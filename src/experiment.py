import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.data import build_dataset, decode_target_values, encode_target_values, inverse_target
from src.models.factory import build_model
from src.postprocess import run_residual_postprocess
from src.trainer import build_loaders, run_inference, train_single_model
from src.utils.metrics import evaluate_metrics


def evaluate_metrics_multistep(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pred_len = y_true.shape[1]
    overall = evaluate_metrics(y_true.reshape(-1), y_pred.reshape(-1))
    per = {}
    for h in range(pred_len):
        per[f"RMSE_h{h + 1}"] = float(np.sqrt(np.mean((y_true[:, h] - y_pred[:, h]) ** 2)))
    return {**overall, **per}


def compute_base_weights(stack_val_x: np.ndarray, y_val: np.ndarray, mode: str, fixed_weights=None) -> np.ndarray:
    n_models = stack_val_x.shape[1]
    if mode == "fixed":
        w = np.asarray(fixed_weights, dtype=np.float32).reshape(-1)
        if w.shape[0] != n_models:
            raise ValueError(f"fixed_weights length {w.shape[0]} != n_models {n_models}")
        w = np.clip(w, 1e-6, None)
        return (w / w.sum()).astype(np.float32)
    if mode == "val_rmse_inverse":
        rmses = []
        for i in range(n_models):
            rmse = float(np.sqrt(np.mean((y_val - stack_val_x[:, i]) ** 2)))
            rmses.append(rmse)
        inv = 1.0 / (np.asarray(rmses, dtype=np.float32) + 1e-8)
        inv = np.clip(inv, 1e-6, None)
        return (inv / inv.sum()).astype(np.float32)
    if mode == "val_linreg":
        coef, *_ = np.linalg.lstsq(stack_val_x, y_val.astype(np.float32), rcond=None)
        w = np.clip(coef.astype(np.float32).reshape(-1), 0.0, None)
        if float(w.sum()) <= 1e-12:
            return (np.ones((n_models,), dtype=np.float32) / n_models).astype(np.float32)
        return (w / w.sum()).astype(np.float32)
    return (np.ones((n_models,), dtype=np.float32) / n_models).astype(np.float32)


def market_phase_mask(series: np.ndarray, win: int = 20, threshold: float = 0.0):
    slope = pd.Series(series).diff(win).fillna(0).values
    trend_mask = slope > threshold
    range_mask = ~trend_mask
    return trend_mask, range_mask


def ensure_prediction_shape(pred: np.ndarray, pred_len: int) -> np.ndarray:
    arr = np.asarray(pred, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim == 2 and arr.shape[1] == 1 and pred_len > 1:
        arr = np.repeat(arr, pred_len, axis=1)
    return arr.astype(np.float32)


def to_absolute_scaled(pred: np.ndarray, last_target: np.ndarray, target_mode: str, pred_len: int) -> np.ndarray:
    arr = ensure_prediction_shape(pred, pred_len)
    return decode_target_values(arr, last_target.astype(np.float32), target_mode).astype(np.float32)


def scaled_prediction_to_inverse(
    pred: np.ndarray,
    last_target: np.ndarray,
    target_mode: str,
    pred_len: int,
    scaler,
    target_idx: int,
    feature_count: int,
) -> np.ndarray:
    abs_scaled = to_absolute_scaled(pred, last_target, target_mode, pred_len)
    return inverse_target(abs_scaled, scaler, target_idx, feature_count)


def train_regime_split_model(
    alias: str,
    base_model_name: str,
    ds: dict,
    cfg: dict,
    logger,
    device: str,
    ckpt_dir: Path,
    fallback_predictions: dict | None = None,
):
    effective_pred_len = ds["y_val"].shape[1] if np.asarray(ds["y_val"]).ndim == 2 else 1
    y_val_shape = ensure_prediction_shape(ds["y_val"], effective_pred_len).shape
    y_test_shape = ensure_prediction_shape(ds["y_test"], effective_pred_len).shape
    val_pred = np.full(y_val_shape, np.nan, dtype=np.float32)
    test_pred = np.full(y_test_shape, np.nan, dtype=np.float32)
    val_filled = np.zeros((y_val_shape[0],), dtype=bool)
    test_filled = np.zeros((y_test_shape[0],), dtype=bool)
    checkpoint_info = {}
    regime_names = ds.get("regime_names") or ["all"]
    regime_values = sorted(set(ds["train_regime"].tolist()) | set(ds["val_regime"].tolist()) | set(ds["test_regime"].tolist()))
    if not regime_values:
        regime_values = [0]

    for regime_value in regime_values:
        regime_name = regime_names[regime_value] if regime_value < len(regime_names) else f"regime_{regime_value}"
        train_mask = ds["train_regime"] == regime_value
        val_mask = ds["val_regime"] == regime_value
        test_mask = ds["test_regime"] == regime_value
        if train_mask.sum() < 32 or val_mask.sum() < 16:
            logger.info(
                "Skip regime model=%s regime=%s due to limited samples train=%d val=%d",
                alias,
                regime_name,
                int(train_mask.sum()),
                int(val_mask.sum()),
            )
            continue

        regime_train_loader, regime_val_loader = build_loaders(
            ds["x_train"][train_mask],
            ds["y_train"][train_mask],
            ds["x_val"][val_mask],
            ds["y_val"][val_mask],
            batch_size=cfg["train"]["batch_size"],
            num_workers=cfg["train"]["num_workers"],
            shuffle_train=bool(cfg["train"].get("shuffle_train", True)),
        )
        regime_model = build_model(
            base_model_name,
            input_size=ds["x_train"].shape[2],
            cfg=cfg,
            dataset_meta={"feature_cols": ds.get("feature_cols", []), "pred_len": effective_pred_len},
        )
        regime_model = train_single_model(
            f"{alias}_{regime_name}",
            regime_model,
            regime_train_loader,
            regime_val_loader,
            cfg,
            logger,
            device,
            wandb_run=None,
        )
        regime_ckpt = ckpt_dir / f"{alias}_{regime_name}_best.pt"
        torch.save({"model_name": base_model_name, "state_dict": regime_model.state_dict(), "config": cfg}, regime_ckpt)
        checkpoint_info[regime_name] = str(regime_ckpt)

        if val_mask.any():
            val_pred[val_mask] = run_inference(regime_model, ds["x_val"][val_mask], device)
            val_filled[val_mask] = True
        if test_mask.any():
            test_pred[test_mask] = run_inference(regime_model, ds["x_test"][test_mask], device)
            test_filled[test_mask] = True

    if fallback_predictions is not None:
        if not val_filled.all():
            val_pred[~val_filled] = ensure_prediction_shape(fallback_predictions["val"], effective_pred_len)[~val_filled]
        if not test_filled.all():
            test_pred[~test_filled] = ensure_prediction_shape(fallback_predictions["test"], effective_pred_len)[~test_filled]

    if np.isnan(val_pred).any() or np.isnan(test_pred).any():
        raise ValueError(f"Regime model {alias} produced incomplete predictions.")

    return val_pred, test_pred, checkpoint_info


def save_visualizations(output_dir: Path, test_dates, y_true, pred_map, analysis_name: str, analysis_pred):
    plt.figure(figsize=(14, 5))
    plt.plot(test_dates, y_true, label="True", color="#FF6347", linewidth=1.8)
    for name, pred in pred_map.items():
        plt.plot(test_dates, pred, label=name, linewidth=1.1)
    plt.plot(test_dates, analysis_pred, label=analysis_name, color="#8A2BE2", linewidth=2.0)
    plt.title("Neural Models Comparison")
    plt.xlabel("Time")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=220)
    plt.close()

    residual = y_true - analysis_pred
    plt.figure(figsize=(10, 4))
    sns.histplot(residual, bins=50, kde=True, color="#FF4500", alpha=0.7)
    plt.title(f"{analysis_name} Residual Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_distribution.png", dpi=220)
    plt.close()

    rolling_rmse = []
    win = min(50, max(10, len(y_true) // 8))
    for i in range(0, len(y_true) - win + 1):
        rolling_rmse.append(np.sqrt(np.mean((y_true[i : i + win] - analysis_pred[i : i + win]) ** 2)))
    plt.figure(figsize=(12, 4))
    plt.plot(rolling_rmse, color="#00BFFF", linewidth=2)
    plt.title(f"Rolling RMSE (window={win})")
    plt.xlabel("Start index")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(output_dir / "rolling_error.png", dpi=220)
    plt.close()


def slice_metrics_by_feature(
    df: pd.DataFrame,
    test_dates: pd.Series,
    y_true_h1: np.ndarray,
    y_pred_h1: np.ndarray,
    feature: str,
    bins,
    labels=None,
):
    if feature not in df.columns:
        return []

    feat = df[feature].reindex(pd.to_datetime(test_dates)).values
    mask = np.isfinite(feat) & np.isfinite(y_true_h1) & np.isfinite(y_pred_h1)
    if not mask.any():
        return []

    feat = feat[mask]
    yt = y_true_h1[mask]
    yp = y_pred_h1[mask]

    if labels is None:
        cats = pd.cut(feat, bins=bins)
    else:
        cats = pd.cut(feat, bins=bins, labels=labels)

    rows = []
    for cat in pd.Series(cats).dropna().unique():
        idx = cats == cat
        if idx.sum() < 50:
            continue
        m = evaluate_metrics(yt[idx], yp[idx])
        rows.append({"feature": feature, "bin": str(cat), "n": int(idx.sum()), **m})
    return rows


def run_experiment(cfg: dict, logger):
    output_dir = Path(cfg["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    logger.info("Loading and preparing dataset...")

    ds = build_dataset(cfg)
    df_full = ds["df"]
    feature_cols = ds.get("feature_cols", [])
    x_train, y_train = ds["x_train"], ds["y_train"]
    x_val, y_val = ds["x_val"], ds["y_val"]
    x_test, y_test = ds["x_test"], ds["y_test"]
    y_val_abs, y_test_abs = ds["y_val_abs"], ds["y_test_abs"]
    scaler, target_idx, feature_count = ds["scaler"], ds["target_idx"], ds["feature_count"]
    test_dates = ds["test_dates"]
    target_mode = ds.get("target_mode", "absolute")
    train_last_target = ds["train_last_target"]
    val_last_target = ds["val_last_target"]
    test_last_target = ds["test_last_target"]
    pred_len = y_train.shape[1] if y_train.ndim == 2 else 1
    logger.info(
        "Dataset ready: feature_count=%d train=%s val=%s test=%s",
        feature_count,
        tuple(x_train.shape),
        tuple(x_val.shape),
        tuple(x_test.shape),
    )
    if feature_cols:
        logger.info("Effective feature columns: %s", feature_cols)
    y_val_inv = inverse_target(y_val_abs, scaler, target_idx, feature_count)
    y_true_inv = inverse_target(y_test_abs, scaler, target_idx, feature_count)

    exp_cfg = cfg.get("experiment", {})
    active_models = [name for name in exp_cfg.get("active_models", ["lstm", "transformer", "tcn"]) if name]
    primary_model = str(exp_cfg.get("primary_model", "lstm"))
    analysis_model_name = primary_model
    if primary_model not in active_models:
        active_models = [primary_model] + [name for name in active_models if name != primary_model]

    def log_test_metrics(model_key: str, pred_scaled: np.ndarray) -> dict:
        arr = to_absolute_scaled(pred_scaled, test_last_target, target_mode, pred_len)
        pred_inv = inverse_target(arr, scaler, target_idx, feature_count)
        m = evaluate_metrics_multistep(y_true_inv, pred_inv)
        logger.info(
            "[TEST] %s => RMSE=%.5f MAE=%.5f MAPE=%.5f R2=%.5f%s",
            model_key,
            m["RMSE"],
            m["MAE"],
            m["MAPE"],
            m["R2"],
            "" if pred_len <= 1 else f" | RMSE_h1={m.get('RMSE_h1', float('nan')):.5f} RMSE_h2={m.get('RMSE_h2', float('nan')):.5f} RMSE_h3={m.get('RMSE_h3', float('nan')):.5f}",
        )
        return m

    train_loader, val_loader = build_loaders(
        x_train,
        y_train,
        x_val,
        y_val,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        shuffle_train=bool(cfg["train"].get("shuffle_train", True)),
    )

    predictions = {}
    checkpoint_map = {}
    logger.info("Training neural sequence models...")
    for name in active_models:
        val_cache = cache_dir / f"{name}_val_pred.npy"
        test_cache = cache_dir / f"{name}_test_pred.npy"
        model_ckpt = ckpt_dir / f"{name}_best.pt"
        if val_cache.exists() and test_cache.exists():
            logger.info("Skip training model=%s (found cached predictions).", name)
            val_pred = np.load(val_cache)
            test_pred = np.load(test_cache)
        else:
            model = build_model(
                name,
                input_size=x_train.shape[2],
                cfg=cfg,
                dataset_meta={"feature_cols": feature_cols, "pred_len": pred_len},
            )
            model = train_single_model(name, model, train_loader, val_loader, cfg, logger, device, wandb_run=None)
            val_pred = run_inference(model, x_val, device)
            test_pred = run_inference(model, x_test, device)
            np.save(val_cache, val_pred)
            np.save(test_cache, test_pred)
            torch.save({"model_name": name, "state_dict": model.state_dict(), "config": cfg}, model_ckpt)
            logger.info("Saved checkpoint: %s", model_ckpt)
        predictions[name] = {"val": val_pred, "test": test_pred}
        checkpoint_map[name] = str(model_ckpt)
        log_test_metrics(name, test_pred)

    regime_cfg = exp_cfg.get("regime_split", {})
    if regime_cfg.get("enabled", False):
        base_model_name = str(regime_cfg.get("base_model", primary_model))
        if base_model_name not in predictions:
            raise ValueError(f"Regime split base_model={base_model_name} must also be listed in experiment.active_models")
        regime_alias = str(regime_cfg.get("alias", f"{base_model_name}_regime"))
        regime_val_cache = cache_dir / f"{regime_alias}_val_pred.npy"
        regime_test_cache = cache_dir / f"{regime_alias}_test_pred.npy"
        if regime_val_cache.exists() and regime_test_cache.exists():
            logger.info("Skip regime training alias=%s (found cached predictions).", regime_alias)
            regime_val_pred = np.load(regime_val_cache)
            regime_test_pred = np.load(regime_test_cache)
            regime_ckpts = {
                regime_name: str(ckpt_dir / f"{regime_alias}_{regime_name}_best.pt")
                for regime_name in (ds.get("regime_names") or [])
            }
        else:
            logger.info("Training regime split model alias=%s base_model=%s", regime_alias, base_model_name)
            regime_val_pred, regime_test_pred, regime_ckpts = train_regime_split_model(
                alias=regime_alias,
                base_model_name=base_model_name,
                ds=ds,
                cfg=cfg,
                logger=logger,
                device=device,
                ckpt_dir=ckpt_dir,
                fallback_predictions=predictions[base_model_name],
            )
            np.save(regime_val_cache, regime_val_pred)
            np.save(regime_test_cache, regime_test_pred)
        predictions[regime_alias] = {"val": regime_val_pred, "test": regime_test_pred}
        checkpoint_map[regime_alias] = regime_ckpts
        if regime_alias not in active_models:
            active_models.append(regime_alias)
        analysis_model_name = regime_alias
        log_test_metrics(regime_alias, regime_test_pred)

    target_series_scaled = x_test[:, :, target_idx]
    persistence_abs_scaled = np.repeat(target_series_scaled[:, -1][:, None], pred_len, axis=1)
    persistence_scaled = encode_target_values(persistence_abs_scaled, test_last_target, target_mode)
    ma_k = min(12, target_series_scaled.shape[1])
    moving_avg_abs_scaled = np.repeat(target_series_scaled[:, -ma_k:].mean(axis=1)[:, None], pred_len, axis=1)
    moving_avg_scaled = encode_target_values(moving_avg_abs_scaled, test_last_target, target_mode)
    log_test_metrics("naive_persistence", persistence_scaled)
    log_test_metrics("naive_moving_avg", moving_avg_scaled)

    fusion_cfg = cfg.get("fusion", {})
    fusion_enabled = bool(fusion_cfg.get("enabled", True)) and len(active_models) > 1
    fusion_val_pred = None
    fusion_pred = None
    fusion_weights_by_h = []
    ablations = {}
    if fusion_enabled:
        logger.info("Building lightweight neural fusion...")
        weighting = fusion_cfg.get("base_weighting", "val_rmse_inverse")
        fixed_weights = fusion_cfg.get("fixed_weights")
        fusion_val_pred = np.zeros((x_val.shape[0], pred_len), dtype=np.float32)
        fusion_pred = np.zeros((x_test.shape[0], pred_len), dtype=np.float32)
        for h in range(pred_len):
            stack_val_x_h = np.vstack([predictions[name]["val"][:, h] for name in active_models]).T
            weights_h = compute_base_weights(stack_val_x_h, y_val[:, h], weighting, fixed_weights=fixed_weights)
            stack_test_x_h = np.vstack([predictions[name]["test"][:, h] for name in active_models]).T
            fusion_val_pred[:, h] = (stack_val_x_h @ weights_h).astype(np.float32)
            fusion_pred[:, h] = (stack_test_x_h @ weights_h).astype(np.float32)
            fusion_weights_by_h.append({"horizon": h + 1, "weights": dict(zip(active_models, np.round(weights_h, 6).tolist()))})
            logger.info("Fusion weights horizon=%d mode=%s weights=%s", h + 1, weighting, fusion_weights_by_h[-1]["weights"])

        if len(active_models) >= 3:
            for omitted in active_models:
                keep = [name for name in active_models if name != omitted]
                ablations[f"fusion_no_{omitted}"] = np.mean(
                    np.stack([predictions[name]["test"] for name in keep], axis=0),
                    axis=0,
                )
        weights_path = output_dir / "fusion_weights.json"
        weights_path.write_text(json.dumps(fusion_weights_by_h, ensure_ascii=False, indent=2), encoding="utf-8")
        checkpoint_map["fusion_main"] = str(weights_path)

    prediction_inv_map = {
        name: {
            "val": scaled_prediction_to_inverse(
                predictions[name]["val"],
                val_last_target,
                target_mode,
                pred_len,
                scaler,
                target_idx,
                feature_count,
            ),
            "test": scaled_prediction_to_inverse(
                predictions[name]["test"],
                test_last_target,
                target_mode,
                pred_len,
                scaler,
                target_idx,
                feature_count,
            ),
        }
        for name in active_models
    }
    if fusion_pred is not None:
        prediction_inv_map["fusion_main"] = {
            "val": scaled_prediction_to_inverse(
                fusion_val_pred,
                val_last_target,
                target_mode,
                pred_len,
                scaler,
                target_idx,
                feature_count,
            ),
            "test": scaled_prediction_to_inverse(
                fusion_pred,
                test_last_target,
                target_mode,
                pred_len,
                scaler,
                target_idx,
                feature_count,
            ),
        }

    postprocess_cfg = cfg.get("postprocess", {}).get("residual_correction", {})
    residual_result = run_residual_postprocess(
        residual_cfg=postprocess_cfg,
        prediction_inv_map=prediction_inv_map,
        x_val=x_val,
        x_test=x_test,
        y_val_inv=y_val_inv,
        y_test_inv=y_true_inv,
        scaler=scaler,
        feature_cols=feature_cols,
        val_regime=ds["val_regime"],
        test_regime=ds["test_regime"],
        regime_names=ds.get("regime_names") or ["all"],
        output_dir=output_dir,
        cache_dir=cache_dir,
        seed=int(cfg["project"].get("seed", 42)),
        logger=logger,
    )
    if residual_result is not None:
        prediction_inv_map[residual_result["alias"]] = {
            "val": residual_result["val_pred_inv"],
            "test": residual_result["test_pred_inv"],
        }
        checkpoint_map[residual_result["alias"]] = residual_result["artifact_path"]
        analysis_model_name = residual_result["alias"]

    metric_rows = []
    pred_for_plot = {}
    all_eval_preds_inv = {
        **{name: prediction_inv_map[name]["test"] for name in active_models},
        "naive_persistence": scaled_prediction_to_inverse(
            persistence_scaled,
            test_last_target,
            target_mode,
            pred_len,
            scaler,
            target_idx,
            feature_count,
        ),
        "naive_moving_avg": scaled_prediction_to_inverse(
            moving_avg_scaled,
            test_last_target,
            target_mode,
            pred_len,
            scaler,
            target_idx,
            feature_count,
        ),
        **({"fusion_main": prediction_inv_map["fusion_main"]["test"]} if fusion_pred is not None else {}),
        **({residual_result["alias"]: residual_result["test_pred_inv"]} if residual_result is not None else {}),
    }
    for name, pred in ablations.items():
        all_eval_preds_inv[name] = scaled_prediction_to_inverse(
            pred,
            test_last_target,
            target_mode,
            pred_len,
            scaler,
            target_idx,
            feature_count,
        )

    for name, pred_inv in all_eval_preds_inv.items():
        m = evaluate_metrics_multistep(y_true_inv, pred_inv)
        metric_rows.append({"model": name, **m})
        if name in active_models:
            pred_for_plot[name] = pred_inv[:, 0]
        logger.info(
            "%s metrics(flat) => RMSE=%.5f MAE=%.5f MAPE=%.5f R2=%.5f",
            name,
            m["RMSE"],
            m["MAE"],
            m["MAPE"],
            m["R2"],
        )

    if residual_result is not None:
        analysis_name = residual_result["alias"]
    else:
        analysis_name = "fusion_main" if fusion_pred is not None else analysis_model_name
    analysis_inv = all_eval_preds_inv[analysis_name]
    save_visualizations(output_dir, test_dates, y_true_inv[:, 0], pred_for_plot, analysis_name, analysis_inv[:, 0])

    trend_mask, range_mask = market_phase_mask(y_true_inv[:, 0])
    phase_metrics = {
        "trend": evaluate_metrics(y_true_inv[trend_mask, 0], analysis_inv[trend_mask, 0]) if trend_mask.any() else {},
        "range": evaluate_metrics(y_true_inv[range_mask, 0], analysis_inv[range_mask, 0]) if range_mask.any() else {},
    }
    logger.info("Phase metrics trend=%s range=%s", phase_metrics["trend"], phase_metrics["range"])

    slice_rows = []
    for feature, bins, labels in [
        ("temporal_hour", [0, 6, 12, 18, 24], ["night(0-6)", "morning(6-12)", "afternoon(12-18)", "evening(18-24)"]),
        ("SolarFlux", None, None),
    ]:
        if feature == "SolarFlux" and feature in df_full.columns:
            sf = df_full["SolarFlux"].reindex(pd.to_datetime(test_dates)).values
            sf = sf[np.isfinite(sf)]
            if len(sf) > 100:
                qs = np.quantile(sf, [0.0, 0.33, 0.66, 1.0]).tolist()
                qs = sorted(list(dict.fromkeys([float(x) for x in qs])))
                if len(qs) >= 3:
                    slice_rows += slice_metrics_by_feature(df_full, test_dates, y_true_inv[:, 0], analysis_inv[:, 0], feature, qs)
        elif feature in df_full.columns:
            slice_rows += slice_metrics_by_feature(df_full, test_dates, y_true_inv[:, 0], analysis_inv[:, 0], feature, bins, labels)

    if slice_rows:
        slice_df = pd.DataFrame(slice_rows).sort_values(["feature", "RMSE"])
        slice_df.to_csv(output_dir / "slice_metrics.csv", index=False, encoding="utf-8-sig")
        logger.info("Saved slice metrics to slice_metrics.csv (%d rows)", len(slice_df))

    metrics_df = pd.DataFrame(metric_rows).sort_values("RMSE")
    metrics_df.to_csv(output_dir / "metrics_all_models.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "phase_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(phase_metrics, f, ensure_ascii=False, indent=2)

    pred_df = pd.DataFrame({"Date": test_dates, "y_true_h1": y_true_inv[:, 0], f"{analysis_name}_pred_h1": analysis_inv[:, 0]})
    if pred_len > 1:
        for h in range(1, pred_len):
            pred_df[f"y_true_h{h + 1}"] = y_true_inv[:, h]
            pred_df[f"{analysis_name}_pred_h{h + 1}"] = analysis_inv[:, h]
    for n, p in pred_for_plot.items():
        pred_df[f"{n}_pred"] = p
    pred_df.to_csv(output_dir / "predictions_test.csv", index=False, encoding="utf-8-sig")

    best_row = metrics_df.iloc[0].to_dict()
    best_model_name = str(best_row["model"])
    best_summary = {
        "primary_model": primary_model,
        "analysis_model": analysis_model_name,
        "best_model": best_model_name,
        "best_metrics": best_row,
        "checkpoint": checkpoint_map.get(best_model_name),
        "all_checkpoints": checkpoint_map,
        "active_models": active_models,
        "fusion_enabled": fusion_enabled,
    }
    with (output_dir / "best_model_summary.json").open("w", encoding="utf-8") as f:
        json.dump(best_summary, f, ensure_ascii=False, indent=2)
    logger.info("Saved best model summary: %s", output_dir / "best_model_summary.json")

    logger.info("Saved all outputs to: %s", output_dir.resolve())
