import argparse
from copy import deepcopy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (
    add_time_features,
    aggregate_target,
    apply_long_gap_mask,
    build_effective_feature_frame,
    build_regime_labels,
    build_segment_ids,
    compute_missing_run_lengths,
    create_sequences_with_context,
    encode_target_values,
    inverse_target,
    read_source_dataframe,
    set_seed,
)
from src.experiment import (
    compute_base_weights,
    evaluate_metrics_multistep,
    save_visualizations,
    scaled_prediction_to_inverse,
    train_regime_split_model,
)
from src.models.factory import build_model
from src.trainer import build_loaders, run_inference, train_single_model
from src.utils.config import load_config
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Strict cross-site transfer: source-site train, target-site test.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cross_site_transfer_to_dataWfon0U.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.epochs",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def add_daylight_proxy(df: pd.DataFrame, proxy_cfg: dict | None) -> pd.DataFrame:
    proxy_cfg = proxy_cfg or {}
    if not proxy_cfg.get("enabled", False):
        return df

    overwrite_existing = bool(proxy_cfg.get("overwrite_existing", True))
    if (not overwrite_existing) and ("is_daylight" in df.columns):
        return df

    start_hour = float(proxy_cfg.get("start_hour", 6.0))
    end_hour = float(proxy_cfg.get("end_hour", 18.0))
    hour = df.index.hour + df.index.minute / 60.0 + df.index.second / 3600.0

    if start_hour <= end_hour:
        mask = (hour >= start_hour) & (hour < end_hour)
    else:
        mask = (hour >= start_hour) | (hour < end_hour)

    out = df.copy()
    out["is_daylight"] = mask.astype(np.float32)
    return out


def prepare_site_dataframe(
    file_path: Path,
    datetime_col: str,
    time_col: str | None,
    feature_cols: list[str],
    target_col: str,
    resample_rule: str | None,
    preprocess_cfg: dict,
    status_col: str | None = None,
    status_keep_values=None,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    df = read_source_dataframe(file_path)

    if time_col and time_col in df.columns:
        datetime_str = df[datetime_col].astype(str).str[:10] + " " + df[time_col].astype(str)
        try:
            dt = pd.to_datetime(datetime_str, format="mixed", errors="coerce")
        except Exception:
            dt = pd.to_datetime(datetime_str, errors="coerce")
    else:
        dt = pd.to_datetime(df[datetime_col], errors="coerce")

    df["datetime"] = dt
    df = df.dropna(subset=["datetime"]).copy()

    if status_col and status_col in df.columns and status_keep_values is not None:
        keep_values = set(status_keep_values)
        df = df[df[status_col].isin(keep_values)].copy()

    df = df.sort_values("datetime").set_index("datetime")
    df = df.drop(columns=[c for c in [datetime_col, time_col] if c and c in df.columns], errors="ignore")
    if preprocess_cfg.get("drop_duplicate_timestamps", True):
        df = df.groupby(level=0).mean(numeric_only=True)

    if resample_rule:
        df = df.resample(resample_rule).mean(numeric_only=True)

    missing_reference = df.isna().copy()
    missing_run_lengths = compute_missing_run_lengths(missing_reference)

    interpolate_cfg = preprocess_cfg.get("interpolate", {})
    if interpolate_cfg.get("enabled", True):
        df = df.interpolate(
            method=interpolate_cfg.get("method", "time"),
            limit=interpolate_cfg.get("limit"),
            limit_direction="both",
        )

    df, long_gap_mask = apply_long_gap_mask(
        df,
        missing_ref=missing_reference,
        missing_run_lengths=missing_run_lengths,
        gap_cfg=preprocess_cfg.get("long_gap_mask"),
    )

    df = add_daylight_proxy(df, preprocess_cfg.get("daylight_proxy"))
    df = add_time_features(df, preprocess_cfg.get("time_features"))

    preprocess_tmp = {
        **preprocess_cfg,
        "_missing_reference": missing_reference,
        "_missing_run_lengths": missing_run_lengths,
    }
    dcfg = {"feature_cols": feature_cols, "target_col": target_col}
    df, effective_feature_cols = build_effective_feature_frame(df, dcfg, preprocess_tmp)

    missing_feature_cols = [col for col in effective_feature_cols if col not in df.columns]
    if missing_feature_cols:
        raise ValueError(f"Missing engineered feature columns: {missing_feature_cols}")

    df = df.dropna(subset=effective_feature_cols).copy()
    if long_gap_mask.any():
        long_gap_mask = long_gap_mask.reindex(df.index, fill_value=False)
    segment_ids = build_segment_ids(df.index, resample_rule)
    return df, effective_feature_cols, segment_ids


def fit_source_clip_bounds(
    feat_df: pd.DataFrame,
    n_train: int,
    clip_cfg: dict | None,
    feature_cols: list[str],
):
    clip_cfg = clip_cfg or {}
    if not clip_cfg.get("enabled", False):
        return feat_df.copy(), {}

    lower_q = float(clip_cfg.get("lower", 0.005))
    upper_q = float(clip_cfg.get("upper", 0.995))
    exclude_cols = set(clip_cfg.get("exclude_cols", []))

    clipped = feat_df.copy()
    train_part = clipped.iloc[:n_train]
    bounds = {}
    for col in feature_cols:
        if col in exclude_cols:
            continue
        lower = float(train_part[col].quantile(lower_q))
        upper = float(train_part[col].quantile(upper_q))
        clipped[col] = clipped[col].clip(lower=lower, upper=upper)
        bounds[col] = {"lower": lower, "upper": upper}
    return clipped, bounds


def apply_clip_bounds(feat_df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    clipped = feat_df.copy()
    for col, value in bounds.items():
        if col not in clipped.columns:
            continue
        clipped[col] = clipped[col].clip(lower=value["lower"], upper=value["upper"])
    return clipped


def summarize_feature_shift(
    source_train_df: pd.DataFrame,
    source_val_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_z_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        source_train = source_train_df[col].dropna()
        source_val = source_val_df[col].dropna()
        target = target_df[col].dropna()
        target_z = target_z_df[col].dropna()
        source_std = float(source_train.std(ddof=1)) if len(source_train) > 1 else float("nan")
        target_mean = float(target.mean()) if len(target) else float("nan")
        source_mean = float(source_train.mean()) if len(source_train) else float("nan")
        if np.isfinite(source_std) and source_std > 0:
            mean_shift_sigma = (target_mean - source_mean) / source_std
        else:
            mean_shift_sigma = float("nan")
        rows.append(
            {
                "feature": col,
                "source_train_mean": source_mean,
                "source_train_std": source_std,
                "source_train_min": float(source_train.min()) if len(source_train) else float("nan"),
                "source_train_max": float(source_train.max()) if len(source_train) else float("nan"),
                "source_val_mean": float(source_val.mean()) if len(source_val) else float("nan"),
                "target_mean": target_mean,
                "target_std": float(target.std(ddof=1)) if len(target) > 1 else float("nan"),
                "target_min": float(target.min()) if len(target) else float("nan"),
                "target_max": float(target.max()) if len(target) else float("nan"),
                "target_z_mean": float(target_z.mean()) if len(target_z) else float("nan"),
                "target_z_min": float(target_z.min()) if len(target_z) else float("nan"),
                "target_z_max": float(target_z.max()) if len(target_z) else float("nan"),
                "target_abs_z_gt3_pct": float((target_z.abs() > 3.0).mean() * 100.0) if len(target_z) else float("nan"),
                "target_abs_z_gt10_pct": float((target_z.abs() > 10.0).mean() * 100.0) if len(target_z) else float("nan"),
                "mean_shift_sigma": float(mean_shift_sigma),
            }
        )
    return pd.DataFrame(rows)


def finalize_sequence_targets(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    target_idx: int,
    target_mode: str,
    agg_mode: str,
):
    y_abs = aggregate_target(y_arr, agg_mode).astype(np.float32)
    last_target = x_arr[:, -1, target_idx : target_idx + 1].astype(np.float32)
    y_model = encode_target_values(y_abs, last_target, target_mode).astype(np.float32)
    return y_abs, y_model, last_target


def log_metrics(logger, stage: str, model_name: str, metrics: dict):
    logger.info(
        "[%s] %s => RMSE=%.5f MAE=%.5f MAPE=%.5f R2=%.5f%s",
        stage,
        model_name,
        metrics["RMSE"],
        metrics["MAE"],
        metrics["MAPE"],
        metrics["R2"],
        ""
        if "RMSE_h1" not in metrics
        else f" | RMSE_h1={metrics.get('RMSE_h1', float('nan')):.5f}"
        + (f" RMSE_h2={metrics.get('RMSE_h2', float('nan')):.5f}" if "RMSE_h2" in metrics else "")
        + (f" RMSE_h3={metrics.get('RMSE_h3', float('nan')):.5f}" if "RMSE_h3" in metrics else ""),
    )


def main():
    args = parse_args()
    cfg_path = resolve_project_path(args.config)
    cfg = load_config(str(cfg_path))
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)

    if args.output_dir:
        output_dir = resolve_project_path(args.output_dir)
    else:
        output_dir = resolve_project_path(cfg["project"]["output_dir"])
    cfg["project"]["output_dir"] = str(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    set_seed(int(cfg["project"]["seed"]))
    logger = setup_logger(str(output_dir), cfg["project"].get("log_level", "INFO"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    logger.info("Running strict cross-site transfer: source train/val -> target test")

    data_cfg = cfg["data"]
    preprocess_cfg = cfg.get("preprocess", {})
    target_col = data_cfg["target_col"]
    feature_cols = list(data_cfg["feature_cols"])
    resample_rule = data_cfg.get("resample_rule")

    source_path = resolve_project_path(data_cfg["source_file_path"])
    target_path = resolve_project_path(data_cfg["target_file_path"])
    logger.info("Source file: %s", source_path)
    logger.info("Target file: %s", target_path)

    source_df, source_feature_cols, source_segment_ids = prepare_site_dataframe(
        file_path=source_path,
        datetime_col=data_cfg["source_datetime_col"],
        time_col=data_cfg.get("source_time_col"),
        feature_cols=feature_cols,
        target_col=target_col,
        resample_rule=resample_rule,
        preprocess_cfg=preprocess_cfg,
        status_col=preprocess_cfg.get("source_status_col"),
        status_keep_values=preprocess_cfg.get("source_status_keep_values"),
    )
    target_df, target_feature_cols, target_segment_ids = prepare_site_dataframe(
        file_path=target_path,
        datetime_col=data_cfg["target_datetime_col"],
        time_col=data_cfg.get("target_time_col"),
        feature_cols=feature_cols,
        target_col=target_col,
        resample_rule=resample_rule,
        preprocess_cfg=preprocess_cfg,
        status_col=preprocess_cfg.get("target_status_col"),
        status_keep_values=preprocess_cfg.get("target_status_keep_values"),
    )

    if source_feature_cols != target_feature_cols:
        raise ValueError(
            "Source and target effective feature schemas do not match.\n"
            f"source={source_feature_cols}\n"
            f"target={target_feature_cols}"
        )
    feature_cols = source_feature_cols
    target_idx = feature_cols.index(target_col)
    feature_count = len(feature_cols)

    n_source_total = len(source_df)
    n_source_train = int(n_source_total * float(data_cfg["source_train_ratio"]))
    n_source_val = int(n_source_total * float(data_cfg["source_val_ratio"]))
    if n_source_train <= 0 or n_source_val <= 0:
        raise ValueError("Invalid source split sizes. Check source_train_ratio and source_val_ratio.")

    source_feat_df, clip_bounds = fit_source_clip_bounds(
        source_df[feature_cols].copy(),
        n_train=n_source_train,
        clip_cfg=preprocess_cfg.get("clip_quantiles"),
        feature_cols=feature_cols,
    )
    target_feat_df = apply_clip_bounds(target_df[feature_cols].copy(), clip_bounds)

    scaler_name = str(preprocess_cfg.get("scaler", "standard")).lower()
    scaler = RobustScaler() if scaler_name == "robust" else StandardScaler()
    scaler.fit(source_feat_df.iloc[:n_source_train].values)

    source_all_scaled = scaler.transform(source_feat_df.values)
    target_all_scaled = scaler.transform(target_feat_df.values)
    target_z_df = pd.DataFrame(target_all_scaled, index=target_feat_df.index, columns=feature_cols)

    feature_shift_df = summarize_feature_shift(
        source_train_df=source_feat_df.iloc[:n_source_train],
        source_val_df=source_feat_df.iloc[n_source_train : n_source_train + n_source_val],
        target_df=target_feat_df,
        target_z_df=target_z_df,
        feature_cols=feature_cols,
    )
    feature_shift_df.to_csv(output_dir / "feature_shift_diagnostics.csv", index=False, encoding="utf-8-sig")
    extreme_features = (
        feature_shift_df.sort_values("target_abs_z_gt3_pct", ascending=False)
        .head(5)
        .to_dict(orient="records")
    )
    (output_dir / "feature_shift_summary.json").write_text(
        json.dumps(
            {
                "scaler": scaler_name,
                "feature_count": len(feature_cols),
                "extreme_features_by_abs_z_gt3_pct": extreme_features,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    seq_len = int(data_cfg["seq_len"])
    raw_pred_len = int(data_cfg["pred_len"])
    target_mode = str(data_cfg.get("target_mode", "absolute")).lower()
    agg_mode = str(data_cfg.get("target_aggregation", "multistep"))

    source_split = create_sequences_with_context(
        source_all_scaled,
        target_idx=target_idx,
        seq_len=seq_len,
        pred_len=raw_pred_len,
        train_end=n_source_train,
        val_end=n_source_train + n_source_val,
        segment_ids=source_segment_ids,
    )
    target_split = create_sequences_with_context(
        target_all_scaled,
        target_idx=target_idx,
        seq_len=seq_len,
        pred_len=raw_pred_len,
        train_end=0,
        val_end=0,
        segment_ids=target_segment_ids,
    )

    x_train, y_train_raw, train_forecast_start = source_split["train"]
    x_val, y_val_raw, val_forecast_start = source_split["val"]
    x_test, y_test_raw, test_forecast_start = target_split["test"]
    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise ValueError("Cross-site transfer produced empty train/val/test sequences.")

    y_train_abs, y_train, train_last_target = finalize_sequence_targets(
        x_arr=x_train,
        y_arr=y_train_raw,
        target_idx=target_idx,
        target_mode=target_mode,
        agg_mode=agg_mode,
    )
    y_val_abs, y_val, val_last_target = finalize_sequence_targets(
        x_arr=x_val,
        y_arr=y_val_raw,
        target_idx=target_idx,
        target_mode=target_mode,
        agg_mode=agg_mode,
    )
    y_test_abs, y_test, test_last_target = finalize_sequence_targets(
        x_arr=x_test,
        y_arr=y_test_raw,
        target_idx=target_idx,
        target_mode=target_mode,
        agg_mode=agg_mode,
    )

    pred_len = int(y_train.shape[1])
    model_cfg = deepcopy(cfg)
    model_cfg["data"]["pred_len"] = pred_len

    y_val_inv = inverse_target(y_val_abs, scaler, target_idx, feature_count)
    y_test_inv = inverse_target(y_test_abs, scaler, target_idx, feature_count)
    test_dates = pd.Series(pd.Index(target_df.index)[test_forecast_start]).reset_index(drop=True)

    regime_cfg = cfg.get("experiment", {}).get("regime_split", {}) or {}
    train_regime, regime_names = build_regime_labels(source_df, train_forecast_start, regime_cfg)
    val_regime, _ = build_regime_labels(source_df, val_forecast_start, regime_cfg)
    test_regime, _ = build_regime_labels(target_df, test_forecast_start, regime_cfg)

    ds = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "pred_len": pred_len,
        "train_regime": train_regime,
        "val_regime": val_regime,
        "test_regime": test_regime,
        "regime_names": regime_names,
    }

    train_loader, val_loader = build_loaders(
        x_train,
        y_train,
        x_val,
        y_val,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        shuffle_train=bool(cfg["train"].get("shuffle_train", True)),
    )

    cache_dir = output_dir / "cache"
    ckpt_dir = output_dir / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = cfg.get("experiment", {})
    active_models = [name for name in exp_cfg.get("active_models", ["gru", "lstm", "cnn_lstm"]) if name]
    primary_model = str(exp_cfg.get("primary_model", "gru"))

    predictions = {}
    checkpoint_map = {}
    source_val_metrics = {}
    target_test_metrics = {}
    prediction_inv_map = {}

    def evaluate_scaled_predictions(pred_scaled: np.ndarray, last_target: np.ndarray, y_true_inv: np.ndarray):
        pred_inv = scaled_prediction_to_inverse(
            pred_scaled,
            last_target=last_target,
            target_mode=target_mode,
            pred_len=pred_len,
            scaler=scaler,
            target_idx=target_idx,
            feature_count=feature_count,
        )
        return pred_inv, evaluate_metrics_multistep(y_true_inv, pred_inv)

    logger.info(
        "Prepared source/target data: source_train=%s source_val=%s target_test=%s feature_count=%d",
        tuple(x_train.shape),
        tuple(x_val.shape),
        tuple(x_test.shape),
        feature_count,
    )
    logger.info("Shared effective feature columns: %s", feature_cols)

    for name in active_models:
        val_cache = cache_dir / f"{name}_source_val_pred.npy"
        test_cache = cache_dir / f"{name}_target_test_pred.npy"
        model_ckpt = ckpt_dir / f"{name}_best.pt"
        if val_cache.exists() and test_cache.exists():
            logger.info("Skip training model=%s (cached transfer predictions found).", name)
            val_pred = np.load(val_cache)
            test_pred = np.load(test_cache)
        else:
            model = build_model(
                name,
                input_size=x_train.shape[2],
                cfg=model_cfg,
                dataset_meta={"pred_len": pred_len, "feature_cols": feature_cols},
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

        val_inv, val_metrics = evaluate_scaled_predictions(val_pred, val_last_target, y_val_inv)
        test_inv, test_metrics = evaluate_scaled_predictions(test_pred, test_last_target, y_test_inv)
        source_val_metrics[name] = val_metrics
        target_test_metrics[name] = test_metrics
        log_metrics(logger, "SOURCE_VAL", name, val_metrics)
        log_metrics(logger, "TARGET_TEST", name, test_metrics)

    if regime_cfg.get("enabled", False):
        base_model_name = str(regime_cfg.get("base_model", primary_model))
        regime_alias = str(regime_cfg.get("alias", f"{base_model_name}_regime"))
        regime_val_cache = cache_dir / f"{regime_alias}_source_val_pred.npy"
        regime_test_cache = cache_dir / f"{regime_alias}_target_test_pred.npy"
        if regime_val_cache.exists() and regime_test_cache.exists():
            logger.info("Skip regime transfer alias=%s (cached predictions found).", regime_alias)
            regime_val_pred = np.load(regime_val_cache)
            regime_test_pred = np.load(regime_test_cache)
            regime_ckpts = {
                regime_name: str(ckpt_dir / f"{regime_alias}_{regime_name}_best.pt")
                for regime_name in regime_names
            }
        else:
            regime_val_pred, regime_test_pred, regime_ckpts = train_regime_split_model(
                alias=regime_alias,
                base_model_name=base_model_name,
                ds=ds,
                cfg=model_cfg,
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

        val_inv, val_metrics = evaluate_scaled_predictions(regime_val_pred, val_last_target, y_val_inv)
        test_inv, test_metrics = evaluate_scaled_predictions(regime_test_pred, test_last_target, y_test_inv)
        source_val_metrics[regime_alias] = val_metrics
        target_test_metrics[regime_alias] = test_metrics
        log_metrics(logger, "SOURCE_VAL", regime_alias, val_metrics)
        log_metrics(logger, "TARGET_TEST", regime_alias, test_metrics)

    target_series_scaled = x_test[:, :, target_idx]
    persistence_abs_scaled = np.repeat(target_series_scaled[:, -1][:, None], pred_len, axis=1)
    persistence_scaled = encode_target_values(persistence_abs_scaled, test_last_target, target_mode)
    moving_avg_k = min(12, target_series_scaled.shape[1])
    moving_avg_abs_scaled = np.repeat(target_series_scaled[:, -moving_avg_k:].mean(axis=1)[:, None], pred_len, axis=1)
    moving_avg_scaled = encode_target_values(moving_avg_abs_scaled, test_last_target, target_mode)

    persistence_inv, persistence_metrics = evaluate_scaled_predictions(persistence_scaled, test_last_target, y_test_inv)
    moving_avg_inv, moving_avg_metrics = evaluate_scaled_predictions(moving_avg_scaled, test_last_target, y_test_inv)
    prediction_inv_map["naive_persistence"] = {"test": persistence_inv}
    prediction_inv_map["naive_moving_avg"] = {"test": moving_avg_inv}
    target_test_metrics["naive_persistence"] = persistence_metrics
    target_test_metrics["naive_moving_avg"] = moving_avg_metrics
    log_metrics(logger, "TARGET_TEST", "naive_persistence", persistence_metrics)
    log_metrics(logger, "TARGET_TEST", "naive_moving_avg", moving_avg_metrics)

    fusion_cfg = cfg.get("fusion", {})
    fusion_enabled = bool(fusion_cfg.get("enabled", True)) and len(active_models) > 1
    for name in active_models:
        prediction_inv_map[name] = {
            "val": scaled_prediction_to_inverse(
                predictions[name]["val"],
                last_target=val_last_target,
                target_mode=target_mode,
                pred_len=pred_len,
                scaler=scaler,
                target_idx=target_idx,
                feature_count=feature_count,
            ),
            "test": scaled_prediction_to_inverse(
                predictions[name]["test"],
                last_target=test_last_target,
                target_mode=target_mode,
                pred_len=pred_len,
                scaler=scaler,
                target_idx=target_idx,
                feature_count=feature_count,
            ),
        }

    fusion_weights_by_h = []
    if fusion_enabled:
        weighting = fusion_cfg.get("base_weighting", "val_linreg")
        fixed_weights = fusion_cfg.get("fixed_weights")
        fusion_val_pred = np.zeros((x_val.shape[0], pred_len), dtype=np.float32)
        fusion_test_pred = np.zeros((x_test.shape[0], pred_len), dtype=np.float32)
        for h in range(pred_len):
            stack_val_x_h = np.vstack([predictions[name]["val"][:, h] for name in active_models]).T
            weights_h = compute_base_weights(stack_val_x_h, y_val[:, h], weighting, fixed_weights=fixed_weights)
            stack_test_x_h = np.vstack([predictions[name]["test"][:, h] for name in active_models]).T
            fusion_val_pred[:, h] = (stack_val_x_h @ weights_h).astype(np.float32)
            fusion_test_pred[:, h] = (stack_test_x_h @ weights_h).astype(np.float32)
            fusion_weights_by_h.append({"horizon": h + 1, "weights": dict(zip(active_models, np.round(weights_h, 6).tolist()))})
            logger.info("Transfer fusion weights horizon=%d mode=%s weights=%s", h + 1, weighting, fusion_weights_by_h[-1]["weights"])

        prediction_inv_map["fusion_main"] = {
            "val": scaled_prediction_to_inverse(
                fusion_val_pred,
                last_target=val_last_target,
                target_mode=target_mode,
                pred_len=pred_len,
                scaler=scaler,
                target_idx=target_idx,
                feature_count=feature_count,
            ),
            "test": scaled_prediction_to_inverse(
                fusion_test_pred,
                last_target=test_last_target,
                target_mode=target_mode,
                pred_len=pred_len,
                scaler=scaler,
                target_idx=target_idx,
                feature_count=feature_count,
            ),
        }
        source_val_metrics["fusion_main"] = evaluate_metrics_multistep(y_val_inv, prediction_inv_map["fusion_main"]["val"])
        target_test_metrics["fusion_main"] = evaluate_metrics_multistep(y_test_inv, prediction_inv_map["fusion_main"]["test"])
        log_metrics(logger, "SOURCE_VAL", "fusion_main", source_val_metrics["fusion_main"])
        log_metrics(logger, "TARGET_TEST", "fusion_main", target_test_metrics["fusion_main"])
        (output_dir / "fusion_weights.json").write_text(json.dumps(fusion_weights_by_h, ensure_ascii=False, indent=2), encoding="utf-8")

        if len(active_models) >= 3:
            for omitted in active_models:
                keep = [name for name in active_models if name != omitted]
                ablation_pred_scaled = np.mean(np.stack([predictions[name]["test"] for name in keep], axis=0), axis=0)
                ablation_name = f"fusion_no_{omitted}"
                prediction_inv_map[ablation_name] = {
                    "test": scaled_prediction_to_inverse(
                        ablation_pred_scaled,
                        last_target=test_last_target,
                        target_mode=target_mode,
                        pred_len=pred_len,
                        scaler=scaler,
                        target_idx=target_idx,
                        feature_count=feature_count,
                    )
                }
                target_test_metrics[ablation_name] = evaluate_metrics_multistep(
                    y_test_inv, prediction_inv_map[ablation_name]["test"]
                )
                log_metrics(logger, "TARGET_TEST", ablation_name, target_test_metrics[ablation_name])

    metrics_df = (
        pd.DataFrame.from_dict(target_test_metrics, orient="index")
        .reset_index()
        .rename(columns={"index": "model"})
        .sort_values("RMSE", ascending=True)
        .reset_index(drop=True)
    )
    metrics_df.to_csv(output_dir / "metrics_all_models.csv", index=False, encoding="utf-8-sig")

    source_val_df = (
        pd.DataFrame.from_dict(source_val_metrics, orient="index")
        .reset_index()
        .rename(columns={"index": "model"})
        .sort_values("RMSE", ascending=True)
        .reset_index(drop=True)
    )
    source_val_df.to_csv(output_dir / "metrics_source_val.csv", index=False, encoding="utf-8-sig")

    best_row = metrics_df.iloc[0].to_dict()
    best_model_name = str(best_row["model"])
    best_summary = {
        "mode": "strict_cross_site_transfer",
        "source_file": str(source_path),
        "target_file": str(target_path),
        "best_model": best_model_name,
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in best_row.items() if k != "model"},
        "source_time_begin": str(source_df.index.min()),
        "source_time_end": str(source_df.index.max()),
        "target_time_begin": str(target_df.index.min()),
        "target_time_end": str(target_df.index.max()),
        "source_windows": {
            "train": int(len(x_train)),
            "val": int(len(x_val)),
        },
        "target_windows": {
            "test": int(len(x_test)),
        },
        "feature_count": int(feature_count),
        "feature_cols": feature_cols,
        "raw_pred_len": raw_pred_len,
        "effective_pred_len": pred_len,
        "target_aggregation": agg_mode,
        "clip_bounds_source_train_only": bool(preprocess_cfg.get("clip_quantiles", {}).get("enabled", False)),
        "daylight_proxy": preprocess_cfg.get("daylight_proxy"),
        "fusion_weights_file": str(output_dir / "fusion_weights.json") if fusion_enabled else None,
    }
    (output_dir / "best_model_summary.json").write_text(json.dumps(best_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    prediction_table = pd.DataFrame({"datetime": test_dates})
    for h in range(pred_len):
        prediction_table[f"y_true_h{h + 1}"] = y_test_inv[:, h]
    for name, pred_map in prediction_inv_map.items():
        if "test" not in pred_map:
            continue
        pred_arr = pred_map["test"]
        for h in range(pred_len):
            prediction_table[f"{name}_h{h + 1}"] = pred_arr[:, h]
    prediction_table.to_csv(output_dir / "predictions_target_test.csv", index=False, encoding="utf-8-sig")

    regime_slice_rows = []
    best_pred_h1 = prediction_inv_map[best_model_name]["test"][:, 0]
    for regime_value, regime_name in enumerate(regime_names):
        mask = test_regime == regime_value
        if int(mask.sum()) < 20:
            continue
        metrics = evaluate_metrics_multistep(y_test_inv[mask], prediction_inv_map[best_model_name]["test"][mask])
        regime_slice_rows.append({"regime": regime_name, "n": int(mask.sum()), **metrics})
    if regime_slice_rows:
        pd.DataFrame(regime_slice_rows).to_csv(output_dir / "regime_slice_metrics.csv", index=False, encoding="utf-8-sig")

    plot_models = {}
    for name in active_models:
        plot_models[name] = prediction_inv_map[name]["test"][:, 0]
    if fusion_enabled and "fusion_main" in prediction_inv_map:
        plot_models["fusion_main"] = prediction_inv_map["fusion_main"]["test"][:, 0]
    save_visualizations(
        output_dir=output_dir,
        test_dates=test_dates,
        y_true=y_test_inv[:, 0],
        pred_map=plot_models,
        analysis_name=best_model_name,
        analysis_pred=best_pred_h1,
    )

    transfer_summary = {
        "mode": "source_train_target_test",
        "source_description": "2021-04-01 to 2021-06-30 source-site records",
        "target_description": "2006-06-09 to 2006-08-08 external site records",
        "shared_feature_cols": feature_cols,
        "source_rows_after_preprocess": int(len(source_df)),
        "target_rows_after_preprocess": int(len(target_df)),
        "source_window_counts": {"train": int(len(x_train)), "val": int(len(x_val))},
        "target_window_counts": {"test": int(len(x_test))},
        "raw_pred_len": raw_pred_len,
        "effective_pred_len": pred_len,
        "target_aggregation": agg_mode,
        "best_model": best_model_name,
        "best_rmse": float(best_row["RMSE"]),
    }
    (output_dir / "transfer_summary.json").write_text(json.dumps(transfer_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Strict cross-site transfer finished. Best model=%s RMSE=%.5f", best_model_name, float(best_row["RMSE"]))


if __name__ == "__main__":
    main()
