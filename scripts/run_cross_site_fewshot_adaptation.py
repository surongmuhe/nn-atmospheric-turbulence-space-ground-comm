import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_current_route_cross_site import make_config
from scripts.run_cross_site_transfer import (
    apply_clip_bounds,
    finalize_sequence_targets,
    fit_source_clip_bounds,
    prepare_site_dataframe,
)
from src.data import (
    create_sequences_with_context,
    encode_target_values,
    inverse_target,
    set_seed,
)
from src.experiment import evaluate_metrics_multistep, scaled_prediction_to_inverse
from src.models.factory import build_model
from src.trainer import build_loaders, run_inference, train_single_model
from src.utils.logger import setup_logger


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def prepare_common_data(cfg: dict):
    data_cfg = cfg["data"]
    preprocess_cfg = cfg.get("preprocess", {})
    target_col = data_cfg["target_col"]
    feature_cols = list(data_cfg["feature_cols"])
    resample_rule = data_cfg.get("resample_rule")

    source_df, source_feature_cols, source_segment_ids = prepare_site_dataframe(
        file_path=resolve_project_path(data_cfg["source_file_path"]),
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
        file_path=resolve_project_path(data_cfg["target_file_path"]),
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
        raise ValueError(f"Source/target feature schemas differ: {source_feature_cols} vs {target_feature_cols}")

    feature_cols = source_feature_cols
    target_idx = feature_cols.index(target_col)
    feature_count = len(feature_cols)

    n_source_total = len(source_df)
    n_source_train = int(n_source_total * float(data_cfg["source_train_ratio"]))
    n_source_val = int(n_source_total * float(data_cfg["source_val_ratio"]))

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

    x_train, y_train_raw, _ = source_split["train"]
    x_val, y_val_raw, _ = source_split["val"]
    _, y_train, _ = finalize_sequence_targets(x_train, y_train_raw, target_idx, target_mode, agg_mode)
    _, y_val, _ = finalize_sequence_targets(x_val, y_val_raw, target_idx, target_mode, agg_mode)

    pred_len = int(y_train.shape[1])
    model_cfg = copy.deepcopy(cfg)
    model_cfg["data"]["pred_len"] = pred_len

    return {
        "cfg": cfg,
        "model_cfg": model_cfg,
        "source_df": source_df,
        "target_df": target_df,
        "source_segment_ids": source_segment_ids,
        "target_segment_ids": target_segment_ids,
        "source_scaled": source_all_scaled,
        "target_scaled": target_all_scaled,
        "x_source_train": x_train,
        "y_source_train": y_train,
        "x_source_val": x_val,
        "y_source_val": y_val,
        "feature_cols": feature_cols,
        "target_idx": target_idx,
        "feature_count": feature_count,
        "seq_len": seq_len,
        "raw_pred_len": raw_pred_len,
        "pred_len": pred_len,
        "target_mode": target_mode,
        "agg_mode": agg_mode,
        "scaler": scaler,
    }


def make_target_split(common: dict, adaptation_ratio: float):
    target_df = common["target_df"]
    n_target_total = len(target_df)
    if adaptation_ratio <= 0:
        train_end = 0
        val_end = 0
    else:
        adapt_end = max(1, int(n_target_total * adaptation_ratio))
        train_end = max(1, int(adapt_end * 0.8))
        val_end = adapt_end

    split = create_sequences_with_context(
        common["target_scaled"],
        target_idx=common["target_idx"],
        seq_len=common["seq_len"],
        pred_len=common["raw_pred_len"],
        train_end=train_end,
        val_end=val_end,
        segment_ids=common["target_segment_ids"],
    )

    result = {"ratio": adaptation_ratio, "train_end": train_end, "val_end": val_end}
    for split_name in ["train", "val", "test"]:
        x_arr, y_raw, forecast_start = split[split_name]
        result[f"x_{split_name}"] = x_arr
        result[f"forecast_start_{split_name}"] = forecast_start
        if len(x_arr):
            y_abs, y_model, last_target = finalize_sequence_targets(
                x_arr,
                y_raw,
                common["target_idx"],
                common["target_mode"],
                common["agg_mode"],
            )
            y_inv = inverse_target(y_abs, common["scaler"], common["target_idx"], common["feature_count"])
            dates = pd.Series(pd.Index(target_df.index)[forecast_start]).reset_index(drop=True)
        else:
            y_model = np.empty((0, common["pred_len"]), dtype=np.float32)
            last_target = np.empty((0, common["pred_len"]), dtype=np.float32)
            y_inv = np.empty((0, common["pred_len"]), dtype=np.float32)
            dates = pd.Series([], dtype="datetime64[ns]")
        result[f"y_{split_name}"] = y_model
        result[f"last_target_{split_name}"] = last_target
        result[f"y_{split_name}_inv"] = y_inv
        result[f"dates_{split_name}"] = dates
    return result


def evaluate_model(model, split: dict, common: dict, device: str, batch_size: int = 256):
    pred_scaled = run_inference(model, split["x_test"], device, batch_size=batch_size)
    pred_inv = scaled_prediction_to_inverse(
        pred_scaled,
        last_target=split["last_target_test"],
        target_mode=common["target_mode"],
        pred_len=common["pred_len"],
        scaler=common["scaler"],
        target_idx=common["target_idx"],
        feature_count=common["feature_count"],
    )
    metrics = evaluate_metrics_multistep(split["y_test_inv"], pred_inv)
    return pred_scaled, pred_inv, metrics


def evaluate_persistence(split: dict, common: dict):
    target_series_scaled = split["x_test"][:, :, common["target_idx"]]
    persistence_abs_scaled = np.repeat(target_series_scaled[:, -1][:, None], common["pred_len"], axis=1)
    persistence_scaled = encode_target_values(
        persistence_abs_scaled,
        split["last_target_test"],
        common["target_mode"],
    )
    persistence_inv = scaled_prediction_to_inverse(
        persistence_scaled,
        last_target=split["last_target_test"],
        target_mode=common["target_mode"],
        pred_len=common["pred_len"],
        scaler=common["scaler"],
        target_idx=common["target_idx"],
        feature_count=common["feature_count"],
    )
    return persistence_inv, evaluate_metrics_multistep(split["y_test_inv"], persistence_inv)


def row_from_metrics(variant: str, ratio: float, method: str, split: dict, metrics: dict, extra: dict | None = None):
    payload = {
        "variant": variant,
        "adaptation_ratio": ratio,
        "method": method,
        "target_train_windows": int(len(split["x_train"])),
        "target_val_windows": int(len(split["x_val"])),
        "target_test_windows": int(len(split["x_test"])),
        "test_start": "" if split["dates_test"].empty else str(split["dates_test"].iloc[0]),
        "test_end": "" if split["dates_test"].empty else str(split["dates_test"].iloc[-1]),
        **{k: float(v) for k, v in metrics.items()},
    }
    if extra:
        payload.update(extra)
    return payload


def plot_results(metrics_df: pd.DataFrame, output_dir: Path):
    plot_df = metrics_df[metrics_df["method"].isin(["source_only", "fewshot_finetuned", "persistence"])].copy()
    if plot_df.empty:
        return

    for metric in ["R2", "RMSE"]:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for (variant, method), group in plot_df.groupby(["variant", "method"]):
            group = group.sort_values("adaptation_ratio")
            label = f"{variant} / {method}"
            ax.plot(group["adaptation_ratio"] * 100, group[metric], marker="o", linewidth=2, label=label)
        ax.set_xlabel("Target-site adaptation budget (%)")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        ax.set_title(f"Few-shot cross-site adaptation: {metric}")
        fig.tight_layout()
        fig.savefig(output_dir / f"fewshot_cross_site_{metric.lower()}.png", dpi=220)
        fig.savefig(output_dir / f"fewshot_cross_site_{metric.lower()}.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot target-site adaptation for the current LSTM route.")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--variants", type=str, default="lstm_raw,lstm_improved")
    parser.add_argument("--ratios", type=str, default="0,0.05,0.10,0.20")
    parser.add_argument("--source-epochs", type=int, default=35)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--finetune-lr", type=float, default=3.0e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else ROOT / "outputs" / "cross_site_fewshot_adaptation" / stamp
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_dir = output_root / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    ratios = [float(item.strip()) for item in args.ratios.split(",") if item.strip()]
    rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for variant in variants:
        variant_dir = output_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(str(variant_dir), "INFO")
        logger.info("Device=%s variant=%s", device, variant)

        set_seed(42)
        cfg = make_config(variant, variant_dir, epochs=args.source_epochs, batch_size=args.batch_size)
        cfg["train"]["epochs"] = int(args.source_epochs)
        cfg["train"]["batch_size"] = int(args.batch_size)
        (config_dir / f"{variant}.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (variant_dir / "config_snapshot.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        common = prepare_common_data(cfg)
        model_name = str(cfg["experiment"]["primary_model"])
        source_train_loader, source_val_loader = build_loaders(
            common["x_source_train"],
            common["y_source_train"],
            common["x_source_val"],
            common["y_source_val"],
            batch_size=cfg["train"]["batch_size"],
            num_workers=cfg["train"]["num_workers"],
            shuffle_train=bool(cfg["train"].get("shuffle_train", True)),
        )
        source_model = build_model(
            model_name,
            input_size=common["x_source_train"].shape[2],
            cfg=common["model_cfg"],
            dataset_meta={"pred_len": common["pred_len"], "feature_cols": common["feature_cols"]},
        )
        source_model = train_single_model(model_name, source_model, source_train_loader, source_val_loader, cfg, logger, device)
        source_state = copy.deepcopy(source_model.state_dict())
        torch.save(
            {"variant": variant, "model_name": model_name, "state_dict": source_state, "config": cfg},
            variant_dir / f"{variant}_source_best.pt",
        )

        for ratio in ratios:
            split = make_target_split(common, ratio)
            if len(split["x_test"]) == 0:
                logger.warning("Skip ratio=%.3f because target test is empty.", ratio)
                continue

            _, _, source_metrics = evaluate_model(source_model, split, common, device)
            rows.append(row_from_metrics(variant, ratio, "source_only", split, source_metrics))

            _, persistence_metrics = evaluate_persistence(split, common)
            rows.append(row_from_metrics(variant, ratio, "persistence", split, persistence_metrics))

            if ratio <= 0:
                continue
            if len(split["x_train"]) == 0 or len(split["x_val"]) == 0:
                logger.warning(
                    "Skip fine-tuning ratio=%.3f due to insufficient target train/val windows: train=%d val=%d",
                    ratio,
                    len(split["x_train"]),
                    len(split["x_val"]),
                )
                continue

            ft_cfg = copy.deepcopy(cfg)
            ft_cfg["train"]["epochs"] = int(args.finetune_epochs)
            ft_cfg["train"]["learning_rate"] = float(args.finetune_lr)
            ft_cfg["train"]["early_stop_patience"] = min(6, int(ft_cfg["train"].get("early_stop_patience", 10)))
            ft_cfg["train"]["batch_size"] = min(int(args.batch_size), max(16, len(split["x_train"])))

            ft_train_loader, ft_val_loader = build_loaders(
                split["x_train"],
                split["y_train"],
                split["x_val"],
                split["y_val"],
                batch_size=ft_cfg["train"]["batch_size"],
                num_workers=ft_cfg["train"]["num_workers"],
                shuffle_train=True,
            )
            ft_model = build_model(
                model_name,
                input_size=common["x_source_train"].shape[2],
                cfg=common["model_cfg"],
                dataset_meta={"pred_len": common["pred_len"], "feature_cols": common["feature_cols"]},
            )
            ft_model.load_state_dict(source_state)
            ft_model = train_single_model(model_name, ft_model, ft_train_loader, ft_val_loader, ft_cfg, logger, device)
            _, _, ft_metrics = evaluate_model(ft_model, split, common, device)
            rows.append(
                row_from_metrics(
                    variant,
                    ratio,
                    "fewshot_finetuned",
                    split,
                    ft_metrics,
                    extra={"finetune_epochs": int(args.finetune_epochs), "finetune_lr": float(args.finetune_lr)},
                )
            )
            torch.save(
                {
                    "variant": variant,
                    "adaptation_ratio": ratio,
                    "model_name": model_name,
                    "state_dict": ft_model.state_dict(),
                    "config": ft_cfg,
                },
                variant_dir / f"{variant}_finetuned_{int(round(ratio * 100)):02d}pct.pt",
            )

    metrics_df = pd.DataFrame(rows)
    metrics_path = output_root / "fewshot_adaptation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    plot_results(metrics_df, output_root)

    best_df = (
        metrics_df.sort_values(["adaptation_ratio", "R2"], ascending=[True, False])
        .groupby("adaptation_ratio", as_index=False)
        .first()
    )
    best_df.to_csv(output_root / "fewshot_adaptation_best_by_ratio.csv", index=False, encoding="utf-8-sig")

    summary = {
        "output_root": str(output_root),
        "variants": variants,
        "ratios": ratios,
        "rows": int(len(metrics_df)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "best_by_ratio": best_df.to_dict(orient="records"),
    }
    (output_root / "fewshot_adaptation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote few-shot adaptation results to {output_root}")


if __name__ == "__main__":
    main()
