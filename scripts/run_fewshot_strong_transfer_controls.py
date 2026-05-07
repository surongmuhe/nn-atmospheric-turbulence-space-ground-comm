from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cross_site_fewshot_adaptation import (  # noqa: E402
    evaluate_model,
    evaluate_persistence,
    make_target_split,
    prepare_common_data,
    row_from_metrics,
)
from scripts.run_current_route_cross_site import make_config  # noqa: E402
from src.data import set_seed  # noqa: E402
from src.experiment import evaluate_metrics_multistep  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.trainer import build_loaders, train_single_model  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402


MODEL_LABELS = {
    "lstm_raw": "LSTM-raw",
    "tcn": "TCN",
    "tft": "TFT",
    "patchtst": "PatchTST",
    "lstm_improved": "Improved LSTM",
}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_true - y_pred
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err**2) / denom) if denom > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "MSE": mse, "R2": r2}


def make_transfer_config(variant: str, output_dir: Path, epochs: int, batch_size: int) -> dict:
    if variant in {"lstm_raw", "lstm_improved"}:
        cfg = make_config(variant, output_dir, epochs=epochs, batch_size=batch_size)
    elif variant in {"tcn", "tft", "patchtst"}:
        # Strong non-LSTM baselines use the same common enhanced cross-site feature pipeline as
        # Improved LSTM so the few-shot comparison tests the model class, not feature deprivation.
        cfg = make_config("lstm_improved", output_dir, epochs=epochs, batch_size=batch_size)
        cfg["project"]["name"] = f"fewshot_strong_transfer_{variant}"
        cfg["experiment"]["primary_model"] = variant
        cfg["experiment"]["active_models"] = [variant]
        cfg["models"].setdefault("tcn", {})
        cfg["models"]["tcn"].update({"channels": [96, 96, 128], "kernel_size": 5, "dropout": 0.15})
        cfg["models"]["tft"] = {
            "hidden_size": 128,
            "lstm_layers": 1,
            "num_heads": 4,
            "dropout": 0.1,
        }
        cfg["models"]["patchtst"] = {
            "patch_len": 6,
            "stride": 3,
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            "ff_dim": 256,
            "dropout": 0.1,
            "pooling": "mean",
            "max_patches": 128,
        }
    else:
        raise ValueError(f"Unknown variant: {variant}")
    cfg["project"]["output_dir"] = str(output_dir)
    cfg["train"]["epochs"] = int(epochs)
    cfg["train"]["batch_size"] = int(batch_size)
    cfg["train"]["show_progress"] = False
    return cfg


def save_predictions(
    output_dir: Path,
    variant: str,
    ratio: float,
    method: str,
    split: dict,
    pred_inv: np.ndarray,
) -> pd.DataFrame:
    dates = split["dates_test"].reset_index(drop=True)
    true = np.asarray(split["y_test_inv"], dtype=np.float64)
    pred = np.asarray(pred_inv, dtype=np.float64)
    if true.ndim == 1:
        true = true[:, None]
    if pred.ndim == 1:
        pred = pred[:, None]
    df = pd.DataFrame(
        {
            "datetime": dates,
            "variant": variant,
            "model_label": MODEL_LABELS.get(variant, variant),
            "adaptation_ratio": float(ratio),
            "method": method,
            "y_true": true[:, 0],
            "y_pred": pred[:, 0],
        }
    )
    path = output_dir / f"{variant}_{method}_{int(round(ratio * 100)):02d}pct_predictions.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def train_source_model(variant: str, cfg: dict, common: dict, logger, device: str):
    model_name = str(cfg["experiment"]["primary_model"])
    train_loader, val_loader = build_loaders(
        common["x_source_train"],
        common["y_source_train"],
        common["x_source_val"],
        common["y_source_val"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        shuffle_train=bool(cfg["train"].get("shuffle_train", True)),
    )
    model = build_model(
        model_name,
        input_size=common["x_source_train"].shape[2],
        cfg=common["model_cfg"],
        dataset_meta={"pred_len": common["pred_len"], "feature_cols": common["feature_cols"]},
    )
    model = train_single_model(model_name, model, train_loader, val_loader, cfg, logger, device)
    return model, copy.deepcopy(model.state_dict())


def fine_tune_model(
    variant: str,
    cfg: dict,
    common: dict,
    source_state: dict,
    split: dict,
    epochs: int,
    lr: float,
    batch_size: int,
    logger,
    device: str,
):
    model_name = str(cfg["experiment"]["primary_model"])
    ft_cfg = copy.deepcopy(cfg)
    ft_cfg["train"]["epochs"] = int(epochs)
    ft_cfg["train"]["learning_rate"] = float(lr)
    ft_cfg["train"]["early_stop_patience"] = min(6, int(ft_cfg["train"].get("early_stop_patience", 10)))
    ft_cfg["train"]["batch_size"] = min(int(batch_size), max(16, len(split["x_train"])))
    train_loader, val_loader = build_loaders(
        split["x_train"],
        split["y_train"],
        split["x_val"],
        split["y_val"],
        batch_size=ft_cfg["train"]["batch_size"],
        num_workers=ft_cfg["train"]["num_workers"],
        shuffle_train=True,
    )
    model = build_model(
        model_name,
        input_size=common["x_source_train"].shape[2],
        cfg=common["model_cfg"],
        dataset_meta={"pred_len": common["pred_len"], "feature_cols": common["feature_cols"]},
    )
    model.load_state_dict(source_state)
    return train_single_model(model_name, model, train_loader, val_loader, ft_cfg, logger, device)


def collect_aligned_metrics(predictions: list[pd.DataFrame], output_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    if not predictions:
        return pd.DataFrame()
    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df.to_csv(output_root / "fewshot_strong_transfer_all_predictions.csv", index=False, encoding="utf-8-sig")

    for ratio, ratio_df in pred_df.groupby("adaptation_ratio"):
        methods = sorted(ratio_df["method_key"].unique())
        common_dates: set[pd.Timestamp] | None = None
        for method_key in methods:
            dates = set(pd.to_datetime(ratio_df.loc[ratio_df["method_key"] == method_key, "datetime"]))
            common_dates = dates if common_dates is None else common_dates.intersection(dates)
        common_dates = common_dates or set()
        if not common_dates:
            continue
        common_index = pd.Index(sorted(common_dates))
        for method_key, group in ratio_df.groupby("method_key"):
            current = group.copy()
            current["datetime"] = pd.to_datetime(current["datetime"])
            current = current[current["datetime"].isin(common_index)].sort_values("datetime")
            metrics = regression_metrics(current["y_true"].to_numpy(), current["y_pred"].to_numpy())
            variant = str(current["variant"].iloc[0])
            method = str(current["method"].iloc[0])
            rows.append(
                {
                    "adaptation_ratio": float(ratio),
                    "variant": variant,
                    "model_label": MODEL_LABELS.get(variant, variant),
                    "method": method,
                    "method_key": method_key,
                    "common_test_timestamps": int(len(current)),
                    "test_start": str(current["datetime"].iloc[0]),
                    "test_end": str(current["datetime"].iloc[-1]),
                    **metrics,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["adaptation_ratio", "method", "RMSE", "variant"]).reset_index(drop=True)
        out.to_csv(output_root / "fewshot_strong_transfer_aligned_metrics.csv", index=False, encoding="utf-8-sig")

        ft = out[out["method"] == "fewshot_finetuned"].copy()
        if not ft.empty:
            best = ft.sort_values(["adaptation_ratio", "R2"], ascending=[True, False]).groupby("adaptation_ratio").head(1)
            best.to_csv(output_root / "fewshot_strong_transfer_best_finetuned_by_ratio.csv", index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run same-timestamp few-shot transfer controls for strong time-series baselines.")
    parser.add_argument("--output-root", default="outputs/fewshot_strong_transfer_controls_20260427")
    parser.add_argument("--variants", default="lstm_raw,tcn,tft,patchtst,lstm_improved")
    parser.add_argument("--ratios", default="0.05,0.10,0.20")
    parser.add_argument("--source-epochs", type=int, default=35)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--finetune-lr", type=float, default=3.0e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_dir = output_root / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for variant in variants:
        set_seed(args.seed)
        variant_dir = output_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(str(variant_dir), "INFO")
        logger.info("Running variant=%s on device=%s", variant, device)
        cfg = make_transfer_config(variant, variant_dir, args.source_epochs, args.batch_size)
        (config_dir / f"{variant}.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        common = prepare_common_data(cfg)
        source_model, source_state = train_source_model(variant, cfg, common, logger, device)

        for ratio in ratios:
            split = make_target_split(common, ratio)
            if len(split["x_test"]) == 0:
                logger.warning("Skip variant=%s ratio=%.2f because target test is empty.", variant, ratio)
                continue

            pred_scaled, pred_inv, source_metrics = evaluate_model(source_model, split, common, device)
            all_metric_rows.append(row_from_metrics(variant, ratio, "source_only", split, source_metrics))
            source_df = save_predictions(variant_dir, variant, ratio, "source_only", split, pred_inv)
            source_df["method_key"] = f"{variant}|source_only"
            prediction_frames.append(source_df)

            persistence_pred, persistence_metrics = evaluate_persistence(split, common)
            all_metric_rows.append(row_from_metrics(variant, ratio, "persistence", split, persistence_metrics))
            if variant == "lstm_improved":
                persistence_df = save_predictions(variant_dir, variant, ratio, "persistence", split, persistence_pred)
                persistence_df["method_key"] = "persistence|persistence"
                prediction_frames.append(persistence_df)

            if len(split["x_train"]) == 0 or len(split["x_val"]) == 0:
                logger.warning("Skip fine-tuning variant=%s ratio=%.2f due to empty target train/val.", variant, ratio)
                continue

            ft_model = fine_tune_model(
                variant,
                cfg,
                common,
                source_state,
                split,
                args.finetune_epochs,
                args.finetune_lr,
                args.batch_size,
                logger,
                device,
            )
            _, ft_pred_inv, ft_metrics = evaluate_model(ft_model, split, common, device)
            all_metric_rows.append(
                row_from_metrics(
                    variant,
                    ratio,
                    "fewshot_finetuned",
                    split,
                    ft_metrics,
                    extra={"finetune_epochs": int(args.finetune_epochs), "finetune_lr": float(args.finetune_lr)},
                )
            )
            ft_df = save_predictions(variant_dir, variant, ratio, "fewshot_finetuned", split, ft_pred_inv)
            ft_df["method_key"] = f"{variant}|fewshot_finetuned"
            prediction_frames.append(ft_df)

    raw_metrics = pd.DataFrame(all_metric_rows)
    raw_metrics.to_csv(output_root / "fewshot_strong_transfer_raw_metrics.csv", index=False, encoding="utf-8-sig")
    aligned = collect_aligned_metrics(prediction_frames, output_root)

    summary = {
        "output_root": str(output_root),
        "variants": variants,
        "ratios": ratios,
        "source_epochs": int(args.source_epochs),
        "finetune_epochs": int(args.finetune_epochs),
        "finetune_lr": float(args.finetune_lr),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_rows": int(len(raw_metrics)),
        "aligned_rows": int(len(aligned)),
    }
    (output_root / "fewshot_strong_transfer_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
