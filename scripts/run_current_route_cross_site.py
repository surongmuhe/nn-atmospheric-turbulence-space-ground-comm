import argparse
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


RAW_FEATURES = ["LogCn2", "Tair", "Pair", "UmSonic", "is_daylight"]
IMPROVED_FEATURES = RAW_FEATURES + ["hour_sin", "hour_cos"]
SENSOR_COLS = ["LogCn2", "Tair", "Pair", "UmSonic"]


BASE_MODELS = {
    "mlp": {"hidden_sizes": [256, 128], "dropout": 0.2},
    "lstm": {
        "hidden_size": 96,
        "num_layers": 2,
        "dropout": 0.1,
        "bidirectional": False,
        "attention_pool": False,
        "input_proj_size": None,
        "ar_window": 0,
        "context_window": 12,
    },
    "gru": {"hidden_size": 64, "num_layers": 2, "dropout": 0.1},
    "transformer": {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "ff_dim": 128,
        "dropout": 0.1,
        "use_positional_encoding": True,
        "pooling": "last",
    },
    "tcn": {"channels": [64, 64], "kernel_size": 3, "dropout": 0.2},
    "cnn_lstm": {"cnn_channels": 64, "lstm_hidden": 128, "lstm_layers": 2, "dropout": 0.2},
}


def make_preprocess(improved: bool) -> dict:
    cfg = {
        "scaler": "standard",
        "source_status_col": "STATUS",
        "source_status_keep_values": [0],
        "target_status_col": None,
        "target_status_keep_values": None,
        "drop_duplicate_timestamps": True,
        "daylight_proxy": {
            "enabled": True,
            "overwrite_existing": True,
            "start_hour": 6.0,
            "end_hour": 18.0,
        },
        "interpolate": {"enabled": improved, "method": "time", "limit": 3},
        "missing_indicators": {"enabled": False, "columns": SENSOR_COLS},
        "missing_context_features": {"enabled": False, "columns": SENSOR_COLS},
        "long_gap_mask": {"enabled": False, "columns": SENSOR_COLS},
        "domain_features": {"enabled": False},
        "solar_regime_features": {"enabled": False},
        "diff_features": {"enabled": False, "columns": SENSOR_COLS, "periods": [1]},
        "rolling_features": {
            "enabled": False,
            "columns": SENSOR_COLS,
            "windows": [3, 12],
            "include_mean": True,
            "include_std": True,
            "include_min": False,
            "include_max": False,
        },
        "clip_quantiles": {"enabled": improved, "lower": 0.005, "upper": 0.995, "exclude_cols": ["LogCn2"]},
        "time_features": {"keep_temporal_hour": improved, "hour_cyclical": improved, "dayofyear_cyclical": False},
    }
    if improved:
        cfg.update(
            {
                "missing_indicators": {"enabled": True, "columns": SENSOR_COLS},
                "missing_context_features": {
                    "enabled": True,
                    "columns": SENSOR_COLS,
                    "include_any_missing": True,
                    "include_missing_fraction": True,
                    "include_max_run_log1p": True,
                    "include_recent_missing": True,
                    "recent_windows": [3, 12],
                },
                "long_gap_mask": {"enabled": True, "columns": SENSOR_COLS, "max_missing_run": 6, "pad_steps": 1},
                "diff_features": {"enabled": True, "columns": ["LogCn2", "Tair", "UmSonic"], "periods": [1]},
                "rolling_features": {
                    "enabled": True,
                    "columns": ["LogCn2", "Tair", "UmSonic"],
                    "windows": [3, 12],
                    "include_mean": True,
                    "include_std": True,
                    "include_min": False,
                    "include_max": False,
                },
            }
        )
    return cfg


def make_config(variant: str, output_dir: Path, epochs: int, batch_size: int) -> dict:
    if variant == "ann_current":
        model_name, seq_len, improved = "mlp", 1, False
    elif variant == "ann_window":
        model_name, seq_len, improved = "mlp", 72, False
    elif variant == "lstm_raw":
        model_name, seq_len, improved = "lstm", 72, False
    elif variant == "lstm_improved":
        model_name, seq_len, improved = "lstm", 72, True
    else:
        raise ValueError(f"Unknown variant: {variant}")

    models = deepcopy(BASE_MODELS)
    if improved:
        models["lstm"].update(
            {
                "hidden_size": 128,
                "num_layers": 3,
                "dropout": 0.15,
                "bidirectional": True,
                "attention_pool": True,
                "input_proj_size": 64,
                "ar_window": 12,
                "context_window": 12,
            }
        )

    return {
        "project": {
            "name": f"current_route_cross_site_{variant}",
            "seed": 42,
            "output_dir": str(output_dir),
            "log_level": "INFO",
        },
        "data": {
            "source_file_path": "data/Sklavounos and Cohn, Spring. 2022.xlsx",
            "source_datetime_col": "Date",
            "source_time_col": "Time",
            "target_file_path": "data/dataWfon0U_ml_ready.csv",
            "target_datetime_col": "time",
            "target_time_col": None,
            "feature_cols": IMPROVED_FEATURES if improved else RAW_FEATURES,
            "target_col": "LogCn2",
            "seq_len": seq_len,
            "pred_len": 6,
            "resample_rule": "5min",
            "source_train_ratio": 0.7,
            "source_val_ratio": 0.15,
            "target_mode": "delta",
            "target_aggregation": "mean",
        },
        "preprocess": make_preprocess(improved),
        "experiment": {
            "primary_model": model_name,
            "active_models": [model_name],
            "regime_split": {"enabled": False},
        },
        "train": {
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "optimizer": "adamw",
            "scheduler": {"type": "plateau", "factor": 0.6, "patience": 2, "min_lr": 1.0e-5},
            "shuffle_train": True,
            "early_stop_patience": 10,
            "num_workers": 0,
            "grad_clip_norm": 1.0,
            "show_progress": False,
            "loss": {"huber_weight": 0.35, "huber_beta": 0.2, "diff_weight": 0.0},
        },
        "fusion": {"enabled": False},
        "models": models,
        "wandb": {"enabled": False, "project": None, "entity": None, "run_name": None},
    }


def collect_results(output_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_root.glob("*/metrics_all_models.csv")):
        variant = path.parent.name
        metrics = pd.read_csv(path)
        for _, row in metrics.iterrows():
            payload = row.to_dict()
            payload["variant"] = variant
            rows.append(payload)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    order = ["variant", "model"] + [c for c in df.columns if c not in {"variant", "model"}]
    return df[order].sort_values(["RMSE", "variant", "model"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the current ANN/LSTM/Improved-LSTM route in strict cross-site mode.")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--variants",
        type=str,
        default="ann_current,ann_window,lstm_raw,lstm_improved",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else ROOT / "outputs" / "current_route_cross_site" / stamp
    output_root = output_root.resolve()
    config_dir = output_root / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    for variant in variants:
        out_dir = output_root / variant
        cfg = make_config(variant, out_dir, epochs=args.epochs, batch_size=args.batch_size)
        cfg_path = config_dir / f"{variant}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        cmd = [
            sys.executable,
            "scripts/run_cross_site_transfer.py",
            "--config",
            str(cfg_path),
            "--output-dir",
            str(out_dir),
        ]
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)

    metrics = collect_results(output_root)
    metrics.to_csv(output_root / "current_route_cross_site_metrics.csv", index=False, encoding="utf-8-sig")

    lines = ["# Current Route Strict Cross-Site Results", ""]
    if metrics.empty:
        lines.append("No metrics were collected.")
    else:
        cols = [c for c in ["variant", "model", "RMSE", "MAE", "MAPE", "R2", "RMSE_h1"] if c in metrics.columns]
        lines.append(metrics[cols].to_markdown(index=False))
    (output_root / "current_route_cross_site_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summary to {output_root}", flush=True)


if __name__ == "__main__":
    main()
