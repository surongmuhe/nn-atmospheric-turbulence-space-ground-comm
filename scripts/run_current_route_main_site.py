import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


RAW_FEATURES = ["LogCn2", "IRtemp", "SolarFlux", "H2Ocon", "Tair", "UmSonic", "Pair"]


def run_command(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def base_train_cfg() -> dict:
    return {
        "batch_size": 128,
        "epochs": 35,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "optimizer": "adamw",
        "scheduler": {"type": "plateau", "factor": 0.6, "patience": 2, "min_lr": 1.0e-5},
        "shuffle_train": True,
        "early_stop_patience": 10,
        "num_workers": 0,
        "grad_clip_norm": 1.0,
        "show_progress": False,
    }


def make_config(variant: str, output_dir: Path, target_mode: str) -> dict:
    if variant == "ann_current":
        model_name = "mlp"
        seq_len = 1
        hidden_sizes = [256, 128]
        mlp_dropout = 0.2
    elif variant == "ann_window":
        model_name = "mlp"
        seq_len = 72
        hidden_sizes = [256, 128]
        mlp_dropout = 0.2
    elif variant == "lstm_raw":
        model_name = "lstm"
        seq_len = 72
        hidden_sizes = None
        mlp_dropout = 0.2
    else:
        raise ValueError(f"Unknown variant: {variant}")

    cfg = {
        "project": {
            "name": f"current_route_main_{variant}",
            "seed": 42,
            "output_dir": str(output_dir),
            "log_level": "INFO",
        },
        "data": {
            "file_path": "data/Sklavounos and Cohn, Spring. 2022.xlsx",
            "datetime_col": "Date",
            "time_col": "Time",
            "feature_cols": RAW_FEATURES,
            "target_col": "LogCn2",
            "seq_len": seq_len,
            "pred_len": 6,
            "resample_rule": "5min",
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "target_mode": target_mode,
            "target_aggregation": "mean",
        },
        "preprocess": {
            "scaler": "standard",
            "status_col": "STATUS",
            "status_keep_values": [0],
            "drop_duplicate_timestamps": True,
            "sequence_split": {"use_context_history": True},
            "interpolate": {"enabled": False},
            "time_features": {
                "keep_temporal_hour": False,
                "hour_cyclical": False,
                "dayofyear_cyclical": False,
            },
        },
        "experiment": {
            "primary_model": model_name,
            "active_models": [model_name],
            "regime_split": {"enabled": False},
        },
        "train": base_train_cfg(),
        "fusion": {"enabled": False},
        "postprocess": {"residual_correction": {"enabled": False}},
        "models": {
            "mlp": {
                "hidden_sizes": hidden_sizes if hidden_sizes is not None else [256, 128],
                "dropout": mlp_dropout,
            },
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
        },
        "wandb": {"enabled": False, "project": None, "entity": None, "run_name": None},
    }
    if target_mode == "delta":
        cfg["train"]["loss"] = {"huber_weight": 0.35, "huber_beta": 0.2, "diff_weight": 0.0}
    return cfg


def collect_best_rows(output_root: Path) -> pd.DataFrame:
    rows = []
    for variant_dir in sorted(p for p in output_root.iterdir() if p.is_dir() and p.name != "_configs"):
        metrics_path = variant_dir / "metrics_all_models.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        metrics = metrics.sort_values("RMSE", ascending=True).reset_index(drop=True)
        best = metrics.iloc[0].to_dict()
        best["variant"] = variant_dir.name
        best["source"] = str(metrics_path)
        rows.append(best)
    return pd.DataFrame(rows)


def load_final_improved_row() -> dict:
    metrics_path = ROOT / "outputs" / "thesis_lstm_final_best_seq72" / "formal_20260424" / "metrics_all_models.csv"
    metrics = pd.read_csv(metrics_path)
    best = metrics.sort_values("RMSE", ascending=True).iloc[0].to_dict()
    best["variant"] = "lstm_improved_final"
    best["source"] = str(metrics_path)
    return best


def write_summary(output_root: Path, best_df: pd.DataFrame, target_mode: str) -> None:
    lines = [
        "# Current Route Main-Site Results",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"All variants target the future 30-minute mean of `LogCn2` with target mode `{target_mode}` for the newly run baselines. `ann_current`, `ann_window`, and `lstm_raw` use raw features; `lstm_improved_final` reuses the finalized seq72 Improved LSTM result.",
        "",
    ]
    if best_df.empty:
        lines.append("No metrics were collected.")
    else:
        display_cols = [c for c in ["variant", "model", "RMSE", "MAE", "MAPE", "R2", "source"] if c in best_df.columns]
        lines.append(best_df[display_cols].to_markdown(index=False))
    (output_root / "current_route_main_site_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align main-site ANN/LSTM baselines with the final future-mean task.")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--variants", type=str, default="ann_current,ann_window,lstm_raw")
    parser.add_argument("--target-mode", type=str, default="absolute", choices=["absolute", "delta"])
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else ROOT / "outputs" / "current_route_main_site" / stamp
    output_root = output_root.resolve()
    config_dir = output_root / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    for variant in variants:
        cfg = make_config(variant, output_root / variant, target_mode=args.target_mode)
        cfg_path = config_dir / f"{variant}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_experiments.py"),
            "--config",
            str(cfg_path),
            "--output-dir",
            str(output_root / variant),
        ]
        run_command(cmd)

    best_df = collect_best_rows(output_root)
    best_df = pd.concat([best_df, pd.DataFrame([load_final_improved_row()])], ignore_index=True)
    best_df = best_df.sort_values("RMSE", ascending=True).reset_index(drop=True)
    best_df.to_csv(output_root / "current_route_main_site_metrics.csv", index=False, encoding="utf-8-sig")
    write_summary(output_root, best_df, target_mode=args.target_mode)
    (output_root / "current_route_main_site_summary.json").write_text(
        json.dumps({"output_root": str(output_root), "rows": int(len(best_df))}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
