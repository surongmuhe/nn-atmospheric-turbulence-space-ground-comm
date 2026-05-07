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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_current_route_cross_site import BASE_MODELS, make_preprocess


RAW_FEATURES = ["LogCn2", "Tair", "Pair", "UmSonic", "is_daylight"]
IMPROVED_FEATURES = RAW_FEATURES + ["hour_sin", "hour_cos"]
SENSOR_COLS = ["LogCn2", "Tair", "Pair", "UmSonic"]


def make_config(variant: str, output_dir: Path, epochs: int, batch_size: int, target_mode: str) -> dict:
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
    if variant == "lstm_raw":
        models["lstm"].update(
            {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.1,
                "bidirectional": False,
                "attention_pool": False,
                "input_proj_size": None,
                "ar_window": 0,
                "context_window": 12,
            }
        )
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

    preprocess = make_preprocess(improved)
    preprocess["status_col"] = None
    preprocess["status_keep_values"] = None
    preprocess["sequence_split"] = {"use_context_history": True}
    for key in ["missing_indicators", "missing_context_features", "long_gap_mask"]:
        if key in preprocess and isinstance(preprocess[key], dict):
            preprocess[key]["columns"] = list(SENSOR_COLS)
    if "diff_features" in preprocess and isinstance(preprocess["diff_features"], dict):
        preprocess["diff_features"]["columns"] = ["LogCn2", "Tair", "UmSonic"]
    if "rolling_features" in preprocess and isinstance(preprocess["rolling_features"], dict):
        preprocess["rolling_features"]["columns"] = ["LogCn2", "Tair", "UmSonic"]

    return {
        "project": {
            "name": f"external_site_four_model_{variant}",
            "seed": 42,
            "output_dir": str(output_dir),
            "log_level": "INFO",
        },
        "data": {
            "file_path": "data/dataWfon0U_ml_ready.csv",
            "datetime_col": "time",
            "time_col": None,
            "feature_cols": IMPROVED_FEATURES if improved else RAW_FEATURES,
            "target_col": "LogCn2",
            "seq_len": seq_len,
            "pred_len": 6,
            "resample_rule": "5min",
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "target_mode": target_mode,
            "target_aggregation": "mean",
        },
        "preprocess": preprocess,
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
        "postprocess": {"residual_correction": {"enabled": False}},
        "models": models,
        "wandb": {"enabled": False, "project": None, "entity": None, "run_name": None},
    }


def collect_results(output_root: Path) -> pd.DataFrame:
    rows = []
    primary = {
        "ann_current": "mlp",
        "ann_window": "mlp",
        "lstm_raw": "lstm",
        "lstm_improved": "lstm",
    }
    for variant, model_name in primary.items():
        metrics_path = output_root / variant / "metrics_all_models.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        subset = metrics[metrics["model"] == model_name]
        if subset.empty:
            subset = metrics.sort_values("RMSE")
        row = subset.iloc[0].to_dict()
        row["variant"] = variant
        row["source"] = str(metrics_path)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    order = ["variant", "model"] + [c for c in df.columns if c not in {"variant", "model"}]
    return df[order].sort_values("RMSE").reset_index(drop=True)


def write_summary(output_root: Path, metrics: pd.DataFrame) -> None:
    lines = [
        "# External-Site Four-Model Route",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This experiment retrains the four-model route on the external station with the same chronological split.",
        "It is an external-site supervised replication, not a zero-shot transfer leaderboard.",
        "",
    ]
    if metrics.empty:
        lines.append("No metrics were collected.")
    else:
        cols = [c for c in ["variant", "model", "RMSE", "MAE", "MAPE", "R2", "RMSE_h1"] if c in metrics.columns]
        lines.append(metrics[cols].to_markdown(index=False))
    (output_root / "external_site_four_model_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the four-model route on the external site with supervised retraining.")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--variants", type=str, default="ann_current,ann_window,lstm_raw,lstm_improved")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-mode", type=str, default="delta", choices=["absolute", "delta"])
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else ROOT / "outputs" / "external_site_four_model_route" / stamp
    output_root = output_root.resolve()
    config_dir = output_root / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    for variant in variants:
        out_dir = output_root / variant
        cfg = make_config(
            variant,
            out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            target_mode=args.target_mode,
        )
        cfg_path = config_dir / f"{variant}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        cmd = [
            sys.executable,
            "scripts/run_experiments.py",
            "--config",
            str(cfg_path),
            "--output-dir",
            str(out_dir),
        ]
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)

    metrics = collect_results(output_root)
    metrics.to_csv(output_root / "external_site_four_model_metrics.csv", index=False, encoding="utf-8-sig")
    write_summary(output_root, metrics)
    summary = {
        "output_root": str(output_root),
        "variants": variants,
        "rows": int(len(metrics)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output_root / "external_site_four_model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote external-site four-model outputs to {output_root}", flush=True)


if __name__ == "__main__":
    main()
