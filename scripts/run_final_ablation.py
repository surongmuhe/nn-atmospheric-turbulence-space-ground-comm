import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import set_seed
from src.experiment import run_experiment
from src.utils.config import load_config
from src.utils.logger import setup_logger


RAW_FEATURES = ["LogCn2", "IRtemp", "SolarFlux", "H2Ocon", "Tair", "UmSonic", "Pair"]
TIME_FEATURES = RAW_FEATURES + ["hour_sin", "hour_cos"]


def set_nested(cfg: dict, path: list[str], value) -> None:
    cur = cfg
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def configure_clean_common(cfg: dict) -> None:
    cfg["experiment"]["primary_model"] = "lstm"
    cfg["experiment"]["active_models"] = ["lstm"]
    set_nested(cfg, ["fusion", "enabled"], False)
    set_nested(cfg, ["postprocess", "residual_correction", "enabled"], False)
    set_nested(cfg, ["preprocess", "missing_indicators", "enabled"], False)
    set_nested(cfg, ["preprocess", "missing_context_features", "enabled"], False)
    set_nested(cfg, ["preprocess", "long_gap_mask", "enabled"], True)
    set_nested(cfg, ["preprocess", "clip_quantiles", "enabled"], True)
    set_nested(cfg, ["preprocess", "diff_features", "enabled"], False)
    set_nested(cfg, ["preprocess", "rolling_features", "enabled"], False)
    set_nested(cfg, ["preprocess", "domain_features", "enabled"], False)
    set_nested(cfg, ["preprocess", "solar_regime_features", "enabled"], False)
    set_nested(cfg, ["preprocess", "time_features", "keep_temporal_hour"], False)
    set_nested(cfg, ["preprocess", "time_features", "hour_cyclical"], False)
    set_nested(cfg, ["data", "feature_cols"], list(RAW_FEATURES))
    set_nested(cfg, ["data", "resample_rule"], "5min")
    set_nested(cfg, ["data", "seq_len"], 144)
    set_nested(cfg, ["data", "pred_len"], 6)
    set_nested(cfg, ["data", "target_aggregation"], "mean")
    set_nested(cfg, ["train", "loss", "diff_weight"], 0.0)


def enable_time_features(cfg: dict) -> None:
    set_nested(cfg, ["data", "feature_cols"], list(TIME_FEATURES))
    set_nested(cfg, ["preprocess", "time_features", "keep_temporal_hour"], True)
    set_nested(cfg, ["preprocess", "time_features", "hour_cyclical"], True)


def enable_physics_features(cfg: dict) -> None:
    enable_time_features(cfg)
    set_nested(cfg, ["preprocess", "domain_features", "enabled"], True)
    set_nested(cfg, ["preprocess", "solar_regime_features", "enabled"], True)


def enable_trend_features(cfg: dict) -> None:
    enable_physics_features(cfg)
    set_nested(cfg, ["preprocess", "diff_features", "enabled"], True)
    set_nested(cfg, ["preprocess", "diff_features", "periods"], [1])
    set_nested(cfg, ["preprocess", "rolling_features", "enabled"], True)
    set_nested(cfg, ["preprocess", "rolling_features", "windows"], [3, 12])


def enable_time_trend_features(cfg: dict) -> None:
    enable_time_features(cfg)
    set_nested(cfg, ["preprocess", "diff_features", "enabled"], True)
    set_nested(cfg, ["preprocess", "diff_features", "periods"], [1])
    set_nested(cfg, ["preprocess", "rolling_features", "enabled"], True)
    set_nested(cfg, ["preprocess", "rolling_features", "windows"], [3, 12])


def apply_ablation(base_cfg: dict, variant: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["project"]["name"] = f"final_ablation_{variant}"
    configure_clean_common(cfg)

    if variant == "01_single_point_final":
        enable_trend_features(cfg)
        set_nested(cfg, ["data", "pred_len"], 1)
        set_nested(cfg, ["data", "target_mode"], "delta")
        set_nested(cfg, ["data", "target_aggregation"], "multistep")
        return cfg

    if variant == "02_future_mean_raw_absolute":
        set_nested(cfg, ["data", "target_mode"], "absolute")
        return cfg

    if variant == "03_future_mean_raw_delta":
        set_nested(cfg, ["data", "target_mode"], "delta")
        return cfg

    if variant == "04_future_mean_time_delta":
        set_nested(cfg, ["data", "target_mode"], "delta")
        enable_time_features(cfg)
        return cfg

    if variant == "05_future_mean_physics_delta":
        set_nested(cfg, ["data", "target_mode"], "delta")
        enable_physics_features(cfg)
        return cfg

    if variant == "06_future_mean_final":
        set_nested(cfg, ["data", "target_mode"], "delta")
        enable_trend_features(cfg)
        return cfg

    if variant == "07_final_no_attention":
        set_nested(cfg, ["data", "target_mode"], "delta")
        enable_trend_features(cfg)
        set_nested(cfg, ["models", "lstm", "attention_pool"], False)
        return cfg

    if variant == "08_final_no_ar_shortcut":
        set_nested(cfg, ["data", "target_mode"], "delta")
        enable_trend_features(cfg)
        set_nested(cfg, ["models", "lstm", "ar_window"], 0)
        set_nested(cfg, ["models", "lstm", "context_window"], 2)
        return cfg

    if variant == "09_final_vanilla_lstm":
        set_nested(cfg, ["data", "target_mode"], "delta")
        enable_trend_features(cfg)
        cfg["experiment"]["primary_model"] = "vanilla_lstm"
        cfg["experiment"]["active_models"] = ["vanilla_lstm"]
        return cfg

    if variant == "10_final_mlp_window":
        set_nested(cfg, ["data", "target_mode"], "delta")
        enable_trend_features(cfg)
        cfg["experiment"]["primary_model"] = "mlp"
        cfg["experiment"]["active_models"] = ["mlp"]
        return cfg

    if variant == "11_final_no_attention_no_ar":
        set_nested(cfg, ["data", "target_mode"], "delta")
        enable_trend_features(cfg)
        set_nested(cfg, ["models", "lstm", "attention_pool"], False)
        set_nested(cfg, ["models", "lstm", "ar_window"], 0)
        set_nested(cfg, ["models", "lstm", "context_window"], 2)
        return cfg

    if variant == "12_time_trend_delta":
        set_nested(cfg, ["data", "target_mode"], "delta")
        enable_time_trend_features(cfg)
        return cfg

    if variant == "13_time_trend_no_attention":
        set_nested(cfg, ["data", "target_mode"], "delta")
        enable_time_trend_features(cfg)
        set_nested(cfg, ["models", "lstm", "attention_pool"], False)
        return cfg

    if variant == "14_final_no_bidirectional":
        set_nested(cfg, ["data", "target_mode"], "delta")
        enable_trend_features(cfg)
        set_nested(cfg, ["models", "lstm", "bidirectional"], False)
        return cfg

    raise ValueError(f"Unsupported ablation variant: {variant}")


def collect_metrics(output_root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        metrics_path = run_dir / "metrics_all_models.csv"
        snapshot_path = run_dir / "config_snapshot.json"
        if not metrics_path.exists():
            continue

        cfg = {}
        if snapshot_path.exists():
            cfg = json.loads(snapshot_path.read_text(encoding="utf-8"))
        metrics_df = pd.read_csv(metrics_path)
        for _, row in metrics_df.iterrows():
            model = str(row.get("model", ""))
            if model.startswith("naive_"):
                continue
            payload = row.to_dict()
            payload["variant"] = run_dir.name
            payload["seq_len"] = cfg.get("data", {}).get("seq_len")
            payload["pred_len"] = cfg.get("data", {}).get("pred_len")
            payload["target_mode"] = cfg.get("data", {}).get("target_mode")
            payload["target_aggregation"] = cfg.get("data", {}).get("target_aggregation")
            payload["feature_count"] = len(cfg.get("data", {}).get("feature_cols", []))
            payload["attention_pool"] = cfg.get("models", {}).get("lstm", {}).get("attention_pool")
            payload["ar_window"] = cfg.get("models", {}).get("lstm", {}).get("ar_window")
            rows.append(payload)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    preferred = [
        "variant",
        "model",
        "RMSE",
        "MAE",
        "MAPE",
        "R2",
        "seq_len",
        "pred_len",
        "target_mode",
        "target_aggregation",
        "feature_count",
        "attention_pool",
        "ar_window",
    ]
    ordered = preferred + [col for col in df.columns if col not in preferred]
    return df[ordered].sort_values(["variant", "model"]).reset_index(drop=True)


def write_summary(output_root: Path, metrics_df: pd.DataFrame) -> None:
    lines = [
        "# Final Improved LSTM Ablation Summary",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Design",
        "",
        "All variants use the same chronological split and the same Sklavounos spring LogCn2 data. "
        "Variants 02-10 predict the 30-minute future mean of LogCn2 from a 12-hour input window; "
        "variant 01 is retained as a single-point target reference.",
        "",
        "## Metrics",
        "",
    ]
    if metrics_df.empty:
        lines.append("No metrics were collected.")
    else:
        display_cols = [
            col
            for col in [
                "variant",
                "model",
                "RMSE",
                "MAE",
                "MAPE",
                "R2",
                "seq_len",
                "pred_len",
                "target_mode",
                "target_aggregation",
                "attention_pool",
                "ar_window",
            ]
            if col in metrics_df.columns
        ]
        lines.append(metrics_df[display_cols].to_markdown(index=False))
        best = metrics_df.sort_values(["R2", "RMSE"], ascending=[False, True]).iloc[0].to_dict()
        lines.extend(
            [
                "",
                "## Best Variant",
                "",
                f"- Variant: `{best.get('variant')}`",
                f"- Model: `{best.get('model')}`",
                f"- RMSE: `{best.get('RMSE')}`",
                f"- R2: `{best.get('R2')}`",
            ]
        )

    (output_root / "final_ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final ablations for the thesis Improved LSTM route.")
    parser.add_argument("--config", type=str, default="configs/thesis_lstm_future_mean.yaml")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument(
        "--variants",
        type=str,
        default=(
            "01_single_point_final,"
            "02_future_mean_raw_absolute,"
            "03_future_mean_raw_delta,"
            "04_future_mean_time_delta,"
            "05_future_mean_physics_delta,"
            "06_future_mean_final,"
            "07_final_no_attention,"
            "08_final_no_ar_shortcut,"
            "09_final_vanilla_lstm,"
            "10_final_mlp_window"
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    base_cfg = load_config(str(cfg_path))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else ROOT / "outputs" / "final_ablation" / stamp
    output_root.mkdir(parents=True, exist_ok=True)

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    for variant in variants:
        run_dir = output_root / variant
        if args.skip_existing and (run_dir / "metrics_all_models.csv").exists():
            print(f"SKIP existing variant: {variant}", flush=True)
            continue

        run_cfg = apply_ablation(base_cfg, variant)
        run_cfg["train"]["epochs"] = int(args.epochs)
        run_cfg["project"]["output_dir"] = str(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config_snapshot.json").write_text(json.dumps(run_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        set_seed(int(run_cfg["project"].get("seed", 42)))
        logger = setup_logger(str(run_dir), run_cfg["project"].get("log_level", "INFO"))
        logger.info("Final ablation run variant=%s epochs=%d", variant, args.epochs)
        run_experiment(run_cfg, logger)

    metrics_df = collect_metrics(output_root)
    if not metrics_df.empty:
        metrics_df.to_csv(output_root / "final_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    write_summary(output_root, metrics_df)
    print(json.dumps({"output_root": str(output_root.resolve()), "rows": int(len(metrics_df))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
