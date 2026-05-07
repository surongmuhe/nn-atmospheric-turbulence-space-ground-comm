import argparse
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

from scripts.run_current_route_cross_site import collect_results, make_config  # noqa: E402


FEATURE_PROFILES = {
    "portable_no_pair": {
        "description": "Shared portable features without the cross-site pressure variable Pair.",
        "base_features": ["LogCn2", "Tair", "UmSonic", "is_daylight"],
        "improved_extra_features": ["hour_sin", "hour_cos"],
        "sensor_cols": ["LogCn2", "Tair", "UmSonic"],
    },
    "history_daylight": {
        "description": "Target-history and day/night proxy only; used to isolate exogenous feature-shift effects.",
        "base_features": ["LogCn2", "is_daylight"],
        "improved_extra_features": ["hour_sin", "hour_cos"],
        "sensor_cols": ["LogCn2"],
    },
}


def update_preprocess_columns(cfg: dict, sensor_cols: list[str]) -> None:
    preprocess = cfg["preprocess"]
    for key in ["missing_indicators", "missing_context_features", "long_gap_mask"]:
        if key in preprocess and isinstance(preprocess[key], dict):
            preprocess[key]["columns"] = list(sensor_cols)

    dynamic_cols = [col for col in ["LogCn2", "Tair", "UmSonic"] if col in sensor_cols]
    if "diff_features" in preprocess and isinstance(preprocess["diff_features"], dict):
        preprocess["diff_features"]["columns"] = dynamic_cols
    if "rolling_features" in preprocess and isinstance(preprocess["rolling_features"], dict):
        preprocess["rolling_features"]["columns"] = dynamic_cols


def make_profile_config(variant: str, profile: dict, output_dir: Path, epochs: int, batch_size: int) -> dict:
    cfg = make_config(variant, output_dir=output_dir, epochs=epochs, batch_size=batch_size)
    cfg = deepcopy(cfg)
    base_features = list(profile["base_features"])
    if variant == "lstm_improved":
        feature_cols = base_features + list(profile.get("improved_extra_features", []))
    else:
        feature_cols = base_features
    cfg["project"]["name"] = f"cross_site_feature_validation_{variant}"
    cfg["project"]["output_dir"] = str(output_dir)
    cfg["data"]["feature_cols"] = feature_cols
    update_preprocess_columns(cfg, list(profile["sensor_cols"]))
    return cfg


def collect_profile_results(profile_root: Path) -> pd.DataFrame:
    metrics = collect_results(profile_root)
    if metrics.empty:
        return metrics
    primary_rows = []
    for variant, model_name in {
        "ann_current": "mlp",
        "ann_window": "mlp",
        "lstm_raw": "lstm",
        "lstm_improved": "lstm",
    }.items():
        subset = metrics[(metrics["variant"] == variant) & (metrics["model"] == model_name)]
        if not subset.empty:
            primary_rows.append(subset.iloc[0].to_dict())
    return pd.DataFrame(primary_rows)


def write_summary(output_root: Path, all_metrics: pd.DataFrame, primary_metrics: pd.DataFrame) -> None:
    lines = [
        "# Cross-Site Feature Validation",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This experiment checks whether the four-model cross-site comparison is sensitive to the selected shared input features.",
        "",
        "## Primary Model Results",
        "",
    ]
    if primary_metrics.empty:
        lines.append("No primary model metrics were collected.")
    else:
        cols = [c for c in ["profile", "variant", "model", "RMSE", "MAE", "MAPE", "R2", "RMSE_h1"] if c in primary_metrics.columns]
        lines.append(primary_metrics[cols].to_markdown(index=False))
    lines.extend(["", "## All Metrics", ""])
    if all_metrics.empty:
        lines.append("No metrics were collected.")
    else:
        cols = [c for c in ["profile", "variant", "model", "RMSE", "MAE", "MAPE", "R2", "RMSE_h1"] if c in all_metrics.columns]
        lines.append(all_metrics[cols].to_markdown(index=False))
    (output_root / "cross_site_feature_validation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate four cross-site models under alternative shared feature profiles.")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--profiles", type=str, default="portable_no_pair")
    parser.add_argument("--variants", type=str, default="ann_current,ann_window,lstm_raw,lstm_improved")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else ROOT / "outputs" / "cross_site_feature_validation" / stamp
    output_root = output_root.resolve()
    config_root = output_root / "_configs"
    config_root.mkdir(parents=True, exist_ok=True)

    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown_profiles = [name for name in profiles if name not in FEATURE_PROFILES]
    if unknown_profiles:
        raise ValueError(f"Unknown feature profiles: {unknown_profiles}. Available: {sorted(FEATURE_PROFILES)}")

    all_frames = []
    primary_frames = []
    profile_manifest = {}
    for profile_name in profiles:
        profile = FEATURE_PROFILES[profile_name]
        profile_root = output_root / profile_name
        profile_config_root = config_root / profile_name
        profile_config_root.mkdir(parents=True, exist_ok=True)
        profile_manifest[profile_name] = profile

        for variant in variants:
            out_dir = profile_root / variant
            cfg = make_profile_config(variant, profile, output_dir=out_dir, epochs=args.epochs, batch_size=args.batch_size)
            cfg_path = profile_config_root / f"{variant}.yaml"
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

        profile_metrics = collect_results(profile_root)
        if not profile_metrics.empty:
            profile_metrics.insert(0, "profile", profile_name)
            all_frames.append(profile_metrics)
        profile_primary = collect_profile_results(profile_root)
        if not profile_primary.empty:
            profile_primary.insert(0, "profile", profile_name)
            primary_frames.append(profile_primary)

    all_metrics = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    primary_metrics = pd.concat(primary_frames, ignore_index=True) if primary_frames else pd.DataFrame()
    all_metrics.to_csv(output_root / "cross_site_feature_validation_all_metrics.csv", index=False, encoding="utf-8-sig")
    primary_metrics.to_csv(output_root / "cross_site_feature_validation_primary_metrics.csv", index=False, encoding="utf-8-sig")
    (output_root / "feature_profile_manifest.yaml").write_text(
        yaml.safe_dump(profile_manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    write_summary(output_root, all_metrics, primary_metrics)
    print(f"Wrote feature validation outputs to {output_root}", flush=True)


if __name__ == "__main__":
    main()
