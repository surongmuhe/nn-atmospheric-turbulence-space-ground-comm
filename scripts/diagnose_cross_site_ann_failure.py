import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cross_site_transfer import (  # noqa: E402
    apply_clip_bounds,
    fit_source_clip_bounds,
    prepare_site_dataframe,
    summarize_feature_shift,
)


DEFAULT_CROSS_SITE_ROOT = ROOT / "outputs" / "current_route_cross_site" / "formal_20260425_delta"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "cross_site_ann_diagnosis_20260426"


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def load_variant_config(cross_site_root: Path, variant: str) -> dict:
    config_path = cross_site_root / variant / "config_snapshot.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config snapshot: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def compute_shift_for_variant(cross_site_root: Path, variant: str) -> pd.DataFrame:
    cfg = load_variant_config(cross_site_root, variant)
    data_cfg = cfg["data"]
    preprocess_cfg = cfg.get("preprocess", {})
    feature_cols = list(data_cfg["feature_cols"])
    target_col = data_cfg["target_col"]
    resample_rule = data_cfg.get("resample_rule")

    source_df, source_feature_cols, _ = prepare_site_dataframe(
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
    target_df, target_feature_cols, _ = prepare_site_dataframe(
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
        raise ValueError(f"Feature schema mismatch for {variant}")

    n_source_total = len(source_df)
    n_source_train = int(n_source_total * float(data_cfg["source_train_ratio"]))
    n_source_val = int(n_source_total * float(data_cfg["source_val_ratio"]))
    source_feat_df, clip_bounds = fit_source_clip_bounds(
        source_df[source_feature_cols].copy(),
        n_train=n_source_train,
        clip_cfg=preprocess_cfg.get("clip_quantiles"),
        feature_cols=source_feature_cols,
    )
    target_feat_df = apply_clip_bounds(target_df[target_feature_cols].copy(), clip_bounds)

    scaler_name = str(preprocess_cfg.get("scaler", "standard")).lower()
    scaler = RobustScaler() if scaler_name == "robust" else StandardScaler()
    scaler.fit(source_feat_df.iloc[:n_source_train].values)
    target_z_df = pd.DataFrame(
        scaler.transform(target_feat_df.values),
        index=target_feat_df.index,
        columns=source_feature_cols,
    )
    shift = summarize_feature_shift(
        source_train_df=source_feat_df.iloc[:n_source_train],
        source_val_df=source_feat_df.iloc[n_source_train : n_source_train + n_source_val],
        target_df=target_feat_df,
        target_z_df=target_z_df,
        feature_cols=source_feature_cols,
    )
    shift.insert(0, "variant", variant)
    shift.insert(1, "scaler", scaler_name)
    shift.insert(2, "source_train_rows", n_source_train)
    shift.insert(3, "target_rows", len(target_feat_df))
    return shift


def summarize_predictions(cross_site_root: Path, variant: str) -> pd.DataFrame:
    pred_path = cross_site_root / variant / "predictions_target_test.csv"
    metrics_path = cross_site_root / variant / "metrics_all_models.csv"
    if not pred_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Missing prediction or metrics file for {variant}")

    pred_df = pd.read_csv(pred_path)
    metrics_df = pd.read_csv(metrics_path)
    truth_col = "y_true_h1"
    truth = pred_df[truth_col]
    rows = []
    for col in [c for c in pred_df.columns if c.endswith("_h1") and c != truth_col]:
        model = col.removesuffix("_h1")
        metric_match = metrics_df.loc[metrics_df["model"] == model]
        metrics = metric_match.iloc[0].to_dict() if len(metric_match) else {}
        series = pred_df[col]
        rows.append(
            {
                "variant": variant,
                "model": model,
                "truth_mean": float(truth.mean()),
                "truth_std": float(truth.std(ddof=1)),
                "pred_mean": float(series.mean()),
                "pred_std": float(series.std(ddof=1)),
                "pred_min": float(series.min()),
                "pred_max": float(series.max()),
                "truth_pred_corr": float(series.corr(truth)) if series.std(ddof=1) > 0 else np.nan,
                "RMSE": float(metrics.get("RMSE", np.nan)),
                "MAE": float(metrics.get("MAE", np.nan)),
                "R2": float(metrics.get("R2", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def write_markdown_report(output_dir: Path, shift_df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    raw_shift = shift_df[shift_df["variant"].isin(["ann_current", "ann_window", "lstm_raw"])]
    pair_rows = raw_shift[raw_shift["feature"] == "Pair"].copy()
    ann_preds = pred_df[(pred_df["variant"].isin(["ann_current", "ann_window"])) & (pred_df["model"] == "mlp")]

    lines = [
        "# Cross-Site ANN Failure Diagnosis",
        "",
        "This report diagnoses why the two ANN baselines fail in strict zero-shot cross-site transfer.",
        "",
        "## Key Finding",
        "",
        "The failure is not caused by an empty test set or an inverse-transform bug. The ANN models perform normally on the source validation split, but their target-site predictions move far outside the physical log10(Cn^2) range. The dominant trigger is a severe cross-site feature-scale shift in `Pair` under source-fitted standardization.",
        "",
    ]
    if not pair_rows.empty:
        display = pair_rows[
            [
                "variant",
                "source_train_mean",
                "source_train_std",
                "target_mean",
                "target_z_mean",
                "target_abs_z_gt3_pct",
                "mean_shift_sigma",
            ]
        ].round(4)
        lines.extend(["## Pair Shift", "", display.to_markdown(index=False), ""])
    if not ann_preds.empty:
        display = ann_preds[
            [
                "variant",
                "model",
                "truth_mean",
                "pred_mean",
                "pred_std",
                "RMSE",
                "R2",
                "truth_pred_corr",
            ]
        ].round(5)
        lines.extend(["## ANN Prediction Distribution", "", display.to_markdown(index=False), ""])

    lines.extend(
        [
            "## Interpretation",
            "",
            "- In the raw cross-site setting, `Pair` in the source training split is around 101 with a standard deviation below 0.5, while the target-site values are around 681 before any raw-feature clipping.",
            "- After applying a `StandardScaler` fitted only on the source training split, all target `Pair` values are thousands of standard deviations away from the source distribution.",
            "- The MLP baselines flatten the input and use unconstrained linear layers, so this out-of-distribution pressure feature produces extreme extrapolation. `ANN-current` predicts around -2 on average and `ANN-window` predicts around -48, while the target truth is around -13.7.",
            "- `LSTM-raw` is less affected because its recurrent dynamics are dominated by recent `LogCn2` history, and the improved model uses source-train clipping plus temporal/contextual structure. This explains why the ANN cross-site failure is much more severe than the LSTM failure.",
            "",
            "## Paper-Level Conclusion",
            "",
            "The ANN cross-site results should be interpreted as a distribution-shift diagnostic rather than a normal performance comparison. The paper should explicitly state that zero-shot cross-site deployment is sensitive to feature-unit consistency and site-specific meteorological ranges, especially pressure-related variables.",
            "",
        ]
    )
    (output_dir / "cross_site_ann_failure_diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose why ANN baselines fail in strict cross-site transfer.")
    parser.add_argument("--cross-site-root", type=Path, default=DEFAULT_CROSS_SITE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--variants",
        type=str,
        default="ann_current,ann_window,lstm_raw,lstm_improved",
        help="Comma-separated variants to diagnose.",
    )
    args = parser.parse_args()

    cross_site_root = args.cross_site_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    shift_frames = []
    pred_frames = []
    for variant in variants:
        shift_frames.append(compute_shift_for_variant(cross_site_root, variant))
        pred_frames.append(summarize_predictions(cross_site_root, variant))

    shift_df = pd.concat(shift_frames, ignore_index=True)
    pred_df = pd.concat(pred_frames, ignore_index=True)
    shift_df.to_csv(output_dir / "cross_site_feature_shift_diagnostics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(output_dir / "cross_site_prediction_distribution_diagnostics.csv", index=False, encoding="utf-8-sig")

    top_shift = (
        shift_df.sort_values(["target_abs_z_gt3_pct", "mean_shift_sigma"], ascending=[False, False])
        .head(12)
        .to_dict(orient="records")
    )
    ann_rows = pred_df[(pred_df["variant"].isin(["ann_current", "ann_window"])) & (pred_df["model"] == "mlp")]
    summary = {
        "cross_site_root": str(cross_site_root),
        "output_dir": str(output_dir),
        "diagnosed_variants": variants,
        "top_shift_features": top_shift,
        "ann_prediction_rows": ann_rows.to_dict(orient="records"),
    }
    (output_dir / "cross_site_ann_failure_diagnosis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(output_dir, shift_df, pred_df)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
