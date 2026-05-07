import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_VARIANTS = ["ann_current", "ann_window", "lstm_raw", "lstm_improved"]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    mask = np.abs(y_true) > 1.0e-12
    mape = float(np.mean(np.abs(err[mask] / y_true[mask])) * 100.0) if np.any(mask) else float("nan")
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err**2) / denom) if denom > 0 else float("nan")
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def load_variant_predictions(root: Path, variant: str) -> pd.DataFrame:
    path = root / variant / "predictions_test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")

    df = pd.read_csv(path)
    if "Date" not in df.columns or "y_true_h1" not in df.columns:
        raise ValueError(f"{path} must contain Date and y_true_h1 columns.")

    pred_cols = [c for c in df.columns if c.endswith("_pred_h1")]
    if not pred_cols:
        pred_cols = [c for c in df.columns if c.endswith("_pred")]
    if not pred_cols:
        raise ValueError(f"Cannot find prediction column in {path}.")

    out = df[["Date", "y_true_h1", pred_cols[0]]].copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.rename(
        columns={
            "y_true_h1": f"{variant}_true",
            pred_cols[0]: f"{variant}_pred",
        }
    )
    return out.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)


def align_predictions(root: Path, variants: list[str]) -> pd.DataFrame:
    aligned = None
    for variant in variants:
        df = load_variant_predictions(root, variant)
        aligned = df if aligned is None else aligned.merge(df, on="Date", how="inner")
    if aligned is None or aligned.empty:
        raise RuntimeError("No common timestamps were found across variants.")
    return aligned.sort_values("Date").reset_index(drop=True)


def summarize(aligned: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    rows = []
    for variant in variants:
        metrics = regression_metrics(aligned[f"{variant}_true"], aligned[f"{variant}_pred"])
        metrics["variant"] = variant
        metrics["n_common"] = int(len(aligned))
        rows.append(metrics)
    cols = ["variant", "n_common", "RMSE", "MAE", "MSE", "MAPE", "R2"]
    return pd.DataFrame(rows)[cols].sort_values("R2").reset_index(drop=True)


def write_markdown(root: Path, aligned: pd.DataFrame, metrics: pd.DataFrame, variants: list[str]) -> None:
    order = metrics.sort_values("R2")["variant"].tolist()
    desired = order == variants
    lines = [
        "# Aligned External-Site Four-Model Metrics",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Experiment root: `{root}`",
        f"Common timestamps: {len(aligned)}",
        "",
        "Metrics below are recomputed only on timestamps shared by all variants.",
        "",
        metrics.to_markdown(index=False, floatfmt=".5f"),
        "",
        f"R2 order from low to high: `{ ' < '.join(order) }`",
        f"Matches expected route order: `{desired}`",
        "",
    ]
    (root / "aligned_four_model_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align external-site four-model predictions on common timestamps.")
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS), type=str)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    aligned = align_predictions(root, variants)
    metrics = summarize(aligned, variants)

    aligned.to_csv(root / "aligned_four_model_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(root / "aligned_four_model_metrics.csv", index=False, encoding="utf-8-sig")
    write_markdown(root, aligned, metrics, variants)
    payload = {
        "root": str(root),
        "variants": variants,
        "n_common": int(len(aligned)),
        "r2_order_low_to_high": metrics.sort_values("R2")["variant"].tolist(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (root / "aligned_four_model_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"Wrote aligned metrics to {root}")


if __name__ == "__main__":
    main()
