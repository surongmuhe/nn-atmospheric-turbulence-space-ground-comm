from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
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


def parse_prediction_files(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*/*_predictions.csv")):
        if "all_predictions" in path.name:
            continue
        df = pd.read_csv(path, parse_dates=["datetime"])
        required = {"datetime", "variant", "method", "y_true", "y_pred"}
        missing = required.difference(df.columns)
        if missing:
            raise KeyError(f"{path} missing columns: {sorted(missing)}")
        variant = str(df["variant"].iloc[0])
        method = str(df["method"].iloc[0])
        df["model_label"] = MODEL_LABELS.get(variant, variant)
        if "adaptation_ratio" not in df.columns:
            raise KeyError(f"{path} missing adaptation_ratio")
        df["method_key"] = f"{variant}|{method}"
        if method == "persistence":
            df["method_key"] = "persistence|persistence"
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No prediction files found below {root}")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["adaptation_ratio", "method_key", "datetime"]).reset_index(drop=True)


def aligned_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ratio, ratio_df in pred_df.groupby("adaptation_ratio"):
        common_dates: set[pd.Timestamp] | None = None
        for _, group in ratio_df.groupby("method_key"):
            dates = set(pd.to_datetime(group["datetime"]))
            common_dates = dates if common_dates is None else common_dates.intersection(dates)
        if not common_dates:
            continue
        common_dates = set(sorted(common_dates))
        for method_key, group in ratio_df.groupby("method_key"):
            current = group[pd.to_datetime(group["datetime"]).isin(common_dates)].copy()
            current = current.sort_values("datetime")
            metrics = regression_metrics(current["y_true"].to_numpy(), current["y_pred"].to_numpy())
            rows.append(
                {
                    "adaptation_ratio": float(ratio),
                    "variant": str(current["variant"].iloc[0]),
                    "model_label": str(current["model_label"].iloc[0]),
                    "method": str(current["method"].iloc[0]),
                    "method_key": method_key,
                    "common_test_timestamps": int(len(current)),
                    "test_start": str(current["datetime"].iloc[0]),
                    "test_end": str(current["datetime"].iloc[-1]),
                    **metrics,
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["adaptation_ratio", "method", "RMSE", "variant"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate all few-shot strong-transfer prediction files on common timestamps.")
    parser.add_argument("--root", default="outputs/fewshot_strong_transfer_controls_20260427")
    args = parser.parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    pred_df = parse_prediction_files(root)
    pred_df.to_csv(root / "fewshot_strong_transfer_all_predictions_all_variants.csv", index=False, encoding="utf-8-sig")
    metrics = aligned_metrics(pred_df)
    metrics.to_csv(root / "fewshot_strong_transfer_aligned_metrics_all_variants.csv", index=False, encoding="utf-8-sig")
    ft = metrics[metrics["method"] == "fewshot_finetuned"].copy()
    if not ft.empty:
        ft.sort_values(["adaptation_ratio", "R2"], ascending=[True, False]).to_csv(
            root / "fewshot_strong_transfer_finetuned_ranking_all_variants.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
