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


DEFAULT_EXTERNAL_ROOT = ROOT / "outputs" / "external_site_four_model_route_20260426_validated"
DEFAULT_FEWSHOT_ROOT = ROOT / "outputs" / "cross_site_fewshot_adaptation" / "formal_20260426"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "reviewer_requested_extras_20260427"
VARIANTS = ["ann_current", "ann_window", "lstm_raw", "lstm_improved"]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err**2) / denom) if denom > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "MSE": mse, "R2": r2}


def _ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_external_site(
    external_root: Path,
    output_root: Path,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = external_root / "aligned_four_model_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing aligned external-site predictions: {pred_path}")

    aligned = pd.read_csv(pred_path)
    if aligned.empty:
        raise RuntimeError("Aligned external-site predictions are empty.")

    n = len(aligned)
    rng = np.random.default_rng(seed)

    model_rows = []
    paired_rows = []
    all_indices = [rng.integers(0, n, size=n) for _ in range(n_boot)]

    for variant in VARIANTS:
        y = aligned[f"{variant}_true"].to_numpy(float)
        p = aligned[f"{variant}_pred"].to_numpy(float)
        point = regression_metrics(y, p)
        boot = {k: [] for k in ["RMSE", "MAE", "MSE", "R2"]}
        for idx in all_indices:
            vals = regression_metrics(y[idx], p[idx])
            for key in boot:
                boot[key].append(vals[key])
        row = {"variant": variant, "n_common": n, **point}
        for key, values in boot.items():
            lo, hi = _ci(np.asarray(values))
            row[f"{key}_ci_low"] = lo
            row[f"{key}_ci_high"] = hi
        model_rows.append(row)

    y_imp = aligned["lstm_improved_true"].to_numpy(float)
    p_imp = aligned["lstm_improved_pred"].to_numpy(float)
    imp_point = regression_metrics(y_imp, p_imp)
    for variant in [v for v in VARIANTS if v != "lstm_improved"]:
        y_base = aligned[f"{variant}_true"].to_numpy(float)
        p_base = aligned[f"{variant}_pred"].to_numpy(float)
        base_point = regression_metrics(y_base, p_base)

        deltas = {
            "RMSE_reduction": [],
            "MAE_reduction": [],
            "MSE_reduction": [],
            "R2_gain": [],
        }
        for idx in all_indices:
            imp = regression_metrics(y_imp[idx], p_imp[idx])
            base = regression_metrics(y_base[idx], p_base[idx])
            deltas["RMSE_reduction"].append(base["RMSE"] - imp["RMSE"])
            deltas["MAE_reduction"].append(base["MAE"] - imp["MAE"])
            deltas["MSE_reduction"].append(base["MSE"] - imp["MSE"])
            deltas["R2_gain"].append(imp["R2"] - base["R2"])

        row = {
            "comparison": f"Improved LSTM vs {variant}",
            "n_common": n,
            "RMSE_reduction": base_point["RMSE"] - imp_point["RMSE"],
            "MAE_reduction": base_point["MAE"] - imp_point["MAE"],
            "MSE_reduction": base_point["MSE"] - imp_point["MSE"],
            "R2_gain": imp_point["R2"] - base_point["R2"],
        }
        for key, values in deltas.items():
            lo, hi = _ci(np.asarray(values))
            row[f"{key}_ci_low"] = lo
            row[f"{key}_ci_high"] = hi
        paired_rows.append(row)

    model_df = pd.DataFrame(model_rows)
    paired_df = pd.DataFrame(paired_rows)
    model_df.to_csv(output_root / "external_site_model_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    paired_df.to_csv(output_root / "external_site_improved_paired_bootstrap.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# External-Site Bootstrap Confidence Intervals",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Common timestamps: {n}",
        f"Bootstrap resamples: {n_boot}",
        "",
        "## Model-level 95% confidence intervals",
        "",
        model_df.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Paired improvements of Improved LSTM",
        "",
        paired_df.to_markdown(index=False, floatfmt=".5f"),
        "",
    ]
    (output_root / "external_site_bootstrap_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return model_df, paired_df


def run_target_scratch_controls(
    fewshot_root: Path,
    output_root: Path,
    ratios: list[float],
    epochs: int,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scratch_dir = output_root / "target_only_scratch_lstm_improved"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(scratch_dir), "INFO")

    cfg = make_config("lstm_improved", scratch_dir, epochs=epochs, batch_size=batch_size)
    cfg = copy.deepcopy(cfg)
    cfg["project"]["seed"] = seed
    cfg["train"]["epochs"] = int(epochs)
    cfg["train"]["batch_size"] = int(batch_size)
    cfg["train"]["learning_rate"] = 0.001
    cfg["train"]["early_stop_patience"] = min(10, int(cfg["train"].get("early_stop_patience", 10)))
    (scratch_dir / "config_snapshot.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    common = prepare_common_data(cfg)
    model_name = str(cfg["experiment"]["primary_model"])
    rows = []
    for ratio in ratios:
        if ratio <= 0:
            continue
        split = make_target_split(common, ratio)
        if len(split["x_train"]) == 0 or len(split["x_val"]) == 0 or len(split["x_test"]) == 0:
            logger.warning(
                "Skip target-only scratch ratio=%.3f train=%d val=%d test=%d",
                ratio,
                len(split["x_train"]),
                len(split["x_val"]),
                len(split["x_test"]),
            )
            continue

        set_seed(seed + int(round(ratio * 1000)))
        scratch_cfg = copy.deepcopy(cfg)
        scratch_cfg["train"]["batch_size"] = min(int(batch_size), max(16, len(split["x_train"])))
        train_loader, val_loader = build_loaders(
            split["x_train"],
            split["y_train"],
            split["x_val"],
            split["y_val"],
            batch_size=scratch_cfg["train"]["batch_size"],
            num_workers=scratch_cfg["train"]["num_workers"],
            shuffle_train=True,
        )
        model = build_model(
            model_name,
            input_size=common["x_source_train"].shape[2],
            cfg=common["model_cfg"],
            dataset_meta={"pred_len": common["pred_len"], "feature_cols": common["feature_cols"]},
        )
        model = train_single_model(model_name, model, train_loader, val_loader, scratch_cfg, logger, device)
        _, _, metrics = evaluate_model(model, split, common, device)
        rows.append(
            row_from_metrics(
                "lstm_improved",
                ratio,
                "target_only_scratch",
                split,
                metrics,
                extra={"scratch_epochs": int(epochs), "scratch_seed": int(seed)},
            )
        )
        torch.save(
            {
                "variant": "lstm_improved",
                "adaptation_ratio": ratio,
                "method": "target_only_scratch",
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "config": scratch_cfg,
            },
            scratch_dir / f"lstm_improved_target_only_scratch_{int(round(ratio * 100)):02d}pct.pt",
        )

    scratch_df = pd.DataFrame(rows)
    scratch_df.to_csv(output_root / "fewshot_target_only_scratch_metrics.csv", index=False, encoding="utf-8-sig")

    base_path = fewshot_root / "fewshot_adaptation_metrics.csv"
    if base_path.exists():
        base = pd.read_csv(base_path)
        keep = base[
            (base["variant"] == "lstm_improved")
            & (base["method"].isin(["source_only", "persistence", "fewshot_finetuned"]))
            & (base["adaptation_ratio"].isin(ratios))
        ].copy()
        merged = pd.concat([keep, scratch_df], ignore_index=True, sort=False)
    else:
        merged = scratch_df
    merged = merged.sort_values(["adaptation_ratio", "variant", "method"]).reset_index(drop=True)
    merged.to_csv(output_root / "fewshot_improved_lstm_augmented_controls.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Few-Shot Target-Only Scratch Control",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    if not merged.empty:
        cols = [
            "adaptation_ratio",
            "method",
            "target_train_windows",
            "target_val_windows",
            "target_test_windows",
            "test_start",
            "RMSE",
            "MAE",
            "R2",
        ]
        cols = [c for c in cols if c in merged.columns]
        lines.append(merged[cols].to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No rows were produced.")
    (output_root / "fewshot_target_only_scratch_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return scratch_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reviewer-requested supplementary experiments.")
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--fewshot-root", type=Path, default=DEFAULT_FEWSHOT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ratios", type=str, default="0.05,0.10,0.20")
    parser.add_argument("--scratch-epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ratios = [float(item.strip()) for item in args.ratios.split(",") if item.strip()]

    model_ci, paired_ci = bootstrap_external_site(args.external_root.resolve(), output_root, args.bootstrap, args.seed)
    scratch_df = run_target_scratch_controls(
        args.fewshot_root.resolve(),
        output_root,
        ratios=ratios,
        epochs=args.scratch_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    summary = {
        "output_root": str(output_root),
        "external_model_ci_rows": int(len(model_ci)),
        "external_paired_ci_rows": int(len(paired_ci)),
        "scratch_rows": int(len(scratch_df)),
        "ratios": ratios,
        "bootstrap": int(args.bootstrap),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output_root / "reviewer_requested_extras_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
