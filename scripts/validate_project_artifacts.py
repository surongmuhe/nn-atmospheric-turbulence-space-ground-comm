from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = {
    "main_site_single_run": ROOT / "outputs/thesis_lstm_final_best_seq72/formal_20260424/metrics_all_models.csv",
    "main_site_cn2": ROOT / "outputs/thesis_lstm_final_best_seq72/formal_20260424/metrics_cn2_logcn2_all_models.csv",
    "strong_baselines": ROOT / "outputs/strong_baselines_seq72_20260425/strong_baselines_seed_summary.csv",
    "forecast_significance": ROOT / "outputs/review_response_improvements_20260426/forecast_pairwise_bootstrap.csv",
    "external_site_four_model": (
        ROOT / "outputs/external_site_four_model_route_20260426_validated/aligned_four_model_metrics.csv"
    ),
    "external_site_lstm_raw_sweep": (
        ROOT / "outputs/external_site_lstm_raw_sweep_20260426/lstm_raw_sweep_metrics.csv"
    ),
    "cross_site_final_model": (
        ROOT / "outputs/current_route_cross_site/formal_20260425_delta/lstm_improved/metrics_all_models.csv"
    ),
    "fewshot_cross_site": ROOT / "outputs/cross_site_fewshot_adaptation/formal_20260426/fewshot_adaptation_metrics.csv",
    "link_policy": ROOT / "outputs/review_response_improvements_20260426/default_link_policy_metrics.csv",
    "link_sensitivity": ROOT / "outputs/review_response_improvements_20260426/link_sensitivity_rank_stability.csv",
    "cross_site_link": ROOT / "outputs/review_response_improvements_20260426/cross_site_link_policy_metrics.csv",
    "cross_site_ann_diagnosis": ROOT / "outputs/cross_site_ann_diagnosis_20260426/cross_site_ann_failure_diagnosis.md",
    "cross_site_feature_validation": (
        ROOT / "outputs/cross_site_feature_validation_20260426/cross_site_feature_validation_primary_metrics.csv"
    ),
    "latest_thesis_md": ROOT / "docs/final_thesis_rewritten_20260425_v28_toc_refs_template.md",
    "latest_thesis_docx": ROOT / "docs/final_thesis_rewritten_20260425_v28_toc_refs_template.docx",
}


def require_artifacts() -> None:
    missing = [f"{name}: {path}" for name, path in REQUIRED_FILES.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))


def read_seed_summary() -> pd.DataFrame:
    rows = []
    model_dirs = {
        "Improved LSTM": "lstm_improved",
        "LSTM-raw": "lstm_raw",
        "TCN": "tcn",
        "TFT": "tft",
        "PatchTST": "patchtst",
    }
    base = ROOT / "outputs/strong_baselines_seq72_20260425"
    for label, dirname in model_dirs.items():
        frame = pd.read_csv(base / dirname / "seed_summary_by_model.csv")
        row = frame.iloc[0].to_dict()
        row["model_label"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    require_artifacts()

    main_metrics = pd.read_csv(REQUIRED_FILES["main_site_single_run"])
    improved = main_metrics.loc[main_metrics["model"] == "lstm"].iloc[0]

    seed_summary = read_seed_summary()
    best_seed = seed_summary.sort_values("RMSE_mean").iloc[0]

    forecast_sig = pd.read_csv(REQUIRED_FILES["forecast_significance"])
    external_four = pd.read_csv(REQUIRED_FILES["external_site_four_model"])
    cross_site_final = pd.read_csv(REQUIRED_FILES["cross_site_final_model"])
    fewshot = pd.read_csv(REQUIRED_FILES["fewshot_cross_site"])
    link_policy = pd.read_csv(REQUIRED_FILES["link_policy"])
    link_sensitivity = pd.read_csv(REQUIRED_FILES["link_sensitivity"])

    cross_improved = cross_site_final.loc[cross_site_final["model"] == "lstm"].iloc[0]
    fewshot_improved = fewshot.loc[
        (fewshot["variant"] == "lstm_improved")
        & (
            ((fewshot["adaptation_ratio"] == 0.0) & (fewshot["method"] == "source_only"))
            | ((fewshot["adaptation_ratio"] > 0.0) & (fewshot["method"] == "fewshot_finetuned"))
        )
    ].sort_values("adaptation_ratio")
    improved_link = link_policy.loc[link_policy["policy"] == "Improved LSTM Policy"].iloc[0]
    best_sensitivity = link_sensitivity.sort_values("mean_rank").iloc[0]
    external_sorted = external_four.sort_values("R2").reset_index(drop=True)

    summary = {
        "main_site_single_run": {
            "model": "Improved LSTM",
            "RMSE": round(float(improved["RMSE"]), 5),
            "MAE": round(float(improved["MAE"]), 5),
            "R2": round(float(improved["R2"]), 5),
        },
        "best_5seed_model": {
            "model": best_seed["model_label"],
            "RMSE_mean": round(float(best_seed["RMSE_mean"]), 5),
            "R2_mean": round(float(best_seed["R2_mean"]), 5),
        },
        "forecast_significance_comparisons": int(len(forecast_sig)),
        "external_site_four_model_order": {
            "order_low_to_high": external_sorted["variant"].tolist(),
            "best_variant": external_four.sort_values("R2", ascending=False).iloc[0]["variant"],
            "best_R2": round(float(external_four["R2"].max()), 5),
        },
        "cross_site_final_model": {
            "model": "Improved LSTM",
            "RMSE": round(float(cross_improved["RMSE"]), 5),
            "MAE": round(float(cross_improved["MAE"]), 5),
            "R2": round(float(cross_improved["R2"]), 5),
        },
        "fewshot_improved_lstm": fewshot_improved[
            ["adaptation_ratio", "variant", "method", "RMSE", "R2"]
        ].to_dict(orient="records"),
        "link_policy": {
            "policy": "Improved LSTM Policy",
            "avg_mission_utility": round(float(improved_link["avg_mission_utility"]), 4),
            "outage_rate": round(float(improved_link["outage_rate"]), 4),
            "availability": round(float(improved_link["availability"]), 4),
        },
        "link_sensitivity_best": {
            "policy": best_sensitivity["policy"],
            "mean_rank": round(float(best_sensitivity["mean_rank"]), 2),
            "best_rank_count": int(best_sensitivity["best_rank_count"]),
        },
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Artifact validation passed.")


if __name__ == "__main__":
    main()
