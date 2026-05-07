from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.link_budget_simulation import (  # noqa: E402
    DEFAULT_MCS_MODES,
    LinkBudgetConfig,
    evaluate_policy,
    mode_table,
)


MAIN_SITE_MODEL_INPUTS = {
    "Improved LSTM Policy": (
        "outputs/strong_baselines_seq72_20260425/lstm_improved/seed_42/predictions_test.csv",
        "lstm_pred_h1",
    ),
    "LSTM-raw Policy": (
        "outputs/strong_baselines_seq72_20260425/lstm_raw/seed_42/predictions_test.csv",
        "lstm_pred_h1",
    ),
    "TCN Policy": (
        "outputs/strong_baselines_seq72_20260425/tcn/seed_42/predictions_test.csv",
        "tcn_pred_h1",
    ),
    "TFT Policy": (
        "outputs/strong_baselines_seq72_20260425/tft/seed_42/predictions_test.csv",
        "tft_pred_h1",
    ),
    "PatchTST Policy": (
        "outputs/strong_baselines_seq72_20260425/patchtst/seed_42/predictions_test.csv",
        "patchtst_pred_h1",
    ),
}

SEED_MODEL_INPUTS = {
    "Improved LSTM": ("lstm_improved", "lstm_pred_h1"),
    "LSTM-raw": ("lstm_raw", "lstm_pred_h1"),
    "TCN": ("tcn", "tcn_pred_h1"),
    "TFT": ("tft", "tft_pred_h1"),
    "PatchTST": ("patchtst", "patchtst_pred_h1"),
}

SEEDS = [42, 52, 62, 72, 82]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def set_plot_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "axes.edgecolor": "#243447",
            "axes.labelcolor": "#243447",
            "axes.titleweight": "bold",
            "grid.color": "#D7DEE8",
            "grid.linewidth": 0.6,
            "figure.dpi": 160,
            "savefig.dpi": 320,
        },
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def read_prediction_file(path: Path, pred_col: str, policy_name: str, date_col: str = "Date") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    required = {date_col, "y_true_h1", pred_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")
    out = df[[date_col, "y_true_h1", pred_col]].copy()
    out = out.rename(
        columns={
            date_col: "Date",
            "y_true_h1": f"y_true_h1_{policy_name}",
            pred_col: policy_name,
        }
    )
    return out


def load_aligned_main_site_predictions(persistence_lag_steps: int = 6) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    true_cols: list[str] = []
    for policy_name, (path_str, pred_col) in MAIN_SITE_MODEL_INPUTS.items():
        current = read_prediction_file(resolve(path_str), pred_col, policy_name)
        true_cols.append(f"y_true_h1_{policy_name}")
        merged = current if merged is None else merged.merge(current, on="Date", how="inner")
    if merged is None or merged.empty:
        raise ValueError("No main-site prediction data was loaded.")

    merged = merged.sort_values("Date").reset_index(drop=True)
    true_stack = merged[true_cols].to_numpy(dtype=np.float64)
    max_true_delta = float(np.nanmax(np.abs(true_stack - true_stack[:, [0]])))
    if max_true_delta > 1e-6:
        raise ValueError(f"Aligned main-site files disagree on y_true_h1; max delta={max_true_delta:.3e}.")
    merged["y_true_h1"] = merged[true_cols[0]]
    merged["Persistence Policy"] = merged["y_true_h1"].shift(persistence_lag_steps)
    merged = merged.drop(columns=true_cols).dropna().reset_index(drop=True)
    return merged


def add_relative_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    fixed_high = float(out.loc[out["policy"] == "Fixed High-Rate", "avg_mission_utility"].iloc[0])
    oracle = float(out.loc[out["policy"] == "Oracle Policy", "avg_mission_utility"].iloc[0])
    out["utility_gain_vs_fixed_high_pct"] = 100.0 * (out["avg_mission_utility"] - fixed_high) / abs(fixed_high)
    out["oracle_utility_gap_pct"] = 100.0 * (oracle - out["avg_mission_utility"]) / abs(oracle)
    return out.sort_values("avg_mission_utility", ascending=False).reset_index(drop=True)


def build_policy_outputs(frame: pd.DataFrame, cfg: LinkBudgetConfig, forecast_policies: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_frames = []
    summaries = []
    fixed_policies = [
        ("Fixed Survival", 0),
        ("Fixed Conservative", 1),
        ("Fixed Balanced", 2),
        ("Fixed High-Rate", 3),
        ("Fixed Peak-Rate", 4),
    ]
    for policy_name, mode_index in fixed_policies:
        policy_frame, summary = evaluate_policy(
            frame,
            policy_name=policy_name,
            true_col="y_true_h1",
            prediction_col=None,
            modes=DEFAULT_MCS_MODES,
            cfg=cfg,
            fixed_mode_index=mode_index,
        )
        policy_frames.append(policy_frame)
        summaries.append(summary)

    for policy_name in forecast_policies:
        policy_frame, summary = evaluate_policy(
            frame,
            policy_name=policy_name,
            true_col="y_true_h1",
            prediction_col=policy_name,
            modes=DEFAULT_MCS_MODES,
            cfg=cfg,
        )
        policy_frames.append(policy_frame)
        summaries.append(summary)

    policy_frame, summary = evaluate_policy(
        frame,
        policy_name="Oracle Policy",
        true_col="y_true_h1",
        prediction_col=None,
        modes=DEFAULT_MCS_MODES,
        cfg=cfg,
    )
    policy_frames.append(policy_frame)
    summaries.append(summary)
    return pd.concat(policy_frames, ignore_index=True), pd.DataFrame(summaries)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 4000) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan, np.nan
    boot = np.empty(n_boot, dtype=np.float64)
    chunk = 200
    filled = 0
    while filled < n_boot:
        take = min(chunk, n_boot - filled)
        idx = rng.integers(0, len(values), size=(take, len(values)))
        boot[filled : filled + take] = values[idx].mean(axis=1)
        filled += take
    mean = float(values.mean())
    lo, hi = np.quantile(boot, [0.025, 0.975])
    prob_positive = float(np.mean(boot > 0.0))
    return mean, float(lo), float(hi), prob_positive


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum(np.square(y_true - y_pred)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def run_forecast_significance(output_dir: Path, rng: np.random.Generator) -> pd.DataFrame:
    base_dir = ROOT / "outputs" / "strong_baselines_seq72_20260425"
    records = []
    seed_records = []
    for baseline_name in ["LSTM-raw", "TCN", "TFT", "PatchTST"]:
        all_abs_diff = []
        all_sq_diff = []
        for seed in SEEDS:
            improved_dir, improved_col = SEED_MODEL_INPUTS["Improved LSTM"]
            baseline_dir, baseline_col = SEED_MODEL_INPUTS[baseline_name]
            improved = read_prediction_file(base_dir / improved_dir / f"seed_{seed}" / "predictions_test.csv", improved_col, "Improved LSTM")
            baseline = read_prediction_file(base_dir / baseline_dir / f"seed_{seed}" / "predictions_test.csv", baseline_col, baseline_name)
            merged = improved.merge(baseline, on="Date", how="inner")
            y_true_delta = np.nanmax(np.abs(merged["y_true_h1_Improved LSTM"] - merged[f"y_true_h1_{baseline_name}"]))
            if y_true_delta > 1e-6:
                raise ValueError(f"Seed {seed} true target mismatch for {baseline_name}: {y_true_delta:.3e}")
            y = merged["y_true_h1_Improved LSTM"].to_numpy(dtype=np.float64)
            pred_imp = merged["Improved LSTM"].to_numpy(dtype=np.float64)
            pred_base = merged[baseline_name].to_numpy(dtype=np.float64)
            abs_diff = np.abs(y - pred_base) - np.abs(y - pred_imp)
            sq_diff = np.square(y - pred_base) - np.square(y - pred_imp)
            all_abs_diff.append(abs_diff)
            all_sq_diff.append(sq_diff)
            seed_records.append(
                {
                    "baseline": baseline_name,
                    "seed": seed,
                    "rows": int(len(merged)),
                    "improved_rmse": rmse(y, pred_imp),
                    "baseline_rmse": rmse(y, pred_base),
                    "rmse_reduction": rmse(y, pred_base) - rmse(y, pred_imp),
                    "improved_mae": mae(y, pred_imp),
                    "baseline_mae": mae(y, pred_base),
                    "mae_reduction": mae(y, pred_base) - mae(y, pred_imp),
                    "improved_r2": r2_score(y, pred_imp),
                    "baseline_r2": r2_score(y, pred_base),
                    "r2_gain": r2_score(y, pred_imp) - r2_score(y, pred_base),
                }
            )

        abs_values = np.concatenate(all_abs_diff)
        sq_values = np.concatenate(all_sq_diff)
        abs_mean, abs_lo, abs_hi, abs_prob_pos = bootstrap_mean_ci(abs_values, rng)
        sq_mean, sq_lo, sq_hi, sq_prob_pos = bootstrap_mean_ci(sq_values, rng)
        seed_df = pd.DataFrame([r for r in seed_records if r["baseline"] == baseline_name])
        records.append(
            {
                "comparison": f"Improved LSTM vs {baseline_name}",
                "paired_rows": int(len(abs_values)),
                "mae_reduction_mean": abs_mean,
                "mae_reduction_ci95_low": abs_lo,
                "mae_reduction_ci95_high": abs_hi,
                "mae_reduction_bootstrap_prob_positive": abs_prob_pos,
                "mse_reduction_mean": sq_mean,
                "mse_reduction_ci95_low": sq_lo,
                "mse_reduction_ci95_high": sq_hi,
                "mse_reduction_bootstrap_prob_positive": sq_prob_pos,
                "seeds_with_rmse_reduction": int((seed_df["rmse_reduction"] > 0).sum()),
                "seed_count": int(len(seed_df)),
                "mean_rmse_reduction": float(seed_df["rmse_reduction"].mean()),
                "mean_r2_gain": float(seed_df["r2_gain"].mean()),
            }
        )

    seed_out = pd.DataFrame(seed_records)
    pairwise_out = pd.DataFrame(records)
    seed_out.to_csv(output_dir / "forecast_seed_level_differences.csv", index=False, encoding="utf-8-sig")
    pairwise_out.to_csv(output_dir / "forecast_pairwise_bootstrap.csv", index=False, encoding="utf-8-sig")
    return pairwise_out


def run_default_link_bootstrap(output_dir: Path, rng: np.random.Generator) -> pd.DataFrame:
    frame = load_aligned_main_site_predictions()
    cfg = LinkBudgetConfig()
    policy_timeseries, summary = build_policy_outputs(
        frame,
        cfg,
        forecast_policies=[
            "Persistence Policy",
            "LSTM-raw Policy",
            "TCN Policy",
            "TFT Policy",
            "PatchTST Policy",
            "Improved LSTM Policy",
        ],
    )
    summary = add_relative_columns(summary)
    policy_timeseries.to_csv(output_dir / "default_link_policy_timeseries.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "default_link_policy_metrics.csv", index=False, encoding="utf-8-sig")
    mode_table(DEFAULT_MCS_MODES).to_csv(output_dir / "mcs_mode_table.csv", index=False, encoding="utf-8-sig")

    records = []
    improved = policy_timeseries[policy_timeseries["policy"] == "Improved LSTM Policy"][["Date", "mission_utility", "expected_goodput", "outage_probability"]]
    for baseline in ["LSTM-raw Policy", "TCN Policy", "TFT Policy", "PatchTST Policy", "Persistence Policy"]:
        other = policy_timeseries[policy_timeseries["policy"] == baseline][["Date", "mission_utility", "expected_goodput", "outage_probability"]]
        merged = improved.merge(other, on="Date", suffixes=("_improved", "_baseline"))
        utility_diff = merged["mission_utility_improved"].to_numpy(dtype=np.float64) - merged["mission_utility_baseline"].to_numpy(dtype=np.float64)
        goodput_diff = merged["expected_goodput_improved"].to_numpy(dtype=np.float64) - merged["expected_goodput_baseline"].to_numpy(dtype=np.float64)
        outage_diff = merged["outage_probability_baseline"].to_numpy(dtype=np.float64) - merged["outage_probability_improved"].to_numpy(dtype=np.float64)
        utility_mean, utility_lo, utility_hi, utility_prob = bootstrap_mean_ci(utility_diff, rng)
        goodput_mean, goodput_lo, goodput_hi, goodput_prob = bootstrap_mean_ci(goodput_diff, rng)
        outage_mean, outage_lo, outage_hi, outage_prob = bootstrap_mean_ci(outage_diff, rng)
        records.append(
            {
                "comparison": f"Improved LSTM Policy vs {baseline}",
                "paired_rows": int(len(merged)),
                "mission_utility_gain_mean": utility_mean,
                "mission_utility_gain_ci95_low": utility_lo,
                "mission_utility_gain_ci95_high": utility_hi,
                "mission_utility_gain_bootstrap_prob_positive": utility_prob,
                "goodput_gain_mean": goodput_mean,
                "goodput_gain_ci95_low": goodput_lo,
                "goodput_gain_ci95_high": goodput_hi,
                "goodput_gain_bootstrap_prob_positive": goodput_prob,
                "outage_reduction_mean": outage_mean,
                "outage_reduction_ci95_low": outage_lo,
                "outage_reduction_ci95_high": outage_hi,
                "outage_reduction_bootstrap_prob_positive": outage_prob,
            }
        )
    out = pd.DataFrame(records)
    out.to_csv(output_dir / "link_policy_pairwise_bootstrap.csv", index=False, encoding="utf-8-sig")
    return out


def run_link_sensitivity(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_aligned_main_site_predictions()
    path_values = [300.0, 500.0, 800.0]
    snr_values = [18.0, 20.0, 22.0]
    alpha_values = [4.0, 8.0, 12.0]
    all_summaries = []
    for path_m in path_values:
        for snr_db in snr_values:
            for alpha in alpha_values:
                scenario = f"L{int(path_m)}_SNR{snr_db:.0f}_A{alpha:.0f}"
                cfg = LinkBudgetConfig(
                    effective_turbulence_path_m=path_m,
                    clear_sky_snr_db=snr_db,
                    outage_penalty=alpha,
                )
                _, summary = build_policy_outputs(
                    frame,
                    cfg,
                    forecast_policies=[
                        "Persistence Policy",
                        "LSTM-raw Policy",
                        "TCN Policy",
                        "TFT Policy",
                        "PatchTST Policy",
                        "Improved LSTM Policy",
                    ],
                )
                summary = add_relative_columns(summary)
                summary["scenario"] = scenario
                summary["effective_path_m"] = path_m
                summary["clear_sky_snr_db"] = snr_db
                summary["outage_penalty"] = alpha
                all_summaries.append(summary)

    all_summary = pd.concat(all_summaries, ignore_index=True)
    all_summary.to_csv(output_dir / "link_sensitivity_policy_metrics.csv", index=False, encoding="utf-8-sig")

    forecast = all_summary[all_summary["policy_kind"] == "forecast"].copy()
    forecast["rank_in_scenario"] = forecast.groupby("scenario")["avg_mission_utility"].rank(ascending=False, method="min")
    winners = forecast.loc[forecast.groupby("scenario")["avg_mission_utility"].idxmax()].copy()
    stability = (
        forecast.groupby("policy", as_index=False)
        .agg(
            mean_rank=("rank_in_scenario", "mean"),
            median_rank=("rank_in_scenario", "median"),
            best_rank_count=("rank_in_scenario", lambda s: int(np.sum(s == 1))),
            mean_utility=("avg_mission_utility", "mean"),
            mean_outage=("outage_rate", "mean"),
            mean_goodput=("avg_goodput", "mean"),
        )
        .sort_values(["mean_rank", "mean_utility"], ascending=[True, False])
    )
    stability.to_csv(output_dir / "link_sensitivity_rank_stability.csv", index=False, encoding="utf-8-sig")
    winners.to_csv(output_dir / "link_sensitivity_winners.csv", index=False, encoding="utf-8-sig")

    improved = forecast[forecast["policy"] == "Improved LSTM Policy"][
        ["scenario", "effective_path_m", "clear_sky_snr_db", "outage_penalty", "avg_mission_utility"]
    ].rename(columns={"avg_mission_utility": "improved_utility"})
    delta_records = []
    for baseline in ["LSTM-raw Policy", "TCN Policy", "TFT Policy", "PatchTST Policy", "Persistence Policy"]:
        base = forecast[forecast["policy"] == baseline][["scenario", "avg_mission_utility"]].rename(columns={"avg_mission_utility": "baseline_utility"})
        merged = improved.merge(base, on="scenario")
        merged["baseline"] = baseline
        merged["utility_delta"] = merged["improved_utility"] - merged["baseline_utility"]
        delta_records.append(merged)
    deltas = pd.concat(delta_records, ignore_index=True)
    deltas.to_csv(output_dir / "link_sensitivity_improved_deltas.csv", index=False, encoding="utf-8-sig")

    plot_sensitivity_rank_stability(stability, output_dir)
    plot_sensitivity_delta_heatmaps(deltas, output_dir)
    return all_summary, stability


def plot_sensitivity_rank_stability(stability: pd.DataFrame, output_dir: Path) -> None:
    plot_df = stability.copy()
    plot_df["policy_short"] = plot_df["policy"].str.replace(" Policy", "", regex=False)
    colors = ["#E8A87C" if p == "Improved LSTM Policy" else "#9DB7D1" for p in plot_df["policy"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    axes[0].barh(plot_df["policy_short"], plot_df["mean_rank"], color=colors, edgecolor="white")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Mean rank across 27 scenarios")
    axes[0].set_title("Rank stability")
    axes[1].barh(plot_df["policy_short"], plot_df["best_rank_count"], color=colors, edgecolor="white")
    axes[1].set_xlabel("Best-policy count")
    axes[1].set_title("Scenario wins")
    save_figure(fig, output_dir, "link_sensitivity_rank_stability")


def plot_sensitivity_delta_heatmaps(deltas: pd.DataFrame, output_dir: Path) -> None:
    subset = deltas[deltas["baseline"].isin(["TCN Policy", "LSTM-raw Policy"])].copy()
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2), sharex=True, sharey=True)
    for row_idx, baseline in enumerate(["TCN Policy", "LSTM-raw Policy"]):
        for col_idx, alpha in enumerate([4.0, 8.0, 12.0]):
            ax = axes[row_idx, col_idx]
            current = subset[(subset["baseline"] == baseline) & (subset["outage_penalty"] == alpha)]
            pivot = current.pivot(index="effective_path_m", columns="clear_sky_snr_db", values="utility_delta").sort_index(ascending=False)
            sns.heatmap(
                pivot,
                cmap=sns.diverging_palette(220, 25, as_cmap=True),
                center=0.0,
                annot=True,
                fmt=".3f",
                linewidths=0.5,
                linecolor="white",
                cbar=col_idx == 2,
                ax=ax,
            )
            ax.set_title(f"vs {baseline.replace(' Policy', '')}, alpha={alpha:.0f}")
            ax.set_xlabel("Clear-sky SNR (dB)")
            ax.set_ylabel("Effective path (m)" if col_idx == 0 else "")
    save_figure(fig, output_dir, "link_sensitivity_improved_delta_heatmaps")


def load_cross_site_predictions() -> pd.DataFrame:
    base = ROOT / "outputs" / "current_route_cross_site" / "formal_20260425_delta"
    raw = pd.read_csv(base / "lstm_raw" / "predictions_target_test.csv", parse_dates=["datetime"])
    improved = pd.read_csv(base / "lstm_improved" / "predictions_target_test.csv", parse_dates=["datetime"])
    raw_frame = raw[["datetime", "y_true_h1", "naive_persistence_h1", "lstm_h1"]].rename(
        columns={
            "datetime": "Date",
            "y_true_h1": "y_true_h1_raw",
            "naive_persistence_h1": "Persistence Policy",
            "lstm_h1": "LSTM-raw Policy",
        }
    )
    improved_frame = improved[["datetime", "y_true_h1", "lstm_h1"]].rename(
        columns={
            "datetime": "Date",
            "y_true_h1": "y_true_h1_improved",
            "lstm_h1": "Improved LSTM Policy",
        }
    )
    merged = raw_frame.merge(improved_frame, on="Date", how="inner")
    true_delta = float(np.nanmax(np.abs(merged["y_true_h1_raw"] - merged["y_true_h1_improved"])))
    if true_delta > 1e-5:
        raise ValueError(f"Cross-site aligned targets disagree; max delta={true_delta:.3e}")
    merged["y_true_h1"] = merged["y_true_h1_raw"]
    return merged.drop(columns=["y_true_h1_raw", "y_true_h1_improved"]).sort_values("Date").reset_index(drop=True)


def run_cross_site_link_simulation(output_dir: Path) -> pd.DataFrame:
    frame = load_cross_site_predictions()
    cfg = LinkBudgetConfig()
    timeseries, summary = build_policy_outputs(
        frame,
        cfg,
        forecast_policies=["Persistence Policy", "LSTM-raw Policy", "Improved LSTM Policy"],
    )
    summary = add_relative_columns(summary)
    frame.to_csv(output_dir / "cross_site_link_inputs.csv", index=False, encoding="utf-8-sig")
    timeseries.to_csv(output_dir / "cross_site_link_policy_timeseries.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "cross_site_link_policy_metrics.csv", index=False, encoding="utf-8-sig")
    plot_cross_site_link_metrics(summary, output_dir)
    return summary


def plot_cross_site_link_metrics(summary: pd.DataFrame, output_dir: Path) -> None:
    policies = ["Fixed High-Rate", "Persistence Policy", "LSTM-raw Policy", "Improved LSTM Policy", "Oracle Policy"]
    plot_df = summary[summary["policy"].isin(policies)].copy()
    plot_df["short"] = plot_df["policy"].str.replace(" Policy", "", regex=False).str.replace("Fixed ", "F-", regex=False)
    colors = ["#E8A87C" if p == "Improved LSTM Policy" else "#9DB7D1" if p in {"LSTM-raw Policy", "Persistence Policy"} else "#C7CDD6" for p in plot_df["policy"]]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    for ax, metric, title in [
        (axes[0], "avg_mission_utility", "Mission utility"),
        (axes[1], "avg_goodput", "Goodput"),
        (axes[2], "outage_rate", "Outage probability"),
    ]:
        ax.barh(plot_df["short"], plot_df[metric], color=colors, edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("")
        if metric == "outage_rate":
            ax.xaxis.set_major_formatter(lambda x, pos: f"{100*x:.1f}%")
    save_figure(fig, output_dir, "cross_site_link_policy_metrics")


def write_summary_markdown(
    output_dir: Path,
    forecast_stats: pd.DataFrame,
    link_stats: pd.DataFrame,
    sensitivity_stability: pd.DataFrame,
    cross_site_summary: pd.DataFrame,
) -> None:
    best_sensitivity = sensitivity_stability.iloc[0]
    improved_row = sensitivity_stability[sensitivity_stability["policy"] == "Improved LSTM Policy"].iloc[0]
    cross_forecast = cross_site_summary[cross_site_summary["policy_kind"] == "forecast"].sort_values("avg_mission_utility", ascending=False)
    summary = {
        "forecast_significance": forecast_stats.to_dict(orient="records"),
        "link_policy_bootstrap": link_stats.to_dict(orient="records"),
        "sensitivity_best_policy": best_sensitivity.to_dict(),
        "sensitivity_improved_policy": improved_row.to_dict(),
        "cross_site_best_forecast_policy": cross_forecast.iloc[0].to_dict(),
    }
    (output_dir / "review_response_improvements_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Review-response experiment additions",
        "",
        "## Forecasting statistical evidence",
        "",
        "| Comparison | Paired rows | MAE reduction | 95% CI | MSE reduction | 95% CI | Seeds improved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in forecast_stats.iterrows():
        lines.append(
            f"| {row['comparison']} | {int(row['paired_rows'])} | {row['mae_reduction_mean']:.5f} | "
            f"[{row['mae_reduction_ci95_low']:.5f}, {row['mae_reduction_ci95_high']:.5f}] | "
            f"{row['mse_reduction_mean']:.5f} | [{row['mse_reduction_ci95_low']:.5f}, {row['mse_reduction_ci95_high']:.5f}] | "
            f"{int(row['seeds_with_rmse_reduction'])}/{int(row['seed_count'])} |"
        )
    lines += [
        "",
        "## Link-policy bootstrap evidence",
        "",
        "| Comparison | Utility gain | 95% CI | Outage reduction | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in link_stats.iterrows():
        lines.append(
            f"| {row['comparison']} | {row['mission_utility_gain_mean']:.5f} | "
            f"[{row['mission_utility_gain_ci95_low']:.5f}, {row['mission_utility_gain_ci95_high']:.5f}] | "
            f"{100*row['outage_reduction_mean']:.3f}% | "
            f"[{100*row['outage_reduction_ci95_low']:.3f}%, {100*row['outage_reduction_ci95_high']:.3f}%] |"
        )
    lines += [
        "",
        "## Sensitivity analysis",
        "",
        f"- Best average rank policy: {best_sensitivity['policy']} (mean rank={best_sensitivity['mean_rank']:.2f}).",
        f"- Improved LSTM Policy: mean rank={improved_row['mean_rank']:.2f}, best-policy count={int(improved_row['best_rank_count'])}/27.",
        "",
        "## Cross-site link adaptation",
        "",
        "| Policy | Mission utility | Goodput | Outage | Availability |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in cross_site_summary[cross_site_summary["policy"].isin(["Persistence Policy", "LSTM-raw Policy", "Improved LSTM Policy", "Oracle Policy"])].iterrows():
        lines.append(
            f"| {row['policy']} | {row['avg_mission_utility']:.4f} | {row['avg_goodput']:.4f} | "
            f"{100*row['outage_rate']:.2f}% | {100*row['availability']:.2f}% |"
        )
    (output_dir / "review_response_improvements_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add review-response experiments: significance, sensitivity, and cross-site link simulation.")
    parser.add_argument("--output-dir", default="outputs/review_response_improvements_20260426")
    parser.add_argument("--bootstrap", type=int, default=4000, help="Reserved for reproducibility note; bootstrap count is fixed inside helper.")
    parser.add_argument("--seed", type=int, default=20260426)
    args = parser.parse_args()

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    rng = np.random.default_rng(args.seed)

    forecast_stats = run_forecast_significance(output_dir, rng)
    link_stats = run_default_link_bootstrap(output_dir, rng)
    _, sensitivity_stability = run_link_sensitivity(output_dir)
    cross_site_summary = run_cross_site_link_simulation(output_dir)
    write_summary_markdown(output_dir, forecast_stats, link_stats, sensitivity_stability, cross_site_summary)

    compact = {
        "output_dir": str(output_dir),
        "forecast_significance_csv": str(output_dir / "forecast_pairwise_bootstrap.csv"),
        "link_policy_bootstrap_csv": str(output_dir / "link_policy_pairwise_bootstrap.csv"),
        "link_sensitivity_csv": str(output_dir / "link_sensitivity_policy_metrics.csv"),
        "cross_site_link_csv": str(output_dir / "cross_site_link_policy_metrics.csv"),
        "config": asdict(LinkBudgetConfig()),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
