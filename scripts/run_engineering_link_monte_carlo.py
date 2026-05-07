from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from math import erfc, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_review_response_improvements import load_aligned_main_site_predictions  # noqa: E402
from src.link_budget_simulation import DEFAULT_MCS_MODES, MCSMode  # noqa: E402


FORECAST_POLICIES = [
    "Persistence Policy",
    "LSTM-raw Policy",
    "TCN Policy",
    "TFT Policy",
    "PatchTST Policy",
    "Improved LSTM Policy",
]

FIXED_POLICIES = {
    "Fixed Survival": 0,
    "Fixed Conservative": 1,
    "Fixed Balanced": 2,
    "Fixed High-Rate": 3,
    "Fixed Peak-Rate": 4,
}

SHORT_LABELS = {
    "Fixed Conservative": "F-Conservative",
    "Fixed Balanced": "F-Balanced",
    "Fixed High-Rate": "F-High",
    "Fixed Peak-Rate": "F-Peak",
    "Persistence Policy": "Persistence",
    "LSTM-raw Policy": "LSTM-raw",
    "TCN Policy": "TCN",
    "TFT Policy": "TFT",
    "PatchTST Policy": "PatchTST",
    "Improved LSTM Policy": "Improved LSTM",
    "Oracle Policy": "Oracle",
}

CODING_GAIN_DB = {
    "1/3": 5.0,
    "1/2": 3.2,
    "0.9": 0.8,
    "3/4": 1.8,
    "5/6": 0.8,
}


@dataclass(frozen=True)
class EngineeringLinkConfig:
    wavelength_m: float = 1550e-9
    base_turbulence_path_m: float = 500.0
    altitude_m: float = 550_000.0
    reference_range_m: float = 550_000.0
    zenith_clear_snr_db: float = 22.0
    min_elevation_deg: float = 12.0
    max_elevation_deg: float = 82.0
    pass_len_steps: int = 36
    atmospheric_loss_db_per_airmass: float = 0.05
    pointing_jitter_urad: float = 3.0
    beam_divergence_urad: float = 20.0
    pointing_a0: float = 0.95
    max_rytov_variance: float = 6.0
    utility_outage_penalty: float = 8.0
    block_bits: int = 1024
    outage_bler_threshold: float = 0.10


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


def qfunc(x: np.ndarray) -> np.ndarray:
    return 0.5 * np.vectorize(erfc)(x / sqrt(2.0))


def parse_qam_order(modulation: str) -> int:
    modulation = modulation.upper().replace("-", "")
    if modulation == "BPSK":
        return 2
    if modulation == "QPSK":
        return 4
    if "QAM" in modulation:
        return int(modulation.split("QAM")[0])
    raise ValueError(f"Unsupported modulation: {modulation}")


def uncoded_ber_from_snr(snr_linear: np.ndarray, modulation: str) -> np.ndarray:
    m = parse_qam_order(modulation)
    snr_linear = np.maximum(np.asarray(snr_linear, dtype=np.float64), 1e-12)
    if m in {2, 4}:
        return qfunc(np.sqrt(2.0 * snr_linear))
    bits_per_symbol = np.log2(m)
    return (4.0 / bits_per_symbol) * (1.0 - 1.0 / np.sqrt(m)) * qfunc(
        np.sqrt(3.0 * bits_per_symbol * snr_linear / (m - 1.0))
    )


def add_pass_geometry(frame: pd.DataFrame, cfg: EngineeringLinkConfig) -> pd.DataFrame:
    out = frame.copy()
    n = len(out)
    phase = ((np.arange(n) % cfg.pass_len_steps) + 0.5) / cfg.pass_len_steps
    shape = np.sin(np.pi * phase) ** 1.15
    elevation_deg = cfg.min_elevation_deg + (cfg.max_elevation_deg - cfg.min_elevation_deg) * shape
    elevation_rad = np.deg2rad(elevation_deg)
    sin_el = np.maximum(np.sin(elevation_rad), 0.05)
    slant_range_m = cfg.altitude_m / sin_el
    airmass = 1.0 / sin_el
    fspl_delta_db = 20.0 * np.log10(slant_range_m / cfg.reference_range_m)
    atmospheric_delta_db = cfg.atmospheric_loss_db_per_airmass * (airmass - 1.0)
    clear_snr_db = cfg.zenith_clear_snr_db - fspl_delta_db - atmospheric_delta_db
    out["pass_id"] = np.arange(n) // cfg.pass_len_steps
    out["pass_phase"] = phase
    out["elevation_deg"] = elevation_deg
    out["slant_range_km"] = slant_range_m / 1000.0
    out["airmass"] = airmass
    out["clear_snr_db"] = clear_snr_db
    return out


def rytov_variance(logcn2: np.ndarray, airmass: np.ndarray, cfg: EngineeringLinkConfig) -> np.ndarray:
    cn2 = np.power(10.0, np.asarray(logcn2, dtype=np.float64))
    wavenumber = 2.0 * np.pi / cfg.wavelength_m
    path_m = cfg.base_turbulence_path_m * np.asarray(airmass, dtype=np.float64)
    sigma_r2 = 1.23 * cn2 * (wavenumber ** (7.0 / 6.0)) * (path_m ** (11.0 / 6.0))
    return np.clip(sigma_r2, 1e-12, cfg.max_rytov_variance)


def gamma_gamma_params(sigma_r2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sigma = np.maximum(np.asarray(sigma_r2, dtype=np.float64), 1e-8)
    alpha = 1.0 / (
        np.exp(0.49 * sigma / ((1.0 + 1.11 * sigma ** (12.0 / 5.0)) ** (7.0 / 6.0))) - 1.0
    )
    beta = 1.0 / (
        np.exp(0.51 * sigma / ((1.0 + 0.69 * sigma ** (12.0 / 5.0)) ** (5.0 / 6.0))) - 1.0
    )
    return np.clip(alpha, 0.2, 200.0), np.clip(beta, 0.2, 200.0)


def sample_channel_gain(
    logcn2: np.ndarray,
    geometry: pd.DataFrame,
    cfg: EngineeringLinkConfig,
    rng: np.random.Generator,
    n_mc: int,
) -> np.ndarray:
    logcn2 = np.asarray(logcn2, dtype=np.float64)
    sigma_r2 = rytov_variance(logcn2, geometry["airmass"].to_numpy(dtype=np.float64), cfg)
    alpha, beta = gamma_gamma_params(sigma_r2)
    small_scale = rng.gamma(shape=alpha[:, None], scale=1.0 / alpha[:, None], size=(len(logcn2), n_mc))
    large_scale = rng.gamma(shape=beta[:, None], scale=1.0 / beta[:, None], size=(len(logcn2), n_mc))
    scintillation = small_scale * large_scale

    range_m = geometry["slant_range_km"].to_numpy(dtype=np.float64) * 1000.0
    beam_radius = np.maximum(cfg.beam_divergence_urad * 1e-6 * range_m, 1e-3)
    jitter_sigma = cfg.pointing_jitter_urad * 1e-6 * range_m
    radial_error = rng.rayleigh(scale=jitter_sigma[:, None], size=(len(logcn2), n_mc))
    pointing = cfg.pointing_a0 * np.exp(-2.0 * np.square(radial_error / beam_radius[:, None]))
    return np.clip(scintillation * pointing, 1e-12, None)


def snr_samples_db(gain: np.ndarray, clear_snr_db: np.ndarray) -> np.ndarray:
    return clear_snr_db[:, None] + 10.0 * np.log10(np.maximum(gain, 1e-12))


def mode_metrics_from_snr(snr_db: np.ndarray, mode: MCSMode, cfg: EngineeringLinkConfig) -> dict[str, np.ndarray]:
    snr_linear = 10.0 ** (snr_db / 10.0)
    coding_gain = 10.0 ** (CODING_GAIN_DB.get(str(mode.code_rate), 0.0) / 10.0)
    coded_ber = uncoded_ber_from_snr(snr_linear * coding_gain, mode.modulation)
    bler = 1.0 - np.exp(-np.clip(coded_ber * cfg.block_bits, 0.0, 60.0))
    threshold_outage = snr_db < mode.required_snr_db
    block_outage = bler > cfg.outage_bler_threshold
    outage = threshold_outage | block_outage
    return {
        "outage_rate": outage.mean(axis=1),
        "avg_coded_ber": coded_ber.mean(axis=1),
        "p99_coded_ber": np.quantile(coded_ber, 0.99, axis=1),
        "bler": bler.mean(axis=1),
        "goodput": mode.spectral_efficiency * (1.0 - bler).mean(axis=1),
    }


def choose_modes_from_samples(snr_db: np.ndarray, modes: tuple[MCSMode, ...], cfg: EngineeringLinkConfig) -> np.ndarray:
    utilities = []
    for mode in modes:
        metrics = mode_metrics_from_snr(snr_db, mode, cfg)
        utility = metrics["goodput"] - cfg.utility_outage_penalty * metrics["outage_rate"]
        utilities.append(utility)
    return np.argmax(np.column_stack(utilities), axis=1).astype(int)


def evaluate_selected_modes(
    snr_db_true: np.ndarray,
    selected_idx: np.ndarray,
    modes: tuple[MCSMode, ...],
    cfg: EngineeringLinkConfig,
) -> pd.DataFrame:
    rows = []
    for idx, mode in enumerate(modes):
        mask = selected_idx == idx
        if not mask.any():
            continue
        metrics = mode_metrics_from_snr(snr_db_true[mask], mode, cfg)
        rows.append(
            pd.DataFrame(
                {
                    "row_index": np.flatnonzero(mask),
                    "selected_mcs_index": idx,
                    "selected_mcs": mode.name,
                    "selected_modulation": mode.modulation,
                    "spectral_efficiency": mode.spectral_efficiency,
                    "outage_rate": metrics["outage_rate"],
                    "avg_coded_ber": metrics["avg_coded_ber"],
                    "p99_coded_ber": metrics["p99_coded_ber"],
                    "avg_bler": metrics["bler"],
                    "goodput": metrics["goodput"],
                }
            )
        )
    out = pd.concat(rows, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    out["mission_utility"] = out["goodput"] - cfg.utility_outage_penalty * out["outage_rate"]
    return out


def summarize_policy(policy_frame: pd.DataFrame) -> dict[str, float | int | str]:
    switches = np.r_[False, policy_frame["selected_mcs_index"].to_numpy()[1:] != policy_frame["selected_mcs_index"].to_numpy()[:-1]]
    return {
        "policy": str(policy_frame["policy"].iloc[0]),
        "policy_kind": str(policy_frame["policy_kind"].iloc[0]),
        "rows": int(len(policy_frame)),
        "avg_mission_utility": float(policy_frame["mission_utility"].mean()),
        "avg_goodput": float(policy_frame["goodput"].mean()),
        "outage_rate": float(policy_frame["outage_rate"].mean()),
        "availability": float(1.0 - policy_frame["outage_rate"].mean()),
        "avg_coded_ber": float(policy_frame["avg_coded_ber"].mean()),
        "p99_coded_ber": float(policy_frame["p99_coded_ber"].quantile(0.99)),
        "avg_bler": float(policy_frame["avg_bler"].mean()),
        "switch_count": int(switches.sum()),
        "switching_rate": float(switches.mean()),
        "robust_mode_share": float(policy_frame["selected_mcs_index"].isin([0, 1]).mean()),
        "high_rate_mode_share": float(policy_frame["selected_mcs_index"].isin([3, 4]).mean()),
    }


def bootstrap_delta(
    policy_timeseries: pd.DataFrame,
    metric_col: str,
    policy_a: str,
    policy_b: str,
    rng: np.random.Generator,
    n_boot: int = 3000,
) -> dict[str, float | str]:
    pivot = policy_timeseries.pivot(index="row_index", columns="policy", values=metric_col)
    delta = (pivot[policy_a] - pivot[policy_b]).dropna().to_numpy(dtype=np.float64)
    boot = np.empty(n_boot, dtype=np.float64)
    chunk = 200
    filled = 0
    while filled < n_boot:
        take = min(chunk, n_boot - filled)
        idx = rng.integers(0, len(delta), size=(take, len(delta)))
        boot[filled : filled + take] = delta[idx].mean(axis=1)
        filled += take
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "metric": metric_col,
        "policy_a": policy_a,
        "policy_b": policy_b,
        "mean_delta_a_minus_b": float(delta.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "prob_delta_positive": float(np.mean(boot > 0.0)),
    }


def run_simulation(frame: pd.DataFrame, cfg: EngineeringLinkConfig, n_mc_eval: int, n_mc_select: int, seed: int):
    rng_select = np.random.default_rng(seed)
    rng_eval = np.random.default_rng(seed + 100_000)
    geom = add_pass_geometry(frame, cfg)
    modes = tuple(DEFAULT_MCS_MODES)

    true_gain = sample_channel_gain(geom["y_true_h1"].to_numpy(), geom, cfg, rng_eval, n_mc_eval)
    true_snr_db = snr_samples_db(true_gain, geom["clear_snr_db"].to_numpy(dtype=np.float64))

    policy_frames = []
    for policy, fixed_idx in FIXED_POLICIES.items():
        selected_idx = np.full(len(geom), fixed_idx, dtype=int)
        evaluated = evaluate_selected_modes(true_snr_db, selected_idx, modes, cfg)
        policy_frames.append(pd.concat([geom.reset_index(names="row_index"), evaluated.drop(columns="row_index")], axis=1).assign(policy=policy, policy_kind="fixed"))

    for policy in FORECAST_POLICIES:
        pred_gain = sample_channel_gain(geom[policy].to_numpy(), geom, cfg, rng_select, n_mc_select)
        pred_snr_db = snr_samples_db(pred_gain, geom["clear_snr_db"].to_numpy(dtype=np.float64))
        selected_idx = choose_modes_from_samples(pred_snr_db, modes, cfg)
        evaluated = evaluate_selected_modes(true_snr_db, selected_idx, modes, cfg)
        policy_frames.append(pd.concat([geom.reset_index(names="row_index"), evaluated.drop(columns="row_index")], axis=1).assign(policy=policy, policy_kind="forecast"))

    oracle_idx = choose_modes_from_samples(true_snr_db, modes, cfg)
    evaluated = evaluate_selected_modes(true_snr_db, oracle_idx, modes, cfg)
    policy_frames.append(pd.concat([geom.reset_index(names="row_index"), evaluated.drop(columns="row_index")], axis=1).assign(policy="Oracle Policy", policy_kind="oracle"))

    policy_timeseries = pd.concat(policy_frames, ignore_index=True)
    cols = ["policy", "policy_kind"] + [c for c in policy_timeseries.columns if c not in {"policy", "policy_kind"}]
    policy_timeseries = policy_timeseries[cols]
    summary = pd.DataFrame([summarize_policy(g) for _, g in policy_timeseries.groupby("policy")])
    summary = summary.sort_values("avg_mission_utility", ascending=False).reset_index(drop=True)
    return geom, policy_timeseries, summary


def plot_policy_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    order = [
        "Fixed Conservative",
        "Fixed Balanced",
        "Fixed High-Rate",
        "Fixed Peak-Rate",
        "Persistence Policy",
        "LSTM-raw Policy",
        "TCN Policy",
        "TFT Policy",
        "PatchTST Policy",
        "Improved LSTM Policy",
        "Oracle Policy",
    ]
    plot_df = summary[summary["policy"].isin(order)].copy()
    plot_df["policy"] = pd.Categorical(plot_df["policy"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("policy")
    plot_df["label"] = plot_df["policy"].astype(str).map(SHORT_LABELS)

    def color(policy: str) -> str:
        if policy == "Improved LSTM Policy":
            return "#E8A87C"
        if policy == "Oracle Policy":
            return "#7FB069"
        if policy.startswith("Fixed"):
            return "#D6D6D6"
        return "#B7C9E2"

    colors = [color(str(p)) for p in plot_df["policy"]]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6))
    metrics = [
        ("avg_mission_utility", "Mission utility"),
        ("avg_goodput", "Expected goodput"),
        ("outage_rate", "Outage rate"),
        ("avg_bler", "Mean BLER"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        ax.barh(plot_df["label"], plot_df[metric], color=colors, edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("")
        if metric in {"outage_rate", "avg_bler"}:
            ax.xaxis.set_major_formatter(lambda x, pos: f"{100*x:.1f}%")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.7)
    save_figure(fig, output_dir, "engineering_policy_summary")


def plot_pass_timeline(policy_timeseries: pd.DataFrame, output_dir: Path, pass_id: int = 3) -> None:
    keep = ["Improved LSTM Policy", "TCN Policy", "LSTM-raw Policy", "Persistence Policy", "Oracle Policy"]
    df = policy_timeseries[(policy_timeseries["pass_id"] == pass_id) & (policy_timeseries["policy"].isin(keep))].copy()
    if df.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12.2, 8.2), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.1, 1.1]})
    base = df[df["policy"] == keep[0]].sort_values("Date")
    axes[0].plot(base["Date"], base["elevation_deg"], color="#4C78A8", linewidth=2.0)
    axes[0].set_ylabel("Elevation (deg)")
    axes[0].set_title("Example pass geometry and policy behavior")
    for policy in keep:
        cur = df[df["policy"] == policy].sort_values("Date")
        axes[1].step(cur["Date"], cur["selected_mcs_index"], where="mid", label=SHORT_LABELS.get(policy, policy), linewidth=1.6)
        axes[2].plot(cur["Date"], cur["outage_rate"], label=SHORT_LABELS.get(policy, policy), linewidth=1.5)
    axes[1].set_ylabel("MCS index")
    axes[1].set_yticks([0, 1, 2, 3, 4])
    axes[2].set_ylabel("Outage rate")
    axes[2].yaxis.set_major_formatter(lambda x, pos: f"{100*x:.0f}%")
    axes[2].legend(ncol=3, fontsize=8)
    axes[2].set_xlabel("Time")
    save_figure(fig, output_dir, "engineering_example_pass_timeline")


def write_outputs(output_dir: Path, cfg: EngineeringLinkConfig, geom: pd.DataFrame, policy_timeseries: pd.DataFrame, summary: pd.DataFrame, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    geom.to_csv(output_dir / "engineering_pass_geometry.csv", index=False, encoding="utf-8-sig")
    policy_timeseries.to_csv(output_dir / "engineering_policy_timeseries.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "engineering_policy_summary.csv", index=False, encoding="utf-8-sig")
    usage = (
        policy_timeseries.groupby(["policy", "selected_mcs"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    usage["share"] = usage["count"] / usage.groupby("policy")["count"].transform("sum")
    usage.to_csv(output_dir / "engineering_mcs_usage.csv", index=False, encoding="utf-8-sig")
    rng = np.random.default_rng(seed + 200_000)
    boot_rows = [
        bootstrap_delta(policy_timeseries, "mission_utility", "Improved LSTM Policy", "TCN Policy", rng),
        bootstrap_delta(policy_timeseries, "mission_utility", "Improved LSTM Policy", "LSTM-raw Policy", rng),
        bootstrap_delta(policy_timeseries, "outage_rate", "TCN Policy", "Improved LSTM Policy", rng),
        bootstrap_delta(policy_timeseries, "outage_rate", "LSTM-raw Policy", "Improved LSTM Policy", rng),
    ]
    bootstrap = pd.DataFrame(boot_rows)
    bootstrap.to_csv(output_dir / "engineering_policy_bootstrap_delta.csv", index=False, encoding="utf-8-sig")
    compact = {
        "config": asdict(cfg),
        "best_policy": summary.iloc[0].to_dict(),
        "improved_lstm": summary.loc[summary["policy"] == "Improved LSTM Policy"].iloc[0].to_dict(),
        "tcn": summary.loc[summary["policy"] == "TCN Policy"].iloc[0].to_dict(),
        "lstm_raw": summary.loc[summary["policy"] == "LSTM-raw Policy"].iloc[0].to_dict(),
        "bootstrap": boot_rows,
    }
    (output_dir / "engineering_link_monte_carlo_summary.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quasi-engineering satellite-ground optical-link Monte Carlo simulation.")
    parser.add_argument("--output-dir", default="outputs/engineering_link_monte_carlo_20260428")
    parser.add_argument("--n-mc-eval", type=int, default=1600)
    parser.add_argument("--n-mc-select", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260428)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    cfg = EngineeringLinkConfig()
    frame = load_aligned_main_site_predictions()
    geom, policy_timeseries, summary = run_simulation(frame, cfg, args.n_mc_eval, args.n_mc_select, args.seed)
    write_outputs(output_dir, cfg, geom, policy_timeseries, summary, args.seed)
    set_plot_style()
    plot_policy_summary(summary, output_dir)
    plot_pass_timeline(policy_timeseries, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "best_policy": summary.iloc[0].to_dict()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
