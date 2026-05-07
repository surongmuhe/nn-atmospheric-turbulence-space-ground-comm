from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LinkActionProfile:
    turbulence_level: str
    fec_profile: str
    modulation_profile: str
    channel_strategy: str
    pointing_tracking: str
    traffic_policy: str


ACTION_PROFILES = {
    "good": LinkActionProfile(
        turbulence_level="good",
        fec_profile="High-rate FEC (e.g. LDPC 5/6)",
        modulation_profile="Higher spectral efficiency (e.g. 16-QAM or mission baseline)",
        channel_strategy="Nominal channel plan and bandwidth allocation",
        pointing_tracking="Nominal tracking loop and standard pointing updates",
        traffic_policy="Allow full-rate payload traffic",
    ),
    "watch": LinkActionProfile(
        turbulence_level="watch",
        fec_profile="Moderate FEC (e.g. LDPC 3/4)",
        modulation_profile="Conservative step-down (e.g. QPSK/8-PSK depending on margin)",
        channel_strategy="Enable adaptive coding and moderate interleaving",
        pointing_tracking="Increase tracking refresh rate and jitter monitoring",
        traffic_policy="Reserve a small margin for retransmission or queue smoothing",
    ),
    "degraded": LinkActionProfile(
        turbulence_level="degraded",
        fec_profile="Robust FEC (e.g. LDPC 1/2)",
        modulation_profile="Robust modulation (e.g. QPSK)",
        channel_strategy="Increase interleaving and consider backup channel or frequency plan",
        pointing_tracking="Strengthen pointing compensation and shorten reacquisition interval",
        traffic_policy="Prioritize critical traffic and throttle high-rate low-priority data",
    ),
    "severe": LinkActionProfile(
        turbulence_level="severe",
        fec_profile="Maximum protection (e.g. LDPC 1/3 plus HARQ/repetition)",
        modulation_profile="Most robust mode (e.g. BPSK/QPSK fallback)",
        channel_strategy="Switch to survival mode, enable diversity or handover if available",
        pointing_tracking="Use aggressive tracking and reacquisition; widen beam/search window when supported",
        traffic_policy="Keep only mission-critical traffic and defer bulk transmission",
    ),
}


def compute_turbulence_thresholds(train_logcn2: np.ndarray, quantiles=(0.25, 0.5, 0.75)) -> dict:
    arr = np.asarray(train_logcn2, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Cannot compute turbulence thresholds from an empty array.")

    qs = np.quantile(arr, quantiles)
    return {
        "good_max": float(qs[0]),
        "watch_max": float(qs[1]),
        "degraded_max": float(qs[2]),
    }


def classify_turbulence_level(value: float, thresholds: dict) -> str:
    if value <= thresholds["good_max"]:
        return "good"
    if value <= thresholds["watch_max"]:
        return "watch"
    if value <= thresholds["degraded_max"]:
        return "degraded"
    return "severe"


def build_adaptation_table(pred_df: pd.DataFrame, pred_prefix: str, thresholds: dict) -> pd.DataFrame:
    required_cols = [f"{pred_prefix}_h1"]
    missing = [col for col in required_cols if col not in pred_df.columns]
    if missing:
        raise KeyError(f"Missing prediction columns: {missing}")

    horizon_cols = [col for col in pred_df.columns if col.startswith(f"{pred_prefix}_h")]
    if not horizon_cols:
        raise KeyError(f"No horizon columns found for prefix '{pred_prefix}'.")

    out = pred_df.copy()
    out["forecast_mean_logcn2"] = out[horizon_cols].mean(axis=1)
    out["forecast_worst_logcn2"] = out[horizon_cols].max(axis=1)
    out["turbulence_level"] = out["forecast_worst_logcn2"].apply(lambda x: classify_turbulence_level(float(x), thresholds))

    profiles = out["turbulence_level"].map(ACTION_PROFILES)
    out["fec_profile"] = profiles.map(lambda x: x.fec_profile)
    out["modulation_profile"] = profiles.map(lambda x: x.modulation_profile)
    out["channel_strategy"] = profiles.map(lambda x: x.channel_strategy)
    out["pointing_tracking"] = profiles.map(lambda x: x.pointing_tracking)
    out["traffic_policy"] = profiles.map(lambda x: x.traffic_policy)
    return out


def summarize_adaptation_table(adaptation_df: pd.DataFrame, thresholds: dict, pred_prefix: str) -> dict:
    horizon_cols = [col for col in adaptation_df.columns if col.startswith(f"{pred_prefix}_h")]
    level_counts = adaptation_df["turbulence_level"].value_counts(dropna=False).to_dict()
    return {
        "thresholds": thresholds,
        "prediction_prefix": pred_prefix,
        "horizon_columns": horizon_cols,
        "rows": int(len(adaptation_df)),
        "level_counts": {str(k): int(v) for k, v in level_counts.items()},
        "worst_case_stats": {
            "min": float(adaptation_df["forecast_worst_logcn2"].min()),
            "max": float(adaptation_df["forecast_worst_logcn2"].max()),
            "mean": float(adaptation_df["forecast_worst_logcn2"].mean()),
        },
    }

