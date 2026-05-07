from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erf, exp, log, pi, sqrt
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LinkBudgetConfig:
    wavelength_m: float = 1550e-9
    effective_turbulence_path_m: float = 500.0
    clear_sky_snr_db: float = 20.0
    outage_penalty: float = 8.0
    outage_floor: float = 1e-9
    outage_ceiling: float = 1.0 - 1e-9
    max_rytov_variance: float = 3.0


@dataclass(frozen=True)
class MCSMode:
    name: str
    modulation: str
    code_rate: str
    spectral_efficiency: float
    required_snr_db: float
    description: str


DEFAULT_MCS_MODES: tuple[MCSMode, ...] = (
    MCSMode(
        name="MCS0-Survival",
        modulation="BPSK",
        code_rate="1/3",
        spectral_efficiency=0.35,
        required_snr_db=3.0,
        description="Emergency mode for severe turbulence and link survival.",
    ),
    MCSMode(
        name="MCS1-Robust",
        modulation="QPSK",
        code_rate="1/2",
        spectral_efficiency=1.00,
        required_snr_db=7.0,
        description="Robust low-rate mode with strong FEC protection.",
    ),
    MCSMode(
        name="MCS2-Balanced",
        modulation="QPSK",
        code_rate="0.9",
        spectral_efficiency=1.80,
        required_snr_db=11.0,
        description="Balanced mode for moderate turbulence.",
    ),
    MCSMode(
        name="MCS3-HighRate",
        modulation="16-QAM",
        code_rate="3/4",
        spectral_efficiency=3.00,
        required_snr_db=15.5,
        description="High-rate mode with moderate coding overhead.",
    ),
    MCSMode(
        name="MCS4-Peak",
        modulation="16-QAM",
        code_rate="5/6",
        spectral_efficiency=4.20,
        required_snr_db=19.0,
        description="Peak-throughput mode used only when the fade margin is high.",
    ),
)


def normal_cdf(x: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(erf)(arr / sqrt(2.0)))


def logcn2_to_cn2(logcn2: np.ndarray | float) -> np.ndarray:
    return np.power(10.0, np.asarray(logcn2, dtype=np.float64))


def rytov_variance(logcn2: np.ndarray | float, cfg: LinkBudgetConfig) -> np.ndarray:
    cn2 = logcn2_to_cn2(logcn2)
    wavenumber = 2.0 * pi / cfg.wavelength_m
    sigma_r2 = 1.23 * cn2 * (wavenumber ** (7.0 / 6.0)) * (cfg.effective_turbulence_path_m ** (11.0 / 6.0))
    return np.clip(sigma_r2, 1e-12, cfg.max_rytov_variance)


def lognormal_irradiance_variance(logcn2: np.ndarray | float, cfg: LinkBudgetConfig) -> np.ndarray:
    sigma_r2 = rytov_variance(logcn2, cfg)
    scintillation_index = np.exp(sigma_r2) - 1.0
    return np.log1p(scintillation_index)


def outage_probability(logcn2: np.ndarray | float, mode: MCSMode, cfg: LinkBudgetConfig) -> np.ndarray:
    """Return P(SNR < required SNR) under a normalized lognormal fading model."""
    variance = lognormal_irradiance_variance(logcn2, cfg)
    std = np.sqrt(np.maximum(variance, 1e-12))
    mean = -0.5 * variance
    irradiance_threshold = 10.0 ** ((mode.required_snr_db - cfg.clear_sky_snr_db) / 10.0)
    z = (log(irradiance_threshold) - mean) / std
    prob = normal_cdf(z)
    return np.clip(prob, cfg.outage_floor, cfg.outage_ceiling)


def quantile_margin_db(logcn2: np.ndarray | float, mode: MCSMode, cfg: LinkBudgetConfig, quantile: float = 0.05) -> np.ndarray:
    """Approximate fade-margin at a low irradiance quantile using Acklam's inverse CDF."""
    variance = lognormal_irradiance_variance(logcn2, cfg)
    std = np.sqrt(np.maximum(variance, 1e-12))
    mean = -0.5 * variance
    z = inverse_normal_cdf(quantile)
    irradiance_q = np.exp(mean + std * z)
    snr_q_db = cfg.clear_sky_snr_db + 10.0 * np.log10(np.maximum(irradiance_q, 1e-12))
    return snr_q_db - mode.required_snr_db


def expected_goodput(logcn2: np.ndarray | float, mode: MCSMode, cfg: LinkBudgetConfig) -> np.ndarray:
    outage = outage_probability(logcn2, mode, cfg)
    return mode.spectral_efficiency * (1.0 - outage)


def mission_utility(logcn2: np.ndarray | float, mode: MCSMode, cfg: LinkBudgetConfig) -> np.ndarray:
    outage = outage_probability(logcn2, mode, cfg)
    return expected_goodput(logcn2, mode, cfg) - cfg.outage_penalty * outage


def choose_modes(predicted_logcn2: np.ndarray, modes: Iterable[MCSMode], cfg: LinkBudgetConfig) -> np.ndarray:
    mode_list = list(modes)
    utilities = np.column_stack([mission_utility(predicted_logcn2, mode, cfg) for mode in mode_list])
    return np.argmax(utilities, axis=1).astype(int)


def evaluate_policy(
    frame: pd.DataFrame,
    policy_name: str,
    true_col: str,
    prediction_col: str | None,
    modes: Iterable[MCSMode],
    cfg: LinkBudgetConfig,
    fixed_mode_index: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    mode_list = list(modes)
    true_logcn2 = frame[true_col].to_numpy(dtype=np.float64)
    if fixed_mode_index is not None:
        selected_idx = np.full(len(frame), fixed_mode_index, dtype=int)
        policy_kind = "fixed"
    elif prediction_col is None:
        selected_idx = choose_modes(true_logcn2, mode_list, cfg)
        policy_kind = "oracle"
    else:
        selected_idx = choose_modes(frame[prediction_col].to_numpy(dtype=np.float64), mode_list, cfg)
        policy_kind = "forecast"

    selected_modes = [mode_list[i] for i in selected_idx]
    outage = np.array([outage_probability(v, mode, cfg) for v, mode in zip(true_logcn2, selected_modes)], dtype=np.float64)
    goodput = np.array([mode.spectral_efficiency for mode in selected_modes], dtype=np.float64) * (1.0 - outage)
    utility = goodput - cfg.outage_penalty * outage
    margin_q05 = np.array([quantile_margin_db(v, mode, cfg, quantile=0.05) for v, mode in zip(true_logcn2, selected_modes)], dtype=np.float64)
    spectral_efficiency = np.array([mode.spectral_efficiency for mode in selected_modes], dtype=np.float64)

    switches = np.r_[False, selected_idx[1:] != selected_idx[:-1]]
    policy_frame = pd.DataFrame(
        {
            "Date": frame["Date"].to_numpy(),
            "policy": policy_name,
            "policy_kind": policy_kind,
            "true_logcn2": true_logcn2,
            "selected_mcs_index": selected_idx,
            "selected_mcs": [mode.name for mode in selected_modes],
            "selected_modulation": [mode.modulation for mode in selected_modes],
            "spectral_efficiency": spectral_efficiency,
            "outage_probability": outage,
            "availability_probability": 1.0 - outage,
            "expected_goodput": goodput,
            "mission_utility": utility,
            "q05_margin_db": margin_q05,
            "switch": switches,
        }
    )

    usage = pd.Series(selected_idx).value_counts(normalize=True).to_dict()
    summary = {
        "policy": policy_name,
        "policy_kind": policy_kind,
        "rows": int(len(frame)),
        "avg_mission_utility": float(np.mean(utility)),
        "avg_goodput": float(np.mean(goodput)),
        "avg_spectral_efficiency": float(np.mean(spectral_efficiency)),
        "outage_rate": float(np.mean(outage)),
        "availability": float(1.0 - np.mean(outage)),
        "q05_margin_db_mean": float(np.mean(margin_q05)),
        "q05_margin_db_p10": float(np.quantile(margin_q05, 0.10)),
        "switch_count": int(np.sum(switches)),
        "switching_rate": float(np.mean(switches)),
        "robust_mode_share": float(sum(usage.get(i, 0.0) for i in [0, 1])),
        "high_rate_mode_share": float(sum(usage.get(i, 0.0) for i in [3, 4])),
    }
    return policy_frame, summary


def mode_table(modes: Iterable[MCSMode]) -> pd.DataFrame:
    return pd.DataFrame([asdict(mode) for mode in modes])


def inverse_normal_cdf(p: float) -> float:
    """Acklam's rational approximation for the standard-normal inverse CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1).")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = sqrt(-2.0 * log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > phigh:
        q = sqrt(-2.0 * log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )
