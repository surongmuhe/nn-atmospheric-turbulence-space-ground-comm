from __future__ import annotations

import argparse
import json
import sys
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_review_response_improvements import (  # noqa: E402
    add_relative_columns,
    build_policy_outputs,
    load_aligned_main_site_predictions,
    load_cross_site_predictions,
)
from src.link_budget_simulation import (  # noqa: E402
    DEFAULT_MCS_MODES,
    LinkBudgetConfig,
    lognormal_irradiance_variance,
)


CODING_GAIN_DB = {
    "1/3": 5.0,
    "1/2": 3.2,
    "0.9": 0.8,
    "3/4": 1.8,
    "5/6": 0.8,
}


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
    if m == 2:
        return qfunc(np.sqrt(2.0 * snr_linear))
    if m == 4:
        return qfunc(np.sqrt(2.0 * snr_linear))
    k = np.log2(m)
    return (4.0 / k) * (1.0 - 1.0 / np.sqrt(m)) * qfunc(np.sqrt(3.0 * k * snr_linear / (m - 1.0)))


def expected_ber_lognormal(
    logcn2: np.ndarray,
    modulation: str,
    code_rate: str,
    cfg: LinkBudgetConfig,
    n_gh: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite expectation of BER under normalized lognormal scintillation."""
    logcn2 = np.asarray(logcn2, dtype=np.float64)
    variance = lognormal_irradiance_variance(logcn2, cfg)
    std = np.sqrt(np.maximum(variance, 1e-12))
    mean = -0.5 * variance
    nodes, weights = np.polynomial.hermite.hermgauss(n_gh)
    z = np.sqrt(2.0) * std[:, None] * nodes[None, :] + mean[:, None]
    irradiance = np.exp(z)
    clear_snr_linear = 10.0 ** (cfg.clear_sky_snr_db / 10.0)
    uncoded_snr = clear_snr_linear * irradiance
    coded_snr = uncoded_snr * (10.0 ** (CODING_GAIN_DB.get(str(code_rate), 0.0) / 10.0))
    uncoded = uncoded_ber_from_snr(uncoded_snr, modulation)
    coded = uncoded_ber_from_snr(coded_snr, modulation)
    norm = np.sqrt(np.pi)
    uncoded_exp = (uncoded * weights[None, :]).sum(axis=1) / norm
    coded_exp = (coded * weights[None, :]).sum(axis=1) / norm
    return np.clip(uncoded_exp, 1e-12, 0.5), np.clip(coded_exp, 1e-12, 0.5)


def add_ber_columns(policy_timeseries: pd.DataFrame, cfg: LinkBudgetConfig) -> pd.DataFrame:
    out = policy_timeseries.copy()
    out["expected_uncoded_ber"] = np.nan
    out["expected_coded_ber_proxy"] = np.nan
    for mode in DEFAULT_MCS_MODES:
        mask = out["selected_mcs"] == mode.name
        if not mask.any():
            continue
        uncoded, coded = expected_ber_lognormal(
            out.loc[mask, "true_logcn2"].to_numpy(dtype=np.float64),
            mode.modulation,
            mode.code_rate,
            cfg,
        )
        out.loc[mask, "expected_uncoded_ber"] = uncoded
        out.loc[mask, "expected_coded_ber_proxy"] = coded
    out["ber_proxy_exceeds_1e_3"] = out["expected_coded_ber_proxy"] > 1e-3
    out["ber_proxy_exceeds_1e_4"] = out["expected_coded_ber_proxy"] > 1e-4
    return out


def summarize_ber(policy_timeseries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in policy_timeseries.groupby("policy"):
        coded = group["expected_coded_ber_proxy"].to_numpy(dtype=np.float64)
        uncoded = group["expected_uncoded_ber"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "policy": policy,
                "rows": int(len(group)),
                "avg_uncoded_ber": float(np.mean(uncoded)),
                "avg_coded_ber_proxy": float(np.mean(coded)),
                "p95_coded_ber_proxy": float(np.quantile(coded, 0.95)),
                "p99_coded_ber_proxy": float(np.quantile(coded, 0.99)),
                "ber_proxy_exceed_1e_3_rate": float(np.mean(coded > 1e-3)),
                "ber_proxy_exceed_1e_4_rate": float(np.mean(coded > 1e-4)),
                "avg_goodput": float(group["expected_goodput"].mean()),
                "outage_rate": float(group["outage_probability"].mean()),
                "avg_mission_utility": float(group["mission_utility"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_mission_utility", ascending=False).reset_index(drop=True)


def run_one(frame: pd.DataFrame, output_dir: Path, prefix: str, forecast_policies: list[str]) -> pd.DataFrame:
    cfg = LinkBudgetConfig()
    timeseries, summary = build_policy_outputs(frame, cfg, forecast_policies=forecast_policies)
    timeseries = add_ber_columns(timeseries, cfg)
    ber_summary = summarize_ber(timeseries)
    summary = add_relative_columns(summary)
    merged_summary = summary.merge(
        ber_summary[
            [
                "policy",
                "avg_uncoded_ber",
                "avg_coded_ber_proxy",
                "p95_coded_ber_proxy",
                "p99_coded_ber_proxy",
                "ber_proxy_exceed_1e_3_rate",
                "ber_proxy_exceed_1e_4_rate",
            ]
        ],
        on="policy",
        how="left",
    )
    timeseries.to_csv(output_dir / f"{prefix}_ber_mcs_policy_timeseries.csv", index=False, encoding="utf-8-sig")
    merged_summary.to_csv(output_dir / f"{prefix}_ber_mcs_policy_summary.csv", index=False, encoding="utf-8-sig")
    return merged_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Add BER-proxy MCS evidence to the link-adaptation simulation.")
    parser.add_argument("--output-dir", default="outputs/ber_mcs_link_enhancement_20260427")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    main_summary = run_one(
        load_aligned_main_site_predictions(),
        output_dir,
        "main_site",
        [
            "Persistence Policy",
            "LSTM-raw Policy",
            "TCN Policy",
            "TFT Policy",
            "PatchTST Policy",
            "Improved LSTM Policy",
        ],
    )
    cross_summary = run_one(
        load_cross_site_predictions(),
        output_dir,
        "cross_site",
        ["Persistence Policy", "LSTM-raw Policy", "Improved LSTM Policy"],
    )
    mode_df = pd.DataFrame(
        [
            {
                "name": mode.name,
                "modulation": mode.modulation,
                "code_rate": mode.code_rate,
                "spectral_efficiency": mode.spectral_efficiency,
                "required_snr_db": mode.required_snr_db,
                "coding_gain_db_used_for_ber_proxy": CODING_GAIN_DB.get(str(mode.code_rate), 0.0),
                "description": mode.description,
            }
            for mode in DEFAULT_MCS_MODES
        ]
    )
    mode_df.to_csv(output_dir / "ber_mcs_mode_assumptions.csv", index=False, encoding="utf-8-sig")
    compact = {
        "output_dir": str(output_dir),
        "main_site_best_forecast": main_summary[main_summary["policy_kind"] == "forecast"].iloc[0].to_dict(),
        "cross_site_improved": cross_summary[cross_summary["policy"] == "Improved LSTM Policy"].iloc[0].to_dict(),
        "ber_proxy_note": "Expected BER is computed by Gauss-Hermite integration over normalized lognormal scintillation; coding gains are engineering proxies tied to MCS code rates.",
    }
    (output_dir / "ber_mcs_link_enhancement_summary.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
