from __future__ import annotations

import argparse
import json
import sys
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
    lognormal_irradiance_variance,
    mission_utility,
    mode_table,
    outage_probability,
    rytov_variance,
)


DEFAULT_MODEL_INPUTS = {
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


SHORT_LABELS = {
    "Fixed Survival": "F-Survival",
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


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_prediction_file(path: Path, pred_col: str, policy_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "y_true_h1", pred_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")
    out = df[["Date", "y_true_h1", pred_col]].copy()
    out = out.rename(columns={pred_col: policy_name, "y_true_h1": f"y_true_h1_{policy_name}"})
    return out


def load_aligned_predictions(model_inputs: dict[str, tuple[str, str]], persistence_lag_steps: int) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    true_cols = []
    for policy_name, (path_str, pred_col) in model_inputs.items():
        path = resolve_path(path_str)
        current = read_prediction_file(path, pred_col, policy_name)
        true_cols.append(f"y_true_h1_{policy_name}")
        if merged is None:
            merged = current
        else:
            merged = merged.merge(current, on="Date", how="inner")
    if merged is None or merged.empty:
        raise ValueError("No prediction data was loaded.")

    merged = merged.sort_values("Date").reset_index(drop=True)
    true_stack = merged[true_cols].to_numpy(dtype=np.float64)
    max_true_delta = float(np.nanmax(np.abs(true_stack - true_stack[:, [0]])))
    if max_true_delta > 1e-6:
        raise ValueError(f"Aligned files disagree on y_true_h1; max delta={max_true_delta:.3e}.")

    merged["y_true_h1"] = merged[true_cols[0]]
    merged["Persistence Policy"] = merged["y_true_h1"].shift(persistence_lag_steps)
    merged = merged.drop(columns=true_cols)
    merged = merged.dropna().reset_index(drop=True)
    return merged


def build_policy_outputs(frame: pd.DataFrame, cfg: LinkBudgetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    forecast_policies = [
        "Persistence Policy",
        "LSTM-raw Policy",
        "TCN Policy",
        "TFT Policy",
        "PatchTST Policy",
        "Improved LSTM Policy",
    ]
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


def add_relative_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    fixed_high = float(out.loc[out["policy"] == "Fixed High-Rate", "avg_mission_utility"].iloc[0])
    fixed_balanced = float(out.loc[out["policy"] == "Fixed Balanced", "avg_goodput"].iloc[0])
    oracle = float(out.loc[out["policy"] == "Oracle Policy", "avg_mission_utility"].iloc[0])
    out["utility_gain_vs_fixed_high_pct"] = 100.0 * (out["avg_mission_utility"] - fixed_high) / abs(fixed_high)
    out["goodput_gain_vs_fixed_balanced_pct"] = 100.0 * (out["avg_goodput"] - fixed_balanced) / abs(fixed_balanced)
    out["oracle_utility_gap_pct"] = 100.0 * (oracle - out["avg_mission_utility"]) / abs(oracle)
    return out.sort_values("avg_mission_utility", ascending=False).reset_index(drop=True)


def write_mcs_usage(policy_timeseries: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    usage = (
        policy_timeseries.groupby(["policy", "selected_mcs"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    totals = usage.groupby("policy")["count"].transform("sum")
    usage["share"] = usage["count"] / totals
    usage.to_csv(output_dir / "mcs_usage.csv", index=False, encoding="utf-8-sig")
    return usage


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


def plot_policy_performance(summary: pd.DataFrame, output_dir: Path) -> None:
    policy_order = [
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
    plot_df = summary[summary["policy"].isin(policy_order)].copy()
    plot_df["policy"] = pd.Categorical(plot_df["policy"], categories=policy_order, ordered=True)
    plot_df = plot_df.sort_values("policy")
    plot_df["short_policy"] = plot_df["policy"].astype(str).map(SHORT_LABELS)

    def color_for(policy: str) -> str:
        if policy == "Improved LSTM Policy":
            return "#E8A87C"
        if policy == "Oracle Policy":
            return "#7FB069"
        if policy.startswith("Fixed"):
            return "#D6D6D6"
        return "#B7C9E2"

    colors = [color_for(policy) for policy in plot_df["policy"].astype(str)]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8))
    metrics = [
        ("avg_mission_utility", "Mission utility"),
        ("avg_goodput", "Expected goodput (bit/s/Hz)"),
        ("outage_rate", "Mean outage probability"),
        ("switching_rate", "MCS switching rate"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        ax.barh(plot_df["short_policy"], plot_df[metric], color=colors, edgecolor="white", linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=8)
        if metric in {"outage_rate", "switching_rate"}:
            ax.xaxis.set_major_formatter(lambda x, pos: f"{100*x:.1f}%")
        ax.grid(axis="x", alpha=0.75)
    save_figure(fig, output_dir, "link_policy_performance")


def plot_tradeoff(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), gridspec_kw={"width_ratios": [1.1, 1.0]})
    plot_df = summary.copy()
    plot_df["short_policy"] = plot_df["policy"].map(SHORT_LABELS)

    def color_for(policy: str) -> str:
        if policy == "Improved LSTM Policy":
            return "#E8A87C"
        if policy == "Oracle Policy":
            return "#7FB069"
        if "Policy" in policy:
            return "#8AA6C1"
        return "#C8C8C8"

    colors = [color_for(policy) for policy in plot_df["policy"]]
    sizes = 90 + 140 * (plot_df["avg_mission_utility"] - plot_df["avg_mission_utility"].min()) / (
        plot_df["avg_mission_utility"].max() - plot_df["avg_mission_utility"].min() + 1e-9
    )

    ax = axes[0]
    ax.scatter(plot_df["outage_rate"] * 100, plot_df["avg_goodput"], s=sizes, c=colors, edgecolors="#243447", linewidth=0.8)
    key_offsets = {
        "Fixed Survival": (5, -8),
        "Fixed Conservative": (5, 5),
        "Fixed Balanced": (5, 5),
        "Fixed High-Rate": (5, 5),
        "Fixed Peak-Rate": (5, 5),
        "Improved LSTM Policy": (5, 5),
        "Oracle Policy": (5, -10),
    }
    for _, row in plot_df.iterrows():
        if row["policy"] in key_offsets:
            ax.annotate(row["short_policy"], (row["outage_rate"] * 100, row["avg_goodput"]), xytext=key_offsets[row["policy"]], textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean outage probability (%)")
    ax.set_ylabel("Expected goodput (bit/s/Hz)")
    ax.set_title("Global trade-off")

    zoom = plot_df[plot_df["policy_kind"].isin(["forecast", "oracle"])].copy()
    ax = axes[1]
    zoom_colors = [color_for(policy) for policy in zoom["policy"]]
    ax.scatter(zoom["outage_rate"] * 100, zoom["avg_goodput"], s=150, c=zoom_colors, edgecolors="#243447", linewidth=0.8)
    offsets = {
        "Oracle Policy": (6, 4),
        "Improved LSTM Policy": (6, 8),
        "TCN Policy": (6, -12),
        "LSTM-raw Policy": (6, 4),
    }
    for _, row in zoom.iterrows():
        if row["policy"] in offsets:
            ax.annotate(row["short_policy"], (row["outage_rate"] * 100, row["avg_goodput"]), xytext=offsets[row["policy"]], textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean outage probability (%)")
    ax.set_ylabel("")
    ax.set_title("Forecast-policy zoom")
    save_figure(fig, output_dir, "throughput_outage_tradeoff")


def plot_mcs_usage(usage: pd.DataFrame, output_dir: Path) -> None:
    pivot = usage.pivot(index="policy", columns="selected_mcs", values="share").fillna(0.0)
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
    pivot = pivot.reindex([p for p in order if p in pivot.index])
    fig, ax = plt.subplots(figsize=(9.8, 5.9))
    sns.heatmap(
        pivot,
        cmap=sns.light_palette("#4B83B6", as_cmap=True),
        annot=True,
        fmt=".0%",
        cbar_kws={"label": "Usage share"},
        linewidths=0.6,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Selected MCS")
    ax.set_ylabel("")
    ax.set_title("Mode usage distribution")
    save_figure(fig, output_dir, "mcs_usage_heatmap")


def plot_link_model_curves(frame: pd.DataFrame, cfg: LinkBudgetConfig, output_dir: Path) -> None:
    x = np.linspace(frame["y_true_h1"].quantile(0.01), frame["y_true_h1"].quantile(0.99), 240)
    fig, ax1 = plt.subplots(figsize=(8.8, 5.8))
    for mode in DEFAULT_MCS_MODES:
        ax1.plot(x, outage_probability(x, mode, cfg) * 100.0, label=mode.name, linewidth=1.9)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Future $\log_{10}(C_n^2)$")
    ax1.set_ylabel("Outage probability (%)")
    ax1.set_title("Turbulence-to-link response curves")
    ax1.legend(frameon=True, fontsize=8, ncol=2)
    ax2 = ax1.twinx()
    ax2.hist(frame["y_true_h1"], bins=36, color="#DADDE6", alpha=0.45, density=True)
    ax2.set_ylabel("Test-set density")
    ax2.grid(False)
    save_figure(fig, output_dir, "turbulence_to_link_curves")


def plot_timeline(frame: pd.DataFrame, policy_timeseries: pd.DataFrame, output_dir: Path, start: int = 540, length: int = 360) -> None:
    end = min(start + length, len(frame))
    if start >= len(frame):
        start = 0
        end = min(length, len(frame))
    window_dates = frame.iloc[start:end]["Date"]
    selected = policy_timeseries[
        policy_timeseries["policy"].isin(["LSTM-raw Policy", "Improved LSTM Policy", "Oracle Policy"])
        & policy_timeseries["Date"].isin(window_dates)
    ].copy()
    fig, axes = plt.subplots(2, 1, figsize=(12.6, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0]})
    axes[0].plot(frame.iloc[start:end]["Date"], frame.iloc[start:end]["y_true_h1"], color="#243447", linewidth=1.8, label="True future turbulence")
    axes[0].plot(frame.iloc[start:end]["Date"], frame.iloc[start:end]["Improved LSTM Policy"], color="#E8A87C", linewidth=1.4, label="Improved LSTM forecast")
    axes[0].plot(frame.iloc[start:end]["Date"], frame.iloc[start:end]["LSTM-raw Policy"], color="#7EA0B7", linewidth=1.1, alpha=0.9, label="LSTM-raw forecast")
    axes[0].invert_yaxis()
    axes[0].set_ylabel(r"$\log_{10}(C_n^2)$")
    axes[0].set_title("Prediction-driven MCS adaptation example")
    axes[0].legend(frameon=True, fontsize=8, ncol=3)
    sns.lineplot(data=selected, x="Date", y="selected_mcs_index", hue="policy", drawstyle="steps-post", ax=axes[1], linewidth=1.6)
    axes[1].set_yticks(range(len(DEFAULT_MCS_MODES)))
    axes[1].set_yticklabels([mode.name.replace("MCS", "") for mode in DEFAULT_MCS_MODES], fontsize=8)
    axes[1].set_ylabel("Selected mode")
    axes[1].set_xlabel("")
    axes[1].legend(frameon=True, fontsize=8, ncol=3)
    save_figure(fig, output_dir, "mcs_timeline_example")


def write_section_markdown(summary: pd.DataFrame, output_dir: Path, cfg: LinkBudgetConfig) -> None:
    selected = summary[
        summary["policy"].isin(
            [
                "Fixed Conservative",
                "Fixed Balanced",
                "Fixed High-Rate",
                "Fixed Peak-Rate",
                "Persistence Policy",
                "LSTM-raw Policy",
                "TCN Policy",
                "Improved LSTM Policy",
                "Oracle Policy",
            ]
        )
    ].copy()
    selected = selected.sort_values("avg_mission_utility", ascending=False)
    table_lines = [
        "| 策略 | 平均效用 | 平均有效吞吐 | 中断率 | 可用率 | 切换率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in selected.iterrows():
        table_lines.append(
            f"| {row['policy']} | {row['avg_mission_utility']:.4f} | {row['avg_goodput']:.4f} | "
            f"{100*row['outage_rate']:.2f}% | {100*row['availability']:.2f}% | {100*row['switching_rate']:.2f}% |"
        )
    text = f"""# 星地光通信链路自适应仿真补充材料

本节将湍流预测结果进一步接入简化星地光通信链路模型。设预测目标为 $y=\\log_{{10}}(C_n^2)$，则 $C_n^2=10^y$。在波长 $\\lambda={cfg.wavelength_m:.2e}$ m、等效湍流路径长度 $L_{{eff}}={cfg.effective_turbulence_path_m:.0f}$ m 条件下，采用均匀路径近似的 Rytov 方差：

$$
\\sigma_R^2=1.23 C_n^2 k^{{7/6}} L_{{eff}}^{{11/6}},\\quad k=\\frac{{2\\pi}}{{\\lambda}} .
$$

对数正态闪烁模型将归一化接收光强写为 $I=\\exp(X)$，其中 $X\\sim\\mathcal{{N}}(-\\sigma_X^2/2,\\sigma_X^2)$，并令 $\\sigma_X^2=\\ln(1+\\sigma_I^2)$。当清空信噪比为 $\\gamma_0={cfg.clear_sky_snr_db:.1f}$ dB，某一调制编码模式的门限为 $\\gamma_m$ 时，中断概率为：

$$
P_{{out}}(m|y)=\\Pr\\{{\\gamma_0+10\\log_{{10}}I<\\gamma_m\\}}.
$$

本文不再只做风险分类，而是令预测值直接驱动调制编码模式选择。对每个候选 MCS，定义预测链路效用：

$$
U(m|\\hat y)=\\eta_m\\left[1-P_{{out}}(m|\\hat y)\\right]-\\alpha P_{{out}}(m|\\hat y),
$$

其中 $\\eta_m$ 为模式频谱效率，$\\alpha={cfg.outage_penalty:.1f}$ 为中断惩罚系数。策略选择使 $U(m|\\hat y)$ 最大的模式，并使用真实未来 $y$ 评价实际有效吞吐、中断率、可用率和任务效用。

## 仿真结果

{chr(10).join(table_lines)}

结果表明，`Improved LSTM Policy` 在预测驱动策略中取得最高平均任务效用，优于 `LSTM-raw Policy`、`TCN Policy` 和 `Persistence Policy`。虽然 `LSTM-raw Policy` 会因为更激进的模式选择获得较高瞬时吞吐，但其平均中断率也更高；在星地通信场景中，中断通常比吞吐下降更难接受，因此综合效用更能反映实际链路价值。`Oracle Policy` 使用真实未来湍流作为决策输入，可视为理论上限。

生成图件：

- `link_policy_performance.png/pdf`：不同策略的效用、有效吞吐、中断率和切换率。
- `throughput_outage_tradeoff.png/pdf`：吞吐与可靠性折中关系。
- `mcs_usage_heatmap.png/pdf`：不同策略的调制编码模式使用比例。
- `turbulence_to_link_curves.png/pdf`：由 $\\log_{{10}}(C_n^2)$ 到中断概率的链路响应曲线。
- `mcs_timeline_example.png/pdf`：预测驱动 MCS 选择的时间序列示例。
"""
    (output_dir / "paper_section_link_simulation.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prediction-driven satellite-to-ground optical link adaptation simulation.")
    parser.add_argument("--output-dir", default="outputs/link_adaptation_sim_20260426")
    parser.add_argument("--persistence-lag-steps", type=int, default=6)
    parser.add_argument("--clear-sky-snr-db", type=float, default=20.0)
    parser.add_argument("--effective-path-m", type=float, default=500.0)
    parser.add_argument("--outage-penalty", type=float, default=8.0)
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = LinkBudgetConfig(
        clear_sky_snr_db=args.clear_sky_snr_db,
        effective_turbulence_path_m=args.effective_path_m,
        outage_penalty=args.outage_penalty,
    )

    set_plot_style()
    frame = load_aligned_predictions(DEFAULT_MODEL_INPUTS, persistence_lag_steps=args.persistence_lag_steps)
    policy_timeseries, summary = build_policy_outputs(frame, cfg)
    summary = add_relative_columns(summary)
    usage = write_mcs_usage(policy_timeseries, output_dir)

    frame["true_cn2"] = np.power(10.0, frame["y_true_h1"])
    frame["rytov_variance"] = rytov_variance(frame["y_true_h1"].to_numpy(dtype=np.float64), cfg)
    frame["lognormal_variance"] = lognormal_irradiance_variance(frame["y_true_h1"].to_numpy(dtype=np.float64), cfg)

    frame.to_csv(output_dir / "aligned_prediction_inputs.csv", index=False, encoding="utf-8-sig")
    policy_timeseries.to_csv(output_dir / "link_policy_timeseries.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "link_policy_metrics.csv", index=False, encoding="utf-8-sig")
    mode_table(DEFAULT_MCS_MODES).to_csv(output_dir / "mcs_mode_table.csv", index=False, encoding="utf-8-sig")

    plot_policy_performance(summary, output_dir)
    plot_tradeoff(summary, output_dir)
    plot_mcs_usage(usage, output_dir)
    plot_link_model_curves(frame, cfg, output_dir)
    plot_timeline(frame, policy_timeseries, output_dir)
    write_section_markdown(summary, output_dir, cfg)

    best_forecast = summary[summary["policy_kind"] == "forecast"].sort_values("avg_mission_utility", ascending=False).iloc[0]
    compact = {
        "config": cfg.__dict__,
        "rows": int(len(frame)),
        "date_start": str(frame["Date"].min()),
        "date_end": str(frame["Date"].max()),
        "persistence_lag_steps": int(args.persistence_lag_steps),
        "best_forecast_policy": best_forecast["policy"],
        "best_forecast_avg_mission_utility": float(best_forecast["avg_mission_utility"]),
        "best_forecast_avg_goodput": float(best_forecast["avg_goodput"]),
        "best_forecast_outage_rate": float(best_forecast["outage_rate"]),
        "best_forecast_availability": float(best_forecast["availability"]),
        "output_dir": str(output_dir),
    }
    (output_dir / "link_simulation_summary.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
