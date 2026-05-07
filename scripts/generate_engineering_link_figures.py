from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


POLICY_ORDER = [
    "Fixed Conservative",
    "Fixed Balanced",
    "Fixed High-Rate",
    "Persistence Policy",
    "LSTM-raw Policy",
    "TCN Policy",
    "TFT Policy",
    "PatchTST Policy",
    "Improved LSTM Policy",
    "Oracle Policy",
]

FORECAST_ORDER = [
    "Persistence Policy",
    "LSTM-raw Policy",
    "TCN Policy",
    "TFT Policy",
    "PatchTST Policy",
    "Improved LSTM Policy",
    "Oracle Policy",
]

SHORT_LABELS = {
    "Fixed Conservative": "Fixed conservative",
    "Fixed Balanced": "Fixed balanced",
    "Fixed High-Rate": "Fixed high-rate",
    "Persistence Policy": "Persistence",
    "LSTM-raw Policy": "LSTM-raw",
    "TCN Policy": "TCN",
    "TFT Policy": "TFT",
    "PatchTST Policy": "PatchTST",
    "Improved LSTM Policy": "Improved LSTM",
    "Oracle Policy": "Oracle",
}

COLORS = {
    "Fixed Conservative": "#C7C7C7",
    "Fixed Balanced": "#A9A9A9",
    "Fixed High-Rate": "#777777",
    "Persistence Policy": "#8FB3D9",
    "LSTM-raw Policy": "#6B9AC4",
    "TCN Policy": "#2F6F9F",
    "TFT Policy": "#7E6BB7",
    "PatchTST Policy": "#9E78A8",
    "Improved LSTM Policy": "#E07A5F",
    "Oracle Policy": "#59A14F",
}

MCS_COLORS = {
    "MCS0-Survival": "#476A6F",
    "MCS1-Robust": "#6B9080",
    "MCS2-Balanced": "#E9C46A",
    "MCS3-HighRate": "#F4A261",
    "MCS4-Peak": "#E76F51",
}


def set_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "axes.edgecolor": "#253241",
            "axes.labelcolor": "#253241",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "grid.color": "#D8DEE8",
            "grid.linewidth": 0.55,
            "figure.dpi": 170,
            "savefig.dpi": 340,
        },
    )


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def pick_pass(timeseries: pd.DataFrame) -> int:
    improved = timeseries[timeseries["policy"] == "Improved LSTM Policy"].copy()
    grouped = improved.groupby("pass_id").agg(
        rows=("row_index", "size"),
        outage_span=("outage_rate", lambda x: float(x.max() - x.min())),
        mcs_span=("selected_mcs_index", lambda x: int(x.max() - x.min())),
        elevation_span=("elevation_deg", lambda x: float(x.max() - x.min())),
    )
    grouped = grouped[grouped["rows"] >= 24].copy()
    grouped["score"] = grouped["outage_span"] * 4.0 + grouped["mcs_span"] + grouped["elevation_span"] / 90.0
    return int(grouped.sort_values("score", ascending=False).index[0])


def plot_pass_timeline(timeseries: pd.DataFrame, output_dir: Path) -> None:
    pass_id = pick_pass(timeseries)
    policies = ["Persistence Policy", "LSTM-raw Policy", "TCN Policy", "Improved LSTM Policy", "Oracle Policy"]
    df = timeseries[(timeseries["pass_id"] == pass_id) & (timeseries["policy"].isin(policies))].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    base = df[df["policy"] == "Improved LSTM Policy"].sort_values("Date")

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11.2, 7.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.15, 1.15]},
    )
    axes[0].plot(base["Date"], base["elevation_deg"], color="#2F6F9F", linewidth=2.2)
    axes[0].fill_between(base["Date"], base["elevation_deg"], color="#2F6F9F", alpha=0.12)
    axes[0].set_ylabel("Elevation (deg)")

    axes[1].plot(base["Date"], base["y_true_h1"], color="#253241", label="True future", linewidth=2.0)
    axes[1].plot(base["Date"], base["Improved LSTM Policy"], color=COLORS["Improved LSTM Policy"], label="Improved LSTM prediction", linewidth=1.8)
    axes[1].plot(base["Date"], base["TCN Policy"], color=COLORS["TCN Policy"], label="TCN prediction", linewidth=1.35, alpha=0.9)
    axes[1].set_ylabel(r"$\log_{10}(C_n^2)$")
    axes[1].legend(ncol=3, loc="upper left", frameon=True)

    for policy in policies:
        cur = df[df["policy"] == policy].sort_values("Date")
        axes[2].step(
            cur["Date"],
            cur["selected_mcs_index"],
            where="mid",
            label=SHORT_LABELS[policy],
            color=COLORS[policy],
            linewidth=1.65,
        )
    axes[2].set_ylabel("MCS index")
    axes[2].set_yticks([0, 1, 2, 3, 4])
    axes[2].legend(ncol=5, loc="upper left", frameon=True)

    for policy in policies:
        cur = df[df["policy"] == policy].sort_values("Date")
        axes[3].plot(
            cur["Date"],
            cur["outage_rate"],
            label=SHORT_LABELS[policy],
            color=COLORS[policy],
            linewidth=1.55,
        )
    axes[3].set_ylabel("Outage rate")
    axes[3].yaxis.set_major_formatter(lambda x, pos: f"{100*x:.0f}%")
    axes[3].set_xlabel("Time")
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    for ax in axes:
        ax.grid(axis="y", alpha=0.75)
    save(fig, output_dir, "fig_engineering_pass_timeline")


def plot_tradeoff(summary: pd.DataFrame, output_dir: Path) -> None:
    df = summary[summary["policy"].isin(POLICY_ORDER)].copy()
    df["label"] = df["policy"].map(SHORT_LABELS)
    df["outage_pct"] = 100.0 * df["outage_rate"]
    df["bler_pct"] = 100.0 * df["avg_bler"]
    sizes = 90 + 260 * (df["avg_mission_utility"] - df["avg_mission_utility"].min()) / (
        df["avg_mission_utility"].max() - df["avg_mission_utility"].min()
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), gridspec_kw={"width_ratios": [1.18, 1.0]})

    def draw_points(ax: plt.Axes, frame: pd.DataFrame, *, annotate: bool, annotate_fixed_only: bool = False) -> None:
        for _, row in frame.iterrows():
            ax.scatter(
                row["outage_pct"],
                row["avg_goodput"],
                s=float(sizes.loc[row.name]),
                color=COLORS.get(row["policy"], "#999999"),
                edgecolor="white",
                linewidth=1.0,
                alpha=0.95,
                zorder=3,
            )
        if not annotate:
            return
        offsets = {
            "Fixed Conservative": (6, 4),
            "Fixed Balanced": (6, 5),
            "Fixed High-Rate": (6, 6),
            "Persistence Policy": (-58, -2),
            "LSTM-raw Policy": (8, -32),
            "TCN Policy": (8, 34),
            "TFT Policy": (8, -42),
            "PatchTST Policy": (-58, 8),
            "Improved LSTM Policy": (8, 8),
            "Oracle Policy": (8, 34),
        }
        for _, row in frame.iterrows():
            if annotate_fixed_only and not row["policy"].startswith("Fixed"):
                continue
            dx, dy = offsets.get(row["policy"], (5, 5))
            ax.annotate(
                row["label"],
                xy=(row["outage_pct"], row["avg_goodput"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.2,
                ha="left" if dx >= 0 else "right",
                va="center",
            )
        ax.axvline(
            df.loc[df["policy"] == "Improved LSTM Policy", "outage_pct"].iloc[0],
            color=COLORS["Improved LSTM Policy"],
            linestyle="--",
            linewidth=0.9,
            alpha=0.6,
        )
        ax.axhline(
            df.loc[df["policy"] == "Improved LSTM Policy", "avg_goodput"].iloc[0],
            color=COLORS["Improved LSTM Policy"],
            linestyle="--",
            linewidth=0.9,
            alpha=0.6,
        )
        ax.grid(True, alpha=0.75)

    draw_points(axes[0], df, annotate=True, annotate_fixed_only=True)
    axes[0].set_title("(a) All strategies")
    axes[0].set_xlabel("Mean outage rate (%)")
    axes[0].set_ylabel("Expected goodput (bit/s/Hz)")
    axes[0].set_xlim(left=max(0, df["outage_pct"].min() - 1.0), right=df["outage_pct"].max() + 3.0)
    cluster = df[df["policy"].isin(FORECAST_ORDER)]
    axes[0].annotate(
        "Forecast-driven strategies",
        xy=(cluster["outage_pct"].mean(), cluster["avg_goodput"].mean()),
        xytext=(38, 16),
        textcoords="offset points",
        fontsize=7.4,
        arrowprops={"arrowstyle": "-", "color": "#253241", "linewidth": 0.7},
    )

    forecast_df = df[df["policy"].isin(FORECAST_ORDER)].copy()
    draw_points(axes[1], forecast_df, annotate=True)
    axes[1].set_title("(b) Forecast-driven strategies")
    axes[1].set_xlabel("Mean outage rate (%)")
    axes[1].set_ylabel("")
    axes[1].set_xlim(forecast_df["outage_pct"].min() - 0.18, forecast_df["outage_pct"].max() + 0.34)
    axes[1].set_ylim(forecast_df["avg_goodput"].min() - 0.025, forecast_df["avg_goodput"].max() + 0.035)

    save(fig, output_dir, "fig_engineering_tradeoff_scatter")


def plot_mcs_usage(usage: pd.DataFrame, output_dir: Path) -> None:
    policies = ["Persistence Policy", "LSTM-raw Policy", "TCN Policy", "TFT Policy", "PatchTST Policy", "Improved LSTM Policy", "Oracle Policy"]
    mcs_order = list(MCS_COLORS)
    pivot = (
        usage[usage["policy"].isin(policies)]
        .pivot(index="policy", columns="selected_mcs", values="share")
        .reindex(policies)
        .fillna(0.0)
        .reindex(columns=mcs_order)
    )
    labels = [SHORT_LABELS[p] for p in pivot.index]
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))
    for mcs in mcs_order:
        values = pivot[mcs].to_numpy()
        ax.barh(y, values, left=left, color=MCS_COLORS[mcs], edgecolor="white", height=0.72, label=mcs.replace("MCS", "MCS "))
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, pos: f"{100*x:.0f}%")
    ax.set_xlabel("Share of selected MCS")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=True)

    for row_idx, (_, row) in enumerate(pivot.iterrows()):
        cumulative = 0.0
        for mcs in mcs_order:
            value = float(row[mcs])
            if value >= 0.08:
                ax.text(cumulative + value / 2.0, row_idx, f"{100*value:.0f}%", va="center", ha="center", fontsize=7, color="#222222")
            cumulative += value
    save(fig, output_dir, "fig_engineering_mcs_usage")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-style figures for engineering link Monte Carlo results.")
    parser.add_argument("--input-dir", default="outputs/engineering_link_monte_carlo_20260428")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir
    summary = pd.read_csv(input_dir / "engineering_policy_summary.csv")
    timeseries = pd.read_csv(input_dir / "engineering_policy_timeseries.csv")
    usage = pd.read_csv(input_dir / "engineering_mcs_usage.csv")
    set_style()
    plot_pass_timeline(timeseries, input_dir)
    plot_tradeoff(summary, input_dir)
    plot_mcs_usage(usage, input_dir)
    print(f"Saved engineering figures to {input_dir}")


if __name__ == "__main__":
    main()
