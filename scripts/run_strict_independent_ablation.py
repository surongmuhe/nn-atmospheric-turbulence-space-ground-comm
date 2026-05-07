from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CFG_PATH = ROOT / "configs" / "thesis_lstm_final_best_seq72.yaml"
DEFAULT_SEEDS = [42]


RAW_FEATURES = ["LogCn2", "IRtemp", "SolarFlux", "H2Ocon", "Tair", "UmSonic", "Pair"]
TIME_FEATURES = RAW_FEATURES + ["hour_sin", "hour_cos"]


VARIANT_META = {
    "full_improved": {
        "cn_name": "完整改进 LSTM 模型",
        "category": "基准模型",
        "single_change": "不改变配置，作为严格消融参照组。",
        "paper_role": "正文主参照组",
        "primary_model": "lstm",
    },
    "no_future_mean_target": {
        "cn_name": "去除未来均值目标",
        "category": "任务定义消融",
        "single_change": "仅将目标聚合方式由未来 30 min 均值改为 6 步多步预测。",
        "paper_role": "检验未来均值目标对短时平均湍流预测的作用。",
        "primary_model": "lstm",
    },
    "no_delta_target": {
        "cn_name": "去除增量预测目标",
        "category": "任务定义消融",
        "single_change": "仅将目标形式由增量预测改为绝对值预测。",
        "paper_role": "检验增量预测对站点偏置和短时变化建模的作用。",
        "primary_model": "lstm",
    },
    "no_time_cycle_features": {
        "cn_name": "去除时间周期特征",
        "category": "输入特征消融",
        "single_change": "仅移除 hour_sin 和 hour_cos，并关闭小时周期特征开关。",
        "paper_role": "检验昼夜周期信息对湍流预测的作用。",
        "primary_model": "lstm",
    },
    "no_domain_physics_features": {
        "cn_name": "去除物理派生特征",
        "category": "输入特征消融",
        "single_change": "仅关闭 domain_features，其余时间、差分、滚动统计设置保持不变。",
        "paper_role": "检验温差、太阳辐射非负化、风速平方等物理派生量的作用。",
        "primary_model": "lstm",
    },
    "no_solar_regime_features": {
        "cn_name": "去除太阳状态特征",
        "category": "输入特征消融",
        "single_change": "仅关闭 solar_regime_features。",
        "paper_role": "检验夜间、晨昏、强日照和太阳辐射跃迁信息的作用。",
        "primary_model": "lstm",
    },
    "no_diff_features": {
        "cn_name": "去除一阶差分特征",
        "category": "输入特征消融",
        "single_change": "仅关闭 diff_features。",
        "paper_role": "检验短时变化率特征对预测的作用。",
        "primary_model": "lstm",
    },
    "no_rolling_features": {
        "cn_name": "去除滚动统计特征",
        "category": "输入特征消融",
        "single_change": "仅关闭 rolling_features。",
        "paper_role": "检验短窗均值和标准差统计对预测的作用。",
        "primary_model": "lstm",
    },
    "no_bidirectional_encoder": {
        "cn_name": "去除双向编码",
        "category": "模型结构消融",
        "single_change": "仅将 LSTM 编码器由双向改为单向。",
        "paper_role": "检验双向历史上下文编码的作用。",
        "primary_model": "lstm",
    },
    "no_attention_pooling": {
        "cn_name": "去除注意力池化",
        "category": "模型结构消融",
        "single_change": "仅关闭 attention_pool。",
        "paper_role": "检验注意力池化对历史窗口信息聚合的作用。",
        "primary_model": "lstm",
    },
    "no_short_ar_branch": {
        "cn_name": "去除短窗自回归支路",
        "category": "模型结构消融",
        "single_change": "仅将 ar_window 由 12 改为 0。",
        "paper_role": "检验短窗目标历史支路对局部惯性的刻画作用。",
        "primary_model": "lstm",
    },
    "no_input_projection": {
        "cn_name": "去除输入投影层",
        "category": "模型结构消融",
        "single_change": "仅将 input_proj_size 由 64 改为 None。",
        "paper_role": "检验输入特征投影压缩对表示学习的作用。",
        "primary_model": "lstm",
    },
    "vanilla_lstm_same_features": {
        "cn_name": "基础 LSTM 结构对照",
        "category": "模型结构对照",
        "single_change": "仅将模型类由改进 LSTM 换为基础 LSTM，输入特征和任务设置保持一致。",
        "paper_role": "检验改进预测头和增强结构相对基础循环模型的整体作用。",
        "primary_model": "vanilla_lstm",
    },
    "mlp_window_same_features": {
        "cn_name": "窗口多层感知机对照",
        "category": "模型结构对照",
        "single_change": "仅将模型类由改进 LSTM 换为窗口 MLP，输入特征和任务设置保持一致。",
        "paper_role": "检验显式时序建模相对窗口展开前馈网络的作用。",
        "primary_model": "mlp",
    },
}


DEFAULT_VARIANTS = list(VARIANT_META.keys())


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def set_nested(cfg: dict, path: list[str], value) -> None:
    cur = cfg
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def apply_variant(base_cfg: dict, variant: str, seed: int, output_dir: Path) -> dict:
    cfg = deepcopy(base_cfg)
    cfg["project"]["name"] = f"strict_ablation_{variant}_seed{seed}"
    cfg["project"]["seed"] = int(seed)
    cfg["project"]["output_dir"] = str(output_dir)
    cfg["train"]["show_progress"] = False
    cfg["experiment"]["primary_model"] = VARIANT_META[variant]["primary_model"]
    cfg["experiment"]["active_models"] = [VARIANT_META[variant]["primary_model"]]

    if variant == "full_improved":
        return cfg

    if variant == "no_future_mean_target":
        set_nested(cfg, ["data", "target_aggregation"], "multistep")
        return cfg

    if variant == "no_delta_target":
        set_nested(cfg, ["data", "target_mode"], "absolute")
        return cfg

    if variant == "no_time_cycle_features":
        set_nested(cfg, ["data", "feature_cols"], list(RAW_FEATURES))
        set_nested(cfg, ["preprocess", "time_features", "keep_temporal_hour"], False)
        set_nested(cfg, ["preprocess", "time_features", "hour_cyclical"], False)
        return cfg

    if variant == "no_domain_physics_features":
        set_nested(cfg, ["preprocess", "domain_features", "enabled"], False)
        return cfg

    if variant == "no_solar_regime_features":
        set_nested(cfg, ["preprocess", "solar_regime_features", "enabled"], False)
        return cfg

    if variant == "no_diff_features":
        set_nested(cfg, ["preprocess", "diff_features", "enabled"], False)
        return cfg

    if variant == "no_rolling_features":
        set_nested(cfg, ["preprocess", "rolling_features", "enabled"], False)
        return cfg

    if variant == "no_bidirectional_encoder":
        set_nested(cfg, ["models", "lstm", "bidirectional"], False)
        return cfg

    if variant == "no_attention_pooling":
        set_nested(cfg, ["models", "lstm", "attention_pool"], False)
        return cfg

    if variant == "no_short_ar_branch":
        set_nested(cfg, ["models", "lstm", "ar_window"], 0)
        return cfg

    if variant == "no_input_projection":
        set_nested(cfg, ["models", "lstm", "input_proj_size"], None)
        return cfg

    if variant == "vanilla_lstm_same_features":
        cfg["experiment"]["primary_model"] = "vanilla_lstm"
        cfg["experiment"]["active_models"] = ["vanilla_lstm"]
        return cfg

    if variant == "mlp_window_same_features":
        cfg["experiment"]["primary_model"] = "mlp"
        cfg["experiment"]["active_models"] = ["mlp"]
        return cfg

    raise ValueError(f"Unsupported variant: {variant}")


def run_command(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_variant(cfg_path: Path, output_dir: Path, skip_existing: bool) -> None:
    metrics_path = output_dir / "metrics_all_models.csv"
    if skip_existing and metrics_path.exists():
        print(f"SKIP existing: {metrics_path}", flush=True)
        return
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_experiments.py"),
        "--config",
        str(cfg_path),
        "--output-dir",
        str(output_dir),
    ]
    run_command(cmd)


def pick_focus_row(metrics_path: Path, primary_model: str) -> dict:
    metrics = pd.read_csv(metrics_path)
    focus = metrics[metrics["model"] == primary_model]
    if focus.empty:
        focus = metrics[~metrics["model"].astype(str).str.startswith("naive_")]
    if focus.empty:
        raise ValueError(f"No focus model row found in {metrics_path}")
    return focus.sort_values("RMSE", ascending=True).iloc[0].to_dict()


def collect_metrics(output_root: Path, variants: list[str], seeds: list[int]) -> pd.DataFrame:
    rows = []
    for variant in variants:
        meta = VARIANT_META[variant]
        for seed in seeds:
            run_dir = output_root / "runs" / variant / f"seed_{seed}"
            metrics_path = run_dir / "metrics_all_models.csv"
            snapshot_path = run_dir / "config_snapshot.json"
            if not metrics_path.exists():
                continue
            row = pick_focus_row(metrics_path, meta["primary_model"])
            cfg = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.exists() else {}
            row.update(
                {
                    "variant": variant,
                    "中文名称": meta["cn_name"],
                    "消融类别": meta["category"],
                    "单一改动": meta["single_change"],
                    "论文用途": meta["paper_role"],
                    "seed": seed,
                    "primary_model": meta["primary_model"],
                    "seq_len": cfg.get("data", {}).get("seq_len"),
                    "pred_len_cfg": cfg.get("data", {}).get("pred_len"),
                    "target_mode": cfg.get("data", {}).get("target_mode"),
                    "target_aggregation": cfg.get("data", {}).get("target_aggregation"),
                    "feature_count": len(cfg.get("data", {}).get("feature_cols", [])),
                    "effective_feature_count": None,
                    "source": str(metrics_path.relative_to(ROOT)),
                }
            )
            feature_cols_path = run_dir / "dataset_summary.json"
            if feature_cols_path.exists():
                try:
                    ds = json.loads(feature_cols_path.read_text(encoding="utf-8"))
                    row["effective_feature_count"] = ds.get("feature_count")
                except json.JSONDecodeError:
                    pass
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    preferred = [
        "variant",
        "中文名称",
        "消融类别",
        "单一改动",
        "论文用途",
        "seed",
        "primary_model",
        "model",
        "RMSE",
        "MAE",
        "MAPE",
        "R2",
        "seq_len",
        "pred_len_cfg",
        "target_mode",
        "target_aggregation",
        "feature_count",
        "effective_feature_count",
        "source",
    ]
    df = pd.DataFrame(rows)
    ordered = preferred + [col for col in df.columns if col not in preferred]
    return df[ordered].sort_values(["消融类别", "variant", "seed"]).reset_index(drop=True)


def summarize(seed_df: pd.DataFrame) -> pd.DataFrame:
    if seed_df.empty:
        return pd.DataFrame()
    numeric_cols = ["RMSE", "MAE", "MAPE", "R2"]
    grouped = seed_df.groupby(["variant", "中文名称", "消融类别", "单一改动", "论文用途"], dropna=False)
    summary = grouped[numeric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join([str(x) for x in col if str(x)])
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    full = summary[summary["variant"] == "full_improved"].iloc[0]
    summary["相对完整模型RMSE变化"] = summary["RMSE_mean"] - float(full["RMSE_mean"])
    summary["相对完整模型MAE变化"] = summary["MAE_mean"] - float(full["MAE_mean"])
    summary["相对完整模型R2变化"] = summary["R2_mean"] - float(full["R2_mean"])
    summary["RMSE相对变化率"] = summary["相对完整模型RMSE变化"] / float(full["RMSE_mean"])
    summary["结论口径"] = summary.apply(conclusion_text, axis=1)
    order_map = {variant: idx for idx, variant in enumerate(DEFAULT_VARIANTS)}
    summary["order"] = summary["variant"].map(order_map).fillna(999)
    return summary.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def conclusion_text(row: pd.Series) -> str:
    if row["variant"] == "full_improved":
        return "完整模型作为单因素消融参照组。"
    delta = float(row["相对完整模型RMSE变化"])
    if delta > 0.002:
        return "去除该因素后 RMSE 上升，说明该因素对最终模型有正向作用。"
    if delta < -0.002:
        return "去除该因素后 RMSE 下降，说明该因素在当前设置下可能引入冗余或需重新调参。"
    return "去除该因素后 RMSE 变化较小，说明该因素影响有限或与其他模块存在替代关系。"


def fmt(value: float, digits: int = 5) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def markdown_table(df: pd.DataFrame, fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for field in fields:
            val = row.get(field, "")
            if isinstance(val, float):
                if "变化率" in field:
                    vals.append(pct(val))
                else:
                    vals.append(fmt(val))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def make_plots(output_root: Path, summary: pd.DataFrame) -> list[str]:
    paths: list[str] = []
    if summary.empty:
        return paths
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        (output_root / "plot_generation_skipped.txt").write_text(str(exc), encoding="utf-8")
        return paths

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plot_df = summary[summary["variant"] != "full_improved"].copy()
    labels = plot_df["中文名称"].tolist()
    delta_rmse = plot_df["相对完整模型RMSE变化"].astype(float).tolist()
    colors = ["#E15759" if v >= 0 else "#59A14F" for v in delta_rmse]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.bar(range(len(labels)), delta_rmse, color=colors, alpha=0.86)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("相对完整模型 RMSE 变化")
    ax.set_title("严格单因素消融的 RMSE 变化")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = output_root / "strict_ablation_rmse_delta.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    r2_delta = plot_df["相对完整模型R2变化"].astype(float).tolist()
    colors = ["#59A14F" if v >= 0 else "#E15759" for v in r2_delta]
    ax.bar(range(len(labels)), r2_delta, color=colors, alpha=0.86)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("相对完整模型 R2 变化")
    ax.set_title("严格单因素消融的 R2 变化")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = output_root / "strict_ablation_r2_delta.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))
    return paths


def write_markdown(output_root: Path, seed_df: pd.DataFrame, summary: pd.DataFrame, plot_paths: list[str]) -> None:
    lines: list[str] = [
        "# 严格单因素消融实验汇总\n\n",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}\n\n",
        "## 实验原则\n\n",
        "以最终 `seq_len=72` 的改进 LSTM 配置为完整模型，除被考察因素外，其余数据划分、训练参数、随机种子、预测步长和评价指标保持一致。任务定义类消融会改变预测目标本身，因此用于回答目标设计是否必要；输入特征类和模型结构类消融用于回答对应模块的独立作用。\n\n",
        "## 正文建议主表\n\n",
    ]
    fields = [
        "中文名称",
        "消融类别",
        "RMSE_mean",
        "MAE_mean",
        "MAPE_mean",
        "R2_mean",
        "相对完整模型RMSE变化",
        "RMSE相对变化率",
        "结论口径",
    ]
    lines.append(markdown_table(summary, fields))
    lines.append("\n\n## 每个实验的单一改动\n\n")
    lines.append(markdown_table(summary, ["中文名称", "消融类别", "单一改动", "论文用途"]))
    lines.append("\n\n## 各随机种子原始指标\n\n")
    seed_fields = ["variant", "中文名称", "seed", "model", "RMSE", "MAE", "MAPE", "R2", "target_mode", "target_aggregation", "source"]
    lines.append(markdown_table(seed_df, seed_fields))
    lines.append("\n\n## 建议插图\n\n")
    for path in plot_paths:
        lines.append(f"- `{path}`\n")
    (output_root / "strict_ablation_summary.md").write_text("".join(lines), encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict one-factor ablation around the final seq72 Improved LSTM.")
    parser.add_argument("--output-root", type=str, default="outputs/strict_independent_ablation_20260501")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    invalid = [variant for variant in args.variants if variant not in VARIANT_META]
    if invalid:
        raise ValueError(f"Unsupported variants: {invalid}")

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    config_root = output_root / "_configs"
    output_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(BASE_CFG_PATH)
    if not args.collect_only:
        for variant in args.variants:
            for seed in args.seeds:
                run_dir = output_root / "runs" / variant / f"seed_{seed}"
                cfg = apply_variant(base_cfg, variant, seed, run_dir)
                cfg_path = config_root / variant / f"seed_{seed}.yaml"
                write_yaml(cfg_path, cfg)
                run_variant(cfg_path, run_dir, skip_existing=args.skip_existing)

    seed_df = collect_metrics(output_root, args.variants, args.seeds)
    if seed_df.empty:
        raise RuntimeError("No strict ablation metrics were collected.")
    summary = summarize(seed_df)
    seed_df.to_csv(output_root / "strict_ablation_seed_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_root / "strict_ablation_summary.csv", index=False, encoding="utf-8-sig")
    plot_paths = make_plots(output_root, summary)
    write_markdown(output_root, seed_df, summary, plot_paths)

    compact = {
        "output_root": str(output_root),
        "seeds": list(args.seeds),
        "variants": list(args.variants),
        "generated_files": [
            "strict_ablation_seed_metrics.csv",
            "strict_ablation_summary.csv",
            "strict_ablation_summary.md",
            *[Path(path).name for path in plot_paths],
        ],
    }
    (output_root / "strict_ablation_summary.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote strict independent ablation outputs to {output_root}")
    for path in sorted(output_root.glob("*")):
        if path.is_file():
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
