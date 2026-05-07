from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "current_stage_link_simulation_20260501"

BER_DIR = ROOT / "outputs" / "ber_mcs_link_enhancement_20260427"
ROBUST_DIR = ROOT / "outputs" / "review_response_improvements_20260426"
ENG_DIR = ROOT / "outputs" / "engineering_link_monte_carlo_20260428"


POLICY_CN = {
    "Oracle Policy": "理想策略（Oracle Policy）",
    "Improved LSTM Policy": "改进 LSTM 预测驱动策略（Improved LSTM Policy）",
    "TCN Policy": "时间卷积网络预测驱动策略（TCN Policy）",
    "TFT Policy": "时间融合 Transformer 预测驱动策略（TFT Policy）",
    "PatchTST Policy": "分块时间序列 Transformer 预测驱动策略（PatchTST Policy）",
    "Persistence Policy": "持续性预测策略（Persistence Policy）",
    "LSTM-raw Policy": "基础 LSTM 预测驱动策略（LSTM-raw Policy）",
    "Fixed High-Rate": "固定高速率策略（Fixed High-Rate）",
    "Fixed Balanced": "固定均衡策略（Fixed Balanced）",
    "Fixed Conservative": "固定保守策略（Fixed Conservative）",
    "Fixed Survival": "固定生存策略（Fixed Survival）",
    "Fixed Peak-Rate": "固定峰值速率策略（Fixed Peak-Rate）",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def pct(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def fixed(value: float, digits: int = 5) -> str:
    return f"{float(value):.{digits}f}"


def as_policy_table(df: pd.DataFrame, policies: list[str]) -> pd.DataFrame:
    out = df[df["policy"].isin(policies)].copy()
    out["策略"] = out["policy"].map(POLICY_CN).fillna(out["policy"])
    keep = [
        "策略",
        "avg_mission_utility",
        "avg_goodput",
        "outage_rate",
        "availability",
        "avg_coded_ber_proxy",
        "p99_coded_ber_proxy",
        "ber_proxy_exceed_1e_4_rate",
    ]
    if "avg_coded_ber_proxy" not in out.columns:
        keep = [
            "策略",
            "avg_mission_utility",
            "avg_goodput",
            "outage_rate",
            "availability",
            "avg_coded_ber",
            "p99_coded_ber",
            "avg_bler",
        ]
    return out[keep]


def markdown_table(df: pd.DataFrame, headers: dict[str, str]) -> str:
    cols = list(headers.keys())
    lines = ["| " + " | ".join(headers.values()) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if col in {"outage_rate", "availability", "ber_proxy_exceed_1e_4_rate", "avg_bler", "mean_outage", "share"}:
                vals.append(pct(val))
            elif isinstance(val, float):
                if "ber" in col:
                    vals.append(f"{val:.3e}")
                else:
                    vals.append(fixed(val))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    main_link = read_csv(BER_DIR / "main_site_ber_mcs_policy_summary.csv")
    cross_link = read_csv(BER_DIR / "cross_site_ber_mcs_policy_summary.csv")
    mcs = read_csv(BER_DIR / "ber_mcs_mode_assumptions.csv")
    pair_boot = read_csv(ROBUST_DIR / "link_policy_pairwise_bootstrap.csv")
    sensitivity = read_csv(ROBUST_DIR / "link_sensitivity_rank_stability.csv")
    eng_policy = read_csv(ENG_DIR / "engineering_policy_summary.csv")
    eng_boot = read_csv(ENG_DIR / "engineering_policy_bootstrap_delta.csv")
    eng_mcs = read_csv(ENG_DIR / "engineering_mcs_usage.csv")
    eng_summary = json.loads((ENG_DIR / "engineering_link_monte_carlo_summary.json").read_text(encoding="utf-8"))

    main_policies = [
        "Oracle Policy",
        "Improved LSTM Policy",
        "TCN Policy",
        "TFT Policy",
        "PatchTST Policy",
        "Persistence Policy",
        "LSTM-raw Policy",
        "Fixed High-Rate",
        "Fixed Balanced",
        "Fixed Conservative",
        "Fixed Peak-Rate",
    ]
    cross_policies = [
        "Oracle Policy",
        "Improved LSTM Policy",
        "Persistence Policy",
        "LSTM-raw Policy",
        "Fixed High-Rate",
        "Fixed Balanced",
        "Fixed Conservative",
        "Fixed Peak-Rate",
    ]
    engineering_policies = [
        "Oracle Policy",
        "Improved LSTM Policy",
        "TCN Policy",
        "LSTM-raw Policy",
        "Persistence Policy",
        "Fixed High-Rate",
        "Fixed Balanced",
        "Fixed Conservative",
        "Fixed Peak-Rate",
    ]

    main_table = as_policy_table(main_link, main_policies)
    cross_table = as_policy_table(cross_link, cross_policies)
    eng_table = as_policy_table(eng_policy, engineering_policies)

    main_table.to_csv(OUT_DIR / "current_stage_main_site_link_table.csv", index=False, encoding="utf-8-sig")
    cross_table.to_csv(OUT_DIR / "current_stage_cross_site_link_table.csv", index=False, encoding="utf-8-sig")
    eng_table.to_csv(OUT_DIR / "current_stage_engineering_random_link_table.csv", index=False, encoding="utf-8-sig")
    mcs.to_csv(OUT_DIR / "current_stage_mcs_assumptions.csv", index=False, encoding="utf-8-sig")
    pair_boot.to_csv(OUT_DIR / "current_stage_default_bootstrap.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(OUT_DIR / "current_stage_sensitivity_rank.csv", index=False, encoding="utf-8-sig")
    eng_boot.to_csv(OUT_DIR / "current_stage_engineering_bootstrap.csv", index=False, encoding="utf-8-sig")
    eng_mcs.to_csv(OUT_DIR / "current_stage_engineering_mcs_usage.csv", index=False, encoding="utf-8-sig")

    imp = main_link[main_link["policy"] == "Improved LSTM Policy"].iloc[0]
    oracle = main_link[main_link["policy"] == "Oracle Policy"].iloc[0]
    fixed_high = main_link[main_link["policy"] == "Fixed High-Rate"].iloc[0]
    eng_imp = eng_policy[eng_policy["policy"] == "Improved LSTM Policy"].iloc[0]
    eng_oracle = eng_policy[eng_policy["policy"] == "Oracle Policy"].iloc[0]
    eng_high = eng_policy[eng_policy["policy"] == "Fixed High-Rate"].iloc[0]

    headers_default = {
        "策略": "策略",
        "avg_mission_utility": "平均任务效用",
        "avg_goodput": "平均吞吐量",
        "outage_rate": "中断率",
        "availability": "可用率",
        "avg_coded_ber_proxy": "平均 BER proxy",
        "p99_coded_ber_proxy": "P99 BER proxy",
        "ber_proxy_exceed_1e_4_rate": "BER proxy>1e-4",
    }
    headers_engineering = {
        "策略": "策略",
        "avg_mission_utility": "平均任务效用",
        "avg_goodput": "平均吞吐量",
        "outage_rate": "中断率",
        "availability": "可用率",
        "avg_coded_ber": "平均编码 BER",
        "p99_coded_ber": "P99 编码 BER",
        "avg_bler": "平均 BLER",
    }

    mcs_display = mcs.rename(
        columns={
            "name": "模式",
            "modulation": "调制方式",
            "code_rate": "码率",
            "spectral_efficiency": "频谱效率",
            "required_snr_db": "所需信噪比/dB",
            "coding_gain_db_used_for_ber_proxy": "BER proxy编码增益/dB",
        }
    )[["模式", "调制方式", "码率", "频谱效率", "所需信噪比/dB", "BER proxy编码增益/dB"]]

    md = []
    md.append("# 现有阶段星地通信链路级仿真汇总\n\n")
    md.append(f"生成时间：{datetime.now().isoformat(timespec='seconds')}\n\n")
    md.append("## 一、仿真闭环\n\n")
    md.append(
        "现有阶段已经形成从湍流预测到链路自适应评价的闭环：神经网络输出未来 30 min 平均 `log10(Cn2)`，经物理尺度转换得到 `Cn2`，再根据链路裕量、调制编码方式和误码率代理计算任务效用、吞吐量、中断率和可用率。该闭环用于比较不同预测策略对链路自适应决策的影响。\n\n"
    )
    md.append("边界说明建议只在第 6 章方法部分出现一次：当前结果属于链路级仿真验证，不等同于真实星地激光通信外场实测。\n\n")
    md.append("## 二、默认 BER/MCS 链路仿真\n\n")
    md.append(
        f"主站点默认设置下，改进 LSTM 预测驱动策略的平均任务效用为 {fixed(imp['avg_mission_utility'])}，接近理想策略的 {fixed(oracle['avg_mission_utility'])}；相较固定高速率策略，中断率由 {pct(fixed_high['outage_rate'])} 降至 {pct(imp['outage_rate'])}，可用率由 {pct(fixed_high['availability'])} 提升至 {pct(imp['availability'])}。\n\n"
    )
    md.append(markdown_table(main_table, headers_default))
    md.append("\n\n## 三、MCS 模式参数\n\n")
    md.append(markdown_table(mcs_display, {col: col for col in mcs_display.columns}))
    md.append("\n\n## 四、跨站点链路补充验证\n\n")
    md.append(markdown_table(cross_table, headers_default))
    md.append("\n\n## 五、bootstrap 与敏感性分析\n\n")
    boot_display = pair_boot.copy()
    boot_display["指标"] = "mission_utility"
    boot_display["效用或中断率变化"] = boot_display["mission_utility_gain_mean"]
    boot_display["95%置信区间"] = boot_display.apply(
        lambda r: f"[{fixed(r['mission_utility_gain_ci95_low'])}, {fixed(r['mission_utility_gain_ci95_high'])}]",
        axis=1,
    )
    boot_display["正向概率"] = boot_display["mission_utility_gain_bootstrap_prob_positive"]
    boot_display = boot_display[["comparison", "指标", "效用或中断率变化", "95%置信区间", "正向概率"]]
    md.append(markdown_table(
        boot_display.rename(columns={"comparison": "比较对象"}),
        {"比较对象": "比较对象", "指标": "指标", "效用或中断率变化": "变化均值", "95%置信区间": "95%置信区间", "正向概率": "正向概率"},
    ))
    md.append("\n\n")
    sens = sensitivity.copy()
    sens["策略"] = sens["policy"].map(POLICY_CN).fillna(sens["policy"])
    sens = sens[["策略", "mean_rank", "median_rank", "best_rank_count", "mean_utility", "mean_outage"]]
    md.append(markdown_table(
        sens,
        {"策略": "策略", "mean_rank": "平均排名", "median_rank": "中位排名", "best_rank_count": "第一名次数", "mean_utility": "平均效用", "mean_outage": "平均中断率"},
    ))
    md.append("\n\n## 六、随机信道补充仿真\n\n")
    cfg = eng_summary["config"]
    md.append(
        f"随机信道补充仿真加入了过境几何、仰角变化、指向抖动、路径损耗和块错误率（Block Error Rate，BLER）评价。主要配置包括轨道高度 {cfg['altitude_m']:.0f} m，最小仰角 {cfg['min_elevation_deg']:.1f}°，最大仰角 {cfg['max_elevation_deg']:.1f}°，指向抖动 {cfg['pointing_jitter_urad']:.1f} μrad，波束发散角 {cfg['beam_divergence_urad']:.1f} μrad。\n\n"
    )
    md.append(
        f"在该补充仿真下，改进 LSTM 预测驱动策略的平均任务效用为 {fixed(eng_imp['avg_mission_utility'])}，接近理想策略的 {fixed(eng_oracle['avg_mission_utility'])}，明显高于固定高速率策略的 {fixed(eng_high['avg_mission_utility'])}；固定高速率策略虽然平均吞吐量较高，但中断率达到 {pct(eng_high['outage_rate'])}。\n\n"
    )
    md.append(markdown_table(eng_table, headers_engineering))
    md.append("\n\n## 七、随机信道补充仿真的 MCS 使用比例\n\n")
    eng_mcs_display = eng_mcs.copy()
    eng_mcs_display["策略"] = eng_mcs_display["policy"].map(POLICY_CN).fillna(eng_mcs_display["policy"])
    eng_mcs_display = eng_mcs_display[["策略", "selected_mcs", "count", "share"]]
    md.append(markdown_table(eng_mcs_display, {"策略": "策略", "selected_mcs": "MCS", "count": "次数", "share": "比例"}))
    md.append("\n\n## 八、建议图表\n\n")
    for path in [
        "outputs/ber_mcs_link_enhancement_20260427/main_site_ber_mcs_policy_summary.csv",
        "outputs/review_response_improvements_20260426/link_sensitivity_rank_stability.png",
        "outputs/engineering_link_monte_carlo_20260428/fig_engineering_tradeoff_scatter.png",
        "outputs/engineering_link_monte_carlo_20260428/fig_engineering_mcs_usage.png",
        "outputs/engineering_link_monte_carlo_20260428/fig_engineering_pass_timeline.png",
    ]:
        md.append(f"- `{path}`\n")

    (OUT_DIR / "current_stage_link_simulation_summary.md").write_text("".join(md), encoding="utf-8-sig")
    (OUT_DIR / "current_stage_link_simulation_summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(OUT_DIR),
                "main_improved_utility": float(imp["avg_mission_utility"]),
                "main_improved_outage_rate": float(imp["outage_rate"]),
                "engineering_improved_utility": float(eng_imp["avg_mission_utility"]),
                "engineering_improved_outage_rate": float(eng_imp["outage_rate"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote current-stage link simulation summary to {OUT_DIR}")
    for path in sorted(OUT_DIR.iterdir()):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
