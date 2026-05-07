# 星地光通信链路自适应仿真补充材料

本节将湍流预测结果进一步接入简化星地光通信链路模型。设预测目标为 $y=\log_{10}(C_n^2)$，则 $C_n^2=10^y$。在波长 $\lambda=1.55e-06$ m、等效湍流路径长度 $L_{eff}=500$ m 条件下，采用均匀路径近似的 Rytov 方差：

$$
\sigma_R^2=1.23 C_n^2 k^{7/6} L_{eff}^{11/6},\quad k=\frac{2\pi}{\lambda} .
$$

对数正态闪烁模型将归一化接收光强写为 $I=\exp(X)$，其中 $X\sim\mathcal{N}(-\sigma_X^2/2,\sigma_X^2)$，并令 $\sigma_X^2=\ln(1+\sigma_I^2)$。当清空信噪比为 $\gamma_0=20.0$ dB，某一调制编码模式的门限为 $\gamma_m$ 时，中断概率为：

$$
P_{out}(m|y)=\Pr\{\gamma_0+10\log_{10}I<\gamma_m\}.
$$

本文不再只做风险分类，而是令预测值直接驱动调制编码模式选择。对每个候选 MCS，定义预测链路效用：

$$
U(m|\hat y)=\eta_m\left[1-P_{out}(m|\hat y)\right]-\alpha P_{out}(m|\hat y),
$$

其中 $\eta_m$ 为模式频谱效率，$\alpha=8.0$ 为中断惩罚系数。策略选择使 $U(m|\hat y)$ 最大的模式，并使用真实未来 $y$ 评价实际有效吞吐、中断率、可用率和任务效用。

## 仿真结果

| 策略 | 平均效用 | 平均有效吞吐 | 中断率 | 可用率 | 切换率 |
|---|---:|---:|---:|---:|---:|
| Oracle Policy | 2.3811 | 2.5490 | 2.10% | 97.90% | 2.14% |
| Improved LSTM Policy | 2.3627 | 2.5464 | 2.30% | 97.70% | 2.20% |
| TCN Policy | 2.3622 | 2.5325 | 2.13% | 97.87% | 3.07% |
| Persistence Policy | 2.3513 | 2.5393 | 2.35% | 97.65% | 2.14% |
| LSTM-raw Policy | 2.3453 | 2.5883 | 3.04% | 96.96% | 3.63% |
| Fixed High-Rate | 2.0927 | 2.7525 | 8.25% | 91.75% | 0.00% |
| Fixed Balanced | 1.7020 | 1.7820 | 1.00% | 99.00% | 0.00% |
| Fixed Conservative | 0.9893 | 0.9988 | 0.12% | 99.88% | 0.00% |
| Fixed Peak-Rate | -0.7479 | 2.4966 | 40.56% | 59.44% | 0.00% |

结果表明，`Improved LSTM Policy` 在预测驱动策略中取得最高平均任务效用，优于 `LSTM-raw Policy`、`TCN Policy` 和 `Persistence Policy`。虽然 `LSTM-raw Policy` 会因为更激进的模式选择获得较高瞬时吞吐，但其平均中断率也更高；在星地通信场景中，中断通常比吞吐下降更难接受，因此综合效用更能反映实际链路价值。`Oracle Policy` 使用真实未来湍流作为决策输入，可视为理论上限。

生成图件：

- `link_policy_performance.png/pdf`：不同策略的效用、有效吞吐、中断率和切换率。
- `throughput_outage_tradeoff.png/pdf`：吞吐与可靠性折中关系。
- `mcs_usage_heatmap.png/pdf`：不同策略的调制编码模式使用比例。
- `turbulence_to_link_curves.png/pdf`：由 $\log_{10}(C_n^2)$ 到中断概率的链路响应曲线。
- `mcs_timeline_example.png/pdf`：预测驱动 MCS 选择的时间序列示例。
