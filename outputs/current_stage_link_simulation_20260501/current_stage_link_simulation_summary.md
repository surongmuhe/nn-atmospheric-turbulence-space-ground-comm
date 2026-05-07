# 现有阶段星地通信链路级仿真汇总

生成时间：2026-05-01T01:30:13

## 一、仿真闭环

现有阶段已经形成从湍流预测到链路自适应评价的闭环：神经网络输出未来 30 min 平均 `log10(Cn2)`，经物理尺度转换得到 `Cn2`，再根据链路裕量、调制编码方式和误码率代理计算任务效用、吞吐量、中断率和可用率。该闭环用于比较不同预测策略对链路自适应决策的影响。

边界说明建议只在第 6 章方法部分出现一次：当前结果属于链路级仿真验证，不等同于真实星地激光通信外场实测。

## 二、默认 BER/MCS 链路仿真

主站点默认设置下，改进 LSTM 预测驱动策略的平均任务效用为 2.36272，接近理想策略的 2.38113；相较固定高速率策略，中断率由 8.25% 降至 2.30%，可用率由 91.75% 提升至 97.70%。

| 策略 | 平均任务效用 | 平均吞吐量 | 中断率 | 可用率 | 平均 BER proxy | P99 BER proxy | BER proxy>1e-4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 理想策略（Oracle Policy） | 2.38113 | 2.54896 | 2.10% | 97.90% | 3.540e-06 | 7.537e-05 | 0.43% |
| 改进 LSTM 预测驱动策略（Improved LSTM Policy） | 2.36272 | 2.54639 | 2.30% | 97.70% | 5.308e-06 | 1.064e-04 | 1.02% |
| 时间卷积网络预测驱动策略（TCN Policy） | 2.36224 | 2.53246 | 2.13% | 97.87% | 5.202e-06 | 1.115e-04 | 1.05% |
| 时间融合 Transformer 预测驱动策略（TFT Policy） | 2.35956 | 2.53923 | 2.25% | 97.75% | 5.000e-06 | 9.646e-05 | 0.99% |
| 分块时间序列 Transformer 预测驱动策略（PatchTST Policy） | 2.35661 | 2.54286 | 2.33% | 97.67% | 5.122e-06 | 9.673e-05 | 0.99% |
| 持续性预测策略（Persistence Policy） | 2.35132 | 2.53929 | 2.35% | 97.65% | 5.409e-06 | 1.134e-04 | 1.12% |
| 基础 LSTM 预测驱动策略（LSTM-raw Policy） | 2.34525 | 2.58827 | 3.04% | 96.96% | 6.214e-06 | 1.200e-04 | 1.24% |
| 固定高速率策略（Fixed High-Rate） | 2.09267 | 2.75255 | 8.25% | 91.75% | 3.069e-05 | 7.710e-04 | 5.83% |
| 固定均衡策略（Fixed Balanced） | 1.70203 | 1.78201 | 1.00% | 99.00% | 7.370e-06 | 2.162e-04 | 1.67% |
| 固定保守策略（Fixed Conservative） | 0.98934 | 0.99882 | 0.12% | 99.88% | 1.730e-06 | 5.173e-05 | 0.43% |
| 固定峰值速率策略（Fixed Peak-Rate） | -0.74786 | 2.49664 | 40.56% | 59.44% | 5.354e-05 | 1.236e-03 | 10.02% |

## 三、MCS 模式参数

| 模式 | 调制方式 | 码率 | 频谱效率 | 所需信噪比/dB | BER proxy编码增益/dB |
| --- | --- | --- | --- | --- | --- |
| MCS0-Survival | BPSK | 1/3 | 0.35000 | 3.00000 | 5.00000 |
| MCS1-Robust | QPSK | 1/2 | 1.00000 | 7.00000 | 3.20000 |
| MCS2-Balanced | QPSK | 0.9 | 1.80000 | 11.00000 | 0.80000 |
| MCS3-HighRate | 16-QAM | 3/4 | 3.00000 | 15.50000 | 1.80000 |
| MCS4-Peak | 16-QAM | 5/6 | 4.20000 | 19.00000 | 0.80000 |

## 四、跨站点链路补充验证

| 策略 | 平均任务效用 | 平均吞吐量 | 中断率 | 可用率 | 平均 BER proxy | P99 BER proxy | BER proxy>1e-4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 理想策略（Oracle Policy） | 2.62748 | 2.82949 | 2.53% | 97.47% | 2.512e-06 | 4.733e-05 | 0.00% |
| 持续性预测策略（Persistence Policy） | 2.53011 | 2.77941 | 3.12% | 96.88% | 2.766e-06 | 5.315e-05 | 0.25% |
| 基础 LSTM 预测驱动策略（LSTM-raw Policy） | 2.52306 | 2.82422 | 3.76% | 96.24% | 3.532e-06 | 6.671e-05 | 0.43% |
| 改进 LSTM 预测驱动策略（Improved LSTM Policy） | 2.50848 | 2.81359 | 3.81% | 96.19% | 3.674e-06 | 6.561e-05 | 0.43% |
| 固定高速率策略（Fixed High-Rate） | 2.20166 | 2.78227 | 7.26% | 92.74% | 1.705e-05 | 2.883e-04 | 5.42% |
| 固定均衡策略（Fixed Balanced） | 1.71855 | 1.78504 | 0.83% | 99.17% | 2.766e-06 | 5.896e-05 | 0.25% |
| 固定保守策略（Fixed Conservative） | 0.99336 | 0.99926 | 0.07% | 99.93% | 4.291e-07 | 1.048e-05 | 0.00% |
| 固定峰值速率策略（Fixed Peak-Rate） | 0.22025 | 2.82992 | 32.62% | 67.38% | 3.284e-05 | 5.071e-04 | 10.05% |

## 五、bootstrap 与敏感性分析

| 比较对象 | 指标 | 变化均值 | 95%置信区间 | 正向概率 |
| --- | --- | --- | --- | --- |
| Improved LSTM Policy vs LSTM-raw Policy | mission_utility | 0.01747 | [0.01321, 0.02219] | 1.00000 |
| Improved LSTM Policy vs TCN Policy | mission_utility | 0.00048 | [-0.00287, 0.00383] | 0.60475 |
| Improved LSTM Policy vs TFT Policy | mission_utility | 0.00316 | [-0.00025, 0.00656] | 0.96550 |
| Improved LSTM Policy vs PatchTST Policy | mission_utility | 0.00611 | [0.00234, 0.00990] | 0.99925 |
| Improved LSTM Policy vs Persistence Policy | mission_utility | 0.01140 | [0.00745, 0.01555] | 1.00000 |

| 策略 | 平均排名 | 中位排名 | 第一名次数 | 平均效用 | 平均中断率 |
| --- | --- | --- | --- | --- | --- |
| 改进 LSTM 预测驱动策略（Improved LSTM Policy） | 1.25926 | 1.00000 | 23 | 2.35009 | 3.82% |
| 时间卷积网络预测驱动策略（TCN Policy） | 2.07407 | 2.00000 | 4 | 2.34784 | 3.65% |
| 时间融合 Transformer 预测驱动策略（TFT Policy） | 3.48148 | 3.00000 | 0 | 2.34509 | 3.66% |
| 分块时间序列 Transformer 预测驱动策略（PatchTST Policy） | 3.51852 | 4.00000 | 0 | 2.34362 | 3.61% |
| 基础 LSTM 预测驱动策略（LSTM-raw Policy） | 4.92593 | 5.00000 | 0 | 2.33539 | 4.44% |
| 持续性预测策略（Persistence Policy） | 5.74074 | 6.00000 | 0 | 2.33334 | 3.86% |

## 六、随机信道补充仿真

随机信道补充仿真加入了过境几何、仰角变化、指向抖动、路径损耗和块错误率（Block Error Rate，BLER）评价。主要配置包括轨道高度 550000 m，最小仰角 12.0°，最大仰角 82.0°，指向抖动 3.0 μrad，波束发散角 20.0 μrad。

在该补充仿真下，改进 LSTM 预测驱动策略的平均任务效用为 1.93742，接近理想策略的 1.95621，明显高于固定高速率策略的 0.54103；固定高速率策略虽然平均吞吐量较高，但中断率达到 28.24%。

| 策略 | 平均任务效用 | 平均吞吐量 | 中断率 | 可用率 | 平均编码 BER | P99 编码 BER | 平均 BLER |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 理想策略（Oracle Policy） | 1.95621 | 2.42623 | 5.88% | 94.12% | 4.646e-05 | 2.151e-02 | 0.54% |
| 改进 LSTM 预测驱动策略（Improved LSTM Policy） | 1.93742 | 2.43478 | 6.22% | 93.78% | 4.742e-05 | 2.166e-02 | 0.55% |
| 时间卷积网络预测驱动策略（TCN Policy） | 1.93646 | 2.43570 | 6.24% | 93.76% | 4.764e-05 | 2.151e-02 | 0.56% |
| 基础 LSTM 预测驱动策略（LSTM-raw Policy） | 1.93390 | 2.44270 | 6.36% | 93.64% | 4.788e-05 | 2.151e-02 | 0.56% |
| 持续性预测策略（Persistence Policy） | 1.93101 | 2.42945 | 6.23% | 93.77% | 4.797e-05 | 2.151e-02 | 0.56% |
| 固定均衡策略（Fixed Balanced） | 0.86000 | 1.75928 | 11.24% | 88.76% | 2.582e-04 | 1.061e-01 | 2.26% |
| 固定保守策略（Fixed Conservative） | 0.76997 | 0.99258 | 2.78% | 97.22% | 7.686e-05 | 5.001e-02 | 0.74% |
| 固定高速率策略（Fixed High-Rate） | 0.54103 | 2.80043 | 28.24% | 71.76% | 7.975e-04 | 1.410e-01 | 6.65% |
| 固定峰值速率策略（Fixed Peak-Rate） | -0.44292 | 3.82098 | 53.30% | 46.70% | 1.242e-03 | 1.613e-01 | 9.02% |

## 七、随机信道补充仿真的 MCS 使用比例

| 策略 | MCS | 次数 | 比例 |
| --- | --- | --- | --- |
| 固定均衡策略（Fixed Balanced） | MCS2-Balanced | 3225 | 100.00% |
| 固定保守策略（Fixed Conservative） | MCS1-Robust | 3225 | 100.00% |
| 固定高速率策略（Fixed High-Rate） | MCS3-HighRate | 3225 | 100.00% |
| 固定峰值速率策略（Fixed Peak-Rate） | MCS4-Peak | 3225 | 100.00% |
| 固定生存策略（Fixed Survival） | MCS0-Survival | 3225 | 100.00% |
| 改进 LSTM 预测驱动策略（Improved LSTM Policy） | MCS0-Survival | 307 | 9.52% |
| 改进 LSTM 预测驱动策略（Improved LSTM Policy） | MCS1-Robust | 369 | 11.44% |
| 改进 LSTM 预测驱动策略（Improved LSTM Policy） | MCS2-Balanced | 661 | 20.50% |
| 改进 LSTM 预测驱动策略（Improved LSTM Policy） | MCS3-HighRate | 1435 | 44.50% |
| 改进 LSTM 预测驱动策略（Improved LSTM Policy） | MCS4-Peak | 453 | 14.05% |
| 基础 LSTM 预测驱动策略（LSTM-raw Policy） | MCS0-Survival | 306 | 9.49% |
| 基础 LSTM 预测驱动策略（LSTM-raw Policy） | MCS1-Robust | 383 | 11.88% |
| 基础 LSTM 预测驱动策略（LSTM-raw Policy） | MCS2-Balanced | 626 | 19.41% |
| 基础 LSTM 预测驱动策略（LSTM-raw Policy） | MCS3-HighRate | 1449 | 44.93% |
| 基础 LSTM 预测驱动策略（LSTM-raw Policy） | MCS4-Peak | 461 | 14.29% |
| 理想策略（Oracle Policy） | MCS0-Survival | 300 | 9.30% |
| 理想策略（Oracle Policy） | MCS1-Robust | 380 | 11.78% |
| 理想策略（Oracle Policy） | MCS2-Balanced | 673 | 20.87% |
| 理想策略（Oracle Policy） | MCS3-HighRate | 1428 | 44.28% |
| 理想策略（Oracle Policy） | MCS4-Peak | 444 | 13.77% |
| 分块时间序列 Transformer 预测驱动策略（PatchTST Policy） | MCS0-Survival | 295 | 9.15% |
| 分块时间序列 Transformer 预测驱动策略（PatchTST Policy） | MCS1-Robust | 401 | 12.43% |
| 分块时间序列 Transformer 预测驱动策略（PatchTST Policy） | MCS2-Balanced | 657 | 20.37% |
| 分块时间序列 Transformer 预测驱动策略（PatchTST Policy） | MCS3-HighRate | 1463 | 45.36% |
| 分块时间序列 Transformer 预测驱动策略（PatchTST Policy） | MCS4-Peak | 409 | 12.68% |
| 持续性预测策略（Persistence Policy） | MCS0-Survival | 298 | 9.24% |
| 持续性预测策略（Persistence Policy） | MCS1-Robust | 391 | 12.12% |
| 持续性预测策略（Persistence Policy） | MCS2-Balanced | 638 | 19.78% |
| 持续性预测策略（Persistence Policy） | MCS3-HighRate | 1465 | 45.43% |
| 持续性预测策略（Persistence Policy） | MCS4-Peak | 433 | 13.43% |
| 时间卷积网络预测驱动策略（TCN Policy） | MCS0-Survival | 301 | 9.33% |
| 时间卷积网络预测驱动策略（TCN Policy） | MCS1-Robust | 376 | 11.66% |
| 时间卷积网络预测驱动策略（TCN Policy） | MCS2-Balanced | 670 | 20.78% |
| 时间卷积网络预测驱动策略（TCN Policy） | MCS3-HighRate | 1415 | 43.88% |
| 时间卷积网络预测驱动策略（TCN Policy） | MCS4-Peak | 463 | 14.36% |
| 时间融合 Transformer 预测驱动策略（TFT Policy） | MCS0-Survival | 296 | 9.18% |
| 时间融合 Transformer 预测驱动策略（TFT Policy） | MCS1-Robust | 377 | 11.69% |
| 时间融合 Transformer 预测驱动策略（TFT Policy） | MCS2-Balanced | 660 | 20.47% |
| 时间融合 Transformer 预测驱动策略（TFT Policy） | MCS3-HighRate | 1438 | 44.59% |
| 时间融合 Transformer 预测驱动策略（TFT Policy） | MCS4-Peak | 454 | 14.08% |

## 八、建议图表

- `outputs/ber_mcs_link_enhancement_20260427/main_site_ber_mcs_policy_summary.csv`
- `outputs/review_response_improvements_20260426/link_sensitivity_rank_stability.png`
- `outputs/engineering_link_monte_carlo_20260428/fig_engineering_tradeoff_scatter.png`
- `outputs/engineering_link_monte_carlo_20260428/fig_engineering_mcs_usage.png`
- `outputs/engineering_link_monte_carlo_20260428/fig_engineering_pass_timeline.png`
