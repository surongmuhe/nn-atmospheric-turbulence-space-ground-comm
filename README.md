# 基于神经网络的大气湍流预测与星地通信

本仓库保存“基于神经网络的大气湍流预测与星地通信”项目的可复现实验代码、模型配置、数据处理流程和关键结果。项目面向星地激光通信链路自适应需求，以大气光学湍流强度 `log10(Cn^2)` 为预测变量，研究未来 30 min 时间窗内平均湍流强度预测，并将预测结果进一步接入调制编码选择和链路性能评价流程。

仓库已经清理与论文写作、Word 排版、引用调整、图表插入相关的临时脚本和文档输出，仅保留数据处理、模型训练、实验复现、关键结果和链路仿真所需内容。

## 1. 项目任务

星地激光通信具有高带宽、窄波束和抗电磁干扰等优势，但近地大气湍流会引起光强闪烁、波前畸变和接收功率波动。折射率结构常数 `Cn^2` 是表征光学湍流强度的重要物理量，由于其数量级很小，项目统一采用 `log10(Cn^2)` 作为建模对象。

本项目的最终主线任务为：基于连续观测序列，预测未来 30 min 时间窗内的平均 `log10(Cn^2)`。该任务比单个未来时刻预测更稳定，也更适合链路自适应决策。模型输出随后被转换回 `Cn^2` 物理尺度，并用于链路裕量、误码率代理、调制编码方案和任务效用评价。

## 2. 数据集与处理

| 数据集 | 文件 | 用途 |
|---|---|---|
| 主站点数据 | `data/Sklavounos and Cohn, Spring. 2022.xlsx` | 主站点训练、验证、测试、强基线对比和严格单因素消融 |
| 外部站点数据 | `data/dataWfon0U_ml_ready.csv` | 外部站点复现、零样本迁移、少样本适配和跨站点链路补充验证 |

主要处理步骤位于 `src/data.py` 和相关实验入口脚本中，包括：

- 时间排序、重复时间戳聚合和 5 min 间隔重采样；
- 对不超过 3 个采样点的连续短缺口进行时间插值，长缺口不跨越构造窗口样本；
- 使用训练集统计量进行特征标准化，避免验证集和测试集信息泄漏；
- 构造长度为 72 的历史窗口，对应 6 h 历史观测；
- 将预测目标设置为未来 30 min 时间窗平均 `log10(Cn^2)`；
- 构造时间周期特征、物理启发特征、一阶差分特征、滚动统计特征和太阳状态特征。

## 3. 模型路线

项目采用递进式建模路线，逐步检验当前特征、历史窗口、显式时序结构和任务增强设计的作用。

| 模型 | 作用 |
|---|---|
| `ANN-current` | 仅使用当前时刻特征，检验无历史窗口条件下的基础预测能力。 |
| `ANN-window` | 将 72 步历史窗口展开输入前馈网络，检验历史窗口是否带来增益。 |
| `LSTM-raw` | 使用基础 LSTM 显式建模时序依赖，作为循环神经网络基线。 |
| `Improved LSTM` | 主模型，引入未来均值目标、增量预测、增强特征、双向编码、注意力池化、短窗自回归支路和门控预测头。 |
| `TCN`、`TFT`、`PatchTST` | 外部强时序基线，用于检验主模型相对现代时序模型的竞争力。 |

## 4. 实验设计与关键结果

### 4.1 主站点递进式对比

结果目录：`outputs/current_route_main_site/formal_20260425_delta`

该实验用于回答：从当前特征到历史窗口，再到显式时序结构和 Improved LSTM，预测性能是否逐步提升。

| 模型 | RMSE | MAE | MSE/MAPE 列按项目输出命名 | R2 |
|---|---:|---:|---:|---:|
| ANN-current | 0.14049 | 0.10128 | 0.00759 | 0.88026 |
| ANN-window | 0.13934 | 0.10006 | 0.00751 | 0.88221 |
| LSTM-raw | 0.13707 | 0.09751 | 0.00733 | 0.88601 |
| Improved LSTM | 0.11635 | 0.08341 | 0.00625 | 0.91908 |

结果表明，单纯展开历史窗口只能带来有限改善，显式时序结构进一步提升性能，而 Improved LSTM 在主站点测试集上取得最优结果。

### 4.2 强基线与 5 组随机种子

结果目录：`outputs/strong_baselines_seq72_20260425`

该实验使用 5 组随机种子，对 Improved LSTM、LSTM-raw、TCN、TFT 和 PatchTST 进行统一任务口径下的对比，避免只依赖单次随机初始化得出结论。

| 模型 | RMSE | MAE | R2 |
|---|---:|---:|---:|
| Improved LSTM | 0.11899±0.00249 | 0.08539±0.00212 | 0.91535±0.00357 |
| TCN | 0.12530±0.00352 | 0.09252±0.00405 | 0.90610±0.00528 |
| TFT | 0.13033±0.00527 | 0.09394±0.00563 | 0.89835±0.00826 |
| LSTM-raw | 0.13675±0.00433 | 0.09760±0.00299 | 0.88646±0.00721 |
| PatchTST | 0.14390±0.02295 | 0.10305±0.01530 | 0.87372±0.04055 |

Improved LSTM 在 5 组随机种子均值上保持最优，说明其优势不依赖单一训练种子。

### 4.3 严格单因素消融实验

结果目录：`outputs/strict_independent_ablation_20260501`

该实验以最终 `seq_len=72` 的 Improved LSTM 为参照组，每次只改变一个因素，其余数据划分、训练参数、随机种子、预测窗口和评价指标保持一致。消融实验分为三类：任务定义消融、输入特征消融和模型结构消融。

| 中文名称 | 单一改动 | RMSE_mean | R2_mean | 相对完整模型 RMSE 变化 | 结论 |
|---|---|---:|---:|---:|---|
| 完整改进 LSTM 模型 | 不改变配置，作为参照组 | 0.11899 | 0.91535 | 0.00000 | 完整模型基准 |
| 去除未来均值目标 | 将未来 30 min 均值目标改为 6 步多步预测 | 0.17545 | 0.83289 | +0.05647 | 未来均值目标是最关键的任务设计 |
| 去除增量预测目标 | 将增量预测改为绝对值预测 | 0.12166 | 0.91150 | +0.00268 | 增量建模有稳定正向作用 |
| 去除时间周期特征 | 移除 `hour_sin` 和 `hour_cos` | 0.11907 | 0.91523 | +0.00009 | 独立影响较小 |
| 去除物理派生特征 | 关闭温差、风速平方等物理派生量 | 0.11799 | 0.91679 | -0.00100 | 与其他特征存在替代关系 |
| 去除太阳状态特征 | 关闭夜间、晨昏、强日照等状态特征 | 0.12032 | 0.91345 | +0.00134 | 有小幅正向作用 |
| 去除一阶差分特征 | 关闭短时变化率特征 | 0.12002 | 0.91390 | +0.00103 | 有小幅正向作用 |
| 去除滚动统计特征 | 关闭短窗均值和标准差特征 | 0.11909 | 0.91523 | +0.00010 | 独立影响较小 |
| 去除双向编码 | 将 LSTM 编码器由双向改为单向 | 0.12138 | 0.91191 | +0.00239 | 双向历史编码有稳定正向作用 |
| 去除注意力池化 | 关闭注意力池化 | 0.11870 | 0.91578 | -0.00029 | 与其他结构存在替代关系 |
| 去除短窗自回归支路 | 将 `ar_window` 由 12 改为 0 | 0.11820 | 0.91648 | -0.00078 | 独立影响较小 |
| 去除输入投影层 | 将 `input_proj_size` 由 64 改为 `None` | 0.11924 | 0.91500 | +0.00026 | 有轻微正向作用 |
| 基础 LSTM 结构对照 | 仅将模型类换为基础 LSTM | 0.11976 | 0.91426 | +0.00078 | 增强结构整体有小幅收益 |
| 窗口多层感知机对照 | 仅将模型类换为窗口 MLP | 0.13283 | 0.89448 | +0.01384 | 显式时序建模明显优于窗口展开前馈网络 |

消融结果的解释口径为：未来均值目标、增量预测、双向编码和显式时序建模是主要收益来源；部分增强特征或结构的独立贡献较小，说明它们与其他模块存在功能重叠，不应被解释为每个模块都带来同等幅度提升。

### 4.4 外部站点四模型复现

结果目录：`outputs/external_site_four_model_route_20260426_validated`

该实验在外部站点重新训练 ANN-current、ANN-window、LSTM-raw 和 Improved LSTM，用于检验递进式建模关系是否能在不同站点复现。

| 模型 | RMSE | MAE | MSE | MAPE | R2 |
|---|---:|---:|---:|---:|---:|
| ANN-current | 0.23135 | 0.17797 | 0.05352 | 1.30659 | 0.84810 |
| ANN-window | 0.19696 | 0.14166 | 0.03879 | 1.04167 | 0.88991 |
| LSTM-raw | 0.18942 | 0.13854 | 0.03588 | 1.01493 | 0.89817 |
| Improved LSTM | 0.15898 | 0.11496 | 0.02527 | 0.84233 | 0.92827 |

外部站点上仍呈现 `ANN-current < ANN-window < LSTM-raw < Improved LSTM` 的性能顺序，说明主线建模关系具有一定可复现性。

### 4.5 零样本迁移与少样本适配

结果目录：

- `outputs/cross_site_fewshot_adaptation/formal_20260426`
- `outputs/fewshot_strong_transfer_controls_20260427`

该实验用于分析新站点冷启动和少量本地数据可用时的部署策略。源站点模型可作为无目标站点标签时的初始预测器；当目标站点积累少量连续历史观测后，可通过微调提升本地预测效果。

| 目标站点适配比例 | 推荐模型 | RMSE | MAE | R2 |
|---:|---|---:|---:|---:|
| 0% | source-only LSTM-raw | 0.27344 | 0.20037 | 0.85475 |
| 5% | few-shot Improved LSTM | 0.26770 | 0.18805 | 0.87282 |
| 10% | few-shot Improved LSTM | 0.26048 | 0.17667 | 0.88070 |
| 20% | few-shot Improved LSTM | 0.24756 | 0.16968 | 0.89365 |

结果表明，零样本迁移可以用于冷启动，但站点分布差异会限制精度；少样本微调能够持续改善外部站点预测效果。

### 4.6 星地通信链路自适应仿真

结果目录：

- `outputs/current_stage_link_simulation_20260501`
- `outputs/ber_mcs_link_enhancement_20260427`
- `outputs/engineering_link_monte_carlo_20260428`

链路仿真将预测得到的 `log10(Cn^2)` 转换为 `Cn^2`，再接入 Rytov 方差、闪烁衰落、链路裕量、调制编码方案和误码率代理评价。该部分属于预测驱动的链路级工程代理验证，不等同于真实星地激光通信终端实测。

| 策略 | 平均任务效用 | 平均吞吐量 | 中断率 | 可用率 | 平均 BER proxy |
|---|---:|---:|---:|---:|---:|
| Oracle Policy | 2.38113 | 2.54896 | 2.10% | 97.90% | 3.540e-06 |
| Improved LSTM Policy | 2.36272 | 2.54639 | 2.30% | 97.70% | 5.308e-06 |
| TCN Policy | 2.36224 | 2.53246 | 2.13% | 97.87% | 5.202e-06 |
| LSTM-raw Policy | 2.34525 | 2.58827 | 3.04% | 96.96% | 6.214e-06 |
| Fixed High-Rate | 2.09267 | 2.75255 | 8.25% | 91.75% | 3.069e-05 |

默认链路仿真中，Improved LSTM Policy 的任务效用接近 Oracle Policy，并显著降低 Fixed High-Rate 的中断风险。Bootstrap 对比显示，Improved LSTM Policy 相对 LSTM-raw Policy 的任务效用增益均值为 0.01747，95% 置信区间为 `[0.01321, 0.02219]`，正向概率为 1.0。

随机信道补充仿真进一步加入过境几何、仰角变化、指向抖动、路径损耗和块错误率评价。在该设置下，Improved LSTM Policy 的平均任务效用为 1.93742，接近 Oracle Policy 的 1.95621，并明显优于 Fixed High-Rate 的 0.54103。

## 5. 目录结构

```text
fusion_forecasting_project/
├── configs/       # 实验配置
├── data/          # 主站点与外部站点数据
├── outputs/       # 关键实验结果、表格和图像
├── scripts/       # 实验入口、数据准备和链路仿真脚本
├── src/           # 数据处理、模型、训练、评估和链路仿真源码
├── README.md
└── requirements.txt
```

## 6. 复现方式

安装依赖：

```bash
pip install -r requirements.txt
```

运行主站点四模型实验：

```bash
python scripts/run_current_route_main_site.py
```

运行最终 Improved LSTM：

```bash
python scripts/run_experiments.py --config configs/thesis_lstm_final_best_seq72.yaml
```

运行强基线与 5 组随机种子：

```bash
python scripts/run_strong_baselines.py
```

运行严格单因素消融：

```bash
python scripts/run_strict_independent_ablation.py
```

运行外部站点四模型复现：

```bash
python scripts/run_external_site_four_model_route.py
python scripts/sweep_external_site_lstm_raw.py
python scripts/align_external_site_four_model_metrics.py
```

运行少样本适配对照：

```bash
python scripts/run_cross_site_fewshot_adaptation.py
python scripts/run_fewshot_strong_transfer_controls.py
python scripts/aggregate_fewshot_strong_transfer_controls.py
```

运行链路级 BER/MCS 仿真：

```bash
python scripts/run_ber_mcs_link_enhancement.py
python scripts/summarize_current_stage_link_simulation.py
python scripts/run_engineering_link_monte_carlo.py
```

## 7. 输出文件索引

| 实验 | 主要输出 |
|---|---|
| 主站点递进式对比 | `outputs/current_route_main_site/formal_20260425_delta` |
| 强基线 5 种子 | `outputs/strong_baselines_seq72_20260425/strong_baselines_seed_summary.csv` |
| 严格单因素消融 | `outputs/strict_independent_ablation_20260501/strict_ablation_summary.md` |
| 外部站点四模型复现 | `outputs/external_site_four_model_route_20260426_validated/aligned_four_model_summary.md` |
| 少样本适配 | `outputs/cross_site_fewshot_adaptation/formal_20260426/fewshot_adaptation_summary.json` |
| 链路自适应仿真 | `outputs/current_stage_link_simulation_20260501/current_stage_link_simulation_summary.md` |

## 8. 发布说明

GitHub 发布快照保留核心源码、模型配置、数据集和关键实验结果。为控制仓库体积，默认不上传训练得到的 `.pt/.pth` 权重、大体积逐样本预测文件、Python 缓存、临时日志和论文写作文档。
