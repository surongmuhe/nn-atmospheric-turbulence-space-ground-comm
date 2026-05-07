# 基于神经网络的大气湍流预测与星地通信

本仓库保存“基于神经网络的大气湍流预测与星地通信”项目的可复现实验代码、模型配置、数据处理流程和关键实验结果。项目以大气光学湍流强度 `log10(Cn^2)` 预测为主线，比较 ANN-current、ANN-window、LSTM-raw、Improved LSTM 以及 TCN、TFT、PatchTST 等时序模型，并将预测结果接入星地光通信链路自适应仿真。

仓库已清理与论文撰写、Word 排版、引用调整、图表插入相关的临时脚本和文档输出，仅保留模型、实验复现和链路仿真所需内容。

## 1. 技术路线

```text
数据预处理
  -> ANN-current / ANN-window
  -> LSTM-raw
  -> Improved LSTM
  -> 强时序基线对比与消融实验
  -> 外部站点复现、零样本迁移和少样本适配
  -> 星地通信链路自适应仿真
```

核心模型含义如下：

- `ANN-current`：仅使用当前时刻气象与湍流相关特征，作为无历史窗口的非递归基线。
- `ANN-window`：将长度为 72 的历史窗口展开为前馈网络输入，用于检验历史观测窗口的增益。
- `LSTM-raw`：使用基础 LSTM 显式建模时间依赖，作为循环时序结构基线。
- `Improved LSTM`：在 LSTM 基础上引入 30 min 平均目标、增量预测、时间周期特征、物理启发特征、双向编码、注意力池化、短窗自回归支路和门控预测头，是本项目主模型。
- `TCN`、`TFT`、`PatchTST`：作为外部强时序基线，用于检验 Improved LSTM 的竞争力。

## 2. 数据集

| 数据集 | 文件 | 用途 |
|---|---|---|
| 主站点数据 | `data/Sklavounos and Cohn, Spring. 2022.xlsx` | 主站点训练、验证、测试、强基线和消融实验 |
| 外部站点数据 | `data/dataWfon0U_ml_ready.csv` | 外部站点复现、零样本迁移和少样本适配 |

数据处理逻辑位于 `src/data.py` 和相关实验入口脚本中，包含时间排序、重采样、窗口样本构造、特征标准化以及预测目标构造等步骤。

## 3. 关键结果

### 3.1 主站点四模型结果

结果目录：`outputs/current_route_main_site/formal_20260425_delta`

| 模型 | RMSE | MAE | MSE | R2 |
|---|---:|---:|---:|---:|
| ANN-current | 0.14049 | 0.10128 | 0.00759 | 0.88026 |
| ANN-window | 0.13934 | 0.10006 | 0.00751 | 0.88221 |
| LSTM-raw | 0.13707 | 0.09751 | 0.00733 | 0.88601 |
| Improved LSTM | 0.11635 | 0.08341 | 0.00625 | 0.91908 |

### 3.2 强基线与 5 组随机种子

结果目录：`outputs/strong_baselines_seq72_20260425`

| 模型 | RMSE | MAE | R2 |
|---|---:|---:|---:|
| Improved LSTM | 0.11899±0.00249 | 0.08539±0.00212 | 0.91535±0.00357 |
| TCN | 0.12530±0.00352 | 0.09252±0.00405 | 0.90610±0.00528 |
| TFT | 0.13033±0.00527 | 0.09394±0.00563 | 0.89835±0.00826 |
| LSTM-raw | 0.13675±0.00433 | 0.09760±0.00299 | 0.88646±0.00721 |
| PatchTST | 0.14390±0.02295 | 0.10305±0.01530 | 0.87372±0.04055 |

### 3.3 外部站点四模型复现

结果目录：`outputs/external_site_four_model_route_20260426_validated`

| 模型 | RMSE | MAE | MSE | R2 |
|---|---:|---:|---:|---:|
| ANN-current | 0.23135 | 0.17797 | 0.05352 | 0.84810 |
| ANN-window | 0.19696 | 0.14167 | 0.03879 | 0.88991 |
| LSTM-raw | 0.18942 | 0.13854 | 0.03588 | 0.89817 |
| Improved LSTM | 0.15898 | 0.11496 | 0.02527 | 0.92827 |

### 3.4 少样本适配

结果目录：

- `outputs/cross_site_fewshot_adaptation/formal_20260426`
- `outputs/fewshot_strong_transfer_controls_20260427`

源站点训练的 Improved LSTM 在外部站点零样本测试中达到 `R2=0.82148`。使用目标站点前 5%、10% 和 20% 数据微调后，R2 分别提升至 `0.87282`、`0.88070` 和 `0.89365`。

### 3.5 星地通信链路自适应仿真

结果目录：

- `outputs/review_response_improvements_20260426`
- `outputs/ber_mcs_link_enhancement_20260427`

| 策略 | 任务效用 | 中断率 | 可用率 | 平均 BER proxy |
|---|---:|---:|---:|---:|
| Oracle Policy | 2.3811 | 2.10% | 97.90% | 3.54e-06 |
| Improved LSTM Policy | 2.3627 | 2.30% | 97.70% | 5.31e-06 |
| TCN Policy | 2.3622 | 2.13% | 97.87% | 5.20e-06 |
| Fixed High-Rate | 2.0927 | 8.25% | 91.75% | 3.07e-05 |

链路仿真为预测驱动的工程代理验证，不等同于真实星地通信终端实测。

## 4. 目录结构

```text
fusion_forecasting_project/
├── configs/       # 实验配置
├── data/          # 主站点与外部站点数据
├── outputs/       # 关键实验结果汇总
├── scripts/       # 实验入口、数据准备和链路仿真脚本
├── src/           # 数据处理、模型、训练、评估和链路仿真源码
├── README.md
└── requirements.txt
```

## 5. 复现方式

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
python scripts/run_fewshot_strong_transfer_controls.py
python scripts/aggregate_fewshot_strong_transfer_controls.py
```

运行链路级 BER/MCS 仿真：

```bash
python scripts/run_ber_mcs_link_enhancement.py
```

## 6. 发布说明

GitHub 发布快照保留核心源码、模型配置、数据集和关键结果汇总。为控制仓库体积，默认不上传训练得到的 `.pt/.pth` 权重、大体积逐样本预测文件、Python 缓存、临时日志和论文写作文档。
