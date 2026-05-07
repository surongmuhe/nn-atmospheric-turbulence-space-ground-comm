# 严格单因素消融实验汇总

生成时间：2026-05-01T03:14:08

## 实验原则

以最终 `seq_len=72` 的改进 LSTM 配置为完整模型，除被考察因素外，其余数据划分、训练参数、随机种子、预测步长和评价指标保持一致。任务定义类消融会改变预测目标本身，因此用于回答目标设计是否必要；输入特征类和模型结构类消融用于回答对应模块的独立作用。

## 正文建议主表

| 中文名称 | 消融类别 | RMSE_mean | MAE_mean | MAPE_mean | R2_mean | 相对完整模型RMSE变化 | RMSE相对变化率 | 结论口径 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 完整改进 LSTM 模型 | 基准模型 | 0.11899 | 0.08539 | 0.00640 | 0.91535 | 0.00000 | 0.00% | 完整模型作为单因素消融参照组。 |
| 去除未来均值目标 | 任务定义消融 | 0.17545 | 0.12654 | 0.00949 | 0.83289 | 0.05647 | 47.46% | 去除该因素后 RMSE 上升，说明该因素对最终模型有正向作用。 |
| 去除增量预测目标 | 任务定义消融 | 0.12166 | 0.08898 | 0.00666 | 0.91150 | 0.00268 | 2.25% | 去除该因素后 RMSE 上升，说明该因素对最终模型有正向作用。 |
| 去除时间周期特征 | 输入特征消融 | 0.11907 | 0.08537 | 0.00640 | 0.91523 | 0.00009 | 0.07% | 去除该因素后 RMSE 变化较小，说明该因素影响有限或与其他模块存在替代关系。 |
| 去除物理派生特征 | 输入特征消融 | 0.11799 | 0.08521 | 0.00638 | 0.91679 | -0.00100 | -0.84% | 去除该因素后 RMSE 变化较小，说明该因素影响有限或与其他模块存在替代关系。 |
| 去除太阳状态特征 | 输入特征消融 | 0.12032 | 0.08802 | 0.00659 | 0.91345 | 0.00134 | 1.12% | 去除该因素后 RMSE 变化较小，说明该因素影响有限或与其他模块存在替代关系。 |
| 去除一阶差分特征 | 输入特征消融 | 0.12002 | 0.08680 | 0.00650 | 0.91390 | 0.00103 | 0.87% | 去除该因素后 RMSE 变化较小，说明该因素影响有限或与其他模块存在替代关系。 |
| 去除滚动统计特征 | 输入特征消融 | 0.11909 | 0.08617 | 0.00645 | 0.91523 | 0.00010 | 0.08% | 去除该因素后 RMSE 变化较小，说明该因素影响有限或与其他模块存在替代关系。 |
| 去除双向编码 | 模型结构消融 | 0.12138 | 0.08687 | 0.00651 | 0.91191 | 0.00239 | 2.01% | 去除该因素后 RMSE 上升，说明该因素对最终模型有正向作用。 |
| 去除注意力池化 | 模型结构消融 | 0.11870 | 0.08484 | 0.00636 | 0.91578 | -0.00029 | -0.24% | 去除该因素后 RMSE 变化较小，说明该因素影响有限或与其他模块存在替代关系。 |
| 去除短窗自回归支路 | 模型结构消融 | 0.11820 | 0.08460 | 0.00634 | 0.91648 | -0.00078 | -0.66% | 去除该因素后 RMSE 变化较小，说明该因素影响有限或与其他模块存在替代关系。 |
| 去除输入投影层 | 模型结构消融 | 0.11924 | 0.08580 | 0.00643 | 0.91500 | 0.00026 | 0.22% | 去除该因素后 RMSE 变化较小，说明该因素影响有限或与其他模块存在替代关系。 |
| 基础 LSTM 结构对照 | 模型结构对照 | 0.11976 | 0.08568 | 0.00642 | 0.91426 | 0.00078 | 0.65% | 去除该因素后 RMSE 变化较小，说明该因素影响有限或与其他模块存在替代关系。 |
| 窗口多层感知机对照 | 模型结构对照 | 0.13283 | 0.09641 | 0.00722 | 0.89448 | 0.01384 | 11.63% | 去除该因素后 RMSE 上升，说明该因素对最终模型有正向作用。 |

## 每个实验的单一改动

| 中文名称 | 消融类别 | 单一改动 | 论文用途 |
| --- | --- | --- | --- |
| 完整改进 LSTM 模型 | 基准模型 | 不改变配置，作为严格消融参照组。 | 正文主参照组 |
| 去除未来均值目标 | 任务定义消融 | 仅将目标聚合方式由未来 30 min 均值改为 6 步多步预测。 | 检验未来均值目标对短时平均湍流预测的作用。 |
| 去除增量预测目标 | 任务定义消融 | 仅将目标形式由增量预测改为绝对值预测。 | 检验增量预测对站点偏置和短时变化建模的作用。 |
| 去除时间周期特征 | 输入特征消融 | 仅移除 hour_sin 和 hour_cos，并关闭小时周期特征开关。 | 检验昼夜周期信息对湍流预测的作用。 |
| 去除物理派生特征 | 输入特征消融 | 仅关闭 domain_features，其余时间、差分、滚动统计设置保持不变。 | 检验温差、太阳辐射非负化、风速平方等物理派生量的作用。 |
| 去除太阳状态特征 | 输入特征消融 | 仅关闭 solar_regime_features。 | 检验夜间、晨昏、强日照和太阳辐射跃迁信息的作用。 |
| 去除一阶差分特征 | 输入特征消融 | 仅关闭 diff_features。 | 检验短时变化率特征对预测的作用。 |
| 去除滚动统计特征 | 输入特征消融 | 仅关闭 rolling_features。 | 检验短窗均值和标准差统计对预测的作用。 |
| 去除双向编码 | 模型结构消融 | 仅将 LSTM 编码器由双向改为单向。 | 检验双向历史上下文编码的作用。 |
| 去除注意力池化 | 模型结构消融 | 仅关闭 attention_pool。 | 检验注意力池化对历史窗口信息聚合的作用。 |
| 去除短窗自回归支路 | 模型结构消融 | 仅将 ar_window 由 12 改为 0。 | 检验短窗目标历史支路对局部惯性的刻画作用。 |
| 去除输入投影层 | 模型结构消融 | 仅将 input_proj_size 由 64 改为 None。 | 检验输入特征投影压缩对表示学习的作用。 |
| 基础 LSTM 结构对照 | 模型结构对照 | 仅将模型类由改进 LSTM 换为基础 LSTM，输入特征和任务设置保持一致。 | 检验改进预测头和增强结构相对基础循环模型的整体作用。 |
| 窗口多层感知机对照 | 模型结构对照 | 仅将模型类由改进 LSTM 换为窗口 MLP，输入特征和任务设置保持一致。 | 检验显式时序建模相对窗口展开前馈网络的作用。 |

## 各随机种子原始指标

| variant | 中文名称 | seed | model | RMSE | MAE | MAPE | R2 | target_mode | target_aggregation | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_delta_target | 去除增量预测目标 | 42 | lstm | 0.12122 | 0.09091 | 0.00681 | 0.91217 | absolute | mean | outputs\strict_independent_ablation_20260501\runs\no_delta_target\seed_42\metrics_all_models.csv |
| no_delta_target | 去除增量预测目标 | 52 | lstm | 0.12156 | 0.08725 | 0.00653 | 0.91168 | absolute | mean | outputs\strict_independent_ablation_20260501\runs\no_delta_target\seed_52\metrics_all_models.csv |
| no_delta_target | 去除增量预测目标 | 62 | lstm | 0.12455 | 0.08940 | 0.00670 | 0.90728 | absolute | mean | outputs\strict_independent_ablation_20260501\runs\no_delta_target\seed_62\metrics_all_models.csv |
| no_delta_target | 去除增量预测目标 | 72 | lstm | 0.12293 | 0.09202 | 0.00688 | 0.90968 | absolute | mean | outputs\strict_independent_ablation_20260501\runs\no_delta_target\seed_72\metrics_all_models.csv |
| no_delta_target | 去除增量预测目标 | 82 | lstm | 0.11805 | 0.08534 | 0.00639 | 0.91671 | absolute | mean | outputs\strict_independent_ablation_20260501\runs\no_delta_target\seed_82\metrics_all_models.csv |
| no_future_mean_target | 去除未来均值目标 | 42 | lstm | 0.17355 | 0.12601 | 0.00944 | 0.83652 | delta | multistep | outputs\strict_independent_ablation_20260501\runs\no_future_mean_target\seed_42\metrics_all_models.csv |
| no_future_mean_target | 去除未来均值目标 | 52 | lstm | 0.17720 | 0.12729 | 0.00956 | 0.82955 | delta | multistep | outputs\strict_independent_ablation_20260501\runs\no_future_mean_target\seed_52\metrics_all_models.csv |
| no_future_mean_target | 去除未来均值目标 | 62 | lstm | 0.17717 | 0.12660 | 0.00951 | 0.82962 | delta | multistep | outputs\strict_independent_ablation_20260501\runs\no_future_mean_target\seed_62\metrics_all_models.csv |
| no_future_mean_target | 去除未来均值目标 | 72 | lstm | 0.17514 | 0.12727 | 0.00954 | 0.83350 | delta | multistep | outputs\strict_independent_ablation_20260501\runs\no_future_mean_target\seed_72\metrics_all_models.csv |
| no_future_mean_target | 去除未来均值目标 | 82 | lstm | 0.17421 | 0.12553 | 0.00942 | 0.83526 | delta | multistep | outputs\strict_independent_ablation_20260501\runs\no_future_mean_target\seed_82\metrics_all_models.csv |
| full_improved | 完整改进 LSTM 模型 | 42 | lstm | 0.11635 | 0.08341 | 0.00625 | 0.91908 | delta | mean | outputs\strict_independent_ablation_20260501\runs\full_improved\seed_42\metrics_all_models.csv |
| full_improved | 完整改进 LSTM 模型 | 52 | lstm | 0.12298 | 0.08768 | 0.00658 | 0.90961 | delta | mean | outputs\strict_independent_ablation_20260501\runs\full_improved\seed_52\metrics_all_models.csv |
| full_improved | 完整改进 LSTM 模型 | 62 | lstm | 0.11824 | 0.08417 | 0.00631 | 0.91644 | delta | mean | outputs\strict_independent_ablation_20260501\runs\full_improved\seed_62\metrics_all_models.csv |
| full_improved | 完整改进 LSTM 模型 | 72 | lstm | 0.11946 | 0.08768 | 0.00656 | 0.91470 | delta | mean | outputs\strict_independent_ablation_20260501\runs\full_improved\seed_72\metrics_all_models.csv |
| full_improved | 完整改进 LSTM 模型 | 82 | lstm | 0.11791 | 0.08398 | 0.00630 | 0.91691 | delta | mean | outputs\strict_independent_ablation_20260501\runs\full_improved\seed_82\metrics_all_models.csv |
| mlp_window_same_features | 窗口多层感知机对照 | 42 | mlp | 0.13289 | 0.09644 | 0.00723 | 0.89445 | delta | mean | outputs\strict_independent_ablation_20260501\runs\mlp_window_same_features\seed_42\metrics_all_models.csv |
| mlp_window_same_features | 窗口多层感知机对照 | 52 | mlp | 0.13474 | 0.09771 | 0.00731 | 0.89149 | delta | mean | outputs\strict_independent_ablation_20260501\runs\mlp_window_same_features\seed_52\metrics_all_models.csv |
| mlp_window_same_features | 窗口多层感知机对照 | 62 | mlp | 0.12902 | 0.09386 | 0.00703 | 0.90051 | delta | mean | outputs\strict_independent_ablation_20260501\runs\mlp_window_same_features\seed_62\metrics_all_models.csv |
| mlp_window_same_features | 窗口多层感知机对照 | 72 | mlp | 0.13783 | 0.10062 | 0.00754 | 0.88646 | delta | mean | outputs\strict_independent_ablation_20260501\runs\mlp_window_same_features\seed_72\metrics_all_models.csv |
| mlp_window_same_features | 窗口多层感知机对照 | 82 | mlp | 0.12967 | 0.09341 | 0.00700 | 0.89951 | delta | mean | outputs\strict_independent_ablation_20260501\runs\mlp_window_same_features\seed_82\metrics_all_models.csv |
| vanilla_lstm_same_features | 基础 LSTM 结构对照 | 42 | vanilla_lstm | 0.12048 | 0.08583 | 0.00643 | 0.91324 | delta | mean | outputs\strict_independent_ablation_20260501\runs\vanilla_lstm_same_features\seed_42\metrics_all_models.csv |
| vanilla_lstm_same_features | 基础 LSTM 结构对照 | 52 | vanilla_lstm | 0.12179 | 0.08942 | 0.00670 | 0.91134 | delta | mean | outputs\strict_independent_ablation_20260501\runs\vanilla_lstm_same_features\seed_52\metrics_all_models.csv |
| vanilla_lstm_same_features | 基础 LSTM 结构对照 | 62 | vanilla_lstm | 0.12012 | 0.08506 | 0.00637 | 0.91375 | delta | mean | outputs\strict_independent_ablation_20260501\runs\vanilla_lstm_same_features\seed_62\metrics_all_models.csv |
| vanilla_lstm_same_features | 基础 LSTM 结构对照 | 72 | vanilla_lstm | 0.11800 | 0.08339 | 0.00625 | 0.91677 | delta | mean | outputs\strict_independent_ablation_20260501\runs\vanilla_lstm_same_features\seed_72\metrics_all_models.csv |
| vanilla_lstm_same_features | 基础 LSTM 结构对照 | 82 | vanilla_lstm | 0.11842 | 0.08470 | 0.00635 | 0.91619 | delta | mean | outputs\strict_independent_ablation_20260501\runs\vanilla_lstm_same_features\seed_82\metrics_all_models.csv |
| no_attention_pooling | 去除注意力池化 | 42 | lstm | 0.11939 | 0.08547 | 0.00640 | 0.91480 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_attention_pooling\seed_42\metrics_all_models.csv |
| no_attention_pooling | 去除注意力池化 | 52 | lstm | 0.11857 | 0.08430 | 0.00631 | 0.91597 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_attention_pooling\seed_52\metrics_all_models.csv |
| no_attention_pooling | 去除注意力池化 | 62 | lstm | 0.11876 | 0.08372 | 0.00628 | 0.91570 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_attention_pooling\seed_62\metrics_all_models.csv |
| no_attention_pooling | 去除注意力池化 | 72 | lstm | 0.12006 | 0.08777 | 0.00658 | 0.91384 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_attention_pooling\seed_72\metrics_all_models.csv |
| no_attention_pooling | 去除注意力池化 | 82 | lstm | 0.11670 | 0.08291 | 0.00621 | 0.91860 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_attention_pooling\seed_82\metrics_all_models.csv |
| no_bidirectional_encoder | 去除双向编码 | 42 | lstm | 0.12527 | 0.08955 | 0.00669 | 0.90620 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_bidirectional_encoder\seed_42\metrics_all_models.csv |
| no_bidirectional_encoder | 去除双向编码 | 52 | lstm | 0.12176 | 0.08757 | 0.00656 | 0.91139 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_bidirectional_encoder\seed_52\metrics_all_models.csv |
| no_bidirectional_encoder | 去除双向编码 | 62 | lstm | 0.12253 | 0.08661 | 0.00649 | 0.91027 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_bidirectional_encoder\seed_62\metrics_all_models.csv |
| no_bidirectional_encoder | 去除双向编码 | 72 | lstm | 0.11745 | 0.08451 | 0.00633 | 0.91755 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_bidirectional_encoder\seed_72\metrics_all_models.csv |
| no_bidirectional_encoder | 去除双向编码 | 82 | lstm | 0.11987 | 0.08612 | 0.00645 | 0.91412 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_bidirectional_encoder\seed_82\metrics_all_models.csv |
| no_input_projection | 去除输入投影层 | 42 | lstm | 0.11666 | 0.08409 | 0.00629 | 0.91866 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_input_projection\seed_42\metrics_all_models.csv |
| no_input_projection | 去除输入投影层 | 52 | lstm | 0.11908 | 0.08676 | 0.00651 | 0.91524 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_input_projection\seed_52\metrics_all_models.csv |
| no_input_projection | 去除输入投影层 | 62 | lstm | 0.11962 | 0.08638 | 0.00647 | 0.91448 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_input_projection\seed_62\metrics_all_models.csv |
| no_input_projection | 去除输入投影层 | 72 | lstm | 0.11945 | 0.08473 | 0.00635 | 0.91472 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_input_projection\seed_72\metrics_all_models.csv |
| no_input_projection | 去除输入投影层 | 82 | lstm | 0.12141 | 0.08702 | 0.00652 | 0.91190 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_input_projection\seed_82\metrics_all_models.csv |
| no_short_ar_branch | 去除短窗自回归支路 | 42 | lstm | 0.11795 | 0.08437 | 0.00632 | 0.91685 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_short_ar_branch\seed_42\metrics_all_models.csv |
| no_short_ar_branch | 去除短窗自回归支路 | 52 | lstm | 0.11861 | 0.08455 | 0.00633 | 0.91592 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_short_ar_branch\seed_52\metrics_all_models.csv |
| no_short_ar_branch | 去除短窗自回归支路 | 62 | lstm | 0.12026 | 0.08720 | 0.00653 | 0.91355 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_short_ar_branch\seed_62\metrics_all_models.csv |
| no_short_ar_branch | 去除短窗自回归支路 | 72 | lstm | 0.11721 | 0.08293 | 0.00622 | 0.91789 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_short_ar_branch\seed_72\metrics_all_models.csv |
| no_short_ar_branch | 去除短窗自回归支路 | 82 | lstm | 0.11700 | 0.08394 | 0.00628 | 0.91818 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_short_ar_branch\seed_82\metrics_all_models.csv |
| no_diff_features | 去除一阶差分特征 | 42 | lstm | 0.11979 | 0.08662 | 0.00649 | 0.91423 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_diff_features\seed_42\metrics_all_models.csv |
| no_diff_features | 去除一阶差分特征 | 52 | lstm | 0.12046 | 0.08789 | 0.00658 | 0.91328 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_diff_features\seed_52\metrics_all_models.csv |
| no_diff_features | 去除一阶差分特征 | 62 | lstm | 0.12174 | 0.08676 | 0.00651 | 0.91142 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_diff_features\seed_62\metrics_all_models.csv |
| no_diff_features | 去除一阶差分特征 | 72 | lstm | 0.11836 | 0.08571 | 0.00642 | 0.91627 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_diff_features\seed_72\metrics_all_models.csv |
| no_diff_features | 去除一阶差分特征 | 82 | lstm | 0.11975 | 0.08704 | 0.00652 | 0.91429 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_diff_features\seed_82\metrics_all_models.csv |
| no_domain_physics_features | 去除物理派生特征 | 42 | lstm | 0.11867 | 0.08679 | 0.00650 | 0.91583 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_domain_physics_features\seed_42\metrics_all_models.csv |
| no_domain_physics_features | 去除物理派生特征 | 52 | lstm | 0.11752 | 0.08375 | 0.00627 | 0.91745 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_domain_physics_features\seed_52\metrics_all_models.csv |
| no_domain_physics_features | 去除物理派生特征 | 62 | lstm | 0.11838 | 0.08584 | 0.00642 | 0.91624 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_domain_physics_features\seed_62\metrics_all_models.csv |
| no_domain_physics_features | 去除物理派生特征 | 72 | lstm | 0.11702 | 0.08512 | 0.00638 | 0.91815 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_domain_physics_features\seed_72\metrics_all_models.csv |
| no_domain_physics_features | 去除物理派生特征 | 82 | lstm | 0.11834 | 0.08455 | 0.00633 | 0.91629 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_domain_physics_features\seed_82\metrics_all_models.csv |
| no_rolling_features | 去除滚动统计特征 | 42 | lstm | 0.11860 | 0.08636 | 0.00646 | 0.91593 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_rolling_features\seed_42\metrics_all_models.csv |
| no_rolling_features | 去除滚动统计特征 | 52 | lstm | 0.12052 | 0.08803 | 0.00659 | 0.91319 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_rolling_features\seed_52\metrics_all_models.csv |
| no_rolling_features | 去除滚动统计特征 | 62 | lstm | 0.11872 | 0.08492 | 0.00635 | 0.91576 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_rolling_features\seed_62\metrics_all_models.csv |
| no_rolling_features | 去除滚动统计特征 | 72 | lstm | 0.11952 | 0.08690 | 0.00652 | 0.91462 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_rolling_features\seed_72\metrics_all_models.csv |
| no_rolling_features | 去除滚动统计特征 | 82 | lstm | 0.11808 | 0.08462 | 0.00634 | 0.91666 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_rolling_features\seed_82\metrics_all_models.csv |
| no_solar_regime_features | 去除太阳状态特征 | 42 | lstm | 0.11903 | 0.08516 | 0.00637 | 0.91532 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_solar_regime_features\seed_42\metrics_all_models.csv |
| no_solar_regime_features | 去除太阳状态特征 | 52 | lstm | 0.12309 | 0.09172 | 0.00686 | 0.90945 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_solar_regime_features\seed_52\metrics_all_models.csv |
| no_solar_regime_features | 去除太阳状态特征 | 62 | lstm | 0.12089 | 0.09005 | 0.00673 | 0.91265 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_solar_regime_features\seed_62\metrics_all_models.csv |
| no_solar_regime_features | 去除太阳状态特征 | 72 | lstm | 0.11819 | 0.08470 | 0.00634 | 0.91651 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_solar_regime_features\seed_72\metrics_all_models.csv |
| no_solar_regime_features | 去除太阳状态特征 | 82 | lstm | 0.12043 | 0.08844 | 0.00662 | 0.91332 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_solar_regime_features\seed_82\metrics_all_models.csv |
| no_time_cycle_features | 去除时间周期特征 | 42 | lstm | 0.12226 | 0.08844 | 0.00665 | 0.91067 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_time_cycle_features\seed_42\metrics_all_models.csv |
| no_time_cycle_features | 去除时间周期特征 | 52 | lstm | 0.11977 | 0.08503 | 0.00638 | 0.91426 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_time_cycle_features\seed_52\metrics_all_models.csv |
| no_time_cycle_features | 去除时间周期特征 | 62 | lstm | 0.11686 | 0.08291 | 0.00622 | 0.91838 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_time_cycle_features\seed_62\metrics_all_models.csv |
| no_time_cycle_features | 去除时间周期特征 | 72 | lstm | 0.11995 | 0.08682 | 0.00651 | 0.91400 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_time_cycle_features\seed_72\metrics_all_models.csv |
| no_time_cycle_features | 去除时间周期特征 | 82 | lstm | 0.11652 | 0.08364 | 0.00627 | 0.91885 | delta | mean | outputs\strict_independent_ablation_20260501\runs\no_time_cycle_features\seed_82\metrics_all_models.csv |

## 建议插图

- `outputs\strict_independent_ablation_20260501\strict_ablation_rmse_delta.png`
- `outputs\strict_independent_ablation_20260501\strict_ablation_r2_delta.png`
