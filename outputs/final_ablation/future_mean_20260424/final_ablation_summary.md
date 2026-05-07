# Final Improved LSTM Ablation Summary

Generated at: 2026-04-24T21:17:38

## Design

All variants use the same chronological split and the same Sklavounos spring LogCn2 data. Variants 02-10 predict the 30-minute future mean of LogCn2 from a 12-hour input window; variant 01 is retained as a single-point target reference.

## Metrics

| variant                      | model        |     RMSE |       MAE |       MAPE |       R2 |   seq_len |   pred_len | target_mode   | target_aggregation   | attention_pool   |   ar_window |
|:-----------------------------|:-------------|---------:|----------:|-----------:|---------:|----------:|-----------:|:--------------|:---------------------|:-----------------|------------:|
| 01_single_point_final        | lstm         | 0.150873 | 0.109144  | 0.00818306 | 0.876867 |       144 |          1 | delta         | multistep            | True             |          12 |
| 02_future_mean_raw_absolute  | lstm         | 0.129861 | 0.0937375 | 0.00703646 | 0.899206 |       144 |          6 | absolute      | mean                 | True             |          12 |
| 03_future_mean_raw_delta     | lstm         | 0.12735  | 0.0908872 | 0.0068107  | 0.903067 |       144 |          6 | delta         | mean                 | True             |          12 |
| 04_future_mean_time_delta    | lstm         | 0.121606 | 0.0863326 | 0.00647968 | 0.911613 |       144 |          6 | delta         | mean                 | True             |          12 |
| 05_future_mean_physics_delta | lstm         | 0.123153 | 0.0882868 | 0.0066069  | 0.909351 |       144 |          6 | delta         | mean                 | True             |          12 |
| 06_future_mean_final         | lstm         | 0.119629 | 0.0856943 | 0.00641503 | 0.914464 |       144 |          6 | delta         | mean                 | True             |          12 |
| 07_final_no_attention        | lstm         | 0.11893  | 0.0862481 | 0.00646087 | 0.915461 |       144 |          6 | delta         | mean                 | False            |          12 |
| 08_final_no_ar_shortcut      | lstm         | 0.119114 | 0.0850006 | 0.00636373 | 0.9152   |       144 |          6 | delta         | mean                 | True             |           0 |
| 09_final_vanilla_lstm        | vanilla_lstm | 0.120797 | 0.0866308 | 0.0064927  | 0.912786 |       144 |          6 | delta         | mean                 | True             |          12 |
| 10_final_mlp_window          | mlp          | 0.136335 | 0.0992237 | 0.00744548 | 0.888905 |       144 |          6 | delta         | mean                 | True             |          12 |
| 11_final_no_attention_no_ar  | lstm         | 0.118511 | 0.0861381 | 0.00644782 | 0.916055 |       144 |          6 | delta         | mean                 | False            |           0 |
| 12_time_trend_delta          | lstm         | 0.118756 | 0.0846534 | 0.00634548 | 0.915708 |       144 |          6 | delta         | mean                 | True             |          12 |
| 13_time_trend_no_attention   | lstm         | 0.120548 | 0.0891169 | 0.00665675 | 0.913145 |       144 |          6 | delta         | mean                 | False            |          12 |
| 14_final_no_bidirectional    | lstm         | 0.120863 | 0.0885468 | 0.00663047 | 0.91269  |       144 |          6 | delta         | mean                 | True             |          12 |

## Best Variant

- Variant: `11_final_no_attention_no_ar`
- Model: `lstm`
- RMSE: `0.1185113647345189`
- R2: `0.9160548887247072`