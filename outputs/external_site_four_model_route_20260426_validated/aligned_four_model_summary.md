# Aligned External-Site Four-Model Metrics

Generated at: 2026-04-26T19:11:37
Experiment root: `C:\Users\33640\Desktop\lstm\fusion_forecasting_project\outputs\external_site_four_model_route_20260426_validated`
Common timestamps: 399

Metrics below are recomputed only on timestamps shared by all variants.

| variant       |   n_common |    RMSE |     MAE |     MSE |    MAPE |      R2 |
|:--------------|-----------:|--------:|--------:|--------:|--------:|--------:|
| ann_current   |        399 | 0.23135 | 0.17797 | 0.05352 | 1.30659 | 0.84810 |
| ann_window    |        399 | 0.19696 | 0.14166 | 0.03879 | 1.04167 | 0.88991 |
| lstm_raw      |        399 | 0.18942 | 0.13854 | 0.03588 | 1.01493 | 0.89817 |
| lstm_improved |        399 | 0.15898 | 0.11496 | 0.02527 | 0.84233 | 0.92827 |

R2 order from low to high: `ann_current < ann_window < lstm_raw < lstm_improved`
Matches expected route order: `True`
