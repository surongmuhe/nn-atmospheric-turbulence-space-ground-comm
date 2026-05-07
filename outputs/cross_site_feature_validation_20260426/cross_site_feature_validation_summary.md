# Cross-Site Feature Validation

Generated at: 2026-04-26T18:18:31

This experiment checks whether the four-model cross-site comparison is sensitive to the selected shared input features.

## Primary Model Results

| profile          | variant       | model   |     RMSE |      MAE |      MAPE |       R2 |   RMSE_h1 |
|:-----------------|:--------------|:--------|---------:|---------:|----------:|---------:|----------:|
| portable_no_pair | ann_current   | mlp     | 0.311803 | 0.235911 | 0.0169948 | 0.825328 |  0.311803 |
| portable_no_pair | ann_window    | mlp     | 0.260287 | 0.187474 | 0.0134964 | 0.868392 |  0.260287 |
| portable_no_pair | lstm_raw      | lstm    | 0.27741  | 0.203394 | 0.0146429 | 0.850506 |  0.27741  |
| portable_no_pair | lstm_improved | lstm    | 0.308667 | 0.222057 | 0.0158014 | 0.830832 |  0.308667 |

## All Metrics

| profile          | variant       | model             |     RMSE |      MAE |      MAPE |       R2 |   RMSE_h1 |
|:-----------------|:--------------|:------------------|---------:|---------:|----------:|---------:|----------:|
| portable_no_pair | ann_window    | mlp               | 0.260287 | 0.187474 | 0.0134964 | 0.868392 |  0.260287 |
| portable_no_pair | lstm_raw      | lstm              | 0.27741  | 0.203394 | 0.0146429 | 0.850506 |  0.27741  |
| portable_no_pair | ann_window    | naive_persistence | 0.284592 | 0.194331 | 0.0139588 | 0.842665 |  0.284592 |
| portable_no_pair | lstm_raw      | naive_persistence | 0.284592 | 0.194331 | 0.0139588 | 0.842665 |  0.284592 |
| portable_no_pair | ann_current   | naive_moving_avg  | 0.297745 | 0.207128 | 0.0147914 | 0.840723 |  0.297745 |
| portable_no_pair | ann_current   | naive_persistence | 0.297745 | 0.207128 | 0.0147914 | 0.840723 |  0.297745 |
| portable_no_pair | lstm_improved | naive_persistence | 0.299816 | 0.207929 | 0.0148328 | 0.840395 |  0.299816 |
| portable_no_pair | lstm_improved | lstm              | 0.308667 | 0.222057 | 0.0158014 | 0.830832 |  0.308667 |
| portable_no_pair | ann_current   | mlp               | 0.311803 | 0.235911 | 0.0169948 | 0.825328 |  0.311803 |
| portable_no_pair | ann_window    | naive_moving_avg  | 0.346712 | 0.245553 | 0.0176023 | 0.766484 |  0.346712 |
| portable_no_pair | lstm_raw      | naive_moving_avg  | 0.346712 | 0.245553 | 0.0176023 | 0.766484 |  0.346712 |
| portable_no_pair | lstm_improved | naive_moving_avg  | 0.35975  | 0.255923 | 0.0181696 | 0.770205 |  0.35975  |