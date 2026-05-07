# Current Route Strict Cross-Site Results

| variant       | model             |      RMSE |       MAE |      MAPE |           R2 |   RMSE_h1 |
|:--------------|:------------------|----------:|----------:|----------:|-------------:|----------:|
| lstm_raw      | lstm              |  0.273442 |  0.200371 | 0.0144134 |     0.854753 |  0.273442 |
| ann_window    | naive_persistence |  0.284592 |  0.194331 | 0.0139588 |     0.842665 |  0.284592 |
| lstm_raw      | naive_persistence |  0.284592 |  0.194331 | 0.0139588 |     0.842665 |  0.284592 |
| ann_current   | naive_moving_avg  |  0.297745 |  0.207128 | 0.0147914 |     0.840723 |  0.297745 |
| ann_current   | naive_persistence |  0.297745 |  0.207128 | 0.0147914 |     0.840723 |  0.297745 |
| lstm_improved | naive_persistence |  0.299816 |  0.207929 | 0.0148328 |     0.840395 |  0.299816 |
| lstm_improved | lstm              |  0.317083 |  0.228204 | 0.0162298 |     0.821481 |  0.317083 |
| ann_window    | naive_moving_avg  |  0.346712 |  0.245553 | 0.0176023 |     0.766484 |  0.346712 |
| lstm_raw      | naive_moving_avg  |  0.346712 |  0.245553 | 0.0176023 |     0.766484 |  0.346712 |
| lstm_improved | naive_moving_avg  |  0.35975  |  0.255923 | 0.0181696 |     0.770205 |  0.35975  |
| ann_current   | mlp               | 11.6941   | 11.6819   | 0.848121  |  -244.695    | 11.6941   |
| ann_window    | mlp               | 34.7685   | 34.7668   | 2.5518    | -2347.28     | 34.7685   |