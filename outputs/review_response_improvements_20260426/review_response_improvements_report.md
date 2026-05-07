# Review-response experiment additions

## Forecasting statistical evidence

| Comparison | Paired rows | MAE reduction | 95% CI | MSE reduction | 95% CI | Seeds improved |
|---|---:|---:|---:|---:|---:|---:|
| Improved LSTM vs LSTM-raw | 16155 | 0.01182 | [0.01083, 0.01283] | 0.00446 | [0.00406, 0.00487] | 5/5 |
| Improved LSTM vs TCN | 16280 | 0.00714 | [0.00632, 0.00794] | 0.00155 | [0.00128, 0.00182] | 5/5 |
| Improved LSTM vs TFT | 16280 | 0.00856 | [0.00768, 0.00946] | 0.00284 | [0.00251, 0.00319] | 5/5 |
| Improved LSTM vs PatchTST | 16280 | 0.01766 | [0.01648, 0.01887] | 0.00697 | [0.00635, 0.00758] | 5/5 |

## Link-policy bootstrap evidence

| Comparison | Utility gain | 95% CI | Outage reduction | 95% CI |
|---|---:|---:|---:|---:|
| Improved LSTM Policy vs LSTM-raw Policy | 0.01747 | [0.01321, 0.02219] | 0.742% | [0.629%, 0.859%] |
| Improved LSTM Policy vs TCN Policy | 0.00048 | [-0.00287, 0.00383] | -0.168% | [-0.233%, -0.105%] |
| Improved LSTM Policy vs TFT Policy | 0.00316 | [-0.00025, 0.00656] | -0.050% | [-0.126%, 0.021%] |
| Improved LSTM Policy vs PatchTST Policy | 0.00611 | [0.00234, 0.00990] | 0.032% | [-0.048%, 0.111%] |
| Improved LSTM Policy vs Persistence Policy | 0.01140 | [0.00745, 0.01555] | 0.054% | [-0.040%, 0.146%] |

## Sensitivity analysis

- Best average rank policy: Improved LSTM Policy (mean rank=1.26).
- Improved LSTM Policy: mean rank=1.26, best-policy count=23/27.

## Cross-site link adaptation

| Policy | Mission utility | Goodput | Outage | Availability |
|---|---:|---:|---:|---:|
| Oracle Policy | 2.6275 | 2.8295 | 2.53% | 97.47% |
| Persistence Policy | 2.5301 | 2.7794 | 3.12% | 96.88% |
| LSTM-raw Policy | 2.5231 | 2.8242 | 3.76% | 96.24% |
| Improved LSTM Policy | 2.5085 | 2.8136 | 3.81% | 96.19% |