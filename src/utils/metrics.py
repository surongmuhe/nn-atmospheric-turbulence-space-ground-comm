import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    eps = 1e-8
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps)))
    r2 = r2_score(y_true, y_pred)
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "R2": float(r2),
    }
