import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - handled at runtime if xgboost is unavailable
    XGBRegressor = None


def build_last_step_feature_frame(x_seq: np.ndarray, scaler, feature_cols: list[str]) -> pd.DataFrame:
    if len(x_seq) == 0:
        return pd.DataFrame(columns=feature_cols)
    last_step = x_seq[:, -1, :]
    values = scaler.inverse_transform(last_step)
    return pd.DataFrame(values, columns=feature_cols)


def build_residual_feature_frames(
    prediction_inv_map: dict,
    last_feature_frames: dict[str, pd.DataFrame],
    val_regime: np.ndarray,
    test_regime: np.ndarray,
    residual_cfg: dict,
    pred_len: int,
):
    source_model = str(residual_cfg.get("source_model", "fusion_main"))
    include_models = residual_cfg.get("include_model_predictions") or [source_model]
    include_models = [name for name in include_models if name in prediction_inv_map]
    if source_model not in prediction_inv_map:
        raise ValueError(f"Residual correction source_model={source_model} not found in prediction map.")
    if not include_models:
        include_models = [source_model]

    include_last_features = residual_cfg.get("include_last_features") or []
    include_prediction_spread = bool(residual_cfg.get("include_prediction_spread", True))
    include_source_summary = bool(residual_cfg.get("include_source_summary", True))
    include_regime_id = bool(residual_cfg.get("include_regime_id", False))

    frames = {}
    regime_map = {"val": np.asarray(val_regime, dtype=np.int64), "test": np.asarray(test_regime, dtype=np.int64)}

    for split in ("val", "test"):
        frame = pd.DataFrame(index=np.arange(len(last_feature_frames[split])))
        last_frame = last_feature_frames[split]
        for col in include_last_features:
            if col in last_frame.columns:
                frame[col] = last_frame[col].astype(np.float32)

        for model_name in include_models:
            preds = np.asarray(prediction_inv_map[model_name][split], dtype=np.float32)
            for h in range(pred_len):
                frame[f"{model_name}_pred_h{h + 1}"] = preds[:, h]

        if include_prediction_spread and len(include_models) >= 2:
            for h in range(pred_len):
                stack = np.vstack([prediction_inv_map[name][split][:, h] for name in include_models]).T.astype(np.float32)
                frame[f"pred_mean_h{h + 1}"] = stack.mean(axis=1)
                frame[f"pred_std_h{h + 1}"] = stack.std(axis=1)
                frame[f"pred_min_h{h + 1}"] = stack.min(axis=1)
                frame[f"pred_max_h{h + 1}"] = stack.max(axis=1)

        if include_source_summary:
            source_pred = np.asarray(prediction_inv_map[source_model][split], dtype=np.float32)
            frame["source_pred_mean"] = source_pred.mean(axis=1)
            frame["source_pred_std"] = source_pred.std(axis=1)
            if pred_len > 1:
                frame["source_pred_span"] = source_pred[:, -1] - source_pred[:, 0]
                for h in range(pred_len - 1):
                    frame[f"source_pred_stepdiff_{h + 1}_{h + 2}"] = source_pred[:, h + 1] - source_pred[:, h]

        if include_regime_id:
            frame["regime_id"] = regime_map[split].astype(np.float32)

        frames[split] = frame.astype(np.float32)

    return frames, include_models


def build_residual_estimator(model_type: str, residual_cfg: dict, seed: int):
    model_type = str(model_type).lower()
    if model_type == "linear":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression(fit_intercept=bool(residual_cfg.get("fit_intercept", True)))),
            ]
        )
    if model_type == "ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=float(residual_cfg.get("ridge_alpha", 1.0)))),
            ]
        )
    if model_type == "hgb":
        hgb_cfg = residual_cfg.get("hgb", {})
        return HistGradientBoostingRegressor(
            max_depth=hgb_cfg.get("max_depth", 3),
            learning_rate=hgb_cfg.get("learning_rate", 0.05),
            max_iter=hgb_cfg.get("max_iter", 200),
            min_samples_leaf=hgb_cfg.get("min_samples_leaf", 20),
            l2_regularization=hgb_cfg.get("l2_regularization", 0.0),
            random_state=seed,
        )
    if model_type == "rf":
        rf_cfg = residual_cfg.get("rf", {})
        return RandomForestRegressor(
            n_estimators=rf_cfg.get("n_estimators", 300),
            max_depth=rf_cfg.get("max_depth", 5),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 5),
            random_state=seed,
            n_jobs=-1,
        )
    if model_type == "xgb":
        if XGBRegressor is None:
            raise ImportError("xgboost is not available, but residual model_type=xgb was requested.")
        xgb_cfg = residual_cfg.get("xgb", {})
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=xgb_cfg.get("n_estimators", 300),
            max_depth=xgb_cfg.get("max_depth", 3),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            subsample=xgb_cfg.get("subsample", 0.9),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            reg_alpha=xgb_cfg.get("reg_alpha", 0.0),
            reg_lambda=xgb_cfg.get("reg_lambda", 1.0),
            random_state=seed,
            n_jobs=xgb_cfg.get("n_jobs", 4),
        )
    raise ValueError(f"Unsupported residual model_type={model_type}")


def extract_feature_scores(estimator, feature_columns: list[str], top_k: int = 15):
    core = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
    if hasattr(core, "coef_"):
        values = np.asarray(core.coef_, dtype=np.float32).reshape(-1)
        score = np.abs(values)
        rows = [
            {"feature": feature_columns[i], "score": float(score[i]), "value": float(values[i])}
            for i in range(min(len(feature_columns), len(values)))
        ]
        rows.sort(key=lambda row: row["score"], reverse=True)
        return rows[:top_k]
    if hasattr(core, "feature_importances_"):
        values = np.asarray(core.feature_importances_, dtype=np.float32).reshape(-1)
        rows = [
            {"feature": feature_columns[i], "score": float(values[i])}
            for i in range(min(len(feature_columns), len(values)))
        ]
        rows.sort(key=lambda row: row["score"], reverse=True)
        return rows[:top_k]
    return []


def run_residual_postprocess(
    residual_cfg: dict,
    prediction_inv_map: dict,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_val_inv: np.ndarray,
    y_test_inv: np.ndarray,
    scaler,
    feature_cols: list[str],
    val_regime: np.ndarray,
    test_regime: np.ndarray,
    regime_names: list[str],
    output_dir: Path,
    cache_dir: Path,
    seed: int,
    logger,
):
    if not residual_cfg.get("enabled", False):
        return None

    alias = str(residual_cfg.get("alias", "residual_main"))
    cache_val = cache_dir / f"{alias}_val_pred_inv.npy"
    cache_test = cache_dir / f"{alias}_test_pred_inv.npy"
    model_path = output_dir / f"{alias}_models.joblib"
    summary_path = output_dir / f"{alias}_summary.json"

    if cache_val.exists() and cache_test.exists() and model_path.exists() and summary_path.exists():
        logger.info("Skip residual correction alias=%s (found cached corrected predictions).", alias)
        return {
            "alias": alias,
            "val_pred_inv": np.load(cache_val),
            "test_pred_inv": np.load(cache_test),
            "artifact_path": str(model_path),
            "summary_path": str(summary_path),
        }

    pred_len = y_val_inv.shape[1]
    source_model = str(residual_cfg.get("source_model", "fusion_main"))
    source_val = np.asarray(prediction_inv_map[source_model]["val"], dtype=np.float32)
    source_test = np.asarray(prediction_inv_map[source_model]["test"], dtype=np.float32)

    last_feature_frames = {
        "val": build_last_step_feature_frame(x_val, scaler, feature_cols),
        "test": build_last_step_feature_frame(x_test, scaler, feature_cols),
    }
    feature_frames, include_models = build_residual_feature_frames(
        prediction_inv_map=prediction_inv_map,
        last_feature_frames=last_feature_frames,
        val_regime=val_regime,
        test_regime=test_regime,
        residual_cfg=residual_cfg,
        pred_len=pred_len,
    )
    feature_columns = feature_frames["val"].columns.tolist()
    if not feature_columns:
        raise ValueError(f"Residual correction alias={alias} produced zero features.")

    x_val_res = feature_frames["val"].to_numpy(dtype=np.float32)
    x_test_res = feature_frames["test"].to_numpy(dtype=np.float32)
    split_by_regime = bool(residual_cfg.get("split_by_regime", False))
    min_samples_per_regime = int(residual_cfg.get("min_samples_per_regime", 64))
    model_type = str(residual_cfg.get("model_type", "ridge"))

    corrected_val = source_val.copy()
    corrected_test = source_test.copy()
    artifact = {
        "alias": alias,
        "source_model": source_model,
        "model_type": model_type,
        "include_models": include_models,
        "feature_columns": feature_columns,
        "split_by_regime": split_by_regime,
        "min_samples_per_regime": min_samples_per_regime,
        "horizons": [],
        "models": {},
    }

    unique_regimes = sorted(set(np.asarray(val_regime, dtype=np.int64).tolist()) | set(np.asarray(test_regime, dtype=np.int64).tolist()))
    regime_name_map = {
        int(regime_id): regime_names[regime_id] if 0 <= int(regime_id) < len(regime_names) else f"regime_{int(regime_id)}"
        for regime_id in unique_regimes
    }

    for h in range(pred_len):
        residual_target = (y_val_inv[:, h] - source_val[:, h]).astype(np.float32)
        horizon_models = {}
        horizon_summary = {
            "horizon": h + 1,
            "val_rmse_before": float(np.sqrt(np.mean((y_val_inv[:, h] - source_val[:, h]) ** 2))),
            "test_rmse_before": float(np.sqrt(np.mean((y_test_inv[:, h] - source_test[:, h]) ** 2))),
            "regimes": {},
            "top_features": [],
        }

        global_model = build_residual_estimator(model_type=model_type, residual_cfg=residual_cfg, seed=seed + h)
        global_model.fit(x_val_res, residual_target)
        global_val_residual = np.asarray(global_model.predict(x_val_res), dtype=np.float32)
        global_test_residual = np.asarray(global_model.predict(x_test_res), dtype=np.float32)
        corrected_val_h = source_val[:, h] + global_val_residual
        corrected_test_h = source_test[:, h] + global_test_residual
        horizon_models["global"] = global_model
        horizon_summary["top_features"] = extract_feature_scores(global_model, feature_columns)

        if split_by_regime:
            corrected_val_h = corrected_val_h.copy()
            corrected_test_h = corrected_test_h.copy()
            for regime_id in unique_regimes:
                train_mask = np.asarray(val_regime, dtype=np.int64) == regime_id
                test_mask = np.asarray(test_regime, dtype=np.int64) == regime_id
                if int(train_mask.sum()) < min_samples_per_regime:
                    horizon_summary["regimes"][regime_name_map[regime_id]] = {
                        "train_samples": int(train_mask.sum()),
                        "used_regime_model": False,
                    }
                    continue

                regime_model = build_residual_estimator(
                    model_type=model_type,
                    residual_cfg=residual_cfg,
                    seed=seed + h + int(regime_id) + 17,
                )
                regime_model.fit(x_val_res[train_mask], residual_target[train_mask])
                if train_mask.any():
                    corrected_val_h[train_mask] = source_val[train_mask, h] + np.asarray(
                        regime_model.predict(x_val_res[train_mask]),
                        dtype=np.float32,
                    )
                if test_mask.any():
                    corrected_test_h[test_mask] = source_test[test_mask, h] + np.asarray(
                        regime_model.predict(x_test_res[test_mask]),
                        dtype=np.float32,
                    )
                horizon_models[regime_name_map[regime_id]] = regime_model
                horizon_summary["regimes"][regime_name_map[regime_id]] = {
                    "train_samples": int(train_mask.sum()),
                    "used_regime_model": True,
                    "top_features": extract_feature_scores(regime_model, feature_columns, top_k=10),
                }

        corrected_val[:, h] = corrected_val_h.astype(np.float32)
        corrected_test[:, h] = corrected_test_h.astype(np.float32)
        horizon_summary["val_rmse_after"] = float(np.sqrt(np.mean((y_val_inv[:, h] - corrected_val[:, h]) ** 2)))
        horizon_summary["test_rmse_after"] = float(np.sqrt(np.mean((y_test_inv[:, h] - corrected_test[:, h]) ** 2)))
        artifact["horizons"].append(horizon_summary)
        artifact["models"][f"horizon_{h + 1}"] = horizon_models

    np.save(cache_val, corrected_val.astype(np.float32))
    np.save(cache_test, corrected_test.astype(np.float32))
    joblib.dump(artifact, model_path)

    summary_payload = {
        "alias": alias,
        "source_model": source_model,
        "model_type": model_type,
        "include_models": include_models,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "split_by_regime": split_by_regime,
        "horizons": artifact["horizons"],
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved residual correction artifacts: %s, %s", model_path, summary_path)

    return {
        "alias": alias,
        "val_pred_inv": corrected_val,
        "test_pred_inv": corrected_test,
        "artifact_path": str(model_path),
        "summary_path": str(summary_path),
    }
