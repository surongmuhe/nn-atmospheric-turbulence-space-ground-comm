import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_sequences(
    data: np.ndarray,
    target_idx: int,
    seq_len: int,
    pred_len: int = 1,
    segment_ids: np.ndarray | None = None,
):
    x_list, y_list = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        if segment_ids is not None:
            seg = segment_ids[i : i + seq_len + pred_len]
            if seg.size == 0 or np.any(seg != seg[0]):
                continue
        x_list.append(data[i : i + seq_len])
        y_list.append(data[i + seq_len : i + seq_len + pred_len, target_idx])
    return np.array(x_list), np.array(y_list)


def create_sequences_with_context(
    data: np.ndarray,
    target_idx: int,
    seq_len: int,
    pred_len: int,
    train_end: int,
    val_end: int,
    segment_ids: np.ndarray | None = None,
):
    feature_count = data.shape[1]
    split_x = {"train": [], "val": [], "test": []}
    split_y = {"train": [], "val": [], "test": []}
    split_forecast_start = {"train": [], "val": [], "test": []}

    max_forecast_start = len(data) - pred_len
    for forecast_start in range(seq_len, max_forecast_start + 1):
        forecast_end = forecast_start + pred_len
        if segment_ids is not None:
            seg = segment_ids[forecast_start - seq_len : forecast_end]
            if seg.size == 0 or np.any(seg != seg[0]):
                continue
        x_seq = data[forecast_start - seq_len : forecast_start]
        y_seq = data[forecast_start:forecast_end, target_idx]

        if forecast_end <= train_end:
            split_name = "train"
        elif forecast_start >= train_end and forecast_end <= val_end:
            split_name = "val"
        elif forecast_start >= val_end and forecast_end <= len(data):
            split_name = "test"
        else:
            continue

        split_x[split_name].append(x_seq)
        split_y[split_name].append(y_seq)
        split_forecast_start[split_name].append(forecast_start)

    def finalize(name: str):
        if split_x[name]:
            x_arr = np.asarray(split_x[name], dtype=np.float32)
            y_arr = np.asarray(split_y[name], dtype=np.float32)
        else:
            x_arr = np.empty((0, seq_len, feature_count), dtype=np.float32)
            y_arr = np.empty((0, pred_len), dtype=np.float32)
        idx_arr = np.asarray(split_forecast_start[name], dtype=np.int64)
        return x_arr, y_arr, idx_arr

    return {
        "train": finalize("train"),
        "val": finalize("val"),
        "test": finalize("test"),
    }


def aggregate_target(y_seq: np.ndarray, mode: str):
    """
    y_seq: [N, pred_len]
    mode:
      - multistep: return [N, pred_len]
      - mean: return [N, 1] averaged target across horizon
    """
    if mode == "mean":
        return y_seq.mean(axis=1, keepdims=True)
    return y_seq


def encode_target_values(y_abs: np.ndarray, last_target: np.ndarray, target_mode: str):
    if target_mode == "delta":
        return y_abs - last_target
    return y_abs


def decode_target_values(y_model: np.ndarray, last_target: np.ndarray, target_mode: str):
    if target_mode == "delta":
        return y_model + last_target
    return y_model


def add_time_features(df: pd.DataFrame, time_cfg: dict | None) -> pd.DataFrame:
    time_cfg = time_cfg or {}
    out = df.copy()
    hour = out.index.hour + out.index.minute / 60.0 + out.index.second / 3600.0
    if time_cfg.get("keep_temporal_hour", False):
        out["temporal_hour"] = hour.astype(np.float32)
    if time_cfg.get("hour_cyclical", True):
        out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
        out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    if time_cfg.get("dayofyear_cyclical", False):
        dayofyear = out.index.dayofyear.astype(np.float32)
        out["doy_sin"] = np.sin(2.0 * np.pi * dayofyear / 365.25).astype(np.float32)
        out["doy_cos"] = np.cos(2.0 * np.pi * dayofyear / 365.25).astype(np.float32)
    return out


def add_season_features(df: pd.DataFrame, season_cfg: dict | None):
    season_cfg = season_cfg or {}
    if not season_cfg.get("enabled", False):
        return df, []

    out = df.copy()
    added_cols = []
    month = out.index.month.astype(np.int32)
    season_id = np.select(
        [
            month.isin([12, 1, 2]),
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8]),
            month.isin([9, 10, 11]),
        ],
        [0, 1, 2, 3],
        default=0,
    ).astype(np.int32)
    season_names = ["winter", "spring", "summer", "autumn"]

    if season_cfg.get("include_season_id", True):
        out["season_id"] = season_id.astype(np.float32)
        added_cols.append("season_id")

    if season_cfg.get("season_one_hot", False):
        for idx, name in enumerate(season_names):
            col = f"season_{name}"
            out[col] = (season_id == idx).astype(np.float32)
            added_cols.append(col)

    if season_cfg.get("month_cyclical", True):
        month_zero = month.astype(np.float32) - 1.0
        out["month_sin"] = np.sin(2.0 * np.pi * month_zero / 12.0).astype(np.float32)
        out["month_cos"] = np.cos(2.0 * np.pi * month_zero / 12.0).astype(np.float32)
        added_cols.extend(["month_sin", "month_cos"])

    if season_cfg.get("include_season_day_regime", True) and "is_daylight" in out.columns:
        day_flag = out["is_daylight"].fillna(0.0).round().clip(0.0, 1.0).astype(np.int32)
        out["season_day_regime"] = (season_id * 2 + day_flag).astype(np.float32)
        added_cols.append("season_day_regime")

    return out, added_cols


def add_missing_indicator_features(df: pd.DataFrame, missing_ref: pd.DataFrame, miss_cfg: dict | None):
    miss_cfg = miss_cfg or {}
    if not miss_cfg.get("enabled", False):
        return df, []

    out = df.copy()
    columns = miss_cfg.get("columns") or []
    added_cols = []
    for col in columns:
        if col not in missing_ref.columns:
            continue
        new_col = f"{col}_missing_flag"
        out[new_col] = missing_ref[col].astype(np.float32)
        added_cols.append(new_col)
    return out, added_cols


def compute_missing_run_lengths(missing_ref: pd.DataFrame) -> pd.DataFrame:
    run_df = pd.DataFrame(index=missing_ref.index)
    for col in missing_ref.columns:
        mask = missing_ref[col].fillna(False).astype(bool)
        group_id = mask.ne(mask.shift(fill_value=False)).cumsum()
        run_df[col] = mask.astype(np.int32).groupby(group_id).cumsum().astype(np.float32)
    return run_df


def add_missing_context_features(
    df: pd.DataFrame,
    missing_ref: pd.DataFrame,
    missing_run_lengths: pd.DataFrame,
    miss_ctx_cfg: dict | None,
):
    miss_ctx_cfg = miss_ctx_cfg or {}
    if not miss_ctx_cfg.get("enabled", False):
        return df, []

    out = df.copy()
    columns = [col for col in (miss_ctx_cfg.get("columns") or []) if col in missing_ref.columns]
    if not columns:
        return out, []

    added_cols = []
    missing_subset = missing_ref[columns].astype(np.float32)
    run_subset = missing_run_lengths[columns].astype(np.float32)

    if miss_ctx_cfg.get("include_any_missing", True):
        out["any_sensor_missing"] = (missing_subset.max(axis=1) > 0).astype(np.float32)
        added_cols.append("any_sensor_missing")

    if miss_ctx_cfg.get("include_missing_fraction", True):
        out["sensor_missing_fraction"] = missing_subset.mean(axis=1).astype(np.float32)
        added_cols.append("sensor_missing_fraction")

    if miss_ctx_cfg.get("include_max_run_log1p", True):
        out["sensor_missing_run_log1p"] = np.log1p(run_subset.max(axis=1)).astype(np.float32)
        added_cols.append("sensor_missing_run_log1p")

    if miss_ctx_cfg.get("include_recent_missing", True):
        windows = [int(x) for x in (miss_ctx_cfg.get("recent_windows") or [3, 12])]
        missing_fraction = missing_subset.mean(axis=1)
        for window in windows:
            new_col = f"sensor_missing_recent_{window}"
            out[new_col] = missing_fraction.rolling(window=window, min_periods=1).mean().astype(np.float32)
            added_cols.append(new_col)

    return out, added_cols


def add_domain_features(df: pd.DataFrame, domain_cfg: dict | None):
    domain_cfg = domain_cfg or {}
    if not domain_cfg.get("enabled", False):
        return df, []

    out = df.copy()
    added_cols = []

    if {"Tair", "IRtemp"}.issubset(out.columns):
        out["Tair_IRtemp_gap"] = (out["Tair"] - out["IRtemp"]).astype(np.float32)
        added_cols.append("Tair_IRtemp_gap")

    if "SolarFlux" in out.columns:
        solar_nonneg = out["SolarFlux"].clip(lower=0.0).astype(np.float32)
        out["SolarFlux_nonneg"] = solar_nonneg
        added_cols.append("SolarFlux_nonneg")

        threshold = float(domain_cfg.get("solar_day_threshold", 20.0))
        out["is_daylight"] = (solar_nonneg > threshold).astype(np.float32)
        added_cols.append("is_daylight")

        out["SolarFlux_log1p"] = np.log1p(solar_nonneg).astype(np.float32)
        added_cols.append("SolarFlux_log1p")

    if "UmSonic" in out.columns:
        out["UmSonic_sq"] = np.square(out["UmSonic"]).astype(np.float32)
        added_cols.append("UmSonic_sq")

    if {"Pout", "Predicted Pout"}.issubset(out.columns):
        pred_missing = out["Predicted Pout"].isna()
        pred_pout = out["Predicted Pout"].astype(np.float32)
        residual = (out["Pout"] - pred_pout).astype(np.float32)
        out["Pout_residual"] = residual.fillna(0.0).astype(np.float32)
        added_cols.append("Pout_residual")
        out["PredictedPout_missing_flag"] = pred_missing.astype(np.float32)
        added_cols.append("PredictedPout_missing_flag")

    return out, added_cols


def add_solar_regime_features(df: pd.DataFrame, regime_cfg: dict | None):
    regime_cfg = regime_cfg or {}
    if not regime_cfg.get("enabled", False):
        return df, []

    out = df.copy()
    added_cols = []
    if "SolarFlux_nonneg" in out.columns:
        solar = out["SolarFlux_nonneg"]
    elif "SolarFlux" in out.columns:
        solar = out["SolarFlux"].clip(lower=0.0).astype(np.float32)
    else:
        return out, []

    twilight_low = float(regime_cfg.get("twilight_low", 5.0))
    twilight_high = float(regime_cfg.get("twilight_high", 120.0))
    strong_sun_threshold = float(regime_cfg.get("strong_sun_threshold", 250.0))

    out["is_night_regime"] = (solar <= twilight_low).astype(np.float32)
    out["is_twilight_regime"] = ((solar > twilight_low) & (solar <= twilight_high)).astype(np.float32)
    out["is_strong_sun"] = (solar > strong_sun_threshold).astype(np.float32)
    added_cols.extend(["is_night_regime", "is_twilight_regime", "is_strong_sun"])

    transition_absdiff = solar.diff().abs().fillna(0.0).astype(np.float32)
    out["solar_transition_absdiff"] = transition_absdiff
    out["solar_transition_log1p"] = np.log1p(transition_absdiff).astype(np.float32)
    added_cols.extend(["solar_transition_absdiff", "solar_transition_log1p"])

    recent_windows = [int(x) for x in (regime_cfg.get("recent_windows") or [6, 12])]
    for window in recent_windows:
        twilight_recent_col = f"twilight_recent_{window}"
        daylight_recent_col = f"daylight_recent_{window}"
        out[twilight_recent_col] = out["is_twilight_regime"].rolling(window=window, min_periods=1).mean().astype(
            np.float32
        )
        out[daylight_recent_col] = (solar > twilight_low).astype(np.float32).rolling(window=window, min_periods=1).mean().astype(
            np.float32
        )
        added_cols.extend([twilight_recent_col, daylight_recent_col])

    if {"UmSonic", "Tair_IRtemp_gap"}.issubset(out.columns):
        out["thermal_wind_coupling"] = (out["UmSonic"] * out["Tair_IRtemp_gap"]).astype(np.float32)
        added_cols.append("thermal_wind_coupling")

    return out, added_cols


def add_diff_features(df: pd.DataFrame, diff_cfg: dict | None):
    diff_cfg = diff_cfg or {}
    if not diff_cfg.get("enabled", False):
        return df, []

    out = df.copy()
    columns = diff_cfg.get("columns") or []
    periods = [int(x) for x in (diff_cfg.get("periods") or [1])]
    added_cols = []

    for col in columns:
        if col not in out.columns:
            continue
        for period in periods:
            new_col = f"{col}_diff_{period}"
            out[new_col] = out[col].diff(period).fillna(0.0).astype(np.float32)
            added_cols.append(new_col)

    return out, added_cols


def add_rolling_features(df: pd.DataFrame, rolling_cfg: dict | None):
    rolling_cfg = rolling_cfg or {}
    if not rolling_cfg.get("enabled", False):
        return df, []

    out = df.copy()
    columns = rolling_cfg.get("columns") or []
    windows = [int(x) for x in (rolling_cfg.get("windows") or [3])]
    include_mean = bool(rolling_cfg.get("include_mean", True))
    include_std = bool(rolling_cfg.get("include_std", True))
    include_min = bool(rolling_cfg.get("include_min", False))
    include_max = bool(rolling_cfg.get("include_max", False))
    added_cols = []

    for col in columns:
        if col not in out.columns:
            continue
        series = out[col]
        for window in windows:
            rolling = series.rolling(window=window, min_periods=1)
            if include_mean:
                new_col = f"{col}_roll_mean_{window}"
                out[new_col] = rolling.mean().astype(np.float32)
                added_cols.append(new_col)
            if include_std:
                new_col = f"{col}_roll_std_{window}"
                out[new_col] = rolling.std().fillna(0.0).astype(np.float32)
                added_cols.append(new_col)
            if include_min:
                new_col = f"{col}_roll_min_{window}"
                out[new_col] = rolling.min().astype(np.float32)
                added_cols.append(new_col)
            if include_max:
                new_col = f"{col}_roll_max_{window}"
                out[new_col] = rolling.max().astype(np.float32)
                added_cols.append(new_col)

    return out, added_cols


def clip_by_train_quantiles(
    feat_df: pd.DataFrame,
    n_train: int,
    clip_cfg: dict | None,
    feature_cols: list[str],
) -> pd.DataFrame:
    clip_cfg = clip_cfg or {}
    if not clip_cfg.get("enabled", False):
        return feat_df

    lower_q = float(clip_cfg.get("lower", 0.005))
    upper_q = float(clip_cfg.get("upper", 0.995))
    exclude_cols = set(clip_cfg.get("exclude_cols", []))

    clipped = feat_df.copy()
    train_part = clipped.iloc[:n_train]
    for col in feature_cols:
        if col in exclude_cols:
            continue
        lower = float(train_part[col].quantile(lower_q))
        upper = float(train_part[col].quantile(upper_q))
        clipped[col] = clipped[col].clip(lower=lower, upper=upper)
    return clipped


def build_effective_feature_frame(df: pd.DataFrame, dcfg: dict, preprocess_cfg: dict):
    feature_cols = list(dcfg["feature_cols"])
    out = df.copy()
    missing_ref = preprocess_cfg.get("_missing_reference", pd.DataFrame(index=out.index))
    missing_run_lengths = preprocess_cfg.get("_missing_run_lengths", pd.DataFrame(index=out.index))

    out, missing_cols = add_missing_indicator_features(
        out,
        missing_ref=missing_ref,
        miss_cfg=preprocess_cfg.get("missing_indicators"),
    )
    feature_cols.extend(missing_cols)

    out, missing_context_cols = add_missing_context_features(
        out,
        missing_ref=missing_ref,
        missing_run_lengths=missing_run_lengths,
        miss_ctx_cfg=preprocess_cfg.get("missing_context_features"),
    )
    feature_cols.extend(missing_context_cols)

    out, domain_cols = add_domain_features(out, preprocess_cfg.get("domain_features"))
    feature_cols.extend(domain_cols)

    out, season_cols = add_season_features(out, preprocess_cfg.get("season_features"))
    feature_cols.extend(season_cols)

    out, solar_regime_cols = add_solar_regime_features(out, preprocess_cfg.get("solar_regime_features"))
    feature_cols.extend(solar_regime_cols)

    out, diff_cols = add_diff_features(out, preprocess_cfg.get("diff_features"))
    feature_cols.extend(diff_cols)

    out, rolling_cols = add_rolling_features(out, preprocess_cfg.get("rolling_features"))
    feature_cols.extend(rolling_cols)

    feature_cols = list(dict.fromkeys(feature_cols))
    return out, feature_cols


def apply_long_gap_mask(
    df: pd.DataFrame,
    missing_ref: pd.DataFrame,
    missing_run_lengths: pd.DataFrame,
    gap_cfg: dict | None,
):
    gap_cfg = gap_cfg or {}
    if not gap_cfg.get("enabled", False):
        return df, pd.Series(False, index=df.index)

    columns = [col for col in (gap_cfg.get("columns") or []) if col in df.columns and col in missing_ref.columns]
    if not columns:
        return df, pd.Series(False, index=df.index)

    max_run = int(gap_cfg.get("max_missing_run", 6))
    pad_steps = int(gap_cfg.get("pad_steps", 0))
    out = df.copy()
    total_mask = pd.Series(False, index=df.index)

    for col in columns:
        invalid = missing_ref[col].astype(bool) & (missing_run_lengths[col] > max_run)
        expanded = invalid.copy()
        for step in range(1, pad_steps + 1):
            expanded |= invalid.shift(step, fill_value=False)
            expanded |= invalid.shift(-step, fill_value=False)
        out.loc[expanded, col] = np.nan
        total_mask |= expanded

    return out, total_mask


def build_segment_ids(index: pd.Index, resample_rule: str | None) -> np.ndarray:
    if len(index) == 0:
        return np.empty((0,), dtype=np.int64)
    if not resample_rule:
        return np.zeros((len(index),), dtype=np.int64)

    expected_delta = pd.to_timedelta(resample_rule)
    diffs = pd.Series(index).diff()
    breaks = diffs.ne(expected_delta).fillna(True)
    return breaks.cumsum().to_numpy(dtype=np.int64)


def build_regime_labels(df: pd.DataFrame, forecast_start: np.ndarray, regime_cfg: dict | None):
    regime_cfg = regime_cfg or {}
    if len(forecast_start) == 0:
        return np.empty((0,), dtype=np.int64), ["all"]

    strategy = str(regime_cfg.get("strategy", "binary_threshold")).lower()
    regime_feature = str(regime_cfg.get("regime_feature", "is_daylight"))
    if regime_feature not in df.columns:
        return np.zeros((len(forecast_start),), dtype=np.int64), ["all"]

    values = df[regime_feature].to_numpy()[forecast_start]
    if strategy in {"feature_bins", "solar_bins"}:
        bins = [float(x) for x in (regime_cfg.get("bins") or [0.5])]
        labels = regime_cfg.get("labels")
        class_ids = np.digitize(values, bins=bins, right=False).astype(np.int64)
        default_names = [f"regime_{i}" for i in range(len(bins) + 1)]
        regime_names = labels if labels and len(labels) == len(default_names) else default_names
        return class_ids, regime_names

    threshold = float(regime_cfg.get("threshold", 0.5))
    labels = regime_cfg.get("labels") or ["night", "day"]
    regime_names = labels if len(labels) >= 2 else ["night", "day"]
    return (values > threshold).astype(np.int64), regime_names[:2]


def resolve_data_paths(dcfg: dict, project_root: Path) -> list[Path]:
    raw_paths = dcfg.get("file_paths")
    if raw_paths:
        entries = list(raw_paths)
    else:
        entries = [dcfg["file_path"]]

    resolved = []
    for entry in entries:
        path = Path(entry)
        if not path.is_absolute():
            path = (project_root / path).resolve()
        resolved.append(path)
    return resolved


def read_source_dataframe(file_path: Path) -> pd.DataFrame:
    if str(file_path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    alias_map = {
        "Predicted": "Predicted Pout",
        "Predicted Pout": "Predicted Pout",
        "Log Pout": "LogPout",
        "LogPout": "LogPout",
    }
    normalized_cols = [alias_map.get(str(col).strip(), str(col).strip()) for col in df.columns]
    df.columns = normalized_cols
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).first().T
    return df


def build_dataset(cfg: dict):
    dcfg = cfg["data"]
    preprocess_cfg = cfg.get("preprocess", {})
    project_root = Path(__file__).resolve().parents[1]
    file_paths = resolve_data_paths(dcfg, project_root)
    missing_paths = [path for path in file_paths if not path.exists()]
    if missing_paths:
        data_dir = project_root / "data"
        candidates = []
        if data_dir.exists():
            candidates = sorted([p.name for p in data_dir.iterdir() if p.is_file()])
        raise FileNotFoundError(
            f"Data files do not exist: {[str(path) for path in missing_paths]}\n"
            f"Please check data.file_path or data.file_paths in the config.\n"
            f"Files under {str(data_dir)}: {candidates}"
        )

    frames = []
    for source_path in file_paths:
        part = read_source_dataframe(source_path)
        part["_source_file"] = source_path.name
        frames.append(part)
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    dt_col = dcfg.get("datetime_col", "Date")
    tm_col = dcfg.get("time_col")
    if tm_col and tm_col in df.columns:
        datetime_str = df[dt_col].astype(str).str[:10] + " " + df[tm_col].astype(str)
        try:
            dt = pd.to_datetime(datetime_str, format="mixed", errors="coerce")
        except Exception:
            dt = pd.to_datetime(datetime_str, errors="coerce")
        if dt.isna().any():
            bad_examples = datetime_str[dt.isna()].head(5).tolist()
            raise ValueError(f"Found unparseable datetime strings, examples: {bad_examples}")
        df["datetime"] = dt
    else:
        df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")
        if df["datetime"].isna().any():
            raise ValueError(f"Column {dt_col} contains unparseable datetime values.")

    df = df.dropna(subset=["datetime"]).copy()

    status_col = preprocess_cfg.get("status_col")
    status_keep_values = preprocess_cfg.get("status_keep_values")
    if status_col and status_col in df.columns and status_keep_values is not None:
        keep_values = set(status_keep_values)
        df = df[df[status_col].isin(keep_values)].copy()

    df = df.sort_values("datetime").set_index("datetime")
    df = df.drop(columns=[c for c in [dt_col, tm_col] if c and c in df.columns], errors="ignore")
    if preprocess_cfg.get("drop_duplicate_timestamps", True):
        df = df.groupby(level=0).mean(numeric_only=True)

    resample_rule = dcfg.get("resample_rule")
    if resample_rule:
        df = df.resample(resample_rule).mean(numeric_only=True)

    missing_reference = df.isna().copy()
    missing_run_lengths = compute_missing_run_lengths(missing_reference)
    interpolate_cfg = preprocess_cfg.get("interpolate", {})
    if interpolate_cfg.get("enabled", True):
        method = interpolate_cfg.get("method", "time")
        limit = interpolate_cfg.get("limit")
        df = df.interpolate(method=method, limit=limit, limit_direction="both")

    df, long_gap_mask = apply_long_gap_mask(
        df,
        missing_ref=missing_reference,
        missing_run_lengths=missing_run_lengths,
        gap_cfg=preprocess_cfg.get("long_gap_mask"),
    )

    df = add_time_features(df, preprocess_cfg.get("time_features"))
    preprocess_cfg = {
        **preprocess_cfg,
        "_missing_reference": missing_reference,
        "_missing_run_lengths": missing_run_lengths,
    }
    df, feature_cols = build_effective_feature_frame(df, dcfg, preprocess_cfg)
    df = df.dropna(subset=feature_cols).copy()
    if long_gap_mask.any():
        long_gap_mask = long_gap_mask.reindex(df.index, fill_value=False)

    n_total = len(df)
    n_train = int(n_total * dcfg["train_ratio"])
    n_val = int(n_total * dcfg["val_ratio"])

    feat_df = clip_by_train_quantiles(
        df[feature_cols].copy(),
        n_train=n_train,
        clip_cfg=preprocess_cfg.get("clip_quantiles"),
        feature_cols=feature_cols,
    )
    values = feat_df.values
    target_idx = feature_cols.index(dcfg["target_col"])

    train_raw = values[:n_train]
    val_raw = values[n_train : n_train + n_val]
    test_raw = values[n_train + n_val :]

    scaler_name = str(preprocess_cfg.get("scaler", "standard")).lower()
    if scaler_name == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    scaler.fit(train_raw)

    train_scaled = scaler.transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    all_scaled = scaler.transform(values)
    segment_ids = build_segment_ids(df.index, resample_rule)

    seq_len = dcfg["seq_len"]
    pred_len = dcfg["pred_len"]
    split_cfg = preprocess_cfg.get("sequence_split", {})
    if split_cfg.get("use_context_history", True):
        split_seq = create_sequences_with_context(
            all_scaled,
            target_idx=target_idx,
            seq_len=seq_len,
            pred_len=pred_len,
            train_end=n_train,
            val_end=n_train + n_val,
            segment_ids=segment_ids,
        )
        x_train, y_train, train_forecast_start = split_seq["train"]
        x_val, y_val, val_forecast_start = split_seq["val"]
        x_test, y_test, test_forecast_start = split_seq["test"]
    else:
        train_segment_ids = segment_ids[:n_train]
        val_segment_ids = segment_ids[n_train : n_train + n_val]
        test_segment_ids = segment_ids[n_train + n_val :]
        x_train, y_train = create_sequences(train_scaled, target_idx, seq_len, pred_len, segment_ids=train_segment_ids)
        x_val, y_val = create_sequences(val_scaled, target_idx, seq_len, pred_len, segment_ids=val_segment_ids)
        x_test, y_test = create_sequences(test_scaled, target_idx, seq_len, pred_len, segment_ids=test_segment_ids)
        train_forecast_start = np.arange(seq_len, seq_len + len(y_train), dtype=np.int64)
        val_forecast_start = np.arange(n_train + seq_len, n_train + seq_len + len(y_val), dtype=np.int64)
        test_forecast_start = np.arange(n_train + n_val + seq_len, n_train + n_val + seq_len + len(y_test), dtype=np.int64)

    agg_mode = dcfg.get("target_aggregation", "multistep")
    target_mode = str(dcfg.get("target_mode", "absolute")).lower()
    y_train_abs = aggregate_target(y_train, agg_mode).astype(np.float32)
    y_val_abs = aggregate_target(y_val, agg_mode).astype(np.float32)
    y_test_abs = aggregate_target(y_test, agg_mode).astype(np.float32)

    train_last_target = x_train[:, -1, target_idx : target_idx + 1].astype(np.float32)
    val_last_target = x_val[:, -1, target_idx : target_idx + 1].astype(np.float32)
    test_last_target = x_test[:, -1, target_idx : target_idx + 1].astype(np.float32)

    y_train = encode_target_values(y_train_abs, train_last_target, target_mode).astype(np.float32)
    y_val = encode_target_values(y_val_abs, val_last_target, target_mode).astype(np.float32)
    y_test = encode_target_values(y_test_abs, test_last_target, target_mode).astype(np.float32)

    # Some academically useful baselines predict the target from exogenous inputs only.
    # In that setting we still need the target column to exist inside the scaled frame so
    # sequence targets and inverse transforms remain well-defined, but we should remove
    # the target channel from the model input tensor after target construction.
    if bool(preprocess_cfg.get("exclude_target_from_inputs", False)):
        x_train = np.delete(x_train, target_idx, axis=2)
        x_val = np.delete(x_val, target_idx, axis=2)
        x_test = np.delete(x_test, target_idx, axis=2)

    all_index = pd.Index(df.index)
    test_dates = pd.Series(all_index[test_forecast_start]).reset_index(drop=True)
    regime_cfg = cfg.get("experiment", {}).get("regime_split", {}) or {}
    train_regime, regime_names = build_regime_labels(df, train_forecast_start, regime_cfg)
    val_regime, _ = build_regime_labels(df, val_forecast_start, regime_cfg)
    test_regime, _ = build_regime_labels(df, test_forecast_start, regime_cfg)

    return {
        "df": df,
        "scaler": scaler,
        "target_idx": target_idx,
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "agg_mode": agg_mode,
        "target_mode": target_mode,
        "n_train_raw": n_train,
        "n_val_raw": n_val,
        "train_forecast_start": train_forecast_start,
        "val_forecast_start": val_forecast_start,
        "test_forecast_start": test_forecast_start,
        "train_last_target": train_last_target,
        "val_last_target": val_last_target,
        "test_last_target": test_last_target,
        "y_train_abs": y_train_abs,
        "y_val_abs": y_val_abs,
        "y_test_abs": y_test_abs,
        "train_regime": train_regime,
        "val_regime": val_regime,
        "test_regime": test_regime,
        "regime_names": regime_names,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "test_dates": test_dates,
        "test_scaled_raw": test_scaled,
    }


def inverse_target(values: np.ndarray, scaler: StandardScaler, target_idx: int, feature_count: int):
    flat = values.reshape(-1)
    holder = np.zeros((len(flat), feature_count))
    holder[:, target_idx] = flat
    return scaler.inverse_transform(holder)[:, target_idx].reshape(values.shape)
