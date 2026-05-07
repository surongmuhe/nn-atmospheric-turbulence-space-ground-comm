import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the dataWfon0U cross-site dataset for the fusion forecasting pipeline.")
    parser.add_argument("--zip-path", required=True, help="Path to dataWfon0U.zip")
    parser.add_argument("--output-csv", required=True, help="Path to save the processed CSV")
    return parser.parse_args()


def build_daylight_flag(timestamp: pd.Series) -> pd.Series:
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    # No solar radiation variable exists in the site file, so use a conservative
    # local-hour proxy to keep the same day/night regime idea available.
    return ((hour >= 6.0) & (hour < 18.0)).astype(np.float32)


def main():
    args = parse_args()
    zip_path = Path(args.zip_path)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("dataWfon0U/isff_5min_ml.csv") as f:
            raw = pd.read_csv(f)

    out = pd.DataFrame()
    out["time"] = pd.to_datetime(raw["time"], errors="coerce")
    out["LogCn2"] = np.where(pd.to_numeric(raw["Cn2_15m"], errors="coerce") > 0, np.log10(raw["Cn2_15m"]), np.nan)
    out["Tair"] = pd.to_numeric(raw["T_2m"], errors="coerce")
    out["RH_2m"] = pd.to_numeric(raw["RH_2m"], errors="coerce")
    out["Pair"] = pd.to_numeric(raw["P_2m"], errors="coerce")
    out["Dir_10m"] = pd.to_numeric(raw["Dir_10m"], errors="coerce")
    out["UmSonic"] = pd.to_numeric(raw["Spd_10m"], errors="coerce")
    out["u_15m"] = pd.to_numeric(raw["u_15m"], errors="coerce")
    out["v_15m"] = pd.to_numeric(raw["v_15m"], errors="coerce")
    out["w_15m"] = pd.to_numeric(raw["w_15m"], errors="coerce")
    out["tc_15m"] = pd.to_numeric(raw["tc_15m"], errors="coerce")
    out["is_daylight"] = build_daylight_flag(out["time"])

    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "zip_path": str(zip_path),
        "output_csv": str(output_csv),
        "rows_total": int(len(out)),
        "rows_logcn2_valid": int(out["LogCn2"].notna().sum()),
        "time_begin": str(out["time"].min()),
        "time_end": str(out["time"].max()),
        "feature_columns": [col for col in out.columns if col != "time"],
        "target_source": "log10(Cn2_15m)",
        "daylight_proxy": "06:00 <= local hour < 18:00",
    }
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
