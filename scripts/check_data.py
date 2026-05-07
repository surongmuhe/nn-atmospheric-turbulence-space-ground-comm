import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import read_source_dataframe, resolve_data_paths
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Check dataset columns/types for turbulence forecasting project")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    cfg = load_config(str(cfg_path))
    dcfg = cfg["data"]

    file_paths = resolve_data_paths(dcfg, ROOT)
    missing = [path for path in file_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {[str(path) for path in missing]}")

    frames = [read_source_dataframe(path) for path in file_paths]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    if len(file_paths) == 1:
        print(f"File: {file_paths[0]}")
    else:
        print("Files:")
        for path in file_paths:
            print(f"- {path}")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    for c in df.columns:
        print(f"- {c} ({df[c].dtype})")

    required = []
    if dcfg.get("datetime_col"):
        required.append(dcfg["datetime_col"])
    if dcfg.get("time_col"):
        required.append(dcfg["time_col"])
    required += list(dcfg.get("feature_cols", []))
    required = list(dict.fromkeys(required))

    derived_features = {"temporal_hour", "hour_sin", "hour_cos", "doy_sin", "doy_cos"}
    required = [c for c in required if c not in derived_features]

    missing = [c for c in required if c not in df.columns]
    print("\nRequired columns check:")
    if missing:
        print("Missing:", missing)
    else:
        print("OK (all present)")

    # Quick preview
    print("\nHead preview:")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()

