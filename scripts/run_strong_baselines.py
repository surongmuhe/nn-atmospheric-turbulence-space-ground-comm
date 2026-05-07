import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CFG_PATH = ROOT / "configs" / "thesis_lstm_final_best_seq72.yaml"
RAW_LSTM_CFG_PATH = ROOT / "configs" / "thesis_lstm_raw_fair_seq72.yaml"
DEFAULT_SEEDS = [42, 52, 62, 72, 82]


def run_command(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def make_baseline_cfg(model_name: str, output_dir: Path) -> dict:
    cfg = deepcopy(load_yaml(BASE_CFG_PATH))
    cfg["project"]["name"] = f"strong_baseline_{model_name}_seq72"
    cfg["project"]["output_dir"] = str(output_dir)
    cfg["experiment"]["primary_model"] = model_name
    cfg["experiment"]["active_models"] = [model_name]
    cfg["fusion"]["enabled"] = False
    cfg["postprocess"]["residual_correction"]["enabled"] = False
    cfg["train"]["show_progress"] = False
    cfg["train"]["epochs"] = 35
    cfg["train"]["early_stop_patience"] = 8
    cfg["train"]["lr_scale_map"] = {
        "tcn": 0.8,
        "tft": 0.8,
        "patchtst": 0.7,
        "lstm": 1.0,
    }
    cfg["models"].setdefault("tcn", {})
    cfg["models"]["tcn"].update({"channels": [96, 96, 128], "kernel_size": 5, "dropout": 0.15})
    cfg["models"]["tft"] = {
        "hidden_size": 128,
        "lstm_layers": 1,
        "num_heads": 4,
        "dropout": 0.1,
    }
    cfg["models"]["patchtst"] = {
        "patch_len": 6,
        "stride": 3,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "ff_dim": 256,
        "dropout": 0.1,
        "pooling": "mean",
        "max_patches": 128,
    }
    return cfg


def make_improved_cfg(output_dir: Path) -> dict:
    cfg = deepcopy(load_yaml(BASE_CFG_PATH))
    cfg["project"]["name"] = "improved_lstm_seq72_seed_sweep"
    cfg["project"]["output_dir"] = str(output_dir)
    cfg["train"]["show_progress"] = False
    return cfg


def make_raw_lstm_cfg(output_dir: Path) -> dict:
    cfg = deepcopy(load_yaml(RAW_LSTM_CFG_PATH))
    cfg["project"]["name"] = "lstm_raw_seq72_seed_sweep"
    cfg["project"]["output_dir"] = str(output_dir)
    cfg["train"]["show_progress"] = False
    return cfg


def run_smoke_test(cfg_path: Path, output_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_experiments.py"),
        "--config",
        str(cfg_path),
        "--output-dir",
        str(output_dir),
        "--epochs",
        "1",
    ]
    run_command(cmd)


def run_seed_sweep(cfg_path: Path, output_root: Path, seeds: list[int], focus_model: str) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_seed_sweep.py"),
        "--config",
        str(cfg_path),
        "--output-root",
        str(output_root),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--focus-models",
        focus_model,
    ]
    run_command(cmd)


def summarize_runs(root: Path, run_names: list[str]) -> pd.DataFrame:
    rows = []
    for run_name in run_names:
        summary_path = root / run_name / "seed_summary_by_model.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        df = df[df["model"].isin(["lstm", "tcn", "tft", "patchtst"])]
        if df.empty:
            continue
        best = df.sort_values("RMSE_mean", ascending=True).iloc[0].to_dict()
        best["run_name"] = run_name
        rows.append(best)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strong baselines and 5-seed summaries on the seq72 main task.")
    parser.add_argument("--output-root", type=str, default="outputs/strong_baselines_seq72_20260425")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["lstm_raw", "lstm_improved", "tcn", "tft", "patchtst"],
        help="Subset of runs to execute.",
    )
    parser.add_argument(
        "--skip-existing-seeds",
        action="store_true",
        help="Skip per-seed runs whose metrics_all_models.csv already exists.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    config_dir = output_root / "_configs"
    smoke_dir = output_root / "_smoke"
    config_dir.mkdir(parents=True, exist_ok=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)

    cfg_map = {
        "lstm_raw": make_raw_lstm_cfg(output_root / "lstm_raw"),
        "lstm_improved": make_improved_cfg(output_root / "lstm_improved"),
        "tcn": make_baseline_cfg("tcn", output_root / "tcn"),
        "tft": make_baseline_cfg("tft", output_root / "tft"),
        "patchtst": make_baseline_cfg("patchtst", output_root / "patchtst"),
    }

    cfg_paths = {}
    for name, cfg in cfg_map.items():
        cfg_path = config_dir / f"{name}.yaml"
        write_yaml(cfg_path, cfg)
        cfg_paths[name] = cfg_path

    selected_models = [name for name in ["lstm_raw", "lstm_improved", "tcn", "tft", "patchtst"] if name in args.models]

    if not args.skip_smoke:
        for name in [m for m in ["tcn", "tft", "patchtst"] if m in selected_models]:
            run_smoke_test(cfg_paths[name], smoke_dir / name)
    if args.smoke_only:
        return

    focus_map = {
        "lstm_raw": "lstm",
        "lstm_improved": "lstm",
        "tcn": "tcn",
        "tft": "tft",
        "patchtst": "patchtst",
    }
    for name in selected_models:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_seed_sweep.py"),
            "--config",
            str(cfg_paths[name]),
            "--output-root",
            str(output_root / name),
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--focus-models",
            focus_map[name],
        ]
        if args.skip_existing_seeds:
            cmd.append("--skip-existing")
        run_command(cmd)

    summary = summarize_runs(output_root, selected_models)
    if not summary.empty:
        summary = summary.sort_values("RMSE_mean", ascending=True).reset_index(drop=True)
        summary.to_csv(output_root / "strong_baselines_seed_summary.csv", index=False, encoding="utf-8-sig")
        compact = {
            "seeds": list(args.seeds),
            "best_run_name": summary.iloc[0]["run_name"],
            "best_model": summary.iloc[0]["model"],
            "best_rmse_mean": float(summary.iloc[0]["RMSE_mean"]),
            "best_r2_mean": float(summary.iloc[0]["R2_mean"]),
        }
    else:
        compact = {"seeds": list(args.seeds), "best_run_name": None, "best_model": None}
    (output_root / "strong_baselines_seed_summary.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
