import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_external_site_four_model_route import make_config


DEFAULT_CANDIDATES = [
    {"hidden_size": 64, "num_layers": 1, "dropout": 0.0, "learning_rate": 0.0010},
    {"hidden_size": 64, "num_layers": 2, "dropout": 0.05, "learning_rate": 0.0010},
    {"hidden_size": 96, "num_layers": 1, "dropout": 0.0, "learning_rate": 0.0010},
    {"hidden_size": 96, "num_layers": 2, "dropout": 0.05, "learning_rate": 0.0010},
    {"hidden_size": 96, "num_layers": 2, "dropout": 0.10, "learning_rate": 0.0010},
    {"hidden_size": 128, "num_layers": 1, "dropout": 0.0, "learning_rate": 0.0010},
    {"hidden_size": 128, "num_layers": 2, "dropout": 0.05, "learning_rate": 0.0010},
    {"hidden_size": 128, "num_layers": 2, "dropout": 0.10, "learning_rate": 0.0010},
    {"hidden_size": 160, "num_layers": 1, "dropout": 0.0, "learning_rate": 0.0007},
    {"hidden_size": 160, "num_layers": 2, "dropout": 0.05, "learning_rate": 0.0007},
    {"hidden_size": 192, "num_layers": 1, "dropout": 0.05, "learning_rate": 0.0007},
    {"hidden_size": 128, "num_layers": 2, "dropout": 0.0, "learning_rate": 0.0005},
    {"hidden_size": 160, "num_layers": 2, "dropout": 0.0, "learning_rate": 0.0005},
]


def parse_min_val_loss(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    pattern = re.compile(r"val_loss=([0-9.eE+-]+)")
    values = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            values.append(float(match.group(1)))
    return min(values) if values else None


def patch_lstm_config(cfg: dict, candidate: dict) -> dict:
    model = cfg["models"]["lstm"]
    model.update(
        {
            "hidden_size": int(candidate["hidden_size"]),
            "num_layers": int(candidate["num_layers"]),
            "dropout": float(candidate["dropout"]),
            "bidirectional": False,
            "attention_pool": False,
            "input_proj_size": None,
            "ar_window": 0,
            "context_window": 12,
        }
    )
    cfg["train"]["learning_rate"] = float(candidate["learning_rate"])
    return cfg


def run_candidate(output_root: Path, candidate: dict, index: int, epochs: int, batch_size: int, target_mode: str) -> dict:
    name = (
        f"hs{candidate['hidden_size']}_l{candidate['num_layers']}"
        f"_do{str(candidate['dropout']).replace('.', 'p')}"
        f"_lr{str(candidate['learning_rate']).replace('.', 'p')}"
    )
    out_dir = output_root / name
    cfg = make_config("lstm_raw", out_dir, epochs=epochs, batch_size=batch_size, target_mode=target_mode)
    cfg = patch_lstm_config(cfg, candidate)

    config_dir = output_root / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = config_dir / f"{index:02d}_{name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/run_experiments.py",
        "--config",
        str(cfg_path),
        "--output-dir",
        str(out_dir),
    ]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

    metrics_path = out_dir / "metrics_all_models.csv"
    metrics = pd.read_csv(metrics_path)
    row = metrics[metrics["model"] == "lstm"].iloc[0].to_dict()
    row.update(candidate)
    row["candidate"] = name
    row["output_dir"] = str(out_dir)
    row["min_val_loss"] = parse_min_val_loss(out_dir / "train.log")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Small validation-oriented sweep for external-site LSTM-raw.")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-mode", type=str, default="delta", choices=["absolute", "delta"])
    parser.add_argument("--max-candidates", type=int, default=None)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else ROOT / "outputs" / "external_site_lstm_raw_sweep" / stamp
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = DEFAULT_CANDIDATES[: args.max_candidates] if args.max_candidates else DEFAULT_CANDIDATES
    rows = []
    for idx, candidate in enumerate(candidates, start=1):
        rows.append(run_candidate(output_root, candidate, idx, args.epochs, args.batch_size, args.target_mode))

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["min_val_loss", "RMSE"], ascending=[True, True]).reset_index(drop=True)
    summary.to_csv(output_root / "lstm_raw_sweep_metrics.csv", index=False, encoding="utf-8-sig")
    (output_root / "lstm_raw_sweep_summary.md").write_text(
        "# External-Site LSTM-Raw Sweep\n\n"
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n\n"
        + summary[
            [
                "candidate",
                "hidden_size",
                "num_layers",
                "dropout",
                "learning_rate",
                "min_val_loss",
                "RMSE",
                "MAE",
                "R2",
            ]
        ].to_markdown(index=False, floatfmt=".5f"),
        encoding="utf-8",
    )
    (output_root / "lstm_raw_sweep_summary.json").write_text(
        json.dumps(
            {
                "output_root": str(output_root),
                "target_mode": args.target_mode,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_candidates": len(candidates),
                "best_by_val": summary.iloc[0].to_dict(),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary[["candidate", "min_val_loss", "RMSE", "MAE", "R2"]].to_string(index=False))
    print(f"Wrote sweep summary to {output_root}")


if __name__ == "__main__":
    main()
