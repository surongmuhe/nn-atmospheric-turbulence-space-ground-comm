import argparse
import sys
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import set_seed
from src.experiment import run_experiment
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Professional fusion forecasting experiment runner")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory (otherwise create timestamped run folder)")
    parser.add_argument("--seq-len", type=int, default=None, help="Override data.seq_len")
    parser.add_argument("--pred-len", type=int, default=None, help="Override data.pred_len")
    parser.add_argument("--resample-rule", type=str, default=None, help="Override data.resample_rule (e.g. 5min). Use empty string to disable.")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        # 允许从任意工作目录运行：相对路径按项目根目录解析
        cfg_path = (ROOT / cfg_path).resolve()
    cfg = load_config(str(cfg_path))

    # CLI overrides
    if args.seq_len is not None:
        cfg["data"]["seq_len"] = int(args.seq_len)
    if args.pred_len is not None:
        cfg["data"]["pred_len"] = int(args.pred_len)
    if args.resample_rule is not None:
        cfg["data"]["resample_rule"] = args.resample_rule if args.resample_rule != "" else None
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "outputs" / f"run_{ts}"
    cfg["project"]["output_dir"] = str(out_dir)

    set_seed(cfg["project"]["seed"])
    logger = setup_logger(cfg["project"]["output_dir"], cfg["project"]["log_level"])
    logger.info("Loaded config from %s", str(cfg_path))
    # snapshot config for reproducibility
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_snapshot.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    run_experiment(cfg, logger)


if __name__ == "__main__":
    main()
