import logging
from pathlib import Path


def setup_logger(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fusion_forecasting")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(Path(output_dir) / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
