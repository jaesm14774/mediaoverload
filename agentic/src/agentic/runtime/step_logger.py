from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def create_run_logger(log_root: Path, run_id: str) -> tuple[logging.Logger, Path]:
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id}.log"
    logger_name = f"mediaoverload.agentic.{run_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, log_path
