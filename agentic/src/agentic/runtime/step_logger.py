from __future__ import annotations

import logging
from pathlib import Path

from agentic.runtime.observability import RunRecorder


def create_run_logger(
    log_root: Path,
    run_id: str,
    *,
    recorder: RunRecorder | None = None,
) -> tuple[logging.Logger, Path]:
    recorder = recorder or RunRecorder(log_root, run_id)
    log_path = recorder.run_dir / "lifecycle.log"
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
    setattr(logger, "run_recorder", recorder)
    return logger, log_path
