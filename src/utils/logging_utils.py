from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class RunLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        run_id = self.extra.get("run_id", "unknown")
        return f"[run_id={run_id}] {msg}", kwargs


def build_logger(run_id: str, logs_dir: Path, logger_name: str = "portfolio_gan") -> RunLoggerAdapter:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        logs_dir / f"{run_id}.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return RunLoggerAdapter(logger, {"run_id": run_id})

