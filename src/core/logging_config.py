"""Structured JSON logging configuration."""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import json as json_logger

from config.settings import settings


def setup_logging(name: str = "semantic_search") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # JSON formatter
    fmt = json_logger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console handler — INFO level
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler — DEBUG level, rotating 10 MB × 5 files
    log_dir = Path(settings.logs_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()
