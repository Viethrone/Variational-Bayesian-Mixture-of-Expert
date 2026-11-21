"""Logging: console + file channels, isolated per run directory.

`setup_logging` runs once per experiment entry point; library code only calls
`get_logger(__name__)` and never configures handlers at import time.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

__all__ = ["get_logger", "setup_logging", "DEFAULT_FORMAT"]

DEFAULT_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "inversion") -> logging.Logger:
    """Get a logger; without explicit configuration a console handler is attached."""
    logger = logging.getLogger(name)
    if not logger.handlers and not logging.root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def setup_logging(
    level: Union[str, int] = "INFO",
    *,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    name: Optional[str] = None,
) -> logging.Logger:
    """Configure the root logger.

    Args:
        level: log level, as a string or a logging constant.
        log_file: log file path; its parent directory is created.
        console: also log to the console.
        name: logger name to configure; None means the root logger.
    """
    logger = logging.getLogger(name) if name else logging.root
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DATE_FORMAT)

    if console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if name:
        logger.propagate = False
    return logger
