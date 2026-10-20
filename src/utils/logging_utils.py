"""日志工具：控制台 + 文件双通道，按 run 目录隔离。

约定：`setup_logging` 在每个实验入口调用一次；库代码内部只用 `get_logger(__name__)`，
不在模块顶层配置 handler，避免污染调用方的日志设置。
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
    """获取 logger。未显式配置时挂一个控制台 handler，保证直接运行也有输出。"""
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
    """配置根日志。

    Args:
        level: 日志级别，接受字符串或 logging 常量。
        log_file: 日志文件路径；提供时自动创建父目录。
        console: 是否输出到控制台。
        name: 指定 logger 名；None 表示配置根 logger。
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
