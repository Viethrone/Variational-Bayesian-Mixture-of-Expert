"""torch 通用工具：设备选择、种子辅助。"""

from __future__ import annotations

import random

import numpy as np
import torch

__all__ = ["get_device", "set_seed"]


def get_device(prefer: str | None = None) -> torch.device:
    """选择可用设备：显式 `prefer`（"cpu" / "cuda" / "cuda:0"）优先，
    否则自动回落 cpu -> cuda。
    """
    if prefer is not None:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """固定所有随机源（numpy / python / torch / cudnn）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)