"""随机种子统一入口。

所有实验在开始前调用一次 `seed_everything(cfg["seed"])`，
保证 numpy / random / torch（若安装）行为一致，并在 run 元信息中留痕。
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

__all__ = ["seed_everything", "get_random_state"]


def seed_everything(seed: int, *, deterministic: bool = False,
                    deterministic_algorithms: bool | str = False) -> int:
    """设置各随机源的种子。

    Args:
        seed: 随机种子。
        deterministic: 为 True 时额外开启 cudnn 确定性模式（可能降低速度）。
        deterministic_algorithms: ``torch.use_deterministic_algorithms`` 开关。
            ``True`` = 严格模式（遇到没有确定性实现的算子直接报错）；
            ``"warn"`` = 只告警（``warn_only=True``，用于部分算子只有原子实现的情形）；
            ``False``（默认）= 不动，保持旧行为。

    Returns:
        实际使用的种子。
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch = _try_import_torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if deterministic_algorithms:
            # deterministic cuBLAS workspace: without it every matmul/linear warns and
            # stays non-deterministic. Must be set before cuBLAS initialises, i.e. this
            # function has to run early in the process.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            warn_only = str(deterministic_algorithms).lower() == "warn"
            if deterministic:
                torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=warn_only)

    return seed


def get_random_state() -> dict:
    """返回当前随机状态快照，便于调试不可复现问题。"""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    torch = _try_import_torch()
    if torch is not None:
        state["torch"] = torch.get_rng_state()
    return state


def _try_import_torch() -> Optional[object]:
    """按需导入 torch，未安装时返回 None，避免硬依赖。"""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    return torch
