"""Single entry point for seeding.

Every experiment calls `seed_everything(cfg["seed"])` once before it starts,
so numpy / random / torch behave consistently and the seed is recorded in run metadata.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

__all__ = ["seed_everything", "get_random_state"]


def seed_everything(seed: int, *, deterministic: bool = False,
                    deterministic_algorithms: bool | str = False) -> int:
    """Seed every random source.

    Args:
        seed: the random seed.
        deterministic: also enable cudnn deterministic mode (may be slower).
        deterministic_algorithms: switch for ``torch.use_deterministic_algorithms``.
            ``True`` = strict (an operator without a deterministic implementation raises);
            ``"warn"`` = warn only (for operators that only have atomic implementations);
            ``False`` (default) = leave the setting alone.

    Returns:
        The seed that was applied.
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
    """Return a snapshot of the RNG state, for debugging irreproducibility."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    torch = _try_import_torch()
    if torch is not None:
        state["torch"] = torch.get_rng_state()
    return state


def _try_import_torch() -> Optional[object]:
    """Import torch lazily; return None when it is absent, so torch stays optional."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    return torch
