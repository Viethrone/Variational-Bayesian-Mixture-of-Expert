"""Shared pytest fixtures / skip guards for the public release.

The public repository intentionally ships **without** the data arrays
(``data/processed/**``, ``data/interim/norm_stats.json``). Tests that need the
rebuilt dataset therefore skip instead of failing, so a fresh clone passes
``pytest`` out of the box. Once the data has been prepared with::

    python -m tools.preprocess.run_preprocess --config configs/data/seismic1d.yaml
    python -m tools.preprocess.run_preprocess --config configs/data/seismic2d.yaml

those tests become active again.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_CONFIGS = {
    "1d": "configs/data/seismic1d.yaml",
    "2d": "configs/data/seismic2d.yaml",
}


def dataset_root(dim: str) -> Path:
    """Resolved ``data/processed/marmousi2_<dim>`` path (project-root relative)."""
    cfg = load_config(str(PROJECT_ROOT / DATA_CONFIGS[dim]))
    root = Path(cfg["data"]["root"])
    return root if root.is_absolute() else PROJECT_ROOT / root


def dataset_ready(dim: str) -> bool:
    """True when every processed array configured for ``dim`` exists on disk."""
    root = dataset_root(dim)
    cfg = load_config(str(PROJECT_ROOT / DATA_CONFIGS[dim]))
    names = [v for v in (cfg["data"].get("files") or {}).values() if v]
    return bool(names) and all((root / n).exists() for n in names)


@pytest.fixture(scope="session")
def data_ready() -> dict:
    """``{"1d": bool, "2d": bool}`` — which processed datasets are present."""
    return {dim: dataset_ready(dim) for dim in ("1d", "2d")}
