"""Data-layer base classes and the shared sample structure.

`InversionSample` is the only contract between the data layer and the algorithms:
    observation  observed data y
    target       ground truth x_true (may be None at inference time)
    meta         per-sample metadata (id, geometry, indices, ...)

Any concrete dataset only produces this structure, so solvers and evaluators stay unaware of the source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.utils.registry import Registry

__all__ = ["DATASETS", "BaseDataset", "InversionSample", "collate_samples", "load_split"]

DATASETS = Registry("dataset")


@dataclass
class InversionSample:
    """One inversion sample.

    Attributes:
        observation: observed array y, shaped as in configs/data/*.yaml.
        target: ground-truth array x_true; None for unsupervised or inference passes.
        meta: extra information such as sample id, geometry or noise level.
    """

    observation: np.ndarray
    target: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def sample_id(self) -> str:
        """Sample identifier used in result filenames; falls back to the index."""
        return str(self.meta.get("id", self.meta.get("index", "sample")))

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict view, for serialisation and debugging."""
        return {
            "observation": self.observation,
            "target": self.target,
            "meta": dict(self.meta),
        }


class BaseDataset(ABC):
    """Dataset base class: index handling and sample loading; subclasses implement `_load_item`.

    A subclass only maps a sample name to an `InversionSample`;
    splits, path resolution and caching live here.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        sample_names: Optional[Sequence[str]] = None,
        split_file: Optional[str | Path] = None,
        transform: Optional[Any] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.sample_names: List[str] = list(
            sample_names if sample_names is not None else load_split(split_file)
        )
        if not self.sample_names:
            raise ValueError(
                f"dataset [{split}] is empty: no sample_names given and {split_file} has no entries."
            )

    # ---- implemented by subclasses ----
    @abstractmethod
    def _load_item(self, name: str) -> InversionSample:
        """Load one sample by name and return the shared structure."""

    # ---- shared interface ----
    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, index: int) -> InversionSample:
        name = self.sample_names[index]
        sample = self._load_item(name)
        if "id" not in sample.meta:
            sample.meta["id"] = name
        sample.meta.setdefault("index", index)
        sample.meta.setdefault("split", self.split)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def names(self) -> Iterable[str]:
        """Iterate over all sample names, for result checking and plots."""
        return iter(self.sample_names)


def collate_samples(batch: Sequence[InversionSample]) -> Dict[str, Any]:
    """Stack a list of samples into a batch dict.

    Returns:
        ``{"observation": ndarray, "target": ndarray|None, "meta": list[dict]}``
        If any sample has no target, the whole batch target is None (inference).
    """
    if not batch:
        raise ValueError("collate_samples got an empty batch")

    observations = np.stack([np.asarray(s.observation) for s in batch], axis=0)
    targets: Optional[np.ndarray] = None
    if all(s.target is not None for s in batch):
        targets = np.stack([np.asarray(s.target) for s in batch], axis=0)

    return {
        "observation": observations,
        "target": targets,
        "meta": [dict(s.meta) for s in batch],
    }


def load_split(split_file: Optional[str | Path]) -> List[str]:
    """Read a split file: one sample name per line, ignoring blanks and `#` comments."""
    if split_file is None:
        return []
    path = Path(split_file)
    if not path.exists():
        raise FileNotFoundError(f"split file not found: {path}")
    names: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item and not item.startswith("#"):
                names.append(item)
    return names
