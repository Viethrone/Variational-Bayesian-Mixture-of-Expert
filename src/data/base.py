"""数据层基类与统一样本结构。

`InversionSample` 是数据层与算法层之间的唯一契约：
    observation  观测数据 y
    target       真值 x_true（训练/评估用，推理时可为 None）
    meta         样本级元信息（id、几何参数、传感器索引等）

任何具体 Dataset 只需产出该结构，反演求解器与评估脚本就无需感知数据来源。
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
    """单条反演样本。

    Attributes:
        observation: 观测数组 y，形状见 configs/data/*.yaml 的 shape.observation。
        target: 真值数组 x_true；无监督或推理阶段为 None。
        meta: 附加信息，例如样本 id、观测几何、噪声水平。
    """

    observation: np.ndarray
    target: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def sample_id(self) -> str:
        """样本标识，用于结果文件命名；缺失时按位置回退。"""
        return str(self.meta.get("id", self.meta.get("index", "sample")))

    def to_dict(self) -> Dict[str, Any]:
        """转为普通 dict，便于序列化与调试。"""
        return {
            "observation": self.observation,
            "target": self.target,
            "meta": dict(self.meta),
        }


class BaseDataset(ABC):
    """数据集基类：负责索引管理与样本读取，子类实现 `_load_item`。

    子类通常只需关心「给定样本名 -> 返回 InversionSample」，
    划分读取、路径解析、缓存等通用逻辑由基类承担。
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
                f"数据集 [{split}] 为空：未提供 sample_names，或划分文件 {split_file} 无内容。"
            )

    # ---- 子类实现 ----
    @abstractmethod
    def _load_item(self, name: str) -> InversionSample:
        """按样本名读取一条数据并返回统一结构。"""

    # ---- 通用接口 ----
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
        """返回全部样本名，便于结果核对与可视化。"""
        return iter(self.sample_names)


def collate_samples(batch: Sequence[InversionSample]) -> Dict[str, Any]:
    """把样本列表堆叠为批字典。

    Returns:
        ``{"observation": ndarray, "target": ndarray|None, "meta": list[dict]}``
        target 中若存在 None，整批 target 返回 None（推理场景）。
    """
    if not batch:
        raise ValueError("collate_samples 收到空批次")

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
    """读取划分文件：每行一个样本名，忽略空行与 `#` 注释。"""
    if split_file is None:
        return []
    path = Path(split_file)
    if not path.exists():
        raise FileNotFoundError(f"划分文件不存在: {path}")
    names: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item and not item.startswith("#"):
                names.append(item)
    return names
