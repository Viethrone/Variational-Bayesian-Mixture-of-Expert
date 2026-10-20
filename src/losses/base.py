"""损失函数基类与常用数据项实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from src.utils.registry import Registry

__all__ = ["LOSSES", "BaseLoss", "MSELoss", "L1Loss", "WeightedMSELoss"]

LOSSES = Registry("loss")


class BaseLoss(ABC):
    """损失基类：同时提供值与对预测的梯度。"""

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = float(weight)

    @abstractmethod
    def raw_value(self, pred: Any, target: Any) -> float:
        """未加权的损失值。"""

    @abstractmethod
    def raw_grad(self, pred: Any, target: Any) -> Any:
        """未加权的、对 pred 的梯度。"""

    def value(self, pred: Any, target: Any) -> float:
        """加权损失值。"""
        return self.weight * float(self.raw_value(pred, target))

    def grad(self, pred: Any, target: Any) -> Any:
        """加权梯度。"""
        return self.weight * self.raw_grad(pred, target)

    def __call__(self, pred: Any, target: Any) -> float:
        return self.value(pred, target)


@LOSSES.register("mse")
@LOSSES.register("l2")
class MSELoss(BaseLoss):
    """均方误差：mean((pred - target)^2)。"""

    def __init__(self, weight: float = 1.0, reduction: str = "mean") -> None:
        super().__init__(weight)
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction 只能是 mean 或 sum，收到 {reduction!r}")
        self.reduction = reduction

    def raw_value(self, pred: Any, target: Any) -> float:
        diff = np.asarray(pred) - np.asarray(target)
        sq = float(np.sum(diff**2))
        return sq / diff.size if self.reduction == "mean" else sq

    def raw_grad(self, pred: Any, target: Any) -> Any:
        diff = np.asarray(pred) - np.asarray(target)
        factor = 2.0 / diff.size if self.reduction == "mean" else 2.0
        return factor * diff


@LOSSES.register("l1")
@LOSSES.register("mae")
class L1Loss(BaseLoss):
    """L1 损失：mean(|pred - target|)，对离群观测更鲁棒。"""

    def __init__(self, weight: float = 1.0, reduction: str = "mean", eps: float = 1e-12) -> None:
        super().__init__(weight)
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction 只能是 mean 或 sum，收到 {reduction!r}")
        self.reduction = reduction
        self.eps = float(eps)

    def raw_value(self, pred: Any, target: Any) -> float:
        diff = np.abs(np.asarray(pred) - np.asarray(target))
        total = float(np.sum(diff))
        return total / diff.size if self.reduction == "mean" else total

    def raw_grad(self, pred: Any, target: Any) -> Any:
        diff = np.asarray(pred) - np.asarray(target)
        grad = diff / (np.abs(diff) + self.eps)
        return grad / diff.size if self.reduction == "mean" else grad


@LOSSES.register("weighted_mse")
class WeightedMSELoss(BaseLoss):
    """带权 MSE：用于观测点可信度不均（如掩膜、方差倒数权重）的场景。

    Args:
        weights: 与 pred 同形的权重数组；None 时退化为普通 MSE。
        normalize: 是否按权重和归一化。
    """

    def __init__(
        self,
        weight: float = 1.0,
        weights: Optional[Any] = None,
        normalize: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__(weight)
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self.normalize = bool(normalize)
        self.reduction = reduction

    def _effective_weights(self, shape: tuple) -> Any:
        if self.weights is None:
            return np.ones(shape, dtype=np.float64)
        if self.weights.shape != shape:
            raise ValueError(
                f"权重形状 {self.weights.shape} 与预测形状 {shape} 不一致，无法计算加权 MSE"
            )
        return self.weights

    def raw_value(self, pred: Any, target: Any) -> float:
        pred_arr = np.asarray(pred, dtype=np.float64)
        w = self._effective_weights(pred_arr.shape)
        sq = w * (pred_arr - np.asarray(target, dtype=np.float64)) ** 2
        denom = float(w.sum()) if self.normalize else float(sq.size)
        return float(sq.sum()) / max(denom, 1e-12)

    def raw_grad(self, pred: Any, target: Any) -> Any:
        pred_arr = np.asarray(pred, dtype=np.float64)
        w = self._effective_weights(pred_arr.shape)
        grad = 2.0 * w * (pred_arr - np.asarray(target, dtype=np.float64))
        denom = float(w.sum()) if self.normalize else float(grad.size)
        return grad / max(denom, 1e-12)
