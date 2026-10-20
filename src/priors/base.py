"""先验基类与常用实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import numpy as np

from src.utils.registry import Registry

__all__ = ["PRIORS", "BasePrior", "NonePrior", "GaussianPrior", "UniformPrior"]

PRIORS = Registry("prior")


class BasePrior(ABC):
    """先验基类。子类实现 log_prob / grad_log_prob，sample 可选。"""

    @abstractmethod
    def log_prob(self, x: Any) -> float:
        """对数先验概率（允许非归一化，只需保证梯度正确）。"""

    @abstractmethod
    def grad_log_prob(self, x: Any) -> Any:
        """对数先验概率对 x 的梯度。"""

    def sample(self, rng: np.random.Generator, shape: Sequence[int]) -> Any:
        """从先验采样。默认未实现。"""
        raise NotImplementedError(f"{type(self).__name__} 未实现 sample")


@PRIORS.register("none")
class NonePrior(BasePrior):
    """无先验：目标函数中不贡献任何项，消融实验的默认占位。"""

    def log_prob(self, x: Any) -> float:
        return 0.0

    def grad_log_prob(self, x: Any) -> Any:
        return np.zeros_like(np.asarray(x, dtype=np.float64))

    def sample(self, rng: np.random.Generator, shape: Sequence[int]) -> Any:
        return np.zeros(shape, dtype=np.float64)


@PRIORS.register("gaussian")
class GaussianPrior(BasePrior):
    """各向同性或给定协方差的高斯先验 N(mean, sigma²I)。

    log p(x) = -||x - mean||² / (2 sigma²) + const
    grad     = -(x - mean) / sigma²
    """

    def __init__(
        self,
        mean: Any = 0.0,
        sigma: float = 1.0,
        weight: float = 1.0,
    ) -> None:
        sigma = float(sigma)
        if sigma <= 0:
            raise ValueError(f"高斯先验的 sigma 必须为正数，收到 {sigma!r}")
        self.mean = mean if np.isscalar(mean) else np.asarray(mean, dtype=np.float64)
        self.sigma = sigma
        self.weight = float(weight)
        self._inv_var = self.weight / (sigma**2)

    def log_prob(self, x: Any) -> float:
        diff = np.asarray(x, dtype=np.float64) - self.mean
        return float(-0.5 * self._inv_var * np.sum(diff**2))

    def grad_log_prob(self, x: Any) -> Any:
        diff = np.asarray(x, dtype=np.float64) - self.mean
        return -self._inv_var * diff

    def sample(self, rng: np.random.Generator, shape: Sequence[int]) -> Any:
        mean = self.mean if np.isscalar(self.mean) else np.broadcast_to(self.mean, shape)
        return rng.normal(loc=mean, scale=self.sigma, size=tuple(shape))


@PRIORS.register("uniform")
class UniformPrior(BasePrior):
    """盒式均匀先验：可行域内为常数，域外为 -inf（用大惩罚值近似以保留梯度）。

    Args:
        low: 下界，可为标量或数组；None 表示不限。
        high: 上界，同上。
        penalty: 越界惩罚系数。
    """

    def __init__(
        self,
        low: Optional[Any] = 0.0,
        high: Optional[Any] = 1.0,
        penalty: float = 1.0e6,
    ) -> None:
        self.low = None if low is None else (low if np.isscalar(low) else np.asarray(low, dtype=np.float64))
        self.high = None if high is None else (high if np.isscalar(high) else np.asarray(high, dtype=np.float64))
        self.penalty = float(penalty)

    def _violation(self, x: np.ndarray) -> np.ndarray:
        violation = np.zeros_like(x)
        if self.low is not None:
            violation += np.minimum(x - self.low, 0.0)
        if self.high is not None:
            violation += np.maximum(x - self.high, 0.0)
        return violation

    def log_prob(self, x: Any) -> float:
        arr = np.asarray(x, dtype=np.float64)
        return float(-0.5 * self.penalty * np.sum(self._violation(arr) ** 2))

    def grad_log_prob(self, x: Any) -> Any:
        arr = np.asarray(x, dtype=np.float64)
        return -self.penalty * self._violation(arr)

    def sample(self, rng: np.random.Generator, shape: Sequence[int]) -> Any:
        low = 0.0 if self.low is None else self.low
        high = 1.0 if self.high is None else self.high
        return rng.uniform(low=low, high=high, size=tuple(shape))
