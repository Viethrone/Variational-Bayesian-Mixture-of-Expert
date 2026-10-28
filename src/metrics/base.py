"""Evaluation metrics: scalar ``Metric`` classes registered in ``METRICS``.

Names are the keys written to ``metrics.json`` and the entries listed in ``cfg["metrics"]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from src.utils.registry import Registry

__all__ = ["METRICS", "Metric", "RMSE", "MAE", "SSIM"]

METRICS = Registry("metric")


class Metric(ABC):
    """Base class.

    Attributes:
        name: key written to ``metrics.json``.
        higher_is_better: direction used when picking the best checkpoint.
    """

    name: str = "metric"
    higher_is_better: bool = False

    @abstractmethod
    def compute(self, pred: Any, target: Any) -> float:
        """Metric value of one batch or sample."""

    def __call__(self, pred: Any, target: Any) -> float:
        return self.compute(pred, target)


@METRICS.register("rmse")
class RMSE(Metric):
    """Root-mean-square error (lower is better)."""

    name = "rmse"
    higher_is_better = False

    def compute(self, pred: Any, target: Any) -> float:
        p = np.asarray(pred, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        _check_same_shape(p, t, self.name)
        return float(np.sqrt(np.mean((p - t) ** 2)))


@METRICS.register("mae")
class MAE(Metric):
    """Mean absolute error (lower is better)."""

    name = "mae"
    higher_is_better = False

    def compute(self, pred: Any, target: Any) -> float:
        p = np.asarray(pred, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        _check_same_shape(p, t, self.name)
        return float(np.mean(np.abs(p - t)))


@METRICS.register("ssim")
class SSIM(Metric):
    """Structural similarity on a 2-D image (higher is better).

    Gaussian-window implementation, so no extra dependency is needed; compute it per
    sample and average for batches.

    Args:
        data_range: dynamic range of the data; ``None`` infers it from ``target``.
        k1 / k2 / sigma / win_size: standard SSIM constants.
    """

    name = "ssim"
    higher_is_better = True

    def __init__(
        self,
        data_range: Optional[float] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        sigma: float = 1.5,
        win_size: int = 11,
    ) -> None:
        self.data_range = None if data_range is None else float(data_range)
        self.k1, self.k2, self.sigma = float(k1), float(k2), float(sigma)
        self.win_size = int(win_size)

    def compute(self, pred: Any, target: Any) -> float:
        p = np.asarray(pred, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        _check_same_shape(p, t, self.name)
        if p.ndim != 2:
            raise ValueError(
                f"SSIM expects a single 2-D image, got ndim={p.ndim}; "
                "compute it per sample and average for batches.")
        data_range = self.data_range or float(t.max() - t.min())
        if data_range <= 0:
            raise ValueError("data_range must be positive; pass it explicitly")

        c1 = (self.k1 * data_range) ** 2
        c2 = (self.k2 * data_range) ** 2

        win = _gaussian_kernel(self.win_size, self.sigma)
        mu_p = _convolve2d_same(p, win)
        mu_t = _convolve2d_same(t, win)

        mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p * mu_t
        sigma_p2 = _convolve2d_same(p * p, win) - mu_p2
        sigma_t2 = _convolve2d_same(t * t, win) - mu_t2
        sigma_pt = _convolve2d_same(p * t, win) - mu_pt

        numerator = (2 * mu_pt + c1) * (2 * sigma_pt + c2)
        denominator = (mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2)
        return float(np.mean(numerator / denominator))


# ---------------------------------------------------------------- internal helpers


def _check_same_shape(p: np.ndarray, t: np.ndarray, metric_name: str) -> None:
    if p.shape != t.shape:
        raise ValueError(f"[{metric_name}] shape mismatch: pred {p.shape} vs target {t.shape}")


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError(f"the Gaussian window must be odd-sized, got {size}")
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = g / g.sum()
    return np.outer(kernel_1d, kernel_1d)


def _convolve2d_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D "same" convolution; uses scipy when available and a numpy fallback otherwise."""
    try:
        from scipy.signal import convolve2d  # noqa: PLC0415

        return convolve2d(image, kernel, mode="same", boundary="symm")
    except ImportError:
        pass

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="symmetric")
    out = np.zeros_like(image, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i : i + image.shape[0], j : j + image.shape[1]]
    return out
