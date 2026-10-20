"""指标基类与常用实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.utils.registry import Registry

__all__ = [
    "METRICS",
    "Metric",
    "MetricCollection",
    "RMSE",
    "MAE",
    "PSNR",
    "SSIM",
    "RelativeL2Error",
    "build_metrics",
]

METRICS = Registry("metric")


class Metric(ABC):
    """指标基类。

    Attributes:
        name: 写入 metrics.json 的键名。
        higher_is_better: 用于挑选最优 checkpoint 的方向。
    """

    name: str = "metric"
    higher_is_better: bool = False

    @abstractmethod
    def compute(self, pred: Any, target: Any) -> float:
        """计算单个批次/样本的指标值。"""

    def __call__(self, pred: Any, target: Any) -> float:
        return self.compute(pred, target)


@METRICS.register("rmse")
class RMSE(Metric):
    """均方根误差，越小越好。"""

    name = "rmse"
    higher_is_better = False

    def compute(self, pred: Any, target: Any) -> float:
        p = np.asarray(pred, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        _check_same_shape(p, t, self.name)
        return float(np.sqrt(np.mean((p - t) ** 2)))


@METRICS.register("mae")
class MAE(Metric):
    """平均绝对误差，越小越好。"""

    name = "mae"
    higher_is_better = False

    def compute(self, pred: Any, target: Any) -> float:
        p = np.asarray(pred, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        _check_same_shape(p, t, self.name)
        return float(np.mean(np.abs(p - t)))


@METRICS.register("relative_l2")
@METRICS.register("rel_error")
class RelativeL2Error(Metric):
    """相对 L2 误差 ||p-t||₂ / ||t||₂，越小越好。"""

    name = "relative_l2"
    higher_is_better = False

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = float(eps)

    def compute(self, pred: Any, target: Any) -> float:
        p = np.asarray(pred, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        _check_same_shape(p, t, self.name)
        denom = float(np.linalg.norm(t))
        return float(np.linalg.norm(p - t) / max(denom, self.eps))


@METRICS.register("psnr")
class PSNR(Metric):
    """峰值信噪比（dB），越大越好。

    Args:
        data_range: 数据动态范围；None 时按 target 的 max-min 推断。
    """

    name = "psnr"
    higher_is_better = True

    def __init__(self, data_range: Optional[float] = None, eps: float = 1e-12) -> None:
        self.data_range = None if data_range is None else float(data_range)
        self.eps = float(eps)

    def compute(self, pred: Any, target: Any) -> float:
        p = np.asarray(pred, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        _check_same_shape(p, t, self.name)
        data_range = self.data_range
        if data_range is None:
            data_range = float(t.max() - t.min())
        if data_range <= 0:
            raise ValueError("data_range 必须为正数；target 可能是常数场，请显式传入 data_range")
        mse = float(np.mean((p - t) ** 2))
        if mse <= self.eps:
            return float("inf")
        return float(20.0 * np.log10(data_range / np.sqrt(mse)))


@METRICS.register("ssim")
class SSIM(Metric):
    """结构相似度（2D 图像），越大越好。

    基于高斯窗的标准实现，避免额外依赖 skimage；
    TODO: 如需与论文对齐，可切换为 skimage.metrics.structural_similarity 并核对参数。
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
                f"SSIM 目前仅支持 2D 单图，收到 ndim={p.ndim}；"
                f"批量数据请逐样本计算后取平均。"
            )
        data_range = self.data_range or float(t.max() - t.min())
        if data_range <= 0:
            raise ValueError("data_range 必须为正数；请显式传入 data_range")

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


class MetricCollection:
    """按配置批量计算多个指标，输出统一 dict。"""

    def __init__(self, metrics: Sequence[Metric]) -> None:
        if not metrics:
            raise ValueError("MetricCollection 至少需要一个指标")
        self.metrics: List[Metric] = list(metrics)

    @property
    def names(self) -> List[str]:
        """全部指标名。"""
        return [m.name for m in self.metrics]

    def compute(self, pred: Any, target: Any) -> Dict[str, float]:
        """计算全部指标，返回 {指标名: 值}。"""
        return {m.name: float(m.compute(pred, target)) for m in self.metrics}

    def compute_batch(self, preds: Any, targets: Any) -> Dict[str, float]:
        """逐样本计算后按样本求平均，避免批内量纲被展平。"""
        preds_arr = np.asarray(preds)
        targets_arr = np.asarray(targets)
        if preds_arr.shape != targets_arr.shape:
            raise ValueError(
                f"批量形状不一致: pred {preds_arr.shape} vs target {targets_arr.shape}"
            )
        if preds_arr.ndim < 2:
            return self.compute(preds_arr, targets_arr)

        per_sample = [self.compute(preds_arr[i], targets_arr[i]) for i in range(preds_arr.shape[0])]
        return {
            name: float(np.mean([item[name] for item in per_sample]))
            for name in self.names
        }

    def __call__(self, pred: Any, target: Any) -> Dict[str, float]:
        return self.compute(pred, target)


def build_metrics(names: Iterable[Any], registry: Optional[Registry] = None) -> MetricCollection:
    """按配置构造 MetricCollection。

    支持两种写法：
        metrics: [rmse, mae]                          # 纯名称
        metrics: [{_target_: psnr, data_range: 1.0}]  # 带参数
    """
    reg = registry or METRICS
    items: List[Metric] = []
    for entry in names:
        if isinstance(entry, str):
            items.append(reg.get(entry)())
        elif isinstance(entry, dict):
            cfg = dict(entry)
            target = cfg.pop("_target_", None) or cfg.pop("name", None)
            if target is None:
                raise ValueError(f"指标配置缺少 `_target_`: {entry}")
            items.append(reg.get(str(target))(**cfg))
        elif isinstance(entry, Metric):
            items.append(entry)
        else:
            raise TypeError(f"无法解析指标配置项: {entry!r}（应为字符串或映射）")
    return MetricCollection(items)


# ---------------------------------------------------------------- 内部工具


def _check_same_shape(p: np.ndarray, t: np.ndarray, metric_name: str) -> None:
    if p.shape != t.shape:
        raise ValueError(
            f"[{metric_name}] 形状不一致: pred {p.shape} vs target {t.shape}"
        )


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError(f"高斯窗大小必须是奇数，收到 {size}")
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = g / g.sum()
    return np.outer(kernel_1d, kernel_1d)


def _convolve2d_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D 'same' 卷积；优先使用 scipy，缺失时退化为纯 numpy 实现。"""
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
