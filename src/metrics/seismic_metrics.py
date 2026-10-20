"""地震 AVA 反演专用指标：R² 与皮尔逊相关系数（逐属性）。

与 legacy ``old/AVA/plot/Evaluation.txt`` 的口径一致：对每个弹性属性
（Vp/Vs/Den）分别计算 R² / PCC / RMSE / MAE / SSIM。RMSE/MAE/SSIM 已在
``src.metrics.base`` 提供（全局标量）；这里补充「逐通道」计算族，并把 R²/PCC
按通道各自分子分母独立聚合（等价于 flatten 每个通道后计算），返回形状 (C,)。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.metrics.base import METRICS, Metric

__all__ = ["METRICS", "R2Score", "PearsonCorrelation", "compute_metrics_per_channel"]


@METRICS.register("r2")
@METRICS.register("r2_score")
class R2Score(Metric):
    """确定系数 R² = 1 - SS_res/SS_tot，越高越好。

    全局模式（单标量）：对全部元素计算；通道模式（``per_channel=True`` 或
    ``channel_axis``）时返回 (C,) 数组（0 维为张量标量）。
    """

    name = "r2"
    higher_is_better = True

    def __init__(self, channel_axis: Optional[int] = 0, per_channel: bool = True) -> None:
        """Args:
            channel_axis: 通道轴；None 表示没有通道轴。
            per_channel: True 返回逐通道数组（供 Evaluation.txt 对照），
                False 返回全体标量。
        """
        self.channel_axis = channel_axis
        self.per_channel = bool(per_channel)

    def compute(self, pred: Any, target: Any) -> Any:
        p = np.asarray(pred, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        if self.channel_axis is None or not self.per_channel:
            ss_res = float(np.sum((p - t) ** 2))
            ss_tot = float(np.sum((t - t.mean()) ** 2))
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        moved = np.moveaxis(p, self.channel_axis, 0)
        tm = np.moveaxis(t, self.channel_axis, 0)
        out = np.empty(moved.shape[0], dtype=np.float64)
        for i in range(moved.shape[0]):
            pi, ti = moved[i].ravel(), tm[i].ravel()
            ss_res = float(np.sum((pi - ti) ** 2))
            ss_tot = float(np.sum((ti - ti.mean()) ** 2))
            out[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return out


@METRICS.register("pearson")
@METRICS.register("pcc")
class PearsonCorrelation(Metric):
    """皮尔逊相关系数（逐通道或全体）。

    PCC = cov(p,t) / (σ_p σ_t)。空间结构相关性度量 —— Evaluation.txt 的 corr 列。
    """

    name = "pearson"
    higher_is_better = True

    def __init__(self, channel_axis: Optional[int] = 0, per_channel: bool = True) -> None:
        self.channel_axis = channel_axis
        self.per_channel = bool(per_channel)

    def compute(self, pred: Any, target: Any) -> Any:
        p = np.asarray(pred, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        if self.channel_axis is None or not self.per_channel:
            return float(_pearson(p.ravel(), t.ravel()))

        moved_t = np.moveaxis(t, self.channel_axis, 0)
        out = np.empty(moved_t.shape[0], dtype=np.float64)
        for i in range(moved_t.shape[0]):
            pi = np.moveaxis(p, self.channel_axis, 0)[i].ravel()
            ti = moved_t[i].ravel()
            out[i] = _pearson(pi, ti)
        return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    n = a.size
    if n == 0:
        return float("nan")
    da = a - a.mean()
    db = b - b.mean()
    denom = float(np.linalg.norm(da) * np.linalg.norm(db))
    if denom == 0:
        return float("nan")
    return float(np.dot(da, db) / denom)


def compute_metrics_per_channel(pred: Any, target: Any) -> dict:
    """对标 legacy Evaluation.txt：每个属性的 r2 / pearson / rmse / mae / ssim。

    返回 ``{"r2": (C,), "pearson": (C,), "rmse": (C,), "mae": (C,)}``；
    ssim 需要二维窗口（逐通道取 2D 切片），此处省略，由 tools/evaluate 单独处理。
    """
    from src.metrics.base import RMSE, MAE

    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    C = p.shape[0]
    out = {}
    for name, metric in (("r2", R2Score()), ("pearson", PearsonCorrelation())):
        out[name] = np.asarray(metric.compute(p, t), dtype=np.float64)
    out["rmse"] = np.array([RMSE().compute(p[i], t[i]) for i in range(C)])
    out["mae"] = np.array([MAE().compute(p[i], t[i]) for i in range(C)])
    return out