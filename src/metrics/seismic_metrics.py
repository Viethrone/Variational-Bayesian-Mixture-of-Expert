"""Seismic AVA inversion metrics: R^2 and Pearson correlation, per property.

The convention matches the legacy ``Evaluation.txt``: for every elastic property
(Vp/Vs/Den) compute R^2 / PCC / RMSE / MAE / SSIM. RMSE/MAE/SSIM are global and come
from ``src.metrics.base``; this module adds the per-channel family, evaluating R^2/PCC
per channel (equivalent to flattening each channel), returning shape (C,).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.metrics.base import METRICS, Metric

__all__ = ["METRICS", "R2Score", "PearsonCorrelation", "compute_metrics_per_channel"]


@METRICS.register("r2")
@METRICS.register("r2_score")
class R2Score(Metric):
    """Coefficient of determination R^2 = 1 - SS_res/SS_tot (higher is better).

    Global mode returns one scalar over all elements; channel mode (``per_channel=True``
    or a given ``channel_axis``) returns a (C,) array.
    """

    name = "r2"
    higher_is_better = True

    def __init__(self, channel_axis: Optional[int] = 0, per_channel: bool = True) -> None:
        """Args:
            channel_axis: channel axis; None means there is none.
            per_channel: True returns the per-channel array (matching Evaluation.txt),
                False returns a single scalar.
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
    """Pearson correlation coefficient (per channel or global).

    PCC = cov(p,t) / (sigma_p sigma_t), the ``corr`` column of Evaluation.txt.
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
    """Match the legacy Evaluation.txt: r2 / pearson / rmse / mae / ssim per property.

    Returns ``{"r2": (C,), "pearson": (C,), "rmse": (C,), "mae": (C,)}``;
    SSIM needs a 2-D window (a slice per channel) and is handled by tools/evaluate.
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