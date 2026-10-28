"""Evaluation metrics (``@METRICS.register("name")``; names are listed in ``cfg["metrics"]``)."""

from src.metrics.base import METRICS, MAE, RMSE, SSIM, Metric
from src.metrics.seismic_metrics import (  # noqa: F401  (registers r2 / pearson)
    PearsonCorrelation,
    R2Score,
    compute_metrics_per_channel,
)

__all__ = [
    "METRICS",
    "Metric",
    "RMSE",
    "MAE",
    "SSIM",
    "R2Score",
    "PearsonCorrelation",
    "compute_metrics_per_channel",
]
