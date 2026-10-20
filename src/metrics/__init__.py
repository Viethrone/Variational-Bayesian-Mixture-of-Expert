"""评价指标。

约定：
    * 每个指标是 `Metric` 子类，实现 `compute(pred, target) -> float`，并用 `@METRICS.register` 登记。
    * `MetricCollection` 按配置批量计算，输出统一 dict，直接写入 metrics.json。
    * 指标必须与数据集无关、可复现；带范围的指标（如 psnr）需显式传入 data_range。
"""

from src.metrics.base import (
    METRICS,
    MAE,
    PSNR,
    RMSE,
    SSIM,
    Metric,
    MetricCollection,
    RelativeL2Error,
    build_metrics,
)
from src.metrics.seismic_metrics import (  # noqa: F401  (注册 r2/pearson)
    PearsonCorrelation,
    R2Score,
    compute_metrics_per_channel,
)

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
    "R2Score",
    "PearsonCorrelation",
    "compute_metrics_per_channel",
]
