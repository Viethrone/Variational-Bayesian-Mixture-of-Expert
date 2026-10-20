"""先验模型。

约定接口：
    log_prob(x)       -> float    # 对数先验概率（可为非归一化）
    grad_log_prob(x)  -> ndarray  # 对数先验对 x 的梯度
    sample(rng, shape)-> ndarray  # 从先验采样，用于合成数据与可视化

在 `InverseProblem` 中先验以「负对数概率」的形式加入目标函数，
因此 `objective` 使用 `-log_prob`，梯度使用 `-grad_log_prob`。
"""

from src.priors.base import PRIORS, BasePrior, GaussianPrior, NonePrior, UniformPrior
from src.priors.gmm import (  # noqa: F401  (注册 gmm_diag)
    GMMPrior,
    GMMPriorFitter,
    SpatialPriorGenerator,
    hungarian_match,
)

__all__ = [
    "PRIORS",
    "BasePrior",
    "GaussianPrior",
    "NonePrior",
    "UniformPrior",
    "GMMPrior",
    "GMMPriorFitter",
    "SpatialPriorGenerator",
    "hungarian_match",
]

# TODO: 按需补充马尔可夫随机场、预训练去噪器（PnP）、扩散模型先验等。
