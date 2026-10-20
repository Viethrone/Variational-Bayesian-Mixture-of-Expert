"""正演算子、观测算子基类与噪声注入。

接口约定（子类实现）：
    forward(x)      正演：参数/场 -> 物理量
    adjoint(y)      伴随算子（线性问题）或雅可比转置作用（非线性问题）
    jacobian(x)     可选：返回雅可比（矩阵或可调用对象）

反演求解器只依赖这三个接口，因此替换正演实现不需要改求解器代码。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import numpy as np

from src.utils.registry import Registry

__all__ = [
    "FORWARD_OPERATORS",
    "OBSERVATION_OPERATORS",
    "ForwardOperator",
    "ObservationOperator",
    "ForwardModel",
    "IdentityOperator",
    "add_noise",
]

FORWARD_OPERATORS = Registry("forward_operator")
OBSERVATION_OPERATORS = Registry("observation_operator")

ArrayLike = Union[np.ndarray, Any]  # 兼容 torch.Tensor 等


class ForwardOperator(ABC):
    """正演算子基类：x -> 物理量场。"""

    @abstractmethod
    def forward(self, x: ArrayLike) -> ArrayLike:
        """正演计算。"""

    def adjoint(self, y: ArrayLike) -> ArrayLike:
        """伴随算子。默认未实现，线性算子必须重写。"""
        raise NotImplementedError(
            f"{type(self).__name__} 未实现 adjoint；"
            f"若使用基于梯度的求解器，请提供 adjoint 或 jacobian。"
        )

    def jacobian(self, x: ArrayLike) -> Union[ArrayLike, Callable[[ArrayLike], ArrayLike]]:
        """雅可比（矩阵或作用函数）。默认由有限差分近似，子类建议给出解析形式。"""

        def jvp(v: ArrayLike, eps: float = 1e-6) -> ArrayLike:
            return (self.forward(x + eps * v) - self.forward(x - eps * v)) / (2 * eps)

        return jvp

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self.forward(x)


class ObservationOperator(ABC):
    """观测算子基类：物理量场 -> 实际观测（采样、积分、投影、降采样等）。"""

    @abstractmethod
    def forward(self, u: ArrayLike) -> ArrayLike:
        """施加观测算子。"""

    def adjoint(self, y: ArrayLike) -> ArrayLike:
        """观测算子的伴随。默认未实现。"""
        raise NotImplementedError(f"{type(self).__name__} 未实现 adjoint")

    def __call__(self, u: ArrayLike) -> ArrayLike:
        return self.forward(u)


@FORWARD_OPERATORS.register("identity")
@OBSERVATION_OPERATORS.register("identity")
class IdentityOperator(ForwardOperator, ObservationOperator):
    """恒等算子：用于骨架自测与占位，正式实验必须替换。"""

    def forward(self, x: ArrayLike) -> ArrayLike:
        return x

    def adjoint(self, y: ArrayLike) -> ArrayLike:
        return y


class ForwardModel:
    """组合正演算子、观测算子与噪声，得到 y = A(F(x)) + n。

    Args:
        forward_operator: 正演算子实例。
        observation_operator: 观测算子实例；None 表示不做额外观测变换。
        noise: 噪声配置，形如 ``{"type": "gaussian", "sigma": 0.01}``。
    """

    def __init__(
        self,
        forward_operator: ForwardOperator,
        observation_operator: Optional[ObservationOperator] = None,
        noise: Optional[dict] = None,
    ) -> None:
        self.forward_operator = forward_operator
        self.observation_operator = observation_operator
        self.noise = dict(noise or {"type": "none"})

    def simulate(self, x: ArrayLike, *, seed: Optional[int] = None) -> ArrayLike:
        """生成含噪观测：完整正演链路。"""
        u = self.forward_operator.forward(x)
        y = self.observation_operator.forward(u) if self.observation_operator is not None else u
        return add_noise(y, self.noise, seed=seed)

    def predict(self, x: ArrayLike) -> ArrayLike:
        """无噪声正演，用于计算数据项残差。"""
        u = self.forward_operator.forward(x)
        return self.observation_operator.forward(u) if self.observation_operator is not None else u

    def residual(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        """数据残差 predict(x) - y，供损失与梯度使用。"""
        return self.predict(x) - y

    def adjoint(self, y: ArrayLike) -> ArrayLike:
        """组合伴随：A^T(F^T(y))，线性情形下可直接用于梯度计算。"""
        if self.observation_operator is None:
            return self.forward_operator.adjoint(y)
        return self.forward_operator.adjoint(self.observation_operator.adjoint(y))

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self.predict(x)


def add_noise(y: ArrayLike, noise: Optional[dict], *, seed: Optional[int] = None) -> ArrayLike:
    """按配置注入噪声。

    支持类型：``none``、``gaussian``（sigma 或 snr_db）、``poisson``（scale）。
    TODO: 按实际观测物理补充其他噪声模型（如散粒噪声、条带噪声、相位噪声）。
    """
    cfg = dict(noise or {"type": "none"})
    kind = str(cfg.get("type", "none")).lower()
    if kind in ("none", "", "null"):
        return y

    rng = np.random.default_rng(seed)
    arr = np.asarray(y)

    if kind == "gaussian":
        sigma = cfg.get("sigma")
        if sigma is None and cfg.get("snr_db") is not None:
            signal_power = float(np.mean(arr.astype(np.float64) ** 2))
            snr = float(cfg["snr_db"])
            sigma = float(np.sqrt(signal_power / (10 ** (snr / 10)))) if signal_power > 0 else 0.0
        sigma = float(sigma or 0.0)
        if sigma == 0.0:
            return y
        return arr + rng.normal(0.0, sigma, size=arr.shape).astype(arr.dtype)

    if kind == "poisson":
        scale = float(cfg.get("scale", 1.0))
        if scale <= 0:
            raise ValueError("poisson 噪声的 scale 必须为正数")
        return (rng.poisson(np.clip(arr, 0, None) * scale) / scale).astype(arr.dtype)

    raise ValueError(
        f"未知噪声类型: {kind!r}。可选: none | gaussian | poisson（其余请在 add_noise 中扩展）"
    )
