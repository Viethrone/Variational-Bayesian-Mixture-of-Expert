"""模型基类与占位实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from src.utils.registry import Registry

__all__ = ["MODELS", "BaseModel", "IdentityModel", "count_parameters"]

MODELS = Registry("model")


class BaseModel(ABC):
    """模型基类：定义 predict / save / load 统一接口。

    基于 torch 的实现建议重写 `module` 属性以暴露原生 nn.Module，
    这样 trainers 可以直接使用优化器与自动求导。
    """

    @abstractmethod
    def predict(self, x: Any) -> Any:
        """前向计算。"""

    @property
    def module(self) -> Optional[Any]:
        """底层框架模块（如 torch.nn.Module）；纯 numpy 实现返回 None。"""
        return None

    def save(self, path: str) -> None:
        """保存权重到指定路径。"""
        raise NotImplementedError(f"{type(self).__name__} 未实现 save")

    def load(self, path: str) -> None:
        """从指定路径加载权重。"""
        raise NotImplementedError(f"{type(self).__name__} 未实现 load")

    def num_parameters(self) -> int:
        """参数总量；默认通过 count_parameters 估算，torch 模型建议重写。"""
        return count_parameters(self)

    def __call__(self, x: Any) -> Any:
        return self.predict(x)


@MODELS.register("identity")
class IdentityModel(BaseModel):
    """恒等模型：骨架自测与占位使用，正式实验必须替换为真实网络。"""

    def predict(self, x: Any) -> Any:
        return x

    def save(self, path: str) -> None:
        np.save(path, np.array([]))

    def load(self, path: str) -> None:
        return None


def count_parameters(model: BaseModel) -> int:
    """统计参数量，兼容 torch 模型与 numpy 模型。"""
    module = model.module
    if module is not None and hasattr(module, "parameters"):
        return int(sum(p.numel() for p in module.parameters() if getattr(p, "requires_grad", True)))
    total = 0
    for value in vars(model).values():
        if isinstance(value, np.ndarray):
            total += int(value.size)
    return total
