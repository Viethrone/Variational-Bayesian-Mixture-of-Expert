"""损失函数（数据项）。

约定接口：每个损失对象提供
    value(pred, target) -> float
    grad(pred, target)  -> ndarray   # 对 pred 的梯度，供求解器链式反传

新增损失时用 `@LOSSES.register("名字")` 登记，即可在配置中通过 `_target_` 使用。
"""

from src.losses.base import LOSSES, BaseLoss, L1Loss, MSELoss, WeightedMSELoss
from src.losses.vbelbo import VBELBOCore

__all__ = ["LOSSES", "BaseLoss", "L1Loss", "MSELoss", "WeightedMSELoss", "VBELBOCore"]
