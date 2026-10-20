"""网络与模型结构。

约定：
    * 所有模型通过 `@MODELS.register("名字")` 登记，配置用 `_target_` 指定。
    * 模型只负责结构，不含训练逻辑；训练流程在 src/trainers。
    * 后端可选 numpy（轻量原型）或 torch/jax（正式训练），
      统一实现 `predict(x)` 接口，避免上游感知框架差异。
"""

from src.models.base import MODELS, BaseModel, IdentityModel, count_parameters
from src.models.vbmile import GeoVBMILE

__all__ = ["MODELS", "BaseModel", "IdentityModel", "count_parameters", "GeoVBMILE"]
