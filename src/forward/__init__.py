"""正演算子与观测算子。

反演问题的标准形式：
    y = A( F(x) ) + noise
其中 F 是正演算子（物理模型），A 是观测算子（采样/积分/投影），noise 为观测噪声。

本模块提供：
    * `FORWARD_OPERATORS`：正演算子注册表
    * `OBSERVATION_OPERATORS`：观测算子注册表
    * `ForwardModel`：把正演 + 观测 + 噪声组合为可调用整体，并统一提供伴随/雅可比接口
"""

from src.forward.base import (
    FORWARD_OPERATORS,
    OBSERVATION_OPERATORS,
    ForwardModel,
    ObservationOperator,
    ForwardOperator,
    add_noise,
)

__all__ = [
    "FORWARD_OPERATORS",
    "OBSERVATION_OPERATORS",
    "ForwardModel",
    "ForwardOperator",
    "ObservationOperator",
    "add_noise",
]

# import concrete forward operators to trigger registration
from src.forward.seismic import SeismicForward  # noqa: E402,F401

# NOTE: ``src.forward.elastic_impedance`` (Connolly EI synthesis / exact Zoeppritz, plus
# the ``.mat`` generator that produced the raw observations) is deliberately NOT imported
# here so that importing this package stays light. Import it explicitly when needed:
#     from src.forward.elastic_impedance import ElasticImpedanceForward
