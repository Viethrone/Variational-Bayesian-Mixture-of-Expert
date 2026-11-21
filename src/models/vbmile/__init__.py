"""Geo-VB-MILE network modules, split by role.

``GeoVBMILE`` is the only model registered with ``MODELS``; the other modules are its
building blocks, kept small so each one can be read on its own.

    _norm.py             NormLayer factories (1-D / 2-D)
    _attention1d.py      CBAM attention used by ``BasicBlock1D``
    _blocks1d.py         1-D residual block + 1-D U-Net encoder
    _partition1d.py      1-D variational partition layer (gating)
    _decoder1d.py        1-D expert decoders
    _padding.py          deterministic reflect padding (copy kernel)
    _encoder2d.py        2-D axis-aware conv + 2-D U-Net encoder
    _partition2d.py      2-D variational partition layer (gating)
    _film_decoder2d.py   2-D FiLM modulation + independent expert decoders
    _proto_helpers.py    marginal variance and tempered-gate helpers
    _model.py            ``GeoVBMILE`` and its registry factories
"""

from __future__ import annotations

from __future__ import annotations

from src.models.vbmile._blocks1d import BasicBlock1D, EncoderBlock1D, UNetEncoder1D
from src.models.vbmile._decoder1d import SharedExpertDecoder1D
from src.models.vbmile._encoder2d import EncoderBlock2D, UNetEncoder2D, _AdaptiveConv2d
from src.models.vbmile._film_decoder2d import ExpertDecoder2D, FiLMBasicBlock2D, SpatialFiLMGenerator
from src.models.vbmile._model import GeoVBMILE
from src.models.vbmile._norm import NormFactory1D, NormFactory2D
from src.models.vbmile._padding import (

    _install_manual_reflect,
    _manual_reflect_forward,
    _manual_reflect_pad_axis,
    _pad_axes,
)
from src.models.vbmile._partition1d import VariationalPartitionLayer1D
from src.models.vbmile._partition2d import VariationalPartitionLayer2D

__all__ = [
    "GeoVBMILE",
    "NormFactory1D",
    "NormFactory2D",
    "BasicBlock1D",
    "EncoderBlock1D",
    "UNetEncoder1D",
    "VariationalPartitionLayer1D",
    "SharedExpertDecoder1D",
    "UNetEncoder2D",
    "EncoderBlock2D",
    "VariationalPartitionLayer2D",
    "SpatialFiLMGenerator",
    "FiLMBasicBlock2D",
    "ExpertDecoder2D",
]
