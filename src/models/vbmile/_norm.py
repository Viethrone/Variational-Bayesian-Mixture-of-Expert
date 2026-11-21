"""NormLayer factories (1-D / 2-D): a config string becomes an ``nn.Module``.

``group`` lowers the group count until it divides ``channels`` (``GroupNorm`` would
otherwise raise); ``layer`` is realised as ``GroupNorm(1, C)``.
"""

from __future__ import annotations

import torch.nn as nn


class NormFactory1D:
    """1-D norm factory: batch / group / instance / layer / none."""

    @staticmethod
    def create_norm(norm_type: str, channels: int, groups: int = 8) -> nn.Module:
        norm_type = norm_type.lower()
        if norm_type == "batch":
            return nn.BatchNorm1d(channels)
        if norm_type == "group":
            actual_groups = min(groups, channels)
            while channels % actual_groups != 0 and actual_groups > 1:
                actual_groups -= 1
            return nn.GroupNorm(actual_groups, channels)
        if norm_type == "instance":
            return nn.InstanceNorm1d(channels, affine=True)
        if norm_type == "layer":
            return nn.GroupNorm(1, channels)
        if norm_type == "none":
            return nn.Identity()
        raise ValueError(f"Unsupported norm type: {norm_type}")


class NormFactory2D:
    """2-D norm factory: batch / group / instance / layer / none."""

    @staticmethod
    def create_norm(norm_type: str, channels: int, groups: int = 8) -> nn.Module:
        norm_type = norm_type.lower()
        if norm_type == "batch":
            return nn.BatchNorm2d(channels)
        if norm_type == "group":
            actual_groups = min(groups, channels)
            while channels % actual_groups != 0 and actual_groups > 1:
                actual_groups -= 1
            return nn.GroupNorm(actual_groups, channels)
        if norm_type == "instance":
            return nn.InstanceNorm2d(channels, affine=True)
        if norm_type == "layer":
            return nn.GroupNorm(1, channels)  # LayerNorm is GroupNorm over a single group
        if norm_type == "none":
            return nn.Identity()
        raise ValueError(f"Unsupported norm type: {norm_type}")


# ============================================================================
# Attention (1-D)
# ============================================================================
