"""CBAM-style attention for the 1-D residual block."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention1D(nn.Module):
    """Channel attention (SE style): average and max pooling share one MLP."""

    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        mid_channel = max(channel // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv1d(channel, mid_channel, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channel, channel, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        avg_out = self.mlp(F.adaptive_avg_pool1d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool1d(x, 1))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention1D(nn.Module):
    """Spatial attention: a 1-D convolution scores every position."""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B,1,L)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,L)
        x_cat = torch.cat([avg_out, max_out], dim=1)  # (B,2,L)
        attention = self.sigmoid(self.conv(x_cat))  # (B,1,L)
        return attention * x


class CBAM1D(nn.Module):
    """CBAM block: channel attention, then spatial attention, with a residual add."""

    def __init__(self, channel: int, reduction: int = 4, spatial_kernel: int = 5):
        super().__init__()
        self.channel_attention = ChannelAttention1D(channel, reduction)
        self.spatial_attention = SpatialAttention1D(spatial_kernel)

    def forward(self, x):
        return x + self.spatial_attention(self.channel_attention(x))


# ============================================================================
# Basic building blocks (1-D)
# ============================================================================
