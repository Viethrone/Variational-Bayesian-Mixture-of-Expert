"""1-D residual block and the 1-D U-Net encoder."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.vbmile._attention1d import CBAM1D
from src.models.vbmile._norm import NormFactory1D


class BasicBlock1D(nn.Module):
    """Two-path residual block: a 1x1 branch plus a dilated-conv branch, followed by CBAM.

    Order: Conv -> Norm -> ReLU -> Dropout.
    """

    def __init__(self, in_channels, out_channels, dropout=0.0, norm_type="group",
                 dilation=5, use_attention=True):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            NormFactory1D.create_norm(norm_type, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.conv3_dilated = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            NormFactory1D.create_norm(norm_type, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention = CBAM1D(channel=out_channels) if use_attention else nn.Identity()

    def forward(self, x):
        out = self.conv1x1(x) + self.conv3_dilated(x)
        return self.attention(out)


class EncoderBlock1D(nn.Module):
    """Downsampling block: ``BasicBlock1D`` followed by a stride-2 convolution."""

    def __init__(self, in_channels, out_channels, norm_type="group"):
        super().__init__()
        self.conv_block = BasicBlock1D(in_channels, out_channels, dropout=0.2, dilation=5,
                                       use_attention=True, norm_type=norm_type)
        self.downsample = nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.downsample(x)
        return x


class UNetEncoder1D(nn.Module):
    """1-D U-Net style encoder returning four feature maps:

        skip1:  (B, base_channels*2, L/2)
        skip2:  (B, base_channels*4, L/4)
        bottom: (B, base_channels*8, L/8)
        global_feat: (B, base_channels, L)
    """

    def __init__(self, in_channels=4, base_channels=8, norm_type="group"):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * 2, 3, padding=1, bias=False),
            NormFactory1D.create_norm(norm_type, in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels * 2, base_channels, 1, bias=False),
            NormFactory1D.create_norm(norm_type, base_channels),
            nn.ReLU(inplace=True),
        )
        self.enc1 = EncoderBlock1D(base_channels, base_channels * 2, norm_type)
        self.enc2 = EncoderBlock1D(base_channels * 2, base_channels * 4, norm_type)
        self.bottleneck_conv = EncoderBlock1D(base_channels * 4, base_channels * 8, norm_type)

    def forward(self, x):
        global_feat = self.input_proj(x)  # (B, base_channels, L)
        skip1 = self.enc1(global_feat)  # (B, base_channels*2, L/2)
        skip2 = self.enc2(skip1)  # (B, base_channels*4, L/4)
        bottom = self.bottleneck_conv(skip2)  # (B, base_channels*8, L/8)
        return [skip1, skip2, bottom, global_feat]


# ============================================================================
# Variational partition layer (gating, 1-D)
# ============================================================================
