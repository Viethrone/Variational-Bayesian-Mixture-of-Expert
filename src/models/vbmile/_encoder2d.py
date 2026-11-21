"""2-D U-Net encoder and its axis-aware convolution.

``_AdaptiveConv2d`` falls back to replicate padding when a reflect window would exceed the
trace axis, which is what keeps narrow windows runnable.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vbmile._norm import NormFactory2D
from src.models.vbmile._padding import (
    _manual_reflect_forward,
    _pad_axes,
)


class _AdaptiveConv2d(nn.Conv2d):
    """Conv2d subclass with size-aware padding (same parameter shapes/keys as nn.Conv2d).

    ``aniso=False`` (default): the forward matches the parent bit for bit.
    ``aniso=True`` checks each axis independently:
      * an output size below 1 (a dilated conv on a narrow window) raises that axis' padding
        to ``dilation`` (size-preserving for k=3, s=1);
      * reflect padding larger than the axis falls back to replicate.
    When neither triggers, the parent path runs and the numerics are unchanged.

    ``manual_reflect=True`` (deterministic protocol) routes reflect through
    ``_manual_reflect_pad_axis``: same numbers as ``F.pad(mode="reflect")``, but deterministic on CUDA.
    """

    def __init__(self, *args, aniso: bool = False, manual_reflect: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.aniso = bool(aniso)
        self.manual_reflect = bool(manual_reflect)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.aniso and not self.manual_reflect:
            return super().forward(x)
        h, w = x.shape[-2:]
        (dh, dw) = self.dilation
        (kh, kw) = self.kernel_size
        (sh, sw) = self.stride
        ph, pw = self.padding
        if self.aniso:
            if (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1 < 1:
                ph = dh * (kh - 1) // 2
            if (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1 < 1:
                pw = dw * (kw - 1) // 2
        reflect_illegal = self.padding_mode == "reflect" and (h <= ph or w <= pw)
        if (ph == self.padding[0] and pw == self.padding[1]
                and not reflect_illegal and not (self.manual_reflect
                                                 and self.padding_mode == "reflect")):
            return super().forward(x)
        y = _pad_axes(x, ph, pw, self.padding_mode, manual_reflect=self.manual_reflect)
        return F.conv2d(y, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class EncoderBlock2D(nn.Module):
    """Downsampling block: a BasicBlock2D followed by a stride-2 convolution.

    ``downsample_mode``:

    * ``"iso"`` (default) — both axes shrink by a factor of 2, which is bit-wise
      identical to the released checkpoint;
    * ``"aniso"`` — decided per axis: an axis whose legacy downsampled size would fall
      below ``min_down_size`` keeps its width instead (conv pad 1, stride 1). The depth
      axis (e.g. 1200) always takes the stride-2 path, so a narrow trace window (e.g.
      width 5) no longer collapses — with ``iso`` the smallest feasible width is 13.
    """

    def __init__(self, in_channels: int, out_channels: int, norm_type: str = "group",
                 downsample_mode: str = "iso", min_down_size: int = 3):
        super().__init__()
        if downsample_mode not in ("iso", "aniso"):
            raise ValueError(f"downsample_mode must be iso/aniso, got {downsample_mode!r}")
        self.downsample_mode = downsample_mode
        self.min_down_size = min_down_size
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=0, bias=False, padding_mode="reflect"),
            NormFactory2D.create_norm(norm_type, out_channels),
            nn.GELU(),
        )
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, bias=False)

    def _axis_plan(self, n: int) -> tuple:
        """Per-axis (padding, stride) decision: keep the legacy path only if the axis stays >= min_down_size.

        legacy chain = conv_block(pad0, n -> n-2) + downsample(stride 2, -> (n-5)//2+1).
        """
        if n >= 5 and (n - 5) // 2 + 1 >= self.min_down_size:
            return 0, 2
        return 1, 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample_mode == "iso":
            return self.downsample(self.conv_block(x))
        h, w = x.shape[-2:]
        ph, sh = self._axis_plan(h)
        pw, sw = self._axis_plan(w)
        conv = self.conv_block[0]
        # deterministic protocol: this branch bypasses the module forward and calls
        # F.conv2d directly, so manual_reflect must be forwarded explicitly (otherwise
        # reflection_pad2d_backward_cuda is still hit)
        mr = bool(getattr(conv, "manual_reflect", False)) or (
            getattr(conv.forward, "__func__", None) is _manual_reflect_forward)
        y = _pad_axes(x, ph, pw, conv.padding_mode, manual_reflect=mr)
        y = F.conv2d(y, conv.weight, None, stride=1, padding=0)
        y = self.conv_block[1](y)
        y = self.conv_block[2](y)
        ds = self.downsample
        y = _pad_axes(y, ph, pw, ds.padding_mode)
        return F.conv2d(y, ds.weight, None, stride=(sh, sw), padding=0)


class UNetEncoder2D(nn.Module):
    """2-D U-Net encoder returning multi-scale features:

        skip1: (B, base*2, H/2, W/2)
        skip2: (B, base*4, H/4, W/4)
        global_feat: (B, base, H, W)

    ``downsample_mode="aniso"`` adapts the downsampling per axis (see EncoderBlock2D):
    depth is halved as usual, while a narrow trace axis keeps its width.
    """

    def __init__(self, in_channels: int, base_channels: int = 8, norm_type: str = "group",
                 downsample_mode: str = "iso", min_down_size: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, bias=False, padding_mode="replicate"),
            NormFactory2D.create_norm(norm_type, in_channels * 2),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, base_channels, 1, bias=False),
            NormFactory2D.create_norm(norm_type, base_channels),
            nn.GELU(),
        )
        self.enc1 = EncoderBlock2D(base_channels, base_channels * 2, norm_type,
                                   downsample_mode, min_down_size)
        self.enc2 = EncoderBlock2D(base_channels * 2, base_channels * 4, norm_type,
                                   downsample_mode, min_down_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_feat = self.input_proj(x)
        skip1 = self.enc1(global_feat)
        skip2 = self.enc2(skip1)
        return skip1, skip2, global_feat


# ============================================================================
# Variational partition layer (gating, 2-D)
# ============================================================================
