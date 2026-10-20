"""Geo-VB-MILE network: variational Bayesian mixture of experts (VB-MoE).

Architecture:

* a U-Net encoder (1-D / 2-D) producing multi-scale features
  ``skip1 / skip2`` (and ``bottom`` for 1-D) plus ``global``;
* a variational partition layer (gating network) emitting ``q(z|x)`` logits and
  providing ``KL(q(z) || p(z))``;
* expert decoders — 1-D: a shared convolution followed by per-expert adapter / output
  convolutions (prior-biased initialization); 2-D: fully independent per-expert
  ``skip2 / final`` plus spatial FiLM and a three-parameter decoupled output head;
* a marginal distribution obtained by fusing the expert distributions with the gating
  probabilities (first / second moments combined).

Sub-module class and attribute names (``encoder / partition_layer / experts`` and the
buffers ``current_temperature / prior_means / prior_vars``) are frozen, so the state
dict stays compatible with the released 1-D checkpoint.
"""

from __future__ import annotations

import math
import types
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import MODELS
from src.utils.fullcov import (
    build_chol,
    chol_apply,
    chol_columns,
    chol_diag,
    chol_trace_term,
    qform,
)

__all__ = [
    "GeoVBMILE",
    "NormFactory1D",
    "NormFactory2D",
    "CBAM1D",
    "BasicBlock1D",
    "EncoderBlock1D",
    "UNetEncoder1D",
    "VariationalPartitionLayer1D",
    "SharedExpertDecoder1D",
    "UNetEncoder2D",
    "VariationalPartitionLayer2D",
    "SpatialFiLMGenerator",
    "FiLMBasicBlock2D",
    "ExpertDecoder2D",
]

# ============================================================================
# 归一化工厂
# ============================================================================


class NormFactory1D:
    """1D 归一化层工厂，支持 batch / group / instance / layer / none（notebook 逐字移植）。"""

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
    """2D 归一化层工厂，支持 batch / group / instance / layer / none（notebook 逐字移植）。"""

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
            return nn.GroupNorm(1, channels)  # LayerNorm 等价于 GroupNorm(1)
        if norm_type == "none":
            return nn.Identity()
        raise ValueError(f"Unsupported norm type: {norm_type}")


# ============================================================================
# 注意力模块（1D）
# ============================================================================


class ChannelAttention1D(nn.Module):
    """通道注意力（SE 风格），使用平均池化和最大池化。"""

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
    """空间注意力，通过 1D 卷积学习注意力权重。"""

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
    """CBAM 模块，顺序应用通道注意力和空间注意力，并带有残差连接。"""

    def __init__(self, channel: int, reduction: int = 4, spatial_kernel: int = 5):
        super().__init__()
        self.channel_attention = ChannelAttention1D(channel, reduction)
        self.spatial_attention = SpatialAttention1D(spatial_kernel)

    def forward(self, x):
        return x + self.spatial_attention(self.channel_attention(x))


# ============================================================================
# 基础构建块（1D）
# ============================================================================


class BasicBlock1D(nn.Module):
    """双路径残差块：1x1 分支 + 空洞卷积分支，相加后经 CBAM 注意力。

    顺序: Conv -> Norm -> ReLU -> Dropout（notebook 逐字移植）。
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
    """下采样块：BasicBlock1D + 步长2卷积下采样。"""

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
    """一维 U-Net 风格编码器，输出四个特征：

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
# 变分分区层（门控网络，1D）
# ============================================================================


class VariationalPartitionLayer1D(nn.Module):
    """输出 q(z|x) 的 logits，并提供 KL(q(z) || p(z))。输入为低频模型。"""

    def __init__(self, in_channels, num_experts=3, norm_type="batch"):
        super().__init__()
        self.num_experts = num_experts
        hidden_dim = min(max(in_channels * 2, num_experts * 4), 24)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1, bias=False),
            NormFactory1D.create_norm(norm_type, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 3, dilation=7, padding=7, bias=False),
            NormFactory1D.create_norm(norm_type, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, num_experts, 1, bias=True),
        )

    def forward(self, low_freq, prior=None):
        """
        Args:
            low_freq: (B, C_low, L) 低频模型
            prior:    (B, K, L) 可选的先验空间分布（各向异性正则提供的概率图）
        Returns:
            logits: (B, K, L)
            kl_z:   标量 KL 散度
        """
        logits = self.net(low_freq)  # (B, K, L)
        probs = F.softmax(logits, dim=1)  # (B, K, L)
        log_probs = F.log_softmax(logits, dim=1)  # (B, K, L)

        if prior is not None:
            log_prior = torch.log(prior + 1e-12)  # (B, K, L)
        else:
            log_prior = -math.log(self.num_experts) * torch.ones_like(log_probs)

        kl_per_pixel = (probs * (log_probs - log_prior)).sum(dim=1)  # (B, L)
        kl_z = kl_per_pixel.mean()
        return logits, kl_z


# ============================================================================
# 共享专家解码器（1D）
# ============================================================================


class SharedExpertDecoder1D(nn.Module):
    """多专家解码器：共享卷积与上采样路径，每专家独立适配器 + 输出卷积。

    支持通过 prior_means / prior_logvars 初始化输出偏置，利用先验加速收敛。

    ``share="independent"``（A1，默认 ``"shared"`` 为 legacy 逐位口径）：把专家
    表达力交集最大的三处 —— ``conv2`` / ``conv1`` / ``global_residual`` —— 改为
    **逐专家独立**（``up_*`` 上采样仍共享）。

    ``full_covariance``：额外输出 3 通道 off-diagonal（``b1,b2,b3`` →
    ``L[1,0], L[2,0], L[2,1]``），使专家后验为 ``Σ = LLᵀ``。只加 3 通道的头，
    不改 6 通道主头（对角仍由 ``logvar`` 给出）；末层 zero-init ⇒ ``b=0`` 时与
    对角路径逐位相同。
    """

    def __init__(self, skip_channels, num_experts, norm_type="group",
                 prior_means=None, prior_logvars=None, share="shared",
                 full_covariance=False):
        super().__init__()
        if share not in ("shared", "independent"):
            raise ValueError(f"share 只能是 shared/independent，收到 {share!r}")
        self.share = str(share)
        self.full_covariance = bool(full_covariance)
        self.num_experts = num_experts
        c1, c2, c3, c4 = skip_channels  # c1: skip1, c2: skip2, c3: bottom, c4: global_feat

        # 共享的上采样转置卷积（3 个阶段）
        self.up_bottom = nn.ConvTranspose1d(c3, c3, 4, 2, 1, bias=False)
        self.up_mid = nn.ConvTranspose1d(c2, c2, 4, 2, 1, bias=False)
        self.up_final = nn.ConvTranspose1d(c1, c1, 4, 2, 1, bias=False)

        # 共享的卷积块（融合不同尺度特征）
        self.conv2 = BasicBlock1D(c3 + c2, c2, use_attention=False, norm_type=norm_type, dilation=3)
        self.conv1 = BasicBlock1D(c2 + c1, c1, use_attention=False, norm_type=norm_type, dilation=3)

        # 全局残差分支：直接从 global_feat 映射到 6 通道输出（所有专家共享）
        self.global_residual = nn.Conv1d(c4, 6, 1, bias=False)

        # 每个专家的适配器（轻量卷积 + 归一化）
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(c1, c1, 3, dilation=3, padding=3, bias=False),
                NormFactory1D.create_norm(norm_type, c1),
            ) for _ in range(num_experts)
        ])
        # 每个专家的输出卷积 (1x1 卷积，输出 6 通道)
        self.out_convs = nn.ModuleList([
            nn.Conv1d(c1, 6, 1, bias=True) for _ in range(num_experts)
        ])

        # 利用先验初始化输出偏置
        if prior_means is not None and prior_logvars is not None:
            assert prior_means.shape == (num_experts, 3) and prior_logvars.shape == (num_experts, 3)
            with torch.no_grad():
                for k in range(num_experts):
                    self.out_convs[k].bias.data[:3] = prior_means[k]
                    self.out_convs[k].bias.data[3:] = prior_logvars[k]

        # A1：可选的去共享分支。整段置于 __init__ 末尾，使 share="shared"（默认）
        # 不消耗任何额外 RNG；即使 share="independent" 共享模块的初始化也与
        # legacy 逐位相同（新模块只在其后补画）。
        if self.share == "independent":
            self.conv2s = nn.ModuleList([
                BasicBlock1D(c3 + c2, c2, use_attention=False, norm_type=norm_type, dilation=3)
                for _ in range(num_experts)
            ])
            self.conv1s = nn.ModuleList([
                BasicBlock1D(c2 + c1, c1, use_attention=False, norm_type=norm_type, dilation=3)
                for _ in range(num_experts)
            ])
            self.global_residuals = nn.ModuleList([
                nn.Conv1d(c4, 6, 1, bias=False) for _ in range(num_experts)
            ])

        # 全协方差 off-diagonal 头（默认关闭 = 对角后验）。只建"新增"的 3 通道头，
        # 不动 out_convs 的 6 通道口径 ⇒ 对角仍走 logvar。
        # 末层 zero-init 是**必要条件**：第 0 步前向必须恒等于对角路径。
        if self.full_covariance:
            self.off_convs = nn.ModuleList([
                nn.Conv1d(c1, 3, 1, bias=True) for _ in range(num_experts)
            ])
            self.global_residual_off = nn.Conv1d(c4, 3, 1, bias=False)
            with torch.no_grad():
                for conv in self.off_convs:
                    conv.weight.zero_()
                    conv.bias.zero_()
                self.global_residual_off.weight.zero_()

    @staticmethod
    def _upsample_conv(x, target_size, conv):
        """转置卷积上采样，再插值到精确长度。"""
        x = conv(x)
        if x.shape[-1] != target_size:
            x = F.interpolate(x, size=target_size, mode="linear", align_corners=False)
        return x

    def forward(self, features, target_size, z_soft):
        """
        Args:
            features: [skip1, skip2, bottom, global_feat]
            target_size: 最终输出序列长度 L
            z_soft: (B, K, L) 软分配概率
        Returns:
            (B, K, 6, L) 每个专家的 6 通道输出 (mu 3 + logvar 3)；
            ``full_covariance`` 时再追加 3 通道 off-diagonal（b1,b2,b3）。
        """
        skip1, skip2, bottom, global_feat = features
        independent = self.share == "independent"
        # 共享分支沿用 legacy 的"先算一次全局残差"口径；独立分支逐专家各算一份
        global_out = None if independent else self.global_residual(global_feat)  # (B, 6, L)

        up_bottom = self._upsample_conv(bottom, skip2.shape[-1], self.up_bottom)

        mus, logvars, offs = [], [], []
        for k in range(self.num_experts):
            w = z_soft[:, k:k + 1]  # (B,1,L)
            conv2 = self.conv2s[k] if independent else self.conv2
            conv1 = self.conv1s[k] if independent else self.conv1
            g_res = self.global_residuals[k](global_feat) if independent else global_out

            # ---- 融合 skip2 ----
            w_skip2 = F.interpolate(w, size=skip2.shape[-1], mode="linear", align_corners=False)
            skip2_weighted = skip2 + skip2 * w_skip2
            x = torch.cat([up_bottom, skip2_weighted], dim=1)
            x = conv2(x)  # (B, c2, L/4)

            # ---- 融合 skip1 ----
            w_skip1 = F.interpolate(w, size=skip1.shape[-1], mode="linear", align_corners=False)
            skip1_weighted = skip1 + skip1 * w_skip1
            x_up = self._upsample_conv(x, skip1.shape[-1], self.up_mid)
            x = torch.cat([x_up, skip1_weighted], dim=1)
            x = conv1(x)  # (B, c1, L/2)

            # ---- 最终上采样到目标长度，并加上专家概率调制 ----
            x = self._upsample_conv(x, target_size, self.up_final)
            x = x + x * w  # 残差式调制

            # ---- 专家专属处理 ----
            x = self.adapters[k](x)
            out = self.out_convs[k](x) + g_res  # (B, 6, L)
            if out.shape[-1] != target_size:
                out = F.interpolate(out, size=target_size, mode="linear", align_corners=False)

            mu = out[:, :3]
            logvar = out[:, 3:].clamp(min=-5, max=2)  # 限制对数方差范围
            mus.append(mu)
            logvars.append(logvar)
            if self.full_covariance:
                # off-diagonal 半支：独立头 + 独立全局残差（均 zero-init）⇒ 初始恒等
                off = self.off_convs[k](x) + self.global_residual_off(global_feat)
                if off.shape[-1] != target_size:
                    off = F.interpolate(off, size=target_size, mode="linear",
                                        align_corners=False)
                offs.append(off)

        mus = torch.stack(mus, dim=1)  # (B, K, 3, L)
        logvars = torch.stack(logvars, dim=1)
        out = torch.cat([mus, logvars], dim=2)  # (B, K, 6, L)
        if self.full_covariance:
            # (B, K, 9, L): 3 mu + 3 logvar + 3 off-diagonal (fixed channel order)
            return torch.cat([out, torch.stack(offs, dim=1)], dim=2)
        return out


# ============================================================================
# U-Net 编码器（2D）
# ============================================================================


def _manual_reflect_pad_axis(x: torch.Tensor, pad: int, dim: int) -> torch.Tensor:
    """Reflect-pad one axis with plain copy kernels (``narrow`` / ``flip`` / ``cat``).

    ``reflection_pad2d_backward_cuda`` has no deterministic implementation, so strict
    ``torch.use_deterministic_algorithms(True)`` rejects it. This helper reproduces
    ``F.pad(mode="reflect")`` bit-for-bit while staying on deterministic copy kernels,
    so the numerics of the network are unchanged.

    Reflect semantics (the edge element is not repeated)::

        input [a b c d], pad=2  =>  [c b | a b c d | c b]

    ``pad`` may exceed the axis length (as ``F.pad`` allows); the mirror is repeated
    ``(pad - 1) // (n - 1) + 1`` times per side.
    """
    n = int(x.shape[dim])
    if pad <= 0:
        return x
    if n == 1 or pad >= n:
        # single-element axis / pad >= axis length: reflect is undefined (F.pad errors),
        # so fall back to replication; ``_pad_axes`` already routes this case to replicate.
        return torch.cat([x] * (2 * pad + 1), dim=dim)

    left = torch.flip(x.narrow(dim, 1, pad), dims=[dim])
    right = torch.flip(x.narrow(dim, n - 1 - pad, pad), dims=[dim])
    return torch.cat([left, x, right], dim=dim)


def _pad_axes(x: torch.Tensor, pad_h: int, pad_w: int, mode: str,
              manual_reflect: bool = False) -> torch.Tensor:
    """Pad each axis separately (``F.conv2d`` takes no ``padding_mode``).

    ``mode`` follows ``nn.Conv2d.padding_mode`` (zeros / reflect / replicate); an axis
    where reflect is illegal (``pad >=`` that axis size) falls back to replicate while
    the other axis keeps its mode.

    ``manual_reflect=True`` (the deterministic protocol) routes reflect through
    ``_manual_reflect_pad_axis``: plain copy kernels, deterministic on CUDA, and
    numerically identical to ``F.pad(mode="reflect")``.
    """

    def _pad1d(t: torch.Tensor, p: int, dim: int) -> torch.Tensor:
        n = int(t.shape[dim])
        m = mode if (mode != "reflect" or n > p) else "replicate"
        if m == "reflect" and manual_reflect and n > 1:
            return _manual_reflect_pad_axis(t, p, dim)
        return F.pad(t, (p, p, 0, 0) if dim == -1 else (0, 0, p, p),
                     mode="constant" if m == "zeros" else m)

    if pad_w > 0:
        x = _pad1d(x, pad_w, -1)
    if pad_h > 0:
        x = _pad1d(x, pad_h, -2)
    return x


def _manual_reflect_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Hand-written reflect forward for a plain ``nn.Conv2d``.

    Installed per instance by ``_install_manual_reflect`` as a bound method (never by
    patching the class, so other instances are unaffected). Numerically identical to
    ``F.conv2d(..., padding_mode="reflect")``.
    """
    ph, pw = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
    y = _pad_axes(x, ph, pw, self.padding_mode, manual_reflect=True)
    return F.conv2d(y, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


def _install_manual_reflect(module: nn.Module) -> int:
    """Route the reflect padding of every ``nn.Conv2d`` in ``module`` through copy kernels.

    ``reflection_pad2d_backward_cuda`` has no deterministic implementation, so strict
    ``torch.use_deterministic_algorithms(True)`` rejects the forward. Swapping the padding
    for ``narrow`` / ``flip`` / ``cat`` keeps reflect semantics bit-for-bit while making
    the backward pass deterministic on CUDA.

    Returns:
        The number of conv layers that were switched.
    """
    n = 0
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and m.padding_mode == "reflect":
            if isinstance(m, _AdaptiveConv2d):
                m.manual_reflect = True
            else:
                m.forward = types.MethodType(_manual_reflect_forward, m)
            n += 1
    return n


class _AdaptiveConv2d(nn.Conv2d):
    """尺寸自适应补边的 Conv2d 子类（参数形状/键名与 nn.Conv2d 完全一致）。

    ``aniso=False``（默认）：forward 与父类逐位一致，保护历史 checkpoint 数值口径。
    ``aniso=True``：逐轴检查原 padding 的合法性——
      * 输出尺寸 < 1（膨胀卷积落在窄窗上）→ 该轴 padding 提升为 dilation（k=3, s=1
        时尺寸保持不变）；
      * reflect 补边不合法（pad ≥ 该轴尺寸）→ 该轴退回 replicate。
    两项都不触发时走父类原路径，数值不变。

    ``manual_reflect=True``（确定性协议）：reflect 改走 ``_manual_reflect_pad_axis``，
    数值与 ``F.pad(mode="reflect")`` 逐位相同但 CUDA 上确定。
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
        """单轴 (padding, stride) 决策：legacy 下采样后仍 ≥ min_down_size 才走 legacy。

        legacy 链路 = conv_block(pad0, n→n-2) + downsample(stride2, →(n-5)//2+1)。
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
    """二维 U-Net 编码器，输出多尺度特征：

        skip1: (B, base*2, H/2, W/2)
        skip2: (B, base*4, H/4, W/4)
        global_feat: (B, base, H, W)

    ``downsample_mode="aniso"`` 时各轴下采样程度自适应（见 EncoderBlock2D）：
    深度轴照常 /2、/4，窄道轴（< min_down_size 可下采样阈值）保持原宽。
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
# 变分分区层（门控网络，2D）
# ============================================================================


class VariationalPartitionLayer2D(nn.Module):
    """输出 q(z|x) 的 logits，并提供 KL(q(z) || p(z))。

    输入：低频模型 + 全局特征（拼接后通道数 = low_freq_channels + base_channels）。
    """

    def __init__(self, in_channels: int, num_experts: int = 3, norm_type: str = "group",
                 tau: float = 0.7, aniso: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.tau = tau  # 温度参数
        # aniso=True 时膨胀卷积在窄窗（宽 ≤ padding）上自动退回 replicate 补边；
        # aniso=False（默认）与 legacy 逐位一致。
        self.net = nn.Sequential(
            _AdaptiveConv2d(in_channels, num_experts, 3, padding=1, bias=False,
                            padding_mode="reflect", aniso=aniso),
            NormFactory2D.create_norm(norm_type, num_experts),
            nn.GELU(),
            _AdaptiveConv2d(num_experts, num_experts, 3, dilation=5, padding=5, bias=False,
                            padding_mode="reflect", aniso=aniso),
            NormFactory2D.create_norm(norm_type, num_experts),
            nn.GELU(),
            nn.Conv2d(num_experts, num_experts, 1, bias=False),
        )

    def forward(self, low_freq: torch.Tensor, global_feat: torch.Tensor,
                prior: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            low_freq: (B, C_low, H, W)
            global_feat: (B, C_global, H, W)
            prior: (B, K, H, W) 空间先验概率，可选
        Returns:
            logits: (B, K, H, W)
            kl_z: scalar
        """
        x = torch.cat([low_freq, global_feat], dim=1)
        logits = self.net(x) / self.tau
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        if prior is not None:
            log_prior = torch.log(prior + 1e-12)
        else:
            log_prior = -math.log(self.num_experts) * torch.ones_like(log_probs)

        kl_per_pixel = (probs * (log_probs - log_prior)).sum(dim=1)
        kl_z = kl_per_pixel.mean()
        return logits, kl_z


# ============================================================================
# 专家解码器（2D，空间 FiLM 调制）
# ============================================================================


class SpatialFiLMGenerator(nn.Module):
    """空间 FiLM 参数生成器：根据门控概率生成 scale 和 shift。"""

    def __init__(self, out_channels: int, hidden_dim: int = 8,
                 kernel_size: int = 3, modulation_scale: float = 1.0):
        super().__init__()
        self.modulation_scale = modulation_scale
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size, padding=padding, bias=False,
                               padding_mode="reflect")
        self.norm = nn.GroupNorm(min(4, hidden_dim), hidden_dim)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_dim, out_channels * 2, 1)

    def forward(self, w_prob: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w_prob: (B, 1, H, W) 门控概率
        Returns:
            (B, out_channels*2, H, W) 拼接的 scale 和 shift
        """
        x = self.conv1(w_prob)
        x = self.norm(x)
        x = self.relu(x)
        out = self.conv2(x)
        scale, shift = out.chunk(2, dim=1)
        scale = scale * self.modulation_scale
        return torch.cat([scale, shift], dim=1)


class FiLMBasicBlock2D(nn.Module):
    """带空间 FiLM 调制的双路径残差块，每个专家有独立的调制生成器。"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0,
                 norm_type: str = "group", dilation: int = 3, film_hidden_dim: int = 8,
                 film_kernel_size: int = 3, modulation_scale: float = 1.0):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            NormFactory2D.create_norm(norm_type, out_channels),
        )
        self.conv3_dilated = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation,
                      bias=False, padding_mode="replicate"),
            NormFactory2D.create_norm(norm_type, out_channels),
        )
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.film_generator = SpatialFiLMGenerator(
            out_channels, film_hidden_dim, film_kernel_size, modulation_scale
        )

    def forward(self, x: torch.Tensor, w_prob: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            w_prob: (B, 1, H, W) 当前专家的门控概率
        Returns:
            (B, out_channels, H, W)
        """
        out = self.conv1x1(x) + self.conv3_dilated(x)
        film_params = self.film_generator(w_prob)
        scale, shift = film_params.chunk(2, dim=1)
        out = out * (1 + torch.tanh(scale)) + torch.tanh(shift)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class ExpertDecoder2D(nn.Module):
    """完全独立的专家解码器（每个专家权重独立，三参数解耦）。

    每个专家独立处理 skip1, skip2, global_feat，输出：
        μ_vp, μ_vs, μ_ρ, logσ²_vp, logσ²_vs, logσ²_ρ

    结构要点（唯一形态，不再有 legacy 分支）：

    * ``skip2`` 在**原生分辨率**上先卷积、再上采样 —— 参数量与插值顺序无关，
      但卷积算力减半（道轴仍在低分辨率时做卷积）；
    * ``final_convs`` / ``global_residual`` 用 ``padding = dilation`` 保持 ``(H, W)``，
      尾部**没有** valid 卷积 + 双线性回写；
    * 每专家一个 **6 通道**输出头（``[μ×3, logσ²×3]``），替代每属性一个 2 通道头
      （参数 −14.2%，精度无可测量差异）；
    * ``expert_dilations`` 是逐专家感受野（默认全部 7）。它只改 dilation，
      **不改参数量、不改 state-dict 键集**：例如 ``[7, 15, 7]`` 只把中间专家的
      融合层感受野 7→15，给深层硬岩专家更强的外推能力。
    """

    def __init__(self, skip_channels: List[int], num_experts: int, norm_type: str = "group",
                 prior_means: Optional[torch.Tensor] = None,
                 prior_logvars: Optional[torch.Tensor] = None,
                 film_hidden_dim: int = 8, film_kernel_size: int = 3,
                 modulation_scale: float = 0.5, aniso: bool = False,
                 expert_dilations: Optional[List[int]] = None,
                 full_covariance: bool = False):
        super().__init__()
        self.num_experts = num_experts
        # 逐专家感受野（在 conv_fuses 构建循环读取之前完成校验）
        if expert_dilations is not None:
            if len(expert_dilations) != num_experts:
                raise ValueError(f"expert_dilations has {len(expert_dilations)} entries "
                                 f"!= num_experts {num_experts}")
            for d in expert_dilations:
                if int(d) < 1:
                    raise ValueError(f"expert_dilations entries must be >= 1, got {d!r}")
        self.expert_dilations = (None if expert_dilations is None
                                 else [int(d) for d in expert_dilations])
        self.aniso = bool(aniso)
        self.full_covariance = bool(full_covariance)
        c1, c2, c4 = skip_channels  # skip1, skip2, global

        # ---------- 全局残差分支（共享）----------
        # padding=dilation（dil=3 ⇒ pad=3）⇒ 输出保持 (H, W)；
        # aniso=True 时窄窗上自动退回 replicate 补边。
        self.global_residual = nn.Sequential(
            _AdaptiveConv2d(c4, c4, 3, dilation=3, padding=3, bias=False, aniso=aniso),
            NormFactory2D.create_norm(norm_type, c4),
            nn.GELU(),
            nn.Conv2d(c4, 6, 1, bias=True),  # 输出 6 通道，与专家输出格式一致
        )

        # ---------- 每个专家独立的组件 ----------
        self.skip2_convs = nn.ModuleList()
        self.final_convs = nn.ModuleList()
        self.conv_fuses = nn.ModuleList()
        # 每专家一个 6 通道输出头：[μ_vp, μ_vs, μ_ρ, logσ²_vp, logσ²_vs, logσ²_ρ]
        self.out_merged = nn.ModuleList()

        for k in range(num_experts):
            skip2_conv = nn.Sequential(
                _AdaptiveConv2d(c2, c2, 3, dilation=3, padding=3, bias=False,
                                padding_mode="replicate", aniso=aniso),
                NormFactory2D.create_norm(norm_type, c2),
                nn.GELU(),
            )
            self.skip2_convs.append(skip2_conv)

            final_conv = nn.Sequential(
                # padding=dilation=5 ⇒ 输出保持 (H,W)，无需插值回写；aniso 窄窗自动补边
                _AdaptiveConv2d(c1, c1, 3, dilation=5, padding=5, bias=False, aniso=aniso),
                NormFactory2D.create_norm(norm_type, c1),
                nn.GELU(),
            )
            self.final_convs.append(final_conv)

            conv_fuse = FiLMBasicBlock2D(
                in_channels=c2 + c1,
                out_channels=c1,
                norm_type=norm_type,
                # per-expert receptive field (default 7 for every expert)
                dilation=(self.expert_dilations[k] if self.expert_dilations else 7),
                film_hidden_dim=film_hidden_dim,
                film_kernel_size=film_kernel_size,
                modulation_scale=modulation_scale,
            )
            self.conv_fuses.append(conv_fuse)

            self.out_merged.append(nn.Sequential(
                nn.Conv2d(c1, c1, 3, padding=1, bias=False),
                NormFactory2D.create_norm(norm_type, c1),
                nn.GELU(),
                nn.Conv2d(c1, 6, 1, bias=True),
            ))

        # ---------- 先验初始化（6 通道头的 μ / logσ² bias）----------
        # 通道序：[μ_vp, μ_vs, μ_ρ, logσ²_vp, logσ²_vs, logσ²_ρ]
        if prior_means is not None and prior_logvars is not None:
            with torch.no_grad():
                for k in range(num_experts):
                    b = self.out_merged[k][-1].bias.data
                    b[0], b[1], b[2] = (prior_means[k, 0], prior_means[k, 1],
                                        prior_means[k, 2])
                    b[3], b[4], b[5] = (prior_logvars[k, 0], prior_logvars[k, 1],
                                        prior_logvars[k, 2])

        # 全协方差 off-diagonal 头（默认关闭 = 对角后验）。Σ = LLᵀ 的三个自由量
        # b1→L[1,0]、b2→L[2,0]、b3→L[2,1]。单独一个 3 通道头（不与 μ/σ 混在
        # 一个 9 通道卷积里：exp 量纲与原始量纲混在一个头里收敛差）。
        # zero-init（权重与偏置）⇒ 第 0 步前向与对角路径逐位相同。
        if self.full_covariance:
            self.off_heads = nn.ModuleList()
            for _ in range(num_experts):
                head = nn.Sequential(
                    nn.Conv2d(c1, c1, 3, padding=1, bias=False),
                    NormFactory2D.create_norm(norm_type, c1),
                    nn.GELU(),
                    nn.Conv2d(c1, 3, 1, bias=True),
                )
                with torch.no_grad():
                    head[-1].weight.zero_()
                    head[-1].bias.zero_()
                self.off_heads.append(head)
            # 与 μ/σ 的 global_residual 同源、但独立一条（zero-init）
            self.global_residual_off = nn.Conv2d(c4, 3, 1, bias=False)
            with torch.no_grad():
                self.global_residual_off.weight.zero_()

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                target_size: Tuple[int, int], z_soft: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (skip1, skip2, global_feat) 每个形状 (B, C, H, W)
            target_size: (H_out, W_out) 最终输出的空间尺寸
            z_soft: (B, num_experts, H, W) 门控概率（软分配）
        Returns:
            expert_outputs: (B, num_experts, 6, H_out, W_out)
                6 通道依次为: μ_vp, μ_vs, μ_ρ, logσ²_vp, logσ²_vs, logσ²_ρ
        """
        skip1, skip2, global_feat = features

        # 全局残差（共享）：padding=dilation ⇒ 顶层分辨率下尺寸即 (H, W)
        global_out = self.global_residual(global_feat)

        expert_outputs = []
        for k in range(self.num_experts):
            w_prob = z_soft[:, k:k + 1]  # (B,1,H,W)
            w_prob_s1 = F.interpolate(w_prob, size=skip1.shape[-2:], mode="bilinear")

            # skip2：先在原生分辨率卷积，再上采样到 skip1 尺寸
            s2 = self.skip2_convs[k](skip2)
            up_skip2 = F.interpolate(s2, size=skip1.shape[-2:], mode="bilinear",
                                     align_corners=False)

            # 融合 skip1 和 up_skip2
            x = torch.cat([up_skip2, skip1], dim=1)  # (B, c2+c1, H, W)
            x = self.conv_fuses[k](x, w_prob_s1)

            # 上采样到最终目标尺寸；final_convs 保持尺寸
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            x = self.final_convs[k](x)  # (B, c1, H_out, W_out)

            o = self.out_merged[k](x)   # (B,6,H_out,W_out)
            mu, logvar = o[:, :3], o[:, 3:]

            # 添加全局残差
            mu = mu + global_out[:, :3]
            logvar = logvar + global_out[:, 3:]
            logvar = logvar.clamp(min=-3, max=3)

            out = torch.cat([mu, logvar], dim=1)
            if self.full_covariance:
                # off-diagonal 半支（zero-init ⇒ 初始恒等）。global_residual_off 与
                # μ/σ 的共享残差独立，避免 b 被 μ 的空间结构绑定。
                off = self.off_heads[k](x) + self.global_residual_off(global_feat)
                out = torch.cat([out, off], dim=1)
            expert_outputs.append(out)

        # 堆叠所有专家输出: (B, num_experts, 6|9, H, W)
        return torch.stack(expert_outputs, dim=1)


# ============================================================================
# 主模型：变分贝叶斯混合专家（单一实现，dim 参数化）
# ============================================================================


def _cov_diagonal(cov: torch.Tensor) -> torch.Tensor:
    """取协方差的（先验）对角：full (K,D,D) -> (K,D)，diag (K,D) 原样返回。"""
    if cov.ndim == 3:
        return torch.diagonal(cov, dim1=-2, dim2=-1)
    return cov


def _tempered_gate(logits: torch.Tensor, temperature, prior=None):
    """按指定温度重算门控软权重与 KL(q(z|x)||p(z))。

    逐位镜像 ``VariationalPartitionLayer1D.forward`` 的后半段（softmax + log_softmax
    分开算、同一 KL 式），使 ``temperature == 1`` 时与分区层原生输出**逐位相同**；
    分区层本体不动（保 state-dict 与默认路径回归）。

    Args:
        logits: (B, K, ...) 分区层输出（2D 已含其自有 ``tau=0.7`` 缩放）。
        temperature: 退火温度（标量张量或 float），作用于 **z_soft** 而非仅 STE 梯度。
        prior: (B, K, ...) 可选先验空间分布。
    Returns:
        (probs, kl_z)：软门控 (B, K, ...) 与标量 KL。
    """
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=1)
    log_probs = F.log_softmax(scaled, dim=1)

    if prior is not None:
        log_prior = torch.log(prior + 1e-12)
    else:
        log_prior = -math.log(logits.shape[1]) * torch.ones_like(log_probs)

    kl_per_pixel = (probs * (log_probs - log_prior)).sum(dim=1)
    return probs, kl_per_pixel.mean()


class GeoVBMILE(nn.Module):
    """Variational Bayesian mixture-of-experts network for elastic-parameter inversion.

    ``dim=1`` — the single-trace variant (shared decoder + per-expert adapters); its
    attribute names correspond one-to-one with the keys of the released 1-D state dict.
    ``dim=2`` — the section variant (fully independent experts + spatial FiLM), which
    supports two-phase training (``set_phase(1) / set_phase(2)``).

    Args:
        seismic_channels: number of observed angle channels (default 6).
        low_freq_channels: number of low-frequency model channels (default 3: Vp/Vs/Density).
        base_channels: base width of the encoder.
        num_experts: number of experts K.
        prior_means / prior_vars: the parameter prior. It is registered as buffers and
            initializes the output bias. ``prior_vars`` may be diagonal ``(K, 3)`` or
            full covariance ``(K, 3, 3)``; ``KL(theta)`` branches on ``ndim`` inside
            ``forward``.
        encoder_downsample: 2-D encoder downsampling mode.
            ``"iso"`` (default) downsamples both axes by 2, which is bit-wise identical
            to the released checkpoint; ``"aniso"`` keeps any axis that would fall below
            ``min_down_size`` at its original width, which makes narrow trace windows
            (e.g. width 5) workable — with ``iso`` the smallest feasible width is 13.
        min_down_size: smallest axis size allowed to be stride-2 downsampled in
            ``aniso`` mode (default 3).
        z_temperature: when ``True`` the temperature also scales the gating soft weights
            (``z_soft = softmax(logits / tau)``) and ``kl_z`` is recomputed from the same
            tempered probabilities. Default ``False`` keeps the plain
            ``softmax(logits)``. The 2-D partition layer already uses ``tau=0.7``, so the
            effective 2-D temperature is ``0.7 * tau``; keep the final value >= 0.1 so the
            gating gradient region does not get too narrow.
        decoder_share: whether the 1-D expert decoder shares its convolutions.
            ``"shared"`` (default) is the released layout; ``"independent"`` gives every
            expert its own ``conv2`` / ``conv1`` / ``global_residual`` (see
            ``SharedExpertDecoder1D``). ``"independent"`` is rejected for ``dim=2``
            because the 2-D decoders are always independent.
        prior_cov_shrinkage: shrinkage of a full-covariance prior toward a spherical
            target, ``Sigma <- (1-d) * Sigma + d * (tr(Sigma)/d) * I``. Default ``0.0``
            keeps the prior untouched.
        expert_dilations: 2-D only. Optional per-expert dilation of the fusion block,
            e.g. ``[7, 15, 7]``. Default ``None`` = 7 for every expert. A dilation does
            not change the parameter count or the state-dict keys.
        variational_covariance: variational family of every expert posterior.
            ``"diag"`` (default) keeps ``q(theta|z=k) = N(mu_k, diag(sigma_k^2))``; the
            released 1-D checkpoint has a 6-channel output head, so ``"diag"`` is the
            state-dict-compatible layout. ``"full"`` adds three off-diagonal channels
            (``b1 -> L[1,0]``, ``b2 -> L[2,0]``, ``b3 -> L[2,1]``) and models
            ``Sigma_k = L_k L_k^T``; the off-diagonal head is zero-initialised, so a fresh
            ``"full"`` model is bit-wise identical to ``"diag"`` at step 0. The delivery
            1-D/2-D recipes use ``"full"``.
        ``expert_dilations`` and ``variational_covariance`` are the only structural degrees
        of freedom; everything else in the 2-D decoder is fixed (merged 6-channel output
        head, ``skip2`` convolved at its native resolution, same-size tail, copy-kernel
        reflect padding).

    The 2-D-only options raise if they are passed with ``dim=1`` instead of being
    silently inert.
    """

    def __init__(self,
                 dim: int = 1,
                 seismic_channels: int = 6,
                 low_freq_channels: int = 3,
                 base_channels: int = 16,
                 norm_type: str = "group",
                 num_experts: int = 3,
                 initial_temperature: float = 1.0,
                 final_temperature: float = 0.1,
                 prior_means: Optional[torch.Tensor] = None,
                 prior_vars: Optional[torch.Tensor] = None,
                 film_hidden_dim: int = 8,
                 film_kernel_size: int = 3,
                 modulation_scale: float = 1.0,
                 encoder_downsample: str = "iso",
                 min_down_size: int = 3,
                 z_temperature: bool = False,
                 decoder_share: str = "shared",
                 prior_cov_shrinkage: float = 0.0,
                 expert_dilations: Optional[List[int]] = None,
                 variational_covariance: str = "diag"):
        super().__init__()
        if dim not in (1, 2):
            raise ValueError(f"dim 只能是 1 或 2，收到 {dim!r}")
        if variational_covariance not in ("diag", "full"):
            raise ValueError(
                f"variational_covariance 只能是 diag/full，收到 {variational_covariance!r}")
        if dim == 1 and expert_dilations is not None:
            raise ValueError("expert_dilations only applies to dim=2")
        if encoder_downsample not in ("iso", "aniso"):
            raise ValueError(f"encoder_downsample 只能是 iso/aniso，收到 {encoder_downsample!r}")
        if dim == 1 and encoder_downsample != "iso":
            raise ValueError("encoder_downsample=aniso 仅支持 dim=2（1D 编码器无道轴）")
        if decoder_share not in ("shared", "independent"):
            raise ValueError(f"decoder_share 只能是 shared/independent，收到 {decoder_share!r}")
        if dim == 2 and decoder_share != "shared":
            raise ValueError("decoder_share=independent 仅支持 dim=1（2D 专家解码器本就全独立）")
        if not 0.0 <= float(prior_cov_shrinkage) < 1.0:
            raise ValueError(
                f"prior_cov_shrinkage 需在 [0, 1)，收到 {prior_cov_shrinkage!r}")
        # 普通属性（非 buffer）：不进 state_dict
        self.z_temperature = bool(z_temperature)
        self.decoder_share = str(decoder_share)
        self.prior_cov_shrinkage = float(prior_cov_shrinkage)
        self.variational_covariance = str(variational_covariance)
        self.dim = int(dim)
        self.num_experts = num_experts
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.low_freq_channels = low_freq_channels
        self.register_buffer("current_temperature", torch.tensor(initial_temperature))

        # 先验（注册为 buffer，供 KL(θ) 与输出偏置初始化使用）。
        # prior_vars 可为对角 (K, 3)（legacy 口径）或全协方差 (K, 3, 3)
        # （GMM.ipynb cell 13 / 2D vbmile 口径）；KL(θ) 在前向里按 ndim 分支。
        if prior_means is not None:
            assert prior_vars is not None
            prior_means = torch.as_tensor(prior_means, dtype=torch.float32)
            prior_vars = torch.as_tensor(prior_vars, dtype=torch.float32)
            # 协方差收缩（opt-in，默认 0.0 = 逐位不变）：
            # 实测 GMM 全协方差先验在归一化空间近乎奇异（cond 最高 5804、λ_min 3e-4），
            # Σp⁻¹ 的最坏方向增益因此达 3220 → KL 被放大两个数量级（单位先验 KL≈1，
            # GMM 先验 KL≈900）。有限样本拟合出的全协方差本就过自信，标准做法是向
            # 「按维度均摊的球面目标」收缩：Σ ← (1−δ)Σ + δ·(tr(Σ)/d)·I。
            if self.prior_cov_shrinkage > 0.0 and prior_vars.ndim == 3:
                d_attr = prior_vars.shape[-1]
                eye = torch.eye(d_attr, dtype=prior_vars.dtype)
                scale = prior_vars.diagonal(dim1=-2, dim2=-1).mean(dim=-1)   # (K,) = tr/d
                prior_vars = ((1.0 - self.prior_cov_shrinkage) * prior_vars
                              + self.prior_cov_shrinkage * scale[:, None, None] * eye)
            self.register_buffer("prior_means", prior_means)          # (K, 3)
            self.register_buffer("prior_vars", prior_vars)            # (K, 3) 或 (K, 3, 3)
            prior_logvars = torch.log(
                torch.clamp(_cov_diagonal(prior_vars), min=1e-12)     # (K, 3)
            )
        else:
            self.register_buffer("prior_means", torch.zeros(num_experts, 3))
            self.register_buffer("prior_vars", torch.ones(num_experts, 3))
            prior_logvars = torch.zeros(num_experts, 3)

        # 编码器（输入：地震 + 低频）
        encoder_in = seismic_channels + low_freq_channels
        if self.dim == 1:
            self.encoder = UNetEncoder1D(encoder_in, base_channels, norm_type)
            self.skip_channels = [base_channels * 2, base_channels * 4,
                                  base_channels * 8, base_channels]
        else:
            self.encoder = UNetEncoder2D(encoder_in, base_channels, norm_type,
                                         downsample_mode=encoder_downsample,
                                         min_down_size=min_down_size)
            self.skip_channels = [base_channels * 2, base_channels * 4, base_channels]

        # 分区层（2D 额外拼接 global 特征）
        if self.dim == 1:
            self.partition_layer = VariationalPartitionLayer1D(
                low_freq_channels, num_experts, norm_type=norm_type
            )
        else:
            partition_in = low_freq_channels + base_channels
            self.partition_layer = VariationalPartitionLayer2D(
                partition_in, num_experts, norm_type,
                aniso=(encoder_downsample == "aniso"),
            )

        # 专家解码器
        if self.dim == 1:
            self.experts = SharedExpertDecoder1D(
                skip_channels=self.skip_channels,
                num_experts=num_experts,
                norm_type=norm_type,
                prior_means=prior_means,
                prior_logvars=prior_logvars,
                share=self.decoder_share,
                full_covariance=(self.variational_covariance == "full"),
            )
        else:
            self.experts = ExpertDecoder2D(
                self.skip_channels, num_experts, norm_type,
                prior_means=prior_means, prior_logvars=prior_logvars,
                film_hidden_dim=film_hidden_dim, film_kernel_size=film_kernel_size,
                modulation_scale=modulation_scale,
                aniso=(encoder_downsample == "aniso"),
                expert_dilations=expert_dilations,
                full_covariance=(self.variational_covariance == "full"),
            )

        # 确定性协议：把 reflect 补边全部换成纯拷贝算子（数值与 F.pad 逐位相同）。
        # 1D 没有 reflect 卷积，计数为 0。
        self.n_manual_reflect_pads = _install_manual_reflect(self)

    # ---- 训练阶段与温度 ----
    def set_phase(self, phase: int) -> None:
        """Set the training phase.

        phase 1 — joint training (everything trainable);
        phase 2 — freeze the gating, train encoder + decoders;
        phase 3 — joint fine-tuning (everything trainable, gating not frozen).
        """
        if phase in (1, 3):
            self.train()
            for p in self.parameters():
                p.requires_grad = True
        elif phase == 2:
            self.encoder.train()
            self.experts.train()
            self.partition_layer.eval()
            for p in self.encoder.parameters():
                p.requires_grad = True
            for p in self.partition_layer.parameters():
                p.requires_grad = False
            for p in self.experts.parameters():
                p.requires_grad = True
        else:
            raise ValueError("phase must be 1, 2 or 3")

    def set_temperature(self, epoch: int, total_epochs: int) -> None:
        """余弦退火更新 Gumbel-Softmax 温度。"""
        if total_epochs <= 0:
            return
        cos_val = (1 + math.cos(math.pi * epoch / total_epochs)) / 2
        new_temp = self.final_temperature + (self.initial_temperature - self.final_temperature) * cos_val
        self.current_temperature.fill_(new_temp)

    def _prior_precision_and_logdet(self):
        """Prior precision ``Sigma_p^-1`` and ``ln|Sigma_p|``, shapes ``(K,3,3)`` / ``(K,)``.

        Used by the full-covariance path only. The prior is a constant, so the result is
        cached against the ``prior_vars`` buffer version to avoid inverting it every step.
        """
        key = (id(self.prior_vars), int(getattr(self.prior_vars, "_version", 0)))
        cached = getattr(self, "_prior_prec_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        if self.prior_vars.ndim == 3:
            # Same operators as the diagonal branch (inv + slogdet) so that
            # "full with b=0 == diag" can be verified at floating-point level. A 1e-6
            # jitter is added only if the prior is numerically singular -- never silently.
            cov = self.prior_vars
            prec = torch.linalg.inv(cov)
            _, logdet_p = torch.linalg.slogdet(cov)
            if not torch.isfinite(prec).all() or not torch.isfinite(logdet_p).all():
                eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
                cov = cov + 1e-6 * eye
                prec = torch.linalg.inv(cov)
                _, logdet_p = torch.linalg.slogdet(cov)
        else:
            var = self.prior_vars
            prec = torch.diag_embed(1.0 / var.clamp_min(1e-12))          # (K,3,3)
            logdet_p = torch.log(var.clamp_min(1e-12)).sum(dim=-1)        # (K,)
        self._prior_prec_cache = (key, prec, logdet_p)
        return prec, logdet_p

    # ---- 1D 兼容助手 ----
    def encode(self, seismic, low_freq):
        """编码器前向：拼接地震与低频模型（1D 兼容接口）。"""
        encoder_input = torch.cat([seismic, low_freq], dim=1)
        return self.encoder(encoder_input)

    def sample_theta(self, expert_mus, expert_logvars, logits, tau, hard=True,
                     expert_off=None):
        """从混合后验 q(θ|x) 中采样一个样本（Gumbel-Softmax + 重参数化）。

        ``expert_off is None``（对角口径）::

            θ̃ = μ + σ ⊙ ε

        ``expert_off`` 非空（全协方差口径）::

            θ̃ = μ + L ε,   L 为下三角、对角 exp(logvar/2)、off-diagonal 取 expert_off

        这是唯一影响梯度路径的公式：若保留 ``σ⊙ε``，即使头输出 9 通道，MC 估计仍
        只覆盖对角方向，与声明的 full 后验不一致。
        """
        z_hard = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)
        eps = torch.randn_like(expert_mus)
        if expert_off is None:
            theta_samples = expert_mus + torch.exp(0.5 * expert_logvars) * eps
        else:
            L = build_chol(expert_logvars, expert_off)
            theta_samples = expert_mus + chol_apply(L, eps)
        theta = (z_hard.unsqueeze(2) * theta_samples).sum(dim=1)
        return theta

    # ---- 前向 ----
    def forward(self, seismic, low_freq_model=None, prior=None, num_samples=1) -> dict:
        """前向计算。输入形状按 dim 分派：1D (B,C,L)，2D (B,C,H,W)。

        Returns:
            dict 包含 final_pred / theta_sample / expert_mus / expert_logvars /
            z_soft / logits / kl_z，以及中间特征与（1D）kl_theta 或（2D）
            kl_theta_per_expert。
        """
        if self.dim == 1:
            return self._forward_1d(seismic, low_freq_model, prior, num_samples)
        return self._forward_2d(seismic, low_freq_model, prior, num_samples)

    def _forward_1d(self, seismic, low_freq_model, prior, num_samples):
        B, _, L = seismic.shape
        if low_freq_model is None:
            low_freq_model = torch.zeros(B, self.low_freq_channels, L, device=seismic.device)

        features = self.encode(seismic, low_freq_model)  # [skip1, skip2, bottom, global]

        logits, kl_z = self.partition_layer(low_freq_model, prior=prior)
        if self.z_temperature:
            # T2：τ 作用于门控软权重，kl_z 随 tempered probs 同步重算
            z_soft, kl_z = _tempered_gate(logits, self.current_temperature, prior)
        else:
            z_soft = F.softmax(logits, dim=1)  # (B, K, L)

        expert_out = self.experts(features, target_size=L, z_soft=z_soft)  # (B,K,6|9,L)
        expert_mus = expert_out[:, :, :3]  # (B, K, 3, L)
        expert_logvars = expert_out[:, :, 3:6]  # (B, K, 3, L)
        expert_off = (expert_out[:, :, 6:] if self.variational_covariance == "full"
                      else None)  # (B, K, 3, L) 或 None
        full_cov = expert_off is not None

        # ---- KL(θ) 计算 ----
        var = torch.exp(expert_logvars)
        if full_cov:
            # full q ‖ full/diag p 的统一闭式：
            #   KL = ½[tr(Σp⁻¹Σq) + δᵀΣp⁻¹δ − D + ln|Σp| − ln|Σq|]
            cols = chol_columns(expert_logvars, expert_off)
            prec, logdet_p = self._prior_precision_and_logdet()          # (K,3,3), (K,)
            # ln|Σq| = 2·Σ log diag(L) = Σ logvar
            logdet_q = expert_logvars.sum(dim=2)                         # (B,K,L)
            tr_t = chol_trace_term(cols, prec)                          # (B,K,L)
            delta = expert_mus - self.prior_means[None, :, :, None]      # (B,K,3,L)
            qf = qform(delta, prec)                                     # (B,K,L)
            kl_theta_per_expert = 0.5 * (
                tr_t + qf - expert_mus.shape[2] - logdet_q
                + logdet_p[None, :, None]
            )  # (B, K, L)
            # 边际方差口径：full 时 diag(Σ_k) ≠ exp(logvar)，b² 项必须计入
            diag_var = chol_diag(expert_logvars, expert_off)            # (B,K,3,L)
        elif self.prior_vars.ndim == 3:
            # 全协方差先验 p = N(prior_means, prior_vars)：
            # KL(q=diag 变分 ‖ p) = ½[tr(Σp⁻¹V) + ΔᵀΣp⁻¹Δ − D + ln|Σp| − Σ_a logvar_a]
            cov_inv = torch.linalg.inv(self.prior_vars)                  # (K, 3, 3)
            _, logdet_p = torch.linalg.slogdet(self.prior_vars)          # (K,)
            d_attr = expert_mus.shape[2]
            tr_t = torch.einsum("bkdl,kd->bkl", var,
                                cov_inv.diagonal(dim1=-2, dim2=-1))      # (B, K, L)
            delta = expert_mus - self.prior_means[None, :, :, None]      # (B, K, 3, L)
            qf = torch.einsum("bkdl,kde,bkel->bkl", delta, cov_inv, delta)
            kl_theta_per_expert = 0.5 * (
                tr_t + qf - d_attr - expert_logvars.sum(dim=2)
                + logdet_p[None, :, None]
            )  # (B, K, L)
            diag_var = var
        else:
            prior_means_exp = self.prior_means[None, :, :, None]   # (1, K, 3, 1)
            prior_vars_exp = self.prior_vars[None, :, :, None]     # (1, K, 3, 1)
            kl_theta_per_expert = 0.5 * (
                (var / prior_vars_exp)
                + ((expert_mus - prior_means_exp) ** 2) / prior_vars_exp
                - 1.0 - expert_logvars + torch.log(prior_vars_exp)
            ).sum(dim=2)  # (B, K, L)
            diag_var = var
        kl_theta = (z_soft * kl_theta_per_expert).sum(dim=1).mean()

        # ---- 边际分布 q(θ|x)：law of total covariance 的**对角** ----
        mu_marginal = (z_soft.unsqueeze(2) * expert_mus).sum(dim=1)  # (B, 3, L)
        second_moment = (z_soft.unsqueeze(2) * (diag_var + expert_mus ** 2)).sum(dim=1)
        var_marginal = second_moment - mu_marginal ** 2
        var_marginal = torch.clamp(var_marginal, min=1e-6)
        logvar_marginal = torch.log(var_marginal)
        final_pred = torch.cat([mu_marginal, logvar_marginal], dim=1)  # (B, 6, L)

        # ---- 可选采样 ----
        theta_sample = None
        if num_samples > 0:
            theta_sample = self.sample_theta(
                expert_mus, expert_logvars, logits, self.current_temperature, hard=True,
                expert_off=expert_off,
            )

        out = {
            "final_pred": final_pred,
            "theta_sample": theta_sample,
            "expert_mus": expert_mus,
            "expert_logvars": expert_logvars,
            "z_soft": z_soft,
            "logits": logits,
            "kl_z": kl_z,
            "kl_theta": kl_theta,
            "kl_theta_per_expert": kl_theta_per_expert,
            "skip1": features[0],
            "skip2": features[1],
            "bottleneck": features[2],
            "global": features[3],
        }
        if full_cov:
            # full 模式才有的键：供损失侧算专家级期望井似然与 mixture-predictive 口径
            out["expert_off"] = expert_off
            out["expert_diag_var"] = diag_var
            out["expert_tr_sigma"] = diag_var.sum(dim=2)                 # (B,K,L)
        return out

    def _forward_2d(self, seismic, low_freq_model, prior, num_samples):
        B, _, H, W = seismic.shape
        if low_freq_model is None:
            low_freq_model = torch.zeros(B, self.low_freq_channels, H, W, device=seismic.device)

        # 编码
        features = self.encoder(torch.cat([seismic, low_freq_model], dim=1))
        skip1, skip2, global_feat = features

        # 门控
        logits, kl_z = self.partition_layer(low_freq_model, global_feat, prior=prior)
        if self.z_temperature:
            # T2：同 1D；2D 分区层自带 tau=0.7 → 有效温度 0.7·τ
            z_soft, kl_z = _tempered_gate(logits, self.current_temperature, prior)
        else:
            z_soft = F.softmax(logits, dim=1)

        # 专家解码
        expert_out = self.experts([skip1, skip2, global_feat], (H, W), z_soft)
        expert_mus = expert_out[:, :, :3]      # (B, K, 3, H, W)
        expert_logvars = expert_out[:, :, 3:6]  # (B, K, 3, H, W)
        expert_off = (expert_out[:, :, 6:] if self.variational_covariance == "full"
                      else None)  # (B, K, 3, H, W) 或 None
        full_cov = expert_off is not None

        # KL(θ) 计算
        var = torch.exp(expert_logvars)
        if full_cov:
            # full q 闭式（同 1D）；除以 D 与 2D 对角口径的 mean(dim=2) 尺度一致。
            cols = chol_columns(expert_logvars, expert_off)
            prec, logdet_p = self._prior_precision_and_logdet()          # (K,3,3), (K,)
            logdet_q = expert_logvars.sum(dim=2)                         # (B,K,H,W)
            tr_t = chol_trace_term(cols, prec)                          # (B,K,H,W)
            delta = expert_mus - self.prior_means[None, :, :, None, None]
            qf = qform(delta, prec)                                     # (B,K,H,W)
            d_attr = expert_mus.shape[2]
            kl_theta_per_expert = 0.5 * (
                tr_t + qf - d_attr - logdet_q + logdet_p[None, :, None, None]
            ) / d_attr  # (B, K, H, W)
            diag_var = chol_diag(expert_logvars, expert_off)            # (B,K,3,H,W)
        elif self.prior_vars.ndim == 3:
            # 全协方差先验（公式同 1D，见 _forward_1d）；除以 D 与 2D 对角口径的
            # mean(dim=2) 尺度一致。
            cov_inv = torch.linalg.inv(self.prior_vars)                  # (K, 3, 3)
            _, logdet_p = torch.linalg.slogdet(self.prior_vars)          # (K,)
            d_attr = expert_mus.shape[2]
            tr_t = torch.einsum("bkdhw,kd->bkhw", var,
                                cov_inv.diagonal(dim1=-2, dim2=-1))
            delta = expert_mus - self.prior_means[None, :, :, None, None]
            qf = torch.einsum("bkdhw,kde,bkehw->bkhw", delta, cov_inv, delta)
            kl_theta_per_expert = 0.5 * (
                tr_t + qf - d_attr - expert_logvars.sum(dim=2)
                + logdet_p[None, :, None, None]
            ) / d_attr  # (B, K, H, W)
            diag_var = var
        else:
            prior_means_exp = self.prior_means[None, :, :, None, None]
            prior_vars_exp = self.prior_vars[None, :, :, None, None]
            kl_theta_per_expert = 0.5 * (
                var / prior_vars_exp
                + (expert_mus - prior_means_exp).pow(2) / prior_vars_exp
                - 1.0 - expert_logvars + torch.log(prior_vars_exp)
            ).mean(dim=2)  # (B, K, H, W)
            diag_var = var

        # 边际分布（diag(Σ_marg) 用 Σ_k 的对角；full 时含 b² 项）
        mu_marginal = (z_soft.unsqueeze(2) * expert_mus).sum(dim=1)  # (B, 3, H, W)
        second_moment = (z_soft.unsqueeze(2) * (diag_var + expert_mus ** 2)).sum(dim=1)
        var_marginal = second_moment - mu_marginal ** 2
        var_marginal = torch.clamp(var_marginal, min=1e-6)
        logvar_marginal = torch.log(var_marginal)
        final_pred = torch.cat([mu_marginal, logvar_marginal], dim=1)  # (B, 6, H, W)

        # 采样（Gumbel-Softmax 硬采样 + 重参数化）
        # 注意顺序：先重参数化噪声 `eps`、后 Gumbel 门控采样。这决定了两条 RNG 流
        # 的消耗顺序，是 2D notebook / 论文口径的固定约定（1D 走 sample_theta，顺序相反）。
        theta_sample = None
        if num_samples > 0:
            eps = torch.randn_like(expert_mus)
            if full_cov:
                # full：θ̃ = μ + Lε（全协方差路径唯一影响梯度的公式）
                theta_samples = expert_mus + chol_apply(
                    build_chol(expert_logvars, expert_off), eps)
            else:
                theta_samples = expert_mus + torch.exp(0.5 * expert_logvars) * eps
            z_hard = F.gumbel_softmax(logits, tau=self.current_temperature, hard=True, dim=1)
            theta_sample = (z_hard.unsqueeze(2) * theta_samples).sum(dim=1)  # (B, 3, H, W)

        out = {
            "final_pred": final_pred,
            "theta_sample": theta_sample,
            "expert_mus": expert_mus,
            "expert_logvars": expert_logvars,
            "z_soft": z_soft,
            "logits": logits,
            "kl_z": kl_z,
            "kl_theta_per_expert": kl_theta_per_expert,
            "skip1": skip1,
            "skip2": skip2,
            "global": global_feat,
        }
        if full_cov:
            out["expert_off"] = expert_off
            out["expert_diag_var"] = diag_var
            out["expert_tr_sigma"] = diag_var.sum(dim=2)                 # (B,K,H,W)
        return out


def _vbmile_factory(dim: int):
    """按 dim 生成注册工厂：config 的 `_target_: vbmile_1d/vbmile_2d` 入口。"""

    def factory(**kwargs):
        kwargs.pop("dim", None)
        return GeoVBMILE(dim=dim, **kwargs)

    factory.__name__ = f"build_vbmile_{dim}d"
    return factory


MODELS.register_obj("vbmile_1d", _vbmile_factory(1))
MODELS.register_obj("vbmile_2d", _vbmile_factory(2))