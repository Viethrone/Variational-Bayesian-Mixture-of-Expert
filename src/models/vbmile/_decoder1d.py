"""1-D expert decoders: a shared trunk plus per-expert adapters and output heads.

The output bias and log-variance can be initialised from the GMM prior
(``prior_means`` / ``prior_logvars``), so the experts start at the prior modes.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vbmile._blocks1d import BasicBlock1D
from src.models.vbmile._norm import NormFactory1D


class SharedExpertDecoder1D(nn.Module):
    """Multi-expert decoder: shared convolutions and upsampling, one adapter + head per expert.

    ``prior_means`` / ``prior_logvars`` initialise the output bias and log-variance.

    ``share="independent"`` (default ``"shared"`` = the released path) makes the three
    most expressive modules (``conv2`` / ``conv1`` / ``global_residual``) per-expert;
    the ``up_*`` upsamplers stay shared.

    ``full_covariance`` adds a 3-channel off-diagonal head (``b1,b2,b3`` ->
    ``L[1,0], L[2,0], L[2,1]``) so the expert posterior is ``Sigma = L L^T``. The 6-channel
    main head is untouched (the diagonal still comes from ``logvar``); zero-init on the
    last layer makes ``b = 0`` bit-wise identical to the diagonal path.
    """

    def __init__(self, skip_channels, num_experts, norm_type="group",
                 prior_means=None, prior_logvars=None, share="shared",
                 full_covariance=False):
        super().__init__()
        if share not in ("shared", "independent"):
            raise ValueError(f"share must be shared/independent, got {share!r}")
        self.share = str(share)
        self.full_covariance = bool(full_covariance)
        self.num_experts = num_experts
        c1, c2, c3, c4 = skip_channels  # c1: skip1, c2: skip2, c3: bottom, c4: global_feat

        # shared transposed-conv upsampling (three stages)
        self.up_bottom = nn.ConvTranspose1d(c3, c3, 4, 2, 1, bias=False)
        self.up_mid = nn.ConvTranspose1d(c2, c2, 4, 2, 1, bias=False)
        self.up_final = nn.ConvTranspose1d(c1, c1, 4, 2, 1, bias=False)

        # shared conv blocks (multi-scale fusion)
        self.conv2 = BasicBlock1D(c3 + c2, c2, use_attention=False, norm_type=norm_type, dilation=3)
        self.conv1 = BasicBlock1D(c2 + c1, c1, use_attention=False, norm_type=norm_type, dilation=3)

        # global residual branch: global_feat -> 6 channels (shared by all experts)
        self.global_residual = nn.Conv1d(c4, 6, 1, bias=False)

        # per-expert adapters (lightweight conv + norm)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(c1, c1, 3, dilation=3, padding=3, bias=False),
                NormFactory1D.create_norm(norm_type, c1),
            ) for _ in range(num_experts)
        ])
        # per-expert output convolution (1x1, 6 channels)
        self.out_convs = nn.ModuleList([
            nn.Conv1d(c1, 6, 1, bias=True) for _ in range(num_experts)
        ])

        # initialise the output bias from the prior
        if prior_means is not None and prior_logvars is not None:
            assert prior_means.shape == (num_experts, 3) and prior_logvars.shape == (num_experts, 3)
            with torch.no_grad():
                for k in range(num_experts):
                    self.out_convs[k].bias.data[:3] = prior_means[k]
                    self.out_convs[k].bias.data[3:] = prior_logvars[k]

        # optional per-expert branch, kept at the end of __init__ so share="shared" (default)
        # consumes no extra RNG and the shared modules keep their released initialisation
        # even when share="independent".
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

        # full-covariance off-diagonal head (off = diagonal posterior). Only the 3 new channels
        # are added; the 6-channel head still carries the diagonal.
        # zero-init is required: step 0 must equal the diagonal path.
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
        """Transposed-conv upsampling followed by interpolation to the exact length."""
        x = conv(x)
        if x.shape[-1] != target_size:
            x = F.interpolate(x, size=target_size, mode="linear", align_corners=False)
        return x

    def forward(self, features, target_size, z_soft):
        """
        Args:
            features: [skip1, skip2, bottom, global_feat]
            target_size: final sequence length L
            z_soft: (B, K, L) soft assignment
        Returns:
            (B, K, 6, L) per-expert output (3 mu + 3 logvar);
            ``full_covariance`` appends the 3 off-diagonal channels (b1, b2, b3).
        """
        skip1, skip2, bottom, global_feat = features
        independent = self.share == "independent"
        # the shared branch computes the global residual once; the independent branch does it per expert
        global_out = None if independent else self.global_residual(global_feat)  # (B, 6, L)

        up_bottom = self._upsample_conv(bottom, skip2.shape[-1], self.up_bottom)

        mus, logvars, offs = [], [], []
        for k in range(self.num_experts):
            w = z_soft[:, k:k + 1]  # (B,1,L)
            conv2 = self.conv2s[k] if independent else self.conv2
            conv1 = self.conv1s[k] if independent else self.conv1
            g_res = self.global_residuals[k](global_feat) if independent else global_out

            # ---- fuse skip2 ----
            w_skip2 = F.interpolate(w, size=skip2.shape[-1], mode="linear", align_corners=False)
            skip2_weighted = skip2 + skip2 * w_skip2
            x = torch.cat([up_bottom, skip2_weighted], dim=1)
            x = conv2(x)  # (B, c2, L/4)

            # ---- fuse skip1 ----
            w_skip1 = F.interpolate(w, size=skip1.shape[-1], mode="linear", align_corners=False)
            skip1_weighted = skip1 + skip1 * w_skip1
            x_up = self._upsample_conv(x, skip1.shape[-1], self.up_mid)
            x = torch.cat([x_up, skip1_weighted], dim=1)
            x = conv1(x)  # (B, c1, L/2)

            # ---- final upsample to the target length, then modulate by the gate ----
            x = self._upsample_conv(x, target_size, self.up_final)
            x = x + x * w  # residual modulation

            # ---- per-expert head ----
            x = self.adapters[k](x)
            out = self.out_convs[k](x) + g_res  # (B, 6, L)
            if out.shape[-1] != target_size:
                out = F.interpolate(out, size=target_size, mode="linear", align_corners=False)

            mu = out[:, :3]
            logvar = out[:, 3:].clamp(min=-5, max=2)  # keep log-variance in range
            mus.append(mu)
            logvars.append(logvar)
            if self.full_covariance:
                # off-diagonal branch: own head + own global residual (both zero-init)
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
# U-Net encoder (2-D)
# ============================================================================
