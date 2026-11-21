"""2-D FiLM modulation and the fully independent per-expert decoders."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vbmile._encoder2d import _AdaptiveConv2d
from src.models.vbmile._norm import NormFactory2D


class SpatialFiLMGenerator(nn.Module):
    """Spatial FiLM generator: turn the gate probabilities into scale and shift maps."""

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
            w_prob: (B, 1, H, W) gate probability
        Returns:
            (B, 2 * out_channels, H, W) concatenated scale and shift
        """
        x = self.conv1(w_prob)
        x = self.norm(x)
        x = self.relu(x)
        out = self.conv2(x)
        scale, shift = out.chunk(2, dim=1)
        scale = scale * self.modulation_scale
        return torch.cat([scale, shift], dim=1)


class FiLMBasicBlock2D(nn.Module):
    """Two-path residual block with spatial FiLM modulation (one generator per expert)."""

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
            w_prob: (B, 1, H, W) gate probability of the current expert
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
    """Fully independent expert decoders (separate weights per expert).

    Every expert processes skip1 / skip2 / global_feat on its own and returns
        μ_vp, μ_vs, μ_ρ, logσ²_vp, logσ²_vs, logσ²_ρ

    Structural points (a single form, no legacy branch):

    * ``skip2`` is convolved at its native resolution before upsampling: the parameter
        count is unchanged while the convolution work on the trace axis halves;
    * ``final_convs`` / ``global_residual`` use ``padding = dilation`` to keep ``(H, W)``,
        so the tail has no valid conv + bilinear write-back;
    * one 6-channel head per expert (``[mu x3, log sigma^2 x3]``) replaces three 2-channel
        heads: 14.2% fewer parameters, no measurable accuracy change;
    * ``expert_dilations`` sets the per-expert receptive field (all 7 by default). It only
        changes dilation - never the parameter count or the state-dict keys: ``[7, 15, 7]``
        widens the middle expert's fusion field from 7 to 15.
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
        # per-expert receptive field, validated before conv_fuses reads it
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

        # ---------- global residual branch (shared) ----------
        # padding=dilation (dil=3 -> pad=3) keeps (H, W);
        # aniso=True falls back to replicate padding on narrow windows.
        self.global_residual = nn.Sequential(
            _AdaptiveConv2d(c4, c4, 3, dilation=3, padding=3, bias=False, aniso=aniso),
            NormFactory2D.create_norm(norm_type, c4),
            nn.GELU(),
            nn.Conv2d(c4, 6, 1, bias=True),  # 6 channels, matching the expert output layout
        )

        # ---------- per-expert modules ----------
        self.skip2_convs = nn.ModuleList()
        self.final_convs = nn.ModuleList()
        self.conv_fuses = nn.ModuleList()
        # one 6-channel head per expert: [mu_vp, mu_vs, mu_rho, log_var_vp, log_var_vs, log_var_rho]
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
                # padding=dilation=5 keeps (H, W); aniso handles narrow windows
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

        # ---------- prior initialisation (mu / log-var bias of the 6-channel head) ----------
        # channel order: [mu_vp, mu_vs, mu_rho, log_var_vp, log_var_vs, log_var_rho]
        if prior_means is not None and prior_logvars is not None:
            with torch.no_grad():
                for k in range(num_experts):
                    b = self.out_merged[k][-1].bias.data
                    b[0], b[1], b[2] = (prior_means[k, 0], prior_means[k, 1],
                                        prior_means[k, 2])
                    b[3], b[4], b[5] = (prior_logvars[k, 0], prior_logvars[k, 1],
                                        prior_logvars[k, 2])

        # full-covariance off-diagonal head (off = diagonal posterior): the three free entries
        # b1 -> L[1,0], b2 -> L[2,0], b3 -> L[2,1]. A separate 3-channel head (mixing them into
        # one 9-channel convolution converges worse because of the mixed scales).
        # zero-init (weights and bias) => step-0 forward equals the diagonal path.
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
            # same source as the mu/sigma global residual, but its own zero-init branch
            self.global_residual_off = nn.Conv2d(c4, 3, 1, bias=False)
            with torch.no_grad():
                self.global_residual_off.weight.zero_()

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                target_size: Tuple[int, int], z_soft: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (skip1, skip2, global_feat), each (B, C, H, W)
            target_size: (H_out, W_out) spatial size of the output
            z_soft: (B, num_experts, H, W) gate probability (soft assignment)
        Returns:
            expert_outputs: (B, num_experts, 6, H_out, W_out)
                the 6 channels are mu_vp, mu_vs, mu_rho, log_var_vp, log_var_vs, log_var_rho
        """
        skip1, skip2, global_feat = features

        # global residual (shared): padding=dilation keeps (H, W) at full resolution
        global_out = self.global_residual(global_feat)

        expert_outputs = []
        for k in range(self.num_experts):
            w_prob = z_soft[:, k:k + 1]  # (B,1,H,W)
            w_prob_s1 = F.interpolate(w_prob, size=skip1.shape[-2:], mode="bilinear")

            # skip2: convolve at native resolution, then upsample to the skip1 size
            s2 = self.skip2_convs[k](skip2)
            up_skip2 = F.interpolate(s2, size=skip1.shape[-2:], mode="bilinear",
                                     align_corners=False)

            # fuse skip1 with the upsampled skip2
            x = torch.cat([up_skip2, skip1], dim=1)  # (B, c2+c1, H, W)
            x = self.conv_fuses[k](x, w_prob_s1)

            # upsample to the target size; final_convs preserves it
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            x = self.final_convs[k](x)  # (B, c1, H_out, W_out)

            o = self.out_merged[k](x)   # (B,6,H_out,W_out)
            mu, logvar = o[:, :3], o[:, 3:]

            # add the global residual
            mu = mu + global_out[:, :3]
            logvar = logvar + global_out[:, 3:]
            logvar = logvar.clamp(min=-3, max=3)

            out = torch.cat([mu, logvar], dim=1)
            if self.full_covariance:
                # off-diagonal branch (zero-init => identity at step 0); global_residual_off is
                # independent of the shared residual so b is not tied to the spatial mu structure.
                off = self.off_heads[k](x) + self.global_residual_off(global_feat)
                out = torch.cat([out, off], dim=1)
            expert_outputs.append(out)

        # stack the expert outputs: (B, num_experts, 6|9, H, W)
        return torch.stack(expert_outputs, dim=1)


# ============================================================================
# main model: variational Bayesian mixture of experts (one dim-parameterised implementation)
# ============================================================================
