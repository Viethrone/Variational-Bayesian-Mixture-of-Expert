"""2-D variational partition layer (gating): spatially varying ``q(z | x)``."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vbmile._encoder2d import _AdaptiveConv2d
from src.models.vbmile._norm import NormFactory2D


class VariationalPartitionLayer2D(nn.Module):
    """Produce ``q(z | x)`` logits and ``KL(q(z) || p(z))``.

    Input: low-frequency model + global feature, concatenated on the channel axis.
    """

    def __init__(self, in_channels: int, num_experts: int = 3, norm_type: str = "group",
                 tau: float = 0.7, aniso: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.tau = tau  # gate temperature
        # aniso=True lets a dilated conv fall back to replicate padding on narrow windows;
        # aniso=False (default) stays bit-wise identical to the released path.
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
            prior: (B, K, H, W) optional spatial prior probability
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
# Expert decoders (2-D, spatial FiLM)
# ============================================================================
