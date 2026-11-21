"""1-D variational partition layer (gating).

Emits ``q(z | x)`` logits from the low-frequency model (an optional spatial prior is
injected multiplicatively) and the per-sample ``KL(q(z) || p(z))``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vbmile._norm import NormFactory1D


class VariationalPartitionLayer1D(nn.Module):
    """Produce ``q(z | x)`` logits from the low-frequency model and ``KL(q(z) || p(z))``."""

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
            low_freq: (B, C_low, L) low-frequency model
            prior:    (B, K, L) optional spatial prior probability map
        Returns:
            logits: (B, K, L)
            kl_z:   scalar KL divergence
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
# Shared expert decoder (1-D)
# ============================================================================
