"""Helpers shared by both model forwards: marginal variance and the tempered gate."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _cov_diagonal(cov: torch.Tensor) -> torch.Tensor:
    """Diagonal of a prior covariance: full (K,D,D) -> (K,D); diag (K,D) passes through."""
    if cov.ndim == 3:
        return torch.diagonal(cov, dim1=-2, dim2=-1)
    return cov


def _tempered_gate(logits: torch.Tensor, temperature, prior=None):
    """Recompute the gate soft weights and ``KL(q(z|x) || p(z))`` at a given temperature.

    Mirrors the second half of ``VariationalPartitionLayer1D.forward`` (softmax and
    log_softmax computed separately, same KL), so ``temperature == 1`` reproduces the
    partition layer bit for bit; the layer itself is untouched.

    Args:
        logits: (B, K, ...) partition output (2-D already applies its own ``tau=0.7``).
        temperature: annealing temperature (tensor or float) applied to ``z_soft`` itself.
        prior: (B, K, ...) optional spatial prior.
    Returns:
        (probs, kl_z): soft gate (B, K, ...) and the scalar KL.
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
