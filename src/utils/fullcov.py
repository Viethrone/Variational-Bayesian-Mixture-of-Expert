"""Closed-form operators for a 3x3 full-covariance posterior.

Both the model (KL, sampling) and the loss (well likelihood) need the same
factorisation, so it lives here instead of inside either module. A single
implementation is what guarantees that both sides agree on the channel mapping of
``Sigma = L L^T``.

Parameterisation (the only one)::

    L = [[e^{a1}, 0,     0   ],
         [b1,    e^{a2}, 0   ],
         [b2,    b3,     e^{a3}]]

    Sigma = L L^T

``a`` comes from ``logvar`` (``a = logvar / 2``, i.e. ``sigma = exp(logvar / 2)``);
the three ``b`` channels fill the lower triangle in **row-major** order:
``b1 -> L[1,0]``, ``b2 -> L[2,0]``, ``b3 -> L[2,1]``. Filling the upper triangle
instead mixes ``L`` with ``L^T`` and makes the KL asymmetric.

Tensor layout: the property axis is fixed at dim 2 (``(B, K, 3, *spatial)``) and the two
matrix axes follow it (``(B, K, 3, 3, *spatial)``), so ``spatial`` may have any length
(``(L,)`` for 1-D or ``(H, W)`` for 2-D). The helpers flatten the pixel axes internally
instead of growing a separate einsum path per dimensionality.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

__all__ = [
    "chol_columns",
    "build_chol",
    "chol_apply",
    "chol_diag",
    "chol_trace_term",
    "qform",
]


def chol_columns(expert_logvars: torch.Tensor,
                 expert_off: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Assemble the three **columns** of the lower-triangular ``L``.

    Args:
        expert_logvars: ``(B, K, 3, *spatial)`` diagonal log-variances (the caller clamps).
        expert_off: ``(B, K, 3, *spatial)`` the three off-diagonal free parameters.

    Returns:
        ``(c0, c1, c2)``, each ``(B, K, 3, *spatial)``; ``torch.stack([...], dim=-2)`` is
        ``L``.
    """
    sigma = torch.exp(0.5 * expert_logvars)          # (B, K, 3, *spatial)
    b1 = expert_off[:, :, 0]
    b2 = expert_off[:, :, 1]
    b3 = expert_off[:, :, 2]
    zero = torch.zeros_like(sigma[:, :, 0])
    c0 = torch.stack([sigma[:, :, 0], b1, b2], dim=2)      # column 0 of L
    c1 = torch.stack([zero, sigma[:, :, 1], b3], dim=2)    # column 1 of L
    c2 = torch.stack([zero, zero, sigma[:, :, 2]], dim=2)  # column 2 of L
    return c0, c1, c2


def build_chol(expert_logvars: torch.Tensor,
               expert_off: torch.Tensor) -> torch.Tensor:
    """Assemble ``L`` (lower triangular, diagonal ``exp(logvar / 2)``).

    Shape ``(B, K, 3, 3, *spatial)``. The column axis is inserted at dim 3 on purpose:
    ``dim=-2`` would land between ``(H, W)`` in 2-D and ``chol_apply``'s reshape would
    then fail.
    """
    c0, c1, c2 = chol_columns(expert_logvars, expert_off)
    return torch.stack([c0, c1, c2], dim=3)


def qform(x: torch.Tensor, prec: torch.Tensor) -> torch.Tensor:
    """Point-wise quadratic form ``x^T M x`` (``x``: ``(B,K,3,*spatial)``, ``prec``: ``(K,3,3)``)."""
    B, K = x.shape[:2]
    spatial = tuple(x.shape[3:])
    flat = x.reshape(B, K, 3, -1)
    out = torch.einsum("bkds,kde,bkes->bks", flat, prec, flat)
    return out.reshape(B, K, *spatial)


def chol_trace_term(cols: Sequence[torch.Tensor], prec: torch.Tensor) -> torch.Tensor:
    """``tr(M Sigma) = tr(M L L^T) = sum_j c_j^T M c_j`` (three quadratic forms)."""
    return sum(qform(c, prec) for c in cols)


def chol_diag(expert_logvars: torch.Tensor,
              expert_off: torch.Tensor) -> torch.Tensor:
    """Closed-form ``diag(Sigma)``: ``[sigma1^2, b1^2+sigma2^2, b2^2+b3^2+sigma3^2]``."""
    var = torch.exp(expert_logvars)                 # (B,K,3,*spatial)
    b1 = expert_off[:, :, 0]
    b2 = expert_off[:, :, 1]
    b3 = expert_off[:, :, 2]
    return torch.stack([
        var[:, :, 0],
        b1 ** 2 + var[:, :, 1],
        b2 ** 2 + b3 ** 2 + var[:, :, 2],
    ], dim=2)


def chol_apply(L: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """``L eps``: ``L`` is ``(B,K,3,3,*spatial)``, ``eps`` is ``(B,K,3,*spatial)``."""
    B, K = L.shape[:2]
    spatial = tuple(L.shape[4:])
    Lf = L.reshape(B * K, 3, 3, -1)
    ef = eps.reshape(B * K, 3, -1)
    out = torch.einsum("bijs,bjs->bis", Lf, ef)
    return out.reshape(B, K, 3, *spatial)
