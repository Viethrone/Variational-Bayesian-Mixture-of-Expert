"""Deterministic reflect padding.

Built from ``narrow`` / ``flip`` / ``cat`` instead of ``F.pad(mode="reflect")``: the numerics
are identical, but the backward pass stays deterministic on CUDA, which is what makes a
same-seed rerun bit-wise reproducible.
"""

from __future__ import annotations

import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    # local import: _encoder2d imports these kernels, so a module-level import would cycle
    from src.models.vbmile._encoder2d import _AdaptiveConv2d

    n = 0
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and m.padding_mode == "reflect":
            if isinstance(m, _AdaptiveConv2d):
                m.manual_reflect = True
            else:
                m.forward = types.MethodType(_manual_reflect_forward, m)
            n += 1
    return n
