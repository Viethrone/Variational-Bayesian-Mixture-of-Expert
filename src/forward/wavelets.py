"""Seismic wavelets (Ricker / Ormsby), single torch source.

The legacy code used ``bruges.filters.wavelets``; this module re-implements the same
mathematics in numpy + torch and guarantees the repository-wide convention of an odd-length
kernel symmetric about 0 (see ``ricker_wavelet``). Both wavelets return a ``(1, 1, L)``
torch.float32 tensor, ready to be used as the ``wavelet`` entry of ``SeismicForward``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

__all__ = ["ricker_wavelet", "ormsby_wavelet"]


def _time_axis(dt: float, length: float, n: int) -> np.ndarray:
    """Odd-length time axis symmetric about 0 (seconds): -a .. +a with step dt."""
    return np.linspace(-(n - 1) * dt / 2, (n - 1) * dt / 2, n, dtype=np.float32)


def _odd_len(length: float, dt: float) -> int:
    n = int(round(length / dt))
    return n + 1 if n % 2 == 0 else n


def ricker_wavelet(dt: float = 0.002, f: float = 30.0, length: float = 0.1) -> torch.Tensor:
    """Ricker (Mexican hat) wavelet, shape (1,1,L).

    w(t) = (1 - 2(π f t)²) exp(-(π f t)²)

    Notes:
        * the legacy code used `np.arange(-l/2, l/2+dt, dt)`, whose parity drifts and yields
            asymmetric or even-length kernels, shifting the convolved length by +-1. Here the
            kernel is always odd and symmetric (101 pts <-> 0.202 s @2 ms, 51 pts <-> 0.1 s @2 ms).
    """
    n = _odd_len(length, dt)
    t = _time_axis(dt, length, n)
    w = (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)
    return torch.tensor(w, dtype=torch.float32).view(1, 1, -1)


def ormsby_wavelet(dt: float = 0.001, f_hz: Sequence[float] = (5, 10, 60, 80),
                   length: float = 0.1) -> torch.Tensor:
    """Ormsby wavelet (rewrite of bruges.filters.wavelets.ormsby), shape (1,1,L).

    Requires f1 < f2 <= f3 < f4 (trapezoidal band). It is a sum of sinc terms:

        w = (n·sinc²(π f4 t) - m·sinc²(π f3 t) - p·sinc²(π f2 t) + q·sinc²(π f1 t))
            · (f4 - f3) · π / 2

    sinc(x) = sin(x)/x with sinc(0)=1, same convention as ``ricker_wavelet``.
    """
    f1, f2, f3, f4 = (float(x) for x in f_hz)
    if f1 < 0 or f2 <= f1 or f3 <= f2 or f4 <= f3:
        raise ValueError(f"frequencies must satisfy f1 < f2 <= f3 < f4, got {f_hz}")
    n = _odd_len(length, dt)
    t = _time_axis(dt, length, n)
    tf = lambda x: np.sinc(x)  # noqa: E731  np.sinc(x)=sin(πx)/(πx)

    def sinc_sq(freq: float) -> np.ndarray:
        return tf(freq * t) ** 2

    coefs = (f4 / (f4 - f3), -f3 / (f4 - f3), -f2 / (f2 - f1), f1 / (f2 - f1))
    w = (coefs[0] * sinc_sq(f4) + coefs[1] * sinc_sq(f3)
         + coefs[2] * sinc_sq(f2) + coefs[3] * sinc_sq(f1))
    w = w * (f4 - f3) * np.pi / 2
    return torch.tensor(w, dtype=torch.float32).view(1, 1, -1)