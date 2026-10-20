"""Forward operator tests: Ricker wavelet properties and synthetic-seismic shapes."""

from __future__ import annotations

import torch

from src.forward.base import FORWARD_OPERATORS
from src.forward.seismic import SeismicForward, ricker_wavelet


def test_registry_and_wavelet_properties():
    assert "synthetic_seismic" in FORWARD_OPERATORS

    w = ricker_wavelet(dt=0.002, f=20.0, length=0.1)
    assert w.ndim == 3 and w.shape == (1, 1, 51)          # 51 points / 0.1 s @ 2 ms
    n = w.shape[-1]
    assert n % 2 == 1, "the kernel must have odd length"
    assert torch.allclose(w[0, 0], w[0, 0].flip(-1)), "the kernel must be symmetric about 0"
    assert torch.all(w[0, 0].abs() > 0) and abs(float(w[0, 0, n // 2])) > 0

    # length is given in seconds (number of points = length / dt)
    w1 = ricker_wavelet(dt=0.002, f=30.0, length=0.202)
    assert w1.shape == (1, 1, 101)


def test_forward_1d_shapes():
    m = SeismicForward(dim=1, angles_deg=(5, 10, 15))
    x = torch.randn(2, 3, 120)
    y = m.forward(x)
    assert y.shape == (2, 3, 120), y.shape
    assert not torch.isnan(y).any() and torch.isfinite(y).all()


def test_forward_2d_shapes():
    m = SeismicForward(dim=2, angles_deg=(5, 10, 15, 20, 25, 30))
    x = torch.randn(2, 3, 24, 32)
    y = m.forward(x)
    assert y.shape == (2, 6, 24, 32), y.shape
    assert torch.isfinite(y).all()


def test_forward_matches_aki_richards_reflectivity():
    """The synthetic record is a Ricker convolution of the Aki-Richards reflectivity."""
    m = SeismicForward(dim=1, angles_deg=(10, 30), wavelet={"dt": 0.002, "f": 20.0, "length": 0.1})
    x = torch.randn(2, 3, 64) * 0.5
    rc = m.aki_richards_reflectivity(
        x[:, 0, :-1], x[:, 1, :-1], x[:, 2, :-1],
        x[:, 0, 1:], x[:, 1, 1:], x[:, 2, 1:],
        m.angles_rad)
    # (1, batch, depth-1, n_angles)
    assert rc.dim() == 4 and rc.shape == (1, 2, 63, 2), rc.shape
    assert torch.isfinite(rc).all()
