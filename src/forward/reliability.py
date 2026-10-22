"""Per-angle reliability weighting of the seismic likelihood (opt-in, off by default).

Why this exists
---------------
The seismic term is a *weak physics constraint*: the forward operator uses the
Aki-Richards approximation (``src/forward/seismic.py``), which degrades systematically near
and beyond the critical angle (the higher-order angle dependence of the reflection
coefficient, transmission loss and far-angle energy decay are absent from the two-term
expansion). The residual is then not random noise with zero mean but a systematic error of
fixed sign; an iid Gaussian likelihood fits it as information and pushes the network away
from the true model - this is the mechanism behind "a weak physics constraint doing harm".

This module makes that failure mode **detectable and maskable**: for every
(angle, depth sample) the analytic critical angle ``theta_c`` is computed and the NLL terms
with ``theta >= theta_c`` are scaled down to ``floor`` (default 0). Their gradient is
therefore zero, while the reliable region (``theta << theta_c``) keeps every bit of its
gradient.

Analytic critical angle (aligned with the forward operator)
----------------------------------------------------------
``SeismicForward`` builds the reflection coefficient from **adjacent depth samples**: at
sample j the incident medium is j-1 and the transmitted medium is j (``vp1 = vp[..., :-1]``,
``vp2 = vp[..., 1:]``). For a downgoing P wave::

    theta_c = arcsin(vp1 / vp2)   (vp2 > vp1, a critical angle exists)
    theta_c = 90 deg              (vp2 <= vp1, no critical angle)

In code ``ratio = clamp(vp1 / vp2, 0, 1)`` covers both cases at once
(``vp2 <= vp1`` gives ``ratio >= 1``, clamped to 1, and ``arcsin(1) = 90 deg``).

Two spatial scales (both supported; the adjacent-sample view is the primary evidence)
-------------------------------------------------------------------------------------
* ``layer_sigma=0`` (default): **adjacent-sample view**, ``theta_c`` per sample, exactly
  consistent with the differential stencil of the forward operator.
* ``layer_sigma>0``: **layer-scale view** (optional), smoothing ``vp`` along depth with a
  Gaussian of ``sigma`` samples before ``theta_c`` is evaluated. The physical reading is
  "the velocity contrast of the layer package, not of two adjacent samples, decides from
  which angle the linearisation stops being trustworthy".

Measured on Marmousi2 (2026-10-22 critical-angle census, Vp 1028-4700 m/s): the most
critical interface has ``theta_c = 32.8 deg``, and the fraction of interfaces with
``theta_c <= 30 deg`` is **0.00%** at the adjacent-sample scale and at block scales
5/10/20/50/100. Consequently a hard ``theta >= theta_c`` mask is a **strict no-op** on
5-30 deg data - which is precisely the claim the paper reports. Only the taper band
(default 5 deg, starting at ``theta_c - 5``) touches <= 0.04% of the pixels (30 -> 32.8 deg
leaves a 2.8 deg margin), and smoothing (``layer_sigma>0``) only **raises** ``theta_c``
(it removes sample-scale spurious contrasts), never fabricates masks. Hence the default 0
and the explicit note that this view is not part of the main evidence.

Weight curve (``theta`` is the incidence angle of that trace)
-------------------------------------------------------------
    theta <= theta_c - taper          -> 1
    theta_c - taper < theta < theta_c -> cosine decay to floor (C1 continuous, so the mask
                                         boundary does not introduce its own gradient kink)
    theta >= theta_c                  -> floor (default 0)

Angle-axis convention: **the weight has the shape of the likelihood tensor**, i.e.
``(B, n_angles, *spatial)`` (``(B, A, L)`` in 1-D and ``(B, A, H, W)`` in 2-D).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Union

import torch
import torch.nn.functional as F

__all__ = [
    "smooth_depth",
    "critical_angle_deg",
    "critical_angle_weights",
    "masked_mean",
    "mask_stats",
]

AngleLike = Union[Sequence[float], torch.Tensor]

_EPS = 1e-8


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _gaussian_kernel1d(sigma: float, *, device, dtype) -> torch.Tensor:
    """Normalised 1-D Gaussian kernel truncated at +-4 sigma (length ``2*ceil(4 sigma)+1``)."""
    radius = max(1, int(math.ceil(4.0 * float(sigma))))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    return k / k.sum()


def _as_angles(angles: AngleLike, like: torch.Tensor) -> torch.Tensor:
    """Angles in degrees -> 1-D tensor on ``like``'s device/dtype."""
    if torch.is_tensor(angles):
        return angles.to(device=like.device, dtype=like.dtype).reshape(-1)
    return torch.as_tensor([float(a) for a in angles], dtype=like.dtype, device=like.device)


def smooth_depth(x: torch.Tensor, sigma: float, *, depth_dim: int = -1) -> torch.Tensor:
    """Gaussian smoothing along ``depth_dim``; ``sigma <= 0`` returns ``x`` unchanged (bit-wise).

    Args:
        x: tensor of any shape whose ``depth_dim`` axis is depth.
        sigma: standard deviation **in samples** (a physical scale, independent of the
            sampling interval).
        depth_dim: depth-axis index (``-1`` in 1-D; ``-2`` for the ``(B, H, W)`` layout of
            2-D data).

    Note:
        Boundaries use reflect padding (replicate when the axis is too short), so the two
        ends of the profile are not flattened into a constant, which would lower the
        ``theta_c`` of the shallow/deep samples.
    """
    if sigma is None or float(sigma) <= 0.0:
        return x
    d = x.dim()
    dd = depth_dim % d
    n = x.shape[dd]
    if n < 2:
        return x
    x_m = x.movedim(dd, -1)
    flat = x_m.reshape(-1, 1, n)
    k = _gaussian_kernel1d(sigma, device=x.device, dtype=x.dtype).view(1, 1, -1)
    r = k.shape[-1] // 2
    mode = "reflect" if n > r else "replicate"
    y = F.conv1d(F.pad(flat, (r, r), mode=mode), k)
    return y.reshape(x_m.shape).movedim(-1, dd)


def critical_angle_deg(vp: torch.Tensor, *, depth_dim: int = -1,
                       layer_sigma: float = 0.0) -> torch.Tensor:
    """Analytic critical angle (degrees) of every interface; same shape as ``vp`` but one
    sample shorter along the depth axis.

    Output sample j refers to the interface between samples j-1 and j, matching the
    reflection-coefficient stencil of ``SeismicForward``.

    Args:
        vp: P-wave velocity with depth along ``depth_dim`` (any unit, positive and
            dimensionally consistent).
        depth_dim: depth-axis index.
        layer_sigma: ``> 0`` smooths ``vp`` along depth first (layer-scale view);
            ``0`` keeps the adjacent-sample view.
    """
    if vp.dim() < 2:
        raise ValueError(
            f"critical_angle_deg needs at least 2 dims (batch + depth), got {tuple(vp.shape)}")
    d = vp.dim()
    dd = depth_dim % d
    n = vp.shape[dd]
    if n < 2:
        raise ValueError(f"the depth axis needs at least 2 samples, got {n}")
    vp_s = smooth_depth(vp, layer_sigma, depth_dim=dd)
    vp1 = vp_s.narrow(dd, 0, n - 1)          # overburden (incident side)
    vp2 = vp_s.narrow(dd, 1, n - 1)          # underburden (transmitted side)
    # vp2 <= vp1 => ratio >= 1 => clamped to 1 => arcsin(1) = 90 deg (no critical angle)
    ratio = torch.clamp(vp1 / vp2.clamp_min(_EPS), min=0.0, max=1.0)
    return torch.rad2deg(torch.asin(ratio))


def critical_angle_weights(vp: torch.Tensor, angles_deg: AngleLike, *,
                           layer_sigma: float = 0.0, taper_deg: float = 5.0,
                           floor: float = 0.0, depth_dim: int = -1) -> torch.Tensor:
    """Per-angle reliability weights with shape ``(B, n_angles, *spatial)`` (the shape of the
    likelihood tensor).

    Args:
        vp: P-wave velocity ``(B, *spatial)`` with depth along ``depth_dim``.
        angles_deg: incidence angle of every trace (degrees); its length must equal the
            angle-axis length of the likelihood.
        layer_sigma: depth smoothing scale in samples; ``0`` = adjacent-sample view
            (exactly consistent with the forward stencil).
        taper_deg: width of the cosine transition band in degrees; ``0`` gives a hard step
            (weight 1 below ``theta_c``, ``floor`` above).
        floor: smallest weight kept beyond the critical region, in ``[0, 1]``; the default
            0 masks the gradient completely.
        depth_dim: depth-axis index (``-1`` in 1-D; ``-2`` for ``(B, H, W)`` in 2-D).

    Returns:
        Weight tensor with ``vp``'s dtype/device and shape ``(B, A, *spatial)``. The first
        depth sample reuses the weight of the first interface, because the forward
        operator zeroes its reflection coefficient (it carries only the wavelet tail, so
        there is no independent interface to judge).

    Note:
        The result is **not** detached (smoothing itself is differentiable). The caller
        should detach it when using it as a mask, so that "whether to mask" is decided by
        the forward physical quantities and not by the backward pass.
    """
    floor = float(floor)
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"floor must be in [0, 1], got {floor!r}")
    taper = float(taper_deg)
    if taper < 0.0:
        raise ValueError(f"taper_deg must be >= 0, got {taper!r}")

    theta_c = critical_angle_deg(vp, depth_dim=depth_dim, layer_sigma=layer_sigma)  # (..., n-1)
    ang = _as_angles(angles_deg, vp)                                               # (A,)
    th = ang.view(*([1] * theta_c.dim()), -1)                                      # (..., 1, A)
    tc = theta_c.unsqueeze(-1)                                                     # (..., n-1, 1)
    if taper > 0.0:
        # x=0 at theta = theta_c - taper (weight 1), x=1 at theta = theta_c (weight floor);
        # outside the band x is clamped and the weight is constant.
        x = ((th - (tc - taper)) / taper).clamp(0.0, 1.0)
        w = floor + (1.0 - floor) * 0.5 * (1.0 + torch.cos(math.pi * x))
    else:
        ones = torch.full_like(th, 1.0)
        zeros = torch.full_like(th, floor)
        w = torch.where(th < tc, ones, zeros)

    d = theta_c.dim()
    dd = depth_dim % d
    # pad the depth axis back by one sample (the difference operator consumed one);
    # the first interface's weight is repeated.
    w = torch.cat([w.narrow(dd, 0, 1), w], dim=dd)
    return w.movedim(-1, 1)


# ---------------------------------------------------------------------------
# reduction and statistics (used by the loss; the convention is fixed here so the
# reported definition stays identical to the paper)
# ---------------------------------------------------------------------------
def masked_mean(nll: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weight-normalised mean ``sum(w*nll) / sum(w)`` (the "mean over the reliable region").

    Relation to ``nll.mean()``: with all weights equal to 1 the two expressions are
    mathematically identical (the reduction tree may differ by the last 1-2 ulp, which is
    why the default path never takes this route and masking is strictly opt-in).
    Dividing by ``sum(w)`` rather than by the element count keeps the **gradient strength
    of the reliable region unchanged** by masking: masking should only delete information,
    not weaken the remaining constraints.
    """
    if nll.shape != weight.shape:
        raise ValueError(
            f"masked_mean: NLL shape {tuple(nll.shape)} does not match weight shape "
            f"{tuple(weight.shape)} - multiplying would broadcast silently and corrupt the loss.")
    w = weight.to(nll.dtype)
    return (nll * w).sum() / w.sum().clamp_min(_EPS)


def mask_stats(weight: torch.Tensor, residual: torch.Tensor | None = None):
    """Mask occupancy statistics (logged into ``loss_dict`` to show that the mechanism is a
    no-op on 5-30 deg data).

    Args:
        weight: weights shaped like the likelihood ``(B, A, ...)``.
        residual: optional ``(syn - obs)`` with the same shape as ``weight``.

    Returns:
        ``(occupancy, occupancy_energy)``:

        * ``occupancy = 1 - mean(weight)`` - the **weight-gap** convention (mean weight).
          With 0/1 weights this is the fraction of fully masked elements; with a taper or
          ``floor > 0`` it is the average amount of weight removed. Both live in ``[0, 1]``.
        * ``occupancy_energy`` = fraction of the **residual energy** (``sum r^2``) that sits
          where the weight was driven to 0. ``None`` when ``residual`` is not given. It
          answers "of the gradient that was blocked, how much was really large residual
          (i.e. exactly the part where the forward operator is seriously wrong)".

    Note:
        Both quantities are computed on ``detach()``-ed weights and do not take part in
        backpropagation.
    """
    w = weight.detach()
    occupancy = 1.0 - w.mean()
    if residual is None:
        return occupancy, None
    if residual.shape != w.shape:
        raise ValueError(
            f"mask_stats: residual shape {tuple(residual.shape)} does not match weight "
            f"shape {tuple(w.shape)}")
    energy = residual.detach() ** 2
    zero = (w <= 0).to(energy.dtype)
    occupancy_energy = (energy * zero).sum() / energy.sum().clamp_min(_EPS)
    return occupancy, occupancy_energy
