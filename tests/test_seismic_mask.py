"""Per-angle reliability masking of the seismic likelihood (analytic critical angle).

Background (reviewer question): the seismic term is a *weak physics constraint* whose
forward operator (Aki-Richards) is systematically wrong near and beyond the critical angle.
That directed error is what an iid Gaussian likelihood mistakes for information, pushing the
network off the true model. This module provides a default-off per-angle weight:
``theta <= theta_c - taper`` -> 1, cosine taper down across the band, ``theta >= theta_c``
-> ``floor`` (default 0); the gradient of the masked angles is exactly zero and the mask
occupancy is reported in ``loss_dict``.

Covered here:
    * (a) with the switch off the numerical path is **bit-wise** identical to the released
      implementation (the ``loss_dict`` key set included);
    * (b) the weight curve: a constructed interface with ``theta_c = 30 deg`` gives weight 0
      at 30 deg, 1 at 5 deg, and a strictly monotone taper in between;
    * (c) 1-D / 2-D shapes match the likelihood tensor dimension by dimension;
    * (d) integration smoke: assemble criterion + trainer from
      ``configs/inversion/vbmile_1d.yaml``, run one forward/backward batch, check that the
      loss is finite and that the occupancy stays in ``[0, 1]``;
    * (e) masked angles receive **exactly zero** gradient (that is what "masking" means);
    * (f) with all weights equal to 1 the loss equals the unmasked value (the mechanism is a
      no-op on 5-30 deg data).

Conventions (fixed here, do not re-word them in reports):
    ``seismic_mask_occupancy = 1 - mean(w)`` with the denominator equal to the number of
    ``(B, n_angles, *spatial)`` weight elements (not only the elements driven to 0). It is
    exactly 0 when every weight is 1 (a no-op).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

# registration side effects (models / losses / forward / trainers are resolved from the
# `_target_` entries of the configs)
import src.forward.seismic  # noqa: E402,F401
import src.losses  # noqa: E402,F401
import src.models  # noqa: E402,F401
from experiments.common import project_root
from src.forward.base import FORWARD_OPERATORS  # noqa: E402
from src.forward.reliability import (
    critical_angle_deg,
    critical_angle_weights,
    mask_stats,
    smooth_depth,
)
from src.losses import VBELBOCore
from src.losses.base import LOSSES  # noqa: E402
from src.models.base import MODELS  # noqa: E402
from src.trainers.base import TrainerConfig
from src.trainers.vbmile_trainer import VBEMLETrainer
from src.utils.config import load_config
from src.utils.registry import build_from_config  # noqa: E402

CFG_DIR = project_root() / "configs"
ANGLES = (5, 10, 15, 20, 25, 30)
NA = len(ANGLES)
L = 32
H, W = 12, 7

LEGACY_KEYS = {"total", "nll_well", "seismic_nll", "kl_z", "kl_theta",
               "well_var", "seismic_var", "mse_monitor"}


# ---------------------------------------------------------------------------
# small constructors
# ---------------------------------------------------------------------------
def _crit(**kw) -> VBELBOCore:
    kwargs = dict(seismic_obs_noise_init=1.0, well_obs_noise_init=0.1,
                  kl_z_weight=0.05, kl_theta_weight=0.01, seismic_weight=0.37)
    kwargs.update(kw)
    return VBELBOCore(**kwargs)


def _out_1d(b=2):
    torch.manual_seed(0)
    return {
        "final_pred": torch.randn(b, 6, L),
        "kl_z": torch.tensor(0.123),
        "kl_theta": torch.tensor(0.045),
        "theta_sample": torch.randn(b, 3, L),
        "z_soft": torch.softmax(torch.randn(b, 3, L), dim=1),
        "expert_mus": torch.randn(b, 3, 3, L),
        "expert_logvars": torch.randn(b, 3, 3, L) * 0.1,
    }


def _pair(shape, seeds=(21, 22, 23)):
    g1 = torch.Generator().manual_seed(seeds[0])
    g2 = torch.Generator().manual_seed(seeds[1])
    g3 = torch.Generator().manual_seed(seeds[2])
    return (torch.randn(*shape, generator=g1), torch.randn(*shape, generator=g2),
            torch.randn(*shape, generator=g3))


def _target(b=2):
    return torch.randn(b, 3, L, generator=torch.Generator().manual_seed(29))


# ---------------------------------------------------------------------------
# (a) switch off => bit-wise identical to the released implementation
# ---------------------------------------------------------------------------
def test_default_mode_is_none_and_adds_no_parameters():
    crit = _crit()
    assert crit.seismic_mask_mode == "none"
    assert crit.seismic_mask_layer_sigma == 0.0        # default = adjacent-sample view
    assert crit.seismic_mask_taper_deg == 5.0
    assert crit.seismic_mask_floor == 0.0
    # plain attributes only => the state_dict key set matches legacy (nothing in ckpt)
    assert not any("seismic_mask" in k for k in crit.state_dict())


def _legacy_nll_seismic(crit, syn, obs):
    """Body of ``_nll_seismic`` before this change (its last line was ``return nll.mean()``)."""
    diff = (syn - obs).clamp(-crit.clip_diff, crit.clip_diff)
    if crit.seismic_noise != "iid":
        nll = crit._nll_seis_corr(diff)
    else:
        nll = 0.5 * (diff**2 / crit.seismic_var + crit.log_two_pi
                     + torch.log(crit.seismic_var))
    nll = torch.nan_to_num(nll, nan=0.0, posinf=1e6, neginf=-1e6)
    return nll.mean()


@pytest.mark.parametrize("noise", ["iid", "coherent", "deviatoric"])
def test_nll_seismic_none_weight_is_bitwise_legacy(noise: str):
    """``mask_weight=None`` takes the original branch, bit-wise equal to before."""
    crit = _crit(seismic_noise=noise, seismic_coh_noise_init=1.1,
                 seismic_white_noise_init=0.9)
    crit.loss_dim = 2
    syn, obs, _ = _pair((2, NA, H, W))
    want = _legacy_nll_seismic(crit, syn, obs)
    got_default = crit._nll_seismic(syn, obs)
    got_explicit_none = crit._nll_seismic(syn, obs, None)
    assert torch.equal(got_default, want), (float(got_default), float(want))
    assert torch.equal(got_explicit_none, want)


def test_forward_without_weight_is_bitwise_legacy_and_keyset_unchanged():
    """Whole forward: ``seismic_mask_weight=None`` is bit-wise identical, key set unchanged."""
    crit = _crit()
    mo = _out_1d()
    syn, obs, _ = _pair((2, NA, L))
    target = _target()
    mask = torch.ones(2, 3, L)
    loss_a, ld_a = crit(mo, target=target, syn_norm=syn, seismic=obs, mask=mask)
    loss_b, ld_b = crit(mo, target=target, syn_norm=syn, seismic=obs, mask=mask,
                        seismic_mask_weight=None)
    assert torch.equal(loss_a, loss_b)
    assert set(ld_a) == LEGACY_KEYS and set(ld_b) == LEGACY_KEYS
    for k in ld_a:
        assert torch.equal(ld_a[k], ld_b[k]), k
    # summation order matches the implementation:
    # total = nll_well + seismic_nll + w_z * kl_z + w_theta * kl_theta
    want = (ld_a["nll_well"] + ld_a["seismic_nll"]
            + crit.kl_z_weight * ld_a["kl_z"] + crit.kl_theta_weight * ld_a["kl_theta"])
    assert torch.equal(ld_a["total"], want)
    assert torch.equal(loss_a, ld_a["total"])


# ---------------------------------------------------------------------------
# (b) weight curve: hard mask + monotone taper + floor
# ---------------------------------------------------------------------------
def test_theta_c_formula_and_90deg_when_no_contrast():
    # vp2 = 2 * vp1 => theta_c = arcsin(1/2) = 30 deg
    assert float(critical_angle_deg(torch.tensor([[3.0, 6.0]]))[0, 0]) == pytest.approx(30.0, abs=1e-4)
    # vp2 <= vp1 => no critical angle => 90 deg
    assert float(critical_angle_deg(torch.tensor([[4.0, 3.0]]))[0, 0]) == pytest.approx(90.0, abs=1e-5)
    assert float(critical_angle_deg(torch.tensor([[4.0, 4.0]]))[0, 0]) == pytest.approx(90.0, abs=1e-5)


def test_weight_curve_hard_mask_taper_and_floor():
    vp = torch.tensor([[3.0, 6.0]])            # single interface: theta_c = 30 deg
    w = critical_angle_weights(vp, ANGLES, taper_deg=5.0)[0, :, 1]   # weights of that interface
    assert float(w[0]) == 1.0                  # 5 deg << theta_c
    assert float(w[-1]) == 0.0                 # 30 deg >= theta_c => fully masked
    assert float(critical_angle_weights(vp, [25.0], taper_deg=5.0)[0, 0, 1]) == pytest.approx(1.0)
    # strictly decreasing inside the taper band (theta_c - taper = 25 deg, theta_c = 30 deg)
    fine = [26.0, 27.0, 28.0, 29.0]
    wf = critical_angle_weights(vp, fine, taper_deg=5.0)[0, :, 1]
    assert torch.all(wf.diff() < 0), wf.tolist()
    assert torch.all((wf > 0) & (wf < 1)), wf.tolist()
    # floor is configurable: keep `floor` beyond the critical angle
    wf2 = critical_angle_weights(vp, ANGLES, taper_deg=5.0, floor=0.2)[0, :, 1]
    assert float(wf2[-1]) == pytest.approx(0.2)
    assert torch.all(wf2 >= 0.2 - 1e-6)
    # taper=0 => hard step (float32 arcsin lands 2e-6 deg off 30 deg, so judge at 30.5 deg)
    hard = critical_angle_weights(vp, [20.0, 30.5], taper_deg=0.0)[0, :, 1]
    assert float(hard[0]) == 1.0 and float(hard[-1]) == 0.0
    # a section with no velocity increase (vp decreasing) => all ones (a no-op)
    flat = critical_angle_weights(torch.tensor([[6.0, 5.0, 4.0, 3.0]]), ANGLES)
    assert torch.equal(flat, torch.ones_like(flat))


# ---------------------------------------------------------------------------
# (c) shapes: aligned with the likelihood tensor (1-D / 2-D)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape,depth_dim", [((2, L), -1), ((2, H, W), -2)])
def test_weight_shape_matches_likelihood(shape, depth_dim):
    torch.manual_seed(5)
    vp = 2.0 + 3.0 * torch.rand(*shape)
    w = critical_angle_weights(vp, ANGLES, depth_dim=depth_dim)
    assert tuple(w.shape) == (shape[0], NA, *shape[1:])
    assert torch.isfinite(w).all()
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0


def test_layer_sigma_is_optional_and_smoothing_raises_theta_c():
    """``layer_sigma=0`` is the adjacent-sample view; > 0 is the layer-scale view.

    Constructed case: sample-scale **spurious contrast** (2/5 km/s alternating). At the
    adjacent-sample scale theta_c = 23.6 deg, so the 30 deg channel is masked; layer-scale
    smoothing (sigma = 4 samples) averages the alternation away, theta_c returns to 90 deg
    and the mask disappears. So the two scales do differ, and smoothing can only raise
    theta_c - never fabricate masks.
    """
    torch.manual_seed(7)
    vp = torch.where(torch.arange(L) % 2 == 0, torch.tensor(2.0), torch.tensor(5.0))
    vp = vp.reshape(1, L)
    w_adj = critical_angle_weights(vp, ANGLES, layer_sigma=0.0)
    occ_adj, _ = mask_stats(w_adj)
    assert float(occ_adj) > 0.0
    w_layer = critical_angle_weights(vp, ANGLES, layer_sigma=4.0)
    occ_layer, _ = mask_stats(w_layer)
    assert float(occ_layer) == pytest.approx(0.0, abs=1e-6)
    # smooth_depth(sigma=0) returns the input unchanged (bit-wise); > 0 keeps the shape and
    # shrinks the variance
    assert torch.equal(smooth_depth(vp, 0.0), vp)
    sm = smooth_depth(vp, 4.0)
    assert tuple(sm.shape) == tuple(vp.shape)
    assert float(sm.std()) < float(vp.std())


# ---------------------------------------------------------------------------
# (e) the substance of masking: exactly zero gradient on masked angles
# ---------------------------------------------------------------------------
def test_masked_angle_receives_exactly_zero_gradient():
    crit = _crit()
    torch.manual_seed(11)
    syn = torch.randn(1, NA, L, requires_grad=True)
    obs = torch.randn(1, NA, L)
    w = torch.ones(1, NA, L)
    w[:, -1] = 0.0                       # mask only the 30 deg trace
    crit._nll_seismic(syn, obs, w).backward()
    assert float(syn.grad[:, -1].abs().sum()) == 0.0     # exactly 0, not "small"
    assert float(syn.grad[:, 0].abs().sum()) > 0.0


def test_all_ones_weight_is_numerically_a_noop():
    """On 5-30 deg data "theta >= theta_c hard mask" is a no-op: unit weights keep the loss."""
    crit = _crit()
    torch.manual_seed(13)
    syn, obs = torch.randn(2, NA, L), torch.randn(2, NA, L)
    base = crit._nll_seismic(syn, obs)
    ones = torch.ones_like(syn)
    got = crit._nll_seismic(syn, obs, ones)
    assert torch.allclose(got, base, rtol=1e-6, atol=1e-8), (float(got), float(base))
    occ, occ_energy = mask_stats(ones, syn - obs)
    assert float(occ) == 0.0
    assert float(occ_energy) == 0.0


def test_mask_stats_denominator_is_all_weight_elements():
    """occupancy = 1 - mean(w), denominator = all elements; energy is weighted by sum r^2."""
    w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    r = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    occ, occ_e = mask_stats(w, r)
    assert float(occ) == pytest.approx(0.75)            # 3 of 4 elements driven to 0
    assert float(occ_e) == pytest.approx(0.8)           # 4/5 of the energy sits in the mask


# ---------------------------------------------------------------------------
# (d) loss side: occupancy is reported only when a weight is passed
# ---------------------------------------------------------------------------
def test_forward_reports_occupancy_only_when_weight_given():
    crit = _crit(seismic_mask_mode="critical_angle")
    mo = _out_1d()
    syn, obs, _ = _pair((2, NA, L))
    target = _target()
    mask = torch.ones(2, 3, L)
    w = torch.ones_like(syn)
    w[:, -1] = 0.0                                      # mask 1/6 of the angle traces
    loss, ld = crit(mo, target=target, syn_norm=syn, seismic=obs, mask=mask,
                    seismic_mask_weight=w)
    assert torch.isfinite(loss)
    assert "seismic_mask_occupancy" in ld
    assert "seismic_mask_occupancy_energy" in ld
    assert float(ld["seismic_mask_occupancy"]) == pytest.approx(1.0 / NA)
    assert 0.0 <= float(ld["seismic_mask_occupancy"]) <= 1.0
    assert 0.0 <= float(ld["seismic_mask_occupancy_energy"]) <= 1.0


def test_mode_enabled_without_weight_raises_instead_of_silently_disabling():
    crit = _crit(seismic_mask_mode="critical_angle")
    mo = _out_1d()
    syn, obs, _ = _pair((2, NA, L))
    with pytest.raises(ValueError, match="seismic_mask_weight"):
        crit(mo, target=_target(), syn_norm=syn, seismic=obs, mask=torch.ones(2, 3, L))


def test_bad_mask_config_is_rejected():
    with pytest.raises(ValueError, match="seismic_mask_mode"):
        VBELBOCore(seismic_mask_mode="layer")
    with pytest.raises(ValueError, match="seismic_mask_layer_sigma"):
        VBELBOCore(seismic_mask_layer_sigma=-1.0)
    with pytest.raises(ValueError, match="seismic_mask_floor"):
        VBELBOCore(seismic_mask_floor=1.5)


# ---------------------------------------------------------------------------
# (d) integration smoke: configs/inversion/vbmile_1d.yaml assembles criterion + trainer
# ---------------------------------------------------------------------------
def _run_info():
    """The trainer only reads ``run_info`` fields and writes nothing for a single batch."""
    return SimpleNamespace(reconstructed_dir=None, checkpoints_dir=None,
                           write_metrics=lambda m: None, path=lambda *p: None)


def _smoke_batch(b=2):
    g = torch.Generator().manual_seed(41)
    return {"seismic": torch.randn(b, NA, 64, generator=g),
            "target": torch.randn(b, 3, 64, generator=g),
            "target_low": torch.randn(b, 3, 64, generator=g),
            "mask": torch.ones(b, 3, 64), "prior": None}


def _smoke_stats():
    # physical magnitudes (Vp in km/s): the critical angle needs real velocities
    return {"target_mean": torch.tensor([4.0, 2.1, 2.5]),
            "target_std": torch.tensor([0.6, 0.35, 0.15]),
            "seismic_mean": torch.zeros(NA), "seismic_std": torch.ones(NA)}


def _build_smoke(loss_overrides=None):
    cfg = load_config(CFG_DIR / "inversion/vbmile_1d.yaml",
                      overrides=["model.base_channels=4", "model.num_experts=2"])
    mcfg = dict(cfg["model"])
    target = mcfg.pop("_target_")
    mcfg.pop("prior_means", None)
    mcfg.pop("prior_vars", None)
    k = int(mcfg.get("num_experts", 3))
    torch.manual_seed(0)
    model = build_from_config(
        MODELS, {"_target_": target, **mcfg},
        extra_kwargs={"prior_means": np.zeros((k, 3), np.float32),
                      "prior_vars": np.ones((k, 3), np.float32)})
    forward = build_from_config(FORWARD_OPERATORS, cfg["forward"])
    criterion = build_from_config(LOSSES, {**cfg["loss"], **(loss_overrides or {})})
    trainer = VBEMLETrainer(
        _run_info(), TrainerConfig(max_epochs=1, extra={"lr": 1e-3, "grad_clip": 1.0}),
        model=model, criterion=criterion, forward=forward,
        train_loader=[_smoke_batch()], val_loader=[_smoke_batch()],
        stats=_smoke_stats(), device="cpu")
    return trainer


def test_smoke_one_batch_with_mask_off_keeps_legacy_keys():
    """Smoke 0: switch off => one forward/backward batch keeps the legacy component keys."""
    tr = _build_smoke()
    opt = torch.optim.SGD(tr.model.parameters(), lr=1e-3)
    loss, comps = tr._train_one_epoch(tr.model, tr.criterion, opt, 0, tr.stats)
    assert np.isfinite(loss)
    assert set(comps) == LEGACY_KEYS
    assert not any("mask" in k for k in comps)


def test_smoke_one_batch_with_critical_angle_mask():
    """Smoke 1: critical-angle masking on (adjacent-sample view, default taper)."""
    tr = _build_smoke({"seismic_mask_mode": "critical_angle"})
    opt = torch.optim.SGD(tr.model.parameters(), lr=1e-3)
    loss, comps = tr._train_one_epoch(tr.model, tr.criterion, opt, 0, tr.stats)
    assert np.isfinite(loss)
    assert "seismic_mask_occupancy" in comps
    assert "seismic_mask_occupancy_energy" in comps
    occ = float(comps["seismic_mask_occupancy"])
    assert 0.0 <= occ <= 1.0
    # the default parameters (theta_c far above 30 deg) make this close to a no-op; the
    # taper edge can still remove a sliver, so do not assert exactly 0
    assert occ < 0.05


def test_smoke_one_batch_with_forced_nonzero_occupancy():
    """Smoke 2: push the taper to 180 deg (**test only**, not a recommended setting) so the
    weight really reaches the loss - this proves the occupancy is not hard-coded to 0."""
    tr = _build_smoke({"seismic_mask_mode": "critical_angle",
                       "seismic_mask_taper_deg": 180.0})
    opt = torch.optim.SGD(tr.model.parameters(), lr=1e-3)
    loss, comps = tr._train_one_epoch(tr.model, tr.criterion, opt, 0, tr.stats)
    assert np.isfinite(loss)
    occ = float(comps["seismic_mask_occupancy"])
    assert 0.0 < occ < 1.0, occ
    assert 0.0 <= float(comps["seismic_mask_occupancy_energy"]) <= 1.0


def test_trainer_mask_weight_shape_1d_and_2d():
    """The trainer computes weights from physical theta: 1-D (N,A,L) / 2-D (N,A,H,W)."""
    tr = _build_smoke({"seismic_mask_mode": "critical_angle"})
    torch.manual_seed(0)
    w1 = tr._seismic_mask_weight(torch.randn(4, 3, L), tr.stats, tr.criterion)
    assert tuple(w1.shape) == (4, NA, L)
    w2 = tr._seismic_mask_weight(torch.randn(4, 3, H, W), tr.stats, tr.criterion)
    assert tuple(w2.shape) == (4, NA, H, W)
    assert torch.isfinite(w1).all() and torch.isfinite(w2).all()
    # switch off => None (nothing extra is computed)
    tr2 = _build_smoke()
    assert tr2._seismic_mask_weight(torch.randn(4, 3, L), tr2.stats, tr2.criterion) is None
