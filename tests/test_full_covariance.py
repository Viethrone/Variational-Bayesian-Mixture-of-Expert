"""Full-covariance variational posterior: numerics, closed-form KL, well likelihood.

The full family adds three off-diagonal channels per expert (``Sigma = L L^T``). These
tests pin: (1) the extension is backwards compatible (zero off-diagonals == the diagonal
path bit-for-bit), (2) the closed-form KL / trace / diag match an independent torch
reference, (3) the two expert-level well-likelihood modes, and (4) the theory contract
that records which of these was actually used.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from experiments.common import resolve_config
from src.losses import VBELBOCore
from src.models import GeoVBMILE
from src.utils.fullcov import (
    build_chol,
    chol_apply,
    chol_columns,
    chol_diag,
    chol_trace_term,
    qform,
)
from src.utils.theory_contract import build_theory_contract
from src.utils.registry import build_from_config

K = 3
D = 3


@pytest.fixture(autouse=True)
def _torch_seed():
    torch.manual_seed(0)


def _prior(K=K, seed=0):
    """A well-conditioned full-covariance prior (random SPD matrices)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(K, D, D, generator=g)
    cov = A @ A.transpose(-1, -2) + 0.5 * torch.eye(D)
    means = torch.randn(K, D, generator=g)
    return means, cov


def _pair(dim=1, seed=0, **kw):
    common = dict(dim=dim, base_channels=8, num_experts=K)
    common.update(kw)
    torch.manual_seed(seed)
    diag = GeoVBMILE(**common, variational_covariance="diag").eval()
    torch.manual_seed(seed)
    full = GeoVBMILE(**common, variational_covariance="full").eval()
    return diag, full


def _inputs(dim, seed=123):
    g = torch.Generator().manual_seed(seed)
    if dim == 1:
        x = torch.randn(2, 6, 96, generator=g)
        low = torch.randn(2, 3, 96, generator=g)
    else:
        x = torch.randn(1, 6, 32, 15, generator=g)
        low = torch.randn(1, 3, 32, 15, generator=g)
    return x, low


def _forward(model, dim, num_samples=0):
    x, low = _inputs(dim)
    with torch.no_grad():
        return model(x, low_freq_model=low, num_samples=num_samples)


# ---------------------------------------------------------------------------
# T1: the extension is backwards compatible
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [1, 2])
def test_full_with_zero_offdiag_equals_diag(dim):
    """``b = 0`` (zero-initialised off head) reproduces the diagonal path.

    The deterministic outputs are bit-wise equal; the KL differs only by summation order
    (three quadratic forms vs one element-wise product), hence the 1e-6 tolerance.
    """
    diag, full = _pair(dim)
    missing, unexpected = full.load_state_dict(diag.state_dict(), strict=False)
    assert not unexpected
    assert all("off" in k for k in missing), missing

    out_d, out_f = _forward(diag, dim), _forward(full, dim)
    for key in ("final_pred", "expert_mus", "expert_logvars", "z_soft", "logits", "kl_z"):
        assert torch.equal(out_d[key], out_f[key]), key
    assert torch.allclose(out_d["kl_theta_per_expert"], out_f["kl_theta_per_expert"],
                          atol=1e-6)
    # the off-diagonal channels are exactly zero at initialisation
    assert torch.equal(out_f["expert_off"], torch.zeros_like(out_f["expert_off"]))
    # marginal identity: b=0 => diag(Sigma_k) == exp(logvar)
    assert torch.allclose(out_f["expert_diag_var"], torch.exp(out_f["expert_logvars"]),
                          atol=1e-6)


def test_offdiag_head_is_zero_init_and_adds_only_new_keys():
    diag, full = _pair(1)
    off_keys = [k for k in full.state_dict() if "off" in k]
    assert off_keys and not any("off" in k for k in diag.state_dict())
    for key in off_keys:
        assert torch.count_nonzero(full.state_dict()[key]) == 0
    assert set(diag.state_dict()) < set(full.state_dict())


def test_released_diag_layout_is_the_default():
    model = GeoVBMILE(dim=1, base_channels=8, num_experts=K)
    assert model.variational_covariance == "diag"
    assert "variational_covariance" not in model.state_dict()
    with pytest.raises(ValueError):
        GeoVBMILE(dim=1, variational_covariance="banana")


# ---------------------------------------------------------------------------
# closed-form operators vs an independent torch reference
# ---------------------------------------------------------------------------

def test_chol_diag_trace_and_columns_match_reference():
    g = torch.Generator().manual_seed(3)
    logvars = torch.randn(2, K, D, generator=g)
    off = torch.randn(2, K, D, generator=g)
    L = build_chol(logvars, off)                        # (B,K,3,3)
    sigma = torch.exp(0.5 * logvars)
    # rebuild L explicitly by the documented row-major mapping
    L_ref = torch.zeros(2, K, D, D)
    b1, b2, b3 = off[:, :, 0], off[:, :, 1], off[:, :, 2]
    L_ref[:, :, 0, 0] = sigma[:, :, 0]
    L_ref[:, :, 1, 0], L_ref[:, :, 1, 1] = b1, sigma[:, :, 1]
    L_ref[:, :, 2, 0], L_ref[:, :, 2, 1], L_ref[:, :, 2, 2] = b2, b3, sigma[:, :, 2]
    assert torch.allclose(L, L_ref, atol=1e-6)

    sigma_mat = L @ L.transpose(-1, -2)
    diag_ref = torch.diagonal(sigma_mat, dim1=-2, dim2=-1)
    assert torch.allclose(chol_diag(logvars, off), diag_ref, atol=1e-6)

    prec = torch.linalg.inv(_prior()[1])                # (K,3,3)
    tr_ref = torch.einsum("bkij,kji->bk", sigma_mat, prec)
    assert torch.allclose(chol_trace_term(chol_columns(logvars, off), prec),
                          tr_ref, atol=1e-6)

    x = torch.randn(2, K, D, generator=g)
    qf_ref = torch.einsum("bkd,kde,bke->bk", x, prec, x)
    assert torch.allclose(qform(x, prec), qf_ref, atol=1e-6)


def test_sampling_covariance_matches_cholesky():
    g = torch.Generator().manual_seed(5)
    logvars = torch.randn(K, D, generator=g)
    off = torch.randn(K, D, generator=g)
    n = 200_000
    eps = torch.randn(n, K, D, generator=g)
    L = build_chol(logvars[None], off[None])[0]                  # (K,3,3)
    samples = chol_apply(L.expand(n, K, D, D), eps)              # (n,K,3)
    emp = torch.einsum("nkd,nke->kde", samples, samples) / n
    sigma = L @ L.transpose(-1, -2)
    assert torch.allclose(emp, sigma, rtol=5e-2, atol=5e-2)


def test_closed_form_kl_matches_torch_distributions():
    """Model-level per-expert KL vs ``MultivariateNormal.kl_divergence``."""
    means, cov = _prior()
    torch.manual_seed(0)
    model = GeoVBMILE(dim=1, base_channels=8, num_experts=K,
                      variational_covariance="full",
                      prior_means=means, prior_vars=cov).eval()
    out = _forward(model, 1)
    # break the zero initialisation so the off-diagonals are non-trivial
    g = torch.Generator().manual_seed(17)
    off = torch.randn(*out["expert_off"].shape, generator=g)
    logvars = torch.randn(*out["expert_logvars"].shape, generator=g) - 1.0
    mus = torch.randn(*out["expert_mus"].shape, generator=g)
    # rebuild the model's closed form by hand
    prec = torch.linalg.inv(cov)
    _, logdet_p = torch.linalg.slogdet(cov)
    cols = chol_columns(logvars, off)
    kl = 0.5 * (chol_trace_term(cols, prec)
                + qform(mus - means[None, :, :, None], prec)
                - D - logvars.sum(dim=2) + logdet_p[None, :, None])
    # independent reference, per expert / pixel
    L = build_chol(logvars, off)
    sig = torch.einsum("bkijs,bkmjs->bkims", L, L)               # (B,K,3,3,*spatial)
    ref = torch.empty_like(kl)
    B, Kk, _, P = mus.shape
    for b in range(B):
        for k in range(Kk):
            p = torch.distributions.MultivariateNormal(means[k], covariance_matrix=cov[k])
            for i in range(P):
                q = torch.distributions.MultivariateNormal(
                    mus[b, k, :, i], covariance_matrix=sig[b, k, :, :, i])
                ref[b, k, i] = torch.distributions.kl_divergence(q, p)
    assert torch.allclose(kl, ref, atol=1e-3)


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [1, 2])
def test_full_q_forward_backward_is_finite(dim):
    torch.manual_seed(0)
    model = GeoVBMILE(dim=dim, base_channels=8, num_experts=K,
                      variational_covariance="full")
    x, low = _inputs(dim)
    # non-zero off-diagonal so the new path actually contributes
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "off" in name:
                param.normal_(0.0, 0.05)
    out = model(x, low_freq_model=low, num_samples=1)
    assert torch.isfinite(out["final_pred"]).all()
    assert "expert_off" in out and "expert_tr_sigma" in out
    loss = out["final_pred"].pow(2).mean() + out["kl_theta_per_expert"].mean()
    loss.backward()
    off_grads = [p.grad for n, p in model.named_parameters() if "off" in n]
    assert off_grads and all(g is not None and torch.isfinite(g).all() for g in off_grads)


def test_prior_shrinkage_brings_condition_number_down():
    g = torch.Generator().manual_seed(7)
    A = torch.randn(K, D, D, generator=g) * 0.05
    cov = A @ A.transpose(-1, -2)                     # nearly singular
    torch.manual_seed(0)
    plain = GeoVBMILE(dim=1, base_channels=8, num_experts=K, prior_means=_prior()[0],
                      prior_vars=cov, prior_cov_shrinkage=0.0)
    torch.manual_seed(0)
    shrunk = GeoVBMILE(dim=1, base_channels=8, num_experts=K, prior_means=_prior()[0],
                       prior_vars=cov, prior_cov_shrinkage=0.1)
    cond = lambda m: torch.linalg.cond(m.prior_vars).max().item()  # noqa: E731
    assert cond(shrunk) < cond(plain)


# ---------------------------------------------------------------------------
# well likelihood: expert (closed form) and mixture_predictive
# ---------------------------------------------------------------------------

def _expert_outputs(B=2, P=16, seed=11):
    g = lambda s: torch.Generator().manual_seed(s)  # noqa: E731
    mus = torch.randn(B, K, D, P, generator=g(seed))
    logvars = torch.randn(B, K, D, P, generator=g(seed + 1)) - 1.0
    off = torch.randn(B, K, D, P, generator=g(seed + 2))
    logits = torch.randn(B, K, P, generator=g(seed + 3))
    z = torch.softmax(logits, dim=1)
    return mus, logvars, off, z


def test_expert_nll_uses_tr_sigma_not_the_marginal():
    """``E_q[-log p]`` must use ``sum_k pi_k tr(Sigma_k)``, not ``tr(Sigma_marg)``."""
    mus, logvars, off, z = _expert_outputs()
    crit = VBELBOCore(well_nll_mode="expert", noise_mode="fixed",
                      seismic_obs_noise_init=5.0, well_obs_noise_init=0.5)
    tr = chol_diag(logvars, off).sum(dim=2)                       # (B,K,P)
    target = torch.randn(2, D, 16, generator=torch.Generator().manual_seed(4))
    mask = torch.ones(2, 1, 16)
    got = crit._nll_well_expert(mus, tr, z, target, mask)
    diff = (target.unsqueeze(1) - mus).clamp(-crit.clip_diff, crit.clip_diff)
    manual = ((z * (diff ** 2).sum(dim=2) + z * tr).sum(dim=1)
              / (2.0 * crit.well_var) + 0.5 * D * crit.log_two_pi
              + 0.5 * D * torch.log(crit.well_var)).mean()
    assert torch.allclose(got, manual, atol=1e-5)

    # the marginal variant would use the moment-matched variance instead
    second = (z.unsqueeze(2) * (chol_diag(logvars, off) + mus ** 2)).sum(dim=1)
    mu_marg = (z.unsqueeze(2) * mus).sum(dim=1)
    var_marg = torch.clamp(second - mu_marg ** 2, min=1e-6)
    marginal = crit._nll_well(
        mu_marg, torch.log(var_marg), target, mask)
    assert not torch.allclose(got, marginal)


def test_expert_mode_requires_the_full_variational_keys():
    crit = VBELBOCore(well_nll_mode="expert", noise_mode="fixed")
    mus, logvars, off, z = _expert_outputs()
    model_output = {
        "final_pred": torch.randn(2, 6, 16),
        "expert_mus": mus,
        "expert_logvars": logvars,
        "z_soft": z,
        "kl_z": torch.tensor(0.1),
        "kl_theta_per_expert": torch.rand(2, K, 16),
    }
    target = torch.randn(2, D, 16)
    with pytest.raises(ValueError, match="variational_covariance"):
        crit(model_output, target, mask=torch.ones(2, 1, 16))


def test_mixture_predictive_runs_and_uses_the_offdiagonals():
    mus, logvars, off, z = _expert_outputs()
    crit = VBELBOCore(well_nll_mode="mixture_predictive", noise_mode="fixed",
                      well_obs_noise_init=0.5)
    target = torch.randn(2, D, 16, generator=torch.Generator().manual_seed(8))
    mask = torch.ones(2, 1, 16)
    base = crit._nll_well_mixture_predictive(mus, off, logvars, z, target, mask)
    off2 = off.clone()
    off2[:, :, 0] += 0.3
    changed = crit._nll_well_mixture_predictive(mus, off2, logvars, z, target, mask)
    assert torch.isfinite(base) and torch.isfinite(changed)
    assert not torch.allclose(base, changed), "off-diagonals must affect this mode"


@pytest.mark.parametrize("mode", ["expert", "mixture_predictive"])
def test_pixel_mask_reduction_is_batch_invariant(mode):
    """Same two pixels scored alone or in a batch of 4 must give the same value."""
    mus, logvars, off, z = _expert_outputs(B=2, P=8, seed=13)
    crit = VBELBOCore(well_nll_mode=mode, noise_mode="fixed", well_obs_noise_init=0.5)
    target = torch.randn(2, D, 8, generator=torch.Generator().manual_seed(9))
    mask = torch.ones(2, 1, 8)
    if mode == "expert":
        tr = chol_diag(logvars, off).sum(dim=2)
        single = crit._nll_well_expert(mus, tr, z, target, mask)
        big = crit._nll_well_expert(
            torch.cat([mus, mus]), torch.cat([tr, tr]), torch.cat([z, z]),
            torch.cat([target, target]), torch.cat([mask, mask]))
    else:
        single = crit._nll_well_mixture_predictive(mus, off, logvars, z, target, mask)
        big = crit._nll_well_mixture_predictive(
            torch.cat([mus, mus]), torch.cat([off, off]), torch.cat([logvars, logvars]),
            torch.cat([z, z]), torch.cat([target, target]), torch.cat([mask, mask]))
    assert torch.allclose(single, big, atol=1e-6)


def test_noise_mode_fixed_removes_the_parameters():
    fixed = VBELBOCore(noise_mode="fixed", well_obs_noise_init=0.5)
    learned = VBELBOCore(noise_mode="learned")
    names_fixed = {n for n, _ in fixed.named_parameters()}
    names_learned = {n for n, _ in learned.named_parameters()}
    assert "log_well_var" not in names_fixed and "log_well_var" in names_learned
    assert "log_well_var" in fixed.state_dict()          # still a buffer
    with pytest.raises(ValueError):
        VBELBOCore(noise_mode="nope")


# ---------------------------------------------------------------------------
# KL objective contract and theory contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,expect", [
    (dict(kl_z_weight=1.0, kl_theta_weight=1.0), "elbo"),
    (dict(kl_z_weight=0.01, kl_theta_weight=0.01), "single_temperature"),
    (dict(kl_z_weight=0.05, kl_theta_weight=0.01), "legacy_weighted"),
    (dict(kl_z_weight=0.01, kl_theta_weight=0.01, expert_lambdas=[0.1, 0.2, 0.3]),
     "legacy_weighted"),
])
def test_kl_objective_contract_modes(kwargs, expect):
    crit = VBELBOCore(**kwargs)
    contract = crit.kl_objective_contract()
    assert contract["mode"] == expect
    assert contract["strict_posterior"] == (expect in ("elbo", "single_temperature"))


def _model(cov="full", **kw):
    torch.manual_seed(0)
    return GeoVBMILE(dim=1, base_channels=8, num_experts=K,
                     variational_covariance=cov, **kw)


def test_theory_contract_detects_mismatched_well_nll_mode():
    contract = build_theory_contract(
        _model("diag"), VBELBOCore(well_nll_mode="expert", noise_mode="fixed"))
    by_name = {c["check"]: c for c in contract["checks"]}
    assert by_name["well_nll_mode_requires_full_q"]["ok"] is False
    assert contract["all_ok"] is False


def test_theory_contract_all_ok_for_the_delivery_setup():
    means, cov = _prior()
    model = _model("full", prior_means=means, prior_vars=cov, prior_cov_shrinkage=0.1)
    crit = VBELBOCore(well_nll_mode="expert", noise_mode="fixed",
                      kl_z_weight=0.01, kl_theta_weight=0.01)
    contract = build_theory_contract(model, crit, require_full=True)
    assert contract["all_ok"] is True
    by_name = {c["check"]: c for c in contract["checks"]}
    assert by_name["variational_is_full"]["mandatory"] is True
    assert by_name["kl_objective_recorded"]["ok"] is True
    assert contract["objective"]["kl"]["mode"] == "single_temperature"


def test_theory_contract_flags_ineffective_shrinkage():
    """``prior_cov_shrinkage`` with a diagonal prior is silently ignored in the model."""
    model = _model("full", prior_cov_shrinkage=0.1)      # unit prior => ndim 2
    contract = build_theory_contract(
        model, VBELBOCore(well_nll_mode="expert", noise_mode="fixed"))
    by_name = {c["check"]: c for c in contract["checks"]}
    assert by_name["prior_cov_shrinkage_is_effective"]["ok"] is False


def test_require_full_is_mandatory_only_when_asked():
    crit = VBELBOCore(well_nll_mode="marginal", noise_mode="learned")
    diag = _model("diag")
    assert build_theory_contract(diag, crit, require_full=True)["all_ok"] is False
    assert build_theory_contract(diag, crit, require_full=False)["all_ok"] is True
    assert build_theory_contract(_model("full"), crit, require_full=True)["all_ok"] is True


# ---------------------------------------------------------------------------
# the released 1-D / 2-D main configs use the delivery contract
# ---------------------------------------------------------------------------

def _cfg(path):
    return resolve_config(argparse.Namespace(
        config=path, overrides=[], seed=None, device=None, name=None))


def test_main_1d_config_is_full_cov_with_the_closed_form_well_term():
    cfg = _cfg("configs/inversion/vbmile_1d.yaml")
    assert cfg["model"]["variational_covariance"] == "full"
    assert cfg["loss"]["well_nll_mode"] == "expert"
    assert cfg["loss"]["kl_z_weight"] == pytest.approx(cfg["loss"]["kl_theta_weight"])
    assert cfg["loss"]["seismic_weight"] == pytest.approx(0.03)
    assert cfg["theory_contract"]["require_full"] is True

    model = build_from_config(__import__("src.models", fromlist=["MODELS"]).MODELS,
                              cfg["model"])
    loss = build_from_config(__import__("src.losses", fromlist=["LOSSES"]).LOSSES,
                             cfg["loss"])
    contract = build_theory_contract(model, loss, require_full=True)
    assert contract["all_ok"] is True
    assert contract["objective"]["kl"]["mode"] == "single_temperature"


def test_main_2d_config_is_full_cov_and_keeps_the_marginal_well_term():
    cfg = _cfg("configs/inversion/vbmile_2d.yaml")
    assert cfg["model"]["variational_covariance"] == "full"
    assert cfg["prior"]["_target_"] == "gmm"          # full q needs the full-cov prior
    assert "well_nll_mode" not in cfg["loss"]        # 2-D keeps the marginal default
    assert cfg["theory_contract"]["require_full"] is True
    # build the model the way the entry point does: the GMM prior is injected, which is
    # what makes ``prior_cov_shrinkage`` effective
    means, cov = _prior()
    mcfg = dict(cfg["model"], prior_means=means, prior_vars=cov)
    model = build_from_config(__import__("src.models", fromlist=["MODELS"]).MODELS, mcfg)
    loss = build_from_config(__import__("src.losses", fromlist=["LOSSES"]).LOSSES,
                             cfg["loss"])
    assert loss.well_nll_mode == "marginal"
    contract = build_theory_contract(model, loss, require_full=True)
    assert contract["all_ok"] is True
    assert loss.kl_objective_contract()["mode"] == "legacy_weighted"  # λ̂ Eq.21
