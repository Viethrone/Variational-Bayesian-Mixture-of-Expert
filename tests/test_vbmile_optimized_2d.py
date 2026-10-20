"""2-D method: single structural form, deterministic padding, seismic noise model.

The 2-D structure has one form only (merged output head, skip2 convolved before
upsampling, same-size tail, copy-kernel reflect padding) — there is no legacy variant to
toggle back to. These tests pin that structure, its numerics, and the seismic likelihood's
noise model.
"""

from __future__ import annotations

import argparse

import pytest
import torch
import torch.nn.functional as F

from experiments.common import resolve_config
from src.losses import VBELBOCore
from src.models import GeoVBMILE
from src.models.vbmile import (
    _AdaptiveConv2d,
    _install_manual_reflect,
    _manual_reflect_pad_axis,
)
from src.trainers.base import TrainerConfig
from src.trainers.vbmile_trainer import VBEMLETrainer
from src.utils.registry import build_from_config
from src.utils.run_info import RunInfo

PARAMS_2D = 335_965          # base_channels=16, K=3: the only 2-D structure
PARAMS_2D_FULL = 364_150     # same, with the full-covariance off-diagonal heads
N_REFLECT_CONV = 7           # reflect-padded convolutions in the 2-D network


@pytest.fixture(autouse=True)
def _torch_seed():
    torch.manual_seed(0)


def _model(**kwargs):
    base = dict(dim=2, base_channels=16, num_experts=3)
    base.update(kwargs)
    torch.manual_seed(0)          # identical weights across separately built models
    return GeoVBMILE(**base)


def _inputs(h=48, w=15, seed=123):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 6, h, w, generator=g)
    low = torch.randn(1, 3, h, w, generator=g)
    prior = torch.softmax(torch.randn(1, 3, h, w, generator=g), dim=1)
    return x, low, prior


def _forward(model, num_samples=0):
    x, low, prior = _inputs()
    model.eval()
    with torch.no_grad():
        return model(x, low_freq_model=low, prior=prior, num_samples=num_samples)


def _nparams(module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# the one and only 2-D structure
# ---------------------------------------------------------------------------

def test_structure_is_the_single_optimised_form():
    model = _model()
    assert _nparams(model) == PARAMS_2D
    # one 6-channel head per expert, no per-property 2-channel heads
    assert len(model.experts.out_merged) == 3
    assert not hasattr(model.experts, "out_vp")
    # same-size tail: padding == dilation
    assert model.experts.final_convs[0][0].padding == (5, 5)
    assert model.experts.global_residual[0].padding == (3, 3)
    # deterministic reflect padding is installed unconditionally
    assert model.n_manual_reflect_pads == N_REFLECT_CONV
    # default per-expert dilation is 7; H, W, K and the state-dict keys are unaffected
    for fuse in model.experts.conv_fuses:
        assert fuse.conv3_dilated[0].dilation == (7, 7)
    for key in ("expert_dilations",):
        assert key not in model.state_dict()


def test_forward_shape_ok():
    out = _forward(_model())
    assert out["final_pred"].shape == (1, 6, 48, 15)


def test_merged_head_takes_the_prior_bias():
    K = 3
    means = torch.arange(K * 3, dtype=torch.float32).reshape(K, 3)
    variances = torch.full((K, 3), 0.25)          # prior_logvars = log(0.25)
    model = _model(prior_means=means, prior_vars=variances)
    for k in range(K):
        bias = model.experts.out_merged[k][-1].bias.data
        assert torch.allclose(bias[:3], means[k])
        assert torch.allclose(bias[3:], torch.full((3,), float(torch.log(torch.tensor(0.25)))))


def test_tail_keeps_full_resolution():
    model = _model()
    conv = model.experts.final_convs[0][0]
    y = conv(torch.randn(1, conv.in_channels, 12, 12))
    assert y.shape[-2:] == (12, 12)
    gres = model.experts.global_residual[0]
    z = gres(torch.randn(1, gres.in_channels, 12, 12))
    assert z.shape[-2:] == (12, 12)


def test_skip2_conv_runs_before_upsampling():
    """The skip2 convolution sees the native (coarse) resolution, not the upsampled one."""
    model = _model()
    seen = []
    for k, conv in enumerate(model.experts.skip2_convs):
        conv.register_forward_hook(lambda m, i, o, k=k: seen.append(i[0].shape[-2:]))
    _forward(model)
    # skip2 is (H/4, W/4) (aniso keeps the 5-trace axis), skip1 is (H/2, W/2)
    assert len(seen) == len(model.experts.skip2_convs)
    for shape in seen:
        assert shape != (24, 15)


# ---------------------------------------------------------------------------
# per-expert receptive field
# ---------------------------------------------------------------------------

def test_expert_dilations_applied_without_new_parameters():
    ref = _model()
    model = _model(expert_dilations=[7, 15, 7])
    for k, d in enumerate([7, 15, 7]):
        assert model.experts.conv_fuses[k].conv3_dilated[0].dilation == (d, d)
    assert _nparams(model) == PARAMS_2D
    assert set(model.state_dict()) == set(ref.state_dict())
    _forward(model)


@pytest.mark.parametrize("dilations", [[7, 15], [7, 15, 7, 7], [7, 0, 7]])
def test_expert_dilations_validated(dilations):
    with pytest.raises(ValueError):
        _model(expert_dilations=dilations)


def test_expert_dilations_rejected_for_1d():
    with pytest.raises(ValueError):
        GeoVBMILE(dim=1, base_channels=8, num_experts=3, expert_dilations=[7, 15, 7])


# ---------------------------------------------------------------------------
# deterministic padding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape,pad", [((1, 2, 9, 11), 2), ((1, 1, 5, 5), 3),
                                       ((2, 3, 4, 6), 1)])
def test_manual_reflect_matches_f_pad(shape, pad):
    x = torch.randn(*shape)
    for dim, squeeze in ((-1, (pad, pad, 0, 0)), (-2, (0, 0, pad, pad))):
        expected = F.pad(x, squeeze, mode="reflect")
        assert torch.equal(_manual_reflect_pad_axis(x, pad, dim), expected)


def test_installed_reflect_numerics_match_f_pad():
    """Reverting the copy-kernel padding reproduces the model output bit-for-bit."""
    model = _model()
    o_det = _forward(model)
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) and m.padding_mode == "reflect":
            if isinstance(m, _AdaptiveConv2d):
                m.manual_reflect = False
            else:
                del m.forward        # restore the class method (plain F.pad reflect)
    o_ref = _forward(model)
    for key in ("final_pred", "z_soft", "logits", "kl_z"):
        assert torch.equal(o_det[key], o_ref[key]), key


def test_install_manual_reflect_patches_instances_only():
    conv = torch.nn.Conv2d(2, 2, 3, padding=1, padding_mode="reflect")
    other = torch.nn.Conv2d(2, 2, 3, padding=1, padding_mode="reflect")
    other.load_state_dict(conv.state_dict())
    assert _install_manual_reflect(conv) == 1
    x = torch.randn(1, 2, 6, 6)
    assert torch.equal(conv(x), other(x))
    # patched per instance only, the class is untouched
    assert other.forward.__func__ is torch.nn.Conv2d.forward


def test_adaptive_conv_manual_reflect_matches_conv2d():
    conv = _AdaptiveConv2d(4, 4, 3, padding=1, padding_mode="reflect", aniso=True)
    man = _AdaptiveConv2d(4, 4, 3, padding=1, padding_mode="reflect", aniso=True,
                          manual_reflect=True)
    man.weight.data.copy_(conv.weight.data)
    man.bias.data.copy_(conv.bias.data)
    x = torch.randn(1, 4, 7, 9)
    assert torch.equal(conv(x), man(x))


# ---------------------------------------------------------------------------
# earthquake-like noise model of the seismic likelihood
# ---------------------------------------------------------------------------

def _loss_outputs(seed=7):
    g = lambda s: torch.Generator().manual_seed(s)  # noqa: E731
    logits = torch.randn(2, 3, 8, 5, generator=g(10))
    out = {
        "final_pred": torch.randn(2, 6, 8, 5, generator=g(7)),
        "expert_mus": torch.randn(2, 3, 3, 8, 5, generator=g(8)),
        "expert_logvars": torch.randn(2, 3, 3, 8, 5, generator=g(9)),
        "logits": logits,
        "z_soft": torch.softmax(logits, dim=1),
        "kl_z": torch.tensor(0.37),
        "kl_theta_per_expert": torch.rand(2, 3, 8, 5, generator=g(14)),
    }
    return out


def test_default_seismic_noise_is_iid_and_leaves_state_dict_alone():
    loss = VBELBOCore()
    assert loss.seismic_noise == "iid"
    assert set(loss.state_dict()) == {"log_seismic_var", "log_well_var"}
    out = _loss_outputs()
    g = lambda s: torch.Generator().manual_seed(s)  # noqa: E731
    target = torch.randn(2, 3, 8, 5, generator=g(11))
    syn = torch.randn(2, 6, 8, 5, generator=g(12))
    obs = torch.randn(2, 6, 8, 5, generator=g(13))
    _, comps = loss(out, target, syn_norm=syn, seismic=obs, mask=torch.ones(2, 1, 8, 5))
    assert "seismic_coh_var" not in comps and "seismic_white_var" not in comps


def test_seismic_noise_validated():
    with pytest.raises(ValueError):
        VBELBOCore(seismic_noise="nope")


def test_deviatoric_marginalises_the_common_mode():
    loss = VBELBOCore(seismic_noise="deviatoric", seismic_white_noise_init=0.4224)
    assert loss.seismic_noise == "deviatoric"
    assert "log_seismic_coh_var" not in loss.state_dict()   # fixed, not learned
    common = torch.randn(1, 6, 8, 5, generator=torch.Generator().manual_seed(21))
    common = common.mean(dim=-1, keepdim=True).expand(-1, -1, -1, 5)   # pure common mode
    dev_only = torch.randn(1, 6, 8, 5, generator=torch.Generator().manual_seed(22))
    dev_only = dev_only - dev_only.mean(dim=-1, keepdim=True)          # pure deviatoric part
    nll_common = loss._nll_seis_corr(common).mean()
    nll_zero = loss._nll_seis_corr(torch.zeros_like(common)).mean()
    nll_dev = loss._nll_seis_corr(dev_only).mean()
    nll_dev_plus_common = loss._nll_seis_corr(dev_only + common).mean()
    assert torch.allclose(nll_common, nll_zero, atol=1e-3)
    assert torch.allclose(nll_dev_plus_common, nll_dev, atol=1e-3)
    # relative to the iid NLL the marginalised common mode only adds a log-determinant term
    W = dev_only.shape[-1]
    iid = 0.5 * (dev_only ** 2 / loss.seismic_white_var + loss.log_two_pi
                 + torch.log(loss.seismic_white_var)).mean()
    offset = 0.5 * torch.log1p(W * loss.seismic_coh_var / loss.seismic_white_var) / W
    assert torch.allclose(nll_dev - iid, offset, rtol=1e-4, atol=1e-6)


def test_coherent_with_zero_common_variance_degenerates_to_iid():
    loss = VBELBOCore(seismic_noise="coherent", seismic_coh_noise_init=1.0,
                      seismic_white_noise_init=0.1)
    with torch.no_grad():
        loss.log_seismic_coh_var.fill_(-60.0)     # sigma_c^2 -> 0
    r = torch.randn(1, 6, 8, 5, generator=torch.Generator().manual_seed(23))
    iid = 0.5 * (r ** 2 / loss.seismic_white_var + loss.log_two_pi
                 + torch.log(loss.seismic_white_var)).mean()
    assert torch.allclose(loss._nll_seis_corr(r).mean(), iid, rtol=1e-3, atol=1e-3)


def test_correlated_noise_model_rejected_for_1d():
    loss = VBELBOCore(seismic_noise="deviatoric")
    loss.loss_dim = 1
    with pytest.raises(NotImplementedError):
        loss._nll_seis_corr(torch.randn(1, 6, 8))


def test_loss_dict_reports_noise_variances_when_enabled():
    loss = VBELBOCore(seismic_noise="deviatoric")
    loss.loss_dim = 2
    out = _loss_outputs()
    g = lambda s: torch.Generator().manual_seed(s)  # noqa: E731
    target = torch.randn(2, 3, 8, 5, generator=g(11))
    syn = torch.randn(2, 6, 8, 5, generator=g(12))
    obs = torch.randn(2, 6, 8, 5, generator=g(13))
    _, comps = loss(out, target, syn_norm=syn, seismic=obs, mask=torch.ones(2, 1, 8, 5))
    assert float(comps["seismic_coh_var"]) == pytest.approx(1e6)
    assert float(comps["seismic_white_var"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# trainer: stage-1 learning rate and checkpoint.save_last
# ---------------------------------------------------------------------------

class _Sample:
    def __init__(self, obs, target, low, mask):
        self.observation = obs
        self.target = target
        self.meta = {"target_low": low, "mask": mask, "prior": None}


class _FakeDataset:
    """Two tiny 2-D windows; ``to_norm`` is the identity (statistics are baked in)."""

    def __init__(self, n=2, h=12, w=5):
        self.n, self.h, self.w = n, h, w

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        obs = torch.randn(6, self.h, self.w, generator=g).numpy()
        target = torch.randn(3, self.h, self.w, generator=g).numpy()
        low = torch.randn(3, self.h, self.w, generator=g).numpy()
        mask = torch.ones(1, self.h, self.w).numpy()
        return _Sample(obs, target, low, mask)

    def to_norm(self, arr, which="seismic"):
        return arr


def _trainer(tmp_path, *, phases, full_config=None, epochs=1):
    model = GeoVBMILE(dim=2, base_channels=4, num_experts=3, encoder_downsample="aniso")
    criterion = VBELBOCore(kl_z_weight=0.01, kl_theta_weight=0.1)
    cfg = TrainerConfig.from_dict({
        "max_epochs": epochs, "lr": 1e-2, "weight_decay": 0.0,
        "warmup_epochs": 0, "warmup_start_lr": 1e-6, "lr_min": 1e-8,
        "grad_clip": 0.0, "batch_size": 2,
    })
    ds = _FakeDataset()
    from src.trainers.vbmile_trainer import build_vbmile_loader
    loader = build_vbmile_loader(ds, batch_size=2, shuffle=False)
    stats = {
        "target_mean": torch.zeros(3), "target_std": torch.ones(3),
        "seismic_mean": torch.zeros(6), "seismic_std": torch.ones(6),
    }
    run_info = RunInfo.create({"exp_name": "t"}, outputs_root=tmp_path, exp_name="t",
                              repo_root=tmp_path)
    trainer = VBEMLETrainer(run_info=run_info, cfg=cfg, model=model, criterion=criterion,
                            train_loader=loader, val_loader=loader,
                            device="cpu", stats=stats, phases=phases)
    if full_config is not None:
        trainer.full_config = full_config
    return trainer, run_info


def test_first_phase_learning_rate_is_applied(tmp_path, monkeypatch):
    created = []
    real_adam = torch.optim.Adam

    def spy(params, **kwargs):
        opt = real_adam(params, **kwargs)
        created.append(opt)
        return opt

    monkeypatch.setattr(torch.optim, "Adam", spy)
    trainer, _ = _trainer(tmp_path, phases=[{"epochs": 1, "lr": 0.25,
                                             "freeze_gating": False}])
    trainer.run()
    assert created, "the trainer must build an optimizer"
    assert created[0].defaults["lr"] == pytest.approx(0.25)


def test_without_phases_the_global_learning_rate_is_used(tmp_path, monkeypatch):
    created = []
    real_adam = torch.optim.Adam

    def spy(params, **kwargs):
        opt = real_adam(params, **kwargs)
        created.append(opt)
        return opt

    monkeypatch.setattr(torch.optim, "Adam", spy)
    trainer, _ = _trainer(tmp_path, phases=None)
    trainer.run()
    assert created[0].defaults["lr"] == pytest.approx(1e-2)


def test_save_last_writes_the_final_weights(tmp_path):
    trainer, run_info = _trainer(
        tmp_path, phases=None, full_config={"checkpoint": {"save_last": True}})
    trainer.run()
    assert (run_info.checkpoints_dir / "last_model.pth").is_file()
    assert (run_info.checkpoints_dir / "best_model.pth").is_file()


def test_save_last_off_by_default(tmp_path):
    trainer, run_info = _trainer(tmp_path, phases=None)
    trainer.run()
    assert not (run_info.checkpoints_dir / "last_model.pth").exists()
    assert (run_info.checkpoints_dir / "best_model.pth").is_file()


# ---------------------------------------------------------------------------
# the released 2-D recipe configures the whole optimised method
# ---------------------------------------------------------------------------

def test_released_2d_config_builds_the_optimised_recipe():
    cfg = resolve_config(argparse.Namespace(
        config="configs/inversion/vbmile_2d.yaml", overrides=[],
        seed=None, device=None, name=None))
    assert cfg["deterministic"] is True
    assert cfg["deterministic_algorithms"] is True
    assert cfg["data"]["loader"]["explicit_generator"] is True
    assert cfg["data"]["loader"]["num_workers"] == 0

    model = build_from_config(__import__("src.models", fromlist=["MODELS"]).MODELS,
                              cfg["model"])
    # the structure is not configurable any more: only geometry / dilations remain
    for gone in ("out_head_share", "skip2_conv_before_upsample", "tail_same_size",
                 "pad_mode"):
        assert gone not in cfg["model"]
    assert model.initial_temperature == pytest.approx(2.0)
    assert model.final_temperature == pytest.approx(0.1)
    assert model.variational_covariance == "full"
    assert model.n_manual_reflect_pads == N_REFLECT_CONV
    assert [f.conv3_dilated[0].dilation[0] for f in model.experts.conv_fuses] == [7, 15, 7]
    # merged head (335,965) + the three off-diagonal heads (364,150 with the full family)
    assert sum(p.numel() for p in model.parameters()) == PARAMS_2D_FULL

    loss = build_from_config(__import__("src.losses", fromlist=["LOSSES"]).LOSSES,
                             cfg["loss"])
    assert loss.seismic_noise == "deviatoric"
    assert loss.seismic_weight == pytest.approx(0.03)
    assert loss.loss_dim == 2

    phases = cfg["trainer"]["phases"]
    assert [p["epochs"] for p in phases] == [50, 250]
    assert phases[0]["lr"] == pytest.approx(5e-4)
    assert phases[0]["freeze_gating"] is False
    assert phases[1]["freeze_gating"] is True
