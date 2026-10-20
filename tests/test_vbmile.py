"""P3 VB-MoE 门：1D/2D 前反向有限、KL 有限、温度退火、两阶段行为。

门原句："B=2/L=64(1D)、B=2/64×24(2D) 前反向有限、KL 有限、
set_temperature/set_phase 行为；tests/test_legacy_checkpoint.py 断言 ……"。
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.losses import LOSSES, VBELBOCore
from src.models import MODELS, GeoVBMILE


@pytest.fixture(autouse=True)
def _torch_seed():
    torch.manual_seed(0)


def _prior_probs(shape):
    return torch.softmax(torch.randn(*shape), dim=1)


# ---------------------------------------------------------------------------
# T2 z_temperature：温度退火作用于门控软权重（默认关，逐位 legacy）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim,shape", [(1, (2, 6, 32)), (2, (2, 6, 32, 24))])
def test_z_temperature_off_is_plain_softmax(dim, shape):
    """默认关：z_soft 就是 softmax(logits)，且 z_temperature 不进 state-dict。"""
    torch.manual_seed(0)
    model = GeoVBMILE(dim=dim, base_channels=8, num_experts=3, z_temperature=False)
    assert model.z_temperature is False
    assert "z_temperature" not in model.state_dict()
    out = model(torch.randn(*shape), low_freq_model=torch.randn(2, 3, *shape[2:]),
                prior=None, num_samples=0)
    assert torch.equal(out["z_soft"], F.softmax(out["logits"], dim=1))


def test_z_temperature_tau_one_is_bitwise_identity():
    """τ=1 时 tempered 门控与原生路径逐位相同（镜像实现，两模型同权）。"""
    s, low = torch.randn(2, 6, 32), torch.randn(2, 3, 32)
    torch.manual_seed(0)
    m_off = GeoVBMILE(dim=1, base_channels=8, num_experts=3, z_temperature=False)
    torch.manual_seed(0)
    m_on = GeoVBMILE(dim=1, base_channels=8, num_experts=3,
                     initial_temperature=1.0, z_temperature=True)
    m_on.set_temperature(0, 100)                      # epoch 0 → τ = initial = 1.0
    assert float(m_on.current_temperature) == pytest.approx(1.0)
    o_off = m_off(s, low_freq_model=low, prior=None, num_samples=0)
    o_on = m_on(s, low_freq_model=low, prior=None, num_samples=0)
    assert torch.equal(o_off["z_soft"], o_on["z_soft"])
    assert torch.equal(o_off["kl_z"], o_on["kl_z"])


def test_z_temperature_sharpens_gate_and_keeps_kl_bounded():
    """退火终点：门控变锐、argmax 不变、kl_z 仍有界、前后向有限。"""
    torch.manual_seed(0)
    model = GeoVBMILE(dim=1, base_channels=8, num_experts=3,
                      initial_temperature=1.0, final_temperature=0.1,
                      z_temperature=True).train()
    s, low = torch.randn(2, 6, 32), torch.randn(2, 3, 32)
    model.set_temperature(0, 100)
    o1 = model(s, low_freq_model=low, prior=None, num_samples=1)
    model.set_temperature(99, 100)
    o2 = model(s, low_freq_model=low, prior=None, num_samples=1)
    assert float(model.current_temperature) < 0.11
    assert torch.equal(o1["z_soft"].argmax(1), o2["z_soft"].argmax(1))
    assert float(o2["z_soft"].amax(1).mean()) > float(o1["z_soft"].amax(1).mean())
    assert 0.0 <= float(o2["kl_z"]) <= math.log(3) + 1e-4
    neg, ld = VBELBOCore(well_nll_mode="mixture")(
        o2, target=torch.randn(2, 3, 32), target_mask=torch.ones(2, 3, 32))
    assert torch.isfinite(neg)
    neg.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


# ---------------------------------------------------------------------------
# 前向 / 反向有限
# ---------------------------------------------------------------------------

def test_1d_forward_backward_finite():
    model = GeoVBMILE(dim=1, base_channels=16, num_experts=3).train()
    seismic = torch.randn(2, 6, 64)
    low = torch.randn(2, 3, 64)
    prior = _prior_probs((2, 3, 64))

    out = model(seismic, low_freq_model=low, prior=prior, num_samples=1)
    assert tuple(out["final_pred"].shape) == (2, 6, 64)
    assert tuple(out["theta_sample"].shape) == (2, 3, 64)

    neg, ld = VBELBOCore()(out, target=torch.randn(2, 3, 64),
                           target_mask=torch.ones(2, 3, 64))
    assert torch.isfinite(neg), neg
    neg.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert torch.isfinite(ld["nll_well"]) and torch.isfinite(ld["seismic_nll"])


def test_2d_forward_backward_finite():
    model = GeoVBMILE(dim=2, base_channels=8, num_experts=3).train()
    seismic = torch.randn(2, 6, 64, 24)
    low = torch.randn(2, 3, 64, 24)
    prior = _prior_probs((2, 3, 64, 24))

    out = model(seismic, low_freq_model=low, prior=prior, num_samples=1)
    assert tuple(out["final_pred"].shape) == (2, 6, 64, 24)
    assert tuple(out["theta_sample"].shape) == (2, 3, 64, 24)

    neg, ld = VBELBOCore()(out, target=torch.randn(2, 3, 64, 24), mask=torch.ones(2, 3, 64, 24))
    assert torch.isfinite(neg), neg
    neg.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------
# KL 有限且有界
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim,shape", [(1, (2, 6, 64)), (2, (2, 6, 24, 32))])
def test_kl_z_bounded(dim, shape):
    model = GeoVBMILE(dim=dim, base_channels=8, num_experts=3).eval()
    seismic = torch.randn(*shape)
    low = torch.randn(2, 3, *shape[2:])
    out = model(seismic, low_freq_model=low, prior=None, num_samples=0)
    kl = float(out["kl_z"])
    assert torch.isfinite(out["kl_z"])
    # KL(q||uniform_K) ∈ [0, ln K]
    assert 0.0 <= kl <= math.log(3) + 1e-4, kl
    # 2D 额外返回逐专家 KL 分量
    if dim == 2:
        assert tuple(out["kl_theta_per_expert"].shape) == (2, 3, *shape[2:])
        assert torch.isfinite(out["kl_theta_per_expert"]).all()


# ---------------------------------------------------------------------------
# set_temperature / set_phase 行为
# ---------------------------------------------------------------------------

def test_set_temperature_anneals():
    model = GeoVBMILE(dim=1, initial_temperature=1.0, final_temperature=0.05)
    model.set_temperature(0, 100)
    t0 = float(model.current_temperature)
    model.set_temperature(99, 100)   # 训练循环最多调用 epoch-1（见 train_model_1d）
    t1 = float(model.current_temperature)
    model.set_temperature(50, 100)
    t2 = float(model.current_temperature)
    assert t0 == pytest.approx(1.0, abs=1e-5)   # epoch 0: cos=1 -> initial
    assert t2 == pytest.approx(0.525, abs=1e-4)  # epoch 50: cos=0 -> 中点
    assert 0.05 < t1 < t2 < t0                   # 单调余弦退火，末值逼近 final


def test_set_phase_1_and_2_requires_grad():
    model = GeoVBMILE(dim=2, base_channels=8)
    model.set_phase(1)
    assert all(p.requires_grad for p in model.parameters())

    model.set_phase(2)
    # 门控冻结，编码器 + 专家保留梯度
    assert all(not p.requires_grad for p in model.partition_layer.parameters())
    assert all(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.experts.parameters())

    # phase 3（N4，M3-D 联合微调）：全参数可训；越界阶段才报错
    model.set_phase(3)
    assert all(p.requires_grad for p in model.partition_layer.parameters())
    assert all(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError):
        model.set_phase(0)
    with pytest.raises(ValueError):
        model.set_phase(4)


# ---------------------------------------------------------------------------
# A1 decoder_share：1D 专家解码器去共享
# ---------------------------------------------------------------------------

def test_decoder_share_shared_keeps_legacy_keys_and_values():
    """默认 shared：与显式 shared 键集/数值逐位相同，且 share 不进 state-dict。"""
    s, low = torch.randn(2, 6, 32), torch.randn(2, 3, 32)
    torch.manual_seed(0)
    implicit = GeoVBMILE(dim=1, base_channels=8, num_experts=3).eval()
    torch.manual_seed(0)
    explicit = GeoVBMILE(dim=1, base_channels=8, num_experts=3,
                         decoder_share="shared").eval()
    assert implicit.decoder_share == "shared" == explicit.decoder_share
    assert set(implicit.state_dict()) == set(explicit.state_dict())
    assert not any(k.startswith("experts.conv1s") for k in implicit.state_dict())
    o1 = implicit(s, low_freq_model=low, prior=None, num_samples=0)
    o2 = explicit(s, low_freq_model=low, prior=None, num_samples=0)
    for key in ("final_pred", "expert_mus", "expert_logvars", "z_soft", "kl_z"):
        assert torch.equal(o1[key], o2[key]), key


def test_decoder_share_independent_preserves_rng_order():
    """independent 只在末尾补画：共享子模块参数与 shared 版逐位相同。"""
    torch.manual_seed(0)
    shared = GeoVBMILE(dim=1, base_channels=8, num_experts=3, decoder_share="shared")
    torch.manual_seed(0)
    indep = GeoVBMILE(dim=1, base_channels=8, num_experts=3, decoder_share="independent")
    sd_s, sd_i = shared.state_dict(), indep.state_dict()
    assert set(sd_s) <= set(sd_i)                       # 纯增量
    assert all(torch.equal(sd_s[k], sd_i[k]) for k in sd_s)
    assert any(k.startswith("experts.conv1s.") for k in sd_i)


def test_decoder_share_invalid_and_dim2():
    with pytest.raises(ValueError):
        GeoVBMILE(dim=1, decoder_share="bogus")
    with pytest.raises(ValueError):
        GeoVBMILE(dim=2, base_channels=8, decoder_share="independent")


def test_decoder_share_independent_forward_backward():
    """independent：参数量增加、前后向有限、逐专家模块梯度互异。"""
    torch.manual_seed(0)
    model = GeoVBMILE(dim=1, base_channels=8, num_experts=3,
                      decoder_share="independent").train()
    n_indep = sum(p.numel() for p in model.parameters())
    torch.manual_seed(0)
    n_shared = sum(p.numel() for p in GeoVBMILE(dim=1, base_channels=8, num_experts=3).parameters())
    assert n_indep > n_shared

    out = model(torch.randn(2, 6, 32), low_freq_model=torch.randn(2, 3, 32),
                prior=None, num_samples=1)
    assert tuple(out["final_pred"].shape) == (2, 6, 32)
    assert tuple(out["expert_mus"].shape) == (2, 3, 3, 32)
    neg, _ = VBELBOCore(well_nll_mode="mixture")(
        out, target=torch.randn(2, 3, 32), target_mask=torch.ones(2, 3, 32))
    assert torch.isfinite(neg)
    neg.backward()
    g0 = model.experts.conv2s[0].conv1x1[0].weight.grad
    g1 = model.experts.conv2s[1].conv1x1[0].weight.grad
    assert g0 is not None and g1 is not None
    assert torch.isfinite(g0).all() and torch.isfinite(g1).all()
    assert not torch.allclose(g0, g1)


# ---------------------------------------------------------------------------
# 注册表接线
# ---------------------------------------------------------------------------

def test_registry_registrations():
    assert "vbmile_1d" in MODELS and "vbmile_2d" in MODELS
    assert "vbelbo_1d" in LOSSES and "vbelbo_2d" in LOSSES

    m1 = MODELS.get("vbmile_1d")(base_channels=8, num_experts=2)
    assert isinstance(m1, GeoVBMILE) and m1.dim == 1
    m2 = MODELS.get("vbmile_2d")(base_channels=8, num_experts=2)
    assert isinstance(m2, GeoVBMILE) and m2.dim == 2

    # vbelbo_1d 默认权重为 1D 脚本口径（0.05/0.01），vbelbo_2d 为 1.0/1.0
    c1 = LOSSES.get("vbelbo_1d")()
    c2 = LOSSES.get("vbelbo_2d")()
    assert c1.kl_z_weight == pytest.approx(0.05) and c1.kl_theta_weight == pytest.approx(0.01)
    assert c2.kl_z_weight == pytest.approx(1.0) and c2.kl_theta_weight == pytest.approx(1.0)


def test_no_prior_buffers_default():
    """未提供先验时注册缓冲为 (K,3) 单位先验，前向仍有限。"""
    model = GeoVBMILE(dim=1, num_experts=2, base_channels=8)
    assert tuple(model.prior_means.shape) == (2, 3)
    out = model(torch.randn(1, 6, 32), num_samples=1)
    assert torch.isfinite(out["kl_theta"])
    assert torch.isfinite(out["final_pred"]).all()


# ---------------------------------------------------------------------------
# T3 井项口径：marginal（默认，逐位 legacy）/ mixture（严格混合 ELBO）
# ---------------------------------------------------------------------------

def _mixture_out(mus, logvars, z_soft):
    """构造最小 model_output：final_pred 用矩匹配边际（marginal 口径的输入）。"""
    mu = (z_soft.unsqueeze(2) * mus).sum(dim=1)
    second = (z_soft.unsqueeze(2) * (torch.exp(logvars) + mus**2)).sum(dim=1)
    logvar = torch.log((second - mu**2).clamp_min(1e-12))
    return {
        "final_pred": torch.cat([mu, logvar], dim=1),
        "expert_mus": mus,
        "expert_logvars": logvars,
        "z_soft": z_soft,
        "kl_z": torch.tensor(0.0),
        "kl_theta": torch.tensor(0.0),
    }


def test_well_nll_mode_default_is_marginal_bitwise():
    """默认构造与显式 marginal 逐位一致；非法取值报错。"""
    assert VBELBOCore().well_nll_mode == "marginal"
    c = VBELBOCore()
    out = _mixture_out(torch.randn(2, 3, 3, 16), torch.full((2, 3, 3, 16), -4.0),
                       torch.softmax(torch.randn(2, 3, 16), dim=1))
    tgt, msk = torch.randn(2, 3, 16), torch.ones(2, 3, 16)
    v_default = c(out, target=tgt, mask=msk)[1]["nll_well"]
    v_explicit = VBELBOCore(well_nll_mode="marginal")(out, target=tgt, mask=msk)[1]["nll_well"]
    assert torch.equal(v_default, v_explicit)
    with pytest.raises(ValueError):
        VBELBOCore(well_nll_mode="nope")
    # 工厂路径也透传该开关
    assert LOSSES.get("vbelbo_1d")(well_nll_mode="mixture").well_nll_mode == "mixture"


def test_well_nll_mixture_equals_marginal_when_experts_identical():
    """专家完全相同时，混合 ELBO 与矩匹配边际数值相等（Σ_k z_k = 1）。"""
    base = torch.randn(1, 1, 3, 12)
    mus = base.expand(1, 3, 3, 12).contiguous()
    logvars = torch.full((1, 3, 3, 12), -3.0)
    z_soft = torch.softmax(torch.randn(1, 3, 12), dim=1)
    out = _mixture_out(mus, logvars, z_soft)
    tgt = torch.randn(1, 3, 12)
    for msk in (torch.ones(1, 1, 12), torch.ones(1, 3, 12), None):
        m = VBELBOCore(well_nll_mode="marginal")(out, target=tgt, mask=msk)[1]["nll_well"]
        x = VBELBOCore(well_nll_mode="mixture")(out, target=tgt, mask=msk)[1]["nll_well"]
        assert torch.isfinite(x)
        assert x.item() == pytest.approx(m.item(), rel=1e-5, abs=1e-6)


def test_well_nll_mixture_differentiates_experts():
    """专家分布差异时，mixture 给每个专家自己的残差梯度；marginal 全部收到同一残差。"""
    d = 2.0
    mus = torch.zeros(1, 2, 3, 4, requires_grad=True)
    with torch.no_grad():
        mus[0, 0] = d
        mus[0, 1] = -d
    logvars = torch.full((1, 2, 3, 4), math.log(0.01))
    z_soft = torch.full((1, 2, 4), 0.5)
    tgt = torch.zeros(1, 3, 4)

    def _grads(mode):
        m = mus.detach().clone().requires_grad_(True)
        out = _mixture_out(m, logvars, z_soft)
        # 边际方差固定为常数，隔离出"残差"这一条通道（否则 logvar 也带梯度）
        out["final_pred"] = torch.cat(
            [out["final_pred"][:, :3], torch.full_like(out["final_pred"][:, 3:], math.log(0.01))],
            dim=1)
        VBELBOCore(well_nll_mode=mode)(out, target=tgt)[0].backward()
        return m.grad[0, 0].mean().item(), m.grad[0, 1].mean().item()

    g0m, g1m = _grads("marginal")
    g0x, g1x = _grads("mixture")
    # marginal：μ_marg 残差为零 → 两个专家梯度都被抹平
    assert abs(g0m) < 1e-6 and abs(g1m) < 1e-6
    # mixture：各自残差符号相反 → 责任度分化
    assert g0x > 0 and g1x < 0 and g0x * g1x < 0


def test_well_nll_mixture_full_model_forward_backward():
    """mixture 口径在真实模型输出上前后向有限，loss_dict 键集不变。"""
    torch.manual_seed(0)
    model = GeoVBMILE(dim=1, base_channels=16, num_experts=3).train()
    out = model(torch.randn(2, 6, 64), low_freq_model=torch.randn(2, 3, 64),
                prior=_prior_probs((2, 3, 64)), num_samples=1)
    crit = VBELBOCore(well_nll_mode="mixture")
    neg, ld = crit(out, target=torch.randn(2, 3, 64), target_mask=torch.ones(2, 3, 64))
    assert torch.isfinite(neg), neg
    neg.backward()
    assert set(ld) == {"total", "nll_well", "seismic_nll", "kl_z", "kl_theta",
                       "well_var", "seismic_var", "mse_monitor"}
    assert all(torch.isfinite(g).all() for p in model.parameters()
               if p.grad is not None for g in [p.grad])

# ---------------------------------------------------------------------------
# prior_cov_shrinkage：全协方差先验的条件数收缩（默认关，逐位 legacy）
# ---------------------------------------------------------------------------

def _ill_conditioned_prior(K=3, d=3, seed=0):
    """构造一个**近乎奇异**的全协方差先验：谱固定为 [1e-5, 1.2, 1.5]（cond ~1.5e5）。

    与实测 GMM 先验同病（归一化空间三通道近乎共线 → λ_min 3e-4 / cond 5804）。
    """
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    cov = q @ torch.diag(torch.tensor([1e-5, 1.2, 1.5])) @ q.T
    cov = 0.5 * (cov + cov.T)
    return torch.zeros(K, d), cov.expand(K, d, d).contiguous()


def test_prior_cov_shrinkage_default_off_is_bitwise_identity():
    """默认 0.0：prior_vars 逐位不变，且开关不进 state-dict（legacy 键集不破）。"""
    means, cov = _ill_conditioned_prior()
    torch.manual_seed(0)
    off = GeoVBMILE(dim=1, base_channels=8, num_experts=3,
                    prior_means=means, prior_vars=cov)
    assert off.prior_cov_shrinkage == 0.0
    assert "prior_cov_shrinkage" not in off.state_dict()
    assert torch.equal(off.prior_vars, cov)

    torch.manual_seed(0)
    tuned = GeoVBMILE(dim=1, base_channels=8, num_experts=3,
                      prior_means=means, prior_vars=cov, prior_cov_shrinkage=0.1)
    assert tuned.prior_cov_shrinkage == 0.1
    assert not torch.equal(tuned.prior_vars, cov)          # 只有开启才动
    assert torch.equal(tuned.prior_means, off.prior_means)  # 均值不受影响


def test_prior_cov_shrinkage_raises_min_eigenvalue():
    """正向：收缩把 λ_min 抬起来、cond 压下去，且仍是对称正定。"""
    means, cov = _ill_conditioned_prior()
    cond_before = float(torch.linalg.cond(cov).max())       # cond 是 (K,) 向量
    assert cond_before > 1e3                                # 前提：先验确实病态

    model = GeoVBMILE(dim=1, base_channels=8, num_experts=3,
                      prior_means=means, prior_vars=cov, prior_cov_shrinkage=0.1)
    pv = model.prior_vars
    assert pv.ndim == 3 and pv.shape == cov.shape
    assert torch.allclose(pv, pv.transpose(-1, -2))         # 对称
    assert torch.linalg.eigvalsh(pv).min() > 0              # 正定
    lam_before = torch.linalg.eigvalsh(cov).min()
    lam_after = torch.linalg.eigvalsh(pv).min()
    assert float(lam_after) > 10 * float(lam_before)         # λ_min 显著抬高
    assert float(torch.linalg.cond(pv).max()) < cond_before / 10   # cond 显著下降


def test_prior_cov_shrinkage_reduces_kl_and_forward_ok():
    """收缩后 KL(θ) 下降一个数量级以上，且前反向仍有限（口径不破）。"""
    means, cov = _ill_conditioned_prior()
    s, low = torch.randn(2, 6, 32), torch.randn(2, 3, 32)

    def _kl(m):
        out = m(s, low_freq_model=low, prior=None, num_samples=0)
        return out["kl_theta"]

    torch.manual_seed(0)
    big = _kl(GeoVBMILE(dim=1, base_channels=8, num_experts=3,
                        prior_means=means, prior_vars=cov))
    torch.manual_seed(0)
    small = _kl(GeoVBMILE(dim=1, base_channels=8, num_experts=3,
                          prior_means=means, prior_vars=cov, prior_cov_shrinkage=0.1))
    assert torch.isfinite(big) and torch.isfinite(small)
    # 收缩直接把 tr(Σp⁻¹V) 项压掉（λ_min 抬约两个数量级）→ KL 至少降数倍；
    # 不写"降 10×"是因为 KL 里 −Σlogvar / ln|Σp| 不随收缩等比变化。
    assert float(small) < float(big) / 5

    model = GeoVBMILE(dim=1, base_channels=8, num_experts=3,
                      prior_means=means, prior_vars=cov,
                      prior_cov_shrinkage=0.1).train()
    out = model(s, low_freq_model=low, prior=None, num_samples=1)
    crit = VBELBOCore()
    neg, ld = crit(out, target=torch.randn(2, 3, 32), target_mask=torch.ones(2, 3, 32))
    neg.backward()
    assert torch.isfinite(neg)
    assert set(ld) == {"total", "nll_well", "seismic_nll", "kl_z", "kl_theta",
                       "well_var", "seismic_var", "mse_monitor"}


def test_prior_cov_shrinkage_rejects_out_of_range():
    """收缩率必须落在 [0, 1)。"""
    with pytest.raises(ValueError, match="prior_cov_shrinkage"):
        GeoVBMILE(dim=1, prior_cov_shrinkage=1.0)
    with pytest.raises(ValueError, match="prior_cov_shrinkage"):
        GeoVBMILE(dim=1, prior_cov_shrinkage=-0.1)


# ---------------------------------------------------------------------------
# Direction A：专家输出对比正则（Hinge 形式，默认关，逐位 legacy）
# ---------------------------------------------------------------------------









