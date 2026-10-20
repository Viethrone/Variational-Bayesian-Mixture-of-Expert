"""P2 先验测试：GMMPrior 数学（对数似然 / 梯度中心差分 / 采样）、fitter 复原、Hungarian。"""

from __future__ import annotations

import numpy as np

from src.priors import PRIORS
from src.priors.gmm import GMMPrior, GMMPriorFitter, SpatialPriorGenerator, hungarian_match


def test_gmm_prior_registered():
    assert "gmm_diag" in PRIORS
    assert "gmm_full" in PRIORS


def test_gmm_prior_grad_center_difference():
    rng = np.random.default_rng(0)
    means = rng.normal(size=(3, 2))
    variances = np.abs(rng.normal(size=(3, 2))) + 0.2
    prior = GMMPrior(means, variances)
    x = np.array([0.3, -0.2])
    assert np.isfinite(prior.log_prob(x))

    g = prior.grad_log_prob(x)
    h = 1e-6
    gd = np.array([
        (prior.log_prob(x + h * e) - prior.log_prob(x - h * e)) / (2 * h) for e in np.eye(2)
    ])
    assert np.allclose(g, gd, atol=1e-5)


def test_gmm_prior_full_cov_grad_center_difference():
    """全协方差先验：log_prob/梯度/采样走 Σ⁻¹ 分支。"""
    rng = np.random.default_rng(7)
    means = rng.normal(size=(3, 2))
    covs = np.stack([np.asarray([[0.4, 0.25], [0.25, 0.6]]),
                     np.asarray([[0.8, -0.3], [-0.3, 0.5]]),
                     np.asarray([[0.3, 0.1], [0.1, 0.4]])])
    prior = GMMPrior(means, covariances=covs)
    x = np.array([0.3, -0.2])
    assert np.isfinite(prior.log_prob(x))

    g = prior.grad_log_prob(x)
    h = 1e-6
    gd = np.array([
        (prior.log_prob(x + h * e) - prior.log_prob(x - h * e)) / (2 * h) for e in np.eye(2)
    ])
    assert np.allclose(g, gd, atol=1e-5)

    s = prior.sample(rng, (2000,))
    assert s.shape == (2000, 2)
    assert np.isfinite(s).all()


def test_gmm_prior_full_cov_matches_diag_when_diagonal():
    """Σ 为对角矩阵时，全协方差分支与对角分支 log_prob 一致。"""
    rng = np.random.default_rng(8)
    means = rng.normal(size=(2, 3))
    variances = np.abs(rng.normal(size=(2, 3))) + 0.3
    covs = np.stack([np.diag(v) for v in variances])
    dg = GMMPrior(means, variances=variances)
    fl = GMMPrior(means, covariances=covs)
    x = rng.normal(size=3)
    assert np.allclose(dg.log_prob(x), fl.log_prob(x), atol=1e-10)
    assert np.allclose(dg.grad_log_prob(x), fl.grad_log_prob(x), atol=1e-10)


def test_gmm_prior_log_prob_greater_far_from_means():
    rng = np.random.default_rng(1)
    prior = GMMPrior(rng.normal(size=(2, 2)), np.ones((2, 2)))
    near = prior.log_prob(np.zeros(2))
    far = prior.log_prob(np.array([100.0, 100.0]))
    assert near > far


def test_gmm_prior_sample_shapes_and_range():
    rng = np.random.default_rng(2)
    prior = GMMPrior(np.array([[-3.0, 0.0], [3.0, 0.0]]), np.array([[0.1, 0.1], [0.1, 0.1]]))
    s = prior.sample(rng, (1000,))
    assert s.shape == (1000, 2)
    # 样本应聚在两簇附近
    dist = np.abs(s).max(axis=1)
    assert dist.mean() < 3.5


def test_fitter_recovers_synthetic_clusters_full_cov():
    """默认 full 口径：返回全协方差 (K,3,3)，均值近似复原合成簇。"""
    rng = np.random.default_rng(0)
    means_t = np.array([[-3, -2], [2, -1], [0, 3.0]])
    vars_t = np.array([[0.2, 0.3], [0.4, 0.15], [0.25, 0.5]])
    X = np.concatenate([rng.normal(means_t[k], np.sqrt(vars_t[k]), (2000, 2)) for k in range(3)], 0)
    pri = GMMPriorFitter(K=3, seed=42).fit(X)
    assert pri["means"].shape == (3, 2)
    assert pri["covariances"].shape == (3, 2, 2)
    assert pri["weights"].shape == (3,)
    # 正则化后仍对称正定
    for c in pri["covariances"]:
        assert np.allclose(c, c.T, atol=1e-12)
        assert np.linalg.eigvalsh(c).min() > 0
    for mk in means_t:
        assert np.linalg.norm(pri["means"] - mk, axis=1).min() < 1.5


def test_fitter_diag_mode_preserves_legacy():
    """covariance_type="diag" 保留 legacy cell 12 输出（variances + 裁剪）。"""
    rng = np.random.default_rng(0)
    means_t = np.array([[-3, -2], [2, -1], [0, 3.0]])
    vars_t = np.array([[0.2, 0.3], [0.4, 0.15], [0.25, 0.5]])
    X = np.concatenate([rng.normal(means_t[k], np.sqrt(vars_t[k]), (2000, 2)) for k in range(3)], 0)
    pri = GMMPriorFitter(K=3, seed=42, covariance_type="diag").fit(X)
    assert pri["means"].shape == (3, 2) and pri["variances"].shape == (3, 2)
    assert (pri["variances"] > 0).all()
    for mk in means_t:
        assert np.linalg.norm(pri["means"] - mk, axis=1).min() < 1.5


def test_fitter_shrinkage_and_clip():
    """退化的单簇数据：diag 模式 shrinkage 把分量方差拉向全局方差 + 裁剪。"""
    rng = np.random.default_rng(5)
    X = rng.normal(0, 0.3, (500, 2))
    pri = GMMPriorFitter(K=2, seed=42, covariance_type="diag",
                         shrinkage=0.1, eps=1e-4).fit(X)
    assert np.isfinite(pri["variances"]).all() and (pri["variances"] > 0).all()
    gv = np.var(X, axis=0)
    # 正则化后方差在 clip 范围内（diag 模式才裁剪；full 模式不裁剪）
    assert ((pri["variances"] >= 0.01 * gv).all()
            and (pri["variances"] <= 10 * gv).all())


def test_fitter_full_shrinkage_posdef_and_no_clip():
    """full 模式：shrunk 协方差 = (1-s)·Σk + s·Σglobal + eps·I，不裁剪但正定。"""
    rng = np.random.default_rng(9)
    X = rng.normal(0, 0.3, (500, 2))
    fitter = GMMPriorFitter(K=2, seed=42, shrinkage=0.1, eps=1e-4)
    pri = fitter.fit(X)
    for c in pri["covariances"]:
        assert np.linalg.eigvalsh(c).min() > 0
    # 无裁剪：不做 [0.01*var, 10*var] 逐元素截断，只要求至少半正定（对应对角口径的 clip）
    assert (pri["covariances"] >= -1e-12).all()
    assert "variances" not in pri


def test_fitter_to_from_npz_roundtrip_full():
    rng = np.random.default_rng(11)
    X = rng.normal(0, 1.0, (800, 3))
    pri = GMMPriorFitter(K=3, seed=42).fit(X)
    import io
    buf = io.BytesIO()
    GMMPriorFitter.to_npz(pri, buf)
    buf.seek(0)
    back = GMMPriorFitter.from_npz(buf)
    assert "covariances" in back and "variances" not in back
    np.testing.assert_allclose(back["covariances"], pri["covariances"])
    np.testing.assert_allclose(back["weights"], pri["weights"])


def test_hungarian_match_permutes():
    a = np.array([[2.0, 0.0], [0.0, 0.0]])
    b = np.array([[0.0, 0.0], [2.0, 0.0]])
    perm = hungarian_match(a, b)
    assert list(perm) == [1, 0]


def test_spatial_prior_generator_shape_and_alignment():
    C, H, W = 3, 8, 10
    # 低频 cube 由 3 个空间区域组成（每区域一种属性组合）
    target_low = np.empty((C, H, W), dtype=np.float64)
    for i in range(C):
        target_low[i] = np.linspace(0, 2, H * W).reshape(H, W) + (0.5 * i)
    well_priors = GMMPriorFitter(K=3, seed=42).fit(
        target_low.transpose(1, 2, 0).reshape(-1, C)[::7]   # 井位样本
    )
    gen = SpatialPriorGenerator(K=3, seed=42)
    prob, perm = gen.generate(target_low, well_priors)
    assert prob.shape == (3, H, W) and perm.shape == (3,)
    np.testing.assert_allclose(prob.sum(axis=0), 1.0, atol=1e-4)
    assert isinstance(perm[0], (int, np.integer))