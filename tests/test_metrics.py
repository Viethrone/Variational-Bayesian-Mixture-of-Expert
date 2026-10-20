"""P2 指标测试：R² / PCC 逐通道（Evaluation.txt 口径）、4D batch、常量目标安全。"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import METRICS, PearsonCorrelation, R2Score
from src.metrics.seismic_metrics import compute_metrics_per_channel


def test_metrics_registered():
    assert "r2" in METRICS and "r2_score" in METRICS
    assert "pearson" in METRICS and "pcc" in METRICS


def test_r2_per_channel():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(3, 40, 50))
    p = 0.9 * t + 0.1 * rng.normal(size=t.shape)
    r2 = R2Score().compute(p, t)
    assert r2.shape == (3,)
    assert np.isfinite(r2).all() and (r2 > 0.7).all()

    # 全体标量模式
    r2_global = R2Score(per_channel=False).compute(p, t)
    assert np.isscalar(r2_global) and np.isfinite(r2_global)

    # 完美预测 -> 1
    assert np.allclose(R2Score().compute(t, t), 1.0, atol=1e-9)


def test_pearson_per_channel_and_4d():
    rng = np.random.default_rng(1)
    t = rng.normal(size=(3, 40, 50))
    p = 0.8 * t + 0.2 * rng.normal(size=t.shape)
    pc = PearsonCorrelation().compute(p, t)
    assert pc.shape == (3,) and (pc > 0.8).all()

    # batch 布局 (B,C,H,W)：显式 channel_axis=1
    pb, tb = p[None], t[None]
    pc_b = PearsonCorrelation(channel_axis=1).compute(pb, tb)
    assert pc_b.shape == (3,)
    np.testing.assert_allclose(pc_b, pc, atol=1e-12)


def test_pearson_perfect_and_constant():
    rng = np.random.default_rng(2)
    t = rng.normal(size=(2, 10, 10))
    assert np.allclose(PearsonCorrelation().compute(t, t), 1.0, atol=1e-9)
    # 常量目标：方差为 0 -> NaN 而非崩溃
    tconst = np.ones((2, 5, 5))
    out = PearsonCorrelation().compute(rng.normal(size=(2, 5, 5)), tconst)
    assert np.isnan(out).all()
    # 常量通道 + 变化通道混合也是 NaN 逐通道处理
    t2 = np.concatenate([np.ones((1, 5, 5)), rng.normal(size=(1, 5, 5))], axis=0)
    p2 = rng.normal(size=(2, 5, 5))
    out2 = PearsonCorrelation().compute(p2, t2)
    assert out2.shape == (2,) and np.isnan(out2[0])


def test_compute_metrics_per_channel_matches_eval_terms():
    rng = np.random.default_rng(3)
    t = rng.normal(size=(3, 20, 20))
    p = t + 0.05 * rng.normal(size=t.shape)
    d = compute_metrics_per_channel(p, t)
    assert set(d) == {"r2", "pearson", "rmse", "mae"}
    assert d["r2"].shape == d["rmse"].shape == (3,)
    assert (d["r2"] > 0.9).all() and (d["pearson"] > 0.95).all()