"""P3 最高风险关：1D 分支 state-dict 与 legacy 权重键集逐字一致。

``data/external/weights/vbmile_1d_best.pth`` 是论文主方法 1D 模型的最终权重
（Geo_VBMILE_1D，base_channels=16、num_experts=3、norm=group）。本次重构的
1D 分支必须能以 ``strict=True`` 加载它 —— 键集一致 + 张量形状一致，
这是"移植正确性"最硬的一条证据（若失败需回退为双方协商的可映射格式，
而非另起炉灶重训练）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models import GeoVBMILE

# The original 1-D checkpoint is not redistributed with the public code release.
# Drop a compatible checkpoint at this path to activate these assertions.
WEIGHTS = "data/external/weights/vbmile_1d_best.pth"

_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / WEIGHTS

pytestmark = pytest.mark.skipif(
    not _WEIGHTS_PATH.exists(),
    reason=f"legacy checkpoint {WEIGHTS} not present in the public release",
)

CKPT_CONFIG = dict(
    dim=1,
    seismic_channels=6,
    low_freq_channels=3,
    base_channels=16,
    norm_type="group",
    num_experts=3,
    initial_temperature=1.0,
    final_temperature=0.05,
)


def _legacy_state_dict():
    sd = torch.load(WEIGHTS, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def _vbmile_1d():
    return GeoVBMILE(**CKPT_CONFIG)


def test_state_dict_key_set_matches_legacy():
    """键集一致：既不缺、也不多（含 buffers current_temperature/prior_means/prior_vars）。"""
    legacy = _legacy_state_dict()
    model = _vbmile_1d()
    model_keys = set(model.state_dict().keys())
    legacy_keys = set(legacy.keys())

    missing = legacy_keys - model_keys
    extra = model_keys - legacy_keys
    assert not missing, f"legacy 中有而模型缺的键: {sorted(missing)}"
    assert not extra, f"模型中有而 legacy 缺的键: {sorted(extra)}"
    assert model_keys == legacy_keys


def test_strict_load_has_no_missing_or_unexpected():
    """strict 加载零 missing / 零 unexpected（键集 + 形状完全对齐）。"""
    model = _vbmile_1d()
    missing, unexpected = model.load_state_dict(_legacy_state_dict(), strict=True)
    assert not missing, f"strict load 缺键: {missing}"
    assert not unexpected, f"strict load 多余键: {unexpected}"


def test_loaded_checkpoint_forward_shapes():
    """加载 legacy 权重后前向形状符合 (B,6,L) 预测合同。"""
    model = _vbmile_1d()
    model.load_state_dict(_legacy_state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 6, 64), low_freq_model=torch.randn(2, 3, 64),
                    prior=None, num_samples=1)
    assert tuple(out["final_pred"].shape) == (2, 6, 64)
    assert tuple(out["theta_sample"].shape) == (2, 3, 64)
    assert tuple(out["expert_mus"].shape) == (2, 3, 3, 64)
    assert tuple(out["z_soft"].shape) == (2, 3, 64)
    assert torch.isfinite(out["final_pred"]).all()