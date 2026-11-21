"""Highest-risk gate: the 1-D state dict must match the legacy weight keys exactly.

``data/external/weights/vbmile_1d_best.pth`` holds the released 1-D weights
(base_channels=16, num_experts=3, norm=group). The 1-D branch must load it with
``strict=True``: identical key set and shapes. That is the hardest evidence that
the port is correct; a failure means falling back to a mapped format rather than
retraining from scratch.
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
    """Same key set, nothing missing or extra (buffers included)."""
    legacy = _legacy_state_dict()
    model = _vbmile_1d()
    model_keys = set(model.state_dict().keys())
    legacy_keys = set(legacy.keys())

    missing = legacy_keys - model_keys
    extra = model_keys - legacy_keys
    assert not missing, f"keys present in legacy but missing here: {sorted(missing)}"
    assert not extra, f"keys here but not in legacy: {sorted(extra)}"
    assert model_keys == legacy_keys


def test_strict_load_has_no_missing_or_unexpected():
    """strict load reports zero missing and zero unexpected keys."""
    model = _vbmile_1d()
    missing, unexpected = model.load_state_dict(_legacy_state_dict(), strict=True)
    assert not missing, f"strict load missing keys: {missing}"
    assert not unexpected, f"strict load unexpected keys: {unexpected}"


def test_loaded_checkpoint_forward_shapes():
    """After loading the legacy weights the forward honours the (B,6,L) contract."""
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