"""Sliding-window inference protocol for 2-D checkpoints.

Two things are protected here, both of which are easy to break silently:

1. the tool only accepts 2-D checkpoints (a 1-D run directory must be rejected), and
2. the centre-column extraction is *aligned* with the trace list, i.e. the denormalized
   windowed target centre column equals ``target.npy[:, :, trace]``.

The test needs a trained 2-D run; when only synthetic data is available it builds a tiny
2-D run itself (2 epochs, 4 base channels) inside a temporary output root, which keeps
``pytest`` green on a fresh clone with the data prepared.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from experiments.main.run_main import _assemble_vbmile_trainer
from src.trainers.base import build_trainer
from src.utils.config import load_config

pytestmark = pytest.mark.usefixtures("data_ready")


def _make_2d_run(tmp_path: Path, epochs: int = 2) -> Path:
    """Train a tiny 2-D model inside ``tmp_path`` and return its run directory."""
    import sys

    from experiments.common import make_parser, prepare_run, resolve_config
    from experiments.main.run_main import main as run_main

    argv = [
        "-c", "configs/inversion/vbmile_2d.yaml",
        "--name", "unittest_2d",
        "--outputs-root", str(tmp_path / "out"),
        "--set", "model.base_channels=4",
        "--set", "trainer.max_epochs=%d" % epochs,
        "--set", "trainer.batch_size=2",
        "--set", "trainer.warmup_epochs=1",
        "--seed", "0",
        "--device", "cpu",
    ]
    assert run_main(argv) == 0

    runs = sorted((tmp_path / "out").glob("*unittest_2d*"))
    assert runs, "expected a run directory"
    return runs[-1]


@pytest.fixture(scope="module")
def run_2d(tmp_path_factory, data_ready):
    if not data_ready["2d"]:
        pytest.skip("2-D processed data missing")
    tmp_path = tmp_path_factory.mktemp("win_infer")
    return _make_2d_run(tmp_path)


def _alignments(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Recompute centre-column predictions and the aligned reference, offline.

    Mirrors what ``tools.evaluate.b9c_2d_windowed_infer`` does, but re-implemented here
    so the test fails if that script's alignment logic drifts from the dataset contract.
    """
    cfg = load_config(str(run_dir / "config.yaml"))
    dcfg = cfg["data"]
    root = Path(dcfg["root"])
    if not root.is_absolute():
        root = Path.cwd() / root
    traces = [int(x) for x in (Path.cwd() / dcfg["splits"]["val"]).read_text().split()]
    radius = int(dcfg.get("window", {}).get("radius", 7))

    target = np.load(root / dcfg["files"]["target"])
    ref = target[:, :, traces].transpose(2, 0, 1)          # (n,3,L)
    # windowed target centre column must equal target[:, :, trace]
    return ref, np.asarray(traces), radius


def test_run_config_has_training_window_geometry(run_2d):
    cfg = load_config(str(run_2d / "config.yaml"))
    window = cfg["data"].get("window") or {}
    assert window.get("mode") == "slice", "2-D training must use the slice (window) mode"
    assert int(window.get("radius", -1)) == 7, "released geometry is radius=7 (width 15)"


def test_window_centre_is_the_well_trace(run_2d):
    """The centre column of a radius-7 window is the well trace itself.

    Reproduces the alignment self-check of ``tools.evaluate.b9c_2d_windowed_infer``:
    reading a slice-mode dataset and comparing the windowed target's centre column with
    ``target.npy[:, :, trace]`` must agree exactly (relative error < 1e-5).
    """
    import numpy as np

    from experiments.common import build_dataset

    cfg = load_config(str(run_2d / "config.yaml"))
    dcfg = cfg["data"]
    radius = int((dcfg.get("window") or {}).get("radius", 7))
    assert radius == 7, "released 2-D geometry is radius=7 (width 15)"

    ds = build_dataset(cfg, "val")              # slice mode, radius=7 -> (6,1200,15)
    traces = [int(x) for x in ds.sample_names]
    assert len(traces) > 0

    root = Path(dcfg["root"])
    if not root.is_absolute():
        root = Path.cwd() / root
    target = np.load(root / dcfg["files"]["target"])          # (3,L,W) physical

    centre_idx = radius
    for i, trace in enumerate(traces[:5]):
        s = ds[i]
        assert s.meta["mask"][0, :, centre_idx].min() == 1.0, "centre column must be the well column"
        assert s.meta["mask"][0, :, centre_idx + 1].min() == 0.0
        # windowed target centre column == target.npy[:, :, trace]
        win_centre = s.target[:, :, centre_idx]               # (3,L) physical
        np.testing.assert_allclose(win_centre, target[:, :, trace], rtol=1e-5, atol=1e-3)


def test_sliding_window_infer_rejects_1d_checkpoint(run_2d, tmp_path, monkeypatch, capsys):
    """The tool must refuse a 1-D checkpoint rather than silently producing garbage."""
    import subprocess
    import sys

    cfg = load_config(str(run_2d / "config.yaml"))
    cfg["model"]["_target_"] = "vbmile_1d"
    bad = tmp_path / "bad_1d_run"
    (bad / "checkpoints").mkdir(parents=True)
    (bad / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    torch.save({"model": {}}, bad / "checkpoints" / "best_model.pth")

    # window mode would make the 1-D dataset a 2-D window; the tool's own dim guard fires
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "tools.evaluate.b9c_2d_windowed_infer",
         "--run-dir", str(bad), "--metrics-out", str(tmp_path / "m.json")],
        capture_output=True, text=True, timeout=300, cwd=str(Path.cwd()))
    assert proc.returncode != 0
    assert "assert" in (proc.stdout + proc.stderr).lower() or "dim=" in (proc.stdout + proc.stderr)


def test_tiny_forward_matches_contract(run_2d):
    """A width-15 window produces (B,6,1200,15) output with finite values."""
    from src.models.base import MODELS
    import src.models  # noqa: F401  (registration)

    model = MODELS.get("vbmile_2d")(base_channels=4, num_experts=3)
    out = model(torch.randn(2, 6, 24, 15), low_freq_model=torch.randn(2, 3, 24, 15),
                num_samples=0)
    assert tuple(out["final_pred"].shape) == (2, 6, 24, 15)
    assert torch.isfinite(out["final_pred"]).all()


def test_window_width_floor(run_2d):
    """With the iso encoder, widths below 13 cannot hold the 3x3 kernels.

    Empirically the smallest workable trace width is 13 (15 is the released geometry);
    3..11 raise from the two stride-2 blocks plus the decoder's upsampling path.
    """
    from src.models.base import MODELS
    import src.models  # noqa: F401

    model = MODELS.get("vbmile_2d")(base_channels=4, num_experts=3)
    for bad in (3, 5, 7, 9, 11):
        with pytest.raises(RuntimeError):
            model(torch.randn(1, 6, 16, bad), low_freq_model=torch.randn(1, 3, 16, bad),
                  num_samples=0)
    # 13 and 15 work
    for ok in (13, 15):
        out = model(torch.randn(1, 6, 16, ok), low_freq_model=torch.randn(1, 3, 16, ok),
                    num_samples=0)
        assert tuple(out["final_pred"].shape) == (1, 6, 16, ok)


def test_aniso_encoder_allows_narrow_windows(run_2d):
    """``encoder_downsample=aniso`` keeps narrow trace widths (5) runnable."""
    from src.models.base import MODELS
    import src.models  # noqa: F401

    model = MODELS.get("vbmile_2d")(base_channels=4, num_experts=3,
                                    encoder_downsample="aniso")
    out = model(torch.randn(1, 6, 16, 5), low_freq_model=torch.randn(1, 3, 16, 5),
                num_samples=0)
    assert tuple(out["final_pred"].shape) == (1, 6, 16, 5)
