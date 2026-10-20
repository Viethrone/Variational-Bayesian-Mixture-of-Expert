"""Audit the ``data/`` layout and the statistical conventions.

Three checks:
1. raw -> processed completeness: shapes and md5 against ``processed/MANIFEST.json``
   (reuses ``run_preprocess.check``);
2. numerical conventions: the physical magnitude of ``processed/target``
   (Vp in [~1000, ~4700] m/s) plus the normalization statistics against reference values;
3. a sample-contract example: prints the shapes of one training trace
   (observation / target / mask / prior).

Usage::

    python -m tools.data_check.check_data --config configs/data/seismic1d.yaml
    python -m tools.data_check.check_data --config configs/data/seismic2d.yaml

Returns 0 when everything passes, 1 when at least one check fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.seismic_dataset import load_norm_stats

# reference values for the target statistics (full-volume, float64)
REF_TARGET_MEAN_LEGACY = np.array([[2856.0], [1343.7], [2.2334]], dtype=np.float32)
REF_TARGET_STD_LEGACY = np.array([[919.5], [711.2], [0.170]], dtype=np.float32)
# this repository uses the strict training subset (float32), so the digits differ from the
# full-volume reference; a relative tolerance of 2% separates "same data, other convention"
# from a genuine error.
TOL_REL = 0.02          # 2% relative tolerance
VP_RANGE = (900.0, 5000.0)   # physical Pp magnitude (typical Marmousi2 range)


def check_manifest(config_path: str) -> int:
    """Reuse the MANIFEST validation of ``run_preprocess.check``."""
    from tools.preprocess.run_preprocess import check

    failures = check(config_path)
    if not failures:
        print("\n[ok] shapes/md5 of processed files match MANIFEST")
    return failures


def check_stats(cfg: dict) -> int:
    """Compare the normalization statistics against the reference."""
    ndiv = (cfg.get("data") or {}).get("normalization", {})
    stats = load_norm_stats(ndiv.get("stats_file", "data/interim/norm_stats.json"),
                            Path((cfg.get("data") or {}).get("root", "data/processed")))
    tm, ts = stats["target_mean"], stats["target_std"]
    failures = 0
    print("target statistics (compared against the reference):")
    for i, name in enumerate(("Vp", "Vs", "Den")):
        rm = abs(float(tm[i, 0]) - float(REF_TARGET_MEAN_LEGACY[i, 0])) / abs(float(REF_TARGET_MEAN_LEGACY[i, 0]))
        rs = abs(float(ts[i, 0]) - float(REF_TARGET_STD_LEGACY[i, 0])) / abs(float(REF_TARGET_STD_LEGACY[i, 0]))
        ok = rm < TOL_REL and rs < TOL_REL
        failures += 0 if ok else 1
        print(f"  {name}: mean={tm[i,0]:8.3f} (ref {REF_TARGET_MEAN_LEGACY[i,0]:8.3f}, {rm*100:5.2f}%) "
              f"std={ts[i,0]:8.4f} (ref {REF_TARGET_STD_LEGACY[i,0]:8.4f}, {rs*100:5.2f}%) -> {'ok' if ok else 'FAIL'}")
    return failures


def check_sample(cfg: dict, split: str = "train") -> int:
    """Walk a real sample through ``build_dataset`` and assert the contract shapes."""
    from experiments.common import build_dataset

    ds = build_dataset(cfg, split)
    sample = ds[0]
    obs, tgt = sample.observation, sample.target
    print(f"\n[sample contract] {ds.name} split={split} len={len(ds)}")
    print(f"  observation {obs.shape} {obs.dtype}  target {tgt.shape} {tgt.dtype}")
    for key in ("target_low", "prior", "mask"):
        v = sample.meta.get(key)
        print(f"  meta[{key}] {None if v is None else v.shape}")
    is_2d = bool(cfg.get("data", {}).get("window", {})) and int(cfg["data"]["window"].get("radius", 0)) > 0
    exp_ndim = 3 if is_2d else 2
    ok = (obs.ndim == exp_ndim and tgt.ndim == exp_ndim
          and obs.shape[0] == 6 and tgt.shape[0] == 3)
    if not ok:
        print(f"  [fail] shapes violate the contract (expected ndim={exp_ndim}, (6,...)/(3,...))")
        return 1
    print("  [ok] shapes satisfy the (6,...)/(3,...) contract")
    return 0


def check_physical_range(cfg: dict) -> int:
    """Spot-check the Pp magnitude (a cheap proxy for a correct rebuild)."""
    from experiments.common import build_dataset

    ds = build_dataset(cfg, "train")
    tgt = ds.target  # (3, 1200, 1360)
    vp = tgt[0]
    ok = vp.min() > VP_RANGE[0] and vp.max() < VP_RANGE[1]
    print(f"\n[physical range] Vp in [{vp.min():.1f}, {vp.max():.1f}] "
          f"(expected ~[{VP_RANGE[0]:.0f},{VP_RANGE[1]:.0f}]) -> {'ok' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", default="configs/data/seismic1d.yaml")
    args = parser.parse_args(argv)

    from src.utils.config import load_config

    cfg = load_config(args.config)

    failures = 0
    failures += check_stats(cfg)
    failures += check_sample(cfg, "train")
    failures += check_physical_range(cfg)
    failures += check_manifest(args.config)

    print(f"\nresult: {'all checks passed' if failures == 0 else f'{failures} failed'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())