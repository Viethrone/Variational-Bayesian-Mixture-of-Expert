"""Rebuild data/processed, data/interim/norm_stats.json and data/splits/* from data/raw.

The recipe follows the legacy notebooks (AVA1d_inverse / AVA2d_inverse_phase):

    seismic6 (6, 1200, 1360)  <- mat['seismic_data'].transpose(2,0,1)[1:]   (0 deg dropped)
    target  (3, 1200, 1360)   <- data2.npy['parameter'].transpose(1,2,0)
    target_low                <- gaussian_filter(target, sigma=preprocess.low_sigma, axes=(1,2))
    spatial_prior (K,H,W)     <- spatial_prior.pt transposed to channel-first
    gmm_prior (K,3,3)         <- GMMPriorFitter refit (full covariance) on normalized train
                                                               (GMM.ipynb cell 13: reg_covar=1e-6 + shrinkage
                                                               0.1 + eps*eye, no clipping; raw npz kept for provenance)

Normalization statistics follow the notebooks: train subset = linspace(20, 1300, 12),
mean/std over (depth, trace) on that subset with +1e-8.

Usage::

    python -m tools.preprocess.run_preprocess --config configs/data/seismic1d.yaml
    python -m tools.preprocess.run_preprocess --config configs/data/seismic1d.yaml --overwrite
    python -m tools.preprocess.run_preprocess --config configs/data/seismic1d.yaml --check  # verify only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

from src.utils.config import load_config

__all__ = ["rebuild", "check", "main"]

_TEX = "├─"
_BR = "└─"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(cfg_val: str, root: Path) -> Path:
    p = Path(cfg_val)
    return p if p.is_absolute() else (root / p)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# main rebuild pipeline
# ---------------------------------------------------------------------------

def rebuild(config_path: str | Path = "configs/data/seismic1d.yaml", *,
            overwrite: bool = False) -> Dict[str, Any]:
    """Rebuild the processed data from the config; returns the manifest (with statistics)."""
    root = project_root()
    cfg = load_config(str(_resolve(str(config_path), root) if not Path(config_path).is_absolute() else config_path))

    data_cfg = cfg.get("data") or {}
    pre = cfg.get("preprocess") or {}
    files = data_cfg.get("files") or {}
    dtype = np.dtype(pre.get("dtype", "float32"))
    low_sigma = float(pre.get("low_sigma", 30))
    num_train = int(pre.get("num_train", 12))
    num_val = int(pre.get("num_val", 7))
    train_start, train_end = int(pre.get("train_start", 20)), int(pre.get("train_end", 1300))
    val_start, val_end = int(pre.get("val_start", 30)), int(pre.get("val_end", 1300))

    raw_obs = _resolve(pre["raw"]["observations"], root)
    raw_priors = _resolve(pre["raw"]["priors"], root)
    out_root = _resolve(pre.get("root", "data/processed"), root)
    interim_dir = _resolve(pre.get("interim_dir", "data/interim"), root)
    splits_dir = _resolve(pre.get("splits_dir", "data/splits"), root)

    out_root.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. read raw ----
    mat_path = raw_obs / "elastic_impedance_results.mat"
    data2_path = raw_obs / "data2.npy"
    if not mat_path.exists() or not data2_path.exists():
        raise FileNotFoundError(
            f"raw observation files missing ({mat_path} / {data2_path})."
            "Run python -m tools.preprocess.import_core_data first"
        )

    mat = loadmat(mat_path)
    data2 = np.load(data2_path, allow_pickle=True).item()

    seismic_phys = np.transpose(mat["seismic_data"], (2, 0, 1))[1:]  # drop 0 deg
    assert seismic_phys.shape[0] == 6, f"seismic must have 6 angles, got {seismic_phys.shape[0]}"
    target_phys = np.transpose(np.asarray(data2["parameter"]), (1, 2, 0))

    H, W = seismic_phys.shape[1:]
    assert seismic_phys.shape[1:] == target_phys.shape[1:] == (H, W), "seismic and target spatial shapes differ"
    log = [f"raw: seismic {seismic_phys.shape} / target {target_phys.shape} (0 deg dropped)"]

    # ---- 2. rebuild arrays ----
    arrays: Dict[str, np.ndarray] = {
        files.get("seismic", "seismic6.npy"): seismic_phys.astype(dtype),
        files.get("target", "target.npy"): target_phys.astype(dtype),
    }

    # low-frequency model
    arrays[files.get("target_low")] = gaussian_filter(
        target_phys.astype(dtype), sigma=low_sigma, axes=(1, 2)
    ).astype(dtype)

    # spatial prior + parameter prior
    spatial_prior_path = raw_priors / "spatial_prior.pt"
    if spatial_prior_path.exists():
        import torch  # lazily loaded here so the rest of the tool stays torch-free
        prior = torch.load(spatial_prior_path, map_location="cpu", weights_only=True)
        prior_np = prior.cpu().numpy().squeeze()
        assert prior_np.ndim == 3 and prior_np.shape == (H, W, 3) or prior_np.shape == (3, H, W), \
            f"unexpected spatial_prior shape {prior_np.shape}, expected (H,W,K) or (K,H,W)"
        if prior_np.shape[0] == 3 and prior_np.shape[1:] == (H, W):
            prior_np = np.transpose(prior_np, (1, 2, 0))
        arrays[files.get("spatial_prior")] = np.transpose(prior_np, (2, 0, 1)).astype(dtype)
    else:
        log.append("warning: spatial_prior.pt missing, skipping the spatial prior (uniform gating)")

    gmm_path = raw_priors / "gmm_priors.npz"
    if not gmm_path.exists():
        raise FileNotFoundError(f"raw prior file missing: {gmm_path} (run import_core_data first)")
    gmm_legacy = np.load(gmm_path)
    K = int(np.asarray(gmm_legacy["means"]).shape[0])

    # spatial-prior K must match the GMM K
    if f := files.get("spatial_prior"):
        arr = arrays.get(f)
        if arr is not None:
            assert arr.shape[0] == K, f"spatial_prior K={arr.shape[0]} != gmm K={K}"

    # ---- 3. normalization statistics (notebook convention) ----
    train_indices = np.linspace(train_start, train_end, num_train, dtype=int)
    val_indices = np.linspace(val_start, val_end, num_val, dtype=int)

    seismic_train = arrays[files["seismic"]][:, :, train_indices]
    target_train = arrays[files["target"]][:, :, train_indices]
    seismic_mean = np.mean(seismic_train, axis=(1, 2), keepdims=True).astype(dtype)   # (6,1,1)
    seismic_std = (np.std(seismic_train, axis=(1, 2), keepdims=True) + 1e-8).astype(dtype)
    target_mean = np.mean(target_train, axis=(1, 2), keepdims=True).astype(dtype)     # (3,1,1)
    target_std = (np.std(target_train, axis=(1, 2), keepdims=True) + 1e-8).astype(dtype)

    # ---- 3.5 GMM parameter prior: full-covariance refit on the normalized train subset ----
    gmm_out = out_root / files.get("gmm_prior", "gmm_prior.npz")
    if overwrite or not gmm_out.exists():
        from src.priors.gmm import GMMPriorFitter  # sklearn dependency (as in gmm_select)

        tm = target_mean.reshape(1, -1).astype(np.float64)
        ts = target_std.reshape(1, -1).astype(np.float64)
        cols = np.asarray(arrays[files["target"]], dtype=np.float64)[:, :, train_indices]
        X_norm = (cols.transpose(1, 2, 0).reshape(-1, cols.shape[0]) - tm) / ts
        prior = GMMPriorFitter(K=K, seed=42).fit(X_norm)   # full covariance, seed=42
        GMMPriorFitter.to_npz(prior, gmm_out)
        print(f"[save] {gmm_out.relative_to(root)}  means{K, 3} / covariances{K, 3, 3}")
    # the raw diagonal gmm_priors.npz is provenance only (legacy cell 12), not a live prior
    log.append(f"GMM prior: K={K}, {gmm_out.name} = full-covariance refit on normalized train"
               f"（cell 13，shrinkage=0.1，eps=1e-4）")

    # ---- 4. write out (atomic: tmp then rename) ----
    entries = []
    for name, arr in arrays.items():
        if arr is None:
            continue
        path = out_root / name
        rel = str(path.relative_to(root)).replace("\\", "/")
        if path.exists() and not overwrite:
            print(f"[skip] {rel} already exists (use --overwrite to rebuild)")
        else:
            _atomic_write_npy(path, arr)
            print(f"[save] {rel}  {arr.shape} {arr.dtype}")
        entries.append({"name": name, "shape": list(arr.shape), "dtype": str(arr.dtype),
                        "size": path.stat().st_size, "md5": _md5(path)})

    # ---- 5. statistics JSON ----
    stats = {
        "seismic_mean": seismic_mean.reshape(-1, 1).tolist(),
        "seismic_std": seismic_std.reshape(-1, 1).tolist(),
        "target_mean": target_mean.reshape(-1, 1).tolist(),
        "target_std": target_std.reshape(-1, 1).tolist(),
        "meta": {
            "trace_indices_train": train_indices.tolist(),
            "trace_indices_val": val_indices.tolist(),
            "num_train": num_train,
            "num_val": num_val,
            "low_sigma": low_sigma,
            "dataset": data_cfg.get("name"),
            "recipe": "notebook convention: z-score on the train subset, std +1e-8, shape (C,1)",
            "seismic_shape_6": list(arrays[files["seismic"]].shape),
        },
    }
    stats_path = _resolve(data_cfg["normalization"]["stats_file"], root)
    _atomic_write_json(stats_path, stats)
    print(f"[save] {stats_path.relative_to(root)}")

    # ---- 6. split files (namespaced by data.name so 1-D/2-D val traces do not collide) ----
    ns = (data_cfg.get("name") or "default").replace("/", "_")
    ns_dir = splits_dir / ns
    splits = {
        "train": train_indices,
        "val": val_indices,
        "test": np.arange(W),
    }
    split_paths: Dict[str, str] = {}
    for split_name, indices in splits.items():
        path = ns_dir / f"{split_name}.txt"
        _atomic_write_text(path, "\n".join(str(int(i)) for i in indices) + "\n")
        split_paths[split_name] = str(path.relative_to(root)).replace("\\", "/")
        print(f"[save] {split_paths[split_name]}  ({len(indices)} traces)")

    # ---- 7. processed MANIFEST ----
    manifest = {
        "generator": "tools/preprocess/run_preprocess.py",
        "config": str(config_path),
        "data_files": entries,
        "norm_stats": str(stats_path.relative_to(root)).replace("\\", "/"),
        "splits": {k: v for k, v in split_paths.items()},
        "trace_indices": {"train": train_indices.tolist(), "val": val_indices.tolist()},
        "notes": log,
    }
    manifest_path = out_root / "MANIFEST.json"
    _atomic_write_json(manifest_path, manifest)
    print("[save] processed/MANIFEST.json\n\n" + "\n".join(f"  {_BR if i == len(log) - 1 else _TEX} {s}"
                                                            for i, s in enumerate(log)))

    return manifest


def _atomic_write_npy(path: Path, arr: np.ndarray) -> None:
    """Atomic .npy write. np.save appends a suffix to names not ending in .npy, so tmp does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def check(config_path: str | Path = "configs/data/seismic1d.yaml") -> int:
    """Verify shapes and hashes against processed/MANIFEST.json and print the statistics."""
    root = project_root()
    cfg = load_config(str(config_path))
    pre = cfg.get("preprocess") or {}
    out_root = _resolve(pre.get("root", "data/processed"), root)
    manifest_path = out_root / "MANIFEST.json"
    if not manifest_path.exists():
        print(f"[err] {manifest_path} missing; run run_preprocess first")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = 0
    for e in manifest["data_files"]:
        path = out_root / e["name"]
        if not path.exists():
            print(f"[MISS] processed/{e['name']}")
            failures += 1
            continue
        arr = np.load(path, mmap_mode="r")
        shape_ok = list(arr.shape) == e["shape"]
        md5_ok = _md5(path) == e["md5"]
        status = "ok" if shape_ok and md5_ok else "MISMATCH"
        if not shape_ok or not md5_ok:
            failures += 1
        print(f"[{status:8s}] {e['name']:<28s} shape={list(arr.shape)} md5={e['md5'][:12]}")

    stats_rel = (cfg.get("data") or {}).get("normalization", {}).get("stats_file", "")
    stats_path = _resolve(stats_rel, root) if stats_rel \
        else root / pre.get("interim_dir", "data/interim") / "norm_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        sm, ss = np.asarray(stats["seismic_mean"]), np.asarray(stats["seismic_std"])
        tm, ts = np.asarray(stats["target_mean"]), np.asarray(stats["target_std"])
        print("\nnormalization statistics (train-subset z-score, shape (C,1)):")
        names = {0: "Vp", 1: "Vs", 2: "Den"}
        for i in range(tm.shape[0]):
            print(f"  target[{names.get(i, i)}]  mean={tm[i, 0]: .4f}  std={ts[i, 0]: .4f}")
        print(f"  seismic6 mean |mean|={np.abs(sm[:, 0]).mean():.4f}  |std|={ss[:, 0].mean():.4f}")
        print(f"  train traces: {stats['meta']['trace_indices_train']}")
        print(f"  val traces:   {stats['meta']['trace_indices_val']}")

    return failures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", default="configs/data/seismic1d.yaml")
    parser.add_argument("--overwrite", action="store_true", help="rebuild files that already exist")
    parser.add_argument("--check", action="store_true", help="verify the existing processed outputs and exit")
    args = parser.parse_args(argv)

    if args.check:
        return check(args.config)
    rebuild(args.config, overwrite=args.overwrite)
    return check(args.config)


if __name__ == "__main__":
    sys.exit(main())