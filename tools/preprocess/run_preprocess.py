"""从 data/raw 重建 data/processed、data/interim/norm_stats.json 与 data/splits/*。

配方与 legacy notebook（AVA1d_inverse / AVA2d_inverse_phase）逐字一致：

    seismic6 (6, 1200, 1360)  <- mat['seismic_data'].transpose(2,0,1)[1:]   （去掉 0 幅角）
    target  (3, 1200, 1360)   <- data2.npy['parameter'].transpose(1,2,0)
    target_low                <- gaussian_filter(target, sigma=preprocess.low_sigma, axes=(1,2))
    spatial_prior (K,H,W)     <- spatial_prior.pt 转置（通道在前）
    gmm_prior (K,3,3)         <- 归一化 train 子集上 GMMPriorFitter 全协方差重拟合
                                  （GMM.ipynb cell 13 口径：reg_covar=1e-6 + shrinkage
                                  0.1 + eps*eye，不裁剪；raw gmm_priors.npz 仅留作溯源）

    # 可选：include_baseline_data: true（默认 false）时额外生成
    #     seismic5_csistmc (1360, 5, 1200) <- data2.npy['tmp_synth']
    # 那是已归档基线（CSI-STMC / MTL）的输入，不属于 Geo-VB-MILE 主链路，
    # 公开版默认不生成（原仓库默认生成）。

归一化统计严格按 notebook 口径：train 子集 = linspace(20, 1300, 12)，
mean/std 在该子集沿 (depth, trace) 计算并 +1e-8。

用法::

    python -m tools.preprocess.run_preprocess --config configs/data/seismic1d.yaml
    python -m tools.preprocess.run_preprocess --config configs/data/seismic1d.yaml --overwrite
    python -m tools.preprocess.run_preprocess --config configs/data/seismic1d.yaml --check  # 只校验
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
# 重建主流程
# ---------------------------------------------------------------------------

def rebuild(config_path: str | Path = "configs/data/seismic1d.yaml", *,
            overwrite: bool = False) -> Dict[str, Any]:
    """按配置重建 processed 数据。返回 manifest（含统计量）。"""
    root = project_root()
    cfg = load_config(str(_resolve(str(config_path), root) if not Path(config_path).is_absolute() else config_path))

    data_cfg = cfg.get("data") or {}
    pre = cfg.get("preprocess") or {}
    files = data_cfg.get("files") or {}
    dtype = np.dtype(pre.get("dtype", "float32"))
    low_sigma = float(pre.get("low_sigma", 30))
    num_train = int(pre.get("num_train", 12))
    num_val = int(pre.get("num_val", 7))
    include_baseline = bool(pre.get("include_baseline_data", False))
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

    # ---- 1. 读 raw ----
    mat_path = raw_obs / "elastic_impedance_results.mat"
    data2_path = raw_obs / "data2.npy"
    if not mat_path.exists() or not data2_path.exists():
        raise FileNotFoundError(
            f"缺少 raw 观测文件（{mat_path} / {data2_path}）。"
            "请先运行: python -m tools.preprocess.import_core_data"
        )

    mat = loadmat(mat_path)
    data2 = np.load(data2_path, allow_pickle=True).item()

    seismic_phys = np.transpose(mat["seismic_data"], (2, 0, 1))[1:]  # 去 0 幅角
    assert seismic_phys.shape[0] == 6, f"seismic 应为 6 角，实际 {seismic_phys.shape[0]}"
    target_phys = np.transpose(np.asarray(data2["parameter"]), (1, 2, 0))

    H, W = seismic_phys.shape[1:]
    assert seismic_phys.shape[1:] == target_phys.shape[1:] == (H, W), "seismic 与 target 空间维度不一致"
    log = [f"raw 观测: seismic {seismic_phys.shape} / target {target_phys.shape}（0 幅角已去掉）"]

    # ---- 2. 重建数组 ----
    arrays: Dict[str, np.ndarray] = {
        files.get("seismic", "seismic6.npy"): seismic_phys.astype(dtype),
        files.get("target", "target.npy"): target_phys.astype(dtype),
    }
    # 5 角道集（已归档基线 CSI-STMC/MTL 的输入）；默认不产出，见 include_baseline_data
    if include_baseline and files.get("seismic5_csistmc"):
        arrays[files["seismic5_csistmc"]] = np.asarray(data2["tmp_synth"]).astype(dtype)

    # 低频模型
    arrays[files.get("target_low")] = gaussian_filter(
        target_phys.astype(dtype), sigma=low_sigma, axes=(1, 2)
    ).astype(dtype)

    # 空间先验 + 参数先验
    spatial_prior_path = raw_priors / "spatial_prior.pt"
    if spatial_prior_path.exists():
        import torch  # lazily loaded here so the rest of the tool stays torch-free
        prior = torch.load(spatial_prior_path, map_location="cpu", weights_only=True)
        prior_np = prior.cpu().numpy().squeeze()
        assert prior_np.ndim == 3 and prior_np.shape == (H, W, 3) or prior_np.shape == (3, H, W), \
            f"spatial_prior 形状异常 {prior_np.shape}，期望 (H,W,K) 或 (K,H,W)"
        if prior_np.shape[0] == 3 and prior_np.shape[1:] == (H, W):
            prior_np = np.transpose(prior_np, (1, 2, 0))
        arrays[files.get("spatial_prior")] = np.transpose(prior_np, (2, 0, 1)).astype(dtype)
    else:
        log.append("warning: 缺失 spatial_prior.pt，跳过空间先验（模型将退化为均匀 gating prior）")

    gmm_path = raw_priors / "gmm_priors.npz"
    if not gmm_path.exists():
        raise FileNotFoundError(f"缺少 raw 先验文件: {gmm_path}（请先运行 import_core_data）")
    gmm_legacy = np.load(gmm_path)
    K = int(np.asarray(gmm_legacy["means"]).shape[0])

    # 空间先验 K 与 GMM K 一致性
    if f := files.get("spatial_prior"):
        arr = arrays.get(f)
        if arr is not None:
            assert arr.shape[0] == K, f"spatial_prior K={arr.shape[0]} != gmm K={K}"

    # ---- 3. 归一化统计（严格 notebook 口径）----
    train_indices = np.linspace(train_start, train_end, num_train, dtype=int)
    val_indices = np.linspace(val_start, val_end, num_val, dtype=int)

    seismic_train = arrays[files["seismic"]][:, :, train_indices]
    target_train = arrays[files["target"]][:, :, train_indices]
    seismic_mean = np.mean(seismic_train, axis=(1, 2), keepdims=True).astype(dtype)   # (6,1,1)
    seismic_std = (np.std(seismic_train, axis=(1, 2), keepdims=True) + 1e-8).astype(dtype)
    target_mean = np.mean(target_train, axis=(1, 2), keepdims=True).astype(dtype)     # (3,1,1)
    target_std = (np.std(target_train, axis=(1, 2), keepdims=True) + 1e-8).astype(dtype)

    # ---- 3.5 GMM 参数先验：归一化 train 子集上全协方差重拟合（cell 13 口径）----
    gmm_out = out_root / files.get("gmm_prior", "gmm_prior.npz")
    if overwrite or not gmm_out.exists():
        from src.priors.gmm import GMMPriorFitter  # sklearn 依赖（与 gmm_select 一致）

        tm = target_mean.reshape(1, -1).astype(np.float64)
        ts = target_std.reshape(1, -1).astype(np.float64)
        cols = np.asarray(arrays[files["target"]], dtype=np.float64)[:, :, train_indices]
        X_norm = (cols.transpose(1, 2, 0).reshape(-1, cols.shape[0]) - tm) / ts
        prior = GMMPriorFitter(K=K, seed=42).fit(X_norm)   # full covariance, seed=42
        GMMPriorFitter.to_npz(prior, gmm_out)
        print(f"[save] {gmm_out.relative_to(root)}  means{K, 3} / covariances{K, 3, 3}")
    # raw 对角 gmm_priors.npz 只作为溯源（legacy cell 12 口径），不再直接作为先验使用
    log.append(f"GMM 先验: K={K}，{gmm_out.name} = 归一化 train 子集全协方差重拟合"
               f"（cell 13，shrinkage=0.1，eps=1e-4）")

    # ---- 4. 落盘（原子写：先写 tmp 再改名）----
    entries = []
    for name, arr in arrays.items():
        if arr is None:
            continue
        path = out_root / name
        rel = str(path.relative_to(root)).replace("\\", "/")
        if path.exists() and not overwrite:
            print(f"[skip] {rel} 已存在（--overwrite 重建）")
        else:
            _atomic_write_npy(path, arr)
            print(f"[save] {rel}  {arr.shape} {arr.dtype}")
        entries.append({"name": name, "shape": list(arr.shape), "dtype": str(arr.dtype),
                        "size": path.stat().st_size, "md5": _md5(path)})

    # ---- 5. 统计量 JSON ----
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
            "recipe": "notebook 口径：train 子集 z-score，std +1e-8；形状 (C,1)",
            "seismic_shape_6": list(arrays[files["seismic"]].shape),
        },
    }
    stats_path = _resolve(data_cfg["normalization"]["stats_file"], root)
    _atomic_write_json(stats_path, stats)
    print(f"[save] {stats_path.relative_to(root)}")

    # ---- 6. 划分文件（按 data.name 命名空间隔离，避免 1D/2D val 道数互相覆盖）----
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
        print(f"[save] {split_paths[split_name]}  ({len(indices)} 道)")

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
    """原子写 .npy。注意 np.save 会给不以 .npy 结尾的文件名追加后缀，故 tmp 也以 .npy 结尾。"""
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
# 校验
# ---------------------------------------------------------------------------

def check(config_path: str | Path = "configs/data/seismic1d.yaml") -> int:
    """对照 processed/MANIFEST.json 校验形状/哈希，并打印统计量。失败返回 1。"""
    root = project_root()
    cfg = load_config(str(config_path))
    pre = cfg.get("preprocess") or {}
    out_root = _resolve(pre.get("root", "data/processed"), root)
    manifest_path = out_root / "MANIFEST.json"
    if not manifest_path.exists():
        print(f"[err] 缺少 {manifest_path}，请先运行 run_preprocess")
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
        print("\n归一化统计量（train 子集 z-score，形状 (C,1)）:")
        names = {0: "Vp", 1: "Vs", 2: "Den"}
        for i in range(tm.shape[0]):
            print(f"  target[{names.get(i, i)}]  mean={tm[i, 0]: .4f}  std={ts[i, 0]: .4f}")
        print(f"  seismic 6 平均 |mean|={np.abs(sm[:, 0]).mean():.4f}  |std|={ss[:, 0].mean():.4f}")
        print(f"  train 道: {stats['meta']['trace_indices_train']}")
        print(f"  val 道:   {stats['meta']['trace_indices_val']}")

    return failures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", default="configs/data/seismic1d.yaml")
    parser.add_argument("--overwrite", action="store_true", help="重建已存在的 processed 文件")
    parser.add_argument("--check", action="store_true", help="只校验现有 processed 产物并退出")
    args = parser.parse_args(argv)

    if args.check:
        return check(args.config)
    rebuild(args.config, overwrite=args.overwrite)
    return check(args.config)


if __name__ == "__main__":
    sys.exit(main())