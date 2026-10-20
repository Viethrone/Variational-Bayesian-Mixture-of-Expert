"""Add Gaussian white noise to the observed angle gathers at a target SNR.

Noise is added in physical space with a single global scale::

    P_signal = mean(seismic ** 2)
    sigma    = sqrt(P_signal * 10 ** (-SNR_dB / 10))      # SNR_dB = 10*log10(Ps/Pn)

so the labels 2 / 5 / 10 / 15 are dB values. Each version is written next to the clean
observation as ``<stem>_snr{SNR}.npy`` (float32, same layout) and selected in training
with ``--set data.files.seismic=<file>``.

Conventions:
  * the normalization statistics stay those of ``data/interim/norm_stats.json``
    (z-scored on the clean training traces) — a linear z-score preserves the SNR, and
    sharing one set of statistics makes the runs comparable;
  * only the physical-space seismic file changes; raw, MANIFEST, target, low-frequency
    model and priors are untouched;
  * white noise in physical space with a single global sigma, seed 0.

用法::

    python -m tools.preprocess.add_noise --config configs/data/seismic1d.yaml \\
        --snr 2 5 10 15 [--seed 0] [--overwrite]

The measured SNR (dB) of every version is printed: the realized noise power can deviate
slightly from the nominal value, and that deviation is reported as-is.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.utils.config import load_config

__all__ = ["add_noise", "main"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_root(root: str) -> Path:
    p = Path(root)
    return p if p.is_absolute() else (project_root() / p)


def snr_fname(stem: str, snr: float) -> str:
    """文件主干加信噪比后缀：``seismic6_snr2.npy``。"""
    tag = f"snr{snr:g}"
    return f"{stem}_{tag}.npy"


def add_noise(
    seismic: np.ndarray,
    snr_dbs: Sequence[float],
    *,
    seed: int = 0,
    out_root: Path,
    fname: str = "seismic6.npy",
    overwrite: bool = False,
) -> dict[float, dict[str, float]]:
    """写加噪版本到 ``out_root``。

    Args:
        seismic: 物理空间观测 (C, ...)。
        snr_dbs: 目标 SNR 列表（**dB** 值，SNR_dB = 10·log10(Ps/Pn)）。
        seed: 随机种子（numpy default_rng）。
        out_root: processed 目录（在何处写 *._snr{N}.npy）。
        fname: 原观测文件名（决定输出主干名）。
        overwrite: 已存在时覆盖；False 则跳过已存在文件。
    Returns:
        ``{snr: {"file": str, "measured_snr_db": float, "sigma": float}}``。
    """
    arr = np.asarray(seismic, dtype=np.float32)
    p_signal = float(np.mean(arr.astype(np.float64) ** 2))
    if p_signal <= 0:
        raise ValueError(f"观测全零/负能量，无法按 SNR 定噪（P_signal={p_signal}）")
    rng = np.random.default_rng(seed)
    stem = Path(fname).stem

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    written: dict[float, dict[str, float]] = {}
    for snr in snr_dbs:
        if snr <= 0:
            raise ValueError(f"SNR 必须为正，收到 {snr}")
        fpath = out_root / snr_fname(stem, snr)
        if fpath.exists() and not overwrite:
            print(f"[skip] {fpath.name} 已存在（--overwrite 可覆盖）")
            written[float(snr)] = {
                "file": fpath.name,
                "measured_snr_db": float("nan"),
                "sigma": float("nan"),
            }
            continue
        sigma = float(np.sqrt(p_signal * 10.0 ** (-snr / 10.0)))
        noise = rng.normal(0.0, sigma, arr.shape).astype(np.float32)
        noisy = arr + noise
        np.save(fpath, noisy)
        p_noise = float(np.mean(noise.astype(np.float64) ** 2))
        measured = float(10.0 * np.log10(p_signal / p_noise))
        written[float(snr)] = {
            "file": fpath.name,
            "measured_snr_db": measured,
            "sigma": sigma,
        }
        print(f"[ok] {fpath.name}: target SNR={snr:g} dB measured SNR={measured:.3f} dB "
              f"sigma={sigma:.4g}")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Noisy observation generation: seismic -> seismic_snr{N}.npy (dB)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=str, default="configs/data/seismic1d.yaml",
                        help="data config (decides the processed root and the observation file)")
    parser.add_argument("--snr", type=float, nargs="+", required=True,
                        help="target SNR list (dB, SNR_dB=10*log10(Ps/Pn)), e.g. --snr 2 5 10 15")
    parser.add_argument("--seed", type=int, default=0, help="noise random seed")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite an existing noisy file with the same name")
    args = parser.parse_args(argv)

    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_config(str(cfg_path))
    data_cfg = cfg.get("data") or {}
    out_root = _resolve_root(str(data_cfg.get("root", "data/processed")))
    fname = str((data_cfg.get("files") or {}).get("seismic", "seismic6.npy"))

    seismic_path = out_root / fname
    if not seismic_path.exists():
        print(f"[error] observation not found: {seismic_path} (run run_preprocess first)",
              file=sys.stderr)
        return 2

    seismic = np.load(seismic_path)
    if seismic.ndim < 3:
        print(f"[error] the observation must be (C, ...), got {seismic.shape}", file=sys.stderr)
        return 2

    add_noise(seismic, args.snr, seed=args.seed, out_root=out_root,
              fname=fname, overwrite=args.overwrite)
    print("normalization statistics stay those of data/interim/norm_stats.json "
          "(clean training traces); nothing to recompute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())