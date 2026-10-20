"""Bootstrap stability criterion for the number of GMM components K.

Motivation: likelihood-based criteria (BIC/AIC decrease monotonically and held-out CV
log-likelihood increases monotonically) have **no discriminative power** on a heavy-tailed
continuous elastic cloud — information criteria inherently favour over-segmentation when
the data are not clean clusters. This script adds likelihood-free evidence.

Method: B bootstrap resamples of the normalized training pixels (``subsize`` pixels per
draw, with replacement), each fitted as a K-component ``GaussianMixture`` (full
covariance, ``reg_covar=1e-3``). The mixture weights are sorted per draw and summarized:

* ``min_weight_mean`` / ``min_weight_q05`` — mean / 5% quantile of the smallest component
  weight ("are all components substantial?"). If q05 drops below the threshold, the model
  has picked up a spurious micro-cluster and that K is not trustworthy;
* ``nbig_mean`` — how many components on average reach ``threshold`` (an upper bound on
  the number of real groups);
* ``w_med`` / ``w_q05`` — the weight profile at each sorted rank;
* ``pair_close_frac`` — fraction of near-duplicate component pairs (Bhattacharyya < 1 in
  normalized space), i.e. whether K splits one group into neighbouring micro-clusters.

Usage::

    python -m tools.preprocess.gmm_bootstrap_stability --config configs/data/seismic1d.yaml \\
        [--k-min 3] [--k-max 6] [--n-boot 40] [--subsize 12000] [--min-weight 0.05]

Writes ``outputs/summary/gmm_k_bootstrap_stability.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import load_config
from tools.preprocess.gmm_select import load_train_pixels

__all__ = ["bootstrap_stability", "main"]

DEFAULT_SEED = 0


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pair_bhatt(means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """K×K 两两 Bhattacharyya 距离（归一化空间，full 协方差，与 gmm_select 同式）。"""
    K = means.shape[0]
    out = np.zeros((K, K))
    for a in range(K):
        for b in range(a + 1, K):
            sm = 0.5 * (covs[a] + covs[b])
            dm = means[a] - means[b]
            h1 = 0.125 * float(dm @ np.linalg.solve(sm, dm))
            _, ld_sm = np.linalg.slogdet(sm)
            _, ld_a = np.linalg.slogdet(covs[a])
            _, ld_b = np.linalg.slogdet(covs[b])
            out[a, b] = out[b, a] = h1 + 0.5 * (ld_sm - 0.5 * (ld_a + ld_b))
    return out


def bootstrap_stability(
    X_norm: np.ndarray, K: int, n_boot: int = 40, subsize: int = 12000,
    threshold: float = 0.05, seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """B 次 bootstrap 拟合 K 分量 GMM 的权重稳定性统计。

    Returns:
        ``{"k", "min_weight_mean", "min_weight_q05", "nbig_mean", "w_med",
        "w_q05", "pair_close_frac"}``。
    """
    from sklearn.mixture import GaussianMixture

    X = np.asarray(X_norm, dtype=np.float64)
    N = X.shape[0]
    if subsize > N:
        subsize = N
    rng = np.random.default_rng(seed)
    min_w, nbig, close = [], [], []
    w_all: list[np.ndarray] = []
    for b in range(n_boot):
        idx = rng.choice(N, subsize, replace=True)
        g = GaussianMixture(n_components=K, covariance_type="full",
                            random_state=b, reg_covar=1e-3).fit(X[idx])
        w = np.sort(g.weights_)
        min_w.append(float(w[0]))
        nbig.append(int((g.weights_ >= threshold).sum()))
        w_all.append(w)
        bp = _pair_bhatt(np.asarray(g.means_), np.asarray(g.covariances_))
        n_pair = K * (K - 1) // 2                          # 仅不同分量对
        close.append(float((bp[np.triu_indices(K, 1)] < 1.0).sum() / n_pair))
    arr = np.array(w_all)
    return {
        "k": int(K),
        "min_weight_mean": float(np.mean(min_w)),
        "min_weight_q05": float(np.quantile(min_w, 0.05)),
        f"nbig_mean(>= {threshold:.2g})": float(np.mean(nbig)),
        "w_med": np.median(arr, axis=0).round(4).tolist(),
        "w_q05": np.quantile(arr, 0.05, axis=0).round(4).tolist(),
        "pair_close_frac": float(np.mean(close)),
        "n_boot": n_boot,
        "subsize": int(subsize),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="测井像素 K 稳定性判据：bootstrap 分量人口占比稳定性（M5 并入 M8）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", default="configs/data/seismic1d.yaml",
                        help="数据配置（同 gmm_select）")
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--n-boot", type=int, default=40)
    parser.add_argument("--subsize", type=int, default=12000,
                        help="每次 bootstrap 采样像素数")
    parser.add_argument("--min-weight", type=float, default=0.05,
                        help="分量人口占比下限（充实判据）")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    root = project_root()
    cfg_path = args.config if Path(args.config).is_absolute() else root / args.config
    cfg = load_config(str(cfg_path))
    data_cfg = cfg.get("data") or {}
    base = root / str(data_cfg.get("root", "data/processed"))
    tgt_path = base / (data_cfg.get("files") or {}).get("target", "target.npy")
    if not tgt_path.exists():
        print(f"[error] 目标场不存在: {tgt_path}", file=sys.stderr)
        return 2
    target = np.load(tgt_path)
    splits = data_cfg.get("splits") or {}
    train_rel = splits.get("train")
    train_path = root / str(train_rel) if train_rel else None
    stats_rel = (data_cfg.get("normalization") or {}).get(
        "stats_file", "data/interim/norm_stats.json")
    stats_f = Path(stats_rel)
    if not stats_f.is_absolute():
        stats_f = root / stats_f
    if not stats_f.exists():
        print(f"[error] 归一化统计不存在: {stats_f}", file=sys.stderr)
        return 2
    norm_stats = json.loads(stats_f.read_text(encoding="utf-8"))
    X = load_train_pixels(target, train_path, norm_stats)
    print(f"[i] train 井像素 (N,3) = {X.shape}")

    rows = [bootstrap_stability(X, K, n_boot=args.n_boot, subsize=args.subsize,
                                threshold=args.min_weight, seed=args.seed)
            for K in range(args.k_min, args.k_max + 1)]

    outdir = root / "outputs/summary"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "gmm_k_bootstrap_stability.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'K':>2} {'minW_mean':>9} {'minW_q05':>9} {'nbig':>6} {'closePair':>9}")
    for r in rows:
        print(f"{r['k']:>2} {r['min_weight_mean']:>9.4f} {r['min_weight_q05']:>9.4f}"
              f" {r[f'nbig_mean(>= {args.min_weight:.2g})']:>6.2f} {r['pair_close_frac']:>9.3f}")
    stable = [r for r in rows if r["min_weight_q05"] >= args.min_weight]
    knee = max((r["k"] for r in stable), default=None)
    print(f"[i] 充实稳定（minW q05 ≥ {args.min_weight:.0%}）的 K 上限 = "
          f"{knee if knee is not None else '无'}（超过此值出现约瑟微簇）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())