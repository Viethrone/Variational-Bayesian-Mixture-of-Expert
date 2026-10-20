"""Choose the number of GMM components K (i.e. the number of experts).

Fits ``K = 1..k_max`` on the **normalized training well pixels** (each trace's samples,
z-scored with the statistics of ``data/interim/norm_stats.json``) using
``sklearn.mixture.GaussianMixture`` and reports:

* ``bic`` / ``aic`` — the usual MLE criteria (sklearn's own, penalizing 2 ln N / 2);
* ``bic_shrink`` — the same BIC recomputed with the shrunk variances the method actually
  uses (``GMMPriorFitter``'s shrinkage recipe);
* ``silhouette`` — silhouette score of the maximum-posterior grouping (Euclidean,
  normalized space);
* ``max_bhattacharyya`` / ``mean_bhattacharyya`` — pairwise component overlap,
  ``B = (mu_a-mu_b)^T (Sigma_a+Sigma_b)^-1 (mu_a-mu_b)/8 + 0.5 ln(|S|/sqrt(|Sa||Sb|))``
  with ``S = (Sa+Sb)/2``;
* ``min_weight`` / ``max_weight`` — extreme mixture weights (a tiny, unstable component
  appears once K is too large);
* ``cv_ll*`` — **held-out cross-validated per-sample log-likelihood**, the decisive
  criterion: K folds, each fold fitted on the in-fold training part and scored on the
  held-out part, reported as mean ± std. Folds are grouped by **depth blocks** (each well
  contributes a contiguous block per fold, all wells sharing a fold) so neighbouring
  depths of the same well never straddle a fold boundary.

Writes ``outputs/summary/gmm_k_selection.csv`` (and ``.json``).

Usage::

    python -m tools.preprocess.gmm_select --config configs/data/seismic1d.yaml [--k-max 5] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.priors.gmm import GMMPriorFitter
from src.utils.config import load_config

__all__ = [
    "load_train_pixels",
    "bhattacharyya_distance",
    "fit_gmm_metrics",
    "fit_shrinkage_bic",
    "cv_heldout_loglik",
    "select_k",
    "main",
]

DEFAULT_K_MAX = 5
DEFAULT_SEED = 0


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (project_root() / p)


def load_train_pixels(
    target_phys: np.ndarray, train_path: str | Path | None,
    norm_stats: dict[str, Any],
) -> np.ndarray:
    """train 井像素 -> 归一化矩阵 (N, 3)。

    Args:
        target_phys: (C, T, X) 物理弹性参数场。
        train_path: train 井索引文件（每行一道索引）；None 时退化为全剖面中线。
        norm_stats: 含 ``target_mean/target_std`` (3,1) 的统计字典。
    Returns:
        归一化空间 (N, 3)（物理场 z-score，沿用 train 统计口径）。
    """
    target = np.asarray(target_phys, dtype=np.float32)
    if target.ndim == 2:
        target = target[:, :, None]
    C, T, X = target.shape
    if train_path is not None:
        idx = sorted(int(x) for x in Path(train_path).read_text().split())
        idx = [i for i in idx if 0 <= i < X]
    else:
        idx = [X // 2]
    cols = target[:, :, idx].transpose(1, 2, 0)          # (T, n, C)
    X_norm = cols.reshape(-1, C).astype(np.float64)

    mu = np.asarray(norm_stats["target_mean"], dtype=np.float64).ravel()      # (C,)
    sd = np.asarray(norm_stats["target_std"], dtype=np.float64).ravel()
    return (X_norm - mu) / sd


def bhattacharyya_distance(
    mean_a: np.ndarray, var_a: np.ndarray,
    mean_b: np.ndarray, var_b: np.ndarray,
) -> float:
    """两个高斯间的 Bhattacharyya 距离（文档 M9 公式）。

    ``var_a/var_b`` 可为对角方差向量 (D,)（自动 ``np.diag``）或全协方差矩阵
    (D, D)（GMMPriorFitter full 口径，直接使用）。
    """
    mu_a = np.asarray(mean_a, dtype=np.float64).ravel()
    mu_b = np.asarray(mean_b, dtype=np.float64).ravel()
    va = np.asarray(var_a, dtype=np.float64)
    vb = np.asarray(var_b, dtype=np.float64)
    sa = np.diag(va.ravel()) if va.ndim <= 1 else va
    sb = np.diag(vb.ravel()) if vb.ndim <= 1 else vb
    if sa.shape != (mu_a.size, mu_a.size) or sb.shape != sa.shape:
        raise ValueError(f"方差输入形状不支持：var_a {va.shape} / var_b {vb.shape}，"
                         f"均值维 D={mu_a.size}")
    sm = 0.5 * (sa + sb)
    dm = mu_a - mu_b
    h1 = 0.125 * float(dm @ np.linalg.solve(sm, dm))
    _, ld_sm = np.linalg.slogdet(sm)
    _, ld_a = np.linalg.slogdet(sa)
    _, ld_b = np.linalg.slogdet(sb)
    h2 = 0.5 * (ld_sm - 0.5 * (ld_a + ld_b))
    return h1 + h2


def _component_overlaps(means: np.ndarray, variances: np.ndarray) -> list[float]:
    K = means.shape[0]
    dists: list[float] = []
    for a in range(K):
        for b in range(a + 1, K):
            dists.append(bhattacharyya_distance(means[a], variances[a],
                                                means[b], variances[b]))
    return dists


def fit_gmm_metrics(
    X_norm: np.ndarray, K: int, seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """diag-GMM（sklearn，MLE 口径）的 BIC/AIC/轮廓/Bhattacharyya。

    Returns:
        ``{"k", "bic", "aic", "silhouette", "max_bhattacharyya",
        "mean_bhattacharyya", "n_samples"}``。
    """
    from sklearn.mixture import GaussianMixture

    X = np.asarray(X_norm, dtype=np.float64)
    gmm = GaussianMixture(n_components=K, covariance_type="diag",
                          random_state=seed).fit(X)
    out: dict[str, Any] = {
        "k": int(K),
        "bic": float(gmm.bic(X)),
        "aic": float(gmm.aic(X)),
        "min_weight": float(gmm.weights_.min()),
        "max_weight": float(gmm.weights_.max()),
        "n_samples": int(X.shape[0]),
    }
    if K >= 2:
        from sklearn.metrics import silhouette_score

        labels = gmm.predict(X)
        out["silhouette"] = float(silhouette_score(X, labels))
        ovl = _component_overlaps(np.asarray(gmm.means_), np.asarray(gmm.covariances_))
        out["max_bhattacharyya"] = float(max(ovl))
        out["mean_bhattacharyya"] = float(np.mean(ovl))
    else:
        out["silhouette"], out["max_bhattacharyya"] = float("nan"), float("nan")
        out["mean_bhattacharyya"] = float("nan")
    return out


def fit_shrinkage_bic(
    X_norm: np.ndarray, K: int, seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """主方法先验口径（``GMMPriorFitter`` shrinkage 正则协方差）的 BIC 对照。

    协方差被平滑（0.9·分量 + 0.1·全局 + eps*eye，full 口径不裁剪），不再是 MLE 解，
    BIC 直接用 shrink 化分布的对数似然仍可对比（统一 K 下的模型证据排序，文档 M5
    注明口径）。full 方差参数数 = D(D+1)/2 每分量。
    """
    fitter = GMMPriorFitter(K=K, seed=seed)
    prior = fitter.fit(X_norm)
    means = prior["means"]                                # (K, 3)
    D = X_norm.shape[1]
    n = X_norm.shape[0]
    if "covariances" in prior:
        prec = np.linalg.inv(np.asarray(prior["covariances"]))          # (K,D,D)
        _, logdet = np.linalg.slogdet(np.asarray(prior["covariances"]))
        diff = X_norm[:, None, :] - means[None]                         # (N,K,D)
        qf = np.einsum("nki,kij,nkj->nk", diff, prec, diff)
        log_norm = -0.5 * (qf + logdet[None] + D * np.log(2 * np.pi))
        n_params = K * (D + D * (D + 1) // 2) + (K - 1)                 # 均值+协方差+权重
    else:
        variances = prior["variances"]
        diff2 = ((X_norm[:, None, :] - means[None]) ** 2) / variances[None]
        log_norm = -0.5 * (diff2.sum(axis=2) + np.log(variances).sum(axis=1)
                           + D * np.log(2 * np.pi))
        n_params = K * (2 * D) + (K - 1)                                # 均值+方差+权重
    lnL = float((np.logaddexp.reduce(log_norm, axis=1) - np.log(K)).sum())
    return {"k": int(K), "bic_shrink": float(-2 * lnL + n_params * np.log(n)),
            "lnL_shrink": lnL}


def cv_heldout_loglik(
    X_norm: np.ndarray, K: int, n_folds: int = 5, seed: int = DEFAULT_SEED,
    block_rows: int | None = None, covariance_type: str = "diag",
) -> dict[str, Any]:
    """留出交叉验证的逐样本对数似然（sklearn GMM，MLE 口径）。

    Args:
        X_norm: (N, D) 归一化测井像素。
        K: 分量数。
        n_folds: 折数。
        seed: 折切分随机种（GMM 拟合沿用同一随机种）。
        block_rows: 若给出，把连续 ``block_rows`` 行划为一个不可再分块、按块折行
            （同井相邻深度无论 train/val 都在同一块内）。测井像素按
            ``load_train_pixels`` 为时间主序：``block_rows = n_wells * (T // n_folds)``
            即把每道井的深度轴均分 ``n_folds`` 段。None 则随机 KFold。
        covariance_type: "diag"（与 BIC/AIC 行同族）或 "full"（与主方法先验同族，
            但 full 在 14400 样本重尾云上有过拟合风险，官方折 std 大，仅作对照）。
    Returns:
        ``{"cv_ll_mean", "cv_ll_std", "cv_ll_folds", "cv_n_folds",
        "cv_block_rows", "cv_covariance_type"}``。均值=各折留出子集上逐样本
        对数似然的平均（``sklearn.GaussianMixture.score``，天然对数）。
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import KFold

    X = np.asarray(X_norm, dtype=np.float64)
    n = X.shape[0]
    splits = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    if block_rows:
        if block_rows < 1:
            raise ValueError(f"block_rows 必须 ≥1，收到 {block_rows}")
        block_idx = np.minimum(np.arange(n) // block_rows, n // block_rows)
        n_blocks = int(block_idx.max()) + 1
        orig = [(tr, va) for tr, va in splits.split(np.arange(n_blocks))]
        folds = [(np.isin(block_idx, trb), np.isin(block_idx, vab))
                 for trb, vab in orig]
    else:
        folds = [(tr, va) for tr, va in splits.split(X)]
    per_fold: list[float] = []
    for tr_mask, va_mask in folds:
        gmm = GaussianMixture(n_components=K, covariance_type=covariance_type,
                              random_state=seed).fit(X[tr_mask])
        per_fold.append(float(gmm.score(X[va_mask])))
    return {"cv_ll_mean": float(np.mean(per_fold)),
            "cv_ll_std": float(np.std(per_fold)),
            "cv_ll_folds": per_fold,
            "cv_n_folds": n_folds,
            "cv_block_rows": block_rows,
            "cv_covariance_type": covariance_type}


def select_k(
    X_norm: np.ndarray, k_max: int = DEFAULT_K_MAX, seed: int = DEFAULT_SEED,
    cv_n_folds: int = 5, cv_block_rows: int | None = None,
    cv_covariance_type: str = "diag",
) -> list[dict[str, Any]]:
    """K=1..k_max 全表（MLE + shrinkage 对照 + 留出交叉验证对数似然）。"""
    rows: list[dict[str, Any]] = []
    for K in range(1, k_max + 1):
        row = fit_gmm_metrics(X_norm, K, seed=seed)
        row.update(fit_shrinkage_bic(X_norm, K, seed=seed))
        row.update(cv_heldout_loglik(X_norm, K, n_folds=cv_n_folds,
                                     seed=seed, block_rows=cv_block_rows,
                                     covariance_type=cv_covariance_type))
        rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="专家数 K 选择：GMM BIC/AIC/轮廓/重叠度（返修实验 M5，N6）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", default="configs/data/seismic1d.yaml",
                        help="data config (data.files.target + splits.train + norm_stats)")
    parser.add_argument("--k-max", type=int, default=DEFAULT_K_MAX,
                        help="largest candidate number of experts")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="number of held-out CV folds (depth blocks, T//folds points each)")
    parser.add_argument("--cv-cov", choices=["diag", "full"], default="diag",
                        help="CV covariance family: diag = same as the BIC row; full = same as the method prior")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="GMM random seed")
    args = parser.parse_args(argv)

    root = project_root()
    cfg_path = args.config if Path(args.config).is_absolute() else root / args.config
    cfg = load_config(str(cfg_path))
    data_cfg = cfg.get("data") or {}
    root_dir = _resolve(str(data_cfg.get("root", "data/processed")))
    files = data_cfg.get("files") or {}
    tgt_path = root_dir / files.get("target", "target.npy")
    if not tgt_path.exists():
        print(f"[error] target field not found: {tgt_path}", file=sys.stderr)
        return 2
    target = np.load(tgt_path)

    splits = data_cfg.get("splits") or {}
    train_rel = splits.get("train")
    train_path = _resolve(str(train_rel)) if train_rel else None

    stats_f = _resolve(str((data_cfg.get("normalization") or {}).get(
        "stats_file", "data/interim/norm_stats.json")))
    if not stats_f.exists():
        print(f"[error] 归一化统计不存在: {stats_f}", file=sys.stderr)
        return 2
    norm_stats = json.loads(stats_f.read_text(encoding="utf-8"))

    X = load_train_pixels(target, train_path, norm_stats)
    print(f"[i] train 井像素 (N,3) = {X.shape}  归一化口径 norm_stats.json")
    n_wells = None
    if train_path is not None and train_path.exists():
        n_wells = len(sorted(int(x) for x in train_path.read_text().split()))
    block_rows = None
    if n_wells:
        block_rows = n_wells * (target.shape[1] // args.cv_folds)
    print(f"[i] CV: {args.cv_folds} 折，时间块折行 block_rows={block_rows}"
          f"（{n_wells} 道 × T/{args.cv_folds}）" if block_rows else
          f"[i] CV: {args.cv_folds} 折，随机 KFold（未取到 train 道数）")
    rows = select_k(X, k_max=args.k_max, seed=args.seed,
                    cv_n_folds=args.cv_folds, cv_block_rows=block_rows,
                    cv_covariance_type=args.cv_cov)

    outdir = root / "outputs/summary"
    outdir.mkdir(parents=True, exist_ok=True)
    cols = ["k", "bic", "aic", "bic_shrink", "silhouette",
            "cv_ll_mean", "cv_ll_std", "max_bhattacharyya", "mean_bhattacharyya",
            "min_weight", "n_samples"]
    lines = [",".join(cols)]
    for r in rows:
        vals = [f"{r[c]:.4g}" if isinstance(r.get(c), float) else str(r.get(c)) for c in cols]
        lines.append(",".join(vals))
    (outdir / "gmm_k_selection.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "gmm_k_selection.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[ok] outputs/summary/gmm_k_selection.csv —— K=1..{args.k_max}, CV {args.cv_folds} 折")
    print(f"{'K':>2} {'BIC':>11} {'silhouette':>10} {'cv_ll':>9} {'cv_std':>7}"
          f" {'maxBha':>8} {'meanBha':>8} {'minW':>6}")
    for r in rows:
        print(f"{r['k']:>2} {r['bic']:>11.4g} {r['silhouette']:>10.4g}"
              f" {r['cv_ll_mean']:>9.4g} {r['cv_ll_std']:>7.4g}"
              f" {r['max_bhattacharyya']:>8.4g} {r['mean_bhattacharyya']:>8.4g}"
              f" {r['min_weight']:>6.3g}")
    best = max(rows, key=lambda r: r["cv_ll_mean"])
    # one-standard-error 规则：取在其最佳值的 1 个标准差内、K 最小者
    thresh = best["cv_ll_mean"] - best["cv_ll_std"]
    cands = [r for r in rows if r["cv_ll_mean"] >= thresh]
    k_selected = min(cands, key=lambda r: r["k"])["k"]
    best_s = min(rows, key=lambda r: r["bic_shrink"])
    print(f"[i] CV 对数似然峰值 K={best['k']}（{best['cv_ll_mean']:.3f}±{best['cv_ll_std']:.3f}）；"
          f"1SE 规则选定 K={k_selected}")
    print(f"[i] 注：BIC 谷值 K={min(rows, key=lambda r: r['bic'])['k']}（单调下降，"
          f"14000+ 样本量级无判别力，不作为选择依据）；shrinkage-BIC 谷值 K={best_s['k']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())