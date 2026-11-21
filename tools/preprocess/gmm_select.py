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
    """Train-well pixels -> normalized matrix (N, 3).

    Args:
        target_phys: (C, T, X) physical elastic-parameter volume.
        train_path: file of train trace indices (one per line); None falls back to the midline.
        norm_stats: statistics dict holding ``target_mean`` / ``target_std`` of shape (3, 1).
    Returns:
        Normalized (N, 3) pixels (z-score with the train statistics).
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
    """Bhattacharyya distance between two Gaussians (the M9 formula).

    ``var_a`` / ``var_b`` may be diagonal variance vectors (D,) (turned into a matrix) or
    full covariance matrices (D, D), used as is.
    """
    mu_a = np.asarray(mean_a, dtype=np.float64).ravel()
    mu_b = np.asarray(mean_b, dtype=np.float64).ravel()
    va = np.asarray(var_a, dtype=np.float64)
    vb = np.asarray(var_b, dtype=np.float64)
    sa = np.diag(va.ravel()) if va.ndim <= 1 else va
    sb = np.diag(vb.ravel()) if vb.ndim <= 1 else vb
    if sa.shape != (mu_a.size, mu_a.size) or sb.shape != sa.shape:
        raise ValueError(f"unsupported variance shape: var_a {va.shape} / var_b {vb.shape}, "
                         f"mean dimension D={mu_a.size}")
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
    """BIC / AIC / silhouette / Bhattacharyya for a diag-GMM fitted by MLE.

    Returns:
        ``{"k", "bic", "aic", "silhouette", "max_bhattacharyya",
        "mean_bhattacharyya", "n_samples"}``.
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
    """BIC reference for the shrinkage-regularised ``GMMPriorFitter`` covariance.

    The covariance is smoothed (0.9 component + 0.1 global + eps*eye, no clipping), so it is
    no longer an MLE; the log-likelihood of the shrunk density still ranks models at fixed K
    (see M5). A full covariance costs D(D+1)/2 parameters per component.
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
        n_params = K * (D + D * (D + 1) // 2) + (K - 1)                 # means + covariance + weights
    else:
        variances = prior["variances"]
        diff2 = ((X_norm[:, None, :] - means[None]) ** 2) / variances[None]
        log_norm = -0.5 * (diff2.sum(axis=2) + np.log(variances).sum(axis=1)
                           + D * np.log(2 * np.pi))
        n_params = K * (2 * D) + (K - 1)                                # means + variances + weights
    lnL = float((np.logaddexp.reduce(log_norm, axis=1) - np.log(K)).sum())
    return {"k": int(K), "bic_shrink": float(-2 * lnL + n_params * np.log(n)),
            "lnL_shrink": lnL}


def cv_heldout_loglik(
    X_norm: np.ndarray, K: int, n_folds: int = 5, seed: int = DEFAULT_SEED,
    block_rows: int | None = None, covariance_type: str = "diag",
) -> dict[str, Any]:
    """Held-out cross-validated per-sample log-likelihood (sklearn GMM, MLE).

    Args:
        X_norm: (N, D) normalized well pixels.
        K: number of components.
        n_folds: number of folds.
        seed: seed for the fold split (also used for the GMM fit).
        block_rows: group consecutive rows into atomic blocks before splitting, so neighbouring
            depths of one well never straddle train/val. Pixels from ``load_train_pixels`` are
            time-major, so ``block_rows = n_wells * (T // n_folds)`` splits each well depth axis
            into ``n_folds`` segments. None means a random KFold.
        covariance_type: "diag" (same family as the BIC/AIC rows) or "full" (same family as the
            method prior; on 14400 heavy-tailed samples it can overfit, so it is a reference only).
    Returns:
        ``{"cv_ll_mean", "cv_ll_std", "cv_ll_folds", "cv_n_folds",
        "cv_block_rows", "cv_covariance_type"}``. Reported values are the mean per-sample
        log-likelihood over the held-out folds (``GaussianMixture.score``).
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import KFold

    X = np.asarray(X_norm, dtype=np.float64)
    n = X.shape[0]
    splits = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    if block_rows:
        if block_rows < 1:
            raise ValueError(f"block_rows must be >= 1, got {block_rows}")
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
    """Full table for K=1..k_max (MLE + shrinkage reference + held-out log-likelihood)."""
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
        description="Choose K: GMM BIC/AIC/silhouette/overlap (revision experiments M5, N6)",
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
        print(f"[error] normalization stats missing: {stats_f}", file=sys.stderr)
        return 2
    norm_stats = json.loads(stats_f.read_text(encoding="utf-8"))

    X = load_train_pixels(target, train_path, norm_stats)
    print(f"[i] train well pixels (N,3) = {X.shape}  normalized with norm_stats.json")
    n_wells = None
    if train_path is not None and train_path.exists():
        n_wells = len(sorted(int(x) for x in train_path.read_text().split()))
    block_rows = None
    if n_wells:
        block_rows = n_wells * (target.shape[1] // args.cv_folds)
    print(f"[i] CV: {args.cv_folds} folds, time-block rows block_rows={block_rows}"
          f"({n_wells} traces x T/{args.cv_folds})" if block_rows else
          f"[i] CV: {args.cv_folds} folds, random KFold (train traces unknown)")
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

    print(f"[ok] outputs/summary/gmm_k_selection.csv - K=1..{args.k_max}, CV {args.cv_folds} folds")
    print(f"{'K':>2} {'BIC':>11} {'silhouette':>10} {'cv_ll':>9} {'cv_std':>7}"
          f" {'maxBha':>8} {'meanBha':>8} {'minW':>6}")
    for r in rows:
        print(f"{r['k']:>2} {r['bic']:>11.4g} {r['silhouette']:>10.4g}"
              f" {r['cv_ll_mean']:>9.4g} {r['cv_ll_std']:>7.4g}"
              f" {r['max_bhattacharyya']:>8.4g} {r['mean_bhattacharyya']:>8.4g}"
              f" {r['min_weight']:>6.3g}")
    best = max(rows, key=lambda r: r["cv_ll_mean"])
    # one-standard-error rule: smallest K within one standard deviation of the best
    thresh = best["cv_ll_mean"] - best["cv_ll_std"]
    cands = [r for r in rows if r["cv_ll_mean"] >= thresh]
    k_selected = min(cands, key=lambda r: r["k"])["k"]
    best_s = min(rows, key=lambda r: r["bic_shrink"])
    print(f"[i] CV log-likelihood peaks at K={best['k']} ({best['cv_ll_mean']:.3f}+-{best['cv_ll_std']:.3f}); "
          f"1SE rule selects K={k_selected}")
    print(f"[i] note: BIC minimum at K={min(rows, key=lambda r: r['bic'])['k']} (monotone,"
          f"uninformative at 14000+ samples, not used to select); shrinkage-BIC minimum K={best_s['k']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
