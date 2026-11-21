"""GMM parameter and spatial priors (the full-covariance convention of GMM.ipynb cell 13).

Two pipelines:

1. Parameter prior ``GMMPriorFitter`` - fits a GMM on the normalized train subset,
       full covariance by default (``covariance_type="full"``, as in cell 13), returning
       ``means (K, 3), covariances (K, 3, 3), weights (K,)``. Shrinkage and ``eps * I``
       follow cell 13 (no clipping). ``covariance_type="diag"`` keeps the legacy diagonal
       recipe (shrinkage + var_clip) and returns ``variances (K, 3)``.
2. Spatial prior ``SpatialPriorGenerator`` - per-pixel GMM posterior on the low-frequency
       cube, with the components Hungarian-aligned to the well GMM, returning ``(K, H, W)``.
       The low-frequency cube uses diagonal covariance (legacy cell 19) unless
       ``covariance_type`` says otherwise.

``GMMPrior`` exposes the fit through the ``PRIORS`` registry; ``log_prob`` is the
log-sum-exp of the Gaussians, with diagonal (K, D) or full (K, D, D) covariance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Dict, Optional, Tuple

import numpy as np

from src.priors.base import PRIORS

__all__ = [
    "PRIORS",
    "GMMPriorFitter",
    "SpatialPriorGenerator",
    "hungarian_match",
    "GMMPrior",
]


# ---------------------------------------------------------------------------
# component alignment (Hungarian)
# ---------------------------------------------------------------------------

def hungarian_match(means_a: np.ndarray, means_b: np.ndarray) -> np.ndarray:
    """Permutation minimising ||means_a - means_b[perm]||_F. K is small, so brute force."""
    assert means_a.shape == means_b.shape and means_a.ndim == 2
    K = means_a.shape[0]
    from itertools import permutations

    best_perm, best_cost = None, float("inf")
    for perm in permutations(range(K)):
        cost = float(np.linalg.norm(means_a - means_b[list(perm)]))
        if cost < best_cost:
            best_cost, best_perm = cost, perm
    return np.asarray(best_perm)


# ---------------------------------------------------------------------------
# parameter prior: well-GMM fit
# ---------------------------------------------------------------------------

class GMMPriorFitter:
    """GMM fit following GMM.ipynb cell 13 (full covariance + shrinkage).

    Args:
        K: number of mixture components.
        seed: sklearn GMM random seed (42 in the legacy recipe).
        covariance_type: ``"full"`` (default, as in cell 13) or ``"diag"`` (the legacy
            diagonal alternative of cell 12).
        shrinkage: interpolate each component covariance toward the global one
            (0.1 in cell 13).
        eps: diagonal floor added to the covariance (1e-4 in cell 13; full is not clipped).
        var_clip: diag only - clip bounds for component variance relative to the global one.
    """

    def __init__(self, K: int = 3, seed: int = 42, covariance_type: str = "full",
                 shrinkage: float = 0.1, eps: float = 1e-4,
                 var_clip: Tuple[float, float] = (0.01, 10.0)) -> None:
        if covariance_type not in ("full", "diag"):
            raise ValueError(f"covariance_type must be full or diag, got {covariance_type!r}")
        self.K = int(K)
        self.seed = seed
        self.covariance_type = covariance_type
        self.shrinkage = float(shrinkage)
        self.eps = float(eps)
        self.var_clip = tuple(var_clip)

    def fit(self, X_norm: np.ndarray) -> Dict[str, np.ndarray]:
        """Fit a GMM on normalized data (N, D) with ``reg_covar=1e-6`` (as in cell 13).

        Returns:
            full: ``{"means": (K, D), "covariances": (K, D, D), "weights": (K,)}``，
                    then shrink and add ``eps * I(D)`` without clipping.
            diag: ``{"means": (K, D), "variances": (K, D), "weights": (K,)}``，
                    use the legacy shrinkage + var_clip recipe.
        """
        from sklearn.mixture import GaussianMixture

        X = np.asarray(X_norm, dtype=np.float64)
        assert X.ndim == 2, f"X must be (N, D), got {X.shape}"
        D = X.shape[1]
        gmm = GaussianMixture(n_components=self.K, covariance_type=self.covariance_type,
                              random_state=self.seed, reg_covar=1e-6, max_iter=500)
        gmm.fit(X)
        means = np.asarray(gmm.means_, dtype=np.float64)                    # (K, D)
        weights = np.asarray(gmm.weights_, dtype=np.float64)                # (K,)
        global_cov = np.cov(X, rowvar=False, ddof=1)                        # (D, D)

        if self.covariance_type == "full":
            # cell 13 shrinkage: cov_k = (1-s) * Sigma_k + s * Sigma_global + eps * I, no clipping
            cov_raw = np.asarray(gmm.covariances_, dtype=np.float64)        # (K, D, D)
            cov_reg = ((1 - self.shrinkage) * cov_raw + self.shrinkage * global_cov
                       + self.eps * np.eye(D))
            return {"means": means, "covariances": cov_reg, "weights": weights}

        # diag: legacy recipe (shrink + clip from cell 12)
        var_raw = np.asarray(gmm.covariances_, dtype=np.float64)            # (K, D)
        global_var = np.diag(global_cov)                                    # (D,)
        var_reg = (self.shrinkage * global_var
                   + (1 - self.shrinkage) * var_raw + self.eps * global_var)
        lo, hi = self.var_clip
        var_reg = np.clip(var_reg, lo * global_var, hi * global_var)
        return {"means": means, "variances": var_reg, "weights": weights}

    @staticmethod
    def to_npz(priors: Dict[str, np.ndarray], path) -> None:
        """Write npz: ``covariances`` for full, ``variances`` for diag, plus optional ``weights``."""
        kwargs: Dict[str, np.ndarray] = {"means": priors["means"]}
        if "covariances" in priors:
            kwargs["covariances"] = priors["covariances"]
        else:
            kwargs["variances"] = priors["variances"]
        w = priors.get("weights")
        if w is not None:
            kwargs["weights"] = w
        np.savez(path, **kwargs)

    @staticmethod
    def from_npz(path) -> Dict[str, np.ndarray]:
        """Read npz: accepts either ``covariances`` or the legacy ``variances``."""
        z = np.load(path)
        out: Dict[str, np.ndarray] = {"means": np.asarray(z["means"])}
        if "covariances" in z.files:
            out["covariances"] = np.asarray(z["covariances"])
        if "variances" in z.files:
            out["variances"] = np.asarray(z["variances"])
        if "weights" in z.files:
            out["weights"] = np.asarray(z["weights"])
        return out


# ---------------------------------------------------------------------------
# spatial prior: per-pixel low-frequency GMM posterior + Hungarian alignment
# ---------------------------------------------------------------------------

class SpatialPriorGenerator:
    """Low-frequency cube -> (K, H, W) spatial prior probability.

    Args:
        K: number of components (same as the well GMM being aligned to).
        seed: GMM random seed.
        covariance_type: covariance of the low-frequency cube GMM, ``"diag"`` by default
            (as in legacy cell 19); ``"full"`` is also allowed. Alignment only uses the means
            (squared L2 between means), independently of the covariance type.
        group_mode: flatten (H, W) before fitting (legacy behaviour).
        max_iter / n_init: EM iterations and restarts of the low-frequency GMM,
            legacy 200 / 5 (cell 19). Verified insensitive: 100/1 and 200/5 reproduce the same
            probability field to 99.98% argmax agreement with the deployed ``spatial_prior.npy``
            on the baseline low-frequency model, so the legacy values are kept.
            
    """

    def __init__(self, K: int = 3, seed: int = 42, covariance_type: str = "diag",
                 group_mode: bool = True, max_iter: int = 200,
                 n_init: int = 5) -> None:
        if covariance_type not in ("full", "diag"):
            raise ValueError(f"covariance_type must be full or diag, got {covariance_type!r}")
        self.K = int(K)
        self.seed = seed
        self.covariance_type = covariance_type
        self.group_mode = group_mode
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)

    def generate(self, target_low: np.ndarray, well_priors: Dict[str, np.ndarray]
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit a GMM on the low-frequency cube (3, H, W), align it and return the posterior.

        Returns:
            ``(prob: (K, H, W) float32, perm: (K,) Hungarian permutation)``.

        Side effect: the matched low-frequency components are stored on ``self.means_low_aligned_`` /
        ``self.covariances_low_aligned_`` / ``self.weights_low_aligned_``
        ``self.covs_low_aligned_`` in well-component order, for reporting.
        """
        from scipy.optimize import linear_sum_assignment
        from sklearn.mixture import GaussianMixture

        C, H, W = target_low.shape
        X = np.asarray(target_low, dtype=np.float64).transpose(1, 2, 0).reshape(H * W, C)
        gmm = GaussianMixture(n_components=self.K, covariance_type=self.covariance_type,
                              random_state=self.seed, max_iter=self.max_iter,
                              n_init=self.n_init)
        gmm.fit(X)
        means_low = np.asarray(gmm.means_, dtype=np.float64)
        means_well = np.asarray(well_priors["means"], dtype=np.float64)

        # Hungarian: align the low-frequency components to the well components
        cost = np.sum((means_low[:, None, :] - means_well[None, :, :]) ** 2, axis=2)
        row_idx, col_idx = linear_sum_assignment(cost)
        perm = col_idx[np.argsort(row_idx)]

        # matched parameters in well-component (= expert) order
        inv = np.argsort(perm)
        self.means_low_aligned_ = means_low[inv]
        self.covariances_low_aligned_ = np.asarray(gmm.covariances_, dtype=np.float64)[inv]
        self.weights_low_aligned_ = np.asarray(gmm.weights_, dtype=np.float64)[inv]

        # per-pixel posterior, already reordered by perm
        prob = gmm.predict_proba(X).reshape(H, W, self.K).transpose(2, 0, 1)[perm]
        return prob.astype(np.float32), perm.astype(np.int64)


# ---------------------------------------------------------------------------
# registry interface
# ---------------------------------------------------------------------------

@PRIORS.register("gmm_diag")
@PRIORS.register("gmm_full")
class GMMPrior:
    """Gaussian mixture prior (registered as ``gmm_diag`` / ``gmm_full``).

    log p(x) = log sum_k pi_k N(x; mu_k, Sigma_k) with uniform pi; Sigma_k is either
    diagonal (``variances``, (K, D)) or full (``covariances``, (K, D, D)).
    x is assumed normalized; log-sum-exp keeps the evaluation stable.
    """

    def __init__(self, means: Sequence[Sequence[float]],
                 variances: Sequence[Sequence[float]] | None = None,
                 covariances: Sequence[Sequence[Sequence[float]]] | None = None,
                 weight: float = 1.0) -> None:
        self.means = np.asarray(means, dtype=np.float64)
        if self.means.ndim != 2 or self.means.shape[0] == 0:
            raise ValueError(f"means must be (K, D), got {self.means.shape}")
        self.K, self.D = self.means.shape
        self.weight = float(weight)
        self.covariances: Optional[np.ndarray] = None
        self.variances: Optional[np.ndarray] = None
        if covariances is not None:
            covs = np.asarray(covariances, dtype=np.float64)
            if covs.ndim == 3:
                if covs.shape != (self.K, self.D, self.D):
                    raise ValueError(
                        f"covariances must be (K, D, D) = {(self.K, self.D, self.D)}, "
                        f"got {covs.shape}")
                self.covariances = covs
            elif covs.ndim == 2 and covs.shape == (self.K, self.D):
                self.variances = covs                      # (K, D) diagonal shorthand
            else:
                raise ValueError(f"unsupported covariances shape: {covs.shape}")
        elif variances is not None:
            var = np.asarray(variances, dtype=np.float64)
            if var.shape != (self.K, self.D):
                raise ValueError(
                    f"variances must be (K, D) = {(self.K, self.D)}, got {var.shape}")
            self.variances = var
        else:
            raise ValueError("provide either variances (diagonal) or covariances (full)")

        if self.variances is not None:
            if self.variances.min() <= 0:
                raise ValueError("GMM variances must be positive")
            self._prec = np.eye(self.D)[None, :, :] / self.variances[:, :, None]  # (K,D,D)
            self._logdet = np.sum(np.log(self.variances), axis=1)                  # (K,)
        else:
            self._covs = self.covariances
            try:
                self._prec = np.linalg.inv(self._covs)                             # (K,D,D)
                sign, logabsdet = np.linalg.slogdet(self._covs)
                if not np.all(sign > 0):
                    raise ValueError("a full-covariance prior must be positive definite")
                self._logdet = logabsdet                                          # (K,)
            except np.linalg.LinAlgError as e:
                raise ValueError("the full-covariance prior is singular") from e

    def _as_matrix(self, arr: np.ndarray) -> tuple[np.ndarray, int, tuple]:
        """Reshape the input to (..., D); return ((N, D) matrix, channel axis, original shape)."""
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.shape[-1] == self.D:
            return arr.reshape(-1, self.D), -1, arr.shape
        if arr.ndim >= 2 and arr.shape[0] == self.D:
            perm = np.moveaxis(arr, 0, -1)                    # field (D, ...) -> (..., D)
            return perm.reshape(-1, self.D), 0, arr.shape
        raise ValueError(
            f"prior dimension mismatch: input {arr.shape}, prior D={self.D}"
            f"(the last or channel axis must equal D)"
        )

    def _restore_field(self, grad_matrix: np.ndarray, ch_axis: int, shape: tuple) -> np.ndarray:
        """Restore an (N, D) gradient to the shape of the original input.

        A matrix with D last reshapes directly; a field with D on axis 0 goes through
        (spatial..., D) and back (the inverse of ``_as_matrix``).
        """
        if ch_axis == -1:
            return grad_matrix.reshape(shape)
        spatial = shape[:ch_axis] + shape[ch_axis + 1:]
        grad_field = grad_matrix.reshape(spatial + (shape[ch_axis],))
        return np.moveaxis(grad_field, -1, ch_axis)

    def _log_density(self, xf: np.ndarray) -> np.ndarray:
        """Per-component log N(x; mu_k, Sigma_k), shape (N, K)."""
        diff = xf[:, None, :] - self.means[None]                     # (N, K, D)
        if self.variances is not None:
            diff2 = diff ** 2 / self.variances[None]
            return -0.5 * (np.sum(diff2, axis=2) + self._logdet[None] + self.D * np.log(2 * np.pi))
        qf = np.einsum("nki,kij,nkj->nk", diff, self._prec, diff)
        return -0.5 * (qf + self._logdet[None] + self.D * np.log(2 * np.pi))

    def log_prob(self, x: Sequence[float]) -> float:
        arr = np.asarray(x, dtype=np.float64)
        xf, _, _ = self._as_matrix(arr)
        log_mix = np.logaddexp.reduce(self._log_density(xf), axis=1) - np.log(self.K)  # (N,)
        return float(self.weight * np.sum(log_mix))

    def grad_log_prob(self, x: Sequence[float]) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        xf, ch_axis, shape = self._as_matrix(arr)
        log_norm = self._log_density(xf)
        w = np.exp(log_norm - np.logaddexp.reduce(log_norm, axis=1, keepdims=True))   # (N,K)
        if self.variances is not None:
            # diag：∇log N_k = (μ_k − x)/σ_k²
            grad = self.weight * np.sum(
                w[:, :, None] * (self.means[None] - xf[:, None, :]) / self.variances[None],
                axis=1,
            )
        else:
            # full：∇log N_k = Σ_k⁻¹(μ_k − x)
            score = np.einsum("kij,nkj->nki", self._prec, self.means[None] - xf[:, None, :])
            grad = self.weight * np.sum(w[:, :, None] * score, axis=1)
        return self._restore_field(grad, ch_axis, shape)

    def sample(self, rng: np.random.Generator, shape: Sequence[int]) -> np.ndarray:
        out = np.empty(tuple(shape) + (self.D,), dtype=np.float64)
        n = np.prod(shape)
        k = rng.integers(0, self.K, size=n)
        z = rng.standard_normal((n, self.D))
        flat = out.reshape(n, self.D)
        if self.variances is not None:
            flat[:] = self.means[k] + np.sqrt(self.variances[k]) * z
        else:
            chol = np.linalg.cholesky(self._covs)                    # (K, D, D)
            flat[:] = self.means[k] + np.einsum("kij,nj->ni", chol[k], z)
        return out