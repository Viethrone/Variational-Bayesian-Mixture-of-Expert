"""GMM 参数先验与空间先验工具（按 legacy GMM.ipynb cell 13 的全协方差口径）。

两条管道：

1. 参数先验 ``GMMPriorFitter`` — 在归一化空间的 train 子集上拟合 GMM，默认
    全协方差（``covariance_type="full"``，与 GMM.ipynb cell 13 / 2D vbmile 一致），
    输出 ``means (K, 3), covariances (K, 3, 3), weights (K,)``；shrinkage 与
    ``eps*eye`` 沿用 cell 13 配方（**不裁剪**）。``covariance_type="diag"`` 保留
    legacy cell 12 的对角口径（shrinkage + var_clip），输出 ``variances (K, 3)``。
2. 空间先验 ``SpatialPriorGenerator`` — 低频模型 cube 上的逐像素 GMM 后验概率，
    并用 Hungarian 算法把组件排列对齐到井位 GMM，输出 ``(K, H, W)`` 概率图。
    （低频 cube 的 GMM 默认「对角协方差」，与 legacy cell 19 ``fit_lowfreq_gmm``
    一致；可通过 ``covariance_type`` 切换。）

``GMMPrior`` 把拟合结果暴露为骨架 ``BasePrior`` 接口（迭代反演路径用），
log_prob 取 Gauss 混合的 log-sum-exp，方差可为对角 (K, D) 或全协方差 (K, D, D)。
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
# 组件排列对齐（Hungarian）
# ---------------------------------------------------------------------------

def hungarian_match(means_a: np.ndarray, means_b: np.ndarray) -> np.ndarray:
    """最小化 ||means_a - means_b[perm]||_F 的排列。K 小，直接穷举。"""
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
# 参数先验：well-GMM 拟合
# ---------------------------------------------------------------------------

class GMMPriorFitter:
    """legacy GMM.ipynb cell 13 的 GMM 拟合口径（full covariance + shrinkage）。

    Args:
        K: 混合分量数。
        seed: sklearn GMM 随机种子（legacy 用 42）。
        covariance_type: ``"full"``（默认，与 GMM.ipynb cell 13 / 2D vbmile 一致）
            或 ``"diag"``（legacy cell 12 的对角替代口径）。
        shrinkage: 协方差收缩系数——归一化空间里全局协方差与分量协方差的插值
            （cell 13 用 0.1）。
        eps: 协方差下限保护，加到对角上（cell 13 用 1e-4；full 不裁剪）。
        var_clip: 仅 diag 模式生效——分量方差相对全局方差的裁剪上下限。
    """

    def __init__(self, K: int = 3, seed: int = 42, covariance_type: str = "full",
                 shrinkage: float = 0.1, eps: float = 1e-4,
                 var_clip: Tuple[float, float] = (0.01, 10.0)) -> None:
        if covariance_type not in ("full", "diag"):
            raise ValueError(f"covariance_type 只能是 full 或 diag，收到 {covariance_type!r}")
        self.K = int(K)
        self.seed = seed
        self.covariance_type = covariance_type
        self.shrinkage = float(shrinkage)
        self.eps = float(eps)
        self.var_clip = tuple(var_clip)

    def fit(self, X_norm: np.ndarray) -> Dict[str, np.ndarray]:
        """在归一化空间数据 (N, D) 上拟合 GMM（``reg_covar=1e-6``，同 cell 13）。

        Returns:
            full: ``{"means": (K, D), "covariances": (K, D, D), "weights": (K,)}``，
                  shrinkage 后 ``+= eps*eye(D)``，**不裁剪**。
            diag: ``{"means": (K, D), "variances": (K, D), "weights": (K,)}``，
                  沿用旧 shrinkage + var_clip 裁剪。
        """
        from sklearn.mixture import GaussianMixture

        X = np.asarray(X_norm, dtype=np.float64)
        assert X.ndim == 2, f"X 应为 (N, D)，实际 {X.shape}"
        D = X.shape[1]
        gmm = GaussianMixture(n_components=self.K, covariance_type=self.covariance_type,
                              random_state=self.seed, reg_covar=1e-6, max_iter=500)
        gmm.fit(X)
        means = np.asarray(gmm.means_, dtype=np.float64)                    # (K, D)
        weights = np.asarray(gmm.weights_, dtype=np.float64)                # (K,)
        global_cov = np.cov(X, rowvar=False, ddof=1)                        # (D, D)

        if self.covariance_type == "full":
            # cell 13 收缩配方：cov_k = (1-s)*Σk + s*Σglobal + eps*eye(D)，不裁剪
            cov_raw = np.asarray(gmm.covariances_, dtype=np.float64)        # (K, D, D)
            cov_reg = ((1 - self.shrinkage) * cov_raw + self.shrinkage * global_cov
                       + self.eps * np.eye(D))
            return {"means": means, "covariances": cov_reg, "weights": weights}

        # diag：旧口径（cell 12 的 shrink + clip）
        var_raw = np.asarray(gmm.covariances_, dtype=np.float64)            # (K, D)
        global_var = np.diag(global_cov)                                    # (D,)
        var_reg = (self.shrinkage * global_var
                   + (1 - self.shrinkage) * var_raw + self.eps * global_var)
        lo, hi = self.var_clip
        var_reg = np.clip(var_reg, lo * global_var, hi * global_var)
        return {"means": means, "variances": var_reg, "weights": weights}

    @staticmethod
    def to_npz(priors: Dict[str, np.ndarray], path) -> None:
        """写 npz：full 存 `covariances`，diag 存 `variances`（均附可选 `weights`）。"""
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
        """读 npz：兼容 `covariances` 与旧 `variances` 两种格式。"""
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
# 空间先验：低频模型逐像素 GMM 后验 + Hungarian 对齐
# ---------------------------------------------------------------------------

class SpatialPriorGenerator:
    """低频模型 cube -> (K, H, W) 空间先验概率图。

    Args:
        K: 混合分量数（与被对齐的 well-GMM 相同）。
        seed: GMM 随机种子。
        covariance_type: 低频 cube GMM 的协方差类型，默认 ``"diag"``（与 legacy
            cell 19 ``fit_lowfreq_gmm`` 一致）；也可 ``"full"``。对齐仍只看均值
            （Hungarian 距离 = 均值间 squared L2），与协方差类型无关。
        group_mode: 是否先对 (H W) 展平后整体拟合（legacy 行为）。
        max_iter / n_init: 低频 cube GMM 的 EM 迭代上限与重启次数，默认沿用
            legacy ``fit_lowfreq_gmm``（cell 19）的 200 / 5。**已核实**：这两项
            取 100/1 或 200/5 时，基准低频模型（``target_low_s30``，归一化空间）
            复现出的概率场相对部署 ``spatial_prior.npy`` 的 argmax 一致率同为
            99.98%，故数值上不敏感；默认取 legacy 口径。
    """

    def __init__(self, K: int = 3, seed: int = 42, covariance_type: str = "diag",
                 group_mode: bool = True, max_iter: int = 200,
                 n_init: int = 5) -> None:
        if covariance_type not in ("full", "diag"):
            raise ValueError(f"covariance_type 只能是 full 或 diag，收到 {covariance_type!r}")
        self.K = int(K)
        self.seed = seed
        self.covariance_type = covariance_type
        self.group_mode = group_mode
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)

    def generate(self, target_low: np.ndarray, well_priors: Dict[str, np.ndarray]
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """拟合低频 cube (C=3, H, W) 的 GMM，对齐到 well-GMM，返回后验概率图。

        Returns:
            ``(prob: (K, H, W) float32, perm: (K,) Hungarian 排列)``。

        Side effect: 匹配后的低频簇参数挂到 ``self.means_low_aligned_`` /
        ``self.covariances_low_aligned_`` / ``self.weights_low_aligned_``
        （专家序 = well 分量序），便于调用方报告"哪个低频簇被标成哪个专家"。
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

        # Hungarian：把低频组件排列到 well 组件
        cost = np.sum((means_low[:, None, :] - means_well[None, :, :]) ** 2, axis=2)
        row_idx, col_idx = linear_sum_assignment(cost)
        perm = col_idx[np.argsort(row_idx)]

        # 匹配后的低频簇参数（专家序 = well 分量序），供调用方判读/落盘
        inv = np.argsort(perm)
        self.means_low_aligned_ = means_low[inv]
        self.covariances_low_aligned_ = np.asarray(gmm.covariances_, dtype=np.float64)[inv]
        self.weights_low_aligned_ = np.asarray(gmm.weights_, dtype=np.float64)[inv]

        # 逐像素后验（已按 perm 重排）
        prob = gmm.predict_proba(X).reshape(H, W, self.K).transpose(2, 0, 1)[perm]
        return prob.astype(np.float32), perm.astype(np.int64)


# ---------------------------------------------------------------------------
# 骨架接口：迭代反演路径
# ---------------------------------------------------------------------------

@PRIORS.register("gmm_diag")
@PRIORS.register("gmm_full")
class GMMPrior:
    """Gauss 混合先验（骨架 BasePrior 接口）。

    log p(x) = log Σ_k π_k · N(x; μ_k, Σ_k)；π 取均匀。Σ_k 可为对角
    （``variances``，(K, D)）或全协方差（``covariances``，(K, D, D)）。
    x 假定已在归一化空间。手段：用 log-sum-exp 保证数值稳定。
    """

    def __init__(self, means: Sequence[Sequence[float]],
                 variances: Sequence[Sequence[float]] | None = None,
                 covariances: Sequence[Sequence[Sequence[float]]] | None = None,
                 weight: float = 1.0) -> None:
        self.means = np.asarray(means, dtype=np.float64)
        if self.means.ndim != 2 or self.means.shape[0] == 0:
            raise ValueError(f"means 应为 (K, D)：{self.means.shape}")
        self.K, self.D = self.means.shape
        self.weight = float(weight)
        self.covariances: Optional[np.ndarray] = None
        self.variances: Optional[np.ndarray] = None
        if covariances is not None:
            covs = np.asarray(covariances, dtype=np.float64)
            if covs.ndim == 3:
                if covs.shape != (self.K, self.D, self.D):
                    raise ValueError(
                        f"covariances 应为 (K, D, D) = {(self.K, self.D, self.D)}，"
                        f"实际 {covs.shape}")
                self.covariances = covs
            elif covs.ndim == 2 and covs.shape == (self.K, self.D):
                self.variances = covs                      # (K, D) 对角简写
            else:
                raise ValueError(f"covariances 形状不支持：{covs.shape}")
        elif variances is not None:
            var = np.asarray(variances, dtype=np.float64)
            if var.shape != (self.K, self.D):
                raise ValueError(
                    f"variances 应为 (K, D) = {(self.K, self.D)}，实际 {var.shape}")
            self.variances = var
        else:
            raise ValueError("须提供 variances（对角）或 covariances（全协方差）")

        if self.variances is not None:
            if self.variances.min() <= 0:
                raise ValueError("GMM 方差必须为正")
            self._prec = np.eye(self.D)[None, :, :] / self.variances[:, :, None]  # (K,D,D)
            self._logdet = np.sum(np.log(self.variances), axis=1)                  # (K,)
        else:
            self._covs = self.covariances
            try:
                self._prec = np.linalg.inv(self._covs)                             # (K,D,D)
                sign, logabsdet = np.linalg.slogdet(self._covs)
                if not np.all(sign > 0):
                    raise ValueError("全协方差先验必须对称正定（slogdet 存在非正 sign）")
                self._logdet = logabsdet                                          # (K,)
            except np.linalg.LinAlgError as e:
                raise ValueError("全协方差先验不可逆——协方差矩阵奇异") from e

    def _as_matrix(self, arr: np.ndarray) -> tuple[np.ndarray, int, tuple]:
        """把输入整理为 (..., D) 且返回 (矩阵 (N,D), 通道轴位置, 原形状)。"""
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.shape[-1] == self.D:
            return arr.reshape(-1, self.D), -1, arr.shape
        if arr.ndim >= 2 and arr.shape[0] == self.D:
            perm = np.moveaxis(arr, 0, -1)                    # 场 (D, ...) -> (..., D)
            return perm.reshape(-1, self.D), 0, arr.shape
        raise ValueError(
            f"先验维度不匹配：输入形状 {arr.shape}，先验维 D={self.D}"
            f"（末维或通道轴须等于 D）"
        )

    def _restore_field(self, grad_matrix: np.ndarray, ch_axis: int, shape: tuple) -> np.ndarray:
        """把 (N,D) 梯度恢复为原输入形状。

        D 在末维的矩阵输入：直接 reshape；D 在 axis 0 的场输入：先重排为
        (spatial..., D) 再 moveaxis 回原通道轴（_as_matrix 的逆）。
        """
        if ch_axis == -1:
            return grad_matrix.reshape(shape)
        spatial = shape[:ch_axis] + shape[ch_axis + 1:]
        grad_field = grad_matrix.reshape(spatial + (shape[ch_axis],))
        return np.moveaxis(grad_field, -1, ch_axis)

    def _log_density(self, xf: np.ndarray) -> np.ndarray:
        """逐分量 log N(x; μ_k, Σ_k)，(N, K)。"""
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