"""
gmm_prior.py

本脚本用于：
1. 从测井数据（Vp, Vs, Den）拟合 GMM，得到每个专家（岩相）的先验均值和对角/全协方差。
2. 对低频模型（平滑后的弹性参数）拟合另一个 GMM，并通过匈牙利算法匹配测井 GMM 的分量顺序。
3. 计算每个像素的空间先验概率（后验概率），保存为 .pt 文件，供地震反演使用。
4. 提供多种可视化：散点图矩阵、概率密度曲线、协方差热图/椭圆、BIC 曲线、空间先验图。

使用方法：
    python gmm_prior.py
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm, chi2
from matplotlib.patches import Ellipse
from typing import Tuple, Optional


# ==================== 全局配置（可修改） ====================
class GMMConfig:
    # ----- 数据路径 -----
    vp_path = './data/Marmous2/Vp.npy'
    vs_path = './data/Marmous2/Vs.npy'
    den_path = './data/Marmous2/Den.npy'
    
    # 输出文件路径
    output_dir = './'
    gmm_prior_file = os.path.join(output_dir, 'gmm_priors.npz')      # 对角协方差
    gmm_full_file = os.path.join(output_dir, 'gmm_priors_full.npz')  # 全协方差
    spatial_prior_file = os.path.join(output_dir, 'spatial_prior.pt')
    
    # ----- GMM 参数 -----
    n_components = 3           # 专家数（岩相数）
    covariance_type = 'diag'   # 'diag' 或 'full'
    random_state = 42
    max_iter = 500
    reg_covar = 1e-6            # 协方差正则化（仅 full）
    shrinkage = 0.1             # 收缩系数（针对对角方差）
    eps = 1e-4                  # 对角扰动
    
    # ----- 低频模型参数 -----
    sigma_low = 30              # 高斯平滑核大小
    train_start, train_end = 20, 650   # 训练道索引范围（用于标准化）
    num_train = 12               # 用于标准化的道数
    
    # ----- 可视化开关 -----
    do_visualization = True


# ==================== 数据加载与预处理 ====================
def load_elastic_data(cfg: GMMConfig):
    """加载 Vp, Vs, Den 原始数据，并裁剪为合适大小"""
    VP = np.load(cfg.vp_path)
    VS = np.load(cfg.vs_path)
    DEN = np.load(cfg.den_path)
    
    # 注：原 notebook 中的索引 (401::2, 1::10) 可能因数据维度不同需调整
    Vp = VP[401::2, 1::10]      # 形状 (H, W)
    Vs = VS[401::2, 1::10]
    Den = DEN[401::2, 1::10]
    
    target = np.stack([Vp, Vs, Den], axis=0)   # (3, H, W)
    return target, (Vp.shape[0], Vp.shape[1])


def compute_lowfreq_model(target: np.ndarray, sigma: float) -> np.ndarray:
    """对弹性参数进行高斯平滑，得到低频模型"""
    low = gaussian_filter(target, sigma=sigma, axes=(1,2))
    return low


def normalize_training_data(target, lowfreq, cfg: GMMConfig):
    """利用训练道（部分水平位置）计算均值和标准差，进行标准化"""
    train_indices = np.linspace(cfg.train_start, cfg.train_end, cfg.num_train, dtype=int)
    target_train = target[:, :, train_indices]               # (3, H, num_train)
    lowfreq_train = lowfreq[:, :, train_indices]
    
    target_mean = np.mean(target_train, axis=(1,2), keepdims=True)
    target_std = np.std(target_train, axis=(1,2), keepdims=True) + 1e-8
    lowfreq_mean = np.mean(lowfreq_train, axis=(1,2), keepdims=True)
    lowfreq_std = np.std(lowfreq_train, axis=(1,2), keepdims=True) + 1e-8
    
    target_norm = (target - target_mean) / target_std
    lowfreq_norm = (lowfreq - lowfreq_mean) / lowfreq_std
    return target_norm, lowfreq_norm, (target_mean, target_std, lowfreq_mean, lowfreq_std)


# ==================== GMM 拟合（对角协方差 + 正则化） ====================
def fit_gmm_diag(X: np.ndarray, n_components: int, random_state=42) -> Tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """
    拟合对角协方差 GMM，并应用收缩正则化。
    返回: (gmm模型, 正则化后的均值, 正则化后的方差)
    """
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag',
                          random_state=random_state, max_iter=500)
    gmm.fit(X)
    labels = gmm.predict(X)
    means = gmm.means_          # (K, 3)
    vars_raw = gmm.covariances_ # (K, 3)
    
    # 收缩估计
    global_var = np.var(X, axis=0)   # (3,)
    shrinkage = 0.1
    eps = 1e-4
    vars_reg = shrinkage * global_var + (1 - shrinkage) * vars_raw
    vars_reg = vars_reg + eps * global_var
    vars_reg = np.clip(vars_reg, 0.01 * global_var, 10 * global_var)
    
    return gmm, means, vars_reg, labels


# ==================== GMM 拟合（全协方差 + 正则化） ====================
def fit_gmm_full(X: np.ndarray, n_components: int, reg_covar=1e-6, shrinkage=0.1, eps=1e-4):
    """
    拟合全协方差 GMM，并对协方差矩阵进行收缩 + 对角扰动。
    返回: (gmm模型, 正则化后的均值, 正则化后的协方差矩阵)
    """
    gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                          random_state=42, max_iter=500, reg_covar=reg_covar)
    gmm.fit(X)
    means = gmm.means_                     # (K, 3)
    covs = gmm.covariances_                # (K, 3, 3)
    global_cov = np.cov(X, rowvar=False)   # (3, 3)
    
    for k in range(n_components):
        cov_k = (1 - shrinkage) * covs[k] + shrinkage * global_cov
        cov_k += eps * np.eye(3)
        covs[k] = cov_k
    return gmm, means, covs


# ==================== 空间先验（低频 GMM 匹配） ====================
def match_gmm_components(means_ref: np.ndarray, means_target: np.ndarray) -> np.ndarray:
    """
    匈牙利算法匹配两个 GMM 的分量顺序。
    返回 perm: perm[target_idx] = ref_idx，表示 target 的第 target_idx 个分量应映射到 ref 的第 ref_idx 个分量。
    """
    K = len(means_ref)
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = np.linalg.norm(means_ref[i] - means_target[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.zeros(K, dtype=int)
    for ref_idx, tgt_idx in zip(row_ind, col_ind):
        perm[tgt_idx] = ref_idx
    return perm


def apply_permutation_to_gmm(gmm: GaussianMixture, perm: np.ndarray):
    """根据 perm 重排 GMM 的均值、协方差和权重"""
    means_orig = gmm.means_
    covs_orig = gmm.covariances_
    weights_orig = gmm.weights_
    K = len(perm)
    means_aligned = np.zeros_like(means_orig)
    covs_aligned = np.zeros_like(covs_orig)
    weights_aligned = np.zeros_like(weights_orig)
    for orig_idx, new_idx in enumerate(perm):
        means_aligned[new_idx] = means_orig[orig_idx]
        covs_aligned[new_idx] = covs_orig[orig_idx]
        weights_aligned[new_idx] = weights_orig[orig_idx]
    return means_aligned, covs_aligned, weights_aligned


def compute_spatial_prior(lowfreq_cube: np.ndarray, means_ref: np.ndarray,
                          n_components: int, covariance_type='diag',
                          reg_covar=1e-6, random_state=42,
                          save_path: Optional[str] = None) -> np.ndarray:
    """
    对低频模型拟合 GMM，匹配测井 GMM 顺序，返回每个像素的后验概率 (H, W, K)。
    """
    H, W, _ = lowfreq_cube.shape
    data_flat = lowfreq_cube.reshape(-1, 3)
    
    # 拟合低频 GMM（与测井 GMM 相同的协方差类型）
    gmm_low = GaussianMixture(n_components=n_components,
                              covariance_type=covariance_type,
                              random_state=random_state,
                              max_iter=200,
                              n_init=5,
                              reg_covar=reg_covar)
    gmm_low.fit(data_flat)
    
    # 匹配顺序
    perm = match_gmm_components(means_ref, gmm_low.means_)
    means_low, covs_low, weights_low = apply_permutation_to_gmm(gmm_low, perm)
    
    # 计算每个像素的后验概率
    if covariance_type == 'diag':
        # 对角协方差
        log_probs = np.zeros((data_flat.shape[0], n_components))
        for k in range(n_components):
            mu = means_low[k]
            var = covs_low[k]
            diff = data_flat - mu
            log_det = np.sum(np.log(var))
            mahal = np.sum((diff**2) / var, axis=1)
            log_probs[:, k] = -0.5 * (mahal + 3 * np.log(2*np.pi) + log_det)
    else:  # full
        log_probs = np.zeros((data_flat.shape[0], n_components))
        for k in range(n_components):
            mu = means_low[k]
            cov = covs_low[k]
            diff = data_flat - mu
            # 求解线性方程组，计算马氏距离平方
            try:
                L = np.linalg.cholesky(cov)
                solve = np.linalg.solve(L, diff.T).T
                mahal = np.sum(solve**2, axis=1)
            except np.linalg.LinAlgError:
                # 若协方差接近奇异，使用伪逆
                inv_cov = np.linalg.pinv(cov)
                mahal = np.sum(diff @ inv_cov * diff, axis=1)
            log_det = np.linalg.slogdet(cov)[1]
            log_probs[:, k] = -0.5 * (mahal + 3 * np.log(2*np.pi) + log_det)
    
    # 加入 log(weights) 并 softmax
    log_weights = np.log(weights_low + 1e-12)
    log_joint = log_probs + log_weights[None, :]
    max_log = np.max(log_joint, axis=1, keepdims=True)
    exp_val = np.exp(log_joint - max_log)
    probs_flat = exp_val / np.sum(exp_val, axis=1, keepdims=True)
    prior_probs = probs_flat.reshape(H, W, n_components)
    
    if save_path is not None:
        if save_path.endswith('.pt'):
            torch.save(torch.from_numpy(prior_probs), save_path)
        else:
            np.save(save_path, prior_probs)
    return prior_probs


# ==================== 可视化函数 ====================
def plot_pairplot_scatter(X, labels, best_k, feature_names=['Vp', 'Vs', 'ρ'], max_points=5000, random_state=42):
    """
    散点图矩阵，按专家着色。自动下采样到 max_points 个点。
    """
    n_samples = X.shape[0]
    if n_samples > max_points:
        # 随机下采样，保持各个专家的比例（因为 random 采样自然保持比例）
        np.random.seed(random_state)
        indices = np.random.choice(n_samples, max_points, replace=False)
        X_sample = X[indices]
        labels_sample = labels[indices]
    else:
        X_sample = X
        labels_sample = labels

    df = pd.DataFrame(X_sample, columns=feature_names)
    df['Lithofacies'] = [f'Expert {l+1}' for l in labels_sample]
    palette = 'tab10' if best_k <= 10 else 'hsv'
    g = sns.pairplot(df, hue='Lithofacies', palette=palette, diag_kind='hist',
                     plot_kws={'s': 2, 'alpha': 0.5, 'edgecolor': 'none'},
                     diag_kws={'alpha': 0.7, 'edgecolor': 'white'})
    # 美化
    for ax in g.axes.flat:
        if ax is not None:
            ax.grid(True, linestyle='--', alpha=0.3)
    # 图例
    handles = g._legend.legend_handles
    labels_leg = [f'Expert {i+1}' for i in range(best_k)]
    g._legend.remove()
    plt.legend(handles, labels_leg, title='Expert Types', ncol=best_k,
               loc='lower center', bbox_to_anchor=(0.5, 1.02),
               frameon=True, shadow=False)
    plt.suptitle(f'Scatter Matrix (K={best_k}, subsampled to {len(X_sample)} points)', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

def plot_expert_pdf(X, labels, prior_vars=None, feature_names=['Vp', 'Vs', 'ρ'],
                    best_k=3, bins=50):
    """每个专家每个特征的直方图 + 高斯拟合曲线"""
    fig, axes = plt.subplots(best_k, 3, figsize=(13, 3.5*best_k))
    if best_k == 1:
        axes = axes.reshape(1, -1)
    colors = plt.cm.tab10(np.linspace(0, 1, best_k))
    for k in range(best_k):
        X_k = X[labels == k]
        for j, feat in enumerate(feature_names):
            ax = axes[k, j]
            data = X_k[:, j]
            ax.hist(data, bins=bins, density=True, alpha=0.6, color=colors[k])
            mu, sigma = np.mean(data), np.std(data)
            x_range = np.linspace(data.min(), data.max(), 500)
            ax.plot(x_range, norm.pdf(x_range, mu, sigma), 'k-', lw=1.5, label='Sample')
            if prior_vars is not None:
                sigma_p = np.sqrt(prior_vars[k, j])
                ax.plot(x_range, norm.pdf(x_range, mu, sigma_p), 'r--', lw=2, label='Prior')
            if j == 0:
                ax.set_ylabel(f'Expert {k+1}', fontweight='bold')
            if k == 0:
                ax.set_title(feat, fontweight='bold')
            ax.grid(alpha=0.3)
            if j == 2:
                ax.legend()
    plt.tight_layout()
    plt.show()


def plot_covariance_ellipses(means, covariances, feature_names=['Vp', 'Vs', 'Den']):
    """绘制全协方差矩阵的置信椭圆"""
    K = means.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [(0,1), (0,2), (1,2)]
    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        for k in range(K):
            mu_ij = means[k][[i, j]]
            cov_ij = covariances[k][[i, j]][:, [i, j]]
            eigvals, eigvecs = np.linalg.eigh(cov_ij)
            angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
            chi2_val = chi2.ppf(0.95, df=2)
            width = 2 * np.sqrt(eigvals[1] * chi2_val)
            height = 2 * np.sqrt(eigvals[0] * chi2_val)
            ellipse = Ellipse(xy=mu_ij, width=width, height=height, angle=angle,
                              edgecolor=plt.cm.tab10(k), facecolor='none', lw=2)
            ax.add_patch(ellipse)
            ax.scatter(mu_ij[0], mu_ij[1], color=plt.cm.tab10(k), s=50)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_spatial_prior(prior_probs: np.ndarray, save_fig=True):
    """显示空间先验概率图（硬分类 + 前三个专家的概率）"""
    H, W, K = prior_probs.shape
    num_show = min(3, K)
    fig, axes = plt.subplots(1, num_show+1, figsize=(16, 5))
    # 硬分类
    hard = np.argmax(prior_probs, axis=2)
    im0 = axes[0].imshow(hard, cmap='tab20', vmin=0, vmax=K-1, interpolation='none')
    axes[0].set_title('Dominant Expert')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], ticks=range(K))
    # 各专家概率
    for i in range(num_show):
        im = axes[i+1].imshow(prior_probs[:, :, i], cmap='plasma', vmin=0, vmax=1)
        axes[i+1].set_title(f'Expert {i+1} Probability')
        axes[i+1].axis('off')
        plt.colorbar(im, ax=axes[i+1])
    plt.tight_layout()
    if save_fig:
        plt.savefig('spatial_prior.png', dpi=300)
    plt.show()


# ==================== 主程序 ====================
def main():
    cfg = GMMConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # 1. 加载数据
    target, (H, W) = load_elastic_data(cfg)
    lowfreq = compute_lowfreq_model(target, cfg.sigma_low)
    target_norm, lowfreq_norm, _ = normalize_training_data(target, lowfreq, cfg)
    
    # 2. 准备测井训练数据（所有像素的空间点）
    X = target_norm.reshape(3, -1).T   # (N, 3)
    print(f"Total samples: {X.shape[0]}")
    
    # 3. 拟合对角协方差 GMM（带正则化）
    print("Fitting diagonal GMM ...")
    gmm_diag, means_diag, vars_diag, labels = fit_gmm_diag(X, cfg.n_components, cfg.random_state)
    np.savez(cfg.gmm_prior_file, means=means_diag, variances=vars_diag)
    print(f"Saved diagonal prior to {cfg.gmm_prior_file}")
    
    # 4. 拟合全协方差 GMM（可选，主要用于可视化）
    if cfg.covariance_type == 'full':
        print("Fitting full-covariance GMM ...")
        gmm_full, means_full, covs_full = fit_gmm_full(X, cfg.n_components, cfg.reg_covar,
                                                        cfg.shrinkage, cfg.eps)
        np.savez(cfg.gmm_full_file, means=means_full, covariances=covs_full,
                 scaler_mean=None, scaler_scale=None)
        print(f"Saved full prior to {cfg.gmm_full_file}")
    else:
        means_full = means_diag
        covs_full = np.array([np.diag(v) for v in vars_diag])
    
    # 5. 计算空间先验（基于低频模型）
    print("Computing spatial prior from low-frequency model ...")
    lowfreq_cube = lowfreq_norm.transpose(1,2,0)   # (H, W, 3)
    spatial_prior = compute_spatial_prior(
        lowfreq_cube=lowfreq_cube,
        means_ref=means_diag,
        n_components=cfg.n_components,
        covariance_type='diag',
        reg_covar=cfg.reg_covar,
        random_state=cfg.random_state,
        save_path=cfg.spatial_prior_file
    )
    print(f"Spatial prior shape: {spatial_prior.shape}, saved to {cfg.spatial_prior_file}")
    
    # 6. 可视化（可选）
    if cfg.do_visualization:
        # 散点图矩阵
        plot_pairplot_scatter(X, labels, cfg.n_components)
        # 概率密度曲线
        plot_expert_pdf(X, labels, prior_vars=vars_diag, best_k=cfg.n_components)
        # 协方差椭圆（全协方差）
        plot_covariance_ellipses(means_full, covs_full)
        # 空间先验图
        plot_spatial_prior(spatial_prior)
    
    print("Done.")


if __name__ == "__main__":
    main()