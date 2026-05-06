import numpy as np
import matplotlib.pyplot as plt

def plot_section_prediction(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None,
                            param_name: str = "Vp", vmin=None, vmax=None):
    """
    展示整个剖面的预测、真实、残差。
    pred, true: (3, H, W) 或 (H, W) 单参数
    """
    if pred.ndim == 3:
        pred = pred[0]  # 取 Vp 通道
        true = true[0]
    H, W = pred.shape

    if vmin is None:
        vmin = min(pred.min(), true.min())
    if vmax is None:
        vmax = max(pred.max(), true.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # 预测
    im0 = axes[0].imshow(pred, cmap='jet', vmin=-2, vmax=2,
                         extent=(0, W, H, 0), aspect='auto')
    axes[0].set_title(f'Predicted {param_name}')
    axes[0].set_xlabel('Trace')
    axes[0].set_ylabel('Sample')
    plt.colorbar(im0, ax=axes[0])

    # 真实
    im1 = axes[1].imshow(true, cmap='jet', vmin=-2, vmax=2,
                         extent=(0, W, H, 0), aspect='auto')
    axes[1].set_title(f'True {param_name}')
    axes[1].set_xlabel('Trace')
    axes[1].set_ylabel('Sample')
    plt.colorbar(im1, ax=axes[1])

    # 残差
    diff = np.abs(pred - true)
    im2 = axes[2].imshow(diff, cmap='gray', vmin=0, vmax=np.percentile(diff, 95),
                         extent=(0, W, H, 0), aspect='auto')
    axes[2].set_title('Absolute Difference')
    axes[2].set_xlabel('Trace')
    axes[2].set_ylabel('Sample')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()


def plot_expert_partition(z_soft: np.ndarray, struct_attrs: np.ndarray = None,
                          dip_channel: int = 0, coherence_channel: int = 1):
    """
    展示硬分区（最大概率）和软分区概率图。
    z_soft: (K, H, W)
    struct_attrs: (C_attr, H, W) 可选的属性，用于对比。
    """
    K, H, W = z_soft.shape
    z_hard = np.argmax(z_soft, axis=0)

    fig, axes = plt.subplots(1, K+1, figsize=(4*(K+1), 4))

    # 硬分区图
    cmap = plt.cm.get_cmap('tab10', K)
    im0 = axes[0].imshow(z_hard, cmap=cmap, vmin=0, vmax=K-1,
                         extent=(0, W, H, 0), aspect='auto')
    axes[0].set_title('Hard Partition (argmax)')
    axes[0].set_xlabel('Trace')
    axes[0].set_ylabel('Sample')
    plt.colorbar(im0, ax=axes[0], ticks=range(K))

    # 各专家软分区概率
    for k in range(K):
        im = axes[k+1].imshow(z_soft[k], cmap='viridis', vmin=0, vmax=1,
                              extent=(0, W, H, 0), aspect='auto')
        axes[k+1].set_title(f'Expert {k} probability')
        axes[k+1].set_xlabel('Trace')
        axes[k+1].set_ylabel('Sample')
        plt.colorbar(im, ax=axes[k+1])

    plt.tight_layout()
    plt.show()

    # 可选：叠合结构属性（例如 dip）
    if struct_attrs is not None:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        # dip
        dip = struct_attrs[dip_channel]
        im_dip = axes2[0].imshow(dip, cmap='jet', extent=(0, W, H, 0), aspect='auto')
        axes2[0].set_title('Dip')
        plt.colorbar(im_dip, ax=axes2[0])
        # coherence
        coh = struct_attrs[coherence_channel]
        im_coh = axes2[1].imshow(coh, cmap='gray', extent=(0, W, H, 0), aspect='auto')
        axes2[1].set_title('Coherence')
        plt.colorbar(im_coh, ax=axes2[1])
        plt.tight_layout()
        plt.show()


def plot_trace_with_uncertainty(mu_section, logvar_section, true_section, trace_id, param_name='Vp'):
    """
    绘制指定参数在某个 trace 上的深度剖面，包含均值和不确定性。

    参数:
        mu_section : ndarray, shape (3, L, N)
            参数的均值，第一维顺序为 [Vp, Vs, density]
        logvar_section : ndarray, shape (3, L, N)
            参数的对数方差
        true_section : ndarray, shape (3, L, N)
            真实参数值
        trace_id : int
            要绘制的水平位置索引 (0 <= trace_id < N)
        param_name : str, 可选, 默认 'Vp'
            要绘制的参数名称，支持 'Vp', 'Vs', 'den' (不区分大小写)
    """
    # 参数名到索引的映射
    param_map = {'vp': 0, 'vs': 1, 'den': 2}
    param_idx = param_map.get(param_name.lower(), 0)
    
    # 提取数据：形状均为 (L,)
    mu = mu_section[param_idx, :, trace_id]
    logvar = logvar_section[param_idx, :, trace_id]
    true = true_section[param_idx, :, trace_id]
    
    # 计算标准差 (σ = exp(logvar/2))
    std = np.exp(logvar / 2)
    
    # 深度轴（使用索引，可替换为实际深度值）
    depth = np.arange(len(mu))
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(depth, mu, 'b-', label='Mean', linewidth=2)
    plt.fill_between(depth, mu - 2*std, mu + 2*std, color='b', alpha=0.3, label='±2σ')
    plt.plot(depth, true, 'r--', label='True', linewidth=2)
    
    plt.xlabel('Depth Index')
    plt.ylabel(param_name.capitalize())
    plt.title(f'{param_name.capitalize()} at Trace {trace_id} with Uncertainty')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_trace_marginal_and_experts(
    z_soft_section,      # (num_experts, L, N)
    expert_means,        # (num_experts, 3, L, N)
    expert_logvars,      # (num_experts, 3, L, N)
    trace_id,            # 要绘制的道索引 (0 <= trace_id < N)
    param_names=['Vp', 'Vs', 'Den'],
    param_for_marginal='Vp',   # 左侧边际分布展示哪个参数
    depth=None,                # 深度数组，默认为索引
    true_section=None,         # 可选真实参数 (3, L, N)
    figsize=(16, 6)
):
    """
    绘制某一道的边际分布和专家分解。

    参数:
        z_soft_section   : (K, L, N) 专家分配概率
        expert_means     : (K, 3, L, N) 专家均值
        expert_logvars   : (K, 3, L, N) 专家对数方差
        trace_id         : 道索引
        param_names      : 参数名称列表，长度3
        param_for_marginal: 左侧边际展示的参数名（必须在 param_names 中）
        depth            : 深度坐标 (L,)，默认为索引
        true_section     : 真实参数 (3, L, N)，可选
        figsize          : 图形大小
    """
    K, L, N = z_soft_section.shape
    assert expert_means.shape == (K, 3, L, N)
    assert expert_logvars.shape == (K, 3, L, N)

    # 确定边际参数索引
    try:
        p_idx_marginal = param_names.index(param_for_marginal)
    except ValueError:
        raise ValueError(f"param_for_marginal '{param_for_marginal}' 不在 {param_names} 中")

    # 提取当前道的数据
    z_trace = z_soft_section[:, :, trace_id]           # (K, L)
    mu_trace = expert_means[:, :, :, trace_id]         # (K, 3, L)
    logvar_trace = expert_logvars[:, :, :, trace_id]   # (K, 3, L)

    # 深度坐标
    if depth is None:
        depth = np.arange(L)
    else:
        assert len(depth) == L

    # 真实值（可选）
    true_trace = None
    if true_section is not None:
        true_trace = true_section[:, :, trace_id]      # (3, L)

    # 创建子图：1行4列
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=True)

    # ---------- 左侧子图：边际分布（均值 ± 2σ） ----------
    ax_left = axes[0]
    # 计算边际均值和方差
    mu_marginal = np.sum(z_trace * mu_trace[:, p_idx_marginal, :], axis=0)  # (L,)
    # 二阶矩：E[θ^2] = Σ z_k * (σ_k^2 + μ_k^2)
    var_k = np.exp(logvar_trace[:, p_idx_marginal, :])  # (K, L)
    second_moment = np.sum(z_trace * (var_k + mu_trace[:, p_idx_marginal, :]**2), axis=0)
    var_marginal = second_moment - mu_marginal**2
    std_marginal = np.sqrt(np.maximum(var_marginal, 1e-12))

    # 绘图
    ax_left.plot(depth, mu_marginal, 'b-', linewidth=2, label='Mean')
    ax_left.fill_between(depth, mu_marginal - 2*std_marginal,
                         mu_marginal + 2*std_marginal,
                         color='b', alpha=0.3, label='±2σ')
    if true_trace is not None:
        ax_left.plot(depth, true_trace[p_idx_marginal], 'r--', linewidth=2, label='True')
    ax_left.set_title(f'Marginal distribution\n{param_for_marginal}')
    ax_left.set_xlabel('Depth')
    ax_left.set_ylabel(param_for_marginal)
    ax_left.legend(loc='upper right')
    ax_left.grid(True, linestyle=':', alpha=0.6)

    # ---------- 右侧三个子图：每个参数的专家均值曲线 ----------
    for p_idx, p_name in enumerate(param_names):
        ax = axes[p_idx + 1]
        for k in range(K):
            # 专家 k 在当前参数上的均值曲线
            mu_k = mu_trace[k, p_idx, :]   # (L,)
            # 可选：用 z_soft 的均值作为透明度或线宽，这里仅用颜色区分
            ax.plot(depth, mu_k, label=f'Expert {k}', linewidth=1.5, alpha=0.7)
        # 可选：叠加真实值
        if true_trace is not None:
            ax.plot(depth, true_trace[p_idx], 'k--', linewidth=2, label='True')
        ax.set_title(f'Expert means for {p_name}')
        ax.set_xlabel('Depth')
        ax.set_ylabel(p_name)
        ax.legend(loc='upper right', fontsize='small', ncol=1)
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()




    
