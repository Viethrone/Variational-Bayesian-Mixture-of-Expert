import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

from config import Config2D
from data.dataset import SeismicDataset, Standardize
from models.forward import SeismicForward, ricker_wavelet
from models.VBMILE2d import GeoVBMILE2D
from models.VBlosses import VBELBO2D

def set_seed(seed, deterministic):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def load_data(cfg, device):
    VP = np.load(cfg.vp_path); VS = np.load(cfg.vs_path); DEN = np.load(cfg.den_path)
    mat_data = loadmat(cfg.seismic_mat_path); SYN = mat_data['seismic_data']

    SYN = mat_data['seismic_data']  

    Vp = VP[401::2, 1::10]; Vs = VS[401::2, 1::10]; Den = DEN[401::2, 1::10]   # (1200, 1360)
    W = Vp.shape[-1]
    spatial_prior = torch.load(cfg.spatial_prior_path)
    if spatial_prior.ndim == 3 and spatial_prior.shape[-1] == cfg.num_experts:
        prior_spatial = np.transpose(spatial_prior.numpy(), (2, 0, 1))   # (K, H, W)
    else:
        prior_spatial = spatial_prior.numpy()

    Syn = np.transpose(SYN, (2, 0, 1))[1:, :, :]     # (6, 1200, 1360)
    target = np.stack([Vp, Vs, Den], axis=0)          # (3, H, W)
    target_low = gaussian_filter(target, sigma=cfg.sigma_low, axes=(1,2))

    # 标准化
    train_indices = np.linspace(cfg.train_start, cfg.train_end, cfg.num_train, dtype=int)
    seismic_train = Syn[:, :, train_indices]
    target_train = target[:, :, train_indices]
    seismic_mean = np.mean(seismic_train, axis=(1,2), keepdims=True)
    seismic_std = np.std(seismic_train, axis=(1,2), keepdims=True) + 1e-8
    target_mean = np.mean(target_train, axis=(1,2), keepdims=True)
    target_std = np.std(target_train, axis=(1,2), keepdims=True) + 1e-8

    seismic_scaler = Standardize(seismic_mean, seismic_std)
    target_scaler = Standardize(target_mean, target_std)
    Syn_norm = seismic_scaler.normalize(Syn)
    target_norm = target_scaler.normalize(target)
    target_low_norm = target_scaler.normalize(target_low)

    stats = {
        'seismic_mean': torch.tensor(seismic_mean, device=device).view(1, 6, 1, 1),
        'seismic_std': torch.tensor(seismic_std, device=device).view(1, 6, 1, 1),
        'target_mean': torch.tensor(target_mean, device=device).view(1, 3, 1, 1),
        'target_std': torch.tensor(target_std, device=device).view(1, 3, 1, 1)
    }
    return (Syn_norm, target_norm, target_low_norm, prior_spatial, stats, W)


def train_one_epoch(model, dataloader, optimizer, criterion, forward, device, stats,
                    epoch: int, num_samples: int = 2, kl_z_weight: float = None,
                    kl_theta_weight: float = None):
    """单 epoch 训练，支持多采样估计地震似然。"""
    model.train()
    if kl_z_weight is not None:
        criterion.kl_z_weight = kl_z_weight
    if kl_theta_weight is not None:
        criterion.kl_theta_weight = kl_theta_weight

    total_loss = 0.0
    loss_components = {}
    num_batches = len(dataloader)

    target_mean = stats['target_mean'].to(device)
    target_std = stats['target_std'].to(device)
    seismic_mean = stats['seismic_mean'].to(device)
    seismic_std = stats['seismic_std'].to(device)

    for batch in dataloader:
        seismic = batch['seismic'].to(device)
        target = batch['target'].to(device)
        target_low = batch['target_low'].to(device)
        mask = batch['mask'].to(device)
        prior_z = batch['spatial_prior'].to(device)

        optimizer.zero_grad()

        syn_norm_list = []
        output = model(seismic, target_low, prior_z, num_samples=0)
        theta_samples = model._sample_from_mixture(
            output['logits'], output['expert_mu'], output['expert_chol_vec'], num_samples=num_samples
        )

        syn_norm_list = []
        for i in range(num_samples):
            theta_norm = theta_samples[:, i]          # (B, 3, H, W)
            theta_phys = theta_norm * target_std + target_mean
            syn_phys = forward(theta_phys)
            syn_norm = (syn_phys - seismic_mean) / (seismic_std + 1e-8)
            syn_norm_list.append(syn_norm)

        syn_norm_avg = torch.stack(syn_norm_list).mean(dim=0)

        loss, loss_dict = criterion(output, target, syn_norm_avg, seismic, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            loss_components[k] = loss_components.get(k, 0.0) + val

    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    return avg_loss, avg_components

def validate(model, dataloader, device):  #smooth_l1
    model.eval()
    total_smooth_l1 = 0.0
    total_pixels = 0
    with torch.no_grad():
        for batch in dataloader:
            seismic = batch['seismic'].to(device)
            target = batch['target'].to(device)
            target_low = batch['target_low'].to(device)
            prior_z = batch['spatial_prior'].to(device)
            out = model(seismic, target_low, prior_z, num_samples=0)
            mu = out['final_pred'][:, :3]
            mask = batch['mask'].to(device)
            
            if mask.shape[1] == 1:
                mask = mask.expand(-1, 3, -1, -1)
            mu_valid = mu[mask.bool()]
            target_valid = target[mask.bool()]
            if mu_valid.numel() > 0:
                loss = F.smooth_l1_loss(mu_valid, target_valid, reduction='sum')
                total_smooth_l1 += loss.item()
                total_pixels += mu_valid.numel()
    return total_smooth_l1 / total_pixels if total_pixels > 0 else 0.0

def train_vb_mile(model, criterion, train_loader, val_loader, forward, device, stats, cfg):
    """
    两阶段训练流程：
        Phase 1: 联合训练 + 强 KL 正则化
        Phase 2: 冻结门控 + 弱 KL 正则化
    cfg 需包含以下属性：
        phase1_epochs, phase1_lr, phase1_kl_z, phase1_kl_theta, phase1_num_samples
        phase2_epochs, phase2_lr, phase2_kl_z, phase2_kl_theta, phase2_num_samples
        save_path
    """
    # ---------- Phase 1 ----------
    print("\n" + "="*60)
    print("Phase 1: Joint training with strong KL regularization")
    print("="*60)
    model.set_phase(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.phase1_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.phase1_epochs)

    best_val = float('inf')
    for epoch in range(1, cfg.phase1_epochs + 1):
        model.set_temperature(epoch, cfg.phase1_epochs)
        train_loss, comp = train_one_epoch(
            model, train_loader, optimizer, criterion, forward, device, stats,
            epoch=epoch, num_samples=cfg.phase1_num_samples,
            kl_z_weight=cfg.phase1_kl_z, kl_theta_weight=cfg.phase1_kl_theta
        )
        val_mse = validate(model, val_loader, device)
        scheduler.step()
        print(f"[P1 E{epoch:03d}] loss={train_loss:.6f} | val_mse={val_mse:.6f} | "
              f"nll_well={comp['nll_well']:.4f} | kl_z={comp['kl_z']:.4f} | kl_theta={comp['kl_theta']:.4f}")
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), cfg.save_path.replace('.pth', '_phase1_best.pth'))

    # ---------- Phase 2 ----------
    print("\n" + "="*60)
    print("Phase 2: Gating frozen, weak KL regularization")
    print("="*60)
    model.set_phase(2)
    # 重新构建优化器（只优化编码器和解码器）
    params = list(model.encoder.parameters()) + list(model.experts.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.phase2_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.phase2_epochs)

    best_val = float('inf')
    for epoch in range(1, cfg.phase2_epochs + 1):
        model.set_temperature(epoch, cfg.phase2_epochs)
        train_loss, comp = train_one_epoch(
            model, train_loader, optimizer, criterion, forward, device, stats,
            epoch=epoch, num_samples=cfg.phase2_num_samples,
            kl_z_weight=cfg.phase2_kl_z, kl_theta_weight=cfg.phase2_kl_theta
        )
        val_mse = validate(model, val_loader, device)
        scheduler.step()
        print(f"[P2 E{epoch:03d}] loss={train_loss:.6f} | val_mse={val_mse:.6f} | "
              f"nll_well={comp['nll_well']:.4f} | kl_z={comp['kl_z']:.4f} | kl_theta={comp['kl_theta']:.4f}")
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), cfg.save_path)

    print(f"\nTraining finished. Best validation MSE: {best_val:.6f}")
    return model


def run_test(cfg, model, model_path, test_dataset, device, stats=None):
    """
    加载模型并在测试集上评估，打印指标并绘图。
    """

    from utils.metrics import compute_metrics
    from utils.plotting import plot_section_prediction, plot_trace_with_uncertainty, plot_expert_partition, plot_trace_marginal_and_experts

    # 加载模型
    model = model

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 创建 DataLoader
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # 收集所有输出
    results = {
        'pred': [],
        'final_pred': [],
        'target': [],
        'mask': [],
        'z_soft': [],
        'expert_mus': [],
        'expert_covs': []
    }
    with torch.no_grad():
        for batch in test_loader:
            seismic = batch['seismic'].to(device)           # (1, C_seis, H, W)
            target = batch['target'].to(device)             # (1, 3, H, W)
            target_low = batch['target_low'].to(device)     # (1, 3, H, W)
            mask = batch['mask'].to(device)                 # (1, 1, H, W)
            spatial_prior = batch['spatial_prior'].to(device) # (1, 1, H, W)

            # 前向传播（推理模式，不采样）
            output = model(seismic, low_freq_model=target_low,
                           prior_z=spatial_prior, num_samples=0)

            results['pred'].append(output['final_pred'][:, :3].squeeze(0).cpu().numpy())
            results['final_pred'].append(output['final_pred'].squeeze(0).cpu().numpy())
            results['target'].append(target.squeeze(0).cpu().numpy())
            results['mask'].append(mask.squeeze(0).cpu().numpy())
            results['z_soft'].append(output['z_soft'].squeeze(0).cpu().numpy())
            results['expert_mus'].append(output['expert_mu'].squeeze(0).cpu().numpy())
            results['expert_covs'].append(output['expert_cov'].squeeze(0).cpu().numpy())

    
    idx = 0
    pred = results['pred'][idx]          # (3, H, W)
    final_pred = results['final_pred'][idx]  # (6, H, W) 前3 mu 后3 logvar
    true = results['target'][idx]        # (3, H, W)
    mask = results['mask'][idx]          # (1, H, W)
    z_soft = results['z_soft'][idx]      # (K, H, W)
    expert_mus = results['expert_mus'][idx]  # (K, 3, H, W)
    expert_covs = results['expert_covs'][idx]

    # 计算指标
    metrics = compute_metrics(pred, true, mask=None)
    print("\n===== Test Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 可视化
    plot_section_prediction(pred, true, param_name='Vp')
    trace_id = pred.shape[2] // 2
    final_pred = results['final_pred'][idx]  # (6, H, W) 前3 mu 后3 logvar
    mu_final = final_pred[:3]
    logvar_final = final_pred[3:]
    # 可视化单道不确定性（取中间道）
    plot_trace_with_uncertainty(mu_final, logvar_final, true, trace_id)
    plot_expert_partition(z_soft)
    # 可视化某一道的输出边际分布
    plot_trace_marginal_and_experts(
        z_soft_section=z_soft, 
        expert_means=expert_mus, 
        expert_cov=expert_covs,
        trace_id=680,
        param_names=['Vp', 'Vs', 'Den'],
        depth=None,                     # 使用索引作为深度
        true_section=true,
        figsize=(16, 6)
    )

    return metrics

if __name__ == "__main__":
    cfg = Config2D()
    set_seed(cfg.seed, cfg.deterministic)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Syn_norm, target_norm, target_low_norm, prior_spatial, stats, W = load_data(cfg, device)

    # 准备数据集
    train_indices = np.linspace(cfg.train_start, cfg.train_end, cfg.num_train, dtype=int)
    val_indices = np.linspace(cfg.val_start, cfg.val_end, cfg.num_val, dtype=int)
    train_dataset = SeismicDataset(Syn_norm, target_norm, target_low_norm, prior_spatial,
                                   train_indices, radius=cfg.radius, mode='slice')
    val_dataset = SeismicDataset(Syn_norm, target_norm, target_low_norm, prior_spatial,
                                 val_indices, radius=cfg.radius, mode='slice')
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # 正演
    wavelet = ricker_wavelet(dt=cfg.dt, f=cfg.f, length=cfg.wavelet_len_sec).to(device)
    forward = SeismicForward(wavelet, cfg.angles_deg).to(device)

    # 模型先验
    prior_means = torch.tensor(cfg.prior_means_np, dtype=torch.float32).to(device)
    prior_cov = torch.tensor(cfg.prior_cov_np, dtype=torch.float32).to(device)

    model = GeoVBMILE2D(
        seismic_channels=cfg.seismic_channels, low_freq_channels=cfg.low_freq_channels,
        base_channels=cfg.base_channels, norm_type=cfg.norm_type, num_experts=cfg.num_experts,
        initial_temperature=cfg.initial_temperature, final_temperature=cfg.final_temperature,
        prior_means=prior_means, prior_cov=prior_cov,
        film_hidden_dim=cfg.film_hidden_dim, film_kernel_size=cfg.film_kernel_size,
        modulation_scale=cfg.modulation_scale
    ).to(device)

    criterion = VBELBO2D(
        seismic_obs_noise_init=cfg.seismic_obs_noise_init,
        well_obs_noise_init=cfg.well_obs_noise_init,
        clip_diff=cfg.clip_diff,
        expert_lambdas=cfg.expert_lambdas
    ).to(device)

    # 训练
    train_vb_mile(model, criterion, train_loader, val_loader, forward, device, stats, cfg)

    print("Training completed. Best model saved to", cfg.save_path)
    
    # 测试及可视化
    testdataset = SeismicDataset(
        seismic=Syn_norm,
        target=target_norm,
        target_low=target_low_norm,
        prior_spatial=prior_spatial,
        well_indices=None,  # 全剖面测试
        radius=None,
        mode='full'
    )

    print("\nTraining finished. Running final test on best model...")
    run_test(cfg, model, cfg.save_path, testdataset, device, stats)
    