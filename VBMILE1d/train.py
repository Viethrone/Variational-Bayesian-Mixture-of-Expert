import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

from config import Config
from data.dataset import Seismic1DDataset, Standardize
from models.VBMILE1d import Geo_VBMILE_1D
from models.VBlosses import VBELBOLoss1D
from models.forward import SeismicForward1D, ricker_wavelet
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR

def set_seed(seed, deterministic):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def load_data(cfg):
    # 加载原始数据
    VP = np.load(cfg.vp_path)
    VS = np.load(cfg.vs_path)
    DEN = np.load(cfg.den_path)
    mat_data = loadmat(cfg.seismic_mat_path)
    SYN = mat_data['seismic_data']

    Vp = VP[401::2, 1::10]          # (1200, 1360)
    Vs = VS[401::2, 1::10]
    Den = DEN[401::2, 1::10]
    W = Vp.shape[-1]

    # 加载先验
    priors = np.load(cfg.gmm_prior_path)
    spatial_prior = torch.load(cfg.spatial_prior_path)
    expert_means = priors['means']      # (K, 3)
    expert_vars = priors['variances']
    prior_spatial = np.transpose(spatial_prior, (2, 0, 1))   # (K, H, W)
    prior_means = torch.tensor(expert_means, dtype=torch.float32)
    prior_vars = torch.tensor(expert_vars, dtype=torch.float32)

    # 正演数据
    Syn = np.transpose(SYN, (2, 0, 1))[1:, :, :]   # (6, 1200, 1360)
    target = np.stack([Vp, Vs, Den], axis=0)   # (3, 1200, 1360)
    target_low = gaussian_filter(target, sigma=cfg.sigma_low, axes=(1,2))

    # 标准化
    train_indices = np.linspace(cfg.train_start, cfg.train_end, cfg.num_train, dtype=int)
    seismic_train = Syn[:, :, train_indices]   # (C_seis, H, num_train)
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
        'seismic_mean': torch.tensor(seismic_mean, device=device).view(1, 6, 1),
        'seismic_std': torch.tensor(seismic_std, device=device).view(1, 6, 1),
        'target_mean': torch.tensor(target_mean, device=device).view(1, 3, 1),
        'target_std': torch.tensor(target_std, device=device).view(1, 3, 1)
    }
    return (Syn_norm, target_norm, target_low_norm, prior_spatial,
            expert_means, expert_vars, stats, W)

def train_one_epoch_1d(model, dataloader, optimizer, criterion, forward, device, epoch, stats, log_interval=10):
    model.train()
    total_loss = 0.0
    loss_components = {}
    num_batches = len(dataloader)

    target_mean = stats['target_mean'].to(device)
    target_std = stats['target_std'].to(device)
    seismic_mean = stats['seismic_mean'].to(device)
    seismic_std = stats['seismic_std'].to(device)

    for batch_idx, batch in enumerate(dataloader):
        seismic = batch['seismic'].to(device)          # (B, C_seis, L)
        target = batch['target'].to(device)            # (B, 3, L)
        target_low = batch['target_low'].to(device)    # (B, 3, L)
        mask = batch['mask'].to(device)                # (B, 1, L)
        spatial_prior = batch['prior'].to(device)

        optimizer.zero_grad()
        model_output = model(seismic, target_low, spatial_prior, num_samples=1)

        theta_sample_norm = model_output['theta_sample']
        theta_sample_phys = theta_sample_norm * target_std + target_mean
        syn_phys = forward(theta_sample_phys)
        syn_norm = (syn_phys - seismic_mean) / (seismic_std + 1e-8)

        total_loss_batch, loss_dict = criterion(
            model_output=model_output,
            target=target,
            syn_norm=syn_norm,
            seismic=seismic,
            target_mask=mask
        )

        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += total_loss_batch.item()
        for k, v in loss_dict.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            loss_components[k] = loss_components.get(k, 0.0) + val

        if batch_idx % log_interval == 0:
            print(f'Epoch {epoch:03d} Batch {batch_idx:04d}: Loss = {total_loss_batch.item():.6f}')

    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    return avg_loss, avg_components

def validate_point_estimate_1d(model, dataloader, device):
    model.eval()
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            seismic = batch['seismic'].to(device)
            target = batch['target'].to(device)
            target_low = batch['target_low'].to(device)
            mask = batch['mask'].to(device)           # (B, 1, L) 或 (B, 3, L)
            prior = batch['prior'].to(device)

            # 调用模型，推理模式无需采样
            model_output = model(seismic, low_freq_model=target_low, prior=prior, num_samples=0)
            mu = model_output['final_pred'][:, :3]     # (B, 3, L)

            # 将 mask 广播到与 mu 相同形状
            if mask.shape[1] == 1:
                mask = mask.expand(-1, 3, -1)          # (B, 3, L)

            mu_valid = mu[mask.bool()]
            target_valid = target[mask.bool()]

            if mu_valid.numel() > 0:
                loss = mse_loss(mu_valid, target_valid)
                total_loss += loss.item() * mu_valid.numel()
                num_samples += mu_valid.numel()

    return total_loss / num_samples if num_samples > 0 else 0.0
   
def train_model_1d(model, train_loader, val_loader, criterion, forward, device, stats,
                   epochs=200, lr=1e-3, weight_decay=1e-2,
                   warmup_epochs=20, warmup_start_lr=1e-6, eta_min=1e-8,
                   save_path='best_model_1d.pth', val_metric='mse'):

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min
    )

    best_val = float('inf')
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        if hasattr(model, 'set_temperature'):
            model.set_temperature(epoch - 1, epochs)

        train_loss, train_comp = train_one_epoch_1d(
            model, train_loader, optimizer, criterion, forward, device, epoch, stats
        )

        if val_metric == 'mse':
            val_loss = validate_point_estimate_1d(model, val_loader, device)
            metric_name = 'MSE'
        else:
            raise ValueError(f"Unknown val_metric: {val_metric}")

        scheduler.step()

        print(f"\nEpoch {epoch:03d}/{epochs} | Train Loss: {train_loss:.6f} | Val {metric_name}: {val_loss:.6f}")
        print(f"   Components: nll_well={train_comp.get('nll_well',0):.6f}, "
              f"seismic_nll={train_comp.get('seismic_nll',0):.6f}, "
              f"kl_z={train_comp.get('kl_z',0):.6f}, "
              f"kl_theta={train_comp.get('kl_theta',0):.6f}, "
              f"well_var={train_comp.get('well_var',0):.6f}, "
              f"seismic_var={train_comp.get('seismic_var',0):.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            print(f"   -> Best model saved (epoch {epoch}, {metric_name}={val_loss:.6f})")

    print(f"\nTraining finished. Best {metric_name} = {best_val:.6f} at epoch {best_epoch}")
    return model

def run_test(cfg, model, model_path, test_dataset, device, stats=None):
    """
    加载模型并在测试集上评估，打印指标并绘图。
    """
    from models.VBMILE1d import Geo_VBMILE_1D
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
    all_mu, all_logvar, all_z_soft, all_expert_mus, all_expert_logvars = [], [], [], [], []
    all_true_mu = []

    with torch.no_grad():
        for batch in test_loader:
            seismic = batch['seismic'].to(device)
            target_low = batch['target_low'].to(device)
            target = batch['target'].to(device)
            prior = batch['prior'].to(device) if batch.get('prior') is not None else None

            outputs = model(seismic, low_freq_model=target_low, prior=prior, num_samples=1)

            mu = outputs['final_pred'][:, :3].cpu()
            logvar = outputs['final_pred'][:, 3:].cpu()
            z_soft = outputs['z_soft'].cpu()
            expert_mus = outputs['expert_mus'].cpu()
            expert_logvars = outputs['expert_logvars'].cpu()
            true_mu = target.cpu()

            all_mu.append(mu)
            all_logvar.append(logvar)
            all_z_soft.append(z_soft)
            all_expert_mus.append(expert_mus)
            all_expert_logvars.append(expert_logvars)
            all_true_mu.append(true_mu)

    # 拼接
    mu_all = torch.cat(all_mu, dim=0).numpy()                     # (N, 3, L)
    logvar_all = torch.cat(all_logvar, dim=0).numpy()
    z_soft_all = torch.cat(all_z_soft, dim=0).numpy()             # (N, K, L)
    expert_mus_all = torch.cat(all_expert_mus, dim=0).numpy()     # (N, K, 3, L)
    expert_logvars_all = torch.cat(all_expert_logvars, dim=0).numpy()
    true_mu_all = torch.cat(all_true_mu, dim=0).numpy()           # (N, 3, L)

    # 重排为剖面格式 (3, L, N)
    mu_section = np.transpose(mu_all, (1, 2, 0))
    true_section = np.transpose(true_mu_all, (1, 2, 0))
    logvar_section = np.transpose(logvar_all, (1, 2, 0))
    z_soft_section = np.transpose(z_soft_all, (1, 2, 0))          # (K, L, N)
    expert_means_section = np.transpose(expert_mus_all, (1, 2, 3, 0))  # (K, 3, L, N)
    expert_logvars_section = np.transpose(expert_logvars_all, (1, 2, 3, 0))

    # 计算指标
    metrics = compute_metrics(mu_section, true_section, mask=None)
    print("\n===== Test Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 可视化
    plot_section_prediction(mu_section, true_section, param_name='Vp')
    trace_id = min(700, mu_section.shape[2] - 1)
    plot_trace_with_uncertainty(mu_section, logvar_section, true_section, trace_id, param_name='Vp')
    plot_expert_partition(z_soft_section)
    plot_trace_marginal_and_experts(
        z_soft_section, expert_means_section, expert_logvars_section,
        trace_id=0, param_names=['Vp', 'Vs', 'Den'], param_for_marginal='Vp',
        depth=None, true_section=true_section, figsize=(16, 6)
    )

    return metrics

if __name__ == "__main__":
    cfg = Config()
    set_seed(cfg.seed, cfg.deterministic)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    (Syn_norm, target_norm, target_low_norm, prior_spatial,
     expert_means, expert_vars, stats, W) = load_data(cfg)

    # 准备训练/验证索引
    train_indices = np.linspace(cfg.train_start, cfg.train_end, cfg.num_train, dtype=int)
    val_indices = np.linspace(cfg.val_start, cfg.val_end, cfg.num_val, dtype=int)

    train_dataset = Seismic1DDataset(Syn_norm, target_norm, target_low_norm,
                                     prior_spatial, train_indices, augment=False)
    val_dataset = Seismic1DDataset(Syn_norm, target_norm, target_low_norm,
                                   prior_spatial, val_indices, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # 模型、损失、正演
    wavelet = ricker_wavelet(cfg.f, cfg.dt, cfg.wavelet_length).to(device)
    forward = SeismicForward1D(wavelet, cfg.angles_deg).to(device)

    prior_means = torch.tensor(expert_means, dtype=torch.float32).to(device)
    prior_vars = torch.tensor(expert_vars, dtype=torch.float32).to(device)

    model = Geo_VBMILE_1D(
        seismic_channels=cfg.seismic_channels,
        low_freq_channels=cfg.low_freq_channels,
        base_channels=cfg.base_channels,
        norm_type=cfg.norm_type,
        num_experts=cfg.num_experts,
        initial_temperature=cfg.initial_temperature,
        final_temperature=cfg.final_temperature,
        prior_means=prior_means,
        prior_vars=prior_vars,
        prior_spatial=prior_spatial   # 传入，模型内部可忽略
    ).to(device)

    criterion = VBELBOLoss1D(
        seismic_obs_noise_init=cfg.seismic_obs_noise_init,
        well_obs_noise_init=cfg.well_obs_noise_init,
        clip_diff=cfg.clip_diff
    ).to(device)

    # 训练
    trained_model = train_model_1d(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        forward=forward,
        device=device,
        stats=stats,
        epochs=cfg.total_epochs,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,
        warmup_start_lr=cfg.warmup_start_lr,
        eta_min=cfg.eta_min,
        save_path=cfg.save_path
    )

    # 测试及可视化
    test_dataset = Seismic1DDataset(
        seismic=Syn_norm,
        target=target_norm,
        target_low=target_low_norm,
        well_indices=np.arange(W, dtype=int),   # 所有道
        prior_spatial=prior_spatial,
        augment=False
    )
    print("\nTraining finished. Running final test on best model...")
    run_test(cfg, model, cfg.save_path, test_dataset, device, stats)
