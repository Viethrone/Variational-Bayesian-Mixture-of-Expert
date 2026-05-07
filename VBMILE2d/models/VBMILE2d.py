# ============================================================================
# 变分贝叶斯混合专家（VB-MILE）网络与训练模块
# 用于地震弹性参数反演（AVA）
# ============================================================================

import math
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 归一化工厂
# ============================================================================

class NormFactory2D:
    """
    2D 归一化层工厂，支持 batch/group/instance/layer/none 归一化类型。
    """
    @staticmethod
    def create_norm(norm_type: str, channels: int, groups: int = 8) -> nn.Module:
        norm_type = norm_type.lower()
        if norm_type == 'batch':
            return nn.BatchNorm2d(channels)
        if norm_type == 'group':
            actual_groups = min(groups, channels)
            while channels % actual_groups != 0 and actual_groups > 1:
                actual_groups -= 1
            return nn.GroupNorm(actual_groups, channels)
        if norm_type == 'instance':
            return nn.InstanceNorm2d(channels, affine=True)
        if norm_type == 'layer':
            return nn.GroupNorm(1, channels)
        if norm_type == 'none':
            return nn.Identity()
        raise ValueError(f"Unsupported norm type: {norm_type}")


# ============================================================================
# U-Net 编码器（2D）
# ============================================================================

class EncoderBlock2D(nn.Module):
    """下采样块：卷积块 + 步长2的卷积下采样。"""
    def __init__(self, in_channels: int, out_channels: int, norm_type: str = 'group'):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=0, bias=False, padding_mode='reflect'),
            NormFactory2D.create_norm(norm_type, out_channels),
            nn.GELU(),
        )
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.conv_block(x))


class UNetEncoder2D(nn.Module):
    """二维 U-Net 编码器，输出三个尺度的特征图。"""
    def __init__(self, in_channels: int, base_channels: int = 8, norm_type: str = 'group'):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, bias=False, padding_mode='replicate'),
            NormFactory2D.create_norm(norm_type, in_channels * 2),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, base_channels, 1, bias=False),
            NormFactory2D.create_norm(norm_type, base_channels),
            nn.GELU()
        )
        self.enc1 = EncoderBlock2D(base_channels, base_channels * 2, norm_type)
        self.enc2 = EncoderBlock2D(base_channels * 2, base_channels * 4, norm_type)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_feat = self.input_proj(x)
        skip1 = self.enc1(global_feat)
        skip2 = self.enc2(skip1)
        return skip1, skip2, global_feat


# ============================================================================
# 变分分区层（门控网络）——支持动态温度
# ============================================================================

class VariationalPartitionLayer2D(nn.Module):
    """输出 q(z|x) 的 logits，并计算 KL(q(z) || p(z))。"""
    def __init__(self, in_channels: int, num_experts: int = 3,
                 norm_type: str = 'group', initial_tau: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.register_buffer('tau', torch.tensor(initial_tau))

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, num_experts, 3, padding=1, bias=False, padding_mode='reflect'),
            NormFactory2D.create_norm(norm_type, num_experts),
            nn.GELU(),
            nn.Conv2d(num_experts, num_experts, 3, dilation=5, padding=5, bias=False, padding_mode='reflect'),
            NormFactory2D.create_norm(norm_type, num_experts),
            nn.GELU(),
            nn.Conv2d(num_experts, num_experts, 1, bias=False),
        )

    def forward(self,
                low_freq: torch.Tensor,
                global_feat: torch.Tensor,
                prior: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([low_freq, global_feat], dim=1)
        logits = self.net(x) / self.tau
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        if prior is not None:
            log_prior = torch.log(prior + 1e-12)
        else:
            log_prior = -math.log(self.num_experts) * torch.ones_like(log_probs)

        kl_per_pixel = (probs * (log_probs - log_prior)).sum(dim=1)
        kl_z = kl_per_pixel.mean()
        return logits, kl_z

    def set_temperature(self, tau: float) -> None:
        """更新 softmax 温度。"""
        self.tau.fill_(tau)


# ============================================================================
# 专家解码器（带空间 FiLM 调制）——保持不变，但统一 padding_mode
# ============================================================================

class SpatialFiLMGenerator(nn.Module):
    """空间 FiLM 参数生成器。"""
    def __init__(self, out_channels: int, hidden_dim: int = 8,
                 kernel_size: int = 3, modulation_scale: float = 1.0):
        super().__init__()
        self.modulation_scale = modulation_scale
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size, padding=padding, bias=False, padding_mode='reflect')
        self.norm = nn.GroupNorm(min(4, hidden_dim), hidden_dim)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_dim, out_channels * 2, 1)

    def forward(self, w_prob: torch.Tensor) -> torch.Tensor:
        x = self.conv1(w_prob)
        x = self.norm(x)
        x = self.relu(x)
        out = self.conv2(x)
        scale, shift = out.chunk(2, dim=1)
        scale = scale * self.modulation_scale
        return torch.cat([scale, shift], dim=1)


class FiLMBasicBlock2D(nn.Module):
    """带空间 FiLM 调制的双路径残差块。"""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0,
                 norm_type: str = 'group', dilation: int = 3,
                 film_hidden_dim: int = 8, film_kernel_size: int = 3,
                 modulation_scale: float = 1.0):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            NormFactory2D.create_norm(norm_type, out_channels),
        )
        self.conv3_dilated = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation,
                      bias=False, padding_mode='replicate'),
            NormFactory2D.create_norm(norm_type, out_channels),
        )
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.film_generator = SpatialFiLMGenerator(
            out_channels, film_hidden_dim, film_kernel_size, modulation_scale
        )

    def forward(self, x: torch.Tensor, w_prob: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x) + self.conv3_dilated(x)
        film_params = self.film_generator(w_prob)
        scale, shift = film_params.chunk(2, dim=1)
        out = out * (1 + torch.tanh(scale)) + torch.tanh(shift)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class ExpertDecoder2D(nn.Module):
    """完全独立的专家解码器，每个专家输出联合高斯分布的均值和 Cholesky 因子。"""
    def __init__(self, skip_channels: List[int], num_experts: int, norm_type: str = 'group',
                 prior_means: Optional[torch.Tensor] = None,
                 prior_cov_chol: Optional[torch.Tensor] = None,
                 film_hidden_dim: int = 8, film_kernel_size: int = 3,
                 modulation_scale: float = 0.5):
        super().__init__()
        self.num_experts = num_experts
        c1, c2, c4 = skip_channels   # base*2, base*4, base

        # 全局残差分支（所有专家共享）
        self.global_residual = nn.Sequential(
            nn.Conv2d(c4, c4, 3, dilation=3, padding=3, bias=False, padding_mode='reflect'),
            NormFactory2D.create_norm(norm_type, c4),
            nn.GELU(),
            nn.Conv2d(c4, 9, 1, bias=True)
        )

        # 每个专家独立的组件
        self.skip2_convs = nn.ModuleList()
        self.final_convs = nn.ModuleList()
        self.conv_fuses = nn.ModuleList()
        self.out_joint = nn.ModuleList()

        for _ in range(num_experts):
            skip2_conv = nn.Sequential(
                nn.Conv2d(c2, c2, 3, dilation=3, padding=3, bias=False, padding_mode='replicate'),
                NormFactory2D.create_norm(norm_type, c2),
                nn.GELU()
            )
            self.skip2_convs.append(skip2_conv)

            final_conv = nn.Sequential(
                nn.Conv2d(c1, c1, 3, dilation=5, padding=5, bias=False, padding_mode='reflect'),
                NormFactory2D.create_norm(norm_type, c1),
                nn.GELU()
            )
            self.final_convs.append(final_conv)

            conv_fuse = FiLMBasicBlock2D(
                in_channels=c2 + c1, out_channels=c1, norm_type=norm_type,
                dilation=7, film_hidden_dim=film_hidden_dim,
                film_kernel_size=film_kernel_size, modulation_scale=modulation_scale
            )
            self.conv_fuses.append(conv_fuse)

            head = nn.Sequential(
                nn.Conv2d(c1, c1, 3, padding=1, bias=False, padding_mode='reflect'),
                NormFactory2D.create_norm(norm_type, c1),
                nn.GELU(),
                nn.Conv2d(c1, 9, 1, bias=True)
            )
            self.out_joint.append(head)

        # 使用先验信息初始化偏置
        if prior_means is not None and prior_cov_chol is not None:
            with torch.no_grad():
                for k in range(num_experts):
                    self.out_joint[k][-1].bias.data[:3] = prior_means[k]
                    idx = torch.tril_indices(3, 3)
                    chol_vec = prior_cov_chol[k][idx[0], idx[1]]
                    self.out_joint[k][-1].bias.data[3:] = chol_vec

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                target_size: Tuple[int, int], z_soft: torch.Tensor) -> torch.Tensor:
        skip1, skip2, global_feat = features
        B, _, H, W = skip1.shape

        global_out = self.global_residual(global_feat)
        if global_out.shape[-2:] != target_size:
            global_out = F.interpolate(global_out, size=target_size, mode='bilinear', align_corners=False)

        expert_outputs = []
        for k in range(self.num_experts):
            w_prob = z_soft[:, k:k+1]                         # (B,1,H,W)
            w_prob_s1 = F.interpolate(w_prob, size=skip1.shape[-2:], mode='bilinear')

            up_skip2 = F.interpolate(skip2, size=skip1.shape[-2:], mode='bilinear', align_corners=False)
            up_skip2 = self.skip2_convs[k](up_skip2)

            x = torch.cat([up_skip2, skip1], dim=1)
            x = self.conv_fuses[k](x, w_prob_s1)

            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            x = self.final_convs[k](x)

            out = self.out_joint[k](x)
            out = out + global_out

            # 确保 Cholesky 对角线为正
            chol_raw = out[:, 3:]
            chol_raw[:, 0] = torch.exp(chol_raw[:, 0])   # L00
            chol_raw[:, 2] = torch.exp(chol_raw[:, 2])   # L11
            chol_raw[:, 5] = torch.exp(chol_raw[:, 5])   # L22
            out = torch.cat([out[:, :3], chol_raw], dim=1)
            expert_outputs.append(out)

        return torch.stack(expert_outputs, dim=1)   # (B, K, 9, H_out, W_out)


# ============================================================================
# 辅助函数：Cholesky ↔ 协方差矩阵
# ============================================================================

def build_covariance_from_chol(chol_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    chol_vec: (B, 6, H, W) -> Sigma: (B, 3, 3, H, W), logdet: (B, H, W)
    """
    B, _, H, W = chol_vec.shape
    device = chol_vec.device
    L = torch.zeros(B, 3, 3, H, W, device=device)
    L[:, 0, 0] = chol_vec[:, 0]
    L[:, 1, 0] = chol_vec[:, 1]
    L[:, 1, 1] = chol_vec[:, 2]
    L[:, 2, 0] = chol_vec[:, 3]
    L[:, 2, 1] = chol_vec[:, 4]
    L[:, 2, 2] = chol_vec[:, 5]

    Sigma = torch.einsum('bikhw,bjkhw->bijhw', L, L)
    logdet = 2.0 * (torch.log(chol_vec[:, 0].clamp(min=1e-6)) +
                    torch.log(chol_vec[:, 2].clamp(min=1e-6)) +
                    torch.log(chol_vec[:, 5].clamp(min=1e-6)))
    return Sigma, logdet


# ============================================================================
# 主模型：变分贝叶斯混合专家（VB-MILE）
# ============================================================================

class GeoVBMILE2D(nn.Module):
    """
    二维变分贝叶斯混合专家模型，用于地震 AVA 反演。
    包含编码器、门控网络、多个专家解码器，输出边际高斯分布并支持采样。
    """
    def __init__(self,
                 seismic_channels: int = 6,
                 low_freq_channels: int = 3,
                 base_channels: int = 16,
                 norm_type: str = 'group',
                 num_experts: int = 3,
                 initial_temperature: float = 1.0,
                 final_temperature: float = 0.1,
                 prior_means: Optional[torch.Tensor] = None,   # (K, 3)
                 prior_cov: Optional[torch.Tensor] = None,     # (K, 3, 3)
                 film_hidden_dim: int = 8,
                 film_kernel_size: int = 3,
                 modulation_scale: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.low_freq_channels = low_freq_channels

        # ----- 先验：全协方差高斯（每个专家独立）-----
        if prior_means is None:
            prior_means = torch.zeros(num_experts, 3)
        self.register_buffer("prior_means", prior_means)

        if prior_cov is None:
            prior_cov = torch.eye(3).unsqueeze(0).repeat(num_experts, 1, 1)
        eps = 1e-6
        reg_cov = prior_cov + eps * torch.eye(3, device=prior_cov.device).unsqueeze(0)
        self.register_buffer("prior_cov", reg_cov)
        self.register_buffer("prior_cov_inv", torch.inverse(reg_cov))
        self.register_buffer("prior_cov_logdet", torch.logdet(reg_cov))
        prior_cov_chol = torch.linalg.cholesky(reg_cov)

        # ----- 编码器 -----
        encoder_in = seismic_channels + low_freq_channels
        self.encoder = UNetEncoder2D(encoder_in, base_channels, norm_type)
        skip_channels = [base_channels * 2, base_channels * 4, base_channels]

        # ----- 门控网络（支持动态温度）-----
        partition_in = low_freq_channels + base_channels
        self.gating = VariationalPartitionLayer2D(
            partition_in, num_experts, norm_type, initial_tau=initial_temperature
        )

        # ----- 专家解码器 -----
        self.experts = ExpertDecoder2D(
            skip_channels, num_experts, norm_type,
            prior_means=prior_means,
            prior_cov_chol=prior_cov_chol,
            film_hidden_dim=film_hidden_dim,
            film_kernel_size=film_kernel_size,
            modulation_scale=modulation_scale
        )

    def set_temperature(self, epoch: int, total_epochs: int) -> None:
        """余弦退火更新门控网络的温度。"""
        if total_epochs <= 0:
            return
        cos_val = (1 + math.cos(math.pi * epoch / total_epochs)) / 2
        new_temp = self.final_temperature + (self.initial_temperature - self.final_temperature) * cos_val
        self.gating.set_temperature(new_temp)

    def set_phase(self, phase: int) -> None:
        """训练阶段控制：1=端到端，2=冻结门控网络。"""
        if phase == 1:
            self.train()
            for p in self.parameters():
                p.requires_grad = True
        elif phase == 2:
            self.experts.train()
            self.encoder.train()
            self.gating.eval()
            for p in self.encoder.parameters():
                p.requires_grad = True
            for p in self.gating.parameters():
                p.requires_grad = False
            for p in self.experts.parameters():
                p.requires_grad = True
        else:
            raise ValueError("phase must be 1 or 2")

    def forward(self,
                seismic: torch.Tensor,
                low_freq_model: Optional[torch.Tensor] = None,
                prior_z: Optional[torch.Tensor] = None,
                num_samples: int = 0) -> Dict[str, Any]:
        """
        前向传播。

        Args:
            seismic: (B, seismic_channels, H, W)
            low_freq_model: (B, low_freq_channels, H, W)，若为 None 则填充 0
            prior_z: (B, K, H, W) 门控先验概率，可选
            num_samples: 若 >0，则从混合后验中采样相应次数（返回的 theta_sample 形状为 (B, num_samples, 3, H, W)）

        Returns:
            dict 包含：
                final_pred: (B, 6, H, W)  边际均值 + 对数方差
                theta_sample: (B, num_samples, 3, H, W) 或 None
                expert_mu: (B, K, 3, H, W)
                expert_cov: (B, K, 3, 3, H, W)
                expert_chol_vec: (B, K, 6, H, W)
                z_soft: (B, K, H, W)
                logits: (B, K, H, W)
                kl_z: 标量
                kl_theta: 标量（加权后的总 KL）
                kl_theta_per_expert: (B, K)
        """
        B, _, H, W = seismic.shape
        if low_freq_model is None:
            low_freq_model = torch.zeros(B, self.low_freq_channels, H, W, device=seismic.device)

        # 编码
        features = self.encoder(torch.cat([seismic, low_freq_model], dim=1))
        skip1, skip2, global_feat = features

        # 门控
        logits, kl_z = self.gating(low_freq_model, global_feat, prior=prior_z)
        z_soft = F.softmax(logits, dim=1)

        # 专家解码 (B, K, 9, H, W)
        expert_out = self.experts([skip1, skip2, global_feat], (H, W), z_soft)
        expert_mu = expert_out[:, :, :3]          # (B, K, 3, H, W)
        expert_chol_vec = expert_out[:, :, 3:]    # (B, K, 6, H, W)

        # 构建协方差矩阵
        expert_cov = []
        expert_logdet = []
        for k in range(self.num_experts):
            cov_k, logdet_k = build_covariance_from_chol(expert_chol_vec[:, k])
            expert_cov.append(cov_k)
            expert_logdet.append(logdet_k)
        expert_cov = torch.stack(expert_cov, dim=1)        # (B, K, 3, 3, H, W)
        expert_logdet = torch.stack(expert_logdet, dim=1)  # (B, K, H, W)

        # 计算 KL(q(θ|z) || p(θ)) 并加权
        kl_theta_per_expert = self._compute_kl_theta_per_expert(expert_mu, expert_cov, expert_logdet)  # (B, K)
        # 加权求和：E_{q(z)}[ KL(...) ] = Σ_k (平均 z_soft) * kl_theta_k
        mean_z = z_soft.mean(dim=(2, 3))                 # (B, K)
        kl_theta = (mean_z * kl_theta_per_expert).sum(dim=1).mean()

        # 边际分布
        mu_marg, var_marg = self._compute_marginal_gaussian(z_soft, expert_mu, expert_cov)
        logvar_marg = torch.log(var_marg.clamp(min=1e-6))
        final_pred = torch.cat([mu_marg, logvar_marg], dim=1)   # (B, 6, H, W)

        # 采样（可选）
        theta_sample = None
        if num_samples > 0:
            theta_sample = self._sample_from_mixture(logits, expert_mu, expert_chol_vec, num_samples)

        return {
            "final_pred": final_pred,
            "theta_sample": theta_sample,
            "expert_mu": expert_mu,
            "expert_cov": expert_cov,
            "expert_chol_vec": expert_chol_vec,
            "z_soft": z_soft,
            "logits": logits,
            "kl_z": kl_z,
            "kl_theta": kl_theta,
            "kl_theta_per_expert": kl_theta_per_expert,
        }

    # ---------- 私有辅助方法 ----------
    def _compute_kl_theta_per_expert(self, expert_mu, expert_cov, expert_logdet):
        B, K, _, H, W = expert_mu.shape
        device = expert_mu.device
        kl = torch.zeros(B, K, device=device)

        for k in range(K):
            mu_k = expert_mu[:, k]                     # (B, 3, H, W)
            cov_k = expert_cov[:, k]                   # (B, 3, 3, H, W)
            logdet_k = expert_logdet[:, k]             # (B, H, W)

            diff = mu_k - self.prior_means[k, :, None, None]   # (B, 3, H, W)

            N = B * H * W
            cov_flat = cov_k.permute(0, 3, 4, 1, 2).reshape(N, 3, 3)   # (N, 3, 3)
            diff_flat = diff.permute(0, 2, 3, 1).reshape(N, 3, 1)      # (N, 3, 1)
            prior_inv = self.prior_cov_inv[k]           # (3, 3)

            product = prior_inv @ cov_flat              # (N, 3, 3)
            trace = torch.einsum('bii->b', product)     # (N,)
            term1 = trace.reshape(B, H, W)

            quad = (diff_flat.transpose(-2, -1) @ prior_inv @ diff_flat).squeeze()  # (N,)
            term2 = quad.reshape(B, H, W)

            term3 = -3.0
            term4 = self.prior_cov_logdet[k] - logdet_k   # (B, H, W)

            kl_pixel = 0.5 * (term1 + term2 + term3 + term4)   # (B, H, W)
            kl[:, k] = kl_pixel.mean(dim=(1, 2))
        return kl


    def _compute_marginal_gaussian(self, z_soft, expert_mu, expert_cov):
        """
        计算混合高斯边际分布的均值和方差（对角线）。
        返回:
            mu_marg: (B, 3, H, W)
            var_marg: (B, 3, H, W)
            cov_marg: (B, 3, 3, H, W)
        """
        mu_marg = (z_soft.unsqueeze(2) * expert_mu).sum(dim=1)          # (B,3,H,W)

        mu_mu_T = torch.einsum('bkihw,bkjhw->bkijhw', expert_mu, expert_mu)
        weighted_sum = (z_soft[:, :, None, None, :, :] * (expert_cov + mu_mu_T)).sum(dim=1)

        mu_marg_muT = torch.einsum('bihw,bjhw->bijhw', mu_marg, mu_marg)
        device = weighted_sum.device
        cov_marg = weighted_sum - mu_marg_muT + 1e-6 * torch.eye(3, device=device).view(1, 3, 3, 1, 1)

        # diagonal 返回 (B, H, W, 3)，需转置为 (B, 3, H, W)
        var_marg = torch.diagonal(cov_marg, dim1=1, dim2=2)   # (B, H, W, 3)
        var_marg = var_marg.permute(0, 3, 1, 2)               # (B, 3, H, W)

        return mu_marg, var_marg


    def _sample_from_mixture(self,
                             logits: torch.Tensor,
                             expert_mu: torch.Tensor,
                             expert_chol_vec: torch.Tensor,
                             num_samples: int = 1) -> torch.Tensor:
        """
        从混合后验中采样弹性参数。
        Returns:
            samples: (B, num_samples, 3, H, W)
        """
        B, K, _, H, W = expert_mu.shape
        device = expert_mu.device
        samples_list = []

        for _ in range(num_samples):
            # Gumbel-Softmax 硬分配
            with torch.no_grad():
                # 使用当前门控温度 (从 gating 模块获取)
                tau = self.gating.tau.item()
                z_hard = F.gumbel_softmax(logits, tau=tau, hard=True, dim=1)  # (B, K, H, W)

            # 构建下三角 Cholesky 矩阵 L (B, K, 3, 3, H, W)
            L = torch.zeros(B, K, 3, 3, H, W, device=device)
            L[:, :, 0, 0] = expert_chol_vec[:, :, 0]
            L[:, :, 1, 0] = expert_chol_vec[:, :, 1]
            L[:, :, 1, 1] = expert_chol_vec[:, :, 2]
            L[:, :, 2, 0] = expert_chol_vec[:, :, 3]
            L[:, :, 2, 1] = expert_chol_vec[:, :, 4]
            L[:, :, 2, 2] = expert_chol_vec[:, :, 5]

            eps = torch.randn(B, 3, H, W, device=device)
            sample = torch.zeros(B, 3, H, W, device=device)

            for k in range(K):
                mask = z_hard[:, k:k+1]   # (B, 1, H, W)
                if mask.sum() == 0:
                    continue
                # μ_k + L_k @ eps
                L_k = L[:, k]             # (B, 3, 3, H, W)
                eps_exp = eps.unsqueeze(2)  # (B, 3, 1, H, W)
                sample_k = expert_mu[:, k] + torch.einsum('bijhw,bjkhw->bikhw', L_k, eps_exp).squeeze(2)
                sample = sample * (1 - mask) + sample_k * mask
            samples_list.append(sample)

        samples = torch.stack(samples_list, dim=1)   # (B, num_samples, 3, H, W)
        return samples
    
    