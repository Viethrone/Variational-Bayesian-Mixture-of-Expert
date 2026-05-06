import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
import numpy as np

# ============================================
# 归一化工厂
# ============================================

class NormFactory1D:
    """1D 归一化层工厂，支持 batch / group / instance / layer / none。"""
    @staticmethod
    def create_norm(norm_type: str, channels: int, groups: int = 8) -> nn.Module:
        norm_type = norm_type.lower()
        if norm_type == 'batch':
            return nn.BatchNorm1d(channels)
        if norm_type == 'group':
            actual_groups = min(groups, channels)
            while channels % actual_groups != 0 and actual_groups > 1:
                actual_groups -= 1
            return nn.GroupNorm(actual_groups, channels)
        if norm_type == 'instance':
            return nn.InstanceNorm1d(channels, affine=True)
        if norm_type == 'layer':
            return nn.GroupNorm(1, channels)
        if norm_type == 'none':
            return nn.Identity()
        raise ValueError(f"Unsupported norm type: {norm_type}")


# ============================================
# 注意力模块 (1D)
# ============================================

class ChannelAttention1D(nn.Module):
    """通道注意力 (SE 风格)，使用平均池化和最大池化。"""
    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        mid_channel = max(channel // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv1d(channel, mid_channel, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channel, channel, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        avg_out = self.mlp(F.adaptive_avg_pool1d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool1d(x, 1))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention1D(nn.Module):
    """空间注意力，通过 1D 卷积学习注意力权重。"""
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B,1,L)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B,1,L)
        x_cat = torch.cat([avg_out, max_out], dim=1)   # (B,2,L)
        attention = self.sigmoid(self.conv(x_cat))     # (B,1,L)
        return attention * x


class CBAM1D(nn.Module):
    """CBAM 模块，顺序应用通道注意力和空间注意力，并带有残差连接。"""
    def __init__(self, channel: int, reduction: int = 4, spatial_kernel: int = 5):
        super().__init__()
        self.channel_attention = ChannelAttention1D(channel, reduction)
        self.spatial_attention = SpatialAttention1D(spatial_kernel)

    def forward(self, x):
        return x + self.spatial_attention(self.channel_attention(x))


# ============================================
# 基础构建块
# ============================================

class BasicBlock1D(nn.Module):
    """
    双路径残差块：
        - 1x1 卷积分支（线性变换）
        - 空洞卷积分支 (默认 dilation=5，感受野较大)
    两条分支输出相加后经过 CBAM 注意力，再输出。
    顺序: Conv -> Norm -> ReLU -> Dropout
    """
    def __init__(self, in_channels, out_channels, dropout=0.0, norm_type='group',
                 dilation=5, use_attention=True):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            NormFactory1D.create_norm(norm_type, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.conv3_dilated = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            NormFactory1D.create_norm(norm_type, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention = CBAM1D(channel=out_channels) if use_attention else nn.Identity()

    def forward(self, x):
        out = self.conv1x1(x) + self.conv3_dilated(x)
        return self.attention(out)


# ============================================
# U-Net 编码器 (1D)
# ============================================

class EncoderBlock1D(nn.Module):
    """下采样块：BasicBlock1D + 步长2卷积下采样。"""
    def __init__(self, in_channels, out_channels, norm_type='group'):
        super().__init__()
        self.conv_block = BasicBlock1D(in_channels, out_channels, dropout=0.2, dilation=5,
                                       use_attention=True, norm_type=norm_type)
        self.downsample = nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.downsample(x)
        return x


class UNetEncoder1D(nn.Module):
    """
    一维 U-Net 风格编码器，输出四个特征：
        skip1:  下采样一次后的特征 (B, base_channels*2, L/2)
        skip2:  下采样两次后的特征 (B, base_channels*4, L/4)
        bottom: 瓶颈特征 (B, base_channels*8, L/8)
        global_feat: 输入投影后的原始分辨率特征 (B, base_channels, L)
    """
    def __init__(self, in_channels=4, base_channels=8, norm_type='group'):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * 2, 3, padding=1, bias=False),
            NormFactory1D.create_norm(norm_type, in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels * 2, base_channels, 1, bias=False),
            NormFactory1D.create_norm(norm_type, base_channels),
            nn.ReLU(inplace=True),
        )
        self.enc1 = EncoderBlock1D(base_channels, base_channels * 2, norm_type)
        self.enc2 = EncoderBlock1D(base_channels * 2, base_channels * 4, norm_type)
        self.bottleneck_conv = EncoderBlock1D(base_channels * 4, base_channels * 8, norm_type)

    def forward(self, x):
        global_feat = self.input_proj(x)           # (B, base_channels, L)
        skip1 = self.enc1(global_feat)             # (B, base_channels*2, L/2)
        skip2 = self.enc2(skip1)                   # (B, base_channels*4, L/4)
        bottom = self.bottleneck_conv(skip2)       # (B, base_channels*8, L/8)
        return [skip1, skip2, bottom, global_feat]


# ============================================
# 变分分区层 (1D)
# ============================================

class VariationalPartitionLayer1D(nn.Module):
    """
    输出 q(z|x) 的 logits，并提供 KL(q(z) || p(z))。
    输入为低频模型（可包含先验空间信息），网络内部使用空洞卷积扩大感受野。
    """
    def __init__(self, in_channels, num_experts=3, norm_type='batch'):
        super().__init__()
        self.num_experts = num_experts
        hidden_dim = min(max(in_channels * 2, num_experts * 4), 24)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1, bias=False),
            NormFactory1D.create_norm(norm_type, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 3, dilation=7, padding=7, bias=False),
            NormFactory1D.create_norm(norm_type, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, num_experts, 1, bias=True),
        )

    def forward(self, low_freq, prior=None):
        """
        Args:
            low_freq: (B, C_low, L)  低频模型
            prior:    (B, K, L) 可选的先验空间分布 (例如各向异性正则提供的概率图)
        Returns:
            logits: (B, K, L)
            kl_z:   标量 KL 散度
        """
        logits = self.net(low_freq)                     # (B, K, L)
        probs = F.softmax(logits, dim=1)                # (B, K, L)
        log_probs = F.log_softmax(logits, dim=1)        # (B, K, L)

        if prior is not None:
            log_prior = torch.log(prior + 1e-12)        # (B, K, L)
        else:
            log_prior = -math.log(self.num_experts) * torch.ones_like(log_probs)

        kl_per_pixel = (probs * (log_probs - log_prior)).sum(dim=1)  # (B, L)
        kl_z = kl_per_pixel.mean()
        return logits, kl_z


# ============================================
# 共享专家解码器 (1D)
# ============================================

class SharedExpertDecoder1D(nn.Module):
    """
    多专家解码器，共享大部分卷积和上采样路径，每个专家拥有独立的适配器和输出卷积。
    输出每个专家的 q(θ|z=k,x) 的均值和对数方差。
    支持通过 prior_means / prior_logvars 初始化输出偏置，以利用先验知识加速收敛。
    先验使用对角方差。
    """
    def __init__(self, skip_channels, num_experts, norm_type='group',
                 prior_means=None, prior_logvars=None):
        super().__init__()
        self.num_experts = num_experts
        c1, c2, c3, c4 = skip_channels   # c1: skip1, c2: skip2, c3: bottom, c4: global_feat

        # 共享的上采样转置卷积（3 个阶段）
        self.up_bottom = nn.ConvTranspose1d(c3, c3, 4, 2, 1, bias=False)
        self.up_mid = nn.ConvTranspose1d(c2, c2, 4, 2, 1, bias=False)
        self.up_final = nn.ConvTranspose1d(c1, c1, 4, 2, 1, bias=False)

        # 共享的卷积块（融合不同尺度特征）
        self.conv2 = BasicBlock1D(c3 + c2, c2, use_attention=False, norm_type=norm_type, dilation=3)
        self.conv1 = BasicBlock1D(c2 + c1, c1, use_attention=False, norm_type=norm_type, dilation=3)

        # 全局残差分支：直接从 global_feat 映射到 6 通道输出（所有专家共享）
        self.global_residual = nn.Conv1d(c4, 6, 1, bias=False)

        # 每个专家的适配器（轻量卷积 + 归一化）
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(c1, c1, 3, dilation=3, padding=3, bias=False),
                NormFactory1D.create_norm(norm_type, c1),
            ) for _ in range(num_experts)
        ])
        # 每个专家的输出卷积 (1x1 卷积，输出 6 通道)
        self.out_convs = nn.ModuleList([
            nn.Conv1d(c1, 6, 1, bias=True) for _ in range(num_experts)
        ])

        # 利用先验初始化输出偏置
        if prior_means is not None and prior_logvars is not None:
            assert prior_means.shape == (num_experts, 3) and prior_logvars.shape == (num_experts, 3)
            with torch.no_grad():
                for k in range(num_experts):
                    self.out_convs[k].bias.data[:3] = prior_means[k]
                    self.out_convs[k].bias.data[3:] = prior_logvars[k]

    @staticmethod
    def _upsample_conv(x, target_size, conv):
        """转置卷积上采样，再插值到精确长度。"""
        x = conv(x)
        if x.shape[-1] != target_size:
            x = F.interpolate(x, size=target_size, mode='linear', align_corners=False)
        return x

    def forward(self, features, target_size, z_soft):
        """
        Args:
            features: [skip1, skip2, bottom, global_feat]
            target_size: 最终输出序列长度 L
            z_soft: (B, K, L) 软分配概率
        Returns:
            (B, K, 6, L) 每个专家的 6 通道输出 (mu 3 + logvar 3)
        """
        skip1, skip2, bottom, global_feat = features
        global_out = self.global_residual(global_feat)          # (B, 6, L)

        up_bottom = self._upsample_conv(bottom, skip2.shape[-1], self.up_bottom)

        mus, logvars = [], []
        for k in range(self.num_experts):
            w = z_soft[:, k:k+1]  # (B,1,L)

            # ---- 融合 skip2 ----
            w_skip2 = F.interpolate(w, size=skip2.shape[-1], mode='linear', align_corners=False)
            skip2_weighted = skip2 + skip2 * w_skip2
            x = torch.cat([up_bottom, skip2_weighted], dim=1)
            x = self.conv2(x)                                  # (B, c2, L/4)

            # ---- 融合 skip1 ----
            w_skip1 = F.interpolate(w, size=skip1.shape[-1], mode='linear', align_corners=False)
            skip1_weighted = skip1 + skip1 * w_skip1
            x_up = self._upsample_conv(x, skip1.shape[-1], self.up_mid)
            x = torch.cat([x_up, skip1_weighted], dim=1)
            x = self.conv1(x)                                  # (B, c1, L/2)

            # ---- 最终上采样到目标长度，并加上专家概率调制 ----
            x = self._upsample_conv(x, target_size, self.up_final)
            x = x + x * w                                      # 残差式调制

            # ---- 专家专属处理 ----
            x = self.adapters[k](x)
            out = self.out_convs[k](x) + global_out            # (B, 6, L)
            if out.shape[-1] != target_size:
                out = F.interpolate(out, size=target_size, mode='linear', align_corners=False)

            mu = out[:, :3]
            logvar = out[:, 3:].clamp(min=-5, max=2)           # 限制对数方差范围
            mus.append(mu)
            logvars.append(logvar)

        mus = torch.stack(mus, dim=1)          # (B, K, 3, L)
        logvars = torch.stack(logvars, dim=1)
        return torch.cat([mus, logvars], dim=2)   # (B, K, 6, L)


# ============================================
# 主模型：变分贝叶斯混合专家 (1D)
# ============================================

class Geo_VBMILE_1D(nn.Module):
    """
    一维变分贝叶斯混合专家地震反演模型。
    输入：地震角道集 (B, seismic_channels, L) 和低频模型 (B, low_freq_channels, L)
    输出：弹性参数 (Vp, Vs, ρ) 的边际分布（均值 + 对数方差），以及后验采样样本等。
    """
    def __init__(self,
                 seismic_channels=6,
                 low_freq_channels=3,
                 base_channels=16,
                 norm_type='group',
                 num_experts=4,
                 initial_temperature=1.0,
                 final_temperature=0.1,
                 prior_means=None,
                 prior_vars=None,
                 prior_spatial=None):   # prior_spatial 暂未使用，保留接口
        super().__init__()
        self.num_experts = num_experts
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.low_freq_channels = low_freq_channels
        self.register_buffer("current_temperature", torch.tensor(initial_temperature))

        # 参数先验（每个专家 3 个参数：Vp, Vs, Rho）
        if prior_means is not None:
            assert prior_vars is not None
            self.register_buffer("prior_means", prior_means)          # (K, 3)
            self.register_buffer("prior_vars", prior_vars)            # (K, 3)
            prior_logvars = torch.log(prior_vars)                     # (K, 3)
        else:
            self.register_buffer("prior_means", torch.zeros(num_experts, 3))
            self.register_buffer("prior_vars", torch.ones(num_experts, 3))
            prior_logvars = torch.zeros(num_experts, 3)

        # 编码器（输入：地震 + 低频）
        encoder_in = seismic_channels + low_freq_channels
        self.encoder = UNetEncoder1D(encoder_in, base_channels, norm_type)
        self.skip_channels = [base_channels*2, base_channels*4, base_channels*8, base_channels]

        # 分区层（仅用低频模型推断专家分配）
        self.partition_layer = VariationalPartitionLayer1D(
            low_freq_channels, num_experts, norm_type=norm_type
        )

        # 共享专家解码器
        self.experts = SharedExpertDecoder1D(
            skip_channels=self.skip_channels,
            num_experts=num_experts,
            norm_type=norm_type,
            prior_means=prior_means,
            prior_logvars=prior_logvars
        )

    def set_temperature(self, epoch, total_epochs):
        """余弦退火更新 Gumbel-Softmax 温度。"""
        if total_epochs <= 0:
            return
        cos_val = (1 + math.cos(math.pi * epoch / total_epochs)) / 2
        new_temp = self.final_temperature + (self.initial_temperature - self.final_temperature) * cos_val
        self.current_temperature.fill_(new_temp)

    def encode(self, seismic, low_freq):
        """编码器前向：拼接地震与低频模型。"""
        encoder_input = torch.cat([seismic, low_freq], dim=1)
        return self.encoder(encoder_input)

    def sample_theta(self, expert_mus, expert_logvars, logits, tau, hard=True):
        """从混合后验 q(θ|x) 中采样一个样本（Gumbel-Softmax + 重参数化）。"""
        z_hard = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)
        eps = torch.randn_like(expert_mus)
        theta_samples = expert_mus + torch.exp(0.5 * expert_logvars) * eps
        theta = (z_hard.unsqueeze(2) * theta_samples).sum(dim=1)
        return theta

    def forward(self, seismic, low_freq_model=None, prior=None, num_samples=1):
        """
        Args:
            seismic: (B, seismic_channels, L)
            low_freq_model: (B, low_freq_channels, L) 或 None (自动补零)
            prior: (B, K, L) 可选的先验分布，用于分区层 KL 计算
            num_samples: 是否从后验采样 theta (1: 采样, 0: 不采样)
        Returns:
            字典包含边际预测、采样 theta、专家分布、各项 KL 及中间特征。
        """
        B, _, L = seismic.shape
        if low_freq_model is None:
            low_freq_model = torch.zeros(B, self.low_freq_channels, L, device=seismic.device)

        features = self.encode(seismic, low_freq_model)          # [skip1, skip2, bottom, global]

        logits, kl_z = self.partition_layer(low_freq_model, prior=prior)
        z_soft = F.softmax(logits, dim=1)                        # (B, K, L)

        expert_out = self.experts(features, target_size=L, z_soft=z_soft)   # (B, K, 6, L)
        expert_mus = expert_out[:, :, :3]                        # (B, K, 3, L)
        expert_logvars = expert_out[:, :, 3:]                    # (B, K, 3, L)

        # ---- KL(θ) 计算 ----
        var = torch.exp(expert_logvars)
        prior_means_exp = self.prior_means[None, :, :, None]     # (1, K, 3, 1)
        prior_vars_exp = self.prior_vars[None, :, :, None]       # (1, K, 3, 1)
        kl_theta_per_expert = 0.5 * (
            (var / prior_vars_exp) +
            ((expert_mus - prior_means_exp) ** 2) / prior_vars_exp -
            1.0 - expert_logvars + torch.log(prior_vars_exp)
        ).sum(dim=2)                                             # (B, K, L)
        kl_theta = (z_soft * kl_theta_per_expert).sum(dim=1).mean()

        # ---- 边际分布 q(θ|x) ----
        mu_marginal = (z_soft.unsqueeze(2) * expert_mus).sum(dim=1)            # (B, 3, L)
        second_moment = (z_soft.unsqueeze(2) * (var + expert_mus**2)).sum(dim=1)
        var_marginal = second_moment - mu_marginal**2
        var_marginal = torch.clamp(var_marginal, min=1e-6)
        logvar_marginal = torch.log(var_marginal)
        final_pred = torch.cat([mu_marginal, logvar_marginal], dim=1)          # (B, 6, L)

        # ---- 可选采样 ----
        theta_sample = None
        if num_samples > 0:
            theta_sample = self.sample_theta(
                expert_mus, expert_logvars, logits, self.current_temperature, hard=True
            )

        return {
            "final_pred": final_pred,
            "theta_sample": theta_sample,
            "expert_mus": expert_mus,
            "expert_logvars": expert_logvars,
            "z_soft": z_soft,
            "logits": logits,
            "kl_z": kl_z,
            "kl_theta": kl_theta,
            "skip1": features[0],
            "skip2": features[1],
            "bottleneck": features[2],
            "global": features[3]
        }

