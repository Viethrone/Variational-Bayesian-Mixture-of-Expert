import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# ============================================
# ELBO 损失函数 (1D)
# ============================================

class VBELBOLoss1D(nn.Module):
    """
    负 ELBO 损失，包含：
        - 井似然期望（解析高斯形式，使用预测的均值和方差）
        - 地震似然期望（基于 theta 样本的正演结果，仅观测噪声）
        - KL(q(z)||p(z)) 和 KL(q(θ)||p(θ))
    支持可学习的观测噪声（井、地震）。
    """
    def __init__(self, seismic_obs_noise_init=1.0, well_obs_noise_init=0.05,
                 aniso_weight=0.1, clip_diff=10.0):
        super().__init__()
        self.log_seismic_var = nn.Parameter(torch.tensor(math.log(seismic_obs_noise_init**2)))
        self.log_well_var = nn.Parameter(torch.tensor(math.log(well_obs_noise_init**2)))
        self.clip_diff = clip_diff
        self.log_two_pi = math.log(2 * math.pi)

    @property
    def seismic_var(self):
        return torch.exp(self.log_seismic_var)

    @property
    def well_var(self):
        return torch.exp(self.log_well_var)

    def _nll_well_expectation(self, mu, logvar, target, mask):
        """
        井似然期望：E_q[log p(y|θ)] = N(y; μ, σ_y² + σ_θ²)
        """
        var_theta = torch.exp(logvar)
        total_var = var_theta + self.well_var
        diff = target - mu
        diff = torch.clamp(diff, -self.clip_diff, self.clip_diff)
        nll = 0.5 * (diff**2 / total_var + self.log_two_pi + torch.log(total_var))
        if mask is not None:
            nll = nll * mask
            return nll.sum() / (mask.sum() + 1e-8)
        else:
            return nll.mean()

    def _nll_seismic(self, syn, obs):
        """地震似然：p(d|θ_sample) 的高斯 NLL。"""
        diff = syn - obs
        diff = torch.clamp(diff, -self.clip_diff, self.clip_diff)
        nll = 0.5 * (diff**2 / self.seismic_var + self.log_two_pi + torch.log(self.seismic_var))
        nll = torch.nan_to_num(nll, nan=0.0, posinf=1e6, neginf=-1e6)
        return nll.mean()

    def forward(self, model_output, target, syn_norm=None, seismic=None, target_mask=None):
        """
        Args:
            model_output: Geo_VBMILE_1D 的输出字典
            target: (B, 3, L) 归一化后的弹性参数标签
            syn_norm: (B, n_angles, L) 正演合成记录（来自 theta_sample）
            seismic: (B, n_angles, L) 观测地震数据（归一化）
            target_mask: (B, 3, L) 井标签掩码（1 表示有标签）
        """
        mu = model_output['final_pred'][:, :3]      # (B,3,L)
        logvar = model_output['final_pred'][:, 3:]  # (B,3,L)

        nll_well = self._nll_well_expectation(mu, logvar, target, target_mask)

        seismic_nll = torch.tensor(0.0, device=mu.device)
        if syn_norm is not None and seismic is not None:
            seismic_nll = self._nll_seismic(syn_norm, seismic)

        kl_z = model_output['kl_z']
        kl_theta = model_output['kl_theta']

        # 监控用 MSE（仅在有标签位置）
        if target_mask is not None:
            masked_diff = (mu - target) * target_mask
            valid_pixels = target_mask.sum() + 1e-8
            mse = (masked_diff**2).sum() / valid_pixels
        else:
            mse = torch.tensor(0.0, device=mu.device)

        neg_elbo = nll_well + seismic_nll + 0.05*kl_z + 0.01*kl_theta

        loss_dict = {
            'total': neg_elbo,
            'nll_well': nll_well,
            'seismic_nll': seismic_nll,
            'kl_z': kl_z,
            'kl_theta': kl_theta,
            'well_var': self.well_var.detach(),
            'seismic_var': self.seismic_var.detach(),
            'mse_monitor': mse,
        }
        return neg_elbo, loss_dict
    
