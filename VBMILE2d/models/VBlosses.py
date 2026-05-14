# ============================================================================
# 损失函数：变分下界 ELBO（2D）
# ============================================================================
import math
from typing import Optional, Union, List, Tuple
import torch
import torch.nn as nn

class VBELBO2D(nn.Module):
    def __init__(self,
                 seismic_obs_noise_init: float = 1.0,
                 well_obs_noise_init: float = 0.1,
                 clip_diff: float = 10.0,
                 kl_z_weight: float = 1.0,
                 kl_theta_weight: float = 1.0,
                 expert_lambdas: Optional[Union[List[float], torch.Tensor]] = None,
                 prior_means: Optional[torch.Tensor] = None,
                 prior_vars: Optional[torch.Tensor] = None):
        super().__init__()
        self.log_seismic_var = nn.Parameter(torch.tensor(math.log(seismic_obs_noise_init ** 2), requires_grad=False))
        self.log_well_var = nn.Parameter(torch.tensor(math.log(well_obs_noise_init ** 2), requires_grad=False))
        self.clip_diff = clip_diff
        self.log_two_pi = math.log(2 * math.pi)

        self.kl_z_weight = kl_z_weight
        self.kl_theta_weight = kl_theta_weight
        self.expert_lambdas = expert_lambdas

    @property
    def seismic_var(self) -> torch.Tensor:
        return torch.exp(self.log_seismic_var)

    @property
    def well_var(self) -> torch.Tensor:
        return torch.exp(self.log_well_var)

    def _nll_well(self, mu: torch.Tensor, logvar: torch.Tensor,
                  target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        var_theta = torch.exp(logvar)
        total_var = var_theta + self.well_var + 1e-4
        diff = (target - mu).clamp(-self.clip_diff, self.clip_diff)
        nll = 0.5 * (diff ** 2 / total_var + self.log_two_pi + torch.log(total_var))
        if mask is not None:
            if mask.shape[1] == 1:
                mask = mask.expand_as(nll)
            nll = nll * mask
            return nll.sum() / (mask.sum() + 1e-8)
        return nll.mean()

    def _nll_seismic(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        diff = (syn - obs).clamp(-self.clip_diff, self.clip_diff)
        var = self.seismic_var + 1e-4
        nll = 0.5 * (diff ** 2 / var + self.log_two_pi + torch.log(var))
        nll = torch.nan_to_num(nll, nan=0.0, posinf=1e6, neginf=-1e6)
        return nll.mean()

    def forward(self,
                model_output: dict,
                target: torch.Tensor,
                syn_norm: Optional[torch.Tensor] = None,
                seismic: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        mu = model_output['final_pred'][:, :3]
        logvar = model_output['final_pred'][:, 3:]

        nll_well = self._nll_well(mu, logvar, target, mask)

        seismic_nll = torch.tensor(0.0, device=mu.device)
        if syn_norm is not None and seismic is not None:
            seismic_nll = self._nll_seismic(syn_norm, seismic)

        kl_z = model_output['kl_z']
        kl_theta_per_expert = model_output['kl_theta_per_expert']   # (B, K)

        # 处理专家权重
        lambdas = self.expert_lambdas
        if lambdas is not None:
            # 转换为张量
            if not isinstance(lambdas, torch.Tensor):
                lambdas = torch.tensor(lambdas, device=kl_theta_per_expert.device, dtype=kl_theta_per_expert.dtype)
            # lambdas 形状应为 (K,)
            # kl_theta_per_expert 形状 (B, K) -> 按专家加权平均
            weighted_kl_theta = (kl_theta_per_expert * lambdas.unsqueeze(0)).mean()
        else:
            weighted_kl_theta = kl_theta_per_expert.mean()

        total_loss = (nll_well + 0.1*seismic_nll +
                      self.kl_z_weight * kl_z +
                      self.kl_theta_weight * weighted_kl_theta)  # 正演模块存在子波与模型假设，根据需要加权

        loss_dict = {
            'total': total_loss,
            'nll_well': nll_well,
            'seismic_nll': seismic_nll,
            'kl_z': kl_z,
            'kl_theta': weighted_kl_theta,
            'well_var': self.well_var.detach(),
            'seismic_var': self.seismic_var.detach(),
        }
        return total_loss, loss_dict
    
    