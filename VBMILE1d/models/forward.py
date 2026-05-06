import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

def ricker_wavelet(dt=0.002, f=30, length=0.1):
    """生成 Ricker 子波，返回形状 (1,1,L) 的 torch.Tensor"""
    t = np.arange(-length/2, length/2 + dt, dt)
    w = (1 - 2*(np.pi*f*t)**2) * np.exp(-(np.pi*f*t)**2)
    return torch.tensor(w, dtype=torch.float32).view(1, 1, -1)


# ===================== 正演模块 =====================
class SeismicForward1D(nn.Module):
    """基于 Aki‑Richards 近似的角度域正演（1D卷积实现）"""
    def __init__(self, wavelet: torch.Tensor, angles_deg: List[float]):
        super().__init__()
        # 确保小波形状为 (1, 1, L_w)
        if wavelet.dim() == 1:
            wavelet = wavelet.view(1, 1, -1)
        elif wavelet.dim() == 2:
            wavelet = wavelet.unsqueeze(0)
        assert wavelet.size(1) == 1, "wavelet must have in_channels=1"
        self.register_buffer('wavelet', wavelet)

        angles_rad = torch.tensor(angles_deg, dtype=torch.float32) * (np.pi / 180.0)
        self.register_buffer('angles_rad', angles_rad)
        self.n_angles = len(angles_rad)

    def aki_richards_reflectivity(self, vp1, vs1, rho1, vp2, vs2, rho2, theta_rad):
        vp_avg = (vp1 + vp2) / 2.0
        vs_avg = (vs1 + vs2) / 2.0
        rho_avg = (rho1 + rho2) / 2.0
        dvp = vp2 - vp1
        dvs = vs2 - vs1
        drho = rho2 - rho1
        eps = 1e-6
        vp_avg = torch.clamp(vp_avg, min=eps)
        vs_avg = torch.clamp(vs_avg, min=eps)
        rho_avg = torch.clamp(rho_avg, min=eps)

        theta_rad = theta_rad.view(1, 1, 1, -1)
        sin_theta = torch.sin(theta_rad)
        cos_theta = torch.cos(theta_rad)
        cos2_inv = 1.0 / (cos_theta**2 + eps)

        p2 = (sin_theta / (vp_avg.unsqueeze(-1) + eps))**2
        term1 = 0.5 * (1 - 4 * (vs_avg.unsqueeze(-1)**2) * p2) * (drho.unsqueeze(-1) / (rho_avg.unsqueeze(-1) + eps))
        term2 = 0.5 * cos2_inv * (dvp.unsqueeze(-1) / (vp_avg.unsqueeze(-1) + eps))
        term3 = -4 * (vs_avg.unsqueeze(-1)**2) * p2 * (dvs.unsqueeze(-1) / (vs_avg.unsqueeze(-1) + eps))
        Rpp = term1 + term2 + term3
        return Rpp

    def forward(self, mu_phys):
        B, C, L = mu_phys.shape
        vp = mu_phys[:, 0]
        vs = mu_phys[:, 1]
        rho = mu_phys[:, 2]

        vp1, vp2 = vp[:, :-1], vp[:, 1:]
        vs1, vs2 = vs[:, :-1], vs[:, 1:]
        rho1, rho2 = rho[:, :-1], rho[:, 1:]

        Rpp = self.aki_richards_reflectivity(vp1, vs1, rho1, vp2, vs2, rho2, self.angles_rad)
        Rpp = Rpp.view(B, L-1, self.n_angles)          # (B, L-1, n_angles)

        # 在时间/深度维度左侧补零
        Rpp_padded = F.pad(Rpp, (0, 0, 1, 0))           # (B, L, n_angles)

        # 卷积
        R_reshaped = Rpp_padded.permute(0, 2, 1).reshape(B * self.n_angles, 1, L)
        syn_reshaped = F.conv1d(R_reshaped, self.wavelet, padding=self.wavelet.shape[-1] // 2)
        syn = syn_reshaped.view(B, self.n_angles, L)
        return syn

