import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def ricker_wavelet(dt=0.002, f=30, length=0.1):
    t = np.arange(-length/2, length/2 + dt, dt)
    w = (1 - 2*(np.pi*f*t)**2) * np.exp(-(np.pi*f*t)**2)
    return torch.tensor(w, dtype=torch.float32).view(1, 1, -1)

class SeismicForward(nn.Module):
    def __init__(self, wavelet: torch.Tensor, angles_deg: List[float]):
        super().__init__()
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
        Rpp = torch.clamp(Rpp, -0.5, 0.5)
        return Rpp

    def forward(self, mu_phys: torch.Tensor) -> torch.Tensor:
        B, C, H, W = mu_phys.shape
        device = mu_phys.device
        vp = mu_phys[:, 0]; vs = mu_phys[:, 1]; rho = mu_phys[:, 2]

        vp1 = vp[:, :-1, :]; vp2 = vp[:, 1:, :]
        vs1 = vs[:, :-1, :]; vs2 = vs[:, 1:, :]
        rho1 = rho[:, :-1, :]; rho2 = rho[:, 1:, :]

        Rpp = self.aki_richards_reflectivity(vp1, vs1, rho1, vp2, vs2, rho2, self.angles_rad)
        Rpp_padded = torch.zeros(B, H, W, self.n_angles, device=device)
        Rpp_padded[:, 1:, :, :] = Rpp

        # 卷积正演
        R_reshaped = Rpp_padded.permute(0, 3, 2, 1).reshape(B * self.n_angles * W, 1, H)
        wavelet_len = self.wavelet.shape[-1]
        padding = wavelet_len // 2
        syn_reshaped = F.conv1d(R_reshaped, self.wavelet, padding=padding)
        syn = syn_reshaped.view(B, self.n_angles, W, H).permute(0, 1, 3, 2)
        return syn
    
    