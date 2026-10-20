"""Seismic AVA forward modelling: Ricker wavelet + Aki-Richards reflectivity + convolution.

The only difference between the two dimensions is the sample layout — ``(B,3,L)`` vs
``(B,3,H,W)``:

* 1-D — the reflectivity is convolved along the trace (deepest axis), trace by trace;
* 2-D — every trace column is convolved independently (``R`` has shape
  ``(B, L, W, n_angles)``).

The wavelet is always generated as ``ricker_wavelet(dt=dt, f=f, length=length_sec)`` with
``length_sec`` in seconds, so both dimensions share one convention.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.forward.base import FORWARD_OPERATORS
from src.forward.wavelets import ricker_wavelet  # noqa: F401  (单一来源)

__all__ = ["FORWARD_OPERATORS", "metadata", "ricker_wavelet", "SeismicForward"]


def metadata() -> dict:
    """注册在 models.json 里的可复现信息。"""
    return {
        "forward": "SeismicForward(Aki-Richards) + Ricker 卷积",
        "angles_deg_default": [5, 10, 15, 20, 25, 30],
    }


@FORWARD_OPERATORS.register("synthetic_seismic")
class SeismicForward(nn.Module):
    """基于 Aki-Richards 近似的角度域正演（1D/2D 统一实现）。

    Args:
        wavelet: (1,1,L_w) 的 Ricker 子波，或传入 (dt, f, length_sec) 字典自动生成。
        angles_deg: 角度列表（度）。
        dim: 1（逐道 (B,3,L)）或 2（切片 (B,3,H,W)）；默认按输入张量 3/4 维自动判定。
    """

    def __init__(
        self,
        wavelet: Optional[Union[torch.Tensor, dict]] = None,
        angles_deg: Sequence[float] = (5, 10, 15, 20, 25, 30),
        dim: int = 1,
    ) -> None:
        super().__init__()
        if wavelet is None:
            w = {"dt": 0.002, "f": 20.0, "length": 0.1}
        elif isinstance(wavelet, dict):
            w = dict(wavelet)
        else:
            w = wavelet
        if isinstance(w, dict):
            wavelet_t = ricker_wavelet(dt=w.get("dt", 0.002), f=w.get("f", 20.0),
                                       length=w.get("length", 0.1))
        else:
            wavelet_t = w
        if wavelet_t.dim() == 1:
            wavelet_t = wavelet_t.view(1, 1, -1)
        elif wavelet_t.dim() == 2:
            wavelet_t = wavelet_t.unsqueeze(0)
        assert wavelet_t.size(1) == 1, "wavelet must have in_channels=1"
        self.register_buffer("wavelet", wavelet_t)

        self.dim = int(dim)
        deg = torch.as_tensor(list(angles_deg), dtype=torch.float32)
        self.register_buffer("angles_rad", deg * (np.pi / 180.0))
        self.n_angles = len(self.angles_rad)

    def aki_richards_reflectivity(
        self, vp1, vs1, rho1, vp2, vs2, rho2, theta_rad: torch.Tensor
    ) -> torch.Tensor:
        """Aki-Richards 近似反射系数；theta_rad 形状含末维角度。"""
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

        p2 = (sin_theta / (vp_avg.unsqueeze(-1) + eps)) ** 2
        term1 = 0.5 * (1 - 4 * vs_avg.unsqueeze(-1) ** 2 * p2) * (
            drho.unsqueeze(-1) / (rho_avg.unsqueeze(-1) + eps)
        )
        term2 = 0.5 * cos2_inv * (dvp.unsqueeze(-1) / (vp_avg.unsqueeze(-1) + eps))
        term3 = -4 * vs_avg.unsqueeze(-1) ** 2 * p2 * (
            dvs.unsqueeze(-1) / (vs_avg.unsqueeze(-1) + eps)
        )
        return term1 + term2 + term3

    def forward(self, mu_phys: torch.Tensor) -> torch.Tensor:
        """物理空间弹性参数 (B,3,*) -> 合成角道集 (B,n_angles,*)。"""
        B, C = mu_phys.shape[:2]
        vp, vs, rho = mu_phys[:, 0], mu_phys[:, 1], mu_phys[:, 2]

        if self.dim == 1 or mu_phys.dim() == 3:
            L = mu_phys.shape[-1]
            vp1, vp2 = vp[:, :-1], vp[:, 1:]
            vs1, vs2 = vs[:, :-1], vs[:, 1:]
            rho1, rho2 = rho[:, :-1], rho[:, 1:]
            Rpp = self.aki_richards_reflectivity(vp1, vs1, rho1, vp2, vs2, rho2, self.angles_rad)
            Rpp = Rpp.view(B, L - 1, self.n_angles)
            Rpp_padded = F.pad(Rpp, (0, 0, 1, 0))                      # (B, L, n_angles)
            R_reshaped = Rpp_padded.permute(0, 2, 1).reshape(B * self.n_angles, 1, L)
            padding = self.wavelet.shape[-1] // 2
            syn_reshaped = F.conv1d(R_reshaped, self.wavelet, padding=padding)
            return syn_reshaped.view(B, self.n_angles, L)
        else:  # 2D
            H, W = mu_phys.shape[2:]
            vp1, vp2 = vp[:, :-1, :], vp[:, 1:, :]
            vs1, vs2 = vs[:, :-1, :], vs[:, 1:, :]
            rho1, rho2 = rho[:, :-1, :], rho[:, 1:, :]
            Rpp = self.aki_richards_reflectivity(vp1, vs1, rho1, vp2, vs2, rho2, self.angles_rad)
            Rpp_padded = torch.zeros(B, H, W, self.n_angles, device=mu_phys.device)
            Rpp_padded[:, 1:, :, :] = Rpp
            R_reshaped = Rpp_padded.permute(0, 3, 2, 1).reshape(B * self.n_angles * W, 1, H)
            padding = self.wavelet.shape[-1] // 2
            syn_reshaped = F.conv1d(R_reshaped, self.wavelet, padding=padding)
            syn = syn_reshaped.view(B, self.n_angles, W, H).permute(0, 1, 3, 2)
            return syn