"""Elastic-impedance (EI) forward modelling: Connolly EI + EI reflection + wavelet convolution.

Numerical core:

* Connolly EI: ``EI(theta) = Vp0 Den0 (Vp/Vp0)^a (Vs/Vs0)^b (Den/Den0)^c`` with
  ``a = 1 + tan^2(theta)``, ``b = -8k sin^2(theta)``, ``c = 1 - 4k sin^2(theta)`` and
  ``k = (Vs0/Vp0)^2`` (computed in the log domain for stability);
* EI reflection coefficient ``R = (EI2 - EI1)/(EI2 + EI1)``;
* Aki-Richards / exact Zoeppritz (Aki & Richards, 1980) reflection coefficients;
* Ricker / Ormsby wavelet convolution into angle gathers.

Two interfaces:

* ``ElasticImpedanceForward`` (``nn.Module``, registered as ``elastic_impedance_synth``):
  a batched differentiable implementation usable through ``build_from_config``;
* ``ElasticImpedanceTool`` (a numpy facade): reproduces
  ``elastic_impedance_results.mat`` from a synthetic model, with matplotlib and prints
  removed but ``add_perturbation / synthesize / save`` kept.

Note: the ``method='ei'`` reflection lives in the *elastic-impedance* domain rather than
in the linear-average domain. The inversion path itself uses Aki-Richards
(``SeismicForward``); this module serves data synthesis.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.forward.base import FORWARD_OPERATORS
from src.forward.wavelets import ormsby_wavelet, ricker_wavelet

__all__ = [
    "FORWARD_OPERATORS",
    "connolly_elastic_impedance",
    "elastic_impedance_reflection",
    "aki_richards_reflection",
    "exact_zoeppritz_pp",
    "ElasticImpedanceForward",
    "ElasticImpedanceTool",
]


# ---------------------------------------------------------------------------
# 无量纲公式（numpy + torch 通用；torch 统一输 (B,n_angles,*)）
# ---------------------------------------------------------------------------

def connolly_elastic_impedance(
    vp: torch.Tensor, vs: torch.Tensor, den: torch.Tensor,
    angles_deg: torch.Tensor, vp0: float = 1.0, vs0: float = 1.0, den0: float = 1.0,
) -> torch.Tensor:
    """Connolly 弹性阻抗（对数域），返回与 vp 同空间形状、末维为角度。

    vp/vs/den 支持任意前导形状 (..., nz) 或 (..., nz, nx)；角度为末维。
    """
    eps = 1e-6
    vp = torch.clamp(vp, min=eps)
    vs = torch.clamp(vs, min=eps)
    den = torch.clamp(den, min=eps)
    vp0 = max(float(vp0), eps)
    vs0 = max(float(vs0), eps)
    den0 = max(float(den0), eps)

    k = (vs0 / vp0) ** 2
    th = angles_deg * (np.pi / 180.0)
    tan2 = torch.tan(th) ** 2                       # (n_angles,)
    sin2 = torch.sin(th) ** 2
    a = 1 + tan2
    b = -8 * k * sin2
    c = 1 - 4 * k * sin2

    log_vp = torch.log(vp / vp0)
    log_vs = torch.log(vs / vs0)
    log_den = torch.log(den / den0)
    # 广播：…把角度扩展为末维
    log_ei = (a * log_vp.unsqueeze(-1) + b * log_vs.unsqueeze(-1)
              + c * log_den.unsqueeze(-1))
    return (vp0 * den0) * torch.exp(log_ei)


def elastic_impedance_reflection(ei: torch.Tensor) -> torch.Tensor:
    """EI 反射系数：(EI2-EI1)/(EI2+EI1)，首采样为 0。ei 形状 (..., nz, n_angles)。"""
    ei1 = ei[..., :-1, :]
    ei2 = ei[..., 1:, :]
    rc = (ei2 - ei1) / (ei2 + ei1 + 1e-12)
    out = torch.zeros_like(ei)
    out[..., 1:, :] = rc
    return torch.clamp(out, -0.5, 0.5)


def _angle_view(th_rad: torch.Tensor, n_leading_dims: int) -> torch.Tensor:
    """把角度张量 (A,) 视图为前导 n_leading_dims 个单例维 + 末维 A（接口轴之后）。"""
    return th_rad.reshape((1,) * n_leading_dims + (-1,))


def aki_richards_reflection(
    vp: torch.Tensor, vs: torch.Tensor, den: torch.Tensor, angles_deg: torch.Tensor,
) -> torch.Tensor:
    """Aki-Richards 近似 Rpp；vp/vs/den 形状 (..., nz[, nx])，角度为末维。"""
    eps = 1e-6
    th = angles_deg * (np.pi / 180.0)
    av = _angle_view(th, vp.ndim)                    # (1,..,1, A)
    vp1, vp2 = vp[..., :-1], vp[..., 1:]
    vs1, vs2 = vs[..., :-1], vs[..., 1:]
    de1, de2 = den[..., :-1], den[..., 1:]
    vpa = torch.clamp((vp1 + vp2) / 2, min=eps).unsqueeze(-1)   # (..., nz-1, 1)
    vsa = torch.clamp((vs1 + vs2) / 2, min=eps).unsqueeze(-1)
    dea = torch.clamp((de1 + de2) / 2, min=eps).unsqueeze(-1)
    dvp, dvs, dde = (vp2 - vp1).unsqueeze(-1), (vs2 - vs1).unsqueeze(-1), (de2 - de1).unsqueeze(-1)

    cos2 = torch.cos(av) ** 2 + eps
    sin2 = torch.sin(av) ** 2
    p2 = sin2 / (vpa + eps) ** 2                     # (..., nz-1, A)
    term1 = 0.5 * (1 - 4 * vsa ** 2 * p2) * (dde / dea)
    term2 = 0.5 * (1 / cos2) * (dvp / vpa)
    term3 = -4 * vsa ** 2 * p2 * (dvs / vsa)
    r = term1 + term2 + term3
    rc_full = torch.zeros(vp.shape + (len(th),), device=vp.device)
    rc_full[..., 1:, :] = r
    return torch.clamp(rc_full, -0.5, 0.5)


def exact_zoeppritz_pp(
    vp: torch.Tensor, vs: torch.Tensor, den: torch.Tensor, angles_deg: torch.Tensor,
) -> torch.Tensor:
    """Aki & Richards (1980) 精确 PP 反射系数（Zoeppritz 显式解）。

    vp/vs/den 形状 (..., nz[, nx])；角度末维。临界角/虚折射角经 arcsin clip 处理。
    """
    # 物理有效域护栏（反演阶段 theta 采样可能越界到 0/负值 → 归一化空间极端样本
    # 会触发临界角/相消奇异；宽上界不伤正常岩性取值，grad 在域内无损传导）
    vp = torch.clamp(vp, 300.0, 7000.0)
    vs = torch.clamp(vs, 150.0, 4000.0)
    den = torch.clamp(den, 1.5, 3.5)
    th = angles_deg * (np.pi / 180.0)
    av = _angle_view(th, vp.ndim)
    vp1, vp2 = torch.clamp(vp[..., :-1], min=1e-6).unsqueeze(-1), torch.clamp(vp[..., 1:], min=1e-6).unsqueeze(-1)
    vs1, vs2 = torch.clamp(vs[..., :-1], min=1e-6).unsqueeze(-1), torch.clamp(vs[..., 1:], min=1e-6).unsqueeze(-1)
    de1, de2 = torch.clamp(den[..., :-1], min=1e-6).unsqueeze(-1), torch.clamp(den[..., 1:], min=1e-6).unsqueeze(-1)

    sin_t = torch.sin(av)                            # (1,..,1, A)
    cos1 = torch.cos(av)
    p = sin_t / vp1
    sin_th2 = (vp2 / vp1) * sin_t
    sin_sp1 = (vs1 / vp1) * sin_t
    sin_sp2 = (vs2 / vp1) * sin_t
    cos_th2 = torch.cos(torch.arcsin(torch.clamp(sin_th2, -1, 1)))
    cos_sp1 = torch.cos(torch.arcsin(torch.clamp(sin_sp1, -1, 1)))
    cos_sp2 = torch.cos(torch.arcsin(torch.clamp(sin_sp2, -1, 1)))

    p2 = p ** 2
    v1s2 = vs1 ** 2
    v2s2 = vs2 ** 2
    a = de2 * (1 - 2 * v2s2 * p2) - de1 * (1 - 2 * v1s2 * p2)
    b = de2 * (1 - 2 * v2s2 * p2) + 2 * de1 * v1s2 * p2
    c = de1 * (1 - 2 * v1s2 * p2) + 2 * de2 * v2s2 * p2
    d = 2 * (de2 * v2s2 - de1 * v1s2)

    E = (b * cos1 / vp1) + (c * cos_th2 / vp2)
    F = (b * cos_sp1 / vs1) + (c * cos_sp2 / vs2)
    G = a - d * cos1 * cos_sp2 / (vp1 * vs2)
    H = a - d * cos_th2 * cos_sp1 / (vp2 * vs1)
    D = E * F + G * H * p2
    num = ((b * cos1 / vp1) - (c * cos_th2 / vp2)) * F \
        - (a + d * cos1 * cos_sp2 / (vp1 * vs2)) * H * p2
    r = torch.real(num / torch.clamp(D, min=1e-12))
    # 数值相消兜底：D≈0 且 num≈0 的 0/0 处反射无物理意义，置 0（梯度截断为该像素 0）
    r = torch.nan_to_num(r, nan=0.0, posinf=0.5, neginf=-0.5)
    rc_full = torch.zeros(vp.shape + (len(th),), device=vp.device)
    rc_full[..., 1:, :] = r
    return torch.clamp(rc_full, -0.5, 0.5)


# ---------------------------------------------------------------------------
# torch 模块（注册 elastic_impedance_synth）
# ---------------------------------------------------------------------------

@FORWARD_OPERATORS.register("elastic_impedance_synth")
class ElasticImpedanceForward(nn.Module):
    """EI 正演：弹性参数场 (B,3,L)/(B,3,H,W) -> 合成角道集 (B,n_angles,*)。

    Args:
        wavelet: (1,1,L_w) 子波，或 ``{"ricker": {dt, f, length}}`` /
            ``{"ormsby": {dt, f_hz, length}}`` 字典自动生成。
        angles_deg: 角度列表（度）。
        vp0 / vs0 / den0: 弹性阻抗归一化参考值；None 时用默认值
            （Vp0=4000, Vs0=2300, Den0=2.3 量级，数据生成时最好显式给出）。
        reflection: 'ei'（Connolly）/ 'aki' / 'zoeppritz'。
        dim: 1（(B,3,L)）或 2（(B,3,H,W)）。
    """

    def __init__(
        self,
        wavelet: Optional[object] = None,          # Tensor / dict
        angles_deg: Sequence[float] = (0, 5, 10, 15, 20, 25, 30),
        vp0: Optional[float] = None,
        vs0: Optional[float] = None,
        den0: Optional[float] = None,
        reflection: str = "ei",
        dim: int = 2,
    ) -> None:
        super().__init__()
        if wavelet is None:
            wt = ricker_wavelet(dt=0.002, f=20.0, length=0.1)
        elif isinstance(wavelet, dict):
            if "ricker" in wavelet:
                wt = ricker_wavelet(**{k: v for k, v in wavelet["ricker"].items() if k != "name"})
            elif "ormsby" in wavelet:
                wt = ormsby_wavelet(**{k: v for k, v in wavelet["ormsby"].items() if k != "name"})
            else:
                raise ValueError(f"wavelet 字典须含 'ricker' 或 'ormsby' 键，收到 {sorted(wavelet)}")
        else:
            wt = torch.as_tensor(wavelet, dtype=torch.float32)
        if wt.dim() == 1:
            wt = wt.view(1, 1, -1)
        self.register_buffer("wavelet", wt)

        deg = torch.as_tensor(list(angles_deg), dtype=torch.float32)
        self.register_buffer("angles_deg", deg)
        self.n_angles = len(deg)

        self.reflection = str(reflection).lower()
        if self.reflection not in ("ei", "aki", "zoeppritz"):
            raise ValueError(f"reflection 可选 ei/aki/zoeppritz，收到 {self.reflection!r}")

        # EI 参考值（仅 reflection='ei' 使用）
        self.vp0 = float(vp0) if vp0 is not None else 4000.0
        self.vs0 = float(vs0) if vs0 is not None else 2300.0
        self.den0 = float(den0) if den0 is not None else 2.3
        self.dim = int(dim)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        B, C = mu.shape[:2]
        vp, vs, den = mu[:, 0], mu[:, 1], mu[:, 2]
        if self.reflection == "ei":
            ei = connolly_elastic_impedance(vp, vs, den, self.angles_deg,
                                            vp0=self.vp0, vs0=self.vs0, den0=self.den0)
            rc = elastic_impedance_reflection(ei)        # (B, L[, W], A)
        elif self.reflection == "zoeppritz":
            rc = exact_zoeppritz_pp(vp, vs, den, self.angles_deg)
        else:
            rc = aki_richards_reflection(vp, vs, den, self.angles_deg)

        n_angles = self.n_angles
        if self.dim == 1 or mu.dim() == 3:
            L = mu.shape[-1]
            rc = rc.view(B, L, n_angles)                 # (B,L,A)
            rc_p = rc.permute(0, 2, 1).reshape(B * n_angles, 1, L)
            pad = self.wavelet.shape[-1] // 2
            syn = F.conv1d(rc_p, self.wavelet, padding=pad)
            return syn.view(B, n_angles, L)
        H, W = mu.shape[2:]
        rc = rc.view(B, H, W, n_angles)                  # legacy 布局：深度在前
        rc_p = rc.permute(0, 3, 2, 1).reshape(B * n_angles * W, 1, H)
        pad = self.wavelet.shape[-1] // 2
        syn = F.conv1d(rc_p, self.wavelet, padding=pad)
        return syn.view(B, n_angles, W, H).permute(0, 1, 3, 2)

    def __repr__(self) -> str:
        return (f"ElasticImpedanceForward(reflection={self.reflection}, "
                f"angles={self.angles_deg.tolist()}, dim={self.dim})")


# ---------------------------------------------------------------------------
# numpy facade：legacy ElasticImpedanceTool 数值平移版（数据合成工具用）
# ---------------------------------------------------------------------------

class ElasticImpedanceTool:
    """弹性阻抗 + 角度域合成地震记录工具（legacy ElasticImpedanceTool 平移版）。

    Args:
        Vp / Vs / Den: 弹性参数场 (nz, nx)。
        Vp0 / Vs0 / Den0: EI 参考值；None 时取各场均值（与 legacy 一致）。
        resolution_ratio: 构造时的降采样因子（≥1）。
    """

    def __init__(self, Vp, Vs, Den, Vp0=None, Vs0=None, Den0=None, resolution_ratio: int = 1):
        rr = int(resolution_ratio)
        self.Vp = np.asarray(Vp, dtype=np.float64)[::rr, ::rr]
        self.Vs = np.asarray(Vs, dtype=np.float64)[::rr, ::rr]
        self.Den = np.asarray(Den, dtype=np.float64)[::rr, ::rr]
        assert self.Vp.shape == self.Vs.shape == self.Den.shape
        assert self.Vp.ndim == 2
        self.nz, self.nx = self.Vp.shape
        self.Vp0 = float(Vp0) if Vp0 is not None else float(self.Vp.mean())
        self.Vs0 = float(Vs0) if Vs0 is not None else float(self.Vs.mean())
        self.Den0 = float(Den0) if Den0 is not None else float(self.Den.mean())
        self.k = (self.Vs0 / self.Vp0) ** 2

    def add_perturbation(self, std_vp=0.05, std_vs=0.08, std_den=0.03, seed=None):
        """对场加相对比例高斯扰动，并夹下界保证物理有效。"""
        rng = np.random.default_rng(seed)
        vp = self.Vp + rng.normal(0, std_vp * self.Vp, self.Vp.shape)
        vs = self.Vs + rng.normal(0, std_vs * self.Vs, self.Vs.shape)
        de = self.Den + rng.normal(0, std_den * self.Den, self.Den.shape)
        return np.maximum(vp, 100), np.maximum(vs, 50), np.maximum(de, 1.0)

    def compute_elastic_impedance(self, angles, Vp=None, Vs=None, Den=None):
        vp = self.Vp if Vp is None else np.asarray(Vp, dtype=np.float64)
        vs = self.Vs if Vs is None else np.asarray(Vs, dtype=np.float64)
        de = self.Den if Den is None else np.asarray(Den, dtype=np.float64)
        vp_t = torch.from_numpy(vp)
        vs_t = torch.from_numpy(vs)
        de_t = torch.from_numpy(de)
        ang = torch.as_tensor(angles, dtype=torch.float32)
        ei = connolly_elastic_impedance(vp_t, vs_t, de_t, ang,
                                        vp0=self.Vp0, vs0=self.Vs0, den0=self.Den0)
        return ei.numpy()                                  # (nz, nx, n_angles)

    def compute_reflection_coefficient(self, angles, method="ei", Vp=None, Vs=None, Den=None):
        vp = self.Vp if Vp is None else np.asarray(Vp, dtype=np.float64)
        vs = self.Vs if Vs is None else np.asarray(Vs, dtype=np.float64)
        de = self.Den if Den is None else np.asarray(Den, dtype=np.float64)
        vp_t, vs_t, de_t = (torch.from_numpy(a) for a in (vp, vs, de))
        ang = torch.as_tensor(angles, dtype=torch.float32)
        if method == "ei":
            ei = connolly_elastic_impedance(vp_t, vs_t, de_t, ang,
                                            vp0=self.Vp0, vs0=self.Vs0, den0=self.Den0)
            rc = elastic_impedance_reflection(ei)
            return rc.numpy(), ei.numpy()
        if method == "zoeppritz":
            return exact_zoeppritz_pp(vp_t, vs_t, de_t, ang).numpy(), None
        if method == "aki":
            return aki_richards_reflection(vp_t, vs_t, de_t, ang).numpy(), None
        raise ValueError(f"不支持的反射系数方法: {method}")

    @staticmethod
    def synthesize_seismic_vectorized(reflection_coefficient, wavelet):
        """每道-角度组合沿深度轴卷积（np.convolve mode='same'，核取奇数长）。

        Args:
            reflection_coefficient: (nz, nx, n_angles)。
            wavelet: 1D 子波（numpy）。

        Returns:
            (nz, nx, n_angles) 合成角道集。
        """
        rc = np.asarray(reflection_coefficient, dtype=np.float64)
        w = np.asarray(wavelet, dtype=np.float64)
        if len(w) % 2 == 0:
            w = w[:-1]
        nz, nx, na = rc.shape
        rc_flat = rc.reshape(nz, nx * na)
        out = np.zeros_like(rc_flat)
        start = (len(w) - 1) // 2
        for i in range(rc_flat.shape[1]):
            # 居中卷积：'full' + 裁剪，核长 ≥ 信号长时也能保持输出长度 = nz
            out[:, i] = np.convolve(rc_flat[:, i], w, mode="full")[start:start + nz]
        return out.reshape(nz, nx, na)

    @staticmethod
    def create_ricker_wavelet(duration=0.2, dt=0.001, frequency=20):
        """返回 (wavelet: ndarray, time: ndarray)，bruges 兼容签名。"""
        w = ricker_wavelet(dt=dt, f=frequency, length=duration)
        n = w.shape[-1]
        t = np.linspace(-(n - 1) * dt / 2, (n - 1) * dt / 2, n)
        return w.numpy().ravel(), t

    @staticmethod
    def create_ormsby_wavelet(duration=0.1, dt=0.001, f_hz=(5, 10, 60, 80)):
        w = ormsby_wavelet(dt=dt, f_hz=f_hz, length=duration)
        n = w.shape[-1]
        t = np.linspace(-(n - 1) * dt / 2, (n - 1) * dt / 2, n)
        return w.numpy().ravel(), t

    def run_full_workflow(self, angles, frequency=40, ei_method="connolly",
                          reflection_method="ei", add_perturbation=False,
                          perturbation_params=None, output_dir=None):
        """复现 legacy 工作流，返回结果字典；``output_dir`` 给定时写 .mat。"""
        if add_perturbation and perturbation_params:
            Vp_use, Vs_use, Den_use = self.add_perturbation(**perturbation_params)
            data_type = "perturbed"
        else:
            Vp_use, Vs_use, Den_use = self.Vp, self.Vs, self.Den
            data_type = "original"
        if ei_method not in ("connolly",):
            raise ValueError(f"EI 方法仅支持 connolly，收到 {ei_method!r}")
        rc, ei = self.compute_reflection_coefficient(angles, method=reflection_method,
                                                     Vp=Vp_use, Vs=Vs_use, Den=Den_use)
        wavelet, wtime = self.create_ricker_wavelet(frequency=frequency)
        seismic = self.synthesize_seismic_vectorized(rc, wavelet)
        results = {
            "angles": np.asarray(angles, dtype=np.float64),
            "frequency": float(frequency),
            "ei_method": ei_method,
            "reflection_method": reflection_method,
            "data_type": data_type,
            "Vp0": self.Vp0, "Vs0": self.Vs0, "Den0": self.Den0,
            "reflection_coefficient": rc,
            "elastic_impedance": ei,
            "seismic_data": seismic,
            "wavelet": wavelet, "wavelet_time": wtime,
        }
        if output_dir is not None:
            from scipy import io

            import os
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "elastic_impedance_results.mat")
            io.savemat(path, {k: v for k, v in results.items() if np.isscalar(v) or hasattr(v, "shape")})
        return results