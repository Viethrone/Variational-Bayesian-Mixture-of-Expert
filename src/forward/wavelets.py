"""地震子波生成（Ricker / Ormsby），torch 单一来源。

原 legacy 依赖 ``bruges.filters.wavelets``；本模块用 numpy+torch 重写相同数学，
并按仓库统一口径保证核为关于 0 对称的**奇数长**（见 ``ricker_wavelet`` 注释）。
两种子波都返回 ``(1, 1, L)`` 的 torch.float32 张量，可直接作 ``SeismicForward``
的 ``wavelet`` 键值或 ``ElasticImpedanceForward.wavelet``。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

__all__ = ["ricker_wavelet", "ormsby_wavelet"]


def _time_axis(dt: float, length: float, n: int) -> np.ndarray:
    """关于 0 对称的奇数长时间轴（秒）：-a .. +a，步长 dt。"""
    return np.linspace(-(n - 1) * dt / 2, (n - 1) * dt / 2, n, dtype=np.float32)


def _odd_len(length: float, dt: float) -> int:
    n = int(round(length / dt))
    return n + 1 if n % 2 == 0 else n


def ricker_wavelet(dt: float = 0.002, f: float = 30.0, length: float = 0.1) -> torch.Tensor:
    """Ricker（Mexican hat）子波，形状 (1,1,L)。

    w(t) = (1 - 2(π f t)²) exp(-(π f t)²)

    Notes:
        * legacy 用 `np.arange(-l/2, l/2+dt, dt)`，采样点个数奇偶漂移会产生
          非对称/偶数长核，导致卷积输出长度 L±偏移。这里保证核为关于 0 对称的
          **奇数长**（101 点 ↔ 0.202s@2ms，51 点 ↔ 0.1s@2ms）。
    """
    n = _odd_len(length, dt)
    t = _time_axis(dt, length, n)
    w = (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)
    return torch.tensor(w, dtype=torch.float32).view(1, 1, -1)


def ormsby_wavelet(dt: float = 0.001, f_hz: Sequence[float] = (5, 10, 60, 80),
                   length: float = 0.1) -> torch.Tensor:
    """Ormsby 子波（bruges.filters.wavelets.ormsby 的重写），形状 (1,1,L)。

    需要 f1 < f2 <= f3 < f4（金字塔形频带）。trapezoid 频谱的 sinc 组合：

        w = (n·sinc²(π f4 t) - m·sinc²(π f3 t) - p·sinc²(π f2 t) + q·sinc²(π f1 t))
            · (f4 - f3) · π / 2

    sinc(x) = sin(x)/x，x=0 处取 1。与 ``ricker_wavelet`` 同口径（对称奇数长核）。
    """
    f1, f2, f3, f4 = (float(x) for x in f_hz)
    if f1 < 0 or f2 <= f1 or f3 <= f2 or f4 <= f3:
        raise ValueError(f"频率须满足 f1 < f2 <= f3 < f4，收到 {f_hz}")
    n = _odd_len(length, dt)
    t = _time_axis(dt, length, n)
    tf = lambda x: np.sinc(x)  # noqa: E731  np.sinc(x)=sin(πx)/(πx)

    def sinc_sq(freq: float) -> np.ndarray:
        return tf(freq * t) ** 2

    coefs = (f4 / (f4 - f3), -f3 / (f4 - f3), -f2 / (f2 - f1), f1 / (f2 - f1))
    w = (coefs[0] * sinc_sq(f4) + coefs[1] * sinc_sq(f3)
         + coefs[2] * sinc_sq(f2) + coefs[3] * sinc_sq(f1))
    w = w * (f4 - f3) * np.pi / 2
    return torch.tensor(w, dtype=torch.float32).view(1, 1, -1)