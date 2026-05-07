import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple

class Standardize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
    def normalize(self, x):
        return (x - self.mean) / self.std
    def unnormalize(self, x):
        return x * self.std + self.mean

class SeismicDataset(Dataset):
    """
    二维地震反演数据集，支持 'slice' 模式（以井位为中心的窗口）和 'full' 模式（全剖面）。
    """
    def __init__(self,
                 seismic: np.ndarray,          # (C_seis, H, W)
                 target: np.ndarray,           # (C_target, H, W)
                 target_low: np.ndarray,       # (C_target, H, W)
                 prior_spatial: np.ndarray,    # (K, H, W) 或 (H, W)
                 well_indices: Optional[List[int]] = None,
                 radius: Optional[int] = None,
                 pad_mode: str = 'reflect',
                 mode: str = 'slice'):
        assert seismic.ndim == 3
        assert target.ndim == 3
        H, W = seismic.shape[1], seismic.shape[2]
        assert target.shape[1:] == (H, W)
        assert target_low.shape[1:] == (H, W)
        if prior_spatial.ndim == 2:
            prior_spatial = prior_spatial[None, ...]   # 扩展为 (1, H, W)
        assert prior_spatial.shape[1:] == (H, W)

        self.seismic = seismic
        self.target = target
        self.target_low = target_low
        self.prior = prior_spatial
        self.well_indices = np.asarray(well_indices) if well_indices is not None else None
        self.mode = mode

        if mode == 'slice':
            assert radius is not None and well_indices is not None
            self.radius = radius
            self.window_width = 2 * radius + 1
            self.pad_mode = pad_mode
        elif mode == 'full':
            self.radius = None
            self.window_width = W
        else:
            raise ValueError("mode must be 'slice' or 'full'")

        self.H, self.W = H, W

    def __len__(self):
        return len(self.well_indices) if self.mode == 'slice' else 1

    def _extract_window(self, data: np.ndarray, center_cdp: int) -> np.ndarray:
        left_pad = max(0, self.radius - center_cdp)
        right_pad = max(0, self.radius - (self.W - 1 - center_cdp))
        start = max(0, center_cdp - self.radius)
        end = min(self.W, center_cdp + self.radius + 1)
        if data.ndim == 3:
            window = data[:, :, start:end]
            pad_width = ((0,0), (0,0), (left_pad, right_pad))
            window = np.pad(window, pad_width, mode=self.pad_mode)
        else:
            raise NotImplementedError("Only 3D data supported")
        return window

    def __getitem__(self, idx):
        if self.mode == 'slice':
            center = self.well_indices[idx]
            seismic_win = self._extract_window(self.seismic, center)
            target_win = self._extract_window(self.target, center)
            target_low_win = self._extract_window(self.target_low, center)
            prior_win = self._extract_window(self.prior, center)
            # 生成掩码：仅在井位列为1
            mask = np.zeros((1, self.H, self.window_width), dtype=np.float32)
            col_in_window = self.radius + (center - max(0, center - self.radius))   # 实际列位置
            mask[:, :, col_in_window] = 1.0
        else:   # full
            seismic_win = self.seismic
            target_win = self.target
            target_low_win = self.target_low
            prior_win = self.prior
            mask = np.ones((1, self.H, self.W), dtype=np.float32)
            if self.well_indices is not None:
                mask[:, :, :] = 0.0
                for cdp in self.well_indices:
                    if 0 <= cdp < self.W:
                        mask[:, :, cdp] = 1.0

        return {
            'seismic': torch.from_numpy(seismic_win).float(),
            'target': torch.from_numpy(target_win).float(),
            'target_low': torch.from_numpy(target_low_win).float(),
            'mask': torch.from_numpy(mask).float(),
            'spatial_prior': torch.from_numpy(prior_win).float()
        }
    