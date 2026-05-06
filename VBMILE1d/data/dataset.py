import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple


class Standardize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def normalize(self, x):
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        return x * self.std + self.mean



class Seismic1DDataset(Dataset):
    """
    One-dimensional: seismic inversion dataset: Each sample corresponds to a CDP line (single-channel data).
    Input: Seismic angle line set, low-frequency model, prior_spatial
    Output: Single-channel seismic data, low-frequency model, mask (always 1) and prior_spatial.

    """

    def __init__(self,
                 seismic: np.ndarray,      # (C_seis, H, W)
                 target: np.ndarray,       # (C_target, H, W)
                 target_low: np.ndarray,   # (C_target, H, W)
                 prior_spatial : np.ndarray,
                 well_indices: List[int] = None,
                 augment: bool = False):
        assert seismic.ndim == 3
        assert target.ndim == 3

        H, W = seismic.shape[1], seismic.shape[2]
        assert target.shape[1:] == (H, W)
        assert target_low.shape[1:] == (H, W)

        self.seismic = seismic
        self.target = target
        self.prior = prior_spatial
        self.target_low = target_low
        self.well_indices = well_indices if well_indices is not None else list(range(W))
        self.augment = augment
        self.H = H
        self.W = W

    def __len__(self):
        return len(self.well_indices)

    def __getitem__(self, idx):
        cdp = self.well_indices[idx]
        # 提取单道数据，形状: (C, H)
        seismic_trace = self.seismic[:, :, cdp]
        target_trace = self.target[:, :, cdp]
        target_low_trace = self.target_low[:, :, cdp]
        prior_trace = self.prior[:, :, cdp]

        if self.augment:
            seismic_trace *= np.random.uniform(0.9, 1.1)

        # 掩码：单道全部有效
        mask = np.ones((1, self.H), dtype=np.float32)

        return {
            'seismic': torch.from_numpy(seismic_trace).float(),
            'target': torch.from_numpy(target_trace).float(),
            'target_low': torch.from_numpy(target_low_trace).float(),
            'mask': torch.from_numpy(mask).float(),
            'prior': prior_trace
        }

