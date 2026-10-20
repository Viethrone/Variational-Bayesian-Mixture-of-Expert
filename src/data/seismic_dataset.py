"""Marmousi2 AVA 反演数据集的 1D/2D 实现（主方法 Geo-VB-MILE 使用）。

数据布局（由 ``tools/preprocess/run_preprocess.py`` 生成，全部为物理空间 float32）：

    processed/seismic6.npy           (6, 1200, 1360)  观测角道集（0° 角已去掉）
    processed/target.npy             (3, 1200, 1360)  弹性参数真值 Vp/Vs/Den
    processed/target_low_s{30,100}   (3, 1200, 1360)  低频模型（高斯平滑，选 1D/2D 口径）
    processed/spatial_prior.npy      (K, 1200, 1360)  空间先验概率（通道在前）
    processed/gmm_prior.npz          (K,3,3)/(K,3)    GMM 参数先验（归一化空间；full 协方差优先）

归一化不在 Dataset 内做（Dataset 输出物理空间样本），而是把 norm_stats.json 的统计量挂在
``dataset.seismic_stats / target_stats`` 上，由训练器/推理管线在需要处 z-score —— 与 legacy
notebook 的用法一致（theta_norm = (theta-phys − μ)/σ，syn_norm = (syn_phys − μ)/σ）。
meta 中额外携带：``target_low``（低频模型 trace）、``prior``（该处空间先验）、``mask``（井位掩码）、
``trace``（道索引号），供模型输入与损失加权使用。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.data.base import DATASETS, BaseDataset, InversionSample

__all__ = ["Seismic1DDataset", "Seismic2DDataset", "load_norm_stats"]


def load_norm_stats(stats_file: str | Path, root: Path) -> dict[str, np.ndarray]:
    """加载 norm_stats.json；按键返回 (channel, 1) 的 numpy 向量。

    stats_file 为配置相对路径（如 data/interim/norm_stats.json），依次尝试：绝对路径 →
    当前工作目录 → 项目根（root 的上上级目录）。
    """
    p = Path(stats_file)
    if not p.is_absolute() and not p.exists():
        cand = Path.cwd() / stats_file
        if not cand.exists():
            cand = root.parent.parent / stats_file
        p = cand
    if not p.exists():
        raise FileNotFoundError(f"找不到归一化统计文件: {stats_file}（请先运行 run_preprocess）")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {
        k: np.asarray(raw[k], dtype=np.float32).reshape(-1, 1)
        for k in ("seismic_mean", "seismic_std", "target_mean", "target_std")
    }


class _SeismicDatasetBase(BaseDataset):
    """共享：cube 懒加载 + 统计量解析 + 样本元信息装配。"""

    files: dict[str, str] = {}
    _seismic: np.ndarray | None = None
    _target: np.ndarray | None = None
    _target_low: np.ndarray | None = None
    _spatial_prior: np.ndarray | None = None
    _prior_means: np.ndarray | None = None
    _prior_vars: np.ndarray | None = None
    _prior_covs: np.ndarray | None = None
    _prior_weights: np.ndarray | None = None

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        sample_names: Sequence[str] | None = None,
        split_file: str | Path | None = None,
        transform: Any | None = None,
        name: str | None = None,
        files: dict[str, str] | None = None,
        shape: dict[str, Any] | None = None,
        normalization: dict[str, Any] | None = None,
        target_columns: Sequence[str] | None = None,
        window: dict[str, Any] | None = None,
        loader: dict[str, Any] | None = None,
        splits_dir: str | None = None,   # 骨架 configs 遗留键，仅吸收不消费
        augment: bool = False,
    ) -> None:
        super().__init__(root, split=split, sample_names=sample_names,
                         split_file=split_file, transform=transform)
        self.name = name or self.__class__.__name__
        self.files = dict(files or {})
        self.shape = dict(shape or {})
        self.target_columns = list(target_columns) if target_columns else ["Vp", "Vs", "Den"]
        self.window = dict(window or {})
        self.loader_cfg = dict(loader or {})
        self.splits_dir = splits_dir
        self.augment = augment

        stats_file = ((normalization or {}).get("stats_file")
                      or "data/interim/norm_stats.json")
        self.seismic_stats = self.target_stats = None
        st = load_norm_stats(stats_file, self.root)
        self.seismic_mean, self.seismic_std = st["seismic_mean"], st["seismic_std"]
        self.target_mean, self.target_std = st["target_mean"], st["target_std"]

    # ---- 懒加载 ----
    def _load_cube(self, key: str) -> np.ndarray:
        attr = f"_{key}"
        if not hasattr(self, attr) or getattr(self, attr) is None:
            # files 条目为空串/None 时视为未配置（消融 no_spatial_prior 用 null 关掉）
            full = np.load(self.root / self.files[key]) if self.files.get(key) else None
            setattr(self, attr, full)
        return getattr(self, attr)

    @property
    def seismic(self) -> np.ndarray:
        v = self._load_cube("seismic")
        if v is None:
            raise FileNotFoundError(f"{self.name}: 配置未给出 seismic 文件名")
        return v

    @property
    def target(self) -> np.ndarray:
        v = self._load_cube("target")
        if v is None:
            raise FileNotFoundError(f"{self.name}: 配置未给出 target 文件名")
        return v

    @property
    def target_low(self) -> np.ndarray:
        v = self._load_cube("target_low")
        if v is None:
            raise FileNotFoundError(f"{self.name}: 配置未给出 target_low 文件名")
        return v

    @property
    def spatial_prior(self) -> np.ndarray | None:
        return self._load_cube("spatial_prior")

    @property
    def gmm_prior(self) -> dict[str, np.ndarray]:
        """归一化空间的 GMM 参数先验。

        full 协方差文件：``{means: (K,3), covariances: (K,3,3), weights: (K,)}``；
        legacy 对角文件：``{means: (K,3), variances: (K,3)}``。透传
        ``variances``（= 对角）与 ``covariances``（= full），以及可选的混合权重
        ``weights``（缺失时下游取均匀权重），供下游按需取用。
        """
        if self._prior_means is None:
            p = self.root / self.files.get("gmm_prior", "gmm_prior.npz")
            if not p.exists():
                raise FileNotFoundError(f"缺少 GMM 先验文件: {p}（请先运行 run_preprocess）")
            z = np.load(p)
            self._prior_means = np.asarray(z["means"], dtype=np.float32)
            if "covariances" in z.files:
                self._prior_covs = np.asarray(z["covariances"], dtype=np.float32)
            if "variances" in z.files:
                self._prior_vars = np.asarray(z["variances"], dtype=np.float32)
            if "weights" in z.files:
                self._prior_weights = np.asarray(z["weights"], dtype=np.float32)
            if self._prior_means.ndim == 1:
                self._prior_means = self._prior_means.reshape(1, -1)
                if self._prior_vars is not None:
                    self._prior_vars = self._prior_vars.reshape(1, -1)
                if self._prior_covs is not None:
                    self._prior_covs = self._prior_covs.reshape(1, -1, -1)
                if self._prior_weights is not None:
                    self._prior_weights = self._prior_weights.reshape(1)
        out = {"means": self._prior_means}
        if self._prior_vars is not None:
            out["variances"] = self._prior_vars
        if self._prior_covs is not None:
            out["covariances"] = self._prior_covs
        if self._prior_weights is not None:
            out["weights"] = self._prior_weights
        return out

    # ---- 归一化助手（物理空间 <-> 归一化空间）----
    @staticmethod
    def _broadcast(stats: np.ndarray, ndim: int) -> np.ndarray:
        """把 (C,1) 统计量 reshape 成可对 (C, ...) 张量广播的 (C,1,1,...)。"""
        return stats.reshape(-1, *([1] * (ndim - 1)))

    def to_norm(self, x: np.ndarray, *, which: str) -> np.ndarray:
        mean, std = (self.seismic_mean, self.seismic_std) if which == "seismic" \
            else (self.target_mean, self.target_std)
        x = np.asarray(x, np.float32)
        return (x - self._broadcast(mean, x.ndim)) / self._broadcast(std, x.ndim)

    def to_physical(self, x_norm: np.ndarray, *, which: str) -> np.ndarray:
        mean, std = (self.seismic_mean, self.seismic_std) if which == "seismic" \
            else (self.target_mean, self.target_std)
        x = np.asarray(x_norm, np.float32)
        return x * self._broadcast(std, x.ndim) + self._broadcast(mean, x.ndim)

    def _base_meta(self, name: str, **extra: Any) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "trace": int(name),
            "dataset": self.name,
            "target_columns": self.target_columns,
        }
        meta.update(extra)
        return meta


@DATASETS.register("seismic_1d")
class Seismic1DDataset(_SeismicDatasetBase):
    """逐道（trace-level）数据集：每样本 = 单道 6 角观测 + 3 参数剖面。

    长度 = splits/*.txt 中的道索引数。``_load_item`` 返回物理空间的
    observation (6, 1200) / target (3, 1200)，meta 携带归一化低频模型、
    空间先验（深度均值）与井位掩码。
    """

    def _load_item(self, name: str) -> InversionSample:
        trace = int(name)
        obs_phys = self.seismic[:, :, trace]           # (6, 1200)
        tgt_phys = self.target[:, :, trace]            # (3, 1200)
        low_norm = self.to_norm(self.target_low[:, :, trace], which="target")

        prior: np.ndarray | None = None
        if self.spatial_prior is not None:
            prior = self.spatial_prior[:, :, trace].mean(axis=1).astype(np.float32)  # (K,)

        meta = self._base_meta(
            name,
            target_low=low_norm,        # (3, 1200) 低频模型（target 空间归一化）
            prior=prior,                # (K,) 或 None
            mask=np.ones((1, obs_phys.shape[-1]), dtype=np.float32),
        )
        sample = InversionSample(observation=obs_phys, target=tgt_phys, meta=meta)
        return self._augment_if(sample)

    def _augment_if(self, sample: InversionSample) -> InversionSample:
        # legacy 用 1.0±0.05 的 gain 乘整道；默认关闭（deterministic 复现优先）
        if self.augment:
            g = float(np.random.uniform(0.95, 1.05))
            sample.observation = (sample.observation * g).astype(np.float32)
            sample.meta["gain"] = g
        return sample


@DATASETS.register("seismic_2d")
class Seismic2DDataset(_SeismicDatasetBase):
    """二维数据集，支持两种输出模式（对齐 legacy ``SeismicDataset.mode``）：

    ``mode="slice"``（默认）以每口井（划分文件内 trace）为中心取定宽窗口 (2·radius+1)；
    ``mode="full"`` 把**整个剖面**作为单条样本（``__len__ == 1``）——推理/评测时对全剖面
    一次前向（legacy ``AVA2d_inverse_phase`` 的 test2D 路径，``well_indices=None`` 即全一 mask）。

    slice：observation (6, H, 2r+1) / target (3, H, 2r+1)，边界 reflect 填充，
    meta.mask (1, H, 2r+1) 标记中心井位列。
    full ：observation (6, H, W) / target (3, H, W)，meta.mask (1, H, W) 在划分文件
    各井位列 = 1、其余 0（test 划分含全部道 → 实际为全一，与 legacy 一致）；
    meta 另带 ``well_traces`` 井位清单、``mode="full"``。空间先验与低频模型取整剖面。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mode = str(self.window.get("mode", "slice"))
        if self.mode == "slice":
            self.radius = int(self.window.get("radius", 7))
            self.pad_mode = self.window.get("pad", "reflect")
        elif self.mode == "full":
            self.radius = None
            self.pad_mode = None
        else:
            raise ValueError(f"window.mode 必须是 'slice' 或 'full'，实际为 {self.mode!r}")

    def __len__(self) -> int:
        return 1 if self.mode == "full" else len(self.sample_names)

    def _window(self, cube: np.ndarray, trace: int) -> np.ndarray:
        r = self.radius
        W = cube.shape[-1]
        lo, hi = max(0, trace - r), min(W, trace + r + 1)
        left, right = max(0, r - trace), max(0, trace + r + 1 - W)
        pad_widths = [(0, 0)] * cube.ndim
        pad_widths[-1] = (left, right)
        seg = cube[:, :, lo:hi]
        return np.pad(seg, pad_widths, mode=self.pad_mode).astype(np.float32)

    def _load_full(self) -> InversionSample:
        obs_phys = self.seismic.astype(np.float32)          # (6, H, W)
        tgt_phys = self.target.astype(np.float32)           # (3, H, W)
        low_norm = self.to_norm(self.target_low, which="target")   # (3, H, W)

        prior: np.ndarray | None = None
        if self.spatial_prior is not None:
            prior = self.spatial_prior.astype(np.float32)   # (K, H, W)

        H, W = obs_phys.shape[1], obs_phys.shape[2]
        mask = np.zeros((1, H, W), dtype=np.float32)
        well_traces = [int(t) for t in self.sample_names]
        for t in well_traces:
            mask[..., t] = 1.0                              # 井位列整列标记

        meta: dict[str, Any] = {
            "trace": -1,
            "id": "full",
            "mode": "full",
            "dataset": self.name,
            "target_columns": self.target_columns,
            "well_traces": well_traces,
            "target_low": low_norm,
            "prior": prior,
            "mask": mask,
        }
        return InversionSample(observation=obs_phys, target=tgt_phys, meta=meta)

    def _load_item(self, name: str) -> InversionSample:
        if self.mode == "full":
            return self._load_full()
        trace = int(name)
        obs_phys = self._window(self.seismic, trace)      # (6, 1200, 2r+1)
        tgt_phys = self._window(self.target, trace)       # (3, 1200, 2r+1)
        low_norm = self.to_norm(self._window(self.target_low, trace), which="target")

        prior: np.ndarray | None = None
        if self.spatial_prior is not None:
            prior = self._window(self.spatial_prior, trace)  # (K, 1200, 2r+1)

        w = obs_phys.shape[-1]
        mask = np.zeros((1, obs_phys.shape[-2], w), dtype=np.float32)
        mask[:, :, w // 2] = 1.0       # 中心井位列

        meta = self._base_meta(
            name,
            target_low=low_norm,
            prior=prior,
            mask=mask,
            center_trace=trace,
        )
        return InversionSample(observation=obs_phys, target=tgt_phys, meta=meta)

    def _augment_if(self, sample: InversionSample) -> InversionSample:
        return sample