"""Marmousi2 AVA inversion datasets (1-D and 2-D) used by Geo-VB-MILE.

Layout (built by ``tools/preprocess/run_preprocess.py``, physical space, float32):

    processed/seismic6.npy           (6, 1200, 1360)  observed angle gathers (0 deg dropped)
    processed/target.npy             (3, 1200, 1360)  ground-truth Vp / Vs / density
    processed/target_low_s{30,100}   (3, 1200, 1360)  low-frequency model (sigma 30 for 1-D, 100 for 2-D)
    processed/spatial_prior.npy      (K, 1200, 1360)  spatial prior probability, channel-first
    processed/gmm_prior.npz          (K,3,3)/(K,3)    GMM parameter prior (normalized; full covariance preferred)

Normalization is not applied inside the dataset: the statistics from norm_stats.json are
exposed as ``seismic_stats / target_stats`` and the trainer z-scores where needed, matching
the legacy notebooks: theta_norm = (theta - mu) / sigma, syn_norm = (syn - mu) / sigma.
meta also carries ``target_low``, ``prior``, ``mask`` and ``trace`` (the trace index).

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
    """Load norm_stats.json; return (channel, 1) numpy vectors keyed by statistic.

    ``stats_file`` may be relative (data/interim/norm_stats.json); it is tried as an
    absolute path, then relative to the cwd, then relative to the project root.
    """
    p = Path(stats_file)
    if not p.is_absolute() and not p.exists():
        cand = Path.cwd() / stats_file
        if not cand.exists():
            cand = root.parent.parent / stats_file
        p = cand
    if not p.exists():
        raise FileNotFoundError(f"normalization stats not found: {stats_file} (run run_preprocess first)")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {
        k: np.asarray(raw[k], dtype=np.float32).reshape(-1, 1)
        for k in ("seismic_mean", "seismic_std", "target_mean", "target_std")
    }


class _SeismicDatasetBase(BaseDataset):
    """Shared core: lazy cube loading, statistics resolution and sample assembly."""

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
        splits_dir: str | None = None,   # accepted for config compatibility, unused
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

    # ---- lazy loading ----
    def _load_cube(self, key: str) -> np.ndarray:
        attr = f"_{key}"
        if not hasattr(self, attr) or getattr(self, attr) is None:
            # an empty or None entry means unset (the no_spatial_prior ablation passes null)
            full = np.load(self.root / self.files[key]) if self.files.get(key) else None
            setattr(self, attr, full)
        return getattr(self, attr)

    @property
    def seismic(self) -> np.ndarray:
        v = self._load_cube("seismic")
        if v is None:
            raise FileNotFoundError(f"{self.name}: config does not name a seismic file")
        return v

    @property
    def target(self) -> np.ndarray:
        v = self._load_cube("target")
        if v is None:
            raise FileNotFoundError(f"{self.name}: config does not name a target file")
        return v

    @property
    def target_low(self) -> np.ndarray:
        v = self._load_cube("target_low")
        if v is None:
            raise FileNotFoundError(f"{self.name}: config does not name a target_low file")
        return v

    @property
    def spatial_prior(self) -> np.ndarray | None:
        return self._load_cube("spatial_prior")

    @property
    def gmm_prior(self) -> dict[str, np.ndarray]:
        """GMM parameter prior in normalized space.

        Full covariance: ``{means: (K,3), covariances: (K,3,3), weights: (K,)}``;
        legacy diagonal: ``{means: (K,3), variances: (K,3)}``. ``variances``, ``covariances``
        and the optional mixture ``weights`` are passed through as-is (uniform when absent).
        
        """
        if self._prior_means is None:
            p = self.root / self.files.get("gmm_prior", "gmm_prior.npz")
            if not p.exists():
                raise FileNotFoundError(f"GMM prior file missing: {p} (run run_preprocess first)")
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

    # ---- normalization helpers (physical <-> normalized) ----
    @staticmethod
    def _broadcast(stats: np.ndarray, ndim: int) -> np.ndarray:
        """Reshape a (C,1) statistic to (C,1,1,...) so it broadcasts over (C, ...) tensors."""
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
    """Per-trace dataset: one sample is a single trace of 6 angles plus the 3-parameter profile.

    Length = number of trace indices in splits/*.txt. ``_load_item`` returns physical-space
    observation (6, 1200) / target (3, 1200); meta carries the normalized low-frequency model,
    the spatial prior (depth mean) and the well mask.
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
            target_low=low_norm,        # (3, 1200) low-frequency model in target space
            prior=prior,                # (K,) or None
            mask=np.ones((1, obs_phys.shape[-1]), dtype=np.float32),
        )
        sample = InversionSample(observation=obs_phys, target=tgt_phys, meta=meta)
        return self._augment_if(sample)

    def _augment_if(self, sample: InversionSample) -> InversionSample:
        # the legacy code applied a 1.0 +- 0.05 gain per trace; off by default for determinism
        if self.augment:
            g = float(np.random.uniform(0.95, 1.05))
            sample.observation = (sample.observation * g).astype(np.float32)
            sample.meta["gain"] = g
        return sample


@DATASETS.register("seismic_2d")
class Seismic2DDataset(_SeismicDatasetBase):
    """2-D dataset with two output modes (matching the legacy ``SeismicDataset.mode``):

    ``mode="slice"`` (default) takes a fixed-width window (2*radius+1) around each well trace;
    ``mode="full"`` treats the whole section as one sample (``__len__ == 1``), used for a single
    whole-section forward pass (the legacy test2D path with an all-ones mask).

    slice: observation (6, H, 2r+1) / target (3, H, 2r+1), reflect-padded at the borders,
        with meta.mask (1, H, 2r+1) marking the centre well column.
    full:  observation (6, H, W) / target (3, H, W); meta.mask (1, H, W) is 1 on the
        split's well columns and 0 elsewhere (the test split covers all traces, so it is
        all ones); meta also carries ``well_traces`` and ``mode``. The prior and the
        low-frequency model cover the full section.
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
            raise ValueError(f"window.mode must be 'slice' or 'full', got {self.mode!r}")

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
            mask[..., t] = 1.0                              # mark the whole well column

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
        mask[:, :, w // 2] = 1.0       # centre well column

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
