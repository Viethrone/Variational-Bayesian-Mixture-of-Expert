"""Data layer: datasets, loaders, preprocessing and normalization.

Conventions:
    * only `data/processed/` is read; `data/raw/` is consumed by tools/preprocess.
    * samples are `InversionSample`; downstream code only depends on that structure.
    * splits come from `data/splits/{train,val,test}.txt`, never from random slicing in code.
"""

from src.data.base import DATASETS, BaseDataset, InversionSample, collate_samples

__all__ = ["DATASETS", "BaseDataset", "InversionSample", "collate_samples"]

# import the concrete datasets to trigger registration
from src.data.seismic_dataset import Seismic1DDataset, Seismic2DDataset  # noqa: E402,F401
