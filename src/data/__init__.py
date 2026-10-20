"""数据层：Dataset、DataLoader、预处理与归一化。

约定：
    * 只从 `data/processed/` 读取，`data/raw/` 由 tools/preprocess 生成 processed 时使用。
    * 样本以 `InversionSample` 统一表示，下游（正演、反演、评估）只依赖该结构。
    * 划分从 `data/splits/{train,val,test}.txt` 读取，不在代码里随机切分。
"""

from src.data.base import DATASETS, BaseDataset, InversionSample, collate_samples

__all__ = ["DATASETS", "BaseDataset", "InversionSample", "collate_samples"]

# 导入具体数据集以触发注册
from src.data.seismic_dataset import Seismic1DDataset, Seismic2DDataset  # noqa: E402,F401
