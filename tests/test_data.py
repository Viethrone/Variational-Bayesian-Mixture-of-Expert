"""P1 数据管线测试：processed 重建 + 数据集契约（InversionSample 形状/元信息/分块）。"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.common import build_dataset, project_root
from src.data.base import collate_samples
from src.data.seismic_dataset import Seismic2DDataset, load_norm_stats
from src.utils.config import load_config
from tests.conftest import dataset_ready


@pytest.fixture(scope="session")
def cfg_1d():
    if not dataset_ready("1d"):
        pytest.skip("1-D processed data missing (see README 'Prepare the data')")
    return load_config(project_root() / "configs/data/seismic1d.yaml")


@pytest.fixture(scope="session")
def cfg_2d():
    if not dataset_ready("2d"):
        pytest.skip("2-D processed data missing (see README 'Prepare the data')")
    return load_config(project_root() / "configs/data/seismic2d.yaml")


def test_registry_has_both_datasets():
    from src.data.base import DATASETS

    assert "seismic_1d" in DATASETS
    assert "seismic_2d" in DATASETS


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_1d_split_lengths(cfg_1d, split):
    ds = build_dataset(cfg_1d, split)
    expected = {"train": 12, "val": 7, "test": 1360}[split]
    assert len(ds) == expected, f"{split} 划分长度应为 {expected}"


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_2d_split_lengths(cfg_2d, split):
    ds = build_dataset(cfg_2d, split)
    expected = {"train": 12, "val": 3, "test": 1360}[split]
    assert len(ds) == expected, f"{split} 划分长度应为 {expected}"


def test_1d_sample_shapes(cfg_1d):
    ds = build_dataset(cfg_1d, "train")
    s = ds[0]
    assert s.observation.shape == (6, 1200)
    assert s.target.shape == (3, 1200)
    assert s.meta["target_low"].shape == (3, 1200)
    assert s.meta["prior"].shape == (3,)          # K 通道先验（深度均值）
    assert s.meta["mask"].shape == (1, 1200)
    assert s.sample_id == ds.sample_names[0]          # id = 划分文件中的道名（字符串）
    assert s.meta["trace"] == int(ds.sample_names[0])
    # 物理量级抽查：Vp ∈ [900, 5000]，Den ≈ 2.x
    assert 900.0 < s.target[0].max() < 5000.0
    assert 1.0 < s.target[2].mean() < 3.0


def test_2d_sample_shapes(cfg_2d):
    ds = build_dataset(cfg_2d, "train")
    s = ds[0]
    w = 2 * 7 + 1
    assert s.observation.shape == (6, 1200, w)
    assert s.target.shape == (3, 1200, w)
    assert s.meta["target_low"].shape == (3, 1200, w)
    assert s.meta["prior"].shape == (3, 1200, w)
    assert s.meta["mask"].shape == (1, 1200, w)
    assert s.meta["mask"].sum() == 1200          # 仅中心井位列标记


def _full_cfg(cfg_2d):
    """深拷贝一份 data 节并覆盖 window.mode=full（避免污染 session fixture）。"""
    cfg = dict(cfg_2d)
    data = cfg_2d["data"]
    cfg["data"] = {**data, "window": {**data["window"], "mode": "full"}}
    return cfg


def test_2d_full_mode_shapes(cfg_2d):
    """整剖面模式：len=1，全 (C,H,W) 样本，mask 在划分井位列=1。"""
    ds = build_dataset(_full_cfg(cfg_2d), "train")
    assert len(ds) == 1
    s = ds[0]
    assert s.meta["id"] == "full" and s.meta["mode"] == "full"
    assert s.observation.shape == (6, 1200, 1360)
    assert s.target.shape == (3, 1200, 1360)
    assert s.meta["target_low"].shape == (3, 1200, 1360)
    assert s.meta["prior"].shape == (3, 1200, 1360)
    m = s.meta["mask"]
    assert m.shape == (1, 1200, 1360)
    # train 划分 12 口井 → 12 条井位列全 1，其余 0
    well_traces = [int(t) for t in ds.sample_names]
    assert s.meta["well_traces"] == well_traces and len(well_traces) == 12
    assert float(m[0, :, well_traces].sum()) == pytest.approx(12 * 1200)
    assert float(m[0, 0, well_traces].min()) == pytest.approx(1.0)
    # 非井位列全 0
    non_well = [c for c in range(1360) if c not in well_traces]
    assert float(m[0, :, non_well].sum()) == 0.0
    # 物理量级抽查（与 slice 样本同源）
    assert 900.0 < s.target[0].max() < 5000.0


def test_2d_full_mode_matches_slice_center(cfg_2d):
    """full 模式的井位列 == slice 模式该井窗口中心列（双模式口径一致）。"""
    ds_full = build_dataset(_full_cfg(cfg_2d), "train")
    ds = build_dataset(cfg_2d, "train")
    r = ds.radius
    trace = int(ds.sample_names[0])                 # 第一口井
    assert ds_full.sample_names == ds.sample_names
    w_slice = ds._load_item(str(trace))
    assert w_slice.observation.shape == (6, 1200, 2 * r + 1)
    sample = ds_full[0]
    np.testing.assert_allclose(w_slice.observation[:, :, r], sample.observation[:, :, trace], rtol=1e-6)
    np.testing.assert_allclose(w_slice.target[:, :, r], sample.target[:, :, trace], rtol=1e-6)


def test_2d_full_mode_invalid_mode_raises(cfg_2d):
    with pytest.raises(ValueError, match="mode"):
        _ = Seismic2DDataset(
            root=project_root() / cfg_2d["data"]["root"],
            split="train",
            split_file=project_root() / cfg_2d["data"]["splits"]["train"],
            window={"mode": "bogus"},
            normalization=cfg_2d["data"]["normalization"],
        )


def test_2d_full_mode_training_batch(cfg_2d):
    """`_VBMLEBatchDataset` 正常消费 full 样本：一个 batch 含整剖面各字段。"""
    from src.trainers.vbmile_trainer import build_vbmile_loader

    ds = build_dataset(_full_cfg(cfg_2d), "val")
    loader = build_vbmile_loader(ds, batch_size=1, shuffle=False, num_experts=3)
    batch = next(iter(loader))
    assert batch["seismic"].shape == (1, 6, 1200, 1360)
    assert batch["target"].shape == (1, 3, 1200, 1360)
    assert batch["mask"].shape == (1, 1, 1200, 1360)
    assert batch["prior"].shape == (1, 3, 1200, 1360)
    assert batch["target_low"].shape == (1, 3, 1200, 1360)


def test_collate_1d(cfg_1d):
    ds = build_dataset(cfg_1d, "train")
    batch = collate_samples([ds[0], ds[1]])
    assert batch["observation"].shape == (2, 6, 1200)
    assert batch["target"].shape == (2, 3, 1200)
    assert len(batch["meta"]) == 2


def test_normalize_roundtrip(cfg_1d):
    ds = build_dataset(cfg_1d, "train")
    # 物理空间真值一小段（深度轴 4 点）
    x = np.array([[2863.0, 1500.0, 3900.0, 1100.0],
                  [1350.0, 700.0, 1800.0, 400.0],
                  [2.23, 2.0, 2.5, 1.9]], np.float32)
    x_norm = ds.to_norm(x, which="target")
    x_back = ds.to_physical(x_norm, which="target")
    np.testing.assert_allclose(x_back, x, rtol=1e-5, atol=1e-3)
    assert x_norm.dtype == np.float32
    # 2D 形状（C, H, W）广播
    x2 = np.random.RandomState(0).rand(3, 8, 5).astype(np.float32) * 2.0 + 0.5
    assert ds.to_norm(x2, which="target").shape == x2.shape
    np.testing.assert_allclose(
        ds.to_physical(ds.to_norm(x2, which="target"), which="target"), x2, rtol=1e-4, atol=1e-4)


def test_gmm_prior_loaded(cfg_1d, cfg_2d):
    for cfg in (cfg_1d, cfg_2d):
        ds = build_dataset(cfg, "train")
        g = ds.gmm_prior
        assert g["means"].shape == (3, 3)
        if "covariances" in g:                      # full 协方差格式（新）
            assert g["covariances"].shape == (3, 3, 3)
            for c in g["covariances"]:
                assert np.linalg.eigvalsh(c).min() > 0
        else:                                       # legacy 对角格式
            assert g["variances"].shape == (3, 3)
            assert g["variances"].min() > 0


def test_processed_manifest_intact(cfg_1d):
    from tools.preprocess.run_preprocess import check

    assert check(project_root() / "configs/data/seismic1d.yaml") == 0


def test_stats_file_reference(cfg_1d):
    """norm_stats.json 与 legacy Evaluation.txt 口径对照（P1 门禁）。"""
    from tools.data_check.check_data import REF_TARGET_MEAN_LEGACY, REF_TARGET_STD_LEGACY, TOL_REL

    ndiv = cfg_1d["data"]["normalization"]
    stats = load_norm_stats(ndiv["stats_file"], project_root() / cfg_1d["data"]["root"])
    tm, ts = stats["target_mean"], stats["target_std"]
    for i in range(3):
        assert abs(float(tm[i, 0]) - float(REF_TARGET_MEAN_LEGACY[i, 0])) / float(REF_TARGET_MEAN_LEGACY[i, 0]) < TOL_REL
        assert abs(float(ts[i, 0]) - float(REF_TARGET_STD_LEGACY[i, 0])) / float(REF_TARGET_STD_LEGACY[i, 0]) < TOL_REL