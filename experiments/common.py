"""实验公共助手：命令行解析、配置解析、run 创建、数据集组装。

所有实验入口都从这里起步，
保证 run 目录结构、元信息落盘与随机种子处理完全一致。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.utils.config import load_config, save_config
from src.utils.logging_utils import get_logger
from src.utils.run_info import RunInfo
from src.utils.seeds import seed_everything

__all__ = [
    "make_parser",
    "resolve_config",
    "prepare_run",
    "build_dataset",
    "save_cmd",
    "set_default",
    "project_root",
]

logger = get_logger(__name__)


def project_root() -> Path:
    """返回项目根目录（本文件的上级目录）。"""
    return Path(__file__).resolve().parent.parent


def set_default(parser: argparse.ArgumentParser, dest: str, value: Any) -> None:
    """调整已定义参数的默认值。

    `make_parser` 已统一声明 --outputs-root 等公共选项，子命令若需要不同默认值，
    用本函数修改而不是重复 add_argument（重复添加会触发 argparse 冲突错误）。
    """
    for action in parser._actions:
        if action.dest == dest:
            action.default = value
            return
    raise ValueError(f"解析器中不存在参数 dest={dest!r}，无法设置默认值")


def make_parser(
    description: str,
    *,
    default_config: str = "configs/inversion/default.yaml",
    extra_args: Optional[Sequence[tuple]] = None,
) -> argparse.ArgumentParser:
    """构造统一的命令行解析器。

    Args:
        description: 子命令说明。
        default_config: `--config` 的默认值。
        extra_args: 追加参数，形如 ``[("--split", {"default": "test"})]``。
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", type=str, default=default_config, help="配置文件路径（支持 _base_ 继承）")
    parser.add_argument("--name", type=str, default=None, help="实验名，用于 run 目录命名")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，覆盖配置中的 seed")
    parser.add_argument("--device", type=str, default=None, help="计算设备，覆盖配置中的 device")
    parser.add_argument("--outputs-root", type=str, default=None, help="run 目录根，默认 outputs/runs")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="点分路径覆盖配置，可多次使用，例如 --set inversion.solver.lr=0.01",
    )
    for flag, kwargs in extra_args or ():
        parser.add_argument(flag, **kwargs)
    return parser


def resolve_config(args: argparse.Namespace, extra_overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """加载配置并应用命令行与额外覆盖项。"""
    overrides = list(getattr(args, "overrides", []) or []) + list(extra_overrides or [])
    if getattr(args, "seed", None) is not None:
        overrides.append(f"seed={args.seed}")
    if getattr(args, "device", None) is not None:
        overrides.append(f"device={args.device}")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    cfg = load_config(config_path, overrides=overrides)

    if getattr(args, "name", None):
        cfg["exp_name"] = args.name
    return cfg


def prepare_run(
    cfg: Dict[str, Any],
    args: Optional[argparse.Namespace] = None,
    *,
    outputs_root: Optional[str] = None,
    subdirs: Optional[Sequence[str]] = None,
) -> RunInfo:
    """设置随机种子并创建 run 目录，写入全部元信息。"""
    seed = int(cfg.get("seed", 0))
    seed_everything(
        seed,
        deterministic=bool(cfg.get("deterministic", False)),
        # top-level ``deterministic_algorithms: true|warn`` (opt-in, default untouched)
        deterministic_algorithms=cfg.get("deterministic_algorithms", False),
    )

    root = outputs_root or (args.outputs_root if args is not None and getattr(args, "outputs_root", None) else None)
    if root is None:
        root = str((cfg.get("paths") or {}).get("runs") or "outputs/runs")
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = project_root() / root_path

    kwargs: Dict[str, Any] = {}
    if subdirs is not None:
        kwargs["subdirs"] = tuple(subdirs)

    run_info = RunInfo.create(
        cfg,
        outputs_root=root_path,
        exp_name=cfg.get("exp_name"),
        repo_root=project_root(),
        **kwargs,
    )
    run_info.setup_logging(cfg)
    return run_info


def build_dataset(cfg: Dict[str, Any], split: str = "test") -> Any:
    """按配置组装数据集实例。

    读取 `cfg["data"]`，从 DATASETS 注册表实例化对应实现。

    TODO: 具体数据集实现完成后，这里无需修改（由 `_target_` 驱动）。
    """
    from src.data.base import DATASETS
    from src.utils.registry import build_from_config

    data_cfg = dict(cfg.get("data") or {})
    target = data_cfg.pop("_target_", None)
    if target is None:
        raise ValueError(
            "配置缺少 data._target_，无法确定数据集实现。"
            "请在 configs/data/*.yaml 中填写，例如 _target_: inversion_dataset"
        )

    splits = dict(data_cfg.get("splits") or {})
    split_file = splits.get(split)
    if split_file is None:
        raise ValueError(f"data.splits 中未配置 {split!r} 划分文件；已有键: {sorted(splits)}")

    root = data_cfg.get("root") or (cfg.get("paths") or {}).get("data_processed") or "data/processed"
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = project_root() / root_path

    split_path = Path(split_file)
    if not split_path.is_absolute():
        split_path = project_root() / split_path

    params = {k: v for k, v in data_cfg.items() if k not in ("splits", "root")}
    return build_from_config(
        DATASETS,
        {"_target_": target, **params},
        extra_kwargs={"root": str(root_path), "split": split, "split_file": str(split_path)},
    )


def save_cmd(run_info: RunInfo, argv: Optional[Sequence[str]] = None) -> Path:
    """把当前命令行写入 run 目录（RunInfo.create 已写过一次，这里用于补充规范化信息）。"""
    argv = list(argv if argv is not None else sys.argv)
    text = "\n".join(
        [
            f"cwd: {os.getcwd()}",
            f"executable: {sys.executable}",
            f"argv: {' '.join(argv)}",
        ]
    )
    return run_info.write_text("cmd.txt", text + "\n")
