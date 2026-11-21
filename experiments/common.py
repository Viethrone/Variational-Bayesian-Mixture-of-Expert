"""Shared experiment helpers: CLI parsing, config resolution, run creation, dataset build.

Every entry point starts here, so the run directory layout, metadata
and seeding behave identically across experiments.
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
    """Project root (the parent directory of this file)."""
    return Path(__file__).resolve().parent.parent


def set_default(parser: argparse.ArgumentParser, dest: str, value: Any) -> None:
    """Change the default of an already declared argument.

    `make_parser` declares the shared options; use this instead of adding the argument again
    (a duplicate would make argparse raise).
    """
    for action in parser._actions:
        if action.dest == dest:
            action.default = value
            return
    raise ValueError(f"the parser has no argument dest={dest!r}")


def make_parser(
    description: str,
    *,
    default_config: str = "configs/inversion/default.yaml",
    extra_args: Optional[Sequence[tuple]] = None,
) -> argparse.ArgumentParser:
    """Build the shared command-line parser.

    Args:
        description: what the command does.
        default_config: default for `--config`.
        extra_args: additional arguments, e.g. ``[("--split", {"default": "test"})]``.
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", type=str, default=default_config, help="config file path (supports _base_ inheritance)")
    parser.add_argument("--name", type=str, default=None, help="experiment name, used for the run directory")
    parser.add_argument("--seed", type=int, default=None, help="random seed, overrides the configured one")
    parser.add_argument("--device", type=str, default=None, help="device, overrides the configured one")
    parser.add_argument("--outputs-root", type=str, default=None, help="run directory root (default outputs/runs)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dot-path config override, e.g. --set trainer.max_epochs=2",
    )
    for flag, kwargs in extra_args or ():
        parser.add_argument(flag, **kwargs)
    return parser


def resolve_config(args: argparse.Namespace, extra_overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load the config and apply the command-line and extra overrides."""
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
    """Seed everything, create the run directory and write its metadata."""
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
    """Build the dataset instance described by the config.

    Reads `cfg["data"]` and instantiates it from the DATASETS registry.

    Nothing here needs to change when a new dataset is added: `_target_` drives it.
    """
    from src.data.base import DATASETS
    from src.utils.registry import build_from_config

    data_cfg = dict(cfg.get("data") or {})
    target = data_cfg.pop("_target_", None)
    if target is None:
        raise ValueError(
            "the config has no data._target_, so the dataset cannot be resolved."
            "set it in configs/data/*.yaml, e.g. _target_: seismic_2d"
        )

    splits = dict(data_cfg.get("splits") or {})
    split_file = splits.get(split)
    if split_file is None:
        raise ValueError(f"data.splits has no {split!r} entry; present keys: {sorted(splits)}")

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
    """Write the command line into the run directory (RunInfo.create already writes cmd.txt)."""
    argv = list(argv if argv is not None else sys.argv)
    text = "\n".join(
        [
            f"cwd: {os.getcwd()}",
            f"executable: {sys.executable}",
            f"argv: {' '.join(argv)}",
        ]
    )
    return run_info.write_text("cmd.txt", text + "\n")
