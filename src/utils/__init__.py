"""Shared utilities: config, registry, seeding, logging, run metadata, IO."""

from src.utils.config import (
    deep_merge,
    load_config,
    override_from_cli,
    save_config,
)
from src.utils.io_utils import ensure_dir, read_json, write_json
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.registry import Registry, build_from_config, register
from src.utils.run_info import RunInfo, collect_env_info, git_commit
from src.utils.seeds import seed_everything

__all__ = [
    "deep_merge",
    "load_config",
    "override_from_cli",
    "save_config",
    "ensure_dir",
    "read_json",
    "write_json",
    "get_logger",
    "setup_logging",
    "Registry",
    "build_from_config",
    "register",
    "RunInfo",
    "collect_env_info",
    "git_commit",
    "seed_everything",
]
