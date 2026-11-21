"""Experiment entry layer: orchestration only, no core logic.

Rules:
    * scripts here read the config, create a run, assemble data/model/trainer, call src, save.
    * algorithm implementations live in src/ so several experiments can reuse them.
    * every run gets a new directory under outputs/runs/; earlier runs are never overwritten.

Subdirectories:
    main        the main experiment
"""

from experiments.common import (
    build_dataset,
    make_parser,
    prepare_run,
    resolve_config,
    save_cmd,
)

__all__ = [
    "build_dataset",
    "make_parser",
    "prepare_run",
    "resolve_config",
    "save_cmd",
]
