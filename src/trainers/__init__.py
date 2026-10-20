"""Training and evaluation flow.

A trainer only orchestrates the loop (epochs, validation, checkpointing, metric
persistence); the concrete method lives in ``src/models``.
Every write happens inside the run directory provided by ``RunInfo``.
"""

from src.trainers.base import TRAINERS, BaseTrainer, TrainerConfig, build_trainer
from src.trainers.vbmile_trainer import VBEMLETrainer

__all__ = [
    "TRAINERS",
    "BaseTrainer",
    "TrainerConfig",
    "VBEMLETrainer",
    "build_trainer",
]
