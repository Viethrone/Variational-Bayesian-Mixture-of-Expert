"""Trainer base class and configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.utils.logging_utils import get_logger
from src.utils.registry import Registry, build_from_config
from src.utils.run_info import RunInfo

__all__ = ["TRAINERS", "BaseTrainer", "TrainerConfig", "build_trainer"]

logger = get_logger(__name__)
TRAINERS = Registry("trainer")


@dataclass
class TrainerConfig:
    """Training-loop settings, built from ``cfg["trainer"]``."""

    max_epochs: int = 50
    val_every: int = 1
    test_at_end: bool = True
    grad_clip: Optional[float] = None
    max_iter_per_sample: int = 100
    save_predictions: bool = True
    save_reconstructed: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "TrainerConfig":
        """Build from a config dict; unknown keys land in `extra` instead of raising."""
        cfg = dict(cfg or {})
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in cfg.items() if k in known}
        extra = {k: v for k, v in cfg.items() if k not in known}
        return cls(extra=extra, **kwargs)


class BaseTrainer(ABC):
    """Trainer base class.

    Subclasses implement `run`; shared helpers for metrics and best-epoch tracking live here.
    """

    def __init__(self, run_info: RunInfo, cfg: Optional[TrainerConfig] = None) -> None:
        self.run_info = run_info
        self.cfg = cfg or TrainerConfig()
        self.logger = get_logger(self.__class__.__name__)
        self._best: Dict[str, Any] = {"metric": None, "value": None, "epoch": -1}

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the full training / inversion loop and return the metrics dict."""

    # ---- shared helpers ----
    def record_best(self, metric_name: str, value: float, epoch: int, *, higher_is_better: bool = False) -> bool:
        """Record a value and report whether it is a new best."""
        current = self._best.get("value")
        improved = (
            current is None
            or (value > current if higher_is_better else value < current)
        )
        if improved:
            self._best = {"metric": metric_name, "value": float(value), "epoch": int(epoch)}
            self.logger.info("new best %s=%.6f @ epoch %d", metric_name, value, epoch)
        return improved

    def finalize(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Finalise: write metrics.json and return the complete metrics dict."""
        payload = dict(metrics)
        if self._best.get("value") is not None:
            payload["best"] = dict(self._best)
        self.run_info.write_metrics(payload)
        self.logger.info("metrics written to %s", self.run_info.path("metrics.json"))
        return payload


def build_trainer(
    cfg: Dict[str, Any],
    run_info: RunInfo,
    registry: Optional[Registry] = None,
) -> BaseTrainer:
    """Build a trainer.

    The config looks like::

        trainer:
          _target_: vbmile
          max_epochs: 500
          lr: 1.0e-3
    """
    trainer_cfg = dict(cfg.get("trainer") or {})
    target = trainer_cfg.pop("_target_", "vbmile")
    reg = registry or TRAINERS
    factory = reg.get(str(target))
    params = TrainerConfig.from_dict(trainer_cfg)
    return factory(run_info=run_info, cfg=params)
