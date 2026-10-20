"""训练器基类与配置。"""

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
    """训练流程参数，由 `cfg["trainer"]` 构造。"""

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
        """从配置字典构造，未知键进入 `extra` 而非报错，便于逐步扩展。"""
        cfg = dict(cfg or {})
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in cfg.items() if k in known}
        extra = {k: v for k, v in cfg.items() if k not in known}
        return cls(extra=extra, **kwargs)


class BaseTrainer(ABC):
    """训练器基类。

    子类实现 `run`；基类提供指标落盘、最优记录等通用助手。
    """

    def __init__(self, run_info: RunInfo, cfg: Optional[TrainerConfig] = None) -> None:
        self.run_info = run_info
        self.cfg = cfg or TrainerConfig()
        self.logger = get_logger(self.__class__.__name__)
        self._best: Dict[str, Any] = {"metric": None, "value": None, "epoch": -1}

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """执行完整训练/反演流程，返回指标字典。"""

    # ---- 通用助手 ----
    def record_best(self, metric_name: str, value: float, epoch: int, *, higher_is_better: bool = False) -> bool:
        """记录并判断是否为当前最优；返回 True 表示刷新最优。"""
        current = self._best.get("value")
        improved = (
            current is None
            or (value > current if higher_is_better else value < current)
        )
        if improved:
            self._best = {"metric": metric_name, "value": float(value), "epoch": int(epoch)}
            self.logger.info("刷新最优 %s=%.6f @ epoch %d", metric_name, value, epoch)
        return improved

    def finalize(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """统一收尾：写 metrics.json 并返回完整指标字典。"""
        payload = dict(metrics)
        if self._best.get("value") is not None:
            payload["best"] = dict(self._best)
        self.run_info.write_metrics(payload)
        self.logger.info("指标已写入 %s", self.run_info.path("metrics.json"))
        return payload


def build_trainer(
    cfg: Dict[str, Any],
    run_info: RunInfo,
    registry: Optional[Registry] = None,
) -> BaseTrainer:
    """构造训练器。

    配置形如::

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
