"""run 元信息：创建带时间戳的 run 目录，并落盘配置副本、命令、commit、环境、指标。

这是可复现性的核心：任何实验入口都应通过 `RunInfo.create(...)` 起步，
之后所有产出（权重、预测、重建场、日志、临时图）都写在 run 目录内。
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.utils.config import save_config
from src.utils.io_utils import atomic_write_text, ensure_dir, read_json, write_json
from src.utils.logging_utils import get_logger, setup_logging

__all__ = ["RunInfo", "collect_env_info", "git_commit", "make_run_name"]

logger = get_logger(__name__)

PathLike = Any

RUN_SUBDIRS = (
    "checkpoints",
    "predictions",
    "reconstructed",
    "logs",
    "figures",
)


def make_run_name(exp_name: str, seed: Any = 0, *, timestamp: Optional[datetime] = None) -> str:
    """生成 run 名称：`YYYY-MM-DD_HHMMSS_<实验名>_seed<种子>`。"""
    ts = timestamp or datetime.now()
    safe_name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(exp_name))
    return f"{ts.strftime('%Y-%m-%d_%H%M%S')}_{safe_name}_seed{seed}"


def git_commit(repo_root: PathLike = ".") -> str:
    """返回当前 HEAD 的短 commit；非 git 仓库或无 git 时返回 "unknown"。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = out.stdout.strip()
    return commit or "unknown"


def collect_env_info() -> Dict[str, Any]:
    """收集环境信息：Python、平台、关键依赖版本、设备可见性。"""
    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "system": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cwd": os.getcwd(),
        "packages": {},
    }
    for pkg in ("numpy", "scipy", "pandas", "matplotlib", "yaml", "torch", "jax", "sklearn"):
        try:
            mod = __import__(pkg)
        except ImportError:
            continue
        info["packages"][pkg] = getattr(mod, "__version__", "unknown")

    try:
        import torch  # noqa: PLC0415

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["torch_cuda_device_count"] = torch.cuda.device_count()
            if torch.cuda.device_count() > 0:   # CUDA_VISIBLE_DEVICES="" 时 device(0) 非法
                info["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_cuda_available"] = False
    return info


@dataclass
class RunInfo:
    """一次实验 run 的目录与元信息句柄。

    Attributes:
        run_dir: run 根目录，所有产出的写入位置。
        run_name: run 目录名。
        config: 合并后的最终配置。
    """

    run_dir: Path
    run_name: str
    config: Dict[str, Any]
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---- 创建 ----
    @classmethod
    def create(
        cls,
        config: Mapping[str, Any],
        *,
        outputs_root: PathLike = "outputs/runs",
        exp_name: Optional[str] = None,
        run_name: Optional[str] = None,
        repo_root: PathLike = ".",
        command: Optional[str] = None,
        subdirs: tuple = RUN_SUBDIRS,
    ) -> "RunInfo":
        """创建 run 目录并写入元信息文件。

        写入内容：
            config.yaml / cmd.txt / git_commit.txt / env.txt / 以及各产出子目录。
        """
        cfg = dict(config)
        name = run_name or make_run_name(
            exp_name or cfg.get("exp_name", "run"),
            cfg.get("seed", 0),
        )
        run_dir = ensure_dir(Path(outputs_root) / name)
        for sub in subdirs:
            ensure_dir(run_dir / sub)

        commit = git_commit(repo_root)
        env = collect_env_info()

        save_config(cfg, run_dir / "config.yaml")
        atomic_write_text(
            run_dir / "cmd.txt",
            (command or " ".join(sys.argv)) + "\n",
        )
        atomic_write_text(run_dir / "git_commit.txt", commit + "\n")
        atomic_write_text(
            run_dir / "env.txt",
            "\n".join(
                [
                    f"python: {env['python']}",
                    f"executable: {env['executable']}",
                    f"platform: {env['platform']}",
                    f"cwd: {env['cwd']}",
                    f"git_commit: {commit}",
                    f"torch_cuda_available: {env.get('torch_cuda_available')}",
                    "",
                    "packages:",
                    *[f"  {k}=={v}" for k, v in sorted(env["packages"].items())],
                ]
            )
            + "\n",
        )
        write_json(run_dir / "env.json", env)

        info = cls(run_dir=run_dir, run_name=name, config=cfg)
        info.setup_logging(cfg)
        logger.info("run 目录: %s", info.run_dir.resolve())
        logger.info("git commit: %s", commit)
        return info

    # ---- 路径助手 ----
    def path(self, *parts: str) -> Path:
        """拼接 run 目录下的子路径。"""
        return self.run_dir.joinpath(*parts)

    def subdir(self, name: str) -> Path:
        """获取（并按需创建）run 目录下的子目录。"""
        return ensure_dir(self.run_dir / name)

    @property
    def checkpoints_dir(self) -> Path:
        return self.subdir("checkpoints")

    @property
    def predictions_dir(self) -> Path:
        return self.subdir("predictions")

    @property
    def reconstructed_dir(self) -> Path:
        return self.subdir("reconstructed")

    @property
    def figures_dir(self) -> Path:
        return self.subdir("figures")

    @property
    def logs_dir(self) -> Path:
        return self.subdir("logs")

    # ---- 日志 ----
    def setup_logging(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        """按配置初始化文件+控制台日志，日志写入 run 目录内。"""
        cfg = cfg if cfg is not None else self.config
        log_cfg = dict(cfg.get("logging") or {})
        rel = log_cfg.get("log_file", "logs/run.log")
        level = log_cfg.get("level", "INFO")
        console = bool(log_cfg.get("console", True))
        setup_logging(level, log_file=self.run_dir / rel, console=console)

    # ---- 指标 ----
    def write_metrics(self, metrics: Mapping[str, Any], *, name: str = "metrics.json") -> Path:
        """写入指标，并附带 run 元信息，便于后续汇总。"""
        payload: Dict[str, Any] = {
            "run_name": self.run_name,
            "exp_name": self.config.get("exp_name"),
            "seed": self.config.get("seed"),
            "git_commit": git_commit(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "metrics": dict(metrics),
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return write_json(self.run_dir / name, payload)

    def read_metrics(self, *, name: str = "metrics.json") -> Any:
        """读取已写入的指标文件；不存在时返回 None。"""
        return read_json(self.run_dir / name, default=None)

    def write_text(self, filename: str, text: str) -> Path:
        """在 run 目录下写文本文件。"""
        return atomic_write_text(self.run_dir / filename, text)

    def __str__(self) -> str:  # pragma: no cover - 调试辅助
        return f"RunInfo({self.run_name})"
