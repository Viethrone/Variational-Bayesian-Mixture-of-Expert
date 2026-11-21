"""File IO helpers: directory creation, JSON read/write, atomic writes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union

__all__ = ["ensure_dir", "read_json", "write_json", "atomic_write_text", "human_size"]

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create a directory (with parents) and return its Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: PathLike, *, default: Any = None) -> Any:
    """Read JSON; return `default` when the file is missing."""
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: PathLike, data: Any, *, indent: int = 2) -> Path:
    """Write JSON through a temporary file so an interrupted write leaves no half file."""
    p = Path(path)
    ensure_dir(p.parent)
    text = json.dumps(data, ensure_ascii=False, indent=indent, default=str)
    atomic_write_text(p, text + "\n")
    return p


def atomic_write_text(path: PathLike, text: str, *, encoding: str = "utf-8") -> Path:
    """Atomic text write: temporary file in the same directory + os.replace."""
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as f:
            f.write(text)
        os.replace(tmp_name, p)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise
    return p


def human_size(num_bytes: float) -> str:
    """Format a byte count for human reading in data audit reports."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"
