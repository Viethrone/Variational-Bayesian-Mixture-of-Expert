"""Config loading: `_base_` inheritance, deep merge, CLI overrides, saved copy.

Usage::

    cfg = load_config("configs/inversion/default.yaml", overrides=["seed=1", "inversion.solver.lr=0.01"])

Conventions:
    * `_base_` is a string or a list of strings; relative paths resolve against the yaml directory.
    * merge order: base (left to right) -> current file -> overrides.
    * overrides use `a.b.c=value`; the value is parsed as a YAML scalar.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Union

import yaml

__all__ = [
    "ConfigDict",
    "deep_merge",
    "load_config",
    "load_yaml",
    "override_from_cli",
    "parse_override",
    "save_config",
]

BASE_KEY = "_base_"
ConfigDict = Dict[str, Any]


def load_yaml(path: Union[str, Path]) -> ConfigDict:
    """Read a single yaml file into a dict; an empty file yields {}."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"the top level of a config must be a mapping, got {type(data).__name__}: {p}")
    return data


def _resolve_bases(raw: Mapping[str, Any], current_dir: Path) -> List[Path]:
    """Resolve `_base_` into a path list ordered from lowest to highest priority."""
    bases = raw.get(BASE_KEY)
    if bases is None:
        return []
    if isinstance(bases, (str, Path)):
        bases = [bases]
    if not isinstance(bases, Sequence):
        raise ValueError(f"`{BASE_KEY}` must be a string or a list of strings, got {type(bases).__name__}")
    resolved = []
    for b in bases:
        p = Path(b)
        resolved.append(p if p.is_absolute() else (current_dir / p).resolve())
    return resolved


def deep_merge(base: ConfigDict, override: Mapping[str, Any]) -> ConfigDict:
    """Recursively merge two dicts into a new one: dicts merge, anything else is replaced."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key == BASE_KEY:
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    path: Union[str, Path],
    overrides: Iterable[str] | None = None,
    *,
    keep_base_key: bool = False,
) -> ConfigDict:
    """Load a config, applying inheritance and overrides.

    Args:
        path: entry yaml path.
        overrides: entries of the form ``["a.b=1", "c=d"]``.
        keep_base_key: keep the `_base_` field (debugging).
    """
    entry = Path(path)
    cfg = _load_recursive(entry, _seen=set())
    if overrides:
        cfg = override_from_cli(cfg, overrides)
    if not keep_base_key:
        cfg.pop(BASE_KEY, None)
    return cfg


def _load_recursive(path: Path, _seen: set) -> ConfigDict:
    """Expand `_base_` recursively and detect cyclic inheritance."""
    resolved = path.resolve()
    if resolved in _seen:
        chain = " -> ".join(str(p) for p in _seen)
        raise ValueError(f"cyclic config inheritance: {chain} -> {resolved}")
    _seen = _seen | {resolved}

    raw = load_yaml(resolved)
    merged: ConfigDict = {}
    for base_path in _resolve_bases(raw, resolved.parent):
        merged = deep_merge(merged, _load_recursive(base_path, _seen))
    return deep_merge(merged, raw)


def parse_override(item: str) -> tuple[str, Any]:
    """Parse ``a.b.c=value`` into (key path, typed value)."""
    if "=" not in item:
        raise ValueError(f"an override must look like key=value, got: {item!r}")
    key, raw_value = item.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"the override key is empty: {item!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError:
        value = raw_value  # unparsable values stay as the raw string
    return key, value


def _set_nested(cfg: ConfigDict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def override_from_cli(cfg: ConfigDict, overrides: Iterable[str]) -> ConfigDict:
    """Apply dot-path overrides and return a new dict."""
    result = copy.deepcopy(cfg)
    for item in overrides:
        key, value = parse_override(item)
        _set_nested(result, key, value)
    return result


def save_config(cfg: Mapping[str, Any], path: Union[str, Path]) -> Path:
    """Write the resolved config (the config.yaml copy kept in a run directory)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            dict(cfg),
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
    return p
