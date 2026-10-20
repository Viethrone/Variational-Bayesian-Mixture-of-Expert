"""配置加载：支持 `_base_` 继承、深合并、命令行覆盖、副本落盘。

用法::

    cfg = load_config("configs/inversion/default.yaml", overrides=["seed=1", "inversion.solver.lr=0.01"])

约定：
    * `_base_` 可以是字符串或字符串列表；相对路径以**当前 yaml 所在目录**为基准。
    * 合并顺序：base(从左到右) -> 当前文件 -> overrides。
    * overrides 语法为 `a.b.c=value`，value 会被解析为 yaml 标量（数字/布尔/null/字符串）。
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
    """读取单个 yaml 文件为 dict；空文件返回 {}。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是映射(dict)，实际为 {type(data).__name__}: {p}")
    return data


def _resolve_bases(raw: Mapping[str, Any], current_dir: Path) -> List[Path]:
    """把 `_base_` 解析为按优先级从低到高排列的路径列表。"""
    bases = raw.get(BASE_KEY)
    if bases is None:
        return []
    if isinstance(bases, (str, Path)):
        bases = [bases]
    if not isinstance(bases, Sequence):
        raise ValueError(f"`{BASE_KEY}` 必须是字符串或字符串列表，实际为 {type(bases).__name__}")
    resolved = []
    for b in bases:
        p = Path(b)
        resolved.append(p if p.is_absolute() else (current_dir / p).resolve())
    return resolved


def deep_merge(base: ConfigDict, override: Mapping[str, Any]) -> ConfigDict:
    """递归合并两个 dict，返回新对象；override 优先，dict 逐层合并，其他类型直接覆盖。"""
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
    """加载配置并应用继承与覆盖。

    Args:
        path: 入口 yaml 路径。
        overrides: 形如 ``["a.b=1", "c=d"]`` 的覆盖项。
        keep_base_key: 为 True 时保留 `_base_` 字段（调试用）。
    """
    entry = Path(path)
    cfg = _load_recursive(entry, _seen=set())
    if overrides:
        cfg = override_from_cli(cfg, overrides)
    if not keep_base_key:
        cfg.pop(BASE_KEY, None)
    return cfg


def _load_recursive(path: Path, _seen: set) -> ConfigDict:
    """递归展开 `_base_`，并检测循环继承。"""
    resolved = path.resolve()
    if resolved in _seen:
        chain = " -> ".join(str(p) for p in _seen)
        raise ValueError(f"检测到配置循环继承: {chain} -> {resolved}")
    _seen = _seen | {resolved}

    raw = load_yaml(resolved)
    merged: ConfigDict = {}
    for base_path in _resolve_bases(raw, resolved.parent):
        merged = deep_merge(merged, _load_recursive(base_path, _seen))
    return deep_merge(merged, raw)


def parse_override(item: str) -> tuple[str, Any]:
    """把 ``a.b.c=value`` 解析为 (键路径, 已转型的值)。"""
    if "=" not in item:
        raise ValueError(f"覆盖项必须是 key=value 形式，收到: {item!r}")
    key, raw_value = item.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"覆盖项的键为空: {item!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError:
        value = raw_value  # 无法解析时按原始字符串处理
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
    """按点分路径应用命令行覆盖，返回新 dict。"""
    result = copy.deepcopy(cfg)
    for item in overrides:
        key, value = parse_override(item)
        _set_nested(result, key, value)
    return result


def save_config(cfg: Mapping[str, Any], path: Union[str, Path]) -> Path:
    """把最终配置写盘（run 目录中的 config.yaml 副本）。"""
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
