"""组件注册表：把配置里的 `_target_` 字符串解析成实际对象。

设计目标：求解器、正演算子、先验、正则化、损失、指标、模型都可以在**不改调用方代码**
的前提下替换，消融实验只需改 yaml。

用法::

    from src.utils.registry import Registry, build_from_config

    SOLVERS = Registry("solver")

    @SOLVERS.register("gradient_descent")
    class GradientDescent:
        def __init__(self, max_iter=100, lr=1e-2):
            ...

    solver = build_from_config(SOLVERS, cfg["inversion"]["solver"])

`build_from_config` 会把除 `_target_` 以外的键作为关键字参数传入构造函数。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, TypeVar

__all__ = ["Registry", "REGISTRIES", "build_from_config", "register"]

T = TypeVar("T")

# 全局命名空间，便于统一 introspection 与跨模块查找
REGISTRIES: Dict[str, "Registry"] = {}


class Registry:
    """按名字登记可调用对象（类或工厂函数）的轻量注册表。"""

    def __init__(self, name: str, *, global_scope: bool = True) -> None:
        self.name = name
        self._items: Dict[str, Callable[..., Any]] = {}
        if global_scope:
            if name in REGISTRIES and REGISTRIES[name] is not self:
                raise ValueError(f"注册表名称冲突: {name!r} 已存在")
            REGISTRIES[name] = self

    # ---- 登记 ----
    def register(self, key: Optional[str] = None, *, force: bool = False) -> Callable[[T], T]:
        """装饰器形式登记。`key` 缺省时使用对象的 `__name__`（转小写）。"""

        def decorator(obj: T) -> T:
            name = key or getattr(obj, "__name__", None)
            if not name:
                raise ValueError("无法推断注册名，请显式传入 key")
            if isinstance(name, str):
                name = name.lower()
            if name in self._items and not force:
                existing = self._items[name]
                if existing is not obj:
                    raise ValueError(
                        f"[{self.name}] 名称 {name!r} 已被 {existing!r} 占用；"
                        f"如需覆盖请传 force=True"
                    )
            self._items[name] = obj
            return obj

        return decorator

    def register_obj(self, key: str, obj: Callable[..., Any], *, force: bool = False) -> None:
        """函数式登记，便于登记第三方实现。"""
        if key in self._items and not force and self._items[key] is not obj:
            raise ValueError(f"[{self.name}] 名称 {key!r} 已存在")
        self._items[key] = obj

    # ---- 查询 ----
    def get(self, key: str) -> Callable[..., Any]:
        if key not in self._items:
            raise KeyError(self._missing_message(key))
        return self._items[key]

    def contains(self, key: str) -> bool:
        return key in self._items

    def keys(self) -> Iterable[str]:
        return self._items.keys()

    def _missing_message(self, key: str) -> str:
        available = ", ".join(sorted(self._items)) or "(空)"
        return (
            f"[{self.name}] 未注册 {key!r}。已注册: {available}。"
            f"提示：确认对应模块已被 import（注册发生在 import 时）。"
        )

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._items

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"Registry({self.name!r}, items={sorted(self._items)})"


def build_from_config(
    registry: Registry,
    cfg: Optional[Mapping[str, Any]],
    *,
    extra_kwargs: Optional[Mapping[str, Any]] = None,
    default: Optional[str] = None,
) -> Any:
    """按配置实例化注册表中的组件。

    Args:
        registry: 目标注册表。
        cfg: 组件配置，需含 `_target_`；为 None 时使用 `default`。
        extra_kwargs: 追加的关键字参数（同名时以 cfg 为准）。
        default: cfg 为空时使用的默认注册名。
    """
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, Mapping):
        raise TypeError(f"组件配置必须是映射，实际为 {type(cfg).__name__}")

    target = cfg.get("_target_", default)
    if target is None:
        raise ValueError(
            f"组件配置缺少 `_target_` 字段，无法在注册表 [{registry.name}] 中定位实现。"
            f"配置内容: {dict(cfg)}"
        )

    factory = registry.get(str(target))
    kwargs: Dict[str, Any] = dict(extra_kwargs or {})
    kwargs.update({k: v for k, v in cfg.items() if k != "_target_"})
    return factory(**kwargs)


def register(registry_name: str, key: str, obj: Callable[..., Any], *, force: bool = False) -> None:
    """向全局命名空间中的注册表登记对象，注册表不存在时自动创建。"""
    registry = REGISTRIES.get(registry_name)
    if registry is None:
        registry = Registry(registry_name)
    registry.register_obj(key, obj, force=force)
