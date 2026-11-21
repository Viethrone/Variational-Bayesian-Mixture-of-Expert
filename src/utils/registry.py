"""Component registry: resolve a `_target_` string from a config into a real object.

Goal: solvers, forward operators, priors, regularisers, losses, metrics and models can all
be swapped without touching the caller; an ablation only edits YAML.

Usage::

    from src.utils.registry import Registry, build_from_config

    SOLVERS = Registry("solver")

    @SOLVERS.register("gradient_descent")
    class GradientDescent:
        def __init__(self, max_iter=100, lr=1e-2):
            ...

    solver = build_from_config(SOLVERS, cfg["inversion"]["solver"])

`build_from_config` passes every key except `_target_` to the constructor as a keyword.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, TypeVar

__all__ = ["Registry", "REGISTRIES", "build_from_config", "register"]

T = TypeVar("T")

# global namespace, so introspection and cross-module lookup stay uniform
REGISTRIES: Dict[str, "Registry"] = {}


class Registry:
    """Lightweight registry mapping a name to a callable (class or factory)."""

    def __init__(self, name: str, *, global_scope: bool = True) -> None:
        self.name = name
        self._items: Dict[str, Callable[..., Any]] = {}
        if global_scope:
            if name in REGISTRIES and REGISTRIES[name] is not self:
                raise ValueError(f"registry name clash: {name!r} already exists")
            REGISTRIES[name] = self

    # ---- registration ----
    def register(self, key: Optional[str] = None, *, force: bool = False) -> Callable[[T], T]:
        """Decorator form. Without `key` the object's `__name__` (lower-cased) is used."""

        def decorator(obj: T) -> T:
            name = key or getattr(obj, "__name__", None)
            if not name:
                raise ValueError("cannot infer a registry key; pass `key` explicitly")
            if isinstance(name, str):
                name = name.lower()
            if name in self._items and not force:
                existing = self._items[name]
                if existing is not obj:
                    raise ValueError(
                        f"[{self.name}] key {name!r} is already taken by {existing!r}; "
                        f"pass force=True to override"
                    )
            self._items[name] = obj
            return obj

        return decorator

    def register_obj(self, key: str, obj: Callable[..., Any], *, force: bool = False) -> None:
        """Functional form, useful for registering third-party implementations."""
        if key in self._items and not force and self._items[key] is not obj:
            raise ValueError(f"[{self.name}] key {key!r} already exists")
        self._items[key] = obj

    # ---- lookup ----
    def get(self, key: str) -> Callable[..., Any]:
        if key not in self._items:
            raise KeyError(self._missing_message(key))
        return self._items[key]

    def contains(self, key: str) -> bool:
        return key in self._items

    def keys(self) -> Iterable[str]:
        return self._items.keys()

    def _missing_message(self, key: str) -> str:
        available = ", ".join(sorted(self._items)) or "(empty)"
        return (
            f"[{self.name}] {key!r} is not registered. Registered: {available}."
            f"Hint: make sure the module has been imported (registration happens on import)."
        )

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._items

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Registry({self.name!r}, items={sorted(self._items)})"


def build_from_config(
    registry: Registry,
    cfg: Optional[Mapping[str, Any]],
    *,
    extra_kwargs: Optional[Mapping[str, Any]] = None,
    default: Optional[str] = None,
) -> Any:
    """Instantiate a registered component from a config.

    Args:
        registry: target registry.
        cfg: component config containing `_target_`; `default` is used when it is None.
        extra_kwargs: additional keyword arguments (cfg wins on a clash).
        default: registry key used when cfg is empty.
    """
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, Mapping):
        raise TypeError(f"component config must be a mapping, got {type(cfg).__name__}")

    target = cfg.get("_target_", default)
    if target is None:
        raise ValueError(
            f"component config has no `_target_`, so [{registry.name}] cannot resolve an implementation."
            f"config: {dict(cfg)}"
        )

    factory = registry.get(str(target))
    kwargs: Dict[str, Any] = dict(extra_kwargs or {})
    kwargs.update({k: v for k, v in cfg.items() if k != "_target_"})
    return factory(**kwargs)


def register(registry_name: str, key: str, obj: Callable[..., Any], *, force: bool = False) -> None:
    """Register an object with a registry in the global namespace, creating it if needed."""
    registry = REGISTRIES.get(registry_name)
    if registry is None:
        registry = Registry(registry_name)
    registry.register_obj(key, obj, force=force)
