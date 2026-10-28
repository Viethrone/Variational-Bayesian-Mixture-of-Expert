"""Forward-operator registry.

Concrete operators register themselves here and are built from a config
``_target_`` entry; ``SeismicForward`` (``synthetic_seismic``) is the only one the
released recipes use.
"""

from __future__ import annotations

from src.utils.registry import Registry

__all__ = ["FORWARD_OPERATORS"]

FORWARD_OPERATORS = Registry("forward_operator")
