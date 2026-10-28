"""Negative-ELBO registry (``LOSSES``).

``VBELBOCore`` registers the ``vbelbo_1d`` / ``vbelbo_2d`` factories; the entry point
builds the criterion from ``cfg["loss"]``.
"""

from __future__ import annotations

from src.utils.registry import Registry

__all__ = ["LOSSES"]

LOSSES = Registry("loss")
