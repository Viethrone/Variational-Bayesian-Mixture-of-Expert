"""Losses (data term).

``VBELBOCore`` registers the ``vbelbo_1d`` / ``vbelbo_2d`` factories and is built from
``cfg["loss"]``; add new losses with ``@LOSSES.register("name")``.
"""

from src.losses.base import LOSSES
from src.losses.vbelbo import VBELBOCore

__all__ = ["LOSSES", "VBELBOCore"]
