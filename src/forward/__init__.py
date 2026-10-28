"""Forward operators: Aki-Richards reflectivity convolved with a Ricker wavelet."""

from __future__ import annotations

from src.forward.base import FORWARD_OPERATORS

__all__ = ["FORWARD_OPERATORS"]

# import the concrete operator to trigger registration
from src.forward.seismic import SeismicForward  # noqa: E402,F401
