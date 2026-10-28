"""Priors: GMM fitting, spatial prior generation and the differentiable GMM prior."""

from src.priors.base import PRIORS
from src.priors.gmm import (  # noqa: F401  (registers gmm_diag / gmm_full)
    GMMPrior,
    GMMPriorFitter,
    SpatialPriorGenerator,
    hungarian_match,
)

__all__ = [
    "PRIORS",
    "GMMPrior",
    "GMMPriorFitter",
    "SpatialPriorGenerator",
    "hungarian_match",
]
