"""Priors: GMM fitting, spatial prior generation and the differentiable GMM prior.

Two layers, and it matters which one a config touches:

* ``prior._target_`` in a config is **not** a registry lookup. The training entry point
  reads it as a *source selector* for the parameter prior, because the three supported
  choices have different inputs (see ``experiments/main/run_main.py``):

  - ``none`` (or omitted) - unit prior ``N(0, I)``, nothing injected;
  - ``gaussian``          - one Gaussian shared by all experts, from the training-target
                            z-score statistics;
  - ``gmm``               - the K-component GMM stored with the processed dataset
                            (``gmm_prior.npz``), refit on the train subset when the model
                            asks for more experts than the file has modes.

  All three end up as the model's ``prior_means`` / ``prior_vars`` buffers, so the choice
  never changes the code path inside the model.

* ``PRIORS`` is the registry for *callable* prior objects with the ``log_prob`` /
  ``grad_log_prob`` interface (``gmm_diag`` / ``gmm_full``). The training loop does not
  need them - the model carries its prior as buffers - but they are the supported way to
  evaluate a prior numerically or to plug one into an iterative solver.
"""

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