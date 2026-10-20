"""Geo-VB-MILE core library.

Submodules::

    data        datasets, loading and normalization
    forward     forward and observation operators
    models      network structures (GeoVBMILE 1-D / 2-D)
    priors      prior models (GMM parameter prior, spatial prior)
    losses      loss functions (negative ELBO)
    metrics     evaluation metrics
    trainers    training / evaluation loops
    utils       configuration, registry, seeding, logging, run metadata

Every replaceable component is registered in ``src.utils.registry`` and instantiated
from the ``_target_`` field of the configuration; concrete classes are never hard-coded
in business logic.
"""

__version__ = "1.0.0"
