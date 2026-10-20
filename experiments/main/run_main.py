"""Training / inference entry point for Geo-VB-MILE (1-D and 2-D).

Usage::

    python -m experiments.main.run_main -c configs/inversion/vbmile_1d.yaml --name ours_full
    python -m experiments.main.run_main -c configs/inversion/vbmile_2d.yaml --name ours_2d

Steps (this module only orchestrates):

1. resolve the configuration (``_base_`` chain + command-line ``--set`` overrides)
2. create the run directory and write its metadata (config copy, command, commit, env)
3. assemble the model, loss, forward operator, data loaders and statistics, and inject
   the GMM prior read from the processed dataset
4. run the inversion and write the metrics

Every run lands in ``outputs/runs/<timestamp>_<name>/`` and never overwrites earlier
runs, which keeps experiment history auditable and reproducible.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any, Dict, Optional

import numpy as np
import torch

import src.forward.seismic  # noqa: E402,F401  (registers `synthetic_seismic`)
from experiments.common import build_dataset, make_parser, prepare_run, resolve_config, save_cmd
from src.trainers.base import build_trainer
from src.utils.logging_utils import get_logger
from src.utils.registry import build_from_config

logger = get_logger(__name__)


def _refit_gmm_prior_k(dataset: Any, K: int) -> Dict[str, Any]:
    """Fit a K-component GMM prior (normalized space) on the training samples.

    Used when the configured number of experts exceeds the components present in
    ``gmm_prior.npz``. The pixels come from the train subset (physical-space target,
    columns selected by sample index) and are normalized before fitting with
    ``GMMPriorFitter(K)`` (full covariance, shrinkage 0.1, ``eps*I``, seed 42).

    Returns:
        ``{"prior_means": (K,3), "prior_vars": (K,3,3)}``
    """
    from src.priors.gmm import GMMPriorFitter

    target = np.asarray(getattr(dataset, "target", None), dtype=np.float32)
    if target is None or target.ndim not in (2, 3):
        raise RuntimeError(f"cannot read the target field from dataset {type(dataset).__name__}")
    if target.ndim == 2:
        target = target[:, :, None]
    C, T, X = target.shape
    idx = [int(s) for s in getattr(dataset, "sample_names", [])]
    idx = [i for i in idx if 0 <= i < X]
    cols = target[:, :, idx].transpose(1, 2, 0).reshape(-1, C).astype(np.float64)

    mu = np.asarray(dataset.target_mean, dtype=np.float64).ravel()          # (C,)
    sd = np.asarray(dataset.target_std, dtype=np.float64).ravel()
    X_norm = (cols - mu) / sd
    prior = GMMPriorFitter(K=int(K), seed=42).fit(X_norm)
    return {
        "prior_means": np.asarray(prior["means"], dtype=np.float32),
        "prior_vars": np.asarray(prior["covariances"], dtype=np.float32),
    }


def _assemble_vbmile_trainer(trainer: Any, cfg: Dict[str, Any], run_info: Any) -> Any:
    """Build the model / loss / forward / loaders and inject them into ``trainer``.

    The GMM parameter prior (normalized space) is taken from the dataset's
    ``gmm_prior`` and passed to the model as ``prior_means`` / ``prior_vars``, which
    initialize the expert output heads and drive the per-expert ``KL(theta)`` term.
    """
    # import side effects register the concrete implementations
    import src.forward.seismic  # noqa: F401
    import src.losses  # noqa: F401
    import src.models  # noqa: F401
    from src.forward.base import FORWARD_OPERATORS
    from src.losses.base import LOSSES
    from src.models.base import MODELS
    from src.trainers.vbmile_trainer import build_vbmile_loader

    train_dataset = build_dataset(cfg, split="train")
    val_dataset = build_dataset(cfg, split="val")

    # ---- model: GMM parameter prior (normalized space (K,3)) ----
    # ``prior._target_: none`` disables the prior (unit-prior ablation)
    mcfg = dict(cfg.get("model") or {})
    m_target = mcfg.pop("_target_", None) or "vbmile_1d"
    mcfg.pop("prior_means", None)
    mcfg.pop("prior_vars", None)
    prior_target = str((cfg.get("prior") or {}).get("_target_", "") or "").lower()
    prior_kwargs: Dict[str, Any] = {}
    if prior_target not in ("", "none"):
        if prior_target == "gaussian":
            # single Gaussian prior shared by all experts = global z-score statistics
            # of the training targets (the targets are already z-scored, so this is
            # close to N(0, I) — qualitatively different from the 3-modal GMM)
            tmean = np.asarray(train_dataset.target_mean, np.float32).reshape(1, -1)
            tstd = np.asarray(train_dataset.target_std, np.float32).reshape(1, -1)
            K = int(mcfg.get("num_experts", 3))
            prior_kwargs = {
                "prior_means": np.repeat(tmean, K, axis=0),
                "prior_vars": np.repeat(tstd ** 2, K, axis=0),
            }
        else:
            gp = train_dataset.gmm_prior
            K = int(mcfg.get("num_experts", 3))
            if K <= len(np.asarray(gp["means"])):
                # fewer experts than modes: keep the first K (smoke / experts_2 ablation).
                # full covariance (K,3,3) preferred, legacy diagonal file falls back to
                # variances (K,3).
                prior_vars = (np.asarray(gp["covariances"]) if "covariances" in gp
                              else np.asarray(gp["variances"]))
                prior_kwargs = {
                    "prior_means": np.asarray(gp["means"])[:K],
                    "prior_vars": prior_vars[:K],
                }
            else:
                # more experts than available modes: refit a K-component GMM on the
                # normalized train subset (same shrinkage recipe as GMMPriorFitter).
                prior_kwargs = _refit_gmm_prior_k(train_dataset, K)
                logger.info("K=%d exceeds the available GMM modes; refit %d on the train subset",
                            K, K)
    model = build_from_config(MODELS, {"_target_": m_target, **mcfg}, extra_kwargs=prior_kwargs)

    # ---- loss / forward (the device is resolved by the trainer) ----
    criterion = build_from_config(LOSSES, cfg.get("loss"))
    forward = build_from_config(FORWARD_OPERATORS, cfg.get("forward"))

    # ---- theory contract: read from the actual objects, write <run>/theory_contract.json
    # so every run carries a machine-readable "what was really computed" record (full vs
    # diagonal posterior, closed-form vs moment-matched well likelihood, weighted vs strict
    # KL objective) plus its consistency checks.
    from src.utils.theory_contract import build_theory_contract, persist_theory_contract

    tc_cfg = dict(cfg.get("theory_contract") or {})
    require_full = bool(tc_cfg.get("require_full", False))
    contract = build_theory_contract(model, criterion, require_full=require_full)
    persist_theory_contract(run_info, contract, strict=bool(tc_cfg.get("strict", False)))
    if hasattr(trainer, "cfg") and isinstance(getattr(trainer.cfg, "extra", None), dict):
        trainer.cfg.extra["theory_contract"] = {
            "variational_covariance": contract["variational"]["covariance"],
            "kl_mode": contract["objective"]["kl"].get("mode"),
            "kl_temperature": contract["objective"]["kl"].get("temperature"),
            "noise_mode": contract["objective"]["noise_mode"],
            "well_nll_mode": contract["objective"]["well_nll_mode"],
            "require_full": require_full,
            "all_ok": contract["all_ok"],
            "mandatory_failures": contract["mandatory_failures"],
        }

    # ---- normalization statistics (per-channel) ----
    stats = {
        "target_mean": torch.as_tensor(np.asarray(train_dataset.target_mean, np.float32)),
        "target_std": torch.as_tensor(np.asarray(train_dataset.target_std, np.float32)),
        "seismic_mean": torch.as_tensor(np.asarray(train_dataset.seismic_mean, np.float32)),
        "seismic_std": torch.as_tensor(np.asarray(train_dataset.seismic_std, np.float32)),
    }

    # ---- loaders ----
    batch_size = int((cfg.get("trainer") or {}).get("batch_size")
                     or (cfg.get("data") or {}).get("loader", {}).get("batch_size")
                     or 20)
    num_workers = int((cfg.get("data") or {}).get("loader", {}).get("num_workers", 0))
    num_experts = int(mcfg.get("num_experts", 3))
    # explicit random source (``data.loader.explicit_generator: true``, default off):
    # both the shuffle order and the worker base seed then come from this generator
    # instead of the global RNG, so ``num_workers`` no longer moves the trajectory
    loader_gen = None
    if bool((cfg.get("data") or {}).get("loader", {}).get("explicit_generator", False)):
        loader_gen = torch.Generator()
        loader_gen.manual_seed(int(cfg.get("seed", 0)))
    train_loader = build_vbmile_loader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        num_experts=num_experts, generator=loader_gen,
    )
    val_loader = build_vbmile_loader(
        val_dataset, batch_size=batch_size, shuffle=False, num_experts=num_experts
    )

    # ---- inject into the trainer ----
    trainer.model = model
    trainer.criterion = criterion
    trainer.forward = forward
    trainer.train_loader = train_loader
    trainer.val_loader = val_loader
    trainer.stats = stats
    # the top-level config (e.g. ``checkpoint.save_last``) is read through this attribute
    trainer.full_config = cfg
    return trainer


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code."""
    parser = make_parser(
        "Geo-VB-MILE: train / invert (1-D and 2-D)",
        default_config="configs/inversion/vbmile_1d.yaml",
    )
    args = parser.parse_args(argv)

    cfg = resolve_config(args)
    run_info = prepare_run(cfg, args)
    save_cmd(run_info, sys.argv if argv is None else ["experiments.main.run_main", *(argv or [])])

    logger.info("experiment: %s", cfg.get("exp_name"))

    trainer = build_trainer(cfg, run_info)
    if (cfg.get("trainer") or {}).get("_target_") != "vbmile":
        logger.error("this entry point only builds the Geo-VB-MILE trainer "
                     "(trainer._target_: vbmile)")
        return 2

    trainer = _assemble_vbmile_trainer(trainer, cfg, run_info)
    metrics = trainer.run()
    logger.info("done. metrics: %s",
                {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))})
    logger.info("run dir: %s", run_info.run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
