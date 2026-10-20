"""Theory contract: record what the code actually computed, and check it.

The point is alignment between the paper, the code and the results. A run can claim a
full-covariance posterior, an analytic expected well likelihood or a tempered posterior
while the code silently computes a diagonal posterior, a moment-matched NLL or unrelated
KL weights. Without a machine-readable record that divergence can only be found by reading
code, and the resulting numbers cannot be traced back.

At the start of every run this module reads the contract from the **actual model and
criterion objects** (not from the config, which may be rewritten by prior injection or by a
phase override), writes it to ``<run_dir>/theory_contract.json`` and runs five explicit
consistency checks:

1. whether ``well_nll_mode`` matches ``variational_covariance`` (an expert-level well term
   needs the full posterior to provide ``tr(Sigma_k)`` / the off-diagonals);
2. whether the KL weights form a strictly writable objective (``elbo`` /
   ``single_temperature`` / ``legacy_weighted``, see
   ``VBELBOCore.kl_objective_contract``);
3. whether the noise mode matches the paper's fixed-noise statement;
4. for a full posterior with a full-covariance prior: whether
   ``prior_cov_shrinkage`` is switched on;
5. whether ``prior_cov_shrinkage`` is actually effective (it is silently ignored when the
   prior is not full covariance).

Only the **mandatory** checks can raise (``strict=True``); weighted objectives and learned
noise are legitimate choices and are merely recorded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

__all__ = ["build_theory_contract", "persist_theory_contract"]

logger = get_logger(__name__)


def build_theory_contract(model: Any, criterion: Any,
                          *, require_full: bool = False) -> Dict[str, Any]:
    """Read the contract off the **actual objects** (never off the config).

    Args:
        model: a ``GeoVBMILE`` instance (or anything exposing the same attributes).
        criterion: a ``VBELBOCore`` instance.
        require_full: when true, "the variational family must be full" becomes a
            **mandatory** check. The delivery configuration enables it; historical
            reproductions that really are diagonal keep it false.

    Returns:
        A JSON-serialisable dict with the ``variational`` / ``objective`` sections and the
        ``checks`` list. ``all_ok`` only counts ``mandatory`` checks — weighted objectives
        and learned noise are allowed and simply recorded.
    """
    variational_cov = str(getattr(model, "variational_covariance", "diag"))
    prior_vars = getattr(model, "prior_vars", None)
    prior_ndim = int(prior_vars.ndim) if prior_vars is not None else None

    kl_contract: Dict[str, Any] = {"mode": "unknown", "strict_posterior": None}
    if hasattr(criterion, "kl_objective_contract"):
        kl_contract = criterion.kl_objective_contract()

    noise_mode = str(getattr(criterion, "noise_mode", "learned"))
    well_nll_mode = str(getattr(criterion, "well_nll_mode", "marginal"))
    likelihood = str(getattr(criterion, "likelihood", "gaussian"))

    checks: List[Dict[str, Any]] = []

    def _check(name: str, ok: bool, detail: str, *, mandatory: bool = True) -> None:
        """Register one check.

        ``mandatory`` separates two classes: **mandatory** checks (the default) treat a
        failure as a contract incident and raise under ``strict=True``; **recorded** checks
        are legitimate contract choices of this release, written into the json for the
        paper wording and cross-run summaries, but never blocking (``all_ok`` only counts
        mandatory ones).
        """
        checks.append({"check": name, "ok": bool(ok), "detail": detail,
                       "mandatory": bool(mandatory)})

    # 0) Full covariance is the delivery requirement: q(theta|z=k) = N(mu_k, Sigma_k).
    #    ``require_full`` is explicit so that a diagonal reproduction is not judged as an
    #    incident by default.
    _check(
        "variational_is_full",
        variational_cov == "full",
        (f"variational_covariance={variational_cov!r}; the delivery contract requires full"
         if require_full else
         f"full covariance not required for this run (require_full=False), "
         f"current {variational_cov!r}"),
        mandatory=bool(require_full),
    )

    # 1) An expert-level well likelihood needs the full posterior (otherwise
    #    expert_tr_sigma / expert_off do not exist).
    needs_full = well_nll_mode in ("expert", "mixture_predictive")
    _check(
        "well_nll_mode_requires_full_q",
        (not needs_full) or variational_cov == "full",
        (f"well_nll_mode={well_nll_mode!r} needs a full variational posterior "
         f"(current {variational_cov!r})"
         if needs_full else "expert-level well likelihood is off; full q not required"),
    )

    # 2) KL objective: recorded, never rejected. A weighted objective is legitimate; what
    #    matters is that the paper can be worded to match.
    strict = kl_contract.get("strict_posterior")
    mode = kl_contract.get("mode")
    temp = kl_contract.get("temperature")
    _check(
        "kl_objective_recorded",
        True,
        (f"KL weight mode = {mode}"
         + (f", single temperature c={temp}" if temp is not None else "")
         + "; a weighted objective is allowed (full covariance mandatory, strict ELBO "
           "optional); the text must state the weights/temperature"
         + (f". Note: {kl_contract.get('note')}" if kl_contract.get("note") else "")),
        mandatory=False,
    )
    _check(
        "kl_objective_strict",
        bool(strict) if strict is not None else False,
        (f"KL weight mode = {mode}; "
         + ("strictly writable objective (ELBO or single temperature)" if strict else
            "weighted objective -- allowed, but the text must not call it a strict ELBO; "
            "state the weights/temperature")),
        mandatory=False,
    )

    # 3) Noise: recorded only (fixed and jointly estimated are both allowed).
    _check(
        "noise_mode_matches_paper",
        noise_mode == "fixed",
        (f"noise_mode={noise_mode!r}; the paper declares fixed noise, so a learned mode "
         "must be described as jointly estimated with its initial values"),
        mandatory=False,
    )

    # 4) Full posterior with a near-singular full prior: is shrinkage on?
    shrinkage = float(getattr(model, "prior_cov_shrinkage", 0.0))
    if variational_cov == "full" and prior_ndim == 3:
        _check(
            "prior_cov_shrinkage_on_for_full",
            shrinkage > 0.0,
            f"full q + full prior with prior_cov_shrinkage={shrinkage} "
            "(the GMM full covariance can reach condition number 5804; 0.05-0.1 advised)",
        )

    # 5) Shrinkage is a paired switch: a value with a non-full prior does nothing at all.
    #    This is the "configured but not effective" class of incident, invisible in config.
    _check(
        "prior_cov_shrinkage_is_effective",
        shrinkage <= 0.0 or prior_ndim == 3,
        (f"prior_cov_shrinkage={shrinkage} is set, but the prior covariance has ndim="
         f"{prior_ndim} (not full) => the switch is inert. Configure prior._target_: gmm, "
         "or drop prior_cov_shrinkage so the contract is not misread."),
    )

    return {
        "variational": {
            "covariance": variational_cov,
            "prior_covariance_ndim": prior_ndim,
            "prior_cov_shrinkage": shrinkage,
            "num_experts": int(getattr(model, "num_experts", 0) or 0),
            "dim": int(getattr(model, "dim", 0) or 0),
        },
        "objective": {
            "kl": kl_contract,
            "noise_mode": noise_mode,
            "well_nll_mode": well_nll_mode,
            "likelihood": likelihood,
            "seismic_noise": str(getattr(criterion, "seismic_noise", "iid")),
        },
        "checks": checks,
        # mandatory checks only: weighted objectives / learned noise / strictness are
        # allowed contract choices
        "all_ok": all(c["ok"] for c in checks if c.get("mandatory", True)),
        "mandatory_failures": [c["check"] for c in checks
                               if c.get("mandatory", True) and not c["ok"]],
        "recorded_non_mandatory": [c["check"] for c in checks
                                   if not c.get("mandatory", True)],
    }


def persist_theory_contract(run_info: Any, contract: Dict[str, Any],
                            *, strict: bool = False,
                            name: str = "theory_contract.json") -> Optional[Path]:
    """Write the contract into the run directory and warn (or raise, with ``strict``).

    Only ``mandatory`` checks warn or block; non-mandatory ones (weighted objective,
    learned noise, KL strictness) are recorded silently because they are allowed choices.

    Args:
        run_info: a ``RunInfo`` (needs ``write_text``).
        contract: the return value of ``build_theory_contract``.
        strict: raise ``RuntimeError`` when any **mandatory** check fails.
        name: file name; a phase boundary can write ``theory_contract_phase2.json`` to keep
            the per-phase contract.
    """
    text = json.dumps(contract, ensure_ascii=False, indent=2, default=str) + "\n"
    path = None
    if run_info is not None and hasattr(run_info, "write_text"):
        path = run_info.write_text(name, text)

    checks = contract.get("checks", [])
    failed = [c for c in checks if c.get("mandatory", True) and not c["ok"]]
    for c in failed:
        logger.warning("theory contract check failed [%s]: %s", c["check"], c["detail"])
    if not failed:
        logger.info("theory contract: all mandatory checks passed "
                    "(variational=%s, kl=%s, noise=%s)",
                    contract["variational"]["covariance"],
                    contract["objective"]["kl"].get("mode"),
                    contract["objective"]["noise_mode"])
    recorded = [c for c in checks if not c.get("mandatory", True)]
    if recorded:
        logger.info("theory contract (recorded): %s", "; ".join(
            f"{c['check']}={'ok' if c['ok'] else 'no'}" for c in recorded))
    if strict and failed:
        raise RuntimeError(
            "theory contract mandatory checks failed (strict=True): "
            + "; ".join(f"{c['check']}: {c['detail']}" for c in failed)
        )
    return path
