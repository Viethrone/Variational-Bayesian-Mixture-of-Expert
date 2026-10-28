"""Negative ELBO loss for the Geo-VB-MILE (VB-MoE) method.

The objective is::

    neg_elbo = nll_well + seismic_nll + kl_z_weight * KL(z) + kl_theta_weight * KL(theta)

where

* ``nll_well``    — expected well likelihood, a Gaussian NLL over the predicted mean and
  variance plus the (learnable) observation noise variance;
* ``seismic_nll`` — Gaussian NLL of the observed gathers given the forward model of a
  ``theta`` sample;
* ``kl_z``        — KL between the gating distribution and the prior partition;
* ``kl_theta``    — per-expert KL between the expert parameter distribution and the
  parameter prior, weighted by the gating probabilities.

The observation noise variances (well / seismic) are learnable parameters.
"""

from __future__ import annotations

import math
from typing import List, Optional, Union

import torch
import torch.nn as nn

from src.forward.reliability import mask_stats, masked_mean
from src.losses.base import LOSSES
from src.utils.fullcov import build_chol

__all__ = ["VBELBOCore"]


class VBELBOCore(nn.Module):
    """Negative ELBO loss for Geo-VB-MILE.

    The objective is::

        neg_elbo = nll_well + seismic_nll + kl_z_weight * KL(z) + kl_theta_weight * KL(theta)

    Args:
        seismic_obs_noise_init: initial seismic observation noise standard deviation.
        well_obs_noise_init: initial well observation noise standard deviation.
        clip_diff: residual clipping magnitude (outlier suppression).
        kl_z_weight: weight of the gating KL term.
        kl_theta_weight: weight of the parameter KL term.
        seismic_weight: weight of the seismic likelihood term (0 disables it).
        expert_lambdas: per-expert weights for ``KL(theta)`` (``(K,)`` tensor or list);
            ``None`` means uniform weighting.
        well_nll_mode: well-likelihood convention.

            * ``"marginal"`` (default) - Gaussian NLL of the moment-matched marginal
              ``N(y; mu_marg, sigma_y^2 + Var_marg)``; the 2-D recipe.
            * ``"expert"`` - analytic ``E_q[-log p(y|theta)]`` using ``tr(Sigma_k)``,
              which needs the full-covariance posterior; the 1-D recipe.
            * ``"mixture"`` - ``-logsumexp_k(log q_k - nll_k)`` with
              ``sigma_y^2 + sigma_k^2`` per expert.
            * ``"mixture_predictive"`` - ``-log sum_k pi_k N(y; mu_k, sigma_y^2 I +
              Sigma_k)``; the only mode in which the off-diagonal structure enters.
        noise_mode: ``"learned"`` (default) makes the observation variances learnable;
            ``"fixed"`` keeps them at their initial values.
        seismic_noise: ``"iid"`` (default), ``"coherent"`` (trace common mode with
            ``C = sigma_c^2 11^T + sigma_w^2 I``) or ``"deviatoric"`` (the common mode is
            marginalised, so only trace-to-trace differences are constrained). The
            correlated models are 2-D only.
        seismic_coh_noise_init: initial common-mode noise level (unused for ``"iid"``).
        seismic_white_noise_init: initial white noise level (unused for ``"iid"``).
        seismic_mask_mode: per-angle reliability masking, ``"none"`` (default) or
            ``"critical_angle"``. The weight comes from
            :func:`src.forward.reliability.critical_angle_weights` and the seismic term is
            then reduced over the reliable region only; ``"none"`` is bit-wise the
            released path.
        seismic_mask_layer_sigma: depth smoothing scale of Vp, in samples (0 = adjacent
            samples, matching the forward stencil).
        seismic_mask_taper_deg: width of the cosine taper below the critical angle.
        seismic_mask_floor: weight kept beyond the critical angle, in ``[0, 1]``.
    """

    def __init__(self,
                 seismic_obs_noise_init: float = 1.0,
                 well_obs_noise_init: float = 0.1,
                 clip_diff: float = 10.0,
                 kl_z_weight: float = 1.0,
                 kl_theta_weight: float = 1.0,
                 seismic_weight: float = 1.0,
                 expert_lambdas: Optional[Union[List[float], str]] = None,
                 well_nll_mode: str = "marginal",
                 noise_mode: str = "learned",
                 seismic_noise: str = "iid",
                 seismic_coh_noise_init: float = 1.0,
                 seismic_white_noise_init: float = 1.0,
                 seismic_mask_mode: str = "none",
                 seismic_mask_layer_sigma: float = 0.0,
                 seismic_mask_taper_deg: float = 5.0,
                 seismic_mask_floor: float = 0.0):
        super().__init__()
        if noise_mode not in ("learned", "fixed"):
            raise ValueError(f"noise_mode must be learned or fixed, got {noise_mode!r}")
        # learned = the legacy layout (parameters in the optimizer); fixed = buffers at the
        # initial values, matching a paper statement of fixed noise. Note that a fixed
        # sigma_y makes the well-likelihood scale hard, so calibrate it on the training
        # residual RMS rather than letting it learn.
        self.noise_mode = str(noise_mode)
        if self.noise_mode == "fixed":
            self.register_buffer(
                "log_seismic_var", torch.tensor(math.log(seismic_obs_noise_init ** 2)))
            self.register_buffer(
                "log_well_var", torch.tensor(math.log(well_obs_noise_init ** 2)))
        else:
            self.log_seismic_var = nn.Parameter(
                torch.tensor(math.log(seismic_obs_noise_init ** 2)))
            self.log_well_var = nn.Parameter(
                torch.tensor(math.log(well_obs_noise_init ** 2)))
        self.clip_diff = clip_diff
        self.log_two_pi = math.log(2 * math.pi)

        if well_nll_mode not in ("marginal", "mixture", "expert", "mixture_predictive"):
            raise ValueError(
                "well_nll_mode must be marginal/mixture/expert/mixture_predictive, "
                f"got {well_nll_mode!r}")
        self.well_nll_mode = str(well_nll_mode)

        if seismic_noise not in ("iid", "coherent", "deviatoric"):
            raise ValueError(
                f"seismic_noise must be iid/coherent/deviatoric, got {seismic_noise!r}")
        self.seismic_noise = str(seismic_noise)
        # set by the registered factory (``vbelbo_1d`` / ``vbelbo_2d``); a plain attribute,
        # so it never enters the state dict. ``None`` means "unknown" and is accepted.
        self.loss_dim: Optional[int] = None
        if self.seismic_noise != "iid":
            sc2 = float(seismic_coh_noise_init) ** 2
            sw2 = float(seismic_white_noise_init) ** 2
            if sc2 <= 0.0 or sw2 <= 0.0:
                raise ValueError("seismic_coh/white_noise_init must be > 0")
            if self.seismic_noise == "coherent":
                self.log_seismic_coh_var = nn.Parameter(torch.tensor(math.log(sc2)))
            else:
                # deviatoric: the common mode is marginalised, i.e. its variance is fixed
                # at a very large value and never optimised
                self.register_buffer("_seis_coh_var_fixed", torch.tensor(1.0e6))
            self.log_seismic_white_var = nn.Parameter(torch.tensor(math.log(sw2)))

        # ---- per-angle reliability masking (opt-in; "none" = bit-wise legacy) ----
        # Plain attributes, so the state_dict key set is unchanged; the convention lives in
        # src/forward/reliability.py and the class docstring.
        if seismic_mask_mode not in ("none", "critical_angle"):
            raise ValueError(
                "seismic_mask_mode must be none/critical_angle, "
                f"got {seismic_mask_mode!r}")
        if float(seismic_mask_layer_sigma) < 0.0:
            raise ValueError(
                f"seismic_mask_layer_sigma must be >= 0, got {seismic_mask_layer_sigma!r}")
        if float(seismic_mask_taper_deg) < 0.0:
            raise ValueError(
                f"seismic_mask_taper_deg must be >= 0, got {seismic_mask_taper_deg!r}")
        if not 0.0 <= float(seismic_mask_floor) <= 1.0:
            raise ValueError(
                f"seismic_mask_floor must be in [0, 1], got {seismic_mask_floor!r}")
        self.seismic_mask_mode = str(seismic_mask_mode)
        self.seismic_mask_layer_sigma = float(seismic_mask_layer_sigma)
        self.seismic_mask_taper_deg = float(seismic_mask_taper_deg)
        self.seismic_mask_floor = float(seismic_mask_floor)

        self.kl_z_weight = float(kl_z_weight)
        self.kl_theta_weight = float(kl_theta_weight)
        self.seismic_weight = float(seismic_weight)
        if isinstance(expert_lambdas, str):
            expert_lambdas = [float(v) for v in expert_lambdas.split(",")]
        if expert_lambdas is not None:
            expert_lambdas = torch.as_tensor(list(expert_lambdas), dtype=torch.float32)
            assert expert_lambdas.dim() == 1
        self.expert_lambdas = expert_lambdas

    @property
    def seismic_var(self) -> torch.Tensor:
        return torch.exp(self.log_seismic_var)

    @property
    def well_var(self) -> torch.Tensor:
        return torch.exp(self.log_well_var)

    def _reduce_well_nll(self, nll, mask):
        """Well-term reduction, shared by both well NLL modes."""
        if mask is not None:
            if mask.shape[1] == 1:
                mask = mask.expand_as(nll)
            nll = nll * mask
            return nll.sum() / (mask.sum() + 1e-8)
        return nll.mean()

    def _reduce_pixel_nll(self, nll, mask, where="well_nll"):
        """Reduce a per-pixel well NLL whose property axis is already summed out.

        The mask must first be reduced to the same rank: multiplying a ``(B,1,L)`` mask by a
        ``(B,L)`` NLL broadcasts to ``(B,B,L)`` and rescales the loss by the batch size.
        """
        if mask is None:
            return nll.mean()
        pix = mask
        while pix.dim() > nll.dim():
            # per-property mask (B,3,.) -> (B,.); amax over a (B,1,.) mask is the identity
            pix = pix.amax(dim=1)
        if pix.shape != nll.shape:
            raise ValueError(
                f"{where}: reduced mask shape {tuple(pix.shape)} does not match the NLL "
                f"shape {tuple(nll.shape)} -- multiplying would broadcast silently and "
                "inflate the loss. Reduce the mask to per-pixel at the call site.")
        pix = pix.to(nll.dtype)
        return (nll * pix).sum() / (pix.sum() + 1e-8)

    def _nll_well(self, mu, logvar, target, mask):
        """Expected well likelihood (marginal mode): N(y; mu_marg, sigma_y^2 + Var_marg)."""
        var_theta = torch.exp(logvar)
        total_var = var_theta + self.well_var
        diff = (target - mu).clamp(-self.clip_diff, self.clip_diff)
        nll = 0.5 * (diff**2 / total_var + self.log_two_pi + torch.log(total_var))
        return self._reduce_well_nll(nll, mask)

    def _require_expert_keys(self, model_output, keys) -> None:
        """Fail loudly when an expert-level mode lacks the keys it needs.

        A silent fallback would make the declared contract diverge from what is computed.
        """
        missing = [k for k in keys if k not in model_output]
        if missing:
            raise ValueError(
                f"well_nll_mode={self.well_nll_mode!r} needs {missing} in model_output, "
                "but they are absent. Set model.variational_covariance='full' -- these "
                "keys only exist on the full-covariance path."
            )

    def _nll_well_expert(self, expert_mus, expert_tr_sigma, z_soft, target, mask):
        """Analytic expected well likelihood (expert level)::

            E_q[-log p(y|theta)] = 1/(2 sigma_y^2) sum_k pi_k (|y - mu_k|^2 + tr(Sigma_k))
                                   + D/2 log(2 pi sigma_y^2)

        The expert-level quantities are required: ``final_pred`` would lose how the variance
        is distributed across experts.
        """
        diff = (target.unsqueeze(1) - expert_mus).clamp(-self.clip_diff, self.clip_diff)
        quad = (diff ** 2).sum(dim=2)                       # (B,K,*spatial) |y - mu_k|^2
        weighted = (z_soft * quad + z_soft * expert_tr_sigma).sum(dim=1)   # (B,*spatial)
        d_attr = expert_mus.shape[2]
        nll = weighted / (2.0 * self.well_var) + 0.5 * d_attr * self.log_two_pi \
            + 0.5 * d_attr * torch.log(self.well_var)
        # the property axis is already summed out => reduce the mask per pixel
        return self._reduce_pixel_nll(nll, mask, where="well_nll_mode='expert'")

    def _nll_well_mixture_predictive(self, expert_mus, expert_off, expert_logvars,
                                     z_soft, target, mask):
        """Mixture predictive density (ablation mode)::

            -log sum_k pi_k N(y; mu_k, sigma_y^2 I + Sigma_k)

        Unlike ``"expert"`` the whole ``Sigma_k`` enters, so the off-diagonals are
        constrained rather than only ``tr(Sigma_k)``.
        """
        L = build_chol(expert_logvars, expert_off)                  # (B,K,3,3,*spatial)
        B, K = L.shape[:2]
        spatial = L.shape[4:]
        eye = torch.eye(3, dtype=L.dtype, device=L.device)
        # Sigma_k + sigma_y^2 I: its Cholesky is not a simple transform of L, so rebuild
        # the covariance per pixel and factorise again.
        S = torch.einsum("bkijs,bkmjs->bkims", L, L)
        S = S + self.well_var * eye.view(1, 1, 3, 3, *([1] * len(spatial)))
        Sf = S.reshape(B * K, 3, 3, -1).permute(0, 3, 1, 2)         # (B*K, P, 3, 3)
        Lc = torch.linalg.cholesky(Sf)
        logdet = 2.0 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)
        r = (target.unsqueeze(1) - expert_mus).clamp(-self.clip_diff, self.clip_diff)
        rf = r.reshape(B * K, 3, -1).permute(0, 2, 1).unsqueeze(-1)  # (B*K, P, 3, 1)
        z = torch.linalg.solve_triangular(Lc, rf, upper=False)
        quad = (z ** 2).sum(dim=(-2, -1)).reshape(B, K, *spatial)
        nll_k = 0.5 * (quad + logdet.reshape(B, K, *spatial) + 3 * self.log_two_pi)
        log_q = torch.log(z_soft.clamp_min(1e-12))
        nll = -torch.logsumexp(log_q - nll_k, dim=1)                # (B,*spatial)
        return self._reduce_pixel_nll(
            nll, mask, where="well_nll_mode='mixture_predictive'")

    def _nll_well_mixture(self, expert_mus, expert_logvars, z_soft, target, mask):
        """Expected well likelihood (exact mixture ELBO mode).

        ``E_q[log p(y|theta)] = sum_k q(z=k|x) log N(y; mu_k, sigma_y^2 + sigma_k^2)
        = logsumexp_k(log q_k + log N_k)``, hence
        ``NLL = -logsumexp_k(log q_k - nll_k)``.

        Per-expert residuals are clipped by ``clip_diff``; the per-expert total variance
        matches the marginal variant.
        """
        var_theta = torch.exp(expert_logvars)                      # (B,K,3,...)
        total_var = var_theta + self.well_var
        diff = (target.unsqueeze(1) - expert_mus).clamp(-self.clip_diff, self.clip_diff)
        nll_k = 0.5 * (diff**2 / total_var + self.log_two_pi + torch.log(total_var))
        log_q = torch.log(z_soft.clamp_min(1e-12))                 # (B,K,...)
        if log_q.dim() < nll_k.dim():                              # -> (B,K,1,...) broadcast
            log_q = log_q.unsqueeze(2)
        nll = -torch.logsumexp(log_q - nll_k, dim=1)               # (B,3,...)
        return self._reduce_well_nll(nll, mask)

    def _nll_seismic(self, syn, obs, mask_weight=None):
        """Seismic likelihood: NLL of p(d | theta_sample) under the configured noise model.

        ``mask_weight=None`` (default) returns ``nll.mean()`` and stays bit-wise identical to
        the released path; when given, the residual is reduced as ``sum(w * nll) / sum(w)``
        over the reliable region only.
        """
        diff = (syn - obs).clamp(-self.clip_diff, self.clip_diff)
        if self.seismic_noise != "iid":
            nll = self._nll_seis_corr(diff)
        else:
            nll = 0.5 * (diff**2 / self.seismic_var + self.log_two_pi + torch.log(self.seismic_var))
        nll = torch.nan_to_num(nll, nan=0.0, posinf=1e6, neginf=-1e6)
        if mask_weight is None:
            return nll.mean()
        return masked_mean(nll, mask_weight)

    # ---- seismic noise model: coherent (trace common mode) + white (trace difference) ----
    @property
    def seismic_coh_var(self) -> torch.Tensor:
        """Common-mode variance sigma_c^2 (a fixed, very large value for ``deviatoric``)."""
        if self.seismic_noise == "deviatoric":
            return self._seis_coh_var_fixed
        return torch.exp(self.log_seismic_coh_var)

    @property
    def seismic_white_var(self) -> torch.Tensor:
        """White (trace-to-trace) variance sigma_w^2."""
        return torch.exp(self.log_seismic_white_var)

    def _nll_seis_corr(self, diff: torch.Tensor) -> torch.Tensor:
        """Gaussian NLL with a covariance correlated along the last (trace) axis.

        With ``C = sigma_c^2 11^T + sigma_w^2 I`` the residual splits into a common mode and
        a deviatoric part (``r = c 1 + w``, ``w`` orthogonal to ``1``), which decouple::

            log|C|      = (W-1) log sigma_w^2 + log(sigma_w^2 + W sigma_c^2)
            r^T C^-1 r  = c^2 W / (W sigma_c^2 + sigma_w^2) + ||w||^2 / sigma_w^2

        Dividing by ``W`` keeps the scale of the element-wise iid NLL, so the seismic weight
        carries over unchanged. ``sigma_c^2 -> 0`` degenerates to iid and
        ``sigma_c^2 -> inf`` keeps only the trace-to-trace differences (``deviatoric``).
        """
        W = diff.shape[-1]
        if getattr(self, "loss_dim", None) == 1:
            raise NotImplementedError(
                "seismic_noise=coherent/deviatoric is 2-D only (the last axis is the trace "
                "axis); in 1-D the last axis is depth and the common mode would be read "
                "along a single trace."
            )
        sc2 = self.seismic_coh_var
        sw2 = self.seismic_white_var
        c = diff.mean(dim=-1, keepdim=True)
        w = diff - c
        quad_c = c ** 2 * W / (W * sc2 + sw2)
        quad_w = (w ** 2).sum(dim=-1, keepdim=True) / sw2
        logdet = (W - 1) * torch.log(sw2 + 1e-12) + torch.log(sw2 + W * sc2 + 1e-12)
        return 0.5 * (quad_c + quad_w + logdet + W * self.log_two_pi) / W

    # ---- weights adjusted at run time (two-phase training) ----
    def set_kl_weights(self, kl_z_weight: float, kl_theta_weight: float) -> None:
        """Update the KL weights (2-D phase 1: heavy KL, phase 2: light KL)."""
        self.kl_z_weight = float(kl_z_weight)
        self.kl_theta_weight = float(kl_theta_weight)

    def kl_objective_contract(self) -> dict:
        """Whether the current KL weights correspond to a strictly writable objective.

        Returned for ``theory_contract.json`` so that "the paper says tempered / the code
        actually computes" stays traceable. Three cases:

        * ``elbo`` -- every KL weight (including the per-expert ``lambda_k``) equals 1;
        * ``single_temperature`` -- every weight equals the same scalar ``c`` (a valid
          tempered objective);
        * ``legacy_weighted`` -- the weights differ, so this is the KL of no single
          posterior; the text must not call it an ELBO/tempered posterior.
        """
        weights = {"kl_z_weight": float(self.kl_z_weight),
                   "kl_theta_weight": float(self.kl_theta_weight)}
        lambdas = None
        if self.expert_lambdas is not None:
            lambdas = [float(v) for v in self.expert_lambdas.tolist()]
            weights["expert_lambdas"] = lambdas
        flat = [weights["kl_z_weight"], weights["kl_theta_weight"]]
        if lambdas:
            flat.extend(lambdas)
        is_elbo = all(abs(w - 1.0) <= 1e-12 for w in flat)
        is_single = all(abs(w - flat[0]) <= 1e-12 for w in flat)
        if is_elbo:
            mode = "elbo"
        elif is_single:
            mode = "single_temperature"
        else:
            mode = "legacy_weighted"
        return {
            "mode": mode,
            "weights": weights,
            "strict_posterior": mode in ("elbo", "single_temperature"),
            "temperature": (1.0 if mode == "elbo" else flat[0]) if is_single else None,
            "note": (
                "valid tempered objective c * KL(q || P_c)" if mode == "single_temperature"
                else ("a true ELBO" if mode == "elbo"
                      else "weights differ => not the KL of any single posterior; the text "
                           "must not call it an ELBO/tempered posterior")
            ),
        }

    def forward(self, model_output, target, syn_norm=None, seismic=None,
                mask=None, target_mask=None, seismic_mask_weight=None):
        """Compute the negative ELBO.

        Args:
            model_output: the dictionary returned by ``GeoVBMILE.forward``.
            target: ``(B, 3, ...)`` normalized elastic-parameter labels.
            syn_norm: ``(B, n_angles, ...)`` forward modelling of ``theta_sample``, in the
                normalized seismic domain. ``None`` skips the seismic term.
            seismic: ``(B, n_angles, ...)`` observed gathers (normalized).
            mask / target_mask: well-label mask.
            seismic_mask_weight: per-angle reliability weights (opt-in, default ``None`` =
                off), shaped like ``syn_norm`` and computed by the trainer from the
                *physical* theta values (:func:`src.forward.reliability.critical_angle_weights`).
                ``None`` bypasses the mechanism entirely and keeps the released numbers
                bit-wise identical; when the criterion was built with
                ``seismic_mask_mode="critical_angle"`` but the weight is missing, a
                ``ValueError`` is raised instead of silently degrading to no masking.

        Returns:
            ``(neg_elbo, loss_dict)``
        """
        mask = target_mask if target_mask is not None else mask
        mu = model_output["final_pred"][:, :3]
        logvar = model_output["final_pred"][:, 3:]

        # Four well-likelihood modes; the default "marginal" matches the 2-D recipe, the
        # expert modes need the full-covariance keys.
        if self.well_nll_mode == "expert":
            self._require_expert_keys(model_output, ("expert_tr_sigma",))
            nll_well = self._nll_well_expert(
                model_output["expert_mus"], model_output["expert_tr_sigma"],
                model_output["z_soft"], target, mask)
        elif self.well_nll_mode == "mixture_predictive":
            self._require_expert_keys(model_output, ("expert_off",))
            nll_well = self._nll_well_mixture_predictive(
                model_output["expert_mus"], model_output["expert_off"],
                model_output["expert_logvars"], model_output["z_soft"], target, mask)
        elif self.well_nll_mode == "mixture" and "expert_mus" in model_output:
            nll_well = self._nll_well_mixture(
                model_output["expert_mus"], model_output["expert_logvars"],
                model_output["z_soft"], target, mask)
        else:
            nll_well = self._nll_well(mu, logvar, target, mask)

        seismic_nll = torch.tensor(0.0, device=mu.device)
        # Mask occupancy, filled only when a weight is actually passed; with
        # seismic_mask_mode="none" neither key appears in ``loss_dict``.
        mask_occupancy = None
        mask_occupancy_energy = None
        if syn_norm is not None and seismic is not None:
            if seismic_mask_weight is None:
                # default path: identical to the released implementation, line for line
                if self.seismic_mask_mode != "none":
                    # a configured switch without the weight means the declared convention
                    # and what is computed have diverged => fail loudly instead of silently
                    # running without masking.
                    raise ValueError(
                        f"seismic_mask_mode={self.seismic_mask_mode!r} requires the caller to "
                        "pass seismic_mask_weight (VBEMLETrainer computes it from the physical "
                        "theta values); got None => silent fallback to no masking, which "
                        "contradicts the configuration.")
                seismic_nll = self.seismic_weight * self._nll_seismic(syn_norm, seismic)
            else:
                if tuple(seismic_mask_weight.shape) != tuple(syn_norm.shape):
                    raise ValueError(
                        "seismic_mask_weight shape does not match syn_norm: "
                        f"{tuple(seismic_mask_weight.shape)} vs {tuple(syn_norm.shape)}")
                seismic_nll = self.seismic_weight * self._nll_seismic(
                    syn_norm, seismic, seismic_mask_weight)
                # occupancy convention: mean over **all** (B, n_angles, *spatial) weight
                # elements, i.e. ``1 - mean(w)`` (not only the elements driven to 0).
                mask_occupancy, mask_occupancy_energy = mask_stats(
                    seismic_mask_weight, syn_norm - seismic)

        kl_z = model_output["kl_z"]

        # KL(theta): per-expert weighted when expert_lambdas is given, otherwise the
        # scalar (1-D) or the z_soft-weighted per-expert mean (2-D).
        if self.expert_lambdas is not None and "kl_theta_per_expert" in model_output:
            kl_theta_per_expert = model_output["kl_theta_per_expert"]
            z_soft = model_output["z_soft"]
            lambdas = self.expert_lambdas.to(z_soft.device).view(
                1, -1, *([1] * (z_soft.dim() - 2))
            )
            kl_theta = (lambdas * z_soft * kl_theta_per_expert).sum(dim=1).mean()
        elif "kl_theta" in model_output:
            kl_theta = model_output["kl_theta"]
        else:
            kl_theta_per_expert = model_output["kl_theta_per_expert"]
            z_soft = model_output["z_soft"]
            kl_theta = (z_soft * kl_theta_per_expert).sum(dim=1).mean()

        neg_elbo = nll_well + seismic_nll + self.kl_z_weight * kl_z + self.kl_theta_weight * kl_theta

        # monitoring MSE at the labelled positions
        if mask is not None:
            masked_diff = (mu - target) * mask
            mse_monitor = (masked_diff**2).sum() / (mask.sum() + 1e-8)
        else:
            mse_monitor = torch.tensor(0.0, device=mu.device)

        loss_dict = {
            "total": neg_elbo,
            "nll_well": nll_well,
            "seismic_nll": seismic_nll,
            "kl_z": kl_z,
            "kl_theta": kl_theta,
            "well_var": self.well_var.detach(),
            "seismic_var": self.seismic_var.detach(),
            "mse_monitor": mse_monitor,
        }
        # seismic noise-model variances: present only for coherent/deviatoric, so the
        # default ("iid") loss_dict keeps the released key set
        if self.seismic_noise != "iid":
            loss_dict["seismic_coh_var"] = self.seismic_coh_var.detach()
            loss_dict["seismic_white_var"] = self.seismic_white_var.detach()
        # reliability masking: present only when a weight was passed, so the default
        # ("none") key set is unchanged
        if mask_occupancy is not None:
            loss_dict["seismic_mask_occupancy"] = mask_occupancy
            if mask_occupancy_energy is not None:
                loss_dict["seismic_mask_occupancy_energy"] = mask_occupancy_energy
        return neg_elbo, loss_dict


def _vbelbo_factory(dim: int):
    """Register ``vbelbo_1d`` / ``vbelbo_2d`` with their respective legacy defaults."""

    def factory(**kwargs):
        kwargs.pop("dim", None)
        if dim == 1:
            defaults = dict(
                seismic_obs_noise_init=5.0,
                well_obs_noise_init=1.0,
                kl_z_weight=0.05,
                kl_theta_weight=0.01,
            )
        else:
            defaults = dict(
                seismic_obs_noise_init=1.0,
                well_obs_noise_init=0.1,
                kl_z_weight=1.0,
                kl_theta_weight=1.0,
            )
        defaults.update(kwargs)
        obj = VBELBOCore(**defaults)
        # lets _nll_seis_corr() verify that the last axis is really the trace axis
        # (a plain attribute, so it never enters the state dict)
        obj.loss_dim = dim
        return obj

    factory.__name__ = f"build_vbelbo_{dim}d"
    return factory


LOSSES.register_obj("vbelbo_1d", _vbelbo_factory(1))
LOSSES.register_obj("vbelbo_2d", _vbelbo_factory(2))
