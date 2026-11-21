"""``GeoVBMILE``: the single ``dim``-parameterised model and its registry factories.

``_forward_1d`` and ``_forward_2d`` share one output contract: ``final_pred``,
``theta_sample``, ``z_soft``, ``kl_z``, the expert keys and - for the full-covariance
family - the off-diagonal outputs and ``tr(Sigma_k)``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import MODELS
from src.models.vbmile._blocks1d import UNetEncoder1D
from src.models.vbmile._decoder1d import SharedExpertDecoder1D
from src.models.vbmile._encoder2d import UNetEncoder2D
from src.models.vbmile._film_decoder2d import ExpertDecoder2D
from src.models.vbmile._padding import _install_manual_reflect
from src.models.vbmile._partition1d import VariationalPartitionLayer1D
from src.models.vbmile._partition2d import VariationalPartitionLayer2D
from src.models.vbmile._proto_helpers import _cov_diagonal, _tempered_gate
from src.utils.fullcov import (
    build_chol,
    chol_apply,
    chol_columns,
    chol_diag,
    chol_trace_term,
    qform,
)


class GeoVBMILE(nn.Module):
    """Variational Bayesian mixture-of-experts network for elastic-parameter inversion.

    ``dim=1`` — the single-trace variant (shared decoder + per-expert adapters); its
    attribute names correspond one-to-one with the keys of the released 1-D state dict.
    ``dim=2`` — the section variant (fully independent experts + spatial FiLM), which
    supports two-phase training (``set_phase(1) / set_phase(2)``).

    Args:
        seismic_channels: number of observed angle channels (default 6).
        low_freq_channels: number of low-frequency model channels (default 3: Vp/Vs/Density).
        base_channels: base width of the encoder.
        num_experts: number of experts K.
        prior_means / prior_vars: the parameter prior. It is registered as buffers and
            initializes the output bias. ``prior_vars`` may be diagonal ``(K, 3)`` or
            full covariance ``(K, 3, 3)``; ``KL(theta)`` branches on ``ndim`` inside
            ``forward``.
        encoder_downsample: 2-D encoder downsampling mode.
            ``"iso"`` (default) downsamples both axes by 2, which is bit-wise identical
            to the released checkpoint; ``"aniso"`` keeps any axis that would fall below
            ``min_down_size`` at its original width, which makes narrow trace windows
            (e.g. width 5) workable — with ``iso`` the smallest feasible width is 13.
        min_down_size: smallest axis size allowed to be stride-2 downsampled in
            ``aniso`` mode (default 3).
        z_temperature: when ``True`` the temperature also scales the gating soft weights
            (``z_soft = softmax(logits / tau)``) and ``kl_z`` is recomputed from the same
            tempered probabilities. Default ``False`` keeps the plain
            ``softmax(logits)``. The 2-D partition layer already uses ``tau=0.7``, so the
            effective 2-D temperature is ``0.7 * tau``; keep the final value >= 0.1 so the
            gating gradient region does not get too narrow.
        decoder_share: whether the 1-D expert decoder shares its convolutions.
            ``"shared"`` (default) is the released layout; ``"independent"`` gives every
            expert its own ``conv2`` / ``conv1`` / ``global_residual`` (see
            ``SharedExpertDecoder1D``). ``"independent"`` is rejected for ``dim=2``
            because the 2-D decoders are always independent.
        prior_cov_shrinkage: shrinkage of a full-covariance prior toward a spherical
            target, ``Sigma <- (1-d) * Sigma + d * (tr(Sigma)/d) * I``. Default ``0.0``
            keeps the prior untouched.
        expert_dilations: 2-D only. Optional per-expert dilation of the fusion block,
            e.g. ``[7, 15, 7]``. Default ``None`` = 7 for every expert. A dilation does
            not change the parameter count or the state-dict keys.
        variational_covariance: variational family of every expert posterior.
            ``"diag"`` (default) keeps ``q(theta|z=k) = N(mu_k, diag(sigma_k^2))``; the
            released 1-D checkpoint has a 6-channel output head, so ``"diag"`` is the
            state-dict-compatible layout. ``"full"`` adds three off-diagonal channels
            (``b1 -> L[1,0]``, ``b2 -> L[2,0]``, ``b3 -> L[2,1]``) and models
            ``Sigma_k = L_k L_k^T``; the off-diagonal head is zero-initialised, so a fresh
            ``"full"`` model is bit-wise identical to ``"diag"`` at step 0. The delivery
            1-D/2-D recipes use ``"full"``.
        ``expert_dilations`` and ``variational_covariance`` are the only structural degrees
        of freedom; everything else in the 2-D decoder is fixed (merged 6-channel output
        head, ``skip2`` convolved at its native resolution, same-size tail, copy-kernel
        reflect padding).

    The 2-D-only options raise if they are passed with ``dim=1`` instead of being
    silently inert.
    """

    def __init__(self,
                 dim: int = 1,
                 seismic_channels: int = 6,
                 low_freq_channels: int = 3,
                 base_channels: int = 16,
                 norm_type: str = "group",
                 num_experts: int = 3,
                 initial_temperature: float = 1.0,
                 final_temperature: float = 0.1,
                 prior_means: Optional[torch.Tensor] = None,
                 prior_vars: Optional[torch.Tensor] = None,
                 film_hidden_dim: int = 8,
                 film_kernel_size: int = 3,
                 modulation_scale: float = 1.0,
                 encoder_downsample: str = "iso",
                 min_down_size: int = 3,
                 z_temperature: bool = False,
                 decoder_share: str = "shared",
                 prior_cov_shrinkage: float = 0.0,
                 expert_dilations: Optional[List[int]] = None,
                 variational_covariance: str = "diag"):
        super().__init__()
        if dim not in (1, 2):
            raise ValueError(f"dim must be 1 or 2, got {dim!r}")
        if variational_covariance not in ("diag", "full"):
            raise ValueError(
                f"variational_covariance must be diag/full, got {variational_covariance!r}")
        if dim == 1 and expert_dilations is not None:
            raise ValueError("expert_dilations only applies to dim=2")
        if encoder_downsample not in ("iso", "aniso"):
            raise ValueError(f"encoder_downsample must be iso/aniso, got {encoder_downsample!r}")
        if dim == 1 and encoder_downsample != "iso":
            raise ValueError("encoder_downsample=aniso is 2-D only (the 1-D encoder has no trace axis)")
        if decoder_share not in ("shared", "independent"):
            raise ValueError(f"decoder_share must be shared/independent, got {decoder_share!r}")
        if dim == 2 and decoder_share != "shared":
            raise ValueError("decoder_share=independent is 1-D only (2-D experts are already independent)")
        if not 0.0 <= float(prior_cov_shrinkage) < 1.0:
            raise ValueError(
                f"prior_cov_shrinkage must be in [0, 1), got {prior_cov_shrinkage!r}")
        # plain attribute (not a buffer): never enters the state dict
        self.z_temperature = bool(z_temperature)
        self.decoder_share = str(decoder_share)
        self.prior_cov_shrinkage = float(prior_cov_shrinkage)
        self.variational_covariance = str(variational_covariance)
        self.dim = int(dim)
        self.num_experts = num_experts
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.low_freq_channels = low_freq_channels
        self.register_buffer("current_temperature", torch.tensor(initial_temperature))

        # prior (registered as buffers; drives KL(theta) and the output-bias init)
        # prior_vars is either diagonal (K, 3) or full covariance (K, 3, 3);
        # KL(theta) branches on the rank inside the forward.
        if prior_means is not None:
            assert prior_vars is not None
            prior_means = torch.as_tensor(prior_means, dtype=torch.float32)
            prior_vars = torch.as_tensor(prior_vars, dtype=torch.float32)
            # covariance shrinkage (opt-in, 0.0 = bit-wise unchanged):
            # a GMM full-covariance prior in normalized space is near singular (cond up to 5804,
            # smallest eigenvalue 3e-4), so the worst-direction gain of the inverse reaches 3220
            # and KL is inflated by two orders of magnitude. Shrinking toward the
            # dimension-averaged target is the standard remedy: Sigma <- (1-d)Sigma + d (tr(Σ)/d) I.
            if self.prior_cov_shrinkage > 0.0 and prior_vars.ndim == 3:
                d_attr = prior_vars.shape[-1]
                eye = torch.eye(d_attr, dtype=prior_vars.dtype)
                scale = prior_vars.diagonal(dim1=-2, dim2=-1).mean(dim=-1)   # (K,) = tr/d
                prior_vars = ((1.0 - self.prior_cov_shrinkage) * prior_vars
                              + self.prior_cov_shrinkage * scale[:, None, None] * eye)
            self.register_buffer("prior_means", prior_means)          # (K, 3)
            self.register_buffer("prior_vars", prior_vars)            # (K, 3) or (K, 3, 3)
            prior_logvars = torch.log(
                torch.clamp(_cov_diagonal(prior_vars), min=1e-12)     # (K, 3)
            )
        else:
            self.register_buffer("prior_means", torch.zeros(num_experts, 3))
            self.register_buffer("prior_vars", torch.ones(num_experts, 3))
            prior_logvars = torch.zeros(num_experts, 3)

        # encoder (seismic + low-frequency model)
        encoder_in = seismic_channels + low_freq_channels
        if self.dim == 1:
            self.encoder = UNetEncoder1D(encoder_in, base_channels, norm_type)
            self.skip_channels = [base_channels * 2, base_channels * 4,
                                  base_channels * 8, base_channels]
        else:
            self.encoder = UNetEncoder2D(encoder_in, base_channels, norm_type,
                                         downsample_mode=encoder_downsample,
                                         min_down_size=min_down_size)
            self.skip_channels = [base_channels * 2, base_channels * 4, base_channels]

        # partition layer (2-D also concatenates the global feature)
        if self.dim == 1:
            self.partition_layer = VariationalPartitionLayer1D(
                low_freq_channels, num_experts, norm_type=norm_type
            )
        else:
            partition_in = low_freq_channels + base_channels
            self.partition_layer = VariationalPartitionLayer2D(
                partition_in, num_experts, norm_type,
                aniso=(encoder_downsample == "aniso"),
            )

        # expert decoders
        if self.dim == 1:
            self.experts = SharedExpertDecoder1D(
                skip_channels=self.skip_channels,
                num_experts=num_experts,
                norm_type=norm_type,
                prior_means=prior_means,
                prior_logvars=prior_logvars,
                share=self.decoder_share,
                full_covariance=(self.variational_covariance == "full"),
            )
        else:
            self.experts = ExpertDecoder2D(
                self.skip_channels, num_experts, norm_type,
                prior_means=prior_means, prior_logvars=prior_logvars,
                film_hidden_dim=film_hidden_dim, film_kernel_size=film_kernel_size,
                modulation_scale=modulation_scale,
                aniso=(encoder_downsample == "aniso"),
                expert_dilations=expert_dilations,
                full_covariance=(self.variational_covariance == "full"),
            )

        # deterministic protocol: every reflect padding becomes a copy kernel (same numbers as F.pad).
        # 1-D has no reflect convolution, so the count is 0.
        self.n_manual_reflect_pads = _install_manual_reflect(self)

    # ---- training phases and temperature ----
    def set_phase(self, phase: int) -> None:
        """Set the training phase.

        phase 1 — joint training (everything trainable);
        phase 2 — freeze the gating, train encoder + decoders;
        phase 3 — joint fine-tuning (everything trainable, gating not frozen).
        """
        if phase in (1, 3):
            self.train()
            for p in self.parameters():
                p.requires_grad = True
        elif phase == 2:
            self.encoder.train()
            self.experts.train()
            self.partition_layer.eval()
            for p in self.encoder.parameters():
                p.requires_grad = True
            for p in self.partition_layer.parameters():
                p.requires_grad = False
            for p in self.experts.parameters():
                p.requires_grad = True
        else:
            raise ValueError("phase must be 1, 2 or 3")

    def set_temperature(self, epoch: int, total_epochs: int) -> None:
        """Cosine-anneal the Gumbel-Softmax temperature."""
        if total_epochs <= 0:
            return
        cos_val = (1 + math.cos(math.pi * epoch / total_epochs)) / 2
        new_temp = self.final_temperature + (self.initial_temperature - self.final_temperature) * cos_val
        self.current_temperature.fill_(new_temp)

    def _prior_precision_and_logdet(self):
        """Prior precision ``Sigma_p^-1`` and ``ln|Sigma_p|``, shapes ``(K,3,3)`` / ``(K,)``.

        Used by the full-covariance path only. The prior is a constant, so the result is
        cached against the ``prior_vars`` buffer version to avoid inverting it every step.
        """
        key = (id(self.prior_vars), int(getattr(self.prior_vars, "_version", 0)))
        cached = getattr(self, "_prior_prec_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        if self.prior_vars.ndim == 3:
            # Same operators as the diagonal branch (inv + slogdet) so that
            # "full with b=0 == diag" can be verified at floating-point level. A 1e-6
            # jitter is added only if the prior is numerically singular -- never silently.
            cov = self.prior_vars
            prec = torch.linalg.inv(cov)
            _, logdet_p = torch.linalg.slogdet(cov)
            if not torch.isfinite(prec).all() or not torch.isfinite(logdet_p).all():
                eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
                cov = cov + 1e-6 * eye
                prec = torch.linalg.inv(cov)
                _, logdet_p = torch.linalg.slogdet(cov)
        else:
            var = self.prior_vars
            prec = torch.diag_embed(1.0 / var.clamp_min(1e-12))          # (K,3,3)
            logdet_p = torch.log(var.clamp_min(1e-12)).sum(dim=-1)        # (K,)
        self._prior_prec_cache = (key, prec, logdet_p)
        return prec, logdet_p

    # ---- 1-D compatibility helpers ----
    def encode(self, seismic, low_freq):
        """Encoder forward: concatenate seismic and the low-frequency model (1-D interface)."""
        encoder_input = torch.cat([seismic, low_freq], dim=1)
        return self.encoder(encoder_input)

    def sample_theta(self, expert_mus, expert_logvars, logits, tau, hard=True,
                     expert_off=None):
        """Draw one sample from ``q(theta | x)`` (Gumbel-Softmax + reparameterisation).

        With ``expert_off is None`` (diagonal family)::

            θ̃ = μ + σ ⊙ ε

        With ``expert_off`` given (full-covariance family)::

            theta = mu + L eps, with L lower-triangular, diagonal exp(logvar / 2) and the

        off-diagonals from ``expert_off``. This is the only formula that shapes the gradient
        path: keeping ``sigma * eps`` would leave the MC estimate diagonal even with a 9-channel head.
        """
        z_hard = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)
        eps = torch.randn_like(expert_mus)
        if expert_off is None:
            theta_samples = expert_mus + torch.exp(0.5 * expert_logvars) * eps
        else:
            L = build_chol(expert_logvars, expert_off)
            theta_samples = expert_mus + chol_apply(L, eps)
        theta = (z_hard.unsqueeze(2) * theta_samples).sum(dim=1)
        return theta

    # ---- forward ----
    def forward(self, seismic, low_freq_model=None, prior=None, num_samples=1) -> dict:
        """Forward pass. The shape follows ``dim``: (B, C, L) in 1-D, (B, C, H, W) in 2-D.

        Returns:
            The returned dict holds final_pred / theta_sample / expert_mus / expert_logvars /
            z_soft / logits / kl_z plus the intermediate features and either (1-D) kl_theta or
            kl_theta_per_expert。
        """
        if self.dim == 1:
            return self._forward_1d(seismic, low_freq_model, prior, num_samples)
        return self._forward_2d(seismic, low_freq_model, prior, num_samples)

    def _forward_1d(self, seismic, low_freq_model, prior, num_samples):
        B, _, L = seismic.shape
        if low_freq_model is None:
            low_freq_model = torch.zeros(B, self.low_freq_channels, L, device=seismic.device)

        features = self.encode(seismic, low_freq_model)  # [skip1, skip2, bottom, global]

        logits, kl_z = self.partition_layer(low_freq_model, prior=prior)
        if self.z_temperature:
            # tau acts on the gate probabilities; kl_z is recomputed from the tempered probs
            z_soft, kl_z = _tempered_gate(logits, self.current_temperature, prior)
        else:
            z_soft = F.softmax(logits, dim=1)  # (B, K, L)

        expert_out = self.experts(features, target_size=L, z_soft=z_soft)  # (B,K,6|9,L)
        expert_mus = expert_out[:, :, :3]  # (B, K, 3, L)
        expert_logvars = expert_out[:, :, 3:6]  # (B, K, 3, L)
        expert_off = (expert_out[:, :, 6:] if self.variational_covariance == "full"
                      else None)  # (B, K, 3, L) or None
        full_cov = expert_off is not None

        # ---- KL(theta) ----
        var = torch.exp(expert_logvars)
        if full_cov:
            # one closed form for full q against a full or diagonal p:
            #   KL = ½[tr(Σp⁻¹Σq) + δᵀΣp⁻¹δ − D + ln|Σp| − ln|Σq|]
            cols = chol_columns(expert_logvars, expert_off)
            prec, logdet_p = self._prior_precision_and_logdet()          # (K,3,3), (K,)
            # ln|Σq| = 2·Σ log diag(L) = Σ logvar
            logdet_q = expert_logvars.sum(dim=2)                         # (B,K,L)
            tr_t = chol_trace_term(cols, prec)                          # (B,K,L)
            delta = expert_mus - self.prior_means[None, :, :, None]      # (B,K,3,L)
            qf = qform(delta, prec)                                     # (B,K,L)
            kl_theta_per_expert = 0.5 * (
                tr_t + qf - expert_mus.shape[2] - logdet_q
                + logdet_p[None, :, None]
            )  # (B, K, L)
            # marginal variance: for full, diag(Sigma_k) != exp(logvar), the b^2 term counts
            diag_var = chol_diag(expert_logvars, expert_off)            # (B,K,3,L)
        elif self.prior_vars.ndim == 3:
            # full-covariance prior p = N(prior_means, prior_vars):
            # KL(q=diag || p) = 1/2 [tr(Sp^-1 V) + d^T Sp^-1 d - D + ln|Sp| - sum_a logvar_a]
            cov_inv = torch.linalg.inv(self.prior_vars)                  # (K, 3, 3)
            _, logdet_p = torch.linalg.slogdet(self.prior_vars)          # (K,)
            d_attr = expert_mus.shape[2]
            tr_t = torch.einsum("bkdl,kd->bkl", var,
                                cov_inv.diagonal(dim1=-2, dim2=-1))      # (B, K, L)
            delta = expert_mus - self.prior_means[None, :, :, None]      # (B, K, 3, L)
            qf = torch.einsum("bkdl,kde,bkel->bkl", delta, cov_inv, delta)
            kl_theta_per_expert = 0.5 * (
                tr_t + qf - d_attr - expert_logvars.sum(dim=2)
                + logdet_p[None, :, None]
            )  # (B, K, L)
            diag_var = var
        else:
            prior_means_exp = self.prior_means[None, :, :, None]   # (1, K, 3, 1)
            prior_vars_exp = self.prior_vars[None, :, :, None]     # (1, K, 3, 1)
            kl_theta_per_expert = 0.5 * (
                (var / prior_vars_exp)
                + ((expert_mus - prior_means_exp) ** 2) / prior_vars_exp
                - 1.0 - expert_logvars + torch.log(prior_vars_exp)
            ).sum(dim=2)  # (B, K, L)
            diag_var = var
        kl_theta = (z_soft * kl_theta_per_expert).sum(dim=1).mean()

        # ---- marginal q(theta | x): the diagonal of the law of total covariance ----
        mu_marginal = (z_soft.unsqueeze(2) * expert_mus).sum(dim=1)  # (B, 3, L)
        second_moment = (z_soft.unsqueeze(2) * (diag_var + expert_mus ** 2)).sum(dim=1)
        var_marginal = second_moment - mu_marginal ** 2
        var_marginal = torch.clamp(var_marginal, min=1e-6)
        logvar_marginal = torch.log(var_marginal)
        final_pred = torch.cat([mu_marginal, logvar_marginal], dim=1)  # (B, 6, L)

        # ---- optional sampling ----
        theta_sample = None
        if num_samples > 0:
            theta_sample = self.sample_theta(
                expert_mus, expert_logvars, logits, self.current_temperature, hard=True,
                expert_off=expert_off,
            )

        out = {
            "final_pred": final_pred,
            "theta_sample": theta_sample,
            "expert_mus": expert_mus,
            "expert_logvars": expert_logvars,
            "z_soft": z_soft,
            "logits": logits,
            "kl_z": kl_z,
            "kl_theta": kl_theta,
            "kl_theta_per_expert": kl_theta_per_expert,
            "skip1": features[0],
            "skip2": features[1],
            "bottleneck": features[2],
            "global": features[3],
        }
        if full_cov:
            # full-only keys: needed by the expert-level and mixture-predictive well likelihoods
            out["expert_off"] = expert_off
            out["expert_diag_var"] = diag_var
            out["expert_tr_sigma"] = diag_var.sum(dim=2)                 # (B,K,L)
        return out

    def _forward_2d(self, seismic, low_freq_model, prior, num_samples):
        B, _, H, W = seismic.shape
        if low_freq_model is None:
            low_freq_model = torch.zeros(B, self.low_freq_channels, H, W, device=seismic.device)

        # encode
        features = self.encoder(torch.cat([seismic, low_freq_model], dim=1))
        skip1, skip2, global_feat = features

        # gate
        logits, kl_z = self.partition_layer(low_freq_model, global_feat, prior=prior)
        if self.z_temperature:
            # as in 1-D; the 2-D partition layer already applies tau=0.7, so the effective temperature is 0.7 * tau
            z_soft, kl_z = _tempered_gate(logits, self.current_temperature, prior)
        else:
            z_soft = F.softmax(logits, dim=1)

        # expert decode
        expert_out = self.experts([skip1, skip2, global_feat], (H, W), z_soft)
        expert_mus = expert_out[:, :, :3]      # (B, K, 3, H, W)
        expert_logvars = expert_out[:, :, 3:6]  # (B, K, 3, H, W)
        expert_off = (expert_out[:, :, 6:] if self.variational_covariance == "full"
                      else None)  # (B, K, 3, H, W) or None
        full_cov = expert_off is not None

        # KL(theta)
        var = torch.exp(expert_logvars)
        if full_cov:
            # full-q closed form (same as 1-D); dividing by D matches the mean(dim=2) scale of the diagonal path.
            cols = chol_columns(expert_logvars, expert_off)
            prec, logdet_p = self._prior_precision_and_logdet()          # (K,3,3), (K,)
            logdet_q = expert_logvars.sum(dim=2)                         # (B,K,H,W)
            tr_t = chol_trace_term(cols, prec)                          # (B,K,H,W)
            delta = expert_mus - self.prior_means[None, :, :, None, None]
            qf = qform(delta, prec)                                     # (B,K,H,W)
            d_attr = expert_mus.shape[2]
            kl_theta_per_expert = 0.5 * (
                tr_t + qf - d_attr - logdet_q + logdet_p[None, :, None, None]
            ) / d_attr  # (B, K, H, W)
            diag_var = chol_diag(expert_logvars, expert_off)            # (B,K,3,H,W)
        elif self.prior_vars.ndim == 3:
            # full-covariance prior (same formula as in _forward_1d); dividing by D matches
            # the mean(dim=2) scale of the 2-D diagonal path.
            cov_inv = torch.linalg.inv(self.prior_vars)                  # (K, 3, 3)
            _, logdet_p = torch.linalg.slogdet(self.prior_vars)          # (K,)
            d_attr = expert_mus.shape[2]
            tr_t = torch.einsum("bkdhw,kd->bkhw", var,
                                cov_inv.diagonal(dim1=-2, dim2=-1))
            delta = expert_mus - self.prior_means[None, :, :, None, None]
            qf = torch.einsum("bkdhw,kde,bkehw->bkhw", delta, cov_inv, delta)
            kl_theta_per_expert = 0.5 * (
                tr_t + qf - d_attr - expert_logvars.sum(dim=2)
                + logdet_p[None, :, None, None]
            ) / d_attr  # (B, K, H, W)
            diag_var = var
        else:
            prior_means_exp = self.prior_means[None, :, :, None, None]
            prior_vars_exp = self.prior_vars[None, :, :, None, None]
            kl_theta_per_expert = 0.5 * (
                var / prior_vars_exp
                + (expert_mus - prior_means_exp).pow(2) / prior_vars_exp
                - 1.0 - expert_logvars + torch.log(prior_vars_exp)
            ).mean(dim=2)  # (B, K, H, W)
            diag_var = var

        # marginal distribution (diag of Sigma_marg; the b^2 term is included for full)
        mu_marginal = (z_soft.unsqueeze(2) * expert_mus).sum(dim=1)  # (B, 3, H, W)
        second_moment = (z_soft.unsqueeze(2) * (diag_var + expert_mus ** 2)).sum(dim=1)
        var_marginal = second_moment - mu_marginal ** 2
        var_marginal = torch.clamp(var_marginal, min=1e-6)
        logvar_marginal = torch.log(var_marginal)
        final_pred = torch.cat([mu_marginal, logvar_marginal], dim=1)  # (B, 6, H, W)

        # sampling (hard Gumbel-Softmax + reparameterisation)
        # order matters: the reparameterisation noise `eps` is drawn before the Gumbel gate.
        # This fixes the order in which the two RNG streams are consumed (1-D uses sample_theta, reversed).
        theta_sample = None
        if num_samples > 0:
            eps = torch.randn_like(expert_mus)
            if full_cov:
                # full: theta = mu + L eps (the only formula that shapes the gradient on this path)
                theta_samples = expert_mus + chol_apply(
                    build_chol(expert_logvars, expert_off), eps)
            else:
                theta_samples = expert_mus + torch.exp(0.5 * expert_logvars) * eps
            z_hard = F.gumbel_softmax(logits, tau=self.current_temperature, hard=True, dim=1)
            theta_sample = (z_hard.unsqueeze(2) * theta_samples).sum(dim=1)  # (B, 3, H, W)

        out = {
            "final_pred": final_pred,
            "theta_sample": theta_sample,
            "expert_mus": expert_mus,
            "expert_logvars": expert_logvars,
            "z_soft": z_soft,
            "logits": logits,
            "kl_z": kl_z,
            "kl_theta_per_expert": kl_theta_per_expert,
            "skip1": skip1,
            "skip2": skip2,
            "global": global_feat,
        }
        if full_cov:
            out["expert_off"] = expert_off
            out["expert_diag_var"] = diag_var
            out["expert_tr_sigma"] = diag_var.sum(dim=2)                 # (B,K,H,W)
        return out


def _vbmile_factory(dim: int):
    """Build the registry factory for ``dim`` (config ``_target_: vbmile_1d / vbmile_2d``)."""

    def factory(**kwargs):
        kwargs.pop("dim", None)
        return GeoVBMILE(dim=dim, **kwargs)

    factory.__name__ = f"build_vbmile_{dim}d"
    return factory


MODELS.register_obj("vbmile_1d", _vbmile_factory(1))
MODELS.register_obj("vbmile_2d", _vbmile_factory(2))
