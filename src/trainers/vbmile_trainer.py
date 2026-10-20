"""Geo-VB-MILE trainer.

Training loop:

1. Adam optimizes the model and the criterion parameters jointly;
2. a ``LinearWarmupCosineAnnealingLR`` schedule (see ``src/utils/lr_scheduler.py``);
3. each epoch anneals the gating temperature first, via
   ``model.set_temperature(epoch-1, epochs)``;
4. inside the data loop a ``theta`` sample is de-normalized to physical space, passed
   through the forward operator, normalized back into the seismic domain and scored by
   the NLL;
5. gradient clipping at ``grad_clip``;
6. validation is the point-estimate MSE under the well mask;
7. the best checkpoint (by validation metric) is written into the run directory.

For ``dim=2`` the ``phases`` option enables a two-phase schedule: phase 1 trains
everything jointly, phase 2 freezes the gating (``model.set_phase``) and switches the
KL weights.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.trainers.base import TRAINERS, BaseTrainer, TrainerConfig
from src.utils.logging_utils import get_logger
from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.torch_utils import get_device

__all__ = ["VBEMLETrainer", "build_vbmile_loader", "build_vbmile_loaders"]

logger = get_logger(__name__)


class _VBMLEBatchDataset(torch.utils.data.Dataset):
    """Adapt the ``InversionSample`` datasets (physical space) into training batches.

    Each sample moves ``seismic`` / ``target`` into the normalized space, keeps
    ``target_low`` normalized, and passes ``mask`` and ``prior`` (possibly ``None``)
    through. Normalization uses the dataset's own ``to_norm`` — the same statistics as
    ``norm_stats.json`` — and everything is emitted as contiguous float32 tensors.

    Args:
        num_experts: number of spatial-prior channels (= the model's K). When it differs
            from the number of experts, the first K channels are kept so the partition
            layer's KL shapes still match.
    """

    def __init__(self, dataset, num_experts: Optional[int] = None) -> None:
        self.ds = dataset
        self.num_experts = int(num_experts) if num_experts else None

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> Dict[str, Optional[torch.Tensor]]:
        s = self.ds[index]
        obs_n = self.ds.to_norm(np.asarray(s.observation, np.float32), which="seismic")
        tgt_n = self.ds.to_norm(np.asarray(s.target, np.float32), which="target")
        prior = s.meta.get("prior")
        if prior is not None and self.num_experts is not None:
            prior = np.asarray(prior)[: self.num_experts]
        return {
            "seismic": torch.from_numpy(np.ascontiguousarray(obs_n)),
            "target": torch.from_numpy(np.ascontiguousarray(tgt_n)),
            "target_low": torch.from_numpy(np.ascontiguousarray(s.meta["target_low"])),
            "mask": torch.from_numpy(np.ascontiguousarray(s.meta["mask"])),
            "prior": None if prior is None
            else torch.from_numpy(np.ascontiguousarray(prior)),
        }


def _collate_vbmile_batch(batch: List[Dict[str, Optional[torch.Tensor]]]) -> Dict[str, Any]:
    """Stack a batch; the whole batch is ``None`` when ``prior`` is ``None``.

    The 1-D prior ``(B,K)`` is broadcast to ``(B,K,L)`` because the model's partition
    layer expects it to extend along depth; the 2-D prior already carries the spatial
    dimensions ``(B,K,H,W)`` and is returned unchanged.
    """
    out: Dict[str, Any] = {}
    for k in ("seismic", "target", "target_low", "mask"):
        out[k] = torch.stack([b[k] for b in batch])
    priors = [b["prior"] for b in batch]
    if any(p is None for p in priors):
        out["prior"] = None
    else:
        prior = torch.stack(priors)  # (B,K) / (B,K,H,W)
        if prior.dim() == 2:
            L = out["seismic"].shape[-1]
            prior = prior.unsqueeze(-1).expand(-1, -1, L).contiguous()
        out["prior"] = prior
    return out


def build_vbmile_loader(dataset, *, batch_size: int = 20, shuffle: bool = False,
                        num_workers: int = 0, num_experts: Optional[int] = None,
                        generator: Optional[torch.Generator] = None,
                        **loader_kwargs) -> DataLoader:
    """Wrap one dataset into a Geo-VB-MILE training-batch loader.

    Args:
        num_experts: channel truncation for the spatial prior (used by the expert-count
            ablation).
        generator: optional explicit random source. Passing one takes both the shuffle
            order and the worker ``base_seed`` from it instead of the global RNG, so
            ``num_workers`` no longer changes the training trajectory.
    """
    return DataLoader(
        _VBMLEBatchDataset(dataset, num_experts=num_experts),
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        collate_fn=_collate_vbmile_batch,
        generator=generator,
        **loader_kwargs,
    )


def build_vbmile_loaders(train_dataset, val_dataset=None, *, batch_size: int = 20,
                         num_workers: int = 0, num_experts: Optional[int] = None,
                         generator: Optional[torch.Generator] = None) -> tuple:
    """Convenience wrapper building ``(train_loader, val_loader)``.

    ``val_dataset`` may be ``None`` (train is reused); ``generator`` drives the training
    loader only (the validation loader never shuffles).
    """
    train_loader = build_vbmile_loader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        num_experts=num_experts, generator=generator,
    )
    val_loader = build_vbmile_loader(
        val_dataset or train_dataset, batch_size=batch_size, shuffle=False,
        num_experts=num_experts,
    )
    return train_loader, val_loader


def _phase_defaults(phase_cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return phase_cfg.get(key, default)


@TRAINERS.register("vbmile")
class VBEMLETrainer(BaseTrainer):
    """Geo-VB-MILE training loop.

    The components are injected after construction by
    ``experiments.main.run_main._assemble_vbmile_trainer``.

    Args:
        run_info: run-directory handle (metrics / checkpoints are written there).
        cfg: ``TrainerConfig`` (epochs, learning rate, ...).
        model: the ``GeoVBMILE`` network.
        criterion: the ``VBELBOCore`` loss.
        forward: ``SeismicForward`` (physical space, ``theta`` sample -> synthetic
            gathers).
        train_loader / val_loader: data loaders whose batches are dictionaries with the
            keys ``seismic / target / target_low / mask / prior``.
        stats: normalization statistics (``target_mean / target_std / seismic_mean /
            seismic_std`` tensors).
        phases: optional two-phase schedule, shared by 1-D and 2-D, of the form
            ``[{"epochs": 100, "lr": 1e-2, "kl_z_weight": 1.0, "kl_theta_weight": 0.1,
            "freeze_gating": false}, {...}]``. ``None`` means a single phase.
            A phase may override the scheduler shape with ``sched_epochs /
            warmup_epochs / warmup_start_lr``; when omitted the global values are used.
            Monte-Carlo sampling is always a single sample (``num_samples=1``).
    """

    def __init__(self, run_info, cfg: Optional[TrainerConfig] = None, *,
                 model: Optional[nn.Module] = None,
                 criterion: Optional[nn.Module] = None,
                 forward: Optional[nn.Module] = None,
                 train_loader=None, val_loader=None,
                 device=None, stats: Optional[Dict[str, torch.Tensor]] = None,
                 phases: Optional[List[Dict[str, Any]]] = None):
        super().__init__(run_info, cfg)
        self.model = model
        self.criterion = criterion
        self.forward = forward
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = get_device() if device is None else torch.device(device)
        self.stats = stats or {}
        self.phases = phases
        # full top-level config (set by the entry point); ``checkpoint.save_last`` and
        # other top-level keys are read through it
        self.full_config: Dict[str, Any] = {}

        # training hyper-parameters (``cfg.extra`` absorbs keys outside TrainerConfig)
        extra = self.cfg.extra
        self.lr = float(extra.get("lr", 1e-3))
        self.weight_decay = float(extra.get("weight_decay", 1e-2))
        self.warmup_epochs = int(extra.get("warmup_epochs", 20))
        self.warmup_start_lr = float(extra.get("warmup_start_lr", 1e-6))
        self.lr_min = float(extra.get("lr_min", 1e-8))
        self.grad_clip = extra.get("grad_clip", 1.0)
        self.max_epochs = int(extra.get("max_epochs", self.cfg.max_epochs))
        self.batch_size = int(extra.get("batch_size", 20))
        self.shuffle_train = bool(extra.get("shuffle_train", True))
        self.temperature_fixed = bool(extra.get("temperature_fixed", False))
        # optional starting point: ``trainer.init_weights=<path>`` pointing at a
        # checkpoint with a "model" state_dict (or a bare state_dict), loaded at the
        # beginning of ``run()``.
        self.init_weights = extra.get("init_weights")
        # separate learning rate for the gating network (partition layer). ``None``
        # shares the base learning rate with everything else (default).
        glr = extra.get("gating_lr")
        self.gating_lr: Optional[float] = float(glr) if glr is not None else None
        # the two-phase schedule can be passed as a constructor argument or live in
        # ``trainer.<extra>`` (dict keys are forwarded to ``extra``)
        self.phases = phases if phases is not None else extra.get("phases")

        self.dim = int(getattr(self.model, "dim", 1)) if self.model is not None else 1

    # ---- main loop ----
    def run(self) -> Dict[str, Any]:
        if self.model is None or self.criterion is None:
            raise RuntimeError("VBEMLETrainer needs a model and a criterion "
                               "(injected by `_target_: vbmile`)")
        # the model is injected after construction, so recompute dim for mask broadcasting
        self.dim = int(getattr(self.model, "dim", 1))
        device = self.device
        model = self.model.to(device)
        criterion = self.criterion.to(device)
        if self.init_weights:
            ck = torch.load(self.init_weights, map_location=device, weights_only=True)
            state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            missing, unexpected = model.load_state_dict(state, strict=True)
            self.logger.info("init_weights: loaded %s (missing=%d, unexpected=%d)",
                             self.init_weights, len(missing), len(unexpected))
        if self.forward is not None:
            self.forward.to(device)

        stats = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in self.stats.items()
        }

        phase_boundaries = self._phase_boundaries()
        # the first phase's learning rate must be used when the optimizer is built:
        # ``phases[0].lr`` is otherwise only read at a phase *boundary* (``b + 1``), and
        # epoch 1 is not one, so a configured stage-1 lr would silently never apply.
        init_lr = self._phase_lr(0) if self.phases else self.lr
        optimizer, scheduler = self._build_optimizer(model, criterion, lr=init_lr)
        self._apply_phase(model, criterion, phase_index=0)

        best_val = float("inf")
        best_epoch = 0
        final_metrics: Dict[str, Any] = {}

        for epoch in range(1, self.max_epochs + 1):
            phase_idx = self._phase_index(epoch)
            if phase_idx is not None and self._phase_boundary(epoch):
                self._apply_phase(model, criterion, phase_idx)
                # a phase may override the scheduler shape (e.g. cosine within the phase,
                # or warmup=0 for an almost constant learning rate); omitted keys keep the
                # global values.
                ph_cfg = self.phases[phase_idx]
                ph_lr = self._phase_lr(phase_idx)
                ph_wu = ph_cfg.get("warmup_epochs")
                ph_ws = ph_cfg.get("warmup_start_lr")
                if ph_ws is None and ph_wu is not None and int(ph_wu) == 0:
                    ph_ws = ph_lr      # warmup=0: the phase peak lr applies from the first step
                optimizer, scheduler = self._build_optimizer(
                    model, criterion, lr=ph_lr,
                    warmup_epochs=ph_wu, sched_epochs=ph_cfg.get("sched_epochs"),
                    warmup_start_lr=ph_ws,
                )
                logger.info("entering phase %d (epoch %d)", phase_idx + 1, epoch)

            if hasattr(model, "set_temperature") and not self.temperature_fixed:
                model.set_temperature(epoch - 1, self.max_epochs)

            train_loss, comps = self._train_one_epoch(model, criterion, optimizer, epoch, stats)
            val_mse = self._validate(model, device)
            scheduler.step()

            # when the loss carries an explicit seismic noise model, also print its
            # variances (default iid keeps the plain released log line)
            if getattr(criterion, "seismic_noise", "iid") != "iid":
                logger.info(
                    "Epoch %03d/%d | Train: %.6f | Val MSE: %.6f | kl_z=%.4f kl_theta=%.4f | "
                    "well_var=%.4f seismic_var=%.4f nll_well=%.4f seismic_nll=%.4f "
                    "coh_var=%.3f white_var=%.3f",
                    epoch, self.max_epochs, train_loss, val_mse,
                    comps.get("kl_z", 0.0), comps.get("kl_theta", 0.0),
                    comps.get("well_var", 0.0), comps.get("seismic_var", 0.0),
                    comps.get("nll_well", 0.0), comps.get("seismic_nll", 0.0),
                    comps.get("seismic_coh_var", float("nan")),
                    comps.get("seismic_white_var", float("nan")),
                )
            else:
                logger.info(
                    "Epoch %03d/%d | Train: %.6f | Val MSE: %.6f | kl_z=%.4f kl_theta=%.4f | "
                    "well_var=%.4f seismic_var=%.4f",
                    epoch, self.max_epochs, train_loss, val_mse,
                    comps.get("kl_z", 0.0), comps.get("kl_theta", 0.0),
                    comps.get("well_var", 0.0), comps.get("seismic_var", 0.0),
                )

            if self.record_best("val_mse", val_mse, epoch, higher_is_better=False):
                best_val, best_epoch = val_mse, epoch
                ckpt_path = self.run_info.checkpoints_dir / "best_model.pth"
                torch.save(
                    {"model": model.state_dict(),
                     "criterion": criterion.state_dict(),
                     "epoch": epoch,
                     "val_mse": float(val_mse),
                     **self._theory_meta()},
                    ckpt_path,
                )

        self._save_predictions(model, device, stats)
        # ``checkpoint.save_last`` (default false) additionally keeps the final weights
        if self._save_last_enabled():
            torch.save(
                {"model": model.state_dict(),
                 "criterion": criterion.state_dict(),
                 "epoch": self.max_epochs,
                 "val_mse": float(val_mse),
                 **self._theory_meta()},
                self.run_info.checkpoints_dir / "last_model.pth",
            )

        final_metrics = {
            "train_loss_final": float(train_loss),
            "val_mse_final": float(val_mse),
            "val_mse_best": float(best_val),
            "best_epoch": int(best_epoch),
            "max_epochs": int(self.max_epochs),
            "lr_final": float(scheduler.get_last_lr()[0]),
            "dim": int(self.dim),
        }
        # theory-contract summary (injected by the entry point); absent by default so the
        # released metrics key set is unchanged
        tc = self.cfg.extra.get("theory_contract")
        if isinstance(tc, dict) and tc:
            final_metrics["theory_contract"] = dict(tc)
        return self.finalize(final_metrics)

    def _theory_meta(self) -> Dict[str, Any]:
        """Theory-contract summary embedded in checkpoints (empty when unavailable).

        Why inside the checkpoint: a ``.pth`` is often copied and reused on its own, and
        then the run-directory ``theory_contract.json`` is gone. Carrying
        ``variational_covariance`` / ``kl_mode`` lets any weight file state whether it is
        full or diagonal and whether the objective was strict.
        """
        tc = self.cfg.extra.get("theory_contract")
        return {"theory_contract": dict(tc)} if isinstance(tc, dict) and tc else {}

    def _save_last_enabled(self) -> bool:
        """Whether ``checkpoint.save_last`` asks for an extra copy of the final weights.

        Read through ``full_config`` (the top-level config, injected by the entry point).
        Without that attribute the answer is ``False``, i.e. the released behaviour.
        """
        cfg = getattr(self, "full_config", None) or {}
        return bool((cfg.get("checkpoint") or {}).get("save_last", False))

    def _save_predictions(self, model, device, stats) -> None:
        """Write the point-estimate inversion field into ``reconstructed/`` of the run.

        Produces ``predictions.npz`` with the physical-space inversion mean ``mu_phys``,
        the normalized mean / log-variance, the gating ``z_soft``, the raw gating
        ``logits``, the normalized ground truth ``target_norm`` and ``mask`` — enough for
        the evaluation tools and the figures to score each property without a dataset
        instance.
        """
        loader = self.val_loader or self.train_loader
        if loader is None or not stats:
            logger.info("no validation loader / statistics; skipping prediction saving")
            return
        model.eval()

        def bc(s: torch.Tensor) -> torch.Tensor:
            # the statistics moved to device in run(); back to CPU and broadcast (1,C,1,...)
            return torch.as_tensor(s).cpu().float().reshape(1, -1, *([1] * self.dim))

        tm, ts = bc(stats["target_mean"]), bc(stats["target_std"])
        parts: Dict[str, list] = {"mu_norm": [], "logvar_norm": [], "target_norm": [],
                                  "z_soft": [], "mask": [], "logits": []}
        with torch.no_grad():
            for batch in loader:
                seismic = batch["seismic"].to(device)
                target = batch["target"].to(device)
                target_low = batch["target_low"].to(device)
                prior = None if batch["prior"] is None else batch["prior"].to(device)
                mask = batch["mask"].to(device)

                out = model(seismic, low_freq_model=target_low, prior=prior, num_samples=0)
                mu = out["final_pred"][:, :3]
                parts["mu_norm"].append(mu.detach().cpu())
                parts["logvar_norm"].append(out["final_pred"][:, 3:].detach().cpu())
                parts["target_norm"].append(target.detach().cpu())
                parts["z_soft"].append(out["z_soft"].detach().cpu())
                # raw gating logits: lets the entropy be re-examined offline
                parts["logits"].append(out["logits"].detach().cpu())
                parts["mask"].append(mask.detach().cpu())

        out_dir = self.run_info.reconstructed_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        merged = {k: torch.cat(v, dim=0).numpy() for k, v in parts.items()}
        mu_phys = merged["mu_norm"] * ts.numpy() + tm.numpy()
        np.savez(out_dir / "predictions.npz", mu_phys=mu_phys, **merged)
        np.save(out_dir / "final_pred_physical.npy",
                np.concatenate([mu_phys, merged["logvar_norm"]], axis=1))
        logger.info("predictions written to %s", out_dir)

    # ---- phase management ----
    def _phase_boundaries(self) -> List[int]:
        """Cumulative epoch counts at which each phase ends."""
        if not self.phases:
            return []
        boundaries, acc = [], 0
        for i, ph in enumerate(self.phases):
            if "epochs" not in ph:
                raise KeyError(f"phases[{i}] is missing the 'epochs' key: {ph!r}")
            n = int(ph["epochs"])
            if n <= 0:
                raise ValueError(f"phases[{i}]['epochs'] must be positive, got {n}")
            acc += n
            boundaries.append(acc)
        return boundaries

    def _phase_boundary(self, epoch: int) -> bool:
        """True on the **first** epoch of a new phase (boundary + 1)."""
        if not self.phases:
            return False
        return any(b + 1 == epoch for b in self._phase_boundaries())

    def _phase_index(self, epoch: int) -> Optional[int]:
        """Index of the phase ``epoch`` belongs to (clamped to the last phase)."""
        if not self.phases:
            return None
        idx = 0
        for i, b in enumerate(self._phase_boundaries()):
            if epoch > b:
                idx = i + 1
        # when max_epochs exceeds sum(phases.epochs), clamp to the last phase
        return min(idx, len(self.phases) - 1)

    def _phase_lr(self, idx: int) -> float:
        return float(_phase_defaults(self.phases[idx], "lr", self.lr))

    def _apply_phase(self, model, criterion, phase_index: int) -> None:
        """Switch the model and the KL weights for ``phase_index``.

        A phase may set ``freeze_gating: false`` to keep the gating network trainable
        (joint fine-tuning, ``set_phase(3)``) instead of frozen.
        """
        if not self.phases:
            if hasattr(model, "set_phase"):
                model.set_phase(1)
            return
        ph = self.phases[phase_index]
        if hasattr(model, "set_phase"):
            freeze = bool(ph.get("freeze_gating", True))
            if phase_index >= 1 and not freeze:
                model.set_phase(3)
            else:
                model.set_phase(phase_index + 1)
        criterion.set_kl_weights(
            float(ph.get("kl_z_weight", criterion.kl_z_weight)),
            float(ph.get("kl_theta_weight", criterion.kl_theta_weight)),
        )
        # A phase boundary rewrites the KL weights, so the strictness of the objective can
        # change per phase (e.g. stage 1 single_temperature, stage 2 legacy_weighted when
        # the weights differ). Keep a per-phase snapshot; only when the entry point already
        # produced a contract, so the default path writes no extra file.
        if isinstance(self.cfg.extra.get("theory_contract"), dict):
            try:
                from src.utils.theory_contract import (
                    build_theory_contract,
                    persist_theory_contract,
                )

                contract = build_theory_contract(
                    model, criterion,
                    require_full=bool(
                        self.cfg.extra["theory_contract"].get("require_full", False)))
                persist_theory_contract(
                    self.run_info, contract,
                    name=f"theory_contract_phase{phase_index + 1}.json")
            except Exception as exc:  # noqa: BLE001 - an audit artefact must not stop training
                logger.warning("theory contract snapshot for phase %d failed: %s",
                               phase_index + 1, exc)

    def _build_optimizer(self, model, criterion, lr: float,
                         warmup_epochs: int | None = None,
                         sched_epochs: int | None = None,
                         warmup_start_lr: float | None = None):
        """Build the optimizer and its warmup-cosine scheduler.

        With ``gating_lr`` set, the partition layer becomes an independent parameter
        group so the gating network can learn at a different rate; otherwise a single
        group is used.
        """
        partition = getattr(model, "partition_layer", None)
        if self.gating_lr is not None and partition is not None:
            gate_ids = {id(p) for p in partition.parameters()}
            base_params = [p for p in model.parameters() if id(p) not in gate_ids]
            groups: List[Any] = [
                {"params": base_params, "lr": float(lr)},
                {"params": list(partition.parameters()), "lr": float(self.gating_lr)},
            ]
            crit_params = list(criterion.parameters())
            if crit_params:
                groups.append({"params": crit_params, "lr": float(lr)})
            params: Any = groups
        else:
            params = list(model.parameters()) + list(criterion.parameters())
        optimizer = torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=self.weight_decay,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            # explicit values at a phase boundary win; None keeps the global defaults
            warmup_epochs=self.warmup_epochs if warmup_epochs is None else int(warmup_epochs),
            max_epochs=self.max_epochs if sched_epochs is None else int(sched_epochs),
            warmup_start_lr=self.warmup_start_lr if warmup_start_lr is None else float(warmup_start_lr),
            eta_min=self.lr_min,
        )
        return optimizer, scheduler

    def _syn_norm(self, theta_norm: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
        """Normalized ``theta`` ``(N, 3, ...)`` -> normalized synthetic gathers ``(N, n_angles, ...)``.

        ``stats`` holds per-channel tensors; reshape to a leading singleton
        ``(1, C, 1, ...)`` so they broadcast correctly over ``(N, C, ...)``.
        """
        target_mean, target_std = stats["target_mean"], stats["target_std"]
        seismic_mean, seismic_std = stats["seismic_mean"], stats["seismic_std"]
        nd_sp = theta_norm.dim() - 2
        tm = target_mean.reshape(1, -1, *([1] * nd_sp))
        ts = target_std.reshape(1, -1, *([1] * nd_sp))
        syn_phys = self.forward(theta_norm * ts + tm)
        sm = seismic_mean.reshape(1, -1, *([1] * (syn_phys.dim() - 2)))
        ss = seismic_std.reshape(1, -1, *([1] * (syn_phys.dim() - 2)))
        return (syn_phys - sm) / (ss + 1e-8)

    # ---- one epoch ----
    def _train_one_epoch(self, model, criterion, optimizer, epoch, stats) -> tuple:
        model.train()
        num_batches = max(1, len(self.train_loader))
        total_loss = 0.0
        comps: Dict[str, float] = {}

        stats_ready = stats.get("seismic_mean") is not None and stats.get("target_mean") is not None

        for batch in self.train_loader:
            seismic = batch["seismic"].to(self.device)
            target = batch["target"].to(self.device)
            target_low = batch["target_low"].to(self.device)
            mask = batch["mask"].to(self.device)
            # the spatial prior may be None (the model falls back to a uniform prior)
            spatial_prior = batch["prior"]
            if spatial_prior is not None:
                spatial_prior = spatial_prior.to(self.device)

            optimizer.zero_grad()
            model_output = model(seismic, target_low, spatial_prior, num_samples=1)

            syn_norm = None
            if self.forward is not None and stats_ready:
                syn_norm = self._syn_norm(model_output["theta_sample"], stats)

            loss, loss_dict = criterion(
                model_output=model_output,
                target=target,
                syn_norm=syn_norm,
                seismic=seismic,
                mask=mask,
            )
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(self.grad_clip))
            optimizer.step()

            total_loss += float(loss.item())
            for k, v in loss_dict.items():
                val = float(v.item()) if torch.is_tensor(v) else float(v)
                comps[k] = comps.get(k, 0.0) + val

        return total_loss / num_batches, {k: v / num_batches for k, v in comps.items()}

    # ---- validation ----
    def _validate(self, model, device) -> float:
        """Point-estimate MSE over the well-masked positions."""
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in self.train_loader if self.val_loader is None else self.val_loader:
                seismic = batch["seismic"].to(device)
                target = batch["target"].to(device)
                target_low = batch["target_low"].to(device)
                mask = batch["mask"].to(device)
                prior = None if batch["prior"] is None else batch["prior"].to(device)

                out = model(seismic, low_freq_model=target_low, prior=prior, num_samples=0)
                mu = out["final_pred"][:, :3]
                if mask.shape[1] == 1:
                    mask = mask.expand(-1, 3, -1) if self.dim == 1 else mask.expand_as(mu)
                mu_valid = mu[mask.bool()]
                target_valid = target[mask.bool()]
                if mu_valid.numel() > 0:
                    total += float(((mu_valid - target_valid) ** 2).sum())
                    n += int(mu_valid.numel())
        return total / n if n > 0 else float("nan")