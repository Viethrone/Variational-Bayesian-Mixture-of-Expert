"""2-D sliding-window inference for a trained Geo-VB-MILE 2-D checkpoint.

Why this exists
---------------
The 2-D model is *trained* on narrow well-centred slices
(``data.window.mode=slice``, ``radius=7`` → 15 traces wide, reflect padded at the
borders). Evaluating the same checkpoint by pushing the whole 1200 x 1360 section
through the network in one forward pass (``window.mode=full``) therefore changes the
spatial context the experts and the FiLM modulation see, so the two protocols are
**not** comparable.

This script re-runs inference with the *training geometry*: for every test trace a
width-15 window centred on it, one forward pass, and only the centre column is kept.

It reuses the training assembly chain (``build_trainer`` +
``experiments.main.run_main._assemble_vbmile_trainer``) so the model, loss, forward
operator, GMM prior and normalization statistics are constructed exactly as during
training; only ``data.window`` and the val split are overridden.

Alignment self-check
-----------------------
Before computing metrics the script compares the *denormalized* centre column of the
windowed target against ``target.npy[:, :, trace]`` element-wise, proving the trace
order and the centre-column index (== ``radius``) are correct. It asserts a relative
error < 1e-5 and aborts otherwise.

Metrics
--------
Per-property R² / Pearson / NRMSE over **all pixels** of the reconstructed section,
plus profile SSIM on [0,1]-normalized sections with ``data_range=1.0`` — the same
formulas as the section-evaluation helpers in ``tools/evaluate``.

Usage (cwd = repository root)::

    python -m tools.evaluate.b9c_2d_windowed_infer \
        --run-dir outputs/runs/<timestamp>_ours_2d_seed0 \
        --metrics-out outputs/summary/vbmile_2d_win15_metrics.json \
        --compare outputs/summary/vbmile_2d_full_metrics.json

Optional flags::

    --split data/splits/marmousi2_2d/test.txt    # trace list to invert
    --radius 7                                   # window half-width (0 = single trace)
    --batch-size 8
    --snr-tag clean                              # label written into the metrics rows
    --device cuda                                # default: cpu (deterministic, portable)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ATTRS = ["Vp", "Vs", "Den"]


def _resolve(p: str) -> Path:
    from experiments.common import project_root

    q = Path(p)
    return q if q.is_absolute() else (project_root() / q)


def _per_property(pred: np.ndarray, tgt: np.ndarray) -> dict[str, dict[str, float]]:
    """pred/tgt (3, L, n) → {name: {r2, pearson, nrmse, ssim}} (2-D section protocol)."""
    from src.metrics.base import SSIM

    out: dict[str, dict[str, float]] = {}
    L, n = pred.shape[1], pred.shape[2]
    for i, name in enumerate(ATTRS):
        pr = pred[i].reshape(L * n)
        tg = tgt[i].reshape(L * n)
        ss_res = float(np.sum((pr - tg) ** 2))
        ss_tot = float(np.sum((tg - tg.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        pcc = float(np.corrcoef(pr, tg)[0, 1]) if pr.std() > 0 else float("nan")
        nrmse = float(np.sqrt(np.mean((pr - tg) ** 2)) / tg.std()) if tg.std() > 0 else float("nan")
        p_img = (pred[i] - pred[i].min()) / max(pred[i].max() - pred[i].min(), 1e-12)
        t_img = (tgt[i] - tgt[i].min()) / max(tgt[i].max() - tgt[i].min(), 1e-12)
        win = min(7, p_img.shape[0], p_img.shape[1])
        ssim = float(SSIM(data_range=1.0).compute(p_img, t_img)) if win >= 3 else float("nan")
        out[name] = {"r2": r2, "pearson": pcc, "nrmse": nrmse, "ssim": ssim, "n_pixels": int(L * n)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="2-D training run directory (config.yaml + checkpoints/best_model.pth)")
    ap.add_argument("--split", default="data/splits/marmousi2_2d/test.txt")
    ap.add_argument("--radius", type=int, default=7,
                    help="window half-width (training geometry radius=7 → width 15)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-root", default=None,
                    help="predictions.npz root (default: <run-dir>/sections_2d_win)")
    ap.add_argument("--metrics-out", required=True)
    ap.add_argument("--snr-tag", default=None,
                    help="label for the metrics rows (default: win<width>_center)")
    ap.add_argument("--compare", default=None,
                    help="metrics json from a different protocol (prints the delta)")
    ap.add_argument("--target-config", default=None,
                    help="data config with target.npy (default: data block of the run config)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    from experiments.main.run_main import _assemble_vbmile_trainer
    from src.trainers.base import build_trainer
    from src.utils.config import load_config
    from src.utils.run_info import RunInfo

    run_dir = Path(args.run_dir).resolve()
    cfg = load_config(str(run_dir / "config.yaml"))
    cfg.setdefault("data", {})
    cfg["data"].setdefault("splits", {})["val"] = str(_resolve(args.split))
    cfg["data"]["window"] = {"mode": "slice", "radius": args.radius, "pad": "reflect"}
    cfg["data"].setdefault("loader", {})["num_workers"] = 0
    cfg.setdefault("trainer", {})["batch_size"] = args.batch_size
    cfg["exp_name"] = f"vbmile_2d_win{2 * args.radius + 1}_center"

    traces = [int(x) for x in _resolve(args.split).read_text(encoding="utf-8").split()]
    out_root = Path(args.out_root) if args.out_root else run_dir / "sections_2d_win"
    run_info = RunInfo.create(cfg, outputs_root=out_root, exp_name=cfg["exp_name"] + "_seed0")

    trainer = build_trainer(cfg, run_info)
    trainer = _assemble_vbmile_trainer(trainer, cfg, run_info)
    trainer.dim = int(getattr(trainer.model, "dim", 1))
    assert trainer.dim == 2, f"this tool only accepts 2-D checkpoints (dim={trainer.dim})"

    ckpt_path = run_dir / "checkpoints" / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    trainer.model.load_state_dict(ckpt["model"])
    device = torch.device(args.device)
    trainer.device = device
    trainer.model.to(device)
    print(f"[i] checkpoint: {ckpt_path.name} (epoch {ckpt.get('epoch')}, "
          f"val_mse {ckpt.get('val_mse'):.6f})")
    print(f"[i] sliding window: {len(traces)} traces x width {2 * args.radius + 1} "
          f"(radius={args.radius}, reflect padding), batch={args.batch_size}, "
          f"centre index={args.radius}")

    stats = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in trainer.stats.items()}
    tm = torch.as_tensor(stats["target_mean"]).float().reshape(1, 3, 1, 1)
    ts = torch.as_tensor(stats["target_std"]).float().reshape(1, 3, 1, 1)

    loader = trainer.val_loader or trainer.train_loader
    mu_centers, tgt_centers = [], []
    trainer.model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            seismic = batch["seismic"].to(device)
            target_low = batch["target_low"].to(device)
            prior = None if batch["prior"] is None else batch["prior"].to(device)
            out = trainer.model(seismic, low_freq_model=target_low, prior=prior, num_samples=0)
            # denormalize with the same broadcast as VBEMLETrainer._save_predictions
            mu_phys = out["final_pred"][:, :3].cpu() * ts + tm
            tgt_phys = batch["target"].cpu() * ts + tm
            mu_centers.append(mu_phys[:, :, :, args.radius])      # (B,3,L) centre trace
            tgt_centers.append(tgt_phys[:, :, :, args.radius])
    wall = time.perf_counter() - t0
    pred = torch.cat(mu_centers, 0).numpy()                       # (n,3,L) in split order
    tgt_win = torch.cat(tgt_centers, 0).numpy()
    print(f"[i] inference done: {pred.shape}, wall={wall:.1f}s "
          f"({wall / max(1, len(traces)) * 1000:.0f} ms/trace)")

    # ---- alignment self-check: windowed target centre column == target.npy[:, :, trace] ----
    dcfg = cfg.get("data") or {}
    root = Path(str(dcfg.get("root", "data/processed")))
    if not root.is_absolute():
        root = _resolve(str(root))
    if args.target_config:
        tcfg = load_config(args.target_config)
        root = Path(str(tcfg["data"]["root"]))
        if not root.is_absolute():
            root = _resolve(str(root))
        tname = str((tcfg["data"].get("files") or {}).get("target", "target.npy"))
    else:
        tname = str((dcfg.get("files") or {}).get("target", "target.npy"))
    target = np.load(root / tname)
    ref = target[:, :, traces].transpose(2, 0, 1)                 # (3,L,n) -> (n,3,L)
    err = float(np.max(np.abs(tgt_win - ref)))
    scale = float(np.max(np.abs(ref))) or 1.0
    print(f"[i] alignment check: max|windowed target centre - target.npy[:,:,traces]| "
          f"= {err:.3e} (scale {scale:.1e})")
    assert err / scale < 1e-5, "trace order / centre-column index misaligned"

    # ---- per-property metrics + persist ----
    section = pred.transpose(1, 2, 0)                             # (3,L,n)
    metrics = _per_property(section, target[:, :, traces])
    npz_dir = run_info.reconstructed_dir
    npz_dir.mkdir(parents=True, exist_ok=True)
    np.savez(npz_dir / "predictions.npz", mu_phys=pred,
             trace_indices=np.asarray(traces, np.int64),
             mode=f"slice_win{2 * args.radius + 1}_center")

    snr_tag = args.snr_tag or f"win{2 * args.radius + 1}_center"
    rows = []
    for i, name in enumerate(ATTRS):
        m = metrics[name]
        rows.append({"method": "vbmile_2d", "snr": snr_tag, "property": name,
                     "r2": float(m["r2"]), "pearson": float(m["pearson"]),
                     "nrmse": float(m["nrmse"]), "ssim": float(m["ssim"]),
                     "n_traces": int(pred.shape[0]), "n_depth": int(pred.shape[2]),
                     "n_pixels": int(m["n_pixels"]),
                     "predictions": str(npz_dir / "predictions.npz")})
    out = {"task": "vbmile_2d_windowed_infer", "run_dir": str(run_dir), "radius": args.radius,
           "window_width": 2 * args.radius + 1, "center_index": args.radius, "wall_s": wall,
           "notes": {
               "protocol": "sliding window (train geometry) + centre-trace extraction; "
                           "reflect padding at section borders; all test traces covered",
               "denorm": "mu_phys = mu_norm*target_std + target_mean "
                         "(same as VBEMLETrainer._save_predictions)",
               "alignment": f"max rel err vs target.npy = {err / scale:.3e}"},
           "rows": rows}
    mp = Path(args.metrics_out)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[ok] {mp}")

    print("\n[table] sliding window + centre trace vs comparison protocol:")
    comp = None
    if args.compare:
        comp = {r["property"]: r for r in json.loads(Path(args.compare).read_text("utf-8"))["rows"]}
    for r in rows:
        line = (f"   {r['property']:<4} win: R2={r['r2']:.4f} PCC={r['pearson']:.4f} "
                f"SSIM={r['ssim']:.4f}")
        if comp and r["property"] in comp:
            c = comp[r["property"]]
            line += (f" | cmp: R2={c['r2']:.4f} PCC={c['pearson']:.4f} SSIM={c['ssim']:.4f}"
                     f" | dR2={r['r2'] - c['r2']:+.4f}")
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
