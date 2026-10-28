# Geo-VB-MILE

Variational Bayesian Mixture-of-Experts (VB-MoE) for seismic elastic-parameter inversion,
with the Marmousi2 experiment suite. Everything is configured through YAML with a
`_target_` registry (no hard-coded hyper-parameters), and every run writes a timestamped,
self-describing directory (config copy, command, git commit, environment, metrics, checkpoints).

## Method

| | |
|---|---|
| **Network** | U-Net encoder → variational partition layer (Gumbel-Softmax gating) → expert decoders → marginal mixture q(theta \| x) |
| **Posterior** | full-covariance expert posterior q(theta \| z=k) = N(mu_k, Sigma_k), Sigma_k = L_k L_k^T (delivery family; a diagonal layout is kept for the released 1-D checkpoint) |
| **Objective** | negative ELBO = `nll_well + seismic_nll + kl_z_weight·KL(z) + kl_theta_weight·KL(theta)`; `nll_well` is the analytic expectation E_q[-log p(y\|theta)] in 1-D and the moment-matched marginal NLL in 2-D |
| **Forward closure** | differentiable Aki–Richards + Ricker convolution inside the training loop |
| **Priors** | GMM parameter prior (full covariance, normalized space) + spatial prior (per-pixel posterior on the low-frequency model, Hungarian-aligned to the well GMM) |
| **Contract** | each run writes `theory_contract.json`: the posterior family, well-likelihood mode, KL-weight mode and noise mode actually used, plus consistency checks |

### 1-D vs 2-D

| | 1-D | 2-D |
|---|---|---|
| Sample | single trace `(6, 1200)` | well-centred window `(6, 1200, 15)` (`radius=7`) |
| Experts | shared decoder + per-expert adapters | independent decoders + spatial FiLM |
| Posterior | full covariance, 9-channel head | full covariance, 6-channel head + 3 off-diagonal channels |
| Gating | logits from the low-frequency model | logits from low-frequency model + global feature (`tau=0.7`) |
| Well likelihood | **analytic expectation** (closed form) | **moment-matched marginal** + per-expert λ̂ (Eq. 21) |
| KL objective | single temperature c = 0.01 (both weights equal) | phase-wise weights (β / λ_max), per-expert λ̂ |
| Training | single phase, 500 epochs | joint (50 ep) → expert refine (250 ep, gating frozen) |
| Gumbel temperature | 1.0 → 0.05 | 2.0 → 0.1 |
| Low-frequency sigma | 30 | 100 |
| Config | `configs/inversion/vbmile_1d.yaml` | `configs/inversion/vbmile_2d.yaml` |
| **Evaluation** | predict each trace directly | **sliding window** (`radius=7`), centre column only |

The 1-D branch is bit-wise state-dict compatible with the original trained checkpoint
(`tests/test_legacy_checkpoint.py` asserts `strict=True` loading with no missing or
unexpected keys).

### Full-covariance posterior

Every expert carries `q(theta | z=k) = N(mu_k, Sigma_k)` with `Sigma_k = L_k L_k^T`:

* the decoder keeps its 6-channel head and adds three off-diagonal channels
  (`L[1,0]`, `L[2,0]`, `L[2,1]`, row-major); the head is zero-initialised, so a fresh
  full-covariance model is bit-wise identical to the diagonal one at step 0;
* `KL(q || p)` is closed form, `½[tr(Sigma_p⁻¹ Sigma_q) + deltaᵀ Sigma_p⁻¹ delta − D +
  ln|Sigma_p| − ln|Sigma_q|]`, and works against diagonal or full priors;
* the marginal output uses `diag(Sigma_k)` from the law of total covariance, so the
  off-diagonals also reach `final_pred`; Monte-Carlo sampling uses `theta = mu + L eps`.

Two well-likelihood modes pair with it: `"expert"` (analytic `E_q[-log p(y|theta)]`, needs
`tr(Sigma_k)`; used in 1-D) and `"mixture_predictive"` (the only mode in which the
off-diagonal structure enters the likelihood). The 2-D delivery keeps the marginal mode.
A full-covariance posterior with a near-singular full prior needs `prior_cov_shrinkage`
(0.05–0.1), otherwise the posterior variance collapses.

### Seismic reliability masking (optional)

`src/forward/reliability.py` gives the seismic likelihood a per-angle weight built from the
analytic critical angle (`theta_c = arcsin(vp1/vp2)`, `90 deg` when `vp2 <= vp1`): unity
below `theta_c - taper`, a cosine taper to `floor`, and `floor` beyond. Masked angles
receive exactly zero gradient, while the reliable region keeps full strength under
`sum(w * nll) / sum(w)`. It is **off by default**:

```yaml
loss:
  seismic_mask_mode: critical_angle   # none (default) | critical_angle
  seismic_mask_layer_sigma: 0.0       # depth smoothing of Vp, samples; 0 = adjacent
  seismic_mask_taper_deg: 5.0         # cosine taper width, degrees
  seismic_mask_floor: 0.0             # weight kept beyond theta_c
```

On the 5–30° Marmousi2 data it is a strict no-op (every interface has `theta_c >= 32.8°`,
median 88.5°), so hard masking removes 0.0000% of the seismic gradient energy; the switch
targets wider-angle data, where Aki–Richards really degrades. With the mode on, the
trainer must supply the weight (a silent fallback raises), and `loss_dict` reports
`seismic_mask_occupancy(_energy)`.

## Install

```bash
pip install -r requirements.txt     # numpy scipy pyyaml pandas matplotlib scikit-learn torch
pip install -e .                    # optional
```

Python >= 3.9; a CUDA build of PyTorch is recommended for 2-D, 1-D also runs on CPU.

## Quick start

```bash
# 1. rebuild the processed data (see data/README.md for the raw inputs)
python -m tools.preprocess.run_preprocess --config configs/data/seismic1d.yaml
python -m tools.preprocess.run_preprocess --config configs/data/seismic2d.yaml

# 2. train / invert
python -m experiments.main.run_main -c configs/inversion/vbmile_1d.yaml --name ours_1d
python -m experiments.main.run_main -c configs/inversion/vbmile_2d.yaml --name ours_2d

# 3. evaluate the 2-D result with the sliding window
python -m tools.evaluate.b9c_2d_windowed_infer \
    --run-dir outputs/runs/<timestamp>_ours_2d_seed0 \
    --metrics-out outputs/summary/vbmile_2d_win15_metrics.json
```

Overrides accept dotted keys: `--seed 0`, `--device cuda`, `--set trainer.max_epochs=2`,
`--set model.base_channels=4`, `--set trainer.batch_size=1`; a 1-D smoke test is
`--set model.base_channels=4 --set trainer.max_epochs=2 --set trainer.batch_size=4`.

A run directory contains `config.yaml`, `cmd.txt`, `git_commit.txt`, `env.txt`,
`checkpoints/best_model.pth`, `checkpoints/last_model.pth` (when `checkpoint.save_last`
is true), `theory_contract.json` (+ phase snapshots), `reconstructed/`, `metrics.json` and
`logs/run.log`; both checkpoints embed the contract summary.

## 2-D inference must be a sliding window

The 2-D model is trained on narrow well-centred slices (`radius=7` → 15 traces, reflect
padding at the borders, only the centre column supervised). Evaluating the same checkpoint
on the whole `1200 x 1360` section changes the spatial context seen by the experts and the
FiLM modulation, so the two protocols are not comparable: always invert with the training
geometry. `--radius` is the window half-width; `--snr-tag` labels a metrics row and
`--compare <metrics.json>` prints the delta against another protocol.

The tool rebuilds the window exactly as the dataset does, keeps only the centre column,
reuses the training assembly (model, GMM prior, normalization) and runs an alignment
self-check (denormalized centre column vs `target.npy[:, :, trace]`, relative error
< 1e-5). It writes per-property R² / Pearson / NRMSE over all pixels plus profile SSIM on
[0,1]-normalized sections to `outputs/summary/<name>.json`.

Measured on the released architecture with `encoder_downsample: iso`, the smallest workable
trace width is **13** (the trace axis collapses before a `3x3` conv below that); width 15 is
the released training geometry, so any `--radius >= 6` works with the released checkpoint.
With the anisotropic encoder (`encoder_downsample: aniso`, already used by the 2-D recipe)
the trace axis stays above `model.min_down_size` and widths 3 / 5 / 15 also run.

## Reference values

Single-seed, delivery recipes, sliding-window protocol (2-D) / direct traces (1-D):

| | Vp | Vs | Den |
|---|---|---|---|
| 1-D (full + closed-form well NLL + c = 0.01) | 0.977 | 0.976 | 0.951 |
| 2-D (full on top of the B3 recipe) | 0.966 | 0.966 | 0.940 |

The run-to-run noise floor is Vp ±0.0034 / Vs ±0.0024 / Den ±0.0044 (1-D) and
Vp ±0.0027 / Vs ±0.0031 / Den ±0.0090 (2-D): differences below those are not resolvable.
The 2-D recipe also uses the deterministic protocol, the deviatoric seismic likelihood at weight 0.03 and per-expert dilations `[7, 15, 7]`.

## GMM prior

The released default is **K = 3**. The fitting is `GMMPriorFitter` (sklearn full-covariance
GaussianMixture, `reg_covar=1e-6`, then `(1-s) Sigma_k + s Sigma_global + eps I` with
`s=0.1`, `eps=1e-4`, `seed=42`); model selection and stability checks are in
`tools/preprocess/gmm_select.py` and `gmm_bootstrap_stability.py`.

The prior is injected when the config declares it:

```yaml
prior:
  _target_: gmm        # read data/processed/<dataset>/gmm_prior.npz
```

The 2-D recipe sets it together with `model.prior_cov_shrinkage: 0.1`; the 1-D delivery
deliberately uses the **unit prior** (`prior: {}`), because the single-temperature
objective `c·KL(q || N(0, I))` states that target. Use `prior._target_: none` for the
unit-prior ablation, or `configs/ablations/gaussian_prior.yaml` for a single Gaussian.
Shrinkage is a paired switch: with a diagonal or unit prior it is ignored and the contract
flags the combination as inert.

## Repository layout

```
vbmile-open/
├── configs/                  base + data + model + inversion + ablations
├── src/                      data, forward, models, priors, losses, trainers, metrics, utils
├── experiments/              common.py + main/run_main.py (train / invert)
├── tools/                    preprocess (data, noise, GMM), evaluate (sliding window), data_check
├── tests/                    model, losses, priors, data, protocol, legacy checkpoint
├── data/                     see data/README.md (splits committed, arrays rebuilt)
└── outputs/                  run artifacts (git-ignored)
```

## Tests

```bash
pytest
```

covers the forward/backward of both models, the temperature and phase mechanics, all
well-NLL modes, the full-covariance posterior, the 2-D structure, the deviatoric seismic
likelihood, the reliability masking, the stage-1 learning rate and `checkpoint.save_last`,
the GMM prior, the forward operators, the dataset contract, the sliding-window protocol and
the legacy 1-D state-dict compatibility.

Tests that need the rebuilt dataset skip automatically when `data/processed/` is absent, so
a fresh clone is green; with the data prepared the suite is 146 passed.

## Citation

If you use this code, please cite the Geo-VB-MILE paper and mention the Marmousi2 model
source proposed by Mejer et al.
