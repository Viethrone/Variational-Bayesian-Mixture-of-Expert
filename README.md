# Geo-VB-MILE

Variational Bayesian Mixture-of-Experts (VB-MoE) for seismic elastic-parameter inversion,
with the complete experiment suite used on the **Marmousi2** model.

The release contains the method itself and the three experiment families that go with it:

1. **Data preprocessing** — raw angle gathers + elastic parameters to the model-ready
   arrays (low-frequency model, spatial prior, GMM prior) and the normalization statistics.
2. **GMM-prior inversion** — fit a K=3 full-covariance Gaussian mixture in the normalized
   space, then inject it into the expert output-bias initialization and the per-expert
   `KL(theta)` term, and invert by loading the processed prior.
3. **SNR robustness** — add Gaussian white noise at 2 / 5 / 10 / 15 dB to the observed
   gathers and quantify the inversion degradation per property.
4. **2-D evaluation protocol** — the 2-D model is trained on narrow well-centred slices,
   so inference is a **sliding window** with centre-column extraction, never a
   whole-section forward pass.

Everything is configurable through YAML with a `_target_` registry, no hard-coded
hyper-parameters in the code. All experiments write a timestamped, self-describing run
directory (config copy, command, git commit, environment, metrics).

## Method

| | |
|---|---|
| **Network** | U-Net encoder → variational partition layer (Gumbel-Softmax gating) → expert decoders → marginal mixture q(theta \| x) |
| **Posterior** | full-covariance expert posterior q(theta \| z=k) = N(mu_k, Sigma_k), Sigma_k = L_k L_k^T (delivery family; a diagonal layout is kept for the released 1-D checkpoint) |
| **Objective** | Negative ELBO = `nll_well + seismic_nll + kl_z_weight·KL(z) + kl_theta_weight·KL(theta)`; `nll_well` is the analytic expectation E_q[-log p(y\|theta)] in 1-D and the moment-matched marginal NLL in 2-D |
| **Forward closure** | differentiable Aki-Richards + Ricker convolution, in the training loop |
| **Priors** | GMM parameter prior (full covariance, in normalized space) + spatial prior (per-pixel posterior on the low-frequency model, Hungarian-aligned to the well GMM) |
| **Contract** | every run writes `theory_contract.json`: the posterior family, well-likelihood mode, KL-weight mode and noise mode actually used, plus their consistency checks |
| **Regularization** | ASO (structure-oriented) can be enabled via config |

### 1-D vs 2-D

| | 1-D | 2-D |
|---|---|---|
| Sample | single trace `(6, 1200)` | well-centred window `(6, 1200, 15)` (`radius=7`) |
| Encoder | `UNetEncoder1D` | `UNetEncoder2D` |
| Experts | shared decoder + per-expert adapters | fully independent decoders + spatial FiLM |
| Posterior | full covariance, 9-channel head | full covariance, 6-channel head + 3 off-diagonal channels |
| Gating | logits from the low-frequency model | logits from low-frequency model + global feature (`tau=0.7` built in) |
| Well likelihood | **analytic expectation** (closed form, no sampling) | **moment-matched marginal** + per-expert λ̂ (Eq. 21) |
| KL objective | single temperature c = 0.01 (both weights equal) | phase-wise weights (β / λ_max), per-expert λ̂ |
| Training | single phase, 500 epochs | two phases: joint (50 ep) → expert refine (250 ep, gating frozen) |
| Gumbel temperature | 1.0 → 0.05 | 2.0 → 0.1 |
| Low-frequency sigma | 30 | 100 |
| Config | `configs/inversion/vbmile_1d.yaml` | `configs/inversion/vbmile_2d.yaml` |
| **Evaluation** | predict each trace directly | **sliding window** (`radius=7`), centre column only — see below |

The 1-D branch is bit-wise state-dict compatible with the original trained checkpoint
(`tests/test_legacy_checkpoint.py` asserts `strict=True` loading with zero
missing / unexpected keys). That checkpoint has a 6-channel diagonal head, which is why
`model.variational_covariance: diag` is still the default layout; the delivery recipes
switch it to `full`.

### Variational posterior: full covariance

Both delivery recipes use ``model.variational_covariance: full``, i.e. every expert
carries ``q(theta | z=k) = N(mu_k, Sigma_k)`` with ``Sigma_k = L_k L_k^T``:

* the decoder keeps its 6-channel head and **adds** three off-diagonal channels, mapped to
  ``L[1,0]``, ``L[2,0]``, ``L[2,1]`` in that order (row-major lower triangle); the head is
  **zero-initialised**, so a fresh full-covariance model is bit-wise identical to the
  diagonal one at step 0 — the change extends the model instead of rewriting it;
* ``KL(q || p)`` is closed form, ``½[tr(Sigma_p^-1 Sigma_q) + delta^T Sigma_p^-1 delta -
  D + ln|Sigma_p| - ln|Sigma_q|]``, and works against a diagonal or a full prior;
* the marginal output uses ``diag(Sigma_k)`` from the law of total covariance, so the
  off-diagonal terms also reach ``final_pred``;
* Monte-Carlo sampling uses ``theta = mu + L eps``; keeping ``sigma * eps`` would leave the
  gradient path diagonal while the head claims otherwise.

Two well-likelihood modes pair with it: ``"expert"`` (the analytic
``E_q[-log p(y|theta)]``, which needs ``tr(Sigma_k)``; used in 1-D) and
``"mixture_predictive"`` (the only mode where the off-diagonal structure enters the
likelihood). The 2-D delivery keeps the marginal mode, which measured no worse on the B3
baseline.

A full-covariance posterior with a near-singular full prior needs ``prior_cov_shrinkage``
(0.05–0.1): without it the posterior variance collapses. The theory contract checks that
pair explicitly — 0.1 with a non-full prior is flagged as inert.


### 2-D recipe

The 2-D network has a **single structural form**; there is no switch that brings the
notebook layout back. `configs/inversion/vbmile_2d.yaml` fixes the remaining degrees of
freedom (encoder geometry, per-expert receptive field, likelihood and schedule):

| Component | What it does | Why |
|---|---|---|
| merged output head | one 6-channel head per expert, `[μ_vp, μ_vs, μ_ρ, logσ²_vp, logσ²_vs, logσ²_ρ]` | −14.2 % parameters versus three 2-channel heads, no measurable accuracy change |
| `skip2` convolution | convolved at its **native** resolution, then upsampled | same parameters, half the convolution work on the trace axis |
| same-size tail | `final_convs` / `global_residual` use `padding = dilation` | no `1200×15 → 1190×5 → 1200×15` bilinear write-back; the head runs at full resolution |
| copy-kernel reflect padding | padding built from `narrow` / `flip` / `cat` | identical numerics to `F.pad(mode="reflect")`, but the backward pass is deterministic on CUDA |
| `model.expert_dilations: [7, 15, 7]` | per-expert dilation of the fusion block | a wider receptive field for the deep hard-rock expert; no new parameters |
| `loss.seismic_noise: deviatoric` + `loss.seismic_weight: 0.03` | seismic likelihood with the trace common mode marginalised | the forward-operator mismatch is coherent along the trace axis; this constrains only the trace-to-trace differences and cuts the systematic bias |

`expert_dilations` is a plain attribute — it never enters the state dict, and it is the
only structural degree of freedom left: a dilation changes the numerics without changing a
single parameter.

Two training-loop details matter for reproducing the released numbers: the first phase's
learning rate is used when the optimizer is built (a later phase only rebuilds it at a
phase boundary), and `data.loader.explicit_generator: true` takes the shuffle order and
the worker seeds from an explicit generator, so `num_workers` cannot move the trajectory.
Together with `deterministic: true` + `deterministic_algorithms: true` a same-seed rerun is
bit-wise identical.


## Install

```bash
pip install -r requirements.txt     # numpy scipy pyyaml pandas matplotlib scikit-learn torch
pip install -e .                    # optional, makes `src` importable from anywhere
```

Python >= 3.9. A CUDA build of PyTorch is recommended for 2-D inverse modelling; 1-D
experiments also run on CPU.

## Quick start

### 1. Prepare the data

Put the raw arrays in place (see [data/README.md](data/README.md)) and rebuild:

```bash
python -m tools.preprocess.run_preprocess --config configs/data/seismic1d.yaml
python -m tools.preprocess.run_preprocess --config configs/data/seismic2d.yaml
```

### 2. Train / invert 1-D

```bash
python -m experiments.main.run_main -c configs/inversion/vbmile_1d.yaml --name ours_full
```

### 3. Train / invert 2-D (two phases)

```bash
python -m experiments.main.run_main -c configs/inversion/vbmile_2d.yaml --name ours_2d
```

This is the recipe documented above (300 epochs, structural subtraction, deterministic
padding, deviatoric seismic likelihood).

### 4. Evaluate the 2-D result with the sliding window

`run_main` already writes a whole-section forward pass into the run directory, but that
is a **convenience output, not the evaluation protocol** — see
[2-D inference must be a sliding window](#2-d-inference-must-be-a-sliding-window):

```bash
python -m tools.evaluate.b9c_2d_windowed_infer \
    --run-dir outputs/runs/<timestamp>_ours_2d_seed0 \
    --metrics-out outputs/summary/vbmile_2d_win15_metrics.json
```

Useful command-line overrides (all keys accept dotted paths):

```bash
--seed 0                                # repeat runs with a fixed seed
--device cuda                           # or cpu / mps
--set trainer.max_epochs=2              # smoke test
--set model.base_channels=4             # smaller network
--set trainer.batch_size=1              # smaller 2-D batch
```

Every run lands in `outputs/runs/<timestamp>_<name>/` with `config.yaml`, `cmd.txt`,
`git_commit.txt`, `env.txt`, `checkpoints/best_model.pth`,
`checkpoints/last_model.pth` (while `checkpoint.save_last` is true),
`theory_contract.json` (+ `theory_contract_phase<N>.json` at phase boundaries),
`reconstructed/predictions.npz`, `reconstructed/final_pred_physical.npy`, `metrics.json`
and `logs/run.log`. Both checkpoints embed the theory-contract summary, so a weight file
copied on its own still states whether it is a full or diagonal posterior and how the
objective was weighted.

`--set model.base_channels=4 --set trainer.max_epochs=2 --set trainer.batch_size=4` is a
complete 1-D smoke test that finishes in seconds.

> **2-D inference protocol**: evaluating a 2-D checkpoint must use the sliding window,
> not the whole-section forward pass — see
> [2-D inference must be a sliding window](#2-d-inference-must-be-a-sliding-window) below.

## Experiment: GMM prior

### Choosing the number of components K

```bash
python -m tools.preprocess.gmm_select                    # BIC/AIC/silhouette/CV log-likelihood
python -m tools.preprocess.gmm_bootstrap_stability       # bootstrap min-weight stability
```

The released default is **K = 3**, selected from:

* the bootstrap minimum-weight q05 stays above 5% for K <= 4 and falls below for K >= 5;
* the silhouette curve reaches a plateau;
* the pairwise Bhattacharyya distance between components explodes at K = 4 (200.3 for
  the k1-k4 pair), i.e. a spurious component splits an existing mode.

The fitting itself is `GMMPriorFitter` in `src/priors/gmm.py`
(sklearn `GaussianMixture`, full covariance, `reg_covar=1e-6`, then
`(1-s)*Sigma_k + s*Sigma_global + eps*I` with `s=0.1`, `eps=1e-4`, `seed=42`).

### Inverting with the processed prior

The prior is injected when the config declares it explicitly:

```yaml
prior:
  _target_: gmm        # read data/processed/<dataset>/gmm_prior.npz
```

`configs/inversion/vbmile_2d.yaml` sets it (together with
`model.prior_cov_shrinkage: 0.1`, which is what makes the full-covariance GMM usable with
a full-covariance posterior). The 1-D delivery recipe deliberately uses the **unit prior**
(`prior: {}`, i.e. `N(0, I)`): with the single-temperature objective
`c·KL(q || N(0, I))` the prior is part of the stated target, and that is the setting the
reported 1-D numbers were produced with. The training entry
(`experiments/main/run_main.py::_assemble_vbmile_trainer`) reads the file, converts the
normalized-space means/covariances into the model's `prior_means` / `prior_vars` buffers,
and the prior then drives

* the expert output-head bias and log-variance initialization, and
* the per-expert `KL(theta)` term inside the model forward pass
  (`_forward_1d` / `_forward_2d` in `src/models/vbmile.py`).

Set `prior._target_: none` to fall back to the unit prior explicitly (ablation), or use
`configs/ablations/gaussian_prior.yaml` for a single Gaussian instead.

### Full-covariance shrinkage

Finite-sample full covariance can be near-singular (condition number up to ~5800,
smallest eigenvalue ~3e-4), which inflates `KL(theta)` by two orders of magnitude. The 2-D
delivery sets

```yaml
model:
  prior_cov_shrinkage: 0.1
```

which shrinks toward `(tr(Sigma)/d)·I` and keeps the optimization stable. It is a **paired
switch**: with a diagonal or unit prior the value is silently ignored, so the theory
contract flags that combination as inert. The default `0.0` keeps the prior untouched.

## 2-D inference must be a sliding window

The 2-D model is **trained** on narrow well-centred slices
(`data.window.mode=slice`, `radius=7` → 15 traces wide, reflect padding at the borders,
only the centre column supervised). Evaluating the same checkpoint by pushing the whole
`1200 x 1360` section through the network in one forward pass (`window.mode=full`)
changes the spatial context seen by the experts and the FiLM modulation, so the two
protocols are **not comparable** — always invert with the training geometry:

```bash
python -m tools.evaluate.b9c_2d_windowed_infer \
    --run-dir outputs/runs/<timestamp>_ours_2d_seed0 \
    --metrics-out outputs/summary/vbmile_2d_win15_metrics.json
```

What the tool does:

* for every trace in `--split` (default `data/splits/marmousi2_2d/test.txt`, 1360 traces)
  it builds a `radius=7` window the same way the dataset does, runs **one** forward pass
  and keeps **only the centre column**;
* it reuses the training assembly chain, so model / GMM prior / normalization statistics
  are identical to training; only `data.window` and the val split are overridden;
* an **alignment self-check** denormalizes the windowed target centre column and compares
  it element-wise against `target.npy[:, :, trace]`, asserting a relative error < 1e-5 —
  this is what proves the trace order and the centre-column index are correct;
* per-property R² / Pearson / NRMSE over **all pixels** of the section plus profile SSIM
  on [0,1]-normalized sections (`data_range=1.0`), written as
  `outputs/summary/<name>.json` with the section itself in
  `<run-dir>/sections_2d_win/.../predictions.npz`.

Useful flags: `--radius` (window half-width), `--batch-size`, `--snr-tag` (label the
metrics row, e.g. `clean` / `snr2`), `--compare <metrics.json>` (prints the delta against
another protocol, e.g. a whole-section forward pass), `--device cuda`.

### Window width has a hard floor

Measured on the released architecture with `encoder_downsample: iso` (the default): the
smallest workable trace width is **13**, because the two stride-2 encoder blocks plus the
decoder's upsampling path need at least 3 traces at the coarsest level:

| width | feasible | note |
| --- | --- | --- |
| 3 / 5 / 7 / 9 / 11 | **no** | the trace axis collapses to 1–2 before a `3x3` conv (`tests/test_2d_windowed_infer.py::test_window_width_floor` pins this) |
| 13 | yes | smallest runnable |
| 15 | yes | **released training geometry** |

The released training geometry (`radius=7`, width 15) is safely above the floor, and any
`--radius` ≥ 6 works with the released checkpoint. If you need the CSI-STMC geometry
(width 5), train **and** infer with the anisotropic encoder:

```bash
python -m experiments.main.run_main -c configs/inversion/vbmile_2d.yaml --name ours_2d_w5 \
    --set model.encoder_downsample=aniso --set data.window.radius=2
python -m tools.evaluate.b9c_2d_windowed_infer \
    --run-dir outputs/runs/<timestamp>_ours_2d_w5_seed0 --radius 2 \
    --metrics-out outputs/summary/vbmile_2d_win5_metrics.json
```

`aniso` keeps the depth-axis downsampling normal and leaves a trace axis that would fall
below `model.min_down_size` (default 3) at its original width, which makes widths
3 / 5 / 15 all runnable (pinned by `test_aniso_encoder_allows_narrow_windows`). The
released 2-D recipe already uses `aniso` (`configs/inversion/vbmile_2d.yaml`), so the
`--set` above only matters when you train with `iso` explicitly.

Reference values, both seed 0 with the delivery recipes, were obtained with the windowed
protocol above, i.e. with the same geometry the model was trained on:

* 1-D (full covariance + closed-form well likelihood + single temperature c = 0.01):
  Vp ≈ 0.977 / Vs ≈ 0.976 / Den ≈ 0.951 on the 1360-trace test set;
* 2-D (full covariance on top of the B3 recipe): Vp ≈ 0.966 / Vs ≈ 0.966 / Den ≈ 0.940
  under the sliding-window protocol (centre column).

These are single-seed numbers: the run-to-run noise floor is Vp ±0.0034 / Vs ±0.0024 /
Den ±0.0044 (1-D) and Vp ±0.0027 / Vs ±0.0031 / Den ±0.0090 (2-D), so differences below
those values are not resolvable.

## Experiment: SNR robustness

### Generate the noisy observations

```bash
python -m tools.preprocess.add_noise --config configs/data/seismic1d.yaml --snr 2 5 10 15 --seed 0
python -m tools.preprocess.add_noise --config configs/data/seismic2d.yaml --snr 2 5 10 15 --seed 0
```

Noise is Gaussian white, added in physical space with a single global scale
`sigma = sqrt(P_signal · 10^(-SNR_dB/10))`, seed 0. Only the seismic file changes;
target, low-frequency model, priors and normalization statistics stay identical so runs
are directly comparable. The script reports the *measured* SNR of each version
(measured values were 2.0048 / 4.9987 / 9.9993 / 14.9976 dB).

### Run the arm and compare

1-D (one run per SNR level, then read `reconstructed/predictions.npz`):

```bash
for snr in 2 5 10 15; do
  python -m experiments.main.run_main -c configs/inversion/vbmile_1d.yaml \
      --name m1_snr${snr} --set data.files.seismic=seismic6_snr${snr}.npy
done
```

2-D (same training switch, but evaluate with the sliding-window protocol and label each
metrics row with `--snr-tag` so the noise levels stay comparable):

```bash
for snr in 2 5 10 15; do
  python -m experiments.main.run_main -c configs/inversion/vbmile_2d.yaml \
      --name m1_2d_snr${snr} --set data.files.seismic=seismic6_snr${snr}.npy
  python -m tools.evaluate.b9c_2d_windowed_infer \
      --run-dir outputs/runs/<timestamp>_m1_2d_snr${snr}_seed0 \
      --snr-tag snr${snr} \
      --metrics-out outputs/summary/m1_2d_snr${snr}_win15_metrics.json
done
```

The per-property R² / PCC / NRMSE / SSIM are in each run's
`reconstructed/predictions.npz` (keys `mu_phys`, `target_norm`, `z_soft`, `mask`) and can
be compared with `tools/evaluate/waveform_compare.py`.

Reference numbers on the 1-D test set (1360 traces, all properties): the clean anchor
sits near R² 0.96 / 0.97 / 0.94 for Vp / Vs / density, and **2 dB is the only level with
a clearly visible drop** (Vp R² ~ 0.9568); 5 / 10 / 15 dB recover to the clean level. The
2-D gating stays stable across all noise levels (entropy ≈ 0.19, hard-assignment ARI
0.91–0.95 against the clean anchor).

## Repository layout

```
vbmile-open/
├── configs/                  YAML configuration (base + data + model + inversion + ablations)
│   ├── base.yaml
│   ├── data/seismic{1d,2d}.yaml
│   ├── model/{base_vbmile,vbmile_1d,vbmile_2d}.yaml
│   ├── inversion/vbmile_1d.yaml|vbmile_2d.yaml
│   └── ablations/{no_prior,gaussian_prior,no_spatial_prior,no_kl_theta}.yaml
├── src/                      reusable library (registry driven)
│   ├── data/                 Seismic1DDataset / Seismic2DDataset, normalization helpers
│   ├── forward/              Aki-Richards + Ricker forward, Connolly EI / exact Zoeppritz,
│   │                         wavelets
│   ├── models/               GeoVBMILE (1-D and 2-D)
│   ├── priors/               GMMPriorFitter / SpatialPriorGenerator / GMMPrior
│   ├── losses/               VBELBOCore (negative ELBO)
│   ├── trainers/             VBEMLETrainer (single- and two-phase)
│   ├── metrics/              rmse / mae / r2 / pearson / ssim / psnr / ...
│   └── utils/                config, registry, run metadata, seeds, lr scheduler,
│                             fullcov (3x3 Cholesky operators), theory_contract
├── experiments/
│   ├── common.py             CLI, config resolution, run directory, dataset build
│   └── main/run_main.py      training + inference entry
├── tools/
│   ├── preprocess/           run_preprocess, add_noise, gmm_select, gmm_bootstrap_stability
│   ├── evaluate/             b9c_2d_windowed_infer (2-D sliding-window protocol)
│   └── data_check/           check_data (layout + statistics audit)
├── data/                     see data/README.md (splits committed, arrays rebuilt)
├── tests/                    test_vbmile.py, test_gmm.py, test_2d_windowed_infer.py, ...
└── outputs/                  run artifacts land here (git-ignored)
```

## Tests

```bash
pytest
```

covers the model forward/backward for both dimensions, the temperature and phase
mechanics, all well-NLL modes, the full-covariance posterior (zero-off-diagonal identity,
closed-form KL against `torch.distributions`, sampling covariance, expert-level well
likelihood, batch-invariant pixel masks, the theory contract and its checks), the 2-D
structure (merged output head, skip2 ordering, same-size tail, per-expert dilations,
deterministic padding), the deviatoric seismic likelihood, the stage-1 learning rate and
`checkpoint.save_last`, the GMM prior and fitter, the forward operators, the dataset
contract, the 2-D sliding-window protocol and its width floor, and the legacy 1-D
state-dict compatibility check.

Tests that need the rebuilt dataset skip automatically when `data/processed/` is absent
(see `tests/conftest.py`), so a fresh clone is green out of the box; with the data
prepared, the full suite (126 tests) runs and passes.

## Citation

If you use this code, please cite the Geo-VB-MILE paper and mention the Marmousi2 model
source proposed by Mejer et al.
