# Data preparation

The repository is committed **without** the large data files, so that cloning stays
fast and stays well inside GitHub limits. Everything that is missing can be rebuilt
from four raw inputs plus the splits, which are committed.

## What you need to supply

Place the following files under `data/raw/` (they are *not* redistributed here, see
"Provenance" below):

```
data/
├── raw/
│   ├── observations/
│   │   ├── elastic_impedance_results.mat   # (1200, 1360, 7) float64, 7-angle stack incl. 0 deg
│   │   ├── data2.npy                        # dict: parameter (1360,3,1200)
│   └── priors/
│       ├── gmm_priors.npz                    # {"means": (K,3), "variances"|"covariances": ...}
│       └── spatial_prior.pt                  # (H, W, K) or (K, H, W) torch tensor
├── interim/      # rebuilt: norm_stats.json
├── processed/    # rebuilt: marmousi2_{1d,2d}/...
└── splits/       # COMMITTED: marmousi2_1d/{train,val,test}.txt, marmousi2_2d/...
```

## One command to rebuild

```bash
# 1-D pipeline (low-frequency sigma = 30, 12 train wells, 7 val wells)
python -m tools.preprocess.run_preprocess --config configs/data/seismic1d.yaml

# 2-D pipeline (low-frequency sigma = 100, 12 train wells, 3 val wells)
python -m tools.preprocess.run_preprocess --config configs/data/seismic2d.yaml
```

Each run writes, atomically:

| output | shape | meaning |
| --- | --- | --- |
| `data/processed/<name>/seismic6.npy` | `(6, 1200, 1360)` | observed angle gathers, 0 deg removed (5-30 deg) |
| `data/processed/<name>/target.npy` | `(3, 1200, 1360)` | ground-truth Vp / Vs / density |
| `data/processed/<name>/target_low_s*.npy` | `(3, 1200, 1360)` | Gaussian low-frequency model (sigma = 30 for 1D, 100 for 2D) |
| `data/processed/<name>/spatial_prior.npy` | `(K, 1200, 1360)` | spatial prior probability, channel-first |
| `data/processed/<name>/gmm_prior.npz` | keys `means` `(K,3)`, `covariances` `(K,3,3)`, `weights` `(K,)` | GMM prior in the normalized space (full covariance) |
| `data/processed/<name>/MANIFEST.json` | — | per-file shape / dtype / md5 provenance record |
| `data/interim/norm_stats.json` | — | per-channel z-score statistics for the 12 train traces |

### Reproducibility details

* `seismic6.npy` = `scipy.io.loadmat("elastic_impedance_results.mat")["seismic_data"]`,
  transposed to `(angle, depth, trace)` with the 0 deg channel dropped.
* `target.npy` = `numpy.load("data2.npy", allow_pickle=True).item()["parameter"]`
  transposed from `(trace, attribute, depth)` to `(attribute, depth, trace)`.
* Low-frequency model: `scipy.ndimage.gaussian_filter(target, sigma=..., axes=(1, 2))`.
* GMM prior: fitted on the **normalized** train-subset pixels by
  `src.priors.gmm.GMMPriorFitter(K=3, seed=42)` — sklearn `GaussianMixture`
  (`covariance_type="full"`, `reg_covar=1e-6`, `max_iter=500`) followed by the
  legacy shrinkage recipe `(1 - 0.1) * Sigma_k + 0.1 * Sigma_global + 1e-4 * I`.
* Normalization statistics: train traces are `linspace(20, 1300, 12)`; mean/std are
  computed over `(depth, trace)` on that subset with `std + 1e-8`.

Verify an existing `processed/` tree with hashes and shapes:

```bash
python -m tools.preprocess.run_preprocess --config configs/data/seismic1d.yaml --check
```

## SNR (noise robustness) variants

```bash
# 1-D observations at 2 / 5 / 10 / 15 dB (SNR_dB = 10*log10(P_signal / P_noise))
python -m tools.preprocess.add_noise --config configs/data/seismic1d.yaml --snr 2 5 10 15 --seed 0

# 2-D observations
python -m tools.preprocess.add_noise --config configs/data/seismic2d.yaml --snr 2 5 10 15 --seed 0
```

This writes `seismic6_snr{2,5,10,15}.npy` next to `seismic6.npy` and leaves every other
file (target, low-frequency model, priors, normalization statistics) untouched, so a
noisy run is selected purely by configuration:

```bash
python -m experiments.main.run_main -c configs/inversion/vbmile_1d.yaml --name m1_snr2 \
    --set data.files.seismic=seismic6_snr2.npy
```

## Provenance

The raw arrays are derived from the **Marmousi2** elastic model (Mejer et al.) through an
Aki-Richards angle-gather synthesis and the elastic-impedance branch of the original
notebook experiments. `data/MANIFEST.json` in the original research workspace records the
md5 hashes and the notebook cells that produced each file. Those raw arrays are not
reproducible from public sources alone, which is why they must be supplied by the user.
