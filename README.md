# Variational-Bayesian-Mixture-of-Expert

This repository provides a **simplified 1D implementation** of the method described in the paper:

> **"Geology-Guided Variational Bayesian Mixture of Experts for AVA Inversion"**  
> *(Full paper uses 2D spatial priors and full‑covariance GMM; this code is a 1D, diagonal‑covariance version for clarity and fast prototyping.)*

## Key differences from the paper
- **1D** instead of 2D (each CDP trace processed independently)
- **Diagonal Gaussian** instead of full covariance for expert proposals

Despite these simplifications, the core variational Bayesian mixture‑of‑experts framework remains the same, and the code can be easily extended to the full 2D version.

---

## Requirements

- Python 3.9+, PyTorch 1.10+, NumPy, SciPy, Matplotlib, scikit‑image, scikit‑learn

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

Place the following files in `./data/` (paths can be changed in `config.py`):

| File                          | Description                                 |
|-------------------------------|---------------------------------------------|
| `Vp.npy`, `Vs.npy`, `Den.npy` | True elastic models (2D sections)           |
| `elastic_impedance_results.mat` | Synthetic angle gathers (6 angles)        |
| `gmm_priors3.npz`             | GMM prior: `means`, `variances` (K×3)       |
| `spatial_prior3.pt`           | Spatial prior maps (K×H×W)                  |


## Repository Structure
```
.
├── config.py
├── train.py
├── requirements.txt
├── README.md
├── data/ # your data files
├── models/
│ ├── VBMILE1d.py # Main model (Geo_VBMILE_1D)
│ ├── VBlosses.py
│ └── forward.py
├── data/ (package)
│ └── dataset.py
└── utils/
├── lr_scheduler.py
├── metrics.py
└── plotting.py
```

## License
MIT License – see `LICENSE`.
