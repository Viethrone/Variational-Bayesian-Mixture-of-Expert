# Variational-Bayesian-Mixture-of-Expert

This repository provides a **simplified 1D implementation** of the method described in the paper:

> **"Geology-Guided Variational Bayesian Mixture of Experts for AVA Inversion"**  
> *(Full paper uses 2D spatial priors and full‑covariance GMM; this code is a 1D, diagonal‑covariance version for clarity and fast prototyping.)*

## Key differences from the paper
- **1D** instead of 2D (each CDP trace processed independently)
- **Diagonal Gaussian** instead of full covariance for expert proposals
- No spatial correlation between neighbouring traces (except via provided spatial prior maps)

Despite these simplifications, the core variational Bayesian mixture‑of‑experts framework remains the same, and the code can be easily extended to the full 2D version.

---

## Requirements

- Python 3.9+, PyTorch 1.10+, NumPy, SciPy, Matplotlib, scikit‑image, scikit‑learn

Install dependencies:
```bash
pip install -r requirements.txt
