# Probabilistic Precipitation Downscaling using U-Net, WGAN-GP, and DDPM

**Journal:** Geoscientific Model Development (GMD)  
**Authors:** [Author 1], [Author 2], ...  
**Zenodo DOI:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)  
**Manuscript DOI:** [add once accepted]  
**License:** MIT

---

## Overview

This repository contains all code used to reproduce the results in the
manuscript. Three deep learning model families are evaluated for statistical
precipitation downscaling over three agricultural regions in the contiguous
United States (CONUS), at two spatial downscaling factors:

| Experiment | LR input | HR target | Factor |
|---|---|---|---|
| 8×  downscaling | 16×16 px | 128×128 px | 8×  |
| 16× downscaling | 8×8 px   | 128×128 px | 16× |

**Models trained:**
- **U-Net + WGAN-GP** — stochastic U-Net generator fine-tuned with Wasserstein GAN-GP (TensorFlow)
- **DDPM** — conditional Denoising Diffusion Probabilistic Model (PyTorch)

All models produce probabilistic ensemble outputs trained over 10 random seeds.

---

## Repository Structure

```
.
├── README.md                        ← You are here
├── environment_tf.yml               ← Conda env: TensorFlow (U-Net / WGAN + analysis)
├── environment_torch.yml            ← Conda env: PyTorch (DDPM)
│
├── preprocessing/                   ← Shared data pipeline (all models)
│   ├── 00_download_era5land.py
│   ├── 01_era5land_hourly_to_daily.py
│   ├── 02_crop_regions.py
│   ├── 03_filter_dry_images.py
│   ├── 04_plot_study_area.py
│   ├── 05_prepare_dataset.py        ← 8×  LR + splits
│   └── 05b_prepare_dataset_16x.py   ← 16× LR + splits
│
├── models/                          ← Model architecture definitions
│   ├── __init__.py
│   ├── models_wgan_8x.py            ← TF: U-Net, critic, WGAN (8×)
│   ├── models_wgan_16x.py           ← TF: U-Net, critic, WGAN (16×)
│   └── models_ddpm.py               ← PyTorch: DDPM U-Net + diffusion schedule
│
├── training/
│   ├── wgan_8x/                     ← TensorFlow — 8× experiment
│   │   ├── 06_train_unet.py
│   │   └── 07_train_wgan.py
│   ├── wgan_16x/                    ← TensorFlow — 16× experiment
│   │   ├── 08_train_unet_16x.py
│   │   └── 09_train_wgan_16x.py
│   ├── ddpm_8x/                     ← PyTorch — 8× experiment
│   │   └── 10_train_ddpm_8x.py
│   └── ddpm_16x/                    ← PyTorch — 16× experiment
│       └── 11_train_ddpm_16x.py
│
├── analysis/                        ← Post-training figures and metrics
│   ├── __init__.py
│   ├── plot_utils.py                ← Shared palette, colormap, helpers
│   ├── 12_plot_loss_curves.py       ← Fig: training/validation loss (all models)
│   ├── 13_plot_sample_grid.py       ← Fig 3/S3: visual comparison grid
│   ├── 14_plot_marginal_statistics.py ← Fig 4/S4: density scatter statistics
│   ├── 15_plot_spatial_correlation.py ← Fig 5/S8: lagged autocorrelation
│   ├── 16_plot_exceedance_and_qq.py ← Fig 7/S7: exceedance & Q–Q plot
│   ├── 17_plot_mass_conservation.py ← Fig 6: mass consistency at multiple scales
│   └── 18_plot_composite_metrics.py ← Fig 8: power spectrum, FSS, ROC, SSIM
│
└── results/                         ← Output figures (populated after running analysis)
    └── README.md
```

---

## Input Data

**ERA5-Land Hourly Reanalysis — Total Precipitation**

| Property | Value |
|---|---|
| Variable | `total_precipitation` (`tp`) |
| Period | January 1980 – December 2014 |
| Domain | CONUS (~25–50°N, ~125–67°W) |
| Resolution | 0.1° × 0.1° (native) |
| Source | [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu) |

> ERA5-Land data are freely available but require a free CDS account.
> Register at: https://cds.climate.copernicus.eu/user/register

---

## Study Regions

| Region | Lat range | Lon range | Tile size |
|---|---|---|---|
| Central Plains | 30.2–43.0°N | 105.4–92.6°W | 128×128 px |
| Northwest      | 36.6–49.4°N | 121.8–109.0°W | 128×128 px |
| Northeast      | 36.6–49.4°N | 90.4–77.6°W  | 128×128 px |

---

## Environment Setup

> **Why two environments?**  
> TensorFlow and PyTorch can coexist in the same conda environment, but
> GPU-accelerated TensorFlow and PyTorch use different CUDA runtime versions
> and can conflict. Keeping them separate avoids installation headaches,
> especially on HPC systems.  
> The preprocessing scripts (00–05b) use only NumPy/xarray and run in
> either environment.

### For U-Net / WGAN-GP (scripts 06–09)

```bash
conda env create -f environment_tf.yml
conda activate precip-downscaling-tf
```

### For DDPM (scripts 10–11)

```bash
conda env create -f environment_torch.yml
conda activate precip-downscaling-torch
```

### CDS API credentials (required for script 00)

1. Register at https://cds.climate.copernicus.eu
2. Go to your profile → API key
3. Create `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <your-personal-API-key>
   ```

---

## Reproducing the Results

### Step 0 — Configure paths

Every script has a `USER CONFIGURATION` block at the top. Edit the path
variables before running. Example:

```python
# ── USER CONFIGURATION ─────────────────────────────────────
HOURLY_DIR = "/your/path/ERA5_land/Hourly"
DAILY_DIR  = "/your/path/ERA5_land/Daily"
# ───────────────────────────────────────────────────────────
```

### Step 1 — Pre-processing  *(shared by all models)*

Run these once, in order, before any model training:

```bash
cd preprocessing/

python 00_download_era5land.py          # ~several days (CDS queue)
python 01_era5land_hourly_to_daily.py   # ~35 min for 1980–2014
python 02_crop_regions.py
python 03_filter_dry_images.py
python 04_plot_study_area.py            # generates Figure 1

# 8× dataset (16×16 LR)
python 05_prepare_dataset.py

# 16× dataset (8×8 LR)
python 05b_prepare_dataset_16x.py
```

> Script 00 skips months already downloaded — safe to interrupt and restart.
> Do **not** manually unzip downloaded files; script 01 handles extraction.

---

### Step 2a — U-Net + WGAN-GP  *(TensorFlow)*

Activate the TF environment first: `conda activate precip-downscaling-tf`

**8× experiment (16×16 → 128×128)**

```bash
cd training/wgan_8x/

# Run script 06 ten times, changing SEED = 1..10 in USER CONFIGURATION each time:
python 06_train_unet.py     # produces unet_generator_best_seed{N}.keras

# After all 10 U-Net checkpoints are ready:
python 07_train_wgan.py     # loops over all 10 seeds automatically
```

**16× experiment (8×8 → 128×128)**

```bash
cd training/wgan_16x/

python 08_train_unet_16x.py   # trains all 10 seeds in one run
python 09_train_wgan_16x.py   # WGAN-GP fine-tuning, all 10 seeds
```

---

### Step 3 — Analysis and figures  *(TensorFlow environment)*

All analysis scripts run in the TF environment and produce publication-ready
figures directly into `OUTPUT_DIR` (configured at the top of each script).

> **Important:** DDPM predictions must be generated from the saved
> `best_model.pth` checkpoints and saved as `seed_{N}/best_model_preds.npy`
> before running analysis scripts 13–18. The DDPM training scripts save
> model weights, not predictions, to keep storage manageable.

```bash
conda activate precip-downscaling-tf
cd analysis/

python 12_plot_loss_curves.py         # training/validation loss for all models
python 13_plot_sample_grid.py         # Figure 3/S3 — visual comparison grid
python 14_plot_marginal_statistics.py # Figure 4/S4 — statistical scatter panels
python 15_plot_spatial_correlation.py # Figure 5/S8 — lagged autocorrelation
python 16_plot_exceedance_and_qq.py   # Figure 7/S7 — exceedance & Q–Q
python 17_plot_mass_conservation.py   # Figure 6   — mass consistency
python 18_plot_composite_metrics.py   # Figure 8   — power, FSS, ROC, SSIM
```

### Step 2b — DDPM  *(PyTorch)*

Activate the PyTorch environment first: `conda activate precip-downscaling-torch`

```bash
# 8× experiment
cd training/ddpm_8x/
python 10_train_ddpm_8x.py    # trains all 10 seeds in one run

# 16× experiment
cd training/ddpm_16x/
python 11_train_ddpm_16x.py   # trains all 10 seeds in one run
```

> DDPM scripts read the same `.npz` split files as the TF scripts.
> No data conversion between frameworks is needed.

---

## Key Methodological Decisions

**ERA5-Land 00Z accumulation convention (script 01)**  
The 00:00 UTC step contains the full 24-hour precipitation total for the
*previous* calendar day. Daily totals are extracted from the 00Z step and
assigned to the preceding day via `daily_from_next00z()`.

**Wet-pixel filter (script 03)**  
Pixels below 1 mm/day are set to zero. Days with fewer than 164 wet pixels
(≈1% of the 128×128 domain) are discarded to remove near-empty training samples.

**Block-average downscaling (scripts 05 / 05b)**  
LR inputs are created by spatially averaging non-overlapping pixel blocks:
8×8 blocks for 8× (→ 16×16 LR), 16×16 blocks for 16× (→ 8×8 LR).

**U-Net architecture (models/models_wgan_8x.py, models_wgan_16x.py)**  
Stochastic U-Net with two encoder stages, a bottleneck, skip connections,
and three progressive transposed-convolution upsampling steps. A Gaussian
noise field is concatenated with the LR input to enable probabilistic output.
For the 16× model, LR and noise fields are bilinearly resized to 16×16 before
the encoder.

**WGAN-GP fine-tuning (script 07 / 09)**  
The pre-trained U-Net is fine-tuned against a conditional critic with
Wasserstein loss + gradient penalty (GP weight = 10, 3 critic steps per
generator step, Adam β₁=0, β₂=0.9).

**DDPM architecture (models/models_ddpm.py)**  
Conditional U-Net noise predictor with Feature-wise Linear Modulation (FiLM)
for diffusion timestep conditioning and sinusoidal time embeddings. Cosine
noise schedule with T=100 timesteps. The LR field is progressively upsampled
inside the model via bilinear Upsample layers before concatenation with the
noisy HR field.

**Ensemble strategy**  
All three model families are trained with 10 independent random seeds
(seeds 1–10) to produce probabilistic ensembles for uncertainty quantification.

---

## Software Versions

| Package | Version | Used by |
|---|---|---|
| Python | 3.10 | all |
| numpy | ≥1.24 | all |
| xarray | ≥2023.1 | preprocessing |
| netCDF4 | ≥1.6 | preprocessing |
| matplotlib | ≥3.7 | all |
| cartopy | ≥0.21 | preprocessing |
| pandas | ≥2.0 | WGAN 16× |
| cdsapi | ≥0.6 | download |
| tensorflow | ≥2.12 | U-Net / WGAN |
| torch | ≥2.0 | DDPM |

---

## License

Code released under the **MIT License** — see [LICENSE](LICENSE).

ERA5-Land data are provided by ECMWF/Copernicus under the
[Copernicus Licence](https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf)
and are not redistributed in this repository.

---

## Citation

If you use this code, please cite both the manuscript and this Zenodo archive:

**Manuscript:**
> [Author 1], [Author 2], et al. ([YEAR]). *[Manuscript Title]*.
> Geoscientific Model Development. https://doi.org/[MANUSCRIPT DOI]

**Code archive:**
> [Author 1], [Author 2], et al. ([YEAR]).
> *Code for: [Manuscript Title]* [Software].
> Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

---

## Contact

For questions about this code, contact [Author Name] at [email@institution.edu].  
For issues with specific scripts, please open a GitHub issue with the script
name and the full error message.
