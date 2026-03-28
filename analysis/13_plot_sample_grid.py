"""
Script:      analysis/13_plot_sample_grid.py
Description: Visual comparison grid — Figure 3 / Supplementary Figure S3.

             Produces a 5-row × N_COLS figure showing, for each selected
             test sample:
               Row 1 — Low-resolution input (LR)
               Row 2 — High-resolution observation (truth)
               Row 3 — U-Net prediction
               Row 4 — WGAN prediction
               Row 5 — DDPM prediction

             A shared colorbar is placed on the right.  The LR row uses its
             own colour scale; all HR rows share a common scale derived from
             the 0.5th–99.5th percentile of the displayed samples.

Inputs:      dataset_splits.npz   (for Xtest, Ytest)
             U-Net and WGAN model checkpoints (one seed each)
             DDPM prediction .npy for one seed
Outputs:     comparison_grid.png  in OUTPUT_DIR

Usage:       python analysis/13_plot_sample_grid.py

Requirements: numpy, tensorflow, matplotlib  (see environment_tf.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
DATASET_PATH  = "/path/to/ERA5_land/Dataset/dataset_splits.npz"

UNET_PATH     = "/path/to/unet_runs/unet_generator_best_seed1.keras"
WGAN_PATH     = "/path/to/unet_runs/WGANs/gen_final_seed1.keras"
DDPM_PRED_PATH= "/path/to/checkpoints/ddpm_8x/seed_1/best_model_preds.npy"
# NOTE: DDPM_PRED_PATH should point to pre-generated predictions (.npy file of shape
#       (N_test, 128, 128)).  Run inference separately using the saved best_model.pth.

OUTPUT_DIR    = "/path/to/results"
N_COLS        = 6      # number of test samples shown as columns
NOISE_STD     = 1.0    # noise standard deviation for U-Net / WGAN inference
RANDOM_SEED   = 42     # for reproducible sample and noise selection
# ────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.plot_utils import custom_cmap, squeeze_hw

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load test data ────────────────────────────────────────────────────────────
print(f"Loading dataset from: {DATASET_PATH}")
splits = np.load(DATASET_PATH)
Xtest  = splits["Xtest"][..., np.newaxis].astype(np.float32)   # (N, 16, 16, 1)
Ytest  = squeeze_hw(splits["Ytest"])                            # (N, 128, 128)

print(f"  Xtest: {Xtest.shape}  Ytest: {Ytest.shape}")

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading U-Net...")
unet_model = tf.keras.models.load_model(UNET_PATH, compile=False)

print("Loading WGAN generator...")
wgan_model = tf.keras.models.load_model(WGAN_PATH, compile=False)

print("Loading DDPM predictions...")
if not os.path.exists(DDPM_PRED_PATH):
    raise FileNotFoundError(
        f"DDPM prediction file not found: {DDPM_PRED_PATH}\n"
        "Generate predictions from the saved DDPM checkpoint before running this script."
    )
diff_preds = squeeze_hw(np.load(DDPM_PRED_PATH))    # (N, 128, 128)

# ── Inference: U-Net and WGAN ─────────────────────────────────────────────────
rng        = np.random.default_rng(seed=RANDOM_SEED)
noise_test = rng.standard_normal(Xtest.shape).astype(np.float32)

print("Running U-Net inference...")
unet_preds = squeeze_hw(unet_model.predict([Xtest, noise_test], verbose=0))

print("Running WGAN inference...")
wgan_preds = squeeze_hw(wgan_model.predict([Xtest, noise_test], verbose=0))

# ── Select samples ────────────────────────────────────────────────────────────
# Evenly spaced across the test set for a representative overview
samples_idx = np.linspace(0, len(Xtest) - 1, N_COLS, dtype=int)

Xtest_sq    = np.squeeze(Xtest)          # (N, 16, 16)
X_samples   = Xtest_sq[samples_idx]     # (N_COLS, 16, 16)
Y_samples   = Ytest[samples_idx]
U_samples   = unet_preds[samples_idx]
W_samples   = wgan_preds[samples_idx]
D_samples   = diff_preds[samples_idx]

# ── Shared colour scale for all HR rows ───────────────────────────────────────
all_hr = np.concatenate([
    Y_samples.ravel(), U_samples.ravel(), W_samples.ravel(), D_samples.ravel()
])
vmin, vmax = float(np.percentile(all_hr, 0.5)), float(np.percentile(all_hr, 99.5))

# ── Build figure ──────────────────────────────────────────────────────────────
row_labels = [
    "Low-Res Input",
    "Observed (HR)",
    "U-Net",
    "WGAN",
    "DDPM",
]

fig, axes = plt.subplots(5, N_COLS, figsize=(2.5 * N_COLS, 14))

for col in range(N_COLS):
    imgs = [X_samples[col], Y_samples[col], U_samples[col],
            W_samples[col], D_samples[col]]

    for row in range(5):
        ax  = axes[row, col]
        img = imgs[row]

        # LR row has its own colour scale; all HR rows share vmin/vmax
        kwargs = dict(cmap=custom_cmap)
        if row > 0:
            kwargs.update(vmin=vmin, vmax=vmax)

        ax.imshow(img, **kwargs)
        ax.axis("off")

        if col == 0:
            ax.set_ylabel(
                row_labels[row],
                fontsize=13,
                rotation=0,
                labelpad=85,
                va="center",
                ha="right",
            )

plt.subplots_adjust(wspace=0.02, hspace=0.02)

# ── Shared colorbar ───────────────────────────────────────────────────────────
cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
norm    = mcolors.Normalize(vmin=vmin, vmax=vmax)
cb      = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap), cax=cbar_ax
)
cb.set_label("Precipitation (mm/day)", fontsize=13)
cb.ax.tick_params(labelsize=11)

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "comparison_grid.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
