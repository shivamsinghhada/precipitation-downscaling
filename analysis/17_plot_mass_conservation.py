"""
Script:      analysis/17_plot_mass_conservation.py
Description: Mass (depth) consistency at multiple spatial aggregation scales
             — Figure 6.

             For each of 8 block-averaging scales (1×1 to 128×128 pixels),
             predicted and observed domain-mean precipitation depths are
             compared in a scatter plot.  Pearson r and Bias are annotated
             in each panel.

             A 2×4 subplot grid is produced with a shared legend.

Inputs:      dataset_splits.npz   (for Xtest, Ytest)
             U-Net and WGAN model checkpoints (one seed, seed=1 by default)
             DDPM prediction .npy file (one seed)
Outputs:     mass_consistency.png  in OUTPUT_DIR

Usage:       python analysis/17_plot_mass_conservation.py

Requirements: numpy, tensorflow, matplotlib, scipy  (see environment_tf.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
DATASET_PATH   = "/path/to/ERA5_land/Dataset/dataset_splits.npz"

UNET_PATH      = "/path/to/unet_runs/unet_generator_best_seed1.keras"
WGAN_PATH      = "/path/to/unet_runs/WGANs/gen_final_seed1.keras"
DDPM_PRED_PATH = "/path/to/checkpoints/ddpm_8x/seed_1/best_model_preds.npy"

OUTPUT_DIR     = "/path/to/results"
NOISE_STD      = 1.0
SCALES         = [1, 2, 4, 8, 16, 32, 64, 128]   # block-averaging side lengths
# ────────────────────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.plot_utils import COLORS, squeeze_hw

os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA_SCATTER = 0.65
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def aggregate_depth(arr: np.ndarray, k: int) -> np.ndarray:
    """
    Block-average each image in arr to (H//k, W//k) and return flattened values.

    Parameters
    ----------
    arr : (N, H, W) float array
    k   : block size (number of pixels per side)

    Returns
    -------
    1-D float array of length N × (H//k) × (W//k)
    """
    N, H, W = arr.shape
    Hk, Wk  = H // k, W // k
    arr      = arr[:, :Hk * k, :Wk * k]
    blocked  = arr.reshape(N, Hk, k, Wk, k).mean(axis=(2, 4))
    return blocked.ravel()


def compute_metrics(truth: np.ndarray, pred: np.ndarray):
    """Return (bias, pearson_r) for paired 1-D arrays."""
    bias = float(np.mean(pred - truth))
    r, _ = pearsonr(truth, pred)
    return bias, float(r)


# ── Load data and models ──────────────────────────────────────────────────────

print(f"Loading dataset from: {DATASET_PATH}")
splits = np.load(DATASET_PATH)
Xtest  = splits["Xtest"][..., np.newaxis].astype(np.float32)
Ytest  = squeeze_hw(splits["Ytest"])

noise  = np.random.default_rng(42).standard_normal(Xtest.shape).astype(np.float32)

print("Loading U-Net...")
unet_model = tf.keras.models.load_model(UNET_PATH, compile=False)
unet_preds = squeeze_hw(unet_model.predict([Xtest, noise], verbose=0))

print("Loading WGAN...")
wgan_model = tf.keras.models.load_model(WGAN_PATH, compile=False)
wgan_preds = squeeze_hw(wgan_model.predict([Xtest, noise], verbose=0))

print("Loading DDPM predictions...")
if not os.path.exists(DDPM_PRED_PATH):
    raise FileNotFoundError(
        f"DDPM predictions not found: {DDPM_PRED_PATH}\n"
        "Generate predictions from saved best_model.pth first."
    )
ddpm_preds = squeeze_hw(np.load(DDPM_PRED_PATH))

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle("Mass (Depth) Consistency", fontsize=20, y=0.97)

for idx, k in enumerate(SCALES):
    row, col = idx // 4, idx % 4
    ax       = axes[row, col]

    truth = aggregate_depth(Ytest,      k)
    u     = aggregate_depth(unet_preds, k)
    w     = aggregate_depth(wgan_preds, k)
    d     = aggregate_depth(ddpm_preds, k)

    ubias, ur = compute_metrics(truth, u)
    wbias, wr = compute_metrics(truth, w)
    dbias, dr = compute_metrics(truth, d)

    max_lim = max(truth.max(), u.max(), w.max(), d.max())

    ax.scatter(truth, u, s=5, alpha=ALPHA_SCATTER,
               color=COLORS["U-Net"], edgecolors="none")
    ax.scatter(truth, w, s=5, alpha=ALPHA_SCATTER,
               color=COLORS["WGAN"],  edgecolors="none")
    ax.scatter(truth, d, s=5, alpha=ALPHA_SCATTER,
               color=COLORS["DDPM"],  edgecolors="none")

    ax.plot([0, max_lim], [0, max_lim], "k--", linewidth=1)
    ax.set_xlim(0, max_lim)
    ax.set_ylim(0, max_lim)
    ax.set_aspect("equal")
    ax.set_title(f"Aggregation scale: {k}×{k} px")
    ax.grid(alpha=0.3)

    stats_text = (
        f"U-Net: Bias={ubias:.3f}, r={ur:.3f}\n"
        f"WGAN:  Bias={wbias:.3f}, r={wr:.3f}\n"
        f"DDPM:  Bias={dbias:.3f}, r={dr:.3f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, va="top",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7))

legend_elements = [
    plt.Line2D([0],[0], marker="o", color=COLORS["U-Net"],
               alpha=ALPHA_SCATTER, markersize=10, linestyle="None", label="U-Net"),
    plt.Line2D([0],[0], marker="o", color=COLORS["WGAN"],
               alpha=ALPHA_SCATTER, markersize=10, linestyle="None", label="WGAN"),
    plt.Line2D([0],[0], marker="o", color=COLORS["DDPM"],
               alpha=ALPHA_SCATTER, markersize=10, linestyle="None", label="DDPM"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.02))

plt.subplots_adjust(top=0.92, bottom=0.10, hspace=0.3, wspace=0.25)

out_path = os.path.join(OUTPUT_DIR, "mass_consistency.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
