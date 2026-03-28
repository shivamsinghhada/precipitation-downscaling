"""
Script:      analysis/14_plot_marginal_statistics.py
Description: Marginal statistics scatter plots — Figure 4 / Supplementary Figure S4.

             Produces a 5-row × 3-column density scatter figure:
               Columns : U-Net | WGAN | DDPM
               Rows    : Dry % | Mean | Variance | Skewness | Kurtosis

             Statistics are computed per test image and compared against
             observed values.  All 10 ensemble seeds are pooled to form a
             single density scatter per model family, giving N × 10 points
             per panel.  Axis limits are shared across columns within each row.

             The Bias and RMSE of each model's statistics against observations
             are annotated in each panel.

Inputs:      dataset_splits.npz   (for Xtest, Ytest)
             U-Net and WGAN model checkpoints (all 10 seeds)
             DDPM prediction .npy files for all 10 seeds
Outputs:     stats_scatter.png  in OUTPUT_DIR

Usage:       python analysis/14_plot_marginal_statistics.py

Requirements: numpy, tensorflow, matplotlib, scipy  (see environment_tf.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
DATASET_PATH   = "/path/to/ERA5_land/Dataset/dataset_splits.npz"

UNET_MODEL_DIR = "/path/to/unet_runs"
UNET_GLOB      = "unet_generator_best_seed*.keras"

WGAN_MODEL_DIR = "/path/to/unet_runs/WGANs"
WGAN_GLOB      = "gen_final_seed*.keras"

DDPM_PRED_DIR  = "/path/to/checkpoints/ddpm_8x"
DDPM_SEEDS     = list(range(1, 11))   # seeds 1–10 (matches training scripts)
# NOTE: DDPM prediction files are expected at:
#   {DDPM_PRED_DIR}/seed_{N}/best_model_preds.npy
#   (shape: N_test × 128 × 128)
# Generate these from the saved best_model.pth before running this script.

OUTPUT_DIR     = "/path/to/results"
N_SAMPLES      = None   # None = use all test samples; int to limit (e.g. 500 for testing)
NOISE_STD      = 1.0
WET_THRESHOLD  = 1.0    # mm/day — pixels below this treated as dry
# ────────────────────────────────────────────────────────────────────────────

import os
import glob
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import moment, gaussian_kde

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.plot_utils import COLORS, squeeze_hw

os.makedirs(OUTPUT_DIR, exist_ok=True)

GRAY_LINE     = "0.25"
GRAY_TEXT     = "0.05"
GRAY_BOX_EDGE = "0.55"


# ── Statistic functions ───────────────────────────────────────────────────────

def prob_dry_percent(img: np.ndarray, thr: float = 1.0) -> float:
    return 100.0 * float(np.mean(img <= thr))


def wet_moments(img: np.ndarray, thr: float = 1.0):
    """Return (mean, variance, skewness, kurtosis) for wet pixels only."""
    vals = img[np.isfinite(img)]
    vals = vals[vals > thr]
    if vals.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(vals)),
        float(moment(vals, moment=2)),
        float(moment(vals, moment=3)),
        float(moment(vals, moment=4)),
    )


def compute_stats_array(images: np.ndarray, thr: float = 1.0) -> dict:
    """Compute per-image statistics for an (N, H, W) array."""
    N = images.shape[0]
    dry  = np.zeros(N, dtype=np.float64)
    mean = np.zeros(N, dtype=np.float64)
    m2   = np.zeros(N, dtype=np.float64)
    m3   = np.zeros(N, dtype=np.float64)
    m4   = np.zeros(N, dtype=np.float64)
    for i in range(N):
        dry[i] = prob_dry_percent(images[i], thr)
        mean[i], m2[i], m3[i], m4[i] = wet_moments(images[i], thr)
    return {"dry_pct": dry, "mean": mean, "m2": m2, "m3": m3, "m4": m4}


def build_pairs(obs_stats: dict, preds_dict: dict) -> dict:
    """Pool all seeds: return {key: (obs_concat, pred_concat)}."""
    result = {k: ([], []) for k in obs_stats}
    for seed in sorted(preds_dict):
        pred_stats = compute_stats_array(preds_dict[seed], WET_THRESHOLD)
        for k in obs_stats:
            result[k][0].append(obs_stats[k])
            result[k][1].append(pred_stats[k])
    return {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in result.items()}


# ── Inference helpers ─────────────────────────────────────────────────────────

def infer_tf_models(model_dir: str, glob_pattern: str, X: np.ndarray,
                    noise: np.ndarray) -> dict:
    """Load all matching TF models and run inference. Returns {seed: preds}."""
    paths = sorted(glob.glob(os.path.join(model_dir, glob_pattern)))
    if not paths:
        raise RuntimeError(f"No models found: {model_dir}/{glob_pattern}")
    out = {}
    for p in paths:
        seed = int(p.split("seed")[-1].split(".")[0])
        model = tf.keras.models.load_model(p, compile=False)
        preds = squeeze_hw(model.predict([X, noise], verbose=0))
        out[seed] = preds
        print(f"  Inference: seed {seed} → {preds.shape}")
    return out


def load_ddpm_preds(pred_dir: str, seeds: list, n_samples=None) -> dict:
    """Load pre-generated DDPM predictions. Returns {seed: preds}."""
    out = {}
    for s in seeds:
        path = os.path.join(pred_dir, f"seed_{s}", "best_model_preds.npy")
        if not os.path.exists(path):
            print(f"  WARNING: DDPM prediction not found: {path}")
            continue
        arr = squeeze_hw(np.load(path))
        if n_samples is not None:
            arr = arr[:n_samples]
        out[s] = arr
        print(f"  Loaded DDPM seed {s}: {arr.shape}")
    return out


# ── Scatter density panel ─────────────────────────────────────────────────────

def scatter_density(ax, x_obs: np.ndarray, y_pred: np.ndarray):
    """Density-coloured scatter with 1:1 line and Bias/RMSE annotation."""
    x = x_obs.ravel(); y = y_pred.ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size == 0:
        return
    try:
        z   = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
        idx = z.argsort()
        ax.scatter(x[idx], y[idx], c=z[idx], cmap="viridis",
                   s=8, alpha=0.85, linewidths=0)
    except np.linalg.LinAlgError:
        ax.scatter(x, y, c="k", s=8, alpha=0.45, linewidths=0)

    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color=GRAY_LINE)

    bias = float(np.mean(y - x))
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    ax.text(0.03, 0.97,
            f"Bias: {bias:.2f}\nRMSE: {rmse:.2f}",
            transform=ax.transAxes,
            va="top", ha="left", fontsize=8, color=GRAY_TEXT,
            bbox=dict(boxstyle="round", facecolor="none",
                      edgecolor=GRAY_BOX_EDGE, linewidth=0.8))


# ── Shared row limits ─────────────────────────────────────────────────────────

def row_limits(pairs_list: list, key: str, pad_frac: float = 0.03):
    all_vals = np.concatenate(
        [np.concatenate(p[key]) for p in pairs_list]
    )
    m = np.isfinite(all_vals)
    lo, hi = float(all_vals[m].min()), float(all_vals[m].max())
    if lo == hi:
        lo -= 1.0; hi += 1.0
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading dataset from: {DATASET_PATH}")
    splits = np.load(DATASET_PATH)
    Xtest  = splits["Xtest"][..., np.newaxis].astype(np.float32)
    Ytest  = squeeze_hw(splits["Ytest"])

    if N_SAMPLES is not None:
        Xtest = Xtest[:N_SAMPLES]
        Ytest = Ytest[:N_SAMPLES]

    noise = np.random.default_rng(0).standard_normal(Xtest.shape).astype(np.float32)

    print("\nRunning U-Net inference...")
    preds_unet = infer_tf_models(UNET_MODEL_DIR, UNET_GLOB, Xtest, noise)

    print("\nRunning WGAN inference...")
    preds_wgan = infer_tf_models(WGAN_MODEL_DIR, WGAN_GLOB, Xtest, noise)

    print("\nLoading DDPM predictions...")
    preds_ddpm = load_ddpm_preds(DDPM_PRED_DIR, DDPM_SEEDS, N_SAMPLES)
    if not preds_ddpm:
        raise RuntimeError("No DDPM predictions loaded.")

    print("\nComputing statistics...")
    obs_stats  = compute_stats_array(Ytest, WET_THRESHOLD)
    pairs_unet = build_pairs(obs_stats, preds_unet)
    pairs_wgan = build_pairs(obs_stats, preds_wgan)
    pairs_ddpm = build_pairs(obs_stats, preds_ddpm)
    pairs_all  = [pairs_unet, pairs_wgan, pairs_ddpm]

    rows = [
        ("dry_pct", "Probability of dry (%)"),
        ("mean",    "Mean (mm/day)"),
        ("m2",      "Variance"),
        ("m3",      "Skewness"),
        ("m4",      "Kurtosis"),
    ]
    cols = ["U-Net", "WGAN", "DDPM"]

    # Shared row limits
    lims = {}
    for key, _ in rows:
        lims[key] = (0.0, 100.0) if key == "dry_pct" else row_limits(pairs_all, key)

    plt.rcParams.update({"axes.grid": False, "font.size": 12})
    fig, axes = plt.subplots(5, 3, figsize=(10, 12), constrained_layout=True)

    for j, title in enumerate(cols):
        axes[0, j].set_title(title, fontsize=12, fontweight="bold", color=GRAY_TEXT)

    for i, (key, row_name) in enumerate(rows):
        for j, pairs in enumerate(pairs_all):
            ax = axes[i, j]
            scatter_density(ax, *pairs[key])
            ax.set_xlim(lims[key])
            ax.set_ylim(lims[key])
            ax.tick_params(labelsize=9, colors=GRAY_TEXT)
            for spine in ax.spines.values():
                spine.set_color("0.75")
            if j == 0:
                ax.set_ylabel("Predicted", fontsize=10, color=GRAY_TEXT)
                ax.annotate(row_name,
                            xy=(-0.40, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", ha="center",
                            fontsize=11, fontweight="bold", color=GRAY_TEXT)
            if i == len(rows) - 1:
                ax.set_xlabel("Observed", fontsize=10, color=GRAY_TEXT)

    out_path = os.path.join(OUTPUT_DIR, "stats_scatter.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
