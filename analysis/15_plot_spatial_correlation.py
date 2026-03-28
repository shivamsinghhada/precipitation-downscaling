"""
Script:      analysis/15_plot_spatial_correlation.py
Description: Lagged spatial autocorrelation figures — Figure 5 and
             Supplementary Figure S8.

             Figure 5: Combined boxplot (one box per model per lag) showing
             horizontal and vertical spatial correlations across all 10 seeds
             and all test images for U-Net, WGAN, DDPM, and Observed.

             Figure S8: Per-model 2×4 boxplot panels showing each seed
             individually (Seed 1 … Seed 10) against the observed distribution,
             one panel per spatial lag.  Produced for U-Net, WGAN, and DDPM
             separately (3 output files).

Inputs:      dataset_splits.npz   (for Xtest, Ytest)
             U-Net and WGAN model checkpoints (all 10 seeds)
             DDPM prediction .npy files for all 10 seeds
Outputs:     spatial_corr_combined.png      — Figure 5
             spatial_corr_unet_seeds.png    — Figure S8 (U-Net)
             spatial_corr_wgan_seeds.png    — Figure S8 (WGAN)
             spatial_corr_ddpm_seeds.png    — Figure S8 (DDPM)
             (all saved to OUTPUT_DIR)

Usage:       python analysis/15_plot_spatial_correlation.py

Requirements: numpy, tensorflow, pandas, seaborn, matplotlib  (see environment_tf.yml)
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
DDPM_SEEDS     = list(range(1, 11))   # seeds 1–10

OUTPUT_DIR     = "/path/to/results"
MAX_LAG        = 8
NOISE_STD      = 1.0
# ────────────────────────────────────────────────────────────────────────────

import os
import glob
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.plot_utils import COLORS, squeeze_hw

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Correlation function ──────────────────────────────────────────────────────

def spatial_correlation_df(images: np.ndarray, max_lag: int,
                            model_name: str, seed=None) -> pd.DataFrame:
    """
    Compute per-image horizontal and vertical spatial correlations up to max_lag.

    Parameters
    ----------
    images     : (N, H, W) float array
    max_lag    : int, maximum pixel lag
    model_name : str, label for the 'Model' column
    seed       : optional seed identifier

    Returns
    -------
    pd.DataFrame with columns: Model, Lag, Correlation, Direction, Seed
    """
    images = np.squeeze(images)
    N, H, W = images.shape
    records = []
    for img in images:
        for lag in range(1, max_lag + 1):
            # Horizontal
            if lag < W:
                x1, x2 = img[:, :-lag].ravel(), img[:, lag:].ravel()
                if np.std(x1) > 1e-6 and np.std(x2) > 1e-6:
                    c = np.corrcoef(x1, x2)[0, 1]
                    if np.isfinite(c):
                        records.append({"Model": model_name, "Lag": lag,
                                        "Correlation": c, "Direction": "Horizontal",
                                        "Seed": seed})
            # Vertical
            if lag < H:
                y1, y2 = img[:-lag, :].ravel(), img[lag:, :].ravel()
                if np.std(y1) > 1e-6 and np.std(y2) > 1e-6:
                    c = np.corrcoef(y1, y2)[0, 1]
                    if np.isfinite(c):
                        records.append({"Model": model_name, "Lag": lag,
                                        "Correlation": c, "Direction": "Vertical",
                                        "Seed": seed})
    return pd.DataFrame.from_records(records)


# ── Inference helpers ─────────────────────────────────────────────────────────

def infer_tf_models(model_dir, glob_pattern, X, noise):
    paths = sorted(glob.glob(os.path.join(model_dir, glob_pattern)))
    if not paths:
        raise RuntimeError(f"No models found: {model_dir}/{glob_pattern}")
    out = {}
    for p in paths:
        seed  = int(p.split("seed")[-1].split(".")[0])
        model = tf.keras.models.load_model(p, compile=False)
        preds = squeeze_hw(model.predict([X, noise], verbose=0))
        out[seed] = preds
        print(f"  Inference: seed {seed} → {preds.shape}")
    return out


def load_ddpm_preds(pred_dir, seeds):
    out = {}
    for s in seeds:
        path = os.path.join(pred_dir, f"seed_{s}", "best_model_preds.npy")
        if not os.path.exists(path):
            print(f"  WARNING: DDPM prediction not found: {path}")
            continue
        out[s] = squeeze_hw(np.load(path))
        print(f"  Loaded DDPM seed {s}: {out[s].shape}")
    return out


# ── Figure 5: combined across all seeds ──────────────────────────────────────

def plot_figure5(df_combined: pd.DataFrame, out_dir: str):
    """Two-panel boxplot: (a) Horizontal, (b) Vertical — all seeds pooled."""
    palette = {k: COLORS[k] for k in ["Observed", "U-Net", "WGAN", "DDPM"]}
    sns.set_theme(style="white", font_scale=1.3)

    for direction, label in [("Horizontal", "(a)"), ("Vertical", "(b)")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=df_combined[df_combined["Direction"] == direction],
            x="Lag", y="Correlation", hue="Model",
            palette=palette, showfliers=False,
            saturation=1, boxprops=dict(alpha=0.65), ax=ax,
        )
        ax.set_xlabel("Lag (pixels)", fontsize=13)
        ax.set_ylabel("Spatial Correlation", fontsize=13)
        ax.set_ylim(0.0, 1.0)
        ax.grid(False)
        ax.text(0.01, 0.97, label, transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="top")
        leg = ax.get_legend()
        if leg:
            leg.set_title("")

        fname = "spatial_corr_horizontal.png" if direction == "Horizontal" \
                else "spatial_corr_vertical.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")


# ── Figure S8: per-seed panels for one model ─────────────────────────────────

def plot_per_seed(df_obs: pd.DataFrame, df_model_seeds: pd.DataFrame,
                  model_name: str, model_color: str,
                  max_lag: int, out_path: str):
    """2×4 grid of boxplots (one per lag) showing per-seed vs observed."""
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
    axes = axes.flatten()

    for lag in range(1, max_lag + 1):
        ax  = axes[lag - 1]
        sub = pd.concat([
            df_obs[df_obs["Lag"] == lag],
            df_model_seeds[df_model_seeds["Lag"] == lag],
        ], ignore_index=True)

        sub["Group"] = sub["Source"].apply(
            lambda x: "Truth" if x == "Truth" else model_name
        )
        palette = {"Truth": COLORS["Observed"], model_name: model_color}

        sns.boxplot(data=sub, x="Source", y="Correlation",
                    hue="Group", palette=palette,
                    showfliers=False, dodge=False, ax=ax)

        ax.set_title(f"Lag {lag}")
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(alpha=0.3)
        leg = ax.get_legend()
        if leg:
            leg.remove()

    axes[0].set_ylabel("Spatial Correlation")
    axes[4].set_ylabel("Spatial Correlation")
    fig.suptitle(
        f"{model_name}: Horizontal Lagged Correlation (Observed vs 10 seeds)",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading dataset from: {DATASET_PATH}")
    splits = np.load(DATASET_PATH)
    Xtest  = splits["Xtest"][..., np.newaxis].astype(np.float32)
    Ytest  = squeeze_hw(splits["Ytest"])

    noise = np.random.default_rng(42).standard_normal(Xtest.shape).astype(np.float32)

    print("\nRunning U-Net inference...")
    preds_unet = infer_tf_models(UNET_MODEL_DIR, UNET_GLOB, Xtest, noise)

    print("\nRunning WGAN inference...")
    preds_wgan = infer_tf_models(WGAN_MODEL_DIR, WGAN_GLOB, Xtest, noise)

    print("\nLoading DDPM predictions...")
    preds_ddpm = load_ddpm_preds(DDPM_PRED_DIR, DDPM_SEEDS)

    # ── Observed correlations (computed once, reused) ─────────────────────────
    print("\nComputing observed correlations...")
    df_obs_combined = spatial_correlation_df(Ytest, MAX_LAG, "Observed")
    df_obs_label    = spatial_correlation_df(Ytest, MAX_LAG, "Truth")
    df_obs_label.rename(columns={"Model": "Source"}, inplace=True)

    # ── Figure 5: combined pooled across seeds ────────────────────────────────
    print("\nBuilding Figure 5 data...")
    all_dfs = [df_obs_combined]

    for seed, preds in sorted(preds_unet.items()):
        all_dfs.append(spatial_correlation_df(preds, MAX_LAG, "U-Net", seed))
    for seed, preds in sorted(preds_wgan.items()):
        all_dfs.append(spatial_correlation_df(preds, MAX_LAG, "WGAN", seed))
    for seed, preds in sorted(preds_ddpm.items()):
        all_dfs.append(spatial_correlation_df(preds, MAX_LAG, "DDPM", seed))

    df_combined = pd.concat(all_dfs, ignore_index=True)
    plot_figure5(df_combined, OUTPUT_DIR)

    # ── Figure S8: per-seed panels ────────────────────────────────────────────
    print("\nBuilding Figure S8 per-seed panels...")

    for model_name, color, preds_dict, fname in [
        ("U-Net", COLORS["U-Net"], preds_unet, "spatial_corr_unet_seeds.png"),
        ("WGAN",  COLORS["WGAN"],  preds_wgan, "spatial_corr_wgan_seeds.png"),
        ("DDPM",  COLORS["DDPM"],  preds_ddpm, "spatial_corr_ddpm_seeds.png"),
    ]:
        seed_dfs = []
        for i, (seed, preds) in enumerate(sorted(preds_dict.items()), start=1):
            df_s = spatial_correlation_df(preds, MAX_LAG, f"Seed {i}", seed)
            df_s.rename(columns={"Model": "Source"}, inplace=True)
            seed_dfs.append(df_s)

        df_model_seeds = pd.concat(seed_dfs, ignore_index=True)
        plot_per_seed(
            df_obs_label, df_model_seeds,
            model_name, color, MAX_LAG,
            os.path.join(OUTPUT_DIR, fname),
        )

    print("\nAll spatial correlation figures saved.")


if __name__ == "__main__":
    main()
