"""
Script:      analysis/16_plot_exceedance_and_qq.py
Description: Exceedance probability (1-CDF) and Q–Q plots — Figure 7 and
             Supplementary Figure S7.

             Figure 7: Three-panel composite (a) U-Net, (b) WGAN, (c) DDPM.
             Each panel shows the exceedance probability curve for all 10
             ensemble seeds overlaid on the observed curve (tail only,
             above EXCEEDANCE_THRESHOLD mm/day).

             Figure S7: Joint Q–Q plot showing mean ± 1σ quantile envelope
             across seeds for U-Net, WGAN, and DDPM against observed
             (restricted to wet values above QQ_THRESHOLD mm/day).

Inputs:      dataset_splits.npz   (for Xtest, Ytest)
             U-Net and WGAN model checkpoints (all 10 seeds)
             DDPM prediction .npy files for all 10 seeds
Outputs:     exceedance_composite.png   — Figure 7
             qq_plot.png                — Figure S7
             (saved to OUTPUT_DIR)

Usage:       python analysis/16_plot_exceedance_and_qq.py

Requirements: numpy, tensorflow, matplotlib  (see environment_tf.yml)
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
DDPM_SEEDS     = list(range(1, 11))

OUTPUT_DIR            = "/path/to/results"
EXCEEDANCE_THRESHOLD  = 10.0   # mm/day — tail threshold for Figure 7
QQ_THRESHOLD          = 20.0   # mm/day — wet threshold for Figure S7
N_QUANTILES           = 200    # quantile grid resolution
NOISE_STD             = 1.0
# ────────────────────────────────────────────────────────────────────────────

import os
import glob
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.plot_utils import COLORS, squeeze_hw, flatten_pos

os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA_SEED = 0.65
LW_SEED    = 1.5
LW_TRUTH   = 2.5


# ── Helpers ───────────────────────────────────────────────────────────────────

def infer_tf_models(model_dir, glob_pattern, X, noise):
    paths = sorted(glob.glob(os.path.join(model_dir, glob_pattern)))
    if not paths:
        raise RuntimeError(f"No models: {model_dir}/{glob_pattern}")
    out = {}
    for p in paths:
        seed  = int(p.split("seed")[-1].split(".")[0])
        model = tf.keras.models.load_model(p, compile=False)
        out[seed] = squeeze_hw(model.predict([X, noise], verbose=0))
        print(f"  Inference: seed {seed}")
    return out


def load_ddpm_preds(pred_dir, seeds):
    out = {}
    for s in seeds:
        path = os.path.join(pred_dir, f"seed_{s}", "best_model_preds.npy")
        if not os.path.exists(path):
            print(f"  WARNING: DDPM prediction not found: {path}")
            continue
        out[s] = squeeze_hw(np.load(path))
    return out


def exceedance_curve(arr: np.ndarray, threshold: float):
    """Return (x_sorted, exceedance_prob) for values > threshold."""
    x = flatten_pos(arr, threshold)
    if x.size == 0:
        return None, None
    x_s = np.sort(x)
    F   = np.arange(1, x_s.size + 1) / x_s.size
    return x_s, 1.0 - F


def quantile_envelope(arrays: list, threshold: float, n_q: int = 200):
    """
    Compute mean ± std quantile envelope across a list of arrays.

    Returns (obs_quantiles, mean_q, std_q).
    """
    q_grid = np.linspace(0, 1, n_q)
    Q_all  = []
    for arr in arrays:
        vals = flatten_pos(arr, threshold)
        if vals.size > 0:
            Q_all.append(np.quantile(vals, q_grid))
    if not Q_all:
        return None, None, None
    Q_all = np.stack(Q_all, axis=0)
    return q_grid, Q_all.mean(axis=0), Q_all.std(axis=0)


# ── Figure 7: exceedance panels ───────────────────────────────────────────────

def plot_exceedance(Ytest, preds_unet, preds_wgan, preds_ddpm, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    configs   = [
        (axes[0], preds_unet, COLORS["U-Net"], "(a)"),
        (axes[1], preds_wgan, COLORS["WGAN"],  "(b)"),
        (axes[2], preds_ddpm, COLORS["DDPM"],  "(c)"),
    ]

    for ax, preds_dict, color, label in configs:
        # Observed
        xt, yt = exceedance_curve(Ytest, EXCEEDANCE_THRESHOLD)
        if xt is not None:
            ax.plot(xt, yt, color=COLORS["Observed"],
                    linewidth=LW_TRUTH, zorder=10)

        # Each seed
        for seed in sorted(preds_dict):
            xp, yp = exceedance_curve(preds_dict[seed], EXCEEDANCE_THRESHOLD)
            if xp is None:
                continue
            ax.plot(xp, yp, color=color, alpha=ALPHA_SEED,
                    linewidth=LW_SEED, zorder=2)

        ax.set_yscale("log")
        ax.grid(False)
        ax.tick_params(labelsize=12)
        ax.set_xlabel("Precipitation (mm/day)", fontsize=13)
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="top")

    axes[0].set_ylabel("Exceedance Probability", fontsize=13)

    legend_handles = [
        Line2D([0], [0], color=COLORS["Observed"], lw=LW_TRUTH, label="Observed"),
        Line2D([0], [0], color=COLORS["U-Net"],    lw=2.0, alpha=ALPHA_SEED, label="U-Net"),
        Line2D([0], [0], color=COLORS["WGAN"],     lw=2.0, alpha=ALPHA_SEED, label="WGAN"),
        Line2D([0], [0], color=COLORS["DDPM"],     lw=2.0, alpha=ALPHA_SEED, label="DDPM"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure S7: Q–Q plot ───────────────────────────────────────────────────────

def plot_qq(Ytest, preds_unet, preds_wgan, preds_ddpm, out_path):
    obs_vals = flatten_pos(Ytest, QQ_THRESHOLD)
    obs_q    = np.quantile(obs_vals, np.linspace(0, 1, N_QUANTILES))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(obs_q, obs_q, "k--", linewidth=2, label="1:1 Line")

    for model_name, color, preds_dict in [
        ("U-Net", COLORS["U-Net"], preds_unet),
        ("WGAN",  COLORS["WGAN"],  preds_wgan),
        ("DDPM",  COLORS["DDPM"],  preds_ddpm),
    ]:
        _, mean_q, std_q = quantile_envelope(
            list(preds_dict.values()), QQ_THRESHOLD, N_QUANTILES
        )
        if mean_q is None:
            continue
        ax.plot(obs_q, mean_q, color=color, linewidth=2, label=model_name)
        ax.fill_between(obs_q, mean_q - std_q, mean_q + std_q,
                        color=color, alpha=0.2)

    ax.set_xlabel("Observed Quantiles (mm/day)", fontsize=13)
    ax.set_ylabel("Model Quantiles (mm/day)", fontsize=13)
    ax.legend(loc="upper left")
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading dataset from: {DATASET_PATH}")
    splits = np.load(DATASET_PATH)
    Xtest  = splits["Xtest"][..., np.newaxis].astype(np.float32)
    Ytest  = squeeze_hw(splits["Ytest"])

    noise  = np.random.default_rng(42).standard_normal(Xtest.shape).astype(np.float32)

    print("\nRunning U-Net inference...")
    preds_unet = infer_tf_models(UNET_MODEL_DIR, UNET_GLOB, Xtest, noise)

    print("\nRunning WGAN inference...")
    preds_wgan = infer_tf_models(WGAN_MODEL_DIR, WGAN_GLOB, Xtest, noise)

    print("\nLoading DDPM predictions...")
    preds_ddpm = load_ddpm_preds(DDPM_PRED_DIR, DDPM_SEEDS)

    plot_exceedance(
        Ytest, preds_unet, preds_wgan, preds_ddpm,
        os.path.join(OUTPUT_DIR, "exceedance_composite.png"),
    )
    plot_qq(
        Ytest, preds_unet, preds_wgan, preds_ddpm,
        os.path.join(OUTPUT_DIR, "qq_plot.png"),
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
