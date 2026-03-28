"""
Script:      analysis/12_plot_loss_curves.py
Description: Plots training and validation loss curves for all three model
             families across all 10 seeds:

               (a) U-Net MSE loss (from all_histories_unet.npy or all_histories.npy)
               (b) WGAN generator loss and critic loss
               (c) DDPM noise-prediction MSE loss

             Each panel overlays all seeds transparently so the spread of
             training behaviour is visible.

Inputs:
  UNET_HISTORY_PATH  : .npy file saved by 07_train_wgan.py or 08_train_unet_16x.py
                       (dict: seed → {"loss": [...], "val_loss": [...]})
  WGAN_HISTORY_PATH  : .npy file saved by 07_train_wgan.py or 09_train_wgan_16x.py
                       (dict: seed → {"mse": [...], "val_mse": [...],
                                      "g_loss": [...], "d_loss": [...]})
  DDPM_CHECKPOINT_DIR: directory containing seed_{N}/train_losses.npy
                       and seed_{N}/val_losses.npy saved by 10/11_train_ddpm*.py

Outputs:     loss_curves_unet.png, loss_curves_wgan.png, loss_curves_ddpm.png
             in OUTPUT_DIR

Usage:       python analysis/12_plot_loss_curves.py

Requirements: numpy, matplotlib  (see environment_tf.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
# Paths for the 8× experiment — change to 16× equivalents as needed.
UNET_HISTORY_PATH   = "/path/to/unet_runs/all_histories.npy"
WGAN_HISTORY_PATH   = "/path/to/unet_runs/WGANs/all_histories.npy"
DDPM_CHECKPOINT_DIR = "/path/to/checkpoints/ddpm_8x"

OUTPUT_DIR = "/path/to/results"

SEEDS_UNET_WGAN = list(range(1, 11))   # seeds 1–10
SEEDS_DDPM      = list(range(1, 11))   # seeds 1–10 (matches 10/11_train_ddpm*.py)
# ────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize":13,
    "xtick.labelsize":12,
    "ytick.labelsize":12,
})


# ── (a) U-Net loss curves ─────────────────────────────────────────────────────

def plot_unet_loss(history_path: str, out_path: str):
    """
    Plot training and validation MSE for all U-Net seeds.

    Expects history dict: {seed: {"loss": [...], "val_loss": [...]}}
    """
    if not os.path.exists(history_path):
        print(f"WARNING: U-Net history not found: {history_path}")
        return

    all_histories = np.load(history_path, allow_pickle=True).item()
    print(f"Loaded {len(all_histories)} U-Net histories")

    fig, ax = plt.subplots(figsize=(10, 6))
    first = True

    for seed, hist in all_histories.items():
        train = np.array(hist.get("loss", []))
        val   = np.array(hist.get("val_loss", []))

        if train.size == 0 or val.size == 0:
            print(f"  WARNING: empty history for seed {seed}, skipping.")
            continue

        epochs = np.arange(len(train))
        ax.plot(epochs, train, color="steelblue", alpha=0.55,
                label="Training" if first else "")
        ax.plot(epochs, val,   color="tomato",    alpha=0.55,
                linestyle="--", label="Validation" if first else "")
        first = False

    ax.set_title("U-Net Training and Validation Loss (all seeds)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── (b) WGAN loss curves ──────────────────────────────────────────────────────

def plot_wgan_loss(history_path: str, out_path: str):
    """
    Plot WGAN generator loss, critic loss, and validation MSE for all seeds.

    Expects history dict: {seed: {"g_loss": [...], "d_loss": [...],
                                   "mse": [...], "val_mse": [...]}}
    """
    if not os.path.exists(history_path):
        print(f"WARNING: WGAN history not found: {history_path}")
        return

    all_histories = np.load(history_path, allow_pickle=True).item()
    print(f"Loaded {len(all_histories)} WGAN histories")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    first = True
    for seed, hist in all_histories.items():
        g_loss = np.array(hist.get("g_loss", []))
        d_loss = np.array(hist.get("d_loss", []))
        mse    = np.array(hist.get("mse",    []))
        val_mse= np.array(hist.get("val_mse",[]))

        if g_loss.size == 0:
            print(f"  WARNING: empty WGAN history for seed {seed}, skipping.")
            continue

        ep = np.arange(len(g_loss))

        axes[0].plot(ep, g_loss, color="steelblue", alpha=0.45,
                     label="Generator" if first else "")
        axes[0].plot(ep, d_loss, color="tomato",    alpha=0.45,
                     linestyle="--", label="Critic" if first else "")

        if mse.size > 0 and val_mse.size > 0:
            ep2 = np.arange(len(mse))
            axes[1].plot(ep2, mse,     color="steelblue", alpha=0.45,
                         label="Train MSE" if first else "")
            axes[1].plot(ep2, val_mse, color="tomato",    alpha=0.45,
                         linestyle="--", label="Val MSE" if first else "")

        first = False

    axes[0].set_title("WGAN Generator & Critic Loss (all seeds)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Wasserstein Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right")

    axes[1].set_title("WGAN Validation MSE (all seeds)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── (c) DDPM loss curves ──────────────────────────────────────────────────────

def plot_ddpm_loss(checkpoint_dir: str, seeds: list, out_path: str):
    """
    Plot DDPM noise-prediction MSE for all seeds.

    Expects per-seed .npy files saved by 10/11_train_ddpm*.py:
      {checkpoint_dir}/seed_{N}/train_losses.npy
      {checkpoint_dir}/seed_{N}/val_losses.npy
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    first = True
    found = 0

    for seed in seeds:
        seed_dir   = os.path.join(checkpoint_dir, f"seed_{seed}")
        train_path = os.path.join(seed_dir, "train_losses.npy")
        val_path   = os.path.join(seed_dir, "val_losses.npy")

        if not (os.path.exists(train_path) and os.path.exists(val_path)):
            print(f"  WARNING: missing loss files for DDPM seed {seed}, skipping.")
            continue

        train_losses = np.load(train_path)
        val_losses   = np.load(val_path)
        epochs       = np.arange(len(train_losses))

        ax.plot(epochs, train_losses, color="steelblue", alpha=0.35,
                label="Training" if first else "")
        ax.plot(epochs, val_losses,   color="tomato",    alpha=0.35,
                linestyle="--", label="Validation" if first else "")

        first = False
        found += 1

    print(f"Plotted {found}/{len(seeds)} DDPM seeds.")
    ax.set_title("DDPM Training and Validation Loss (all seeds)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Noise Prediction MSE")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_unet_loss(
        history_path=UNET_HISTORY_PATH,
        out_path=os.path.join(OUTPUT_DIR, "loss_curves_unet.png"),
    )
    plot_wgan_loss(
        history_path=WGAN_HISTORY_PATH,
        out_path=os.path.join(OUTPUT_DIR, "loss_curves_wgan.png"),
    )
    plot_ddpm_loss(
        checkpoint_dir=DDPM_CHECKPOINT_DIR,
        seeds=SEEDS_DDPM,
        out_path=os.path.join(OUTPUT_DIR, "loss_curves_ddpm.png"),
    )
    print("\nAll loss curves saved.")
