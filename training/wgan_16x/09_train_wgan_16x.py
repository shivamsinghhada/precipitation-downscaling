"""
Script:      09_train_wgan_16x.py
Description: Fine-tunes a WGAN-GP ensemble for the 16× downscaling experiment
             (8×8 → 128×128).

             For each seed (1–10), the best U-Net checkpoint saved by
             script 08 is loaded as the generator initialisation, and a
             fresh conditional critic is built.  Training proceeds with the
             Wasserstein loss + gradient penalty (WGAN-GP).

             Compared with the 8× WGAN experiment (script 07):
               - LR input shape is (8, 8, 1)
               - Generator and critic from models_16x.py (WGAN16x) are used
               - Per-seed training history is saved as both .npy and .csv

Inputs:      dataset_splits_16x.npz produced by 05b_prepare_dataset_16x.py
             unet_generator_best_seed{1..10}.keras from 08_train_unet_16x.py
Outputs:     In WGAN_DIR:
               gen_best_seed{seed}.keras      — best checkpoint per seed
               gen_final_seed{seed}.keras     — final state per seed
               history_seed{seed}.csv         — per-epoch metrics as CSV
               all_histories_wgan.npy         — all training histories
               loss_curves_all_seeds.png      — combined MSE plot

Usage:       python 09_train_wgan_16x.py

             To resume after interruption, set SEEDS to the remaining seeds,
             e.g.  SEEDS = list(range(9, 11))  to redo only seeds 9 and 10.

Requirements: numpy, pandas, tensorflow>=2.12, matplotlib  (see environment.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
DATASET_PATH = "/path/to/ERA5_land/Dataset/dataset_splits_16x.npz"
UNET_DIR     = "/path/to/unet_runs_16x"           # output of script 08
WGAN_DIR     = "/path/to/unet_runs_16x/WGANs"     # output directory

SEEDS      = list(range(1, 11))   # change to e.g. [9, 10] to resume
EPOCHS     = 200
BATCH_SIZE = 32
LR         = 1e-4
GP_WEIGHT  = 10.0
D_STEPS    = 3
# ────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from models.models_wgan_16x import build_conditional_critic_16x, WGAN16x

os.makedirs(WGAN_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading dataset from: {DATASET_PATH}")
splits = np.load(DATASET_PATH)

Xtrain = splits["Xtrain"][..., np.newaxis].astype(np.float32)   # (N, 8,   8,   1)
Xval   = splits["Xval"][...,   np.newaxis].astype(np.float32)
Ytrain = splits["Ytrain"][..., np.newaxis].astype(np.float32)   # (N, 128, 128, 1)
Yval   = splits["Yval"][...,   np.newaxis].astype(np.float32)

print(f"  Xtrain: {Xtrain.shape}  Ytrain: {Ytrain.shape}")
print(f"  Xval:   {Xval.shape}    Yval:   {Yval.shape}")

# ── tf.data pipelines ────────────────────────────────────────────────────────
train_dataset = (
    tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain))
    .shuffle(2048, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
val_dataset = (
    tf.data.Dataset.from_tensor_slices((Xval, Yval))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ── Multi-GPU strategy ────────────────────────────────────────────────────────
strategy = tf.distribute.MirroredStrategy()
print(f"Using {strategy.num_replicas_in_sync} GPU(s)\n")

# ── Loss functions ────────────────────────────────────────────────────────────
def d_loss_fn(real_logits, fake_logits):
    """Wasserstein critic loss."""
    return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

def g_loss_fn(fake_logits):
    """Wasserstein generator loss."""
    return -tf.reduce_mean(fake_logits)

# ── Training loop over seeds ──────────────────────────────────────────────────
all_histories = {}

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"  Training WGAN (16×) — seed {seed} / {SEEDS[-1]}")
    print(f"{'='*50}")

    gen_path = os.path.join(UNET_DIR, f"unet_generator_best_seed{seed}.keras")
    if not os.path.exists(gen_path):
        raise FileNotFoundError(
            f"U-Net checkpoint not found: {gen_path}\n"
            f"Run script 08 with seed {seed} first."
        )

    with strategy.scope():
        generator = tf.keras.models.load_model(gen_path, compile=False)
        critic    = build_conditional_critic_16x()

        g_opt = tf.keras.optimizers.Adam(LR, beta_1=0.0, beta_2=0.9)
        d_opt = tf.keras.optimizers.Adam(LR, beta_1=0.0, beta_2=0.9)

        wgan = WGAN16x(generator, critic, gp_weight=GP_WEIGHT, d_steps=D_STEPS)
        wgan.compile(
            g_optimizer=g_opt,
            d_optimizer=d_opt,
            g_loss_fn=g_loss_fn,
            d_loss_fn=d_loss_fn,
        )

    # Checkpoint monitors val_mse — matches WGAN16x.test_step return key
    ckpt_path = os.path.join(WGAN_DIR, f"gen_best_seed{seed}.keras")
    ckpt_cb   = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor="val_mse",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    history = wgan.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[ckpt_cb],
        verbose=1,
    )

    # Save final generator
    final_path = os.path.join(WGAN_DIR, f"gen_final_seed{seed}.keras")
    generator.save(final_path)

    # Save per-seed history as CSV (convenient for inspection) and collect
    hist_df   = pd.DataFrame(history.history)
    hist_path = os.path.join(WGAN_DIR, f"history_seed{seed}.csv")
    hist_df.to_csv(hist_path, index=False)

    all_histories[seed] = history.history
    print(f"  Seed {seed} done. Best checkpoint: {ckpt_path}")
    print(f"  History CSV: {hist_path}")

# ── Save combined histories ───────────────────────────────────────────────────
npy_path = os.path.join(WGAN_DIR, "all_histories_wgan.npy")
np.save(npy_path, all_histories)
print(f"\nAll histories saved to: {npy_path}")

# ── Plot combined loss curves ─────────────────────────────────────────────────
# WGAN16x.train_step → "mse"  |  WGAN16x.test_step → "val_mse"
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize":13,
    "xtick.labelsize":12,
    "ytick.labelsize":12,
})

fig, ax = plt.subplots(figsize=(10, 6))
first = True

for seed, hist in all_histories.items():
    train_mse = np.array(hist.get("mse",     []))
    val_mse   = np.array(hist.get("val_mse", []))

    if train_mse.size == 0 or val_mse.size == 0:
        print(f"  WARNING: empty history for seed {seed}, skipping.")
        continue

    epochs_x = np.arange(len(train_mse))
    ax.plot(epochs_x, train_mse, color="steelblue", alpha=0.55,
            label="Train MSE" if first else "")
    ax.plot(epochs_x, val_mse,   color="tomato",    alpha=0.55,
            linestyle="--", label="Val MSE" if first else "")
    first = False

ax.set_title("WGAN (16×) Training and Validation MSE — all seeds")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper right")
fig.tight_layout()

curve_path = os.path.join(WGAN_DIR, "loss_curves_all_seeds.png")
fig.savefig(curve_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Loss curves saved to: {curve_path}")
