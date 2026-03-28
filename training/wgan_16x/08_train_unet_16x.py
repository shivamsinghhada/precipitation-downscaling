"""
Script:      08_train_unet_16x.py
Description: Trains an ensemble of 10 stochastic U-Net generators for 16×
             precipitation downscaling (8×8 → 128×128) using MSE loss.

             For each seed (1–10), a fresh U-Net is built and trained from
             scratch.  EarlyStopping (patience=10) is used to prevent
             over-fitting; the best checkpoint (lowest val_loss) is saved
             for subsequent WGAN-GP fine-tuning in script 09.

             Compared with the 8× U-Net experiment (script 06):
               - LR input is 8×8 (not 16×16)
               - The generator internally upsamples LR and noise to 16×16
                 before the encoder (see build_unet_generator_16x in models_16x.py)
               - 10 seeds are trained in a single run (not one seed at a time)
               - EarlyStopping is enabled with patience=10
               - max epochs = 500

Inputs:      dataset_splits_16x.npz produced by 05b_prepare_dataset_16x.py
Outputs:     In OUTPUT_DIR:
               unet_generator_best_seed{1..10}.keras — best checkpoints
               all_histories_unet.npy                — all training histories
               training_curves_all_seeds.png          — combined loss plot

Usage:       python 08_train_unet_16x.py

Requirements: numpy, tensorflow>=2.12, matplotlib  (see environment.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
DATASET_PATH = "/path/to/ERA5_land/Dataset/dataset_splits_16x.npz"
OUTPUT_DIR   = "/path/to/unet_runs_16x"

SEEDS      = list(range(1, 11))   # seeds 1–10
EPOCHS     = 500
BATCH_SIZE = 32
LR         = 1e-4                 # Adam learning rate
ES_PATIENCE= 10                   # EarlyStopping patience (epochs)
# ────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.models_wgan_16x import build_unet_generator_16x

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading dataset from: {DATASET_PATH}")
splits = np.load(DATASET_PATH)

# Add channel dimension: (N, H, W) → (N, H, W, 1)
Xtrain = splits["Xtrain"][..., np.newaxis].astype(np.float32)   # (N, 8,   8,   1)
Xval   = splits["Xval"][...,   np.newaxis].astype(np.float32)
Ytrain = splits["Ytrain"][..., np.newaxis].astype(np.float32)   # (N, 128, 128, 1)
Yval   = splits["Yval"][...,   np.newaxis].astype(np.float32)

print(f"  Xtrain: {Xtrain.shape}  Ytrain: {Ytrain.shape}")
print(f"  Xval:   {Xval.shape}    Yval:   {Yval.shape}")

# ── Training loop ─────────────────────────────────────────────────────────────
all_histories = {}

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"  Training U-Net (16×) — seed {seed} / {SEEDS[-1]}")
    print(f"{'='*50}")

    # Reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Build a fresh model for each seed
    unet = build_unet_generator_16x(
        lr_shape=Xtrain.shape[1:],
        noise_shape=Xtrain.shape[1:],
    )
    unet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="mse",
    )

    # Fixed noise for this seed — reproducible across re-runs
    rng         = np.random.default_rng(seed=seed)
    noise_train = rng.standard_normal(Xtrain.shape).astype(np.float32)
    noise_val   = rng.standard_normal(Xval.shape).astype(np.float32)

    # Callbacks
    checkpoint_path = os.path.join(
        OUTPUT_DIR, f"unet_generator_best_seed{seed}.keras"
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=ES_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    history = unet.fit(
        [Xtrain, noise_train], Ytrain,
        validation_data=([Xval, noise_val], Yval),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint],
        shuffle=True,
        verbose=1,
    )

    all_histories[seed] = history.history
    stopped_epoch = len(history.history["loss"])
    print(f"  Seed {seed} stopped at epoch {stopped_epoch}. "
          f"Best checkpoint: {checkpoint_path}")

# ── Save all histories ────────────────────────────────────────────────────────
hist_path = os.path.join(OUTPUT_DIR, "all_histories_unet.npy")
np.save(hist_path, all_histories)
print(f"\nAll U-Net histories saved to: {hist_path}")

# ── Plot combined loss curves ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for seed, hist in all_histories.items():
    epochs_x = np.arange(len(hist["loss"]))
    ax.plot(epochs_x, hist["loss"],     color="steelblue", alpha=0.55,
            label="Train MSE" if seed == SEEDS[0] else "")
    ax.plot(epochs_x, hist["val_loss"], color="tomato",    alpha=0.55,
            linestyle="--", label="Val MSE" if seed == SEEDS[0] else "")

ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("U-Net (16×) Training and Validation Loss — all seeds")
ax.legend(ncol=2, fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()

curve_path = os.path.join(OUTPUT_DIR, "training_curves_all_seeds.png")
fig.savefig(curve_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Loss curves saved to: {curve_path}")
print(f"\nNext step: run  09_train_wgan_16x.py")
