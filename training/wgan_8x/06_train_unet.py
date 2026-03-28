"""
Script:      06_train_unet.py
Description: Trains the stochastic U-Net generator for 8× precipitation
             downscaling (16×16 → 128×128) using MSE loss.

             A fixed Gaussian noise field is concatenated with the LR input
             at each forward pass, enabling the model to learn stochastic
             mappings.  The best checkpoint (lowest val_loss) is saved for
             subsequent WGAN-GP fine-tuning in script 07.

             To train an ensemble of 10 generators (one per random seed),
             run script 07 which loops over seeds 1–10.  This script trains
             only a single seed (default: seed = 1) for quick validation.

Inputs:      dataset_splits.npz produced by 05_prepare_dataset.py
Outputs:     In OUTPUT_DIR:
               unet_generator_best_seed{seed}.keras  — best checkpoint
               training_curve_seed{seed}.png          — loss curve
               training_time_seed{seed}.txt           — wall-clock time

Usage:       python 06_train_unet.py

Requirements: numpy, tensorflow>=2.12, matplotlib  (see environment.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
DATASET_PATH = "/path/to/ERA5_land/Dataset/dataset_splits.npz"
OUTPUT_DIR   = "/path/to/unet_runs"

SEED       = 1     # random seed for this run
EPOCHS     = 200
BATCH_SIZE = 32
LR         = 1e-4  # Adam learning rate
# ────────────────────────────────────────────────────────────────────────────

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.models_wgan_8x import build_unet_generator

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading dataset from: {DATASET_PATH}")
splits = np.load(DATASET_PATH)

# Add channel dimension: (N, H, W) → (N, H, W, 1)
Xtrain = splits["Xtrain"][..., np.newaxis].astype(np.float32)
Xval   = splits["Xval"][...,   np.newaxis].astype(np.float32)
Ytrain = splits["Ytrain"][..., np.newaxis].astype(np.float32)
Yval   = splits["Yval"][...,   np.newaxis].astype(np.float32)

print(f"  Xtrain: {Xtrain.shape}  Ytrain: {Ytrain.shape}")
print(f"  Xval:   {Xval.shape}    Yval:   {Yval.shape}")

# ── Generate fixed noise inputs ───────────────────────────────────────────────
# Using a fixed seed ensures noise arrays are reproducible across runs.
rng          = np.random.default_rng(seed=SEED)
noise_train  = rng.standard_normal(Xtrain.shape).astype(np.float32)
noise_val    = rng.standard_normal(Xval.shape).astype(np.float32)

# ── Build and compile model ───────────────────────────────────────────────────
print(f"\nBuilding U-Net generator (seed {SEED})...")
unet = build_unet_generator(
    lr_shape=Xtrain.shape[1:],
    noise_shape=Xtrain.shape[1:],
)
unet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="mse",
)
unet.summary()

# ── Callbacks ────────────────────────────────────────────────────────────────
checkpoint_path = os.path.join(OUTPUT_DIR, f"unet_generator_best_seed{SEED}.keras")
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"\nTraining U-Net (seed={SEED}, epochs={EPOCHS}, batch={BATCH_SIZE})...")
start_time = time.time()

history = unet.fit(
    [Xtrain, noise_train], Ytrain,
    validation_data=([Xval, noise_val], Yval),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint],
    shuffle=True,
    verbose=1,
)

elapsed = time.time() - start_time

# ── Save training time ────────────────────────────────────────────────────────
time_path = os.path.join(OUTPUT_DIR, f"training_time_seed{SEED}.txt")
with open(time_path, "w") as f:
    f.write(f"Training time (seconds): {elapsed:.2f}\n")
    f.write(f"Training time (minutes): {elapsed / 60:.2f}\n")
    f.write(f"Training time (hours):   {elapsed / 3600:.2f}\n")

print(f"\nTraining complete in {elapsed / 60:.1f} min ({elapsed / 3600:.2f} h)")
print(f"Best checkpoint saved to: {checkpoint_path}")

# ── Plot loss curve ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history.history["loss"],     label="Train MSE")
ax.plot(history.history["val_loss"], label="Val MSE", linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (MSE)")
ax.set_title(f"U-Net Training Curve (Seed {SEED})")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

curve_path = os.path.join(OUTPUT_DIR, f"training_curve_seed{SEED}.png")
fig.savefig(curve_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Loss curve saved to: {curve_path}")
print(f"\nNext step: run  07_train_wgan.py")
