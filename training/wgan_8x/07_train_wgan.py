"""
Script:      07_train_wgan.py
Description: Fine-tunes a WGAN-GP ensemble of 10 generators for 8×
             precipitation downscaling (16×16 → 128×128).

             For each seed (1–10), the best U-Net checkpoint saved by
             script 06 is loaded as the generator initialisation, and a
             fresh conditional critic is built.  Training uses the
             Wasserstein loss with gradient penalty (WGAN-GP).

             Multi-GPU training is enabled automatically via
             tf.distribute.MirroredStrategy when multiple GPUs are available.

             Validation MSE is tracked after every epoch; the best generator
             (lowest val_mse) and the final generator are both saved.

Inputs:      dataset_splits.npz produced by 05_prepare_dataset.py
             unet_generator_best_seed{1..10}.keras from 06_train_unet.py
             (run script 06 ten times with SEED = 1..10 to produce these)
Outputs:     In WGAN_DIR:
               gen_best_seed{seed}.keras     — best checkpoint per seed
               gen_final_seed{seed}.keras    — final state per seed
               training_time_seed{seed}.txt  — wall-clock time per seed
               all_histories.npy             — dict of all training histories
               loss_curves_all_seeds.png     — combined loss plot

Usage:       python 07_train_wgan.py

Requirements: numpy, tensorflow>=2.12, matplotlib  (see environment.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
DATASET_PATH = "/path/to/ERA5_land/Dataset/dataset_splits.npz"
UNET_DIR     = "/path/to/unet_runs"           # where script 06 saved U-Net checkpoints
WGAN_DIR     = "/path/to/unet_runs/WGANs"     # output directory for WGAN results

SEEDS      = list(range(1, 11))   # seeds 1–10; change to [1] for a quick single run
EPOCHS     = 200
BATCH_SIZE = 32
LR         = 1e-4   # Adam learning rate for both generator and critic
GP_WEIGHT  = 10.0   # gradient penalty coefficient
D_STEPS    = 3      # critic updates per generator update
# ────────────────────────────────────────────────────────────────────────────

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.models_wgan_8x import build_conditional_critic, WGAN

os.makedirs(WGAN_DIR, exist_ok=True)

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

# ── tf.data pipelines (built once, reused across seeds) ───────────────────────
# Noise is generated inside WGAN.train_step / test_step, so only LR and HR
# fields are needed in the dataset.
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

# ── Loss functions (defined outside strategy scope — they are pure functions) ──
def d_loss_fn(real_logits, fake_logits):
    """Wasserstein critic loss (maximise real − fake)."""
    return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

def g_loss_fn(fake_logits):
    """Wasserstein generator loss (minimise −fake)."""
    return -tf.reduce_mean(fake_logits)

# ── Training loop over seeds ──────────────────────────────────────────────────
all_histories = {}

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"  Training WGAN — seed {seed} / {SEEDS[-1]}")
    print(f"{'='*50}")

    gen_path = os.path.join(UNET_DIR, f"unet_generator_best_seed{seed}.keras")
    if not os.path.exists(gen_path):
        raise FileNotFoundError(
            f"U-Net checkpoint not found: {gen_path}\n"
            "Run script 06 with SEED={seed} first."
        )

    with strategy.scope():
        # Load pre-trained U-Net generator; build fresh critic
        generator = tf.keras.models.load_model(gen_path, compile=False)
        critic    = build_conditional_critic()

        g_opt = tf.keras.optimizers.Adam(LR, beta_1=0.0, beta_2=0.9)
        d_opt = tf.keras.optimizers.Adam(LR, beta_1=0.0, beta_2=0.9)

        wgan = WGAN(generator, critic, gp_weight=GP_WEIGHT, d_steps=D_STEPS)
        wgan.compile(
            g_optimizer=g_opt,
            d_optimizer=d_opt,
            g_loss_fn=g_loss_fn,
            d_loss_fn=d_loss_fn,
        )

    # Checkpoint: monitor val_mse (logged by WGAN.test_step)
    ckpt_path = os.path.join(WGAN_DIR, f"gen_best_seed{seed}.keras")
    ckpt_cb   = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor="val_mse",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    start_time = time.time()

    history = wgan.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[ckpt_cb],
        verbose=1,
    )

    elapsed = time.time() - start_time

    # Save final generator state
    final_path = os.path.join(WGAN_DIR, f"gen_final_seed{seed}.keras")
    generator.save(final_path)

    # Save wall-clock time
    time_path = os.path.join(WGAN_DIR, f"training_time_seed{seed}.txt")
    with open(time_path, "w") as f:
        f.write(f"Training time (seconds): {elapsed:.2f}\n")
        f.write(f"Training time (minutes): {elapsed / 60:.2f}\n")
        f.write(f"Training time (hours):   {elapsed / 3600:.2f}\n")

    all_histories[seed] = history.history
    print(f"  Seed {seed} done in {elapsed / 60:.1f} min — "
          f"best checkpoint: {ckpt_path}")

# ── Save all histories ────────────────────────────────────────────────────────
hist_path = os.path.join(WGAN_DIR, "all_histories.npy")
np.save(hist_path, all_histories)
print(f"\nAll histories saved to: {hist_path}")

# ── Plot combined loss curves ─────────────────────────────────────────────────
# WGAN.train_step returns "mse"; WGAN.test_step returns "val_mse".
# (Not "loss"/"val_loss" — those keys do not exist in WGAN history.)

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

ax.set_title("WGAN Training and Validation MSE (8× downscaling, all seeds)")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper right")
fig.tight_layout()

curve_path = os.path.join(WGAN_DIR, "loss_curves_all_seeds.png")
fig.savefig(curve_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Loss curves saved to: {curve_path}")
