"""
Script:      11_train_ddpm_16x.py
Description: Trains an ensemble of 10 conditional DDPM models for 16×
             precipitation downscaling (8×8 → 128×128) using PyTorch.

             Identical training procedure to 10_train_ddpm_8x.py except:
               - Reads dataset_splits_16x.npz (LR: 8×8, HR: 128×128)
               - Uses scale_factor=16 in build_conditional_unet(), which
                 adds one extra bilinear Upsample step in cond_up
                 (4 steps: 8→16→32→64→128 instead of 3: 16→32→64→128)
               - Outputs saved to a separate BASE_CHECKPOINT_DIR

             To resume a partial run, set SEEDS to the remaining seeds,
             e.g.  SEEDS = [9, 10]  to redo only seeds 9 and 10.

Inputs:      dataset_splits_16x.npz produced by 05b_prepare_dataset_16x.py
             (LR: 8×8, HR: 128×128)
Outputs:     Same structure as 10_train_ddpm_8x.py, in BASE_CHECKPOINT_DIR.

Usage:       python 11_train_ddpm_16x.py

Requirements: torch>=2.0, numpy, matplotlib  (see environment.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
DATASET_PATH        = "/path/to/ERA5_land/Dataset/dataset_splits_16x.npz"
BASE_CHECKPOINT_DIR = "/path/to/checkpoints/ddpm_16x"

SEEDS        = list(range(1, 11))   # seeds 1–10; set e.g. [9, 10] to resume
EPOCHS       = 200
BATCH_SIZE   = 16
LR           = 1e-4
T            = 100    # diffusion timesteps (cosine schedule)
SCALE_FACTOR = 16     # LR → HR upscaling factor (8×8 → 128×128)
# ────────────────────────────────────────────────────────────────────────────

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.models_ddpm import Diffusion, build_conditional_unet

os.makedirs(BASE_CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Reproducibility helper ────────────────────────────────────────────────────

def set_seed(seed: int):
    """Set all random seeds for full reproducibility across PyTorch and NumPy."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Data loading and normalisation ───────────────────────────────────────────

def clean_precip(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0 and clip negative values to 0."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, 0.0, None)


def log1p_minmax_normalize(
    tensor: torch.Tensor,
    global_min: torch.Tensor = None,
    global_max: torch.Tensor = None,
) -> tuple:
    """
    Apply log1p then min-max normalisation to [0, 1].

    Training statistics (global_min, global_max) must be computed on the
    training set only and then passed when normalising val/test data.

    Returns (normalised_tensor, global_min, global_max).
    """
    tensor = torch.log1p(tensor)
    if global_min is None:
        global_min = tensor.min()
    if global_max is None:
        global_max = tensor.max()
    return (tensor - global_min) / (global_max - global_min + 1e-8), global_min, global_max


print(f"Loading dataset from: {DATASET_PATH}")
splits = np.load(DATASET_PATH)

Xtrain_raw = torch.from_numpy(clean_precip(splits["Xtrain"])).float()  # (N, 8, 8)
Ytrain_raw = torch.from_numpy(clean_precip(splits["Ytrain"])).float()  # (N, 128, 128)
Xval_raw   = torch.from_numpy(clean_precip(splits["Xval"])).float()
Yval_raw   = torch.from_numpy(clean_precip(splits["Yval"])).float()

print(f"  Xtrain: {Xtrain_raw.shape}  Ytrain: {Ytrain_raw.shape}")
print(f"  Xval:   {Xval_raw.shape}    Yval:   {Yval_raw.shape}")

# Normalise using training statistics only
Ytrain_norm, y_min, y_max = log1p_minmax_normalize(Ytrain_raw)
Xtrain_norm, x_min, x_max = log1p_minmax_normalize(Xtrain_raw)
Xval_norm, *_ = log1p_minmax_normalize(Xval_raw, x_min, x_max)
Yval_norm, *_ = log1p_minmax_normalize(Yval_raw, y_min, y_max)

# Save normalisation stats for inference/denormalisation
norm_path = os.path.join(BASE_CHECKPOINT_DIR, "global_denorm.pth")
torch.save({"x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max}, norm_path)
print(f"Normalisation stats saved to: {norm_path}")

# Add channel dimension: (N, H, W) → (N, 1, H, W)
train_dataset = TensorDataset(
    Xtrain_norm.unsqueeze(1), Ytrain_norm.unsqueeze(1)
)
val_dataset = TensorDataset(
    Xval_norm.unsqueeze(1), Yval_norm.unsqueeze(1)
)

torch.save(SEEDS, os.path.join(BASE_CHECKPOINT_DIR, "seeds_used.pth"))

# ── Training loop over seeds ──────────────────────────────────────────────────

training_times = []
criterion = nn.MSELoss()

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"  Training DDPM (16×) — seed {seed} / {SEEDS[-1]}")
    print(f"{'='*60}")

    set_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )

    seed_dir = os.path.join(BASE_CHECKPOINT_DIR, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # scale_factor=16 adds a 4th Upsample step in cond_up (8→16→32→64→128)
    model     = build_conditional_unet(scale_factor=SCALE_FACTOR).to(DEVICE)
    diffusion = Diffusion(T=T, device=DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    train_losses, val_losses = [], []
    best_val_loss   = float("inf")
    best_model_path = os.path.join(seed_dir, "best_model.pth")
    start_time      = time.time()

    for epoch in range(1, EPOCHS + 1):

        # ── Training ──────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0
        for x_lr, x_hr in train_loader:
            x_lr = x_lr.to(DEVICE, non_blocking=True)
            x_hr = x_hr.to(DEVICE, non_blocking=True)
            t          = diffusion.sample_timesteps(x_hr.size(0))
            x_t, noise = diffusion.add_noise(x_hr, t)
            pred_noise = model(x_t, t, x_lr)
            loss = criterion(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0
        with torch.inference_mode():
            for x_lr, x_hr in val_loader:
                x_lr = x_lr.to(DEVICE, non_blocking=True)
                x_hr = x_hr.to(DEVICE, non_blocking=True)
                t          = diffusion.sample_timesteps(x_hr.size(0))
                x_t, noise = diffusion.add_noise(x_hr, t)
                pred_noise = model(x_t, t, x_lr)
                total_val_loss += criterion(pred_noise, noise).item()

        avg_val = total_val_loss / len(val_loader)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_model_path)

        print(f"  [Seed {seed}] Epoch {epoch:3d}/{EPOCHS} | "
              f"Train: {avg_train:.6f} | Val: {avg_val:.6f}"
              + (" ← best" if avg_val == best_val_loss else ""))

        if epoch % 10 == 0 or epoch == EPOCHS:
            torch.save(
                model.state_dict(),
                os.path.join(seed_dir, f"epoch_{epoch:03d}.pth"),
            )

    elapsed = time.time() - start_time
    training_times.append(elapsed)

    np.save(os.path.join(seed_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(seed_dir, "val_losses.npy"),   np.array(val_losses))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label="Training Loss",   linewidth=2)
    ax.plot(val_losses,   label="Validation Loss", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Noise Prediction MSE", fontsize=12)
    ax.set_title(f"DDPM (16×) Training Progress — Seed {seed}", fontsize=14)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(seed_dir, f"ddpm_loss_curve_seed_{seed}.png"), dpi=150)
    plt.close(fig)

    h, rem = divmod(elapsed, 3600)
    m, s   = divmod(rem, 60)
    with open(os.path.join(seed_dir, "training_info.txt"), "w") as f:
        f.write(f"Seed:                  {seed}\n")
        f.write(f"Training time:         {int(h):02d}:{int(m):02d}:{s:05.2f} (HH:MM:SS)\n")
        f.write(f"Final train loss:      {avg_train:.6f}\n")
        f.write(f"Best val loss:         {best_val_loss:.6f}\n")
        f.write(f"Best model:            {best_model_path}\n")

    print(f"  Seed {seed} done in {int(h):02d}:{int(m):02d}:{s:05.2f}")

# ── Summary report ────────────────────────────────────────────────────────────

total_time = sum(training_times)
avg_time   = total_time / len(SEEDS)

summary_path = os.path.join(BASE_CHECKPOINT_DIR, "training_summary.txt")
with open(summary_path, "w") as f:
    f.write("DDPM (16×) Multi-Seed Training Summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"Scale factor   : {SCALE_FACTOR}× (8×8 → 128×128)\n")
    f.write(f"Timesteps (T)  : {T}\n")
    f.write(f"Learning rate  : {LR}\n")
    f.write(f"Batch size     : {BATCH_SIZE}\n")
    f.write(f"Epochs         : {EPOCHS}\n")
    f.write(f"Seeds          : {SEEDS}\n\n")
    f.write("Per-seed training times:\n")
    for seed, t in zip(SEEDS, training_times):
        h, rem = divmod(t, 3600)
        m, s   = divmod(rem, 60)
        f.write(f"  Seed {seed:2d}: {int(h):02d}:{int(m):02d}:{s:05.2f}\n")
    h, rem = divmod(avg_time, 3600)
    m, s   = divmod(rem, 60)
    f.write(f"\nAverage per seed : {int(h):02d}:{int(m):02d}:{s:05.2f}\n")
    h, rem = divmod(total_time, 3600)
    m, s   = divmod(rem, 60)
    f.write(f"Total wall time  : {int(h):02d}:{int(m):02d}:{s:05.2f}\n")

print(f"\nSummary saved to: {summary_path}")
