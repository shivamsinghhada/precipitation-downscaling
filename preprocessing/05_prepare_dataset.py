"""
Script:      05_prepare_dataset.py
Description: Loads the three filtered regional precipitation arrays, concatenates
             them into a single dataset, creates paired low-resolution (LR) inputs
             via 8× block averaging (128×128 → 16×16), and splits the data into
             train / validation / test sets.

             The LR–HR pairs form the input–target pairs for model training:
               LR (16×16)  : low-resolution input  → X
               HR (128×128): high-resolution target → Y

             Split fractions (approximate):
               Train : ~54.7 %  (indices   0 – 19199)
               Val   :  ~9.4 %  (indices 19200 – 22503)
               Test  : ~35.9 %  (indices 22504 – end)

             The split indices were chosen to preserve temporal ordering
             (no random shuffling before splitting).

Inputs:      Filtered .npz files produced by 03_filter_dry_images.py
Outputs:     dataset_splits.npz in OUTPUT_DIR containing:
               Xtrain, Xval, Xtest  — LR arrays  (N, 16, 16)
               Ytrain, Yval, Ytest  — HR arrays  (N, 128, 128)

Usage:       python 05_prepare_dataset.py

Requirements: numpy  (see environment.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
FILTERED_DIR = "/path/to/ERA5_land/Filtered"   # output of script 03
OUTPUT_DIR   = "/path/to/ERA5_land/Dataset"    # output directory for splits
# ────────────────────────────────────────────────────────────────────────────

import os
import numpy as np

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load filtered regional arrays ────────────────────────────────────────────

regions = ["CentralPlains", "Northwest", "Northeast"]
arrays  = []

for region in regions:
    path = os.path.join(FILTERED_DIR, f"filtered_precip_{region}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}. Run script 03 first."
        )
    arr = np.load(path)["data"]   # shape (N_region, 128, 128)
    print(f"  {region}: {arr.shape[0]} images")
    arrays.append(arr)

# Concatenate all three regions along the sample axis
data_hr = np.concatenate(arrays, axis=0)   # (N_total, 128, 128)
print(f"\nCombined HR dataset: {data_hr.shape}")

# ── Block-average downscaling: 128×128 → 16×16 (factor 8) ───────────────────

def block_average(arr: np.ndarray, factor: int = 8) -> np.ndarray:
    """
    Downscale a 2-D spatial field by block averaging.

    Parameters
    ----------
    arr    : np.ndarray of shape (H, W)
    factor : int, downscaling factor (must divide H and W exactly)

    Returns
    -------
    np.ndarray of shape (H // factor, W // factor)
    """
    H, W = arr.shape
    if H % factor != 0 or W % factor != 0:
        raise ValueError(
            f"Spatial dimensions ({H}, {W}) must be divisible by factor {factor}."
        )
    return arr.reshape(H // factor, factor, W // factor, factor).mean(axis=(1, 3))


print("Creating LR fields via 8× block averaging (128→16)...")
data_lr = np.stack(
    [block_average(frame, factor=8) for frame in data_hr], axis=0
).astype(np.float32)                         # (N_total, 16, 16)
data_hr = data_hr.astype(np.float32)

print(f"HR shape : {data_hr.shape}")         # (N_total, 128, 128)
print(f"LR shape : {data_lr.shape}")         # (N_total,  16,  16)

# ── Train / validation / test split ──────────────────────────────────────────
# Indices preserve temporal ordering within each region (no pre-shuffle).
# Adjust these boundaries if your total sample count differs.

N = data_hr.shape[0]

# Approximate split: 55 % train | 9 % val | 36 % test
TRAIN_END = 19200
VAL_END   = 22504

if VAL_END > N:
    raise ValueError(
        f"Dataset has only {N} samples but VAL_END={VAL_END}. "
        "Reduce split indices or add more data."
    )

Xtrain, Ytrain = data_lr[:TRAIN_END],         data_hr[:TRAIN_END]
Xval,   Yval   = data_lr[TRAIN_END:VAL_END],  data_hr[TRAIN_END:VAL_END]
Xtest,  Ytest  = data_lr[VAL_END:],           data_hr[VAL_END:]

print(f"\nSplit summary (total = {N} samples):")
print(f"  Train : Xtrain {Xtrain.shape}, Ytrain {Ytrain.shape}")
print(f"  Val   : Xval   {Xval.shape},   Yval   {Yval.shape}")
print(f"  Test  : Xtest  {Xtest.shape},  Ytest  {Ytest.shape}")

# ── Save ─────────────────────────────────────────────────────────────────────

out_path = os.path.join(OUTPUT_DIR, "dataset_splits.npz")
np.savez_compressed(
    out_path,
    Xtrain=Xtrain, Ytrain=Ytrain,
    Xval=Xval,     Yval=Yval,
    Xtest=Xtest,   Ytest=Ytest,
)
print(f"\nSaved splits to: {out_path}")
print("Next step: run  06_train_unet.py")
