"""
Script:      05b_prepare_dataset_16x.py
Description: Prepares LR–HR pairs for the 16× downscaling experiment
             (8×8 → 128×128).

             Steps:
               1. Load the three filtered regional arrays (128×128 HR fields)
                  produced by 03_filter_dry_images.py.
               2. Concatenate the three regions into a single dataset.
               3. Create 8×8 LR inputs by 16× block averaging (128 → 8 pixels).
               4. Split into train / validation / test sets preserving
                  temporal ordering (no pre-shuffle).

Inputs:      Filtered .npz files produced by 03_filter_dry_images.py
             (same files used by the 8× experiment in 05_prepare_dataset.py)
Outputs:     dataset_splits_16x.npz in OUTPUT_DIR containing:
               Xtrain, Xval, Xtest  — LR arrays  (N, 8, 8)
               Ytrain, Yval, Ytest  — HR arrays  (N, 128, 128)

Usage:       python 05b_prepare_dataset_16x.py

Requirements: numpy  (see environment.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
FILTERED_DIR = "/path/to/ERA5_land/Filtered"   # output of script 03
OUTPUT_DIR   = "/path/to/ERA5_land/Dataset"    # output directory (same as 05)
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

data_hr = np.concatenate(arrays, axis=0).astype(np.float32)  # (N_total, 128, 128)
print(f"\nCombined HR dataset: {data_hr.shape}")

# ── Block-average downscaling: 128×128 → 8×8 (factor 16) ────────────────────

def block_average(arr: np.ndarray, factor: int) -> np.ndarray:
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


print("Creating LR fields via 16× block averaging (128→8)...")
data_lr = np.stack(
    [block_average(frame, factor=16) for frame in data_hr], axis=0
).astype(np.float32)                           # (N_total, 8, 8)

print(f"HR shape : {data_hr.shape}")           # (N_total, 128, 128)
print(f"LR shape : {data_lr.shape}")           # (N_total,   8,   8)

# ── Train / validation / test split ──────────────────────────────────────────
# Uses the same index boundaries as the 8× experiment (05_prepare_dataset.py)
# so results are directly comparable (same samples in each split).

N = data_hr.shape[0]

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

out_path = os.path.join(OUTPUT_DIR, "dataset_splits_16x.npz")
np.savez_compressed(
    out_path,
    Xtrain=Xtrain, Ytrain=Ytrain,
    Xval=Xval,     Yval=Yval,
    Xtest=Xtest,   Ytest=Ytest,
)
print(f"\nSaved splits to: {out_path}")
print("Next step: run  08_train_unet_16x.py")
