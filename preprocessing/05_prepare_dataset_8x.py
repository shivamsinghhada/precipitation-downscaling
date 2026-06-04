"""
Script:      05_prepare_dataset.py

Description:
    Loads the three filtered regional precipitation arrays, concatenates them
    in the order:

        CentralPlains → Northwest → Northeast

    creates paired low-resolution inputs using 8× block averaging
    (128×128 → 16×16), and applies the original index-based split.

    Important:
        The test set starts at index 22772, which is exactly where the
        Northeast region begins after concatenating:

            CentralPlains = 11025 samples
            Northwest     = 11747 samples
            Northeast     = 12348 samples

            11025 + 11747 = 22772

    Therefore, the test set contains Northeast samples only.

Splits:
    Train : indices 0     – 19199
    Val   : indices 19200 – 22503
    Gap   : indices 22504 – 22771
    Test  : indices 22772 – 35119

Outputs:
    dataset_splits.npz containing:
        Xtrain, Xval, Xtest
        Ytrain, Yval, Ytest
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
FILTERED_DIR = "/path/to/ERA5_land/Filtered"
OUTPUT_DIR   = "/path/to/ERA5_land/Dataset"

DOWNSCALE_FACTOR = 8   # 128×128 → 16×16
# ────────────────────────────────────────────────────────────────────────────

import os
import numpy as np

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Fixed regional order ────────────────────────────────────────────────────
regions = ["CentralPlains", "Northwest", "Northeast"]
arrays = []

print("\nLoading filtered regional arrays...")

for region in regions:
    path = os.path.join(FILTERED_DIR, f"filtered_precip_{region}.npz")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}. Run 03_filter_dry_images.py first."
        )

    arr = np.load(path)["data"].astype(np.float32)
    print(f"  {region}: {arr.shape}")

    arrays.append(arr)


# ── Concatenate regions ─────────────────────────────────────────────────────
data_hr = np.concatenate(arrays, axis=0).astype(np.float32)

n_central = arrays[0].shape[0]
n_northwest = arrays[1].shape[0]
n_northeast = arrays[2].shape[0]

central_start = 0
northwest_start = n_central
northeast_start = n_central + n_northwest

print("\nRegional boundaries after concatenation:")
print(f"  CentralPlains: {central_start} to {northwest_start - 1}")
print(f"  Northwest    : {northwest_start} to {northeast_start - 1}")
print(f"  Northeast    : {northeast_start} to {data_hr.shape[0] - 1}")

print(f"\nCombined HR dataset shape: {data_hr.shape}")


# ── Block-average downscaling ───────────────────────────────────────────────
def block_average_batch(arr: np.ndarray, factor: int = 8) -> np.ndarray:
    """
    Downscale a batch of 2-D fields using block averaging.

    Input shape:
        arr: (N, H, W)

    Output shape:
        (N, H/factor, W/factor)
    """
    N, H, W = arr.shape

    if H % factor != 0 or W % factor != 0:
        raise ValueError(
            f"Spatial dimensions ({H}, {W}) must be divisible by factor {factor}."
        )

    return arr.reshape(
        N,
        H // factor,
        factor,
        W // factor,
        factor
    ).mean(axis=(2, 4))


print(f"\nCreating LR fields using {DOWNSCALE_FACTOR}× block averaging...")
data_lr = block_average_batch(data_hr, factor=DOWNSCALE_FACTOR).astype(np.float32)

print(f"HR shape: {data_hr.shape}")
print(f"LR shape: {data_lr.shape}")


# ── Original split indices ──────────────────────────────────────────────────
TRAIN_END = 19200
VAL_END = 22504
TEST_START = 22772

N = data_hr.shape[0]

if TEST_START != northeast_start:
    raise ValueError(
        f"TEST_START={TEST_START}, but Northeast starts at {northeast_start}. "
        "Check regional sample counts or concatenation order."
    )

if N != 35120:
    print(
        f"\nWarning: Total sample count is {N}, not 35120. "
        "Proceeding with the available data, but please verify consistency."
    )

if TEST_START > N:
    raise ValueError(
        f"TEST_START={TEST_START} exceeds total sample count N={N}."
    )


# ── Apply split ─────────────────────────────────────────────────────────────
Xtrain = data_lr[0:TRAIN_END]
Xval   = data_lr[TRAIN_END:VAL_END]
Xtest  = data_lr[TEST_START:N]

Ytrain = data_hr[0:TRAIN_END]
Yval   = data_hr[TRAIN_END:VAL_END]
Ytest  = data_hr[TEST_START:N]


# ── Summary ─────────────────────────────────────────────────────────────────
print("\nFinal split summary:")
print(f"  Xtrain: {Xtrain.shape}")
print(f"  Xval  : {Xval.shape}")
print(f"  Xtest : {Xtest.shape}")

print(f"  Ytrain: {Ytrain.shape}")
print(f"  Yval  : {Yval.shape}")
print(f"  Ytest : {Ytest.shape}")

print("\nSplit interpretation:")
print("  Train : CentralPlains + part of Northwest")
print("  Val   : Northwest only")
print("  Gap   : remaining Northwest samples, not used")
print("  Test  : Northeast only")


# ── Save ────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "dataset_splits.npz")

np.savez_compressed(
    out_path,
    Xtrain=Xtrain,
    Ytrain=Ytrain,
    Xval=Xval,
    Yval=Yval,
    Xtest=Xtest,
    Ytest=Ytest,
)

print(f"\nSaved dataset splits to: {out_path}")
print("Next step: run 06_train_unet.py")