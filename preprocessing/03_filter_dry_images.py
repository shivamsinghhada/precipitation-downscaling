"""
Script:      03_filter_dry_images.py
Description: Removes near-dry daily precipitation images from the three
             regional 128×128 tiles before model training.

             Filtering criteria (applied per daily image):
               1. Soft threshold: pixels below WET_THRESHOLD mm/day are set
                  to zero (treats trace/drizzle as dry).
               2. Coverage threshold: images with fewer than MIN_WET_PIXELS
                  wet pixels (> 0 after step 1) are discarded.

             MIN_WET_PIXELS = 164 corresponds to ≈1 % of the 128×128 = 16,384
             pixel domain, ensuring that only days with spatially meaningful
             precipitation are retained for training.

Inputs:      Cropped NetCDF tiles produced by 02_crop_regions.py
             Variable: daily_precip (time, latitude, longitude)
Outputs:     Three NumPy arrays (one per region) saved as compressed .npz:
               filtered_precip_CentralPlains.npz
               filtered_precip_Northwest.npz
               filtered_precip_Northeast.npz
             Each .npz contains key 'data' with shape (N_kept, 128, 128).

Usage:       python 03_filter_dry_images.py

Requirements: numpy, xarray  (see environment.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
CROP_DIR   = "/path/to/ERA5_land/Crops"    # output of script 02
OUTPUT_DIR = "/path/to/ERA5_land/Filtered" # output directory for .npz files

# Filtering thresholds
WET_THRESHOLD  = 1.0   # mm/day — pixels below this are treated as dry
MIN_WET_PIXELS = 164   # minimum wet pixels to keep an image
                       # 164 ≈ 1 % of 128×128 = 16,384 pixels
# ────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import xarray as xr

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Region file mapping ───────────────────────────────────────────────────────
REGIONS = {
    "CentralPlains": os.path.join(CROP_DIR, "era5land_precip_crop_CentralPlains.nc"),
    "Northwest":     os.path.join(CROP_DIR, "era5land_precip_crop_Northwest.nc"),
    "Northeast":     os.path.join(CROP_DIR, "era5land_precip_crop_Northeast.nc"),
}

# ── Process each region ──────────────────────────────────────────────────────
for region, nc_path in REGIONS.items():
    print(f"\nProcessing {region}...")

    if not os.path.exists(nc_path):
        raise FileNotFoundError(
            f"File not found: {nc_path}. Run script 02 first."
        )

    # Load data
    with xr.open_dataset(nc_path) as ds:
        arr = ds["daily_precip"].values  # shape: (time, 128, 128), float32, mm/day

    n_total = arr.shape[0]
    print(f"  Input images : {n_total}")

    # Step 1 — soft threshold: set sub-threshold pixels to zero
    arr_thresh = np.where(arr >= WET_THRESHOLD, arr, 0.0).astype("float32")

    # Step 2 — coverage filter: discard images with too few wet pixels
    wet_pixel_counts = np.sum(arr_thresh > 0, axis=(1, 2))
    keep_mask = wet_pixel_counts >= MIN_WET_PIXELS
    arr_clean = arr_thresh[keep_mask]

    n_kept = arr_clean.shape[0]
    pct_kept = 100.0 * n_kept / n_total
    print(f"  Kept images  : {n_kept} / {n_total}  ({pct_kept:.1f} %)")
    print(f"  Output shape : {arr_clean.shape}")

    # Save as compressed NumPy archive
    out_path = os.path.join(OUTPUT_DIR, f"filtered_precip_{region}.npz")
    np.savez_compressed(out_path, data=arr_clean)
    print(f"  Saved: {out_path}")

print("\nFiltering complete for all regions.")
