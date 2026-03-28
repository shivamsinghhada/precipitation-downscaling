"""
Script:      02_crop_regions.py
Description: Crops the continental-US ERA5-Land daily precipitation data
             into three 128×128-pixel regional tiles used in the study:
               1. Central Plains   (30.2–43.0°N, 105.4–92.6°W)
               2. Northwest        (36.6–49.4°N, 121.8–109.0°W)
               3. Northeast        (36.6–49.4°N,  90.4–77.6°W)

             ERA5-Land latitudes are stored in descending order; the slice
             direction is therefore reversed (lat_max → lat_min).

Inputs:      Annual daily NetCDF files produced by 01_era5land_hourly_to_daily.py
             Variable: daily_precip (time, latitude, longitude)
Outputs:     One NetCDF per region in CROP_DIR,
             e.g. era5land_precip_crop_CentralPlains.nc

Usage:       python 02_crop_regions.py

Requirements: xarray, glob  (see environment.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
DAILY_DIR = "/path/to/ERA5_land/Daily"   # output of script 01
CROP_DIR  = "/path/to/ERA5_land/Crops"  # output directory for cropped tiles
# ────────────────────────────────────────────────────────────────────────────

import os
import glob
import xarray as xr

os.makedirs(CROP_DIR, exist_ok=True)

# ── Region definitions ───────────────────────────────────────────────────────
# Each tuple: (lat_min, lat_max, lon_min, lon_max, label)
# Bounding boxes chosen to produce ~128×128 pixel tiles at ERA5-Land resolution
# (0.1°).  Verify exact output shapes with the printed diagnostics below.
TILES = [
    (30.2, 43.0, -105.4,  -92.6, "CentralPlains"),
    (36.6, 49.4, -121.8, -109.0, "Northwest"),
    (36.6, 49.4,  -90.4,  -77.6, "Northeast"),
]

# ── Load all annual files ────────────────────────────────────────────────────
daily_files = sorted(glob.glob(os.path.join(DAILY_DIR, "*.nc")))
if not daily_files:
    raise FileNotFoundError(
        f"No NetCDF files found in {DAILY_DIR}. "
        "Run script 01 first to generate daily files."
    )

print(f"Found {len(daily_files)} annual files in {DAILY_DIR}")
print("Loading ERA5-Land daily data (this may take a moment)...")
ds = xr.open_mfdataset(daily_files, combine="by_coords")
print(f"Loaded dataset: {dict(ds.dims)}\n")

# ── Crop and save each tile ──────────────────────────────────────────────────
for lat_min, lat_max, lon_min, lon_max, label in TILES:
    print(f"Cropping {label}  ({lat_min}–{lat_max}°N, {lon_min}–{lon_max}°E)...")

    # ERA5 latitudes are in descending order → slice(lat_max, lat_min)
    ds_crop = ds.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max),
    )

    n_lat = ds_crop.latitude.size
    n_lon = ds_crop.longitude.size
    print(f"  Tile shape: {ds_crop['daily_precip'].shape}  "
          f"(lat={n_lat}, lon={n_lon})")

    if n_lat != 128 or n_lon != 128:
        print(f"  WARNING: expected 128×128 but got {n_lat}×{n_lon}. "
              "Adjust bounding-box coordinates if needed.")

    out_file = os.path.join(CROP_DIR, f"era5land_precip_crop_{label}.nc")
    ds_crop.to_netcdf(out_file)
    print(f"  Saved: {out_file}")

print("\nAll tiles cropped and saved successfully.")
