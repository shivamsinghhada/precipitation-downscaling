"""
Script:      01_era5land_hourly_to_daily.py
Description: Converts ERA5-Land hourly precipitation ZIP archives into
             annual daily-total NetCDF files (mm/day).

             ERA5-Land short-forecast convention:
               - The 00:00 UTC value equals the full accumulated total
                 for the PREVIOUS calendar day.
               - Values at 01–23 UTC are cumulative since that day's 00:00.
             Therefore: daily total for day D = tp[00Z of day D+1].

Inputs:      Monthly ZIP files containing hourly ERA5-Land NetCDF,
             named according to ZIP_PATTERN in the USER CONFIGURATION block.
Outputs:     One NetCDF per year in DAILY_DIR,
             e.g. era5land_daily_precip_mm_1980.nc
             Variables: daily_precip (float32, mm/day)
             Coordinates: time (daily), latitude, longitude

Usage:       python 01_era5land_hourly_to_daily.py

Requirements: numpy, xarray  (see environment.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
# Set these paths to match your local directory structure.
HOURLY_DIR = "/path/to/ERA5_land/Hourly"     # directory holding monthly ZIPs
DAILY_DIR  = "/path/to/ERA5_land/Daily"      # output directory for annual files

# File name templates (change only if your files use a different naming scheme)
ZIP_PATTERN = "era5_hourly_precip_{ym}.zip"   # e.g. era5_hourly_precip_1980_01.zip
NC_PATTERN  = "era5land_hourly_tp_{ym}.nc"    # normalized monthly NC we create

# Processing range (inclusive)
YEAR_START = 1980
YEAR_END   = 2014
# ────────────────────────────────────────────────────────────────────────────

import os
import zipfile
import numpy as np
import xarray as xr

os.makedirs(DAILY_DIR, exist_ok=True)


# ── Helper functions ─────────────────────────────────────────────────────────

def ensure_month_nc(year: int, month: int) -> str:
    """
    Return the path to the normalized monthly NetCDF for (year, month).
    If it does not yet exist, extract the first *.nc from the corresponding
    ZIP archive and rename it to the standard NC_PATTERN filename.
    """
    ym = f"{year}_{month:02d}"
    nc_path = os.path.join(HOURLY_DIR, NC_PATTERN.format(ym=ym))
    if os.path.exists(nc_path):
        return nc_path

    zpath = os.path.join(HOURLY_DIR, ZIP_PATTERN.format(ym=ym))
    if not os.path.exists(zpath):
        raise FileNotFoundError(f"Missing ZIP: {zpath}")

    with zipfile.ZipFile(zpath, "r") as z:
        nc_members = [n for n in z.namelist() if n.lower().endswith(".nc")]
        if not nc_members:
            raise FileNotFoundError(f"No .nc file found inside {zpath}")
        extracted = z.extract(nc_members[0], path=HOURLY_DIR)

    os.replace(extracted, nc_path)
    return nc_path


def open_month(path: str) -> xr.Dataset:
    """
    Open a monthly hourly NetCDF, normalising the time coordinate name.
    ERA5 CDS downloads sometimes use 'valid_time' instead of 'time'.
    """
    ds = xr.open_dataset(path, decode_times=True)
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds


def daily_from_next00z(tp: xr.DataArray) -> xr.DataArray:
    """
    Compute daily precipitation totals from ERA5-Land hourly accumulations.

    ERA5-Land convention: the 00:00 UTC step holds the 24-hour accumulation
    for the PREVIOUS day.  Selecting every 00Z frame and shifting back one
    day therefore gives one clean daily total per calendar day.

    Parameters
    ----------
    tp : xr.DataArray
        Hourly total_precipitation in metres (m).

    Returns
    -------
    xr.DataArray
        Daily precipitation in mm/day, named 'daily_precip'.
    """
    tp = tp.clip(min=0)                                    # remove floating-point negatives
    tp00 = tp.sel(time=tp.time.dt.hour == 0)               # select all 00Z time steps
    day_tot = tp00.shift(time=-1)                          # shift 00Z value to previous day
    day_tot = day_tot.assign_coords(
        time=day_tot.time - np.timedelta64(1, "D")
    )
    day_mm = (day_tot * 1000.0).astype("float32")          # convert m → mm
    day_mm.name = "daily_precip"
    day_mm.attrs.update(
        units="mm/day",
        long_name="Daily total precipitation (derived from 00Z of next day)",
        source="ERA5-Land hourly reanalysis, ECMWF",
    )
    return day_mm


# ── Main processing function ─────────────────────────────────────────────────

def process_year(year: int):
    """
    Build the annual daily-precipitation NetCDF for a single year.
    Requires the 12 monthly ZIPs for that year plus January of year+1
    (needed to derive the daily total for 31 December).
    """
    out_path = os.path.join(DAILY_DIR, f"era5land_daily_precip_mm_{year}.nc")
    if os.path.exists(out_path):
        print(f"  Already exists, skipping: {out_path}")
        return

    # Collect monthly NC paths: Jan–Dec of this year + Jan of next year
    paths = [ensure_month_nc(year, m) for m in range(1, 13)]
    try:
        paths.append(ensure_month_nc(year + 1, 1))
    except FileNotFoundError as e:
        print(f"  WARNING: {year + 1}-01 not found; 31 Dec {year} will be missing. ({e})")

    # Open and concatenate all monthly files
    ds_list = [open_month(p) for p in paths]
    ds = xr.concat(ds_list, dim="time").sortby("time")

    # Remove duplicate time steps (can occur at monthly boundaries)
    time_idx = ds.indexes["time"]
    if hasattr(time_idx, "duplicated") and time_idx.duplicated().any():
        ds = ds.sel(time=~time_idx.duplicated())

    # Identify the precipitation variable (name varies by CDS version)
    var = next((v for v in ("tp", "total_precipitation") if v in ds.data_vars), None)
    if var is None:
        raise ValueError(
            f"Precipitation variable not found in {paths[0]}. "
            "Expected 'tp' or 'total_precipitation'."
        )

    # Compute daily totals and trim to the target year
    daily = daily_from_next00z(ds[var])
    daily = daily.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))

    # Save with lossless compression
    encoding = {
        "daily_precip": {"zlib": True, "complevel": 4, "dtype": "float32"}
    }
    daily.to_netcdf(out_path, encoding=encoding)
    print(f"  Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Processing ERA5-Land daily precipitation for {YEAR_START}–{YEAR_END}...\n")
    for yr in range(YEAR_START, YEAR_END + 1):
        print(f"Year {yr}:")
        process_year(yr)
    print("\nDone.")
