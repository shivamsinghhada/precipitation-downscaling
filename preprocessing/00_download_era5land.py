"""
Script:      00_download_era5land.py
Description: Downloads ERA5-Land hourly total precipitation from the
             Copernicus Climate Data Store (CDS) for the CONUS domain.

             Output files are saved as monthly ZIP archives named:
               era5_hourly_precip_{YYYY_MM}.zip
             which is exactly the format expected by the next step:
               01_era5land_hourly_to_daily.py

             The ZIPs are intentionally kept on disk so that script 01 can
             extract and normalise them in its own controlled way.  Do NOT
             manually unzip or rename the files.

Prerequisites:
             1. Create a free CDS account: https://cds.climate.copernicus.eu
             2. Install the CDS API client:
                  pip install cdsapi
             3. Create the credentials file ~/.cdsapirc with:
                  url: https://cds.climate.copernicus.eu/api
                  key: <your-personal-API-key>
                (See https://cds.climate.copernicus.eu/how-to-api for details)

Outputs:     Monthly ZIP files in HOURLY_DIR,
             e.g. era5_hourly_precip_1980_01.zip

Usage:       python 00_download_era5land.py

             Downloads ~35 years × 12 months = 420 requests.
             CDS rate-limits to ~10 active requests; expect this to take
             several days depending on queue length.  Already-downloaded
             months are automatically skipped.

Requirements: cdsapi  (pip install cdsapi)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
HOURLY_DIR = "/path/to/ERA5_land/Hourly"  # must match HOURLY_DIR in script 01

# Spatial bounding box: [North, West, South, East]  (CONUS + buffer)
AREA = [50, -125, 24, -60]

# Year range — must match YEAR_START/YEAR_END in script 01
YEAR_START = 1980
YEAR_END   = 2014
# ────────────────────────────────────────────────────────────────────────────

import os
import cdsapi

os.makedirs(HOURLY_DIR, exist_ok=True)

client = cdsapi.Client()

for year in range(YEAR_START, YEAR_END + 1):
    for month in range(1, 13):
        ym_str   = f"{year}_{month:02d}"
        zip_path = os.path.join(HOURLY_DIR, f"era5_hourly_precip_{ym_str}.zip")

        # Skip if ZIP already downloaded
        if os.path.exists(zip_path):
            print(f"  Already exists, skipping: {zip_path}")
            continue

        print(f"Downloading ERA5-Land hourly data for {ym_str}...")
        client.retrieve(
            "reanalysis-era5-land",
            {
                "variable":    "total_precipitation",
                "year":        str(year),
                "month":       f"{month:02d}",
                "day":         [f"{d:02d}" for d in range(1, 32)],
                "time":        [f"{h:02d}:00" for h in range(24)],
                "area":        AREA,
                "data_format": "netcdf",          # 'format' is deprecated in CDS API v2
                "download_format": "zip",
            },
            zip_path,
        )
        print(f"  Saved: {zip_path}")

print("\nAll downloads complete (or already present).")
print(f"Next step: run  01_era5land_hourly_to_daily.py")
