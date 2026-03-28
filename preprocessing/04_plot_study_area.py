"""
Script:      04_plot_study_area.py
Description: Produces a publication-quality map of the three study regions
             overlaid on a CONUS basemap.  The three 128×128-pixel ERA5-Land
             tiles are drawn as semi-transparent coloured rectangles with
             bold labels.

             Region colours (hex):
               Central Plains : #2baad3  (blue)
               Northwest      : #c57a3e  (orange-brown)
               Northeast      : #986bc5  (purple)

Inputs:      None — uses bounding-box coordinates defined below.
Outputs:     study_area_map.png  (300 dpi) in FIGURE_DIR

Usage:       python 04_plot_study_area.py

Requirements: matplotlib, cartopy  (see environment.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
FIGURE_DIR = "/path/to/figures"   # output directory for the PNG
# ────────────────────────────────────────────────────────────────────────────

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

os.makedirs(FIGURE_DIR, exist_ok=True)

# ── Region definitions ───────────────────────────────────────────────────────
# Each tuple: (lat_min, lat_max, lon_min, lon_max, label, hex_color)
TILES = [
    (30.2, 43.0, -105.4,  -92.6, "Central Plains", "#2baad3"),
    (36.6, 49.4, -121.8, -109.0, "Northwest",      "#c57a3e"),
    (36.6, 49.4,  -90.4,  -77.6, "Northeast",      "#986bc5"),
]

# ── Figure setup ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# Basemap layers
ax.add_feature(cfeature.LAND,      facecolor="white",  edgecolor="none")
ax.add_feature(cfeature.OCEAN,     facecolor="white")
ax.add_feature(cfeature.BORDERS,   linewidth=0.6)
ax.add_feature(cfeature.STATES,    linewidth=0.4)
ax.add_feature(cfeature.COASTLINE, linewidth=0.7)

# CONUS extent
ax.set_extent([-125, -66.5, 25, 50], crs=ccrs.PlateCarree())

# ── Draw tiles ───────────────────────────────────────────────────────────────
for lat_min, lat_max, lon_min, lon_max, label, color in TILES:
    rect = patches.Rectangle(
        (lon_min, lat_min),
        lon_max - lon_min,
        lat_max - lat_min,
        linewidth=1.8,
        edgecolor=color,
        facecolor=color,
        alpha=0.25,
        transform=ccrs.PlateCarree(),
    )
    ax.add_patch(rect)

    ax.text(
        (lon_min + lon_max) / 2,
        (lat_min + lat_max) / 2,
        label,
        color=color,
        fontsize=12,
        ha="center",
        va="center",
        fontweight="bold",
        transform=ccrs.PlateCarree(),
    )

# ── Save ─────────────────────────────────────────────────────────────────────
out_png = os.path.join(FIGURE_DIR, "study_area_map.png")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_png}")
