"""
Module:      analysis/plot_utils.py
Description: Shared plotting utilities used across all analysis scripts.

             Defines:
               COLORS        — consistent model color palette
               custom_cmap   — precipitation colormap (white → blue → green → yellow → red)
               squeeze_hw()  — array shape normalisation helper
               flatten_pos() — extract positive (wet) pixel values

             Import in any analysis script with:
               from analysis.plot_utils import COLORS, custom_cmap, squeeze_hw, flatten_pos

Requirements: numpy, matplotlib  (see environment_tf.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── Color palette ─────────────────────────────────────────────────────────────
# Consistent across all figures in the manuscript.

COLORS = {
    "Observed":   "#2baad3",   # blue
    "U-Net":      "#c57a3e",   # orange-brown
    "WGAN":       "#986bc5",   # purple
    "DDPM":       "#ca5478",   # pink-red
    "DDPM-T500":  "#ca5478",   # same as DDPM (T=500 experiment)
    "DDPM-T100":  "#73a450",   # green  (T=100 experiment, for comparison figure)
}


# ── Precipitation colormap ────────────────────────────────────────────────────
# Transitions: white (dry) → light-blue → blue → cyan → green → yellow → red
# Suitable for precipitation fields where 0 mm/day should appear white.

_precip_colors = [
    (1.00, 1.00, 1.00),   # 0.0  white     — dry
    (0.80, 0.90, 1.00),   # 0.1  pale blue
    (0.40, 0.70, 1.00),   # 0.25 light blue
    (0.00, 0.45, 0.80),   # 0.45 blue
    (0.00, 0.70, 0.50),   # 0.60 cyan-green
    (0.50, 0.85, 0.00),   # 0.75 yellow-green
    (1.00, 0.75, 0.00),   # 0.87 amber
    (1.00, 0.30, 0.00),   # 0.94 orange
    (0.80, 0.00, 0.00),   # 1.0  dark red
]

custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    "precip", _precip_colors, N=256
)


# ── Array helpers ─────────────────────────────────────────────────────────────

def squeeze_hw(arr: np.ndarray) -> np.ndarray:
    """
    Normalise array shape to (N, H, W) float32.

    Removes a trailing channel dimension of size 1 if present,
    i.e. (N, H, W, 1) → (N, H, W).
    """
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def flatten_pos(arr: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """
    Flatten arr to 1-D and return only finite values above threshold.

    Parameters
    ----------
    arr       : array of any shape
    threshold : float, values <= threshold are excluded (default 1.0 mm/day)

    Returns
    -------
    1-D float32 array of values above threshold
    """
    x = np.ravel(np.asarray(arr, dtype=np.float32))
    x = x[np.isfinite(x)]
    return x[x > threshold]
