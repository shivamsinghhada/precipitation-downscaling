#!/usr/bin/env python3
# ============================================================
# Script: analysis/plot_DDPM_T100_T50_composite_figureS11.py
#
# Figure S11: DDPM accelerated inference diagnostics
#
# Rows:
#   Row 1: DDPM (T=100)
#   Row 2: DDPM (T=50)
#
# Columns:
#   (a,d) Exceedance probability
#   (b,e) Q-Q plot
#   (c,f) Radial power spectrum
#
# Notes:
#   - Uses revised 16x Northeast-only test split.
#   - Uses saved DDPM T100 and T50 predictions.
#   - No model loading or inference.
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# USER CONFIGURATION
# ============================================================

DATASET_PATH = "/scratch/wpa8me/ERA5_land/Dataset/dataset_splits_16x.npz"

PRED_DIRS = {
    "T100": "/scratch/wpa8me/New16x/predictions_multiseed_16x1/T100",
    "T50":  "/scratch/wpa8me/New16x/predictions_multiseed_16x1/T50",
}

OUT_DIR = "./FigureS11_DDPM_T100_T50_diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(
    OUT_DIR,
    "FigureS11_DDPM_T100_T50_exceedance_QQ_spectrum.png"
)

OUT_PDF = os.path.join(
    OUT_DIR,
    "FigureS11_DDPM_T100_T50_exceedance_QQ_spectrum.pdf"
)

SEEDS = list(range(1, 11))


# ============================================================
# SETTINGS
# ============================================================

N_SAMPLES = None

RAIN_THRESHOLD = 1.0
TAIL_THRESHOLD = 10.0

QQ_POINTS = 200
POWER_NSAMPLES = 250

DX_KM = 9.0

FIG_WIDTH = 18
FIG_HEIGHT = 10

FONTSIZE_LABEL = 16
FONTSIZE_TICK = 13
FONTSIZE_PANEL = 16
FONTSIZE_LEGEND = 14
FONTSIZE_ROW = 17

LW_TARGET = 2.7
LW_SEED = 1.35
ALPHA_SEED = 0.45

COLORS = {
    "Target (ERA5-Land)": "#2baad3",
    "T100": "#986bc5",
    "T50": "#ca5478",
}

plt.rcParams.update({
    "font.size": FONTSIZE_LABEL,
    "axes.labelsize": FONTSIZE_LABEL,
    "xtick.labelsize": FONTSIZE_TICK,
    "ytick.labelsize": FONTSIZE_TICK,
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
})


# ============================================================
# HELPERS
# ============================================================

def squeeze_hw(arr):
    arr = np.asarray(arr)

    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    arr = np.squeeze(arr)

    if arr.ndim != 3:
        raise ValueError(f"Expected shape (N,H,W), got {arr.shape}")

    return arr.astype(np.float32)


def load_target(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset split not found: {dataset_path}")

    data = np.load(dataset_path)
    y = squeeze_hw(data["Ytest"])

    if N_SAMPLES is not None:
        y = y[:N_SAMPLES]

    return y


def load_seed_predictions(pred_dir, seeds, n_samples=None):
    out = {}

    for seed in seeds:

        path = os.path.join(
            pred_dir,
            f"Xtest_predictions_{seed}.npy"
        )

        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue

        arr = squeeze_hw(np.load(path))

        if n_samples is not None:
            arr = arr[:n_samples]

        out[seed] = arr

        print(f"Loaded {os.path.basename(pred_dir)} seed {seed}: {arr.shape}")

    if len(out) == 0:
        raise RuntimeError(f"No predictions loaded from {pred_dir}")

    return out


def filter_flat(arr, threshold):
    arr = np.asarray(arr).ravel()
    arr = arr[np.isfinite(arr)]

    return arr[arr > threshold]


def exceedance_curve(arr, threshold):
    vals = filter_flat(arr, threshold)

    if vals.size == 0:
        return None, None

    vals = np.sort(vals)

    F = np.arange(1, vals.size + 1) / vals.size

    return vals, 1.0 - F


def quantile_curve(arr, threshold=1.0, n_points=200):
    vals = filter_flat(arr, threshold)

    if vals.size == 0:
        return None

    q = np.linspace(0, 1, n_points)

    return np.quantile(vals, q)


def radial_power_spectrum(image):
    image = np.asarray(image, dtype=np.float64)

    h, w = image.shape

    fft_image = np.fft.fftshift(np.fft.fft2(image))
    power = np.abs(fft_image) ** 2

    cy, cx = h // 2, w // 2

    yy, xx = np.indices((h, w))

    radius = np.sqrt(
        (xx - cx) ** 2 +
        (yy - cy) ** 2
    ).astype(int)

    radial_sum = np.bincount(radius.ravel(), power.ravel())
    radial_count = np.bincount(radius.ravel())

    spectrum = radial_sum / (radial_count + 1e-12)

    return spectrum


def mean_power_spectrum(images, n_samples=250):
    images = squeeze_hw(images)

    n = min(n_samples, images.shape[0])

    spectra = [
        radial_power_spectrum(images[i])
        for i in range(n)
    ]

    min_len = min(len(spectrum) for spectrum in spectra)

    spectra = np.asarray([
        spectrum[:min_len]
        for spectrum in spectra
    ])

    return np.mean(spectra, axis=0)


def to_db(power):
    return 10.0 * np.log10(
        np.clip(power, 1e-12, None)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Loading target and DDPM T100/T50 predictions")
    print("=" * 70)

    ytest = load_target(DATASET_PATH)

    print(f"Target: {ytest.shape}")

    preds = {}

    for tag, pred_dir in PRED_DIRS.items():

        print("\n" + "=" * 70)
        print(f"Loading {tag}")
        print("=" * 70)

        preds[tag] = load_seed_predictions(
            pred_dir=pred_dir,
            seeds=SEEDS,
            n_samples=N_SAMPLES
        )

    # --------------------------------------------------------
    # Align common sample count
    # --------------------------------------------------------
    n_common = min(
        ytest.shape[0],
        min(arr.shape[0] for arr in preds["T100"].values()),
        min(arr.shape[0] for arr in preds["T50"].values())
    )

    ytest = ytest[:n_common]

    for tag in preds:
        for seed in preds[tag]:
            preds[tag][seed] = preds[tag][seed][:n_common]

    print(f"Common N: {n_common}")

    # --------------------------------------------------------
    # Target curves
    # --------------------------------------------------------
    target_ex_x, target_ex_y = exceedance_curve(
        ytest,
        TAIL_THRESHOLD
    )

    target_q = quantile_curve(
        ytest,
        threshold=RAIN_THRESHOLD,
        n_points=QQ_POINTS
    )

    target_power = mean_power_spectrum(
        ytest,
        n_samples=POWER_NSAMPLES
    )

    target_power_plot = target_power[1:]

    k_power = np.arange(
        1,
        len(target_power_plot) + 1
    )

    h = ytest.shape[1]
    domain_length_km = h * DX_KM
    wavelength_km = domain_length_km / k_power

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------
    print("\nCreating Figure S11")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        constrained_layout=False
    )

    panel_labels = {
        (0, 0): "(a)",
        (0, 1): "(b)",
        (0, 2): "(c)",
        (1, 0): "(d)",
        (1, 1): "(e)",
        (1, 2): "(f)",
    }

    for row_idx, tag in enumerate(["T100", "T50"]):

        color = COLORS[tag]

        # ====================================================
        # Exceedance probability
        # ====================================================
        ax = axes[row_idx, 0]

        ax.plot(
            target_ex_x,
            target_ex_y,
            color=COLORS["Target (ERA5-Land)"],
            linewidth=LW_TARGET,
            zorder=10
        )

        for seed in SEEDS:
            if seed not in preds[tag]:
                continue

            x, y = exceedance_curve(
                preds[tag][seed],
                TAIL_THRESHOLD
            )

            if x is None:
                continue

            ax.plot(
                x,
                y,
                color=color,
                linewidth=LW_SEED,
                alpha=ALPHA_SEED
            )

        ax.set_yscale("log")
        ax.set_xlim(TAIL_THRESHOLD, 280)
        ax.set_ylim(1e-8, 1.0)

        ax.set_xlabel("Precipitation (mm/day)", labelpad=5)
        ax.set_ylabel("Exceedance probability", labelpad=5)
        ax.grid(False)

        # ====================================================
        # Q-Q plot
        # ====================================================
        ax = axes[row_idx, 1]

        ax.plot(
            target_q,
            target_q,
            color="black",
            linestyle="--",
            linewidth=1.8,
            zorder=10
        )

        for seed in SEEDS:
            if seed not in preds[tag]:
                continue

            pred_q = quantile_curve(
                preds[tag][seed],
                threshold=RAIN_THRESHOLD,
                n_points=QQ_POINTS
            )

            if pred_q is None:
                continue

            ax.plot(
                target_q,
                pred_q,
                color=color,
                linewidth=LW_SEED,
                alpha=ALPHA_SEED
            )

        ax.set_xlabel("Target quantiles", labelpad=5)
        ax.set_ylabel("Predicted quantiles", labelpad=5)
        ax.grid(False)

        # ====================================================
        # Power spectrum
        # ====================================================
        ax = axes[row_idx, 2]

        ax.plot(
            wavelength_km,
            to_db(target_power_plot),
            color=COLORS["Target (ERA5-Land)"],
            linewidth=LW_TARGET,
            zorder=10
        )

        for seed in SEEDS:
            if seed not in preds[tag]:
                continue

            pred_power = mean_power_spectrum(
                preds[tag][seed],
                n_samples=POWER_NSAMPLES
            )

            pred_power_plot = pred_power[1:len(target_power)]

            ax.plot(
                wavelength_km,
                to_db(pred_power_plot),
                color=color,
                linewidth=LW_SEED,
                alpha=ALPHA_SEED
            )

        ax.set_xscale("log")
        ax.invert_xaxis()

        ax.set_xlabel("Wavelength (km)", labelpad=5)
        ax.set_ylabel("Power (dB)", labelpad=5)
        ax.grid(False)

        # ====================================================
        # Row label
        # ====================================================
        axes[row_idx, 0].text(
            -0.28,
            0.5,
            f"DDPM ({tag.replace('T', 'T=')})",
            transform=axes[row_idx, 0].transAxes,
            fontsize=FONTSIZE_ROW,
            fontweight="normal",
            rotation=90,
            va="center",
            ha="center"
        )

    # --------------------------------------------------------
    # Panel labels
    # --------------------------------------------------------
    for (i, j), label in panel_labels.items():
        axes[i, j].text(
            0.015,
            0.98,
            label,
            transform=axes[i, j].transAxes,
            fontsize=FONTSIZE_PANEL,
            fontweight="normal",
            va="top",
            ha="left"
        )

    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------
    handles = [
        Line2D(
            [0], [0],
            color=COLORS["Target (ERA5-Land)"],
            lw=LW_TARGET,
            label="Target (ERA5-Land)"
        ),
        Line2D(
            [0], [0],
            color=COLORS["T100"],
            lw=2.2,
            label="DDPM (T=100)"
        ),
        Line2D(
            [0], [0],
            color=COLORS["T50"],
            lw=2.2,
            label="DDPM (T=50)"
        ),
        Line2D(
            [0], [0],
            color="black",
            lw=1.8,
            linestyle="--",
            label="1:1"
        ),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=FONTSIZE_LEGEND,
        bbox_to_anchor=(0.5, 0.015)
    )

    plt.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.965,
        bottom=0.125,
        wspace=0.26,
        hspace=0.30
    )

    fig.savefig(
        OUT_PNG,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.03
    )

    fig.savefig(
        OUT_PDF,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.03
    )

    plt.show()

    print("\nSaved:")
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
