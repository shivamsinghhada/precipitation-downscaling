#!/usr/bin/env python3
# ============================================================
# Script: analysis/plot_temporal_correlation_figure10.py
#
# Figure 10: Temporal correlation diagnostics
#
# Panels:
#   (a) Pixel-wise temporal correlation distribution
#   (b) Bias in temporal correlation
#   (c) RMSE in temporal correlation
#
# Notes:
#   - Uses the revised 16x Northeast-only test split.
#   - Uses the last/first N_SAMPLES consecutive test samples as provided.
#   - No bold text is used.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER CONFIGURATION
# ============================================================

DATASET_PATH = "/scratch/wpa8me/ERA5_land/Dataset/dataset_splits_16x.npz"

UNET_PATH = (
    "./ensemble_generation_best_models_NE/"
    "Xtest_predictions_unet_seed5_samples10.npy"
)

WGAN_PATH = (
    "./ensemble_generation_best_models_NE/"
    "Xtest_predictions_wgan_seed3_samples10.npy"
)

DDPM_PATH = (
    "./predictions_multisample_16x1_bestseed/T500/Temporal/"
    "Xtest_predictions_seed1_samples10.npy"
)

OUT_DIR = "./Figure10_temporal_correlation"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, "Figure10_temporal_correlation.png")
OUT_PDF = os.path.join(OUT_DIR, "Figure10_temporal_correlation.pdf")


# ============================================================
# SETTINGS
# ============================================================

N_SAMPLES = 1000
MAX_LAG = 5
GEN_ID = 0

FIG_WIDTH = 15.5
FIG_HEIGHT = 4.8

FONTSIZE_LABEL = 13
FONTSIZE_TICK = 11
FONTSIZE_PANEL = 13
FONTSIZE_LEGEND = 11


# ============================================================
# HELPERS
# ============================================================

def to_nhw(arr, name="array"):
    arr = np.asarray(arr)

    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    arr = np.squeeze(arr)

    if arr.ndim != 3:
        raise ValueError(f"{name} must be (N,H,W), got {arr.shape}")

    return arr.astype(np.float32)


def to_nshw(arr, name="ensemble"):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim != 4:
        raise ValueError(f"{name} must be (N,S,H,W), got {arr.shape}")

    return arr.astype(np.float32)


def load_target(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset split not found: {dataset_path}")

    data = np.load(dataset_path)
    return to_nhw(data["Ytest"], "Target")


def load_ensemble(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} prediction file not found: {path}")

    return to_nshw(np.load(path), name)


# ============================================================
# TEMPORAL AUTOCORRELATION
# ============================================================

def temporal_autocorrelation_distribution(images, max_lag=5):
    """
    Compute pixel-wise temporal autocorrelation.

    images: (N,H,W)

    For each pixel and lag:
        corr[x(t), x(t+lag)]

    Returns:
        dict[lag] = array of pixel-wise correlations
    """
    images = to_nhw(images)

    n, h, w = images.shape
    out = {}

    for lag in range(1, max_lag + 1):

        vals = []

        if lag >= n:
            out[lag] = np.array([], dtype=np.float32)
            continue

        for i in range(h):
            for j in range(w):

                ts1 = images[:-lag, i, j]
                ts2 = images[lag:, i, j]

                if np.std(ts1) < 1e-12 or np.std(ts2) < 1e-12:
                    continue

                corr = np.corrcoef(ts1, ts2)[0, 1]

                if np.isfinite(corr):
                    vals.append(corr)

        out[lag] = np.asarray(vals, dtype=np.float32)

    return out


# ============================================================
# METRICS
# ============================================================

def compute_bias_rmse(model_dict, truth_dict, max_lag):
    lags = []
    bias = []
    rmse = []

    for lag in range(1, max_lag + 1):

        truth_vals = truth_dict[lag]
        model_vals = model_dict[lag]

        n = min(len(truth_vals), len(model_vals))

        if n == 0:
            lags.append(lag)
            bias.append(np.nan)
            rmse.append(np.nan)
            continue

        diff = model_vals[:n] - truth_vals[:n]

        lags.append(lag)
        bias.append(np.mean(diff))
        rmse.append(np.sqrt(np.mean(diff ** 2)))

    return np.asarray(lags), np.asarray(bias), np.asarray(rmse)


def dict_to_df(data_dict, model_name):
    rows = []

    for lag, arr in data_dict.items():
        for value in arr:
            rows.append({
                "Model": model_name,
                "Lag": lag,
                "Correlation": value
            })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Loading Figure 10 data")
    print("=" * 70)

    Y = load_target(DATASET_PATH)

    unet_all = load_ensemble(UNET_PATH, "U-Net")
    wgan_all = load_ensemble(WGAN_PATH, "WGAN")
    ddpm_all = load_ensemble(DDPM_PATH, "DDPM")

    if N_SAMPLES is not None:
        Y = Y[:N_SAMPLES]
        unet_all = unet_all[:N_SAMPLES]
        wgan_all = wgan_all[:N_SAMPLES]
        ddpm_all = ddpm_all[:N_SAMPLES]

    n_common = min(
        len(Y),
        len(unet_all),
        len(wgan_all),
        len(ddpm_all)
    )

    Y = Y[:n_common]
    unet_all = unet_all[:n_common]
    wgan_all = wgan_all[:n_common]
    ddpm_all = ddpm_all[:n_common]

    if GEN_ID >= unet_all.shape[1]:
        raise ValueError(f"GEN_ID={GEN_ID} exceeds U-Net samples {unet_all.shape[1]}")

    if GEN_ID >= wgan_all.shape[1]:
        raise ValueError(f"GEN_ID={GEN_ID} exceeds WGAN samples {wgan_all.shape[1]}")

    if GEN_ID >= ddpm_all.shape[1]:
        raise ValueError(f"GEN_ID={GEN_ID} exceeds DDPM samples {ddpm_all.shape[1]}")

    unet = unet_all[:, GEN_ID]
    wgan = wgan_all[:, GEN_ID]
    ddpm = ddpm_all[:, GEN_ID]

    print(f"Target: {Y.shape}")
    print(f"U-Net : {unet.shape}")
    print(f"WGAN  : {wgan.shape}")
    print(f"DDPM  : {ddpm.shape}")

    # --------------------------------------------------------
    # Compute temporal correlations
    # --------------------------------------------------------
    print("\nComputing temporal autocorrelation")

    acs_truth = temporal_autocorrelation_distribution(Y, MAX_LAG)
    acs_unet = temporal_autocorrelation_distribution(unet, MAX_LAG)
    acs_wgan = temporal_autocorrelation_distribution(wgan, MAX_LAG)
    acs_ddpm = temporal_autocorrelation_distribution(ddpm, MAX_LAG)

    # --------------------------------------------------------
    # Bias / RMSE
    # --------------------------------------------------------
    lags, bias_unet, rmse_unet = compute_bias_rmse(
        acs_unet,
        acs_truth,
        MAX_LAG
    )

    _, bias_wgan, rmse_wgan = compute_bias_rmse(
        acs_wgan,
        acs_truth,
        MAX_LAG
    )

    _, bias_ddpm, rmse_ddpm = compute_bias_rmse(
        acs_ddpm,
        acs_truth,
        MAX_LAG
    )

    # --------------------------------------------------------
    # DataFrame for boxplot
    # --------------------------------------------------------
    df = pd.concat([
        dict_to_df(acs_truth, "Target (ERA5-Land)"),
        dict_to_df(acs_unet, "U-Net"),
        dict_to_df(acs_wgan, "WGAN"),
        dict_to_df(acs_ddpm, "DDPM"),
    ], ignore_index=True)

    # --------------------------------------------------------
    # Style
    # --------------------------------------------------------
    plt.rcParams.update({
        "font.size": FONTSIZE_LABEL,
        "axes.labelsize": FONTSIZE_LABEL,
        "xtick.labelsize": FONTSIZE_TICK,
        "ytick.labelsize": FONTSIZE_TICK,
        "legend.fontsize": FONTSIZE_LEGEND,
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
    })

    colors = {
        "Target (ERA5-Land)": "#2baad3",
        "U-Net": "#c57a3e",
        "WGAN": "#986bc5",
        "DDPM": "#ca5478",
    }

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------
    print("\nCreating Figure 10")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        constrained_layout=False
    )

    ax1, ax2, ax3 = axes

    # ========================================================
    # (a) Temporal correlation boxplot
    # ========================================================
    models = [
        "Target (ERA5-Land)",
        "U-Net",
        "WGAN",
        "DDPM"
    ]

    offsets = [-0.27, -0.09, 0.09, 0.27]
    width = 0.14

    for model, offset in zip(models, offsets):

        positions = []
        box_data = []

        for lag in range(1, MAX_LAG + 1):

            vals = df[
                (df["Model"] == model) &
                (df["Lag"] == lag)
            ]["Correlation"].values

            positions.append(lag + offset)
            box_data.append(vals)

        bp = ax1.boxplot(
            box_data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(
                color="black",
                linewidth=0.9
            ),
            whiskerprops=dict(
                color="0.35",
                linewidth=0.8
            ),
            capprops=dict(
                color="0.35",
                linewidth=0.8
            ),
            boxprops=dict(
                edgecolor="0.35",
                linewidth=0.8
            )
        )

        for patch in bp["boxes"]:
            patch.set_facecolor(colors[model])
            patch.set_alpha(0.76)

    ax1.set_xlim(0.4, MAX_LAG + 0.6)
    ax1.set_xticks(np.arange(1, MAX_LAG + 1))
    ax1.set_xticklabels([str(i) for i in range(1, MAX_LAG + 1)])

    ax1.set_xlabel("Temporal lag")
    ax1.set_ylabel("Temporal correlation")

    ax1.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.45
    )

    ax1.grid(axis="y", alpha=0.16)

    # ========================================================
    # (b) Bias
    # ========================================================
    ax2.plot(
        lags,
        bias_unet,
        marker="o",
        linewidth=2.1,
        color=colors["U-Net"],
        label="U-Net"
    )

    ax2.plot(
        lags,
        bias_wgan,
        marker="o",
        linewidth=2.1,
        color=colors["WGAN"],
        label="WGAN"
    )

    ax2.plot(
        lags,
        bias_ddpm,
        marker="o",
        linewidth=2.1,
        color=colors["DDPM"],
        label="DDPM"
    )

    ax2.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.45
    )

    ax2.set_xlabel("Temporal lag")
    ax2.set_ylabel("Bias")

    ax2.set_xticks(np.arange(1, MAX_LAG + 1))
    ax2.set_xticklabels([str(i) for i in range(1, MAX_LAG + 1)])
    ax2.grid(axis="y", alpha=0.16)

    # ========================================================
    # (c) RMSE
    # ========================================================
    ax3.plot(
        lags,
        rmse_unet,
        marker="o",
        linewidth=2.1,
        color=colors["U-Net"],
        label="U-Net"
    )

    ax3.plot(
        lags,
        rmse_wgan,
        marker="o",
        linewidth=2.1,
        color=colors["WGAN"],
        label="WGAN"
    )

    ax3.plot(
        lags,
        rmse_ddpm,
        marker="o",
        linewidth=2.1,
        color=colors["DDPM"],
        label="DDPM"
    )

    ax3.set_xlabel("Temporal lag")
    ax3.set_ylabel("RMSE")

    ax3.set_xticks(np.arange(1, MAX_LAG + 1))
    ax3.set_xticklabels([str(i) for i in range(1, MAX_LAG + 1)])
    ax3.grid(axis="y", alpha=0.16)

    # --------------------------------------------------------
    # Panel labels
    # --------------------------------------------------------
    panel_labels = {
        ax1: "(a)",
        ax2: "(b)",
        ax3: "(c)",
    }

    for ax, label in panel_labels.items():
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONTSIZE_PANEL,
            fontweight="normal"
        )

    # --------------------------------------------------------
    # Clean style
    # --------------------------------------------------------
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=FONTSIZE_TICK)

    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------
    legend_handles = [
        plt.Line2D(
            [0], [0],
            color=colors["Target (ERA5-Land)"],
            lw=6,
            label="Target (ERA5-Land)"
        ),
        plt.Line2D(
            [0], [0],
            color=colors["U-Net"],
            lw=6,
            label="U-Net"
        ),
        plt.Line2D(
            [0], [0],
            color=colors["WGAN"],
            lw=6,
            label="WGAN"
        ),
        plt.Line2D(
            [0], [0],
            color=colors["DDPM"],
            lw=6,
            label="DDPM"
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=FONTSIZE_LEGEND,
        bbox_to_anchor=(0.5, 0.005)
    )

    plt.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.955,
        bottom=0.20,
        wspace=0.28
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
