#!/usr/bin/env python3
# ============================================================
# Script: analysis/plot_marginal_statistics_Figure5_S6.py
#
# Description:
#     Marginal precipitation statistics scatter plots for model evaluation.
#
#     Produces a 5-row x 3-column density scatter figure:
#
#         Columns : U-Net | WGAN | DDPM
#         Rows    : Probability of dry (%) |
#                   Mean |
#                   Second L-moment |
#                   L-skewness |
#                   L-kurtosis
#
#     Statistics are computed per test image and compared against
#     Target (ERA5-Land). All available ensemble seeds are pooled to form
#     a single density scatter per model family, giving N x seeds points
#     per panel.
#
#     Bias and RMSE of each model statistic relative to the target statistic
#     are annotated in each panel.
#
# Inputs:
#     dataset_splits_16x.npz
#         Xtest : LR inputs, shape (N, 8, 8)
#         Ytest : HR targets, shape (N, 128, 128)
#
#     U-Net checkpoints:
#         unet_generator_best_seed{seed}.h5
#
#     WGAN checkpoints:
#         gen_final_seed{seed}.keras
#
#     DDPM predictions:
#         Xtest_predictions_{seed}.npy
#
# Outputs:
#     Figure5_Lmoments_16x_allseeds_clean.png
#     Figure5_Lmoments_16x_allseeds_clean.pdf
#
# Usage:
#     python analysis/14_plot_marginal_lmoments.py
#
# Requirements:
#     numpy, tensorflow, matplotlib, scipy
# ============================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ============================================================
# USER CONFIGURATION
# ============================================================

DATASET_PATH = "/scratch/wpa8me/ERA5_land/Dataset/dataset_splits_16x.npz"

UNET_MODEL_DIR = "/scratch/wpa8me/unet_runs_16x/Same"
WGAN_MODEL_DIR = os.path.join(UNET_MODEL_DIR, "WGANs")

DDPM_PRED_DIR = "/scratch/wpa8me/New16x/predictions_multiseed_16x1/T500"

SEEDS = list(range(1, 11))

OUTPUT_DIR = "/scratch/wpa8me/unet_runs_16x/Same/figure5_lmoments_16x_allseeds_clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUTPUT_DIR, "Figure5_Lmoments_16x_allseeds_clean.png")
OUT_PDF = os.path.join(OUTPUT_DIR, "Figure5_Lmoments_16x_allseeds_clean.pdf")


# ============================================================
# SETTINGS
# ============================================================

N_SAMPLES = None          # None = use all test samples
DRY_THRESHOLD = 1.0       # mm/day
WET_THRESHOLD = 1.0       # mm/day
MIN_WET_PIXELS = 30       # minimum wet pixels required for L-moments

NOISE_STD = 1.0
NOISE_BASE_SEED = 2026
BATCH_SIZE = 32

FIG_WIDTH = 17.5
FIG_HEIGHT = 20.0

FONTSIZE_TITLE = 23
FONTSIZE_LABEL = 19
FONTSIZE_TICK = 16
FONTSIZE_ROW = 21
FONTSIZE_TEXT = 15

POINT_SIZE = 8

LMOMENT_AXIS_LIMITS = {
    "L-skewness": (0.0, 1.0),
    "L-kurtosis": (0.0, 1.0),
}

plt.rcParams.update({
    "font.size": FONTSIZE_LABEL,
    "axes.labelsize": FONTSIZE_LABEL,
    "axes.titlesize": FONTSIZE_TITLE,
    "xtick.labelsize": FONTSIZE_TICK,
    "ytick.labelsize": FONTSIZE_TICK,
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.grid": False,
})


# ============================================================
# BASIC HELPERS
# ============================================================

def squeeze_nhw(arr):
    """
    Convert arrays to shape (N, H, W).
    Handles arrays with singleton channel dimension.
    """
    arr = np.asarray(arr)

    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    return np.squeeze(arr).astype(np.float32)


def ensure_channel(arr):
    """
    Convert arrays from (N, H, W) to (N, H, W, 1).
    """
    arr = np.asarray(arr)

    if arr.ndim == 3:
        arr = arr[..., np.newaxis]

    return arr.astype(np.float32)


def load_dataset(dataset_path):
    """
    Load Xtest and Ytest from compressed dataset split file.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = np.load(dataset_path)

    Xtest = data["Xtest"].astype(np.float32)
    Ytest = squeeze_nhw(data["Ytest"])

    if N_SAMPLES is not None:
        Xtest = Xtest[:N_SAMPLES]
        Ytest = Ytest[:N_SAMPLES]

    return Xtest, Ytest


# ============================================================
# MODEL INFERENCE / PREDICTION LOADING
# ============================================================

def generate_tf_prediction(model_path, X, seed):
    """
    Load a TensorFlow/Keras generator and produce predictions.

    The generator is assumed to take:
        [LR input, noise input]

    where both inputs have shape (N, H, W, 1).
    """
    model = tf.keras.models.load_model(model_path, compile=False)

    X_in = ensure_channel(X)

    rng = np.random.default_rng(NOISE_BASE_SEED + seed)
    noise = rng.normal(
        loc=0.0,
        scale=NOISE_STD,
        size=X_in.shape
    ).astype(np.float32)

    preds = model.predict(
        [X_in, noise],
        batch_size=BATCH_SIZE,
        verbose=0
    )

    preds = squeeze_nhw(preds)

    del model
    tf.keras.backend.clear_session()

    return preds.astype(np.float32)


def load_ddpm_prediction(ddpm_path):
    """
    Load pre-generated DDPM predictions.
    """
    preds = squeeze_nhw(np.load(ddpm_path))

    if N_SAMPLES is not None:
        preds = preds[:N_SAMPLES]

    return preds.astype(np.float32)


# ============================================================
# STATISTICS
# ============================================================

def probability_dry(image, threshold=1.0):
    """
    Probability of dry pixels in percentage.
    """
    return 100.0 * np.mean(image <= threshold)


def sample_lmoments(vals):
    """
    Compute sample L-moments from a one-dimensional array.

    Returns:
        l1 : first L-moment, equivalent to mean
        l2 : second L-moment
        t3 : L-skewness = l3 / l2
        t4 : L-kurtosis = l4 / l2
    """
    vals = np.sort(np.asarray(vals, dtype=np.float64))
    n = len(vals)

    if n < 4:
        return np.nan, np.nan, np.nan, np.nan

    j = np.arange(1, n + 1, dtype=np.float64)

    b0 = np.mean(vals)

    b1 = np.mean(
        ((j - 1) / (n - 1)) * vals
    )

    b2 = np.mean(
        ((j - 1) * (j - 2)) /
        ((n - 1) * (n - 2)) * vals
    )

    b3 = np.mean(
        ((j - 1) * (j - 2) * (j - 3)) /
        ((n - 1) * (n - 2) * (n - 3)) * vals
    )

    l1 = b0
    l2 = 2.0 * b1 - b0
    l3 = 6.0 * b2 - 6.0 * b1 + b0
    l4 = 20.0 * b3 - 30.0 * b2 + 12.0 * b1 - b0

    if np.abs(l2) < 1e-12:
        t3 = np.nan
        t4 = np.nan
    else:
        t3 = l3 / l2
        t4 = l4 / l2

    return l1, l2, t3, t4


def compute_all_statistics(images):
    """
    Compute per-image marginal statistics.

    Statistics:
        Probability of dry (%) : fraction of pixels <= DRY_THRESHOLD
        Mean                   : first L-moment of wet pixels
        Second L-moment         : L-scale of wet pixels
        L-skewness              : L3 / L2
        L-kurtosis              : L4 / L2
    """
    images = squeeze_nhw(images)

    prob_dry = []
    mean_vals = []
    l2_vals = []
    lskew_vals = []
    lkurt_vals = []

    for i in range(images.shape[0]):

        img = images[i]

        prob_dry.append(
            probability_dry(img, threshold=DRY_THRESHOLD)
        )

        vals = img[np.isfinite(img)]
        vals = vals[vals > WET_THRESHOLD].ravel()

        if vals.size < MIN_WET_PIXELS:
            mean_vals.append(np.nan)
            l2_vals.append(np.nan)
            lskew_vals.append(np.nan)
            lkurt_vals.append(np.nan)
            continue

        l1, l2, t3, t4 = sample_lmoments(vals)

        mean_vals.append(l1)
        l2_vals.append(l2)
        lskew_vals.append(t3)
        lkurt_vals.append(t4)

    return {
        "Probability of dry (%)": np.asarray(prob_dry, dtype=np.float64),
        "Mean": np.asarray(mean_vals, dtype=np.float64),
        "Second L-moment": np.asarray(l2_vals, dtype=np.float64),
        "L-skewness": np.asarray(lskew_vals, dtype=np.float64),
        "L-kurtosis": np.asarray(lkurt_vals, dtype=np.float64),
    }


def calculate_bias_rmse(target, pred):
    """
    Calculate bias and RMSE after removing invalid values.
    """
    target = np.asarray(target)
    pred = np.asarray(pred)

    valid = np.isfinite(target) & np.isfinite(pred)

    target = target[valid]
    pred = pred[valid]

    if target.size == 0:
        return np.nan, np.nan

    bias = np.mean(pred - target)
    rmse = np.sqrt(np.mean((pred - target) ** 2))

    return bias, rmse


def repeat_target_stats(target_stats, n_repeat):
    """
    Repeat target statistics for pooling against multiple model seeds.
    """
    return {
        key: np.concatenate([target_stats[key] for _ in range(n_repeat)])
        for key in target_stats
    }


def pool_model_seed_statistics(model_stats_list):
    """
    Pool statistics from all available seeds for a model family.
    """
    pooled = {}
    keys = model_stats_list[0].keys()

    for key in keys:
        pooled[key] = np.concatenate([
            stats[key] for stats in model_stats_list
        ])

    return pooled


# ============================================================
# PLOTTING
# ============================================================

def scatter_density_panel(ax, target, pred, title, stat_name):
    """
    Create one density-colored scatter panel with 1:1 reference line
    and bias/RMSE annotation.
    """
    target = np.asarray(target)
    pred = np.asarray(pred)

    valid = np.isfinite(target) & np.isfinite(pred)

    target = target[valid]
    pred = pred[valid]

    if target.size == 0:
        ax.text(
            0.5,
            0.5,
            "No valid data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FONTSIZE_TEXT
        )
        return

    xy = np.vstack([target, pred])

    try:
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()

        ax.scatter(
            target[idx],
            pred[idx],
            c=z[idx],
            cmap="viridis",
            s=POINT_SIZE,
            edgecolors="none"
        )

    except Exception:
        ax.scatter(
            target,
            pred,
            color="#440154",
            s=POINT_SIZE,
            alpha=0.65,
            edgecolors="none"
        )

    if stat_name in LMOMENT_AXIS_LIMITS:
        xy_min, xy_max = LMOMENT_AXIS_LIMITS[stat_name]

    elif stat_name == "Probability of dry (%)":
        xy_min, xy_max = 0.0, 100.0

    else:
        xy_min = min(np.nanmin(target), np.nanmin(pred))
        xy_max = max(np.nanmax(target), np.nanmax(pred))

        pad = 0.04 * (xy_max - xy_min + 1e-8)
        xy_min -= pad
        xy_max += pad

        if xy_min > 0:
            xy_min = 0.0

    ax.plot(
        [xy_min, xy_max],
        [xy_min, xy_max],
        linestyle="--",
        linewidth=1.2,
        color="red"
    )

    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)

    bias, rmse = calculate_bias_rmse(target, pred)

    ax.text(
        0.05,
        0.94,
        f"Bias: {bias:.2f}\nRMSE: {rmse:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=FONTSIZE_TEXT,
        fontweight="normal",
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            linewidth=0.7,
            alpha=0.85,
            boxstyle="round,pad=0.25"
        )
    )

    ax.set_title(
        title,
        fontsize=FONTSIZE_TITLE,
        fontweight="normal"
    )

    ax.set_xlabel(
        "Target (ERA5-Land)",
        fontsize=FONTSIZE_LABEL,
        fontweight="normal",
        labelpad=3
    )

    ax.set_ylabel(
        "Predicted",
        fontsize=FONTSIZE_LABEL,
        fontweight="normal",
        labelpad=3
    )

    ax.tick_params(labelsize=FONTSIZE_TICK)


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():

    print("=" * 70)
    print("Loading 16x test dataset")
    print("=" * 70)

    Xtest_arr, Ytest_arr = load_dataset(DATASET_PATH)

    print(f"Xtest : {Xtest_arr.shape}")
    print(f"Ytest : {Ytest_arr.shape}")

    print("\nComputing target statistics...")
    target_stats_single = compute_all_statistics(Ytest_arr)

    unet_stats_all = []
    wgan_stats_all = []
    ddpm_stats_all = []

    used_unet_seeds = []
    used_wgan_seeds = []
    used_ddpm_seeds = []

    for seed in SEEDS:

        print("\n" + "=" * 70)
        print(f"Processing seed {seed}")
        print("=" * 70)

        # ----------------------------------------------------
        # U-Net
        # ----------------------------------------------------
        unet_path = os.path.join(
            UNET_MODEL_DIR,
            f"unet_generator_best_seed{seed}.h5"
        )

        if os.path.exists(unet_path):

            print(f"Generating U-Net prediction for seed {seed}")

            unet_pred = generate_tf_prediction(
                model_path=unet_path,
                X=Xtest_arr,
                seed=seed
            )

            n_common = min(Ytest_arr.shape[0], unet_pred.shape[0])

            unet_stats_all.append(
                compute_all_statistics(unet_pred[:n_common])
            )

            used_unet_seeds.append(seed)

        else:
            print(f"Missing U-Net seed {seed}: {unet_path}")

        # ----------------------------------------------------
        # WGAN
        # ----------------------------------------------------
        wgan_path = os.path.join(
            WGAN_MODEL_DIR,
            f"gen_final_seed{seed}.keras"
        )

        if os.path.exists(wgan_path):

            print(f"Generating WGAN prediction for seed {seed}")

            wgan_pred = generate_tf_prediction(
                model_path=wgan_path,
                X=Xtest_arr,
                seed=seed
            )

            n_common = min(Ytest_arr.shape[0], wgan_pred.shape[0])

            wgan_stats_all.append(
                compute_all_statistics(wgan_pred[:n_common])
            )

            used_wgan_seeds.append(seed)

        else:
            print(f"Missing WGAN seed {seed}: {wgan_path}")

        # ----------------------------------------------------
        # DDPM
        # ----------------------------------------------------
        ddpm_path = os.path.join(
            DDPM_PRED_DIR,
            f"Xtest_predictions_{seed}.npy"
        )

        if os.path.exists(ddpm_path):

            print(f"Loading DDPM prediction for seed {seed}")

            ddpm_pred = load_ddpm_prediction(ddpm_path)

            n_common = min(Ytest_arr.shape[0], ddpm_pred.shape[0])

            ddpm_stats_all.append(
                compute_all_statistics(ddpm_pred[:n_common])
            )

            used_ddpm_seeds.append(seed)

        else:
            print(f"Missing DDPM seed {seed}: {ddpm_path}")

    if len(unet_stats_all) == 0:
        raise RuntimeError("No U-Net statistics computed. Check U-Net model paths.")

    if len(wgan_stats_all) == 0:
        raise RuntimeError("No WGAN statistics computed. Check WGAN model paths.")

    if len(ddpm_stats_all) == 0:
        raise RuntimeError("No DDPM statistics computed. Check DDPM prediction paths.")

    print("\nUsed seeds:")
    print(f"  U-Net: {used_unet_seeds}")
    print(f"  WGAN : {used_wgan_seeds}")
    print(f"  DDPM : {used_ddpm_seeds}")

    # --------------------------------------------------------
    # Pool target and model statistics
    # --------------------------------------------------------
    target_stats_unet = repeat_target_stats(
        target_stats_single,
        len(unet_stats_all)
    )

    target_stats_wgan = repeat_target_stats(
        target_stats_single,
        len(wgan_stats_all)
    )

    target_stats_ddpm = repeat_target_stats(
        target_stats_single,
        len(ddpm_stats_all)
    )

    unet_stats = pool_model_seed_statistics(unet_stats_all)
    wgan_stats = pool_model_seed_statistics(wgan_stats_all)
    ddpm_stats = pool_model_seed_statistics(ddpm_stats_all)

    target_stats_by_model = {
        "U-Net": target_stats_unet,
        "WGAN": target_stats_wgan,
        "DDPM": target_stats_ddpm,
    }

    model_stats_by_model = {
        "U-Net": unet_stats,
        "WGAN": wgan_stats,
        "DDPM": ddpm_stats,
    }

    row_names = [
        "Probability of dry (%)",
        "Mean",
        "Second L-moment",
        "L-skewness",
        "L-kurtosis",
    ]

    col_names = [
        "U-Net",
        "WGAN",
        "DDPM",
    ]

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------
    print("\nCreating L-moment marginal statistics figure...")

    fig, axes = plt.subplots(
        5,
        3,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        constrained_layout=False
    )

    for i, stat_name in enumerate(row_names):

        for j, model_name in enumerate(col_names):

            ax = axes[i, j]

            title = model_name if i == 0 else ""

            scatter_density_panel(
                ax=ax,
                target=target_stats_by_model[model_name][stat_name],
                pred=model_stats_by_model[model_name][stat_name],
                title=title,
                stat_name=stat_name
            )

            if j == 0:
                ax.text(
                    -0.24,
                    0.50,
                    stat_name,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=FONTSIZE_ROW,
                    fontweight="normal"
                )

    plt.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.965,
        bottom=0.060,
        wspace=0.24,
        hspace=0.34
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
    print(f"  {OUT_PNG}")
    print(f"  {OUT_PDF}")


if __name__ == "__main__":
    main()