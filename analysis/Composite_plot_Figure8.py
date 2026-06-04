#!/usr/bin/env python3
# ============================================================
# Script: analysis/	Composite_plot_Figure8.py.py
#Figure n8
#
# Description:
#     Composite metrics figure for precipitation downscaling models.
#
#     Produces a 2 x 2 figure:
#         (a) Radial power spectrum
#         (b) Fractions Skill Score (FSS)
#         (c) ROC curve with AUC annotation
#         (d) Quantile-Quantile (Q-Q) plot
#
#     The SSIM panel used in the earlier version is replaced by
#     the Q-Q analysis panel in the revised version.
#
# Inputs:
#     dataset_splits_16x.npz
#         Xtest : LR inputs
#         Ytest : HR target fields, Target (ERA5-Land)
#
#     U-Net checkpoints:
#         unet_generator_best_seed{seed}.h5
#
#     WGAN checkpoints:
#         gen_final_seed{seed}.keras
#
#     DDPM predictions:
#         Xtest_predictions_seed{seed}_sample1.npy
#
# Outputs:
#     Figure8_composite_metrics_QQ.png
#     Figure8_composite_metrics_QQ.pdf
#
# Usage:
#     python analysis/18_plot_composite_metrics_qq.py
#
# Requirements:
#     numpy, tensorflow, matplotlib, scipy, scikit-learn
# ============================================================

import os
import numpy as np
import numpy.fft as fft
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.ndimage import uniform_filter
from sklearn.metrics import roc_curve, auc
from matplotlib.lines import Line2D


# ============================================================
# USER CONFIGURATION
# ============================================================

DATASET_PATH = "/scratch/wpa8me/ERA5_land/Dataset/dataset_splits_16x.npz"

UNET_DIR = "/scratch/wpa8me/unet_runs_16x/Same"
WGAN_DIR = os.path.join(UNET_DIR, "WGANs")

DDPM_DIR = (
    "/scratch/wpa8me/New16x/"
    "predictions_multi_sample/T500"
)

OUTPUT_DIR = "./Figure8_composite_metrics_QQ"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(
    OUTPUT_DIR,
    "Figure8_composite_metrics_QQ.png"
)

OUT_PDF = os.path.join(
    OUTPUT_DIR,
    "Figure8_composite_metrics_QQ.pdf"
)

SEEDS = list(range(1, 11))


# ============================================================
# METRIC SETTINGS
# ============================================================

DX_KM = 9.0
RAIN_THR = 1.0

FSS_WINDOWS = [1, 2, 4, 8, 16, 32]

NSAMP_POWER = 250

QQ_THRESHOLD = 20.0
QQ_POINTS = 200
QQ_XMAX = 200
QQ_YMAX = 250

NOISE_STD = 1.0
NOISE_SEED = 42
BATCH_SIZE = 32

N_SAMPLES = None   # None = use all test samples


# ============================================================
# FIGURE SETTINGS
# ============================================================

FIG_W = 15
FIG_H = 11

LW_TRUTH = 2.8
LW_SEED = 1.6
ALPHA_SEED = 0.35

FS_LABEL = 16
FS_TICK = 14
FS_LEGEND = 14
FS_PANEL = 16

plt.rcParams.update({
    "font.size": FS_LABEL,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.grid": False,
})


# ============================================================
# COLORS
# ============================================================

COLORS = {
    "Target": "#2baad3",
    "U-Net":  "#c57a3e",
    "WGAN":   "#986bc5",
    "DDPM":   "#ca5478",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def squeeze_hw(arr):
    """
    Convert array to shape (N, H, W).
    """
    arr = np.asarray(arr)

    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    return np.squeeze(arr).astype(np.float32)


def ensure_channel(arr):
    """
    Convert array from (N, H, W) to (N, H, W, 1).
    """
    arr = np.asarray(arr)

    if arr.ndim == 3:
        arr = arr[..., np.newaxis]

    return arr.astype(np.float32)


def load_dataset(dataset_path):
    """
    Load Xtest and Ytest from dataset split file.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = np.load(dataset_path)

    Xtest = data["Xtest"].astype(np.float32)
    Ytest = squeeze_hw(data["Ytest"])

    if N_SAMPLES is not None:
        Xtest = Xtest[:N_SAMPLES]
        Ytest = Ytest[:N_SAMPLES]

    return ensure_channel(Xtest), Ytest


# ============================================================
# MODEL LOADING / PREDICTION
# ============================================================

def load_tf_prediction(model_path, Xtest, seed):
    """
    Load a TensorFlow/Keras model and generate prediction.

    The model is assumed to take:
        [LR input, noise input]
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    rng = np.random.RandomState(NOISE_SEED + seed)

    noise = rng.normal(
        loc=0.0,
        scale=NOISE_STD,
        size=Xtest.shape
    ).astype(np.float32)

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    pred = model.predict(
        [Xtest, noise],
        batch_size=BATCH_SIZE,
        verbose=0
    )

    pred = squeeze_hw(pred)

    del model
    tf.keras.backend.clear_session()

    return pred


def load_ddpm_prediction(ddpm_path):
    """
    Load one DDPM prediction array.
    """
    if not os.path.exists(ddpm_path):
        raise FileNotFoundError(f"DDPM prediction not found: {ddpm_path}")

    pred = squeeze_hw(np.load(ddpm_path))

    if N_SAMPLES is not None:
        pred = pred[:N_SAMPLES]

    return pred


def load_all_predictions(Xtest):
    """
    Load/generate predictions for all available seeds.
    """
    preds_unet = {}
    preds_wgan = {}
    preds_ddpm = {}

    used_unet = []
    used_wgan = []
    used_ddpm = []

    for seed in SEEDS:

        print("\n" + "=" * 70)
        print(f"Processing seed {seed}")
        print("=" * 70)

        # -------------------------
        # U-Net
        # -------------------------
        unet_path = os.path.join(
            UNET_DIR,
            f"unet_generator_best_seed{seed}.h5"
        )

        if os.path.exists(unet_path):
            print(f"Generating U-Net prediction: seed {seed}")
            preds_unet[seed] = load_tf_prediction(
                model_path=unet_path,
                Xtest=Xtest,
                seed=seed
            )
            used_unet.append(seed)
        else:
            print(f"Missing U-Net model: {unet_path}")

        # -------------------------
        # WGAN
        # -------------------------
        wgan_path = os.path.join(
            WGAN_DIR,
            f"gen_final_seed{seed}.keras"
        )

        if os.path.exists(wgan_path):
            print(f"Generating WGAN prediction: seed {seed}")
            preds_wgan[seed] = load_tf_prediction(
                model_path=wgan_path,
                Xtest=Xtest,
                seed=seed
            )
            used_wgan.append(seed)
        else:
            print(f"Missing WGAN model: {wgan_path}")

        # -------------------------
        # DDPM
        # -------------------------
        ddpm_path = os.path.join(
            DDPM_DIR,
            f"seed{seed}",
            f"Xtest_predictions_seed{seed}_sample1.npy"
        )

        if os.path.exists(ddpm_path):
            print(f"Loading DDPM prediction: seed {seed}")
            preds_ddpm[seed] = load_ddpm_prediction(ddpm_path)
            used_ddpm.append(seed)
        else:
            print(f"Missing DDPM prediction: {ddpm_path}")

    if len(preds_unet) == 0:
        raise RuntimeError("No U-Net predictions were generated.")

    if len(preds_wgan) == 0:
        raise RuntimeError("No WGAN predictions were generated.")

    if len(preds_ddpm) == 0:
        raise RuntimeError("No DDPM predictions were loaded.")

    print("\nUsed seeds:")
    print(f"  U-Net: {used_unet}")
    print(f"  WGAN : {used_wgan}")
    print(f"  DDPM : {used_ddpm}")

    return preds_unet, preds_wgan, preds_ddpm


# ============================================================
# POWER SPECTRUM
# ============================================================

def radial_power(img):
    """
    Mean radial power spectrum of a single 2-D field.
    """
    F = fft.fftshift(fft.fft2(img))
    P = np.abs(F) ** 2

    H, W = P.shape
    cy, cx = H // 2, W // 2

    yy, xx = np.indices((H, W))

    r = np.sqrt(
        (xx - cx) ** 2 +
        (yy - cy) ** 2
    ).astype(int)

    rb = np.bincount(r.ravel(), P.ravel())
    ct = np.bincount(r.ravel())

    return rb / (ct + 1e-8)


def to_dB(ps):
    """
    Convert power spectrum to decibels.
    """
    return 10.0 * np.log10(
        np.clip(ps, 1e-12, None)
    )


def power_curve_mean(fields, nsamp):
    """
    Mean radial power spectrum across samples.
    """
    ns = min(nsamp, fields.shape[0])

    specs = [
        radial_power(fields[i])
        for i in range(ns)
    ]

    return np.mean(specs, axis=0)


def power_per_seed(pred_dict, nsamp, nyq):
    """
    Compute mean radial power spectrum for each seed.
    """
    out = {}

    for seed in sorted(pred_dict):
        out[seed] = power_curve_mean(
            pred_dict[seed],
            nsamp
        )[1:nyq + 1]

    return out


# ============================================================
# FSS
# ============================================================

def fss_score(obs, pred, thr=1.0, window=3):
    """
    Fractions Skill Score for one image pair.
    """
    obs_bin = (obs >= thr).astype(np.float32)
    pred_bin = (pred >= thr).astype(np.float32)

    obs_frac = uniform_filter(
        obs_bin,
        size=window,
        mode="nearest"
    )

    pred_frac = uniform_filter(
        pred_bin,
        size=window,
        mode="nearest"
    )

    numerator = ((pred_frac - obs_frac) ** 2).mean()

    denominator = (
        (pred_frac ** 2 + obs_frac ** 2).mean()
        + 1e-8
    )

    return 1.0 - numerator / denominator


def fss_per_seed(pred_dict, Ytest):
    """
    Compute FSS curve for each seed.
    """
    out = {}

    for seed in sorted(pred_dict):

        pred = pred_dict[seed]
        n_common = min(Ytest.shape[0], pred.shape[0])

        curve = []

        for window in FSS_WINDOWS:

            vals = [
                fss_score(
                    Ytest[i],
                    pred[i],
                    thr=RAIN_THR,
                    window=window
                )
                for i in range(n_common)
            ]

            curve.append(np.mean(vals))

        out[seed] = np.asarray(curve)

    return out


# ============================================================
# ROC
# ============================================================

def roc_per_seed(pred_dict, Ytest):
    """
    Compute ROC curve and AUC for each seed.
    """
    fpr_grid = np.linspace(0, 1, 300)

    tpr_dict = {}
    auc_dict = {}

    y_true = (
        Ytest >= RAIN_THR
    ).ravel().astype(np.uint8)

    for seed in sorted(pred_dict):

        pred = pred_dict[seed]

        n_common = min(Ytest.shape[0], pred.shape[0])

        y_true_seed = (
            Ytest[:n_common] >= RAIN_THR
        ).ravel().astype(np.uint8)

        y_score = pred[:n_common].ravel()

        fpr, tpr, _ = roc_curve(
            y_true_seed,
            y_score
        )

        tpr_i = np.interp(
            fpr_grid,
            fpr,
            tpr
        )

        tpr_i[0] = 0.0

        tpr_dict[seed] = tpr_i
        auc_dict[seed] = auc(fpr, tpr)

    return fpr_grid, tpr_dict, auc_dict


# ============================================================
# Q-Q ANALYSIS
# ============================================================

def flatten_positive_tail(arr, threshold):
    """
    Flatten finite precipitation values above threshold.
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr).ravel()

    arr = arr[np.isfinite(arr)]

    return arr[arr > threshold]


def compute_quantile_curve(arr, n_points):
    """
    Compute quantile curve from flattened values.
    """
    if arr.size == 0:
        return np.full(n_points, np.nan)

    q = np.linspace(0, 1, n_points)

    return np.quantile(arr, q)


def qq_per_seed(pred_dict, Ytest):
    """
    Compute Q-Q curves for each seed using values above QQ_THRESHOLD.
    """
    truth_flat = flatten_positive_tail(
        Ytest,
        threshold=QQ_THRESHOLD
    )

    truth_q = compute_quantile_curve(
        truth_flat,
        n_points=QQ_POINTS
    )

    qq_dict = {}

    for seed in sorted(pred_dict):

        pred_flat = flatten_positive_tail(
            pred_dict[seed],
            threshold=QQ_THRESHOLD
        )

        qq_dict[seed] = compute_quantile_curve(
            pred_flat,
            n_points=QQ_POINTS
        )

    return truth_q, qq_dict


# ============================================================
# PLOTTING HELPERS
# ============================================================

def add_panel_label(ax, label):
    ax.text(
        0.01,
        0.97,
        label,
        transform=ax.transAxes,
        fontsize=FS_PANEL,
        va="top",
        ha="left"
    )


def plot_seed_curves(ax, x, curves, color, marker=None):
    for seed in sorted(curves):
        ax.plot(
            x,
            curves[seed],
            color=color,
            lw=LW_SEED,
            alpha=ALPHA_SEED,
            marker=marker
        )


def auc_text(auc_dict):
    vals = np.asarray(list(auc_dict.values()), dtype=np.float64)

    return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Loading dataset")
    print("=" * 70)

    Xtest, Ytest = load_dataset(DATASET_PATH)

    print(f"Xtest: {Xtest.shape}")
    print(f"Ytest: {Ytest.shape}")

    N, H, W = Ytest.shape

    nyq = H // 2
    domain_length_km = H * DX_KM

    print("\nLoading/generating predictions...")
    preds_unet, preds_wgan, preds_ddpm = load_all_predictions(Xtest)

    # --------------------------------------------------------
    # Compute metrics
    # --------------------------------------------------------
    print("\nComputing composite metrics...")

    modes = np.arange(1, nyq + 1)
    wavelength_km = domain_length_km / modes

    truth_power = power_curve_mean(
        Ytest,
        NSAMP_POWER
    )[1:nyq + 1]

    unet_power = power_per_seed(preds_unet, NSAMP_POWER, nyq)
    wgan_power = power_per_seed(preds_wgan, NSAMP_POWER, nyq)
    ddpm_power = power_per_seed(preds_ddpm, NSAMP_POWER, nyq)

    unet_fss = fss_per_seed(preds_unet, Ytest)
    wgan_fss = fss_per_seed(preds_wgan, Ytest)
    ddpm_fss = fss_per_seed(preds_ddpm, Ytest)

    fpr_grid, unet_tpr, unet_auc = roc_per_seed(preds_unet, Ytest)
    _, wgan_tpr, wgan_auc = roc_per_seed(preds_wgan, Ytest)
    _, ddpm_tpr, ddpm_auc = roc_per_seed(preds_ddpm, Ytest)

    truth_q, unet_qq = qq_per_seed(preds_unet, Ytest)
    _, wgan_qq = qq_per_seed(preds_wgan, Ytest)
    _, ddpm_qq = qq_per_seed(preds_ddpm, Ytest)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    print("\nCreating composite figure with Q-Q panel...")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIG_W, FIG_H)
    )

    axA, axB, axC, axD = axes.ravel()

    # ========================================================
    # (a) Power spectrum
    # ========================================================
    axA.plot(
        wavelength_km,
        to_dB(truth_power),
        color=COLORS["Target"],
        lw=LW_TRUTH
    )

    for seed in sorted(unet_power):
        axA.plot(
            wavelength_km,
            to_dB(unet_power[seed]),
            color=COLORS["U-Net"],
            lw=LW_SEED,
            alpha=ALPHA_SEED
        )

    for seed in sorted(wgan_power):
        axA.plot(
            wavelength_km,
            to_dB(wgan_power[seed]),
            color=COLORS["WGAN"],
            lw=LW_SEED,
            alpha=ALPHA_SEED
        )

    for seed in sorted(ddpm_power):
        axA.plot(
            wavelength_km,
            to_dB(ddpm_power[seed]),
            color=COLORS["DDPM"],
            lw=LW_SEED,
            alpha=ALPHA_SEED
        )

    axA.set_xscale("log")
    axA.invert_xaxis()

    axA.set_xlabel("Wavelength (km)")
    axA.set_ylabel("Power (dB)")
    axA.grid(False)

    add_panel_label(axA, "(a)")

    # ========================================================
    # (b) FSS
    # ========================================================
    plot_seed_curves(
        axB,
        FSS_WINDOWS,
        unet_fss,
        COLORS["U-Net"],
        marker="o"
    )

    plot_seed_curves(
        axB,
        FSS_WINDOWS,
        wgan_fss,
        COLORS["WGAN"],
        marker="s"
    )

    plot_seed_curves(
        axB,
        FSS_WINDOWS,
        ddpm_fss,
        COLORS["DDPM"],
        marker="^"
    )

    axB.set_ylim(0, 1.05)

    axB.set_xlabel("Window size (pixels)")
    axB.set_ylabel("FSS")
    axB.grid(False)

    add_panel_label(axB, "(b)")

    # ========================================================
    # (c) ROC
    # ========================================================
    plot_seed_curves(
        axC,
        fpr_grid,
        unet_tpr,
        COLORS["U-Net"]
    )

    plot_seed_curves(
        axC,
        fpr_grid,
        wgan_tpr,
        COLORS["WGAN"]
    )

    plot_seed_curves(
        axC,
        fpr_grid,
        ddpm_tpr,
        COLORS["DDPM"]
    )

    axC.plot(
        [0, 1],
        [0, 1],
        "k--",
        lw=1.0
    )

    axC.set_xlabel("False positive rate")
    axC.set_ylabel("True positive rate")
    axC.grid(False)

    add_panel_label(axC, "(c)")

    axC.text(
        0.53,
        0.05,
        (
            f"U-Net: AUC = {auc_text(unet_auc)}\n"
            f"WGAN: AUC = {auc_text(wgan_auc)}\n"
            f"DDPM: AUC = {auc_text(ddpm_auc)}"
        ),
        transform=axC.transAxes,
        fontsize=11,
        va="bottom",
        bbox=dict(
            facecolor="white",
            edgecolor="gray",
            alpha=0.9
        )
    )

    # ========================================================
    # (d) Q-Q plot
    # ========================================================
    axD.plot(
        truth_q,
        truth_q,
        "k--",
        lw=1.8
    )

    plot_seed_curves(
        axD,
        truth_q,
        unet_qq,
        COLORS["U-Net"]
    )

    plot_seed_curves(
        axD,
        truth_q,
        wgan_qq,
        COLORS["WGAN"]
    )

    plot_seed_curves(
        axD,
        truth_q,
        ddpm_qq,
        COLORS["DDPM"]
    )

    axD.set_xlabel("Target quantiles (ERA5-Land; mm/day)")
    axD.set_ylabel("Model quantiles (mm/day)")

    axD.set_xlim(0, QQ_XMAX)
    axD.set_ylim(0, QQ_YMAX)

    axD.grid(False)

    add_panel_label(axD, "(d)")

    # ========================================================
    # Shared legend
    # ========================================================
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["Target"],
            lw=LW_TRUTH,
            label="Target (ERA5-Land)"
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["U-Net"],
            lw=2.5,
            label="U-Net"
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["WGAN"],
            lw=2.5,
            label="WGAN"
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["DDPM"],
            lw=2.5,
            label="DDPM"
        ),
        Line2D(
            [0],
            [0],
            color="black",
            lw=1.8,
            linestyle="--",
            label="1:1 line"
        ),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=FS_LEGEND,
        bbox_to_anchor=(0.5, 0.01)
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    fig.savefig(
        OUT_PNG,
        dpi=400,
        bbox_inches="tight"
    )

    fig.savefig(
        OUT_PDF,
        dpi=400,
        bbox_inches="tight"
    )

    plt.show()

    print("\nSaved:")
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()