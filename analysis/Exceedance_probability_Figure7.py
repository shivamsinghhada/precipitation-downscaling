#!/usr/bin/env python3
# ============================================================
# Figure 7: Exceedance Probability Composite
#
# Rows:
#   8x  : U-Net | WGAN | DDPM
#   16x : U-Net | WGAN | DDPM
#
# Columns:
#   U-Net, WGAN, DDPM
#
# The test set is Northeast only, as defined by:
#   05_prepare_dataset.py      -> dataset_splits.npz
#   05b_prepare_dataset_16x.py -> dataset_splits_16x.npz
# ============================================================

import os
import glob
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# CONFIG
# ============================================================

DATASET_8X_PATH = "/scratch/wpa8me/ERA5_land/Dataset/dataset_splits.npz"
DATASET_16X_PATH = "/scratch/wpa8me/ERA5_land/Dataset/dataset_splits_16x.npz"

# ---- 8x model paths ----
UNET_8X_DIR = "/scratch/wpa8me/unet_runs"
WGAN_8X_DIR = os.path.join(UNET_8X_DIR, "WGANs")
DDPM_8X_DIR = "/scratch/wpa8me/New_DDPM8x/predictions_multiseed_8x/T500"

# ---- 16x model paths ----
UNET_16X_DIR = "/scratch/wpa8me/unet_runs_16x/Same"
WGAN_16X_DIR = os.path.join(UNET_16X_DIR, "WGANs")
DDPM_16X_DIR = "/scratch/wpa8me/New16x/predictions_multiseed_16x1/T500"

UNET_GLOB = "unet_generator_best_seed*.h5"
WGAN_GLOB = "gen_final_seed*.keras"

SEEDS = list(range(1, 11))

OUT_DIR = "/scratch/wpa8me/Figure7_exceedance_probability"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, "Figure7_exceedance_probability_8x_16x.png")
OUT_PDF = os.path.join(OUT_DIR, "Figure7_exceedance_probability_8x_16x.pdf")


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 10.0
N_SAMPLES = None

NOISE_STD = 1.0
NOISE_SEED = 2026
BATCH_SIZE = 32

YMIN = 1e-8
YMAX = 1.0
XMIN = THRESHOLD
XMAX = 280.0

FIG_WIDTH = 18
FIG_HEIGHT = 10.4

FONTSIZE_LABEL = 16
FONTSIZE_TICK = 13
FONTSIZE_PANEL = 16
FONTSIZE_LEGEND = 14
FONTSIZE_ROW = 17

ALPHA_SEED = 0.62
LW_SEED = 1.45
LW_TRUTH = 2.8

COLORS = {
    "Target": "#2baad3",
    "U-Net":  "#c57a3e",
    "WGAN":   "#986bc5",
    "DDPM":   "#ca5478",
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


def ensure_channel(arr):
    arr = np.asarray(arr)

    if arr.ndim == 3:
        arr = arr[..., np.newaxis]

    return arr.astype(np.float32)


def load_test_split(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset split not found: {path}")

    data = np.load(path)

    Xtest = data["Xtest"].astype(np.float32)
    Ytest = squeeze_hw(data["Ytest"])

    if N_SAMPLES is not None:
        Xtest = Xtest[:N_SAMPLES]
        Ytest = Ytest[:N_SAMPLES]

    return Xtest, Ytest


def parse_seed(path):
    base = os.path.basename(path)
    match = re.search(r"seed(\d+)", base)

    if match is None:
        raise ValueError(f"Could not parse seed from: {base}")

    return int(match.group(1))


def exceedance_curve(arr, threshold):
    vals = np.ravel(arr)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > threshold]

    if vals.size == 0:
        return None, None

    vals_sorted = np.sort(vals)
    F = np.arange(1, vals_sorted.size + 1) / vals_sorted.size

    return vals_sorted, 1.0 - F


def infer_tf_models(model_dir, model_glob, Xtest, scale_label):
    model_paths = sorted(
        glob.glob(os.path.join(model_dir, model_glob)),
        key=parse_seed
    )

    if len(model_paths) == 0:
        raise RuntimeError(f"No models found in {model_dir}")

    preds = {}

    X_in = ensure_channel(Xtest)

    for path in model_paths:
        seed = parse_seed(path)

        rng = np.random.default_rng(NOISE_SEED + seed)

        noise = rng.normal(
            0.0,
            NOISE_STD,
            size=X_in.shape
        ).astype(np.float32)

        print(f"{scale_label}: running TF inference seed {seed}")

        model = tf.keras.models.load_model(path, compile=False)

        pred = model.predict(
            [X_in, noise],
            batch_size=BATCH_SIZE,
            verbose=0
        )

        preds[seed] = squeeze_hw(pred)

        del model
        tf.keras.backend.clear_session()

    return preds


def load_ddpm_8x(pred_dir):
    """
    8x DDPM naming used in the current run:
        Xtest_predictions_seed1_1.npy
        ...
        Xtest_predictions_seed1_10.npy
    """
    preds = {}

    for seed in SEEDS:
        path = os.path.join(
            pred_dir,
            f"Xtest_predictions_seed1_{seed}.npy"
        )

        if not os.path.exists(path):
            print(f"Missing 8x DDPM file: {path}")
            continue

        arr = squeeze_hw(np.load(path))

        if N_SAMPLES is not None:
            arr = arr[:N_SAMPLES]

        preds[seed] = arr
        print(f"8x: loaded DDPM seed {seed}: {arr.shape}")

    if len(preds) == 0:
        raise RuntimeError(f"No 8x DDPM predictions loaded from {pred_dir}")

    return preds


def load_ddpm_16x(pred_dir):
    """
    16x DDPM naming used in the current run:
        Xtest_predictions_1.npy
        ...
        Xtest_predictions_10.npy
    """
    preds = {}

    for seed in SEEDS:
        path = os.path.join(
            pred_dir,
            f"Xtest_predictions_{seed}.npy"
        )

        if not os.path.exists(path):
            print(f"Missing 16x DDPM file: {path}")
            continue

        arr = squeeze_hw(np.load(path))

        if N_SAMPLES is not None:
            arr = arr[:N_SAMPLES]

        preds[seed] = arr
        print(f"16x: loaded DDPM seed {seed}: {arr.shape}")

    if len(preds) == 0:
        raise RuntimeError(f"No 16x DDPM predictions loaded from {pred_dir}")

    return preds


def align_all(Ytest, pred_dicts):
    n_common = len(Ytest)

    for pred_dict in pred_dicts:
        n_common = min(
            n_common,
            min(arr.shape[0] for arr in pred_dict.values())
        )

    Ytest = Ytest[:n_common]

    aligned = []

    for pred_dict in pred_dicts:
        aligned.append({
            seed: arr[:n_common]
            for seed, arr in pred_dict.items()
        })

    return Ytest, aligned


def plot_exceedance_panel(
    ax,
    target,
    preds_dict,
    model_name,
    panel_label
):
    xt, yt = exceedance_curve(target, THRESHOLD)

    if xt is not None:
        ax.plot(
            xt,
            yt,
            color=COLORS["Target"],
            alpha=1.0,
            linewidth=LW_TRUTH,
            zorder=10
        )

    for seed in sorted(preds_dict):
        xp, yp = exceedance_curve(
            preds_dict[seed],
            THRESHOLD
        )

        if xp is None:
            continue

        ax.plot(
            xp,
            yp,
            color=COLORS[model_name],
            alpha=ALPHA_SEED,
            linewidth=LW_SEED,
            zorder=3
        )

    ax.set_yscale("log")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)

    ax.grid(False)

    ax.tick_params(
        axis="both",
        which="both",
        labelsize=FONTSIZE_TICK
    )

    ax.text(
        0.02,
        0.98,
        panel_label,
        transform=ax.transAxes,
        fontsize=FONTSIZE_PANEL,
        fontweight="normal",
        va="top",
        ha="left"
    )


def prepare_scale(scale_name, dataset_path, unet_dir, wgan_dir, ddpm_dir):
    print("\n" + "=" * 70)
    print(f"Preparing {scale_name}")
    print("=" * 70)

    Xtest, Ytest = load_test_split(dataset_path)

    print(f"{scale_name} Xtest: {Xtest.shape}")
    print(f"{scale_name} Ytest: {Ytest.shape}")

    preds_unet = infer_tf_models(
        model_dir=unet_dir,
        model_glob=UNET_GLOB,
        Xtest=Xtest,
        scale_label=f"{scale_name} U-Net"
    )

    preds_wgan = infer_tf_models(
        model_dir=wgan_dir,
        model_glob=WGAN_GLOB,
        Xtest=Xtest,
        scale_label=f"{scale_name} WGAN"
    )

    if scale_name == "8x":
        preds_ddpm = load_ddpm_8x(ddpm_dir)
    elif scale_name == "16x":
        preds_ddpm = load_ddpm_16x(ddpm_dir)
    else:
        raise ValueError(f"Unknown scale name: {scale_name}")

    Ytest, aligned = align_all(
        Ytest,
        [preds_unet, preds_wgan, preds_ddpm]
    )

    preds_unet, preds_wgan, preds_ddpm = aligned

    print(f"{scale_name}: common test samples = {Ytest.shape[0]}")

    return Ytest, preds_unet, preds_wgan, preds_ddpm


# ============================================================
# MAIN
# ============================================================

def main():

    Ytest_8x, unet_8x, wgan_8x, ddpm_8x = prepare_scale(
        scale_name="8x",
        dataset_path=DATASET_8X_PATH,
        unet_dir=UNET_8X_DIR,
        wgan_dir=WGAN_8X_DIR,
        ddpm_dir=DDPM_8X_DIR
    )

    Ytest_16x, unet_16x, wgan_16x, ddpm_16x = prepare_scale(
        scale_name="16x",
        dataset_path=DATASET_16X_PATH,
        unet_dir=UNET_16X_DIR,
        wgan_dir=WGAN_16X_DIR,
        ddpm_dir=DDPM_16X_DIR
    )

    print("\nCreating Figure 7 exceedance probability composite")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        sharey=True,
        constrained_layout=False
    )

    # -------------------------
    # 8x row
    # -------------------------
    plot_exceedance_panel(
        axes[0, 0],
        Ytest_8x,
        unet_8x,
        "U-Net",
        "(a)"
    )

    plot_exceedance_panel(
        axes[0, 1],
        Ytest_8x,
        wgan_8x,
        "WGAN",
        "(b)"
    )

    plot_exceedance_panel(
        axes[0, 2],
        Ytest_8x,
        ddpm_8x,
        "DDPM",
        "(c)"
    )

    # -------------------------
    # 16x row
    # -------------------------
    plot_exceedance_panel(
        axes[1, 0],
        Ytest_16x,
        unet_16x,
        "U-Net",
        "(d)"
    )

    plot_exceedance_panel(
        axes[1, 1],
        Ytest_16x,
        wgan_16x,
        "WGAN",
        "(e)"
    )

    plot_exceedance_panel(
        axes[1, 2],
        Ytest_16x,
        ddpm_16x,
        "DDPM",
        "(f)"
    )

    # Column titles
    axes[0, 0].set_title("U-Net", fontsize=FONTSIZE_LABEL, fontweight="normal")
    axes[0, 1].set_title("WGAN", fontsize=FONTSIZE_LABEL, fontweight="normal")
    axes[0, 2].set_title("DDPM", fontsize=FONTSIZE_LABEL, fontweight="normal")

    # Axis labels
    for ax in axes[1, :]:
        ax.set_xlabel(
            "Precipitation (mm/day)",
            fontsize=FONTSIZE_LABEL,
            fontweight="normal",
            labelpad=4
        )

    axes[0, 0].set_ylabel(
        "Exceedance probability",
        fontsize=FONTSIZE_LABEL,
        fontweight="normal",
        labelpad=4
    )

    axes[1, 0].set_ylabel(
        "Exceedance probability",
        fontsize=FONTSIZE_LABEL,
        fontweight="normal",
        labelpad=4
    )

    # Row labels
    axes[0, 0].text(
        -0.23,
        0.50,
        "8×",
        transform=axes[0, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=FONTSIZE_ROW,
        fontweight="normal"
    )

    axes[1, 0].text(
        -0.23,
        0.50,
        "16×",
        transform=axes[1, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=FONTSIZE_ROW,
        fontweight="normal"
    )

    # Legend
    legend_handles = [
        Line2D(
            [0], [0],
            color=COLORS["Target"],
            lw=LW_TRUTH,
            label="Target (ERA5-Land)"
        ),
        Line2D(
            [0], [0],
            color=COLORS["U-Net"],
            lw=2.2,
            alpha=ALPHA_SEED,
            label="U-Net"
        ),
        Line2D(
            [0], [0],
            color=COLORS["WGAN"],
            lw=2.2,
            alpha=ALPHA_SEED,
            label="WGAN"
        ),
        Line2D(
            [0], [0],
            color=COLORS["DDPM"],
            lw=2.2,
            alpha=ALPHA_SEED,
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
        left=0.075,
        right=0.995,
        top=0.94,
        bottom=0.105,
        wspace=0.08,
        hspace=0.18
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