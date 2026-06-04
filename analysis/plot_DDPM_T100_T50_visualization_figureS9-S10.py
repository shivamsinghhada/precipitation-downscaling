#!/usr/bin/env python3
# ============================================================
# Script: analysis/plot_DDPM_T100_T50_visualization_figureS9-S10.py
#
# Supplementary Figures S9-S10:
#     DDPM accelerated inference examples for 16x downscaling.
#
# Figures:
#     Figure S9  : DDPM T=100, 5 examples, 5 seeds
#     Figure S10 : DDPM T=50,  5 examples, 5 seeds
#
# Notes:
#     - Uses revised 16x Northeast-only test split.
#     - Uses already saved DDPM predictions.
#     - No model loading or inference.
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# USER CONFIGURATION
# ============================================================

DATASET_PATH = "/scratch/wpa8me/ERA5_land/Dataset/dataset_splits_16x.npz"

PRED_DIRS = {
    "S9_T100": "/scratch/wpa8me/New16x/predictions_multiseed_16x1/T100",
    "S10_T50": "/scratch/wpa8me/New16x/predictions_multiseed_16x1/T50",
}

OUT_DIR = "./Supplementary_Figures_S9_S10"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [1, 2, 3, 4, 5]
SAMPLE_INDICES = [0, 1, 2, 3, 4]


# ============================================================
# STYLE
# ============================================================

NUM_SAMPLES = len(SAMPLE_INDICES)

FIG_WIDTH = 18
FIG_HEIGHT = 3.1 * NUM_SAMPLES

FONTSIZE_TITLE = 15
FONTSIZE_LABEL = 13
FONTSIZE_TICK = 10

plt.rcParams.update({
    "font.size": FONTSIZE_LABEL,
    "axes.titlesize": FONTSIZE_TITLE,
    "axes.labelsize": FONTSIZE_LABEL,
    "xtick.labelsize": FONTSIZE_TICK,
    "ytick.labelsize": FONTSIZE_TICK,
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
})


# ============================================================
# COLORMAP
# ============================================================

CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "precip_cmap",
    [
        "#ffffff",
        "#d7f0ff",
        "#7fcdbb",
        "#41b6c4",
        "#2c7fb8",
        "#253494",
        "#fed976",
        "#fd8d3c",
        "#e31a1c",
        "#800026",
    ],
    N=256
)


# ============================================================
# HELPERS
# ============================================================

def squeeze_hw(arr):
    arr = np.asarray(arr)

    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    return np.squeeze(arr).astype(np.float32)


def load_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset split not found: {dataset_path}")

    data = np.load(dataset_path)

    Xtest = squeeze_hw(data["Xtest"])
    Ytest = squeeze_hw(data["Ytest"])

    return Xtest, Ytest


def load_predictions(pred_dir, seeds, sample_indices):
    preds = {}

    for seed in seeds:
        path = os.path.join(
            pred_dir,
            f"Xtest_predictions_{seed}.npy"
        )

        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue

        arr = squeeze_hw(np.load(path))
        preds[seed] = arr[sample_indices]

        print(f"Loaded seed {seed}: {preds[seed].shape}")

    if len(preds) == 0:
        raise RuntimeError(f"No predictions loaded from {pred_dir}")

    return preds


def make_figure(X_vis, Y_vis, preds, tag, fig_label):
    n_cols = 2 + len(SEEDS)

    fig, axes = plt.subplots(
        NUM_SAMPLES,
        n_cols,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        constrained_layout=False
    )

    if NUM_SAMPLES == 1:
        axes = axes[np.newaxis, :]

    column_titles = (
        ["Low-resolution input", "Target (ERA5-Land)"] +
        [f"Seed {seed}" for seed in SEEDS]
    )

    for i in range(NUM_SAMPLES):

        available_preds = [
            preds[seed][i].ravel()
            for seed in SEEDS
            if seed in preds
        ]

        row_max = np.nanpercentile(
            np.concatenate(
                [Y_vis[i].ravel()] + available_preds
            ),
            99.5
        )

        row_min = 0.0

        panels = [X_vis[i], Y_vis[i]]

        for seed in SEEDS:
            if seed in preds:
                panels.append(preds[seed][i])
            else:
                panels.append(np.full_like(Y_vis[i], np.nan))

        for j, img in enumerate(panels):

            ax = axes[i, j]

            im = ax.imshow(
                img,
                cmap=CUSTOM_CMAP,
                origin="lower",
                vmin=row_min,
                vmax=row_max
            )

            if i == 0:
                ax.set_title(
                    column_titles[j],
                    fontsize=FONTSIZE_TITLE,
                    fontweight="normal",
                    pad=8
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if j == 0:
                ax.set_ylabel(
                    f"Sample {i + 1}",
                    fontsize=FONTSIZE_LABEL,
                    fontweight="normal",
                    labelpad=8
                )

            cbar = plt.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.030
            )

            cbar.ax.tick_params(labelsize=FONTSIZE_TICK)

    plt.subplots_adjust(
        left=0.045,
        right=0.995,
        top=0.925,
        bottom=0.045,
        wspace=0.25,
        hspace=0.14
    )

    out_png = os.path.join(
        OUT_DIR,
        f"Figure{fig_label}_DDPM_{tag}_5examples_5seeds.png"
    )

    out_pdf = os.path.join(
        OUT_DIR,
        f"Figure{fig_label}_DDPM_{tag}_5examples_5seeds.pdf"
    )

    fig.savefig(
        out_png,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.03
    )

    fig.savefig(
        out_pdf,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.03
    )

    plt.show()

    print("\nSaved:")
    print(out_png)
    print(out_pdf)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Loading revised 16x test split")
    print("=" * 70)

    Xtest, Ytest = load_dataset(DATASET_PATH)

    X_vis = Xtest[SAMPLE_INDICES]
    Y_vis = Ytest[SAMPLE_INDICES]

    print("X_vis:", X_vis.shape)
    print("Y_vis:", Y_vis.shape)

    for key, pred_dir in PRED_DIRS.items():

        if key == "S9_T100":
            tag = "T100"
            fig_label = "S9"
        elif key == "S10_T50":
            tag = "T50"
            fig_label = "S10"
        else:
            raise ValueError(f"Unknown figure key: {key}")

        print("\n" + "=" * 70)
        print(f"Creating Figure {fig_label}: DDPM {tag}")
        print("=" * 70)

        preds = load_predictions(
            pred_dir=pred_dir,
            seeds=SEEDS,
            sample_indices=SAMPLE_INDICES
        )

        make_figure(
            X_vis=X_vis,
            Y_vis=Y_vis,
            preds=preds,
            tag=tag,
            fig_label=fig_label
        )


if __name__ == "__main__":
    main()
