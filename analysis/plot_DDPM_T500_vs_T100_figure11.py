#!/usr/bin/env python3
# ============================================================
# Script: analysis/plot_DDPM_T500_vs_T100_figure11.py
#
# Figure 11: DDPM T500 vs T100 comparison, 16x
#
# Panels:
#   (a) Horizontal spatial autocorrelation
#   (b) Vertical spatial autocorrelation
#   (c) Exceedance probability
#   (d) Q-Q plot
#
# Notes:
#   - Uses revised 16x Northeast-only test split.
#   - Q-Q panel uses independent seed-wise lines.
#   - No bold text is used.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# USER CONFIGURATION
# ============================================================

DATASET_PATH = "/scratch/wpa8me/ERA5_land/Dataset/dataset_splits_16x.npz"

PRED_DIR_DDPM500 = "/scratch/wpa8me/New16x/predictions_multiseed_16x1/T500"
PRED_DIR_DDPM100 = "/scratch/wpa8me/New16x/predictions_multiseed_16x1/T100"

OUT_DIR = "./Figure11_DDPM_T500_T100"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(
    OUT_DIR,
    "Figure11_DDPM_T500_vs_T100_seedwise_QQ_composite.png"
)

OUT_PDF = os.path.join(
    OUT_DIR,
    "Figure11_DDPM_T500_vs_T100_seedwise_QQ_composite.pdf"
)

SEED_LIST_T500 = list(range(1, 11))
SEED_LIST_T100 = list(range(1, 11))


# ============================================================
# SETTINGS
# ============================================================

MAX_LAG = 8
THRESHOLD_TAIL = 10.0
QQ_THRESHOLD = 1.0
QQ_POINTS = 200

N_SAMPLES = None

FIG_WIDTH = 14
FIG_HEIGHT = 10

FONTSIZE_LABEL = 17
FONTSIZE_TICK = 14
FONTSIZE_PANEL = 16
FONTSIZE_LEGEND = 14

ALPHA_LINES = 0.45
BOX_ALPHA = 0.68

COLORS = {
    "Target (ERA5-Land)": "#2baad3",
    "DDPM (T=500)": "#ca5478",
    "DDPM (T=100)": "#986bc5",
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

sns.set_theme(style="white")


# ============================================================
# HELPERS
# ============================================================

def squeeze_nhw(arr):
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
    y = squeeze_nhw(data["Ytest"])

    if N_SAMPLES is not None:
        y = y[:N_SAMPLES]

    return y


def filter_flat(arr, threshold):
    arr = np.ravel(arr)
    arr = arr[np.isfinite(arr)]
    return arr[arr > threshold]


def load_predictions(pred_dir, seed_list, label, n_samples):
    pred_dict = {}

    for seed in seed_list:
        fpath = os.path.join(
            pred_dir,
            f"Xtest_predictions_{seed}.npy"
        )

        if not os.path.exists(fpath):
            print(f"Missing {label} seed {seed}: {fpath}")
            continue

        arr = squeeze_nhw(np.load(fpath))[:n_samples]
        pred_dict[seed] = arr

        print(f"Loaded {label} seed {seed}: {arr.shape}")

    if len(pred_dict) == 0:
        raise RuntimeError(f"No predictions loaded for {label} from {pred_dir}")

    return pred_dict


# ============================================================
# SPATIAL AUTOCORRELATION
# ============================================================

def spatial_corr(images, max_lag=8, model_name="Unknown", run_id=None):
    images = squeeze_nhw(images)

    _, h, w = images.shape
    rows = []

    for img in images:

        for lag in range(1, max_lag + 1):

            if lag < w:
                x1 = img[:, :-lag].ravel()
                x2 = img[:, lag:].ravel()

                if np.std(x1) > 1e-6 and np.std(x2) > 1e-6:
                    c = np.corrcoef(x1, x2)[0, 1]

                    if np.isfinite(c):
                        rows.append([
                            model_name,
                            lag,
                            c,
                            "Horizontal",
                            run_id
                        ])

            if lag < h:
                y1 = img[:-lag, :].ravel()
                y2 = img[lag:, :].ravel()

                if np.std(y1) > 1e-6 and np.std(y2) > 1e-6:
                    c = np.corrcoef(y1, y2)[0, 1]

                    if np.isfinite(c):
                        rows.append([
                            model_name,
                            lag,
                            c,
                            "Vertical",
                            run_id
                        ])

    return pd.DataFrame(
        rows,
        columns=["Model", "Lag", "Correlation", "Direction", "Run"]
    )


# ============================================================
# DISTRIBUTIONAL PANELS
# ============================================================

def exceedance_xy(arr, threshold):
    vals = filter_flat(arr, threshold)

    if vals.size == 0:
        return None, None

    xs = np.sort(vals)
    F = np.arange(1, xs.size + 1) / xs.size

    return xs, 1.0 - F


def quantile_line(arr, q_space, threshold=1.0):
    vals = filter_flat(arr, threshold)

    if vals.size == 0:
        return None

    return np.quantile(vals, q_space)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Loading target and DDPM predictions")
    print("=" * 70)

    y_ref = load_target(DATASET_PATH)
    n_samples = len(y_ref)

    print(f"Target: {y_ref.shape}")

    t500_preds = load_predictions(
        PRED_DIR_DDPM500,
        SEED_LIST_T500,
        "T500",
        n_samples
    )

    t100_preds = load_predictions(
        PRED_DIR_DDPM100,
        SEED_LIST_T100,
        "T100",
        n_samples
    )

    n_common = min(
        len(y_ref),
        min(arr.shape[0] for arr in t500_preds.values()),
        min(arr.shape[0] for arr in t100_preds.values())
    )

    y_ref = y_ref[:n_common]

    for seed in t500_preds:
        t500_preds[seed] = t500_preds[seed][:n_common]

    for seed in t100_preds:
        t100_preds[seed] = t100_preds[seed][:n_common]

    print(f"Common N: {n_common}")

    # --------------------------------------------------------
    # Spatial correlations
    # --------------------------------------------------------
    print("\nComputing spatial autocorrelations")

    dfs = []

    dfs.append(
        spatial_corr(
            y_ref,
            max_lag=MAX_LAG,
            model_name="Target (ERA5-Land)",
            run_id=0
        )
    )

    for seed, arr in t500_preds.items():
        dfs.append(
            spatial_corr(
                arr,
                max_lag=MAX_LAG,
                model_name="DDPM (T=500)",
                run_id=seed
            )
        )

    for seed, arr in t100_preds.items():
        dfs.append(
            spatial_corr(
                arr,
                max_lag=MAX_LAG,
                model_name="DDPM (T=100)",
                run_id=seed
            )
        )

    df = pd.concat(dfs, ignore_index=True)

    print(f"Combined lagged correlations: {df.shape}")

    # --------------------------------------------------------
    # Target curves
    # --------------------------------------------------------
    q_space = np.linspace(0.001, 0.999, QQ_POINTS)

    target_ex_x, target_ex_y = exceedance_xy(
        y_ref,
        THRESHOLD_TAIL
    )

    truth_q_all = quantile_line(
        y_ref,
        q_space,
        threshold=QQ_THRESHOLD
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------
    print("\nCreating Figure 11")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        constrained_layout=False
    )

    axA, axB, axC, axD = axes.ravel()

    # ========================================================
    # (a) Horizontal spatial autocorrelation
    # ========================================================
    sns.boxplot(
        data=df[df["Direction"] == "Horizontal"],
        x="Lag",
        y="Correlation",
        hue="Model",
        palette=COLORS,
        showfliers=False,
        saturation=1,
        boxprops=dict(alpha=BOX_ALPHA),
        linewidth=1.0,
        ax=axA
    )

    axA.set_xlabel("Spatial lag (pixels)", labelpad=6)
    axA.set_ylabel("Spatial autocorrelation\n(horizontal direction)", labelpad=6)
    axA.set_ylim(0.0, 1.0)
    axA.grid(False)

    if axA.get_legend() is not None:
        axA.get_legend().remove()

    axA.text(
        0.01,
        0.98,
        "(a)",
        transform=axA.transAxes,
        fontsize=FONTSIZE_PANEL,
        fontweight="normal",
        va="top",
        ha="left"
    )

    # ========================================================
    # (b) Vertical spatial autocorrelation
    # ========================================================
    sns.boxplot(
        data=df[df["Direction"] == "Vertical"],
        x="Lag",
        y="Correlation",
        hue="Model",
        palette=COLORS,
        showfliers=False,
        saturation=1,
        boxprops=dict(alpha=BOX_ALPHA),
        linewidth=1.0,
        ax=axB
    )

    axB.set_xlabel("Spatial lag (pixels)", labelpad=6)
    axB.set_ylabel("Spatial autocorrelation\n(vertical direction)", labelpad=6)
    axB.set_ylim(0.0, 1.0)
    axB.grid(False)

    if axB.get_legend() is not None:
        axB.get_legend().remove()

    axB.text(
        0.01,
        0.98,
        "(b)",
        transform=axB.transAxes,
        fontsize=FONTSIZE_PANEL,
        fontweight="normal",
        va="top",
        ha="left"
    )

    # ========================================================
    # (c) Exceedance probability
    # ========================================================
    if target_ex_x is not None:
        axC.plot(
            target_ex_x,
            target_ex_y,
            color=COLORS["Target (ERA5-Land)"],
            linewidth=2.4,
            zorder=5
        )

    for _, arr in t500_preds.items():
        xs, ys = exceedance_xy(arr, THRESHOLD_TAIL)

        if xs is None:
            continue

        axC.plot(
            xs,
            ys,
            color=COLORS["DDPM (T=500)"],
            alpha=ALPHA_LINES,
            linewidth=1.2
        )

    for _, arr in t100_preds.items():
        xs, ys = exceedance_xy(arr, THRESHOLD_TAIL)

        if xs is None:
            continue

        axC.plot(
            xs,
            ys,
            color=COLORS["DDPM (T=100)"],
            alpha=ALPHA_LINES,
            linewidth=1.2
        )

    axC.set_yscale("log")
    axC.set_xlim(THRESHOLD_TAIL, 250.0)
    axC.set_ylim(1e-8, 1.0)

    axC.set_xlabel("Precipitation threshold (mm/day)", labelpad=6)
    axC.set_ylabel("Exceedance probability", labelpad=6)
    axC.grid(False)

    axC.text(
        0.01,
        0.98,
        "(c)",
        transform=axC.transAxes,
        fontsize=FONTSIZE_PANEL,
        fontweight="normal",
        va="top",
        ha="left"
    )

    # ========================================================
    # (d) Q-Q plot
    # ========================================================
    axD.plot(
        truth_q_all,
        truth_q_all,
        linestyle="--",
        linewidth=2.0,
        color="black",
        label="1:1",
        zorder=10
    )

    qq_values = [truth_q_all]

    for _, arr in t500_preds.items():
        pred_q = quantile_line(
            arr,
            q_space,
            threshold=QQ_THRESHOLD
        )

        if pred_q is None:
            continue

        qq_values.append(pred_q)

        axD.plot(
            truth_q_all,
            pred_q,
            color=COLORS["DDPM (T=500)"],
            linewidth=1.2,
            alpha=ALPHA_LINES
        )

    for _, arr in t100_preds.items():
        pred_q = quantile_line(
            arr,
            q_space,
            threshold=QQ_THRESHOLD
        )

        if pred_q is None:
            continue

        qq_values.append(pred_q)

        axD.plot(
            truth_q_all,
            pred_q,
            color=COLORS["DDPM (T=100)"],
            linewidth=1.2,
            alpha=ALPHA_LINES
        )

    qq_max = np.nanmax([
        np.nanmax(q)
        for q in qq_values
        if q is not None
    ])

    axD.set_xlim(0, qq_max)
    axD.set_ylim(0, qq_max)

    axD.set_xlabel("Target quantiles (mm/day)", labelpad=6)
    axD.set_ylabel("Predicted quantiles (mm/day)", labelpad=6)
    axD.grid(False)

    axD.text(
        0.01,
        0.98,
        "(d)",
        transform=axD.transAxes,
        fontsize=FONTSIZE_PANEL,
        fontweight="normal",
        va="top",
        ha="left"
    )

    # ========================================================
    # Shared legend
    # ========================================================
    handles = [
        plt.Line2D(
            [0], [0],
            color=COLORS["Target (ERA5-Land)"],
            lw=2.4,
            label="Target (ERA5-Land)"
        ),
        plt.Line2D(
            [0], [0],
            color=COLORS["DDPM (T=500)"],
            lw=2.0,
            label="DDPM (T=500)"
        ),
        plt.Line2D(
            [0], [0],
            color=COLORS["DDPM (T=100)"],
            lw=2.0,
            label="DDPM (T=100)"
        ),
        plt.Line2D(
            [0], [0],
            color="black",
            lw=2.0,
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
        bbox_to_anchor=(0.5, 0.01)
    )

    plt.subplots_adjust(
        left=0.08,
        right=0.99,
        top=0.97,
        bottom=0.12,
        wspace=0.22,
        hspace=0.22
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


