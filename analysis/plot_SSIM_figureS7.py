#!/usr/bin/env python3
# ============================================================
# Script: analysis/plot_SSIM_figureS7.py
#
# Figure S6: Sample-wise SSIM distributions
#
# Models: U-Net, WGAN, DDPM
# Scales: 8x and 16x
#
# Notes:
#   - Uses revised Northeast-only test splits:
#       dataset_splits.npz      for 8x
#       dataset_splits_16x.npz  for 16x
#   - SSIM is computed for each predicted image against
#     Target (ERA5-Land), pooled across all 10 seeds.
#   - No bold text is used.
# ============================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# ============================================================
# USER CONFIGURATION
# ============================================================

DATASET_8X_PATH = "/scratch/wpa8me/ERA5_land/Dataset/dataset_splits.npz"
DATASET_16X_PATH = "/scratch/wpa8me/ERA5_land/Dataset/dataset_splits_16x.npz"

UNET_DIR_8X = "/scratch/wpa8me/unet_runs"
WGAN_DIR_8X = "/scratch/wpa8me/unet_runs/WGANs"
DDPM_DIR_8X = "/scratch/wpa8me/New_DDPM8x/predictions_multiseed_8x/T500"

UNET_DIR_16X = "/scratch/wpa8me/unet_runs_16x/Same"
WGAN_DIR_16X = "/scratch/wpa8me/unet_runs_16x/Same/WGANs"
DDPM_DIR_16X = "/scratch/wpa8me/New16x/predictions_multiseed_16x1/T500"

OUT_DIR = "./FigureS6_SSIM_violin"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "FigureS6_SSIM_seedwise_values.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "FigureS6_SSIM_seedwise_summary.csv")

OUT_PNG = os.path.join(OUT_DIR, "FigureS6_SSIM_violin_8x_16x.png")
OUT_PDF = os.path.join(OUT_DIR, "FigureS6_SSIM_violin_8x_16x.pdf")


# ============================================================
# SETTINGS
# ============================================================

SEEDS = list(range(1, 11))

N_SAMPLES = None
BATCH_SIZE = 32

NOISE_STD = 1.0
NOISE_SEED = 2026

DATA_RANGE_MODE = "target_global"
RECOMPUTE_SSIM = True


# ============================================================
# STYLE
# ============================================================

MODEL_ORDER = ["U-Net", "WGAN", "DDPM"]
SCALE_ORDER = ["8x", "16x"]

COLORS = {
    "U-Net": "#c57a3e",
    "WGAN": "#986bc5",
    "DDPM": "#ca5478",
}

FONTSIZE_TITLE = 15
FONTSIZE_LABEL = 14
FONTSIZE_TICK = 12
FONTSIZE_PANEL = 14

YMIN = 0.60
YMAX = 1.00

plt.rcParams.update({
    "font.size": FONTSIZE_LABEL,
    "axes.labelsize": FONTSIZE_LABEL,
    "axes.titlesize": FONTSIZE_TITLE,
    "xtick.labelsize": FONTSIZE_TICK,
    "ytick.labelsize": FONTSIZE_TICK,
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
})


# ============================================================
# HELPERS
# ============================================================

def squeeze_nhw(arr, name="array"):
    arr = np.asarray(arr)

    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    arr = np.squeeze(arr)

    if arr.ndim != 3:
        raise ValueError(f"{name}: expected shape (N,H,W), got {arr.shape}")

    return arr.astype(np.float32)


def ensure_channel(arr):
    arr = np.asarray(arr)

    if arr.ndim == 3:
        arr = arr[..., np.newaxis]

    return arr.astype(np.float32)


def load_test_split(dataset_path, scale):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{scale} dataset split not found: {dataset_path}")

    data = np.load(dataset_path)

    X = data["Xtest"].astype(np.float32)
    Y = squeeze_nhw(data["Ytest"], f"Target {scale}")

    if N_SAMPLES is not None:
        X = X[:N_SAMPLES]
        Y = Y[:N_SAMPLES]

    return X, Y


def compute_ssim_per_sample(target, pred, data_range_mode="target_global"):
    target = squeeze_nhw(target, "target")
    pred = squeeze_nhw(pred, "prediction")

    n = min(len(target), len(pred))
    target = target[:n]
    pred = pred[:n]

    if data_range_mode == "target_global":
        global_range = float(np.nanmax(target) - np.nanmin(target))
        if global_range <= 0:
            global_range = 1.0

    values = []

    for i in range(n):
        y = target[i]
        p = pred[i]

        if data_range_mode == "pair":
            dr = float(
                max(np.nanmax(y), np.nanmax(p)) -
                min(np.nanmin(y), np.nanmin(p))
            )
            if dr <= 0:
                dr = 1.0
        else:
            dr = global_range

        values.append(ssim(y, p, data_range=dr))

    return np.asarray(values, dtype=np.float32)


def infer_tf_model(model_path, X, seed):
    if not os.path.exists(model_path):
        print(f"Missing model: {model_path}")
        return None

    X_in = ensure_channel(X)

    rng = np.random.default_rng(NOISE_SEED + seed)

    noise = rng.normal(
        0.0,
        NOISE_STD,
        size=X_in.shape
    ).astype(np.float32)

    model = tf.keras.models.load_model(model_path, compile=False)

    preds = model.predict(
        [X_in, noise],
        batch_size=BATCH_SIZE,
        verbose=0
    )

    preds = squeeze_nhw(preds, "preds")

    del model
    tf.keras.backend.clear_session()

    return preds


def load_ddpm_prediction(ddpm_dir, seed, scale):
    if scale == "8x":
        fname = f"Xtest_predictions_seed1_{seed}.npy"
    elif scale == "16x":
        fname = f"Xtest_predictions_{seed}.npy"
    else:
        raise ValueError("scale must be '8x' or '16x'")

    path = os.path.join(ddpm_dir, fname)

    if not os.path.exists(path):
        print(f"Missing DDPM prediction: {path}")
        return None

    return squeeze_nhw(np.load(path), f"DDPM {scale} seed {seed}")


def collect_ssim_for_scale(scale, dataset_path, unet_dir, wgan_dir, ddpm_dir):
    print("=" * 70)
    print(f"Processing {scale}")
    print("=" * 70)

    X, Y = load_test_split(dataset_path, scale)

    print(f"{scale} Xtest: {X.shape}")
    print(f"{scale} Ytest: {Y.shape}")

    records = []

    for seed in SEEDS:
        print(f"\n{scale} | seed {seed}")

        # ----------------------------
        # U-Net
        # ----------------------------
        unet_path = os.path.join(
            unet_dir,
            f"unet_generator_best_seed{seed}.h5"
        )

        unet_pred = infer_tf_model(
            unet_path,
            X,
            seed
        )

        if unet_pred is not None:
            vals = compute_ssim_per_sample(
                Y,
                unet_pred,
                DATA_RANGE_MODE
            )

            records.extend([
                {
                    "Scale": scale,
                    "Model": "U-Net",
                    "Seed": seed,
                    "SSIM": float(v)
                }
                for v in vals
            ])

            print(f"  U-Net SSIM mean: {np.mean(vals):.4f}")

        # ----------------------------
        # WGAN
        # ----------------------------
        wgan_path = os.path.join(
            wgan_dir,
            f"gen_final_seed{seed}.keras"
        )

        wgan_pred = infer_tf_model(
            wgan_path,
            X,
            seed
        )

        if wgan_pred is not None:
            vals = compute_ssim_per_sample(
                Y,
                wgan_pred,
                DATA_RANGE_MODE
            )

            records.extend([
                {
                    "Scale": scale,
                    "Model": "WGAN",
                    "Seed": seed,
                    "SSIM": float(v)
                }
                for v in vals
            ])

            print(f"  WGAN SSIM mean: {np.mean(vals):.4f}")

        # ----------------------------
        # DDPM
        # ----------------------------
        ddpm_pred = load_ddpm_prediction(
            ddpm_dir,
            seed,
            scale
        )

        if ddpm_pred is not None:
            vals = compute_ssim_per_sample(
                Y,
                ddpm_pred,
                DATA_RANGE_MODE
            )

            records.extend([
                {
                    "Scale": scale,
                    "Model": "DDPM",
                    "Seed": seed,
                    "SSIM": float(v)
                }
                for v in vals
            ])

            print(f"  DDPM SSIM mean: {np.mean(vals):.4f}")

    return pd.DataFrame(records)


def plot_violin(df):
    required_cols = {"Scale", "Model", "Seed", "SSIM"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in SSIM dataframe: {missing}")

    df = df[np.isfinite(df["SSIM"])].copy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.8),
        sharey=True,
        constrained_layout=False
    )

    panel_labels = ["(a)", "(b)"]

    for ax, scale, panel in zip(axes, SCALE_ORDER, panel_labels):

        sub = df[df["Scale"] == scale]

        data = []

        for model in MODEL_ORDER:
            vals = sub[sub["Model"] == model]["SSIM"].values
            vals = vals[np.isfinite(vals)]
            data.append(vals)

            print(
                f"{scale} {model}: n={len(vals)}, "
                f"median={np.median(vals):.4f}, "
                f"IQR=({np.percentile(vals, 25):.4f}, "
                f"{np.percentile(vals, 75):.4f})"
            )

        positions = np.arange(1, len(MODEL_ORDER) + 1)

        vp = ax.violinplot(
            data,
            positions=positions,
            widths=0.80,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )

        for body, model in zip(vp["bodies"], MODEL_ORDER):
            body.set_facecolor(COLORS[model])
            body.set_edgecolor("0.25")
            body.set_alpha(0.62)
            body.set_linewidth(0.8)

        for pos, vals in zip(positions, data):

            q1 = np.percentile(vals, 25)
            med = np.percentile(vals, 50)
            q3 = np.percentile(vals, 75)

            w_low = np.percentile(vals, 5)
            w_high = np.percentile(vals, 95)

            ax.plot(
                [pos, pos],
                [w_low, w_high],
                color="0.20",
                linewidth=1.1,
                zorder=5
            )

            box_width = 0.06

            rect = plt.Rectangle(
                (pos - box_width / 2, q1),
                box_width,
                q3 - q1,
                facecolor="0.20",
                edgecolor="0.20",
                linewidth=0.8,
                alpha=0.90,
                zorder=6
            )

            ax.add_patch(rect)

            ax.scatter(
                pos,
                med,
                s=18,
                color="white",
                edgecolor="0.20",
                linewidth=0.5,
                zorder=7
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(MODEL_ORDER, fontsize=FONTSIZE_TICK)

        ax.set_ylim(YMIN, YMAX)

        ax.set_title(
            f"{scale} downscaling",
            fontsize=FONTSIZE_TITLE,
            fontweight="normal",
            pad=8
        )

        ax.grid(axis="y", alpha=0.14)

        ax.text(
            0.03,
            0.97,
            panel,
            transform=ax.transAxes,
            fontsize=FONTSIZE_PANEL,
            fontweight="normal",
            va="top",
            ha="left"
        )

        ax.tick_params(labelsize=FONTSIZE_TICK)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Sample-wise SSIM",
        fontsize=FONTSIZE_LABEL,
        fontweight="normal"
    )

    axes[1].set_ylabel("")

    plt.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.16,
        top=0.90,
        wspace=0.08
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

    print("\nSaved figure:")
    print(OUT_PNG)
    print(OUT_PDF)


# ============================================================
# MAIN
# ============================================================

def main():

    if RECOMPUTE_SSIM or not os.path.exists(OUT_CSV):

        df8 = collect_ssim_for_scale(
            scale="8x",
            dataset_path=DATASET_8X_PATH,
            unet_dir=UNET_DIR_8X,
            wgan_dir=WGAN_DIR_8X,
            ddpm_dir=DDPM_DIR_8X
        )

        df16 = collect_ssim_for_scale(
            scale="16x",
            dataset_path=DATASET_16X_PATH,
            unet_dir=UNET_DIR_16X,
            wgan_dir=WGAN_DIR_16X,
            ddpm_dir=DDPM_DIR_16X
        )

        df = pd.concat([df8, df16], ignore_index=True)

        df.to_csv(OUT_CSV, index=False)

        summary = (
            df.groupby(["Scale", "Model", "Seed"])["SSIM"]
            .agg(["mean", "median", "std", "count"])
            .reset_index()
        )

        summary.to_csv(OUT_SUMMARY, index=False)

        print("\nSaved SSIM values:")
        print(OUT_CSV)
        print(OUT_SUMMARY)

    else:
        print(f"Loading existing SSIM values from {OUT_CSV}")
        df = pd.read_csv(OUT_CSV)

    plot_violin(df)


if __name__ == "__main__":
    main()
