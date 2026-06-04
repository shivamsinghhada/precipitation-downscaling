#!/usr/bin/env python3
# ============================================================
# Script: analysis/plot_variability_composite_Figure9.py
#
# Figure 9: Within-seed and across-seed variability composite
#
# Panels:
#   (a1, a2) Horizontal spatial autocorrelation
#   (b1, b2) Exceedance probability
#   (c1, c2) Radial power spectrum
#
# Columns:
#   Left  : within-seed variability
#   Right : across-seed variability
#
# Notes:
#   - Uses the revised 16x Northeast-only test split.
#   - Exceedance probability is shown from 10 to 250 mm/day.
#   - Power spectrum is shown as wavelength (km) vs PSD in dB.
#   - No bold text is used.
#
# Usage:
#   python analysis/plot_variability_composite_Figure9.py
# ============================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


# ============================================================
# USER CONFIGURATION
# ============================================================

DATASET_PATH = "/scratch/wpa8me/ERA5_land/Dataset/dataset_splits_16x.npz"

UNET_DIR = "/scratch/wpa8me/unet_runs_16x/Same"
WGAN_DIR = os.path.join(UNET_DIR, "WGANs")

BEST_UNET_SEED = 5
BEST_WGAN_SEED = 3

SEED_LIST = list(range(1, 11))

DDPM_WITHIN_SEED = 10

DDPM_WITHIN_DIR = (
    f"/scratch/wpa8me/New16x/predictions_multi_sample/T500/"
    f"seed{DDPM_WITHIN_SEED}"
)

DDPM_ACROSS_DIR = "/scratch/wpa8me/New16x/predictions_multiseed_16x1/T500"

OUT_DIR = "./Figure9_variability_composite"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, "Figure9_variability_composite.png")
OUT_PDF = os.path.join(OUT_DIR, "Figure9_variability_composite.pdf")


# ============================================================
# SETTINGS
# ============================================================

N_SAMPLES = None

INTRA_DRAWS = 10

NOISE_STD_INTRA = 1.0
NOISE_BASE_SEED_INTRA = 2026

NOISE_STD_INTER = 1.0
NOISE_BASE_SEED_INTER = 42

BATCH_SIZE = 32

MAX_LAG = 8

TAIL_THRESHOLD = 10.0
TAIL_XMAX = 250.0
N_THRESHOLDS = 300

DX_KM = 9.0
DOMAIN_PIXELS = 128
DOMAIN_LENGTH_KM = DOMAIN_PIXELS * DX_KM

REGENERATE_INTRA_UNET = True
REGENERATE_INTRA_WGAN = True
REGENERATE_INTER_UNET = True
REGENERATE_INTER_WGAN = True

INTRA_UNET_CACHE = os.path.join(
    OUT_DIR,
    f"cache_intra_unet_seed{BEST_UNET_SEED}.npy"
)

INTRA_WGAN_CACHE = os.path.join(
    OUT_DIR,
    f"cache_intra_wgan_seed{BEST_WGAN_SEED}.npy"
)

INTER_UNET_CACHE = os.path.join(
    OUT_DIR,
    "cache_inter_unet.npy"
)

INTER_WGAN_CACHE = os.path.join(
    OUT_DIR,
    "cache_inter_wgan.npy"
)


# ============================================================
# STYLE
# ============================================================

FONTSIZE_TITLE = 15
FONTSIZE_LABEL = 13
FONTSIZE_TICK = 11
FONTSIZE_PANEL = 13
FONTSIZE_LEGEND = 12

LINEWIDTH_TARGET = 2.8
LINEWIDTH_MEMBER = 1.2
LINE_ALPHA = 0.42

FIG_WIDTH = 15.0
FIG_HEIGHT = 13.0

TARGET_LABEL = "Target (ERA5-Land)"

UNET_WITHIN = "U-Net within-seed"
WGAN_WITHIN = "WGAN within-seed"
DDPM_WITHIN = "DDPM within-seed"

UNET_ACROSS = "U-Net across-seed"
WGAN_ACROSS = "WGAN across-seed"
DDPM_ACROSS = "DDPM across-seed"

COLORS = {
    TARGET_LABEL: "#2baad3",
    "U-Net": "#c57a3e",
    "WGAN": "#986bc5",
    "DDPM": "#ca5478",
}

MODEL_COLOR = {
    TARGET_LABEL: COLORS[TARGET_LABEL],

    UNET_WITHIN: COLORS["U-Net"],
    WGAN_WITHIN: COLORS["WGAN"],
    DDPM_WITHIN: COLORS["DDPM"],

    UNET_ACROSS: COLORS["U-Net"],
    WGAN_ACROSS: COLORS["WGAN"],
    DDPM_ACROSS: COLORS["DDPM"],
}

MODEL_ORDER_WITHIN = [
    TARGET_LABEL,
    UNET_WITHIN,
    WGAN_WITHIN,
    DDPM_WITHIN,
]

MODEL_ORDER_ACROSS = [
    TARGET_LABEL,
    UNET_ACROSS,
    WGAN_ACROSS,
    DDPM_ACROSS,
]

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
# BASIC HELPERS
# ============================================================

def squeeze_nhw(arr):
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


def stack_members(arr4d):
    if arr4d.ndim != 4:
        raise ValueError(f"Expected shape (N,M,H,W), got {arr4d.shape}")

    n, m, h, w = arr4d.shape
    return arr4d.reshape(n * m, h, w)


def standardize_members_4d(arr, expected_n=None, name="array"):
    arr = np.squeeze(np.asarray(arr, dtype=np.float32))

    if arr.ndim == 3:
        arr = arr[:, np.newaxis, :, :]

    elif arr.ndim == 4:

        if expected_n is not None:

            if arr.shape[0] == expected_n:
                pass

            elif arr.shape[1] == expected_n:
                arr = np.transpose(arr, (1, 0, 2, 3))

            elif arr.shape[0] < expected_n:
                print(f"Warning: {name} has fewer samples than expected.")

            else:
                raise ValueError(
                    f"{name}: cannot identify sample axis. "
                    f"Shape={arr.shape}, expected_n={expected_n}"
                )

    else:
        raise ValueError(f"{name}: expected 3D or 4D array, got {arr.shape}")

    return arr.astype(np.float32)


def load_test_data(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset split not found: {dataset_path}")

    data = np.load(dataset_path)

    Xtest = data["Xtest"].astype(np.float32)
    Ytest = squeeze_nhw(data["Ytest"])

    if N_SAMPLES is not None:
        Xtest = Xtest[:N_SAMPLES]
        Ytest = Ytest[:N_SAMPLES]

    return Xtest, Ytest


# ============================================================
# TENSORFLOW GENERATION
# ============================================================

def generate_tf_samples(
    model,
    X,
    n_draws,
    noise_std,
    noise_base_seed,
    batch_size=32,
):
    X_in = ensure_channel(X)
    preds_all = []

    for draw_idx in range(n_draws):

        rng = np.random.default_rng(noise_base_seed + draw_idx)

        noise = rng.normal(
            0.0,
            noise_std,
            size=X_in.shape
        ).astype(np.float32)

        preds = model.predict(
            [X_in, noise],
            batch_size=batch_size,
            verbose=0
        )

        preds_all.append(squeeze_nhw(preds))

        print(f"    draw {draw_idx + 1}/{n_draws} done")

    return np.stack(preds_all, axis=1).astype(np.float32)


def generate_one_draw_per_seed_tf(
    seed_list,
    model_template,
    X,
    noise_std,
    noise_seed,
    batch_size=32,
):
    X_in = ensure_channel(X)

    rng = np.random.default_rng(noise_seed)

    noise = rng.normal(
        0.0,
        noise_std,
        size=X_in.shape
    ).astype(np.float32)

    all_draws = []
    used_seeds = []

    for seed in seed_list:

        model_path = model_template.format(seed=seed)

        if not os.path.exists(model_path):
            print(f"Missing model: {model_path}")
            continue

        model = tf.keras.models.load_model(model_path, compile=False)

        preds = model.predict(
            [X_in, noise],
            batch_size=batch_size,
            verbose=0
        )

        preds = squeeze_nhw(preds)

        all_draws.append(preds)
        used_seeds.append(seed)

        print(f"  seed {seed}: {preds.shape}")

        del model
        tf.keras.backend.clear_session()

    if len(all_draws) == 0:
        raise RuntimeError("No model predictions generated.")

    return np.stack(all_draws, axis=1).astype(np.float32), used_seeds


# ============================================================
# DDPM LOADING
# ============================================================

def load_ddpm_within_seed(ddpm_dir, seed, n_draws, n_samples):
    draws = []
    loaded = []

    for sample_id in range(1, n_draws + 1):

        path = os.path.join(
            ddpm_dir,
            f"Xtest_predictions_seed{seed}_sample{sample_id}.npy"
        )

        if not os.path.exists(path):
            print(f"Missing DDPM within-seed sample: {path}")
            continue

        arr = squeeze_nhw(np.load(path))[:n_samples]

        draws.append(arr)
        loaded.append(sample_id)

        print(f"  DDPM within sample {sample_id}: {arr.shape}")

    if len(draws) == 0:
        raise RuntimeError("No DDPM within-seed samples loaded.")

    print(f"Loaded DDPM within-seed samples: {loaded}")

    return np.stack(draws, axis=1).astype(np.float32)


def load_ddpm_across_seed(ddpm_dir, seed_list, n_samples):
    draws = []
    loaded = []

    for seed in seed_list:

        path = os.path.join(
            ddpm_dir,
            f"Xtest_predictions_{seed}.npy"
        )

        if not os.path.exists(path):
            print(f"Missing DDPM across-seed file: {path}")
            continue

        arr = squeeze_nhw(np.load(path))[:n_samples]

        draws.append(arr)
        loaded.append(seed)

        print(f"  DDPM across seed {seed}: {arr.shape}")

    if len(draws) == 0:
        raise RuntimeError("No DDPM across-seed predictions loaded.")

    print(f"Loaded DDPM across-seed files: {loaded}")

    return np.stack(draws, axis=1).astype(np.float32)


# ============================================================
# HORIZONTAL LAGGED AUTOCORRELATION
# ============================================================

def horizontal_correlation_distribution(images, max_lag, model_name):
    images = squeeze_nhw(images)
    records = []

    for img in images:

        h, w = img.shape

        for lag in range(1, max_lag + 1):

            if lag < w:

                x1 = img[:, :-lag].ravel()
                x2 = img[:, lag:].ravel()

                if np.std(x1) > 1e-6 and np.std(x2) > 1e-6:

                    corr = np.corrcoef(x1, x2)[0, 1]

                    if np.isfinite(corr):
                        records.append([model_name, lag, corr])

    return pd.DataFrame(
        records,
        columns=["Model", "Lag", "Correlation"]
    )


def draw_lag_boxplots(ax, df_sub, model_order, show_ylabel=True):
    lags = sorted(df_sub["Lag"].unique())

    positions = []
    data = []
    facecolors = []

    group_gap = 2.05
    intra_gap = 0.34
    box_width = 0.25

    xticks = []
    xticklabels = []

    for lag in lags:

        base = (lag - 1) * group_gap

        for m_idx, model in enumerate(model_order):

            vals = df_sub[
                (df_sub["Lag"] == lag) &
                (df_sub["Model"] == model)
            ]["Correlation"].values

            positions.append(base + m_idx * intra_gap)
            data.append(vals)
            facecolors.append(MODEL_COLOR[model])

        xticks.append(base + 1.5 * intra_gap)
        xticklabels.append(str(lag))

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(
            color="black",
            linewidth=0.9
        ),
        whiskerprops=dict(
            linewidth=0.7,
            color="0.25"
        ),
        capprops=dict(
            linewidth=0.7,
            color="0.25"
        ),
        boxprops=dict(
            linewidth=0.7,
            color="0.25"
        ),
    )

    for patch, color in zip(bp["boxes"], facecolors):
        patch.set_facecolor(color)
        patch.set_alpha(0.76)
        patch.set_edgecolor("0.25")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_xlabel(
        "Spatial lag (pixels)",
        fontsize=FONTSIZE_LABEL,
        fontweight="normal"
    )

    if show_ylabel:
        ax.set_ylabel(
            "Spatial autocorrelation\n(horizontal direction)",
            fontsize=FONTSIZE_LABEL,
            fontweight="normal"
        )
    else:
        ax.set_ylabel("")

    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.10)
    ax.tick_params(labelsize=FONTSIZE_TICK)


# ============================================================
# EXCEEDANCE PROBABILITY
# ============================================================

def exceedance_curve(values, thresholds):
    vals = values[np.isfinite(values)]
    vals = vals[vals >= 0]
    vals = np.sort(vals)

    probs = np.empty_like(thresholds, dtype=np.float64)

    for i, threshold in enumerate(thresholds):

        idx = np.searchsorted(
            vals,
            threshold,
            side="right"
        )

        probs[i] = (len(vals) - idx) / max(len(vals), 1)

    return probs


def eventwise_exceedance(images, thresholds):
    return np.asarray([
        exceedance_curve(img.ravel(), thresholds)
        for img in images
    ])


def memberwise_mean_exceedance(arr4d, thresholds):
    out = []

    for member in range(arr4d.shape[1]):

        curves = eventwise_exceedance(
            arr4d[:, member],
            thresholds
        )

        out.append(np.mean(curves, axis=0))

    return np.asarray(out)


def target_mean_exceedance(target3d, thresholds):
    curves = eventwise_exceedance(target3d, thresholds)
    return np.mean(curves, axis=0)


def plot_exceedance_lines(ax, thresholds, target_exc, member_dict, show_ylabel=True):
    ax.plot(
        thresholds,
        target_exc,
        color=COLORS[TARGET_LABEL],
        lw=LINEWIDTH_TARGET,
        zorder=10,
    )

    for name, lines in member_dict.items():

        color = MODEL_COLOR[name]

        for i in range(lines.shape[0]):

            ax.plot(
                thresholds,
                lines[i],
                color=color,
                lw=LINEWIDTH_MEMBER,
                alpha=LINE_ALPHA,
                zorder=3,
            )

    ax.set_yscale("log")
    ax.set_xlim(TAIL_THRESHOLD, TAIL_XMAX)
    ax.set_ylim(1e-8, 1.0)

    ax.set_xlabel(
        "Precipitation threshold (mm/day)",
        fontsize=FONTSIZE_LABEL,
        fontweight="normal"
    )

    if show_ylabel:
        ax.set_ylabel(
            "Exceedance probability",
            fontsize=FONTSIZE_LABEL,
            fontweight="normal"
        )
    else:
        ax.set_ylabel("")

    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(True, which="both", alpha=0.08)


# ============================================================
# RADIAL POWER SPECTRUM
# ============================================================

def radial_power_spectrum(image):
    img = np.asarray(image, dtype=np.float64)

    h, w = img.shape

    img = img - np.nanmean(img)

    fft_img = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(fft_img) ** 2

    ky = np.arange(-h // 2, h // 2)
    kx = np.arange(-w // 2, w // 2)

    kx_grid, ky_grid = np.meshgrid(kx, ky)

    kr = np.sqrt(kx_grid**2 + ky_grid**2).astype(int)

    max_bin = min(h, w) // 2

    spectrum = np.zeros(max_bin + 1)

    for r in range(max_bin + 1):

        mask = kr == r

        if np.any(mask):
            spectrum[r] = np.nanmean(power[mask])

    return spectrum


def eventwise_spectra(images):
    return np.asarray([
        radial_power_spectrum(img)
        for img in images
    ])


def memberwise_mean_spectra(arr4d):
    out = []

    for member in range(arr4d.shape[1]):

        spectra = eventwise_spectra(arr4d[:, member])

        out.append(
            np.nanmean(spectra[:, 1:], axis=0)
        )

    return np.asarray(out)


def target_mean_spectrum(target3d):
    spectra = eventwise_spectra(target3d)

    return np.nanmean(
        spectra[:, 1:],
        axis=0
    )


def to_db(power):
    return 10.0 * np.log10(
        np.clip(power, 1e-12, None)
    )


def plot_spectrum_lines(ax, wavelength_km, target_spec, member_dict, show_ylabel=True):
    ax.plot(
        wavelength_km,
        to_db(target_spec),
        color=COLORS[TARGET_LABEL],
        lw=LINEWIDTH_TARGET,
        zorder=10,
    )

    for name, lines in member_dict.items():

        color = MODEL_COLOR[name]

        for i in range(lines.shape[0]):

            ax.plot(
                wavelength_km,
                to_db(lines[i]),
                color=color,
                lw=LINEWIDTH_MEMBER,
                alpha=LINE_ALPHA,
                zorder=3,
            )

    ax.set_xscale("log")
    ax.invert_xaxis()

    ax.set_xlabel(
        "Wavelength (km)",
        fontsize=FONTSIZE_LABEL,
        fontweight="normal"
    )

    if show_ylabel:
        ax.set_ylabel(
            "Power spectral density (dB)\n$10\\log_{10}(P)$",
            fontsize=FONTSIZE_LABEL,
            fontweight="normal"
        )
    else:
        ax.set_ylabel("")

    ax.set_xticks([1000, 500, 200, 100, 50, 20])

    ax.get_xaxis().set_major_formatter(
        ScalarFormatter()
    )

    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(True, which="both", alpha=0.12)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Loading revised 16x test data")
    print("=" * 70)

    Xtest_arr, Ytest_arr = load_test_data(DATASET_PATH)

    n = len(Ytest_arr)

    print(f"Xtest: {Xtest_arr.shape}")
    print(f"Target: {Ytest_arr.shape}")

    # --------------------------------------------------------
    # Within-seed U-Net
    # --------------------------------------------------------
    print("\nPreparing within-seed U-Net")

    if REGENERATE_INTRA_UNET or not os.path.exists(INTRA_UNET_CACHE):

        path = os.path.join(
            UNET_DIR,
            f"unet_generator_best_seed{BEST_UNET_SEED}.h5"
        )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing U-Net model: {path}")

        model = tf.keras.models.load_model(path, compile=False)

        intra_unet_preds = generate_tf_samples(
            model=model,
            X=Xtest_arr,
            n_draws=INTRA_DRAWS,
            noise_std=NOISE_STD_INTRA,
            noise_base_seed=NOISE_BASE_SEED_INTRA,
            batch_size=BATCH_SIZE,
        )

        np.save(INTRA_UNET_CACHE, intra_unet_preds)

        del model
        tf.keras.backend.clear_session()

    else:
        intra_unet_preds = standardize_members_4d(
            np.load(INTRA_UNET_CACHE),
            expected_n=n,
            name="intra_unet"
        )

    # --------------------------------------------------------
    # Within-seed WGAN
    # --------------------------------------------------------
    print("\nPreparing within-seed WGAN")

    if REGENERATE_INTRA_WGAN or not os.path.exists(INTRA_WGAN_CACHE):

        path = os.path.join(
            WGAN_DIR,
            f"gen_final_seed{BEST_WGAN_SEED}.keras"
        )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing WGAN model: {path}")

        model = tf.keras.models.load_model(path, compile=False)

        intra_wgan_preds = generate_tf_samples(
            model=model,
            X=Xtest_arr,
            n_draws=INTRA_DRAWS,
            noise_std=NOISE_STD_INTRA,
            noise_base_seed=NOISE_BASE_SEED_INTRA,
            batch_size=BATCH_SIZE,
        )

        np.save(INTRA_WGAN_CACHE, intra_wgan_preds)

        del model
        tf.keras.backend.clear_session()

    else:
        intra_wgan_preds = standardize_members_4d(
            np.load(INTRA_WGAN_CACHE),
            expected_n=n,
            name="intra_wgan"
        )

    # --------------------------------------------------------
    # Within-seed DDPM
    # --------------------------------------------------------
    print("\nPreparing within-seed DDPM")

    intra_ddpm_preds = load_ddpm_within_seed(
        ddpm_dir=DDPM_WITHIN_DIR,
        seed=DDPM_WITHIN_SEED,
        n_draws=INTRA_DRAWS,
        n_samples=n,
    )

    # --------------------------------------------------------
    # Across-seed U-Net
    # --------------------------------------------------------
    print("\nPreparing across-seed U-Net")

    if REGENERATE_INTER_UNET or not os.path.exists(INTER_UNET_CACHE):

        template = os.path.join(
            UNET_DIR,
            "unet_generator_best_seed{seed}.h5"
        )

        inter_unet_preds, _ = generate_one_draw_per_seed_tf(
            seed_list=SEED_LIST,
            model_template=template,
            X=Xtest_arr,
            noise_std=NOISE_STD_INTER,
            noise_seed=NOISE_BASE_SEED_INTER,
            batch_size=BATCH_SIZE,
        )

        np.save(INTER_UNET_CACHE, inter_unet_preds)

    else:
        inter_unet_preds = standardize_members_4d(
            np.load(INTER_UNET_CACHE),
            expected_n=n,
            name="inter_unet"
        )

    # --------------------------------------------------------
    # Across-seed WGAN
    # --------------------------------------------------------
    print("\nPreparing across-seed WGAN")

    if REGENERATE_INTER_WGAN or not os.path.exists(INTER_WGAN_CACHE):

        template = os.path.join(
            WGAN_DIR,
            "gen_final_seed{seed}.keras"
        )

        inter_wgan_preds, _ = generate_one_draw_per_seed_tf(
            seed_list=SEED_LIST,
            model_template=template,
            X=Xtest_arr,
            noise_std=NOISE_STD_INTER,
            noise_seed=NOISE_BASE_SEED_INTER,
            batch_size=BATCH_SIZE,
        )

        np.save(INTER_WGAN_CACHE, inter_wgan_preds)

    else:
        inter_wgan_preds = standardize_members_4d(
            np.load(INTER_WGAN_CACHE),
            expected_n=n,
            name="inter_wgan"
        )

    # --------------------------------------------------------
    # Across-seed DDPM
    # --------------------------------------------------------
    print("\nPreparing across-seed DDPM")

    inter_ddpm_preds = load_ddpm_across_seed(
        ddpm_dir=DDPM_ACROSS_DIR,
        seed_list=SEED_LIST,
        n_samples=n,
    )

    # --------------------------------------------------------
    # Align common sample count
    # --------------------------------------------------------
    n_common = min(
        Ytest_arr.shape[0],
        intra_unet_preds.shape[0],
        intra_wgan_preds.shape[0],
        intra_ddpm_preds.shape[0],
        inter_unet_preds.shape[0],
        inter_wgan_preds.shape[0],
        inter_ddpm_preds.shape[0],
    )

    if n_common < Ytest_arr.shape[0]:
        print(f"Warning: using common N = {n_common}")

    Ytest_arr = Ytest_arr[:n_common]

    intra_unet_preds = intra_unet_preds[:n_common]
    intra_wgan_preds = intra_wgan_preds[:n_common]
    intra_ddpm_preds = intra_ddpm_preds[:n_common]

    inter_unet_preds = inter_unet_preds[:n_common]
    inter_wgan_preds = inter_wgan_preds[:n_common]
    inter_ddpm_preds = inter_ddpm_preds[:n_common]

    # --------------------------------------------------------
    # Dataset containers
    # --------------------------------------------------------
    within_4d = {
        UNET_WITHIN: intra_unet_preds,
        WGAN_WITHIN: intra_wgan_preds,
        DDPM_WITHIN: intra_ddpm_preds,
    }

    across_4d = {
        UNET_ACROSS: inter_unet_preds,
        WGAN_ACROSS: inter_wgan_preds,
        DDPM_ACROSS: inter_ddpm_preds,
    }

    datasets_lag_within = {
        TARGET_LABEL: Ytest_arr,
        UNET_WITHIN: stack_members(intra_unet_preds),
        WGAN_WITHIN: stack_members(intra_wgan_preds),
        DDPM_WITHIN: stack_members(intra_ddpm_preds),
    }

    datasets_lag_across = {
        TARGET_LABEL: Ytest_arr,
        UNET_ACROSS: stack_members(inter_unet_preds),
        WGAN_ACROSS: stack_members(inter_wgan_preds),
        DDPM_ACROSS: stack_members(inter_ddpm_preds),
    }

    # --------------------------------------------------------
    # Compute horizontal lagged autocorrelation
    # --------------------------------------------------------
    print("\nComputing horizontal lagged autocorrelation")

    df_lag_within = pd.concat(
        [
            horizontal_correlation_distribution(
                datasets_lag_within[name],
                MAX_LAG,
                name
            )
            for name in MODEL_ORDER_WITHIN
        ],
        ignore_index=True
    )

    df_lag_across = pd.concat(
        [
            horizontal_correlation_distribution(
                datasets_lag_across[name],
                MAX_LAG,
                name
            )
            for name in MODEL_ORDER_ACROSS
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Compute exceedance probability
    # --------------------------------------------------------
    print("\nComputing exceedance probability")

    thresholds = np.linspace(
        TAIL_THRESHOLD,
        TAIL_XMAX,
        N_THRESHOLDS
    )

    target_exc = target_mean_exceedance(
        Ytest_arr,
        thresholds
    )

    exc_lines_within = {
        name: memberwise_mean_exceedance(arr, thresholds)
        for name, arr in within_4d.items()
    }

    exc_lines_across = {
        name: memberwise_mean_exceedance(arr, thresholds)
        for name, arr in across_4d.items()
    }

    # --------------------------------------------------------
    # Compute spectra
    # --------------------------------------------------------
    print("\nComputing spectra")

    target_spec = target_mean_spectrum(Ytest_arr)

    k_spec = np.arange(
        1,
        target_spec.shape[0] + 1
    )

    wavelength_km = DOMAIN_LENGTH_KM / k_spec

    spec_lines_within = {
        name: memberwise_mean_spectra(arr)
        for name, arr in within_4d.items()
    }

    spec_lines_across = {
        name: memberwise_mean_spectra(arr)
        for name, arr in across_4d.items()
    }

    # --------------------------------------------------------
    # Create composite figure
    # --------------------------------------------------------
    print("\nCreating Figure 9 composite")

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        constrained_layout=False,
    )

    ax_a1, ax_a2 = axes[0]
    ax_b1, ax_b2 = axes[1]
    ax_c1, ax_c2 = axes[2]

    ax_a1.set_title(
        "Within-seed variability",
        fontsize=FONTSIZE_TITLE,
        fontweight="normal",
        pad=12
    )

    ax_a2.set_title(
        "Across-seed variability",
        fontsize=FONTSIZE_TITLE,
        fontweight="normal",
        pad=12
    )

    draw_lag_boxplots(
        ax_a1,
        df_lag_within,
        MODEL_ORDER_WITHIN,
        show_ylabel=True,
    )

    draw_lag_boxplots(
        ax_a2,
        df_lag_across,
        MODEL_ORDER_ACROSS,
        show_ylabel=False,
    )

    plot_exceedance_lines(
        ax_b1,
        thresholds,
        target_exc,
        exc_lines_within,
        show_ylabel=True,
    )

    plot_exceedance_lines(
        ax_b2,
        thresholds,
        target_exc,
        exc_lines_across,
        show_ylabel=False,
    )

    plot_spectrum_lines(
        ax_c1,
        wavelength_km,
        target_spec,
        spec_lines_within,
        show_ylabel=True,
    )

    plot_spectrum_lines(
        ax_c2,
        wavelength_km,
        target_spec,
        spec_lines_across,
        show_ylabel=False,
    )

    panel_labels = {
        ax_a1: "(a1)",
        ax_a2: "(a2)",
        ax_b1: "(b1)",
        ax_b2: "(b2)",
        ax_c1: "(c1)",
        ax_c2: "(c2)",
    }

    for ax, label in panel_labels.items():

        ax.text(
            0.012,
            0.982,
            label,
            transform=ax.transAxes,
            fontsize=FONTSIZE_PANEL,
            fontweight="normal",
            va="top",
            ha="left",
        )

    legend_handles = [
        Line2D(
            [0], [0],
            color=COLORS[TARGET_LABEL],
            lw=LINEWIDTH_TARGET,
            label=TARGET_LABEL
        ),
        Line2D(
            [0], [0],
            color=COLORS["U-Net"],
            lw=2.3,
            label="U-Net"
        ),
        Line2D(
            [0], [0],
            color=COLORS["WGAN"],
            lw=2.3,
            label="WGAN"
        ),
        Line2D(
            [0], [0],
            color=COLORS["DDPM"],
            lw=2.3,
            label="DDPM"
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=FONTSIZE_LEGEND,
        bbox_to_anchor=(0.5, 0.012),
    )

    plt.subplots_adjust(
        left=0.090,
        right=0.985,
        top=0.950,
        bottom=0.095,
        wspace=0.12,
        hspace=0.20,
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
