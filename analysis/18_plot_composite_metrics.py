"""
Script:      analysis/18_plot_composite_metrics.py
Description: Composite metrics figure — Figure 8.

             Four-panel (2×2) figure showing:
               (a) Radial power spectrum (wavelength vs power in dB)
               (b) Fractions Skill Score (FSS) across window sizes
               (c) ROC curve with AUC annotation (mean across seeds)
               (d) SSIM violin plot (one value per trained seed)

             Mean curves are plotted across all 10 seeds for (a), (b), (c).
             Seed-level scatter dots are overlaid on violins in (d).

Inputs:      dataset_splits.npz   (for Xtest, Ytest)
             U-Net and WGAN model checkpoints (all 10 seeds)
             DDPM prediction .npy files for all 10 seeds
Outputs:     composite_metrics.png  in OUTPUT_DIR

Usage:       python analysis/18_plot_composite_metrics.py

Requirements: numpy, tensorflow, matplotlib, scipy, scikit-image, scikit-learn
              (see environment_tf.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────────────
DATASET_PATH   = "/path/to/ERA5_land/Dataset/dataset_splits.npz"

UNET_MODEL_DIR = "/path/to/unet_runs"
UNET_GLOB      = "unet_generator_best_seed*.keras"

WGAN_MODEL_DIR = "/path/to/unet_runs/WGANs"
WGAN_GLOB      = "gen_final_seed*.keras"

DDPM_PRED_DIR  = "/path/to/checkpoints/ddpm_8x"
DDPM_SEEDS     = list(range(1, 11))

OUTPUT_DIR     = "/path/to/results"
DX_KM          = 9.0        # grid spacing in km (ERA5-Land at ~0.1° ≈ 9 km)
RAIN_THR       = 1.0        # mm/day — wet/dry threshold for FSS and ROC
FSS_WINDOWS    = [1, 2, 4, 8, 16, 32]   # pixel window sizes
NSAMP_POWER    = 250        # number of test images for power spectrum (speed)
NOISE_STD      = 1.0
NOISE_SEED     = 42         # fixed noise for reproducible/fair comparison
# ────────────────────────────────────────────────────────────────────────────

import os
import glob
import sys
import numpy as np
import numpy.fft as fft
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.plot_utils import COLORS, squeeze_hw

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13})

LW_TRUTH = 2.5
LW_MEAN  = 2.5


# ── Inference helpers ─────────────────────────────────────────────────────────

def infer_tf_models(model_dir, glob_pattern, X, noise):
    paths = sorted(glob.glob(os.path.join(model_dir, glob_pattern)))
    if not paths:
        raise RuntimeError(f"No models: {model_dir}/{glob_pattern}")
    out = {}
    for p in paths:
        seed  = int(p.split("seed")[-1].split(".")[0])
        model = tf.keras.models.load_model(p, compile=False)
        out[seed] = squeeze_hw(model.predict([X, noise], verbose=0))
        print(f"  Inference: seed {seed}")
    return out


def load_ddpm_preds(pred_dir, seeds):
    out = {}
    for s in seeds:
        path = os.path.join(pred_dir, f"seed_{s}", "best_model_preds.npy")
        if not os.path.exists(path):
            print(f"  WARNING: DDPM not found: {path}")
            continue
        out[s] = squeeze_hw(np.load(path))
    return out


# ── Metric functions ──────────────────────────────────────────────────────────

def radial_power(img: np.ndarray) -> np.ndarray:
    """Mean radial power spectrum of a single 2-D image."""
    F  = fft.fftshift(fft.fft2(img))
    P  = np.abs(F) ** 2
    H, W = P.shape
    cy, cx = H // 2, W // 2
    r  = np.sqrt((np.indices((H, W))[1] - cx) ** 2 +
                 (np.indices((H, W))[0] - cy) ** 2).astype(int)
    rb = np.bincount(r.ravel(), P.ravel())
    ct = np.bincount(r.ravel())
    return rb / (ct + 1e-8)


def to_dB(ps: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.clip(ps, 1e-12, None))


def power_curve_mean(pred_3d: np.ndarray, nsamp: int) -> np.ndarray:
    ns    = min(nsamp, pred_3d.shape[0])
    specs = [radial_power(pred_3d[i]) for i in range(ns)]
    return np.mean(specs, axis=0)


def fss_score(obs: np.ndarray, pred: np.ndarray,
              thr: float = 1.0, window: int = 3) -> float:
    o  = (obs  >= thr).astype(np.float32)
    f  = (pred >= thr).astype(np.float32)
    Po = uniform_filter(o, size=window, mode="constant")
    Pf = uniform_filter(f, size=window, mode="constant")
    num = ((Pf - Po) ** 2).mean()
    den = (Pf ** 2 + Po ** 2).mean() + 1e-8
    return 1.0 - num / den


def safe_ssim(a: np.ndarray, b: np.ndarray) -> float:
    dr = max(a.max(), b.max()) - min(a.min(), b.min())
    return float(ssim(a, b, data_range=max(dr, 1.0)))


# ── Aggregate over seeds ──────────────────────────────────────────────────────

def mean_power(pred_dict: dict, truth: np.ndarray, nsamp: int, nyq: int) -> np.ndarray:
    curves = [power_curve_mean(pred_dict[s], nsamp)[1:nyq + 1]
              for s in sorted(pred_dict)]
    return np.mean(curves, axis=0)


def mean_fss(pred_dict: dict, truth: np.ndarray) -> np.ndarray:
    all_curves = []
    for s in sorted(pred_dict):
        P     = pred_dict[s]
        curve = [np.mean([fss_score(truth[i], P[i], RAIN_THR, w)
                          for i in range(len(truth))]) for w in FSS_WINDOWS]
        all_curves.append(curve)
    return np.mean(all_curves, axis=0)


def mean_roc(pred_dict: dict, truth: np.ndarray):
    y_true  = (truth >= RAIN_THR).ravel().astype(np.uint8)
    fpr_grid = np.linspace(0, 1, 300)
    tprs, aucs = [], []
    for s in sorted(pred_dict):
        fpr, tpr, _ = roc_curve(y_true, pred_dict[s].ravel())
        tpr_i = np.interp(fpr_grid, fpr, tpr); tpr_i[0] = 0.0
        tprs.append(tpr_i); aucs.append(auc(fpr, tpr))
    return (fpr_grid, np.mean(tprs, axis=0),
            float(np.mean(aucs)), float(np.std(aucs)))


def ssim_per_seed(pred_dict: dict, truth: np.ndarray) -> np.ndarray:
    return np.array([
        np.mean([safe_ssim(truth[i], pred_dict[s][i])
                 for i in range(len(truth))])
        for s in sorted(pred_dict)
    ])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading dataset from: {DATASET_PATH}")
    splits = np.load(DATASET_PATH)
    Xtest  = splits["Xtest"][..., np.newaxis].astype(np.float32)
    Ytest  = squeeze_hw(splits["Ytest"])
    N, H, W = Ytest.shape
    NYQ     = H // 2
    L_KM    = H * DX_KM

    rng   = np.random.RandomState(NOISE_SEED)
    noise = rng.normal(0, NOISE_STD, size=Xtest.shape).astype(np.float32)

    print("\nRunning U-Net inference...")
    preds_unet = infer_tf_models(UNET_MODEL_DIR, UNET_GLOB, Xtest, noise)

    print("\nRunning WGAN inference...")
    preds_wgan = infer_tf_models(WGAN_MODEL_DIR, WGAN_GLOB, Xtest, noise)

    print("\nLoading DDPM predictions...")
    preds_ddpm = load_ddpm_preds(DDPM_PRED_DIR, DDPM_SEEDS)

    # Truth power spectrum
    modes    = np.arange(1, NYQ + 1)
    lam      = L_KM / modes
    truth_ps = power_curve_mean(Ytest, NSAMP_POWER)[1:NYQ + 1]

    print("\nComputing metrics (this may take a few minutes)...")
    unet_ps  = mean_power(preds_unet, Ytest, NSAMP_POWER, NYQ)
    wgan_ps  = mean_power(preds_wgan, Ytest, NSAMP_POWER, NYQ)
    ddpm_ps  = mean_power(preds_ddpm, Ytest, NSAMP_POWER, NYQ)

    unet_fss = mean_fss(preds_unet, Ytest)
    wgan_fss = mean_fss(preds_wgan, Ytest)
    ddpm_fss = mean_fss(preds_ddpm, Ytest)

    fpr, unet_tpr, unet_auc_mu, unet_auc_sd = mean_roc(preds_unet, Ytest)
    _,   wgan_tpr, wgan_auc_mu, wgan_auc_sd = mean_roc(preds_wgan, Ytest)
    _,   ddpm_tpr, ddpm_auc_mu, ddpm_auc_sd = mean_roc(preds_ddpm, Ytest)

    unet_ssim = ssim_per_seed(preds_unet, Ytest)
    wgan_ssim = ssim_per_seed(preds_wgan, Ytest)
    ddpm_ssim = ssim_per_seed(preds_ddpm, Ytest)

    # ── Plotting ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axA, axB, axC, axD = axes.ravel()

    # (a) Power spectrum
    axA.plot(lam, to_dB(truth_ps),  color=COLORS["Observed"], lw=LW_TRUTH, label="Observed")
    axA.plot(lam, to_dB(unet_ps),   color=COLORS["U-Net"],    lw=LW_MEAN,  label="U-Net")
    axA.plot(lam, to_dB(wgan_ps),   color=COLORS["WGAN"],     lw=LW_MEAN,  label="WGAN")
    axA.plot(lam, to_dB(ddpm_ps),   color=COLORS["DDPM"],     lw=LW_MEAN,  label="DDPM")
    axA.set_xscale("log"); axA.invert_xaxis()
    axA.set_xlabel("Wavelength (km)"); axA.set_ylabel("Power (dB)")
    axA.grid(False)
    axA.text(0.01, 0.97, "(a)", transform=axA.transAxes,
             fontsize=14, fontweight="bold", va="top")

    # (b) FSS
    axB.plot(FSS_WINDOWS, unet_fss, color=COLORS["U-Net"], lw=LW_MEAN, marker="o")
    axB.plot(FSS_WINDOWS, wgan_fss, color=COLORS["WGAN"],  lw=LW_MEAN, marker="s")
    axB.plot(FSS_WINDOWS, ddpm_fss, color=COLORS["DDPM"],  lw=LW_MEAN, marker="^")
    axB.set_ylim(0, 1.05)
    axB.set_xlabel("Window size (pixels)"); axB.set_ylabel("FSS")
    axB.grid(False)
    axB.text(0.01, 0.97, "(b)", transform=axB.transAxes,
             fontsize=14, fontweight="bold", va="top")

    # (c) ROC
    axC.plot(fpr, unet_tpr, color=COLORS["U-Net"], lw=LW_MEAN)
    axC.plot(fpr, wgan_tpr, color=COLORS["WGAN"],  lw=LW_MEAN)
    axC.plot(fpr, ddpm_tpr, color=COLORS["DDPM"],  lw=LW_MEAN)
    axC.plot([0, 1], [0, 1], "k--", lw=1)
    axC.set_xlabel("False positive rate"); axC.set_ylabel("True positive rate")
    axC.grid(False)
    axC.text(0.01, 0.97, "(c)", transform=axC.transAxes,
             fontsize=14, fontweight="bold", va="top")
    axC.text(0.55, 0.05,
             f"U-Net: AUC = {unet_auc_mu:.3f} ± {unet_auc_sd:.3f}\n"
             f"WGAN:  AUC = {wgan_auc_mu:.3f} ± {wgan_auc_sd:.3f}\n"
             f"DDPM:  AUC = {ddpm_auc_mu:.3f} ± {ddpm_auc_sd:.3f}",
             transform=axC.transAxes, fontsize=10, va="bottom",
             bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9))

    # (d) SSIM violin
    data = [unet_ssim, wgan_ssim, ddpm_ssim]
    vp   = axD.violinplot(data, showmeans=True, showextrema=False)
    for body, col in zip(vp["bodies"],
                         [COLORS["U-Net"], COLORS["WGAN"], COLORS["DDPM"]]):
        body.set_facecolor(col); body.set_edgecolor("black"); body.set_alpha(0.55)
    if "cmeans" in vp:
        vp["cmeans"].set_color("black"); vp["cmeans"].set_linewidth(1.6)
    for i, vals in enumerate(data, start=1):
        axD.scatter(np.full_like(vals, i, dtype=float), vals,
                    s=22, alpha=0.8, color="black", zorder=5)
    axD.set_xticks([1, 2, 3])
    axD.set_xticklabels(["U-Net", "WGAN", "DDPM"])
    axD.set_ylabel("SSIM"); axD.grid(False)
    axD.text(0.01, 0.97, "(d)", transform=axD.transAxes,
             fontsize=14, fontweight="bold", va="top")
    pad = 0.005
    all_ssim = np.concatenate(data)
    axD.set_ylim(all_ssim.min() - pad, all_ssim.max() + pad)

    # Shared legend
    handles = [
        plt.Line2D([0],[0], color=COLORS["Observed"], lw=LW_TRUTH, label="Observed"),
        plt.Line2D([0],[0], color=COLORS["U-Net"],    lw=LW_MEAN,  label="U-Net"),
        plt.Line2D([0],[0], color=COLORS["WGAN"],     lw=LW_MEAN,  label="WGAN"),
        plt.Line2D([0],[0], color=COLORS["DDPM"],     lw=LW_MEAN,  label="DDPM"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(OUTPUT_DIR, "composite_metrics.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
