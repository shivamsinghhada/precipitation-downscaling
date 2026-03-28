"""
Module:      ddpm_model.py
Description: PyTorch model definitions for the conditional Denoising Diffusion
             Probabilistic Model (DDPM) used in both the 8× and 16×
             downscaling experiments.

             This module is shared by both training scripts:
               10_train_ddpm_8x.py   — 16×16 → 128×128 (8×  factor)
               11_train_ddpm_16x.py  — 8×8   → 128×128 (16× factor)

             The only difference between the two experiments is the number of
             upsampling steps in ConditionalUNet.cond_up and the LR input
             shape — both are configured via the `scale_factor` argument to
             build_conditional_unet().

Classes
-------
Diffusion           : cosine noise schedule + forward-process utilities
FiLM                : Feature-wise Linear Modulation for time conditioning
DoubleConv          : two Conv2D + FiLM + SiLU blocks (basic U-Net unit)
ConditionalUNet     : full U-Net with sinusoidal time embeddings and FiLM

Functions
---------
build_conditional_unet(scale_factor) : convenience factory for 8× or 16×

Requirements: torch>=2.0  (see environment.yml)
Reference:    Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
              https://arxiv.org/abs/2006.11239
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

import math
import torch
import torch.nn as nn


# ── Diffusion Schedule ────────────────────────────────────────────────────────

class Diffusion:
    """
    Cosine noise schedule for DDPM (Nichol & Dhariwal, 2021).

    Precomputes alpha_bar, alpha, and beta tensors on the target device.
    The cosine schedule provides a smoother noise ramp than the original
    linear schedule, particularly beneficial for small T.

    Parameters
    ----------
    T      : int,   total diffusion timesteps (e.g. 100)
    device : str or torch.device

    Key attributes
    --------------
    alpha_bar : cumulative product of (1 - beta), shape (T,)
    alpha     : per-step alpha,  shape (T,)
    beta      : per-step beta,   shape (T,)
    """

    def __init__(self, T: int = 100, device: str = "cpu"):
        self.T      = T
        self.device = device

        s = 0.008   # small offset to avoid alpha_bar ≈ 1 at t=0
        steps           = torch.arange(T + 1, dtype=torch.float64) / T
        alphas_cumprod  = torch.cos((steps + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod  = alphas_cumprod / alphas_cumprod[0]
        alphas_cumprod  = torch.clamp(alphas_cumprod, 0.0001, 0.9999)

        self.alpha_bar  = alphas_cumprod[1:].to(device).float()
        alpha_bar_prev  = torch.cat([
            torch.tensor([1.0], device=device), self.alpha_bar[:-1]
        ])
        self.alpha = self.alpha_bar / alpha_bar_prev
        self.beta  = 1.0 - self.alpha

    def sample_timesteps(self, n: int) -> torch.Tensor:
        """Uniformly sample n random timesteps in [0, T)."""
        return torch.randint(0, self.T, (n,), device=self.device)

    def add_noise(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: corrupt x0 at timestep t.

        Parameters
        ----------
        x0 : clean HR field, shape (B, C, H, W)
        t  : integer timestep indices, shape (B,)

        Returns
        -------
        x_t    : noisy sample at timestep t
        noise  : the Gaussian noise that was added (training target)
        """
        sqrt_alpha_bar          = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]
        noise = torch.randn_like(x0)
        x_t   = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise


# ── FiLM Conditioning ─────────────────────────────────────────────────────────

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for time-step conditioning.

    Projects the time embedding to per-channel scale and shift parameters,
    then applies: output = x * (scale + 1) + shift
    The '+1' ensures identity initialisation (scale ≈ 0 at init → no
    distortion before training begins).

    Parameters
    ----------
    time_emb_dim : int, dimensionality of the sinusoidal time embedding
    num_features : int, number of feature-map channels to modulate (C)
    """

    def __init__(self, time_emb_dim: int, num_features: int):
        super().__init__()
        self.proj = nn.Linear(time_emb_dim, num_features * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W    = x.shape
        scale_shift    = self.proj(t_emb)                          # (B, 2C)
        scale, shift   = scale_shift.chunk(2, dim=1)               # each (B, C)
        return x * (scale.view(B, C, 1, 1) + 1) + shift.view(B, C, 1, 1)


# ── Basic U-Net Block ─────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    Two successive Conv2d → FiLM → SiLU blocks.

    Used at every encoder, decoder, and bottleneck level of the U-Net.
    FiLM modulation injects the diffusion timestep information into each
    activation map.

    Parameters
    ----------
    in_ch        : int, number of input channels
    out_ch       : int, number of output channels
    time_emb_dim : int, dimensionality of the time embedding
    """

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.film1 = FiLM(time_emb_dim, out_ch)
        self.act1  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.film2 = FiLM(time_emb_dim, out_ch)
        self.act2  = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.film1(x, t_emb)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.film2(x, t_emb)
        return self.act2(x)


# ── Conditional U-Net ─────────────────────────────────────────────────────────

class ConditionalUNet(nn.Module):
    """
    Conditional U-Net noise predictor for DDPM-based precipitation downscaling.

    The model receives three inputs at every forward pass:
      x    : noisy HR field at timestep t,  shape (B, 1, 128, 128)
      t    : integer timestep indices,       shape (B,)
      cond : LR conditioning field,          shape (B, 1, H_lr, W_lr)

    The LR conditioning field is progressively upsampled to 128×128 via
    `cond_up` (a sequence of bilinear Upsample + Conv layers), then
    concatenated channel-wise with x before the encoder.

    Architecture
    ------------
    Encoder    : four DoubleConv blocks with MaxPool2d downsampling
    Bottleneck : one DoubleConv block (no spatial change)
    Decoder    : four DoubleConv blocks with ConvTranspose2d upsampling
                 and skip connections from the corresponding encoder level
    Output     : 1×1 Conv → clamp to [-3, 3]  (predicted noise)

    Time conditioning is applied at every DoubleConv via FiLM layers.
    The sinusoidal embedding follows Vaswani et al. (2017).

    Parameters
    ----------
    in_ch        : int, channels of noisy HR input (default 1)
    out_ch       : int, channels of predicted noise (default 1)
    cond_ch      : int, channels of LR conditioning field (default 1)
    time_emb_dim : int, sinusoidal embedding dimension (default 256)
    base_ch      : int, base channel width; doubles at each encoder level
                   (default 64 → 64, 128, 256, 512)
    scale_factor : int, spatial upscaling factor for the LR field.
                   Must be a power of 2: 8 (16×16→128) or 16 (8×8→128).
                   Determines the number of Upsample steps in cond_up.
    """

    def __init__(
        self,
        in_ch:        int = 1,
        out_ch:       int = 1,
        cond_ch:      int = 1,
        time_emb_dim: int = 256,
        base_ch:      int = 64,
        scale_factor: int = 8,
    ):
        super().__init__()

        if scale_factor not in (8, 16):
            raise ValueError(
                f"scale_factor must be 8 or 16, got {scale_factor}."
            )

        self.time_emb_dim = time_emb_dim

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # LR conditioning upsampler: each Upsample ×2 step doubles spatial size
        # 8×  experiment: 16×16 → 32 → 64 → 128  (3 steps)
        # 16× experiment:  8×8  → 16 → 32 → 64 → 128  (4 steps)
        n_upsample = int(math.log2(scale_factor))   # 3 for 8×, 4 for 16×
        cond_layers = []
        for i in range(n_upsample):
            in_c  = cond_ch if i == 0 else base_ch // 2
            cond_layers += [
                nn.Conv2d(in_c, base_ch // 2, 3, padding=1),
                nn.SiLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ]
        self.cond_up = nn.Sequential(*cond_layers)

        # Encoder
        self.enc1 = DoubleConv(in_ch + base_ch // 2, base_ch,     time_emb_dim)
        self.enc2 = DoubleConv(base_ch,               base_ch * 2, time_emb_dim)
        self.enc3 = DoubleConv(base_ch * 2,           base_ch * 4, time_emb_dim)
        self.enc4 = DoubleConv(base_ch * 4,           base_ch * 8, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 8, time_emb_dim)

        # Decoder
        self.up4  = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec4 = DoubleConv(base_ch * 8, base_ch * 4, time_emb_dim)
        self.up3  = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec3 = DoubleConv(base_ch * 4, base_ch * 2, time_emb_dim)
        self.up2  = nn.ConvTranspose2d(base_ch * 2, base_ch,     2, stride=2)
        self.dec2 = DoubleConv(base_ch * 2, base_ch,     time_emb_dim)

        # Output head
        self.final = nn.Conv2d(base_ch, out_ch, 1)

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Standard sinusoidal time embedding (Vaswani et al., 2017).

        Parameters
        ----------
        t   : integer timestep tensor, shape (B,)
        dim : embedding dimensionality

        Returns
        -------
        Embedding tensor, shape (B, dim)
        """
        device   = t.device
        half_dim = dim // 2
        emb  = math.log(10000) / (half_dim - 1)
        emb  = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb  = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb  = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        x:    torch.Tensor,
        t:    torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : noisy HR field at timestep t,  (B, 1, 128, 128)
        t    : timestep indices,               (B,)
        cond : LR conditioning field,          (B, 1, H_lr, W_lr)

        Returns
        -------
        Predicted noise field,  (B, 1, 128, 128), clamped to [-3, 3]
        """
        t_emb   = self.sinusoidal_embedding(t, self.time_emb_dim)
        t_emb   = self.time_mlp(t_emb)
        cond_up = self.cond_up(cond)                              # → (B, base_ch//2, 128, 128)

        x  = torch.cat([x, cond_up], dim=1)
        e1 = self.enc1(x,            t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)
        e4 = self.enc4(self.pool(e3), t_emb)

        b  = self.bottleneck(e4, t_emb)

        d4 = self.dec4(torch.cat([self.up4(b),  e3], dim=1), t_emb)
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1), t_emb)

        return torch.clamp(self.final(d2), -3.0, 3.0)


# ── Convenience factory ───────────────────────────────────────────────────────

def build_conditional_unet(scale_factor: int = 8, **kwargs) -> ConditionalUNet:
    """
    Build a ConditionalUNet configured for the requested downscaling factor.

    Parameters
    ----------
    scale_factor : int, 8 for 16×16→128×128, 16 for 8×8→128×128
    **kwargs     : passed directly to ConditionalUNet (e.g. base_ch, time_emb_dim)

    Returns
    -------
    ConditionalUNet instance (not yet moved to a device)
    """
    return ConditionalUNet(scale_factor=scale_factor, **kwargs)
