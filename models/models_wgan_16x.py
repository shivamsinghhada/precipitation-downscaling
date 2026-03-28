"""
Module:      models_16x.py
Description: Neural network architectures for the 16× statistical downscaling
             experiment (8×8 → 128×128 precipitation).

             Differences from models.py (8× experiment):
               - LR input is 8×8 instead of 16×16.
               - Both the U-Net generator and the conditional critic apply a
                 bilinear/nearest-neighbor resize to bring 8×8 feature maps
                 to 16×16 before the encoder and merge stages respectively.
                 This is necessary because the encoder requires at least two
                 2×2 max-pool operations (8→4→2), and the critic merges HR
                 and LR branches at 16×16.
               - The WGAN class is identical in logic to models.py but is
                 included here for self-containment and clarity.

             Functions
             ----------
               build_unet_generator_16x()    : stochastic U-Net (8×8 → 128×128)
               build_conditional_critic_16x(): conditional WGAN-GP critic
               WGAN16x                       : WGAN-GP training loop

             Both training scripts (08_train_unet_16x.py, 09_train_wgan_16x.py)
             import directly from this module.

Requirements: tensorflow>=2.12  (see environment.yml)
Author:       [Your Name]
Date:         [YYYY-MM-DD]
"""

import tensorflow as tf
from tensorflow.keras import layers, models


# ── U-Net Generator (16×) ────────────────────────────────────────────────────

def build_unet_generator_16x(lr_shape=(8, 8, 1), noise_shape=(8, 8, 1)):
    """
    Stochastic U-Net that upscales 8×8 low-resolution (LR) precipitation
    to 128×128 high-resolution (HR) fields (16× spatial factor).

    Because the encoder requires two 2×2 max-pool steps (→ 4×4 → 2×2 bottleneck
    if applied at 8×8), both the LR input and the noise field are first resized
    to 16×16 before concatenation:
      - LR input  : bilinear resize  (preserves spatial gradients)
      - Noise     : nearest-neighbor resize  (preserves noise texture)
    This keeps the encoder/bottleneck/decoder structure identical to the 8×
    generator while accommodating the smaller LR footprint.

    Architecture
    ------------
    Pre-processing : LR  8×8  → bilinear   → 16×16
                     Noise 8×8 → nearest    → 16×16
    Concatenate    : (16,16,2)
    Encoder        : 16×16 → 8×8 → 4×4  (two MaxPooling steps)
    Bottleneck     : 4×4
    Decoder        : 4×4 → 8×8 → 16×16  (skip connections from encoder)
    Progressive upsampling: 16×16 → 32×32 → 64×64 → 128×128

    Parameters
    ----------
    lr_shape    : tuple, spatial shape of raw LR input  (H, W, C) = (8, 8, 1)
    noise_shape : tuple, spatial shape of noise input   (H, W, C) = (8, 8, 1)

    Returns
    -------
    tf.keras.Model with inputs [lr_input, noise_input] and output shape
    (batch, 128, 128, 1).
    """
    lr_input    = layers.Input(shape=lr_shape,    name="lr_input")
    noise_input = layers.Input(shape=noise_shape, name="noise_input")

    # Resize both inputs to 16×16 before encoding
    # Bilinear for LR (smooth spatial gradients); nearest for noise (preserve texture)
    up_lr    = layers.Resizing(16, 16, interpolation="bilinear",  name="lr_upsample")(lr_input)
    up_noise = layers.Resizing(16, 16, interpolation="nearest",   name="noise_upsample")(noise_input)

    x_in = layers.Concatenate(axis=-1)([up_lr, up_noise])         # (16,16,2)

    # --- Encoder ---
    c1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x_in)
    c1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2), padding="same")(c1)           # 8×8

    c2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2), padding="same")(c2)           # 4×4

    # --- Bottleneck ---
    bn = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p2)
    bn = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(bn)

    # --- Decoder (with skip connections) ---
    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(bn)   # 8×8
    u1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u1)
    c3 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c3)

    u2 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c3)   # 16×16
    u2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(u2)
    c4 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c4)

    # --- Progressive upsampling: 16×16 → 32×32 → 64×64 → 128×128 ---
    u3 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c4)   # 32×32
    u3 = layers.LeakyReLU(negative_slope=0.2)(u3)

    u4 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(u3)   # 64×64
    u4 = layers.LeakyReLU(negative_slope=0.2)(u4)

    u5 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(u4)   # 128×128
    u5 = layers.LeakyReLU(negative_slope=0.2)(u5)

    # Final refinement convolutions at full resolution
    uc1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(u5)
    uc2 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(uc1)

    output = layers.Conv2D(1, (1, 1), activation="linear", padding="same")(uc2)

    return models.Model([lr_input, noise_input], output, name="UNetGenerator_16x")


# ── Conditional Critic (16×) ──────────────────────────────────────────────────

def build_conditional_critic_16x(hr_shape=(128, 128, 1), lr_shape=(8, 8, 1)):
    """
    Conditional critic for WGAN-GP (16× experiment).

    The HR branch is downsampled via strided convolutions to 16×16.
    The LR branch (8×8) is bilinearly upsampled to 16×16 to match, then
    processed by three Conv2D layers before merging with the HR branch.

    Architecture
    ------------
    HR branch : 128×128 → 64×64 → 32×32 → 16×16  (strided Conv2D)
    LR branch : 8×8 → bilinear resize → 16×16 → three Conv2D layers
    Merge     : concatenate at 16×16 → Conv2D → GlobalAveragePooling → Dense(1)

    Parameters
    ----------
    hr_shape : tuple  (128, 128, 1)
    lr_shape : tuple  (8,   8,   1)

    Returns
    -------
    tf.keras.Model with inputs [hr_input, lr_input] and scalar output.
    """
    # --- HR branch ---
    hr_input = layers.Input(shape=hr_shape, name="hr_input")
    x_hr = layers.Conv2D(32,  (4, 4), strides=(2, 2), padding="same")(hr_input)  # 64×64
    x_hr = layers.LeakyReLU(negative_slope=0.2)(x_hr)
    x_hr = layers.Conv2D(64,  (4, 4), strides=(2, 2), padding="same")(x_hr)      # 32×32
    x_hr = layers.LeakyReLU(negative_slope=0.2)(x_hr)
    x_hr = layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same")(x_hr)      # 16×16
    x_hr = layers.LeakyReLU(negative_slope=0.2)(x_hr)

    # --- LR branch: upsample 8×8 → 16×16 to match HR feature map size ---
    lr_input = layers.Input(shape=lr_shape, name="lr_input")
    x_lr     = layers.Resizing(16, 16, interpolation="bilinear", name="lr_upsample")(lr_input)
    x_lr = layers.Conv2D(32,  (3, 3), padding="same")(x_lr)
    x_lr = layers.LeakyReLU(negative_slope=0.2)(x_lr)
    x_lr = layers.Conv2D(64,  (3, 3), padding="same")(x_lr)
    x_lr = layers.LeakyReLU(negative_slope=0.2)(x_lr)
    x_lr = layers.Conv2D(128, (3, 3), padding="same")(x_lr)
    x_lr = layers.LeakyReLU(negative_slope=0.2)(x_lr)

    # --- Merge at 16×16 ---
    x = layers.Concatenate()([x_hr, x_lr])                        # 16×16, 256 ch
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Dense(1)(x)                                    # Wasserstein score

    return models.Model([hr_input, lr_input], output, name="ConditionalCritic_16x")


# ── WGAN-GP Model (16×) ───────────────────────────────────────────────────────

class WGAN16x(tf.keras.Model):
    """
    Wasserstein GAN with Gradient Penalty (WGAN-GP) for the 16× downscaling
    experiment (8×8 → 128×128).

    Identical in logic to the WGAN class in models.py, but operates on
    8×8 LR inputs.  Noise is sampled at the same shape as X_lr (8×8)
    and the generator's internal Resizing layers handle the 8→16 step.

    Parameters
    ----------
    generator  : tf.keras.Model  (16× U-Net generator)
    critic     : tf.keras.Model  (16× conditional critic)
    gp_weight  : float, gradient penalty coefficient (default 10.0)
    d_steps    : int,   number of critic updates per generator update (default 3)

    Tracked metrics
    ---------------
    train_step : d_loss, g_loss, gp, mse
    test_step  : val_mse
    """

    def __init__(self, generator, critic, gp_weight=10.0, d_steps=3):
        super().__init__()
        self.generator  = generator
        self.critic     = critic
        self.gp_weight  = gp_weight
        self.d_steps    = d_steps
        # Define metric in __init__ so it is always registered before compile()
        self.mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")

    def compile(self, g_optimizer, d_optimizer, g_loss_fn, d_loss_fn):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss_fn   = g_loss_fn
        self.d_loss_fn   = d_loss_fn

    @property
    def metrics(self):
        # Returning metrics here ensures Keras resets them each epoch.
        return [self.mse_metric]

    def gradient_penalty(self, real_hr, fake_hr, cond_lr):
        """Gradient penalty on interpolated HR samples (Gulrajani et al., 2017)."""
        batch_size   = tf.shape(real_hr)[0]
        alpha        = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = real_hr + alpha * (fake_hr - real_hr)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic([interpolated, cond_lr], training=True)

        grads = gp_tape.gradient(pred, interpolated)
        norm  = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-12)
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, data):
        X_lr, Y_hr = data

        # === Train critic (d_steps times per generator update) ===
        for _ in range(self.d_steps):
            # Noise sampled at LR shape (8×8); generator resizes internally
            noise = tf.random.normal(tf.shape(X_lr))
            with tf.GradientTape() as d_tape:
                fake_hr      = self.generator([X_lr, noise], training=True)
                real_logits  = self.critic([Y_hr,    X_lr],  training=True)
                fake_logits  = self.critic([fake_hr, X_lr],  training=True)
                d_loss       = self.d_loss_fn(real_logits, fake_logits)
                gp           = self.gradient_penalty(Y_hr, fake_hr, X_lr)
                d_loss_total = d_loss + self.gp_weight * gp

            d_grads = d_tape.gradient(d_loss_total, self.critic.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_grads, self.critic.trainable_variables)
            )

        # === Train generator (one step) ===
        noise = tf.random.normal(tf.shape(X_lr))
        with tf.GradientTape() as g_tape:
            fake_hr     = self.generator([X_lr, noise], training=True)
            fake_logits = self.critic([fake_hr, X_lr],  training=True)
            g_loss      = self.g_loss_fn(fake_logits)

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables)
        )

        self.mse_metric.update_state(Y_hr, fake_hr)
        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "gp":     gp,
            "mse":    self.mse_metric.result(),
        }

    def test_step(self, data):
        """
        Validation step.
        Returns 'val_mse' — this key is monitored by ModelCheckpoint in
        09_train_wgan_16x.py.  Must match the monitor= argument exactly.
        """
        X_lr, Y_hr = data
        noise  = tf.random.normal(tf.shape(X_lr))
        preds  = self.generator([X_lr, noise], training=False)
        self.mse_metric.update_state(Y_hr, preds)
        return {"val_mse": self.mse_metric.result()}    # ← key must be val_mse
