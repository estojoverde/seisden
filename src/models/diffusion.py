# src/models/diffusion.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, Sequence

import math
import torch
from torch import nn

from ..utils import PML_kw
from ..features.fourier import PML_build_fourier_feature_stack
from .unet_blocks import PML_UNet


# =========================
# Utilities (spectral masks)
# =========================

def _pml_rfftfreq(n_H: int, f_dt: float, device, dtype) -> torch.Tensor:
    """
    Return positive frequency axis for rFFT along time dimension (length n_H).

    Args:
        n_H: int, number of time samples (rows)
        f_dt: float, sample interval [s]
    Returns:
        (n_H//2 + 1,) frequencies in Hz, dtype/device matched.
    """
    return torch.fft.rfftfreq(n_H, d=f_dt, device=device, dtype=dtype)


def _pml_highpass_raised_cosine(
    f_axis: torch.Tensor,
    f_cut: torch.Tensor,
    f_trans: float,
) -> torch.Tensor:
    """
    Per-frequency high-pass gain using a raised-cosine transition.

    A(f) =
      0                                        if f < f_cut
      0.5 - 0.5*cos(pi * (f-f_cut)/f_trans)    if f ∈ [f_cut, f_cut + f_trans]
      1                                        if f > f_cut + f_trans

    Shapes:
      f_axis:  (F,)
      f_cut:   (B,1,1) or scalar → broadcast
    Returns:
      A: (B,1,F) in [0,1]
    """
    # Ensure z has batch shape (B,1,F)
    F = f_axis.shape[0]
    # broadcast f_axis to (1,1,F) then to (B,1,F)
    z = (f_axis.view(1, 1, F) - f_cut) / max(f_trans, 1e-12)  # (B,1,F)

    A_mid = 0.5 - 0.5 * torch.cos(math.pi * z)
    A = torch.where(
        z <= 0.0, torch.zeros_like(z),
        torch.where(z >= 1.0, torch.ones_like(z), A_mid)
    )
    return A  # (B,1,F)



def _pml_expand_A_to_spec(A_1d: torch.Tensor, n_W: int) -> torch.Tensor:
    """
    Expand (F,) -> (F, n_W) for broadcasting over offset (W).
    """
    return A_1d.view(-1, 1).expand(-1, n_W)


def _pml_fft_time(x: torch.Tensor) -> torch.Tensor:
    """
    rFFT along time (dim=-2). Input (B,1,H,W) -> (B,1,F,W) complex.
    """
    return torch.fft.rfft(x, n=x.shape[-2], dim=-2, norm="ortho")


def _pml_ifft_time(X: torch.Tensor, n_H: int) -> torch.Tensor:
    """
    irFFT along time back to length n_H. Input (B,1,F,W) complex -> (B,1,H,W) real.
    """
    return torch.fft.irfft(X, n=n_H, dim=-2, norm="ortho")


def _pml_make_time_window(n_H: int, **kwargs) -> torch.Tensor:
    """
    Build a 1D time window (H,) then reshape to (1,1,H,1) for broadcasting.

    Optional **kwargs:
        c_type: "tukey" | "hann" = "tukey"
        f_alpha: float (Tukey shape) = 0.25
        f_mix: float in [0,1] (applied later) = 1.0  # not used here
        device, dtype
    """
    c_type = PML_kw("c_type", kwargs, "tukey").lower()
    f_alpha = float(PML_kw("f_alpha", kwargs, 0.25))
    device = PML_kw("device", kwargs, "cpu")
    dtype = PML_kw("dtype", kwargs, torch.float32)

    t = torch.linspace(0.0, 1.0, n_H, device=device, dtype=dtype)
    if c_type == "hann":
        w = 0.5 - 0.5 * torch.cos(2.0 * math.pi * t)
    else:
        # Tukey: 0..alpha/2 rise, flat, alpha/2..1 fall (here we just build a raised window)
        f_a = max(min(f_alpha, 1.0), 1e-8)
        w = torch.ones_like(t)
        # first segment
        idx1 = t < (f_a / 2.0)
        w[idx1] = 0.5 * (1 - torch.cos(2.0 * math.pi * t[idx1] / f_a))
        # last segment
        idx3 = t > (1 - f_a / 2.0)
        w[idx3] = 0.5 * (1 - torch.cos(2.0 * math.pi * (1 - t[idx3]) / f_a))
    return w.view(1, 1, n_H, 1)


# ===================================
# Configs (fully parameterized, no magic)
# ===================================

@dataclass
class PML_SpectralDiffusionConfig:
    """
    Configuration for Spectral Diffusion (LF-loss corruption).

    n_T (int): number of diffusion timesteps (training grid).
    c_schedule (str): mapping t -> cutoff frequency. "linear" | "cosine".
    f_lf_max (float): maximum LF cutoff at t = n_T-1 [Hz]. Controls how much LF can be removed.
    f_transition (float): transition bandwidth for the high-pass raised-cosine [Hz].
    b_band_noise (bool): if True, inject band-limited noise in LF via B_t(f) = sqrt(1 - A_t(f)^2).
    f_noise_cap (float): cap for B_t(f) to avoid instability; 0.0 -> deterministic.
    b_v_prediction (bool): if True, use v-prediction; else epsilon-prediction (default).
    n_ddim_steps (int): number of steps during DDIM sampling (reverse).
    b_clip_denoised (bool): clamp denoised x0 estimate into [-1,1] (safety).
    b_predict_residual (bool): if True, model predicts LF residual in time domain.
    dic_window_blend (dict): {"c_type": "tukey"|"hann", "f_alpha": float, "f_mix": float in [0,1]}.
    c_axis (str): "time" (default) or "radial" (future). Currently only "time" implemented.
    b_concat_t (bool): concatenate a constant channel with normalized timestep to the model input.
    b_append_noisy_residual (bool): append x_t (noisy residual) as an extra channel for the denoiser.
    n_in_feat (int): number of channels produced by feature stack (default 6).
    """
    n_T: int = 1000
    c_schedule: str = "linear"
    f_lf_max: float = 12.0
    f_transition: float = 2.0
    b_band_noise: bool = True
    f_noise_cap: float = 0.1
    b_v_prediction: bool = False
    n_ddim_steps: int = 50
    b_clip_denoised: bool = True
    b_predict_residual: bool = True
    dic_window_blend: Dict[str, Any] = None
    c_axis: str = "time"
    b_concat_t: bool = True
    b_append_noisy_residual: bool = True
    n_in_feat: int = 6  # from PML_build_fourier_feature_stack default

    def __post_init__(self):
        if self.dic_window_blend is None:
            self.dic_window_blend = {"c_type": "tukey", "f_alpha": 0.25, "f_mix": 1.0}


@dataclass
class PML_VanillaDiffusionConfig:
    """
    Standard DDPM config (kept as a tool).
    """
    n_T: int = 1000
    c_beta_schedule: str = "linear"  # "linear"|"cosine"
    b_v_prediction: bool = False
    n_ddim_steps: int = 50
    b_clip_denoised: bool = True


# =========================
# Denoiser (UNet wrapper)
# =========================

class PML_SpectralUNetDenoiser(nn.Module):
    """
    Wrapper to construct a U-Net denoiser with dynamic input channels.

    Args:
        n_in_ch: int, input channels
        n_out_ch: int, output channels (1 for epsilon in time-domain residual)
        dic_unet_kwargs: forwarded to PML_UNet (widths, blocks, etc.)
    """
    def __init__(self, n_in_ch: int, n_out_ch: int = 1, dic_unet_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        if dic_unet_kwargs is None:
            dic_unet_kwargs = {}
        self.net = PML_UNet(n_in_ch, n_out_ch, **dic_unet_kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net(x, **kwargs)


# =========================
# Spectral DDPM
# =========================

class PML_SpectralDDPM(nn.Module):
    """
    Diffusion where corruption corresponds to **progressive loss of low frequencies** along time.

    Training:
      - Target space x0 is the **LF residual**: residual = y_fullband - x_lowcut.
      - At timestep t, we form x_t by applying a high-pass A_t(f) in frequency (time axis)
        and optionally adding band-limited noise via B_t(f) in the LF band only.
      - The denoiser predicts epsilon in time domain (colored, due to B_t).

    Sampling:
      - DDIM (eta=0) by default: invert using predicted epsilon to estimate X0_hat,
        then reapply A_{t-1}, B_{t-1} in frequency to get x_{t-1}.

    Public methods:
      - forward(...) -> training loss
      - predict_residual(x_lowcut, f_dt, **kwargs)
      - reconstruct_fullband(x_lowcut, f_dt, **kwargs)
    """
    def __init__(
        self,
        cfg: PML_SpectralDiffusionConfig,
        dic_unet_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.dic_unet_kwargs = {} if dic_unet_kwargs is None else dict(dic_unet_kwargs)
        # Input channels = features + optional x_t + optional t
        n_in_ch = cfg.n_in_feat + (1 if cfg.b_append_noisy_residual else 0) + (1 if cfg.b_concat_t else 0)
        self.denoiser = PML_SpectralUNetDenoiser(n_in_ch=n_in_ch, n_out_ch=1, dic_unet_kwargs=self.dic_unet_kwargs)

    # ---------- schedules and operators ----------

    def _cutoff_at_t(self, t_int: torch.Tensor, f_lf_max: float) -> torch.Tensor:
        """
        Map integer t ∈ [0, n_T-1] -> cutoff frequency f_c(t) [Hz].
        Supports "linear" and "cosine" schedules.
        Returns shape (B,1,1) broadcastable over (F,W).
        """
        # normalize t in [0,1]
        tau = t_int.to(torch.float32) / max(self.cfg.n_T - 1, 1)
        if self.cfg.c_schedule.lower() == "cosine":
            # start slow, end fast
            tau = torch.sin(0.5 * math.pi * tau)  # [0,1]
        f_c = tau * f_lf_max
        return f_c.view(-1, 1, 1)

    def _compute_AB(self, n_H: int, n_W: int, f_dt: float, t_int: torch.Tensor, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build A_t(f), B_t(f) grids for batch, expanded to (B,1,F,W).
        """
        f_axis = _pml_rfftfreq(n_H, f_dt, device, dtype)            # (F,)
        f_c = self._cutoff_at_t(t_int, self.cfg.f_lf_max)            # (B,1,1)
        A_1d = _pml_highpass_raised_cosine(f_axis, f_c, self.cfg.f_transition)  # (B,1,F)
        A = A_1d.unsqueeze(-1).expand(-1, -1, -1, n_W)               # (B,1,F,W)

        if self.cfg.b_band_noise:
            B = torch.sqrt(torch.clamp(1.0 - A * A, min=0.0))
            if self.cfg.f_noise_cap < 1.0:
                B = torch.clamp(B, max=self.cfg.f_noise_cap)
        else:
            B = torch.zeros_like(A)
        return A, B

    # ---------- q(x_t | x0) and targets ----------

    def _q_sample(
        self,
        residual_true: torch.Tensor,
        t_int: torch.Tensor,
        f_dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply frequency high-pass A_t and band-noise B_t (in LF) to residual_true.

        Args:
            residual_true: (B,1,H,W)
            t_int: (B,) integer timesteps
            f_dt: float sample interval [s]

        Returns:
            x_t: (B,1,H,W) time-domain noisy residual
            eps_time: (B,1,H,W) colored noise in time domain (what we train to predict)
        """
        Bsz, _, n_H, n_W = residual_true.shape
        device, dtype = residual_true.device, residual_true.dtype

        A, B = self._compute_AB(n_H, n_W, f_dt, t_int, device, dtype)  # (1,1,F,W)
        X0 = _pml_fft_time(residual_true)  # (B,1,F,W) complex
        # complex Gaussian noise Zc ~ N(0, I) in complex domain, unit variance per dim
        Zc_real = torch.randn_like(X0.real)
        Zc_imag = torch.randn_like(X0.imag)
        Zc = torch.complex(Zc_real, Zc_imag)

        X_t = A * X0 + B * Zc  # broadcast over batch
        x_t = _pml_ifft_time(X_t, n_H)

        # time-domain noise equals irfft(B * Zc)
        eps_time = _pml_ifft_time(B * Zc, n_H)
        return x_t, eps_time

    # ---------- training ----------

    def forward(
        self,
        x_lowcut: torch.Tensor,
        y_fullband: torch.Tensor,
        f_dt: float,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute training loss (MSE on epsilon) for Spectral Diffusion.

        Args:
            x_lowcut:  (B,1,H,W)
            y_fullband:(B,1,H,W)
            f_dt: float sample interval [s]
        Optional **kwargs:
            dic_features: dict forwarded to PML_build_fourier_feature_stack for conditioning.
            b_return_debug: bool (False) if True, also return aux dict.

        Returns:
            loss (Tensor scalar) or (loss, dic_debug) if b_return_debug=True
        """
        b_return_debug = bool(PML_kw("b_return_debug", kwargs, False))
        dic_features = PML_kw("dic_features", kwargs, {}) or {}

        residual_true = y_fullband - x_lowcut  # (B,1,H,W)

        n_B = residual_true.shape[0]
        t_int = torch.randint(low=0, high=self.cfg.n_T, size=(n_B,), device=residual_true.device)

        x_t, eps_target = self._q_sample(residual_true, t_int, f_dt)

        # Build conditioning features from low-cut input
        feat = PML_build_fourier_feature_stack(
            x_lowcut, f_dt,
            **dic_features
        )  # (B, n_in_feat, H, W) default 6

        l_inputs = [feat]
        if self.cfg.b_append_noisy_residual:
            l_inputs.append(x_t)  # +1 channel
        if self.cfg.b_concat_t:
            # normalized t in [0,1], broadcast to HxW
            t_norm = (t_int.to(feat.dtype) / max(self.cfg.n_T - 1, 1)).view(-1, 1, 1, 1).expand(-1, 1, feat.shape[-2], feat.shape[-1])
            l_inputs.append(t_norm)

        den_in = torch.cat(l_inputs, dim=1)  # (B, n_in, H, W)

        # Predict colored time-domain noise
        eps_pred = self.denoiser(den_in)  # (B,1,H,W)

        loss = torch.mean((eps_pred - eps_target) ** 2)

        if not b_return_debug:
            return loss

        with torch.no_grad():
            dic_debug = {
                "t_int": t_int,
                "eps_target_std": eps_target.std().detach().cpu(),
                "eps_pred_std": eps_pred.std().detach().cpu(),
            }
        return loss, dic_debug

    # ---------- sampling ----------

    @torch.no_grad()
    def predict_residual(
        self,
        x_lowcut: torch.Tensor,
        f_dt: float,
        **kwargs,
    ) -> torch.Tensor:
        """
        DDIM deterministic sampling to estimate residual from x_T (heavily LF-removed) to x_0.

        Args:
            x_lowcut: (B,1,H,W)
            f_dt: float

        Optional **kwargs:
            dic_features: kwargs to PML_build_fourier_feature_stack
            n_steps: int override for DDIM steps
        """
        device, dtype = x_lowcut.device, x_lowcut.dtype
        n_steps = int(PML_kw("n_steps", kwargs, self.cfg.n_ddim_steps))
        dic_features = PML_kw("dic_features", kwargs, {}) or {}

        Bsz, _, n_H, n_W = x_lowcut.shape

        # Initialize x_t by applying the *max* corruption to a dummy residual (zeros)
        # We will run DDIM from t=T-1 -> 0 starting from zeros (or could sample).
        x_t = torch.zeros(Bsz, 1, n_H, n_W, device=device, dtype=dtype)

        # Choose a DDIM time grid
        ts = torch.linspace(self.cfg.n_T - 1, 0, n_steps, device=device)
        ts = ts.round().to(torch.long)

        for i, t in enumerate(ts):
            t_batch = torch.full((Bsz,), t, device=device, dtype=torch.long)

            # Build conditioning
            feat = PML_build_fourier_feature_stack(x_lowcut, f_dt, **dic_features)
            l_inputs = [feat]
            if self.cfg.b_append_noisy_residual:
                l_inputs.append(x_t)
            if self.cfg.b_concat_t:
                t_norm = (t_batch.to(dtype) / max(self.cfg.n_T - 1, 1)).view(-1, 1, 1, 1).expand(-1, 1, n_H, n_W)
                l_inputs.append(t_norm)
            den_in = torch.cat(l_inputs, dim=1)

            eps_pred = self.denoiser(den_in)  # (B,1,H,W)

            # In frequency domain: X_t = A_t X0 + B_t Z  ->  estimate X0
            A_t, B_t = self._compute_AB(n_H, n_W, f_dt, t_batch, device, dtype)
            X_t = _pml_fft_time(x_t)
            N_pred_f = _pml_fft_time(eps_pred)  # equals B_t * Z_hat ideally

            # Estimate X0: guard division by A_t (eps to avoid explode)
            A_eps = torch.clamp(A_t, min=1e-6)
            X0_hat = (X_t - N_pred_f) / A_eps

            # Deterministic DDIM: set noise term at t-1 to zero (eta=0)
            t_prev = ts[i + 1] if i + 1 < len(ts) else torch.tensor(0, device=device, dtype=torch.long)
            A_prev, B_prev = self._compute_AB(n_H, n_W, f_dt, t_prev.view(1), device, dtype)

            X_prev = A_prev * X0_hat  # + 0 * (B_prev * Z)
            x_t = _pml_ifft_time(X_prev, n_H)

            if self.cfg.b_clip_denoised:
                x_t = x_t.clamp_(-1.0, 1.0)

        # Final estimate is x_0
        residual_hat = x_t
        return residual_hat

    @torch.no_grad()
    def reconstruct_fullband(self, x_lowcut: torch.Tensor, f_dt: float, **kwargs) -> torch.Tensor:
        """
        Predict residual and blend back into low-cut input via a time-domain window.

        Optional **kwargs:
            dic_features: forwarded to PML_build_fourier_feature_stack
            dic_window_blend: override (c_type, f_alpha, f_mix)
        """
        residual = self.predict_residual(x_lowcut, f_dt, **kwargs)
        dic_window_blend = PML_kw("dic_window_blend", kwargs, self.cfg.dic_window_blend) or self.cfg.dic_window_blend
        f_mix = float(PML_kw("f_mix", dic_window_blend, 1.0))
        w = _pml_make_time_window(x_lowcut.shape[-2], **dic_window_blend).to(device=x_lowcut.device, dtype=x_lowcut.dtype)
        return x_lowcut + f_mix * w * residual


# =========================
# Vanilla DDPM (kept for future use)
# =========================

def _pml_betas_linear(n_T: int, f_beta_start: float = 1e-4, f_beta_end: float = 2e-2, device="cpu", dtype=torch.float32):
    """
    Linear beta schedule; params are explicit.
    """
    return torch.linspace(f_beta_start, f_beta_end, n_T, device=device, dtype=dtype)


class PML_DDPM(nn.Module):
    """
    Vanilla DDPM (time-domain Gaussian noise). Kept as a tool for future baselines.
    """
    def __init__(self, cfg: PML_VanillaDiffusionConfig, dic_unet_kwargs: Optional[Dict[str, Any]] = None, n_in_ch: int = 6):
        super().__init__()
        self.cfg = cfg
        self.dic_unet_kwargs = {} if dic_unet_kwargs is None else dict(dic_unet_kwargs)

        device = torch.device("cpu")
        betas = _pml_betas_linear(cfg.n_T, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # model: input = features (+ maybe extra channels if you wish to add)
        self.denoiser = PML_UNet(n_in_ch, 1, **(self.dic_unet_kwargs))

    def forward(self, x_lowcut: torch.Tensor, y_fullband: torch.Tensor, f_dt: float, **kwargs) -> torch.Tensor:
        """
        Baseline DDPM objective: predict epsilon for time-domain Gaussian noise.
        We still aim at residual = y_fullband - x_lowcut.
        """
        residual_true = y_fullband - x_lowcut
        n_B = residual_true.shape[0]
        device = residual_true.device
        t = torch.randint(0, self.cfg.n_T, (n_B,), device=device)

        # standard q(x_t|x0)
        a_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        eps = torch.randn_like(residual_true)
        x_t = torch.sqrt(a_bar) * residual_true + torch.sqrt(1.0 - a_bar) * eps

        # features from low-cut
        feat = PML_build_fourier_feature_stack(x_lowcut, f_dt)
        # optionally append x_t and/or t; kept minimal here
        den_in = torch.cat([feat, x_t], dim=1)  # expect n_in_ch matched at construction

        eps_pred = self.denoiser(den_in)
        return torch.mean((eps_pred - eps) ** 2)



# --- append at end of src/models/diffusion.py ---

# Ensure prefixed aliases exist for training/integration code
try:
    PML_SpectralDDPM
except NameError:
    try:
        PML_SpectralDDPM = SpectralDDPM  # type: ignore[name-defined]
    except NameError:
        pass

try:
    PML_DiffusionConfig
except NameError:
    try:
        PML_DiffusionConfig = DiffusionConfig  # type: ignore[name-defined]
    except NameError:
        pass

# (Optional) If you also have a vanilla DDPM/Config, alias those too:
try:
    PML_DDPM
except NameError:
    try:
        PML_DDPM = DDPM  # type: ignore[name-defined]
    except NameError:
        pass

try:
    PML_VanillaDiffusionConfig
except NameError:
    try:
        PML_VanillaDiffusionConfig = VanillaDiffusionConfig  # type: ignore[name-defined]
    except NameError:
        pass
    
