# src/metrics.py
from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

import torch

from .utils import PML_kw

__all__ = [
    "PML_compute_freqs_1d",
    "PML_build_band_mask",
    "PML_spectral_l2_bands",
    "PML_snr_lowband",
]


def PML_compute_freqs_1d(n_H: int, f_dt: float, **kwargs: Any) -> torch.Tensor:
    """
    One-sided rFFT frequency axis for the time dimension.

    Args:
        n_H: number of time samples (rows)
        f_dt: sample interval [s]

    Optional **kwargs:
        device = "cpu"
        dtype = torch.float32

    Returns:
        (n_H//2+1,) tensor of Hz
    """
    device = PML_kw("device", kwargs, "cpu")
    dtype = PML_kw("dtype", kwargs, torch.float32)
    # torch.fft.rfftfreq introduced in newer versions; emulate via arange
    n = int(n_H)
    freqs = torch.arange(0, n // 2 + 1, device=device, dtype=dtype) / (n * f_dt)
    return freqs


def PML_build_band_mask(
    n_H: int,
    f_dt: float,
    tn_band: Tuple[float, float],
    **kwargs: Any,
) -> torch.Tensor:
    """
    Boolean mask over rFFT bins selecting [f_lo, f_hi) Hz (hi inclusive if equals Nyquist).

    Args:
        n_H: time samples
        f_dt: sample interval [s]
        tn_band: (f_lo, f_hi) in Hz

    Optional:
        device, dtype

    Returns:
        (n_H//2+1,) bool tensor
    """
    device = PML_kw("device", kwargs, "cpu")
    dtype = PML_kw("dtype", kwargs, torch.float32)

    f_lo = float(tn_band[0])
    f_hi = float(tn_band[1])

    freqs = PML_compute_freqs_1d(n_H, f_dt, device=device, dtype=dtype)
    f_nyq = 0.5 / f_dt
    # clamp hi to nyquist
    f_hi_eff = min(f_hi, f_nyq + 1e-12)

    mask = (freqs >= f_lo) & (freqs < f_hi_eff)
    # if hi matches nyquist exactly, include the last bin
    if abs(f_hi - f_nyq) < 1e-9:
        mask = (freqs >= f_lo) & (freqs <= f_hi_eff)
    return mask


def _rfft_time(x: torch.Tensor, c_norm: str = "ortho") -> torch.Tensor:
    """
    rFFT across time axis (dim=-2). Preserves batch/channel/offset dims.
    """
    return torch.fft.rfft(x, dim=-2, norm=c_norm)


def PML_spectral_l2_bands(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    f_dt: float,
    l_bands: Sequence[Tuple[float, float]],
    **kwargs: Any,
) -> torch.Tensor:
    """
    Per-band complex L2 (mean of |Yh - Y|^2 within band) over the rFFT along time.

    Args:
        y_hat, y_true: (B, C, H, W)
        f_dt: sample interval [s]
        l_bands: list of (f_lo, f_hi) in Hz

    Optional:
        c_norm (str) = "ortho"

    Returns:
        (B, len(l_bands)) tensor of bandwise L2 errors
    """
    assert y_hat.shape == y_true.shape and y_hat.ndim == 4
    c_norm = PML_kw("c_norm", kwargs, "ortho")

    B, C, H, W = y_hat.shape
    Yh = _rfft_time(y_hat, c_norm)  # (B,C,Hf,W)
    Yt = _rfft_time(y_true, c_norm)
    diff = Yh - Yt

    # power per bin
    pwr = (diff.real ** 2 + diff.imag ** 2)  # (B,C,Hf,W)

    l_vals: List[torch.Tensor] = []
    for tn in l_bands:
        m = PML_build_band_mask(H, f_dt, tn, device=y_hat.device, dtype=y_hat.dtype)  # (Hf,)
        # mean over channels, offsets, and selected freq bins
        # avoid empty band: if no bin selected, return 0 for that band
        if not torch.any(m):
            l_vals.append(torch.zeros(B, device=y_hat.device, dtype=y_hat.dtype))
            continue
        sel = pwr[:, :, m, :]  # (B,C,Hsel,W)
        val = sel.mean(dim=(1, 2, 3))  # (B,)
        l_vals.append(val)

    return torch.stack(l_vals, dim=1)  # (B, n_bands)


def PML_snr_lowband(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    f_dt: float,
    f_fmax_low: float,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SNR (dB) in low-frequency band [0, f_fmax_low] using rFFT on time axis.

    Args:
        y_hat, y_true: (B,C,H,W)
        f_dt: sample interval [s]
        f_fmax_low: band upper edge in Hz (inclusive clamped to Nyquist)

    Optional:
        c_norm (str) = "ortho"
        f_eps (float) = 1e-12

    Returns:
        (B,) tensor of SNR dB per sample
    """
    c_norm = PML_kw("c_norm", kwargs, "ortho")
    f_eps = float(PML_kw("f_eps", kwargs, 1e-12))

    B, C, H, W = y_true.shape
    Yt = _rfft_time(y_true, c_norm)  # (B,C,Hf,W)
    Yh = _rfft_time(y_hat, c_norm)
    freqs = PML_compute_freqs_1d(H, f_dt, device=y_true.device, dtype=y_true.dtype)
    f_nyq = 0.5 / f_dt
    f_hi = min(float(f_fmax_low), f_nyq + 1e-12)
    m = (freqs >= 0.0) & (freqs <= f_hi)
    if not torch.any(m):
        return torch.full((B,), float("-inf"), device=y_true.device, dtype=y_true.dtype)

    sig = (Yt[:, :, m, :].real ** 2 + Yt[:, :, m, :].imag ** 2).mean(dim=(1, 2, 3))  # (B,)
    err = ( (Yh - Yt)[:, :, m, :].real ** 2 + (Yh - Yt)[:, :, m, :].imag ** 2 ).mean(dim=(1, 2, 3)) + f_eps
    snr = 10.0 * torch.log10((sig + f_eps) / err)
    return snr
