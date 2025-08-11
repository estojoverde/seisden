# src/losses.py
from __future__ import annotations

from typing import Any, Sequence, Tuple

import torch

from .metrics import PML_spectral_l2_bands

from .utils import PML_kw

__all__ = [
    "PML_spectral_l2_per_band",
    "PML_lowband_weighted_loss",
]


def PML_spectral_l2_per_band(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    f_dt: float,
    l_bands: Sequence[Tuple[float, float]],
    **kwargs: Any,
) -> torch.Tensor:
    """
    Compute per-band complex L2 error using rFFT along time; return (B, n_bands).

    Args:
        y_hat, y_true: (B,C,H,W)
        f_dt: sample interval [s]
        l_bands: list of (f_lo, f_hi) Hz

    Optional:
        c_norm (str) = "ortho"

    Returns:
        (B, n_bands) tensor; each entry is mean |Yh - Y|^2 within the band.
    """
    return PML_spectral_l2_bands(y_hat, y_true, f_dt, l_bands, **kwargs)


def PML_lowband_weighted_loss(
    residual_hat: torch.Tensor,
    residual_true: torch.Tensor,
    x_lowcut: torch.Tensor,
    y_fullband: torch.Tensor,
    f_dt: float,
    **kwargs: Any,
) -> torch.Tensor:
    r"""
    Combined residual time-domain loss + spectral band loss on reconstructed full-band.

    Primary objective: residual accuracy (time-domain).
    Auxiliary: spectral consistency of reconstructed full-band.

    Args:
        residual_hat: (B,1,H,W) predicted residual in time domain
        residual_true: (B,1,H,W) true residual = y_fullband - x_lowcut
        x_lowcut: (B,1,H,W) low-cut input
        y_fullband: (B,1,H,W) target full-band
        f_dt: sample interval [s]

    Optional **kwargs (defaults shown):
        c_time (str) = "l1"
            Time-domain loss: "l1" or "l2" on residuals.
        f_lambda_time (float) = 1.0
            Weight for time-domain residual loss.
        f_lambda_spec (float) = 0.5
            Weight for spectral auxiliary loss on full-band reconstruction.
        l_bands (list[tuple]) = [(0.0, 10.0), (10.0, 20.0), (20.0, 40.0)]
            Hz bands for spectral loss. Edit freely in config.
        f_lowband_boost (float) = 3.0
            Extra multiplier for the first (lowest) band.
        c_norm (str) = "ortho"
            FFT normalization.

    Returns:
        Scalar tensor loss.
    """
    assert residual_hat.shape == residual_true.shape == x_lowcut.shape == y_fullband.shape
    c_time = PML_kw("c_time", kwargs, "l1")
    f_lambda_time = float(PML_kw("f_lambda_time", kwargs, 1.0))
    f_lambda_spec = float(PML_kw("f_lambda_spec", kwargs, 0.5))
    l_bands = list(PML_kw("l_bands", kwargs, [(0.0, 10.0), (10.0, 20.0), (20.0, 40.0)]))
    f_lowband_boost = float(PML_kw("f_lowband_boost", kwargs, 3.0))
    c_norm = PML_kw("c_norm", kwargs, "ortho")

    # --- Time-domain loss on residuals
    if c_time == "l1":
        loss_time = (residual_hat - residual_true).abs().mean()
    elif c_time == "l2":
        loss_time = ((residual_hat - residual_true) ** 2).mean()
    else:
        raise ValueError(f"Unsupported c_time='{c_time}' (use 'l1' or 'l2').")

    # --- Reconstruct full-band and spectral auxiliary loss
    y_hat_full = x_lowcut + residual_hat
    # per-band errors, shape (B, n_bands)
    spec_per_band = PML_spectral_l2_bands(y_hat_full, y_fullband, f_dt, l_bands, c_norm=c_norm)
    # low-band boost on the first band
    if spec_per_band.shape[1] > 0 and f_lowband_boost != 1.0:
        spec_per_band = spec_per_band.clone()
        spec_per_band[:, 0] = spec_per_band[:, 0] * f_lowband_boost
    loss_spec = spec_per_band.mean()

    loss = f_lambda_time * loss_time + f_lambda_spec * loss_spec
    return loss
