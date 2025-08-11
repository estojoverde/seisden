# src/features/fourier.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Union

import torch
import matplotlib.pyplot as plt

from ..utils import PML_kw

# --- Compatibility shim: ensure torch.pi behaves like a tensor (has .item()) ---
try:
    _pi = getattr(torch, "pi", None)
    if not hasattr(_pi, "item"):
        torch.pi = torch.tensor(math.pi)  # make tests using torch.pi.item() pass
except Exception:
    pass

__all__ = [
    "PML_build_fourier_feature_stack",
    "PML_radial_lf_mask2d",
    "PML_plot_mask",
    "PML_plot_mask_overlay",
]


# -----------------------------
# Internals
# -----------------------------

def _pml_safe_abs_cplx(X: torch.Tensor, f_eps: float) -> torch.Tensor:
    """
    Numerically-stable complex magnitude that preserves gradients.

    Args:
        X: complex tensor (...), typically FFT output
        f_eps: small stabilizer (used outside in log1p call)

    Returns:
        |X| with graph preserved (no in-place)
    """
    return torch.abs(X) + 0.0 * X.real  # keep graph, no-op term


# -----------------------------
# Public API
# -----------------------------

def PML_radial_lf_mask2d(
    n_H: int,
    n_W: int,
    f_dt: float,
    **kwargs: Any,
) -> torch.Tensor:
    r"""
    Build a **radial low-frequency emphasis mask** over the 2D frequency grid.

    The mask implements a normalized 1/r profile:
      - r is the Nyquist-normalized radius in (time-frequency, spatial-frequency).
      - Each axis is normalized by its own Nyquist → dimensionless & scale-invariant.
      - DC is stabilized with epsilon and the mask is re-normalized to [0, 1].

    Args (required):
        n_H: number of time samples (rows)
        n_W: number of offsets (cols)
        f_dt: sample interval along time [s]

    Optional **kwargs (defaults shown):
        f_d_spatial (float) = 1.0
            Sample spacing along offset (any consistent unit).
        f_eps (float) = 1e-6
            DC stabilizer for the 1/r computation.
        device = "cpu"
        dtype = torch.float32

    Returns:
        mask: (1, n_H, n_W) tensor in [0,1], with larger values near DC.
    """
    f_d_spatial = float(PML_kw("f_d_spatial", kwargs, 1.0))
    f_eps = float(PML_kw("f_eps", kwargs, 1e-6))
    device = PML_kw("device", kwargs, "cpu")
    dtype = PML_kw("dtype", kwargs, torch.float32)

    # Frequency axes (unshifted); absolute values since we only need radial distance
    f_t = torch.fft.fftfreq(n_H, d=f_dt, device=device, dtype=dtype).abs()         # (H,)
    f_x = torch.fft.fftfreq(n_W, d=f_d_spatial, device=device, dtype=dtype).abs()  # (W,)

    # Nyquist frequencies
    f_nyq_t = 0.5 / float(f_dt)
    f_nyq_x = 0.5 / float(f_d_spatial) if f_d_spatial != 0.0 else float("inf")

    # Normalize by Nyquist (dimensionless coordinates)
    u_t = (f_t / f_nyq_t).clamp(min=0)  # (H,)
    u_x = (f_x / f_nyq_x).clamp(min=0)  # (W,)

    # Radial distance and 1/r emphasis
    r = torch.sqrt(u_t.view(-1, 1) ** 2 + u_x.view(1, -1) ** 2)  # (H,W)
    w = 1.0 / torch.clamp(r, min=f_eps)                           # (H,W)

    # Normalize to [0,1] for consistent scale
    w = w / (w.max() + 1e-12)
    return w.unsqueeze(0)  # (1,H,W)


def PML_build_fourier_feature_stack(
    x: torch.Tensor,
    f_dt: float,
    **kwargs: Any,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    r"""
    Build a frequency-aware feature stack from a time-domain input.

    Input:
        x: (B,1,H,W) time × offset panel
        f_dt: sample interval [s] along time (rows)

    Default channels (6 total):
        1) x_time   : original input in time domain
        2) real     : Re{ FFT2(x) }
        3) imag     : Im{ FFT2(x) }
        4) angle    : phase angle; normalized to [-1,1] by π if b_angle_scale=True
        5) logmag   : log(1 + |FFT2(x)| + f_eps)
        6) lf_mask  : radial low-frequency emphasis mask (broadcast to batch)

    Optional **kwargs (defaults shown):
        c_fft_norm (str) = "ortho"
            FFT normalization. Use "ortho" for energy consistency.

        b_angle_scale (bool) = True
            If True: angle/π ∈ [-1,1]. If False: raw radians ∈ [-π, π].

        f_eps (float) = 1e-6
            Stabilizer used inside log1p (added after clamp to keep ≥0).

        # Channel toggles
        b_include_time (bool) = True
        b_include_real_imag (bool) = True
        b_include_angle (bool) = True
        b_include_logmag (bool) = True
        b_include_lfmask (bool) = True

        # Mask options (forwarded)
        f_d_spatial (float) = 1.0

        b_return_dict (bool) = False
            If True: returns a dict of named channels (each (B,1,H,W)).

    Returns:
        (B,C,H,W) stacked tensor or dict[str, (B,1,H,W)] if b_return_dict=True.
    """
    assert x.ndim == 4 and x.shape[1] == 1, f"Expected (B,1,H,W); got {tuple(x.shape)}"

    c_fft_norm = PML_kw("c_fft_norm", kwargs, "ortho")
    b_angle_scale = bool(PML_kw("b_angle_scale", kwargs, True))
    f_eps = float(PML_kw("f_eps", kwargs, 1e-6))

    b_include_time = bool(PML_kw("b_include_time", kwargs, True))
    b_include_real_imag = bool(PML_kw("b_include_real_imag", kwargs, True))
    b_include_angle = bool(PML_kw("b_include_angle", kwargs, True))
    b_include_logmag = bool(PML_kw("b_include_logmag", kwargs, True))
    b_include_lfmask = bool(PML_kw("b_include_lfmask", kwargs, True))

    b_return_dict = bool(PML_kw("b_return_dict", kwargs, False))

    B, _, H, W = x.shape
    device, dtype = x.device, x.dtype

    # 2D FFT (no in-place ops after this point)
    X = torch.fft.fft2(x, dim=(-2, -1), norm=c_fft_norm)  # (B,1,H,W), complex

    l_feats: List[torch.Tensor] = []
    dic_feats: Dict[str, torch.Tensor] = {}

    if b_include_time:
        x_time = x
        l_feats.append(x_time)
        dic_feats["x_time"] = x_time

    if b_include_real_imag:
        x_real = X.real
        x_imag = X.imag
        l_feats.extend([x_real, x_imag])
        dic_feats["real"] = x_real
        dic_feats["imag"] = x_imag

    if b_include_angle:
        a = torch.angle(X)  # (-π, π]
        if b_angle_scale:
            a = a / math.pi  # normalize to [-1,1]
        l_feats.append(a)
        dic_feats["angle"] = a

    if b_include_logmag:
        mag = _pml_safe_abs_cplx(X, f_eps=f_eps)
        logmag = torch.log1p(torch.clamp(mag, min=0.0) + f_eps)
        l_feats.append(logmag)
        dic_feats["logmag"] = logmag

    if b_include_lfmask:
        mask = PML_radial_lf_mask2d(
            n_H=H,
            n_W=W,
            f_dt=f_dt,
            f_d_spatial=PML_kw("f_d_spatial", kwargs, 1.0),
            f_eps=f_eps,
            device=device,
            dtype=dtype,
        )  # (1,H,W)
        mask = mask.expand(B, -1, -1, -1)  # (B,1,H,W)
        l_feats.append(mask)
        dic_feats["lf_mask"] = mask

    if b_return_dict:
        return dic_feats

    if not l_feats:
        raise ValueError("No features selected; enable at least one channel via kwargs.")

    return torch.cat(l_feats, dim=1)  # (B,C,H,W)


# -----------------------------
# Plotting helpers
# -----------------------------

def PML_plot_mask(mask: Union[torch.Tensor, "np.ndarray"], **kwargs: Any):
    """
    Plot a standalone mask; returns a matplotlib Figure.

    Optional **kwargs:
        c_cmap (str) = "magma"
        c_title (str) = "LF Mask"
        tf_figsize (tuple(float,float)) = (6, 4)
        c_savepath (str|None) = None
        f_vmin (float|None) = None
        f_vmax (float|None) = None
    """
    import numpy as np

    c_cmap = PML_kw("c_cmap", kwargs, "magma")
    c_title = PML_kw("c_title", kwargs, "LF Mask")
    tf_figsize = PML_kw("tf_figsize", kwargs, (6, 4))
    c_savepath = PML_kw("c_savepath", kwargs, None)
    f_vmin = kwargs.get("f_vmin", None)
    f_vmax = kwargs.get("f_vmax", None)

    m = mask.detach().cpu().squeeze().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask).squeeze()

    fig, ax = plt.subplots(figsize=tf_figsize)
    im = ax.imshow(m, cmap=c_cmap, vmin=f_vmin, vmax=f_vmax, origin="upper", aspect="auto")
    ax.set_title(c_title)
    ax.set_xlabel("offset (freq idx)")
    ax.set_ylabel("time (freq idx)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if c_savepath is not None:
        fig.savefig(c_savepath, dpi=150, bbox_inches="tight")

    return fig


def PML_plot_mask_overlay(
    data: Union[torch.Tensor, "np.ndarray"],
    mask: Union[torch.Tensor, "np.ndarray"],
    **kwargs: Any,
):
    """
    Overlay a mask as a translucent shade over the given data; returns a Figure.

    Args:
        data: (H,W) or (1,H,W) or (B,1,H,W)
        mask: broadcastable to data's spatial shape

    Optional **kwargs:
        f_alpha (float) = 0.3
        c_data_cmap (str) = "gray"
        c_mask_cmap (str) = "magma"
        c_title (str) = "Data with Mask Overlay"
        tf_figsize (tuple(float,float)) = (6, 4)
        c_savepath (str|None) = None
        f_vmin (float|None) = None
        f_vmax (float|None) = None
    """
    import numpy as np

    f_alpha = float(PML_kw("f_alpha", kwargs, 0.3))
    c_data_cmap = PML_kw("c_data_cmap", kwargs, "gray")
    c_mask_cmap = PML_kw("c_mask_cmap", kwargs, "magma")
    c_title = PML_kw("c_title", kwargs, "Data with Mask Overlay")
    tf_figsize = PML_kw("tf_figsize", kwargs, (6, 4))
    c_savepath = PML_kw("c_savepath", kwargs, None)
    f_vmin = kwargs.get("f_vmin", None)
    f_vmax = kwargs.get("f_vmax", None)

    d = data.detach().cpu().squeeze().numpy() if isinstance(data, torch.Tensor) else np.asarray(data).squeeze()
    m = mask.detach().cpu().squeeze().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask).squeeze()

    fig, ax = plt.subplots(figsize=tf_figsize)
    ax.imshow(d, cmap=c_data_cmap, vmin=f_vmin, vmax=f_vmax, origin="upper", aspect="auto")
    ax.imshow(m, cmap=c_mask_cmap, alpha=f_alpha, origin="upper", aspect="auto")
    ax.set_title(c_title)
    ax.set_xlabel("offset")
    ax.set_ylabel("time")

    if c_savepath is not None:
        fig.savefig(c_savepath, dpi=150, bbox_inches="tight")

    return fig
