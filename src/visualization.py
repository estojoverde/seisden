# src/visualization.py
from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # safe for headless test runners
import matplotlib.pyplot as plt

from .utils import PML_kw


__all__ = [
    "PML_to_numpy_hw",
    "PML_plot_time_section",
    "PML_plot_mean_spectrum",
    "PML_plot_mean_spectrum_compare",
    "PML_save_examples_grid",
]


ArrayLike = Union[np.ndarray, torch.Tensor]


def PML_to_numpy_hw(x: ArrayLike) -> np.ndarray:
    """
    Convert tensor/array to numpy (H,W). Accepts (H,W), (1,H,W), (C,H,W), (B,1,H,W).
    Keeps values (no normalization).

    Returns:
        np.ndarray (H, W), float32
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 2:
        hw = x
    elif x.ndim == 3:
        hw = x[0]
    elif x.ndim == 4:
        hw = x[0, 0]
    else:
        raise ValueError(f"Expected 2D/3D/4D array. Got shape {x.shape}")
    return np.asarray(hw, dtype=np.float32, order="C")


def _time_axis_seconds(n_H: int, f_dt: float) -> np.ndarray:
    return np.arange(n_H, dtype=np.float64) * float(f_dt)


def _compute_mean_amp_spectrum_hw(
    hw: np.ndarray,
    f_dt: float,
    *,
    c_norm: str = "ortho",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rFFT along time (axis=0) and average amplitude over offsets.

    Args:
        hw: (H,W) float32
        f_dt: sample interval [s]
        c_norm: FFT normalization

    Returns:
        freqs (Hf,), mean_amp (Hf,)
    """
    H, W = hw.shape
    X = np.fft.rfft(hw, axis=0, norm=c_norm)  # (Hf, W)
    amp = np.abs(X).mean(axis=1)              # (Hf,)
    freqs = np.fft.rfftfreq(H, d=f_dt)        # (Hf,)
    return freqs, amp


def _apply_percentile_clip(hw: np.ndarray, f_pmin: float, f_pmax: float) -> Tuple[float, float]:
    vmin = np.percentile(hw, f_pmin) if not np.isnan(f_pmin) else None
    vmax = np.percentile(hw, f_pmax) if not np.isnan(f_pmax) else None
    return float(vmin) if vmin is not None else None, float(vmax) if vmax is not None else None


def PML_plot_time_section(
    x: ArrayLike,
    f_dt: float,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a time-offset section as an image.

    Required:
        x: (H,W) or (1,H,W) or (B,1,H,W)
        f_dt: sample interval [s]

    Optional **kwargs:
        c_cmap="gray", f_aspect=0.5 (time:offset pixel ratio),
        f_pmin=1.0, f_pmax=99.0 (percentile clip), c_title=None,
        tn_figsize=(6,4), c_savepath=None
        # Axes labels
        c_xlabel="Offset (traces)", c_ylabel="Time (s)"
    """
    c_cmap = PML_kw("c_cmap", kwargs, "gray")
    f_aspect = float(PML_kw("f_aspect", kwargs, 0.5))
    f_pmin = float(PML_kw("f_pmin", kwargs, 1.0))
    f_pmax = float(PML_kw("f_pmax", kwargs, 99.0))
    c_title = PML_kw("c_title", kwargs, None)
    tn_figsize = PML_kw("tn_figsize", kwargs, (6, 4))
    c_savepath = PML_kw("c_savepath", kwargs, None)
    c_xlabel = PML_kw("c_xlabel", kwargs, "Offset (traces)")
    c_ylabel = PML_kw("c_ylabel", kwargs, "Time (s)")

    hw = PML_to_numpy_hw(x)
    H, W = hw.shape
    t = _time_axis_seconds(H, f_dt)
    vmin, vmax = _apply_percentile_clip(hw, f_pmin, f_pmax)

    fig, ax = plt.subplots(1, 1, figsize=tn_figsize, constrained_layout=True)
    im = ax.imshow(
        hw,
        origin="upper",
        cmap=c_cmap,
        aspect=f_aspect,
        vmin=vmin,
        vmax=vmax,
        extent=[0, W, t[-1], t[0]],  # y is time in seconds (top is 0s)
    )
    ax.set_xlabel(c_xlabel)
    ax.set_ylabel(c_ylabel)
    if c_title is not None:
        ax.set_title(str(c_title))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if c_savepath:
        os.makedirs(os.path.dirname(c_savepath), exist_ok=True)
        fig.savefig(c_savepath, dpi=150)

    return fig, ax


def PML_plot_mean_spectrum(
    x: ArrayLike,
    f_dt: float,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes, Tuple[np.ndarray, np.ndarray]]:
    """
    Plot mean amplitude spectrum over offsets (rFFT along time).

    Required:
        x: (H,W)/(1,H,W)/(B,1,H,W)
        f_dt: sample interval [s]

    Optional:
        c_color=None, c_label=None, b_logy=False, c_title=None,
        tn_figsize=(6,3), c_savepath=None, l_bands=None for shaded bands
    """
    c_color = PML_kw("c_color", kwargs, None)
    c_label = PML_kw("c_label", kwargs, None)
    b_logy = bool(PML_kw("b_logy", kwargs, False))
    c_title = PML_kw("c_title", kwargs, None)
    tn_figsize = PML_kw("tn_figsize", kwargs, (6, 3))
    c_savepath = PML_kw("c_savepath", kwargs, None)
    l_bands: Optional[Sequence[Tuple[float, float]]] = PML_kw("l_bands", kwargs, None)

    hw = PML_to_numpy_hw(x)
    freqs, amp = _compute_mean_amp_spectrum_hw(hw, f_dt, c_norm="ortho")

    fig, ax = plt.subplots(1, 1, figsize=tn_figsize, constrained_layout=True)
    ax.plot(freqs, amp, label=c_label, color=c_color)
    if b_logy:
        ax.set_yscale("log")
    if l_bands:
        y_top = float(amp.max() if amp.size > 0 else 1.0)
        for f0, f1 in l_bands:
            ax.axvspan(float(f0), float(f1), color="tab:gray", alpha=0.15, lw=0)
        ax.set_ylim(bottom=0.0, top=y_top * 1.05)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean amplitude")
    if c_title is not None:
        ax.set_title(str(c_title))
    if c_label is not None:
        ax.legend()

    if c_savepath:
        os.makedirs(os.path.dirname(c_savepath), exist_ok=True)
        fig.savefig(c_savepath, dpi=150)

    return fig, ax, (freqs, amp)


def PML_plot_mean_spectrum_compare(
    l_x: Sequence[ArrayLike],
    f_dt: float,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Plot multiple spectra on the same axes.

    Required:
        l_x: list of arrays/tensors (each -> (H,W))
        f_dt: sample interval [s]

    Optional:
        l_labels=None, b_logy=False, tn_figsize=(7,4), c_title=None,
        c_savepath=None, l_bands=None
    """
    l_labels = PML_kw("l_labels", kwargs, None)
    b_logy = bool(PML_kw("b_logy", kwargs, False))
    tn_figsize = PML_kw("tn_figsize", kwargs, (7, 4))
    c_title = PML_kw("c_title", kwargs, None)
    c_savepath = PML_kw("c_savepath", kwargs, None)
    l_bands: Optional[Sequence[Tuple[float, float]]] = PML_kw("l_bands", kwargs, None)

    fig, ax = plt.subplots(1, 1, figsize=tn_figsize, constrained_layout=True)
    outs: List[Tuple[np.ndarray, np.ndarray]] = []

    for i, xi in enumerate(l_x):
        hw = PML_to_numpy_hw(xi)
        freqs, amp = _compute_mean_amp_spectrum_hw(hw, f_dt, c_norm="ortho")
        outs.append((freqs, amp))
        label = l_labels[i] if (l_labels is not None and i < len(l_labels)) else None
        ax.plot(freqs, amp, label=label)

    if b_logy:
        ax.set_yscale("log")
    if l_bands:
        y_top = max([amp.max() if len(amp) else 1.0 for (_, amp) in outs] + [1.0])
        for f0, f1 in l_bands:
            ax.axvspan(float(f0), float(f1), color="tab:gray", alpha=0.15, lw=0)
        ax.set_ylim(bottom=0.0, top=y_top * 1.05)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean amplitude")
    if c_title is not None:
        ax.set_title(str(c_title))
    if l_labels is not None:
        ax.legend()

    if c_savepath:
        os.makedirs(os.path.dirname(c_savepath), exist_ok=True)
        fig.savefig(c_savepath, dpi=150)

    return fig, ax, outs


def PML_save_examples_grid(
    x_lowcut: ArrayLike,
    y_fullband: ArrayLike,
    y_recon: ArrayLike,
    f_dt: float,
    **kwargs: Any,
) -> List[str]:
    """
    Save a small grid per-sample with: input, reconstruction, target, error, and spectra.

    Required:
        x_lowcut, y_fullband, y_recon: (B,1,H,W) or  (H,W)

    Optional **kwargs:
        n_max=3, c_dir="runs/vis", c_prefix="sample",
        f_pmin=1.0, f_pmax=99.0, l_bands=None, b_logy=False
    """
    n_max = int(PML_kw("n_max", kwargs, 3))
    c_dir = PML_kw("c_dir", kwargs, "runs/vis")
    c_prefix = PML_kw("c_prefix", kwargs, "sample")
    f_pmin = float(PML_kw("f_pmin", kwargs, 1.0))
    f_pmax = float(PML_kw("f_pmax", kwargs, 99.0))
    l_bands: Optional[Sequence[Tuple[float, float]]] = PML_kw("l_bands", kwargs, None)
    b_logy = bool(PML_kw("b_logy", kwargs, False))

    # Normalize inputs to numpy batches
    def _to_batch_hw(a: ArrayLike) -> np.ndarray:
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        a = np.asarray(a)
        if a.ndim == 2:
            a = a[None, None]
        elif a.ndim == 3:
            a = a[None]
        elif a.ndim == 4:
            pass
        else:
            raise ValueError(f"Expected 2D/3D/4D. Got {a.shape}")
        return a.astype(np.float32, copy=False)

    xa = _to_batch_hw(x_lowcut)
    ya = _to_batch_hw(y_fullband)
    za = _to_batch_hw(y_recon)

    assert xa.shape == ya.shape == za.shape, "All inputs must share (B,1,H,W) shape."
    B, _, H, W = xa.shape

    os.makedirs(c_dir, exist_ok=True)
    saved: List[str] = []

    for i in range(min(B, n_max)):
        x_hw = xa[i, 0]
        y_hw = ya[i, 0]
        z_hw = za[i, 0]
        e_hw = z_hw - y_hw

        # Percentile limits shared across panels for readability
        vmin, vmax = _apply_percentile_clip(np.stack([x_hw, z_hw, y_hw], axis=0), f_pmin, f_pmax)

        fig = plt.figure(figsize=(10, 6), constrained_layout=True)
        gs = fig.add_gridspec(2, 3)

        # Row 1: images
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        im1 = ax1.imshow(x_hw, origin="upper", cmap="gray", vmin=vmin, vmax=vmax,
                         extent=[0, W, (H-1)*f_dt, 0])
        ax1.set_title("Low-cut input")
        ax1.set_xlabel("Offset"); ax1.set_ylabel("Time (s)")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(z_hw, origin="upper", cmap="gray", vmin=vmin, vmax=vmax,
                         extent=[0, W, (H-1)*f_dt, 0])
        ax2.set_title("Reconstruction")
        ax2.set_xlabel("Offset"); ax2.set_ylabel("Time (s)")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        im3 = ax3.imshow(y_hw, origin="upper", cmap="gray", vmin=vmin, vmax=vmax,
                         extent=[0, W, (H-1)*f_dt, 0])
        ax3.set_title("Target full-band")
        ax3.set_xlabel("Offset"); ax3.set_ylabel("Time (s)")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Row 2: error + spectra
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(e_hw, origin="upper", cmap="magma",
                         extent=[0, W, (H-1)*f_dt, 0])
        ax4.set_title("Error (recon - target)")
        ax4.set_xlabel("Offset"); ax4.set_ylabel("Time (s)")
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        ax5 = fig.add_subplot(gs[1, 1:])
        # spectra compare
        freqs_x, amp_x = _compute_mean_amp_spectrum_hw(x_hw, f_dt, c_norm="ortho")
        freqs_z, amp_z = _compute_mean_amp_spectrum_hw(z_hw, f_dt, c_norm="ortho")
        freqs_y, amp_y = _compute_mean_amp_spectrum_hw(y_hw, f_dt, c_norm="ortho")
        ax5.plot(freqs_x, amp_x, label="input")
        ax5.plot(freqs_z, amp_z, label="recon")
        ax5.plot(freqs_y, amp_y, label="target")
        if b_logy:
            ax5.set_yscale("log")
        if l_bands:
            for f0, f1 in l_bands:
                ax5.axvspan(float(f0), float(f1), color="tab:gray", alpha=0.15, lw=0)
        ax5.set_xlabel("Frequency (Hz)")
        ax5.set_ylabel("Mean amplitude")
        ax5.set_title("Mean spectra")
        ax5.legend()

        c_path = os.path.join(c_dir, f"{c_prefix}_{i:03d}.png")
        fig.savefig(c_path, dpi=150)
        plt.close(fig)
        saved.append(c_path)

    return saved
