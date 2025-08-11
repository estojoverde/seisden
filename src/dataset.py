# src/dataset.py
from __future__ import annotations

from typing import Optional, Tuple, Union, Dict, Any
import os
import numpy as np
import torch

try:
    # Base class provided by the repo
    from .templates import PML_BasicDataset
except Exception as e:  # pragma: no cover - make the error explicit if missing
    raise ImportError(
        "Could not import PML_BasicDataset from src.templates. "
        "Please ensure templates.py defines class `PML_BasicDataset`."
    ) from e

from .utils import PML_kw

ArrayLike = Union[str, np.ndarray]  # path to .npy or already-loaded ndarray

__all__ = [
    "PML_NpyPairedSeismic",
    "PML_apply_lowcut_fft",
    "PML_rfftfreq",
]


# ---------- Internal helpers (private) ----------

def _ensure_4d(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array shape is (N, 1, H, W). Accepts (N, H, W) or (N, 1, H, W).

    Raises:
        ValueError if the input array cannot be coerced to (N,1,H,W).
    """
    if arr.ndim == 3:
        arr = arr[:, None, ...]
    if arr.ndim != 4 or arr.shape[1] != 1:
        raise ValueError(
            f"Expected array with shape (N,1,H,W) or (N,H,W). Got {arr.shape}."
        )
    return arr


def _load_npy(maybe_path: Optional[ArrayLike]) -> Optional[np.ndarray]:
    """
    Load .npy if string path, else return array copy as float32 C-contiguous.
    Allows None (returns None).
    """
    if maybe_path is None:
        return None
    if isinstance(maybe_path, str):
        if not os.path.isfile(maybe_path):
            raise FileNotFoundError(f"File not found: {maybe_path}")
        arr = np.load(maybe_path)
    elif isinstance(maybe_path, np.ndarray):
        arr = maybe_path
    else:
        raise TypeError("Expected str path to .npy, numpy.ndarray, or None.")
    arr = np.asarray(arr, dtype=np.float32, order="C")
    return arr


# ---------- Public utilities (prefixed) ----------

def PML_rfftfreq(n: int, f_dt: float | None = None, **kwargs) -> np.ndarray:
    """
    Frequency bins for rFFT (one-sided), NumPy-compatible.

    Args:
        n: number of time samples (H).
        f_dt: sample interval in seconds (dt).

    Legacy aliases (accepted for back-compat/tests):
        d: same as f_dt (NumPy-style)

    Returns:
        np.ndarray of shape (n//2 + 1,) in Hz.
    """
    if f_dt is None:
        f_dt = kwargs.get("d", None)
    if f_dt is None:
        raise TypeError("PML_rfftfreq: must provide f_dt (or legacy d=).")
    return np.fft.rfftfreq(n, d=float(f_dt))


def PML_apply_lowcut_fft(
    panel_hw: np.ndarray,
    f_dt: float | None = None,
    f_low: float = 0.0,
    c_norm: str | None = None,
    **kwargs,
) -> np.ndarray:
    """
    High-pass (aka low-cut) a time–offset panel via 1D rFFT along time.

    Args:
        panel_hw: (H, W) time x offset, float32.
        f_dt: sample interval in seconds (e.g., 0.004).
        f_low: cutoff frequency in Hz. Components with f < f_low are removed.
        c_norm: FFT normalization ("backward", "ortho", or "forward"). Use "ortho"
                for energy-consistent transforms.

    Legacy aliases (accepted for back-compat/tests):
        dt -> f_dt
        norm -> c_norm

    Returns:
        filtered: (H, W) float32, same shape/order.
    """
    if f_dt is None:
        f_dt = kwargs.get("dt", None)
    if f_dt is None:
        raise TypeError("PML_apply_lowcut_fft: must provide f_dt (or legacy dt=).")
    if c_norm is None:
        c_norm = kwargs.get("norm", "ortho")
    c_norm = str(c_norm)

    H, W = panel_hw.shape
    if f_low <= 0.0:
        # No-op for non-positive cutoff (preserve phase/amplitude)
        return panel_hw.copy()

    f_nyq = 0.5 / float(f_dt)
    if f_low >= f_nyq:
        # All frequencies removed -> zeros
        return np.zeros_like(panel_hw, dtype=np.float32)

    X = np.fft.rfft(panel_hw, axis=0, norm=c_norm)  # (H//2+1, W), complex
    freqs = PML_rfftfreq(H, f_dt=float(f_dt))        # (H//2+1,)
    mask = (freqs >= float(f_low)).astype(np.float32)  # (H//2+1,)
    X_hp = X * mask[:, None]
    filtered = np.fft.irfft(X_hp, n=H, axis=0, norm=c_norm).astype(np.float32, copy=False)
    return filtered


# ---------- Dataset (public, prefixed) ----------

class PML_NpyPairedSeismic(PML_BasicDataset):
    r"""
    PyTorch Dataset for paired low-cut and full-band seismic panels.

    Each sample is a 2D time×offset panel, returned as tensors with shape (1, H, W).

    Supervision by default (paired mode):
        x_lowcut:   (1, H, W), low-frequency-attenuated ("low-cut") input
        y_fullband: (1, H, W), full-band target

    FULLBAND-ONLY INPUT mode:
        Set b_input_is_fullband=True to have the dataset return
        x = y = fullband (uncorrupted). This is useful when the **model**
        (e.g., Spectral DDPM) is responsible for applying all training-time
        corruption. In this mode, `lowcut` may be None.

    Optional on-the-fly augmentation (paired mode only):
        With probability f_p_aug, recompute x_lowcut by applying a *random low-cut*
        to either the fullband or existing lowcut array using FFT masking.

    Args (required):
        lowcut:   path to .npy or np.ndarray with shape (N,1,H,W) or (N,H,W).
                  May be None if b_input_is_fullband=True.
        fullband: path to .npy or np.ndarray with shape (N,1,H,W) or (N,H,W).

    Optional **kwargs (typed prefixes; legacy keys kept for compatibility):
        f_dt (float) = 0.004
            Sample interval in seconds.

        b_input_is_fullband (bool) = False
            If True: return x==y==fullband (uncorrupted input). Ignores augmentation.

        b_augment_lowcut (bool) = False
            Enable on-the-fly low-cut augmentation (paired mode only).

        dic_aug_params (dict):
            {
              "f_p": float in [0,1], probability of applying augmentation (default 0.5),
              "tf_f_low_range": (f_min, f_max) in Hz, inclusive (default (4.0, 12.0)),
              "c_source": "fullband" or "lowcut" (default "fullband"),
              "c_norm": "ortho" | "backward" | "forward" (default "ortho")
            }

        n_seed (int) = None
            Global base seed for determinism.

        b_deterministic (bool) = True
            If True, augmentation randomness depends on (seed, index).

        transform (callable) = None
            Optional callable applied to BOTH x and y tensors: (x, y) -> (x, y)

    Back-compat legacy kwargs (still accepted):
        dt -> f_dt, augment_lowcut -> b_augment_lowcut, aug_params -> dic_aug_params,
        seed -> n_seed, deterministic -> b_deterministic, norm -> c_norm
    """

    # Only the must-have args are positional; everything else comes from **kwargs.
    def __init__(self, lowcut: Optional[ArrayLike], fullband: ArrayLike, **kwargs: Any) -> None:
        # Optional: call base init if present
        try:
            super().__init__()
        except TypeError:
            pass

        # --- Resolve kwargs with back-compat ---
        f_dt = float(PML_kw("f_dt", kwargs, PML_kw("dt", kwargs, 0.004)))
        b_input_is_fullband = bool(PML_kw("b_input_is_fullband", kwargs, False))
        b_augment_lowcut = bool(PML_kw("b_augment_lowcut", kwargs, PML_kw("augment_lowcut", kwargs, False)))
        dic_aug_params = PML_kw("dic_aug_params", kwargs, PML_kw("aug_params", kwargs, {})) or {}
        n_seed = PML_kw("n_seed", kwargs, PML_kw("seed", kwargs, None))
        n_seed = int(n_seed) if n_seed is not None else None
        b_deterministic = bool(PML_kw("b_deterministic", kwargs, PML_kw("deterministic", kwargs, True)))
        transform = kwargs.get("transform", None)

        # --- Load arrays ---
        arr_full = _load_npy(fullband)
        if arr_full is None:
            raise ValueError("`fullband` must be provided (str path or ndarray).")
        arr_full = _ensure_4d(arr_full)

        arr_low = _load_npy(lowcut)
        if not b_input_is_fullband:
            if arr_low is None:
                raise ValueError("`lowcut` must be provided unless b_input_is_fullband=True.")
            arr_low = _ensure_4d(arr_low)
            if arr_low.shape != arr_full.shape:
                raise ValueError(f"Shape mismatch: lowcut {arr_low.shape} vs fullband {arr_full.shape}")

        # --- Store core ---
        self.full = np.asarray(arr_full, dtype=np.float32, order="C")
        self.low = np.asarray(arr_low, dtype=np.float32, order="C") if arr_low is not None else None
        self.N, self.C, self.H, self.W = self.full.shape

        self.f_dt = f_dt
        self.f_nyquist = 0.5 / self.f_dt

        self.b_input_is_fullband = b_input_is_fullband
        self.b_augment_lowcut = b_augment_lowcut and (not b_input_is_fullband)  # aug meaningless in fullband-only mode
        self.transform = transform
        self.n_seed = n_seed
        self.b_deterministic = b_deterministic

        # --- Aug params (paired mode only) ---
        dic_ap = dict(dic_aug_params) if isinstance(dic_aug_params, dict) else {}
        self.f_p_aug: float = float(PML_kw("f_p", dic_ap, PML_kw("p", dic_ap, 0.5)))
        tf_f_low_range = PML_kw("tf_f_low_range", dic_ap, PML_kw("f_low_range", dic_ap, (4.0, 12.0)))
        if not (isinstance(tf_f_low_range, (tuple, list)) and len(tf_f_low_range) == 2):
            raise ValueError("dic_aug_params['tf_f_low_range'] must be a (f_min, f_max) tuple.")
        self.f_low_min: float = float(min(tf_f_low_range))
        self.f_low_max: float = float(max(tf_f_low_range))
        self.c_aug_source: str = str(PML_kw("c_source", dic_ap, PML_kw("source", dic_ap, "fullband"))).lower()
        if self.c_aug_source not in ("fullband", "lowcut"):
            raise ValueError("dic_aug_params['c_source'] must be 'fullband' or 'lowcut'.")
        self.c_fft_norm: str = str(PML_kw("c_norm", dic_ap, PML_kw("norm", dic_ap, "ortho")))

    def __len__(self) -> int:
        return self.N

    # --- deterministic split ---

    @staticmethod
    def split_indices(
        n: int,
        f_val_frac: float | None = None,
        n_seed: int | None = None,
        b_shuffle: bool | None = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Deterministic index split for train/val.

        Args:
            f_val_frac: validation fraction in (0,1)
            n_seed: RNG seed
            b_shuffle: whether to shuffle before split

        Legacy aliases (kept for tests/back-compat):
            val_frac -> f_val_frac
            seed -> n_seed
            shuffle -> b_shuffle

        Returns:
            train_idx, val_idx (np.ndarray of ints)
        """
        if f_val_frac is None:
            f_val_frac = kwargs.get("val_frac", 0.1)
        if n_seed is None:
            n_seed = kwargs.get("seed", 42)
        if b_shuffle is None:
            b_shuffle = kwargs.get("shuffle", True)

        f_val_frac = float(f_val_frac)
        if not (0.0 < f_val_frac < 1.0):
            raise ValueError("f_val_frac must be in (0,1).")
        rng = np.random.RandomState(int(n_seed))
        idx = np.arange(n, dtype=np.int64)
        if bool(b_shuffle):
            rng.shuffle(idx)
        n_val = int(round(f_val_frac * n))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        return train_idx, val_idx

    # --- RNG per index for deterministic augmentation ---

    def _rng_for_index(self, i: int) -> np.random.RandomState:
        if self.n_seed is None:
            return np.random.RandomState()  # seeded from /dev/urandom
        return np.random.RandomState(self.n_seed ^ (i * 0x9E3779B1))

    # --- Augmentation (paired mode only) ---

    def _maybe_augment_lowcut(self, i: int, x_low_hw: np.ndarray, y_full_hw: np.ndarray) -> np.ndarray:
        if not self.b_augment_lowcut or self.f_p_aug <= 0.0:
            return x_low_hw

        rng = self._rng_for_index(i) if self.b_deterministic else np.random.RandomState()
        if rng.rand() >= self.f_p_aug:
            return x_low_hw

        f_low = float(rng.uniform(self.f_low_min, self.f_low_max))
        source = y_full_hw if self.c_aug_source == "fullband" else x_low_hw
        aug = PML_apply_lowcut_fft(source, f_dt=self.f_dt, f_low=f_low, c_norm=self.c_fft_norm)
        return aug

    # --- Main access ---

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_input:    torch.float32, (1, H, W)
            y_fullband: torch.float32, (1, H, W)

        Modes:
            - b_input_is_fullband=True:
                x_input == y_fullband == fullband[i]
            - Paired (default):
                x_input == lowcut[i] (optionally re-generated by augmentation)
                y_fullband == fullband[i]
        """
        y_np = self.full[i, 0]  # (H, W)

        if self.b_input_is_fullband:
            x_np = y_np
        else:
            x_np = self.low[i, 0]
            x_np = self._maybe_augment_lowcut(i, x_np, y_np)

        # Add channel dim
        x_np = x_np[None, ...]  # (1,H,W)
        y_np = y_np[None, ...]  # (1,H,W)

        # to tensors (contiguous)
        x = torch.from_numpy(np.ascontiguousarray(x_np))
        y = torch.from_numpy(np.ascontiguousarray(y_np))

        if self.transform is not None:
            x, y = self.transform(x, y)

        return x, y
