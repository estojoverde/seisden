# train_diffusion.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Local imports
from src.dataset import PML_NpyPairedSeismic, PML_apply_lowcut_fft
from src.metrics import PML_snr_lowband, PML_spectral_l2_bands
from src.utils import PML_kw

# We import diffusion and unet lazily inside builders to avoid import issues for users who only need utilities.


# -----------------------------
# Seeding
# -----------------------------
def PML_seed_everything(n_seed: int = 42) -> None:
    """
    Set seeds for reproducibility across numpy/torch.
    """
    import random
    random.seed(n_seed)
    np.random.seed(n_seed)
    torch.manual_seed(n_seed)
    torch.cuda.manual_seed_all(n_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # speed


# -----------------------------
# Datasets & loaders
# -----------------------------
def PML_build_dataset_from_paths(
    c_lowcut_path: str,
    c_fullband_path: str,
    **kwargs: Any,
) -> PML_NpyPairedSeismic:
    """
    Build PML_NpyPairedSeismic with the repo defaults; **kwargs forward to dataset.
    """
    # Defaults for our training: we generally pass b_fullband_only=True
    # to serve uncorrupted pairs and let SpectralDDPM apply corruption.
    dt = float(PML_kw("f_dt", kwargs, 0.004))
    b_fullband_only = bool(PML_kw("b_fullband_only", kwargs, True))
    augment_lowcut = bool(PML_kw("b_augment_lowcut", kwargs, False))
    aug_params = PML_kw("dic_aug_params", kwargs, None)
    seed = PML_kw("n_seed", kwargs, None)
    deterministic = bool(PML_kw("b_deterministic", kwargs, True))
    transform = PML_kw("transform", kwargs, None)

    ds = PML_NpyPairedSeismic(
        lowcut=c_lowcut_path,
        fullband=c_fullband_path,
        f_dt=dt,
        b_fullband_only=b_fullband_only,
        b_augment_lowcut=augment_lowcut,
        dic_aug_params=aug_params,
        n_seed=seed,
        b_deterministic=deterministic,
        transform=transform,
    )
    return ds


def PML_build_dataloaders(
    ds_train: Dataset,
    ds_val: Dataset,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader]:
    """
    Make torch DataLoaders with sensible defaults.
    """
    n_bs = int(PML_kw("n_batch_size", kwargs, 4))
    n_workers = int(PML_kw("n_workers", kwargs, 0))
    b_pin = bool(PML_kw("b_pin_memory", kwargs, True))
    b_shuffle = bool(PML_kw("b_shuffle", kwargs, True))
    n_val_bs = int(PML_kw("n_val_batch_size", kwargs, max(1, n_bs // 2)))

    train_loader = DataLoader(
        ds_train, batch_size=n_bs, shuffle=b_shuffle, num_workers=n_workers, pin_memory=b_pin, drop_last=True
    )
    val_loader = DataLoader(
        ds_val, batch_size=n_val_bs, shuffle=False, num_workers=n_workers, pin_memory=b_pin, drop_last=False
    )
    return train_loader, val_loader


# -----------------------------
# Model builder
# -----------------------------
def PML_build_spectral_model(
    n_in: int = 7,
    n_out: int = 1,
    **kwargs: Any,
) -> nn.Module:
    """
    Construct SpectralDDPM with a UNet backbone.

    Optional **kwargs (selected):
        l_n_channels: list[int] feature widths per level (default [64,128,256,512])
        n_groups_gn: int for GroupNorm (default 8)
        b_use_attention: bool (default True)
        f_dropout: float (default 0.0)
        n_T: diffusion steps (default 1000)
        c_beta_schedule: "linear"|"cosine" (default "linear")
        n_ddim_steps: int steps for sampling (default 50)
        b_clip_denoised: bool (default True)
        b_predict_residual: bool (default True)
        dic_window_blend: dict for blending (type, alpha, mix)
    """
    # --- begin patch in train_diffusion.py (inside PML_build_spectral_model) ---

    # UNet
    from src.models.unet_blocks import PML_UNet

    # Spectral diffusion model class (try several canonical names)
    SpectralCls = None
    for _name in [
        "PML_SpectralDDPM",
        "SpectralDDPM",
        "PML_SpectralDiffusion",
        "SpectralDiffusion",
    ]:
        try:
            SpectralCls = getattr(__import__("src.models.diffusion", fromlist=[_name]), _name)
            break
        except Exception:
            pass
    if SpectralCls is None:
        raise ImportError(
            "Could not locate a Spectral Diffusion class in src.models.diffusion. "
            "Tried: PML_SpectralDDPM, SpectralDDPM, PML_SpectralDiffusion, SpectralDiffusion."
        )

    # Config class (try several canonical names)
    CfgCls = None
    for _cfg in [
        "PML_DiffusionConfig",
        "DiffusionConfig",
        "PML_SpectralDiffusionConfig",
        "SpectralDiffusionConfig",
    ]:
        try:
            CfgCls = getattr(__import__("src.models.diffusion", fromlist=[_cfg]), _cfg)
            break
        except Exception:
            pass
    if CfgCls is None:
        raise ImportError(
            "Could not locate a diffusion config class in src.models.diffusion. "
            "Tried: PML_DiffusionConfig, DiffusionConfig, PML_SpectralDiffusionConfig, SpectralDiffusionConfig."
        )

    # --- end patch ---
    try:
        # Prefer prefixed names
        from src.models.diffusion import PML_SpectralDDPM, PML_DiffusionConfig
    except Exception:
        # Backward-fallback
        from src.models.diffusion import SpectralDDPM as PML_SpectralDDPM  # type: ignore
        from src.models.diffusion import DiffusionConfig as PML_DiffusionConfig  # type: ignore

    # UNet
    unet = PML_UNet(
        n_in=n_in,
        n_out=n_out,
        l_n_channels=PML_kw("l_n_channels", kwargs, [64, 128, 256, 512]),
        n_groups_gn=PML_kw("n_groups_gn", kwargs, 8),
        b_use_attention=PML_kw("b_use_attention", kwargs, True),
        f_dropout=PML_kw("f_dropout", kwargs, 0.0),
        ksize=PML_kw("n_ksize", kwargs, 3),
        c_padding=PML_kw("c_padding", kwargs, "same"),
    )

    # Diffusion config
    cfg = PML_DiffusionConfig(
        n_T=int(PML_kw("n_T", kwargs, 1000)),
        c_beta_schedule=PML_kw("c_beta_schedule", kwargs, "linear"),
        b_v_prediction=bool(PML_kw("b_v_prediction", kwargs, False)),
        n_ddim_steps=int(PML_kw("n_ddim_steps", kwargs, 50)),
        b_clip_denoised=bool(PML_kw("b_clip_denoised", kwargs, True)),
        b_predict_residual=bool(PML_kw("b_predict_residual", kwargs, True)),
        dic_window_blend=PML_kw("dic_window_blend", kwargs, {"type": "tukey", "alpha": 0.25, "mix": 1.0}),
        # Spectral noise schedule params (no magic numbers; all configurable)
        f_dt=float(PML_kw("f_dt", kwargs, 0.004)),
        f_fmax_target=float(PML_kw("f_fmax_target", kwargs, 12.0)),     # target LF gap we want to learn (< Nyq)
        c_f_ramp=PML_kw("c_f_ramp", kwargs, "linear"),                  # how LF grows with t: "linear"|"sqrt"|"cosine"
        f_noise_gain=float(PML_kw("f_noise_gain", kwargs, 1.0)),        # spectral noise multiplier
        f_eps_stab=float(PML_kw("f_eps_stab", kwargs, 1e-6)),           # numerical epsilon in spectral ops
    )

    model = PML_SpectralDDPM(unet, cfg)
    return model


# -----------------------------
# Training / Validation loops
# -----------------------------
def PML_train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    f_dt: float,
    optim: torch.optim.Optimizer,
    **kwargs: Any,
) -> float:
    """
    Train for one epoch; returns mean loss.
    """
    b_amp = bool(PML_kw("b_amp", kwargs, True))
    f_clip = float(PML_kw("f_grad_clip", kwargs, 1.0))
    device = PML_kw("device", kwargs, "cuda" if torch.cuda.is_available() else "cpu")

    scaler = torch.cuda.amp.GradScaler(enabled=b_amp)
    model.train().to(device)
    total = 0.0
    n = 0

    for x_low, y_full in loader:
        x_low = x_low.to(device, non_blocking=True)
        y_full = y_full.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=b_amp):
            loss = model(x_low, y_full, f_dt)  # model returns training loss internally
        scaler.scale(loss).backward()
        if f_clip and f_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), f_clip)
        scaler.step(optim)
        scaler.update()

        total += float(loss.item())
        n += 1

    return total / max(1, n)


@torch.no_grad()
def PML_validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    f_dt: float,
    **kwargs: Any,
) -> Dict[str, float]:
    """
    Evaluate SNR LF and spectral L2 band errors on the validation loader.
    """
    device = PML_kw("device", kwargs, "cuda" if torch.cuda.is_available() else "cpu")
    f_fmax_low = float(PML_kw("f_fmax_low", kwargs, 10.0))
    l_bands = PML_kw("l_bands", kwargs, [(0.0, 10.0), (10.0, 20.0), (20.0, 40.0)])

    model.eval().to(device)

    l_snr = []
    l_err = []

    for x_low, y_full in loader:
        x_low = x_low.to(device, non_blocking=True)
        y_full = y_full.to(device, non_blocking=True)
        y_recon = model.reconstruct_fullband(x_low, f_dt)  # (B,1,H,W)

        snr = PML_snr_lowband(y_recon, y_full, f_dt, f_fmax_low=f_fmax_low)  # (B,)
        err = PML_spectral_l2_bands(y_recon, y_full, f_dt, l_bands)          # (B, nb)

        l_snr.append(snr.detach().cpu())
        l_err.append(err.detach().cpu())

    snr_mean = torch.cat(l_snr, 0).mean().item() if l_snr else float("nan")
    err_mean = torch.cat(l_err, 0).mean().item() if l_err else float("nan")
    return {"snr_lf_db": snr_mean, "spec_l2_mean": err_mean}


# -----------------------------
# Minimal train entry that tests quickly on CPU
# -----------------------------
def PML_minimal_training_run(
    a_low: np.ndarray,
    a_full: np.ndarray,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Convenience helper used by tests and quick sanity checks.

    Required:
        a_low, a_full: numpy arrays with shape (N,1,H,W) or (N,H,W)

    Optional:
        f_dt=0.004, n_steps=12, n_batch_size=2, n_lr=1e-3, n_T=200, n_ddim_steps=10,
        l_n_channels=[32,64], device="cpu"
    """
    f_dt = float(PML_kw("f_dt", kwargs, 0.004))
    n_steps = int(PML_kw("n_steps", kwargs, 12))
    n_bs = int(PML_kw("n_batch_size", kwargs, 2))
    n_lr = float(PML_kw("n_lr", kwargs, 1e-3))
    device = PML_kw("device", kwargs, "cpu")

    # Dataset (we directly use the dataset class with arrays)
    ds = PML_NpyPairedSeismic(a_low, a_full, f_dt=f_dt, b_fullband_only=False, b_augment_lowcut=False)
    train_loader = DataLoader(ds, batch_size=n_bs, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(ds, batch_size=n_bs, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

    # Model
    model = PML_build_spectral_model(
        n_in=7, n_out=1,
        l_n_channels=PML_kw("l_n_channels", kwargs, [32, 64]),
        n_T=int(PML_kw("n_T", kwargs, 200)),
        n_ddim_steps=int(PML_kw("n_ddim_steps", kwargs, 10)),
        f_dt=f_dt, b_use_attention=False,
    )
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=n_lr)

    # Train a few steps
    hist: List[float] = []
    for _ in range(n_steps):
        loss = PML_train_one_epoch(model, train_loader, f_dt, optim, b_amp=False, device=device, f_grad_clip=0.0)
        hist.append(loss)

    # Quick val
    val = PML_validate_epoch(model, val_loader, f_dt, device=device, f_fmax_low=min(12.0, 0.5/f_dt - 1e-6))
    return {"train_loss_hist": hist, "val": val, "model": model}


# -----------------------------
# CLI
# -----------------------------
def PML_parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("PipeML SpectralDDPM training")
    p.add_argument("--lowcut", type=str, required=True, help="Path to lowcut .npy")
    p.add_argument("--fullband", type=str, required=True, help="Path to fullband .npy")
    p.add_argument("--dt", type=float, default=0.004, help="Sample interval (s)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="runs/train_diffusion")
    p.add_argument("--use_attention", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)
    # Diffusion specifics
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"])
    p.add_argument("--fmax_target", type=float, default=12.0)
    return p.parse_args()


def main() -> None:
    args = PML_parse_args()
    PML_seed_everything(args.seed)

    # Datasets
    ds = PML_build_dataset_from_paths(
        args.lowcut, args.fullband,
        f_dt=args.dt, b_fullband_only=True, b_augment_lowcut=False,
        n_seed=args.seed, b_deterministic=True,
    )
    # simple split
    tr_idx, va_idx = PML_NpyPairedSeismic.split_indices(ds.__len__(), f_val_frac=0.1, n_seed=args.seed, b_shuffle=True)
    ds_train = torch.utils.data.Subset(ds, tr_idx.tolist())
    ds_val = torch.utils.data.Subset(ds, va_idx.tolist())
    train_loader, val_loader = PML_build_dataloaders(ds_train, ds_val, n_batch_size=args.bs, n_workers=2, b_pin_memory=True)

    # Model
    model = PML_build_spectral_model(
        n_in=7, n_out=1,
        l_n_channels=[64, 128, 256, 512],
        n_T=args.T, n_ddim_steps=args.ddim_steps,
        c_beta_schedule=args.beta_schedule,
        f_dt=args.dt, f_fmax_target=args.fmax_target,
        b_use_attention=args.use_attention,
    )
    model.to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Logging dir
    os.makedirs(args.save_dir, exist_ok=True)
    cfg_json = os.path.join(args.save_dir, "config.json")
    with open(cfg_json, "w") as f:
        json.dump({"args": vars(args)}, f, indent=2)

    # Train loop (simple)
    best = -1e9
    for ep in range(1, args.epochs + 1):
        tr_loss = PML_train_one_epoch(
            model, train_loader, args.dt, optim,
            b_amp=not args.no_amp, f_grad_clip=args.grad_clip, device=args.device
        )
        val = PML_validate_epoch(model, val_loader, args.dt, device=args.device, f_fmax_low=min(12.0, 0.5/args.dt - 1e-6))
        print(f"[Ep {ep:03d}] train_loss={tr_loss:.6f}  val_snrLF={val['snr_lf_db']:.3f} dB  val_specL2={val['spec_l2_mean']:.3e}")

        # rudimentary checkpointing on SNR
        if val["snr_lf_db"] > best:
            best = val["snr_lf_db"]
            ckpt = os.path.join(args.save_dir, "best.pt")
            torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "snr_lf_db": best}, ckpt)

    print("Training finished. Best SNR LF (dB):", best)


if __name__ == "__main__":
    main()
