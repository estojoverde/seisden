import unittest
import numpy as np
import torch

from torch.utils.data import DataLoader

from src.dataset import PML_apply_lowcut_fft, PML_NpyPairedSeismic
from src.metrics import PML_snr_lowband, PML_spectral_l2_bands
from train_diffusion import (
    PML_minimal_training_run,
    PML_train_one_epoch,
    PML_validate_epoch,
    PML_build_spectral_model,
)

F_DT = 0.004  # 4 ms


def _make_tiny_pairs(H=128, W=48, f_low=10.0, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(H) * F_DT
    # gentle band-limited-ish signal with mixed tones
    base = (np.sin(2*np.pi*3.0*t) + 0.6*np.sin(2*np.pi*15.0*t + 0.3) +
            0.4*np.sin(2*np.pi*26.0*t + 1.1))
    full = np.repeat(base[:, None], W, axis=1).astype(np.float32)
    low = PML_apply_lowcut_fft(full, f_dt=F_DT, f_low=f_low, norm="ortho")
    # batch of N panels
    full = np.stack([full, full, full, full], axis=0)[:, None]  # (N,1,H,W)
    low  = np.stack([low,  low,  low,  low ], axis=0)[:, None]
    return low, full


class TestTrainIntegration(unittest.TestCase):

    def test_minimal_training_updates_and_improves_metrics(self):
        low, full = _make_tiny_pairs(H=96, W=40, f_low=10.0, seed=11)

        # Baselines before training (on first sample)
        x0 = torch.from_numpy(low[:1]).float()
        y0 = torch.from_numpy(full[:1]).float()
        snr_in = float(PML_snr_lowband(x0, y0, F_DT, f_fmax_low=10.0).item())
        err_in = float(PML_spectral_l2_bands(x0, y0, F_DT, [(0.0, 10.0)])[0, 0].item())

        # Quick training
        out = PML_minimal_training_run(
            low, full, f_dt=F_DT, n_steps=10, n_batch_size=2, n_lr=2e-3,
            l_n_channels=[16, 32], n_T=100, n_ddim_steps=5, device="cpu"
        )
        self.assertTrue(len(out["train_loss_hist"]) == 10)
        self.assertTrue(np.isfinite(out["val"]["snr_lf_db"]))

        model = out["model"].eval()

        # Reconstruct first sample
        with torch.no_grad():
            y_recon = model.reconstruct_fullband(x0, F_DT)

        # Metrics should improve over the input lowcut
        snr_out = float(PML_snr_lowband(y_recon, y0, F_DT, f_fmax_low=10.0).item())
        err_out = float(PML_spectral_l2_bands(y_recon, y0, F_DT, [(0.0, 10.0)])[0, 0].item())

        self.assertGreater(snr_out, snr_in - 1e-6)
        self.assertLess(err_out, err_in + 1e-9)

    def test_param_update_occurs_in_train_one_epoch(self):
        low, full = _make_tiny_pairs(H=64, W=32, f_low=12.0, seed=7)
        ds = PML_NpyPairedSeismic(low, full, f_dt=F_DT, b_fullband_only=False, b_augment_lowcut=False)
        loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

        model = PML_build_spectral_model(n_in=7, n_out=1, l_n_channels=[16, 32], n_T=80, n_ddim_steps=5, f_dt=F_DT, b_use_attention=False).cpu()
        optim = torch.optim.AdamW(model.parameters(), lr=2e-3)

        # snapshot of a parameter tensor
        p0 = None
        for p in model.parameters():
            if p.requires_grad and p.data.ndim > 1:
                p0 = p.data.clone()
                break
        self.assertIsNotNone(p0)

        loss = PML_train_one_epoch(model, loader, F_DT, optim, b_amp=False, device="cpu", f_grad_clip=0.0)
        self.assertTrue(np.isfinite(float(loss)))

        # check parameter changed
        changed = False
        for p in model.parameters():
            if p.requires_grad and p.data.ndim > 1:
                if not torch.allclose(p0, p.data):
                    changed = True
                    break
        self.assertTrue(changed, "Parameters did not change after one epoch")

        # quick validation path runs
        _ = PML_validate_epoch(model, loader, F_DT, device="cpu", f_fmax_low=10.0)  # just ensure no crash


if __name__ == "__main__":
    unittest.main()
