import unittest
import os
import tempfile
import numpy as np
import torch

from src.visualization import (
    PML_plot_time_section,
    PML_plot_mean_spectrum,
    PML_plot_mean_spectrum_compare,
    PML_save_examples_grid,
)
from src.dataset import PML_apply_lowcut_fft

F_DT = 0.004  # 4 ms


def _two_tone_panel(H=512, W=8, f1=2.0, f2=24.0, amp2=0.7, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(H) * F_DT
    sig = np.sin(2*np.pi*f1*t)[:, None] + amp2*np.sin(2*np.pi*f2*t + rng.rand()*2*np.pi)[:, None]
    return np.repeat(sig, W, axis=1).astype(np.float32)


class TestVisualization(unittest.TestCase):

    def test_time_section_returns_fig_axes_and_saves(self):
        H, W = 128, 16
        x = np.random.randn(H, W).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "sec.png")
            fig, ax = PML_plot_time_section(x, F_DT, c_title="Demo", c_savepath=p, f_pmin=5.0, f_pmax=95.0)
            self.assertTrue(hasattr(fig, "savefig"))
            self.assertTrue(hasattr(ax, "imshow"))
            self.assertTrue(os.path.isfile(p))

    def test_mean_spectrum_value_behavior(self):
        H, W = 512, 8
        full = _two_tone_panel(H, W, f1=3.0, f2=26.0, amp2=0.8, seed=3)
        low = PML_apply_lowcut_fft(full, f_dt=F_DT, f_low=10.0, norm="ortho")

        fig, ax, (f_full, a_full) = PML_plot_mean_spectrum(full, F_DT, c_label="full")
        fig2, ax2, (f_low, a_low) = PML_plot_mean_spectrum(low, F_DT, c_label="low")
        # Compare low frequency bin magnitude
        k3 = int(np.argmin(np.abs(f_full - 3.0)))
        self.assertGreater(a_full[k3], a_low[k3])
        # High freq bin (26 Hz) should be similar
        k26 = int(np.argmin(np.abs(f_full - 26.0)))
        self.assertLess(abs(a_full[k26] - a_low[k26]), 1e-3)

    def test_spectrum_compare_multiple_and_bands(self):
        H, W = 256, 6
        full = _two_tone_panel(H, W, seed=5)
        low = PML_apply_lowcut_fft(full, f_dt=F_DT, f_low=8.0, norm="ortho")
        fig, ax, outs = PML_plot_mean_spectrum_compare(
            [full, low], F_DT, l_labels=["full", "low"], l_bands=[(0.0, 8.0), (8.0, 16.0)]
        )
        self.assertEqual(len(outs), 2)
        f0, a0 = outs[0]
        f1, a1 = outs[1]
        self.assertTrue(np.allclose(f0, f1))  # same freq axis

    def test_save_examples_grid_creates_files_and_error_has_energy(self):
        H, W = 384, 5
        full = _two_tone_panel(H, W, seed=11)
        low = PML_apply_lowcut_fft(full, f_dt=F_DT, f_low=12.0, norm="ortho")
        recon = low + (full - low)  # perfect recon

        x = torch.from_numpy(low)[None, None]
        y = torch.from_numpy(full)[None, None]
        z = torch.from_numpy(recon)[None, None]

        with tempfile.TemporaryDirectory() as td:
            paths = PML_save_examples_grid(x, y, z, F_DT, c_dir=td, n_max=1, c_prefix="eg")
            self.assertEqual(len(paths), 1)
            self.assertTrue(os.path.isfile(paths[0]))

        # Now suboptimal recon -> error nonzero
        z_bad = torch.from_numpy(low)[None, None]
        e = (z_bad - y).abs().mean().item()
        self.assertGreater(e, 1e-6)
