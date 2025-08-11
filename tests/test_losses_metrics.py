import unittest
import numpy as np
import torch

from src.dataset import PML_apply_lowcut_fft
from src.metrics import (
    PML_compute_freqs_1d,
    PML_build_band_mask,
    PML_spectral_l2_bands,
    PML_snr_lowband,
)
from src.losses import (
    PML_spectral_l2_per_band,
    PML_lowband_weighted_loss,
)

F_DT = 0.004  # 4 ms
TORCH_DTYPE = torch.float32
DEVICE = "cpu"


def _panel_two_tone(H=1024, W=6, f1=3.0, f2=30.0, amp2=0.7, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(H) * F_DT
    sig = np.sin(2*np.pi*f1*t)[:, None] + amp2*np.sin(2*np.pi*f2*t + rng.rand()*2*np.pi)[:, None]
    return np.repeat(sig, W, axis=1).astype(np.float32)


class TestLossesMetricsExtra(unittest.TestCase):

    def test_highband_energy_conservation_after_lowcut(self):
        """
        After brick-wall lowcut at f_low, the rFFT coefficients ABOVE f_low should match original.
        We compare energy error on bins strictly > f_low (allow tiny numerical drift).
        """
        H, W = 1024, 8
        f_low = 10.0
        panel = _panel_two_tone(H, W, f1=3.0, f2=25.0, amp2=0.8, seed=1)
        filt = PML_apply_lowcut_fft(panel, f_dt=F_DT, f_low=f_low, norm="ortho")

        X0 = np.fft.rfft(panel, axis=0, norm="ortho")
        X1 = np.fft.rfft(filt, axis=0, norm="ortho")
        freqs = np.fft.rfftfreq(H, d=F_DT)
        m_hi = freqs > f_low

        # energy difference above f_low should be nearly zero
        err = np.mean(np.abs(X1[m_hi] - X0[m_hi])**2)
        self.assertLess(err, 1e-10)

    def test_band_layout_no_overlap_full_coverage(self):
        """
        Build 3 disjoint bands that exactly partition [0, Nyquist].
        Masks should be disjoint and their OR should cover all bins.
        """
        H = 480
        fnyq = 0.5 / F_DT
        bands = [(0.0, 10.0), (10.0, 20.0), (20.0, fnyq)]
        masks = [PML_build_band_mask(H, F_DT, b) for b in bands]

        # no overlaps: pairwise AND should be all False
        for i in range(len(masks)):
            for j in range(i+1, len(masks)):
                self.assertFalse(bool(torch.any(masks[i] & masks[j])))

        # full coverage: OR across all masks is True for every bin
        cover = masks[0]
        for m in masks[1:]:
            cover = cover | m
        self.assertTrue(bool(cover.all()))

    def test_weighted_additivity_over_arbitrary_partitions(self):
        """
        The global mean spectral L2 equals the weighted sum of per-band means,
        with weights proportional to number of freq bins in each band.
        """
        H, W = 512, 5
        # Make two random panels to compare
        rng = np.random.RandomState(5)
        a = rng.randn(H, W).astype(np.float32)
        b = rng.randn(H, W).astype(np.float32)

        Ya = np.fft.rfft(a, axis=0, norm="ortho")
        Yb = np.fft.rfft(b, axis=0, norm="ortho")
        pwr = np.abs(Ya - Yb)**2  # (Hf, W)
        global_mean = pwr.mean()   # average over all freq bins and W

        # Arbitrary partition points (sorted, unique, within [0, Nyq])
        fnyq = 0.5 / F_DT
        splits = [0.0, 7.5, 13.25, 19.0, 28.0, fnyq]
        bands = list(zip(splits[:-1], splits[1:]))

        # Torch path (mirrors training code)
        A = torch.from_numpy(a)[None, None]
        B = torch.from_numpy(b)[None, None]
        per_band = PML_spectral_l2_bands(A, B, F_DT, bands)  # (1, n_bands)

        Hf = H//2 + 1
        freqs = np.fft.rfftfreq(H, d=F_DT)
        weights = []
        for lo, hi in bands:
            m = (freqs >= lo) & (freqs < min(hi, fnyq + 1e-12))
            if abs(hi - fnyq) < 1e-9:
                m = (freqs >= lo) & (freqs <= hi + 1e-12)
            weights.append(m.sum() / Hf)

        weighted_sum = float((per_band[0] * torch.tensor(weights)).sum().item())
        self.assertAlmostEqual(global_mean, weighted_sum, places=6)

    def test_snr_lowband_scale_invariance(self):
        """
        Scaling both y_hat and y_true by the same positive factor should not change SNR (dB).
        """
        H, W = 512, 6
        full = _panel_two_tone(H, W, seed=9)
        low = PML_apply_lowcut_fft(full, f_dt=F_DT, f_low=12.0, norm="ortho")
        y_true = torch.from_numpy(full)[None, None].float()
        y_hat = torch.from_numpy(low)[None, None].float()

        s1 = PML_snr_lowband(y_hat, y_true, F_DT, f_fmax_low=12.0)
        k = 3.7
        s2 = PML_snr_lowband(k*y_hat, k*y_true, F_DT, f_fmax_low=12.0)
        self.assertTrue(torch.allclose(s1, s2, atol=1e-6))

    def test_lowband_weighted_loss_weights_identity(self):
        """
        If f_lambda_spec=0, loss == pure time loss; if f_lambda_time=0, loss == spectral loss.
        """
        H, W = 384, 6
        full = _panel_two_tone(H, W, seed=11)
        low = PML_apply_lowcut_fft(full, f_dt=F_DT, f_low=10.0, norm="ortho")
        x_low = torch.from_numpy(low)[None, None].float()
        y_full = torch.from_numpy(full)[None, None].float()
        residual_true = y_full - x_low

        # A wrong residual for contrast
        wrong = 0.5 * residual_true

        # Time-only
        loss_time_only = PML_lowband_weighted_loss(
            wrong, residual_true, x_low, y_full, F_DT,
            c_time="l1", f_lambda_time=1.0, f_lambda_spec=0.0,
            l_bands=[(0.0, 10.0), (10.0, 20.0)]
        )
        manual_time = (wrong - residual_true).abs().mean()
        self.assertTrue(torch.allclose(loss_time_only, manual_time, atol=1e-12))

        # Spec-only (weighted by bands; just check it matches helper)
        loss_spec_only = PML_lowband_weighted_loss(
            wrong, residual_true, x_low, y_full, F_DT,
            c_time="l1", f_lambda_time=0.0, f_lambda_spec=1.0,
            l_bands=[(0.0, 10.0), (10.0, 20.0)], f_lowband_boost=2.0
        )
        spec_pb = PML_spectral_l2_per_band(x_low + wrong, y_full, F_DT, [(0.0, 10.0), (10.0, 20.0)])
        spec_pb = spec_pb.clone()
        spec_pb[:, 0] = spec_pb[:, 0] * 2.0  # low-band boost
        manual_spec = spec_pb.mean()
        self.assertTrue(torch.allclose(loss_spec_only, manual_spec, atol=1e-12))

    def test_band_layouts_do_not_change_total_error(self):
        """
        Sum of weighted per-band errors is invariant to how we partition the spectrum.
        Compare fine partition vs coarse partition.
        """
        H, W = 640, 4
        rng = np.random.RandomState(123)
        a = rng.randn(H, W).astype(np.float32)
        b = rng.randn(H, W).astype(np.float32)

        A = torch.from_numpy(a)[None, None]
        B = torch.from_numpy(b)[None, None]
        fnyq = 0.5 / F_DT

        # coarse: 3 bands
        coarse = [(0.0, 10.0), (10.0, 20.0), (20.0, fnyq)]
        # fine: 12 equal bands
        fine_edges = np.linspace(0.0, fnyq, 13).tolist()
        fine = list(zip(fine_edges[:-1], fine_edges[1:]))

        def total_weighted(per_band, H):
            Hf = H//2 + 1
            freqs = np.fft.rfftfreq(H, d=F_DT)
            weights = []
            for lo, hi in per_band[1]:
                m = (freqs >= lo) & (freqs < min(hi, fnyq + 1e-12))
                if abs(hi - fnyq) < 1e-9:
                    m = (freqs >= lo) & (freqs <= hi + 1e-12)
                weights.append(m.sum() / Hf)
            return float((per_band[0] * torch.tensor(weights)).sum().item())

        pb_coarse = (PML_spectral_l2_bands(A, B, F_DT, coarse), coarse)
        pb_fine   = (PML_spectral_l2_bands(A, B, F_DT, fine), fine)

        self.assertAlmostEqual(total_weighted(pb_coarse, H), total_weighted(pb_fine, H), places=6)


if __name__ == "__main__":
    unittest.main()
