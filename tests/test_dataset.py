# tests/test_dataset.py
import os
import tempfile
import unittest
import numpy as np
import torch

from src.dataset import (
    PML_NpyPairedSeismic,
    PML_apply_lowcut_fft,
    PML_rfftfreq,
)
from src.templates import PML_BasicDataset  # verify subclassing

DT = 0.004  # 4 ms sample interval
FS = 1.0 / DT
NYQ = 0.5 * FS


def synth_fullband_batch(N=10, H=128, W=32, seed=123):
    """
    Build a synthetic full-band batch: sum of low and mid frequencies.
    Returns arrays shaped (N,1,H,W) float32.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(H) * DT  # seconds
    f_low = 5.0   # Hz
    f_mid = 20.0  # Hz
    batch = []
    for _ in range(N):
        phase1 = rng.uniform(0, 2*np.pi, size=(W,))
        phase2 = rng.uniform(0, 2*np.pi, size=(W,))
        amp1 = 1.0 + 0.1 * rng.randn(W)
        amp2 = 0.7 + 0.1 * rng.randn(W)
        panel = np.zeros((H, W), dtype=np.float32)
        for w in range(W):
            panel[:, w] = (
                amp1[w] * np.sin(2*np.pi*f_low*t + phase1[w]) +
                amp2[w] * np.sin(2*np.pi*f_mid*t + phase2[w])
            )
        panel += 0.01 * rng.randn(H, W).astype(np.float32)
        batch.append(panel[None, ...])  # (1,H,W)
    X = np.stack(batch, axis=0).astype(np.float32)  # (N,1,H,W)
    return X


def lowband_energy(panel_hw: np.ndarray, fmax: float) -> float:
    """
    Average magnitude in [0, fmax] using rFFT along time.
    """
    H, W = panel_hw.shape
    X = np.fft.rfft(panel_hw, axis=0, norm="ortho")
    freqs = np.fft.rfftfreq(H, d=DT)
    band = freqs <= fmax
    mag = np.abs(X[band, :])
    return float(mag.mean())


class TestDataset(unittest.TestCase):

    def test_is_subclass(self):
        self.assertTrue(issubclass(PML_NpyPairedSeismic, PML_BasicDataset))

    def test_rfftfreq(self):
        H = 128
        freqs = PML_rfftfreq(H, d=DT)
        self.assertAlmostEqual(freqs[0], 0.0, places=12)
        self.assertAlmostEqual(freqs[-1], NYQ, places=6)
        self.assertEqual(freqs.shape[0], H // 2 + 1)

    def test_apply_lowcut_fft_roundtrip(self):
        H, W = 128, 16
        rng = np.random.RandomState(0)
        panel = rng.randn(H, W).astype(np.float32)

        # f_low = 0 -> no change
        out0 = PML_apply_lowcut_fft(panel, dt=DT, f_low=0.0)
        np.testing.assert_allclose(out0, panel, rtol=1e-5, atol=1e-5)

        # Very high cutoff -> near zeros
        out_hi = PML_apply_lowcut_fft(panel, dt=DT, f_low=NYQ + 1.0)
        self.assertLess(np.abs(out_hi).mean(), 1e-6)

    def test_load_from_arrays(self):
        N, H, W = 6, 64, 32
        full = synth_fullband_batch(N=N, H=H, W=W, seed=1)
        low = full.copy()

        ds = PML_NpyPairedSeismic(low, full, dt=DT, augment_lowcut=False)
        self.assertEqual(len(ds), N)
        x, y = ds[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertEqual(tuple(x.shape), (1, H, W))
        self.assertEqual(tuple(y.shape), (1, H, W))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)

    def test_load_from_paths(self):
        N, H, W = 4, 32, 16
        full = synth_fullband_batch(N=N, H=H, W=W, seed=7)
        low = full.copy()

        with tempfile.TemporaryDirectory() as td:
            p_low = os.path.join(td, "low.npy")
            p_full = os.path.join(td, "full.npy")
            np.save(p_low, low)
            np.save(p_full, full)

            ds = PML_NpyPairedSeismic(p_low, p_full, dt=DT, augment_lowcut=False)
            self.assertEqual(len(ds), N)
            x, y = ds[1]
            self.assertEqual(tuple(x.shape), (1, H, W))
            self.assertEqual(tuple(y.shape), (1, H, W))

    def test_deterministic_augmentation(self):
        '''
        Validate two things:
        1) Deterministic augmentation yields the same sample on repeated calls.
        2) Different fixed cutoffs produce different augmented panels when the data
        contains energy near the cutoff (we add a 10 Hz component).
        '''
        N, H, W = 3, 128, 16
        rng = np.random.RandomState(11)
        t = np.arange(H) * DT

        # Build full-band with 5 Hz, 10 Hz, and 20 Hz components
        full = np.zeros((N, 1, H, W), dtype=np.float32)
        for i in range(N):
            phase5 = rng.uniform(0, 2*np.pi, size=(W,))
            phase10 = rng.uniform(0, 2*np.pi, size=(W,))
            phase20 = rng.uniform(0, 2*np.pi, size=(W,))
            amp5 = 1.0 + 0.1 * rng.randn(W)
            amp10 = 0.6 + 0.1 * rng.randn(W)
            amp20 = 0.7 + 0.1 * rng.randn(W)
            for w in range(W):
                panel = (
                    amp5[w]  * np.sin(2*np.pi*5.0  * t + phase5[w]) +
                    amp10[w] * np.sin(2*np.pi*10.0 * t + phase10[w]) +
                    amp20[w] * np.sin(2*np.pi*20.0 * t + phase20[w])
                )
                full[i, 0, :, w] = panel.astype(np.float32)
        low = full.copy()  # baseline

        # (1) Deterministic: same f_low and seed → identical outputs for same index
        ds_det = PML_NpyPairedSeismic(
            low, full, dt=DT,
            augment_lowcut=True,
            aug_params={"p": 1.0, "f_low_range": (9.0, 9.0), "source": "fullband"},
            seed=123, deterministic=True
        )
        x_a1, _ = ds_det[0]
        x_a2, _ = ds_det[0]
        np.testing.assert_allclose(x_a1.numpy(), x_a2.numpy(), rtol=0, atol=0)

        # (2) Different fixed cutoffs: 9 Hz vs 12 Hz must differ because of 10 Hz energy
        ds_cut12 = PML_NpyPairedSeismic(
            low, full, dt=DT,
            augment_lowcut=True,
            aug_params={"p": 1.0, "f_low_range": (12.0, 12.0), "source": "fullband"},
            seed=999, deterministic=True
        )
        x_b, _ = ds_cut12[0]
        self.assertFalse(np.allclose(x_a1.numpy(), x_b.numpy()))


    def test_lowband_energy_reduced_by_augmentation(self):
        """
        If baseline lowcut == fullband, and we augment with f_low in ~[10,12] Hz,
        then lowband energy up to 8 Hz should be reduced in the augmented x.
        """
        N, H, W = 2, 128, 24
        full = synth_fullband_batch(N=N, H=H, W=W, seed=5)
        low = full.copy()

        ds = PML_NpyPairedSeismic(
            low, full, dt=DT,
            augment_lowcut=True,
            aug_params={"p": 1.0, "f_low_range": (10.0, 12.0), "source": "fullband"},
            seed=777, deterministic=True
        )
        x_aug, y = ds[0]
        x_base = low[0, 0]  # (H,W)
        x_aug_hw = x_aug.numpy()[0]

        def lowband_energy(panel_hw, fmax):
            Hh, _ = panel_hw.shape
            X = np.fft.rfft(panel_hw, axis=0, norm="ortho")
            freqs = np.fft.rfftfreq(Hh, d=DT)
            band = freqs <= fmax
            mag = np.abs(X[band, :])
            return float(mag.mean())

        e_base = lowband_energy(x_base, fmax=8.0)
        e_aug = lowband_energy(x_aug_hw, fmax=8.0)
        self.assertLess(e_aug, e_base)

    def test_split_indices(self):
        train_idx, val_idx = PML_NpyPairedSeismic.split_indices(
            n=100, val_frac=0.2, seed=42, shuffle=True
        )
        self.assertEqual(len(val_idx), 20)
        self.assertEqual(len(train_idx), 80)
        self.assertEqual(len(set(train_idx) & set(val_idx)), 0)
        self.assertEqual(len(set(train_idx) | set(val_idx)), 100)
        
    def test_fullband_only_mode_returns_identical_pairs(self):
        N, H, W = 4, 32, 24
        full = np.random.randn(N, 1, H, W).astype(np.float32)
        # lowcut can be None in fullband-only mode
        ds = PML_NpyPairedSeismic(
            lowcut=None,
            fullband=full,
            b_input_is_fullband=True,  # NEW MODE
            b_augment_lowcut=True,     # should be ignored in this mode
            dic_aug_params={"f_p": 1.0, "tf_f_low_range": (8.0, 12.0)},
        )
        x, y = ds[0]
        assert x.shape == (1, H, W) and y.shape == (1, H, W)
        # x must equal y exactly (no corruption here)
        assert torch.allclose(x, y)



if __name__ == "__main__":
    unittest.main()
