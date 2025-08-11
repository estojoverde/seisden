import unittest
import math
import numpy as np
import torch
import torch.nn as nn

from src.models.unet_blocks import PML_ResidualBlock
from src.features.fourier import (
    PML_build_fourier_feature_stack,
    PML_radial_lf_mask2d,
)
from src.dataset import PML_apply_lowcut_fft


F_DT = 0.004  # 4 ms
TORCH_DTYPE = torch.float32
DEVICE = "cpu"


class TestValueFocused(unittest.TestCase):

    def test_fft2_roundtrip_ortho(self):
        """ifft2(fft2(x, 'ortho'), 'ortho') ≈ x (value equality, not just shape)."""
        B, C, H, W = 2, 1, 64, 48
        x = torch.randn(B, C, H, W, dtype=TORCH_DTYPE, device=DEVICE)
        X = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
        x_rec = torch.fft.ifft2(X, dim=(-2, -1), norm="ortho").real
        self.assertTrue(torch.allclose(x, x_rec, atol=1e-6, rtol=1e-6))

    def test_fourier_scaling_effects(self):
        """
        Scaling x by k should scale Re/Im by k, keep angle invariant, and increase logmag.
        """
        B, C, H, W = 1, 1, 64, 64
        x = torch.randn(B, C, H, W, dtype=TORCH_DTYPE, device=DEVICE)
        k = 2.0
        fkws = dict(b_return_dict=True, b_include_time=False)  # we only need spectral channels

        d1 = PML_build_fourier_feature_stack(x, f_dt=F_DT, **fkws)
        d2 = PML_build_fourier_feature_stack(k * x, f_dt=F_DT, **fkws)

        # Real/Imag scale ~k
        self.assertIn("real", d1)
        self.assertIn("imag", d1)
        r1, i1 = d1["real"], d1["imag"]
        r2, i2 = d2["real"], d2["imag"]
        self.assertTrue(torch.allclose(r2, k * r1, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(i2, k * i1, atol=1e-5, rtol=1e-4))

        # Angle invariant to positive scalar
        a1 = d1["angle"]
        a2 = d2["angle"]
        self.assertTrue(torch.allclose(a1, a2, atol=1e-6, rtol=1e-6))

        # logmag should increase with scaling by k>1
        lm1 = d1["logmag"]
        lm2 = d2["logmag"]
        self.assertGreater(float((lm2 - lm1).mean().item()), 0.0)

    def test_radial_mask_dc_is_max(self):
        """DC (0,0) should be the maximum of the normalized radial LF mask."""
        H, W = 64, 64
        m = PML_radial_lf_mask2d(H, W, f_dt=F_DT)  # (1,H,W)
        m = m[0]  # (H,W)
        dc = m[0, 0].item()
        self.assertAlmostEqual(dc, float(m.max().item()), places=7)
        # Range strictly within [0,1]
        self.assertGreaterEqual(m.min().item(), 0.0)
        self.assertLessEqual(m.max().item(), 1.0)

    def test_lowcut_selectivity_two_tones(self):
        """
        Two-tone signal (2 Hz + 20 Hz). After lowcut f_low=10 Hz:
        - 2 Hz component ≪ 20 Hz component in the filtered signal.
        """
        H, W = 512, 4
        t = np.arange(H) * F_DT
        f1, f2 = 2.0, 20.0
        sig = np.sin(2 * np.pi * f1 * t)[:, None] + np.sin(2 * np.pi * f2 * t)[:, None]  # (H,1)
        panel = np.repeat(sig, W, axis=1).astype(np.float32, copy=False)  # (H,W)

        filt = PML_apply_lowcut_fft(panel, f_dt=F_DT, f_low=10.0, norm="ortho")
        X = np.fft.rfft(filt, axis=0, norm="ortho")  # (H//2+1,W)
        freqs = np.fft.rfftfreq(H, d=F_DT)

        # Find nearest bins
        k1 = int(np.argmin(np.abs(freqs - f1)))
        k2 = int(np.argmin(np.abs(freqs - f2)))

        a1 = np.abs(X[k1]).mean()
        a2 = np.abs(X[k2]).mean()

        # 2 Hz should be strongly attenuated
        self.assertLess(a1, 0.1 * a2)

    def test_residual_block_film_identity_equivalence(self):
        """
        When FiLM gamma=1, beta=0, the block output equals the no-FiLM output (value equality).
        """
        B, C, H, W = 2, 8, 31, 29
        x = torch.randn(B, C, H, W, dtype=TORCH_DTYPE)
        blk = PML_ResidualBlock(C, C)

        y_no = blk(x.clone())
        y_id = blk(x.clone(), gamma=torch.ones(C), beta=torch.zeros(C))
        self.assertTrue(torch.allclose(y_no, y_id, atol=1e-6, rtol=1e-6))

    def test_residual_block_film_beta2_exact_injection(self):
        """
        Make the block algebra predictable to check exact FiLM effect:
        - Zero the conv weights (so conv outputs are 0).
        - Replace activations with Identity.
        - Feed x=0 so residual=0.
        Then: output should equal beta2 broadcast exactly.
        """
        B, C, H, W = 1, 6, 17, 13
        x = torch.zeros(B, C, H, W, dtype=TORCH_DTYPE)

        blk = PML_ResidualBlock(C, C)
        # Zero conv weights
        with torch.no_grad():
            blk.conv1.weight.zero_()
            blk.conv2.weight.zero_()
        # Identity activations
        blk.act1 = nn.Identity()
        blk.act2 = nn.Identity()

        # Provide only beta2; conv2->GN2 yields 0, FiLM2 adds beta2, residual is 0 -> y == beta2
        beta2 = torch.linspace(-0.5, 0.5, C)  # (C,)
        y = blk(x, beta2=beta2)

        expected = beta2.view(1, C, 1, 1).expand_as(y)
        self.assertTrue(torch.allclose(y, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
