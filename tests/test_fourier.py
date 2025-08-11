import unittest
import torch

from src.features.fourier import (
    PML_build_fourier_feature_stack,
    PML_radial_lf_mask2d,
    PML_plot_mask,
    PML_plot_mask_overlay,
)

F_DT = 0.004


class TestFourierFeatures(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def _toy(self, n_B=2, n_H=64, n_W=48):
        x = torch.randn(n_B, 1, n_H, n_W, requires_grad=True)
        return x

    def test_stack_default_channels(self):
        x = self._toy()
        s = PML_build_fourier_feature_stack(x, F_DT)
        self.assertEqual(s.shape[1], 6)   # [time, real, imag, angle, logmag, lf_mask]
        self.assertEqual(tuple(s.shape[:2]), (x.shape[0], 6))

    def test_stack_dict_keys_and_shapes(self):
        x = self._toy(n_B=1, n_H=32, n_W=24)
        d = PML_build_fourier_feature_stack(x, F_DT, b_return_dict=True)
        for k in ["x_time", "real", "imag", "angle", "logmag", "lf_mask"]:
            self.assertIn(k, d)
            self.assertEqual(tuple(d[k].shape), (1, 1, 32, 24))

    def test_toggle_channels(self):
        x = self._toy()
        s = PML_build_fourier_feature_stack(
            x, F_DT,
            b_include_time=False,
            b_include_angle=False,
            b_include_lfmask=False,
        )
        # left with: real, imag, logmag = 3
        self.assertEqual(s.shape[1], 3)

    def test_angle_scaling_toggle(self):
        x = self._toy(n_B=1, n_H=16, n_W=12)
        d1 = PML_build_fourier_feature_stack(x, F_DT, b_return_dict=True, b_angle_scale=True)
        d2 = PML_build_fourier_feature_stack(x, F_DT, b_return_dict=True, b_angle_scale=False)
        a1 = d1["angle"]
        a2 = d2["angle"]
        self.assertTrue(a1.min().item() >= -1.0000001 and a1.max().item() <= 1.0000001)
        self.assertTrue(a2.min().item() >= -torch.pi.item() - 1e-6 and a2.max().item() <= torch.pi.item() + 1e-6)

    def test_mask_shape_and_monotonicity(self):
        n_H, n_W = 64, 48
        m = PML_radial_lf_mask2d(n_H, n_W, F_DT)
        self.assertEqual(tuple(m.shape), (1, n_H, n_W))
        # Highest near DC (index 0,0 for unshifted FFT), smaller away
        self.assertGreater(m[0, 0, 0].item(), m[0, n_H // 2, n_W // 2].item())
        self.assertGreaterEqual(m.min().item(), 0.0)
        self.assertLessEqual(m.max().item(), 1.0)

    def test_mask_kwargs_f_d_spatial_invariant_under_axis_normalization(self):
        n_H, n_W = 64, 48
        m1 = PML_radial_lf_mask2d(n_H, n_W, F_DT, f_d_spatial=1.0)
        m2 = PML_radial_lf_mask2d(n_H, n_W, F_DT, f_d_spatial=2.0)
        # Nyquist-normalization makes the radial shape invariant to f_d_spatial scale
        self.assertTrue(torch.allclose(m1, m2, atol=1e-6, rtol=1e-6))

    def test_no_inplace_and_grad_flow(self):
        x = self._toy(n_B=1, n_H=16, n_W=12)
        s = PML_build_fourier_feature_stack(x, F_DT)
        loss = s.mean()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(tuple(x.grad.shape), tuple(x.shape))

    def test_plotting_helpers_return_figures(self):
        import matplotlib.figure
        n_H, n_W = 32, 20
        m = PML_radial_lf_mask2d(n_H, n_W, F_DT)
        fig1 = PML_plot_mask(m)
        self.assertIsInstance(fig1, matplotlib.figure.Figure)

        d = torch.randn(1, 1, n_H, n_W)
        fig2 = PML_plot_mask_overlay(d, m, f_alpha=0.2)
        self.assertIsInstance(fig2, matplotlib.figure.Figure)


if __name__ == "__main__":
    unittest.main()
