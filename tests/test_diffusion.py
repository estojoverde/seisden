# tests/test_diffusion.py
import unittest
import torch

from src.models.diffusion import (
    PML_SpectralDiffusionConfig,
    PML_SpectralDDPM,
    PML_VanillaDiffusionConfig,
    PML_DDPM,
)

F_DT = 0.004  # 4 ms


class TestSpectralDiffusion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def _toy_batch(self, n_B=2, n_H=96, n_W=64):
        # synthetic: y_fullband has more LF than x_lowcut
        x_low = torch.randn(n_B, 1, n_H, n_W) * 0.1
        y_full = x_low + torch.randn(n_B, 1, n_H, n_W) * 0.05
        return x_low, y_full

    def test_q_sample_monotonic_lf_loss(self):
        cfg = PML_SpectralDiffusionConfig(
            n_T=100,
            c_schedule="linear",
            f_lf_max=12.0,
            f_transition=2.0,
            b_band_noise=False,   # deterministic for this test
            f_noise_cap=0.0,
            b_concat_t=True,
            b_append_noisy_residual=True,
            n_in_feat=6,
        )
        model = PML_SpectralDDPM(cfg)
        x_low, y_full = self._toy_batch()
        residual_true = y_full - x_low  # (B,1,H,W)
        B, _, H, W = residual_true.shape

        # early t (almost no loss) vs late t (strong LF loss)
        t_early = torch.zeros(B, dtype=torch.long)
        t_late = torch.full((B,), cfg.n_T - 1, dtype=torch.long)

        x_t_early, _ = model._q_sample(residual_true, t_early, F_DT)
        x_t_late, _ = model._q_sample(residual_true, t_late, F_DT)

        # Compare LF energy via rFFT bins ~[0, f_lf_max]
        X0 = torch.fft.rfft(residual_true, dim=-2, norm="ortho")
        Xe = torch.fft.rfft(x_t_early, dim=-2, norm="ortho")
        Xl = torch.fft.rfft(x_t_late, dim=-2, norm="ortho")

        f = torch.fft.rfftfreq(H, d=F_DT)
        lf_idx = f <= cfg.f_lf_max + 1e-6

        def band_energy(X):
            return (X[..., lf_idx, :] * X[..., lf_idx, :].conj()).real.mean()

        self.assertGreater(band_energy(Xe), band_energy(Xl))  # LF energy goes down with t

    def test_training_forward_shapes_and_grad(self):
        cfg = PML_SpectralDiffusionConfig(n_T=50, n_ddim_steps=5, f_lf_max=10.0, f_transition=2.0)
        model = PML_SpectralDDPM(cfg)
        x_low, y_full = self._toy_batch(n_B=2, n_H=64, n_W=48)
        loss = model(x_low, y_full, F_DT)
        self.assertTrue(torch.is_tensor(loss) and loss.ndim == 0)
        loss.backward()  # smoke test

    def test_ddim_sampling_shapes(self):
        cfg = PML_SpectralDiffusionConfig(n_T=50, n_ddim_steps=5, f_lf_max=8.0, f_transition=2.0, b_clip_denoised=True)
        model = PML_SpectralDDPM(cfg)
        x_low, _ = self._toy_batch(n_B=1, n_H=64, n_W=32)
        r = model.predict_residual(x_low, F_DT, n_steps=5)
        self.assertEqual(tuple(r.shape), tuple(x_low.shape))

    def test_reconstruct_fullband(self):
        cfg = PML_SpectralDiffusionConfig(n_T=20, n_ddim_steps=4, f_lf_max=6.0, f_transition=1.5)
        model = PML_SpectralDDPM(cfg)
        x_low, _ = self._toy_batch(n_B=1, n_H=48, n_W=40)
        y_hat = model.reconstruct_fullband(x_low, F_DT)
        self.assertEqual(tuple(y_hat.shape), tuple(x_low.shape))


class TestVanillaDiffusion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_vanilla_forward(self):
        cfg = PML_VanillaDiffusionConfig(n_T=10)
        # n_in_ch = features(6) + x_t(1) = 7
        model = PML_DDPM(cfg, n_in_ch=7)
        B, H, W = 2, 32, 24
        x_low = torch.randn(B, 1, H, W)
        y_full = torch.randn(B, 1, H, W)
        loss = model(x_low, y_full, F_DT)
        self.assertTrue(loss.ndim == 0)


if __name__ == "__main__":
    unittest.main()
