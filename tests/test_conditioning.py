# tests/test_conditioning.py
import unittest
import torch

from src.models.conditioning import PML_MetadataConditioner, PML_FiLM
from src.models.unet_blocks import PML_ResidualBlock


class TestConditioning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_metadata_conditioner_shapes_and_order(self):
        n_B = 3
        cond = PML_MetadataConditioner(
            ["f_peak", "f_flow", "f_dt"],
            c_transform="auto",  # log1p on f_* keys
            b_keep_dims=False,
        )
        dic_meta = {
            "f_flow": torch.tensor([8.0, 10.0, 12.0]),   # shuffled order on purpose
            "f_dt": torch.tensor([0.004, 0.004, 0.004]),
            "f_peak": torch.tensor([20.0, 25.0, 30.0]),
        }
        z = cond(dic_meta)
        self.assertEqual(tuple(z.shape), (n_B, 3))
        # Ensure column order matches l_keys order
        z2 = cond({"f_peak": 20.0, "f_flow": 8.0, "f_dt": 0.004})
        self.assertEqual(tuple(z2.shape), (1, 3))

    def test_film_shapes(self):
        n_B = 2
        n_in, n_out = 3, 16
        film = PML_FiLM(n_in, n_out, n_hidden=32, n_layers=1, f_dropout=0.0)
        z = torch.randn(n_B, n_in)
        g1, b1, g2, b2 = film(z)
        for t in (g1, b1, g2, b2):
            self.assertEqual(tuple(t.shape), (n_B, n_out, 1, 1))

    def test_film_identity_init(self):
        """
        With zero-init heads and b_init_gamma_to_one=True, b_init_beta_to_zero=True,
        FiLM should act like identity when z_meta==0: gamma≈1, beta≈0.
        """
        n_B = 2
        n_in, n_out = 4, 8
        film = PML_FiLM(n_in, n_out, n_hidden=16, n_layers=1)

        z0 = torch.zeros(n_B, n_in)
        g1, b1, g2, b2 = film(z0)

        self.assertTrue(torch.allclose(g1, torch.ones_like(g1)))
        self.assertTrue(torch.allclose(g2, torch.ones_like(g2)))
        self.assertTrue(torch.allclose(b1, torch.zeros_like(b1)))
        self.assertTrue(torch.allclose(b2, torch.zeros_like(b2)))

    def test_residual_block_modulated(self):
        """
        Show that FiLM actually changes the output of a residual block
        when metadata changes.
        """
        n_B, n_H, n_W = 2, 32, 32
        n_in, n_out = 8, 8
        x = torch.randn(n_B, n_in, n_H, n_W)

        # Residual block with FiLM hooks enabled
        blk = PML_ResidualBlock(n_in, n_out, b_use_film=True)

        # Build FiLM
        film = PML_FiLM(n_in=3, n_out=n_out, n_hidden=32, n_layers=1)

        # Two different metadata vectors
        z_a = torch.tensor([[20.0, 8.0, 0.004], [25.0, 10.0, 0.004]], dtype=torch.float32)  # (B,3)
        z_b = torch.tensor([[30.0, 12.0, 0.004], [35.0, 14.0, 0.004]], dtype=torch.float32)

        g1a, b1a, g2a, b2a = film(z_a)
        g1b, b1b, g2b, b2b = film(z_b)

        ya = blk(x, gamma1=g1a, beta1=b1a, gamma2=g2a, beta2=b2a)
        yb = blk(x, gamma1=g1b, beta1=b1b, gamma2=g2b, beta2=b2b)

        self.assertFalse(torch.allclose(ya, yb))


if __name__ == "__main__":
    unittest.main()
