# tests/test_unet_blocks.py
import unittest
import torch

from src.models.unet_blocks import (
    PML_ResidualBlock,
    PML_AttentionBlock,
    PML_UNet,
)


class TestUNetBlocks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_residual_block_shapes_and_grad(self):
        n_B, n_H, n_W = 2, 64, 64
        n_in, n_out = 8, 16
        x = torch.randn(n_B, n_in, n_H, n_W, requires_grad=True)
        blk = PML_ResidualBlock(n_in, n_out, n_gn_groups=8, f_dropout=0.0)

        y = blk(x)
        self.assertEqual(tuple(y.shape), (n_B, n_out, n_H, n_W))
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    def test_attention_block_shape(self):
        n_B, n_H, n_W = 1, 32, 48
        n_c = 32
        x = torch.randn(n_B, n_c, n_H, n_W)
        attn = PML_AttentionBlock(n_c, n_heads=4)
        y = attn(x)
        self.assertEqual(tuple(y.shape), (n_B, n_c, n_H, n_W))

    def test_unet_forward_default(self):
        n_B, n_H, n_W = 2, 96, 96
        n_in = 6   # from features default: 6 channels
        n_out = 1
        x = torch.randn(n_B, n_in, n_H, n_W)

        net = PML_UNet(
            n_in, n_out,
            l_widths=[64, 128, 256, 512],
            n_blocks_per_res=2,
            b_use_attention=True,
            n_attn_heads=4,
            n_gn_groups=8,
            f_dropout=0.0,
        )
        y = net(x)
        self.assertEqual(tuple(y.shape), (n_B, n_out, n_H, n_W))

    def test_unet_no_attention_and_odd_sizes(self):
        n_B, n_H, n_W = 1, 97, 95  # odd sizes to exercise center-crop for skips
        x = torch.randn(n_B, 4, n_H, n_W)

        net = PML_UNet(
            4, 2,
            l_widths=[32, 64, 128],
            n_blocks_per_res=1,
            b_use_attention=False,  # disable mid-attn
            n_gn_groups=8,
        )
        y = net(x)
        self.assertEqual(tuple(y.shape), (n_B, 2, n_H, n_W))

    def test_unet_groupnorm_divisibility(self):
        # widths not divisible by requested groups -> fallback to valid GN groups
        n_B, n_H, n_W = 1, 64, 64
        x = torch.randn(n_B, 3, n_H, n_W)
        net = PML_UNet(
            3, 1,
            l_widths=[30, 50],  # awkward channel counts
            n_blocks_per_res=1,
            n_gn_groups=16,     # too many relative to channels
        )
        y = net(x)
        self.assertEqual(tuple(y.shape), (n_B, 1, n_H, n_W))

    def test_no_inplace(self):
        n_B, n_H, n_W = 1, 32, 32
        x = torch.randn(n_B, 6, n_H, n_W, requires_grad=True)
        x_clone = x.detach().clone()
        net = PML_UNet(6, 1)
        y = net(x)
        self.assertTrue(torch.allclose(x.detach(), x_clone, atol=0.0, rtol=0.0))
        y.sum().backward()
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
