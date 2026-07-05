"""Milestone 3: kaiming init - the principled scale replaces the magic number."""
import unittest

import torch

from mlp import init_params, kaiming_scale


class TestKaiming(unittest.TestCase):

    def test_kaiming_scale_values(self):
        self.assertAlmostEqual(kaiming_scale(9), (5 / 3) / 3, places=6,
                               msg="kaiming_scale(fan_in) = gain / sqrt(fan_in), "
                                   "default gain 5/3")
        self.assertAlmostEqual(kaiming_scale(30), 0.3042903, places=5)

    def test_kaiming_scale_gain_is_respected(self):
        self.assertAlmostEqual(kaiming_scale(25, gain=1.0), 0.2, places=6,
                               msg="the gain must be a parameter, not hard-coded")
        self.assertAlmostEqual(kaiming_scale(100, gain=2.0), 0.2, places=6)

    def test_W1_uses_kaiming(self):
        params = init_params(generator=torch.Generator().manual_seed(7))
        expected = kaiming_scale(30)
        std = params[1].std().item()
        self.assertTrue(
            0.9 * expected < std < 1.1 * expected,
            f"W1 std is {std:.4f}, expected about {expected:.4f} = "
            "(5/3)/sqrt(30). Replace the eyeballed factor with "
            "kaiming_scale(block_size * emb_dim).")

    def test_W1_kaiming_tracks_fan_in(self):
        params = init_params(block_size=5, emb_dim=4, hidden=64,
                             generator=torch.Generator().manual_seed(8))
        expected = (5 / 3) / 20 ** 0.5
        std = params[1].std().item()
        self.assertTrue(
            0.9 * expected < std < 1.1 * expected,
            f"W1 std is {std:.4f} for fan_in 20, expected about {expected:.4f} "
            "- the scale must follow block_size * emb_dim, not stay 30.")


if __name__ == "__main__":
    unittest.main()
