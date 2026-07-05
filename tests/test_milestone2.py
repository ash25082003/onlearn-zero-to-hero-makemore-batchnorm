"""Milestone 2: the saturated tanh - measure it, then fix it."""
import unittest

import torch
import torch.nn.functional as F

from mlp import init_params, saturation


class TestSaturation(unittest.TestCase):

    def test_saturation_counts(self):
        h = torch.tensor([[0.6, -0.99], [0.98, 0.1]])
        self.assertAlmostEqual(
            saturation(h), 0.5, places=6,
            msg="two of these four entries have absolute value above 0.97")
        self.assertIsInstance(saturation(h), float,
                              "return a plain Python float (.item())")

    def test_saturation_threshold_is_respected(self):
        h = torch.tensor([[0.6, -0.99], [0.98, 0.1]])
        self.assertAlmostEqual(saturation(h, threshold=0.5), 0.75, places=6,
                               msg="with threshold 0.5, three of four entries "
                                   "count as saturated")

    def test_saturation_of_all_calm_values(self):
        self.assertAlmostEqual(saturation(torch.zeros(8, 8)), 0.0, places=6)

    def test_hidden_layer_healthy_at_init(self):
        for seed in (2147483647, 42, 1337):
            g = torch.Generator().manual_seed(seed)
            C, W1, b1, W2, b2 = init_params(generator=g)
            gd = torch.Generator().manual_seed(seed + 1)
            X = torch.randint(0, 27, (256, 3), generator=gd)
            emb = C[X].view(256, -1)
            h = torch.tanh(emb @ W1 + b1)
            sat = saturation(h)
            self.assertLess(
                sat, 0.30,
                f"{sat * 100:.0f}% of tanh outputs are pinned past 0.97 at "
                f"init (seed {seed}) - the naive init saturates ~70%. Shrink "
                "W1 in init_params (the lecture eyeballs * 0.2).")

    def test_loss_still_calibrated(self):
        g = torch.Generator().manual_seed(2147483647)
        C, W1, b1, W2, b2 = init_params(generator=g)
        gd = torch.Generator().manual_seed(5)
        X = torch.randint(0, 27, (256, 3), generator=gd)
        Y = torch.randint(0, 27, (256,), generator=gd)
        emb = C[X].view(256, -1)
        logits = torch.tanh(emb @ W1 + b1) @ W2 + b2
        loss = F.cross_entropy(logits, Y).item()
        self.assertLess(loss, 3.5,
                        "fixing W1 must not undo Milestone 1 - the initial "
                        "loss should still be about 3.3")


if __name__ == "__main__":
    unittest.main()
