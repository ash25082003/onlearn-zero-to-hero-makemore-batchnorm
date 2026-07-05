"""Milestone 1: fix the confidently-wrong init (the loss that should be 3.3)."""
import unittest

import torch
import torch.nn.functional as F

from mlp import init_params


def flat_forward(X, params):
    C, W1, b1, W2, b2 = params
    emb = C[X]
    h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
    return h @ W2 + b2


def init_loss(seed, **kwargs):
    g = torch.Generator().manual_seed(seed)
    params = init_params(generator=g, **kwargs)
    gd = torch.Generator().manual_seed(seed + 1)
    X = torch.randint(0, 27, (256, 3), generator=gd)
    Y = torch.randint(0, 27, (256,), generator=gd)
    with torch.no_grad():
        return F.cross_entropy(flat_forward(X, params), Y).item()


class TestInitFix(unittest.TestCase):

    def test_shapes_and_grad(self):
        params = init_params(generator=torch.Generator().manual_seed(1))
        self.assertEqual(len(params), 5, "expected [C, W1, b1, W2, b2]")
        C, W1, b1, W2, b2 = params
        self.assertEqual(tuple(C.shape), (27, 10))
        self.assertEqual(tuple(W1.shape), (30, 200))
        self.assertEqual(tuple(b1.shape), (200,))
        self.assertEqual(tuple(W2.shape), (200, 27))
        self.assertEqual(tuple(b2.shape), (27,))
        for p in params:
            self.assertTrue(p.requires_grad, "every parameter needs requires_grad=True")

    def test_custom_dims(self):
        params = init_params(vocab_size=27, block_size=5, emb_dim=4, hidden=64,
                             generator=torch.Generator().manual_seed(2))
        C, W1, b1, W2, b2 = params
        self.assertEqual(tuple(C.shape), (27, 4))
        self.assertEqual(tuple(W1.shape), (20, 64))
        self.assertEqual(tuple(W2.shape), (64, 27))

    def test_reproducible_with_seed(self):
        a = init_params(generator=torch.Generator().manual_seed(7))
        b = init_params(generator=torch.Generator().manual_seed(7))
        for pa, pb in zip(a, b):
            self.assertTrue(torch.equal(pa, pb),
                            "same seed must give identical parameters - pass "
                            "generator= to every torch.randn")

    def test_initial_loss_is_calibrated(self):
        for seed in (2147483647, 42, 1337):
            loss = init_loss(seed)
            self.assertLess(
                loss, 3.5,
                f"initial loss {loss:.2f} (seed {seed}) - an unsure network "
                "scores -log(1/27) = 3.3. Yours is confidently wrong: scale "
                "W2 way down (* 0.01) and start b2 at zero.")

    def test_output_layer_scaled_not_deleted(self):
        params = init_params(generator=torch.Generator().manual_seed(3))
        W2 = params[3]
        self.assertGreater(
            W2.abs().max().item(), 0.0,
            "W2 is all zeros - scale it down (* 0.01), don't erase it: "
            "identical zero weights mean identical gradients for every "
            "output unit.")


if __name__ == "__main__":
    unittest.main()
