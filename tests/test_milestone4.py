"""Milestone 4: BatchNorm's forward pass, written by hand."""
import unittest

import torch

from mlp import batchnorm, init_bn_params


class TestBatchnorm(unittest.TestCase):

    def test_hand_computed(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        gain = torch.ones((1, 2))
        bias = torch.zeros((1, 2))
        out = batchnorm(x, gain, bias)
        # column means [3, 4], column stds (torch.std over dim 0) [2, 2]
        expected = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        self.assertTrue(
            torch.allclose(out, expected, atol=1e-4),
            f"got {out.tolist()}, expected {expected.tolist()} - center with "
            "x.mean(0, keepdim=True), scale with x.std(0, keepdim=True)")

    def test_gain_and_bias_applied(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        gain = torch.tensor([[3.0, 2.0]])
        bias = torch.tensor([[1.0, -1.0]])
        out = batchnorm(x, gain, bias)
        expected = torch.tensor([[-2.0, -3.0], [1.0, -1.0], [4.0, 1.0]])
        self.assertTrue(
            torch.allclose(out, expected, atol=1e-4),
            "after normalizing, multiply by gain and add bias - "
            "gain * xhat + bias")

    def test_output_statistics(self):
        g = torch.Generator().manual_seed(11)
        x = torch.randn((64, 8), generator=g) * 5 + 2
        gain = torch.full((1, 8), 2.0)
        bias = torch.full((1, 8), 0.5)
        out = batchnorm(x, gain, bias)
        self.assertEqual(tuple(out.shape), (64, 8), "shape must be preserved")
        self.assertTrue(torch.allclose(out.mean(0), torch.full((8,), 0.5), atol=1e-4),
                        "every column's mean should land exactly on the bias")
        self.assertTrue(torch.allclose(out.std(0), torch.full((8,), 2.0), atol=1e-3),
                        "every column's std should land exactly on the gain")

    def test_gradient_flows(self):
        g = torch.Generator().manual_seed(12)
        x = torch.randn((16, 4), generator=g)
        x.requires_grad = True
        gain, bias = init_bn_params(hidden=4)
        out = batchnorm(x, gain, bias)
        out.sum().backward()
        self.assertIsNotNone(x.grad, "batchnorm must stay differentiable - "
                                     "no .item()/.detach() inside")
        self.assertIsNotNone(gain.grad, "gain is a trainable parameter; "
                                        "gradients must reach it")

    def test_init_bn_params(self):
        gain, bias = init_bn_params(hidden=200)
        self.assertEqual(tuple(gain.shape), (1, 200))
        self.assertEqual(tuple(bias.shape), (1, 200))
        self.assertTrue(torch.equal(gain.data, torch.ones((1, 200))),
                        "gain starts at all ones - BatchNorm begins as a no-op")
        self.assertTrue(torch.equal(bias.data, torch.zeros((1, 200))),
                        "bias starts at all zeros")
        self.assertTrue(gain.requires_grad and bias.requires_grad,
                        "both are trained by backprop - requires_grad=True")

    def test_init_bn_params_custom_size(self):
        gain, bias = init_bn_params(hidden=64)
        self.assertEqual(tuple(gain.shape), (1, 64))


if __name__ == "__main__":
    unittest.main()
