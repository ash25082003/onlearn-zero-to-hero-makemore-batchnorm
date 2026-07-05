"""Milestone 5: running statistics - BatchNorm without a batch."""
import unittest

import torch

from mlp import batchnorm_infer, update_running


class TestRunningStats(unittest.TestCase):

    def test_update_running_math(self):
        r = update_running(torch.zeros(3), torch.tensor([1.0, 2.0, 3.0]),
                           momentum=0.1)
        self.assertTrue(
            torch.allclose(r, torch.tensor([0.1, 0.2, 0.3]), atol=1e-6),
            f"got {r.tolist()} - one EMA step from zeros with momentum 0.1 "
            "is (1 - 0.1) * 0 + 0.1 * batch = [0.1, 0.2, 0.3]")

    def test_update_running_default_momentum(self):
        r = update_running(torch.tensor([1.0]), torch.tensor([2.0]))
        self.assertAlmostEqual(
            r.item(), 0.999 * 1.0 + 0.001 * 2.0, places=6,
            msg="default momentum is 0.001 - the running value barely moves "
                "per step, which is the point")

    def test_update_running_moves_toward_batch(self):
        r = torch.tensor([0.0])
        for _ in range(2000):
            r = update_running(r, torch.tensor([5.0]), momentum=0.01)
        self.assertAlmostEqual(r.item(), 5.0, places=1,
                               msg="fed the same batch value forever, the "
                                   "running estimate must converge to it")

    def test_infer_uses_given_stats(self):
        x = torch.tensor([[5.0, 10.0]])
        out = batchnorm_infer(
            x,
            gain=torch.ones((1, 2)), bias=torch.zeros((1, 2)),
            running_mean=torch.tensor([[3.0, 8.0]]),
            running_std=torch.tensor([[2.0, 4.0]]),
        )
        expected = torch.tensor([[1.0, 0.5]])
        self.assertTrue(
            torch.allclose(out, expected, atol=1e-5),
            f"got {out.tolist()}, expected {expected.tolist()} - normalize "
            "with the running stats you're handed: (x - running_mean) / "
            "running_std, then gain/bias")

    def test_infer_works_on_a_single_example(self):
        # THE point of running stats: a batch of one has no batch statistics
        # (its std is undefined), yet inference must still work.
        x = torch.tensor([[7.0, -2.0, 0.0]])
        out = batchnorm_infer(
            x,
            gain=torch.full((1, 3), 2.0), bias=torch.full((1, 3), 1.0),
            running_mean=torch.tensor([[5.0, -4.0, 0.0]]),
            running_std=torch.tensor([[1.0, 2.0, 4.0]]),
        )
        self.assertTrue(torch.isfinite(out).all(),
                        "inference on ONE example produced nan/inf - you're "
                        "computing statistics from the batch instead of using "
                        "the running ones")
        expected = torch.tensor([[5.0, 3.0, 1.0]])
        self.assertTrue(torch.allclose(out, expected, atol=1e-5),
                        f"got {out.tolist()}, expected {expected.tolist()}")

    def test_infer_ignores_batch_statistics(self):
        g = torch.Generator().manual_seed(13)
        x = torch.randn((32, 4), generator=g) * 10 + 50  # wild batch stats
        gain = torch.ones((1, 4))
        bias = torch.zeros((1, 4))
        rm = torch.zeros((1, 4))
        rs = torch.ones((1, 4))
        out = batchnorm_infer(x, gain, bias, rm, rs)
        self.assertTrue(
            torch.allclose(out, x, atol=1e-5),
            "with running_mean 0 and running_std 1, inference is the "
            "identity - if the output looks normalized, you used the batch "
            "statistics instead of the running ones")


if __name__ == "__main__":
    unittest.main()
