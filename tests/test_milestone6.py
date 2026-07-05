"""Milestone 6: your own torch.nn - Linear, BatchNorm1d, Tanh, and the deep net."""
import unittest

import torch
import torch.nn.functional as F

from nn import BatchNorm1d, Linear, Tanh, build_network, forward_net


class TestLinear(unittest.TestCase):

    def test_matmul_with_known_weights(self):
        lin = Linear(3, 2, generator=torch.Generator().manual_seed(1))
        lin.weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        lin.bias = torch.tensor([10.0, 20.0])
        x = torch.tensor([[1.0, 2.0, 3.0]])
        out = lin(x)
        expected = torch.tensor([[14.0, 25.0]])
        self.assertTrue(torch.allclose(out, expected, atol=1e-5),
                        f"got {out.tolist()}, expected {expected.tolist()} - "
                        "__call__ is x @ weight + bias")
        self.assertTrue(torch.equal(lin.out, out),
                        "store the result in self.out - Milestone 7's "
                        "diagnostics read it")

    def test_parameters_with_and_without_bias(self):
        lin = Linear(4, 3, generator=torch.Generator().manual_seed(2))
        self.assertEqual(len(lin.parameters()), 2,
                         "with a bias, parameters() lists [weight, bias]")
        nb = Linear(4, 3, bias=False, generator=torch.Generator().manual_seed(3))
        self.assertEqual(len(nb.parameters()), 1,
                         "bias=False means parameters() is just [weight]")
        out = nb(torch.ones(2, 4))
        self.assertEqual(tuple(out.shape), (2, 3),
                         "__call__ must work without a bias")

    def test_init_scale_is_kaiming(self):
        lin = Linear(100, 100, generator=torch.Generator().manual_seed(4))
        std = lin.weight.std().item()
        self.assertTrue(0.09 < std < 0.11,
                        f"weight std is {std:.4f}, expected about 0.1 = "
                        "1/sqrt(100) - divide torch.randn by fan_in**0.5")
        self.assertTrue(lin.bias.abs().max().item() == 0.0,
                        "the bias starts at zeros, not randn")


class TestBatchNorm1d(unittest.TestCase):

    def test_training_mode_normalizes(self):
        bn = BatchNorm1d(4)
        self.assertTrue(getattr(bn, "training", None) is True,
                        "a fresh BatchNorm1d starts in training mode "
                        "(self.training = True)")
        g = torch.Generator().manual_seed(5)
        x = torch.randn((64, 4), generator=g) * 5 + 3
        out = bn(x)
        self.assertTrue(torch.allclose(out.mean(0), torch.zeros(4), atol=1e-4),
                        "in training mode, each column of the output should "
                        "have mean ~0 (gamma starts 1, beta starts 0)")
        self.assertTrue(torch.allclose(out.std(0), torch.ones(4), atol=1e-2),
                        "and std ~1 - normalize with this batch's mean and "
                        "variance")

    def test_running_buffers_update(self):
        bn = BatchNorm1d(3, momentum=0.1)
        g = torch.Generator().manual_seed(6)
        x = torch.randn((32, 3), generator=g) * 2 + 7
        bn(x)
        exp_mean = 0.9 * torch.zeros(3) + 0.1 * x.mean(0)
        exp_var = 0.9 * torch.ones(3) + 0.1 * x.var(0)
        self.assertTrue(
            torch.allclose(bn.running_mean.reshape(-1), exp_mean, atol=1e-4),
            "after one training-mode call, running_mean = 0.9 * old + "
            "0.1 * batch_mean (your Milestone 5 EMA with momentum 0.1)")
        self.assertTrue(
            torch.allclose(bn.running_var.reshape(-1), exp_var, atol=1e-4),
            "running_var updates the same way - note it tracks VARIANCE "
            "(x.var(0)), not std")

    def test_eval_mode_uses_running_stats(self):
        bn = BatchNorm1d(2)
        bn.running_mean = torch.tensor([1.0, 2.0])
        bn.running_var = torch.tensor([4.0, 0.25])
        bn.training = False
        x = torch.tensor([[3.0, 1.0]])  # a batch of ONE
        out = bn(x)
        expected = (x - torch.tensor([1.0, 2.0])) / torch.sqrt(
            torch.tensor([4.0, 0.25]) + 1e-5)
        self.assertTrue(torch.isfinite(out).all(),
                        "eval mode on a single example produced nan/inf - "
                        "you're still using batch statistics")
        self.assertTrue(torch.allclose(out, expected, atol=1e-4),
                        f"got {out.tolist()}, expected {expected.tolist()} - "
                        "eval mode normalizes with the running buffers")

    def test_eval_mode_does_not_touch_buffers(self):
        bn = BatchNorm1d(3)
        bn.training = False
        before = bn.running_mean.clone().reshape(-1)
        g = torch.Generator().manual_seed(7)
        bn(torch.randn((16, 3), generator=g) + 9)
        self.assertTrue(torch.allclose(bn.running_mean.reshape(-1), before),
                        "the running buffers only update in training mode")

    def test_parameters_are_gamma_and_beta(self):
        bn = BatchNorm1d(5)
        params = bn.parameters()
        self.assertEqual(len(params), 2,
                         "parameters() is [gamma, beta] - the running buffers "
                         "are state, not parameters")
        for p in params:
            self.assertEqual(p.ndim, 1,
                             "gamma and beta are 1D (dim,) tensors, exactly "
                             "like the real nn.BatchNorm1d's weight and bias")
        datas = [p.detach().reshape(-1) for p in params]
        self.assertTrue(any(torch.equal(d, torch.ones(5)) for d in datas),
                        "gamma starts at ones")
        self.assertTrue(any(torch.equal(d, torch.zeros(5)) for d in datas),
                        "beta starts at zeros")


class TestTanh(unittest.TestCase):

    def test_tanh(self):
        t = Tanh()
        x = torch.tensor([[0.0, 100.0, -2.0]])
        out = t(x)
        self.assertTrue(torch.allclose(out, torch.tanh(x)))
        self.assertTrue(torch.equal(t.out, out),
                        "store the result in self.out - the activation "
                        "histograms read it")
        self.assertEqual(t.parameters(), [],
                         "a Tanh has no trainable parameters")


class TestBuildNetwork(unittest.TestCase):

    def test_parameter_count(self):
        C, layers, params = build_network(
            generator=torch.Generator().manual_seed(2147483647))
        n = sum(p.nelement() for p in params)
        self.assertEqual(
            n, 47024,
            f"got {n:,} parameters, expected 47,024: C (27x10) + six "
            "bias-free Linears (30-100, 4x 100-100, 100-27) + gamma/beta "
            "for every BatchNorm1d")

    def test_parameter_count_tracks_arguments(self):
        C, layers, params = build_network(
            hidden=32, n_hidden_layers=2,
            generator=torch.Generator().manual_seed(1))
        n = sum(p.nelement() for p in params)
        self.assertEqual(n, 3300,
                         f"got {n:,}, expected 3,300 for hidden=32, "
                         "n_hidden_layers=2 - the stack must follow the "
                         "arguments, not stay hard-coded")

    def test_initial_loss_is_calibrated(self):
        C, layers, params = build_network(
            generator=torch.Generator().manual_seed(9))
        gd = torch.Generator().manual_seed(10)
        X = torch.randint(0, 27, (256, 3), generator=gd)
        Y = torch.randint(0, 27, (256,), generator=gd)
        with torch.no_grad():
            loss = F.cross_entropy(forward_net(X, C, layers), Y).item()
        self.assertLess(loss, 3.5,
                        f"initial loss {loss:.2f} - Milestone 1's lesson, "
                        "BatchNorm edition: scale the LAST BatchNorm's gamma "
                        "by 0.1 so the logits start timid")

    def test_all_parameters_require_grad(self):
        C, layers, params = build_network(
            generator=torch.Generator().manual_seed(11))
        for p in params:
            self.assertTrue(p.requires_grad)
        self.assertTrue(any(p is C for p in params),
                        "C is a trainable parameter too - include it")

    def test_reproducible_with_seed(self):
        a = build_network(generator=torch.Generator().manual_seed(12))
        b = build_network(generator=torch.Generator().manual_seed(12))
        self.assertTrue(torch.equal(a[0], b[0]),
                        "same seed, same C - pass generator through")
        self.assertTrue(torch.equal(a[1][0].weight, b[1][0].weight),
                        "same seed, same first-layer weights")


class TestForwardNet(unittest.TestCase):

    def test_embedding_and_flatten(self):
        C = torch.arange(54, dtype=torch.float32).view(27, 2)
        X = torch.tensor([[0, 1, 2]])
        out = forward_net(X, C, [])  # no layers: just embed + flatten
        expected = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
        self.assertTrue(torch.equal(out, expected),
                        "forward_net starts exactly like Module 3: C[X], "
                        "then .view(N, block_size * emb_dim)")

    def test_layers_applied_in_order(self):
        C = torch.eye(27)
        lin = Linear(81, 4, generator=torch.Generator().manual_seed(13))
        X = torch.tensor([[1, 2, 3], [4, 5, 6]])
        out = forward_net(X, C, [lin])
        emb = C[X].view(2, -1)
        self.assertTrue(torch.allclose(out, emb @ lin.weight + lin.bias,
                                       atol=1e-5),
                        "after flattening, push x through every layer in "
                        "order: for layer in layers: x = layer(x)")

    def test_output_shape(self):
        C, layers, params = build_network(
            generator=torch.Generator().manual_seed(14))
        X = torch.randint(0, 27, (64, 3),
                          generator=torch.Generator().manual_seed(15))
        out = forward_net(X, C, layers)
        self.assertEqual(tuple(out.shape), (64, 27),
                         "the network ends in logits: one score per character")


if __name__ == "__main__":
    unittest.main()
