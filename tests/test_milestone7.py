"""Milestone 7: training telemetry - the loop that watches itself learn."""
import unittest

import torch
import torch.nn.functional as F

from data import build_dataset, build_vocab, load_words, split_dataset
from nn import BatchNorm1d, Linear, Tanh, build_network, forward_net
from train import activation_stats, evaluate, train, update_ratios

Xtr = Ytr = Xdev = Ydev = None


def setUpModule():
    global Xtr, Ytr, Xdev, Ydev
    words = load_words()
    stoi, itos = build_vocab(words)
    g = torch.Generator().manual_seed(42)
    tr_words, dev_words, te_words = split_dataset(words, generator=g)
    Xtr, Ytr = build_dataset(tr_words, stoi)
    Xdev, Ydev = build_dataset(dev_words, stoi)


def tiny_net(seed=2147483647):
    return build_network(hidden=32, n_hidden_layers=2,
                         generator=torch.Generator().manual_seed(seed))


class TestActivationStats(unittest.TestCase):

    def make_layers(self):
        lin = Linear(2, 2, generator=torch.Generator().manual_seed(1))
        bn = BatchNorm1d(2)
        t1, t2 = Tanh(), Tanh()
        t1(torch.tensor([[3.0, 0.0], [-3.0, 0.5]]))
        t2(torch.tensor([[0.1, 0.2], [0.3, 5.0]]))
        return [lin, t1, bn, t2], t1, t2

    def test_reports_only_tanh_layers(self):
        layers, t1, t2 = self.make_layers()
        stats = activation_stats(layers)
        self.assertEqual(len(stats), 2,
                         "one report per Tanh layer - Linears and BatchNorms "
                         "aren't activations")
        self.assertEqual([s["layer"] for s in stats], [1, 3],
                         "'layer' is the index in the layers list "
                         "(enumerate!), so plots line up with the network")

    def test_values_are_correct(self):
        layers, t1, t2 = self.make_layers()
        stats = activation_stats(layers)
        self.assertAlmostEqual(stats[0]["mean"], t1.out.mean().item(), places=5)
        self.assertAlmostEqual(stats[0]["std"], t1.out.std().item(), places=5)
        # tanh(+/-3) = 0.9951 is past 0.97; the other two entries aren't
        self.assertAlmostEqual(stats[0]["saturated"], 0.5, places=5,
                               msg="two of t1's four outputs sit past 0.97")
        self.assertAlmostEqual(stats[1]["saturated"], 0.25, places=5,
                               msg="only tanh(5.0) = 0.9999 counts in t2")

    def test_threshold_is_respected(self):
        layers, t1, t2 = self.make_layers()
        stats = activation_stats(layers, threshold=0.4)
        self.assertAlmostEqual(stats[0]["saturated"], 0.75, places=5,
                               msg="with threshold 0.4, tanh(0.5) = 0.46 "
                                   "counts as saturated too")


class TestUpdateRatios(unittest.TestCase):

    def test_hand_computed(self):
        p = torch.tensor([[1.0, 2.0], [3.0, 5.0]], requires_grad=True)
        p.grad = torch.tensor([[0.1, 0.3], [0.2, 0.6]])
        q = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        q.grad = torch.ones(3)
        ratios = update_ratios([p, q], lr=0.1)
        self.assertEqual(len(ratios), 1,
                         "only 2D parameters are measured - q (1D) is skipped")
        expected = ((0.1 * p.grad).std() / p.data.std()).log10().item()
        self.assertAlmostEqual(ratios[0], expected, places=5,
                               msg="ratio = log10((lr * grad).std() / "
                                   "data.std())")

    def test_lr_shifts_the_ratio(self):
        p = torch.tensor([[1.0, 2.0], [3.0, 5.0]], requires_grad=True)
        p.grad = torch.tensor([[0.1, 0.3], [0.2, 0.6]])
        r1 = update_ratios([p], lr=0.1)[0]
        r2 = update_ratios([p], lr=0.01)[0]
        self.assertAlmostEqual(r1 - r2, 1.0, places=5,
                               msg="10x the learning rate is exactly +1 on "
                                   "the log10 ratio")


class TestTrain(unittest.TestCase):

    def test_training_learns_and_records(self):
        C, layers, params = tiny_net()
        C_before = C.detach().clone()
        lossi, ud = train(Xtr, Ytr, C, layers, params, steps=300, lr=0.1,
                          generator=torch.Generator().manual_seed(2147483647))
        self.assertEqual(len(lossi), 300, "one loss per step")
        self.assertIsInstance(lossi[0], float)
        first = sum(lossi[:20]) / 20
        last = sum(lossi[-20:]) / 20
        self.assertLess(last, first * 0.92,
                        f"loss went {first:.3f} -> {last:.3f} over 300 steps; "
                        "it should be clearly falling")
        self.assertLess(last, 3.1,
                        f"after 300 steps the batch loss should be under 3.1 "
                        f"(got {last:.3f})")
        self.assertFalse(torch.equal(C, C_before),
                         "the embedding table must move - is C in the "
                         "parameter update loop?")
        self.assertEqual(len(ud), 300, "one update_ratios entry per step")
        self.assertEqual(len(ud[0]), 4,
                         "the tiny test net has four 2D parameters: C and "
                         "three Linear weights")
        flat = [v for step in ud for v in step]
        self.assertTrue(all(-8.0 < v < 0.0 for v in flat),
                        "update:data ratios live around -3 on the log10 "
                        "scale; values outside (-8, 0) mean the formula is "
                        "off")

    def test_arguments_are_respected(self):
        C, layers, params = tiny_net(seed=99)
        lossi, ud = train(Xtr, Ytr, C, layers, params, steps=3, batch_size=4,
                          generator=torch.Generator().manual_seed(1))
        self.assertEqual(len(lossi), 3,
                         "steps must control the loop - no hard-coded counts")
        self.assertEqual(len(ud), 3)

    def test_intermediate_grads_are_retained(self):
        C, layers, params = tiny_net(seed=7)
        train(Xtr, Ytr, C, layers, params, steps=2, lr=0.1,
              generator=torch.Generator().manual_seed(7))
        grads = [layer.out.grad for layer in layers]
        self.assertTrue(all(g is not None for g in grads),
                        "call layer.out.retain_grad() before backward - the "
                        "gradient histograms in explore.ipynb need gradients "
                        "on every layer's output")


class TestEvaluate(unittest.TestCase):

    def test_matches_manual_loss_in_eval_mode(self):
        C, layers, params = tiny_net(seed=3)
        for layer in layers:
            layer.training = False
        X, Y = Xdev[:512], Ydev[:512]
        got = evaluate(X, Y, C, layers)
        self.assertIsInstance(got, float, "return loss.item(), a plain float")
        with torch.no_grad():
            expected = F.cross_entropy(forward_net(X, C, layers), Y).item()
        self.assertAlmostEqual(got, expected, places=4)

    def test_single_example_in_eval_mode(self):
        # the whole point of running stats: evaluation without a batch
        C, layers, params = tiny_net(seed=4)
        for layer in layers:
            layer.training = False
        loss = evaluate(Xdev[:1], Ydev[:1], C, layers)
        self.assertTrue(torch.isfinite(torch.tensor(loss)),
                        "evaluating ONE example returned nan/inf - in eval "
                        "mode BatchNorm must use its running buffers")


if __name__ == "__main__":
    unittest.main()
