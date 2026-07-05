"""
mlp.py - your Module 3 network, about to get fixed.

This is YOUR file. It starts with your MLP init exactly as you wrote it in
Module 3 - which means it starts broken in a very specific, very instructive
way: the first loss is ~27 when an honest "no idea yet" scores 3.3. Milestones
1-3 fix the initialization; Milestones 4-5 build BatchNorm by hand. The
dataset helpers in `data.py` are complete and provided.

Check your progress any time with:  python check.py
"""
import torch


def kaiming_scale(fan_in, gain=5 / 3):
    # TODO (Milestone 3): the principled init scale, straight from the Kaiming
    # He paper: gain / sqrt(fan_in). Dividing by sqrt(fan_in) keeps the output
    # of a layer at roughly the same scale as its input; the gain (5/3 for
    # tanh) compensates for tanh squashing things back down. Return the float.
    raise NotImplementedError("Milestone 3: kaiming_scale")


def init_params(vocab_size=27, block_size=3, emb_dim=10, hidden=200, generator=None):
    # This is your Module 3 init, verbatim - every tensor a raw torch.randn.
    # It trains, but watch `python check.py` interrogate its first loss.
    # You'll fix it in three passes:
    #
    # TODO (Milestone 1): at init the network should be UNSURE, not confidently
    # wrong. The logits are h @ W2 + b2, so make them start near zero: scale
    # W2 way down (the lecture uses * 0.01 - small, not exactly zero) and
    # start b2 at zero.
    #
    # TODO (Milestone 2): the tanh comes up next. W1 at scale 1.0 makes
    # h = tanh(...) slam into +/-1 for ~70% of entries, and saturated tanhs
    # kill gradients. Shrink W1 (the lecture eyeballs * 0.2; also shrink b1,
    # e.g. * 0.01) until the saturation test is happy.
    #
    # TODO (Milestone 3): replace the eyeballed W1 factor with your
    # kaiming_scale(block_size * emb_dim) - same idea, no magic number.
    C = torch.randn((vocab_size, emb_dim), generator=generator)
    W1 = torch.randn((block_size * emb_dim, hidden), generator=generator)
    b1 = torch.randn(hidden, generator=generator)
    W2 = torch.randn((hidden, vocab_size), generator=generator)
    b2 = torch.randn(vocab_size, generator=generator)
    params = [C, W1, b1, W2, b2]
    for p in params:
        p.requires_grad = True
    return params


def saturation(h, threshold=0.97):
    # TODO (Milestone 2): what fraction of the entries of h are pinned to the
    # rails - absolute value above `threshold`? One line of tensor ops:
    # compare, cast to float, mean, .item(). Return a plain Python float.
    raise NotImplementedError("Milestone 2: saturation")


def batchnorm(x, gain, bias):
    # TODO (Milestone 4): BatchNorm's forward pass, written by hand. Per
    # COLUMN (each hidden unit, across the batch dimension 0):
    #   1. center: subtract x.mean(0, keepdim=True)
    #   2. scale to unit std: divide by x.std(0, keepdim=True)
    #   3. then let the network undo it where it wants to: * gain + bias
    # Return the result. (gain and bias broadcast - shapes (1, hidden).)
    raise NotImplementedError("Milestone 4: batchnorm")


def init_bn_params(hidden=200):
    # TODO (Milestone 4): BatchNorm's two trainable parameters, at their
    # do-nothing start: gain = torch.ones((1, hidden)), bias =
    # torch.zeros((1, hidden)). requires_grad=True on both; return
    # (gain, bias).
    raise NotImplementedError("Milestone 4: init_bn_params")


def update_running(running, batch_value, momentum=0.001):
    # TODO (Milestone 5): one step of the exponential moving average that
    # BatchNorm keeps on the side (no gradients involved):
    # (1 - momentum) * running + momentum * batch_value. Return the new value.
    raise NotImplementedError("Milestone 5: update_running")


def batchnorm_infer(x, gain, bias, running_mean, running_std):
    # TODO (Milestone 5): BatchNorm at inference time. Same
    # center-scale-shift as `batchnorm`, but using the running statistics
    # you're given instead of batch statistics - so it works on a batch of
    # ONE. No .mean() or .std() calls on x here.
    raise NotImplementedError("Milestone 5: batchnorm_infer")
