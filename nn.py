"""
nn.py - your own torch.nn, built from scratch.

This is YOUR file (Milestone 6). Three small classes with the same API as
PyTorch's real nn.Linear / nn.BatchNorm1d / nn.Tanh, then a function that
stacks them into a 6-layer network. Every class follows one pattern: do the
work in __call__, keep the result in self.out (the diagnostics in Milestone 7
read it), and list the trainable tensors in parameters().

Check your progress any time with:  python check.py
"""
import torch


class Linear:

    def __init__(self, fan_in, fan_out, bias=True, generator=None):
        # TODO (Milestone 6): self.weight = torch.randn((fan_in, fan_out),
        # generator=generator) / fan_in**0.5  - the sqrt(fan_in) part of
        # kaiming, built in. self.bias = torch.zeros(fan_out) if bias is
        # requested, else None.
        raise NotImplementedError("Milestone 6: Linear.__init__")

    def __call__(self, x):
        # TODO (Milestone 6): self.out = x @ self.weight, plus the bias if
        # there is one. Store it in self.out AND return it.
        raise NotImplementedError("Milestone 6: Linear.__call__")

    def parameters(self):
        # TODO (Milestone 6): [self.weight], plus self.bias if there is one.
        raise NotImplementedError("Milestone 6: Linear.parameters")


class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        # TODO (Milestone 6): everything BatchNorm needs to remember:
        #   self.eps, self.momentum, self.training = True
        #   trained by backprop:  self.gamma = torch.ones(dim),
        #                         self.beta = torch.zeros(dim)
        #   running estimates (no gradients, updated by EMA - Milestone 5!):
        #     self.running_mean = torch.zeros(dim),
        #     self.running_var = torch.ones(dim)
        # Note it keeps the running VARIANCE, not std - that's what the real
        # nn.BatchNorm1d stores.
        raise NotImplementedError("Milestone 6: BatchNorm1d.__init__")

    def __call__(self, x):
        # TODO (Milestone 6): the two personalities of BatchNorm:
        #   training:  mean/var from THIS batch (x.mean(0, keepdim=True),
        #              x.var(0, keepdim=True))
        #   inference: self.running_mean / self.running_var instead
        # Normalize: (x - mean) / torch.sqrt(var + self.eps), then
        # self.out = self.gamma * xhat + self.beta.
        # If training, update both running buffers with your Milestone 5 EMA
        # (inside `with torch.no_grad():`). Return self.out.
        raise NotImplementedError("Milestone 6: BatchNorm1d.__call__")

    def parameters(self):
        # TODO (Milestone 6): only the two backprop-trained tensors -
        # [self.gamma, self.beta]. The running buffers are NOT parameters.
        raise NotImplementedError("Milestone 6: BatchNorm1d.parameters")


class Tanh:

    def __call__(self, x):
        # TODO (Milestone 6): self.out = torch.tanh(x); return it.
        raise NotImplementedError("Milestone 6: Tanh.__call__")

    def parameters(self):
        # TODO (Milestone 6): a Tanh has nothing to train.
        raise NotImplementedError("Milestone 6: Tanh.parameters")


def build_network(vocab_size=27, block_size=3, emb_dim=10, hidden=100,
                  n_hidden_layers=5, generator=None):
    # TODO (Milestone 6): assemble the deep network.
    #   1. C = torch.randn((vocab_size, emb_dim), generator=generator)
    #   2. layers: a [Linear(block_size * emb_dim, hidden, bias=False),
    #      BatchNorm1d(hidden), Tanh()] block, then (n_hidden_layers - 1)
    #      more [Linear(hidden, hidden, bias=False), BatchNorm1d(hidden),
    #      Tanh()] blocks, then the output pair: Linear(hidden, vocab_size,
    #      bias=False), BatchNorm1d(vocab_size). No biases anywhere - every
    #      Linear feeds a BatchNorm, which cancels biases (Milestone 4's
    #      centering step eats them).
    #   3. Make the init unsure, not confidently wrong (Milestone 1's lesson,
    #      BatchNorm edition): inside `with torch.no_grad():`, scale the LAST
    #      layer's gamma by 0.1.
    #   4. parameters = [C] + every layer's parameters();
    #      requires_grad=True on all of them.
    # Return (C, layers, parameters).
    raise NotImplementedError("Milestone 6: build_network")


def forward_net(X, C, layers):
    # TODO (Milestone 6): the whole forward pass: emb = C[X], flatten to
    # (N, block_size * emb_dim) with .view, then just push x through every
    # layer in order: `for layer in layers: x = layer(x)`. Return x (the
    # logits).
    raise NotImplementedError("Milestone 6: forward_net")
