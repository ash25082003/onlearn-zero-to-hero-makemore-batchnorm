"""
baseline.py - your Module 3 network, frozen in time.

This file is PROVIDED and complete. It's the "before" picture: the naive init
and the plain forward pass exactly as Module 3 left them. You don't edit it -
it exists so you (and `explore.ipynb`) can put the broken version next to your
fixed one and *see* the difference.
"""
import torch


def init_params_naive(vocab_size=27, block_size=3, emb_dim=10, hidden=200, generator=None):
    """The Module 3 init: every tensor a raw torch.randn. First loss: ~27."""
    C = torch.randn((vocab_size, emb_dim), generator=generator)
    W1 = torch.randn((block_size * emb_dim, hidden), generator=generator)
    b1 = torch.randn(hidden, generator=generator)
    W2 = torch.randn((hidden, vocab_size), generator=generator)
    b2 = torch.randn(vocab_size, generator=generator)
    params = [C, W1, b1, W2, b2]
    for p in params:
        p.requires_grad = True
    return params


def forward(X, params):
    """The Module 3 forward pass: embed, tanh hidden layer, logits."""
    C, W1, b1, W2, b2 = params
    emb = C[X]
    h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
    return h @ W2 + b2
