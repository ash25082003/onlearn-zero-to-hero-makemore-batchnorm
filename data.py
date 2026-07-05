"""
data.py - dataset loading, vocabulary, context windows, and the split.

This file is PROVIDED and complete - you built every one of these functions
yourself in Modules 2 and 3, so here they're simply given. You don't edit
this file; you import from it.
"""
import torch


def load_words(path="names.txt"):
    """Return the 32,033 names, one string per entry."""
    with open(path) as f:
        return f.read().splitlines()


def build_vocab(words):
    """Return (stoi, itos): '.' is 0, 'a'..'z' are 1..26."""
    chars = sorted(set("".join(words)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def build_dataset(words, stoi, block_size=3):
    """Slide a block_size window over every name. You wrote this in Module 3."""
    xs, ys = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            xs.append(context)
            ys.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(xs), torch.tensor(ys)


def split_dataset(words, generator=None):
    """Shuffle and cut 80/10/10 into train/dev/test. You wrote this in Module 3."""
    perm = torch.randperm(len(words), generator=generator)
    shuffled = [words[i] for i in perm]
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    return shuffled[:n1], shuffled[n1:n2], shuffled[n2:]
