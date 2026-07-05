"""
train.py - training with telemetry, and the payoff.

You write the three functions (Milestone 7): a training loop that records
what's happening inside the network while it learns, plus the two readouts
the diagnostic plots are built from. The block at the bottom is the reward
for finishing everything. Run it with:  python train.py
"""
import torch
import torch.nn.functional as F

from data import load_words, build_vocab, build_dataset, split_dataset
from nn import Tanh, build_network, forward_net


def activation_stats(layers, threshold=0.97):
    # TODO (Milestone 7): the health report for every Tanh in the network.
    # Walk `layers` with enumerate; for each layer that
    # `isinstance(layer, Tanh)`, read its saved `layer.out` and append
    #   {"layer": i, "mean": ..., "std": ..., "saturated": ...}
    # where saturated is the fraction of entries with absolute value above
    # `threshold` (your Milestone 2 saturation, inlined). Plain floats via
    # .item(). Return the list of dicts.
    raise NotImplementedError("Milestone 7: activation_stats")


def update_ratios(parameters, lr):
    # TODO (Milestone 7): for every 2D parameter (p.ndim == 2 - the weight
    # matrices; biases and gains start too uniform to measure), how big was
    # this step's update relative to the data it updated?
    #   ((lr * p.grad).std() / p.data.std()).log10().item()
    # Wrap the whole thing in torch.no_grad(). Return the list of floats.
    # Healthy training sits around -3: updates about 1/1000th of the weights.
    raise NotImplementedError("Milestone 7: update_ratios")


def train(X, Y, C, layers, parameters, steps=200, lr=0.1, batch_size=32, generator=None):
    # TODO (Milestone 7): the Module 3 training loop, upgraded for a layered
    # network and instrumented. Each step:
    #   1. ix = torch.randint(0, X.shape[0], (batch_size,), generator=generator)
    #   2. logits = forward_net(X[ix], C, layers); loss vs Y[ix]
    #      (F.cross_entropy)
    #   3. for every layer: layer.out.retain_grad() - the gradient plots in
    #      explore.ipynb need gradients on intermediate outputs, which torch
    #      normally discards
    #   4. zero grads (p.grad = None), loss.backward()
    #   5. nudge: p.data += -lr * p.grad
    #   6. record loss.item() into lossi, and update_ratios(parameters, lr)
    #      into ud
    # Return (lossi, ud).
    raise NotImplementedError("Milestone 7: train")


def evaluate(X, Y, C, layers):
    # TODO (Milestone 7): the full-dataset loss as a plain float: forward_net,
    # F.cross_entropy, .item(), inside torch.no_grad(). NOTE: this does NOT
    # flip the BatchNorm layers to inference mode for you - whoever calls it
    # sets layer.training first. Forgetting that is the classic BatchNorm bug,
    # and you're going to see why it matters.
    raise NotImplementedError("Milestone 7: evaluate")


def sample_name(C, layers, itos, block_size=3, generator=None):
    """PROVIDED - you wrote this loop in Module 3. Note the batch size: ONE."""
    out = []
    context = [0] * block_size
    while True:
        logits = forward_net(torch.tensor([context]), C, layers)
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
        if ix == 0:
            break
        out.append(itos[ix])
        context = context[1:] + [ix]
    return "".join(out)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # The payoff. Finish all 7 milestones, then run:  python train.py
    # Takes a few minutes on a laptop CPU.
    # ------------------------------------------------------------------
    words = load_words()
    stoi, itos = build_vocab(words)

    g = torch.Generator().manual_seed(42)
    train_words, dev_words, test_words = split_dataset(words, generator=g)
    Xtr, Ytr = build_dataset(train_words, stoi)
    Xdv, Ydv = build_dataset(dev_words, stoi)

    C, layers, params = build_network(generator=torch.Generator().manual_seed(2147483647))
    n_params = sum(p.nelement() for p in params)
    print(f"a deep MLP: 6 linear layers, BatchNorm everywhere, {n_params:,} parameters")

    with torch.no_grad():
        loss0 = F.cross_entropy(forward_net(Xtr[:256], C, layers), Ytr[:256]).item()
    print(f"  loss before training: {loss0:.2f} - Module 3 started near 27. "
          f"That's your init fixes.")

    g = torch.Generator().manual_seed(2147483647)
    lossi1, _ = train(Xtr, Ytr, C, layers, params, steps=9000, lr=0.1, generator=g)
    print(f"  after 9,000 steps (lr 0.1):  batch loss ~{sum(lossi1[-200:]) / 200:.4f}")
    lossi2, _ = train(Xtr, Ytr, C, layers, params, steps=3000, lr=0.01, generator=g)
    print(f"  after 3,000 more (lr 0.01):  batch loss ~{sum(lossi2[-200:]) / 200:.4f}")

    # The moment Milestone 5 exists for: flip every BatchNorm to inference
    # mode, so evaluation uses the running statistics - no batch required.
    for layer in layers:
        layer.training = False

    train_loss = evaluate(Xtr, Ytr, C, layers)
    val_loss = evaluate(Xdv, Ydv, C, layers)
    print()
    print(f"  train loss: {train_loss:.4f}")
    print(f"  val loss:   {val_loss:.4f}")
    print(f"  for scale - the bigram: 2.4540, Module 3's MLP: 2.2564")
    if val_loss < 2.2564:
        print("  >>> past your Module 3 number - with a loss that started sane,")
        print("      layers that stayed healthy, and an init you can defend.")
    else:
        print("  (above Module 3's number - train longer or check your setup)")

    print()
    print("20 names, sampled one at a time (batch size 1 - your running stats at work):")
    gs = torch.Generator().manual_seed(2147483647 + 10)
    for _ in range(20):
        print("  " + sample_name(C, layers, itos, generator=gs))
