#!/usr/bin/env python3
"""
check.py - your progress scoreboard for activations, gradients & BatchNorm.

Run it any time from the project root:

    python check.py

It runs your milestones in order and stops at the first one that isn't finished,
so you only ever see what you're working on now - not a wall of red for
milestones you haven't reached yet. Completed milestones stay green; everything
ahead shows as locked until you get there.

The only dependency is PyTorch (see SETUP.md); the tests themselves run on
Python's built-in test tools.
"""
import sys
import unittest
import warnings

# torch prints a harmless warning if numpy isn't installed - not our problem here
warnings.filterwarnings("ignore", message=".*NumPy.*")

try:
    import torch  # noqa: F401
except ImportError:
    print()
    print("  PyTorch isn't installed (or your virtual environment isn't active).")
    print("  This project needs it - it's the one thing to install:")
    print()
    print("      pip install torch")
    print()
    print("  See SETUP.md for the full setup, then run `python check.py` again.")
    print()
    sys.exit(1)

# Ordered milestones: number, title, test module, and a warm hint shown only
# when this is the milestone you're currently on.
MILESTONES = [
    (1, "Fix the confidently-wrong init", "tests.test_milestone1",
     "At init the network should be unsure, not confidently wrong: every "
     "character equally likely, loss = -log(1/27) = 3.3. Logits are "
     "h @ W2 + b2, so in init_params scale W2 way down (* 0.01) and start "
     "b2 at zero."),
    (2, "The saturated tanh", "tests.test_milestone2",
     "saturation(h): (h.abs() > threshold).float().mean().item(). Then calm "
     "the hidden layer in init_params: W1 at scale 1.0 pins ~70% of tanh "
     "entries to the rails - shrink it (try * 0.2, and b1 * 0.01)."),
    (3, "Kaiming init", "tests.test_milestone3",
     "kaiming_scale(fan_in, gain) = gain / fan_in**0.5, and tanh's gain is "
     "5/3. Use it on W1: the fan-in there is block_size * emb_dim."),
    (4, "BatchNorm, written by hand", "tests.test_milestone4",
     "Per column across the batch: (x - x.mean(0, keepdim=True)) / "
     "x.std(0, keepdim=True), then * gain + bias. init_bn_params: gain "
     "torch.ones((1, hidden)), bias torch.zeros((1, hidden)), "
     "requires_grad=True on both."),
    (5, "Running stats: BatchNorm without a batch", "tests.test_milestone5",
     "update_running: (1 - momentum) * running + momentum * batch_value. "
     "batchnorm_infer: same center-scale-shift, but with the running stats "
     "you're handed - never .mean(0)/.std(0) on x, so a batch of one works."),
    (6, "Build your own torch.nn", "tests.test_milestone6",
     "Build in order - Linear first: __init__ (weight randn / fan_in**0.5, "
     "zero bias), __call__ (x @ weight + bias, kept in self.out), then "
     "parameters(). Then Tanh (same pattern, nothing to train). Then finish "
     "BatchNorm1d: batch stats when training, running buffers when not. "
     "Only once the classes pass, build_network: [Linear, BatchNorm1d, "
     "Tanh] x5 then Linear + BatchNorm1d, no biases, last gamma * 0.1 - "
     "and forward_net to run it."),
    (7, "Training telemetry", "tests.test_milestone7",
     "activation_stats: one dict per Tanh from layer.out. update_ratios: "
     "log10((lr * p.grad).std() / p.data.std()) for 2D params, in no_grad. "
     "train: the Module 3 loop + layer.out.retain_grad() before backward, "
     "recording lossi and ud. evaluate: forward_net + cross_entropy in "
     "no_grad."),
]

GLYPH = {"done": "✅", "current": "▶ ", "locked": "\U0001f512"}  # ✅ ▶ 🔒


class _OrderedResult(unittest.TestResult):
    """Records failures and errors in the order the tests ran, so the
    scoreboard always points at the earliest unfinished step - not at
    whichever kind of problem unittest happens to list first."""

    def __init__(self):
        super().__init__()
        self.problems = []

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.problems.append((test, self._exc_info_to_string(err, test)))

    def addError(self, test, err):
        super().addError(test, err)
        self.problems.append((test, self._exc_info_to_string(err, test)))


def run_module(module_name):
    """Run one milestone's tests. Returns (passed, total, first_problem_detail)."""
    loader = unittest.TestLoader()
    try:
        suite = loader.loadTestsFromName(module_name)
    except Exception as exc:  # tests couldn't even import (e.g. broken mlp.py)
        return 0, 1, f"could not load tests ({exc})"
    result = _OrderedResult()
    suite.run(result)
    total = result.testsRun
    passed = total - len(result.problems)
    detail = None
    if result.problems:
        test, traceback = result.problems[0]
        lines = traceback.strip().splitlines()
        last = lines[-1] if lines else "a check failed"
        detail = f"{test.id().split('.')[-1]} - {last}"
    return passed, total, detail


def main():
    statuses = []  # each: (num, title, state, passed, total, hint, detail)
    blocked = False
    for num, title, module, hint in MILESTONES:
        if blocked:
            statuses.append((num, title, "locked", 0, 0, hint, None))
            continue
        passed, total, detail = run_module(module)
        if total > 0 and passed == total:
            statuses.append((num, title, "done", passed, total, hint, None))
        else:
            statuses.append((num, title, "current", passed, total, hint, detail))
            blocked = True

    done_count = sum(1 for s in statuses if s[2] == "done")
    total_count = len(statuses)

    print()
    print("=" * 62)
    print("  Activations, gradients & BatchNorm - your progress")
    print("=" * 62)
    for num, title, state, passed, total, hint, detail in statuses:
        line = f"  {GLYPH[state]}  Milestone {num}  {title}"
        if state == "current":
            extra = f"{passed} of {total} checks passing" if total else "in progress"
            line += f"   <- you are here ({extra})"
        print(line)
    print("-" * 62)

    if not blocked:
        print(f"  \U0001f389  All {total_count} milestones complete!")
        print("      Run `python train.py` for the payoff: a deep network that")
        print("      trains sanely from step one - then open explore.ipynb.")
        print("=" * 62)
        print()
        return 0

    current = next(s for s in statuses if s[2] == "current")
    print(f"  {done_count} of {total_count} milestones complete.")
    print()
    print(f"  ▶  Milestone {current[0]} - {current[1]}: what to do next")
    print(f"     {current[5]}")
    if current[6]:
        print(f"     Failing check: {current[6]}")
    print("=" * 62)
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
