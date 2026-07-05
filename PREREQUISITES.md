# Module 0 — Prerequisites (taught inline)

Before the diagnostics begin, the learner needs a few things checked. Run a quick
**diagnostic** at the very first session, then teach only the warm-ups they actually need.
A confident learner can skip straight to Milestone 1.

## Step 0 — Is PyTorch installed?

Run `python check.py`. If it complains about torch, walk them through `SETUP.md`
(`pip install torch`, CPU is fine) before anything else. This is the only required install.
(`explore.ipynb` also wants `jupyter` and `matplotlib` — optional but strongly recommended
in this module; it can wait until after Milestone 1.)

## The diagnostic (ask conversationally, don't make it feel like a test)

Find out, in a friendly back-and-forth:

1. **Did they do Module 3 (the MLP)?** This project starts from that exact network — the
   provided `baseline.py` *is* their Module 3 init and forward pass — and its finale is
   beating their own 2.2564. If they haven't done it, strongly suggest starting there; if
   they insist on starting here, do Warm-up A properly.
2. **The mystery** — do they remember their Module 3 loss starting around 26-27? Can they
   say what loss a network with *no idea* should score? (-log(1/27) = 3.3 — don't give it
   away; nudge them to compute it. If they get this, Milestone 1 is already half done.)
3. **Training-loop recall** — can they sketch the loop? Forward, loss, zero grads,
   `loss.backward()`, nudge weights against the gradient.
4. **Scale reasoning** — roughly what happens to the std of `x @ W` as fan-in grows, if x
   and W are unit gaussians? (It grows like sqrt(fan_in) — this module leans on that one
   fact over and over.)

Based on their answers, route them:
- Solid on all → go straight to Milestone 1.
- Shaky on one → do just that warm-up, then continue.
- New to this → warm-ups first, and set expectations warmly (this module has less new
  machinery than Module 3 but deeper ideas; every milestone is small).

## Warm-up A — The Module 3 story in three minutes (only if needed)
Goal: enough context for this project to make sense. A network that predicts the next
character from THREE characters: each character looks up a learned 10-dimensional vector
(the embedding table C), the three vectors get concatenated (30 numbers), pushed through a
tanh hidden layer (200 neurons), and out come 27 logits. Cross-entropy loss, minibatch
SGD, train/dev/test split; final dev loss 2.2564, beating the bigram's 2.454. This module
does not add capability — it adds *health*: the same ideas, initialized and instrumented
properly, stacked deeper. Whiteboard-style, no code.

## Warm-up B — The scale of a matmul (only if needed)
Goal: the one bit of math this module leans on. If x has std 1 and W's entries have std 1,
then (x @ W) has std ~sqrt(fan_in) — every output sums fan_in independent products, and
variances add. Let them verify it in a REPL in three lines (`torch.randn(1000, 30) @
torch.randn(30, 200)` — what's the std?). Corollary: divide W by sqrt(fan_in) and the scale
is preserved. That corollary is Milestones 2, 3, and half of 6.

## Warm-up C — The training loop in one breath (only if needed)
Goal: reactivate the Module 3 loop before Milestone 7 instruments it. Minibatch of 32,
forward, cross-entropy, `p.grad = None`, `loss.backward()`, `p.data += -lr * p.grad`,
repeat. That exact loop returns here — the only news is what we *record* while it runs.

When the needed warm-ups are done (or skipped), continue into the main CURRICULUM.md.
