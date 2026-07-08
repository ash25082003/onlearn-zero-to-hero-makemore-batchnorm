# Curriculum — Activations, gradients & BatchNorm

Goal: the student takes their Module 3 MLP and learns why training breaks and how to see
it. Three acts: fix the initialization (the confident-wrong output, the saturated tanh,
kaiming), build BatchNorm by hand (batch stats, gain/bias, running statistics), then
package everything as their own mini `torch.nn` — Linear, BatchNorm1d, Tanh classes — and
train a 6-layer, 47,024-parameter network with real telemetry. The finale: `python
train.py` starts at loss ~3.3 instead of ~27, trains sanely, and evaluates at a val loss
of about 2.19 — **past Module 3's 2.2564** — and `explore.ipynb` turns into an x-ray
machine for the whole thing.

Reveal these milestones **one at a time**. Do not show the whole list to the student.

## Milestone 1 — Fix the confidently-wrong init
Module 3 planted the mystery: the first loss was ~27 when "no idea yet" scores
-log(1/27) = 3.3. Diagnose it (random-scale W2 and b2 make the logits huge — the network
is *confidently wrong* before it has seen anything), then fix `init_params`: W2 * 0.01,
b2 starts at zero.
*Concept:* at init you want maximum uncertainty. A badly-initialized network spends its
first thousands of steps just un-learning its own confident garbage — that's the
hockey-stick loss curve.

## Milestone 2 — The saturated tanh
Write `saturation(h)` and point it at the hidden layer: with W1 at scale 1.0, ~70% of tanh
outputs are pinned past 0.97. tanh is flat out there, flat means (chain rule!) almost no
gradient, and a neuron that's saturated for every input is a **dead neuron** — it will
never learn. Shrink W1 (the lecture eyeballs * 0.2) until the hidden layer breathes.
*Concept:* vanishing gradients through squashing nonlinearities — the failure mode that
haunted neural nets for decades, measured on your own network.

## Milestone 3 — Kaiming init
Replace the eyeballed 0.2 with the principled version: `kaiming_scale(fan_in, gain)` =
gain / sqrt(fan_in), with tanh's gain of 5/3. Dividing by sqrt(fan_in) keeps a layer's
output at the same scale as its input; the gain compensates for tanh squashing.
*Concept:* init as science instead of folklore — this is exactly what
`torch.nn.init.kaiming_normal_` computes, and after this milestone they know why.

## Milestone 4 — BatchNorm, written by hand
Careful init keeps preactivations in range at step 0 — BatchNorm *forces* them into range
at every step. Write it as a function: center each column over the batch, scale to unit
std, then let the network undo it where it wants to (`* gain + bias`, both trainable).
*Concept:* normalize-then-reparameterize. Plus BatchNorm's weird soul: every example's
output now depends on its batchmates — a mild regularizer, and a legendary source of bugs.
(A bias feeding into a BatchNorm is dead weight — the centering erases it.)

## Milestone 5 — Running stats: BatchNorm without a batch
Batch statistics need a batch — but inference often has exactly one example. Keep
exponential moving averages of the mean and std during training (`update_running`, the
0.999/0.001 nudge), and normalize with *those* at inference (`batchnorm_infer`).
*Concept:* why `model.train()` / `model.eval()` exist at all, and why forgetting `.eval()`
is one of the most-Googled PyTorch bugs. The single-example test in this milestone is that
bug, bottled.

## Milestone 6 — Build your own torch.nn
Package the whole toolkit into three classes with the real `torch.nn` API, built in this
order: `Linear` (kaiming baked in), `Tanh`, then `BatchNorm1d` (two personalities: batch
stats when training, running buffers when not). Only then `build_network` — it composes
all three: six Linears deep, BatchNorm after every one, no biases (BatchNorm eats them),
and the last gamma scaled by 0.1 (Milestone 1's lesson, BatchNorm edition) — and
`forward_net` to run it. 47,024 parameters.
*Concept:* a module is parameters + forward + state. After this, `nn.Linear` and
`nn.BatchNorm1d` are never black boxes again. (One honest detail: the class tracks running
*variance* with an eps, not std — that's what the real BatchNorm1d stores.)

## Milestone 7 — Training telemetry
The Module 3 training loop, instrumented: `activation_stats` (mean/std/saturation for
every Tanh), `update_ratios` (log10 of update-size over weight-size for every weight
matrix), `retain_grad` so even intermediate outputs keep their gradients, and `evaluate`.
*Concept:* professionals don't stare at the loss alone. Healthy training has signatures —
activation histograms that look alike layer after layer, saturation in the low percents,
update:data ratios hovering near the -3 line (updates ~1/1000th of the weights).

Each milestone has tests in `tests/`. A milestone is **done when its tests pass**.

When all tests pass, run `python train.py` for the payoff: a deep network whose first
loss is ~3.3 (remember starting at 27?), a val loss about 2.19 that beats Module 3's
2.2564, and names sampled one at a time — batch size one, powered by their running stats.
Then open `explore.ipynb`: the autopsy of the naive init, the saturation histograms, the
BatchNorm before/afters, and the full diagnostic dashboard. Next lecture: backprop
through all of it by hand (makemore Part 4: becoming a backprop ninja).
