# Fix your init & build BatchNorm

**Neural Networks: Zero to Hero - Module 4**
Companion lecture: [Building makemore Part 3: Activations & Gradients, BatchNorm](https://youtu.be/P6sfmUTpUmc) by Andrej Karpathy.

The MLP you built in Module 3 works - and it's been hiding something. Its very first
loss was ~27, when a network with *no idea* should score 3.3. It spent thousands of steps
un-learning its own confidently-wrong start, most of a tanh layer arrived dead on the
scene, and the loss curve never told you. This project is about noticing. You diagnose
and fix the initialization (down to **Kaiming init**, derived, not pasted), build
**BatchNorm from scratch** - batch statistics, gain and bias, running stats for
batch-of-one inference - and then package it all as your own mini `torch.nn`: `Linear`,
`BatchNorm1d`, `Tanh` classes, stacked six layers deep, 47,024 parameters, with the
telemetry professionals actually watch: activation histograms, gradient stats, and
update-to-data ratios.

You'll build every piece yourself, with an AI tutor guiding you one step at a time.

## What you need
- Python 3.9+ and **PyTorch** - the one required install: `pip install torch` (CPU is
  plenty). Progress checks run on Python's built-in `unittest` via `python check.py`.
- Any coding agent or AI assistant that can read the project files and run shell commands
  (Claude, Codex, Cursor, or another tool)
- **Module 3 (the MLP)** under your belt - this project starts from that exact network
  (its naive init ships in `baseline.py` as the "before" picture) and ends by beating
  your own 2.2564.

## How to start
1. Download or clone this project and `cd` into it.
2. Open it with your preferred coding agent.
3. Just say hi. Your tutor takes it from there.

For tool-specific setup notes, see [SETUP.md](./SETUP.md). It covers Claude Code, Codex,
Cursor, and generic coding assistants.

Your code goes in `mlp.py` (the fixes and hand-built BatchNorm), `nn.py` (your own
torch.nn), and `train.py` (the instrumented training loop). The dataset machinery in
`data.py` is provided - you wrote it in Modules 2-3. Ask for a hint when stuck, ask the
tutor to check your work to run the tests, and ask to move on when ready for the next
milestone. Plan on roughly two sittings; seven milestones in all.

The tests decide when you're done - not the tutor. When they all pass, run
`python train.py`: a few minutes of training whose *first* printed loss is ~3.3
(remember 27?), ending past Module 3's number - and sampling names one character at a
time, batch size one, which only works because of the running statistics you built.

**The dashboard:** this module's `explore.ipynb` (`pip install jupyter matplotlib`) is
the biggest one yet, because this module is about *seeing*: the autopsy of the naive
init, the dead-neuron map, tanh saturation before and after your fixes, BatchNorm taming
a wild layer, and the per-layer activation, gradient, and update-ratio dashboards for
your deep network.

Next up: **[Module 5 - Become a backprop ninja](https://onlearn.app/projects/nn-zero-to-hero/backprop-ninja)**,
where the `loss.backward()` you've been trusting gets rebuilt by hand, gradient by
gradient, through every layer you just stacked (yes, through BatchNorm too).

---

## More from Onlearn

This is **Module 4** of **[Neural Networks: Zero to Hero](https://onlearn.app/projects/nn-zero-to-hero)**
on Onlearn — guided, build-it-yourself projects with an AI tutor that checks your work against
the tests.

**[→ Explore more courses](https://onlearn.app/projects)**
