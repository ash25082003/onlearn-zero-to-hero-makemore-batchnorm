# Setup Guide

This project works with any coding assistant that can read local files and run shell
commands. The shared tutoring rules live in `AGENTS.md`; tool-specific files are only
thin adapters that point back to the same guide.

## Common Setup

1. Clone or download this project.
2. Open a terminal in the project directory.
3. Install **PyTorch** - the one required dependency (CPU is plenty for this project):

   ```bash
   pip install torch
   ```

   Prefer a virtual environment? Make ONE for the whole course and reuse it in every
   module - all modules run on the same few packages:
   `python -m venv ~/zero-to-hero-env && source ~/zero-to-hero-env/bin/activate`,
   then `pip install torch regex` (once - covers every module). On Windows, activate
   with `%USERPROFILE%\zero-to-hero-env\Scripts\activate`.

4. Confirm it worked - this also shows your progress scoreboard:

   ```bash
   python check.py
   ```

   The tests themselves run on Python's built-in `unittest`; nothing else to install.

5. Open the project with your preferred coding assistant and say something like:

   ```text
   Hi, please read the project instructions and tutor me through this exercise.
   ```

The assistant should read `progress.md`, start with the Module 0 diagnostic if this is a
fresh session, and reveal only one milestone at a time.

## Recommended: the companion notebook

`explore.ipynb` is the visual side of the project, and this module is *about* seeing
inside a network - the init autopsy, saturation histograms, BatchNorm before/afters, and
the full diagnostic dashboard. It earns its keep from Milestone 1 on and needs two extras:

```bash
pip install jupyter matplotlib
jupyter notebook explore.ipynb   # or open it in VS Code / Cursor directly
```

The milestones all work without it - but this is the one module where you really want it.

## Claude Code

Claude Code can use the compatibility files included in this repo:

- `CLAUDE.md` imports `AGENTS.md` (`@AGENTS.md`), so Claude follows the same guide.
- `.claude/commands/hint.md`, `.claude/commands/check.md`, and `.claude/commands/next.md`
  provide optional slash-command shortcuts.

Suggested start:

```text
Read CLAUDE.md and tutor me through this project.
```

You can use `/hint`, `/check`, and `/next` if your Claude Code environment supports them.
Plain-language requests like "give me a hint", "run the tests", and "move on" should work
the same way.

## Codex

Codex reads `AGENTS.md` for project instructions — it contains the full tutoring method.

Suggested start:

```text
Read AGENTS.md and tutor me through this project. Do not write mlp.py, nn.py, or train.py for me.
```

Useful requests:

- "Give me a small hint."
- "Run the tests."
- "Move to the next milestone if I am ready."

## Cursor

Cursor reads `AGENTS.md` natively, and the included rule file reinforces it:

- `.cursor/rules/tutoring.mdc` tells Cursor to follow `AGENTS.md`.

Suggested start in Cursor chat:

```text
Read AGENTS.md and tutor me through this project. I want to write the code myself.
```

If Cursor does not automatically apply the rule, mention `AGENTS.md` directly in your
first message.

## Other Assistants

For any other coding environment, start by asking the assistant to read:

- `AGENTS.md`
- `progress.md`
- `CURRICULUM.md`

Then ask it to tutor, not solve. The assistant should only edit `progress.md` during
milestone tracking; the learner writes `mlp.py`, `nn.py`, and `train.py`.
