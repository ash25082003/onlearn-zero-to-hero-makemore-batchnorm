Run the progress scoreboard and react like a teacher, not a CI bot.

1. Run: `python check.py` (the only dependency is torch - see SETUP.md if it's missing).
2. It prints a milestone scoreboard and stops at the current milestone. Completed
   milestones stay ✅; future ones show as 🔒 locked, not as failures.
3. If the current milestone passes: celebrate briefly (one line), update `progress.md`,
   and ask if they want to go to `/next`.
4. If it's still failing: the scoreboard already prints a one-line hint and the failing
   check. Do NOT paste the full output or fix it for them. Add the smallest nudge toward
   the problem - a sentence or two - and let them try again. Talk only about the current
   milestone, never the locked ones.
