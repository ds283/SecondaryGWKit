# levin-phase-floor 05a §6 — a note on T1's scored reading (added 2026-10-26, after the run)

`state/predictions.md` and `state/predictions_attempt1_wrong_window_lines.md` are left exactly as
they were written. This note sits beside them and changes neither.

## The user's decision, as relayed

The coordinator relayed it to this session at about 13:23Z on 2026-10-26, from the laptop
orchestration session.
- **The question put to the user:** "Which reading of §9's resolution rule should be T1's scored
  verdict?"
- **The user chose:** "True step (Recommended)".
- **That option read:** "T1 is scored on windows whose true step |dθ/dz|(b−a)/64 is below π/2,
  narrowing further where it isn't. This is what §9 meant by 'resolved'. The literal
  unwrapped-step reading is reported beside it, unscored. Recorded as an amendment of §9's
  wording, made before any T1 number exists."

The laptop session amends §9's wording in 05a's records commit; this session does not.

## What it changes here

- **T1's scored verdict** is now taken on windows resolved by the true step: the first halving
  level at which $|d\theta_T/dz|(b-a)/64 < \pi/2$ under both $c$ and $t$. Every other clause of
  §9's T1 is unchanged.
- **The literal reading** fixed in `predictions.md`, `max|diff(numpy.unwrap(θ))| < π/2`, is still
  computed. It is reported beside the scored verdict, unscored.

## Timing: only what the files show

| When (UTC, 2026-10-26) | What |
|---|---|
| 12:29:30 | `state/predictions_attempt1_wrong_window_lines.md` written (file mtime) |
| 12:30:03 | `state/predictions.md` written (file mtime). It fixes both readings before the run: the literal one to be scored, the true-step one to be reported apart |
| 12:30:47 | the run launched (`out/cmd/16_launch.txt`) |
| 12:35:49 | both readings first scored: `out/cmd/23_analyse.txt`, `out/wider_05a.txt`, `state/analysis.json` and `state/probe_windows.json`. The literal reading gave fails, the true step holds |
| 12:36:26 | both readings re-scored from the saved windows (`out/cmd/24_rescore_t1.txt`) |
| about 13:23 | the decision relayed to this session |
| 13:24:32 | T1 scored by the true step alone (`out/cmd/26_T1_true_step_scored.txt`) |

When the user made the decision is for the laptop session to record. This session cannot say
whether any T1 number had reached the user by then.
