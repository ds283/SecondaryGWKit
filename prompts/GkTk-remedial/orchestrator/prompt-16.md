# Orchestrator prompt — Prompt 16 (the `has_unresolved_osc` print policy)

You are orchestrating prompt 16 of the Gk/Tk WKB phase remedial campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `gktk-remedial`). You do not write code
yourself.

This is not a workstream — it is the follow-up prompt README §7 D2 anticipated ("closes when the
chosen policy lands (a follow-up prompt if not (i))"). Prompt 11 made the oscillation-resolution
test faithful and measured its consequence: it now fires on **2,149 of 2,149** $G_k$-like objects
(0 of 6 $T_k$-like), which at ~65,000 objects per model is ~$1.3\times10^5$ printed lines where
production prints none. **The user took decision D2 on 2026-09-11: option (ii)** — store the flag,
print a per-$k$ summary in `main.py`. Prompt 16 implements exactly that.

It runs **between prompts 11 and 12** of Workstream E, at the user's request; 12 is unaffected by
it and may follow immediately.

## What to read

The prompt: [`../16-unresolved-osc-print-policy.md`](../16-unresolved-osc-print-policy.md) — in
full, before dispatch. `../README.md` §2 **(h)**, §7 **D2**, §5; `../IMPLEMENTATION_STATE.md` —
row 11, M17, M18, and the `[00-unresolved-osc-print-policy]` entry in §3;
[`README.md`](README.md) (campaign-wide stop conditions); `logs/11-numeric-diagnostics-and-units.md`
Result section and "State handed to the next prompt"; review §13.1.

## The stop condition that does *not* apply here, and the one that does

README §2 (h) protects the `has_unresolved_osc` warning, and `../README.md` §4.3 makes deleting it
a stop. **Relocating it is the user's decision D2, already taken**, so a deviation that moves the
warning from the integrator to `main.py` is not a §2 (h) stop. What *is* still a stop: the flag's
value changing, the three payload keys moving, the factories being touched, or the `print` calls
being deleted outright rather than gated (which would make the relocation irreversible and leave a
direct caller of the integrator with no warning at all).

## Preconditions

`git status` clean; board row 11 ✅/⚠️ and row 16 ⬜. The prompt cites `RayWorkPool` line numbers
written against commit `8f606c4`; confirm before dispatch:

```bash
grep -n "post_handler" RayTools/RayWorkPool.py
grep -n "store_results" main.py | head
```

`post_handler` must still be called at the task exit points, and the two numeric queues must still
pass `store_results=False` and no `post_handler`. If an intervening commit changed either, fix the
prompt's references in a planning commit rather than dispatching a stale prompt.

Baseline: `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3`
— expect **273 tests, OK** at `8f606c4`.

## Dispatching

Standard dispatch text (`workstream-A.md`), with `NN-<name>` = `16-unresolved-osc-print-policy`.
Model: **Opus**.

## Reviewing prompt 16

The five checks of `../README.md` §4.3, plus:

4. Tests:
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_numeric_phase_cut ComputeTargets.tests.test_main_plumbing -v
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
   ```
   No test may be lost: 273 at `8f606c4`, and this prompt only adds.
6. **The warning is gated, not deleted**:
   `grep -n "print" Quadrature/integrators/numeric_with_phase_cut.py` still shows both warning
   lines, and a test proves `warn=True` still prints them. Deletion is a stop.
7. **The flag did not move.** Prompt 11's bit-identity constants still pass, and a test asserts the
   returned dict is equal under `warn=True` and `warn=False`. `git diff HEAD~1 --stat` shows no
   `Datastore/`.
8. **The wiring is guarded.** `test_main_plumbing.py` carries an `ast` check that both numeric
   `RayWorkPool` constructions pass a `post_handler` and that the formatter is called after each.
   Without it a later edit silently reverts the feature — this is the prompt's main exposure, and a
   missing guard is a stop.
9. **The summary functions are module-level** in `main.py` and the formatter *returns* lines.
   `load_main_py_functions` extracts only module-level `FunctionDef`s, so a class or a closure
   inside `run_pipeline` cannot be tested and check 8 would be hollow.
10. **`main.py` stayed in scope**: `git diff HEAD~1 -- main.py` touches only the two queue
    constructions, the accumulators, the summary calls and the two new functions. The `delta_logz=`
    arguments, the `0.85` constants and the tolerance plumbing are unchanged.
11. **The log shows the replacement output** — the actual formatted summary block for one $k$, so a
    reader can see what replaced the 1.3e5 lines.
12. **The issue closed correctly**: `[00-unresolved-osc-print-policy]` moved to §4 with the
    decision and date, `docs/OPEN_ISSUES.md` count 51 → 50 in the same commit, and the §4 entry
    records that the *semantic* question — whether the response grid should resolve the mode
    through the seam — is **not** closed here and is owned by the hand-over campaign
    (`docs/OPEN_ISSUES.md` §1.1). Closing it without that hand-off loses the finding.

## Continue or stop

Continue on a `COMPLETE`-class log with only `IMPLEMENTATION CHOICE` deviations, then proceed to
prompt 12 of Workstream E. Stop on any campaign-wide condition or on checks 6, 7, 8 or 9.

## Completion criterion

Row 16 ✅/⚠️; `[00-unresolved-osc-print-policy]` closed; the suite passing with no test lost.
Report the summary block the log quotes, and confirm that the datastore is unaffected by this
prompt (it changes only where information is printed) — the regeneration Workstream E requires is
prompt 12's $T_k$ tolerance, not this.
