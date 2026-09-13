# Orchestrator prompt — Workstream E (prompts 11, 12)

You are orchestrating Workstream E of the Gk/Tk WKB phase remedial campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `gktk-remedial`). You do not write code
yourself.

This workstream is **independent** of A–D and may run at any point. It cleans up a region the
review found sound, so its acceptance is largely *bit-identity*: the numeric samples must not
change unless a repair changes them by design and the test says so. It also contains the campaign's
one semantic decision that must go to the user **after** implementation, because its consequence is
measured, not predicted: D2, the corrected `has_unresolved_osc` fire rate. **This workstream always
stops after prompt 11.**

## What to read

`../README.md` §0.3, §2 (h), §4.3, §5, §6, **§7 D2**; `../RECONCILIATION.md` §1 items 8, 9, 14
and §2 items 8, 9; `../IMPLEMENTATION_STATE.md` (board, §3 `[00-unresolved-osc-print-policy]`,
`[00-tk-superhorizon-ic-series]`, §5); `orchestrator/README.md`; review §10, §12.5, §13.1.

Read each prompt only when about to dispatch it.

## Preconditions

`git status` clean; rows 11, 12 ⬜. If Workstream A has not run, the prompts build their stand-ins
locally — note this in your report. Baseline:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
```

## Dispatching

Standard dispatch text (`workstream-A.md`). Models: **11 → Opus**, **12 → Opus**. Run 11, stop
and report, then 12 when the user has responded (12 does not depend on the D2 answer, so the user
may say "continue" without deciding D2 — record that).

## Reviewing prompt 11 — diagnostics off the RHS, units, stop-point repairs

Structural checks; allowed files: `numeric_with_phase_cut.py`, `supervisors/numeric.py`,
`integration_tools.py`, `GkNumericIntegration.py`, `TkNumericIntegration.py`, `main.py`
(**two comment-only hunks at `:595-598`, `:1160-1165`**), the new test, log, board.

4. Tests:
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_numeric_phase_cut -v
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
   ```
6. **The warning survives**: `grep -n "unresolved" Quadrature/integrators/numeric_with_phase_cut.py`
   shows the flag computed on the sample grid and the single warning line printed; the payload keys
   `has_unresolved_osc`, `unresolved_z`, `unresolved_efolds_subh` unchanged (the factories are
   untouched — `git diff HEAD~1 --stat` shows no `Datastore/`). Deleting the warning is a stop
   (design fact (h), review §13.1).
7. **Off the RHS**: `grep -n "omegaEff_sq\|report_wavelength" ComputeTargets/GkNumericIntegration.py ComputeTargets/TkNumericIntegration.py`
   shows no call inside either `RHS`.
8. **The sampling change is stated** in a code comment and in the log (review §13.1's last
   rider).
9. **Bit-identity test** exists with the pre-change constants and the `HEAD~1` SHA in a comment;
   the stop-point tolerance is as the prompt specifies.
10. **`main.py`**: `git diff HEAD~1 -- main.py` shows only comment lines; the `delta_logz=`
    arguments and the `0.85` constants are unchanged. The hand-over window attributes are
    untouched (`grep -n "z_exit_subh_e3\|z_exit_subh_e6" ComputeTargets/GkNumericIntegration.py`
    unchanged from `HEAD~1`).
11. **The D2 measurement is in the log's first paragraph**: fire fractions for $G_k$ (response
    grid) and $T_k$ (source grid) at three $k$, with the $x$ at which the flag fires.
12. Speed-up reported (≥30 %); RHS counts unchanged.

**Then stop, regardless of outcome**, and report the D2 fractions with the three options of
README §7 D2. Resume with 12 when the user replies.

## Reviewing prompt 12 — the transfer-function numeric `atol`

Structural checks; allowed files: `config/defaults.py`, `main.py`, `test_tk_numeric_atol.py`,
`test_main_plumbing.py`, log, board.

4. Tests:
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_tk_numeric_atol ComputeTargets.tests.test_main_plumbing -v
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
   ```
6. **Every site**: `grep -n '"TkNumericIntegration"' main.py` — every hit inside an `object_get`
   carries `atol=Tk_numeric_atol`; the `ast` guard in `test_main_plumbing.py` exists and asserts
   both directions (Tk numeric uses the new name; every other `*Integration` uses `atol`). A
   missed site is a stop: it makes datastore lookups miss silently (`RECONCILIATION.md` §1 item 14).
7. `TkNumericIntegration.py` untouched; the initial condition untouched (`[00-tk-superhorizon-ic-series]`).
8. Numbers: baseline $9$–$13\times10^{-6}$ reproduced; new $\le3\times10^{-6}$; exact-IC
   $\le5\times10^{-7}$; RHS evaluations $+\le25\,\%$; $G_k$ unaffected to $10^{-9}$.

## Continue or stop

Always stop after 11 (D2). Otherwise continue on `COMPLETE`-class logs; stop on the campaign-wide
conditions, 11 checks 6, 7, 10, 12 check 6, or a missed threshold.

## Completion criterion

Rows 11, 12 ✅/⚠️. Report: "Workstream E complete; the tree is at `<SHA>`. The D2 decision is
`<decided / pending>`. A fresh datastore is required before any pipeline run (the $T_k$ tolerance
is a row key). Ready for Workstream F."
