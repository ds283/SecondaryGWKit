# Orchestrator prompt — Workstream D (prompts 08, 09, 10)

You are orchestrating Workstream D of the Gk/Tk WKB phase remedial campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `gktk-remedial`). You do not write code
yourself.

This workstream cures the consumer-side error — a cubic spline of a phase that grows to $10^{12}$
rad — by evaluating the leading term from the tables and splining only the small residual. Prompt
08 is a deletion with a frozen signature; prompt 09 is the design-critical one (the $\varphi$
decomposition and its interplay with the `GkSource` rectifier); prompt 10 is its transfer-function
twin and carries the campaign's one file overlap with the in-flight `transfer-remedial` campaign.

## What to read

`../README.md` §0.2, §0.4, §2 (**(c), (e), (g)**), §4.2, §4.3, §5, §6, §7 D4–D6;
`../RECONCILIATION.md` §1 items 5–7 and §2 items 6, 7, 13; `../IMPLEMENTATION_STATE.md` (board,
§3 — `[00-consumer-anchoring-floor]`, `[00-transfer-remedial-test-file-overlap]` — and §5);
`orchestrator/README.md`; review §5, §7, §8.3, §12.6, §13.3, §13.4; `logs/06-…` (the
$\delta$-wrap case), `logs/07-…`.

Read each prompt only when about to dispatch it. **Do not read 11–13.**

## Preconditions

`git status` clean; rows 06, 07 ✅/⚠️; 08–10 ⬜. **Before dispatching 08**: the user's decision
(README §4.2 item 1) is that Workstream D runs only after `transfer-remedial` has been merged into
this branch. Confirm it: `git log --oneline | grep -i "bessel"` should show that campaign's
prompts 01–09 (its 08 is "fixture revalidation"), and `LiouvilleGreen/bessel_phase.py` should no
longer import `phase_spline`. If the merge has not happened, **stop and ask** — do not dispatch
08, 09 or 10.

## Dispatching

Standard dispatch text (`workstream-A.md`). Models: **08 → Sonnet**, **09 → Fable** (Opus if
unavailable), **10 → Opus**. Run 08 → 09 → 10.

## Reviewing prompt 08 — `phase_spline` de-chunk

Structural checks; allowed files: `LiouvilleGreen/phase_spline.py`, `LiouvilleGreen/tests/test_phase_spline.py`,
log, board — **nothing else**. `git diff HEAD~1 --stat` is the review.

4. Tests:
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_phase_spline -v
   PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_phase -v
   PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_phase_groups ComputeTargets.tests.test_tk_source_functions ComputeTargets.tests.test_gk_source_policy ComputeTargets.tests.test_quadsource_integral -v
   ```
   All callers untouched and passing is the acceptance test. `test_three_bessel` if it finishes in
   ten minutes; otherwise note it.
6. `grep -n "_match_chunk\|_build_log_chunks\|MINIMUM_SPLINE_DATA_POINTS" LiouvilleGreen/phase_spline.py` empty.
7. `python -c "import inspect; from LiouvilleGreen.phase_spline import phase_spline; print(inspect.signature(phase_spline.__init__))"`
   shows the unchanged signature; `num_chunks` returns 1.
8. Test 1 reproduces the $h^4x/384$ law ($7$–$10\times10^{-5}$ rad at $k=10^6$) — this is the number
   prompt 09 must beat, so it must be in 08's log.
9. The module docstring says what the object is **not** (a cure for the growth) and points at
   `PrimitivePhase`.

## Reviewing prompt 09 — `PrimitivePhase` and the Green's-function consumer

Structural checks; allowed files: `primitive_phase.py`, `GkSourcePolicyData.py`,
`ComputeTargets/__init__.py`, three test modules, log, board. **`GkSource.py` untouched**
(`git diff HEAD~1 -- ComputeTargets/GkSource.py` empty — D5).

4. Tests:
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_primitive_phase ComputeTargets.tests.test_gk_source_primitive_phase ComputeTargets.tests.test_gk_source_policy -v
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
   ```
   `test_quadsource_integral.py`, `test_phase_groups.py` **unchanged** and passing (protocol check).
6. **Read `PrimitivePhase.raw_theta`**: leading term from `leading.delta(...)`, residual from a
   spline of $\varphi$ — never a spline of the full phase (a stop, design fact (g)); `theta_deriv`
   is closed-form $\mp k(1+z)/H$ plus $\varphi'$; `theta_mod_2pi` uses `WKB_mod_2pi` on `raw_theta`
   and the docstring names the D6 floor.
7. **$\varphi$ is built from the rectified `theta_div_2pi`**, not `raw_theta_div_2pi`: read
   `_create_functions`. The stand-in test (4.2 item 1) shows $2\pi$ jumps *before* rectification
   at the $\delta$-wrap points from 06's log and none after — both halves must be asserted.
8. `_classify_Levin` uses the same `PrimitivePhase`; `grep -n "phase_spline" ComputeTargets/GkSourcePolicyData.py`
   empty; the false chunking comment (`:666-670`) is gone.
9. **The ratio test**: `PrimitivePhase` vs `phase_spline` on the same samples at $k=10^8$,
   ratio $>10^5$ (README §6 consumer row). Quote it.
10. How `model` was obtained in `_create_functions` is stated; if `GkSource` had no model proxy,
    it is a `STRUCTURALLY REQUIRED` note and the plumbing is described.
11. `pure-WKB` objects: zero rectifier corrections asserted.

## Reviewing prompt 10 — the transfer-function consumer

Structural checks; allowed files: `TkSourceFunctions.py`, `test_tk_source_functions.py`
(fixtures and new assertions), `test_phase_groups.py` (**stand-in construction only**), log, board.

4. Tests:
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_tk_source_functions ComputeTargets.tests.test_phase_groups ComputeTargets.tests.test_quadsource_integral -v
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
   ```
6. **The overlap discipline**: `git diff HEAD~1 -- ComputeTargets/tests/test_phase_groups.py`
   and `-- ComputeTargets/tests/test_tk_source_functions.py` touch **no tolerance constant and no
   tolerance comment**. Any such hunk is a stop (README §4.2 item 2).
7. `grep -n "phase_spline\|PHASE_SPLINE_CHUNK_LOGSTEP\|friction_RHS" ComputeTargets/TkSourceFunctions.py` empty.
8. `friction()` reads `friction_F.delta(crossover_z, z)`; the construction-time cross-check
   against the stored samples exists and is tested to raise on inconsistent input.
9. The module docstring's protocol paragraph is rewritten truthfully (read it against the code).
10. Numbers: test 3.1's ratio $>10^5$ at $x_T=10^6$; `omega == phase.theta_deriv` to $10^{-10}$.

## Continue or stop

Continue on `COMPLETE`-class logs. Stop on the campaign-wide conditions; on 08 check 5 (any file
beyond the two); on 09 checks 6, 7 or a touched `GkSource.py`; on 10 check 6; on any missed
threshold; and **before 10** if the overlap question is unanswered.

## Completion criterion

Rows 08–10 ✅/⚠️. Report: "Workstream D complete; the tree is at `<SHA>`. The campaign's accuracy
claims now hold end to end (README §4.1). Ready for Workstream E (11, 12) if not already run, then
F." Include the two ratio tests, the $\varphi$ statistics, and the exact fixture lines changed in
the two shared test files.
