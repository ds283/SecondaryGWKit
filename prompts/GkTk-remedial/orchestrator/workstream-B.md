# Orchestrator prompt — Workstream B (prompts 03, 04)

You are orchestrating Workstream B of the Gk/Tk WKB phase remedial campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `gktk-remedial`). You do not write code
yourself. You dispatch one fresh-context subagent per prompt, review against fixed criteria, and
either continue or stop and report to the user.

**This is the foundation, and a schema change.** Prompt 03 replaces how conformal time is computed,
stored and read; prompts 04–10 are built on the two accessors it defines (`tau(z)`,
`tau.delta(z_a, z_b)`). A wrong sign convention, a `delta` implemented as a difference of two
pointwise values, or a low limb lost in the persistence round trip would propagate silently into
every later acceptance test. After 03 the datastore must be regenerated.

## What to read

`../README.md` §0–§2 (**§2 (b), (c) twice**), §3, §4, §4.3, §5, §5.1, §6, **§7 D1**;
`../RECONCILIATION.md` §1 items 10–14, §2 items 1, 2, 5, 11, 12; `../IMPLEMENTATION_STATE.md`
(board, §3 — in particular `[00-tau-storage-decision]` — and §5); `orchestrator/README.md`;
review §7, §12.7, §13.2, §13.3; `logs/01-…` and `logs/02-…` "State handed to the next prompt"
(you need the Gauss orders and the throughput figures to review 03).

Read 03 only when about to dispatch it; likewise 04. **Do not read 05–13.**

## Preconditions

`git status` clean; rows 01, 02 ✅/⚠️; rows 03, 04 ⬜. D1 is confirmed by the user (2026-09-10, README §7); note it in your report and proceed.
Confirm the harness:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_wkb_reference ComputeTargets.tests.test_background_derivatives -v
```

## Dispatching

The standard dispatch text of `workstream-A.md`. Models: **03 → Fable** (Opus if unavailable —
then apply every check below twice), **04 → Opus**. Run 03 → 04.

## Reviewing prompt 03 — the τ primitive

Structural checks 1–3 and 5 as in `../README.md` §4.3 (allowed files: `cumulative_table.py`,
`BackgroundModel.py`, the `BackgroundModel` factory, the `main.py` solver hunk, the named tests,
log, board; `docs/OPEN_ISSUES.md` if `[03-backgroundmodelvalue-build-path]` was opened).

4. **Tests pass when you run them:**
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_cumulative_table ComputeTargets.tests.test_background_tau -v
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
   ```
6. **Read `CumulativeTable.delta`.** It must be partial + [(hi_b − hi_a) + (lo_b − lo_a)] +
   partial. If it calls `value()` twice, or forms `(hi_b + lo_b) − (hi_a + lo_a)`, **stop** —
   design fact (c). Its docstring must state the 9e-4 rad reason.
7. **Read the sign convention.** `tau.delta(z_a, z_b) == tau(z_b) - tau(z_a)`, positive for
   $z_b<z_a$; the test asserts it on exact radiation. A flipped sign here flips every phase in 06.
8. **The floor is demonstrated**: the single-double control test exists and shows the loss; a
   test that only asserts the double-double accuracy fails this check.
9. **Node lookup is exact** (`node_index` by equality; no tolerance in $z$; no `expm1(log1p(z))`
   round trip used for matching): read the code.
10. **Schema and round trip**: `tau_lo_Mpc` column present, `nullable=False`; written and read in
    both factory paths; the exact-round-trip test exists; the regeneration comment is present.
    `grep -n "solve_ivp" ComputeTargets/BackgroundModel.py` empty.
11. **`main.py`**: `git diff HEAD~1 -- main.py` shows one hunk near `:2809-2821` and nothing else.
12. **The two latent factory defects** (`RECONCILIATION.md` §2 item 11) were **not** silently fixed:
    `git diff HEAD~1 -- Datastore/SQL/ObjectFactories/BackgroundModel.py` must not touch `:674`'s
    `"wkb_serial"` or `:705`'s `row_data.Hubble` unless the log records it as an opened issue
    *and* the user is asked. Fixing them is scope creep by the campaign's own rules.
13. **Numbers**: LambdaCDM nodes $\le2\times10^{-14}$; adjacent-node `delta` $\le10^{-13}$
    relative; build $\le0.5$ s; the shipped-object throughput re-measured and within 2× of the
    prototype's. The oracle-improvement note (~2 rad at $k=10^5$) is present.
14. **Stand-ins still construct**: `test_tk_source_functions.py` and `test_phase_groups.py`
    unchanged and passing (this prompt adds no `ModelFunctions` field, so a failure here means
    something else moved).

## Reviewing prompt 04 — τ_s and F

Structural checks; allowed files: `BackgroundModel.py`, its factory, the new test, log, board.

4. Tests:
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_background_cs_tau_friction -v
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
   ```
6. **`ModelFunctions` defaults**: `python -c "from ComputeTargets.BackgroundModel import ModelFunctions; print(ModelFunctions._field_defaults)"`
   shows `cs_tau` and `friction_F` defaulting to `None`; the thirteen-positional-argument test
   exists; `test_tk_source_functions.py` and `test_phase_groups.py` are **untouched** (`git diff
   HEAD~1 --stat`) and pass. This is design fact / rule 5 §7 and the guard for prompt 10.
7. The `wPerturbations < 0` guard exists and is tested or at least reasoned about in the log.
8. The ODE-vs-table friction test exists and reports $2$–$4\times10^{-7}$ — the documentation that
   the *ODE* was wrong.
9. Columns `cs_tau_Mpc`, `cs_tau_lo_Mpc`, `friction_F` present; `friction_F` has no unit scaling.
10. Numbers: $\tau_s$ $\le2\times10^{-14}$, $F$ $\le10^{-13}$ on LambdaCDM; QCD within 3× the
    reference floor; the $c_s^2$ transition intervals are among the checked baselines.

## Continue or stop

Continue on `COMPLETE`-class logs with only `IMPLEMENTATION CHOICE` deviations. Stop on the
campaign-wide conditions, on checks 6, 7, 9, 12 of prompt 03, on check 6 of prompt 04, or if any
accuracy row is missed.

## Completion criterion

Rows 03, 04 ✅/⚠️. Report: "Workstream B complete; the tree is at `<SHA>`. `BackgroundModel`
builds τ, τ_s and F as tables; `functions.tau/cs_tau/friction_F` expose `__call__` and `.delta`;
**every datastore created before `<03 SHA>` must be regenerated**. The WKB producers still run the
ODE (README §4.1: a usable stopping point). Ready for Workstream C (prompt 05)." Include the
accessor names, column names, solver label and Gauss orders verbatim from the logs.
