# Orchestrator prompt — Prompt 20 (key the numeric break-point policy)

You are orchestrating prompt 20 of the Gk/Tk WKB phase remedial campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `gktk-remedial`). You do not write code
yourself.

Not a workstream — the datastore half of `[18-numeric-solver-not-in-lookup-key]`, which prompt 18
opened and reserved to the user and which prompt 19 made bite a second time. The user took the
decision on 2026-09-13, in two parts:

1. **Key the configuration, not the solver.** `break_point_kind` is a real degree of freedom since
   prompt 19 and moves the stored QCD answer by up to 1.61e-04 of the envelope; the *solver* is
   hard-coded (`method="DOP853"` at two sites, no argument, a constant label, and `main.py`'s
   `solvers` dict never indexed), so `solver_serial` cannot vary and filtering on it would be a
   no-op. The prompt's §1 exists to stop an agent implementing the no-op, which is what the
   issue's own recorded "next step" still proposes.
2. **Audit all five compute targets**, but — under the design chosen in part 1 — only the two
   numeric ones need a key change. The other three are report-only (§5 of the prompt).

**It runs before prompt 13**, like 18 and 19 before it: it changes the numeric lookup key and makes
an old-schema datastore fail loudly, which is precisely what prompt 13 needs settled before it
builds one.

**This prompt changes production datastore code** — two factories, two compute targets, possibly
`main.py`. Review it as such. But note what it must *not* change: no computed value may move.

## What to read

The prompt: [`../20-key-the-break-point-policy.md`](../20-key-the-break-point-policy.md), in full.
`../README.md` §2 (h), §5, §6; `../IMPLEMENTATION_STATE.md` row 19 and the
`[18-numeric-solver-not-in-lookup-key]` entry in §3;
[`README.md`](README.md); `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` §9.4 and §10.5;
`logs/19-per-sector-break-point-policy.md`, "State handed to the next prompt".

## Preconditions

`git status` clean; row 19 ⚠️ and row 20 ⬜. Baseline **328 tests**. The machine-load-sensitive
`test_tk_wkb_phase.TestCost.test_wall_time_per_object` is still a wall-clock budget with no margin
(measured 0.0511–0.0520 s on 2026-09-13 against a hard-coded 0.06 s, but 0.0603–0.0681 s a day
earlier): measure it before dispatch so you can tell it from a regression, and tell the subagent to
leave it alone. Confirm the shape prompt 19 left, and the hard-coding the prompt's §1 rests on:

```bash
grep -n "break_point_kind" ComputeTargets/TkNumericIntegration.py ComputeTargets/GkNumericIntegration.py
grep -n "method=\"DOP853\"\|solver_label" Quadrature/integrators/numeric_with_phase_cut.py
grep -n "table.c.solver_serial ==" Datastore/SQL/ObjectFactories/*.py   # must be empty
```

## Dispatching

Standard dispatch text (`workstream-A.md`), with `NN-<name>` = `20-key-the-break-point-policy`.
Model: **Opus**.

## Reviewing prompt 20

The five checks of `../README.md` §4.3, plus:

4. Tests: `discover -s ComputeTargets/tests -t .` → **328 plus the new cases**, none removed and
   **no existing expectation changed**. An existing expectation that moved is a stop: this is a
   keying change and no number may move.
5. **The integrator did not change.** `git diff HEAD~1 HEAD --stat -- Quadrature/` must be empty.
   So must `config/defaults.py` and `CosmologyModels/`. A change to `numeric_with_phase_cut`
   is a stop, however small and however well argued — prompt 19 settled that file.
6. **No computed value moved, shown by measurement.** The prompt's §3 item 1 requires a payload
   compared bit-for-bit against `HEAD~1`. The log must quote it as something that was *run*. "The
   change is only to the key, so nothing can move" is reasoning, not evidence, and is not
   sufficient here.
7. **The solver was left alone.** `git diff HEAD~1 HEAD | grep -in "solver_serial\|IntegrationSolver\|solve_ivp+DOP853"`
   should show nothing but prose in the log, the board and the issue entry. Code that adds
   `solver_serial` to a query means the agent implemented the no-op §1 forbids — stop, and say
   that §1 was read and disregarded.
8. **The lookup is actually filtered, and demonstrated.** §3 item 2: a lookup under the wrong
   policy must be shown to miss. A test that only asserts the column exists, or that only reads
   the source with `ast`, has not shown it. The prompt names compiling the query without a
   connection as the way to do this; any equivalent is fine, an absent demonstration is not.
9. **One source of truth.** §2.1: the value passed to `numeric_with_phase_cut` and the value used
   in the key must be provably the same object, with a test that would fail if they drifted. Three
   literals that happen to agree today is the thing this prompt exists to prevent.
10. **The old-schema error exists and names the prompt** (§2.3), modelled on
    `Datastore/SQL/ObjectFactories/BackgroundModel.py:300-309`. No default value was supplied for
    the new field — a default silently blesses pre-prompt-18 rows and is a stop.
11. **The §5 audit was done and is report-only.** All four other solver-storing factories
    (`BackgroundModel`, `GkWKBIntegration`, `TkWKBIntegration`'s two) are covered in the log with
    their actual lookup keys; anything found is a §3 issue, not an edit. A diff touching them is a
    stop. The log must also answer the WKB-consumes-numeric-values question §5 ends on, since
    prompt 13 depends on it.
12. **The corrected diagnosis is written down** (§7): the issue entry and `docs/OPEN_ISSUES.md`
    must say that adding `solver_serial` to the queries is a no-op and why, so the next reader does
    not re-propose it. An entry that closes the issue without recording that is incomplete.

## Continue or stop

**Stop after this prompt** and report. Prompt 13 is next and needs to know what it may keep.
Stop early on checks 5, 6, 7, 9 or 10, or on any campaign-wide condition.

## Completion criterion

Row 20 ✅/⚠️; `[18-numeric-solver-not-in-lookup-key]` closed or narrowed with the reason; the suite
at 328 plus the new cases. Report: whether a lookup now distinguishes the two policies and how that
was shown; that no computed value moved, with the bit-identity check quoted; what the §5 audit
found on the other four solver columns and whether the WKB staleness hazard is real; and what
prompt 13 may now keep rather than regenerate.
