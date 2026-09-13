# Orchestrator prompt — Prompt 18 (declared discontinuities in the numeric ODE)

You are orchestrating prompt 18 of the Gk/Tk WKB phase remedial campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `gktk-remedial`). You do not write code
yourself.

Not a workstream — the repair prompt 17's check 6 forced, approved by the user 2026-09-12. Prompt
17 could not demonstrate its reference converged on `QCDModel` at four of fifty wavenumbers,
because `QCD_Cosmology`'s $H(z)$ jumps and `numeric_with_phase_cut` integrates straight across the
jumps in one `solve_ivp` call. Prompt 02 already built the declaration protocol the fix needs;
prompt 18 wires it to the ODE and re-measures.

**It runs before prompt 13.** The split changes computed values on QCD in both sectors, and
`solver_serial` is not part of the `GkNumericIntegration` lookup key (prompt 18 §3.1), so a QCD
datastore built before this commit would hold pre-split rows that the post-split lookups cannot
tell apart.

**This prompt changes production code**, unlike 17 — a shared integrator that both
`GkNumericIntegration` and `TkNumericIntegration` run through. Review it as such.

## What to read

The prompt: [`../18-numeric-ode-break-points.md`](../18-numeric-ode-break-points.md), in full.
`../README.md` §2 (d) and (h), §5, §6; `../IMPLEMENTATION_STATE.md` row 17, M22, and the
`[17-qcd-reference-not-converged]` entry in §3; [`README.md`](README.md);
`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` §4; `logs/17-tk-numeric-atol-k-sweep.md` and
`logs/02-qcd-residual-convergence.md`, both "State handed to the next prompt" sections.

## Preconditions

`git status` clean; row 17 ✅ and row 18 ⬜. Baseline **299 tests, OK**. Confirm the protocol the
prompt builds on still looks as it did:

```bash
grep -n "break_temperatures_GeV" CosmologyModels/GenericEOS/GenericEOS.py CosmologyModels/GenericEOS/QCD_EOS.py
grep -n "def integration_break_points" CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py
grep -n "def _cosmology_break_points" ComputeTargets/BackgroundModel.py
grep -n "solve_ivp(" Quadrature/integrators/numeric_with_phase_cut.py
```

## Dispatching

Standard dispatch text (`workstream-A.md`), with `NN-<name>` = `18-numeric-ode-break-points`.
Model: **Opus**.

## Reviewing prompt 18

The five checks of `../README.md` §4.3, plus:

4. Tests: `discover -s ComputeTargets/tests -t .` → **299 plus the new
   `test_numeric_break_points.py` cases**, none removed and no existing expectation changed. If a
   pre-existing test's expected value moved, that is a stop: on a smooth cosmology this change is
   supposed to be a no-op.
5. **The generic property survived.** Nothing under `Quadrature/` or `ComputeTargets/` names a
   temperature, a model, or an equation of state:
   ```bash
   git diff HEAD~1 -- Quadrature/ ComputeTargets/ | grep -in "qcd\|saikawa\|T_LO\|T_HI\|120_MEV\|MeV"
   ```
   must be empty of anything but comment prose citing the campaign. A hard-coded break point is a
   stop — it is the specific thing the user objected to, and the whole reason the declaration
   protocol exists.
6. **`GenericEOS`'s default is still "smooth"** and the quadrature path is unchanged:
   `test_background_tau.test_qcd_break_points` and `ComputeTargets/tests/test_cumulative_table.py`
   pass **with their expectations untouched**. If the agent changed what the cumulative tables
   split on, it has broken prompt 02's result — stop.
7. **The acceptance test of §4 passed**: QCD reference drift $\le3.4\times10^{-8}$ of the envelope
   at all 50 wavenumbers. If it did not, the agent should have stopped and reported the
   measurement; if it continued, or reached the threshold by tightening a tolerance, stop here.
   **`config/defaults.py` must be untouched** — no tolerance moves in this prompt, `rtol` included.
8. **The smooth-model regression is exact, not close.** Prompt 17's control figures reproduce at
   2.534e-6 and 2.56e-4 with RHS-evaluation counts **7403 and 8483 unchanged**. "Within a few per
   cent" is a fail: on a cosmology declaring no discontinuities the code path is supposed to be
   the present one.
9. **The $G_k$ arm is present** (§3 item 2). Drift figures for `GkNumericIntegration` on all three
   models, on $G_k$'s own production geometry. This has never been measured and is half the value
   of the prompt; a $T_k$-only report is incomplete.
10. **The diagnostic was preserved** — `has_unresolved_osc`, `unresolved_z`,
    `unresolved_efolds_subh` aggregate across segments rather than reporting the last one
    (`../README.md` §2 (h); losing it is a campaign-wide stop). Likewise `RHS_evaluations` and
    `sol.nfev`.
11. **The §3.1 datastore-key finding is reported and not acted on**, and opened as a §3 issue with
    `docs/OPEN_ISSUES.md` updated; **counts, not wall time** for the cost.
12. **`[17-qcd-reference-not-converged]` is closed only if check 7 passed**, narrowed otherwise.

## Continue or stop

**Stop after this prompt** and report, whatever the outcome — there is a datastore question waiting
on it (§3 item 5 and §3.1) that only the user can settle before prompt 13 runs. Stop early on
checks 5, 6, 7, 8 or 10, or on any campaign-wide condition.

## Completion criterion

Row 18 ✅/⚠️; the new section of `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` present and
additive; the suite at 299 plus the new cases. Report: whether the QCD reference now converges and
at what cost; the $G_k$ drift figures; whether a QCD datastore built before this commit is usable
by prompt 13, and what must be regenerated if not; and the shape the declaration ended up with,
since it is now part of the `CosmologyModels` API.
