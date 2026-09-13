# Orchestrator prompt — Prompt 19 (per-sector break-point policy)

You are orchestrating prompt 19 of the Gk/Tk WKB phase remedial campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `gktk-remedial`). You do not write code
yourself.

Not a workstream — the decision prompt 18 §4 reserved for the user, taken 2026-09-13. Prompt 18
split the numeric ODE at the cosmology's declared *jumps* and left `QCDModel`'s $T_k$ reference
above the convergence criterion at 3 of 50 wavenumbers; closing it needs the $T(z)$ spline's $C^2$
knots as well, at +219 % ($T_k$) / +155 % ($G_k$) of the production evaluations. The user's ruling:
the cosmology declares everything and each *consumer* decides, so the reusable integrator takes the
choice as an argument — $T_k$ splits at jumps and kinks (necessary, by measurement), $G_k$ at jumps
only (unnecessary, by measurement), each with a note saying so.

**It runs before prompt 13**, for prompt 18's reason: the split changes computed $T_k$ values on
QCD and `solver_serial` is in neither numeric lookup key
(`[18-numeric-solver-not-in-lookup-key]`), so a QCD datastore built before this commit cannot be
told apart from one built after it.

**This prompt changes production code** — the same shared integrator as prompt 18, plus both
production call sites. Review it as such.

## What to read

The prompt: [`../19-per-sector-break-point-policy.md`](../19-per-sector-break-point-policy.md), in
full. `../README.md` §2 (d) and (h), §5, §6; `../IMPLEMENTATION_STATE.md` row 18, M22, and the
`[17-qcd-reference-not-converged]` and `[18-numeric-solver-not-in-lookup-key]` entries in §3;
[`README.md`](README.md); `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` §9 (§9.1, §9.3, §9.7
especially); `logs/18-numeric-ode-break-points.md`, "State handed to the next prompt".

## Preconditions

`git status` clean; row 18 ⚠️ and row 19 ⬜. Baseline **320 tests**. Note that
`test_tk_wkb_phase.TestCost.test_wall_time_per_object` is a machine-load-sensitive wall-clock
budget with no margin (0.0603–0.0681 s against a hard-coded 0.06 s) and fails intermittently on a
clean tree: measure it before dispatch so you can tell it from a regression, and tell the subagent
to leave it alone. Confirm the shape prompt 18 left:

```bash
grep -n "BREAK_POINT_ALL\|BREAK_POINT_DISCONTINUITY" CosmologyModels/GenericEOS/GenericEOS.py ComputeTargets/BackgroundModel.py
grep -n "declared_discontinuities_in_z" Quadrature/integrators/numeric_with_phase_cut.py
grep -n "numeric_with_phase_cut.remote" ComputeTargets/TkNumericIntegration.py ComputeTargets/GkNumericIntegration.py
```

## Dispatching

Standard dispatch text (`workstream-A.md`), with `NN-<name>` = `19-per-sector-break-point-policy`.
Model: **Opus**.

## Reviewing prompt 19

The five checks of `../README.md` §4.3, plus:

4. Tests: `discover -s ComputeTargets/tests -t .` → **320 plus the new cases**, none removed and no
   existing expectation changed. If a pre-existing test's expected value moved, that is a stop:
   $G_k$ and both smooth models are supposed to be bit-for-bit unchanged.
5. **The generic property survived.** The caller chooses a *kind*, never a point:
   ```bash
   git diff HEAD~1 HEAD -- Quadrature/ ComputeTargets/ | grep -in "qcd\|saikawa\|T_LO\|T_HI\|120_MEV\|MeV"
   ```
   must be empty of anything but comment prose citing the campaign and the new tests' own
   fixtures. A hard-coded break point is a stop. `CosmologyModels/` must not appear in the diff at
   all — the declaration was finished by prompt 18.
6. **The $G_k$ regression is exact, not close.** §9's $G_k$ figures reproduce: 1.94e-11, 2.1e-11,
   8.41e-09 worst over the grid on the three models, and 13320 evaluations per QCD object. $G_k$
   did not change its policy, so if a $G_k$ number moved, the shared driver changed behaviour for a
   sector that did not ask — stop.
7. **The smooth-model regression is exact.** 2.534e-6 and 2.56e-4 with RHS counts **7403 and 8483**
   unchanged, under *either* policy, and 1 segment at every smooth wavenumber.
8. **The acceptance test of §4 passed**: QCD $T_k$ drift $\le3.4\times10^{-8}$ at **all 50**
   wavenumbers — not the 3 prompt 18 measured. If it did not, the agent should have stopped and
   reported; if it continued, or reached the threshold by tightening a tolerance or moving
   `BREAK_POINT_STANDOFF`, stop here. **`config/defaults.py` must be untouched.**
9. **Both call sites pass the argument explicitly and say why** (§2.2), $G_k$'s included even
   though it is the default, each citing the measurement. A $G_k$ site that relies on the default,
   or a comment that asserts the asymmetry without pointing at §9.1/§9.7, is the thing this prompt
   exists to prevent.
10. **The many-segment hazards of §2.3 were addressed by measurement, not assertion** — empty
    segments, near-coincident boundaries, a break point landing on a sample, `mode="stop"` through
    ~400 dense segments. An agent that says these "should be fine" without a test or a number has
    not done §2.3.
11. **The diagnostic was preserved** — `has_unresolved_osc`, `unresolved_z`,
    `unresolved_efolds_subh` aggregate across segments rather than reporting the last
    (`../README.md` §2 (h); losing it is a campaign-wide stop). Likewise `RHS_evaluations` and
    `sol.nfev`.
12. **The cost is reported in counts first** (§5 note 14), with seconds as an explicitly secondary,
    scoping-only figure. Counts-only is incomplete here — the user asked for the seconds — but
    seconds presented as the measure is wrong.
13. **`[17-qcd-reference-not-converged]` is closed only if check 8 passed**, narrowed otherwise;
    the `prompts/tolerance-convergence` paragraph in `docs/OPEN_ISSUES.md` §1 is updated with it,
    since it currently records the three-wavenumber caveat.

## Continue or stop

**Stop after this prompt** and report. `[18-numeric-solver-not-in-lookup-key]` is still open and
still the user's, and prompt 13 needs it settled along with the QCD datastore question. Stop early
on checks 5, 6, 7, 8 or 11, or on any campaign-wide condition.

## Completion criterion

Row 19 ✅/⚠️; the new section of `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` present and
additive; the suite at 320 plus the new cases. Report: whether the QCD $T_k$ reference now
converges at all 50 wavenumbers and at what cost in evaluations and in seconds; that $G_k$ and both
smooth models are unchanged; whether §2.3 forced any change to the shared driver; and what prompt
13 must regenerate.
