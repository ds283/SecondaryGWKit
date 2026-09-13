# Log 19 — Let each numeric sector choose which declared break points it splits at

**Prompt:** prompts/GkTk-remedial/19-per-sector-break-point-policy.md
**Commit:** *(this commit)* — Let each numeric sector choose which declared break points it splits at
**Model:** Claude Opus 5
**Date:** 2026-09-13
**Result:** COMPLETE WITH DEVIATIONS — the §4 acceptance test **passes** at all 50 QCD $T_k$
wavenumbers (worst drift 8.72e-09 against the 3.4e-08 criterion), the $G_k$ and smooth-model
regressions are exact, and the suite is 328 tests OK. Three deviations, one of them
`STRUCTURALLY REQUIRED` — the segment-boundary separation guard of §2.3 item 2, which the
prompt asks for explicitly and which touches code both sectors run through; the exact $G_k$
regression the prompt requires in that case was re-run and is bit-identical.

## What shipped

### `Quadrature/integrators/numeric_with_phase_cut.py` — the parameter

- **`declared_discontinuities_in_z(model, z_lo, z_hi)` → `declared_discontinuities_in_z(model,
  z_lo, z_hi, kind=BREAK_POINT_DISCONTINUITY)`** (`:94`). `kind` is handed straight to
  `_cosmology_break_points`, replacing the hard-coded `BREAK_POINT_DISCONTINUITY` at the old
  `:524`. **Its own default is unchanged**, as prompt §2.1 requires, so
  `TestQCDReferenceConvergence.test_the_branch_crossing_is_inside_the_range` and
  `TestSplitAtDeclaredJump.test_only_the_jump_is_returned_to_the_ode`, which call it with three
  arguments, keep their meaning and their expectations. Docstring rewritten to say what the two
  kinds are and that which one is asked for is the caller's decision.
- **`numeric_with_phase_cut(..., break_point_kind: str = BREAK_POINT_DISCONTINUITY)`** (`:531`),
  **appended last in the signature**, exactly as prompt 16 appended `warn_unresolved_osc`, so no
  positional index moved. The remote gains its first docstring, which says what the parameter
  does, that the default reproduces every pre-prompt-19 call bit for bit, and why both production
  call sites name it anyway. Consumed at `:625`, where `break_z` is now
  `declared_discontinuities_in_z(model, z_min, z_init.z, kind=break_point_kind)`.
- **`_separated_boundaries(interior, z_top, z_bottom) -> List[float]`** (`:166`), new, and the one
  change to the shared driver beyond passing the parameter through (prompt §2.3 item 2). It thins
  the already-descending, already-in-range boundary list so that consecutive boundaries, and the
  two endpoints, are separated by at least `BREAK_POINT_STANDOFF` relative in $(1+z)$; a boundary
  that does not clear its predecessor is dropped, never moved. Called from `_solve_segmented`
  immediately after the standoff is applied and the out-of-range boundaries are discarded. Its
  docstring carries the measurement that says it is inert in production and the reason it exists
  anyway. `BREAK_POINT_STANDOFF` itself is untouched at `1.0e-12`.
- Module docstring: the paragraph that said this module "asks for *jumps only*" is replaced by one
  saying it asks the cosmology and the *caller* chooses the kind, with the per-sector decision and
  its evidence (§9.1, §9.7, §10). `_solve_segmented`'s docstring gains a paragraph on the
  segment-holding-no-requested-sample case (§2.3 item 1), which is now the common case rather than
  an edge.
- **No other change to the driver.** The `num_requested == 0` branch, the
  `len(segment_t) == num_requested + 1` guard, the `sol.y`-empty-list normalisation, the
  terminal-event break, `_SegmentedDenseOutput` and the supervisor's aggregated accounting all
  already handled ~400 segments correctly and were left alone; §2.3 items 1, 3 and 4 are
  discharged by test rather than by edit.

### The two call sites

- **`ComputeTargets/TkNumericIntegration.py`** `:378-412`: imports `BREAK_POINT_ALL` from
  `ComputeTargets.BackgroundModel` (the re-export prompt 18 provided) and passes
  `break_point_kind=BREAK_POINT_ALL`, under a comment block in the shape of the neighbouring
  `warn_unresolved_osc=False` one: necessary by measurement (3 of 50 wavenumbers above the
  criterion with jumps alone, worst 1.97e-07 against 3.4e-08; 4.65e-09 or better with the knots),
  affordable because the sector is one object per $k$, citing §9.7.
- **`ComputeTargets/GkNumericIntegration.py`** `:345-379`: imports `BREAK_POINT_DISCONTINUITY` and
  passes `break_point_kind=BREAK_POINT_DISCONTINUITY` — **the module default, named explicitly**,
  because the point of the prompt is that the choice is a measured decision in both sectors.
  Comment: unnecessary by measurement (converges at all 50 on all three models, worst 8.41e-09),
  and expensive at ~65,000 objects per model (+155 %), citing §9.1 and §9.7.

### `ComputeTargets/tests/test_numeric_break_points.py` — eight new cases

New stand-in `_ManyBreakCosmology(z_jump, z_lo, z_hi, count, coincident=())`: one jump plus
`count` non-jump breaks uniform in $\log(1+z)$, plus optionally a pair placed closer than the
standoff. New helpers `_segmented_solve(model, grid, kind, events=None, dense_output=False)`
(drives `_solve_segmented` directly, the only way to observe the *number* of segments — the
payload deliberately does not carry it) and `_payloads_are_bit_identical(a, b)` (`==` on every
returned double plus the evaluation count).

- `TestPerSectorPolicy.test_the_parameter_selects_the_number_of_segments` — 1 declared point and 2
  segments under `BREAK_POINT_DISCONTINUITY`, 2 and 3 under `BREAK_POINT_ALL`, on a stand-in
  declaring one jump and one kink.
- `…test_the_default_reproduces_the_jumps_only_result_exactly` — argument omitted is bit-identical
  to `BREAK_POINT_DISCONTINUITY`, and *not* to `BREAK_POINT_ALL` (so the first assertion is not
  comparing one code path with itself).
- `…test_a_cosmology_declaring_nothing_takes_the_single_call_path_either_way` — empty declaration
  under both kinds for `RadiationModel`, `LambdaCDMModel` and the non-declaring stand-in; one
  segment; the two policies bit-identical.
- `…test_each_production_call_site_passes_the_kind_its_sector_decided_on` — `ast`, following
  `test_numeric_phase_cut.test_both_integrators_pass_warn_unresolved_osc_False`: exactly one
  `numeric_with_phase_cut.remote(...)` per module, `break_point_kind` present, the right `ast.Name`,
  and that name imported from `ComputeTargets.BackgroundModel`.
- `TestManyBreakPoints.test_most_segments_hold_no_requested_sample` — 402 segments against 200
  samples; assembled `t` equals the requested grid element-for-element under `==`, `y` has the
  matching shape.
- `…test_splitting_where_the_right_hand_side_is_smooth_changes_nothing_material` — the 400 kinks
  are declared where the coefficient is continuous, so the extra restarts cost evaluations and not
  accuracy; `has_unresolved_osc` agrees between policies and the accounting aggregates.
- `…test_boundaries_closer_than_the_standoff_collapse` — three boundaries within 0.2 × the standoff
  of each other collapse to one, the segment count is what it would have been without them, and
  the run still matches the closed form.
- `…test_stop_mode_survives_a_long_chain_of_segments` — terminal event in a late segment stops the
  whole integration, `find_phase_extremum` finds the extremum through a 402-segment composite dense
  output to 1e-6 of the envelope against the closed form, and it is the same extremum the
  jumps-only policy finds.

### `docs/gktk-remedial/`

- `tk_numeric_atol_sweep.py`: **additive third entry point** `--per-sector` →
  `main_per_sector()`. Above the prompt-19 banner only three signatures changed, each gaining
  `break_point_kind` defaulting to `BREAK_POINT_DISCONTINUITY` — what the integrator did
  unconditionally when §9 was taken — so `--break-points` still emits §9: `run`, `run_gk`,
  `reproduce_control`. New below the banner: `SECTOR_POLICY`, `SECTION_9_GK_DRIFT`,
  `_bitwise_equal`, `_matches_to_printed_precision`, `per_sector_sweep`, `timing_experiment`,
  `report_policy`, `report_gk_regression`, `report_smooth_regression`, `report_per_sector_cost`,
  `report_per_sector_shift`, `report_per_sector_detail`, `main_per_sector`.
- `TK-NUMERIC-ATOL-SWEEP.md`: **new §10**, appended. §§1–9 untouched.

## Deviations from the prompt

### 1. The parameter is optional, not required — IMPLEMENTATION CHOICE

Prompt §2.1 states that the default must reproduce today's behaviour and adds "If you think the
parameter should instead be required, say so in the log and ask — do not make it required
unilaterally." **I do not think it should be required, so there is nothing to ask.** The reasons,
for a later reader who may disagree:

- Making it required would change the signature of a function four test modules, four `docs/`
  reproduction scripts and `main.py`'s two integrators reach, for no gain in either production
  path — both of which name it anyway. The prompt's own §4 requires the `docs/` scripts and the
  tests to keep their numbers bit for bit; an optional parameter gives that for free.
- The thing the prompt actually wants — that neither sector looks as though it inherited a policy
  by omission — is delivered by the *call sites* naming it, which they do, and by a test that
  fails if either stops. Requiredness would enforce it one level too low, on callers that have no
  decision to make.
- `warn_unresolved_osc` (prompt 16) set the precedent one prompt earlier, for the same reason.

The default is asserted twice: `test_the_default_reproduces_the_jumps_only_result_exactly` in the
suite, and §10.2's "default matches explicit policy: all 50" for each of the three models in the
sweep.

### 2. One change to the shared driver beyond passing the parameter through — STRUCTURALLY REQUIRED

Prompt §2.3 anticipates this ("If any of this forces a change to the driver beyond passing the
parameter through, that is expected — but … say so plainly in the log and re-run the $G_k$
regression"). `_separated_boundaries` is that change, and it answers §2.3 item 2. Stated plainly:

- **What forced it.** §2.3 item 2 requires establishing whether two production boundaries can lie
  closer than `BREAK_POINT_STANDOFF`, "and if it cannot, say what the minimum spacing actually is
  and guard it anyway". The guard is therefore mandated by the prompt, not chosen.
- **The measurement.** They cannot. The `T(z)` spline's knots are uniform in $u = \log(1+z)$ at
  **2.85e-02**, and the closest a declared temperature crossing comes to a knot anywhere in
  $(z = 0.1,\ 10^{14})$ is **3.4e-03** — nine orders above the 1e-12 standoff. The closest a
  declared break point comes to a *requested sample* is **1.88e-06** relative, six orders above it,
  so no boundary can cross a sample after the standoff either. (Scratch measurement over the
  $T_k$ source and $G_k$ response grids at $k = 10^5$, $5.94\times10^6$ and $3\times10^8$ on
  `QCD_Cosmology`, and over the whole declared range.) The guard is inert on every production
  geometry.
- **Why a guard at all.** With `BREAK_POINT_ALL` a production $T_k$ object is cut at ~125–127
  boundaries rather than one, so "two declared points closer than the standoff" stops being
  hypothetical by inspection of the code; if it happened the standoff would carry one boundary onto
  or past its neighbour and `solve_ivp` would be called over a zero-length or inverted `t_span`.
  A future equation of state now degrades into one boundary instead of a solver error.
- **Dropping, not merging**, because prompt 18 §2.3 item 1's rule — never round a break point onto
  a sample or onto another break point — leaves only "keep it" and "leave it out" as safe repairs.
- **The $G_k$ regression the prompt asks for in this case was re-run and is exact**: §10.2, three
  models, worst drift reproducing §9.1 to the printed digit and the QCD per-object evaluation count
  at 13320 = §9.3's figure, with the argument-omitted run bit-identical at all 50 wavenumbers on
  all three models.

### 3. `MANY_BREAK_COUNT = 400` rather than the production ~125 — IMPLEMENTATION CHOICE

The fixture declares 400 non-jump breaks against 200 samples, where production is ~125 against
~100. Both put the driver in the "most segments hold no sample" regime; 400 exaggerates it, keeps
the ratio at 2:1 rather than 1.25:1, and costs 0.3 s. The alternative — matching production exactly
— would have tested the same branch with less margin. Recorded because the number is arbitrary.

## Verification performed

All figures are from
`PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/tk_numeric_atol_sweep.py --per-sector`, one run,
573.7 s, emitted as §10 of `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md`. 50 wavenumbers × 3 models
× 2 sectors.

### §4's acceptance test — **passes**, on `QCDModel` $T_k$ at all 50 wavenumbers

| | jumps only (§9's policy) | shipped policy (jumps + knots) |
|---|---|---|
| worst drift | **1.97e-07** (at $k = 4.287\times10^6$) | **8.72e-09** (at $k = 1.0\times10^5$) |
| median drift | 5.12e-09 | **3.96e-09** |
| wavenumbers above 3.4e-08 | **3** | **0** |

Prompt §3 item 1's worry — that §9.7 measured three wavenumbers and the other 47 might be
disturbed — does not materialise: the per-$k$ table (§10.6) shows every wavenumber at or below
8.72e-09, and the three §9.7 measured reproduce it exactly (6.89e-10 at $8.366\times10^5$,
**4.65e-09** at $4.287\times10^6$, **2.30e-09** at $4.223\times10^7$ — the same three numbers
§9.7 printed). `BREAK_POINT_STANDOFF` was **not** moved and no tolerance was touched.

### §3 item 2 — the $G_k$ regression, and it is exact

| model | §9.1 | measured now | per-object evals (§9.3: 13320) | argument-omitted run bit-identical |
|---|---|---|---|---|
| RadiationModel | 1.94e-11 | **1.94e-11** | 12743 | all 50 |
| LambdaCDMModel | 2.1e-11 | **2.1e-11** | 12804 | all 50 |
| QCDModel | 8.41e-09 | **8.41e-09** | **13320** | all 50 |

The worst-drift wavenumbers also reproduce ($1.561\times10^8$, $2.197\times10^7$,
$6.034\times10^5$), as do §9.2's four named wavenumbers in the $G_k$ column. §10.4 records the
$G_k$ QCD cost as **13320 → 13320 per object, 666011 → 666011 over the grid, +0.00 %** — the same
integers, not the same to rounding. Nothing moved for the sector that did not ask.

### §3 item 3 — the smooth models

Both models, both sectors, all 50 wavenumbers: **one segment under either policy**, and for $T_k$
(where the two policies are different requests) the production run under `BREAK_POINT_ALL` is
bit-identical to the run under `BREAK_POINT_DISCONTINUITY` at **all 50**. Prompt 17's two control
figures against the exact $T$, under both policies: **2.53e-06 in 7403** evaluations and
**2.56e-04 in 8483** — the expected counts, identical between policies.
`test_a_cosmology_declaring_nothing_takes_the_single_call_path_either_way` makes the same
assertion in the suite.

### §3 item 4 — the cost

Right-hand-side evaluations, `QCDModel`, production tolerances (the reproducible measure):

| sector | per object, jumps only | per object, shipped | grid total, jumps only | grid total, shipped | change |
|---|---|---|---|---|---|
| Tk | 9843 | 31521 | 492158 | 1576030 | **+220.23 %** |
| Gk | 13320 | 13320 | 666011 | 666011 | **+0.00 %** |

Seconds per object, best of **N = 5** single-core runs at $k = 4.972\times10^7$/Mpc, *for scoping
only*: $T_k$ on QCD **0.9805 s** shipped against 0.3188 s jumps-only, i.e. **49.0 s for the whole
sector** at 50 objects per model; $G_k$ on QCD 0.2084 s against 0.2093 s (the difference is timing
noise on an unchanged computation), i.e. ~3.8 core-hours per model either way — which is the
number that makes the asymmetry legible: the same +155 % applied there would have been ~5.9
additional core-hours per model to improve a quantity already at 8.41e-09. The prompt's
expectations of "+219 %" and "~34 s" are met at +220.23 % and 49.0 s; the seconds differ because
this machine is not the one §9.7's estimate was scaled on, which is why the counts are the measure.

### §3 item 5 — how far the $T_k$ answer moves on QCD

Production-tolerance run under the jumps-only policy, scored against the converged reference under
the shipped policy, the same envelope-relative measure as §9.4: **worst 1.61e-04** at
$k = 4.972\times10^7$, median 8.27e-07, smallest 1.71e-07. This is on top of prompt 18's 2.82e-04.
**Consequence, stated explicitly:** `solver_serial` is in neither numeric lookup key
(`[18-numeric-solver-not-in-lookup-key]`, still open and still the user's), so a `QCDModel`
datastore holding `TkNumericIntegration` rows computed before this commit is indistinguishable by
key from one computed after it while differing by up to 1.6e-04 of the envelope. It may not be
used. `GkNumericIntegration` rows are bit-identical and need no regeneration on any model, and
`LambdaCDMModel` and `RadiationModel` rows are bit-identical in both sectors.

### The suite

`PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` →
**`Ran 328 tests in 154.748s` / `OK`**, against the 320 handed over: 8 added, none removed, none
changed in expectation. `test_background_tau.test_qcd_break_points` and `test_cumulative_table.py`
are untouched and pass — the quadrature path is not what this prompt changes, and
`_cosmology_break_points`'s own `BREAK_POINT_ALL` default is unchanged.
`test_numeric_phase_cut.test_both_integrators_pass_warn_unresolved_osc_False` passes. The
machine-load-sensitive `test_tk_wkb_phase.TestCost.test_wall_time_per_object` passed on the single
full-suite run; it was not re-run and its limit was not touched.

`black --check` clean on all six files touched. `git diff HEAD --stat` touches
`Quadrature/integrators/numeric_with_phase_cut.py`, `ComputeTargets/{Tk,Gk}NumericIntegration.py`,
`ComputeTargets/tests/test_numeric_break_points.py`, `docs/gktk-remedial/tk_numeric_atol_sweep.py`,
`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md`, plus the log, the board and `docs/OPEN_ISSUES.md`.
**Not** `config/defaults.py`, **not** `CosmologyModels/`, **not** `Datastore/`.

## Observations not acted on

1. **Two `CosmologyModels/` docstrings now overstate the jump/kink distinction.**
   `GenericEOS.py:97-101` says "an *adaptive* ODE solver only has to be split at a jump, because a
   C2 point does not invalidate an embedded Runge–Kutta error estimator", and
   `LambdaCDM_GenericEOS.py:277-283` says the jumps-only set "is what
   `Quadrature/integrators/numeric_with_phase_cut.py` asks for". Both were true when prompt 18 wrote
   them and both are now half-true: the $T_k$ sector asks for `BREAK_POINT_ALL`, and the
   measurement is that on `QCD_Cosmology` the C2 knots *do* cost about an order of magnitude of
   reference convergence — a $C^2$ point does not invalidate the estimator but it does degrade it.
   Not edited: prompt §1's file list puts `CosmologyModels/` out of bounds and says "the
   declaration is finished". Opened as `[19-cosmologymodels-docstrings-predate-per-sector-policy]`.

2. **The knots cost an order of magnitude more than an adaptive stepper "absorbing" them suggests.**
   The shipped $T_k$ policy is +220 % of the evaluations for a factor ~23 in worst drift
   (1.97e-07 → 8.72e-09). Prompt 18's framing — kinks are absorbed at the cost of a few extra
   steps — is right about *steps* and wrong about *convergence*: what the restart buys is that
   DOP853's embedded estimator sees a smooth right-hand side over every step, which matters to the
   reference-convergence test even where it barely matters to the answer. The answer itself moves
   by only 1.6e-04 of the envelope (§10.5). Recorded because it is the physical content of the
   number, not an action.

3. **`_SegmentedDenseOutput.__call__` is a linear scan.** With ~400 segments and a root-find that
   evaluates the dense output tens of times, this is ~10⁴ comparisons per object — invisible next
   to 31,521 right-hand-side evaluations, and it is a list of tuples, so a bisection would need the
   boundaries kept separately. Left alone; measured cost is in §10.4 and it is not visible there.

4. **§10.4's seconds are a single machine's.** Best-of-5 removes scheduler noise but not the
   machine. The $G_k$ shipped-vs-jumps columns differ by 0.4 % on a computation that is
   bit-identical, which calibrates how much of the $T_k$ column to believe. The counts are the
   measure and the document says so twice.

## State handed to the next prompt

**Prompt 13 may still not use a `QCDModel` datastore built before this commit, and the reason has
widened rather than changed.** Prompt 18 said no because the split moved `TkNumericIntegration` and
`GkNumericIntegration` values on that model by up to 2.82e-04 of the envelope. This commit moves
`TkNumericIntegration` on `QCDModel` a *second* time, by up to **1.61e-04** of the envelope (worst
at $k = 4.972\times10^7$; median 8.27e-07), and `solver_serial` is still in neither numeric lookup
key. What has to be regenerated is therefore what prompt 18 named, minus one sector: every
`TkNumericIntegration` row on `QCDModel` and everything downstream of it —
`TkWKBIntegration` takes its initial data from the numeric stop point, so the whole QCD transfer
chain follows. **`GkNumericIntegration` is new information: its rows are bit-identical across this
commit on all three models** (§10.2, §10.4: 13320 → 13320 per object, +0.00 %), so anything
regenerated for prompt 18 stays valid for the Green's-function sector. `LambdaCDMModel` and
`RadiationModel` rows are bit-identical in both sectors. Nothing in the schema or the payload keys
moved, and no tolerance changed.

**The per-sector policy, now part of `numeric_with_phase_cut`'s public signature:**

```python
# Quadrature/integrators/numeric_with_phase_cut.py
def declared_discontinuities_in_z(
    model, z_lo, z_hi, kind: str = BREAK_POINT_DISCONTINUITY
) -> List[float]: ...

@ray.remote
def numeric_with_phase_cut(
    ...,                                   # unchanged, in order
    warn_unresolved_osc: bool = True,
    break_point_kind: str = BREAK_POINT_DISCONTINUITY,   # NEW, appended last
) -> dict: ...

def _separated_boundaries(interior, z_top, z_bottom) -> List[float]: ...   # NEW, private

# ComputeTargets/TkNumericIntegration.py   ->  break_point_kind=BREAK_POINT_ALL
# ComputeTargets/GkNumericIntegration.py   ->  break_point_kind=BREAK_POINT_DISCONTINUITY
```

The default is `BREAK_POINT_DISCONTINUITY`, so every caller that does not name it — the four
`docs/gk-wkb-review-fable-2026-09-09/` scripts, `test_numeric_phase_cut.py`,
`test_tk_numeric_atol.py`, a future integrator — keeps its numbers bit for bit, and §9's figures
remain valid for the sector that kept them. `BREAK_POINT_STANDOFF` is unchanged at `1.0e-12`;
`config/defaults.py` is unchanged (`atol = 1e-10`, `DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13`,
`rtol = 1e-8`); `CosmologyModels/` and `Datastore/` are unchanged.

**The cost, so prompt 13 can budget a pipeline run.** Per QCD object at production tolerances:
$T_k$ **9843 → 31521** right-hand-side evaluations (+220.23 %), $G_k$ **13320 → 13320** (+0.00 %);
grid totals 492,158 → 1,576,030 and 666,011 → 666,011. In seconds, best of 5 at
$k = 4.972\times10^7$/Mpc on one core: a $T_k$ QCD object **0.98 s** (0.32 s under the old policy),
so ~**49 s for the whole sector per model**; a $G_k$ QCD object 0.21 s, unchanged, ~3.8 core-hours
per model. On `LambdaCDMModel` and `RadiationModel` nothing moves: 0.048–0.078 s per object in
either sector, identical under either policy. **The counts are the reproducible measure; the
seconds are one machine's.**

**One change was forced on the shared driver** (deviation 2): `_separated_boundaries`, which drops
a segment boundary that does not clear its predecessor or an endpoint by `BREAK_POINT_STANDOFF`
relative in $(1+z)$. It is inert on every production geometry — the measured minimum separation of
declared points on `QCD_Cosmology` is 3.4e-03 in $\log(1+z)$, nine orders above the standoff — and
the $G_k$ regression above proves the sector that did not change its policy did not change its
numbers.

**`[17-qcd-reference-not-converged]` is CLOSED** (§4 of the board). The acceptance test passes at
all 50 QCD $T_k$ wavenumbers: worst drift 8.72e-09 against the 3.4e-08 criterion, median 3.96e-09,
zero offenders. `prompts/tolerance-convergence` is now unblocked without a caveat on QCD: both
sectors converge at all 50 wavenumbers on all three models.

**`[18-numeric-solver-not-in-lookup-key]` remains open and remains the user's**, and this commit is
its second instance. **`[19-cosmologymodels-docstrings-predate-per-sector-policy]` is opened**: two
`CosmologyModels/` docstrings say the ODE path wants jumps only, which is now true of one sector
and not the other; out of bounds for this prompt.
