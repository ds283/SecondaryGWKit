# Log 01 — The convergence harness, and one production grid

**Prompt:** `prompts/tolerance-convergence/01-convergence-harness-and-grid.md`
**Commit:** *(this commit)* — Build the convergence harness and name the source-grid generations
**Model:** Opus 5
**Date:** 2026-09-16
**Result:** COMPLETE WITH DEVIATIONS — the facility and the hoist both land, the version-0
reproduction is **bit-identical over the whole of `tk_numeric_atol_sweep.py`'s output** rather than
merely to the published digits, and the version-2 statistic is reported for all three models. Six
deviations, none of them numerical: two `STRUCTURALLY REQUIRED` because the prompt's §3.2 and §3.3
each describe something that is not in the tree, one more forced by the new issue below, and three
`IMPLEMENTATION CHOICE`. **One new issue, and it is not small**: `main.source_grid_spacing_profile`
**raises** on `QCD_Cosmology` when the source grid is anchored where a QCD production run anchors
it, so the version-2 QCD row is taken at the LambdaCDM anchor that every recorded figure uses.

---

## What shipped

### 1. `ComputeTargets/tests/convergence_reference.py` — new, the facility (board item T1)

The public API, in full, because prompts 03, 04 and 06 are written against it.

**The knobs.** `AccuracyKnob` is the base: `tighter(steps=1)`, `looser(steps=1)`, `label`,
`ladder(steps)` (loose to tight, README §6.1 rule 3). Two implementations:

| | step | notes |
|---|---|---|
| `TolerancePair(atol, rtol, axis="both")` | **one decade**, as an exact decimal shift | `axis` is `"both"`, `"atol"` or `"rtol"`; `rtol_step_is_effective` says whether SciPy would clamp it |
| `GaussOrder(order, name="N")` | **one order** (`+1`) | for $N_\tau$, $N_{c_s\tau}$, $N_F$, $N_\rho$ |

`TolerancePair(1e-18, 1e-12).tighter()` is `(1e-19, 1e-13)` exactly and not
`(1.0000000000000001e-19, 1e-13)`: the step is `Decimal(repr(value)).scaleb(-steps)`, because an
accuracy parameter is part of its object's datastore lookup key (README §2 (g)) and a value one
ulp from the one a later prompt writes into `config/defaults.py` is a different key.

**The drift, with its verdict attached.**

```python
reference_drift(build, knob, *, error_measure, smallest_reported_difference,
                criterion_ratio=CRITERION_RATIO, reference=None,
                tightened_knob=None) -> DriftVerdict
converged_reference(build, knob, *, error_measure, smallest_reported_difference,
                    criterion_ratio=CRITERION_RATIO,
                    tightened_knob=None) -> ConvergedReference
```

`build(knob)` is the caller's — it returns whatever the caller's `error_measure(candidate,
reference)` consumes, so the same two functions serve a solver payload, a cumulative table and a
phase. `smallest_reported_difference` is **required and has no default**; that is README §5 rule 5
expressed as a signature. `DriftVerdict` carries `knob`, `tightened`, `drift` (the full
`summarise` dict), `smallest_reported_difference`, `criterion_ratio` and `notes`, and answers
`threshold` (= smallest / 10), `passed`, `max`, `median`, `max_z`, `headroom`. There is no public
route to a drift number that does not also hand over the verdict. `ConvergedReference` adds
`payload`, `converged`, `__bool__`, `errors(candidate)` and `score(candidate)`, the last of which
returns the summary with `reference_drift` and `reference_converged` already in it.

`CRITERION_RATIO = 10.0` is a *ratio*, not a threshold; the threshold is always the caller's own
smallest reported difference divided by it. `SCIPY_RTOL_FLOOR = 2.220446049250313e-14` is
`100 * eps`, SciPy's own clamp (`scipy/integrate/_ivp/common.py:47-51`), and a reference or
tightening below it is recorded in `DriftVerdict.notes` rather than silently measuring nothing.

**The anchors** (README §3.1's table, all of it). `radiation_anchors(model)` returns
`{name: (kind, callable)}` for the eleven quantities `T`, `Tprime`, `G`, `Gprime`, `tau`,
`cs_tau`, `friction_F`, `theta_G`, `rho_G`, `rho_T`, `z_exit`; `kind` is `"value"` (scored
envelope-relative), `"interval"` (difference error, and the accessor is the *delta*, never a
difference of primitives) or `"phase"` (radians). `anchor_error(kind, value, exact, envelope=None)`
picks the measure the row calls for. Two of the eleven are new here:
`exact_z_exit(H0, k_inv_Mpc, efolds_subh=0) = k/(H0 e^N) - 1`, and `rho_G ≡ 0`, which makes the
Green's-function residual a pure quadrature-error measurement with no reference to build.
`analytic_G`, `analytic_Gprime`, `analytic_T`, `analytic_Tprime` wrap
`ComputeTargets/analytic_{Gk,Tk}.py` in the harness's variables, at the model's own constant $w$.

**The error measures** are `wkb_reference.py`'s, re-exported through `ERROR_MEASURES` and used by
`anchor_error` and `sector_errors`. No fourth definition was written.

**The two numeric sectors**, folded from `docs/gktk-remedial/tk_numeric_atol_sweep.py`:
`_Wavenumber`, `_KExit`, `_Proxy`, `tk_geometry`, `gk_geometry`, `tk_run`, `gk_run`, `SECTORS`,
`x_local`, `sector_errors`, `summarise`, plus `sector_error_measure(sector, model, k, geo)` and

```python
sector_reference(sector, model, cosmology, k_inv_Mpc, grid, knob, *,
                 smallest_reported_difference, criterion_ratio=CRITERION_RATIO,
                 break_point_kind=BREAK_POINT_DISCONTINUITY,
                 geo=None) -> (geo, ConvergedReference)
```

which is the whole of README §0.2's self-convergence test for one (sector, model, $k$, grid
generation) in one call.

**The grid is named, not defaulted.** `SourceGridSpec(generation, universal, z_init, z_end,
samples_per_log10z, k_inv_Mpc)` is what a geometry is built from, and `tk_geometry` /
`gk_geometry` take a built one as a **required positional argument**. `universal` is the second
axis and is not the same question as the generation: `universal=False` is prompt 17's one lattice
per wavenumber (available on version 0 only, because it is the only construction that consults no
cosmology), `universal=True` is `main.py`'s one grid per model, truncated per work item
(`main.py:1209-1211`). Two module constants are provided: `V0_PER_K_GRID` and
`V2_PRODUCTION_GRID`. `spec.build(cosmology)` returns a `BuiltSourceGrid` whose `generation`,
`samples`, `digest` (`redshift_grid_digest`) and `label` go straight into a table, which is
README §5 rule 6 made mechanical.

### 2. `ComputeTargets/tests/test_convergence_reference.py` — new, 32 tests

Six classes: the knob's step; the drift-and-verdict constraint; the named generations against the
campaign's recorded lengths and digests; that the fold did not move the geometry; the constant-$w$
anchors; and one end-to-end use on the control with the oracle beside it.

### 3. `ComputeTargets/tests/wkb_reference.py` — the grid helpers only (board item T2)

New constants: `PRODUCTION_NUMBER_K_VALUES = 50`, `PRODUCTION_K_GRID_INV_MPC` (the fifty
production wavenumbers), `PRODUCTION_Z_INIT = 2.0636395964161516e16`, `SOURCE_GRID_V0` / `_V1` /
`_V2`, `SOURCE_GRID_GENERATIONS`, `SOURCE_GRID_V2_REPRODUCES_VERSION = 2`.

New functions:

```python
main_py_grid_helpers() -> dict          # cached lift of main.py's three grid helpers
cosmology_grid_features(cosmology, z_end, z_init) -> (break_z, feature_z)
source_grid_spacing(cosmology, base_z_values, k_inv_Mpc) -> (u_profile, h_profile)
source_grid(generation, z_init, z_end, samples_per_log10z, *, cosmology, k_inv_Mpc) -> SourceGrid
source_grid_z_values(generation, ...) -> np.ndarray
source_grid_redshifts(generation, ...) -> redshift_array
```

`source_grid` has **no default generation** and refuses a cosmology for version 0 and refuses the
absence of one for versions 1 and 2. `production_source_grid` keeps its behaviour exactly and its
docstring now opens "**The version-0 source grid**" and says why the name survives (deviation 1).

### 4. The three call sites repointed

| Site | Before | After |
|---|---|---|
| `test_source_grid.py:127` `_production_grid` | the full v2 construction, private to that module | `source_grid(SOURCE_GRID_V2 or _V1, ...)` |
| `test_background_segmentation.py:90` `production_source_grid` | its own `build_z_sample` call with the v1 arguments | `cosmology_grid_features` + `source_grid(SOURCE_GRID_V1, ...)` |
| `tk_numeric_atol_sweep.py` `geometry` / `gk_geometry` | the geometry inline, on an unnamed grid | `tk_geometry` / `gk_geometry` with `V0_PER_K_GRID`, and the machinery imported back |

`test_source_grid.py:152` `_production_base_grid` is untouched, as the prompt requires: it mirrors
`main.py:944`'s own base-grid step.

---

## Deviations from the prompt

### 1. `production_source_grid` is documented as version 0 rather than made to fail loudly — STRUCTURALLY REQUIRED

**What the prompt assumed.** §3.2: "repoint #1's callers, including `tk_numeric_atol_sweep.py:215`,
at the named **v0** helper", and "if you keep that name, it must fail loudly rather than default".

**What is actually there.** `wkb_reference.production_source_grid` has **28 call sites in 20
files** outside the prompt's "files you may create or touch" list — 11 test modules under
`ComputeTargets/tests/` and 9 scripts under `docs/gktk-remedial/` and
`docs/qcd-background-audit/`. Every one of them is version 0 by intent: their published figures
were scored there. Making the name fail loudly, or removing it, edits all twenty, which is scope
creep of exactly the kind README §5 rule 4 forbids and which would put the prompt's revert
boundary around twenty files it was not given.

**What was done instead.** The name keeps its behaviour and gains a docstring that says, in its
first line, that it is the version-0 grid, that it is not "the production grid" and has not been
since `SOURCE_GRID_CONSTRUCTION_VERSION` reached 1, and that new callers name a generation. The
three sites the prompt lists are repointed. **The issue is therefore narrowed, not closed**, and
the residue — 28 call sites on the bare name — is recorded on the board and in
`docs/OPEN_ISSUES.md` with the count, so that a later prompt given those files can finish it.

### 2. The prompt's stand-in for the lifted grid code does not exist — STRUCTURALLY REQUIRED

**What the prompt assumed.** §3.3: "`test_source_grid.py:185-203` already carries one that can
[answer both equality redshifts], and says in terms that it is deliberately not a `BaseCosmology`.
**Reuse it** — hoist it with the grid rather than writing a second."

**What is actually there.** `test_source_grid.py:185-203` is `_BrokenCosmology`, whose own
docstring is "A cosmology that declares non-smoothness but **cannot** say where its own equality
redshifts are", and whose only use is
`test_a_cosmology_that_cannot_answer_raises_instead_of_falling_back`. It is the stand-in that hits
the `RuntimeError`, not one that clears it. There is no stand-in in that module that both declares
break points and answers the two attributes.

**What was done instead.** Nothing was hoisted, because nothing needed to be: the grid helpers take
the **real** cosmology (`QCD_Cosmology`, `LambdaCDM`), which is what `main.py` hands them and which
answers both attributes. No fallback was added to `cosmology_feature_redshifts` and the
`RuntimeError` was never reached. The one adapter written is
`convergence_reference._GridCosmologyView`, a view that forwards `wPerturbations` from
`model.functions` for `RadiationModel` — which declares no break points, so it returns
`([], [])` from `cosmology_feature_redshifts` and never reaches the equality-redshift path at all.
It supplies no value the wrapped object does not already have.

### 3. `main.py`'s grid helpers are lifted with a function-local import of `ComputeTargets.phase_residual` — IMPLEMENTATION CHOICE

`wkb_reference.py`'s header states that the module "deliberately imports nothing from …
`ComputeTargets/phase_residual.py`", because a reference built from the object under test conceals
common error. The version-2 construction needs `phase_residual_integrand` and
`residual_node_range` to lift `source_grid_spacing_profile`.

**Alternatives.** (a) Import at module scope and amend the header — rejected: the statement is
load-bearing for the phase references and a reader should not have to work out that the exception
is only about the grid. (b) Put the generations in a new module — rejected: README §3's file table
and prompt §3.2 both put the grid helpers in `wkb_reference.py`, and a fourth module named
"production grid" is the defect being removed. (c) **Chosen**: `main_py_grid_helpers()` does the
imports inside itself, cached, with a docstring that says why the invariant is unaffected — the
phase residual enters only as the criterion that sets the grid's *density*, which is a property of
the geometry every candidate and every reference share, never as a reference value. It also keeps
`main.py`'s `ast` lift off the import path of the twenty-odd modules that import
`wkb_reference.py` for other reasons.

### 4. The script keeps two-argument `geometry` / `gk_geometry` wrappers — IMPLEMENTATION CHOICE

The facility's `tk_geometry(cosmology, k, grid)` takes the grid as a required argument, so a caller
must name its generation. `tk_numeric_atol_sweep.py` reaches `geometry` / `gk_geometry` at nine sites
across its three entry points — four direct calls and five through
`SECTORS[sector]["geometry"]` — two of which (`main_break_points`, `main_per_sector`) are
`GkTk-remedial` prompts 18–20's and which prompt §2.2 says are "not yours to restructure". Its
`geometry` and `gk_geometry` are therefore one-line wrappers that pass `V0_PER_K_GRID`, with
docstrings saying that every figure the script publishes is version 0 and that production has not
been. The alternative — editing the nine call sites — changes prompts 18–20's code for no
measurement.

### 5. `summarise` no longer raises on a single-sample error list — IMPLEMENTATION CHOICE

Folded unchanged except that `"second"` reports the maximum again when there is one sample, instead
of `IndexError`. Nothing the fold has to reproduce ever sees one — both sectors sample hundreds —
but a Gauss order scored at a single node does, and prompt 04 will. The change cannot move a
published figure: it is reachable only on inputs the old code refused.

### 6. The version-2 QCD row is anchored at LambdaCDM's `z_init` — STRUCTURALLY REQUIRED

Forced by the defect in "Observations not acted on" item 1 below. `main.source_grid_spacing_profile`
**raises** on `QCD_Cosmology` when the grid starts at that cosmology's own `z_exit_suph_e5` for
$k = 3\times10^8$, which is where `main.py:944` starts it in a QCD run. The row is taken at
`PRODUCTION_Z_INIT = 2.0636395964161516e16`, LambdaCDM's anchor, which is where **every** recorded
version-2 figure was taken — including `test_source_grid.py`'s 1,996 samples and digest
`4849552b`. The table below says so in its own `z_init` column. No production file was touched.

---

## Verification performed

### V1. §4.1, the construction check — reproduce prompt 17 on prompt 17's grid

Stronger than the prompt asks. `docs/gktk-remedial/tk_numeric_atol_sweep.py` was run to completion
on this tree and on a `git worktree` of `9f04f3a` (the commit before this one), and the two
outputs **diff to two lines, both of which are the wall clock the script prints about itself**
(292 s before, 300 s after). Every measured number in all 294 lines — §3's control, §4's reference
convergence, §5's three per-model tables at all 50 wavenumbers, §6's costs, §7's `rtol` ladder and
$k$-sensitivity, §8's initial-condition floor — is identical.

The two rows the prompt names as the reproduction target, against `TK-NUMERIC-ATOL-SWEEP.md` §4:

| model | published worst drift | measured | at $k$ [1/Mpc] | published median | measured median |
|---|---|---|---|---|---|
| `RadiationModel` | **4.21e-11** | **4.21e-11** | 3e+08 (published 3e+08) | **1.92e-11** | **1.92e-11** |
| `LambdaCDMModel` | **5.7e-11** | **5.7e-11** | 1.561e+08 (published 1.561e+08) | **3.76e-11** | **3.76e-11** |

The prompt 12 control (§3 of the same document) also reproduces: 2.534e-6 at $k = 10^6$ against a
published 2.53e-6 (0.02 % off, 7,403 RHS evaluations) and 2.560e-4 at $k = 3\times10^8$ against a
published 2.56e-4 (0.01 % off, at $x = 10.78$ against a published ~10.8, 8,483 evaluations).

**`QCDModel`, which is not a reproduction target.** Measured on this tree, version 0 per-k, under
the module default `BREAK_POINT_DISCONTINUITY`: **5.57e-09**, worst at $k = 2.23\times10^6$/Mpc,
median over the grid 1.57e-09, smallest candidate error reported 3.52e-07, **criterion met**. The
three published values it stands beside:

| figure | value | tree it was taken on |
|---|---|---|
| `TK-NUMERIC-ATOL-SWEEP.md` §4 | 6.17e-06, criterion **missed** at 4 of 50 $k$ | before `GkTk-remedial` prompt 18 split the ODE at the declared discontinuities |
| `TK-NUMERIC-ATOL-SWEEP.md` §9.1 | 1.97e-07, criterion missed at 3 of 50 $k$ | after the split, before `qcd-background-audit` replaced the $T(z)$ representation |
| board / `PER-SECTOR-POLICY.md` §2 | 7.08e-09, zero offenders | `acd5b8e`, under the sector's own `BREAK_POINT_ALL` |
| **this run** | **5.57e-09**, zero offenders | this commit, under `BREAK_POINT_DISCONTINUITY` — and bit-identically at `9f04f3a` |

Nothing was tuned to land on any of them, and the run that produced it is bit-identical to the one
the previous commit produces.

### V2. §4.2, the tree check — the same statistic on version 2

Taken through the facility's own public API (`SourceGridSpec`, `sector_reference`,
`ConvergedReference.score`), transfer-function sector, reference `(atol, rtol) = (1e-18, 1e-12)`
one decade tighter, all 50 production wavenumbers, all three models, under
`BREAK_POINT_DISCONTINUITY` throughout so that **the grid generation is the only variable**. The
criterion's denominator is each row's own smallest candidate error over
`atol ∈ {1e-10, 1e-13, 1e-16}` at `rtol = 1e-8`, as prompt 17 computes it.

The middle row of each model is the control that separates the two things that change at once: a
version-0 grid *rebuilt per wavenumber* (prompt 17's) against the **same generation** built as one
universal grid and truncated per work item (`main.py`'s shape).

| model | grid generation | $z_{\rm init}$ | universal samples | digest | worst reference drift | at $k$ [1/Mpc] | its median | median drift over the grid | smallest candidate error reported | drift $\le\frac1{10}$ of it? |
|---|---|---|---|---|---|---|---|---|---|---|
| `RadiationModel` | **v0 per-k** | 4.4524e+10 | — | — | **4.21e-11** | 3e+08 | 3.54e-12 | **1.92e-11** | 1.32e-07 | yes |
| `RadiationModel` | v0 universal | 4.4524e+10 | 1,165 | `6f0507b3` | 4.17e-11 | 3e+08 | 3.34e-12 | 1.98e-11 | 1.32e-07 | yes |
| `RadiationModel` | **v2 universal** | 4.4524e+10 | 2,306 | `3bef2c06` | **4.26e-11** | 8.118e+07 | 3.87e-12 | **2.08e-11** | 1.35e-07 | yes |
| `LambdaCDMModel` | **v0 per-k** | 2.0636e+16 | — | — | **5.7e-11** | 1.561e+08 | 4.66e-12 | **3.76e-11** | 3.73e-07 | yes |
| `LambdaCDMModel` | v0 universal | 2.0636e+16 | 1,732 | `0960e169` | 5.23e-11 | 1.325e+08 | 3.74e-12 | 3.82e-11 | 3.72e-07 | yes |
| `LambdaCDMModel` | **v2 universal** | 2.0636e+16 | 1,778 | `60a3205a` | **5.23e-11** | 1.325e+08 | 3.74e-12 | **3.82e-11** | 3.72e-07 | yes |
| `QCDModel` | **v0 per-k** | 3.3003e+16 | — | — | **5.57e-09** | 2.23e+06 | 3.88e-09 | **1.57e-09** | 3.52e-07 | yes |
| `QCDModel` | v0 universal | 3.3003e+16 | 1,752 | `1d8c159e` | 8.48e-09 | 5.048e+06 | 5.62e-09 | 1.72e-09 | 3.52e-07 | yes |
| `QCDModel` | **v2 universal** | **2.0636e+16** (deviation 6) | 1,996 | `4849552b` | **7.11e-09** | 3.092e+06 | 4.86e-09 | **1.23e-09** | 4.04e-07 | yes |

Per-object sample counts and cost, same runs:

| model | grid | samples per work item | 50 $k$ in |
|---|---|---|---|
| `RadiationModel` | v0 per-k / v0 universal / **v2** | 486 / 485–486 / **771–1,607** | 25.9 s / 25.1 s / 26.3 s |
| `LambdaCDMModel` | v0 per-k / v0 universal / **v2** | 486 / 485–486 / **485–532** | 31.7 s / 32.5 s / 31.4 s |
| `QCDModel` | v0 per-k / v0 universal / **v2** | 501–504 / 500–504 / **603–741** | 154.0 s / 154.0 s / 161.5 s |

**What the difference is, and it is not explained away.**

* **The statistic barely moves, and the criterion is met on every one of the nine rows.**
  Version 0 to version 2: 4.21e-11 → 4.26e-11 on radiation, 5.7e-11 → 5.23e-11 on LambdaCDM,
  5.57e-09 → 7.11e-09 on QCD. Every figure is two to four orders below its own row's smallest
  reported candidate difference. **The tolerance figures this campaign inherits were not made
  wrong by the grid moving** — which is a result, and it was not knowable before the measurement:
  `[00-three-production-grid-reproductions]`'s impact statement says a tolerance chosen from those
  figures would be chosen for a grid production does not use, and the answer is that on this
  statistic it would have been the same choice.
* **The two changes are separable and the control row is what separates them.** On LambdaCDM the
  v0-universal and v2 rows are *identical to three figures in every column* — the density
  criterion adds 46 samples in 1,732 and moves nothing — while v0-per-k to v0-universal moves the
  worst drift by 8 % and relocates it from $k = 1.561\times10^8$ to $1.325\times10^8$. On
  LambdaCDM the whole of the difference is **per-k versus universal**, not the generation.
* **On QCD both figures improve, and the comparison is the weakest of the three.** Against the
  v0-universal row the worst drift falls 8.48e-09 → 7.11e-09 and the median over the grid falls
  1.72e-09 → 1.23e-09 (−28 %), with the worst moving from $k = 5.048\times10^6$ to
  $3.092\times10^6$. **But the two rows have different `z_init`** — 3.3003e+16 against 2.0636e+16,
  forced by deviation 6 — so the generation is not the only thing that changed and this row does
  not carry the weight the other two do. What it does establish is that the version-2 grid is not
  *worse* on QCD, and that the drift stays two orders below the smallest reported candidate
  difference either way.
* **Radiation is the row where version 2 costs something.** 2,306 samples against 1,165, and
  771–1,607 per work item against a flat 486, for a drift that moves from 4.17e-11 to 4.26e-11.
  The criterion has no late-time structure to spend samples on there and spends them on the
  transfer-function band instead. `RadiationModel` is a control, not a production model, so this
  is a note and not a finding.


### V3. §4.3, the mechanical checks

| Check | Threshold | Measured |
|---|---|---|
| Hoisted v2 against `test_source_grid._production_grid` before the move | **bit-identical** on `QCD_Cosmology` and `LambdaCDM` | **bit-identical** — SHA-256 of the raw `float64` bytes matches on both, as does the protected-point array on QCD. QCD **1,996** samples, digest `4849552b`; LambdaCDM **1,778**, digest `60a3205a` |
| Named v1 against `test_background_segmentation`'s construction | bit-identical; that module's assertions pass unmodified | **bit-identical** (QCD 1,773 / `81c6e682`; LambdaCDM 1,732 / `0960e169`). `test_background_segmentation` passes, 12 tests |
| Named v0 against `wkb_reference.production_source_grid` | bit-identical, and `tk_numeric_atol_sweep.py`'s §4 figures unchanged | **bit-identical** (1,732 / `0960e169`). The script's figures are unchanged — see V1: its entire output is unchanged |
| `ComputeTargets` suite | 452 → 452 + *n*, OK | **452 → 484, OK**, *n* = **32**, all of them `test_convergence_reference.py`'s. **195.6 s**, against **188.6 s** for the same suite on a `git worktree` of `9f04f3a` measured in the same session — so the addition costs **7.0 s**, which is the new module's own 7.1 s. The board's 164 s was taken on another day; 188.6 s is today's baseline, and both are far inside the ~240 s the prompt sets |
| `CosmologyModels` suite | 39 → 39, OK | **39 → 39, OK**, 0.78 s |
| Production files in the diff | **zero** | **zero**. The diff is 3 files under `ComputeTargets/tests/`, 1 script under `docs/gktk-remedial/`, plus 2 new test-tree modules and the campaign's log, board and `docs/OPEN_ISSUES.md` |
| `black --check` on every file touched | clean | clean, 6 files |
| Runtime of the new test module | quoted | **7.1 s**, 32 tests. `QCD_Cosmology` and `LambdaCDM` are built once in `setUpClass` of the one class that needs them |

Two further checks, neither asked for, both worth more than the ones that were:

| Check | Result |
|---|---|
| The **whole** of `tk_numeric_atol_sweep.py`'s output, this tree against a `git worktree` of `9f04f3a` | identical in all 294 lines but the two that print the script's own wall clock (V1) |
| The $G_k$ machinery, which `main()` does not exercise: `gk_geometry` + `run_gk` + `sector_errors` at $k \in \{10^5, 10^7, 3\times10^8\}$ on `RadiationModel` and `LambdaCDMModel`, under **both** break-point policies, plus the $T_k$ side under `BREAK_POINT_ALL` | **bit-identical across the fold** — grid bytes, reference `value_sample` bytes, RHS evaluation counts and the summarised max/median all match, 14 cases |


---

## Observations not acted on

### 1. `main.source_grid_spacing_profile` raises on `QCD_Cosmology` at the anchor a QCD production run uses — **new issue**

Building the version-2 grid for `QCD_Cosmology` at that cosmology's own `z_exit_suph_e5` for
$k = 3\times10^8$ raises:

```
ValueError: phase_residual[Gk]: the Liouville-Green frequency is not positive at
z = 8.6447769e+11 for k = 266544.64 (leading = 3.8377293e-26, correction = -2.2713588e-21,
omega^2 = -2.2713204e-21) on cosmology QCD_Cosmology (store_id=0); the residual is only
defined inside the WKB region
```

$z = 8.6448\times10^{11}$ is the cosmology's own third declared crossing, and $k = 266544.64$ is
the smallest production wavenumber in the cosmology's units. The anchor is not a harness invention:
`CosmologyConcepts.wavenumber._solve_horizon_exit(QCD_Cosmology, k=3e8, -5)` — production's own
solver, run directly — returns **3.30033444460513e+16**, and `main.py:944` passes
`k_exit_earliest.z_exit_suph_e5` straight to `build_z_sample`. The same construction succeeds at
LambdaCDM's anchor, 2.0636395964161516e+16, where it gives the recorded 1,996 samples and digest
`4849552b`; the version-**1** grid succeeds at the QCD anchor (1,793 samples), so it is the density
criterion alone. The failure therefore depends on where the base lattice's nodes fall relative to
the crossing mask (`SOURCE_GRID_CROSSING_MASK_U`), and one anchor escapes it while the other does
not.

Two consequences, and neither is this prompt's to act on. (i) Whether a QCD production run can
build its source grid at all is now an open question. (ii) Every recorded version-2 QCD figure —
`test_source_grid.py`'s 1,996 / `4849552b`, `RECONCILIATION.md` §2.7's "1,996 samples on QCD",
`docs/qcd-background-verification.md` §10 — is anchored at **LambdaCDM's** `z_init`, not QCD's, so
"the production QCD grid" in the record is not the grid a QCD run would build. That is not a wrong
number; it is an unmarked one, which is the same species as
`[00-three-production-grid-reproductions]` itself.

Opened as `[01-v2-density-raises-at-the-qcd-production-anchor]` on the board and in
`docs/OPEN_ISSUES.md`. Fixing it is a production change in `main.py` or
`ComputeTargets/phase_residual.py`, which prompt 01 may not make and which README §0.5 puts outside
this campaign's boundary in any case.

### 2. The residue of `[00-three-production-grid-reproductions]`

28 call sites in 20 files still import `production_source_grid` by its bare name (deviation 1).
They are all version 0 and all correct; what they do not do is *say so* at the call site. Recorded
in the board's narrowing.

### 3. `RadiationModel`'s version-2 grid refines hard

2,306 samples against version 0's 1,165 at the same geometry, and a per-object sample count of
771–1,607 against a flat 486 on version 0. The density criterion has no LambdaCDM-like late-time
structure to spend samples on there, so it spends them on the transfer-function band. Not acted
on: `RadiationModel` is a control, not a production model, and README §0.5 holds the grid fixed.

### 4. `ComputeTargets/QuadSourceIntegral.py:1550` still cites `DEFAULT_QUADRATURE_ATOL = 1e-25`

Already on the board as a recorded-not-owned item; the file is out of bounds (README §0.4).
Unchanged, and repeated here only so that the next reader does not think it was missed.

---

## State handed to the next prompt

### The facility's public API

Import from `ComputeTargets.tests.convergence_reference`.

**Knobs.** `AccuracyKnob` — `tighter(steps=1)`, `looser(steps=1)`, `ladder(steps)`, `label`.
`TolerancePair(atol, rtol, axis="both")` — a step is one decade, as an exact decimal shift;
`axis` selects `"both"` / `"atol"` / `"rtol"`; `rtol_step_is_effective` against
`SCIPY_RTOL_FLOOR = 2.220446049250313e-14`. `GaussOrder(order, name="N")` — a step is `+1`.
**Prompt 04 uses `GaussOrder`; it does not need a tolerance anywhere.**

**Drift and reference.**
`reference_drift(build, knob, *, error_measure, smallest_reported_difference, criterion_ratio=10.0, reference=None, tightened_knob=None) -> DriftVerdict`
and `converged_reference(...)` with the same signature `-> ConvergedReference`.
`DriftVerdict`: `.threshold`, `.passed`, `.max`, `.median`, `.max_z`, `.headroom`, `.notes`,
`.drift` (the `summarise` dict), `str()`. `ConvergedReference`: `.payload`, `.drift`,
`.converged`, `bool()`, `.errors(candidate)`, `.score(candidate)` — the last returns the summary
with `reference_drift` and `reference_converged` in it, which is the shape a table row wants.
`smallest_reported_difference` has no default and never will have one.

**Anchors.** `radiation_anchors(model) -> {name: (kind, callable)}` over `T`, `Tprime`, `G`,
`Gprime`, `tau`, `cs_tau`, `friction_F`, `theta_G`, `rho_G`, `rho_T`, `z_exit`;
`anchor_error(kind, value, exact, envelope=None)`; `exact_z_exit(H0, k_inv_Mpc, efolds_subh=0)`;
`analytic_{G,Gprime,T,Tprime}(model, k_inv_Mpc, …, z)`. `rho_G` returns `0.0` identically.
`rho_T` raises `ValueError` outside $1 + z < k/(\sqrt6 H_0)$ and that raise is a caller error.

**Sectors.** `SECTORS = {"Tk": …, "Gk": …}`;
`tk_geometry(cosmology, k_inv_Mpc, grid)` and `gk_geometry(...)` — **`grid` is required**;
`tk_run(model, k, geo, atol, rtol, ic=None, break_point_kind=BREAK_POINT_DISCONTINUITY)`;
`gk_run(model, k, geo, atol, rtol, break_point_kind=BREAK_POINT_DISCONTINUITY)`;
`sector_errors(sector, model, k, geo, candidate, reference)`;
`sector_error_measure(sector, model, k, geo)`;
`sector_reference(sector, model, cosmology, k, grid, knob, *, smallest_reported_difference, criterion_ratio=10.0, break_point_kind=BREAK_POINT_DISCONTINUITY, geo=None) -> (geo, ConvergedReference)`.
`summarise(errors)` returns `max`, `max_z`, `max_x`, `second`, `median`, `terminal`, `terminal_x`,
`samples`.

**Note for prompt 03.** Both `run` defaults are `BREAK_POINT_DISCONTINUITY`, which is the module
default and the $G_k$ sector's production policy. **`TkNumericIntegration`'s production policy is
`BREAK_POINT_ALL`** (`GkTk-remedial` prompt 19) and every figure in this log and in
`TK-NUMERIC-ATOL-SWEEP.md` §4 was taken at the module default. Prompt 03 has to pass
`break_point_kind=BREAK_POINT_ALL` for its $T_k$ rows, as its own §3.3 says.

### How a caller selects a grid generation

From `ComputeTargets.tests.wkb_reference`: `source_grid(generation, z_init, z_end,
samples_per_log10z, *, cosmology=None, k_inv_Mpc=PRODUCTION_K_GRID_INV_MPC)` returns the
`SourceGrid` NamedTuple; `source_grid_z_values` and `source_grid_redshifts` return the array and
the `redshift_array`. `generation` is `SOURCE_GRID_V0`, `SOURCE_GRID_V1` or `SOURCE_GRID_V2` and
there is no default. Through the facility: `SourceGridSpec(generation=…, universal=…,
z_init=…).build(cosmology)`, and the two ready-made ones are `V0_PER_K_GRID` and
`V2_PRODUCTION_GRID`. `production_source_grid` still exists and is **version 0**.

### The version-2 grid, per model

| model | `z_init` | samples | `redshift_grid_digest` |
|---|---|---|---|
| `RadiationModel` | 4.4524e+10 (its own `z_exit_suph_e5` at $k = 3\times10^8$) | 2,306 | `3bef2c06` |
| `LambdaCDMModel` | 2.0636395964161516e+16 | **1,778** | **`60a3205a`** |
| `QCDModel` | **2.0636395964161516e+16 — LambdaCDM's anchor**, because the criterion raises at QCD's own 3.30033444460513e+16 (observations item 1) | **1,996** | **`4849552b`** |

Version 1 for comparison: QCD 1,773 / `81c6e682` at the LambdaCDM anchor and 1,793 at QCD's own;
version 0: 1,732 / `0960e169`. `SOURCE_GRID_V2_REPRODUCES_VERSION = 2`, cross-checked against
`CosmologyConcepts.wavenumber.SOURCE_GRID_CONSTRUCTION_VERSION` in
`test_convergence_reference.py` with a failure message that says what to do when production moves.


### The reference setting and drift figures, so prompt 03 does not re-derive them

Reference `TolerancePair(1e-18, 1e-12)`, one decade tighter to `(1e-19, 1e-13)`. **One decade and
not two**: `rtol = 1e-14` is below SciPy's `100 * eps` clamp and the facility says so in
`DriftVerdict.notes`. The worst drift and the criterion verdict per (model, generation) are the
table in V2. In short, on the version-2 production grid and the transfer-function sector:

| model | worst drift | at $k$ | median over the grid | criterion |
|---|---|---|---|---|
| `RadiationModel` | 4.26e-11 | 8.118e+07 | 2.08e-11 | met, by 317× |
| `LambdaCDMModel` | 5.23e-11 | 1.325e+08 | 3.82e-11 | met, by 711× |
| `QCDModel` | 7.11e-09 | 3.092e+06 | 1.23e-09 | met, by 5.7× |

`(1e-18, 1e-12)` is therefore a usable reference on all three models **on the version-2 grid and in
the transfer-function sector under `BREAK_POINT_DISCONTINUITY`**, with three orders of headroom on
the two smooth models and a factor of 5.7 on QCD. Prompt 03 should not assume the same headroom in
the $G_k$ sector or under `BREAK_POINT_ALL`; neither is measured here.


**The floors, re-confirmed but not re-derived.** The $T = 1, T' = 0$ initial condition holds
**2.52e-06** of the envelope on the radiation control at $k = 10^6$ (asserted in
`test_convergence_reference.py`; `GkTk-remedial` prompt 17 §8's figure, unmoved). The reference
run from **exact** initial data reproduces the closed-form $T$ to 1.34e-11 against its own
self-convergence drift of 1.22e-11 at the same wavenumber — the same order, which is what licenses
the drift statistic on the two models where no oracle exists.
