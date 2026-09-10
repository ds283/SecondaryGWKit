# Log 08 — Tighten the tests to the acceptance table and revalidate the fixtures

**Prompt:** prompts/transfer-remedial/08-fixture-revalidation.md
**Commit:** *(this commit; SHA not self-embedded)* — Tighten the Bessel tests to the new accuracy
**Model:** Claude Opus 5
**Date:** 2026-09-10
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

Four test files, no production code. `git diff HEAD~1 --stat` touches only those four, this log
and the board (plus `docs/OPEN_ISSUES.md` per `CLAUDE.md`).

### `LiouvilleGreen/tests/test_bessel_phase.py` (162 → 458 lines)

Rewritten around the campaign's own metrics. New module docstring states the measure change and
the division of labour with `test_bessel_two_region.py` / `test_bessel_compatibility.py`, so that
nothing here duplicates them.

New module-level constants: `LOW_ORDER_TARGET = 1e-11`, `LOW_ORDER_DERIV_TARGET = 1e-9`,
`HIGH_ORDER_TARGET = 1e-6`, `CORNER_TARGET = 1e-14` — README §6's numbers, named.

New private helpers:

```python
_check_points(data, max_x) -> np.ndarray   # the historical linspace, plus x_star, plus the
                                           # near-region nodes and their midpoints
_score(nu, xs, data, tier) -> ((E_theta, x), (E_A, x), (E_deriv, x))
```

* `_test_bessel_value` (`:14-42` → `:116-156`): `REL_DIFF = 0.5` on `|ours/theirs - 1|` against
  `jv`/`yv` replaced by `bessel_reference.phase_pair_error` / `amplitude_error` /
  `derivative_error` at `1e-11`, `1e-11`, `1e-9`, on the `exact` half-integer tier. The grid gains
  `x_star`, the interpolation nodes and their midpoints.
* `test_high_order` (`:86-119` → `:206-254`): `relerr < 1e-3` → the three metrics at `1e-6`; the
  sweep now starts at `x_0` rather than `1.5 x_0 + 1`; docstring rewritten (the `asin(J/m)`
  initial-condition paragraph is gone, the non-Limber use-case sentence kept).
* `test_phase_derivative` (`:121-142` → `:256-310`): `1e-6` at every order → `1e-9` at
  `nu = 2.5` and `1e-6` at 20.5 and 100.5, and the lower bound moves from `2 x_0 + 1` to `x_0`
  with 40 extra points inside the turning-point interval.
* `test_bessel_J_integral` (`:144-158` → `:312-349`): `MAX_INTEGRAL_RELERR` `1e-3` → `1e-10`, and
  the four expected values go from 10 to 20 significant digits. `_integrate_bessel_J` now supplies
  the fourth Levin phase key, `theta_abserr`.
* New `TestProductionOrdersAtLargeArgument` (`:351-455`), two tests: every cached 40-digit corner
  through `x = 1e15` at `nu = 1/2` and `5/2` at `1e-14`; and the logarithmic input mode scored
  against live `mpmath` at `exp(u)`, the argument the accessor is actually handed.

### `LiouvilleGreen/tests/test_3bessel_analytic.py`

* New module docstring carrying the seven-oracle before/after table, the Chebyshev-order finding
  and the random-draw margins.
* `ABS_TOLERANCE` `1e-6` → `1e-8`; `REL_TOLERANCE` `1e-5` → `1e-7`.
* `SINGULARITY_ABS_TOLERANCE` and `SINGULARITY_REL_TOLERANCE` unchanged at `1e-3` / `1e-2` — see
  "Verification performed" for the measurement and Deviation 4 for why.
* `test_abserr_bounds_truth`: `@unittest.expectedFailure` **removed**, assertion kept, docstring
  rewritten to say what changed and which half of the predicted cure did the work. This closes
  board issue `[07-abserr-bounds-truth-is-now-an-unexpected-success]`, and with it the module
  passes as a module again.
* The deprecated `atol`/`rtol` build arguments in `plot_and_compute_3Bessel`, in
  `test_abserr_bounds_truth` and in `test_YJJ_log_scaling` migrated to
  `phase_atol` / `amplitude_rtol`, at the values the ignored arguments were already silently
  getting (`DEFAULT_PHASE_ATOL` = `DEFAULT_AMPLITUDE_RTOL` = `1e-11`). Numerically identical —
  verified by re-running `test_abserr_bounds_truth` and reproducing all seven measurements to
  every printed digit. Deviation 2.

### `ComputeTargets/tests/test_tk_source_functions.py` — comments and one constant

* Module docstring: new closing paragraph separating the three error sources (Bessel oracle,
  consumer re-spline, LG truncation) and saying which this campaign moves.
* `M_exact` docstring: the stale `bessel_phase.py:270-282` citation replaced by the module
  docstring reference `bessel_phase.py:13`, with a note that prompt 05 rewrote the file and the
  convention is unchanged.
* `theta_exact` docstring: the `theta = pi - vartheta` rotation re-checked clause by clause, with
  the measured numbers, plus the zero-point check.
* the `err_scipy` comment ("carries `bessel_phase`'s own phase-function error and so saturates
  near 2e-6") rewritten with the before and after numbers.
* **`err_M`: `1e-8` → `1e-12`** (the only tolerance changed in this file).
* `err_T` (`1e-7`), `test_phase_convention`'s `1e-6`,
  `test_spline_error_dominates_on_the_production_grid`'s `1e-5`,
  `test_dlnM_dz_matches_finite_difference`'s and `test_omega_matches_phase_derivative`'s `1e-6`:
  each keeps its value and gains a comment naming what limits it.

### `ComputeTargets/tests/test_phase_groups.py` — comments and two constants

* Module docstring: the `~x * 1e-8` claim rewritten, plus a paragraph recording the before/after
  re-measurement and the single place the improvement is visible.
* `REALISTIC_THRESHOLD`'s comment: the stale §2.4 citation removed and the "not `bessel_phase`"
  clause upgraded from inference to measurement.
* `test_exact_fixture_seam`'s docstring: "the phase re-spline and `bessel_phase` floors" separated
  into four named terms with numbers.
* **`test_composed_phase_with_exact_constituents`: `1e-10` → `1e-13` and `1e-8` → `1e-10`.**
* `test_composed_phase_through_phase_spline`'s `1e-6` and `1e-8` keep their values and gain
  comments naming `phase_spline`'s own rounding as the limit.

## Deviations from the prompt

Five, of which three are choices the prompt left open. The two tagged STRUCTURALLY REQUIRED are
both of the same kind and neither is a departure from the prompt's *instructions* — in both cases
the prompt was followed exactly and doing so left something the campaign documents expected to be
done. They are recorded as deviations, rather than buried in "Observations", precisely so the
orchestrator surfaces them: each is a scope question only the user can settle, and neither touches
anything on README §4.3's list of load-bearing design decisions (the zero point, the sign
convention, the Wronskian relation, the exactness of §2 (a), which series supplies the tail, the
two-region structure, or `theta' = e^{-2 ell}`). Nothing in this commit's numbers depends on
either.

### 1. `test_three_bessel.py` was not touched — the prompt and README §4 disagree about it (STRUCTURALLY REQUIRED)

**What the prompt assumed.** README §4's hard-dependency list says "**07 before 08.** 08
re-tightens `test_three_bessel.py`'s and `test_3bessel_analytic.py`'s tolerances, which 07 changes
the accuracy of."

**What is actually there.** `test_three_bessel.py` is **not** in prompt 08's "Files you may
touch", nor in README §3's file column for prompt 08, and the prompt has no section for it — its
§2 covers `test_bessel_phase.py`, §3 `test_3bessel_analytic.py` and §4 the two `ComputeTargets`
fixtures. Its §6 acceptance says "`git diff HEAD~1 --stat` shows changes confined to the four test
files", which `test_three_bessel.py` would make five.

**What was done.** Nothing. The prompt file and README §3 agree with each other against README
§4's prose, so the prose is the odd one out, and touching a file the prompt forbids is a stop
condition (README §4.3). Recorded as board issue
`[08-test-three-bessel-tolerances-unassigned]`; its tolerances are still the pre-campaign ones and
prompt 07 left the module passing.

### 2. The deprecated `atol`/`rtol` call sites in `test_3bessel_analytic.py` were migrated (IMPLEMENTATION CHOICE)

The prompt does not mention them. `IMPLEMENTATION_STATE.md` standing note 20 does, and assigns
them to prompt 08 by name: "Five call sites in the tree still pass the old names and now emit one
`DeprecationWarning` each: `test_three_bessel.py` and `test_3bessel_analytic.py` (prompt 08) …".

**Alternatives considered.** (a) Leave them, on README §5 item 5 ("do not fix things the prompt
did not ask for") and record the conflict. (b) Migrate them to the *values printed in the source*,
`phase_atol=1e-25, amplitude_rtol=5e-14`. (c) Migrate them to the values the ignored arguments were
already producing.

**Chosen (c).** (b) is wrong: it would silently *change* the build accuracy of a module whose
tolerances this prompt is setting from measurement, and `1e-25` is not a meaningful radian budget
anyway. Between (a) and (c) the deciding argument is that this prompt's whole job in §3 is to set
tolerances honestly from measurement, and a tolerance justified by a build whose stated arguments
are silently discarded is not honest — a later reader would believe the phase had been built at
`rtol = 5e-14`. (c) is exactly meaning-preserving, which was checked rather than assumed: the
seven `test_abserr_bounds_truth` measurements reproduce to every printed digit before and after.
`test_three_bessel.py`'s call sites are untouched (Deviation 1).

### 3. `test_bessel_phase.py`'s four closed-form integral values were re-quoted to 20 digits (IMPLEMENTATION CHOICE)

The prompt says of `test_bessel_J_integral`: "Tighten as far as the measurement supports and say
what limits it… the floor may well be the quadrature or the closed-form reference values (which
are quoted to 10 digits at `:154-158`)… If so, that is the answer — record it and do not tighten
past it."

The reference values *were* the floor: the two `nu = 3/2` expectations are wrong by 1.94e-10 and
1.52e-10 relative to their own true values, i.e. one and a half orders above the quadrature. Two
readings of "do not tighten past it" are available — leave the threshold at 2e-10, or remove the
floor. Removing it was chosen, because a 10-digit constant is not a floor of the *calculation*,
only of the constant, and the campaign's method throughout has been to build a reference good
enough to score against. The four values were regenerated with `mpmath.quad` at 50 and 80 decimal
digits over two different panelizations (one panel; and unit-width panels, so the oscillation is
resolved) and all four agree to 25 digits; the reproduction line is in the code comment. They agree
with the values they replace to all 10 digits those carried, so no claim about the integrand
changes. The threshold is then set by the quadrature at 1e-10, and the comment says so.

### 4. `SINGULARITY_ABS_TOLERANCE` and `SINGULARITY_REL_TOLERANCE` were not tightened (IMPLEMENTATION CHOICE)

Measured — see "Verification performed" — and left alone. Reasoning there.

### 5. No assertion was added on `test_tk_source_functions`'s `err_scipy` (STRUCTURALLY REQUIRED)

`err_scipy` is the one number in that file that measured the Bessel oracle, and it is the one that
moved (1.985e-06 → 3.021e-08). Asserting it would be the natural way to lock the campaign's result
into that fixture. The prompt and README §4.2 allow "tolerance constants and comments only" in
`ComputeTargets/tests/`, and "anything more is a stop condition", so a new `assertLess` was not
added. Recorded as board issue `[08-tk-fixture-scipy-comparison-unasserted]`, and the printed
number plus the attribution table below are the evidence in the meantime.

## Verification performed

Everything below was run. Nothing in this section is inference except where it says so.

Environment: Python 3.12.14, SciPy 1.15.2, NumPy 2.2.4, mpmath 1.3.0, Darwin 25.5.0 (arm64), from
the repository root with `PYTHONPATH=.` and `MPLBACKEND=Agg`.

**Before/after methodology.** "Before" means a detached worktree at `f71401d` (prompt 01 — the old
phase-ODE `bessel_phase`, with prompt 01's `bessel_reference` already present so the same metrics
can be applied) for the oracle numbers, and at `f17f2d4` (the commit before prompt 01, i.e. the
tree as it was when the campaign started) for the two `ComputeTargets` fixtures and the
three-Bessel oracles, which need no new reference module. The *same script* was run on both trees
in each case, so the comparisons are on identical grids with identical metrics.

### 1. `test_bessel_phase.py`: every threshold, old → new, with the measurement

Two grids are involved and the table keeps them apart. "same grid" is the grid the *old* test
used, run on both trees by one script, so before and after are strictly comparable. "shipped" is
what the test as committed measures, on the enriched grid `_check_points` builds (the old
`linspace` plus `x_star`, the near-region nodes and their midpoints) — a strictly harder grid,
which is why some shipped numbers are larger than the same-grid ones.

| test | threshold before | after | same grid: before | same grid: after | shipped | at (nu, x) |
|---|---|---|---:|---:|---:|---|
| `_test_bessel_value` `E_theta`, nu=3/2 | 0.5 on `|ours/theirs-1|` | 1e-11 | 2.005e-06 | 5.222e-13 | **2.482e-12** | (1.5, 34.41 = `x_star`) |
| `_test_bessel_value` `E_theta`, nu=5/2 | 0.5 | 1e-11 | 2.903e-06 | 7.546e-13 | **2.484e-12** | (2.5, 64.47 = `x_star`) |
| `_test_bessel_value` `E_A`, nu=3/2 | *(not tested)* | 1e-11 | 1.404e-13 | 5.396e-14 | 2.545e-13 | (1.5, 34.41) |
| `_test_bessel_value` `E_A`, nu=5/2 | *(not tested)* | 1e-11 | 1.404e-13 | 5.018e-14 | 1.358e-13 | (2.5, 64.47) |
| `_test_bessel_value` `theta'`, nu=3/2 | *(not tested)* | 1e-9 | 5.150e-08 | 1.081e-13 | 5.090e-13 | (1.5, 34.41) |
| `_test_bessel_value` `theta'`, nu=5/2 | *(not tested)* | 1e-9 | 6.148e-08 | 1.004e-13 | 2.713e-13 | (2.5, 64.47) |
| `test_high_order` `E_theta`, nu=2.5 | 1e-3 | 1e-6 | 2.284e-06 | 3.993e-13 | 2.484e-12 | (2.5, 64.47) |
| `test_high_order` `E_theta`, nu=20.5 | 1e-3 | 1e-6 | 9.702e-07 | 1.511e-12 | 2.418e-12 | (20.5, 664.4) |
| `test_high_order` `E_theta`, nu=100.5 | 1e-3 | 1e-6 | 3.058e-07 | 2.261e-13 | 3.768e-13 | (100.5, 959.3) |
| `test_high_order` `E_theta`, nu=400.5 | 1e-3 | 1e-6 | 7.396e-07 | 1.311e-12 | 2.037e-12 | (400.5, 3787) |
| `test_high_order` `E_theta`, nu=1000.5 | 1e-3 | 1e-6 | 2.036e-06 | 4.956e-12 | **8.776e-12** | (1000.5, 9962) |
| `test_high_order` `theta'`, nu=1000.5 | *(not tested)* | 1e-6 | 2.156e-08 | 8.648e-12 | 8.095e-12 | (1000.5, 9505) |
| `test_phase_derivative`, nu=2.5 | 1e-6 from `2 x_0+1` | **1e-9 from `x_0`** | 6.263e-08 | 4.197e-14 | **4.197e-14** | (2.5, 81.41) |
| `test_phase_derivative`, nu=20.5 | 1e-6 from `2 x_0+1` | 1e-6 from `x_0` | **2.894e-06** | 2.265e-14 | 2.265e-14 | (20.5, 678.9) |
| `test_phase_derivative`, nu=100.5 | 1e-6 from `2 x_0+1` | 1e-6 from `x_0` | **6.388e-05** | 3.643e-13 | 3.643e-13 | (100.5, 919.2) |
| `test_bessel_J_integral` | 1e-3 | 1e-10 | *(reference-limited, see below)* | — | 8.894e-12 | (2.5, [5,100]) |
| corners through 1e15, nu=1/2 | *(not tested)* | 1e-14 | **1.509** | 1.110e-16 | 1.110e-16 | (0.5, 1e12 before / 10 after) |
| corners through 1e15, nu=5/2 | *(not tested)* | 1e-14 | **1.512** | 5.274e-16 | 5.274e-16 | (2.5, 1e15 before / 3.674 after) |
| log input mode vs `mpmath` at `exp(u)` | *(not tested)* | 1e-14 | — | — | 5.117e-16 | (2.5, 25) |

The `test_phase_derivative` rows show the same value in both "after" columns because that test's
grid is the same in the script and in the shipped test; the corner rows likewise, since the corner
set is fixed by `bessel_reference_data.json`.

Three of those rows are worth reading twice.

1. **The corner rows.** The old construction's envelope-normalized phase error at `x = 1e12` and
   `1e15` is **O(1)** — 1.509 and 1.512, i.e. the reconstructed `sin theta` bears no relation to
   `J_nu/A`. That is the `eps * theta` rounding of `sin(raw_theta(x))` (0.03 rad at 1e15,
   `RECONCILIATION.md` §1), not a defect of the phase itself, and it is exactly what prompt 05's
   angle addition removes. The new numbers are one and five ulps.
2. **`test_phase_derivative` at nu = 20.5 and 100.5.** Extending the lower bound from `2 x_0 + 1`
   down to `x_0` is not a cosmetic tightening: on the pre-campaign tree the same test with the
   turning-point interval included measures 2.894e-06 and 6.388e-05, i.e. it would have **failed
   its own 1e-6 contract by up to 64x**. `RECONCILIATION.md` C4 flags the exclusion; this measures
   what it was hiding. The new construction passes the extended test by 44x and 2700x.
3. **Every low-order `E_theta` maximum is at `x_star`**, which is the tail series remainder and
   not interpolation, reproducing prompt 05's finding on this file's own grids. That is why
   `_check_points` puts `x_star` in the grid explicitly, and it is worth 4.8x at nu = 3/2
   (5.222e-13 on the plain `linspace`, 2.482e-12 once `x_star` and the nodes are in): the old
   100-point `linspace` misses the maximum by construction. The 1e-11 target is set against the
   harder grid and still clears it by 4x.

`test_bessel_J_integral`, in full. Value against the 20-digit reference, and the reported Levin
`abserr` now that `theta_abserr` is supplied:

| nu | span | relative error vs the 20-digit value | reported `abserr` | regions |
|---|---|---:|---:|---:|
| 3/2 | [5, 10] | 6.117e-16 | 9.147e-12 | 1 (direct) |
| 3/2 | [5, 100] | 1.160e-12 | 2.148e-12 | 2 (Levin) |
| 5/2 | [5, 10] | 5.138e-18 | 9.557e-12 | 1 (direct) |
| 5/2 | [5, 100] | 8.894e-12 | 3.513e-12 | 2 (Levin) |

**What limits it: the quadrature.** The declared phase error over these spans is 5.0e-12 rad and
the reported `abserr` is 2.1e-12 to 9.6e-12, so 1e-10 is one order above the largest measured
residual and the same order as the quadrature's own reported error. Tightening further would be
measuring `adaptive_levin_sincos`. Note the two `[5, 100]` rows are the only ones where the value
is not correct to nearly every bit, and they are the two that actually use the Levin rule.

Also recorded, since the prompt asks for the reference-value floor by name: the 10-digit
expectations this replaces are wrong by **1.942e-10** (nu=3/2, [5,10]), **1.524e-10** (3/2,
[5,100]), 6.596e-16 (5/2, [5,10]) and 7.139e-17 (5/2, [5,100]) relative to their own true values.
The first two are the floor Deviation 3 removes.

Module run: `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_phase`
→ **6 tests, OK, 0.209 s** (prompt 01's baseline for this module: 4 tests, 1.2 s).

### 2. `test_3bessel_analytic.py`

**The seven oracles at the fixed triple** k, q, s = 1.3, 1.7, 2.1, `max_x = 1e12`, `atol = 1e-14`,
`rtol = 1e-10`. Relative error against the closed form, and whether the reported `abserr` bounds
the true error:

| oracle | relerr before | relerr after | gain | true/reported before | after |
|---|---:|---:|---:|---:|---:|
| J000 | 1.625e-08 | 1.397e-10 | 116x | 0.0179 (bounds) | 1.54e-04 (bounds) |
| J110 | 4.615e-08 | 5.857e-12 | 7880x | **1.03** (fails) | 1.30e-04 (bounds) |
| J220 | 2.561e-08 | 2.521e-14 | 1.0e6x | **9.98** (fails) | 9.77e-06 (bounds) |
| J222 | 9.552e-09 | 3.504e-14 | 2.7e5x | **3.93** (fails) | 1.42e-05 (bounds) |
| J231 | 3.349e-08 | 1.491e-13 | 2.2e5x | **11.47** (fails) | 5.08e-05 (bounds) |
| Y000 | 2.156e-08 | 4.771e-11 | 452x | 0.0161 (bounds) | 3.55e-05 (bounds) |
| Y022 | 2.014e-08 | 1.543e-14 | 1.3e6x | **7.27** (fails) | 7.11e-05 (bounds) |

2 of 7 bounding before, 7 of 7 after — reproducing on this tree exactly what prompt 07 measured on
`bc31493` and `f9cc891`, and confirming that dropping the `@unittest.expectedFailure` is the
correct repair rather than a tolerance change.

**Is `DEFAULT_3BESSEL_CHEBYSHEV_ORDER = 12` now the limit? Yes, for the two (0,0,0) oracles.**
Sweeping it at the same triple, relative error:

| oracle | order 12 | order 20 | order 32 |
|---|---:|---:|---:|
| J000 | 1.397e-10 | **2.071e-13** | 4.494e-14 |
| J110 | 5.857e-12 | 2.248e-12 | 9.167e-09 |
| J220 | 2.521e-14 | 1.582e-12 | 9.329e-13 |
| J222 | 3.504e-14 | 3.423e-12 | 3.561e-13 |
| J231 | 1.491e-13 | 7.045e-13 | 1.101e-11 |
| Y000 | 4.771e-11 | **3.508e-13** | 3.390e-14 |
| Y022 | 1.543e-14 | 4.745e-13 | 2.597e-11 |

So J000 and Y000 improve by three orders when the spectral order rises, and the other five get
between 4x and 1500x **worse** — conditioning of the higher-order Chebyshev fit. The constant is
therefore genuinely limiting for two of the seven and genuinely well chosen for the other five,
which is not a one-line change; per the prompt it was left alone and board issue
`[08-3bessel-chebyshev-order-is-now-the-limit]` opened.

This also **corrects prompt 07's observation 4**, which attributed J000's and Y000's residuals to
the `MAX_X = 1e12` truncation. Sweeping `max_x` at order 12 gives J000 1.397e-10 → 1.512e-11
(1e13) → 5.273e-12 (1e14), so the truncation is real but smaller: at `max_x = 1e12` raising the
spectral order alone reaches 2.071e-13, which is below anything the `max_x` sweep achieves, so at
order 12 the spectral order dominates the truncation.

**Random draws.** `test_JJJ`, `test_YJJ` and `test_YJJ_log_singularity` draw
`k, q, s ~ uniform(0.1, 5)` unseeded, so the tolerances have to survive draws that were not
measured. 42 seeded draws (6 per oracle) at `max_x = 1e12`, order 12:

* worst relative error on a triangle configuration: **1.740e-09** (Y000, k,q,s = 4.269, 4.483,
  0.416);
* worst absolute error on a non-triangle configuration, where the closed form is 0 and
  `ABS_TOLERANCE` is the only gate: **4.056e-10** (J000, 2.420, 4.522, 1.987);
* every other oracle stayed below 1.5e-11 relative.

`REL_TOLERANCE = 1e-7` and `ABS_TOLERANCE = 1e-8` are 100x tighter than the values they replace
and keep factors of 57 and 25 over those maxima. Tighter would be measuring the draw.

**The two singularity bands: measured, and left alone.** `test_YJJ_log_singularity` was run to
completion — the first time in this campaign, closing board issue
`[05-3bessel-analytic-not-run-to-completion]`. 40 cases, 21.2 min, OK. It walks `s` to within
`eps` of `|k - q|` (the "lo" branch) and of `k + q` ("hi") for `eps` from 0.1 down to 1e-10. All 40
cases, `|relerr|` / `|abserr|`:

| eps | Y000 lo | Y000 hi | Y022 lo | Y022 hi |
|---|---|---|---|---|
| 1e-01 | 1.095e-09 / 4.003e-11 | 1.948e-09 / 2.235e-11 | 1.692e-10 / 1.325e-10 | 1.447e-10 / 3.650e-12 |
| 1e-02 | 5.583e-12 / 3.859e-13 | 5.521e-11 / 9.388e-13 | 4.992e-11 / 2.040e-11 | 2.392e-11 / 2.635e-12 |
| 1e-03 | 8.288e-11 / 8.293e-12 | 9.704e-11 / 2.195e-12 | 2.669e-10 / 5.503e-10 | 8.788e-11 / 1.651e-11 |
| 1e-04 | 9.109e-10 / 1.190e-10 | 8.322e-10 / 2.351e-11 | 2.035e-09 / 8.873e-09 | 1.132e-09 / 2.995e-10 |
| 1e-05 | 3.477e-09 / 5.604e-10 | 3.071e-09 / 1.041e-10 | 6.084e-09 / 3.994e-08 | 4.090e-09 / 1.395e-09 |
| 1e-06 | 2.421e-08 / 4.640e-09 | 2.164e-08 / 8.552e-10 | 3.798e-08 / 3.324e-07 | 2.782e-08 / 1.161e-08 |
| 1e-07 | 2.131e-08 / 4.736e-09 | 1.876e-08 / 8.473e-10 | 3.102e-08 / 3.393e-07 | 2.415e-08 / 1.193e-08 |
| 1e-08 | 1.602e-06 / 4.049e-07 | 1.476e-06 / 7.496e-08 | 2.210e-06 / 2.901e-05 | 1.778e-06 / 1.014e-06 |
| 1e-09 | 3.867e-05 / 1.095e-05 | 3.583e-05 / 2.022e-06 | 5.123e-05 / 7.845e-04 | 4.242e-05 / 2.743e-05 |
| 1e-10 | **2.175e-04** / 6.824e-05 | 2.030e-04 / 1.260e-05 | **2.794e-04** / **4.888e-03** | 2.363e-04 / 1.709e-04 |

Against `SINGULARITY_REL_TOLERANCE = 1e-2` and `SINGULARITY_ABS_TOLERANCE = 1e-3`. Two things
follow, and both argue for leaving the constants where they are.

1. **Below `eps = 1e-3` the error grows by roughly an order for every order `eps` falls** — flat
   at 1e-11 to 1e-09 for `eps >= 1e-3`, then 1e-09, 3e-09, 2e-08, 2e-08, 2e-06, 4e-05, 3e-04. (It
   is not monotone at the top: `eps = 1e-2` is *better* than `eps = 0.1`, because at `eps = 0.1`
   the configuration is not yet near-singular and the residual is the ordinary one.) A factor 36 of
   headroom on a quantity with that scaling, at wavenumbers the test redraws on every run, is not
   margin to spend: a draw that put `k` and `q` an order closer together would eat it.
2. **The `or` in those assertions is load-bearing.** At `eps = 1e-10` the *absolute* error
   4.888e-03 **exceeds** `SINGULARITY_ABS_TOLERANCE = 1e-3`, and the case passes on the relative
   band alone. So the absolute band cannot be tightened either — it is already inactive where the
   test is hardest, and lowering it would change nothing except to make the `or` mandatory in
   writing as well as in fact. Both facts are now in a comment above the two constants.

**Attribution of the singularity bands, and what could not be measured.** The `eps` scaling is
itself the attribution: an error that is 1e-11 at `eps = 1e-2` and 3e-04 at `eps = 1e-10`, growing
like the closed form's own `log|(k-q+s)(k+q-s) / ((k+q+s)(k-q-s))|` conditioning, cannot be a
property of a phase representation that is uniformly accurate to 5e-12 rad. So these two bands are
**genuine near-singular behaviour**, the fourth candidate the prompt's §3 lists, and not the
Bessel oracle.

A directly comparable before/after was attempted and **is not reported, because the pre-campaign
run did not finish**. The script fixes `k, q = 1.3, 1.7` and `s = |k - q| + eps` for
`eps = 1e-2, 1e-6, 1e-10` on both Y oracles, with the singularity test's own quadrature tolerances
(`atol = 1e-10`, `rtol = 1e-8`) and no plotting. On this tree all six cases complete in about nine
minutes:

| oracle | eps | s | analytic | relerr | abserr |
|---|---:|---:|---:|---:|---:|
| Y000 | 1e-02 | 0.4100000000 | -1.288352 | 2.966e-08 | 3.821e-08 |
| Y000 | 1e-06 | 0.4000010000 | -3.919852 | 2.528e-08 | 9.911e-08 |
| Y000 | 1e-10 | 0.4000000001 | -6.524596 | 2.232e-04 | 1.456e-03 |
| Y022 | 1e-02 | 0.4100000000 | -0.4048635 | 4.373e-11 | 1.770e-11 |
| Y022 | 1e-06 | 0.4000010000 | -3.071417 | 3.220e-08 | 9.890e-08 |
| Y022 | 1e-10 | 0.4000000001 | -5.676180 | 2.565e-04 | 1.456e-03 |

The same script on `f17f2d4` had **not produced its first line after 29.5 minutes** and was
abandoned, so no "before" column exists. That is itself worth
recording rather than hiding: it means the old route is at least an order of magnitude slower on a
near-singular YJJ configuration, which is consistent with prompt 07's account of why — the old
`_phase_group` differenced independently reconstructed large phases, so the phase
`adaptive_levin_sincos` was handed was noisy at ~`eps * theta`, and the subdivider fights noise by
bisecting. **Marked as not measured** rather than inferred; nothing in this prompt's decisions
depends on it, since the bands were left unchanged.

The 2.97e-08 at `eps = 1e-2` in the table above is worth one note, because it looks out of place
next to the 4.373e-11 on the other oracle: it is Y000 at this *particular* `(k, q)`, where
`s = 0.41` is small and `s * max_x` is correspondingly small, and it is the same
`DEFAULT_3BESSEL_CHEBYSHEV_ORDER = 12` effect as in the fixed-triple table — Y000 and J000 are the
two oracles that constant limits.

Module runs, all with `MPLBACKEND=Agg`:

| target | result | wall clock |
|---|---|---:|
| `test_abserr_bounds_truth` | OK (no longer an expected failure) | 2.4 s |
| `test_JJJ` + `test_YJJ` together | OK | 665.9 s (11.1 min) |
| `test_YJJ_log_singularity` | OK, all 40 cases | 1270.3 s (21.2 min) |
| `test_YJJ_log_scaling` | ran as part of the module discovery run below; it contains no assertion | — |

`test_JJJ`/`test_YJJ`'s own draws on that run, for the record: worst triangle relative error
**1.047e-09** (Y000, k,q,s = 3.838, 1.464, 3.486) against `REL_TOLERANCE = 1e-7`, a factor 95; worst
non-triangle absolute error **1.293e-13** against `ABS_TOLERANCE = 1e-8`, a factor 77000. Both
consistent with the 42-draw sample above.

### 3. `ComputeTargets/tests/test_tk_source_functions.py`

Every printed number, on `f17f2d4` and on this tree, from the same test module:

| quantity | before | after | threshold |
|---|---:|---:|---|
| `err_M`, w=1/3 | 1.268e-13 | 1.272e-13 | 1e-12 (was 1e-8) |
| `err_M`, w=0.2 | 7.708e-14 | 7.843e-14 | 1e-12 |
| `err_T`, w=1/3 | 3.021e-08 | 3.021e-08 | 1e-7, unchanged |
| `err_T`, w=0.2 | 2.215e-08 | 2.234e-08 | 1e-7, unchanged |
| `err_scipy`, w=1/3 | **1.985e-06** | **3.021e-08** | not asserted |
| `err_scipy`, w=0.2 | **1.550e-06** | **2.234e-08** | not asserted |
| `[grid refinement]` 100/decade | 6.090e-06 | 6.090e-06 | 1e-5, unchanged |
| `[grid refinement]` 300/decade | 3.021e-08 | 3.021e-08 | < 1/10 of the above |
| `test_phase_convention` `sin` diff | — | 7.342e-08 | 1e-6, unchanged |
| `[LG fixture] d ln M/dz` vs FD, w=1/3 | 6.355e-11 | 6.355e-11 | 1e-6, unchanged |
| `[LG fixture] omega` vs `theta_deriv`, w=1/3 | 4.249e-08 | 4.249e-08 | 1e-6, unchanged |
| `[exact envelope] d ln M/dz` vs FD | 8.528e-06 | 8.528e-06 | 1e-3, unchanged |
| numeric-region midpoint `dT/dz` | 2.720e-04 | 2.720e-04 | 1e-3, unchanged |

The single decisive line is `err_scipy`: it falls by 66x and lands **exactly on `err_T`**, to
every printed digit, at both `w`. `err_T` compares against this fixture's own re-splined phase and
`err_scipy` against `scipy`'s `J_nu`; their difference is precisely the Bessel oracle's
contribution, and it has gone from 1.955e-06 to below the last printed digit of a 3.0e-08 number.
That is the answer to `docs/lg-phase-and-handover-followup-2026-09.md` §2.4's claim that every
"exact" constant-w fixture here carries a phase floor of order `x * 1e-8`: for this fixture the
claim was correct and is now false.

**The `theta_exact = pi - vartheta` convention check, clause by clause** (prompt §4.1 and §7).
Measured on both `w`; the docstring now carries the same:

| clause | verdict | evidence |
|---|---|---|
| "`bessel_phase`'s vartheta *increases* with x" | holds | `theta_deriv = a_nu^-2 >= 0.99727` (w=1/3) and `>= 0.99591` (w=0.2) over x in [18.5, 1000]; the sampled `vartheta` is strictly increasing on a 4001-point grid, minimum forward difference 0.2444 |
| "hence decreases with z" | holds | `x = k c_s a0 tau(z)` is strictly decreasing along the stored `z_WKB` (which is descending), checked elementwise |
| "`theta = pi - vartheta` therefore increases with z" | holds | strictly monotone on the stored grid, both `w` |
| "as the code's `d(theta)/dz = +omega_eff` convention requires" | holds | central difference of `pi - vartheta(x(z))` in z is positive everywhere and agrees with `+sqrt(Tk_omegaEff_sq)` to 1.074e-05 (w=1/3) and 1.856e-04 (w=0.2) relative. That residual is the LG truncation of the closed-form `omega_eff` against the exact Bessel phase's derivative — the same O(x^-4) quantity `test_omega_matches_phase_derivative` sees at 4.2e-08 on the *LG* fixture — and not a phase error |
| "`sin(theta) = sin(vartheta)` keeps the amplitude positive" | holds | `|sin(pi - vartheta) - sin(vartheta)| <= 1.078e-14`, and `min M_exact` is 3.0e-06 (w=1/3), 7.7e-07 (w=0.2), both positive |
| the new zero point | exact | `phase.c_nu == pi/4 - pi nu/2` bit for bit at nu = 3/2 and 7/4, and `raw_theta(x) - (x + c_nu + residual(x))` is exactly 0.0 |

One new number worth recording: the 1.078e-14 in the last-but-one row is not zero, and it is the
fixture's own arithmetic, not `bessel_phase`'s — `pi` is a 53-bit approximation and `vartheta`
reaches 1000, so forming `pi - vartheta` rounds at ~ulp(1000). It is a floor this rotation
introduces, three orders below the 3.0e-08 re-spline term that dominates `err_T`, and it is now
stated in the docstring so that nobody later reads it as an oracle error.

Module run: 12 tests, OK, 1.5 s (was 0.58 s before the added docstrings — the difference is run
noise, not the docstrings).

### 4. `ComputeTargets/tests/test_phase_groups.py`

| quantity | before | after | threshold |
|---|---:|---:|---|
| Oracle 1, exact stand-ins (worst) | 6.669e-16 | 6.669e-16 | 1e-12 region, unchanged |
| Oracle 2, exact stand-ins (worst) | 3.412e-14 | 3.412e-14 | 1e-8, unchanged |
| realistic, w=1/3, 100/decade | 7.015e-06 | 7.048e-06 | `REALISTIC_THRESHOLD` 5e-4 |
| realistic, w=0.2, 100/decade | 1.376e-04 | 1.375e-04 | 5e-4 |
| realistic, w=1/3, 300/decade, `x_q > 100` | **1.661e-06** | **7.502e-09** | printed only |
| realistic, w=0.2, 300/decade, `x_q > 100` | 5.710e-06 | 5.568e-06 | printed only |
| LG-truncation attribution, w=1/3 | 5.453e-06 → 1.474e-06 | 5.466e-06 → 1.221e-06 | printed only |
| LG-truncation attribution, w=0.2 | 7.778e-05 → 1.439e-06 | 7.775e-05 → **2.866e-07** | printed only |
| composed phase, exact constituents, `|Psi|` ~ 2.2e6 | 1.776e-15 | 1.776e-15 | **1e-13** (was 1e-10) |
| composed phase, exact constituents, derivative | 2.332e-12 | 2.332e-12 | **1e-10** (was 1e-8) |
| composed phase, `phase_spline` constituents, `|Psi|` ~ 2.2e6 | 3.782e-08 | 3.782e-08 | 1e-6, unchanged |
| exact-fixture seam, nodes (w=0.2) | 1.514e-04 | 1.514e-04 | 5e-4, unchanged |
| exact-fixture seam, midpoints (w=1/3) | 1.031e-03 | 1.031e-03 | 2e-3, unchanged |

**Where the improvement is, and where it is not.** Only one asserted-adjacent number moves: the
`x_q > 100`, 300/decade restriction at w = 1/3 falls by **221x**, from 1.661e-06 to 7.502e-09.
That restriction is precisely the one that removes the other two floors — 300/decade pushes the
re-spline down, `x_q > 100` moves away from the hand-over where the LG truncation of `omega` and
`d ln M/dz` lives — so it is the only place in this module where the Bessel oracle was ever the
binding term, and there it improves by the campaign's expected margin. The same restriction at
w = 0.2 barely moves (5.710e-06 → 5.568e-06) because at w = 0.2 the LG truncation is an order
larger and still dominates. The LG-truncation attribution test agrees: with exact `omega` and
`d ln M/dz` substituted, the w = 0.2 residual falls from 1.439e-06 to 2.866e-07, a factor 5,
which is the oracle contribution that used to sit under it.

`REALISTIC_THRESHOLD` therefore stays at 5e-4: its own measurement moved from 1.376e-04 to
1.375e-04, a factor 1.0007, while the oracle under it changed by eight orders. The comment now
records that as the *measurement* which replaces the previous inference.

Module run: 18 tests, OK, 3.7 s (before: 4.0 s).

### 5. The other two suites the prompt names as tripwires

| suite | result | wall clock |
|---|---|---:|
| `unittest discover -s AdaptiveLevin/tests -t .` | 32 tests, **OK** | 0.05 s |
| `unittest discover -s CosmologyModels/tests -t .` | 11 tests, **OK** | 0.05 s |
| `unittest discover -s ComputeTargets/tests -t .` | 97 tests, **OK** | 138.2 s |
| `unittest discover -s LiouvilleGreen/tests -t .` | **133 tests, OK** | 1524.5 s (**25.4 min**) |

Neither `AdaptiveLevin/tests` nor `CosmologyModels/tests` was expected to move and neither did.

**The `LiouvilleGreen/tests` discovery run completes, and this is the first time in the campaign
that it has.** `RECONCILIATION.md` §3.4 and `docs/transfer-remedial/baseline-2026-09.md` §5 record
it as not finishing within 50 minutes on the planning machine, and prompt 01 timed out at 15
minutes; prompt 05 started it and did not see it finish. It now runs 133 tests in 25.4 min. Two
things changed: prompt 05 made construction ~25x cheaper (~2.5 ms against ~60-100 ms), and the
`@unittest.expectedFailure` that made the module report `FAILED` regardless is gone. Together they
close board issue `[05-3bessel-analytic-not-run-to-completion]`.

**The suite wall clock against prompt 01's baseline**, per module where a comparison exists:

| module | baseline (log 01 / prompt 05, 06) | now |
|---|---:|---:|
| `test_bessel_phase` | 1.2 s (4 tests) → 0.18 s after prompt 05 | **0.21 s** (6 tests) |
| `test_bessel_reference` | 0.6 s | unchanged (not touched) |
| `test_range_reduce` | 0.1 s | unchanged (not touched) |
| `test_three_bessel` | 7.5 s → 4.1 s after prompt 05 | unchanged (not touched) |
| `test_bessel_compatibility` | 2.2 s (prompt 06) | unchanged (not touched) |
| `test_bessel_two_region` | 3.8 s (prompt 05) | unchanged (not touched) |
| `test_3bessel_analytic` | TIMEOUT at 15 min (prompt 01) / 50 min (planning) | ~25 min, **completes** |
| whole `LiouvilleGreen/tests` discovery | did not complete | **25.4 min** |

25.4 min is under the prompt's ~30 min line, so the "say so prominently" clause is not triggered;
but it is within a factor 1.2 of it and essentially all of it is one module, so the proposal the
prompt asks for is recorded anyway. **This prompt tightening the thresholds did not make the suite
slower**: `test_bessel_phase` went from 1.2 s to 0.21 s while gaining two tests and the orders
400.5 and 1000.5, because the new construction builds in milliseconds and because the reference is
the closed-form `exact` tier or the cached corner table rather than live `mpmath` (`mpmath` is
used at six points only, in the log-input-mode test).

**Proposed, not implemented** — what to skip by default, if the suite ever needs to be faster
(board issue `[08-3bessel-plot-cost-dominates-the-suite]`):

1. **Gate `plot_and_compute_3Bessel`'s 250-point plot grid** behind an environment variable or a
   module flag defaulting to *off*. It evaluates 250 full three-Bessel integrals per case purely to
   draw a figure, then evaluates the one integral the assertion uses; 80 cases across
   `test_YJJ_log_singularity` and `test_YJJ_log_scaling` makes that 20,000 integrals for
   diagnostics against 80 for verification. This alone would take the module from ~25 min to well
   under a minute and changes no threshold and no assertion.
2. **Move `test_YJJ_log_scaling` out of `unittest` discovery.** It asserts nothing: it draws two
   figures per Y integral and prints timings. It belongs beside the other diagnostic scripts in
   `docs/`, or behind the same flag.
3. **Do not** thin the `eps` list or the oracle list. Those are the coverage, and the `eps = 1e-9`
   and `1e-10` cases are where the singularity bands are actually exercised.

### 6. Formatting

`./venv/bin/python -m black` on all four files; the tree is clean under `--check`.

## The §5 attribution table — the required deliverable

One row per fixture comparison. "Bessel oracle" means improved by this campaign; "consumer
re-spline" means the `h^4` fit of a sampled fixture on the production grid, which this campaign
does not touch; "LG truncation" is physical; "quadrature" and "reference value" are what they say.

Oracle rows are quoted on the **same grid** on both trees (§1's "same grid" columns), so the
ratios are strictly comparable; the shipped tests measure on a harder grid and their values are in
§1.

| comparison | error before | error after | attribution |
|---|---:|---:|---|
| `test_bessel_phase` `_test_bessel_value` `E_theta`, nu=3/2 | 2.005e-06 | 5.222e-13 | **Bessel oracle** (the residual is now the tail series remainder at `x_star`) |
| `_test_bessel_value` `E_theta`, nu=5/2 | 2.903e-06 | 7.546e-13 | **Bessel oracle** |
| `_test_bessel_value` `E_A`, nu=3/2 | 1.404e-13 | 5.396e-14 | Bessel oracle, but it was never the limit here — both trees sit near the amplitude sampling floor and the gain is a factor 2.6, not six orders |
| `_test_bessel_value` `theta'`, nu=3/2 | 5.150e-08 | 1.081e-13 | **Bessel oracle** |
| `test_high_order` `E_theta`, nu=1000.5 | 2.036e-06 | 4.956e-12 | **Bessel oracle** |
| `test_phase_derivative`, nu=100.5, from `x_0` | 6.388e-05 | 3.643e-13 | **Bessel oracle** (the old value violated the test's own 1e-6 contract; the old lower bound hid it) |
| `test_bessel_phase` cached corners, nu=1/2 through 1e15 | 1.509 | 1.110e-16 | **Bessel oracle** — specifically the split evaluation; the old failure is `eps * theta` rounding of `sin(x + d)` |
| `test_bessel_J_integral`, nu=5/2, [5,100] | (1e-3 threshold; 8.894e-12 vs truth) | 8.894e-12 | **quadrature** — and, before Deviation 3, **reference value** at 1.5e-10 |
| `test_3bessel_analytic` J110 at the fixed triple | 4.615e-08 | 5.857e-12 | **Bessel oracle** |
| `test_3bessel_analytic` J220 / J222 / J231 / Y022 | 9.6e-09 – 3.3e-08 | 2.5e-14 – 1.5e-13 | **Bessel oracle** |
| `test_3bessel_analytic` J000 / Y000 | 1.6e-08 / 2.2e-08 | 1.397e-10 / 4.771e-11 | **Bessel oracle** for the improvement; the residual is **quadrature** — `DEFAULT_3BESSEL_CHEBYSHEV_ORDER = 12`, worth three further orders |
| `test_3bessel_analytic` reported `abserr` bounds truth | 2 of 7 | 7 of 7 | **Bessel oracle** (prompt 05 dropped the true error) plus the declared `theta_abserr` (prompt 07, +0–1.9 % on the reported value) |
| `test_3bessel_analytic` singularity bands, `eps >= 1e-3` | *(not measured; see §2)* | 1e-11 – 1e-09 rel | **quadrature** at `eps` this size, and `DEFAULT_3BESSEL_CHEBYSHEV_ORDER` on the (0,0,0) oracles |
| `test_3bessel_analytic` singularity bands, `eps = 1e-10` | *(not measured; see §2)* | 2.794e-04 rel | **genuine near-singular behaviour** — the error grows an order per order of `eps`, which no 5e-12 rad phase can produce |
| `test_tk_source_functions` `err_M`, w=1/3 | 1.268e-13 | 1.272e-13 | **consumer re-spline** (amplitude path) |
| `test_tk_source_functions` `err_T`, w=1/3 | 3.021e-08 | 3.021e-08 | **consumer re-spline** (phase path, `h^4`) |
| `test_tk_source_functions` `err_scipy`, w=1/3 | 1.985e-06 | 3.021e-08 | **Bessel oracle** removed; what remains *is* `err_T`, i.e. **consumer re-spline** |
| `test_tk_source_functions` `err_scipy`, w=0.2 | 1.550e-06 | 2.234e-08 | as above |
| `test_tk_source_functions` `[grid refinement]` 100/decade | 6.090e-06 | 6.090e-06 | **consumer re-spline** |
| `test_tk_source_functions` `omega` vs `theta_deriv` (LG fixture) | 4.249e-08 | 4.249e-08 | **consumer re-spline** (a `phase_spline` derivative through the exact `omega` integral) |
| `test_tk_source_functions` `d ln M/dz` vs FD (exact envelope) | 8.528e-06 | 8.528e-06 | **LG truncation** |
| `test_tk_source_functions` numeric-region midpoints | 2.720e-04 | 2.720e-04 | **consumer re-spline** (the numeric `dT/dz` cubic; board `[05-numeric-region-is-now-the-accuracy-floor]`) |
| `test_phase_groups` Oracle 1, exact stand-ins | 6.669e-16 | 6.669e-16 | rounding; no Bessel function involved |
| `test_phase_groups` Oracle 2, exact stand-ins | 3.412e-14 | 3.412e-14 | rounding of `scipy`-built stand-ins |
| `test_phase_groups` realistic, w=0.2, 100/decade | 1.376e-04 | 1.375e-04 | **LG truncation** of `omega` and `d ln M/dz` |
| `test_phase_groups` realistic, w=1/3, 300/decade, `x_q>100` | 1.661e-06 | 7.502e-09 | **Bessel oracle** — the only place in this module where it was the binding term |
| `test_phase_groups` realistic, w=0.2, 300/decade, `x_q>100` | 5.710e-06 | 5.568e-06 | **LG truncation** |
| `test_phase_groups` composed phase, exact constituents | 1.776e-15 | 1.776e-15 | rounding of the composition arithmetic |
| `test_phase_groups` composed phase, `phase_spline` constituents | 3.782e-08 | 3.782e-08 | **consumer re-spline** — `phase_spline`'s ~20 eps `|theta|`, out of scope |
| `test_phase_groups` exact-fixture seam, nodes | 1.514e-04 | 1.514e-04 | **LG truncation** |
| `test_phase_groups` exact-fixture seam, midpoints | 1.031e-03 | 1.031e-03 | **consumer re-spline** (numeric `dT/dz`) |

No row is attributed to "Bessel oracle" while barely moving, and no row that moved is attributed
to anything else. The two apparent exceptions are stated as such: `_test_bessel_value`'s `E_A`
sits at the amplitude sampling floor on both trees, and J000/Y000's *residual* after the
improvement is the quadrature rather than the oracle.

## Observations not acted on

1. **`err_scipy` in `test_tk_source_functions` is the campaign's headline number for that fixture
   and nothing asserts it.** Deviation 5; board issue
   `[08-tk-fixture-scipy-comparison-unasserted]`. One `self.assertLess(err_scipy, 1.0e-7)` would
   lock it in, and it belongs to whoever is allowed to add an assertion to that file.
2. **`DEFAULT_3BESSEL_CHEBYSHEV_ORDER = 12` costs J000 and Y000 three orders and buys the other
   five oracles between 4x and 1500x.** So it is not a constant to raise; it is an argument for
   choosing the order per integrand, or for a convergence check. Board issue
   `[08-3bessel-chebyshev-order-is-now-the-limit]`.
3. **`test_3bessel_analytic.py` spends nearly all of its wall clock on diagnostic plots, not on
   assertions.** `plot_and_compute_3Bessel` evaluates a 250-point `logspace` grid of full
   three-Bessel integrals per case purely to draw a figure, then evaluates the one integral the
   assertion uses. `test_YJJ_log_singularity` has 40 cases and `test_YJJ_log_scaling` — which
   asserts **nothing at all** — has 40 more. See "the suite wall clock" below for the proposal;
   nothing was implemented, per the prompt.
4. **`test_YJJ_log_scaling` has no assertions.** It builds two figures per Y integral and prints
   timings. It is the second most expensive test in the repository and cannot fail except by
   raising. Not touched.
5. **The three random-draw tests are unseeded**, so their pass/fail is a sample. Seeding them
   would make the tolerances above exactly reproducible at the cost of coverage; the tolerances
   were given 25x–57x margin instead. Not changed, because changing the draw is a change of what
   the test covers and this prompt is about thresholds.
6. **`test_bessel_phase.py`'s `_integrate_bessel_J` builds a fresh `bessel_phase` per call.** Four
   calls, 2.5 ms each, so it does not matter here; `test_bessel_two_region.py` has a `build()`
   cache for the same reason and this module could share it if it ever grows.
7. **`test_3bessel_analytic.py` emits two pre-existing `SyntaxWarning`s** for `"$\epsilon$"` at
   `:785` and `:839` (invalid escape `\e`). One character each, and outside this prompt's subject.
8. **The old route is at least an order of magnitude slower on a near-singular YJJ
   configuration.** The six-case fixed-triple script of §2 runs in about nine minutes here and had
   not produced its first line on `f17f2d4` after 29.5 minutes. Consistent with prompt 07's
   diagnosis — the old `_phase_group` handed `adaptive_levin_sincos` a phase differenced from
   independently reconstructed large values, and the subdivider bisects against that noise — but
   not measured to completion, so it is an observation and not a claim. If prompt 09 wants a
   speed statement about the phase-group restructure, this is the case to measure; note standing
   note 3's rule that *construction* speed is not a thing to sell, which this is not.
9. **`test_3bessel_analytic.py`'s figures are written into `test_3bessel_analytic/` at the
   repository root**, one directory per test invocation, ~80 files per singularity run. It is
   git-ignored, so nothing leaks into a commit, but a machine that runs the suite a few times
   accumulates hundreds of megabytes. Same root cause as observation 3; not touched.

## State handed to the next prompt

Prompt 09 records the campaign's measured outcome in `docs/` and re-runs the benchmark. Everything
it needs from here:

### The numbers to carry into `docs/`

1. **The §5 attribution table above is the deliverable prompt 09 inherits.** It is complete and
   every row is measured on both trees, so it can be transcribed rather than re-derived. The
   before tree is `f71401d` for oracle numbers (prompt 01, old construction + new references) and
   `f17f2d4` for the fixtures.
2. **The headline fixture number**, for `docs/lg-phase-and-handover-followup-2026-09.md` §2.4's
   stale `x * 1e-8` claim: `test_tk_source_functions`'s `err_scipy` falls from **1.985e-06 to
   3.021e-08** (w = 1/3) and **1.550e-06 to 2.234e-08** (w = 0.2), landing exactly on `err_T`,
   the fixture's own re-spline error. §2.4's second bullet — "the 05 log's 'vs scipy J_nu' column
   (1.985e-06) is this floor" — is confirmed as a correct historical measurement and is now
   superseded. **Keep it as history; do not delete the measurement.**
3. **The most striking single improvement is not an accuracy figure but a correctness one**: at the
   orders production builds, the old construction's envelope-normalized phase error at `x = 1e12`
   and `1e15` was **1.509 and 1.512** — O(1) — against 1.110e-16 and 5.274e-16 now. That is the
   split evaluation of `DRAFT-PLAN.md` §6.3/§7.4, measured on the production orders rather than on
   3/2 and 7/4.
4. **`test_phase_derivative` at nu = 20.5 and 100.5 would have failed its own 1e-6 contract**
   (2.894e-06 and 6.388e-05) if its lower bound had reached the turning point on the old tree.
   `RECONCILIATION.md` C4 predicted the exclusion mattered; this is the number.
5. **The 221x row** in `test_phase_groups` (`x_q > 100`, 300/decade, w = 1/3: 1.661e-06 →
   7.502e-09) is the only place in the two `ComputeTargets` fixtures where the oracle was the
   binding term. Everything else there is re-spline or LG truncation and moves by less than 1.1x.
   If `docs/` claims a downstream improvement, this is the one to claim, and the honest framing is
   "the oracle stopped being the limit", not "the fixtures got better".
6. **Do not claim the three-Bessel integrals are now good to 1e-14 generally.** Five of the seven
   oracles are (2.5e-14 to 1.5e-13), but J000 and Y000 sit at 1.4e-10 and 4.8e-11 because of
   `DEFAULT_3BESSEL_CHEBYSHEV_ORDER = 12`, and the worst of 42 random draws was 1.7e-09. Board
   `[07-generic-K-product-rounding]` is a further caveat at large `max_x` for non-resonant
   triples.

### Board state

* **Closed:** `[07-abserr-bounds-truth-is-now-an-unexpected-success]` (the decorator is gone and
  the assertion passes) and `[05-3bessel-analytic-not-run-to-completion]` — see the wall-clock
  section for what was and was not run.
* **Opened:** `[08-3bessel-chebyshev-order-is-now-the-limit]`,
  `[08-tk-fixture-scipy-comparison-unasserted]`,
  `[08-test-three-bessel-tolerances-unassigned]`, `[08-3bessel-plot-cost-dominates-the-suite]`.
* **Untouched and still prompt 09's:** `[00-plan-vs-tree-corrections]`,
  `[00-qsi-three-bessel-levin-excluded]`, `[03-draft-plan-tail-coefficient-wrong]`,
  `[06-measure-bessel-phase-num-chunks]`, `[07-generic-K-product-rounding]`.

### Interfaces and constants a later reader may need

```python
# LiouvilleGreen/tests/test_bessel_phase.py
LOW_ORDER_TARGET        = 1.0e-11   # E_theta, E_A at nu = 3/2, 5/2
LOW_ORDER_DERIV_TARGET  = 1.0e-9    # theta' relative, low order
HIGH_ORDER_TARGET       = 1.0e-6    # everything at nu >= 20.5
CORNER_TARGET           = 1.0e-14   # cached 40-digit corners, and the log input mode
_check_points(data, max_x)          # linspace + x_star + near-region nodes + midpoints
_score(nu, xs, data, tier)          # -> ((E_theta, x), (E_A, x), (E_deriv, x))
class TestProductionOrdersAtLargeArgument   # nu = 1/2 and 5/2 through x = 1e15

# LiouvilleGreen/tests/test_3bessel_analytic.py
ABS_TOLERANCE = 1e-8                # was 1e-6
REL_TOLERANCE = 1e-7                # was 1e-5
SINGULARITY_ABS_TOLERANCE = 1e-3    # unchanged
SINGULARITY_REL_TOLERANCE = 1e-2    # unchanged
plot_and_compute_3Bessel(..., phase_atol=DEFAULT_PHASE_ATOL,
                              amplitude_rtol=DEFAULT_AMPLITUDE_RTOL, ...)  # renamed kwargs
```

`test_3bessel_analytic.py`'s two `phase_rtol` keyword arguments no longer exist; a caller passing
them will raise `TypeError`. Nothing in the tree calls that helper from outside the module.
