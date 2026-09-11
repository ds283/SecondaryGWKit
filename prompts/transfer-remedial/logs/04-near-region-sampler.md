# Log 04 — The near-region sampler: branch tracking and two-sided adaptivity

**Prompt:** prompts/transfer-remedial/04-near-region-sampler.md
**Commit:** *(this commit; SHA not self-embedded)* — Add the branch-tracked near-region Bessel sampler
**Model:** Claude Opus 5
**Date:** 2026-09-10
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

Two new files. No existing file was modified except the campaign board and the project-wide issue
index.

### `LiouvilleGreen/bessel_near_region.py` (new, 1032 lines)

The near region of the two-region construction: sample the exponentially scaled Hankel function
below `x_star`, branch-track its argument into a continuous `r_nu`, and interpolate `r` and
`ell = log a` in `u = log x` on adaptively bisected Chebyshev–Lobatto panels.

Public surface, verbatim:

```python
# --- constants ---------------------------------------------------------------------------------
AMPLITUDE_BAND_LO = 0.99
AMPLITUDE_BAND_HI = 8.0
DEFAULT_PANEL_DEGREE = 16
MIN_PANEL_DEGREE = 8
DEFAULT_INITIAL_PANEL_WIDTH = 0.5          # e-folds
BRANCH_SAFETY_FRACTION = 0.5               # of pi, per node gap
BRANCH_CONSISTENCY_FRACTION = 0.25         # of pi, snap discrepancy
DEFAULT_MAX_REFINEMENT_PASSES = 24
DEFAULT_MAX_PANELS = 8192
DEGENERATE_SPAN_TOL = 1e-12
INTERPOLANT_RANGE_SLACK = 1e-9
TWO_PI = 2.0 * math.pi

# --- exceptions --------------------------------------------------------------------------------
class NearRegionError(ValueError)
class AmplitudeBandError(NearRegionError)

# --- sampling ----------------------------------------------------------------------------------
scaled_hankel_phase_constant(nu: float) -> float
check_amplitude_band(nu, x, a, hankel=None, lo=AMPLITUDE_BAND_LO, hi=AMPLITUDE_BAND_HI) -> None
sample_scaled_hankel(nu: float, x, band: Tuple[float, float] = None)   # -> (a, raw_angle, hankel)
residual_log_derivative(x, a)                                          # x (a^-2 - 1)

# --- interpolation -----------------------------------------------------------------------------
class PiecewiseChebyshev:
    breakpoints: np.ndarray
    coefficients: List[np.ndarray]
    n_panels: int
    __call__(u)
    derivative() -> PiecewiseChebyshev
    on_panel(index: int, u)

# --- the result and the builder ------------------------------------------------------------------
@dataclass(frozen=True) class NearRegionData        # fields listed under "State handed on"
build_near_region(nu, x_lo, x_star, phase_atol, amplitude_rtol,
                  deriv_rtol=None, degree=DEFAULT_PANEL_DEGREE,
                  initial_panel_width=DEFAULT_INITIAL_PANEL_WIDTH,
                  max_refinement_passes=DEFAULT_MAX_REFINEMENT_PASSES,
                  max_panels=DEFAULT_MAX_PANELS,
                  amplitude_band=(AMPLITUDE_BAND_LO, AMPLITUDE_BAND_HI),
                  anchor_residual=None, tail_terms=DEFAULT_TAIL_TERMS) -> NearRegionData
```

The module imports only `math`, `numpy`, `numpy.polynomial.chebyshev`, `scipy.special.hankel1e`
and `LiouvilleGreen.bessel_tail`. It does **not** import `bessel_phase`, `phase_spline` or
`range_reduce_mod_2pi`, and uses no `np.unwrap`, no `simple_mod_2pi` and no root solve — asserted
through the AST in `TestModuleHygiene`, not by a text search, because several of those names are
discussed in the docstrings.

How each of `DRAFT-PLAN.md` §7.3's four branch-tracking requirements is met:

1. **Anchor consistent with both `J` and `Y`.** `arg S_nu` is the argument of
   `sqrt(pi x/2) exp(i(pi nu/2 + pi/4)) hankel1e(nu, x)`, and `hankel1e` carries `J + iY`, so the
   anchor uses both. The anchor is taken at the **top** node rather than the bottom, and the
   otherwise-free constant integer-cycle offset is fixed by `bessel_tail.tail_residual(nu, x_star)`
   — see Deviation 4. No root solve.
2. **Refine before accepting.** Two independent conditions per node gap. The crude one is
   `max|dr/d log x| * h <= 0.5 pi` with `dr/d log x = x(a^-2 - 1)` exact from the sampled `a`; the
   sharp one is that the resolved value lie within `0.25 pi` of the value predicted by spectrally
   integrating that same estimator across the gap. Both are checked before a panel is accepted,
   and either failure bisects it.
3. **Track the continuous branch.** A single downward walk over the global node list, re-anchoring
   each prediction on the previous resolved node so the integration error cannot accumulate over
   the ~90 cycles the region spans at `nu = 1000.5`.
4. **Validate continuity and derivative behaviour across boundaries.** Lobatto grids place a node
   on every panel edge and adjacent panels interpolate the same sampled value there, so `C0` is
   exact by construction; `test_continuity_at_panel_boundaries` measures the one-sided limits
   *at* the edge (through `on_panel`, not across a finite step) and gets `r` jumps of 2.8e-16
   (`nu = 2.5`), 5.7e-14 (100.5) and 4.5e-13 (1000.5), all below `phase_atol`. `C1` is not imposed
   and the `dr/du` jump is reported rather than asserted: 4.1e-13, 8.1e-11, 8.8e-09.

The two explicit prohibitions of §7.3 are honoured: endpoint principal-angle differences are never
used on their own, and there is no fixed-grid `np.unwrap` in the tracker.

### `LiouvilleGreen/tests/test_bessel_near_region.py` (new, 850 lines)

27 tests in 12 classes, all scored against `bessel_reference` (prompt 01). Runtime **0.65 s**.
Every one of the prompt's nine test items is covered; the map is in "Verification performed".

## Deviations from the prompt

### 1. Piecewise Chebyshev panels, not quintic splines — IMPLEMENTATION CHOICE

The prompt makes quintic the starting candidate and says "Piecewise Chebyshev is a valid
alternative if it makes error estimation simpler; if you take it, say why." Taken, for three
reasons in order of weight:

- **Error estimation.** The Chebyshev coefficient tail is a per-panel error estimate that costs
  nothing beyond the fit, so refinement can be driven panel-by-panel without a second grid. A
  quintic spline has no equivalent local estimator; the alternative would have been a
  halve-the-grid refinement comparison, which doubles the sample count on every pass.
- **Derivatives.** `dr/du` comes from differentiating the Chebyshev series exactly, rather than
  from a spline's own piecewise-polynomial derivative whose accuracy at the interval ends is the
  weakest part of the fit — and `DRAFT-PLAN.md` §4.7 locates every derivative maximum in the
  interval adjacent to the turning point.
- **Interpolation-boundary continuity.** Lobatto grids put a node on each panel edge, so `C0` is
  exact rather than a tolerance to check (measured above at 2.8e-16 to 4.5e-13).

Against it: the result is `C0` but not `C1`, where a quintic spline would be `C4`. Measured
`dr/du` jumps are 4.1e-13 to 8.8e-09, i.e. below the derivative budget everywhere, and the
derivative is not read from `r` on the shipped route anyway (`theta' = exp(-2 ell)`).

Degree 16 is a *stronger* local approximation than quintic, so `DRAFT-PLAN.md` §4.7's finding —
cubic misses 1e-9 at every order, quintic at 250/e-fold misses it at `nu = 100.5` — is respected
rather than contradicted; degree remains not the acceptance criterion.

### 2. Accuracy of the alternative derivative route is measured and reported, never used to drive refinement — IMPLEMENTATION CHOICE

A criterion on `1 + r_u/x` was implemented first, as the Chebyshev tail of the differentiated `r`
series against `deriv_rtol * x * theta'`. **It does not converge, and the failure mode is
instructive**, so the measurement is recorded here rather than lost:

| pass | panels | panels failing this criterion | worst ratio to threshold |
|---|---:|---:|---:|
| 5 | 51 | 1 | 1.62 |
| 6 | 52 | 2 | 1.16 |
| 7 | 54 | 3 | 3.90 |
| 8 | 57 | 6 | 3.24 |
| 9 | 63 | 12 | 10.7 |

Every other criterion was satisfied from pass 5 onward. The mechanism is that differentiating an
interpolant amplifies the sampling floor of `r` by roughly `N^2/h`: as panels shrink the estimate
*grows* while the threshold stays fixed, so bisection makes the criterion worse and the failing
count doubles every pass. Left in, the build ran to 6193 panels and 99089 nodes at `nu = 1000.5`
and still reported `converged = False`.

What ships instead: refinement is driven by the value accuracy of `r` and `ell` plus the two
branch conditions, and `achieved_deriv_alt_relerr` reports the alternative route's measured
agreement. That is sound because the *shipped* derivative route is `theta' = exp(-2 ell)`, whose
relative error is exactly twice `ell`'s and is therefore contracted by `amplitude_rtol`
(`README.md` §2 (f), `DRAFT-PLAN.md` §7.3). Measured, the alternative route meets every budget
anyway: worst 1.4e-13 at low order and 6.9e-11 at `nu = 1000.5`, against targets of 1e-9 and 1e-6.

Alternative considered and rejected: keep the criterion but floor the threshold at an estimated
noise level. Rejected because the floor would be a fitted constant with no measurement behind it,
and because it would silently stop refining for a reason the caller could not see — whereas a
reported number is visible.

### 3. The degenerate near region returns a one-node structure, not two — IMPLEMENTATION CHOICE

The prompt's test item 7 says the sampler "is either never called or returns a degenerate two-node
structure". It returns a **one**-node structure: `log_x_nodes` has length 1, `n_panels == 0`,
`degenerate == True`, and `r_interp` / `log_a_interp` are constants. A second node would have to
sit above `x_star`, and §4 forbids sampling there for any reason. At `nu = 1/2`, where this case
actually arises, the constant is not an extrapolation but the exact answer: `mu - 1 = 0` makes
`r == 0` and `a == 1` identically at every `x`. Asserted to 1e-15 in
`TestDegenerateHalfOrder`.

The constant interpolants are given a nominal unit-width domain about `u_lo`, because a zero-width
panel cannot be mapped to `[-1, 1]`.

### 4. The anchor is taken at the top node, with its integer cycle from the tail series — IMPLEMENTATION CHOICE

The prompt suggests "`arg S_nu` at the lowest node is the natural anchor". The shipped code
anchors on `arg S_nu` at the **highest** node and fixes the constant integer-cycle offset from
`bessel_tail.tail_residual(nu, x_star)`:

```python
r[top] = raw_angle[top] + 2 pi * round((tail_residual(nu, x_star) - raw_angle[top]) / (2 pi))
```

Why. The residual is free up to a constant `2 pi n` given `J` and `Y` alone, and *some* convention
must be chosen. Anchoring at the bottom and folding into `(-pi, pi]` would put `r(x_0)` at
`-0.9498` instead of `570.820039` at `nu = 1000.5`, i.e. a whole construction offset by exactly
`-91 * 2 pi` from the branch the tail series, `RECONCILIATION.md` C2 and
`bessel_reference_data.json` all use. Prompt 05 stitches the two regions and compares the
interpolant against the series at `x_star`; making that comparison carry an integer-cycle
correction is an invitation to a sign error that no test at the level of `J` and `Y` would catch,
because `E_theta` is blind to it.

Only the **integer** is taken from the series, so none of the series' own accuracy (at most
`phase_atol`, by construction of `x_star`) enters the residual: the value is the sampled
`arg S_nu`. The anchor is at the top rather than the bottom because that is where the series is
valid. `anchor_residual` overrides it, so a test can pin the anchor without the tail series at all.

Measured: `r(x_0)` comes out at 0.615480 (`nu = 3/2`), 0.757566 (7/4), 1.183200 (5/2), 11.443651
(20.5), 57.104238 (100.5) and **570.820039** (1000.5) — `RECONCILIATION.md` C2 and the prompt 01
corner table, to every digit either records. `test_the_residual_is_on_the_tail_continuous_branch`
pins all six.

### 5. The rotation constant is reduced mod 2 pi before multiplication by pi — IMPLEMENTATION CHOICE

`scaled_hankel_phase_constant(nu)` returns `pi * fmod((2 nu + 1)/4, 2)` rather than
`pi nu/2 + pi/4`. For a half-integer order `(2 nu + 1)/4` is exact in binary and `fmod` is exact,
so the constant costs one rounding of `pi * q` with `|q| <= 2`, i.e. `<= 7e-16`. The naive form at
`nu = 1000.5` is `1572.367...`, whose ulp is **2.27e-13** and which is wrong by **6.57e-14** even
after an `fmod` reduction — at or above the 2.96e-13 sampling floor `DRAFT-PLAN.md` §4.4 tabulates
for that order. `a_nu` is likewise formed as `sqrt(pi x/2) |hankel1e|` rather than as the modulus
of the rotated value.

This is a choice about how `S_nu` is formed inside new code the prompt asked for, not a change to
anything existing. `test_the_rotation_constant_is_reduced_before_multiplication` pins both the
exact `pi/2` and the two costs of not doing it.

### 6. The `nu = 1000.5` branch-tracking test runs at the 1e-11 budget, because `wraps_tracked` depends on `x_star` — STRUCTURALLY REQUIRED

The prompt's test item 4 asks that `wraps_tracked` equal `90 +/- 1`, citing
`RECONCILIATION.md` C2's 90.05 cycles. That figure is measured over `[x_0, 100 nu]`. Prompt 03's
remainder-tested crossover is **not** `100 nu` and is not a fixed multiple of `nu`: at `nu = 1000.5`
it is `58.09 nu` (budget 1e-11) or `11.22 nu` (budget 1e-6). Since `r(x_star)` is 8.69 rad in the
first case and 44.60 in the second, the number of `2 pi` boundaries crossed is **89** and **83**
respectively — the second is outside `90 +/- 1` and no implementation choice can change it.

So the test is run on the 1e-11 build, where `wraps_tracked == 89` and the prompt's acceptance
holds. This is recorded as structurally required rather than quietly satisfied because prompt 08
would otherwise re-assert `90 +/- 1` on a loose-budget build and fail. `wraps_tracked` is a
property of the constructed interval, not of the order.

### 7. The two-sided-coarsening test asserts a factor above 2, and the factor is not defined below `nu = 20.5` — IMPLEMENTATION CHOICE

Prompt test item 6 asks for a coarsening factor "you measure and record" and fixes no value. The
measured median-gap ratio between the top decade and the turning-point decade is **2.44**
(`nu = 20.5`), **2.66** (100.5) and **6.01** (1000.5); the test asserts `> 2.0` for the last two.
At `nu <= 2.5` the factor is exactly 1.00, because no panel anywhere fails a criterion at the
initial 0.5-e-fold spacing and no refinement happens at all — there is nothing to coarsen relative
to. The second half of the assertion is the one that carries the design intent and holds at every
order: the top decade runs at 26.5–77.9 nodes per e-fold, i.e. 3.2x to 9.4x **coarser** than the
prototype's uniform 250 per e-fold.

## Verification performed

Everything below was run, not reasoned. Environment: Python 3.12.14, NumPy 2.2.4, SciPy 1.15.2,
mpmath 1.3.0, macOS arm64.

### The prompt's nine test items

| item | where | outcome |
|---|---|---|
| 1 accuracy, low orders, three point sets | `TestLowOrderAccuracy.test_phase_and_amplitude` | pass |
| 2 derivative, low orders | `TestLowOrderAccuracy.test_phase_derivative` | pass |
| 3 plausibility band, both sides | `TestAmplitudePlausibilityBand` (4 tests) | pass |
| 4 branch tracking, negative control + positive result | `TestBranchTracking` (4 tests) | pass |
| 5 accuracy, high orders | `TestHighOrderAccuracy` (2 tests) | pass |
| 6 two-sided adaptivity | `TestTwoSidedAdaptivity` | pass |
| 7 `nu = 1/2` needs no near region | `TestDegenerateHalfOrder` | pass |
| 8 consistency of the two derivative routes | `TestDerivativeRoutes` (2 tests) | pass |
| 9 cost | `TestCost` | pass, recorded below |

Plus `TestScaledHankelSampling` (the exactness of §2 (a) on samples alone, and the rotation
constant), `TestInterpolantStructure` (§7.3 item 4), `TestRefinementCap` and `TestModuleHygiene`.

### Cost and structure, per order, budget 1e-11

Best of three builds.

| nu | `x_star` | nodes | panels | passes | build | `wraps_tracked` |
|---|---:|---:|---:|---:|---:|---:|
| 1/2 | 1.0e-05 | 1 | 0 | 0 | 0.000 s | 0 |
| 3/2 | 34.4119 | 113 | 7 | 0 | 0.002 s | 0 |
| 7/4 | 54.5456 | 113 | 7 | 0 | 0.002 s | 0 |
| 5/2 | 64.4688 | 113 | 7 | 0 | 0.002 s | 0 |
| 20.5 | 664.389 | 145 | 9 | 2 | 0.005 s | 1 |
| 100.5 | 4199.98 | 177 | 11 | 3 | 0.008 s | 9 |
| 1000.5 | 58122.8 | **817** | 51 | 5 | **0.042 s** | **89** |

The expensive case is 0.042 s, three orders below the ~30 s at which the prompt asked for a
prominent warning. Nothing here needs a cost caveat for prompt 08.

### Accuracy, per order, over nodes + midpoints + endpoint intervals combined

Scored against `bessel_reference` at tier `scipy`. Every maximum is quoted with its `x`.

| nu | `E_theta` | at `x` | `E_A` | at `x` | `theta'` relerr | at `x` | `1 + r_u/x` relerr | at `x` |
|---|---:|---|---:|---|---:|---|---:|---|
| 1/2 | 2.220e-16 | 1e-05 | 1.110e-16 | 1e-05 | 4.441e-16 | 1e-05 | 4.441e-16 | 1e-05 |
| 3/2 | 9.770e-15 | 16.6093 | 2.331e-14 | 9.53078 | 4.663e-14 | 9.53078 | 1.416e-13 | 1.41421 |
| 7/4 | 1.787e-14 | 14.9834 | 3.153e-14 | 12.9894 | 6.284e-14 | 12.9894 | 1.437e-13 | 1.67705 |
| 5/2 | 9.881e-15 | 16.5104 | 2.143e-14 | 16.752 | 4.285e-14 | 16.752 | 8.105e-14 | 2.44949 |
| 20.5 | 4.619e-14 | 149.245 | 4.929e-14 | 201.328 | 9.848e-14 | 201.328 | 2.103e-12 | 20.4939 |
| 100.5 | 2.714e-12 | 4192.92 | 1.432e-12 | 4196.45 | 2.864e-12 | 4196.45 | 1.859e-11 | 100.499 |
| 1000.5 | 4.045e-11 | 56540.2 | 3.443e-11 | 57926.2 | 6.887e-11 | 57926.2 | 6.887e-11 | 57926.2 |

Against the acceptance table: low orders need `1e-11` for `E_theta`/`E_A` and `1e-9` for the
derivative — margins of **317x to 1023x** and **1.6e4x to 2.3e4x**. High orders need `1e-6`
throughout — margins of **1.45e4x** at `nu = 1000.5` (the derivative, the tightest of the three;
2.5e4x on `E_theta`) and better below.

Two observations on where the maxima fall. At low order the phase and amplitude maxima are in the
middle of the region, not at the turning point; the `1 + r_u/x` maximum *is* at the turning point
at every order up to 100.5, consistent with `DRAFT-PLAN.md` §4.7. At `nu = 1000.5` every maximum
has moved to the top of the region, which is the reference's doing rather than the
construction's — see the next table.

### The high-order numbers are limited by SciPy, not by this construction

Scored against the committed 40-digit `mpmath` corners
(`LiouvilleGreen/tests/bessel_reference_data.json`), which is the only comparison in which the
reference is better than the construction at every order:

| nu | `x` | `E_theta` | `E_A` | `theta'` relerr | `r_ours - r_cached` |
|---|---:|---:|---:|---:|---:|
| 20.5 | 20.4939 | 2.554e-15 | 1.332e-15 | 3.331e-15 | -1.776e-15 |
| 20.5 | 30.74085 | 1.305e-15 | 4.441e-16 | 3.331e-16 | -3.553e-15 |
| 20.5 | 205 | 5.940e-15 | 0 | 2.220e-16 | +6.661e-16 |
| 100.5 | 100.4988 | 8.521e-14 | 1.021e-14 | 2.043e-14 | +9.948e-14 |
| 100.5 | 150.7481 | 7.938e-15 | 0 | 2.220e-16 | +1.421e-14 |
| 100.5 | 1000 | 5.718e-15 | 2.220e-16 | 0 | +7.105e-15 |
| 100.5 | 1005 | 1.314e-14 | 2.220e-16 | 2.220e-16 | -8.882e-16 |
| 1000.5 | 1000.5 | 8.954e-14 | 2.354e-14 | 4.707e-14 | -2.274e-13 |
| 1000.5 | 1500.75 | 4.269e-13 | 9.992e-15 | 2.021e-14 | +1.705e-13 |
| 1000.5 | 10005 | 4.383e-13 | 5.285e-14 | 1.058e-13 | +2.629e-13 |

So at `nu = 1000.5` the construction is good to **4.4e-13**, not the 4.0e-11 the SciPy-scored sweep
reports; the difference is the `jv`/`yv` floor board note 15 records (8.4e-12 in `r` at that order).
`test_against_the_cached_mpmath_corners` asserts 1e-11 on all three metrics and 1e-9 on
`|r_ours - r_cached|` — the last being the assertion that the branch is right and not merely the
value.

### Branch tracking

**Negative control** (`test_fixed_density_unwrap_fails_at_nu_1000`), a uniform 250.1 samples per
e-fold at `nu = 1000.5` with a quintic `make_interp_spline` through `np.unwrap`'d angles:

- maximum residual advance per interval **3.6820 rad**, above `pi` (`RECONCILIATION.md` C2 measures
  3.6847);
- `E_theta` at the grid midpoints **1.9954e+00** — order unity, against the prompt's `> 0.1`;
- `E_theta` **at the nodes themselves 5.4388e-11**.

That last number is asserted too, because it is the whole trap: an integer `2 pi` error leaves `J`
and `Y` unchanged, so the mis-unwrapped representation looks perfect wherever it was sampled and
is catastrophic between the samples. No density and no interpolation degree repairs it.

**Positive result** (`test_the_shipped_tracker_succeeds_at_nu_1000`): `E_theta = 4.0447e-11` at
`x = 56540.2` against a `1e-6` target, `wraps_tracked = 89` (within `90 +/- 1`), maximum branch
advance **1.5635 rad** against the `0.5 pi = 1.5708` limit, maximum snap discrepancy **6.637e-11
rad** against the `0.25 pi = 0.7854` limit.

The branch-safety criterion is therefore *binding* at `nu = 1000.5` — it is what sets the node
count there, not accuracy. Per-order figures:

| nu | max advance (rad) | max snap (rad) | top-decade nodes/e-fold | turning-point-decade nodes/e-fold | coarsening |
|---|---:|---:|---:|---:|---:|
| 3/2 | 0.0194 | 8.743e-16 | 35.3 | 35.3 | 1.00 |
| 7/4 | 0.0268 | 2.179e-15 | 26.5 | 26.5 | 1.00 |
| 5/2 | 0.0425 | 1.527e-15 | 28.3 | 28.3 | 1.00 |
| 20.5 | 0.2685 | 1.066e-14 | 26.6 | 64.7 | 2.44 |
| 100.5 | 1.2981 | 1.271e-12 | 28.3 | 75.3 | 2.66 |
| 1000.5 | 1.5635 | 6.637e-11 | 77.9 | 467.9 | **6.01** |

### The plausibility band

`hankel1e(100.5, 1e9)` was re-confirmed to be exactly `-0j` on this tree, `np.isfinite` to accept
it, and `np.log(np.abs(.))` to give `-inf`; the band rejects it with `nu`, `x`, the `hankel1e`
value and the band in the message, and the message says so. The upper side is exercised with a
fabricated `a = 42.0`. Every sampled `a` in every build lies inside `[0.99, 8.0]`:
min 1.0000000000, max 3.5459685 at `nu = 1000.5`, matching `RECONCILIATION.md` §3.1 to every digit
it records.

### The `achieved_*` estimators, their method, and how they compare

**Method.** After the refinement loop settles, each panel is resampled at the `u`-midpoints of
consecutive Lobatto nodes — points strictly interior to the panel and never nodes of it — and the
interpolants are compared against those fresh samples. `achieved_phase_abserr` is
`max |r_interp - r_sampled|`, `achieved_amplitude_relerr` is `max |ell_interp - ell_sampled|`, and
`achieved_deriv_relerr` is `max |exp(-2 ell_interp) a_sampled^2 - 1|`. The same points feed
`achieved_deriv_alt_relerr` for the `1 + r_u/x` route. This is a residual estimate at interior
points, the first of the two methods the prompt offers; it costs one extra `hankel1e` batch per
panel per pass and no second grid.

| nu | `achieved_phase_abserr` | `achieved_amplitude_relerr` | `achieved_deriv_relerr` | `achieved_deriv_alt_relerr` | worst `\|dr\|` at the cached corners | ratio |
|---|---:|---:|---:|---:|---:|---:|
| 1/2 | 0 | 0 | 0 | 0 | 2.220e-16 | — |
| 3/2 | 6.106e-16 | 7.377e-16 | 1.554e-15 | 5.407e-14 | 1.665e-16 | 3.67 |
| 7/4 | 1.374e-15 | 8.257e-16 | 1.665e-15 | 7.538e-14 | 7.772e-16 | 1.77 |
| 5/2 | 1.110e-15 | 7.082e-16 | 1.332e-15 | 5.862e-14 | 4.441e-16 | 2.50 |
| 20.5 | 1.599e-14 | 2.831e-15 | 5.773e-15 | 9.255e-13 | 3.553e-15 | 4.50 |
| 100.5 | 1.279e-13 | 2.228e-14 | 4.463e-14 | 9.922e-12 | 9.948e-14 | 1.29 |
| 1000.5 | 1.251e-12 | 2.833e-13 | 5.665e-13 | 2.168e-11 | 2.629e-13 | 4.76 |

**They over-report by 1.3x to 4.8x, at every order tested.** That is the direction the prompt asks
for — an honest over-estimate is better than an optimistic under-estimate, because prompt 05
propagates these into `theta_abserr`. `test_the_achieved_estimates_bracket_the_measured_errors`
asserts the estimator never falls below a tenth of the corner-measured error. The structural
limitation is recorded as a §3 issue: the estimator resamples the *same* function it interpolates,
so it measures interpolation error and would not see a systematic bias in `hankel1e` itself.

### The panel degree, swept

`DEFAULT_PANEL_DEGREE = 16` and `MIN_PANEL_DEGREE = 8` are set from a sweep at
`phase_atol = amplitude_rtol = 1e-11`, total nodes for degrees 4/6/8/10/12/16/20/24:

| nu | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5/2 | 28673* | 1765 | 265 | 111 | 97 | 113 | 141 | 169 |
| 100.5 | 32769* | 6355 | 793 | 321 | 217 | 177 | 201 | 241 |
| 1000.5 | 18433* | 12589 | 1393 | 861 | 829 | 817 | 841 | 841 |

`*` = hit the panel cap with `converged = False`. Flat to within 15 % from degree 12 to 24, rising
sharply below 10, and pathological at 4. The mechanism at low degree is not approximation but the
estimator: `_coefficient_tail` sums the last three Chebyshev coefficients, and with only five
coefficients those three are a large fraction of the function itself rather than of its truncation
error, so the criterion never clears however small the panel becomes — the degree-4 runs above
have interior residuals of 2.0e-15 (`nu = 5/2`) and 1.3e-13 (100.5) while still reporting
`converged = False`. `build_near_region` therefore refuses `degree < 8` and refuses odd degrees,
with the reason in the message; `test_the_panel_degree_floor_is_enforced` pins both.

### The refinement cap

`build_near_region(..., max_refinement_passes=1)` at `nu = 1000.5` returns `converged = False` with
`achieved_phase_abserr = 7.143e-05` against the requested 1e-11 and 15 panels — reported, not
silently accepted, with the decision to raise left to prompt 05. The unconstrained build of the
same case returns `converged = True` at pass 5.

### Test suite

Per module, from the repository root, following board standing note 13 (prefer per-module runs;
the discovery run is dominated by a pre-existing cost):

| module | result | time |
|---|---|---:|
| `test_bessel_near_region` (new) | OK, 27 tests | 0.65 s |
| `test_bessel_phase` | OK, 4 tests | 0.20 s |
| `test_bessel_reference` | OK, 12 tests | 0.49 s |
| `test_bessel_tail` | OK, 19 tests | 0.17 s |
| `test_range_reduce` | OK, 4 tests | 0.00 s |
| `test_scipy_bessel_domain` | OK, 7 tests | 0.03 s |
| `test_three_bessel` | OK, 2 tests | 4.60 s |
| `test_3bessel_analytic` | **not run separately** | pre-existing cost; log 01 records >15 min without finishing |

The prompt also asks for `PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .`
to pass. **It was attempted and did not complete**: run under a 900 s cap it was still executing
when the cap fired, having emitted no test summary at all — only SciPy's standing
`rtol is too small` `UserWarning` from the *existing* `bessel_phase` ODE. That reproduces board
standing note 13 and log 01's measurement exactly, and it is a pre-existing property of
`test_3bessel_analytic.py` rebuilding `bessel_phase` objects per case, not something this commit
introduces: the seven modules above, which are every module in the directory except that one, run
to completion in 6.3 s in total. Per README standing note 13, the per-module results are the
verification of record.

`test_bessel_phase`'s `test_phase_derivative` — the campaign's standing regression gate at 1e-6 —
passes unchanged. No production file outside the new module was touched, so it could not have been
otherwise, but it was run rather than assumed.

`./venv/bin/python -m black --check LiouvilleGreen/` reports 20 files unchanged.

## Observations not acted on

1. **`hankel1e`'s own phase floor is not separately characterised.** The corner comparisons above
   bound the *total* error of samples plus interpolation, and the interior-residual estimator
   cannot separate the two because both sides of its comparison come from `hankel1e`. Deviation 5
   removed one identified contributor (the rotation constant), but `DRAFT-PLAN.md` §4.4's residual
   2.96e-13 at `nu = 1000.5` was measured before that fix and has not been re-measured. Nothing in
   the campaign's budgets is threatened either way. Opened as `[04-achieved-estimates-exclude-the-sampling-floor]`.
2. **Node count grows linearly in `nu` once branch safety binds.** Total variation of `r` across
   the near region is `~ pi nu / 2`, and each gap may advance at most `pi/2`, so at least `~nu`
   gaps are needed however accurate the interpolation is. At `nu = 1000.5` that is the binding
   constraint (817 nodes, advance 1.5635 against a 1.5708 limit). It is not a problem at any order
   the campaign supports — 0.042 s — but it means the cost of extending the order ceiling is
   linear, not logarithmic. Not acted on: extending the ceiling is out of scope (`README.md` §1.1).
3. **The initial panel width of 0.5 e-folds is not tuned.** It sets the tail density at ~32 nodes
   per e-fold and is why no refinement at all happens below `nu = 20.5`. A wider initial panel
   would be cheaper still; the current setting was left alone because cost is already three orders
   inside the prompt's threshold and a tuned constant would need its own justification.
4. **`achieved_deriv_relerr` is exactly twice `achieved_amplitude_relerr` at every order**, to two
   digits, which is the first-order propagation `delta theta'/theta' = -2 delta ell` being exact.
   It is reported separately anyway, because prompt 05 consumes the derivative estimate directly
   and should not have to know the relation.
5. **`bessel_tail.construction_min_x` is duplicated in `bessel_reference`** and the tail module's
   docstring says so deliberately (production code must not import a test module). This module
   imports neither and takes `x_lo` as an argument, so the duplication is untouched.

## State handed to the next prompt

### Import path and public API

    from LiouvilleGreen import bessel_near_region

`NearRegionData` is a frozen dataclass. Field names, verbatim and in order:

```python
nu: float
x_lo: float
x_star: float
log_x_nodes: np.ndarray            # u = log x, ascending, panel edges appearing once
a_nodes: np.ndarray                # sampled a_nu at those nodes
r_nodes: np.ndarray                # continuously tracked r_nu, NOT reduced mod 2 pi
r_interp: PiecewiseChebyshev       # u -> r,   .derivative() -> u -> dr/du
log_a_interp: PiecewiseChebyshev   # u -> ell, .derivative() -> u -> d ell/du
achieved_phase_abserr: float
achieved_amplitude_relerr: float
achieved_deriv_relerr: float       # for theta' = exp(-2 ell), the shipped route
achieved_deriv_alt_relerr: float   # for 1 + r_u/x, reported only -- see Deviation 2
phase_atol: float
amplitude_rtol: float
deriv_rtol: float
interp_degree: int
refinement_passes: int
wraps_tracked: int
n_panels: int
panel_edges: np.ndarray
converged: bool
degenerate: bool
amplitude_band: Tuple[float, float]
max_branch_advance: float          # max |dr/d log x| * h over any node gap, rad
max_branch_snap: float             # max |r_resolved - r_predicted| over any node, rad
```

plus the property `n_nodes -> int`. And the builder:

```python
build_near_region(nu, x_lo, x_star, phase_atol, amplitude_rtol,
                  deriv_rtol=None,                     # defaults to 2 * amplitude_rtol
                  degree=16,                           # even, >= MIN_PANEL_DEGREE = 8
                  initial_panel_width=0.5,             # e-folds
                  max_refinement_passes=24,
                  max_panels=8192,
                  amplitude_band=(0.99, 8.0),
                  anchor_residual=None,                # integer cycle only; see Deviation 4
                  tail_terms=3) -> NearRegionData
```

`PiecewiseChebyshev` exposes `breakpoints`, `coefficients`, `n_panels`, `__call__(u)`,
`derivative()` and `on_panel(index, u)`. Scalars in, scalars out; arrays in, arrays out. It
**raises** `ValueError` outside `[log x_lo, log x_star]` (slack `1e-9` in `u`), deliberately:
above `x_star` the closed-form tail is the representation, and silently extrapolating a Chebyshev
panel there would be a wrong answer with no symptom.

**Raising behaviour.** `NearRegionError` (a `ValueError`) for a non-positive `x_lo`, `x_star < x_lo`,
a non-positive budget, or a `degree` that is not an even integer `>= MIN_PANEL_DEGREE` (8). `AmplitudeBandError` (a
`NearRegionError`) for any sample outside the band — construction fails loudly, and no sample is
dropped, interpolated across or warned about and kept.

### Things prompt 05 must not get wrong

1. **`r` is on the tail-continuous branch, `r -> 0` as `x -> infinity`.** It is *not* folded into
   `(-pi, pi]` and must never be: `r(x_0) = 570.820039` at `nu = 1000.5`. This is the same branch
   as `bessel_tail.tail_residual` and as `bessel_reference_data.json`, so the crossover comparison
   at `x_star` is a direct subtraction with no integer bookkeeping. That is by construction, not
   by luck — see Deviation 4.
2. **`theta' = exp(-2 ell)` is the shipped route**, read off `log_a_interp` as a value.
   `1 + r_u/x` is the independent consistency check and is *reported*, not contracted; do not
   substitute it. The Wronskian check `a^2 theta' = 1` is a tautology here — measured at 1e-15,
   which is exactly the point.
3. **`achieved_*` are practical estimators, not supremum bounds**, and they measure interpolation
   error against resampled `hankel1e`. They over-report the corner-measured error by 1.3x–4.8x at
   every order tested, so propagating them into `theta_abserr` is conservative in the right
   direction; but they do not include a `hankel1e` bias term (§3 issue
   `[04-achieved-estimates-exclude-the-sampling-floor]`).
4. **`wraps_tracked` is a property of `[x_lo, x_star]`, not of the order.** 89 at `nu = 1000.5`
   with the 1e-11 crossover (`x_star = 58.09 nu`) and **83** with the 1e-6 one
   (`x_star = 11.22 nu`). `RECONCILIATION.md` C2's 90.05 is measured to `100 nu`. Any later test
   that re-asserts `90 +/- 1` must say which build it means.
5. **`nu = 1/2` returns `degenerate = True`, one node, `n_panels = 0`,** and constant interpolants
   with `r == 0`, `ell == 0`. Prompt 03's `tail_crossover` returns `x_star = x_0 = 1e-5` there, so
   the whole domain is closed form; prompt 05 may either skip the call or accept the degenerate
   structure, but must not treat `n_nodes == 1` as a failure.
6. **`converged = False` is the signal to act on**, not an exception. The refinement cap reports
   unmet accuracy through `converged` plus the `achieved_*` values; the policy decision — raise,
   warn, or accept — is prompt 05's, in one place.

### Values prompt 05 will want without re-deriving them

Budget 1e-11 on both phase and amplitude, defaults otherwise.

| nu | `x_star` | nodes | panels | build | `wraps_tracked` | `E_theta` | `E_A` | `theta'` relerr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1/2 | 1.0e-05 | 1 | 0 | 0.000 s | 0 | 2.2e-16 | 1.1e-16 | 4.4e-16 |
| 3/2 | 34.4119 | 113 | 7 | 0.002 s | 0 | 9.8e-15 | 2.3e-14 | 4.7e-14 |
| 7/4 | 54.5456 | 113 | 7 | 0.002 s | 0 | 1.8e-14 | 3.2e-14 | 6.3e-14 |
| 5/2 | 64.4688 | 113 | 7 | 0.002 s | 0 | 9.9e-15 | 2.1e-14 | 4.3e-14 |
| 20.5 | 664.389 | 145 | 9 | 0.005 s | 1 | 4.6e-14 | 4.9e-14 | 9.8e-14 |
| 100.5 | 4199.98 | 177 | 11 | 0.008 s | 9 | 2.7e-12 | 1.4e-12 | 2.9e-12 |
| 1000.5 | 58122.8 | 817 | 51 | 0.042 s | 89 | 4.0e-11 | 3.4e-11 | 6.9e-11 |

Errors are against the SciPy tier over nodes, midpoints and endpoint intervals combined; at
`nu >= 100.5` they are dominated by the reference, and the corner-scored values are 8.5e-14 and
4.4e-13. **Cost is never a consideration**: the whole seven-order sweep builds in 0.06 s.

### Test helpers prompt 05 and prompt 08 can reuse

`LiouvilleGreen/tests/test_bessel_near_region.py` exposes three module-level functions that are
not tests and are meant to be imported: `build(nu, budget)` (cached), `reconstruct(nu, x, data)`
returning `(sin theta, -cos theta, A, theta'_from_ell, theta'_from_r)` by angle addition, and
`score(nu, x, data, tier)` returning every metric with the `x` at which it peaks. `point_sets(data)`
returns the three separate point sets `DRAFT-PLAN.md` §9 Stage 1 requires.
