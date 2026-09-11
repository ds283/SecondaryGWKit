# Log 05 — Replace `bessel_phase` with the two-region construction

**Prompt:** prompts/transfer-remedial/05-two-region-construction.md
**Commit:** *(this commit; SHA not self-embedded)* — Rebuild the Bessel phase from a two-region amplitude and residual
**Model:** Claude Opus 5
**Date:** 2026-09-10
**Result:** COMPLETE WITH DEVIATIONS

---

## What shipped

### `LiouvilleGreen/bessel_phase.py` — rewritten (293 → 1010 lines)

The ODE, the root solve, the full-phase spline and the `(div_2pi, mod_2pi)` reduction are gone;
`grep -n "solve_ivp\|root_scalar\|phase_spline\|simple_mod_2pi"` returns nothing, and a test asserts
it. `_weff_sq`, `MINIMUM_X` and `DEFAULT_SAMPLE_DENSITY` are gone with them (nothing outside the
file referenced any of them; `QuadSourceIntegral_debug.py:216` mentions `MINIMUM_X` only inside a
commented-out line).

**Module constants** (each carries its measurement in a `#:` comment):

| name | value | role |
|---|---|---|
| `DEFAULT_PHASE_ATOL` | `1e-11` | absolute phase budget; the §10 low-order target used directly |
| `DEFAULT_AMPLITUDE_RTOL` | `1e-11` | relative amplitude budget |
| `MIN_SUPPORTED_NU` | `0.5` | declared order floor |
| `MAX_SUPPORTED_X` | `1e16` | declared `max_x` ceiling, set by `sin`/`cos`, not by Amos |
| `DOMAIN_CUSHION` | `0.01` | unchanged clamp cushion from the old `XSplineWrapper` |
| `SAMPLED_PHASE_FLOOR` | `3e-13` | `hankel1e` phase floor the near region's estimator cannot see |
| `SAMPLED_AMPLITUDE_FLOOR` | `3e-13` | same, relative, for `a` |
| `EVALUATION_FLOOR` | `4 eps = 8.88e-16` | the angle-addition arithmetic itself |
| `MIN_INITIAL_PANEL_WIDTH` | `1e-3` | floor on what `sample_points` may request |

**Exceptions.** `BesselPhaseError(RuntimeError)` and `BesselPhaseAccuracyError(BesselPhaseError)`.
Derived from `RuntimeError` because that is what the old domain check raised.

**Free functions.**

```python
c_nu(nu: float) -> float                 # pi/4 - pi nu/2, exactly as before
c_nu_reduced(nu: float) -> float         # pi * fmod((1 - 2 nu)/4, 2); sin/cos use only this
```

**`XSplineWrapper`** — kept, importable, same `__call__(x, is_log=False)` behaviour and same
1 % clamp cushion. Gained one method, `_resolve(x, is_log) -> (raw_x, log_x)`, and array support.

**`BesselAmplitude(XSplineWrapper)`** — the `"mod"` member.

```python
__call__(x, is_log=False)     # A_nu(x) = sqrt(2/(pi x)) a_nu(x) == sqrt(J^2 + Y^2)
a(x, is_log=False)            # the normalized amplitude a_nu
log_deriv(x, is_log=False)    # d log A/dx = -1/(2x) + ell_u(log x)/x
```

**`BesselPhaseFunction`** — the `"phase"` member. Attributes `nu`, `min_x`, `max_x`, `x_star`,
`c_nu`, `c_nu_reduced`, `theta_abserr`.

```python
raw_theta(x, x_is_log=False)                                  # x + c_nu + r; eps*theta limited
theta_mod_2pi(x, x_is_log=False)                              # atan2 of the split pair, (-pi, pi]
theta_deriv(x, x_is_log=False, log_derivative=False)          # exp(-2 ell) / 1 + r'
residual(x, x_is_log=False)                                   # r_nu, tail-continuous branch
residual_log_deriv(x, x_is_log=False)                         # dr/d log x
sin_cos_theta(x, x_is_log=False) -> (sin theta, cos theta)    # angle addition
theta_deriv_from_residual(x, x_is_log=False)                  # 1 + r_u/x, the check route
bessel_j(x, is_log=False)
bessel_y(x, is_log=False)
```

Every accessor takes scalars or arrays and returns the matching type.

**`_TwoRegionCorrections`** — private; the shared core that dispatches `r`, `ell`, `a`, `theta'` and
the two log-derivatives to the tail for `x >= x_star` and to the near-region interpolants below. No
blending, taper or overlap of any kind.

**`_DeprecatedQ(XSplineWrapper)`** — the `"Q"` member, `__call__(x, is_log=False)` returning
`raw_theta(x)/x`.

**`bessel_phase()`** — new signature:

```python
bessel_phase(nu, max_x, sample_points=None, atol=None, rtol=None,
             phase_atol=None, amplitude_rtol=None, tail_terms=3,
             interp_degree=16, strict_accuracy=True) -> dict
```

Returned dict: the six members §3 requires (`phase`, `mod`, `bessel_j`, `bessel_y`, `min_x`,
`max_x`), plus `Q` and `phi` unchanged in name, plus `nu`, `x_star`, `theta_abserr`,
`amplitude_relerr`, `theta_deriv_relerr`, `accuracy` (an 18-key breakdown), `accuracy_met`,
`crossover` (the `TailCrossover`) and `near_region` (the `NearRegionData`, or `None`).

### `LiouvilleGreen/tests/test_bessel_two_region.py` — new, 24 tests, 3.8 s

Module-level helpers `build(nu, max_x, **kw)` (cached), `point_sets(data)` (nodes / midpoints /
endpoint intervals / tail, kept separate) and `score(nu, xs, data, tier)`. Test classes:
`TestAcceptanceTable`, `TestDegenerateOrder`, `TestCrossover`, `TestSplitEvaluation`,
`TestDerivativeRoutes`, `TestStructure`, `TestCostAndTheCliff`, `TestZerosAndExtrema`,
`TestNoLegacyMachinery`.

---

## Deviations from the prompt

### 1. The near region is built over the whole of `[min_x, x_star]`, even when `max_x < x_star` — IMPLEMENTATION CHOICE

§2.1's sketch is `x_star = tail_crossover(...)` then `build_near_region(nu, x_lo, x_star, ...)`, and
it does not say what happens when the caller's `max_x` falls *below* `x_star`. That case is not
hypothetical: it is what the standing regression gate does. `test_bessel_phase.test_high_order`
builds `nu = 1000.5` with `max_x = 10005` while `x_star = 58122.8`, and `test_phase_derivative`
builds `nu = 100.5` with `max_x = 1005` while `x_star = 4199.98`.

Alternatives considered:

* **clamp the near region to `min(x_star, max_x)`** — cheaper, and it is what a first reading of
  §2.1 suggests. Rejected on two grounds. (i) `build_near_region` fixes the residual's integer cycle
  from `tail_residual(nu, x_top)`, and the series is certified to `phase_atol` only at `x_star`; at
  `nu = 1000.5, x = 1.5 x_0 = 1500` the three-term series is already wrong by ~0.33 rad and its
  terms have stopped decreasing usefully, so a caller with a small `max_x` at high order would get
  an anchor whose integer cycle is not reliably determined. (ii) §2.2 requires the crossover
  agreement to be computed at construction time and to raise if it fails; with a clamped near region
  there is no common point at which to compare, so the campaign's central structural test would
  simply not run for those builds.
* **build to `x_star` always** — chosen. The cost of the unused part is bounded by the whole near
  region, which log 04 measures at 817 nodes and 0.042 s at `nu = 1000.5`; measured here, the
  `nu = 1000.5, max_x = 10005` build takes 0.043 s against ~0.025 s if clamped. Buying an
  unconditional seam test and a certified anchor for 18 ms is a good trade.

The served domain is still `[min_x, max_x]` exactly: `_resolve` rejects or clamps against `max_x`,
and nothing above it is evaluable.

### 2. `sin`/`cos` use `c_nu` reduced mod `2 pi`; `raw_theta` uses the unreduced `c_nu` — IMPLEMENTATION CHOICE

§2.4 gives the angle addition on `d = c_nu + r_nu(x)` but does not say how `c_nu` itself is formed.
Formed directly, `c_nu = pi/4 - pi nu/2` is `-1570.87` at `nu = 1000.5`, whose ulp is **2.3e-13** —
so `d` would carry that much error into every evaluated angle at high order, for no reason, since
only `sin d` and `cos d` are ever needed and any multiple of `2 pi` may be removed first.
`c_nu_reduced(nu) = pi * fmod((1 - 2 nu)/4, 2)` is exact in binary through the `fmod` for a
half-integer order and costs one rounding of `pi q` with `|q| <= 2`, below 7e-16. This is the same
device `bessel_near_region.scaled_hankel_phase_constant` already uses for the rotation constant, so
the two are consistent.

`raw_theta` keeps the unreduced `c_nu`, because it is documented to return `x + c_nu + r` and its
own precision limit is `eps * theta` in any case. A test asserts the two differ by an exact integer
number of cycles at every order. **Nothing about the zero-point itself changed**: `c_nu` is still
`pi/4 - pi nu/2` and is exported as such.

### 3. A `sample_points` interpretation had to be chosen, and it is "floor on initial density" — IMPLEMENTATION CHOICE

§2.7 offers "make it a *floor* on initial density or deprecate it explicitly, and say which". Chosen:
floor on initial density, no warning, because the argument retains a meaning under the new design
and `docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py:82` passes it deliberately. The
translation is `density = sample_points / (log max_x - log min_x)` nodes per e-fold, then
`initial_panel_width = min(0.5, degree/density)`, clamped below at `MIN_INITIAL_PANEL_WIDTH = 1e-3`.
It can only make the initial grid finer, never coarser, so refinement still decides the outcome.
Tested both ways: `sample_points=20000` increases the node count, `sample_points=1` does not
decrease it.

### 4. `strict_accuracy` is a parameter, defaulting to raising — IMPLEMENTATION CHOICE

§2.8 says "either raise, or return the object with an unmistakable `accuracy_met = False` and a
warning. State which you chose; raising is the safer default". Shipped: **raising is the default**
(`BesselPhaseAccuracyError`, naming nu, the interval, the pass and panel counts, all three achieved
values and all three budgets). `strict_accuracy=False` gives the second behaviour — a
`RuntimeWarning` with the same message and `accuracy_met = False` on the object — because a
diagnostic that wants to *see* an unmet accuracy would otherwise have no way to obtain the object at
all. `accuracy_met` is present either way, so a consumer need not know which mode was used.

### 5. An `EVALUATION_FLOOR` term was added to the reported accuracy — IMPLEMENTATION CHOICE

`DRAFT-PLAN.md` §10 requires phase error to be allocated among six mechanisms, the last of which is
"evaluation arithmetic". At `nu = 1/2` all five representation terms are identically zero (the
series is exact), so the object initially reported `theta_abserr = 0.0` while the measured
phase-pair error against the committed 40-digit corners was 1.11e-16 — half an ulp of unity, from
the four multiplications and one addition of the angle addition. A declared error *below* the true
error is exactly the failure `theta_abserr` exists to prevent, so four ulps are added to
`theta_abserr` and `amplitude_relerr` (eight to `theta_deriv_relerr`). It is negligible wherever a
near region exists.

### 6. The `SAMPLED_PHASE_FLOOR` term, discharging board issue `[04-achieved-estimates-exclude-the-sampling-floor]` — IMPLEMENTATION CHOICE

That issue's "next step" is that prompt 05 "either adds a documented sampling-floor term to what it
publishes as `theta_abserr` or states the assumption explicitly". The first option was taken:
`SAMPLED_PHASE_FLOOR = SAMPLED_AMPLITUDE_FLOOR = 3e-13`, added to the near region's `achieved_*`
values before they enter the reported accuracy. The value is `DRAFT-PLAN.md` §4.4's measured
2.96e-13 `hankel1e` phase floor at `nu = 1000.5`, rounded up and applied at every order.

The alternative — a fresh `mpmath` measurement of the post-prompt-04 floor — was not done, because
it is out of this prompt's scope and because the floor is not binding: the crossover's series
remainder (2.5e-12) dominates it by a factor 8 at every order. The issue is therefore **narrowed**
rather than closed: the *published* number now includes a floor term, and what remains open is only
whether 3e-13 is the right size for it.

### 7. `nu >= 1/2` is now rejected explicitly, and `max_x <= 1e16` with it — IMPLEMENTATION CHOICE

§2.1 requires the supported `(nu, x_max)` domain to be stated and requests outside it rejected
"rather than extending behaviour implicitly". Neither bound existed before: the old code accepted
any order and any `max_x`, and simply produced an unjustifiable initial condition below `nu = 1/2`
and hung above `x ~ 2.5e15`. Both new bounds are therefore *narrowings* of the nominally accepted
set, not widenings, and the lower `x` bound `sqrt(nu^2 - 1/4)` is untouched. Every call site in the
repository builds `nu >= 0.5` (checked: `main.py`, both audit scripts, the benchmark tier and all
seven test modules). `MAX_SUPPORTED_X = 1e16` is where `RECONCILIATION.md` §1's verification of
`np.sin`/`math.sin` stops; prompt 09 measures how far the representation actually goes.

### 8. `theta_mod_2pi` returns `(-pi, pi]`, where the old one returned `fmod(theta, 2 pi)` in `(-2 pi, 2 pi)` — STRUCTURALLY REQUIRED

§2.5 requires the bounded accessor to be `atan2(sin_theta, cos_theta)`, whose range is `(-pi, pi]`.
The old `phase_spline.theta_mod_2pi` returned `fmod(theta, TWO_PI)`, which keeps the sign of `theta`
and so lives in `(-2 pi, 2 pi)`. Both are valid representatives modulo `2 pi` and the change is
forced by the prompt. Checked that no consumer depends on the interval: `AdaptiveLevin` uses the
value only inside `np.sin`/`np.cos` (`levin_quadrature.py:1103-1105`, `:1120-1121`), and
`three_bessel_integrals._phase_group` only sums such values and takes trigonometric functions of the
sum. Verified by running both `test_three_bessel` and `test_3bessel_analytic`.

### 9. A `"nu"` key was added to the returned dict — IMPLEMENTATION CHOICE

Prompt 07 will need the order to assemble phase groups, and `ComputeTargets/QuadSourceIntegral.py:680`
documents at length that it has to check the order *numerically* because "`bessel_phase()` does not
record its own order". Adding the key costs nothing and breaks nothing (that check is numeric and
still passes). It makes that docstring stale, but `ComputeTargets/` is not mine to edit; recorded
under "Observations not acted on".

### 10. The module docstring says "chunked full-phase spline module" rather than naming it — STRUCTURALLY REQUIRED

§5's acceptance criterion is that `grep -n "solve_ivp\|root_scalar\|phase_spline\|simple_mod_2pi"`
returns *nothing* from the file. Explaining what was removed and why — which §1 requires — normally
means naming those things. The prose therefore refers to "the repository's chunked full-phase spline
module", "the full-phase spline" and "the cycle-count-plus-remainder `(div_2pi, mod_2pi)`
representation" instead. All four removals are still explained with their measured justifications.

---

## Verification performed

All runs from the repository root with `PYTHONPATH=. ./venv/bin/python`. Every number below was
printed by a run, not reasoned about.

### The new module: `test_bessel_two_region`, 24 tests, **OK in 3.785 s**

**Acceptance table, low orders through `x_max = 1e7`** (target 1e-11 for `E_theta`/`E_A`, 1e-9 for
`theta'`), scored against the SciPy tier, point sets kept separate:

| nu | set | `E_theta` @ x | `E_A` @ x | `theta'` relerr @ x |
|---|---|---|---|---|
| 0.5 | tail | 8.660e-15 @ 14.6357 | 1.821e-14 @ 14.6357 | 3.642e-14 @ 14.6357 |
| 1.5 | nodes | 2.482e-12 @ 34.4119 | 2.545e-13 @ 34.4119 | 5.090e-13 @ 34.4119 |
| 1.5 | midpoints | 9.326e-15 @ 16.2579 | 2.331e-14 @ 9.53078 | 4.663e-14 @ 9.53078 |
| 1.5 | endpoints | 6.661e-16 @ 1.41623 | 4.441e-16 @ 1.41669 | 8.882e-16 @ 1.41669 |
| 1.5 | tail | 2.482e-12 @ 34.4119 | 2.545e-13 @ 34.4119 | 5.090e-13 @ 34.4119 |
| 1.75 | nodes | 1.787e-14 @ 14.9834 | 2.387e-14 @ 16.5104 | 4.774e-14 @ 16.5104 |
| 1.75 | midpoints | 1.454e-14 @ 16.9 | 3.153e-14 @ 12.9894 | 6.284e-14 @ 12.9894 |
| 1.75 | endpoints | 7.772e-16 @ 54.305 | 5.551e-16 @ 1.68207 | 8.882e-16 @ 1.68207 |
| 1.75 | tail | 1.786e-12 @ 54.5456 | 1.599e-13 @ 54.5456 | 3.195e-13 @ 54.5456 |
| 2.5 | nodes | 2.484e-12 @ 64.4688 | 1.358e-13 @ 64.4688 | 2.713e-13 @ 64.4688 |
| 2.5 | midpoints | 9.548e-15 @ 11.7464 | 2.143e-14 @ 16.752 | 4.285e-14 @ 16.752 |
| 2.5 | endpoints | 8.882e-16 @ 2.45086 | 6.661e-16 @ 64.1945 | 8.882e-16 @ 64.2161 |
| 2.5 | tail | 2.484e-12 @ 64.4688 | 1.357e-13 @ 64.4688 | 2.713e-13 @ 64.4688 |

Worst low-order `E_theta` is **2.484e-12** (`nu = 5/2` at `x = x_star = 64.4688`), a factor 4.0
inside the 1e-11 target; worst `E_A` **2.545e-13**; worst derivative **5.090e-13**, 1960x inside the
1e-9 target. Every maximum sits at `x_star`, i.e. it is the *series remainder*, not interpolation —
the `safety = 0.25` margin of prompt 03's crossover at work.

**Acceptance table, high orders** (target 1e-6 throughout), `max_x = max(1000, 10 nu)`:

| nu | set | `E_theta` @ x | `E_A` @ x | `theta'` relerr @ x |
|---|---|---|---|---|
| 20.5 | nodes | 2.418e-12 @ 664.389 | 4.929e-14 @ 201.328 | 9.848e-14 @ 201.328 |
| 20.5 | midpoints | 4.188e-14 @ 149.245 | 4.208e-14 @ 119.55 | 8.393e-14 @ 119.55 |
| 20.5 | endpoints | 6.550e-15 @ 20.5135 | 3.109e-15 @ 20.4951 | 5.995e-15 @ 20.4951 |
| 20.5 | tail | 2.462e-12 @ 665.752 | 1.321e-14 @ 665.752 | 2.642e-14 @ 665.07 |
| 100.5 | nodes | 2.194e-13 @ 996.012 | 1.985e-13 @ 897.004 | 3.970e-13 @ 897.004 |
| 100.5 | midpoints | 1.966e-13 @ 981.668 | 2.287e-13 @ 801.934 | 4.579e-13 @ 801.934 |
| 100.5 | endpoints | 1.023e-13 @ 100.5 | 1.243e-14 @ 100.5 | 2.465e-14 @ 100.5 |
| 1000.5 | nodes | 6.307e-12 @ 8175.84 | 3.325e-12 @ 7625.74 | 6.650e-12 @ 7625.74 |
| 1000.5 | midpoints | 8.776e-12 @ 9962.19 | 4.047e-12 @ 9505.02 | 8.095e-12 @ 9505.02 |
| 1000.5 | endpoints | 5.753e-13 @ 1000.56 | 4.541e-14 @ 1000.6 | 9.104e-14 @ 1000.6 |

Worst high-order value anywhere is **8.776e-12**, i.e. **1.1e5 times inside** the 1e-6 target. Note
standing note 15: at `nu >= 100.5` these are at or below SciPy's own floor (1.2e-12 and 8.4e-12 in
`r`), so they are *bounded by the reference*, not by the construction. Scored against the 40-digit
`mpmath` corners instead, the same objects give 8.610e-14 (`nu = 100.5`) and 2.443e-13
(`nu = 1000.5`).

**Declared accuracy is an over-estimate at every order** (the `theta_abserr` honesty test):

| nu | corner `E_theta` | declared `theta_abserr` | corner `E_A` | declared `amplitude_relerr` |
|---|---|---|---|---|
| 0.5 | 1.110e-16 | 8.882e-16 | 2.220e-16 | 8.882e-16 |
| 1.5 | 2.220e-16 | 2.500e-12 | 4.441e-16 | 3.007e-13 |
| 1.75 | 7.910e-16 | 2.500e-12 | 4.441e-16 | 3.008e-13 |
| 2.5 | 5.274e-16 | 2.501e-12 | 4.441e-16 | 3.007e-13 |
| 20.5 | 1.324e-13 | 2.501e-12 | 1.332e-15 | 3.028e-13 |
| 100.5 | 8.610e-14 | 2.502e-12 | 1.021e-14 | 3.223e-13 |
| 1000.5 | 2.443e-13 | 2.524e-12 | 5.285e-14 | 5.833e-13 |

**`nu = 1/2` is exact and all tail.** `near_region is None`, `x_star == min_x == 1e-5`,
`residual(x) == 0.0` and `a(x) == 1.0` *identically* over 500 points, `c_nu(1/2) == 0.0`. Against
the exact half-integer tier over the whole domain to `1e7`: `E_theta = 2.220e-16`,
`E_A = 2.220e-16`.

**Crossover agreement at `x_star`**, both budgets 1e-11:

| nu | `x_star` | phase agreement | amplitude agreement |
|---|---:|---:|---:|
| 1.5 | 34.4119 | 2.499e-12 | 2.541e-13 |
| 1.75 | 54.5456 | 2.491e-12 | 1.601e-13 |
| 2.5 | 64.4688 | 2.501e-12 | 1.361e-13 |
| 20.5 | 664.389 | 2.501e-12 | 1.354e-14 |
| 100.5 | 4199.98 | 2.502e-12 | 2.442e-15 |
| 1000.5 | 58122.8 | 2.524e-12 | 2.220e-16 |

A factor 4.0 inside the phase budget at every order, and 39x to 4.5e4x inside the amplitude budget.
The stored `accuracy["crossover_phase_agreement"]` equals the measured value bit for bit (asserted
with `delta=0.0`).

**Seam continuity** at `x_star(1 +/- 1e-9)`, with the genuine variation across the pair subtracted
(see the note in the test — not subtracting it measures 5.6e-11 at `nu = 3/2`, which is the true
change in `r`, not a discontinuity):

| nu | `r` | `a` | `theta'` |
|---|---:|---:|---:|
| 1.5 | 2.499e-12 | 2.539e-13 | 5.082e-13 |
| 2.5 | 2.501e-12 | 1.360e-13 | 2.719e-13 |
| 20.5 | 2.501e-12 | 1.373e-14 | 2.723e-14 |
| 100.5 | 2.503e-12 | 2.311e-15 | 4.843e-15 |
| 1000.5 | 2.521e-12 | 4.089e-16 | 3.736e-16 |

**Split evaluation against 70-digit `mpmath` at the supplied argument** (`mpf(float(x))`):

| nu | x | naive `sin(x + d)` | angle addition |
|---|---|---:|---:|
| 1.5 | 1e3 | 5.951e-14 | **0** |
| 1.5 | 1e7 | 1.210e-10 | **0** |
| 1.5 | 1e12 | 2.723e-06 | 2.220e-16 |
| 1.5 | 1e15 | **4.725e-02** | 1.110e-16 |
| 1.75 | 1e3 | 2.243e-14 | 1.110e-16 |
| 1.75 | 1e7 | 1.140e-09 | 2.220e-16 |
| 1.75 | 1e12 | 4.831e-06 | 1.110e-16 |
| 1.75 | 1e15 | **3.620e-02** | 2.776e-17 |

`RECONCILIATION.md` §1's table for `nu = 3/2` is reproduced to three figures at `1e7`, `1e12` and
`1e15` (1.210e-10, 2.723e-06, 4.725e-02). `E_A` at those points is 0 or 2.220e-16. The bounded-angle
accessor agrees with `sin`/`cos` of the split pair to <= 1e-15 at all four arguments.

**The two derivative routes** (README §2 (f)), over nodes + midpoints + endpoints + tail:

| nu | `exp(-2 ell)` vs `1 + r_u/x` | shipped vs reference | alternative vs reference |
|---|---:|---:|---:|
| 0.5 | 0 (both identically 1) | 3.642e-14 | 3.642e-14 |
| 1.5 | 1.421e-13 | 5.090e-13 | 5.090e-13 |
| 1.75 | 1.434e-13 | 3.195e-13 | 3.195e-13 |
| 2.5 | 9.293e-14 | 2.713e-13 | 2.713e-13 |
| 20.5 | 2.099e-12 | 9.848e-14 | 2.103e-12 |
| 100.5 | 1.805e-11 | 4.579e-13 | 1.807e-11 |
| 1000.5 | 3.958e-11 | 8.095e-12 | 3.961e-11 |

The shipped route beats the alternative against the reference by 21x (`nu = 20.5`), 39x (100.5) and
4.9x (1000.5) — consistent with `DRAFT-PLAN.md` §4.7's "4x to 15x in 7 of 8 configurations", and
with log 04's warning that the alternative amplifies the sampling floor. The Wronskian check
`a^2 theta' = 1` is deliberately not asserted: it is a tautology here, and the test docstring says
so.

**Cost, and the cliff.** `nu = 5/2`, build times 0.0024 s (`max_x = 1e3`), 0.0025 s (`1e7`),
0.0023 s (`1e11`), 0.0034 s (`1e13`), 0.0022 s (`1e15`) — flat, as designed, because the near region
does not depend on `max_x`. **`max_x = 3e15` builds in 0.0025 s and `8.6e15` in 0.0024 s**; both are
cases where the old construction does not return (log 01: abandoned at 60 s). At the declared
ceiling `max_x = 1e16` the build takes 0.0027 s and gives `E_theta = 2.220e-16`, `E_A = 0` against
the cached `nu = 5/2, x = 1e16` corner. At the `x = 2.5e15` corner, `E_theta = 2.220e-16`, `E_A = 0`.
No ratio against the old build is claimed (`RECONCILIATION.md` C1: it would be noise).

**Zeros and extrema.** 12 zeros of `J_nu`, 12 of `Y_nu` and 12 extrema of `J_nu` per order, located
by `brentq`: `E_theta` = 2.447e-12 (`nu = 3/2`, at `x = 34.514`), 1.713e-15 (5/2), 7.480e-15 (20.5);
`E_A` = 2.482e-13, 2.065e-14, 1.332e-15. At each zero of `J_nu` the reconstructed value is below
`1e-11` times the local envelope. The relative error against `jv()` itself is deliberately not
asserted there.

**Structural.** `phi == 0.0` exactly at every order tested. The dict keys `phase`, `mod`, `Q`,
`phi`, `bessel_j`, `bessel_y`, `min_x`, `max_x` are all present and `mod` is an `XSplineWrapper`.
Raw and logarithmic input modes agree. Nothing is sampled above `x_star`, and `x_star` is below
`scipy_reference_max_x(nu)` at every order. No non-finite value appears in `mod`, `mod.a`,
`residual`, `theta_deriv`, `bessel_j` or `bessel_y` over 2000-point sweeps at `nu = 5/2, 100.5,
1000.5`. `nu = 0.25`, `max_x = 1e17` and `bessel_phase(100.5, 50.0)` all raise `BesselPhaseError`;
`phase_atol = amplitude_rtol = 1e-20` raises. The deprecation warning fires when `atol`/`rtol` are
supplied, does not fire when they are not, and says "take precedence" when the new arguments are
supplied too.

### The standing regression gate: `test_bessel_phase`, 4 tests, **OK in 0.177 s** (baseline 1.2 s)

`test_phase_derivative`'s measured margin against its 1e-6 contract, which is what the prompt asks
for explicitly:

| nu | worst relerr | at x | margin |
|---|---:|---|---:|
| 2.5 | 3.077e-14 | 84.5741 | **3.25e7 x** |
| 20.5 | 3.830e-14 | 193.323 | **2.61e7 x** |
| 100.5 | 4.249e-13 | 735.197 | **2.35e6 x** |

Log 01 records the old margins as 16.5x, 44.3x and 45.2x, so the gate's headroom improves by
1.97e6, 5.89e5 and 5.21e4 respectively. `DRAFT-PLAN.md` §4.7 predicted 2.09e-8 at `nu = 100.5` for
the new derivative route at 250 samples per e-fold; the adaptive panels deliver 4.249e-13, four and
a half orders better. The test is untouched.

### The rest of `LiouvilleGreen/tests`

Run per module, per `RECONCILIATION.md` §3.4 and standing note 13.

| module | result | time |
|---|---|---|
| `test_bessel_reference` | OK | (69 tests across five modules in 0.782 s) |
| `test_scipy_bessel_domain` | OK | " |
| `test_bessel_tail` | OK | " |
| `test_bessel_near_region` | OK | " |
| `test_range_reduce` | OK | " |
| `test_bessel_phase` | OK, 4 tests | 0.177 s (baseline 1.2 s) |
| `test_bessel_two_region` | OK, 24 tests | 3.785 s (new) |
| `test_three_bessel` | OK, 2 tests | 4.134 s (baseline 7.5 s) |
| `test_3bessel_analytic` | see below | pre-existing multi-hour module |

`test_3bessel_analytic` is the module `RECONCILIATION.md` §3.4 and log 01 record as not finishing
within 50 minutes and 15 minutes respectively on the planning and prompt-01 machines; it rebuilds
`bessel_phase` objects per case and there are many cases. It was started here and made normal
progress (tests passing, no failures, only the expected `DeprecationWarning`s from its `atol`/`rtol`
calls), but it is a pre-existing cost and **its completion is not asserted in this log**. Its two
tolerance bands (1e-5/1e-6 and 1e-2/1e-3 near singularities) are prompt 08's to revisit. **This is
the one verification step a later prompt or the user may wish to complete on a machine that can give
it hours.**

### Consumers outside `LiouvilleGreen/`, checked but not modified

`ComputeTargets/tests/test_tk_source_functions` and `ComputeTargets/tests/test_phase_groups` — the
two fixture modules that read `mod`, `phase.raw_theta` and a real phase object — **Ran 30 tests,
OK, 4.427 s**. Neither file was touched (prompt 08 owns their tolerances).

### Serialization

Not this prompt's acceptance criterion (prompt 06 owns `BesselPhaseProxy`), but cheap to check and a
nasty surprise if wrong: the returned dict now round-trips through **plain `pickle`** (18148 bytes)
with `bessel_j(137.0)` reproducing exactly. The old dict could not: it held module-level closures
(`bessel_j`, `bessel_y` were local functions), which plain `pickle` cannot serialize and which
relied on Ray's `cloudpickle`. The new members are bound methods of picklable objects.

### Greps required by §5

* `grep -n "solve_ivp\|root_scalar\|phase_spline\|simple_mod_2pi" LiouvilleGreen/bessel_phase.py` —
  **no matches** (exit 1). Asserted as a test.
* "never exceeds a cycle" and equivalents — **absent**, asserted as a test, which also asserts the
  contrary fact (`max_abs_residual > 500` at `nu = 1000.5`; measured 570.820039).

### Formatting

`./venv/bin/python -m black LiouvilleGreen/` — 21 files unchanged, the two touched files
reformatted then clean under `--check`.

---

## Observations not acted on

1. **`ComputeTargets/QuadSourceIntegral.py:680` `_check_bessel_order`'s docstring is now stale.** It
   states that the returned dict "carries phase, mod, Q, phi, bessel_j, bessel_y, min_x, max_x and
   no `nu`", and explains at length that the order therefore has to be checked numerically. There is
   now a `nu` key. The numeric check still works and still passes, and `ComputeTargets/` is out of
   scope for this campaign, so nothing was changed. A later prompt in *that* campaign could simplify
   the check to a direct comparison.

2. **The low-order `E_theta` is entirely the tail series remainder.** Every low-order maximum sits
   exactly at `x_star`, at 2.48e-12 against a 1e-11 target — the interpolated near region is at
   1e-14 and the evaluation is at 1e-16. If a later prompt wants another order of magnitude at low
   order, the cheapest lever by far is `bessel_tail`'s `DEFAULT_CROSSOVER_SAFETY` or a fourth series
   term, not a denser near region. Out of scope here (README §1.1 defers tightening) and recorded
   only so the lever is not looked for in the wrong place.

3. **`plot_besssel_phase.py` is still broken and now differently so.** It read `data["x_min"]`
   (never a key) and called `phase(x)` (never callable), and it also reads `Q`. `Q` survives, but
   the first two failures are unchanged. `RECONCILIATION.md` C3 makes repair-or-delete a user
   decision and prompt 06 owns it; nothing was done here.

4. **`docs/transfer-remedial/measure_bessel_phase.py` and
   `docs/source-remediation-verification/run_quadsource_integrals.py` pass `atol`/`rtol`** and will
   now emit `DeprecationWarning`s. Both still run. `docs/spec-code-audit/scripts/QI_02_analytic_numeric.py`
   does the same. None was touched; prompt 06 migrates `main.py` and the live diagnostics, prompt 09
   the benchmark.

5. **`bessel_tail.tail_crossover_max_x = 4800 max(nu, 1)` is never approached** at the campaign's
   budgets — the tightest crossover used here is `58.09 nu`. It would only bind at a budget around
   1e-17, which already fails for other reasons. Nothing to do; noted because the constant looks
   arbitrary until one checks how far away it is.

---

## State handed to the next prompt

### Import path and public API of `LiouvilleGreen/bessel_phase.py`

    from LiouvilleGreen.bessel_phase import bessel_phase, XSplineWrapper

```python
# --- constants ---------------------------------------------------------------------------------
DEFAULT_PHASE_ATOL       = 1.0e-11
DEFAULT_AMPLITUDE_RTOL   = 1.0e-11
MIN_SUPPORTED_NU         = 0.5
MAX_SUPPORTED_X          = 1.0e16
DOMAIN_CUSHION           = 0.01     # 1% clamp cushion, unchanged from the old XSplineWrapper
SAMPLED_PHASE_FLOOR      = 3.0e-13
SAMPLED_AMPLITUDE_FLOOR  = 3.0e-13
EVALUATION_FLOOR         = 4 * eps  # 8.881784197001252e-16
MIN_INITIAL_PANEL_WIDTH  = 1.0e-3

# --- exceptions --------------------------------------------------------------------------------
class BesselPhaseError(RuntimeError)                    # domain, crossover, seam
class BesselPhaseAccuracyError(BesselPhaseError)        # refinement cap hit

# --- free functions ----------------------------------------------------------------------------
c_nu(nu: float) -> float                 # pi/4 - pi nu/2
c_nu_reduced(nu: float) -> float         # the same, mod 2 pi; used only inside sin/cos

# --- the constructor ---------------------------------------------------------------------------
bessel_phase(nu, max_x, sample_points=None, atol=None, rtol=None,
             phase_atol=None, amplitude_rtol=None, tail_terms=3,
             interp_degree=16, strict_accuracy=True) -> dict
```

`AmplitudeBandError` from `bessel_near_region` propagates unchanged — construction fails loudly, and
this is the guard that catches `hankel1e` returning exactly `-0j` (finite, so `isfinite` passes it).

### The returned dict, every key

| key | type | meaning |
|---|---|---|
| `phase` | `BesselPhaseFunction` | the phase accessors |
| `mod` | `BesselAmplitude` (a `XSplineWrapper`) | `A_nu(x) = sqrt(J^2 + Y^2)` |
| `Q` | `_DeprecatedQ` (a `XSplineWrapper`) | `raw_theta(x)/x`; compatibility only |
| `phi` | `float` | **identically `0.0`** |
| `bessel_j`, `bessel_y` | bound methods | `f(x, is_log=False)` |
| `min_x`, `max_x` | `float` | the served domain |
| `nu` | `float` | **new**; the order |
| `x_star` | `float` | **new**; the crossover actually used |
| `theta_abserr` | `float` | **new**; declared absolute phase error, rad |
| `amplitude_relerr` | `float` | **new**; declared relative amplitude error |
| `theta_deriv_relerr` | `float` | **new**; declared relative `theta'` error |
| `accuracy` | `dict` | **new**; the 18-key breakdown below |
| `accuracy_met` | `bool` | **new**; `False` only under `strict_accuracy=False` |
| `crossover` | `TailCrossover` | **new**; prompt 03's object, verbatim |
| `near_region` | `NearRegionData` or `None` | **new**; prompt 04's object. `None` at `nu = 1/2` |

`accuracy` keys: `phase_atol`, `amplitude_rtol`, `theta_abserr`, `amplitude_relerr`,
`theta_deriv_relerr`, `near_region_phase_abserr`, `near_region_amplitude_relerr`,
`near_region_deriv_relerr`, `near_region_deriv_alt_relerr`, `sampled_phase_floor`,
`sampled_amplitude_floor`, `tail_first_omitted_at_x_star`, `tail_amplitude_at_x_star`,
`crossover_phase_agreement`, `crossover_amplitude_agreement`, `residual_resolution`,
`evaluation_floor`, `max_abs_residual`, `accuracy_met`. The four `near_region_*` entries are `None`
when there is no near region.

### Accessor signatures, verbatim

```python
# phase  -- BesselPhaseFunction
raw_theta(x, x_is_log=False)                            # x + c_nu + r; eps*theta limited
theta_mod_2pi(x, x_is_log=False)                        # atan2(sin, cos) of the split pair, (-pi, pi]
theta_deriv(x, x_is_log=False, log_derivative=False)    # exp(-2 ell) below x_star, 1 + r' above
residual(x, x_is_log=False)                             # r_nu, tail-continuous branch, never mod 2 pi
residual_log_deriv(x, x_is_log=False)                   # dr/d log x
sin_cos_theta(x, x_is_log=False) -> (sin theta, cos theta)
theta_deriv_from_residual(x, x_is_log=False)            # 1 + r_u/x; the check route, never evaluated for values
bessel_j(x, is_log=False)
bessel_y(x, is_log=False)
# attributes: nu, min_x, max_x, x_star, c_nu, c_nu_reduced, theta_abserr

# mod  -- BesselAmplitude
__call__(x, is_log=False)      # A_nu = sqrt(2/(pi x)) a_nu
a(x, is_log=False)             # a_nu
log_deriv(x, is_log=False)     # d log A/dx = -1/(2x) + ell_u/x
```

Scalars in, scalars out; arrays in, arrays out — the old `mod` accepted scalars only. **Note the
keyword names differ between the two objects and this is deliberate compatibility**: `phase` uses
`x_is_log` (matching the old `phase_spline`) and `mod` uses `is_log` (matching the old
`XSplineWrapper`). Prompt 06 should not "fix" one to match the other without checking the call
sites.

### Exactly what happened to `Q`, `phi`, `sample_points`, `XSplineWrapper`, `atol`, `rtol`

* **`Q`** — kept, as an `XSplineWrapper` subclass returning `raw_theta(x)/x`. This *is* the quantity
  the old diagnostic plotted: with `phi` identically zero and no cycle rebasing, `theta/x` and the
  old pre-offset ODE state coincide, so the meaning is preserved rather than silently replaced.
  It inherits `raw_theta`'s `eps * theta` limit. `ComputeTargets/QuadSourceIntegral_debug.py:55,72`
  is the only live consumer; prompt 06 decides whether to migrate it to `residual`.
* **`phi`** — kept, and **identically `0.0`** at every order, `nu = 1/2` included (there is no root
  solve at all, so `DRAFT-PLAN.md` §4.3's caveat about `nu = 1/2`'s interior match point no longer
  applies). Reported rather than dropped, per §4.3.
* **`sample_points`** — kept and honoured, reinterpreted as a **floor on the initial node density**
  (deviation 3). `bessel_tier.py:82` keeps working. No warning is emitted.
* **`XSplineWrapper`** — kept in this module, importable by name, same `__call__` semantics and same
  cushion; `mod` is now an instance of a subclass, so `test_three_bessel.py:10,67`'s import *and*
  its annotation both remain correct.
* **`atol` / `rtol`** — accepted, **ignored**, and warned about. See the next section.

### The deprecation translation and precedence rule, as shipped

* Defaults are `None`, not `config.defaults`' values, so a caller that does not pass them gets no
  warning. `config.defaults.DEFAULT_ABS_TOLERANCE`/`DEFAULT_REL_TOLERANCE` are no longer imported by
  this module.
* Supplying either emits a `DeprecationWarning` (`stacklevel=2`) naming both replacements.
* **`atol` maps to nothing and `rtol` maps to nothing.** They named tolerances of an ODE solve that
  no longer exists, so any translation would be invented. The new defaults are used.
* If old and new are supplied together, **the new win** and the warning text contains
  "phase_atol/amplitude_rtol were also supplied and take precedence".
* Tested three ways in `TestStructure.test_deprecated_tolerance_arguments`.

### `x_star` as actually used, and the per-order measured accuracy

At the shipped defaults (`phase_atol = amplitude_rtol = 1e-11`, `tail_terms = 3`). `E_theta`,
`E_A` and the derivative are the worst over nodes, midpoints, endpoint intervals and tail combined,
scored against the SciPy tier; the corner column is the same object scored against the committed
40-digit `mpmath` corners, and is the honest number at `nu >= 100.5` where SciPy is the limit.

| nu | `x_star` | worst `E_theta` (at x) | worst `E_A` | worst `theta'` relerr | corner `E_theta` | declared `theta_abserr` |
|---|---:|---:|---:|---:|---:|---:|
| 0.5 | 1.0e-05 (= `min_x`) | 8.660e-15 (14.64) | 1.821e-14 | 3.642e-14 | 1.110e-16 | 8.882e-16 |
| 1.5 | 34.4119 | 2.482e-12 (34.41) | 2.545e-13 | 5.090e-13 | 2.220e-16 | 2.500e-12 |
| 1.75 | 54.5456 | 1.786e-12 (54.55) | 1.599e-13 | 3.195e-13 | 7.910e-16 | 2.500e-12 |
| 2.5 | 64.4688 | 2.484e-12 (64.47) | 1.358e-13 | 2.713e-13 | 5.274e-16 | 2.501e-12 |
| 20.5 | 664.389 | 2.462e-12 (665.8) | 4.929e-14 | 9.848e-14 | 1.324e-13 | 2.501e-12 |
| 100.5 | 4199.98 | 2.194e-13 (996.0) | 2.287e-13 | 4.579e-13 | 8.610e-14 | 2.502e-12 |
| 1000.5 | 58122.8 | 8.776e-12 (9962) | 4.047e-12 | 8.095e-12 | 2.443e-13 | 2.524e-12 |

**Every low-order maximum is at `x_star`**, i.e. it is the series remainder, not interpolation.

### Facts prompts 06 and 07 can rely on without re-deriving

1. **`theta_abserr` is ready to pass straight to `AdaptiveLevin`.** It is a float on the phase
   object *and* a top-level dict key, and `levin_quadrature.py:962` accepts either a scalar or a
   callable. It over-reports the corner-measured error by 4x to 8000x at every order (table above),
   which is the honest direction. It includes five separately reported terms: near-region
   interpolation, the `hankel1e` sampling floor (3e-13, deviation 6), the series remainder, the seam
   agreement, and `eps |r|_max`.
2. **`theta` is still a required key of the Levin phase dict** — `_Basis_SinCos.__init__` raises
   without it — so `raw_theta` must be supplied even though Levin never *evaluates* it when
   `theta_mod_2pi` and `theta_deriv` are both present (`:1038`). "Compatibility-only" is not
   "optional".
3. **`theta_mod_2pi` now returns `(-pi, pi]`, not `fmod(theta, 2 pi)`.** Both are valid mod-`2 pi`
   representatives; every consumer checked takes only `sin`/`cos` of it (deviation 8).
4. **Do not build a phase group by differencing `raw_theta` values.** `raw_theta` is `eps * theta`
   limited by construction: 2.2e-1 rad at `x = 1e15`. Prompt 07's `K t + C + R(t)` decomposition
   should combine the leading coefficients (`k + e_nu q + e_sigma s`) and the `c_nu` constants
   *before* multiplying by `t`, and take `R(t)` from `phase.residual`, which is the accessor that
   exists for exactly this. `phase.c_nu` and `phase.c_nu_reduced` are both exposed, and
   `c_nu_reduced` is the one to sum when only `sin`/`cos` of the group are needed.
5. **`residual` is on the tail-continuous branch and is never reduced mod `2 pi`.** It reaches
   570.820039 rad at `x_0` for `nu = 1000.5`. Folding it into `(-pi, pi]` would give a value wrong
   by an exact multiple of `2 pi` and poison every downstream comparison.
6. **The returned dict pickles with plain `pickle`**, which the old one did not (it held local
   closures). Ray's `cloudpickle` path in `BesselPhaseProxy` is therefore strictly easier than
   before, but prompt 06 still owns the actual `ray.put` check.
7. **`near_region` is `None` at `nu = 1/2` and only there** (at the shipped budgets). Anything that
   walks `data["near_region"].log_x_nodes` must handle `None`; `data["crossover"]` is always
   present.
8. **Build cost is flat in `max_x`** (0.0022-0.0034 s for `nu = 5/2` across `1e3` to `1e16`) and
   depends only on the order: 0.043 s at `nu = 1000.5`. Nothing in the campaign should be scheduled
   around construction cost, and **no speed-up ratio against the old build should be claimed**
   (`RECONCILIATION.md` C1: the old build was ~0.1 s over the whole range, so a ratio would be
   noise). The result to claim is that `3e15` and `8.6e15`, which the old construction could not
   complete, now build in 2.5 ms.
9. **Deprecated-argument noise.** `main.py:520-528`, `docs/transfer-remedial/measure_bessel_phase.py`,
   `docs/source-remediation-verification/run_quadsource_integrals.py`,
   `docs/spec-code-audit/scripts/QI_02_analytic_numeric.py`, `LiouvilleGreen/tests/test_three_bessel.py`
   and `LiouvilleGreen/tests/test_3bessel_analytic.py` all pass `atol`/`rtol` and now emit a
   `DeprecationWarning` each. All still run. Prompt 06 owns `main.py` and the live diagnostics;
   prompt 08 owns the two test modules; prompt 09 owns the benchmark. `bessel_tier.py` already wraps
   its build in `warnings.catch_warnings()`.
10. **`test_3bessel_analytic` was started but not seen to completion** (see "Verification
    performed"). It is the pre-existing multi-hour module. Prompt 08 inherits its tolerances; if it
    exposes a *different* limiting error now that the oracle is eight orders better, that is the
    finding `DRAFT-PLAN.md` §9 Stage 4 anticipates, not a nuisance.
