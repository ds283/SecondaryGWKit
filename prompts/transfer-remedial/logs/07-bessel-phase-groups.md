# Log 07 — Preserve the leading term in three-Bessel phase groups

**Prompt:** prompts/transfer-remedial/07-bessel-phase-groups.md
**Commit:** *(this commit; SHA not self-embedded)* — Preserve the leading term in Bessel phase groups
**Model:** Claude Opus 5
**Date:** 2026-09-10
**Result:** COMPLETE WITH DEVIATIONS — **but one acceptance item in prompt §4 is not met, for a
pre-existing reason in a file this prompt forbids touching.** `test_3bessel_analytic.py` fails as a
module on `f9cc891` (this prompt's parent) and on this commit, because
`test_abserr_bounds_truth` — an `@unittest.expectedFailure` — now *passes*, which `unittest`
reports as `FAILED (unexpected successes=1)`. It is prompt 05's eight-order oracle improvement that
did that, not this commit; measured on both trees (Deviation 1). That file is prompt 08's. Read
Deviation 1 before deciding whether to proceed.

## What shipped

### `LiouvilleGreen/three_bessel_integrals.py`

* **`_coefficient_sum(terms)`** (new, `:190`). `math.fsum` of a short list. Used three times per
  group, at construction, for `K`, `C` and `C_reduced`. Its docstring states why the *per-point*
  sums deliberately do not use it (below).
* **`_PhaseGroup`** (new class, `:216-451`) replaces the three closures the old `_phase_group`
  returned. Public surface:
  - `__init__(phases, coefficients, signs)` — `phases` is `(phase_mu, phase_nu, phase_sigma)`
    (prompt 05 `BesselPhaseFunction` objects), `coefficients` is `(k, q, s)`, `signs` is
    `(1.0, e_nu, e_sigma)`. Computes and stores `self.K = fsum(e_i m_i)`,
    `self.C = fsum(e_i c_nu_i)` and `self.C_reduced = fsum(e_i c_nu_reduced_i)`.
  - `arguments(x) -> (k x, q x, s x)`
  - `residual(x)` — `R(x) = sum_i e_i r_i(m_i x)`, from `phase.residual`
  - `theta(log_x)` — `K x + C + R(x)`
  - `sin_cos(log_x)` — `(sin, cos)` by angle addition on `(K x)` and `(C_reduced + R)`
  - `theta_mod_2pi(log_x)` — `atan2` of that pair, in `(-pi, pi]`
  - `theta_deriv(log_x)` — `K x + sum_i e_i (dr_i/dlog y)|_{m_i x}`, from
    `phase.residual_log_deriv`
  - `theta_abserr(log_x)` — `sum_i phase_i.theta_abserr_at(m_i x)`, **linear**
  - `levin_theta() -> dict` — the four keys `theta`, `theta_mod_2pi`, `theta_deriv`,
    `theta_abserr`
* **`_phase_group(phase_mu, phase_nu, phase_sigma, k, q, s, e_nu, e_sigma)`** (`:454`) keeps its
  name and signature but now returns a `_PhaseGroup`; the single call site, `_Levin_3bessel`
  (`:494`), calls `.levin_theta()` on it. `_Levin_JJJ`, `_Levin_YJJ`, `quad_JJJ`, `quad_YJJ`,
  `_direct_JJJ`, `_direct_YJJ` are otherwise unchanged, so **all eight `adaptive_levin_sincos`
  calls of a `quad_JJJ` + `quad_YJJ` pair now carry a `theta_abserr`** (asserted, not assumed:
  `test_every_levin_call_receives_a_declared_phase_error` spies on the module's
  `adaptive_levin_sincos`).
* **`BesselIntegralResult`'s docstring** (`:25-60`): the paragraph claiming the phase/modulus fit
  error "is invisible from inside this module", quoting a uniform ~2e-8 floor and saying "nothing
  in `LiouvilleGreen/` supplies one yet", is replaced by four labelled paragraphs — what `abserr`
  now includes (the declared phase error, through each region's floor and hence through
  `phase_limited`), what it still does not (the *modulus* fit error, the truncation at `max_x`,
  and uncertainty in `k, q, s`), and the re-measured floor. The reasoning about linear rather than
  quadrature combination is left intact, as the prompt requires.

### `LiouvilleGreen/tests/test_three_bessel.py`

`TestBessel` (2 tests) is untouched. Added a module-level reference block and
**`TestPhaseGroups` (7 tests, 3.6 s)**:

| test | what it pins |
|---|---|
| `test_the_group_phase_survives_cancellation_of_its_leading_term` | `|delta Theta|` at every point `<= group.theta_abserr + 2 eps |K| x`, for K exactly 0, `K/max ~ 1e-6`, `1e-10` and generic; plus new-vs-old ratios |
| `test_the_group_derivative_is_scored_against_the_constituent_frequencies` | `|delta dTheta/dlog x| / (max(k,q,s) x) < 1e-11`; absolute new-vs-old ratio; and that at exact resonance the group derivative is O(1) against a constituent scale of 1e12, so a relative metric is meaningless there |
| `test_the_bounded_angle_is_the_angle_of_the_group_not_a_sum_of_angles` | the `(sin, cos)` pair error against mpmath, new vs the summed-bounded-angle route; `theta_mod_2pi` reproduces its own pair and lies in `(-pi, pi]` |
| `test_the_four_groups_reconstruct_the_triple_product` | `sin A sin B sin C = (-G1+G2+G3-G4)/4` and `(-cos A) sin B sin C = (G1-G2-G3+G4)/4` through the group machinery, with each constituent's `(sin, cos)` anchored to `J/A`, `-Y/A` from `bessel_reference` at `TIER_EXACT` |
| `test_every_levin_call_receives_a_declared_phase_error` | all 8 Levin calls get all four keys; the group declaration is the linear sum of its constituents' and is larger than the quadrature sum |
| `test_quadrature_refinement_is_not_a_certificate_of_phase_accuracy` | the value improves and then stops; the driver reports `phase_limited`; the residual disagreement with the oracle is far above the requested `rtol` |
| `test_the_three_bessel_integrals_still_match_their_closed_forms` | `quad_JJJ`/`quad_YJJ` against J000 and Y000, re-derived locally, `relerr < 1e-6` |

The reference is 60-digit `mpmath` at the **exact** products `k.x, q.x, s.x`
(`_group_reference`, `_mpmath_theta_and_deriv`), not at the doubles `fl(k x)`: a route that forms
`fl(m_i x)` for each leading term commits an error `~eps m_i x` with unit sensitivity, and a
reference taken at `fl(m_i x)` would hide exactly the error the restructure removes. The residual
branch needs no walk because `|r| <= pi` at every order used here. `_legacy_group_theta` keeps the
route this commit replaced, so the improvement is re-measured on every run rather than quoted from
this log.

## Deviations from the prompt

### 1. `test_3bessel_analytic.py` cannot pass, and could not before this commit — STRUCTURALLY REQUIRED

Prompt §4 requires `unittest discover -s LiouvilleGreen/tests -t .` to pass "including
`test_3bessel_analytic.py` **at its existing tolerances**". It does not, and the cause is neither
this commit nor a tolerance:

```
$ git rev-parse HEAD                     # bc31493, prompt 04, before bessel_phase was rewritten
$ python -m unittest ...test_abserr_bounds_truth
OK (expected failures=1)
   (JJJ 1,1,0): true abserr=3.0039e-10, reported abserr=2.9174e-10, ratio=1.0296, bounds=False
   ... 5 of 7 oracles underbound, up to 11.467x

$ git rev-parse HEAD                     # f9cc891, prompt 06, this prompt's parent
$ python -m unittest ...test_abserr_bounds_truth
FAILED (unexpected successes=1)
   (JJJ 1,1,0): true abserr=3.8121e-14, reported abserr=2.8889e-10, ratio=0.00013196, bounds=True
   ... 7 of 7 bound
```

`test_abserr_bounds_truth` is an `@unittest.expectedFailure` asserting that the reported `abserr`
does *not* bound the true error against the analytic oracles. Prompt 05 improved the oracle by
about eight orders (true `abserr` 3.0e-10 → 3.8e-14 on that row) while the reported `abserr` barely
moved, so the inequality flipped and the deliberate xfail became an **unexpected success** —
which `unittest` scores as a module failure. `RECONCILIATION.md` §3.4 and issue
`[05-3bessel-analytic-not-run-to-completion]` explain why nobody saw this: that module takes hours
and prompt 05 could not run it to completion.

This commit makes the reported `abserr` slightly *larger* (0–1.9 % at these tolerances), so it
moves in the direction that keeps the test passing; it cannot restore the failure and should not
try. Repairing the test means editing `test_3bessel_analytic.py`, which prompt 07 is explicitly
forbidden to touch ("prompt 08's") and which prompt 08 owns by the dependency table. Recorded as
issue `[07-abserr-bounds-truth-is-now-an-unexpected-success]` and handed to prompt 08. **This is
the item the orchestrator should surface to the user** (README §4.3: "any test the prompt says must
pass fails").

`test_JJJ` and `test_YJJ`, the two tolerance-bearing tests in that module, do pass — see
"Verification performed".

### 2. `_phase_group` became a class rather than three rewritten closures — IMPLEMENTATION CHOICE

The prompt describes the restructure but not its shape. Alternatives:

* rewrite the three closures in place, keeping `K`, `C`, `R` in the enclosing scope;
* a `NamedTuple` of callables, as `ComputeTargets/phase_groups.PhaseGroup` is;
* a small class with methods and a `levin_theta()`.

I took the third, and named the method `levin_theta()` to match `ComputeTargets/phase_groups`
(`:194`) — the sibling module that solves the same problem for the source integrand, which
`RECONCILIATION.md` §3.2 explicitly compares this one against. It buys three things the closures
could not: `K`, `C` and `C_reduced` are visible for a test to assert against (and `K` is asserted
to be exactly 0.0 in the resonant case), `sin_cos` is reachable so the four-group identity can be
checked without going through `atan2` and back, and the docstrings attach to the quantities they
describe rather than to one function. Cost: one more name in a module that had none. A `NamedTuple`
of callables would have been closer to the sibling's literal shape but cannot expose `K`.

`_phase_group()` itself is kept, with its old name and argument list, as a one-line factory, so
the diff at the call site is one method call.

### 3. `math.fsum` for `K`, `C`, `C_reduced` only; ordinary summation per point — IMPLEMENTATION CHOICE

The prompt asks for "appropriately accurate summation" for the three-term sums and for a statement
of what was used. I used `math.fsum` where the sum is *cancellative and computed once* — `K`, `C`,
`C_reduced` — and ordinary left-to-right summation for `theta`, `residual`, `theta_deriv` and
`theta_abserr`.

I first wrote a helper that used `fsum` for scalars and ordinary summation for arrays, and measured
that it was wrong to do so: `levin_quadrature._detect_vectorized()` accepts an array-sampling path
only if the array result is **bit-identical** to the scalar results, so a scalar path using `fsum`
and an array path that cannot silently loses the array sampling of `theta'` over the Chebyshev grid
— 13.5 ms against 71.8 ms for 200 samplings of a 13-point grid here, and `_sample_vectorized()`'s
own docstring puts the Python loop at 20–37 % of a subregion evaluation. With ordinary summation all
four callables are detected as vectorising (checked directly). The accuracy given up is nil in
context: the measured worst-case change from dropping `fsum` was 2.9e-11 at `K/max ~ 1e-6`, which
is one ulp of `K x` at that point — the floor of Deviation 4 — and the declared bound still covers
it with a factor 2.

### 4. The prompt's error model for the leading term is incomplete, and the shipped docstring says so — STRUCTURALLY REQUIRED (documentation only)

Prompt §2 item 1 gives the leading-term error as `t delta K` with `delta K ~ eps max(k,q,s)`, i.e.
it accounts for the rounding of `K` but not for the rounding of the **product** `K*x`. Measured, the
product rounding is what binds whenever `K` is not small: at `K = 0.1, x = 1e12` the group phase is
1e11 rad and `|delta Theta| = 1.526e-05` for *both* routes, exactly one ulp of `K x`. So the
docstring states the estimate as `~eps (|K| x + |C + R|)` rather than as `t delta K`, and says
plainly that a non-resonant group gains nothing from the restructure. Recorded as issue
`[07-generic-K-product-rounding]`, with the remedy (a two-product split of `K*x`, folding the low
part into the small angle) deliberately **not** implemented — the prompt prescribes handing the
unreduced product to libm, and this would be a second, unrequested change to the same expression.

### 5. The derivative acceptance metric and the derivative *decision* use different quantities — IMPLEMENTATION CHOICE

Prompt §3 item 2 requires `|delta theta'_group|` to be scored against `max(k,q,s)`, and prompt §2
item 3 requires the two derivative routes to be measured and the better one taken. Those turn out
to be different measurements, so the test reports both:

* scaled by `max(k,q,s) x` per point, the two routes are equal to within 1 % (1.45e-14 against
  1.44e-14) and both attain their maximum at the **bottom** of the near region, where `x` is small,
  there is no large leading term to cancel, and both sit on the interpolation floor. This is the
  prompt's metric, and by itself it does not choose a route.
* in absolute radians the restructured route wins by 3.0e7 (K = 0) to 1.2e3 (generic), because the
  route it replaced carries `eps max(m) x` and this one does not.

I took the restructured route on the absolute measurement, and on the sharpest case: at exact
resonance and `x = 1e12` the true group log-derivative is 6.666667e-12, which the restructured
route returns to the last bit while the old route is wrong by 100 % of it. Levin uses `theta'` for
basis conditioning *and* for subdivision (`phase_span`, `levin_quadrature.py:1090`), so an absolute
error of 9e-5 rad per unit `log x` in a quantity whose true value is 7e-12 is not a metric
technicality. The scaled metric is asserted as the acceptance threshold, at 1e-11, and both numbers
are printed.

Note that this route uses `residual_log_deriv` — a differentiated interpolant, the weaker of prompt
05's two derivative primitives (standing note 17) — where the old route used
`theta_deriv = exp(-2 ell)`. That is not a contradiction of standing note 17, which warns against
using the differentiated route as a *refinement criterion*; here it enters only as a correction to
`K x`, and where its relative error is worst `x` is smallest.

### 6. The four-group identity is checked against an `x`-dependent bound — IMPLEMENTATION CHOICE

The `(+, +)` group has `K = k + q + s`, so at the top of the grid its phase is 4e12 rad and one ulp
of it is 1e-3. A constant tolerance would therefore either fail at large `x` or be too loose to
detect a sign slip at small `x`. The test asserts
`error <= 1e-14 + 4 eps max_i |K_i| x` at every point and prints the worst ratio (0.0563), so the
strict part of the test is at the bottom of the grid where a sign error has nowhere to hide.

### 7. Requirement 3's choice: closed forms re-derived, not imported — IMPLEMENTATION CHOICE

Prompt §3 item 3 offers a choice. J000 (`(pi/4)/(kqs)`) and Y000 (the log formula) are re-derived
in `test_three_bessel.py` rather than imported from `test_3bessel_analytic.py`, so that this
module's tests do not couple to prompt 08's file (and do not import a module whose collection cost
is minutes). Two closed forms are enough for a smoke check; the seven-oracle comparison lives in
that other file.

## Verification performed

All runs from the repository root with `PYTHONPATH=. ./venv/bin/python`, on this commit unless
stated.

### The prompt's per-item measurements

`|delta Theta|` and `|delta dTheta/dlog x|`, before and after, at orders (1/2, 3/2, 5/2), signs
`(+, -, -)`, 41 log-spaced points over `34.5 <= x <= 1e12`, against 60-digit `mpmath` at the exact
products. Printed by `TestPhaseGroups`; "old" is `_legacy_group_theta` in the same file.

| case | `K` | `|delta Theta|` old | new | ratio | worst err / declared bound |
|---|---|---|---|---|---|
| K = 0 exactly | `0.0` | 2.8622e-05 (x=2.72e11) | **1.4211e-13** (x=34.5) | 2.01e8 | 0.4812 |
| K/max ~ 1e-6 | `9.5367431640625e-07` | 6.0307e-05 (x=5.22e11) | **2.9104e-11** (x=1.42e11) | 2.07e6 | 0.4845 |
| K/max ~ 1e-10 | `1.1641532182693481e-10` | 7.6272e-05 (x=5.22e11) | **1.4211e-13** (x=34.5) | 5.37e8 | 0.4812 |
| generic | `0.10000000000000009` | 1.5259e-05 (x=1.42e11) | 1.5259e-05 (x=1e12) | 1.00 | 0.4594 |

| case | `|delta dTheta/dlog x|` old | new | ratio | scaled by `max(k,q,s) x`: old / new |
|---|---|---|---|---|
| K = 0 exactly | 3.0518e-05 (x=5.22e11) | **1.0030e-12** (x=34.5) | 3.04e7 | 1.4406e-14 / 1.4517e-14 |
| K/max ~ 1e-6 | 7.3671e-05 | 1.0030e-12 | 7.35e7 | 1.4443e-14 / 1.4517e-14 |
| K/max ~ 1e-10 | 8.9636e-05 | 1.0030e-12 | 8.94e7 | 1.4327e-14 / 1.4517e-14 |
| generic | 1.7166e-05 | 1.4901e-08 (x=7.76e8) | 1.15e3 | 1.3627e-14 / 1.3823e-14 |

At exact resonance, `x = 1e12`: `dTheta/dlog x = 6.666667e-12` against a constituent scale of
2e12; relative error (the metric the test refuses to use) **0.000e+00** new, **1.000e+00** old.

The `(sin, cos)` pair error — what a Levin consumer actually sees — old / new: 3.0518e-05 /
**1.3947e-13** (K = 0), 6.9543e-05 / 5.6801e-11 (1e-6), 7.6389e-05 / **1.3936e-13** (1e-10),
1.4232e-05 / 1.2410e-05 (generic).

**Which derivative route won, and by how much:** the restructured `K x + sum e_i dr_i/dlog x`, by
3.0e7 in absolute radians at exact resonance and 1.15e3 at a generic `K`; by nothing at all in the
prompt's scaled metric (a 1 % difference, both routes at 1.45e-14). Deviation 5 has the reasoning.

**Summation scheme:** `math.fsum` for `K`, `C`, `C_reduced` at construction; ordinary summation per
point. Deviation 3 has the measurement that decided it.

**The new `abserr` floor for `quad_JJJ`/`quad_YJJ`, against the ~2e-8 the old docstring records.**
At `k,q,s = 1.3,1.7,2.1`, `max_x = 1e12`, `atol = 1e-14`, `rtol = 1e-10`, over all seven closed
forms of `test_3bessel_analytic.py` (measured with both routes in one process, by monkeypatching
`_phase_group`):

| oracle | true relerr | reported abserr, old route | new route | bounds truth? |
|---|---|---|---|---|
| J000 | 1.3970e-10 | 1.5331e-07 | 1.5331e-07 | yes |
| J110 | 5.8567e-12 | 2.8889e-10 | 2.9270e-10 | yes |
| J220 | 2.5205e-14 | 2.1353e-10 | 2.1734e-10 | yes |
| J222 | 3.5035e-14 | 2.1451e-10 | 2.2052e-10 | yes |
| J231 | 1.4946e-13 | 1.7902e-10 | 1.8239e-10 | yes |
| Y000 | 4.7708e-11 | 1.5335e-07 | 1.5335e-07 | yes |
| Y022 | 1.9705e-13 | 2.1411e-10 | 2.1715e-10 | yes |

So the uniform ~2e-8 relative floor is gone — it was the old construction's phase and modulus fit
error, removed by prompt 05, not by this commit. What this commit changes is that the declaration
reaches the driver: the reported `abserr` rises by 0–1.9 %, and the *floor* it reports is now the
round-off-plus-declared-phase floor rather than an unbounded subdivision. A visible consequence:
`quad_YJJ(Y022)` at `rtol = 1e-14` takes **0.44 s** with the declaration and **2.15 s** without, for
a value agreeing to 3.3e-14 relative — the driver stops subdividing against a phase it cannot
resolve and says `phase_limited` instead.

Tolerance refinement (`test_quadrature_refinement_is_not_a_certificate_of_phase_accuracy`, J000 at
`max_x = 1e12`, `atol = 1e-16`): `|value - analytic|` = 1.4203e-11 (`rtol=1e-8`), 2.3641e-11
(1e-10), 2.3904e-11 (1e-12), 2.3904e-11 (1e-13), 2.3904e-11 (1e-14); the spread over the last
three is 1.1102e-16 and `phase_limited` is `True` throughout. The residual 1.4125e-10 relative
disagreement with the closed form is four orders above the requested `rtol`, which is the point of
the test.

Four-group identity: worst `|direct - groups|` = 1.2463e-04 at `x = 1e12` against an
`eps`-scaled bound of 2.2e-03 (worst error/bound 0.0563); worst
`|constituent (sin, cos) - reference|` = 1.4252e-13 at `TIER_EXACT`.

Group declaration: `theta_abserr` is the linear sum of the three constituents' to within 1e-18,
and exceeds their quadrature sum. All 8 Levin calls (`quad_JJJ` + `quad_YJJ`) received exactly
`['theta', 'theta_abserr', 'theta_deriv', 'theta_mod_2pi']`.

Closed-form smoke check at `k,q,s = 1.3,1.7,2.1`, `max_x = 1e12`: J000 relerr 1.3970e-10, Y000
relerr 4.7708e-11.

### Test runs

| module | result | time |
|---|---|---|
| `LiouvilleGreen.tests.test_three_bessel` | **OK, 9 tests** (2 pre-existing + 7 new) | 10.0 s (was 4.1 s for 2) |
| `LiouvilleGreen.tests.test_three_bessel.TestPhaseGroups` | OK, 7 tests | 3.6 s |
| `LiouvilleGreen.tests.test_bessel_compatibility` | OK, 20 | 2.1 s |
| `LiouvilleGreen.tests.test_bessel_near_region` | OK, 27 | 0.3 s |
| `LiouvilleGreen.tests.test_bessel_phase` | OK, 4 | 0.2 s |
| `LiouvilleGreen.tests.test_bessel_reference` | OK, 12 | 0.3 s |
| `LiouvilleGreen.tests.test_bessel_tail` | OK, 19 | 0.2 s |
| `LiouvilleGreen.tests.test_bessel_two_region` | OK, 24 | 4.5 s |
| `LiouvilleGreen.tests.test_range_reduce` | OK, 4 | 0.0 s |
| `LiouvilleGreen.tests.test_scipy_bessel_domain` | OK, 7 | 0.0 s |
| `ComputeTargets.tests.test_quadsource_integral` | OK, 33 | 124 s |
| `LiouvilleGreen.tests.test_3bessel_analytic` | **not run as a module** (hours; `RECONCILIATION.md` §3.4) | — |
| `...test_3bessel_analytic` `test_JJJ` + `test_YJJ` | **OK, 2 tests** (relerr 2.3e-13 to 5.5e-10) | 401 s |
| `...test_3bessel_analytic.test_abserr_bounds_truth` | **unexpected success** — Deviation 1 | 6.9 s |

`test_phase_derivative` (`test_bessel_phase.py:139`, the standing 1e-6 regression gate of
`RECONCILIATION.md` C4) passes. `ComputeTargets.tests.test_quadsource_integral` was run because
`QuadSourceIntegral._three_bessel_integrals` calls `quad_JJJ`/`quad_YJJ` and is therefore a live
consumer of this change, even though no `ComputeTargets/` file was touched.

`test_JJJ`/`test_YJJ` on the parent commit `f9cc891`, for comparison: **OK, 559.6 s**, with
relative errors 9.2e-13 to 5.4e-12 on its three triangle-satisfying draws. Those two tests draw
`k, q, s` from `uniform(0.1, 5.0)` on each run, so the before/after comparison is of pass/fail and
order of magnitude, not of a fixed case.

`black --check LiouvilleGreen/` is clean (22 files).

### What was *not* verified

* `test_3bessel_analytic.py` as a whole module. Three of its six tests were run individually; the
  three plotting/singularity sweeps (`test_YJJ_log_singularity`, `test_YJJ_log_scaling` and the
  `MAX_X = 1e12` plot loops inside `test_JJJ`/`test_YJJ`) are the multi-hour part.
  `[05-3bessel-analytic-not-run-to-completion]` stays open, narrowed.
* Anything at `nu > 5/2`. This module is only ever called at orders `mu + 0.5` with `mu` a small
  integer, and the local `mpmath` reference relies on `|r| <= pi`, which fails above roughly
  `nu = 6`. The test module says so.
* Ray serialization of a `_PhaseGroup`. It is constructed inside `_Levin_3bessel` and never
  crosses a process boundary; `BesselPhaseProxy` transfers the `bessel_phase` dict, which prompt
  06 verified.

## Observations not acted on

1. **`levin_quadrature._sample_vectorized()`'s docstring is stale** in the same way
   `[06-levin-theta-docstring-stale]` records for `:2750`: it says the `three_bessel_integrals.py`
   / `QuadSourceIntegral.py` callables "do not [vectorize], as of this commit: their phase and
   modulus splines branch on a scalar argument". Prompt 05's `BesselPhaseFunction` accessors are
   array-safe, so both this assembly and the one it replaced are detected as vectorising (checked
   directly with `_detect_vectorized`). `AdaptiveLevin/` is forbidden here; the existing issue was
   extended rather than a second one opened.
2. **`[09-abserr-does-not-bound-phase-spline-floor]` (the `levin-refactor` campaign's issue) is now
   false as stated**: "`quad_JJJ`/`quad_YJJ`'s `abserr` misses the true error against the analytic
   oracle by up to 11.5× on 5 of 7 three-Bessel closed forms" — measured here as 0 of 7, with
   margins from 12× to 6.5e3×. That is prompt 05's doing, plus this commit's declaration.
   Closing it means editing another campaign's board, which is out of scope; prompt 09 makes the
   cross-campaign hand-offs and should carry this one too.
3. **`DEFAULT_3BESSEL_CHEBYSHEV_ORDER`'s comment is now questionable.** It justifies 12 by
   "the relative error is *identical* at orders 12 through 64 — it is set by the accuracy of the
   phase and modulus splines, not by the spectral order", measured at `max_x = 1e5` against the old
   construction. The phase splines are eight orders better now, so that measurement no longer
   supports the conclusion and the order may well be re-optimisable. Not touched: it is a
   performance parameter, no prompt owns it, and changing it would move every three-Bessel number
   in the campaign.
4. **The `max_x` truncation, not the phase, is now the largest error in the lowest-order
   oracles.** J000 and Y000 disagree with their closed forms by 1.4e-10 and 4.8e-11 relative at
   `max_x = 1e12` while J220/J222/J231/Y022 sit at 2.5e-14 to 2.0e-13; the low-order integrands
   decay slowest, so the omitted tail dominates. This bears directly on prompt 08's re-tightening
   of `test_3bessel_analytic.py`: a tolerance tightened to the phase floor there will be
   measuring `MAX_X`, not the oracle. Noted in `BesselIntegralResult`'s docstring.

## State handed to the next prompt

### For prompt 08 (fixture revalidation), which owns `test_3bessel_analytic.py`

1. **`test_abserr_bounds_truth` is an unexpected success and the module therefore fails.** Not
   caused by this commit; measured on `bc31493` (expected failure, 5 of 7 underbound by up to
   11.5×) and on `f9cc891` (unexpected success, 7 of 7 bound). Issue
   `[07-abserr-bounds-truth-is-now-an-unexpected-success]`. The test's docstring says closing it
   needs "`phase_spline` to report its own fit accuracy and `theta_abserr` wired up" — the
   `theta_abserr` half is now done for this module, so the honest repair is to drop the
   `@unittest.expectedFailure` and keep the assertion, not to weaken it.
2. **Its tolerances are not currently limited by the Bessel phase.** At `k,q,s = 1.3,1.7,2.1`,
   `max_x = 1e12` the seven oracles measure 2.5e-14 to 1.4e-10 relative against `REL_TOLERANCE =
   1e-5`, i.e. five to nine orders of headroom. The two largest residuals are J000 (1.4e-10) and
   Y000 (4.8e-11), and observation 4 argues they are the `MAX_X = 1e12` truncation rather than the
   phase. Tightening below ~1e-9 relative on those two will be measuring `MAX_X`; the other five
   have room to ~1e-12. The singularity bands (1e-2/1e-3) were not measured here.
3. `test_3bessel_analytic.py` still emits three `DeprecationWarning`s per case from its
   `atol`/`rtol` call sites (`:47-56`, `:504-512`), as prompt 06 recorded.

### For prompt 09 (benchmark and docs)

* Two cross-campaign hand-offs now, not one: `[00-qsi-three-bessel-levin-excluded]` as planned,
  and observation 2 above — `levin-refactor`'s `[09-abserr-does-not-bound-phase-spline-floor]` is
  measured false on this tree.
* `[07-generic-K-product-rounding]` is the one new numerical limit this commit documents rather
  than removes: a non-resonant group phase is accurate to one ulp of `K x` (1.5e-5 rad at
  `K = 0.1`, `x = 1e12`) and both routes measure the same there. If `docs/` records the campaign's
  achieved accuracy, this is the caveat that belongs next to the three-Bessel numbers.

### Interfaces, for anyone touching this module

```python
group = _phase_group(phase_mu, phase_nu, phase_sigma, k, q, s, e_nu, e_sigma)  # -> _PhaseGroup
group.K, group.C, group.C_reduced          # floats, formed once by math.fsum
group.arguments(x)                         # (k x, q x, s x)
group.residual(x)                          # R(x) = sum_i e_i r_i(m_i x)
group.theta(log_x)                         # K x + C + R(x)
group.sin_cos(log_x)                       # (sin, cos) by angle addition on (K x), (C_red + R)
group.theta_mod_2pi(log_x)                 # atan2 of that pair, in (-pi, pi]
group.theta_deriv(log_x)                   # d/dlog x, = K x + sum_i e_i dr_i/dlog x
group.theta_abserr(log_x)                  # sum of the constituents' theta_abserr_at
group.levin_theta()                        # the four-key dict for adaptive_levin_sincos
```

Every accessor takes `log_x` (the Levin integration variable) except `arguments`, `residual` and
`sin_cos`'s internals, which take `x`. All four callables in `levin_theta()` are array-safe **and
bit-identical between the array and scalar paths**; keep it that way, or
`levin_quadrature._detect_vectorized()` will silently fall back to a Python loop (Deviation 3).
