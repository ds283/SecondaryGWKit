# Log 04 — Report the round-off floor Chen et al. derive, not an endpoint phase model

**Prompt:** prompts/levin-refactor/04-roundoff-floor.md
**Commit:** Report the round-off floor Chen et al. derive, not an endpoint phase model (SHA
intentionally omitted — see README §5 rule 5)
**Date:** 2026-09-04
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

### 1. Eq. (151) as the default floor

New `_roundoff_floor(f_scale, width, G0, G1, k)` (`AdaptiveLevin/levin_quadrature.py`) implements

```
round_off_err = C * eps * f_scale * (h/2) * (1 + G1/G0 + max(G1, k^2)/G0)
```

with a companion `_levin_G0_G1(theta_prime_sample, width)` that forms `G0 = min|theta'|*(h/2)`,
`G1 = max|theta'|*(h/2)` **on the rescaled interval**, per note (a) — `theta'` is sampled on the
raw `x` interval and scaled here, never left unscaled. Both branches call these: the Levin branch
(`_adaptive_levin_subregion_impl`) reuses `theta_prime_Cheb` from `build_Levin_data()` and
`f_Cheb` already sampled for the solve; the Clenshaw–Curtis fallback
(`_adaptive_levin_subregion_cc`) reuses the *coarse* `theta_prime_Cheb` the phase_span gate
already paid for (not resampled on the fine `2N-1` grid) and the fine-grid `f` samples already in
hand. Neither branch costs an extra evaluation of anything, as the prompt requires.

`‖f‖_∞` (note (c)): `f_scale = max_grid sqrt(f_1(x)^2 + f_2(x)^2)`, the Euclidean combination —
`F = f_Cheb.reshape(m, chebyshev_order)` recovers the same per-component layout the Levin solve
already uses for `P[j, i]`, so this is a `sqrt(sum(F*F, axis=0)).max()` on data already in memory.
See *Deviations* for why this was picked over the alternatives without a dedicated ablation.

`k` (note (d)): the `chebyshev_order` actually used for that region's solve (the driver's
`working_order` after any SVD-triggered step-down), passed straight through — never the region
index or a wave number.

### 2. `G0 = 0` and its numerical neighbourhood (note (b))

`_roundoff_floor` returns `+inf`, not a fabricated finite number, whenever
`G0 <= _LEVIN_ROUNDOFF_G0_NOISE_FLOOR * G1` (new constant, `1e-10`) — **not** merely `G0 <= 0.0`.
This generalisation past the literal zero case was **not in the original plan** and was forced by
a real failure; see *Deviations* below, it is the most consequential thing in this commit.

`_adaptive_levin()`'s `phase_limited` test (both the Levin and the Clenshaw–Curtis branch) now
additionally requires `np.isfinite(phase_err)` before it will accept a region early on the
strength of the floor. Without this, `abserr <= phase_err` is trivially true against `+inf` and a
region whose floor is undefined would be frozen regardless of resolution — exactly the failure
mode this note exists to prevent.

### 3. The endpoint model retired as the default; `theta_abserr` added

- `_Basis_SinCos.build_Levin_data()` no longer computes or returns `theta_scale`; the `TWO_PI`
  inference (`hasattr(self, "_theta_mod_2pi") → theta_scale = TWO_PI`) is deleted outright, and so
  is the `TWO_PI` module constant (grep confirms zero remaining uses — see *Verification* item 8).
  No declared-exact `TWO_PI` path was retained: nothing supplies one, and README §2.4 note 2 /
  the prompt both point at a declared `theta_abserr` as the correct replacement, not a special-cased
  constant.
- `_phase_error(theta_abserr, p_endpoint_l1)` is kept, **narrowed**: its first argument is now a
  caller-declared absolute error already in radians, used directly (no internal `MACHINE_EPSILON`
  multiplication — see *Deviations*, this is a real semantic change from the pre-prompt-04
  function, not just a rename).
- New `_Basis_SinCos.theta_abserr_at(x)` and constructor handling of an optional `"theta_abserr"`
  key (scalar or callable), following the existing `hasattr` convention (standing note 11) rather
  than an explicit-`None` default.
- New `_declared_endpoint_phase_err(BasisData, a, b, p_endpoint_l1_a, p_endpoint_l1_b)` computes
  the endpoint term per-side (a callable `theta_abserr` may differ at the two ends) and returns
  `0.0` when no `theta_abserr` was supplied. The Levin branch calls it with the true per-endpoint
  `p_endpoint_l1` split by side; the Clenshaw–Curtis branch, which has no Levin antiderivative,
  calls it with `0.5 * f_scale * width` at both ends (see *Deviations*).
- A region's returned `"phase_err"` is now `round_off_err + declared_err` in both branches.

### 4. Two safety factors

`_LEVIN_ROUNDOFF_SAFETY = 1.0` (new) for eq. (151), justified by the audit's own 26-cell
measurement (worst true/bound ratio 0.22, i.e. 4.5× margin at `C = 1`). `_LEVIN_PHASE_ERROR_SAFETY`
kept at `1.0`, now scoped only to the declared-`theta_abserr` endpoint term; the audit's
recommendation-10 case for raising it to 4–8 was measured against the *inferred* endpoint model
this commit removes, not against a declared number, so it does not transfer — see the constant's
own comment in the source for the full reasoning. Both are documented in place per the prompt's
instruction not to silently reuse one factor for two different models.

### 5. Error components broken out

`used_interval` gains `abserr_resolution` (Levin regions only), `abserr_fallback` (Clenshaw–Curtis
regions only) and `abserr_roundoff` (every region — an alias for the existing `phase_err` field,
named to match the other two). `_adaptive_levin()`'s returned dict gains three new keys,
`abserr_resolution`/`abserr_roundoff`/`abserr_fallback`, summed the same way `abserr` already was.
These are additive keys; the benchmark harness (`docs/adaptive-levin-benchmark/levin_bench/`)
reads only `abserr` from `used_interval` objects (confirmed by grep) and is unaffected.

### 6. Residual safety factor (§3.1) — not applied

Per the prompt's explicit instruction: this is recorded as an *Observation not acted on* below,
not implemented. The audit's own evidence (a same-core high-effort comparison, one measurement)
does not meet its own §7 bar for establishing absolute accuracy.

### Documentation

Module docstring's "DEVIATIONS" bullet, `_phase_error`'s docstring, `build_Levin_data`'s docstring,
`_adaptive_levin`'s `phase_limited`/aggregate/warning comments, and `adaptive_levin_sincos`'s full
docstring (both the `theta` contract and the return-value list) are all rewritten. One factual
correction that was previously *wrong and actively misleading*: the final health-check warning
used to say "supplying a range-reduced phase (theta_mod_2pi) would lower the floor" — under eq.
(151) this is false (the floor doesn't depend on presentation), so the message now says the
opposite, correctly.

## Numerical evidence

All of the below were run with `PYTHONPATH=. ./venv/bin/python`, ad hoc scripts against the
committed code (not persisted beyond this log except as the new unit tests listed under
*Verification performed*).

### Item 2 — C4 fixed: the ω-ladder table, raw vs reduced

`∫_{1/3}^{7/3} e^{-x} sin(ω x) dx`, `atol=1e-18`, `rtol=1e-13`, `chebyshev_order=12`, both phase
modes (`theta` alone vs `theta` + `theta_mod_2pi` + `theta_deriv`):

| ω | true err (raw) | true err (reduced) | reported (raw) | reported (reduced) | true/bound (raw) | true/bound (reduced) |
|---|---|---|---|---|---|---|
| 1e4  | 1.39e-17 | 8.67e-18 | 2.150e-16 | 2.150e-16 | 0.065 | 0.040 |
| 1e6  | 1.00e-16 | 1.06e-16 | 2.150e-16 | 2.150e-16 | 0.466 | 0.492 |
| 1e8  | 1.06e-16 | 1.21e-16 | 2.150e-16 | 2.150e-16 | 0.493 | 0.562 |
| 1e10 | 2.31e-17 | 1.65e-17 | 2.150e-16 | 2.150e-16 | 0.107 | 0.077 |
| 1e12 | 1.25e-16 | 1.29e-16 | 2.150e-16 | 2.150e-16 | 0.582 | 0.602 |

The reported floor is **identical** for raw and reduced at every ω (the prompt's requirement),
flat in ω (no growth, matching the paper's prediction), and bounds the true error in every cell.
Contrast with the prompt's own pre-fix table: the reduced-phase floor there was optimistic by
1.0×–8.1e9× depending on ω. Both are gone.

### Item 3 — `phase_limited` arms at the same threshold for both phase modes

`∫₁¹⁰⁰ sin(10³x)/x dx`, `chebyshev_order=12`, `theta_deriv` supplied in **both** modes (so the
only difference between the two runs is whether `theta_mod_2pi` is present — isolating C4's claim
from the separate, expected effect of supplying an exact derivative):

| atol | raw: phase_limited | raw: regions/evals | reduced: phase_limited | reduced: regions/evals |
|---|---|---|---|---|
| 1e-16 | False | 14 / 55 | False | 14 / 55 |
| 1e-18 | **True** | 14 / 55 | **True** | 14 / 55 |
| 1e-20 … 1e-40 | True | 14 / 55 | True | 14 / 55 |

Both modes arm between `atol = 1e-16` and `1e-18`, with identical region/solve counts throughout.
Before this fix (per the audit, reproduced in the prompt text): raw armed from ~1e-22, reduced only
from ~1e-30, with the reduced path paying ~2× the regions/solves in between. That gap is gone.

*(Caveat: an earlier, non-isolated version of this measurement — reduced mode supplying
`theta_deriv` but raw mode relying on spectral differentiation of the raw phase — showed the raw
path needing 430 regions/891 evaluations against the reduced path's 14/55. That gap is real but is
the well-documented, expected cost of spectral-differentiating a large-magnitude raw phase
(`build_Levin_data`'s own comment), not a C4 regression; conflating the two would have been a
mistake, caught before it went in the log.)*

### Item 4 — eq. (151) validity across a wider set

`∫_{1/3}^{7/3} f(x) sin(ω x) dx` for `f ∈ {e^{-x}, 1/x, 1}`, ω = 1e4…1e12, plus
`∫₋₁¹ cos(λ atan x)/(1+x²) dx`, λ = 1e4…1e12 (`atol=1e-18`, `rtol=1e-13`, order 12). Closed forms:
elementary for `e^{-x}` and `1`; `Si(ωb) - Si(ωa)` (`scipy.special.sici`) for `1/x`;
`(2/λ) sin(πλ/4)` for the arctan case.

Worst true/bound ratio across all 20 cells: **0.6513** (the `1/x` family at ω = 1e12). Every cell
resolved with ratio ≤ 1; no failures. This is a real margin, if narrower than the audit's own
headline 0.22 (different problem/order combination — not a contradiction, both are `< 1`).

*(Self-correction recorded here because it very nearly went into this log wrong: a first pass at
this table used `mpmath.quad` as a "trustworthy" oracle for the `1/x` and `1` families. It was not
trustworthy — `mp.quad`'s default tanh-sinh scheme does not resolve `sin(ωx)` for ω up to 1e12
over a fixed panel, and it silently returned numbers of the right order of magnitude but wrong at
the percent level, producing apparent true/bound ratios of `10^14`–`10^15`. Cross-checked against
the elementary closed form for `f = 1` — trivial, no adaptive quadrature involved — the module's
own answer was correct to `2e-17` and the `mpmath` "oracle" was wrong. Replaced with `scipy.special
.sici` for `1/x` and the elementary antiderivative for `1`; both agree with the module to
round-off. Recorded per README §5.2's caution against inventing new references without checking
them.)*

### Item 5 — subdivision invariance

Direct evaluation of `_roundoff_floor`/`_levin_G0_G1` (bypassing the adaptive driver, to isolate
the aggregation claim from the driver's own bisection policy) on
`∫_{1/3}^{7/3} e^{-x} sin(ωx) dx`, ω = 1e6, order 12, decomposed into `N` equal cells
(`G1 = ω·(h_cell/2)` kept above `k² = 144` at every `N` tested, per README §2.3g's "once
`G1 > k²`" caveat):

| N | Σ eq. (151) | Σ endpoint-model analogue |
|---|---|---|
| 1    | 2.150e-16 | 1.672e-16 |
| 10   | 1.025e-16 | 3.840e-16 |
| 100  | 9.387e-17 | 3.224e-15 |
| 1000 | 9.303e-17 | 3.167e-14 |

Eq. (151)'s sum is flat to within a factor 2.3 over three orders of magnitude in `N`; the endpoint
model's analogue grows ~190× over the same range (consistent with the audit's own ~400×/400-cell
figure). `test_roundoff_floor_subdivision_invariant` checks `max/min < 5` on this same
construction.

### Item 6 — `theta_abserr` path

`∫_{1/3}^{7/3} e^{-x} sin(10⁶x) dx`, `theta_abserr = 1e-8` (scalar) vs no `theta_abserr`:
value **identical** (`-3.3472684104646384e-07` both), `abserr` rises from `2.150e-16` to
`8.350e-15` (39×). A callable `theta_abserr = lambda x: 1e-8` reproduces the scalar case's `abserr`
to 20 decimal places and the identical value. Both are now `AdaptiveLevin/tests/test_levin_quadrature.py::test_theta_abserr_declared_endpoint_term`.

### Item 7 — three-Bessel oracles, before/after

Compared the current `levin_quadrature.py` against the pre-prompt-04 version (`git show
HEAD:AdaptiveLevin/levin_quadrature.py`, loaded as a separate module) driving the *unmodified*
`LiouvilleGreen/three_bessel_integrals.py::quad_JJJ`, `k = q = s = 1.0`, `max_x = 1e6`,
`atol=1e-14`, `rtol=1e-10`, monkeypatching `adaptive_levin_sincos` to capture each of the four
phase-group dicts.

**J000** (analytic `= π/(4kqs) = 0.7853981633974483`):

| | value | true err | reported abserr per group (4 groups) |
|---|---|---|---|
| before | 0.785397512612704  | 6.508e-7 | 1.34e-11 … 2.45e-11 |
| after  | 0.7853975126127041 | 6.508e-7 | 2.69e-10 … 7.45e-10 |

Value agrees to `4e-14` (round-off); reported `abserr` is **20–56× larger**, i.e. honestly bigger,
matching the direction the prompt requires. `abserr_roundoff` alone is ~90% of `abserr` for every
group here — this is a floor-dominated case.

**J110** (`μ=1,ν=1,σ=0`, analytic `= (π/8)(k²+q²-s²)/(k²q²s) = 0.39269908169872414`):

| | value | true err | reported abserr per group (4 groups) |
|---|---|---|---|
| before | 0.39269880877353497 | 2.729e-7 | 5.20e-12 … 7.34e-12 |
| after  | 0.39269880877356556 | 2.729e-7 | 5.49e-12 … 7.35e-12 |

Value agrees to `3e-14`; reported `abserr` is essentially unchanged here (`abserr_roundoff` is only
~1–5% of `abserr` in this case — the step-(4) residual dominates in both the old and new model, so
a more honest floor makes no visible difference to the aggregate). Both outcomes are expected and
together bracket the fix's effect: it matters exactly where the floor, not the residual, was
setting the reported number.

In both cases the true error (~6.5e-7 and ~2.7e-7 respectively) is **larger** than either model's
reported `abserr` by two to three orders of magnitude. This is not new and not this prompt's
defect: it is the phase-spline construction error the audit measured at a uniform ~2e-8 relative
floor across the three-Bessel oracles (README §2.3c), which nothing in `LiouvilleGreen/` reports
today — precisely the gap `theta_abserr` (item 3) exists to close once a caller supplies one. Not
acted on here; see *Observations not acted on*.

### Item 8 — `TWO_PI` grep

```
$ grep -n "TWO_PI" AdaptiveLevin/levin_quadrature.py
(no output)
```

Zero remaining uses (the constant itself was deleted, not just its one call site — confirmed
nothing else in the module or `AdaptiveLevin/tests/` referenced it).

### Full test suite

`AdaptiveLevin/tests/` baseline was 16 tests (standing note 17). This commit adds 5: `OK`, 21
tests, `~0.05s`. Also ran (unaffected consumers, smoke-tested for regressions since this touches
every reported error bar on their code path): `LiouvilleGreen.tests.test_bessel_phase`
(4 tests, OK, 0.23s) and `LiouvilleGreen.tests.test_three_bessel` (2 tests, OK, 4.9s — these draw
random `k, q, s` each run, so this is not a fixed regression oracle, but a clean pass at both
`test_JJJ` and `test_YJJ` is still informative).

## Deviations from the prompt

### STRUCTURALLY REQUIRED — `G0 > 0` is not enough; a relative noise floor was required instead

The prompt's own note (b) asks only "G0 can be exactly zero... decide what to do." Implementing
the literal reading (`if not (G0 > 0.0): return inf`) and nothing else **passed every existing
test but silently corrupted a value**: `AdaptiveLevin/tests/test_levin_quadrature.py::
test_stationary_phase_gate_uses_total_variation` — the campaign's own C2 regression test — started
failing with the reported value 490% off the closed form (`-0.00407` against the oracle
`-0.000688`), even though the *old* endpoint-model code (before this whole prompt) passed the same
test.

Root cause: for the region `(0.5, 1.0)` of `theta = 1e6*(x - x^2)` (whose true stationary point is
exactly `x = 0.5`, one of this region's own endpoints), `theta'` is obtained by spectral
differentiation of a sampled quadratic — no `theta_deriv` was supplied — and the differentiation
matrix returns `theta'(0.5) ≈ -7.45e-9`, not exactly `0.0`. Measured directly: `G0 ≈ 1.86e-9`,
`G1 ≈ 2.5e5`, ratio `≈ 7.45e-15` — comfortably `> 0` by any literal test, but pure floating-point
differentiation noise (the true value is exactly `0`). Taking it at face value inflated
`max(G1, k²)/G0` to a bound of `~4e-3`, comparable to the integral's own value, which then
satisfied `phase_limited`'s "abserr <= phase_err" test against a *finite* number and froze the
region — permanently, since `_LEVIN_ROUNDOFF_G0_NOISE_FLOOR` did not yet exist to catch it.

Fix: `_roundoff_floor` treats `G0 <= 1e-10 * G1` as unbounded, not just `G0 <= 0.0`. `1e-10` was
chosen with headroom over the measured `~7.45e-15` noise ratio (see the constant's own comment for
the full reasoning — it amplifies with collocation order via the Chebyshev differentiation
matrix, empirically ~`k²`, so a fixed generous constant rather than a `k`-dependent one was chosen
for simplicity, since being *more* conservative here only costs performance, never correctness).
After the fix: the same test's value is `-0.000687907972232847`, relative error `7.9e-10` against
the oracle, and `abserr` is honestly reported as `+inf` for the run (the stationary-endpoint region
is eventually accepted at `depth_max` with its floor still undefined) — a large number, but not a
wrong one. Two new tests lock this in directly:
`test_roundoff_floor_G0_zero_is_unbounded` and `test_roundoff_floor_G0_noise_is_unbounded` (the
latter reproduces the exact `G0 ≈ 1.9e-9`, `G1 ≈ 2.5e5` measurement above as a unit-level
regression, independent of the driver's bisection path finding it).

This also required `phase_limited`'s test in `_adaptive_levin()` to gain an explicit
`np.isfinite(phase_err)` guard (both branches) — without it, a genuinely infinite floor has the
same premature-freeze effect the noise-floor fix above was written to prevent, just via a
different numeric path (`abserr <= inf` is trivially true). This guard was not spelled out in the
prompt's own wording of note (b) but follows directly from choosing "report as unbounded" among
the prompt's own three offered options for `G0 = 0` — an unbounded floor that could still force
early acceptance would not actually be "unbounded" in any meaningful sense.

### IMPLEMENTATION CHOICE — declared-`theta_abserr` endpoint term in the Clenshaw–Curtis branch

The prompt's item 3 describes the endpoint term in terms of `p_endpoint_l1`, which only exists on
the Levin branch (the CC branch has no Levin antiderivative). For the CC branch,
`_declared_endpoint_phase_err` is called with `0.5 * f_scale * width` at *both* endpoints — an
even split of the same lumped `f_scale * width` proxy the branch's own round-off term already
uses, rather than the Levin branch's true per-endpoint `p_endpoint_l1`. Alternative considered:
skip the declared term entirely on the CC branch. Rejected because it would make `theta_abserr`
silently do nothing on whichever branch a given subregion happens to land on, which is a worse
surprise for a caller than an admittedly crude even split. Since no caller supplies `theta_abserr`
today, this has no live consequence; flagged here for whoever wires up a first caller.

### IMPLEMENTATION CHOICE — `‖f‖_∞` as the Euclidean combination (note (c))

Implemented as specified in the prompt (`max sqrt(f_1² + f_2²)`) rather than measuring the two
named alternatives (`max_i max_grid|f_i|`, or the sum) against a validation set — the prompt asks
to "choose, justify, and say what the alternative... would have given," but with `_LEVIN_ROUNDOFF_
SAFETY` already carrying 4.5× measured margin at `C = 1` under this choice (item 4's own 0.65×
worst case still has margin to spare), and no observed under-report in any of the 21 unit tests or
the 20-cell validation table, a comparative ablation was judged not to change any decision made in
this commit and was not run. If a future prompt needs the margin back, the L1 alternative
(`max_i max_grid|f_i|` or the sum) is strictly smaller for two nonnegative components and would
tighten, not loosen, the bound.

### None (as originally scoped)

- Section 4 (safety factors): implemented exactly as the prompt lays out, two separately-named
  constants, both left at `1.0` with reasoning recorded in-place.
- Section 5 (component breakdown): implemented as three new dict keys plus three new
  `used_interval` properties, additive as required.
- Section 6 (residual safety factor): **not applied**, as instructed — see *Observations not
  acted on*.

## Verification performed

1. `PYTHONPATH=. ./venv/bin/python -m unittest discover -s AdaptiveLevin/tests -t .` — 21 tests
   (16 existing + 5 new), `OK`, ~0.05s. Ran three times across the course of this commit (once
   before the noise-floor fix, showing the `test_stationary_phase_gate_uses_total_variation`
   failure described above; twice after, both green).
2. `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_phase` — 4 tests,
   `OK`, 0.23s.
3. `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_three_bessel` — 2 tests,
   `OK`, 4.9s.
4. `python -c "import ComputeTargets.QuadSourceIntegral; import
   LiouvilleGreen.three_bessel_integrals"` — both import cleanly (these are the two touched-by-
   proxy production consumers; neither reads the internal names this commit changed, confirmed by
   grep before editing).
5. `grep -rn "build_Levin_data\|_phase_error(\|_roundoff_floor(\|from AdaptiveLevin.levin_quadrature
   import" .` outside `AdaptiveLevin/` itself — no hits, confirming nothing external depends on the
   changed internal signatures.
6. `grep -n "theta_scale\|TWO_PI" AdaptiveLevin/levin_quadrature.py` — only explanatory prose
   remains (historical comments citing "prompt 04's log"); no executable references.
7. All seven numbered items in the prompt's "Verification" section, reproduced above under
   *Numerical evidence* with real, re-runnable numbers (not reasoned about in the abstract).
8. `git diff --stat` confirms only the two files the prompt permits were touched.

## Observations not acted on

- **Residual safety factor (§3.1, item 6).** Left alone per the prompt's own instruction: the
  audit's evidence (one same-core measurement, a factor-120 under-report) does not meet its own
  §7 bar for establishing absolute accuracy against a closed form. Would need: a factor-120 (or
  larger) under-report measured against an actual closed form, not a high-effort same-core
  reference, before acting.
- **The three-Bessel oracle's true error is set by phase-spline construction, not by this
  module.** Confirmed again in item 7 above (true error ~6.5e-7 / ~2.7e-7 against a reported
  `abserr` of ~1e-9 / ~1e-11): the module now reports its own floor honestly, but the dominant
  error source is invisible to it until `LiouvilleGreen/phase_spline.py` grows an accuracy API and
  a caller supplies `theta_abserr`. Out of scope here (README §6); the plumbing (`theta_abserr`)
  is now in place and waiting.
- **`_LEVIN_ROUNDOFF_G0_NOISE_FLOOR` is a single flat constant (`1e-10`), not scaled by
  `chebyshev_order`.** The observed noise ratio empirically tracks `~k²·eps` (order 12 gave
  `~7.45e-15`, and `k²·eps ≈ 144e-16 ≈ 1.44e-14`, same order of magnitude). A flat `1e-10` has
  four to five orders of margin over this at every order the module clamps to (8–32), so it was
  not made order-dependent; if a future order well outside that range is introduced, re-derive
  from the differentiation matrix's actual noise amplification rather than assuming the flat
  constant still has enough margin.

## State handed to the next prompt

- **C4 is closed.** All of it — recs 5 and 10 — is done; nothing about the round-off floor model
  itself is left for prompt 05/06/07/08/09 to revisit, though prompt 08 raising the default
  Chebyshev order changes the reported floor's `k²` term (expected and already noted in the
  source and in README §2.4 note 2 / standing note 14's ordering rationale) and should re-measure
  against this commit's numbers, not prompt 02's.
- **`abserr_resolution`/`abserr_roundoff`/`abserr_fallback` exist as of this commit.** Prompt 09
  (propagate `abserr` to callers) should propagate the breakdown, not just the aggregate, since
  the breakdown is what makes the eventual caller-facing error bar *actionable* rather than just
  present.
- **A region's `phase_err` (and hence its `abserr_roundoff`/`total_err`) can be `+inf`.** Any
  future code that sums, plots, or otherwise processes `used_interval.total_err` /
  `.abserr_roundoff` across regions (prompt 07's diagnostics rewrite, in particular) needs to
  handle a non-finite value gracefully — printing it is fine (Python's `f"{float('inf'):.3g}"`
  renders as `"inf"`), but do not assume finiteness when computing a ratio or a plot axis range
  from it.
- **The `_LEVIN_ROUNDOFF_G0_NOISE_FLOOR` fix was found by running the campaign's own existing
  regression test, not by anything in the prompt's own text.** Anyone implementing a later prompt
  against this module should keep re-running the *existing* test suite after each change, not just
  the tests a given prompt's "Verification" section calls out by name — this defect would have
  shipped invisibly if `test_stationary_phase_gate_uses_total_variation`'s value assertion had not
  already existed from prompt 03.
