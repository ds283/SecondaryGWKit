# Log 01 — Independent references and the measurement harness

**Prompt:** prompts/transfer-remedial/01-reference-harness.md
**Commit:** *(SHA intentionally not embedded — the log and the board are inside the commit they
would name; the same precedent `prompts/source-remediation` logs 01 and 11 set)* — "Add independent
Bessel references and a measurement harness", the single commit that adds this log
**Model:** Claude Opus 5
**Date:** 2026-09-10
**Result:** COMPLETE WITH DEVIATIONS

**Tree:** executed on `f17f2d4`, 16 commits after the campaign baseline `95cc326`.
`git diff --stat 95cc326..f17f2d4 -- LiouvilleGreen/` is **empty**, so `RECONCILIATION.md` §1, §2
and §3.1 apply verbatim; nine of their measurements were re-confirmed independently here (see
"Verification performed"). Also worth recording because README §4.2 flags it as a risk to
Workstream B: `source-remediation` prompt 12 has now run (`5b82149`), so the scheduling hazard it
describes has cleared.

---

## What shipped

Four new files. **No production code was modified** — `git diff HEAD~1 --stat` touches only
`LiouvilleGreen/tests/` (new files), `docs/transfer-remedial/` (new), `docs/OPEN_ISSUES.md` and this
campaign's board and log.

### `LiouvilleGreen/tests/bessel_reference.py` (new, 1093 lines)

The campaign's single home for its metrics and references. Contains no `TestCase`, so
`unittest discover` does not collect it. Imports `numpy` and `scipy.special.{jv,yv}` at module
scope and **`mpmath` only lazily**, inside the four functions that need it.

*Error metrics* — `DRAFT-PLAN.md` §6.1 / README §6, each returning `(max_error, argmax_index)`:

- `phase_pair_error(sin_theta, minus_cos_theta, J, Y, amplitude) -> (float, int)`
- `amplitude_error(A_ours, amplitude) -> (float, int)`
- `derivative_error(theta_prime_ours, theta_prime_ref) -> (float, int)`

All three accept scalars or arrays (flattened; a scalar reports index 0).

*Geometry of the construction*, reproduced rather than imported so that nothing here depends on
`bessel_phase`:

- `c_nu(nu) -> float` — `pi/4 - pi*nu/2`
- `construction_min_x(nu) -> float` — `sqrt(nu^2 - 1/4)` for `nu > 1/2`, else `1e-5`
  (mirrors `bessel_phase.py:85-88`)
- `tail_residual_series(nu, x)` — the two-term DLMF 10.18.18 residual, used **only** to anchor the
  residual branch in the generator; prompt 03 owns the tail proper

*Derived reference quantities*, tier-independent: `reference_amplitude(J, Y)`,
`reference_theta_deriv(x, J, Y)`, `reference_theta_principal(J, Y)`, `reference_a(x, J, Y)`.

*The three tiers*, plus an explicit selector: `exact_half_integer(nu, x)`, `scipy_reference(nu, x)`,
`mpmath_reference(nu, x, dps=70)`, `mpmath_reference_extended(nu, x, dps=70)`,
`reference_JY(nu, x, tier, dps=70)`, `reference_bundle(nu, x, tier, dps=70)`,
`best_available_tier(nu, x_max)`, `scipy_reference_max_x(nu)`.

*The cached corner table*: `load_reference_table`, `cached_corners(nu=None)`, `cached_orders()`,
`cached_corner(nu, x)`, `corner_x_values(nu)`, `regenerate_reference_table(...)`.

### `LiouvilleGreen/tests/bessel_reference_data.json` (new, 73 entries, 38 kB)

70-digit `mpmath` corner references for seven orders, with an environment block. Schema, contents
and regeneration in "State handed to the next prompt".

### `LiouvilleGreen/tests/test_bessel_reference.py` (new, 12 tests, 0.29 s)

Tests of the harness: the five the prompt names, plus six more (the high-order SciPy refusal, the
corners' internal self-consistency including branch closure, the cross-check against
`RECONCILIATION.md` C2/§3.1, the `nu=1/2` exactness identities, the tier-selector contract, and
`test_no_test_imports_mpmath`). The module never imports `mpmath`.

### `docs/transfer-remedial/measure_bessel_phase.py` + `baseline-2026-09.md` (new)

The diagnostic and its committed output: 305 lines, five sections — accuracy (24 cases × 3 point
sets), cost/chunking/`phi`/evaluation cost, the cost-and-cliff sweep with the RHS-noise mechanism
and the new order-dependent SciPy boundary, the `test_phase_derivative` margin, and the test suite's
per-module wall clock.

---

## Deviations from the prompt

### 1. `scipy_reference` also refuses above an *order-dependent* boundary — STRUCTURALLY REQUIRED

**The prompt assumed** one boundary: "Must **refuse**, by raising, for
\(x>\texttt{SCIPY\_REFERENCE\_MAX\_X}\), a module constant you set to \(2\times10^{15}\)".

**What is actually there:** the Amos boundary is order dependent, and the collapse applies to
`jv`/`yv`, not only to `hankel1e`. `RECONCILIATION.md` §1 measured 7.13e8 for `hankel1e` at
\(\nu\in\{100.5,1000.5\}\) but did not measure `jv`/`yv`, which are what this tier, three of the
four existing `test_bessel_phase` tests and `DRAFT-PLAN.md` §12.1's own script are built from.
Probing with \(a=\sqrt{\pi x/2}\,\lvert(J,Y)\rvert\), which is \(1+O(\nu^2/x^2)\) and so must be 1
to twelve places over \(10^8\le x\le2\times10^{15}\):

| \(\nu\) | 2.5 | 20.5 | 50.5 | 70.5 | 80.5 | **85.5** | **88.5** | 89.5 | 90.5 | 100.5 | 1000.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| \(\max\lvert a-1\rvert\) | 4.4e-16 | 1.0e-14 | 6.4e-14 | 1.2e-13 | 1.6e-13 | 1.8e-13 | **1** | 0.998 | 1 | 1 | 1 |

and in \(x\) at \(\nu=100.5\) the transition is abrupt: \(a=0.9999999872\) at \(x=7.108\times10^8\),
\(0.0848\) at \(7.188\times10^8\), then wandering over \([0.012,\,0.99]\) — finite, non-zero, and
wrong by up to a factor 100, so `isfinite` passes it. That reproduces `RECONCILIATION.md` §1's
7.13e8 to three figures.

**What was done instead:** `SCIPY_REFERENCE_MAX_X = 2.0e15` is present and unchanged, and
`scipy_reference` enforces it. Two further constants were added,
`SCIPY_REFERENCE_HIGH_ORDER_NU = 85.5` and `SCIPY_REFERENCE_HIGH_ORDER_MAX_X = 7.13e8`, and the
refusal is taken from `scipy_reference_max_x(nu)`, which returns the collapsed boundary for
\(\nu>85.5\) and 2e15 otherwise. The order threshold is only *bracketed* — 85.5 clean, 88.5
destroyed — and is set at the conservative end.

This is classified structurally required rather than as a choice because the prompt's own words
leave no room: "A silent wrong answer here is the exact failure this campaign is fixing; do not let
a reference reproduce it." Shipping a reference that returns \(a=0.55\) at \(\nu=100.5,\,x=10^9\)
would have done exactly that, inside the instrument used to measure the fix. It touches none of the
six items README §4.3 makes a stop condition. Opened as board issue
`[01-scipy-jv-yv-high-order-boundary]` and handed to prompt 02, which owns pinning the boundaries.

### 2. Two `x` values were added to the cached grid — IMPLEMENTATION CHOICE

The prompt's grid is `{x_0, 1.5 x_0, 10nu, 100nu, 1e3, 1e7, 1e12, 1e15}` restricted to `x >= x_0`,
"plus 2.5e15 and 1e16 for the orders whose supported domain reaches there", and says "at minimum".
Two additions:

- **`x = 10` for every order with `x_0 < 10`** (0.5, 1.5, 1.75, 2.5). Required by the prompt's own
  §2.3 item 1, which asks the three tiers to be compared at `x in {10, 1e3, 1e7}` *against the
  cached corners* while §2.3 also requires the test not to touch `mpmath`; 10 is neither `10nu` nor
  `100nu` for any covered order, so without it that test could not be written as specified.
  Alternative considered: let the test call `mpmath` directly for the `x = 10` point. Rejected —
  §2.3's "must not import `mpmath` at test time" is the stronger constraint, and one extra cached
  row costs 0.5 kB.
- **`2.5e15` and `1e16` for *all* seven orders**, not a subset. The supported \((\nu,x_{\max})\)
  domain is not declared until prompt 05, so restricting the grid now would mean guessing it; every
  order has \(x_0\) at least eleven decades below \(2.5\times10^{15}\), and the marginal cost is
  14 rows and ~0.02 s of generation. Alternative considered: restrict to \(\nu\in\{1/2,5/2\}\), the
  orders production builds. Rejected because `bessel_tier.py`'s \(\kappa=1000\) case reaches
  \(x\approx8.6\times10^{15}\) and prompts 05, 06 and 09 will need references above the cliff at
  whatever orders they test.

### 3. The residual branch walk uses a double-precision probe, not `mpmath`, between corners — IMPLEMENTATION CHOICE

The prompt specifies the *method* (anchor on the series at the largest `x`, track down) but not what
supplies the phase between corners. `mpmath` cannot: at \(\nu=1000.5,\,x=8436.9\) its `0F1` series
escalates to 7323 bits of working precision and then raises `hypsum() failed to converge`, and with
`maxprec=1e6` bits it succeeds but costs **1.5 s per point** — and the walk needs ~2000 points at
that order.

So the walk takes `theta_principal` and `a` from the cheapest *valid* tier (`_walk_probe`): SciPy
below `scipy_reference_max_x(nu)`, `mpmath` above it, where the step count is capped by
`_WALK_MAX_DLOGX` rather than by the residual's motion. The *arithmetic* is still `mpmath` — at
\(x=10^{16}\) the cycle count is \(1.59\times10^{15}\), so \(2\pi n\) carries ~2 rad of
double-precision rounding and a double walk could not represent its own residual — and every
**stored** residual is recomputed from `mpmath_reference_extended` at the corner and required to
agree with the walked value, so the probe fixes only the branch, never a stored number.
Alternatives considered: (i) `mpmath` throughout — rejected on the 1.5 s/point cost, ~50 min at
\(\nu=1000.5\) alone; (ii) resolve every corner directly from the series instead of walking —
rejected because the series is not valid at \(x_0\), which is precisely where the branch is 91
cycles from zero and where the answer matters. Whole-table generation now costs **0.7 s**.

Two consequences worth flagging for later prompts. `MPMATH_MAXTERMS = 1e7` and
`MPMATH_MAXPREC = 1e6` are needed on every `besselj`/`bessely` call at high order near the turning
point, defaults are not enough, and the failure is a raise rather than a wrong answer. And the walk
is self-validating: it predicts `r` by first-order Taylor from the exact identity
\(dr/d\log x=x(a^{-2}-1)\), aims at 0.3 rad of advance per step, and halves the step if the lifted
value disagrees with the prediction by more than 1.2 rad — an order of margin below the \(\pi\) at
which the lift becomes ambiguous.

### 4. The report's five sections do not map one-to-one onto the prompt's §2.4/§4 list — IMPLEMENTATION CHOICE

Everything §2.4 and §4 ask for is present, but grouped as accuracy / cost / cliff / gate / suite
rather than in the prompt's listing order, because the three point sets multiply the accuracy table
by three and merging cost columns into it made a 15-column table unreadable. §1 of the report and
§2 of the report carry the same 24 cases in the same order, so the two are read side by side.

The `nfev` figure the prompt asks for "if obtainable" is obtainable two ways, and both are reported:
a `jv`/`yv` call count taken by rebinding those names in `LiouvilleGreen.bessel_phase` for the
duration of the build (which mixes the ODE with the modulus samples, the initial condition and the
`root_scalar` bracket search), and an independent `solve_ivp` replication of the same ODE — same
right-hand side, stepper, initial condition and tolerances — which gives a clean `nfev` and accepted
step count.

### 5. `test_bessel_reference.py` has 12 tests, not the 5 the prompt lists — IMPLEMENTATION CHOICE

The five named are all present. The additions are: the high-order refusal of deviation 1; the
corners' internal self-consistency, including that the stored branch closes
(`theta_principal + 2*pi*theta_div_2pi == x + c_nu + r`, to an `eps*x` budget); the cross-check of
`r(x_0)` and `a(x_0)` against `RECONCILIATION.md` C2 and §3.1, which were measured by a *different*
route (continuously tracked `hankel1e`, no series anchor, no `mpmath`) and so are a genuine
independent confirmation of the branch rather than a restatement; the `nu=1/2` exactness identities
(`a==1`, `r==0`, `theta'==1`); and the tier-selector contract. Also `test_no_test_imports_mpmath`
asserts §2.3's and §5's cost requirement in a subprocess rather than leaving it as a comment: it
re-runs the module's other eleven tests in a fresh interpreter and checks `sys.modules` afterwards,
which is strictly stronger than checking the import, since it is the test *bodies* that would
otherwise reach for `mpmath`.

### 6. No deviation on the phase-pair metric's calibration, but a fact worth recording — none

Not a deviation, recorded because a later prompt will otherwise re-derive it: a phase error
\(\delta\) at a point where the phase is \(\theta\) registers as
\(\delta\max(\lvert\cos\theta\rvert,\lvert\sin\theta\rvert)\), i.e. between \(\delta/\sqrt2\) and
\(\delta\), never more and never zero. That bounded, \(x\)-independent relationship is asserted by
`test_phase_pair_error_on_perfect_and_perturbed_input`. So `E_theta` may be read as a phase error in
radians to within a factor \(\sqrt2\), and a target of \(10^{-11}\) in `E_theta` is a target of
\(\le1.42\times10^{-11}\) rad.

---

## Verification performed

Everything below was **run**, on `f17f2d4`, and the numbers are quoted from the run. Nothing in this
section is a reasoned claim, and nothing needs a run the user must do.

### The prompt's four acceptance criteria

**(1) `test_bessel_reference.py` passes in under ~10 s without importing `mpmath`.**
`PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_reference` →
`Ran 12 tests in 0.292s / OK`, 0.72 s of wall clock including interpreter start.
`test_no_test_imports_mpmath` asserts the second half literally: it re-runs this module's other
eleven tests in a fresh interpreter and prints `wasSuccessful()` and `'mpmath' in sys.modules` →
`True False`. Two tests were reshaped to make that true rather than nearly true — the high-order
refusal test now demonstrates the wrong `jv`/`yv` value against the *cached* corner at
\(\nu=100.5,\,x=10^{12}\) (where `a` should be 1 and `jv`/`yv` give **0.611**) instead of calling
`mpmath` live, and the tier-selector test no longer evaluates the `mpmath` tier. That tier is not
thereby untested: it produced `bessel_reference_data.json`, which eight of the twelve tests score
against.

**(2) The three reference tiers agree to \(10^{-14}\) where they overlap.** Worst relative
disagreement over \(\nu\in\{1/2,3/2,5/2\}\times x\in\{10,10^3,10^7\}\) on \(J\), \(Y\) and \(A\):
**5.44e-15**, at `scipy` \(J_{1/2}(10)\). The `exact` tier alone is much tighter — \(\le\)**2.22e-16**
relative against the cached `mpmath` corners at every one of those nine points — which is the figure
that matters, since `exact` is the tier with no shared library at all. Wronskian
\(\lvert a^2\theta'-1\rvert\) over all **73** cached corners: worst **2.22e-16**, at
\(\nu=3/2,\,x=10\) (tolerance 1e-13).

**(3) The existing suite does not regress.** The full discovery run cannot be used as an acceptance
gate on this machine — `RECONCILIATION.md` §3.4 records it as not completing in 50 minutes — so it
was run per module, as standing note 13 directs:

| module | status | wall clock |
|---|---|---:|
| `test_bessel_phase` | **OK** (4 tests, including `test_phase_derivative` at 1e-6) | 1.2 s |
| `test_bessel_reference` | **OK** (12 tests, new) | 0.6 s |
| `test_range_reduce` | **OK** | 0.1 s |
| `test_three_bessel` | **OK** | 7.5 s |
| `test_3bessel_analytic` | **TIMEOUT** at the 15-minute cap; pre-existing (`RECONCILIATION.md` §3.4) | 15.0 min |

`test_3bessel_analytic` was then re-run with a 90-minute cap to check it is slow rather than broken;
see "Observations not acted on" for the outcome. This commit adds only new files and changes no
existing test module and no production module, so it cannot affect it.

**(4) `baseline-2026-09.md` contains every quantity §2.4 and §4 list.** It does: build wall clock,
`jv`/`yv` counts and an independent ODE `nfev` plus accepted steps, sample count and
`phase_spline.num_chunks`, per-call evaluation cost for `theta_mod_2pi` / `raw_theta` /
`theta_deriv` / `mod`, \(E_\theta\) / \(E_A\) / derivative relative error each with its argmax \(x\),
`phi`, the three point sets kept separate, the cliff sweep with per-case timeout, the five-adjacent-
doubles RHS noise samples, and the test-suite wall clock.

### Nine of `RECONCILIATION.md`'s measurements, re-confirmed by an independent route

This was not asked for, but it is the cheapest possible check that the harness is measuring the
right object, and it is the reason to believe the rest of the numbers.

| Quantity | `RECONCILIATION.md` | Measured here | Route here |
|---|---|---|---|
| `phi`, \(\nu=3/2\) | \(-1.149353\times10^{-8}\) | \(-1.149353\times10^{-8}\) | `bessel_phase` return value |
| `phi`, \(\nu=7/4\) | \(-2.044386\times10^{-8}\) | \(-2.044386\times10^{-8}\) | as above |
| `phi`, \(\nu=5/2\) | \(-4.836537\times10^{-8}\) | \(-4.836537\times10^{-8}\) | as above |
| \(E_\theta\), fixture tol, \(x\le10^3\) (README §1) | 2.0e-6 | **2.018e-6** at \(\nu=3/2,\,x=578.0\) | `phase_pair_error`, `exact` tier |
| \(r(x_0)\): 3/2, 5/2, 20.5, 100.5, 1000.5 | 0.6155, 1.1832, 11.4437, 57.1042, 570.8200 | **0.615480, 1.183200, 11.443651, 57.104238, 570.820039** | `mpmath` + series-anchored branch walk (C2 used tracked `hankel1e`) |
| \(r(x_\star=100\nu)\): same orders | 0.006667, 0.012000, 0.102440, 0.502492, 5.002540 | **identical to six places** | as above |
| \(\max a=a(x_0)\): 5/2, 20.5, 100.5, 1000.5 | 1.32288, 1.85684, 2.41795, 3.54597 | **1.322876, 1.856838, 2.417953, 3.545969** | as above |
| \(\min a=a(x_\star)\): same orders | 1.0000240009, 1.0000249867, 1.0000250009, 1.0000250016 | **identical to ten places** | as above |
| `num_chunks` vs \(x_{\max}\) | 2 (1e3), 4 (1e7), 5 (1e10), 7 (1e13), 8 (1e15) | **2 (1e3), 4 (1e7), 6 (1e11), 7 (1e13), 7 (1e14), 8 (1e15), 8 (2e15)** | `phase_spline.num_chunks` |
| `hankel1e`/`jv`/`yv` boundary 7.13e8 at high \(\nu\) | for `hankel1e` | **7.11e8–7.19e8 for `jv`/`yv` too** | \(a\) probe, deviation 1 |

The \(\nu=1000.5\) residual agreement is the one that matters most: 570.820039 against 570.8200,
obtained by `mpmath` with a two-term-series anchor versus continuously tracked `hankel1e` with no
anchor. A branch error of one cycle would show as a discrepancy of 6.28.

### The baseline itself — headline figures

Full tables in `docs/transfer-remedial/baseline-2026-09.md`. Every maximum below carries its
\((\nu,x)\).

**Fixture tolerances** (`rtol=1e-8, atol=1e-10`), worst \(E_\theta\) per order at
\(x_{\max}=10^3\) / \(10^7\), over log-interval midpoints:

| \(\nu\) | 1/2 | 3/2 | 7/4 | 5/2 | 20.5 | 100.5 |
|---|---|---|---|---|---|---|
| \(x_{\max}=10^3\) | 2.228e-7 | 1.981e-6 | 1.551e-6 | 2.978e-6 | 1.114e-6 | 3.884e-7 |
| \(x_{\max}=10^7\) | **1.990e+00** | 1.587e-3 | 1.953e-3 | 1.732e-3 | 3.683e-3 | 1.385e-2 |

The \(\nu=1/2,\,x_{\max}=10^7\) figure of 1.990 (at \(x=6.896\times10^5\)) is not a typo: at fixture
tolerances that case has no useful accuracy at all. It is the worst manifestation of M1 —
\(\delta\theta=x\,\delta Q\) with \(\nu=1/2\)'s lower bound at \(10^{-5}\), i.e. 27.6 e-folds of
integration.

**Production tolerances** (`rtol=5e-14, atol=1e-25`), the same cells:

| \(\nu\) | 1/2 | 3/2 | 7/4 | 5/2 | 20.5 | 100.5 |
|---|---|---|---|---|---|---|
| \(x_{\max}=10^3\), nodes | 1.453e-8 | 1.153e-8 | 2.047e-8 | 4.840e-8 | 1.550e-7 | 1.648e-7 |
| \(\lvert\)`phi`\(\rvert\) | 1.452e-8 | 1.149e-8 | 2.044e-8 | 4.837e-8 | 1.550e-7 | 1.648e-7 |
| \(x_{\max}=10^7\), midpoints | 6.465e-6 | 1.108e-7 | 1.236e-7 | 1.754e-7 | 3.476e-7 | 2.919e-3 |
| \(x_{\max}=10^7\), endpoints | 7.094e-5 | 7.148e-5 | 7.192e-5 | 7.065e-5 | 7.064e-5 | 3.015e-3 |

Row 1 against row 2 is M2 stated as a measurement: at production tolerances and
\(x_{\max}=10^3\), \(E_\theta\) **is** \(\lvert\)`phi`\(\rvert\) to three significant figures at
every order. The endpoint row is a separate effect the campaign has not named — the last
log-interval is ~7e-5 at \(x_{\max}=10^7\) regardless of order, three decades worse than the
interior, which is `XSplineWrapper`'s clamping at `max_x` plus the interpolant's end interval.

Worst \(E_A\): **2.902e-6** at \(\nu=100.5,\,x=100.6\) (the turning-point interval); **1.170e-10**
at \(\nu=5/2,\,x=2.453\); \(\le\)3.6e-11 at \(\nu\le7/4\). Worst derivative relative error:
**6.377e-5** at \(\nu=100.5,\,x=100.5\), i.e. exactly at \(x_0\) — confirming `DRAFT-PLAN.md` §4.7's
claim that every derivative maximum falls in the interval adjacent to the turning point, which is
what the argmax output of the metrics exists to establish.

**`test_phase_derivative`'s margin against its \(10^{-6}\) contract** (`RECONCILIATION.md` C4),
measured over exactly its own 25 points and default tolerances:

| \(\nu\) | relative error | at \(x\) | margin |
|---|---|---|---|
| 2.5 | 6.072e-8 | 123.9 | **16.5×** |
| 20.5 | 2.256e-8 | 533.8 | **44.3×** |
| 100.5 | 2.212e-8 | 327.5 | **45.2×** |

Identical to seven figures whether scored against the `exact` tier or against `jv`/`yv` as the test
itself does. Note the contrast with the 6.377e-5 above: the test's lower bound of \(2x_0+1\) excludes
the interval where the error is 2900× larger, so 16.5× of margin is a statement about the interior
only.

**The cliff sweep**, production tolerances, 60 s per case, each in its own subprocess:

| \(x_{\max}\) | 1e11 | 1e13 | 1e14 | 1e15 | 2e15 | 3e15 | 8.6e15 |
|---|---|---|---|---|---|---|---|
| \(\nu=1/2\) | 0.059 s | 0.068 s | 0.070 s | 0.074 s | 0.077 s | **TIMEOUT** | **TIMEOUT** |
| \(\nu=5/2\) | 0.065 s | 0.061 s | 0.066 s | 0.069 s | 0.072 s | **TIMEOUT** | **TIMEOUT** |

**Last \(x_{\max}\) at which the existing construction completes: \(2\times10^{15}\), at both
orders.** The build is flat at 0.06–0.08 s across four decades — `DRAFT-PLAN.md` §4.7's ">600 s (did
not complete)" at \(10^{13}\) does not reproduce, exactly as `RECONCILIATION.md` C1 says. And the
mechanism, \((2/\pi)/(x\,m(x))\) at five adjacent doubles:

| \(\nu\) | \(x\) | five adjacent values |
|---|---|---|
| 1/2 | 1e15 | 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 1/2 | 3e15 | 0.988677, 0.939033, 0.944549, 1.001489, 1.064185 |
| 1/2 | 1e16 | 0.750691, 1.685941, 0.833489, 0.872832, 1.639993 |
| 5/2 | 3e15 | 0.946862, 0.758635, 0.776589, 1.007340, 1.419559 |

within the prompt's expected 0.73–1.68.

**A cost figure the campaign did not have.** At \(\nu=100.5,\,x_{\max}=10^7\) and *production*
tolerances the build takes **5.355 s** and the replicated ODE needs **647 642** right-hand-side
evaluations over 42 761 accepted steps — against 0.018–0.068 s and 158–1034 `nfev` for every other
case in the table. So the stall mechanism of C1 is not confined to \(x\gtrsim2.5\times10^{15}\): it
also appears at high order and moderate \(x_{\max}\), two and a half orders of magnitude below the
declared cliff, and it costs 300× in wall clock while still delivering \(E_\theta=2.9\times10^{-3}\).
Recorded as an observation, not acted on.

### Formatting and hygiene

`./venv/bin/python -m black --check LiouvilleGreen/ docs/transfer-remedial/` → `16 files would be
left unchanged`.

---

## Observations not acted on

1. **`test_3bessel_analytic.py` is the suite's whole cost, and it is worse than "slow".** It did not
   finish within the 15-minute per-module cap of the recorded baseline. A 90-minute re-run was
   started to establish whether it terminates at all; whatever it shows, nothing here can affect it
   (this commit adds only new files). `RECONCILIATION.md` §3.4 already records the behaviour; the
   new information is that it is not merely the discovery run that is slow but this one module,
   while the other four together take 9.4 s.

2. **The last log-interval is ~7e-5 in \(E_\theta\) at \(x_{\max}=10^7\), at every order, even at
   production tolerances** — three decades worse than the interior of the same construction, and
   independent of \(\nu\). `XSplineWrapper.__call__` (`bessel_phase.py:42-58`) accepts
   `x <= 1.01*max_x` and then *clamps* to `max_x`, and `phase_spline` does the same, so the top
   1 % of the requested range is served by a constant. Production asks for
   `1.075 * largest_x` (`main.py:507`) precisely to stay off this edge, so it is not currently a
   production defect — but prompt 05 should not reproduce the clamp, and prompt 06's endpoint checks
   should assert against it rather than around it. Not opened as a board issue because prompt 05
   replaces the object outright.

3. **The `nu=100.5, x_max=1e7` production build is a 300× cost outlier** (5.355 s, 647 642 `nfev`),
   as recorded above. This is the C1 stall mechanism appearing far below \(2.5\times10^{15}\).
   Not opened as a board issue: the replacement removes the ODE, so it cannot survive prompt 05.
   It does mean prompt 09 has a second, sharper cost claim available than the one C1 supplies.

4. **`phi` is independent of the ODE tolerances above \(\nu=1/2\).** Identical to seven figures at
   `rtol=1e-8` and `rtol=5e-14` for \(\nu\in\{3/2,7/4,5/2,20.5,100.5\}\), because `root_scalar` is
   called with `xtol=1e-6, rtol=1e-4` (`bessel_phase.py:214-219`) and *its* tolerance, not the
   ODE's, sets the floor. `DRAFT-PLAN.md` §4.3 says `phi` is a spurious artefact; this says the
   artefact's size is set by a hard-coded root-solve tolerance that no caller can reach. Relevant to
   M2 and to prompt 05's justification, not to anything before it.

5. **`LiouvilleGreen/tests/bessel_reference.py` re-exports `jv`, `yv` and `NamedTuple`** into its
   namespace as a side effect of its imports. Harmless, not worth an `__all__`, noted so that a
   later prompt does not read `bessel_reference.jv` as a deliberate part of the surface. Use
   `scipy_reference`.

6. **Nothing was done to `phase_spline`**, its chunking or its `_build_log_chunks_positive` progress
   guard (README §1.1). The baseline records `num_chunks` per case, which is the input prompt 05's
   removal argument needs, and nothing more.

---

## State handed to the next prompt

### Import path and public API of `bessel_reference.py`

    from LiouvilleGreen.tests import bessel_reference as br

Every public name, with its exact signature as shipped:

```python
# --- error metrics; all return (max_error, argmax_index); scalars or arrays accepted -------
phase_pair_error(sin_theta, minus_cos_theta, J, Y, amplitude) -> Tuple[float, int]
amplitude_error(A_ours, amplitude)                            -> Tuple[float, int]
derivative_error(theta_prime_ours, theta_prime_ref)           -> Tuple[float, int]

# --- geometry of the construction ----------------------------------------------------------
c_nu(nu: float) -> float                        # pi/4 - pi*nu/2
construction_min_x(nu: float) -> float          # sqrt(nu^2 - 1/4), or 1e-5 for nu <= 1/2
tail_residual_series(nu: float, x)              # (mu-1)/(8x) + (mu-1)(mu-25)/(384 x^3)

# --- derived reference quantities, tier independent ----------------------------------------
reference_amplitude(J, Y)                       # hypot(J, Y)
reference_theta_deriv(x, J, Y)                  # (2/pi)/(x (J^2 + Y^2)) -- an identity
reference_theta_principal(J, Y)                 # atan2(J, -Y), in (-pi, pi]
reference_a(x, J, Y)                            # sqrt(pi x/2) hypot(J, Y)

# --- the three tiers ------------------------------------------------------------------------
exact_half_integer(nu: float, x)                        -> (J, Y)
scipy_reference(nu: float, x)                           -> (J, Y)   # raises above the boundary
mpmath_reference(nu: float, x, dps: int = 70)           -> (J, Y)   # floats
mpmath_reference_extended(nu: float, x: float, dps: int = 70) -> dict  # mpf values

# --- the explicit selector ------------------------------------------------------------------
reference_JY(nu: float, x, tier: str, dps: int = 70)     -> (J, Y)
reference_bundle(nu: float, x, tier: str, dps: int = 70) -> BesselReference
best_available_tier(nu: float, x_max: float)             -> str
scipy_reference_max_x(nu: float)                         -> float

# --- the cached corner table ----------------------------------------------------------------
load_reference_table(path=REFERENCE_DATA_PATH)           -> dict
cached_corners(nu: Optional[float] = None, path=...)     -> Sequence[CachedCorner]
cached_orders(path=...)                                  -> Sequence[float]
cached_corner(nu: float, x: float, path=...)             -> CachedCorner
corner_x_values(nu: float)                               -> Sequence[float]
regenerate_reference_table(path=..., orders=CACHED_ORDERS, dps=70, verbose=True) -> dict
```

`BesselReference` is a `NamedTuple` with fields
`(nu, tier, x, J, Y, amplitude, a, theta_principal, theta_deriv)`; the array-valued fields carry
whatever shape `x` had.

`CachedCorner` is a `NamedTuple` with fields
`(nu, x, J, Y, amplitude, a, r, theta_principal, theta_div_2pi, theta_deriv, strings)`, all floats
except `theta_div_2pi` (`int`) and `strings` (the raw 40-digit decimal strings of that row). It has
one property, `theta_continuous == theta_principal + 2*pi*theta_div_2pi`, which is \(\varepsilon x\)-
limited at large \(x\) and should not be differenced against `x`.

### Tier-selection convention

`tier` is a **required** argument of `reference_JY` and `reference_bundle` and takes one of
`REFERENCE_TIERS = ("exact", "scipy", "mpmath")`, exposed as `TIER_EXACT`, `TIER_SCIPY`,
`TIER_MPMATH`. There is no default and no automatic promotion; an unknown tier raises `ValueError`.
`best_available_tier(nu, x_max)` returns the cheapest *valid* tier as a convenience for a diagnostic
that sweeps orders, but it is not a default: it returns a name the caller then passes and records.

Boundary constants, and the function that combines them:

| name | value | meaning |
|---|---|---|
| `SCIPY_REFERENCE_MAX_X` | **2.0e15** | universal Amos argument-reduction boundary (`RECONCILIATION.md` C1) |
| `SCIPY_REFERENCE_HIGH_ORDER_NU` | **85.5** | above this order the boundary collapses; bracketed [85.5, 88.5] |
| `SCIPY_REFERENCE_HIGH_ORDER_MAX_X` | **7.13e8** | the collapsed boundary |
| `scipy_reference_max_x(nu)` | 7.13e8 if `nu > 85.5` else 2.0e15 | what `scipy_reference` enforces |
| `EXACT_HALF_INTEGER_ORDERS` | `(-0.5, 0.5, 1.5, 2.5)` | where tier 1 exists; **7/4 is not among them** |
| `DEFAULT_MPMATH_DPS` | 70 | |
| `MPMATH_MAXTERMS`, `MPMATH_MAXPREC` | 1e7, 1e6 | required at high order near the turning point |

`scipy_reference` raises `ValueError` naming `RECONCILIATION.md` and `mpmath`; array arguments are
judged on their largest element.

### `bessel_reference_data.json`

Schema version **1**. Top-level keys: `schema_version`, `generated_by`, `campaign`, `environment`
(`date`, `python`, `numpy`, `scipy`, `mpmath`, `platform`, `mpmath_dps`), `conventions` (prose), and
`entries`. Each entry:

```json
{"nu": "1000.5", "x": "100050.0", "x_hex": "0x1.86d2000000000p+16",
 "J": "...", "Y": "...", "amplitude": "...", "a": "...", "r": "...",
 "theta_principal": "...", "theta_div_2pi": 15674, "theta_deriv": "..."}
```

All numeric fields are **decimal strings with 40 significant digits** except `theta_div_2pi`, which
is a JSON integer. `x` is `repr(float)` and round-trips exactly; `x_hex` is `float.hex()` and
`cached_corners` raises if the two disagree. `theta_principal` is `atan2(J, -Y)` in \((-\pi,\pi]\);
`r` is on its **correct branch**, so `theta_principal + 2*pi*theta_div_2pi == x + c_nu + r`.

Coverage — **73 entries**, orders `CACHED_ORDERS = (0.5, 1.5, 1.75, 2.5, 20.5, 100.5, 1000.5)`:

| \(\nu\) | \(x\) values |
|---|---|
| 0.5 | 1e-5, 1.5e-5, 5, 10, 50, 1e3, 1e7, 1e12, 1e15, 2.5e15, 1e16 |
| 1.5 | 1.41421, 2.12132, 10, 15, 150, 1e3, 1e7, 1e12, 1e15, 2.5e15, 1e16 |
| 1.75 | 1.67705, 2.51558, 10, 17.5, 175, 1e3, 1e7, 1e12, 1e15, 2.5e15, 1e16 |
| 2.5 | 2.44949, 3.67423, 10, 25, 250, 1e3, 1e7, 1e12, 1e15, 2.5e15, 1e16 |
| 20.5 | 20.4939, 30.7409, 205, 1e3, 2050, 1e7, 1e12, 1e15, 2.5e15, 1e16 |
| 100.5 | 100.499, 150.748, 1e3, 1005, 10050, 1e7, 1e12, 1e15, 2.5e15, 1e16 |
| 1000.5 | 1000.5, 1500.75, 10005, 100050, 1e7, 1e12, 1e15, 2.5e15, 1e16 |

i.e. \(x_0\), \(1.5x_0\), \(10\nu\), \(100\nu\), 10, 1e3, 1e7, 1e12, 1e15, 2.5e15, 1e16 restricted to
\(x\ge x_0\). Regenerate with

    PYTHONPATH=. ./venv/bin/python -c \
      "from LiouvilleGreen.tests.bessel_reference import regenerate_reference_table as g; g()"

which takes **0.7 s** and asserts, per order, that the residual at the largest \(x\) matches
`tail_residual_series` (achieved: 0 to **4.17e-38**, worst at \(\nu=1000.5\)) and that the
`mpmath` phase at every corner lifts to within 1.2 rad of the walked residual. It raises rather
than writing a table if either fails. Environment recorded in the committed file: Python 3.12.14,
numpy 2.2.4, scipy 1.15.2, mpmath 1.3.0, macOS-26.5.2-arm64, `mpmath_dps` 70, 2026-09-10.

### Reference values prompts 03, 04 and 05 will want first

From the cached table, so they need not re-derive them. \(x_\star=100\nu\) is *the plan's*
crossover; prompt 03 owns choosing the real one.

| \(\nu\) | \(x_0\) | \(r(x_0)\) | \(a(x_0)\) | \(r(100\nu)\) | \(a(100\nu)\) |
|---|---|---|---|---|---|
| 0.5 | 1e-5 | 0.000000 | 1.0000000000 | 0.000000 | 1.0000000000 |
| 1.5 | 1.414214 | 0.615480 | 1.2247449 | 0.006667 | 1.0000222220 |
| 1.75 | 1.677051 | 0.757566 | 1.2532327 | 0.008036 | 1.0000229594 |
| 2.5 | 2.449490 | 1.183200 | 1.3228757 | 0.012000 | 1.0000240009 |
| 20.5 | 20.493902 | 11.443651 | 1.8568381 | 0.102440 | 1.0000249867 |
| 100.5 | 100.498756 | 57.104238 | 2.4179531 | 0.502492 | 1.0000250009 |
| 1000.5 | 1000.499875 | **570.820039** | **3.5459685** | 5.002540 | 1.0000250016 |

\(\nu=1/2\) is exactly degenerate: \(\mu-1=0\), so \(r\equiv0\), \(a\equiv1\) and
\(\theta'\equiv1\) at **every** \(x\), stored as exact `"0.0"` and `"1.0"` in the table.
`test_derived_quantities_against_the_exact_tier` asserts it; prompt 03 can use it as a free
exactness test.

### Baseline numbers later prompts are scored against

- **Baseline document:** `docs/transfer-remedial/baseline-2026-09.md`, generated by
  `docs/transfer-remedial/measure_bessel_phase.py` (accepts `--skip-suite` and `--suite-timeout`).
- **Last \(x_{\max}\) at which the existing construction completes: \(2\times10^{15}\)**, at both
  \(\nu=1/2\) and \(\nu=5/2\), production tolerances, 60 s cap. \(3\times10^{15}\) and
  \(8.6\times10^{15}\) time out. Builds are flat at 0.06–0.08 s from \(10^{11}\) to
  \(2\times10^{15}\).
- **\(E_\theta\) to beat, production tolerances:** 1.15e-8 (\(\nu=3/2\), \(x_{\max}=10^3\), nodes),
  1.11e-7 (\(\nu=3/2\), \(10^7\), midpoints), 7.15e-5 (\(\nu=3/2\), \(10^7\), endpoint interval),
  2.92e-3 (\(\nu=100.5\), \(10^7\), midpoints). At fixture tolerances, 2.02e-6 (\(\nu=3/2\),
  \(10^3\)) and 1.99e+00 (\(\nu=1/2\), \(10^7\)).
- **\(E_A\) to beat:** 2.90e-6 (\(\nu=100.5\), \(x=100.6\)); 1.17e-10 (\(\nu=5/2\), \(x=2.453\)).
- **Derivative relative error to beat:** 6.38e-5 at \(\nu=100.5\), \(x=100.5\) — *at* \(x_0\).
  Every derivative maximum in every case falls in the interval adjacent to the turning point or at
  the far endpoint, as `DRAFT-PLAN.md` §4.7 predicts.
- **`test_phase_derivative`'s margin against \(10^{-6}\):** 16.5× (\(\nu=2.5\), worst 6.072e-8 at
  \(x=123.9\)), 44.3× (20.5, 2.256e-8 at 533.8), 45.2× (100.5, 2.212e-8 at 327.5). This is the
  standing gate; from prompt 05 onward it must pass, and it is never loosened.
- **Test suite per module:** `test_bessel_phase` 1.2 s OK, `test_bessel_reference` 0.6 s OK,
  `test_range_reduce` 0.1 s OK, `test_three_bessel` 7.5 s OK, `test_3bessel_analytic` **does not
  finish in 15 min** (pre-existing). Prefer per-module runs; do not read a long run as a regression.

### Two traps a later prompt would otherwise hit

1. **`mpmath.besselj`/`bessely` fail at high order near the turning point with default caps.**
   Pass `maxterms=MPMATH_MAXTERMS, maxprec=MPMATH_MAXPREC`, as `mpmath_reference` and
   `mpmath_reference_extended` do. Without `maxprec` the call *raises*
   (`hypsum() failed to converge to the requested 288 bits ... using a working precision of 7323
   bits`) at, for instance, \(\nu=1000.5,\,x=8436.9\). With it, the worst interior points cost
   ~1.5 s each — do not put that tier inside a loop over thousands of points.
2. **`jv`/`yv` are not a reference above 7.13e8 for \(\nu\gtrsim86\)**, not merely above
   \(2.5\times10^{15}\) (deviation 1, board issue `[01-scipy-jv-yv-high-order-boundary]`). Prompt 02
   pins it; prompt 04's plausibility band should be calibrated against
   `scipy_reference_max_x(nu)`, and any high-order test above 7.13e8 must use the cached corners.
