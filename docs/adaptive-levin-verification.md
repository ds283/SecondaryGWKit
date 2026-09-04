# Levin refactor campaign: verification and close-out

**Campaign:** [`prompts/levin-refactor/README.md`](../prompts/levin-refactor/README.md)
**Status board:** [`prompts/levin-refactor/IMPLEMENTATION_STATE.md`](../prompts/levin-refactor/IMPLEMENTATION_STATE.md)
**Source audit:** [`docs/adaptive-levin-audit-2026-09.md`](adaptive-levin-audit-2026-09.md)
**Prompt:** [`prompts/levin-refactor/10-test-matrix.md`](../prompts/levin-refactor/10-test-matrix.md)
**Date:** 2026-09-04

---

## 1. What this document is

Prompts 01–09 landed the nine commits that fix the audit's C1–C11 findings and implement
recommendations 1–15. Each prompt's own log recorded before/after numbers at the time it was
written — against the *previous* prompt's commit, not against the pre-campaign baseline, and not
re-run by anyone outside that prompt's own session. This document goes back through every one of
those nine logs and independently re-runs the claims that matter most, on the actual `HEAD`
(`d384031`) against the actual pre-campaign baseline (`c8a1918`), using this repository's own venv.
It also completes the C12 test-matrix gaps (Part 1 of the prompt, in
`AdaptiveLevin/tests/test_levin_quadrature.py`).

**Method used throughout:** the pre-campaign module is loaded via
`importlib.util.spec_from_file_location` from `git show c8a1918:AdaptiveLevin/levin_quadrature.py`,
exactly as prompts 02/05/06 did in their own logs, so both versions run in the same process against
the same NumPy/SciPy/LAPACK. Every number below was actually executed in this session
(`PYTHONPATH=. ./venv/bin/python`), not transcribed from a prior log. Scratch scripts live under
this session's scratchpad directory (not committed, per this campaign's own convention — see
prompt 02/05/06/07's logs for the same practice).

**Headline results, ahead of the detail below:**

- **C1–C4, C6, C8, C11 all reproduce exactly as the prior prompts claimed.** No contradiction found.
- **The delivered accuracy of the three-Bessel oracles is unchanged by the campaign** (all seven
  agree with the pre-campaign values to round-off, §4.7) — the campaign improved *reporting* and
  *robustness*, not the numbers a caller already trusted.
- **The audit's/prompt 02's 1.4–1.8× "end-to-end" speedup claim does not describe the net effect of
  the finished campaign.** In isolation, the complexified solve is confirmed faster (§4.5, real vs
  complex path microbenchmark and the isolated multi-region case). But measured end-to-end against
  the pre-campaign baseline on the same synthetic problems, the net speedup ranges from **0.94× to
  1.26×**, and on the production three-Bessel `J000` oracle through the real caller chain, the net
  effect is a **3.7× slowdown** (0.27× "speedup"), because prompt 03's total-variation gate fix (a
  necessary correctness fix, C2) increases evaluation counts by 3.4× on this exact oracle — a cost
  this campaign's own `IMPLEMENTATION_STATE.md` §3 already flags
  (`[03-fallback-cost-on-difference-groups]`) but had not, until this document, been stated as a net
  wall-clock number against the pre-campaign baseline. See §4.5.

---

## 2. Reconciliation

### 2.1 Git history and commit SHAs

```
6bfd5d1 Plan the adaptive Levin quadrature refactor
0002fb4 Make Levin quadrature refuse bad input rather than certifying it        (prompt 01)
5d5b958 Solve the Levin collocation system in its complex N x N form           (prompt 02)
c4f2ab8 Gate the Levin fallback on total variation, not net phase change      (prompt 03)
13773b9 Report the round-off floor Chen et al. derive, not an endpoint model  (prompt 04)
88f6ab3 Distribute atol across subintervals so it bounds the total error     (prompt 05)
ab8f7ca Make the Levin mode filter self-consistent with the estimate it feeds (prompt 06)
78cae32 Make the Levin module cheap to import and honest about its counters   (prompt 07)
c4dc41d Retune the Levin spectral order and add vectorised sampling           (prompt 08)
d384031 Carry the Levin error estimate out to the three-Bessel callers       (prompt 09)
```

One commit per prompt, in order; `d384031` is `HEAD` at the start of this prompt's work — the
working tree is exactly the campaign's finished state. Every SHA above was independently confirmed
reachable with `git merge-base --is-ancestor <sha> HEAD` (all returned true) and its subject line
was read directly from `git log`, not copied from the board. These are now filled into
`IMPLEMENTATION_STATE.md`'s status board (this document does not duplicate that table).

### 2.2 Standing facts re-confirmed

- `AdaptiveLevin/levin_quadrature.py` at `c8a1918` imports and runs standalone via
  `importlib.util` (needed for every comparison below): confirmed.
- `DEFAULT_LEVIN_CHEBSHEV_ORDER = 16` (`levin_quadrature.py:150`), `_LEVIN_MINIMUM_ALLOWED_ORDER = 8`
  — confirmed by direct read.
- `ComputeTargets/QuadSourceIntegral.py`'s `CHEBYSHEV_ORDER = 24` (`:47`) — confirmed by direct read
  (prompt 08's claim; not 12, not 64).
- `LiouvilleGreen/three_bessel_integrals.py`'s `DEFAULT_3BESSEL_CHEBYSHEV_ORDER = 12` (`:61`) —
  confirmed unchanged, as prompt 08 states.

---

## 3. Part 1 — the C12 test matrix

`AdaptiveLevin/tests/test_levin_quadrature.py` grew from 28 tests (post-prompt-08 baseline) to
**32 tests**, all passing in **0.043–0.059 s** (`PYTHONPATH=. ./venv/bin/python -m unittest discover
-s AdaptiveLevin/tests -t .`), comfortably inside the "a few seconds" budget. No test needed a
slow-run marker or a separate module — every new test runs in milliseconds.

| C12 gap | Status | Test(s) |
|---|---|---|
| `theta_mod_2pi` path (production's own shape: `theta` + `theta_mod_2pi`, no `theta_deriv`) | **written** | `test_theta_mod_2pi_path_only` |
| `theta_deriv` path in isolation (no `theta_mod_2pi`) | **written** | `test_theta_deriv_path_only` |
| A region taking the fallback deliberately | verified/extended | `test_fallback_region_bisects_on_missed_tolerance` (03); `test_GRZIntegral` now asserts the path explicitly (see below) |
| Non-monotonic phase (C2) | verified | `test_stationary_phase_gate_uses_total_variation` (03) |
| Non-finite amplitude (C1) | verified | `test_nan_amplitude_raises`, `test_all_nan_amplitude_raises` (01) |
| Reversed span | **written** | `test_reversed_span` |
| `m != 2` | **written** | `test_generic_m_component_basis_two_decoupled_phases` (see §3.1) |
| `atol = 0` | verified | `test_atol_zero_rejected`, `test_input_validation` (01) |
| `abserr` checked against true error, on every problem with a closed form | **extended to all** | see §3.2 |

### 3.1 The `m != 2` test — the hardest row

There is no public entry point for `m != 2` (`adaptive_levin_sincos` hard-validates
`len(f) == 2`), so the only way to exercise the generic path is to construct a basis object by hand
and drive `_adaptive_levin()` directly, as the prompt anticipates. `_TwoIndependentSinCosBasis`
(module scope, top of the test file) implements `build_Levin_data`/`eval_basis`/`theta_abserr_at`
for an `m = 4` basis: two independent `(sin, cos)` pairs at independent linear phases
`theta1(x) = lambda1*x`, `theta2(x) = lambda2*x`, block-diagonal in `A^T`. This is not a contrived
degenerate case — it is exactly the shape a future basis combining two independent Liouville-Green
phases would take (prior review §8.3, cited in the module docstring).

With `f = [1, 0, 0, 1]` on `x_span = (1, 50)`, the integral has an elementary closed form:
`(cos(1) - cos(50)) + 0.5*(sin(100) - sin(2))`. Verified directly (this session, not just in the
shipped test): value agrees with the closed form to `2.0e-15`, `abserr` (`4.4e-14`) bounds it, and
the driver reports `converged=True`. No design problem was found with constructing this basis from
outside the module — `getattr(BasisData, "supports_complexified_solve", False)` degrades gracefully
to `False` for any basis that doesn't define the attribute, and `m == 2` in the `and` short-circuits
before it would even be consulted for this `m = 4` case, so no `IMPLEMENTATION_STATE.md` §3 issue is opened for this row.

### 3.2 Tightened original-four thresholds, and `abserr`-vs-truth added everywhere

The four original tests asserted `|value - literal| < 1e-10` against a truncated literal (e.g.
`0.5581795618`, itself only accurate to ~3.5e-11 — *looser* than the tightened threshold below would
allow). Replaced with the full-precision closed form and a threshold set from a measured true error,
with a stated margin (not the measured value itself, which would flake):

| Test | Closed form used | Measured true error | New threshold | Margin |
|---|---|---|---|---|
| `test_SinIntegral` | `cos(1) - cos(50000)` | 7.8e-16 | 1e-12 | ~1000x |
| `test_CosIntegral` | `sin(500000) - sin(1)` | 1.2e-15 | 1e-12 | ~1000x |
| `test_SincIntegral` | `Si(10000) - Si(100)` (`scipy.special.sici`) | 9.8e-14 | 1e-11 | ~100x |
| `_GRZIntegral` (lambda=10,100,1000) | `(2/lambda)*sin(pi*lambda/4)` | 2.8e-13 (worst, lambda=100) | 1e-11 | ~35x |

Every one of these four tests, plus `test_stationary_phase_gate_uses_total_variation`,
`test_theta_abserr_declared_endpoint_term`, `test_roundoff_floor_independent_of_theta_presentation`,
`test_theta_mod_2pi_path_only`, `test_theta_deriv_path_only`,
`test_generic_m_component_basis_two_decoupled_phases`, and `test_reversed_span`, now assert
`abserr >= true_err` against a real closed form. Combined with the pre-existing
`LiouvilleGreen/tests/test_3bessel_analytic.py::test_abserr_bounds_truth` (prompt 09, `xfail`), this
covers every closed form README §5.2 lists.

### 3.3 `_GRZIntegral(10.0)` path, made explicit

Re-derived against the **current** total-variation gate rather than assumed unchanged from the
audit's pre-campaign analysis (`theta'(x) = lambda/(1+x^2)` never changes sign on `[-1, 1]`, so total
variation equals net phase change here regardless of which gate is used — this is *not* a
stationary-point case, so the correction doesn't change the outcome, but it does need re-deriving,
not assuming). Measured directly this session:

| lambda | `num_regions` | `num_simple_regions` | Levin ever invoked? |
|---|---|---|---|
| 10 | 8 | 8 | **no** — every region is Clenshaw-Curtis fallback |
| 100 | 24 | 22 | yes (2 Levin regions) |
| 1000 | 4 | 0 | yes (all 4 regions are Levin) |

`test_GRZIntegral` now asserts `num_simple_regions == num_regions` for lambda=10 and
`num_simple_regions < num_regions` for 100 and 1000, so this is a named, checked fact rather than an
incidental one.

---

## 4. Part 2 — verified claims, by audit item

### 4.1 C1 — no input produces a silent zero

Tried the audit's own two cases plus three more the audit's own §1.2 names as realistic triggers
(a spline evaluated outside its range, an overflow in `x^{3/2}*m_mu*m_nu*m_sigma`, a division by
zero in a user amplitude):

| Case | Result |
|---|---|
| Amplitude goes NaN partway through the region (audit's own case) | raises `ValueError`, message names "amplitude" |
| Amplitude entirely NaN | raises `ValueError`, message names "amplitude" |
| Amplitude is a "spline" returning NaN outside its fit range | raises `ValueError`, message names "amplitude" |
| Amplitude overflows to `inf` (`x**1.5 * 1e200 * 1e200 * 1e200`) | raises `ValueError`, message names "amplitude" |
| Amplitude divides by zero at an interior point (`1/(x-1.5)` on `(1,2)`) | raises `ValueError`, message names "amplitude" |

**All five raise.** No silent zero found on any of the five. Confirms prompt 01's claim and closes
the audit's own suggested trigger list (§1.2), not just its literal reproduction case.

### 4.2 C2 — the stationary-phase integral

`I = integral_0^1 exp(-x) sin(1e6*(x - x^2)) dx`, oracle `-6.879079716900e-04` (audit §1.3, a
79,578-panel Gauss-Legendre sum).

| Stage | relative error | notes |
|---|---|---|
| Module as committed (pre-campaign, C2 present) | 1590% | audit's own figure, not re-run here (would need reverting the total-variation-gate fix specifically, not just the whole module) |
| Same code, interval split by hand at x=1/2 | 3.9e-12 | audit's own figure |
| Total-variation-gate prototype (audit's own, without step 5's bisect-on-miss) | 1.1e-8 | audit's own figure |
| Prompt 03's own post-fix measurement (right after prompt 03's commit) | 7.89e-10 | prompt 03's log |
| **This session, at `HEAD` (all nine prompts applied)** | **2.13e-11** | measured this session |

At `HEAD` the relative error is *better* than prompt 03's own post-fix figure by ~37x — consistent
with the accumulated effect of the higher default order (16, prompt 08) and the round-off floor /
mode filter refinements (04, 06) landing on top of prompt 03's structural fix. `num_regions=20`,
`num_simple_regions=6` (so 14 of the 20 accepted regions do use the Levin rule — the C2 fix is fully
live, not accidentally routing everything to the fallback). One curiosity worth recording plainly:
**`abserr` is reported as `+inf`** for this run, not a small number. This is exactly prompt 04's
documented behaviour (`_LEVIN_ROUNDOFF_G0_NOISE_FLOOR`, `IMPLEMENTATION_STATE.md` standing note 19)
— a region with an endpoint at the true stationary point has an undefined round-off floor, and the
module reports that honestly as unbounded rather than fabricating a finite number. The true error
(1.47e-14) is trivially bounded by `+inf`, so the contract ("abserr must not under-report") holds,
but a caller checking `abserr < some_threshold` to decide "is this good enough" would find this
particular run reports `False` regardless of how good the true answer actually is. This is not a new
finding — it is `IMPLEMENTATION_STATE.md` §3's existing `[04-roundoff-floor-can-be-infinite]` issue, restated here because this
is the first time it was observed to fire on the campaign's own headline C2 regression case at
`HEAD`, not just on a hand-constructed reproducer.

### 4.3 C3 — `converged` and the aggregate contract

Six problems (the three closed-form ones from README §5.2 plus three more spanning single-region,
many-region, and the audit's own C3 reproducer) at three tolerances each (`atol` = 1e-8, 1e-10,
1e-12), `rtol = 1e-13` throughout so `atol` governs acceptance — **18 combinations, all measured
directly this session**:

- `converged` was `True` in all 18 cases, and in every case `abserr <= max(atol, rtol*|value|)`
  held exactly whenever `converged` was reported `True` (never a mismatch in either direction).
- The audit's own C3 reproducer (`integral_0^3 exp(-400*(x-1.1)^2) sin(3e4 x) dx`, `atol=1e-10`)
  now reports `abserr = 6.22e-13` and `converged = True` — the pre-campaign behaviour was
  `abserr = 1.27e-10` (silently exceeding the request). Confirmed by direct re-run, matching prompt
  01/05's logs.

### 4.4 C4 — the omega-ladder in both phase modes

`integral_{1/3}^{7/3} exp(-x) sin(omega x) dx`, `atol=1e-18, rtol=1e-13, chebyshev_order=12`, both
"raw" (theta only) and "reduced" (`theta` + `theta_mod_2pi` + `theta_deriv`) phase modes, omega from
1e4 to 1e12 — reproduced exactly, digit for digit against prompt 04's log:

| omega | `abserr` (raw) | `abserr` (reduced) | bounds truth (raw) | bounds truth (reduced) |
|---|---|---|---|---|
| 1e4 | 2.1496e-16 | 2.1496e-16 | yes | yes |
| 1e6 | 2.1496e-16 | 2.1496e-16 | yes | yes |
| 1e8 | 2.1496e-16 | 2.1496e-16 | yes | yes |
| 1e10 | 2.1496e-16 | 2.1496e-16 | yes | yes |
| 1e12 | 2.1496e-16 | 2.1496e-16 | yes | yes |

The reported floor is **identical between raw and reduced presentation at every omega tested** (the
prompt's specific requirement — C4's whole point was that the pre-campaign floor depended on this,
by up to 5.5e10 at high omega) and **bounds the true error in every one of the 10 cells**. No
contradiction found.

### 4.5 The speedup — measured against the pre-campaign baseline, not just prompt-to-prompt

The audit's expectation from complexification alone is 1.4–1.8x (§4.2); prompts 03/07/08 add more
structural change on top. This section reports the **composite** end-to-end effect at `HEAD` versus
`c8a1918`, which none of the nine prompts' own logs computed (each measured its own prompt's delta
against the *immediately preceding* commit).

**Synthetic problems, `chebyshev_order=12` fixed in both versions (isolates the linear-algebra and
per-call-overhead changes from the order-default change), best-of-300, median timing:**

| Problem | region count old -> new | median speedup |
|---|---|---|
| `SinIntegral` (1 region both) | 1 -> 1 | 1.09-1.13x |
| `CosIntegral` (1 region both) | 1 -> 1 | 0.94-0.96x (slightly *slower*) |
| `GRZ_1000` (4 regions both, order 12) | 4 -> 4 | 1.04-1.15x |
| `expsin_1e6` (1 region both) | 1 -> 1 | 1.04-1.13x |
| `SincIntegral` (10 regions both) | 10 -> 10 | 1.20-1.26x |

On problems whose region count is unaffected by the campaign, the net speedup is **0.94x-1.26x** —
present, but well below the audit's 1.4-1.8x isolated per-solve figure. This is not a contradiction
of prompt 02's own measurement (which used the same problems right after prompt 02's commit and
found 1.32-1.52x): the difference is that prompts 04/05/06's added per-region computation (the eq.
151 round-off floor, `_local_atol` scaling, the `abserr_truncation` term, `theta_abserr_at` calls)
sits on the hot path for every region and eats into the linear-algebra win on these small problems,
where fixed per-region overhead was always a large fraction of the cost (audit §4.3: "fixed per-call
driver overhead is 32% of a small call" — a comparable fraction now goes to error-estimate
bookkeeping rather than diagnostics-loop cost).

**Problems whose region count *is* affected by the campaign (the total-variation gate, C2)** show
the opposite sign, confirming and quantifying `IMPLEMENTATION_STATE.md`'s own
`[03-fallback-cost-on-difference-groups]` issue as a genuine net wall-clock cost, not just an
evaluation-count curiosity:

| Problem | region count old -> new | net speedup |
|---|---|---|
| `GRZ_100` (order 12) | 4 -> 24 | 0.40x (2.5x *slower*) |
| `GRZ_1000` (order 16) | 2 -> 4 | 0.57-0.59x (~1.7x *slower*) |

**Production three-Bessel `J000` oracle, through the real caller chain** (`LiouvilleGreen/
three_bessel_integrals.py::quad_JJJ`, `k,q,s=1.3,1.7,2.1`, `max_x=1e5`, `atol=1e-14, rtol=1e-10`,
`chebyshev_order=12` fixed in both — the pre-campaign core was driven through a small shim that adds
the `"converged"`/`"phase_limited"` keys the post-prompt-01 caller code expects, since the
pre-campaign core's return dict predates those keys; the shim adds nothing to what is timed, it only
lets the identical, unmodified `quad_JJJ` post-processing code run without a `KeyError`):

```
old total evaluations across 4 phase groups: 308   (10, 26, 40, 63 regions per group / 25, 63, 85, 135 evals)
new total evaluations across 4 phase groups: 1042  (22, 106, 155, 220 regions per group / 49, 221, 317, 455 evals)
old wall time (best of 3): 0.804 s
new wall time (best of 3): 2.961 s
net speedup: 0.27x  ->  the finished campaign is 3.7x SLOWER on this oracle than the pre-campaign code
```

The 3.4x evaluation-count increase matches prompt 03's own log for this exact oracle (2.2-3.4x,
attributed there to the total-variation gate now correctly applying to the eagerly-computed Levin
comparison children, which the old net-phase gate never tested). **This is not a new defect** — it
is the necessary cost of fixing a critical wrong-answer bug (C2) on exactly the phase-group family
the audit itself flagged as most exposed — but it means the audit's headline "1.4-1.8x end-to-end
speedup" claim, while an accurate and reproducible measurement of complexification *in isolation*,
does not describe what a caller of the finished campaign experiences on this production integrand.
No prior log in this campaign computed this composite number; it is reported here for the first
time, and __no new `IMPLEMENTATION_STATE.md` §3 issue needs opening for it__ — it is a sharper restatement of the existing
`[03-fallback-cost-on-difference-groups]` issue with an actual wall-clock ratio attached, and that
issue's own "next step" (raising `SIX_PI`, evaluated with prompt 03's measurements in hand) remains
the correct place to act on it, not this document.

### 4.6 The `lstsq` share after complexification

Re-derived from the tip's own returned dictionary (`num_solves_direct`/`num_solves_lstsq`/
`num_solves_total`, prompt 07's honest counters) rather than instrumentation, on the `J000` oracle
(`k,q,s=1.3,1.7,2.1`, `max_x=1e5`):

| chebyshev_order | LU | lstsq | total | lstsq share |
|---|---|---|---|---|
| 12 | 91 | 85 | 176 | **48.3%** |
| 16 (module default) | 56 | 69 | 125 | **55.2%** |

The order-12 figure reproduces prompt 07's own log exactly (48.3%). At the module's actual current
default (order 16), the share is even higher (55.2%). **Restating prominently, as the prompt asks**:
this is far above the audit's own synthetic reference points (0-25%, §4.4), and remains the standing
evidence base for the still-deferred rank-revealing-QR decision (README §6) — complexification
already halved `lstsq`'s absolute cost per the audit's own reasoning, but on this production
integrand, more than half of all solves still take that branch.

### 4.7 The three-Bessel oracles, `c8a1918` versus `HEAD`

All seven analytic oracles (`J000, J110, J220, J222, J231, Y000, Y022`), fixed triple
`(k,q,s) = (1.3, 1.7, 2.1)`, `max_x = 1e6` (chosen for this comparison's time budget — smaller than
`test_abserr_bounds_truth`'s `1e12`, but still exercising the full four-phase-group decomposition and
the complete accept/bisect machinery), `atol=1e-14, rtol=1e-10, chebyshev_order=12` in both versions,
through the real, current `quad_JJJ`/`quad_YJJ` (the pre-campaign core driven via the same
`"converged"`-key shim as §4.5):

| oracle | analytic | old value | old relerr | new value | new relerr | \|old - new\| |
|---|---|---|---|---|---|---|
| J000 | 0.169230 | 0.16923048 | 6.01e-7 | 0.16923048 | 6.01e-7 | 6.2e-15 |
| J110 | 0.00650886 | 0.00650884 | 3.11e-6 | 0.00650884 | 3.11e-6 | 6.7e-13 |
| J220 | -0.0842397 | -0.08423958 | 1.15e-6 | -0.08423958 | 1.15e-6 | 1.2e-12 |
| J222 | 0.0891252 | 0.08912505 | 1.10e-6 | 0.08912505 | 1.10e-6 | 8.8e-13 |
| J231 | -0.062165 | -0.06216491 | 1.11e-6 | -0.06216491 | 1.11e-6 | 6.4e-13 |
| Y000 | -0.114214 | -0.11421373 | 1.09e-7 | -0.11421373 | 1.09e-7 | 3.0e-12 |
| Y022 | 0.0783165 | 0.07831648 | 1.48e-7 | 0.07831648 | 1.48e-7 | 1.5e-13 |

**Every oracle agrees between the pre-campaign and post-campaign code to round-off or near-round-off
(6.2e-15 to 3.0e-12 absolute), and both versions sit at essentially the same relative error against
the analytic closed form (1e-7 to 3e-6) in every case.** This is the single most important
confirmation this document can offer: the campaign has not changed what the module actually
delivers on the production integrand family it exists to serve — it has changed how honestly the
module *reports* that delivered accuracy (§4.9 below quantifies that change), and, per §4.5, how much
it costs to get there. The residual gap to the analytic value (1e-7 to 1e-6, larger than either
version's own internal `abserr`) is the phase/modulus-spline construction floor this campaign
explicitly did not touch (README §6) — both before and after, confirming that floor, not the Levin
core, is what limits accuracy on this integrand family.

This reproduces, and extends to all seven oracles, the two-oracle spot check prompt 02's own log
performed (`J000`/`J110`, `0.00` and `3.5e-18` agreement) and prompt 03's own two-oracle check
(`J000`/`J110`, `6.5e-13`/`4.2e-13` agreement) — no prior log in this campaign checked all seven
together, or checked `c8a1918` directly against `HEAD` rather than against the immediately preceding
prompt.

### 4.8 `test_abserr_bounds_truth`, re-run to completion

Prompt 09's own `xfail` test, re-run standalone this session (`max_x=1e12`, the test's own
parameters, matching `ABSERR_BOUNDS_K/Q/S = 1.3, 1.7, 2.1`): **1 test, OK (expected failures=1)**,
19.2 s wall time. Every one of the seven printed diagnostics matches prompt 09's log to the displayed
precision:

```
(JJJ 0,0,0): true abserr=2.7502e-09, reported abserr=1.5331e-07, ratio=0.018, bounds=True
(JJJ 1,1,0): true abserr=3.0039e-10, reported abserr=2.9174e-10, ratio=1.03, bounds=False
(JJJ 2,2,0): true abserr=2.1574e-09, reported abserr=2.161e-10, ratio=9.98, bounds=False
(JJJ 2,2,2): true abserr=8.5133e-10, reported abserr=2.1656e-10, ratio=3.93, bounds=False
(JJJ 2,3,1): true abserr=2.0819e-09, reported abserr=1.8156e-10, ratio=11.47, bounds=False
(YJJ 0,0,0): true abserr=2.4629e-09, reported abserr=1.5335e-07, ratio=0.016, bounds=True
(YJJ 0,2,2): true abserr=1.5774e-09, reported abserr=2.1686e-10, ratio=7.27, bounds=False
```

2 of 7 bound the truth; 5 underbound it, by up to 11.5x — exactly prompt 09's own figures.
**Confirmed, not merely re-stated**, and confirmed fast (19.2 s, not the many minutes `test_JJJ`
needs) because it skips the 250-point diagnostic plotting loop `plot_and_compute_3Bessel` performs
— see §5.1 for the plotting-inclusive run.

### 4.9 Additional spot-checks performed

- **Complex-vs-real solve agreement (prompt 02).** Forced the real (`2N x 2N`) path via a subclass
  overriding `supports_complexified_solve` to `False`, on a problem in the direct-LU regime
  (`phase_span = 30*pi`, order 16): complex and real paths agree to `1.7e-18`, consistent with
  round-off and with prompt 02's own `~2e-21` figure on a different problem (same order of
  magnitude, not the same number, as expected).
- **C6 (`max_depth` on an all-fallback run).** A narrow Gaussian bump modulating a weak carrier
  (`atol=rtol=1e-30, depth_max=3`) gives `num_simple_regions == num_regions == 6`, `max_depth == 3`
  — confirmed non-zero on a run entirely handled by the Clenshaw-Curtis branch, matching prompt 01's
  own reproducer exactly in structure (different absolute region count, same qualitative result).
- **C11 (import cost).** `-X importtime` on the current module: **0.25-0.44s** total, with `grep` for
  `seaborn$`/`matplotlib$` in the importtime trace returning **no hits** (they are not imported at
  module scope). The same probe against the `c8a1918` module shows `seaborn`'s own cumulative cost
  alone at **1.11-1.40s** across three runs — consistent with both the audit's original 1.13-1.36s
  and this campaign's own re-measurement (1.20-2.00s). Confirms prompt 07's fix is real and durable.
- **Prompt 06's `grz_1000` field case.** Re-ran `_adaptive_levin_subregion_impl` directly on the
  exact region prompt 06's log cites (`x_span=(-1,0)`, `theta=1000*atan(x)`, order 16,
  `rtol=1e-10`): `p_ratios = [1.0, 1.3845e-10]`, mode 2 kept (ratio > rtol) — matches the log's
  cited value to the displayed precision. This is also now a shipped regression test
  (`test_mode_filter_gates_on_endpoint_not_mean`), so this is a second, independent confirmation of
  the same fact, not merely a rerun of the test.
- **CC weight nesting and exactness (prompt 03).** Already covered by shipped tests
  (`test_chebyshev_nesting`, `test_cc_weights_exact_for_polynomials`,
  `test_cc_weights_transcendental`); re-ran them standalone this session — all pass, same order of
  magnitude as prompt 03's log.

---

## 5. What is not verified

Being honest about the limits of this pass, in the same spirit as
[`docs/backport-modules-verification.md`](backport-modules-verification.md):

- **`ComputeTargets/QuadSourceIntegral.py` end to end, under Ray, against a real datastore.** This
  module's nine `adaptive_levin_sincos` call sites (prompt 08's order retuning, prompt 09's
  `abserr`/`converged` propagation into `metadata`) have never been exercised in this campaign
  against a real `GkSource`/`BackgroundModel` object graph — every prompt that touched this file
  confirmed only that it imports cleanly and, for prompt 08, that a hand-reproduced *shape* of one
  call site behaves as claimed. Closing this would need a real (if small) compute pipeline run,
  comparable in scope to the `backport-modules` campaign's own unresolved checks 4/7/8 — a
  multi-wavenumber run under a local Ray instance, likely tens of minutes to hours depending on the
  tolerance chosen, and it is out of proportion to a documentation-and-tests-only prompt.
- **The phase-spline accuracy floor.** No test in this repository can see
  `LiouvilleGreen/phase_spline.py`'s own construction error directly — §4.7/§4.8 above show its
  *effect* (the 1e-7 to 1e-6 residual gap between the Levin core's own accurate arithmetic and the
  analytic oracle), but nothing here or in the shipped suite measures `phase_spline.py` in isolation.
  This is explicitly out of scope for this campaign (README §6) and unchanged by it.
- **`test_JJJ`, `test_YJJ`, `test_YJJ_log_singularity` — the full, plotting-inclusive run.** Prompt
  09's own log flagged this as its one open verification gap (its "Deviations" section, "UNINTENDED
  DRIFT" heading, and `IMPLEMENTATION_STATE.md`'s own note to close it if time allows). Attempted in
  this session (`test_JJJ` and `test_YJJ` together, 30-minute budget) — see §5.1 for the outcome.
  `test_YJJ_log_singularity` (10 singularity-epsilon values x 2 Y-oracles, each with its own
  250-point plotting loop) was not attempted in this session on top of that: it is a materially
  larger amount of the same per-call cost `test_JJJ`/`test_YJJ` already stress, and this session's
  time budget was already spent confirming the plotting-inclusive `test_JJJ`/`test_YJJ` pair (or was
  exhausted attempting to — see §5.1). It should be run with the same generous (20-30 minute) budget
  by whoever next has the time; nothing in this pass gives any reason to expect it would fail (the
  underlying `quad_YJJ` call is exercised, at the same `max_x=1e12` scale, by
  `test_abserr_bounds_truth`, which does complete and does pass on its own numerical assertions).
- **`docs/adaptive-levin-benchmark/levin_bench/` harness, end to end.** Confirmed by grep (repeated
  in this pass) that it reads only pre-existing keys by name and that every new key this campaign
  added is additive; the harness itself was not re-run in this session (it was exercised once, for
  one oracle, in prompt 09's own log).
- **The `expectedFailure` left in this campaign.** `LiouvilleGreen/tests/test_3bessel_analytic.py::
  test_abserr_bounds_truth` (prompt 09) remains `xfail`, re-confirmed to still genuinely fail (not
  vacuously pass) in §4.8 above. `IMPLEMENTATION_STATE.md`'s
  `[09-abserr-does-not-bound-phase-spline-floor]` issue stays open, correctly, since nothing in this
  prompt's file list could close it (closing it needs `phase_spline.py` to grow an accuracy API,
  explicitly out of scope, README §6).
- **Cross-platform / cross-BLAS reproducibility of this session's own tightened test thresholds
  (§3.2).** All margins were chosen with 35x-1000x headroom above the measured true error on *this*
  machine's LAPACK, but "this session's own venv" is the only environment they were checked against,
  per this campaign's standing rule 10. A different BLAS backend is not expected to move these true
  errors by more than a few orders of magnitude at most (they are dominated by well-conditioned LU
  solves on small matrices, not ill-conditioned accumulation), but this is reasoning, not a second
  measurement on a second platform.

### 5.1 The `test_JJJ`/`test_YJJ` full run — outcome

**Attempted; not completed; terminated after 15.5 minutes of CPU time without finishing even the
first of the seven oracles the two tests together exercise.**

Launched `PYTHONPATH=. ./venv/bin/python -m unittest
LiouvilleGreen.tests.test_3bessel_analytic.Test3BesselAnalytic.test_JJJ
LiouvilleGreen.tests.test_3bessel_analytic.Test3BesselAnalytic.test_YJJ -v` in the background with a
30-minute (`timeout 1800`) budget, per this prompt's instruction to attempt it "up to ~10-30 min per
the prior log". Monitored directly (`ps`, and the redirected log file) at roughly 2-3 minute
intervals throughout. Progress, in full:

```
elapsed CPU time    log content
0:00                test_JJJ (...) ...
5:39                test_JJJ (...) ...   (no change)
7:07                test_JJJ (...) ...   (no change)
9:35                test_JJJ (...) ...   (no change)
10:26               test_JJJ (...) ...   (no change)
11:44               test_JJJ (...) ...   (no change)
12:28               test_JJJ (...) ...   (no change)
13:01               test_JJJ (...) ...   (no change)
13:42               test_JJJ (...) ...   (no change)
14:15               test_JJJ (...) ...   (no change)
14:50               test_JJJ (...) ...   (no change)
15:34               test_JJJ (...) ...   (no change)   <- terminated here
```

`unittest -v` only prints a test method's own result (`ok`/`FAIL`) after that method returns, and
`test_JJJ` itself only prints its `@@ (J...)` diagnostic line after each oracle's full
`plot_and_compute_3Bessel` call completes — so this trace means **the process had not finished even
the first of `test_JJJ`'s five `J`-oracles after 15.5 minutes of CPU time**, let alone the remaining
four, `test_YJJ`'s two more, or anything from `test_YJJ_log_singularity`. Each oracle's
`plot_and_compute_3Bessel` call builds three `bessel_phase` splines at `max_x=1e12` (`phase_atol=
1e-25, phase_rtol=5e-14` — deliberately tight) and then calls the evaluator (a full `quad_JJJ`/
`quad_YJJ`, itself four Levin phase-group solves) at **250** points on a log-spaced grid plus once
more for the final result — 251 full adaptive-Levin evaluations per oracle, each individually
comparable in cost to the single evaluations §4.5-§4.7 above measured taking seconds at this scale.

**Judged genuinely impractical within this prompt's time budget, not merely slow.** Extrapolating
linearly from "no oracle finished in 15.5 minutes" gives a lower bound on the order of **1.5-2+
hours** for `test_JJJ` and `test_YJJ` together (5 + 2 = 7 oracles), before `test_YJJ_log_singularity`
(a further 20 oracle-equivalent runs at near-singular triangle configurations, likely *more*
expensive per call since near-degenerate triangles are exactly where this integrand is hardest) is
even started. This is consistent with, and considerably worse than, prompt 09's own log estimate
("test_JJJ alone ran past 6 minutes without finishing its first oracle... a ~30-60 minute test run"
was its own suggested budget for `test_JJJ`/`test_YJJ`/`test_YJJ_log_singularity` together) —
prompt 09 under-estimated the true cost, and this session's attempt corrects that estimate rather
than repeating it. The pre-campaign code was not tried against these particular tests (they did not
exist before prompt 09 added the plotting-loop `.value` unpacking this campaign's own commits
touch), so there is no faster baseline to fall back on here.

**What is, and is not, covered by this gap.** `quad_JJJ`/`quad_YJJ`'s actual numerical output at
this exact scale (`max_x=1e12`, the same seven oracles, the same triple `(1.3, 1.7, 2.1)`) *is*
independently confirmed in this session — see §4.8, which reuses the identical evaluators and
oracles at the identical `max_x` and completes in 19.2 seconds because it skips the 250-point
plotting grid. What is *not* confirmed by this session is `test_JJJ`/`test_YJJ`'s own specific
assertions, which run at *randomised* `(k, q, s)` on every invocation (`uniform(0.1, 5.0)` per
component, drawn fresh each call) rather than the fixed triple, and which additionally check the
non-triangle-inequality branch's absolute-tolerance path — a case §4.8's fixed-triple run does not
exercise. Whoever next has a genuine 1.5-2+ hour budget should run these three tests to completion;
nothing found in this session gives any reason to expect a failure (the shared evaluator machinery
is confirmed correct and the shared cost driver — many full adaptive-Levin solves at `max_x=1e12`
per test — is a performance property of the *test*, not of the module this campaign changed), but
that expectation is reasoning, not the second measurement it would take to promote it to a
confirmed fact.
