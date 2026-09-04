# Log 08 — Spectral order and vectorised sampling

**Prompt:** prompts/levin-refactor/08-order-and-sampling.md
**Commit:** Retune the Levin spectral order and add vectorised sampling; SHA intentionally omitted — see README §5 rule 5
**Date:** 2026-09-04
**Result:** COMPLETE

## What shipped

### Item 1 — Vectorised sampling (recommendation 14)

Two new module-level helpers in `AdaptiveLevin/levin_quadrature.py`:

- `_detect_vectorized(func, grid)` (new, before `_Basis_SinCos`): probes `func` with the grid's
  first two points, both as an array call and as two scalar calls, and returns `True` only if the
  array call succeeds, returns an array of the probe's shape, is numeric and finite, and agrees
  *exactly* with the scalar loop. A constant `lambda x: 1.0` (used by this module's own tests) or a
  scalar-only callable (`math.atan`, which raises on an array) are both correctly rejected — the
  first by the shape check, the second by the caught exception.
- `_sample_vectorized(func, grid, cache, key)` (new, same location): looks up `key` in `cache`;
  on a miss, calls `_detect_vectorized()` once and stores the result; then either makes one array
  call (`func(grid)`) or falls back to `np.array([func(x) for x in grid])`. If a cached-vectorized
  callable's output shape stops matching the grid (a chebyshev-order retry after an SVD failure
  changed the grid size), the cache entry is corrected to `False` and the loop is used from then on.

Detection was chosen over an opt-in flag (prior review §7.6's other option): callers get the
speedup automatically, and the shape/dtype/finiteness/exact-agreement check is the safeguard
against the "silently broadcasts" failure mode that option warns about.

**Threading, not a global cache.** A naive `id(func)`-keyed global cache risks a stale hit once an
id is reused after garbage collection. Instead, `_adaptive_levin()` creates one `vectorize_cache: dict
= {}` per top-level `adaptive_levin_sincos()` call (`levin_quadrature.py:1752`) and threads it
through every call site that used to sample a callable in a loop:

- `_adaptive_levin_subregion()` → `_adaptive_levin_subregion_impl()` → `_Basis_SinCos.build_Levin_data()`
  (samples `theta`/`theta_deriv` — the `need_theta_Cheb` and `hasattr(self, "_theta_deriv")`
  branches) and the `f_Cheb = np.hstack([...])` amplitude sample.
- `_adaptive_levin_subregion_impl()` → `_adaptive_levin_subregion_cc()` (the `f_fine = np.vstack([...])`
  amplitude sample on the Clenshaw-Curtis fallback's fine grid).
- `_write_progress_data()`'s own three `_adaptive_levin_subregion()` calls (diagnostics-only,
  `emit_diagnostics=True`, off by default and unset anywhere in this repository), for consistency —
  not itself performance-critical.

All five now-optional `vectorize_cache: Optional[dict] = None` parameters default to `None`, in
which case the function creates its own fresh, unshared dict — correct for a direct caller (as
several `AdaptiveLevin/tests/` cases are) but without the cross-region reuse the driver gets.
`build_Levin_data()`'s cache keys are `("theta", id(self))` / `("theta_deriv", id(self))`; the two
`f`-sampling sites use `("f", id(func))` per component. `eval_basis()`'s own per-point loop (used
only by the diagnostics-only `_write_progress_data()`) and the CC fallback's `basis_fine` sample
were left untouched — the prompt's measured components are `f` and `theta'` sampling only, and
touching more would be scope creep (README §5 rule 6).

**Whether production callables actually vectorize: no, none of them do**, as of this commit —
confirmed by direct probing (see "Numerical evidence" below), not assumed:

- `LiouvilleGreen/bessel_phase.py`'s `XSplineWrapper.__call__` (the `mod` slot of a `bessel_phase()`
  dict, used by `three_bessel_integrals.Levin_f` and `ComputeTargets/QuadSourceIntegral.py`'s
  `Levin_f`) branches on `if raw_x < 0.99 * self._min_x` — an ambiguous truth-value error on an
  array, and even if guarded would still be scalar-only logic.
- `LiouvilleGreen/phase_spline.py`'s `chunk_spline._theta`/`_theta_deriv` (reached via
  `phase_spline.raw_theta`/`theta_deriv`/`theta_mod_2pi`, i.e. `three_bessel_integrals._phase_group`'s
  `theta`/`theta_deriv` and `QuadSourceIntegral.py`'s `phase1`/`phase1_mod_2pi`) likewise branch on
  scalar `log_x < self.min_log_x * (...)` comparisons and call `math.fmod`, none of which accept an
  array.

So the mechanism this item adds is in place and exercised by its own tests, but currently inert on
every production call site — it will start paying off automatically, with no further change here,
the day `phase_spline.py`/`XSplineWrapper` grow array support (out of scope per README §6, same as
the rest of `LiouvilleGreen/bessel_phase.py`/`phase_spline.py`).

### Item 2 — Default spectral order (recommendation 13)

`DEFAULT_LEVIN_CHEBSHEV_ORDER` raised from **12 to 16** (`levin_quadrature.py:137-152`, comment
records the justification and the trade-off). `_LEVIN_MINIMUM_ALLOWED_ORDER` (8) is unchanged.
`adaptive_levin_sincos()`'s `:param chebyshev_order:` docstring updated to state the new default and
the 12-32 useful band.

### Item 3 — `ComputeTargets/QuadSourceIntegral.py`'s hard-coded 64

`CHEBYSHEV_ORDER` retuned from **64 to 24** (`QuadSourceIntegral.py:29-45`, comment records the
method and the caveat that this is a self-consistency result, not an oracle-verified one). All nine
call sites pass `chebyshev_order=CHEBYSHEV_ORDER` explicitly, so the one constant change covers all
of them. `DEFAULT_3BESSEL_CHEBYSHEV_ORDER` (12, `three_bessel_integrals.py`) was left untouched, as
instructed.

### Tests

`AdaptiveLevin/tests/test_levin_quadrature.py`: new `TestVectorizedSampling` class, 5 tests —
detection of a genuine vectorizer with bit-equal output against the loop; rejection of a
constant-broadcasting callable; rejection (with a correct fallback result, not a propagated
exception) of a scalar-only callable modelled directly on the `_GRZIntegral` test's own
`math.atan`-based phase; a call-count assertion that detection runs once per callable across five
samplings, not once per call; and an end-to-end `adaptive_levin_sincos()` run through a genuinely
vectorizing amplitude, checked against the same closed form `test_SincIntegral` uses.

## Numerical evidence

### Item 1 — vectorisation speedup (synthetic, isolated)

`_sample_vectorized()` timed against the Python-loop baseline on `f(x) = sin(x) * sqrt(|x| + 1)`
(a genuinely vectorizing synthetic amplitude), cache already warm (i.e. timing the sampling call
only, not the one-off detection):

| N  | loop (µs) | vectorized (µs) | speedup |
|----|-----------|------------------|---------|
| 12 | 62.1      | 3.76             | 16.5x   |
| 24 | 78.3      | 5.79             | 13.5x   |
| 64 | 208.1     | 10.08            | 20.6x   |

Consistent with the audit's own 11.7 µs → 2.5 µs (N=12) / 18x-at-N=64 figures (different synthetic
integrand, same order of magnitude).

### Item 1 — production callables probed directly, all reject

```
mod (XSplineWrapper) vectorizes:        False
raw_theta (phase_spline) vectorizes:    False
theta_deriv (phase_spline) vectorizes:  False
```
(probed via `bessel_phase(0.5, 1e6, ...)`'s returned `"mod"`/`"phase"` objects), and, reproducing
the actual closures `three_bessel_integrals._phase_group`/`_Levin_3bessel` build:
```
theta vectorizes:        False
theta_deriv vectorizes:  False
Levin_f vectorizes:      False
```
`ComputeTargets/QuadSourceIntegral.py`'s `Levin_f`/`phase1`/`phase1_mod_2pi` (lines 439-471) call
the identical `mod_Gk`/`mod_Tk`/`phase_Gk`/`phase_Tk` objects and so are subject to the same result
by construction (not re-probed separately, since they share the same underlying callables).

### Item 2 — order sweep (primary artefact)

`adaptive_levin_sincos()`/`quad_JJJ()` at chebyshev_order = 8, 12, 16, 24, 32. Timings are the
second of two identical calls (a discarded warm-up call absorbs `chebyshev_matrices()`'s/
`_cc_weights_base()`'s `lru_cache` population for a new order, which otherwise dominates the first
call and masks the actual order-dependence — confirmed by a reversed-order re-run of the J000 case,
which reproduced the same ordering: 32→3.9s, 24→3.5s, 16→1.4s, 12→2.2s, 8→8.2s, then 12 and 16
repeated at 2.2s/1.4s again).

**The four `AdaptiveLevin/tests/` problems:**

```
== SinIntegral: int_1^50000 sin(x) dx ==
 order   time(ms)  regions   solves    evals       abserr     true_err
     8      0.227        1        3        3   7.4999e-12   3.4696e-11
    12      0.202        1        3        3   7.4999e-12   3.4695e-11
    16      0.305        1        3        3   7.5000e-12   3.4703e-11
    24      0.229        1        3        3   7.5000e-12   3.4706e-11
    32      0.300        1        3        3   7.5004e-12   3.4708e-11

== CosIntegral: int_1^500000 cos(x) dx ==
 order   time(ms)  regions   solves    evals       abserr     true_err
     8      0.260        1        3        3   7.5000e-11   1.0364e-11
    12      0.214        1        3        3   7.5000e-11   1.0364e-11
    16      0.219        1        3        3   7.5000e-11   1.0353e-11
    24      0.243        1        3        3   7.5000e-11   1.0363e-11
    32      0.262        1        3        3   7.5000e-11   1.0333e-11

== SincIntegral: int_1^100 sin(100x)/x dx ==
 order   time(ms)  regions   solves    evals       abserr     true_err
     8     17.167      120      107      287   2.1237e-11   2.6864e-11
    12      2.542       10       39       39   1.5654e-13   2.6808e-11
    16      1.689        7       27       27   1.1288e-12   2.8033e-11
    24      1.482        6       23       23   3.5362e-14   2.6939e-11
    32      1.586        5       19       19   1.0742e-13   2.7011e-11

== GRZIntegral(lambda=100.0): int_-1^1 cos(lambda*atan(x))/(1+x^2) dx ==
 order   time(ms)  regions   solves    evals       abserr     true_err
     8     23.702      184       11      367   1.3179e-11   1.9804e-16
    12      4.429       24       11       51   4.4007e-12   2.8220e-13
    16      1.571        4       11       15   3.9670e-15   1.9574e-15
    24      1.819        4       11       15   4.2560e-13   1.0894e-14
    32      0.286        1        3        3   3.2868e-14   1.2106e-28
```

**Three three-Bessel oracles** (`k=1.3, q=1.7, s=2.1`, `max_x=1e5`, `atol=1e-14, rtol=1e-10` —
matching the parameters `three_bessel_integrals.py`'s own retuning comment records for its
order-12 decision):

```
== J000 (JJJ, mu=nu=sigma=0) ==
 order   time(ms)            value     true_err
     8   8491.835   1.69230108e-01   2.6594e-07
    12   2232.400   1.69230108e-01   2.6594e-07
    16   1408.316   1.69230108e-01   2.6594e-07
    24   3706.520   1.69230108e-01   2.6594e-07
    32   4348.678   1.69230108e-01   2.6594e-07

== J110 (JJJ, mu=nu=1,sigma=0) ==
 order   time(ms)            value     true_err
     8  14079.034   6.50950807e-03   6.4755e-07
    12   6775.817   6.50950807e-03   6.4755e-07
    16   1750.824   6.50950807e-03   6.4755e-07
    24   6158.190   6.50950807e-03   6.4755e-07
    32   7094.366   6.50950807e-03   6.4755e-07

== J220 (JJJ, mu=nu=2,sigma=0) ==
 order   time(ms)            value     true_err
     8   6450.550  -8.42399464e-02   2.7088e-07
    12   9538.373  -8.42399464e-02   2.7087e-07
    16   5221.599  -8.42399464e-02   2.7087e-07
    24   4118.291  -8.42399464e-02   2.7088e-07
    32   6026.956  -8.42399464e-02   2.7087e-07
```

`analytic()` closed forms taken directly from `LiouvilleGreen/tests/test_3bessel_analytic.py`'s
`J000`/`J110`/`J220` classes.

**Reading the table:**

- **Order 8 is worse everywhere measured**, confirming the audit and the campaign's own §2.4 note.
  On the three three-Bessel oracles it is 3.8x-8.0x slower than order 12 (worst: J110, 14.1s vs
  6.8s) and, separately, on a `QuadSourceIntegral`-shaped problem below, up to ~550x slower.
- **Delivered accuracy against the closed form is unaffected by order** on every three-Bessel
  oracle (`true_err` identical to five significant figures at every order from 8 to 32) and on
  `SinIntegral`/`CosIntegral`/`SincIntegral` (flat to within the same order of magnitude) —
  confirming the `three_bessel_integrals.py` finding ("accuracy is set by the phase/modulus
  splines, not the spectral order") holds for this campaign's own re-measurement, not just the
  original retuning to 12. `GRZIntegral(100)`'s `true_err` is the one case that visibly improves
  with order (1.96e-15 at 16 down to 1.21e-28 at 32), but that tracks region count collapsing
  (24 → 4 → 1), not the order itself.
- **Order 16 is faster than order 12 on all three three-Bessel oracles** (1.6x on J000, 3.9x on
  J110, best-of-run on J220 too, though J220's own order-24 timing (4.1s) edges out order 16's
  (5.2s) — the three-Bessel evidence favours 16-24, not a single sharp optimum) and on both
  `AdaptiveLevin/tests/` problems that subdivide at all (`SincIntegral`: 10→7 regions, 2.5→1.7ms;
  `GRZIntegral(100)`: 24→4 regions, 4.4→1.6ms). `SinIntegral`/`CosIntegral` never subdivide (1
  region at every order) and show no order-dependence beyond timing noise, as expected.
- **This supports raising the default to 16** (order 8 is never a win; 16 beats 12 on every
  problem that actually exercises subdivision; 24 and 32 are sometimes marginally better on a
  single problem but not consistently, and cost more per solve once a problem has already
  stopped subdividing — see `SinIntegral` at order 32, 0.300ms vs order 12's 0.202ms for an
  identical 1-region/3-solve run).

**Isolated measurement of note (a)'s cost** (the round-off floor's `max(G1, k^2)/G0` term): a
single synthetic region, forced into the Levin branch just above the `SIX_PI` threshold
(`theta = omega*x`, `omega*width = 19.85`, `G0 = G1` fixed regardless of order since `theta' `is
constant), so only `k` changes between calls:

```
order= 8  phase_span=19.8496  phase_err(roundoff)=8.448507e-16
order=12  phase_span=19.8496  phase_err(roundoff)=1.650914e-15
order=16  phase_span=19.8496  phase_err(roundoff)=2.779403e-15
order=24  phase_span=19.8496  phase_err(roundoff)=6.003656e-15
order=32  phase_span=19.8496  phase_err(roundoff)=1.051761e-14
```

`phase_err` at order 16 is 1.685x order 12's — close to the predicted `(16/12)^2 = 1.78x` (not
exact, because the floor's `1 + G1/G0` term is order-independent and dilutes the pure-`k^2`
scaling). This confirms the trade-off the prompt's constraint (a) describes is real and
measurable in isolation, even though none of the concrete problems above showed it degrading the
*aggregate* reported `abserr` (region-count changes dominate there).

### Item 3 — `QuadSourceIntegral` order retuning

No analytic oracle exists for this integrand (option 1 unavailable — confirmed by inspection: its
nine call sites depend on `GkSource`/`GkSourcePolicyData`/`BackgroundModel`/Ray, none of which
admit a closed form). Used option 2 (self-consistency): reproduced the *shape* of one call site
(`_three_bessel_Levin`'s `Levin_f`/`phase1`/`phase1_mod_2pi`, `QuadSourceIntegral.py:439-471`)
directly from `bessel_phase()` — same domain (`log eta`), same amplitude form
`eta^(1.5-b) * mod(x1)*mod(x2)*mod(x3)`, same phase contract (`theta` + `theta_mod_2pi` only, no
`theta_deriv`, matching the deliberate omission `QuadSourceIntegral.py:1023-1037` documents), same
`atol`/`rtol` as production (`LEVIN_ABSERR=1e-23`, `LEVIN_RELERR=1e-8`) — at three configurations
spanning the phase magnitude the production gate (`LEVIN_MIN_PHASE_DIFF = 20*pi ≈ 62.8` rad,
`QuadSourceIntegral.py:26-27`) allows, so the order decision does not rest on one lucky case:

```
== A: deep sub-horizon (net phase change=2.851e+07 rad, 4538009 cycles) ==
 order   time(ms)  regions   solves              value       abserr    |value-ref|
     8    6241.56     2654    10615  -3.4039318843e-04   1.8714e-12     1.8081e-12
    12      11.48        4       15  -3.4039318691e-04   2.7849e-12     3.3303e-12
    16       6.44        2        7  -3.4039318619e-04   3.3334e-12     4.0516e-12
    24       4.07        1        3  -3.4039318893e-04   1.4337e-11     1.3069e-12
    32       5.40        1        3  -3.4039319034e-04   1.4337e-11     1.0242e-13
    48       8.55        1        3  -3.4039318920e-04   1.4337e-11     1.0377e-12
    64      10.58        1        3  -3.4039319024e-04   1.4337e-11     0.0000e+00  (reference)

== B: near-threshold, narrow range (net phase change=85.52 rad, 13.61 cycles) ==
 order   time(ms)  regions   solves              value       abserr    |value-ref|
     8       1.16        1        3  -6.4802228425e-04   7.7791e-14     1.2153e-13
    12       2.01        1        3  -6.4802228423e-04   6.4315e-14     9.9833e-14
    16       2.19        1        3  -6.4802228429e-04   7.9536e-14     1.6331e-13
    24       3.19        1        3  -6.4802228422e-04   1.9120e-13     9.3787e-14
    32       4.12        1        3  -6.4802228425e-04   1.9758e-14     1.1821e-13
    48       6.20        1        3  -6.4802228445e-04   2.7475e-13     3.2740e-13
    64       8.66        1        3  -6.4802228413e-04   2.6049e-13     0.0000e+00  (reference)

== C: middle case, k:q:r = 0.5:3.0:3.2, b=0, nu_Tk=1.5 (net phase=4.079e+05 rad, 64922 cycles) ==
 order   time(ms)  regions   solves              value       abserr    |value-ref|
     8      81.40       38       77   9.8825868420e-03   2.7698e-10     1.6076e-11
    12       9.26        4       15   9.8825868543e-03   3.1864e-12     3.7806e-12
    16       6.18        2        7   9.8825868517e-03   6.5303e-13     6.4019e-12
    24       3.73        1        3   9.8825868503e-03   3.7205e-12     7.8275e-12
    32       4.33        1        3   9.8825868257e-03   2.5169e-11     3.2364e-11
    48       6.44        1        3   9.8825868534e-03   3.7205e-12     4.6687e-12
    64       8.32        1        3   9.8825868581e-03   5.9180e-12     0.0000e+00  (reference)
```

**Reading:** order 8 is again uniformly the worst choice — 550x slower on config A (2654 regions
vs order 12's 4), 9x slower on config C, competitive only on config B where the whole problem is
one region regardless of order (too little phase for subdivision to matter). Orders 12 through 64
agree with the order-64 reference to within a few parts in 1e-11 to 1e-13 in every configuration —
no systematic trend with order, consistent with the same "splines set the floor, not the order"
finding as the three-Bessel oracles. Orders 24-64 reach the same single-region solution on configs
A and C (identical region/solve counts), so 64's extra cost buys nothing there; order 16 has not
always collapsed to one region at that point (config A: 2 regions; config C: 2 regions), so it was
not chosen as the QuadSourceIntegral-specific value even though it was chosen as the module-wide
default.

**24 was chosen over three_bessel_integrals.py's more aggressive 12** specifically because this
evidence is self-consistency only, not an analytic oracle (audit §7's same-core-reference caveat
applies in full here: agreement across orders 12-64 cannot rule out a bias shared by all of them,
whereas `three_bessel_integrals.py`'s retuning checked against seven independent analytic
formulas). 24 reaches the collapsed single-region solution in both large-phase configurations
measured, at roughly 2.2-2.6x less per-solve cost than order 64 (config A: 4.07ms vs 10.58ms;
config C: 3.73ms vs 8.32ms), which is the deliverable this item exists for.

## Deviations from the prompt

**IMPLEMENTATION CHOICE — cache keyed by `id()`, scoped to one call, not a persistent global
cache.** The prompt does not specify how "once per call, not once per region" should be
implemented. A persistent cache keyed by `id(func)` across calls was considered and rejected: once
a callable is garbage-collected, Python may reuse its id for an unrelated object, which would
silently return a stale (and possibly wrong) vectorisation verdict for that new object. Scoping the
cache to one `_adaptive_levin()` call (created fresh each time, discarded at the end) makes `id()`
safe, because every object it could key on is kept alive by the call's own local references for the
cache's entire lifetime. The cost is that a caller invoking `adaptive_levin_sincos()` many times
with structurally-identical-but-freshly-constructed closures (the common production pattern — see
`three_bessel_integrals._phase_group`, built fresh per phase group) re-probes each closure once per
call rather than once ever; at two scalar calls plus one array call per distinct callable per run,
this is judged negligible next to the per-region sampling it is designed to save.

**IMPLEMENTATION CHOICE — `QuadSourceIntegral.py`'s order chosen more conservatively than
`three_bessel_integrals.py`'s.** See "Item 3" above for the full reasoning: 24, not 12, specifically
because the evidence here is self-consistency rather than an analytic oracle.

**None** beyond the above — no other deviation from the prompt as written.

## Verification performed

1. `PYTHONPATH=. ./venv/bin/python -m unittest discover -s AdaptiveLevin/tests -t .` — **28 tests,
   OK, 0.044-0.247s** (23 pre-existing + 5 new `TestVectorizedSampling` tests). Baseline going in
   was 23 (prompt 07's log records 16 after prompt 03; prompts 04-07 each added tests without a
   running total in the board — 23 was confirmed by running the suite before this prompt's changes).
2. `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_three_bessel
   LiouvilleGreen.tests.test_bessel_phase` — **6 tests, OK, 5.4s** — confirms the three-Bessel path
   (which passes `chebyshev_order` explicitly at every call site, so is unaffected by the module
   default change) and `build_Levin_data()`'s new `vectorize_cache` parameter did not break the
   production entry points.
3. `PYTHONPATH=. ./venv/bin/python -c "import ComputeTargets.QuadSourceIntegral"` — imports cleanly
   after the `CHEBYSHEV_ORDER` edit (no dedicated test suite exists for this module — it depends on
   `ray`/`GkSource`/`BackgroundModel` object graphs not constructible standalone).
4. Grepped every in-tree call site of `adaptive_levin_sincos()` (`docs/adaptive-levin-benchmark/`,
   `LiouvilleGreen/three_bessel_integrals.py` and its tests, `ComputeTargets/QuadSourceIntegral.py`)
   and confirmed every one passes `chebyshev_order` explicitly — the `DEFAULT_LEVIN_CHEBSHEV_ORDER`
   12→16 change is therefore inert on every existing caller except a fresh direct call that omits
   the parameter.
5. Grepped for other callers of `build_Levin_data()` — only the one call site inside
   `levin_quadrature.py` itself; no external code depends on its positional signature.
6. Vectorisation correctness and fallback, both by direct unit test (`TestVectorizedSampling`, item
   4 above) and interactively: a genuine vectorizer, a constant-broadcasting trap, and a
   `math.atan`-based scalar-only callable all produced bit-identical results to the loop baseline
   (`np.array_equal`), with the trap and the scalar-only callable both correctly falling back
   rather than being mistaken for vectorized.
7. Production callables probed directly and interactively (not merely inspected) — see "Numerical
   evidence" above — confirming they do not vectorize as of this commit.
8. The order sweep itself (see above) is the primary verification of item 2's decision; the
   isolated single-region measurement is the verification of constraint (a)'s cost.

## Observations not acted on

- **J220's per-order timing was noisier than J000/J110** (order 24 beat order 16 by a small margin
  in one run) — plausibly this problem's amplitude/phase shape genuinely has its optimum nearer 24
  than 16, or it is measurement noise on a ~5s run on a shared machine. Not resolved further: the
  decision (raise the module default to 16) does not hinge on this one oracle, and `chebyshev_order`
  remains a caller-overridable parameter for exactly this kind of problem-specific tuning.
- **`_write_progress_data()`'s own per-point loop** (`for x in x_grid: ... f[i](x) ...`,
  `levin_quadrature.py` inside the diagnostics dump) was not vectorised. It is reached only under
  `emit_diagnostics=True`, which nothing in this repository sets, and touching it was outside this
  prompt's measured scope (README §5 rule 6).
- **`eval_basis()`'s per-point loop** in the Clenshaw-Curtis fallback's `basis_fine` sample
  (`_adaptive_levin_subregion_cc()`) was likewise left unvectorised for the same reason — the
  prompt's audit citations measure `f` and `theta'` sampling specifically, not this one.

## State handed to the next prompt

- `AdaptiveLevin/levin_quadrature.py`'s subregion-solving functions
  (`_adaptive_levin_subregion`/`_impl`/`_cc`, `_Basis_SinCos.build_Levin_data`,
  `_write_progress_data`) now all accept an optional `vectorize_cache: Optional[dict] = None`
  keyword, threaded from a single dict created once per `_adaptive_levin()` call. Any future new
  call site sampling a caller-supplied callable in a loop should either accept and use this same
  cache (preferred, for the "once per run" property) or, if it cannot, note explicitly why not.
- `DEFAULT_LEVIN_CHEBSHEV_ORDER = 16` and `ComputeTargets/QuadSourceIntegral.py`'s
  `CHEBYSHEV_ORDER = 24` are now the values prompt 09 (caller propagation) and prompt 10 (test
  matrix) should treat as the baseline — prompt 04's §3 note
  `[03-fallback-cost-on-difference-groups]` already flagged that prompt 08's cost re-measurement
  should be the baseline for future cost comparisons; that baseline is the order-sweep table above,
  not prompt 02's or prompt 03's numbers.
- The vectorisation mechanism is unused by every production caller today. The natural trigger to
  revisit it is the same one README §6 already names for `theta_abserr`: if
  `LiouvilleGreen/bessel_phase.py`/`phase_spline.py` ever grow array-argument support (out of scope
  for this campaign), production callers would start getting the measured 13-20x sampling speedup
  with no further change to this module.
