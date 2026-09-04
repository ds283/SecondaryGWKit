# Prompt 08 — Spectral order and vectorised sampling

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit sections:** §4.3, §4.5; §5 recommendations 13 and 14
**Depends on:** prompt 04 (eq. (151) contains a `max(G₁, k²)` term, so the order feeds the reported
floor) and prompt 03 (the sampling site is only final after the restructure)
**Files you may touch:** `AdaptiveLevin/levin_quadrature.py`, `ComputeTargets/QuadSourceIntegral.py`,
`AdaptiveLevin/tests/test_levin_quadrature.py`, plus the log and the status board.

---

## Character of this commit

Two tuning changes, both requiring **measurement before action**. The audit is explicit that §4.5 is
"the weakest-supported item in §5" — a ±2×, problem-dependent effect on four problems — and that it
"should be re-measured on the real three-Bessel integrands before being treated as settled".

**You are the re-measurement.** If the numbers do not support the change, do not make it. A log that
says "measured, does not pay, left at 12" is a successful outcome for this prompt.

---

## Item 1 — Vectorised sampling (recommendation 14)

`f` is sampled in a Python loop, `f_Cheb = np.hstack([[func(x) for x in grid] for func in f])`
(`levin_quadrature.py:645`), and `θ′` likewise (`:474` or `:478`).

**Measured cost** (audit §4.1, per subregion evaluation, µs):

| component | N=12 | N=24 | N=32 |
|---|---|---|---|
| `f` sampling, Python loop (2N calls) | 11.7 | 20.6 | 26.8 |
| `θ′` sampling, Python loop (N calls) | 9.2 | 18.6 | 23.7 |

Together that is **37% of a 56 µs subregion evaluation at N = 12** — more than the linear solve.
Audit §4.3: `f` sampling drops from **11.7 µs to 2.5 µs** at N = 12 when the callables accept arrays,
and stays ~2.7 µs at N = 64 against 50.5 µs for the loop — an 18× saving at high order.

**The catch:** in production the integrand is a stack of spline evaluations. Check whether the real
callables vectorise before assuming the saving is available:

- `three_bessel_integrals.Levin_f` (`:190-193`) is `x^{3/2} · m_μ(kx) · m_ν(qx) · m_σ(sx)` where each
  `m` is a modulus spline. `np.pow` and `np.exp` vectorise; whether `phase_spline`'s modulus callable
  does is a question you must answer by reading it, not by assuming.
- `three_bessel_integrals._phase_group`'s `theta_deriv` calls `phase.theta_deriv(m*x,
  log_derivative=True)` — same question.
- `QuadSourceIntegral.py`'s `Levin_f` and phase functions — same question.

**Implement it as opt-in or by one-time detection at entry**, per the prior review's §7.6. Detection
is friendlier (callers get the speedup for free) and riskier (a callable that *accepts* an array but
returns something wrong, e.g. by broadcasting a scalar, produces silent garbage). If you detect,
**validate the result** — right shape, right dtype, finite, and agreeing with a scalar call at one
point — and fall back to the loop otherwise. Do the validation once per call, not once per region.

Record which you chose and why. If detection: show the validation. If opt-in: say what a caller has
to do, and add the flag to the docstring.

**Whether the production callables actually vectorise is the deliverable here**, more than the flag
itself. If they do not, the honest outcome is "the mechanism is in place and unused; making
`phase_spline` vectorise is follow-up work" — record that, with what you found.

## Item 2 — Default spectral order (recommendation 13)

`DEFAULT_LEVIN_CHEBSHEV_ORDER = 12` (`:79`). The audit recommends 16, keeping the
`_LEVIN_MINIMUM_ALLOWED_ORDER = 8` floor, and documenting that 12–32 is the useful band.

**The audit's measurement** (§4.5, wall time on the real module, ms):

| problem | k=8 | k=12 | k=16 | k=24 | k=32 | k=48 |
|---|---|---|---|---|---|---|
| sinc (1 region) | 0.24 | 0.26 | 0.27 | 0.31 | 0.35 | 0.47 |
| lorentz peak | 14.88 | **9.75** | 10.83 | 12.28 | 12.19 | 11.59 |
| gauss peak | 6.63 | 3.80 | 3.01 | 2.69 | **2.00** | 2.72 |
| log-x Bessel-like | 2.14 | 1.61 | 1.34 | **0.47** | 0.56 | 0.78 |

Regions fall monotonically with order (lorentz peak: 45 → 18), so on problems that subdivide, higher
orders do strictly less work. Order 8 is bad everywhere.

**A spot check run during this campaign's planning, on four differently-parameterised problems,
found higher order monotonically better all the way to 32** — no k=12 optimum anywhere, and 3–5×
faster at k=32 than at k=8. That does not contradict the audit (different problems), but it does
mean the k=12 optimum on `lorentz peak` is not robust across parameterisations, and it strengthens
rather than weakens the case for raising the default.

### Three constraints

**(a) The order feeds the reported error floor.** Prompt 04 implemented eq. (151), which contains
`max(G₁, k²)/G₀` with `k` the collocation order. Raising the default from 12 to 16 raises `k²` from
144 to 256 — a factor 1.78 on the round-off floor wherever `k² > G₁`, i.e. on regions with a small
rescaled phase derivative. **Measure the effect on reported `abserr`, not just on wall time.** A
change that is 10% faster and reports 1.8× more error is not obviously a win, and the trade must be
stated.

**(b) There is no odd-order constraint.** §4.5 says "odd orders are preferable if §2.2's nested
Clenshaw–Curtis fallback is adopted (nesting needs `N−1` even)". **That is wrong** — nesting holds
for every `N ≥ 2`, verified exactly at N = 12, 13, 16, 17, 25, 33 (README §2.3a, and prompt 03
should have a test asserting it). Do not let a phantom constraint push you to an odd order.

**(c) The SVD-failure recovery steps down in twos to a floor of 8.** `three_bessel_integrals.py:24-25`
notes that 12 "leaves it two steps of headroom". 16 leaves four. Not a constraint, but worth stating.

### Re-measure before changing

Run the order sweep on:

- the four problems in `AdaptiveLevin/tests/`;
- **at least three three-Bessel oracles** from `LiouvilleGreen/tests/test_3bessel_analytic.py` —
  this is the audit's explicit request and the reason the recommendation is not yet settled;
- at least one `QuadSourceIntegral`-shaped problem if you can construct one cheaply.

Report wall time, region count, solve count, integrand-evaluation count, delivered error against the
closed form, and reported `abserr`, at k = 8, 12, 16, 24, 32.

**Then decide.** Raise to 16 if the evidence supports it; leave at 12 if not. Either way, document
the useful band and the fact that the optimum depends on how much the amplitude subdivides. Update
the comment at `:78-80`.

## Item 3 — `ComputeTargets/QuadSourceIntegral.py`'s hard-coded 64

> **Correction to the audit.** §4.5 ends: "`three_bessel_integrals.py`'s hard-coded 64 is worth
> re-tuning on its own integrands." **It is already 12.** `DEFAULT_3BESSEL_CHEBYSHEV_ORDER = 12` at
> `three_bessel_integrals.py:26`, retuned in commit `cc64ae4`, with the measurement recorded in the
> comment at `:12-25`: identical relative error at orders 12 through 64 against the analytic oracles,
> against a 3–6.6× runtime rise. The audit's sentence is stale.
>
> **The file that still hard-codes 64 is `ComputeTargets/QuadSourceIntegral.py:29`**
> (`CHEBYSHEV_ORDER = 64`), used at nine `adaptive_levin_sincos` call sites (`:479`–`:611`, `:1038`).

Retune it, using the same method the `three_bessel` comment records: sweep the order against a fixed
reference and find where the delivered accuracy stops improving.

**This is harder than the three-Bessel case** because `QuadSourceIntegral` has no analytic oracle.
Options, in order of preference:

1. If a closed form or an independent reference exists for any configuration, use it.
2. Otherwise use self-consistency: the value at order 64 against the value at each lower order, with
   the reported `abserr` as the acceptance criterion. This is legitimate for *choosing an order* and
   is **not** evidence of absolute accuracy — say so explicitly, and note that the audit's §7 caveat
   about same-core references applies.
3. If neither is tractable in reasonable time, **do not change it**. Leave 64, record what you tried
   and what it would take, and open a §3 issue. A wrong order here degrades a production pipeline
   that the campaign cannot easily test.

If the `three_bessel` finding transfers — that accuracy is set by the phase and modulus splines, not
the spectral order — then 64 is costing several times the necessary work for nothing, exactly as it
was in `three_bessel_integrals.py`. That is the hypothesis; test it, do not assume it.

**Note the interaction:** if you change `DEFAULT_LEVIN_CHEBSHEV_ORDER` in item 2 *and*
`CHEBYSHEV_ORDER` here, `QuadSourceIntegral.py` passes its order explicitly at all nine sites, so it
is unaffected by the default. They are independent decisions. Make them separately.

---

## Do not

- Do not change `_LEVIN_MINIMUM_ALLOWED_ORDER`. The step-down recovery depends on it.
- Do not change `DEFAULT_3BESSEL_CHEBYSHEV_ORDER`. It was measured and retuned in `cc64ae4`; there is
  no evidence to overturn it, and its comment records the evidence for 12.
- Do not introduce an odd-order constraint.
- Do not change any error-estimate logic. If the order sweep suggests eq. (151)'s `k²` term is badly
  calibrated, that is an observation for the log and a §3 issue, not a change here.
- Do not change the numerics of prompts 02–06.

---

## Verification

1. `AdaptiveLevin/tests/` passes at whatever default you settle on.
2. **The full order sweep table**, all problems, all metrics listed in item 2. This is the commit's
   primary artefact.
3. **The effect on reported `abserr`** of any order change, separately from the effect on wall time.
4. **Vectorised sampling: correctness first.** Values must be unchanged (bit-equal where the sampling
   is the only difference — `np.fromiter`/array evaluation of the same callable at the same points
   should give identical floats unless the callable itself branches on input type). Then the speedup,
   on both a synthetic vectorisable integrand and, if possible, a real one.
5. **The fallback path when a callable does not vectorise** — construct one and show it is caught,
   not silently broadcast into garbage.
6. **`QuadSourceIntegral` retuning**: the sweep, the criterion used, and an explicit statement of
   whether it establishes absolute accuracy (it probably does not — say so).
7. Three-Bessel oracles unchanged at the default order they actually pass (12), before and after —
   this commit must not move them.

---

## Finish

1. Write `prompts/levin-refactor/logs/08-order-and-sampling.md`. The order sweep table goes under
   *Numerical evidence* in full — it is the evidence for a recommendation the audit itself flagged as
   under-supported, and the next person to touch the order should not have to re-run it. Record
   explicitly: the order decision and its justification; the vectorisation approach (opt-in vs
   detection) and **whether the production callables actually vectorise**; and the
   `QuadSourceIntegral` decision including "left at 64" if that is where you land.
2. Update `IMPLEMENTATION_STATE.md`. If you left the default at 12 or left `CHEBYSHEV_ORDER` at 64,
   mark rec 13 as ⚠️ with a one-line reason, not ✅.
3. Commit in one commit. Suggested message (adjust heavily to what you actually did):

```
Retune the Levin spectral order and add vectorised sampling

Sampling the amplitude and the phase derivative in Python loops accounted for
37% of a subregion evaluation at the default order -- more than the linear
solve -- and 18x more than an array call at order 64. Callables that accept
arrays are now <detected/opted in> and sampled in one call, with a fallback
to the element-wise loop.

The default Chebyshev order is <raised to 16 / left at 12>. <Summarise the
measurement: which problems, what the sweep showed, and what it costs in
reported error through the k^2 term of the round-off floor.> The useful band
is 12-32; the optimum depends on how much the amplitude subdivides, since
regions fall monotonically with order while the cost per solve rises. Order 8
is worse than 12 on every problem measured and the minimum-allowed-order floor
stays where it is.

ComputeTargets/QuadSourceIntegral.py's hard-coded order 64 is <retuned to N /
left at 64 pending a reference>. <Give the evidence.> This is the same finding
that took three_bessel_integrals.py from 64 to 12 in cc64ae4, where the
delivered error turned out to be set by the accuracy of the phase and modulus
splines rather than by the spectral order.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```
