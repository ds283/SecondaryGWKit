# Performance characterisation of `AdaptiveLevin`

**Scope.** An empirical characterisation of the adaptive Levin quadrature in
`AdaptiveLevin/levin_quadrature.py` against brute-force adaptive quadrature, in
accuracy and cost, as the oscillation frequency grows. Written to support
extraction of the component into a standalone package.

**Bottom line.** The Levin core is frequency-independent to machine precision:
given an exactly-represented phase it returns 16 correct digits at
ω = 10¹², where adaptive Gauss–Kronrod is wrong by nine orders of magnitude. On
the repository's own three-Bessel application at 10⁵–10⁸ oscillations, `quad`
failed to converge in **0 of 12** cells within 60 seconds while Levin returned
eight correct digits in under one second. Everything that degrades at high
frequency degrades in the *phase representation*, not in the algorithm.

One genuine defect, and one packaging gap. The defect is local: the bespoke
argument reduction is both ~59× slower than `fmod` and *less accurate* than
simply calling `sin`, so it should be deleted. The gap is that the returned
convergence residual measures *resolution*, not total error — it runs 10¹⁰ below
the delivered error at ω = 10¹¹ — and the dictionary carries no aggregate error
and no conditioning term at all. This is not a flaw in Chen et al.'s algorithm,
whose authors identify the endpoint conditioning loss explicitly (§4.0); it is
an API that will mislead a caller who has not read the paper, which matters
precisely because the plan is to hand this component to another project.

---

## 1. What was measured

Two tiers, all against independent ground truth.

**Tier A — synthetic problems with closed-form oracles.** Five integrands
spanning the behaviours that matter, each with an analytic value verified
against 60-digit `mpmath` quadrature to ≤ 2×10⁻¹⁷:

| problem | integrand | phase | why it is here |
|---|---|---|---|
| `damped_sine` | e^{-x} sin(ωx) on [0,1] | linear | clean baseline |
| `sinc` | sin(ωx)/x on [1,100] | linear | wide span, θ_max = 99ω |
| `grz` | Gradshteyn–Ryzhik closed form | linear | independent oracle family |
| `chirp` | amplitude × sin(ωx²) | **nonlinear** | the case Levin exists for |
| `near_singular` | (x²+δ²)^{-1} sin(ωx) | linear | amplitude structure |

**Tier B — three-Bessel products**, the repository's actual application, using
the seven closed-form oracles already in `LiouvilleGreen/tests/`
(`J000`, `J110`, `J220`, `J222`, `J231`, `Y000`, `Y022`) via the
Liouville–Green phase representation.

**Baselines.** `scipy.integrate.quad` (adaptive Gauss–Kronrod — the brute-force
comparison requested) and, where the phase is linear so that it legitimately
applies, `quad`'s QAWO oscillatory rule. QAWO is included deliberately as the
*harder* baseline: it is the method a numerical analyst would actually reach for
on these integrands, and comparing only against Gauss–Kronrod would flatter
Levin.

Frequency ladder: ω = 10 → 10¹², twelve decades.

---

## 2. Accuracy and cost versus frequency

![Frequency ladder](figs/fig1_ladder.png)

Three facts, all read off the saved ladder table.

**Brute force does not merely slow down — it returns garbage.** Median
relative error by decade of ω:

| ω | 10³ | 10⁵ | 10⁶ | 10⁸ | 10¹⁰ | 10¹² |
|---|---|---|---|---|---|---|
| `quad` | 3.4×10⁻¹³ | 8.3×10⁻¹¹ | **2.3×10²** | 4.3×10³ | 1.7×10⁶ | 2.4×10⁸ |
| Levin | 3.6×10⁻¹⁴ | 7.0×10⁻¹⁴ | 7.7×10⁻¹² | 1.2×10⁻⁹ | 6.1×10⁻⁸ | 4.6×10⁻⁶ |

`quad` crosses from "converging" to "meaningless" between ω = 10⁵ and 10⁶: it
runs out of subdivision budget before it can resolve a single period, and the
returned number has no relationship to the integral. Levin loses roughly one
digit per decade of ω and still has five to six correct digits at ω = 10¹².

**Levin's cost is flat in frequency; brute force's is not.** Levin wall time is
essentially independent of ω — the region structure is set by the *amplitude*,
which does not change — whereas `quad` grows until it hits its subdivision
limit and then stops (returning a wrong answer cheaply, which is worse than
being slow).

**Credit where due: `quad` is wrong but honest.** Of 36 cells with relative
error > 10⁻³, **all 36** raised `IntegrationWarning`, and above ω ≈ 10⁵ its
self-reported error estimate stayed within about one order of magnitude of the
truth. A caller who checks warnings is never misled by `quad`. That is exactly
the property Levin lacks — see §4.

**Where Levin is not the right tool.** At ω = 10¹², on the three problems with
linear phase:

| problem | Levin | QAWO | `quad` |
|---|---|---|---|
| `damped_sine` | 1.2×10⁻⁵ | **0** (exact) | 1.0×10⁵ |
| `sinc` | 1.9×10⁻⁵ | **5.5×10⁻¹⁴** | 2.4×10⁸ |
| `near_singular` | 2.3×10⁻⁸ | **3.7×10⁻¹³** | 1.5×10⁶ |

QAWO beats Levin by seven to nine orders of magnitude wherever it applies,
because it never forms ωx in double precision at all. **Levin's defensible
territory is nonlinear phase** — `chirp`, and the Bessel products — where QAWO
cannot be used and brute force fails outright. On `chirp` at ω ≈ 10¹² Levin
delivers 4.6×10⁻⁶ against `quad`'s 1.5×10⁹, and there is no oscillatory rule to
fall back on. This is the honest case for the component, and it is a strong one;
it is just narrower than "Levin beats quad at high frequency".

---

## 3. Mechanism: the error is in the phase, not the algorithm

![Mechanism](figs/fig4_mechanism.png)

This is the most consequential experiment in the campaign. Fixing the integrand
and varying only *how the phase is evaluated*:

On **[0,1]**, where the endpoints and ωx are exactly representable in binary64:

| ω | exact phase (mpmath) | `sin(ωx)` direct | pre-reduced (`fmod`) | `range_reduce_mod_2pi` |
|---|---|---|---|---|
| 10⁶ | 3.2×10⁻¹⁶ | 2.8×10⁻¹⁵ | 7.7×10⁻¹² | 7.7×10⁻¹² |
| 10⁹ | 1.5×10⁻¹⁶ | 1.5×10⁻¹⁶ | 1.1×10⁻⁸ | 1.1×10⁻⁸ |
| 10¹² | 3.2×10⁻¹³ | 3.2×10⁻¹³ | 1.2×10⁻⁵ | 1.2×10⁻⁵ |

**The Levin algorithm is not the limiting factor at any frequency tested.** With
an exact phase it holds ~10⁻¹⁶ to ω = 10¹². The interior of a region depends
only on θ′; θ itself enters only at region *endpoints*. So the entire
high-frequency error budget is the rounding of two numbers.

**Pre-reducing the phase actively destroys accuracy.** Calling `sin(ωx)` and
letting libm perform its own (exact, infinite-precision) argument reduction
matches the mpmath phase to ~10⁻¹⁶ over most of the ladder. Pre-reducing in
double precision first — by either `fmod` or the repository's
`range_reduce_mod_2pi` — is **up to 7.9 orders of magnitude worse** (worst gap
at ω = 10⁹; 7.6 decades at ω = 10¹²), because it rounds the argument before libm
ever sees it. The reduction is solving a problem libm has already solved
correctly.

On **generic endpoints [0.07, 0.93]**, where ωx is not representable, all four
modes tie: θ is already inexact at the eps·θ level before any reduction, the
*absolute* error is pinned near 10⁻¹⁶ independent of ω, and since |I| ~ 1/ω from
oscillatory cancellation, the *relative* error grows linearly in ω. That is the
observed one-digit-per-decade decay.

This yields a usable a priori bound. Writing θ_max for the total phase
traversed, the delivered relative error never exceeded **0.25 · eps · θ_max** in
any of 35 high-frequency cells across all five problems. `eps·θ_max` is
therefore a safe estimate of the attainable accuracy, and is the natural
health-check predicate: a caller requesting `rtol` below it is asking for
something no double-precision phase can deliver.

Two further observations from the same table: the error is **independent of
Chebyshev order from 4 to 32** (confirming it is not collocation error), and
`range_reduce_mod_2pi` costs **10–26 µs per evaluation against 0.2–0.36 µs for
`fmod`** — a median **59× penalty** (range 45–99×) for negative accuracy
benefit.

---

## 4. The convergence test measures resolution, not total error

![Estimator reliability](figs/fig2_estimator.png)

**Read §4.0 first: this is not a defect in Chen et al.'s algorithm, nor an
implementation error.** An earlier draft of this report characterised it as
both. See §4.0 for the correction; the practical recommendation survives, its
justification changes.

### 4.0 What the paper says (correction to an earlier draft)

The implementation is faithful to arXiv:2211.13400v3 §5. The algorithm proper,
step 4 (p. 30), accepts an interval exactly when the parent-versus-bisected
difference falls below the tolerance, and accumulates the **parent** value —
which is what `levin_quadrature.py:831` does.

More importantly, the authors **explicitly anticipate the mechanism measured in
§3**. In the paragraph immediately following the algorithm (p. 30) they note
that the condition number of the oscillatory integral grows with the magnitude
of g, and that for the adaptive Levin method "the principal loss of accuracy
occurs when exponentials of large magnitude are evaluated" in their formula
(171) — the endpoint expression `p(b)exp(ig(b)) − p(a)exp(ig(a))`. They further
predict that because |I| typically shrinks as g grows, the *absolute* error
"often remains constant or even decays" with frequency.

That is precisely what §3 measured: absolute error pinned near 10⁻¹⁶
independent of ω, relative error therefore growing like ω. The paper's §4 error
analysis (their eq. 151) bounds the *discretization* error — the residual in the
Levin equation — and shows it independent of the magnitude of g′. It does not
claim to bound the floating-point conditioning loss at the endpoints, and the
authors say so in prose rather than folding it into the bound.

**So step 4 is a valid test of what it tests**: whether the interval is resolved
by the k-point Chebyshev discretization. It is a *resolution* criterion, and it
is sound. The 1.5×10¹⁰ ratio tabulated below compares it against a quantity it
was never intended to bound.

The engineering point that survives is narrower but still real, and it is about
packaging rather than numerics: **a library API that returns a resolution
residual, with no aggregate error and no conditioning term, invites the caller
to read it as an error bar.** For the in-repository use that is a documentation
matter. For the planned extraction and reuse it is worth fixing, because the
next caller will not have read p. 30.

### 4.1 What the returned dictionary provides

First, `adaptive_levin_sincos` returns **no aggregate error estimate at all**:
the return dict carries `value`, `num_regions`, `regions`, `evaluations`,
`max_depth` and diagnostics, but no `abserr`/`relerr`. A caller wanting an error
bar must walk `regions` and combine the per-region figures itself.

Second, the natural aggregate built from what *is* returned — the sum of
per-region `abserr` over accepted regions, which is what "internal estimate"
means below — is unreliable at high frequency. Comparing it against the true
error versus the oracles:

| ω | 10² | 10⁴ | 10⁶ | 10⁸ | 10¹⁰ | 10¹¹ |
|---|---|---|---|---|---|---|
| median true/reported | 0.26 | 1.0 | 1.9 | 2.0×10⁶ | 1.0×10³ | 8.0×10⁹ |
| worst | 0.42 | 146 | 3.5×10⁵ | 2.2×10⁷ | 2.1×10³ | **1.5×10¹⁰** |

Up to ω ≈ 10⁵ the resolution residual and the total error coincide, because
discretization dominates. Above that they separate, because conditioning
dominates and the residual does not track it. Concretely, on `damped_sine` at
ω = 10¹¹ the summed residual is 1.9×10⁻¹⁶ — machine precision — while the
delivered relative error is 1.5×10⁻⁶. A caller who reads the residual as an
error bar believes it has 16 digits and has 6. That is the packaging hazard of
§4.0, not a miscomputation.

The cause is structural, not a tuning problem. A region's Levin value is
`upper_limit - lower_limit` (line 584), a difference of `Σ p[i]·w[i]` evaluated
**at the two endpoints only**, where `w` carries (cos θ, sin θ). Bisecting at
`c` gives `(a,c) + (c,b)`, so the interior point cancels between the children
and parent and children are built from the *identical* θ(a), θ(b). Endpoint
phase rounding is therefore common-mode and subtracts out exactly. It is
invisible by construction, and no amount of subdivision will reveal it.

This also explains §6: the estimator sees only collocation error, which is
exactly the component the Chebyshev order controls — and exactly the component
that is negligible at high frequency.

Two consequences follow. The first is local to this implementation; the second
is a property of the method:

- **Acceptance is disjunctive, and that part is *not* from the paper.** Chen et
  al. step 4 uses a single absolute test, `|val₀ − valL − valR| < ϵ`. Line 830
  accepts a region when `(abserr < atol or relerr < rtol) or depth >=
  depth_max`; the source comment marks the relative branch as a local
  adaptation. Because the added `rtol` branch is satisfied by a residual that
  cannot see the conditioning error, tightening `atol` has no effect: driving it
  from 10⁻¹² to **10⁻³⁰** left the region count at 1 and the delivered error
  unchanged. Under the paper's absolute-only test this particular inertness
  would not arise.
- **Region counts collapse to 1 where delivered error is worst.** This is
  correct behaviour, not a malfunction: the region *is* fully resolved, and
  subdividing it would add rounded endpoints rather than accuracy. It is only
  surprising if one expects the residual to track total error.

The tolerance is a real control at low frequency and stops being one at high
frequency. On `near_singular` at ω = 10⁴, requesting 10⁻¹² versus 10⁻⁴ gives
2.8×10⁻¹³ versus 1.8×10⁻²  — the knob works. At ω = 10¹², all five requested
tolerances from 10⁻¹² to 10⁻⁴ return **identically** 2.3×10⁻⁸.

---

## 5. Bessel tier: the real application

![Bessel tier and cost](figs/fig3_bessel_cost.png)

**Truncation versus the phase floor.** Relative error against the closed forms
falls as 1/x_max — genuine truncation of the tail — until it reaches a floor at
**≈ 2×10⁻⁸**, uniform across all seven oracles. That floor is the accuracy of
the Liouville–Green `bessel_phase` construction, not of the quadrature.
Extending x_max past that point buys nothing.

**Frequency independence, confirmed on the real problem.** Scaling the
wavenumbers by κ = 1 → 100, i.e. from 8×10¹¹ to 8×10¹³ oscillations across the
domain:

| κ | oscillations | median rel. err | Levin time | phase-build time | phase share |
|---|---|---|---|---|---|
| 1 | 8.1×10¹¹ | 2.15×10⁻⁸ | 4.55 s | 8.23 s | 64% |
| 10 | 8.1×10¹² | 2.15×10⁻⁸ | 3.48 s | 8.91 s | 72% |
| 100 | 8.1×10¹³ | 2.16×10⁻⁸ | 0.96 s | 10.9 s | **92%** |

Accuracy is *flat to three significant figures* across two decades of
oscillation count, and the Levin call gets **cheaper**, not more expensive. This
is the campaign's cleanest vindication of the method: at 8×10¹³ oscillations,
brute-force quadrature is not merely impractical, it is unthinkable.

**Head-to-head against brute force on the real problem.** On `J000` and `J222`
across κ = 1 → 300 (8×10⁵ → 2.4×10⁸ oscillations), with `quad` given a 60-second
budget per call:

| κ | oscillations | Levin rel. err | Levin time | `quad` |
|---|---|---|---|---|
| 1 | 8.1×10⁵ | 6.0×10⁻⁷ | 0.92 s | timeout (8.4×10⁵ evals) |
| 10 | 8.1×10⁶ | 5.0×10⁻⁸ | 0.75 s | timeout (7.6×10⁵ evals) |
| 100 | 8.1×10⁷ | 1.5×10⁻⁸ | 0.20 s | timeout (6.2×10⁵ evals) |
| 300 | 2.4×10⁸ | 1.7×10⁻⁸ | 0.29 s | timeout (3.7×10⁵ evals) |

`quad` converged in **0 of 12 cells**. Every call exhausted the full 60 seconds
after 0.4–1.3 million integrand evaluations without reaching its tolerance, so
there is no accuracy number to report for it — the comparison is not "slower",
it is "does not produce an answer". Levin returned eight to nine correct digits
in under a second throughout, and again got *faster* as κ rose. This is the
result that most directly answers the original question, and on the repository's
own application it is unambiguous.

**But the phase layer is now the bottleneck, and it is the scalability limit.**
Phase construction rises to 92% of total cost at κ = 100, and at κ = 1000
(x_max ≈ 8×10¹⁵) a `bessel_phase` build **failed to complete within ~25
minutes** and had to be abandoned. The Levin core scales; the phase table
construction does not.

---

## 6. Chebyshev order economics

Accuracy is identical from order 4 to order 32 — as §3 predicts, since the error
is phase-floor limited rather than collocation limited. Order therefore trades
only cost: raising it reduces subdivision depth and total Levin solve count on
problems that subdivide. The current default of 12 is not optimal on the
problems measured; higher orders did strictly less work for the same answer.

---

## 7. Change list for extraction, ordered by measured payoff

1. **Return an aggregate error estimate, and make it account for the phase
   floor.** There is currently no `abserr` in the returned dict at all. Add a
   per-region phase bound `phase_err = eps · phase_span · |estimate|` — note
   that `phase_span` is *already computed* per region at line 345 for
   conditioning purposes and explicitly "costs no further evaluations of the
   phase function"; it merely needs propagating out of
   `_adaptive_levin_subregion`. Then return
   `sum(max(r.abserr, r.phase_err))` together with a `phase_limited` flag, and
   warn when a caller requested better than `eps·θ_max`.

   This is an *extension* of Chen et al., not a correction to it: their step 4
   residual is a resolution criterion and is sound as such (§4.0), and they
   describe the conditioning loss in prose. The change makes that loss
   machine-readable so a caller who has not read p. 30 of the paper cannot
   mistake the residual for an error bar. It is the single change that most
   affects whether the component is safe to hand to another project.
   *(§4; the conditioning term is what the residual omits, measured up to
   1.5×10¹⁰ below the true error.)*

2. **Delete `range_reduce_mod_2pi` from the phase path; call `sin`/`cos` on the
   unreduced argument.** Measured: 59× faster (median) *and* up to 7.9 orders
   of magnitude more accurate, because libm's own argument reduction is exact
   and pre-reduction in double precision is not. *(§3.)*

3. **Separate "unresolved" from "precision-limited" in region acceptance.**
   The inert `atol` traces to the locally-added `rtol` branch, not to the paper
   (Chen et al. step 4 is absolute-only), so this is a decision about the local
   adaptation. `atol = 10⁻³⁰` was measured to be inert (§4), but the fix is
   *not* to make
   acceptance conjunctive: once endpoint-phase rounding dominates, subdivision
   cannot reduce the error — it adds more rounded endpoints — so a conjunctive
   test would drive every region to `depth_max` for no accuracy gain, at up to
   two orders of magnitude more cost. Instead branch on the cause:

   ```python
   total_err = max(abserr, phase_err)
   if phase_err > atol and phase_err > rtol * abs(estimate):
       accept(region); region.phase_limited = True   # subdivision cannot help
   elif total_err < atol or total_err < rtol * abs(estimate) or depth >= depth_max:
       accept(region)
   else:
       bisect(region)                                # genuinely unresolved
   ```

   At low frequency `phase_err` is negligible and behaviour is unchanged, so
   this is not a regression for the well-conditioned case. *(§4.)*

4. **Document the applicability boundary.** For linear phase, dispatch to QAWO:
   it is seven to nine orders of magnitude more accurate at ω = 10¹² and the
   component should not compete with it. Levin's value is nonlinear phase.
   *(§2.)*

5. **Raise the default Chebyshev order above 12** — free reduction in solve
   count at unchanged accuracy. *(§6.)*

6. **Two free correctness wins noticed while reading the source** (not measured
   separately, both low-risk):
   - Line 831 accumulates `val + estimate`, the *parent* value, although
     `refined_estimate = dataL + dataR` is already computed and is strictly the
     better approximation. Note this is **faithful to the paper** — step 4
     specifies `val = val + val₀` — so it is a proposed improvement on Chen et
     al., not a fix. QUADPACK and `scipy` accumulate the refined value; the
     children have already been evaluated so it costs nothing, and in the accept
     branch the change is bounded by `atol` by construction. Being a deviation
     from the reference algorithm, it should be benchmarked rather than assumed.
   - Line 823's relative-error denominator `min(|estimate|, |refined|)` blows up
     near an accidental zero of the integral, forcing pointless subdivision.
     This affects the `grz` problem family in the harness. Guard it with the
     `atol` scale.

7. **Try a rank-revealing QR in place of the truncated SVD.** Not from my
   measurements — from Remark 2 of the paper (p. 30), which the implementation
   appears not to have taken up: the authors report that in their own
   implementation RRQR replaced the TSVD, was about **5× faster**, and cost no
   apparent accuracy. The linear solve is the per-region inner loop here
   (`levin_solves` counts in the thousands on subdividing problems), so this is
   the largest speedup available on the Levin side that requires no new
   analysis. Unmeasured in this campaign — worth a direct A/B.

8. **Treat `bessel_phase` construction as the scaling frontier.** It is 92% of
   cost at κ = 100 and does not complete at κ = 1000. Any effort spent
   optimising the Levin core in production settings is misdirected until the
   phase build is addressed. *(§5.)*

---

## 8. Reproducing this

`levin_bench/` is self-contained; see its `README.md`. One entry point
regenerates every table and figure:

```
PYTHONPATH=/path/to/SecondaryGWKit python -m levin_bench.campaign all
```

**Caveats, stated plainly.**

- Headline *timing* claims rest on the tier-A frequency ladder, which was run
  alone and serially. Some accuracy-only sweeps ran concurrently with the
  background Bessel scan; their wall-clock columns should not be quoted as
  timing results.
- The `grz` oracle passes through accidental zeros of the integral at certain ω
  (e.g. sin(250π) at ω = 1000), where relative error is not meaningful; those
  cells are handled by the harness but the problem returns near-exact answers
  over much of the ladder and should not be read as representative accuracy.
- `eps·θ_max` is reported as an empirical **upper bound** (never exceeded, with
  0.25 margin, in 35 cells), not as a fitted law: the ratio of delivered error
  to `eps·θ_max` varies by several decades across problems.
- §4 was revised after reading arXiv:2211.13400v3. An earlier draft called the
  convergence test an invalid estimate and an extraction-blocking defect; both
  characterisations were wrong. The measurements are unchanged — only their
  interpretation. Recommendation 1 is now framed as an extension for reuse
  rather than a bug fix, and recommendation 7 comes from the paper rather than
  from this campaign. Anything cited to a page number is from v3; if you work
  from v1 or v2, check the numbering.
- Accuracy floors quoted for tier B are properties of the Liouville–Green phase
  construction in this repository, not of Levin quadrature in general.
