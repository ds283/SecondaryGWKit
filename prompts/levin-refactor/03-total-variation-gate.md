# Prompt 03 — Gate on total variation; replace `quad` with a nested Clenshaw–Curtis rule

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit sections:** §1.3 (C2), §1.7 (C7), §2.1, §2.2, §2.4; §5 recommendation 7
**Depends on:** prompt 02 (you restructure the routine whose solve 02 rewrites)
**Files you may touch:** `AdaptiveLevin/levin_quadrature.py`, `AdaptiveLevin/tests/test_levin_quadrature.py`,
plus the log and the status board.

---

## Character of this commit

**The largest and riskiest change in the campaign, and the one that fixes a wrong answer.**

It closes the second critical finding, replaces an unbounded fallback with a bounded one that
supplies its own error estimate, removes the module's dependency on `Quadrature/`, and eliminates
two phase evaluations per region — all from one restructure.

Read audit §1.3, §2.1 and §2.2 in full before starting. They are three pages and they contain the
whole design.

---

## The defect (C2)

`levin_quadrature.py:924`:

```python
phase_diff = np.fabs(BasisData.raw_theta(b) - BasisData.raw_theta(a))
if np.fabs(phase_diff) < SIX_PI:      # -> hand the region to scipy.quad
```

This is the **net** phase change. The quantity that decides whether Levin has an advantage is the
**total variation** `∫|θ′| dx`. They differ whenever `θ′` changes sign inside the region.

The module already computes the right quantity — `phase_span = mean|θ′| · width` at `:498-500` —
but only *inside* `build_Levin_data`, i.e. after the gate has already fired.

**Reproduced at HEAD.** `I = ∫₀¹ e^{−x} sin(10⁶(x − x²)) dx`, whose phase returns to zero at `x = 1`
(net change 0, total variation 5×10⁵ rad, ≈1.6×10⁵ oscillations):

```
oracle (audit §1.3, 79 578-panel Gauss-Legendre)   I = -6.879079716900e-04
module as committed                                  = +1.024944516387e-02
                                                       rel. err 1.590e+01  (1590%)
                                                       1 region, 0 Levin solves, reported abserr 0.189
same code, called separately on (0, 1/2) and (1/2, 1)= -6.879079722372e-04
                                                       rel. err 8.0e-10, 16 regions
```

The same code delivers ten correct digits when the gate sees a monotone phase and **zero** correct
digits otherwise. That the split run reproduces the audit's independent oracle to 8×10⁻¹⁰ is
confirmation that the oracle is right and the module is wrong.

**This is reachable in production.** `_phase_group` in `three_bessel_integrals.py:111-156` builds
`θ_μ(kx) ± θ_ν(qx) ± θ_σ(sx)`. Each Liouville–Green phase has vanishing derivative at its own
turning point, so a difference group's `θ′` changes sign near `x ≈ max(min_x)/min(k,q,s)` — the
lower end of the Levin range, exactly where the code splits. `QuadSourceIntegral.py`'s `phase2`,
`phase3` and `phase4` have the same structure. This is not a synthetic hazard.

## The second defect (C7)

In the `phase_diff < SIX_PI` branch (`:927-972`):

- `data["abserr"]` from `quad` is recorded into the `used_interval` but **never tested**. The region
  is accepted no matter what `quad` reports.
- The **global** `atol`/`rtol` are passed straight to `quad` as `epsabs`/`epsrel`, so on a hard
  problem each of hundreds of small panels chases a tolerance meant for the whole integral, inside
  `limit=100` subdivisions. Cost is unbounded.
- `scipy`'s `IntegrationWarning` is emitted to stderr as an ordinary Python warning but is not acted
  on and does not reach the returned dictionary.
- The region's cached Levin estimate (from its parent's comparison solve) is discarded. Measured
  waste: 4% of solves at worst — minor, and this restructure removes it anyway.

Measured cost of the current fallback (audit §2.1, `∫₁² e^{−t}/(1+t)·sin(Wt) dt`): 21 integrand
evaluations at a phase span ≤ 2π, **63** at 4π–6π, **147** at 10π — and the `simple_quadrature`
wrapper costs **1.58–1.63×** a bare `scipy.quad` call, about 0.9 µs per evaluation of pure
`QuadSupervisor`/`RHS_timer` overhead (two `perf_counter()` and one `time.time()` per evaluation;
see `Quadrature/simple_quadrature.py:84-92`).

---

## The replacement

Restructure `_adaptive_levin_subregion_impl` (and the driver's use of it) as:

1. Sample `fᵢ` and `θ′` on the Chebyshev grid — **already done**.
2. Compute `phase_span = mean|θ′| · width` — **already done**; just make it available *before* the
   solve, and to the driver.
3. **If `phase_span < threshold`:** skip the linear solve. Evaluate the integrand on the
   `2N−1`-point extremal grid and return the nested pair
   `(CC_{2N−1}, |CC_{2N−1} − CC_N|)` — a value plus a genuine error estimate at fixed, bounded cost.
4. Otherwise solve the Levin system as now.
5. **Feed the fallback estimate into the same accept/bisect logic as a Levin estimate**, so a
   fallback region that misses its tolerance is bisected rather than accepted. This is the C7 fix
   and it is not optional.

### Why this shape

- **It fixes C2 for free.** The gate becomes the total variation, computed from samples that were
  needed anyway. It also removes the two `raw_theta` calls per region that the current gate costs —
  in the three-Bessel application each `raw_theta` is three spline evaluations, so this is six
  spline evaluations per region saved.
- **The fallback cost becomes bounded and predictable:** `2N−1` integrand evaluations, always —
  25 at N = 13 against `quad`'s 21 / 63 / 147, and against *unbounded* when `atol` is tight.
- **It supplies its own error estimate**, so the fallback stops being a silent trust boundary.
- **No wasted solve** on a fallback region.
- **It removes `simple_quadrature` from the hot path**, with its 1.6× wrapper overhead and its
  dependency on `Quadrature/`, `Datastore/` and `utilities/`.

### Clenshaw–Curtis: the facts you need

**The extremal Chebyshev grid the Levin solve already uses *is* the Clenshaw–Curtis grid.**
`_chebyshev_base(N)` returns `cos(kπ/(N−1))`, `k = 0…N−1` (in ascending order after the reversal at
`:379`; `chebyshev_matrices` then maps it to a descending grid on `[a, b]`).

**Nesting holds for every `N ≥ 2`, not only odd `N`.** The `2N−1`-point grid is `cos(jπ/(2N−2))`,
and `j = 2k` gives `cos(kπ/(N−1))` because `2N−2` is even by construction. Verified exactly
(`max|diff| = 0.0` against `chebyshev_matrices`) at **N = 12, 13, 16, 17, 25, 33**.

> **This corrects the audit.** §4.5 says "odd orders are preferable if §2.2's nested Clenshaw–Curtis
> fallback is adopted (nesting needs `N−1` even)". That constraint does not exist. **Do not
> introduce an odd-order requirement**, and do not let it constrain prompt 08's choice of default
> order.

**Accuracy of the proposed rule**, `∫₁² e^{−t}/(1+t)·sin(Wt) dt` at N = 13 — audit §2.2's table,
reproduced here to within a factor 1.1 at every span:

| phase span | `CC₁₃` error | `CC₂₅` error | estimate `\|CC₂₅−CC₁₃\|` | `quad` error |
|---|---|---|---|---|
| 0.5π | 1.4×10⁻¹⁶ | 1.4×10⁻¹⁷ | 1.3×10⁻¹⁶ | 1.4×10⁻¹⁷ |
| 2π | 8.2×10⁻¹³ | 1.4×10⁻¹⁷ | 8.2×10⁻¹³ | 6.9×10⁻¹⁸ |
| 4π | 4.3×10⁻⁸ | 3.5×10⁻¹⁸ | 4.3×10⁻⁸ | 3.5×10⁻¹⁸ |
| 6π | 3.1×10⁻⁶ | 3.2×10⁻¹⁴ | 3.1×10⁻⁶ | 6.1×10⁻¹⁸ |
| 10π | 8.7×10⁻⁵ | 2.9×10⁻⁹ | 8.7×10⁻⁵ | 2.2×10⁻¹⁷ |

`CC₂₅` is comfortably accurate through 6π, and where it is not the estimate says so loudly — the
estimate is the `CC₁₃` error, hence a conservative bound on the `CC₂₅` error. `quad` is more accurate
in absolute terms, which is the honest trade: **you give up two to five digits on the
weakly-oscillatory minority of regions in exchange for bounded cost, a real error estimate, and a
correct gate.** Where those digits matter, step 5 recovers them by bisection, at `2(2N−1)`
evaluations — still cheaper than one `quad` call at 4π.

### Implementation notes for the CC rule

- **Weights.** Clenshaw–Curtis weights on the extremal grid. Compute them once per order and cache
  them next to `_chebyshev_base` (which is already `lru_cache(maxsize=32)`d) — the driver evaluates
  many subregions at the same order. Either the closed-form cosine sum or the standard DCT/FFT
  construction is fine; whichever you choose, **verify it** by integrating a few polynomials of
  degree < N exactly and a known transcendental integral, and say in the log what you verified
  against. Getting the endpoint weights wrong is the classic error and produces a rule that is right
  to several digits, which is exactly the failure that will not be caught by the existing tests.
- **Mark the returned arrays read-only** and document that they are shared, matching
  `_chebyshev_base`'s convention at `:391-392`.
- **Node reuse.** The `N`-point values are a subset of the `2N−1`-point values, so evaluate the
  integrand once on the `2N−1` grid and form both rules from it. Do not evaluate twice.
- **What the CC rule integrates** is the *full* integrand `Σᵢ fᵢ(x) wᵢ(x)`, so it needs `wᵢ` — i.e.
  `sin θ`, `cos θ` — at each of the `2N−1` nodes, which the Levin path does *not* currently compute
  (it needs only `θ′` on the grid and `θ` at the two endpoints). Use `BasisData.eval_basis`, which
  already prefers `theta_mod_2pi` when available (`:534-538`). **Be honest about the cost in the
  log:** a CC region costs `2N−1` evaluations of each `fᵢ` *and* `2N−1` evaluations of the phase,
  on top of the `N` phase-derivative evaluations already spent on the gate. It is still bounded and
  still much cheaper than `quad` at 4π and above, but it is not free and the audit's "`2N−1`
  integrand evaluations, always" understates it.
- **The threshold.** `SIX_PI` need not change; with a nested pair it becomes self-policing, because
  a region where `CC₂₅` is not good enough now gets bisected rather than accepted. The audit notes
  it could be raised to reduce the number of Levin solves at the bottom of the tree. **Do not raise
  it in this commit** — one change at a time. If you measure a case for raising it, record the
  measurement as an observation for a later prompt.

### Feeding the fallback into accept/bisect (step 5, the C7 fix)

The driver's acceptance logic at `:1046-1092` consumes `estimate`, `refined_estimate`, `abserr`,
`relerr` and `phase_err`. A CC region supplies `value = CC_{2N−1}` and `abserr = |CC_{2N−1} − CC_N|`
directly — it has no `refined_estimate` and needs none, because unlike the step-(4) residual its
error estimate comes from a nested pair rather than from a comparison against children.

So a CC region should:
- be accepted when `abserr < atol or relerr < rtol`, on the same terms as a Levin region;
- otherwise be **bisected**, exactly like a Levin region, subject to the same `depth_max`;
- still be recorded as `INTERVAL_TYPE_DIRECT` in `used_regions` so `num_simple_regions` keeps
  meaning what it means, and the benchmark harness (which reads `num_simple_regions`) keeps working.

**A bisected CC region's children re-enter the gate**, and both will have roughly half the phase
span, so they stay on the CC side. That is intended: it is `h`-refinement of the CC rule, which
converges fast on a weakly-oscillatory integrand.

**Guard against infinite descent.** A region containing a genuine stationary point may fail its
tolerance at every depth. `depth_max` catches this — confirm it does, and confirm the depth-limit
warning at `:1197` now fires for such a region (prompt 01 moved the `max_depth` update to the top
of the loop precisely so that it can).

The audit's own measurement of what step 5 buys: with a correct total-variation gate but *without*
step 5, the stationary-phase problem plateaus at a relative error of 1.1×10⁻⁸ because the two
fallback regions straddling the stationary point are accepted with whatever they return. The
reported `abserr` of 1.6×10⁻⁶ correctly detects it; the driver just does not act on it.

### Documentation (§2.4)

Levin's method is **not applicable at a stationary point** of the phase: `p ≈ f/(iθ′)` is singular
there and no amount of refinement fixes it. The correct structural response is what the module does
— refine until the stationary neighbourhood is weakly oscillatory, then use a direct rule on it —
and §1.3 shows this works once the gate measures total variation.

Add this to the module docstring's "DEVIATIONS FROM THE REFERENCE ALGORITHM" list (`:26-54`), and
update the existing third bullet, which currently reads "Weakly oscillatory regions (phase span
< 6*pi) are handed to ordinary adaptive quadrature" — after this commit that is wrong in two ways
(it was net phase, not span; and it is no longer ordinary adaptive quadrature).

Also update `adaptive_levin_sincos`'s docstring (written in prompt 01) to describe the new fallback
and its error estimate.

---

## Do not

- **Do not reintroduce p-refinement or global worst-first error balancing.** Audit §2.3 measured
  both. p-refinement costs 1.5–2.2× more integrand evaluations, and in production the integrand is a
  stack of spline evaluations, so evaluations are the currency. Global balancing is *unsound* with
  this estimator: the step-(4) residual is a difference of two rules, not a per-region bound that
  decreases under refinement, so the children's residuals can exceed the parent's and a heap driven
  by it need not converge — it ran to 8 192 regions and 827 353 evaluations where the local scheme
  used 3 665. Do not re-derive this from first principles; it is settled.
- **Do not raise `SIX_PI`** in this commit.
- **Do not add an odd-order constraint.** Nesting does not need one (see above).
- Do not change `theta_scale` or `_phase_error` — prompt 04. The CC branch's phase floor is prompt
  04's problem; in *this* commit, give the CC region the same `phase_err` treatment the current
  direct branch gets (`:947-958`), adapted to the new structure, and note in the log that prompt 04
  will replace it.
- Do not change the acceptance tolerance semantics — prompt 05.
- Do not remove `Quadrature/simple_quadrature` from the repository. Other modules use it
  (`three_bessel_integrals.py`, and others). Only remove the *import in this file*, and only if
  nothing else in `AdaptiveLevin/` needs it.

---

## Verification

1. `AdaptiveLevin/tests/` passes.
2. **The C2 case is fixed.** `∫₀¹ e^{−x} sin(10⁶(x − x²)) dx` must now agree with the oracle
   `-6.879079716900e-04`. The audit's total-variation prototype achieved 1.1×10⁻⁸ relative *without*
   step 5; with step 5 you should do better. Report the relative error, the region count, the
   fallback-region count and the reported `abserr`. **The reported `abserr` must exceed the true
   error** — if it does not, stop and investigate, because that is the property the whole campaign
   is defending.
3. **CC weights are right.** Show the check: exact integration of polynomials up to degree `N−1`,
   and agreement with a known transcendental integral.
4. **Nesting is right.** Assert `chebyshev_matrices((a,b), 2*N-1)[0][::2] == chebyshev_matrices((a,b), N)[0]`
   for several `N` and spans — this belongs in the test file, not just in the log.
5. **The fallback error estimate is honest.** For each row of the `CC₁₃`/`CC₂₅` table above, confirm
   the estimate exceeds the true `CC₂₅` error.
6. **A fallback region that misses its tolerance is bisected.** Construct one and show it.
7. **No accuracy regression on well-resolved problems.** The five-problem A/B from prompt 02, before
   and after: `value`, `abserr`, `num_regions`, `num_simple_regions`, wall time. Regions that were
   already Levin regions must be unaffected; the changes should appear only in the fallback
   population.
8. **Three-Bessel oracles.** At least two from `LiouvilleGreen/tests/test_3bessel_analytic.py`,
   before and after. These are the production integrands and they have difference-type phase groups,
   so this is where C2 could actually be biting today. **If any oracle improves, say so loudly** —
   that would mean C2 was affecting production results, which is worth the user knowing.
9. **Cost of the fallback.** Measured integrand evaluations and wall time per fallback region,
   before and after, at phase spans of 2π, 4π and 6π.
10. `grep -n simple_quadrature AdaptiveLevin/levin_quadrature.py` returns nothing (or only a comment
    explaining why it was removed).

---

## Finish

1. Write `prompts/levin-refactor/logs/03-total-variation-gate.md`. This is the campaign's most
   consequential log — a later reader must be able to understand the new region taxonomy without
   reading the diff. Under *What shipped*, describe the two cell types, what each returns, and how
   each enters accept/bisect. Under *Numerical evidence*, items 2, 5, 7, 8 and 9 above.
   Judgement calls that must be recorded: the CC weight construction chosen; whether the phase
   evaluations on the `2N−1` grid reuse anything; how the CC region's `phase_err` is set pending
   prompt 04; and whether anything about `SIX_PI` was measured but deliberately not changed.
2. Update `IMPLEMENTATION_STATE.md`. **C2 and C7 close here** — if either does not fully close, say
   so and open a §3 issue rather than marking it done.
3. Commit in one commit. Suggested message:

```
Gate the Levin fallback on total variation, not net phase change

The weakly-oscillatory gate compared theta(b) with theta(a) and handed the
region to scipy.quad when their difference fell below 6*pi. That is the net
phase change; the quantity that decides whether the Levin rule has an
advantage is the total variation of the phase. They differ whenever theta'
changes sign inside the region.

On int_0^1 exp(-x) sin(1e6 (x - x^2)) dx, whose phase returns to zero at
x = 1, the net change is zero and the total variation is 5e5 radians. The
whole integral was therefore handed to quad in a single region, the Levin
rule was never invoked, and the answer was wrong by 1590%. Splitting the same
integral at x = 1/2 -- so that each half has a monotone phase -- gave ten
correct digits from the same code.

This is reachable from production: three_bessel_integrals._phase_group builds
difference-type phase groups whose derivative changes sign near each Bessel
turning point, at the lower end of the Levin range.

The module already computed the right quantity, mean|theta'| * width, but
only inside build_Levin_data, after the gate had fired. The subregion routine
is restructured to sample once, gate on the sampled phase derivative, and
only then decide between the Levin solve and the fallback.

The fallback itself is replaced. The extremal Chebyshev grid used by the
Levin solve is the Clenshaw-Curtis grid, and the N-point grid is exactly
every other node of the (2N-1)-point grid, so a nested Clenshaw-Curtis pair
gives a value and a genuine error estimate for 2N-1 evaluations, always. In
place of that, quad cost 21 evaluations below 2*pi, 63 at 4*pi-6*pi and 147
at 10*pi -- unbounded when atol is tight, since it was handed the global
tolerance for each small panel -- through a wrapper measured at 1.6x the cost
of a bare scipy.quad call.

The fallback estimate now enters the same accept/bisect logic as a Levin
estimate, so a fallback region that misses its tolerance is bisected instead
of being accepted unconditionally, which is what previously happened to quad's
own reported error: it was recorded and never tested.

This trades two to five digits on the weakly-oscillatory minority of regions
for bounded cost, a real error estimate, a correct gate, and the removal of
the Quadrature/ dependency from this module.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```
