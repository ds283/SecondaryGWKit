# `AdaptiveLevin`: correctness, efficiency and error-estimate audit

**Subject.** `AdaptiveLevin/levin_quadrature.py` at commit `68cff5d` (1489 lines), together with
`Quadrature/simple_quadrature.py`, on which its fallback path depends.

**Reference.** Bremer, Chen & Yang, *Rapid evaluation of oscillatory integrals…*,
arXiv:2211.13400**v3** (page and equation numbers below are v3, from the copy in
`docs/adaptive-levin-benchmark/`). The adaptive scheme is §5, p. 28–30; the round-off analysis
is §4, culminating in eq. (151).

**Prior work this builds on.** `docs/adaptive-levin-benchmark/ADAPTIVE-LEVIN-REVIEW.md`
(static review) and `LEVIN-PERFORMANCE-REPORT.md` (measurement campaign). Both predate
`b76570e`/`68cff5d`; several of their recommendations are now implemented (§0.2).

**Method.** Everything asserted below is either a derivation checked against the paper or a
measurement. Reproduction scripts are listed in §7. No repository files were modified.

---

## 0. Summary

### 0.1 Verdict

The Levin core is correct and, given a resolved region, delivers the frequency-independent
absolute accuracy the paper predicts (measured: ≤1.2×10⁻¹⁶ absolute from ω = 10⁴ to 10¹², §3.2).
The spectral differentiation matrix, the descending-grid endpoint convention, the `Aᵀ`
construction and the endpoint extraction all verify exactly (§1.1). The error-estimate work
added since the campaign is a real improvement and it has, as a side effect, closed the
"tolerance trap" that the campaign identified as the module's worst operational hazard (§0.2).

Three defects, however, break the property the module most needs in order to be handed to
another project: **that it either returns a meaningful result or refuses**.

| | Finding | Severity |
|---|---|---|
| **C1** | A non-finite amplitude sample makes a region contribute **exactly 0 with a reported error of exactly 0**. The caller receives a partial integral certified at machine precision. | critical |
| **C2** | The weakly-oscillatory gate tests the **net** phase change, not the total variation. On a phase with an interior stationary point the entire integral is handed to `quad` and the answer is wrong by **1590%** — while the Levin rule is never invoked at all. | critical |
| **C3** | `atol` is a **per-region** tolerance. The returned aggregate `abserr` is never compared with what the caller asked for, and was measured exceeding it with no warning. | high |

Two further findings concern the error estimate itself:

| | Finding | Severity |
|---|---|---|
| **C4** | For a range-reduced phase the floor is hardwired to `theta_scale = 2π`, making it optimistic by the oscillation count — measured **5.5×10¹⁰** too small at ω = 10¹². It stays optimistic even for an *exactly* reduced phase, because the residual floor is not endpoint-phase rounding at all (§3.3). | high |
| **C5** | The `p_use` mode filter gates on the **mean** of `\|p\|` over the collocation points but the estimate uses only its **endpoint** values; it makes the returned value a function of `rtol` at fixed mesh; and it is the mechanism behind C1. | medium |

On efficiency, the measured picture inverts the campaign's assumption: at the default order
**the linear solve is 15% of a subregion evaluation and `np.block` assembly alone is 35%**
(§4.1). Solving the equivalent complexified `N×N` system instead of the real `2N×2N` one, and
assembling into a preallocated buffer, gives a **1.4–1.8× end-to-end speedup with
bit-identical results** (§4.2). Two optimisations were measured and found *not* to pay:
p-refinement before bisection (§2.3) and prioritising a rank-revealing QR (§4.4).

### 0.2 What has already been fixed since the campaign

Credit where due, so the reader does not re-litigate closed items:

- An aggregate `abserr`/`relerr` is now returned, with a `phase_limited` flag
  (report §7 item 1). ✔
- The relative-error denominator is floored at `atol` (report §7 item 6b), which stops
  pointless subdivision at accidental zeros of a region's contribution. ✔
- Child subregion estimates are cached and reused (review §7.4). ✔ Measured residual waste:
  4% of solves at worst (§4.6) — no further action needed.
- The Chebyshev base grid and differentiation matrix are `lru_cache`d (review §7.5). ✔
- A direct LU fast path, gated on phase span, bypasses `lstsq` on well-conditioned regions. ✔
  Measured: it covers **75–100%** of solves (§4.4).
- **The tolerance trap is closed.** Re-running the review's §6.2 scenario (`atol=1e-30`,
  `rtol=1e-16` on `∫₁¹⁰⁰ sin(10³x)/x dx`) now yields 11 regions / 43 solves, not 892 regions /
  2781 solves: the `phase_limited` acceptance branch terminates the runaway. ✔
  **Caveat:** the guard fires at a threshold set by the phase floor, so C4 delays it on the
  range-reduced path. Measured (`e30_trapmod.py`) on the same integral: the raw path arms the
  guard from `atol ≈ 10⁻²²` downwards; the reduced path only from `atol ≈ 10⁻³⁰`, and pays
  23 regions / 91 solves against the raw path's 11 / 43 — about 2×, not the campaign's 80×.
  I could not reproduce the 892-region blowup at `HEAD` on any problem tried.

---

## 1. Correctness

### 1.1 Structural verification (all pass)

Checked by derivation and numerically (`e01_struct.py`):

- `chebyshev_matrices` returns a **descending** grid, `x[0] = b`, `x[-1] = a`, and `D`
  differentiates correctly on it (`max|Df − f′|` = 3.7×10⁻¹⁰ at N=12, 8.9×10⁻¹³ at N=32 for
  `f = exp`; spectral convergence as expected). Odd N works. The comment at line 414 documenting
  the convention is accurate and load-bearing.
- With `w = (sin θ, cos θ)`, `w′ = A w` and `(p·w)′ = f·w` give `p′ + Aᵀp = f` with
  `Aᵀ = [[0, −θ′],[θ′, 0]]` — matching lines 490–493. `LevinL = blockdiag(D,D) + Aᵀ`
  matches (168). Endpoint extraction `p₁ sin θ + p₂ cos θ` differenced across the region
  matches (171). ✔
- A **reversed span** (`b < a`) works and returns the correctly negated value. Untested, but
  correct.
- A **zero-width span** returns 0.0. ✔
- `lstsq(..., rcond=None)` truncates at `eps·max(M,N)·σ₁`, which is the paper's step 5
  truncation at `‖A‖ε₀`. ✔

### 1.2 C1 (critical) — a non-finite amplitude yields a silent zero

`levin_quadrature.py:662` checks `np.isfinite(LevinL).all()` but **nothing checks `f_Cheb`**.
The failure then routes through `p_use` (line 769):

```python
p_use = [r > rtol for r in p_ratios]      # NaN > rtol is False
```

A NaN anywhere in the sampled amplitude propagates to `p`, hence to `p_ratios`, hence
`p_use == [False, False]`, hence `lower_limit = upper_limit = 0.0` **and**
`p_endpoint_l1 = 0.0`, so `phase_err = 0.0`. The driver then sees `estimate = 0`,
`refined = 0 + 0 = 0`, `abserr = 0 < atol`, and accepts.

Measured (`e21_nan.py`, `e22_nan2.py`), amplitude `f₀(x) = NaN if x > 1.9 else 1`, θ = 10⁵x
on [1, 2], default tolerances:

```
region (1.875, 2)  →  abserr=0, phase_err=0, p-ratios=[nan, nan],  contribution = 0.0
driver             →  value = -5.191968760836506e-07,  abserr = 1.03e-15,  4 regions
```

The interval whose grid saw the NaN contributed nothing, the other three were integrated
normally, and the caller is handed a **partial integral certified at 10⁻¹⁵**. A spline
evaluated outside its range, an overflow in `x^{3/2}·m_μ m_ν m_σ`, or a division by zero in a
user amplitude all trigger this. All-NaN input gives `value = 0.0, abserr = 0.0` — the most
dangerous possible response.

**Recommendation.** Validate `f_Cheb` and the sampled `θ′` for finiteness immediately after
sampling and raise (the existing `LevinL` check is the right model); make `p_use` reject
non-finite ratios explicitly rather than by accident; assert `np.isfinite(p).all()` after the
solve. Cost: one `np.isfinite(...).all()` on an `mN` vector per solve, ~0.5 µs at N=12.

### 1.3 C2 (critical) — the weakly-oscillatory gate uses net phase, not total variation

`levin_quadrature.py:924`:

```python
phase_diff = np.fabs(BasisData.raw_theta(b) - BasisData.raw_theta(a))
if np.fabs(phase_diff) < SIX_PI:      # -> hand the region to scipy.quad
```

This is the *net* phase change. The quantity that decides whether Levin has an advantage is the
**total variation** `∫|θ′|dx`. They differ whenever `θ′` changes sign inside the region — and
the module already computes the right quantity, `phase_span = mean|θ′|·width`
(line 498), but only *inside* `build_Levin_data`, i.e. too late to be used by the gate.

Measured (`e26_oracle.py`, `e27_tvgate.py`). Test integral
`I = ∫₀¹ e^{−x} sin(10⁶(x − x²)) dx`, whose phase returns to zero at `x = 1`
(net change 0, total variation 5×10⁵ rad, ≈1.6×10⁵ oscillations). Independent oracle by the
exact reduction `I = e^{−1/2}∫₀¹ cosh(v/2) sin(10⁶(1−v²)/4) dv`, evaluated with
79 578 Gauss–Legendre panels aligned to the zeros of the phase; two independent
double-precision variants agree to 6.3×10⁻¹⁵:

```
oracle                                   I = -6.879079716900e-04
module as committed                        = +1.024944516387e-02   rel. err 1.59e+01  (1590%)
                                             1 region, 0 Levin solves, reported abserr 1.9e-01
same code, interval split at x = 1/2       = -6.879079716894e-04   rel. err 3.9e-12
                                             20 regions, 74 solves, reported abserr 1.4e-13
prototype with a total-variation gate      = -6.879079796062e-04   rel. err 1.1e-08
                                             16 regions (2 fallback), reported abserr 1.6e-06
```

The same code delivers twelve correct digits when the gate sees a monotone phase and **zero**
correct digits otherwise. The reported `abserr` (0.19) does at least exceed the actual error
(0.011), so the aggregate estimate is honest — but no exception and no warning is raised, and
the module's own callers discard `abserr` (`three_bessel_integrals.py:207` keeps only
`data["value"]`).

**Reachability in the real application.** `_phase_group` in `three_bessel_integrals.py` builds
`θ_μ(kx) ± θ_ν(qx) ± θ_σ(sx)`. Each Liouville–Green phase has vanishing derivative at its own
turning point, so a difference group's `θ′` changes sign in the neighbourhood of
`x ≈ max(min_x)/min(k,q,s)` — the lower end of the Levin range, exactly where the code splits.
This is not a synthetic hazard.

**Recommendation.** Gate on total variation. The clean way is a restructure that is also the
answer to the graceful-degradation question — see §2.2.

### 1.4 C3 (high) — `atol` is a per-region tolerance and the aggregate is never checked

Acceptance (line 1083) is `abserr < atol or relerr < rtol` **per region**. `abserr_total` is
then the sum over accepted regions (line 1173), so the delivered absolute error scales like
`N_regions × atol`. The paper has the same structure (their step 4 uses a per-interval `ϵ`) and
gets away with it because they run at `ϵ = 10⁻¹³` with few intervals; for a library API this
is a broken contract.

Measured (`e15_policy2.py`, prototype sharing the module's core). `∫₀³ e^{−400(x−1.1)²}
sin(3×10⁴x) dx`, `atol = 10⁻¹⁰`:

```
delivered absolute error = 1.27e-10       (exceeds the requested atol)
reported   abserr        = 1.26e-10       (also exceeds it — and nothing says so)
```

Two independent fixes, both cheap:

1. **Compare the aggregate with the request.** After summing, if
   `abserr_total > max(atol, rtol·|val|)`, return `converged = False` and warn. Zero cost. This
   is the single most valuable change in the audit relative to effort.
2. **Distribute the absolute tolerance by interval length**: accept when
   `abserr < atol·(b−a)/(b₀−a₀)`, which makes `Σ abserr < atol` true by construction.
   Measured (`e16_scaled.py`, five problems × three tolerances): this is **free at matched
   delivered accuracy** — `local-h` at `atol = 10⁻¹²` and `local-h-scaled` at `atol = 10⁻¹⁰`
   produce identical evaluation counts, solve counts, region counts and errors on three of the
   five problems. It is a relabelling that makes `atol` mean what a caller assumes it means.
   At fixed nominal `atol` it costs 1.1–1.5× more evaluations and buys 10–100× accuracy.

### 1.5 C5 (medium) — the `p_use` mode filter

Three separate problems with lines 764–789.

**(a) It gates on the wrong quantity.** `p_ratios` is built from `mean|P[j,:]|` over the
collocation points, but what the estimate consumes is `P[j, 0]` and `P[j, −1]` — the endpoint
values. A mode with a small mean and a large endpoint value would be discarded wrongly. (I did
not find a case where this actually bites: over 400 randomised spike-amplitude problems the
worst perturbation from the gate was 4.1×10⁻⁷ against `rtol = 10⁻⁷`, i.e. bounded by `rtol` as
intended — `e17_puse.py`.) The fix is free: gate on `(|p_i(a)| + |p_i(b)|) / Σ_j(|p_j(a)| +
|p_j(b)|)`, the quantity actually used.

**(b) It makes the returned value a function of `rtol` at fixed mesh.** Measured
(`e18_puse2.py`), one region, order 16, mode ratios `[1.06×10⁻⁶, 1]`:

```
rtol = 1e-2, 1e-4      ->  value = 1.81743030831524e-07
rtol <= 1e-6           ->  value = 1.81736473688358e-07     (relative jump 3.6e-5)
```

The jump is bounded by `rtol`, so it is defensible — but it is invisible to `abserr`, because
parent and children use identical gating and the discarded amount is common-mode in the step-(4)
residual. A caller with `atol = 10⁻²⁰, rtol = 10⁻³` (a legal combination) gets a value perturbed
at the 10⁻³ level with an error bar near machine precision. **Fix:** add the discarded endpoint
contribution `Σ_{i not used} (|p_i(a)| + |p_i(b)|)` to the region's `abserr`. A handful of flops,
and it makes the heuristic self-consistent.

**(c) It is the mechanism behind C1.** See §1.2.

### 1.6 C6 (medium) — `max_depth` is not updated on the fallback branch

The direct-quadrature branch `continue`s at line 972, before the
`if current_region.depth > max_depth` update at line 1112. So a run that bisects to the `6π`
floor and terminates in direct quadrature never raises `max_depth`, and the depth-limit health
warning at line 1197 cannot fire — which is precisely the non-convergence mode the warning
exists to catch. **Fix:** move the update to immediately after `regions.pop()`.

### 1.7 C7 (medium) — fallback regions are accepted unconditionally

In the `phase_diff < SIX_PI` branch (lines 927–972):

- `data["abserr"]` from `quad` is recorded but **never tested**. The region is accepted no
  matter what `quad` reports, and `scipy`'s `IntegrationWarning` is neither caught nor
  surfaced. A fallback region that fails its tolerance is silently accepted.
- The **global** `atol`/`rtol` are passed straight to `quad` as `epsabs`/`epsrel`. On a hard
  problem this makes each of hundreds of small panels chase a tolerance meant for the whole
  integral, inside `limit=100` subdivisions.
- The region's cached Levin estimate (from its parent's comparison solve) is discarded.
  Measured waste: 4% of solves at worst (§4.6) — minor.

This is visible in §1.3's numbers: even with a correct total-variation gate the answer plateaus
at 1.1×10⁻⁸ because the two fallback regions straddling the stationary point are accepted with
whatever they return; the reported `abserr` of 1.6×10⁻⁶ correctly detects it, but the driver
does not act on it.

**Fix:** treat the fallback estimate exactly like a Levin estimate — if its own error estimate
fails the tolerance, bisect the region.

### 1.8 C8 (medium) — `atol = 0` can subdivide without bound

With `atol = 0` the relative-error denominator floor (line 1059) is inert, `relerr` becomes
`0/0 = nan` for a region with a vanishing contribution, and the `phase_limited` branch cannot
fire because `phase_err > atol` is `0 > 0`. Measured (`e04_edge.py`), identically-zero
integrand, `atol = 0`, `depth_max = 8`:

```
256 regions, 1023 solves   (correct answer: 1 region, 3 solves)
```

At the default `depth_max = 20` that is 2²⁰ regions. A caller wanting pure relative control
would naturally write `atol = 0`. **Fix:** reject non-positive `atol` (or floor it at, say,
`1e-300` and floor `relerr_denom` at a positive number derived from the running total).

### 1.9 C9 (low) — no input validation

Measured (`e20_valid.py`):

| Input | Current behaviour | Should be |
|---|---|---|
| `f` of length ≠ 2 | `ValueError: operands could not be broadcast together with shapes (12,12) (24,24)` | clear error naming the 2-component contract of `_Basis_SinCos` |
| `f = []` | `ValueError: need at least one array to concatenate` | clear error |
| `x_span` of length 3 | **silently integrates `(x_span[0], x_span[1])`** | error |
| `atol < 0`, `rtol < 0` | silently accepted | error |
| `depth_max < 0` | silently accepted; every region accepted at depth 0 | error |
| `chebyshev_order < 8` | silently clamped to 8 | documented, or a warning |
| `theta` passed as a bare callable | `TypeError: argument of type 'function' is not iterable` | clear error (this is an easy mistake — the parameter is *named* `theta` and the module's own tests pass `theta={"theta": …}`) |
| NaN endpoint | `ValueError` from the `LevinL` finiteness check | fine, but the message misattributes the cause |

`adaptive_levin_sincos` has **no docstring**.

### 1.10 C10 (low) — diagnostic counters under-count by ~3×

`num_direct_solves`, `num_SVD_errors` and `num_order_changes` are accumulated only from
`data["metadata"]` (lines 1005–1009) — the *parent* solve. The two comparison solves `dataL`,
`dataR` are never inspected. Measured (`e23_solver.py`, `e24_solver2.py`):

```
lorentz peak, k=12:  true solves 95, true LU 71, true lstsq 24  |  reported num_direct_solves 39
gauss peak,  k=12:   true solves 43, true LU 43, true lstsq  0  |  reported num_direct_solves 21
```

Consequently the source comment's claim that the fast path "removes the SVD from the
well-conditioned majority of regions" cannot be checked from the returned dictionary — it is in
fact true (§4.4), but only measurable by instrumentation. Also: `evaluations` counts subregion
*solves*, not integrand evaluations, which invites misreading (the review's §2.2 read it as
"linear solves ÷ 3"). And when every order down to the floor fails, the reported
`chebyshev_order` is `6` — an order that was never used.

### 1.11 C11 (low) — operational hygiene (unchanged since the review)

- `seaborn` and `matplotlib.pyplot` are imported at module scope (lines 66–67) for the sole
  benefit of `_write_progress_data`. Measured: **1.13–1.36 s of the module's 1.57 s import
  time**. Under a Ray driver that is paid per worker.
- The `lstsq` failure path writes `LevinL_<isoformat>.txt` and `f_Cheb_<isoformat>.txt` into the
  **current working directory** at one-second timestamp resolution — collisions between
  parallel workers, unbounded disk use, arbitrary location.
- `_write_progress_data` writes to a cwd-relative `SlowLevinData/…` path.
- All warnings and progress notices go to `print`, not `logging`.
- `_write_progress_data` re-solves three subregions per region and evaluates the integrand
  500×(m+1) times per region, and uses the *unguarded* `min(|est|,|ref|)` denominator (line
  1393) that was fixed in the main path. Diagnostics-only, but inconsistent.

### 1.12 C12 (low) — test coverage gaps

`AdaptiveLevin/tests/test_levin_quadrature.py` has four tests, all passing in 0.010 s. Not
covered:

- The `theta_mod_2pi` path — i.e. the branch production actually uses.
- The `theta_deriv` path (all four tests use spectral differentiation).
- Any region taking the direct-quadrature fallback deliberately. (`_GRZIntegral(10.0)` takes it
  by accident: `10·(atan 1 − atan −1) = 15.7 < 6π`, so **that test never exercises Levin at
  all** — worth noting, since it is one of the four.)
- Non-monotonic phase (C2), non-finite amplitude (C1), reversed span, `m ≠ 2`, `atol = 0`.
- Any check of the returned `abserr` against the true error.

---

## 2. Graceful degradation

### 2.1 What is there now, and what is wrong with it

The escalation ladder is: Levin at fixed order → bisect → bisect → … until either a tolerance
is met, the phase floor is hit, `depth_max` is reached, or the *net* phase change falls below
`6π`, at which point `simple_quadrature(method="quad")` takes the region.

Measured costs of that last step (`e10_fallback.py`, `∫₁² e^{−t}/(1+t)·sin(Wt) dt`):

| phase span | `quad` integrand evaluations |
|---|---|
| ≤ 2π | 21 |
| 4π–6π | 63 |
| 10π | 147 |

and the `simple_quadrature` wrapper costs **1.58–1.63×** the bare `scipy.quad` call — about
0.9 µs per integrand evaluation of pure `QuadSupervisor`/`RHS_timer` overhead (two
`perf_counter()` and one `time.time()` per evaluation).

Structural problems: the gate measures the wrong thing (C2); the inner solver is *adaptive*, so
its cost is unbounded and driven by a tolerance meant for the whole integral (C7); its
convergence flag is discarded (C7); and the already-computed Levin estimate for the region is
thrown away.

### 2.2 Recommended replacement: sample once, gate on `phase_span`, fall back to nested Clenshaw–Curtis

The extremal Chebyshev grid the Levin solve already uses **is** the Clenshaw–Curtis grid, and
for odd `N` the `N`-point grid is exactly every other node of the `2N−1`-point grid
(verified, `e10_fallback.py`). So restructure `_adaptive_levin_subregion_impl` as:

1. Sample `f_i` and `θ′` on the grid (already done).
2. Compute `phase_span = mean|θ′|·width` (already done — just move it before the solve).
3. **If `phase_span < threshold`**: skip the linear solve. Evaluate the integrand at the
   `2N−1`-point extremal grid and return the nested pair `(CC_{2N−1}, |CC_{2N−1} − CC_N|)` —
   a value plus a genuine error estimate at **fixed, bounded cost**.
4. Otherwise solve the Levin system as now.
5. Feed the fallback estimate into the same accept/bisect logic as a Levin estimate, so a
   fallback region that misses its tolerance is bisected rather than accepted (C7).

Why this is the right shape:

- **It fixes C2 for free.** The gate becomes the total variation, computed from samples that
  were needed anyway. It also removes the two `raw_theta` calls per region that the current
  gate costs — in the three-Bessel application `raw_theta` is three spline evaluations.
- **The fallback cost becomes bounded and predictable.** `2N−1` integrand evaluations, always
  — 25 at `N = 13` versus `quad`'s 21 / 63 / 147, and versus *unbounded* when `atol` is tight.
- **It supplies its own error estimate**, so the fallback stops being a silent trust boundary.
- **No wasted solve** on a fallback region.
- **It removes `simple_quadrature` from the hot path** along with its 1.6× wrapper overhead
  and its dependency on `Quadrature/`, `Datastore/` and `utilities/` — relevant to the
  extraction plan.

Accuracy of the proposed rule, `∫₁² e^{−t}/(1+t)·sin(Wt) dt` at `N = 13` (`e09_cc.py`):

| phase span | `CC₁₃` error | `CC₂₅` error | estimate `\|CC₂₅−CC₁₃\|` | `quad` error |
|---|---|---|---|---|
| 0.5π | 1.4×10⁻¹⁶ | 0 | 1.4×10⁻¹⁶ | 1.4×10⁻¹⁷ |
| 2π | 8.2×10⁻¹³ | 3.5×10⁻¹⁸ | 8.2×10⁻¹³ | 6.9×10⁻¹⁸ |
| 4π | 4.3×10⁻⁸ | 1.6×10⁻¹⁷ | 4.3×10⁻⁸ | 3.5×10⁻¹⁸ |
| 6π | 3.1×10⁻⁶ | 3.2×10⁻¹⁴ | 3.1×10⁻⁶ | 6.1×10⁻¹⁸ |
| 10π | 8.7×10⁻⁵ | 2.9×10⁻⁹ | 8.7×10⁻⁵ | 2.2×10⁻¹⁷ |

`CC₂₅` is comfortably accurate through 6π, and where it is not the estimate says so loudly
(it is the `CC₁₃` error, hence a conservative bound on the `CC₂₅` error). `quad` is more
accurate in absolute terms, which is the honest trade: **you give up two to five digits on the
weakly-oscillatory minority of regions in exchange for bounded cost, a real error estimate, and
a correct gate.** If those digits matter, step 5 recovers them by bisection — at `2(2N−1)`
evaluations, still cheaper than one `quad` call at 4π.

The `6π` threshold need not change; with a nested pair it becomes self-policing, so it could
also be raised to reduce the number of Levin solves at the bottom of the tree.

### 2.3 Two escalation strategies measured and rejected

**p-refinement before bisection.** Raise the order on the same interval before splitting it —
the natural response when the amplitude is analytic (which it is for Liouville–Green moduli).
Measured (`e15_policy2.py`, one `order → 2·order` attempt before each bisection):

| problem | `atol` | policy | f evaluations | solves | regions | true error |
|---|---|---|---|---|---|---|
| lorentz peak | 10⁻¹³ | h only | 3 665 | 95 | 26 | 6.3×10⁻¹⁵ |
| lorentz peak | 10⁻¹³ | p then h | **8 229** | 156 | 22 | 5.5×10⁻¹⁸ |
| 1/x on [10⁻², 1] | 10⁻¹³ | h only | 1 311 | 35 | 9 | 7.4×10⁻¹⁵ |
| 1/x on [10⁻², 1] | 10⁻¹³ | p then h | **2 157** | 41 | 6 | 8.6×10⁻¹⁵ |

p-refinement reliably reduces the *region* count and often improves accuracy, but costs
**1.5–2.2× more integrand evaluations** — and in production the integrand is a stack of spline
evaluations, so evaluations, not regions, are the currency. **Verdict: measured, does not pay.**

**Global error balancing (worst-first refinement with a global stopping test).** The obvious
QUADPACK-style restructure: max-heap on per-region error, subdivide the worst region, stop when
`Σ err < max(atol, rtol|val|)`. Measured (`e15_policy2.py`) it is either identical to the
current local scheme or **catastrophically worse**: on `lorentz peak` at `atol = 10⁻¹³` it ran
to 8 192 regions and 827 353 evaluations against 3 665 for the local scheme.

The reason is instructive and worth recording so nobody tries it again: **the step-(4) residual
`|val₀ − val_L − val_R|` is not a per-region error bound that decreases under refinement.** It
is a difference of two rules, and the sum of the children's residuals can exceed the parent's,
so a heap driven by it need not converge. QUADPACK's Gauss–Kronrod estimator does not have
this property. **Verdict: unsound with this estimator.** The contract benefit that motivated it
(the delivered error matching the request) is obtained far more cheaply by §1.4's
length-proportional tolerance, which is *free at matched accuracy*.

### 2.4 Stationary points — document the boundary

Levin's method is not applicable at a stationary point of the phase: `p ≈ f/(iθ′)` is singular
there, and no amount of refinement fixes it. The correct structural response is what the module
already does — refine until the stationary neighbourhood is weakly oscillatory, then use a
direct rule on it — and §1.3 shows that this *works* (1.1×10⁻⁸, and better with §2.2 step 5)
once the gate measures total variation. This should be stated in the module docstring alongside
the existing "deviations from the reference algorithm" list, since a caller integrating a
difference-type phase group will meet it.

---

## 3. Error estimates

### 3.1 The step-(4) resolution residual

Correct as implemented and faithful to the paper (step 4, p. 30). The
`docs/adaptive-levin-benchmark` analysis of why it cannot see endpoint phase rounding
(common-mode between parent and children) is right, and the module's comment at lines 145–150
states it accurately.

One caveat worth acting on: it is a difference of two estimates, not a bound, and it can
under-report the collocation error. Measured (`e15_policy2.py`, chirp at ω = 3×10⁴ against a
high-order reference of the same core): reported 2.4×10⁻¹⁷ against a true 2.8×10⁻¹⁵ — a factor
120. A safety factor on the residual (as QUADPACK applies to its own) would be prudent.

### 3.2 `_phase_error`: the derivation is right

`value = Σᵢ pᵢ(b)wᵢ(b) − Σᵢ pᵢ(a)wᵢ(a)` and `|dw/dθ| ≤ 1` componentwise, so
`|d value| ≤ dθ·(Σ|pᵢ(b)| + Σ|pᵢ(a)|)`. Correct, and conservative by up to √2 (the true
sensitivity is `√(p₁² + p₂²)`, not `|p₁| + |p₂|`). ✔

For the **raw** phase path (`theta_scale = max|θ_Cheb|`) the bound is empirically sound.
Measured (`e08_bound.py`), `∫_{1/3}^{7/3} e^{−x} sin(ωx) dx` against a closed form:

| ω | delivered abs. error | `eps·\|θ\|·Σ\|p\|` | ratio |
|---|---|---|---|
| 10⁴ | 4.3×10⁻¹⁸ | 1.90×10⁻¹⁶ | 0.02 |
| 10⁶ | 8.6×10⁻¹⁷ | 1.90×10⁻¹⁶ | 0.45 |
| 10⁸ | 1.03×10⁻¹⁶ | 1.90×10⁻¹⁶ | 0.54 |
| 10¹⁰ | 3.6×10⁻¹⁸ | 1.90×10⁻¹⁶ | 0.02 |
| 10¹² | 1.03×10⁻¹⁶ | 1.90×10⁻¹⁶ | 0.54 |

Never exceeded, and the delivered absolute error is flat in ω — exactly the behaviour the paper
predicts in prose on p. 30. On a wider set (`e29_paperbound.py`, adding
`∫₋₁¹ cos(λ atan x)/(1+x²) dx` at λ = 10⁴…10¹²) the worst ratio rises to **1.24**, i.e. the
bound is marginally exceeded. The module's comment at lines 99–103 records a 1.6×–36× margin on
an 18-cell test; on 30 cells it fails once. `_LEVIN_PHASE_ERROR_SAFETY` exists precisely to be
retuned — **raise it to 4–8**.

### 3.3 C4 (high) — `theta_scale = TWO_PI` is wrong, and for a subtler reason than it looks

Line 521 sets `theta_scale = TWO_PI` whenever `theta_mod_2pi` is supplied, on the reasoning that
a range-reduced phase hands `sin`/`cos` an `O(2π)` argument.

That reasoning is incomplete, and the measured consequence is severe. Same problem as above,
same code path, with `theta_mod_2pi = math.remainder(ωx, 2π)` (`e08_bound.py`):

| ω | delivered abs. error | reported floor (`2π`) | optimistic by |
|---|---|---|---|
| 10⁴ | 2.6×10⁻¹⁷ | 6.7×10⁻¹⁸ | 3.9× |
| 10⁶ | 9.8×10⁻¹⁸ | 1.39×10⁻²¹ | 7.0×10³ |
| 10⁸ | 3.9×10⁻¹⁷ | 1.39×10⁻²³ | 2.8×10⁶ |
| 10¹⁰ | 4.5×10⁻¹⁷ | 1.39×10⁻²⁵ | 3.2×10⁸ |
| 10¹² | 7.7×10⁻¹⁷ | 1.39×10⁻²⁷ | **5.5×10¹⁰** |

Note also that the reduced path is **not more accurate than the raw path** here (both sit at
~10⁻¹⁶ absolute), consistent with the campaign's §3 finding that on generic endpoints all
reduction modes tie.

The obvious diagnosis — "the product `ωx` is formed in binary64 before reduction, so
`dθ ≈ eps·|θ|`" — is only half the story. Repeating the experiment with an **exactly** reduced
phase (residue computed in 60-digit arithmetic from the double-precision `x`, then rounded;
`e28_reduction.py`) leaves the delivered error essentially unchanged:

| ω | raw | reduced in binary64 | reduced exactly | reduced exactly + 10⁻¹² fit error |
|---|---|---|---|---|
| 10⁶ | 9.9×10⁻¹⁷ | 3.9×10⁻¹⁸ | 1.0×10⁻¹⁸ | 1.1×10⁻¹⁸ |
| 10⁹ | 1.2×10⁻¹⁶ | 1.0×10⁻¹⁷ | 1.1×10⁻¹⁷ | 1.1×10⁻¹⁷ |
| 10¹² | 1.2×10⁻¹⁶ | 9.7×10⁻¹⁷ | 8.6×10⁻¹⁷ | 8.6×10⁻¹⁷ |

So the residual floor is **not endpoint-phase rounding**. It is the round-off floor that
Chen et al. actually derive, eq. (151):

> `|I₁ − I| ≲ ε (1 + G₁/G₀ + max{G₁,k²}/G₀) · |W| min(1, 1/|W|) · ‖f‖_{L∞}`

With `G₀ = min|g′|`, `G₁ = max|g′|` on the rescaled interval, the bracket tends to a small
constant when `g′` is roughly constant and large, and `|W|min(1,1/|W|) ≤ 1`, leaving
`≈ ε‖f‖_∞`. For the table above, `ε·‖f‖_∞·(h/2) = 2.2×10⁻¹⁶ × 0.717 × 1 = 1.6×10⁻¹⁶` —
matching every measured value.

This also *derives* the campaign's empirical `eps·θ_max` relative bound: since
`|I| ~ ‖f‖_∞/θ′` and `θ_max = θ′h`, `ε‖f‖_∞h ≈ ε·θ_max·|I|`. And it explains why the raw-path
endpoint bound works numerically: `Σ|p| ≈ 2‖f‖_∞/θ′` and `theta_scale ≈ θ′b`, so
`eps·θ_scale·Σ|p| ≈ 2b·eps·‖f‖_∞` — the same quantity up to a geometric factor.

**Recommendation, in order of preference:**

1. **Implement eq. (151) as the round-off floor**, replacing the endpoint model. Everything it
   needs — `max|f|` on the grid, `min|θ′|`, `max|θ′|`, `h`, `k` — is already sampled, so it
   costs no extra evaluations of anything:
   ```
   round_off_err = C · eps · max_grid|f| · (h/2) · (1 + G1/G0 + max(G1, k²)/G0)
   ```
   Measured validity (`e29_paperbound.py`): across 26 resolved cells — three problem families
   × five decades of ω × two orders, excluding four `sinc` cells that are under-resolved as a
   single region, where the step-(4) residual and not round-off dominates — the worst
   true/bound ratio is **0.22** with `C = 1`, against **1.24** for the endpoint model. It is both valid and tighter, and it is the paper's own bound rather than a
   local invention. It is also **independent of whether the phase is range-reduced**, which
   removes the failure mode above entirely.
   *Caveat:* it bounds round-off only. Where the region is under-resolved the step-(4) residual
   dominates and `total_err = max(residual, round_off_err)` is still the right aggregate.
2. **Make phase accuracy part of the contract.** Add an optional `theta_abserr` (a scalar, or a
   callable, in radians) to the `theta` dict, and use it in place of `eps·theta_scale`. This is
   the only way to represent the error a *phase spline* introduces — `bessel_phase`'s
   construction accuracy is a property of the spline, invisible from inside the quadrature. The
   campaign measured a uniform 2×10⁻⁸ relative floor on all seven three-Bessel oracles that is
   attributable to the phase construction; the module cannot currently express it, and reports
   ~10⁻¹⁵ instead.
3. **Retain `TWO_PI` only when the phase function declares that it reduces exactly.** Never as
   a default inferred from the mere presence of `theta_mod_2pi`.

**Knock-on effect, and why C4 is scored "high" rather than "medium".** `phase_limited` (lines
1075–1092) is the branch that closes the tolerance trap (§0.2), and its first condition is
`phase_err > atol`. Understating `phase_err` by the oscillation count pushes the threshold at
which the guard arms down by the same factor. Measured (`e30_trapmod.py`) on
`∫₁¹⁰⁰ sin(10³x)/x dx`: the raw path arms it from `atol ≈ 10⁻²²`, the reduced path only from
`atol ≈ 10⁻³⁰`, and the reduced path pays ~2× the regions and solves in between. The window is
therefore widened by roughly eight decades of `atol` on exactly the path that
`three_bessel_integrals.py` and `QuadSourceIntegral.py` use. Fixing the floor arms the guard
where it belongs. (The guard is not the only thing that terminates the loop — the step-(4)
residual can itself fall below a tight `atol`, because it is blind to the phase error — which is
why the effect is a 2× cost widening rather than an unbounded one.)

### 3.4 The aggregate

The linear (rather than in-quadrature) sum over regions is the right choice, and the comment at
lines 1168–1172 justifies it correctly: neighbouring regions share endpoints and a systematically
inaccurate phase produces a common drift, so the contributions are not independent.

Remaining gaps:

- `abserr_total` is never compared with the request (§1.4). Add `converged`.
- Nothing is reported for regions dropped by `p_use` (§1.5b).
- The aggregate mixes a resolution residual, a round-off floor and `quad`'s own estimate into a
  single scalar. Returning the components separately (`abserr_resolution`,
  `abserr_roundoff`, `abserr_fallback`) costs nothing and lets a caller see *why* they cannot
  get more digits — which is the actionable information.
- Callers discard it. `three_bessel_integrals.py:207` combines four phase groups as
  `(−G₁+G₂+G₃−G₄)/4` and keeps only `value`. Because that combination is cancellative, the
  *relative* error of the total is amplified by the cancellation factor, and only the summed
  `abserr` can reveal it. Propagating `abserr` through `quad_JJJ`/`quad_YJJ` should be part of
  any follow-up.

---

## 4. Efficiency

### 4.1 Where the time actually goes

Per subregion evaluation, microseconds, measured component by component (`e02_profile.py`) and
confirmed by `cProfile` on full runs (`e03_cprofile.py`):

| component | N=12 | N=24 | N=32 |
|---|---|---|---|
| `chebyshev_matrices` (cached base + rescale) | 2.4 | 2.8 | 2.9 |
| `f` sampling, Python loop (2N calls) | 11.7 | 20.6 | 26.8 |
| `θ′` sampling, Python loop (N calls) | 9.2 | 18.6 | 23.7 |
| **`np.block` assembly of `LevinL`** | **19.9** | **23.9** | **26.9** |
| `np.linalg.solve`, real 2N×2N | 8.2 | 17.4 | 24.4 |
| residual sanity check (extra matmul + 2 norms) | 4.6 | 4.4 | 5.4 |
| **total** | **≈56** | **≈88** | **≈110** |

**At the default order the linear solve is 15% of the cost and `np.block` alone is 35%.** In the
`cProfile` view of a 380-solve run, `numpy._core.shape_base` (`_block_check_depths_match`,
`_block_dispatcher`, `_block`) plus its `concatenate` calls account for ~21% of total time
against 6% for `numpy.linalg.solve`. This inverts the premise behind the campaign's
recommendation 7.

### 4.2 The main optimisation: solve the complexified `N×N` system

For the `(sin θ, cos θ)` basis the real `2N×2N` system is the realification of a complex `N×N`
one. With `q = p₁ + i p₂`:

```
p₁′ − θ′p₂ = f₁ ,  p₂′ + θ′p₁ = f₂     <=>     (D + i θ′) q = f₁ + i f₂
I_region = Im[ q(b) e^{iθ(b)} − q(a) e^{iθ(a)} ]
```

This is the paper's own formulation, (168) and (171). It is exactly equivalent, not an
approximation: verified (`e01_struct.py`) that the two systems have **identical condition
numbers** and that their minimum-norm least-squares solutions agree to ≤7.5×10⁻¹⁶ across phase
spans from 5 to 600 rad and N = 12, 24. It halves the flop count (complex `N³` LU ≈ `8N³/3`
against real `(2N)³` LU ≈ `16N³/3`), halves memory traffic, and makes assembly trivial — the
operator is `D` with `iθ′` added on the diagonal, so `np.block` disappears.

Measured per-operation (`e02_profile.py`), real 2N against complex N:

| N | `solve` | `lstsq` |
|---|---|---|
| 12 | 8.2 → 5.8 µs (1.42×) | 88.4 → 42.3 µs (2.09×) |
| 24 | 17.4 → 9.2 µs (1.89×) | 220 → 115 µs (1.91×) |
| 32 | 24.4 → 13.8 µs (1.77×) | 404 → 193 µs (2.09×) |

Measured end to end (`e12_fast2.py`), as a drop-in replacement for
`_adaptive_levin_subregion_impl` combining complexification, assembly into a preallocated
buffer, and `np.fromiter` sampling:

| problem | order | current | fast | speedup | \|Δvalue\| |
|---|---|---|---|---|---|
| sinc, rtol 10⁻¹³ | 12 | 0.26 ms | 0.17 ms | **1.47×** | 0 |
| peaked amplitude | 12 | 3.61 ms | 2.52 ms | **1.43×** | 1.7×10⁻²⁴ |
| chirp, rtol 10⁻¹³ | 12 | 0.28 ms | 0.19 ms | **1.47×** | 3.3×10⁻²⁴ |
| GRZ λ=10⁴ | 12 | 0.28 ms | 0.20 ms | **1.39×** | 0 |
| peaked amplitude | 24 | 3.35 ms | 1.88 ms | **1.78×** | 1.7×10⁻²⁴ |
| sinc | 32 | 0.38 ms | 0.24 ms | **1.60×** | 0 |

**1.4–1.8× across the board, with results identical to ≤3.3×10⁻²⁴.** This is the single
largest measured win available, it is the paper's own formulation, and it simultaneously
simplifies the code. It applies only to the two-dimensional `(sin, cos)` basis, so the generic
`m`-component path should be retained for the future bases sketched in the review's §8.3.

### 4.3 Smaller items, with numbers

- **Vectorised sampling.** `f` sampling drops from 11.7 µs to 2.5 µs at N=12 (and stays ~2.7 µs
  at N=64, against 50.5 µs for the loop) when the callables accept arrays. Worth adding as an
  opt-in (`vectorised=True`) or by one-time detection at entry, per the review's §7.6.
- **The residual sanity check costs 4.6 µs against the 8.2 µs solve it guards** — 56%. The
  code's own comment (lines 682–685) says the phase span is the primary guard and the residual
  is only "a secondary guard against non-finite values and outright failure". A bare
  `np.isfinite(q).all()` would discharge that purpose at ~0.3 µs. Worth an A/B before removing,
  but the cost/benefit is poor as it stands.
- **Fixed per-call driver overhead ≈ 80 µs**: a minimal single-region call takes 248 µs for
  three solves that account for ~168 µs. `uuid4()` is 2.3 µs of it; the rest is dict/list/object
  construction, the two `raw_theta` gate calls, `time.time()`, sorting, and the two health-check
  loops. That is **32% of a small call** — and `three_bessel_integrals.py` makes four calls per
  integral. Cheap fixes: generate the uuid lazily (only when a message is emitted), skip the
  `phase_diff` calls per §2.2, and build the diagnostics portion of the return dict only when
  asked.
- **Import cost**: 1.13–1.36 s of the 1.57 s module import is `seaborn` + `matplotlib`.

### 4.4 Rank-revealing QR (paper Remark 2): the payoff is bounded, and smaller than it looks

Remark 2 reports RRQR ~5× faster than the truncated SVD. But the LU fast path already removes
`lstsq` from most regions. Measured true split (`e24_solver2.py` — the returned
`num_direct_solves` under-counts, see §1.10):

| problem | k | solves | LU | `lstsq` | `lstsq` share |
|---|---|---|---|---|---|
| sinc | 12 | 3 | 3 | 0 | 0% |
| lorentz peak | 12 | 95 | 71 | 24 | 25% |
| lorentz peak | 32 | 79 | 71 | 8 | 10% |
| gauss peak | 12 | 43 | 43 | 0 | 0% |
| log-x Bessel-like | 12 | 15 | 15 | 0 | 0% |

So RRQR can address at most 25% of solves on the worst of these problems, and complexification
(§4.2) already halves `lstsq`. **Do complexification first, then re-measure the `lstsq` share
on the real three-Bessel integrands before writing any pivoted-QR-plus-rank-determination
code.** The module's existing comment on this (lines 703–713) reaches the same conclusion for
the same reasons and should stand.

### 4.5 Chebyshev order economics

The campaign's recommendation 5 ("raise the default above 12") is supported on subdividing
problems and not universally. Measured wall time on the real module (`e19_order.py`):

| problem | k=8 | k=12 | k=16 | k=24 | k=32 | k=48 |
|---|---|---|---|---|---|---|
| sinc (1 region) | 0.24 | 0.26 | 0.27 | 0.31 | 0.35 | 0.47 ms |
| lorentz peak | 14.88 | **9.75** | 10.83 | 12.28 | 12.19 | 11.59 ms |
| gauss peak | 6.63 | 3.80 | 3.01 | 2.69 | **2.00** | 2.72 ms |
| log-x Bessel-like | 2.14 | 1.61 | 1.34 | **0.47** | 0.56 | 0.78 ms |

Regions fall monotonically with order (lorentz peak: 45 → 18), so on problems that subdivide the
higher orders do strictly less work — up to **3.4× faster** at k=24 on the log-x problem, and
1.9× at k=32 on the gaussian peak. On problems that converge in one region, higher order simply
costs more. Order 8 is bad everywhere (and on lorentz peak it pushes 12 regions into the
fallback path where k=12 pushes only 4).

**Recommendation:** raise the default to 16 (a win on two of four problems, a 4% loss on the
one-region case, a 10% loss on lorentz peak), keep the `_LEVIN_MINIMUM_ALLOWED_ORDER = 8` floor,
document that 12–32 is the useful band and that the optimum depends on how much the amplitude
subdivides, and note that odd orders are preferable if §2.2's nested Clenshaw–Curtis fallback is
adopted (nesting needs `N−1` even). Not worth more than that: the effect is ±2× and
problem-dependent, and `three_bessel_integrals.py`'s hard-coded 64 is worth re-tuning on its
own integrands.

### 4.6 Measured and not worth doing

- **Recovering the Levin solve discarded when a child region falls to the fallback branch.**
  Measured (`e19_order.py`): 4% of solves on `lorentz peak`, 0% on three other problems.
  §2.2's restructure removes it anyway.
- **p-refinement** (§2.3): 1.5–2.2× more integrand evaluations.
- **Global error balancing** (§2.3): unsound with the step-(4) residual.

---

## 5. Prioritised recommendations

Ordered by (severity or measured payoff) ÷ effort.

| # | Change | Why | Effort |
|---|---|---|---|
| 1 | Check `f_Cheb`, sampled `θ′` and the solved `p` for finiteness and raise; make `p_use` reject non-finite ratios (C1) | eliminates a silent-wrong-answer path; the module currently certifies a partial integral at machine precision | trivial |
| 2 | Compare `abserr_total` with `max(atol, rtol·\|val\|)`; return `converged` and warn (C3) | the returned error bar is currently never checked against the request | trivial |
| 3 | Fix `max_depth` to update on every popped region (C6) | re-arms the non-convergence health check | trivial |
| 4 | Reject `atol <= 0`, `rtol < 0`, `depth_max < 0`, `len(f) != 2`, `len(x_span) != 2`; add a docstring (C8, C9) | `atol=0` gives 2²⁰ regions; a 3-tuple span is silently truncated | trivial |
| 5 | Replace the endpoint phase floor with eq. (151) (`C·eps·max\|f\|·(h/2)·(1+G₁/G₀+max(G₁,k²)/G₀)`), and add an optional `theta_abserr` to the phase contract (C4) | reported floor is optimistic by up to 5.5×10¹⁰ on the range-reduced path; eq. (151) is valid with 4.5× margin on 30 cells and needs no new evaluations; arms `phase_limited` ~8 decades of `atol` earlier on the production path | small |
| 6 | Complexify the `(sin, cos)` solve to `N×N`; assemble into a preallocated buffer instead of `np.block` (§4.2) | 1.4–1.8× end to end, bit-identical, and it is the paper's own formulation | small |
| 7 | Restructure the subregion routine to sample once → gate on `phase_span` (total variation) → nested Clenshaw–Curtis fallback → feed the fallback estimate into the same accept/bisect logic (C2, C7, §2.2) | fixes a 1590%-error case; bounded fallback cost with a real error estimate; drops the `simple_quadrature` dependency and its 1.6× wrapper overhead | medium |
| 8 | Distribute `atol` in proportion to interval length (C3 fix 2) | makes `atol` a global tolerance; measured free at matched delivered accuracy | small |
| 9 | Gate `p_use` on endpoint magnitudes rather than the mean, and add the discarded endpoint contribution to the region `abserr` (C5) | removes the value's dependence on `rtol` from the error budget's blind spot | small |
| 10 | Raise `_LEVIN_PHASE_ERROR_SAFETY` to 4–8 (§3.2) | the bound was exceeded by 1.24× in one of 30 cells at the current factor of 1.0 | trivial |
| 11 | Count `direct_solve`/`SVD_errors`/`num_order_changes` from `dataL`/`dataR` too; rename `evaluations`; return `abserr` components (C10, §3.4) | diagnostics under-count by ~3×, which is why the fast path's coverage was not measurable from the API | small |
| 12 | Move `seaborn`/`matplotlib` behind `emit_diagnostics`; parameterise the diagnostics and failure-dump paths; route messages through `logging` (C11) | 1.2 s of import time per worker; failure dumps land in arbitrary cwds and collide between workers | small |
| 13 | Raise the default order to 16; document the 12–32 band (§4.5) | up to 3.4× on subdividing problems, small loss elsewhere | trivial |
| 14 | Optional vectorised sampling (§4.3) | 4.7× on the sampling component at N=12, 18× at N=64 | small |
| 15 | Propagate `abserr` out of `quad_JJJ`/`quad_YJJ` (§3.4) | the four-group combination is cancellative, so only the aggregate error reveals precision loss | small |
| 16 | Tests: non-monotonic phase, non-finite amplitude, `theta_mod_2pi` and `theta_deriv` paths, deliberate fallback regions, reversed span, `atol=0`, and an `abserr`-vs-truth assertion on every problem (C12) | none of these are covered; `_GRZIntegral(10.0)` never reaches Levin at all | medium |

Recommendations 1–4 and 10 are one-line-scale changes that between them close the two critical
findings' *reporting* gaps and both trivial-severity classes; 5–8 are the substantive work.

---

## 6. What this audit did **not** examine

- `LiouvilleGreen/bessel_phase.py`, `phase_spline.py`, `range_reduce_mod_2pi.py`. The review's
  §7.1 initial-condition defect (`/ m` should be `/ sqrt(m)`) and the campaign's
  recommendation 2 (delete the bespoke range reduction) are unaddressed here and remain live.
  §3.3's `theta_abserr` recommendation implies work in `phase_spline.py` to report its own
  fit accuracy.
- The `_write_progress_data` diagnostics path beyond noting its inconsistencies (§1.11).
- Whether the four-phase-group decomposition in `three_bessel_integrals.py` is optimal.
- Thread/process safety under Ray beyond the cwd-relative file writes.

---

## 7. Reproduction

All measurements used the repository virtualenv (`./venv`, Python 3.12, NumPy 2.2.4,
SciPy 1.15.2, mpmath) with `PYTHONPATH=.` from the repository root, on the machine's idle
state. Scripts (throwaway, written to a scratch directory; none are in the repository):

| script | what it establishes |
|---|---|
| `e01_struct.py` | `D` correctness on the descending grid; reversed span; exact equivalence and equal conditioning of the real 2N and complex N systems |
| `e02_profile.py` | component-by-component cost of a subregion evaluation; real-vs-complex solve and `lstsq` timings |
| `e03_cprofile.py` | `cProfile` confirmation that `np.block` dominates `np.linalg.solve` |
| `e04_edge.py` | non-monotonic-phase failure; `max_depth` reporting; `atol=0` runaway |
| `e05_trap.py`, `e06_modtrap.py`, `e30_trapmod.py` | tolerance trap: closed at `HEAD`, with the guard arming ~8 decades of `atol` later on the reduced path |
| `e07_phaseerr.py`, `e08_bound.py` | reported-vs-delivered error across the ω ladder for both phase modes |
| `e09_cc.py`, `e10_fallback.py` | Clenshaw–Curtis nesting and accuracy; `quad` evaluation counts; `simple_quadrature` wrapper overhead |
| `e11_fastimpl.py`, `e12_fast2.py` | end-to-end A/B of the complexified implementation |
| `e13`–`e16` | policy A/B: p-refinement, global error balancing, length-proportional tolerance |
| `e17_puse.py`, `e18_puse2.py` | `p_use` perturbation bound and `rtol`-dependence of the value |
| `e19_order.py` | order economics; discarded-cached-solve fraction; fixed per-call overhead |
| `e20_valid.py`, `e21_nan.py`, `e22_nan2.py` | input validation matrix; silent-zero mechanism |
| `e24_solver2.py` | true LU/`lstsq` split versus the reported counter |
| `e26_oracle.py`, `e27_tvgate.py` | independent 79 578-panel oracle for the stationary-phase integral; total-variation gate verification |
| `e28_reduction.py`, `e29_paperbound.py` | exact-versus-binary64 range reduction; eq. (151) as an implementable floor |

**Caveats stated plainly.**

- Timings are single-machine, single-run wall clock with repetition counts chosen to give
  ≥0.25 s per measurement; they are stable to roughly ±5% between repeats but should not be
  quoted to two significant figures.
- The policy A/B in §2.3 uses a prototype driver that shares the module's Chebyshev machinery
  and Levin core but reimplements the adaptive loop. It is a fair comparison *between policies*;
  its absolute numbers are not the module's.
- Where no closed form exists the reference is a high-effort run of the same core (order 48–64,
  `atol` 10⁻²⁴, floor disabled). That is legitimate for comparing policies and not for
  establishing absolute accuracy. Every absolute-accuracy claim in §1 and §3 rests on a closed
  form or on the independent panel-sum oracle of §1.3.
- Several `mpmath` cross-checks attempted during this work were themselves unreliable at high
  frequency (`mp.quad` at `maxdegree=14` cannot resolve 10⁵ oscillations). One intermediate
  figure derived that way was discarded and replaced by the panel-sum oracle in §1.3; no
  `mp.quad`-derived number for an unresolved oscillatory integral is quoted above.
- §4.5's order recommendation is a ±2×, problem-dependent effect on four problems. It is the
  weakest-supported item in §5 and should be re-measured on the real three-Bessel integrands
  before being treated as settled.
