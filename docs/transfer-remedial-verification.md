# Transfer-remedial campaign verification and close-out

**Campaign:** [`prompts/transfer-remedial/README.md`](../prompts/transfer-remedial/README.md)
**Status board:** [`prompts/transfer-remedial/IMPLEMENTATION_STATE.md`](../prompts/transfer-remedial/IMPLEMENTATION_STATE.md)
**Design:** [`prompts/transfer-remedial/DRAFT-PLAN.md`](../prompts/transfer-remedial/DRAFT-PLAN.md) (revision 2, not edited by this campaign)
**Reconciliation:** [`prompts/transfer-remedial/RECONCILIATION.md`](../prompts/transfer-remedial/RECONCILIATION.md)
**Prompt:** [`prompts/transfer-remedial/09-benchmark-and-docs.md`](../prompts/transfer-remedial/09-benchmark-and-docs.md)
**Date:** 2026-09-10

---

## 1. What this document is

Nine prompts replaced `LiouvilleGreen/bessel_phase.py`'s Liouville–Green amplitude–phase
construction for the Bessel functions $J_\nu$, $Y_\nu$ — the ODE-and-root-solve construction that
integrated a normalized phase $Q=\theta/x$, matched a spurious offset `phi`, and splined the full
phase in chunks — with a two-region representation: a sampled near region below a remainder-tested
crossover $x_\star$, and a closed-form asymptotic tail above it. This document collects, in one
place and without requiring a reader to open any prompt log: what shipped in each commit, the
acceptance table with achieved values, the attribution of every downstream fixture number to its
actual cause, the cost/domain result, the remaining error floors, and the two cross-campaign
hand-offs this campaign could not act on itself.

**Headline result.** The Bessel oracle's envelope-normalized phase-pair error $E_\theta$ falls by
six to eight orders of magnitude at the orders and arguments production uses, and a hard
construction failure above $x\approx2.5\times10^{15}$ (SciPy/Amos `jv`/`yv` losing
argument-reduction accuracy, which drove the old ODE's adaptive stepper to stall) is removed
because the tail never evaluates a Bessel routine there at all. Every load-bearing design fact in
`README.md` §2 survived nine prompts unchanged: the residual algebra is exact (§2(a)), one series
supplies both the tail phase and amplitude via the Wronskian (§2(b)), branch tracking through ~90
cycles at the highest supported order is verified rather than assumed (§2(d)), and the split
evaluation removes an $O(1)$ correctness failure at $x\gtrsim10^{12}$, not merely an accuracy one.

---

## 2. What changed, per prompt

| # | Commit | Subject | What it did |
|---|---|---|---|
| 01 | `f71401d` | Add independent Bessel references and a measurement harness | New `LiouvilleGreen/tests/bessel_reference.py` (three independent reference tiers: exact half-integer closed forms, `scipy`, cached 70-digit `mpmath` corners) and the campaign's error metrics. No production code touched. |
| 02 | `fe33e8e` | Pin the SciPy Bessel domain boundaries as tests | New `LiouvilleGreen/tests/test_scipy_bessel_domain.py`, pinning four measured facts about `hankel1e`/`jv`/`yv`'s Amos-library failure boundaries as regression tests. No production code touched. |
| 03 | `ffbb36c` | Add the closed-form Bessel tail and its crossover test | New `LiouvilleGreen/bessel_tail.py`: the DLMF 10.18.18 asymptotic series (phase and, via the Wronskian, amplitude) and a remainder-tested crossover $x_\star(\nu,\text{budget})$. Corrected a factor-of-3 transcription error in the plan's third series coefficient (§3 below). Not yet consumed by production. |
| 04 | `bc31493` | Add the branch-tracked near-region Bessel sampler | New `LiouvilleGreen/bessel_near_region.py`: samples the exponentially scaled Hankel function below $x_\star$, branch-tracks the residual through as many as ~90 cycles at $\nu=1000.5$, and adaptively refines on two-sided panels. Not yet consumed by production. |
| 05 | `f6cbb29` | Rebuild the Bessel phase from a two-region amplitude and residual | Rewrote `LiouvilleGreen/bessel_phase.py` (293 → 1010 lines) to stitch prompts 03–04 into the public `bessel_phase()` entry point: the ODE, `phi`'s root solve, and the chunked full-phase spline are gone; split sin/cos evaluation and a declared `theta_abserr` are new. First commit to touch a pre-existing `LiouvilleGreen/` file. |
| 06 | `f9cc891` | Adapt the Bessel phase interface to its consumers | Migrated `main.py`'s two production Bessel-construction call sites and the live `QuadSourceIntegral_debug` diagnostic to the new interface; removed the now-meaningless `Q` member (raises an informative `KeyError`); repaired `plot_besssel_phase.py`, which was already dead code before this campaign. First commit to reach outside `LiouvilleGreen/`. |
| 07 | `69c37a9` | Preserve the leading term in Bessel phase groups | Rewrote `LiouvilleGreen/three_bessel_integrals.py`'s phase-group assembly as $Kt+C+R(t)$ (leading coefficients combined before multiplying by $t$, residual from `phase.residual`), removing near-resonant cancellation of independently reconstructed large phases; every phase group now declares a `theta_abserr` to `AdaptiveLevin`. |
| 08 | `8ba9159` | Tighten the Bessel tests to the new accuracy | Re-tightened `LiouvilleGreen/tests/test_bessel_phase.py` and `test_3bessel_analytic.py`, and the comments/tolerance constants (only) of `ComputeTargets/tests/test_tk_source_functions.py` and `test_phase_groups.py`, to the accuracy the new construction actually delivers; produced the before/after attribution table this document reproduces in §5. |
| 09 | *(this commit)* | Record the measured outcome of the Bessel rebuild | This document, the `docs/lg-phase-and-handover-followup-2026-09.md` supersession, the re-run benchmark tier, and the cross-campaign hand-off. Documentation and one benchmark run; no production code or test touched. |

Every SHA above resolves (`git cat-file -e <sha>`) on this branch.

---

## 3. Corrections to `DRAFT-PLAN.md`, recorded here because the plan itself is not edited

`RECONCILIATION.md` §2 and campaign board issue `[00-plan-vs-tree-corrections]` record four places
where `DRAFT-PLAN.md` revision 2 does not survive contact with the tree or with measurement. None
of them are load-bearing on the design — the two-region construction, the closed-form tail, the
Wronskian amplitude and the split evaluation all stand — but an agent working from the plan text
alone would write a false justification into a docstring or a commit message, so they are recorded
here where the campaign's own documents live.

- **C1 — the performance motivation.** The plan (§4.7) claimed the old ODE construction took
  ">600 s (did not complete)" at $x_{\max}=10^{13}$ and framed the replacement as removing an
  $O(x_{\max})$ cost curve. Measured: the old build was **flat at 0.06–0.09 s across five decades**
  of $x_{\max}$ and stalled only above $x\approx2.5\times10^{15}$, where SciPy/Amos `jv`/`yv` lose
  argument-reduction accuracy, the ODE right-hand side $(2/\pi)/(xm)$ becomes O(1)-relatively
  noisy, and DOP853 at `rtol=5e-14` cannot pass its error test on noise and stalls. The correct
  claim is "**a hard cliff was removed**," not "a cost curve was removed" — see §6 below for the
  measured cost.
- **C2 — the residual's size at high order.** The plan (§1, §4.6) states the interpolated residual
  "never exceeds a cycle." Measured at $\nu=1000.5$: the residual spans **565.82 rad ≈ 90 cycles**
  across the near region. The correct justification for still removing the chunked full-phase
  spline (`phase_spline`) from this module is that
  $\varepsilon\lvert r\rvert_{\max}\approx1.3\times10^{-13}$ at the largest supported order — below
  every acceptance target — not that the residual is small. `IMPLEMENTATION_STATE.md` standing
  note 2 forbids writing "never exceeds a cycle" into any code, docstring or commit message in this
  tree, and none of this campaign's commits do.
- **C3 — `plot_besssel_phase.py`'s state at the start of the campaign.** The plan lists this script
  as a diagnostic to migrate. It could not run at all before this campaign started: it read a
  `data["x_min"]` key `bessel_phase` has never returned (the dict has `min_x`), and called
  `phase(x)`, which neither `phase_spline` nor the new `BesselPhaseFunction` defines `__call__`
  for. So "migrate it" was a repair-or-delete decision, not a mechanical re-point; prompt 06
  repaired it (see §2 above), and it now runs to completion.
- **C4 — the existing derivative test.** The plan's Stage 1 justifies the new measurement harness
  by citing only the 50 %-relative and $10^{-3}$ thresholds of `test_bessel_phase.py`. It does not
  mention that `test_phase_derivative` already contracted the phase derivative to $10^{-6}$ — the
  campaign's standing regression gate throughout, tightened further by prompt 08 (see §4).

A fifth, independent correction was found by prompt 03, not by the planning pass:

- **The third DLMF 10.18.18 coefficient.** `DRAFT-PLAN.md` §7.2 and prompt 03's own text give the
  third series coefficient's denominator as **15360**; the correct value, determined by fitting
  against a 120-digit `mpmath` residual and cross-checked against Abramowitz & Stegun 9.2.29's
  $(8x)^k$ grouping, is **5120** — a factor of three. The shipped `LiouvilleGreen/bessel_tail.py`
  uses 5120 (and the independently-derived fourth coefficient, 229376) and
  `test_coefficients_match_the_published_series` pins both so neither is ever "corrected" back to
  the plan's number. Had the plan's value shipped, the three-term series would have removed only a
  third of the two-term error at $\nu=5/2,x=125$ (1.769e-10 → 1.179e-10, instead of → 2.42e-14).

`DRAFT-PLAN.md` itself remains unedited, as the revision-2 review record (`RECONCILIATION.md` §4).

---

## 4. The acceptance table, achieved

`README.md` §6 reproduces `DRAFT-PLAN.md` §10's targets; this table adds the **achieved** value at
the $(\nu,x)$ where it occurred, from prompts 05 and 08's measurements against the independent
references prompt 01 built. "Achieved" figures are the worst case over nodes, midpoints, endpoint
intervals and the tail, i.e. the harder grid, not a cherry-picked point.

| Coverage | Target | Achieved | At $(\nu,x)$ |
|---|---|---|---|
| Orders $1/2,3/2,7/4,5/2$, existing domain through $10^7$ | $E_\theta,E_A\le10^{-11}$ | $E_\theta=2.484\times10^{-12}$, $E_A=2.545\times10^{-13}$ | $(5/2,\,x_\star=64.4688)$ — the series remainder, not interpolation |
| Ordinary phase derivative, low orders | rel. err. $\le10^{-9}$ | $5.090\times10^{-13}$ (1960× inside) | $(3/2,\,34.41)$ |
| Orders $\ge20.5$, through $\max(1000,10\nu)$ | $E_\theta,E_A,\theta'\le10^{-6}$ | worst anywhere $8.776\times10^{-12}$ ($1.1\times10^5$× inside); corner-scored (below the SciPy floor) $2.443\times10^{-13}$ | $(1000.5,\,9962)$ |
| Orders $\ge20.5$, structural | fails loudly; two-sided $a_\nu$ band; branch tracking *verified* | `AmplitudeBandError` on the exact `-0j` failure mode; negative control (fixed-density `unwrap`, $\nu=1000.5$) measures $E_\theta=1.9954$ (order unity) between samples while the shipped tracker measures $4.045\times10^{-11}$ with `wraps_tracked=89` (within $90\pm1$) and max branch advance $1.5635$ rad (< $\pi/2$ limit) | $(1000.5,\,56540)$ |
| Crossover to the tail | interpolant/series agree to budget at $x_\star$; first omitted term below budget; $x_\star$ recorded | phase agreement $2.5\times10^{-12}$, amplitude $\le1.4\times10^{-13}$ (factor ~4 inside the `safety=0.25` budget) at every order; $x_\star$ table in §6 | all seven orders |
| Supported domain boundary | fails loudly; no `-inf`/`-0j` reaches an interpolant | $\nu<0.5$, $\nu>0$ but $\mu-1<0$, `max_x`$>10^{16}$, and any `hankel1e` sample outside $[0.99,8.0]$ in $a_\nu$ all raise before reaching an interpolant | tested at $\nu=0.25$, `max_x`$=10^{17}$, `bessel_phase(100.5, 50.0)` |
| Phase groups near cancellation | absolute deriv. checks scaled to constituent freq.; independent group reference; no division by a vanishing group derivative | $\lvert\delta\Theta\rvert$: $2.86\times10^{-5}\to1.42\times10^{-13}$ at exact resonance ($2.0\times10^8$×); scaled derivative metric $1.45\times10^{-14}$ for both routes (below the $10^{-11}$ acceptance) | $K=0$ exactly, $(\mu,\nu,\sigma)=(1/2,3/2,5/2)$ |
| Selected large arguments through $10^{15}$ | independent split-evaluation checks against `mpmath` at identical arguments | naive `sin(x+d)`: $4.725\times10^{-2}$ at $x=10^{15}$; split evaluation: $\le2.22\times10^{-16}$ at every tested argument through $10^{15}$ | $(3/2,\,10^{15})$ |

No row was missed and no row is reported as "not met."

---

## 5. The attribution table (from prompt 08 §5, verbatim)

One row per fixture comparison, on identical grids on both trees. "Bessel oracle" means improved
by this campaign; "consumer re-spline" means the $h^4$ fit of a sampled fixture on the production
grid, which this campaign does not touch; "LG truncation" is physical; "quadrature" and "reference
value" are what they say.

| comparison | error before | error after | attribution |
|---|---:|---:|---|
| `test_bessel_phase` `_test_bessel_value` $E_\theta$, $\nu=3/2$ | 2.005e-06 | 5.222e-13 | **Bessel oracle** (the residual is now the tail series remainder at $x_\star$) |
| `_test_bessel_value` $E_\theta$, $\nu=5/2$ | 2.903e-06 | 7.546e-13 | **Bessel oracle** |
| `_test_bessel_value` $E_A$, $\nu=3/2$ | 1.404e-13 | 5.396e-14 | Bessel oracle, but it was never the limit here — both trees sit near the amplitude sampling floor and the gain is a factor 2.6, not six orders |
| `_test_bessel_value` $\theta'$, $\nu=3/2$ | 5.150e-08 | 1.081e-13 | **Bessel oracle** |
| `test_high_order` $E_\theta$, $\nu=1000.5$ | 2.036e-06 | 4.956e-12 | **Bessel oracle** |
| `test_phase_derivative`, $\nu=100.5$, from $x_0$ | 6.388e-05 | 3.643e-13 | **Bessel oracle** (the old value violated the test's own $10^{-6}$ contract; the old lower bound hid it) |
| `test_bessel_phase` cached corners, $\nu=1/2$ through $10^{15}$ | 1.509 | 1.110e-16 | **Bessel oracle** — specifically the split evaluation; the old failure is `eps*theta` rounding of `sin(x+d)` |
| `test_bessel_J_integral`, $\nu=5/2$, $[5,100]$ | (1e-3 threshold; 8.894e-12 vs truth) | 8.894e-12 | **quadrature** — and, before the reference-value correction, **reference value** at 1.5e-10 |
| `test_3bessel_analytic` J110 at the fixed triple | 4.615e-08 | 5.857e-12 | **Bessel oracle** |
| `test_3bessel_analytic` J220 / J222 / J231 / Y022 | 9.6e-09 – 3.3e-08 | 2.5e-14 – 1.5e-13 | **Bessel oracle** |
| `test_3bessel_analytic` J000 / Y000 | 1.6e-08 / 2.2e-08 | 1.397e-10 / 4.771e-11 | **Bessel oracle** for the improvement; the residual is **quadrature** — `DEFAULT_3BESSEL_CHEBYSHEV_ORDER=12`, worth three further orders |
| `test_3bessel_analytic` reported `abserr` bounds truth | 2 of 7 | 7 of 7 | **Bessel oracle** (prompt 05 dropped the true error) plus the declared `theta_abserr` (prompt 07, +0–1.9% on the reported value) |
| `test_3bessel_analytic` singularity bands, $\epsilon\ge10^{-3}$ | *(not measured before)* | 1e-11 – 1e-09 rel | **quadrature** at this $\epsilon$, and `DEFAULT_3BESSEL_CHEBYSHEV_ORDER` on the $(0,0,0)$ oracles |
| `test_3bessel_analytic` singularity bands, $\epsilon=10^{-10}$ | *(not measured before)* | 2.794e-04 rel | **genuine near-singular behaviour** — the error grows an order per order of $\epsilon$, which no 5e-12 rad phase can produce |
| `test_tk_source_functions` `err_M`, $w=1/3$ | 1.268e-13 | 1.272e-13 | **consumer re-spline** (amplitude path) |
| `test_tk_source_functions` `err_T`, $w=1/3$ | 3.021e-08 | 3.021e-08 | **consumer re-spline** (phase path, $h^4$) |
| `test_tk_source_functions` `err_scipy`, $w=1/3$ | 1.985e-06 | 3.021e-08 | **Bessel oracle** removed; what remains *is* `err_T`, i.e. **consumer re-spline** |
| `test_tk_source_functions` `err_scipy`, $w=0.2$ | 1.550e-06 | 2.234e-08 | as above |
| `test_tk_source_functions` [grid refinement] 100/decade | 6.090e-06 | 6.090e-06 | **consumer re-spline** |
| `test_tk_source_functions` `omega` vs `theta_deriv` (LG fixture) | 4.249e-08 | 4.249e-08 | **consumer re-spline** (a `phase_spline` derivative through the exact `omega` integral) |
| `test_tk_source_functions` $d\ln M/dz$ vs FD (exact envelope) | 8.528e-06 | 8.528e-06 | **LG truncation** |
| `test_tk_source_functions` numeric-region midpoints | 2.720e-04 | 2.720e-04 | **consumer re-spline** (the numeric $dT/dz$ cubic) |
| `test_phase_groups` Oracle 1, exact stand-ins | 6.669e-16 | 6.669e-16 | rounding; no Bessel function involved |
| `test_phase_groups` Oracle 2, exact stand-ins | 3.412e-14 | 3.412e-14 | rounding of `scipy`-built stand-ins |
| `test_phase_groups` realistic, $w=0.2$, 100/decade | 1.376e-04 | 1.375e-04 | **LG truncation** of `omega` and $d\ln M/dz$ |
| `test_phase_groups` realistic, $w=1/3$, 300/decade, $x_q>100$ | 1.661e-06 | 7.502e-09 | **Bessel oracle** — the only place in this module where it was the binding term |
| `test_phase_groups` realistic, $w=0.2$, 300/decade, $x_q>100$ | 5.710e-06 | 5.568e-06 | **LG truncation** |
| `test_phase_groups` composed phase, exact constituents | 1.776e-15 | 1.776e-15 | rounding of the composition arithmetic |
| `test_phase_groups` composed phase, `phase_spline` constituents | 3.782e-08 | 3.782e-08 | **consumer re-spline** — `phase_spline`'s own ~20$\varepsilon\lvert\theta\rvert$, out of scope |
| `test_phase_groups` exact-fixture seam, nodes | 1.514e-04 | 1.514e-04 | **LG truncation** |
| `test_phase_groups` exact-fixture seam, midpoints | 1.031e-03 | 1.031e-03 | **consumer re-spline** (numeric $dT/dz$) |

No row is attributed to "Bessel oracle" while barely moving, and no row that moved is attributed to
anything else. **The one clean downstream result to claim** is `test_phase_groups`'s $x_q>100$,
300/decade row at $w=1/3$ (1.661e-06 → 7.502e-09, a factor 221): it is the only place in either
`ComputeTargets` fixture where the Bessel oracle, rather than the consumer's own re-spline or
physical LG truncation, was the binding term. The honest framing is "the oracle stopped being the
limit," not "the fixtures got more accurate."

---

## 6. Cost and the domain boundary

**Do not sell this campaign on construction speed** (`IMPLEMENTATION_STATE.md` standing note 3).
The old ODE build was already $\sim0.06$–$0.09$ s, flat, across five decades of $x_{\max}$; the new
build is $\sim0.002$–$0.043$ s depending only on the order. A speed-up ratio between the two would
be noise (`RECONCILIATION.md` C1).

**What the replacement actually buys is a supported domain that no longer has a silent-failure
cliff.**

| | old construction | new construction |
|---|---|---|
| Cost vs. $x_{\max}$ | flat, $0.06$–$0.09$ s, $x_{\max}\in[10^{11},2\times10^{15}]$ | flat, depends only on $\nu$: $0.0022$–$0.0034$ s at $\nu=5/2$, $0.043$ s at $\nu=1000.5$, across $x_{\max}\in[10^3,10^{16}]$ |
| Largest $x_{\max}$ that completes | $2\times10^{15}$ (production tolerances; $3\times10^{15}$ and $8.6\times10^{15}$ time out after 60 s) | $10^{16}$ (the declared ceiling, `MAX_SUPPORTED_X`) |
| Mechanism at the boundary | SciPy/Amos `jv`/`yv` lose argument-reduction accuracy above $x\approx2.5\times10^{15}$; the ODE right-hand side $(2/\pi)/(xm)$ becomes O(1)-relatively noisy; DOP853 at `rtol=5e-14` cannot pass its error test on noise and stalls (never returns) | the tail series never evaluates `hankel1e`/`jv`/`yv` above $x_\star\sim100\nu$, so the noise is never sampled; $x=3\times10^{15}$ and $8.6\times10^{15}$ — both points where the old construction did not return — build in $\sim2.5$ ms |
| Crossover $x_\star$ (budget $10^{-11}$) | — | $\nu=3/2$: 34.41 (22.9$\nu$); $\nu=1000.5$: 58123 (58.1$\nu$) — **not** a fixed multiple of $\nu$; see the full table in `prompts/transfer-remedial/logs/03-closed-form-tail.md` |

$x_\star/\nu$ grows with order (from ~4.4 at the loose budget and low $\nu$ to ~58 at the tight
budget and $\nu=1000.5$) because the series' governing coefficient grows like $\nu^8$ while the
power of $x$ in the leading omitted term is fixed at 7 — `DRAFT-PLAN.md` §1's "fixed $\approx4.6$
e-folds" does not hold at any order tested.

### The benchmark tier (`docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py`)

See §7 below.

---

## 7. Benchmark tier re-run at $\kappa=1000$

**It completes.** `docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py`'s `production_scan`
capped `B1_KAPPA` at 100 because $\kappa=1000$ (three phases whose `max_x` reaches
$\approx2.3\times10^{15}$ for the largest momentum) did not complete within $\sim25$ min under the
old construction. Re-run against this tree, all seven oracles at $\kappa=1000$, $k,q,s=1300,1700,2100$,
`B1_MAX_X`$=10^{12}$ ($n_{\rm osc}=8.117\times10^{14}$ per oracle):

| oracle | phase build (3 phases) | Levin evaluation | total | rel. err. vs. closed form |
|---|---:|---:|---:|---:|
| J000 | 0.0002 s | 0.149 s | 0.150 s | 2.28e-11 |
| J110 | 0.0065 s | 0.314 s | 0.321 s | 3.25e-10 |
| J220 | 0.0060 s | 0.298 s | 0.304 s | 2.81e-11 |
| J222 | 0.0078 s | 0.386 s | 0.394 s | 5.06e-11 |
| J231 | 0.0081 s | 0.357 s | 0.365 s | 3.32e-12 |
| Y000 | 0.0001 s | 0.156 s | 0.156 s | 1.78e-11 |
| Y022 | 0.0052 s | 0.278 s | 0.283 s | 1.37e-11 |

**All seven together: under 2 s wall clock**, against "did not complete in $\sim25$ min" before.
`B1_KAPPA` now includes 1000.0; the historical claim is kept as history in the module comment,
with the corrected mechanism (`RECONCILIATION.md` C1): it was never that the phase construction
was slow (it was flat at $\sim0.1$ s across the whole range even before this campaign), it was that
SciPy/Amos `jv`/`yv` lose argument-reduction accuracy above $x\approx2.5\times10^{15}$, driving the
old ODE's adaptive stepper to stall. **The phase layer is not the limit at this tier — nor, at
these numbers, is the Levin core**: both complete in well under a second per oracle. Every
`converged=False` above reflects the requested Levin tolerance
(`QUAD_ATOL=1e-14, QUAD_RTOL=1e-10`, tighter than the measured true error in every row), not a
non-terminating or a failing evaluation — this is the same "declared error is honest, not
optimistic" behaviour §5's attribution table documents elsewhere in the campaign, not a new defect.

**A methodological finding, worth recording because it would silently mislead a future re-run.**
`bessel_tier.py:40` hardcodes `REPO = "/Users/ds283/Documents/Code/SecondaryGWKit"` — the main
checkout's absolute path — and inserts it into `sys.path` at import time. Running the script from
a **worktree** (as this campaign's every commit has been made from) therefore silently imports
`LiouvilleGreen` from the **main checkout**, not from the worktree whose code is actually being
measured, because `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` — the script's
other `sys.path` entry — resolves to `docs/adaptive-levin-benchmark`, which does not itself contain
a `LiouvilleGreen` package, so it supplies nothing to shadow `REPO` with. The first re-run attempt
for this section did exactly that: it reproduced the *old* construction's cliff (a `solve_ivp`
stall inside `bessel_phase.py`'s now-superseded ODE, confirmed by interrupting the process and
reading the traceback) because the main checkout (`main`, `9ff59d5` at the time) predates this
entire campaign. The number above is from a corrected invocation that pre-imports
`LiouvilleGreen.bessel_phase`, `LiouvilleGreen.three_bessel_integrals` and
`LiouvilleGreen.tests.test_3bessel_analytic` from this worktree before `bessel_tier` runs, so
Python's module cache resolves them here rather than in the shadowing path; `bessel_tier.py`
itself is **not** modified to fix this (out of this prompt's "one constant, one comment" scope,
and `REPO`'s hardcoding may be intentional for the benchmark's normal invocation from the main
checkout). Recorded as board issue `[09-bessel-tier-hardcoded-repo-path]` (`IMPLEMENTATION_STATE.md`
§3) so a future re-run of this specific tier from a worktree is not silently measuring the wrong
tree.

---

## 8. Remaining floors

The Bessel oracle's own error is no longer the binding term almost everywhere it is measured
directly (§5). What is left, named and quantified:

- **Consumer re-spline error.** `TkSourceFunctions` and the phase-group fixtures still sample the
  Bessel/LG phase onto a production grid and fit a cubic spline through it; that fit's own $h^4$
  error is now the largest term in most of §5's rows (`err_T` $=3.021\times10^{-8}$ at $w=1/3$; the
  numeric-region midpoint error at $2.720\times10^{-4}$). This campaign does not touch it —
  `ComputeTargets/` is out of scope (`README.md` §1.1, §4.2).
- **Physical Liouville–Green truncation.** The closed-form $\omega_{\rm eff}$ and $d\ln M/dz$ used
  by the LG branch are themselves an asymptotic approximation; `test_tk_source_functions`'s
  $d\ln M/dz$-vs-finite-difference row ($8.528\times10^{-6}$) and `test_phase_groups`'s LG-truncation
  attribution rows are this floor, unaffected by the Bessel oracle's accuracy.
- **The Levin quadrature and `DEFAULT_3BESSEL_CHEBYSHEV_ORDER=12`.** Prompt 08 found this constant
  now binds two of the seven three-Bessel closed forms (J000, Y000): raising the spectral order to
  20 buys them three further orders, while making the other five 4×–1500× *worse* (a conditioning
  effect of the higher-order Chebyshev fit). It is not simply too low — it is a per-integrand
  trade-off nobody has resolved (board issue `[08-3bessel-chebyshev-order-is-now-the-limit]`,
  still open).
- **Product rounding in a non-resonant three-Bessel phase group.** Prompt 07's $Kt+C+R(t)$
  restructure removes near-resonant cancellation but not the ordinary rounding of the product
  $K\cdot x$ itself: at $K=0.1,\,x=10^{12}$, $\lvert\delta\Theta\rvert=1.526\times10^{-5}$ rad — one
  ulp of $Kx$ — for **both** the new and the old route, so a generic (non-resonant) group gains
  nothing from the restructure. Accepted and recorded here rather than fixed (board issue
  `[07-generic-K-product-rounding]`, now closed on that basis): the remedy would be a two-product
  split of $K\cdot x$ (Dekker/Veltkamp, or `math.fma` on Python 3.13+) folding the low part into the
  small-angle correction, which the campaign's prompt 07 deliberately did not implement, since the
  prompt prescribed handing the unreduced product to libm and this would be a second, unrequested
  change to the same expression.
- **Input-coordinate error in $x=k\eta$ at large $x$** (`DRAFT-PLAN.md` §7.5). The representation
  can accurately evaluate Bessel functions at the supplied floating-point $x$; it cannot recover
  uncertainty already present in the caller's $x=k\eta$, nor undo a lossy `exp(log(x))` round trip.
  This is a property of the caller's arithmetic, not measured by this campaign, and is recorded here
  because at very large $x$ it can dominate the residual interpolation error by many orders of
  magnitude.
- **The SciPy/Amos boundaries.** `hankel1e` and `jv`/`yv` become unusable above
  $\approx2.247\times10^{15}$ for $\nu\lesssim85.5$ and $\approx7.13\times10^8$ for
  $\nu\gtrsim88.5$ (the order threshold between them is only *bracketed*, at $[85.5,88.5]$, not
  pinned — board issue `[01-scipy-jv-yv-high-order-boundary]`, still open). These are now
  **properties pinned by tests** (`LiouvilleGreen/tests/test_scipy_bessel_domain.py`,
  `bessel_reference.scipy_reference_max_x`), not assumptions, and the two-region construction stays
  at least three decades below them by design ($x_\star\sim100\nu$).

---

## 9. Environment

Python 3.12.14, SciPy 1.15.2, NumPy 2.2.4, mpmath 1.3.0, Darwin 25.5.0 (arm64) — recorded because
§8's SciPy/Amos boundaries are properties of the bundled Amos library, not guarantees, and this
record is what a future SciPy upgrade should be checked against
(`LiouvilleGreen/tests/test_scipy_bessel_domain.py`'s module docstring says so explicitly).

---

## 10. Deferred and handed-over work

From `README.md` §7, restated here as the closing record:

- Tightening the high-order accuracy target beyond $10^{-6}$ (e.g. for non-Limber angular power
  spectra, `test_bessel_phase.py:87-92`'s named use case) — needs only a denser near region and a
  node-count cost budget; the tail, the evaluation path and the consumers are unaffected.
- `phase_spline`'s chunking on its own terms, and its `_build_log_chunks_positive` progress guard —
  both latent, unrelated to this campaign's replacement, and now measured harsher for the
  cosmological case by `docs/gk-wkb-numerical-review-2026-09.md` §3.
- The `QuadSourceIntegral._three_bessel_Levin` phase-group and missing-`theta_deriv` findings —
  handed to `prompts/source-remediation` by this prompt; see §11.
- A validated integer cycle-count algorithm. `wraps_tracked` is a diagnostic of the constructed
  interval, not a certified cycle count; a consumer that genuinely needs one is separate design and
  validation work.
- The §5.3 hybrid (quadrature near the turning point, series in the tail) as a designated fallback
  for the near region only, if a future SciPy change ever degrades `hankel1e`'s phase there.

Two smaller items are also closed by documentation in this prompt, per the boards' own recorded
"next steps," rather than by code:

- **`[07-generic-K-product-rounding]`** — accepted and recorded above (§8), not fixed; see that
  section for the reasoning.
- **`[08-test-three-bessel-tolerances-unassigned]`** — `test_three_bessel.py`'s tolerances remain at
  their pre-campaign values. `README.md` §4's prose said prompt 08 would re-tighten this file, but
  neither the prompt's own file list nor `README.md` §3's per-prompt file column include it, and
  prompt 08's own acceptance criterion restricted its diff to four files. This is recorded as an
  explicit decision, not an oversight: a future prompt in this campaign (or a successor) may
  re-tighten it if the eight-order oracle improvement is judged to change its accuracy the way it
  changed `test_3bessel_analytic.py`'s.

Left genuinely open at campaign close (in `IMPLEMENTATION_STATE.md` §3, not resolved by this
prompt, because resolving them would mean touching a production module, a test, or a second entry
on another campaign's board that this prompt's own instructions restrict to one):

- `[01-scipy-jv-yv-high-order-boundary]` — the order threshold between the two Amos failure modes
  is bracketed at $[85.5,88.5]$, not pinned.
- `[04-achieved-estimates-exclude-the-sampling-floor]` — the near region's `achieved_*` estimators
  cannot see a systematic bias in `hankel1e` itself; not binding at any budget tested, but the
  3e-13 sampling-floor constant it carries has not been re-measured since prompt 04's rotation-fix.
- `[05-quadsource-order-check-docstring-stale]`, `[08-tk-fixture-scipy-comparison-unasserted]` —
  both anticipated riding alongside this prompt's `source-remediation` hand-off; this prompt's own
  text restricts that hand-off to one entry (§11), so both remain open here, undischarged.
- `[06-levin-theta-docstring-stale]` — two stale docstrings in `AdaptiveLevin/levin_quadrature.py`;
  that directory is forbidden to this campaign.
- `[06-measure-bessel-phase-num-chunks]`, `[06-three-bessel-plot-calls-a-non-callable-phase]` —
  both name files outside this prompt's "files you may touch" list.
- `[08-3bessel-chebyshev-order-is-now-the-limit]` — named and quantified in §8; the design decision
  (per-integrand order, or a convergence check) is unresolved.
- `[08-3bessel-plot-cost-dominates-the-suite]` — a proposal (gate the diagnostic plot grid behind a
  flag) was recorded, not implemented.

---

## 11. The hand-off to `prompts/source-remediation`

`RECONCILIATION.md` §3.2 and campaign board issue `[00-qsi-three-bessel-levin-excluded]` record a
finding this campaign deliberately did not act on, because `ComputeTargets/QuadSourceIntegral.py`
belongs to the `source-remediation` campaign and this campaign's rules forbid touching it
(`README.md` §1.1, §4.2):

`QuadSourceIntegral.py`'s `_three_bessel_Levin` (`:1175-1442`) makes **eight**
`adaptive_levin_sincos` calls whose phases are signed sums of three `bessel_phase` `raw_theta`
values (`:1226, :1267, :1308, :1349`), supplying **no `theta_deriv`** and no `theta_abserr`. So
`need_theta_Cheb` is `True` there and Levin obtains $\theta'$ by spectral differentiation of the raw
phase — exactly the route `LiouvilleGreen/three_bessel_integrals.py`'s phase-group assembly was
rewritten (prompt 07, above) to avoid, and it has the same near-resonant cancellation problem that
rewrite fixed in the sibling module.

**The gap survived `source-remediation`'s own rewrite of this file** (its prompts 08–10, landed as
`4afd531`/`ffc50ae`/`815217b`), and is now anomalous within it: the file's *own* new phase-group
route (`:1008`, `LEVIN_USE_THETA_DERIV=True` at `:93`) does pass `theta_deriv`, while these eight
calls do not, and the file's own comment at `:74` notes they are the odd one out. Checked before
filing (as this prompt's text requires): that campaign's board records only its item B6
(`atol`/`rtol` forwarding) against these call sites — not the missing derivative or the missing
declared error, which this hand-off is the first record of.

**There is a working pattern to copy, not a design to invent.**
`LiouvilleGreen/three_bessel_integrals.py`'s `_PhaseGroup` (prompt 07) supplies exactly this: it
forms $K=\sum_i\epsilon_i m_i$ and $C=\sum_i\epsilon_i c_{\nu_i}$ once (by `math.fsum`, before
multiplying by the shared variable $t$), takes $R(t)=\sum_i\epsilon_i r_i(m_it)$ from each
constituent's `phase.residual`, and its `levin_theta()` method returns the four-key dictionary
`{"theta", "theta_mod_2pi", "theta_deriv", "theta_abserr"}` — the last two from
`phase.residual_log_deriv`/`phase.theta_deriv` and the linear sum of each constituent's
`phase.theta_abserr_at` at its own argument. Measured in the sibling module, this restructure took
the group phase error from 2.86e-05 to 1.42e-13 rad at exact resonance (2.0e8× tighter) and made the
Levin-reported `abserr` for the seven three-Bessel closed forms bound the true error in all seven
cases, against two of seven before (see §5, and `docs/OPEN_ISSUES.md`'s note on
`[09-abserr-does-not-bound-phase-spline-floor]`, the `levin-refactor` campaign's own record of the
same symptom, now measured false on this tree for the sibling module).

**Recorded in two places**, per this prompt's instructions, because a finding in one campaign's
docs is invisible to the other's agents: here, and as a new entry
`[transfer-remedial-qsi-phase-groups]` in `prompts/source-remediation/IMPLEMENTATION_STATE.md` §3 —
the only edit this prompt makes to that file, adding one entry and changing nothing else.
