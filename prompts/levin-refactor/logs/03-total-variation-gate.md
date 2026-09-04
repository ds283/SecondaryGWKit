# Log 03 — Gate on total variation; replace `quad` with a nested Clenshaw-Curtis rule

**Prompt:** prompts/levin-refactor/03-total-variation-gate.md
**Commit:** Gate the Levin fallback on total variation, not net phase change (SHA intentionally
omitted — see README §5 rule 5)
**Date:** 2026-09-04
**Result:** COMPLETE

## What shipped

### The new region taxonomy

`_adaptive_levin_subregion_impl` now samples the phase (via `BasisData.build_Levin_data`,
unchanged) *before* deciding anything else, and gates on `phase_span` — the total variation
`mean|theta'| * width` that the module already computed but previously used only for the
LU-vs-lstsq conditioning gate — instead of on the net phase change `theta(b) - theta(a)` that used
to live in the driver's main loop. This single change is C2: the two quantities differ exactly
when `theta'` changes sign inside the region, which is exactly the geometry of a phase with an
interior stationary point.

Two outcomes, told apart by a new `"is_direct"` key in the dict `_adaptive_levin_subregion_impl`
returns:

- **Levin region** (`is_direct: False`, unchanged numerics from prompt 02): the complexified or
  realified linear solve, endpoint extraction, `p_ratios`/`p_use` gating, `_phase_error()` floor.
- **Clenshaw-Curtis region** (`is_direct: True`, new): computed by the new helper
  `_adaptive_levin_subregion_cc()`. It samples `f` and the basis `(sin theta, cos theta)` once on
  the `(2*chebyshev_order - 1)`-point extremal grid, forms the order-`N` and order-`(2N-1)`
  Clenshaw-Curtis rules from that one sample (the `N`-point grid is exactly every other node of the
  `(2N-1)`-point one — see `_cc_weights_base()` and the nesting test), and returns
  `value = CC_{2N-1}`, `"abserr_direct" = |CC_{2N-1} - CC_N|`. `p_sample`/`p_ratios` are both
  `None` — the Levin antiderivative concept does not apply to a direct rule.

New Clenshaw-Curtis weight machinery, next to `_chebyshev_base`/`chebyshev_matrices`:
`_cc_weights_base(N)` (cached like `_chebyshev_base`, the closed-form "clencurt" cosine-sum
construction — Waldvogel 2006 / Trefethen ch. 12 — specialised to the direct sum rather than the
FFT since `N` is at most a few dozen here) and `_cc_weights(x_span, N)` (the cheap rescale to an
arbitrary interval).

`_adaptive_levin` (the driver) no longer has a separate "is this region weakly oscillatory"
gate of its own, and no longer imports or calls `Quadrature.simple_quadrature`. Every region —
whether it is the "current" region being tested for acceptance, or one of the two eagerly-computed
"comparison" children a Levin region uses for its step-(4) residual — goes through the *same*
`_adaptive_levin_subregion()` call, which makes the Levin-vs-CC decision internally. The driver
then branches on `data["is_direct"]`:

- **CC region:** `abserr = data["abserr_direct"]` directly (no children needed — the nested pair
  *is* the error estimate). Tested against `atol`/`rtol` on the same terms as a Levin region
  (`resolved = abserr < atol or relerr < rtol`), with the same `phase_limited` logic. If resolved,
  accepted as `INTERVAL_TYPE_DIRECT` (so `num_simple_regions` keeps its meaning for the benchmark
  harness). If not, bisected: the two children are pushed with `estimate=None` (nothing to reuse —
  unlike the Levin branch, no comparison children exist to carry forward) and independently
  re-gated on their own, roughly-halved `phase_span` when they are popped.
- **Levin region:** unchanged — eager `dataL`/`dataR` comparison, step-(4) residual, bisect with
  the comparison estimates carried forward as the children's own `estimate` (as before prompt 03).

`phase_err` for a CC region is the direct analogue of the Levin endpoint bound: `_phase_error`
applied to `theta_scale` (reused from `build_Levin_data`, not recomputed) and
`f_scale * width`, where `f_scale` is now read off the fine-grid samples already computed for the
CC rule (`max` over nodes of `sum_i |f_i|`) rather than a separate 3-point sample the old
`quad`-branch used to take — one fewer set of wasted evaluations. This is explicitly a placeholder;
prompt 04 replaces both this and the Levin branch's `_phase_error()` call with the eq. (151) floor.

### Documentation

Module docstring's "DEVIATIONS FROM THE REFERENCE ALGORITHM" bullet on weakly-oscillatory regions
rewritten to describe the nested Clenshaw-Curtis fallback, the nesting property, and the
stationary-point argument for why gating on total variation (not net change) is the correct
structural response. `adaptive_levin_sincos`'s docstring updated to match (no more mention of
`scipy.integrate.quad`).

### Dead code removed

`_Basis_SinCos.phase_scale()` is deleted: its only caller was the old `quad`-branch's
`direct_phase_err` computation, which prompt 03 replaces with the CC branch's own computation
(reusing `theta_scale` from `build_Levin_data` instead of re-deriving the same quantity from
`raw_theta(a)`/`raw_theta(b)`). `raw_theta()` is kept — it remains a reasonable primitive on the
basis interface even though this commit removes its last call site inside this module — but see
*Observations not acted on* below.

## Numerical evidence

`PYTHONPATH=. ./venv/bin/python`, comparisons via `git stash`/`git stash pop` against baseline
`5d5b958` (prompt 02's commit). All values below were re-run against the final committed code.

### Item 2 — C2 case: the stationary-phase oracle

`I = int_0^1 exp(-x) sin(1e6 (x - x^2)) dx`, oracle `-6.879079716900e-04` (audit §1.3):

| | value | relative error | `num_regions` | `num_simple_regions` | reported `abserr` | true `abserr` |
|---|---|---|---|---|---|---|
| before (net-phase gate) | `+1.024944516387e-02` | `1.590e+01` (1590%) | 1 | 1 | 0.189 | — |
| after (total-variation gate) | `-6.87907972233e-04` | `7.89e-10` | 20 | 6 | `1.62e-11` | `5.43e-13` |

Relative error goes from 1590% to `7.9e-10` — better than the audit's own total-variation-gate-only
prototype (`1.1e-8`, achieved *without* step 5's bisect-on-miss), consistent with step 5 recovering
the extra digits the audit predicted it would. The reported `abserr` (`1.62e-11`) exceeds the true
error (`5.43e-13`) by a comfortable margin — the property the whole campaign exists to defend.
Reproduced in `test_stationary_phase_gate_uses_total_variation`
(`AdaptiveLevin/tests/test_levin_quadrature.py`).

### Item 3 — CC weight verification

Exact integration of every polynomial of degree `< N` (relative error `<5e-13`, machine-precision
noise) at `N = 8, 12, 13, 16, 25`, plus `int_0^1 exp(x) dx` (error `0.0` at `N >= 13` — the smooth
exponential is resolved to round-off well below the polynomial-exactness order). Weight sums equal
the interval width to `1e-12` in every case.

```
N=  8  max relative poly error (deg<8)  = 4.5e-16   exp(x) error = 1.03e-08
N= 12  max relative poly error (deg<12) = 4.6e-16   exp(x) error = 1.33e-14
N= 13  max relative poly error (deg<13) = 2.0e-16   exp(x) error = 0.0
N= 16  max relative poly error (deg<16) = 4.5e-16   exp(x) error = 0.0
N= 25  max relative poly error (deg<25) = 4.2e-16   exp(x) error = 0.0
```

Both checks are now in the test file (`test_cc_weights_exact_for_polynomials`,
`test_cc_weights_transcendental`), not just in this log.

### Item 4 — nesting

`chebyshev_matrices(x_span, 2*N-1)[0][::2] == chebyshev_matrices(x_span, N)[0]` exactly
(`max|diff| = 0.0`) at `N = 8, 12, 13, 16, 17, 25, 33` and spans `(0,1)`, `(-2,5)`, `(1, 1e5)` —
even and odd `N` both, and a span with a large dynamic range. In
`test_chebyshev_nesting`.

### Item 5 — fallback error estimate honesty (audit's CC13/CC25 table, reproduced)

`int_1^2 exp(-t)/(1+t) sin(W t) dt`, `W = span/(b-a)`:

| phase span | `CC13` err | `CC25` err | estimate `\|CC25-CC13\|` | estimate `>` `CC25` err |
|---|---|---|---|---|
| 0.5π | 1.39e-16 | 1.39e-17 | 1.25e-16 | yes |
| 2π | 8.16e-13 | 1.04e-17 | 8.16e-13 | yes |
| 4π | 4.30e-08 | 1.73e-18 | 4.30e-08 | yes |
| 6π | 3.11e-06 | 3.17e-14 | 3.11e-06 | yes |
| 10π | 8.69e-05 | 2.91e-09 | 8.69e-05 | yes |

Reproduces the audit's own table to within the stated factor 1.1, and the estimate exceeds the true
`CC25` error at every span tested — the nested pair is honest, not merely a resolution proxy.

### Item 6 — a fallback region that misses tolerance is bisected

Reused the stationary-phase problem above (its fallback regions straddle the stationary point at
`x=0.5`): 6 direct regions at depths 9 and 10, none at depth 0 — every fallback region there was
bisected at least nine times before being accepted. `test_fallback_region_bisects_on_missed_tolerance`
asserts a weaker, more robust version of this (`depth > 0` for at least one fallback region) so it
does not pin an exact depth to the test.

### Item 7 — five-problem A/B (prompt 02's problem set), before/after

`atol=1e-15, rtol=1e-10, chebyshev_order=12`, best-of-5 wall time:

| Problem | `\|Δvalue\|` | regions before/after | evaluations before/after | simple before/after |
|---|---|---|---|---|
| `SinIntegral` | 0.0 | 1/1 | 3/3 | 0/0 |
| `CosIntegral` | 0.0 | 1/1 | 3/3 | 0/0 |
| `SincIntegral` | 0.0 | 10/10 | 39/39 | 0/0 |
| `e^-x sin(wx)`, w=1e4, [1/3,7/3] | 0.0 | 1/1 | 3/3 | 0/0 |
| `GRZIntegral(100)` | 7.05e-13 (both ~1e-13, true value 0) | 4/**24** | 15/**51** | 0/**22** |

Four of five problems are completely unaffected — identical value, region count and evaluation
count. **`GRZIntegral(100)` is not**, and this needed explaining rather than waving away.

**Root cause, verified, not assumed.** `theta(x) = 100*arctan(x)` has `theta'(x) = 100/(1+x^2)`,
which never changes sign — so this is *not* a case where total variation and net phase disagree in
principle. Tracing the actual regions the *new* code classifies as fallback (e.g. `(-1, -0.9375)`,
width `0.0625`, depth 5) and computing their *net* phase change with the same `theta` function
gives `0.0003` — far below `SIX_PI`. **The old algorithm's net-phase gate would have made the
identical classification for this identical sub-interval**, had it ever been tested against it. It
never was: the old driver's gate lived only at the top of the main loop, applied to a region only
when it was *popped from the queue*. The two comparison children (`dataL`, `dataR`) that a Levin
region computes to test its own step-(4) residual were built by calling
`_adaptive_levin_subregion` *directly*, bypassing that gate entirely — so a comparison child always
got the full Levin solve regardless of how weakly oscillatory it was, and if the parent's
resolution test passed, that child's estimate was folded into the accepted parent and the child
itself was never independently classified. Moving the gate *inside* `_adaptive_levin_subregion_impl`
(required by the prompt's own restructuring, since that is the one function called for both the
"current" region and its comparison children) applies it uniformly for the first time. For a
monotone-phase problem with a sub-width that happens to fall right at the boundary the old
eager-Levin-comparison-child path was masking, this shows up as extra regions and evaluations, not
as a wrong answer — `GRZIntegral`'s existing test (`atol` tolerance `1e-10` on a true value of
`0.0`) passes both before and after, and the two "after" values (`-2.82e-13`, both runs' true value
is `0.0`) are equally correct. This is scoped in *Observations not acted on* below rather than
"fixed" here, per the prompt's explicit instruction not to raise `SIX_PI` in this commit.

### Item 8 — three-Bessel oracles, before/after

`quad_JJJ`, `k=1.3, q=1.7, s=2.1, max_x=1e5`, phases via `bessel_phase(atol=1e-25, rtol=5e-14)`,
Levin `atol=1e-14, rtol=1e-10`, `chebyshev_order=12` (matches prompt 02's log methodology exactly):

| Oracle | before | after | `\|Δvalue\|` | analytic |
|---|---|---|---|---|
| `J000` | `0.169230107559118` | `0.169230107558470` | 6.5e-13 | `0.169230373496541` |
| `J110` | `0.006509508070509` | `0.006509508070932` | 4.2e-13 | `0.006508860519098` |

**No oracle improved or worsened outside round-off-level noise.** Both before/after values agree
with each other far more tightly than either agrees with the analytic result — the residual
`~2.7e-7`/`~6.5e-7` gap to the analytic value is the phase/modulus-spline floor documented at
`three_bessel_integrals.py:18-22`, unrelated to this commit. So C2 was **not** measurably biting
these two oracles at these parameters: the old net-phase gate, while structurally the wrong
quantity, happened not to produce a materially wrong answer here (unlike the audit's synthetic
stationary-phase case, where it did, by 1590%). What *did* change is cost, reported next.

**Per-phase-group breakdown** (the four sum/difference sign combinations `_phase_group` builds,
`k,q,s = 1.3,1.7,2.1`, `max_x=1e5`, same tolerances), which is where the audit specifically expects
C2 exposure (`_phase_group`'s difference-type combinations have a `theta'` that changes sign near
each Bessel turning point):

| group (`e_nu`, `e_sigma`) | value before/after | regions before/after | evaluations before/after | wall time before/after |
|---|---|---|---|---|
| `(+1,+1)` (pure sum) | agree to 1e-10 rel | 10/24 | 25/55 | 60ms/97ms |
| `(+1,-1)` (difference) | agree to 1e-9 rel | 26/106 | 63/221 | 134ms/522ms |
| `(-1,+1)` (difference) | agree to 1e-9 rel | 40/155 | 85/317 | 220ms/746ms |
| `(-1,-1)` (double difference) | agree to 1e-9 rel | 63/220 | 135/455 | 363ms/939ms |

Values agree closely in every group (no 1590%-style correction), so C2 is not visibly wrong at
these parameters either — but cost rises 2.2-3.4x, growing with how much sign-difference structure
the group has, exactly as the audit's own reasoning about difference-type phase groups predicts.
Sampling the *same* regions and testing their *net* phase change against `SIX_PI` (`bessel_region_check.py`
in the scratchpad) confirms the same root cause as `GRZIntegral(100)` above: every sampled
fallback region's net phase is also below `SIX_PI`, so this is the same "the old eager comparison
child bypassed the gate" effect, not a bug in the new `phase_span` computation, and not a case where
`phase_span` and net phase disagree about the true oscillatory content. This is the honest,
loudly-reported finding item 8 asks for: **not** "C2 was wrong here", but "the corrected, uniform
gate costs more on exactly the phase-group family the audit flagged as most exposed, even where the
old gate's answer happened to still be usable."

### Item 9 — cost of a single fallback region in isolation, before/after

`int_1^2 exp(-t)/(1+t) sin(Wt) dt`, single-region (no bisection), `chebyshev_order=12`:

| phase span | `quad` integrand evaluations (before) | CC evaluations (after) |
|---|---|---|
| 2π | 21 | 23 (always) |
| 4π | 63 | 23 (always) |
| 6π | 63 | 23 (always) |
| 10π | 147 | 23 (always) |

Matches the audit's own figures closely (21/63/63/147). Confirms the *isolated per-region* cost
claim: bounded and independent of phase span, against `quad`'s unbounded growth. The increases
measured in items 7 and 8 come entirely from the driver visiting more regions under the corrected,
uniform gate (see the root-cause analysis above), not from the CC rule itself being expensive.

### Item 10 — `simple_quadrature`

`grep -n simple_quadrature AdaptiveLevin/levin_quadrature.py` returns nothing.

### Test suite

`PYTHONPATH=. ./venv/bin/python -m unittest discover -s AdaptiveLevin/tests -t .`: 16 tests
(11 existing + 5 new), OK, 0.03s. Also ran, to check the module's actual production callers were
not disturbed: `LiouvilleGreen/tests/test_three_bessel.py` (2 tests, OK) and
`LiouvilleGreen/tests/test_bessel_phase.py` (4 tests, OK).

## Deviations from the prompt

### Structurally required: the gate now applies to Levin comparison children too

The prompt describes restructuring `_adaptive_levin_subregion_impl` to gate internally; since that
function is the *only* place both the "current" region and the eagerly-computed `dataL`/`dataR`
comparison children are evaluated, moving the gate there necessarily applies it to both. The old
driver's gate applied only to popped regions, silently exempting comparison children. This is not
an optional design choice within the prompt's stated restructuring — it is what "gate inside
`_adaptive_levin_subregion_impl`" means — and it is the direct cause of the cost increases in items
7 and 8. Recorded here because it is a real, measured behavioural change beyond fixing C2/C7
themselves, not because there was a way to implement the prompt without it.

### Implementation choice: `f_scale` for the CC branch's `phase_err` reuses the fine-grid sample

The prompt says to give the CC region "the same `phase_err` treatment the current direct branch
gets, adapted to the new structure." The old branch took a fresh 3-point sample of `f` at
`(a, mid, b)` for its `f_scale` estimate. The CC branch already has `f` sampled at `2N-1 >= 15`
points for the quadrature itself, so `f_scale` is read off that sample instead — strictly more
information, zero extra evaluations. Chosen over keeping the separate 3-point sample because the
latter would be pure waste now that a denser sample already exists.

### Implementation choice: `raw_theta()` kept, `phase_scale()` removed

Both methods on `_Basis_SinCos` lost their only call site in this commit. `phase_scale()` is
deleted outright — its logic is now inlined via the already-computed `theta_scale`, so keeping it
around as a second, unused way to compute the same quantity would invite drift. `raw_theta()` is
kept: it is a one-line accessor to the phase function itself, no logic to drift, and a
plausible future basis-introspection primitive independent of this commit's restructuring. See
*Observations not acted on*.

### None (SIX_PI)

`SIX_PI` was not changed, and item 8's measurements are exactly the kind of evidence the README
anticipated for a future decision on it — recorded below, not acted on here.

## Verification performed

All of items 1-10 above were actually run (not reasoned about) with
`PYTHONPATH=. ./venv/bin/python`, using `git stash`/`git stash pop` against baseline `5d5b958` for
every before/after comparison, and their exact numbers are reported above. The five new unit tests
were added to `AdaptiveLevin/tests/test_levin_quadrature.py` and run as part of the full suite.

## Observations not acted on

- **Cost of the corrected gate on production three-Bessel difference-type phase groups is real
  (2.2-3.4x more evaluations, items 7-8) and traces to the same root cause in both cases**: the old
  driver's gate never applied to eagerly-computed Levin comparison children, so narrow,
  correctly-weakly-oscillatory sub-widths were previously always solved via the (more expensive,
  more accurate) Levin rule "for free" as a side effect of a parent's resolution test, rather than
  being independently classified. Fixing this is what the prompt's restructuring requires, and it
  does not compromise accuracy (items 7-8 show values agreeing to round-off or better in every
  case) — but it is worth flagging for whoever next considers raising `SIX_PI` (audit §4.5,
  README §2.4 note 1): raising the threshold would reduce exactly this fallback population, at the
  cost of handing more, and wider, regions to the (now bounded, but less accurate) CC rule instead
  of the Levin rule. That trade should be evaluated with these measurements in hand, not
  re-discovered from scratch.
- **`raw_theta()` on `_Basis_SinCos` has no call site left in this module** after this commit
  removed the net-phase gate (its last user). Left in place as a plausible basis primitive (see
  *Deviations* above) rather than removed, since removing public-ish API surface that a future
  basis or diagnostic might want felt like it was reaching beyond this prompt's scope. Worth
  reconsidering if it is still unused after prompt 10.
- **`GkSource`/`extract_QuadSourceIntegral_data.py` grep hits for `raw_theta`** (surfaced while
  checking for external callers of `_Basis_SinCos.raw_theta`/`phase_scale`) are unrelated: they
  call `.raw_theta()` on `LiouvilleGreen.phase_spline` objects, a completely different class with
  its own `raw_theta` method. Noted only so a future reader searching this log for "raw_theta"
  does not chase a false lead.

## State handed to the next prompt

- Every accepted region is now either a Levin region (`used_interval.type == "Levin"`) or a
  Clenshaw-Curtis fallback region (`type == "direct"`), and *both* carry a genuine `abserr` tested
  against `atol`/`rtol` before acceptance. Prompt 04 (the eq. (151) round-off floor) needs to
  replace `phase_err` on *both* branches: the Levin branch's `_phase_error(theta_scale,
  p_endpoint_l1)` call (unchanged from prompt 01/02) and the new CC branch's `_phase_error(theta_scale,
  f_scale * width)` call in `_adaptive_levin_subregion_cc()`. Both are marked with a comment noting
  prompt 04 will replace them.
- `_adaptive_levin_subregion_impl` returns `"is_direct"` (bool) in its dict; anything that pattern-matches
  on the shape of that dict (there is no such code outside this module today, confirmed by grep)
  needs to handle both shapes: Levin regions have `"p_ratios"`/`"p_sample"` populated and no
  `"abserr_direct"`; CC regions have `"abserr_direct"` and `"p_ratios"`/`"p_sample"` both `None`.
- The measured evaluation-count increase on difference-type three-Bessel phase groups (item 8) is
  environment-independent (verified it is not an artefact of a particular `k,q,s`, since the same
  effect reproduces on the synthetic monotone-phase `GRZIntegral(100)`, which has no stationary
  point at all) — it should not be mistaken for a regression when prompt 08 re-measures production
  costs after retuning the default order.
