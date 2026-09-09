# Prompt 05 — Replace `bessel_phase` with the two-region construction

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Design:** `DRAFT-PLAN.md` §9 Stage 2, §7.3 (accessors and derivatives), §7.4 (split evaluation), §7.5 (argument accuracy), §4.3 (the spurious offset)
**Reconciliation items:** C1 (the real cliff and its mechanism), C2 (the correct reason to drop `phase_spline`), C4 (the standing derivative gate), §3.3 (`XSplineWrapper` and `sample_points` are public)
**Depends on:** 01, 02, 03, 04 — all hard.
**Recommended model:** Opus — the second-hardest prompt. If a stronger model is available, use it here.
**Files you may touch:** `LiouvilleGreen/bessel_phase.py` (substantial rewrite), new
`LiouvilleGreen/tests/test_bessel_two_region.py`, plus the log and the status board.
**Do not touch:** `LiouvilleGreen/bessel_tail.py`, `LiouvilleGreen/bessel_near_region.py`,
`LiouvilleGreen/phase_spline.py`, `main.py`, `LiouvilleGreen/three_bessel_integrals.py`,
`LiouvilleGreen/tests/test_bessel_phase.py`, or anything under `ComputeTargets/`. Consumer
migration is prompt 06's; test tightening is prompt 08's.

Read first: README §2 (all six facts) and §6; `RECONCILIATION.md` C1, C2, C4 and §3.3;
`DRAFT-PLAN.md` §9 Stage 2, §7.3, §7.4, §7.5, §4.1, §4.2, §4.3; then the current
`LiouvilleGreen/bessel_phase.py` in full, and the "State handed to the next prompt" sections of
logs 01, 03 and 04.

---

## 1. Character of this commit

The replacement itself. `bessel_phase` stops integrating an ODE and stops splining a growing phase;
it becomes an assembly of the two regions prompts 03 and 04 built, with accessors that preserve the
leading-plus-residual split all the way to `sin`/`cos`.

**This commit changes what production computes.** It is the point of no return for the campaign, and
the one where a subtle sign or convention slip would be most expensive. Four things are removed and
each removal has a specific, measured justification — put each one in the commit message:

| Removed | Why |
|---|---|
| the \(Q=\theta/x\) ODE | \(\delta\theta=x\,\delta Q\): controlling the relative error of a state approaching unity gives no uniform absolute-phase bound (§4.1). And the phase is a **quadrature**, not an ODE — the right-hand side does not contain \(\theta\) (§5.1) |
| the `phi` root solve | For every \(\nu>1/2\) the match point *is* the initial node, where the phase was already fixed exactly, so the matching function vanishes identically at \(\phi=0\). The computed offset is entirely an artefact of `xtol=1e-6, rtol=1e-4`, and it **is** the whole tight-tolerance error: measured \(\phi=-4.836537\times10^{-8}\) and \(E_\theta=4.873\times10^{-8}\) at \(\nu=5/2\) (§4.3) |
| the full-phase `phase_spline` | For \(\theta\simeq x=e^u\) the fourth \(u\)-derivative is also \(\approx x\), so cubic interpolation errs by \(h^4x/384\) — 6.7e-10 at \(x=10^3\), 6.7e-6 at \(10^7\), both confirmed. Subtracting cycles changes the constant term, not the fourth derivative (§4.2), and chunking was measured to have **no** effect on it (§4.6). A second, independent reason has since been measured: chunk selection is a **hard switch** between two fits, with a 1.08e-4 rad phase jump and a 3.51e-8 relative derivative jump at the switch point (`docs/gk-wkb-numerical-review-2026-09.md` §3, defect 3) — a discontinuity a Levin consumer sees. Cite both |
| `simple_mod_2pi` / `(div_2pi, mod_2pi)` | With the tail closed-form, \(\lvert r\rvert\le571\) rad at the largest supported order, whose \(\varepsilon\)-resolution \(1.3\times10^{-13}\) is an order below the \(10^{-11}\) target (`RECONCILIATION.md` C2) |

**Note the last row carefully.** `DRAFT-PLAN.md` §4.6 gives a *different* reason — that \(r\) "never
exceeds a cycle" — and that is **false** above \(\nu\approx630\): measured \(r(x_0)=570.82\) rad at
\(\nu=1000.5\). The conclusion stands; the false justification must not appear in the code, the
docstrings or the commit message.

## 2. What to build

### 2.1 Structure

```
bessel_phase(nu, max_x, ...) ->
    x_star   = bessel_tail.tail_crossover(nu, phase_atol, amplitude_rtol)
    near     = bessel_near_region.build_near_region(nu, x_lo, x_star, ...)   # only if x_star > x_lo
    assemble -> the returned object
```

Domain lower bound: keep the current rule (`bessel_phase.py:85-88`) — \(x_{\rm lo}=\sqrt{\nu^2-\tfrac14}\)
for \(\nu>1/2\), else \(10^{-5}\). **Retain the current domain restrictions initially** and state
the supported \((\nu,x_{\max})\) domain explicitly, rejecting requests outside it rather than
extending behaviour implicitly (`DRAFT-PLAN.md` §9 Stage 2). Widening the domain is a stop
condition (README §4.3), even though the tail makes it tempting — the honest new statement is that
\(x_{\max}\) is now limited by `sin`/`cos` (correct to \(10^{16}\)) rather than by Amos, and
prompt 09 measures how far that actually goes.

The \(\nu=1/2\) case has \(x_\star=x_{\rm lo}\) (prompt 03), so the object is **all tail** and no
sampling happens. Handle it as a first-class case, not a degenerate accident, and test that it is
exact.

### 2.2 The region stitch

Above \(x_\star\), evaluate \(r_\nu\) and \(a_\nu\) from `bessel_tail`. Below, from `near`'s
interpolants. At \(x_\star\) itself the two must agree to the accuracy budget — this is the second
half of the remainder test `DRAFT-PLAN.md` §7.2 requires and prompt 03 could only do the series half
of:

> require in addition that the near-region interpolant and the series agree to the accuracy budget at
> the crossover. Record \(x_\star\) and the agreement achieved.

Compute that agreement **at construction time**, for both phase and amplitude, and store it on the
object. If it exceeds the budget, **raise**, naming \(\nu\), \(x_\star\), both agreements and both
budgets. Do not blend, taper or average across the crossover: a discontinuity that has to be hidden
by blending is a construction that failed its own test, and blending would also break the
\(C^1\) property the Levin quadrature's subdivision logic depends on. If you find that a small
overlap is genuinely needed for derivative continuity, that is an `IMPLEMENTATION CHOICE` requiring
explicit justification and a continuity test at the seam.

### 2.3 Accessors

Amplitude and its logarithmic derivative (`DRAFT-PLAN.md` §7.3):

$$A(x)=\sqrt{\frac{2}{\pi x}}\,e^{\ell(\log x)},\qquad
\frac{d\log A}{dx}=-\frac1{2x}+\frac{\ell_u(\log x)}{x}.$$

Phase derivative **from the amplitude interpolant as a value**, not by differentiating the residual:

$$\theta'(x)=a^{-2}=e^{-2\ell(\log x)}$$

§4.7 measures this as better in 7 of 8 configurations, by 4× to 15× (4.66e-14 vs 1.94e-13 at
\(\nu=3/2\); 2.09e-8 vs 8.06e-8 at \(\nu=100.5\)), and it avoids differentiating an interpolant.
Changing this choice is a stop condition. Its consequence for verification is that the Wronskian
becomes a tautology, so §4 item 6 below is mandatory.

In the tail, \(\theta'=1+r_\nu'\) directly from the series — which is a **primitive** there, not a
differentiated interpolant, so it is exact to the series remainder.

### 2.4 Split evaluation — §7.4

Let \(d=c_\nu+r_\nu(x)\). Evaluate

$$\sin\theta=\sin x\cos d+\cos x\sin d,\qquad \cos\theta=\cos x\cos d-\sin x\sin d.$$

Forming `x + d` first rounds away part or all of the correction. Measured against 70-digit
`mpmath` at \(\nu=3/2\) (`RECONCILIATION.md` §1):

| \(x\) | naive \(\sin(x+d)\) | angle addition |
|---|---:|---:|
| \(10^3\) | 3.475e-14 | 1.11e-16 |
| \(10^7\) | 1.210e-10 | 0 |
| \(10^{12}\) | 2.723e-06 | 0 |
| \(10^{15}\) | **4.725e-02** | 1.11e-16 |

Fourteen orders at \(x=10^{15}\). Pass the *original* \(x\) to `sin`/`cos`, unreduced: a quality
libm does Payne–Hanek reduction against a multi-hundred-bit \(\pi\) and is correctly rounded to
\(10^{16}\) (verified — `RECONCILIATION.md` §1), whereas any reduction you perform yourself is
done in double precision against a 53-bit \(2\pi\) and is strictly worse. This is exactly the
argument `LiouvilleGreen/range_reduce_mod_2pi.py`'s module docstring already makes; do not
contradict it.

`bessel_j` and `bessel_y` must go through this path.

### 2.5 Bounded angle, and `raw_theta`

A bounded-angle accessor should use `atan2(sin_theta, cos_theta)` on the split-evaluated pair,
with any interval convention applied **only** to that bounded result (§7.4). Reducing a huge phase
against a double-precision `TWO_PI` creates an error proportional to the cycle count; `fmod` is
exact for the arguments it is given but faithfully computes the remainder with respect to the
*wrong* modulus.

Keep `raw_theta`, and **document its \(\varepsilon x\) precision limit in its docstring**. It is
compatibility and diagnostics only; accurate oscillatory evaluation uses the split or the bounded
angle. Note that `theta` is a *required* key of the Levin phase dict — `_Basis_SinCos.__init__`
raises without it (`levin_quadrature.py:948-952`) — so "compatibility-only" does not mean
"optional" (`RECONCILIATION.md` §1). Prompt 06 wires the adapter; this prompt only has to make sure
`raw_theta` exists and is honest about itself.

**If a consumer genuinely requires an accurate integer cycle count, that is separate design and
validation work** (§7.4, README §7). Do not provide one here, and do not let `raw_theta` be mistaken
for one.

### 2.6 Argument accuracy — §7.5

When raw \(x\) is supplied, **preserve it for the leading oscillation** and use \(\log x\) only to
query the correction interpolants. If only \(u=\log x\) is supplied, define the evaluation as being
at the computed \(e^u\) and document that. The representation cannot recover uncertainty already
present in \(x=k\eta\), nor undo a lossy `exp(log(x))` round trip; at very large \(x\)
input-coordinate error can exceed the residual interpolation error by many orders. Say so in the
module docstring.

Every accessor keeps the existing `is_log` / `x_is_log` calling convention (see `XSplineWrapper.__call__`
and `phase_spline`'s accessors) so consumers do not have to change how they call. Prompt 06 settles
the exact adapter surface; do not pre-empt it, but do not gratuitously rename either.

### 2.7 Accuracy arguments

`atol`/`rtol` currently describe ODE tolerances and **will no longer have a referent**
(`DRAFT-PLAN.md` §8.1). Introduce explicit settings — `phase_atol` (absolute radians) and
`amplitude_rtol` — and thread them into `tail_crossover` and `build_near_region`.

Accept the deprecated `atol`/`rtol` **temporarily**, and document the translation and precedence
(§8.1 requires this explicitly). Suggested and defensible: `rtol` maps to nothing, `atol` maps to
nothing, both emit a deprecation warning naming the new arguments, and the new defaults are used;
if a caller passes both old and new, the new win and the warning says so. Whatever you choose, the
mapping must be documented and tested. Do not silently claim the old arguments have identical
semantics.

Keep `sample_points` accepted — `docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py:82`
passes it (`RECONCILIATION.md` §3.3). Under adaptive sampling it can no longer mean "use exactly
this many"; make it a *floor* on initial density or deprecate it explicitly, and say which.

Keep `XSplineWrapper` importable from this module: `LiouvilleGreen/tests/test_three_bessel.py:10`
imports it by name and annotates with it (`:67`). If your amplitude accessor is no longer that
class, either keep a compatible shim or note it for prompt 06 — but do not break the import in
*this* commit, because `test_three_bessel.py` is not yours to edit.

### 2.8 Report achieved accuracy

The object must report its own achieved phase and amplitude accuracy (§8.1, §9 Stage 2): the
near-region estimates from `NearRegionData`, the tail's series remainder, the crossover agreement,
and the maximum over the whole domain. Prompt 06 turns that into `theta_abserr` for the Levin
quadrature. Include a **refinement cap report**: if `build_near_region` could not meet the requested
accuracy, the construction must surface that rather than silently accepting it — either raise, or
return the object with an unmistakable `accuracy_met = False` and a warning. State which you chose;
raising is the safer default and matches "expose requested accuracy and report failure when it
cannot meet it" (§10).

## 3. What must not change

- The convention \(J_\nu=A_\nu\sin\theta_\nu\), \(Y_\nu=-A_\nu\cos\theta_\nu\), \(\theta\)
  increasing in \(x\). DLMF differs by \(+\pi/2\); that difference is absorbed in \(c_\nu\) and is
  not an error to fix.
- \(c_\nu=\pi/4-\pi\nu/2\).
- The returned dict's useful members: `phase`, `mod`, `bessel_j`, `bessel_y`, `min_x`, `max_x`
  (§8.1). `Q` and `phi` are prompt 06's to decide; leave them present and behaving as before if you
  can, or note precisely what you had to do. Note §4.3's point that `phi` should be reported as
  **identically zero** for \(\nu>1/2\), not quietly dropped.
- The domain lower bound and the supported order set.

## 4. Tests

`LiouvilleGreen/tests/test_bessel_two_region.py`, scored against `bessel_reference`.

1. **The acceptance table** (README §6), in full: low orders \(\{1/2,3/2,7/4,5/2\}\) to
   \(10^{-11}\) in \(E_\theta\) and \(E_A\) through \(x_{\max}=10^7\), including endpoint checks;
   derivative to \(10^{-9}\) relative; orders \(\{20.5,100.5,1000.5\}\) to \(10^{-6}\) in all
   three. Keep sample nodes, interior midpoints and endpoint intervals separate.
2. **\(\nu=1/2\) is exact and all-tail.** \(E_\theta\), \(E_A\) at machine precision; assert no
   near-region sampling occurred.
3. **The crossover.** At \(x_\star\) and at \(x_\star(1\pm10^{-9})\): the two regions agree to the
   budget in phase **and** amplitude, and \(\theta'\) is continuous across the seam to the
   derivative budget. Assert the stored crossover agreement matches what the test measures.
4. **Split evaluation.** Against 70-digit `mpmath` at \(x\in\{10^3,10^7,10^{12},10^{15}\}\) for
   \(\nu\in\{3/2,7/4\}\), passing `mpf(float(x))` so the reference uses the **same supplied
   argument** (§7.5). Also assert the split path beats a deliberately naive `sin(x + d)` by the
   orders in §2.4 — the test that keeps the design decision from being quietly undone.
5. **`phi` is zero.** For every \(\nu>1/2\) tested, the reported `phi` is exactly 0.0 (or the
   member is documented as removed — prompt 06 decides which, but if it is present it must be
   zero, per §4.3).
6. **The two derivative routes agree** (README §2 (f), mandatory): \(e^{-2\ell}\) vs \(1+r_u/x\) vs
   the reference \(\theta'\), all three, at midpoints. The docstring must say why a bare Wronskian
   check would be a tautology here.
7. **Loud failure.** Requests outside the declared domain raise; a construction that cannot meet its
   requested accuracy raises (or reports, per §2.8) rather than returning silently; and no `-inf`
   or `-0j` can reach an interpolant.
8. **Cost, and the cliff.** Build times at \(x_{\max}\in\{10^3,10^7,10^{11},10^{13},10^{15}\}\)
   should be roughly independent of \(x_{\max}\) — that is the design claim. **Additionally** build
   at \(x_{\max}=3\times10^{15}\) and \(8.6\times10^{15}\) and assert it *completes*: those are the
   cases where the old construction stalls (`RECONCILIATION.md` C1), and clearing them is the real
   scalability result. Do **not** assert a speed-up ratio against the old construction: the old
   build was ~0.1 s over the whole production range, so a ratio claim would be noise.
9. **Bessel zeros and extrema** (README §6): test near zeros of \(J_\nu\) and of \(Y_\nu\), where
   pointwise relative error is misleading and only the envelope-normalized measure is meaningful.

Runtime: keep under ~3 minutes if you can. If \(\nu=1000.5\) dominates, mark that case so prompt 08
knows what it is inheriting.

## 5. Verification and acceptance

- `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_two_region -v`
  passes.
- `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_phase -v` **passes
  unchanged**, including `test_phase_derivative`'s \(10^{-6}\) derivative contract at
  \(\nu\in\{2.5,20.5,100.5\}\) (`RECONCILIATION.md` C4). This is the campaign's standing regression
  gate. `DRAFT-PLAN.md` §4.7 measures the new derivative route at 2.09e-8 at \(\nu=100.5\), so the
  margin is under two orders — quote the measured margin in the log. Loosening this test is a stop
  condition; it is prompt 08's to *tighten*, nobody's to relax.
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .` passes.
  `test_three_bessel.py` and `test_3bessel_analytic.py` are the interesting ones: their tolerances
  were set against the old accuracy, and an eight-order improvement can **expose a different
  limiting error rather than simply passing more easily** (§9 Stage 4). If either fails, that is a
  finding, not a nuisance: record the numbers and stop.
- `grep -n "solve_ivp\|root_scalar\|phase_spline\|simple_mod_2pi" LiouvilleGreen/bessel_phase.py`
  returns nothing.
- No occurrence of the false justification "never exceeds a cycle" (or equivalent) anywhere in the
  file.

## 6. Log and commit

Follow README §5 and §5.1. In "State handed to the next prompt":

- the returned object's members and every accessor signature, verbatim — prompt 06 programs against
  these and prompt 07 against the residual and leading-coefficient accessors;
- what happened to `Q`, `phi`, `sample_points`, `XSplineWrapper`, `atol` and `rtol`, precisely;
- the deprecation translation and precedence rule, as shipped;
- the achieved-accuracy fields, and the **measured** \(E_\theta\), \(E_A\) and derivative error per
  order with the \(x\) of each maximum;
- the crossover agreement achieved per order, and \(x_\star\) as actually used;
- `test_phase_derivative`'s measured margin;
- the largest \(x_{\max}\) at which construction now completes, and its build time.

Commit subject, or something equally specific: `Rebuild the Bessel phase from a two-region
amplitude and residual`.
