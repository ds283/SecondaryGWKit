# Prompt 04 — The near-region sampler: branch tracking and two-sided adaptivity

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Design:** `DRAFT-PLAN.md` §7.1 (the sampler and its boundary), §7.3 (branch tracking, interpolation, derivatives), §4.5 (the conditioning inversion), §4.4 (the plausibility guard)
**Reconciliation items:** C2 (\(r\) traverses 90 cycles at \(\nu=1000.5\)), §3.1 (the measured amplitude band)
**Depends on:** 01 (references, metrics), 02 (pinned boundaries), 03 (the crossover \(x_\star\))
**Recommended model:** Opus — **this is the hardest prompt in the campaign.** If a stronger model
is available, use it here.
**Files you may touch:** new `LiouvilleGreen/bessel_near_region.py`, new
`LiouvilleGreen/tests/test_bessel_near_region.py`, plus the log and the status board.
**Do not touch:** `LiouvilleGreen/bessel_phase.py`, `LiouvilleGreen/bessel_tail.py`,
`LiouvilleGreen/phase_spline.py`.

Read first: README §2 (a), (c), (d), (e) and (f); `RECONCILIATION.md` C2 and §3.1;
`DRAFT-PLAN.md` §7.1, §7.3, §4.5, §4.4 and §5.3 (the designated fallback, which is why the module
boundary matters); then `logs/03-closed-form-tail.md` §"State handed to the next prompt" for the
\(x_\star\) table, and `logs/01-…` for the reference API.

---

## 1. Character of this commit

The one piece of the construction that is genuinely hard. Everything else in the campaign is
algebra, plumbing or measurement; this is where a wrong choice produces a representation that looks
fine on smooth low-order cases and is silently, unrecoverably wrong at high order.

**Why it is hard, stated precisely.** `DRAFT-PLAN.md` §4.5 overturns the naive expectation that the
residual is always better behaved than the full phase. From the exact identity
\(dr/d\log x=x(a^{-2}-1)\):

| \(\nu\) | \(x/\nu\) | \(\lvert d\theta/d\log x\rvert\) | \(\lvert dr/d\log x\rvert\) | gain |
|---|---:|---:|---:|---:|
| 1.5 | 1.0 | 0.943 | 0.471 | 2.0 |
| 100.5 | 1.0 | 17.2 | 83.3 | **0.21** |
| 1000.5 | 1.0 | 79.6 | 920.9 | **0.086** |
| 1000.5 | 1.5 | 1119 | 382 | 2.9 |
| 1000.5 | 100 | 1.0e5 | 5.0 | 2.0e4 |

At the turning point the residual is **11× worse** conditioned than the full phase at
\(\nu=1000.5\). The split pays only for \(x\gtrsim1.5\nu\), and pays enormously only in the tail.
So this module must refine hard where the residual is worst — near the turning point, at high
order — and coarsen where it is trivial, and it must never assume the residual is the easy
quantity.

**The failure this must prevent.** At a uniform 250 samples per e-fold the residual advance per
interval is 0.0019 rad (\(\nu=1.5\)), 0.058 (20.5), 0.333 (100.5) and **3.685 (1000.5)** — above
\(\pi\). `np.unwrap` cannot recover from that, raising the interpolation degree cannot repair an
incorrectly unwrapped sample, and no density fixes it after the fact. Across the whole near region
\(r\) traverses **90.05 cycles** at \(\nu=1000.5\) (`RECONCILIATION.md` C2), so there are ~90
genuine wraps to get right, not a handful.

## 2. Interface, and why it is a boundary

`DRAFT-PLAN.md` §7.1 requires the sampler to sit behind a **narrow internal boundary** returning
\((a_\nu,r_\nu)\) on a grid, so that §5.3's hybrid (quadrature near the turning point, series in the
tail) could later replace this region alone without touching the tail, the interpolation, the
evaluation path or the consumers. Honour that literally: the module's public surface is a request
for sampled data and interpolants over \([x_{\rm lo},x_\star]\), and it knows nothing about
`bessel_phase`'s dict, its accessors, or its consumers.

Suggested shape — adapt names, but keep the boundary this thin:

```python
@dataclass(frozen=True)
class NearRegionData:
    nu: float
    log_x_nodes: np.ndarray        # u = log x, ascending
    a_nodes: np.ndarray            # a_nu at the nodes
    r_nodes: np.ndarray            # continuously tracked r_nu at the nodes (NOT wrapped)
    r_interp: ...                  # callable u -> r, with .derivative()
    log_a_interp: ...              # callable u -> ell = log a, with .derivative()
    achieved_phase_abserr: float   # estimated, see §5
    achieved_amplitude_relerr: float
    achieved_deriv_relerr: float
    interp_degree: int
    refinement_passes: int
    wraps_tracked: int             # number of 2*pi branch crossings resolved
    x_lo: float
    x_star: float

def build_near_region(nu, x_lo, x_star, phase_atol, amplitude_rtol, ...) -> NearRegionData: ...
```

`wraps_tracked` is not decoration: §6 asserts it equals the expected count per order, and it is the
campaign's evidence that branch tracking was *verified* rather than assumed (README §6, the
high-order structural row).

## 3. The construction

### 3.1 Sampling

Build \(S_\nu(x)=\sqrt{\pi x/2}\,e^{i(\pi\nu/2+\pi/4)}\operatorname{hankel1e}(\nu,x)\) — exact, by
the cancellation of README §2 (a) — and take \(a_\nu=\lvert S_\nu\rvert\),
\(r_\nu=\arg S_\nu\) continuously tracked. **Never** obtain the residual by subtracting \(x\) from a
computed phase; the scaled routine supplies the oscillation-removed quantity directly, and that is
the whole reason this route works.

Sample only for \(x\le x_\star\), with \(x_\star\) from `bessel_tail.tail_crossover` (prompt 03).
Do not compute your own crossover.

### 3.2 The plausibility guard — two-sided, not `isfinite`

`hankel1e(100.5, 1e9)` is exactly `-0j`, which **is finite**, so a finite-value check passes and
`log(abs(·))` becomes `-inf` (README §2 (e), prompt 02 Fact 1). The guard is a band on \(a_\nu\).
Measured over \([x_0,\,100\nu]\) (`RECONCILIATION.md` §3.1):

| \(\nu\) | \(\min a\) | \(\max a=a(x_0)\) |
|---|---:|---:|
| 1/2 | 1.0000000000 | 1.000000 |
| 5/2 | 1.0000240009 | 1.32288 |
| 20.5 | 1.0000249867 | 1.85684 |
| 100.5 | 1.0000250009 | 2.41795 |
| 1000.5 | 1.0000250016 | 3.54597 |

\(a\) is monotone decreasing in \(x\), attains its maximum at the turning point, and that maximum
grows only like \(\nu^{1/6}\). So a two-sided \(O(1)\) band is available and there is no excuse for
a one-sided test: pick something like \(0.99\le a\le 8\), justify the constants against this table
in a comment, and **test both sides** (§6 item 3). A one-sided \(a\gtrsim1\) check would pass a
spuriously large value.

Fail construction **loudly** — raise, naming \(\nu\), the offending \(x\), the value of
\(\operatorname{hankel1e}\) there and the band — if any sample is rejected. Do not drop the sample,
do not interpolate across it, do not warn and continue.

### 3.3 Branch tracking

`DRAFT-PLAN.md` §7.3 sets four requirements. Implement all four and say in the log how each is met:

1. **Anchor** consistently with both \(J\) and \(Y\), and document the allowed constant
   integer-cycle offset. \(\arg S_\nu\) at the lowest node is the natural anchor; if a separate
   check is wanted, \(\theta_0=\operatorname{atan2}(J_\nu(x_0),-Y_\nu(x_0))\) uses both functions
   and needs no root solve — which is the point of §4.3's finding that the existing `phi` is a
   pure artefact of a loose root solve. **Do not add a root solve.**
2. **Refine before accepting** an interval whose residual variation could conceal a wrap. The
   estimator is the exact \(dr/d\log x=x(a^{-2}-1)\), computed from the *sampled* \(a\) — so it
   costs nothing beyond the samples you already have. Its known limitation is that it cancels in
   the tail (§5.2), which is harmless here: the tail is closed-form and the estimator is needed only
   where \(a^{-2}-1\) is \(O(1)\).
3. **Track the continuous branch** through accepted intervals.
4. **Validate continuity and derivative behaviour** across refinement and interpolation boundaries.

Two explicit prohibitions, both from §7.3:

- **Do not rely on endpoint principal-angle differences alone.** An interval may contain an
  undetected full turn. Use the derivative estimator to bound the variation *before* accepting.
- **Do not use a bare `np.unwrap` on a fixed grid** as the tracker. `DRAFT-PLAN.md` §12 says the
  fixed-grid `unwrap` prototype "is intentionally limited to the low-order comparison and must not
  be copied as the high-order branch algorithm". Using it is a stop condition.

A reasonable acceptance criterion per interval, which you may adopt or improve on with reasons: an
interval \([u_i,u_{i+1}]\) is branch-safe when \(\max\lvert dr/d\log x\rvert\cdot h\) over its
endpoints is below some fraction of \(\pi\) (a half or a third), *and* the endpoint principal-angle
difference is consistent with that bound. State the fraction and why.

### 3.4 Two-sided adaptive interpolation

Adaptivity must be **two-sided** (§4.5): refine near the turning point, and *coarsen* in the tail
where \(r\approx Ce^{-u}\) makes every \(u\)-derivative as small as \(r\) itself and the required
density collapses far below 250 per e-fold. Revision 1 of the plan discussed only refining; do not
repeat that.

Interpolate \(r(u)\) and \(\ell(u)=\log a(e^u)\) in \(u=\log x\). **Quintic is the starting
candidate**, on the evidence of `DRAFT-PLAN.md` §4.7 and §6.2 — cubic fails the \(10^{-9}\)
derivative target at *every* order, and quintic at 250 per e-fold reaches 1.9e-13 to 8.1e-13 at low
order. But degree alone is **not** the acceptance criterion (§7.3): check at additional points,
refine, and include endpoint intervals. Piecewise Chebyshev is a valid alternative if it makes error
estimation simpler; if you take it, say why.

Interpolate \(\ell=\log a\), not \(a\): it preserves positivity, and \(\theta'=e^{-2\ell}\) is then
a value read off an interpolant rather than a differentiated one (README §2 (f)).

**A refinement cap is mandatory**, and on hitting it the function must **report unmet accuracy
rather than silently accepting it** (`DRAFT-PLAN.md` §9 Stage 2). Return the achieved errors in
`NearRegionData` either way, and let prompt 05 decide whether to raise; that keeps the policy
decision in one place.

## 4. Do not

- Do not build a `(div_2pi, mod_2pi)` representation, and do not use `phase_spline` or
  `simple_mod_2pi`. \(r\) is interpolated directly as a float. **The reason is not that \(r\) is
  sub-cycle** — it reaches 570.82 rad at \(\nu=1000.5\) (`RECONCILIATION.md` C2, correcting
  `DRAFT-PLAN.md` §4.6) — but that \(\varepsilon\cdot571\approx1.3\times10^{-13}\) sits an order
  below the \(10^{-11}\) low-order target and seven below the \(10^{-6}\) high-order one. Put that
  argument, with the number, in the module docstring: the false version ("never exceeds a cycle")
  must not enter the codebase.
- Do not sample above \(x_\star\), for any reason, including "to check the crossover". Prompt 05
  checks the crossover by comparing the *interpolant* against the *series* at \(x_\star\).
- Do not compute your own \(x_\star\), your own series, or your own references.
- Do not import from `bessel_phase.py`.

## 5. Achieved-accuracy estimates

`NearRegionData` reports three numbers, and prompt 05 propagates them into `theta_abserr` for the
Levin quadrature (`DRAFT-PLAN.md` §8.1). They must be honest estimates of *this construction's*
error, obtained without a reference the shipped object could not compute for itself:

- `achieved_phase_abserr` — from refinement comparison (the change in \(r\) at test points when the
  grid is halved), or a residual estimate of the interpolant at interior points. State the method.
- `achieved_amplitude_relerr` — likewise for \(\ell\).
- `achieved_deriv_relerr` — for \(\theta'=e^{-2\ell}\), which §4.7 finds is the binding quantity,
  harder than the value by two to three orders.

State plainly in the docstring that these are **practical estimators, not supremum bounds**
(README §6). An honest over-estimate is much better than an optimistic under-estimate: the whole
point of `theta_abserr` is that "the caller sees an honest number instead of an artificially small
one" (`levin_quadrature.py:2360`).

## 6. Tests

`LiouvilleGreen/tests/test_bessel_near_region.py`, scored against `bessel_reference` (prompt 01).

1. **Accuracy, low orders.** \(\nu\in\{1/2,3/2,7/4,5/2\}\): \(E_\theta,E_A\le10^{-11}\) at sample
   nodes, at log-interval midpoints, **and in the endpoint intervals** (all three point sets kept
   separate — `DRAFT-PLAN.md` §9 Stage 1). Reconstruct \(\theta=x+c_\nu+r\) and use
   `phase_pair_error`.
2. **Derivative, low orders.** \(\theta'=e^{-2\ell}\) to \(\le10^{-9}\) relative against
   \((2/\pi)/(x(J^2+Y^2))\), measured at midpoints. Report where the maximum falls; §4.7 predicts
   the interval adjacent to the turning point, and a maximum somewhere else is a signal worth a
   sentence in the log.
3. **The plausibility band, both sides.** Feed the guard a fabricated `-0j` sample and a fabricated
   large one, and assert it raises in both cases with \(\nu\) and \(x\) in the message. Assert too
   that `isfinite` alone would have passed the `-0j` case — the test that makes the design decision
   legible.
4. **Branch tracking is verified, not assumed.** This is the acceptance criterion of README §6's
   high-order structural row and the most important test in the prompt. Two halves:
   - **The negative control:** a fixed-density 250-per-e-fold `np.unwrap` at \(\nu=1000.5\) produces
     an order-unity phase-pair error. Assert that it does (\(E_\theta>0.1\)), so the test documents
     the failure the shipped tracker exists to avoid.
   - **The positive result:** the shipped tracker at \(\nu=1000.5\) gives \(E_\theta\le10^{-6}\),
     and `wraps_tracked` equals 90 ± 1 (`RECONCILIATION.md` C2 measures 90.05 cycles traversed).
5. **Accuracy, high orders.** \(\nu\in\{20.5,100.5,1000.5\}\): \(E_\theta,E_A\le10^{-6}\) and
   derivative relative error \(\le10^{-6}\), over the lower bound through \(\min(x_\star,\max(1000,10\nu))\).
6. **Two-sided adaptivity actually happens.** Assert the node spacing in the top decade of the near
   region is **coarser** than at the turning point, by a factor you measure and record. A
   construction that only ever refines passes every accuracy test and still fails the design; this
   is the test that catches it.
7. **\(\nu=1/2\) needs no near region.** With \(x_\star\) = the lower bound (prompt 03), the sampler
   is either never called or returns a degenerate two-node structure. Assert whichever your design
   produces, and assert \(r\equiv0\), \(a\equiv1\) to \(10^{-15}\) if it does run.
8. **Consistency of the two derivative routes** (README §2 (f), `DRAFT-PLAN.md` §7.3): \(e^{-2\ell}\)
   and \(1+r_u/x\) agree to the budget, and both agree with the reference \(\theta'\). This is
   **required, not optional** — with \(\theta'\) derived from \(\ell\), the Wronskian check
   \(a^2\theta'=1\) is satisfied by construction and checks nothing, so this is the only independent
   test of the two interpolants against each other.
9. **Cost.** Record node count and build time per order. \(\nu=1000.5\) will be the expensive case;
   report it rather than asserting a bound, unless it exceeds ~30 s, in which case say so
   prominently — it would make prompt 08's high-order tests painful.

## 7. Verification and acceptance

- `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_near_region -v`
  passes.
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .` passes; no
  production code changed, so `test_bessel_phase.py` must be unaffected.
- No `phase_spline`, `simple_mod_2pi` or `np.unwrap` in the shipped tracker (a `np.unwrap` inside
  the *negative-control test* is expected and correct).
- `bessel_near_region.py` does not import `bessel_phase`.

**Stop and report rather than working around**, per README §4.3, if: the branch-tracking test at
\(\nu=1000.5\) cannot be met inside the refinement cap; you conclude the supported order ceiling
should be lowered; or you find that the residual's conditioning near the turning point makes
\(10^{-11}\) unreachable at low order. Any of those is a design question for the user, not an
implementation detail.

## 8. Log and commit

Follow README §5 and §5.1. In "State handed to the next prompt":

- `NearRegionData`'s final field names and `build_near_region`'s signature, verbatim;
- the plausibility band constants and their justification;
- the branch-safety criterion (the fraction of \(\pi\)) and the interpolation degree shipped;
- the refinement cap, and what is reported when it binds;
- **per order** \(\nu\in\{1/2,3/2,7/4,5/2,20.5,100.5,1000.5\}\): node count, build time,
  `wraps_tracked`, \(E_\theta\), \(E_A\), derivative relative error, and the \(x\) of each maximum;
- the method behind each of the three `achieved_*` estimates, and how they compare to the measured
  errors (an estimator that under-reports by an order is a problem prompt 05 needs to know about);
- the measured coarsening factor from §6 item 6.

Commit subject, or something equally specific: `Add the branch-tracked near-region Bessel sampler`.
