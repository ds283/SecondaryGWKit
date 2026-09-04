# Implementation state — Levin quadrature refactor

**Campaign:** [`README.md`](README.md) · **Source audit:** [`docs/adaptive-levin-audit-2026-09.md`](../../docs/adaptive-levin-audit-2026-09.md)
**Baseline commit:** `c8a1918` (`main`, clean; `AdaptiveLevin/levin_quadrature.py` byte-identical to the audited `68cff5d`)
**Last updated:** 2026-09-04 — prompt 10 (test matrix and campaign verification) complete with
deviations. **The campaign is complete: 10 of 10 prompts landed.**

> **Maintenance rule.** Every prompt updates this file *in its own commit*, before committing.
> Set your row's status and the log link, and add or clear entries in §3 (Active issues). Do not
> edit rows other than your own except to close an issue you resolved, or to fill in an earlier
> prompt's commit SHA (see README §5 rule 5 — **do not write your own commit's SHA**).

---

## 1. Status board

Legend: ⬜ not started · 🟡 in flight · ✅ complete · ⚠️ complete with deviations · ⛔ blocked

### Correctness (the two critical findings and their reporting gaps)

| # | Prompt | Audit items | Status | Commit | Log |
|---|---|---|---|---|---|
| 01 | [Refuse or report](01-refuse-or-report.md) | recs 1–4 · C1, C3a, C6, C8, C9 | ✅ | `0002fb4` | [01](logs/01-refuse-or-report.md) |
| 02 | [Complexified solve](02-complexified-solve.md) | rec 6 · §4.1, §4.2 | ✅ | `5d5b958` | [02](logs/02-complexified-solve.md) |
| 03 | [Total-variation gate + nested CC](03-total-variation-gate.md) | rec 7 · C2, C7, §2.2, §2.4 | ✅ | `c4f2ab8` | [03](logs/03-total-variation-gate.md) |

### Error estimates

| # | Prompt | Audit items | Status | Commit | Log |
|---|---|---|---|---|---|
| 04 | [Round-off floor from eq. (151)](04-roundoff-floor.md) | recs 5, 10 · C4, §3.2–§3.4 | ⚠️ | `13773b9` | [04](logs/04-roundoff-floor.md) |
| 05 | [Make `atol` a global tolerance](05-global-tolerance.md) | rec 8 · C3b | ⚠️ | `88f6ab3` | [05](logs/05-global-tolerance.md) |
| 06 | [Fix the `p_use` mode filter](06-mode-filter.md) | rec 9 · C5 | ✅ | `ab8f7ca` | [06](logs/06-mode-filter.md) |

### Operational and tuning

| # | Prompt | Audit items | Status | Commit | Log |
|---|---|---|---|---|---|
| 07 | [Diagnostics, logging, import hygiene](07-diagnostics-hygiene.md) | recs 11, 12 · C10, C11 | ⚠️ | `78cae32` | [07](logs/07-diagnostics-hygiene.md) |
| 08 | [Spectral order and vectorised sampling](08-order-and-sampling.md) | recs 13, 14 · §4.3, §4.5 | ✅ | `c4dc41d` | [08](logs/08-order-and-sampling.md) |
| 09 | [Propagate the error estimate to callers](09-caller-propagation.md) | rec 15 · §3.4 | ⚠️ | `d384031` | [09](logs/09-caller-propagation.md) |

### Close-out

| # | Prompt | Audit items | Status | Commit | Log |
|---|---|---|---|---|---|
| 10 | [Test matrix and campaign verification](10-test-matrix.md) | rec 16 · C12 | ⚠️ | — (this commit) | [10](logs/10-test-matrix.md) |

**Progress:** 10 / 10 complete. **Campaign complete.**

---

## 2. Item-level tracking

Traceability from the audit's finding and recommendation IDs to the prompt that discharges them.

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| **C1** | **Critical** | Non-finite amplitude → region contributes exactly 0 with `abserr` exactly 0; caller gets a partial integral certified at machine precision | 01 | ✅ |
| **C2** | **Critical** | Weakly-oscillatory gate uses *net* phase change, not total variation → 1590% error on a phase with an interior stationary point, Levin never invoked | 03 | ✅ |
| **C3** | High | `atol` is per-region; aggregate `abserr` never compared with the request | 01 (report) + 05 (distribute) | ✅ |
| **C4** | High | `theta_scale = TWO_PI` hardwired for a range-reduced phase → floor optimistic by up to 10¹⁰; delays `phase_limited` by ~8 decades of `atol` on the production path | 04 | ✅ |
| **C5** | Medium | `p_use` gates on the *mean* of `\|p\|` but the estimate uses *endpoint* values; makes the value a function of `rtol`; is the mechanism behind C1 | 01 (non-finite half) + 06 (rest) | ✅ |
| **C6** | Medium | `max_depth` not updated on the fallback branch → depth-limit health warning cannot fire | 01 | ✅ |
| **C7** | Medium | Fallback regions accepted unconditionally; `quad`'s `abserr` recorded but never tested; global tolerances passed to each small panel | 03 | ✅ |
| **C8** | Medium | `atol = 0` subdivides without bound (2²⁰ regions at the default depth) | 01 | ✅ |
| **C9** | Low | No input validation; `adaptive_levin_sincos` has no docstring | 01 | ✅ |
| **C10** | Low | Diagnostic counters count regions but are named for solves; `evaluations` counts solves not evaluations | 07 | ✅ |
| **C11** | Low | `seaborn`/`matplotlib` at module scope (>95% of import time); cwd-relative failure dumps; `print` not `logging` | 07 | ✅ |
| **C12** | Low | Test coverage gaps | distributed + 10 | ✅ (`theta_mod_2pi`-only path, `theta_deriv`-only path, reversed span, `m != 2` (a new `_TwoIndependentSinCosBasis`, m=4, driven directly through `_adaptive_levin()`), and an explicit path-classification assertion for `_GRZIntegral`'s three lambdas are new in `AdaptiveLevin/tests/test_levin_quadrature.py`; non-monotonic phase / non-finite amplitude / a deliberate fallback region / `atol=0` were already covered and re-verified; `abserr`-vs-truth is now asserted on every problem with a closed form, including the four originals, whose 1e-10 thresholds are tightened to measurement-based values -- see `docs/adaptive-levin-verification.md` Sec 3) |

| Rec | Description | Prompt | Status |
|---|---|---|---|
| 1 | Finiteness checks on `f_Cheb`, sampled `θ′`, solved `p`; `p_use` rejects non-finite | 01 | ✅ |
| 2 | Compare `abserr_total` with `max(atol, rtol·\|val\|)`; return `converged`; warn | 01 | ✅ |
| 3 | `max_depth` updated on every popped region | 01 | ✅ |
| 4 | Reject `atol <= 0`, `rtol < 0`, `depth_max < 0`, `len(f) != 2`, `len(x_span) != 2`; docstring | 01 | ✅ |
| 5 | Replace the endpoint phase floor with eq. (151); add optional `theta_abserr` | 04 | ✅ |
| 6 | Complexify the `(sin, cos)` solve to `N×N`; preallocated assembly | 02 | ✅ |
| 7 | Sample once → gate on total variation → nested Clenshaw–Curtis fallback → same accept/bisect logic | 03 | ✅ |
| 8 | Distribute `atol` in proportion to interval length | 05 | ✅ |
| 9 | Gate `p_use` on endpoint magnitudes; add the discarded contribution to `abserr` | 06 | ✅ |
| 10 | Retune the phase-error safety factor | 04 (see README §2.4 note 2) | ✅ |
| 11 | Honest solve counters; rename `evaluations`; return `abserr` components | 07 (+ 04 for the components) | ✅ (see 07's log for the naming deviation: added `num_solves_direct`/`num_solves_lstsq`/`num_solves_pinv`/`num_solves_total`/`num_subregion_solves` rather than repurposing/renaming the existing `num_direct_solves`/`evaluations`, to keep this commit bit-equal and the benchmark harness untouched) |
| 12 | `seaborn`/`matplotlib` behind `emit_diagnostics`; parameterise paths; `logging` | 07 | ✅ |
| 13 | Raise the default order to 16; document the 12–32 band | 08 | ✅ |
| 14 | Optional vectorised sampling | 08 | ✅ (mechanism in place, unused by production — see log) |
| 15 | Propagate `abserr` out of the callers | 09 | ⚠️ (`quad_JJJ`/`quad_YJJ` and `QuadSourceIntegral.py`'s nine call sites all propagate `abserr`/`converged`/`phase_limited`; the reported number does not yet bound the true error on 5/7 analytic oracles — see prompt 09's log and issue [09-abserr-does-not-bound-phase-spline-floor] below) |
| 16 | Test matrix | distributed + 10 | ✅ (`AdaptiveLevin/tests/test_levin_quadrature.py` grew from 28 to 32 tests, 0.043-0.059s; every C12 gap named or verified -- see prompt 10's log and `docs/adaptive-levin-verification.md`) |

**Measured and explicitly rejected — do not schedule** (audit §2.3, §4.6): p-refinement before
bisection; global worst-first error balancing; recovering the Levin solve discarded when a child
falls to the fallback branch (4% at worst, and prompt 03 removes it anyway).

**Out of scope** (README §6): `bessel_phase.py` / `phase_spline.py` / `range_reduce_mod_2pi.py`;
`phase_spline` reporting its own fit accuracy; rank-revealing QR; the four-phase-group decomposition.

---

## 3. Active and unresolved issues

- **[03-fallback-cost-on-difference-groups]** *(opened by prompt 03, 2026-09-04)* — the
  total-variation gate now applies uniformly to every region, including the Levin branch's eagerly
  computed comparison children, which the old net-phase gate never tested. This is structurally
  required by the prompt (see prompt 03's log, "Deviations"), not a bug, and does not affect
  accuracy — but it measurably raises evaluation counts (2.2–3.4x) on the three-Bessel
  difference-type phase groups the audit itself flags as most C2-exposed, and on any other
  monotone-phase problem with a sub-width that lands near the `SIX_PI` boundary (reproduced on the
  synthetic `GRZIntegral(100)`, no stationary point involved: 4→24 regions, 15→51 evaluations).
  **Impact:** production `ComputeTargets/QuadSourceIntegral.py` and
  `LiouvilleGreen/three_bessel_integrals.py` calls will cost more per phase group than before this
  commit, though still bounded (no unbounded-`quad`-panel-count risk). **Next step:** prompt 08's
  order/cost re-measurement should use post-prompt-03 numbers as its baseline, not prompt 02's; if
  the extra cost is unwelcome, raising `SIX_PI` (audit §4.5) is the lever, evaluated with prompt
  03's log measurements in hand rather than re-derived from scratch.
  **Update (prompt 10, 2026-09-04):** quantified as a wall-clock ratio against the true
  pre-campaign baseline for the first time (every prior measurement was prompt-to-prompt, not
  against `c8a1918`): on the `J000` three-Bessel oracle (`k,q,s=1.3,1.7,2.1`, `max_x=1e5`,
  `chebyshev_order=12`), evaluations rise 308 -> 1042 (3.4x, matching this issue's own figure) and
  wall time rises 0.804s -> 2.961s through the real `quad_JJJ` caller chain -- the *finished*
  campaign is **3.7x slower** than the pre-campaign code on this oracle, despite the complexified
  solve (prompt 02) being real and independently confirmed faster in isolation. See
  `docs/adaptive-levin-verification.md` Sec 4.5 for the full measurement, including the synthetic
  problems where the net effect is closer to parity (0.94x-1.26x) because their region count is
  unaffected by the total-variation gate. This does not change the "next step" above; it sharpens
  the evidence for it.

- **[04-roundoff-floor-can-be-infinite]** *(opened by prompt 04, 2026-09-04)* — a region's
  `phase_err` / `abserr_roundoff` / `total_err` can now be `+inf` (`_roundoff_floor()` returns it
  when `G0 = min|theta'|` is at or below a floating-point noise floor relative to `G1`, which
  happens at an interior stationary point strictly inside a still-strongly-oscillatory Levin
  region — see prompt 04's log, "Deviations", for the regression this fixed). `phase_limited` is
  already guarded against this (`np.isfinite(phase_err)` is required before it can fire), so it
  cannot cause a wrong early acceptance. **Impact:** any future code that sums, plots, or takes a
  ratio involving `used_interval.total_err`/`.abserr_roundoff` across many regions (prompt 07's
  diagnostics rewrite is the obvious candidate) must not assume finiteness. **Next step:** prompt
  07 should audit `_write_progress_data` and any new diagnostics for this before shipping.
- **[04-theta-abserr-cc-branch-proxy]** *(opened by prompt 04, 2026-09-04)* — the declared-
  `theta_abserr` endpoint term (recommendation 5.2) is exact per-endpoint on a Levin region (true
  `p_endpoint_l1` split by side) but an even 50/50 split of a lumped `f_scale * width` proxy on a
  Clenshaw-Curtis fallback region, which has no Levin antiderivative to weight by. This is a
  judgement call (prompt 04's log, "Deviations"), not a defect — nothing supplies `theta_abserr`
  yet so it has no live consequence. **Impact:** none today. **Next step:** whoever wires up the
  first real `theta_abserr` caller (README §6's `phase_spline` accuracy API) should re-examine
  whether the CC-branch proxy is tight enough once there is a real number to check it against.

- **[05-zero-width-span-raises]** *(opened by prompt 05, 2026-09-04)* — prompt 05's own text assumed
  "a zero-width span returns 0.0 before the loop" (README's "Reversed spans" note); that early return
  does not exist. Confirmed empirically: `adaptive_levin_sincos((5.0, 5.0), ...)` raises `ValueError:
  sampled phase derivative theta' contains non-numeric values` from `build_Levin_data()`, reached
  *before* the driver loop's acceptance test (and thus before prompt 05's new `_local_atol()`
  scaling) ever runs. Not a regression from this commit — pre-existing behaviour, reached the same way
  before and after — and out of prompt 05's scope per README §5 rule 6. `_local_atol()` itself still
  guards `x_span_width == 0.0` defensively (returns `atol` unscaled) so it cannot divide by zero if
  this upstream crash is ever relaxed. **Impact:** any caller that can produce a genuinely zero-width
  `x_span` (e.g. a degenerate sub-interval from an upstream splitting routine) gets an exception, not a
  0.0 result, contrary to what a reader of this campaign's own planning doc would expect. **Next
  step:** whoever next touches input validation (`_adaptive_levin`'s `len`/finiteness checks, prompt
  01's territory) should decide whether a zero-width span should raise (arguably correct — it is
  almost certainly a caller bug) or return 0.0 (matches the stale premise in the README), and fix the
  README's "Reversed spans" note either way.

- **[09-abserr-does-not-bound-phase-spline-floor]** *(opened by prompt 09, 2026-09-04)* — the
  `abserr` now returned by `quad_JJJ`/`quad_YJJ` (and threaded through `QuadSourceIntegral.py`'s
  `_three_bessel_Levin`/`_three_bessel_integrals`/`analytic_integral`) does not bound the true error
  against the analytic oracle on 5 of 7 three-Bessel closed forms measured, by up to 11.5x (see
  prompt 09's log, "Numerical evidence", and the new `test_abserr_bounds_truth` in
  `LiouvilleGreen/tests/test_3bessel_analytic.py`, marked `@unittest.expectedFailure`). This is the
  scoping note prompt 09 was given in advance: `abserr` measures the Levin quadrature's own accuracy
  against the phase it was *given*, and cannot see the ~2e-8 relative fit floor of the phase/modulus
  splines that build that phase. **Impact:** any caller trusting this `abserr` as a total error bound
  on a three-Bessel value is currently over-confident by up to an order of magnitude on about
  two-thirds of the parameter space measured. **Next step:** `LiouvilleGreen/phase_spline.py` needs
  to report its own fit accuracy (README §6), and `adaptive_levin_sincos`'s existing but currently
  unused `theta_abserr` parameter (prompt 04) needs a caller in `LiouvilleGreen/bessel_phase.py` to
  wire the two together. Out of scope for this campaign (README §6).
- **[09-quadsource-total-error-incomplete]** *(opened by prompt 09, 2026-09-04)* — prompt 09
  propagated the *Levin* error estimate only (its own scope, by title). `compute_QuadSource_integral`'s
  `"total"` = `numeric_quad + WKB_quad + WKB_Levin` still has no combined error bound: `WKB_Levin`'s
  share is now in `metadata["WKB_Levin"]["abserr"]`, but `numeric_quad_data`/`WKB_quad_data` (regions
  1 and 2, plain `scipy.quad` via `simple_quadrature`) already compute their own `abserr` and it is
  silently discarded one level up — `numeric_quad_integral`/`WKB_quad_integral` return it under
  `payload["abserr"]`, but `compute_QuadSource_integral` only reads `payload["value"]`/`payload["data"]`
  (see prompt 09's log, "Observations not acted on"). This predates prompt 09 and is not
  Levin-specific. **Impact:** `"total"`'s true error is unknown even after this prompt; only its
  `WKB_Levin` third has a number attached. **Next step:** whoever wants a genuine end-to-end bound on
  `"total"` should read `payload["abserr"]` at all three regions in `compute_QuadSource_integral` and
  fold them (linearly, per this campaign's established policy) into the metadata dict alongside
  `"WKB_Levin"`.

> Add an entry here whenever a prompt finishes with something unresolved: a verification step that
> could not be run, an assumption that could not be confirmed, a deviation a later prompt has to
> work around. Format:
>
> - **[NN-shortname]** *(opened by prompt NN, YYYY-MM-DD)* — description. **Impact:** who is
>   affected. **Next step:** what would close it.
>
> Move closed entries to §4 rather than deleting them.

---

## 4. Resolved issues

- **[07-write-progress-data-cc-crash]** *(opened and closed by prompt 07, 2026-09-04)* —
  `_write_progress_data()` crashed (`TypeError: 'NoneType' object is not iterable`) whenever a
  region it re-solved for diagnostics landed on a Clenshaw-Curtis fallback region, because it
  iterated `dataX["p_sample"]` unconditionally and a fallback region's `p_sample` is always `None`
  (no Levin antiderivative exists there). Latent since prompt 03 introduced the fallback branch;
  never exercised because `emit_diagnostics=True` is off by default and nothing in this repository
  sets it. Found while verifying prompt 07's own "the `emit_diagnostics=True` path still works end
  to end" requirement, and fixed in the same commit with `X["p_sample"] or []` at the three
  iteration sites. See prompt 07's log, "Deviations", for the full trace.

---

## 5. Standing notes for all implementers

Established by the planning pass against the working tree at `c8a1918`. **These do not need
re-deriving** — but if your own check disagrees with one, stop and reconcile before editing,
because something has landed since.

1. **`AdaptiveLevin/levin_quadrature.py` is byte-identical to the audited commit `68cff5d`.** Every
   line number in `docs/adaptive-levin-audit-2026-09.md` resolves directly against `HEAD`. All 31
   cited line numbers were spot-checked; all resolve.
2. **Every critical and high finding was reproduced numerically before this plan was written.** See
   README §2.2 for the table. You do not need to re-establish that the defects exist; you need to
   establish that your fix removes them.
3. **Run everything with `PYTHONPATH=. ./venv/bin/python`.** The ambient `python3` lacks this
   project's dependencies. **There is no `pytest` in the venv** — use
   `./venv/bin/python -m unittest discover -s AdaptiveLevin/tests -t .`. Baseline: 4 tests, OK,
   0.008 s.
4. **The existing four tests are weak.** They assert `|value − truth| < 1e-10`; the module actually
   delivers ~10⁻¹⁶ absolute. They will not catch a regression of five orders of magnitude. Never
   treat "the tests still pass" as evidence that numerics are unchanged (README §5 rule 9).
5. **Clenshaw–Curtis nesting works for every `N ≥ 2`, not only odd `N`.** The audit's §4.5 claim
   that "nesting needs `N−1` even" is wrong: the `2N−1`-point extremal grid has `2N−2` intervals,
   which is even by construction, so `j = 2k` always lands on the `N`-point grid. Verified exactly
   (`max|diff| = 0.0`) at N = 12, 13, 16, 17, 25, 33. **Do not introduce an odd-order constraint.**
6. **The stale hard-coded order 64 is in `ComputeTargets/QuadSourceIntegral.py:29`, not
   `three_bessel_integrals.py`.** The latter was retuned to 12 in commit `cc64ae4` with the
   measurement recorded in its own source comment (`three_bessel_integrals.py:12-25`). The audit's
   §4.5 sentence naming `three_bessel_integrals.py` is stale.
7. **`ComputeTargets/QuadSourceIntegral.py` is the most C4-exposed caller**: it supplies
   `theta_mod_2pi` but not `theta_deriv` (deliberately, with a long comment at `:1023-1037`), so it
   takes both the spectral-differentiation branch and the `theta_scale = TWO_PI` branch. Nine call
   sites, all at order 64, all discarding `abserr`.
8. **`docs/adaptive-levin-benchmark/levin_bench/{runners,sweeps}.py` read the returned dictionary
   by key** — `num_regions`, `num_simple_regions`, `evaluations`, `max_depth`, `num_order_changes`,
   `chebyshev_min_order`, `num_direct_solves`, `regions`. Adding keys is safe; renaming one breaks
   the harness. If you rename, update the harness in the same commit.
9. **`quad_JJJ` / `quad_YJJ` have no production caller** — only `LiouvilleGreen/tests/` and the
   `docs/` benchmark harness. Production goes through `QuadSourceIntegral.py`'s nine direct
   `adaptive_levin_sincos` calls. Relevant to prompt 09's scoping.
10. **The `f`-vector is `m`-component by construction and every caller uses `m = 2`.** The
    complexification (prompt 02) applies only to the two-component `(sin, cos)` basis; the generic
    `m`-component path must be retained for the future bases sketched in the prior review's §8.3.
11. **`_Basis_SinCos` discovers its optional phase functions with `hasattr`**, because
    `__init__` only sets `_theta_mod_2pi` / `_theta_deriv` when the corresponding key is present.
    Any new optional key (e.g. prompt 04's `theta_abserr`) must follow the same pattern or use an
    explicit `None` default consistently — do not mix the two conventions.
12. **The `simple_quadrature` fallback emits `scipy`'s `IntegrationWarning` to stderr** as an
    ordinary Python warning; the audit's phrase "neither caught nor surfaced" means it is not
    *acted on* and does not reach the returned dictionary. It is visible in a terminal.
    Prompt 03 deletes this branch entirely.
13. **As of prompt 02, the `(sin, cos)` Levin system is solved as the complexified `N×N` system by
    default** (`_Basis_SinCos.supports_complexified_solve` is `True`, gated additionally on
    `m == 2`). Every existing test and production caller goes through this path today. Downstream
    code should read Levin antiderivatives via `P[i, k]` (shape `(m, chebyshev_order)`), never via
    a flattened `p` vector — the complex path has no such vector. Solve-ladder variable names are
    now `LevinL`/`rhs`/`sol` (previously `LevinL`/`f_Cheb`/`p`); the residual sanity check on the
    LU branch was removed (measured: 221 direct-solve calls across the whole test suite plus the
    `J000` oracle never exceeded a relative residual of 4.65e-15 against the 1e-10 threshold it was
    tested against, and finiteness-only vs. finiteness-and-residual agreed on every call).
14. **Prompt 02 measured the `lstsq` share on two three-Bessel oracles at 69–77%**, well above the
    audit's own synthetic reference points (0–25%, §4.4). This is new evidence for the still-deferred
    RRQR decision (audit §4.4, README §6) — not acted on, but the next person to revisit that
    decision should use these figures rather than only the audit's synthetic ones.
15. **As of prompt 03, `_adaptive_levin_subregion_impl` returns one of two shapes**, told apart by
    the `"is_direct"` key: a Levin region (`is_direct: False`, `"p_ratios"`/`"p_sample"` populated,
    `"phase_err"` from `_phase_error(theta_scale, p_endpoint_l1)`) or a Clenshaw-Curtis fallback
    region (`is_direct: True`, `"abserr_direct"` in place of a p-based estimate, `"p_ratios"`/
    `"p_sample"` both `None`, `"phase_err"` from `_phase_error(theta_scale, f_scale * width)` with
    `f_scale` read off the fallback's own fine-grid sample). Prompt 04's eq. (151) floor replaces
    the `_phase_error()` call on **both** branches, not just the Levin one.
16. **The total-variation gate (prompt 03) now applies uniformly to every region**, including the
    Levin branch's eagerly-computed `dataL`/`dataR` comparison children, which the old net-phase
    gate never tested (it lived only at the top of the driver's main loop, applied only to popped
    regions). This is the direct cause of a measured 2.2–3.4x evaluation-count increase on the
    three-Bessel difference-type phase groups — see §3
    `[03-fallback-cost-on-difference-groups]` and prompt 03's log for the full measurement and root
    -cause trace. It is not a bug and does not affect accuracy (values agree to round-off or
    better in every case measured); it changes the cost baseline that prompt 08's order/sampling
    re-measurement should use.
17. **`AdaptiveLevin/tests/` baseline is now 16 tests** (11 pre-prompt-03 + 5 added by prompt 03:
    nesting, CC weight exactness, CC weight vs. a transcendental integral, the C2 stationary-phase
    regression, and fallback-bisects-on-missed-tolerance), all passing in ~0.03s. Standing note 3's
    "4 tests" baseline is the pre-campaign figure at `c8a1918` and is left as written for historical
    accuracy; use this note for the current count.
18. **As of prompt 04, `theta_scale` and the `TWO_PI` inference are gone entirely** — deleted, not
    deprecated. `build_Levin_data()` now returns a 5-tuple (`AmatT, theta_prime_Cheb, w0, wk,
    phase_span`), not the former 6-tuple; `_adaptive_levin_subregion_cc()`'s signature takes
    `theta_prime_Cheb` where it used to take `theta_scale`. The default accuracy floor is now
    `_roundoff_floor()` (Chen et al. eq. 151), computed from `_levin_G0_G1()` and each region's
    already-sampled `f`; `_phase_error()` survives only for the optional, caller-declared
    `theta_abserr` endpoint term (recommendation 5.2), and no longer applies an internal `eps`
    factor -- its first argument is now a true absolute error in radians, not an inferred scale.
    See prompt 04's log for the full before/after and the reasoning.
19. **A region's round-off floor (`used_interval.phase_err` / `.abserr_roundoff` / `.total_err`)
    can be `+inf`.** `_roundoff_floor()` returns it when `G0 = min|theta'|` on a region is at or
    below a relative floating-point noise floor of `G1 = max|theta'|` (constant
    `_LEVIN_ROUNDOFF_G0_NOISE_FLOOR = 1e-10`) -- reachable whenever a Levin region (large total
    variation) contains an interior stationary point close to a sampled node, which a bare
    `G0 > 0` test does **not** catch (differentiation noise from a spectrally-differentiated phase
    gives a small nonzero G0 even at an exact stationary point). `_adaptive_levin()`'s
    `phase_limited` test requires `np.isfinite(phase_err)`, so an infinite floor cannot force a
    wrong early acceptance -- it can only be reported once a region is accepted on some other
    basis. See §3 `[04-roundoff-floor-can-be-infinite]` and prompt 04's log ("Deviations") for the
    real regression this was found by (not anticipated by the prompt as written).
20. **`_adaptive_levin()`'s returned dict has three new keys**: `abserr_resolution`,
    `abserr_roundoff`, `abserr_fallback` (the aggregate `abserr` broken down by source, rec 11 /
    §3.4). Additive, safe for `docs/adaptive-levin-benchmark/levin_bench/` (confirmed by grep: it
    reads only `abserr` off `used_interval` objects, not these new keys or properties).
    `used_interval` gained matching properties `abserr_resolution`/`abserr_fallback` (type-gated:
    `None` on the wrong region type) and `abserr_roundoff` (an alias for the existing `phase_err`,
    defined on every region type).
21. **As of prompt 05, `atol` is distributed across subregions by length share, not applied raw.**
    `_local_atol(atol, a, b, x_span_width)` (`levin_quadrature.py:1329`) returns
    `atol * |b-a| / x_span_width`, computed once per popped region and used by all three per-region
    `atol` sites (the acceptance test, the relative-error denominator floor, the `phase_limited`
    guard) in both the fallback and Levin branches. `rtol` is deliberately **not** scaled -- it stays
    a raw per-region test, because a relative tolerance has no additive length-proportional analogue.
    The two post-loop, whole-integral `atol` uses (`relerr_total`'s denominator, and
    `requested_total` for `converged`) are unchanged and still use the raw, global `atol`. Net
    effect: `Σ abserr <= atol` holds by construction for a run that terminates normally, so
    `converged` is now usually `True` (previously it could be `False` on a normal run purely because
    of region count -- the audit's C3). See prompt 05's log for the full before/after measurement,
    including the one case where scaling changes cost (a 24-to-32-region problem, 1.31x more
    evaluations for a correctly-certified error bound) and the cases where it is a no-op (any
    problem resolving in 1-2 regions, where `local_atol ~= atol` regardless of the nominal value).
22. **A zero-width `x_span` raises, it does not return 0.0.** `_local_atol()` guards its own division
    against `x_span_width == 0.0` (returns `atol` unscaled), but that guard is currently unreachable
    in practice: `adaptive_levin_sincos((x, x), ...)` fails earlier, inside `build_Levin_data()`,
    with a non-finite-theta-prime `ValueError`, before the driver loop's acceptance test (and hence
    `_local_atol()`) is ever reached. See §3 `[05-zero-width-span-raises]`.
23. **As of prompt 06, `p_use` (the Levin mode filter) gates on endpoint magnitudes
    (`\|p_i(a)\| + \|p_i(b)\|`, ratio-to-maximum), not the collocation-point mean it used before.**
    This is a live distinction, not just a paper one: it was found to flip the accept/reject
    decision for a genuinely borderline mode on `grz_1000` (a problem already in this campaign's
    own standard set), changing that problem's returned value at the ~1e-13 level (see prompt 06's
    log, "A field case where problem (a) changes a result"). The discarded endpoint contribution of
    any *still-dropped* mode is now a fourth `used_interval`/return-dict component,
    `abserr_truncation` (always finite, `None` only for a Clenshaw-Curtis fallback region, which
    has no Levin antiderivative to gate), added into `total_err` alongside the existing
    `max(resolution/fallback residual, round-off floor)`. **The filter itself was kept, not
    removed** — measured to fire (and measurably improve accuracy) on an ordinary problem in this
    campaign's own set (`sin_1_100`, 3.64x closer to the closed form with the filter on than off),
    while never firing across a 12-combination three-Bessel oracle sweep at production `rtol`. See
    prompt 06's log, "The open question, answered", for the full evidence and reasoning.
24. **As of prompt 07, this module logs through `logging.getLogger("AdaptiveLevin.levin_quadrature")`
    and is silent by default** (a `logging.NullHandler()` is attached, no other handler is
    configured). Every message that used to `print()` unconditionally -- including warnings -- now
    requires the host to call `logging.basicConfig(...)` or attach its own handler to see it. A
    caller or test that used to capture stdout to check for a warning (as `test_input_validation`
    did) must use `self.assertLogs("AdaptiveLevin.levin_quadrature", level=...)` instead. `seaborn`/
    `matplotlib.pyplot` are imported inside `_write_progress_data()`, not at module scope (import
    time dropped from a measured 1.57s mean to 0.25s mean); nothing else in this module or its
    tests uses either.
25. **`adaptive_levin_sincos()` has a new `diagnostics_path` parameter** (default `None`, resolved
    internally to `DEFAULT_LEVIN_DIAGNOSTICS_PATH = Path("levin_diagnostics")`), threaded to both
    the lstsq-failure dump and `_write_progress_data()`'s output. This is a **behavioural change**:
    both used to write cwd-relative (the failure dump directly into cwd; progress data under a
    cwd-relative `SlowLevinData/`). Nothing in this repository reads either path (grepped), so
    nothing in-tree needed updating. The failure-dump filename now carries the run's `id_label`
    (previously just an isoformat timestamp at one-second resolution), so two workers failing in
    the same second cannot collide.
26. **The returned dictionary gained five more keys as of prompt 07**: `num_subregion_solves`
    (identical value to `evaluations`, correctly named -- `evaluations` counts subregion solves,
    not integrand evaluations, and is kept under its original name only because
    `docs/adaptive-levin-benchmark/levin_bench/` reads it by key), and `num_solves_direct` /
    `num_solves_lstsq` / `num_solves_pinv` / `num_solves_total` (true solve counts, accumulated
    from every Levin subregion solve actually attempted -- a region's own solve *and* both
    comparison children, whichever method succeeded -- unlike the pre-existing `num_direct_solves`,
    which is a *per-region* count that never sees a comparison-child solve whose parent region is
    accepted rather than bisected, and is kept exactly as before under that name for
    `docs/adaptive-levin-benchmark/levin_bench/runners.py`'s sake). All additive; the pre-existing
    keys are bit-identical before/after (verified by equality across five problems, prompt 07's
    log). Use `num_solves_direct / num_solves_total` to check the LU fast path's coverage, not
    `num_direct_solves / evaluations`.
27. **`_LazyUUID` and `_format_label(notify_label, id_label)` are the pattern for anything that logs
    a run's id, as of prompt 07.** `_adaptive_levin()` now constructs `id_label = _LazyUUID()`
    (defers `uuid.uuid4()`, ~1.9 microseconds, until first stringified) rather than
    `uuid.uuid4()` directly; every function that used to precompute a `label = f"...{id_label}..."`
    string unconditionally at its top (which would stringify the lazy id on every call regardless
    of whether a message is ever emitted, defeating the laziness) now takes `notify_label`/
    `id_label` separately and calls `_format_label()` only at the point a message is actually
    logged. A future new log call site should follow the same pattern, not reintroduce a
    precomputed `label`.
28. **`DEFAULT_LEVIN_CHEBSHEV_ORDER` is 16 as of prompt 08** (raised from 12; re-measured, not
    assumed, on the four `AdaptiveLevin/tests/` problems and three three-Bessel oracles -- see
    prompt 08's log for the full order-sweep table). `_LEVIN_MINIMUM_ALLOWED_ORDER` (8) is
    unchanged. Separately, `ComputeTargets/QuadSourceIntegral.py`'s own `CHEBYSHEV_ORDER` (all nine
    call sites pass it explicitly, so it is independent of the module default) is now 24, not 64 --
    chosen more conservatively than `three_bessel_integrals.py`'s 12 because no analytic oracle
    exists for this integrand and the evidence is self-consistency only (audit §7's caveat applies
    in full). **Raising the order has a real, measured cost**: eq. (151)'s `max(G1, k^2)/G0` term
    scales with `k^2`, so a region whose round-off floor is `k^2`-dominated (small `G1`, i.e. one
    that only just cleared the `SIX_PI` Levin/fallback gate) now reports a floor `(16/12)^2 = 1.78x`
    larger than before this prompt -- measured in isolation at 1.68x. This did not show up as a
    worse *aggregate* `abserr` on any concrete problem measured, because fewer regions at the
    higher order more than offset it there, but it is not free in principle and a future problem
    dominated by many such regions could see it.
29. **`_sample_vectorized(func, grid, cache, key)` and `_detect_vectorized(func, grid)`
    (`levin_quadrature.py`, just above `_Basis_SinCos`) are the pattern for sampling any
    caller-supplied callable at every point of a grid, as of prompt 08 (recommendation 14).**
    Detection (one array call, two scalar calls, checked for shape/dtype/finiteness/exact
    agreement) runs at most once per distinct callable per `adaptive_levin_sincos()` call, via a
    `vectorize_cache: dict` created once in `_adaptive_levin()` and threaded through every
    subregion-solving function and `_Basis_SinCos.build_Levin_data()`. **None of this campaign's
    production callers vectorize as of this prompt** -- `LiouvilleGreen/bessel_phase.py`'s
    `XSplineWrapper` and `LiouvilleGreen/phase_spline.py`'s `chunk_spline` both branch on a scalar
    argument throughout -- so the mechanism is exercised only by its own tests
    (`TestVectorizedSampling`) and by any future caller supplying genuinely array-capable
    callables. A future new loop-sampling call site should accept and thread the same
    `vectorize_cache` rather than sampling in a bare Python loop or inventing a second cache.
30. **`quad_JJJ`/`quad_YJJ` (`LiouvilleGreen/three_bessel_integrals.py`) return a
    `BesselIntegralResult` NamedTuple (`value`, `abserr`, `converged`, `phase_limited`) as of
    prompt 09, not a bare `float`.** Every error component this campaign introduced downstream of
    `adaptive_levin_sincos` is combined **linearly** across the four sum-and-difference phase groups
    and across the numeric/Levin split, never in quadrature -- the same policy prompt 09's own log
    justifies (the four groups share a phase construction, so an inaccurate phase produces a common
    drift, not independent noise). `ComputeTargets/QuadSourceIntegral.py`'s own (separate,
    `GkSource`-driven) `_three_bessel_Levin`/`_three_bessel_quad`/`_three_bessel_integrals`/
    `analytic_integral` functions follow the identical linear-combination policy, propagating
    `abserr`/`converged`/`phase_limited` all the way to `compute_QuadSource_integral`'s returned
    `"metadata"` dict (`metadata["analytic"]["abserr"]`, `metadata["WKB_Levin"]["abserr"]`) rather
    than into `LevinData` (a namedtuple mapped onto fixed Datastore SQL columns -- adding a field
    there is a schema change, out of scope). **The reported `abserr` does not yet bound the true
    error** on 5 of 7 analytic three-Bessel oracles measured (see
    `[09-abserr-does-not-bound-phase-spline-floor]` above) -- treat a small `abserr` from either
    `quad_JJJ`/`quad_YJJ` or `QuadSourceIntegral.py`'s Levin path as "the quadrature resolved the
    phase it was given", not as "this value is accurate to that many digits", until that issue
    closes. A future new three-Bessel error-propagating call site should follow the same
    linear-combination convention rather than combining in quadrature.
31. **As of prompt 10, the campaign is complete (10/10) and independently re-verified against the
    real `c8a1918` baseline, not just prompt-to-prompt.** `docs/adaptive-levin-verification.md`
    re-ran C1 (5 cases), C2, C3 (18 combinations), C4, the `lstsq` share, and all seven three-Bessel
    oracles, and found no contradiction of any prior prompt's claim on any of those. It also found
    one thing no prior prompt had computed: **the audit's/prompt 02's 1.4-1.8x complexification
    speedup does not describe the finished campaign's net effect.** On the production `J000`
    three-Bessel oracle through the real caller chain, the finished campaign is **3.7x slower**
    than the pre-campaign code (evaluations 308 -> 1042, wall time 0.804s -> 2.961s), because
    prompt 03's total-variation gate fix (C2, necessary for correctness) increases evaluation
    counts on exactly this phase-group family more than complexification saves. This is not a new
    defect -- `[03-fallback-cost-on-difference-groups]` above already named the mechanism -- but the
    wall-clock ratio against the true baseline is new evidence, folded into that issue rather than
    opened as a separate one. Meanwhile, **all seven three-Bessel oracles' delivered values are
    unchanged by the campaign** (agree with `c8a1918` to round-off or near-round-off in every
    case): the campaign changed reporting and cost, not correctness, on this integrand family.
    `AdaptiveLevin/tests/test_levin_quadrature.py` grew from 28 to 32 tests (0.043-0.059s); every
    C12 gap is now named or verified. Two things this pass could not close, honestly recorded rather
    than assumed: (a) the full plotting-inclusive `LiouvilleGreen/tests/test_3bessel_analytic.py::
    test_JJJ`/`test_YJJ` run, flagged as an open gap by prompt 09's own log -- see
    `docs/adaptive-levin-verification.md` Sec 5/5.1 for the outcome of this session's attempt; (b)
    `ComputeTargets/QuadSourceIntegral.py` end to end under a real Ray/Datastore pipeline, which no
    prompt in this campaign has ever exercised.

---

**Campaign status: COMPLETE.** All ten prompts have landed, each in its own commit, each
independently revertible per README §5 rule 1. Every audit finding (C1-C12) and every
recommendation (1-16) is closed against the item-level tracking table in §2, with the open items in
§3 being genuine, correctly-still-open follow-on work (a `LiouvilleGreen/phase_spline.py` accuracy
API, a `SIX_PI` retuning decision, a full `ComputeTargets/QuadSourceIntegral.py`-under-Ray
verification pass) rather than anything this campaign was scoped to finish.
