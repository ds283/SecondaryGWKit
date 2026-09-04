# Implementation state — Levin quadrature refactor

**Campaign:** [`README.md`](README.md) · **Source audit:** [`docs/adaptive-levin-audit-2026-09.md`](../../docs/adaptive-levin-audit-2026-09.md)
**Baseline commit:** `c8a1918` (`main`, clean; `AdaptiveLevin/levin_quadrature.py` byte-identical to the audited `68cff5d`)
**Last updated:** 2026-09-04 — prompt 03 (total-variation gate + nested CC fallback) complete.

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
| 03 | [Total-variation gate + nested CC](03-total-variation-gate.md) | rec 7 · C2, C7, §2.2, §2.4 | ✅ | — (this commit) | [03](logs/03-total-variation-gate.md) |

### Error estimates

| # | Prompt | Audit items | Status | Commit | Log |
|---|---|---|---|---|---|
| 04 | [Round-off floor from eq. (151)](04-roundoff-floor.md) | recs 5, 10 · C4, §3.2–§3.4 | ⬜ | — | — |
| 05 | [Make `atol` a global tolerance](05-global-tolerance.md) | rec 8 · C3b | ⬜ | — | — |
| 06 | [Fix the `p_use` mode filter](06-mode-filter.md) | rec 9 · C5 | ⬜ | — | — |

### Operational and tuning

| # | Prompt | Audit items | Status | Commit | Log |
|---|---|---|---|---|---|
| 07 | [Diagnostics, logging, import hygiene](07-diagnostics-hygiene.md) | recs 11, 12 · C10, C11 | ⬜ | — | — |
| 08 | [Spectral order and vectorised sampling](08-order-and-sampling.md) | recs 13, 14 · §4.3, §4.5 | ⬜ | — | — |
| 09 | [Propagate the error estimate to callers](09-caller-propagation.md) | rec 15 · §3.4 | ⬜ | — | — |

### Close-out

| # | Prompt | Audit items | Status | Commit | Log |
|---|---|---|---|---|---|
| 10 | [Test matrix and campaign verification](10-test-matrix.md) | rec 16 · C12 | ⬜ | — | — |

**Progress:** 3 / 10 complete.

---

## 2. Item-level tracking

Traceability from the audit's finding and recommendation IDs to the prompt that discharges them.

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| **C1** | **Critical** | Non-finite amplitude → region contributes exactly 0 with `abserr` exactly 0; caller gets a partial integral certified at machine precision | 01 | ✅ |
| **C2** | **Critical** | Weakly-oscillatory gate uses *net* phase change, not total variation → 1590% error on a phase with an interior stationary point, Levin never invoked | 03 | ✅ |
| **C3** | High | `atol` is per-region; aggregate `abserr` never compared with the request | 01 (report) + 05 (distribute) | 🟡 (report done, distribute pending) |
| **C4** | High | `theta_scale = TWO_PI` hardwired for a range-reduced phase → floor optimistic by up to 10¹⁰; delays `phase_limited` by ~8 decades of `atol` on the production path | 04 | ⬜ |
| **C5** | Medium | `p_use` gates on the *mean* of `\|p\|` but the estimate uses *endpoint* values; makes the value a function of `rtol`; is the mechanism behind C1 | 01 (non-finite half) + 06 (rest) | 🟡 (non-finite half done, rest pending) |
| **C6** | Medium | `max_depth` not updated on the fallback branch → depth-limit health warning cannot fire | 01 | ✅ |
| **C7** | Medium | Fallback regions accepted unconditionally; `quad`'s `abserr` recorded but never tested; global tolerances passed to each small panel | 03 | ✅ |
| **C8** | Medium | `atol = 0` subdivides without bound (2²⁰ regions at the default depth) | 01 | ✅ |
| **C9** | Low | No input validation; `adaptive_levin_sincos` has no docstring | 01 | ✅ |
| **C10** | Low | Diagnostic counters count regions but are named for solves; `evaluations` counts solves not evaluations | 07 | ⬜ |
| **C11** | Low | `seaborn`/`matplotlib` at module scope (>95% of import time); cwd-relative failure dumps; `print` not `logging` | 07 | ⬜ |
| **C12** | Low | Test coverage gaps | distributed + 10 | ⬜ |

| Rec | Description | Prompt | Status |
|---|---|---|---|
| 1 | Finiteness checks on `f_Cheb`, sampled `θ′`, solved `p`; `p_use` rejects non-finite | 01 | ✅ |
| 2 | Compare `abserr_total` with `max(atol, rtol·\|val\|)`; return `converged`; warn | 01 | ✅ |
| 3 | `max_depth` updated on every popped region | 01 | ✅ |
| 4 | Reject `atol <= 0`, `rtol < 0`, `depth_max < 0`, `len(f) != 2`, `len(x_span) != 2`; docstring | 01 | ✅ |
| 5 | Replace the endpoint phase floor with eq. (151); add optional `theta_abserr` | 04 | ⬜ |
| 6 | Complexify the `(sin, cos)` solve to `N×N`; preallocated assembly | 02 | ✅ |
| 7 | Sample once → gate on total variation → nested Clenshaw–Curtis fallback → same accept/bisect logic | 03 | ✅ |
| 8 | Distribute `atol` in proportion to interval length | 05 | ⬜ |
| 9 | Gate `p_use` on endpoint magnitudes; add the discarded contribution to `abserr` | 06 | ⬜ |
| 10 | Retune the phase-error safety factor | 04 (see README §2.4 note 2) | ⬜ |
| 11 | Honest solve counters; rename `evaluations`; return `abserr` components | 07 (+ 04 for the components) | ⬜ |
| 12 | `seaborn`/`matplotlib` behind `emit_diagnostics`; parameterise paths; `logging` | 07 | ⬜ |
| 13 | Raise the default order to 16; document the 12–32 band | 08 | ⬜ |
| 14 | Optional vectorised sampling | 08 | ⬜ |
| 15 | Propagate `abserr` out of the callers | 09 | ⬜ |
| 16 | Test matrix | distributed + 10 | ⬜ |

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

*(none yet)*

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
