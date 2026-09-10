# Open issues — project-wide index

**Last updated:** 2026-09-10 · **33 open** across four campaigns.

This file exists so that an issue opened by one campaign is not lost when that campaign closes.
It is an **index, not a record**: one line per issue, pointing at the campaign status board that
holds the measurements, the impact statement and the next step. Never put issue content here — if
the two disagree, the board is right.

> **Maintenance rule.** Whenever you add, narrow or close an entry in a campaign board's
> §3 (Active and unresolved issues) or §4 (Resolved issues), update this file **in the same
> commit**. Add the line, move it between sections here, or delete it, and correct the count and
> the date above. See `CLAUDE.md`.

**Boards.** [`source-remediation`](../prompts/source-remediation/IMPLEMENTATION_STATE.md) ·
[`levin-refactor`](../prompts/levin-refactor/IMPLEMENTATION_STATE.md) ·
[`backport-modules`](../prompts/backport-modules/IMPLEMENTATION_STATE.md) ·
[`transfer-remedial`](../prompts/transfer-remedial/IMPLEMENTATION_STATE.md)

---

## 1. Assigned to a future campaign

Work is identified and owned; the issue is parked deliberately, not forgotten.

### 1.1 The hand-over campaign

The numeric→Liouville–Green seam of $T_k$ and $G_k$. These six are **one place** and must be
attacked together — prompt 12 of `source-remediation` could not separate their contributions by
measurement alone at production $x$. Background reading:
[`docs/lg-phase-and-handover-followup-2026-09.md`](lg-phase-and-handover-followup-2026-09.md).

| Issue | Board | Hook |
|---|---|---|
| `[08-handover-clamp-error]` | source-remediation | The WKB grid starts below `crossover_z`, so the LG accessors are clamped across a gap; a one-step gap moves `total` by 5.1e-03. |
| `[12-handover-clamp-error-in-production]` | source-remediation | Supersedes the above for magnitude: on real rows the gap is universal and costs 6.6e-02–4.6e-01 against 6.2e-05 unclamped. **The campaign's main outstanding accuracy decision.** |
| `[12-phase-spline-error-grows-with-x]` | source-remediation | Stored-phase re-spline error $\simeq h^4x/384$, growing **linearly in $x$**; ~1 % of envelope extrapolated to production, and unmeasured beyond $x=10^4$. |
| `[05-numeric-region-is-now-the-accuracy-floor]` | source-remediation | A cubic spline loses 1–2 orders in its last two intervals, and the numeric grid *ends* at the hand-over. |
| `[06-source-spline-residual-vs-handover]` | source-remediation | $f$ oscillates at twice the transfer-function phase, so its spline is the worst of the three; 1.4e-04 of envelope on real rows, but $O(1)$ if the hand-over is allowed to fall to the bottom of `main.py`'s search window. |
| `[07-lg-derivative-truncation-at-handover]` | source-remediation | The irreducible one: `omega`/`dlnM_dz` are LG quantities, off by $O(x^{-4})$. Grid-independent — only a deeper hand-over helps. |

### 1.2 The `AdaptiveLevin` Clenshaw–Curtis fallback campaign

| Issue | Board | Hook |
|---|---|---|
| `[10-levin-wholesale-cc-fallback]` | source-remediation | 2.65–3.56× wall clock for 1.00–1.44× integrand evaluations, i.e. per-region overhead, not wasted work. Route wholesale to Clenshaw–Curtis at the outset instead of bisecting into it. **Do not** re-add a threshold in `QuadSourcePolicy` — that is defect A4 in weaker form. |
| `[03-fallback-cost-on-difference-groups]` | levin-refactor | The total-variation gate now tests the Levin branch's eagerly computed comparison children too; structurally required, but raises evaluation counts 2.2–3.4× on three-Bessel groups. |
| `[04-roundoff-floor-can-be-infinite]` | levin-refactor | `phase_err`/`abserr_roundoff`/`total_err` can be `+inf` at an interior stationary point inside a strongly oscillatory Levin region. |
| `[04-theta-abserr-cc-branch-proxy]` | levin-refactor | The endpoint term is exact on a Levin region but a 50/50 split of a lumped proxy on a Clenshaw–Curtis one, which has no Levin antiderivative to weight by. |
| `[05-zero-width-span-raises]` | levin-refactor | `adaptive_levin_sincos((5.0, 5.0), …)` raises `ValueError` from `build_Levin_data()`; the early return the README describes does not exist. |

### 1.3 The $T_k$ / $G_k$ numerical-precision campaign

| Issue | Board | Hook |
|---|---|---|
| *(none open)* | | `[07-phase-spline-chunking-precision]` was closed WONTFIX on the expectation that this campaign **removes** the chunked splines; if that plan changes, reopen it. |

---

## 2. Error-bound completeness

A family: several layers compute an error estimate that is correct for what it claims and does not
bound the true error. Closing any of them properly needs the representation error of §1.1 first.

| Issue | Board | Hook |
|---|---|---|
| `[09-abserr-is-a-quadrature-bound]` | source-remediation | `total_abserr` is the linear sum of quadrature estimates and nothing else; the true residual is up to 4.4e4× larger. `total_converged = False` is not a failure. |
| `[12-atol-too-loose-for-the-source-integral]` | source-remediation | 58 % of work items have a raw integral below `DEFAULT_QUADRATURE_ATOL = 1e-25`, so their tolerance is met before any work is done. Needs a production decision: scale `atol` with the integrand, go `rtol`-only, or lower the default for this stage. |
| `[09-abserr-does-not-bound-phase-spline-floor]` | levin-refactor | `quad_JJJ`/`quad_YJJ`'s `abserr` misses the true error against the analytic oracle by up to 11.5× on 5 of 7 three-Bessel closed forms. |
| `[09-quadsource-total-error-incomplete]` | levin-refactor | Partly superseded: `source-remediation` prompt 09 added `total_abserr`. Re-read against the current tree before acting. |

---

## 3. Inert — recorded so a later reader does not misread a residual

No action defined. These are floors on what a test may *assert*, not on what the code computes.

| Issue | Board | Hook |
|---|---|---|
| `[01-genericeos-tz-spline-floor]` | source-remediation | Whether the `T(z)` spline grid is adequately defined. A hot-fix's fixed 500 points give 1.3e-9 at `max_z=1e4`, 6.4e-7 at the default 1e20; it is why prompt 01's test asserts 1e-8, not 1e-10. Two sign bugs in the grid's *range* were fixed 2026-09-10 and are not part of this. |
| `[03-derivative-pad-clamp-on-coarse-grids]` | source-remediation | The background derivative-fit padding is clamped near $z=0$; harmless at the shipped 100 samples/decade, binds at 50. A trap only if `source_samples_log10z` is lowered. |

---

## 4. Verification debt

Something was asserted statically or on a stand-in, and a live exercise is still owed.

| Issue | Board | Hook |
|---|---|---|
| `[04-read-table-service]` | backport-modules | Audit §8 item 6; item 5 was closed live by prompt 10. |
| `[05-persist-handler-split]` | backport-modules | Audit §8 item 7: a real driver run exercising the `store_handler`/`persist_handler` split end to end. |
| `[00-plan-vs-tree-corrections]` | transfer-remedial | Four `DRAFT-PLAN.md` claims that do not survive reconciliation against the tree; the prompts are built on the corrected versions. |
| `[00-qsi-three-bessel-levin-excluded]` | transfer-remedial | `_three_bessel_Levin`'s eight `adaptive_levin_sincos` calls supply no `theta_deriv`, so Levin differentiates the raw phase spectrally there. |
| `[01-scipy-jv-yv-high-order-boundary]` | transfer-remedial | The silent Amos boundary is order dependent and applies to `jv`/`yv`, not only `hankel1e`: 7.13e8 above $\nu\approx86$. Guarded in the harness; the order threshold is bracketed [85.5, 88.5], not pinned, and not yet a test. |
| `[03-draft-plan-tail-coefficient-wrong]` | transfer-remedial | `DRAFT-PLAN.md` §7.2 and prompt 03 print 15360 for the third DLMF 10.18.18 denominator; it is 5120 (fourth: 229376). Shipped code is right and pins it; the plan text is not edited, so prompt 09 owes the correction in `docs/`. |
| `[04-achieved-estimates-exclude-the-sampling-floor]` | transfer-remedial | `NearRegionData.achieved_*` resamples the same `hankel1e` it interpolates, so it estimates interpolation error only. **Narrowed by prompt 05:** the published `theta_abserr` now adds a 3e-13 sampling floor and 4ε of evaluation arithmetic, and is tested never to under-report; only the size of the 3e-13 constant is still open. |
| `[05-quadsource-order-check-docstring-stale]` | transfer-remedial | `bessel_phase` now returns a `"nu"` key, so `QuadSourceIntegral._check_bessel_order`'s docstring (and its reason for checking the order numerically) is obsolete. No functional impact; hand to `source-remediation`. |
| `[06-levin-theta-docstring-stale]` | transfer-remedial | `levin_quadrature.py:2750` says `theta` is always used to decide subdivision; it is not (`:1038`, `:1090`), and prompt 06 has direct evidence — a `theta` that raises gives bit-identical results. `AdaptiveLevin/` is forbidden here. |
| `[06-measure-bessel-phase-num-chunks]` | transfer-remedial | Prompt 01's own diagnostic script reads `phase.num_chunks`, which prompt 05 removed, so its current-tree sections raise `AttributeError`. Prompt 09 owns `docs/`. |
| `[06-three-bessel-plot-calls-a-non-callable-phase]` | transfer-remedial | `QuadSourceIntegral_debug.three_bessel_plot` calls the phase object directly; no phase class has ever defined `__call__`, so it is dead in the same way `plot_besssel_phase.py` was. Repair-or-delete, unowned. |
| `[07-generic-K-product-rounding]` | transfer-remedial | A three-Bessel phase group still carries one rounding of the product $Kx$: 1.5e-5 rad at $K=0.1$, $x=10^{12}$, the same for the new and old routes, so a *non-resonant* group gains nothing from the $Kt+C+R$ restructure. A floor on prompt 08. |
| `[08-3bessel-chebyshev-order-is-now-the-limit]` | transfer-remedial | `DEFAULT_3BESSEL_CHEBYSHEV_ORDER = 12` now binds the two $(0,0,0)$ three-Bessel oracles — order 20 buys J000 and Y000 three orders — while making the other five 4×–1500× worse. A per-integrand or convergence-checked order, not a constant to bump. |
| `[08-tk-fixture-scipy-comparison-unasserted]` | transfer-remedial | `test_tk_source_functions`'s `err_scipy` is the campaign's headline downstream number (1.985e-06 → 3.021e-08) and is printed, not asserted; prompt 08 may only change comments and tolerances in that file. |
| `[08-test-three-bessel-tolerances-unassigned]` | transfer-remedial | README §4 says prompt 08 re-tightens `test_three_bessel.py`; the prompt's own file list and README §3 exclude it, so it was not touched. Needs a decision, not analysis. |
| `[08-3bessel-plot-cost-dominates-the-suite]` | transfer-remedial | `test_3bessel_analytic` spends its whole wall clock (21.2 min for one test) evaluating 250-point grids of three-Bessel integrals to draw figures, not on assertions — which is why a module-level failure survived three prompts. Proposal only; nothing implemented. |

---

## 5. Standing caveat that is not an issue

**The verification runs never reached production $x$.** `source-remediation`'s run A used
`zend = 1e7` to stay inside radiation domination where `analytic_rad` is a valid oracle, so its
largest accumulated phase was $x = 4.63\times10^5$ against $x\sim1.4\times10^7$ at the production
`zend = 0.1` for the largest $k$. Every "verified live" claim in that campaign carries this
ceiling, and `[12-phase-spline-error-grows-with-x]` is the term it matters most for.
