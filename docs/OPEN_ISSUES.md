# Open issues — project-wide index

**Last updated:** 2026-09-11 · **50 open** across six campaigns.

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
[`transfer-remedial`](../prompts/transfer-remedial/IMPLEMENTATION_STATE.md) ·
[`GkTk-remedial`](../prompts/GkTk-remedial/IMPLEMENTATION_STATE.md) ·
[`qsi-phase-groups`](../prompts/qsi-phase-groups/IMPLEMENTATION_STATE.md)

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
| `[05-numeric-region-is-now-the-accuracy-floor]` | source-remediation | A cubic spline loses 1–2 orders in its last two intervals, and the numeric grid *ends* at the hand-over. |
| `[06-source-spline-residual-vs-handover]` | source-remediation | $f$ oscillates at twice the transfer-function phase, so its spline is the worst of the three; 1.4e-04 of envelope on real rows, but $O(1)$ if the hand-over is allowed to fall to the bottom of `main.py`'s search window. |
| `[07-lg-derivative-truncation-at-handover]` | source-remediation | The irreducible one: `omega`/`dlnM_dz` are LG quantities, off by $O(x^{-4})$. Grid-independent — only a deeper hand-over helps. |
| `[00-tk-lg-truncation-floor]` | GkTk-remedial | The $T_k$ LG representation is not exact even in radiation: ~$1.4\times10^{-4}$ of the envelope at the production hand-over $x_T\approx15.5$, scaling as $x_i^{-3}$. A later hand-over ($x_T=50$ gives $4\times10^{-6}$), a higher-order LG frequency, or the Bessel exact form are the levers. |

### 1.2 The `QuadSourceIntegral` phase-groups campaign

`prompts/qsi-phase-groups` — one prompt, opened and completed 2026-09-11. `_three_bessel_Levin`'s
eight Levin calls were the last Bessel-phase consumer in the tree assembled by summing raw phases;
they now pass `BesselPhaseGroup.levin_theta()`.

| Issue | Board | Hook |
|---|---|---|
| *(none open)* | | All three entries closed by prompt 01. The one thing it left behind is a *different* call site and is indexed in §2 below. |

### 1.3 The `AdaptiveLevin` Clenshaw–Curtis fallback campaign

| Issue | Board | Hook |
|---|---|---|
| `[10-levin-wholesale-cc-fallback]` | source-remediation | 2.65–3.56× wall clock for 1.00–1.44× integrand evaluations, i.e. per-region overhead, not wasted work. Route wholesale to Clenshaw–Curtis at the outset instead of bisecting into it. **Do not** re-add a threshold in `QuadSourcePolicy` — that is defect A4 in weaker form. |
| `[03-fallback-cost-on-difference-groups]` | levin-refactor | The total-variation gate now tests the Levin branch's eagerly computed comparison children too; structurally required, but raises evaluation counts 2.2–3.4× on three-Bessel groups. |
| `[04-roundoff-floor-can-be-infinite]` | levin-refactor | `phase_err`/`abserr_roundoff`/`total_err` can be `+inf` at an interior stationary point inside a strongly oscillatory Levin region. |
| `[04-theta-abserr-cc-branch-proxy]` | levin-refactor | The endpoint term is exact on a Levin region but a 50/50 split of a lumped proxy on a Clenshaw–Curtis one, which has no Levin antiderivative to weight by. |
| `[05-zero-width-span-raises]` | levin-refactor | `adaptive_levin_sincos((5.0, 5.0), …)` raises `ValueError` from `build_Levin_data()`; the early return the README describes does not exist. |

### 1.4 The $T_k$ / $G_k$ numerical-precision campaign

Now running as [`prompts/GkTk-remedial/`](../prompts/GkTk-remedial/README.md) (2026-09-10; 13
prompts plus two follow-ups, 12 executed). `[07-phase-spline-chunking-precision]` was closed WONTFIX on
the expectation that this campaign **removes** the chunked splines — its prompt 08 has done so.

| Issue | Board | Hook |
|---|---|---|
| `[12-phase-spline-error-grows-with-x]` | source-remediation | Stored-phase re-spline error $\simeq h^4x/384$, growing **linearly in $x$**; ~1 % of envelope extrapolated to production. **Reassigned here from §1.1** (2026-09-10): the campaign's prompts 09–10 evaluate the leading term from a table and spline only the residual. |
| `[00-unresolved-osc-print-policy]` | GkTk-remedial | With its units fixed and evaluated on the caller's actual grid, `has_unresolved_osc` will fire on essentially every $G_k$ numeric object. Prompt 11 measures the rate; the user chooses the print policy. |
| `[00-consumer-anchoring-floor]` | GkTk-remedial | `PrimitivePhase` reduces $k\Delta\tau$ against a global anchor, so `theta_mod_2pi` carries the $\varepsilon k\tau$ floor ($9\times10^{-4}$ rad at $k=3\times10^8$). Per-region anchoring is the follow-up. **Measured by prompt 09:** the floor is now the whole error — 4.189e-8 rad = 2.81 ulp of the span at $k=10^8$, with $\varphi$ itself recovered to 2.157e-10 rad. |
| `[01-offgrid-accessor-cost-on-qcd]` | GkTk-remedial | The interval accessor costs 102 µs on `QCD_Cosmology` with both endpoints off-grid, above README §4.3's 50 µs stop threshold. **Narrowed by prompt 03:** the shipped accessor is 52 µs both-off-grid, 26 µs one-off-grid, 0.33 µs on-grid (the production case). **Narrowed by prompt 06:** the producers only ever pair the off-grid anchor with an on-grid sample — 464 partial evaluations inside a 0.031 s object. **Narrowed by prompt 09, not closed:** the Levin consumer evaluates off-grid by construction, but always with exactly one off-grid endpoint (the anchor is a grid node), so the 26 µs figure applies and the 50 µs line is not crossed. |
| `[07-tk-per-object-cost-is-all-setup]` | GkTk-remedial | A `TkWKBIntegration` object at $k=3\times10^8$ costs 0.049–0.052 s, straddling prompt 07 §3 item 6's 0.05 s; all of it is setup. 5,840 of its 11,376 integrand evaluations build a per-$k$ residual table that, at one object per $k$, nothing amortises, and 5,536 are the leading table's off-grid anchor panel recomputed once per sample — the split prompt 14 applied to $\rho$ but not to $\tau_s$. **Widened by prompt 09:** the same recomputation would hit any consumer with an off-grid anchor, which is prompt 10's $z_{\rm init}$. |
| `[08-docs-scripts-reference-removed-chunking]` | GkTk-remedial | Two `docs/` reproduction scripts (`t5_spline.py`, `measure.py`) read `phase_spline` internals (`_chunk_list`, `_splines`, `_match_chunk`) that prompt 08 deleted with chunking; they documented the chunked tree they ran on and were not edited. |
| `[10-residual-spline-end-condition]` | GkTk-remedial | Prompt 10 §3 item 3's 1e-10 relative on $\omega$ vs `theta_deriv` is missed at one abscissa per equation of state — 1.0492e-10 at $w=1/3$, the not-a-knot end condition of the cubic residual spline at the top of the WKB region (5.6e-12 from the fifth sample inwards, 4.249e-08 before). `spline_order=5` gives 9.7e-12 over the whole region but needs six samples against `MIN_SPLINE_DATA_POINTS = 5`. **Assigned (2026-09-11): prompt 13**, to re-measure on the real background before anyone pays for the quintic. |
| `[10-transfer-remedial-tolerance-comments-stale]` | GkTk-remedial | Five tolerance comments `8ba9159` wrote in `test_tk_source_functions.py` now describe the consumer re-spline prompt 10 deleted and quote numbers three to four orders above the new measurements. Not edited — `8ba9159`'s text was a stop condition for prompt 10 — and every assertion still passes. |
| `[10-wrap-theta-loop-at-large-phase]` | GkTk-remedial | `wrap_theta` reduces by adding $2\pi$ in a loop, so at $|\theta|\sim10^6$ rad it takes ~1.6e5 iterations and reconstructs $\theta$ only to 1.39e-06 rad. Inert in production (its one caller passes `mod + delta`), a trap for fixtures; `WKB_mod_2pi` is exact. |

---

## 2. Error-bound completeness

A family: several layers compute an error estimate that is correct for what it claims and does not
bound the true error. Closing any of them properly needs the representation error of §1.1 first.

| Issue | Board | Hook |
|---|---|---|
| `[09-abserr-is-a-quadrature-bound]` | source-remediation | `total_abserr` is the linear sum of quadrature estimates and nothing else; the true residual is up to 4.4e4× larger. `total_converged = False` is not a failure. |
| `[12-atol-too-loose-for-the-source-integral]` | source-remediation | 58 % of work items have a raw integral below `DEFAULT_QUADRATURE_ATOL = 1e-25`, so their tolerance is met before any work is done. Needs a production decision: scale `atol` with the integrand, go `rtol`-only, or lower the default for this stage. |
| `[09-abserr-does-not-bound-phase-spline-floor]` | levin-refactor | `quad_JJJ`/`quad_YJJ`'s `abserr` misses the true error against the analytic oracle by up to 11.5× on 5 of 7 three-Bessel closed forms. **Measured false on the current tree** (`transfer-remedial` prompt 08): 7 of 7 now bound, `true/reported` between 9.8e-06 and 1.5e-04. |
| `[09-quadsource-total-error-incomplete]` | levin-refactor | Partly superseded: `source-remediation` prompt 09 added `total_abserr`. Re-read against the current tree before acting. |
| `[01-cosmological-group-declares-no-phase-error]` | qsi-phase-groups | The ninth `adaptive_levin_sincos` call in `QuadSourceIntegral.py` — `phase_group_Levin_integral`'s, over cosmological `phase_spline` phases — still supplies three keys, not four; `phase_spline` reports no fit accuracy to declare. |

---

## 3. Inert — recorded so a later reader does not misread a residual

No action defined. These are floors on what a test may *assert*, not on what the code computes.

| Issue | Board | Hook |
|---|---|---|
| `[01-genericeos-tz-spline-floor]` | source-remediation | Whether the `T(z)` spline grid is adequately defined. A hot-fix's fixed 500 points give 1.3e-9 at `max_z=1e4`, 6.4e-7 at the default 1e20; it is why prompt 01's test asserts 1e-8, not 1e-10. Two sign bugs in the grid's *range* were fixed 2026-09-10 and are not part of this. |
| `[03-derivative-pad-clamp-on-coarse-grids]` | source-remediation | The background derivative-fit padding is clamped near $z=0$; harmless at the shipped 100 samples/decade, binds at 50. A trap only if `source_samples_log10z` is lowered. |
| `[00-tk-superhorizon-ic-series]` | GkTk-remedial | Once the $T_k$ numeric `atol` is fixed (prompt 12), the floor is the super-horizon initial condition $T=1,T'=0$ at $2.5\times10^{-6}$; removable with the series $T\approx1-x^2/10$, a spec-level decision. |
| `[01-lambdacdm-hubble-rounding-floor]` | GkTk-remedial | `LambdaCDM.Hubble` carries 2–9e-15 relative in double precision, which floors $\Delta\tau$ over one grid interval at $4$–$6\times10^{-5}$ rad at $k=3\times10^8$ whatever the Gauss order or storage width. |
| `[02-qcd-reference-floor]` | GkTk-remedial | The QCD $\tau$/$\tau_s$ references in `wkb_reference_data.json` are themselves good only to 1.9e-14 relative, which is exactly where prompt 02's order-4 tables land. Do not assert tighter for QCD $\tau$ at the nodes. |
| `[02-qcd-T-z-spline-node-tolerance]` | GkTk-remedial | `_solve_T_z`'s `root_scalar(xtol=1e-6, rtol=1e-4)` leaves the $T(z)$ spline's node values up to 2.1e-5 relative from a tight re-solve ($\sim4\times10^{-5}$ in $H$). A model-fidelity bound, not a quadrature error; distinct from `[01-genericeos-tz-spline-floor]`, which is about the grid. |
| `[03-qcd-short-baseline-reference-endpoint-rounding]` | GkTk-remedial | The QCD short-baseline references in `wkb_reference_data.json` integrate between rounded `log1p(z)` endpoints and carry up to ulp(u)/W ≈ 1e-13 relative on the 37 % fractions; the shipped table agrees with an exact-endpoint `quad` to ≤ 8.8e-16. Assert README §6's 1e-13 for QCD short baselines, not the JSON's self-agreement. |
| `[03-integrationsolver-stepping-minimum-lookup]` | GkTk-remedial | `IntegrationSolver` lookups match `stepping >= requested`; harmless while every table is order 4, but a second Gauss order under the label `cumulative-GL` could be served by the other order's row. |
| `[04-background-rhs-evaluations-count]` | GkTk-remedial | `compute_background` builds three tables but `IntegrationData` has one counter, which prompt 03's test pins to the $\tau$ table alone; the other two counts are payload keys, so the persisted `RHS_evaluations` understates the build 3×. |
| `[06-metadata-column-headroom]` | GkTk-remedial | The WKB integrations' `metadata` column is `String(256)`. **Narrowed by prompt 14:** with its `rho_reused` key the longest payload is 227 characters, so 29 remain; a test now asserts the length rather than letting it overflow. SQLite does not enforce it, PostgreSQL would. Count before adding a key. |
| `[06-docs-scripts-reference-removed-ode]` | GkTk-remedial | Nine `docs/` reproduction scripts import symbols the campaign removed: seven the phase ODE (`integrate_phase_function`, `stage_*_evolution`, `DEFAULT_OMEGA_WKB_SQ_MAX`, prompt 06), and `TK_04_WKB_reconstruction.py` and `baseline_k1e5.py` the friction ODE `friction_RHS` (prompt 07, relocated into prompt 04's test). They documented the tree they ran on and were not edited. |
| `[14-residual-range-top-margin]` | GkTk-remedial | The residual table's nodes stop where the LG frequency falls below half its leading term, not at its turning point: on QCD the sign near the turning point is unresolved and a panel with two positive nodes can hold an abscissa with $\omega^2<0$. Inert: production anchors sit three e-folds inside the horizon and the cut at ~1.25, a margin of 1.74–16 e-folds over all twelve (model, sector, $k$) cases. **Corrected 2026-09-11:** log 14's stated reason — that the cut is above every criterion-satisfying node — is not general (QCD/`Gk`/$k=10^7$), and no test pins either statement; prompt 13 measures the margin. |
| `[14-rhs-evaluations-depend-on-build-order]` | GkTk-remedial | The WKB integrations' `RHS_evaluations` now counts only what a call spent, so the first object of a $(model, k, sector)$ in a worker stores ~7,000 and every later one a few hundred; which is first depends on the scheduler. Payload data, part of no lookup key. |

---

## 4. Verification debt

Something was asserted statically or on a stand-in, and a live exercise is still owed.

| Issue | Board | Hook |
|---|---|---|
| `[04-read-table-service]` | backport-modules | Audit §8 item 6; item 5 was closed live by prompt 10. |
| `[05-persist-handler-split]` | backport-modules | Audit §8 item 7: a real driver run exercising the `store_handler`/`persist_handler` split end to end. |
| `[01-scipy-jv-yv-high-order-boundary]` | transfer-remedial | The silent Amos boundary is order dependent and applies to `jv`/`yv`, not only `hankel1e`: 7.13e8 above $\nu\approx86$. Guarded in the harness; the order threshold is bracketed [85.5, 88.5], not pinned, and not yet a test. |
| `[04-achieved-estimates-exclude-the-sampling-floor]` | transfer-remedial | `NearRegionData.achieved_*` resamples the same `hankel1e` it interpolates, so it estimates interpolation error only. **Narrowed by prompt 05:** the published `theta_abserr` now adds a 3e-13 sampling floor and 4ε of evaluation arithmetic, and is tested never to under-report; only the size of the 3e-13 constant is still open. |
| `[06-levin-theta-docstring-stale]` | transfer-remedial | `levin_quadrature.py:2750` says `theta` is always used to decide subdivision; it is not (`:1038`, `:1090`), and prompt 06 has direct evidence — a `theta` that raises gives bit-identical results. `AdaptiveLevin/` is forbidden here. |
| `[06-measure-bessel-phase-num-chunks]` | transfer-remedial | Prompt 01's own diagnostic script reads `phase.num_chunks`, which prompt 05 removed, so its current-tree sections raise `AttributeError`. Not in prompt 09's file list, so left unfixed. |
| `[06-three-bessel-plot-calls-a-non-callable-phase]` | transfer-remedial | `QuadSourceIntegral_debug.three_bessel_plot` calls the phase object directly; no phase class has ever defined `__call__`, so it is dead in the same way `plot_besssel_phase.py` was. Repair-or-delete, unowned. |
| `[08-3bessel-chebyshev-order-is-now-the-limit]` | transfer-remedial | `DEFAULT_3BESSEL_CHEBYSHEV_ORDER = 12` now binds the two $(0,0,0)$ three-Bessel oracles — order 20 buys J000 and Y000 three orders — while making the other five 4×–1500× worse. A per-integrand or convergence-checked order, not a constant to bump. |
| `[08-tk-fixture-scipy-comparison-unasserted]` | transfer-remedial | `test_tk_source_functions`'s `err_scipy` is the campaign's headline downstream number (1.985e-06 → 3.021e-08) and is printed, not asserted; prompt 08 may only change comments and tolerances in that file, and prompt 09's `source-remediation` hand-off is restricted to one entry, so this was not folded in either. |
| `[08-3bessel-plot-cost-dominates-the-suite]` | transfer-remedial | `test_3bessel_analytic` spends its whole wall clock (21.2 min for one test) evaluating 250-point grids of three-Bessel integrals to draw figures, not on assertions — which is why a module-level failure survived three prompts. Proposal only; nothing implemented. |
| `[03-backgroundmodelvalue-build-path]` | GkTk-remedial | `sqla_BackgroundModelValue_factory.build()`'s query-existing-row branch inserts with key `"wkb_serial"` (column is `model_serial`) and reads `row_data.Hubble` (select has `Hubble_GeV`); confirmed by prompt 03, never exercised by production, not repaired. |
| `[09-bessel-tier-hardcoded-repo-path]` | transfer-remedial | `bessel_tier.py`'s hardcoded main-checkout `sys.path` entry silently shadows a worktree's own `LiouvilleGreen` package; a re-run from a worktree measures the wrong tree with no warning. Discovered while re-running the $\kappa=1000$ benchmark tier. |

---

## 5. Standing caveat that is not an issue

**The verification runs never reached production $x$.** `source-remediation`'s run A used
`zend = 1e7` to stay inside radiation domination where `analytic_rad` is a valid oracle, so its
largest accumulated phase was $x = 4.63\times10^5$ against $x\sim1.4\times10^7$ at the production
`zend = 0.1` for the largest $k$. Every "verified live" claim in that campaign carries this
ceiling, and `[12-phase-spline-error-grows-with-x]` is the term it matters most for.
