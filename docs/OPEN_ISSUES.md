# Open issues — project-wide index

**Last updated:** 2026-09-17 · **81 open** across ten campaigns.

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
[`qsi-phase-groups`](../prompts/qsi-phase-groups/IMPLEMENTATION_STATE.md) ·
[`phase-representation`](../prompts/phase-representation/IMPLEMENTATION_STATE.md) ·
[`qcd-background-audit`](../prompts/qcd-background-audit/IMPLEMENTATION_STATE.md) ·
[`tolerance-convergence`](../prompts/tolerance-convergence/IMPLEMENTATION_STATE.md) ·
[`background-solver-robustness`](../prompts/background-solver-robustness/IMPLEMENTATION_STATE.md)

---

## 1. Assigned to a future campaign

Work is identified and owned; the issue is parked deliberately, not forgotten.

### 1.1 The hand-over campaign

The numeric→Liouville–Green seam of $T_k$ and $G_k$. These eight are **one place** and must be
attacked together — prompt 12 of `source-remediation` could not separate their contributions by
measurement alone at production $x$. (`[11-stop-point-root-tolerance]` joined them 2026-09-12: the
numeric stop point $z_{\rm init}$ is a `root_scalar` root, so where the seam sits and how precisely
it is located are the same decision.) Background reading:
[`docs/lg-phase-and-handover-followup-2026-09.md`](lg-phase-and-handover-followup-2026-09.md).

| Issue | Board | Hook |
|---|---|---|
| `[08-handover-clamp-error]` | source-remediation | The WKB grid starts below `crossover_z`, so the LG accessors are clamped across a gap; a one-step gap moves `total` by 5.1e-03. |
| `[12-handover-clamp-error-in-production]` | source-remediation | Supersedes the above for magnitude: on real rows the gap is universal and costs 6.6e-02–4.6e-01 against 6.2e-05 unclamped. **The campaign's main outstanding accuracy decision.** |
| `[05-numeric-region-is-now-the-accuracy-floor]` | source-remediation | A cubic spline loses 1–2 orders in its last two intervals, and the numeric grid *ends* at the hand-over. |
| `[06-source-spline-residual-vs-handover]` | source-remediation | $f$ oscillates at twice the transfer-function phase, so its spline is the worst of the three; 1.4e-04 of envelope on real rows, but $O(1)$ if the hand-over is allowed to fall to the bottom of `main.py`'s search window. |
| `[07-lg-derivative-truncation-at-handover]` | source-remediation | The irreducible one: `omega`/`dlnM_dz` are LG quantities, off by $O(x^{-4})$. Grid-independent — only a deeper hand-over helps. |
| `[00-tk-lg-truncation-floor]` | GkTk-remedial | The $T_k$ LG representation is not exact even in radiation: ~$1.4\times10^{-4}$ of the envelope at the production hand-over $x_T\approx15.5$, scaling as $x_i^{-3}$. A later hand-over ($x_T=50$ gives $4\times10^{-6}$), a higher-order LG frequency, or the Bessel exact form are the levers. |
| `[11-stop-point-root-tolerance]` | GkTk-remedial | `find_phase_extremum`'s `root_scalar(xtol=1e-6, rtol=1e-4)` places the stop point only to $\sim10^{-4}z$, so $|G'|/(|G|\omega)$ there is 9.8e-6 (pre-prompt-11) to 6.5e-5, not the 1e-12 prompt 11 §3 asked to assert. Harmless downstream — `store()` rotates $(G,G')$ — but $z_{\rm init}$ is this root, so tightening it is a hand-over-campaign decision with a datastore regeneration attached. |
| `[03-numeric-g-consumer-spline-is-the-dominant-error-near-the-hand-over]` | tolerance-convergence | The consumer of the numeric $G_k$ is a cubic `make_interp_spline` in $\log(1+z_{\rm source})$ over the **source-grid** nodes (`GkSourcePolicyData.py:654-680`), and its interpolation error is **1.6e-04 to 1.9e-04** of the envelope three e-folds inside the horizon and up to **9.4e-03** at four, where `main.py:1884` stops building the target — against **2.6e-07** for the solver at the production `(1e-10, 1e-8)`. Measured at 50 $k$ on three models, **version-2 grid at each cosmology's own anchor**, floor uncertainty 0.00 %, probe within 5.6e-12 of the exact $G$ on the control. It confirms review §10.1's inherited 1e-5–1e-4 and shows the "two orders" understates the dominance (×631 to ×37,700). The cause is spacing, not tolerance: the source lattice carries up to **1.24 rad** of $G$'s oscillation per interval there, and the density criterion that sets it is sized for the phase-residual spline, not for $G$. **Assigned 2026-09-17 on the user's D1 acceptance: the source grid's spacing at the hand-over is post-campaign work.** A design decision, not a tuning one — the density criterion gains $G$'s oscillation as a second consumer, the hand-over moves deeper, or `GkSourcePolicyData` splines the LG amplitude and phase instead. |

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

Ran as [`prompts/GkTk-remedial/`](../prompts/GkTk-remedial/README.md) (2026-09-10 to 2026-09-13;
13 prompts plus seven follow-ups — 14, 15, 16, 17, 18, 19, 20 — **all twenty executed**).
`[07-phase-spline-chunking-precision]` was closed WONTFIX on the expectation that this campaign
**removes** the chunked splines — its prompt 08 has done so. The close-out measurements are
[`docs/gktk-remedial-verification.md`](gktk-remedial-verification.md); the rows below are what
prompt 13 left open.

| Issue | Board | Hook |
|---|---|---|
| `[12-phase-spline-error-grows-with-x]` | source-remediation | Stored-phase re-spline error $\simeq h^4x/384$, growing **linearly in $x$**; ~1 % of envelope extrapolated to production. **Reassigned here from §1.1** (2026-09-10): the campaign's prompts 09–10 evaluate the leading term from a table and spline only the residual. |
| `[00-consumer-anchoring-floor]` | GkTk-remedial | `PrimitivePhase` reduces $k\Delta\tau$ against a global anchor, so `theta_mod_2pi` carries the $\varepsilon k\tau$ floor ($9\times10^{-4}$ rad at $k=3\times10^8$). Per-region anchoring is the follow-up. **Measured by prompt 09:** the floor is now the whole error — 4.189e-8 rad = 2.81 ulp of the span at $k=10^8$, with $\varphi$ itself recovered to 2.157e-10 rad. |
| `[07-tk-per-object-cost-is-all-setup]` | GkTk-remedial | A `TkWKBIntegration` object at $k=3\times10^8$ costs 0.049–0.052 s, straddling prompt 07 §3 item 6's 0.05 s; all of it is setup. 5,840 of its 11,376 integrand evaluations build a per-$k$ residual table that, at one object per $k$, nothing amortises, and 5,536 are the leading table's off-grid anchor panel recomputed once per sample — the split prompt 14 applied to $\rho$ but not to $\tau_s$. **Widened by prompt 09:** the same recomputation would hit any consumer with an off-grid anchor, which is prompt 10's $z_{\rm init}$. |
| `[08-docs-scripts-reference-removed-chunking]` | GkTk-remedial | Two `docs/` reproduction scripts (`t5_spline.py`, `measure.py`) read `phase_spline` internals (`_chunk_list`, `_splines`, `_match_chunk`) that prompt 08 deleted with chunking; they documented the chunked tree they ran on and were not edited. |
| `[10-transfer-remedial-tolerance-comments-stale]` | GkTk-remedial | Five tolerance comments `8ba9159` wrote in `test_tk_source_functions.py` now describe the consumer re-spline prompt 10 deleted and quote numbers three to four orders above the new measurements. Not edited — `8ba9159`'s text was a stop condition for prompt 10 — and every assertion still passes. |
| `[20-wkb-rows-consume-numeric-initial-data]` | GkTk-remedial | `Gk`/`TkWKBIntegration` take $z_{\rm init}$, $G_{\rm init}$/$T_{\rm init}$ and the derivative from the numeric stop point and are keyed independently of the numeric row: no foreign key, and the initial values are stored `nullable=False` but never filtered. Covered in practice only because `z_init` is filtered as an absolute `1e-7` against $z\sim10^{12}$, i.e. exactly — measured on QCD at $k=4.972\times10^7$, the two break-point policies move $z_{\rm init}$ by 4.59e5 and the lookup misses. A change moving the stop *values* without moving $z_{\rm init}$ would be served a stale row. |
| `[10-wrap-theta-loop-at-large-phase]` | GkTk-remedial | `wrap_theta` reduces by adding $2\pi$ in a loop, so at $|\theta|\sim10^6$ rad it takes ~1.6e5 iterations and reconstructs $\theta$ only to 1.39e-06 rad. Inert in production (its one caller passes `mod + delta`), a trap for fixtures. The companion defect in `WKB_mod_2pi`'s cycle count was fixed by `phase-representation` prompt 01 (2026-09-13), so that function is now exact in both halves; this loop is not. |
| `[13-scoped-run-driver-k-grid-literal]` | GkTk-remedial | `docs/source-remediation-verification/scoped_pipeline_run.py` matches a `main.py` k-grid literal that `f17f2d4` renamed to `NUMBER_SOURCE_K_VALUES`/`NUMBER_RESPONSE_K_VALUES`, so it finds zero occurrences and raises rather than running. The `source-remediation` Layer 2 is not reproducible by its own documented command; prompt 13 copied the driver into `docs/gktk-remedial/` rather than editing another campaign's file. |

---

### 1.5 The tolerance and convergence campaign

Planned as [`prompts/tolerance-convergence/`](../prompts/tolerance-convergence/README.md)
(2026-09-12; **rebased 2026-09-16 at `acd5b8e`, re-anchored the same day at `bc6dc97` onto the
tree `prompts/background-solver-robustness` left — six prompts plus the insertion 02a; 01, 02,
02a, 03, 03a and 04 have all run, and 05–06 wait on the user**). One reusable
convergence test applied to every accuracy parameter in the pipeline on all three models, anchored
to the constant-$w$ closed forms in `ComputeTargets/analytic_{Gk,Tk}.py`, and the parameters
decoupled so each quantity carries its own justified one. **Unblocked** by `prompts/GkTk-remedial`
prompts 18 and 19 (2026-09-13): the numeric ODE is split at the cosmology's declared
non-smoothness, *which* kind is the sector's choice, and `[17-qcd-reference-not-converged]` is
**closed**. `prompts/qcd-background-audit` then improved the same figures again — worst QCD
reference-convergence drift **7.08e-09** ($T_k$) and **3.67e-09** ($G_k$) against the 3.4e-08
criterion, zero offenders at all 50 production wavenumbers
(`docs/qcd-background-audit/PER-SECTOR-POLICY.md` §2, §4).

**The rebase changed the campaign's subject**
([`RECONCILIATION.md`](../prompts/tolerance-convergence/RECONCILIATION.md) scores every claim of the
2026-09-12 plan against the tree). The plan counted five compute targets sharing two constants
across four object types; measured, **eight** object types are keyed on an accuracy parameter, the
shared pair keys six of them, and of those six **only one uses the value it is given**. Two targets
the plan never mentioned are now in scope: `wavenumber_exit_time`, a live `root_scalar` in
$\log(1+z)$ that fixes where every grid begins, and `BackgroundModel`, whose three Gauss orders are
the largest instance of the case where the knob is an integer order rather than a tolerance.

**Prompt 02 took that count against the tree, 2026-09-16, and it is off by one in the same
direction.** [`docs/tolerance-convergence/TOLERANCE-INVENTORY.md`](tolerance-convergence/TOLERANCE-INVENTORY.md)
derives the keyed tables from `Datastore/SQL/Datastore.py`'s factory map rather than from any
document: **nine** object types are keyed on an `atol`/`rtol` pair, not eight — the ninth,
`OneLoopIntegral`, is a schema with no computation behind it — and two policy tables are keyed on
`Levin_threshold` besides. Every row of README §2 (a) is **confirmed**; what is wrong is the
count, and the campaign's own one-sentence summary, which says one of the six `(atol, rtol)`
sharers uses the value when **two** do. Seventeen accuracy parameters set the accuracy of a stored
quantity and sit in no lookup key at all, of which `[20-wkb-gauss-orders-not-in-lookup-key]` names
five.

**Two cost figures this index previously carried were wrong and are corrected here.** (i) A QCD
$T_k$ numeric object is **8,986** right-hand-side evaluations under its own `BREAK_POINT_ALL`
policy, not ~31.5k: `qcd-background-audit` prompt 07 took the ~404 spline knots out of
`integration_break_points`, so the wider policy costs **+0.99 %** rather than +220 %
(`PER-SECTOR-POLICY.md` §5). (ii) The ~65,000-objects-per-model sector is $G_k$, not $T_k$:
`TkNumericIntegration` is one object per $k$, 50 per model. One decade of `rtol` at +23–25 %
evaluations is therefore free where it was measured and is the campaign's whole compute decision
where it was not.

| Issue | Board | Hook |
|---|---|---|
| `[04-convergence-floor-used-as-a-test-threshold]` | tolerance-convergence | Two test modules assert *production error ≤ 3 × `convergence.models.QCDModel["branch+knots"][q]["json_vs_reference_max_rel"]`* — `test_background_tau.py:324` ($\tau$) and `test_background_cs_tau_friction.py:606` ($c_s\tau$) — and the two sides are **independent quantities**: the right is how far the JSON's reference values sit from a converged adaptive reference, the left how far the production order-4 table sits from those same JSON values. It only ever worked because both sat at ~2e-14. Prompt 04's regeneration takes the floors to **3.223619e-16** and **4.354138e-16** (58× and 43× tighter, the QCD background having been replaced) while the numerators stay at **2.254e-15** and **2.212e-15** to every digit, so both tests fail by **6.99×** and **5.08×**. **This is prompt 04 §11's first stop condition: the `convergence` block was not written**, so `[01-convergence-block-has-a-separate-generator]` stays open and `QCD_BREAK_POINT_ALIGNMENT_TOL` stays at 1.5e-04 when it is measured to go to **1.421085e-14**. One of the two modules is outside prompt 04's D5 carve-out. **Unassigned — the user decides** between raising both `QCD_FLOOR_FACTOR`s past 7 and giving the tests a bound that is not the reference's floor; either lands in the same commit as the block. |
| `[00-three-production-grid-reproductions]` | tolerance-convergence | **Four** constructions in the test tree were each called "the production source grid" and are different grids. **Narrowed again by prompt 01, 2026-09-16**: the generations are now named and bit-identically hoisted into `wkb_reference.source_grid(SOURCE_GRID_V0/_V1/_V2, …)`, with no default, and the four sites repointed. What is left is the bare `production_source_grid` name, which **28 call sites in 20 files** outside that prompt's scope still import; all are v0 and correct, none says so. |
| `[01-density-criterion-imposed-outside-the-wkb-region]` | tolerance-convergence | `source_grid_spacing_profile` imposes the fourth-derivative spline criterion over `residual_node_range`'s band, which reaches **1.5–2.1 e-folds outside the horizon** — ~5 e-folds beyond the phase spline it protects. 69% of the samples v2 adds on QCD lie above horizon crossing for the smallest production $k$. **Narrowed 2026-09-17** — prompt 02a made the grid buildable without touching the band. **Correction, 2026-09-17: prompt 02a's guarded-node census was briefly recorded here as “the direct measure of the overreach”; it is not, and that claim is withdrawn.** All 53 guards are one node, interior to the band and sub-horizon in 42 of the 53 cases, at a declared break — `[02a-stencil-reaches-across-a-declared-break-before-the-mask]`. **This issue is still without a measurement**; the 69% figure above remains its only evidence. Unjustified rather than wrong; the decision remains **T7**.. **MEASURED by prompt 04, 2026-09-17, for the first time** (`ORDER-AUDIT.md` §8), and it is larger in e-folds and much smaller in consequence than the body claims: in the `Gk` sector **4 %–49 %** of the band's nodes are super-horizon with the band top **13.0 e-folds** outside on Radiation and LambdaCDM and 3.66 on QCD; in the `Tk` sector it is **zero at every wavenumber on every model**, $\omega_T^2 > 0$ already requiring the mode to be inside the sound horizon. Rebuilding the grid with `main.source_grid_spacing_profile` executed unmodified over a horizon-limited band costs **QCD 114 of 2,034 samples and the other two cosmologies none**. **Still unassigned**; the decision now has a cost to set against it. |
| `[02a-stencil-reaches-across-a-declared-break-before-the-mask]` | tolerance-convergence | `source_grid_spacing_profile` evaluates its five-point stencil (reach **2.0e-03** in $u$) over every band node **before** applying the declared-crossing mask (**6.0e-03**), so a node the criterion has already decided to discard is still evaluated and its arms land across the break, where $H$ steps and $\omega^2$ goes non-positive. This was a hard crash until `effb70b`. Measured: all **53** guarded cases on QCD at its own anchor are the **same** node, **1.000e-03** in $u$ from the third declared break, interior to the band and **sub-horizon in 42 of 53**. Because every guarded node is inside the mask, prompt 02a's guard changes **no `usable` outcome** and can alter no profile. But **the published grids' zero is lattice alignment, not construction** — LambdaCDM's anchor straddles the break at 7.582e-03 — so a change to `z_init` or `samples_per_log10z` could start guarding. Reordering the mask would have prevented all 53; it is unmeasured against the published digests. |
| `[02a-grid-digest-not-reproducible]` | tolerance-convergence | The grid tag digests the **exact bits** of the grid's values, but `z_init` is a root-solve output: Brent stops at `xtol + rtol*|u|`, so with `rtol = 1e-8` at `u ≈ 37.6` the anchor is pinned only to **3.8e-7** relative — *coarser* than the `1e-7` at which the redshift table matches a row, and not the `xtol = 1e-10` the record quotes. Across two anchors 3.6e-13 apart all 1778 samples differ bitwise and none exceeds 1e-7, so every row is re-matched and reused while the tag turns over: silent total cache invalidation, ~65,000 `Gk` objects per model. Bit-stable within one datastore lineage. **Part (i) delivered by prompt 03a, 2026-09-17**: both anchor figures confirmed as a swept target (3.62e-13 / 2.50e-09 at $k=3\times10^8$, offset −5), `xtol` measured to bind at **0 of 150** (k, offset) pairs, and the **achieved** worst displacement 7.86e-08 against a **guaranteed** 3.8e-07 — 03a recommends `rtol = 1e-9`, the loosest whose *bound* clears 1e-7. Two cautions for **prompt 05**, which owns part (ii): with `xtol = 1e-10` the pair cannot pin the anchor better than 1e-10 relative however far `rtol` goes, and the "digest the determining data" alternative fails on QCD (`[03a-qcd-v2-grid-sample-count-is-not-reproducible]`). |
| `[02-oneloopintegral-is-a-ninth-keyed-object-type]` | tolerance-convergence | **README §2 (a) counts eight keyed object types; there are nine.** `OneLoopIntegral` carries `atol_serial`/`rtol_serial` as indexed non-nullable foreign keys into `tolerance` and filters on both, is registered, sharded and budgeted — and `main.py` never builds one, its `compute()` being a label-replacing stub with an inverted guard. Object count 0. Prompt 02 §8's stop condition: **unassigned, the user decides** whether prompt 05 decouples it, before any row exists to invalidate, or whether decoupling a target that computes nothing is premature. |
| `[02-wavenumber-exit-time-tolerance-is-an-inequality-key]` | tolerance-convergence | README §2 (g)'s "a new accuracy parameter makes every existing row unreachable" is true of eight of the nine. `wavenumber_exit_time`'s lookup filters `stored.log10_tol − requested <= DEFAULT_FLOAT_PRECISION` and orders descending, so it takes the **loosest row at least as tight as the request**: tightening misses, **loosening silently reuses a tighter row** and the object then reports the stored pair, not the requested one. Prompt 03a must sweep it through `_solve_horizon_exit` and never through the datastore. |
| `[02-shared-atol-doubles-as-a-float-comparison-epsilon]` | tolerance-convergence | `DEFAULT_ABS_TOLERANCE` is also a bare `fabs(a − b) <` epsilon at seven sites with no connection to the $G_k$ ODE (`GkSource.py:96`, `:104`, `:275`; `numeric_with_phase_cut.py:618`, `:737`, `:790`; `WKBtools.py:83`). Nothing is wrong at 1e-10; the hazard is that prompt 05 retunes or splits the constant and all seven move with it, in modules outside that prompt's file list. |
| `[02-extract-tkwkb-queries-tk-numeric-under-the-shared-atol]` | tolerance-convergence | `extract_TkWKB_data.py:433-445` queries `TkNumericIntegration` with `atol = DEFAULT_ABS_TOLERANCE` while `main.py` writes it under `DEFAULT_TK_NUMERIC_ABS_TOLERANCE`, and that target's lookup filters `atol_serial ==` — so the query cannot match a production row. None of the six `extract_*.py` readers imports the split constant. Live since `GkTk-remedial` prompt 12; prompt 05 revisits all six readers anyway. |
| `[03a-tk-numeric-excursion-is-sporadic-in-rtol]` | tolerance-convergence | `TkNumericIntegration`'s maximum envelope-relative error over the production grid is **not monotone in `rtol`**. Measured at 50 $k$ × 3 models × nine settings, v2 grid at each cosmology's own anchor, `BREAK_POINT_ALL`: worst over three models runs 3.36e-04 (`1e-8`), 6.60e-04 (`3e-9`), 9.08e-04 (`1e-9`), 5.76e-04 (`3e-10`), 1.31e-04 (`1e-10`), **3.88e-08** (`3e-11`). From `1e-9` down, **exactly one** of the 150 runs is above 3e-6 at each setting and it is a **different** run each time. It is prompt 17 §7 (b)'s step-selection accident, and `atol` selects which wavenumber draws it (median ≤2.1×, maximum up to 205×). So README §6.1's `rtol = 3e-11` is the loosest that clears *in this sweep*, not a bound. **The user accepted `3e-11` on 2026-09-17 with this caveat standing**, so it ships as a measured setting and not as a bound, and prompt 05's provenance note must say which. **Unassigned.** |
| `[03a-qcd-v2-grid-sample-count-is-not-reproducible]` | tolerance-convergence | Rebuilding the version-2 source grid at a perturbed anchor changes the **number of samples** on `QCD_Cosmology`: 2034 → 2032 / 2016 / 2022 / 2013 / 2029 / 2018 / 2025 at relative shifts of 1e-14 … 1e-6, while LambdaCDM stays at 1778 and Radiation at 2306 throughout. 1e-14 is six orders below the displacement the production anchor solve achieves. The base lattice cannot move at that shift, so what moves is the density criterion's **integer subdivision vector** — which removes one of the two options `[02a-grid-digest-not-reproducible]` offers prompt 05 (digest the determining data, "whose ties are a measured 6e-3 clear"), and means a QCD grid rebuilt elsewhere may hold a different *number* of redshift rows, not merely different bits. **Unassigned — candidate for prompt 05.** |
| `[03a-scipy-rtol-floor-is-the-wrong-floor-for-a-root-solve]` | tolerance-convergence | `convergence_reference.SCIPY_RTOL_FLOOR = 100 eps` is `solve_ivp`'s clamp, and `TolerancePair.rtol_step_is_effective` applies it to **any** pair — including one driving `_solve_horizon_exit`, whose floor is `brentq`'s `4 eps = 8.88e-16`. So a root-solve `rtol = 1e-15` is reported as "silently ignored" when it is not; the measured displacement there is 7.11e-15, one ulp of $u$, which proves the step was applied. Cosmetic today; the hazard is a later prompt stopping its root-solve axis a decade early. The floor belongs to the method, not to the pair. **Unassigned, additive fix.** |
| `[03-outermost-z-source-is-not-the-least-favourable]` | tolerance-convergence | `gk_geometry` takes one source redshift per wavenumber, the outermost, "the longest and therefore the least favourable run", and prompt 03 §2.2 makes that the premise on which fifty runs per model **bound** a 29,000–58,000-object sector. Measured at 3 $k$ × 3 models × 7 source redshifts against converged references: **it is not the worst at any of the nine probes** (×1.01 to ×1.45), because the maximum is *flat* in $z_{\rm source}$ with no trend — what rises monotonically is the median, and what falls is the evaluation count. So the sweep characterises the sector to ~1.5× and does not bound it; no figure in `GK-NUMERIC-SWEEP.md` may be quoted as a maximum over the $(k, z_{\rm source})$ plane. Nothing turns on 1.5× against a 700× floor. **Prompt 03 §9's stop: the user decides** whether a genuine bound is wanted or whether the wording should say "representative". |
| `[00-gk-numeric-never-swept-and-carries-the-cost]` | tolerance-convergence | "The error is set by `rtol`" is one clean measurement in the 50-object sector and one `(atol, rtol)` diagonal in the 65,000-object one. $G_k$ numeric has never been swept in either axis, and review §10.1 puts the consumer spline that reads it two orders above its solver error — so the honest answer may be "tighten nothing". Assigned to prompt 03. |
| `[12-tk-numeric-atol-largest-k-excursion]` | GkTk-remedial → tolerance-convergence | Prompt 12's `atol=1e-13` left excursions above README §6's 3e-6 of the envelope that prompt 17 measured across the production grid: 3 / 13 / 8 of 50 wavenumbers on Radiation / LambdaCDM / QCD, worst 8.64e-4. **Re-taken by prompt 03a, 2026-09-17** on the version-2 grid under `BREAK_POINT_ALL`: **1 / 9 / 4**, worst **3.36e-04** — most of the improvement from the grid, not the policy — and `rtol = 3e-11` takes it to 0 / 150 and 3.88e-08. **The user settled the constant 2026-09-12: `1e-13` stays** — `atol` is not the lever. What remains is the `rtol` retuning. **Assigned (2026-09-12): `prompts/tolerance-convergence`**; its cost figures corrected at the 2026-09-16 rebase, and its sweep re-taken on the v2 grid. |
| `[01-convergence-block-has-a-separate-generator]` | qcd-background-audit → tolerance-convergence | `wkb_reference_data.json`'s `convergence` block records $N_\tau = N_{c_s\tau} = N_F = N_\rho = 4$, was generated 2026-09-10, and names `"branch+knots"` as its winning scheme — a knot set `qcd-background-audit` prompt 07 removed. One tolerance is owed on its account (`QCD_BREAK_POINT_ALIGNMENT_TOL = 1.5e-04`). Declined on scope by prompts 08 and 09 of that campaign. **Assigned (2026-09-16): `prompts/tolerance-convergence` prompt 04**, the first prompt anywhere whose charter is the orders themselves. **Prompt 04 ran the generator on 2026-09-17 and did not write the block**: all four orders come back **unchanged at 4**, `recommended_scheme` becomes `branch` (the knots are measured to buy nothing, confirming prompt 07 at the quadrature level), and the tolerance would go to **1.421085e-14** — ten orders tighter than the 1.4e-05 hoped for, so the whole of the present excess is the block's age. What blocks the write is `[04-convergence-floor-used-as-a-test-threshold]`. The issue has moved from *blocked on scope* to *blocked on a decision*. |
| `[20-wkb-gauss-orders-not-in-lookup-key]` | GkTk-remedial → tolerance-convergence | `TAU_GAUSS_ORDER`, `CS_TAU_GAUSS_ORDER`, `FRICTION_F_GAUSS_ORDER`, `RHO_GAUSS_ORDER` (all 4) and `RESIDUAL_WKB_REGION_MARGIN = 0.5` are configuration axes in no `BackgroundModel`, `GkWKBIntegration` or `TkWKBIntegration` lookup key, while the `atol`/`rtol` columns that *are* in the key describe nothing. **Assigned (2026-09-16): `prompts/tolerance-convergence` prompts 04 and 05** — putting the order in the key instead of the tolerance is the campaign's stated target (README §7 D3). **Prompt 04 reported 2026-09-17** (**T7**): all four orders `unchanged` at 4 and the margin `unchanged` under README §6.1 rule 6, both measured for the first time; **D3's recommendation is `replace with the orders`** — three integer columns on `BackgroundModel`, one on each WKB target, `drop` for `GkSource`, which integrates nothing. Prompt 05 implements once the user settles D3. |

---

### 1.6 The phase-representation campaign

[`prompts/phase-representation/`](../prompts/phase-representation/README.md) (2026-09-13; two
prompts, **01 executed, 02 blocked; CLOSED at 1 / 2 on 2026-09-13**). Both close a defect that `prompts/GkTk-remedial` prompt 13
**measured and was forbidden to fix**, a verification prompt being barred from touching production
code. Both are about how a WKB phase is represented and reconstructed for a consumer: one in the
`(cycle count, remainder)` pair the producers store, one in the spline the consumers build. The
measurements are `docs/gktk-remedial-verification.md` §3.5, §3.6 and §3.7 and are re-runnable from
`docs/gktk-remedial/verify_production_path.py`. Prompt 01 closed
`[13-wkb-mod-2pi-cycle-count-inconsistent]` on 2026-09-13 — the cycle count now comes from the
exact `fmod` remainder and the production geometry is 0 inconsistent of 77,975. **Prompt 02
stopped** on its own §2 item 2 without changing production code: the remedy the remaining issue
names does not construct, and the issue's own attribution did not survive measurement. The
README §7 D2 decision was then taken on the evidence of
[`qcd-background-audit-2026-09.md`](qcd-background-audit-2026-09.md) — 404 of the 407
`BREAK_POINT_ALL` points are knots of the `T(z)` spline, so the remedy is upstream — and the
campaign was **closed**, its remaining issue passing to the `qcd-background-audit` campaign (§1.7). Close-out:
`docs/gktk-remedial-verification.md` §8.

| Issue | Board | Hook |
|---|---|---|
| `[02-consumer-phi-below-the-storage-granularity]` | phase-representation | $\varphi$ is recovered as a difference of two numbers of size $k\tau$, so on QCD $G_k$ its whole range is 6.0 ulp of the stored phase at $k=10^7$ and **2.0 ulp (3 distinct values over 1,377 samples)** at $3\times10^8$. Differentiating that staircase makes `theta_deriv` **3× and 10× worse than omitting $\varphi$ altogether**, and is what §3.6's two failing rows actually are. `[00-consumer-anchoring-floor]` seen in the derivative. |

---

### 1.7 The QCD background campaign

[`prompts/qcd-background-audit/`](../prompts/qcd-background-audit/README.md) (2026-09-13; twelve
prompts in four workstreams, closed at 12 / 12 on 2026-09-15 and **reopened the same day as
workstream E**, four more prompts, **all four of which have now landed: 16 / 16** — the ungated
chain 01–09 completed 2026-09-14, workstream D's prompts 10, 11 and 12 ran on 2026-09-15, prompt 13
closed the defect prompt 12 was forbidden to act on, prompt 14 gave a run a name and the grid's
construction a version, prompt 15 took prompt 12's density recommendation and prompt 16 retired the
samples-per-decade tag it left false). It
implements [`qcd-background-audit-2026-09.md`](qcd-background-audit-2026-09.md), which measured that
`QCD_Cosmology`'s temperature is a cubic spline over 500 points solved to `rtol=1e-4`, built as $T$
against $\log(1+z)$ and run across three points at which $T(z)$ genuinely **jumps** — so the
background carries a systematic **3.461e-08** relative error in conformal time, worth of order
**1.4e5 rad** at $k=3\times10^8$/Mpc against a 9.15e-04 rad floor. That error is **common mode
between every producer and every consumer**, which is why no test in the tree can see it and why
`docs/gktk-remedial-verification.md` §3.5 reads 1.00 ulp while §5 of the audit is true. The
campaign replaces the representation with a segmented entropy-factor spline (prompts 04–06),
collapses `BREAK_POINT_ALL` to 3 (07–08) — it was 407 when the audit was written and prompt 06's
node count took it to 2,414, **all but three of them knots of that auxiliary interpolant rather
than cosmology** — and adds the background-against-background test that would
have caught it (01). It was a **precondition for `prompts/tolerance-convergence`** (§1.5), whose
QCD half would otherwise have been measured against a background about to move; the campaign closed
and merged (`acd5b8e`) and that campaign was rebased on its result on 2026-09-16.

**Prompt 01 landed 2026-09-14** and that test now exists:
`CosmologyModels/tests/test_T_z_representation.py`, scored against
`CosmologyModels/tests/T_z_reference.py`, seven cases, no production file touched. It reproduced
the audit's 3.4605e-08 in conformal time in 0.013 s, and pins the branch joins below.

**Prompts 02–06 landed 2026-09-14, and T1 is closed.** The datastore can now see the
representation (`T_Z_REPRESENTATION_VERSION`, at **4**), the node solve is exact, the entropy
factor is what is tabulated, and it is tabulated **one spline per branch** with the segment edges
*bisected* onto the jumps at 3,000 nodes of order 5. $T(z)$ on the audit's probe set reads
**6.807e-11 / 3.237e-15 / 1.765e-16** against the shipped 7.177e-04 / 1.323e-05 / 1.890e-07, $H(z)$
on the production grid **1.690e-10 / 6.276e-15 / 2.804e-16**, and the conformal-time guard reads
**0.0** — the shipped background's $\int\mathrm{d}z/H$ is bit-identical to the exact background's
at all 17 digits, so the 1.4e5 rad at $k=3\times10^8$/Mpc is 0.000e+00 rad.

**Prompt 07 landed 2026-09-14, and G1 is closed.** `integration_break_points` declares the
equation of state's temperature crossings and nothing else: `BREAK_POINT_ALL` on the production
source grid is **3** (from 407 when the audit was written and 2,414 after prompt 06's node count)
and `BREAK_POINT_DISCONTINUITY` **2**, none of them a knot of anything, all three bit-identical to
prompt 06's bisected segment edges. The knots were dropped **on measurement, not on principle**: at
order 5 the first discontinuous derivative of the interpolant is the *fifth*, three levels below
the deepest one anything in the tree builds, and the observable residual across a knot is 6.3e-12
in $H$ against the **2.1e-04** the old 500-node cubic lattice carried. The QCD `BackgroundModel`
build falls from 16,580 integrand evaluations to **6,936** — 0.17 % above LambdaCDM's break-free
6,924 — with the references unmoved (they are adaptive quadratures and never saw the panel
structure), `cs_tau` and `friction_F` scoring against them unchanged to every digit printed and
`tau` moving 2.104e-15 → 2.254e-15, at 12 % of its 1.879e-14 floor.
`T_Z_REPRESENTATION_VERSION` is **5**.

**Prompt 08 landed 2026-09-14 and found "state (a)": the $T_k$ sector no longer needs
`BREAK_POINT_ALL`.** `GkTk-remedial` prompt 19 chose it on measurement — 3 of 50 QCD wavenumbers
above the 3.4e-08 convergence criterion with the jumps alone, worst 1.97e-07 — against knots
carrying the $10^{-4}$-level defect prompts 04–07 have since removed. Re-taken across the full
matrix (50 wavenumbers × 2 sectors × 2 policies × 3 models, 744 s,
[`per_sector_policy_remeasure.py`](qcd-background-audit/per_sector_policy_remeasure.py) →
[`PER-SECTOR-POLICY.md`](qcd-background-audit/PER-SECTOR-POLICY.md)): QCD $T_k$ converges at **all
50** under *either* policy (**7.08e-09** under `BREAK_POINT_ALL`, **8.85e-09** under
`BREAK_POINT_DISCONTINUITY`, zero offenders in both), $G_k$ improved to **3.67e-09** under both,
and the two models that declare nothing are **bit-identical between the policies** with prompt 19's
grid totals reproduced as exact integers. The wider policy now costs **+0.99 %** of the $T_k$
evaluations rather than +220 % — a QCD object is 8,986 where prompt 19 measured 31,521.
**Neither `BREAK_POINT_KIND` was changed**: they are in a datastore lookup key and the two policies
still differ by up to 2.86e-04 of the envelope on QCD, so that is README §7 **D5**, reported to the
user and not decided, with the recommendation to leave both alone. What *is* still load-bearing is
the **jumps**: with the declaration suppressed altogether, 19 of 50 QCD $T_k$ wavenumbers go above
the criterion, worst 9.61e-06.

**Prompt 09 landed 2026-09-14 and closed the ungated chain.** Verification only; no production file
in the diff and `T_Z_REPRESENTATION_VERSION` unchanged at **5**. The record is
[`qcd-background-verification.md`](qcd-background-verification.md), with a dated **§9** appended to
[`gktk-remedial-verification.md`](gktk-remedial-verification.md) (121 insertions, 0 deletions,
nothing at or above §8 touched). `verify_production_path.py` was re-run **unedited** at the base and
at the close: **every LambdaCDM number is bit-identical** — all six of its §3.5 rows, all six of its
§3.6 `theta_deriv` column groups and all six of its §3.7 rows, with §3.7 unchanged (0 inconsistent)
for both models — and `LambdaCDM(Planck2018)` and
`RadiationModel` are **byte-identical** over 7,007 `float.hex` lines across the whole nine-commit
span. The audit script's §1 and §2 — the equation of state — are character-for-character identical.
On QCD every producer number improved: all six $\theta_G$/$\theta_T$ rows fell towards their
representation floor — all six were above the script's printed `eps*|theta|_max` floor at the base,
by 1.07× to 5.94×, and three are now below it with the other three within 11 % of it — while `tau`
goes 2.104e-14 → **2.254e-15** and `cs_tau` 2.108e-14 → **2.212e-15**, and per-object build costs
$G_k$ 8,380 → **6,892** and $T_k$ 12,896 →
**11,532** integrand evaluations. **Two consumer rows went the other way**, by 4.25× and 4.39×
(§3.5's QCD $k=10^5$ pair, 8.1062e-06 and 1.3982e-05 rad), and the cause is measured rather than
guessed: the old `T(z)` spline's knot lattice smeared the equation of state's genuine step over ~4
production grid intervals, leaving **25.55 %** of it inside the crossing's own interval, where the
corrected background puts **99.84 %** of a step 2.78× taller there. That is
`[13-consumer-spline-crosses-eos-break-points]` seen undiluted — the campaign's one un-discharged
consequence, recorded as a miss of README §6.4's "nothing got worse" and left to **prompt 10**.
The counterpart is §3.6's QCD $T_k$ row at $k=10^7$, **1,413× better** and now carrying LambdaCDM's
own end-condition signature. Suites 11 → **30**, 339 → **359**, 148 → **148** (full set).

**Prompt 10 landed 2026-09-15 and re-attributed that consequence.** Verification and measurement
only; **no production file in the diff** and `T_Z_REPRESENTATION_VERSION` unchanged at **5**. It
re-took `prompts/phase-representation` prompt 02's measurement on the corrected background, scoring
**nine knot schemes** over all twelve (model, sector, $k$) rows with
[`consumer_knot_scheme_scan.py`](qcd-background-audit/consumer_knot_scheme_scan.py) (one command,
~120 s, no Ray, no datastore), and **none of them helps**: the repeated multiplicity-`spline_order`
($C^0$) knot vector the issue's own next step named is **2.09× and 2.10× worse**, per-segment
splines **5.00× and 5.06× worse**, and the best of four controls is 1.21× better against the 4.25×
and 4.39× that would have to be recovered. Prompt 02's kink fit, re-taken as the `GkTk-remedial`
board required, is still **window-dependent** — $[\varphi']$ moves two orders and changes sign
between 1-, 2- and 3-interval windows — which is the signature of smooth-but-*unresolved* data and
the reason a $C^0$ knot has no corner to turn. **What fixes it is samples**: refining the ±5 grid
intervals around the crossing by 2× (10 extra samples in 1,016) takes the two rows to **1.64 ulp**
and **74.60 ulp**, both inside the 1e-06 rad consumer target, where refining the crossing's own
interval alone stalls at 1.96×. The `theta_deriv` residue is split and attributed: the break points'
share is **35.8 % / 35.5 %** and only at $k=10^5$, while
`[02-consumer-phi-below-the-storage-granularity]`'s is **100 %** of the two QCD $G_k$ rows that miss
$10^{-6}$ (recovered $\varphi$ spans 6.0 and 2.0 ulp there; removing $\varphi'$ altogether
*improves* them 1.53× and 15.9×). `PrimitivePhase` therefore keeps `make_interp_spline`'s default
knots, on measurement, and the defect moves to the source grid. Record:
[`qcd-background-verification.md`](qcd-background-verification.md) §8. Suites 30 / 339 → **361** /
143 (fast set).

**Prompt 11 landed 2026-09-15 and closed it.** The production source grid now carries what the
cosmology declares: a pair straddling each of `QCD_Cosmology`'s three equation-of-state crossings at
a quarter of a grid interval, the eleven intervals around each refined by two, and the two equality
redshifts -- **41 extra samples in 1,732 (2.37 %)**, taking the two rows to **1.61 ulp**
(3.8296e-07 rad, $G_k$) and **65.78 ulp** (4.9012e-07 rad, $T_k$), both inside the 1e-06 rad
consumer target and both better than the campaign base. **The two straddling samples are not the
remedy on their own**: measured alone they buy 1.96× and 1.98× and leave both rows outside the
target, which is prompt 10's "the feature is three to five grid intervals wide" confirmed on the
grid actually built. The standoff was scored from 1/2 of an interval down to 1e-4 of one and is a
*fraction of the grid spacing*, not an absolute number -- below ~1/32 the pair saturates 1.5×--1.7×
**worse** than no pair at all, because the slope it implies drowns in
`[02-consumer-phi-below-the-storage-granularity]`'s storage granularity, which is why
`BREAK_POINT_STANDOFF = 1e-12` must not be borrowed for a sample location. LambdaCDM's grid is
**bit-identical, element for element**, and `main.py` gates the whole path on the cosmology
declaring something so that it stays that way. `winnow` now retains a protected set, and the grid
tags carry a digest of the grid's own values, which closes the equal-length collision and
**invalidates every stored object of eight types** -- the bill is quantified in log 11 §5 and
[`qcd-background-verification.md`](qcd-background-verification.md) §9.6. Suites 30 / 361 → **380** /
143 (fast set).

**Prompt 12 landed 2026-09-15 and closed the campaign, by measuring and recommending rather than
by changing anything.** No production file is in its diff, `T_Z_REPRESENTATION_VERSION` is **5**
before and after, and no grid was changed. The uniform `source_samples_per_log10z = 100` is wrong
in **both** directions: scored against the phase residual itself — not against $\varphi$ recovered
from a stored $\theta$, which is floor-limited by
`[02-consumer-phi-below-the-storage-granularity]` — the consumer's cubic misses the storage floor
by **7.86×** and **7.84×** in the top decade of the $T_k$ band at $k=10^5$ on LambdaCDM and QCD,
and has up to **2.1e+19** of headroom at the bottom of the range, the spacing being constant to
four digits across fourteen decades. The criterion that fixes it,
$h^4|\varphi''''|/384 \le \varepsilon$ with $\varphi' = -(1+z)C/(\omega+\omega_0)$, is
**computable before the grid exists** from $H$, $c_s^2$ and $k$ (0.01–0.27 s against
`compute_background`'s 0.599 s) and predicts the realised error **to ±2 %** over 500-odd intervals
in the $T_k$ sector on both models at all three wavenumbers. One universal envelope grid gives QCD
1,773 → **1,761** samples with every row at 0.14× its target, or **1,015** (1.75× fewer) at twice
today's spacing; LambdaCDM 1,732 → **1,634** or **842** (2.06× fewer). A second candidate —
equidistributing $\varphi$ itself — is **refuted** at 114,281 samples. Above $k\approx10^7$ none of
this is visible, the floor growing like $k$ while $\varphi$ falls like $1/k$. Audit §9's $k\tau$
bullet is discharged and the answer is that **no grid size exists**: the median response interval
advances 30 to 47 complete cycles at the smallest production $k$, and four samples per cycle would
need $6\times10^6$ to $1.8\times10^{10}$ times the response samples the grid carries. The two
issues below are prompt 12's; the first **is** the recommendation, recorded so that it outlives the
campaign, and the decision is the user's. Record:
[`grid_density_criterion.py`](qcd-background-audit/grid_density_criterion.py) (one command, 88.4 s,
no Ray, no datastore) → [`qcd-background-verification.md`](qcd-background-verification.md) §10.
Suites 30 / 380 / 143 (fast set), unchanged.

**Prompt 13 landed 2026-09-15 and closed the defect prompt 12 opened.** `BackgroundModel` splined
background quantities at **two** sites — `compute_background`'s order-5 fit of $\log H$ over a
padded, 3× refined copy of the source grid, and `_create_functions`'s cubic through the *stored*
samples — and neither was split at `integration_break_points`, while $H$ genuinely **steps** at two
of the three declared crossings. Both now fit **one spline per branch**, dispatching on
$u=\log(1+z)$ and refusing a branch too narrow for `order + 1` nodes rather than dropping the order
or fitting across the step. `epsilon` against a central difference of the cosmology's own `Hubble`
on the production grid: `T_LO` **2.036e-02 → 1.931e-09**, `T_120_MEV` **1.034e-03 → 4.070e-09**,
both inside the 3.907e-09 that holds away from a crossing, and the `EOS_T_LO` control
**1.609e-09 → 1.609e-09**, the same float. **Neither site is redundant** — the fit alone leaves
1.91e-04 at `T_120_MEV`. `BREAK_POINT_ALL` rather than the two jumps, on measurement:
`d_wPerturbations_dz` at `EOS_T_LO` is **346 % wrong** without the third crossing and 7.8e-04 with
it. The production configuration of the four-way table goes **2.4859e-05 → 5.8437e-08 rad** at QCD
$T_k$, $k=10^5$ (**425×**), the background's grid has stopped mattering (four cells agreeing to four
digits where they spanned 300×), and the maximum has left the crossing.
`T_Z_REPRESENTATION_VERSION` is **6**; the QCD fixture was regenerated (`rho_G` 7.278e-02 largest
relative move) while `tau`, `cs_tau` and `friction_F` did not move by a bit; integrand counts are
exactly unchanged; and every cosmology that declares nothing is **byte-identical**. Suites
30 / 380 → **392** / 148 (full set).

**Prompt 14 landed 2026-09-15 and closed the other thing prompt 11 left behind** —
`[11-background-model-not-keyed-on-the-source-grid]`, deleted from this index. A run is now **named
at write time** (`main.py --run-label`; a `store_tag` carried by and filtered on for everything the
run writes, in `TkProductionTag`'s manner) and **verified at read time**:
`SOURCE_GRID_CONSTRUCTION_VERSION = 1` names the grid's *algorithm* beside prompt 11's content
digest, `sqla_BackgroundModelFactory` gains `source_grid_digest` and `source_grid_construction`
columns and filters on both, and all six `extract_*.py` — whose exclusion the user lifted **for that
prompt only** — select on the run label and **refuse a mixture, naming every generation found**,
rather than picking. A pre-prompt-14 store cannot be computed into (a `RuntimeError` naming the
campaign, "regenerated", "no migration") and can still be **read**, reported as an unknown
generation — the archival requirement, with the measured limit recorded below. No background
number moves:
`T_Z_REPRESENTATION_VERSION` is **6** before and after and the production grid is **byte-identical**
(3,814 `float.hex()` lines, MD5 `d5ecc0aa85f38578d8c57c051d3f11e0`). Suites 30 / 392 → **424** / 148
(full set).

**Opened by prompt 15**, which closed
`[12-source-grid-density-is-uniform-over-a-curvature-that-spans-eight-orders]` — the user took
prompt 12's recommendation and the source grid's base density is now set by the measured curvature
criterion, under the cap `SOURCE_GRID_MAX_SPACING_FACTOR = 1.0` ("never coarser than the uniform
lattice puts there, anywhere"). QCD 1,773 → **1,996**, LambdaCDM 1,732 → **1,778**, the two rows
that missed their storage floor **7.86 → 0.69** and **7.84 → 0.11** ulp in the production
configuration, `SOURCE_GRID_CONSTRUCTION_VERSION` **1 → 2**, `T_Z_REPRESENTATION_VERSION` still 6
and no background value moved. Suites 30 / 424 → **439** / 143 (fast set):

| Issue | Board | Hook |
|---|---|---|
| `[15-the-edge-factor-is-applied-at-the-band-edge-not-at-the-consumers-own-end]` | qcd-background-audit | The criterion uses the cubic's *interior* error constant; prompt 15 measured the realised constant at the outermost intervals of a band at **9.9× / 4.3× / 1.2×** that (the not-a-knot end condition), and tightens the target there with `SOURCE_GRID_SPLINE_EDGE_INTERVALS = 3`, `SOURCE_GRID_SPLINE_EDGE_FACTOR = 10.0`. But a production `PrimitivePhase` spline ends at its object's own anchor, *inside* the band, where the same amplification applies and no tightening does — so such an end can carry up to 10ε where the criterion promises ε. Not measured in either direction. **Next step:** score one QCD $T_k$ object at $k=10^5$ at its own anchor against the residual oracle; log 15 deviation 3 costs both remedies. |
| `[15-the-grid-now-depends-on-the-wavenumber-sample-and-no-tag-says-so]` | qcd-background-audit | The grid was a pure function of $(z_{\rm init}, z_{\rm end}, \texttt{samples\_per\_log10z})$; the criterion's envelope is taken over every wavenumber the run serves, so it now also depends on the wavenumber sample — a hard-coded `NUMBER_SOURCE_K_VALUES = 50` at `main.py:3292`. Caught by the content digest and the construction version (different grids get different tags and `BackgroundModel` refuses the mixture). **Prompt 16 closed the tag half** (`SourceSamplesPerLog10ZTag`, which mislabelled the base density, is retired outright); **the wavenumber-set half is the user's explicit no-action decision** (2026-09-15, quoted on the board) — no `SourceKSampleTag`, bear it in mind at calculation time instead. Correctness unaffected. **Next step:** none. |

**Opened by prompt 14** — the first from a measurement it made in passing, the second in the
user's framing with nothing built towards it:

| Issue | Board | Hook |
|---|---|---|
| `[14-archival-read-stops-at-the-pre-gktk-value-columns]` | qcd-background-audit | Prompt 14's read path makes a **pre-prompt-14** store readable — the missing grid-identity columns are caught, the query re-issued without them, the row reported as an unknown generation. It does **not** make a **pre-`GkTk-remedial` 03/04** store readable: that refusal, on `BackgroundModelValue.tau_lo_Mpc`, is unconditional on both paths. Measured on the only datastore in the tree, which is exactly that old: the prompt-14 fallback fires and finds the row, then the `tau_lo_Mpc` message stops it. The two guards differ in kind — a grid identity is metadata never *recorded*, `tau_lo_Mpc` a value never *computed* — so softening the second means fabricating background values. **Next step:** if the oldest stores must stay archival, it needs a reader that stops at `BackgroundModel` and never asks for the value rows. A design, not a patch; nobody has asked for it. |
| `[14-no-archive-of-grid-construction-algorithms]` | qcd-background-audit | A datastore records *which* construction built its grid, but the **code** of a superseded construction lives only in git history, so a run written under version 1 cannot be re-derived once version 2 has replaced `build_z_sample` — which prompt 15 will do. Harmless today (there is one construction); it bites the first time someone wants to reproduce, rather than merely re-read, an archived run. Prompt 14 built nothing beyond the version integer that would be its key, deliberately: an archive means every retired constructor kept alive and tested forever. **Next step:** none proposed — the decision is the user's, and the first thing to settle is whether a retired construction must keep *running* or only be *readable*. |

**Narrowed by `background-solver-robustness` prompt 07 (2026-09-16):**

| Issue | Board | Hook |
|---|---|---|
| `[03-qcd-inventory-does-not-report-the-representation]` | qcd-background-audit | Originally both `sqla_QCDCosmology_factory.inventory()` and (from prompt 14) `sqla_BackgroundModelFactory.inventory()` omitted their tables' identity columns. **The `QCD_Cosmology` half is resolved** — `background-solver-robustness` prompt 07 added `T_z_representation` to the former, demonstrated against two rows differing only in it. **The `BackgroundModel` half is not**: prompt 07's files-may-touch list did not include `BackgroundModel.py`, and its own stop condition treats a second reporting site with the same gap as a new issue to record, not fix. `sqla_BackgroundModelFactory.inventory()` still says nothing about `source_grid_digest` or `source_grid_construction`. **Next step:** add both columns to its per-bucket report, in whichever prompt next has `Datastore/SQL/ObjectFactories/BackgroundModel.py` in scope. Not assigned. |

**Opened by prompt 13**, which closed `[12-background-derivative-fit-grid-rings-at-a-step]`:

| Issue | Board | Hook |
|---|---|---|
| `[13-segmenting-costs-accuracy-on-a-grid-that-does-not-resolve-the-crossing]` | qcd-background-audit | A segment edge creates two *interior* spline ends that **cannot be padded** — beyond an edge lies the other branch, and the cosmology exposes no continuation past the crossing; at the stored-sample site the node set *is* the grid, so there is nothing to pad with. Accuracy at a crossing is therefore set by how close the nearest samples are, and on a grid that does not resolve one a cut is worse than the smooth fit it replaces: at the `EOS_T_LO` control `epsilon` goes **1.889e-09 → 8.190e-08** on the uniform base grid, **43× worse**, while on prompt 11's production grid it is the same float. Harmless today. **But `[12-source-grid-density-…]`'s cap-2× column is coarser than the base grid everywhere** and its §10.5 table was taken on a tree in which no spline was segmented. **Priced by prompt 15 and it did not bind** (2026-09-15): the cap that shipped only ever refines, so the production grid is a strict superset of the one this was measured on and the three-crossing row is 1.6896e-09 / 1.9824e-09 / 4.0453e-09, identical to the shipped grid's at two crossings and better at the third. **Open, narrowed:** the statement is still true and still unguarded. **Next step:** a test that pins that row, so a later coarsening announces itself. |
| `[13-crossing-neighbourhood-refinement-was-sized-at-k-1e5]` | qcd-background-audit | `SOURCE_GRID_BREAK_HALF_WIDTH = 5` and `SOURCE_GRID_BREAK_REFINEMENT = 2` were fixed by prompt 10's ladder at $k=10^5$, and prompt 10's "the crossing is a $k=10^5$ phenomenon" was scored against $\varphi$ recovered from a **stored** $\theta$, whose granularity at $k=10^7$ is 3.05e-05 rad — larger than the effect. Against prompt 12's residual oracle the production configuration at QCD $T_k$, $k=10^7$ reads **5.6892e-05 rad near a crossing**, **59.7 ulp** of the band's span against 0.008 ulp away from one ($G_k$ is 1.59 ulp, at the floor). It improved **7.9×** in prompt 13 and is in the same regime as the 65.78 ulp prompt 11 shipped and called a success, so not a regression — the statement is that the neighbourhood was sized for one wavenumber. **Prompt 15 did not take it** (2026-09-15): its §2 item 3 requires prompt 11's neighbourhoods to survive unchanged and its criterion masks the neighbourhood of every crossing out, so the density change cannot reach this figure. **Next step:** re-run prompt 10's ±$n$ × $m$ ladder at $k=10^7$ against the residual oracle, and say whether the half-width should depend on the band; it is a separate lever from the base density and sits beside it in `build_z_sample`. |

**Opened by this campaign and owned by no prompt of it.** The chain is closed, so each of these
waits on a prompt that has the right files in scope; the board holds the measurements.

| Issue | Board | Hook |
|---|---|---|
| `[00-eos-branch-joins-do-not-match]` | qcd-background-audit | `QCD_EOS`'s branch joins at $10^{16}$, 0.12 and $10^{-5}$ GeV jump by +1.395e-02, −3.744e-04 and −2.284e-03 in $g_s$, forcing steps in $T(z)$; the join at 0.002 GeV matches to 1.751e-11, and that asymmetry is the evidence the other three are a transcription defect. Origin of the 4.4e-04 jump in $H(z)$ at $z=4.24\times10^7$ that `GkTk-remedial` log 02 measured without attribution. **Upstream data fixture; pinned in a test by prompt 01 (2026-09-14), which confirmed every figure here to the digits quoted; not repaired.** The question for its authors is that campaign's README §7 D6. |
| `[04-unsplit-tk-run-now-meets-the-criterion]` | qcd-background-audit | `test_split_converges_where_unsplit_does_not` asserted that an unsplit $T_k$ numeric run *fails* the 3.4e-08 criterion at $k=4.972\times10^7$/Mpc. False since prompt 05: unsplit drift 1.0213e-06 → **2.2767e-08** (45× better) while the split run barely moved, so most of what the split rescued was the old representation's interpolation noise, not the jump in $H(z)$. First measured evidence that `BREAK_POINT_KIND = BREAK_POINT_ALL` may no longer be load-bearing — README §2 (f), §7 **D5**, and `BREAK_POINT_KIND` is in a datastore lookup key. One wavenumber of fifty; **not decided by prompt 05**. **Prompt 08 took the column across all fifty** (2026-09-14): unsplit, QCD $T_k$ is above the criterion at **19 of 50** wavenumbers, worst **9.61e-06** at $k=4.223\times10^7$ — this entry's own $k$ reads 2.91e-08 and passes, but it is not representative, so splitting at the **jumps** is still load-bearing and only the `ALL`-vs-`DISCONTINUITY` distinction is vestigial (7.08e-09 against 8.85e-09, zero offenders either way). **Narrowed, not closed:** the assertion in the tree is still pinned to a $k$ at which its original statement is false. **Next step:** re-point `test_split_converges_where_unsplit_does_not` at $k=4.223\times10^7$, where unsplit is 9.61e-06 against a split 5.60e-10; out of bounds for prompt 08. |
| `[08-gk-declared-split-buys-nothing-measurably]` | qcd-background-audit | The $G_k$ numeric sector splits at the declared jumps; prompt 08 measured what that buys on QCD — worst reference-convergence drift **3.67e-09** split against **3.52e-09** unsplit over 50 wavenumbers, zero above the criterion either way, 13,343 against 13,320 evaluations per object. Within the noise of the measure it buys nothing, unlike $T_k$ where suppressing the split puts 19 of 50 above the criterion. No action proposed: it costs 0.17 %, it is the mechanism $T_k$ needs, and `BREAK_POINT_KIND` is in a lookup key. Recorded so a later reader weighing README §7 D5 need not re-derive it. |

---

### 1.8 The background solver robustness campaign

Planned as [`prompts/background-solver-robustness/`](../prompts/background-solver-robustness/README.md)
(2026-09-16 at `f023eb8`; grown to **nine prompts in five workstreams** when the user decided
README §7 D2 — **and closed on 2026-09-16 with all seven authorised prompts landed**, 01–06 and 09;
workstream D, prompts 07 and 08, was **authorised by the user on 2026-09-16, after that
close-out, and has since landed in full — all nine prompts are now complete**). It implements
[`AUDIT.md`](../prompts/background-solver-robustness/AUDIT.md), which measured that
`LambdaCDM_GenericEOS._find_rho_equality` is an **unbracketed secant** at `xtol=1e-6, rtol=1e-4` —
two orders looser than the file's other two solves, which `prompts/qcd-background-audit` tightened —
that returns the right answer to **−4.00e-16** relative in one to three evaluations **because its
caller hands it the closed-form root**, not because its tolerances are adequate; and whose failure
mode (`ValueError` at −30 % of the guess, a `T(z)` bounds error from negative $z$ at −50 %) escapes
its own `converged` guard entirely. **The campaign changes no computed quantity**: both results are
printed with `:.4g` and discarded. What it buys is a solve correct by construction, a reachable
guard, and provenance `docs/TOLERANCE-PROVENANCE.md` can state — which is why
[`AUDIT.md`](../prompts/background-solver-robustness/AUDIT.md) §7 wants it to land **before**
`prompts/tolerance-convergence` prompt 02 runs, and that campaign has not started. **It did land
first**, and that campaign's 2026-09-16 re-anchor records the three solves as arriving already
settled, with provenance prompt 02 lifts rather than re-derives.

**Prompt 02 shipped that fix on 2026-09-16** and closed `[00-equality-solve-is-unbracketed-and-loose]`
on that board's §4: a $\sqrt2$ geometric bracket in $1+z$, clamped to the $T(z)$ representation's
own bounds at both ends and at the guess, then Brent at `xtol=1e-300, rtol=8.9e-16` — Brent's own
$4\varepsilon$ floor, and **not** the campaign's original `rtol=1e-14`, which was measured to stop
7 ulp from the independent reference because the residual is a cancellation between two densities
of order $10^{112}$ whose sign change spans several floats (the user amended that campaign's
README §7 **D1** on the measurement). The two matter–radiation roots moved **+3 and +1 ulp, onto**
the reference; the two matter–$\Lambda$ roots are bit-identical; both printed banner lines are
unchanged; `T_Z_REPRESENTATION_VERSION` stays 6.

[`RECONCILIATION.md`](../prompts/background-solver-robustness/RECONCILIATION.md) scores the audit
against the tree at `f023eb8`: **every figure reproduces to the digit**, and one conclusion does
not. Audit §2.1's *"the blast radius of this solve is two banner lines"* is true of the **solve**
and false of the **quantity** — `main.py:553-555` (`:549-551` before prompt 03) recomputes both
equality redshifts from the same closed forms and forces them into the production source grid,
whose content digest is a `BackgroundModel` lookup-key column, so on `QCD_Cosmology` they are
production sample locations inside a datastore identity.

**Prompt 03 wrote that down on 2026-09-16**, changing no production behaviour: one docstring
sentence in `main.py` and one new test. It measured the **three closed-form sites identical bit for
bit** on `QCD_Cosmology`, the pure-radiation stand-in and `LambdaCDM(Planck2018)`, on both pairs,
despite `LambdaCDM.py` reaching `math.pow` where the other two get the builtin; the production
source-grid digests are `a2c32f67` (QCD, 1,996 samples) and `60a3205a` (LambdaCDM, 1,778),
**identical at `7fdc49b`, `921f41c` and prompt 03's commit**; and README §7 **D2** option (iii) is
priced at a single measured number — substituting the solve's answer for the closed form in
`feature_z` takes the QCD digest to `4849552b` on a 7-ulp move in one sample of 1,996, invalidating
eight stored object types. The campaign recommended **(i)**; **the user decided (iii)** on
2026-09-16, since regeneration is not a cost in the build phase and the closed form at that site is
only accidentally right. That adds **prompt 09 and workstream E** to the campaign — nine prompts in
five workstreams.

**Prompt 09 shipped it on 2026-09-16 and closed the issue.** `BaseCosmology` now declares
`z_matter_radiation_equality` and `z_matter_lambda_equality`; `LambdaCDM` answers with the closed
form, which is exact for a model with no equation of state; `LambdaCDM_GenericEOS` answers with the
bracketed solve its constructor already ran; `main.py` asks and computes nothing, with **no
fallback**. The `QCD_Cosmology` production source-grid digest is **`4849552b`** at 1,996 samples,
moved from `a2c32f67` on **exactly one** sample (index 1540, +7 ulp) — the value prompt 03
predicted — and `LambdaCDM(Planck2018)` is `60a3205a` at 1,778, unmoved. No regeneration follows
(the user, 2026-09-16: this is the build phase).

**Prompt 06 closed the campaign on 2026-09-16.** All seven prompts authorised at that point
landed (01–06 and 09). **The user then opened README §7 D3's gate on 2026-09-16**, so workstream D
(07, 08) was authorised, and both prompts have since landed.
[`PROVENANCE.md`](../prompts/background-solver-robustness/PROVENANCE.md) settles all three of the
file's `root_scalar` sites in the shape `docs/TOLERANCE-PROVENANCE.md` will want, and
`prompts/tolerance-convergence`'s board and README §3.2 now point at it instead of carrying an
unowned bullet. **`AUDIT.md` §5's "fixing this changes no computed quantity in the pipeline" held
for workstreams A–C and was deliberately superseded by workstream E**, which the user added after
the audit was written: nothing moved through prompt 05 except the two matter–radiation equality
redshifts by +3 and +1 ulp *onto* the independent reference (both printed with `:.4g`, both banner
lines character-identical), and prompt 09 then moved the `QCD_Cosmology` source-grid digest on one
sample of 1,996 on purpose.

**Prompt 07 landed on 2026-09-16**, reporting `T_z_representation` in the QCD cosmology inventory
(see §1.7 above) and moving `ComputeTargets` to 452. **Prompt 08 landed on 2026-09-16**, the last
prompt on this board: it re-measured `test_wPerturbations.py`'s agreement figures (unchanged from
prompt 01), rewrote the module comment to describe the segmented entropy-factor representation
that replaced the 500-point $T(z)$ spline it used to describe, and tightened `AGREEMENT_RTOL` from
`1.0e-8` to `1.0e-14`. **Nothing on this board remains open or gated.**
`T_Z_REPRESENTATION_VERSION` is 6 at every commit; final suites `CosmologyModels` **39**,
`ComputeTargets` **452**, both OK.

**Opened by prompt 06 (2026-09-16):**

| Issue | Board | Hook |
|---|---|---|
| `[06-node-solve-comment-quotes-a-superseded-node-count]` | background-solver-robustness | `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:627`, inside the comment that justifies `_solve_T_z`'s `xtol=1e-300, rtol=1e-14`, says the solve is "paid once per node at build time (~500 nodes, 8.3 us each -- a few ms total)". `DEFAULT_T_Z_SPLINE_SAMPLES` has been **3,000** since `qcd-background-audit` prompt 06, and the measured count is **3,176** calls per `QCD_Cosmology(max_z=1e12)` construction — about **6×** the quoted figure, and the "a few ms" arithmetic follows it. Harmless: the figure is an aside inside an argument about *uncorrelated node scatter*, which the node count does not affect, and the tolerance is right for the reason the rest of the comment gives. But it is a stale figure inside a tolerance justification, which is the class of thing `prompts/tolerance-convergence` exists to remove. **Next step:** two numbers, in whichever prompt next has that file in scope; `prompts/background-solver-robustness/PROVENANCE.md` §1 already carries the measured count. |

**Opened by prompt 05 (2026-09-16):**

| Issue | Board | Hook |
|---|---|---|
| `[05-black-check-is-not-clean-at-the-repository-root]` | background-solver-robustness | `CLAUDE.md` and every campaign README say the tree is clean under `black --check`. It is not, and was not before prompt 05: `./venv/bin/python -m black --check .` reports **54 files would be reformatted**, all of them under `docs/` in per-review or per-benchmark scratch directories (`docs/gk-wkb-review-fable-2026-09-09/`, `docs/adaptive-levin-benchmark/levin_bench/`, …). No production file and no file any campaign has touched is among them — including `docs/qcd-background-audit/measure_T_z_representation.py`, which prompt 05 edited and which is clean. What is wrong is the statement: an agent running the rule as written sees 54 failures it did not cause, and either reformats them or learns to ignore the rule. **Next step:** either run `black` over `docs/` in a prompt whose whole job that is, or narrow the convention to the packages it actually governs — a decision for the user. |

**Opened by prompt 09 (2026-09-16):**

| Issue | Board | Hook |
|---|---|---|
| `[09-retire-tag-test-docstring-cites-a-superseded-digest]` | background-solver-robustness | `ComputeTargets/tests/test_retire_samples_per_decade_tag.py:29-32`'s module docstring says `test_source_grid.py` pins the production grids to "1,996 samples / digest `a2c32f67` (QCD) and 1,778 / `60a3205a` (LambdaCDM)". Since prompt 09 the QCD digest is `4849552b`; the count and the `LambdaCDM` half are still right. Harmless — that file asserts only `SOURCE_GRID_CONSTRUCTION_VERSION` and `T_Z_REPRESENTATION_VERSION`, the digest appears in prose explaining why it need not assert the grid, and the suite is green at 449 — but it is a stale figure in a docstring arguing that a prompt moved no grid. The file is not in prompt 09's permitted list. **Next step:** one word, in whichever prompt next has that file in scope. |

**Opened by prompt 02 (2026-09-16):**

| Issue | Board | Hook |
|---|---|---|
| `[02-bracketed-reference-is-not-the-exact-root]` | background-solver-robustness | README §3.1 makes the bracketed `brentq` reference "the anchor every measurement is scored against", and on the one pair where an exact oracle exists the anchor is the less accurate of the two. `match_rho` for matter = $\Lambda$ is exactly $\rho_{m0}(1+z)^3-\rho_\Lambda$, so the root follows in closed form from the model's own floats: at 60 digits it is `0.303423032996407410561312801228`. The closed form and the solve give `0.30342303299640738` (**−0.506 ulp, the nearest double**); `bracketed_reference` gives `0.30342303299640749` (**+1.494 ulp, the second-nearest, on the wrong side**). So `LAMBDA_CLOSED_FORM_ULP = 2` measures the reference's own error and the "−2.0 ulp" both logs report for that pair is the anchor, not the solve. Harmless — every assertion passes and the shipped answer is the better of the two — but three more prompts score in ulp against this anchor. **Next step:** prompt 03 reads it before scoring the three closed-form sites; re-wording README §3.1 around an exact oracle is a planning question and the user's, since it would change what prompt 01's tests assert. |

**Adopted from `qcd-background-audit`** — each named "whichever prompt next has these files in
scope" as its next step, and this is the first campaign that does
([`RECONCILIATION.md`](../prompts/background-solver-robustness/RECONCILIATION.md) §9.2). Each is
still that board's issue and closes on **its** §4. **All five have**: `[08-…]` by prompt 04,
`[07-t-photon-range-logic-recomputes-its-bounds]`,
`[06-t-photon-call-cost-needs-a-quiet-machine]` and
`[09-audit-script-section-5-prose-counts-the-wrong-set]` by prompt 05, and
`[03-qcd-inventory-does-not-report-the-representation]`'s `QCD_Cosmology` half by prompt 07 —
whose `BackgroundModel` half prompt 14 of that campaign widened it with stays open, narrowed, in
§1.7 above, unassigned (prompt 07's files-may-touch list did not extend there).

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
| `[03-derivative-pad-clamp-on-coarse-grids]` | source-remediation | The background derivative-fit padding is clamped near $z=0$; harmless at the shipped 100 samples/decade, binds at 50. A trap only if `source_samples_log10z` is lowered. |
| `[00-tk-superhorizon-ic-series]` | GkTk-remedial | Once the $T_k$ numeric `atol` is fixed (prompt 12), the floor is the super-horizon initial condition $T=1,T'=0$ at $2.5\times10^{-6}$; removable with the series $T\approx1-x^2/10$, a spec-level decision. |
| `[01-lambdacdm-hubble-rounding-floor]` | GkTk-remedial | `LambdaCDM.Hubble` carries 2–9e-15 relative in double precision, which floors $\Delta\tau$ over one grid interval at $4$–$6\times10^{-5}$ rad at $k=3\times10^8$ whatever the Gauss order or storage width. |
| `[02-qcd-reference-floor]` | GkTk-remedial | The QCD $\tau$/$\tau_s$ references in `wkb_reference_data.json` are themselves good only to 1.9e-14 relative, which is exactly where prompt 02's order-4 tables land. Do not assert tighter for QCD $\tau$ at the nodes. **Narrowed by `qcd-background-audit` prompt 06 (2026-09-14):** the circularity is gone — the references are now built on a background that reproduces the defining equation to 6.807e-11 — and the model-against-JSON agreement is 2.104e-15 / 2.212e-15, an order *below* the recorded floor. The floor itself needs prompt 08's `residual_convergence.py` re-run before this can close. |
| `[03-qcd-short-baseline-reference-endpoint-rounding]` | GkTk-remedial | The QCD short-baseline references in `wkb_reference_data.json` integrate between rounded `log1p(z)` endpoints and carry up to ulp(u)/W ≈ 1e-13 relative on the 37 % fractions; the shipped table agrees with an exact-endpoint `quad` to ≤ 8.8e-16. Assert README §6's 1e-13 for QCD short baselines, not the JSON's self-agreement. |
| `[03-integrationsolver-stepping-minimum-lookup]` | GkTk-remedial | `IntegrationSolver` lookups match `stepping >= requested`; harmless while every table is order 4, but a second Gauss order under the label `cumulative-GL` could be served by the other order's row. |
| `[04-background-rhs-evaluations-count]` | GkTk-remedial | `compute_background` builds three tables but `IntegrationData` has one counter, which prompt 03's test pins to the $\tau$ table alone; the other two counts are payload keys, so the persisted `RHS_evaluations` understates the build 3×. |
| `[06-metadata-column-headroom]` | GkTk-remedial | The WKB integrations' `metadata` column is `String(256)`. **Narrowed by prompt 14:** with its `rho_reused` key the longest payload is 227 characters, so 29 remain; a test now asserts the length rather than letting it overflow. SQLite does not enforce it, PostgreSQL would. Count before adding a key. |
| `[06-docs-scripts-reference-removed-ode]` | GkTk-remedial | Nine `docs/` reproduction scripts import symbols the campaign removed: seven the phase ODE (`integrate_phase_function`, `stage_*_evolution`, `DEFAULT_OMEGA_WKB_SQ_MAX`, prompt 06), and `TK_04_WKB_reconstruction.py` and `baseline_k1e5.py` the friction ODE `friction_RHS` (prompt 07, relocated into prompt 04's test). They documented the tree they ran on and were not edited. |
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
| `[02-verify-script-builds-its-own-Gk-consumer]` | phase-representation | `docs/gktk-remedial/verify_production_path.py` calls `PrimitivePhase(...)` directly at `:557` and `:1173` instead of going through `GkSourcePolicyData._build_phase`, so six of §3.5's twelve rows and both `theta_deriv` $G_k$ columns are blind to anything the production $G_k$ call site passes. Its $T_k$ half does use `TkSourceFunctions`. Same class as `[13-scoped-run-driver-k-grid-literal]`. |

---

## 5. Standing caveat that is not an issue

**The verification runs never reached production $x$.** `source-remediation`'s run A used
`zend = 1e7` to stay inside radiation domination where `analytic_rad` is a valid oracle, so its
largest accumulated phase was $x = 4.63\times10^5$ against $x\sim1.4\times10^7$ at the production
`zend = 0.1` for the largest $k$. Every "verified live" claim in that campaign carries this
ceiling, and `[12-phase-spline-error-grows-with-x]` is the term it matters most for.
