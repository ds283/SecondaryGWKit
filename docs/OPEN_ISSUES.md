# Open issues — project-wide index

**Last updated:** 2026-09-14 · **61 open** across eight campaigns.

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
[`qcd-background-audit`](../prompts/qcd-background-audit/IMPLEMENTATION_STATE.md)

---

## 1. Assigned to a future campaign

Work is identified and owned; the issue is parked deliberately, not forgotten.

### 1.1 The hand-over campaign

The numeric→Liouville–Green seam of $T_k$ and $G_k$. These seven are **one place** and must be
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
| `[20-wkb-gauss-orders-not-in-lookup-key]` | GkTk-remedial | Prompt 20's §5 audit refutes "the other four compute targets have no equivalent free parameter": `TAU_GAUSS_ORDER`, `CS_TAU_GAUSS_ORDER`, `FRICTION_F_GAUSS_ORDER`, `RHO_GAUSS_ORDER` (all 4) and `RESIDUAL_WKB_REGION_MARGIN = 0.5` are configuration axes in no `BackgroundModel`, `GkWKBIntegration` or `TkWKBIntegration` lookup key. The orders at least move a solver label, which no factory filters on; the margin moves no label, tag or column at all. Latent, not live — prompt 14 measured the margin at $\le1.4\times10^{-17}$ rad in $\rho$ with $\theta$ bit-identical. |
| `[20-wkb-rows-consume-numeric-initial-data]` | GkTk-remedial | `Gk`/`TkWKBIntegration` take $z_{\rm init}$, $G_{\rm init}$/$T_{\rm init}$ and the derivative from the numeric stop point and are keyed independently of the numeric row: no foreign key, and the initial values are stored `nullable=False` but never filtered. Covered in practice only because `z_init` is filtered as an absolute `1e-7` against $z\sim10^{12}$, i.e. exactly — measured on QCD at $k=4.972\times10^7$, the two break-point policies move $z_{\rm init}$ by 4.59e5 and the lookup misses. A change moving the stop *values* without moving $z_{\rm init}$ would be served a stale row. |
| `[10-wrap-theta-loop-at-large-phase]` | GkTk-remedial | `wrap_theta` reduces by adding $2\pi$ in a loop, so at $|\theta|\sim10^6$ rad it takes ~1.6e5 iterations and reconstructs $\theta$ only to 1.39e-06 rad. Inert in production (its one caller passes `mod + delta`), a trap for fixtures. The companion defect in `WKB_mod_2pi`'s cycle count was fixed by `phase-representation` prompt 01 (2026-09-13), so that function is now exact in both halves; this loop is not. |
| `[13-scoped-run-driver-k-grid-literal]` | GkTk-remedial | `docs/source-remediation-verification/scoped_pipeline_run.py` matches a `main.py` k-grid literal that `f17f2d4` renamed to `NUMBER_SOURCE_K_VALUES`/`NUMBER_RESPONSE_K_VALUES`, so it finds zero occurrences and raises rather than running. The `source-remediation` Layer 2 is not reproducible by its own documented command; prompt 13 copied the driver into `docs/gktk-remedial/` rather than editing another campaign's file. |

---

### 1.5 The tolerance and convergence campaign

Planned as [`prompts/tolerance-convergence/`](../prompts/tolerance-convergence/README.md)
(2026-09-12; five prompts, none executed). One reusable convergence test applied to all five
compute targets on all three models, anchored to the constant-$w$ closed forms in
`ComputeTargets/analytic_{Gk,Tk}.py`, and `atol`/`rtol` decoupled so each quantity carries its own
justified pair. **Unblocked, without a caveat, by `prompts/GkTk-remedial` prompts 18 and 19
(2026-09-13)** — the numeric ODE is now split at the cosmology's declared non-smoothness, and
*which* kind is the caller's choice: the $T_k$ integrator asks for every declared break point, the
$G_k$ integrator for the jumps alone, each on measurement. **Both sectors now converge at all 50
production wavenumbers on all three models** — worst reference-convergence drift on `QCDModel`
8.72e-09 ($T_k$) and 8.41e-09 ($G_k$) against the 3.4e-08 criterion, where prompt 18 still left
three $T_k$ wavenumbers at up to 1.97e-07. `[17-qcd-reference-not-converged]` is **closed**, and a
$T_k$ tolerance may now be measured on QCD. Note the cost this campaign inherits: a QCD $T_k$
numeric object is ~31.5k right-hand-side evaluations rather than ~9.8k (+220 %, ~49 s for the
50-object sector per model); $G_k$ is unchanged at ~13.3k.

| Issue | Board | Hook |
|---|---|---|
| `[12-tk-numeric-atol-largest-k-excursion]` | GkTk-remedial → tolerance-convergence | Prompt 12's `atol=1e-13` left excursions above README §6's 3e-6 of the envelope that prompt 17 then measured across the production grid: 3 / 13 / 8 of 50 wavenumbers on Radiation / LambdaCDM / QCD, worst 8.64e-4, each a raised level rather than one bad sample. `atol=1e-16` does not fix it and is not cheaper. **The user settled the constant 2026-09-12: `1e-13` stays** — `atol` is not the lever. One decade of `rtol` removes every excursion for +23–25 % evaluations, and `rtol` is one shared number keying every integration object. **Assigned (2026-09-12): `prompts/tolerance-convergence`**, which sweeps it per sector and decouples the constants. | |

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
prompts in four workstreams, **6 executed**; prompts 10–12 gated on that README's §7 D7). It
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
have caught it (01). It is a **precondition for `prompts/tolerance-convergence`** (§1.5), whose QCD
half would otherwise be measured against a background about to move.

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
at all 17 digits, so the 1.4e5 rad at $k=3\times10^8$/Mpc is 0.000e+00 rad. What is left of the
campaign's chain is the break-point set: prompt 06 pushed `BREAK_POINT_ALL` from 407 to **2,414**
(the node count's price, all of it knots), and prompts 07–08 take it to 3.

| Issue | Board | Hook |
|---|---|---|
| `[00-eos-branch-joins-do-not-match]` | qcd-background-audit | `QCD_EOS`'s branch joins at $10^{16}$, 0.12 and $10^{-5}$ GeV jump by +1.395e-02, −3.744e-04 and −2.284e-03 in $g_s$, forcing steps in $T(z)$; the join at 0.002 GeV matches to 1.751e-11, and that asymmetry is the evidence the other three are a transcription defect. Origin of the 4.4e-04 jump in $H(z)$ at $z=4.24\times10^7$ that `GkTk-remedial` log 02 measured without attribution. **Upstream data fixture; pinned in a test by prompt 01 (2026-09-14), which confirmed every figure here to the digits quoted; not repaired.** The question for its authors is that campaign's README §7 D6. |
| `[19-cosmologymodels-docstrings-predate-per-sector-policy]` | GkTk-remedial → qcd-background-audit | `GenericEOS.py:97-101` and `LambdaCDM_GenericEOS.py:277-283` say an adaptive ODE solver only has to be split at a jump, and that the jumps-only set is what `numeric_with_phase_cut` asks for. True when prompt 18 wrote them; prompt 19 measured the C2 knots costing an order of magnitude of reference convergence in the $T_k$ sector, which now asks for `BREAK_POINT_ALL`. Documentation only; `CosmologyModels/` was out of bounds for prompt 19. |
| `[13-consumer-spline-crosses-eos-break-points]` | GkTk-remedial → qcd-background-audit | `PrimitivePhase`'s cubic spline of $\varphi$ uses default knots across `QCD_EOS`'s declared break points: 1.9e-6 / 3.2e-6 rad at $z=4.24\times10^7$ ($G_k$ / $T_k$, $k=10^5$) against 1 ulp elsewhere. **Narrowed by prompt 02 (2026-09-13), which stopped rather than fixing it:** a repeated-knot vector is singular on all six production grids at `BREAK_POINT_ALL` and 2× worse at `BREAK_POINT_DISCONTINUITY`, no break point coincides with a sample, and the kink itself is only 1.6e-8 / 1.4e-7 rad — 1 % and 4 % — of the error. **Superseded 2026-09-13** by `qcd-background-audit-2026-09.md`: 404 of the 407 `BREAK_POINT_ALL` points are knots of the `T(z)` spline itself, not cosmology, so the fix is to remove that artefact upstream — after which the set falls to 3 and a knot vector constructs. Owned by the `qcd-background-audit` campaign, prompt 10. |
| `[01-convergence-block-has-a-separate-generator]` | qcd-background-audit | `wkb_reference_data.json`'s top-level `convergence` block (geometry, per-order convergence, `decision.N_*`) is written by `docs/gktk-remedial/residual_convergence.py`, not by prompt 02's QCD-only regenerator; five tests read it directly. Stale the moment 04–06 move the QCD block unless something re-runs it. **Next step:** prompt 08, whose charter already re-takes this measurement. |
| `[02-fixture-tests-pinned-to-todays-break-point-artefact]` | qcd-background-audit | Two assertions hard-code today's ~404-knot `BREAK_POINT_ALL` count and go **false**, not just stale, once prompt 07 collapses it to 3: `test_numeric_break_points.py::test_kind_selects_knots_or_jumps` (`len(every) > 100`) and `test_background_tau.py::test_qcd_break_points`. **Next step:** prompt 07 rewrites both in its own commit. |
| `[03-qcd-inventory-does-not-report-the-representation]` | qcd-background-audit | `sqla_QCDCosmology_factory.inventory()` and `tools/inventory_report.py` show QCD cosmology rows without the `T_z_representation` column prompt 03 added, so from prompt 04 rows differing only in their representation render as indistinguishable duplicates to the only tool that inspects a datastore. One line in `inventory()`; out of scope for prompt 03, whose §2 item 4 fixes the key and nothing else. |
| `[04-unsplit-tk-run-now-meets-the-criterion]` | qcd-background-audit | `test_split_converges_where_unsplit_does_not` asserted that an unsplit $T_k$ numeric run *fails* the 3.4e-08 criterion at $k=4.972\times10^7$/Mpc. False since prompt 05: unsplit drift 1.0213e-06 → **2.2767e-08** (45× better) while the split run barely moved, so most of what the split rescued was the old representation's interpolation noise, not the jump in $H(z)$. First measured evidence that `BREAK_POINT_KIND = BREAK_POINT_ALL` may no longer be load-bearing — README §2 (f), §7 **D5**, and `BREAK_POINT_KIND` is in a datastore lookup key. One wavenumber of fifty; **not decided by prompt 05**. **Next step:** prompt 08, which re-takes the measurement across all fifty. |
| `[05-break-point-set-grew-with-the-node-count]` | qcd-background-audit | `BREAK_POINT_ALL` on the production grid is **2,414** where it was 407 (2,411 tabulation knots + 3 crossings, median spacing 0.67× the grid): prompt 06 needed 3,000 nodes rather than 500 to reach the p90 and median, and every node is a declared break point. Costs 2.337× the order×intervals baseline in the `rho` table where it cost 1.24×, so `COST_BREAK_POINT_FACTOR` went 1.30 → 2.40. **Next step:** prompt 07, which removes the knots from the set; the factor should then come back below 1.30, not to it. |
| `[06-t-photon-call-cost-needs-a-quiet-machine]` | qcd-background-audit | `T_photon` measured **2.53 µs/call** on a quiet machine against README §6.2's ≤ 2.5 µs, before the segment dispatch was inlined; after the inline, ratios on a loaded machine put the segmentation at ~1.00–1.05× the unsegmented cost and the whole excess over prompt 05 at the order-5 spline evaluation (1.09–1.13×). A scaled estimate is ~2.4 µs but that is an inference. **Next step:** re-run log 06 deviation 5's three-candidate comparison on a quiet machine. |
| `[07-t-photon-range-logic-recomputes-its-bounds]` | qcd-background-audit | `TemperatureRepresentation.__call__` and `ZSplineWrapper.__call__` evaluate `_outward(bound, ±1)` on every call, though both bounds are fixed at construction: 0.056 µs each, measured, of a ~2.5 µs call. Hoisting them into `__init__` is numerically null. Out of scope for prompt 06, whose prompt did not cover prompt 05's range logic. |

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
