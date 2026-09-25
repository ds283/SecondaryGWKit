# Implementation state — the QCD background campaign

**Campaign:** [`README.md`](README.md) · **Source document:**
[`docs/qcd-background-audit-2026-09.md`](../../docs/qcd-background-audit-2026-09.md)
**Baseline commit:** `e8f746d` (`qcd-background-audit`, clean; identical to `main`)
**Last updated:** 2026-09-16 — **16 / 16: workstream E is complete and the campaign is closed
again.** Four of this board's §3 issues have since been closed *by another campaign*:
`prompts/background-solver-robustness` adopted them under its `RECONCILIATION.md` §9.2, because
each named "whichever prompt next has these files in scope" as its next step. Its prompt 04 closed
`[08-temperature-crossing-solver-is-test-only]`, and its prompt 05 closed
`[07-t-photon-range-logic-recomputes-its-bounds]`,
`[06-t-photon-call-cost-needs-a-quiet-machine]` — **README §6.2's ≤ 2.5 µs `T_photon` row now
holds, at 2.4854 µs** — and `[09-audit-script-section-5-prose-counts-the-wrong-set]`. All four are
in §4 below; the measurements are in that campaign's logs and the entries say so. **Its prompt 07
has now landed too, and closed half a fifth**: `[03-qcd-inventory-does-not-report-the-representation]`'s
`QCD_Cosmology` half is in §4; the `BackgroundModel` half prompt 14 widened that issue with is
**not** in that campaign's files-may-touch list, so it stays open here, narrowed, under the same
ID. **Updated 2026-09-18: a sixth is closed, also by another campaign.**
`prompts/tolerance-convergence` prompt **04b** regenerated the `convergence` block of
`ComputeTargets/tests/wkb_reference_data.json` and took `QCD_BREAK_POINT_ALIGNMENT_TOL` from
1.5e-04 to **3.0e-14**, so `[01-convergence-block-has-a-separate-generator]` — declined on scope
by this campaign's prompts 08 and 09 — is in §4, and both `QCD_FLOOR_FACTOR`s are gone, the two
tests that read the block now bounding their own production quantities. Nothing in this
campaign's own work was re-opened. **Updated 2026-09-25: the fifth is now closed in full,
by `prompts/store-fingerprint` prompt 03.** `[03-qcd-inventory-does-not-report-the-representation]`'s
`BackgroundModel` half is in §4 with the rest of it: that campaign retired every factory's
`inventory()`, and its display renders each background's `source_grid_digest` and
`source_grid_construction`.

> **The campaign reopened after it closed.** Prompt 12 measured a defect it was forbidden to act on
> — `BackgroundModel` splined `d_lnH_dz` over a padded refinement of the source grid that was
> **not** split at the cosmology's break points, so `epsilon` rang by 2.04e-02 relative at `T_LO`
> and 1.03e-03 at `T_120_MEV` on the production grid — and that is why prompt 11's fix did not
> reach production: in the configuration `main.py` actually uses, the consumer's phase error at the
> crossing read **2.4859e-05 rad** where prompt 11 measured 3.9744e-07 with the background held on
> the base grid. **Prompt 13 has now closed it.** Both spline sites are segmented at
> `_cosmology_break_points`, the ringing reads **1.9e-09 / 1.6e-09 / 4.1e-09** at the three
> crossings against the 3.9e-09 that holds away from one, and the production row is **5.8437e-08
> rad** — **425× better** and 6.8× inside the acceptance. `T_Z_REPRESENTATION_VERSION` is **6**.
>
> **Prompt 14 has now closed the other thing prompt 11 left behind: nothing recorded *which run* an
> object belonged to.** A run is named at write time (`main.py --run-label`, a `store_tag` carried
> by and filtered on for everything the run writes, in `TkProductionTag`'s manner), the grid's
> construction algorithm is named by `SOURCE_GRID_CONSTRUCTION_VERSION = 1` beside prompt 11's
> content digest, `BackgroundModel`'s lookup key carries both the digest and the version, and all
> six `extract_*.py` select on the run label and refuse a mixture rather than picking. **A store
> written before prompt 14 cannot be computed into and can still be read** — two different answers,
> deliberately. The production grid is **byte-identical** to prompt 11's (3,814 `float.hex()` lines,
> MD5 `d5ecc0aa85f38578d8c57c051d3f11e0`) and `T_Z_REPRESENTATION_VERSION` is still **6**.
>
> **Prompt 15 has now taken prompt 12's recommendation, which the user decided to take, and the
> campaign is closed again.** The source grid's base density is no longer a uniform
> `source_samples_per_log10z`: every base interval the criterion
> $h^4|\varphi''''|/384 \le \varepsilon$ finds too wide is subdivided, in the base grid's own
> coordinate, by the smallest integer that satisfies it — under the cap
> `SOURCE_GRID_MAX_SPACING_FACTOR = 1.0`, **"never coarser than the uniform lattice puts there,
> anywhere"**, which is the user's instruction and is measured as exactly **1.0000×** on both
> models. The two rows prompt 12 recorded as a miss come inside their floors in the **production**
> configuration — LambdaCDM $T_k$ at $k=10^5$ **7.86 → 0.69 ulp**, QCD **7.84 → 0.11** — and the
> grids go QCD 1,773 → **1,996** and LambdaCDM 1,732 → **1,778**, *larger* than prompt 12's cap-1×
> column rather than smaller, because that column's cap was a single scalar over a lattice that is
> uniform in $\log_{10}z$ and not in $u$ (log 15 deviation 1). `SOURCE_GRID_CONSTRUCTION_VERSION`
> is **2**; `T_Z_REPRESENTATION_VERSION` is still **6** and no background value moves.
>
> **Prompt 16 has now retired the samples-per-decade tag that prompt 15 left false.**
> `SourceSamplesPerLog10ZTag` labelled stored objects `SourceSamplesPerLog10Z_100` as though the
> grid had 100 samples per decade of z, when it now merely bounds a density the curvature
> criterion sets; the tag is gone from its one declaration, its one label construction and all 27
> `tags=[...]` uses in `main.py`, and so are the two vestigial `delta_logz=` arguments it fed
> (established as unused by anything since prompt 11 of `prompts/GkTk-remedial` — the
> oscillation-resolution diagnostic now scores the returned sample grid, not a nominal spacing).
> **`--source-samples-log10z` stays**, its help text corrected to say it sets the base lattice the
> cap is measured against, per the user's decision (prompt 16 §1). Measured, not assumed: every
> tagged `object_get` in the tree joins once per *requested* tag with no check that a row carries
> only those tags, so dropping a tag from the list can only broaden a query — a pre-prompt-16 row,
> which still carries `SourceSamplesPerLog10ZTag` as harmless history, still resolves against the
> shorter post-prompt-16 list. **This commit carries no datastore regeneration.**
> `SOURCE_GRID_CONSTRUCTION_VERSION` stays **2** and `T_Z_REPRESENTATION_VERSION` stays **6** —
> nothing about the grid's construction or the background changes, only what main.py says about
> the grid's density. `docs/source-remediation-verification/run_quadsource_integrals.py` is left
> untouched: it already predates prompts 11/14 of this campaign (no `RunLabelTag`, no
> `SourceGridConstructionTag`, the pre-prompt-11 size-only `SourceRedshiftGrid_{len}` tag), so
> editing out one tag among six would misrepresent a script that is already several tagging
> generations behind main.py as though it had been kept current. Closes the *tag* half of
> `[15-the-grid-now-depends-on-the-wavenumber-sample-and-no-tag-says-so]`; its *wavenumber-set*
> half stands as the user's explicit no-action decision (§3 below).

The record of the original twelve follows. **12 / 12: that chain is closed.** The ungated chain 01–09 is complete; T1 and G1 are closed and verified,
G1's per-sector question is answered, and **P2 is answered, though not in the way the plan
expected**: prompt 10 measured every constructible break-point knot scheme against the corrected
background and **none of them helps**, so `PrimitivePhase` keeps its default knots and the defect
moved to the source grid — where **prompt 11 has now fixed it**, by building the grid around the
points the cosmology declares. **Prompt 12 has now measured the density question the audit left
open and stated the answer**: the uniform `source_samples_per_log10z` is wrong in *both* directions
— under-resolved by 7.8× the storage floor at the top of the $T_k$ band at $k=10^5$, over-resolved
by up to $10^{19}$ at the bottom — and a criterion computable before the grid exists fixes it at
the same cost or saves 1.75×–2.06× at the same accuracy. **It changes nothing; the decision is the
user's.** Twelve prompts in four workstreams.

> **What a reader should conclude.** The QCD background is now correct — its $\int\mathrm{d}z/H$ is
> *bit-identical* to an independently root-solved exact background where it carried 3.461e-08, and
> the 1.43e5 radians that error was worth at $k=3\times10^8$/Mpc are 0.000e+00 — at the price of
> making one consumer defect **4.3× more visible**, because a cubic spline of $\varphi$ meets the
> equation of state's genuine step undiluted where the old representation smeared three quarters of
> it over the neighbouring grid intervals. That is
> `[13-consumer-spline-crosses-eos-break-points]`, it is **prompt 10's**, and it is the one reason
> to release workstream D. **Prompts 10 and 11 have now closed it.** Prompt 10 re-attributed it and
> prompt 11 fixed it where the attribution pointed: the production source grid carries the
> cosmology's declared crossings, 41 extra samples in 1,732, and the two rows read **1.61 ulp**
> and **65.78 ulp** — better than the campaign base and inside the 1e-06 rad consumer target. What
> follows is prompt 10's re-attribution, which is why the fix is in the grid and not in the spline.
> **Prompt 10 ran and re-attributed it** (2026-09-15): the consumer's knot
> placement cannot recover it at all — the $C^0$ repeated knot is 2.09× worse and per-segment
> splines 5.00× worse — while **10 extra samples in 1,016**, placed over the ±5 grid intervals
> around the crossing, bring both rows inside the 1e-06 rad consumer target. The feature is three
> to five grid intervals wide and the production source grid does not resolve it; that entry is
> closed and `[10-consumer-phi-unresolved-at-the-eos-crossing]` takes its place, assigned to
> **prompt 11**.
Every figure below is the audit's, and prompt 01 re-measured the representation, the branch joins,
the jump locations and the conformal-time error **from the test tree** on `2a5e0fa`: all of them
reproduce the audit to every digit it quotes. The audit's script
(`PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/measure_T_z_representation.py`) remains
the reproduction for the figures prompt 01 did not take.

---

## 1. Status board

Legend: ⬜ not started · 🟡 in flight · ✅ complete · ⚠️ complete with deviations · ⛔ blocked

### Workstream A — the representation (prompts 01–06)

| # | Prompt | Model | Status | Commit | Log |
|---|---|---|---|---|---|
| 01 | [The background-against-background harness](01-background-reference-harness.md) | Opus | ⚠️ | *"Add a background-against-background test for the QCD temperature"* (SHA not embedded, per the campaign convention) | [`logs/01-background-reference-harness.md`](logs/01-background-reference-harness.md) |
| 02 | [Make the QCD reference fixture regenerable](02-regenerable-qcd-references.md) | Sonnet | ⚠️ | *"Make the QCD reference fixture regenerable before the background moves"* (SHA not embedded, per the campaign convention) | [`logs/02-regenerable-qcd-references.md`](logs/02-regenerable-qcd-references.md) |
| 03 | [Key the `T(z)` representation](03-key-the-representation.md) | Opus | ⚠️ | *"Key the QCD cosmology on its temperature representation"* (SHA not embedded, per the campaign convention) | [`logs/03-key-the-representation.md`](logs/03-key-the-representation.md) |
| 04 | [Tighten `_solve_T_z` (T2)](04-tighten-node-solve.md) | Sonnet | ⚠️ | *"Solve the T(z) spline nodes to a useful tolerance"* (SHA not embedded, per the campaign convention) | [`logs/04-tighten-node-solve.md`](logs/04-tighten-node-solve.md) |
| 05 | [Spline the entropy factor (T3)](05-entropy-factor-representation.md) | Opus | ⚠️ | *"Spline the entropy factor rather than the temperature itself"* (SHA not embedded, per the campaign convention) | [`logs/05-entropy-factor-representation.md`](logs/05-entropy-factor-representation.md) |
| 06 | [Segment at the jumps (T4)](06-segment-at-the-jumps.md) | Opus | ⚠️ | *"Segment the QCD temperature at its genuine discontinuities"* (SHA not embedded, per the campaign convention) | [`logs/06-segment-at-the-jumps.md`](logs/06-segment-at-the-jumps.md) |

### Workstream B — the break-point set (prompts 07–08)

| # | Prompt | Model | Status | Commit | Log |
|---|---|---|---|---|---|
| 07 | [Re-derive `integration_break_points` (G1)](07-rederive-break-points.md) | Opus | ⚠️ | *"Declare only the cosmology's own break points, not the spline's knots"* (SHA not embedded, per the campaign convention) | [`logs/07-rederive-break-points.md`](logs/07-rederive-break-points.md) |
| 08 | [Re-measure the per-sector break-point policy](08-per-sector-policy-remeasure.md) | Opus | ⚠️ | *"Re-measure the numeric break-point policy on the corrected background"* (SHA not embedded, per the campaign convention) | [`logs/08-per-sector-policy-remeasure.md`](logs/08-per-sector-policy-remeasure.md) |

### Workstream C — close-out (prompt 09)

| # | Prompt | Model | Status | Commit | Log |
|---|---|---|---|---|---|
| 09 | [The consumer tables under a corrected background](09-close-out-verification.md) | Opus | ⚠️ | *"Verify the QCD background remediation against both consumers"* (SHA not embedded, per the campaign convention) | [`logs/09-close-out-verification.md`](logs/09-close-out-verification.md) |

### Workstream D — the source grid and the consumer spline (prompts 10–12) — **gated on README §7 D7**

| # | Prompt | Model | Status | Commit | Log |
|---|---|---|---|---|---|
| 10 | [`PrimitivePhase` on the 3-point break set](10-primitive-phase-break-points.md) | Opus | ⚠️ | *"Measure the consumer phase spline against the cosmology's break points"* (SHA not embedded, per the campaign convention) | [`logs/10-primitive-phase-break-points.md`](logs/10-primitive-phase-break-points.md) |
| 11 | [A cosmology-aware source grid](11-cosmology-aware-source-grid.md) | Opus | ⚠️ | *"Build the source grid around the features the cosmology declares"* (SHA not embedded, per the campaign convention) | [`logs/11-cosmology-aware-source-grid.md`](logs/11-cosmology-aware-source-grid.md) |
| 12 | [A measured grid-density criterion](12-grid-density-criterion.md) | Opus | ⚠️ | *"Measure what the source grid's density buys and what it wastes"* (SHA not embedded, per the campaign convention) | [`logs/12-grid-density-criterion.md`](logs/12-grid-density-criterion.md) |

### Workstream E — the background's own derivative lattice, and the grid's identity (prompts 13–16)

| # | Prompt | Model | Status | Commit | Log |
|---|---|---|---|---|---|
| 13 | [Segment every background spline at the cosmology's break points](13-segment-background-derivative-splines.md) | Opus | ⚠️ | *"Segment the background derivative splines at the declared break points"* (SHA not embedded, per the campaign convention) | [`logs/13-segment-background-derivative-splines.md`](logs/13-segment-background-derivative-splines.md) |
| 14 | [Key the source-grid construction](14-key-the-source-grid-construction.md) | Opus | ⚠️ | *"Name a run at write time and verify it at read time"* (SHA not embedded, per the campaign convention) | [`logs/14-key-the-source-grid-construction.md`](logs/14-key-the-source-grid-construction.md) |
| 15 | [Equidistribute the source grid](15-equidistribute-the-source-grid.md) | Opus | ⚠️ | *"Equidistribute the source grid on the phase residual's curvature"* (SHA not embedded, per the campaign convention) | [`logs/15-equidistribute-the-source-grid.md`](logs/15-equidistribute-the-source-grid.md) |
| 16 | [Retire the samples-per-decade tag](16-retire-the-samples-per-decade-tag.md) | Sonnet | ⚠️ | *"Retire the samples-per-decade tag the grid no longer obeys"* (SHA not embedded, per the campaign convention) | [`logs/16-retire-the-samples-per-decade-tag.md`](logs/16-retire-the-samples-per-decade-tag.md) |

**Progress:** **16 / 16 complete.** (12 / 12 in the original campaign; 4 / 4 in workstream E, which
reopened it.) The line below describes the original twelve. (9 / 9 in the ungated chain 01–09;
workstream D released and all three of 10, 11 and 12 have run.) Prompt 12 **measures and
recommends and changes nothing**, which is what its §4 asks for: the `Result` is `COMPLETE` when
the measurement is made and the recommendation is stated, not when a grid changes. The
recommendation is `docs/qcd-background-verification.md` §10.0 and
`[12-source-grid-density-is-uniform-over-a-curvature-that-spans-eight-orders]`, now in §4 below,
and **the decision was the user's**, taken with a full regeneration of eight stored object types in
front of them. **It was taken, and prompt 15 implemented it.**

**Prompt 01 landed `COMPLETE WITH DEVIATIONS`** — no production file changed; two new modules,
`CosmologyModels/tests/T_z_reference.py` and `CosmologyModels/tests/test_T_z_representation.py`,
seven new tests, `CosmologyModels` 11 → 18. The five deviations are all recorded in
[its log](logs/01-background-reference-harness.md) and none touches a README §2 design fact; the
two that matter to later prompts are (2) the jump height is **7.6229229003969e-04** in $F$ and
**7.625829e-04** in $T$, the audit's 7.614e-04 being §1's *linearised* figure, and (3) a
`root_scalar` bracket on $T(z)-T_{\rm break}$ reports `converged` and returns a **non-root** —
which is audit §2's claim confirmed, in the second of the two forms prompt 01 allowed for.

**Prompt 02 landed `COMPLETE WITH DEVIATIONS`** — no production file changed, no test changed, and
`ComputeTargets/tests/wkb_reference_data.json` is untouched (`git status --porcelain` empty before
and after). Two new files: `docs/qcd-background-audit/generate_qcd_references.py` (the
QCD-only reduction of `docs/gktk-remedial/generate_references.py`; `--dry-run` reproduces the
shipped QCD block's science content bit for bit in 112.3 s, no Ray, no datastore) and
`docs/qcd-background-audit/REFERENCE-FIXTURE.md` (the map: 45 QCD-dependent test methods across
the eleven `ComputeTargets/tests/` modules plus `CosmologyModels/tests/test_T_z_representation.py`,
classified by what they are scored against and which of {node, shape, breaks, none} would move
them). Two findings opened as new issues below: the JSON's separate top-level `convergence` block
(written by `docs/gktk-remedial/residual_convergence.py`, not this prompt's generator) also carries
QCD-specific figures that prompts 04–08 must account for explicitly; and two existing test
assertions (`test_kind_selects_knots_or_jumps`, `test_qcd_break_points`) are pinned to today's
~404-knot artefact and will need editing, not just re-measuring, once prompt 07 lands. Suite counts
unchanged: `CosmologyModels` 18, `ComputeTargets` 339, `LiouvilleGreen` 143/143 on the fast set
(`test_3bessel_analytic` excluded; see the log).

**Prompt 03 landed `COMPLETE WITH DEVIATIONS`** — **no number moved**, demonstrated rather than
asserted: `T_photon`, `Hubble` and `rho` as exact `float.hex()` on 4,001 points over
$z\in[0,10^{19}]$ for `QCD_Cosmology`, `LambdaCDM` and `LambdaCDM_GenericEOS(PureRadiationEOS)` are
**byte-identical** to `3478ae3` (12,003 lines, MD5 `022cbbc1faf075233f54f84d5d0959f8`), and
`generate_qcd_references.py --dry-run` reports no change in all 12 science keys. README §7 **D1
option (i)** was taken, unchanged: `LambdaCDM_GenericEOS.T_Z_REPRESENTATION_VERSION = 1`
(`LambdaCDM_GenericEOS.py:73`, inherited by `QCD_Cosmology`, not shadowed) and a
`T_z_representation` integer column on the QCD cosmology table, filtered on as an equality in
`build()` and written by `insert_data`. A pre-prompt-03 datastore raises `RuntimeError` from
`sqla_QCDCosmology_factory.build()` naming this campaign and demanding regeneration — the mismatch
reaches `build()` as SQLite's `no such column`, **not** as a missing attribute, because
`Datastore._build_schema()` builds the `Table` from the code while `_ensure_tables()` never alters
an existing table. One new module, `ComputeTargets/tests/test_cosmology_representation_key.py`,
15 tests in 0.19 s; `ComputeTargets` 339 → 354. Five deviations, all in
[its log](logs/03-key-the-representation.md); the only `STRUCTURALLY REQUIRED` one is that the
prompt's named precedent (`test_numeric_break_point_key.py`) stands up **no** in-memory datastore —
it compiles queries against a stand-in connection — so the tests create an SQLite engine directly
and prompt §3 items 1 and 3 are scored against real SQL rather than a compiled query. None touches
a README §2 design fact.

**Prompt 04 landed `COMPLETE WITH DEVIATIONS`** — one production line changed:
`_solve_T_z`'s `root_scalar(..., xtol=1e-6, rtol=1e-4)` is now `xtol=1e-300, rtol=1e-14`
(`LambdaCDM_GenericEOS.py:220`), `T_Z_REPRESENTATION_VERSION` is **2**. The node solve is now
bit-identical to the `rtol=1e-14` defining-equation reference on the audit's 640-point probe set
(max relative error **2.496e-05 → 0.0**), and $H(z)$ built from the node solve alone follows it to
**0.0**. The full `T(z)` spline (still one global, un-segmented spline; T3/T4 are prompts 05/06)
improves max/p90/median from **7.177e-04/1.323e-05/1.890e-07** to
**7.2615e-04/1.936e-07/1.071e-07** — the p90 falls as the audit's §4 table predicts, the max is
untouched (and nudges very slightly worse) because it is pinned at the un-segmented jump height,
not set by node accuracy. **The $\pm0.1$-scale $\omega^2/\omega_0^2$ scatter
`ComputeTargets/phase_residual.py:226` attributes to this issue survives essentially unchanged**
(measured span $[-0.0713,+0.2336]\to[-0.0676,+0.2376]$ over 11 production nodes near
$z\sim4\times10^{15}$, $k=3\times10^8$, Gk sector — 0.1 % different): **this narrows, rather than
closes, the question of what causes it**, and `RESIDUAL_WKB_REGION_MARGIN = 0.5` is untouched, as
the prompt requires. The QCD reference fixture was regenerated in this commit (largest relative
move: `rho_G` 1.10e-02); every one of the 45 tests prompt 02 mapped re-scored green except two
whose expected values were pinned to the shipped nodes' branch-crossing location — a hardcoded
literal in `test_numeric_break_points.py` and an alignment tolerance in `test_background_tau.py`
against the JSON's separate, not-regenerated-here `convergence` block, loosened once from `1e-9`
to `1.4e-05` (`[01-convergence-block-has-a-separate-generator]`, not a new defect). `CosmologyModels`
18, `ComputeTargets` 354, `LiouvilleGreen` 143/143 fast set — all unchanged. Closes
`[02-qcd-T-z-spline-node-tolerance]` on the `GkTk-remedial` board (§4 there). Full record:
[`logs/04-tighten-node-solve.md`](logs/04-tighten-node-solve.md).

**Prompt 05 landed `COMPLETE WITH DEVIATIONS`** — `_build_T_z_spline` now tabulates the **entropy
factor** $F(u)=\log\big(T/[T_{\rm CMB}(1+z)]\big)$ rather than $T$ itself, from the *same*
`_solve_T_z` call (no second solve), and a new `TemperatureRepresentation` class in
`LambdaCDM_GenericEOS.py` multiplies the closed-form ramp back in on evaluation
(README §7 **D2**, shape (ii): the range logic and the representation in one place, which is what
prompt 06 segments; `ComputeTargets/spline_wrappers.py` was *not* touched — `_outward` is
imported). **`T_Z_REPRESENTATION_VERSION` is 3.** README §7 **D3**: **500 nodes at $k=3$ kept**,
deliberately — the max is pinned at the jump height at 500, 1,000, 2,000 and 3,000 nodes alike, so
extra nodes buy only the p90 and the median and are paid for one-for-one in declared break points;
holding the count fixed leaves `BREAK_POINT_ALL` at **407** (404 knots + 3 crossings) and
`BREAK_POINT_DISCONTINUITY` at **2**, exactly as prompt 07 inherits them. On the audit's 640-point
probe set the representation goes **7.2615e-04 / 1.936e-07 / 1.071e-07** →
**7.236e-04 / 8.912e-08 / 2.599e-10** (max / p90 / median), reproducing audit §4's entropy-factor
row to the digit; a constant-$g_s$ equation of state is now **exact** (2.928e-16, ~1.3 ulp, against
1.940e-07). Two results beyond the prompt's targets: the **T1 guard fell 3.4509e-08 → 5.4264e-10**,
a factor of 64 from the entropy factor alone (the threshold stays at 4.0e-08 — prompt 06 owns it —
with the measurement recorded in its comment), and $H(z)$ on the probe set reads
1.280e-03 / 1.923e-07 / 5.430e-10. Costs *fell*: `T_photon` 2.240 µs/call, `_build_T_z_spline`
13.3 ms. LambdaCDM and `RadiationModel` are **byte-identical** to `71b842a` (6,008 `float.hex()`
lines, MD5 `cb93a1a382c822a8077d025572faf5b1`). The QCD fixture was regenerated in this commit
(largest relative move: `rho_G` 7.475e-03). **Three tolerances had to loosen** — one more than
prompt §2 item 6 allows, so the prompt **stopped and asked**, and the user chose to land it: all
three are `[01-convergence-block-has-a-separate-generator]` and nothing else, and prompt 08 should
take them all back. A fourth assertion's **premise was falsified** and is the new
`[04-unsplit-tk-run-now-meets-the-criterion]` below. Suites: `CosmologyModels` 18,
`ComputeTargets` 354, `LiouvilleGreen` 143/143 fast set — none falls. Narrows
`[01-genericeos-tz-spline-floor]` on the `source-remediation` board. Full record:
[`logs/05-entropy-factor-representation.md`](logs/05-entropy-factor-representation.md).

**Prompt 06 landed `COMPLETE WITH DEVIATIONS` — and closes T1.** `_build_T_z_spline` now tabulates
the entropy factor **one spline per branch**, with the segment edges placed on the redshifts at
which $T(z)$ genuinely jumps and located by **bisecting the monotone $T(z)$** — never by
root-finding on $T(z)-T_{\rm break}$ (README §2 (b); log 06 enumerates every `root_scalar` in the
file and shows none of them is on the edge path). New in
`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`: `SegmentedEntropyFactor`,
`build_segmented_entropy_spline(...)`, `LambdaCDM_GenericEOS._entropy_factor_log1pz`,
`._entropy_segment_edges_log1pz` and `._bisect_temperature_crossing_log1pz`, plus
`SEGMENT_EDGE_PAD_LOG1PZ = 1e-12`. **`T_Z_REPRESENTATION_VERSION` is 4.** README §7 **D3**:
**3,000 nodes at $k=5$**, the audit's recommendation, measured against a segmented 2,000 / $k=5$
(which reaches the same median and the same bit-identical conformal time but misses the max by
four orders and the p90 by 4.5×) and a segmented 500 / $k=3$ (which misses everything, including
the T1 row). README §7 **D4**: **`break_temperatures_GeV`**, all four, three of them in range.

On the audit's 640-point probe set the representation goes **7.236e-04 / 8.912e-08 / 2.599e-10** →
**6.807e-11 / 3.237e-15 / 1.765e-16** (max / p90 / median), and $H(z)$ on the production grid
**1.280e-03 / 1.923e-07 / 5.430e-10** → **1.690e-10 / 6.276e-15 / 2.804e-16** — both the audit's §4
and §5 rows to the digit. **The T1 guard reads `0.0`**: $\int\mathrm{d}z/H$ from the shipped
background is *bit-identical* to the same integral from the exact background,
`1.3320002507788795e+03` at all 17 digits, and the equivalent phase is `0.000e+00` rad at all three
production wavenumbers. The segment edges, to 17 digits, are `17.565806941870026`,
`23.197460552819653` and `27.485391822044257` in $u$, each verified to be the crossing itself
($T(\text{edge})\ge T_{\rm break}$, $T(\text{edge}-1\,\text{ulp}) < T_{\rm break}$) and agreeing
with `T_z_reference.jump_locations` to 0, 1 and 0 ulp. An edge deliberately misplaced by **one
node** puts the full **7.054e-04** back inside the displaced window while leaving the p90 and the
median untouched — the silent failure, now a standing guard.

**`BREAK_POINT_ALL` is 2,414** on the production source grid (2,411 knots + 3 crossings), up from
407, with median spacing 0.67× the grid rather than 4.04×; `BREAK_POINT_DISCONTINUITY` is **2**.
That is the node count's price and it is **prompt 07's to remove** — it is also why
`COST_BREAK_POINT_FACTOR` had to go 1.30 → 2.40 (`[05-break-point-set-grew-with-the-node-count]`).
`LambdaCDM` and `RadiationModel` are **byte-identical** to `8d6e913` (7,014 `float.hex` lines, MD5
`78f633c20adcf05528e3141ea06d62a1`), and the single-segment code path reproduces prompt 05's
construction **bit for bit** on 2,001 of 2,001 probes at the same nodes and order. The QCD fixture
was regenerated in this commit (186.2 s; largest move `rho_G` at $k=3\times10^8$, **1.613e-01**
relative) and its `method` string now records that the reference is no longer circular. **Two of
prompt 05's three loosened tolerances were taken back to 3.0**; one new tolerance loosened
(`COST_BREAK_POINT_FACTOR`) and one moved again for the known staleness
(`QCD_BREAK_POINT_ALIGNMENT_TOL` → 1.5e-04). **`T_photon`'s cost per call is a possible miss** of
the ≤ 2.5 µs row and is `[06-t-photon-call-cost-needs-a-quiet-machine]`. Suites:
`CosmologyModels` 18 → **30**, `ComputeTargets` 354, `LiouvilleGreen` 143/143 fast set. Closes
`[01-genericeos-tz-spline-floor]` on the `source-remediation` board. Full record:
[`logs/06-segment-at-the-jumps.md`](logs/06-segment-at-the-jumps.md).

**Prompt 07 landed `COMPLETE WITH DEVIATIONS` — and closes G1.** `integration_break_points` now
returns the crossings of the equation of state's `break_temperatures_GeV` (`BREAK_POINT_ALL`) or of
its `discontinuity_temperatures_GeV` (`BREAK_POINT_DISCONTINUITY`) and **nothing else**: **3** and
**2** on the production source grid, from 2,414 and 2, with **0** of them a knot of any
interpolant. The three are `17.565806941870026`, `23.197460552819653`, `27.485391822044257` —
**bit-identical** to prompt 06's segment edges, because both now read one cache,
`_break_point_crossings_log1pz`, bisected once in `__init__` by the new
`_build_break_point_crossings_log1pz()`. `_temperature_crossing_log1pz`, whose `root_scalar`
brackets what is now a genuine step, is **off the production path** (README §2 (b), prompt §2
item 3's preferred resolution). **`T_Z_REPRESENTATION_VERSION` is 5.**

**The knots were dropped on measurement, and the measurement is the point** (prompt §2 item 1,
both halves). *The derivative:* an order-`k` interpolating spline through simple interior knots is
`C(k-1)` there, so at the shipped `k = 5` the first genuinely discontinuous derivative of $F$ is
the **fifth** (measured over all 2,405 interior knots in range: d1–d4 agree to 1.3e-13, 7.1e-12,
5.0e-10 and 6.2e-08 absolute, which is the noise of a one-sided Horner evaluation; d5 jumps by
26 %). The deepest derivative anything in the tree builds is `d3_lnH_dz3` — three levels below.
*The observable:* across a knot the step in $\mathrm{d}\ln H/\mathrm{d}u$ is **8.337e-11** at worst
and $|H/H_{\rm exact}-1|$ in a knot's neighbourhood **6.319e-12**, against **6.294e-07** and
**2.097e-04** for the 500-node `k = 3` lattice that used to be declared — 7,550× and 3.3e7×. That
2.1e-04 is the $10^{-4}$-level defect the old lattice carried, and there is nothing left at a knot
of this representation for a panel edge to protect against.

**The build gets cheaper and the tables do not lose accuracy.** QCD `BackgroundModel`:
tau/cs_tau/friction_F **16,580 → 6,936** integrand evaluations each, all three tables
**0.781 → 0.418 s**, whole `compute_background` **0.959 → 0.599 s**; the off-grid `delta` accessor
**39.48 → 29.23 µs** (on/off) and **77.97 → 57.66 µs** (off/off). QCD's 6,936 is now 0.17 % above
LambdaCDM's break-free 6,924 — three extra Gauss panels in 1,731 intervals. The `rho` residual
table costs **1.002×** the order×intervals baseline against 2.337× with the knots, so
`COST_BREAK_POINT_FACTOR` comes back **2.40 → 1.01**, below its original 1.30 rather than merely to
it. Scored against the (unchanged) references, `cs_tau` **2.212e-15** and `friction_F` **3.340e-16**
are unchanged to every digit printed and `tau` moves **2.104e-15 → 2.254e-15**, a 1.5e-16 absolute
move at 12 % of the 1.879e-14 floor it is scored against — **a narrow miss of prompt §4's "no
worse", recorded and not argued away**, and the evidence that the 404 splits were buying nothing.

**The reference fixture was regenerated and did not move**: 12 of 12 science keys bit-identical,
nothing written, 193.3 s — prompt §2 item 6's expectation that it would move is **falsified**, and
correctly so, because prompt 02's generator computes every reference by converged *adaptive*
quadrature and never consults `integration_break_points`. The stronger statement holds: `T_photon`,
`Hubble` and `rho` as exact `float.hex()` over 2,001 points to $z=10^{19}$ for `LambdaCDM`,
`RadiationModel`, `LambdaCDM_GenericEOS(PureRadiationEOS)` **and `QCD_Cosmology`** are
byte-identical to `a1d667a` (18,017 lines, MD5 `e9799e4299250b2ececb5b00a2a82514`) — **this commit
moves no background value at all**, only which points a quadrature splits at.

**`prompts/phase-representation` prompt 02's blocker is gone, measured.** A
multiplicity-`spline_order` knot vector **constructs on all six** of the (sector, $k$) geometries
that log measured as singular, and the same construction with the knot lattice restored still
raises `LinAlgError` on all six — both asserted in the tree by the new
`TestConsumerKnotVectorConstructs`, whose six bands are recovered from that log's sample counts and
validated against its own `DISC` column. Suites: `CosmologyModels` 30, `ComputeTargets`
354 → **359**, `LiouvilleGreen` 143/143 fast set. Closes
`[19-cosmologymodels-docstrings-predate-per-sector-policy]` on the `GkTk-remedial` board,
`[02-fixture-tests-pinned-to-todays-break-point-artefact]` and
`[05-break-point-set-grew-with-the-node-count]` here, and **narrows**
`[13-consumer-spline-crosses-eos-break-points]`. Full record:
[`logs/07-rederive-break-points.md`](logs/07-rederive-break-points.md).

**Prompt 08 landed `COMPLETE WITH DEVIATIONS` — and found state (a).** Prompt §1 named three
possible states of the world and the measurement picks the first: *the knots were a proxy for the
representation's own defect*. On the corrected background QCD $T_k$ converges at **all 50**
production wavenumbers under **either** policy — worst reference-convergence drift **7.08e-09**
under `BREAK_POINT_ALL` and **8.85e-09** under `BREAK_POINT_DISCONTINUITY`, median 1.96e-09 either
way, zero above the 3.4e-08 criterion in both — where `GkTk-remedial` prompt 19 measured 1.97e-07
and **three** offenders with the jumps alone. The three wavenumbers prompt 19's decision actually
turned on read **8.89e-10 / 2.03e-09 / 5.60e-10** with the jumps alone, against 6.1e-08 / 2.0e-07 /
3.5e-08 then — factors of 69, 98 and 63 with no change to the integrator. State (b) is excluded
(no wavenumber in either sector needs more than the cosmology declares, so G1's claim does not
narrow) and state (c) is excluded ($G_k$ moved **downwards**: worst 8.41e-09 → **3.67e-09**, median
2.05e-09 → 2.16e-10, and 6.0e-11 at prompt 19's own worst wavenumber). **Neither `BREAK_POINT_KIND`
was touched and `T_Z_REPRESENTATION_VERSION` stays at 5** — nothing in the commit moves a number.
The two models that declare nothing are **bit-identical between the policies at all 50 wavenumbers
in both sectors** and reproduce prompt 19's grid totals as exact integers (401,677 / 429,178 /
637,138 / 640,213), prompt 17's two control figures included (2.53e-06 in 7,403 evaluations;
2.56e-04 in 8,483). **The cost collapsed with the set prompt 07 removed:** a QCD $T_k$ object under
`BREAK_POINT_ALL` is **8,986** right-hand-side evaluations where prompt 19 measured **31,521**
(3.51×), and the wider policy now costs **+0.99 %** over the jumps alone rather than +220 %; $G_k$
is +0.57 % rather than +155 %. Wall time was taken on a machine that had been under load all day
and is reported as a *ratio* against a same-process null control (smooth-model rows 1.028 / 0.985 /
1.024 / 1.024; QCD 1.001 and 0.984) — no conclusion rests on it.

**Beyond what the prompt asked, one column decides more than the two policies do.** With the
cosmology's declaration suppressed *altogether*, QCD $T_k$ is above the criterion at **19 of 50**
wavenumbers, worst **9.61e-06** — so splitting at the equation of state's genuine **jumps** is
emphatically still load-bearing and prompt 18's result stands; only the distinction between the two
*kinds* has become vestigial. That narrows rather than closes
`[04-unsplit-tk-run-now-meets-the-criterion]`, whose single wavenumber is one of the 31 that now
pass. **README §7 D5 is reported, not decided** (the prompt's instruction): keeping
`BREAK_POINT_ALL` costs +0.99 %, changing it would move every stored QCD $T_k$ value by up to
**2.86e-04** of the envelope and demand a regeneration, so log 08 recommends leaving both constants
where they are and leaves the decision with the user. `docs/gktk-remedial/tk_numeric_atol_sweep.py`
was **copied, not edited**; the reduction is `docs/qcd-background-audit/per_sector_policy_remeasure.py`
and its output `docs/qcd-background-audit/PER-SECTOR-POLICY.md` (one command, 744 s, no Ray, no
datastore). Suites: `CosmologyModels` 30, `ComputeTargets` 359, `LiouvilleGreen` 143/143 fast set —
none falls. Full record: [`logs/08-per-sector-policy-remeasure.md`](logs/08-per-sector-policy-remeasure.md).

**Prompt 09 landed `COMPLETE WITH DEVIATIONS` — and the ungated chain is closed.** No production
file is in the diff; `T_Z_REPRESENTATION_VERSION` is **5** before and after. Three deliverables:
`docs/qcd-background-verification.md` (new, the campaign's verification document, additive
thereafter), `docs/qcd-background-audit/consumer_break_point_profile.py` (new), and a dated **§9
appended to `docs/gktk-remedial-verification.md`** — **121 insertions, 0 deletions**, nothing at or
above §8 touched.

**Every LambdaCDM number is bit-identical between the campaign base and this commit**, in all six
§3.5 rows, all six §3.6 `theta_deriv` column groups (decay ladders included), all six §3.7 rows and
both producer tables; every LambdaCDM line in the 274-line `verify_production_path.py` diff is a
wall clock. `LambdaCDM(Planck2018)` (5,005 `float.hex` lines, MD5
`157b1c61436106ba52058ddf3533d2df`) and `RadiationModel` (2,002 lines, MD5
`265454eb751f9503a965f204e9e68bf4`) are **byte-identical to `2a5e0fa`** across the whole
nine-commit span. `LambdaCDM_GenericEOS(PureRadiationEOS)` is *not*, and must not be: README §2 (g)
makes the new representation **exact** on a constant-$g_s$ equation of state, and against the closed
form it goes 2.270323e-07 / 1.904049e-07 / 1.006789e-07 → **3.330669e-16 / 2.220446e-16 / 0.0**.

**The consumers: six LambdaCDM rows bit-identical, four QCD rows unchanged in value, one 1,413×
better, and two 4.3× worse.** README §6.4 says the consumer numbers are not required to improve —
§3.5 and §3.6 score a consumer against a producer built from the same background, so the error this
campaign removed cancels in them — but it also says "nothing got worse", and **that clause is
missed**. §3.5's QCD $k=10^5$ rows go 1.9073e-06 → **8.1062e-06** rad ($G_k$, 8.00 → 34.00 ulp) and
3.1859e-06 → **1.3982e-05** ($T_k$, 427.60 → 1876.61 ulp), with §3.6's two rows at the same $k$
2.00× and 2.79× worse. **All four are at the `T_LO` crossing and the cause is measured**, by
`consumer_break_point_profile.py` on both trees: the pre-campaign 500-node order-3 spline had a
knot spacing 4.04× the production grid, smeared the equation of state's genuine step over ~4 grid
intervals and left only **25.55 %** of the total deviation in $\mathrm{d}\ln H/\mathrm{d}u$ inside
the crossing's own interval; the corrected background delivers a step **2.78× taller** with
**99.84 %** of it in that one interval. The background is right; what rose is
`[13-consumer-spline-crosses-eos-break-points]` seen undiluted, and it is **prompt 10's**. Recorded
as a miss, not argued away and not repaired (prompt 09 may not touch production code).

**The one row that says the campaign worked, on the consumer side.** QCD $T_k$ `theta_deriv` at
$k=10^7$ goes 4.8896e-07 / 4.8896e-07 / 4.3292e-07 → **1.8853e-08 / 3.4112e-10 / 3.0630e-10**
(max / `[3:-3]` / deep interior), and *what it became* matters more than the 1,413×: its last five
samples at the hand-over are now a clean geometric ladder rising ~3.8× per sample inwards
(`8.63e-11 3.41e-10 1.29e-09 4.94e-09 1.89e-08`), matching LambdaCDM at the same $k$ to within 20 %,
where before they were flat and the size of the interior. The derivative of a producer–consumer
difference is **not** common mode in the background's *smoothness*, which is why this one row could
see what §3.5 could not.

**Producer-side, everything improved.** All six QCD $\theta_G$ / $\theta_T$ rows fell to their
floors — all six were above the script's printed floor at the base, by 1.07× to 5.94×, and at
`HEAD` three are below it and three within 11 % of it; QCD `tau` 2.104e-14 →
**2.254e-15** and `cs_tau` 2.108e-14 → **2.212e-15**, from above their references' 1.88e-14 floor to
an order below it; QCD per-object build costs $G_k$ **8,380 → 6,892** and $T_k$ **12,896 → 11,532**
integrand evaluations with every LambdaCDM and every cached count exactly unchanged; the QCD
off-grid `raw_theta` accessor needs **4.00** evaluations per call where it needed 4.27. §3.7 is
unchanged, all zero. The audit script's **§1 and §2 — the equation of state — are
character-for-character identical**, which is README §0.5's boundary held.

**`T_photon` is a confirmed miss**, no longer an open question: **2.596 µs mean, range 2.505–2.671
over five runs on a quiet machine**, against README §6.2's ≤ 2.5 µs — about 3.8 % over, all of it
the order-5 spline evaluation (the segment dispatch is free, 1.00–1.05×), and order 5 is not
optional because a cubic needs ~25,000 nodes to reach the required p90.
`[06-t-photon-call-cost-needs-a-quiet-machine]` is narrowed to that statement, with
`[07-t-photon-range-logic-recomputes-its-bounds]`'s measured 0.11 µs as the one cheap saving
available. Suites: `CosmologyModels` 11 → **30**, `ComputeTargets` 339 → **359**, `LiouvilleGreen`
148 → **148** on the **full** set (not the fast set prompts 02–08 used) — none falls. Full record:
[`logs/09-close-out-verification.md`](logs/09-close-out-verification.md).

**Prompt 10 landed `COMPLETE WITH DEVIATIONS` — and the answer is "not the knots".** No production
file is in the diff; `T_Z_REPRESENTATION_VERSION` is **5** before and after, and `num_chunks` is
still 1. Prompt §1 said the first job is not to build anything but to re-take
`prompts/phase-representation` prompt 02's measurement on the corrected background, and the
measurement decides the prompt. Two deliverables plus tests:
`docs/qcd-background-audit/consumer_knot_scheme_scan.py` (new; reproduces
`verify_production_path.py`'s `consumers` geometry for both models, both sectors and all three
wavenumbers, scores **nine knot schemes plus two controls** against one set of producer runs, one
command, ~120 s, no Ray, no datastore) and a dated **§8 appended to
`docs/qcd-background-verification.md`**; plus
`ComputeTargets/tests/test_primitive_phase.py::TestBreakPointKnotsBuyWhatTheSamplesResolve`
(2 tests, 0.007 s).

**No scheme recovers the two rows prompt 09 recorded as a miss, and the two the prompt names make
them worse.** At QCD $k=10^5$, against 8.1062e-06 rad (34.00 ulp, $G_k$) and 1.3982e-05 rad
(1876.61 ulp, $T_k$): the **repeated multiplicity-`spline_order` ($C^0$) knot vector** — prompt §2's
default and the issue's own "next step" — gives 1.6928e-05 (71.00) and 2.9315e-05 (3934.6), **2.09×
and 2.10× worse**; **per-segment splines** give 4.0531e-05 (170.0) and 7.0770e-05 (9498.6), **5.00×
and 5.06× worse**; the best of four non-$C^0$ controls is 6.6757e-06 (28.00) and 1.1528e-05
(1547.2), **1.21× better** against the 4.25× and 4.39× that would have to be recovered and 3.5×
above the campaign base. **All six LambdaCDM rows are identical under every scheme**, and — new
since prompt 02, and prompt 07's doing — **the ten rows at 1.00 ulp stay there under every scheme**,
`ALLx1` included, because `BREAK_POINT_ALL` is now 1–3 points on those grids rather than 226–325.
So the family was scored without prompt 02's trap and still nothing reaches the target. **Prompt §2
item 1's "measure which `kind` serves $\varphi$ better; do not assume" is answered *neither***:
`DISCx3` and `ALLx3` agree to every printed digit on every row, as do the segment and multiplicity-1
pairs.

**Prompt 02's kink fit, re-taken as the `GkTk-remedial` board required, says why.** One-sided cubics
on $\varphi$ from the dense reference either side of `T_LO`, over windows of 1, 2 and 3 grid
intervals, give $[\varphi'] = +7.5232$e-05, $-3.0026$e-03, $-6.1529$e-03 ($G_k$) and $-2.1692$e-04,
$+5.1925$e-03, $+1.0634$e-02 ($T_k$). **A genuine slope discontinuity gives a window-independent
jump; this moves two orders and changes sign** — the signature of smooth-but-*unresolved* data —
on a background whose $T(z)$ error has fallen 7.18e-04 → 6.807e-11. There is no resolved corner for
a $C^0$ knot to turn.

**What does work is samples, and the number is small.** A plain cubic of the same $\varphi$, default
knots and no break-point treatment at all: refining the **±5 grid intervals around the crossing by
2×** — **10 extra samples in 1,016, 1.0 %** — takes the two rows to **1.64 ulp** (3.9121e-07 rad)
and **74.60 ulp** (5.5584e-07 rad), both inside prompt §5's 1e-06 rad target and $G_k$ inside its
2-ulp one; a uniform 2× gives 1.48 and 72.76 ulp. Refining the crossing's *own* interval alone buys
1.96× and stalls, so **the feature is three to five grid intervals wide and a grid design that
protects only the crossing will not work** — which prompt 11 needs before it designs anything. At
$k=10^7$ and $3\times10^8$ the same ladder reads **0.00 ulp near the break at every density**, in
both sectors: the crossing is a $k=10^5$ phenomenon.

**The `theta_deriv` residue, split and attributed.** The break points' share is **35.8 %** ($G_k$)
and **35.5 %** ($T_k$) of the base figure, at $k=10^5$ only, and costs 2.09×/2.10× of consumer phase
to buy — prompt 02 measured 38 % at the same price on the defective background, so that half of its
finding is unchanged. `[02-consumer-phi-below-the-storage-granularity]`'s share is **100 %** of the
two QCD $G_k$ rows that miss $10^{-6}$ away from $k=10^5$: there the recovered $\varphi$ spans
**6.0** and **2.0 ulp** of the stored phase, no scheme moves either figure by a printed digit, and
removing $\varphi'$ altogether *improves* them by **1.53×** and **15.9×**. And QCD $T_k$ at
$3\times10^8$ is a **1.70× regression** under both $C^0$ schemes — prompt 02's warning that a
one-number acceptance test would ship `DISC × 3`, firing in a row it did not name.

**Cost:** unchanged, because nothing changed; measured for the record at load average 12.28, best of
9 — `PrimitivePhase(...)` builds in 3.18e-04 s (LambdaCDM, 1,361 samples) and 3.38e-04 s (QCD,
1,377) with **0 integrand evaluations**, against prompt §3's 0.0010 s figure, which is
`gktk-remedial-verification` §3.9's cost of a whole *producer* object rather than of this build.
**Closes `[13-consumer-spline-crosses-eos-break-points]`** on the `GkTk-remedial` board §4 and opens
`[10-consumer-phi-unresolved-at-the-eos-crossing]` in §3 below, assigned to prompt 11 (deviation 2
of the log argues that choice and says how to reverse it). Suites: `CosmologyModels` 30,
`ComputeTargets` 359 → **361**, `LiouvilleGreen` 143/143 fast set — none falls. Full record:
[`logs/10-primitive-phase-break-points.md`](logs/10-primitive-phase-break-points.md).

**Prompt 11 landed `COMPLETE WITH DEVIATIONS` — and the source grid now knows the cosmology.**
`T_Z_REPRESENTATION_VERSION` is **5** before and after; no `CosmologyModels/` file and no compute
target is in the diff, and **no background value moves** — only which redshifts are sampled. Three
production files: `CosmologyConcepts/wavenumber.py` (`build_z_sample`, `SourceGrid`,
`populate_source_grid`, and the four constants), `CosmologyConcepts/redshift.py`
(`winnow(sparseness, protect=...)`, `redshift_grid_digest`) and `main.py`'s grid-construction and
tag hunks alone, plus two new module-level helpers there (`cosmology_feature_redshifts`,
`build_grid_tag_labels`) so that `load_main_py_functions` can **execute** them rather than assert
against their text.

**The production grid.** QCD 1,732 → **1,773** samples (+41, **+2.37 %**): per declared crossing, a
**pair straddling it** at a quarter of a grid interval plus the **eleven intervals from −5 to +5
refined by 2**, and the two equality redshifts. No base sample is displaced, the grid is strictly
descending, and its closest approach between neighbouring samples is **1.028e-03** relative in
$(1+z)$ — four orders above the 1e-07 at which the datastore would treat two samples as one row.
**LambdaCDM's grid is bit-identical, element for element** (1,732, and its response grid 145): the
whole cosmology-aware path is gated in `main.py` on the cosmology declaring some non-smoothness, so
every LambdaCDM model, `RadiationModel` and stand-in takes the unchanged code path — README §0.5 and
§2 (g), a stop condition rather than an expectation.

**The two rows prompt 09 recorded as a miss are recovered, and the prompt's own remedy is not what
recovers them.** At QCD $k=10^5$, against 8.1062e-06 rad (34.11 ulp, $G_k$) and 1.3982e-05 rad
(1876.61 ulp, $T_k$): the shipped grid gives **3.8296e-07 rad (1.61 ulp)** and **4.9012e-07 rad
(65.78 ulp)** — **21.2× and 28.5×**, both inside prompt 10 §5's 1e-06 rad target, $G_k$ inside its
2-ulp one, and both better than the campaign base (1.9073e-06 / 3.1859e-06). **The straddling pair
alone buys 1.96× and 1.98×** — 17.43 and 949.24 ulp, still far outside the target — so prompt §2
item 2 is necessary and nowhere near sufficient, and what carries the rows is prompt 10's ±5 × 2
neighbourhood (1.64 / 74.60 on its own). The pair adds a further 2 % and 12 % on top of it. All
three wavenumbers were scored, in both sectors, because that is the trap prompts 02 and 10 both
warn about: $k=10^7$ and $3\times10^8$ are unmoved (QCD $T_k$ at $10^7$ improves 0.15 → **0.01 ulp**
near the break), with **one row moving the wrong way** — QCD $T_k$ at $3\times10^8$, 3.8674e-05 →
3.8999e-05 rad, **+0.84 %**, 1.27 → 1.28 ulp of the span, not at the crossing and at the floor ten
of the twelve rows already sit at. Recorded, not argued away.

**The standoff was scored, not assumed, and it is a fraction of the grid spacing.**
`SOURCE_GRID_BREAK_STANDOFF = 0.25` of a base interval (5.82e-03 relative in $(1+z)$ on the
production grid) is the only value in a scan from 1/2 down to 1e-4 of an interval that beats the
no-pair column on **both** rows. Below ~1/32 the pair stops helping and saturates **1.5×–1.7× worse
than no pair at all**, because a pair a distance $d$ apart implies a slope carrying
$\sim2\,\mathrm{ulp}/d$ of `[02-consumer-phi-below-the-storage-granularity]`'s storage granularity.
**That is why `numeric_with_phase_cut.BREAK_POINT_STANDOFF = 1e-12` must not be borrowed here** —
it places an ODE restart, where nothing is interpolated and nothing is stored — and `build_z_sample`
**refuses** a standoff below the datastore's redshift resolution rather than silently making the two
samples one row.

**The grid tag now identifies the grid, and that invalidates a datastore.**
`SourceRedshiftGrid_{len}` labelled size alone; it is now `SourceRedshiftGrid_{len}_{digest}`
(`blake2b` over the grid's exact bits). Eight `pool.object_get` call sites across **seven stored
object types** filter on the grid tags — `TkNumericIntegration`, `TkWKBIntegration`, `QuadSource`,
`GkNumericIntegration`, `GkWKBIntegration`, `GkSource`, `QuadSourceIntegral` — and their factories
treat the sample grid as *"a target rather than a selection criterion"*, so the tag is the **only**
record of which grid an object was computed on; `GkSourcePolicyData` goes with them transitively,
making eight. From the only measured production-shaped run in the tree
(`docs/gktk-remedial-verification.md` §4.2, one model, 5×5 wavenumbers, 1,584 nodes): **15,020
objects, 6 m 35 s plus a `QuadSourceIntegral` stage that did not finish 3,300 objects in three
hours**. Production is ×10 in each wavenumber sample and two models (×85 on `QuadSource`, of order
×850 on `QuadSourceIntegral`) — an extrapolation, labelled as one. **The QCD half was invalidated
anyway** (its grid length changed, and prompt 03's key had already invalidated its cosmology row);
**the LambdaCDM half is invalidated by the tag change alone**, and because `store_tag` is keyed on
its label a LambdaCDM-only datastore can be carried across by relabelling two rows instead of
recomputing — log 11 §5 gives the SQL and the reason it must not be run on a store that also holds
pre-prompt-11 QCD objects.

`[03-derivative-pad-clamp-on-coarse-grids]` was **measured and does not bind**: `h_lo` is the same
float (2.115878e-03) on both grids, the lowest protected sample is 48.19 base intervals above
$z_{\rm end}$, and the bottom 40 samples are element-for-element the base grid's. Nothing was
changed there. Suites: `CosmologyModels` 30, `ComputeTargets` 361 → **380**, `LiouvilleGreen`
143/143 fast set — none falls. **Closes `[10-consumer-phi-unresolved-at-the-eos-crossing]`** (§4)
and opens `[11-background-model-not-keyed-on-the-source-grid]` (§3). Full record:
[`logs/11-cosmology-aware-source-grid.md`](logs/11-cosmology-aware-source-grid.md).

**Prompt 12 landed `COMPLETE WITH DEVIATIONS` — it measures and recommends, and changes nothing.**
No production file is in the diff; `T_Z_REPRESENTATION_VERSION` is **5** before and after, no test
and no fixture moved, and no grid was changed. Two deliverables:
`docs/qcd-background-audit/grid_density_criterion.py` (new; one command, **88.4 s**, no Ray and no
datastore, run twice with every printed figure **bit-identical** between the runs) and a dated
**§10 appended to `docs/qcd-background-verification.md`**.

**The oracle had to change, and that is the prompt's one `STRUCTURALLY REQUIRED` deviation.** The
error this prompt has to resolve runs from 5.9e-08 down to 7e-20 rad, and $\varphi$ recovered from
a *stored* $\theta$ — what prompts 10 and 11 scored — carries
`[02-consumer-phi-below-the-storage-granularity]`'s 3.05e-07 rad floor at $k=10^5$. Ten of the
twelve rows would have read "noise". So the reference is the phase residual $\rho$ itself, from
`build_phase_residual`'s `CumulativeTable`, whose `delta` does genuine Gauss quadrature between
arbitrary endpoints; `PrimitivePhase`'s docstring is the warrant, since $\varphi$ is $\rho$ plus
constants and a constant is invisible to an interpolation error. **It reproduces prompt 11's own
number on that completely different path** — the configuration prompt 11 measured reads
1.4055e-05 → 3.9744e-07 here against its 1.3982e-05 → 4.9012e-07, 0.5 % at the start and 23 % at
the end.

**Sub-question 1: the grid is wrong in both directions, and the miss and the waste are in the same
band.** Two of the twelve production rows miss the storage floor and they are the same row on both
models — $T_k$ at $k=10^5$, **7.86 ulp** (LambdaCDM, 5.8576e-08 rad against 7.4506e-09) and **7.84
ulp** (QCD, 5.8437e-08) — with the maximum in the **top decade of the band**, a few grid intervals
inside horizon entry. The other ten are inside, the smallest at **1.87e-12** of its floor. Per
decade the error falls about an order per decade downwards and spans **eight orders inside one
band**, while $h_u$ is constant to four digits over fourteen decades and then *falls* to 5.49e-03
below $z=1$ because `populate_z_sample` is log-spaced in $z$ rather than in $1+z$ — the density is
highest exactly where the curvature is lowest. On LambdaCDM $G_k$ the headroom is **8.6e+04 to
2.1e+19** and **no density is doing any work at all**.

**Sub-question 2: the criterion, and it is pre-grid.**
$h(u)^4|\varphi''''(u)|/384 \le \varepsilon$ with
$\varphi'(u) = -(1+z)\,C(z)/(\omega+\omega_0)$ — the residual integrand, which needs $H$, $c_s^2$,
their $z$-derivatives and $k$ and nothing else. On QCD those derivatives are not cosmology methods,
so a pre-grid criterion must finite-difference the cosmology's pointwise `Hubble`; that agrees with
the grid-splined ones to **3.9e-09 relative away from a declared crossing** (median 8.1e-10), which
is the licence. Cost $5N$ closed-form evaluations, **0.01–0.27 s** against `compute_background`'s
0.599 s. **It predicts the realised error to ±2 %**: in the $T_k$ sector, both models, all three
wavenumbers, median 0.923–0.928 with a p10-to-p90 spread of **1.0×** over 517–551 intervals, using
the textbook $1/384$ and no fitting. QCD $G_k$ median 1.02–1.25, p90 3.2–10.4; LambdaCDM $G_k$
over-predicts 100×, which is the five-point stencil's roundoff where $\varphi'\sim10^{-8}$ and is
conservative.

**Sub-question 3: one universal envelope grid, costed over a cap ladder.** QCD 1,773 → **1,761** at
"never coarser than today" with every row at **0.14×** its target, or **1,015** (**1.75× fewer**) at
twice today's spacing with every row at 0.21×; LambdaCDM 1,732 → **1,634** (0.34×) or **842**
(**2.06× fewer**, one row 9 % over). At the **shipped** sample count the criterion reads 2.1427e-12,
1.0849e-11 and 9.7801e-11 rad where the grid reads 5.8576e-08, 2.4859e-05 and 4.4783e-04 — 2.7e+04
to 4.6e+06. Uniformly, `source_samples_per_log10z` would have to go 100 → about **167** to meet the
same floor, against 204 and 295 band samples under the criterion. **The second candidate is
refuted**: equidistributing $\varphi$ itself needs **114,281** samples (QCD) and 4,229 (LambdaCDM)
and still misses by up to 1,723×.

**Three things bound all of it, and they are stated where the recommendation is, not in a
footnote.** (i) **The saving and `[03-derivative-pad-clamp-on-coarse-grids]` are the same lever**:
the whole saving is coarsening at low $z$, and the derivative-fit padding clamps the moment the
lowest interval passes $-\log(0.9)/12 = 8.7800\text{e-}03$ in $u$ — measured as *not binding* on
both shipped grids and on both cap-1× criterion grids (7.6773e-03) and as **binding** on every
coarser one. (ii) The source grid also carries the numeric ODE's samples, four `CumulativeTable`s'
Gauss panels and `QuadSourceIntegral`'s abscissae, **none of whose requirements is measured here**,
so a $\varphi$-only criterion is a lower bound on the density and never an upper one — which is
why the answer is a ladder and not a number. (iii) `docs/OPEN_ISSUES.md` §5: **no pipeline has been
run on any grid in this section, including the one that ships.** (iv) Above $k\approx10^7$ nothing
here is visible: the $T_k$ away/floor column reads 7.84 at $k=10^5$, **1.62e-02** at $10^7$ and
**1.92e-03** at $3\times10^8$, because the floor grows like $k$ while $\varphi$ falls like $1/k$ —
the same fact prompt 10 measured from the other side (6.0 and 2.0 ulp of span).

**The $k\tau$ oscillation, written down so audit §9's bullet is not lost.** $k\tau$ over the whole
range is 1.3728e+09 / 1.3728e+11 / 4.1184e+12 rad; the **median** response interval advances
**185–296 rad at $k=10^5$** — 30 to 47 complete cycles — and four samples per cycle would need
8.740e+08 to 2.622e+12 response samples, a shortfall of **5.9e+06× to 1.8e+10×**. **The conclusion
is not a grid size but that no grid size exists**: an $\Omega_{\rm GW}$ post-processing step must
get its RMS amplitude from an oscillatory quadrature, never from sampling the response grid more
finely. The response grid stays a subset of the source grid under a blind stride on all twenty
candidate grids.

**And one finding the prompt did not ask for.** Building the QCD background on the grid prompt 11
ships — which is what `main.py` does — takes the consumer's $\varphi$ error at the `T_LO` crossing
to **2.4859e-05 rad** where prompt 11's configuration (background on the *base* grid, samples from
the new one) reads **3.9744e-07**: **1.77× worse than the base grid**, not 35× better. `epsilon` on
QCD is a stacked spline of $\log H$ over a refinement of the source grid, that lattice is **not**
split at the equation of state's break points, and refining it at a genuine step makes the ringing
**narrower faster than it makes it smaller** (the amplitude does improve, 3.66e-02 → 2.04e-02 at
`T_LO` and 6.48e-03 → 1.03e-03 at `T_120_MEV`; `EOS_T_LO`, the kink where $g_s$ is continuous, is
the control and reads 1.6e-09 on both). **Away from a crossing prompt 11's grid is better on every
one of the six QCD rows.** Opened as `[12-background-derivative-fit-grid-rings-at-a-step]`. Suites:
`CosmologyModels` 30, `ComputeTargets` 380, `LiouvilleGreen` 143/143 fast set — none falls, and
none could, since nothing outside `docs/` and `prompts/` is in the diff. Full record:
[`logs/12-grid-density-criterion.md`](logs/12-grid-density-criterion.md).

**Prompt 13 landed `COMPLETE WITH DEVIATIONS` — and the last smooth interpolant across a declared
step is gone.** `BackgroundModel` splined background quantities at **two** sites and both ran
straight across the equation of state's crossings: `compute_background._build_derivative`'s order-5
fit of $\log H$ over a padded, 3× refined copy of the source grid, and
`_create_functions._build_func`'s cubic through the *stored* samples. Both now fit **one spline per
branch**, separated by `_cosmology_break_points` and dispatched by `bisect_right` on $u$ — never on
a recovered $z$ — with a degenerate branch **refused** rather than silently dropped to a lower order
or fitted across the step after all. New in `ComputeTargets/BackgroundModel.py`: `SegmentedSpline`,
`_segment_slices`, `_refuse_degenerate_segment`, `build_stored_sample_spline` and
`STORED_SAMPLE_SPLINE_ORDER = 3`. **`T_Z_REPRESENTATION_VERSION` is 6.**

**The ringing is gone at both genuine steps and the control did not move by a bit.** Scored the way
`docs/qcd-background-verification.md` §10.4 scored it — `epsilon` from the model against a central
difference of the cosmology's own pointwise `Hubble`, at eight offsets either side of each crossing
— on the production 1,773-sample grid: `T_LO` **2.036e-02 → 1.931e-09**, `T_120_MEV`
**1.034e-03 → 4.070e-09**, both inside the **3.907e-09 max / 8.091e-10 median** that holds away from
any crossing, and the `EOS_T_LO` control **1.609e-09 → 1.609e-09**, the same float. **Neither site
is redundant**: segmenting the fit alone carries `T_LO` the whole way (2.0e-02 → 5.0e-09, because
$\epsilon$ is flat at 2.0 there) but leaves **1.91e-04** at `T_120_MEV`, where the crossing sits in
the QCD crossover and $\epsilon$ runs from 1.9460 to 1.9101 over 0.1 in $u$; only segmenting the
stored cubic too closes that, a further **47,000×**.

**`BREAK_POINT_ALL` is load-bearing, and the row that proves it is the control's own.** Scored
against `BREAK_POINT_DISCONTINUITY`'s two points, `d_wPerturbations_dz` at `EOS_T_LO` is **346 %
wrong** — 3.459 relative — under both "no segmentation" and "jumps only", and **7.752e-04** under
all three crossings: a factor of **4,462**. `EOS_T_LO` is exactly where the equation of state clamps
$w$ (README §7 D4) and $c_s^2$ is `wPerturbations`, so the third crossing earns its place for the
quantity it was declared for, at no cost in $\epsilon$ on the production grid.
`d_wPerturbations_dz` was also **8.2 %** and **3.7 %** wrong at the two genuine steps and is now
1.1e-06 and 3.5e-07 — a second defect this commit closes, which nothing in the campaign had
measured.

**The four-way table, and the acceptance.** `docs/qcd-background-audit/grid_density_criterion.py`
reused **unedited**. The production configuration — background built on the shipped grid, samples
from the shipped grid — goes **2.4859e-05 → 5.8437e-08 rad** at QCD $T_k$, $k=10^5$ (**425×**),
against prompt 13 §4's ≤ 3.9744e-07, and improves at every (sector, $k$) scored: $G_k$ $10^5$
1.4356e-05 → 1.6782e-08 (855×), $T_k$ $10^7$ 4.4783e-04 → 5.6892e-05 (7.9×), $G_k$ $10^7$
2.1574e-04 → 2.4203e-05 (8.9×). **The background's grid has stopped mattering**, which is the
identity that says the defect is gone rather than merely smaller: the four cells at $T_k$, $k=10^5$
were 1.41e-05 / 3.97e-07 / 1.19e-04 / 2.49e-05, a 300× spread set by which grid the background was
built on, and are now 5.905e-08 / 5.844e-08 / 5.905e-08 / 5.844e-08. **And the maximum has left the
crossing**: it now sits at $z=1.44\times10^{10}$, away from any, and is
`[12-source-grid-density-…]`'s 7.84-ulp row — prompt 15's, not this one's.

**Nothing that declares nothing moved.** `LambdaCDM(Planck2018)`, **`LambdaCDM` with its five
analytic derivatives hidden** (the stand-in that drives the spline branch with no break points),
`LambdaCDM_GenericEOS(PureRadiationEOS)` and `RadiationModel` are **byte-identical** to `21d80b2`
across 101,110 `float.hex()` lines — the whole `compute_background` payload plus nine derived
accessors at 1,301 off-grid probes — and only `QCD_Cosmology`'s 28,933 lines differ. Integrand
evaluations are **exactly unchanged** (7,100 per cumulative table on the 1,773-sample grid, both
before and after; §4's 6,936 is prompt 07's figure on the 1,732-sample base grid) and
`compute_background`'s wall time moved −1.3 %. The QCD fixture was regenerated in this commit
(197.6 s): `rho_G` moves **7.278e-02** relative at $k=3\times10^8$ and `rho_T` **2.086e-03**, while
`tau`, `cs_tau` and `friction_F` do **not move by a bit** — the check on the change's scope, since
those are quadratures that never see a derivative spline. **No tolerance was loosened**; the count
is zero.

**One deviation is `STRUCTURALLY REQUIRED` and is worth reading.**
`ComputeTargets/tests/wkb_reference.py` carried a **third** copy of the stored-sample site, in the
function whose own docstring undertakes to reproduce it — and every `QCDModel` in the tree, the one
this prompt's acceptance tool builds included, goes through it. The three lines were lifted into
`build_stored_sample_spline` and both callers now call it; without that the harness scored the
repaired background as a partial failure (1.91e-04 against 4.07e-09). Two issues opened: segmenting
**costs** 43× at the control crossing on a grid that does not resolve it (1.889e-09 → 8.190e-08 on
the base grid), which prompt 15 must pay for if it coarsens; and the crossing neighbourhood prompt
11 sized at $k=10^5$ leaves **59.7 ulp** at $k=10^7$, which prompt 10 could not see because the
stored-$\theta$ floor there is 3.05e-05 rad. New module
`ComputeTargets/tests/test_background_segmentation.py` (12 tests, 0.21 s). Suites:
`CosmologyModels` 30, `ComputeTargets` 380 → **392**, `LiouvilleGreen` 148 on the **full** set —
none falls. **Closes `[12-background-derivative-fit-grid-rings-at-a-step]`** (§4). Full record:
[`logs/13-segment-background-derivative-splines.md`](logs/13-segment-background-derivative-splines.md).

**Prompt 14 landed `COMPLETE WITH DEVIATIONS` — and a run now has a name.** No background number
moves; `T_Z_REPRESENTATION_VERSION` is **6** before and after, and the production source grid is
**byte-identical** to prompt 11's — 3,814 `float.hex()` lines over both models' source, response and
protected sets, MD5 `d5ecc0aa85f38578d8c57c051d3f11e0`, identical at `HEAD` and in a detached
worktree at `9d57ff3`, with all four of log 11 §5's tag labels reproduced to the character (QCD
1,773 / `303f9ce7` / 156 / `197b46de`; LambdaCDM 1,732 / `0960e169` / 145 / `69050b4c`). **Ten**
production files: `CosmologyConcepts/wavenumber.py` (the new constant alone),
`Datastore/SQL/ObjectFactories/BackgroundModel.py`, `main.py`'s argument, tag and tagging hunks,
`extract_common.py` and all six `extract_*.py` — **the exclusion on `extract_*.py` was lifted for
this prompt only**, by the user, and remains in force for prompt 15.

**A human-readable handle plus a derived check, which is the campaign's house style one level out.**
`main.py --run-label` names a run; the name becomes a `store_tag` (`Run_<label>`) carried by, and
**filtered on for**, everything the run writes — `TkProductionTag`'s mechanism, as prompt §2 item 1
required, at the `BackgroundModel` lookup and the **26** further tag lists that name a
production tag.
`SOURCE_GRID_CONSTRUCTION_VERSION = 1` (`CosmologyConcepts/wavenumber.py`) names the *algorithm*
that built the grid and travels beside it as `SourceGridConstruction_1`; prompt 11's content digest
says which exact grid, and the two are not substitutable — a digest cannot be inverted or
range-queried, and two cosmologies in one run legitimately have different grids at the same
construction. `sqla_BackgroundModelFactory` gains `source_grid_digest` and
`source_grid_construction` columns, both read from their single declarations and **never written as
literals** (asserted by `ast`, prompt 03's test of the same name), so
`[11-background-model-not-keyed-on-the-source-grid]` — the one surviving row in a store every
compute target of which prompt 11 had already invalidated — is closed.

**The two paths give different answers to a pre-prompt-14 datastore, and that is the prompt's
archival requirement rather than an inconsistency.** The compute path supplies `z_sample`, filters
on both columns, and on a store that lacks them raises a `RuntimeError` naming this campaign,
"regenerated" and "no migration", with SQLite's `OperationalError` preserved as `__cause__`. The
read path supplies `z_sample=None` — it is asking which grid was used, not asserting one — selects
the two columns instead, falls back to the pre-prompt-14 query when they are absent, and reports
the rows as an **unknown generation** that stays queryable and plottable. Where the tags it was
given match more than one generation it **refuses, naming every `(construction, digest)` it
found** and pointing at `--run-label`; it never picks. All six scripts select on the label, and the
single-run ergonomics survive: no label plus exactly one run in the store uses it and says which,
and only an ambiguous store requires the user to choose.

**Three choices a later reader may want to disagree with, all argued in the log.** The default run
label is the fixed string `"default"` rather than a generated unique one, because the label is a
*selection criterion*: a label that changed per run would make every unlabelled run miss everything
and pay log 11 §5's 15,020-object bill. Verification is "one generation **per lookup**" rather than
"one digest per label", because a production run holds two models whose grids legitimately differ.
And it lives in the factory rather than in `extract_common.py`, because only the factory can see a
column. **`source_samples_per_log10z` was traced rather than assumed** (prompt §2 item 5): it feeds
the grid density, `SourceSamplesPerLog10ZTag`, and `delta_logz` at `main.py:916` and `:1533` — and
that third is **vestigial**, accepted by `numeric_with_phase_cut` only "so that callers need not
change" and consumed by nothing since `GkTk-remedial` prompt 11, so prompt 15 can retire the tag
without a numerical consequence. **No extract script was executed**: each needs a populated
datastore, so their query construction is asserted by `ast` and the read path itself is exercised
against real SQLite. One new module,
`ComputeTargets/tests/test_run_identity.py` (30 tests, 0.3 s), which runs the production `store()`,
`validate()` and `build()` against an in-memory database built from the factories' own
`register()`. Suites: `CosmologyModels` 30, `ComputeTargets` 392 → **424**, `LiouvilleGreen` 148 on
the full set — none falls. Closes `[11-background-model-not-keyed-on-the-source-grid]` (§4), opens
`[14-archival-read-stops-at-the-pre-gktk-value-columns]` and
`[14-no-archive-of-grid-construction-algorithms]` (§3), and widens
`[03-qcd-inventory-does-not-report-the-representation]`. **The archival guarantee has a measured
limit:** run against the one datastore in the tree, the prompt-14 fallback fires and finds the
background row, and then the *pre-existing* `GkTk-remedial` 03/04 refusal on
`BackgroundModelValue.tau_lo_Mpc` stops it — metadata never recorded can be defaulted to
"unknown", values never computed cannot. Full record:
[`logs/14-key-the-source-grid-construction.md`](logs/14-key-the-source-grid-construction.md).

**Prompt 15 landed `COMPLETE WITH DEVIATIONS` — and the grid's density is now a measurement rather
than a command-line number.** `T_Z_REPRESENTATION_VERSION` is **6** before and after; no
`CosmologyModels/` file is in the diff and **no background value moves**, only which redshifts are
sampled. `SOURCE_GRID_CONSTRUCTION_VERSION` is **1 → 2**. Three production files:
`CosmologyConcepts/wavenumber.py` (the version-2 row, the cap and seven criterion constants, and
`build_z_sample(..., spacing=(u_profile, h_profile))`), `CosmologyConcepts/__init__.py` (the
re-exports) and `main.py`'s grid-construction hunk alone, where two new module-level functions
`pre_grid_background_proxy` and `source_grid_spacing_profile` compute the profile — so that
`load_main_py_functions` can **execute** them, as prompt 11 established.

**The criterion is prompt 12's, used unmodified, and it runs before any `BackgroundModel` exists.**
$h(u)^4|\varphi''''(u)|/384 \le \varepsilon$ with $\varphi'$ the integrand of
`phase_residual_integrand` times $(1+z)$, $\varphi''''$ by the five-point stencil, the band from
`residual_node_range` and $\varepsilon = \min(\mathrm{ulp}(k\tau_{\rm span}), 10^{-6})$ per case.
What supplies it is a memoised duck-typed proxy over the cosmology's own `Hubble` and
`wPerturbations`, with `epsilon`, `d_epsilon_dz` and `d_wPerturbations_dz` closed-form where the
cosmology has them and central-differenced in $u$ where it does not — §10.1's licence, asserted in
the tree. The envelope is over **all fifty production wavenumbers** and both sectors, not §10.5's
three, because `main.py` builds one universal grid and each wavenumber has its own band top; it
costs **0.9 s** (LambdaCDM) and **1.8 s** (QCD) against `compute_background`'s 0.599 s.

**The cap is the user's decision and it is implemented *locally*.** `SOURCE_GRID_MAX_SPACING_FACTOR
= 1.0` multiplies the base lattice's **own** interval at each place, and the construction
*subdivides* base intervals rather than marching, so the new grid is a strict **superset** of
prompt 11's and "never coarser than today, anywhere" is a property of the construction: measured,
the widest interval against the base lattice is exactly **1.0000×** on both models. That is where
this prompt parts company with §10.5's cap-1× column, whose cap is a single scalar
(`max(diff(u))` = 2.3032e-02) over a lattice uniform in $\log_{10}z$ and not in $u$, and which
therefore coarsens the bottom two decades by up to 10.9× — which is how LambdaCDM's "1,634" comes
to be *fewer* than the 1,732 it may not undercut. The grids are consequently **larger** than
§4's targets: QCD 1,773 → **1,996** (digest `a2c32f67`), LambdaCDM 1,732 → **1,778** (digest
`60a3205a`), response grids 156 → **175** and 145 → **149**, both still decimations of their source
grids. **Recorded as a miss of the two sample-count rows, in the safe direction the user's
instruction names.**

**The two rows prompt 12 recorded as a miss come inside their floors, in the production
configuration.** Background rebuilt on the grid the run samples, samples taken from it — §10.4's
row 4, which is the trap prompt 11 fell into: LambdaCDM $T_k$ at $k=10^5$ **5.8576e-08 →
5.1164e-09**, 7.86 → **0.69** ulp, and QCD $T_k$ **5.8437e-08 → 8.4161e-10**, 7.84 → **0.11**. Ten
of the twelve rows improve or are unchanged; **two move the wrong way and are recorded** — QCD
$G_k$ at $10^7$ and $3\times10^8$, 1.26× and 1.21× worse at 7.5e-05 and 1.3e-06 ulp of their
floors. The four-way table's two background columns now agree to four digits (LambdaCDM) and
1.5e-04 relative (QCD), where §10.4 spanned a factor of 300: **prompt 13 turned the grid back into
a sample set**, and what is left at a crossing is prompt 11's neighbourhood at 1.0024e-08 rad,
**1.35 ulp**, against a 1e-06 rad target.

**One thing the criterion alone could not do, and it is measured.** At $\varepsilon$ = the
per-case target the criterion calls the shipped spacing adequate at the very interval that misses
by 7.86×. The cause is the interpolating cubic's **not-a-knot end condition**: realised/predicted
is 0.92–0.94 in the interior but **9.90 / 4.29 / 1.23** in the outermost three intervals of the
band, and §10.2's misses are in the topmost interval. Prompt 12 absorbed it into its bisected scale
(8.053e-10 against a 7.4506e-09 target — the same factor of 9.25), which a production criterion
cannot do because the bisection needs the oracle. `SOURCE_GRID_SPLINE_EDGE_INTERVALS = 3` and
`SOURCE_GRID_SPLINE_EDGE_FACTOR = 10.0` tighten the target at each band edge instead; a global
safety factor of ten was measured against it and refines 560 production intervals rather than 226,
almost all where the realised error is $10^{-12}$ against a $10^{-7}$ floor.

`[03-derivative-pad-clamp-on-coarse-grids]` **does not bind** (lowest interval 2.1159e-03 against
$-\log(0.9)/12 = 8.7800\times10^{-3}$), and log 13 §1's three-crossing `epsilon` row is
**identical to the shipped grid's** at all three crossings — which is what
`[13-segmenting-costs-accuracy-on-a-grid-that-does-not-resolve-the-crossing]` asked whoever took
this recommendation to report, and it narrows that entry: a cap that can only refine cannot put a
segment edge on a coarser lattice. The QCD reference fixture was regenerated in this commit and
**12 of 12 science keys are bit-identical** (178.2 s, nothing written) — structurally, because the
test tree's reference grid is `wkb_reference.production_source_z_values`, the bare `logspace`,
which prompt 15 does not touch. **No tolerance loosened and none tightened.** Suites:
`CosmologyModels` 30, `ComputeTargets` 424 → **439**, `LiouvilleGreen` 143/143 fast set — none
falls. Closes
`[12-source-grid-density-is-uniform-over-a-curvature-that-spans-eight-orders]` (§4), narrows
`[13-segmenting-costs-accuracy-on-a-grid-that-does-not-resolve-the-crossing]`, and opens
`[15-the-edge-factor-is-applied-at-the-band-edge-not-at-the-consumers-own-end]` and
`[15-the-grid-now-depends-on-the-wavenumber-sample-and-no-tag-says-so]` (§3). Full record:
[`logs/15-equidistribute-the-source-grid.md`](logs/15-equidistribute-the-source-grid.md).

**The representation version.** `T_Z_REPRESENTATION_VERSION` is introduced by prompt 03 and bumped
by **04, 05, 06, 07 and 13**. Its value at each prompt boundary is recorded here as the campaign runs,
because it is the only thing that tells a datastore that its QCD rows are stale. **Bump it on
`LambdaCDM_GenericEOS`, not on `QCD_Cosmology`**, and add a row to the table in the comment block
above the declaration.

| After prompt | `T_Z_REPRESENTATION_VERSION` | What changed |
|---|---|---|
| 01 | *(does not exist)* | no production file touched |
| 02 | *(does not exist)* | no production file touched |
| 03 | **1** | nothing numerically; the key exists |
| 04 | **2** | `_solve_T_z` tightened from `xtol=1e-6, rtol=1e-4` to `xtol=1e-300, rtol=1e-14` |
| 05 | **3** | `F(u) = log(T / [T_CMB (1+z)])` is splined, not `T`; 500 nodes at `k=3` unchanged |
| 06 | **4** | `F` splined per branch, edges bisected onto the jumps; 500 / `k=3` → 3000 / `k=5` |
| 07 | **5** | `integration_break_points` declares the EOS crossings alone, not the knots |
| 08 | **5** | unchanged — no `BREAK_POINT_KIND` value moved, so no number moved |
| 09 | **5** | unchanged — a verification prompt; no production file is in the diff |
| 10 | **5** | unchanged — the measurement says change nothing; no production file is in the diff |
| 11 | **5** | unchanged — the source grid moves, the cosmology does not; no `CosmologyModels/` file is in the diff |
| 12 | **5** | unchanged — a measurement and a recommendation; no production file is in the diff at all |
| 13 | **6** | `BackgroundModel` splines its derivative fields **one per branch**, at both sites, so no smooth interpolant runs across a declared crossing; every stored QCD $\omega_{\rm eff}$, and therefore every stored phase, moves |
| 14 | **6** | unchanged — no `CosmologyModels/` file is in the diff and no background number moves; what changes is which objects a query matches and what a row records about itself |
| 15 | **6** | unchanged — no `CosmologyModels/` file is in the diff and no background number moves; what changes is which redshifts are sampled |

**The source-grid construction version.** `SOURCE_GRID_CONSTRUCTION_VERSION`
(`CosmologyConcepts/wavenumber.py`) is introduced by prompt 14 and is the same kind of thing one
level out: an integer naming the *algorithm* that builds the source grid, whose bump is the only
signal a datastore gets that the algorithm has moved. **Prompt 15 bumped it in the same commit that
changed how the grid is built**, and added a row to the table in the comment block above the
declaration. `ComputeTargets/tests/test_source_grid.py::TestTheConstructionVersionNamesThisAlgorithm`
pins the version to its constants and to all four production grids' digests, and will fail if any
of them moves without the bump — deliberately.

| After prompt | `SOURCE_GRID_CONSTRUCTION_VERSION` | What the algorithm is |
|---|---|---|
| 11 | *(does not exist)* | the construction itself lands: the straddling pair, the ±5 × 2 neighbourhood, the equality redshifts |
| 12, 13 | *(does not exist)* | the grid does not move |
| 14 | **1** | nothing numerically; the version exists and `BackgroundModel` is keyed on it and on the grid's digest |
| 15 | **2** | version 1 plus a base density set by the curvature criterion, under the cap `SOURCE_GRID_MAX_SPACING_FACTOR = 1.0`; QCD 1,773 → 1,996 (`a2c32f67`), LambdaCDM 1,732 → 1,778 (`60a3205a`) |

---

## 2. Mechanism-level tracking

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| **T1** | **DEFECT, critical** | Was 3.461e-08 relative in $\int\mathrm{d}z/H$ — of order 47.5 / 4.75e3 / 1.43e5 rad at $k=10^5/10^7/3\times10^8$ against 1-ulp floors of 3.05e-7 / 3.05e-5 / 9.15e-4 rad, **common mode** between producer and consumer and so invisible to every test in the tree. **Closed by prompt 06:** the guard reads **0.0** — the shipped background's $\int\mathrm{d}z/H$ is bit-identical to the exact background's at all 17 digits — and the equivalent phase is 0.000e+00 rad at all three wavenumbers. **Verified by prompt 09** (`docs/qcd-background-verification.md` §2): the guard reads 0.0 at `HEAD`, the equivalent phases are 0.000e+00 rad at all three wavenumbers, and the six QCD producer rows of `docs/gktk-remedial-verification.md` §3.2/§3.3 — which are common mode in the background's *value* but not in its *smoothness* — all fell to their 1-ulp floors, three of them from above. | 01, 04, 05, 06, 09 | ⚠️ |
| **T2** | **DEFECT, high** | `_solve_T_z` root-solves each spline node to `rtol=1e-4`; neighbouring nodes carry uncorrelated errors up to 2.496e-05. The root of `[02-qcd-T-z-spline-node-tolerance]`, and (measured, not asserted) of the $\pm0.1$ scatter in $\omega^2/\omega_0^2$ that `RESIDUAL_WKB_REGION_MARGIN = 0.5` exists to survive (`ComputeTargets/phase_residual.py:220`). | 04 | ⚠️ |
| **T3** | **DEFECT, medium** | $T$ was splined against $u$, spending resolution on the $(1+z)$ ramp known in closed form. **Fixed by prompt 05:** `TemperatureRepresentation` splines $F(u)=\log(T/[T_{\rm CMB}(1+z)])$ and multiplies the ramp back in, improving the median 412× at the same 500 nodes (1.071e-07 → **2.599e-10**) and the p90 2.2× (1.936e-07 → **8.912e-08**) at *lower* cost per call (2.240 µs). Exact on a constant-$g_s$ equation of state (2.928e-16). | 05 | ⚠️ |
| **T4** | **DEFECT, high** | One global spline across three points at which $T(z)$ genuinely **jumps** (7.625829e-04 in $T$ at $z_c=4.25337\times10^7$). The max error was pinned near the jump height at 500, 2,000 and 5,000 nodes alike. **Fixed by prompt 06:** `SegmentedEntropyFactor` interpolates $F$ one spline per branch with the edges *bisected* onto the jumps, and the max falls **7.236e-04 → 6.807e-11** while the step itself is reproduced to six figures on both sides. An edge misplaced by one node restores 7.054e-04, and a test asserts that it does. | 06 | ⚠️ |
| **G1** | **DEFECT, high** | 404 of the 407 `BREAK_POINT_ALL` points were knots of the auxiliary interpolant — 2,411 of 2,414 after prompt 06's node count — a Gauss panel split every 0.67 grid intervals throughout `BackgroundModel`, and the sole cause of `prompts/phase-representation` prompt 02's Schoenberg–Whitney failure. **Fixed by prompt 07:** `integration_break_points` declares the equation of state's temperature crossings and nothing else, **3** and **2** on the production grid with **0** knots, measured rather than asserted (at order 5 the first discontinuous derivative of $F$ is the fifth, three levels below `d3_lnH_dz3`; the observable residual across a knot is 6.3e-12 in $H$ against 2.1e-04 for the old cubic lattice). The QCD build falls 16,580 → **6,936** integrand evaluations and 0.959 → **0.599 s**; the references do not move at all; `cs_tau` and `friction_F` score against them unchanged to every digit printed and `tau` moves 2.104e-15 → 2.254e-15, at 12 % of its floor; and a repeated-knot vector constructs on all six grids. **Prompt 08 re-took `GkTk-remedial` prompt 19's per-sector policy measurement against the new set and found state (a):** the $T_k$ sector converges at all 50 QCD wavenumbers under *either* policy (7.08e-09 / 8.85e-09 worst, zero offenders, against 1.97e-07 and three offenders then), so the 404 knots were standing in for the representation's defect and not for anything the integrator needed; $G_k$ improved to 3.67e-09 under both policies; the smooth models are bit-identical between the policies and reproduce prompt 19's grid totals as exact integers. Neither `BREAK_POINT_KIND` was changed — README §7 D5 is reported, not decided. **Prompt 09 re-measured the cost on the production path:** QCD per-object build costs fall $G_k$ 8,380 → **6,892** and $T_k$ 12,896 → **11,532** integrand evaluations, the off-grid `raw_theta` accessor needs **4.00** evaluations per call where it needed 4.27, and every LambdaCDM and every cached-evaluation count is exactly unchanged. | 07, 08, 09 | ⚠️ |
| **P2** | **DEFECT, accuracy** | Inherited `[13-consumer-spline-crosses-eos-break-points]`: `PrimitivePhase` splines $\varphi$ with default knots across the declared break points. 1.907e-6 rad ($G_k$, 8 ulp) and 3.186e-6 rad ($T_k$, 428 ulp) at $z=4.24\times10^7$ at the campaign base; **4.3× larger on the corrected background** (prompt 09: 8.107e-6 / 1.398e-5 rad, 34 / 1877 ulp), because the old representation was smearing the equation of state's step over ~4 grid intervals (25.55 % of it inside the crossing's own interval, against 99.84 % of a step 2.78× taller now). **Answered by prompt 10, and not as the plan expected: the knots are not the remedy.** Nine schemes scored over all twelve production rows on the corrected background — the repeated multiplicity-`spline_order` ($C^0$) knot vector is **2.09×/2.10× worse**, per-segment splines **5.00×/5.06× worse**, the best non-$C^0$ control 1.21× better, the ten rows at 1.00 ulp unmoved by all nine, and LambdaCDM identical throughout. Prompt 02's kink fit, re-taken, is still **window-dependent** ($[\varphi']$ moves two orders and changes sign between 1-, 2- and 3-interval windows), which is smooth-but-unresolved data and not a corner a $C^0$ knot can turn. **The production source grid is the limit**: ±5 grid intervals at 2× — 10 extra samples in 1,016 — give **1.64 ulp** and **74.60 ulp**, both inside the 1e-06 rad target, while refining the crossing's own interval alone stalls at 1.96×. `PrimitivePhase` keeps its default knots, `T_Z_REPRESENTATION_VERSION` stays 5, `num_chunks` stays 1, and no production file changed. The entry closes on the `GkTk-remedial` board §4 and the unfixed accuracy defect re-opens as `[10-consumer-phi-unresolved-at-the-eos-crossing]` (§3), **assigned to prompt 11**. | 10 | ⚠️ |
| **G2** | **DESIGN** | The source grid never consulted the cosmology: `populate_z_sample` was a bare `logspace`, `winnow` a blind stride `[::-n]`, and the tag `SourceRedshiftGrid_{len}` labelled size only, so two different grids of equal length collided in the datastore. **Prompt 11 landed the mechanical half.** `build_z_sample` takes the points the cosmology declares — values, not a cosmology object, and no equation-of-state import in `CosmologyConcepts/` — and gives each a **pair straddling it** at a quarter of a grid interval plus the **±5 intervals refined by 2** that prompt 10 measured; `winnow(sparseness, protect=...)` retains them, matching on `store_id` so nothing compares a recovered redshift for equality; and the tags carry a `blake2b` digest of the grid's own values. QCD 1,732 → **1,773** samples (+2.37 %), LambdaCDM **bit-identical**, and the two consumer rows prompt 09 recorded as a miss go 34.11 → **1.61 ulp** and 1876.61 → **65.78 ulp**, both inside the 1e-06 rad target. **The prompt's own remedy — two straddling samples — buys only 1.96×/1.98× on its own**; the neighbourhood is what carries it. The tag change invalidates eight stored object types and the bill is quantified in log 11 §5. **Prompt 12 answered the density half and changed nothing, which is what its §4 asks for.** The uniform `samples_per_log10z = 100` is wrong in **both** directions: measured against the phase residual itself (not against $\varphi$ recovered from a stored $\theta$, which is floor-limited), the consumer's cubic misses the storage floor by **7.86×** and **7.84×** in the top decade of the $T_k$ band at $k=10^5$ on LambdaCDM and QCD, and has up to **2.1e+19** of headroom at the bottom of the range, with the spacing constant to four digits across fourteen decades of it. The criterion that fixes it is $h^4|\varphi''''|/384 \le \varepsilon$ with $\varphi' = -(1+z)C/(\omega+\omega_0)$ — **computable before the grid exists** from $H$, $c_s^2$ and $k$, at $5N$ closed-form evaluations (0.01–0.27 s against `compute_background`'s 0.599 s) — and it predicts the realised error to **±2 %** over 500-odd intervals in the $T_k$ sector on both models at all three wavenumbers. One universal envelope grid: at the same sample count (1,761 against 1,773 on QCD, 1,634 against 1,732 on LambdaCDM) every row is inside its target where the shipped grid misses two; at the same accuracy it needs **1.75×** and **2.06×** fewer. The second candidate, equidistributing $\varphi$ itself, is **refuted** (114,281 samples and still missing two rows). **The saving and `[03-derivative-pad-clamp-on-coarse-grids]` are the same lever** — the clamp binds at the first coarsening step — and the grid also carries the numeric ODE, four cumulative tables and `QuadSourceIntegral`, **none of whose requirements is measured**, so the criterion is a lower bound on the density and never an upper one. **Prompt 14 closed the identity half.** The digest told two grids apart but nothing recorded *which run* an object belonged to, `BackgroundModel` was keyed on neither the grid nor its construction, and all seven `extract_*.py` passed zero tags against `main.py`'s eight-plus tagged call sites — safe while a store held one run, and silent the moment it held two. Now: `main.py --run-label` names a run and the name is a `store_tag` carried by and filtered on for everything the run writes; `SOURCE_GRID_CONSTRUCTION_VERSION = 1` names the algorithm beside the digest that names the grid; `sqla_BackgroundModelFactory` filters on `source_grid_digest` and `source_grid_construction` on the compute path and **refuses a mixture, naming every generation it found**, on the read path; and a pre-prompt-14 store, which cannot be computed into, is still **read** and reported as an unknown generation. The grid itself is byte-identical. **Prompt 15 closed the density half, which is prompt 12's recommendation and the user's decision.** `build_z_sample` takes a `spacing` profile — two arrays, still no cosmology object and still no equation-of-state import in `CosmologyConcepts/` — and subdivides every base interval that $h^4|\varphi''''|/384 \le \varepsilon$ finds too wide, under `SOURCE_GRID_MAX_SPACING_FACTOR = 1.0`: **never coarser than the uniform lattice puts there, anywhere**, measured at exactly 1.0000× on both models, so the new grid is a strict superset of prompt 11's and every sample prompt 11 placed is where prompt 11 placed it. The profile comes from `main.source_grid_spacing_profile`, which builds it from the cosmology's own `Hubble` and `wPerturbations` alone — no `BackgroundModel`, because none exists yet — over all fifty production wavenumbers and both sectors, in 0.9 s (LambdaCDM) and 1.8 s (QCD). QCD 1,773 → **1,996**, LambdaCDM 1,732 → **1,778**, response grids 156 → **175** and 145 → **149**; the two rows prompt 12 recorded as a miss go **7.86 → 0.69** and **7.84 → 0.11** ulp in the production configuration, with two negligible rows 1.2× the wrong way and `[03-derivative-pad-clamp-on-coarse-grids]` still not binding. `SOURCE_GRID_CONSTRUCTION_VERSION` is **2**. §4's ~1,761 / ~1,634 sample counts are **missed in the safe direction**: prompt 12's cap was a scalar over a lattice uniform in $\log_{10}z$, so its cap-1× column coarsens the bottom two decades. | 11, 12, 14, 15 | ⚠️ |
| **B1** | **DEFECT, accuracy** | *(workstream E, opened by prompt 12 as `[12-background-derivative-fit-grid-rings-at-a-step]`)* `QCD_Cosmology` supplies no `d_lnH_dz`, so `BackgroundModel` built one — and then splined the **stored** result again in `_create_functions` — with neither lattice split at `integration_break_points`, while $H$ genuinely **steps** at two of the three declared crossings. `epsilon`, `d_epsilon_dz` and `d2_epsilon_dz2` rang: **2.04e-02** relative at `T_LO` and **1.03e-03** at `T_120_MEV` on the production grid against **3.9e-09 max / 8.1e-10 median** away from a crossing, with `EOS_T_LO` (where only $w$ kinks) the control at 1.6e-09. It is why prompt 11's fix did not reach production: in the configuration `main.py` uses, the consumer's phase at the crossing read 2.4859e-05 rad against the 3.9744e-07 prompt 11 measured with the background held on the base grid. **Fixed by prompt 13:** both sites fit one spline per branch, the two genuine steps read **1.931e-09** and **4.070e-09**, the control is the same float to the bit, and the production row is **5.8437e-08 rad**, 425× better and with its maximum no longer at a crossing. The four cells of the background×samples table now agree to four digits where they spanned 300×. `BREAK_POINT_ALL` rather than the two jumps, measured: `d_wPerturbations_dz` at `EOS_T_LO` is 346 % wrong without the third crossing and 7.8e-04 with it. Integrand counts exactly unchanged; every cosmology that declares nothing byte-identical. | 12, 13 | ⚠️ |

**T1 is closed, and the record of how it fell is the point.** Prompt 01 put
`CosmologyModels/tests/test_T_z_representation.py::test_conformal_time_matches_the_exact_background`
in the tree: $\int\mathrm{d}z/H$ over $z\in[10^2,10^{12}]$ computed twice from the same cosmology,
once as shipped and once with the temperature replaced by the accurate root solve, so that nothing
cancels. It measured **3.4605051e-08** on prompt 01's tree, against the audit's 3.461e-08, in
0.013 s. Prompt 04 left it at 3.4509e-08; **prompt 05 took it to 5.4264e-10**, a factor of 64,
which says that most of the conformal-time error was never the jump — a jump is a set of measure
zero in an integral — but the interpolation error carried across the whole range, which is what
the median measures. **Prompt 06 took the remaining 5.4e-10 (0.74 rad at $k=10^5$/Mpc) to zero**:
the two integrals are now bit-identical, `1.3320002507788795e+03` both, and `CONFORMAL_TIME_REL`
is 1e-15 — asserted as a threshold rather than as an equality, because the identity is the last bit
of a sum over a few hundred quadrature panels; the three phase floors are asserted alongside it.

**Out of scope (do not schedule here):** `[00-consumer-anchoring-floor]` and
`[02-consumer-phi-below-the-storage-granularity]` — per-region anchoring, untouched by background
accuracy (audit §9); the numeric→WKB hand-over (`docs/OPEN_ISSUES.md` §1.1); tolerances and Gauss
orders (`prompts/tolerance-convergence`); the `QCD_EOS` fitting coefficients and branch boundaries
(README §0.5, §7 D6); `AdaptiveLevin/` and the Levin consumers.

---

## 3. Active and unresolved issues

Opened by **prompt 15**, 2026-09-15:

- **[15-the-edge-factor-is-applied-at-the-band-edge-not-at-the-consumers-own-end]** *(opened by
  prompt 15, 2026-09-15)* — the curvature criterion sets the spacing from
  $h^4|\varphi''''|/384 \le \varepsilon$ with the textbook interior constant, and prompt 15
  measured that the *realised* constant at the outermost intervals of a band is about **ten times**
  larger (9.897 / 4.292 / 1.230 against an interior 0.923 on LambdaCDM $T_k$ at $k=10^5$, and the
  same pattern on QCD in both sectors), because `make_interp_spline` closes a cubic with a
  not-a-knot end condition. `SOURCE_GRID_SPLINE_EDGE_INTERVALS = 3` and
  `SOURCE_GRID_SPLINE_EDGE_FACTOR = 10.0` tighten the target at the edges of the band
  `residual_node_range` defines — which is where §10.2's two misses live, and where
  $|\varphi''''|$ is largest. **But a production `PrimitivePhase` object's spline does not end
  there.** It ends at that object's own anchor, three e-folds inside the horizon and therefore
  *inside* the band, where $|\varphi''''|$ is smaller but the end condition's amplification is the
  same, so such an end carries up to $10\varepsilon$ where the criterion promises $\varepsilon$.
  **Impact:** not measured, in either direction — every measurement in verification-document §10
  and in log 15 scores the band, not an object. The rows that *are* scored against real stored
  objects (`gktk-remedial-verification` §3.5) were 1.61 and 65.78 ulp at prompt 11 and nothing in
  prompts 13–15 moves them the wrong way, so this is a gap in the argument rather than a known
  error. **Next step:** score `PrimitivePhase`'s error at its own anchor against the residual
  oracle, for one QCD $T_k$ object at $k=10^5$, and if it is above the target either widen
  `SOURCE_GRID_SPLINE_EDGE_INTERVALS` or apply the factor as a global safety factor — log 15
  deviation 3 costs both (global 10: 560 intervals refined and +614 samples on QCD, against 226 and
  +242).

- **[15-the-grid-now-depends-on-the-wavenumber-sample-and-no-tag-says-so]** *(opened by prompt 15,
  2026-09-15; narrowed by prompt 16, 2026-09-15)* — the source grid was a pure function of
  $(z_{\rm init}, z_{\rm end}, \texttt{samples\_per\_log10z})$; prompt 15's envelope is taken over
  every wavenumber the run serves, so it is now also a function of the **wavenumber sample**. Two
  runs differing only in `NUMBER_SOURCE_K_VALUES` — a hard-coded 50 at `main.py:3292`, not a
  command-line argument — get different grids. **That is caught**, by prompt 11's content digest
  and prompt 14's construction version: the grids get different tags and `BackgroundModel` refuses
  the mixture. **Prompt 16 closed the tag half**: `SourceSamplesPerLog10ZTag`, which used to
  mislabel the base density as though it were the density, is retired outright (declaration,
  label construction and all 27 `tags=[...]` uses), so it no longer misdescribes anything. **The
  wavenumber-set half stands, by the user's explicit decision** (prompt 16 §1, quoted): *"we have
  to go further back for small `k`. I don't think there is anything we need to do about this; we
  just need to bear it in mind when we're doing a concrete calculation."* No `SourceKSampleTag` is
  to be added and the wavenumber envelope's digest is not to be folded into the construction
  version; a reader of a store still sees two digests with no way to tell which knob moved, and
  that is accepted. **Impact:** none on correctness. **Next step:** none — recorded so a later
  reader does not reopen this as an oversight.

Opened by **prompt 14**, 2026-09-15:

- **[14-archival-read-stops-at-the-pre-gktk-value-columns]** *(opened by prompt 14, 2026-09-15)* —
  prompt 14's read path makes a datastore that predates **prompt 14** readable: `build()` catches
  the missing `source_grid_digest` / `source_grid_construction` columns, re-issues the query
  without them, and reports the row as an unknown generation. It does **not** make a store
  readable that predates `GkTk-remedial` prompts 03/04, whose refusal on
  `BackgroundModelValue.tau_lo_Mpc` is unconditional on both paths. **Measured** against
  `physics-test-n20-lambdacdm-zend0p1-shard0000.sqlite`, the only datastore in the tree: the
  prompt-14 fallback fires and finds the background row, `store_tag` yields nine labels and zero
  runs, `choose_run_label` reports the unnamed-run case correctly — and then `build()` raises the
  `tau_lo_Mpc` message. **Impact:** the user's requirement that a superseded datastore keep
  archival value holds for stores written after `GkTk-remedial` prompt 04 and not for anything
  older, and there is no store in the tree on which the happy path can be demonstrated end to end.
  The two guards are not the same kind of thing — a grid identity is metadata the old store never
  *recorded*, while `tau_lo_Mpc` / `cs_tau_Mpc` / `cs_tau_lo_Mpc` / `friction_F` are values it
  never *computed* — so softening the second would mean fabricating background values and was not
  considered. **Next step:** if the archival case is wanted for the oldest stores, it needs a
  reader that stops at `BackgroundModel` and never asks for the value rows (or a `BackgroundModel`
  that can be deserialised without them). That is a design, not a patch, and nobody has asked for
  it.

- **[14-no-archive-of-grid-construction-algorithms]** *(opened by prompt 14, 2026-09-15, in the
  user's framing; **nothing was built towards it**)* — the user has noted that **an archive of
  grid-construction algorithms may be wanted later**: a datastore records which construction built
  its grid (`SOURCE_GRID_CONSTRUCTION_VERSION`, carried as a `store_tag` and as a
  `BackgroundModel` column), but the *code* of a superseded construction lives only in git
  history, so a run written under version 1 cannot be reproduced once version 2 has replaced
  `build_z_sample`. **Impact: no longer hypothetical — prompt 15 made version 2 the construction**
  (2026-09-15), so version 1 is now reachable only from git history and a version-1 run can be
  *read* but not *re-derived*. A version-1 grid is still recoverable in practice, because version 2
  is version 1 plus a step and `build_z_sample` called without a `spacing` profile still builds
  version 1's grid exactly — which is asserted in the tree — but that is a property of this
  particular succession and not of the scheme.
  **Prompt 14 deliberately built nothing towards this** beyond the version integer that would be
  its key (prompt §2 item 6): an archive is a design with a maintenance cost — every retired
  constructor kept alive and tested forever — and the decision is the user's. **Next step:** none
  proposed. If it is taken, the shape is a registry keyed on the version integer, and the thing to
  decide first is whether a retired construction has to keep *running* or only has to be
  *readable*, because the two have very different prices.

Opened by **prompt 13**, 2026-09-15:

- **[13-segmenting-costs-accuracy-on-a-grid-that-does-not-resolve-the-crossing]** *(opened by
  prompt 13, 2026-09-15; **priced by prompt 15, 2026-09-15, and narrowed**)* — segmenting a spline at a declared
  crossing creates two new *interior* ends, and unlike the outer ends of
  `_build_derivative_fit_grid` they **cannot be padded**: what lies beyond an edge is the other
  branch, whose values are exactly what must not enter the fit, and the cosmology exposes no
  analytic continuation past the crossing. At the stored-sample site padding is not merely
  unavailable but meaningless — the node set *is* the sample grid and the quantity splined is one
  the cosmology does not supply. So the accuracy at a crossing is set by how close the nearest
  samples are, and on a grid that does not resolve it a cut is worse than the (wrong) smooth fit it
  replaces. **Measured** at the `EOS_T_LO` control, where $H$ does not step and nothing rang:
  `epsilon` against a central difference of the cosmology's own `Hubble` goes **1.889e-09 →
  8.190e-08** on the uniform 1,732-sample base grid, **43× worse**. On prompt 11's production grid
  it is **1.609e-09 → 1.609e-09**, the same float, because that grid puts a straddling pair at a
  quarter of an interval and refines ±5 intervals by 2 around every crossing. **Impact:** none
  today — production uses the cosmology-aware grid. **But
  `[12-source-grid-density-is-uniform-over-a-curvature-that-spans-eight-orders]`'s cap-2× column is
  coarser than the base grid everywhere**, and its own §10.5 table does not mention this
  constraint, because prompt 12 measured a tree in which no spline was segmented.

  **Priced by prompt 15, and the risk did not materialise** (2026-09-15). The cap that shipped is
  `SOURCE_GRID_MAX_SPACING_FACTOR = 1.0` applied to the base lattice's *own local* interval, so the
  production grid is a strict **superset** of the one this row was measured on and no crossing
  neighbourhood can be coarsened by it. Log 13 §1's row, re-taken on the 1,996-sample grid:
  `T_LO` **1.6896e-09**, `EOS_T_LO` **1.9824e-09**, `T_120_MEV` **4.0453e-09**, against the shipped
  grid's 1.6896e-09 / 1.9824e-09 / 4.0481e-09 — identical at two crossings and marginally better at
  the third, and two orders below the base grid's 4.6e-08 / 8.2e-08 / 4.3e-06. **The entry stays
  open** because the *statement* is still true and still unguarded: a segment edge's accuracy is set
  by how close the nearest samples are, and nothing in the tree asserts that a future grid keeps
  them close. **Next step (narrowed):** a test that pins the three-crossing `epsilon` row, so that a
  later coarsening announces itself; the measurement now exists twice and only the assertion is
  missing.

- **[13-crossing-neighbourhood-refinement-was-sized-at-k-1e5]** *(opened by prompt 13,
  2026-09-15)* — `SOURCE_GRID_BREAK_HALF_WIDTH = 5` and `SOURCE_GRID_BREAK_REFINEMENT = 2` were
  fixed by prompt 10's ladder at $k=10^5$, and prompt 10 concluded that "the crossing is a $k=10^5$
  phenomenon" because at $k=10^7$ and $3\times10^8$ its ladder read **0.00 ulp near the break at
  every density**. That was scored against $\varphi$ recovered from a **stored** $\theta$, whose
  granularity at $k=10^7$ is 3.05e-05 rad — larger than the effect. Against prompt 12's residual
  oracle, which sees past that floor, the production configuration at QCD $T_k$, $k=10^7$ reads
  **5.6892e-05 rad near a crossing** after prompt 13, **59.7 ulp** of that band's span, against
  0.008 ulp away from one; $G_k$ at $10^7$ reads 2.4203e-05 rad, **1.59 ulp**, which is at the
  floor. **Impact:** the $T_k$ figure improved 7.9× in prompt 13 (from 470 ulp) and is in the same
  regime as the **65.78 ulp** prompt 11 shipped at $k=10^5$ and called a success, so this is not a
  regression and not urgent; it is the statement that the neighbourhood was sized for one wavenumber
  and the band geometry is different at another. One non-production cell of the four-way table moved
  the wrong way for the same reason (base background / shipped samples, $T_k$ $10^7$:
  4.9524e-06 → 5.6892e-05), by the mechanism prompt 09 measured from the other side — a corrected
  background delivers the step undiluted, so a cubic across it has more to bridge. **Next step:**
  re-run prompt 10's ±$n$ × $m$ ladder at $k=10^7$ against the residual oracle rather than the
  stored phase, and say whether `SOURCE_GRID_BREAK_HALF_WIDTH` should depend on the band rather than
  be a constant. **Prompt 15 did not take it** (2026-09-15): its §2 item 3 requires prompt 11's
  protected set and break neighbourhoods to survive unchanged, and its criterion masks out the
  neighbourhood of every declared crossing by construction, so the density change cannot reach this
  figure and did not — the grid is a strict superset of the one the figure was measured on. It now
  belongs to whichever prompt next owns the crossing neighbourhood, which is a different lever from
  the base density and sits beside it in `build_z_sample`.

Opened by this campaign's planning, 2026-09-13:

- **[00-eos-branch-joins-do-not-match]** *(planning, 2026-09-13; **pinned in a test by prompt 01,
  2026-09-14**, which re-measured every figure below from the test tree and confirmed all of them
  to the digits quoted)* — `QCD_EOS`'s branch joins at
  $T=10^{16}$, 0.12 and $10^{-5}$ GeV jump by **+1.395e-02**, **−3.744e-04** and **−2.284e-03** in
  $g_s$ (and +1.454e-02, −2.075e-04, +8.876e-04 in $g$), forcing steps in $T(z)$ at fixed $z$ of
  −4.649e-03, +1.248e-04 and **+7.614e-04**. The join at 0.002 GeV matches to **1.751e-11**, and
  that asymmetry is the evidence that the other three are a defect of the transcription rather than
  the parametrisation's intent: if discontinuous joins were designed in, none of the four would
  match to 1.8e-11. **Impact:** the $10^{-5}$ GeV join is the origin of the $4.4\times10^{-4}$ jump
  in $H(z)$ at $z=4.24\times10^7$ that `GkTk-remedial` log 02 measured without attribution, and of
  the consumer's worst QCD error in `docs/gktk-remedial-verification.md` §3.5 in **both** sectors.
  **This campaign does not repair it** — the equation of state is an upstream data fixture and a
  segmented representation reproduces a discontinuous fixture exactly (audit §0.2). **Next step:**
  the question in README §7 D6, put to the authors of the Saikawa & Shirai transcription. Prompt
  01's `test_the_branch_joins_are_where_the_fixture_puts_them`
  (`CosmologyModels/tests/test_T_z_representation.py`) now **pins all four joins**, and also pins
  the *set* of `break_temperatures_GeV`, so a later correction announces itself as a test failure
  rather than as a silent change of cosmology. That test's docstring says in terms that a failure
  means the fixture changed — not that the code regressed — and that the campaign's segment edges
  must then be re-derived. Measurements: audit §1 and §2, reproducible in 1.0 s, and re-measured
  by prompt 01 in 0.11 s as part of the `CosmologyModels` suite.

- **[01-convergence-block-has-a-separate-generator]** *(prompt 02, 2026-09-14)* —
  `ComputeTargets/tests/wkb_reference_data.json`'s top-level `convergence` block (node/interval
  geometry, per-Gauss-order convergence figures, `decision.N_tau`/`N_cs_tau`/`N_F`/`N_rho`,
  `decision.rho_adaptive_fallback_required`) is written by a **different** script,
  `docs/gktk-remedial/residual_convergence.py`, not by prompt 02's
  `docs/qcd-background-audit/generate_qcd_references.py`. Five tests read it directly
  (`docs/qcd-background-audit/REFERENCE-FIXTURE.md` §1, §4): `test_qcd_nodes_against_adaptive_reference`,
  `test_qcd_checkpoints`, `test_qcd_short_baselines_including_the_transitions`,
  `test_qcd_break_points` and `test_no_adaptive_fallback_was_required`. **Impact:** a prompt that
  regenerates only the QCD block of the JSON (04, 05, 06) leaves this block stale; its figures
  (floors, `T_spline_knots_in_range`, `branch_boundaries` interval indices) silently disagree with
  the representation actually in the tree until something re-runs `residual_convergence.py`.
  **Next step:** prompt 08's charter ("re-derive `integration_break_points`... and re-measure the
  per-sector break-point policy") is exactly what that script computes, so it is the natural place
  to close this — but 04–07 should not assume it is regenerated in the meantime, and should say so
  in their own logs if a `convergence`-scored test's accuracy figure (not just its break-point
  count) moves.
  **Escalated by prompt 05 (2026-09-14): it is now the sole cause of three loosened tolerances,
  and prompt 08 should take all three back in the commit that re-runs the script.** Prompt 04 had
  loosened one; prompt 05 regenerated the QCD block a second time and had to loosen three, which
  is one more than its own §2 item 6 allows — the prompt **stopped and asked**, and the user
  authorised landing them (option (a)). Each is a figure measured on the entropy-factor background
  scored against a floor recorded on the $T$-against-$u$ one:

  | constant | module | was | now | measured |
  |---|---|---|---|---|
  | `QCD_BREAK_POINT_ALIGNMENT_TOL` | `test_background_tau.py` | 1.4e-05 | **3.1e-05** | 3.046858e-05 (`T_120_MEV`) |
  | `QCD_FLOOR_FACTOR` | `test_background_tau.py` | 3.0 | **3.2** | 5.8348e-14 / 1.879e-14 = 3.106 |
  | `QCD_FLOOR_FACTOR` | `test_background_cs_tau_friction.py` | 3.0 | **8.3** | 1.5501e-13 / 1.887e-14 = 8.213 |

  The two `QCD_FLOOR_FACTOR`s multiply `json_vs_reference_max_rel` from the stale block, and the
  quantity scored against it is the same kind of thing one level down — model-against-JSON, where
  the floor is JSON-against-adaptive-reference — so both sides are floors, at the 1e-13 level, a
  few hundred ulp of a cumulative quadrature over twenty decades. The worst point moved from
  $z=1.005\times10^7$ to $z=1.007\times10^{11}$. **Prompt 08 must re-measure all three after
  re-running the script and put them back to 1e-9 / 3.0 / 3.0 if they will go**; if one will not,
  that is a finding about the representation rather than about the block's age, and it should be
  said so explicitly.

  **Two of the three came back at prompt 06 (2026-09-14), one prompt early**, because segmenting
  the representation moved the numerators by two orders: `QCD_FLOOR_FACTOR` is **3.0** again in
  both modules (`tau` 5.8348e-14 → **2.104e-15**, `cs_tau` 1.5501e-13 → **2.212e-15**, against
  floors still recorded as 1.879e-14 and 1.887e-14 — the model's fixed-order table now agrees with
  the JSON an *order below* the floor recorded for the JSON itself). The third moved **further**:
  `QCD_BREAK_POINT_ALIGNMENT_TOL` 3.1e-05 → **1.5e-04**, measured 1.418851e-04 at `T_120_MEV`,
  because the crossing `_temperature_crossing_log1pz` finds is now on the genuine jump
  ($u = 27.485391822$, within 3 ulp of the independent bisection) while the block still records
  where a smooth interpolant passed through 0.12 GeV ($u = 27.485249937$). **Prompt 08 inherits one
  figure, not three, and should expect the tree's to be the right one.**

  **Prompt 07 re-measured it and it did not move** (2026-09-14): 1.418851e-04 at `T_120_MEV` to
  every digit, with 1.728034e-05 at `T_LO` and 1.060594e-06 at `EOS_T_LO`. Prompt 07's break points
  are the *bisected* crossings rather than the root-found ones and the two differ by ~1e-14 in $u$,
  so the whole of this figure remains the age of the block. `QCD_BREAK_POINT_ALIGNMENT_TOL` is left
  at **1.5e-04**, untouched, and the other two constants are still at 3.0.

  **Prompt 08 could not take it, and re-points it at prompt 09** (2026-09-14). The entry named
  prompt 08 as the natural place to re-run `docs/gktk-remedial/residual_convergence.py`, but that
  prompt's "Files you may touch" is `docs/qcd-background-audit/`, the two `BREAK_POINT_KIND`
  comment blocks, the log, the board and `docs/OPEN_ISSUES.md` — it includes neither
  `ComputeTargets/tests/wkb_reference_data.json` nor `ComputeTargets/tests/test_background_tau.py`,
  and its §4 acceptance table carries no tolerance row. Regenerating the block and taking
  `QCD_BREAK_POINT_ALIGNMENT_TOL` back would have been scope creep (README §5 rule 5), so it was
  not done and the figure is unchanged, prompt 08 having moved no number at all.
  **Prompt 09 could not take it either, and it is now unassigned** (2026-09-14). Prompt 08 named
  prompt 09 as "the first prompt after this one with the JSON and that test module naturally in
  scope". **They are not in scope.** Prompt 09's "Files you may touch" is
  `docs/qcd-background-verification.md`, `docs/qcd-background-audit/`,
  `docs/gktk-remedial-verification.md` (§9 only), its log, this board and `docs/OPEN_ISSUES.md`;
  its "Do not touch" is **any production file**, with the stated reason that a verification prompt
  changing production code is what created `prompts/phase-representation`. Closing this needs
  `docs/gktk-remedial/residual_convergence.py` to *write* `ComputeTargets/tests/wkb_reference_data.json`
  and then `ComputeTargets/tests/test_background_tau.py` to be edited, and neither is on the list —
  so it would have been scope creep under README §5 rule 5, and it was not done. `QCD_BREAK_POINT_ALIGNMENT_TOL`
  is unchanged at **1.5e-04**, measured 1.418851e-04, and both `QCD_FLOOR_FACTOR`s are still at 3.0.
  **Next step:** whichever prompt next has that JSON and that test module in scope, which is *not*
  a prompt of this campaign's ungated chain — the chain is closed. `[02-qcd-reference-floor]` on the
  `GkTk-remedial` board waits on the same run.

  **Assigned (2026-09-16): `prompts/tolerance-convergence`, prompt 04.** That campaign's rebase
  (`prompts/tolerance-convergence/RECONCILIATION.md` §5) found the block's staleness to be wider
  than a tolerance owed: `decision.recommended_scheme` is `"branch+knots"`, and prompt 07 of this
  campaign removed the knots, so **the evidence for `N_tau = N_cs_tau = N_F = N_rho = 4` names a
  break-point scheme that no longer exists** and was measured on the `T(z)` representation prompts
  04-06 replaced. Its prompt 04's charter is the Gauss orders themselves, so it cannot avoid
  re-running the generator; that this makes an audit prompt write a fixture and edit
  `test_background_tau.py` is that campaign's README §7 **D5**, put to the user rather than
  assumed. Re-running the script also needs its scheme sweep updated, since one of its three
  schemes is gone. `[02-qcd-reference-floor]` still waits on the same run.

  **Prompt 04 of `prompts/tolerance-convergence` ran the generator on 2026-09-17 and could not
  write the block either — but for a new reason, and the reason is measured.** Everything the
  entry asks for is now established: the four orders are **unchanged at 4** on the corrected
  background and the 3-point break set, `decision.recommended_scheme` becomes **`branch`** (the
  knots are retained as a control and are measured to buy nothing — `branch` 4.835e-16 against
  `branch+knots` 6.011e-16 on QCD's $\tau$ at order 4, confirming prompt 07 at the quadrature
  level), and **`QCD_BREAK_POINT_ALIGNMENT_TOL` would go to 1.421085e-14** — ten orders tighter
  than the 1.4e-05 this entry hoped for, which settles beyond doubt that the whole of the present
  1.418851e-04 is the block's age and none of it is the representation.

  **What blocks the write is the two `QCD_FLOOR_FACTOR` sites this entry already names.** The
  regenerated `json_vs_reference_max_rel` is **3.223619e-16** ($\tau$) and **4.354138e-16**
  ($c_s\tau$) against the recorded 1.878541e-14 and 1.886653e-14 — 58x and 43x tighter, because
  prompts 04–06 of this campaign replaced the representation under the JSON's own QCD reference
  values. The quantity scored against them does not move at all: **2.254e-15** and **2.212e-15**,
  the same figures prompt 06 recorded. So *production error ≤ 3 × floor* fails by **6.99×** and
  **5.08×**, and one of the two modules (`test_background_cs_tau_friction.py`) is outside that
  campaign's D5 carve-out. This entry's own prediction — "if one will not go back, that is a
  finding about the representation rather than about the block's age" — resolves the other way:
  **the tolerances go back, and it is the *test construction* that does not survive**, because it
  divides the model's accuracy by the reference's and the two have stopped being the same size.

  **Next step, and it is a decision rather than a scope question:** give the two tests a bound
  that is not the reference's floor, or raise both `QCD_FLOOR_FACTOR`s past 7; then regenerate the
  block and take `QCD_BREAK_POINT_ALIGNMENT_TOL` to just above 1.421085e-14 **in the same
  commit**. Tracked as `[04-convergence-floor-used-as-a-test-threshold]` on the
  `tolerance-convergence` board, which holds the measurements. Evidence:
  `docs/tolerance-convergence/ORDER-AUDIT.md` §1, §9.

  **CLOSED 2026-09-18 by `prompts/tolerance-convergence` prompt 04b — see §4.** The user removed
  the scope boundary (that campaign's README §7 **D8**, which widens D5 by
  `test_background_cs_tau_friction.py`), the block was regenerated into the tree, both thresholds
  were **rebuilt rather than raised** — absolute bounds of 1.0e-14 on the production quantities,
  against their own measured accumulation floors, with `QCD_FLOOR_FACTOR` removed from both
  modules — and **`QCD_BREAK_POINT_ALIGNMENT_TOL` went 1.5e-04 → 3.0e-14** in the same commit.
  This entry's prediction held in full: the whole of the excess was the block's age.
- **[04-unsplit-tk-run-now-meets-the-criterion]** *(prompt 05, 2026-09-14; **assigned to prompt
  08**)* — `ComputeTargets/tests/test_numeric_break_points.py::TestQCDReferenceConvergence::test_split_converges_where_unsplit_does_not`
  asserted `unsplit > 1e-6`: that a $T_k$ numeric run which does **not** split at the declared
  `T_120_MEV` discontinuity fails `GkTk-remedial` prompt 17 §2.1's convergence criterion
  (3.4e-08) at $k=4.972\times10^7$/Mpc. **That premise is now false.** Measured directly, on the
  two trees either side of this commit, by driving the test class's own `_drift`:

  | tree | split drift | unsplit drift | criterion |
  |---|---|---|---|
  | prompt 04 (`71b842a`) | 2.2136e-09 | **1.0213e-06** — fails, and cleared the `1e-6` bound by 2 % | 3.4e-08 |
  | prompt 05 (this commit) | 2.9753e-09 | **2.2767e-08** — passes | 3.4e-08 |

  The unsplit run improved **45×** while the split run barely moved, which says most of what the
  split was rescuing was never the jump in $H(z)$ at all: it was the $10^{-7}$-level interpolation
  noise the $T$-against-$u$ spline carried across the whole range, and the entropy factor removes
  it (audit §3, T3). **Impact:** this is the first measured evidence that
  `TkNumericIntegration.BREAK_POINT_KIND = BREAK_POINT_ALL` may no longer be load-bearing, which
  is README §2 (f) and §7 **D5** — and `BREAK_POINT_KIND` is in a datastore lookup key, so moving
  it is a production decision with a regeneration attached. It is **one** wavenumber of fifty, in
  one sector, and prompt 05 deliberately did not decide it. **Next step:** prompt 08, whose
  charter is exactly to re-take `GkTk-remedial` prompt 19's measurement across all fifty QCD
  wavenumbers against the new representation; it should read this row before it starts, and
  README §6.3's "≤ 3.4e-08 without them, else stop" is the test. **Prompt 07 changes what the
  question is** (2026-09-14): `BREAK_POINT_ALL` and `BREAK_POINT_DISCONTINUITY` now differ by a
  single kink, `EOS_T_LO`, rather than by 2,411 knots, so re-taking prompt 19's measurement is a
  question about that one crossing. This row's own figures are unaffected — both sets still contain
  `T_120_MEV`, which is the crossing the measurement is about — and neither `BREAK_POINT_KIND` was
  touched. Meanwhile the test asserts the
  weaker true statement that splitting still buys a factor (`UNSPLIT_PENALTY_FACTOR = 5.0`,
  measured 7.65×), and its docstring carries both measurements above.

  **Prompt 08 took the column across all fifty wavenumbers and the answer is "at that one $k$,
  yes; in general, emphatically no"** (2026-09-14; `docs/qcd-background-audit/PER-SECTOR-POLICY.md`
  §2b). With `integration_break_points` suppressed altogether, QCD $T_k$ is above the 3.4e-08
  criterion at **19 of the 50** production wavenumbers, worst **9.61e-06** at
  $k = 4.223\times10^7$/Mpc, median 1.80e-08. At this entry's own wavenumber
  ($k = 4.972\times10^7$) the unsplit run reads **2.91e-08** on prompt 07's tree and passes —
  confirming the entry's measurement and prompt 05's 2.2767e-08 — but that wavenumber is one of
  the 31 that pass, and it is **not representative**. So: splitting at the equation of state's
  genuine **jumps** is still load-bearing and prompt 18's result stands undisturbed; what prompt 08
  measured to be vestigial is only the distinction between the two *kinds*
  (`BREAK_POINT_ALL` 7.08e-09 against `BREAK_POINT_DISCONTINUITY` 8.85e-09, zero offenders either
  way). **This narrows the entry rather than closing it**, because the assertion in the tree is
  still pinned to a wavenumber at which the original, stronger statement is false.
  **Next step:** whoever next has `ComputeTargets/tests/test_numeric_break_points.py` in scope can
  restore the original statement by re-pointing `test_split_converges_where_unsplit_does_not` at
  $k = 4.223\times10^7$/Mpc, where unsplit reads 9.61e-06 against a split 5.60e-10 — a factor of
  17,000, against the 7.65× the weakened assertion now measures. Prompt 08 did not do it: that
  module is not among the files its prompt may touch. README §7 **D5** is meanwhile **reported and
  not decided**, which is what prompt 08 was asked to do.

- **[08-gk-declared-split-buys-nothing-measurably]** *(prompt 08, 2026-09-14)* — the $G_k$
  numeric sector splits at the cosmology's declared jumps (`BREAK_POINT_DISCONTINUITY`,
  `GkNumericIntegration.BREAK_POINT_KIND`), and prompt 08 measured what that buys: on QCD, worst
  reference-convergence drift **3.67e-09** split against **3.52e-09** unsplit over the 50
  production wavenumbers, with zero above the 3.4e-08 criterion in both cases and 13,343 against
  13,320 right-hand-side evaluations per object
  (`docs/qcd-background-audit/PER-SECTOR-POLICY.md` §2, §2b). The split is, within the noise of the
  measure, buying nothing in this sector — unlike $T_k$, where suppressing it puts **19 of 50**
  wavenumbers above the criterion. **Impact:** none today. It costs 0.17 % of the evaluations, it
  is the same mechanism $T_k$ genuinely needs, and `BREAK_POINT_KIND` is in a datastore lookup key,
  so there is no case for removing it and prompt 08 makes none. It is recorded because no document
  previously said what the $G_k$ split is worth, and a later reader weighing README §7 D5 should
  not have to re-derive it. **Next step:** none proposed; re-measure if the equation of state or
  the response grid changes.

Inherited, and **assigned to this campaign** (each is owned by the board named, which holds its
measurements and its history; the closure is recorded there):

| Issue | Owning board | Closed by | Note |
|---|---|---|---|
| `[13-consumer-spline-crosses-eos-break-points]` | GkTk-remedial | prompt 10 | Assigned 2026-09-13 by `prompts/phase-representation`'s close-out; unblocked by prompt 07 and narrowed twice more (07, 09) on that board. **Closed by prompt 10**, 2026-09-15, in that board's §4, on the measurement that the remedy it names does not work: the $C^0$ repeated knot vector is 2.09×/2.10× worse and per-segment splines 5.00×/5.06× worse at QCD $k=10^5$, nothing in the family reaches the target or returns to the campaign base, and the re-taken kink fit is window-dependent. The unfixed accuracy defect re-opens here as `[10-consumer-phi-unresolved-at-the-eos-crossing]`; the index row is swapped, not deleted, so the count is unchanged |
| `[01-genericeos-tz-spline-floor]` | source-remediation | prompts 05, 06 | **Closed by prompt 06**, 2026-09-14, in that board's §4: all three of the audit's defects are fixed and the entry's own question — whether the fixed 500-point grid is adequately defined — is answered with 6.807e-11 / 3.237e-15 / 1.765e-16 at a tunable 3,000 / `k=5`. The row is deleted from `docs/OPEN_ISSUES.md` |
| `[19-cosmologymodels-docstrings-predate-per-sector-policy]` | GkTk-remedial | prompt 07 | **Closed by prompt 07**, 2026-09-14, in that board's §4. Both texts now say the kind is the consumer's choice, taken on measurement; four further texts that described the knot lattice as part of what the cosmology declares were corrected in the same pass (`GenericEOS.py`'s `BREAK_POINT_*` block and `break_temperatures_GeV`, `BackgroundModel._cosmology_break_points`, three paragraphs of `numeric_with_phase_cut.py`). The row is deleted from `docs/OPEN_ISSUES.md` |

Re-measured but **not owned** here (they stay where they are; a prompt that moves one says so):

| Issue | Owning board | Touched by |
|---|---|---|
| `[02-qcd-reference-floor]` | GkTk-remedial | prompts 02, 04, 05, 06 — the QCD references are regenerated and stop being circular |
| `[03-qcd-short-baseline-reference-endpoint-rounding]` | GkTk-remedial | prompts 02, 06 |
| `[20-wkb-gauss-orders-not-in-lookup-key]` | GkTk-remedial | prompt 04 — `RESIDUAL_WKB_REGION_MARGIN`'s reason for existing is measured, not changed |
| `[20-wkb-rows-consume-numeric-initial-data]` | GkTk-remedial | prompt 03 — the same class of defect one level up; prompt 03 does not fix that one |

---

## 4. Resolved issues

- **[01-convergence-block-has-a-separate-generator]** *(prompt 02, 2026-09-14; **assigned
  2026-09-16 to `prompts/tolerance-convergence` prompt 04 and closed by its prompt 04b**,
  2026-09-18)* — the `convergence` block of `ComputeTargets/tests/wkb_reference_data.json` is
  **regenerated and in the tree**: `generated` **2026-09-18**, `campaign`
  `prompts/tolerance-convergence (prompt 04, board item T7)`, `schema_version` 2, and
  `decision.recommended_scheme` **`branch`** — the scheme production can execute — against the
  `branch+knots` this entry records, which named a knot set prompt 07 of this campaign removed.
  `branch+knots` survives as a populated **control** because test modules index it by name; prompt
  04 measured that it buys nothing (`branch` 1.658e-16 on QCD's $\tau$ floor against the control's
  3.224e-16), which **confirms prompt 07 at the quadrature level**. The four orders come back
  $N_\tau = N_{c_s\tau} = N_F = N_\rho = 4$, now measured on the corrected $T(z)$ representation
  and the 3-point break set. Written by `docs/gktk-remedial/residual_convergence.py` with no flags
  and never by hand, in 538.4 s; only that one key of the fixture differs from the parent commit.

  **All three loosened constants this entry tracked are paid.** `QCD_BREAK_POINT_ALIGNMENT_TOL`
  goes **1.5e-04 → 3.0e-14**, against measured offsets of 3.552714e-15, 7.105427e-15 and
  **1.421085e-14** — 1, 2 and 4 ulp of $u$ — so the whole of the excess was the block's age, as
  this entry maintained through four prompts and as prompts 06 and 07 each re-measured. The two
  `QCD_FLOOR_FACTOR`s are **gone**: prompt 04 found that regenerating the block takes the recorded
  floors 58× and 43× tighter (1.878541e-14 → 3.223619e-16, 1.886653e-14 → 4.354138e-16) while the
  quantities scored against them do not move (2.254e-15, 2.212e-15), so the *construction* rather
  than the figure was what failed. The user chose the repair
  (`prompts/tolerance-convergence` README §7 **D8**): each test now bounds its own production
  quantity absolutely, at **1.0e-14**, against the double-precision accumulation floor measured at
  2.16e-16 ($\tau$) and 3.30e-16 ($c_s\tau$), with ×4.4 and ×4.5 of headroom.

  Declined on scope by prompts 08 and 09 of this campaign, exactly as those entries record, and
  closed by the first prompt anywhere given the fixture **and** both test modules. Measurements:
  `docs/tolerance-convergence/ORDER-AUDIT.md` §§1, 9, **12**, and
  [`prompts/tolerance-convergence/logs/04b-regenerate-the-convergence-block.md`](../tolerance-convergence/logs/04b-regenerate-the-convergence-block.md).
  `[02-qcd-reference-floor]` on the `GkTk-remedial` board waited on the same run; the run has now
  happened and its figures are in the tree, but that issue is not this prompt's and stays open on
  its own board.

- **[07-t-photon-range-logic-recomputes-its-bounds]** *(prompt 06, 2026-09-14; **assigned
  2026-09-16 and closed by `prompts/background-solver-robustness` prompt 05**, 2026-09-16)* —
  `TemperatureRepresentation.__call__`, `ZSplineWrapper.__call__` and
  `GkWKBSplineWrapper.__call__` each evaluated `_outward(bound, ±1)` on **every call**, twice for
  the comparisons and twice more inside the two f-strings, although all four bounds are fixed at
  construction. Prompt 05 hoisted **four** values per class — `_reject_above_log_z` and
  `_reject_below_log_z` (the `log(1+z)` thresholds the comparisons use) and `_recommended_max_z`
  and `_recommended_min_z` (the raw-`z` bounds the messages quote) — into each `__init__`, with a
  comment at each site saying they are loop invariants of a hot path. `_outward` itself,
  `SPLINE_BOUND_SLACK`, the `type(self).__name__` prefix `f023eb8` settled, the message text, the
  segment dispatch and the spline order and node count are untouched.

  **Demonstrated numerically null, not argued.** Eight wrapper constructions spanning the three
  classes — four `ZSplineWrapper` geometries including the negative `min_z` that is
  `test_spline_wrappers.py`'s discriminating case, two `GkWKBSplineWrapper`, and
  `TemperatureRepresentation` on the production `QCD_Cosmology` and on the pure-radiation stand-in
  — probed over the full tabulated range of each plus both slack bands and points outside them,
  through both entry points: **3,979 returned values bit-identical by `float.hex()`**, 37 in-probe
  rejections identical in position, and all **six** `RuntimeError` messages character-identical by
  diff. Independently, the whole of `measure_T_z_representation.py`'s output above §6 — §0's
  accuracy table, §1, §2, §3, §4's $H(z)$ and $\int\mathrm{d}z/H$ figures and §5 — is
  **byte-identical across all ten timing runs at both trees**. The cost it buys is the row below.
  Full figures in that campaign's [log 05](../background-solver-robustness/logs/05-hoist-the-range-logic.md).

- **[06-t-photon-call-cost-needs-a-quiet-machine]** *(prompt 06, 2026-09-14; **assigned
  2026-09-16 and closed by `prompts/background-solver-robustness` prompt 05**, 2026-09-16)* —
  README §6.2 and prompt 06 §4 set `T_photon` at **≤ 2.5 µs/call**, and the row was a
  **confirmed miss at 2.596 µs** (mean, range 2.505–2.671 over five runs on a quiet machine after
  prompt 06, restated by prompt 09) — about 3.8 % over, with the whole of the excess in the order-5
  `BSpline.__call__` and order 5 not optional. This entry's own next step was to hoist the two
  loop-invariant `_outward` calls of `[07-…]` and re-measure, with a predicted landing at ~2.49 µs.

  **Measured after the hoist, and it clears.** Same instrument
  (`docs/qcd-background-audit/measure_T_z_representation.py` §6, the 200-point probe set, best of
  3, the *same file* at both trees), five runs at each tree alternating, on a settled machine
  (load average 2.83–3.88, 10 cores, no orphaned busy shells):

  | Row | pre-hoist mean | range | post-hoist mean | range | ratio |
  |---|---|---|---|---|---|
  | **shipped spline** (`T_photon`) | **2.6744** | 2.532–2.856 | **2.4854** | 2.418–2.546 | **0.9293** |
  | entropy factor, 2000 pts *(control)* | 2.1716 | 2.081–2.263 | 2.1608 | 2.110–2.206 | 0.9950 |
  | segmented entropy factor *(control)* | 2.3412 | 2.233–2.474 | 2.2960 | 2.226–2.340 | 0.9807 |
  | accurate root solve *(control)* | 8.5056 | 8.139–8.860 | 8.5332 | 8.404–8.629 | 1.0032 |

  (µs/call.) The three controls — none of which goes through `TemperatureRepresentation.__call__`
  — move by −0.50 %, −1.93 % and +0.32 %, well inside per-tree spreads of 4–10 %, so the run
  stands by its own quietness test. A first ten-run pass at load average 4.5–5.3 had the
  entropy-factor control moving +2.0 % and **is void**; no figure from it is used.

  **$\bar t = 2.4854$ µs ≤ 2.5 µs, so this row closes**, and
  `prompts/background-solver-robustness` README §7 **D4 is not invoked** — the row does not go to
  the user. Two caveats, recorded because the margin is small and the next person to measure should
  not be surprised: the margin is **0.6 %**, and one of the five post-hoist runs (2.546) is above
  target on its own; and this machine reads **+3.0 % high** on this instrument (its pre-hoist
  2.6744 is the same code as this entry's 2.596), so the ratio 0.9293 is the more portable figure.
  The saving, **0.189 µs**, is 1.7× the 0.11 µs `[07-…]` costed, because what was removed is two
  Python function calls rather than two arithmetic operations.

- **[09-audit-script-section-5-prose-counts-the-wrong-set]** *(prompt 07, 2026-09-14; **assigned
  2026-09-16 and closed by `prompts/background-solver-robustness` prompt 05**, 2026-09-16)* —
  `docs/qcd-background-audit/measure_T_z_representation.py` §5 printed *"Of the BREAK_POINT_ALL
  points, 2411 are knots of the T(z) spline itself"*, having computed the tabulation's interior
  knots inside the production range instead of the intersection with the declared set. Prompt 05,
  which re-ran the script for its own §6 cost row and so had the file in scope, replaced
  `knots[(knots > u_nodes.min()) & (knots < u_nodes.max())]` with
  `np.intersect1d(all_points, knots)`, `all_points` being
  `integration_break_points(z_lo, z_hi, kind=BREAK_POINT_ALL)` — the declared set the sentence is
  about. It now prints

  ```
     0 of those 3 BREAK_POINT_ALL points are also knots of the T(z)
     tabulation. A representation that forces its own interpolation lattice into the
     break-point set contributes knots here, which are an artefact of how T(z) is
     approximated and not a feature of the cosmology; a representation that does not,
     contributes none, leaving only the genuine crossings of section 2.
  ```

  which is **0**, as prompt 07 made it, and reads correctly at 0 as well as at a positive count.
  The rewritten sentence also drops the description of the tabulation as "a uniform lattice of an
  auxiliary 500-point interpolant", which has been wrong since prompts 05 and 06 made it a
  segmented entropy factor at 3,000 nodes of order 5; it quotes no node count at all, so it cannot
  go stale the same way (that campaign's log 05, deviation D2). **§5's table above it and the whole
  of §0–§4 print byte-identically** to the unedited script — verified by diffing the two scripts'
  full output, where the only difference is these five lines against the four they replace.

- **[08-temperature-crossing-solver-is-test-only]** *(prompt 07, 2026-09-14; **assigned
  2026-09-16 and closed by `prompts/background-solver-robustness` prompt 04**, 2026-09-16)* —
  `LambdaCDM_GenericEOS._temperature_crossing_log1pz` had no production caller since prompt 07 took
  `integration_break_points` onto the bisected `_break_point_crossings_log1pz`, and a private method
  on a production class whose only callers were tests was a maintenance trap. Prompt 04 moved it,
  whole, to `CosmologyModels/tests/T_z_reference.temperature_crossing_log1pz` as a module-level
  function taking the cosmology as its first argument — its `root_scalar` line is
  character-identical, `xtol=1e-15, rtol=1e-15` untouched. `ComputeTargets/tests/
  test_numeric_break_points.py`'s one caller (`:514`) now imports it from
  `CosmologyModels.tests.T_z_reference` (option (a) of the prompt's open choice: the import works
  cleanly under both `discover -s ComputeTargets/tests -t .` and `-s CosmologyModels/tests -t .`,
  confirmed by running both). The docstring's opening sentence now states plainly that this is test
  machinery kept for two reasons — the neighbourhood probe and the illustration of the trap — rather
  than "nothing in production calls this", and a closing sentence records that it lived on
  `LambdaCDM_GenericEOS` until this campaign moved it. `LambdaCDM_GenericEOS.py`'s
  `_build_break_point_crossings_log1pz` docstring, which contrasted bisection against "a
  `root_scalar` bracket ... there", now names the new home instead of an implicit self-reference.
  **Nothing numeric moved:** `test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink`'s
  three measured steps are unchanged (1.969955e-03 at `T_LO` against the test's 1.97e-3, 9.27e-11 at
  `EOS_T_LO` against its `< 1e-8`, 1.377111e-04 at `T_120_MEV` against 1.38e-4), `CosmologyModels`
  tests 38 → 38 OK and `ComputeTargets` tests 447 → 447 OK, and `T_Z_REPRESENTATION_VERSION` stayed
  **6**. Full record:
  [`logs/04-relocate-the-crossing-probe.md`](../background-solver-robustness/logs/04-relocate-the-crossing-probe.md)
  on the `background-solver-robustness` board.

- **[03-qcd-inventory-does-not-report-the-representation]** *(prompt 03, 2026-09-14; **the
  `QCD_Cosmology` half closed by `prompts/background-solver-robustness` prompt 07**, 2026-09-16;
  **the `BackgroundModel` half prompt 14 widened it with closed by `prompts/store-fingerprint`
  prompt 03**, 2026-09-25, **which closes the issue**. Until then that half stayed open in §3 under
  the same ID; its §3 text is kept below)* — `sqla_QCDCosmology_factory.inventory()`
  (`Datastore/SQL/ObjectFactories/QCD_Cosmology.py:150`) now selects and reports
  `T_z_representation`, placed immediately before `log10_max_z` rather than appended at the end:
  the two of them describe how a row's background was computed and over what range, which
  `name`/`omega_m`/`omega_cc`/`h` do not. `tools/inventory_report.py` needed no change — it renders
  every `values` entry generically by `str()`, naming no column — confirmed by reading it rather
  than assumed. **Demonstrated, not argued:** a new test,
  `ComputeTargets/tests/test_qcd_cosmology_inventory.py` (3 tests), builds an in-memory SQLite
  table from the factory's own `register()` (the pattern
  `ComputeTargets/tests/test_cosmology_representation_key.py` already uses for this same factory)
  and inserts two rows sharing every other reported field but differing only in
  `T_z_representation`: they now render as two distinct label dicts, `{4, 6}`, rather than one
  indistinguishable pair. `build()`, the lookup key and the table schema are untouched; neither is
  in the diff. **The `BackgroundModel` half is out of scope** — prompt 07's files-may-touch list
  named only `QCD_Cosmology.py` and `tools/inventory_report.py`, and its own stop condition treats a
  second reporting site with the same gap as a new issue to record, not fix — so it remains open in
  §3 above, narrowed to that half and unassigned. Full record:
  [`logs/07-inventory-representation.md`](../background-solver-robustness/logs/07-inventory-representation.md)
  on the `background-solver-robustness` board.

  **Widened by prompt 14 (2026-09-15), and this half is still open.** The same defect exists one
  level out, on `BackgroundModel`: `sqla_BackgroundModelFactory.inventory()` reports labels and
  timestamps bucketed by `validated`, and says nothing about `source_grid_digest` or
  `source_grid_construction`. From prompt 14 a datastore can legitimately hold several background
  rows for the same cosmology and tolerances differing **only** in the grid they were tabulated on,
  and `tools/inventory_report.py` will render them as indistinguishable duplicates — precisely the
  confusion the columns exist to remove. Same severity and same reason for not acting: prompt 14 §2
  item 3 is about the key, and `inventory()` is not part of it.

  **`prompts/background-solver-robustness` prompt 07 (2026-09-16) declined to extend to this half**:
  its files-may-touch list named `Datastore/SQL/ObjectFactories/QCD_Cosmology.py` and
  `tools/inventory_report.py` only, not `BackgroundModel.py`, and its own stop condition (§4 item 3
  of that prompt) is explicit that a second reporting site with the same gap is recorded, not fixed,
  in the prompt that finds it. **Next step:** add `source_grid_digest` and `source_grid_construction`
  to `sqla_BackgroundModelFactory.inventory()`'s per-bucket report, in whichever prompt next has
  `Datastore/SQL/ObjectFactories/BackgroundModel.py` in scope. Not assigned to any campaign.

  **Assigned (2026-09-24):** [`prompts/store-fingerprint`](../store-fingerprint/README.md) prompt
  03. That campaign replaces every factory's `inventory()` with a structured record whose key
  carries each class's full lookup identity, `BackgroundModel`'s `source_grid_digest` and
  `source_grid_construction` included (its prompt 02). Its prompt 03 moves the report onto those
  records, which closes this half.

  **Closed (2026-09-25) by `prompts/store-fingerprint` prompt 03**, in its commit *"Move the
  inventory report and run labels onto one read-only service"*. Every factory's `inventory()` is
  retired, `sqla_BackgroundModelFactory.inventory()` and `sqla_QCDCosmology_factory.inventory()`
  included, and `main.py --inventory` renders the structured inventory of that campaign's prompt 02
  (`tools/inventory_report.format_inventory_report`). A record there is rendered by its whole
  physical key, so a `BackgroundModel` record shows `source_grid_digest` and
  `source_grid_construction`, and a `QCD_Cosmology` record shows `T_z_representation`, directly
  after `log10_max_z`. **Tested** by `Datastore/tests/test_inventory_report.py`
  `TestTheRepresentationAndTheGridIdentity` (the grid identity on two background models, and on a
  lone one, and the representation). Removing `source_grid_digest` from the display is that
  prompt's deliberate breakage (ii), which those tests catch. Two stores differing only in a
  background's grid now render differently. **The `QCD_Cosmology` half's test was re-expressed
  under that campaign's decision D4 (2026-09-25):** `ComputeTargets/tests/test_qcd_cosmology_inventory.py`
  tested the retired method and is deleted. `ComputeTargets/tests/test_qcd_cosmology_inventory_record.py`
  replaces its three tests. Two rows differing only in `T_z_representation` are two records and
  render differently, and the representation sits beside `log10_max_z`. The third claim changes
  meaning: two rows differing only in `name` are one key held twice, named a `duplicate`. Full
  record: [`logs/03-one-inventory-service.md`](../store-fingerprint/logs/03-one-inventory-service.md)
  on the `store-fingerprint` board.

- **[12-source-grid-density-is-uniform-over-a-curvature-that-spans-eight-orders]** *(prompt 12,
  2026-09-15; **the user decided to take the recommendation, and prompt 15 implemented it**,
  2026-09-15)* — `main.py` built one universal source grid whose density was a uniform
  `source_samples_per_log10z = 100`, a command-line number, over a $\varphi$ curvature spanning
  **eight orders inside a single Liouville–Green band**. Measured against the phase residual itself
  (verification document §10.2), the consumer's cubic was **under-resolved by 7.86× and 7.84× the
  storage floor** in the top decade of the $T_k$ band at $k=10^5$ on LambdaCDM and QCD, and
  **over-resolved by up to 2.1e+19** at the bottom of the range. The criterion that fixes it,
  $h(u)^4|\varphi''''(u)|/384 \le \varepsilon$ with
  $\varphi'(u) = -(1+z)C(z)/(\omega+\omega_0)$, is **computable before the grid exists** — it needs
  $H$, $c_s^2$, their $z$-derivatives and $k$, and no `BackgroundModel` — and predicts the realised
  error to **±2 %** in the $T_k$ sector on both models at all three reference wavenumbers.
  Equidistributing $\varphi$ itself is refuted (114,281 samples and still two misses). Prompt 12
  measured it and **changed nothing**, which is what its §4 asked for; the decision was the user's,
  taken with a full regeneration of eight stored object types in front of them.

  **The decision, verbatim** (prompt 15 §1): *"This is a science code. We want to use compute
  resource sensibly, efficiently, and without extravagance, but not take any short cuts or risks:
  there is no reward for doing so. The only thing we get is an unreliable published result. I would
  much rather have the grid slightly more dense than needed, than have it underdense."* So the
  **cap-1× column** was taken and the measured 1.75×/2.06× saving at cap 2× **declined**: all of it
  is coarsening at low $z$, where the derivative-fit padding clamps and where three consumers the
  criterion cannot see — the numeric ODE's samples, the four `CumulativeTable`s' Gauss panels and
  `QuadSourceIntegral`'s abscissae — all live.

  **Closed by prompt 15.** `build_z_sample` takes a `spacing` profile (two arrays; still no
  cosmology object and no equation-of-state import in `CosmologyConcepts/`) and subdivides every
  base interval the criterion finds too wide, in the base grid's own coordinate, under
  `SOURCE_GRID_MAX_SPACING_FACTOR = 1.0` — **never coarser than the uniform lattice puts there,
  anywhere**, measured at exactly **1.0000×** on both models, so the grid is a strict superset of
  prompt 11's. The profile comes from `main.source_grid_spacing_profile` over **all fifty**
  production wavenumbers and both sectors, in 0.9 s (LambdaCDM) and 1.8 s (QCD) against a QCD
  `compute_background`'s 0.599 s. QCD **1,773 → 1,996** (`303f9ce7` → `a2c32f67`), LambdaCDM
  **1,732 → 1,778** (`0960e169` → `60a3205a`), response grids **156 → 175** and **145 → 149**, both
  still decimations. In the production configuration the two rows that missed read **7.86 → 0.69**
  and **7.84 → 0.11** of their floors; ten of twelve rows improve or are unchanged and two move
  1.2× the wrong way at 1e-05 ulp. `SOURCE_GRID_CONSTRUCTION_VERSION` is **2** and
  `T_Z_REPRESENTATION_VERSION` stays **6**: no background value moves, only which redshifts are
  sampled.

  **Two things did not go as the recommendation predicted, and both are recorded.** The sample
  counts came out **larger** than §10.5's cap-1× column (1,996 and 1,778 against 1,761 and 1,634)
  because that column's cap is a single scalar, `max(diff(u)) = 2.3032e-02`, over a lattice uniform
  in $\log_{10}z$ rather than in $u$ — so it coarsens the bottom two decades by up to 10.9×, which
  is how "1,634" came to be fewer than the 1,732 it may not undercut. And the criterion at
  $\varepsilon$ = the per-case target does **not** recover the two rows on its own: the
  interpolating cubic's not-a-knot **end condition** raises the realised error constant to 9.9× /
  4.3× / 1.2× the interior one over the outermost three intervals of a band, which is exactly where
  the misses are, and prompt 12's bisected scale had absorbed it silently (8.053e-10 against a
  7.4506e-09 target). `SOURCE_GRID_SPLINE_EDGE_INTERVALS = 3` and
  `SOURCE_GRID_SPLINE_EDGE_FACTOR = 10.0` carry it; the gap that leaves is
  `[15-the-edge-factor-is-applied-at-the-band-edge-not-at-the-consumers-own-end]` in §3. Full
  record: [`logs/15-equidistribute-the-source-grid.md`](logs/15-equidistribute-the-source-grid.md);
  the measurement is
  [`docs/qcd-background-audit/equidistributed_grid_check.py`](../../docs/qcd-background-audit/equidistributed_grid_check.py).

- **[11-background-model-not-keyed-on-the-source-grid]** *(prompt 11, 2026-09-15; **closed by
  prompt 14**, 2026-09-15)* — `sqla_BackgroundModelFactory.build()` filtered on
  `(cosmology_type, cosmology_serial, atol_serial, rtol_serial)` plus whatever tags it was given,
  and `main.py` gave it `LargestSourceZTag`, `SmallestSourceZTag` and `SourceSamplesPerLog10ZTag`
  — **all three unchanged when the grid's shape changes**, because the endpoints and the
  samples-per-decade do not change — while the factory never filtered on `z_sample` at all. So a
  pre-prompt-11 datastore served its **1,732-node** background for the new **1,773-node** QCD grid:
  the *one surviving row* in a store whose every compute target the grid-tag change had already
  invalidated, which made it worse than a miss.

  **The fix is a content digest and a construction version, as columns.** `source_grid_digest`
  (`z_sample.digest()`, prompt 11's `blake2b` over the grid's exact bits) and
  `source_grid_construction` (`CosmologyConcepts.wavenumber.SOURCE_GRID_CONSTRUCTION_VERSION`) are
  registered on the `BackgroundModel` table, written by `store()` and filtered on by `build()` —
  both read from their single declarations, **never as literals**, asserted by `ast` in the shape
  of prompt 03's `test_the_filter_is_not_a_literal`. The two are not substitutable and the log says
  why: a digest cannot be inverted or range-queried, and two cosmologies in one run legitimately
  have different grids at the same construction.

  **The prompt-11 "next step" was `SourceZGridSizeTag` on the lookup, and that is *not* what
  shipped.** A tag would have keyed the background on the grid's *label*; a column keys it on the
  grid's *values*, which is the thing that was wrong, and it also lets the read path — which
  passes `z_sample=None` and therefore cannot filter on a digest it does not have — **select** the
  two columns and refuse a mixture, naming every `(construction, digest)` it found, instead of
  picking one. That asymmetry is the closure's substance: the compute path cannot write into a
  pre-prompt-14 store (a `RuntimeError` naming the campaign, "regenerated", "no migration"), and
  the read path still reads one, reporting it as an **unknown generation**, because a superseded
  datastore keeps its archival value. `check_zsample` still has no callers and is now redundant
  rather than merely unused; nothing was deleted. Record:
  [`logs/14-key-the-source-grid-construction.md`](logs/14-key-the-source-grid-construction.md).

- **[12-background-derivative-fit-grid-rings-at-a-step]** *(prompt 12, 2026-09-15; **closed by
  prompt 13**, 2026-09-15)* — `BackgroundModel` splined background quantities at **two** sites and
  neither was split at `integration_break_points`: `compute_background._build_derivative`'s order-5
  fit of $\log H$ over a padded, 3× refined copy of the source grid, and
  `_create_functions._build_func`'s cubic through the stored samples. $H$ genuinely **steps** at two
  of the three crossings `QCD_Cosmology` declares, so `epsilon` and its derivatives — and therefore
  $\omega_{\rm eff}$, and therefore every stored phase — rang there: **2.04e-02** relative at
  `T_LO` and **1.03e-03** at `T_120_MEV` on the production grid, against **3.9e-09 max / 8.1e-10
  median** away from a crossing and a control at `EOS_T_LO` of 1.6e-09.

  **Both sites now fit one spline per branch, and neither is redundant.** On the production grid the
  two genuine steps read **1.931e-09** and **4.070e-09** — inside the away-from-a-crossing regime —
  and the control is **1.609e-09** before and after, the same float. Segmenting the derivative fit
  alone carries `T_LO` the whole way but leaves **1.91e-04** at `T_120_MEV`, where the crossing sits
  in the QCD crossover and $\epsilon$ runs from 1.9460 to 1.9101 over 0.1 in $u$; the stored cubic
  is what closes that, a further 47,000×. Dispatch is `bisect_right` on $u$, never on a recovered
  $z$, and a branch too narrow for `order + 1` nodes is **refused** rather than silently dropped to
  a lower order or fitted across the step.

  **What the closure also established.** *The consumer:* the production configuration of
  `docs/qcd-background-verification.md` §10.4's four-way table goes **2.4859e-05 → 5.8437e-08 rad**
  at QCD $T_k$, $k=10^5$ (425×, against the ≤ 3.9744e-07 the prompt asked for), it improves at every
  (sector, $k$) scored, **the background's grid has stopped mattering** (the four cells agree to four
  digits where they spanned 300×), and the maximum has left the crossing altogether. *The kind:*
  `BREAK_POINT_ALL` rather than `BREAK_POINT_DISCONTINUITY`, on measurement —
  `d_wPerturbations_dz` at `EOS_T_LO` is **346 % wrong** without the third crossing and 7.752e-04
  with it, a factor of 4,462, which is the clamp in $w$ that crossing was declared for; the same
  quantity was 8.2 % and 3.7 % wrong at the two genuine steps and is now 1.1e-06 and 3.5e-07, a
  second defect nothing in the campaign had measured. *The cost:* integrand evaluations **exactly
  unchanged** (7,100 per cumulative table), wall time −1.3 %. *The boundary:* `LambdaCDM`,
  `LambdaCDM` with its analytic derivatives hidden, `LambdaCDM_GenericEOS(PureRadiationEOS)` and
  `RadiationModel` are **byte-identical** to `21d80b2` across 101,110 `float.hex()` lines. Record:
  [`logs/13-segment-background-derivative-splines.md`](logs/13-segment-background-derivative-splines.md).

- **[10-consumer-phi-unresolved-at-the-eos-crossing]** *(prompt 10, 2026-09-15; **closed by
  prompt 11**, 2026-09-15)* — the production source grid did not resolve $\varphi$ at `QCD_EOS`'s
  `T_LO` crossing ($u = 17.565806941870026$), which cost the consumer **8.1062e-06 rad**
  ($G_k$, 34.11 ulp of the span) and **1.3982e-05 rad** ($T_k$, 1876.61 ulp) at $k=10^5$ against
  1.00 ulp in ten of the twelve production rows. **It does now.** `build_z_sample` gives each
  declared crossing a pair of samples straddling it at a quarter of a grid interval *and* refines
  the eleven intervals from $-5$ to $+5$ around it by two — **41 extra samples in 1,732, 2.37 %** —
  and the two rows read **3.8296e-07 rad (1.61 ulp)** and **4.9012e-07 rad (65.78 ulp)**, both
  inside prompt 10 §5's 1e-06 rad target, $G_k$ inside its 2-ulp one, and both better than the
  campaign base as well as than `HEAD` (21.2× and 28.5×).

  **The entry's own design point held, and it decided the design.** Its warning that *refining the
  crossing's own interval is not enough* was re-measured on the grid actually built: the straddling
  pair alone — which is all prompt 11 §2 item 2 asks for — buys **1.96× and 1.98×** (17.43 and
  949.24 ulp) and leaves both rows outside the target, while the ±5 × 2 neighbourhood alone reaches
  1.64 / 74.60 and the two together 1.61 / 65.78. The feature is three to five grid intervals wide,
  as prompt 10 said.

  **What the closure also established.** The standoff is a *fraction of the grid spacing* (1/4 of an
  interval), scored from 1/2 down to 1e-4 of one: below ~1/32 the pair saturates 1.5×–1.7× **worse
  than no pair at all**, because the slope it implies drowns in
  `[02-consumer-phi-below-the-storage-granularity]`'s storage granularity — so
  `numeric_with_phase_cut.BREAK_POINT_STANDOFF = 1e-12` is the right number for an ODE restart and
  the wrong one for a sample. All three wavenumbers were scored in both sectors: $k=10^7$ and
  $3\times10^8$ are unmoved, with one row **+0.84 %** the wrong way (QCD $T_k$ at $3\times10^8$,
  1.27 → 1.28 ulp, away from the crossing), against the 1.70× the $C^0$ knot schemes moved it.
  LambdaCDM's grid is bit-identical. Record:
  [`logs/11-cosmology-aware-source-grid.md`](logs/11-cosmology-aware-source-grid.md) and
  [`docs/qcd-background-verification.md`](../../docs/qcd-background-verification.md) §9.

- **[02-fixture-tests-pinned-to-todays-break-point-artefact]** *(prompt 02, 2026-09-14; **closed by
  prompt 07**, 2026-09-14)* — the two assertions that hard-coded the knot-lattice era of
  `BREAK_POINT_ALL`. `test_background_tau.py::test_qcd_break_points` had already been re-sourced by
  prompt 06, which made it false rather than merely stale; prompt 07 rewrote it again to expect the
  **3** crossings and to assert directly that the intersection with the tabulation's 2,411 interior
  knots is empty. `test_numeric_break_points.py::test_kind_selects_knots_or_jumps`'s
  `assertGreater(len(every), 100)` became `assertEqual(len(every), 3)` and the case is renamed
  `test_kind_selects_the_kink_or_only_the_jumps`, since the two kinds now differ by one kink
  (`EOS_T_LO`, where $w$ changes analytic form but $g_s$ does not step) rather than by a lattice.
  Two new standing guards were added beside it: `test_no_declared_break_point_is_a_knot` (finding G1
  as an assertion) and `test_the_declared_points_are_prompt_06s_segment_edges` (the declared points
  against prompt 06's 17-digit handover, transcribed rather than re-derived). Record:
  [`logs/07-rederive-break-points.md`](logs/07-rederive-break-points.md).

- **[05-break-point-set-grew-with-the-node-count]** *(prompt 06, 2026-09-14; **closed by prompt
  07**, 2026-09-14)* — `BREAK_POINT_ALL` was 2,414 on the production source grid because prompt 06
  needed 3,000 tabulation nodes to reach README §6.1's p90 and median and every interior knot was
  declared. **All 2,411 are gone**: the set is the 3 equation-of-state crossings, median spacing
  215× the grid where it was 0.67×. The cost the entry measured came back with them — the `rho`
  residual table is **1.002×** the order×intervals baseline against the 2.337× it recorded, so
  `COST_BREAK_POINT_FACTOR` is **1.01**, below the 1.30 it started at rather than merely back to
  it, which is what the entry's "next step" predicted; the QCD `BackgroundModel` cumulative build
  takes **20,822** `Hubble` evaluations and 0.608 s against the 40,110 and 0.972 s it takes with
  the knots restored in the same process (1.93× and 1.61×), and the tau table's own counter goes
  16,580 → **6,936** against LambdaCDM's break-free 6,924. The node count is now free of the
  break-point set and can be raised on accuracy alone, which is recorded beside
  `DEFAULT_T_Z_SPLINE_SAMPLES`. Record:
  [`logs/07-rederive-break-points.md`](logs/07-rederive-break-points.md).

- **[02-qcd-T-z-spline-node-tolerance]** *(GkTk-remedial, opened 2026-09-10; **closed by prompt
  04**, 2026-09-14)* — T2 at its root. `_solve_T_z`'s `root_scalar(xtol=1e-6, rtol=1e-4)` is now
  `xtol=1e-300, rtol=1e-14`; the node solve is bit-identical to the `rtol=1e-14` defining-equation
  reference on the audit's probe set (max error 2.496e-05 → 0.0). Full record, including the
  re-measured $\omega^2/\omega_0^2$ scatter (essentially unchanged — this issue is not its cause),
  is on the `GkTk-remedial` board's §4 and in
  [`logs/04-tighten-node-solve.md`](logs/04-tighten-node-solve.md).

---

## 5. Standing notes for implementers

1. **A test that passes both before and after proves nothing.** README §0.2. The error this
   campaign removes is common mode between every producer and every consumer in the tree. **The
   harness exists as of prompt 01**: `CosmologyModels/tests/T_z_reference.py` (the reference) and
   `CosmologyModels/tests/test_T_z_representation.py` (seven cases). Every later prompt is scored
   against it, and every threshold in it is a named module constant carrying the prompt that
   tightens it.
2. **Never use the shipped `_solve_T_z` as a reference.** It is one of the things being measured.
   The reference is `T_z_reference.accurate_T(cosmology, z, rtol=1.0e-14)` — the defining equation
   (README §2 (a)).
3. **Bisect for a segment edge; never root-find on $T(z)-T_{\rm break}$.** README §2 (b). Use
   `T_z_reference.jump_locations(cosmology)`; its three values on the production QCD model, to 17
   digits, are in log 01's handover. Prompt 01 measured the trap: a `root_scalar` bracket on
   $T(z)-T_{\rm break}$ reports `converged=True` and returns a **non-root** whose relative residual
   is 8.844e-06 — 1.2 % of the jump height — because the difference changes sign inside one ulp of
   $u$ without passing through zero. Its position depends on the tolerance (`+1.126e-12`,
   `+3.304e-13` or `+1.421e-14` in $u$), and the first of those is larger than the `pad = 1e-12` the
   audit's segmented build uses, which is how a first attempt left the full 5.7e-04 error in place.
4. **The jump height is 7.6229229003969e-04 in $F$ and 7.625829e-04 in $T$.** The **7.614e-04** in
   audit §2's prose and README §2 (b) is §1's *linearised* $-\tfrac13\Delta g_s/g_s$: correct, but a
   different quantity, 0.15 % away. Log 01 deviation 2.
5. **The improved representation is cheaper per call**, not more expensive: 2.19–2.21 µs against
   2.26–2.44 µs. A reported regression means something other than the measured design was built.
6. **Every number this campaign moves was invisible to the datastore's lookup key** until prompt
   03 landed. That is why prompt 03 came before prompt 04 and not after. **As of prompt 03 the key
   can see it, but only if the prompt that moves the number bumps
   `LambdaCDM_GenericEOS.T_Z_REPRESENTATION_VERSION`** (`LambdaCDM_GenericEOS.py:73`) in the same
   commit. Bumping it is the whole of what is required and is required of every one of 04, 05, 06
   and 07; the column, the filter and the insert all read that single declaration, and
   `ComputeTargets/tests/test_cosmology_representation_key.py` fails if the link is broken.
7. **The QCD half of `wkb_reference_data.json` is built from the shipped `T(z)`** and moves with it
   (README §2 (e)). Ten test modules assert against it. Prompt 02 is the map.
8. **`BREAK_POINT_ALL` was load-bearing; prompt 07 changed what it *is* and prompt 08 measured
   that it no longer is.** `GkTk-remedial` prompt 19 measured that the $T_k$ numeric sector needed
   it: 3 of 50 QCD wavenumbers missed the criterion with jumps alone, worst 1.97e-07. That
   measurement was taken against knots carrying a $10^{-4}$-level defect. As of prompt 07 the set
   is 3 points, not 2,414, and the two kinds differ by a single kink (`EOS_T_LO`).
   **Prompt 08 re-took the measurement and found state (a)**: both policies converge at all 50
   wavenumbers (7.08e-09 / 8.85e-09 worst, zero offenders), the per-sector distinction is
   vestigial on today's cosmology, and the wider policy costs +0.99 % rather than +220 %.
   **Neither `BREAK_POINT_KIND` has been touched**, both are still in a datastore lookup key, and
   moving one is still a production decision with a regeneration attached, now quantified: up to
   2.86e-04 of the envelope on every stored QCD $T_k$ object (README §2 (f), §7 D5 — the user's,
   and log 08 recommends leaving both where they are). **What is still load-bearing is the
   *jumps*:** suppress the declaration altogether and 19 of 50 QCD $T_k$ wavenumbers go above the
   criterion, worst 9.61e-06.
9. **LambdaCDM has no `T(z)` spline** — `CosmologyModels/LambdaCDM/LambdaCDM.py:129` returns
   $T_{\rm CMB}(1+z)$ in closed form and the class declares no break points. **Measured across the
   whole campaign** by prompt 09, base against close: `LambdaCDM(Planck2018)` (5,005 `float.hex`
   lines, MD5 `157b1c61436106ba52058ddf3533d2df`) and `RadiationModel` (2,002 lines, MD5
   `265454eb751f9503a965f204e9e68bf4`) are **byte-identical**, and no LambdaCDM value moves in any
   table of `docs/gktk-remedial-verification.md`. **`LambdaCDM_GenericEOS(PureRadiationEOS)` is the
   one exception and is meant to be**: it declares no break points but it *does* build a `T(z)`
   representation, and on a constant-$g_s$ equation of state the new one is **exact** rather than
   merely accurate (README §2 (g), prompt 05's acceptance test) — against the closed form it goes
   2.270323e-07 / 1.904049e-07 / 1.006789e-07 → **3.330669e-16 / 2.220446e-16 / 0.0**. Do not read
   §0.5's "every stand-in number must be bit-identical" as covering that model; read it as covering
   the models that take the unchanged code path entirely.

10. **The campaign's verification document is
    [`docs/qcd-background-verification.md`](../../docs/qcd-background-verification.md)**, written by
    prompt 09 and **additive thereafter**: a later re-measurement appends a section and rewrites
    nothing, because what is written was correct for the tree it was taken on (`CLAUDE.md`). The
    same holds for `docs/gktk-remedial-verification.md`, which now carries a §9 and whose §§1–7 and
    §8 must not be touched.

11. **Two consumer facts a later reader will otherwise get backwards.** (i) §3.5 and §3.6 of
    `docs/gktk-remedial-verification.md` score a consumer against a producer built from the **same**
    background, so a background error cancels in them and those numbers were never the acceptance
    test for T1; prompt 01's guard is. (ii) Correcting the background made the QCD $k=10^5$
    consumer rows **4.3× worse**, and that is not a regression: the old representation smeared the
    equation of state's genuine step across ~4 production grid intervals, and the corrected one
    delivers 99.84 % of a step 2.78× taller inside one. `docs/qcd-background-verification.md` §3.3
    and `docs/qcd-background-audit/consumer_break_point_profile.py`.

12. **README §7 D5 is settled, and the decision is the user's, taken 2026-09-15.** Both
    `BREAK_POINT_KIND` values **stay as they are** — `TkNumericIntegration.BREAK_POINT_ALL` and
    `GkNumericIntegration.BREAK_POINT_DISCONTINUITY` — which is what prompt 08 recommended: keeping
    the wider policy costs **+0.99 %** in right-hand-side evaluations, while changing it would move
    every stored QCD $T_k$ value by up to 2.86e-04 of the envelope and demand a full regeneration.
    Recorded by prompt 13, which changed no value. This is not reopened by a later prompt without
    the user saying so.

13. **Prompt 12's density recommendation was taken, by the user, and prompt 15 implemented it**
    (2026-09-15). It is `[12-source-grid-density-is-uniform-over-a-curvature-that-spans-eight-orders]`,
    now in §4, and the cap that shipped is the **1× column**:
    `SOURCE_GRID_MAX_SPACING_FACTOR = 1.0`, "never coarser than the uniform lattice puts there,
    anywhere", with the measured 1.75×/2.06× saving at cap 2× **declined** on the user's
    instruction. Raising that constant does not by itself coarsen anything — the construction only
    ever subdivides — so revisiting the decision means writing a code path that drops base samples,
    which is deliberate. The constraint prompt 13 added,
    `[13-segmenting-costs-accuracy-on-a-grid-that-does-not-resolve-the-crossing]`, was **priced and
    did not bind**: the shipped grid is a strict superset of the one that entry was measured on.
