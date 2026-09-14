# Implementation state — the QCD background campaign

**Campaign:** [`README.md`](README.md) · **Source document:**
[`docs/qcd-background-audit-2026-09.md`](../../docs/qcd-background-audit-2026-09.md)
**Baseline commit:** `e8f746d` (`qcd-background-audit`, clean; identical to `main`)
**Last updated:** 2026-09-14 — **prompt 07 complete; 7 / 12. T1 and G1 are closed.** Twelve
prompts in four workstreams.
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
| 08 | [Re-measure the per-sector break-point policy](08-per-sector-policy-remeasure.md) | Opus | ⬜ | | |

### Workstream C — close-out (prompt 09)

| # | Prompt | Model | Status | Commit | Log |
|---|---|---|---|---|---|
| 09 | [The consumer tables under a corrected background](09-close-out-verification.md) | Opus | ⬜ | | |

### Workstream D — the source grid and the consumer spline (prompts 10–12) — **gated on README §7 D7**

| # | Prompt | Model | Status | Commit | Log |
|---|---|---|---|---|---|
| 10 | [`PrimitivePhase` on the 3-point break set](10-primitive-phase-break-points.md) | Opus | ⬜ | | |
| 11 | [A cosmology-aware source grid](11-cosmology-aware-source-grid.md) | Opus | ⬜ | | |
| 12 | [A measured grid-density criterion](12-grid-density-criterion.md) | Opus | ⬜ | | |

**Progress:** 7 / 12 complete (7 / 9 in the ungated chain 01–09).

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

**The representation version.** `T_Z_REPRESENTATION_VERSION` is introduced by prompt 03 and bumped
by **04, 05, 06 and 07**. Its value at each prompt boundary is recorded here as the campaign runs,
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

---

## 2. Mechanism-level tracking

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| **T1** | **DEFECT, critical** | Was 3.461e-08 relative in $\int\mathrm{d}z/H$ — of order 47.5 / 4.75e3 / 1.43e5 rad at $k=10^5/10^7/3\times10^8$ against 1-ulp floors of 3.05e-7 / 3.05e-5 / 9.15e-4 rad, **common mode** between producer and consumer and so invisible to every test in the tree. **Closed by prompt 06:** the guard reads **0.0** — the shipped background's $\int\mathrm{d}z/H$ is bit-identical to the exact background's at all 17 digits — and the equivalent phase is 0.000e+00 rad at all three wavenumbers. Prompt 09 scores the *consumers* under the corrected background (README §6.4); that is verification, not the fix. | 01, 04, 05, 06 | ⚠️ |
| **T2** | **DEFECT, high** | `_solve_T_z` root-solves each spline node to `rtol=1e-4`; neighbouring nodes carry uncorrelated errors up to 2.496e-05. The root of `[02-qcd-T-z-spline-node-tolerance]`, and (measured, not asserted) of the $\pm0.1$ scatter in $\omega^2/\omega_0^2$ that `RESIDUAL_WKB_REGION_MARGIN = 0.5` exists to survive (`ComputeTargets/phase_residual.py:220`). | 04 | ⚠️ |
| **T3** | **DEFECT, medium** | $T$ was splined against $u$, spending resolution on the $(1+z)$ ramp known in closed form. **Fixed by prompt 05:** `TemperatureRepresentation` splines $F(u)=\log(T/[T_{\rm CMB}(1+z)])$ and multiplies the ramp back in, improving the median 412× at the same 500 nodes (1.071e-07 → **2.599e-10**) and the p90 2.2× (1.936e-07 → **8.912e-08**) at *lower* cost per call (2.240 µs). Exact on a constant-$g_s$ equation of state (2.928e-16). | 05 | ⚠️ |
| **T4** | **DEFECT, high** | One global spline across three points at which $T(z)$ genuinely **jumps** (7.625829e-04 in $T$ at $z_c=4.25337\times10^7$). The max error was pinned near the jump height at 500, 2,000 and 5,000 nodes alike. **Fixed by prompt 06:** `SegmentedEntropyFactor` interpolates $F$ one spline per branch with the edges *bisected* onto the jumps, and the max falls **7.236e-04 → 6.807e-11** while the step itself is reproduced to six figures on both sides. An edge misplaced by one node restores 7.054e-04, and a test asserts that it does. | 06 | ⚠️ |
| **G1** | **DEFECT, high** | 404 of the 407 `BREAK_POINT_ALL` points were knots of the auxiliary interpolant — 2,411 of 2,414 after prompt 06's node count — a Gauss panel split every 0.67 grid intervals throughout `BackgroundModel`, and the sole cause of `prompts/phase-representation` prompt 02's Schoenberg–Whitney failure. **Fixed by prompt 07:** `integration_break_points` declares the equation of state's temperature crossings and nothing else, **3** and **2** on the production grid with **0** knots, measured rather than asserted (at order 5 the first discontinuous derivative of $F$ is the fifth, three levels below `d3_lnH_dz3`; the observable residual across a knot is 6.3e-12 in $H$ against 2.1e-04 for the old cubic lattice). The QCD build falls 16,580 → **6,936** integrand evaluations and 0.959 → **0.599 s**; the references do not move at all; `cs_tau` and `friction_F` score against them unchanged to every digit printed and `tau` moves 2.104e-15 → 2.254e-15, at 12 % of its floor; and a repeated-knot vector constructs on all six grids. Prompt 08 re-takes `GkTk-remedial` prompt 19's per-sector policy measurement against the new set. | 07, 08 | ⚠️ |
| **P2** | **DEFECT, accuracy** | Inherited `[13-consumer-spline-crosses-eos-break-points]`: `PrimitivePhase` splines $\varphi$ with default knots across the declared break points. 1.907e-6 rad ($G_k$, 8 ulp) and 3.186e-6 rad ($T_k$, 428 ulp) at $z=4.24\times10^7$. **Unblocked by prompt 07** — the knot vector now constructs on all six grids, asserted in `TestConsumerKnotVectorConstructs` — but the defect is untouched and whether a $C^0$ knot is the right representation for $\varphi$ is still open. | 10 | ⬜ |
| **G2** | **DESIGN** | The source grid never consults the cosmology: `populate_z_sample` is a bare `logspace`, `winnow` a blind stride `[::-n]`, and the tag `SourceRedshiftGrid_{len}` labels size only, so two different grids of equal length collide in the datastore. | 11, 12 | ⬜ |

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
- **[03-qcd-inventory-does-not-report-the-representation]** *(prompt 03, 2026-09-14)* —
  `sqla_QCDCosmology_factory.inventory()`
  (`Datastore/SQL/ObjectFactories/QCD_Cosmology.py`) reports `name`, `omega_m`, `omega_cc`, `h`
  and `log10_max_z` per row, and `tools/inventory_report.py:46` lists `QCD_Cosmology` among the
  tables it summarises. Neither shows the new `T_z_representation` column. **Impact:** from prompt
  04 onward a datastore can legitimately hold several QCD cosmology rows differing **only** in
  their representation — same name, same seven parameters, same `log10_max_z` — and the only tool
  that inspects a datastore will render them as indistinguishable duplicates, which is precisely
  the confusion the column exists to remove, moved one layer out. Low severity: no computation
  reads `inventory()`, and the lookup key itself is correct. **Next step:** add
  `"T_z_representation": row.T_z_representation` to the `values` list in `inventory()` and to the
  selected columns above it. Not done in prompt 03 because its §2 item 4 is explicit that the
  commit changes the key and nothing else, and `inventory()` is not part of the key. Worth doing
  before prompt 09, which is the first prompt likely to look at a datastore holding rows at two
  representations.

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

- **[06-t-photon-call-cost-needs-a-quiet-machine]** *(prompt 06, 2026-09-14)* — README §6.2 and
  prompt 06 §4 set `T_photon` at **≤ 2.5 µs/call** and make a regression a stop (README §2 (c)).
  Measured on a **quiet** machine before the dispatch was inlined: prompt 05's shape 2.21 µs,
  unsegmented 3,000 / `k=5` 2.40 µs, segmented 3,000 / `k=5` **2.53 µs** — a ~1 % miss. The
  dispatch was then moved in line into `TemperatureRepresentation.__call__`, which on a **loaded**
  machine (load average 11–15, every candidate reading ~20 % high) takes the shipped-to-unsegmented
  ratio from ~1.05 to **1.00–1.05** — i.e. the segmentation is now free — leaving the
  shipped-to-prompt-05 ratio at **1.09–1.13**, which is the order-5 spline evaluation and nothing
  else. Order 5 is not optional: a cubic needs ~25,000 nodes to reach the required p90. Scaling the
  quiet-machine 2.21 µs by the measured 1.09 gives ~2.4 µs, inside the target, but **that is an
  inference and not a measurement**. **Impact:** one row of prompt 06 §4 is unresolved; nothing
  downstream is affected, since `T_photon` costs ~2.5 µs inside a ~10 µs `Hubble` call.
  **Next step:** re-run the three-candidate `timeit` comparison in log 06 deviation 5 on a quiet
  machine and record the absolute figure; if it is above 2.5 µs, `[07-...]` below is 0.11 µs of it.

- **[07-t-photon-range-logic-recomputes-its-bounds]** *(prompt 06, 2026-09-14)* —
  `TemperatureRepresentation.__call__` (`LambdaCDM_GenericEOS.py:302`) evaluates
  `_outward(self._max_log_z, +1)` and `_outward(self._min_log_z, -1)` on **every call**. Both are
  loop-invariant: the bounds are set in `__init__` and never mutated. Measured at **0.056 µs each**
  on the quiet machine, i.e. ~0.11 µs of a ~2.5 µs call, and hoisting them into `__init__` is
  numerically null (the same two floats, compared the same way). `ZSplineWrapper.__call__`
  (`ComputeTargets/spline_wrappers.py:64`) has the same shape and is on many more hot paths.
  **Impact:** ~4 % of every `T_photon` call and of every wrapped spline evaluation in the tree.
  Not done in prompt 06 because the range logic is prompt 05's code and outside what prompt 06 was
  asked to change (README §5 rule 5). **Next step:** hoist both, in whichever prompt next has
  reason to touch that method; re-measure `[06-...]` afterwards.

- **[08-temperature-crossing-solver-is-test-only]** *(prompt 07, 2026-09-14)* —
  `LambdaCDM_GenericEOS._temperature_crossing_log1pz` (`:810`) has no production caller. Prompt 07
  took `integration_break_points` onto the bisected `_break_point_crossings_log1pz`, which is what
  README §2 (b) requires now that `T_photon` genuinely jumps at exactly these temperatures, and the
  `root_scalar` bracket that used to locate them is left behind. It is deliberately kept: it is the
  probe `ComputeTargets/tests/test_numeric_break_points.py::test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink`
  uses to find a *neighbourhood* of a crossing (a use its ~1e-12 offset does not disturb), it is
  cited by name in `CosmologyModels/tests/T_z_reference.py:45` for the redshift-arithmetic rule, and
  it is the documented illustration of the trap, sitting next to the bisector that replaced it. Its
  docstring now opens "Nothing in production calls this, and nothing may put it back on the
  break-point path." **Impact:** none numerically; a private method on a production class whose only
  callers are tests is a maintenance trap, and README §2 (b) makes the specific trap it embodies a
  stop condition. **Next step:** move it into `CosmologyModels/tests/T_z_reference.py` beside
  `jump_locations`, or delete it and have that one test bisect, for whichever prompt next has both
  files in scope. Not done in prompt 07 because `T_z_reference.py` is not among the files that
  prompt may touch.

- **[09-audit-script-section-5-prose-counts-the-wrong-set]** *(prompt 07, 2026-09-14)* —
  `docs/qcd-background-audit/measure_T_z_representation.py:401-405` prints, beneath its §5 table,
  "Of the BREAK_POINT_ALL points, {n} are knots of the T(z) spline itself". It never computes that
  intersection: `n` is `len(knots[(knots > u_min) & (knots < u_max)])`, the tabulation's interior
  knots inside the production range, which is **2,411** and has nothing to do with the declared set
  any more. The table above it is correct — `all 3 points in range`, `discontinuity 2 points in
  range` — so the script does reproduce prompt 07's result; only the sentence is wrong. It was true
  while the two sets coincided, which is the whole history of this script until this commit.
  **Impact:** a reader running the campaign's own reproduction is told that 2,411 of 3 points are
  knots. **Next step:** intersect `knots` with the declared points before printing, and reword.
  Not done in prompt 07: its §3 item 6 requires the script to run **unedited**, and the script is
  not among the files that prompt may touch.

Inherited, and **assigned to this campaign** (each is owned by the board named, which holds its
measurements and its history; the closure is recorded there):

| Issue | Owning board | Closed by | Note |
|---|---|---|---|
| `[13-consumer-spline-crosses-eos-break-points]` | GkTk-remedial | prompt 10 | Assigned 2026-09-13 by `prompts/phase-representation`'s close-out. **Unblocked and narrowed by prompt 07**, 2026-09-14, on that board: the multiplicity-`spline_order` knot vector now constructs on all six production grids and still fails on the old set, both asserted by `TestConsumerKnotVectorConstructs`. The defect itself is untouched — prompt 02's items 2–4 stand |
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
8. **`BREAK_POINT_ALL` was load-bearing, and prompt 07 changed what it *is*.** `GkTk-remedial`
   prompt 19 measured that the $T_k$ numeric sector needed it: 3 of 50 QCD wavenumbers missed the
   criterion with jumps alone. That measurement was taken against knots carrying a $10^{-4}$-level
   defect. **As of prompt 07 the set is 3 points, not 2,414**, and `BREAK_POINT_ALL` differs from
   `BREAK_POINT_DISCONTINUITY` by a single kink (`EOS_T_LO`) rather than by a knot lattice — so
   prompt 19's question narrows to that one crossing. Prompt 08 re-takes the measurement; neither
   `BREAK_POINT_KIND` has been touched, and moving one is still a production decision with a
   datastore regeneration attached (README §2 (f), §7 D5).
9. **LambdaCDM has no `T(z)` spline** — `CosmologyModels/LambdaCDM/LambdaCDM.py:129` returns
   $T_{\rm CMB}(1+z)$ in closed form and the class declares no break points. Every LambdaCDM,
   `RadiationModel` and stand-in number is bit-identical across this entire campaign. So is a
   `LambdaCDM_GenericEOS` built on a constant-$g_s$ equation of state, for which the new
   representation is **exact** rather than merely accurate (README §2 (g)).
