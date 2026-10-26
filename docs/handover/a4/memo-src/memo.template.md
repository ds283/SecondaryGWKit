# Does A2's Levin cost growth survive production's phases? (handover A4)

**Status: STOPPED.** The run ended `failed`: 34 of 60 cells are `ok`, 6 `raised` and 20 were
`skipped`. Arms E and S are complete. Arms P₀ and P have two cells each, `together` and `T-first`
at 980. Every raised cell raises in the test fixture's own $G$ phase table (§2.4). The readings of
§1 **cannot be answered for arm P** from these cells (§3). How A4 proceeds is the user's decision.

## 0. The objects

| object | what it is | whose |
|---|---|---|
| tree | `handover-remedial` at `28ebaf0743e0555fb2c4512ce714884005a60eff` (`28ebaf0`), clean at every cell (`git status --short` empty, recorded per cell) | the dispatch SHA |
| host | the Mac Studio (mps026298), Python 3.13.16, Ray-free, no datastore, no lock | — |
| harness | `scripts/a4_cells.py` (SHA-256 `8d7dc6ec…`), a sibling of A2's `realistic_large_x.py` (`c1cd3598…`, imported, not edited). It replaces A2's clamp reads with `partition["handover"]` | this prompt |
| reference | eq. (22) minus the head, at 50 digits (`reference_of`); `REF_ATOL = 1e-45`, `REF_RTOL = 1e-12` | A2's |
| **E** | $G$ is `ExactGk`; $T_q$, $T_r$ are `ExactTkFunctions`; the exact flavour's seam | A2's exact flavour |
| **S** | $G$'s phase is `OffsetBesselPhaseGk`'s `phase_spline`, A2's construction | A2's $G$ |
| **P₀** | $G$'s phase is a `PrimitivePhase` on a `CumulativeTable` of $1/H$ with `exact_remainder` off (`_ProductionPhaseGk.old_phase`): production before `levin-phase-floor` 05 | test fixture |
| **P** | the same `PrimitivePhase`, built by `_build_phase` with `exact_remainder` on (`_ProductionPhaseGk.phase`): production since 05 | test fixture, production's builder |
| $T$ in S, P₀, P | `TkSourceFunctions(fixture.exact_envelope_model(), k, Tn, Tw)` with no keyword; a `PrimitivePhase` on a table `cs_tau` with `exact_remainder` on | production's construction |
| seam in S, P₀, P | production's since B1: the first Liouville–Green sample is dropped | production's |

**How the realistic $T$'s table is built.** It is built on the union of the fixture's numeric and
WKB nodes, without the one numeric node equal to $z_{\rm init}$. So $z_{\rm init}$ is off a node,
with $p_R \ne 0$ (Table I). This is the user's choice at A4's two stops. 05's
`_TableCsTauFixture` builds its table on the dropped WKB grid, which starts one step below
$z_{\rm init}$, and refuses $z_{\rm init}$ at rungs $10^4$–$10^6$ by one ulp of `log1p`.

**Only `.phase` differs between S, P₀ and P.** The policy's `GkSourceFunctions` is `_replace`d in
`phase` alone. Every `ok` cell records its phase objects by class and `exact_remainder`, and they
are what the arms say (Table J).

**Why A2's recorded cells are context, not a column.** They were computed before B1, on the clamp
seam, with the fixture's old $T$. Every cell here is computed at `HEAD` on one tree, so the arms
differ in $G$'s phase alone. Set beside A2's numbers, B1's change and $T$'s would be mixed in with
$G$'s.

## 1. The readings

Quoted from prompt §4, as copied to `state/readings.md`:

{{READINGS}}

`state/readings.md` was written at **15:26:53 UTC**. The run's manifest was written at
**15:27:04 UTC**, 11 s later. The run ended at about 16:07:47 UTC (its last heartbeat).

## 2. The tables

### 2.1 Regions and maximum depth

{{A}}

### 2.2 Growth per shape, and the level at 980

{{B}}

{{C}}

### 2.3 `integral_time` with the load

{{D}}

### 2.4 The cells not computed

No cell reached its 3 h budget, so **no cell is `not reached`**. Six `raised` and twenty were
`skipped` because a rung below them on the same shape and arm had raised.

{{E}}

**Why they raised** (`out/probe_tau_table_bottom.txt`, `out/probe_tau_table_margin.txt`; objects
only, no cell re-run).
- **The object.** The raising object is $G$'s `PrimitivePhase` leading primitive in P₀ and P: the
  `tau` `CumulativeTable` that `test_g_phase_reanchor._ProductionPhaseGk` builds on
  `sorted({z_grid} ∪ {z_resp})`.
- **The near-duplicate.** A4 passes `Case`'s own $G$ grid,
  `log_grid(1 + G_crossover_z, 1 + z_resp, …)`. Its last node is
  `exp(log(1 + z_resp)) - 1`, which at 13 of the 15 rungs is $z_{\rm resp}$ **minus one ulp**. At
  `together` $10^8$ and `q-smooth` $10^6$ it is equal. So the table holds two nodes one ulp apart
  in $z$, and both have **the same** $u = \log(1+z)$: its bottom interval is zero-wide in $u$.
- **Why it raises.** `CumulativeTable._nearest_node` refuses an abscissa "more than one grid
  interval" below the bottom node, so it refuses every $u$ even one ulp below it.
- **Which cells.** Whether a cell trips this depends on whether the Levin bisection forms an
  abscissa a little below $\log(1+z_{\rm resp})$. It did in six cells. It did not in
  `together` 980 and `T-first` 980, which have the same degenerate table.
- **Not a production construction.** Production's `functions.tau` is the background model's own
  table. Its response redshift is a node once, and the table extends below it. It is the A4 arm's
  construction (`_ProductionPhaseGk` on `Case`'s grid) through a test fixture. This is proposed as
  an issue in `issues.md`.

### 2.5 A2's recorded curves, labelled pre-B1

{{F}}

## 3. What the numbers support

**Neither reading can be answered for arm P from these cells.**
- **Arm P has no growth to measure.** Arm P is `ok` only at 980, on `together` and `T-first`; on
  `q-smooth` it raised at 980. Every rung above 980 raised or was skipped, so P's growth across
  the ladder (§1's measure) is not defined on any shape. The same holds for P₀. No reading is
  chosen.
- **The attribution cannot be formed.** "S against P₀" and "P₀ against P", as ratios of growths,
  need P₀'s and P's growths.

**The numbers that exist:**
- **Growth, E** (highest rung over 980): `together` 1.29 (7→9), `T-first` 1.86 (7→13), `q-smooth`
  0.94 (17→16).
- **Growth, S:** `together` 6.8 (1,254→8,526 at $10^8$; it peaks at 58,532 at $10^7$), `T-first`
  46.9 (716→33,570), `q-smooth` 40.2 (839→33,752 at $10^6$; it peaks at 88,593 at $10^5$).
- **Level at 980, regions over E's:**
  - S: 179 (`together`), 102 (`T-first`), 49.4 (`q-smooth`);
  - P₀: 7.57 and 4.86;
  - P: 6.57 and 4.86.

  So at the one rung where all four arms ran, P₀ and P sit 24–27× (`together`) and 21×
  (`T-first`) below S. That is a level, not a growth.
- **Maximum depth.** S reaches the cap of 20 from $10^5$ on every shape (from $10^4$ it is 17–19).
  E's maximum is 8. P₀ and P reach 5–7 at 980.

## 4. The depth record

Per arm and shape: the group carrying the deepest regions, where in $z'$ they lie, and the
phase-limited counts. In S, the deepest regions sit in the all-oscillatory sub-interval (`Gqr`) at
every rung above 980. At 980 on `together` and `T-first` they are in its `G-q-r` group. At $10^5$
and above, several `Gqr` groups share depth 20, with the deepest regions spread over
$z' \approx 6$–$4\times10^3$. Within the ladder, the larger $x_{\rm resp}$, the larger the
$z'$ they reach. In P₀ and P at 980 the deepest regions are near the hand-overs ($z' \approx
160$–$330$), not near $z_{\rm resp}$.

{{G}}

## 5. The representation term, recorded, no verdict

$N_{\rm arm} - N_{\rm E}$ for S, P₀ and P per cell, each with both cells' declared errors. B2
re-takes it properly.

{{H}}

**Realistic $T$ anchors.**

{{I}}

## 6. What this does not cover

- **$b = 0$ only.**
- **One seam:** production's, in S, P₀ and P; the exact flavour's in E.
- **The fixture's $T$ is production's in every realistic arm**, so $T$'s own floor is not measured
  here; 05a did that on A3 v2. Its table is the union table of §0, not 05's `_TableCsTauFixture`.
- **$x_{\rm resp} \le 10^8$**, not production's $4\times10^{12}$.
- **The producers' stored phase samples** (`levin-phase-floor` 06+), which the fixture does not
  model.
- **Arms P₀ and P above 980, and on `q-smooth` at all.** They raised or were skipped (§2.4), so
  this memo says nothing about how they grow.
- **Times are indicative only.** Up to four cells ran at once, at a 1-minute load of 4.7–12.9.

{{J}}

*Tables generated by `scripts/a4_tables.py` from the run's checkpoint
(`var/runs/handover-04-cost-growth-20261026T152704/checkpoint.jsonl`).*
