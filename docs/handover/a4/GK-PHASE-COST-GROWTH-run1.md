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

**Provenance.** 60 records; driver SHA-256 ['70dd520be4ddef238745c13dbdb4ade272d4b55c35d855e054a9b37ec99a95f5']; a4_cells SHA-256 ['8d7dc6ecd618ea9263438f4dff2d5ac09a6b324fde6b00b64798bf0f9b78711d']; launch git_head ['28ebaf0743e0555fb2c4512ce714884005a60eff'], dirty [False]; per-cell tree_head ['28ebaf0743e0555fb2c4512ce714884005a60eff']; per-cell `git status --short` [''].

## 1. The readings

Quoted from prompt §4, as copied to `state/readings.md`:

> ## 4. The readings, fixed before the run
>
> There is no threshold. The measure is the **growth across the ladder** in each shape: the region
> count at the highest rung reached over the count at 980, for each arm, set beside arm E's. The
> level at 980 is reported apart from it. Copy this section into `state/readings.md` unchanged before
> the launch.
>
> - **Reading 1:** if arm P's region count is flat in $x$, like arm E's, the growth was the
>   fixture's $G$, and `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` closes.
> - **Reading 2:** if arm P's still grows, the $G$ construction is not the cause. The issue is
>   narrowed, and the cause is looked for elsewhere.
> - **The attribution:** S against P₀ measures what the `phase_spline` floor adds; P₀ against P
>   measures what the anchoring floor adds. Each is stated per shape, as a ratio of growths, with the
>   group that carries the deepest regions.
> - **If the numbers support neither reading cleanly,** say so. An example would be P growing by a
>   factor between E's and S's, or the shapes disagreeing. Give the numbers, and leave the reading to
>   the user. Do not choose a reading after seeing the numbers.

`state/readings.md` was written at **15:26:53 UTC**. The run's manifest was written at
**15:27:04 UTC**, 11 s later. The run ended at about 16:07:47 UTC (its last heartbeat).

## 2. The tables

### 2.1 Regions and maximum depth

**Table A -- Levin regions / maximum depth, per cell and arm.**

| shape | x_resp | x = k tau | E | S | P0 | P |
|---|---|---|---|---|---|---|
| together | 980 | 1.5560e+03 | 7 / 0 | 1254 / 12 | 53 / 6 | 46 / 5 |
| together | 10000 | 1.5877e+04 | 8 / 1 | 4943 / 19 | **raised** | **raised** |
| together | 100000 | 1.5877e+05 | 8 / 1 | 34606 / 20 | **skipped** | **skipped** |
| together | 1e+06 | 1.5877e+06 | 9 / 2 | 50678 / 20 | **skipped** | **skipped** |
| together | 1e+07 | 1.5877e+07 | 9 / 2 | 58532 / 20 | **skipped** | **skipped** |
| together | 1e+08 | 1.5877e+08 | 9 / 2 | 8526 / 20 | **skipped** | **skipped** |
| T-first | 980 | 1.4145e+02 | 7 / 0 | 716 / 11 | 34 / 7 | 34 / 7 |
| T-first | 10000 | 1.4434e+03 | 10 / 2 | 3050 / 17 | **raised** | **raised** |
| T-first | 100000 | 1.4434e+04 | 9 / 1 | 11410 / 20 | **skipped** | **skipped** |
| T-first | 1e+06 | 1.4434e+05 | 12 / 3 | 12148 / 20 | **skipped** | **skipped** |
| T-first | 1e+07 | 1.4434e+06 | 13 / 4 | 33570 / 20 | **skipped** | **skipped** |
| q-smooth | 980 | 1.5431e+03 | 17 / 8 | 839 / 11 | **raised** | **raised** |
| q-smooth | 10000 | 1.5746e+04 | 21 / 6 | 8528 / 18 | **skipped** | **skipped** |
| q-smooth | 100000 | 1.5746e+05 | 21 / 6 | 88593 / 20 | **skipped** | **skipped** |
| q-smooth | 1e+06 | 1.5746e+06 | 16 / 3 | 33752 / 20 | **skipped** | **skipped** |

### 2.2 Growth per shape, and the level at 980

**Table B -- growth across the ladder:** regions at the highest rung reached over regions at 980, per shape and arm (prompt 04 section 4's measure). `rungs` is the rungs the arm has `ok` on that shape.

| shape | arm | rungs ok | regions at 980 | highest rung ok | regions there | growth |
|---|---|---|---|---|---|---|
| together | E | 6 of 6 | 7 | 1e+08 | 9 | 1.29 |
| together | S | 6 of 6 | 1254 | 1e+08 | 8526 | 6.8 |
| together | P0 | 1 of 6 | 53 | 980 | 53 | 1 |
| together | P | 1 of 6 | 46 | 980 | 46 | 1 |
| T-first | E | 5 of 5 | 7 | 1e+07 | 13 | 1.86 |
| T-first | S | 5 of 5 | 716 | 1e+07 | 33570 | 46.9 |
| T-first | P0 | 1 of 5 | 34 | 980 | 34 | 1 |
| T-first | P | 1 of 5 | 34 | 980 | 34 | 1 |
| q-smooth | E | 4 of 4 | 17 | 1e+06 | 16 | 0.941 |
| q-smooth | S | 4 of 4 | 839 | 1e+06 | 33752 | 40.2 |
| q-smooth | P0 | 0 of 4 | **raised** | -- | -- | -- |
| q-smooth | P | 0 of 4 | **raised** | -- | -- | -- |

**Table C -- the level at 980:** regions per arm over arm E's.

| shape | E | S | S/E | P0 | P0/E | P | P/E |
|---|---|---|---|---|---|---|---|
| together | 7 | 1254 | 179 | 53 | 7.57 | 46 | 6.57 |
| T-first | 7 | 716 | 102 | 34 | 4.86 | 34 | 4.86 |
| q-smooth | 17 | 839 | 49.4 | **raised** | -- | **raised** | -- |

### 2.3 `integral_time` with the load

**Table D -- `integral_time` (s) with the 1-minute load average at the cell's start and end.** Up to four cells ran at once; times are indicative only.

| shape | x_resp | arm | integral_time | setup_time | load start | load end |
|---|---|---|---|---|---|---|
| together | 980 | E | 0.57 | 0.023 | 6.67 | 6.67 |
| together | 980 | S | 19.01 | 0.128 | 6.69 | 7.43 |
| together | 980 | P0 | 1.01 | 0.133 | 6.67 | 6.69 |
| together | 980 | P | 0.91 | 0.139 | 6.67 | 6.67 |
| together | 10000 | E | 0.48 | 0.060 | 6.67 | 6.67 |
| together | 10000 | S | 72.25 | 0.167 | 8.36 | 8.21 |
| together | 10000 | P0 | **raised** | -- | -- | -- |
| together | 10000 | P | **raised** | -- | -- | -- |
| together | 100000 | E | 0.54 | 0.072 | 6.69 | 6.69 |
| together | 100000 | S | 529.22 | 0.184 | 8.03 | 6.52 |
| together | 100000 | P0 | **skipped** | -- | -- | -- |
| together | 100000 | P | **skipped** | -- | -- | -- |
| together | 1e+06 | E | 0.58 | 0.065 | 6.64 | 6.64 |
| together | 1e+06 | S | 766.41 | 0.217 | 6.40 | 6.56 |
| together | 1e+06 | P0 | **skipped** | -- | -- | -- |
| together | 1e+06 | P | **skipped** | -- | -- | -- |
| together | 1e+07 | E | 0.64 | 0.070 | 7.07 | 7.07 |
| together | 1e+07 | S | 885.35 | 0.258 | 6.91 | 4.74 |
| together | 1e+07 | P0 | **skipped** | -- | -- | -- |
| together | 1e+07 | P | **skipped** | -- | -- | -- |
| together | 1e+08 | E | 0.57 | 0.073 | 7.38 | 7.38 |
| together | 1e+08 | S | 127.76 | 0.279 | 5.24 | 5.06 |
| together | 1e+08 | P0 | **skipped** | -- | -- | -- |
| together | 1e+08 | P | **skipped** | -- | -- | -- |
| T-first | 980 | E | 0.23 | 0.026 | 6.67 | 6.67 |
| T-first | 980 | S | 7.81 | 0.120 | 7.07 | 7.43 |
| T-first | 980 | P0 | 0.52 | 0.127 | 6.64 | 7.07 |
| T-first | 980 | P | 0.50 | 0.125 | 6.64 | 6.64 |
| T-first | 10000 | E | 0.26 | 0.052 | 6.67 | 6.67 |
| T-first | 10000 | S | 43.67 | 0.152 | 8.36 | 9.37 |
| T-first | 10000 | P0 | **raised** | -- | -- | -- |
| T-first | 10000 | P | **raised** | -- | -- | -- |
| T-first | 100000 | E | 0.25 | 0.071 | 6.69 | 6.69 |
| T-first | 100000 | S | 172.27 | 0.186 | 9.34 | 12.90 |
| T-first | 100000 | P0 | **skipped** | -- | -- | -- |
| T-first | 100000 | P | **skipped** | -- | -- | -- |
| T-first | 1e+06 | E | 0.28 | 0.065 | 6.64 | 6.64 |
| T-first | 1e+06 | S | 182.93 | 0.218 | 12.90 | 8.25 |
| T-first | 1e+06 | P0 | **skipped** | -- | -- | -- |
| T-first | 1e+06 | P | **skipped** | -- | -- | -- |
| T-first | 1e+07 | E | 0.34 | 0.071 | 6.64 | 7.07 |
| T-first | 1e+07 | S | 508.13 | 0.255 | 7.99 | 6.79 |
| T-first | 1e+07 | P0 | **skipped** | -- | -- | -- |
| T-first | 1e+07 | P | **skipped** | -- | -- | -- |
| q-smooth | 980 | E | 0.22 | 0.026 | 6.67 | 6.67 |
| q-smooth | 980 | S | 7.21 | 0.130 | 7.43 | 8.36 |
| q-smooth | 980 | P0 | **raised** | -- | -- | -- |
| q-smooth | 980 | P | **raised** | -- | -- | -- |
| q-smooth | 10000 | E | 0.23 | 0.027 | 6.67 | 6.67 |
| q-smooth | 10000 | S | 117.52 | 0.148 | 8.97 | 10.93 |
| q-smooth | 10000 | P0 | **skipped** | -- | -- | -- |
| q-smooth | 10000 | P | **skipped** | -- | -- | -- |
| q-smooth | 100000 | E | 0.20 | 0.052 | 6.69 | 6.69 |
| q-smooth | 100000 | S | 1370.74 | 0.152 | 10.77 | 6.04 |
| q-smooth | 100000 | P0 | **skipped** | -- | -- | -- |
| q-smooth | 100000 | P | **skipped** | -- | -- | -- |
| q-smooth | 1e+06 | E | 0.21 | 0.070 | 6.69 | 6.69 |
| q-smooth | 1e+06 | S | 495.89 | 0.189 | 5.88 | 5.75 |
| q-smooth | 1e+06 | P0 | **skipped** | -- | -- | -- |
| q-smooth | 1e+06 | P | **skipped** | -- | -- | -- |

### 2.4 The cells not computed

No cell reached its 3 h budget, so **no cell is `not reached`**. Six `raised` and twenty were
`skipped` because a rung below them on the same shape and arm had raised.

**Table E -- cells not computed.** None is `not reached`: no cell hit its budget.

| cell | status | reason |
|---|---|---|
| P|q-smooth|980 | raised | RuntimeError: CumulativeTable[tau]: z=5.48046 lies more than one grid interval below the table (min node z=5.48046) |
| P|q-smooth|10000 | skipped | the rung below, P|q-smooth|980, is raised |
| P|q-smooth|100000 | skipped | the rung below, P|q-smooth|10000, is skipped |
| P|q-smooth|1e+06 | skipped | the rung below, P|q-smooth|100000, is skipped |
| P0|q-smooth|980 | raised | RuntimeError: CumulativeTable[tau]: z=5.48046 lies more than one grid interval below the table (min node z=5.48046) |
| P0|q-smooth|10000 | skipped | the rung below, P0|q-smooth|980, is raised |
| P0|q-smooth|100000 | skipped | the rung below, P0|q-smooth|10000, is skipped |
| P0|q-smooth|1e+06 | skipped | the rung below, P0|q-smooth|100000, is skipped |
| P|together|10000 | raised | RuntimeError: CumulativeTable[tau]: z=6.0696 lies more than one grid interval below the table (min node z=6.0696) |
| P0|together|10000 | raised | RuntimeError: CumulativeTable[tau]: z=6.0696 lies more than one grid interval below the table (min node z=6.0696) |
| P|together|100000 | skipped | the rung below, P|together|10000, is raised |
| P0|together|100000 | skipped | the rung below, P0|together|10000, is raised |
| P|together|1e+06 | skipped | the rung below, P|together|100000, is skipped |
| P0|together|1e+06 | skipped | the rung below, P0|together|100000, is skipped |
| P|together|1e+07 | skipped | the rung below, P|together|1e+06, is skipped |
| P0|together|1e+07 | skipped | the rung below, P0|together|1e+06, is skipped |
| P|together|1e+08 | skipped | the rung below, P|together|1e+07, is skipped |
| P0|together|1e+08 | skipped | the rung below, P0|together|1e+07, is skipped |
| P|T-first|10000 | raised | RuntimeError: CumulativeTable[tau]: z=6.0696 lies more than one grid interval below the table (min node z=6.0696) |
| P0|T-first|10000 | raised | RuntimeError: CumulativeTable[tau]: z=6.0696 lies more than one grid interval below the table (min node z=6.0696) |
| P|T-first|100000 | skipped | the rung below, P|T-first|10000, is raised |
| P0|T-first|100000 | skipped | the rung below, P0|T-first|10000, is raised |
| P|T-first|1e+06 | skipped | the rung below, P|T-first|100000, is skipped |
| P0|T-first|1e+06 | skipped | the rung below, P0|T-first|100000, is skipped |
| P|T-first|1e+07 | skipped | the rung below, P|T-first|1e+06, is skipped |
| P0|T-first|1e+07 | skipped | the rung below, P0|T-first|1e+06, is skipped |

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

**Table F -- A2's recorded curves, pre-B1, context only** (`REALISTIC-LARGE-X.md` Table 8: exact and realistic regions, closed seam, realistic integral seconds on the laptop), beside this run's E and S regions.

| shape | x_resp | A2 exact | A2 realistic | A2 realistic s | A4 E | A4 S |
|---|---|---|---|---|---|---|
| together | 980 | 7 | 1304 | 18.5 | 7 | 1254 |
| together | 10000 | 8 | 12068 | 173.4 | 8 | 4943 |
| together | 100000 | 8 | 27253 | 410.2 | 8 | 34606 |
| together | 1e+06 | 9 | 20253 | 299.2 | 9 | 50678 |
| together | 1e+07 | 9 | 50463 | 773.7 | 9 | 58532 |
| together | 1e+08 | 9 | 15827 | 241.9 | 9 | 8526 |
| T-first | 980 | 7 | 720 | 7.2 | 7 | 716 |
| T-first | 10000 | 10 | 3444 | 48.1 | 10 | 3050 |
| T-first | 100000 | 9 | 19375 | 310.7 | 9 | 11410 |
| T-first | 1e+06 | 12 | 36728 | 576.0 | 12 | 12148 |
| T-first | 1e+07 | 13 | 75570 | 1229.8 | 13 | 33570 |
| q-smooth | 980 | 17 | 855 | 6.8 | 17 | 839 |
| q-smooth | 10000 | 21 | 5018 | 62.4 | 21 | 8528 |
| q-smooth | 100000 | 21 | 31358 | 514.8 | 21 | 88593 |
| q-smooth | 1e+06 | 16 | not reached | -- | 16 | 33752 |

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
and above, one to four `Gqr` groups share depth 20. The deepest regions' $z'$-range moves up the
ladder with $x_{\rm resp}$: from $z' \approx 6$–$80$ at $10^5$ to $1.5\times10^4$–$2.8\times10^5$
at `together` $10^8$. In P₀ and P at 980 the deepest regions are near the hand-overs ($z' \approx
160$–$330$), not near $z_{\rm resp}$.

**Table G -- the depth record:** per cell, the group carrying the maximum depth (Levin sub-interval index : regime : group), its z-range of deepest regions, every group at that depth, and the phase-limited region count.

| shape | x_resp | arm | max depth | carrier | its regions | deepest regions | their z' range | all groups at max depth | phase-limited regions |
|---|---|---|---|---|---|---|---|---|---|
| together | 980 | E | 0 | 0:r:r | 1 | 1 | 332.333 - 361.369 | 0:r:r, 1:Gr:G+r, 1:Gr:G-r, 2:Gqr:G+q+r, 2:Gqr:G+q-r, 2:Gqr:G-q+r, 2:Gqr:G-q-r | 0 |
| together | 10000 | E | 1 | 2:Gqr:G+q+r | 2 | 2 | 6.0696 - 3080.37 | 2:Gqr:G+q+r | 1 |
| together | 100000 | E | 1 | 2:Gqr:G+q+r | 2 | 2 | 6.0696 - 30812.7 | 2:Gqr:G+q+r | 3 |
| together | 1e+06 | E | 2 | 2:Gqr:G+q+r | 3 | 2 | 1474.94 - 308136 | 2:Gqr:G+q+r | 3 |
| together | 1e+07 | E | 2 | 2:Gqr:G+q+r | 3 | 2 | 4666.33 - 3.08136e+06 | 2:Gqr:G+q+r | 4 |
| together | 1e+08 | E | 2 | 2:Gqr:G+q+r | 3 | 2 | 14758.4 - 3.08137e+07 | 2:Gqr:G+q+r | 4 |
| together | 980 | S | 12 | 2:Gqr:G-q-r | 414 | 14 | 6.41256 - 7.09004 | 2:Gqr:G-q-r | 11 |
| together | 10000 | S | 19 | 2:Gqr:G+q-r | 1247 | 2 | 6.58064 - 6.58081 | 2:Gqr:G+q-r, 2:Gqr:G-q+r | 69 |
| together | 100000 | S | 20 | 2:Gqr:G-q+r | 17058 | 7040 | 6.92528 - 79.7834 | 2:Gqr:G+q+r, 2:Gqr:G+q-r, 2:Gqr:G-q+r | 1324 |
| together | 1e+06 | S | 20 | 2:Gqr:G+q+r | 39886 | 25768 | 8.75887 - 443.228 | 2:Gqr:G+q+r, 2:Gqr:G-q+r | 1810 |
| together | 1e+07 | S | 20 | 2:Gqr:G+q+r | 42920 | 29946 | 184.824 - 4271.02 | 2:Gqr:G+q+r, 2:Gqr:G+q-r, 2:Gqr:G-q+r | 1286 |
| together | 1e+08 | S | 20 | 2:Gqr:G+q+r | 8445 | 1334 | 14781.9 - 283681 | 2:Gqr:G+q+r | 3363 |
| together | 980 | P0 | 6 | 2:Gqr:G+q-r | 10 | 2 | 163.726 - 184.099 | 2:Gqr:G+q-r | 2 |
| together | 980 | P | 5 | 1:Gr:G+r | 16 | 6 | 307.898 - 332.333 | 0:r:r, 1:Gr:G+r, 1:Gr:G-r | 1 |
| T-first | 980 | E | 0 | 0:r:r | 1 | 1 | 300.974 - 361.369 | 0:r:r, 1:qr:q+r, 1:qr:q-r, 2:Gqr:G+q+r, 2:Gqr:G+q-r, 2:Gqr:G-q+r, 2:Gqr:G-q-r | 0 |
| T-first | 10000 | E | 2 | 2:Gqr:G-q+r | 3 | 2 | 133.293 - 2550.02 | 2:Gqr:G-q+r | 0 |
| T-first | 100000 | E | 1 | 2:Gqr:G+q-r | 2 | 2 | 6.0696 - 25509.2 | 2:Gqr:G+q-r, 2:Gqr:G-q+r | 1 |
| T-first | 1e+06 | E | 3 | 2:Gqr:G-q+r | 4 | 2 | 18508 - 255101 | 2:Gqr:G-q+r | 3 |
| T-first | 1e+07 | E | 4 | 2:Gqr:G-q+r | 5 | 2 | 104083 - 515286 | 2:Gqr:G-q+r | 4 |
| T-first | 980 | S | 11 | 2:Gqr:G-q-r | 247 | 10 | 6.47464 - 6.63245 | 2:Gqr:G-q-r | 13 |
| T-first | 10000 | S | 17 | 2:Gqr:G-q-r | 1383 | 8 | 6.59101 - 7.12529 | 2:Gqr:G-q-r | 49 |
| T-first | 100000 | S | 20 | 2:Gqr:G+q-r | 3926 | 28 | 6.6751 - 8.96605 | 2:Gqr:G+q+r, 2:Gqr:G+q-r, 2:Gqr:G-q-r | 109 |
| T-first | 1e+06 | S | 20 | 2:Gqr:G-q+r | 6835 | 2798 | 6.38504 - 40.4948 | 2:Gqr:G+q-r, 2:Gqr:G-q+r | 753 |
| T-first | 1e+07 | S | 20 | 2:Gqr:G-q+r | 27688 | 16812 | 6.57405 - 403.163 | 2:Gqr:G+q+r, 2:Gqr:G-q+r | 1152 |
| T-first | 980 | P0 | 7 | 0:r:r | 24 | 4 | 307.866 - 330.661 | 0:r:r | 2 |
| T-first | 980 | P | 7 | 0:r:r | 24 | 4 | 307.866 - 330.661 | 0:r:r | 2 |
| q-smooth | 980 | E | 8 | 1:Gr:G-r | 10 | 2 | 16.3398 - 16.8814 | 1:Gr:G-r | 1 |
| q-smooth | 10000 | E | 6 | 1:Gr:G-r | 9 | 4 | 54.452 - 98.791 | 1:Gr:G-r | 0 |
| q-smooth | 100000 | E | 6 | 1:Gr:G-r | 9 | 4 | 553.52 - 996.91 | 1:Gr:G-r | 3 |
| q-smooth | 1e+06 | E | 3 | 0:G:G | 4 | 2 | 338949 - 561409 | 0:G:G, 1:Gr:G+r, 1:Gr:G-r | 4 |
| q-smooth | 980 | S | 11 | 1:Gr:G-r | 385 | 12 | 5.78461 - 9.25169 | 1:Gr:G-r | 12 |
| q-smooth | 10000 | S | 18 | 2:Gqr:G+q-r | 2799 | 2 | 6.61998 - 6.62007 | 2:Gqr:G+q-r, 2:Gqr:G-q-r | 61 |
| q-smooth | 100000 | S | 20 | 2:Gqr:G-q-r | 29337 | 14604 | 5.72801 - 73.314 | 2:Gqr:G+q+r, 2:Gqr:G+q-r, 2:Gqr:G-q+r, 2:Gqr:G-q-r | 4220 |
| q-smooth | 1e+06 | S | 20 | 2:Gqr:G+q+r | 14326 | 6026 | 12.5362 - 407.157 | 2:Gqr:G+q+r, 2:Gqr:G+q-r, 2:Gqr:G-q+r | 1544 |

## 5. The representation term, recorded, no verdict

$N_{\rm arm} - N_{\rm E}$ for S, P₀ and P per cell, each with both cells' declared errors. B2
re-takes it properly.

**Table H -- the representation term, recorded, no verdict:** N_arm - N_E at the same rung, both scored against the same 50-digit reference; `+/-` is the sum of the two cells' declared errors converted to N (A2's `term`).

| shape | x_resp | N_E + 9/8 | E declared | arm | N_arm + 9/8 | arm declared | N_arm - N_E | +/- |
|---|---|---|---|---|---|---|---|---|
| together | 980 | +6.994e-14 | 8.7e-12 | S | +3.470e-03 | 2.5e-08 | +3.470e-03 | 2.7e-08 |
| together | 980 | +6.994e-14 | 8.7e-12 | P0 | +3.622e-03 | 2.4e-08 | +3.622e-03 | 2.7e-08 |
| together | 980 | +6.994e-14 | 8.7e-12 | P | +3.623e-03 | 2.4e-08 | +3.623e-03 | 2.7e-08 |
| together | 10000 | +3.109e-15 | 1.3e-12 | S | -1.035e-04 | 2.6e-09 | -1.035e-04 | 3.0e-09 |
| together | 10000 | +3.109e-15 | 1.3e-12 | P0 | **raised** | -- | -- | -- |
| together | 10000 | +3.109e-15 | 1.3e-12 | P | **raised** | -- | -- | -- |
| together | 100000 | +1.723e-13 | 1.7e-11 | S | +1.495e-04 | 1.9e-09 | +1.495e-04 | 2.2e-09 |
| together | 100000 | +1.723e-13 | 1.7e-11 | P0 | **skipped** | -- | -- | -- |
| together | 100000 | +1.723e-13 | 1.7e-11 | P | **skipped** | -- | -- | -- |
| together | 1e+06 | -1.561e-12 | 1.4e-10 | S | -8.795e-05 | 1.7e-09 | -8.795e-05 | 2.0e-09 |
| together | 1e+06 | -1.561e-12 | 1.4e-10 | P0 | **skipped** | -- | -- | -- |
| together | 1e+06 | -1.561e-12 | 1.4e-10 | P | **skipped** | -- | -- | -- |
| together | 1e+07 | -3.101e-10 | 9.9e-09 | S | -1.090e-03 | 9.7e-09 | -1.090e-03 | 2.2e-08 |
| together | 1e+07 | -3.101e-10 | 9.9e-09 | P0 | **skipped** | -- | -- | -- |
| together | 1e+07 | -3.101e-10 | 9.9e-09 | P | **skipped** | -- | -- | -- |
| together | 1e+08 | -1.461e-10 | 3.5e-08 | S | -3.249e-04 | 3.9e-08 | -3.249e-04 | 8.4e-08 |
| together | 1e+08 | -1.461e-10 | 3.5e-08 | P0 | **skipped** | -- | -- | -- |
| together | 1e+08 | -1.461e-10 | 3.5e-08 | P | **skipped** | -- | -- | -- |
| T-first | 980 | -2.931e-14 | 4.8e-12 | S | -6.037e-05 | 3.5e-09 | -6.037e-05 | 3.9e-09 |
| T-first | 980 | -2.931e-14 | 4.8e-12 | P0 | -6.042e-05 | 3.5e-09 | -6.042e-05 | 3.9e-09 |
| T-first | 980 | -2.931e-14 | 4.8e-12 | P | -6.042e-05 | 3.5e-09 | -6.042e-05 | 3.9e-09 |
| T-first | 10000 | -2.220e-16 | 1.0e-12 | S | -1.334e-05 | 2.9e-09 | -1.334e-05 | 3.3e-09 |
| T-first | 10000 | -2.220e-16 | 1.0e-12 | P0 | **raised** | -- | -- | -- |
| T-first | 10000 | -2.220e-16 | 1.0e-12 | P | **raised** | -- | -- | -- |
| T-first | 100000 | +6.461e-14 | 1.6e-12 | S | -1.380e-05 | 3.0e-09 | -1.380e-05 | 3.4e-09 |
| T-first | 100000 | +6.461e-14 | 1.6e-12 | P0 | **skipped** | -- | -- | -- |
| T-first | 100000 | +6.461e-14 | 1.6e-12 | P | **skipped** | -- | -- | -- |
| T-first | 1e+06 | -2.310e-12 | 7.8e-11 | S | -3.087e-05 | 1.4e-09 | -3.087e-05 | 1.6e-09 |
| T-first | 1e+06 | -2.310e-12 | 7.8e-11 | P0 | **skipped** | -- | -- | -- |
| T-first | 1e+06 | -2.310e-12 | 7.8e-11 | P | **skipped** | -- | -- | -- |
| T-first | 1e+07 | +4.674e-12 | 3.1e-10 | S | -8.214e-06 | 5.5e-10 | -8.214e-06 | 9.6e-10 |
| T-first | 1e+07 | +4.674e-12 | 3.1e-10 | P0 | **skipped** | -- | -- | -- |
| T-first | 1e+07 | +4.674e-12 | 3.1e-10 | P | **skipped** | -- | -- | -- |
| q-smooth | 980 | -3.644e-13 | 5.2e-12 | S | -5.405e-05 | 9.9e-11 | -5.405e-05 | 1.2e-10 |
| q-smooth | 980 | -3.644e-13 | 5.2e-12 | P0 | **raised** | -- | -- | -- |
| q-smooth | 980 | -3.644e-13 | 5.2e-12 | P | **raised** | -- | -- | -- |
| q-smooth | 10000 | -1.088e-13 | 1.0e-11 | S | -9.654e-05 | 3.9e-10 | -9.654e-05 | 4.5e-10 |
| q-smooth | 10000 | -1.088e-13 | 1.0e-11 | P0 | **skipped** | -- | -- | -- |
| q-smooth | 10000 | -1.088e-13 | 1.0e-11 | P | **skipped** | -- | -- | -- |
| q-smooth | 100000 | -3.855e-13 | 2.1e-11 | S | -1.148e-06 | 3.0e-10 | -1.148e-06 | 3.6e-10 |
| q-smooth | 100000 | -3.855e-13 | 2.1e-11 | P0 | **skipped** | -- | -- | -- |
| q-smooth | 100000 | -3.855e-13 | 2.1e-11 | P | **skipped** | -- | -- | -- |
| q-smooth | 1e+06 | +2.003e-13 | 5.4e-10 | S | -1.049e-04 | 7.1e-10 | -1.049e-04 | 1.4e-09 |
| q-smooth | 1e+06 | +2.003e-13 | 5.4e-10 | P0 | **skipped** | -- | -- | -- |
| q-smooth | 1e+06 | +2.003e-13 | 5.4e-10 | P | **skipped** | -- | -- | -- |

**Realistic $T$ anchors.**

**Table I -- the realistic T anchors:** p_R per fixture (z_init off a node).

| shape | x_resp | p_R, T_q | p_R, T_r | anchor node, T_q | anchor node, T_r |
|---|---|---|---|---|---|
| together | 980 | -4.4496e-05 | +3.6282e-05 | None | None |
| together | 10000 | -4.3653e-06 | -3.6366e-06 | None | None |
| together | 100000 | -4.3650e-07 | -3.6367e-07 | None | None |
| together | 1e+06 | -4.3649e-08 | -3.6368e-08 | None | None |
| together | 1e+07 | -4.3648e-09 | -3.6368e-09 | None | None |
| together | 1e+08 | -4.3647e-10 | -3.6368e-10 | None | None |
| T-first | 980 | -4.4496e-05 | +3.6282e-05 | None | None |
| T-first | 10000 | -4.3653e-06 | -3.6366e-06 | None | None |
| T-first | 100000 | -4.3650e-07 | -3.6367e-07 | None | None |
| T-first | 1e+06 | -4.3649e-08 | -3.6368e-08 | None | None |
| T-first | 1e+07 | -4.3648e-09 | -3.6368e-09 | None | None |
| q-smooth | 980 | -4.4496e-03 | +3.9580e-05 | None | None |
| q-smooth | 10000 | -4.3606e-04 | -3.9672e-06 | None | None |
| q-smooth | 100000 | -4.3606e-05 | -3.9673e-07 | None | None |
| q-smooth | 1e+06 | -4.3616e-06 | -3.9674e-08 | None | None |

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

**Table J -- Phase objects, as every `ok` cell records them** (class / exact_remainder):

- E: G _ExactPhase/None, Tq _ExactPhase/None/None, Tr _ExactPhase/None/None, construction_ok True -- 15 cells
- P: G PrimitivePhase/True, Tq PrimitivePhase/True/TablePrimitive, Tr PrimitivePhase/True/TablePrimitive, construction_ok True -- 2 cells
- P0: G PrimitivePhase/False, Tq PrimitivePhase/True/TablePrimitive, Tr PrimitivePhase/True/TablePrimitive, construction_ok True -- 2 cells
- S: G phase_spline/None, Tq PrimitivePhase/True/TablePrimitive, Tr PrimitivePhase/True/TablePrimitive, construction_ok True -- 15 cells

*Tables generated by `scripts/a4_tables.py` from the run's checkpoint
(`var/runs/handover-04-cost-growth-20261026T152704/checkpoint.jsonl`).*
