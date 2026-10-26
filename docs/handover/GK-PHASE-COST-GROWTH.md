# Does A2's Levin cost growth survive production's phases? (handover A4)

> **Committed on the laptop** in A4's records commit. The Mac Studio wrote this under `var/handover-04/`; every path below of the form `out/…`, `scripts/…`, `state/…`, `probes/…` or `*-run1*` is committed under [`docs/handover/a4/`](a4/). Checkpoints stay in the run directories on the Mac Studio.

**Status: complete.** Run 2, `handover-04-cost-growth-20261026T173720`, ended `done` with all 60
cells `ok`. None was `not reached`, none raised, none was skipped.

**What the numbers support:** neither reading cleanly (§3).
- **Arm P's growth lies between E's and S's on every shape:** 6.26 (`together`), 3.29 (`T-first`),
  2.16 (`q-smooth`), against E's 1.29 / 1.86 / 0.94 and S's 6.8 / 46.9 / 40.2.
- **Arm P₀,** the `PrimitivePhase` without the node anchor, grows by far the most: 8,930 / 2,360 /
  619.

## 0. The objects

| object | what it is | whose |
|---|---|---|
| tree | `handover-remedial` at `28ebaf0743e0555fb2c4512ce714884005a60eff` (`28ebaf0`), clean at every cell (`git status --short` empty, recorded per cell) | the dispatch SHA |
| host | the Mac Studio (mps026298), Python 3.13.16, Ray-free, no datastore, no lock | — |
| harness | `scripts/a4_cells.py` (SHA-256 `67badeab…`), a sibling of A2's `realistic_large_x.py` (`c1cd3598…`, imported, not edited). It replaces A2's clamp reads with `partition["handover"]` | this prompt |
| reference | eq. (22) minus the head, at 50 digits (`reference_of`); `REF_ATOL = 1e-45`, `REF_RTOL = 1e-12` | A2's |
| **E** | $G$ is `ExactGk`; $T_q$, $T_r$ are `ExactTkFunctions`; the exact flavour's seam | A2's exact flavour |
| **S** | $G$'s phase is `OffsetBesselPhaseGk`'s `phase_spline`, A2's construction | A2's $G$ |
| **P₀** | $G$'s phase is a `PrimitivePhase` on a `CumulativeTable` of $1/H$ with `exact_remainder` off (`_ProductionPhaseGk.old_phase`): production before `levin-phase-floor` 05 | test fixture |
| **P** | the same `PrimitivePhase`, built by `_build_phase` with `exact_remainder` on (`_ProductionPhaseGk.phase`): production since 05 | test fixture, production's builder |
| $T$ in S, P₀, P | `TkSourceFunctions(fixture.exact_envelope_model(), k, Tn, Tw)` with no keyword; a `PrimitivePhase` on a table `cs_tau` with `exact_remainder` on | production's construction |
| seam in S, P₀, P | production's since B1: the first Liouville–Green sample is dropped | production's |

**Two fixture adaptations, each the user's choice at a stop** (`log.md` deviations 9 and 14).
Neither edits a test file.
1. **$T$'s `cs_tau` table.** It is built on the union of the fixture's numeric and WKB nodes,
   without the numeric node equal to $z_{\rm init}$. So $z_{\rm init}$ is covered and off a node,
   with $p_R \ne 0$ (Table I). 05's `_TableCsTauFixture` does not reach $z_{\rm init}$ once the
   first WKB sample is dropped.
2. **$G$'s grid in P₀ and P.** `_ProductionPhaseGk` is passed `Case`'s own $G$ grid with its last
   node set exactly to $z_{\rm resp}$; it had been one ulp below at 13 of 15 rungs. Unchanged, the
   grid gives `_ProductionPhaseGk`'s `tau` table a zero-width bottom interval. Run 1 failed on that
   at six cells.

**Every cell checks its phases before its integral.** It evaluates the phase of every oscillatory
factor at the bottom of the range: at $a = \log(1+z_{\min})$, at $a$ minus 1–8 ulp, and at the
lowest abscissa the Levin driver can form there, 0–4 ulp below $a$. All 60 pass.

**Only `.phase` differs between S, P₀ and P**, and every cell records its phase objects by class and
`exact_remainder` (Table J).

**Run 1 is a record, not a column.** Run 1 (`handover-04-cost-growth-20261026T152704`, memo
`GK-PHASE-COST-GROWTH-run1.md`) ended `failed` on the zero-width interval: 34 `ok`, 6 `raised`, 20
`skipped`. Its 30 E and S cells are identical to run 2's in `total`, `total_abserr`, regions and depth. Its four P₀ and P cells at 980 match run 2's in `total` and regions; their `total_abserr` differs in the last digits, because the $G$ grid changed (`out/run1_vs_run2_ok_cells.txt`).

**Why A2's recorded cells are context, not a column.** They were computed before B1, on the clamp
seam, with the fixture's old $T$. Every cell here is computed at `HEAD` on one tree, so the arms
differ in $G$'s phase alone. Set beside A2's numbers, B1's change and $T$'s would be mixed in with
$G$'s.

**Provenance.** 60 records; driver SHA-256 ['70dd520be4ddef238745c13dbdb4ade272d4b55c35d855e054a9b37ec99a95f5']; a4_cells SHA-256 ['67badeab27e49e0f97a709ca67f8911d9663e0d8763e789c5946415cc9e64798']; launch git_head ['28ebaf0743e0555fb2c4512ce714884005a60eff'], dirty [False]; per-cell tree_head ['28ebaf0743e0555fb2c4512ce714884005a60eff']; per-cell `git status --short` [''].

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

Run 2's `state/readings.md` was written at **17:37:09 UTC**. The run's manifest was written at
**17:37:21 UTC**, 12 s later. The run ended `done` at about 21:20:08 UTC (its last heartbeat).
Run 1's copy, `state/readings-run1.md` (15:26:53 UTC, before run 1's manifest at 15:27:04), has
identical content.

## 2. The tables

### 2.1 Regions and maximum depth

**Table A -- Levin regions / maximum depth, per cell and arm.**

| shape | x_resp | x = k tau | E | S | P0 | P |
|---|---|---|---|---|---|---|
| together | 980 | 1.5560e+03 | 7 / 0 | 1254 / 12 | 53 / 6 | 46 / 5 |
| together | 10000 | 1.5877e+04 | 8 / 1 | 4943 / 19 | 78 / 8 | 66 / 6 |
| together | 100000 | 1.5877e+05 | 8 / 1 | 34606 / 20 | 1551 / 19 | 80 / 8 |
| together | 1e+06 | 1.5877e+06 | 9 / 2 | 50678 / 20 | 56451 / 20 | 102 / 9 |
| together | 1e+07 | 1.5877e+07 | 9 / 2 | 58532 / 20 | 308667 / 20 | 313 / 14 |
| together | 1e+08 | 1.5877e+08 | 9 / 2 | 8526 / 20 | 473552 / 20 | 288 / 13 |
| T-first | 980 | 1.4145e+02 | 7 / 0 | 716 / 11 | 34 / 7 | 34 / 7 |
| T-first | 10000 | 1.4434e+03 | 10 / 2 | 3050 / 17 | 44 / 7 | 44 / 7 |
| T-first | 100000 | 1.4434e+04 | 9 / 1 | 11410 / 20 | 62 / 7 | 51 / 7 |
| T-first | 1e+06 | 1.4434e+05 | 12 / 3 | 12148 / 20 | 607 / 17 | 45 / 7 |
| T-first | 1e+07 | 1.4434e+06 | 13 / 4 | 33570 / 20 | 80117 / 20 | 112 / 9 |
| q-smooth | 980 | 1.5431e+03 | 17 / 8 | 839 / 11 | 687 / 11 | 686 / 11 |
| q-smooth | 10000 | 1.5746e+04 | 21 / 6 | 8528 / 18 | 1540 / 14 | 1191 / 14 |
| q-smooth | 100000 | 1.5746e+05 | 21 / 6 | 88593 / 20 | 31814 / 20 | 1204 / 14 |
| q-smooth | 1e+06 | 1.5746e+06 | 16 / 3 | 33752 / 20 | 425137 / 20 | 1480 / 14 |

### 2.2 Growth per shape, the attribution, and the level at 980

**Table B -- growth across the ladder:** regions at the highest rung reached over regions at 980, per shape and arm (prompt 04 section 4's measure). `rungs` is the rungs the arm has `ok` on that shape.

| shape | arm | rungs ok | regions at 980 | highest rung ok | regions there | growth |
|---|---|---|---|---|---|---|
| together | E | 6 of 6 | 7 | 1e+08 | 9 | 1.29 |
| together | S | 6 of 6 | 1254 | 1e+08 | 8526 | 6.8 |
| together | P0 | 6 of 6 | 53 | 1e+08 | 473552 | 8.93e+03 |
| together | P | 6 of 6 | 46 | 1e+08 | 288 | 6.26 |
| T-first | E | 5 of 5 | 7 | 1e+07 | 13 | 1.86 |
| T-first | S | 5 of 5 | 716 | 1e+07 | 33570 | 46.9 |
| T-first | P0 | 5 of 5 | 34 | 1e+07 | 80117 | 2.36e+03 |
| T-first | P | 5 of 5 | 34 | 1e+07 | 112 | 3.29 |
| q-smooth | E | 4 of 4 | 17 | 1e+06 | 16 | 0.941 |
| q-smooth | S | 4 of 4 | 839 | 1e+06 | 33752 | 40.2 |
| q-smooth | P0 | 4 of 4 | 687 | 1e+06 | 425137 | 619 |
| q-smooth | P | 4 of 4 | 686 | 1e+06 | 1480 | 2.16 |

**Table K -- the attribution, as ratios of growths** (prompt 04 section 4): growth is regions at the top rung over regions at 980, on shapes where the arm has every rung. `S/P0` is S's growth over P0's (what the `phase_spline` floor adds, against P0); `P0/P` is P0's over P's (what the anchoring floor adds). `carrier` is the group carrying the maximum depth at the top rung (Levin sub-interval : regime : group), with that depth.

| shape | top rung | growth E | growth S | growth P0 | growth P | S/P0 | P0/P | P/E | carrier S | carrier P0 | carrier P |
|---|---|---|---|---|---|---|---|---|---|---|---|
| together | 1e+08 | 1.29 | 6.8 | 8.93e+03 | 6.26 | 0.000761 | 1.43e+03 | 4.87 | 2:Gqr:G+q+r (depth 20) | 2:Gqr:G+q+r (depth 20) | 2:Gqr:G+q+r (depth 13) |
| T-first | 1e+07 | 1.86 | 46.9 | 2.36e+03 | 3.29 | 0.0199 | 715 | 1.77 | 2:Gqr:G-q+r (depth 20) | 2:Gqr:G+q-r (depth 20) | 2:Gqr:G+q-r (depth 9) |
| q-smooth | 1e+06 | 0.941 | 40.2 | 619 | 2.16 | 0.065 | 287 | 2.29 | 2:Gqr:G+q+r (depth 20) | 1:Gr:G-r (depth 20) | 1:Gr:G-r (depth 14) |

**Table C -- the level at 980:** regions per arm over arm E's.

| shape | E | S | S/E | P0 | P0/E | P | P/E |
|---|---|---|---|---|---|---|---|
| together | 7 | 1254 | 179 | 53 | 7.57 | 46 | 6.57 |
| T-first | 7 | 716 | 102 | 34 | 4.86 | 34 | 4.86 |
| q-smooth | 17 | 839 | 49.4 | 687 | 40.4 | 686 | 40.4 |

### 2.3 `integral_time` with the load

Another job shared the host for most of the run, so the 1-minute load ran from 5.4 to 21.1. Times
are indicative only.

**Table D -- `integral_time` (s) with the 1-minute load average at the cell's start and end.** Up to four cells ran at once; times are indicative only.

| shape | x_resp | arm | integral_time | setup_time | load start | load end |
|---|---|---|---|---|---|---|
| together | 980 | E | 0.54 | 0.021 | 13.39 | 13.39 |
| together | 980 | S | 18.04 | 0.133 | 13.28 | 12.44 |
| together | 980 | P0 | 1.07 | 0.139 | 13.28 | 13.28 |
| together | 980 | P | 0.92 | 0.132 | 13.39 | 13.39 |
| together | 10000 | E | 0.57 | 0.030 | 13.28 | 13.28 |
| together | 10000 | S | 73.36 | 0.163 | 12.49 | 17.21 |
| together | 10000 | P0 | 1.35 | 0.160 | 12.49 | 12.49 |
| together | 10000 | P | 1.15 | 0.161 | 12.49 | 12.49 |
| together | 100000 | E | 0.57 | 0.034 | 13.28 | 13.10 |
| together | 100000 | S | 536.65 | 0.183 | 17.21 | 17.05 |
| together | 100000 | P0 | 25.55 | 0.193 | 13.84 | 14.52 |
| together | 100000 | P | 1.36 | 0.201 | 13.84 | 13.84 |
| together | 1e+06 | E | 0.59 | 0.038 | 13.10 | 13.10 |
| together | 1e+06 | S | 782.47 | 0.218 | 17.05 | 8.15 |
| together | 1e+06 | P0 | 927.36 | 0.233 | 15.98 | 8.01 |
| together | 1e+06 | P | 1.71 | 0.239 | 15.78 | 15.72 |
| together | 1e+07 | E | 0.60 | 0.043 | 12.53 | 12.53 |
| together | 1e+07 | S | 902.19 | 0.254 | 8.15 | 8.56 |
| together | 1e+07 | P0 | 4949.80 | 0.277 | 8.50 | 5.58 |
| together | 1e+07 | P | 4.78 | 0.277 | 8.81 | 8.50 |
| together | 1e+08 | E | 0.57 | 0.076 | 12.57 | 12.57 |
| together | 1e+08 | S | 134.84 | 0.294 | 12.12 | 9.68 |
| together | 1e+08 | P0 | 7198.04 | 0.312 | 5.37 | 6.73 |
| together | 1e+08 | P | 4.49 | 0.328 | 12.48 | 12.12 |
| T-first | 980 | E | 0.19 | 0.021 | 13.39 | 13.39 |
| T-first | 980 | S | 7.24 | 0.125 | 12.53 | 12.44 |
| T-first | 980 | P0 | 0.51 | 0.130 | 12.53 | 12.53 |
| T-first | 980 | P | 0.51 | 0.128 | 13.10 | 13.10 |
| T-first | 10000 | E | 0.29 | 0.028 | 13.39 | 13.39 |
| T-first | 10000 | S | 44.72 | 0.157 | 12.21 | 14.52 |
| T-first | 10000 | P0 | 0.75 | 0.168 | 12.49 | 12.21 |
| T-first | 10000 | P | 0.74 | 0.169 | 12.49 | 12.21 |
| T-first | 100000 | E | 0.29 | 0.034 | 13.28 | 13.28 |
| T-first | 100000 | S | 169.85 | 0.185 | 14.73 | 15.78 |
| T-first | 100000 | P0 | 0.97 | 0.196 | 14.80 | 14.80 |
| T-first | 100000 | P | 0.79 | 0.194 | 14.80 | 14.80 |
| T-first | 1e+06 | E | 0.33 | 0.040 | 13.28 | 13.28 |
| T-first | 1e+06 | S | 188.89 | 0.235 | 15.72 | 16.87 |
| T-first | 1e+06 | P0 | 10.45 | 0.243 | 15.57 | 16.22 |
| T-first | 1e+06 | P | 0.83 | 0.242 | 15.57 | 15.57 |
| T-first | 1e+07 | E | 0.34 | 0.043 | 13.10 | 13.10 |
| T-first | 1e+07 | S | 515.04 | 0.254 | 8.27 | 11.48 |
| T-first | 1e+07 | P0 | 1317.30 | 0.273 | 10.16 | 7.88 |
| T-first | 1e+07 | P | 1.63 | 0.274 | 10.35 | 10.16 |
| q-smooth | 980 | E | 0.22 | 0.026 | 13.39 | 13.39 |
| q-smooth | 980 | S | 6.74 | 0.118 | 12.44 | 12.21 |
| q-smooth | 980 | P0 | 5.32 | 0.128 | 12.44 | 12.49 |
| q-smooth | 980 | P | 5.26 | 0.132 | 12.57 | 12.44 |
| q-smooth | 10000 | E | 0.22 | 0.024 | 13.39 | 13.39 |
| q-smooth | 10000 | S | 118.96 | 0.145 | 12.43 | 21.06 |
| q-smooth | 10000 | P0 | 13.87 | 0.157 | 12.21 | 12.43 |
| q-smooth | 10000 | P | 10.14 | 0.157 | 12.21 | 12.56 |
| q-smooth | 100000 | E | 0.22 | 0.067 | 13.28 | 13.28 |
| q-smooth | 100000 | S | 1372.76 | 0.163 | 20.82 | 8.31 |
| q-smooth | 100000 | P0 | 340.20 | 0.178 | 17.21 | 15.35 |
| q-smooth | 100000 | P | 10.38 | 0.174 | 14.73 | 15.83 |
| q-smooth | 1e+06 | E | 0.21 | 0.064 | 13.28 | 13.28 |
| q-smooth | 1e+06 | S | 505.21 | 0.188 | 8.05 | 10.35 |
| q-smooth | 1e+06 | P0 | 4421.76 | 0.206 | 17.29 | 6.34 |
| q-smooth | 1e+06 | P | 13.00 | 0.206 | 17.68 | 17.75 |

### 2.4 The cells not reached

**None.** All 60 cells are `ok`. The longest was `P0|together|1e+08`, at 7,198 s of integral,
under its 3 h budget.

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

Prompt §4 is applied as written. The measure is regions at the top rung over regions at 980, per
shape (Table B, Table K):

| shape | E | S | P₀ | P |
|---|---|---|---|---|
| `together` (980→$10^8$) | 1.29 | 6.8 | 8,930 | **6.26** |
| `T-first` (980→$10^7$) | 1.86 | 46.9 | 2,360 | **3.29** |
| `q-smooth` (980→$10^6$) | 0.94 | 40.2 | 619 | **2.16** |

- **Neither reading cleanly.**
  - **Arm P is not flat like arm E.** Over the ladder its growth is 4.9×, 1.8× and 2.3× E's.
  - **It is not unambiguously "still growing" like S.** On every shape its growth lies **between
    E's and S's**, which is the case §4 names as its example of neither.
  - **The shapes disagree in how far between.** On `together`, P's growth (6.26) is close to S's
    (6.8). On `T-first` and `q-smooth` it is 14× and 19× below S's, and within a factor of 1.8–2.3
    of E's.
  - **P is not monotone.** On `together` it reaches 313 regions at $10^7$ and 288 at $10^8$.
  - **P's maximum depth** rises 5→14 (`together`), 7→9 (`T-first`) and 11→14 (`q-smooth`). It
    never reaches the cap of 20, which S and P₀ reach on every shape (from $10^5$ to $10^7$).

  Per §4, the reading is left to the user.
- **The attribution** (Table K), as ratios of growths, with the group carrying the deepest regions
  at the top rung:
  - **S against P₀: S/P₀ = 7.6e-4 (`together`), 0.020 (`T-first`), 0.065 (`q-smooth`).** S's
    `phase_spline` $G$ grows **less** than P₀'s un-anchored `PrimitivePhase` $G$, by factors of
    1,300, 50 and 15. Replacing the `phase_spline` with a `PrimitivePhase` but no node anchor
    increases the growth on every shape.
    - Carriers on `together`: S `2:Gqr:G+q+r` at depth 20, and P₀ the same group at depth 20.
    - On `T-first`: S `2:Gqr:G-q+r` at depth 20, P₀ `2:Gqr:G+q-r` at depth 20.
    - On `q-smooth`: S `2:Gqr:G+q+r` at depth 20, P₀ `1:Gr:G-r` at depth 20.
  - **P₀ against P: P₀/P = 1,430 (`together`), 715 (`T-first`), 287 (`q-smooth`).** The
    anchoring floor accounts for nearly all of P₀'s growth.
    - P's carriers are `2:Gqr:G+q+r` at depth 13 (`together`), `2:Gqr:G+q-r` at depth 9
      (`T-first`) and `1:Gr:G-r` at depth 14 (`q-smooth`).
    - So on each shape P's deepest group at the top rung is P₀'s, at 6–11 fewer levels.
- **Level at 980** (reported apart, Table C). Regions over E's:
  - S: 179 / 102 / 49.4;
  - P₀: 7.57 / 4.86 / 40.4;
  - P: 6.57 / 4.86 / 40.4.

  At the base rung P₀ and P are within 15 % of each other on every shape. On `q-smooth` both sit
  near S's level (686–687 against 839), all carried by `1:Gr:G-r`.

## 4. The depth record

Table G lists, for every cell, the group carrying the maximum depth, its $z'$-range and the
phase-limited count.
- **E** stays at depth ≤4 on `together` and `T-first`, and at depth 3–8 on `q-smooth`.
- **S** reaches 20 from $10^5$ on every shape. Its deepest regions are in the all-oscillatory
  sub-interval (`Gqr`). They start near $z' \approx 6$ at the lower rungs and move up the ladder
  to $1.5\times10^4$–$2.8\times10^5$ at `together` $10^8$.
- **P₀** reaches 20 from $10^6$ on `together`, from $10^5$ on `q-smooth` and at $10^7$ on
  `T-first`. Its
  deepest regions spread over most of the range at the top rungs: on `together` $10^8$, 335,626
  regions at depth 20 over $z' = 7.3\times10^4$–$3.0\times10^7$. Its phase-limited counts grow
  to 162,594 (`together` $10^8$), 68,048 (`T-first` $10^7$) and 264,172 (`q-smooth` $10^6$).
- **P** stays at depth ≤14. At the top rungs it has only 2–46 regions at its maximum depth, in
  narrow $z'$-windows, and 0–28 phase-limited regions per cell.

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
| together | 10000 | P0 | 8 | 2:Gqr:G+q+r | 33 | 6 | 151.971 - 2738.41 | 2:Gqr:G+q+r | 5 |
| together | 100000 | P0 | 19 | 2:Gqr:G+q+r | 1428 | 2 | 485.65 - 485.666 | 2:Gqr:G+q+r | 986 |
| together | 1e+06 | P0 | 20 | 2:Gqr:G+q+r | 55201 | 23068 | 2839.61 - 27262 | 2:Gqr:G+q+r | 44969 |
| together | 1e+07 | P0 | 20 | 2:Gqr:G+q+r | 298447 | 252640 | 8470.18 - 2.94984e+06 | 2:Gqr:G+q+r | 102251 |
| together | 1e+08 | P0 | 20 | 2:Gqr:G+q+r | 347910 | 335626 | 72988 - 3.01087e+07 | 1:Gr:G-r, 2:Gqr:G+q+r | 162594 |
| together | 980 | P | 5 | 1:Gr:G+r | 16 | 6 | 307.898 - 332.333 | 0:r:r, 1:Gr:G+r, 1:Gr:G-r | 1 |
| together | 10000 | P | 6 | 2:Gqr:G+q+r | 21 | 8 | 144.904 - 3010.21 | 2:Gqr:G+q+r | 2 |
| together | 100000 | P | 8 | 2:Gqr:G+q+r | 36 | 8 | 598.081 - 776.862 | 2:Gqr:G+q+r | 5 |
| together | 1e+06 | P | 9 | 2:Gqr:G+q+r | 57 | 12 | 3497 - 4489.76 | 2:Gqr:G+q+r | 4 |
| together | 1e+07 | P | 14 | 2:Gqr:G+q+r | 268 | 2 | 8924.26 - 8938.39 | 2:Gqr:G+q+r | 9 |
| together | 1e+08 | P | 13 | 2:Gqr:G+q+r | 244 | 46 | 98340.5 - 110796 | 2:Gqr:G+q+r | 6 |
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
| T-first | 10000 | P0 | 7 | 0:r:r | 20 | 2 | 3373.67 - 3383.3 | 0:r:r | 0 |
| T-first | 100000 | P0 | 7 | 2:Gqr:G-q+r | 21 | 8 | 253.52 - 707.576 | 0:r:r, 2:Gqr:G+q-r, 2:Gqr:G-q+r | 2 |
| T-first | 1e+06 | P0 | 17 | 2:Gqr:G-q+r | 581 | 8 | 1357.94 - 1433.74 | 2:Gqr:G-q+r | 392 |
| T-first | 1e+07 | P0 | 20 | 2:Gqr:G+q-r | 41180 | 14540 | 57135.9 - 355714 | 2:Gqr:G+q-r, 2:Gqr:G-q+r | 68048 |
| T-first | 980 | P | 7 | 0:r:r | 24 | 4 | 307.866 - 330.661 | 0:r:r | 2 |
| T-first | 10000 | P | 7 | 0:r:r | 20 | 2 | 3373.67 - 3383.3 | 0:r:r | 0 |
| T-first | 100000 | P | 7 | 0:r:r | 20 | 2 | 33745.7 - 33842 | 0:r:r | 2 |
| T-first | 1e+06 | P | 7 | 0:r:r | 20 | 2 | 337466 - 338429 | 0:r:r, 2:Gqr:G-q+r | 4 |
| T-first | 1e+07 | P | 9 | 2:Gqr:G+q-r | 49 | 10 | 57131.7 - 89588.9 | 2:Gqr:G+q-r | 4 |
| q-smooth | 980 | E | 8 | 1:Gr:G-r | 10 | 2 | 16.3398 - 16.8814 | 1:Gr:G-r | 1 |
| q-smooth | 10000 | E | 6 | 1:Gr:G-r | 9 | 4 | 54.452 - 98.791 | 1:Gr:G-r | 0 |
| q-smooth | 100000 | E | 6 | 1:Gr:G-r | 9 | 4 | 553.52 - 996.91 | 1:Gr:G-r | 3 |
| q-smooth | 1e+06 | E | 3 | 0:G:G | 4 | 2 | 338949 - 561409 | 0:G:G, 1:Gr:G+r, 1:Gr:G-r | 4 |
| q-smooth | 980 | S | 11 | 1:Gr:G-r | 385 | 12 | 5.78461 - 9.25169 | 1:Gr:G-r | 12 |
| q-smooth | 10000 | S | 18 | 2:Gqr:G+q-r | 2799 | 2 | 6.61998 - 6.62007 | 2:Gqr:G+q-r, 2:Gqr:G-q-r | 61 |
| q-smooth | 100000 | S | 20 | 2:Gqr:G-q-r | 29337 | 14604 | 5.72801 - 73.314 | 2:Gqr:G+q+r, 2:Gqr:G+q-r, 2:Gqr:G-q+r, 2:Gqr:G-q-r | 4220 |
| q-smooth | 1e+06 | S | 20 | 2:Gqr:G+q+r | 14326 | 6026 | 12.5362 - 407.157 | 2:Gqr:G+q+r, 2:Gqr:G+q-r, 2:Gqr:G-q+r | 1544 |
| q-smooth | 980 | P0 | 11 | 1:Gr:G-r | 307 | 12 | 8.31746 - 9.93992 | 1:Gr:G-r | 14 |
| q-smooth | 10000 | P0 | 14 | 1:Gr:G-r | 973 | 20 | 30.1855 - 37.9618 | 1:Gr:G-r | 23 |
| q-smooth | 100000 | P0 | 20 | 1:Gr:G-r | 25064 | 4332 | 302.925 - 1213.6 | 1:Gr:G-r | 20162 |
| q-smooth | 1e+06 | P0 | 20 | 1:Gr:G-r | 292870 | 192868 | 3067.88 - 125101 | 1:Gr:G+r, 1:Gr:G-r | 264172 |
| q-smooth | 980 | P | 11 | 1:Gr:G-r | 307 | 10 | 8.31746 - 9.25169 | 1:Gr:G-r | 11 |
| q-smooth | 10000 | P | 14 | 1:Gr:G-r | 679 | 2 | 34.3814 - 34.4017 | 1:Gr:G-r | 15 |
| q-smooth | 100000 | P | 14 | 1:Gr:G-r | 703 | 2 | 333.276 - 333.468 | 1:Gr:G-r | 28 |
| q-smooth | 1e+06 | P | 14 | 1:Gr:G-r | 948 | 2 | 3121.11 - 3122.9 | 1:Gr:G-r | 19 |

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
| together | 10000 | +3.109e-15 | 1.3e-12 | P0 | -1.035e-04 | 2.6e-09 | -1.035e-04 | 3.0e-09 |
| together | 10000 | +3.109e-15 | 1.3e-12 | P | -1.035e-04 | 2.6e-09 | -1.035e-04 | 3.0e-09 |
| together | 100000 | +1.723e-13 | 1.7e-11 | S | +1.495e-04 | 1.9e-09 | +1.495e-04 | 2.2e-09 |
| together | 100000 | +1.723e-13 | 1.7e-11 | P0 | +1.498e-04 | 1.9e-09 | +1.498e-04 | 2.2e-09 |
| together | 100000 | +1.723e-13 | 1.7e-11 | P | +1.498e-04 | 1.9e-09 | +1.498e-04 | 2.2e-09 |
| together | 1e+06 | -1.561e-12 | 1.4e-10 | S | -8.795e-05 | 1.7e-09 | -8.795e-05 | 2.0e-09 |
| together | 1e+06 | -1.561e-12 | 1.4e-10 | P0 | -8.787e-05 | 1.7e-09 | -8.787e-05 | 2.1e-09 |
| together | 1e+06 | -1.561e-12 | 1.4e-10 | P | -8.787e-05 | 1.7e-09 | -8.787e-05 | 2.1e-09 |
| together | 1e+07 | -3.101e-10 | 9.9e-09 | S | -1.090e-03 | 9.7e-09 | -1.090e-03 | 2.2e-08 |
| together | 1e+07 | -3.101e-10 | 9.9e-09 | P0 | -1.090e-03 | 1.9e-08 | -1.090e-03 | 3.2e-08 |
| together | 1e+07 | -3.101e-10 | 9.9e-09 | P | -1.090e-03 | 1.6e-08 | -1.090e-03 | 2.9e-08 |
| together | 1e+08 | -1.461e-10 | 3.5e-08 | S | -3.249e-04 | 3.9e-08 | -3.249e-04 | 8.4e-08 |
| together | 1e+08 | -1.461e-10 | 3.5e-08 | P0 | -3.250e-04 | 4.9e-08 | -3.250e-04 | 9.5e-08 |
| together | 1e+08 | -1.461e-10 | 3.5e-08 | P | -3.250e-04 | 3.9e-08 | -3.250e-04 | 8.4e-08 |
| T-first | 980 | -2.931e-14 | 4.8e-12 | S | -6.037e-05 | 3.5e-09 | -6.037e-05 | 3.9e-09 |
| T-first | 980 | -2.931e-14 | 4.8e-12 | P0 | -6.042e-05 | 3.5e-09 | -6.042e-05 | 3.9e-09 |
| T-first | 980 | -2.931e-14 | 4.8e-12 | P | -6.042e-05 | 3.5e-09 | -6.042e-05 | 3.9e-09 |
| T-first | 10000 | -2.220e-16 | 1.0e-12 | S | -1.334e-05 | 2.9e-09 | -1.334e-05 | 3.3e-09 |
| T-first | 10000 | -2.220e-16 | 1.0e-12 | P0 | -1.332e-05 | 2.9e-09 | -1.332e-05 | 3.3e-09 |
| T-first | 10000 | -2.220e-16 | 1.0e-12 | P | -1.332e-05 | 2.9e-09 | -1.332e-05 | 3.3e-09 |
| T-first | 100000 | +6.461e-14 | 1.6e-12 | S | -1.380e-05 | 3.0e-09 | -1.380e-05 | 3.4e-09 |
| T-first | 100000 | +6.461e-14 | 1.6e-12 | P0 | -1.381e-05 | 3.0e-09 | -1.381e-05 | 3.4e-09 |
| T-first | 100000 | +6.461e-14 | 1.6e-12 | P | -1.381e-05 | 3.0e-09 | -1.381e-05 | 3.4e-09 |
| T-first | 1e+06 | -2.310e-12 | 7.8e-11 | S | -3.087e-05 | 1.4e-09 | -3.087e-05 | 1.6e-09 |
| T-first | 1e+06 | -2.310e-12 | 7.8e-11 | P0 | -3.104e-05 | 1.5e-09 | -3.104e-05 | 1.8e-09 |
| T-first | 1e+06 | -2.310e-12 | 7.8e-11 | P | -3.104e-05 | 1.5e-09 | -3.104e-05 | 1.8e-09 |
| T-first | 1e+07 | +4.674e-12 | 3.1e-10 | S | -8.214e-06 | 5.5e-10 | -8.214e-06 | 9.6e-10 |
| T-first | 1e+07 | +4.674e-12 | 3.1e-10 | P0 | -8.177e-06 | 8.8e-10 | -8.177e-06 | 1.3e-09 |
| T-first | 1e+07 | +4.674e-12 | 3.1e-10 | P | -8.177e-06 | 8.6e-10 | -8.177e-06 | 1.3e-09 |
| q-smooth | 980 | -3.644e-13 | 5.2e-12 | S | -5.405e-05 | 9.9e-11 | -5.405e-05 | 1.2e-10 |
| q-smooth | 980 | -3.644e-13 | 5.2e-12 | P0 | -5.638e-05 | 7.6e-11 | -5.638e-05 | 9.2e-11 |
| q-smooth | 980 | -3.644e-13 | 5.2e-12 | P | -5.638e-05 | 7.7e-11 | -5.638e-05 | 9.3e-11 |
| q-smooth | 10000 | -1.088e-13 | 1.0e-11 | S | -9.654e-05 | 3.9e-10 | -9.654e-05 | 4.5e-10 |
| q-smooth | 10000 | -1.088e-13 | 1.0e-11 | P0 | -9.081e-05 | 3.2e-10 | -9.081e-05 | 3.7e-10 |
| q-smooth | 10000 | -1.088e-13 | 1.0e-11 | P | -9.081e-05 | 1.7e-10 | -9.081e-05 | 2.1e-10 |
| q-smooth | 100000 | -3.855e-13 | 2.1e-11 | S | -1.148e-06 | 3.0e-10 | -1.148e-06 | 3.6e-10 |
| q-smooth | 100000 | -3.855e-13 | 2.1e-11 | P0 | -1.136e-05 | 7.7e-10 | -1.136e-05 | 8.9e-10 |
| q-smooth | 100000 | -3.855e-13 | 2.1e-11 | P | -1.136e-05 | 9.6e-11 | -1.136e-05 | 1.3e-10 |
| q-smooth | 1e+06 | +2.003e-13 | 5.4e-10 | S | -1.049e-04 | 7.1e-10 | -1.049e-04 | 1.4e-09 |
| q-smooth | 1e+06 | +2.003e-13 | 5.4e-10 | P0 | -8.780e-05 | 2.1e-08 | -8.780e-05 | 2.4e-08 |
| q-smooth | 1e+06 | +2.003e-13 | 5.4e-10 | P | -8.780e-05 | 7.5e-10 | -8.780e-05 | 1.4e-09 |

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
  here; 05a did that on A3 v2. Its `cs_tau` table is the union table of §0.
- **$x_{\rm resp} \le 10^8$**, not production's $4\times10^{12}$.
- **The producers' stored phase samples** (`levin-phase-floor` 06+), which the fixture does not
  model.
- **$G$'s grid in P₀ and P ends exactly at $z_{\rm resp}$** (§0 item 2). That is one ulp from
  `Case`'s grid at 13 of 15 rungs.
- **Times are indicative only.** Up to four cells ran at once, beside another job, at a 1-minute
  load of 5.4–21.1.

**Table J -- Phase objects, as every `ok` cell records them** (class / exact_remainder):

- E: G _ExactPhase/None, Tq _ExactPhase/None/None, Tr _ExactPhase/None/None, construction_ok True -- 15 cells
- P: G PrimitivePhase/True, Tq PrimitivePhase/True/TablePrimitive, Tr PrimitivePhase/True/TablePrimitive, construction_ok True -- 15 cells
- P0: G PrimitivePhase/False, Tq PrimitivePhase/True/TablePrimitive, Tr PrimitivePhase/True/TablePrimitive, construction_ok True -- 15 cells
- S: G phase_spline/None, Tq PrimitivePhase/True/TablePrimitive, Tr PrimitivePhase/True/TablePrimitive, construction_ok True -- 15 cells

*Tables generated by `scripts/a4_tables.py` (SHA-256 `4ae8f56a…`) from run 2's checkpoint
(`var/runs/handover-04-cost-growth-20261026T173720/checkpoint.jsonl`).*
