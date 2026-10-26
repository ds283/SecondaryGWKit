# Does A2's Levin cost growth survive production's phases? (handover A4)

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

## 1. The readings

Quoted from prompt §4, as copied to `state/readings.md`:

{{READINGS}}

Run 2's `state/readings.md` was written at **17:37:09 UTC**. The run's manifest was written at
**17:37:21 UTC**, 12 s later. The run ended `done` at about 21:20:08 UTC (its last heartbeat).
Run 1's copy, `state/readings-run1.md` (15:26:53 UTC, before run 1's manifest at 15:27:04), has
identical content.

## 2. The tables

### 2.1 Regions and maximum depth

{{A}}

### 2.2 Growth per shape, the attribution, and the level at 980

{{B}}

{{K}}

{{C}}

### 2.3 `integral_time` with the load

Another job shared the host for most of the run, so the 1-minute load ran from 5.4 to 21.1. Times
are indicative only.

{{D}}

### 2.4 The cells not reached

**None.** All 60 cells are `ok`. The longest was `P0|together|1e+08`, at 7,198 s of integral,
under its 3 h budget.

### 2.5 A2's recorded curves, labelled pre-B1

{{F}}

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
  here; 05a did that on A3 v2. Its `cs_tau` table is the union table of §0.
- **$x_{\rm resp} \le 10^8$**, not production's $4\times10^{12}$.
- **The producers' stored phase samples** (`levin-phase-floor` 06+), which the fixture does not
  model.
- **$G$'s grid in P₀ and P ends exactly at $z_{\rm resp}$** (§0 item 2). That is one ulp from
  `Case`'s grid at 13 of 15 rungs.
- **Times are indicative only.** Up to four cells ran at once, beside another job, at a 1-minute
  load of 5.4–21.1.

{{J}}

*Tables generated by `scripts/a4_tables.py` (SHA-256 `4ae8f56a…`) from run 2's checkpoint
(`var/runs/handover-04-cost-growth-20261026T173720/checkpoint.jsonl`).*
