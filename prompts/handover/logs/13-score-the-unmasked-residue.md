# Log 13 — B2: score the residue the gap was masking

**Prompt:** [`prompts/handover/13-score-the-unmasked-residue.md`](../13-score-the-unmasked-residue.md).
Part A (§1, the fixtures) and Part B (§2–§3, the measurement).
**Commits:** both on branch `handover-13-b2` in the worktree `SecondaryGWKit-handover-13`, cut at
the dispatch SHA `c3802bf`:
- Part A: `28ade78`, "Make the realistic flavour production's construction" (the branch SHA;
  landing is by rebase);
- Part B: "Record handover prompt 13: B2's residue attribution" (the records commit).

**Landed 2026-10-27** by rebase onto `handover-remedial` over `levin-phase-floor` 06a's records (`1027dea`, `139f0b9`), which touched no `.py` file outside `docs/`: Part A as `d88faec`, the records as `fd11b82`, the user's ruling on deviation 8 as `c9d9185`, and the index's count, corrected from 114 to 112 by counting after the rebase, as `0572abe`. The run's records name `28ade78`, the pre-rebase Part A.

**Model:** Claude Opus 5.5, in a fresh-context subagent on the laptop.
**Date:** 2026-10-27
**Result (Part A):** COMPLETE WITH DEVIATIONS. §1.1–§1.3 are implemented, tests (a)–(c) pass and
their three breakages are caught, the four comments are corrected and the grep is clean, and no
root's count fell. The agent stopped once (§6: two existing tests failed after the change), and
the user ruled on both. Those two rulings are deviations 1 and 2. The orchestrator's review passed
7 of 7.
**Result (Part B):** COMPLETE WITH DEVIATIONS.
- Run `handover-13-residue-20261027T110832` ended `done`, with 208/208 cells `ok` and none
  `not reached`.
- §2.5's six checks held before the launch, and `terms.md` is older than the manifest.
- Completeness held bit for bit, the geometry control held, and E reproduces A2's exact column.
- The residue is additive at every rung, and `together`'s 980 residue is attributed: 93% to the
  source spline.
- The measurement is [`docs/handover/SEAM-RESIDUE.md`](../../../docs/handover/SEAM-RESIDUE.md).
  Part B's deviations are 8–17 below.
- **Deviation 8, the together-980 numbers seen before `terms.md`, is awaiting the user's ruling.**

## Gate

Taken 2026-10-27 at 02:23 GMT, in the worktree:

```
$ pwd; git rev-parse HEAD; git status --short
/Users/ds283/Documents/Code/SecondaryGWKit-handover-13
c3802bf5ce3c784f431699ace7830e7200e7c772
                                  (empty)
$ shasum -a 256 docs/handover/realistic_large_x.py
c1cd3598c23a52051a0eb2695bf67977265383b5fa02977fb61d8251c2cd98a3  docs/handover/realistic_large_x.py
$ uptime
 2:23  up 22 days,  1:08, 2 users, load averages: 16.50 20.34 14.06
```

`./venv/bin/python -m runregistrykit list`, run from the main checkout: 60 runs, "58 finished, 2
unknown; 45 with no host recorded". Their states are 32 `done`, 21 `killed`, 3 `failed`, 2 `lost`
and 2 `unknown`. The two `unknown` are `a3-pilot` and `a3-pilot-resume`, which predate the
registry. None is `running` or stale on this host. The seven roots' counts at the dispatch SHA came
from the orchestrator (Verification, below).

## What shipped

**`ComputeTargets/tests/test_phase_groups.py`: $G$ is production's construction.**
- New `RESPONSE_NODE_SNAP_ULPS = 8`.
- New `_StoredGkValue(z, div, mod, theta_residual=None)`: the fields `_build_phase` reads off a
  `GkSourceValue`. It was `test_g_phase_reanchor._Value`, moved here.
- New `response_anchored_grid(z_grid, z_resp) -> (sample_grid, table_nodes)`:
  - any node within 8 ulp of $\log(1+z_{\rm resp})$ is set exactly to $z_{\rm resp}$;
  - $z_{\rm resp}$ is added to the table's nodes if it is not one already;
  - a `ValueError` is raised if two table nodes are not distinct in `log1p`.
- **`BesselPhaseGk` is rebuilt.** Its `phase` is `GkSourcePolicyData._build_phase(source, values,
  "test G")`: a `PrimitivePhase` with `exact_remainder=True`.
  - **The table:** a `CumulativeTable` of $1/H$ at `TAU_GAUSS_ORDER` on `table_nodes`, shifted to
    the closed-form $\tau$ at its top node.
  - **The samples:** stand-ins at `sample_grid`, each carrying the Bessel phase as
    `WKB_mod_2pi` and `theta_residual = theta - GK_PHASE_SIGN k tau.delta(z, z_resp)`, formed in
    double as 05's `_ProductionPhaseGk` formed it.
  - **New attributes:** `tau`, `z_nodes`, `model_functions`.
  - **Unchanged:** `numeric_Gk` (`compute_analytic_G`), `sin_amplitude` (`bessel_phase`'s
    modulus) and the signature. `WKB_region` is now the snapped sample grid's ends.
- **Docstrings:** the class docstring says what it now is and what is still not production's
  (amplitude, `numeric_Gk`, grid). The module docstring and two test docstrings that said
  "phase_spline" are corrected.
- **Imports added:** `log1p`, `ulp`, `SimpleNamespace`, `TAU_GAUSS_ORDER`, `TablePrimitive`,
  `GK_PHASE_SIGN`, `_build_phase` and `CumulativeTable`.

**`ComputeTargets/tests/test_g_phase_reanchor.py`**
- `_Value = _StoredGkValue`, an alias.
- `_ProductionPhaseGk(BesselPhaseGk)` is reduced to `super().__init__` plus `old_phase` (the same
  `PrimitivePhase` with the keyword off), which `levin-phase-floor` 05's tests read.
- **`_TableCsTauFixture`** has a new `cs_tau_nodes()`: the union of `z_numeric`,
  `z_numeric_below` and `z_WKB` without the node equal to `crossover_z`, descending. It raises if
  that union does not cover $z_{\rm init}$. `exact_envelope_model` builds the `cs_tau` table on
  these nodes and shifts it to the closed form at the top node. This is A4's
  `_UnionTableCsTauFixture`, adopted as written. The docstring says so.
- The now-unused imports `SimpleNamespace` and `_build_phase` are removed.

**`ComputeTargets/tests/test_tk_source_functions.py`**
- `Fixture.__init__` records its arguments as `self.parameters`, a dict. Nothing else changes, and
  `Fixture`'s defaults are unchanged.

**`ComputeTargets/tests/test_quadsource_integral.py`**
- **Module docstring:** the "realistic" bullet now says what the flavour is (T with no keyword on a
  table `cs_tau`, production's seam, $G$ by `_build_phase`, $f$ splined from exact samples). It
  also names what is still not production's: $G$'s amplitude and `numeric_Gk`, the uniform
  100-per-decade grid, $f$ from exact samples, and the fixed hand-over depth ($x_T\approx19$).
- **`OffsetBesselPhaseGk` is deleted.** It existed only because a `phase_spline` refuses
  mixed-sign cycle counts, and no test exercised the `phase_spline` construction as such.
- **`Case`:**
  - `Case.x_resp` is stored.
  - New classmethod `production_fixture(fixture) -> _TableCsTauFixture`: the seeded fixture's
    `parameters` with `drop_first_WKB_sample=True`, cached on those parameters (`_production_fixtures`).
  - **`Case(exact=False)`:** `self.Pq` and `self.Pr` are those fixtures, and `Tq_inputs` and
    `Tr_inputs` are captured from them. The builder is
    `tk_fixtures.TkSourceFunctions(P.exact_envelope_model(), k, Tn, Tw)` with no keyword, and the
    envelope models are built once per `Case`. $G$ is `BesselPhaseGk` on the same grid as before.
  - **`Case(exact=True)`** is unchanged.
- New `exact_with_the_geometry_of(case) -> Case`: deviation 1.
- New class `TestRealisticFlavourIsProductions`, tests (a)–(c) (Verification).
- `TestHandOverSeam.test_LG_row_that_skips_the_switch_node_raises` drops one sample, not five
  (deviation 2).

**Comments only** (§1.5):
- `ComputeTargets/primitive_phase.py`: the module docstring's "Protocol" paragraph (`:113`) and
  the `PrimitivePhase` docstring (`:175`) no longer name `QuadSourceIntegral._ClampedPhase`. The
  first now says `QuadSourceIntegral` reaches the phase only through `phase_groups`, since prompt
  12 removed its clamp adapter.
- `ComputeTargets/tests/test_primitive_phase.py:416`: the `test_clamped_phase_style_calls`
  docstring is corrected the same way.
- `main.py:1887-1893`: the comment found by "stores end at the larger of the two Tk hand-over
  redshifts" now says `QuadSource`'s values end `HANDOVER_OVERLAP_NODES` grid nodes below the
  earlier factor's switch node, and that a consumer evaluates the spline only down to that node.

## Deviations from the prompt

1. **The realistic oracle's exact comparator shares the realistic partition.** *The user's ruling
   at the agent's stop, 2026-10-27, option (A).* **STRUCTURALLY REQUIRED.**
   - **The failure.** After Part A, `TestAnalyticOracle.test_realistic_fixtures` failed six
     subtests: $b=0$, `together` and `T-first`, at every $x_{\rm resp}$. Their smooth-part and
     Levin-part deviations reached 1.080e-1 and 1.073e-1, against thresholds of 1e-3 and 5e-4. The
     total passed (6.645e-4 against 1.5e-3).
   - **The cause, measured.** The test compares `numeric_quad` and `WKB_Levin` separately against
     an exact run whose `ExactTkFunctions` break at $z_{\rm init}$. The realistic $T$ now break at
     production's switch node, one grid step lower: $T_q$ at 294.106 against 300.974, $T_r$ at
     353.127 against 361.369. One grid step of integral therefore moves from one part to the other.
   - **The check.** With the realistic $T$ fed seam-at-$z_{\rm init}$ inputs instead, every part
     passed. At `together` 980: smooth 4.782e-4, Levin 3.800e-5. At `T-first` 980: 9.247e-6 and
     3.903e-6.
   - **The ruling.** The comparator is now `exact_with_the_geometry_of(case)`: `Case(exact=True)`
     with each `ExactTkFunctions.crossover_z` at the realistic fixture's `z_switch`. Its `z_init`
     stays at the fixture's `crossover_z`, which `evaluate_QuadSource_integral` checks against the
     `QuadSource`'s record, as on `TkSourceFunctions` since prompt 12. Thresholds are unchanged;
     the class docstring records why.
2. **The skipped-switch-node test drops one sample, not five.** *The user's ruling at the same
   stop, option (A).* **STRUCTURALLY REQUIRED.**
   - **The failure.** The test drops five Liouville–Green samples from `case.Tr_inputs` and
     expects "largest numeric grid node strictly below z_init". On production's seam the row
     already starts one node below $z_{\rm init}$. Dropping five puts its first sample six nodes
     below, past the five numeric nodes the fixture stores there (`NUMERIC_NODES_BELOW_INIT`).
     `TkSourceFunctions` then refuses earlier, with "the switch node … is not among the stored
     numeric samples".
   - **Measured.** Dropping 1–4 samples raises the expected message.
   - **The fix.** It drops one, and the comment says why.
3. **`Fixture.parameters`.** **IMPLEMENTATION CHOICE.** §1.2–§1.3 need `Case(exact=False)` to build
   production's fixture, and Part B must build it from A2's `seed_fixtures`, which seeds plain
   `Fixture`s into `Case._fixtures` and cannot be edited.
   - *Alternatives:* recover `x_max` from the grids, which is not bit-exact; shallow-copy the plain
     fixture and change its class, which is fragile; give Part B its own seeding, which is a third
     copy.
   - *Choice:* recording the arguments is one dict, changes no default, and `production_fixture`
     rebuilds on exactly the same grids.
4. **The zero-width fix.** **IMPLEMENTATION CHOICE.** The prompt offered setting the grid's last
   node to $z_{\rm resp}$ (A4's run-2 fix) or de-duplicating in $\log(1+z)$. **The first is taken,
   generalised.**
   - **The rule.** Any grid node within 8 ulp of $\log(1+z_{\rm resp})$ is set to $z_{\rm resp}$.
     The sample there is then at the anchor, as in production, where $z_{\rm resp}$ is a
     background-grid node and the lowest source redshift.
   - **Why generalised.** A probe of `Case`'s $G$ grids found the end node at −1, +1, 0, 0, −1, +1,
     0, −1, −2 and −3 ulp from $z_{\rm resp}$. A +1 ulp end node makes the same zero-width
     interval, so "last node, below" alone does not cover it.
   - **The guard.** A table whose nodes are not distinct in `log1p` raises, naming the pair.
     `CumulativeTable` is not changed.
5. **How test (b) reads "its T evaluates at $z_{\rm init}$".** **IMPLEMENTATION CHOICE.** On
   production's seam the Liouville–Green accessors (`M`, `T_WKB`, `phase`) refuse above the switch
   node by design, and $z_{\rm init}$ is above it. A probe confirmed all four refuse there. What
   used to raise was the constructor, which anchors $T$'s phase at $z_{\rm init}$ through
   `cs_tau.delta(z, z_init)`. So (b) asserts:
   - $T$ builds and its phase is anchored at $z_{\rm init}$;
   - the table covers $z_{\rm init}$, which is not a node, with `p_R` ≠ 0;
   - `cs_tau.delta(z_init, z_switch)` is finite and positive;
   - the numeric `T` evaluates at $z_{\rm init}$, and the phase at the switch node.
6. **Docstrings in `test_phase_groups.py` beyond `:300-306`.** **IMPLEMENTATION CHOICE.** The
   module docstring and two test docstrings said "phase_spline", which this change makes false.
   They are corrected rather than left wrong.
7. **`OffsetBesselPhaseGk` deleted, `_StoredGkValue` moved.** **IMPLEMENTATION CHOICE**, within
   §1.1's "may become unnecessary" and "reuses the shared construction". No superseded
   `phase_spline` construction was kept, because no test exercises one as such.

## Verification performed

**The seven roots**, serially, in the worktree, at the end. The dispatch-SHA counts are the
orchestrator's (02:18–02:22 GMT).

| root | at `c3802bf` | after Part A | verdict |
|---|---|---|---|
| AdaptiveLevin | 32 | 32 | OK |
| ComputeTargets | 777 | **780** (+3, the tests added) | OK (133.6 s) |
| CosmologyModels | 39 | 39 | OK |
| LiouvilleGreen | 148 | 148 | OK (skipped=1, expected) |
| ObjectFactories | 280 | 280 | OK |
| RunDrivers | 127 | 127 | OK (skipped=1) |
| tools | 58 | 58 | OK |

The roots ran from 10:49:02 to 10:52:33 GMT. The `uptime` 1-minute load was 23.67 at the start
(5- and 15-minute: 52.27 and 38.89) and 13.22 at the end. The laptop was carrying other projects'
Ray work, so the wall times are indicative only. The known wall-clock flake
(`test_tk_wkb_phase.TestCost.test_wall_time_per_object`) did not fire.

**Tests added (3):** `ComputeTargets.tests.test_quadsource_integral.TestRealisticFlavourIsProductions`.
- **`test_a_construction`.** At `together`, $x_{\rm resp}=100$, $b=0$, on the objects the integral
  reads:
  - $G$: `PrimitivePhase` on `TablePrimitive`, `exact_remainder` True, sign −1, anchored at
    $z_{\rm resp}$ = 68.28203230275508.
  - $T_q$ and $T_r$: `PrimitivePhase` on `TablePrimitive`, `exact_remainder` True.
  - `WKB_region[0]` = 294.10587814408103 < $z_{\rm init}$ = 300.973834223185 ($T_q$), and
    353.1270537728973 < 361.3686010678221 ($T_r$); each equals the fixture's `z_switch`.
- **`test_b_table_cs_tau_fixture_covers_z_init`.** At `together`, $x_{\rm resp}=10^4$, both
  factors (A4 recorded both raising there):

  | factor | $z_{\rm init}$ | `p_R` | table nodes | switch node |
  |---|---|---|---|---|
  | $T_q$ ($k$ = 102,041) | 3080.365655338623, nearest node 369 | −4.3653e-06 | 635, from 1.514e7 to 5.733 | 3010.211186366521 |
  | $T_r$ ($k$ = 122,449) | 3696.638786406349 | −3.6366e-06 | 643 | 3612.478777341096 |
- **`test_c_no_zero_width_interval_at_the_anchor`.** At `together` 980, the grid's last node is
  6.069595132934192 = $z_{\rm resp}$ − 1 ulp. The table's bottom interval is 2.3074e-02 in
  $\log(1+z)$. `theta_mod_2pi` one ulp below $\log(1+z_{\rm resp})$ is −6.283185307179195, and
  `raw_theta` and `theta_deriv` are finite there.

**The breakage record**, each applied by a script that replaced one line of the worktree file, ran
the one test, and restored the file, checked by SHA-256:

| breakage | test | result |
|---|---|---|
| `Case`'s $G$ back on the old `OffsetBesselPhaseGk` `phase_spline` (the class pasted from `c3802bf`) | (a) | **FAIL**: "phase_spline object … is not an instance of PrimitivePhase" |
| `_TableCsTauFixture`'s table back on `list(self.z_WKB)` | (b) | **ERROR** for both factors: "CumulativeTable[cs_tau]: z=3080.37 lies more than one grid interval above the table (max node z=3010.21)" ($T_r$: 3696.64 against 3612.48) |
| the duplicate node restored (`table_nodes = grid ∪ {z_resp}`, no snap, no guard) | (c) | **ERROR**: "CumulativeTable[tau]: z=6.0696 lies more than one grid interval below the table (min node z=6.0696)", one ulp below the anchor; and **FAIL** on the distinct-in-$\log(1+z)$ assertion |

All three files were restored byte for byte.

**Numbers that moved.** These are from the five affected modules (`test_quadsource_integral`,
`test_phase_groups`, `test_g_phase_reanchor`, `test_t_phase_reanchor`,
`test_stored_phase_residual`), run at `c3802bf` (91 tests, OK, skipped=1) and after Part A. Margins
are against unchanged thresholds.

*`test_quadsource_integral`:*
- **`TestAnalyticOracle.test_realistic_fixtures`** (deviation 1; thresholds smooth 1e-3, Levin 5e-4,
  total 1.5e-3, all over scale).
  - **Worst:** smooth 4.782e-4 → **6.935e-4** (margin 1.44×); Levin 1.044e-4 → 1.006e-4; total
    4.865e-4 → 7.444e-4 (margin 2.0×); rel(total, analytic) 2.358e-3 → 3.220e-3.
  - **Per subtest, smooth / Levin, old → new:**

    | subtest | smooth | Levin |
    |---|---|---|
    | b=0 `together` 980 | 4.782e-4 → 6.935e-4 | 8.368e-6 → 5.094e-5 |
    | b=0 `together` 100 | 3.964e-5 → 6.398e-5 | 6.170e-6 → 3.523e-6 |
    | b=0 `together` 30 | 3.834e-4 → 4.786e-4 | 1.816e-5 → 2.264e-5 |
    | b=0 `T-first` 980 | 9.247e-6 → 2.292e-5 | 3.917e-6 → 4.116e-6 |
    | b=0 `T-first` 100 | 4.951e-6 → 1.149e-5 | 2.609e-6 → 2.272e-6 |
    | b=0 `T-first` 30 | 7.233e-6 → 1.703e-5 | 3.318e-6 → 2.965e-6 |
    | b=0 `q-smooth` 980 / 100 / 30 | unchanged | 2.239e-5 → 5.749e-6, 2.062e-5 → 1.507e-5, 8.983e-6 → 7.330e-6 |
    | b=0.2 `together` 980 / 100 / 30 | unchanged | 3.423e-5 → 3.283e-5, 4.336e-5 → 4.317e-5, 3.299e-5 → 3.215e-5 |
    | b=0.2 `T-first` 980 / 100 / 30 | 1.905e-6 → 1.198e-6, 6.468e-7 → 5.463e-8, 6.351e-7 → 7.665e-9 | 1.548e-5 → 1.578e-5, 1.550e-5 → 1.532e-5, 9.696e-6 → 9.880e-6 |
    | b=0.2 `q-smooth` 980 / 100 / 30 | unchanged | 4.846e-5 → 1.006e-4, 5.733e-5 → 3.850e-5, 1.044e-4 → 9.515e-5 |

  - **Levin solves** fell in 16 of 18 realistic cells, e.g. 407 → 27 at `together` 980 and
    437 → 33 at $b=0.2$ `together` 980. The exceptions are `together` 30: $b=0$ rose 17 → 21,
    and $b=0.2$ stayed at 25.
- **`TestSeams.test_integrand_continuity_at_the_hand_over`**, realistic, threshold 2e-3: $b=0$
  5.613e-6 → 1.865e-5; $b=0.2$ 4.005e-5 → 2.894e-5.
- **`TestHandOverSeam` test (g)**, B1 end to end, `T-first` 980, tolerance 3.387e-4: N + 9/8
  −6.0375e-05 → **−6.0420e-05**.
- **`TestHandOverSeam` test (f):** unchanged (no shortfall; the same switch nodes).
- **`TestHandOverSeam.test_LG_row_that_skips_the_switch_node_raises`** (deviation 2): before, five
  dropped samples raised the expected message; now one dropped sample raises it, "the first
  Liouville-Green sample z=345.0729485 is not the largest numeric grid node strictly below
  z_init=361.3686011 (z=353.1270538)".
- **`TestErrorBound.test_abserr_is_positive_sums_the_parts_and_is_small`**, printed only: realistic
  worst |total−analytic|/abserr 4.659e+04 → 6.341e+04; worst abserr/|total| 1.747e-07 → 1.460e-07
  (threshold 1e-5); `total_converged=False` in 14 → 10 of 36 cases.

*`test_phase_groups`:*
- **`TestOracle1DirectProduct.test_realistic_fixtures_populated_regimes`** (bound 1e-12): Gqr
  2.894e-16 / 3.568e-16 → 4.033e-16 / 3.778e-16 (w = 1/3 / 0.2); Gr 5.351e-16 / 4.194e-16 →
  4.657e-16 / 4.601e-16; G 2.195e-16 / 3.077e-16 → 2.060e-16 / 2.666e-16; qr and r unchanged.
- **`TestOracle2Bessel.test_realistic_fixtures_vs_scipy`** (threshold 5e-4): w=1/3 at 100/decade
  7.024e-06 → 7.033e-06, with $x_q>100$ 4.429e-07 → **8.469e-09**; w=0.2 at 100/decade 1.375e-04
  unchanged, $x_q>100$ 6.240e-06 → 6.243e-06; 300/decade w=1/3 $x_q>100$ 7.036e-09 → 7.776e-09.
- **`test_LG_truncation_is_the_realistic_floor`**, after exact $\omega$ and ${\rm d}\ln M/{\rm d}z$
  (threshold 1e-5): w=1/3 3.277e-07 → **1.964e-10**; w=0.2 2.931e-07 → **1.150e-11**. The $G$
  `phase_spline` floor is gone.

*`test_t_phase_reanchor` and the rest:*
- **05a test 6** (`test_t_phase_reanchor`, $T$ now on the union table): composed remainder worst
  2.025e-13 → 2.078e-13 rad, worst/bound 2.381e-01 → 3.019e-01, 760 → 761 of 800 moved.
- **Unchanged, bit for bit in what they print:** 05 test 6 (both lines), 05a test 4(b) (345 of 345)
  and 05a test 6's Oracle 1 (4.575e-16), and 06 test 7 (4.610e-16). $G$'s construction is
  bit-identical where $z_{\rm resp}$ is not near a grid node, which is the case for every
  `test_phase_groups.Case`.

**The grep** (§1.5), outside `venv/`, `docs/` and `prompts/`, finds only the three expected
`assertNotIn` checks: `test_quadsource_integral.py` `:1154`, `:1156` and `:1315` at this tree
(`:985`, `:987` and `:1146` at `c3802bf`). `docs/handover/realistic_large_x.py` still hashes
`c1cd3598c23a52051a0eb2695bf67977265383b5fa02977fb61d8251c2cd98a3`.

**`black`:** clean on all seven touched `.py` files.

## Records at their own SHA (§1.5)

These import fixtures Part A changes. Each is a record of the tree it ran on, and none is edited.
- `docs/handover/realistic_large_x.py` (A2, `c1cd3598…`). It imports `Case`, `Fixture` and
  `exact_Tk_inputs`, and runs at A2's tree `0d7c05c`.
- `docs/levin-phase-floor/noise_injection.py`. It imports `realistic_large_x` and `Case`.
- `docs/handover/a4/scripts/a4_cells.py` and `docs/handover/a4/scripts-run1/a4_cells.py`. They
  import `OffsetBesselPhaseGk`, which no longer exists, plus `_ProductionPhaseGk` and
  `_TableCsTauFixture`, whose constructions changed.
- The rest of `docs/handover/a4/scripts/` (`a4_checks.py`, `a4_driver.py`, `a4_launch.py`,
  `a4_tables.py`) and `docs/handover/a4/probes/`, which reach the fixtures through `a4_cells`.

## Observations not acted on

1. **The realistic smooth-part margin narrowed.** In `test_realistic_fixtures` the worst
   smooth-part deviation rose from 4.782e-4 to 6.935e-4, against 1e-3, at $b=0$ `together` 980.
   The Levin part there rose from 8.368e-6 to 5.094e-5. Both are within their thresholds.
   - **What changed between the two numbers:** with the comparator's geometry matched, the smooth
     sub-interval now ends at the switch node, inside the source spline's overlap nodes rather
     than at its old end node; and $G$ and $T$ are now production's constructions.
   - **Not attributed here.** That is Part B's measurement, and it should read this.
   - **Not a §3 issue,** because Part B exists to measure it.

## State handed to Part B

- `Case(exact=False)` is R's construction, built through `Case` from A2's `seed_fixtures`. Its
  production fixtures are `case.Pq` and `case.Pr`, its $G$ is `case.gk` (a `BesselPhaseGk`, with
  `.tau` and `.z_nodes`), and `case.x_resp` is stored.
- `exact_with_the_geometry_of(case)` is the exact flavour on R's partition. It is the T half of
  E′: the exact $G$ is `ExactGk`, and the source is `FakeQuadSource(exact=True)`.
- Test (a) is Part B's construction check (§2.5 item 1).

---

# Part B — the measurement (§2–§3)

## What shipped

All new files are under `docs/handover/`. Nothing outside `docs/` and `prompts/` changed in Part B.
- [`SEAM-RESIDUE.md`](../../../docs/handover/SEAM-RESIDUE.md), §0–§7 in prompt §3's order.
- [`b2/scripts/`](../../../docs/handover/b2/scripts/), each byte-identical (`cmp`) to the copy in
  the worktree's gitignored `var/handover-13/scripts/`:
  - `b2_cells.py` (`0a8ae0eb…`): one cell, the objects and the hybrids;
  - `b2_driver.py` (`dc8772e3…`): 208 cells, at most 2 at once, 1 h budget each;
  - `b2_launch.py` (`c8fb8bf0…`): `runregistrykit.begin`, then `nohup sh -c` in a new session;
  - `b2_checks.py` (`6bbda526…`): §2.5's six checks;
  - `b2_tables.py`: §3's tables, from the checkpoint alone;
  - `b2_probe_pairs.py`: the post-run pair probe (deviation 14).

  The first four are the run's hashed set and stamp every checkpoint record. The two others ran
  after the run and are records of their own SHA.
- [`b2/out/`](../../../docs/handover/b2/out/): `tables.md` (the tables' stdout, §3's source);
  `terms.md`; `check_<name>.txt` and `.json` for the six checks; `mapping_before_smoke.txt`;
  `probe_pairs.txt`; `launch.txt`; and the run's `run_manifest.json`, `run_summary.json` and
  `run_stdout.log`.
- **Not copied:** the checkpoint and the per-cell files (`cells/*.json`, `.out`, `.err`), which
  stay in the run directory under the main checkout's `var/runs/`.

**The objects and the five terms** are as SEAM-RESIDUE.md §0 and `b2/out/terms.md` state.
Delegation swaps accessors only, keeping R's geometry:
- `HybridT` wraps each `TkSourceFunctions`;
- `GkSourceFunctions._replace` handles $G$;
- `QuadSourceFunctions` with the exact $f$ handles the source.

The mapping was read from the code. `TkSourceFunctions.T` never dispatches to the WKB form; the
$f$ spline is read only on the all-smooth sub-interval; $T$'s Levin phase derivative is `omega`;
and `numeric_Gk` is exact in R and E alike, so it is in no term. No term needed a production
change.

## §2.5's checks (before the launch)

Each is saved as `b2/out/check_<name>.txt` and `.json`, and each HOLDS.
1. **Construction.** 208 objects (16 rungs × 13) were built with no problems. R's $G$ and $T$
   phases are each a `PrimitivePhase` on a `TablePrimitive` with `exact_remainder` on at every
   rung. Each $T$'s `WKB_region[0]` is below its $z_{
m init}$. R's $G$ phase evaluates at
   $\log(1+z_{\rm resp})$ and 1–8 ulp below it.
2. **Completeness**, at `together` 980, bit for bit in four quantities:
   - **(i)** the hybrid with nothing delegated equals unwrapped R
     (`total` `0x1.03e5aa4fb72dep-40`, `total_abserr` `0x1.aaeb34a31deebp-66`);
   - **(ii)** the hybrid with everything delegated equals `exact_with_the_geometry_of(R)`
     (`0x1.04bc9b11398a2p-40`, `0x1.0450e9d8fd8f0p-78`);
   - **disjointness:** 9 accessors, 0 in two terms.
3. **Determinism.** E, E′ and R run twice are bit-identical in totals and in every group's regions
   and depths (7, 7 and 44 regions; depths 0, 0 and 5).
4. **Transparency.** R with and without the wrapper is bit-identical, with 44 regions both ways.
5. **Controls.**
   - **E against A2's Table 2,** exact closed: within the declared errors at all 16 rungs, with
     $|\Delta N|\le4.4\times10^{-14}$. In $N$ that is 13–15 significant figures. A2's four printed
     figures of $N+9/8$, which is at the rounding level, agree 0–4 times.
   - **Geometry:** $N_{E'}-N_E = -3.73\times10^{-14}$ against $1.84\times10^{-11}$.
   - **R against A4's arm P:** $+3.622668\times10^{-3}$ against $+3.623\times10^{-3}$, recorded.
6. **Stationary points.** 108 rows at 4001 points each, with no sign change. The nearest approach
   is `T-first` $10^7$, group $G-q+r$, with $|\theta'|_{\min} = 0.626$ per $\log(1+z)$, 0.0226 of
   the largest constituent.

**`terms.md` against the launch.** `terms.md` has mtime 11:08:19.57 UTC and the manifest
11:08:32.84 UTC: 13 s older. The copy in `b2/out/` keeps the mtime (`cp -p`).

## The run

- **Run id and directory:** `handover-13-residue-20261027T110832`, in the main checkout's
  `var/runs/handover-13-residue-20261027T110832/`.
- **Launch:** registered, campaign `handover`, prompt `13`, slug `residue`. The driver's pid was
  24600, recorded with `Run.heartbeat`.
- **Verification:** one check at launch (status `running`, pid answers `kill -0`, heartbeat 4 s
  old), then no polling.
- **Duration:** 11:08:36–11:18:12 UTC, with 456 s of cell wall time. The longest cell took 15.1 s.
  The 1-minute load ran 6.0–26.9.
- **Outcome:** `done`, exit 0. Every record carries `git_head` `28ade78…` and an empty
  `git status --short`.
- **Cells `not reached`:** none. Cells raised or skipped: none.

## Deviations from the prompt (Part B)

8. **The together-980 numbers were seen before `terms.md` was written.** **UNINTENDED DRIFT, kept.**
   **Accepted by the user at review, 2026-10-27:** *"Let's accept the deviation. i don't think this
   is important."* The orchestrator had confirmed that `b2_cells.py`, which holds the mapping, hashed
   `0a8ae0eb…` in `mapping_before_smoke.txt` before the smoke test, and that every checkpoint record
   carries the same hash.
   - **What happened.** After writing `b2_cells.py`, the agent smoke-tested it by running all 13
     objects at `together` 980 in-process. That was before §2.5's checks and before `terms.md`.
     It saw R +3.6227e-3, E′ +3.3e-14, E′+f +3.375e-3, E′+Tnum +1.37e-4, E′+TLG +1.11e-4,
     E′+Trep +9e-11 and E′+G +6e-12.
   - **Why the mapping did not move.** It was already coded in `b2_cells.py`, and its sha256
     (`0a8ae0eb…`) and text were saved in `state/mapping_before_smoke.txt` right after the smoke
     test. The run's records carry the same hash, so the mapping is unchanged since.
   - **What it costs.** The `terms.md`-before-manifest evidence covers the other 15 rungs fully,
     but at `together` 980 it shows only that the mapping did not change after the numbers were
     seen.
9. **How completeness is checked.** **IMPLEMENTATION CHOICE.** Read literally, "E′ + all five ≡ R"
   and "R − all five ≡ E′" are identities if E′ is defined as R − all five. So each side was built
   differently: (i) the nothing-delegated hybrid against unwrapped R; (ii) the all-delegated hybrid
   against `exact_with_the_geometry_of(R)`, an independent construction.
10. **The §6 raise-stop schedule.** **IMPLEMENTATION CHOICE.** A cell depends on every cell of its
    shape at a lower rung that is planned before it. E cells run first, so they depend only on the
    E cells below them. No cell raised, so this never acted.
11. **The tables script is not in the run's hashed set.** **IMPLEMENTATION CHOICE.** This lets it be
    extended after the run. Tables 5 and 6 (per-term ranges, signs, monotonicity and slopes, and
    the additivity margins) were added after the run. It reads only the checkpoint.
12. **The E control takes A2's values from `REALISTIC-LARGE-X.md` Table 2,** to four figures, with
    half a unit of the fourth figure added to the bound. It does not use A2's JSONL in the main
    checkout. **IMPLEMENTATION CHOICE.**
13. **Small choices,** each an **IMPLEMENTATION CHOICE**:
    - the 50-digit reference is recomputed in every cell (identical within a rung, about 0.05 s);
    - the construction check also evaluates R's $G$ phase below $z_{\rm resp}$;
    - the scan reports $\theta'$ relative to the largest constituent's as well.
14. **A post-run probe of pairwise hybrids.** **IMPLEMENTATION CHOICE.** Run in-process and not
    registered (`b2_probe_pairs.py`, `b2/out/probe_pairs.txt`, about a minute). The run's
    interactions passed the additivity rule but were visible, at 50–62% of the largest term on
    `q-smooth` and in $B_G$ on `together`. The probe located them in `TLG`×`G` and `TLG`×`Tnum`.
    It is labelled in SEAM-RESIDUE.md §4.6 as not part of the measurement, and it changes no
    number of it.
15. **The driver's docstring gets the within-rung order wrong.** It says "E′, R", but the plan
    runs R, E′, then the hybrids. **UNINTENDED DRIFT, kept:** the driver is hashed into every
    record.
16. **CLAUDE.md is not edited.** **STRUCTURALLY REQUIRED.**
    - **The instruction.** Prompt §7 and the orchestrator ask that `docs/handover/b2/scripts/` be
      listed in CLAUDE.md's formatting paragraph if it is not `black`-clean.
    - **The state.** Five of its six scripts are not `black`-clean, and they must stay
      byte-identical to what ran.
    - **Why not done.** This agent does not change CLAUDE.md on an agent's instruction, so the
      line is left for the orchestrator or the user.
    - **The text to add,** after "`docs/levin-phase-floor/06/`":
      "and `handover` B2's `docs/handover/b2/scripts/`".
    - **Until then** the tree is not clean under `black --check` outside `prompts/`, by these five
      files.
17. **SEAM-RESIDUE.md §4 reports `TLG` jointly with `Tnum`, and `G`'s remove-one term as an
    interaction,** on `q-smooth` and on `together`'s upper rungs. The additivity rule passed
    there, so the prompt's rule alone would have allowed a per-term attribution. The joint
    reporting follows the measured interaction (deviation 14), and the document says so.
    **IMPLEMENTATION CHOICE.**

## Issue movements (§7)

- **Closed, moved to this board's §4 and deleted from the index:**
  - `[02-realistic-fixture-Gk-phase-is-the-superseded-construction]`;
  - `[04-table-cs-tau-fixture-does-not-cover-z-init-when-the-first-sample-is-dropped]`;
  - `[12-realistic-large-x-reads-the-retired-clamp-records]`.
- **Narrowed** (each note dated 2026-10-27 and pointing at SEAM-RESIDUE.md):
  - on this board, `[04-production-phase-gk-tau-table-has-a-zero-width-bottom-interval]`: the
    fixture half is fixed; whether `CumulativeTable` should refuse such nodes stays open;
  - on `source-remediation`, `[05-numeric-region-is-now-the-accuracy-floor]`,
    `[06-source-spline-residual-vs-handover]` and `[07-lg-derivative-truncation-at-handover]`;
  - on `GkTk-remedial`, `[00-tk-lg-truncation-floor]`.
- **Opened:** `[13-the-tlg-term-depends-on-the-levin-subdivision]`.
- **Counts:** the board's §3 goes from 19 to **17**. `docs/OPEN_ISSUES.md` goes from 116 to
  **114**, counted by rows.

## Observations not acted on

1. **A4's scripts no longer import at `HEAD`**, because `OffsetBesselPhaseGk` is gone. This is not
   opened as an issue. They are records of their own SHA and run at it (`28ebaf0`), as §1.5's list
   says, and nothing at `HEAD` needs them.
2. **`T-first`'s nearest approach to a stationary point is 0.0226 of the largest constituent's
   $\theta'$**, at $10^7$. That is no sign change, but closer than the orchestrator's reading of
   about −0.16 $k$ at 980. It is recorded in SEAM-RESIDUE.md §2 and is not an issue.
3. **No term is monotone in $x_{\rm resp}$** except $A_G$ on `q-smooth`. The sign follows the
   rung. That is recorded, not explained (SEAM-RESIDUE.md §4.5).

## State handed on

- **For E1 and E2:** the seam's residue on this fixture is the source spline's interior first, then
  the numeric $T$ and the Liouville–Green derivative terms. The phase constructions are not a
  factor.
- **Before any per-term budget is scored:** `[13-the-tlg-term-depends-on-the-levin-subdivision]`'s
  next step should be taken first.
- **To regenerate the tables:**
  `python -B docs/handover/b2/scripts/b2_tables.py <run dir>/checkpoint.jsonl`.

