# Log 12 — B1: remove the $T_k$ seam gap, by overlap

**Prompt:** [`prompts/handover/12-remove-the-tk-gap.md`](../12-remove-the-tk-gap.md), Part A
(§1–§5, the code). §6, rehearsal R8, runs later on the Mac Studio, and its record is appended below
by the orchestrator.
**Commit:** the commit on branch `handover-12-b1` that adds this log — "Switch the T_k hand-over at
the first grid node below z_init" (its SHA cannot be embedded in itself)
**Model:** Claude Opus 5.5, in a fresh-context subagent on the laptop, dispatched at `98e68d3`
**Date:** 2026-10-25
**Result:** COMPLETE WITH DEVIATIONS (Part A). §1–§3 are implemented as written, tests (a)–(g) pass,
both breakages are caught, and the seven roots rise only by the tests added. No §8 stop condition
fired. R8 is to come.

The measurements are in [`docs/handover/HANDOVER-B1.md`](../../../docs/handover/HANDOVER-B1.md).
Suite logs, the breakage outputs and the probe scripts went to the session scratchpad.

## What shipped

**New `ComputeTargets/handover.py`** (79 lines), the one module both consumers use:
- `HANDOVER_OVERLAP_NODES = 2` (`:32`), with README §7 D1's wording as its comment.
- `node_z(node) -> float` (`:37`): the `.z` of a redshift-like node, or the float itself.
- `same_redshift(a, b) -> bool` (`:45`): `store_id` equality where both carry one, else exact float
  equality of the stored z. No tolerance and no $\log(1+z)$ round trip.
- `handover_switch_node(z_init, grid) -> Optional[node]` (`:61`): the largest node of `grid`
  strictly below `z_init`, returned as given so a caller can match it by identity; `None` if there
  is none.

**`ComputeTargets/TkSourceFunctions.py`**
- `self._crossover_z` is now `self._z_init` everywhere it meant the initialisation point: the
  cross-check against `z_exit - stop_deltaz_subh` (unchanged, message now says "initialisation
  redshift"), the phase anchor, `friction_F.delta(z_init, z)`, `H_init`.
- `__init__` builds the Liouville–Green side first, then the numeric side, because the numeric cut is
  made relative to the switch node. `_build_WKB` (`:445`) keeps its filter and now records the
  switch sample (its first sample), `_switch_z`, and `_switch_on_init` (the first sample lies at
  $z_{\rm init}$ within the filter's own tolerance, `:470`).
- `_build_numeric` (`:354`), rewritten: sorts the numeric values descending; finds the switch node
  among them by `same_redshift` (raises naming $k$, $z_{\rm init}$, $z_{\rm sw}$ if absent); unless
  `_switch_on_init`, checks that it is `handover_switch_node(z_init, numeric grid)` (raises if the
  Liouville–Green row skipped a node); requires `HANDOVER_OVERLAP_NODES` samples below it (raises,
  "only N numeric sample(s) lie below it"); splines everything at or above the switch node plus the
  overlap; keeps the switch and overlap samples for the continuity record.
- New property `z_init` (`:583`). `crossover_z` (`:591`) now returns the switch node.
  `numeric_region` (`:600`) is documented as reaching below `crossover_z`.
- New property `handover_continuity -> List[dict]` (`:610`): at the switch node and each overlap
  node, `{z, dT_rel, dTprime_rel}` with $(T_{\rm num}-T_{\rm LG})/|M|$ and
  $(T'_{\rm num}-T'_{\rm LG})/(|M|\omega)$, $T'_{\rm LG} = M(\mathrm{d}\ln M/\mathrm{d}z\sin\theta+\omega\cos\theta)$,
  from the stored numeric samples and the Liouville–Green accessors at those stored nodes. Computed
  on first access and cached; a copy is returned.
- The module docstring's "Regions and the sampled ranges" (`:148`) is rewritten: the three
  redshifts, that the numeric data do not stop at the hand-over (118–126 samples below
  $z_{\rm init}$ on A3 v2), the new regions, and that both representations are evaluable on
  $[{\tt numeric\_region}[1], {\tt crossover\_z}]$. `_check_numeric`/`_check_WKB` messages name the
  switch node and $z_{\rm init}$.

**`ComputeTargets/QuadSource.py`**
- `numeric_crossover_z` renamed `numeric_z_init` (`:34`), with `numeric_crossover_z = numeric_z_init`
  kept as an alias (`:64`): `ComputeTargets/tests/test_quadsource.py` imports the old name, and
  nothing else outside this file uses it.
- The region walk in `compute_quad_source` (`:131-180`): if either factor has a $z_{\rm init}$, the
  region is the source grid, sorted descending, down to `handover_switch_node(max z_init, grid)` and
  `HANDOVER_OVERLAP_NODES` nodes below it; it raises if no node lies below $z_{\rm init}$ or the
  grid ends inside the overlap. It no longer clamps to the factors' sampled coverage: every node
  walked must be in both factors' values, and `_lookup` raises otherwise. With no $z_{\rm init}$ at
  all, the old shared-coverage walk is kept. The payload's `crossover_z_q`/`_r` are the two
  $z_{\rm init}$ (as before).
- The "audit A3" comment that said the numeric grids stop at the hand-over, the
  `QuadSourceFunctions` comment, the `crossover_z_q`/`_r`, `numeric_region` and `store()` comments
  are corrected; the class docstring says what the stored values cover, that this is a value change
  and not a schema change, and that a consumer evaluates the spline only down to the switch node.

**`ComputeTargets/QuadSourceIntegral.py`**
- Removed: `HANDOVER_CLAMP_MAX_GRID_STEPS`, `_ClampedTk`, `_ClampedSource`, `_check_gap`,
  `_mean_source_grid_step`, and the private helpers only they used, `_ClampedPhase`, `_as_log` and
  `_log1p_z`. None of the five had a user other than the clamp. A comment at `:104` says what the
  clamp was, what handover 11 measured, and why nothing replaces it.
- `build_partition` (`:187`): the $T$ breakpoints are `crossover_z`, now the switch nodes. Each
  $T$ factor gets `_check_region_covers` — the Liouville–Green region where oscillatory, the numeric
  region open above where smooth — labelled "Liouville-Green/numeric representation of Tq (switch
  node z=…, z_init=…)"; the source spline gets it on every all-smooth sub-interval. The "lies below
  the lowest Liouville-Green sample" check is kept. Sub-interval dicts carry the
  `TkSourceFunctions` objects themselves.
- Metadata: `max_clamp_gap_log1pz` and every `clamp_gaps_log1pz` are gone; new `handover`
  (`:413`): per factor `{z_init, z_switch, handover_continuity}`. `crossover_z_q`/`_r` stay, and are
  now the switch nodes.
- `evaluate_QuadSource_integral`: the cross-check compares `TkSourceFunctions.z_init` with
  `QuadSource`'s recorded $z_{\rm init}$ (`:653`); `source_f` is the spline itself (`:674`).
- Docstrings of `phase_group_Levin_integral`, `numeric_quad_integral` and `total_abserr` no longer
  mention the clamp.
- **No `extract_*` reader reads a removed key** (grep of the repository outside `docs/` and
  `prompts/`), so §3.3's reader adjustment is empty.

**Tests** (`ComputeTargets/tests/`)
- `test_tk_source_functions.py`: `Fixture` gains `numeric_nodes_below_init`
  (default `NUMERIC_NODES_BELOW_INIT = HANDOVER_OVERLAP_NODES + 3`): its numeric samples continue on
  the Liouville–Green grid's own nodes past $z_{\rm init}$; `z_numeric` keeps its meaning and the
  extra nodes are `z_numeric_below`; `z_switch` is the first stored Liouville–Green sample.
  `drop_first_WKB_sample`'s default is unchanged. `AnalyticRadiationFixture` gets the same five nodes.
  New class `TestHandOverSwitch` with (a)–(d). `DECIDED_OVERLAP_NODES = 2` is the number the overlap
  tests assert.
- `test_quadsource.py`: `TestQuadSourceRegion` in production's shape ($z_{\rm init}$ at the
  log-midpoint of two nodes, values five nodes past it); new (e).
- `test_quadsource_integral.py`: `Case`'s source grid continues `HANDOVER_OVERLAP_NODES + 1` steps
  past the earlier $z_{\rm init}$; `ExactTkFunctions` and `_MinimalTk` gain `z_init` and an empty
  `handover_continuity`; `TestSeams` passes the functions unwrapped; `TestHandOverClamp` becomes
  `TestHandOverSeam` with (f), (g), `test_LG_row_that_skips_the_switch_node_raises` and the kept
  `test_response_below_the_LG_samples_raises`; helpers `production_seam_inputs` and `_ShortenedLG`.

The seven tests of prompt §4.2:
- (a) `test_tk_source_functions.TestHandOverSwitch.test_a_z_init_is_the_initialisation_point_and_is_cross_checked`
- (b) `…TestHandOverSwitch.test_b_crossover_is_the_first_LG_sample_and_the_switch_node`
- (c) `…TestHandOverSwitch.test_c_numeric_spline_reaches_exactly_the_overlap`
- (d) `…TestHandOverSwitch.test_d_handover_continuity_is_at_the_representation_floor`
- (e) `test_quadsource.TestQuadSourceRegion.test_e_region_reaches_the_overlap_below_the_earlier_switch_node`
- (f) `test_quadsource_integral.TestHandOverSeam.test_f_partition_records_no_gap_and_a_shortened_representation_raises`
- (g) `test_quadsource_integral.TestHandOverSeam.test_g_production_seam_reproduces_the_representation_term_not_the_clamp`

The kept `:906-911` test is `TestConsistencyChecks.test_crossover_inconsistency_raises`.

## Deviations from the prompt

### 1. The Gate, as amended by the orchestrator — STRUCTURALLY REQUIRED

The Gate asks that `runregistrykit list` show nothing `running` or stale. It showed one,
`levin-phase-floor-02-noise-probe-20261025T183809` (pid 70545), flagged stale because its heartbeat
was old; its pid answered `kill -0`. It belongs to another session, uses no Ray and no store, and is
not this prompt's to touch or end. This prompt launches no run, uses no Ray and opens no store, so
the orchestrator amended the Gate to allow exactly that run. No other run was `running` or stale.
`git status --short` was empty at dispatch.

### 2. A git worktree — STRUCTURALLY REQUIRED

The work was done in the worktree `SecondaryGWKit-handover-12`, on branch `handover-12-b1` cut at
`98e68d3`, not in the main checkout. Another session's running script
(`docs/levin-phase-floor/noise_injection.py`, untracked in the main checkout) imports two of the files
B1 changes, so editing them in place would have changed a running measurement's code.

### 3. The helper's module — IMPLEMENTATION CHOICE

§2.1 asks for "one helper, in one module". It is a new module, `ComputeTargets/handover.py`, which
also holds `HANDOVER_OVERLAP_NODES` and `same_redshift`. The alternative was `TkSourceFunctions.py`.
A separate module gives the overlap constant one definition that `TkSourceFunctions`, `QuadSource`
and the tests all import, so the breakage of §4.3 is one edit, and `QuadSource` does not import the
transfer-function assembly to get a grid helper.

### 4. Names kept — IMPLEMENTATION CHOICE

- `numeric_crossover_z` is renamed `numeric_z_init` and the old name kept as an alias, as §2.1 allows,
  because `test_quadsource.py` imports it.
- `QuadSource.crossover_z_q`/`_r` and the payload keys of the same names keep their names, and are
  documented as $z_{\rm init}$. Renaming them was not asked for, and they are what §3.4's cross-check
  reads.
- `metadata["partition"]["crossover_z_q"]`/`_r` keep their names and now hold the switch nodes,
  with a comment saying so. `docs/handover/a3-v2/seam/scripts/skipped_check.py` reads them.

### 5. A Liouville–Green sample at $z_{\rm init}$ itself is the switch — IMPLEMENTATION CHOICE

§1.1 defines `crossover_z` as the largest Liouville–Green sample *at or below* $z_{\rm init}$, and
§2.1 the helper as the largest node *strictly below* it. They differ only when a sample sits at
$z_{\rm init}$. `Fixture`'s default shape (`drop_first_WKB_sample=False`) is exactly that, and §4.1
forbids changing the default. So `TkSourceFunctions` treats a first sample within the
Liouville–Green filter's own tolerance of $z_{\rm init}$ ($10^{-7}(1+z_{\rm init})$) as the switch,
and skips the helper check there. Both representations are stored there, so there is no gap.
- **Production.** A grid node that close to a root is measure-zero. If it happened, `main.py`'s
  `truncate(z_init, keep="lower")` would include it in the WKB grid, so refusing it would refuse a
  geometry production can produce and that has no gap.
- **`QuadSource`.** It uses the strict helper in that case too, so its region reaches one node
  further. That costs nothing.
- **The alternative,** raising, would have broken every test on the default fixture.

### 6. Fixtures besides `Fixture` — STRUCTURALLY REQUIRED

§4.1 names `Fixture`. Three more fixtures had the pre-B1 shape and would have raised under the new
code, so they were given production's shape as well.
- **`AnalyticRadiationFixture`** builds a `TkSourceFunctions`.
- **`TestQuadSourceRegion`'s `build_Tk`** placed $z_{\rm init}$ exactly on a node and stopped the
  values there; it now puts it between two nodes and continues five past.
- **`Case`'s `FakeQuadSource` grid** now continues past the earlier $z_{\rm init}$, as `QuadSource`
  now stores.

No production code was changed to do this (§8, third condition).

### 7. Tests removed, renamed or rewritten because their premise was the clamp — STRUCTURALLY REQUIRED

- **Removed:**
  - `test_quadsource.TestQuadSourceRegion.test_region_is_clamped_to_the_sampled_coverage`. It
    asserted that the region is clamped to the factors' sampled coverage, which §2.2 removes; (e)
    asserts the replacement.
  - `test_quadsource_integral.TestHandOverClamp.test_gap_is_recorded_and_bridged`. It asserted that
    a one-step gap is recorded and bridged; (f) and (g) replace it. Its last number at `98e68d3` is
    in HANDOVER-B1.md §4.
- **Renamed:** `test_gap_beyond_tolerance_raises` became
  `test_LG_row_that_skips_the_switch_node_raises`. The fixture manipulation is the same: five
  Liouville–Green samples dropped. The refusal now comes from `TkSourceFunctions`, since there is no
  tolerance to exceed, and its message is asserted. The class `TestHandOverClamp` is now
  `TestHandOverSeam`.
- **Rewritten to B1's semantics:**
  - `test_tk_source_functions.TestConsistencyChecks.test_WKB_grid_starting_below_the_hand_over`:
    `WKB_region[0] == crossover_z < z_init`, evaluation above refused at $z_{\rm init}$, numeric region
    below the switch.
  - `TestNumericRegion.test_reproduces_exact_solution`: `numeric_region[1]` is now the second overlap
    node.
  - `test_crossover_inconsistency_raises`, the `:906-911` test that §4.2 (a) says to keep: kept, with
    its consistency assertion now on `functions.z_init`. On that fixture the two coincide, so the
    test passes either way.
  - `TestQuadSourceRegion`'s region and store tests: the new extents.
  - `TestSeams`: `_ClampedTk(...)` unwrapped.

Every other existing test whose *numbers* moved because of the extended grids is listed, with
before and after, in HANDOVER-B1.md §4.

### 8. Test (g): what "reproduce" is taken to mean — IMPLEMENTATION CHOICE

- **The configuration.** T-first at $x_{\rm resp}=980$, $b=0$, built as A2 built it: the
  `fixture_x_max` rule, tolerances (1e-45, 1e-12), `Case._fixtures` seeded and then restored.
  T-first because both transfer functions hand over inside the range there; it is the cheaper of
  the two shapes for which that holds (9–11 s, against 28 s for `together`). In `q-smooth`, $T_q$
  never hands over in range.
- **The reference.** `kohri_terada.total_from_I_RD` in double precision, whose returned error
  (1.7e-14 here) is asserted below 1e-3 of the tolerance. A2 used 50 digits for the small-$u$
  shapes, which T-first is not.
- **The tolerance.** B1 cannot reproduce A2's closed cell to that cell's declared error. It switches
  one node lower and takes the $T$ numeric and source splines' end intervals off the seam, and
  HANDOVER-B1.md §2.2 measures each of those moving N + 9/8 by up to 5e-3. So the test asserts the
  scale: $|N_{\rm B1}+9/8| < |N_{\rm A2,closed}+9/8| = 3.387\times10^{-4}$ (measured 6.04e-05), and
  more than half a clamp term from the open cell. That is the reading of "reproduce its
  representation term, not its clamp term" that can be asserted.
- **What it does not cover.** On `together` B1 gives 3.47e-03, larger than A2's 2.68e-03
  representation term but 130 times below its clamp term. The test does not cover this, and it is
  recorded for B2 (below).

### 9. How (d) catches `HANDOVER_OVERLAP_NODES = 0` — IMPLEMENTATION CHOICE

§4.3 says "(c) and the continuity bound must fail". The residuals at the switch node compare stored
samples, so their values do not depend on the overlap. (d) therefore asserts the record's coverage
as well: the switch node and `DECIDED_OVERLAP_NODES` overlap nodes, by identity with the fixture's
nodes. With no overlap it holds one node and fails. The overlap tests assert the decided number, 2,
and not the module constant, so that a change to the constant is caught rather than followed.

### 10. More docstring than §1.5 named — IMPLEMENTATION CHOICE

§1.5 names the "Regions" section and the comment in `_build_numeric`. Three more places said the
same false thing, so they were corrected too:
- `TkSourceFunctions`'s module docstring said the hand-over has "no overlap region", and described
  the stop as integrating "down to a phase minimum";
- `QuadSourceIntegral.total_abserr` listed "the hand-over clamp" among the ingredient errors;
- `numeric_quad_integral` described `source_f` as "the clamped wrapper".

### 11. The smooth $T$ check is open above — IMPLEMENTATION CHOICE

A smooth factor's coverage check uses `(inf, numeric_region[1])`, because $T = 1$, $T' = 0$ exactly
above its first numeric sample (spec 01 §2.8) and the accessors already return that. A check against
`numeric_region[0]` would refuse every super-horizon sub-interval.

## Verification performed

**The Gate** (at dispatch, in the worktree): `git status --short` empty; `runregistrykit list`, one
stale-but-alive run of another session, pid answering `kill -0` (deviation 1); the grep, outside
`docs/`, `prompts/` and `venv/`, found `ComputeTargets/QuadSourceIntegral.py` (`:111`, `:196`,
`:243`, `:301`, `:310`, `:348`, `:438`, `:509`, `:548`, `:800`, `:1036`) and
`ComputeTargets/tests/test_quadsource_integral.py` (`:47`, `:51`, `:847-848`, `:891-892`, `:943`,
`:950`), as the orchestrator recorded.

**The seven roots.** Run serially, each by its own command from `CLAUDE.md`.

| root | before (`98e68d3`, orchestrator) | after (B1) |
|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK |
| ComputeTargets | 737, FAILED (failures=1: `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, the known wall-clock flake) | **742 OK** |
| CosmologyModels | 39 OK | 39 OK |
| LiouvilleGreen | 148 OK (skipped=1) | 148 OK (skipped=1) |
| ObjectFactories | 280 OK | 280 OK |
| RunDrivers | 127 OK (skipped=1) | 127 OK (skipped=1) |
| tools | 49 OK | 49 OK |

The rise of 5 in `ComputeTargets` equals the tests added:
- seven new tests, (a)–(g);
- one test renamed (`test_gap_beyond_tolerance_raises` → `test_LG_row_that_skips_the_switch_node_raises`),
  which leaves the count alone;
- two removed (deviation 7).

That is 737 + 7 − 2 = 742. The wall-clock flake passed on both after-runs. An independent run of
the `ComputeTargets` root on an export of `98e68d3` in the scratchpad gave 737 OK (skipped=1: its
`git show` test cannot run without a `.git`). That run is the "before" of HANDOVER-B1.md §4.

**Tests (a)–(g).** All pass. Their numbers:
- **(d)**, on the exact fixture in production's shape: $|\Delta T_{\rm rel}| \le 3.2\times10^{-14}$
  against a bound of 1e-11. $|\Delta T'_{\rm rel}| \le 8.6\times10^{-6}$ ($w=1/3$, bound 2.2e-5) and
  $\le 6.5\times10^{-5}$ ($w=0.2$, bound 3.8e-4). Each bound's provenance is in HANDOVER-B1.md §1.
- **(g)**: N + 9/8 = −6.0375e-05, against A2 closed +3.387e-04 (declared 9.2e-09) and open −3.574e-03.
  The eq. (22) reference error is 1.7e-14.
- **(f)**: the largest shortfall of any representation on any sub-interval is 0.0 in
  $\log(1+z')$. Shortened by one node, the source spline raises "sub-interval bottom z=353.13 is
  out-of-bounds for the QuadSource spline of f", and $T_r$'s Liouville–Green region raises
  "sub-interval top z=353.13 is out-of-bounds for the Liouville-Green representation of Tr (switch
  node z=353.12705, z_init=361.3686) region (345.07, 5.9282)".

**Deliberate breakage** (§4.3). Each run was on `test_tk_source_functions`, `test_quadsource`,
`test_quadsource_integral` and `test_phase_groups` (88 tests), and each change was reverted with
`sed` and the revert checked by `grep`.
- **`HANDOVER_OVERLAP_NODES = 0`:** 13 failures and 1 error. They include (c) at both $w$ and (d) at
  both $w$, as required, plus (e), (f), (g) and five existing region tests.
- **The switch back at $z_{\rm init}$** (`crossover_z` returning `self._z_init`): 7 failures and 2
  errors. They include (b) at both $w$ and (f), as required, plus (a), (d), (g) and the WKB-grid
  test.

The full lists are in HANDOVER-B1.md §3. `git status` afterwards showed only B1's files, and the
`ComputeTargets` root was re-run after both reverts.

**The Gate's grep, re-run** at the end, outside `docs/`, `prompts/` and `venv/`, has one hit:
`ComputeTargets/tests/test_quadsource_integral.py`, test (f)'s `assertNotIn("clamp_gaps_log1pz", record)`,
which asserts the key's absence. Every other hit is under `docs/`:
- **`docs/handover/realistic_large_x.py`** (`:45`, `:194`, `:332`, `:496`, `:608`, `:666`): A2's
  script. The prompt forbids editing it, and a re-run after B1 is B2's. Its `:332` and `:666` read
  `item["clamp_gaps_log1pz"]`, and `:356` reads `partition["max_clamp_gap_log1pz"]`; after B1 each
  raises `KeyError`. Opened as an issue (below).
- **`docs/handover/a3-v2/seam/scripts/seam_census.py`**: a record, byte-identical to its run, whose
  outputs record its SHA-256. Not edited.
- **`docs/source-remediation-verification/analyse_quadsource_integral.py:297`** reads the key with
  `.get(…, {})`, and `:254`/`:290` read `max_clamp_gap_log1pz` with `.get`. A post-B1 row reads as
  gap-free, with `nan`/0 for the allowance. It does not raise.

**`black`** on every touched `.py` file: clean (`--check`, 7 files).

## Observations not acted on

- **`docs/handover/realistic_large_x.py` no longer runs at `HEAD`.** It reads
  `item["clamp_gaps_log1pz"]` (`:332`, `:666`) and `partition["max_clamp_gap_log1pz"]` (`:356`),
  which B1 removes, so each cell raises `KeyError`. Its docstring also describes `_ClampedTk`
  (`:45`, `:194`). The prompt forbids editing it (its checkpoint is gated on its own SHA-256), and
  §4.1 makes a re-run after B1 B2's. Opened as
  `[12-realistic-large-x-reads-the-retired-clamp-records]`, owner **B2**.
- **`docs/levin-phase-floor/noise_injection.py`.** It is another session's, untracked in the main
  checkout and so not in this worktree. It imports `realistic_large_x` and
  `test_quadsource_integral`'s `Case`. After B1 it is a record of the SHA it ran at, not a script
  that runs at `HEAD`. Recorded in the same issue.
- **`docs/handover/a3-v2/seam/scripts/seam_census.py`** reads the removed keys. It is a record,
  byte-identical to its run, so it stays as written.
- **Stale mentions of the clamp outside B1's paths:**
  - `ComputeTargets/primitive_phase.py:73` and `:102` list `QuadSourceIntegral._ClampedPhase` among
    the consumers of the phase protocol;
  - `ComputeTargets/tests/test_primitive_phase.py:416`'s docstring says the same;
  - `main.py:1887-1893` says the stored `QuadSource` values "end at the larger of the two Tk
    hand-over redshifts".

  `primitive_phase.py` and `main.py` are outside this prompt's paths. The test docstring is inside
  them but was not asked for. Recorded in the same issue.
- **A pre-B1 store's `QuadSource` rows cannot feed a B1 `QuadSourceIntegral`.** Their values stop at
  the last node at or above $z_{\rm init}$, and the strict source check now refuses them ("…
  out-of-bounds for the QuadSource spline of f"). A3 v2's are such rows. A store is read at its own
  SHA (no version-aware readers), and R8 builds a fresh store, so nothing here needs it. But a
  resume of an old store at B1's tree must recompute `QuadSource` before its integrals. Its
  `TkNumericIntegration` rows are fine: they already hold the overlap nodes. Recorded in the same
  issue.
- **`together` at $x_{\rm resp}=980$ after B1 is 3.47e-03 from the reference.** That is larger than
  A2's representation term there, 2.68e-03 (HANDOVER-B1.md §2.2). The other two shapes fall by five
  to eight times. This is a measurement for B2 to attribute; it is not a defect found here.

## State handed to the next step

- **For R8** (prompt §6, the Mac Studio): the code is the commit on `handover-12-b1` that adds this
  log, cut at `98e68d3`. The laptop's counts are `ComputeTargets` 742 and `ObjectFactories` 280.
  R8 must find differences from R3 in `QuadSource` and `QuadSourceIntegral` only:
  - **`QuadSource`:** `HANDOVER_OVERLAP_NODES` + 1 = 3 more `QuadSourceValue` rows per row. The
    walk used to end at the last node above the earlier $z_{\rm init}$; it now ends at the switch
    node, the next one down, and two nodes below that. The stored values at the old nodes are
    unchanged.
  - **`QuadSourceIntegral`:** changed `total`s and `metadata`, with `partition.handover` and no
    `clamp_gaps_log1pz`.
- **The header's issues are not moved here.** `[08-handover-clamp-error]`,
  `[12-handover-clamp-error-in-production]` and `[05-numeric-region-is-now-the-accuracy-floor]` move
  to the `source-remediation` board's §4, and `[06-source-spline-residual-vs-handover]` is narrowed,
  only after R8 holds, in B1's records commit (prompt §5).
- **For B2:**
  - the continuity record is `metadata["partition"]["handover"][label]["handover_continuity"]` on
    every new row, and `TkSourceFunctions.handover_continuity` on any $T$ row, A3 v2's included;
  - HANDOVER-B1.md §2.2 gives A2's three $x_{\rm resp}=980$ cells re-measured at B1, with the
    end-interval terms separated;
  - `realistic_large_x.py` needs its two key reads replaced before it can re-run (the issue above).

---

## §6 — rehearsal R8 on the Mac Studio

*To be appended by the orchestrator after R8 runs.*
