# Prompt 12 — B1: remove the $T_k$ seam gap, by overlap

**Campaign:** [`README.md`](README.md) · **Board item:** **B1** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-25, on the user's decision of the same day: README §7 **D1** is (b), with the
Liouville–Green initialisation kept at the phase extremum.
**Where it runs:** in two parts.
- **§1–§5, the code:** on the laptop, in a fresh-context subagent, as 01–05 and 09 did. One commit.
- **§6, rehearsal R8:** on the Mac Studio, orchestrated there, after the code commit is pushed.
  The laptop's Ray is in use by ChamPBH and StochasticInstantons (the user's decision U2,
  `prompts/levin-phase-floor/README.md` §6.1). The records are committed on the laptop afterwards, in B1's records commit.

**Closes, when §6 holds:**
- `[08-handover-clamp-error]` and `[12-handover-clamp-error-in-production]`;
- `[05-numeric-region-is-now-the-accuracy-floor]`, as far as the $T_k$ hand-over goes.

All three are on the `source-remediation` board. **Narrows:** `[06-source-spline-residual-vs-handover]`
(the source spline no longer ends at the hand-over).
**Recommended model:** **Opus**.

**Read first:**
1. README §2 (a) **with its correction of 2026-10-25**, §2 (b), and §7 **D1**, decided.
2. [`docs/handover/HANDOVER-MECHANISM.md`](../../docs/handover/HANDOVER-MECHANISM.md) §0.1 and §3,
   and [`docs/lg-phase-and-handover-followup-2026-09.md`](../../docs/lg-phase-and-handover-followup-2026-09.md)
   §1.4, the continuity check this change makes possible.
3. `ComputeTargets/TkSourceFunctions.py`, in full. Note especially:
   - the module docstring's "Regions and the sampled ranges";
   - `__init__`'s cross-check of `crossover_z` against `z_exit - stop_deltaz_subh`;
   - `_build_numeric`, whose filter is the cut;
   - `_build_WKB`.
4. `ComputeTargets/QuadSource.py`: `numeric_crossover_z` (`:31-53`) and the region walk
   (`:110-150`), which is the second cut.
5. `ComputeTargets/QuadSourceIntegral.py`. The clamp:
   - `HANDOVER_CLAMP_MAX_GRID_STEPS` (`:111`), `_ClampedTk` (`:196`) and `_ClampedSource` (`:243`);
   - `_check_gap` (`:291`), `_mean_source_grid_step` (`:316`) and `build_partition` (`:327-570`);
   - the crossover cross-check in `evaluate_QuadSource_integral` (`:784-795`).
6. `Quadrature/integrators/numeric_with_phase_cut.py:649-660` and `:758-776`. The stop-mode
   solve runs to the window's end, $z_{\rm exit,subh,e6}$, and the extremum is found afterwards.
   Also `main.py:1366-1374` (the numeric grid, down to $0.85\,z_{\rm e6}$) and `:1593-1600` (the
   WKB grid: the nodes below $z_{\rm init}$).
7. `ComputeTargets/tests/test_tk_source_functions.py`: `Fixture` (`:197-270`), with its
   `drop_first_WKB_sample`, and the hand-over tests near `:900-960`.
   `ComputeTargets/tests/test_quadsource_integral.py`: every test that names the clamp
   (`grep -n 'clamp\|Clamp\|gap' …`).
8. `docs/handover/a3-v2/seam/out/seam_census_11.txt` (handover 11): the gaps this removes, on all
   of A3 v2.

**Gate:**
- `git status --short` is empty, and `runregistrykit list` shows nothing `running` or stale;
- the seven roots' counts are taken at the dispatch `HEAD` (`CLAUDE.md`);
- `grep -rn 'HANDOVER_CLAMP_MAX_GRID_STEPS\|_ClampedTk\|_ClampedSource\|clamp_gaps_log1pz' --include='*.py' .`
  outside `docs/`, `prompts/` and `venv/` is recorded, so that §5 can show every use is accounted
  for.

---

## 0. What this prompt is

### 0.1 Why the gap exists

The gap is not in the solution, and not in what is stored. Around each $T_k$'s
$z_{\rm init}$ there are two grid nodes, $N_{\rm above} > z_{\rm init} > N_{\rm below}$:
- **The numeric row.** `TkNumericIntegration` is solved to the end of the stop window, and stores
  samples at every grid node down to it. On A3 v2 that is 118–126 nodes below $z_{\rm init}$,
  $N_{\rm below}$ among them (README §2 (a), corrected).
- **The Liouville–Green row.** `TkWKBIntegration` is initialised at $z_{\rm init}$, at the
  extremum, and stores samples from $N_{\rm below}$ down.
- **The two cuts.** `TkSourceFunctions._build_numeric` throws away every numeric sample below
  $z_{\rm init}$. `QuadSource` walks only nodes above $\max(z_{\rm init,q}, z_{\rm init,r})$.
- **The switch.** `QuadSourceIntegral` switches representation at $z_{\rm init}$ itself.

So neither spline covers $(N_{\rm below}, N_{\rm above})$, and the clamp bridges it. Handover 11
measured the result: median 0.5 and maximum 0.98 grid step, on every row with a Levin
sub-interval.

### 0.2 What changes

Three definitions, held apart from here on:
- **$z_{\rm init}$:** the Liouville–Green initialisation point, the extremum. **Unchanged.** It is
  still cross-checked against `z_exit - stop_deltaz_subh`, and still keys the WKB row.
- **The switch node, $z_{\rm sw}$:** the largest source-grid node strictly below $z_{\rm init}$.
  It is the Liouville–Green row's first sample, and the point where consumers change
  representation.
- **The overlap:** `HANDOVER_OVERLAP_NODES = 2`, the number of numeric nodes kept below
  $z_{\rm sw}$. That keeps the switch node at least two intervals inside the numeric spline, out of
  its end intervals (`[05-numeric-region-is-now-the-accuracy-floor]`).

**Nothing about the solution changes.** `TkNumericIntegration`, `TkWKBIntegration`,
`numeric_with_phase_cut`, `find_phase_extremum`, every `store()`, the stop window and every grid
are untouched. **No table's schema changes.** What changes is which stored samples the two
consumers read, where they switch, and the clamp's removal.

## 1. `TkSourceFunctions`

1. **Expose $z_{\rm init}$ and the switch node separately.** `crossover_z` becomes the switch node.
   - `z_init` is the extremum. The existing cross-check moves onto it, unchanged.
   - `crossover_z` is $z_{\rm sw}$, the largest Liouville–Green sample at or below $z_{\rm init}$.
   - Check that the same node is in the numeric values, **by redshift identity** (the `redshift`
     object's `store_id` where both carry one, else exact float equality of the stored `z`). It is
     never matched by a $\log(1+z)$ round trip (`CLAUDE.md`, "Redshift arithmetic"). A node
     missing from the numeric values raises, naming $k$, $z_{\rm init}$ and $z_{\rm sw}$.
2. **The numeric spline** keeps every sample at or above $z_{\rm sw}$, plus the
   `HANDOVER_OVERLAP_NODES` numeric samples immediately below it. Fewer available raises, naming
   how many were found. `numeric_region` reports the true sampled extent, which now reaches below
   `crossover_z`.
3. **The Liouville–Green spline is unchanged.** It keeps samples at or below $z_{\rm sw}$, which is
   every sample it has.
4. **The continuity record.** At $z_{\rm sw}$ and at each overlap node, record:
   - $(T_{\rm num} - T_{\rm LG})/|M|$ and $(T'_{\rm num} - T'_{\rm LG})/(|M|\,\omega)$, with $M$ the
     Liouville–Green amplitude and $\omega$ its frequency there;
   - as a property `handover_continuity`: a list of `{z, dT_rel, dTprime_rel}`.

   Both representations are **stored** at these nodes, so this compares stored samples. It is not
   an interpolation of either. This is the follow-up §1.4 check.
5. **The docstring.** Correct the module docstring's "Regions" section and the comment in
   `_build_numeric`, both of which say the numeric data stop at the hand-over. Say what is now
   true, with the A3 v2 count.

## 2. `QuadSource`

1. **A shared helper.** `numeric_crossover_z` keeps returning $z_{\rm init}$, and is renamed to
   say so (keep the old name as an alias if anything outside this file uses it). Add one helper,
   in one module, that both `QuadSource` and `TkSourceFunctions` use for the switch node:
   `handover_switch_node(z_init, grid)`, the largest node of `grid` strictly below `z_init`.
   `QuadSource` has no `TkWKBIntegration`, so it finds the node on its own `z_sample`.
   `TkSourceFunctions` checks that its Liouville–Green first sample is that same node.
2. **The region walk** keeps nodes down to the switch node of the factor that hands over
   **earlier**, plus `HANDOVER_OVERLAP_NODES` nodes below it. Every such node must be present in
   both factors' numeric values. The existing `_lookup` raises otherwise, and it should.
3. **The comments** that say the numeric grids stop at the hand-over (`:120-125`, "audit A3") are
   corrected.
4. **What the stored values now mean.** A `QuadSource` row stores the source at more nodes. Its
   `numeric_region` grows downwards by `HANDOVER_OVERLAP_NODES` + 1 nodes or so. **That is a value
   change, not a schema change.** Consumers may evaluate the spline only above the switch node;
   the extra nodes exist for the spline's end intervals. Say so in the class docstring.

## 3. `QuadSourceIntegral`

1. **The breakpoints** of `build_partition` are the switch nodes. `crossover_z` from
   `TkSourceFunctions` is now $z_{\rm sw}$, so this follows. The $G_k$ breakpoint is unchanged.
2. **The clamp is removed.**
   - `HANDOVER_CLAMP_MAX_GRID_STEPS`, `_ClampedTk`, `_ClampedSource`, `_check_gap` and
     `_mean_source_grid_step` go, unless something other than the clamp uses one; say which.
   - Each $T$ factor and the source spline then gets the strict coverage check $G_k$ already has
     (`_check_region_covers`). A representation that does not cover its sub-interval **raises**,
     naming the factor, the sub-interval and the region.
   - There is no tolerance. The geometry now guarantees coverage, so a gap is a defect to report,
     not to bridge.
3. **The metadata.** `partition` loses `max_clamp_gap_log1pz` and every `clamp_gaps_log1pz`. It
   gains `handover`: per factor, `z_init`, `z_switch`, and the factor's `handover_continuity`.
   Stored values change; `metadata` is a JSON string, so no schema changes. `extract_*` readers
   that read the removed keys are adjusted (the Gate's grep says whether any do).
4. **The cross-check** in `evaluate_QuadSource_integral` compares like with like:
   `TkSourceFunctions.z_init` against `QuadSource`'s recorded $z_{\rm init}$.

## 4. Tests

The stand-in pattern of `test_tk_source_functions.py`. No Ray, and no datastore.

1. **The fixture becomes production's shape.** `Fixture`'s numeric grid stops at the crossover
   today. Production's runs past it, so extend the fixture's numeric samples to at least
   `HANDOVER_OVERLAP_NODES + 3` nodes below $z_{\rm init}$, as `numeric_with_phase_cut` does.
   **Do not change `drop_first_WKB_sample`'s default.** `docs/handover/realistic_large_x.py`
   imports `Fixture` and must not be edited (its checkpoint is gated on its own SHA-256). The
   tests that model production pass `drop_first_WKB_sample=True` explicitly. Every existing test
   whose numbers change because of the extended numeric grid is listed in the log, with why.
   - A2's committed tables stay the record of the pre-B1 tree. A re-run of
     `realistic_large_x.py` after B1 would measure the new consumer, which is B2's job, not this
     prompt's.
2. **New tests**, each named in the log:
   - (a) $z_{\rm init}$ is the extremum, unchanged, and the cross-check still fires on a perturbed
     `stop_deltaz_subh` (keep `:906-911`'s test);
   - (b) `crossover_z` is the first Liouville–Green sample and equals `handover_switch_node`;
   - (c) the numeric spline reaches exactly `HANDOVER_OVERLAP_NODES` nodes below it, and raises
     with one node fewer;
   - (d) on the exact fixture, `handover_continuity` is at the fixture's own representation floor.
     Quote the bound's provenance from the existing tolerance comments;
   - (e) `QuadSource`'s region reaches the overlap below the earlier switch node, and raises if a
     node is missing;
   - (f) `build_partition` on a fixture records no gap anywhere, and a representation shortened by
     one node raises;
   - (g) end to end on the realistic fixture with the seam as production has it, `total` against
     the exact reference. **The number to compare** is A2's "seam closed" realistic cell at the
     same configuration (`docs/handover/REALISTIC-LARGE-X.md` Tables 2–4). B1 should reproduce
     its representation term, not its clamp term.

     Choose one configuration at $x_{\rm resp} = 980$, so that the test is cheap, and state the
     tolerance with its reference's own error (README §5 rule 8).
3. **Deliberate breakage.**
   - Set `HANDOVER_OVERLAP_NODES = 0`: (c) and the continuity bound must fail.
   - Put the switch back at $z_{\rm init}$: (b) and (f) must fail.

   Record each in the log, then revert it.
4. **The seven roots** at the end: counts before and after. A fall is a stop. A rise equals the
   tests added.

## 5. Records

- The Gate's grep is re-run, and every remaining hit is explained.
- `black`, then one commit, with the log and the board (B1's row and item). The issues of the
  header move to their boards' §4 only after §6 holds, in the records commit.

## 6. Rehearsal R8 — on the Mac Studio, after the code commit is pushed

Orchestrated there, as handover 10 was. It writes under `var/handover-12/`, and its records are
committed on the laptop.

1. **The checkout.** Fast-forward to the code commit. The roots `ComputeTargets` and
   `ObjectFactories` give the laptop's counts.
2. **The lock:** `acquire`, then `set-pid` on the launch.
3. **R8 build.** `RunDrivers/quadsource_atol_sweep.py --build --rehearsal` as R3 ran it
   (`rrk-v020-adoption` 03 §2), into `var/datastores/handover-12-reference.sqlite`, registered
   (`--campaign handover --prompt 12`). The run ends `done`.
4. **The fingerprint moves**, and it must move only where B1 says.
   - `compare_with_recorded.py` against R3's recorded reference must show differences **only** in
     the `QuadSource` and `QuadSourceIntegral` tables.
   - Any other table that differs is a **stop**: it would mean a producer changed.
   - Record the new fingerprint as R8, the rehearsal reference from here on.
5. **On R8's store, read-only:**
   - every `QuadSourceIntegral` row has a `partition.handover` record, and no `clamp_gaps_log1pz`;
   - the distribution of `handover_continuity` (`dT_rel`, `dTprime_rel`), by $k$ and node;
   - the `QuadSource` rows' extent against R3's, and the number of nodes added.
6. **The readers** (`docs/dk-and-rpk-migration/rehearsal/run_readers.sh`): every one exits 0, and
   the store is byte-identical before and after.
7. **Release the lock.** `ray-processes` is empty.

The continuity residuals are **recorded, not judged**. What size they should be is B2's question,
on the exact and realistic fixtures. A3 v2's own $T$ rows already hold both representations at
the overlap nodes, so B2 can also score production's $T_k$ without a rebuild.

## 7. Acceptance

1. §1–§3 as written. Every hit of the Gate's grep is removed or explained.
2. §4: (a)–(g) pass; both breakages caught; the seven roots, with counts that rise only by the
   tests added.
3. §6: R8 ends `done`; it differs from R3 in `QuadSource` and `QuadSourceIntegral` only; no row
   records a clamp gap; the readers exit 0; the lock is released.
4. Logs: `logs/12-remove-the-tk-gap.md`, in `qcd-background-audit` README §5.1's form, with §6's
   Mac Studio record appended by the orchestrator; `docs/handover/HANDOVER-B1.md` holds the
   measurements.

## 8. Stop conditions — stop and ask

- The Gate fails.
- A production `TkNumericIntegration` would have to change, or a `TkWKBIntegration`, the stop
  window, a grid or a schema. That is outside D1 as decided.
- A test fixture cannot be given production's shape without a production change.
- R8 differs from R3 in any table other than `QuadSource` and `QuadSourceIntegral`.
- A seven-roots count falls.
- The lock is held, or will not release.
