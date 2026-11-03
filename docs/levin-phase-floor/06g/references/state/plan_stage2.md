# levin-phase-floor 06g §5, the references — plan for stages 2 and 3 (written in stage 1)

Stage 1 (done, 2026-11-01 ~11:10–11:30Z): read; copied 06h's census scripts (the census agent's
starting point, byte-identical to its current copies) into `scripts/` and
`state/scripts_06h_census_reference/`, 06h's phase driver/launcher/selector, 06e's slicer and
replay, 06i's scorer into `state/scripts_*_reference/` (`out/cmd/00`, `state/scripts_references.sha256`);
wrote the new scripts and the one changed copy, `x1_capture.py` (`out/scripts_diff_stage1.txt`).
**Nothing computed:** no capture, draw, reading, hash of a store or checkpoint, or unpickling.

All commands from `var/levin-phase-floor-06g/references/`, each through `scripts/rec.sh NAME ...`
(output in `out/cmd/NAME.txt` and `out/commands.log`); `PY=/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python`,
always `-B` with `PYTHONDONTWRITEBYTECODE=1`; `G=/Users/ds283/Documents/Code/SecondaryGWKit-lpf06g-1c5336e`.

## Stage 2 (when resumed, after the census is scored)

1. **State.** `git --no-optional-locks -C $G rev-parse HEAD` / `status --short --ignored` (and the
   parent worktree, the main checkout); `$PY -B -m runregistrykit list` (the census run `done`).
2. **The scratch copy, before:** `$PY -B scripts/hash_store.py "06g references before"
   <a3v2_copy>` → `10_copy_files_before`; `bash scripts/compare_hashes.sh
   <copy_files_certified.txt> out/cmd/10_copy_files_before.txt` → `11_copy_vs_certified` (EQUAL, no
   journal/-wal/-shm).
3. **The g census checkpoint:** its SHA-256, size, record count and `status.json` → `12_census06g`;
   06h's checkpoint `3104c18e…` (the draw and the selection check it again).
4. **The harness:** diff the census agent's final `census/scripts/` against
   `state/scripts_06h_census_reference/` → `13_census_harness_diff`; copy its scripts into
   `scripts/` (and unchanged into `state/scripts_06g_census_reference/`), re-applying my four
   marked `x1_capture.py` changes on top of its `x1_capture.py` if it changed it; check that the nine
   files of `worker.py`'s `SCRIPTS` equal, by SHA-256, the `scripts_sha256` in the g census's
   checkpoint records → `14_harness_equals_census`. Re-run the import check. Record the diff of my
   scripts against both sources → `out/scripts_diff_at_launch.txt`. Read the census agent's
   `worker.py` for the field that carries ΣE and fill the predictions' one **[stage 2]** item.
   *If the census agent changed the unit spec (`run_census.py`), mirror it in `x1_capture.py`
   (each change a deviation).*
5. **Predictions:** `state/predictions_draft_stage1.md` → `state/predictions_<UTC>.md`, the
   [stage 2] item filled and nothing else changed, `chmod 444`; the same text into `log.md`'s
   Predictions section, with the time.
6. **The draw:** `$PY -B scripts/draw_refs.py 20261101 state/predictions_<UTC>.md` → `20_draw`
   (writes `out/sample/draw.json`).
7. **The selection:** `$PY -B scripts/select_refs.py` → `21_select` (writes
   `out/sample/selection.json`: 90 calls, both censuses' records).
8. **Launch:** `$PY -B -m runregistrykit list` → `22_registry_list_prelaunch`; then
   `$PY -B scripts/launch_refs.py` → `23_launch` (copied to `state/launch_references.txt`); then
   **one** check → `24_launch_check`: `kill -0` on the driver's pid from `status.json`, the
   `status.json`, the first checkpoint line or heartbeat. **End the turn** with the run id, run
   directory, the driver's and the `sh` pids, and that line.

**The run** (`scripts/run_refs.py`, slug `references`, 368 units): capture (x1_capture.py, six
census workers, ~20–30 min) → check_captures.py → slice_refs.py → compare13.py → routes_refs.py →
the tool on 06i's 13 (52 readings, ~30–40 min on 06i's laptop times) and the command line on z25 and
z13 → compare06i.py → the 77 new samples (308 readings, each stopped at 30 min). Every stop condition
of the predictions file ends it `failed` with the stage named in its heartbeat; then I report and
ask.

## Stage 3 (when resumed with the final `status.json`)

1. The scratch copy again (`hash_store.py` → `40_copy_files_after`, `compare_hashes.sh` →
   `41_copy_after_vs_certified`); 06h's and the g census's checkpoints unchanged; the worktrees clean
   with `--ignored`, no `__pycache__`; nothing under `var/datastores/` newer than the stage-2 marker;
   the registry.
2. **Score** (`scripts/score_refs.py`, written then, from the predictions file's rules only):
   per-sample table (four readings, scatter, scatter over the tolerance, wall times, counts or
   why not); the 77's count (stop if < 39); V3; V4; the correction $-\sum E$; plus the 13's
   comparison from `out/tool_vs_06i.json`.
3. Records: `log.md` (README §5.1, every heading), `verification.md` (links `06g/references/…`),
   `issues.md`, `KEPT_ON_MAC_STUDIO.txt` (every pickle — `state/capture/*/inputs_*.pkl`,
   `regions_*.pkl`, `state/inputs/*.pkl` — and every file over 2 MB, with SHA-256 and size).

## Deviations recorded so far (for log.md; numbering may change)

1. **The captures, their checks, the group comparison and the tool on 06i's 13 are stages of the one
   registered run**, not in-session steps before the launch. STRUCTURALLY REQUIRED: the captures
   re-run about 90 census units (06h's phase capture took ~4 min for 20), and anything over about ten
   minutes is registered and detached (CLAUDE.md; README §5 rule 7). Each check stops the run
   before any reference reading.
2. **One reading per process** (`one_reading.py`, the tool's `reading()` with the command line's
   exact arguments), not the command line's four readings in one process. STRUCTURALLY REQUIRED by
   the brief's 30-minute stop and per-reading wall time. The command line itself is run on z25 and
   z13 and must agree bit for bit.
3. **The tool on 06i's 13 reads the captures at g, not 06i's inputs.** STRUCTURALLY REQUIRED: there
   is no worktree at 06e's tree on this host and making one is a git write; 06i's inputs loaded at g
   would be neither tree's integrand.
4. **06i's inputs' provenance.** The brief says they were made at 06e's census tree with 06e's
   census harness; 06e's `references/slice_inputs.py` copied them out of 06c's depth20-tiling
   captures (06c's census tree `e5764a6`), and 06e's `replay_g.py` showed they reproduce 06e's census
   calls; 06i read them at 06e's tree. No effect on the comparison (made with g on the path).
5. **The count rule also requires Route B's `all_monotone`** and four readings present.
   IMPLEMENTATION CHOICE, from the tool's docstring.
6. **06i's 13 are counted under 06g's tolerance** (their own four readings), as V3 counts "of the 90".
   IMPLEMENTATION CHOICE; 06i's own count reported beside.
7. **The harness copied in stage 1 is 06h's** (the census agent's copy was unchanged at 11:05Z); the
   census agent's final harness is synced in stage 2 and checked against the census run's recorded
   hashes. STRUCTURALLY REQUIRED.
8. **"At most six processes":** a reading runs six spawned worker processes under a parent that
   waits in `pool.map`, under the driver, which waits too; a capture runs six census workers under
   x1_capture.py, which polls. Six computing processes, as 06h's phase agent and 06i ran. Reading of
   the brief, to be confirmed.
9. **compare13.py checks the integrand as well as the criterion,** and **check_captures.py the region
   arrays and the harness's hashes as well as x1_capture.py's tuple.** IMPLEMENTATION CHOICE.
10. **A reading stopped at 30 minutes does not skip its sample's other readings.** IMPLEMENTATION
    CHOICE (the reference is dropped either way; the others are reported).
11. **The draw reads "finite abserr" on 06h's census record,** and **V4 is undecided if one of its four
    samples does not count.** Readings of the rules, fixed in the predictions draft.

## Stage 2 as run (2026-11-01, from 17:48Z), after the coordinator's message and addenda 1 and 2

- Harness synced to the census agent's final scripts (`out/cmd/11`); the nine `SCRIPTS` equal the
  census records' hashes (`out/cmd/12`; one set over all 18,560 records; `run_census.py` 5f21f072).
  Scratch copy EQUAL to its certification (`13`, `14`). Checkpoints: g census `03adcbba…`, 06h
  `3104c18e…` (`15`).
- Predictions `state/predictions_20261101T175120Z.md` (`825b6f04…`, 444); rule files
  `state/v8_rule_20261101T175144Z.md` (`04bacdc5…`) and `state/v1_rule_20261101T175144Z.md`
  (`d03fde45…`).
- Draw `20` (`out/sample/draw.json` `6e26e253…`); selection `21` (102 calls: Z3's 30, r00–r59,
  v00–v11; no V8 call among the 90; `f1d76007…`).
- **Part 1 launched** 17:52:31Z: `levin-phase-floor-06g-references-20261101T175231`, driver pid
  5223, sh pid 5219, 60 units: captures of all 102, checks, slice, compare13, routes, the tool on
  06i's 13, the command line on z25/z13, compare06i; it ends `done` there.

**Next (stage 2b, when part 1 ends):** project the total wall time from 06i's 13 (the coordinator's
5 h 40 min gate); build V1's rehearsal-store capture (rebuild.py's A3 v2 constants, store names and
rows for the R8 configuration, at p against R23 and at g against R24, the stores hashed before and
after) and part 2 (V3's 77 → V1's 12 → V8's 12); launch part 2.

Deviations added:
12. **The references are two registered runs (parts 1 and 2), both slug `references`,** split at
    the coordinator's cost gate after 06i's 13. STRUCTURALLY REQUIRED.
13. **V8's 12 are captured in part 1 with V3's 90** (their readings in part 2, after V1's).
    IMPLEMENTATION CHOICE.
14. **`scripts/harness_equals_census.py`** was first run from the session scratchpad (`out/cmd/12`)
    and copied into `scripts/` afterwards, byte-identical (both SHA-256 appended to `out/cmd/12`).
    UNINTENDED DRIFT.

## Part 1 ended (18:40:42Z, done, 60/60): results in out/part1_results.md

Cost projection for part 2 (`out/cost_projection_part2.txt`): by the mean per sample 3.59 h, but
Route A's QAWO time scales with the phase span and part 2's Route A spans reach 1.6e12 rad against
06i's 1.5e8; with each reading capped at 30 min, **about 20.2 h**, with 17 Route A references
projected to be dropped (QAWO res 2 over 30 min). Over the coordinator's 5 h 40 min gate: nothing
launched, V1's capture not built; the question goes to the coordinator.

## Stage 2b (2026-11-01, 19:59–20:15Z): V1's capture, part 2 built, per addendum 3

- Rehearsal stores hashed before any harness touched them (`out/cmd/40`, `41`): EQUAL to the
  rehearsal agent's baseline; again after the captures (`46`): EQUAL, no journal/-wal/-shm.
- V1 harness: `v1/scripts/` = the census agent's harness with marked changes (deviation 15); the stores
  through symlink directories `v1/stores/r23`, `r24` (deviation 16). Development trial on w00 at p
  (`43`): ran, total bit-identical to R23's. `v1_rows.py` (`42`), `v1_capture.py` (`44`): 24 of 24
  reproduce the stored group bit for bit in its stored form (1 + z_response) × the call's value and
  abserr (deviation 17); `v1_compare_groups.py` (`45`): 12 of 12 groups equal between p and g.
- `one_reading.py --progress` (deviation 18); checked bit-identical to part 1 on six readings (`47`).
- `make_part2_plan.py` (`48`): 380 readings, projected 48,670 s = 13.52 h; driver part 2 with the
  monitor and backstop (deviation 19), dry-tested in the session scratchpad on two readings
  (monitor line, backstop (b) stop, both worked; not kept).

Deviations added:
15. **The census harness pointed at the rehearsal stores** (addendum 1 allows it): in `v1/scripts/`,
    `rebuild.py`'s `A3V2_K_SAMPLE` = logspace(log10 1e5, log10 1e6, 3) and `A3V2_ZEND` = 250.0 (the
    rehearsal scope of `RunDrivers/quadsource_atol_sweep.py`), `prov.py`'s depth assertion and
    `STORE_DIR_BY_TREE` (p → R23, g → R24), `worker.py`'s 1e-8 tolerance made optional; units with no
    budget, no scan, no W3. STRUCTURALLY REQUIRED; no computed value changes (24 of 24 reproduce).
16. **The stores are reached through symbolic links in `v1/stores/`** carrying the file names
    `rebuild.py`/`rerun.py` open and those the primary's shards table records; nothing is created
    beside the stores. IMPLEMENTATION CHOICE (fewer code changes than renaming in four places).
17. **V1's "bit for bit" is against the store's form of the group:** `phase_group_Levin_integral`
    stores (1 + z_response) × the Levin call's value and abserr; the comparison applies that one
    double product. STRUCTURALLY REQUIRED. The references and ratios use the call's own values.
18. **`one_reading.py --progress`** replaces the tool's `_run_jobs` in its own process by the same
    runner with `pool.imap` and a progress file; values bit-identical (checked). IMPLEMENTATION
    CHOICE, for addendum 3's progress check.
19. **Part 2's driver**: no 30-minute stop; the monitor and the backstop of the ruling file.
    STRUCTURALLY REQUIRED by addendum 3.
20. **A read-only `sqlite3 -readonly` query of R23's primary `shards` table preceded the first
    store hash** (`out/cmd/40`). UNINTENDED DRIFT; the hashes equal the rehearsal baseline.
21. **V1's 24 captures ran in session** (about 6 minutes, two passes of 24 units at six workers),
    not registered. IMPLEMENTATION CHOICE (under ten minutes).

**Part 2 launched** 20:14:16Z: `levin-phase-floor-06g-references-20261101T201415`, driver pid 14269,
sh pid 14264, 380 units (`out/cmd/51`, `52`). Projected end 2026-11-02 ~09:45Z (48,670 s): V1 ~0.63 h,
V8 ~0.54 h, rest of V3 ~2.76 h (ends ~00:10Z), then the ten long QAWO r1 readings ~9.6 h.

## After part 2 (ended 2026-11-02 18:46:25Z; results in out/part2_results.md): the rerun batch

Addenda 4 (`5c8de4ad…`) and 5 (`44fd54ad…`) read in full. Plan `out/rerun_plan.md` / `.json`
(`out/cmd/60`): 67 readings, estimate 116,259–216,707 s (32.3–60.2 h).
Deviations added:
22. **One-reading's out-of-order progress** (`--progress-order unordered`: `imap_unordered` over
    indexed chunks, results reassembled in chunk order), and a `--rerun` marker. IMPLEMENTATION
    CHOICE per addendum 4/5; covered by the batch's six bit-for-bit re-reads (a development check on
    r55 qawo r1 in the scratchpad also matched, not kept).
23. **Stage order 3 → 2 → 1** (re-reads, then the 56 stationary-point readings, then the stopped
    QAWO readings shortest first). IMPLEMENTATION CHOICE: the long QAWO readings hold nothing else up.
24. **The w samples' PYTHONPATH is the tool's tree then `v1/scripts`**, as addendum 5 says; the
    unpickling function `harness._rebuild_lru_cache` resolves from one_reading.py's own directory
    (`references/scripts`), whose `harness.py` is byte-identical to `v1/scripts/harness.py`.
    IMPLEMENTATION CHOICE.
25. **r24 qawo r2 is skipped as the addenda list it,** though its phase-span estimate is 7.1–11.9 h
    (the addenda's "about 2 days" came from part 2's ordered progress). Reported, not decided.
26. **r26 qawo r1 runs:** its estimate is 14.4–23.9 h, under a day at both ends (by 187 s at the
    high end).

**Rerun batch launched** 2026-11-02T18:53:24Z: `levin-phase-floor-06g-references-rerun-20261102T185324`,
driver pid 83838, sh pid 83833, 67 units (`out/cmd/63`, `64`; the first re-read, z13 qawo r1,
bit-identical). Projected end 2026-11-04 03:11Z to 2026-11-05 07:05Z (116,259–216,707 s): re-reads
~2 min; the 56 stationary-point readings 1.8–9.3 h; the five stopped QAWO readings 30.5–50.8 h.

## Scored after stage 2 of the rerun batch (2026-11-02T20:44:20Z), per the user's ruling

`scripts/score_refs.py` (one process; the batch untouched) → `out/scores_after_stage2/`, kept as
`report_20261102T204420Z.md` / `scores_20261102T204420Z.json`.
27. **The scorer was run twice** (`out/cmd/70` at 20:42:56Z, `71` at 20:44:20Z): the second adds the
    V1 table's per-reading distances and the 720-row marks; no rule or number of the first changed;
    the first run's outputs were overwritten. UNINTENDED DRIFT (the first was a draft).
28. **"As written" requires clause (i)** (all_monotone, the predictions' text), so z00 and z12, usable
    only by clause (ii), cannot count as written; as ruled they are scored. IMPLEMENTATION CHOICE.

## Rescored per addendum 6 (2026-11-03T02:08:15Z)

`scripts/score_refs_a6.py` (score_refs.py's rules with clause (iii) and addendum 6's reading
sources; one process; the batch untouched) → `out/scores_addendum6/` (kept as `*_20261103T020815Z.*`);
`scripts/compare_z00_z12.py` → `out/compare_z00_z12.json`.
29. **The rescoring keeps every non-14 sample on its stage-2 readings,** leaving out the stage-3 QAWO
    r1 readings of r24 and r53 that have landed: they are for the separate recheck when stage 3 ends.
    IMPLEMENTATION CHOICE ("everything else unchanged except where these readings enter").
30. **The report's V1 heading and clause line keep score_refs.py's stage-2 wording** ("the rerun batch's
    readings", "every V1 reading meets clause (ii)"); the clause list under it shows (ii) for QAWO and
    (iii) for FCC, which is what holds. UNINTENDED DRIFT, cosmetic.

## The stage-3 recheck (2026-11-03T06:51:58Z)

Rerun batch done (67/67: 64 done, 3 stopped). `scripts/score_refs_s3.py` → `out/scores_stage3/`
(kept `*_20261103T065158Z.*`; `changes.txt`); `scripts/stage3_projection.py` →
`out/stage3_projection.json`; finding drafted in `issues.md`.
31. **The stage-3 recheck builds on the addendum-6 scoring** (its readings and clause (iii)) and adds
    only r24's and r53's QAWO r1; the comparison is against the addendum-6 scores. IMPLEMENTATION
    CHOICE.
