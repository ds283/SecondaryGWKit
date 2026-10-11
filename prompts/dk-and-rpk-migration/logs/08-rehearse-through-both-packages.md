# Log — prompt 08, rehearse through both packages (R2)

**Prompt:** [`../08-rehearse-through-both-packages.md`](../08-rehearse-through-both-packages.md) ·
**Board items:** L1–L4 (R2), G-RPK · **Date:** 2026-10-11 · **Model:** Opus 5.5 ·
**Commit:** `f539015` · **Result:** landed.

**R2 holds:**
- Both stores are `39809dca…`, all 21 classes. They are equal to each other and to the
  fingerprints recorded by R1's reference, R0's reference, 07's reference and 09's rehearsal.
- L1–L4 hold against R1, the resumed chain's prune identity vacuously (below).
- **`main.py:3247` (C8) is measured:** every count of the statistics block is R1's.
- **RayPoolKit's G2 holds.**
- Every suite is at its dispatch count. Nothing was opened or closed.

**Where it ran.** Dispatched at `5631f0e` ("Write the orchestrator note for dk-and-rpk-migration
prompt 08"), whose code is `643c3b2`'s but for one test comment.
- The tree was clean, in the only worktree.
- `RunRegistry list` showed 25 runs (23 finished, 2 `unknown`), none `running`.
- There was no Ray process at all, and `RAY_ADDRESS` was unset.
- The venv holds `datastorekit` 0.2.2 at `6d8ee9ae…` and `raypoolkit` 0.2.0 at `eaef0ce6…`, neither
  editable.
- Every launch ran at `5631f0e` with `git status --short` empty.

The record is
[`docs/dk-and-rpk-migration-verification.md`](../../../docs/dk-and-rpk-migration-verification.md)
§3. This log points into it rather than repeat it.

## What shipped

- **`docs/dk-and-rpk-migration-verification.md`**: a new **§3, "R2 — through both packages"**.
  - It has §2's subsections in the same order, every command verbatim.
  - It adds §3.5.6, the printed lines and the registry's fields, and §3.11, "`main.py:3247`",
    which holds C8's measurement.
  - §1 and §2 are not edited.
- **`docs/dk-and-rpk-migration/rehearsal/`**: nineteen saved outputs, each named `*_08*`:
  - the run records and pre-launch checks: `run_records_08.txt`, `prelaunch_checks_08.txt`;
  - the interrupter: `interrupter_dummy_test_08.txt`, `interrupter_build_dummy_test_08.txt`,
    `interrupter_firings_08.txt`;
  - the stores: `store_sidecars_08.txt`, `closed_store_state_08.txt`, `prune_check_08.txt`;
  - the fingerprints and integrity: `fingerprint_comparison_08.txt`,
    `fingerprint_against_recorded_08.txt`, `census_08.txt`, `store_soundness_08.txt`;
  - the guard and L4: `guard_08.txt`, `stage_counts_08.txt`, `policy_statistics_08_reference.txt`,
    and, for C8, `policy_statistics_08_resumed-b.txt` and `policy_statistics_08_resumed-c.txt`;
  - the readers: `readers_08.txt`;
  - `log_comparison_08.txt`. It holds the printed-line counts, the line-kind comparison of every
    log with R1's same slug, the `** ` sequences, the `status.json` stages, the stage times and the
    reader logs' comparison.

  No script there changed, and no new script was added.
- **`prompts/dk-and-rpk-migration/measure/out/08-raypoolkit-measure_client_pools.md`**:
  RayPoolKit's `measure_client_pools.py sgk --commit 5631f0e`, written under `var/` first and
  copied here after the last launch.
- **This log, and the board.** `docs/OPEN_ISSUES.md` is unchanged, since nothing was opened.
  README is unchanged.
- **Under `var/`, not committed:**
  - the two stores and their sidecars;
  - four run directories and their launch logs;
  - `var/dk-and-rpk-migration-08/` (verification §3.9).

## The runs

| Slug | Run id | Python pid (`sh` pid) | Interrupter pid | State | Exit code | Driver exit status | Wall time | Load before / after |
|---|---|---|---|---|---|---|---|---|
| `reference` | `dk-and-rpk-migration-08-reference-20261011T224347` | 55979 (55974) | — | `done` | 0 | 0 | 11 m 25 s | 12.82 15.94 14.77 / 33.02 29.06 22.79 |
| `resumed-a` | `dk-and-rpk-migration-08-resumed-a-20261011T225623` | 64487 (64482) | 64545 | `killed` | 143 | 143 | 4 m 41 s | 22.83 26.98 22.25 / 32.48 32.31 26.83 |
| `resumed-b` | `dk-and-rpk-migration-08-resumed-b-20261011T230305` | 69056 (69051) | 69114 | `killed` | 143 | 143 | 2 m 48 s | 16.79 27.73 25.57 / 19.22 27.64 26.17 |
| `resumed-c` | `dk-and-rpk-migration-08-resumed-c-20261011T230717` | 71947 (71942) | — | `done` | 0 | 0 | 5 m 39 s | 15.16 25.83 25.56 / 32.77 57.25 43.64 |

**Fingerprints.** All four are format 6, problems `{}`, 21 classes:
- `reference` and `resumed-c`:
  `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`;
- `resumed-a`: `ff6c4257…`;
- `resumed-b`: `aa860ed8…`.

**Every manifest** has:
- `git_head` `5631f0ed…` and `git_dirty` false;
- `packages` equal, as JSON, to the prompt's §2 record, both packages at their tags' commits,
  neither editable;
- `script_sha256` `b7935877…`, the same in all four, and R1's.

After each launch the agent made one check and ended its turn. The orchestrator's one wait per
launch ended `ENDED finished` each time, with the heartbeat never stale. The machine was shared:
over the builds the one-minute load at the checks ran from 13 to 33, and the five-minute load
reached 57.

**The interrupter.**
- **The dummy tests** ran before the first launch, 22:43:01–22:43:26, each into a new scratch
  directory. Both exited 0, with every verdict as R0's and R1's records: five cases, then three
  `--build` cases. No child or interrupter was left alive (16 pids, `ps -p` empty).
- **The points**, from R2's own reference `stdout.log`. Neither stage printed a progress line, and
  both queues have a 300 s minimum.
  - **`resumed-a`:** `'^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 10`,
    R0's and R1's arguments.
    - The reference's `GkSource` stage took 41.8 s, so the signal falls 10–15 s in, well before
      the end even at R0's 32.2 s.
    - It became eligible at 23:00:52 and sent SIGTERM at 23:01:02. The handler's line is stamped
      23:01:02, a delay under 1 s.
    - **0 of 291 `GkSource` rows were stored** (below, observation 1).
  - **`resumed-b`:** `'^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 100`, not
    anchored at the end.
    - The reference's QSI stage took 299.2 s, so 100 s is about a third of it, and before the end
      even at R0's 212.3 s. The orchestrator confirmed the hold before the launch.
    - It became eligible at 23:04:04 and sent SIGTERM at 23:05:44. The handler's line is stamped
      23:05:45.
    - 281 of 864 `QuadSourceIntegral` rows were stored. No progress line appeared during the hold.
- Both `stderr.log`s end `SystemExit(1); recording killed, exit code 143`. Each was recorded
  `killed`/143, and each driver exited 143.

**The killed launches' Ray agents.** There were none to wait for.
- The checks were the whole-table search for `SecondaryGWKit/venv` with `ray`, `pgrep -lf
  'gcs_server|raylet|ray::'` with other sessions' shells excluded, and `ps -p` on the run's pids,
  the interrupter's and every Ray pid its logs name.
- All were empty at 23:01:45 (~41 s after `resumed-a`'s final heartbeat) and at 23:06:41 (~48 s
  after `resumed-b`'s).
- No wait by pid was run. The agents' exit time is "none listed at the first check", as at R0 and
  R1.

## Deviations from the prompt

1. **The gate's "no commit after 07's touches code".** `af26914`, 07's review, changed one comment
   in `RunRegistry/tests/test_pipeline_adoption.py`. The gate cannot hold literally. It holds in
   substance:
   - `git diff --stat 643c3b2 5631f0e -- . ':!prompts'` lists that file alone;
   - its `ast` is equal at the two commits;
   - so the code that runs at the dispatch `HEAD` is `643c3b2`'s.

   **STRUCTURALLY REQUIRED**, by the orchestrator's note (§0, correction 1).
2. **The printed lines are compared as the completion lines, not progress lines.** R1 printed no
   progress line in any build, so there was none to compare. R2 compares, slug by slug:
   - the count of `-- ALL WORK ITEMS COMPLETE in time …`, each followed by `Queue summary: …`;
   - the `(no compute items …)` lines, in form and number;
   - the registry's `stage` fields.

   R2's builds printed no progress line either. The two R2 printed, in the GkWKB reader, are in the
   copy's form (verification §3.5.6). **STRUCTURALLY REQUIRED**, by the orchestrator's note (§0,
   correction 2).
3. **`prune_serials.py` was not run.** It hardcodes `a3-v2-readiness` 07's serials. `prune_check.py`
   ran in its place, on R2's own `closed_store_state.py` output. **IMPLEMENTATION CHOICE**, by the
   orchestrator's note (§0, correction 3).
4. **The saved measurement waited for the last launch.** Writing
   `measure/out/08-raypoolkit-measure_client_pools.md` before a launch would have stamped that
   manifest `git_dirty: true`. It was written into `var/dk-and-rpk-migration-08/` at 23:20, after
   `resumed-c` had ended, and copied into `measure/out/` unchanged. **STRUCTURALLY REQUIRED**, by
   the orchestrator's note (§0, correction 4).
5. **C8 is traced through the installed `datastorekit`, not the files the checklist names.**
   RayPoolKit's checklist item 5 cites SGK's `Datastore/SQL/ShardedPool.py`,
   `Datastore/SQL/Datastore.py` and `Datastore/object.py`, which 04 deleted. The trace uses
   `datastorekit/SQL/ShardedPool.py:3441`, `:3531` and `datastorekit/SQL/Datastore.py:661`, `:733`
   (verification §3.11). **STRUCTURALLY REQUIRED**, by the orchestrator's note (§0, correction 5).
6. **The statistics block is compared as counts and as sorted lines.** README §2.2 says "line for
   line". R2's reference block differs from R1's in the order of `complete` and `minimal` under
   `mixed`, and so does `resumed-b`'s from R1's `resumed-b`. Sorted, each pair is equal, and every
   count is equal. This follows `[02-the-policy-statistics-block-orders-its-lines-by-completion]`.
   **STRUCTURALLY REQUIRED**, as the prompt directs.
7. **L4's resumed chain is compared by identity, not by count.** The raw counts depend on where each
   launch was killed. R2 compares 05's four identities instead (verification §3.5.3).
   **STRUCTURALLY REQUIRED**, as 05 directs.
8. **`resumed-b`'s hold is 100 s**, against R1's 95 and R0's 70: a third of R2's own QSI stage,
   299.2 s. **IMPLEMENTATION CHOICE.**
9. **The Ray-process check excluded other sessions' shells.** Before `reference`, `pgrep -lf
   'gcs_server|raylet|ray::'` listed two pids. Both were `zsh -c` shells of another session, working
   in DatastoreKit's scratchpad, whose command text holds the pattern. From then on the check
   listed the same table with `zsh -c` and `grep` lines excluded, beside the unchanged whole-table
   search. Both were empty before every launch. **IMPLEMENTATION CHOICE.**
10. **The log comparison used a throwaway script**, kept in the agent's scratchpad and not
    committed, with §2.2.5's normalisation. Its output is `log_comparison_08.txt`. It also
    compares the printed-line counts, the `status.json` stages and the reader logs.
    **IMPLEMENTATION CHOICE.**
11. **Two more statistics blocks are saved**, `policy_statistics_08_resumed-b.txt` and `_resumed-c.txt`.
    R1 saved only its reference block, but C8 is measured on `resumed-b` too. **IMPLEMENTATION
    CHOICE.**
12. **The readers ran under the Bash tool's background mode**, inside the session, with their
    output to `var/dk-and-rpk-migration-08/readers.txt`. While they ran, the agent ran only reads
    that open no store: the stage counts, the statistics, the log comparison and RayPoolKit's
    script, which reads SGK through `git`. The suites ran after the readers and the guard, serially,
    in the foreground. **IMPLEMENTATION CHOICE.**
13. **The first dummy-test pid check missed one child.** It listed only `pid N` lines, so case C's
    child of the `--build` test (named as `child 55697`) was not in it. A second check at 22:43:39
    took every `pid N` and `child N` pid (16) and found none alive. Both are in
    `prelaunch_checks_08.txt`. **UNINTENDED DRIFT**, in the agent's tooling, corrected before the
    first launch.

## Verification performed

**L1–L3** (verification §3.2–§3.4):
- **L1.**
  - Every launch ended as recorded, and each driver's exit status agrees.
  - No Ray process of this repository was left after any launch.
  - Every manifest is as the prompt's §2 requires, and every `stdout.log` is as §1.2.5's table.
  - The guard refused before Ray started, named the five files, changed nothing and created no run.
- **L2.**
  - **The fingerprints.**
    - Both stores are format 6, `39809dca…`, all 21 classes equal, each equal to its sidecar's.
    - `compare_fingerprints` is empty between them, and against the recorded fingerprints of:
      - `dk-and-rpk-migration-05-reference` (R1);
      - `dk-and-rpk-migration-02-reference` (R0);
      - `a3-v2-readiness-07-reference`;
      - `datastore-generic-09-rehearsal`.
  - **The prune and the recompute.**
    - `resumed-a` left no `GkSource` row, so none unvalidated; `prune_check.py` reports "nothing to
      check" after `resumed-b` and after `resumed-c`.
    - `GkSource` 0 + 291 − 0 = 291; QSI 281 + 583 = 864. No stored row was recomputed, and
      nothing is duplicated.
  - **Integrity.**
    - The version row `(1, '2025.1.1')` is on all 8 shards after every launch.
    - The census gives `no divergence` on both stores, and equals R1's output once the prompt
      number is normalised.
    - On all 10 files, `integrity_check` is `ok`, `foreign_key_check` is empty, and the package's
      `read_inventory` finds no problem. The output equals R1's `store_soundness_05.txt` once the
      prompt number is normalised.
- **L3.**
  - All six readers exit 0, and so does `extract_Gk_data.py` without `--models`, with the skip line.
  - Nothing refused, and every store file is byte-identical. `store fingerprint` afterwards matches
    `resumed-c`'s.
  - Lookups equal R1's reader by reader. File counts differ in the three readers that draw random
    subsamples.

**What runs is the package** (verification §3.1.1):
- `RayTools/` is absent from the tree and from `git ls-tree` at `643c3b2` and `5631f0e`.
- In a fresh `python -I`, `import RayTools` and `import RayTools.RayWorkPool` raise
  `ModuleNotFoundError`.
- `raypoolkit.__file__` is in the venv's `site-packages`, and `RayWorkPool.__module__` is
  `raypoolkit.RayWorkPool`.

**L4 against R1** (verification §3.5):

| | R1 | R2 |
|---|---|---|
| Statistics: total | 291 | 291 |
| numeric | 109, all complete | 109, all complete |
| mixed | 26: 25 complete, 1 minimal | 26: 25 complete, 1 minimal |
| WKB | 156, all complete | 156, all complete |
| Block, line for line | — | differs in the order of two lines under `mixed`; equal sorted; equal to R0's line for line |
| `resumed-b`'s block | the same counts | the same counts; order differs from R1's `resumed-b`; equal sorted |
| `resumed-c`'s block | the same counts | equal to R1's `resumed-c` line for line |

| Stage | R1 `reference` | R2 `reference` |
|---|---|---|
| HORIZON EXIT TIMES, SOURCE K-SAMPLE | 3/3/3 | 3/3/3 |
| HORIZON EXIT TIMES, RESPONSE K-SAMPLE | 3/0/0 | 3/0/0 |
| NUMERICAL PART OF MATTER TRANSFER FUNCTIONS | 3/3/3 | 3/3/3 |
| WKB PART OF MATTER TRANSFER FUNCTIONS | 3/3/3 | 3/3/3 |
| QUADRATIC SOURCE TERMS | 6/6/6 | 6/6/6 |
| NUMERICAL PART OF TENSOR GREEN FUNCTIONS | 1368/1368/1368 | 1368/1368/1368 |
| WKB PART OF TENSOR GREEN FUNCTIONS | 3477/3477/3477 | 3477/3477/3477 |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | 291/291/291 | 291/291/291 |
| APPLY GKSOURCE POLICIES | 291/291/291 | 291/291/291 |
| QUADRATIC SOURCE INTEGRALS | 864/864/864 | 864/864/864 |

| Resumed-chain identity | R1 | R2 |
|---|---|---|
| Finished stages compute 0 on the next launch | yes | yes |
| Pruned at `resumed-b`'s open = left unvalidated by `resumed-a` | 45 = 45 | 0 = 0 (vacuous) |
| `resumed-b` `GkSource` = 291 − stored + pruned | 235 = 291 − 101 + 45 | pruned 0; computed 291 − 0 + 0 = 291 |
| `resumed-c` QSI = 864 − stored by `resumed-b`, nothing else | 530 = 864 − 334 | 583 = 864 − 281 |
| Totals over the chain | 291, 864 | 291, 864 |

**The stage sequence.** The reference's 26 `** ` lines equal R1's in order, once the prompt number,
the path and the numbers are normalised. Its 15 stage banners are equal verbatim.

**The printed lines and the registry** (verification §3.5.6):
- In all four runs, the completion lines (10, 7, 9, 10), the `Queue summary` lines and the
  `(no compute items …)` lines (1, 1, 7, 9) equal R1's in number. Normalised, their forms equal
  R1's.
- No build printed a progress line, at R1 or R2. The GkWKB reader's two progress lines have the
  copy's form.
- Each run's `status.json` `stage` is R1's: `reference` and `resumed-c` end `CALCULATE QUADRATIC
  SOURCE INTEGRALS, all work items complete`. `resumed-a` was killed at `CALCULATE GREENS
  FUNCTIONS FOR SOURCE REDSHIFT` and `resumed-b` at `CALCULATE QUADRATIC SOURCE INTEGRALS`.
  `progress` is `null` in all four, as at R1. Both interrupters read their stage.

**C8, `main.py:3247`** (verification §3.11):
- **By reading.** The queue's computed rows leave the pool at `raypoolkit/RayWorkPool.py:529-530`,
  where the post handler now gets the persisted object. That object is the local object pickled to
  the shard actor and back, with `store_id` set (`datastorekit/SQL/ShardedPool.py:3441`, `:3531`;
  `datastorekit/SQL/Datastore.py:661`, `:733`; the factory's `store`, `:172`, `:174`). The handler
  reads only `policy`, `type` and `quality`, which travel with it.
- **The measurement.** In `reference` and `resumed-b` the stage stored 291 rows, and the block
  totals 291, so the handler ran 291 times on the persisted object. Every count is R1's.
- **Not exercised.** `resumed-c`'s block came through the lookup exit, with `obj`, and does not
  exercise C8.
- **C8 changes nothing the block records.** No issue is opened.

**The stage times** (not compared):

| Stage | R2 reference | R1 reference | R0 reference |
|---|---|---|---|
| `GkSource` | 41.8 s | 55.1 s | 32.2 s |
| QSI | 4 m 59.2 s | 4 m 49.7 s | 3 m 32.3 s |
| The whole build | 11 m 25 s | 12 m 13 s | 8 m 0 s |

**The log differences from R1, slug by slug** (verification §3.2.5). With times, pids, paths and
numbers stripped:
- **`reference` and `resumed-a`:** identical in all three logs, line count and kinds.
- **`resumed-b`:**
  - It lacks the SciPy `IntegrationWarning`'s four lines, which R1 had there. R2 has them in
    `resumed-c`, as R0 did. R2's `resumed-b` stored 281 QSI rows before its point, against R1's
    334, so that one integral fell after R2's point.
  - It lacks the integrity warning's 12 kinds (40 `stdout.log` lines). R2's `resumed-b` had nothing
    unvalidated to prune.
- **`resumed-c`:** gains the warning's four lines, in `stderr.log` and the launch log.
- **Both killed handlers** print `SystemExit(1)`, as at R1.
- **Nothing else.** There is no traceback and no new warning in any build log. There is no line
  from the new pool, and no `ValueError`, `ray.kill`, killed actor, `ActorDiedError` or
  `RayActorError`.

**G-RPK, part by part** (verification §3.10):

| Part | Rests on | Holds |
|---|---|---|
| 1. `RayTools/` is gone | `643c3b2` (07, reviewed); `git ls-tree 643c3b2 RayTools` and `git ls-tree 5631f0e RayTools` empty; re-checked at `5631f0e` | ✅ |
| 2. 0 imports, 0 `import_module` calls, 54 sites `constructs` | RayPoolKit's script at `5631f0e` (this prompt), saved as `measure/out/08-raypoolkit-measure_client_pools.md` | ✅ |
| 3. The pin is in place and installed | `b6cc119` (06, reviewed); `requirements.txt:76`; all four R2 manifests record `raypoolkit` 0.2.0 at `eaef0ce6…`, not editable | ✅ |
| 4. The six roots pass | this prompt's suites at `5631f0e` (code of `643c3b2`), below | ✅ |
| 5. R2 reproduces `39809dca…` | `dk-and-rpk-migration-08-reference-20261011T224347` and `dk-and-rpk-migration-08-resumed-c-20261011T230717`, both `done`, both `39809dca…`, equal to R1's, R0's, 07's and 09's | ✅ |

**RayPoolKit's measurement against the expected.** Run from RayPoolKit's root (clean, `c9b0e11`,
unchanged after) with its venv. It equals 07's saved run but for three things:
- the commit;
- the tracked files, 1474 against 1471;
- the non-`.py` lines naming `RayTools` or `RayWorkPool`, 651 against 475, all 176 more in `prompts/`.

Against the orchestrator's run at `af26914` (1473, 638), the one more file and 13 more lines are
the note committed at `5631f0e`.

**R0 and R1 untouched.** Their 24 files have the same SHA-256, mtime (to the nanosecond) and size
before the first launch (22:42) as after the last step (23:31). They equal the orchestrator's
dispatch table. There is no journal file.

**The suites.** Run serially, in the foreground, after the last launch, the readers and the guard,
with README §7's command.

| Root | Dispatch baseline | Now | Load (1 min) |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 25.6 |
| CosmologyModels | 39 OK | 39 OK | 25.6 |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 23.7 |
| RunRegistry | 286 OK | 286 OK | 19.8 |
| ComputeTargets | 693, 1 wall-clock flake | **693 OK** | 42.6 |
| Datastore | 320 = 54 + 103 + 101 + 62 OK | 320 = 54 + 103 + 101 + 62 OK | 6–9 |

`test_tk_wkb_phase.TestCost.test_wall_time_per_object` failed at dispatch and passed here, so it
was not re-run alone. No Python file was added, so `black --check` had nothing to check.

## The deliberate-breakage record

None (README §5 rule 8). This prompt adds no code and no test, and names no mutation. The check is
the fingerprint comparison itself, which R0 showed can fail: 02's review compared against recorded
fingerprints from other trees. The scripts' failure paths were exercised on real inputs:
- `policy_statistics.py` exits 1 on `resumed-a`, which has no block.
- `prune_check.py` reports "nothing to check" on the state after `resumed-a` and after `resumed-b`.
- The guard refuses an existing store.

## Observations not acted on

1. **`resumed-a` stored no `GkSource` row, so R2 does not exercise the `GkSource` prune at open.**
   - **The evidence.**
     - The interrupter first read the `GkSource` banner at 23:00:52 and sent `SIGTERM` at
       23:01:02, 10–15 s into the stage.
     - The closed store after it holds 0 `GkSource` rows on all four shards.
     - At the same `--hold 10`, R1's `resumed-a` had stored 101 rows (45 unvalidated), and R0's 119
       (43 unvalidated).
   - **The consequence.** `resumed-b` pruned nothing and computed 291 − 0 + 0 = 291. L2's "the
     prune removed only unvalidated rows" holds at R2 vacuously. What covers that path is R1, run
     with the same `datastorekit` 0.2.2: its prune of 45 rows, all serials absent after
     (verification §2.3.2). The prune runs in the package's shard actors at open, not in the work
     queue.
   - **What is not known.** The record does not show why no row had landed 10–15 s in.
     **Inference, not measurement:** the load. The WKB stage before it took 2 m 50.7 s against
     R1's 2 m 15.1 s, and the load at the check after the run was 32.
   - **Why it is not an issue.** The pool's loop schedules as the copy did: a diff of `RayTools/RayWorkPool.py` at
     `789b72d` against the installed file gives only the licence header, the imports, the
     constructor's checks, `ref.hex()`, `results`' size and C8. So nothing read points at the pool.
     No issue opened, by the orchestrator's ruling. A third store is forbidden.
2. **The statistics block's order moved again.**
   - R2's reference and `resumed-b` print `minimal` first, R0's reference order.
   - R1's reference and `resumed-b` printed `complete` first.
   - `resumed-c` prints `minimal` first at R1 and R2.

   This is `[02-the-policy-statistics-block-orders-its-lines-by-completion]`, already open, with
   one more instance and no new fact. Nothing was opened.
3. **`extract_GkWKB_data.py`'s log holds two matplotlib `UserWarning`s that no earlier reader log
   holds.** They are from one `plot_Gk` worker: `Data has no positive values, and therefore cannot
   be log-scaled` and `No artists with labels found to put in legend`.
   - The reader draws its source redshifts at random (`sample`, `extract_GkWKB_data.py:439-441`).
     Each plot task takes its two `object_get` refs straight from the task builder (`:448-479`),
     not through the queue's results. So the queue cannot change what a plot is given, and the
     files written are 2436, R1's.
   - R1's record names only three readers as drawing random subsamples. `extract_GkWKB_data.py`
     draws one too (`z_subsample`), though its file count does not vary.
4. **No registered run has yet printed a pool progress line** under the package. R2's reference QSI
   stage took 299.2 s, 0.8 s inside the queue's 300 s minimum.
   - So `StageTracker`'s reading of a progress line into `status.json` `progress` is shown only
     by `RunRegistry/tests/test_pipeline_adoption.py`, which passes. The GkWKB reader's two lines
     show the strings, not a registry run.
   - A longer build would print one.
5. **`pgrep -lf 'gcs_server|raylet|ray::'` is fooled by another session's shell** whose command
   line holds the pattern (deviation 9).
   - CLAUDE.md's liveness rule forbids pattern matches for waiting, not for this pre-launch listing.
   - This false positive cannot hide a Ray process; it can only list a non-Ray one.
6. **The machine was heavily loaded by processes not of this repository.**
   - The five-minute load reached 57 during the builds, and the one-minute load 252 during the
     readers. The top CPU users were OrbStack, `runningboardd` and the system's `deleted` cache
     daemon.
   - `var/`'s volume is 93 % full (68 GB free).
   - The load slowed the stages, and changed nothing they wrote.
7. **The intermediate fingerprints differ from R1's** (`ff6c4257…` and `aa860ed8…` against
   `b6b89d84…` and `09170e40…`), as they must: each digests a store killed at a different point.
8. **DatastoreKit's checkout is mid-release** in another session, towards `0.2.3`. R2 ran at the
   `v0.2.2` pin. Nothing there was read or changed.

## State handed to the next prompt

- **R2 holds, and RayPoolKit's G2 holds** (board §2), resting on:
  - `b6cc119` (06) and `643c3b2` (07);
  - `f539015`;
  - the runs `dk-and-rpk-migration-08-reference-20261011T224347` and
    `dk-and-rpk-migration-08-resumed-c-20261011T230717`.
- **R2's stores** are `var/datastores/dk-and-rpk-migration-08-{reference,resumed}.sqlite`, with
  sidecars carrying `39809dca…`, and are not retired. All six of the campaign's stores stay for the
  user to retire after 09's review (README §3).
- **For 09.**
  - RayPoolKit's G2 record, to draft for its own board (C12), is verification §3.10.
  - DatastoreKit's is §2.10.
  - The final measurement at `HEAD` can be compared with `measure/out/08-raypoolkit-measure_client_pools.md`.
- **C8 is closed as a measurement:** the package's post handler at `main.py:3247` changes nothing
  the statistics record.
- **The venv is unchanged:** 08 installed nothing. `docs/OPEN_ISSUES.md` is at 108.
- **The registry** holds 29 runs: 27 finished, 2 `unknown`, none `running`.
