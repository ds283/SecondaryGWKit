# Log — prompt 05, rehearse through `datastorekit` (R1)

**Prompt:** [`../05-rehearse-through-datastorekit.md`](../05-rehearse-through-datastorekit.md) ·
**Board items:** L1–L4 (R1), G-DK · **Date:** 2026-10-11 · **Model:** Opus 5.5 ·
**Commit:** this commit · **Result:** landed.

**R1 holds:**
- Both stores are `39809dca…`, all 21 classes. They are equal to each other and to the
  fingerprints recorded by R0's reference, 07's reference and 09's rehearsal.
- L1–L4 hold against R0.
- **DatastoreKit's G2 holds.**
- Every suite is at its dispatch count. Nothing was opened or closed.

**Where it ran.** Dispatched at `6861a9f` ("Write the orchestrator note for dk-and-rpk-migration
prompt 05"), whose code is `0c9e2cf`'s.
- The tree was clean, in the only worktree.
- `RunRegistry list` showed 21 runs (19 finished, 2 `unknown`), none `running`.
- There was no Ray process at all, and `RAY_ADDRESS` was unset.
- The venv holds `datastorekit` 0.2.2 at `6d8ee9ae…`, not editable, and no `raypoolkit`.
- Every launch ran at `6861a9f` with `git status --short` empty.

The record is
[`docs/dk-and-rpk-migration-verification.md`](../../../docs/dk-and-rpk-migration-verification.md)
§2. This log points into it rather than repeat it.

## What shipped

- **`docs/dk-and-rpk-migration-verification.md`**: a new **§2, "R1 — through `datastorekit`"**.
  - It has §1's subsections in the same order, every command verbatim.
  - It adds §2.10, G-DK part by part.
  - §1 is not edited.
- **`docs/dk-and-rpk-migration/rehearsal/`**: seventeen saved outputs, each named `*_05*`:
  - the run records and pre-launch checks: `run_records_05.txt`, `prelaunch_checks_05.txt`;
  - the interrupter: `interrupter_dummy_test_05.txt`, `interrupter_build_dummy_test_05.txt`,
    `interrupter_firings_05.txt`;
  - the stores: `store_sidecars_05.txt`, `closed_store_state_05.txt`, `prune_check_05.txt`;
  - the fingerprints and integrity: `fingerprint_comparison_05.txt`,
    `fingerprint_against_recorded_05.txt`, `census_05.txt`, `store_soundness_05.txt`;
  - the guard and L4: `guard_05.txt`, `stage_counts_05.txt`, `policy_statistics_05_reference.txt`;
  - the readers: `readers_05.txt`;
  - one with no R0 counterpart, `log_comparison_05.txt`. It holds the line-kind comparison of every
    log with R0's same slug, the `** ` sequences, and the stage times.

  No script there changed, and no new script was added.
- **This log, and the board.** `docs/OPEN_ISSUES.md` is unchanged, since nothing was opened.
  README is unchanged.
- **Under `var/`, not committed:**
  - the two stores and their sidecars;
  - four run directories and their launch logs;
  - `var/dk-and-rpk-migration-05/` (verification §2.9).

## The runs

| Slug | Run id | Python pid (`sh` pid) | Interrupter pid | State | Exit code | Driver exit status | Wall time | Load before / after |
|---|---|---|---|---|---|---|---|---|
| `reference` | `dk-and-rpk-migration-05-reference-20261011T192802` | 83632 (83628) | — | `done` | 0 | 0 | 12 m 13 s | 13.84 29.00 32.74 / 38.56 42.53 43.83 |
| `resumed-a` | `dk-and-rpk-migration-05-resumed-a-20261011T194149` | 92759 (92751) | 92862 | `killed` | 143 | 143 | 4 m 23 s | 23.81 38.25 42.20 / 23.93 46.87 47.11 |
| `resumed-b` | `dk-and-rpk-migration-05-resumed-b-20261011T194703` | 96222 (96219) | 96313 | `killed` | 143 | 143 | 2 m 51 s | 21.18 45.52 46.62 / 35.93 44.64 45.99 |
| `resumed-c` | `dk-and-rpk-migration-05-resumed-c-20261011T195044` | 98713 (98710) | — | `done` | 0 | 0 | 4 m 13 s | 31.43 43.39 45.53 / 13.79 30.05 38.99 |

**Fingerprints.** All four are format 6, problems `{}`, 21 classes:
- `reference` and `resumed-c`:
  `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`;
- `resumed-a`: `b6b89d84…`;
- `resumed-b`: `09170e40…`.

**Every manifest** has:
- `git_head` `6861a9fc…` and `git_dirty` false;
- `packages` exactly the prompt's §2 record;
- `script_sha256` `b7935877…`, the same in all four.

After each launch the agent made one check and ended its turn. The orchestrator's one wait per
launch ended `ENDED finished` each time, with the heartbeat never stale. At `reference`'s launch the
load was 77.69 (shared machine). Over the four builds it ran from 14 to 78, against R0's 6 to 36.

**The interrupter.**
- **The dummy tests** ran before the first launch, 19:27:20–19:27:45, each into a new scratch
  directory. Both exited 0, with every verdict as R0's record: five cases, then three `--build`
  cases. No child was left alive.
- **The points**, from R1's own reference `stdout.log`. Neither stage printed a progress line, and
  both queues have a 300 s minimum.
  - **`resumed-a`:** `'^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 10`,
    R0's arguments.
    - The reference's `GkSource` stage took 55.1 s, so the signal falls 10–15 s in, well before
      the end even at R0's 32.2 s.
    - It became eligible at 19:46:00 and sent SIGTERM at 19:46:10. The handler's line is stamped
      19:46:10, a delay under 1 s.
    - 101 of 291 `GkSource` rows were stored, 45 of them unvalidated.
  - **`resumed-b`:** `'^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 95`, not
    anchored at the end.
    - The reference's QSI stage took 289.7 s, so 95 s is about a third of it, and before the end
      even at R0's 212.3 s.
    - It became eligible at 19:48:10 and sent SIGTERM at 19:49:45. The handler's line is stamped
      19:49:45, a delay under 1 s.
    - 334 of 864 `QuadSourceIntegral` rows were stored. No progress line appeared during the hold.
- Both `stderr.log`s end `SystemExit(1); recording killed, exit code 143`. Each was recorded
  `killed`/143, and each driver exited 143.

**The killed launches' Ray agents.** There were none to wait for.
- The checks were `pgrep -lf 'gcs_server|raylet|ray::'`, the whole-table search for
  `SecondaryGWKit/venv` with `ray`, the run's pids and the driver's process group.
- All were empty at 19:46:45 (~33 s after `resumed-a`'s final heartbeat) and at 19:50:24 (~30 s
  after `resumed-b`'s).
- No wait by pid was run, and the agents' exit time is "none listed at the first check", as at R0.
- `v0.2.2`'s `_kill_actors()` left no line in any log.

## Deviations from the prompt

1. **`prune_serials.py` was not run.** The prompt's §1.3 says it runs from where R0 ran it. It
   hardcodes 07's serials, so R0 did not run it either. `prune_check.py` ran in its place, reading
   the serials from R1's own `closed_store_state.py` output. **IMPLEMENTATION CHOICE**, by the
   orchestrator's note for 05 (§0, correction 1).
2. **The manifests' `script_sha256` is `b7935877cdbc4e54…`, not R0's `d745783e65bfff99…`.** The
   prompt's §2 is silent on it. 04 rewrote `docs/handover/quadsource_atol_sweep.py:693` (`from
   Datastore.shard_paths` → `from datastorekit.shard_paths`), its only change since R0. All four
   R1 manifests hold `b7935877…`, which was also the file's hash at the pre-launch check.
   **IMPLEMENTATION CHOICE**, by the orchestrator's note (§0, correction 2).
3. **L4's resumed chain is compared by identity, not by count.** README §2.2 asks that "what the
   resume recomputed" be R0's. The raw counts cannot be, since they depend on where each launch
   was killed. R1 compares the four identities of the prompt's §2 instead, and all hold
   (verification §2.5.3). **STRUCTURALLY REQUIRED**, as the prompt directs.
4. **The statistics block is compared as counts and as sorted lines.** R1's reference block
   differs from R0's line for line in the order of `complete` and `minimal` under `mixed`. Sorted,
   the two are equal, and every count is equal. This follows
   `[02-the-policy-statistics-block-orders-its-lines-by-completion]`. **STRUCTURALLY REQUIRED**, as
   the prompt directs.
5. **`resumed-b`'s hold is 95 s, not R0's 70.** R1's QSI stage took 289.7 s against R0's 212.3 s,
   so a third of it is longer. The prompt says to size the point from R1's own reference.
   **IMPLEMENTATION CHOICE.**
6. **The log comparison used a throwaway script**, kept in the agent's scratchpad and not
   committed.
   - It normalises ANSI codes, the prompt number, the repository path, Ray's `[repeated Nx]` count
     and every number, then compares the sets of line kinds. Its output is in
     `log_comparison_05.txt`, and the method in verification §2.2.5.
   - No script was added beside the rehearsal scripts. Since numbers are normalised, the one value
     difference it cannot see, `SystemExit(1)` against R0's `SystemExit(143)` in `resumed-a`, was
     read directly and is listed separately.

   **IMPLEMENTATION CHOICE.**
7. **Three saved records are truncated or reshaped.**
   - The pre-launch capture before `resumed-c` cut its lines at 160 characters. The capture command
     piped through `cut`, so the `RunRegistry list` rows lose the end of their purpose. The facts
     the check records (HEAD, status, `RAY_ADDRESS`, `pgrep`, the search, the pids, the run count)
     are whole.
   - `prelaunch_checks_05.txt` also carries the environment, the layer check and R0's hashes. R0's
     carried only the per-launch checks.

   **UNINTENDED DRIFT** in the first, of the agent's tooling, not of the measurement. **IMPLEMENTATION
   CHOICE** in the second.
8. **`prune_check.py` for `after-resumed-a` ran twice.** It ran at 19:50:54, once `resumed-c` had
   opened the store, as the orchestrator asked, and again after `resumed-c`, with the same result.
   **IMPLEMENTATION CHOICE.**

## Verification performed

**L1–L3** (verification §2.2–§2.4):
- **L1.**
  - Every launch ended as recorded, and each driver's exit status agrees.
  - No Ray process of this repository was left after any launch.
  - Every manifest is as the prompt's §2 requires, and every `stdout.log` is as §1.2.5's table.
  - The guard refused before Ray started, named the five files, changed nothing and created no run.
- **L2.**
  - **The fingerprints.**
    - Both stores are format 6, `39809dca…`, all 21 classes equal, each equal to its sidecar's.
    - `compare_fingerprints` is empty between them, and against the recorded fingerprints of
      `dk-and-rpk-migration-02-reference` (R0), `a3-v2-readiness-07-reference` and
      `datastore-generic-09-rehearsal`.
  - **The prune and the recompute.**
    - The 45 unvalidated `GkSource` serials are all absent after `resumed-b`'s open, none reused.
      15 + 30 = 45 in the folded warnings.
    - `GkSource` 101 + 235 − 45 = 291; QSI 334 + 530 = 864. No stored row was recomputed, and
      nothing is duplicated.
  - **Integrity.**
    - The version row `(1, '2025.1.1')` is on all 8 shards.
    - The census gives `no divergence` on both stores, and equals R0's output once the prompt
      number is normalised.
    - On all 10 files, `integrity_check` is `ok`, `foreign_key_check` is empty, and the package's
      `read_inventory` finds no problem. The output equals R0's `store_soundness_02.txt` once the
      prompt number is normalised.
- **L3.**
  - All six readers exit 0, and so does `extract_Gk_data.py` without `--models`, with the skip line.
  - Nothing refused, and every store file is byte-identical. `store fingerprint` afterwards matches
    `resumed-c`'s.
  - Lookups equal R0's reader by reader. File counts differ: the readers draw random subsamples
    (`random.sample`), as 02's review found.

**What runs is the package** (verification §2.1.1):
- Each of 04's 15 deleted layer modules gives `ModuleNotFoundError` in a fresh `python -I` (15 of
  15).
- Both `__init__.py` are 0 bytes.
- `datastorekit.__file__`, `datastorekit.SQL.ShardedPool.__file__` and
  `datastorekit.store_inventory.__file__` are in the venv's `site-packages`.

**L4 against R0** (verification §2.5):

| | R0 | R1 |
|---|---|---|
| Statistics: total | 291 | 291 |
| numeric | 109, all complete | 109, all complete |
| mixed | 26: 25 complete, 1 minimal | 26: 25 complete, 1 minimal |
| WKB | 156, all complete | 156, all complete |
| Block, line for line | — | differs in the order of two lines under `mixed`; equal sorted; equal to 09's line for line |

| Stage | R0 `reference` | R1 `reference` |
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

| Resumed-chain identity | R0 | R1 |
|---|---|---|
| Finished stages compute 0 on the next launch | yes | yes |
| Pruned at `resumed-b`'s open = left unvalidated by `resumed-a` | 43 = 43 | 45 = 45 |
| `resumed-b` `GkSource` = 291 − stored + pruned | 215 = 291 − 119 + 43 | 235 = 291 − 101 + 45 |
| `resumed-c` QSI = 864 − stored by `resumed-b`, nothing else | 532 = 864 − 332 | 530 = 864 − 334 |
| Totals over the chain | 291, 864 | 291, 864 |

**The stage sequence.** The reference's 26 `** ` lines equal R0's in order, once the prompt number
and the run stamp are normalised. 15 of them are the stage banners.

**The stage times** (for 08; not compared):

| Stage | R1 reference | R0 reference |
|---|---|---|
| `GkSource` | 55.1 s | 32.2 s |
| QSI | 4 m 49.7 s | 3 m 32.3 s |
| The whole build | 12 m 13 s | 8 m 0 s |

R1 ran at loads of 14–78. Its QSI stage came within 10 s of the 300 s at which a progress line may
first print.

**The log differences from R0, slug by slug** (verification §2.2.5). With times, pids, paths and
numbers stripped:
- **`reference` and `resumed-a`:** identical in all three logs, line count and kinds.
- **`resumed-b`:** gains the SciPy `IntegrationWarning`'s four lines, which R0 had in `resumed-c`
  and R1's `resumed-c` lacks.
  - The warning belongs to one integral. R1's `resumed-b` ran about 95 s of the QSI stage, against
    R0's 70 s, and stored 334 rows against 332. That integral completed before R1's point and
    after R0's.
- **`resumed-b`'s integrity warning** shows 15 entries in full on shard 0, against R0's 11. Ray
  folds the rest as `[repeated 30x]`, against R0's `32x`, since R1 pruned 45 rows, not 43. That
  accounts for 8 more `stdout.log` lines.
- **`resumed-a`'s handler** prints `SystemExit(1)`, where R0's printed `SystemExit(143)`. This is
  07's recorded case: the signal lands in `ray.wait`. The record is `killed`/143 either way.
- **Nothing else.** There is no traceback, no new warning, and no line from the package's close
  (`ray.kill`, killed actor, `ActorDiedError`, `RayActorError`).

**G-DK, part by part** (verification §2.10):

| Part | Rests on | Holds |
|---|---|---|
| 1. The 17 files are gone, two emptied | `0c9e2cf` (04, reviewed); re-checked at `6861a9f` (15 of 15 `ModuleNotFoundError`, both `__init__.py` 0 bytes) | ✅ |
| 2. The pin is in place and installed | `9ec603a` (03, reviewed); re-checked at `6861a9f`; all four R1 manifests record `datastorekit` 0.2.2 at `6d8ee9ae…`, not editable | ✅ |
| 3. The six roots pass | this prompt's suites at `6861a9f` (code of `0c9e2cf`), below | ✅ |
| 4. R1 reproduces `39809dca…` | `dk-and-rpk-migration-05-reference-20261011T192802` and `dk-and-rpk-migration-05-resumed-c-20261011T195044`, both `done`, both `39809dca…`, equal to R0's, 07's and 09's | ✅ |

**R0 untouched.** Its twelve files have the same SHA-256, mtime and size before the first launch
(19:26) as after the last step (20:16, and again just before the commit). They equal the
orchestrator's dispatch table. There is no journal file.

**The suites.** Run serially, in the foreground, after the last launch and the guard, with README
§7's command.

| Root | Dispatch baseline | Now | Load (1 min) |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 74.5 |
| CosmologyModels | 39 OK | 39 OK | 74.5 |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 72.6 |
| RunRegistry | 286 OK | 286 OK | 40.5 |
| ComputeTargets | 693, 1 wall-clock flake | **693 OK** | 13.4 |
| Datastore | 320 = 54 + 103 + 101 + 62 OK | 320 = 54 + 103 + 101 + 62 OK | 7–8 |

`test_tk_wkb_phase.TestCost.test_wall_time_per_object` failed at dispatch and passed here, so it
was not re-run alone. No Python file was added, so `black --check` had nothing to check.

## The deliberate-breakage record

None (README §5 rule 8). This prompt adds no code and no test, and names no mutation. The check is
the fingerprint comparison itself, which R0 showed can fail: 02's review compared against recorded
fingerprints from other trees. The scripts' failure paths were exercised on real inputs:
- `policy_statistics.py` exits 1 on `resumed-a`, which has no block.
- `prune_check.py` reports "nothing to check" on the state after `resumed-b`.
- The guard refuses an existing store.

## Observations not acted on

1. **The statistics block's order moved again.** R1's reference prints 09's order (`complete`
   first); R0's printed the other. R1's `resumed-c`, which computed no policy rows, prints R0's
   order. This is `[02-the-policy-statistics-block-orders-its-lines-by-completion]`, already open,
   with no new fact but one more instance. Nothing was opened.
2. **The intermediate fingerprints differ from R0's** (`b6b89d84…` and `09170e40…` against
   `eaf590af…` and `fc02616e…`). This is expected: each digests a store killed at a different
   point. Only the finished stores are compared.
3. **`v0.2.2`'s close (`_kill_actors`) leaves no visible trace.** It left no log line and no
   orphan, and none was left at R0 either. So R1 cannot say whether it shortens
   `[07-a-killed-builds-ray-agents-outlive-the-driver]`'s orphans. That issue is
   `a3-v2-readiness`'s, and stays open.
4. **The machine was heavily loaded by processes not of this repository.** During the readers the
   load reached 130–157, with PyCharm and a `unittest discover -s datastorek…` process at the top.
   It slowed the builds and the readers, and changed nothing they wrote.
5. **R1's reference QSI stage came within 10 s of the queue's 300 s minimum notify interval.** At a
   higher load, R2's reference may print a QSI progress line. That is harmless to the un-anchored
   pattern, but 08 should read its own reference for it.
6. **The readers' file counts vary run to run** (`random.sample`), so they cannot be compared
   between rehearsals. Their lookups can, and are equal.

## State handed to the next prompt

- **R1 holds, and DatastoreKit's G2 holds.**
  - R1's stores are `var/datastores/dk-and-rpk-migration-05-{reference,resumed}.sqlite`, with
    sidecars carrying `39809dca…`, and are not retired.
  - R2 (08) compares against `var/runs/dk-and-rpk-migration-05-reference-20261011T192802` through
    `compare_with_recorded.py --against`, and against R1's L4 tables in verification §2.5.
- **For 08's points.** R1's reference stage times are in verification §2.5.5: `GkSource` 55.1 s and
  QSI 289.7 s, at loads of 14–78. Neither printed a progress line.
- **Recording G2 on DatastoreKit's own board** is a commit in that repository (README §6.2 C12),
  drafted at 09.
- The venv is unchanged: 05 installed nothing. `docs/OPEN_ISSUES.md` is at 108. The registry holds
  25 runs, none `running`.
