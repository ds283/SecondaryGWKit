# Log — runregistry-extraction prompt 08, rehearse through both packages (R2)

- **Subject:** [`08-rehearse-through-both-packages.md`](../08-rehearse-through-both-packages.md),
  board items L1–L4 (R2) and G-RRK; gates R2 and RunRegistryKit's G2.
- **Commit:** this commit (parent `b9daf44`).
- **Result:** landed. **R2 holds, and RunRegistryKit's G2 holds.** Every acceptance item passes, and
  no stop condition fired.
- **Model:** Opus 5.5.
- **Date:** 2026-10-16 (dispatched 2026-10-15).

**What R2 shows:**
- Both stores are `39809dca…`, format 6, **`encoding_format` 1**, all 21 classes. They are equal to
  each other, to their sidecars, and to the fingerprints R2's own `reference` and `resumed-c`
  recorded. The swap to `runregistrykit` is invisible to a build: the one log line that names it is
  the launcher's banner, and no line names `datastorekit`.
- **W1 across stores:** field by field, `encoding_format` and `taken` set aside, each R2 store's
  fingerprint equals each of the four fingerprints SGK's own code recorded for R0 and R1.
  `fingerprint_both_ways.py` exits 0, 20 of 20 checks on each store.
- L1–L4 hold against R1. The prune at open is exercised a third time: 44 rows pruned, all absent
  afterwards.
- The lock was taken before `reference`, its `pid` set at each launch, held through the readers and
  the guard, and released; it is free now.
- RunRegistryKit's checklist item 9 holds clause by clause.
- Every suite is at 07's count, with no failure. Nothing was opened or closed.

**Where it ran.** Dispatched at `b9daf44` ("Write the orchestrator note for runregistry-extraction
prompt 08"), whose code is `a9ed928`'s.
- The tree was clean, in the only worktree, at every launch.
- `./venv/bin/python -m runregistrykit list` showed 39 runs (37 finished, 2 `unknown`), none
  `running` or stale.
- The lock was free, `ray-processes` listed nothing, and `RAY_ADDRESS` was unset.
- The venv holds `runregistrykit` 0.1.0 at `9f8ebe43…`, `datastorekit` 0.4.0 at `c8570e84…` and
  `raypoolkit` 0.2.0 at `eaef0ce6…`, none editable.

The record is
[`docs/runregistry-extraction-verification.md`](../../../docs/runregistry-extraction-verification.md)
§3. This log points into it rather than repeat it. The orchestrator's facts and corrections for this
prompt are cited as "the orchestrator's note".

## What shipped

- **`docs/runregistry-extraction-verification.md`**: **§3, "R2 — through both packages"**,
  appended, with §2's subsections §2.1–§2.9 as §3.1–§3.9 in the same order (§3.1.3 the lock),
  **§3.10, W1 across stores**, and **§3.11, RunRegistryKit's G2**. §1, §2 and the header are not
  edited; the header already names §3 as R2's.
- **`docs/runregistry-extraction/rehearsal/`**: 23 saved outputs named `*_08*`:
  - the run records and pre-launch checks: `run_records_08.txt` (with the registry's listing after
    R2, counted by state), `prelaunch_checks_08.txt`, `var_datastores_08.txt` (the 18 files before,
    the 30 after);
  - the lock: `lock_08.txt`;
  - the interrupter: `interrupter_dummy_test_08.txt`, `interrupter_build_dummy_test_08.txt`,
    `interrupter_firings_08.txt`;
  - the stores: `store_sidecars_08.txt`, `closed_store_state_08.txt`, `prune_check_08.txt`;
  - the fingerprints and integrity: `fingerprint_comparison_08.txt`,
    `fingerprint_against_recorded_08.txt`, `census_08.txt`, `store_soundness_08.txt`;
  - the guard and L4: `guard_08.txt`, `stage_counts_08.txt`, and
    `policy_statistics_08_reference.txt`, `_resumed-b.txt` and `_resumed-c.txt`;
  - the readers: `readers_08.txt`;
  - `log_comparison_08.txt`: the line-kind comparison of every build log and reader log with R1's
    same slug, the searches, the printed-line counts and forms, the `** ` sequences, the
    `status.json` fields, the stage times, and the fingerprint after the readers;
  - W1: `fingerprint_both_ways_08.txt`;
  - G2: `g2_measure_08.txt` (RunRegistryKit's measure, whole; then correction 8's grep; then
    `store show` on the 18 tombstones).
- **This log, and the board.** `docs/OPEN_ISSUES.md` is unchanged, since nothing was opened.
  README and `prompts/INDEX.md` are unchanged. No code, test or script changed.
- **Under `var/`, not committed:** the two stores and their sidecars; four run directories and their
  launch logs; `var/runregistry-extraction-08/` (verification §3.9). Neither store is retired.

## The runs

| Slug | Run id | Python pid (`sh` pid) | Interrupter pid | State | Exit code | Driver exit status | Wall time | Load before / after |
|---|---|---|---|---|---|---|---|---|
| `reference` | `runregistry-extraction-08-reference-20261015T235716` | 76434 (76430) | — | `done` | 0 | 0 | 4 m 33 s | 5.95 7.17 8.92 / 18.79 23.61 16.60 |
| `resumed-a` | `runregistry-extraction-08-resumed-a-20261016T000342` | 83931 (83928) | 84032 | `killed` | 143 | 143 | 2 m 2 s | 11.97 21.38 16.07 / 5.94 15.15 14.49 |
| `resumed-b` | `runregistry-extraction-08-resumed-b-20261016T000708` | 87907 (87904) | 88011 | `killed` | 143 | 143 | 1 m 27 s | 7.17 14.57 14.31 / 10.71 14.62 14.45 |
| `resumed-c` | `runregistry-extraction-08-resumed-c-20261016T000959` | 91423 (91420) | — | `done` | 0 | 0 | 1 m 40 s | 8.54 13.71 14.13 / 5.59 11.12 13.04 |

**Fingerprints.** All four are format 6, `encoding_format` 1, problems `{}`, 21 classes:
- `reference` and `resumed-c`:
  `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`;
- `resumed-a`: `87e55e8f…`;
- `resumed-b`: `8978b72e…`.

**Every manifest** has:
- `git_head` `b9daf449…` and `git_dirty` false;
- `packages` equal, as JSON, to the note's record: `runregistrykit` 0.1.0 at `9f8ebe43…`
  (`v0.1.0`), `datastorekit` 0.4.0 at `c8570e84…` (`v0.4.0`) and `raypoolkit` 0.2.0 at `eaef0ce6…`
  (`v0.2.0`), none editable; the last two R1's exactly;
- `script_sha256` `2a985303…`, the same in all four;
- R1's nineteen keys.

After each launch the agent made one check, ran `set-pid` in it, and ended its turn. The
orchestrator's one wait per launch ended on the driver pid's death each time, with the heartbeat
never stale, and resumed the agent with the final `status.json`.

**The interrupter.**
- **The dummy tests** ran before the first launch, 23:56:18–23:56:42, each into a new scratch
  directory. Both exited 0. With times, pids and paths stripped, each output equals R1's line for
  line (40 and 20 lines). No child or interrupter was left alive (16 pids; `ps -p` empty).
- **The points**, from R2's own reference `stdout.log`, sized by the orchestrator's note
  (corrections 2 and 3).
  - **`resumed-a`:** `'^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 11`.
    - The reference's `GkSource` stage took 27.4 s; 40 % of that is 10.96 s, so the hold is 11 s,
      just above the 10 s floor.
    - It became eligible at 00:05:32 and sent SIGTERM at 00:05:43, 11–16 s into the stage. The
      handler's line is stamped 00:05:43.
    - **165 of 291 `GkSource` rows were stored, 44 unvalidated.**
  - **`resumed-b`:** `'^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 40`, not
    anchored at its end.
    - The reference's QSI stage took 119.8 s. A third is 39.9 s, so the hold is 40 s, under the
      250 s cap.
    - It became eligible at 00:07:52 and sent SIGTERM at 00:08:32. The handler's line is stamped
      00:08:33.
    - 297 of 864 `QuadSourceIntegral` rows were stored. No progress line appeared during the hold.
- `resumed-a`'s `stderr.log` ends `SystemExit(1); …` and `resumed-b`'s `SystemExit(143); …`, the
  other way round from R0 and R1 (verification §3.2.2). Each was recorded `killed`/143, and each
  driver exited 143.

**The killed launches' Ray agents.** There were none to wait for. `ray-processes` exited 0 with
nothing listed, `pgrep` was empty, and `ps -p` on the run's pids, the interrupter's and every Ray pid
its logs name found none: at 00:06:28 (~44 s after `resumed-a`'s final heartbeat) and at 00:09:22
(~47 s after `resumed-b`'s). No wait by pid was run, as at R0 and R1.

**The lock** ([`lock_08.txt`](../../../docs/runregistry-extraction/rehearsal/lock_08.txt),
verification §3.1.3). Every command exited 0:
- `acquire --purpose "runregistry-extraction 08 (R2)"` at 23:57:01, before `reference`;
- `set-pid` 76434 (23:57:37), 83931 (00:03:52), 87907 (00:07:16) and 91423 (00:10:03), each in the
  one check after its launch;
- `status` after each launch and after each run ended, the pid dead between launches;
- after the readers and the guard, `status`, `ray-processes`, then `release` at 00:16:31;
- then `status` (`lock: free`) and `ray-processes` (none).

## Deviations from the prompt

1. **04's corrections carried over** (the orchestrator's note, correction 1): `prune_serials.py` was
   not run, and `prune_check.py` ran on R2's own `closed_store_state.py` output; the six scripts of
   `docs/dk-and-rpk-migration/rehearsal/` ran as 06 left them, and the other tools from where R1 ran
   them, none copied; nothing was written in the repository outside `var/` until `resumed-c` had
   ended; the pre-launch Ray check was `ray-processes`, exit 0, with the machine-wide `pgrep` also
   recorded. **IMPLEMENTATION CHOICE**, by the orchestrator's note.
2. **`resumed-a`'s hold is 11 s**: 40 % of R2's own `GkSource` stage, 27.4 s, rounded, just above
   the 10 s floor, which did not bind. It left 165 rows stored and 44 unvalidated, so the prune at
   open is a measured clause again. **IMPLEMENTATION CHOICE**, by the orchestrator's note
   (correction 2, and its addendum after `reference` to keep the floor and record the reasoning).
3. **`resumed-b`'s hold is 40 s**: a third of R2's own QSI stage, 119.8 s, under the 250 s cap.
   **IMPLEMENTATION CHOICE**, by the orchestrator's note (correction 3).
4. **No progress line appeared, and that is not a finding.** No stage of R2 ran past the queue's
   300 s minimum notify interval; R1's `reference` and `resumed-c` printed one each. The forms of
   every other pool line are compared with R1's (verification §3.5.6). **IMPLEMENTATION CHOICE**, by
   the orchestrator's note (correction 4).
5. **`packages` is compared as JSON, not in order.** The prompt's §2 says every `packages` names the
   three "in this order"; every record is written with `sort_keys`, so on disk the keys are
   alphabetical. Each manifest's `packages` was compared with the note's record as parsed
   dictionaries. **STRUCTURALLY REQUIRED**, by the orchestrator's note (correction 5).
6. **The launcher's hash and banner line moved.** Every manifest's `script_sha256` is `2a985303…`,
   not R1's `b7935877…`, and every `stdout.log` and launch log has the banner line
   `**   list it with: ./venv/bin/python -m runregistrykit list` where R1's named `RunRegistry`. The
   log comparison lists it as the one expected line naming `runregistrykit`, citing 06; no other line
   names it. **IMPLEMENTATION CHOICE**, by the orchestrator's note (correction 6).
7. **`fingerprint_both_ways.py`'s count is 20 of 20 per store**, four checks of its own and four per
   recorded fingerprint, with `--expect-digest` at its default. **IMPLEMENTATION CHOICE**, by the
   orchestrator's note (correction 7).
8. **G2's item 1 is RunRegistryKit's measure, re-run** from its root with its own venv at its `HEAD`
   `caad864`, `--commit b9daf449…`, saved whole as `g2_measure_08.txt`, with correction 8's grep.
   Nothing in RunRegistryKit changed (`git status --short` empty before and after).
   **IMPLEMENTATION CHOICE**, by the orchestrator's note (correction 8).
9. **The verification record's §3 carries §3.1–§3.9 in §2's order, §3.1.3 the lock, §3.10 W1 across
   stores and §3.11 G2.** §2.10, R1's W1, became §3.10's cross-store form. **IMPLEMENTATION
   CHOICE**, by the orchestrator's note (correction 9).
10. **The board's header says 9 of 10 landed**; `prompts/INDEX.md` is the orchestrator's and is not
    touched. **IMPLEMENTATION CHOICE**, by the orchestrator's note (correction 10).
11. **The statistics block is compared as counts and as sorted lines.** R2's reference block differs
    from R1's in the order of `complete` and `minimal` under `mixed`; sorted, they are equal, and
    every count is equal. R2's reference block is R0's byte for byte
    (`[02-the-policy-statistics-block-orders-its-lines-by-completion]`). **STRUCTURALLY REQUIRED**,
    as the prompt directs through 04, 02 and `dk-and-rpk-migration` 08.
12. **L4's resumed chain is compared by identity, not by count.** The raw counts depend on where each
    launch was killed. **STRUCTURALLY REQUIRED**, as 04 and 02 direct.
13. **The purpose was passed to `sh -c` as `$0`**, as at R0 and R1. The Python `argv` is R1's form
    exactly. **IMPLEMENTATION CHOICE.**
14. **`resumed-b`'s stored `QuadSourceIntegral` count, 297, is read from its recorded fingerprint**,
    as at R0 and R1; `closed_store_state.py` does not count that class. `resumed-c`'s 567 confirms
    it. **IMPLEMENTATION CHOICE.**
15. **The readers ran as a background shell task of this session**, their output to
    `var/runregistry-extraction-08/readers.txt`, the lock held. While they ran, the agent ran only
    reads that open none of the resumed store's files: the stage counts, the statistics, the
    build-log comparison, and G2's grep and measure (git only). The comparisons on the stores, the
    guard, W1 and the suites ran after the readers. **IMPLEMENTATION CHOICE.**
16. **The log and reader comparisons used a throwaway script** in the agent's scratchpad, not
    committed, with §1.2.5's normalisation, and searched also for `ConfigError`, `RunRegistry`,
    `runregistrykit`, the raylet's `file_system_monitor` line and each `SystemExit` code. Its output
    is `log_comparison_08.txt`. **IMPLEMENTATION CHOICE.**
17. **`closed_store_state.py` was also run after `reference`**, as at R0 and R1. **IMPLEMENTATION
    CHOICE.**
18. **G2's item 3 was also re-measured here**: the 29 runs not of this campaign compared by id and
    state with `measure/out/01-list.txt`, and `store show` run on the 18 tombstones (read-only; every
    file under `var/datastores/` unchanged by it). Both are appended to `g2_measure_08.txt` after the
    measure. **IMPLEMENTATION CHOICE.**
19. **Two annotations of the agent's own were corrected before the records were committed:**
    `lock.txt`'s `status` after `release` had been labelled "(after the readers and the guard)", like
    the one before it, and is labelled "(after the release)"; and `fingerprint_both_ways_08.txt`'s
    exit line had its clock times filled in from the two epoch stamps the shell captured, since
    `date -r` here is GNU `date` and printed an error in their place. No command output was changed.
    **IMPLEMENTATION CHOICE.**
20. **One transient file went to `/tmp`.** The first pre-launch capture wrote the `runregistrykit
    list` output to `/tmp/.rrk_list_<pid>` for two `grep`s and deleted it in the same command, before
    the fix to the agent's scratchpad took effect. Nothing of the record depends on it; every later
    capture used the scratchpad. **UNINTENDED DRIFT.**

## Verification performed

**The acceptance items** (prompt §3):
1. **L1–L3 hold.** Both stores are `39809dca…`, format 6, `encoding_format` 1, equal to each other
   and to their sidecars; `fingerprint_both_ways.py` exits 0 against R0's and R1's four recorded
   fingerprints, every field equal but `encoding_format` (and `taken`). ✅
2. **Every manifest's `packages`** names `runregistrykit`, `datastorekit` and `raypoolkit` at their
   pins, none editable, equal as JSON to the note's record; every `git_dirty` is `false`. ✅
3. **L4 against R1.** The statistics counts, the reference stage counts, the chain's identities,
   the stage banners, the printed lines and the registry fields are R1's. Each difference in the
   logs is recorded and explained below; none opens an issue. ✅
4. **`list`** shows 43 runs, 01's 29 and this campaign's fourteen, nothing `running`. ✅
5. **The board records RunRegistryKit's G2 as holding**, clause by clause, and the R2 row of §2. ✅
6. **The lock** was taken before `reference`, its `pid` set at each launch, held through the
   readers, and released, every command and exit status in `lock_08.txt`. Afterwards `status`
   reports no lock and `ray-processes` lists nothing. ✅
7. **The seven roots pass at 07's counts**, with no failure. ✅
8. **`docs/runregistry-extraction-verification.md` gains §3**, in §2's shape, with §3.10 for W1
   across stores and §3.11 for G2. §1 and §2 are not edited. ✅

**L1–L3** (verification §3.2–§3.4):
- **L1.**
  - Every launch ended as recorded, and each driver's exit status agrees.
  - No Ray process of this repository was left after any launch.
  - Every manifest is as the prompt's §2 requires, with R1's keys. Every `status.json` has R1's
    keys, and its fingerprint carries `encoding_format`.
  - Every `stdout.log` is as §3.2.5's table: the banner line the only one naming `runregistrykit`,
    0 naming `datastorekit`, 0 `DeprecationWarning`, 0 `ConfigError`.
  - The guard refused before Ray started, named the five files, changed nothing and created no run.
- **L2.**
  - **The fingerprints.**
    - Both stores are format 6, `encoding_format` 1, `39809dca…`, all 21 classes equal, each equal
      to its sidecar's.
    - `compare_fingerprints` is empty between them, and against the fingerprints recorded by R2's
      own `reference` and `resumed-c` (`compare_with_recorded.py`, exit 0).
  - **The prune and the recompute.**
    - `resumed-a` left 165 `GkSource` rows, 44 unvalidated, listed by serial.
    - `prune_check.py` after `resumed-b`, and again after `resumed-c`, gives 44 checked, 44 absent,
      0 reused, 0 still unvalidated. After `resumed-b` there is nothing to check.
    - `GkSource` 165 + 170 − 44 = 291; QSI 297 + 567 = 864. No stored row was recomputed, and
      nothing is duplicated.
  - **Integrity.** The version row `(1, '2025.1.1')` is on all 8 shards after every launch;
    `replication_in_flight` is empty and there is no journal after every launch. The census gives
    `no divergence` on both stores, and `store_soundness.py` `=> clean`; each output, and
    `compare_stores.py`'s, equals R1's once the campaign and prompt are normalised.
- **L3.**
  - All six readers exit 0, and so does `extract_Gk_data.py` without `--models`, with the skip
    line.
  - Nothing refused, and every store file is byte-identical. `store fingerprint` afterwards matches
    `resumed-c`'s.
  - The lookups equal R1's queue by queue. File counts differ in the readers that draw random
    subsamples.

**W1 across stores** (verification §3.10). On each store all 20 checks hold: its `encoding_format`
is `ENCODING_FORMAT`, 1; its digest `39809dca…`; the sidecar records a fingerprint and compares
empty; and against each of the four records, the record lacks `encoding_format`, its digest is
`39809dca…`, field by field with `encoding_format` and `taken` set aside there are 0 differing
paths, and the package's `compare_fingerprints(recorded, current)` is one `encoding` entry (recorded
`None`, current 1). Exit 0, 5 s. Every file of the two stores and their sidecars (12) had the same
SHA-256, size and `mtime_ns` before and after.

**L4 against R1** (verification §3.5):

| | R1 | R2 |
|---|---|---|
| Statistics: total | 291 | 291 |
| numeric | 109, all complete | 109, all complete |
| mixed | 26: 25 complete, 1 minimal | 26: 25 complete, 1 minimal |
| WKB | 156, all complete | 156, all complete |
| `reference`'s block, line for line | `minimal` first | `complete` first; equal sorted; equal to R0's `reference` byte for byte |
| `resumed-b`'s block | the same counts | equal to R1's `resumed-b` byte for byte |
| `resumed-c`'s block | the same counts | equal to R1's `resumed-c` byte for byte |

The reference's stage counts equal R1's in every stage (verification §3.3.1).

| Resumed-chain identity | R1 | R2 |
|---|---|---|
| Finished stages compute 0 on the next launch | yes | yes |
| Pruned at `resumed-b`'s open = left unvalidated by `resumed-a` | 48 = 48, all absent, none reused | **44 = 44, all absent, none reused** |
| `resumed-b` `GkSource` = 291 − stored + pruned | 132 = 291 − 207 + 48 | 170 = 291 − 165 + 44 |
| `resumed-c` QSI = 864 − stored by `resumed-b`, nothing else | 440 = 864 − 424 | 567 = 864 − 297 |
| Totals over the chain | 291, 864 | 291, 864 |

**The stage sequences.** Each run's `** ` lines (26, 23, 26, 26) equal R1's same slug in order, once
the campaign and prompt, the path and the numbers are normalised, but for the banner's
`list it with:` line. The reference's 15 stage banners are equal verbatim.

**The printed lines and the registry** (verification §3.5.6):
- In all four runs, the completion lines (10, 7, 9, 10), the `Queue summary` lines and the
  `(no compute items …)` lines (1, 1, 7, 9) equal R1's in number. Normalised, their forms equal
  R1's.
- R2 printed no progress line; R1's `reference` and `resumed-c` printed one each.
- Each run's `status.json` has R1's keys, its `stage` is R1's, and `units_done`/`units_total` are
  0/`null`, as at R1. `runregistrykit` read the pool's lines as `RunRegistry` did.

**The stage times** (not compared):

| Stage | R2 reference | R1 reference | R0 reference |
|---|---|---|---|
| `GkSource` | 27.4 s | 1 m 42.9 s | 1 m 18 s |
| QSI | 1 m 59.8 s | 8 m 14.9 s | 20 m 30.1 s |
| The whole build | 4 m 33 s | 13 m 57 s | 29 m 19 s |

**The log differences from R1, slug by slug** (verification §3.2.5). With times, pids, paths and
numbers stripped:
- **Every slug:** the banner's `list it with:` line names `runregistrykit` (`stdout.log` and launch
  log), and the raylet's disk-space line is new in `stderr.log` and the launch log (26, 11, 8, 9
  lines).
- **`reference`:** also one `Queue summary` in `/sec` that R1 printed in `/min`, and no progress
  line or `inflight` line (R1 printed one of each).
- **`resumed-a`:** nothing else. `stderr.log` ends `SystemExit(1)` where R1's ended `SystemExit(143)`.
- **`resumed-b`:** nothing else in line kinds; `stdout.log` is 12 lines shorter because shard 3's
  integrity-warning block lists 9 pruned rows against R1's 15. `stderr.log` ends `SystemExit(143)`
  where R1's ended `SystemExit(1)`.
- **`resumed-c`:** no progress line or `inflight` line (R1 printed one of each). The
  `IntegrationWarning` is here and in `reference`, as at R1.
- **Nothing else.** There is no traceback and no new warning in any build log but the machine's
  disk-space line, and no line naming `datastorekit`, `DeprecationWarning`, `ConfigError` or
  `_timestamp`.

**The reader logs against R1's.** Three are identical in line kinds. The other four differ by the
disk-space line, by `extract_GkWKB_data`'s queue finishing in 44.3 s (no progress line, no
completion time in `Nm Ns`), and by how many of Matplotlib's repeated warnings Ray folded.

**`var/` outside the two stores.** The 18 files under `var/datastores/` have the same SHA-256, mtime
and size before `reference` (23:56:07) as after the last step. The digest over them is
`1fbaf5b7…`, the orchestrator's. The only new files are R2's ten store files and two sidecars.
Nothing under `var/` is newer than the first listing but R2's own runs, stores and working directory.

**The registry after R2.** `./venv/bin/python -m runregistrykit list`: 43 runs, 41 finished and 2
`unknown`; by state 23 `done`, 15 `killed`, 3 `failed`, 2 `unknown`; nothing `running` or stale.
01's 29 by id and state equal `01-list.txt`; this campaign's fourteen are R0's, R1's and R2's twelve
launches and 06's two worktree runs.

**G2, clause by clause** (verification §3.11):
1. No `RunRegistry` outside `prompts/` and dated `docs/` prose: the measure at `b9daf44`, 422 files,
   0 imports, names, bound calls, string literals and `-m RunRegistry`, in live code and the staying
   tests; the grep prints `CLAUDE.md:82` and `config/fingerprint.py:33` alone; `import RunRegistry`
   fails. ✅
2. The roots pass, RunDrivers 99 and ObjectFactories 279 among them (06, 07, here). ✅
3. `list` gives 01's 29 with their states and purposes (06 §7.1; ids and states re-checked here), and
   `store show` reads all 18 sidecars (exit 0, `registry`, problems none, `retired`). ✅
4. The worktree run is recorded under the main checkout's `var/runs/` (06 §7.4). ✅
5. Every R2 fingerprint carries `encoding_format` beside the digest; 06's `TestConfigFingerprint`
   test 2 shows the digest is `fingerprint_of`'s on the test store, and passes here. ✅
6. The pin at `requirements.txt:81`, installed from it, not editable; every R2 manifest's `packages`
   names the three. ✅

**The suites.** Run serially, in the foreground, after the last launch, the readers, the guard and
the release, with README §8's command.

| Root | Dispatch baseline (07) | Now | Load (1 min) |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 7.9 |
| CosmologyModels | 39 OK | 39 OK | 7.9 |
| ObjectFactories | 279 OK | 279 OK | 7.4 |
| tools | 91 OK | 91 OK | 6.9 |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 6.3 |
| RunDrivers | 99 OK | 99 OK | 5.6 |
| ComputeTargets | 693 OK | 693 OK | 14.3 |

No failure, so no known flake needed a re-run. No `.py` file was touched, so `black` was not run.

## The deliberate-breakage record

None. This prompt adds no code, no test and no script (README §5 rule 8). The breakages of
`fingerprint_both_ways.py` are 06's, on the same code.

## Observations not acted on

1. **Ray's disk-space warning.** Every raylet of R2's builds and readers warned every 10 s that the
   volume holding `/tmp/ray` "is over 95% full … Object creation will fail if spilling is required"
   (54 lines over the four builds' `stderr.log`s, 8 over the readers). The volume is 96 % used:
   44 GB free before `reference`, 40 GB after, against 72 GB at R1 and 38 GB at the orchestrator's
   check. Ray's threshold is 95 %. R2 spilled no object and both stores are `39809dca…`, so it
   changed nothing a build wrote or read; it is the machine's, not the swap's. A longer build that
   needed to spill, such as A3 v2, could fail on it. Not opened, at the orchestrator's ruling.
2. **The killed handler's exit, swapped against R1.** `resumed-a` ended `SystemExit(1)` and
   `resumed-b` `SystemExit(143)`, the reverse of R0's and R1's order. §1.2.2's explanation holds:
   Ray's signal check converts the handler's `SystemExit(143)` when the signal lands inside
   `ray.wait`, and where it lands is the point's. The record is `killed`/143 and the exit 143 either
   way. Not a finding, at the orchestrator's ruling.
3. **The 40 % rule came close to its floor.** At this load the reference's `GkSource` stage took
   27.4 s, so 40 % was 10.96 s against a 10 s floor. It still left 44 of 165 stored rows
   unvalidated. Under a lighter load, or a faster machine, the floor would bind, and a 10 s hold
   with a 5 s read interval could signal before any row is stored (as at `dk-and-rpk-migration`'s
   R2, which then held the prune clause only vacuously).
4. **The statistics block's order moved back.** R2's reference prints `complete` first, R0's order,
   byte for byte R0's block; R1's printed `minimal` first. `resumed-b` and `resumed-c` match R1's
   same slug. This is `[02-the-policy-statistics-block-orders-its-lines-by-completion]`, already
   open, with one more instance and no new fact.
5. **`[07-a-killed-builds-ray-agents-outlive-the-driver]` did not recur** after either killed launch
   (checks ~44 s and ~47 s after the final heartbeats, at one-minute loads of 6 and 11). The issue
   stays open, and is not this campaign's.
6. **G2's measure parses 422 files**, against 06's 421: 07 added
   `ObjectFactories/tests/test_fingerprint_format_guard.py`. Every figure is still 0.
7. **No other project's Ray cluster appeared during R2.** The machine-wide `pgrep` was empty before
   every launch and after every run. The lock was free at `acquire`, and `~/ScienceData/` held
   `ChamPBH-rehearsals/` alone before and after.

## State handed to the next prompt

- **R2 holds** (board §2), resting on:
  - the dispatch `HEAD` `b9daf44`, whose code is `a9ed928`'s;
  - this commit;
  - the runs `runregistry-extraction-08-reference-20261015T235716` and
    `runregistry-extraction-08-resumed-c-20261016T000959`, both `done`, both `39809dca…` with
    `encoding_format` 1, equal field by field to R0's and R1's recorded fingerprints.
- **RunRegistryKit's G2 holds** (board §2), resting on 06 (`740bafc`), 07 (`a9ed928`) and this
  commit. 09 drafts the record for RunRegistryKit's board, with the five registry issues for its
  `docs/OPEN_ISSUES.md` (board §3).
- **R2's two stores** are `var/datastores/runregistry-extraction-08-{reference,resumed}.sqlite`, each
  with a sidecar carrying the package's fingerprint with `encoding_format` 1, and neither retired.
  The user retires them after 09's review, or not (README §3).
- **The lock is free** and no Ray process of this repository is running.
- **The venv is unchanged:** 08 installed nothing. `docs/OPEN_ISSUES.md` stays at 106.
- **The registry** holds 43 runs: 41 finished, 2 `unknown`, none `running`.
- **The roots:** AdaptiveLevin 32, CosmologyModels 39, ObjectFactories 279, tools 91,
  LiouvilleGreen 148 (1 skipped), RunDrivers 99, ComputeTargets 693.
