# Log — runregistry-extraction prompt 02, rehearse the start tree (R0)

- **Subject:** [`02-rehearse-the-start-tree.md`](../02-rehearse-the-start-tree.md), board items
  L1–L4 (R0); gate R0.
- **Commit:** this commit (parent `fe16d8e`).
- **Result:** landed. **R0 holds.** Every acceptance item passes, and no stop condition fired.
- **Model:** Opus 5.5.
- **Date:** 2026-10-12.

**What R0 shows:**
- Both stores are `39809dca…`, all 21 classes. They are equal to each other and to the
  fingerprints recorded by `dk-and-rpk-migration`'s R2 `reference` and `resumed-c`.
- The factory move, `511ef5f`, is invisible to a build: no build or reader raised on an import or an
  unpickling, and the fingerprint did not move.
- L1–L4 hold against R2. Unlike R2's, this resumed chain exercises the prune at open: 63 rows
  pruned, all absent afterwards.
- Every suite is at 01's count. Nothing was opened or closed.

**Where it ran.** Dispatched at `fe16d8e` ("Write the orchestrator note for runregistry-extraction
prompt 02"), whose code is `a45aefc`'s.
- The tree was clean, in the only worktree.
- `RunRegistry list` showed 29 runs (27 finished, 2 `unknown`), none `running`.
- There was no Ray process of this repository, and `RAY_ADDRESS` was unset.
- The venv holds `datastorekit` 0.2.2 at `6d8ee9ae…` and `raypoolkit` 0.2.0 at `eaef0ce6…`, neither
  editable, and no `runregistrykit`.
- Every launch ran at `fe16d8e` with `git status --short` empty.

The record is
[`docs/runregistry-extraction-verification.md`](../../../docs/runregistry-extraction-verification.md)
§1. This log points into it rather than repeat it.

## What shipped

- **`docs/runregistry-extraction-verification.md`**, new.
  - A header names the campaign, the purpose and how the record grows: §1 R0 here, §2 R1 at 04, §3
    R2 at 08, §4 the close at 09.
  - **§1, "R0 — the start tree"**, has `dk-and-rpk-migration-verification.md` §3's subsections §3.1–§3.9
    in the same order, every command verbatim.
  - §1.10, "The factory move, seen by a build", is this R0's own. It is the orchestrator's R7 (the
    note's correction 8).
- **`docs/runregistry-extraction/rehearsal/`**, new: twenty saved outputs, each named `*_02*`:
  - the run records and pre-launch checks: `run_records_02.txt`, `prelaunch_checks_02.txt`, and
    `var_datastores_02.txt` (the 14 sidecars before and after);
  - the interrupter: `interrupter_dummy_test_02.txt`, `interrupter_build_dummy_test_02.txt`,
    `interrupter_firings_02.txt`;
  - the stores: `store_sidecars_02.txt`, `closed_store_state_02.txt`, `prune_check_02.txt`;
  - the fingerprints and integrity: `fingerprint_comparison_02.txt`,
    `fingerprint_against_recorded_02.txt`, `census_02.txt`, `store_soundness_02.txt`;
  - the guard and L4: `guard_02.txt`, `stage_counts_02.txt`, and `policy_statistics_02_reference.txt`,
    `_resumed-b.txt` and `_resumed-c.txt`;
  - the readers: `readers_02.txt`;
  - `log_comparison_02.txt`. It holds the line-kind comparison of every build log with R2's same
    slug, the searches of the build logs, the printed-line counts and forms, the `** ` sequence, the
    `status.json` fields, the stage times and the reader logs' comparison.

  No script anywhere changed, and no new script was added.
- **This log, and the board.** `docs/OPEN_ISSUES.md` is unchanged, since nothing was opened.
  README and `prompts/INDEX.md` are unchanged.
- **Under `var/`, not committed:**
  - the two stores and their sidecars;
  - four run directories and their launch logs;
  - `var/runregistry-extraction-02/` (verification §1.9).

## The runs

| Slug | Run id | Python pid (`sh` pid) | Interrupter pid | State | Exit code | Driver exit status | Wall time | Load before / after |
|---|---|---|---|---|---|---|---|---|
| `reference` | `runregistry-extraction-02-reference-20261012T091921` | 69779 (69774) | — | `done` | 0 | 0 | 29 m 19 s | 69.76 54.73 65.04 / 157.91 339.39 312.04 |
| `resumed-a` | `runregistry-extraction-02-resumed-a-20261012T095041` | 90851 (90847) | 90922 | `killed` | 143 | 143 | 12 m 14 s | 127.20 309.05 302.19 / 73.20 182.33 241.64 |
| `resumed-b` | `runregistry-extraction-02-resumed-b-20261012T100353` | 99176 (99172) | 99219 | `killed` | 143 | 143 | 6 m 37 s | 57.90 169.94 235.46 / 103.45 109.75 175.56 |
| `resumed-c` | `runregistry-extraction-02-resumed-c-20261012T101145` | 4791 (4615) | — | `done` | 0 | 0 | 5 m 31 s | 90.21 106.44 172.86 / 92.01 107.30 146.96 |

**Fingerprints.** All four are format 6, problems `{}`, 21 classes:
- `reference` and `resumed-c`:
  `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`;
- `resumed-a`: `9bb753bd…`;
- `resumed-b`: `a0d053af…`.

**Every manifest** has:
- `git_head` `fe16d8e7…` and `git_dirty` false;
- `packages` equal, as JSON, to R2's record, both packages at their tags' commits, neither
  editable;
- `script_sha256` `b7935877…`, the same in all four, and R2's.

After each launch the agent made one check and ended its turn. The orchestrator's one wait per
launch ended `ENDED finished` each time, with the heartbeat never stale.

**The machine was heavily loaded**, by processes not of this repository. At the checks the
one-minute load ran from 58 to 158, and the five-minute load reached 339 at 09:49. R2's checks ran
at 13–33.

**The interrupter.**
- **The dummy tests** ran before the first launch, 09:18:33–09:18:59, each into a new scratch
  directory. Both exited 0. With times, pids and paths stripped, each output equals R2's line for
  line. No child or interrupter was left alive (16 pids, every `pid N` and `child N`; `ps -p`
  empty).
- **The points**, from R0's own reference `stdout.log`, sized by the orchestrator's note
  (corrections 4 and 5).
  - **`resumed-a`:** `'^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 31`.
    - The reference's `GkSource` stage took 78 s; 40 % of that is 31.2 s, so the hold is 31 s.
    - It became eligible at 10:02:20 and sent SIGTERM at 10:02:51, 31–36 s into the stage. The
      handler's line is stamped 10:02:51, a delay under 1 s.
    - **164 of 291 `GkSource` rows were stored, 63 unvalidated.**
  - **`resumed-b`:** `'^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 250`, not
    anchored at its end.
    - The reference's QSI stage took 1230.1 s. A third is 410 s, so the hold takes the 250 s cap,
      inside the queue's 300 s minimum notify interval.
    - It became eligible at 10:06:07 and sent SIGTERM at 10:10:17. The handler's line is stamped
      10:10:18.
    - 486 of 864 `QuadSourceIntegral` rows were stored. No progress line appeared during the hold.
- `resumed-a`'s `stderr.log` ends `SystemExit(143); recording killed, exit code 143` and
  `resumed-b`'s `SystemExit(1); …`. Each was recorded `killed`/143, and each driver exited 143.

**The killed launches' Ray agents.** There were none to wait for.
- The checks were the whole-table search for `SecondaryGWKit/venv` with `ray`, `pgrep -lf
  'gcs_server|raylet|ray::'` with other repositories' venvs excluded, and `ps -p` on the run's pids,
  the interrupter's and every Ray pid its logs name.
- All were empty at 10:03:18 (~23 s after `resumed-a`'s final heartbeat) and at 10:10:54 (~24 s
  after `resumed-b`'s).
- No wait by pid was run. The agents' exit time is "none listed at the first check", as at
  `dk-and-rpk-migration`'s R0, R1 and R2.

## Deviations from the prompt

1. **The gate's "no commit after 01's touches code (anything outside `prompts/`)".** `cf978c6`, 01's
   review, changed two lines of `CLAUDE.md`, adding `a45aefc` to the freeze section. So the gate
   cannot hold literally. It holds in substance:
   - `git diff --stat a45aefc fe16d8e -- . ':!prompts'` lists `CLAUDE.md` alone;
   - no build, test or script reads `CLAUDE.md`;
   - so the code that runs at the dispatch `HEAD` is `a45aefc`'s.

   **STRUCTURALLY REQUIRED**, by the orchestrator's note (§0, correction 1).
2. **`prune_serials.py` was not run.** It hardcodes `a3-v2-readiness` 07's serials, and comes to
   this prompt through 08, 05 and 02. `prune_check.py` ran in its place, on R0's own
   `closed_store_state.py` output, as at `dk-and-rpk-migration`'s three rehearsals.
   **IMPLEMENTATION CHOICE**, by the orchestrator's note (§0, correction 2).
3. **`store_soundness.py` is `dk-and-rpk-migration`'s**, under `docs/dk-and-rpk-migration/rehearsal/`.
   `a3-v2-readiness`'s copy raises `TypeError`
   (`[02-07s-store-soundness-calls-read-inventory-without-its-registry]`) and was not run. The other
   tools ran from where R2 ran them, none copied. **IMPLEMENTATION CHOICE**, by the orchestrator's
   note (§0, correction 3).
4. **`resumed-a`'s hold is 31 s, not R2's 10 s.** The prompt says the points are chosen "by the same
   reasoning". R2's 10 s stored no `GkSource` row, so its prune at open held only vacuously. The
   hold was sized at 40 % of R0's own `GkSource` stage time, 78 s, so that `resumed-a` would leave
   stored and unvalidated rows for `resumed-b` to prune. It did: 164 stored, 63 unvalidated.
   **IMPLEMENTATION CHOICE**, by the orchestrator's note (§0, correction 4).
5. **`resumed-b`'s hold is 250 s, not a third of the stage.** A third of R0's 1230.1 s QSI stage is
   410 s, which would pass the queue's 300 s minimum notify interval. The cap keeps the hold inside
   it. **IMPLEMENTATION CHOICE**, by the orchestrator's note (§0, correction 5).
6. **The builds printed progress lines, and the comparison of printed lines covers them.** R2's
   builds printed none. R0's reference printed three in its QSI stage and `resumed-a` one in its
   GkWKB stage, because each ran past 300 s under this load. Their form is compared with the copy's
   last build line (`a3-v2-readiness-07-reference`'s `stdout.log:100-101`) and R2's GkWKB reader's
   (verification §1.5.6). **STRUCTURALLY REQUIRED**, by the orchestrator's note (§0, correction 6).
7. **The saved outputs waited for the last launch.** `docs/runregistry-extraction/` did not exist,
   and creating it would have stamped a manifest `git_dirty: true`. Every capture was written under
   `var/runregistry-extraction-02/` and copied into `docs/runregistry-extraction/rehearsal/` after
   `resumed-c` had ended. **STRUCTURALLY REQUIRED**, by the orchestrator's note (§0, correction 7).
8. **The verification record's §1 carries §3.1–§3.9 of `dk-and-rpk-migration`'s §3, and adds §1.10.**
   That campaign's §3.10 (G-RPK) and §3.11 (`main.py:3247`) record its RayPoolKit gate and C8, which
   have no counterpart here. §1.10 records the factory move as a build sees it (R7).
   **STRUCTURALLY REQUIRED**, by the orchestrator's note (§0, correction 8).
9. **The statistics block is compared as counts and as sorted lines.** R0's reference and
   `resumed-b` blocks differ from R2's in the order of `complete` and `minimal` under `mixed`.
   Sorted, each pair is equal, and every count is equal. This follows
   `[02-the-policy-statistics-block-orders-its-lines-by-completion]`. **STRUCTURALLY REQUIRED**, as
   the prompt directs (§2, L4).
10. **L4's resumed chain is compared by identity, not by count.** The raw counts depend on where
    each launch was killed. R0 compares 05's four identities instead (verification §1.5.3).
    **STRUCTURALLY REQUIRED**, as 05 directs through 08.
11. **The purpose was passed to `sh -c` as `$0`.** The launch command is R2's but for that: its
    `--purpose "<one line>"` became `--purpose "$0"`, with the line given after the command string.
    This keeps the purpose's parentheses and semicolons out of the quoted command. The Python `argv`
    in each manifest is R2's form exactly. **IMPLEMENTATION CHOICE.**
12. **`resumed-b`'s stored `QuadSourceIntegral` count, 486, is read from its recorded fingerprint.**
    R2 counted its 281 by shard from the closed store. `closed_store_state.py` does not count that
    class. The fingerprint `finish` took on the closed store records the per-class counts, read-only,
    before `resumed-c` opened it. `resumed-c`'s 378 computed rows confirm it: 864 − 486 = 378.
    **IMPLEMENTATION CHOICE.**
13. **The Ray-process check excluded other repositories' venvs by path.** Before `resumed-c`,
    `pgrep -lf 'gcs_server|raylet|ray::'` listed a StochasticInstantons cluster (4084, 4173, 4178),
    started by another session at 10:11:04. The check before the next step lists the same table with
    paths under `StochasticInstantons/venv` excluded, beside the unchanged whole-table search for
    `SecondaryGWKit/venv`. The cluster was not touched. The build starts its own instance with
    `address="local"`. **IMPLEMENTATION CHOICE**, confirmed by the orchestrator.
14. **The readers ran under the Monitor tool**, inside the session, with their output to
    `var/runregistry-extraction-02/readers.txt` and only the `===`/`exit` lines streamed. While they
    ran, the agent ran only reads that open no store: the stage counts, the statistics and the log
    comparison. The guard and the suites ran after the readers, serially, in the foreground.
    **IMPLEMENTATION CHOICE.**
15. **The log and reader comparisons used throwaway scripts**, kept in the agent's scratchpad and
    not committed, with R2's §3.2.5 normalisation. Their output is `log_comparison_02.txt`.
    **IMPLEMENTATION CHOICE.**
16. **`closed_store_state.py` was also run after `reference`**, as R2 did, though the protocol needs
    it only after the killed launches. **IMPLEMENTATION CHOICE.**

No `UNINTENDED DRIFT`.

## Verification performed

**The acceptance items** (prompt §3):
1. **L1–L3 hold.** Both stores equal each other, `39809dca…`, format 6, 21 classes, and R2's
   recorded fingerprints (`dk-and-rpk-migration-08-reference` and `-resumed-c`). ✅
2. **Every manifest's `packages` is R2's record**, compared as JSON, and every `git_dirty` is
   `false`. ✅
3. **L4 against R2.** The statistics counts are equal, the reference stage counts are equal, the
   chain's identities hold, the stage sequence is equal, and the printed lines and registry fields
   are R2's. Each difference in the logs is recorded and explained below; none opens an issue. ✅
4. **The seven roots pass at 01's counts**, with the known wall-clock flake. ✅
5. **`docs/runregistry-extraction-verification.md` exists**, with §1 in the shape of
   `dk-and-rpk-migration-verification.md` §3. ✅
6. **The board records R0 as holding**, naming `runregistry-extraction-02-reference-20261012T091921`
   and `runregistry-extraction-02-resumed-c-20261012T101145`. ✅
7. **No deliberate breakage.** 02 adds no code and no test. ✅

**L1–L3** (verification §1.2–§1.4):
- **L1.**
  - Every launch ended as recorded, and each driver's exit status agrees.
  - No Ray process of this repository was left after any launch.
  - Every manifest is as the prompt's §2 requires: `git_head` `fe16d8e7…`, `script_sha256`
    `b7935877…`. Every `stdout.log` is as R2's §3.2.5 table.
  - The guard refused before Ray started, named the five files, changed nothing and created no run.
- **L2.**
  - **The fingerprints.**
    - Both stores are format 6, `39809dca…`, all 21 classes equal, each equal to its sidecar's.
    - `compare_fingerprints` is empty between them, and against the fingerprints recorded by
      `dk-and-rpk-migration-08-reference-20261011T224347` and
      `dk-and-rpk-migration-08-resumed-c-20261011T230717`.
  - **The prune and the recompute.**
    - `resumed-a` left 164 `GkSource` rows, 63 unvalidated, listed by serial.
    - `prune_check.py` after `resumed-b`, and again after `resumed-c`, gives 63 checked, 63 absent,
      0 reused, 0 still unvalidated. After `resumed-b` there is nothing to check.
    - `GkSource` 164 + 190 − 63 = 291; QSI 486 + 378 = 864. No stored row was recomputed, and
      nothing is duplicated.
  - **Integrity.**
    - The version row `(1, '2025.1.1')` is on all 8 shards after every launch.
    - `replication_in_flight` is empty, and there is no journal, after every launch.
    - The census gives `no divergence` on both stores, and equals R2's output once the campaign and
      prompt are normalised.
    - On all 10 files, `integrity_check` is `ok`, `foreign_key_check` is empty, and the package's
      `read_inventory` finds no problem. The output equals R2's `store_soundness_08.txt` once the
      campaign and prompt are normalised.
- **L3.**
  - All six readers exit 0, and so does `extract_Gk_data.py` without `--models`, with the skip
    line.
  - Nothing refused, and every store file is byte-identical. `store fingerprint` afterwards matches
    `resumed-c`'s.
  - The lookups equal R2's queue by queue. File counts differ in the three readers that draw random
    subsamples.

**The factory move, seen by a build** (verification §1.10):
- In a fresh `python -I`, all 38 registered factories come from `ObjectFactories`, and
  `import Datastore.SQL.ObjectFactories` raises `ModuleNotFoundError`.
- In the twelve build logs and seven reader logs there are 0 lines of `Traceback`,
  `ModuleNotFoundError`, `ImportError`, `AttributeError`, `pickle` or `unpickl`, and none naming
  `Datastore.SQL`.
- The fingerprint did not move.

**L4 against R2** (verification §1.5):

| | R2 | R0 |
|---|---|---|
| Statistics: total | 291 | 291 |
| numeric | 109, all complete | 109, all complete |
| mixed | 26: 25 complete, 1 minimal | 26: 25 complete, 1 minimal |
| WKB | 156, all complete | 156, all complete |
| Block, line for line | — | differs in the order of two lines under `mixed`; equal sorted; equal to `dk-and-rpk-migration` R1's line for line |
| `resumed-b`'s block | the same counts | the same counts; order differs from R2's `resumed-b`; equal sorted; equal to R0's reference |
| `resumed-c`'s block | the same counts | equal to R2's `resumed-c` line for line |

| Stage | R2 `reference` | R0 `reference` |
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

| Resumed-chain identity | R2 | R0 |
|---|---|---|
| Finished stages compute 0 on the next launch | yes | yes |
| Pruned at `resumed-b`'s open = left unvalidated by `resumed-a` | 0 = 0 (vacuous) | **63 = 63, all absent, none reused** |
| `resumed-b` `GkSource` = 291 − stored + pruned | 291 = 291 − 0 + 0 | 190 = 291 − 164 + 63 |
| `resumed-c` QSI = 864 − stored by `resumed-b`, nothing else | 583 = 864 − 281 | 378 = 864 − 486 |
| Totals over the chain | 291, 864 | 291, 864 |

**The stage sequence.** The reference's 26 `** ` lines equal R2's in order, once the campaign and
prompt, the path and the numbers are normalised. Its 15 stage banners are equal verbatim.

**The printed lines and the registry** (verification §1.5.6):
- In all four runs, the completion lines (10, 7, 9, 10), the `Queue summary` lines and the
  `(no compute items …)` lines (1, 1, 7, 9) equal R2's in number. Normalised, their forms equal
  R2's.
- R0's `reference` printed three progress lines and `resumed-a` one; R2's builds printed none. Each
  has the copy's form, and the GkWKB reader's two have R2's.
- `resumed-a`'s interrupter read one progress line out of `status.json`, in `StageTracker`'s form.
- Each run's `status.json` has R2's keys, its `stage` is R2's, and `units_done`/`units_total` are
  0/`null`, as at R2.

**The stage times** (not compared):

| Stage | R0 reference | R2 reference | R1 reference | `dk-and-rpk-migration` R0 reference |
|---|---|---|---|---|
| `GkSource` | 1 m 18 s | 41.8 s | 55.1 s | 32.2 s |
| QSI | 20 m 30.1 s | 4 m 59.2 s | 4 m 49.7 s | 3 m 32.3 s |
| The whole build | 29 m 19 s | 11 m 25 s | 12 m 13 s | 8 m 0 s |

**The log differences from R2, slug by slug** (verification §1.2.5). With times, pids, paths and
numbers stripped:
- **`reference`:** `stderr.log` identical. `stdout.log` and the launch log add the three progress
  lines' kinds (correction 6, the load).
- **`resumed-a`:** `stderr.log` identical but for `SystemExit(143)`, which normalises to the same
  kind. `stdout.log` adds one progress line and its `inflight` line. One `Queue summary` prints its
  rate per minute where R2's printed it per second, the same line at a lower rate.
- **`resumed-b`:**
  - It gains the SciPy `IntegrationWarning`'s four lines, which R2 had in `resumed-c`. R0's
    `resumed-b` stored 486 QSI rows before its point, against R2's 281, so the one integral fell
    before R0's point, as at R1.
  - It gains the integrity warning's 12 kinds, 50 `stdout.log` lines: the prune at open of the 63
    unvalidated rows.
  - The `GkSource` completion time prints as `1m 10s`, against R2's `30.4s`.
- **`resumed-c`:** loses the warning's four lines; `stdout.log` is identical.
- **Nothing else.** There is no traceback and no new warning in any build log. There is no
  `ValueError`, `ray.kill`, killed actor, `ActorDiedError` or `RayActorError`.

**The reader logs against R2's.** Five are identical in line kinds. `extract_GkWKB_data`'s lacks
R2's two random matplotlib `UserWarning`s. `extract_GkSource_data`'s has R2's
`MatplotlibDeprecationWarning` once more in a Ray-folded form.

**`var/` outside the two stores.** The 14 sidecars have the same SHA-256, mtime and size before
`reference` (09:18:23) as after the last step (10:38). The digest over them is `023d139f…`, the
orchestrator's.

**The suites.** Run serially, in the foreground, after the last launch, the readers and the guard,
with README §8's command.

| Root | Dispatch baseline | Now | Load (1 min) |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 67.9 |
| CosmologyModels | 39 OK | 39 OK | 63.5 |
| ObjectFactories | 271 OK | 271 OK | 63.5 |
| tools | 49 OK | 49 OK | 51.1 |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 39.3 |
| RunRegistry | 286 OK | 286 OK | 29.0 |
| ComputeTargets | 693, 1 wall-clock flake | 693, the same flake (0.0658 s against 0.06 s) | 24.9 |

`test_tk_wkb_phase.TestCost.test_wall_time_per_object` was re-run alone three times at load 17. It
passed, passed and failed (0.0709 s): the known flake of README §8. No Python file was added, so
`black --check` had nothing to check.

## The deliberate-breakage record

None (README §5 rule 8). This prompt adds no code and no test, and names no mutation. The check is
the fingerprint comparison itself, which `dk-and-rpk-migration`'s 02 review showed can fail. The
scripts' failure and non-trivial paths were exercised on real inputs:
- `policy_statistics.py` exits 1 on `resumed-a`, which has no block.
- `prune_check.py` reports 63 serials absent after `resumed-a`'s state, and "nothing to check" on the
  state after `resumed-b`.
- The guard refuses an existing store.
- The interrupter's dummy tests refuse a wrong pid and an empty `argv[0]`.

## Observations not acted on

1. **The machine was heavily loaded by processes not of this repository.**
   - The five-minute load reached 339 during `reference`'s QSI stage, and the one-minute load ran
     from 58 to 158 at the checks. R2's ran at 13–33.
   - The reference took 29 m 19 s, against R2's 11 m 25 s. Its QSI stage took 1230.1 s, against
     299.2 s.
   - The load slowed the stages, and changed nothing they wrote.
   - `var/`'s volume is 93–94 % full (65 GB free).
2. **These are the first registered builds to print a pool progress line under `raypoolkit`.**
   R2's observation 4 recorded that none had, and that `StageTracker`'s reading of one was shown only
   by a test.
   - Here the reference's QSI stage printed three, and `resumed-a`'s GkWKB stage one.
   - `resumed-a`'s interrupter read `CALCULATE WKB PART OF TENSOR GREEN FUNCTIONS, 66.67% (8 of 24
     work items remaining)` from `status.json` two seconds after the line was printed. So a real run
     now shows the registry reading the pool's progress line.
   - Nothing to open: it is what the code is meant to do.
3. **The prune at open is exercised again**, after R2 held it only vacuously.
   - The orchestrator's 40 % hold put `resumed-a`'s signal 31–36 s into a 78 s stage. It left 164
     rows, 63 unvalidated, and `resumed-b` pruned exactly those.
   - **A sizing rule from the reference's own stage time works under a different load.** R2's fixed
     10 s missed the stored rows at a load of about 30. Here the stage ran at about half R2's speed,
     and the rule still placed the signal inside it, with rows stored.
4. **The statistics block's order moved again.**
   - R0's reference and `resumed-b` print `complete` first, R1's order.
   - R2's reference and `resumed-b` printed `minimal` first.
   - `resumed-c` prints `minimal` first, as at R1 and R2.

   This is `[02-the-policy-statistics-block-orders-its-lines-by-completion]`, already open, with
   one more instance and no new fact. Nothing was opened.
5. **`[07-a-killed-builds-ray-agents-outlive-the-driver]` did not recur** after either killed launch.
   The checks ran about 23–24 s after each final heartbeat, at one-minute loads of 73 and 103. The
   issue's record has agents outliving the driver by 1–2 min at loads of 65–123. So load alone does
   not explain when it recurs. The issue stays open, and is not this campaign's.
6. **Another repository's Ray cluster appeared between launches.** StochasticInstantons' cluster
   (4084, 4173, 4178) started at 10:11:04, nine seconds before `resumed-c`'s pre-launch check, and
   was gone by the check after it.
   - The build cannot join it (`address="local"`).
   - The check excluded it by venv path, as the orchestrator confirmed, and never touched it.
   - `pgrep 'gcs_server|raylet|ray::'` alone cannot tell this repository's Ray from another's; only
     the venv path can. `CLAUDE.md`'s liveness rule already forbids pattern matching for waiting;
     this is the pre-launch listing, where the path is the distinguishing fact.
   - A saved copy of that listing in `prelaunch_checks_02.txt` truncates the three lines: one
     carried the other process's environment.
7. **`SystemExit(143)` in `resumed-a`**, where R2 printed `SystemExit(1)`. Where the signal lands
   decides it (verification §1.2.2). The record is `killed`/143 either way.
8. **The intermediate fingerprints differ from R2's** (`9bb753bd…` and `a0d053af…` against
   `ff6c4257…` and `aa860ed8…`), as they must: each digests a store killed at a different point.

## State handed to the next prompt

- **R0 holds** (board §2), resting on:
  - the dispatch `HEAD` `fe16d8e`, whose code is `a45aefc`'s;
  - this commit;
  - the runs `runregistry-extraction-02-reference-20261012T091921` and
    `runregistry-extraction-02-resumed-c-20261012T101145`, both `done`, both `39809dca…`, equal to
    R2's recorded fingerprints.
- **The start tree is sound**, and the factory move is invisible to a build. R1 (04) can isolate the
  DatastoreKit pin: a miss there is DatastoreKit's (README §4.1).
- **R0's stores** are `var/datastores/runregistry-extraction-02-{reference,resumed}.sqlite`, with
  sidecars carrying `39809dca…`, and are not retired (README §3). 04 compares with R0 through the
  fingerprints recorded in the two run directories above, and through this record's §1.
- **For 04's points.** R0's reference stage times are the slowest on record: `GkSource` 78 s and QSI
  1230.1 s, at a load of up to 339. 04 sizes its points from its own reference, as here, with the
  orchestrator's 40 % and 250 s rules. Under a lighter load, a QSI stage near 300 s will put the
  progress-line question back where R2 left it.
- **The venv is unchanged:** 02 installed nothing. `docs/OPEN_ISSUES.md` stays at 105.
- **The registry** holds 33 runs: 31 finished, 2 `unknown`, none `running`.
