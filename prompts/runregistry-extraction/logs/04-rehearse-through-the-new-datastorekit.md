# Log — runregistry-extraction prompt 04, rehearse through the new DatastoreKit (R1)

- **Subject:** [`04-rehearse-through-the-new-datastorekit.md`](../04-rehearse-through-the-new-datastorekit.md),
  board items L1–L4 (R1) and W1; gate R1.
- **Commit:** this commit (parent `89ff8b7`).
- **Result:** landed. **R1 holds, and W1 holds.** Every acceptance item passes, and no stop
  condition fired.
- **Model:** Opus 5.5.
- **Date:** 2026-10-14.

**What R1 shows:**
- Both stores are `39809dca…`, all 21 classes. They are equal to each other and to the fingerprints
  recorded by R0's `reference` and `resumed-c`. The pin `v0.2.2` → `v0.4.0` is invisible to a
  build: no log line names `datastorekit`, and none holds a `DeprecationWarning` or `_timestamp`.
- **The fingerprint both ways (W1):** DatastoreKit's `fingerprint_of`, given SGK's inventory and
  format 6, equals SGK's in every field but `encoding_format`, on R1's two stores and R0's two.
- L1–L4 hold against R0. The prune at open is exercised again: 48 rows pruned, all absent
  afterwards.
- The first rehearsal under the shared lock: taken, `pid` set at each launch, held through the
  readers, released; free now.
- Every suite is at 03's count, with README §8's known wall-clock tests. Nothing was opened or
  closed.

**Where it ran.** Dispatched at `89ff8b7` ("Write the orchestrator note for runregistry-extraction
prompt 04"), whose code is `5b81f2e`'s.
- The tree was clean, in the only worktree, at every launch.
- `RunRegistry list` showed 33 runs (31 finished, 2 `unknown`), none `running`.
- The lock was free, `ray-processes` listed nothing, and `RAY_ADDRESS` was unset.
- The venv holds `datastorekit` 0.4.0 at `c8570e84…` and `raypoolkit` 0.2.0 at `eaef0ce6…`,
  neither editable, and no `runregistrykit`.

The record is
[`docs/runregistry-extraction-verification.md`](../../../docs/runregistry-extraction-verification.md)
§2. This log points into it rather than repeat it. The orchestrator's facts and corrections for this
prompt are cited as "the orchestrator's note".

## What shipped

- **`docs/runregistry-extraction/fingerprint_both_ways.py`**, new: W1's verification script
  (verification §2.10). It imports `RunRegistry` and `datastorekit.store_fingerprint`, writes
  nothing, and is `black`-clean.
- **`docs/runregistry-extraction-verification.md`**: **§2, "R1 — through DatastoreKit `v0.4.0`"**,
  appended, with §1's subsections §1.1–§1.9 as §2.1–§2.9 in the same order, a §2.1.3 for the lock,
  and **§2.10, W1**. §1 and the header are not edited; the header already names §2 as R1's.
- **`docs/runregistry-extraction/rehearsal/`**: 22 saved outputs named `*_04*`:
  - the run records and pre-launch checks: `run_records_04.txt`, `prelaunch_checks_04.txt`,
    `var_datastores_04.txt` (the 26 files before, the 38 after);
  - the lock: `lock_04.txt`;
  - the interrupter: `interrupter_dummy_test_04.txt`, `interrupter_build_dummy_test_04.txt`,
    `interrupter_firings_04.txt`;
  - the stores: `store_sidecars_04.txt`, `closed_store_state_04.txt`, `prune_check_04.txt`;
  - the fingerprints and integrity: `fingerprint_comparison_04.txt`,
    `fingerprint_against_recorded_04.txt`, `census_04.txt`, `store_soundness_04.txt`;
  - the guard and L4: `guard_04.txt`, `stage_counts_04.txt`, and
    `policy_statistics_04_reference.txt`, `_resumed-b.txt` and `_resumed-c.txt`;
  - the readers: `readers_04.txt`;
  - `log_comparison_04.txt`: the line-kind comparison of every build log and reader log with R0's
    same slug, the searches, the printed-line counts and forms, the `** ` sequences, the
    `status.json` fields, the stage times, and the fingerprint after the readers;
  - W1: `fingerprint_both_ways_04.txt`.
- **This log, and the board.** `docs/OPEN_ISSUES.md` is unchanged, since nothing was opened.
  README and `prompts/INDEX.md` are unchanged.
- **Under `var/`, not committed:** the two stores and their sidecars; four run directories and their
  launch logs; `var/runregistry-extraction-04/` (verification §2.9).

## The runs

| Slug | Run id | Python pid (`sh` pid) | Interrupter pid | State | Exit code | Driver exit status | Wall time | Load before / after |
|---|---|---|---|---|---|---|---|---|
| `reference` | `runregistry-extraction-04-reference-20261014T154651` | 3243 (3239) | — | `done` | 0 | 0 | 13 m 57 s | 11.37 31.55 34.93 / 100.13 94.44 62.59 |
| `resumed-a` | `runregistry-extraction-04-resumed-a-20261014T160226` | 11561 (11558) | 11805 | `killed` | 143 | 143 | 8 m 49 s | 59.78 84.55 60.76 / 20.53 61.19 67.51 |
| `resumed-b` | `runregistry-extraction-04-resumed-b-20261014T161222` | 17571 (17565) | 17691 | `killed` | 143 | 143 | 4 m 16 s | 16.46 56.99 65.82 / 56.62 50.74 59.30 |
| `resumed-c` | `runregistry-extraction-04-resumed-c-20261014T161739` | 20688 (20685) | — | `done` | 0 | 0 | 7 m 12 s | 40.63 47.46 57.87 / 38.73 59.13 59.23 |

**Fingerprints.** All four are format 6, problems `{}`, 21 classes:
- `reference` and `resumed-c`:
  `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`;
- `resumed-a`: `a410cce0…`;
- `resumed-b`: `8146dab8…`.

**Every manifest** has:
- `git_head` `89ff8b72…` and `git_dirty` false;
- `packages` equal, as JSON, to the note's record: `datastorekit` 0.4.0 at `c8570e84…`,
  `requested_revision` `v0.4.0`, and `raypoolkit` 0.2.0 at `eaef0ce6…`, R0's, neither editable;
- `script_sha256` `b7935877…`, the same in all four, and R0's.

After each launch the agent made one check, ran `set-pid` in it, and ended its turn. The
orchestrator's one wait per launch ended `ENDED finished` each time, with the heartbeat never stale.

**The interrupter.**
- **The dummy tests** ran before the first launch, 15:45:47–15:46:12, each into a new scratch
  directory. Both exited 0. With times, pids and paths stripped, each output equals R0's line for
  line. No child or interrupter was left alive (16 pids; `ps -p` empty).
- **The points**, from R1's own reference `stdout.log`, sized by the orchestrator's note
  (corrections 2 and 3).
  - **`resumed-a`:** `'^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 41`.
    - The reference's `GkSource` stage took 102.9 s; 40 % of that is 41.2 s, so the hold is 41 s.
    - It became eligible at 16:10:28 and sent SIGTERM at 16:11:09, 41–46 s into the stage. The
      handler's line is stamped 16:11:09.
    - **207 of 291 `GkSource` rows were stored, 48 unvalidated.**
  - **`resumed-b`:** `'^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 165`, not
    anchored at its end.
    - The reference's QSI stage took 494.9 s. A third is 165 s, under the 250 s cap.
    - It became eligible at 16:13:37 and sent SIGTERM at 16:16:22. The handler's line is stamped
      16:16:23.
    - 424 of 864 `QuadSourceIntegral` rows were stored. No progress line appeared during the hold.
- `resumed-a`'s `stderr.log` ends `SystemExit(143); …` and `resumed-b`'s `SystemExit(1); …`, as at
  R0. Each was recorded `killed`/143, and each driver exited 143.

**The killed launches' Ray agents.** There were none to wait for. `ray-processes` exited 0 with
nothing listed, `pgrep` was empty, and `ps -p` on the run's pids, the interrupter's and every Ray pid
its logs name found none: at 16:11:46 (~31 s after `resumed-a`'s final heartbeat) and at 16:17:05
(~27 s after `resumed-b`'s). No wait by pid was run, as at R0.

**The lock** ([`lock_04.txt`](../../../docs/runregistry-extraction/rehearsal/lock_04.txt),
verification §2.1.3). Every command exited 0:
- `acquire --purpose "runregistry-extraction 04 (R1)"` at 15:46:42, before `reference`;
- `set-pid` 3243 (15:47:01), 11561 (16:02:37), 17571 (16:12:32) and 20688 (16:17:48), each in the
  one check after its launch;
- `status` after each launch and after each run ended, the pid dead between launches;
- after the readers and the guard, `status`, `ray-processes`, then `release` at 16:33:42;
- then `status` (`lock: free`) and `ray-processes` (none).

## Deviations from the prompt

1. **02's corrections carried over** (the orchestrator's note, correction 1): `prune_serials.py` was
   not run, and `prune_check.py` ran on R1's own `closed_store_state.py` output;
   `store_soundness.py` is `dk-and-rpk-migration`'s; the other tools ran from where R0 ran them,
   none copied. **IMPLEMENTATION CHOICE**, by the orchestrator's note.
2. **`resumed-a`'s hold is 41 s**: 40 % of R1's own `GkSource` stage, 102.9 s, rounded, above the
   10 s floor. It left 207 rows stored and 48 unvalidated, so the prune at open is a measured clause
   again. **IMPLEMENTATION CHOICE**, by the orchestrator's note (correction 2).
3. **`resumed-b`'s hold is 165 s**: a third of R1's own QSI stage, 494.9 s, under the 250 s cap.
   **IMPLEMENTATION CHOICE**, by the orchestrator's note (correction 3).
4. **The progress lines are compared by form, wherever they fell.** R1's `reference` and `resumed-c`
   printed one each in their QSI stages; R0's `reference` printed three and `resumed-a` one. Whether
   a stage prints one is the 300 s threshold and the load. Each is compared with R0's first-line
   form (verification §2.5.6). **IMPLEMENTATION CHOICE**, by the orchestrator's note (correction 4).
5. **The W1 script checks order explicitly.** The prompt's §2 asks for "every class, in the same
   order", which Python's `dict` equality does not compare. The script also checks
   `list(pkg_without_encoding) == list(sgk)` and `list(pkg["classes"]) == list(sgk["classes"])`,
   printing the first differing position. `tag_sets` are lists, so `==` already compares their
   order. No sixth breakage was added for it; breakage 1 also trips the key-order check, at
   position 1. **STRUCTURALLY REQUIRED**, by the orchestrator's note (correction 5).
6. **DatastoreKit was read through `git -C … show v0.4.0:…`**, and its installed
   `store_fingerprint.py` read in the venv. Nothing in DatastoreKit was run, changed or checked out.
   **IMPLEMENTATION CHOICE**, by the orchestrator's note (correction 6).
7. **The verification record's §2 carries §2.1–§2.9 in §1's order, adds §2.1.3 for the lock, and
   §2.10 is W1.** §1.10, the factory move, was R0's own. **STRUCTURALLY REQUIRED**, by the
   orchestrator's note (correction 7).
8. **The pre-launch Ray check is `ray-processes`**, exit 0, nothing listed. The machine-wide `pgrep`
   was also recorded, and was empty before every launch. **IMPLEMENTATION CHOICE**, by the
   orchestrator's note (correction 8).
9. **The statistics block is compared as counts and as sorted lines.** R1's reference block differs
   from R0's in the order of `complete` and `minimal` under `mixed`; sorted, they are equal, and every
   count is equal (`[02-the-policy-statistics-block-orders-its-lines-by-completion]`).
   **STRUCTURALLY REQUIRED**, as the prompt directs through 02 and 08.
10. **L4's resumed chain is compared by identity, not by count.** The raw counts depend on where each
    launch was killed. **STRUCTURALLY REQUIRED**, as 02 and 05 direct.
11. **The purpose was passed to `sh -c` as `$0`**, as at R0. The Python `argv` is R0's form exactly.
    **IMPLEMENTATION CHOICE.**
12. **`resumed-b`'s stored `QuadSourceIntegral` count, 424, is read from its recorded fingerprint**,
    as at R0; `closed_store_state.py` does not count that class. `resumed-c`'s 440 confirms it.
    **IMPLEMENTATION CHOICE.**
13. **The readers ran as a background shell task of this session**, their output to
    `var/runregistry-extraction-04/readers.txt`, the lock held. R0 ran them under the Monitor tool.
    While they ran, the agent ran only reads that open none of the resumed store's files: the stage
    counts, the statistics, the build-log comparison and the run records; then a trial of the W1
    draft on R0's reference store and the five breakages on R1's reference store, each
    hashed before and after and unchanged. The comparisons on the resumed store, the guard and the
    suites ran after the readers. **IMPLEMENTATION CHOICE.**
14. **The W1 script was drafted in the scratchpad while the readers ran**, and copied into
    `docs/runregistry-extraction/` after they had ended, identical (`cmp`). The prompt has it written
    after the readers; nothing in the repository was created before then. **IMPLEMENTATION
    CHOICE.**
15. **The W1 script checks a little more than §2 lists:** `fingerprint_format` 6 and `taken` `None`
    on each side explicitly, the listings' lengths, and that the sidecar records a fingerprint at
    all. Each is implied by §2's checks or needed before one. **IMPLEMENTATION CHOICE.**
16. **The log and reader comparisons used a throwaway script** in the agent's scratchpad, not
    committed, with §1.2.5's normalisation. Its output is `log_comparison_04.txt`.
    **IMPLEMENTATION CHOICE.**
17. **`closed_store_state.py` was also run after `reference`**, as at R0. **IMPLEMENTATION CHOICE.**
18. **The first pre-launch capture's `pgrep` exit status was `cut`'s**, since the command was piped.
    It was re-run at once with its own status recorded (rc 1, nothing listed), and every later
    capture records `pgrep`'s own status. `pgrep` is not the test. **IMPLEMENTATION CHOICE.**

No `UNINTENDED DRIFT`.

## Verification performed

**The acceptance items** (prompt §4):
1. **L1–L3 hold.** Both stores equal each other, `39809dca…`, format 6, 21 classes, and R0's
   recorded fingerprints (`runregistry-extraction-02-reference-20261012T091921` and
   `-resumed-c-20261012T101145`). Every manifest's `packages` names `v0.4.0` at `c8570e84…`, and
   every `git_dirty` is `false`. ✅
2. **The fingerprint both ways.** `fingerprint_both_ways.py` exits 0 on R1's two stores and R0's
   two, every field equal but `encoding_format`; the five breakages are each caught. ✅
3. **L4 against R0.** The statistics counts, the reference stage counts, the chain's identities,
   the stage sequences, the printed lines and the registry fields are R0's. Each difference in the
   logs is recorded and explained below; none opens an issue. ✅
4. **The lock** was taken before `reference`, its `pid` set at each launch, held through the
   readers, and released, every command and exit status in `lock_04.txt`. Afterwards `status`
   reports no lock and `ray-processes` lists nothing. ✅
5. **The seven roots pass at 03's counts**, with README §8's known wall-clock tests. ✅
6. **`docs/runregistry-extraction-verification.md` gains §2**, in §1's shape, with §2.10 for W1.
   §1 is not edited. ✅
7. **The board records R1 as holding**, naming `runregistry-extraction-04-reference-20261014T154651`
   and `runregistry-extraction-04-resumed-c-20261014T161739`. ✅

**L1–L3** (verification §2.2–§2.4):
- **L1.**
  - Every launch ended as recorded, and each driver's exit status agrees.
  - No Ray process of this repository was left after any launch.
  - Every manifest is as the prompt's §3 requires. Every `stdout.log` is as §1.2.5's table, with
    0 lines naming `datastorekit`, 0 `DeprecationWarning` and 0 `_timestamp`.
  - The guard refused before Ray started, named the five files, changed nothing and created no run.
- **L2.**
  - **The fingerprints.**
    - Both stores are format 6, `39809dca…`, all 21 classes equal, each equal to its sidecar's.
    - `compare_fingerprints` is empty between them, and against the fingerprints recorded by R0's
      two run directories.
  - **The prune and the recompute.**
    - `resumed-a` left 207 `GkSource` rows, 48 unvalidated, listed by serial.
    - `prune_check.py` after `resumed-b`, and again after `resumed-c`, gives 48 checked, 48 absent,
      0 reused, 0 still unvalidated. After `resumed-b` there is nothing to check.
    - `GkSource` 207 + 132 − 48 = 291; QSI 424 + 440 = 864. No stored row was recomputed, and
      nothing is duplicated.
  - **Integrity.** The version row `(1, '2025.1.1')` is on all 8 shards after every launch;
    `replication_in_flight` is empty and there is no journal after every launch. The census gives
    `no divergence` on both stores, and `store_soundness.py` `=> clean`; each output equals R0's
    once the campaign and prompt are normalised.
- **L3.**
  - All six readers exit 0, and so does `extract_Gk_data.py` without `--models`, with the skip
    line.
  - Nothing refused, and every store file is byte-identical. `store fingerprint` afterwards matches
    `resumed-c`'s.
  - The lookups equal R0's queue by queue. File counts differ in the readers that draw random
    subsamples.

**W1** (verification §2.10). On each of the four stores all 20 checks hold: the package's
`encoding_format` is `ENCODING_FORMAT`, 1; SGK's has none; without it the two fingerprints are
equal, keys and classes in the same order; format 6 and `taken` `None` on both; the digest
`39809dca…` on both; the listings 7542 lines each, equal but the first, which gains
` encoding_format 1`; `compare_fingerprints(pkg, pkg)` and SGK's `(sgk, sgk)` empty, and the
package's `(sgk, pkg)` one `encoding` entry (recorded `None`, current 1); the sidecar's recorded
fingerprint compares empty with SGK's. Exit 0, 11 s. Every file of the four stores (24) had the same
SHA-256, `mtime_ns` and size before and after.

**L4 against R0** (verification §2.5):

| | R0 | R1 |
|---|---|---|
| Statistics: total | 291 | 291 |
| numeric | 109, all complete | 109, all complete |
| mixed | 26: 25 complete, 1 minimal | 26: 25 complete, 1 minimal |
| WKB | 156, all complete | 156, all complete |
| `reference`'s block, line for line | `complete` first | `minimal` first; equal sorted; equal to `dk-and-rpk-migration` R2's line for line |
| `resumed-b`'s block | the same counts | equal to R0's `resumed-b` line for line |
| `resumed-c`'s block | the same counts | equal to R0's `resumed-c` line for line |

The reference's stage counts equal R0's in every stage (verification §2.3.1).

| Resumed-chain identity | R0 | R1 |
|---|---|---|
| Finished stages compute 0 on the next launch | yes | yes |
| Pruned at `resumed-b`'s open = left unvalidated by `resumed-a` | 63 = 63, all absent, none reused | **48 = 48, all absent, none reused** |
| `resumed-b` `GkSource` = 291 − stored + pruned | 190 = 291 − 164 + 63 | 132 = 291 − 207 + 48 |
| `resumed-c` QSI = 864 − stored by `resumed-b`, nothing else | 378 = 864 − 486 | 440 = 864 − 424 |
| Totals over the chain | 291, 864 | 291, 864 |

**The stage sequences.** Each run's `** ` lines (26, 23, 26, 26) equal R0's same slug in order, once
the campaign and prompt, the path and the numbers are normalised. The reference's 15 stage banners
are equal verbatim.

**The printed lines and the registry** (verification §2.5.6):
- In all four runs, the completion lines (10, 7, 9, 10), the `Queue summary` lines and the
  `(no compute items …)` lines (1, 1, 7, 9) equal R0's in number. Normalised, their forms equal
  R0's.
- R1's `reference` and `resumed-c` printed one progress line each, in R0's first-line form.
- Each run's `status.json` has R0's keys, its `stage` is R0's, and `units_done`/`units_total` are
  0/`null`, as at R0.

**The stage times** (not compared):

| Stage | R1 reference | R0 reference | `dk-and-rpk-migration` R2 reference |
|---|---|---|---|
| `GkSource` | 1 m 42.9 s | 1 m 18 s | 41.8 s |
| QSI | 8 m 14.9 s | 20 m 30.1 s | 4 m 59.2 s |
| The whole build | 13 m 57 s | 29 m 19 s | 11 m 25 s |

**The log differences from R0, slug by slug** (verification §2.2.5). With times, pids, paths and
numbers stripped:
- **`reference`:** `stderr.log` identical. `stdout.log` and the launch log lack two of R0's kinds:
  one `Queue summary` in `/sec`, and the two-rate `inflight` line of R0's later progress lines
  (R1's QSI stage printed one progress line, R0's three).
- **`resumed-a`:** `stderr.log` identical. R1 has one `Queue summary` in `/sec` that R0 printed in
  `/min`, and lacks R0's GkWKB progress line and its `inflight` line (the stage took 4 m 0.4 s here).
- **`resumed-b`:** loses the SciPy `IntegrationWarning`'s four lines (R1's `resumed-b` stored 424 QSI
  rows before the point against R0's 486, so the one integral fell after it), and R0's completion
  time in `Nm Ns` (`GkSource` 32.5 s here against 1 m 10 s). The integrity warning's kinds are R0's.
- **`resumed-c`:** gains the `IntegrationWarning`'s four lines, and one progress line and its
  `inflight` line (its QSI stage took 5 m 43.8 s).
- **Nothing else.** There is no traceback and no new warning in any build log, and no line naming
  `datastorekit`, `DeprecationWarning` or `_timestamp`.

**The reader logs against R0's.** Six are identical in line kinds. `extract_GkWKB_data`'s queue
printed one progress line where R0's printed two; it lacks R0's second, two-rate `inflight` kind.

**`var/` outside the two stores.** The 26 files under `var/datastores/` have the same SHA-256, mtime
and size before `reference` (15:45:40) as after the last step (16:41:55). The digests are `3e0e6bad…`
over the 26 and `023d139f…` over the 14 tombstones, the orchestrator's. The only new files are R1's
ten store files and two sidecars.

**The suites.** Run serially, in the foreground, after the last launch, the readers, the guard and
the release, with README §8's command.

| Root | Dispatch baseline | Now | Load (1 min) |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 29.5 |
| CosmologyModels | 39 OK | 39 OK | 29.5 |
| ObjectFactories | 271 OK | 271 OK | 29.5 |
| tools | 91 OK | 91 OK | 25.5 |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 19.0 |
| RunRegistry | 286 OK | 286 OK | 14.2 |
| ComputeTargets | 693 OK | 693, 2 failures: `test_build_cost` (0.653 s against 0.5 s) and `test_wall_time_per_object` (0.0923 s against 0.06 s) | 7.8 |

Both failures are README §8's known wall-clock tests. Run alone three times each at load 11–13:
`test_build_cost` passed all three; `test_wall_time_per_object` failed all three (0.0805 s,
0.105 s, 0.113 s), as it failed alone twice at 03. No code either runs has changed. `black --check`
on `docs/runregistry-extraction/fingerprint_both_ways.py`: clean.

## The deliberate-breakage record

Five mutations of the W1 script (prompt §2; README §5 rule 8). Each was applied to a copy of the
committed script in the agent's scratchpad (`breakage/fingerprint_both_ways_m<n>.py`), never
committed, and run as
`PYTHONPATH=. ./venv/bin/python <copy> var/datastores/runregistry-extraction-04-reference.sqlite`.
The reference store's files were unchanged by the five runs. The diffs, exactly as applied:

**1. `encoding_format` is not removed from `pkg` before the dictionary comparison.**

```diff
--- fingerprint_both_ways.py
+++ breakage/fingerprint_both_ways_m1.py
@@ -154,7 +154,7 @@
     )
 
     # 3. equal but for encoding_format, as dictionaries and in order
-    stripped = {k: v for k, v in pkg.items() if k != "encoding_format"}
+    stripped = dict(pkg)
     diffs = differences(stripped, sgk)
     check(
         "package without encoding_format == SGK's, as dictionaries",
```

Exit 1, 18 of 20 checks holding. The lines that name the failure:
```
  [FAIL] package without encoding_format == SGK's, as dictionaries: 1 differing path(s)
      encoding_format: package 1, SGK '<absent>'
  [FAIL] the keys in the same order: first differ at position 1: package 'encoding_format', SGK 'classes'
```

**2. One class's `digest` in `pkg` is altered after it is taken.**

```diff
--- fingerprint_both_ways.py
+++ breakage/fingerprint_both_ways_m2.py
@@ -126,6 +126,7 @@
     inventory = read_inventory(str(primary), factories)
     sgk = sgk_stores.fingerprint_of(inventory)
     pkg = package.fingerprint_of(inventory, sgk_stores.FINGERPRINT_FORMAT)
+    pkg["classes"]["GkSource"]["digest"] = "0" * 64
     report(
         f"  SGK format {sgk.get('fingerprint_format')}, digest {sgk.get('digest')}, "
         f"{len(sgk.get('classes') or {})} classes"
```

Exit 1, 18 of 20. The output names the class:
```
  [FAIL] package without encoding_format == SGK's, as dictionaries: 1 differing path(s)
      classes.GkSource.digest: package '0000000000000000000000000000000000000000000000000000000000000000', SGK '6a001ba4e73566f1d6f1fc04529d71eec38c38dbe2f9607309e0c46d90bee79a'
  [FAIL] listings equal line for line but the first: 1 differing line(s), first at line 6082
```

**3. Format `5` is passed to the package's `fingerprint_of`.**

```diff
--- fingerprint_both_ways.py
+++ breakage/fingerprint_both_ways_m3.py
@@ -125,7 +125,7 @@
 
     inventory = read_inventory(str(primary), factories)
     sgk = sgk_stores.fingerprint_of(inventory)
-    pkg = package.fingerprint_of(inventory, sgk_stores.FINGERPRINT_FORMAT)
+    pkg = package.fingerprint_of(inventory, 5)
     report(
         f"  SGK format {sgk.get('fingerprint_format')}, digest {sgk.get('digest')}, "
         f"{len(sgk.get('classes') or {})} classes"
```

Exit 1, 15 of 20. The overall digest differs, and `fingerprint_format` is named:
```
  [FAIL] package without encoding_format == SGK's, as dictionaries: 2 differing path(s)
      fingerprint_format: package 5, SGK 6
      digest: package 'b40736b500a0e6983d6efb00edf08ce0c8188dae485d30d305e013604b43a33f', SGK '39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2'
  [FAIL] package fingerprint_format == 6: 5
  [FAIL] package digest == 39809dca…: b40736b500a0e6983d6efb00edf08ce0c8188dae485d30d305e013604b43a33f
```
The first-line and `compare_fingerprints(sgk, pkg)` checks fail too (the latter gives a `format`
entry and an `encoding` entry).

**4. One tag set's `count` in `sgk` is incremented.**

```diff
--- fingerprint_both_ways.py
+++ breakage/fingerprint_both_ways_m4.py
@@ -125,6 +125,7 @@
 
     inventory = read_inventory(str(primary), factories)
     sgk = sgk_stores.fingerprint_of(inventory)
+    sgk["classes"]["GkSource"]["tag_sets"][0]["count"] += 1
     pkg = package.fingerprint_of(inventory, sgk_stores.FINGERPRINT_FORMAT)
     report(
         f"  SGK format {sgk.get('fingerprint_format')}, digest {sgk.get('digest')}, "
```

Exit 1, 18 of 20. The output names the class and the tags:
```
  [FAIL] package without encoding_format == SGK's, as dictionaries: 1 differing path(s)
      classes.GkSource.tag_sets[0] tags [GkOneLoopDensity, LargestSourceRedshift_6.8788e+13, ResponseRedshiftGrid_97_d128e5d7, ResponseSparsenessZ_12, Run_default, SmallestSourceRedshift_250, SourceGridConstruction_2, SourceRedshiftGrid_1159_7139cb9c].count: package 291, SGK 292
  [FAIL] listings equal line for line but the first: 1 differing line(s), first at line 6083
```

**5. The listing check compares the first lines unmodified.**

```diff
--- fingerprint_both_ways.py
+++ breakage/fingerprint_both_ways_m5.py
@@ -217,7 +217,7 @@
     expected_first = f"{sgk_lines[0]} encoding_format {EXPECTED_ENCODING}"
     check(
         "package's first line is SGK's + ' encoding_format 1'",
-        pkg_lines[:1] == [expected_first],
+        pkg_lines[:1] == sgk_lines[:1],
         f"SGK {sgk_lines[0]!r}; package {pkg_lines[0]!r}",
     )
 
```

Exit 1, 19 of 20:
```
  [FAIL] package's first line is SGK's + ' encoding_format 1': SGK '# fingerprint_format 6 digest 39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2'; package '# fingerprint_format 6 digest 39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2 encoding_format 1'
```

**Order** (the orchestrator's note, correction 5) is checked by the two order lines of check 3.
No sixth mutation was added for it. Breakage 1 shows the key-order check failing, at position 1.

Each breakage run ends `summary: 1 store(s); FAILED on 1: var/datastores/runregistry-extraction-04-reference.sqlite`.
The diffs and outputs are kept under `var/runregistry-extraction-04/breakage/`, not committed.

## Observations not acted on

1. **A killed run's `status.json` says `killed` a few seconds before its record is complete.** The
   orchestrator's first wait, on `resumed-a`, ended when `status.json` first said `killed`
   (16:11:09). The fingerprint was taken at 16:11:14 and the final heartbeat came at 16:11:15, while
   the driver was still alive. This is the ordering the launcher and the registry intend:
   - the driver's signal handler records `killed` at once, with the pool still open
     (`docs/gktk-remedial/scoped_pipeline_run.py`, `install_killed_handler`);
   - `finish_after_exec` then calls `Run.finish(..., fingerprint=True)` once the pipeline has
     unwound, which adds the fingerprint (`RunRegistry/__init__.py`, `Run.finish`'s docstring: "Pass
     it only once the store is closed: a driver's signal handler … does not").

   A wait that ends on the pid's death, as `CLAUDE.md`'s liveness rule defines it, sees the complete
   record. The orchestrator's later waits did so. `RunRegistry/` is frozen, and this is its design,
   not a defect, so no issue is opened. A waiter that reads `state` alone should know it.
2. **The load was lighter than at R0's builds, but not light.** At the checks the one-minute load
   ran 11–100, and it reached 161 during the readers. R0's ran 58–158, with a five-minute load up to
   339. The reference took 13 m 57 s against R0's 29 m 19 s. Its `GkSource` stage was slower than
   R0's (102.9 s against 78 s) while its QSI stage was much faster (494.9 s against 1230.1 s). So
   stage times under a shared load do not scale together. The 40 % rule still placed `resumed-a`'s
   signal inside its stage, with rows stored and unvalidated.
3. **The prune at open is exercised for the second rehearsal running.** A hold sized from the
   reference's own stage time worked under two different loads: 63 of 164 rows unvalidated at R0,
   48 of 207 here.
4. **The statistics block's order moved again.** R1's reference prints `minimal` first, R2's order;
   R0's printed `complete` first. `resumed-b` and `resumed-c` match R0's same slug. This is
   `[02-the-policy-statistics-block-orders-its-lines-by-completion]`, already open, with one more
   instance and no new fact.
5. **`[07-a-killed-builds-ray-agents-outlive-the-driver]` did not recur** after either killed launch
   (checks ~31 s and ~27 s after the final heartbeats, at one-minute loads of 21 and 57). The issue
   stays open, and is not this campaign's.
6. **The wall-clock test failed alone three times** at load 11–13. It now fails alone on most runs
   at this machine's load, as at 03. README §8 lists it, and 09's final measurement already plans to
   re-run it alone at low load (board §5.2).
7. **No other project's Ray cluster appeared during R1.** The machine-wide `pgrep` was empty before
   every launch and after every run. The lock was free at `acquire`, and `~/ScienceData/` held
   `ChamPBH-rehearsals/` alone before and after.

## State handed to the next prompt

- **R1 holds** (board §2), resting on:
  - the dispatch `HEAD` `89ff8b7`, whose code is `5b81f2e`'s;
  - this commit;
  - the runs `runregistry-extraction-04-reference-20261014T154651` and
    `runregistry-extraction-04-resumed-c-20261014T161739`, both `done`, both `39809dca…`, equal to
    R0's recorded fingerprints.
- **W1 holds.** DatastoreKit `v0.4.0`'s `fingerprint_of`, given SGK's inventory and format 6, equals
  SGK's in every field but `encoding_format`, on four real stores. 06 can make SGK call it. 08 runs
  `docs/runregistry-extraction/fingerprint_both_ways.py` again. By 08, R0's and R1's stores are
  retired, and SGK's own `fingerprint_of` is gone with `RunRegistry/`. So the script will need
  inputs or imports adjusted to the tree 06 leaves; that is 08's to state.
- **R1's and R0's stores** are `var/datastores/runregistry-extraction-0{2,4}-{reference,resumed}.sqlite`,
  each with a sidecar carrying an SGK fingerprint without `encoding_format`, and none retired. By D6
  the orchestrator retires all four after this prompt's review, while SGK's own code still compares
  them (`[00-recorded-fingerprints-lack-the-encoding-format]`). Their run directories keep the
  recorded fingerprints, which 08 compares field by field.
- **The lock is free** and no Ray process of this repository is running.
- **The venv is unchanged:** 04 installed nothing. `docs/OPEN_ISSUES.md` stays at 107.
- **The registry** holds 37 runs: 35 finished, 2 `unknown`, none `running`.
