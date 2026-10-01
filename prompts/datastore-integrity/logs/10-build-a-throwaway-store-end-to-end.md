# Log 10 — Build a throwaway store end to end, interrupt it, resume it, and read it back

**Prompt:** [`prompts/datastore-integrity/10-build-a-throwaway-store-end-to-end.md`](../10-build-a-throwaway-store-end-to-end.md)
**Commit:** *(this commit)* — "Build a throwaway store end to end, interrupt it and read it back"
**Model:** Claude Opus 5.5
**Date:** 2026-10-01 to 2026-10-02
**Record:** [`docs/datastore-integrity-verification.md`](../../../docs/datastore-integrity-verification.md),
which holds every measurement; this log says what was done, how it departed from the prompt, and
what it found.
**Result:** COMPLETE, with five decisions of the user's taken on the agent's stops (below). V1–V3
done. After the user's pruned resume, `compare_fingerprints(reference, resumed)` is empty at format
6, and the census, `integrity_check`, `foreign_key_check` and the inventory are clean on both
stores. It closes `[09d-distinct-lines-test-is-satisfied-by-the-label-it-cannot-police]` and opens
four issues, one of them assigned to `handover`.

Suites after this change, serially, on the staged tree (2026-10-02, 00:45–00:51, load average
10.20 at the start and 17.73 at the end), every count at the pre-dispatch baseline:

| Suite | Baseline at `d144d04` | This tree |
|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK |
| ComputeTargets | 665 OK | 665 OK (the known wall-clock flake did not fire) |
| CosmologyModels | 39 OK | 39 OK |
| Datastore | 551 OK | 551 OK, unchanged in count: the one test amended, none added |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) |
| RunRegistry | 199 OK | 199 OK |

```
AdaptiveLevin: Ran 32 tests in 0.052s OK
ComputeTargets: Ran 665 tests in 152.775s OK
CosmologyModels: Ran 39 tests in 0.679s OK
Datastore: Ran 551 tests in 131.323s OK
LiouvilleGreen: Ran 148 tests in 24.370s OK (skipped=1)
RunRegistry: Ran 199 tests in 62.253s OK
```

The index `docs/OPEN_ISSUES.md` goes from **99 to 102** rows, counted
(`grep -c '^| \`\[' docs/OPEN_ISSUES.md`): one deleted (the 09d issue), four opened.

## The decisions, as the brief gave them

- **D10-1 to D10-5 were taken by the user on 2026-10-01, all as recommended**, exactly as the
  prompt's §1.1 states them: LambdaCDM only, three wavenumbers over $10^5$–$10^6$/Mpc, `--zend 250`,
  `--source-samples-log10z 100`, `--shards 4`, `--cpus 6`, `main.py`'s defaults; two builds
  compared; two `SIGTERM` interruptions, in the `GkSource` and `QuadSourceIntegral` stages; four
  fault copies outside `var/`; both stores kept.
- **Decided on the agent's stops** (relayed by the orchestrator, each the user's):
  1. *2026-10-01, after `resumed-a`.* Accept `failed`/1 for an interrupted launch, with the
     interrupter's log, Ray's `SIGTERM received` banner, no traceback and the part-stored stage as
     its evidence; open an issue against the driver's `terminal_state`; carry on.
  2. *2026-10-01, after `resumed-b`.* `SIGKILL` the orphaned `gcs_server` 6426 after checking its
     command line; run `resumed-c` and write up, accepting the `GkSource` duplicate's consequences.
     Then, authorised explicitly: end `resumed-b` as `killed` with `Run.finish(…, exit_code=None)`,
     after confirming `kill -0 6384` fails.
  3. *2026-10-01, after `resumed-c`.* "The bug here is the `--no-prune-unvaiidated`. Pruning is there
     precisely to deal with this situation." So the `GkSource` existence query is not the defect;
     open the issue against A3 v2's launch, owned by `handover`; add **`resumed-d`**, the same launch
     without `--no-prune-unvalidated`; after it, the prompt's standard in full. The board header uses
     the prompt's own text only if R1–R2 hold on the pruned store (they do).
  4. *2026-10-02, at §2.3.* Accept each extraction script's exit 1 (a full LambdaCDM read, then
     `QCD_Cosmology`'s `RuntimeError`) in place of "each must exit 0"; no harness; open an issue
     against the six scripts; the fault copies still open through an extraction script.

## Every run

All on 2026-10-01 (BST), at `d144d04`, `git_dirty: true` (only this prompt's new files were in the
tree, `?? docs/datastore-integrity-verification/`).

| Run id | Slug | Pid | State, exit | Wall | Load average at launch | Interrupter pid |
|---|---|---|---|---|---|---|
| `datastore-integrity-10-reference-20261001T204320` | `reference` | 84388 | `done`, 0 | 14 m 58 s | 22.52 20.59 19.74 | — |
| `datastore-integrity-10-resumed-a-20261001T210301` | `resumed-a` | 1238 | `failed`, 1 | 2 m 25 s | 15.36 18.50 22.48 | 1264 |
| `datastore-integrity-10-resumed-b-20261001T230813` | `resumed-b` | 6384 | none written (aborted 23:15:32); `killed`, `null` by a person at 23:44:51 | 7 m 19 s | 10.67 10.13 11.51 | 6395 |
| `datastore-integrity-10-resumed-c-20261001T234503` | `resumed-c` | 38930 | `done`, 0 | 8 m 14 s | 12.35 16.12 19.20 | — |
| `datastore-integrity-10-resumed-d-20261001T235455` | `resumed-d` | 48129 | `done`, 0 | 33 s | 10.62 19.08 21.16 | — |

Before each: `RunRegistry list` showed nothing `running`, `ray status` found no instance, and
`RAY_ADDRESS` was unset. After each, the one check, and the turn ended; the orchestrator's wait
ended `finished` for every run but `resumed-b`, whose ended `stale`.

## What shipped

- **`Datastore/tests/test_inventory_report.py`**, the one test (§3): line 32 of the staged file,
  `from Datastore.tests.test_inventory_report_labels import OWN_LABEL`; lines 163–165, the comment
  gains why the label is removed; lines 173–176, the comparison is of
  `{(group, OWN_LABEL.sub("", line)) for group, line in section.records}` against
  `len(set(cls.records))`. Nothing else in the test or the module changed.
- **`docs/datastore-integrity-verification.md`**, new: the record.
- **`docs/datastore-integrity-verification/`**, new:
  - `interrupt_when.py` (the interrupter), `interrupter_dummy_test.sh` and its output
    `interrupter_dummy_test.txt`, `interrupter_firings.txt` (both real logs);
  - `compare_stores.py`, `stage_counts.py`, `qsi_resolution_cost.py`, `inventory_reports.py`,
    `fault_copies.py`, each with its saved output (`fingerprint_comparison.txt`, `stage_counts.txt`,
    `qsi_resolution_cost.txt`, `inventory_reports.txt`, `fault_cases.txt`);
  - `sigterm_in_ray_wait_probe.py`, `sigterm_alarm_probe.py` and `sigterm_probes.txt`;
  - `census.txt`, `integrity_and_foreign_keys.txt`, `declared_foreign_keys.txt`,
    `inventory_problems.txt`, `fingerprint_after_reading.txt`, `extractions.txt`,
    `resumed_d_pruning.txt`, `substitute_main_py.txt`.
- **The board**, **`docs/OPEN_ISSUES.md`** and this log.

No production code, no driver, no `tools/`, no `RunRegistry/`, no `config/`, no extraction script
and no audit probe changed.

## The interrupter: its dummy test and its two firings

`interrupt_when.py <run dir> <stage regex> <percent> [--interval 15] [--stale-after 900] [--hold S]`.
Every 15 s it reads `status.json`; it exits without signalling if the run is terminal or its pid
fails `kill -0`, waits while `RunRegistry.liveness(status, 900)` is `stale`, and when `alive` with
the stage matching at or above the percentage (a bare banner reads 0 %; `… all work items complete`
is never eligible) it checks `kill -0` and `ps -p <pid> -o command=` for `scoped_pipeline_run.py`
and `--register <slug>` (the slug from the manifest's `argv`), sends `SIGTERM` once and exits. Its
log is `<run dir>/interrupter.log`.

**The dummy test**, re-run on the committed script (`bash
docs/datastore-integrity-verification/interrupter_dummy_test.sh <work dir>`, polling every 1 s); the
logs, from `interrupter_dummy_test.txt`, with the work directory masked:

```
=== case A: signals when the stage matches
interrupter exit=0
child exit=143 (143 = killed by SIGTERM)
… interrupter pid … watching <work dir>/case-a (slug 'dummy-a'); stage regex '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, ' at >= 40.0%; every 1 s; stale after 900 s; hold 0 s
… liveness stale; pid …; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, 45.00% (11 of 20 work items remaining)'
… liveness alive; pid …; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, 10.00% (18 of 20 work items remaining)'
… liveness alive; pid …; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, 45.00% (11 of 20 work items remaining)'
… sent SIGTERM to pid … at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, 45.00% (11 of 20 work items remaining)' (45% >= 40%); command 'python docs/gktk-remedial/scoped_pipeline_run.py --register dummy-a 120'; exiting
=== case B: exits silently when the child dies first
child exit=0 (0 = sleep ended by itself)
interrupter exit=0
… liveness alive; pid …; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, 10.00% (18 of 20 work items remaining)'
… run's pid … no longer answers kill -0; exiting without signalling
```

Cases C (a plain `sleep`: refused, exit 2, child alive), D (`--hold 4` at a bare banner: holds,
reads again, signals) and E (`--hold 4`, the stage completes in the hold: no signal) are in the same
file. The file holds the timestamps and pids; they are elided here only.

**The two firings** (`interrupter_firings.txt`):
- `resumed-a`, pid 1264, `'^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --hold 8`: banner seen
  at 21:05:17, "holding 8 s", `SIGTERM` to 1238 at 21:05:25.
- `resumed-b`, pid 6395, `'^CALCULATE QUADRATIC SOURCE INTEGRALS, ' 100`: `CALCULATE QUADRATIC SOURCE
  INTEGRALS, 100.00% (0 of 2 work items remaining)` at 23:15:28, `SIGTERM` to 6384 at 23:15:29.

Each logged the pid's full command line, which named `scoped_pipeline_run.py` and the slug.

**That each stage had stored some rows and not all.** `resumed-a`'s `GkSource` stage printed no
progress line; the store it left held 161 of 291 `GkSource` rows and 0 `GkSourcePolicyData`, and
`resumed-b` computed 175 of 291. `resumed-b`'s last progress line before the signal read `864
lookup … 485 compute … 451 store`, with 379 computes in flight; the store held 485 of 864, and
`resumed-c` computed 379.

## §2.1's tables

Lookup / compute / store per stage, from each `stdout.log` (`stage_counts.txt`):

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` | `resumed-d` |
|---|---|---|---|---|---|
| exit times, source k | 3/3/3 | 3/3/3 | 3/0/0 | 3/0/0 | 3/0/0 |
| exit times, response k | 3/0/0 | 3/0/0 | 3/0/0 | 3/0/0 | 3/0/0 |
| Tk numeric | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 | 0/0/0 |
| Tk WKB | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 | 0/0/0 |
| quadratic source terms | 6/6/6 | 6/6/6 | 0/0/0 | 0/0/0 | 0/0/0 |
| Gk numeric | 1368/1368/1368 | 1368/1368/1368 | 0/0/0 | 0/0/0 | 0/0/0 |
| Gk WKB | 3477/3477/3477 | 3477/3477/3477 | 0/0/0 | 0/0/0 | 0/0/0 |
| `GkSource` | 291/291/291 | killed, no progress line | 175/175/175 | 0/0/0 | 0/0/0 |
| `GkSourcePolicyData` | 291/291/291 | — | 291/291/291 | 0/0/0 | 0/0/0 |
| `QuadSourceIntegral` | 864/864/864 | — | killed at 864/485/451 | 379/379/379 | 0/0/0 |

- Every stage finished before an interruption computed zero on the next launch.
- `GkSource`: of `resumed-a`'s 161 stored rows, 45 had `validated = 0`, stored but not yet
  validated at the `SIGTERM`; under `--no-prune-unvalidated` `resumed-b` recomputed them beside the
  130 missing. No row stored **validated** was recomputed, so §5's condition was not met.
- `QuadSourceIntegral`: the 379 in flight at the `SIGTERM` were never stored; `resumed-c` computed
  exactly those.
- Summed over `resumed-a`–`c`, less the 45 that `resumed-d` pruned, each stage stores what the
  reference stored (the user's form of §2.1's sum, decision 3).

The `QuadSourceIntegral` stage's banner to its first compute (`qsi_resolution_cost.txt`):
`reference` 17.0 s, `resumed-b` 16.9 s, `resumed-c` (485 of 864 stored) 16.3 s; `resumed-d` (all
stored) ran the whole stage in 6.78 s.

D10-2, after `resumed-d` (`fingerprint_comparison.txt`): `compare_fingerprints(reference, resumed)`
**0 differences**; both format 6, digest `10b8725f36bd2aa14f74cc8912efe3dd80961c4fc015fe31729b52aad4f61a1f`;
all 21 classes equal in count and digest (the record's §2.1.5 has the table).

## Deviations from the prompt

1. **`PYTHONPATH=.` on every launch.** §1.3's command omits it; the driver's `register()` imports
   `RunRegistry`, and `main.py` its packages, from the repository root, and the driver's usage line
   sets it. STRUCTURALLY REQUIRED.
2. **The interrupter is not registered as a run of its own.** Its lifetime is bounded by the build
   it watches; its log is in that build's run directory and starts with its own pid and arguments.
   Launched detached (`nohup`, `disown`), output to `/dev/null`, its log written by itself.
   IMPLEMENTATION CHOICE.
3. **`--hold S`, added to the interrupter after its first dummy test.** The `GkSource` stage printed
   no intermediate progress line, so it could only be interrupted from its banner, and with a 15 s
   poll a signal on the first reading could land before anything was stored. `--hold 8` put it
   8–23 s into a 34 s stage. The dummy test was re-run with two more cases. IMPLEMENTATION CHOICE.
4. **The dummy test polls every 1 s and has three cases beyond the two asked** (a pid whose command
   line does not name the run; the hold; a stage that completes during the hold). IMPLEMENTATION
   CHOICE.
5. **The `QuadSourceIntegral` interruption is at 100 %.** The pool's percentage counts work items
   taken off the queue, and this scope's 864 integrals are 2 work items, so every progress line reads
   100.00 %; the first line (5 m 01 s in, 594 of 864 stored in `reference`) is the only point a
   percentage can pick. STRUCTURALLY REQUIRED.
6. **`resumed-a` is `failed`/1, not `killed`** (§2.1), by the user's decision 1. STRUCTURALLY
   REQUIRED by the driver: a `SIGTERM` in `ray.wait` reaches Python as `SystemExit(1)`.
7. **`resumed-b` wrote no terminal state, and was ended `killed` by a person**, `Run.finish('killed',
   exit_code=None)`, after `kill -0 6384` failed, on the user's explicit authorisation (decision 2).
   The evidence for `killed`: the interrupter's `SIGTERM` at 23:15:29, the driver's failure at
   23:15:32.164 in the GCS log, the probe's abort status 134. The resulting `status.json`:
   `{"exit_code": null, "heartbeat": "2026-10-01T23:44:51+01:00", "pid": 6384, "stage": "CALCULATE
   QUADRATIC SOURCE INTEGRALS, 100.00% (0 of 2 work items remaining)", "state": "killed",
   "units_done": 0, "units_total": null}`. The orphaned `gcs_server` 6426 (parent 1, started
   23:08:16, `--log_dir=/tmp/ray/session_2026-10-01_23-08-16_091862_6384/logs`) was killed with
   `SIGKILL` after its command line was checked, on the user's decision; it was gone, `ray status`
   found no instance, and no Ray process remained. STRUCTURALLY REQUIRED (`fingerprint_store`
   refuses a store any `running` run names, and §1.3 requires nothing `running` before a launch).
8. **A fifth launch, `resumed-d`**, by the user's decision 3: `--allow-existing`, no interrupter,
   and without `--no-prune-unvalidated`, the only difference. §2.2–§2.5 and §3 ran on the store it
   left, and §2.1's sum reads "less the 45 pruned". STRUCTURALLY REQUIRED by the finding, the
   user's to decide.
9. **Two diagnostic probes, each with its own local Ray instance, no store**
   (`sigterm_in_ray_wait_probe.py`, `sigterm_alarm_probe.py`), run while no build was alive. The
   second left its own `gcs_server` (pid 17217), hung after `SIGTERM`; I killed it with `SIGKILL`,
   since it was mine, before asking about 6426. IMPLEMENTATION CHOICE.
10. **The store after `resumed-b` was read with `read_inventory` directly**, not `store
    fingerprint`, which refuses while a `running` run names it; every file's SHA-256 was the same
    before and after. IMPLEMENTATION CHOICE.
11. **§2.3: every extraction script exits 1**, after a full LambdaCDM read, on `QCD_Cosmology`; the
    result stands in place of "each must exit 0" by the user's decision 4. Each was run once, with no
    harness.
12. **§2.5's opener is `extract_TkWKB_data.py`**, the quickest reader (15 s). The repair case
    passes on its open proceeding and LambdaCDM being read; its later exit 1 is decision 4's.
    IMPLEMENTATION CHOICE.
13. **The mutation of §3 drops the value count as well as the fields.** With the fields alone, the
    fixture's `| values: N` suffixes keep the lines of six of the seven labelled classes distinct,
    and only `GkSourcePolicyData` failed. STRUCTURALLY REQUIRED. And **it is caught for four of the
    seven labelled classes, not each**: the fixture holds one record per tag set of `QuadSource`,
    `GkNumericIntegration` and `GkWKBIntegration`, and the test compares within a tag set, so no
    mutation can fail it for them. STRUCTURALLY REQUIRED; opened as an issue.
14. **The test imports `OWN_LABEL` from 09d's test module** rather than defining a regex, which
    would also need `import re`: one import, as the brief allows. IMPLEMENTATION CHOICE.
15. **The fault copies' work directory was in the session scratchpad**, outside `var/` as §2.5
    requires, and was deleted afterwards; the outputs kept are in `fault_cases.txt`.
    IMPLEMENTATION CHOICE.
16. **A stray file outside the tree.** A `cp` meant for a temporary path copied
    `test_inventory_report.py` into an existing directory `/tmp/x/` that is not mine; I removed only
    that file, after checking it was identical to the source and timestamped that minute.
    UNINTENDED DRIFT, reverted.

## Verification performed

- **The pre-launch checks**, before every launch (above); both store names absent (`store show`:
  kind `absent`, no runs) before `reference`.
- **§2.1–§2.5**, all in the record: the five launches, the stage counts, the resolution cost, the
  pruned resume, D10-2's comparison, the census, `integrity_check` and `foreign_key_check` on ten
  files, the inventory, the six readers with a fingerprint after each, the four inventory reports,
  the four fault copies.
- **`substitute_main_py()`**, on this tree, with no Ray and no store (`substitute_main_py.txt`): its
  five `** quadsource_atol_sweep: replaced 1 occurrence of …` lines, and `count=1` for each literal,
  at `main.py` lines 4354, 4366, 4255, 4256 and 4148.
- **The fingerprint does not move by reading**: after §2.3–§2.5 both stores' `store fingerprint`
  matches its sidecar, digest `10b8725f…1a1f`.
- **`black --check`** is clean on every Python file this prompt adds or changes.
- **The suites**, serially, on the staged tree (above).

## The deliberate-breakage record

The mutation, of `tools/inventory_report.py`, pasted from `git diff` exactly as applied:

```diff
diff --git a/tools/inventory_report.py b/tools/inventory_report.py
index 2b95338..a708845 100644
--- a/tools/inventory_report.py
+++ b/tools/inventory_report.py
@@ -551,6 +551,8 @@ class _Renderer:
 
     def record_line(self, name: str, record: Record, fields: Sequence[str]) -> str:
         text = ", ".join(f"{f}={self.value_text(name, record, f)}" for f in fields)
+        if name in self.labelled():
+            text = ""
         if self.has_validated(name):
             if record.validated is False:
                 text += "  [unvalidated]"
@@ -559,7 +561,7 @@ class _Renderer:
         if name in self.labelled():
             # beyond the first double space, where a parser of the fields stops
             text += f"  #{self.ordinals(name)[0][id(record)]}"
-        if record.value_count is not None:
+        if record.value_count is not None and name not in self.labelled():
             text += f"  | values: {record.value_count:,}"
         return text
 
```

A labelled class's record line then shows its markers and its own label, and none of its fields or
its value count. It was applied with `git apply` in a `git archive` export of `HEAD` with this
prompt's test file in it, made into its own git repository (`git init`, add, commit) with `venv`
linked, and reverted with `git apply -R`.

**The amended test, mutated**:
`PYTHONPATH=. ./venv/bin/python -m unittest -v Datastore.tests.test_inventory_report.TestTheDisplaySaysWhatTheInventoryHolds.test_verbose_renders_every_distinct_record_as_a_distinct_line`
— 4 sub-tests fail, counted, then pasted:

```
FAIL: … (cls='TkNumericIntegration')    AssertionError: 2 != 3
FAIL: … (cls='TkWKBIntegration')        AssertionError: 2 != 3
FAIL: … (cls='GkSource')                AssertionError: 2 != 3
FAIL: … (cls='GkSourcePolicyData')      AssertionError: 1 != 3
Ran 1 test … FAILED (failures=4)
```

**The parent commit's test, mutated** (`git show HEAD:Datastore/tests/test_inventory_report.py` in
the export): the same command, `Ran 1 test … OK`.

Over the whole module: the amended one fails 11 (those 4, and 7 sub-tests of
`test_verbose_shows_every_key_field_of_every_class`, which sees the dropped fields); the parent's
fails the same 7 and not the distinct-lines test. Unmutated, the amended module runs 18 tests OK.

**Replays.** Before the commit, the block above was extracted from this file and applied with plain
`git apply` in a fresh `git archive` export of the staged tree made into its own git repository,
`venv` linked; the result goes in the final report, since this file cannot record a run that
follows itself. After the commit, the same from `git show HEAD:prompts/datastore-integrity/logs/10-build-a-throwaway-store-end-to-end.md`;
that result goes to the orchestrator.

## Observations not acted on

1. **`--no-prune-unvalidated` duplicates rows on a resume** — opened as
   `[10-a3-v2-launch-resumes-without-pruning-unvalidated-rows]`, assigned to `handover`. From the code,
   every class with `validate_on_startup` can do it; only `GkSource` was caught here.
2. **A `SIGTERM` is never recorded `killed`, and can leave a `gcs_server`** — opened as
   `[10-a-sigterm-to-the-scoped-driver-is-recorded-as-failed-or-not-at-all]`.
3. **The extraction scripts exit 1 on a LambdaCDM-only store** — opened as
   `[10-extraction-scripts-exit-1-on-a-store-without-every-model]`.
4. **The amended test cannot fail for three labelled classes on this fixture** — opened as
   `[10-distinct-lines-test-cannot-fail-for-three-labelled-classes-on-the-fixture]`.
5. **Ray's log deduplication hides the integrity warning.** At `resumed-d`'s startup shard 3 listed
   its 13 unvalidated rows; shards 0 and 1's 32 appear only as `[repeated 32x across cluster]`. A
   person reading the log cannot see which rows were pruned there (`RAY_DEDUP_LOGS=0` would show
   them). Not a defect of this tree.
6. **A stage with no intermediate progress line is invisible to the heartbeat between banner and
   summary**, and the `QuadSourceIntegral` stage's percentage counts batches, not rows. Neither is
   wrong; both limit what `status.json`'s `stage` can say.
7. **The verbose report of this small store is 13.3 MB** (881 kB not verbose, 7,499 records), most
   of it parent-set lines, one member per `z_source`. 09d's observations 2 and 3 stand; the size on
   a production store is unmeasured.
8. **The extraction scripts warn `number of shards read from database (=4) does not match
   specified number of shards (=10)`**: they pass their default shard count to `ShardedPool`, which
   reads the store's. Cosmetic.
9. **README §2's row for 10 still reads "written 2026-10-01 … after 09d"**: outside this prompt's
   files (§7). Residue for the orchestrator.
10. **The disk.** Free space on the volume holding `var/` was 36–45 GB through the run and Ray warned
    at every launch that `/tmp/ray` was over 95 % full. A3 v2's longer build will spill more.

## State handed to the next prompt

- **Names.** `docs/datastore-integrity-verification/interrupt_when.py` (and `--hold`),
  `compare_stores.py`, `stage_counts.py`, `qsi_resolution_cost.py`, `inventory_reports.py`,
  `fault_copies.py`; the two stores `var/datastores/datastore-integrity-10-reference.sqlite` and
  `…-resumed.sqlite`, with their sidecars, kept for the review (D10-5); `store retire` is the
  user's.
- **For A3 v2**, the checklist in the record's §2.6: in particular, resume without
  `--no-prune-unvalidated` (`handover`'s issue), and check an interrupted launch's state and `ray
  status` by hand.
- **The campaign.** 12 of 12 prompts landed. Its work is done; it is closed by the user after the
  orchestrator's review of 10.
- **Baselines** after this commit: AdaptiveLevin 32, ComputeTargets 665, CosmologyModels 39, Datastore
  551, LiouvilleGreen 148 (1 skipped), RunRegistry 199, all unchanged.
