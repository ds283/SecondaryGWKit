# Store retirement audit — what removing a datastore has to leave behind

**Date:** 2026-09-25 · **Tree:** `handover-remedial` at `42d4910`, clean · **Author:** Claude Opus
5.5 · **Status:** read-only. No file in the repository was changed to write it. Under `var/`, the
three primaries' `shards` tables were read `mode=ro`, and the sidecars and run records were read.
Nothing under `var/` was written. The throwaway probes in §2 ran in the session scratchpad, and
were deleted.

**Why it exists.** Two stores under `var/datastores/` hold nothing that is not held elsewhere: the
tolerance-sweep store and the pre-resume backup of the A3 baseline (§3). The user asked whether
they can be removed. The first answer was "yes, delete the sweep store now". That answer was wrong
about the *how*. The registry now owns every other event in a store's life: create, adopt, copy,
move and fingerprint each go through `RunRegistry.stores` and leave a `history` entry. A plain
`rm` would be the one event it cannot see. Four run manifests would then name a path that answers
nothing, which is exactly the "moved or lost?" failure `datastore-portability` was built to catch.
The user's decision (2026-09-25) is that a store is removed by a **registry operation that retires
it**: the store's files go, and the sidecar stays behind, marked retired. This audit establishes
what that operation has to do, what in the tree would defeat it, and in what order the work has to
land. The campaign that builds it is
[`prompts/store-retirement/`](../prompts/store-retirement/README.md).

**How to read the citations.** **[checked]** marks a claim re-read in the source while writing this
document. **[probed]** marks one also run, read-only or in the scratchpad. There are no
agent-reported claims in this audit.

---

## 1. The stores, and everything that names them

Measured 2026-09-25 at `42d4910`. `RunRegistry list` shows nothing `running`. **[probed]**

| Store | Files | Sidecar | `store_id` | Fingerprint (overall) | Runs naming it |
|---|---|---|---|---|---|
| live A3, `var/datastores/handover-A3-baseline-lambdacdm.sqlite` | primary + 4 shards, 339 MB | registry, adopted 2026-09-24 | `5f58ac53…` | `433b7fc3…` | `handover-03-a3-baseline-resume-20260923T024847`, by `results` |
| sweep, `var/datastores/handover-atol-sweep.sqlite` | primary + 4 shards, 338 MB | registry, adopted 2026-09-24 | `04198f22…` | `2c2dde68…` | the four `handover--quadsource-atol-sweep-*` runs, by `results` |
| backup, `var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite` | primary + 4 shards, 333 MB | registry, adopted 2026-09-24 | `4c2ce77b…` | `eedcdfb2…` | none |

Every run above predates `results_store_id` and matches by path alone (`runs_naming`,
`RunRegistry/stores.py:542-573`). **[probed]**

**References that are not run manifests.** Each is classified by whether it states something
about the past, which stays true after retirement, or about the present, which would not.

| Where | What it says | Tense |
|---|---|---|
| live A3 sidecar, `backup` (an unknown field) | `{"path": "var/datastores/backup-pre-resume-20260921T091011", "retained": true, "reason": "The resume did NOT succeed. Backup kept until a resume completes cleanly."}` | **present**: false once the backup is retired |
| sweep sidecar, `copied_from` | the legacy string `var/datastores/handover-A3-baseline-lambdacdm.sqlite` | past |
| every sidecar's `history` | where the store was at each operation | past |
| backup and live sidecars, `status_files` (unknown) | `var/runs/a3-pilot/run.{pid,err,out}` | past; points the other way |
| `var/runs/a3-pilot/BACKUP_PATH` | the backup's directory | past; pre-registry evidence |
| `docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md:15-17` | the sweep store, "**deleted after this document was written**" | **false today**, true once the sweep store is retired |
| `prompts/store-fingerprint/logs/05-fingerprints.json` | all three fingerprints as written, committed | past; survives any retirement |
| boards and logs (`datastore-portability`, `run-registry`, `store-fingerprint`, `handover`, `datastore-readback`) | prose about the stores | past |

The sweep doc's claim at `:15-17` was written in anticipation, and the store was never deleted.
The additive rule (`CLAUDE.md` campaign invariant 6) means the doc is not edited. It becomes true
when the store is retired, and the retirement record says when.

---

## 2. What the code does today at a store that is gone

**2.1 The reader calls a sidecar with no primary a problem.** `read_sidecar` adds
`_primary_problems` (`stores.py:184-193`, called at `:349`): *"the primary … does not exist, so this
sidecar describes no store beside it"*. The reading is `registry` with `ok == False`. **[probed]**
on a scratch primary with a created sidecar, then the primary removed. So a tombstone, written
with today's code, reads as a broken sidecar. It cannot be told apart from a store that was moved
by hand or lost, which is the one distinction a tombstone exists to make.

**2.2 The history format has no word for retirement.** `_history_problems` (`stores.py:200-230`)
allows only `create` or `adopt` at index 0, and only `copy` or `move` after it
(`RELOCATION_OPERATIONS`, `:102`). A `retire` entry is a problem: *"history entry #1 records
operation 'retire', where only ('copy', 'move') can stand"*. **[probed]** Every writer checks what
it is about to write (`_check_before_writing`, `:380-386`), so no writer could record a retirement
at all. The format has to grow. By the standing rule, a needed schema change is always the right
answer.

**2.3 Through the registry, a retired name cannot be reused, and that holds already.**
`create_sidecar` refuses if the sidecar name is taken (`stores.py:458-463`). **[probed]** Copy and
move refuse a destination whose sidecar name, `.tmp` name or `.incomplete-move` name exists
(`_prepare`, `:667-678`). **[checked]** So a tombstone that stays at `<stem>.manifest.json` blocks
every registry route to a new store at that name, with no new code. That is what a reference by
path needs: it keeps resolving to the tombstone, and never to an unrelated later store.

**2.4 Outside the registry, a store is recreated at a retired name silently.** The `ShardedPool`
constructor, given a primary that does not exist, creates a new store there. It refuses only if a
shard file of the new name already exists (`Datastore/SQL/ShardedPool.py:116-142`). **[checked]**
It knows nothing about sidecars. That is a user decision (`datastore-portability` README §6.3), and
this campaign does not reopen it. So after a retirement, `main.py --database <retired path>`
creates a fresh store beside the tombstone. The reader must therefore call "retired, but a primary
exists again" a problem. That is the one case in which the tombstone and the files disagree.

**2.5 A registered run at a retired name proceeds, and records no store.** `begin(results=…)`
reads the sidecar only to take its `store_id`. For anything but a problem-free registry sidecar
that is `None`, and `begin` does not refuse (`RunRegistry/__init__.py:549-553`). **[checked]** Both
registered builders reach it:
- `docs/gktk-remedial/scoped_pipeline_run.py:291-302`, with `results=str(database)`;
- `docs/handover/quadsource_atol_sweep.py --build --database …` (`:887-926`, `begin` at `:473`),
  which builds the A3 v2 store at a path it is given.

`begin` is the one point where every registered writer can be refused. Refusing there is the
registry's answer to §2.4, as far as the registry can reach.

**2.6 The sweep's `prepare()` would destroy the sweep store's tombstone, even without `--force`.**
`prepare()` (`docs/handover/quadsource_atol_sweep.py:623-686`) decides that the sweep store
"already exists" from the shards and the primary alone (`:635-642`). The sidecar is not among them.
It then writes the sidecar with `Path.write_text` (`:669-686`), which replaces whatever is there.
**[checked]** After the sweep store is retired, nothing it checks exists. So a plain
`--prepare` re-copies the baseline, and **overwrites the tombstone** with a new hand-written legacy
sidecar. `--force` would do the same. This is `datastore-portability`'s
`[03-atol-sweep-prepare-writes-its-store-sidecar-by-hand]`, whose recorded impact was "none today".
Retirement gives it one. That issue's own next step is "if the script is ever edited again for
another reason, replace its hand copy, its `UPDATE` and its sidecar with one
`RunRegistry.stores.copy_store` call". Retirement is that reason. `copy_store` refuses because the
destination sidecar exists (§2.3), which is the behaviour a retired name needs. The same edit
discharges `[01-atol-sweep-check-expects-absolute-shard-records]`, since `copy_store` writes bare
records and the hand `UPDATE` goes. Nothing else in the tree writes a `<stem>.manifest.json`: the
only other mention is `tools/sharded_store.py:17`, which says it never copies one. **[checked, by
`grep` over `docs/`, `tools/`, `Datastore/`, `ComputeTargets/` and `main.py`]**

**The second issue has an impact its board did not record: `--build --resume` refuses every store
built since `datastore-portability` prompt 01, the A3 v2 store included.**
`assert_store_is_self_consistent` (`:590-620`) requires the `shards` rows to equal the absolute
paths of the expected siblings (`:607`). The board's impact statement is "none in the script's own
workflow, because `SWEEP_STORE` is only ever made by `prepare()`". But the function has three
callers:
- `run_child` (`:372`), on the sweep store;
- `prepare()` (`:661`);
- **`run_build`'s resume branch (`:460-461`), on the `--database` it is given.**

A store the pipeline creates now records bare names (`ShardedPool._write_shard_data`, `:327-344`).
A four-shard store written by the fixtures' `write_new_store`, then checked by the sweep's own two
functions extracted with `ast`, is refused: *"the `shards` table inside v2.sqlite does not name its
own shard files"*. **[probed]** The advice it prints, "Re-run --prepare --force", is wrong for a
build. The A3 v2 rebuild is planned in stages. `QUADSOURCE-TOLERANCE-SWEEP.md` §7 says "one short
session yields 8192 of 9280 items", so `--resume` is its intended workflow, and it cannot resume
today. Comparing through `resolve_shard_path`, which the issue's own next step asks for, accepts
bare and legacy records alike. It is part of the same edit.

One existing test reads `quadsource_atol_sweep.py`. It is
`RunRegistry/tests/test_store_fingerprint.py` test 12 (`:797-842`), and it counts only `run.finish(`
calls, 6 outside the `terminal` handlers and 2 inside. `prepare()` has none, so the edit leaves it
passing. **[checked]**

**2.7 The backup's primary names the live store's shards. A delete that follows the stored
strings would delete the live A3 store.** All three primaries hold legacy absolute records.
**[probed, `mode=ro`]** The backup's four rows are, byte for byte, the live primary's four rows:
`/Users/ds283/…/var/datastores/handover-A3-baseline-lambdacdm-shard000N.sqlite`. The one resolver
(`ShardedPool._resolve_shard_rows`, `:540-582`, through `Datastore/shard_paths.py:78`
`resolve_shard_path`) reads each record as the sibling of that name. `_read_closed_store` (`:674-709`)
on the backup's primary returns the four files in
`var/datastores/backup-pre-resume-20260921T091011/`. **[probed]** So the deletion must enumerate a
store's files through `_read_closed_store`, and through nothing else. A deletion that opened a
stored record as a path would pass every test on a store built after `datastore-portability` prompt
01, whose records are bare names. On the backup it would destroy the store the backup exists to
protect. This is the campaign's most important deliberate-breakage case. A test fixture with legacy
absolute records naming *another existing store's* shards must show that only the siblings go.

**2.8 Nothing that exists deletes a file, and three places say so.**
- `CLAUDE.md:52`: the registry *"records; it does not schedule, supervise, restart, lock or
  delete"*. `datastore-portability` README §6.5 re-affirmed it on 2026-09-24, when copy and move
  were admitted (`:203-208`).
- `ShardedPool`'s closed-store block (`:613-624`): *"They never delete a file … Deleting is for a
  person."*
- The `RunRegistry/stores.py` docstring (`:47-49`), and the `RunRegistry/__main__.py` docstring
  (`:13`): *"none deletes anything"*.

`ShardedPool.move_store` removes a source only by renaming it. **[checked]** Retirement is the
first operation that unlinks a store's files. It needs the `CLAUDE.md` limit amended explicitly, as
copy and move needed §6.5's charter decision. The memory rule that charters are not dogma does not
reach this one: it distinguishes local boundaries from **`CLAUDE.md`'s limits, which bind**. The
rationale behind the limit is `run-registry` README §0 item 4: a datastore that lived in a
scratchpad was lost, and nothing on disk said where it had gone. A retirement that leaves the
sidecar, and deletes no record, is the opposite of that failure. But the limit is written without
qualification, and only the user can qualify it.

**2.9 "Is anything using this store?" can be answered only for registered runs.**
`_running_runs_naming` (`stores.py:576-591`) refuses on any `running` run, alive **or stale**, that
names the store by path or `store_id`, and copy, move and fingerprint share it. It cannot see a
process that opened the store without registering. A rollback-journal store leaves no file while
idle (`datastore-portability` README §6.4, last paragraph). A **hot journal** (`-journal`, `-wal`,
`-shm` beside a file) is visible, and `_plan_relocation` refuses it (`:731-738`). The same two
checks are all a deletion can make, and it should make both. A deletion is irreversible where a
copy is not, so it is also the one operation that must refuse the stale case rather than merely
report it, and it does, by sharing the check.

**2.10 The fingerprint prerequisite, and what a crash does to it.** Both stores to be retired carry
a fingerprint written on 2026-09-25 at `50a24ac`. `store fingerprint` reports each as `matches`
(`store-fingerprint` log 05, Phase B). "Refuse unless a recorded fingerprint matches the current
content" is a fresh `fingerprint_of(read_inventory(primary))`, compared with `compare_fingerprints`
(`stores.py:886-1003`), with no differences, `problems` included. It is a read of 5–7 s per store
(log 05). The crash cases, in order of how often they will occur:

1. **The store's writer died before `Run.finish(…, fingerprint=True)`**, so no fingerprint was
   written. Its run still says `running`, with a stale heartbeat. This is **not** manual
   intervention. A person ends the run with `Run.finish("killed")`. The registry never decides
   that a run is dead (`stores.py:602-614`). Then `store fingerprint --write` records what is
   there, and retirement proceeds. A fingerprint taken *now* is the right one: it must describe
   what is deleted, which is the current content, not what the run would have recorded.
2. **The store cannot be fingerprinted.** A hot journal is refused by the reader. A missing or
   unreadable shard, or a primary whose `shards` table cannot be read, is refused by
   `_read_closed_store`. This is the one genuinely manual case today, and "manual" here means
   `rm`, the thing this campaign exists to replace. A fingerprint whose `problems` are non-empty is
   still a fingerprint (it counts problems beside the digests, `:845-847`), and does not fall
   into this case.
3. **The retirement itself is interrupted.** The primary's `shards` table is the only list of the
   shard files. If the primary goes first, an interrupted deletion leaves shards that nothing
   names. So the shards go first and the primary last, and the list of files to be deleted is
   written into the tombstone **before** anything is deleted. An interrupted deletion then leaves
   a primary with some shards missing, which the constructor refuses to open (`_check_shard_files`, `:150-152`). The
   tombstone says a deletion was under way, and a second retire completes it.

**2.11 The two pre-existing `var/` observations the first answer made.** Neither is a store, and
both are recorded here because they were found in the same pass.
- **`var/bootstrap-a3-resume.log`** (787 185 bytes, 2026-09-23 10:52) is a strict superset of the
  registered run `handover-03-a3-baseline-resume-20260923T024847`'s `stdout.log` and `stderr.log`.
  It has every line of theirs, and 67 more. Those are the scoped $k$ sample, printed before
  `begin`, and a **SIGTERM trace** at 10:52:13 (`*** SIGTERM received at time=1790157133 ***`).
  **[probed]** It is the launcher's capture of the child's combined output. It is the only record
  on disk that the run was signalled, and it sits outside the run directory, attributed by nothing.
- **That run is recorded `failed`, exit code 1, while the `handover` board says `killed`**
  (`prompts/handover/IMPLEMENTATION_STATE.md:309-313`). It ran at `704a12e`, after `ee46e5c`
  added `terminal_state`, which maps `SystemExit(15)` to `killed`
  (`docs/gktk-remedial/scoped_pipeline_run.py:248-264`). The recorded exit code 1 is the generic
  `except BaseException` branch (`:495-496`). The trace shows the signal arriving inside
  `ray::core::CoreWorker::Wait`, where Ray's own handler has replaced the script's. What reached
  the `exec` frame is not established. One instance.

**2.12 A registry copy carries its source's present-tense claims.** `copy_store` deep-copies the
source's fields, and replaces only `store_id`, `datastore`, `name`, `purpose`, `created`,
`copied_from` and `history` (`stores.py:719-733`). **[checked]** Every unknown field is carried
verbatim, by design (`datastore-portability` README §6.5 point 1). The live A3 sidecar's unknown
fields include:
- `backup`, which says `"retained": true` of a backup of *the live store*;
- `restart.command`, whose `--database` is `var/datastores/handover-A3-baseline-lambdacdm.sqlite`;
- `run_history`, the live store's runs.

`store-fingerprint` prompt 05 Phase C made a registry copy of the live A3 store. Its log lists the
fields that differ, and these are not among them, so the copy carried all three. **[checked]**
After prompt 02 of this campaign, `--prepare` makes the sweep store by `copy_store`, and the new
sweep sidecar will say that it has a retained backup, and that it restarts by running the pipeline
**into the live store**. Nothing reads either field. A person following the copy's `restart` would
write into the live A3 store, which is the 2026-09-23 accident (`2ebb7b6`) by another route. The
recommended `store amend` (README D5) is the remedy for a given copy. Whether copy should drop or
flag such fields is a question about `copy_store`'s design, and it is outside this campaign.

---

## 3. Whether each store may be retired

**The sweep store: yes.**
- Its sidecar declares it "Disposable".
- Every number is in `QUADSOURCE-TOLERANCE-SWEEP.md`, per-case totals to seven figures.
- The user's decision of 2026-09-23 regenerates the A3 store from scratch at `(1e-32, 1e-7)`
  (`handover` board, `:295-313`), so none of its rows is reused.
- Its fingerprint survives in git (`05-fingerprints.json`).
- The four runs naming it name a path, which the tombstone will answer.

**The backup: yes.**
- `store-fingerprint` log 05 compared it with the live A3 store, class by class and tag set by tag
  set, and found **nothing only in the backup**. Every difference is an addition attributed to a
  recorded writer: 48 production-pair records from the registered resume, plus the first sweep
  run's 54 records and 6 tolerance values at non-production pairs.
- Its stated retention condition, "until a resume completes cleanly", can now never be met. The
  store it backs up is to be regenerated, not resumed.
- Its one other role, as the real instance of a primary naming another store's shards, was
  discharged by `datastore-portability` prompt 01, and synthetic tests now cover it.
- Retiring it makes one present-tense claim false, the live sidecar's `backup` block (§1), which
  the campaign must correct through the registry.

**The live A3 store: no.** It is the only comparator until the v2 store exists and is verified.

**Every run directory, and the loose files in `var/runs/`: not stores, and not this campaign's.**
They stay, per `run-registry`'s `[01-var-runs-holds-unattributable-loose-files]`.

---

## 4. Issues

**Opened by this audit** (board: `store-retirement` §3):
- `[00-a-sigterm-pipeline-run-is-recorded-as-failed]` — §2.11, second item.
- `[00-a-launch-log-lives-outside-its-run-directory]` — §2.11, first item.

- `[00-a-copy-carries-its-sources-present-tense-fields]` — §2.12.

**Assigned to this campaign** (from `datastore-portability` §3):
- `[03-atol-sweep-prepare-writes-its-store-sidecar-by-hand]` — its impact is now real (§2.6).
- `[01-atol-sweep-check-expects-absolute-shard-records]` — its impact was understated. It blocks
  `--build --resume` of the A3 v2 store (§2.6), and the same edit discharges it.

**Not opened.** The false claim at `QUADSOURCE-TOLERANCE-SWEEP.md:15-17` is not an issue. It is
the *reason* for one retirement, and it becomes true when that retirement lands.
