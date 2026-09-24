# Log 03 — The registry owns the store sidecar, and copies and moves stores with it

**Prompt:** [`prompts/datastore-portability/03-the-registry-owns-the-store-sidecar.md`](../03-the-registry-owns-the-store-sidecar.md)
**Commit:** *(this commit)* — "Make the registry own the store sidecar and copy and move stores"
**Model:** Claude Opus 5.5
**Date:** 2026-09-24
**Result:** COMPLETE. No §7 stop condition was met. All three real sidecars round-trip
value-identically, and the prescribed order satisfies the P11 property in every layout. Every
*prompt's choice* was kept. Four refusals beyond the prompt's list, and a check of every write
before it is made, were added; the orchestrator should look at them ("Deviations", items 2–6).

## What shipped

**`RunRegistry/stores.py`** (new, 718 lines): the one owner of the store sidecar.

- **The name.** `sidecar_path(primary)` (`:103`) returns `primary.parent / f"{primary.stem}.manifest.json"`.
  It is the only place in the package that spells `.manifest.json` outside a docstring, and
  `test_the_pattern_is_spelled_once` checks that with `ast`. The move's temporary name is
  `<dst sidecar name>.incomplete-move` (`INCOMPLETE_MOVE_SUFFIX`, `_incomplete_move_path`).
- **The reader.** `read_sidecar(primary) -> SidecarReading` (`:268`). It returns `kind`
  (`absent` / `unreadable` / `legacy` / `registry`), `fields`, `problems`, and `legacy_path`.
  `.ok` means a problem-free registry sidecar. `.store_id` is that sidecar's id, and `None`
  otherwise. It opens the sidecar for reading and nothing else, and catches `OSError` and
  `ValueError`, so it never raises on bad content. The registry checks are
  `_registry_problems` (`:210`) and `_history_problems` (`:177`). A legacy `datastore` is read
  by its final component, splitting on `/` and `\` (`_final_component`), which is the same pair of
  separators `Datastore/shard_paths.py` refuses in a shard record. That module is not imported:
  `Datastore/__init__.py` imports `.object`, and so `sqlalchemy`.
- **The writer.** `_write_new_sidecar` (`:341`) refuses if the name or its `.tmp` exists.
  `_update_sidecar` (`:354`) refuses if the file is not an existing regular file or if its
  `.tmp` exists. Both call `_check_before_writing` first, which runs `_registry_problems` on what
  is about to be written (Deviations, item 6). Both then call `write_json_atomic`, looked up in
  the module's namespace at call time, so tests can make it fail. `_update_sidecar` has exactly
  two callers: `adopt_sidecar`, and `move_store` on the temporary file.
- **P10.** `create_sidecar(primary, purpose)` (`:394`) and `adopt_sidecar(primary, purpose=None)`
  (`:432`). Neither opens the primary.
- **P11.** `copy_store(src, dst, purpose, runs_root=None)` (`:639`) and
  `move_store(src, dst, runs_root=None)` (`:678`). The shared refusals are in `_prepare` (`:583`).
  The in-use check is `_refuse_if_in_use` (`:527`) over `runs_naming` (`:493`). `runs_naming` is
  also what `store show` prints. `_store_files` calls `ShardedPool.copy_store` / `move_store` and
  adds the sidecar note to any error. `_failure` names the failed step and lists every file
  beside each end whose name begins with that end's stem. `ShardedPool` is imported inside
  `copy_store` and `move_store` only.
- There is no `os.remove`, `unlink`, `rmtree` or `rmdir` in the module. The only renames are the
  move's two sidecar renames. The only in-place updates are adopt's and the move's temporary file.

**`RunRegistry/__init__.py`**: **P12.** `begin()` computes `results_store_id` with
`read_sidecar(results).store_id`, importing `read_sidecar` inside the function, because `stores`
imports this module. It writes the value to the manifest after `results`. Its docstring's last
paragraph now says the field is the one exception to "a field README §0 would have caught
something with". It is §6.5 point 5, and it matches a run to its store after the store has
moved. The package docstring's first paragraph keeps "schedules nothing, supervises nothing,
locks nothing and deletes nothing", and adds the charter decision: the registry also creates,
adopts, copies and moves the stores it manages, and still deletes nothing.

**`RunRegistry/__main__.py`**: `python -m RunRegistry store {show,create,adopt,copy,move}`.
`show` prints the kind, the problems, whether `datastore` is a legacy path and the name it is
read by, the fields, and every run under `--runs-root` whose `results` or `results_store_id`
names the store, with its state and liveness. It returns 0. The others print `>> <command>:
<sidecar>` and the sidecar's JSON, or `!! <reason>` on stderr and exit 1. `list` is unchanged.

**Tests**: 46 new test methods in three modules, plus a fixtures module.
`test_run_registry.py` and `test_pipeline_adoption.py` are unmodified, and so is everything in
`Datastore/tests/`. No test initialises Ray, calls the `ShardedPool` constructor, or opens anything
under `var/`.

| Module | Methods | Prompt §3 test |
|---|---|---|
| `store_fixtures.py` | — | `StoreTestCase(RegistryTestCase)`: one temporary directory holding the stores and, in `runs/`, the runs root, so one `tree_state` covers both. The A3-shaped and sweep-shaped legacy sidecars are written inline. `assertOpens` (the constructor's read-and-check), `assertDescribes` and `assertProperty` (the P11 property) |
| `test_store_sidecar.py` | 17 | **1** name, reader, each problem, never writes (`TestName` 2, `TestReader` 4) · **3** create (3) · **4** adopt (4) · **2** unknown fields through adopt, copy and move (1) · **9** run manifests (3) |
| `test_store_copy_move.py` | 27 | **5** copy (2) · **6** move in the three layouts (1) · **7** refusals (11) · **8** interruption, one method per table row, every layout a subtest (13) |
| `test_store_command_line.py` | 2 | **10** every `store` subcommand, succeeding and refusing, Ray never initialised; `import RunRegistry` and `list` load neither `ray` nor `sqlalchemy` |

## The sidecar format as shipped, and the prompt's choices

The format is exactly P9's table:

| Field | Required | Written by | As shipped |
|---|---|---|---|
| `sidecar_format` | yes | create, adopt | the integer `1`. `True` and any other value are a problem |
| `store_id` | yes | create, adopt, copy (new); move (kept) | `uuid.uuid4().hex`, checked against `[0-9a-f]{32}` |
| `datastore` | yes | create, adopt, copy, move | the primary's bare file name. A registry value containing `/` or `\` is a problem |
| `name` | yes | create, adopt, copy, move | the primary's stem |
| `purpose` | yes | create; adopt (kept if present); copy (given) | a non-empty string |
| `created` | yes | create, copy; adopt keeps a legacy value | `now_iso()` |
| `copied_from` | no | copy | `{"store_id", "datastore"}`, the parent's id and its primary path at the time. A legacy string, or `null`, is kept verbatim by adopt |
| `history` | yes | every operation appends one | `{"operation", "from", "to", "when", "git_head", "git_dirty"}`. Entry 0 is `create` or `adopt` with `from: null`; later entries are `copy` or `move` with a `from` |

Paths in `copied_from` and `history` are written by `RunRegistry._repo_path`. Nothing reads them to
find a file. Every other field is carried, value-identical.

**Every *prompt's choice* was kept**:

| Choice | Kept? | Note |
|---|---|---|
| Field names from the existing sidecars | kept | `name`, `purpose`, `datastore`, `created`, `copied_from` are the sidecars' own |
| History entries for create and adopt | kept | The reader requires entry 0 to be one of them (Deviations, item 7) |
| Create does not open the primary | kept | `test_it_does_not_open_the_primary` creates a sidecar beside a file that is not a database |
| Adopt refuses a legacy object with `store_id` or `history` | kept | |
| Copy and move never upgrade a legacy sidecar | kept | §4 step 4 |
| Refuse any `running` run, alive or stale | kept | mutation (iv) |
| The move order: store; rename to `.incomplete-move` in the destination directory; update; rename last | kept | mutation (viii) |

## The round-trip check of the three real sidecars

Read-only, with `json.load` alone, before anything else, and again in §4 step 1. For each file,
`json.loads(json.dumps(v, indent=2, sort_keys=True)) == v`. The stricter check also held:
`json.dumps(…, sort_keys=True)` of the original and of the round-tripped value are the same
string, so no `1` became `1.0` and no `True` became `1`.

| Sidecar | Top-level keys | Value-identical | Canonical form identical |
|---|---:|---|---|
| `var/datastores/handover-A3-baseline-lambdacdm.manifest.json` | 14 | yes | yes |
| `var/datastores/handover-atol-sweep.manifest.json` | 5 | yes | yes |
| `var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.manifest.json` | 13 | yes | yes |

## The interruption table

**Reachable** means found at the source or destination names by `read_sidecar`, by the constructor,
or by prompt 02's read-and-check. The property: *after any step, every reachable sidecar either
(a) is a problem-free registry sidecar that describes the store beside it with a complete history,
or (b) is reported by the reader as having a problem; and every reachable store is in one of
prompt 02's states.* Neither exclusion may happen: a problem-free sidecar beside the wrong store or
missing a history entry, or two problem-free sidecars with one `store_id`.

Every row runs in all three layouts (same directory with a new stem; new directory with the same
stem; new directory with a new stem), as subtests of the named test. Every row calls
`assertProperty`, which checks the property at both names. For a problem-free sidecar it checks
the id, the history length, and that the last history entry's `to` is this location. It also
checks that no two problem-free sidecars share an id. "*n*" is the source's history length before
the operation (1 in the tests: `create`). A process that dies inside `write_json_atomic` leaves a
`.tmp` beside the file being written. That is not a sidecar name, so it is not reachable. The
failure message lists it, and a rerun refuses because of it.

**Copy** (the source's sidecar is never opened for writing):

| Row | Process dies after | Source sidecar | Source store | Destination store | Destination sidecar name | Backed by |
|---|---|---|---|---|---|---|
| C0 | any pre-check | (a), *n* | opens | absent | absent | `TestRefusals` (11 tests; every one asserts `tree_state` unchanged) |
| C1 | the pre-checks, before `ShardedPool.copy_store` writes | (a), *n* | opens | absent | absent | `test_C1_the_store_copy_fails_before_writing` |
| C2 | part of `ShardedPool.copy_store` (prompt 02 rows C1–C6) | (a), *n* | opens | shards and no primary: prompt 02's guard | absent | `test_C2_the_store_copy_fails_part_way` (shard #1's `copy2` fails) |
| C3 | the store copied, before the sidecar is written | (a), *n* | opens | **opens** against its own files | absent | `test_C3_the_store_is_copied_and_the_sidecar_write_fails` |
| C4 | the sidecar written to its `.tmp` name, before `os.replace` | (a), *n* | opens | opens | absent; `<dst sidecar>.tmp` holds the complete sidecar | `test_C4_…_and_the_replace_fails` (also: a second copy is refused) |
| C5 | complete | (a), *n*, its own id | opens | opens | (a), *n*+1, a **new** id | `test_C5_complete`, `TestCopy` (2), `TestUnknownFieldsSurvive` |

**Move** (the one sidecar is renamed, never rewritten under a sidecar name):

| Row | Process dies after | Source sidecar name | Source store | Destination store | Destination sidecar name | `.incomplete-move` | Backed by |
|---|---|---|---|---|---|---|---|
| M0 | any pre-check | (a), *n* | opens | absent | absent | absent | `TestRefusals` |
| M1 | the pre-checks, before `ShardedPool.move_store` writes | (a), *n* | opens | absent | absent | absent | `test_M1_the_store_move_fails_before_writing` |
| M2 | part of `ShardedPool.move_store`, before the primary is renamed (prompt 02 M2–M3) | (a), *n*: it describes the store beside it, which **prompt 02's check refuses** because the renamed shards are missing | refused | shards and no primary: guard | absent | absent | `test_M2_…_after_a_shard_is_renamed` |
| M3 | the primary renamed, before its rows are rewritten (prompt 02 M4) | **(b)**: the primary does not exist (orphaned) | absent | prompt 02's M4: refused, or opens against its own files in the same-stem layout | absent | absent | `test_M3_…_after_the_primary_is_renamed` |
| M4 | the store moved, before the sidecar is renamed | **(b)**, orphaned | absent | opens | absent | absent | `test_M4_…_the_rename_to_the_temporary_name_fails` |
| M5 | the sidecar renamed to `.incomplete-move`, before the update | absent | absent | opens | absent | the un-updated sidecar, not reachable | `test_M5_…_and_the_update_fails` |
| M6 | the update written to `<.incomplete-move>.tmp`, before its `os.replace` | absent | absent | opens | absent | un-updated, beside the updated `.tmp` | `test_M6_…_and_the_replace_fails` |
| M7 | the update, before the final rename | absent | absent | opens | absent | the updated sidecar, *n*+1 entries, same id | `test_M7_…_the_final_rename_fails` (also: a person's `mv` completes it) |
| M8 | complete | absent | absent | opens | (a), *n*+1, the **same** id; no other sidecar, `.tmp` or `.incomplete-move` anywhere | absent | `test_M8_complete`, `TestMove.test_in_each_layout` |

**The two exclusions, against the code.**

- **A problem-free sidecar beside the wrong store, or with a missing history entry.** A copy's
  destination sidecar is written once, complete, after its store is in place (C3–C5). A move's
  sidecar is never under a sidecar name while it is un-updated. It leaves the source name only by
  a rename to `.incomplete-move`, and reaches the destination name only after the update (M5–M8).
  The one problem-free sidecar during a move sits beside a store that prompt 02 refuses (M2), and
  it is that store's own sidecar. Mutation (viii) is the counter-example the temporary name
  prevents. Under it, in the "new directory, same stem" layout at row M5, the un-updated sidecar
  sits at the destination name and passes every check the reader makes (`datastore` and `name`
  are unchanged and correct there). It still lacks its `move` entry, and `assertProperty` catches
  it by the history.
- **Two problem-free sidecars with one `store_id` after a move.** A move writes no second file.
  The only file carrying the id is renamed twice, so it is never at two names at once. A copy's
  destination always gets a new id: mutation (i).

**What M5–M7 leave.** The store is at the destination with no reachable sidecar, and
`begin(results=…)` records `null` for it. The failure message names the `.incomplete-move` file.
`store create` and `store adopt` refuse at that name (Deviations, item 4), so a person does not
fork the identity by giving the store a new one. The remedy, which the error implies and
`test_M7` exercises, is for a person to rename the file after M7. After M5 or M6, they rename it
and then re-run the update by hand or with adopt. That is for a person; nothing cleans up.

## The deliberate-breakage record

The final tree was staged. Each mutation was applied to the working tree, and its diff was taken
with `git diff` against the index, which is this commit's content. Each was checked with
`git apply --check -R`. Then the whole `RunRegistry/tests` suite ran
(`PYTHONPATH=. ./venv/bin/python -m unittest discover -s RunRegistry/tests -t .`), and the file
was restored with `git checkout --`. The script is scratch `agent03/a03_mutate.py`. Unmutated:
`Ran 84 tests … OK`. Each diff below replays with `git apply` against this commit. Failures are
counted per subtest, and the tests are listed by method.

**(i) copy keeps the source's `store_id`.** `Ran 84 tests FAILED (failures=5)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 357643f..b3d96b1 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -657,7 +657,6 @@ def copy_store(src, dst, purpose, runs_root=None) -> dict:
     step = "write the destination's sidecar"
     try:
         fields = _copy.deepcopy(reading.fields)
-        fields["store_id"] = new_store_id()
         fields["datastore"] = dst.name
         fields["name"] = dst.stem
         fields["purpose"] = purpose
```

Failed:

- FAIL `test_store_copy_move.TestCopy.test_a_copy_of_a_copy_names_its_immediate_parent`
- FAIL `test_store_copy_move.TestCopy.test_the_destination_is_a_new_store_and_the_source_is_untouched`
- FAIL `test_store_copy_move.TestInterruption.test_C5_complete [same directory, new stem]`
- FAIL `test_store_copy_move.TestInterruption.test_C5_complete [new directory, new stem]`
- FAIL `test_store_copy_move.TestInterruption.test_C5_complete [new directory, same stem]`

The copy's sidecar carries the source's id. Every test that compares the two ids fails, including all three layouts of row C5. `TestUnknownFieldsSurvive` still passes, correctly: it checks unknown fields, not identity.

**(ii) move assigns a new `store_id`.** `Ran 84 tests FAILED (failures=9)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 357643f..1e793e3 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -703,6 +703,7 @@ def move_store(src, dst, runs_root=None) -> dict:
                 f'"{pending}" no longer holds what was read from "{reading.path}" before the move'
             )
         fields = _copy.deepcopy(reading.fields)
+        fields["store_id"] = new_store_id()
         fields["datastore"] = dst.name
         fields["name"] = dst.stem
         fields["history"] = list(fields["history"]) + [
```

Failed:

- FAIL `test_store_copy_move.TestInterruption.test_M7_the_sidecar_is_updated_and_the_final_rename_fails [new directory, same stem]`
- FAIL `test_store_copy_move.TestInterruption.test_M7_the_sidecar_is_updated_and_the_final_rename_fails [new directory, new stem]`
- FAIL `test_store_copy_move.TestInterruption.test_M7_the_sidecar_is_updated_and_the_final_rename_fails [same directory, new stem]`
- FAIL `test_store_copy_move.TestInterruption.test_M8_complete [new directory, new stem]`
- FAIL `test_store_copy_move.TestInterruption.test_M8_complete [same directory, new stem]`
- FAIL `test_store_copy_move.TestInterruption.test_M8_complete [new directory, same stem]`
- FAIL `test_store_copy_move.TestMove.test_in_each_layout [new directory, new stem]`
- FAIL `test_store_copy_move.TestMove.test_in_each_layout [same directory, new stem]`
- FAIL `test_store_copy_move.TestMove.test_in_each_layout [new directory, same stem]`

The moved sidecar gets a new id. `TestMove` and row M8 fail in all three layouts. So does M7, whose updated temporary file must carry the old id.

**(iii) the writer keeps only the known fields.** `Ran 84 tests FAILED (failures=5)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 357643f..e4f2e68 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -348,7 +348,7 @@ def _write_new_sidecar(path: Path, fields: dict, primary: Path) -> None:
             f'refusing to write a new sidecar at "{path}": {", ".join(taken)} already exist, '
             f"and nothing is ever overwritten"
         )
-    write_json_atomic(str(path), fields)
+    write_json_atomic(str(path), {k: v for k, v in fields.items() if k in KNOWN_FIELDS})
 
 
 def _update_sidecar(path: Path, fields: dict, primary: Path) -> None:
@@ -363,7 +363,7 @@ def _update_sidecar(path: Path, fields: dict, primary: Path) -> None:
         raise RuntimeError(
             f'refusing to update the sidecar "{path}": "{_tmp_path(path)}" already exists'
         )
-    write_json_atomic(str(path), fields)
+    write_json_atomic(str(path), {k: v for k, v in fields.items() if k in KNOWN_FIELDS})
 
 
 def _history_entry(operation, source, destination, provenance) -> dict:
```

Failed:

- FAIL `test_store_sidecar.TestAdopt.test_an_a3_shaped_sidecar_keeps_every_unknown_field`
- FAIL `test_store_sidecar.TestUnknownFieldsSurvive.test_an_a3_shaped_sidecar_through_adopt_copy_and_move [adopt]`
- FAIL `test_store_sidecar.TestUnknownFieldsSurvive.test_an_a3_shaped_sidecar_through_adopt_copy_and_move`
- FAIL `test_store_sidecar.TestUnknownFieldsSurvive.test_an_a3_shaped_sidecar_through_adopt_copy_and_move [move]`
- FAIL `test_store_sidecar.TestUnknownFieldsSurvive.test_an_a3_shaped_sidecar_through_adopt_copy_and_move [copy]`

Only the tests that check unknown fields catch this, which is why the A3-shaped test exists. Every other test passes, because it checks only known fields. The failures are the A3-shaped adopt, and each of the adopt, copy and move steps.

**(iv) the running-run check considers only runs whose liveness is `alive`.** `Ran 84 tests FAILED (failures=2)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 357643f..775cda3 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -538,7 +538,7 @@ def _refuse_if_in_use(operation, src, dst, store_id, runs_root) -> None:
     running = [
         entry
         for entry in runs_naming([src, dst], store_id, runs_root=root)
-        if entry["state"] == "running"
+        if entry["liveness"] == "alive"
     ]
     if running:
         described = "; ".join(
```

Failed:

- FAIL `test_store_copy_move.TestRefusals.test_a_running_run_that_is_stale_still_refuses [move]`
- FAIL `test_store_copy_move.TestRefusals.test_a_running_run_that_is_stale_still_refuses [copy]`

Only the stale-run test fails, for copy and for move. The alive-run test still refuses. This mutation is exactly the check that looks only at `alive` runs.

**(v) the running-run check ignores `results_store_id`.** `Ran 84 tests FAILED (failures=2)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 357643f..14a1006 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -537,7 +537,7 @@ def _refuse_if_in_use(operation, src, dst, store_id, runs_root) -> None:
         )
     running = [
         entry
-        for entry in runs_naming([src, dst], store_id, runs_root=root)
+        for entry in runs_naming([src, dst], None, runs_root=root)
         if entry["state"] == "running"
     ]
     if running:
```

Failed:

- FAIL `test_store_copy_move.TestRefusals.test_a_match_by_results_store_id_alone [copy]`
- FAIL `test_store_copy_move.TestRefusals.test_a_match_by_results_store_id_alone [move]`

Only the id-alone test fails, for copy and for move. There the run's `results` names the store's old directory, which was moved by hand, so the path match finds nothing. Every path-matched refusal still passes.

**(vi) `datastore` is written as a repository path, not a bare name.** `Ran 84 tests FAILED (failures=2, errors=67)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 357643f..1560706 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -419,7 +419,7 @@ def create_sidecar(primary, purpose) -> dict:
     fields = {
         "sidecar_format": SIDECAR_FORMAT,
         "store_id": new_store_id(),
-        "datastore": primary.name,
+        "datastore": _repo_path(primary),
         "name": primary.stem,
         "purpose": purpose,
         "created": now_iso(),
@@ -476,7 +476,7 @@ def adopt_sidecar(primary, purpose=None) -> dict:
     fields = _copy.deepcopy(reading.fields)
     fields["sidecar_format"] = SIDECAR_FORMAT
     fields["store_id"] = new_store_id()
-    fields["datastore"] = primary.name
+    fields["datastore"] = _repo_path(primary)
     fields["name"] = primary.stem
     fields["purpose"] = purpose
     if not _nonempty_string(fields.get("created")):
@@ -658,7 +658,7 @@ def copy_store(src, dst, purpose, runs_root=None) -> dict:
     try:
         fields = _copy.deepcopy(reading.fields)
         fields["store_id"] = new_store_id()
-        fields["datastore"] = dst.name
+        fields["datastore"] = _repo_path(dst)
         fields["name"] = dst.stem
         fields["purpose"] = purpose
         fields["created"] = now_iso()
@@ -703,7 +703,7 @@ def move_store(src, dst, runs_root=None) -> dict:
                 f'"{pending}" no longer holds what was read from "{reading.path}" before the move'
             )
         fields = _copy.deepcopy(reading.fields)
-        fields["datastore"] = dst.name
+        fields["datastore"] = _repo_path(dst)
         fields["name"] = dst.stem
         fields["history"] = list(fields["history"]) + [
             _history_entry("move", src, dst, git_provenance())
```

Failed or errored (40 methods): `test_store_command_line.TestStoreCommands.test_show_create_adopt_copy_and_move`, `test_store_copy_move.TestCopy.test_a_copy_of_a_copy_names_its_immediate_parent`, `test_store_copy_move.TestCopy.test_the_destination_is_a_new_store_and_the_source_is_untouched`, `test_store_copy_move.TestInterruption.test_C1_the_store_copy_fails_before_writing`, `test_store_copy_move.TestInterruption.test_C2_the_store_copy_fails_part_way`, `test_store_copy_move.TestInterruption.test_C3_the_store_is_copied_and_the_sidecar_write_fails`, `test_store_copy_move.TestInterruption.test_C4_the_sidecar_is_written_to_its_tmp_name_and_the_replace_fails`, `test_store_copy_move.TestInterruption.test_C5_complete`, `test_store_copy_move.TestInterruption.test_M1_the_store_move_fails_before_writing`, `test_store_copy_move.TestInterruption.test_M2_the_store_move_fails_after_a_shard_is_renamed`, `test_store_copy_move.TestInterruption.test_M3_the_store_move_fails_after_the_primary_is_renamed`, `test_store_copy_move.TestInterruption.test_M4_the_store_is_moved_and_the_rename_to_the_temporary_name_fails`, `test_store_copy_move.TestInterruption.test_M5_the_sidecar_is_at_its_temporary_name_and_the_update_fails`, `test_store_copy_move.TestInterruption.test_M6_the_update_is_written_to_its_tmp_name_and_the_replace_fails`, `test_store_copy_move.TestInterruption.test_M7_the_sidecar_is_updated_and_the_final_rename_fails`, `test_store_copy_move.TestInterruption.test_M8_complete`, `test_store_copy_move.TestMove.test_in_each_layout`, `test_store_copy_move.TestRefusals.test_a_copy_needs_its_own_purpose`, `test_store_copy_move.TestRefusals.test_a_finished_run_does_not_refuse`, `test_store_copy_move.TestRefusals.test_a_match_by_results_store_id_alone`, `test_store_copy_move.TestRefusals.test_a_run_begun_before_the_sidecar_is_matched_by_its_path`, `test_store_copy_move.TestRefusals.test_a_running_run_that_is_alive_names_the_source_by_path`, `test_store_copy_move.TestRefusals.test_a_running_run_that_is_stale_still_refuses`, `test_store_copy_move.TestRefusals.test_a_running_run_that_names_the_destination`, `test_store_copy_move.TestRefusals.test_a_runs_root_that_is_not_there`, `test_store_copy_move.TestRefusals.test_a_source_sidecar_that_is_not_a_problem_free_registry_sidecar`, `test_store_copy_move.TestRefusals.test_prompt_02s_refusals_pass_through`, `test_store_copy_move.TestRefusals.test_the_destination_sidecar_names_exist`, `test_store_sidecar.TestAdopt.test_a_purpose_is_taken_only_when_the_sidecar_has_none`, `test_store_sidecar.TestAdopt.test_a_sweep_shaped_legacy_sidecar`, `test_store_sidecar.TestAdopt.test_an_a3_shaped_sidecar_keeps_every_unknown_field`, `test_store_sidecar.TestAdopt.test_each_refusal`, `test_store_sidecar.TestCreate.test_it_does_not_open_the_primary`, `test_store_sidecar.TestCreate.test_the_fields_and_one_create_entry`, `test_store_sidecar.TestReader.test_it_classifies_absent_unreadable_legacy_and_registry`, `test_store_sidecar.TestReader.test_it_reports_each_problem`, `test_store_sidecar.TestRunManifests.test_a_manifest_without_the_field_lists_and_is_checked_by_path`, `test_store_sidecar.TestRunManifests.test_null_for_no_store_no_sidecar_legacy_and_a_problem`, `test_store_sidecar.TestRunManifests.test_the_store_id_of_a_registry_sidecar`, `test_store_sidecar.TestUnknownFieldsSurvive.test_an_a3_shaped_sidecar_through_adopt_copy_and_move`.

`_check_before_writing` refuses every write the mutation makes (`datastore '…/B/store.sqlite' is a path; a registry sidecar names its store by bare file name, and the registry never writes one`). So every test that creates, adopts, copies or moves a sidecar errors: 67 errors and 2 failures, across 40 test methods. Without that check, the reader would still report the written sidecar as having a problem, and the same tests would fail one step later.

**(vii) the reader reads a legacy `datastore` as a path, not by name.** `Ran 84 tests FAILED (failures=3, errors=3)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 357643f..00b121d 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -312,9 +312,9 @@ def read_sidecar(primary) -> SidecarReading:
         if not _nonempty_string(value):
             problems.append(f"datastore {value!r} is not a file name or path")
         else:
-            named = _final_component(value)
-            legacy_path = named != value
-            if named != primary.name:
+            named = os.path.abspath(_resolve(value))
+            legacy_path = False
+            if named != str(primary):
                 problems.append(
                     f'datastore {value!r} names "{named}", not the primary beside it, "{primary.name}"'
                 )
```

Failed:

- FAIL `test_store_command_line.TestStoreCommands.test_show_create_adopt_copy_and_move`
- ERROR `test_store_sidecar.TestAdopt.test_a_sweep_shaped_legacy_sidecar`
- ERROR `test_store_sidecar.TestAdopt.test_an_a3_shaped_sidecar_keeps_every_unknown_field`
- FAIL `test_store_sidecar.TestReader.test_a_legacy_datastore_path_is_read_by_its_name`
- FAIL `test_store_sidecar.TestReader.test_it_reports_each_problem [legacy datastore names another file]`
- ERROR `test_store_sidecar.TestUnknownFieldsSurvive.test_an_a3_shaped_sidecar_through_adopt_copy_and_move`

The sweep-shaped and A3-shaped legacy sidecars, whose `datastore` names another directory, now read as naming that directory, so the reader reports a problem. The reader test, adopt, the unknown-fields test, `store show` (whose `legacy path, read by its name` line is gone) and the problem test's message check all fail.

**(viii) move renames the sidecar straight to its final name and updates it there.** `Ran 84 tests FAILED (failures=12)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 357643f..b30628c 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -691,7 +691,7 @@ def move_store(src, dst, runs_root=None) -> dict:
 
     _store_files("move", ShardedPool.move_store, src, dst)
 
-    pending = _incomplete_move_path(dst_sidecar)
+    pending = dst_sidecar
     step = "rename the source's sidecar to its temporary name"
     try:
         _no_overwrite(pending)
@@ -710,9 +710,6 @@ def move_store(src, dst, runs_root=None) -> dict:
         ]
         _update_sidecar(pending, fields, dst)
 
-        step = "rename the temporary sidecar to the destination's sidecar name"
-        _no_overwrite(dst_sidecar)
-        os.rename(pending, dst_sidecar)
     except Exception as e:
         raise _failure("move", src, dst, step, e) from e
     return fields
```

Failed:

- FAIL `test_store_copy_move.TestInterruption.test_M4_the_store_is_moved_and_the_rename_to_the_temporary_name_fails [same directory, new stem]`
- FAIL `test_store_copy_move.TestInterruption.test_M4_the_store_is_moved_and_the_rename_to_the_temporary_name_fails [new directory, new stem]`
- FAIL `test_store_copy_move.TestInterruption.test_M4_the_store_is_moved_and_the_rename_to_the_temporary_name_fails [new directory, same stem]`
- FAIL `test_store_copy_move.TestInterruption.test_M5_the_sidecar_is_at_its_temporary_name_and_the_update_fails [same directory, new stem]`
- FAIL `test_store_copy_move.TestInterruption.test_M5_the_sidecar_is_at_its_temporary_name_and_the_update_fails [new directory, new stem]`
- FAIL `test_store_copy_move.TestInterruption.test_M5_the_sidecar_is_at_its_temporary_name_and_the_update_fails [new directory, same stem]`
- FAIL `test_store_copy_move.TestInterruption.test_M6_the_update_is_written_to_its_tmp_name_and_the_replace_fails [new directory, new stem]`
- FAIL `test_store_copy_move.TestInterruption.test_M6_the_update_is_written_to_its_tmp_name_and_the_replace_fails [new directory, same stem]`
- FAIL `test_store_copy_move.TestInterruption.test_M6_the_update_is_written_to_its_tmp_name_and_the_replace_fails [same directory, new stem]`
- FAIL `test_store_copy_move.TestInterruption.test_M7_the_sidecar_is_updated_and_the_final_rename_fails [new directory, same stem]`
- FAIL `test_store_copy_move.TestInterruption.test_M7_the_sidecar_is_updated_and_the_final_rename_fails [same directory, new stem]`
- FAIL `test_store_copy_move.TestInterruption.test_M7_the_sidecar_is_updated_and_the_final_rename_fails [new directory, new stem]`

No happy-path test fails, because a completed move ends in the same state. The interruption rows M4–M7 fail in all three layouts. At M4, the rename now goes straight to the destination name, so the injected failure, which matches only the temporary name, never fires. At M5 and M6, the un-updated sidecar sits at the destination name. In the "new directory, same stem" layout it has no reader problem, yet it lacks its `move` entry, and `assertProperty` fails on the history. At M7 there is no final rename left to fail.

**(ix) `begin()` does not record `results_store_id`.** `Ran 84 tests FAILED (failures=3)`.

```diff
diff --git a/RunRegistry/__init__.py b/RunRegistry/__init__.py
index 8445c86..d0de009 100644
--- a/RunRegistry/__init__.py
+++ b/RunRegistry/__init__.py
@@ -510,7 +510,7 @@ def begin(
         "expected_units": expected_units,
         "checkpoint": _repo_path(checkpoint) if checkpoint else None,
         "results": _repo_path(results) if results else None,
-        "results_store_id": results_store_id,
+        "results_store_id": None,
         "scope": scope,
         "heartbeat_means": heartbeat_means,
     }
```

Failed:

- FAIL `test_store_command_line.TestStoreCommands.test_show_create_adopt_copy_and_move`
- FAIL `test_store_copy_move.TestRefusals.test_a_match_by_results_store_id_alone`
- FAIL `test_store_sidecar.TestRunManifests.test_the_store_id_of_a_registry_sidecar`

The field is still written, as `null`. The run-manifest test fails, and so do the id-alone refusal and the command line's `by results and results_store_id` line.

## The real-store demonstration (§4)

All work was in `var/portability-check-03/`, with `--runs-root var/portability-check-03/runs` on
every `store copy` / `store move`, and `root=` that directory in the throwaway `begin()`. **No new
code, `ShardedPool` or `main.py` was pointed at an original, the backup or their sidecars.**
Nothing was written under `var/runs/` or `var/datastores/`. The code that ran was this commit's
production code, from the working tree before it was committed: the history entries record
`git_head` `266a70f…` with `git_dirty: true`. The only change made after the demonstration is
test-only (Deviations, item 12).

**1. Before.** `python -m RunRegistry list`: 7 runs, 5 finished and 2 unknown, none `running`.
Free space: 3.3 GB. I took a read-only snapshot with scratch `a03_snapshot.py`, using only
`sqlite3` `mode=ro`, SHA-256 and `os.stat`. It covered, for the three stores, each file's
`st_mtime_ns` and size, per-table row counts in every file, the primaries' SHA-256 and `shards`
rows, and any `-journal` / `-wal` / `-shm` file (none). For **the three sidecars** it took
SHA-256, `st_mtime_ns` and size. For **`var/runs/`** it took every entry (39: directories and
files) with mtime and size, and the SHA-256 of every file under 50 MB, which includes all 5
`manifest.json` and 5 `status.json`. The round-trip check above was repeated and held.

**2. The hand-made source.** `cp -p` of `var/datastores/handover-atol-sweep.sqlite`, its four shards
and **its sidecar** into `var/portability-check-03/src/`, under their own names. The primary's
rows are the four absolute paths into `var/datastores/`. The sidecar is legacy, with
`"datastore": "var/datastores/handover-atol-sweep.sqlite"` and `"copied_from":
"var/datastores/handover-A3-baseline-lambdacdm.sqlite"`: the backup's defect, reproduced on a copy.
Then, with `sqlite3` on **the copy's shard 0 only**, I ran `DELETE FROM QuadSourceIntegral WHERE
serial = 329386`. That is the highest serial in that shard, the same row prompt 02 deleted, with
label `handover-atol-sweep-a1e-32-r1e-08-QuadSourceIntegral-k3.05e+07-q3.05e+07-r3.05e+07-zresponse0.1-2026-09-24T00:17:34`.
The shard's `QuadSourceIntegral` count went from **1,955 to 1,954**. Then
`var/portability-check-03/a3probe/` got a `cp -p` of the **A3 sidecar** beside an empty placeholder
`handover-A3-baseline-lambdacdm.sqlite` (`: >`).

**3. `store show` on the source** (exit 0):

```
kind:     legacy
problems: none
datastore: 'var/datastores/handover-atol-sweep.sqlite' is a legacy path, read by its name 'handover-atol-sweep.sqlite', never as a path
runs naming this store, under var/portability-check-03/runs: none
```

**4. `store copy` before adoption** (exit 1):

```
!! Cannot copy ".../portability-check-03/src/handover-atol-sweep.sqlite": its sidecar ".../src/handover-atol-sweep.manifest.json"
   is not a problem-free registry sidecar (it is a legacy sidecar). Run `python -m RunRegistry store create` or `store adopt`
   on it first; copy never upgrades a sidecar implicitly. Nothing was written
```

`src/`'s six files had the same mtime_ns, size and SHA-256 before and after (`cmp` of the two
listings), and no `dst/` was created.

**5. Adopt.** The source's sidecar went from the five legacy fields to:

```json
{
  "copied_from": "var/datastores/handover-A3-baseline-lambdacdm.sqlite",
  "created": "2026-09-23T12:41:04+0100",
  "datastore": "handover-atol-sweep.sqlite",
  "history": [{"from": null, "git_dirty": true, "git_head": "266a70f2de7d3ea647b86e1a8ff46724e2f3ea71",
               "operation": "adopt", "to": "var/portability-check-03/src/handover-atol-sweep.sqlite",
               "when": "2026-09-24T19:03:19+01:00"}],
  "name": "handover-atol-sweep",
  "purpose": "Working copy of the A3 baseline store for docs/handover/quadsource_atol_sweep.py. Disposable: …",
  "sidecar_format": 1,
  "store_id": "ee50b83932c44df5b52587219ba801f1"
}
```

`datastore` became bare, `copied_from` stayed the legacy string verbatim, `created` and
`purpose` were kept, and there is one `adopt` entry.

The probe: before adoption, `store show` read it as legacy, with a legacy path read by its name
`handover-A3-baseline-lambdacdm.sqlite` and no problems. After `store adopt` (exit 0, no
`--purpose`, since the sidecar has one), it had `sidecar_format` 1, `store_id`
`99565c23af5b4e06ac90213f97eec08a`, one `adopt` entry, `datastore`
`handover-A3-baseline-lambdacdm.sqlite`, and no `copied_from`. `created`
(`2026-09-20T21:02:30+01:00`), `purpose` and `name` equal the real sidecar's. I compared, with
`json.load` on the real A3 sidecar (read-only) and on the adopted probe, the fields outside
`KNOWN_FIELDS`. Those are the ten unknown fields `backup`, `driver`, `git_dirty`, `git_head`,
`grid_criterion`, `note`, `restart`, `run_history`, `scope` and `status_files`. They are
**value-identical, and identical in canonical form** (`json.dumps(…, sort_keys=True)`). That
covers the nested `run_history` entries with their lists, `restart`, `backup` and `status_files`.

**6. In use.** Scratch `a03_begin.py` called `RunRegistry.begin(campaign="datastore-portability",
prompt="03", slug="demo-in-use", results="var/portability-check-03/src/handover-atol-sweep.sqlite",
root=<repo>/var/portability-check-03/runs, …)` and exited without finishing. Its manifest had
`"results_store_id": "ee50b83932c44df5b52587219ba801f1"`, **equal to the source's `store_id`**. The
status was `running`, pid 22932, gone. `list --root var/portability-check-03/runs` showed it `!!`
stale. `store copy` then exited 1:

```
!! Cannot copy ".../src/handover-atol-sweep.sqlite" to ".../dst/pcopy.sqlite": run datastore-portability-03-demo-in-use-20260924T190330
   is running (stale, pid 22932) and names this store by its results and results_store_id. A stale run is ended by a person,
   with Run.finish, before its store is moved; the registry does not decide that a run is dead. Nothing was written
```

Scratch `a03_finish.py` then called `Run(<that directory>).finish("killed")`. `store show` then
listed `datastore-portability-03-demo-in-use-20260924T190330  killed    finished  by results and results_store_id`.

**7. `store copy`** to `var/portability-check-03/dst/pcopy.sqlite`, with a purpose, exit 0 in
3.8 s. `ShardedPool` printed its `!!` legacy-record notice for the source.

- `dst/` held six files: `pcopy.sqlite`, `pcopy-shard000{0..3}.sqlite`, `pcopy.manifest.json`.
  There was no `.tmp` and no `.incomplete-copy`.
- Its sidecar: `store_id` **`d4d466f5cdce4dcf91c92fb2a733dff1`** (new).
  `copied_from = {"datastore": "var/portability-check-03/src/handover-atol-sweep.sqlite", "store_id": "ee50b83932c44df5b52587219ba801f1"}`.
  History: two entries, `adopt` (`from` null, `to` `…/src/handover-atol-sweep.sqlite`) and `copy`
  (`from` `…/src/handover-atol-sweep.sqlite`, `to` `…/dst/pcopy.sqlite`, `when`
  `2026-09-24T19:03:49+01:00`). `datastore` is `pcopy.sqlite` and `name` is `pcopy`.
- Its `shards` rows: `(0, 'pcopy-shard0000.sqlite')`, `(1, 'pcopy-shard0001.sqlite')`,
  `(2, 'pcopy-shard0002.sqlite')`, `(3, 'pcopy-shard0003.sqlite')`.
- The source's six files, sidecar included, had the same mtime_ns, size and SHA-256 before and
  after (`cmp` of the two listings).

**8. `main.py --database var/portability-check-03/dst/pcopy.sqlite --inventory --no-prune-unvalidated
--shards 4 --ray-address local`**, run as a script with `PYTHONPATH=.`: exit 0 in 12 s,
`>> Opened existing sharded datastore ".../dst/pcopy.sqlite" with 4 shards`, and no `!!` line. The
counts are in the table under step 9.

**9. `store move`** to `var/portability-check-03/moved/pmoved.sqlite`, exit 0.

- `store_id` **`d4d466f5cdce4dcf91c92fb2a733dff1`**, the same as `pcopy`'s.
- A three-entry history: `adopt` → `copy` → `move` (`from` `…/dst/pcopy.sqlite`, `to`
  `…/moved/pmoved.sqlite`).
- `datastore` `pmoved.sqlite`, `name` `pmoved`. `copied_from` is unchanged, naming `ee50b839…`.
- Rows `pmoved-shard000{0..3}.sqlite`. `moved/` held six files.
- **`dst/` held no file at all** (`ls -A` empty): no sidecar, no `.tmp`, no `.incomplete-move`.
  The directory itself remains, since nothing is deleted.

The same `main.py --inventory` on `pmoved.sqlite` exited 0 (77 s, mostly Ray start-up). The two
inventory bodies are identical line for line (`diff`, after the first two lines, which name the
path). Original: step 1's snapshot of the sweep store, replicated tables from shard 0 (all four
shards agree), sharded tables summed.

| Table | Kind | Original | Inventory of `pcopy` | Inventory of `pmoved` |
|---|---|---:|---:|---:|
| version | replicated | 1 | 1 | 1 |
| store_tag | replicated | 10 | 10 | 10 |
| redshift | replicated | 1,740 | 1,740 | 1,740 |
| wavenumber | replicated | 8 | 8 | 8 |
| wavenumber_exit_time | replicated | 8 | 8 | 8 |
| tolerance | replicated | 13 | 13 | 13 |
| LambdaCDM | replicated | 1 | 1 | 1 |
| QCD_Cosmology | replicated | 1 | 1 | 1 |
| IntegrationSolver | replicated | 7 | 7 | 7 |
| BackgroundModel | replicated | 1 | 1 | 1 |
| BackgroundModelValue | replicated | 1,740 | 1,740 | 1,740 |
| GkSourcePolicy | replicated | 2 | 2 | 2 |
| QuadSourcePolicy | replicated | 2 | 2 | 2 |
| TkNumericIntegration | sharded | 8 | 8 | 8 |
| TkNumericValue | sharded | 3,865 | 3,865 | 3,865 |
| TkWKBIntegration | sharded | 8 | 8 | 8 |
| TkWKBValue | sharded | 9,627 | 9,627 | 9,627 |
| QuadSource | sharded | 36 | 36 | 36 |
| QuadSourceValue | sharded | 17,178 | 17,178 | 17,178 |
| GkNumericIntegration | sharded | 4,549 | 4,549 | 4,549 |
| GkNumericValue | sharded | 146,445 | 146,445 | 146,445 |
| GkWKBIntegration | sharded | 13,920 | 13,920 | 13,920 |
| GkWKBValue | sharded | 916,937 | 916,937 | 916,937 |
| GkSource | sharded | 1,160 | 1,160 | 1,160 |
| GkSourceValue | sharded | 1,016,160 | 1,016,160 | 1,016,160 |
| GkSourcePolicyData | sharded | 1,160 | 1,160 | 1,160 |
| **QuadSourceIntegral** | sharded | **7,706** | **7,705** | **7,705** |
| OneLoopIntegral | sharded | 0 | 0 | 0 |

**27 of 28 tables equal the original's, and `QuadSourceIntegral` is exactly one lower in both.**
That row shows which *store files* each inventory read. The ids and histories show which
*sidecar* went where: `pcopy`'s id differs from the adopted source's, and `pmoved`'s equals
`pcopy`'s. The histories have one entry at the source, two at `pcopy` and three at `pmoved`.

**10. Originals untouched.** I re-took step 1's snapshot, and `cmp` with the first gave
**identical**. That covers the 15 store files, the three sidecars, the 39 `var/runs/` entries
with their hashes, and no journal files before or after. `python -m RunRegistry list` still
showed 7 runs, 5 finished and 2 unknown.

| File | mtime (before = after) | Size | Rows, all tables | SHA-256 |
|---|---|---:|---:|---|
| `handover-atol-sweep.sqlite` | 2026-09-23T12:41:04 | 32,768 | 41 | `c3cda49d8e35…` |
| `handover-atol-sweep-shard0000…3` | 2026-09-24T02:58:24 / 09-23T23:00:42 / 02:58:24 / 02:58:24 | 87,482,368 / 87,334,912 / 87,814,144 / 86,994,944 | 596,542 / 593,534 / 590,516 / 599,069 | |
| `handover-A3-baseline-lambdacdm.sqlite` | 2026-09-20T21:03:10 | 32,768 | 41 | `fdbe93a54108…` |
| `handover-A3-baseline-lambdacdm-shard0000…3` | 2026-09-23T11:22:04 / 11:11:00 / 11:22:04 / 11:22:04 | 87,367,680 / 87,302,144 / 87,711,744 / 86,892,544 | 596,351 / 593,493 / 590,385 / 598,908 | |
| backup primary | 2026-09-20T21:03:10 | 32,768 | 41 | `fdbe93a54108…` |
| backup `-shard0000…3` | 2026-09-21T04:32:10 / 04:32:24 ×3 | 87,044,096 / 87,302,144 / 87,588,864 / 86,609,920 | 595,885 / 593,487 / 590,199 / 598,522 | |
| `handover-A3-baseline-lambdacdm.manifest.json` | 2026-09-21T09:13:49 | 4,110 | | `3b519690a57b…` |
| `handover-atol-sweep.manifest.json` | 2026-09-23T12:41:04 | 481 | | `dad4fabc7cf4…` |
| backup `handover-A3-baseline-lambdacdm.manifest.json` | 2026-09-21T07:38:04 | 2,636 | | `1e9c11da680e…` |

**11. Deleted.** `rm -rf var/portability-check-03`. `var/` holds `.DS_Store`,
`bootstrap-a3-resume.log`, `datastores/` and `runs/`, as before. Nothing from §4 is in the commit.
The throwaway scripts and outputs are in session scratch space (`agent03/a03_*`).

## Deviations from the prompt

1. **`store show` takes `--runs-root DIR`**, defaulting to `var/runs/`. Class: **STRUCTURALLY
   REQUIRED.** §2 gives `store show PRIMARY` with no option, but `show` must print the runs that
   name the store. §4 step 6 needs it to list a run under `var/portability-check-03/runs`, and
   nothing may be written under `var/runs/`.
2. **Copy and move refuse when the runs root is not a directory.** Class: **IMPLEMENTATION
   CHOICE.** `list_runs` returns nothing for a missing root. So a mistyped `--runs-root`, or a
   fresh clone with no `var/runs/`, would pass the in-use check without looking at anything. The
   refusal says the check cannot be made. `test_a_runs_root_that_is_not_there`.
3. **A move also refuses if `<dst sidecar>.incomplete-move.tmp` exists.** Class: **IMPLEMENTATION
   CHOICE.** `_update_sidecar` refuses that name, which is correct. But it would refuse at M5,
   after the store had already moved. Checking it with the other destination names moves the
   refusal before any write.
4. **Create and adopt refuse where `<sidecar>.incomplete-move` exists.** Class: **IMPLEMENTATION
   CHOICE.** After M5–M7 that file carries the store's identity. A new sidecar with a new
   `store_id` would fork it, and a person's later `mv` of the temporary file would overwrite
   one or the other. The refusal names the file and says to rename it by hand.
5. **Adopt refuses a `--purpose` that differs from the legacy sidecar's.** Class: **IMPLEMENTATION
   CHOICE.** P10 says adopt keeps `purpose` if present, and it does. The prompt does not say what
   a given, different purpose means. Silently ignoring it, or silently replacing a field the
   prompt says to keep, would each be worse than refusing. A given purpose equal to the kept one
   is accepted.
6. **Both writers check what they are about to write against the reader's registry checks,
   and refuse before writing** (`_check_before_writing`). Class: **IMPLEMENTATION CHOICE.** This
   is prompt 02's read-back idea, moved before the write. It means the writers cannot produce a
   sidecar the reader would call a problem. Mutation (vi) is caught by it, so it fails loudly
   rather than quietly.
7. **What "malformed" means in the reader.** Class: **IMPLEMENTATION CHOICE.** The prompt lists the
   problem classes and leaves "malformed" to the implementation. I read it as follows:
   - `history` must be a non-empty list of objects with the six keys. Entry 0 must be `create` or
     `adopt` with `from: null`, and later entries `copy` or `move` with a `from`. `to`, `when` and
     `git_head` must be non-empty strings and `git_dirty` a boolean. Extra keys in an entry are
     allowed.
   - A `sidecar_format` other than `1` is kind `registry` with a problem. It is not a legacy
     sidecar, since it claims a format, and the file is readable JSON.
   - A sidecar that is itself a symbolic link, or not a regular file, is a problem. This mirrors
     prompt 02's treatment of a symlinked primary, and a symlinked primary is a problem too.
   - A legacy sidecar with no `datastore` or no `name` makes no claim, and is not a problem. Its
     own file name is what ties it to the primary. A legacy `datastore` that is present and names
     another file is a problem, as P9 requires.
8. **The move re-reads the temporary sidecar and requires it to equal what the pre-checks read**
   before updating it. Class: **IMPLEMENTATION CHOICE.** The update starts from the pre-check's
   fields, so an edit to the file in between would be lost. Refusing at that point means it is
   not lost.
9. **Prompt 02's errors gain the sidecar files at each end, not only "no sidecar was written".**
   Class: **IMPLEMENTATION CHOICE.** P11 says to let prompt 02's refusals through unchanged,
   "adding only that no sidecar was written". The same wrapper also carries prompt 02's
   *mid-operation* failures, for which P11 asks that the store and sidecar files at each end be
   listed. So the one wrapper appends `No sidecar was written or moved: sidecar files at the
   destination […]; at the source […]`. Prompt 02's text comes first, unchanged, including its
   `Nothing was written`. `test_prompt_02s_refusals_pass_through`.
10. **Every destination name is checked again immediately before each sidecar rename**
    (`_no_overwrite`). Class: **IMPLEMENTATION CHOICE**, as in prompt 02, because `os.rename`
    replaces silently. The window is narrowed, not closed.
11. **The tests' base class imports `RegistryTestCase` from `test_run_registry`**
    (`store_fixtures.py`), and puts the runs root inside the stores' temporary directory. Class:
    **IMPLEMENTATION CHOICE.** This follows the prompt's "the `RegistryTestCase` pattern" by reuse
    rather than by copy. `test_run_registry.py` is unmodified.
12. **After the demonstration, and after a first pass of the mutations, I changed the interruption
    tests' harness.** Class: **IMPLEMENTATION CHOICE.** The first pass showed that
    `_each_layout`, written as a generator, ran each row's assertions *outside* the layout's
    `subTest`. So the first failed assertion ended the whole method, and the other layouts were
    never checked. It now takes the row's checks as a callback and runs them inside the subtest.
    `test_M5_…` now calls `assertProperty` first, so that mutation (viii) is reported, layout by
    layout, as the property it breaks. `TestMove.test_in_each_layout` clears its directories at
    the start of each layout rather than at the end. These are test-only changes; no production
    file changed after the demonstration. All nine mutations were then re-run, and the record
    above is from that run.
13. **`__main__.py` imports `RunRegistry.stores` at module level**, so `list` loads it too. Class:
    **IMPLEMENTATION CHOICE.** `stores` is standard library only at import. `test_import_and_list_load_neither_ray_nor_sqlalchemy`
    checks `list` in a child interpreter.
14. **The CLI tests run the package with `runpy.run_module("RunRegistry", run_name="__main__",
    alter_sys=True)` in a child interpreter,** which is what `python -m RunRegistry` does. They
    do not literally invoke `-m`. Class: **IMPLEMENTATION CHOICE**: the child can then report
    `ray.is_initialized()` from the same process, as prompt 02's script test does. The existing
    `test_run_registry` tests still run `-m RunRegistry list` literally.
15. **My `var/runs/` snapshot hashed every file under 50 MB**, not only the manifests. Class:
    **IMPLEMENTATION CHOICE**: a superset of what §4 step 1 asks. Every file there is under 50 MB.

The diff touches `RunRegistry/` (the new `stores.py`, `__init__.py`'s docstring and `begin()`,
`__main__.py`, and four new test files), this campaign's board and log, and `docs/OPEN_ISSUES.md`.
`git diff HEAD~1 HEAD -- Datastore/ tools/ main.py docs/handover/ docs/gktk-remedial/ CLAUDE.md` is
empty.

## Verification performed

1. **Suites**, each run as `PYTHONPATH=. ./venv/bin/python -m unittest discover -s
   <package>/tests -t .`, with `THREE_BESSEL_DIAGNOSTIC_PLOTS` unset:

   | Suite | Baseline (`266a70f`, measured by the orchestrator) | Now |
   |---|---|---|
   | AdaptiveLevin | 32 OK | 32 OK |
   | ComputeTargets | 552 OK (the `test_wall_time_per_object` flake is known) | 552 OK (the flake passed this time) |
   | CosmologyModels | 39 OK | 39 OK |
   | Datastore | 70 OK | 70 OK |
   | LiouvilleGreen | 148 OK (skipped=1) | 148 OK (skipped=1) |
   | RunRegistry | 38 OK | **84 OK** (+46, exactly the methods added) |

2. `./venv/bin/python -m black --check .`: clean.
3. **One definition.** `git grep -n -F '.manifest.json' -- RunRegistry/ ':!RunRegistry/tests'`
   finds `stores.py:107`, which is `sidecar_path`. Every other hit is in a docstring: the module
   docstrings of `__init__.py` (`:7`), `__main__.py` (`:7`) and `stores.py` (`:4`),
   `sidecar_path`'s own (`:104`), and two that predate this prompt and name the A3 sidecar
   (`__init__.py:293`, `Run.heartbeat`; `:474`, `begin()`). `begin()` calls `read_sidecar` and
   parses nothing itself.
4. **Layering.** `git diff --cached --stat -- Datastore/ tools/ main.py docs/handover/ docs/gktk-remedial/ CLAUDE.md`
   before committing: empty. In a child interpreter, `import RunRegistry; import RunRegistry.stores`
   leaves `ray` and `sqlalchemy` out of `sys.modules`, and so does `list` run through `runpy`
   (`test_import_and_list_load_neither_ray_nor_sqlalchemy`). `store show`, `create` and `adopt`
   do not import `ray`. `copy` and `move` import it, for `ShardedPool`, and never initialise it.
5. **No delete.** `grep -n "remove\|unlink\|rmtree\|rmdir" RunRegistry/stores.py` finds none.
6. `python -m RunRegistry list` at the start and before §4: nothing `running`.

## Observations not acted on

1. **`docs/handover/quadsource_atol_sweep.py` `prepare()` still writes its sidecar by hand**
   (`:667`, `manifest.write_text(json.dumps(…))`), in the legacy shape. It contradicts README §6.5
   point 1. The script is a measurement record and is out of scope (§6). **Opened
   `[03-atol-sweep-prepare-writes-its-store-sidecar-by-hand]`** in §3, indexed in
   `docs/OPEN_ISSUES.md` §1.12.
2. **Unknown fields that hold paths are carried verbatim and go stale.** The A3 sidecar's
   `restart.command` (`--database var/datastores/handover-A3-baseline-lambdacdm.sqlite`),
   `backup.path` and `status_files` were carried into the probe unchanged. After a copy or move of
   a real store, they would still name the old location. That is the design (§6: unknown fields
   are not interpreted). No issue opened.
3. **A copy's `copied_from` names only its immediate parent.** The sweep sidecar's legacy
   `copied_from` (`…/handover-A3-baseline-lambdacdm.sqlite`) is in the source's sidecar and not in
   `pcopy`'s. `pcopy`'s history starts at the source's `adopt`. The lineage further back is found
   by following `copied_from` to the parent's sidecar. That is §6.5 point 4 as decided. No issue
   opened.
4. **The in-use check is a check, not a lock.** A run begun between the check and the rename is
   not seen. The charter excludes locking (README §6.5), and the stores' docstring says so. No
   issue opened.
5. **No driver creates a sidecar.** A store made by `main.py` or `scoped_pipeline_run.py` has none
   until a person runs `store create`, and `begin()` records `results_store_id: null` for it until
   then. So the id match in P11 protects only stores that have been given a sidecar. The path
   match protects the rest. Whether drivers should create sidecars is the user's call (§6). No
   issue opened.
6. **The three real sidecars stay legacy.** So `store copy` and `store move` refuse the real
   stores until someone runs `store adopt` on them, which is §6.5 point 7 as decided. The backup's
   `datastore` still records the live store's path. The reader now reads it by name, as the
   backup's own sibling, so it no longer misleads code. It is fixed only if the user asks. No
   issue opened.
7. **The index header prose is still stale** (prompt 02's Observations, item 2): its "Of the 90
   above, 87 are spread across the boards" does not match its count line. I corrected only the
   count line's date and this campaign's sentences. No issue opened.

## State handed to the next prompt

All three of this campaign's written prompts have landed. `[store-sidecar-manifests-have-no-owner]`
is closed on this board's §4. Open in §3: `[01-atol-sweep-check-expects-absolute-shard-records]`,
unchanged, and `[03-atol-sweep-prepare-writes-its-store-sidecar-by-hand]`, new. The registry now
creates, adopts, copies and moves stores, with `python -m RunRegistry store`. New run manifests
carry `results_store_id`. No existing sidecar, store or run manifest was rewritten. Whether
anything follows, such as making the pipeline drivers create sidecars or adopting the real
sidecars, is the user's call.
