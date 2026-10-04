# Log — prompt 04: delete the legacy shard records

**Subject:** *Delete the legacy shard records* · **Model:** Claude Sonnet 5.5 · **Date:**
2026-10-04 · **Base:** `2ed3469`, clean · **Result:** landed.
- **L1:** `resolve_shard_path` accepts a bare file name only; `is_legacy_record` and the relocation
  notice are gone.
- **L2:** the audit tool lost its pre-`a2bd966` branch and its tolerance of a primary without
  `shard_key_config`, and refuses an unusable record; the sweep check raises its own `RuntimeError`.
- **L3:** twelve `Datastore` and three `RunRegistry` tests deleted; the rest re-fixtured, split or
  kept; the safety property pinned anew at every reader.

No stop condition fired. Closes nothing; narrows
`[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]`. Opens
`[04-closed-store-refusals-repeat-their-prefix]`. Index 107 → 108. The commit is this commit.

## What shipped

**`Datastore/shard_paths.py`.**
- `resolve_shard_path` sends every record through `_require_bare_name(stored, stored)`; the
  `PurePath(stored).is_absolute()` branch is deleted. Its docstring describes one form and says an
  absolute path is never read, not even by its final component.
- `is_legacy_record` is deleted, and so is the `PurePath` import (`Path` stays).
- The module docstring describes the one form of record, says an absolute record is refused and
  why (a copied primary's absolute records name the original's files, 2026-09-23), and drops the
  legacy paragraph, "Old rows are interpreted when read" and the stale sentence that `prepare()`
  re-points a primary. The paragraphs on the one definition, the naming rule, failing closed and
  the standard-library-only import are kept.
- Kept: `_require_bare_name`, `shard_file_problem`.

**`Datastore/SQL/ShardedPool.py`** (the import, `_resolve_shard_rows` and three docstrings).
- `is_legacy_record` is out of the import.
- `_resolve_shard_rows`: `relocated_legacy`, the `if is_legacy_record(...)` append and the notice
  are deleted. Its docstring says a record that is not a bare name, an absolute path included, is
  refused as unusable.
- `copy_store`'s docstring loses the sentence on a legacy source. `delete_store`'s says that a
  record that is not a bare file name is refused before anything is deleted, so a primary whose
  records name another store's files deletes nothing and never those files.
- Kept: `_shard_file_problems`' "both resolve" check, `_plan_deletion`'s directory guard,
  `_write_shard_data`'s round-trip check.

**`tools/shard_key_audit.py`.**
- The `wavenumber_serial` branch is deleted. A `shard_keys` table without `key_serial` meets the
  one "Unrecognised shard_keys schema" refusal (exit 2), which does not mention `a2bd966`.
- `shard_key_config` is a required table, with `shard_keys` and `shards`. A primary without it is
  refused (exit 2), and so is one whose `shard_key_config` holds no row ("holds no row").
- **The one behaviour change beyond deletion** (confirmed by the user, 2026-10-04): every `shards`
  record goes through `resolve_shard_path` before anything else is checked, and one the resolver
  refuses is a refusal, exit 2, "shard #N record is unusable: <the resolver's reason>", with no
  verdict. It used to print the same line and end "VERDICT: OK". All records are resolved, not
  only shard 0's: a record that is unusable in a later shard is refused too. A missing shard
  *file* keeps today's behaviour: the cross-file check is skipped and said to be.
- The module docstring says that a record that is not a bare name is refused.

**`docs/handover/quadsource_atol_sweep.py`, `assert_store_is_self_consistent` only.**
- A record that `resolve_shard_path` refuses is reported among the mismatches ("serial N: record
  '…' is unusable: <reason>"), so the check raises its own `RuntimeError`. The list `mismatched`
  now holds the report lines themselves rather than tuples, so that the two kinds of line share
  one list; `lines.extend(mismatched)` replaces the loop that formatted them. Nothing outside the
  function changed.
- The docstring describes the one form, keeps the account of 2026-09-23, and says the resolver
  refuses an absolute record.

**`Datastore/tests/shard_store_fixtures.py`.**
- `write_legacy_primary` is renamed `write_hand_built_primary`, in every remaining caller. Its
  docstring says it builds no `replication_in_flight` table, so that the constructor refuses what
  it builds since prompt 03, and that its users go through `read_pool`, the closed-store
  operations, the audit tool or the sweep check, none of which reads that table.
- The module docstring no longer says it stands in for rows the old code wrote.
- `read_pool`'s stdout capture and `pool.read_stdout` are removed (after a grep: their only
  readers were the deleted `TestLegacyRecords` tests), and with them the `contextlib` and `io`
  imports.

## L3, test by test

### Deleted (twelve in `Datastore`, three in `RunRegistry`; counted with `ast` against `HEAD`)

| Module | Class.method |
|---|---|
| `Datastore/tests/test_shard_paths.py` | `TestResolveShardPath.test_legacy_absolute_record_resolves_to_the_sibling_by_name` |
| | `TestResolveShardPath.test_legacy_record_is_not_used_even_when_it_exists_and_differs` |
| | `TestResolveShardPath.test_is_legacy_record` |
| `Datastore/tests/test_shardedpool_shard_paths.py` | `TestLegacyRecords.test_legacy_absolute_records_resolve_to_the_primarys_directory` (the class went with its two tests) |
| | `TestLegacyRecords.test_legacy_records_that_name_their_own_directory_are_silent` |
| | `TestFailClosed.test_missing_sibling_is_refused_even_when_the_absolute_record_exists` |
| | `TestNoSideEffect.test_reading_a_relocated_legacy_store_leaves_the_primary_unchanged` |
| `Datastore/tests/test_copy_move_store.py` | `TestCopyOfALegacySource.test_legacy_source_copied_to_a_new_stem_reads_its_own_files` (the class went with it) |
| | `TestMove.test_legacy_source_in_each_layout` |
| `Datastore/tests/test_delete_store.py` | `TestLegacyRecordsNamingAnotherStore.test_every_listed_file_is_in_b` (the class went with its two tests) |
| | `TestLegacyRecordsNamingAnotherStore.test_delete_removes_only_b_files_and_a_is_untouched` |
| | `TestInterruption.test_the_legacy_shape_other_directory_is_untouched_throughout` |
| `RunRegistry/tests/test_quadsource_atol_sweep_prepare.py` | `SweepPrepareTestCase.test_the_check_accepts_a_legacy_store_naming_its_own_siblings` |
| | `SweepPrepareTestCase.test_the_check_accepts_the_backups_shape` |
| `RunRegistry/tests/test_store_retire.py` | `TestLegacyShape.test_a_primary_naming_another_populated_stores_shards` (the class went with it) |

`ast` shows exactly these removed names, plus the one rename below, and nothing else.

### Re-fixtured (only the records the test builds change, unless stated)

- `test_shardedpool_shard_paths`:
  - `TestCopiedStore._original` builds a bare-name store with `write_new_store`; it serves
    `test_copied_directory_reads_its_own_shards_not_the_originals` and
    `test_copied_with_a_renamed_primary_reads_its_own_shards`. In the first, the assertion that B's
    primary "still names A's shards, which exist" (it needed absolute records) becomes an
    assertion that B's records are the bare names. The assertions on what the pool reads, on the
    markers and on A being untouched are unchanged.
  - `TestFailClosed.test_moved_store_with_legacy_records_and_no_shards_is_refused` is now
    **`test_moved_store_with_no_shards_is_refused`** (renamed to drop "legacy_records"; counted by
    `ast` as one removal and one addition): a bare-name store whose primary is moved alone.
  - `TestFailClosed.test_two_records_resolving_to_one_file_are_refused`: records `{0: n0, 1: n0}`.
- `test_copy_move_store.TestRefusals.test_source_shard_missing_symlinked_irregular_or_shared`: the
  shared case's records are `{0: n0, 1: n0}`.
- `test_delete_store.TestRefusals`:
  - `test_two_serials_share_a_file`: `{0: n0, 1: n0}`;
  - `test_file_outside_the_primary_directory`: two bare-name stores (`new_store("A")`,
    `new_store("B")`), with the resolver mocked to resolve into A's directory
    (`a / stored`). Its expectations are unchanged.
- `test_shard_key_audit_copy.TestAuditOfACopiedStore.setUp`: A's records are bare names. Its three
  tests keep their assertions, including `test_audit_does_not_fall_back_to_the_original_when_the_copy_lacks_shard_0`
  (its premise, that an absolute record does not fall back to the original, now lives in the new
  audit refusal tests).
- `test_quadsource_atol_sweep_prepare.SweepPrepareTestCase`: `test_the_check_refuses_a_renamed_record`
  (the renamed record is the bare name `not-the-right-name.sqlite`),
  `test_the_check_refuses_a_missing_serial`, `…_an_extra_serial` and
  `…_two_swapped_serials` build bare-name records.

### Split (the legacy case dropped, the new-style case unchanged)

- `test_shard_paths.TestResolveShardPath.test_result_is_absolute_and_in_the_primarys_directory`:
  the value `/x/y/store-shard0003.sqlite` is dropped.
- `test_copy_move_store.TestInterruption`: `test_copy`, `test_copy_rewrite_is_one_transaction`,
  `test_move`, `test_move_rewrite_is_one_transaction`: `"legacy"` is out of the case list.
- `test_delete_store.TestInterruption.test_every_point_of_interruption`: `SHAPES` loses `"legacy"`.

### Kept, with the fixture renamed and nothing else changed

- `test_shardedpool_shard_paths.TestFailClosed.test_record_that_is_not_a_bare_name_is_refused_on_read`;
- `test_copy_move_store.TestRefusals.test_source_shard_record_unusable` and
  `test_source_with_no_shards_or_no_shard_zero`;
- `test_delete_store.TestRefusals.test_no_shard_recorded` and `test_unusable_record`.

### Helpers removed (each after a grep; `IMPLEMENTATION CHOICE`)

- `test_shardedpool_shard_paths`: `_legacy_store`.
- `test_copy_move_store`: `legacy_source`, `source(kind)`, `_cases`, and the `kind` parameter of
  `TestMove._move` and `TestInterruption._run`. Each `for kind, layout in self._cases()` is now
  `for layout in LAYOUTS`, and `kind` is gone from the sub-test labels. The surviving cases are the
  same ones, one layout each.
- `test_delete_store`: `legacy_pair`, `store(shape)`, `SHAPES`, the `shape` parameter of
  `_interrupt`, and the `other` parameter of `assertCleanlyDeleted` and `_assertOthersUntouched`
  (always `None` after the deletions; `_assertOthersUntouched` never used it); the imports
  `marker`, `stored_records` and `Tuple`. `quiet` stays (many calls use it); its docstring no
  longer names the notice.
- `test_store_retire`: the `contextlib` redirect for the notice and the `stored_records` call went
  with the deleted test (`stored_records` is used by its replacement; `contextlib` and `io` are
  used elsewhere).
- `shard_store_fixtures`: `read_pool`'s stdout capture, `pool.read_stdout`, and the `contextlib`
  and `io` imports.
- A script (`unused.py`, in the scratchpad) compared the imports of every touched module with the
  names used: none unused.
- Module docstrings corrected where they listed a deleted item:
  `test_shard_paths`, `test_shardedpool_shard_paths`, `test_copy_move_store`, `test_delete_store`
  and the `TestMove`, `TestInterruption` and `TestNoSideEffect` class docstrings.

### Added

- **`Datastore/tests/test_absolute_shard_record_refused.py`** (new; 12 tests, one per entry point
  and one control per class). Stores A (bare names), B (the absolute paths of A's existing shard
  files, with shard files of its own beside it) and C (absolute paths naming its own siblings),
  built in a temporary directory with stand-in shard files. Each refusal asserts "Shard #0",
  "unusable record", "is not a bare file name", the record and the primary's path in the message,
  and that `tree_state` of the whole root, both stores, is unchanged.
  - `TestTheResolver.test_resolve_shard_path_refuses_an_absolute_record` (`ValueError`, each of
    B's and C's six records, as sub-tests);
  - `TestTheResolver.test_the_bare_name_of_the_same_file_is_not_refused` (control);
  - `TestEveryReader.test_the_constructors_read_of_the_shards_table` (`read_pool`, the
    constructor's `_read_shard_data`);
  - `TestEveryReader.test_a_primary_naming_its_own_siblings_is_refused_the_same_way` (`read_pool`
    on C, the one test the prompt asks for);
  - `TestEveryReader.test_copy_store`, `test_move_store` (no destination directory is created);
  - `TestEveryReader.test_closed_store_files`, `test_closed_store_files_resuming`,
    `test_delete_store`, `test_delete_store_resuming`;
  - `TestEveryReader.test_the_read_only_reader` (`store_reader.open_read_only`);
  - `TestEveryReader.test_the_store_it_names_is_not_refused` (control: A is read, listed, and
    unchanged).
- **`Datastore/tests/test_shard_key_audit_refusals.py`** (new; 7 tests, each run as a subprocess
  from an unrelated directory without `PYTHONPATH`; each refusal asserts exit 2, no "VERDICT" in
  the output, empty stderr and an unchanged tree):
  - `test_a_primary_naming_another_stores_shards_by_absolute_path`;
  - `test_a_primary_naming_its_own_siblings_by_absolute_path`;
  - `test_a_record_that_is_unusable_in_a_later_shard_is_refused_too` (my addition, for the
    "all records" choice above);
  - `test_a_primary_without_shard_key_config`;
  - `test_a_primary_whose_shard_key_config_is_empty`;
  - `test_a_shard_keys_table_without_key_serial` (the message says "Unrecognised shard_keys
    schema" and does not contain `a2bd966`);
  - `test_a_current_store_is_still_audited` (control: exit 0 and "VERDICT: OK").
- **`RunRegistry/tests/test_quadsource_atol_sweep_prepare.py`:**
  `SweepPrepareTestCase.test_the_check_refuses_an_absolute_record`. Two sub-tests: B's records
  (another populated store's shards) and a store's own siblings. Each raises `RuntimeError`, not
  `ValueError`, naming `serial N: record '<record>'` for every serial and "is unusable", and the
  tree is unchanged.
- **`RunRegistry/tests/test_store_retire.py`:**
  `TestAnotherStoresShards.test_a_primary_naming_another_populated_stores_shards_is_refused`
  (replaces `TestLegacyShape`; module docstring item 2 reworded). `retire_store` on B, with a
  registry sidecar, under `without_fingerprint=False` and `True` (sub-tests), refuses naming
  "Shard #0", "unusable record" and the record, ending "Nothing was written or deleted"; both
  stores' files and sidecars are byte-identical and neither sidecar gains `retired`.

## The measured double-prefix message

Measured again at the staged tree (script in the scratchpad), on a hand-built primary with
`{0: "../outside.sqlite"}`:

```
closed_store_files / delete_store:
Cannot delete sharded datastore "<p>": Cannot delete sharded datastore "<p>": Shard #0 of primary database "<p>" has an unusable record: shard record '../outside.sqlite' is not a bare file name; ShardedPool only ever records shards that are siblings of the primary, so this row was written by something else. Nothing was deleted

copy_store:
Cannot copy sharded datastore "<p>" to "<d>": Cannot copy sharded datastore "<p>": Shard #0 of primary database "<p>" has an unusable record: … Nothing was written
move_store: the same with "Cannot move".
```

With an absolute record the two `<p>`s differ in the delete case: `/var/folders/…` as given and
`/private/var/folders/…` resolved. `store_reader.open_read_only`'s message is not doubled
(`Cannot read sharded datastore "<p>": Shard #0 …`). Not fixed; opened as
`[04-closed-store-refusals-repeat-their-prefix]`. The new tests assert fragments, not the prefix.

## Deviations from the prompt

1. **`shard_key_audit.py` resolves every record up front, not only shard 0's.** `IMPLEMENTATION
   CHOICE`. The prompt says an unusable record is a refusal; the tool before read only shard 0's.
   One later unusable record would otherwise still end "VERDICT: OK". Pinned by its own test.
2. **The audit tool's refusal of an unusable record prints two lines**: the record and reason, and
   "A ShardedPool primary records each shard by its bare file name. This tool will not attempt to
   audit this datastore." `IMPLEMENTATION CHOICE`.
3. **`TestCopiedStore.test_copied_directory_reads_its_own_shards_not_the_originals` has a changed
   precondition**: B's records are asserted to be the bare names, where they were asserted to name
   A's files. `STRUCTURALLY REQUIRED` (the precondition needs absolute records), and the property
   under test is unchanged.
4. **The sweep check's `mismatched` list holds report lines, not tuples.** `IMPLEMENTATION
   CHOICE`, inside `assert_store_is_self_consistent` only.
5. **The sweep and audit new tests use two shapes (B's, and a store's own siblings)** where the
   prompt asked for B's. `IMPLEMENTATION CHOICE`; the second is one extra sub-test or test.
6. **`TestFailClosed.test_moved_store_with_legacy_records_and_no_shards_is_refused` is renamed**
   `test_moved_store_with_no_shards_is_refused`, as the prompt allows.
7. **The retire refusal test is in a new class `TestAnotherStoresShards`**, replacing
   `TestLegacyShape`, and the module's section header "2. the legacy shape" is reworded.
   `IMPLEMENTATION CHOICE`.

No `UNINTENDED DRIFT`.

## Verification performed

- **Suites**, serially on the staged tree, README §7's command, output under `<scratch>/agent04/`,
  verdicts grepped from the files. The Datastore suite ran in four groups of modules (a Bash call
  cannot run longer than ten minutes), 122 + 132 + 201 + 181 = 636.

  | Suite | Before (orchestrator, `2ed3469`) | After | Difference |
  |---|---|---|---|
  | AdaptiveLevin | 32 OK | 32 OK | 0 |
  | ComputeTargets | 693, 2 failures (`test_background_tau.test_build_cost`, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`) | 693, the same 2 failures | 0 |
  | CosmologyModels | 39 OK | 39 OK | 0 |
  | Datastore | 629 OK | 636 OK | −12 deleted, +12 (`test_absolute_shard_record_refused`), +7 (`test_shard_key_audit_refusals`) |
  | LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 0 |
  | RunRegistry | 281 OK | 280 OK | −3 deleted, +1 sweep, +1 retire |

  - The two ComputeTargets failures are the known wall-clock tests. Alone, at load 77:
    `test_build_cost` 0.68 s against 0.5 s, `test_wall_time_per_object` 0.085 s against 0.05 s.
    `test_quadsource_policy_main` run alone: 19 tests, OK.
  - Every test outside §2 L3 passed untouched: no test outside the list failed, so no stop
    condition fired.
- **The §3.2 greps**, run at the staged tree:
  `grep -rnE 'legacy|is_legacy_record' Datastore/shard_paths.py Datastore/SQL/ShardedPool.py tools/shard_key_audit.py Datastore/tests/shard_store_fixtures.py`
  prints nothing (exit 1); `grep -n 'a2bd966\|wavenumber_serial' tools/shard_key_audit.py` prints
  nothing (exit 1). `grep -rn 'is_legacy_record\|legacy absolute' --include='*.py'` over the
  repository (without `venv`) prints nothing.
- **`black --check`** on all fourteen touched Python files: clean.
- **Staging** by explicit path; `git diff --cached --name-only` shown before the commit.

## The deliberate-breakage record

**How each was run.** Each mutation was applied to the working tree over the staged final tree,
and its diff taken with `git diff -- <file>`, so `git apply` replays it from a clean checkout of
this commit. For each: `git apply -R --check`, run, `git apply -R`, then `git apply --check`
(forward) on the clean tree. All passed, and the tree was clean against the index after each. None
is committed. (i) was run on the new module, the audit and sweep tests, the retire test and
the five Datastore modules the prompt re-fixtured; (ii) on the new module, the audit and sweep
tests, the retire test and `test_delete_store`; (iii)–(v) on the two audit modules and the sweep
module.

**(i) Restore the absolute branch that reads a record's final component as a sibling.**
Ran 98, **FAILED (failures=20, errors=1)**, all in the new tests, none elsewhere:
- `test_absolute_shard_record_refused`: every `TestEveryReader` test but the control
  (`test_the_read_only_reader` errors, the others fail), and `TestTheResolver`'s refusal test, six
  sub-tests;
- `test_shard_key_audit_refusals`: `test_a_primary_naming_another_stores_shards_by_absolute_path`
  and `test_a_primary_naming_its_own_siblings_by_absolute_path`;
- `test_quadsource_atol_sweep_prepare`: `test_the_check_refuses_an_absolute_record`, both
  sub-tests;
- `test_store_retire`: `TestAnotherStoresShards…`, both sub-tests.

```diff
diff --git a/Datastore/shard_paths.py b/Datastore/shard_paths.py
index 4642219..ff5e293 100644
--- a/Datastore/shard_paths.py
+++ b/Datastore/shard_paths.py
@@ -98,7 +98,10 @@ def resolve_shard_path(primary: Union[str, Path], stored: str) -> Path:
             f"shard record {stored!r} is not a string (type {type(stored).__name__})"
         )
 
-    name = _require_bare_name(stored, stored)
+    if Path(stored).is_absolute():
+        name = _require_bare_name(stored.rsplit("/", 1)[-1], stored)
+    else:
+        name = _require_bare_name(stored, stored)
 
     return primary.parent / name
 
```

**(ii) The 2026-09-23 failure: return `Path(stored)` for an absolute record whose file exists.**
Run on the new module, the audit and sweep tests, `TestAnotherStoresShards` and
`test_delete_store`: Ran 56, **FAILED (failures=20, errors=1)**, the same tests as (i), and none
of `test_delete_store`. Only temporary directories are involved: every store in the new module is
built in one.

```diff
diff --git a/Datastore/shard_paths.py b/Datastore/shard_paths.py
index 4642219..5dccb6e 100644
--- a/Datastore/shard_paths.py
+++ b/Datastore/shard_paths.py
@@ -98,6 +98,9 @@ def resolve_shard_path(primary: Union[str, Path], stored: str) -> Path:
             f"shard record {stored!r} is not a string (type {type(stored).__name__})"
         )
 
+    if Path(stored).is_absolute() and Path(stored).exists():
+        return Path(stored)
+
     name = _require_bare_name(stored, stored)
 
     return primary.parent / name
```

**(iii) In the audit tool, let a primary without `shard_key_config` through.** Run on the two audit
modules and the sweep module: Ran 20, **FAILED (failures=1)**:
`test_shard_key_audit_refusals.TestTheAuditRefuses.test_a_primary_without_shard_key_config`.

```diff
diff --git a/tools/shard_key_audit.py b/tools/shard_key_audit.py
index 0b870e3..4fe6dab 100644
--- a/tools/shard_key_audit.py
+++ b/tools/shard_key_audit.py
@@ -78,7 +78,7 @@ def main(argv: List[str]) -> int:
         print(f"!! Could not open {primary_path} read-only: {e}")
         return 2
 
-    if not {"shard_keys", "shards", "shard_key_config"} <= tables:
+    if not {"shard_keys", "shards"} <= tables:
         print(
             f"!! {primary_path} does not look like a completely written ShardedPool "
             f"primary database (expected tables 'shard_keys', 'shards' and "
@@ -99,7 +99,11 @@ def main(argv: List[str]) -> int:
         )
         return 2
 
-    config_row = conn.execute("SELECT key_type FROM shard_key_config").fetchone()
+    config_row = (
+        conn.execute("SELECT key_type FROM shard_key_config").fetchone()
+        if "shard_key_config" in tables
+        else (None,)
+    )
     if config_row is None:
         print(
             f"!! The shard_key_config table of {primary_path} holds no row. This is "
```

**(iv) In the audit tool, let an unusable record end in "VERDICT: OK".** Ran 20, **FAILED
(failures=3)**: `test_a_primary_naming_another_stores_shards_by_absolute_path`,
`test_a_primary_naming_its_own_siblings_by_absolute_path` and
`test_a_record_that_is_unusable_in_a_later_shard_is_refused_too`, all in
`test_shard_key_audit_refusals.TestTheAuditRefuses`.

```diff
diff --git a/tools/shard_key_audit.py b/tools/shard_key_audit.py
index 0b870e3..e0f4fe3 100644
--- a/tools/shard_key_audit.py
+++ b/tools/shard_key_audit.py
@@ -120,11 +120,8 @@ def main(argv: List[str]) -> int:
             shard_paths[shard_serial] = resolve_shard_path(primary_path, shard_filename)
         except ValueError as e:
             print(f"!! shard #{shard_serial} record is unusable: {e}")
-            print(
-                "!! A ShardedPool primary records each shard by its bare file name. "
-                "This tool will not attempt to audit this datastore."
-            )
-            return 2
+            shard_files = []
+            break
 
     print(f">> {primary_path}: current-generation schema (key_serial present)")
 
```

**(v) In the sweep check, let the resolver's `ValueError` escape.** Ran 20, **FAILED
(errors=2)**: `test_quadsource_atol_sweep_prepare.SweepPrepareTestCase.test_the_check_refuses_an_absolute_record`,
both sub-tests, each erroring with the `ValueError`.

```diff
diff --git a/docs/handover/quadsource_atol_sweep.py b/docs/handover/quadsource_atol_sweep.py
index 4a2b4ba..1653f30 100644
--- a/docs/handover/quadsource_atol_sweep.py
+++ b/docs/handover/quadsource_atol_sweep.py
@@ -703,13 +703,7 @@ def assert_store_is_self_consistent(primary):
     extra = sorted(set(found) - set(expected))
     mismatched = []
     for serial in sorted(set(expected) & set(found)):
-        try:
-            resolved = resolve_shard_path(primary, found[serial])
-        except ValueError as e:
-            mismatched.append(
-                f"  serial {serial}: record {found[serial]!r} is unusable: {e}"
-            )
-            continue
+        resolved = resolve_shard_path(primary, found[serial])
         if resolved != expected[serial]:
             mismatched.append(
                 f"  serial {serial}: record {found[serial]!r} resolves to {resolved}, "
```

## Observations not acted on

- **Scripts under `docs/` read `shards.filename` without the resolver** (stop condition 3,
  checked: none behaves differently, because none ever used it):
  - `docs/a3-v2-readiness/rehearsal/closed_store_state.py:24-39` reads each record's final
    component as a sibling (`Path(name).name`), which is the legacy reading, kept in a
    verification helper;
  - `docs/datastore-integrity-verification/fault_copies.py:45` and
    `qsi_resolution_cost.py:58` join the stored value to the primary's directory.

  They are outside §6 and are measurements of past runs. Not changed.
- **`RunRegistry/stores.py:81-82`** explains the sidecar rule as "prompt 01's rule for legacy shard
  records, applied to JSON". After 04 that rule no longer exists for shard records. The sentence
  is U3's to keep or change (the prompt says to record it).
- **`tools/shard_key_audit.py` still ends in a traceback (exit 1) on a shard file that is not a
  database** (`sqlite3.DatabaseError: file is not a database`, from `ATTACH`; it catches only
  `OperationalError`), and on a `shard_key_config` table with no `key_type` column. Exit 1 is also
  its INCONSISTENT code. Unchanged by 04 (found while building the control fixture). No issue is
  opened; the orchestrator may want one.
- **`quiet(...)` in `test_delete_store`** is now a pass-through that captures stdout from calls that
  no longer print. Kept, to leave its many call sites alone.
- **`test_copy_move_store`'s module docstring** still refers to `prompts/datastore-portability`
  prompts, as before.

## State handed to the next prompt

- A `shards` record is a bare file name, or every reader refuses it. `Datastore.shard_paths` has no
  `is_legacy_record`. `ShardedPool` prints no relocation notice.
- `shard_store_fixtures.write_hand_built_primary` builds a primary without `replication_in_flight`
  (the constructor refuses it).
- 05's list is unchanged: B5, the four WKB `numeric_serial` selects, the `OperationalError` for a
  primary without its record table, the split grouping's exclusion of `timestamp`.
- `Datastore` 636, `RunRegistry` 280; every other suite unchanged. The index holds 108.
