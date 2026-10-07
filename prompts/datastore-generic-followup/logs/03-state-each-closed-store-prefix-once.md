# Log — prompt 03: state each closed-store prefix once

**Subject:** *State each closed-store prefix once* · **Model:** Claude Sonnet 5.5 ·
**Date:** 2026-10-07 · **Base:** `21ee420`, plus the interrupted predecessor's uncommitted tree
(see below; code as at `4c34f13`) · **Result:** landed.
- **P1:** `ShardedPool._read_closed_store` raises the bare reason for each of its three refusals
  and takes no `verb`.
- **P2:** `store_reader.open_read_only` wraps that reason with its own `_refuse`.

No stop condition fired. Closes `[04-closed-store-refusals-repeat-their-prefix]`
(`datastore-generic`), and with it the last of that campaign's issues and this campaign. Narrows
nothing. Opens nothing. Index 105 -> 104. The commit is `086c81a`.

**Resumed work.** A previous agent on this prompt was killed when the machine slept. It left
uncommitted changes to `Datastore/SQL/ShardedPool.py`, `Datastore/store_reader.py` and
`Datastore/tests/test_store_reader.py`, and a new `Datastore/tests/test_closed_store_refusals.py`,
and wrote no log and no records. This agent adopted that tree as its starting point after reading
every line of it against the prompt and the orchestrator's §0, and then owned it: the changes were
kept as they were (the code is the prompt's change; the new module already held five tests with the
three fixtures of the orchestrator's addition), and every measurement, suite run, breakage and
record in this log was taken by this agent on the final tree. The predecessor's scratch outputs
were not used.

## What shipped

**`Datastore/SQL/ShardedPool.py`, `_read_closed_store` and its three calls.**
- Signature `_read_closed_store(primary, *, missing_ok=False)`; `verb` is gone.
- The three `raise`s give only the reason: `its shards table could not be read (<e>)`; the
  resolver's `RuntimeError`, which now propagates unwrapped (the `try`/`except` that re-wrapped it
  is gone); and the joined file problems, `"; ".join(problems)`.
- The docstring says it raises `RuntimeError` naming the problem, and that each caller states the
  operation and the store.
- `_plan_relocation`, `_read_back` and `_plan_deletion` stop passing a verb; each keeps its own
  wrapping (`refuse(str(e))`, `_failure_message`) exactly. `_failure_message` is untouched.

**`Datastore/store_reader.py`, `open_read_only` only.** The call becomes
`try: ... except RuntimeError as e: raise _refuse(primary_path, str(e)) from e`. `_VERB` stays:
`_refuse` uses it.

**`Datastore/tests/test_store_reader.py`, one test.**
`TestRefusals.test_missing_shard_is_refused_by_read_closed_store` calls
`ShardedPool._read_closed_store(self.store.primary)`, asserts that the reader's message contains
that call's message, and that it holds `Cannot read sharded datastore` exactly once. Its comment
says the refusal is `_read_closed_store`'s, under the reader's one prefix.

**`Datastore/tests/test_closed_store_refusals.py`, new, 5 tests.** No Ray, nothing under `var/`.
"Once" means the message begins with the operation's prefix and holds `sharded datastore` exactly
once. Fixtures: a shard record `../outside.sqlite` (built as `test_copy_move_store` builds it), a
dropped `shards` table, and a missing shard file.
1. `test_delete_and_closed_store_files_state_their_prefix_once`;
2. `test_copy_and_move_state_their_prefix_once` (the destination named, nothing written);
3. `test_the_reader_states_its_prefix_once` (and the message ends `Nothing was read or repaired`);
4. `test_an_unreadable_shards_table_is_named_once_by_each` (`delete_store`, `copy_store`,
   `open_read_only`; each holds "its shards table could not be read" once);
5. `test_read_closed_store_states_no_prefix`: for **three** fixtures (unusable record, dropped
   `shards` table, missing shard file) the message does not begin `Cannot`, does not hold
   `sharded datastore`, and holds the reason's fragment.

**Records.** `datastore-generic`'s board: the issue moved from §3.1 to §4 with a dated "Closed"
line naming this campaign, the change and the fourth caller; §3.1 says none open; header date.
`docs/OPEN_ISSUES.md`: the row left §1.19 (its table is gone), §1.18 and §1.19 say both campaigns'
issues are closed, count 104, date. This board: header (CLOSED, 3 of 3), the §1 row, P1-P2, §3,
§4 and §5.4. This campaign's README header and §2 status. `prompts/INDEX.md`: this campaign's row
(closed 3 / 3, 2026-10-07, 0 open issues), `datastore-generic`'s row (open issues 0, last-touched
date), the header counts.

## Deviations from the prompt

1. **Test 5 covers three fixtures, and there is a fifth breakage (v)**, at the orchestrator's
   direction: the file-problem `raise`, which neither of the prompt's two fixtures reaches, is
   covered by a missing shard file. The count of new tests stays 5. `IMPLEMENTATION CHOICE`, made
   at the orchestrator's direction.
2. **The resolver branch lets the `RuntimeError` propagate** rather than re-raising it. The
   re-raise would add nothing, and the orchestrator named either as equally right. Breakage (i)
   adds the wrap back. `IMPLEMENTATION CHOICE`.
3. **The `../outside.sqlite` fixture** is built with `write_hand_built_primary(primary, {0:
   "../outside.sqlite"})` plus a placeholder at `outside.sqlite`, as `test_copy_move_store`
   builds it, since the prompt's "absolute-record fixture ... (a shard record `../outside.sqlite`)"
   names two fixtures; both reach the resolver's refusal. `IMPLEMENTATION CHOICE`.
4. **The docstring** names the removed prefix's replacement ("each caller states the operation and
   the store"), beyond "raises RuntimeError naming the problem". `IMPLEMENTATION CHOICE`.
5. **Records beyond the prompt's list of edits**: `datastore-generic`'s board's header date and
   §3.1's "None open", `docs/OPEN_ISSUES.md` §1.18's closing sentence, and `prompts/INDEX.md`'s
   `datastore-generic` row (open issues 3 -> 0, last-touched date) and header counts, all in files
   §6 lists. `IMPLEMENTATION CHOICE`.
6. **Breakage scope.** Each breakage was run against the Datastore and RunRegistry modules that
   mention these operations or messages (16 modules, 282 tests), not the whole suites. The full
   suites ran on the final tree. `IMPLEMENTATION CHOICE`.

No `UNINTENDED DRIFT`.

## Verification performed

**Suites**, README §7's command, each in the foreground, serially, output to a file, the verdict
grepped, on the final tree. Baselines are the brief's, at `21ee420`.

| Suite | Before (`21ee420`) | After | Difference |
|---|---|---|---|
| Datastore `[a-f]` | 158 OK | 163 OK | +5: the five tests of `test_closed_store_refusals` |
| Datastore `[g-o]` | 185 OK | 185 OK | none |
| Datastore `[p-r]` | 200 OK | 200 OK | none |
| Datastore `[s-z]` | 191 OK | 191 OK | none |
| **Datastore** | **734** | **739** | +5 |
| RunRegistry | 280 OK | 280 OK | none (`test_store_copy_move` and `test_store_retire` unedited) |
| ComputeTargets | 693 OK | 693 OK | none |
| AdaptiveLevin | 32 OK | 32 OK | none |
| CosmologyModels | 39 OK | 39 OK | none |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | none |

No known flake fired. `test_store_reader` (15 tests) and `test_closed_store_refusals` (5) pass
together (20). `test_layer_is_generic` is in `[g-o]`, which passes.
`black --check` is clean on the four Python files.

`git grep -n '_read_closed_store(' -- '*.py'` shows the definition, the three calls in
`ShardedPool.py` (`_plan_relocation`, `_read_back`, `_plan_deletion`), `store_reader.py`'s one
call, `test_store_reader`'s changed test (which names the function in its own name and calls it
without a verb) and the orchestrator's scratch script; none passes a verb.

**The messages, before and after** (prompt §3.2). `m3_messages.py` (the orchestrator's) was run at
an export of `HEAD` (`21ee420`, `git archive`) and at the changed tree, for the absolute record,
the missing shard file and the dropped `shards` table; `m3_compare.py` on the two outputs. Paths:
`<R>` is the resolved temporary root, `<T>` the root as given. The parent's output is
byte-identical to the orchestrator's `messages-parent.json`, and the changed tree's to its
`messages-rough.json`.

```
== absolute / _read_closed_store: as expected
   before (1x): RuntimeError: Cannot read sharded datastore "<R>/B/store.sqlite": Shard #0 of primary database "<R>/B/store.sqlite" has an unusable record: shard record '<T>/A/store-shard0000.sqlite' is not a bare file name; ShardedPool only ever records shards that are siblings of the primary, so this row was written by something else
   after  (0x): RuntimeError: Shard #0 of primary database "<R>/B/store.sqlite" has an unusable record: shard record '<T>/A/store-shard0000.sqlite' is not a bare file name; ShardedPool only ever records shards that are siblings of the primary, so this row was written by something else
== absolute / closed_store_files: as expected
   before (2x): RuntimeError: Cannot delete sharded datastore "<T>/B/store.sqlite": Cannot delete sharded datastore "<R>/B/store.sqlite": Shard #0 of primary database "<R>/B/store.sqlite" has an unusable record: shard record '<T>/A/store-shard0000.sqlite' is not a bare file name; ShardedPool only ever records shards that are siblings of the primary, so this row was written by something else. Nothing was deleted
   after  (1x): RuntimeError: Cannot delete sharded datastore "<T>/B/store.sqlite": Shard #0 of primary database "<R>/B/store.sqlite" has an unusable record: shard record '<T>/A/store-shard0000.sqlite' is not a bare file name; ShardedPool only ever records shards that are siblings of the primary, so this row was written by something else. Nothing was deleted
== absolute / copy_store: as expected
   before (2x): RuntimeError: Cannot copy sharded datastore "<T>/B/store.sqlite" to "<T>/D/absolute.sqlite": Cannot copy sharded datastore "<R>/B/store.sqlite": Shard #0 of primary database "<R>/B/store.sqlite" has an unusable record: shard record '<T>/A/store-shard0000.sqlite' is not a bare file name; ShardedPool only ever records shards that are siblings of the primary, so this row was written by something else. Nothing was written
   after  (1x): RuntimeError: Cannot copy sharded datastore "<T>/B/store.sqlite" to "<T>/D/absolute.sqlite": Shard #0 of primary database "<R>/B/store.sqlite" has an unusable record: shard record '<T>/A/store-shard0000.sqlite' is not a bare file name; ShardedPool only ever records shards that are siblings of the primary, so this row was written by something else. Nothing was written
== absolute / delete_store: as expected
   before (2x): RuntimeError: Cannot delete sharded datastore "<T>/B/store.sqlite": Cannot delete sharded datastore "<R>/B/store.sqlite": Shard #0 of primary database "<R>/B/store.sqlite" has an unusable record: shard record '<T>/A/store-shard0000.sqlite' is not a bare file name; ShardedPool only ever records shards that are siblings of the primary, so this row was written by something else. Nothing was deleted
   after  (1x): RuntimeError: Cannot delete sharded datastore "<T>/B/store.sqlite": Shard #0 of primary database "<R>/B/store.sqlite" has an unusable record: shard record '<T>/A/store-shard0000.sqlite' is not a bare file name; ShardedPool only ever records shards that are siblings of the primary, so this row was written by something else. Nothing was deleted
== absolute / move_store: as expected
   before (2x): RuntimeError: Cannot move sharded datastore "<T>/B/store.sqlite" to "<T>/D/absolute.sqlite": Cannot move sharded datastore "<R>/B/store.sqlite": Shard #0 of primary database "<R>/B/store.sqlite" has an unusable record: shard record '<T>/A/store-shard0000.sqlite' is not a bare file name; ShardedPool only ever records shards that are siblings of the primary, so this row was written by something else. Nothing was written
   after  (1x): RuntimeError: Cannot move sharded datastore "<T>/B/store.sqlite" to "<T>/D/absolute.sqlite": Shard #0 of primary database "<R>/B/store.sqlite" has an unusable record: shard record '<T>/A/store-shard0000.sqlite' is not a bare file name; ShardedPool only ever records shards that are siblings of the primary, so this row was written by something else. Nothing was written
== absolute / open_read_only: as expected
   before (1x): RuntimeError: Cannot read sharded datastore "<R>/B/store.sqlite": Shard #0 of primary database "<R>/B/store.sqlite" has an unusable record: shard record '<T>/A/store-shard0000.sqlite' is not a bare file name; ShardedPool only ever records shards that are siblings of the primary, so this row was written by something else
   after  (1x): RuntimeError: Cannot read sharded datastore "<R>/B/store.sqlite": Shard #0 of primary database "<R>/B/store.sqlite" has an unusable record: shard record '<T>/A/store-shard0000.sqlite' is not a bare file name; ShardedPool only ever records shards that are siblings of the primary, so this row was written by something else. Nothing was read or repaired
== missing_shard / _read_closed_store: as expected
   before (1x): RuntimeError: Cannot read sharded datastore "<R>/M/store.sqlite": shard #1: stored record "store-shard0001.sqlite" resolves to "<R>/M/store-shard0001.sqlite", which does not exist
   after  (0x): RuntimeError: shard #1: stored record "store-shard0001.sqlite" resolves to "<R>/M/store-shard0001.sqlite", which does not exist
== missing_shard / closed_store_files: as expected
   before (2x): RuntimeError: Cannot delete sharded datastore "<T>/M/store.sqlite": Cannot delete sharded datastore "<R>/M/store.sqlite": shard #1: stored record "store-shard0001.sqlite" resolves to "<R>/M/store-shard0001.sqlite", which does not exist. Nothing was deleted
   after  (1x): RuntimeError: Cannot delete sharded datastore "<T>/M/store.sqlite": shard #1: stored record "store-shard0001.sqlite" resolves to "<R>/M/store-shard0001.sqlite", which does not exist. Nothing was deleted
== missing_shard / copy_store: as expected
   before (2x): RuntimeError: Cannot copy sharded datastore "<T>/M/store.sqlite" to "<T>/D/missing_shard.sqlite": Cannot copy sharded datastore "<R>/M/store.sqlite": shard #1: stored record "store-shard0001.sqlite" resolves to "<R>/M/store-shard0001.sqlite", which does not exist. Nothing was written
   after  (1x): RuntimeError: Cannot copy sharded datastore "<T>/M/store.sqlite" to "<T>/D/missing_shard.sqlite": shard #1: stored record "store-shard0001.sqlite" resolves to "<R>/M/store-shard0001.sqlite", which does not exist. Nothing was written
== missing_shard / delete_store: as expected
   before (2x): RuntimeError: Cannot delete sharded datastore "<T>/M/store.sqlite": Cannot delete sharded datastore "<R>/M/store.sqlite": shard #1: stored record "store-shard0001.sqlite" resolves to "<R>/M/store-shard0001.sqlite", which does not exist. Nothing was deleted
   after  (1x): RuntimeError: Cannot delete sharded datastore "<T>/M/store.sqlite": shard #1: stored record "store-shard0001.sqlite" resolves to "<R>/M/store-shard0001.sqlite", which does not exist. Nothing was deleted
== missing_shard / move_store: as expected
   before (2x): RuntimeError: Cannot move sharded datastore "<T>/M/store.sqlite" to "<T>/D/missing_shard.sqlite": Cannot move sharded datastore "<R>/M/store.sqlite": shard #1: stored record "store-shard0001.sqlite" resolves to "<R>/M/store-shard0001.sqlite", which does not exist. Nothing was written
   after  (1x): RuntimeError: Cannot move sharded datastore "<T>/M/store.sqlite" to "<T>/D/missing_shard.sqlite": shard #1: stored record "store-shard0001.sqlite" resolves to "<R>/M/store-shard0001.sqlite", which does not exist. Nothing was written
== missing_shard / open_read_only: as expected
   before (1x): RuntimeError: Cannot read sharded datastore "<R>/M/store.sqlite": shard #1: stored record "store-shard0001.sqlite" resolves to "<R>/M/store-shard0001.sqlite", which does not exist
   after  (1x): RuntimeError: Cannot read sharded datastore "<R>/M/store.sqlite": shard #1: stored record "store-shard0001.sqlite" resolves to "<R>/M/store-shard0001.sqlite", which does not exist. Nothing was read or repaired
== no_shards_table / _read_closed_store: as expected
   before (1x): RuntimeError: Cannot read sharded datastore "<R>/N/store.sqlite": its shards table could not be read (no such table: shards)
   after  (0x): RuntimeError: its shards table could not be read (no such table: shards)
== no_shards_table / closed_store_files: as expected
   before (2x): RuntimeError: Cannot delete sharded datastore "<T>/N/store.sqlite": Cannot delete sharded datastore "<R>/N/store.sqlite": its shards table could not be read (no such table: shards). Nothing was deleted
   after  (1x): RuntimeError: Cannot delete sharded datastore "<T>/N/store.sqlite": its shards table could not be read (no such table: shards). Nothing was deleted
== no_shards_table / copy_store: as expected
   before (2x): RuntimeError: Cannot copy sharded datastore "<T>/N/store.sqlite" to "<T>/D/no_shards_table.sqlite": Cannot copy sharded datastore "<R>/N/store.sqlite": its shards table could not be read (no such table: shards). Nothing was written
   after  (1x): RuntimeError: Cannot copy sharded datastore "<T>/N/store.sqlite" to "<T>/D/no_shards_table.sqlite": its shards table could not be read (no such table: shards). Nothing was written
== no_shards_table / delete_store: as expected
   before (2x): RuntimeError: Cannot delete sharded datastore "<T>/N/store.sqlite": Cannot delete sharded datastore "<R>/N/store.sqlite": its shards table could not be read (no such table: shards). Nothing was deleted
   after  (1x): RuntimeError: Cannot delete sharded datastore "<T>/N/store.sqlite": its shards table could not be read (no such table: shards). Nothing was deleted
== no_shards_table / move_store: as expected
   before (2x): RuntimeError: Cannot move sharded datastore "<T>/N/store.sqlite" to "<T>/D/no_shards_table.sqlite": Cannot move sharded datastore "<R>/N/store.sqlite": its shards table could not be read (no such table: shards). Nothing was written
   after  (1x): RuntimeError: Cannot move sharded datastore "<T>/N/store.sqlite" to "<T>/D/no_shards_table.sqlite": its shards table could not be read (no such table: shards). Nothing was written
== no_shards_table / open_read_only: as expected
   before (1x): RuntimeError: Cannot read sharded datastore "<R>/N/store.sqlite": its shards table could not be read (no such table: shards)
   after  (1x): RuntimeError: Cannot read sharded datastore "<R>/N/store.sqlite": its shards table could not be read (no such table: shards). Nothing was read or repaired
0 unexpected
```

The resolved path leaves some messages (`<R>` for `delete_store`, `copy_store` and `move_store`
on a dropped `shards` table): the removed prefix was the one that named it, and that reason names no
file. This is the removed repetition, not a changed reason.

**No Ray was started, and nothing under `var/` was opened.**

## The deliberate-breakage record

Each diff is exactly as applied, against `086c81a`'s tree, produced by `git diff` after staging
the change, then checked in `bash` with `git apply --check` (forward, on the clean tree) and
`git apply -R --check` (applied), run, and reversed with `git apply -R`, leaving `git status` clean.
The modules run were the 16 that mention these operations or messages: `test_closed_store_refusals`,
`test_absolute_shard_record_refused`, `test_delete_store`, `test_copy_move_store`,
`test_drop_refuses_dangling_references`, `test_layer_registry`, `test_store_reader`,
`test_shard_file_name`, `test_sharded_store_script`, `test_store_schema` (Datastore) and
`test_store_retire_references`, `test_store_amend`, `test_store_fingerprint`,
`test_store_copy_move`, `test_store_retire`, `test_store_sidecar` (RunRegistry): 282 tests, which
pass on `086c81a`.

**(i) The resolver branch's `raise` prefixed again.**

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -2665,7 +2665,12 @@ class ShardedPool:
 
         files: Dict[int, Path] = {}
         records: Dict[int, str] = {}
-        ShardedPool._resolve_shard_rows(primary, rows, files, records)
+        try:
+            ShardedPool._resolve_shard_rows(primary, rows, files, records)
+        except RuntimeError as e:
+            raise RuntimeError(
+                f'Cannot open sharded datastore "{str(primary)}": {e}'
+            ) from e
 
         # missing_ok is for delete_store(resume=True) only: a shard file that is not there is not
         # refused. Every serial is still returned, present or not
```

Failed (5 failures and subtest failures in 4 tests; the tests the prompt names are in bold; the prompt names tests 1, 2, 3 and 5):
  - **`test_closed_store_refusals.TestEachPrefixOnce.test_copy_and_move_state_their_prefix_once`** (2 subtest failures)
  - **`test_closed_store_refusals.TestEachPrefixOnce.test_delete_and_closed_store_files_state_their_prefix_once`** (1 subtest failures)
  - **`test_closed_store_refusals.TestEachPrefixOnce.test_the_reader_states_its_prefix_once`** (1 subtest failures)
  - **`test_closed_store_refusals.TestTheBareReason.test_read_closed_store_states_no_prefix`** (1 subtest failures)

**(ii) The same prefix in the shards-table branch.**

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -2661,7 +2661,9 @@ class ShardedPool:
             finally:
                 conn.close()
         except sqlite3.Error as e:
-            raise RuntimeError(f"its shards table could not be read ({e})") from e
+            raise RuntimeError(
+                f'Cannot open sharded datastore "{str(primary)}": its shards table could not be read ({e})'
+            ) from e
 
         files: Dict[int, Path] = {}
         records: Dict[int, str] = {}
```

Failed (4 failures and subtest failures in 2 tests; the tests the prompt names are in bold; the prompt names tests 4 and 5):
  - **`test_closed_store_refusals.TestEachPrefixOnce.test_an_unreadable_shards_table_is_named_once_by_each`** (3 subtest failures)
  - **`test_closed_store_refusals.TestTheBareReason.test_read_closed_store_states_no_prefix`** (1 subtest failures)

**(iii) The reader's `try` removed, so that its message has no prefix.**

```diff
diff --git a/Datastore/store_reader.py b/Datastore/store_reader.py
--- a/Datastore/store_reader.py
+++ b/Datastore/store_reader.py
@@ -203,10 +203,7 @@ def open_read_only(primary: PathType, factories: Mapping) -> Iterator[ReadOnlySt
 
     _refuse_journals(primary_path, primary_path, "the primary")
 
-    try:
-        files, _records = ShardedPool._read_closed_store(primary_path)
-    except RuntimeError as e:
-        raise _refuse(primary_path, str(e)) from e
+    files, _records = ShardedPool._read_closed_store(primary_path)
     if len(files) == 0:
         raise _refuse(
             primary_path, f'the primary "{str(primary_path)}" records no shards'
```

Failed (4 failures and subtest failures in 4 tests; the tests the prompt names are in bold; the prompt names test 3 and the changed `test_store_reader` test):
  - `test_closed_store_refusals.TestEachPrefixOnce.test_an_unreadable_shards_table_is_named_once_by_each` (1 subtest failures)
  - **`test_closed_store_refusals.TestEachPrefixOnce.test_the_reader_states_its_prefix_once`** (1 subtest failures)
  - **`test_store_reader.TestRefusals.test_missing_shard_is_refused_by_read_closed_store`** (1 subtest failures)
  - `test_store_reader.TestRefusals.test_primary_with_no_shards_table` (1 subtest failures)

**(iv) `_plan_deletion` stops wrapping (`refuse(str(e))` becomes `RuntimeError(str(e))`).**

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -2963,7 +2963,7 @@ class ShardedPool:
                 primary_file, missing_ok=resume
             )
         except RuntimeError as e:
-            raise refuse(str(e)) from e
+            raise RuntimeError(str(e)) from e
 
         if len(recorded) == 0:
             raise refuse(f'the primary "{str(primary_file)}" records no shards')
```

Failed (56 failures and subtest failures in 10 tests; the tests the prompt names are in bold; the prompt names test 1):
  - `test_closed_store_refusals.TestEachPrefixOnce.test_an_unreadable_shards_table_is_named_once_by_each` (1 subtest failures)
  - **`test_closed_store_refusals.TestEachPrefixOnce.test_delete_and_closed_store_files_state_their_prefix_once`** (1 subtest failures)
  - `test_delete_store.TestInterruption.test_every_point_of_interruption` (8 subtest failures)
  - `test_delete_store.TestRefusals.test_missing_shard` (2 subtest failures)
  - `test_delete_store.TestRefusals.test_record_resolves_to_a_non_regular_file` (4 subtest failures)
  - `test_delete_store.TestRefusals.test_record_resolves_to_a_symbolic_link` (4 subtest failures)
  - `test_delete_store.TestRefusals.test_shards_table_cannot_be_read` (8 subtest failures)
  - `test_delete_store.TestRefusals.test_two_serials_share_a_file` (8 subtest failures)
  - `test_delete_store.TestRefusals.test_unusable_record` (16 subtest failures)
  - `test_delete_store.TestResumeRelaxesOneThingOnly.test_a_symbolic_link_is_still_refused` (4 subtest failures)

**(v) The file-problem `raise` prefixed again with the same literal as (i) and (ii) (the orchestrator's addition).**

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -2673,7 +2673,10 @@ class ShardedPool:
             files, records, missing_ok=missing_ok
         )
         if len(problems) > 0:
-            raise RuntimeError("; ".join(problems))
+            raise RuntimeError(
+                f'Cannot open sharded datastore "{str(primary)}": '
+                + "; ".join(problems)
+            )
 
         return files, records
 
```

Failed (1 failures and subtest failures in 1 tests; the tests the prompt names are in bold; the prompt names test 5 (the missing-shard-file fixture)):
  - **`test_closed_store_refusals.TestTheBareReason.test_read_closed_store_states_no_prefix`** (1 subtest failures)

Every failure of every breakage is in the new module, `test_store_reader`, or
`test_delete_store`; no RunRegistry test fails under any of them.

## Observations not acted on

- **`_read_back`'s refusal is reached by no test.** `_read_back` reads the copy back through
  `_read_closed_store` mid-operation, and `_failure_message` wraps the message as `copy of sharded
  datastore "<src>" to "<dst>" failed at step "read back the destination": RuntimeError: <reason>`.
  The reason used to start with `Cannot copy sharded datastore "<file read back>": `, and no longer
  does. The step names which file was read back, and the message lists the files at the destination.
  The prompt does not change `_failure_message`, and it was not changed.
- **Breakage (iv) fails 54 subtests of `test_delete_store`** besides tests 1 and 4: that module
  asserts the prefix through `delete_store` and `closed_store_files` for every refusal, so it
  already pinned `_plan_deletion`'s wrap.
- **Breakage (v) fails no existing test**, as the orchestrator measured: the new test 5's missing
  shard fixture alone pins the file-problem `raise`.

## State handed to the next prompt

The campaign is closed. `datastore-generic` and `datastore-generic-followup` have no open issue.
The index counts 104.
