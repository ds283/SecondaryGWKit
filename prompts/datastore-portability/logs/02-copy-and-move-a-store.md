# Log 02 — Copy and move a whole store, through one interface on `ShardedPool`

**Prompt:** [`prompts/datastore-portability/02-copy-and-move-a-store.md`](../02-copy-and-move-a-store.md)
**Commit:** *(this commit)* — "Copy and move a closed sharded store under a new name"
**Model:** Claude Opus 5.5
**Date:** 2026-09-24
**Result:** COMPLETE. No §7 stop condition was met (the schema check is below). Three refusals
beyond P6's list were needed for the interruption property to hold; they are "Deviations" items
1–2, and the orchestrator should check them.

## What shipped

**`Datastore/shard_paths.py`** (+17): **P5**, `shard_file_name(primary, serial) -> str` (`:65`),
the bare name `<stem>-shard<serial:04d><suffix>`. It is the only place in production code that
spells the pattern. The module is still standard library only.

**`Datastore/SQL/ShardedPool.py`** (+489 / −45).

- **P5.** The constructor's new-store branch names shard *i* as
  `self._primary_file.parent / shard_file_name(self._primary_file, i)` (`:126–130`). The
  resulting `Path` is the same as the old `self._primary_file.with_stem(f"{stem}-shard{i:04d}")`.
- **Read-and-check, factored once.** The body of the `shards` loop in `_read_shard_data` moved,
  unchanged, into the static `_resolve_shard_rows(primary_file, rows, shard_db_files,
  shard_records)` (`:534`). The body of `_check_shard_files` moved into the static
  `_shard_file_problems(shard_db_files, shard_records) -> List[str]` (`:579`). That function
  calls `shard_file_problem` for each shard and holds the duplicate-file check.
  `_read_shard_data` passes its own `self._shard_db_files` / `self._shard_records`, so they are
  filled in place exactly as before, the `!!` legacy notice is printed by the same code, and
  every error message is character-for-character the same. `_check_shard_files` formats the
  problems with its existing `Cannot open sharded datastore "…": ` prefix. Prompt 01's 27 tests
  pass unmodified.
- **P6.** `ShardedPool.copy_store(src, dst)` (`:621`) and `ShardedPool.move_store(src, dst)`
  (`:643`) are static methods. Both call `_relocate_store` (`:854`). That in turn calls:
  - `_plan_relocation` (`:706`): every refusal, before anything is written;
  - `_read_closed_store` (`:668`): opens a primary `mode=ro` with stdlib `sqlite3`, and routes its
    rows through `_resolve_shard_rows` and `_shard_file_problems`. It is used for the source, for
    the copy's temporary primary, and for the finished destination;
  - `_write_shard_records` (`:816`): one transaction of `UPDATE shards SET filename = ? WHERE
    serial = ?`, one per serial. It requires one row per serial and exactly as many rows as
    serials. The connection is `mode=rw`, so it can never create a file;
  - `_read_back` (`:842`): the destination's rows must be exactly its bare names, resolving to
    exactly its files;
  - `_failure_message` (`:910`): names the step that failed and the store files that exist at
    each end, and adds the copy-then-delete remedy for `EXDEV`.

  No Ray, no actor, no instance. There is no `unlink`, `remove`, `rmtree` or `rmdir` anywhere in
  the new code. The only write-mode connection is to the destination (the copy's temporary
  primary, or the moved primary).
- Nothing in `ShardedPool` mentions a sidecar or the registry
  (`git grep -n -i "manifest\|registry" -- Datastore/SQL/` is empty).

**`tools/sharded_store.py`** (new, 80 lines): **P7**. It runs as `python tools/sharded_store.py
{copy,move} SRC DST` and calls `ShardedPool.copy_store` / `move_store` and nothing else. On
success it prints the destination's shard map. On refusal or failure it prints `!! <reason>` to
stderr and exits 1. It puts its repository root on `sys.path` the way `tools/shard_key_audit.py`
does, and it never calls `ray.init`. The module docstring, which is also the `--help` text, makes
the three statements P7 requires: it handles the primary and shards only and leaves a
`<stem>.manifest.json` where it is; it cannot tell whether the store is open, because a
rollback-journal store leaves no file while idle, so checking is the caller's job; and a
registry-level tool is the place that does both (README §6.3–§6.4).

**`Datastore/tests/shard_store_fixtures.py`**: `write_new_store` now names shards with
`shard_file_name`. `tree_state(root)` is new. It maps every entry under a root to (kind,
`st_mtime_ns`, SHA-256), for the "nothing was created, changed or removed" assertions.

**Tests:** 33 new test methods in three modules. None starts Ray, opens anything under `var/`, or
calls the constructor. Prompt 01's three modules are unmodified
(`git diff HEAD~1 -- Datastore/tests/test_shard_paths.py Datastore/tests/test_shardedpool_shard_paths.py Datastore/tests/test_shard_key_audit_copy.py`
is empty).

| Module | Methods | Prompt §3 test |
|---|---|---|
| `test_shard_file_name.py` | 4 | **1**: old names reproduced for 11 (stem, serial, suffix) cases, checked against literals and the old constructor expression; bare and directory-independent; the fixtures' new store is named by it; the pattern appears in production code only in `shard_paths.py` |
| `test_copy_move_store.py` | 24 | **2** copy to a new directory and stem, plus "no other table changed" (`iterdump` equal apart from `shards` rows) · **3** legacy source, records into a populated *X*, files in *B*, copied to *C* · **4** move in the three layouts, and a legacy source in each · **5** ten refusal tests (each case a subtest, 55 refusals asserted) · **6** interruption: copy (7 points × 3 layouts × 2 source kinds) and move (5 × 3 × 2), both rewrites made to fail part-way by a trigger, and the cross-filesystem move · **7** manifest and unrelated file, after a copy and after a move |
| `test_sharded_store_script.py` | 5 | **8**: copy and move from an unrelated directory with no `PYTHONPATH`; refusal exits non-zero with no writes; Ray imported, never initialised; `--help` carries the three statements |

## Where the naming function and the read-and-check live, and why

**`shard_file_name` is in `Datastore/shard_paths.py`**, beside the resolver. The prompt asked for
it there, and it has the same constraint as the resolver: it must be importable without `ray`.
What a shard is called and where a shard lives are the two halves of one convention, so they
belong in one module. The constructor, `_plan_relocation` and the fixtures all call it.

**The read-and-check is two static methods on `ShardedPool`** (`_resolve_shard_rows` and
`_shard_file_problems`). The instance methods `_read_shard_data` and `_check_shard_files` call
them, and so does `_read_closed_store`. I kept them in `ShardedPool` and did not move them to
`shard_paths.py`, for three reasons:

- they produce `ShardedPool`'s own error messages, which prompt 01's tests pin;
- they print the `!!` notice in `ShardedPool`'s voice;
- the audit tool, the other client of `shard_paths.py`, does not need them. It checks only
  shard 0.

Static rather than instance methods, because the interface has no instance and none of the
constructor's arguments (shard-key type, table lists). Being static lets both callers share one
body without an `object.__new__` instance in production code.

The one thing not shared is the `SELECT serial, filename FROM shards` itself. The constructor
issues it through its SQLAlchemy engine inside the transaction that also reads the key type and
the table lists. The interface issues it through stdlib `sqlite3` opened `mode=ro`, because
SQLAlchemy's engine in `_create_engine` opens read-write, and the prompt requires a source to be
opened only `mode=ro`. The rows then go through the same code. Deviations, item 6.

## The schema check behind the first stop condition

The condition is met if anything in a primary other than `shards.filename`, or anything in a
shard, records a store's path, name or stem. I checked read-only (`sqlite3` `mode=ro`) on the
sweep store's primary and all four shards, and on the A3 store's primary and all four shards.
In each file:

1. **The schema.** Primaries have 5 tables (`shards`, `shard_key_config`, `shard_keys`,
   `replicated_tables`, `sharded_tables`), 2 indexes, no views and no triggers. Shards have 37
   tables, 125 indexes, no views and no triggers. The only text columns are `label`, `name`,
   `metadata`, `break_point_kind`, `numeric_policy`, `source_grid_digest`, `filename`, `table`,
   `key_attr` and `key_type`.
2. **Every distinct text value in every column of every table** (about 79,000 per shard), searched
   for `/`, `\`, `.sqlite`, `shard`, `atol-sweep`, `lambdacdm`, `datastores`, `primarydb` and
   `manifest`.

**Findings.**

- In the primaries, the only hits are `shards.filename` (the four absolute legacy paths) and
  `replicated_tables.table` = `LambdaCDM` (a class name).
- In the shards, the only hit is `QuadSourceIntegral.label` containing `atol-sweep`: 43 / 4 / 31 /
  28 rows in the sweep store's shards, and 24 / 0 / 18 / 12 (54 in all) in the A3 store's. These
  labels are `<--job-name>-QuadSourceIntegral-k…-q…-r…-zresponse…-<timestamp>`, with job names
  `handover-atol-sweep-a1e-22-r1e-08` and the like (`main.py:3427`). The same prefix sits in the
  A3 store, whose stem is `handover-A3-baseline-lambdacdm`. Those are the 54 rows of the
  2026-09-23 incident. **A label records which job computed a row, not which store holds it.**
  It stays true after a rename, and nothing reads it to find a file. The other labels carry the
  job name `handover-A3-baseline-lcdm`, which is not either store's stem.
- The `main.py:290` label, which contains `primarydb-"{args.database}"`, goes to the
  `ProfileAgent`'s own `--profile-db` database, not to the store.
- No text value in any shard contains `/`, `.sqlite`, `shard`, `datastores` or either store's
  stem. The SQLite file format stores no file name in the file.

**Not met.** `shards.filename` is the only stored reference to where a store's files are.

## The order of operations and the interruption table

The prescribed order is used for both operations, with the shards taken in ascending serial order
(serial 0 first). The copy writes its primary as `<dst name>.incomplete-copy` and `os.replace`s it
last. The copy also reads the temporary primary back through the read-and-check before the
rename. The table's claim is this: *after any step, every store a later opener could find, at the
source or the destination, either opens against the right files, or is refused by prompt 01's
check, or is a destination with shards and no primary, which the constructor refuses
(`ShardedPool.py:132–135`, "Primary database is missing, but shard … already exists").* That
last guard catches every such state because it tests shard 0 for any `--shards` value, and shard
0 is the first file written (Deviations, item 1).

Every row was exercised for **both source kinds** (new-style bare records, and legacy absolute
records naming a directory *X* that exists and holds same-named files) and **all three layouts**
(same directory with a new stem; new directory with the same stem; new directory with a new
stem). That is 6 combinations per row, as subtests of the named test. "Opens" means
`_read_shard_data` + `_check_shard_files` accept the primary, resolve exactly the expected paths,
and shard *i* holds the bytes the source's shard *i* held.

**Copy** (source never written):

| Row | Process dies after | Source | Destination name | Temporary primary | Backed by |
|---|---|---|---|---|---|
| C0 | any pre-check | opens (untouched) | absent | absent | `TestRefusals` (10 tests, every one asserts `tree_state` unchanged) |
| C1 | creating the destination directory | opens | absent: no primary, no shard, the same as an unused name | absent | `TestInterruption.test_copy`, row C1 |
| C2 | part of shard #0's copy | opens | no primary, partial shard #0: **guard** | absent | row C2 (`copy2` writes half the file, then raises) |
| C3 | shard #0, or #0–#1, copied | opens | **guard** | absent | rows C3 (both) |
| C4 | every shard copied, before the primary | opens | **guard** | absent | row C4 |
| C5 | the primary copied to its temporary name, before the rewrite commits (a rewrite interrupted part-way rolls back to here) | opens | **guard** | the source's rows | row C5; `test_copy_rewrite_is_one_transaction` (trigger aborts the 2nd `UPDATE`; temp rows equal the source's) |
| C6 | the rewrite committed and read back, before `os.replace` | opens | **guard** | the destination's rows | row C6 (`os.replace` raises) |
| C7 | `os.replace` | opens | **opens** against its own files, bare rows | gone | `TestCopy`, `TestCopyOfALegacySource`, `TestNothingElseIsTouched.test_copy`, the script tests |

**Move:**

| Row | Process dies after | Source | Destination name | Backed by |
|---|---|---|---|---|
| M0 | any pre-check | opens (untouched) | absent | `TestRefusals` |
| M1 | creating the destination directory (also: a cross-filesystem first rename) | opens | absent | `TestInterruption.test_move` row M1; `test_move_across_filesystems_fails_before_anything_moves` |
| M2 | shard #0, or #0–#1, renamed | **refused** by prompt 01's check (renamed shards missing) | **guard** | rows M2 (both) |
| M3 | every shard renamed, before the primary | **refused** (every shard missing) | **guard** | row M3 |
| M4 | the primary renamed, before the rewrite commits (a rewrite interrupted part-way rolls back to here) | no primary and no shard at the source names | the source's rows read in the new directory: **same directory, new stem → refused** (the old names were renamed away); **new directory, same stem → opens against its own files** (old names = new names); **new directory, new stem → refused** (the old names do not exist there, which the pre-check of Deviations item 2 guarantees) | row M4; `test_move_rewrite_is_one_transaction` |
| M5 | the rewrite committed | no primary and no shard at the source names | **opens** against its own files, bare rows | `TestMove` (4 tests), `TestNothingElseIsTouched.test_move`, `test_move_succeeds` |

Two notes on the table.

- **At the source after M4.** Nothing is left for an opener to find. An opener given the old
  primary path would create a new, empty store there, which is what it does at any unused name
  and what it does after any completed move. That is the constructor's contract for a path with
  nothing at it, not a mixed state.
- **The temporary primary is not a store name**, so it is outside the property. It is neither
  the source's nor the destination's primary name. It is called `<dst>.incomplete-copy`, it
  appears in the failure message, and a rerun refuses because of it. In a same-directory copy,
  if someone opened it by that name through `main.py` before the rewrite committed, it would
  resolve to the source's shards, as any byte copy of a primary in that directory would. That
  cannot be made refused within the prescribed order without a temporary directory, which would
  leave a directory entry behind. Observations, item 4.

A real crash during the rewrite leaves a hot `-journal` beside the primary being rewritten. The
next SQLite opener rolls that back to the pre-rewrite rows, which is state C5 or M4. The tests
reach the same rows through a statement that aborts inside the transaction. They do not kill a
process part-way through a write.

## The deliberate-breakage record

Each mutation was applied to the staged final tree, the whole `Datastore/tests` suite was run
(`PYTHONPATH=. ./venv/bin/python -m unittest discover -s Datastore/tests -t .`), and the file was
restored with `git checkout --` from the index. `git diff --stat` was checked empty afterwards.
The diffs below are exactly as applied, taken with `git diff` against the blobs that this commit
contains, and each was checked with `git apply --check`. Unmutated: `Ran 70 tests … OK`.

**(i) skip the `shards` rewrite.** `Ran 70 … FAILED (failures=12, errors=11)`.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 5ce685a..15a6d94 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -878,7 +878,6 @@ class ShardedPool:
                 shutil.copy2(plan.src_primary, plan.temp_primary)
 
                 step = "rewrite the temporary primary's shards rows"
-                ShardedPool._write_shard_records(plan.temp_primary, plan.dst_records)
 
                 step = "read back the temporary primary"
                 ShardedPool._read_back(plan, plan.temp_primary)
@@ -898,7 +897,6 @@ class ShardedPool:
                 os.rename(plan.src_primary, plan.dst_primary)
 
                 step = "rewrite the destination primary's shards rows"
-                ShardedPool._write_shard_records(plan.dst_primary, plan.dst_records)
 
             step = "read back the destination"
             return ShardedPool._read_back(plan, plan.dst_primary)
```

Failed: `TestCopy.test_copy_to_a_new_directory_and_stem_reads_its_own_files`,
`TestCopy.test_no_other_table_is_changed`,
`TestCopyOfALegacySource.test_legacy_source_copied_to_a_new_stem_reads_its_own_files`,
`TestMove.test_same_directory_new_stem`, `TestMove.test_new_directory_new_stem`,
`TestMove.test_legacy_source_in_each_layout` (all 3 layouts),
`TestRefusals.test_move_into_a_directory_holding_a_source_shard_name` (its closing copy),
`TestNothingElseIsTouched.test_copy`, `.test_move`, `TestInterruption.test_copy` (row C5),
`TestInterruption.test_copy_rewrite_is_one_transaction` (all 6), `TestInterruption.test_move`
(row M4), `TestInterruption.test_move_rewrite_is_one_transaction`, and
`test_sharded_store_script`'s `test_copy_succeeds_…`, `test_move_succeeds` and
`test_ray_is_imported_but_never_initialised`. The read-back is what catches it in production:
every copy ends at "read back the temporary primary", with no destination primary (state C6).
`TestMove.test_new_directory_same_stem` passes, correctly. There the old names *are* the new
names, and only the literal rows differ. `_read_back` then refuses legacy rows, but a new-style
source's rows are already right.

**(ii) write the rows as absolute paths.** `Ran 70 … FAILED (failures=3, errors=12)`.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 5ce685a..2fc5efd 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -824,7 +824,7 @@ class ShardedPool:
                 for serial, name in sorted(records.items()):
                     cursor = conn.execute(
                         "UPDATE shards SET filename = ? WHERE serial = ?",
-                        (name, serial),
+                        (str(primary.parent / name), serial),
                     )
                     if cursor.rowcount != 1:
                         raise RuntimeError(
```

Failed: `TestCopy` (both), `TestCopyOfALegacySource`, `TestMove` (all four, including
`test_new_directory_same_stem`), `TestRefusals.test_move_into_a_directory_holding_a_source_shard_name`,
`TestNothingElseIsTouched` (both), and the three succeeding script tests. The absolute rows
*resolve* to the right files, because the resolver reads an absolute record by its name. So it
is `_read_back`'s comparison with the bare names, and the tests' `stored_records(dst) ==
bare_names`, that detect the mutation.

**(iii) drop the source-journal check.** `Ran 70 … FAILED (failures=12)`.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 5ce685a..050f2d0 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -729,7 +729,6 @@ class ShardedPool:
                         f'{what} "{str(path)}" has "{str(journal)}" beside it, so it was not closed cleanly (a copy made without that file is corrupt)'
                     )
 
-        refuse_journals(src_primary, "the source primary")
 
         # the source's shard records, read and checked exactly as the constructor reads them
         try:
@@ -747,8 +746,6 @@ class ShardedPool:
                 f'the source primary "{str(src_primary)}" records no shard #0 (serials {sorted(src_files)}), so an interrupted {mode} could not be told apart from an unused name'
             )
 
-        for serial, path in sorted(src_files.items()):
-            refuse_journals(Path(path), f"source shard #{serial}")
 
         # the destination primary: a new file, and not the source
         dst_given = Path(dst)
```

Failed: `TestRefusals.test_source_journal_files`, all 12 subtests (3 journal names × primary or
shard 2 × copy or move). In the suite, the later subtests also see the files the first unrefused
copy wrote, so I probed each case on a fresh store under the mutation (scratch
`a02_iii_probe.py`). With a `-journal` beside the *primary*, SQLite's `mode=ro` open still fails,
with `attempt to write a readonly database`: a refusal, but for the wrong reason, and one that
never names the journal. In the other **10 of 12** cases (`-wal` or `-shm` beside the primary,
and any of the three beside a shard, which the interface never opens), the copy or move **went
ahead**. So the explicit check is the only thing that catches them.

**(iv) drop the destination-journal check.** `Ran 70 … FAILED (failures=9, errors=2)`.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 5ce685a..d14f99b 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -783,7 +783,7 @@ class ShardedPool:
         for path in [dst_primary, *[dst_files[s] for s in sorted(dst_files)]] + (
             [temp_primary] if temp_primary is not None else []
         ):
-            for name in [path, *ShardedPool._journal_paths(path)]:
+            for name in [path]:
                 if os.path.lexists(name):
                     taken.append(f'"{str(name)}"')
         if len(taken) > 0:
```

Failed: `TestRefusals.test_destination_names_or_their_journals_exist` (the four journal cases,
copy and move) and `TestRefusals.test_copy_temporary_name_or_its_journal_exists` (the temporary
primary's journal): `RuntimeError not raised`. The two ERRORs are the test's own clean-up
meeting what the unrefused operation did. One is `symlink_to` finding the `copy-shard0001.sqlite`
that the copy wrote. The other is worth recording: `unlink` of the placeholder
`copy.sqlite.incomplete-copy-journal` raised `FileNotFoundError`, because **SQLite had consumed
it.** The rewrite's `mode=rw` open took the stale file beside the temporary primary for that
database's journal, and deleted it. This is the hazard P6 names ("a stale journal at a
destination name would be replayed into the new file by the next opener"), observed. The cases
where the file itself exists still pass, as they should, because the name checks remain.

**(v) in copy, write the primary directly to its final name.** `Ran 70 … FAILED (failures=12,
errors=6)`.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 5ce685a..61da3a9 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -873,19 +873,12 @@ class ShardedPool:
                     no_overwrite(plan.dst_files[serial])
                     shutil.copy2(plan.src_files[serial], plan.dst_files[serial])
 
-                step = "copy the primary to its temporary name"
-                no_overwrite(plan.temp_primary)
-                shutil.copy2(plan.src_primary, plan.temp_primary)
-
-                step = "rewrite the temporary primary's shards rows"
-                ShardedPool._write_shard_records(plan.temp_primary, plan.dst_records)
-
-                step = "read back the temporary primary"
-                ShardedPool._read_back(plan, plan.temp_primary)
-
-                step = "rename the temporary primary to the destination primary"
+                step = "copy the primary to its final name"
                 no_overwrite(plan.dst_primary)
-                os.replace(plan.temp_primary, plan.dst_primary)
+                shutil.copy2(plan.src_primary, plan.dst_primary)
+
+                step = "rewrite the destination primary's shards rows"
+                ShardedPool._write_shard_records(plan.dst_primary, plan.dst_records)
 
             else:
                 for serial in serials:
```

Failed: `TestInterruption.test_copy` rows C5 and C6 in all 6 combinations. For C5, the
destination primary exists holding the source's rows, where the guard was expected. For C6, no
`os.replace` is called, so no failure is injected, and a "died before the rename" state never
happens. Also failed: `TestInterruption.test_copy_rewrite_is_one_transaction`, all 6: there is no
temporary primary, and the destination primary holds the source's rows. In the same-directory
layout, that destination primary resolves to **the source's shards**. That is exactly the state
the temporary name exists to prevent. The happy-path copy tests pass, which is why this mutation
needed the interruption tests.

**(vi) change the zero-padding in `shard_file_name`.** `Ran 70 … FAILED (failures=13,
errors=2)`.

```diff
diff --git a/Datastore/shard_paths.py b/Datastore/shard_paths.py
index 58e20b0..315798c 100644
--- a/Datastore/shard_paths.py
+++ b/Datastore/shard_paths.py
@@ -72,7 +72,7 @@ def shard_file_name(primary: Union[str, Path], serial: int) -> str:
     the primary (``primary.parent / shard_file_name(primary, serial)``). This function does no I/O.
     """
     primary = Path(primary)
-    return primary.with_stem(f"{primary.stem}-shard{serial:04d}").name
+    return primary.with_stem(f"{primary.stem}-shard{serial:03d}").name
 
 
 def resolve_shard_path(primary: Union[str, Path], stored: str) -> Path:
```

Failed: `test_shard_file_name.test_reproduces_the_constructors_old_names` (9 of 11 cases; 9999
and 12345 are unaffected by a narrower pad),
`…test_a_new_store_written_through_the_fixtures_is_named_by_it`,
`TestCopy.test_copy_to_a_new_directory_and_stem_reads_its_own_files` (its literal
`copy-shard0000.sqlite`), and four of **prompt 01's** tests, whose literal `SHARD_NAMES` no
longer match the fixtures: `TestRoundTrip.test_new_store_records_bare_names_and_survives_a_move`,
`…test_renaming_the_primary_alone_keeps_its_shards`,
`TestFailClosed.test_missing_shard_of_a_new_store_is_refused_by_name`, and
`…test_symlinked_shard_is_refused`. Tests that name files only through `shard_file_name` still
pass under this mutation. That is expected: a consistent rule is still consistent. The literals
are what pin the rule to the old names.

## The real-store demonstration (§4)

All work was in `var/portability-check-02/`. **The script, `ShardedPool` and `main.py` were never
pointed at an original or at the backup.**

**1. Before.** `python -m RunRegistry list` showed 7 runs: 5 done or failed, 2 unknown legacy
runs, none `running`. Free disk space was 15 GB. I took a read-only snapshot (`sqlite3`
`mode=ro`) of the three stores: every file's `st_mtime_ns` and size, the per-table row counts of
every primary and shard, the primaries' SHA-256, and a search for `-journal` / `-wal` / `-shm`
files (none).

**2. The hand-made source.** `cp -p var/datastores/handover-atol-sweep.sqlite` and its four shards
into `var/portability-check-02/src/`, under their own names. Its primary's records were the four
absolute paths into `var/datastores/`: a legacy record naming another store that exists. Then,
with `sqlite3` on **the hand-made copy's shard 0 only**, I ran `DELETE FROM QuadSourceIntegral
WHERE serial = 329386`, the highest serial in that shard (label
`handover-atol-sweep-a1e-32-r1e-08-QuadSourceIntegral-k3.05e+07-q3.05e+07-r3.05e+07-zresponse0.1-2026-09-24T00:17:34`).
That took the shard's `QuadSourceIntegral` count from **1,955 to 1,954**. I then recorded the
hand-made source's five files (mtime_ns, size, SHA-256).

**3. Copy with the script,** run from `/` with no `PYTHONPATH`:

```
$ python tools/sharded_store.py copy var/portability-check-02/src/handover-atol-sweep.sqlite var/portability-check-02/dst/pcopy.sqlite
!! Primary database ".../portability-check-02/src/handover-atol-sweep.sqlite" records 4 shard(s) by legacy absolute path in
   ".../var/datastores"; reading them as siblings in ".../portability-check-02/src" instead (stored records not rewritten)
>> copy: ".../portability-check-02/dst/pcopy.sqlite" with 4 shards
>>   shard #0: .../portability-check-02/dst/pcopy-shard0000.sqlite
>>   shard #1 … #3: likewise
exit 0
```

- The destination's `shards` rows were `[(0, 'pcopy-shard0000.sqlite'), (1,
  'pcopy-shard0001.sqlite'), (2, 'pcopy-shard0002.sqlite'), (3, 'pcopy-shard0003.sqlite')]`.
- The hand-made source's five files were identical in mtime_ns, size and SHA-256 (`cmp` of the
  before and after records). Its primary's hash stayed `c3cda49d8e35…`, the original's, with its
  legacy rows.
- `dst/` held exactly five files: `pcopy.sqlite` and `pcopy-shard000{0..3}.sqlite`. No
  `.incomplete-copy`, no journal.

**4. Audit tool on `pcopy.sqlite`,** run from `/` with no `PYTHONPATH`:

```
>> cross-file check against shard #0: .../portability-check-02/dst/pcopy-shard0000.sqlite (record 'pcopy-shard0000.sqlite')
>> 'wavenumber' table (shard #0) row count: 8
VERDICT: OK -- no inconsistency found in .../portability-check-02/dst/pcopy.sqlite.
```

It attached `pcopy-shard0000.sqlite`.

**5. `main.py --database var/portability-check-02/dst/pcopy.sqlite --inventory
--no-prune-unvalidated --shards 4 --ray-address local`**, run as a script. It exited 0 in 11.5 s
and printed `>> Opened existing sharded datastore ".../dst/pcopy.sqlite" with 4 shards`. There
was no `!!` line, because the rows are bare now. The original's column is from step 1's snapshot:
replicated tables from shard 0, where all four shards agree; sharded tables summed over the four
shards.

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

**27 of 28 tables equal the original's, and `QuadSourceIntegral` is exactly one lower.** That row
is the discriminator: the inventory read the copy's files, which carry the hand-made source's
edit, and not the originals', which do not.

**6. Move with the script** to `var/portability-check-02/moved/pmoved.sqlite` (a new directory
and a new stem), run from `/tmp` with no `PYTHONPATH`. It exited 0, and the rows are
`[(0, 'pmoved-shard0000.sqlite'), … (3, 'pmoved-shard0003.sqlite')]`. `dst/` was left **empty**,
the directory itself remaining, since nothing is ever deleted. `moved/` held exactly five files.
The same `main.py --inventory` command on `pmoved.sqlite` exited 0 and gave the column above:
the same counts as step 5. The two inventory bodies are identical line for line (`diff` of the
report text).

**7. The originals were untouched.** I re-took step 1's snapshot, and `cmp` with the first gave
**identical**: 18 entries, and no journal files before or after. The hand-made source was also
still identical at the end.

| File | mtime (before = after) | Size | Rows, all tables | SHA-256 (primaries) |
|---|---|---:|---:|---|
| `handover-atol-sweep.sqlite` | 2026-09-23T12:41:04 | 32,768 | 41 | `c3cda49d8e35…` |
| `handover-atol-sweep-shard0000.sqlite` | 2026-09-24T02:58:24 | 87,482,368 | 596,542 | |
| `handover-atol-sweep-shard0001.sqlite` | 2026-09-23T23:00:42 | 87,334,912 | 593,534 | |
| `handover-atol-sweep-shard0002.sqlite` | 2026-09-24T02:58:24 | 87,814,144 | 590,516 | |
| `handover-atol-sweep-shard0003.sqlite` | 2026-09-24T02:58:24 | 86,994,944 | 599,069 | |
| `handover-A3-baseline-lambdacdm.sqlite` | 2026-09-20T21:03:10 | 32,768 | 41 | `fdbe93a54108…` |
| `handover-A3-baseline-lambdacdm-shard0000…3` | 2026-09-23T11:22:04 / 11:11:00 / 11:22:04 / 11:22:04 | 87,367,680 / 87,302,144 / 87,711,744 / 86,892,544 | 596,351 / 593,493 / 590,385 / 598,908 | |
| backup primary | 2026-09-20T21:03:10 | 32,768 | 41 | `fdbe93a54108…` |
| backup `-shard0000…3` | 2026-09-21T04:32:10 / 04:32:24 ×3 | 87,044,096 / 87,302,144 / 87,588,864 / 86,609,920 | 595,885 / 593,487 / 590,199 / 598,522 | |

**8. Deleted.** `rm -rf var/portability-check-02`. `var/` holds `.DS_Store`,
`bootstrap-a3-resume.log`, `datastores/` and `runs/`, as before. Nothing from §4 is in the commit.
No Ray process remained. The throwaway scripts are in session scratch space (`agent02/a02_*`).

## Deviations from the prompt

1. **Two extra refusals: a source that records no shards, and one with no shard #0.** Class:
   **STRUCTURALLY REQUIRED.** In every interrupted state, the destination either has nothing or
   has shards and no primary. The table rests on the constructor refusing the second case. Its
   guard tests the names `shard_file_name(dst, i)` for `i in range(--shards)`, which always
   includes 0. So the guard catches every partial state **only if shard 0 is the first file
   written**. For a source whose serials do not include 0 (the creator always writes
   `range(n)`, so only a hand-built store could be like this), an interrupted copy would leave
   shards the guard does not look for. The constructor would then create a new empty store
   beside them, which the prompt forbids. Refusing such a source before any write is within
   scope, and it keeps "the serials are those in the table, not `range(n)`": the names come from
   the table's serials. A store with no shard rows is refused too, because the constructor
   refuses to open one ("No shard records were read").
2. **A move is refused if the destination directory holds a file with the name of a source
   shard.** Class: **STRUCTURALLY REQUIRED.** In the prescribed move order, the primary is
   renamed before its rows are rewritten. In between (row M4), the destination primary holds the
   source's records, which the resolver reads *by name in the destination directory*. For a new
   directory with a new stem, those names are the old ones. If the destination directory held
   files of those names, say another store of the source's stem, that state would **open
   against another store's files**. The pre-check (`_plan_relocation`, the last block) makes M4
   always refused in that layout. A copy does not need the check, because its destination
   primary appears only with its final rows. `test_move_into_a_directory_holding_a_source_shard_name`
   shows the refusal, and that a copy into the same directory proceeds. I considered the other
   order (rewrite the source primary's rows before renaming it). It has the mirror-image
   exposure in the *source* directory, and it writes the source, so it is not stronger.
3. **The copy reads its temporary primary back through the read-and-check before `os.replace`,
   and a move or copy reads the finished destination back.** Class: **IMPLEMENTATION CHOICE.**
   If the rewrite ever wrote something other than the destination's bare names, the destination
   primary never appears (state C6). The return value is the map from that read-back, as P6
   asks. Mutations (i) and (ii) show the read-back catching both defects.
4. **Every destination name is checked again immediately before it is written.** Class:
   **IMPLEMENTATION CHOICE.** `shutil.copy2`, `os.rename` and `os.replace` all replace an
   existing file silently. The plan checks every name once. A second `lexists` just before each
   write narrows the window in which something else could create one. It does not close it,
   because POSIX has no exclusive rename.
5. **A symlinked source primary is refused**, by the same `shard_file_problem` that refuses a
   symlinked shard. Class: **IMPLEMENTATION CHOICE.** Moving a symlink's target would leave the
   link dangling, and copying through one hides where the store is. "Not a regular file" is read
   as the resolver's definition. A dangling symlink at a destination name counts as taken
   (`os.path.lexists`).
6. **The interface uses stdlib `sqlite3`, not the SQLAlchemy engine,** to read the source's rows
   (`mode=ro`) and to rewrite the destination's (`mode=rw`, so a missing file is an error and not
   a new database). Class: **IMPLEMENTATION CHOICE.** `_create_engine` opens read-write, and it
   builds instance state the interface has no use for. The `SELECT` is one line. The reading
   that matters, resolution and checking, is the shared code.
7. **The fixtures now import `Datastore.shard_paths`.** Class: **STRUCTURALLY REQUIRED** (P5:
   "Make it call the function"). Prompt 01's two `ShardedPool`-side test modules say in their
   docstrings that they "deliberately do not import `Datastore.shard_paths`", so that they load
   against the pre-01 tree. They still do not import it themselves, but their fixture now does.
   So prompt 01's deliberate-breakage recipe (move `shard_paths.py` aside) no longer loads them.
   Those modules are unmodified, as required. Observations, item 3.
8. **33 tests, beyond §3's minimum**, and a `tree_state` fixture. Class: **IMPLEMENTATION
   CHOICE.** Each refusal case is a subtest, so a mutation reports every case it breaks.
9. **§4's `main.py` runs added `--shards 4 --ray-address local`,** as prompt 01's did. Class:
   **IMPLEMENTATION CHOICE.** The default address, `auto`, looks for a running cluster.
10. **My read-only store snapshot included the two stores' `.manifest.json` files** (mtime, size,
    SHA-256), in addition to what §4.1 lists. Class: **UNINTENDED DRIFT.** I globbed them into
    the "untouched" check without thinking about §6's "does not … read … any sidecar file". That
    line is about the interface and the script, which never touch one. The snapshot is a
    verification artefact in scratch space, not code. The effect is nil: they were read-only
    hashes, and they are identical before and after. I am recording it because it is not what
    §4.1 asked for.

The diff touches `Datastore/SQL/ShardedPool.py`, `Datastore/shard_paths.py`, the new
`tools/sharded_store.py`, `Datastore/tests/`, and this campaign's board and log plus
`docs/OPEN_ISSUES.md`. It does not touch `Datastore/SQL/Datastore.py`, `main.py`,
`docs/handover/`, `RunRegistry/`, the object factories, or the `shards` table's columns.

## Verification performed

1. **Suites**, each run as `PYTHONPATH=. ./venv/bin/python -m unittest discover -s
   <package>/tests -t .`, with `THREE_BESSEL_DIAGNOSTIC_PLOTS` unset:

   | Suite | Baseline (`1770a61`) | Now |
   |---|---|---|
   | AdaptiveLevin | 32 OK | 32 OK |
   | ComputeTargets | 552, FAILED (failures=1): the `test_wall_time_per_object` flake | 552, FAILED (failures=1): the same flake, `0.06103… not less than or equal to 0.06`; the module alone fails it too (`0.06090…`), as at the baseline |
   | CosmologyModels | 39 OK | 39 OK |
   | Datastore | 37 OK | **70 OK** (+33, exactly the methods added) |
   | LiouvilleGreen | 148 OK (skipped=1) | 148 OK (skipped=1) |
   | RunRegistry | 38 OK | 38 OK |

2. `./venv/bin/python -m black --check .`: clean.
3. **One naming rule.** `git grep -n -e '-shard{' -- Datastore/ tools/` finds these:
   `Datastore/shard_paths.py:75`, the rule itself; the literal expected names in prompt 01's
   test modules; and, in `test_shard_file_name.py`, a docstring quoting the old expression and
   the old expression used as an oracle (`:52`). `ShardedPool.py` and `shard_store_fixtures.py`
   no longer match. `test_the_pattern_is_written_once_outside_the_tests` pins this.
4. **Layering.** `git grep -n -i "manifest\|RunRegistry" -- Datastore/ tools/sharded_store.py`
   finds `tools/sharded_store.py:17,21` (the required docstring and `--help` statements) and the
   two test modules that §3 tests 7 and 8 require: a `.manifest.json` written beside a source to
   show it is left alone, and assertions on the `--help` text. There is nothing in
   `Datastore/SQL/`.
5. **Ray.** `test_ray_is_imported_but_never_initialised` runs the script through `runpy` in a child
   interpreter, from an unrelated directory with `PYTHONPATH` unset. After the script returns,
   the child prints `'ray' in sys.modules` (True) and `ray.is_initialized()` (False). It is
   checked in the same process that ran the script, and no Ray call is needed to ask.
6. `python -m RunRegistry list` before any store was touched: nothing `running`.

## Observations not acted on

1. **`docs/handover/quadsource_atol_sweep.py` spells the shard naming pattern twice more**
   (`:583`, `:711`). They agree with `shard_file_name` today, and the rule is pinned by
   `test_reproduces_the_constructors_old_names`. The script is a measurement record, so it was
   not edited (prompt §2, README §1). No issue opened: it is not wrong, and the prompt asked for
   it to be listed here.
2. **The index's header prose is stale.** `docs/OPEN_ISSUES.md` says "Of the 90 above, 87 are
   spread across the boards" while its count line says 95 (94 after this commit). The paragraph
   predates this prompt. I corrected the count line and the one sentence about this campaign,
   and left the rest. No issue opened: it is the index's own prose, and whoever next edits the
   header should recount.
3. **Prompt 01's deliberate-breakage recipe no longer replays as written** (Deviations, item 7).
   Its record in `logs/01-…` was correct for the tree it ran on. Replaying it now means also
   reverting `shard_store_fixtures.py` to `b04671f`. No issue opened: that record is closed
   history, and the orchestrator reproduced it at the time.
4. **The copy's temporary primary, if opened by its own name before the rewrite commits, reads
   the source's shards in a same-directory copy** (the interruption table, second note). It is
   neither the source's nor the destination's name, and it is named to say what it is. No issue
   opened: closing it would need a temporary directory, which the never-delete rule would leave
   behind, and nothing opens a `*.incomplete-copy` file unless a person asks it to.
5. **A move to a new directory leaves the source directory in place**, empty if the store was
   all it held. In §4 this was `dst/`. Likewise, a cross-filesystem move leaves the destination
   directory it created. Both follow from "never deletes", and are for a person to tidy. No
   issue opened.

## State handed to the next prompt

`[01-whole-store-rename-is-unsupported]` is closed on this board's §4. A closed store can now be
copied or moved under a new stem, with `ShardedPool.copy_store` / `move_store` or
`tools/sharded_store.py`, and the destination's rows are bare names. The originals, the backup
and their rows are unchanged. **Prompt 03 stays held** on README §6.4's two decisions.
`[store-sidecar-manifests-have-no-owner]` and `[01-atol-sweep-check-expects-absolute-shard-records]`
are open and untouched. Prompt 03, when written, calls `copy_store` / `move_store`. It must make
its own decision about the sidecar, and about checking that no `running` run names the store,
which is the check these methods say is the caller's job.
