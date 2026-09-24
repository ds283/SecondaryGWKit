# Log 01 — One schema builder, and a read-only reader over a closed store

**Prompt:** [`prompts/store-fingerprint/01-a-read-only-store-reader.md`](../01-a-read-only-store-reader.md)
**Commit:** *(this commit)* — "Add one schema builder and a read-only store reader"
**Base:** `417c6471c08b405e718d4f9d8b244ae7e022b475` ("Record the user's decisions D1-D3 for the
store fingerprint"), clean
**Model:** Claude Opus 5.5
**Date:** 2026-09-25
**Result:** COMPLETE. Items F1, F2 and F3 shipped, and `[00-build-schema-reads-registration-before-its-none-check]`
is closed. No §7 stop condition arose. All six deliberate-breakage mutations were caught. The §4
demonstration changed no byte of the copy or of any original.

## What shipped

**`Datastore/SQL/schema.py`** (new, 135 lines, imports only `sqlalchemy`): **F1**.
`build_schema(metadata, factories) -> BuiltSchema(tables, records)` is the loop that was
`Datastore._build_schema`, moved, **minus the inserters**. It declares one `Table` per factory into
`metadata`, with the prepended `serial` / `version` / `timestamp` / `stepping` columns in that
order and then the factory's columns, and a schema record per class. It creates nothing. The
records hold every field they held before except `insert`.

- **The fix of `[00-build-schema-reads-registration-before-its-none-check]`.** `registration_data`
  is checked for `None` before anything is read from it. A factory whose `register()` returns
  `None` gets `{"name": ..., "validate_on_startup": False, "table": None}` and no table.

**`Datastore/SQL/Datastore.py`** (+8 / −85): `_build_schema` now calls `build_schema(self._metadata,
self._factories)`, adds the inserter to each record that has a table (`functools.partial(self._insert,
schema, tab)`, as before, and `None` otherwise), and fills `self._tables`, `self._inserters` and
`self._schema` in place and in the same order. `self._metadata` is the `MetaData` that `_create_engine`
made, as before. The duplicate-registration check (`RuntimeWarning` when a class is already in
`self._schema`) stays in the actor and runs before the build. Nothing else in the file changed:
`_ensure_tables`, the drop actions, the pruning and the `version` row are untouched. The factory
map stays where it was (the prompt's §2 F1.4 made moving it optional).

**`Datastore/store_reader.py`** (new, 200 lines): **F2**. `open_read_only(primary)` is a context
manager yielding a `ReadOnlyStore(primary, shards, tables, records)`, with `shard(serial)`. Each
`ReadOnlyShard` has `serial`, `path`, `engine` (read-only), `tables` (the one set of `Table`
objects, shared by every shard), `absent_tables`, `absent_columns` and `extra_columns` (per table
present: declared columns missing from the file, in declaration order, and file columns the code
does not declare, in file order; empty tuples when there are none), and `extra_tables`.

- **Shards are found through `ShardedPool._read_closed_store(primary, "read")`**. That method
  reads `shards` `mode=ro`, resolves every record through `Datastore/shard_paths.py`, and refuses
  a missing, non-regular or symlinked shard. Its refusal reaches the caller unchanged.
- **Every file is opened as `sqlite:///file:{path}?mode=ro&uri=true`** (`read_only_url`). It is
  never opened read-write, and never with `immutable=1`.
- **Journals are refused.** Before the primary is opened, and again for every shard before any
  shard is opened, each name `ShardedPool._journal_paths` gives (`-journal`, `-wal`, `-shm`) is
  checked with `os.path.lexists`. If one exists, the reader raises `RuntimeError` naming both the
  database and the journal file. It repairs nothing.
- **The primary itself** must pass `shard_file_problem` (it must exist, be a regular file, and not
  be a symlink), and it must record at least one shard. Both checks mirror `copy_store`'s
  `_plan_relocation`.
- **The tables are built once**, with `build_schema(sqla.MetaData(), _factories)`. Absent and
  extra tables come from `sqlite_master`, and absent and extra columns from `PRAGMA table_info`.
  They are reported, not refused.
- **There is no write path.** It has no `create_all`, no `_ensure_tables`, no DDL and no DML. It
  never initialises Ray, and never constructs a `Datastore` actor or a `ShardedPool`.
- **Every engine is disposed on exit**, including when a refusal comes after some engines exist.

**`Datastore/tests/real_store_fixtures.py`** (new, 406 lines): **F3**. `build_real_store(directory,
...)` builds a real store with no Ray:
- a primary written by `ShardedPool._write_shard_data` itself, on
  `shard_store_fixtures.bare_pool`, with `config/sharding.py`'s replicated and sharded lists;
  shard-key rows go into the pool's own `shard_keys` table;
- two shards by default. Each has every `build_schema` table, created with `create_all` on a
  temporary read-write engine, which is disposed afterwards;
- `REPLICATED_ROWS` copied into every shard with the same serials: `version`, `store_tag`,
  `redshift`, `wavenumber`, `tolerance`, `LambdaCDM`, `IntegrationSolver`, `wavenumber_exit_time`,
  `BackgroundModel` and `BackgroundModel_tags`;
- `SHARDED_ROWS`: on each of shards 0 and 1, one `TkNumericIntegration` with two
  `TkNumeric_tags` rows and three `TkNumericValue` rows.

Row sets are plain dicts, table → list of rows, with explicit serials. `with_rows` extends them
for prompt 02. The builder also takes `missing_tables`, `missing_columns` and `extra_sql` per
shard. `build_old_store` gives shard 1 without `OneLoopIntegral_tags`, and shard 0's
`TkNumericIntegration` without `stop_Tprime` (dropped after the rows are written).
`independent_row_counts` (stdlib `sqlite3`, `mode=ro`), `expected_row_counts` and `file_state`
(listing, and each file's SHA-256, size and `st_mtime_ns`) are the helpers the tests use.

**`Datastore/tests/schema_description.py`** (new, 204 lines): the deterministic description the
witness is written in. `actor_with_built_schema()` gives an actor instance with `_build_schema`
run on it and nothing else. It is also the capture script (`__main__`).

**`Datastore/tests/data/schema_at_base.json`** (new, 11 491 lines, SHA-256
`a8a7404c42831f81306162e931e3112fb5ba0d1e9e64be9e1cd72e38aa1d9a82`): the witness. It covers all
37 classes.

**Tests.** 22 new test methods in two modules. None needs Ray, and none opens anything under
`var/`.

- `test_schema_builder.py` (7):
  - `build_schema` reproduces the witness, class by class and byte for byte;
  - `build_schema` records hold no inserter;
  - the actor's `_build_schema` reproduces the witness, called as in the capture;
  - the actor adds only the inserters, each a partial of its own `_insert` over its record and
    table;
  - the actor's tables are in its `_metadata`;
  - the `None` fix, through `build_schema` and through the actor.
- `test_store_reader.py` (15):
  - it reads: serials and paths, the counts against an independent count, replicated serials
    equal on every shard, engines disposed on exit, `mode=ro` / `uri=true` / no `immutable` on
    every engine URL;
  - it never writes: file state before, during and after a full read, and an INSERT and a DDL
    through each engine raise `OperationalError` "readonly";
  - old stores: the missing table and column are named on the right shard, everything else
    reads, and extra columns and tables are reported;
  - refusals, each naming the file and leaving the file state unchanged: a journal beside the
    primary (all three suffixes), a journal beside each shard (three suffixes, two shards), a
    missing shard (whose message is `_read_closed_store`'s own, compared as a string), a primary
    with no `shards` table, and a missing primary;
  - no Ray: a child interpreter opens the store and reads every table, and `ray.is_initialized()`
    stays false.

## How `schema_at_base.json` was captured

At base `417c647`, **before `Datastore/SQL/Datastore.py` was edited**, and before `schema.py`
existed, it was captured twice. The only new file in the tree was `Datastore/tests/schema_description.py`,
which imports only the standard library and `sqlalchemy` at module scope.

1. **From a pristine extraction of the base commit**, into the session scratchpad, with the
   helper loaded by path from the working tree:
   ```bash
   REPO=/Users/ds283/Documents/Code/SecondaryGWKit
   BASE=<scratchpad>/impl01_base
   mkdir -p $BASE && cd $REPO && git archive 417c6471c08b405e718d4f9d8b244ae7e022b475 | tar -x -C $BASE
   cd $BASE && PYTHONPATH=. $REPO/venv/bin/python $REPO/Datastore/tests/schema_description.py <scratchpad>/impl01_schema_from_archive.json
   ```
   `importlib.import_module("Datastore.SQL.Datastore").__file__` was checked to resolve to
   `$BASE/Datastore/SQL/Datastore.py`, so the code described was the base's, not the tree's.
2. **From the working tree**, with `git diff --quiet HEAD -- Datastore/SQL/` true:
   `PYTHONPATH=. ./venv/bin/python Datastore/tests/schema_description.py <scratchpad>/impl01_schema_from_worktree.json`.

The two files are identical (`cmp`). Re-running (1) under `PYTHONHASHSEED=1` and
`PYTHONHASHSEED=12345` gave the same SHA-256, `a8a7404c…1d9a82`. The archive capture was then
copied to `Datastore/tests/data/schema_at_base.json`.

**What the capture does** (`actor_with_built_schema`):
- it takes `Datastore.__ray_metadata__.modified_class`, and makes `object.__new__(cls)`, never
  calling `__init__`;
- it sets `_factories = {}` and calls `register_factories(_factories)`, as `__init__` does;
- it sets `_metadata = sqla.MetaData()`, as `_create_engine` does, and `_tables`, `_inserters` and
  `_schema` to empty dicts;
- it calls `_build_schema()`.

`describe_schema` then records, per class:
- `has_table`;
- the record's non-callable fields. Column and table references are written as
  `{column, of_table}` and `{table}`, and the inserter is left out;
- for the table, each column in order: `name`, `repr(type)`, the compiled SQLite type,
  `nullable`, `primary_key`, foreign keys (target, ondelete, onupdate), `index`, `unique`,
  `default`, `server_default` and `autoincrement`;
- the table's constraints, sorted, and its indexes by name;
- the table's compiled SQLite `CREATE TABLE` and `CREATE INDEX` DDL.

It also records the class order, the table order and the sorted table names in the `MetaData`. The
JSON is `sort_keys=True, indent=1`, with a trailing newline.

**To re-capture at `HEAD~1` of this commit and compare:**
```bash
git archive HEAD~1 | tar -x -C <dir>
cd <dir> && PYTHONPATH=. <repo>/venv/bin/python <repo>/Datastore/tests/schema_description.py /path/out.json
cmp /path/out.json <repo>/Datastore/tests/data/schema_at_base.json
```
`HEAD~1` is `417c647`.

## What importing `Datastore/store_reader.py` loads

Measured in a fresh interpreter (`sys.modules` before and after `import Datastore.store_reader`):
- 1 663 new modules, in about 3.2 s;
- `ray` is imported, and **`ray.is_initialized()` is `False`**;
- the project packages `AdaptiveLevin`, `ComputeTargets`, `CosmologyConcepts`, `CosmologyModels`,
  `Datastore`, `LiouvilleGreen`, `MetadataConcepts`, `Quadrature`, `Units`, `config` and
  `utilities`;
- the third-party packages `sqlalchemy`, `numpy`, `scipy`, `pandas`, `pyarrow`, `ray` and its
  dependencies (`google`, `msgpack`, `psutil`, `jsonschema`, `yaml`, `requests`, …).

The weight comes from the factory map (`Datastore/SQL/Datastore.py` imports every factory and
`ray`) and from `ShardedPool`. `Datastore/SQL/schema.py` itself imports only `sqlalchemy`, but
importing it runs `Datastore/SQL/__init__.py`, which imports the actor module, so it too loads
`ray` without initialising it. As the audit (§3) records, the registry must therefore import the
reader inside its operation.

## Deviations from the prompt

1. **The description helper existed before the capture.** *IMPLEMENTATION CHOICE.* F1.1 says
   "before you change any code, capture". `Datastore/tests/schema_description.py` was written
   first, and it is a new file that changes no existing code. `Datastore.py` was unchanged at
   capture, and the capture was also taken from a `git archive` of the base, which does not
   contain the helper. Keeping the description in one committed module means the tests and the
   capture cannot describe the schema differently.
2. **The witness records more than F1.1 lists**: the compiled DDL, `default`, `server_default`,
   `autoincrement`, the class and table orders and the `MetaData` table set. *IMPLEMENTATION
   CHOICE.* It is strictly stronger evidence. The listed fields are all present.
3. **`build_schema` returns a `NamedTuple`, `BuiltSchema(tables, records)`.** *IMPLEMENTATION
   CHOICE.* README §4 fixes the name and arguments, not the return type. It unpacks as a pair.
4. **The actor keeps its duplicate-registration check, and runs it before calling
   `build_schema`.** *IMPLEMENTATION CHOICE.* In the old loop, the check fired before any table was
   declared for the offending class. Running it first keeps that, and `build_schema` has its own
   check over its own records.
5. **The shape of a `None` record** is `{"name", "validate_on_startup": False, "table": None}`, and
   the actor adds `"insert": None`. *IMPLEMENTATION CHOICE.* These are the keys the old `else`
   branch would have written had it been reachable. `validate_on_startup` is `False`, because a
   class with no table has nothing to validate.
6. **The reader also reports `extra_tables`**, refuses a missing, non-regular or symlinked
   primary, and refuses a primary that records no shards. Its store also exposes `tables`,
   `records` and `shard(serial)`. *IMPLEMENTATION CHOICE.* These are small additions to new code.
   The refusals mirror `copy_store`'s `_plan_relocation`. `extra_tables` makes the §4 report
   complete. None of them is a write path.
7. **The fixture's primary** uses `config/sharding.py`'s table lists, and not
   `shard_store_fixtures`' placeholder lists. It also has shard-key rows, and an `extra_sql` option
   per shard. *IMPLEMENTATION CHOICE.* This is closer to a real store, and it gives prompt 02 room.
   `config/sharding.py` is read, not changed.
8. **§4 found no absent or extra table or column.** Not a deviation, but the prompt expected some
   ("built before some columns existed"). An independent check with stdlib `sqlite3`
   (`PRAGMA table_info` against the witness's column lists, order-sensitive) agrees. Every one of
   the 37 declared tables is present on all four shards of the copy, with exactly the declared
   columns in the declared order.

## Verification performed

- **Baselines at `417c647`** (measured by the orchestrator before dispatch): AdaptiveLevin 32,
  ComputeTargets 552, CosmologyModels 39, Datastore 70, LiouvilleGreen 148 (1 skipped),
  RunRegistry 84.
- **After the change** (`PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t .`,
  the six suites run concurrently):
  - AdaptiveLevin 32 OK;
  - ComputeTargets 552 OK (the wall-clock flake did not fire);
  - CosmologyModels 39 OK;
  - **Datastore 92 OK**, which is 70 + 22 new;
  - LiouvilleGreen 148 OK (skipped=1);
  - RunRegistry 84 OK.
- **The witness.** Both `build_schema` and the actor's `_build_schema` reproduce it byte for byte
  after the change (the tests). Mutation (iii) shows that the actor test would catch a private
  copy.
- **Scope.** `git diff --cached -- Datastore/SQL/ShardedPool.py Datastore/SQL/ObjectFactories/ main.py RunRegistry/ tools/ config/`
  is empty. No existing test module was modified.
- **`black --check`** is clean on all seven Python files added or changed.
- **The §4 demonstration** is below.

### §4 — the reader on a copy of the sweep store

1. `python -m RunRegistry list`: 7 runs, **none `running`** (5 `done` or `failed`, 2 `unknown`
   pre-registry). About 70 GB free.
2. **The originals were snapshotted read-only** (`os.lstat`, SHA-256 of bytes, and stdlib `sqlite3`
   `mode=ro` `COUNT(*)` per table). This covered the two top-level stores, the backup, and the
   three `.manifest.json` sidecars: every file's size and `st_mtime_ns`, the per-table row counts
   of every `.sqlite`, the SHA-256 of the primaries and sidecars, and both directory listings. The
   primaries' SHA-256 prefixes: A3 `fdbe93a5…` (live and backup, the same file content), sweep
   `c3cda49d…`. The sidecars' prefixes: A3 `f7220408…`, sweep `a595b7c9…`, backup `ce7b476a…`.
3. `cp -p` of `handover-atol-sweep.sqlite` and its four `-shardNNNN.sqlite` files into
   `var/store-fingerprint-check-01/`. The copy was then snapshotted: the listing, and each file's
   SHA-256, size and `st_mtime_ns`.
4. **`open_read_only` on the copy's primary.** Its `shards` rows are legacy absolute paths into
   `var/datastores/`. The resolver read them as siblings **in the copy directory**, and printed its
   one `!!` line. The demonstration script asserted that every shard path's parent was the copy
   directory.
   - **Shards:** 0, 1, 2 and 3, i.e. `handover-atol-sweep-shard0000.sqlite` to `…-shard0003.sqlite`,
     all in `var/store-fingerprint-check-01/`.
   - **Absent tables, absent columns, extra columns, extra tables: none, on every shard.**
   - **Row counts.** Three counts agree for every table on every shard: the reader's
     `SELECT count(*)` through its engine, the rows streamed by a full `SELECT` of the columns
     present, and an independent stdlib `sqlite3` `mode=ro` count of the file. The primary's own
     tables are not factory tables, and are not read by the reader.

| Table | shard 0 | shard 1 | shard 2 | shard 3 | agree |
|---|---:|---:|---:|---:|---|
| `BackgroundModel` | 1 | 1 | 1 | 1 | yes |
| `BackgroundModelValue` | 1 740 | 1 740 | 1 740 | 1 740 | yes |
| `BackgroundModel_tags` | 4 | 4 | 4 | 4 | yes |
| `GkNumericIntegration` | 1 188 | 1 089 | 985 | 1 287 | yes |
| `GkNumericValue` | 39 179 | 33 876 | 28 525 | 44 865 | yes |
| `GkNumeric_tags` | 9 504 | 8 712 | 7 880 | 10 296 | yes |
| `GkSource` | 290 | 290 | 290 | 290 | yes |
| `GkSourcePolicy` | 2 | 2 | 2 | 2 | yes |
| `GkSourcePolicyData` | 290 | 290 | 290 | 290 | yes |
| `GkSourceValue` | 254 040 | 254 040 | 254 040 | 254 040 | yes |
| `GkSource_tags` | 2 320 | 2 320 | 2 320 | 2 320 | yes |
| `GkWKBIntegration` | 3 480 | 3 480 | 3 480 | 3 480 | yes |
| `GkWKBValue` | 227 083 | 231 942 | 235 866 | 222 046 | yes |
| `GkWKB_tags` | 27 840 | 27 840 | 27 840 | 27 840 | yes |
| `IntegrationSolver` | 7 | 7 | 7 | 7 | yes |
| `LambdaCDM` | 1 | 1 | 1 | 1 | yes |
| `OneLoopIntegral` | 0 | 0 | 0 | 0 | yes |
| `OneLoopIntegral_tags` | 0 | 0 | 0 | 0 | yes |
| `QCD_Cosmology` | 1 | 1 | 1 | 1 | yes |
| `QuadSource` | 10 | 8 | 6 | 12 | yes |
| `QuadSourceIntegral` | 1 955 | 1 892 | 1 920 | 1 939 | yes |
| `QuadSourceIntegral_tags` | 17 595 | 17 028 | 17 280 | 17 451 | yes |
| `QuadSourcePolicy` | 2 | 2 | 2 | 2 | yes |
| `QuadSourceValue` | 4 807 | 3 681 | 2 658 | 6 032 | yes |
| `QuadSource_tags` | 70 | 56 | 42 | 84 | yes |
| `TkNumericIntegration` | 2 | 2 | 2 | 2 | yes |
| `TkNumericValue` | 967 | 963 | 963 | 972 | yes |
| `TkNumeric_tags` | 14 | 14 | 14 | 14 | yes |
| `TkWKBIntegration` | 2 | 2 | 2 | 2 | yes |
| `TkWKBValue` | 2 354 | 2 457 | 2 561 | 2 255 | yes |
| `TkWKB_tags` | 14 | 14 | 14 | 14 | yes |
| `redshift` | 1 740 | 1 740 | 1 740 | 1 740 | yes |
| `store_tag` | 10 | 10 | 10 | 10 | yes |
| `tolerance` | 13 | 13 | 13 | 13 | yes |
| `version` | 1 | 1 | 1 | 1 | yes |
| `wavenumber` | 8 | 8 | 8 | 8 | yes |
| `wavenumber_exit_time` | 8 | 8 | 8 | 8 | yes |
| **total rows** | 596 542 | 593 534 | 590 516 | 599 069 | yes |

5. **The copy was re-snapshotted**: identical to step 3 (`cmp` of the two JSON snapshots), with no
   new file. The SHA-256 prefixes: shard0000 `4000a990…`, shard0001 `8cf35f4c…`, shard0002
   `6c6995ad…`, shard0003 `362ad59e…`, primary `c3cda49d…` (the same as the original's).
6. **The originals were re-snapshotted**: identical to step 1 (`cmp`).
7. **`var/store-fingerprint-check-01/` was deleted.** `var/` holds `bootstrap-a3-resume.log`,
   `datastores` and `runs`, as before. Nothing from §4 is committed. The scripts were
   `impl01_snapshot_originals.py`, `impl01_snapshot_copy.py` and `impl01_demo_reader.py`, in the
   session scratchpad.

## The deliberate-breakage record

Each mutation was applied to the working tree with every change of this prompt staged. The diff
is `git diff` (working tree against the index), so it applies with `git apply` against this
commit. The full Datastore suite was run under each mutation, and then the file was restored with
`git checkout -- <file>`, after which `git diff --quiet` was true. `git apply --check` of each diff
succeeded against the staged tree. **No mutation is committed.**

### (i) The reader opens files read-write, without `mode=ro`

```diff
diff --git a/Datastore/store_reader.py b/Datastore/store_reader.py
index 61bfc09..662498c 100644
--- a/Datastore/store_reader.py
+++ b/Datastore/store_reader.py
@@ -102,7 +102,7 @@ def _refuse_journals(primary: Path, path: Path, what: str) -> None:
 
 def read_only_url(path: Path) -> str:
     """The SQLAlchemy URL of ``path`` opened read-only: ``mode=ro``, never ``immutable=1``."""
-    return f"sqlite:///file:{path}?mode=ro&uri=true"
+    return f"sqlite:///file:{path}?uri=true"
 
 
 def _read_only_engine(path: Path) -> sqla.Engine:
```
**Failed** (6 failures, 92 run):
- `test_store_reader.TestReaderNeverWrites.test_a_write_through_a_reader_engine_raises`, for
  INSERT and DDL on shards 0 and 1;
- `test_store_reader.TestReaderReads.test_engines_open_mode_ro_and_never_immutable`.

### (ii) `build_schema` omits one of the prepended columns (`timestamp`)

```diff
diff --git a/Datastore/SQL/schema.py b/Datastore/SQL/schema.py
index 226a947..b84ab3c 100644
--- a/Datastore/SQL/schema.py
+++ b/Datastore/SQL/schema.py
@@ -99,7 +99,7 @@ def build_schema(metadata: sqla.MetaData, factories: Mapping[str, Any]) -> Built
         schema["use_timestamp"] = use_timestamp
         if use_timestamp:
             timestamp_col = sqla.Column("timestamp", sqla.DateTime())
-            tab.append_column(timestamp_col)
+            # tab.append_column(timestamp_col)
             schema["timestamp_col"] = timestamp_col
 
         use_stepping = registration_data.get("stepping", False)
```
**Failed** (61 failures):
- `test_schema_builder.TestSchemaIsUnchanged.test_build_schema_reproduces_the_witness`: the whole
  test, and a subtest for each of the 29 classes that register a timestamp;
- `test_actor_build_schema_reproduces_the_witness`: likewise, because the actor delegates;
- `TestNoneRegistration.test_build_schema_gives_a_record_with_no_table`, whose `OneColumn` table
  lost `timestamp`.

### (iii) `_build_schema` keeps a private copy of the loop, with the `version` column's nullability changed

The private copy is the base's loop verbatim, with `nullable=False` added to the `version`
column.

```diff
diff --git a/Datastore/SQL/Datastore.py b/Datastore/SQL/Datastore.py
index 8bdf1fb..b2101c1 100644
--- a/Datastore/SQL/Datastore.py
+++ b/Datastore/SQL/Datastore.py
@@ -299,19 +299,89 @@ class Datastore:
             # print(f"Registered storable class factory '{cls_name}'")
 
     def _build_schema(self):
-        for cls_name in self._factories:
+        # iterate through all registered storage adapters, querying them for the columns
+        # they need to persist their data
+        for cls_name, factory in self._factories.items():
             if cls_name in self._schema:
                 raise RuntimeWarning(
                     f"Duplicate registered factory for storable class '{cls_name}'"
                 )
 
-        # the tables and schema records come from the one schema builder (Datastore/SQL/schema.py);
-        # the actor adds only the inserters, which are bound to its own _insert
-        built = build_schema(self._metadata, self._factories)
+            # query class for a list of columns that it wants to store
+            registration_data = factory.register()
+
+            schema = {
+                "name": cls_name,
+                "validate_on_startup": registration_data.get(
+                    "validate_on_startup", False
+                ),
+            }
+
+            # does this storage object require its own table?
+            if registration_data is not None:
+                # generate main table for this adapter class
+                tab = sqla.Table(
+                    cls_name,
+                    self._metadata,
+                )
+
+                use_serial = registration_data.get("serial", True)
+                schema["use_serial"] = use_serial
+                if use_serial:
+                    serial_col = sqla.Column("serial", sqla.Integer, primary_key=True)
+                    tab.append_column(serial_col)
+                    schema["serial_col"] = serial_col
+
+                # attach pre-defined columns
+                use_version = registration_data.get("version", False)
+                schema["use_version"] = use_version
+                if use_version:
+                    version_col = sqla.Column(
+                        "version",
+                        sqla.Integer,
+                        sqla.ForeignKey("version.serial"),
+                        index=True,
+                        nullable=False,
+                    )
+                    tab.append_column(version_col)
+                    schema["version_col"] = version_col
+
+                use_timestamp = registration_data.get("timestamp", False)
+                schema["use_timestamp"] = use_timestamp
+                if use_timestamp:
+                    timestamp_col = sqla.Column("timestamp", sqla.DateTime())
+                    tab.append_column(timestamp_col)
+                    schema["timestamp_col"] = timestamp_col
+
+                use_stepping = registration_data.get("stepping", False)
+                if isinstance(use_stepping, str):
+                    if use_stepping not in ["minimum", "exact"]:
+                        print(
+                            f"!! Warning: ignored stepping selection '{use_stepping}' when registering storable class factory for '{cls_name}'"
+                        )
+                        use_stepping = False
+
+                _use_stepping = isinstance(use_stepping, str) or use_stepping is True
+                schema["use_stepping"] = _use_stepping
+                if _use_stepping:
+                    stepping_col = sqla.Column("stepping", sqla.Integer)
+                    tab.append_column(stepping_col)
+                    schema["stepping_col"] = stepping_col
+
+                    _stepping_mode = (
+                        None if not isinstance(use_stepping, str) else use_stepping
+                    )
+                    schema["stepping_mode"] = _stepping_mode
+
+                # append all columns supplied by the class
+                sqla_columns = registration_data.get("columns", [])
+                for col in sqla_columns:
+                    tab.append_column(col)
+                schema["columns"] = sqla_columns
+
+                # store in table cache
+                schema["table"] = tab
 
-        for cls_name, schema in built.records.items():
-            tab = schema["table"]
-            if tab is not None:
                 # build inserter
                 inserter = functools.partial(self._insert, schema, tab)
                 schema["insert"] = inserter
@@ -319,9 +389,18 @@ class Datastore:
                 # also store table and inserter in their own separate cache
                 self._tables[cls_name] = tab
                 self._inserters[cls_name] = inserter
+
+                # print(
+                #     f"Registered storage schema for storable class adapter '{cls_name}' with database table '{tab.name}'"
+                # )
             else:
+                schema["table"] = None
                 schema["insert"] = None
 
+                # print(
+                #     f"Registered storage schema for storable class adapter '{cls_name}' without database table"
+                # )
+
             self._schema[cls_name] = schema
 
     def _ensure_tables(self):
```
**Failed** (14 failures, 1 error):
- `test_schema_builder.TestSchemaIsUnchanged.test_actor_build_schema_reproduces_the_witness`: the
  whole test, and subtests for the 13 classes that register `version` (`BackgroundModel`,
  `GkNumericIntegration`, `GkSource`, `GkSourcePolicy`, `GkSourcePolicyData`, `GkWKBIntegration`,
  `OneLoopIntegral`, `QuadSource`, `QuadSourceIntegral`, `QuadSourcePolicy`,
  `TkNumericIntegration`, `TkWKBIntegration`, `wavenumber_exit_time`);
- ERROR `TestNoneRegistration.test_actor_gives_a_record_with_no_table_and_no_inserter`, because
  the copy still has the old ordering bug.

`test_build_schema_reproduces_the_witness` **passed**, as it should: the function was not
changed. The two witness tests therefore discriminate between the function and the actor.

### (iv) The absent-column check always returns nothing

```diff
diff --git a/Datastore/store_reader.py b/Datastore/store_reader.py
index 61bfc09..4dffd2f 100644
--- a/Datastore/store_reader.py
+++ b/Datastore/store_reader.py
@@ -133,7 +133,7 @@ def _describe_shard(
                 row[1] for row in conn.exec_driver_sql(f'PRAGMA table_info("{quoted}")')
             ]
             declared = [c.name for c in table.columns]
-            absent_columns[name] = tuple(c for c in declared if c not in file_columns)
+            absent_columns[name] = ()
             extra_columns[name] = tuple(c for c in file_columns if c not in declared)
 
     return ReadOnlyShard(
```
**Failed** (1 failure, 1 error): `test_store_reader.TestOldStores.test_missing_table_and_column_are_reported_on_the_right_shard`.
It fails on shard 0's report. Its full read then errors with "no such column", because the
missing column was no longer excluded.

### (v) The journal refusal is removed

```diff
diff --git a/Datastore/store_reader.py b/Datastore/store_reader.py
index 61bfc09..d10a40c 100644
--- a/Datastore/store_reader.py
+++ b/Datastore/store_reader.py
@@ -92,6 +92,7 @@ def _refuse(primary: Path, reason: str) -> RuntimeError:
 
 
 def _refuse_journals(primary: Path, path: Path, what: str) -> None:
+    return
     for journal in ShardedPool._journal_paths(path):
         if os.path.lexists(journal):
             raise _refuse(
```
**Failed** (9 failures):
- `test_store_reader.TestRefusals.test_journal_beside_the_primary`, for `-journal`, `-wal` and
  `-shm`;
- `test_journal_beside_a_shard`, for the same three suffixes on shards 0 and 1.

### (vi) The `None` fix is reverted

```diff
diff --git a/Datastore/SQL/schema.py b/Datastore/SQL/schema.py
index 226a947..6e51861 100644
--- a/Datastore/SQL/schema.py
+++ b/Datastore/SQL/schema.py
@@ -55,6 +55,11 @@ def build_schema(metadata: sqla.MetaData, factories: Mapping[str, Any]) -> Built
         # query class for a list of columns that it wants to store
         registration_data = factory.register()
 
+        schema = {
+            "name": cls_name,
+            "validate_on_startup": registration_data.get("validate_on_startup", False),
+        }
+
         # does this storage object require its own table?
         if registration_data is None:
             records[cls_name] = {
```
**Failed** (2 errors, `AttributeError: 'NoneType' object has no attribute 'get'`):
- `test_schema_builder.TestNoneRegistration.test_build_schema_gives_a_record_with_no_table`;
- `test_actor_gives_a_record_with_no_table_and_no_inserter`.

## Observations not acted on

None of these is a code defect, so no §3 issue was opened.

- **`Datastore/SQL/__init__.py` rebinds `Datastore.SQL.Datastore` to the actor class** (`from
  .Datastore import Datastore`). `from Datastore.SQL import Datastore` therefore gives the class
  and not the module. The capture helper uses `importlib.import_module("Datastore.SQL.Datastore")`
  for this reason. It is long-standing and harmless, and it is noted so that nobody trips on it.
- **The witness is written in SQLAlchemy 2.0.39's reprs and compiled DDL.** A SQLAlchemy upgrade
  that changed `repr(type)` or DDL formatting would fail the witness tests without any schema
  change. If that happens, the right response is to re-capture from the base as recorded above,
  under the new version, and compare. The file must never be regenerated from the new code.
- **Per-shard counts in the sweep copy.** The copy is uniform for `GkSource` (290 per shard),
  `GkSourceValue` (254 040), `GkWKBIntegration` (3 480) and their tag tables, while
  `GkNumericIntegration` and the value tables vary. With 8 wavenumbers over 4 shards this is what
  the sharding would give. It is noted only because prompt 02's replicated-versus-sharded
  comparison will look at such tables.
- **An idle store in rollback-journal mode leaves no file** while something holds it open, so the
  journal refusal cannot see an open-but-idle writer. `mode=ro` still prevents the reader from
  writing. This is inherent, and audit §7 already records it; the registry's `running` check is
  the guard.
- **A stale sentence in `docs/OPEN_ISSUES.md`'s preamble.** The paragraph under the board list
  says "Of the 90 above, 87 are spread across the boards", while the header has said 94 and then
  102 since before this campaign. The header count and the table rows agree: 101 rows after this
  commit's deletion. The sentence predates this prompt and is prose, not an issue row, so it was
  left alone. The next edit to that preamble should correct it.

## State handed to the next prompt

- `from Datastore.store_reader import open_read_only`. `with open_read_only(primary) as store:` gives
  `store.shards`, sorted by serial. Each shard has `.serial`, `.path`, `.engine` (read-only),
  `.tables`, `.absent_tables`, `.absent_columns[name]`, `.extra_columns[name]` and
  `.extra_tables`, and the store also has `store.tables`, `store.records` and `store.shard(n)`.
  To read a table on a shard missing a column, select only the columns not in
  `absent_columns[name]`; `select(func.count()).select_from(table)` works regardless.
- `from Datastore.SQL.schema import build_schema`. There is one schema builder. The four
  hand-copies in existing tests remain, by README §5 rule 7.
- `Datastore/tests/real_store_fixtures.py`: extend `REPLICATED_ROWS` and `SHARDED_ROWS`, or pass
  your own through `with_rows`. `build_old_store` and the `missing_tables`, `missing_columns` and
  `extra_sql` parameters make old or odd stores.
- The sweep store's copy reads cleanly with the reader, in about 12 s for a full stream of every
  row on all four shards. It has no absent or extra schema, so prompt 02's old-store handling will
  be exercised by fixtures, not by this store.
- The `Datastore` baseline is now **92**.
