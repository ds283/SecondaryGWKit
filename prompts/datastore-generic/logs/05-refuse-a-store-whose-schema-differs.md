# Log — prompt 05: refuse a store whose schema differs

**Subject:** *Refuse a store whose files differ from the declared schema* · **Model:** Claude Opus
5.5 · **Date:** 2026-10-05 · **Base:** `6ba15c7`, clean · **Result:** landed.
- **S1:** one comparison, `schema_differences`, and one refusal, `StoreSchemaMismatch(RuntimeError)`,
  both in `Datastore/SQL/schema.py`. The reader refuses any difference, so the inventory and the
  fingerprint do; the reader's four fields and the inventory's `absent-table` and `incomplete`
  problems (and `ShardRead.comparable`) are gone.
- **S2:** the read-only pool refuses through the reader; its and the read-only actor's absent-table
  reporting is gone. The reconcile reads the declared columns as they are; its column subset,
  `unkeyed` and the `columns` difference kind are gone.
- **S3:** a read-write open refuses every difference but an absent table (U1) before the in-flight
  record is read; both constructors refuse a primary that differs from `_create_engine`'s six
  tables; the four WKB `numeric_serial` selects and `build_old_store` are gone.

No stop condition fired. Closes `[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]`.
Opens nothing. Index 108 → 107. The commit is this commit.

## What shipped

**`Datastore/SQL/schema.py`.**
- `SchemaDifferences` (a `NamedTuple`): `absent_tables` (declaration order), `extra_tables` (file
  order), `absent_columns` and `extra_columns` (declared table → names, declaration and file order
  respectively; only tables with a difference are keys). `empty` is true iff all four are empty;
  `describe()` gives every difference by kind, naming each table and column.
- `schema_differences(conn, tables)`: reads `sqlite_master` (type `table`, without `sqlite_*`) and
  `PRAGMA table_info` of each declared table the file has. Nothing else; it issues no write
  (pinned by `test_it_issues_no_write`, which records every statement on a read-write engine).
- `StoreSchemaMismatch(RuntimeError)`: attributes `verb`, `primary`, `file` (`shard #N "<path>"` or
  `the primary "<path>"`) and `differences`; one sentence, `Cannot <verb> sharded datastore
  "<primary>": <file> differs from the tables this code declares (<describe()>), so nothing was
  read`. `__reduce__` as `Datastore/replication.py`'s exceptions have it. It is raised where it is
  built and never wrapped.
- The module docstring says the module is also the one comparison of a file against the declared
  tables. It still imports only `sqlalchemy` (and `typing`), and names no project table.

**`Datastore/store_reader.py`.**
- `_describe_shard(primary, serial, path, engine, tables)` calls `schema_differences` and raises
  `StoreSchemaMismatch("read", …)` on any difference, an absent table included. It is called inside
  `open_read_only`'s `try`, so every engine made so far is disposed and nothing is yielded.
- `ReadOnlyShard` is `serial`, `path`, `engine`, `tables`; its docstring says a shard that differs
  is refused before it is yielded.
- The "Old stores." paragraph is replaced by one saying that a store whose files differ is refused
  by name; `open_read_only`'s docstring lists the refusal. `Dict` and `FrozenSet` imports removed.

**`Datastore/store_inventory.py`.**
- The `absent-table` and `incomplete` problems are gone, with every branch that made them (the
  class's own table, the key-column check and `absent`, the parent-set member table, the
  association table, the value table); the association and value reads are unwrapped from their
  `else:`, unchanged otherwise.
- `has_validated` is `validated`; `has_timestamp` is `"timestamp" in table.c`.
- `ShardContext` loses `absent_tables` and `absent_columns`; `read_inventory` stops passing them.
- `ShardRead.comparable` is gone, with `_read`'s argument and the assignments; `_combine`'s
  `comparable` is now `sorted(reads)`, so every shard is compared. A grep found no factory that
  builds a `ShardRead` or reads `comparable`.
- The docstring's two problem names and its "left out of that class's replicated comparison"
  paragraph are gone; `FrozenSet` import removed.

**`Datastore/SQL/ShardedPool.py`.**
- Imports `StoreSchemaMismatch` and `schema_differences` beside `build_schema`.
- `_refuse_a_primary_that_differs()` (new): compares the primary with the six tables
  `_create_engine` declares, on the pool's own engine, and raises `StoreSchemaMismatch("open",
  …, 'the primary "<path>"', …)` on any difference. Called by the read-write constructor after
  `_create_engine()` and before `_read_shard_data()` (it is the primary's first read, so it is what
  rolls back a hot primary journal; `primary_journal` is noted before it, as before), and by
  `_open_read_only` after the mode=ro engine is made and before `_read_shard_data()`. Not called by
  `_read_closed_store`, `open_read_only`, `copy_store`, `move_store`, `closed_store_files` or
  `delete_store`.
- `_refuse_shards_that_differ(built, shard_ids)` (new): called by `_reconcile_replicated_tables`
  as step "1b", after the hot-journal rollback and once `built` is made, before
  `_read_in_flight_record` (so before the interrupted-prune branch too). Each shard is compared on
  a mode=ro, `NullPool` engine through `store_reader.read_only_url` (imported at function scope, as
  `_open_read_only` does); the first shard, in serial order, with any difference other than an
  absent table raises `StoreSchemaMismatch("open", …)`. Its docstring says why an absent table is
  allowed (U1: an interrupted first open, each `create` of `_ensure_tables` committing on its own,
  or an interrupted drop; the actors complete it).
- "THE READ-ONLY POOL" comment, steps 2 and 3: a differing shard is refused by the reader, the
  primary by `_refuse_a_primary_that_differs`; "the tables it lacks are reported" is gone.
- `_open_read_only`: `self.absent_tables` and its print are gone; `open_read_only` is entered only
  for its refusals, and the comment says so.
- `_version_labels_present`: the branch for a shard without `version`, and its docstring clause,
  are gone.
- `_read_replicated_tables`: a table the shard lacks is still read as empty (`"columns": None`, as
  before; detected by its `PRAGMA table_info`); otherwise it selects `spec["compared"]` as it is.
  The `present` subset and `unkeyed` are gone; the docstring says the schema check at open has
  refused every other difference.
- The `columns` difference kind is gone from `_replicated_differences` (and its docstring),
  `_describe_differences` and `_plan_replicated_repair` (`kind == "duplicate"`).
- `_read_full_rows` selects every column of the table.
- Kept: `_prune_needed`'s and `_prune_shard`'s `has_table` skips (U1).

**`Datastore/SQL/Datastore.py`, the read-only actor only.** `self._inspector`, the absent-table
block and its comment, `_skipped_validation`, and the skip in `_validate_on_startup_read_only` (with
its docstring sentence) are gone. `read_only_state` returns `{"read_only": …}`; its docstring says
the pool keeps the call because it waits on every constructor. `_ensure_tables` and
`_drop_actions` are untouched (B10, U1).

**`TkWKBIntegration.py`, `GkWKBIntegration.py`.** The four `wkb_table.c.numeric_serial` selected
columns and their two-line comments are deleted (`_build_impl_model` and `read_batch` in each). An
`ast` read of the four functions shows `numeric_serial` only in `payload.get` and the conditional
`_numeric_parent_filter`, which stay.

**`Datastore/tests/real_store_fixtures.py`.** `build_old_store` and the five `OLD_MISSING_*`
constants are deleted. The module docstring's "Old stores." paragraph is now "A store whose schema
differs", saying `missing_tables`, `missing_columns` and `extra_sql` build the refusal tests'
shapes. `build_real_store`'s parameters are kept; `_write_shard`'s comment "an old store's table
lacks a column" now reads "a shard whose table lacks a column".

## The tests, test by test

### Deleted (nine, all in `Datastore`; counted with `ast` against `HEAD`)

| Module | Class.method |
|---|---|
| `test_store_inventory` | `TestOldStoresAndOrphans.test_an_absent_table` |
| | `TestOldStoresAndOrphans.test_an_absent_association_table` |
| | `TestOldStoresAndOrphans.test_an_absent_value_table` |
| | `TestOldStoresAndOrphans.test_an_absent_replicated_table` |
| | `TestOldStoresAndOrphans.test_an_absent_replicated_association_table` |
| | `TestOldStoresAndOrphans.test_a_missing_key_column_makes_the_class_incomplete` |
| | `TestOldStoresAndOrphans.test_a_missing_column_the_key_does_not_need` |
| `test_gksource_parent_set` | `TestTheInventoryResolvesTheSet.test_a_store_without_the_table_is_named_absent` |
| `test_read_only_pool` | `TestAtOpen.test_a_lookup_on_a_missing_table_fails_in_sqlite` |

`test_store_inventory`'s `check` helper and `data/schema_at_datastore-integrity-06.json` stay; its
module docstring item 8 now reads "orphans are named problems".

### Converted to a refusal (renamed, as the prompt allows; `ast` counts each as one removal and one addition)

- `test_store_reader.TestOldStores.test_missing_table_and_column_are_reported_on_the_right_shard`
  → **`TestADifferingSchemaIsRefused.test_a_missing_table_or_column_is_refused_naming_the_shard`**.
  Two sub-tests, the same shapes as `build_old_store` built (`OneLoopIntegral_tags` missing on
  shard 1; `TkNumericIntegration.stop_Tprime` missing on shard 0), each its own store. Each asserts
  `StoreSchemaMismatch` (a `RuntimeError`), `verb == "read"`, the exact `differences`, the message's
  prefix and tail, `shard #N "<path>"`, the table (and column), and an unchanged `file_state`.
- `test_store_reader.TestOldStores.test_extra_columns_and_tables_are_reported` →
  **`TestADifferingSchemaIsRefused.test_an_extra_column_or_table_is_refused_naming_the_shard`**: the
  same, for `GkSource.legacy_note` and `retired_table` on shard 1, as two sub-tests. The class is
  renamed `TestADifferingSchemaIsRefused`.
- `test_read_only_pool.TestAtOpen.test_a_shard_lacking_a_table_is_reported_and_the_store_opens` →
  **`test_a_shard_lacking_a_table_is_refused_naming_it`**. Both sub-tests (`GkSource`, which
  validates; `OneLoopIntegral`, which does not) drop the table from shard 1 and assert
  `StoreSchemaMismatch` with `absent_tables == (cls_name,)`, `shard #1 "<path>"` and `it lacks the
  table(s) "<cls_name>"`; no actor constructed (`sp._Options.remote` counted, as
  `test_reconcile_at_open` does) and no actor call; checksums unchanged. The module docstring item 4
  says each is refused. **This is the one intended move of behaviour**: a store left by an
  interrupted drop now opens only read-write.

### Re-fixtured (only the rows the test builds change)

- `test_inventory_consumers.TestRunLabels.test_a_problem_elsewhere_does_not_matter`:
  `extra_sql={1: ['INSERT INTO "QuadSourceIntegral_tags" (parent_serial, tag_serial) VALUES (999,
  1)']}` replaces `missing_tables`. Measured (scratch script): the inventory then holds exactly one
  problem, `orphan-tag: QuadSourceIntegral: 1 QuadSourceIntegral_tags row(s) on shard #1 name a
  QuadSourceIntegral row that is not there; e.g. parent serials 999`, and `available_run_labels`
  returns `['fixture']`.
- `test_inventory_report.TestTheDisplaySaysWhatTheInventoryHolds.test_every_problem_appears_in_full_when_there_are_many`:
  seven shards; one orphan row in `QuadSourceIntegral_tags` on shards 2–6, and on shard 1 one in
  each of the seven tag tables, each through its column naming the tagged row
  (`integration_serial`, `wkb_serial` or `parent_serial`). Measured: 12 problems, all `orphan-tag`,
  6 on `QuadSourceIntegral`, against `MAX_SHOWN` 5. The assertions are unchanged.

### Changed (an assertion or a construction, as the prompt names)

- `test_reconcile_at_open.TestNoRecordRefuses.test_a_primary_without_the_record_table_is_refused`:
  `assertRaises(OperationalError)` → `assertRaises(StoreSchemaMismatch)`; the `OperationalError`
  import goes, `StoreSchemaMismatch` is imported. Both sub-tests pass; the rest of the test is
  unchanged.
- `test_database_errors_reach_the_caller.TestTheWKBValueReads`: in its four tests
  `columns=(RHO_GAUSS_ORDER,)` replaces `(NUMERIC_SERIAL,)`; the class docstring says none of the
  reads selects `numeric_serial` and that the statement is recognised by `rho_gauss_order`. The
  harness and the other classes are untouched; `NUMERIC_SERIAL` is still used by
  `TestTheWKBIntegrations`.
- `test_quadsource_parent_key` (`TestCrossShardParentUnit`'s hand-built `ShardContext`) and
  `test_wkb_numeric_parent_key` (`TestNullableParent`'s): `absent_tables=frozenset(),
  absent_columns={}` removed.
- `test_store_reader`, beyond the conversions: the `build_old_store` and `OLD_MISSING_*` imports go
  (`StoreSchemaMismatch` is imported); `_readable_columns` is deleted and `read_every_table` selects
  every column of every table; `_CHILD`'s absent-table skip goes;
  `TestReaderReads.test_every_shard_with_its_serial_and_path` loses its five assertions on the
  removed fields; `TestNoRay`'s `expected` is `sum(len(s.tables) for s in opened.shards)`; the
  module docstring's old-store item is replaced by the refusal.

### Added: `Datastore/tests/test_store_schema.py` (new; 13 tests)

- `TestSchemaDifferences.test_every_file_of_a_current_store_has_no_difference` — every shard and
  the primary of `build_real_store` and `build_full_store` (the primary against the six tables,
  whose names it also asserts);
- `TestSchemaDifferences.test_each_kind_is_found_on_the_right_shard_by_name` — a three-shard store
  with all four kinds, each on its shard, exactly; absent tables in declaration order, extra tables
  in file order; `describe()`'s text;
- `TestSchemaDifferences.test_it_issues_no_write` — on a read-write engine, every statement is the
  `sqlite_master` select or a `PRAGMA table_info`, and no file changes;
- `TestTheReader.test_a_refusal_yields_nothing_and_disposes_every_engine` — shard 1 of 3 refused:
  engines were made for shards 0 and 1 only, both disposed, nothing yielded;
- `TestTheInventory.test_read_inventory_refuses_a_store_missing_a_table` — `read_inventory` on a
  full store without `QuadSourceIntegral_tags` on shard 1;
- `TestTheReadWriteOpenRefuses.test_an_absent_column_an_extra_column_and_an_extra_table` — through
  `standin_pool`, on shard 2 (`TkNumericIntegration.stop_Tprime` dropped; `tolerance.legacy_note`
  added; `retired_table` created): `StoreSchemaMismatch`, `verb == "open"`, exact `differences`,
  no actor constructed, `_read_in_flight_record` never called, checksums unchanged;
- `TestTheReadWriteOpenRefuses.test_an_absent_table_beside_another_difference_is_still_refused`;
- `TestTheReadWriteOpenRefuses.test_a_prune_record_is_not_completed` — a `prune` record of
  `BackgroundModel` on the primary and an extra table on shard 0: refused, record kept;
- `TestTheReadWriteOpenRecoversAnAbsentTable.test_a_fresh_store_has_no_difference` — a store the
  real `ShardedPool` created on stand-in shards: every shard and the primary;
- `TestTheReadWriteOpenRecoversAnAbsentTable.test_an_interrupted_drop_of_a_sharded_class` —
  `GkSourceValue`, `GkSource_tags`, `GkSource_parents`, `GkSource` dropped from shard 0;
- `TestTheReadWriteOpenRecoversAnAbsentTable.test_an_interrupted_drop_of_a_replicated_class` —
  `GkSourcePolicy` dropped from shard 1, empty elsewhere;
- `TestTheReadWriteOpenRecoversAnAbsentTable.test_an_interrupted_first_open` — the second half of
  the declared tables dropped from shard 2 and the version row deleted from every shard;

  each of the three opens read-write, and afterwards every shard has no difference and holds the
  version row `standin`. These are the first tests of U1 (breakage (ii)).
- `TestThePrimaryIsRefused.test_an_extra_table_and_a_missing_column` — `retired_table` added to, and
  `sharded_tables.key_attr` dropped from, the primary; each read-write and read-only: refused,
  `the primary "<path>"` named, no actor, no in-flight read, checksums unchanged.

### Helpers and fields removed

- `ReadOnlyShard.absent_tables`, `.absent_columns`, `.extra_columns`, `.extra_tables`;
  `ShardContext.absent_tables`, `.absent_columns`; `ShardRead.comparable`;
  `ShardedPool.absent_tables` (read-only pool); `Datastore._absent_tables`, `._skipped_validation`
  and `._inspector` (read-only actor), and the `absent_tables` and `skipped_validation` keys of
  `read_only_state`.
- `real_store_fixtures.build_old_store`, `OLD_MISSING_TABLE`, `OLD_MISSING_TABLE_SHARD`,
  `OLD_MISSING_COLUMN_TABLE`, `OLD_MISSING_COLUMN`, `OLD_MISSING_COLUMN_SHARD`.
- `test_store_reader._readable_columns` (used only to subset by absent columns). Grep: no other
  user.
- Imports left unused: `Dict`, `FrozenSet` (`store_reader`); `FrozenSet` (`store_inventory`);
  `OperationalError` (`test_reconcile_at_open`). An `ast` check of every touched module finds no
  unused import that was not already unused at `HEAD` (three were: `TkNumericIntegration` in
  `test_gksource_parent_set`, `sqla_TkWKBValue_factory` in `test_wkb_numeric_parent_key`, `Record`
  in `test_store_inventory`; left as they are).

## The measured messages

Measured on the staged tree (scratch script; `<dir>` is the temporary directory).

From the reader (`open_read_only`, so also `read_inventory`, `store fingerprint` and the read-only
pool's shard refusals):

```
absent table:  Cannot read sharded datastore "<dir>/fixture-store.sqlite": shard #1 "<dir>/fixture-store-shard0001.sqlite" differs from the tables this code declares (it lacks the table(s) "OneLoopIntegral_tags"), so nothing was read
absent column: Cannot read sharded datastore "<dir>/fixture-store.sqlite": shard #0 "<dir>/fixture-store-shard0000.sqlite" differs from the tables this code declares (its table "TkNumericIntegration" lacks the column(s) "stop_Tprime"), so nothing was read
extra column:  Cannot read sharded datastore "<dir>/fixture-store.sqlite": shard #1 "<dir>/fixture-store-shard0001.sqlite" differs from the tables this code declares (its table "GkSource" has the column(s) "legacy_note", which this code does not declare), so nothing was read
extra table:   Cannot read sharded datastore "<dir>/fixture-store.sqlite": shard #1 "<dir>/fixture-store-shard0001.sqlite" differs from the tables this code declares (it has the table(s) "retired_table", which this code does not declare), so nothing was read
```

From a read-write open (stand-in shards):

```
absent table:  opens (U1); the actors create the table
absent column: Cannot open sharded datastore "<dir>/store.sqlite": shard #0 "<dir>/store-shard0000.sqlite" differs from the tables this code declares (its table "TkNumericIntegration" lacks the column(s) "stop_Tprime"), so nothing was read
extra column:  Cannot open sharded datastore "<dir>/store.sqlite": shard #1 "<dir>/store-shard0001.sqlite" differs from the tables this code declares (its table "GkSource" has the column(s) "legacy_note", which this code does not declare), so nothing was read
extra table:   Cannot open sharded datastore "<dir>/store.sqlite": shard #1 "<dir>/store-shard0001.sqlite" differs from the tables this code declares (it has the table(s) "retired_table", which this code does not declare), so nothing was read
```

From either constructor, the primary (identical for `read_only=False` and `True`):

```
Cannot open sharded datastore "<dir>/store.sqlite": the primary "<dir>/store.sqlite" differs from the tables this code declares (it has the table(s) "retired_table", which this code does not declare), so nothing was read
Cannot open sharded datastore "<dir>/store.sqlite": the primary "<dir>/store.sqlite" differs from the tables this code declares (its table "sharded_tables" lacks the column(s) "key_attr"), so nothing was read
Cannot open sharded datastore "<dir>/store.sqlite": the primary "<dir>/store.sqlite" differs from the tables this code declares (it lacks the table(s) "replication_in_flight"), so nothing was read
```

`python -m RunRegistry store fingerprint` on a full store without `QuadSourceIntegral_tags` on
shard 1: exit 1, stdout empty, stderr `!! Cannot read sharded datastore "<dir>/full-store.sqlite":
shard #1 "<dir>/full-store-shard0001.sqlite" differs from the tables this code declares (it lacks
the table(s) "QuadSourceIntegral_tags"), so nothing was read`, no traceback, and every file in the
store's directory unchanged (`file_state`).

## The fingerprint (§3.3)

A scratch script built `build_full_store` in a temporary directory, took
`fingerprint_of(read_inventory(primary))`, removed `taken`, and wrote it as `json.dumps(…,
sort_keys=True, default=str)`; once with the staged tree and once with an export of `6ba15c7`
(`git archive HEAD`, run with `PYTHONPATH` at the export). The two files are byte-identical
(`cmp`). Fingerprint `digest` **`6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759`**,
`problems` `{}`; SHA-256 of the whole fingerprint without `taken`:
`05e1cf3b3d4dd8f4d79faeeda065fda085777eff9cfef6eb29d7dee42fa53053`.

## Deviations from the prompt

1. **`SchemaDifferences`' column maps hold only tables with a difference** (the reader's old
   `absent_columns`/`extra_columns` mapped every table present, most to `()`), and it has a
   `describe()` beside `empty`, which `StoreSchemaMismatch` uses for its message. `IMPLEMENTATION
   CHOICE` (the prompt leaves the return type to the implementer).
2. **`StoreSchemaMismatch` carries `verb`** beside the primary, the file and the differences, and
   both verbs end "so nothing was read". A read-write open may already have rolled back a hot
   journal (step 1, or the primary's first read), so it does not claim "nothing was written".
   `IMPLEMENTATION CHOICE`.
3. **A read-write refusal names every difference of the refused shard**, an absent table included,
   although only the other kinds cause it. Pinned by
   `test_an_absent_table_beside_another_difference_is_still_refused`. `IMPLEMENTATION CHOICE`.
4. **The two checks are methods**, `_refuse_shards_that_differ` (called as step "1b" of the
   reconcile) and `_refuse_a_primary_that_differs` (called by both constructors), and the shard
   check opens each shard mode=ro through `store_reader.read_only_url`, imported at function scope.
   `IMPLEMENTATION CHOICE`.
5. **`_read_replicated_tables` keeps one `PRAGMA table_info` per table**, now only to detect an
   absent table, which it still reads as `{"columns": None, "rows": {}, …}` exactly as before (U1).
   `IMPLEMENTATION CHOICE`: the prompt keeps the absent-table reading and removes only the subset.
6. **`_combine` keeps its local name `comparable`**, now `sorted(reads)`, to keep the diff to the
   filter. `IMPLEMENTATION CHOICE`.
7. **The read-only pool conversion** labels its sub-tests `table=…, validates=…` (so the loop's
   `validates` stays meaningful), counts constructed actors as `test_reconcile_at_open` does, and
   records each message in the module's `MESSAGES`. `IMPLEMENTATION CHOICE`.
8. **`test_store_reader`'s two conversions cover all four kinds** (one sub-test per kind, each its
   own store), so `test_store_schema` has no further per-kind reader test; it adds the "nothing
   yielded, every engine disposed" test instead. `IMPLEMENTATION CHOICE`.
9. **Five tests beyond the prompt's list** in `test_store_schema`: `test_it_issues_no_write`,
   `test_a_refusal_yields_nothing_and_disposes_every_engine`,
   `test_an_absent_table_beside_another_difference_is_still_refused`,
   `test_a_prune_record_is_not_completed` (the prompt says the check covers step 2) and
   `test_a_fresh_store_has_no_difference`. `IMPLEMENTATION CHOICE`.
10. **`real_store_fixtures._write_shard`'s comment** "an old store's table lacks a column the code
    now declares" reads "a shard whose table lacks a column the code declares". Comment only.
    `IMPLEMENTATION CHOICE`.
11. **`test_store_inventory.TestOldStoresAndOrphans` keeps its name**, though it now holds only
    orphan and parent tests, so that the ids of its surviving tests do not change. `IMPLEMENTATION
    CHOICE`.

No `UNINTENDED DRIFT`.

## Verification performed

- **Suites**, serially on the staged tree, README §7's command, output under the session
  scratchpad (`orch05/agent-work/final_*.txt`), verdicts grepped from the files, at load 11–40.
  Datastore in four groups (a Bash call cannot run longer than ten minutes):
  122 + 131 + 200 + 191 = **644** (before: 122 + 132 + 201 + 185 = 640).

  | Suite | Before (orchestrator, `6ba15c7`) | After | Difference |
  |---|---|---|---|
  | AdaptiveLevin | 32 OK | 32 OK | 0 |
  | ComputeTargets | 693 OK | 693 OK | 0 |
  | CosmologyModels | 39 OK | 39 OK | 0 |
  | Datastore | 640 OK | 644 OK | −9 deleted, +13 in `test_store_schema` (the three conversions are renames) |
  | LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 0 |
  | RunRegistry | 280 OK | 280 OK | 0 |

  By group: `[a-f]` 122 → 122; `[g-o]` 132 → 131 (`test_gksource_parent_set` −1); `[p-r]` 201 →
  200 (`test_read_only_pool` −1); `[s-z]` 185 → 191 (`test_store_inventory` −7,
  `test_store_schema` +13). No test outside §2's plan failed; no wall-clock flake fired. No
  RunRegistry test changed.
- **The §3.2 greps**, at the staged tree:
  - `grep -rnE 'absent-table|unkeyed|skipped_validation' Datastore/ --include='*.py' --exclude-dir=tests`: nothing (exit 1);
  - `grep -n 'absent_tables\|absent_columns' Datastore/store_inventory.py Datastore/SQL/Datastore.py`: nothing (exit 1);
  - `grep -nE '"kind": "columns"|kind == "columns"|\("columns", ' Datastore/SQL/ShardedPool.py`: nothing (exit 1);
  - `grep -rn 'whether or not a parent is given' Datastore/SQL/ObjectFactories/`: nothing (exit 1);
  - `grep -rn 'build_old_store\|OLD_MISSING' --include='*.py' .`, outside `venv/`: nothing outside
    `prompts/datastore-generic/orchestrator/measure-05/scripts/` (`run_store_reader_shim.py`,
    `run_shims.py`: the measurement's own shims, which supply the deleted fixture and are not run
    by any suite).
  - `ReadOnlyShard`'s fields, by `dataclasses.fields`: `['serial', 'path', 'engine', 'tables']`.
- **Current stores have no difference** (scratch script, and now `test_store_schema`): every shard
  and the primary of `build_real_store` and `build_full_store`, and of a store a fresh pool
  created on stand-in shards.
- **`black --check`** on the 19 touched Python files: clean.
- **Staging** by explicit path; `git diff --cached --name-only` shown before the commit.

## The deliberate-breakage record

**How each was run.** The code and tests were staged; each mutation was made in the working tree
and its diff taken with `git diff -- <file>` against the index (the tree of this commit), so
`git apply` replays it from a clean checkout of this commit. For each: `git apply -R --check`, the
run, `git apply -R`, then `git apply --check` (forward) on the restored tree; all passed, and
`git diff` was empty after each. None is committed. Each was run, in `bash`, on six modules:
`test_store_schema`, `test_store_reader`, `test_read_only_pool`, `test_reconcile_at_open`,
`test_version_row_at_open` and `test_wkb_numeric_parent_key` (143 tests; OK unmutated).

**(i) The reader lets an absent table through.** Ran 143, **FAILED (failures=3, errors=2)**:
- `test_store_reader.TestADifferingSchemaIsRefused.test_a_missing_table_or_column_is_refused_naming_the_shard`
  (sub-test "a missing table": the store was opened);
- `test_read_only_pool.TestAtOpen.test_a_shard_lacking_a_table_is_refused_naming_it`, both
  sub-tests (`GkSource`: error, `OperationalError: no such table: GkSource` from the read-only
  actor's validation; `OneLoopIntegral`: fail, `StoreSchemaMismatch not raised`, the pool opened);
- `test_store_schema.TestTheReader.test_a_refusal_yields_nothing_and_disposes_every_engine` (fail:
  the store was opened);
- `test_store_schema.TestTheInventory.test_read_inventory_refuses_a_store_missing_a_table` (error:
  `OperationalError: no such table: QuadSourceIntegral_tags`).

```diff
diff --git a/Datastore/store_reader.py b/Datastore/store_reader.py
index 07c7310..d4811a4 100644
--- a/Datastore/store_reader.py
+++ b/Datastore/store_reader.py
@@ -122,7 +122,7 @@ def _describe_shard(
         differences = schema_differences(conn, tables)
         conn.rollback()
 
-    if not differences.empty:
+    if not differences._replace(absent_tables=()).empty:
         raise StoreSchemaMismatch(
             _VERB, primary, f'shard #{serial} "{str(path)}"', differences
         )
```

**(ii) The read-write check refuses an absent table too.** Ran 143, **FAILED (errors=3)**: the
three U1 tests, `TestTheReadWriteOpenRecoversAnAbsentTable.test_an_interrupted_drop_of_a_sharded_class`,
`…test_an_interrupted_drop_of_a_replicated_class` and `…test_an_interrupted_first_open`, each
erroring with `StoreSchemaMismatch: Cannot open …` naming the shard it was given. Nothing else
failed: `test_version_row_at_open.TestNewStoreInterrupted.test_interrupted_at_the_version_write`,
`test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_store` and
`test_wkb_numeric_parent_key.TestDropOrder` pass under it, because no shard of theirs lacks a table
when it is reopened. The three new tests are U1's only pins.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 40fc1e8..61d12dc 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -1190,7 +1190,7 @@ class ShardedPool:
                     conn.rollback()
             finally:
                 engine.dispose()
-            refused = differences._replace(absent_tables=())
+            refused = differences
             if not refused.empty:
                 raise StoreSchemaMismatch(
                     "open",
```

**(iii) The read-write check is skipped.** Ran 143, **FAILED (failures=5)**, every one
`StoreSchemaMismatch not raised` (the store opened):
`TestTheReadWriteOpenRefuses.test_an_absent_column_an_extra_column_and_an_extra_table` (all three
sub-tests), `…test_an_absent_table_beside_another_difference_is_still_refused` and
`…test_a_prune_record_is_not_completed` (the prune was completed). (A first version of the last
test named `GkSourcePolicy` in its record; under (iii) it then errored with `ReplicatedDivergence`,
because that class does not prune at open. It was changed to `BackgroundModel`, the one replicated
class that does, so that the mutation shows the record completed; the run above is with that
version.)

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 40fc1e8..00f7447 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -1049,7 +1049,6 @@ class ShardedPool:
         # 1b. every shard holds the declared tables with exactly their declared columns, or the
         # store is refused by name, before the record is read and before anything is compared or
         # written
-        self._refuse_shards_that_differ(built, shard_ids)
 
         specs = self._replicated_table_specs(built)
         reconciliation["compared"] = [spec["name"] for spec in specs]
```

**(iv) The primary check is skipped in both constructors.** Ran 143, **FAILED (failures=2,
errors=4)**:
- `test_reconcile_at_open.TestNoRecordRefuses.test_a_primary_without_the_record_table_is_refused`,
  both sub-tests, error: `OperationalError: no such table: replication_in_flight`;
- `test_store_schema.TestThePrimaryIsRefused.test_an_extra_table_and_a_missing_column`: "an extra
  table" fails in both sub-tests (`StoreSchemaMismatch not raised`: the store opened); "a missing
  column" errors in both (`OperationalError: no such column: sharded_tables.key_attr`).

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 40fc1e8..9243148 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -222,7 +222,6 @@ class ShardedPool:
             # the primary holds exactly the tables _create_engine declares, or the store is
             # refused by name before anything reads them. This is the primary's first read, so it
             # is what rolls back a hot journal on the primary
-            self._refuse_a_primary_that_differs()
 
             self._read_shard_data()
 
@@ -434,7 +433,6 @@ class ShardedPool:
         self._engine = sqla.create_engine(
             read_only_url(self._primary_file), future=True, connect_args=connect_args
         )
-        self._refuse_a_primary_that_differs()
         self._read_shard_data()
         self._check_shard_files()
 
```

**(v) `schema_differences` ignores extra columns.** Ran 143, **FAILED (failures=3)**:
- `test_store_schema.TestSchemaDifferences.test_each_kind_is_found_on_the_right_shard_by_name`;
- `test_store_schema.TestTheReadWriteOpenRefuses.test_an_absent_column_an_extra_column_and_an_extra_table`
  (sub-test "an extra column");
- `test_store_reader.TestADifferingSchemaIsRefused.test_an_extra_column_or_table_is_refused_naming_the_shard`
  (sub-test "an extra column").

```diff
diff --git a/Datastore/SQL/schema.py b/Datastore/SQL/schema.py
index 4f7f5ac..a45b2bc 100644
--- a/Datastore/SQL/schema.py
+++ b/Datastore/SQL/schema.py
@@ -224,7 +224,7 @@ def schema_differences(conn, tables: Mapping[str, sqla.Table]) -> SchemaDifferen
         ]
         declared = [c.name for c in table.columns]
         absent = tuple(c for c in declared if c not in file_columns)
-        extra = tuple(c for c in file_columns if c not in declared)
+        extra = ()
         if len(absent) > 0:
             absent_columns[name] = absent
         if len(extra) > 0:
```

## Observations not acted on

- **No test before this one reached U1.** Under (ii), the three tests the prompt names as U1's
  (`TestNewStoreInterrupted`, `TestPruningAfterRepair.test_an_interrupted_store`, `TestDropOrder`)
  pass: none leaves a shard without a table at the next open. The new U1 tests are the only pins.
- **Under (iii) the read-write open accepted every other kind silently**: an absent
  `TkNumericIntegration.stop_Tprime`, an extra column on the replicated `tolerance`, and an extra
  table all opened with no error. The check at open is the only guard on that path, as intended.
- **`[00-a-drop-action-drops-a-replicated-table-outside-the-in-flight-record]`**
  (`a3-v2-readiness`): measured here, a `GkSourcePolicy` table dropped from one shard while empty
  everywhere opens read-write and the actors re-create it (U1 test). A read-only open of such a
  store is now refused by name. Its board and index row are `a3-v2-readiness`'s and unchanged.
- **Three unused imports, unused at `HEAD` too** (`TkNumericIntegration` in
  `test_gksource_parent_set`, `sqla_TkWKBValue_factory` in `test_wkb_numeric_parent_key`, `Record`
  in `test_store_inventory`). Not touched.
- **`docs/a3-v2-readiness/readonly_open_probe.py:124`** builds its own `_inspector` on a probe
  object; it is a measurement script of a past run, outside §6, and does not use the actor's.

None opens an issue.

## State handed to the next prompt

- A store whose files differ from what the code declares meets `StoreSchemaMismatch`
  (`Datastore/SQL/schema.py`, a `RuntimeError`) from the reader, the inventory, the fingerprint,
  the read-only pool and a read-write open, naming the file and each table and column. Only a
  read-write open allows an absent table, which it reads as empty and the actors create (U1). Both
  constructors refuse a primary that differs from `_create_engine`'s six tables; the closed-store
  operations and the reader read only the `shards` table and do not check the primary.
- `schema_differences(conn, tables)` is the one comparison; 06–08 should not write another.
- `ReadOnlyShard` is `serial`, `path`, `engine`, `tables`; `ShardContext` is `shard`, `keys`,
  `digests`, `all_digests`; `ShardRead` has no `comparable`.
- The split grouping's exclusion of `timestamp` (03) is current behaviour and untouched.
- `Datastore` 644, every other suite unchanged. The index holds 107.
