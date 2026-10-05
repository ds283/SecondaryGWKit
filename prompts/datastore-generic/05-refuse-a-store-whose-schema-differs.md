# Prompt 05 — refuse a store whose schema differs

**Campaign:** [`README.md`](README.md) · **Board items:** **S1**–**S3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** 04 landed (`bbdba04`) and reviewed (`bc3e944`). `git status` clean. Never in one
checkout while another prompt's change to `ShardedPool.py` is uncommitted.
**Closes:** `[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]`.
**Opens:** nothing expected; any issue the work finds.
**Recommended model:** **Opus**. The change decides whether a store opens at all, on every path:
the reader, the inventory and fingerprint built on it, the read-only pool, and the read-write
check at open. On the last path it must keep the crash recovery U1 protects.

**Read first:**

1. [`README.md`](README.md): §0.1, §1, §3, §4, §5 (rules 4, 7–11), §6.1 **D2**, §6.2 **U1**.
2. [`docs/datastore-generic-audit.md`](../../docs/datastore-generic-audit.md): **B5**, **B10**.
3. `Datastore/SQL/schema.py`, all of it.
4. `Datastore/store_reader.py`, all of it.
5. `Datastore/store_inventory.py`: the module docstring, `ShardContext`, `ShardRead`,
   `read_records`, `_combine` and `read_inventory`.
6. `Datastore/SQL/ShardedPool.py`:
   - the constructor's existing-store branch;
   - the comment block "THE READ-ONLY POOL", `_open_read_only` and `_version_labels_present`;
   - `_create_engine`;
   - `_reconcile_replicated_tables` (its step 1 especially), `_read_replicated_tables`,
     `_replicated_differences`, `_describe_differences`, `_plan_replicated_repair`,
     `_read_full_rows`, `_prune_needed` and `_prune_shard`.
7. `Datastore/SQL/Datastore.py`: the read-only branch of the constructor,
   `_validate_on_startup_read_only`, `read_only_state`, `_ensure_tables` and `_drop_actions`.
8. `Datastore/SQL/ObjectFactories/TkWKBIntegration.py` and `GkWKBIntegration.py`: the value
   factories' `_build_impl_model` and `read_batch`.
9. `Datastore/tests/real_store_fixtures.py`, `Datastore/tests/standin_pool.py`, and the test
   modules named in §2's test plan.

Line numbers below are at `bc3e944`. Commits after it, up to this prompt's `HEAD`, change no code.

---

## 1. What is wanted

A shard written by the current code holds exactly the tables `build_schema` declares, each with
exactly the declared columns. A primary holds exactly the six tables `ShardedPool._create_engine`
declares. This was measured on 2026-10-05, on every shard and the primary of
`real_store_fixtures.build_real_store` and `build_full_store`, and on a store the real
`ShardedPool` created on stand-in shards. Every difference was empty.

Today the layer tolerates a difference instead of refusing it (audit B5):
- the reader reports `absent_tables`, `absent_columns`, `extra_tables` and `extra_columns`, and
  opens anyway;
- the inventory turns an absence into the named problems `absent-table` and `incomplete`;
- the read-only pool reports a missing table, opens, and skips its validation;
- the reconcile compares the subset of columns a shard has, marks a table without its key column
  `unkeyed`, and has a `columns` difference kind for shards whose column sets differ;
- four WKB value reads select a column they never use, so that an old table raises.

Under D2 these tolerances go:

- **S1 — one comparison, one refusal.** A store whose files differ from what the code declares is
  refused by name, before anything reads its tables, by one comparison and one exception:
  - the comparison is `schema_differences` and the exception `StoreSchemaMismatch`, both in
    `Datastore/SQL/schema.py` (README §4: "05 names the exception");
  - the reader, and so the inventory and the fingerprint, refuse any difference;
  - the reader's four fields and the inventory's two problems go.
- **S2 — the read-only pool and the reconcile.**
  - The read-only pool refuses through the reader, and its absent-table reporting goes, in the
    pool and in the read-only actor.
  - The reconcile compares the declared columns as they are. Its column-subset logic, `unkeyed`
    and the `columns` difference kind go.
- **S3 — the read-write open, the primary, the selects and the fixtures.**
  - A read-write open refuses a difference before the in-flight record is read, **except an
    absent table** (U1). The actors create the absent table, as they do today. That is recovery
    from an interrupted first open or an interrupted drop, not compatibility.
  - Both constructors refuse a primary that differs from `_create_engine`'s tables. This settles
    the bare `OperationalError` that prompt 03 left for a primary without `replication_in_flight`.
  - The four `numeric_serial` selects go.
  - `real_store_fixtures.build_old_store` goes.

After 05, a store written by older code meets `StoreSchemaMismatch`, naming each table and column.
That is loud and true.

**What it compares.** Table names (from `sqlite_master`, type `table`, excluding names that start
`sqlite_`) and column names (`PRAGMA table_info`). Not types, nullability, keys, defaults or
indexes. The reader compares exactly this today, and nothing in the B5 inventory needs more.

---

## 2. What to change

**S1 — `Datastore/SQL/schema.py`.**
- **`schema_differences(conn, tables)`.**
  - `conn` is a connection to one SQLite file. `tables` maps name to `sqla.Table`.
  - It returns the four differences: tables the file lacks; tables the file has that `tables`
    does not declare; for each declared table the file has, the declared columns it lacks; and
    the file's columns that `tables` does not declare.
  - Names are in declaration order and file order respectively, as the reader orders them today.
  - It reads only `sqlite_master` and `PRAGMA table_info`, and never writes.
  - Its return type is the implementer's choice. A `NamedTuple` with an "is empty" test is
    enough.
- **`StoreSchemaMismatch(RuntimeError)`.**
  - **Attributes:** the primary, a description of the file ("shard #N" with its path, or "the
    primary" with its path), and the differences.
  - **Message:** one sentence, with the verb of the operation that refused ("Cannot read
    sharded datastore …" from the reader, "Cannot open …" from a read-write open). It names the
    file and every table and column that differs, by kind, and says that nothing was read.
  - **One prefix, not two.** `[04-closed-store-refusals-repeat-their-prefix]` records what
    happens when a refusal is wrapped again. Raise this exception where it is built, and do not
    wrap it.
  - It subclasses `RuntimeError` so that every caller that today reports a refusal by catching
    `RuntimeError` still does: `RunRegistry/__main__.py:297` included. Measured: `store
    fingerprint` on a store missing a table then exits 1 with the message on stderr, and writes
    nothing.
- `schema.py`'s docstring says that this module is also the one comparison of a file against the
  declared tables. It still imports only `sqlalchemy`.
- No project table, column or package name enters `schema.py` (README §5 rule 11).

**S1 — `Datastore/store_reader.py`.**
- `_describe_shard` (`:112-150`) calls `schema_differences` with `built.tables`, and raises
  `StoreSchemaMismatch` on **any** difference, an absent table included (U1).
  - It raises inside `open_read_only`'s `try` (`:184-200`), so that every engine made so far is
    disposed. Nothing is yielded.
- `ReadOnlyShard` loses `absent_tables`, `absent_columns`, `extra_columns` and `extra_tables`
  (fields `:66-69`, docstring `:53-60`).
- The module docstring's "Old stores." paragraph (`:25-30`) becomes a paragraph saying that a
  store whose files differ from the declared tables is refused, naming the file and each
  difference. The refusal list in `open_read_only`'s docstring (`:158-161`) gains it.
- An import left unused goes (`Dict`, `FrozenSet` at `:37`, after a grep).

**S1 — `Datastore/store_inventory.py`.**
- The `absent-table` and `incomplete` problems go, with their branches in `read_records`:
  - the class's own table: `:512-520`;
  - `absent` at `:522`, and the key-column check at `:530-540`;
  - the parent-set member table: `:542-573`;
  - the association table: `:591-603`;
  - the value table: `:644-656`.
- `has_validated` (`:576`) and `has_timestamp` (`:579`) stop consulting absent columns.
- `ShardContext` loses `absent_tables` and `absent_columns` (`:295-296`), and `read_inventory`
  stops passing them (`:967-968`).
- **`ShardRead.comparable` is now always true**, because every `comparable = False` was in a
  deleted branch. It goes, with:
  - `_read`'s argument (`:499`, `:505`);
  - the assignment at `:585` and the return at `:819`;
  - the filter in `_combine` (`:849-855`), which then takes every shard.

  Grep first that no factory's `inventory_records` builds a `ShardRead` or reads `comparable`.
- The docstring's two problem names (`:46-48`) and its "left out of that class's replicated
  comparison" paragraph (`:70-72`) go. `FrozenSet` (`:90`) goes if unused.
- **The fingerprint is unchanged for every store the current code writes.** Measured: the
  fingerprint of `build_full_store`, with `taken` removed, was byte-identical before and after a
  rough version of this change (digest `6576b85d…`, problems `{}`). §3.3 measures it again.

**S2 — the read-only pool.**
- `ShardedPool._open_read_only`:
  - the block at `:409-421` loses `self.absent_tables` and its print;
  - `open_read_only` is still entered for its refusals, now including the schema refusal, and a
    comment says so;
  - the "THE READ-ONLY POOL" comment (`:368`) stops saying that absent tables are reported.
- `_version_labels_present` (`:567-588`) loses the branch for a shard without the `version`
  table (`:575-582`), and its docstring clause (`:569-570`). Only the read-only pool calls it, and
  that pool now refuses such a shard first.
- **`Datastore/SQL/Datastore.py`**, the read-only actor:
  - the absent-table block and its comment (`:348-357`) go;
  - so does `_skipped_validation` (`:359`), and the skip in `_validate_on_startup_read_only`
    (`:375-382`, with its docstring sentence `:365-367`);
  - `read_only_state` (`:461-469`) returns `read_only` only, and its docstring says why the pool
    still calls it (it waits on every constructor);
  - `self._inspector` (`:336`), if nothing on the read-only path uses it any more.

  **The read-write actor is not touched.** `_ensure_tables` and `_drop_actions`' `has_table`
  skip are B10, and U1's recovery.

**S2 — the reconcile (`ShardedPool`).**
- `_read_replicated_tables` (`:1223-1261`):
  - **keeps** "a table the shard lacks is read as empty". That is U1, reached by a read-write
    open;
  - selects `spec["compared"]` as it is, with no `present` subset and no `unkeyed`;
  - its docstring says that the schema check at open has already refused every other
    difference.
- The `columns` difference kind goes:
  - from `_replicated_differences` (`:1285-1296`, and its docstring at `:1273`);
  - from `_describe_differences` (`:1412-1423`);
  - from `_plan_replicated_repair` (`:1560`, `("columns", "duplicate")` → `"duplicate"`).
- `_read_full_rows` (`:1719-1738`) selects every column of the table (no `present` filter).
- **Kept:** `_prune_needed`'s `has_table` skip (`:1977`), and `_prune_shard`'s (`:1998-2000`).
  Both are U1.

**S3 — the read-write open (`ShardedPool._reconcile_replicated_tables`).**
- Insert the schema check after step 1 (`:1000-1016`, the hot-journal rollback) and once `built`
  is made (`:1018`), **before** `_read_in_flight_record` (`:1022`). It so covers the
  interrupted-prune branch (step 2) as well as the comparison.
- Every shard is compared with `schema_differences` against `built.tables`, on a connection that
  reads only. After step 1 no journal is left to roll back.
- **An absent table is allowed here**, and nothing else is: an absent column, an extra column or
  an extra table raises `StoreSchemaMismatch` ("Cannot open …"), before any actor or the broker
  exists and before anything is written.
- Say in a comment why the absent table is allowed: U1, an interrupted first open (each
  `create` of `_ensure_tables` commits on its own) or an interrupted drop, which the actors
  complete by creating the table.

**S3 — the primary, in the constructors only.**
- The primary file is compared with `schema_differences` against the six tables `_create_engine`
  declares (`:660-721`):
  - `shards`, `shard_key_config`, `shard_keys`, `replicated_tables`, `sharded_tables` and
    `replication_in_flight`;
  - no difference allowed. An existing primary is written whole by `_write_shard_data`.
- **Read-write:** after `_create_engine()` (`:216`) and before `_read_shard_data()` (`:217`), on
  the pool's read-write engine. Its first read rolls back a hot journal on the primary, as
  `_read_shard_data`'s does today; `primary_journal` is noted before that, as now.
- **Read-only:** after the mode=ro engine is made (`:430-432`) and before `_read_shard_data()`
  (`:433`).
- **Not** in `_read_closed_store`, `open_read_only`, `copy_store`, `move_store`,
  `closed_store_files` or `delete_store`. Each of these reads only the primary's `shards` table,
  and `shard_store_fixtures.write_hand_built_primary` builds a primary for them without
  `replication_in_flight` on purpose (prompt 04). So the reader, the inventory and the
  fingerprint read a primary without `replication_in_flight`, and the pool refuses it. That is
  intended: the fingerprint is of the shards' content.
  *Confirmed by the user, 2026-10-05.*

**S3 — the four `numeric_serial` selects.** In the value factories, each selects
`wkb_table.c.numeric_serial` "whether or not a parent is given, so that a table without the column
raises". Nothing reads the value. Delete the selected column and its two-line comment in each:
- `TkWKBIntegration.py`: `:1243-1245` (`_build_impl_model`) and `:1363-1365` (`read_batch`);
- `GkWKBIntegration.py`: `:1236-1238` (`_build_impl_model`) and `:1358-1360` (`read_batch`).

The conditional filters on a given parent stay (Tk `:1259-1262`, `:1378-1381`; Gk `:1252-1255`,
`:1391-1394`).

**S3 — `Datastore/tests/real_store_fixtures.py`.**
- Delete `build_old_store` (`:360-369`) and the `OLD_MISSING_*` constants (`:194-198`).
- **Keep** `build_real_store`'s `missing_tables`, `missing_columns` and `extra_sql`. They build
  the shapes the refusal tests need.
- The module docstring's "Old stores." paragraph (`:25-29`) becomes one about building a store
  whose schema differs, for the refusal tests.

**The tests (S1–S3).** The plan was measured on 2026-10-05 at `bc3e944`, with a rough version of
this change in a scratch worktree. After it, Datastore ran 622 with 6 failures and 23 errors.
Every one is listed below. RunRegistry (280) and ComputeTargets (693) were unchanged. README §5
rule 7 applies to every test not named here.

- **Delete** (each exists only to assert that a difference is tolerated, reported or named):

  | Module | Class.method (line at `bc3e944`) |
  |---|---|
  | `Datastore/tests/test_store_inventory.py` | `TestOldStoresAndOrphans.test_an_absent_table` (874) |
  | | `TestOldStoresAndOrphans.test_an_absent_association_table` (883) |
  | | `TestOldStoresAndOrphans.test_an_absent_value_table` (892) |
  | | `TestOldStoresAndOrphans.test_an_absent_replicated_table` (898) |
  | | `TestOldStoresAndOrphans.test_an_absent_replicated_association_table` (906) |
  | | `TestOldStoresAndOrphans.test_a_missing_key_column_makes_the_class_incomplete` (914) |
  | | `TestOldStoresAndOrphans.test_a_missing_column_the_key_does_not_need` (952) |
  | `Datastore/tests/test_gksource_parent_set.py` | `TestTheInventoryResolvesTheSet.test_a_store_without_the_table_is_named_absent` (902) |
  | `Datastore/tests/test_read_only_pool.py` | `TestAtOpen.test_a_lookup_on_a_missing_table_fails_in_sqlite` (850) |

  That is nine, all in `Datastore`. `test_store_inventory`'s `check` helper (`:842`) and
  `data/schema_at_datastore-integrity-06.json` stay: the orphan tests and `test_schema_builder`
  use them. That module's docstring (`:14`) is corrected.
- **Convert to a refusal** (the shape stays; the assertion becomes the refusal). The method may be
  renamed; the log says so.
  - `test_store_reader.TestOldStores.test_missing_table_and_column_are_reported_on_the_right_shard`
    (200): the reader refuses each shape with `StoreSchemaMismatch`, naming the shard, and the
    table or the table and column; `file_state` is unchanged.
  - `test_store_reader.TestOldStores.test_extra_columns_and_tables_are_reported` (233): the same,
    for an extra column and an extra table. The class may be renamed.
  - `test_read_only_pool.TestAtOpen.test_a_shard_lacking_a_table_is_reported_and_the_store_opens`
    (829), both sub-tests: the read-only pool refuses with `StoreSchemaMismatch` naming the
    table. No actor is constructed, and the checksums are unchanged. **This is the one intended
    move of behaviour for a shape the current code can reach** (an interrupted drop): only a
    read-write open recovers it now (U1).
- **Re-fixture** (the test used a missing table only to cause something else). Both were measured.
  - `test_inventory_consumers.TestRunLabels.test_a_problem_elsewhere_does_not_matter` (271):
    `extra_sql={1: ['INSERT INTO "QuadSourceIntegral_tags" (parent_serial, tag_serial) VALUES
    (999, 1)']}` replaces `missing_tables`. The inventory then holds exactly one problem, an
    `orphan-tag` on `QuadSourceIntegral`, and `available_run_labels` still returns
    `['fixture']`.
  - `test_inventory_report.TestTheDisplaySaysWhatTheInventoryHolds.test_every_problem_appears_in_full_when_there_are_many`
    (236): seven shards, with orphan tag rows replacing the absent tables.
    - One orphan row goes in `QuadSourceIntegral_tags` on each of shards 1–6, and on shard 1 one
      in each of the seven tag tables.
    - Each table's parent column is its foreign key that is not to `store_tag`: `parent_serial`
      in `QuadSourceIntegral_tags`, `integration_serial` in `TkNumeric_tags`.
    - Measured: 12 problems, 6 on `QuadSourceIntegral`, against `MAX_SHOWN` 5. Both of the
      test's inequalities hold.
- **Change an assertion or a construction** (named, with the reason):
  - `test_reconcile_at_open.TestNoRecordRefuses.test_a_primary_without_the_record_table_is_refused`
    (763): `assertRaises(OperationalError)` (`:780`) becomes `StoreSchemaMismatch`, for both
    `read_only` values. Measured: with that change alone, both pass. The message names
    `replication_in_flight`, no actor is constructed, the checksums are unchanged, and the table
    is not created. The `OperationalError` import (`:46`) is then unused and goes.
  - `test_database_errors_reach_the_caller.TestTheWKBValueReads` (416):
    - its four tests (444, 460, 476, 492) assert that the failing statement names
      `numeric_serial`, which a read naming no parent no longer selects;
    - `columns=(RHO_GAUSS_ORDER,)` replaces `(NUMERIC_SERIAL,)`, and all four pass (measured).
      `RHO_GAUSS_ORDER` is at `:104`;
    - the class docstring (`:417-419`) is corrected;
    - the harness and the other classes are untouched.
  - Built by hand, `ShardContext` loses `absent_tables=frozenset(), absent_columns={}`:
    - in `test_quadsource_parent_key.py:445-448`, which fixes the five failing
      `TestCrossShardParentUnit` tests;
    - in `test_wkb_numeric_parent_key.py:993-996`, which fixes the two `TestNullableParent`
      tests.
  - `test_store_reader`, beyond the conversions:
    - the imports of `build_old_store` and `OLD_MISSING_*` (`:36-41`) go;
    - `_readable_columns` and `read_every_table` (`:52-73`) select every column of every table;
    - `_CHILD`'s absent-table skip (`:332-333`) goes;
    - `TestReaderReads.test_every_shard_with_its_serial_and_path` loses its assertions on the four
      removed fields (`:99-104`);
    - `TestNoRay.test_reading_in_a_child_interpreter_leaves_ray_uninitialised`'s `expected`
      (`:357`) becomes `sum(len(s.tables) for s in opened.shards)`;
    - the module docstring (`:9-10`) is corrected.
- **Must pass unchanged**, among them the crash-recovery tests U1 protects:
  - `test_version_row_at_open.TestNewStoreInterrupted.test_interrupted_at_the_version_write`;
  - `test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_store`, which uses
    `drop_actions`;
  - `test_wkb_numeric_parent_key.TestDropOrder`;
  - every other user of `standin_pool`;
  - `test_copy_move_store`, `test_delete_store` and `test_absolute_shard_record_refused`.
- **New tests.**
  - **`Datastore/tests/test_store_schema.py`** (new).
    - **`schema_differences`:**
      - every shard and the primary of `build_real_store` and `build_full_store` give no
        difference;
      - each of the four kinds is found, on the right shard, by name.
    - **The reader** refuses each kind. Two of them are the conversions above; one test per
      remaining kind is enough.
    - **`read_inventory`** refuses a store missing a table with `StoreSchemaMismatch`, as
      `store fingerprint` inherits it.
    - **The read-write open, through `standin_pool`:**
      - an absent column, an extra column and an extra table are each refused with
        `StoreSchemaMismatch`;
      - the refusal comes before any actor is constructed and before the in-flight record is
        read;
      - every file's checksum is unchanged.
    - **U1, the read-write open of a shard that lacks tables, opens.** No test covered it before.
      Three cases, as measured:
      - the `gk-source` tables (`GkSource`, `_tags`, `Value`, `_parents`) dropped from shard 0;
      - the replicated `GkSourcePolicy` dropped from shard 1 and empty elsewhere;
      - one shard lacking half its tables, with no version row anywhere (an interrupted first
        open).

      Each opens. Afterwards every shard has no schema difference and holds the version row.
    - **The primary**, read-write and read-only: an extra table and a missing column are each
      refused, naming "the primary". The missing `replication_in_flight` case is the changed
      test above.

---

## 3. Verification

1. **Every suite**, with counts before and after, and each difference accounted for:
   - Datastore: 640 − 9 + the new tests (640 since `b97cf06` added four audit-tool tests);
   - RunRegistry: 280, unchanged;
   - ComputeTargets: 693, unchanged.

   No RunRegistry test changes.
2. **Greps.** Each of these prints nothing; show the output:
   - `grep -rnE 'absent-table|unkeyed|skipped_validation' Datastore/ --include='*.py' --exclude-dir=tests`
     (10 lines at `bc3e944`);
   - `grep -n 'absent_tables\|absent_columns' Datastore/store_inventory.py Datastore/SQL/Datastore.py`;
   - `grep -nE '"kind": "columns"|kind == "columns"|\("columns", ' Datastore/SQL/ShardedPool.py`
     (3 lines at `bc3e944`: `:1294`, `:1412`, `:1560`);
   - `grep -rn 'whether or not a parent is given' Datastore/SQL/ObjectFactories/`;
   - `grep -rn 'build_old_store\|OLD_MISSING' --include='*.py' .`, outside `venv/`.

   `ReadOnlyShard` has exactly the fields `serial`, `path`, `engine` and `tables`.
3. **The fingerprint**, from a scratch script, not committed. The fingerprint of
   `build_full_store` with `taken` removed is byte-identical at this commit and at an export of
   its parent. Give the digest.
4. `black --check` is clean on every touched Python file.
5. **Deliberate breakage.** Each mutation must make a named test fail. Then restore it.
   - (i) The reader lets an absent table through. The reader's absent-table refusal fails, and so
     does the read-only pool's converted test.
   - (ii) The read-write check refuses an absent table too. The U1 tests fail.
   - (iii) The read-write check is skipped. The read-write refusals fail.
   - (iv) The primary check is skipped in both constructors.
     `test_a_primary_without_the_record_table_is_refused` fails or errors in both sub-tests, and
     so do the new primary tests.
   - (v) `schema_differences` ignores extra columns. The extra-column refusals fail.

   Record each in the log as a diff, exactly as applied, with the failing tests.

---

## 4. Acceptance

1. One comparison (`schema_differences`) and one exception (`StoreSchemaMismatch`), both in
   `Datastore/SQL/schema.py`, used by the reader, the read-write check at open and both
   constructors' primary check. No other code in `Datastore/` compares a file's tables or columns
   to tolerate a difference, apart from U1's absent table on the read-write path (§2).
2. The reader, the inventory, the fingerprint and the read-only pool refuse any difference. The
   read-write open refuses all but an absent table, and recovers that one as before.
3. Every test of §2's test plan has the fate listed, and the log lists each by module, class and
   method. The new tests exist, and the deliberate-breakage record (i)–(v) is in the log.
4. `[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]` closed:
   - its board entry moves to §4 with a dated "Closed (prompt 05)" line naming what 05 deleted;
   - its index row is deleted, and the count corrected.
5. `black --check` clean. On this board: the §1 row, items S1–S3, §3/§4 and §5. All in the same
   commit.

---

## 5. Stop conditions — stop and ask the user

- A test not named in §2's test plan fails after the change. That is current behaviour moving, or a
  tolerance the inventory missed. Never edit it to pass.
- A store written by the current code shows any difference other than U1's absent table on a
  read-write open.
- The U1 cases cannot be made to open without tolerating more than an absent table.
- The change would write, open or touch any store under `var/` (README §3), or change what any
  code writes (README §5 rule 9).

---

## 6. What this prompt does not do

- **Files it changes:**
  - `Datastore/SQL/schema.py`, `Datastore/store_reader.py` and `Datastore/store_inventory.py`;
  - `Datastore/SQL/ShardedPool.py` and `Datastore/SQL/Datastore.py` (the read-only actor only);
  - `Datastore/SQL/ObjectFactories/TkWKBIntegration.py` and `GkWKBIntegration.py` (the four
    selects only);
  - `Datastore/tests/real_store_fixtures.py`;
  - the test modules of §2's test plan, and the new one;
  - the log, this board and `docs/OPEN_ISSUES.md`.

  Nothing else.
- **It does not touch `tools/shard_key_audit.py`.** It reads files with `sqlite3` directly and
  never goes through the reader. Since `b97cf06` (2026-10-05) it refuses with exit code 2 a primary
  without a column it reads, a shard 0 that is not a database, and a shard 0 without the
  shard-key table, so it does not tolerate what 05 refuses.
- It does not compare types, keys or indexes (§1).
- It does not touch:
  - `copy_store`, `move_store`, `closed_store_files` or `delete_store`, which read only the
    `shards` table;
  - `RunRegistry/` code;
  - `main.py`;
  - the extraction scripts.
- It does not fix `[00-a-drop-action-drops-a-replicated-table-outside-the-in-flight-record]`
  (`a3-v2-readiness`). U1 bears on it but does not decide it. A read-only open of a store left by
  such a drop is now refused by name rather than opened.
- It changes no table, column or key (README §5 rule 9).

---

## 7. The log and the board

`logs/05-refuse-a-store-whose-schema-differs.md`, using README §5.1. In addition, it gives:
- every test deleted, converted, re-fixtured, changed and added, each by module, class and
  method;
- the helpers and fields removed;
- the measured message for each kind, from the reader and from a read-write open;
- the fingerprint digest of §3.3.

`IMPLEMENTATION_STATE.md`:
- the §1 row for 05, and items S1–S3;
- the fallbacks issue closed into §4;
- §5.

`docs/OPEN_ISSUES.md`: the fallbacks row deleted, the count and the date.
