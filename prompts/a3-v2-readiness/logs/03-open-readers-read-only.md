# Log — prompt 03: open the readers read-only

**Subject:** *Open the extraction scripts on a read-only pool* · **Model:** Claude Opus 5.5 ·
**Date:** 2026-10-03 · **Base:** `e4d6ccd`, clean · **Result:** landed. `ShardedPool(read_only=True)`
opens an existing store with every file `mode=ro` and writes nothing; a lookup that would insert
raises `ReadOnlyMiss`, and any other write raises `ReadOnlyWrite`. The six readers open it, take
`--models`, and skip a model whose background the store does not hold.
`[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]` and
`[10-extraction-scripts-exit-1-on-a-store-without-every-model]` are closed on the
`datastore-integrity` board. No stop condition of the prompt's §5 fired. The commit is the one that
adds this file ("this commit" below, since its SHA does not exist yet).

The orchestrator's facts for this prompt (its note, checked at `f1999b1`) are cited below as "the
orchestrator's", where this prompt relies on them without re-deriving them.

## What shipped

Line counts are `git diff --cached --numstat` of this commit.

- **`Datastore/replication.py`** (95 added, 0 deleted): `ReadOnlyMiss(class_name, payload, store,
  detail)` and `ReadOnlyWrite(store, operation, class_name, detail)`, beside
  `ReplicatedDivergence`. Both subclass `RuntimeError`, as the module's other exceptions do, and
  pickle with their fields (`__reduce__`), since under Ray an actor's exception reaches the driver
  pickled. The module docstring gains one paragraph.
- **`Datastore/SQL/ShardedPool.py`** (302 added, 0 deleted):
  - `__init__` gains `read_only: bool = False`, last. After the attributes are set, `read_only`
    sends construction to `_open_read_only` and returns; no line of the read-write constructor
    below it runs for a read-only pool.
  - New: `_open_read_only`, `_check_replicated_tables_read_only`, `_version_labels_present`,
    `_get_impl_replicated_table_read_only`, `_refuse_write`, and the comment block above them,
    which lists the order of events at a read-only open.
  - Guards, each `if getattr(self, "_read_only", False):` followed by new code only: at the top of
    `_get_impl_replicated_table` (to the read-only get), of `object_store` and of
    `object_validate` (`ReadOnlyWrite`), and in `_assign_shard_keys` after its early return, before
    the insert (`ReadOnlyWrite`).
  - `ReadOnlyMiss` and `ReadOnlyWrite` are imported by a second `from Datastore.replication import`
    line, so they are importable from the pool's module and the existing import is untouched.
  - `_replicated_write`, `_plan_replicated_repair`, `_prune_replicated_tables`,
    `_complete_interrupted_prune`, `_find_version_row` and `_reconcile_replicated_tables` are
    unchanged (an `ast.unparse` comparison of each function against `e4d6ccd`).
- **`Datastore/SQL/Datastore.py`** (221 added, 0 deleted):
  - `__init__` gains `read_only: bool = False`, last, and a docstring paragraph. After the factory
    registration it sets `self._read_only`, and under `read_only` calls `_open_read_only`, sets
    `_read_table_config` and returns.
  - New: `_open_read_only`, `_validate_on_startup_read_only`, `_refuse_insert`,
    `_read_only_refusal`, `read_only_state`, and a comment block.
  - Guards: `object_store` and `object_validate` raise `ReadOnlyWrite` at their top;
    `_inserters_for` returns the actor's (refusing) inserters whatever the timestamp; the existing
    `except SQLAlchemyError` of `object_get` and `object_read_batch` calls `_read_only_refusal`
    before its `raise e`. `_read_only_refusal` returns at once on a read-write actor.
  - `import sqlite3`, and a second `from Datastore.replication import` line.
- **The six `extract_*.py`** (10 added, 8 deleted each): `read_only=True` in place of
  `prune_unvalidated=False`; `add_models_argument(parser)` after `add_run_selection_argument`;
  `model_list = select_models(build_model_list(pool, units), args.models)`;
  `read_models(model_list, run_pipeline)` in place of the `for` loop; and in `run_pipeline`,
  `require_background(model, model_label)` in place of the `if not model.available: raise
  RuntimeError(...)`. Four names added to the `from extract_common import (...)` list. Nothing
  else: every lookup is where it was, so `test_main_plumbing` and `test_quadrature_tolerance_pin`
  pass unchanged.
- **`extract_common.py`** (110 added, 0 deleted): `add_models_argument` and `select_models` beside
  `add_run_selection_argument`; `NO_BACKGROUND_REASON`, `NoBackgroundModel`, `require_background`
  and `read_models` after `describe_background_generation`.
- **`Datastore/tests/test_read_only_pool.py`** (new, 24 tests) and
  **`ComputeTargets/tests/test_reader_read_only.py`** (new, 17 tests), below.
- **`Datastore/tests/standin_pool.py`**: not changed. Nothing was added to it.
- Not touched: `Datastore/store_reader.py`, `config/model_list.py`, `config/sharding.py`,
  `main.py`, every factory and driver, every existing test module, every probe.

### M1's table, as implemented

M1's table predates prompt 02. Its row "the version insert in `Datastore.__init__`" is now the
pool's `_find_version_row` and its replicated version write, and the prune row is now the pool's.
So this copy gains a row for each of 02's writes at open: `_prune_replicated_tables`,
`_complete_interrupted_prune` and the replicated version write (with `set_version` beside it).
Function names are `ShardedPool`'s unless marked `Datastore.`; "the read-only open" is
`ShardedPool._open_read_only`, whose steps are numbered in its comment block.

| Write at open (read-write, where) | M1 / prompt | Under `read_only=True` | Implemented by | Pinned by |
|---|---|---|---|---|
| New-store branch: `mkdir`, `_write_shard_data`, shard files and their DDL (`__init__`, `Datastore.__init__`) | refuse | A missing primary is refused by `store_reader`, and nothing is made, not its directory. The new-store branch is never reached. An actor given a missing file refuses rather than create it | `_open_read_only` step 2 → `store_reader.open_read_only` (`shard_file_problem` on the primary); `Datastore._open_read_only` | `TestAtOpen.test_a_missing_primary_creates_nothing` |
| Hot-journal rollback of the primary: its first read-write connection, in `_read_shard_data` | refuse if a journal is beside the primary | Refused by presence, any size, `-journal`/`-wal`/`-shm`, before any engine is made; the pool's engine is `mode=ro` | `_open_read_only` step 2 → `store_reader._refuse_journals`; step 3, the `mode=ro` engine | `TestAtOpen.test_an_empty_journal_beside_the_primary_is_refused` |
| Hot-journal rollback of each shard: `_reconcile_replicated_tables` step 1, a read-write `sqlite3.connect` | refuse if a journal is beside any shard | Refused by presence, an empty one included, before any engine is made. Step 1 is not run: `_reconcile_replicated_tables` is not called | `_open_read_only` step 2 → `store_reader._refuse_journals`; `_check_replicated_tables_read_only` has no read-write connect | `TestAtOpen.test_an_empty_journal_beside_a_shard_is_refused`, `test_a_hot_journal_beside_a_shard_is_refused_and_not_rolled_back`, `test_every_connection_is_opened_read_only` |
| The comparison of replicated tables (`_reconcile_replicated_tables` step 3) | keep, `mode=ro` | Kept, on `mode=ro` engines; a difference refuses with `ReplicatedDivergence` | `_check_replicated_tables_read_only`, reusing `_read_replicated_tables`, `_replicated_differences`, `_describe_differences` | `TestAtOpen.test_diverged_shards_are_refused_and_nothing_repaired` |
| Repair of an interrupted replication, and clearing the record (steps 5–6) | refuse when the primary holds a record | `ReadOnlyWrite` naming the record, before anything is compared. Nothing is cleared | `_check_replicated_tables_read_only`, first statement after the record is read (`_read_in_flight_record`, on the `mode=ro` engine) | `TestAtOpen.test_a_record_of_any_operation_is_refused_by_name_and_not_cleared` (`get`) |
| **02:** `_complete_interrupted_prune`, which a `prune` record reaches in `_reconcile_replicated_tables` step 2 before any comparison | refuse, by name, before that branch | The same `ReadOnlyWrite`, naming `operation="prune"`. Neither `_reconcile_replicated_tables` nor `_complete_interrupted_prune` is called | `_check_replicated_tables_read_only` | the same test (`prune`), with both functions patched to fail if called |
| `replication_in_flight.create(checkfirst=True)` | skip | Skipped: the read-only open never reaches it. `_read_in_flight_record` reads a missing table as no record | `_open_read_only` (not called) | the SHA-256 checks of every `TestAtOpen` test |
| `_drop_actions` (`Datastore.__init__`) | refuse if passed (`ReadOnlyWrite`) | `ReadOnlyWrite` before anything is opened; the actor refuses it too | `_open_read_only` step 1; `Datastore._open_read_only` | `TestOtherWritesRaiseReadOnlyWrite.test_drop_actions_and_prune_refuse_before_anything_is_opened` |
| `_ensure_tables` (`Datastore.__init__`) | skip; report the tables a shard lacks | Skipped. Reported as `store_reader` reports it (`ReadOnlyShard.absent_tables`), as `pool.absent_tables` and a printed line; each actor reports its own. A lookup on the table fails in SQLite | `_open_read_only` step 2; `Datastore._open_read_only` | `TestAtOpen.test_a_shard_lacking_a_table_is_reported_and_the_store_opens` (`GkSource`, `OneLoopIntegral`), `test_a_lookup_on_a_missing_table_fails_in_sqlite` |
| `_validate_on_startup` with `prune_unvalidated=False` | keep; skip and report a validating class whose table the shard lacks (amended O1) | Report-only, `prune=False`, on a connection rolled back; a class whose table the file lacks is skipped and reported (`actor_states[sid]["skipped_validation"]`) | `Datastore._validate_on_startup_read_only` | `TestAtOpen.test_a_shard_lacking_a_table_is_reported_and_the_store_opens` (`GkSource` validates; `OneLoopIntegral` does not) |
| The prune, `prune_unvalidated=True` (**02:** `_prune_replicated_tables`, and the actors' sharded prune) | refuse (`ReadOnlyWrite`) at construction | `ReadOnlyWrite` before anything is opened; `_prune_replicated_tables` is never reached. The actor refuses it too | `_open_read_only` step 1; `Datastore._open_read_only` | `test_drop_actions_and_prune_refuse_before_anything_is_opened` |
| **02:** the version row: `_find_version_row`, then the replicated version write through `_get_impl_replicated_table` when the label is absent | find by label; refuse when absent | Found by `_find_version_row` (`mode=ro` `sqlite3`), before any actor exists. An absent label raises `ReadOnlyMiss("version", {"label": …})` naming the labels present. The write is never reached | `_open_read_only` step 5; `_version_labels_present` | `TestAtOpen.test_an_absent_version_label_is_refused_naming_the_labels_present` |
| **02:** `set_version` on every actor | the prompt's choice | Not called (choice 1 below) | `_open_read_only` | `TestNothingWrittenOnAFullStore` (the sequence needs no serial) |
| The broker, `read_largest_store_ids`, `notify_largest_store_ids` | skip (economy) | Skipped; actors get `serial_broker=None`. The pool waits for every constructor through `read_only_state` | `_open_read_only` step 6 | `TestOtherWritesRaiseReadOnlyWrite.test_drop_actions_and_prune_refuse_before_anything_is_opened` sees no actor call; every test opens without a broker |
| `SerialPoolManager` in each actor | keep (inert) | Kept, with `broker=None` | `Datastore.__init__` (unchanged) | — |
| `ShardedPool.__exit__`, `Datastore.__exit__` | keep | Kept, unchanged | — | every SHA-256 check is taken after `__exit__` |

Per lookup:

| What | Under `read_only=True` | Implemented by | Pinned by |
|---|---|---|---|
| A replicated `object_get` | One shard's actor, drawn as `read_table` draws it; no record, no replication; `_replicated_write` never entered | `_get_impl_replicated_table` → `_get_impl_replicated_table_read_only` | `TestNothingWrittenOnAFullStore` (both tests) |
| A get of the shard-key class | Its `_assign_shard_keys` is kept; where it would insert a key, `ReadOnlyWrite` | `_get_impl_replicated_table_read_only`; `_assign_shard_keys` | `TestOtherWritesRaiseReadOnlyWrite.test_a_new_shard_key` |
| An insert-on-miss of a replicated row | `ReadOnlyMiss(class, payload)`, before any `INSERT` and any serial lease | `Datastore._open_read_only` (every inserter replaced), `_inserters_for`, `_refuse_insert` | `TestEachMissRaisesReadOnlyMiss` (5 tests) |
| An insert of any other row (a vectorized get that reaches an inserter) | `ReadOnlyWrite` | `Datastore._refuse_insert` | `test_a_vectorized_get_that_reaches_an_inserter` |
| `object_store`, `object_validate` | `ReadOnlyWrite`, on the pool before any actor call, and on the actor | `ShardedPool.object_store`/`object_validate` → `_refuse_write`; `Datastore.object_store`/`object_validate` | `test_object_store_sharded_and_replicated`, `test_object_validate`, `test_the_actors_refuse_store_and_validate_themselves` |
| A write that reaches SQLite (`redshift`'s flag update on a hit) | The `mode=ro` file refuses it (`SQLITE_READONLY`); re-raised as `ReadOnlyWrite`, chained, naming the actor. `no such table` and `SQLITE_READONLY_ROLLBACK` are not translated | `Datastore._read_only_refusal`, from `object_get`'s and `object_read_batch`'s handlers and `_validate_on_startup_read_only` | `test_the_backstop_a_flag_update_on_a_hit` |

### The new exceptions' messages, as raised in the tests

Collected from the two test modules' `MESSAGES` records in one run of both (40 tests, all
passing). Temporary paths are as they were.

`ReadOnlyMiss`:
- `tolerance` (`TestEachMissRaisesReadOnlyMiss.test_tolerance`):
  `Datastore "shard0000-store" was opened read-only, and a lookup of "tolerance" matched no row; a read-write store would have inserted one with payload {'log10_tol': -7.0}. Nothing was written`
- `store_tag`: `Datastore "shard0001-store" was opened read-only, and a lookup of "store_tag" matched no row; a read-write store would have inserted one with payload {'label': 'Run_not-in-this-store'}. Nothing was written`
- `LambdaCDM`: `Datastore "shard0000-store" was opened read-only, and a lookup of "LambdaCDM" matched no row; a read-write store would have inserted one with payload {'name': 'Planck2018 68% central values TT+TE+EE+lowP+lensing+BAO', 'omega_m': 0.31110000000000004, 'omega_cc': 0.6889, 'h': 0.6766, 'f_baryon': 0.15817, 'T_CMB_Kelvin': 2.7255, 'Neff': 3.046}. Nothing was written`
- `QCD_Cosmology`: `Datastore "shard0000-store" was opened read-only, and a lookup of "QCD_Cosmology" matched no row; a read-write store would have inserted one with payload {'name': 'Planck2018 68% central values TT+TE+EE+lowP+lensing+BAO', 'omega_m': 0.31110000000000004, 'omega_cc': 0.6889, 'h': 0.6766, 'f_baryon': 0.15817, 'T_CMB_Kelvin': 2.7255, 'Neff': 3.046, 'log10_max_z': 20.0, 'T_z_representation': 6}. Nothing was written`
- `GkSourcePolicy`: `Datastore "shard0000-store" was opened read-only, and a lookup of "GkSourcePolicy" matched no row; a read-write store would have inserted one with payload {'label': 'policy="maximize-WKB"-Levin-threshold="5.0"', 'Levin_threshold': 5.0, 'numeric_policy': 'maximize-WKB'}. Nothing was written`
- the version label (`TestAtOpen.test_an_absent_version_label_is_refused_naming_the_labels_present`):
  `Datastore "/private/var/folders/tm/y9rd_z8x1ks3m09yhkjc0h7800128f/T/tmpoipk_43j/copy-1/store.sqlite" was opened read-only, and a lookup of "version" matched no row; a read-write store would have inserted one with payload {'label': '2025.1.1'} (the store holds the version label(s) ['standin']). Nothing was written`

`ReadOnlyWrite`:
- `object_store`: `Datastore "/private/var/folders/tm/y9rd_z8x1ks3m09yhkjc0h7800128f/T/tmpbjnu4srv/copy-1/store.sqlite" was opened read-only, and object_store of "GkSourcePolicyData" would write to it. Nothing was written`
- `object_validate`: `Datastore "/private/var/folders/tm/y9rd_z8x1ks3m09yhkjc0h7800128f/T/tmpkowtrr38/copy-1/store.sqlite" was opened read-only, and object_validate of "BackgroundModel" would write to it. Nothing was written`
- a new shard key: `Datastore "/private/var/folders/tm/y9rd_z8x1ks3m09yhkjc0h7800128f/T/tmpv9g_e04j/copy-1/store.sqlite" was opened read-only, and assigning a shard key of "wavenumber" would write to it (store_id(s) [102] have no row in the primary's shard_keys). Nothing was written`
- a vectorized get that reaches an inserter: `Datastore "shard0002-store" was opened read-only, and an insert of "GkNumericValue" would write to it (payload {'integration_serial': 1, 'z_serial': 1, 'G': 0.5, 'Gprime': 0.25, 'analytic_G_rad': None, 'analytic_Gprime_rad': None, 'analytic_G_w': None, 'analytic_Gprime_w': None, 'omega_WKB_sq': None, 'WKB_criterion': None}). Nothing was written`
- `drop_actions`: `Datastore "/private/var/folders/tm/y9rd_z8x1ks3m09yhkjc0h7800128f/T/tmpdx9apf6_/nowhere/store.sqlite" was opened read-only, and drop_actions=['gk-source-policy'] would write to it (a drop action drops tables). Nothing was written`
- `prune_unvalidated=True`: `Datastore "/private/var/folders/tm/y9rd_z8x1ks3m09yhkjc0h7800128f/T/tmpdx9apf6_/nowhere/store.sqlite" was opened read-only, and prune_unvalidated=True would write to it (the prune at open deletes unvalidated rows). Nothing was written`
- a `get` record: `Datastore "/private/var/folders/tm/y9rd_z8x1ks3m09yhkjc0h7800128f/T/tmpsbk64c6x/copy-1/store.sqlite" was opened read-only, and completing the write in flight (operation="get", class="tolerance", controlling shard=1, store_id=None, started=2026-10-03 09:00:00) would write to it (the primary holds this replication_in_flight record, so the shards may differ; the check at open of a read-write pool completes or repairs it. The record was not cleared). Nothing was written`
- a `prune` record: `Datastore "/private/var/folders/tm/y9rd_z8x1ks3m09yhkjc0h7800128f/T/tmpsbk64c6x/copy-2/store.sqlite" was opened read-only, and completing the write in flight (operation="prune", class="BackgroundModel", controlling shard=None, store_id=None, started=2026-10-03 09:00:00) would write to it (the primary holds this replication_in_flight record, so the shards may differ; the check at open of a read-write pool completes or repairs it. The record was not cleared). Nothing was written`
- the backstop: `Datastore "shard0000-store" was opened read-only, and object_get of "redshift" would write to it (SQLite refused the write to the read-only file "/private/var/folders/tm/y9rd_z8x1ks3m09yhkjc0h7800128f/T/tmp6_p017f9/copy-1/store-shard0000.sqlite": attempt to write a readonly database). Nothing was written`

`store_reader`'s refusals, reused unchanged (`RuntimeError`):
- missing primary: `Cannot read sharded datastore "/private/var/folders/tm/y9rd_z8x1ks3m09yhkjc0h7800128f/T/tmpc43vd174/nowhere/deeper/store.sqlite": the primary "/private/var/folders/tm/y9rd_z8x1ks3m09yhkjc0h7800128f/T/tmpc43vd174/nowhere/deeper/store.sqlite" does not exist. Nothing was read or repaired`
- empty journal beside a shard: `Cannot read sharded datastore "/private/var/folders/tm/y9rd_z8x1ks3m09yhkjc0h7800128f/T/tmpib_nz3qc/copy-1/store.sqlite": shard #1 "/private/var/folders/tm/y9rd_z8x1ks3m09yhkjc0h7800128f/T/tmpib_nz3qc/copy-1/store-shard0001.sqlite" has "/private/var/folders/tm/y9rd_z8x1ks3m09yhkjc0h7800128f/T/tmpib_nz3qc/copy-1/store-shard0001.sqlite-journal" beside it, so the store is open or was not closed cleanly. Nothing was read or repaired`
- hot journal beside a shard: the same text, for its own copy (`…/tmpec9z6r55/copy-1/…`).

`extract_common` (`ComputeTargets/tests/test_reader_read_only.py`):
- an unknown `--models` label (`RuntimeError`): `extract: --models names LambdaCMD, which config.model_list.build_model_list does not return. Models it returns: LambdaCDM, QCD_Cosmology.`
- no model read (`SystemExit`, status 1): `extract: no model was read. LambdaCDM was skipped: the datastore holds no background model for it; QCD_Cosmology was skipped: the datastore holds no background model for it.`
- the skip line (printed): `>> SKIPPING MODEL QCD_Cosmology: the datastore holds no background model for it`

### The prompt's choices

1. **`set_version` is not called.** A read-only actor inserts nothing: every inserter it holds
   refuses (`_refuse_insert`), and `object_store`/`object_validate` refuse before any inserter. The
   serial is read by nothing but `_insert` (the orchestrator's; audit V2). Leaving it unset keeps
   02's refusal in `_insert` as a second, independent stop: an insert that somehow got past the
   refusing inserters into `_insert` would raise before a serial is leased, and the `mode=ro` file
   is the third. The pool keeps the row it found as `self._version`.
2. **The broker, `read_largest_store_ids` and `notify_largest_store_ids` are skipped.** No serial
   is allocated on a read-only pool, so they are economy (M1). Each actor is built with
   `serial_broker=None`, which makes its `SerialPoolManager` inert (the orchestrator's:
   `ClientPool.py`). `read_largest_store_ids` was also where the read-write pool waited for every
   actor constructor; the read-only pool waits through a new actor method, `read_only_state`, which
   returns the tables the actor's file lacks and the classes it did not validate
   (`pool.actor_states`).
3. **`_find_version_row` runs before any actor exists**, right after the check at open (step 5).
   It needs only the shard files. So an absent label refuses before anything on the Ray side is
   made, and a store the check at open has refused is never asked for its version row.
4. **The test imports the probe's `build_full_store`, `qsi_sequence` and `other_sharded_lookups`**
   (with `Recorder`, `ReplicatedWriteLog`, `GKSOURCE_POLICY_5PT0` and `FACTS`), by `importlib`
   from `docs/a3-v2-readiness/reader_writes_probe.py`. The probe imports cleanly: its run is behind
   `__main__`. The prompt prefers the import where it is clean, and the reason is the comparison §3
   case 1 asks for: the sequence run here is the one R2 measured, not a second transcription that
   could drift from it. The cost is that a committed probe under `docs/` is now a test dependency.
   The probe is a record (CLAUDE.md rule 6) and is not edited; observation 2 says what follows.
   It is registered in `sys.modules` under its own name, as `readonly_open_probe.py` registers it,
   because its `Proxy` objects are pickled through the stand-in actors. Only the hot-journal child
   (10 lines) is transcribed, from `readonly_open_probe.py`.

### The tests

`Datastore/tests/test_read_only_pool.py`, 24 tests, on a full store built once per module by the
probe's `build_full_store` and copied per test. The instrument is `standin_pool.store_checksums`,
every file's SHA-256 in the store's directory, so a file that appears or goes is seen too. Beside
it, a class-level `before_cursor_execute` listener on `sqlalchemy.engine.Engine` records every
statement any engine issues:
- `TestNothingWrittenOnAFullStore` (2):
  - `test_the_instrument_counts_a_read_write_pool_then_read_only_writes_nothing`: on a read-write
    copy, QSI's sequence and the other readers' sharded lookups change `store.sqlite` and nothing
    else, with 21 `_replicated_write` calls (R2 Run 1). Then on a read-only copy: every file's
    SHA-256 unchanged open to close; no `_replicated_write` call; no INSERT, UPDATE, DELETE or DDL
    statement from any engine; every step's file delta empty. Each step's outcome equals the
    read-write run's. The hits R2 Run 1 recorded are asserted: `available=True` with the store's
    background serial, `3 source, 3 response`, `6 of 6 available`, `12 source, 6 response`, the
    two `GkSourcePolicy` serials the store holds, the stored `GkSourcePolicyData` available, and the
    sequence stopped by the script's own `RuntimeError` at `QCD_Cosmology`'s background (its type
    exactly `RuntimeError`, so not a `ReadOnlyMiss`).
  - `test_a_replicated_get_is_one_call_to_the_drawn_shard`: with the stand-in's draw pinned to
    each shard in turn, a `tolerance` get is exactly one `object_get` on that shard, with no
    `insert_timestamp`.
- `TestEachMissRaisesReadOnlyMiss` (5): `store_tag`, `LambdaCDM`, `QCD_Cosmology`, `tolerance`
  (`DEFAULT_QUADRATURE_RTOL`'s row) and `GkSourcePolicy` (the 5.0 policy), each deleted from every
  shard of a closed copy, as R2's Runs 2–9 did. Each raises `ReadOnlyMiss` naming the class, with
  the payload checked, no write statement from any engine, and no file changed. `store_tag`'s
  lookup is made directly: with the run's tag deleted, `resolve_run_selection` makes no lookup
  (R2 Run 2).
- `TestOtherWritesRaiseReadOnlyWrite` (7): `object_store` of a sharded and of a replicated object,
  with no actor call made; `object_validate`; the actors' own refusals; a new shard key (a
  wavenumber on every shard whose `shard_keys` row was deleted while closed); a vectorized get of
  `GkNumericValue` that reaches its inserter; `drop_actions` and `prune_unvalidated=True`, each on
  a path where no store exists (the refusal is `ReadOnlyWrite`, not the missing primary's, so it
  came first, and the directory is not made) and on an existing store; and the backstop: `redshift`
  `z = 1e5`, a source redshift only, asked for as a response redshift too. The listener sees the
  one `UPDATE` reach SQLite; the pool raises `ReadOnlyWrite` naming the shard's actor; called on
  the actor object directly, its `__cause__` is `sqlalchemy.exc.OperationalError` with
  `sqlite_errorcode == SQLITE_READONLY`. No file changed.
- `TestAtOpen` (10): the missing primary (then, as the instrument, a read-write pool on the same
  path creates it); an empty `-journal`, `-wal` and `-shm` beside the primary; an empty journal
  beside a shard; a hot journal beside a shard (made by the transcribed child; as the instrument,
  a read-write `sqlite3.connect` of a copy rolls it back, changing the shard's SHA-256 and removing
  the journal; the read-only open refuses, and the shard's and journal's SHA-256 are unchanged); a
  `get` record and a `prune` record, with `_reconcile_replicated_tables` and
  `_complete_interrupted_prune` patched to fail if called, the record still on the primary
  afterwards and no actor built; an absent version label; diverged shards; `GkSource` (validates)
  and `OneLoopIntegral` (does not) each dropped from shard 1 only: the store opens, the table is
  in `pool.absent_tables[1]`, and `GkSource` is in shard 1's `skipped_validation`; `GkSource`
  dropped from every shard: the sequence's lookup fails with `no such table: GkSource`; and every
  connection, below.
  - `test_every_connection_is_opened_read_only` observes the connections themselves.
    `Datastore.SQL.ShardedPool.sqlite3.connect` is patched with a recorder, and a class-level
    `engine_connect` listener records every SQLAlchemy connection's URL, through the open and QSI's
    sequence. Every `sqlite3.connect` call is a `file:…?mode=ro` URI with `uri=True`. Every engine
    URL carries `?mode=ro&uri=true`, and the files opened are exactly the primary and the three
    shards.

`ComputeTargets/tests/test_reader_read_only.py`, 17 tests:
- `TestTheSixReaders` (5, each with a sub-test per script, read with `ast`): one `ShardedPool(...)`
  call, the `with` block's, with `read_only=True` and neither `prune_unvalidated` nor
  `drop_actions`; `add_models_argument(parser)` at module level before `parse_args`;
  `model_list = select_models(build_model_list(...), args.models)` and
  `read_models(model_list, run_pipeline)` in the `with` block, and no `for` loop over `model_list`
  left; `require_background(model, model_label)` in `run_pipeline`, and no "Could not locate
  suitable background model" anywhere in the script; the four helpers imported from
  `extract_common`.
- `TestModelsArgument` (2), `TestSelectModels` (3: `None` keeps all; the filter keeps
  `build_model_list`'s order; an unknown label, alone or beside a known one, is an error naming the
  known ones).
- `TestTheSkip` (7): a model with no background is skipped with its one line and the next model is
  read; the order does not matter; every model held prints nothing; all skipped raises
  `SystemExit` naming each model and the reason; an empty list exits non-zero; any other error is
  not a skip; `NoBackgroundModel` is not a `RuntimeError`.

Every test runs with no Ray; each module's `tearDownModule` asserts `ray.is_initialized()` is
false. Nothing under `var/` is opened.

## Deviations from the prompt

1. **The read-only check at open is a function of its own, `_check_replicated_tables_read_only`,
   not a branch inside `_reconcile_replicated_tables`.** The read-only open never calls
   `_reconcile_replicated_tables`, so the refusal of a record of any operation is necessarily
   before its `prune` branch, and that function's diff is empty. It reuses
   `_read_in_flight_record`, `_replicated_table_specs`, `_read_replicated_tables`,
   `_replicated_differences` and `_describe_differences` unchanged. **IMPLEMENTATION CHOICE.**
2. **The refusal of a record is `ReadOnlyWrite`.** The prompt says it refuses, naming the record,
   and does not name the exception. Completing or repairing the record is the write refused, and
   README §4 lists "repair" among `ReadOnlyWrite`'s. **IMPLEMENTATION CHOICE.**
3. **`_refuse_insert` raises `ReadOnlyMiss` for a row of a replicated class and `ReadOnlyWrite` for
   any other row.** O2 says every `build` is handed an inserter that raises `ReadOnlyMiss`, and also
   that any `object_get_vectorized` that reaches an inserter raises `ReadOnlyWrite`.
   `object_get_vectorized` is sharded-only, and a sharded `build` reaches an inserter only to write
   value rows of computed data (audit R1). Deciding by the inserted row's class satisfies both
   sentences: the five insert-on-miss classes are replicated. **IMPLEMENTATION CHOICE.**
4. **The pool's engine is made by calling `_create_engine`, disposing the engine it makes, and
   replacing it with a `mode=ro` one.** The pool's table objects are built inside
   `_create_engine`, which the read-write path uses and this prompt may not change.
   `create_engine` opens no connection, so the discarded engine was never connected;
   `test_every_connection_is_opened_read_only` sees no connection that is not `mode=ro`.
   **STRUCTURALLY REQUIRED.**
5. **Two helpers in each script, not one.** The prompt asks for one skip helper called from the
   model loop. The background is looked up inside `run_pipeline`, so only there is its absence
   known. Seeing it from the loop would need a second lookup, and O3 forbids rewriting a lookup.
   So `require_background` replaces the `RuntimeError` at its site and raises `NoBackgroundModel`,
   and `read_models`, which replaces the loop, catches it, prints the line and goes on. The skip
   itself (the line, the next model, the exit) is in `read_models` alone. **STRUCTURALLY
   REQUIRED.**
6. **`read_models` exits by raising `SystemExit` with the message** (status 1), rather than
   returning for each script to call `sys.exit`. Each script's body then changes only at the
   helper calls. Inside the `with ShardedPool(...)` block it leaves through `__exit__`.
   **IMPLEMENTATION CHOICE.**
7. **The actor also refuses `drop_actions`, `prune_unvalidated=True` and a missing file**, beside
   the pool. The pool never passes the first two, and the third is refused by `store_reader` first.
   This is defence in depth for an actor built directly. **IMPLEMENTATION CHOICE.**
8. **New actor method `read_only_state`, and pool attributes `absent_tables` and
   `actor_states`** (choice 2). **IMPLEMENTATION CHOICE.**
9. **The backstop is applied in `object_get`, `object_read_batch` and the read-only startup
   validation, not in `read_table`.** `read_table` has no `SQLAlchemyError` handler, and adding one
   would change its read-write body. No factory's `read_table` writes (audit R1). A write there would
   still be refused by the `mode=ro` file, as SQLAlchemy's `OperationalError`. **IMPLEMENTATION
   CHOICE.**
10. **`ReadOnlyMiss` has an optional `detail`.** The version refusal names the labels present in
    it. **IMPLEMENTATION CHOICE.**
11. **The missing-table lookup test drops `GkSource` from every shard.** Then the sequence's lookup
    is on a shard that lacks it, whichever shard the store's keys name. The single-shard case is
    tested for open, report and skip. **IMPLEMENTATION CHOICE.**
12. **Breakage (iv) mutates `Datastore/store_reader.py`.** The empty-journal refusal is
    `store_reader`'s, reused as the prompt requires. The mutation is never committed.
    **STRUCTURALLY REQUIRED.**
13. **The read-write path evaluates the read-only flag at eleven places**: both constructors'
    dispatch (`self._read_only`, set from the new parameter); and `getattr(self, "_read_only",
    False)` in `_get_impl_replicated_table`, `object_store`, `object_validate` and
    `_assign_shard_keys` on the pool, and in `object_store`, `object_validate`, `_inserters_for`
    and, through `_read_only_refusal`, the two `except` handlers on the actor. Each guards new code
    only. On a read-write pool every
    guard is false and execution continues into the unchanged lines below. No existing line of
    either pool file was changed or deleted (numstat `302 0`, `221 0`). The Datastore suite's 574
    existing tests pass unchanged. `getattr` rather than the attribute, because some tests build a
    pool with `object.__new__` (`shard_store_fixtures`, `test_inventory_consumers`). Stated for
    the review. **STRUCTURALLY REQUIRED.**

No `UNINTENDED DRIFT`.

## Verification performed

### Suites

All six, serially, with README §7's command, each suite's output to a file in the scratch
directory and its verdict grepped from it, at the tree of this commit:

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | **686 OK** |
| CosmologyModels | 39 OK |
| Datastore | **598 OK** |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

Measured 10:14–10:24 on 2026-10-03, at load averages (1-minute) of 8–29 on 10 CPUs, on a
machine shared with an unrelated Ray cluster (another of the user's projects), left alone.
ComputeTargets is 669 + this prompt's 17 (`test_reader_read_only`); it includes
`test_main_plumbing` and `test_quadrature_tolerance_pin`, unchanged and passing. Datastore is 574 +
this prompt's 24 (`test_read_only_pool`). Every other count is unchanged from the brief's baselines.
No test failed, so no wall-clock flake needed re-running.

### Read-write unchanged

- `git diff --cached --numstat` shows no deleted line in `ShardedPool.py`, `Datastore.py` or
  `replication.py`.
- An `ast.unparse` comparison of every function against `e4d6ccd` finds these changed:
  `__init__`, `_assign_shard_keys`, `_get_impl_replicated_table`, `object_store` and
  `object_validate` on the pool; `__init__`, `_inserters_for`, `object_get`, `object_read_batch`,
  `object_store` and `object_validate` on the actor. Each changed only by the guards of deviation
  13, and the two constructors also by the new parameter and their docstrings. Unchanged:
  `_replicated_write`, `_plan_replicated_repair`, `_prune_replicated_tables`,
  `_complete_interrupted_prune`, `_find_version_row` and `_reconcile_replicated_tables`.

### Formatting and staging

- `black` on the twelve touched Python files, then `black --check` on them: clean.
- Staged by explicit path. `git diff --cached --name-only` was shown before the commit.

## The deliberate-breakage record

Each diff was made in the working tree against the staged tree of this commit, recorded with
`git diff -- <file>`, run against the named module, and restored with `git checkout -- <file>`
from the index, after which `git diff --name-only -- '*.py'` was empty. The harness ran in `bash`.
The record below is its second run, on the final staged tree (24 tests in `test_read_only_pool`).

**(i) A read-only replicated `object_get` goes through `_replicated_write` again.** The record's
insert then meets the `mode=ro` primary. Fails
`TestNothingWrittenOnAFullStore.test_the_instrument_counts_a_read_write_pool_then_read_only_writes_nothing`
(error) and `test_a_replicated_get_is_one_call_to_the_drawn_shard` (3 sub-tests). It also fails
every `TestEachMissRaisesReadOnlyMiss` test, five of `TestOtherWritesRaiseReadOnlyWrite`, and
`TestAtOpen.test_every_connection_is_opened_read_only` and
`test_a_lookup_on_a_missing_table_fails_in_sqlite`: 1 failure and 16 errors (the drawn-shard test
errs in each of its 3 sub-tests) in a module of 24.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 5b6ae40..c49231a 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -3026,9 +3026,6 @@ class ShardedPool:
 
     def _get_impl_replicated_table(self, cls_name, kwargs):
         # a read-only pool answers from one shard, with no record and no replication (prompt 03)
-        if getattr(self, "_read_only", False):
-            return self._get_impl_replicated_table_read_only(cls_name, kwargs)
-
         # we push an initial 'get' to the controlling shard. If a new database object was created
         # (or an existing one updated) by the get, we then push a replica to all the other shards,
         # with the controller's store_id, and require each to answer with that store_id.
```

**(ii) The read-only inserter inserts instead of raising.** Fails every
`TestEachMissRaisesReadOnlyMiss` test. The insert reaches `_insert`, which for `tolerance` sends the
`INSERT` to SQLite, refused as `ReadOnlyWrite`; for the versioned classes `_insert` refuses first,
since `set_version` was not called. It also fails
`TestOtherWritesRaiseReadOnlyWrite.test_a_vectorized_get_that_reaches_an_inserter`: 6 errors.

```diff
diff --git a/Datastore/SQL/Datastore.py b/Datastore/SQL/Datastore.py
index 4cb3c24..05ac793 100644
--- a/Datastore/SQL/Datastore.py
+++ b/Datastore/SQL/Datastore.py
@@ -426,6 +426,9 @@ class Datastore:
         ``ReadOnlyMiss``. Any other row (a sharded value row that a vectorized get builds, a tag
         row) is a write of computed data, so it is a ``ReadOnlyWrite``.
         """
+        return self._insert(
+            schema, schema["table"], conn, payload, insert_timestamp=insert_timestamp
+        )
         cls_name = schema["name"]
         if cls_name in self._replicated_tables:
             raise ReadOnlyMiss(cls_name, dict(payload), store=self._my_name)
```

**(iii) The actor's engine is opened read-write under `read_only=True`.** Fails
`TestOtherWritesRaiseReadOnlyWrite.test_the_backstop_a_flag_update_on_a_hit`: the `UPDATE` succeeds,
so no `ReadOnlyWrite` is raised. It also fails `TestAtOpen.test_every_connection_is_opened_read_only`:
a shard engine's URL lacks `mode=ro`. 2 failures.

```diff
diff --git a/Datastore/SQL/Datastore.py b/Datastore/SQL/Datastore.py
index 4cb3c24..3d9f761 100644
--- a/Datastore/SQL/Datastore.py
+++ b/Datastore/SQL/Datastore.py
@@ -328,7 +328,7 @@ class Datastore:
         if self._timeout is not None:
             connect_args["timeout"] = self._timeout
         self._engine = sqla.create_engine(
-            read_only_url(self._db_file),
+            f"sqlite:///{self._db_file}",
             future=True,
             connect_args=connect_args,
         )
```

**(iv) The empty-journal refusal is removed** (in `store_reader._refuse_journals`, deviation 12: a
journal of size 0 is let through). Fails `TestAtOpen.test_an_empty_journal_beside_a_shard_is_refused` and
`test_an_empty_journal_beside_the_primary_is_refused` (sub-tests `-journal`, `-wal`, `-shm`): 4
failures. The hot-journal test still passes, as it should: its journal is not empty.

```diff
diff --git a/Datastore/store_reader.py b/Datastore/store_reader.py
index 61bfc09..ff37ef0 100644
--- a/Datastore/store_reader.py
+++ b/Datastore/store_reader.py
@@ -93,7 +93,7 @@ def _refuse(primary: Path, reason: str) -> RuntimeError:
 
 def _refuse_journals(primary: Path, path: Path, what: str) -> None:
     for journal in ShardedPool._journal_paths(path):
-        if os.path.lexists(journal):
+        if os.path.lexists(journal) and os.path.getsize(journal) > 0:
             raise _refuse(
                 primary,
                 f'{what} "{str(path)}" has "{str(journal)}" beside it, so the store is open or was not closed cleanly',
```

**(v) The check at open's step-1 read-write `sqlite3.connect` runs under `read_only=True`.** No
journal is there, so no file changes, and every checksum test passes. Fails
`TestAtOpen.test_every_connection_is_opened_read_only`, which records the three
`sqlite3.connect(str(path))` calls without `mode=ro`. 1 failure.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 5b6ae40..3729ab7 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -536,6 +536,13 @@ class ShardedPool:
                 "record was not cleared",
             )
 
+        for sid in shard_ids:
+            conn = sqlite3.connect(str(Path(self._shard_db_files[sid])))
+            try:
+                conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
+            finally:
+                conn.close()
+
         built = build_schema(sqla.MetaData(), _factories)
         specs = self._replicated_table_specs(built)
         reconciliation["compared"] = [spec["name"] for spec in specs]
```

**(vi) One script keeps `prune_unvalidated=False` and drops `read_only=True`** (`extract_Gk_data.py`).
Fails `TestTheSixReaders.test_each_opens_its_pool_read_only` (sub-test `extract_Gk_data.py`).

```diff
diff --git a/extract_Gk_data.py b/extract_Gk_data.py
index 3ef454e..e969200 100644
--- a/extract_Gk_data.py
+++ b/extract_Gk_data.py
@@ -459,7 +459,7 @@ with ShardedPool(
     timeout=args.db_timeout,
     profile_agent=profile_agent,
     job_name="extract_Gk_data",
-    read_only=True,
+    prune_unvalidated=False,
     read_table_config=read_table_config,
 ) as pool:
 
```

**(vii) The skip helper raises again for a model with no background.** Fails
`TestTheSkip.test_a_model_with_no_background_is_skipped_with_its_line`,
`test_the_skip_does_not_stop_the_next_model` and
`test_all_skipped_exits_non_zero_with_the_reason`: 3 errors.

```diff
diff --git a/extract_common.py b/extract_common.py
index 1f248b0..dc36b61 100644
--- a/extract_common.py
+++ b/extract_common.py
@@ -323,6 +323,9 @@ def read_models(model_list: Sequence[dict], run_pipeline) -> List[str]:
         try:
             run_pipeline(model_data)
         except NoBackgroundModel as e:
+            raise RuntimeError(
+                "Could not locate suitable background model instance in the datastore"
+            ) from e
             if e.model_label != label:
                 raise
             print(f"\n>> SKIPPING MODEL {label}: {NO_BACKGROUND_REASON}")
```

### Replay

Before the commit, each of the seven files was checked with `git apply --check --cached <file>`
against the staged tree, which is this commit's tree: all seven apply. (iv)'s file,
`Datastore/store_reader.py`, is not in this commit, so it applies to `e4d6ccd`'s copy, which is the
committed one.

## Observations not acted on

1. **The six scripts' `--database` help still says "read/write work items using the specified
   database cache".** They now open the store read-only. The help string is in each script's
   body, which this prompt changes only at the helper calls. It is residue for a follow-up, not a
   defect in behaviour.
2. **`docs/a3-v2-readiness/reader_writes_probe.py` is now imported by a test** (choice 4). The
   probe is a record of `5707851` and is not edited (CLAUDE.md rule 6). If a later prompt has to
   change what it transcribes, the test should then take a copy of the sequence instead of
   editing the probe.
3. **`ReadOnlyMiss` and `ReadOnlyWrite` are `RuntimeError`s**, as the module's other exceptions
   are. A caller's `except RuntimeError` catches them: the probe's own `run_sequence` would read a
   `ReadOnlyMiss` as "the sequence stopped where the script would exit 1". The test's miss cases
   call `qsi_sequence` directly for that reason. None of the six scripts catches `RuntimeError`.
4. **Journals are refused at open, not on every read.** A writer that opens the store after a
   read-only pool has opened it leaves a journal the pool does not re-check. A `mode=ro` read
   beside a hot journal then raises `SQLITE_READONLY_ROLLBACK` (audit M2d), which
   `_read_only_refusal` leaves as SQLAlchemy's `OperationalError`, by design: it is not a write
   this pool attempted.
5. **A read-only pool cannot open a store a build has open**, since an open store has a journal
   beside it. That is `store_reader`'s rule, which the prompt asks for. 07's L3 reads its store
   after the build has exited.
6. **`extract_common.available_run_labels`' docstring** says "each actor's opening writes have
   committed, and nothing writes until the script's first `object_get`". On a read-only pool
   nothing writes at all. It is still true, and is left.
7. **The audit's probes still describe `5707851`.** Run here as instruments only, through the
   test's imports. Their committed outputs are not re-taken (prompt §6; no committed probe output).

No new issue is opened. The index count is 102: 104 less the two rows this prompt closes.

## State handed to the next prompt

- **For 07 (L3).** Every reader opens `read_only=True` and takes `--models`. Against a
  LambdaCDM-only store, a reader with no `--models` reads LambdaCDM, prints
  `>> SKIPPING MODEL QCD_Cosmology: the datastore holds no background model for it`, and exits 0.
  With `--models LambdaCDM` it reads LambdaCDM alone. A reader given a path with no store, a store
  with a journal, a record on its primary, or a version label other than `2025.1.1` refuses at
  open and writes nothing.
- **Names.** `ShardedPool(read_only=True)`; `ReadOnlyMiss`, `ReadOnlyWrite` in
  `Datastore/replication.py`, importable from `Datastore.SQL.ShardedPool`;
  `extract_common.add_models_argument`, `select_models`, `require_background`, `read_models`.
- **Pool attributes on a read-only pool:** `absent_tables` (shard → tables its file lacks),
  `actor_states` (shard → `read_only_state()`), `reconciliation` (with `read_only: True`).
- **Baselines** after this commit: §5.6 of the board. ComputeTargets 686, Datastore 598.
- **Index:** 102 open.
