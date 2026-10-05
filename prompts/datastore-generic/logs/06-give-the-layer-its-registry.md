# Log — prompt 06: give the layer its registry

**Subject:** *Give the datastore layer its registries instead of importing them* · **Model:** Claude
Opus 5.5 · **Date:** 2026-10-06 · **Base:** `154d868`, clean · **Result:** landed.
- **G1:** `ShardedPool` takes `factories` (required, keyword-only) and `serial_batch_sizes`, and
  gives both to every actor; the actor takes the same and registers `factories`. `Datastore.py`
  imports no factory but `base` and holds none of the three registries; `ClientPool` holds no table
  of batch sizes.
- **G2:** `open_read_only(primary, factories)` and `read_inventory(primary, factories)`. The reader
  reads the primary's `replicated_tables` record (`ReadOnlyStore.replicated_tables`), and the
  inventory compares exactly the classes it names. Neither imports `_factories` or
  `config.sharding`.
- **G3:** `config/datastore.py` holds `factories`, `drop_groups` (keys in `main.py`'s old choice
  order), `serial_batch_sizes` and `tables_to_drop`. The layer is given `drop_tables`; the actor
  drops them in the order `schema.drop_order` derives from foreign keys. `_drop_order` is deleted.

No stop condition fired. Closes `[07-drop-actions-list-a-parent-table-before-its-children]`
(`datastore-integrity`). Narrows `[00-the-generic-datastore-layer-hard-codes-project-schema-facts]`
(audit C1 rows 9, 10, 11, 14). Opens nothing. Index 108 → 107. The commit is `19f07ee`.

## What shipped

**`config/datastore.py` (new).**
- `factories`: `Datastore.py`'s `_factories`, with its factory imports. Same keys, same order,
  same classes and the same comment (`ast.dump` of the dict and of the import statements equal to
  `HEAD`'s).
- `drop_groups`: `_drop_actions`' groups with the same tables in the same lists. The keys are in
  the order of `main.py`'s old `allowed_drop_actions` literal (checked by `ast` against `HEAD`).
- `serial_batch_sizes`: `ClientPool._default_serial_batch_size`, unchanged, comment included.
- `tables_to_drop(actions) -> List[str]`: each table once, in the order the actions and then each
  group's list give; an unknown action raises `ValueError` naming it and listing the groups.
- The module docstring says what each is, that the groups are a client fact because they are the
  command line's vocabulary (`main.py` offers them as `--drop`'s choices in this order and gives
  the pool only the tables), and that the order of the groups and of the tables within one decides
  nothing about a drop's order, which is `Datastore.SQL.schema.drop_order`'s.
- `import config.datastore` first in a fresh interpreter works (it is what the import test's
  stand-ins and `RunRegistry`'s fingerprint do).

**`Datastore/SQL/schema.py`.** `drop_order(tables) -> List[sqla.Table]`: the given tables (each
once, by identity) in an order in which each can be dropped while the others not yet dropped are
present. A foreign key into another given table puts the referencing table first; a foreign key
into a table not given, or into the table itself, does not constrain. Among the tables free to go
next, the one given first goes first. A cycle raises `ValueError` naming the tables on or behind
it. The module docstring names it; the module still imports only `sqlalchemy` (and `typing`) and
names no project table.

**`Datastore/SQL/Datastore.py` (the actor).**
- The factory imports (all but `ObjectFactories.base`), `_factories`, `_drop_actions` and
  `_drop_order` are deleted.
- The constructor takes `factories` (required) and `serial_batch_sizes` (optional), keyword-only,
  after `read_only`; it registers `factories` and passes `serial_batch_sizes` to
  `SerialPoolManager(batch_sizes=…)`. `drop_actions` is `drop_tables`. The docstring says what each
  of the three is.
- `_open_read_only(drop_tables)`: the refusal reads `drop_tables=[…]`, detail "dropping a table is
  a write".
- `_drop_tables(tables)` replaces `_drop_actions(actions)`: a name the actor's `_tables` does not
  hold raises `RuntimeError` naming it before anything is dropped; the declared tables are dropped
  in `drop_order`'s order; a table the file lacks is skipped (`has_table`); one line per table
  dropped, `Datastore: dropping table "<name>"`; the inspector is regenerated after. A bare string
  is refused as before ("Could not interpret drop_tables type").
- By `ast` against `HEAD` (docstrings stripped), only the module's statements, `__init__`,
  `_open_read_only` and the drop method changed.

**`Datastore/SQL/ClientPool.py`.** `_default_serial_batch_size` is deleted. `SerialPoolManager`
takes `batch_sizes: Optional[Mapping[str, int]] = None` (kept as a copy); a table it does not name
leases 500 at a time, `ClientPool`'s own default.

**`Datastore/SQL/ShardedPool.py`.**
- The `_factories` import is gone. The constructor takes `factories` (required) and
  `serial_batch_sizes` keyword-only, after `read_only`, and keeps a copy of each and of
  `drop_tables`. The docstring says what each of the three is.
- Every `_factories` is `self._factories` (in `_check_replicated_tables_read_only`,
  `_reconcile_replicated_tables`, `_recompute_validated` (still `["BackgroundModel"]`, C1 row 6),
  `_prune_needed`, `_prune_shard`, `_prune_replicated_tables`).
- Both actor constructions pass `factories=` and `serial_batch_sizes=`; the read-write one passes
  `drop_tables=`, the read-only one `drop_tables=None`.
- `factories`, a read-only property returning a copy.
- **An undeclared drop table.** Read-only: step 1 of `_open_read_only` is unchanged in place, and
  refuses any non-empty `drop_tables` with `ReadOnlyWrite` first. Read-write: right after the
  read-only branch returns, `_refuse_undeclared_drop_tables()` refuses a name that
  `build_schema(…, self._factories).tables` does not hold, before the primary is created or
  opened.
- `_open_read_only` calls `open_read_only(self._primary_file, self._factories)`. The comment block
  "THE READ-ONLY POOL" says at step 1 `drop_tables`, and at step 2 that the reader reads the
  primary's `replicated_tables` record, so a primary whose record cannot be read, or names an
  undeclared class, meets the reader's `RuntimeError` there, before step 3's check of the primary.
- By `ast`, only `__init__`, `_open_read_only`, the six methods that used `_factories` and the
  module's imports changed; `factories` and `_refuse_undeclared_drop_tables` are new.

**`Datastore/store_reader.py`.**
- The `_factories` import is gone. `open_read_only(primary, factories)` builds the schema from
  `factories`.
- `_read_replicated_tables(primary, factories)` (new) reads `SELECT serial, "table" FROM
  replicated_tables ORDER BY serial` on a `sqlite3` `mode=ro` connection (as
  `ShardedPool._read_closed_store` reads `shards`), after the shard and journal checks and before
  `build_schema`. A `sqlite3.Error` is refused through `_refuse` naming the table and the error; a
  recorded class `factories` does not declare is refused the same way. An empty record is
  returned as it is.
- `ReadOnlyStore.replicated_tables: Tuple[str, ...]`, the record in serial order.
- The module docstring has a paragraph "The replicated set is the store's", and no longer says the
  module imports `ray` through the factory map; `open_read_only`'s refusal list names both new
  refusals.

**`Datastore/store_inventory.py`.** `read_inventory(primary, factories)`. It imports neither
`_factories` nor `config.sharding`; the replicated set is `set(store.replicated_tables)` and
`factories[name]` is each class's factory. The module docstring says which classes are replicated
is what the primary records in `replicated_tables`, and that the registry is given.

**The callers** (plumbing; behaviour measured unchanged below).
- `main.py`: imports `drop_groups`, `factories`, `serial_batch_sizes` and `tables_to_drop` from
  `config.datastore`; `allowed_drop_actions = list(drop_groups)`; `drop_tables =
  tables_to_drop(drop_actions)` after `drop_actions`; the `ShardedPool(` call passes
  `drop_tables=`, `factories=` and `serial_batch_sizes=`; `--inventory` calls
  `read_inventory(args.database, factories)`.
- The six `extract_*.py` scripts: `from config.datastore import factories`, and
  `factories=factories` in each `ShardedPool(` call. None passes batch sizes or drop tables.
- `extract_common.available_run_labels`: `read_inventory(pool.primary, pool.factories)`; the
  docstring says so (the paragraph is reflowed).
- `RunRegistry/stores.py`'s `fingerprint_store`: imports `factories` from `config.datastore`
  inside the function beside `read_inventory`, passes it; docstring steps 2 and 3 say so.
- `tools/inventory_report.py`'s `_schema_tables`: builds from `config.datastore.factories`.

## The tests, test by test

Grouped as the prompt's §2 groups them. Where a module's change is an import line only, its
location is `<module>`. Counted with an `ast` map of every changed line to its enclosing
definition.

### Test helpers
- `Datastore/tests/standin_pool.py`: `StandinCluster.open_pool` and `open_pool_output` import
  `factories` from `config.datastore` and `kwargs.setdefault("factories", factories)`.
  (`docs/a3-v2-readiness/reader_writes_probe.py`, unedited, reaches the pool through it.)
- `Datastore/tests/schema_description.py`: `actor_with_built_schema` registers
  `config.datastore.factories` (function-scope import) where it read `datastore_module._factories`.
- `Datastore/tests/real_store_fixtures.py`: `from config.datastore import factories as _factories`.

### Import only (`from config.datastore import factories as _factories`)
- `Datastore/tests/`: `test_exact_identity_lookups`, `test_factory_select_columns`,
  `test_integral_read_batch`, `test_inventory_retired`, `test_prune_at_open`,
  `test_read_path_mechanics`, `test_schema_builder`, `test_value_reads_parent_key`.
- `ComputeTargets/tests/`: `test_gksource_parent_set_main`, `test_quadsource_integral_parent_main`.

### The registry passed to `open_read_only` or `read_inventory`, with its import
| Module | Where |
|---|---|
| `Datastore/tests/test_store_schema` | `TestTheReader.test_a_refusal_yields_nothing_and_disposes_every_engine`, `TestTheInventory.test_read_inventory_refuses_a_store_missing_a_table` (its `_factories` import moved to `config.datastore`) |
| `Datastore/tests/test_store_inventory` | `_Stores.inventory`, `TestValueCounts.test_value_tables_are_only_counted`, `TestReadOnlyAndNoRay.test_read_inventory_writes_nothing`; the child-interpreter string of `TestReadOnlyAndNoRay.test_no_ray`; and the function-scope import in `store_inventory_tables` (deviation 9) |
| `Datastore/tests/test_store_reader` | `TestReaderReads` (all five methods), `TestReaderNeverWrites` (both), `TestADifferingSchemaIsRefused.assert_refused`, `TestRefusals.assert_refused_naming`, `TestNoRay.test_reading_in_a_child_interpreter_leaves_ray_uninitialised`: ten calls; and the `_CHILD` string |
| `Datastore/tests/test_absolute_shard_record_refused` | `TestEveryReader.test_the_read_only_reader` |
| `Datastore/tests/test_gksource_parent_set` | `TestTheInventoryResolvesTheSet.read` |
| `Datastore/tests/test_inventory_report` | `_Stores.setUpClass`, `_Stores.read` |
| `Datastore/tests/test_inventory_report_labels` | `_Fixture.read` |
| `Datastore/tests/test_quadsource_integral_parent_key` | `read_the_store` |
| `Datastore/tests/test_quadsource_parent_key` | `read_the_store` |
| `Datastore/tests/test_quadsource_policy_key` | `TestTheInventoryIdentityIncludesIt.read_the_store` |
| `RunRegistry/tests/test_store_fingerprint` | `_FullStores.setUpClass`, `_FullStores.fingerprint`, `TestLocalised.test_deleting_one_quadsourceintegral_row`, `…test_adding_a_tag_to_one_record`, `…test_deleting_one_value_row`, `…test_flipping_one_validated_flag`, `TestNotContent.test_timestamps_labels_and_solver_serials`, `_FullStoreCase.expected`: eight |
| `RunRegistry/tests/test_store_retire` | `TestWithoutFingerprint.test_a_corrupt_shard` |
| `ComputeTargets/tests/test_qcd_cosmology_inventory_record` | `TestTheRepresentationInTheInventory.inventory_with_second_qcd_row`, `…test_the_representation_sits_beside_log10_max_z` |

### The registry passed to a pool or an actor
- `test_read_only_pool._ReadOnlyTestCase.open`: `factories=factories`.
- `test_version_row_at_open._VersionTestCase.open` and `TestInsertBeforeSetVersion.bare_actor`
  (the direct `Datastore.options(…).remote(…)`): `factories=_factories`, the import from
  `config.datastore`.

### `drop_actions` → `drop_tables`
- `test_read_only_pool.TestOtherWritesRaiseReadOnlyWrite.test_drop_actions_and_prune_refuse_before_anything_is_opened`
  (both loops): `{"drop_tables": ["GkSourcePolicyData"]}`, asserted text `"drop_tables"`. The
  method name stays. The module docstring's item 3 says `drop_tables` (deviation 7).
- `test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_store`:
  `drop_tables=tables_to_drop(["gk-source", "gk-source-policy-records"])`. The class docstring
  says `drop_tables` (deviation 7).
- `ComputeTargets/tests/test_reader_read_only.TestTheSixReaders.test_each_opens_its_pool_read_only`:
  `assertNotIn("drop_tables", keywords)`.

### `test_inventory_consumers.TestRunLabels`
`test_the_run_labels_from_the_structured_inventory`, `test_a_problem_in_store_tag_is_a_refusal`,
`test_a_problem_elsewhere_does_not_matter`: the fake pool is
`SimpleNamespace(primary=…, factories=factories)`, with `from config.datastore import factories`.

### `test_wkb_numeric_parent_key.TestDropOrder` (converted)
`_OLD_DROP_ORDER` is deleted. The imports are `from config.datastore import drop_groups, factories
as _factories, tables_to_drop` and `from Datastore.SQL.schema import build_schema, drop_order`. The
module docstring's item 7 says the order is derived and respects every foreign key between the
dropped tables. `_drop` calls the real actor's `_drop_tables(fake, tables_to_drop(actions))`, and
the fake has `_my_name`.

| Method | Fate |
|---|---|
| `test_the_corrected_order_drops_both` | **converted** to `test_the_derived_order_drops_both`: both `["tk-numeric", "tk-wkb"]` and `["tk-wkb", "tk-numeric"]`, as sub-tests, drop both tables under enforcement |
| `test_the_old_order_is_refused_by_enforcement` | **converted** to `test_an_unsorted_order_is_refused_by_enforcement`: with the actor module's `drop_order` patched to the identity, `["tk-numeric", "tk-wkb"]` raises `IntegrityError` (deviation 1: the prompt's `["tk-wkb", "tk-numeric"]` does not) |
| `test_tk_wkb_drops_before_tk_numeric_and_gk_wkb_before_gk_numeric` | **converted**, name kept: `drop_order` of every group's tables puts `TkWKBIntegration` before `TkNumericIntegration` and `GkWKBIntegration` before `GkNumericIntegration` |
| `test_every_declared_foreign_key_between_actions_is_respected` | **converted** to `test_every_declared_foreign_key_between_dropped_tables_is_respected`: for each group alone and for all of them, `drop_order` returns exactly the group's tables and puts every referencing table before the dropped table it references; the "all" case has at least 3 pairs |

No test is deleted.

### Added: `Datastore/tests/test_layer_registry.py` (new; 20 tests)
- `TestTheClientRegistries`
  - `test_tables_to_drop_gives_each_table_once_and_names_an_unknown_action`;
  - `test_every_table_of_every_group_is_declared_by_the_registry`;
  - `test_the_groups_are_in_the_command_line_order` (beyond the plan; deviation 10).
- `TestDropOrder`
  - `test_a_referencing_table_comes_before_what_it_references` (a chain, a duplicate, a free table,
    a self-reference and a reference outside the set);
  - `test_a_cycle_raises`.
- `TestTheDropOnAFullStore` (the real `_drop_tables` on each shard of `build_full_store`, with
  `PRAGMA foreign_keys = ON` on every connection; the store holds rows in each of the 25 dropped
  tables on one shard or the other, and is clean under `foreign_key_check` before)
  - `test_every_group_drops_from_both_shards_and_leaves_no_violation`: every table gone, enforcement
    on, `foreign_key_check` empty, and the printed lines are exactly one `Datastore: dropping table
    "<name>"` per table in `drop_order`'s order;
  - `test_unsorted_the_same_drop_is_refused_by_enforcement` (`drop_order` patched to the
    identity: `IntegrityError`).
- `TestAnUndeclaredDropTable` (`drop_tables=["GkSource", "NotATable"]`, stand-in shards)
  - `test_read_write_the_pool_refuses_it_before_anything_is_opened`: on a missing path and on an
    existing store, `RuntimeError` (not `ReadOnlyWrite`) with one "Cannot open sharded datastore"
    prefix, `'NotATable'` named and `'GkSource'` not, ending "Nothing was opened"; no actor
    constructed, `_create_engine` never called, no actor call, the missing path's directory not
    created, the existing store's checksums unchanged;
  - `test_the_actor_refuses_it_before_dropping_anything`: the real `_drop_tables` on a full-store
    shard; `GkSource` still there, every file unchanged;
  - `test_read_only_the_refusal_is_read_only_write`: on both paths, `ReadOnlyWrite` naming
    `drop_tables=`, not the registry refusal; nothing constructed, opened or changed.
- `TestSerialBatchSizes`
  - `test_the_manager_uses_the_sizes_it_is_given_and_500_otherwise`;
  - `test_the_pool_gives_every_actor_its_sizes_and_its_registry` (beyond the plan; deviation 10).
- `TestTheRecordedReplicatedSet`
  - `test_the_reader_yields_the_record_in_serial_order`: as written, `config.sharding`'s list; with
    `serial = 100 - serial`, the reversed tuple;
  - `test_a_store_the_pool_wrote_records_the_configured_set` (beyond the plan; deviation 10);
  - `test_a_primary_without_the_record_is_refused` and
    `test_a_record_naming_an_undeclared_class_is_refused`: each by the reader and by
    `read_inventory`, one "Cannot read sharded datastore" prefix, the table and the error or class
    named, every file unchanged;
  - `test_the_inventory_compares_exactly_the_recorded_classes`: with `tolerance` deleted from the
    record, `tolerance` is combined as sharded (`replicated` false, twice the records, one
    `duplicate` problem), and `redshift` is still replicated with the same records.
- `TestThePoolUsesItsRegistry.test_a_registry_without_a_class_builds_shards_without_its_tables`:
  without `OneLoopIntegral` and `OneLoopIntegral_tags`, no shard has either table and every shard
  has `QuadSourceIntegral`'s; `pool.factories` equals the registry and is a copy.
- `TestTheLayerImportsNoClient`
  - `test_a_fresh_interpreter_loads_no_client_registry`: importing the five layer modules loads
    neither `config.datastore` nor `config.sharding`, no `ObjectFactories` module but the package
    and `base`, nothing under `ComputeTargets`, `CosmologyConcepts` or `CosmologyModels`. A comment
    says why `MetadataConcepts` and `utilities` are allowed (`store_tag` is prompt 07's, U4) and
    that 09's guard generalises this;
  - `test_no_layer_file_imports_a_registry`: an `ast` check of the five files for `config.datastore`,
    `config.sharding` and `_factories`.

Running the module as a script prints the measured refusal messages below.

### Helpers and imports removed
- `_OLD_DROP_ORDER` (`test_wkb_numeric_parent_key`), with its test's subject.
- `Datastore._drop_actions` (method), `_factories`, `_drop_actions`, `_drop_order`
  (`Datastore.py`); `_default_serial_batch_size` (`ClientPool.py`); the 20 factory import
  statements of `Datastore.py` (moved to `config/datastore.py`; `base`'s stays).
- An `ast` scan of every touched Python file finds no import left unused that was used at
  `154d868` (four were unused before and are untouched: `redshift`, `wavenumber_exit_time` in
  `test_exact_identity_lookups`; `ClassInventory` in `test_inventory_report_labels`; `Optional` in
  `test_read_path_mechanics`; `tolerance` in `main.py`).

## The import test's module lists (§3, before and after)

Importing `Datastore.SQL.ShardedPool`, `Datastore.SQL.Datastore`, `Datastore.SQL.ClientPool`,
`Datastore.store_reader` and `Datastore.store_inventory` in a fresh interpreter (scratch script,
grouped by package):

- **At `154d868`:** `Datastore.SQL.ObjectFactories` 22 (the package, `base` and twenty factory
  modules); `ComputeTargets` 21; `CosmologyConcepts` 3; `CosmologyModels` 11; `MetadataConcepts` 6;
  `LiouvilleGreen` 7; `AdaptiveLevin` 2; `Units` 3; `config` 2 (`config`, `config.defaults`);
  `utilities` 1.
- **At `19f07ee`:** `Datastore.SQL.ObjectFactories` 2 (the package and `base`);
  `MetadataConcepts` 6 (`MetadataConcepts`, `.GkSourcePolicy`, `.QuadSourcePolicy`, `.store_tag`,
  `.tolerance`, `.version`); `config` 2 (`config`, `config.defaults`); `utilities` 1. Nothing from
  `ComputeTargets`, `CosmologyConcepts`, `CosmologyModels`, `LiouvilleGreen`, `AdaptiveLevin` or
  `Units`. `ray` is still loaded, through `Datastore/SQL/__init__.py`.

## The measured messages

From `test_layer_registry` run as a script (`<dir>` the temporary directory):

```
pool (undeclared drop table, read-write):
  Cannot open sharded datastore "<dir>/store/store.sqlite": cannot drop ['NotATable'], which the registry does not declare as tables. Nothing was opened
actor (its own _drop_tables):
  Datastore "registry-test": cannot drop ['NotATable'], which the registry does not declare as tables. Nothing was dropped
a primary without replicated_tables (reader; read_inventory identical):
  Cannot read sharded datastore "<dir>/full/full-store.sqlite": the replicated_tables table of the primary "<dir>/full/full-store.sqlite" could not be read (no such table: replicated_tables). Nothing was read or repaired
a record naming an undeclared class (reader; read_inventory identical):
  Cannot read sharded datastore "<dir>/full/full-store.sqlite": the replicated_tables table of the primary "<dir>/full/full-store.sqlite" names the class(es) ['NotAClass'], which the registry it was given does not declare. Nothing was read or repaired
```

The read-only pool on a primary without `replicated_tables` (scratch script, stand-in shards):
- at `19f07ee`: `RuntimeError`, the reader's message above, at its step 2;
- at `154d868`: `StoreSchemaMismatch: Cannot open sharded datastore "<dir>/store.sqlite": the
  primary "<dir>/store.sqlite" differs from the tables this code declares (it lacks the table(s)
  "replicated_tables"), so nothing was read`, at step 3.

In both trees no actor call was made by the read-only open and every file was unchanged.

## The fingerprint (§3.3)

A scratch script (the measurement's `m1_fingerprint.py`, unchanged) built `build_full_store` and
`build_real_store` in the scratchpad, took `fingerprint_of(read_inventory(…))` with a fixed
provenance, removed `taken`, and wrote it and the flattened inventory (records, problems,
`replicated`) as JSON; once at the staged tree and once at an export of `154d868` (`git archive
HEAD`, `venv` symlinked). All four pairs of files are byte-identical (`cmp`).
- `build_full_store`: digest **`6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759`**,
  problems `{}`; SHA-256 of the fingerprint file without `taken`:
  `27beb9a13f07ae25db37831f6cf1b35cb9005ce78e784fcd1dd9ae274888b26c`.
- `build_real_store`: digest `d932853819d07ce40fbb9cce544dabe4bbe968e4a7c908bb33e80eecbca3264f`,
  problems `{}`; records and problems identical.

`python -m RunRegistry store fingerprint` on a `build_full_store` primary, with a scratch
`--runs-root`, exits 0 at both trees with digest `6576b85d…`; the runs roots stayed empty and the
store unchanged.

## Deviations from the prompt

1. **`test_an_unsorted_order_is_refused_by_enforcement` names `["tk-numeric", "tk-wkb"]`, not the
   prompt's `["tk-wkb", "tk-numeric"]`.** With `drop_order` patched to the identity, the actor
   drops in the order named. `TestDropOrder`'s store holds no tag or value rows, so
   `["tk-wkb", "tk-numeric"]` is then a valid order and the test failed, "IntegrityError not
   raised" (measured). The rough version patched `sort_tables`, whose result the actor reversed, so
   the same names dropped in the other order. Named the other way round, the identity drops
   `TkNumericIntegration` while the `TkWKBIntegration` row references it, which is what the test
   exists to pin (that the actor sorts). A comment says so. `STRUCTURALLY REQUIRED`.
2. **The actor's `factories` and `serial_batch_sizes` are keyword-only, after `read_only`**, as the
   pool's are. A required parameter cannot follow the defaulted ones positionally; the only callers
   (the pool's two constructions and `test_version_row_at_open`'s `bare_actor`) pass keywords.
   `IMPLEMENTATION CHOICE`.
3. **`drop_order` is its own topological sort**, not the reverse of `sqlalchemy.schema.sort_tables`
   (which the prompt offers as an example). It breaks ties by the order given, so the order is
   fixed by the input, and raises `ValueError` on a cycle (`sort_tables` warns and proceeds).
   `IMPLEMENTATION CHOICE`.
4. **"Declared as a table" means a table `build_schema` builds from the registry**, in both the
   pool's check (`_refuse_undeclared_drop_tables`, a method) and the actor's (`self._tables`), not
   merely a registry key. The pool's message names only the undeclared names. `IMPLEMENTATION
   CHOICE`.
5. **The reader reads `replicated_tables` with `sqlite3` `mode=ro`**, as `_read_closed_store`
   reads `shards`, in a new `_read_replicated_tables`; both refusals name the table and the primary
   after the prefix. `IMPLEMENTATION CHOICE`.
6. **`tables_to_drop`'s `ValueError` also lists the groups**, and the actor keeps refusing a bare
   string as "Could not interpret drop_tables type". `IMPLEMENTATION CHOICE`.
7. **Two docstrings in named tests say `drop_tables`**: `test_read_only_pool`'s module docstring
   item 3 and `TestPruningAfterRepair`'s class docstring, each naming the renamed parameter.
   `IMPLEMENTATION CHOICE`.
8. **`test_wkb_numeric_parent_key`**: `drop_groups` is imported under its own name (the rough
   version aliased it `_drop_actions`); the comment on `datastore_module` says it is the module
   whose `drop_order` `TestDropOrder` patches; `TestDropOrder`'s docstring loses the clause that the
   order within one action is not what it checks, and says the order is derived. `IMPLEMENTATION
   CHOICE`.
9. **`test_store_inventory.store_inventory_tables`' function-scope import** of `_factories` moved to
   `config.datastore`. The plan lists the module's three calls and child string; without this the
   helper fails at its call. `STRUCTURALLY REQUIRED`.
10. **Three tests beyond the plan** in `test_layer_registry`
    (`test_the_groups_are_in_the_command_line_order`,
    `test_the_pool_gives_every_actor_its_sizes_and_its_registry`,
    `test_a_store_the_pool_wrote_records_the_configured_set`), and, inside planned tests, the
    printed-lines check of the full-store drop and the count of `_create_engine` calls in the pool
    refusal. `IMPLEMENTATION CHOICE`.
11. **`standin_pool`** imports `factories` with the other `config` imports and has a one-line
    comment beside the `setdefault`. `IMPLEMENTATION CHOICE`.
12. **The §3.2 greps 3 and 5 print lines** (below). Grep 3 matches three factory docstrings that
    mention `config/sharding.py` in prose (the pattern's `.` matches `/`); they are in
    `ObjectFactories/`, which this prompt does not touch, and were there before. Grep 5 matches
    `main.py`'s `allowed_drop_actions`, `specified_drop_actions` and `drop_actions`, which §2 keeps
    by name (`allowed_drop_actions = list(drop_groups)`; `tables_to_drop(drop_actions)`), and
    `test_read_only_pool`'s method name, which §2 says stays. No `drop_actions` remains in any layer
    file or as a keyword. `STRUCTURALLY REQUIRED`.

No `UNINTENDED DRIFT`.

## Verification performed

- **Suites**, serially on the staged tree, README §7's command, output under the session
  scratchpad (`agent06/*.txt`), verdicts grepped from the files, at load 7–27. Datastore in four
  groups: 122 + 151 + 200 + 191 = **664** (before: 122 + 131 + 200 + 191 = 644).

  | Suite | Before (orchestrator, `154d868`) | After | Difference |
  |---|---|---|---|
  | AdaptiveLevin | 32 OK | 32 OK | 0 |
  | ComputeTargets | 693 OK | 693, 1 failure: `test_tk_wkb_phase.TestCost.test_wall_time_per_object` (wall clock) | 0 |
  | CosmologyModels | 39 OK | 39 OK | 0 |
  | Datastore | 644 OK | 664 OK | +20 in `test_layer_registry` (`[g-o]`); four `TestDropOrder` methods converted, three renamed |
  | LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 0 |
  | RunRegistry | 280 OK | 280 OK | 0 |

  The wall-clock test failed alone 3 times of 3 at load ~23 (0.105 s, 0.106 s, 0.072 s against
  0.06 s), passed alone a fourth time at load 27, and failed alone 2 times of 3 on the export of
  `154d868` (0.072 s, 0.081 s). It touches nothing 06 changes. No other test outside §2's plan
  failed.
- **The §3.2 greps**, at the staged tree:
  - `grep -rn '_factories\b' Datastore/SQL/ShardedPool.py Datastore/store_reader.py Datastore/store_inventory.py | grep -v 'self\._factories'`: nothing (exit 1);
  - `grep -rnE '^_factories|^_drop_actions|^_drop_order|_default_serial_batch_size' Datastore/ --include='*.py'`: nothing (exit 1);
  - `grep -rn 'config.sharding\|config.datastore' Datastore/SQL/ Datastore/store_reader.py Datastore/store_inventory.py`:
    ```
    Datastore/SQL/ObjectFactories/GkWKBIntegration.py:35:config/sharding.py), and **NULL means pure WKB**. build() filters on it. The initial data are
    Datastore/SQL/ObjectFactories/TkWKBIntegration.py:33:on k, config/sharding.py, so the parent is always on this shard), and build() filters on it. The
    Datastore/SQL/ObjectFactories/GkSource.py:18:the GkSource's own shard (every Gk row shards on k, config/sharding.py): one row per z_source,
    ```
    (factory docstrings; deviation 12);
  - `grep -rnE 'from Datastore.SQL.ObjectFactories\.[A-Za-z_]+ import' Datastore/SQL/Datastore.py | grep -v '\.base import'`: nothing (exit 1);
  - `grep -rn 'drop_actions' Datastore/ main.py extract_*.py --include='*.py'`:
    ```
    Datastore/tests/test_read_only_pool.py:618:    def test_drop_actions_and_prune_refuse_before_anything_is_opened(self):
    main.py:121:allowed_drop_actions = list(drop_groups)
    main.py:241:    choices=allowed_drop_actions,
    main.py:312:specified_drop_actions = [x.lower() for x in args.drop]
    main.py:313:drop_actions = [x for x in specified_drop_actions if x in allowed_drop_actions]
    main.py:314:drop_tables = tables_to_drop(drop_actions)
    ```
    (deviation 12).
- **The fingerprint (§3.3)**: above.
- **`main.py` (§3.4)**, at the staged tree and at the export of `154d868`, on a `build_full_store`
  primary in the scratchpad: `--inventory` exit 0, stdout 14847 bytes and stderr empty,
  byte-identical; `--inventory --inventory-verbose` likewise (14959 bytes); `--inventory --drop
  gk-source` exit 2, stdout empty, stderr identical (`!! --inventory is read-only and does not drop
  anything; it refuses --drop (gk-source). Run the drop without --inventory`); `--drop nonsense`
  exit 2, stderr identical (1465 bytes; the choices `tk-numeric, tk-wkb, quad-source, gk-numeric,
  gk-wkb, gk-source, gk-source-policy-records, gk-source-policy, quad-source-integral,
  1loop-integral`); `--help` exit 0, stdout identical (5343 bytes). The store's files were
  unchanged.
- **The drop on stand-in shards (§3.5)**: the measurement's `m3b_pool_drop.py`, unchanged, at both
  trees: a read-write stand-in pool on a full store given every group's tables
  (`tables_to_drop(list(drop_groups))`) here and `drop_actions` naming every group at `154d868`.
  The result (each shard's `sqlite_master`, row counts and `foreign_key_check`) is byte-identical
  (`cmp`): 176 schema objects per shard, every dropped table present again and empty (the actors
  re-create it), `foreign_key_check` empty. The actors printed 25 lines per shard here, 10 there.
- **The drop under enforcement** (scratch script, copies of both full-store shards, `PRAGMA
  foreign_keys = ON`): all groups together, the derived order dropped all 25 tables on both shards
  with `foreign_key_check` empty; the old order failed at `OneLoopIntegral` (shard 0) and
  `QuadSourceIntegral` (shard 1). Alone, `tk-wkb`, `quad-source`, `quad-source-integral` and
  `1loop-integral` drop in the derived order on both shards (the old order failed all four on
  shard 0, the first three on shard 1); `tk-numeric`, `gk-numeric`, `gk-wkb`, `gk-source`,
  `gk-source-policy-records` and `gk-source-policy` fail in both orders at the group's parent
  table, as `[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]` says.
- **The recorded replicated set** (the measurement's `m2_recorded_replicated.py`, unchanged):
  `build_real_store`, `build_full_store` and a store the real `ShardedPool` created on stand-in
  shards each record exactly `config.sharding.replicated_tables`, in order (13 classes).
  `shard_store_fixtures.write_hand_built_primary` records `['version', 'wavenumber']`; it is a
  fixture's hand-written primary, not one the code writes, and none of its users reaches the
  inventory.
- **No cycle**: `drop_order` raised for no group and not for all together (the converted
  `TestDropOrder` and the full-store test call it on each).
- **`ast` against `HEAD`**, docstrings stripped: the changed definitions are the ones listed under
  "What shipped"; `config/datastore.py`'s `factories`, `serial_batch_sizes`, `drop_groups` contents
  and factory imports equal the old ones.
- **`black --check`** on the 50 touched Python files: clean.
- **Staging** by explicit path; `git diff --cached --name-only` shown before the commit.

## The deliberate-breakage record

**How each was run.** Code and tests were staged; each mutation was made in the working tree by a
scratch script and its diff taken with `git diff -- <file>` against the index (the tree of
`19f07ee`), so `git apply` replays it from a clean checkout. For each, in `bash`: `git apply -R
--check`, the run, `git apply -R`, then `git apply --check` (forward) on the restored tree; all
passed and `git diff` was empty after each. None is committed. Each ran three modules:
`test_layer_registry`, `test_wkb_numeric_parent_key` and `test_store_reader` (74 tests; OK
unmutated).

**(i) `drop_order` returns its input unchanged.** Ran 74, **FAILED (failures=41, errors=3)**:
- `test_layer_registry.TestTheDropOnAFullStore.test_every_group_drops_from_both_shards_and_leaves_no_violation`,
  both shards (error, `IntegrityError`);
- `test_wkb_numeric_parent_key.TestDropOrder.test_the_derived_order_drops_both`, sub-test
  `['tk-numeric', 'tk-wkb']` (error, `IntegrityError`);
- `test_layer_registry.TestDropOrder.test_a_referencing_table_comes_before_what_it_references`;
- `TestDropOrder.test_tk_wkb_drops_before_tk_numeric_and_gk_wkb_before_gk_numeric`;
- `TestDropOrder.test_every_declared_foreign_key_between_dropped_tables_is_respected`, 39
  sub-tests (every referencing pair within each group, and every pair in "all").

```diff
diff --git a/Datastore/SQL/schema.py b/Datastore/SQL/schema.py
index 8e9300d..dc1817f 100644
--- a/Datastore/SQL/schema.py
+++ b/Datastore/SQL/schema.py
@@ -184,7 +184,7 @@ def drop_order(tables: Iterable[sqla.Table]) -> List[sqla.Table]:
         ordered.append(free[0])
         remaining.remove(free[0])
 
-    return ordered
+    return given
 
 
 class SchemaDifferences(NamedTuple):
```

**(ii) `read_inventory` takes its replicated set from the registry's keys.** Ran 74, **FAILED
(failures=1)**: `test_layer_registry.TestTheRecordedReplicatedSet.test_the_inventory_compares_exactly_the_recorded_classes`
(`True is not false`: `tolerance` was compared as replicated).

```diff
diff --git a/Datastore/store_inventory.py b/Datastore/store_inventory.py
index 527f2c1..0617fef 100644
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -850,7 +850,7 @@ def read_inventory(primary: PathType, factories: Mapping[str, Any]) -> StoreInve
     from Datastore.store_reader import open_read_only
 
     with open_read_only(primary, factories) as store, contextlib.ExitStack() as stack:
-        replicated = set(store.replicated_tables)
+        replicated = set(factories)
         conns = {
             shard.serial: stack.enter_context(shard.engine.connect())
             for shard in store.shards
```

**(iii) The pool's undeclared-drop-table check is skipped.** Ran 74, **FAILED (failures=2)**:
`test_layer_registry.TestAnUndeclaredDropTable.test_read_write_the_pool_refuses_it_before_anything_is_opened`,
both sub-tests: on the missing path, `RuntimeError not raised` (the store was created); on the
existing store, the refusal came from the actors' own `_drop_tables`
(`Datastore "shard0002-store": cannot drop ['NotATable'], …`), after the primary was opened and
the actors constructed, so the "Cannot open sharded datastore" prefix assertion failed.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 39c0045..4dff617 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -206,7 +206,6 @@ class ShardedPool:
 
         # a drop table the registry does not declare is refused before the primary is created or
         # opened
-        self._refuse_undeclared_drop_tables()
 
         # if primary file is absent, all shard databases should be likewise absent
         if self._primary_file.is_dir():
```

**(iv) `SerialPoolManager` ignores `batch_sizes`.** Ran 74, **FAILED (failures=1)**:
`test_layer_registry.TestSerialBatchSizes.test_the_manager_uses_the_sizes_it_is_given_and_500_otherwise`
(`500 != 7`). `test_the_pool_gives_every_actor_its_sizes_and_its_registry` passes: it checks what
each actor was given, not what it leases.

```diff
diff --git a/Datastore/SQL/ClientPool.py b/Datastore/SQL/ClientPool.py
index 6c9e814..fbb502b 100644
--- a/Datastore/SQL/ClientPool.py
+++ b/Datastore/SQL/ClientPool.py
@@ -127,7 +127,7 @@ class SerialPoolManager:
                 table,
                 self._profiler,
                 self._broker,
-                default_batch_size=self._batch_sizes.get(table, 500),
+                default_batch_size=500,
             )
 
         serial = self._tables[table].lease_serial()
```

**(v) The reader stops refusing a recorded class the registry does not declare.** Ran 74,
**FAILED (failures=2)**:
`test_layer_registry.TestTheRecordedReplicatedSet.test_a_record_naming_an_undeclared_class_is_refused`,
sub-tests `reader` and `inventory`, each `RuntimeError not raised`. (A first run also errored with
an `IndexError` in the test's helper, which returned the first message when there was none; the
helper now returns `None` then, and the run above is with that version.)

```diff
diff --git a/Datastore/store_reader.py b/Datastore/store_reader.py
index dd7e4a1..4ac07bb 100644
--- a/Datastore/store_reader.py
+++ b/Datastore/store_reader.py
@@ -163,7 +163,7 @@ def _read_replicated_tables(primary: Path, factories: Mapping) -> Tuple[str, ...
         ) from e
 
     names = tuple(row[1] for row in rows)
-    undeclared = [name for name in names if name not in factories]
+    undeclared = []
     if len(undeclared) > 0:
         raise _refuse(
             primary,
```

## Observations not acted on

- **Under (iii), a new store accepts any drop table.** The actor's own check runs only in
  `_drop_tables`, which a new file never reaches, so without the pool's check a read-write open of
  a missing path given `NotATable` creates the store silently; on an existing store the actors
  refuse, but only after the primary is opened, the check at open has run and every actor exists.
  The pool's check is the only one before anything opens. As intended; recorded.
- **Under (v), `read_inventory` reads a record naming an unknown class without complaint**: it
  consults the set only for `INVENTORY_CLASSES`. The reader's refusal is the only guard.
- **`read_inventory` needs every `INVENTORY_CLASSES` name in the registry it is given**
  (`factories[name]`, `shard.tables[name]`); a smaller registry raises `KeyError`. The inventory's
  class list is audit C1 row 12, prompt 08's.
- **`[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]`**: the six groups it names
  still fail alone under enforcement in the derived order (above). Its board entry's next step
  says a cascade would be "derivable from the schema, as `schema.drop_order` will be"; it now is.
  Not 06's.
- **Factory docstrings name `config/sharding.py`** (`GkWKBIntegration.py:35`,
  `TkWKBIntegration.py:33`, `GkSource.py:18`). Prose in the client, not an import.
- **Both reader refusals name the primary twice** (the prefix and "the replicated_tables table of
  the primary …"), as the reader's existing "the primary … records no shards" does.
- **`ShardedPool._recompute_validated`** still reads `self._factories["BackgroundModel"]`, and the
  pool still imports `MetadataConcepts.store_tag` (C1 rows 1 and 6, prompt 07).

None opens an issue.

## State handed to the next prompt

- The layer is given its registries: `ShardedPool(…, *, factories, serial_batch_sizes=None)` with
  `drop_tables`; the actor takes the same three; `ShardedPool.factories` returns a copy;
  `open_read_only(primary, factories)`, `read_inventory(primary, factories)`;
  `ReadOnlyStore.replicated_tables` is the primary's record. Every `build_schema` in the pool uses
  `self._factories`, so a fact a factory declares in `register()` reaches the pool through
  `built.records` (07).
- `Datastore.SQL.schema.drop_order(tables)` orders any set of tables by their foreign keys; 07's
  copy-as-a-unit rule can derive from the same keys.
- `config/datastore.py` holds `factories`, `drop_groups`, `serial_batch_sizes` and
  `tables_to_drop`; `main.py`, the extract scripts, `extract_common`, `RunRegistry/stores.py` and
  `tools/inventory_report.py` pass or read them.
- Importing the layer loads `MetadataConcepts` (through `ShardedPool`'s `store_tag`, U4) and
  `utilities`; `test_layer_registry.TestTheLayerImportsNoClient` pins only what 06 removed, and 09
  generalises it.
- `Datastore` 664, every other suite unchanged. The index holds 107.
