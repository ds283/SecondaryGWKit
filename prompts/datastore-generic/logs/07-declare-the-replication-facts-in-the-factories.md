# Log — prompt 07: declare the replication facts in the factories

**Subject:** *Declare the replication facts in the factories, not the pool* · **Model:** Claude
Opus 5.5 · **Date:** 2026-10-06 · **Base:** `f1b8d02`, clean · **Result:** landed.
- **F1:** `BackgroundModelValue` declares `owner_column`; `ShardedPool._unit_tables` gives the copy
  unit of the check at open and the prune unit, read from foreign keys through the declared owner
  and the tag table. The hard-coded `BackgroundModel` unit and `_prune_unit` are gone.
- **F2:** `redshift` and `wavenumber` declare `monotone_flags`; the repair licence's flags and its
  refusal text come from the declarations.
- **F3:** `BackgroundModel` declares `validated_column` and implements `revalidate`; the check at
  open recomputes through it. `_StoredModel` is gone. `read_records(validated=…)` takes the
  column's name.
- **F4:** `Datastore/contract.py` names the version and tag tables and their columns once. The
  pool builds the version object through the version table's factory and imports no
  `MetadataConcepts`; `owned_serials` replaces `_value_serials`; `ShardedPool.py` names no
  registry table but the contract's, in code, comments or messages.

No stop condition fired. Closes nothing. Narrows
`[00-the-generic-datastore-layer-hard-codes-project-schema-facts]` (audit C1 rows 1, 2, 4, 5, 6, 7
and 8). Opens nothing. The index stays at 107. The commit is `3f51939`.

## What shipped

**`Datastore/contract.py` (new).** `VERSION_TABLE = "version"`, `VERSION_LABEL = "label"`,
`TAG_TABLE = "store_tag"`, `TAG_LABEL = "label"`, `TAG_SERIAL = "tag_serial"`. It imports nothing.
Its docstring says why these are the layer's (every storable row's prepended `version` column
references the version table, a pool is opened under a version label, and every tagged class
associates with the tag table through a `tag_serial` column), and that the layer builds their
objects through the factories registered under these names.

**`Datastore/SQL/schema.py`.**
- The prepended `version` column's foreign key is `f"{VERSION_TABLE}.serial"`; the DDL is
  unchanged (the witness's table descriptions are identical, below).
- `build_schema` puts `owner_column`, `monotone_flags` (a tuple) and `validated_column` into the
  record of every class with a table, each checked against the table by
  `_declared_owner_column`, `_declared_monotone_flags` and `_declared_validated_column`; a misfit
  raises `ValueError` through `_refuse_declaration`. A record of a class with no table is
  unchanged.
- The module docstring says it imports `sqlalchemy` and `Datastore.contract`, and names the three
  keys; `build_schema`'s docstring says what each record now holds.

**`Datastore/SQL/ObjectFactories/base.py`.** `revalidate(serial, conn, table, tables) -> bool`
raises `NotImplementedError`; `owned_serials(obj)` returns `None`. Both static, not abstract.

**`Datastore/SQL/ObjectFactories/BackgroundModel.py`.**
- `sqla_BackgroundModelFactory.register()`: `"validated_column": "validated"`, with a one-line
  comment.
- `_validate_row(serial, label, conn, table, tables)` holds the rule (value-row count against
  `z_samples`), the warning (same text) and the write. `validate(obj, …)` keeps its `obj.available`
  check and calls it with `obj.store_id`, `obj.label`; `revalidate(serial, …)` reads the label from
  the row (`scalar_one()`) and calls it.
- `owned_serials(obj)`: `[getattr(v, "_my_id", None) for v in obj._values]`, or `None` with no
  `_values` (what `_value_serials` returned).
- `inventory_records`: `validated="validated"`.
- `sqla_BackgroundModelValue_factory.register()`: `"owner_column": "model_serial"`, with a
  one-line comment.

**`redshift.py`, `wavenumber.py`.** `"monotone_flags": ("source", "response")`, each with a
one-line comment.

**Six factories** (`GkNumericIntegration.py:656`, `GkSource.py:874`, `GkWKBIntegration.py:986`,
`QuadSource.py:566`, `TkNumericIntegration.py:658`, `TkWKBIntegration.py:981`):
`validated=True` → `validated="validated"`.

**`Datastore/store_inventory.py`** (only `read_records` and the import):
- `validated: Optional[str] = None`, selected as `table.c[validated].label("validated")`, so the
  row mapping and the record are unchanged; the docstring's bullet says so.
- The tag lookup uses `TAG_TABLE`, `TAG_LABEL` and `TAG_SERIAL`; the "orphan-tag" message is
  `f"{TAG_TABLE} that is not there; …"`, the same text.

**`Datastore/SQL/Datastore.py` (the actor).** Imports `VERSION_TABLE`; `_insert`'s lease test is
`cls_name != VERSION_TABLE` and its comment says "the version table"; `_refuse_insert`'s
docstring names no class. Nothing else.

**`Datastore/SQL/ShardedPool.py`.**
- Imports: `from MetadataConcepts import store_tag` and `from Datastore.object import
  DatastoreObject` deleted; `TAG_TABLE`, `VERSION_LABEL`, `VERSION_TABLE` imported.
- The version row: the get in `__init__`, `_version_labels_present` and the read-only
  `ReadOnlyMiss` use the constants. `_find_version_row` keeps its raw `mode=ro` read of every
  shard, its duplicate and disagreement checks (constants in the SQL), then builds the object by
  `self._factories[VERSION_TABLE].build(payload={VERSION_LABEL: label}, conn=…, table=…,
  inserter=refuse_insert, tables=…, inserters={})` on a `mode=ro` SQLAlchemy connection
  (`store_reader.read_only_url`, imported in the function) to the lowest-serial shard holding
  the row, rolling back and disposing of the engine after. `refuse_insert` raises
  `RuntimeError`. No new refusal.
- `_replicated_table_specs` excludes `TAG_TABLE`; each spec carries `owner_column`,
  `monotone_flags` and `validated_column` from the record. `_plan_replicated_repair` keeps its
  signature.
- `_unit_tables(cls_name, specs)` (static): the class, then its tag tables
  (`_is_tag_table_of`: a foreign key to `TAG_TABLE` and every other one to the class), then the
  tables whose declared `owner_column` references it (`_declares_owner`), each group in the
  specs' order. `_column_naming(spec, cls_name)` gives a member's column naming the class.
- The licence: `unit = self._unit_tables(cls_name, specs)`; `flag_columns` from the specs; the
  flag-or-validated refusal's words from the declared flags (their union in specs' order joined
  by " or ") and the declared validated columns; the recompute licensed when `name == cls_name`,
  the class declares a `validated_column` and `differing == [it]`. The as-unit check runs when the
  unit has more than one table, its refusals name `cls_name`, and `models`/`model_column`/`model_of`
  are `owners`/`owner_index`/`owner_of`. "model row", "part of this model" and "value rows" stay,
  with a comment saying why.
- `_recompute_validated` reads the class from the record, calls `factory.revalidate(serial,
  conns[sid], table, built.tables)` in the same savepoint, reads the stored flag from the
  declared column, and its refusal says `f"{column}=True"`. `_StoredModel` is deleted; the action
  keeps `"validated recomputed"` and `"validated": True`.
- `_prune_shard` uses `_unit_tables`; its refusal names `unit_named`, the class then the rest of
  the unit in the specs' order. `_prune_unit` is deleted.
- `_value_serials` is deleted; `_store_impl_replicated_table`'s `replication` looks up
  `self._factories[cls_name]` and calls `owned_serials` on the controller's object and on each
  answer. The `ReplicationMismatch` detail is unchanged.
- The `shard_key` detail: `f"the primary assigns {self._ShardKeyType_name} serial …"`.
- Comments and docstrings (the "THE REPLICATED TABLES" block, the specs, differences, licence,
  apply, recompute, prune block, `_store_impl_replicated_table`, `_assign_shard_keys`): no
  registry table, project column or package; "the shard-key class", "the version table", "the tag
  table".

By `ast` against `f1b8d02` (scratch), with docstrings stripped, the pool's functions whose code
changed are exactly `__init__`, `_find_version_row` (with its nested `refuse_insert`, new),
`_open_read_only`, `_version_labels_present`, `_replicated_table_specs`, `_describe_differences`,
`_plan_replicated_repair`, `_recompute_validated`, `_prune_shard` and
`_store_impl_replicated_table` (with its nested `replication` and `check`). Docstring only:
`_replicated_differences` and `_apply_replicated_repair`. Comment only: `_assign_shard_keys` and
the two block comments. New: `_is_tag_table_of`, `_declares_owner`, `_unit_tables`,
`_column_naming`. Gone: `_prune_unit`, `_value_serials`.

## The tests, test by test

### Edited
- `Datastore/tests/test_schema_builder.py`: `WITNESS` names
  `data/schema_at_datastore-generic-07.json`; the module docstring's list of witnesses gains it
  first, with the reason ("no table changes; every record of a class with a table carries three
  more keys…, non-default on `redshift`, `wavenumber`, `BackgroundModel` and
  `BackgroundModelValue`"). No method changes. Both
  `TestSchemaIsUnchanged.test_build_schema_reproduces_the_witness` and
  `.test_actor_build_schema_reproduces_the_witness` now read the new witness.
- `Datastore/tests/test_layer_registry.py`: `FORBIDDEN_PACKAGES` gains `"MetadataConcepts"`; the
  comment above it says `utilities` is the one project package still allowed (the actor's
  `WallclockTimer`), and that `MetadataConcepts` is no longer loaded because the pool builds the
  version object through the version factory. No method changes.

No test is deleted. No other test is edited.

### Added: `Datastore/tests/data/schema_at_datastore-generic-07.json`
Captured with `PYTHONPATH=. ./venv/bin/python Datastore/tests/schema_description.py
Datastore/tests/data/schema_at_datastore-generic-07.json` at the staged tree. The old witness
`schema_at_datastore-integrity-09c.json` is untouched.

### Added: `Datastore/tests/test_declared_facts.py` (new; 28 tests)
- `TestTheRecords`
  - `test_every_record_of_a_class_with_a_table_carries_the_three_keys`;
  - `test_exactly_the_four_declarations_are_not_default`;
  - `test_a_record_with_no_table_is_unchanged`.
- `TestBuildSchemaRefusesAMisfit` (a stand-in `Member` beside `Owner` and `Other`; each refusal
  asserts `ValueError`, `"Member"`, the key, the value and the misfit)
  - `test_the_fitting_declarations_are_accepted` (beyond the plan, deviation 7);
  - `test_an_owner_column_that_is_not_a_column`, `test_an_owner_column_with_no_foreign_key`,
    `test_an_owner_column_with_two_foreign_keys`;
  - `test_monotone_flags_given_as_a_string`, `test_monotone_flags_naming_a_column_that_is_absent`,
    `test_monotone_flags_naming_a_column_that_is_not_boolean`;
  - `test_a_validated_column_that_is_not_a_column`, `test_a_validated_column_that_is_not_boolean`.
- `TestTheUnit`
  - `test_every_compared_table_of_the_registry` (14 compared tables: `BackgroundModel`'s three in
    order, each other alone);
  - `test_a_declared_owner_joins_and_an_undeclared_reference_does_not` (hand-built: `Owner`'s unit
    is `[Owner, Owner_tags, Owned]`, `Referrer` with a foreign key and no declaration is not in it,
    every other table is alone; `_column_naming` gives `owner_ref` and `owner_id`).
- `TestTheFlags`
  - `test_the_declared_flags_reach_the_specs` (and the declared validated and owner columns);
  - `test_a_declared_flag_is_set_after_a_get`, `test_an_undeclared_flag_is_refused_in_the_declarations_words`,
    `test_the_declared_validated_column_is_recomputed_after_a_validate` (beyond the plan's wording,
    deviation 7: `_plan_replicated_repair` on two hand-built shard files, whose refusal reads "only a
    on or lit flag after a get, or ok after a validate, …").
- `TestRevalidate`
  - `test_the_same_result_flag_and_warning_as_stored_and_with_a_value_row_deleted` (a copy of
    `build_full_store`'s shard 0, on the pool's `_reconcile_engine`, each call in a rolled-back
    savepoint; the copy's checksum unchanged);
  - `test_the_base_hook_raises`.
- `TestOwnedSerials`
  - `test_the_base_owns_nothing`, `test_a_background_model_names_its_value_rows`;
  - `test_the_pool_checks_each_answer_through_the_hook` (beyond the plan, deviation 7; see
    breakage (v)).
- `TestTheVersionObject.test_built_by_the_registry_version_factory_with_the_attributes_it_had`
  (a recording subclass of the registry's version factory; read-write and read-only reopen: one
  `mode=ro` build with payload `{"label": "standin"}` and `inserters={}`, whose inserter raises
  `RuntimeError`; `vars()` exactly `_my_id`, `_label`, `_deserialized`; the type the fresh
  store's get built).
- `TestThePruneRefusalNamesItsUnit.test_a_prune_that_deletes_outside_its_unit` (M7's `AFTER
  DELETE` trigger on `tolerance`; subclasses `test_prune_at_open._PruneTestCase`, which has no test
  methods of its own).
- `TestNoProjectNameInThePool`
  - `test_no_string_or_name_is_a_registry_table` (`ast`: every string constant's words and every
    name, attribute, function, class, argument and keyword, and their `_`-separated parts);
  - `test_no_import_names_a_project_package` (an allow-list: the standard library, `ray`,
    `sqlalchemy`, `Datastore` but no factory module except `base`, and `config.defaults`);
  - `test_no_comment_or_docstring_names_one` (`tokenize` comments, and docstrings).

  Forbidden: the registry's tables but the contract's; the repository's packages but `Datastore`
  and `config`; the registry's columns that are identifiers (an underscore or a capital), but the
  layer's own; and the phrases "source or response" and "source/response flag". A plain-word
  column (`source`, `response`, `type`, `value`) cannot be told from prose — the closed-store
  operations say "the source store" — so the two declared flags are checked as phrases. "model"
  and "value rows" are allowed and the docstring says why. Pointed at `f1b8d02`'s `ShardedPool.py`
  (scratch), all three fail.

## The schema witnesses

`witness_diff.py` (the measurement's) on the two files:

```
added record key 'monotone_flags' on 38 classes; non-default: {'redshift': ['source', 'response'], 'wavenumber': ['source', 'response']}
added record key 'owner_column' on 38 classes; non-default: {'BackgroundModelValue': 'model_serial'}
added record key 'validated_column' on 38 classes; non-default: {'BackgroundModel': 'validated'}
```

No top-level key, class order, table description or `has_table` differs. `diff` gives 145 added
lines and 25 removed: 114 are the three keys on 38 records, 6 the two non-empty flag lists' items
and brackets, and 25 pairs are a record's `"validate_on_startup": false` gaining a comma where
`validated_column` now follows it. The hunks with a non-default value (`diff -U2`):

```diff
@@ -135,4 +137,5 @@
     "use_version": true,
     "validate_on_startup": true,
+    "validated_column": "validated",
     "version_col": {
      "column": "version",
@@ -638,5 +641,7 @@
      }
     ],
+    "monotone_flags": [],
     "name": "BackgroundModelValue",
+    "owner_column": "model_serial",
     "serial_col": {
      "column": "serial",
@@ -10912,5 +11008,10 @@
      }
     ],
+    "monotone_flags": [
+     "source",
+     "response"
+    ],
     "name": "redshift",
+    "owner_column": null,
     "serial_col": {
      "column": "serial",
@@ -11275,5 +11386,10 @@
      }
     ],
+    "monotone_flags": [
+     "source",
+     "response"
+    ],
     "name": "wavenumber",
+    "owner_column": null,
     "serial_col": {
      "column": "serial",
```

## `build_schema`'s refusal messages, as measured

```
build_schema: the factory of "Member" declares owner_column='nonesuch', which is not a column of its table; the columns of its table are ['serial', 'plain', 'owner_serial', 'two_keys', 'flag']
build_schema: the factory of "Member" declares owner_column='plain', a column with 0 foreign keys, where the column naming a row's owner has exactly one; the columns of its table are [...]
build_schema: the factory of "Member" declares owner_column='two_keys', a column with 2 foreign keys, where the column naming a row's owner has exactly one; the columns of its table are [...]
build_schema: the factory of "Member" declares monotone_flags='flag', a string, where a sequence of column names is expected; the columns of its table are [...]
build_schema: the factory of "Member" declares monotone_flags=('flag', 'nonesuch'), and ['nonesuch'] are not columns of its table; the columns of its table are [...]
build_schema: the factory of "Member" declares monotone_flags=('flag', 'plain'), and ['plain'] are not Boolean columns; the columns of its table are [...]
build_schema: the factory of "Member" declares validated_column='nonesuch', which is not a column of its table; the columns of its table are [...]
build_schema: the factory of "Member" declares validated_column='plain', a column of type Integer(), which is not Boolean; the columns of its table are [...]
```

(`[...]` is the same list as the first line.)

## Imports (§3.6)

Each of the five layer modules imported alone in a fresh interpreter (scratch), project modules
loaded, by package:

| Module | At `f1b8d02` | At `3f51939` |
|---|---|---|
| `Datastore.SQL.ShardedPool` | `ObjectFactories` (package, `base`); `config`, `config.defaults`; `MetadataConcepts` 6 (`MetadataConcepts`, `.GkSourcePolicy`, `.QuadSourcePolicy`, `.store_tag`, `.tolerance`, `.version`); `utilities` | `ObjectFactories` (package, `base`); `config`, `config.defaults`; `utilities`; `Datastore.contract` |
| `Datastore.SQL.Datastore` | `ObjectFactories` (package, `base`); `config`, `config.defaults`; `utilities` | the same, and `Datastore.contract` |
| `Datastore.SQL.ClientPool` | the same as the actor | the same, and `Datastore.contract` |
| `Datastore.store_reader` | as `ShardedPool`, with the 6 `MetadataConcepts` modules | as `ShardedPool`, no `MetadataConcepts` |
| `Datastore.store_inventory` | none | `Datastore.contract` |

No `MetadataConcepts` module is loaded by any of the five. (Importing `ClientPool` runs
`Datastore/SQL/__init__.py`, which imports the actor, hence its list.)

## Project names left in the five layer files

A scratch scan (`tokenize`) for the registry's tables, the repository's packages, the registry's
identifier columns and the words `source`, `response`, `validated`, `label`, `model`, `value`:

- **`ShardedPool.py`**
  - `version` (comments 195, 291, 457, 521, 1065, 3018; strings 356–644, 1621): the version row
    and version label — **contract**;
  - `label` (339, 356, 383, 390, 404, 458, 529, 570, 644; locals 656–658): the version label —
    **contract**;
  - `validated` (comments 1089, 1673, 3474; strings 1621, 1881, 1940, 1970, 1975, 2028, 2030): the
    declared validated column in prose, the action record's `"validated recomputed"` and
    `"validated": True`, and the printed "recomputed validated of …" — **layer vocabulary**;
  - `model` (comment 1809; messages 1838, 1850: "holds its model row", "part of this model") —
    **layer vocabulary**, kept so that no message changes;
  - `value` (comment 174; strings 1427, 1987, 2029, 2429, 3375; locals `value_serials` 3280–3281):
    "same value", "its own value rows", "its value rows have serials" — **generic English / layer
    vocabulary**;
  - `source` (closed-store operations, 912–2918): a source store — **generic English**;
  - `config` (the `config.defaults` import, audit C2's generic constant; the locals
    `for class_name, config in read_table_config.items()`) — **generic**.
- **`Datastore.py`**: `version` and `label` (docstrings 56, 147–160, 720–721; comments 91, 167,
  169, 715, 733; the payload key `"version"` at 745, the prepended column) — **contract / layer's
  prepended column**; `value` (270, 873) — **generic English**; `config` (locals 863–864) —
  **generic**.
- **`schema.py`** *(added at the orchestrator's review; the scan above did not list it)*:
  `version` (the module and `build_schema` docstrings; the registration key `"version"` at 117
  and the prepended column's name at 121) — **contract / layer's prepended column**; `validated`
  (the module docstring and `_declared_validated_column`'s, naming the new key) — **layer
  vocabulary**.
- **`ClientPool.py`, `store_reader.py`**: none.
- **`store_inventory.py`**: the class list and its comments (100–142: every class name,
  `CosmologyModels`, `cosmology_type`), the module docstring (line 1), and the `Parent` and
  `ParentSet` docstrings' examples (239, 265: `QuadSource`, `GkWKBIntegration`, `GkSource`,
  `GkSource_parents`) — **prompt 08's (C1 rows 12, 13) and the audit's comment list (SI:247-256,
  268-270)**, untouched here (§2); in `read_records`, the locals `parent_serial` and `tag_serial`
  (536–578) — **generic local names**, and `validated` (the record field, the parameter and its
  label) — **layer vocabulary**.

## Deviations from the prompt

1. **`_unit_tables` is helped by three static methods**: `_is_tag_table_of`, `_declares_owner`
   and `_column_naming` (the prompt's "a helper gives a member's column naming the class").
   `IMPLEMENTATION CHOICE`.
2. **`model_of` is renamed `owner_of`**, beside the two renames the prompt names (`models` →
   `owners`, `model_column` → `owner_index`). No text changes. `IMPLEMENTATION CHOICE`.
3. **`build_schema`'s refusals also cover a declared value that is not a string** (an
   `owner_column`, `validated_column` or flag that is not a `str` is "not a column of its
   table"), and every message ends with the table's columns. `IMPLEMENTATION CHOICE`.
4. **The pool's factory lookup for `owned_serials` is inside `replication`**, so nothing new runs
   before `_replicated_write` begins. `IMPLEMENTATION CHOICE`.
5. **`revalidate` reads the label with `scalar_one()`**: a serial with no row raises
   `NoResultFound`. Its only caller has already found the row in the snapshot.
   `IMPLEMENTATION CHOICE`.
6. **`_find_version_row`'s unreachable inserter** raises `RuntimeError('Sharded datastore "<p>":
   building the version row of label "<l>" would insert {…}, though every shard holds it')`.
   `IMPLEMENTATION CHOICE`.
7. **Tests beyond the plan**: `test_the_fitting_declarations_are_accepted`; the flags pinned
   through `_plan_replicated_repair` on hand-built shards (three tests, where the plan says "the
   flags reach the specs, and `flag_columns`" — `flag_columns` is a local, so it is pinned by what
   it licenses and refuses); `test_the_pool_checks_each_answer_through_the_hook`;
   `test_no_import_names_a_project_package`; the `_column_naming` assertions. `IMPLEMENTATION
   CHOICE`.
8. **The text check treats the two declared flags as phrases**, not words, because the
   closed-store operations' "source" is a source store (the prompt's §2 keeps it). `IMPLEMENTATION
   CHOICE`.
9. **Breakage (iv) was run in two forms** (below). Read literally ("returns True without
   counting", writing nothing), it fails the new `revalidate` test and two `test_reconcile_at_open`
   tests, but not `test_a_recomputed_validated_that_disagrees`: the unwritten flag leaves the
   shards differing, and the comparison after the repair refuses with the words that test asserts.
   Writing `validated=True` and returning `True` without counting fails that test, as the prompt
   expects. `STRUCTURALLY REQUIRED` (for the named test to fail, the mutation must write).
10. **README §4's "07 ships"** also names `read_records(validated=<column name or None>)` and which
    factories declare what. `IMPLEMENTATION CHOICE`.

No `UNINTENDED DRIFT`.

## Verification performed

- **Suites**, serially on the staged tree, README §7's command, output under the session
  scratchpad (`agent07/*.txt`), verdicts grepped from the files, at load 7–30. Datastore in four
  groups: 150 + 151 + 200 + 191 = **692** (before: 122 + 151 + 200 + 191 = 664).

  | Suite | Before (orchestrator, `f1b8d02`) | After | Difference |
  |---|---|---|---|
  | AdaptiveLevin | 32 OK | 32 OK | 0 |
  | ComputeTargets | 693 OK | 693 OK | 0 |
  | CosmologyModels | 39 OK | 39 OK | 0 |
  | Datastore | 664 OK | 692 OK | +28 in `test_declared_facts` (`[a-f]`) |
  | LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 0 |
  | RunRegistry | 280 OK | 280 OK | 0 |

  No test failed, and no flake appeared.
- **The §3.2 greps**, at the staged tree:
  - `grep -nE 'MetadataConcepts|store_tag\b|BackgroundModel|redshift|wavenumber|_StoredModel|_value_serials|_prune_unit' Datastore/SQL/ShardedPool.py`:
    nothing (exit 1).
  - `grep -nE '"(version|store_tag|label|tag_serial)"' Datastore/SQL/ShardedPool.py Datastore/SQL/Datastore.py Datastore/SQL/schema.py Datastore/store_inventory.py | grep -v 'label("validated")'`:
    five lines (14 at `f1b8d02`):
    ```
    Datastore/SQL/Datastore.py:745:                payload = payload | {"version": self._version_serial}
    Datastore/store_inventory.py:116:    "version",
    Datastore/store_inventory.py:117:    "store_tag",
    Datastore/SQL/schema.py:117:        use_version = registration_data.get("version", False)
    Datastore/SQL/schema.py:121:                "version",
    ```
    The actor's payload key and `schema.py`'s registration key and prepended column are the
    layer's own `version` column; `store_inventory.py:116-117` is the inventory's class list,
    prompt 08's.
  - `grep -nF 'table.c.validated' Datastore/store_inventory.py` and
    `grep -nF 'validated=True' Datastore/SQL/ObjectFactories/*.py`: nothing (exit 1 each).
- **The fingerprint (§3.3).** The measurement's `m1_fingerprint.py`, unchanged, at the staged tree
  and at an export of `f1b8d02` (`git archive`, `venv` symlinked): `build_full_store`'s
  fingerprint without `taken`, its inventory, and `build_real_store`'s fingerprint and inventory,
  four pairs of files, each byte-identical (`cmp`).
  - `build_full_store`: digest **`6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759`**,
    problems `{}`; the fingerprint file's SHA-256 `27beb9a13f07ae25db37831f6cf1b35cb9005ce78e784fcd1dd9ae274888b26c`.
  - `build_real_store`: digest `d932853819d07ce40fbb9cce544dabe4bbe968e4a7c908bb33e80eecbca3264f`,
    problems `{}`.
  - `sqlite_master` of all six files (two stores, primary and two shards each) is identical.
  - `main.py --database <full primary> --inventory`: exit 0, stdout 14844 bytes, stderr empty,
    byte-identical at both trees; the store's files unchanged.
- **The check at open, recorded (§3.4).** The measurement's `m2_open_behaviour.py`, extended to
  record the version object's module and `vars()`, run with `PYTHONHASHSEED=0` and `random`
  seeded per test, at the staged tree and at the export: **111 tests** of `test_reconcile_at_open`,
  `test_prune_at_open`, `test_version_row_at_open`, `test_replicated_write` and
  `test_read_only_pool` (110 with a construction), **328 constructions** (272 opened, 56 raised),
  **982 printed lines**; 35 with something repaired and 2 with something pruned; 126 version
  objects from `_find_version_row` (`vars()` `_my_id`, `_label`, `_deserialized`, type
  `MetadataConcepts.store_tag.store_tag`) and 146 from a get. The two records (304203 bytes, SHA-256
  `5699299b4c4062c49d834da59aada89536ff855211dc8f6381aea2aa36c9dd2b`) are byte-identical (`cmp`),
  and again at the final staged tree.
- **The messages (§3.5).** The measurement's `m7_messages.py`, unchanged, at both trees: 16
  scenarios, byte-identical (`cmp`): the two as-unit refusals and their order, the flag-or-validated
  refusal, the recompute refusal, the `shard_key` detail, the `ReplicationMismatch` details (owned
  serials differ; no owned rows; an equal answer accepted), the inventory's "orphan-tag" problems,
  the version `ReadOnlyMiss` and its fields, the prune refusal, and the printed lines and actions
  of a repaired validate.
- **`revalidate` and `validate`** (the measurement's `m9_revalidate.py`) on copies of both
  `build_full_store` shards: `validate` at both trees, and `revalidate` here, give the same return,
  flag and warning for both models, as stored and with a value row deleted; each copy's checksum
  is unchanged.
- **The unit** (the measurement's `m3_units.py`): `_unit_tables` equals the old copy unit for every
  one of the 14 compared tables; for `BackgroundModel`, the one class that prunes, it is the old
  prune unit as a set.
- **Imports (§3.6)**: above.
- **`black --check`** on the 18 touched Python files: clean.
- **Unused imports**: an `ast` scan of every touched Python file finds none; `DatastoreObject` and
  `store_tag` were removed from the pool with their last uses.
- **Staging** by explicit path; `git diff --cached --name-only` shown before the commit.

## The deliberate-breakage record

**How each was run.** Code and tests were staged; each mutation was made in the working tree by a
scratch script and its diff taken with `git diff -- <file>` against the index (`3f51939`'s
tree), so `git apply` replays it from a clean checkout. For each, in `bash`: the run, `git apply -R
--check` and `git apply -R`, then `git apply --check` (forward) on the restored tree; all passed
and `git diff` was empty after each. None is committed. Unmutated, the modules run below pass.

**(i) `_unit_tables` returns `[cls_name]`.** `test_declared_facts`, `test_reconcile_at_open`,
`test_prune_at_open`: Ran 79, **FAILED (failures=6, errors=14)**:
- `test_declared_facts.TestTheUnit.test_every_compared_table_of_the_registry` (`BackgroundModel`),
  `…test_a_declared_owner_joins_and_an_undeclared_reference_does_not`,
  `TestThePruneRefusalNamesItsUnit.test_a_prune_that_deletes_outside_its_unit`;
- `test_reconcile_at_open`: `TestKillAndReopen.test_store_of_a_background_model` (four points),
  `…test_store_of_a_second_background_model_beside_the_first`,
  `TestPruningAfterRepair.test_an_interrupted_store` (both), `TestMismatchLeftByTheCheck.test_a_validate`,
  `TestRecordDoesNotExplain.test_part_of_the_recorded_background_model`;
- `test_prune_at_open`: `TestInterruptedPrune` (both, four sub-tests), `TestRecordsStayApart` (two),
  `TestUninterruptedPrune.test_every_shard_identical_and_no_record_left`,
  `TestPruneFailsInTheFactory.test_the_record_stays_and_the_next_open_completes_it`.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 3d87a67..b8cdd61 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -2116,7 +2116,7 @@ class ShardedPool:
         for spec in specs:
             if spec["name"] != cls_name and ShardedPool._declares_owner(spec, cls_name):
                 unit.append(spec["name"])
-        return unit
+        return [cls_name]
 
     @staticmethod
     def _column_naming(spec: dict, cls_name: str) -> str:
```

**(ii) `BackgroundModelValue`'s `owner_column` declaration removed.** The same three modules: Ran
79, **FAILED (failures=7, errors=14)**: the same `test_reconcile_at_open` and `test_prune_at_open`
tests as (i) (value rows outside the unit), and `test_declared_facts`'
`TestTheRecords.test_exactly_the_four_declarations_are_not_default`,
`TestTheFlags.test_the_declared_flags_reach_the_specs`,
`TestTheUnit.test_every_compared_table_of_the_registry` (`BackgroundModel`) and
`TestThePruneRefusalNamesItsUnit.test_a_prune_that_deletes_outside_its_unit`.

```diff
diff --git a/Datastore/SQL/ObjectFactories/BackgroundModel.py b/Datastore/SQL/ObjectFactories/BackgroundModel.py
index ee7d4c8..a6e8bf8 100644
--- a/Datastore/SQL/ObjectFactories/BackgroundModel.py
+++ b/Datastore/SQL/ObjectFactories/BackgroundModel.py
@@ -919,8 +919,6 @@ class sqla_BackgroundModelValue_factory(SQLAFactoryBase):
             "version": False,
             "timestamp": False,
             "stepping": False,
-            # the pool copies and prunes a value row with the model this column names, as a unit
-            "owner_column": "model_serial",
             "columns": [
                 sqla.Column(
                     "model_serial",
```

**(iii) `flag_columns` is empty.** `test_declared_facts`, `test_reconcile_at_open`: Ran 69,
**FAILED (failures=1, errors=2)**:
`test_reconcile_at_open.TestKillAndReopen.test_scalar_get_that_turns_a_flag_on`,
`…test_vectorized_get_that_inserts_and_turns_a_flag_on`, and
`test_declared_facts.TestTheFlags.test_a_declared_flag_is_set_after_a_get`.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 3d87a67..c3fe84d 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -1672,11 +1672,7 @@ class ShardedPool:
         unit = self._unit_tables(cls_name, specs)
         # the flags a get may turn on, and the validated column a validate recomputes, as the
         # factories declare them
-        flag_columns = {
-            spec["name"]: spec["monotone_flags"]
-            for spec in specs
-            if len(spec["monotone_flags"]) > 0
-        }
+        flag_columns = {}
         validated_column = by_name[cls_name]["validated_column"]
         flag_names = list(
             dict.fromkeys(f for spec in specs for f in spec["monotone_flags"])
```

**(iv) `BackgroundModel`'s `revalidate` returns `True` without counting** (deviation 9).
- **(iv-a), writing nothing.** `test_declared_facts`, `test_reconcile_at_open`: Ran 69, **FAILED
  (failures=3, errors=5)**: `test_declared_facts.TestRevalidate.test_the_same_result_flag_and_warning_as_stored_and_with_a_value_row_deleted`
  (three of four sub-tests), `test_reconcile_at_open.TestKillAndReopen.test_validate_of_a_background_model`
  (three points), `TestPruningAfterRepair.test_an_interrupted_validate` (both).
  `TestRecordDoesNotExplain.test_a_recomputed_validated_that_disagrees` passes: the flag is not
  written, so the comparison after the repair still refuses.

  ```diff
  diff --git a/Datastore/SQL/ObjectFactories/BackgroundModel.py b/Datastore/SQL/ObjectFactories/BackgroundModel.py
  index ee7d4c8..f9dcdea 100644
  --- a/Datastore/SQL/ObjectFactories/BackgroundModel.py
  +++ b/Datastore/SQL/ObjectFactories/BackgroundModel.py
  @@ -781,6 +781,7 @@ class sqla_BackgroundModelFactory(SQLAFactoryBase):
       def revalidate(serial, conn, table, tables) -> bool:
           # validate()'s rule for a stored row, with no object: the label the warning names is read
           # from the row (prompts/datastore-generic, prompt 07)
  +        return True
           label = conn.execute(
               sqla.select(table.c.label).filter(table.c.serial == serial)
           ).scalar_one()
  ```
- **(iv-b), writing `validated=True`.** The same modules: Ran 69, **FAILED (failures=3)**:
  `test_declared_facts.TestRevalidate.test_the_same_result_flag_and_warning_as_stored_and_with_a_value_row_deleted`
  (both short sub-tests) and `test_reconcile_at_open.TestRecordDoesNotExplain.test_a_recomputed_validated_that_disagrees`.

  ```diff
  diff --git a/Datastore/SQL/ObjectFactories/BackgroundModel.py b/Datastore/SQL/ObjectFactories/BackgroundModel.py
  index ee7d4c8..87fcf8f 100644
  --- a/Datastore/SQL/ObjectFactories/BackgroundModel.py
  +++ b/Datastore/SQL/ObjectFactories/BackgroundModel.py
  @@ -784,9 +784,10 @@ class sqla_BackgroundModelFactory(SQLAFactoryBase):
           label = conn.execute(
               sqla.select(table.c.label).filter(table.c.serial == serial)
           ).scalar_one()
  -        return sqla_BackgroundModelFactory._validate_row(
  -            serial, label, conn, table, tables
  +        conn.execute(
  +            sqla.update(table).where(table.c.serial == serial).values(validated=True)
           )
  +        return True
   
       @staticmethod
       def owned_serials(obj) -> Optional[List[Optional[int]]]:
  ```

**(v) `owned_serials` on `BackgroundModel` returns `None`.** `test_declared_facts`,
`test_replicated_write`, `test_reconcile_at_open`: Ran 94, **FAILED (failures=2)**:
`test_declared_facts.TestOwnedSerials.test_a_background_model_names_its_value_rows` and
`…test_the_pool_checks_each_answer_through_the_hook`. **No `test_replicated_write` test fails.**
Why: the pool's comparison of owned serials is a backstop no test there reaches.
- Every `TestIdempotentReplicaStores` replay calls the actor directly
  (`self.pool._shards[sid].object_store`), not through the pool, so the pool's check never runs;
  their mismatches are raised by the actor's own store.
- `TestReplicaSerialMismatch` goes through the pool, but for `tolerance`, `redshift` and
  `wavenumber_exit_time`, whose factories own no rows.
- `TestCleanWrite.test_every_copy_is_row_identical` reads the stored value serials from the files,
  which the actors write under the controller's serials whether or not the pool compares them.

With `None` on both sides the comparison passes, and no test builds a replica answer whose model
serial agrees but whose value serials differ. The same was true of `_value_serials` at `f1b8d02`:
the detail text "its value rows have serials" appears in no test before `3f51939`.
`test_the_pool_checks_each_answer_through_the_hook` now pins it.

```diff
diff --git a/Datastore/SQL/ObjectFactories/BackgroundModel.py b/Datastore/SQL/ObjectFactories/BackgroundModel.py
index ee7d4c8..4077782 100644
--- a/Datastore/SQL/ObjectFactories/BackgroundModel.py
+++ b/Datastore/SQL/ObjectFactories/BackgroundModel.py
@@ -792,6 +792,7 @@ class sqla_BackgroundModelFactory(SQLAFactoryBase):
     def owned_serials(obj) -> Optional[List[Optional[int]]]:
         # the serials of the model's BackgroundModelValue rows, which a replicated store's
         # controlling shard assigns; None for an object that carries no value rows
+        return None
         values = getattr(obj, "_values", None)
         if values is None:
             return None
```

**(vi) The prune refusal names the unit in `_unit_tables`' order.** `test_declared_facts`,
`test_prune_at_open`: Ran 38, **FAILED (failures=1)**:
`test_declared_facts.TestThePruneRefusalNamesItsUnit.test_a_prune_that_deletes_outside_its_unit`
(the message read `… deletes only from ['BackgroundModel', 'BackgroundModel_tags',
'BackgroundModelValue']`).

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 3d87a67..b6dec77 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -2205,7 +2205,7 @@ class ShardedPool:
             if len(deleted) > 0 and name not in unit:
                 problems.append(
                     f'"{name}" lost rows {deleted}, and a prune of "{cls_name}" deletes only '
-                    f"from {unit_named}"
+                    f"from {unit}"
                 )
             if len(deleted) > 0:
                 actions.append(
```

**(vii) `build_schema` accepts a non-`Boolean` `monotone_flags` column.** `test_declared_facts`,
`test_schema_builder`: Ran 35, **FAILED (failures=1)**:
`test_declared_facts.TestBuildSchemaRefusesAMisfit.test_monotone_flags_naming_a_column_that_is_not_boolean`
(`ValueError not raised`).

```diff
diff --git a/Datastore/SQL/schema.py b/Datastore/SQL/schema.py
index 18b0dd6..1f324e9 100644
--- a/Datastore/SQL/schema.py
+++ b/Datastore/SQL/schema.py
@@ -236,7 +236,7 @@ def _declared_monotone_flags(
             tab,
             f"and {absent} are not columns of its table",
         )
-    not_boolean = [f for f in flags if not isinstance(tab.c[f].type, sqla.Boolean)]
+    not_boolean = []
     if len(not_boolean) > 0:
         raise _refuse_declaration(
             cls_name,
```

## Observations not acted on

- **The pool's owned-serials check had no test before `3f51939`** (breakage (v)). It is a
  second check behind the actors' own store; `test_declared_facts` now pins it on a stand-in pool.
- **The recompute's refusal at the second comparison** catches a `revalidate` that claims `True`
  without writing (breakage (iv-a)): the shards still differ, and the open refuses. Only a
  `revalidate` that writes a wrong `True` reaches `test_a_recomputed_validated_that_disagrees`.
- **`store_inventory.py`'s `Parent` and `ParentSet` docstrings** name `QuadSource`,
  `GkWKBIntegration`, `GkSource` and `GkSource_parents` as examples (lines 239, 265). They are in
  the audit's list of comments (SI:247-256, 268-270) and outside `read_records`, so untouched here;
  09's guard will meet them.
- **`_find_version_row` builds the whole schema** (`build_schema` over the registry) to get the
  version table, once per open of an existing store. It writes nothing.

None opens an issue.

## State handed to the next prompt

- The layer's contract is `Datastore/contract.py`. `ShardedPool.py` names no registry table but
  the contract's, pinned by `test_declared_facts.TestNoProjectNameInThePool`; the actor and
  `schema.py` name only the contract and the layer's own prepended `version` column.
- `build_schema`'s records carry `owner_column`, `monotone_flags` and `validated_column`; the
  factory hooks `revalidate` and `owned_serials` are on `SQLAFactoryBase`.
  `ShardedPool._unit_tables(cls_name, specs)` is the one unit.
- Importing the five layer modules loads no `MetadataConcepts`, `ComputeTargets`,
  `CosmologyConcepts` or `CosmologyModels` module (`test_layer_registry`); `utilities` and
  `config.defaults` remain.
- Left for 08: `store_inventory.py`'s class list and order, the cosmology reference and its
  module docstring (C1 rows 12, 13). Left for 09: the guard.
- `Datastore` 692, every other suite unchanged. The index holds 107.
