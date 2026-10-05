# Prompt 06 — give the layer its registry

**Campaign:** [`README.md`](README.md) · **Board items:** **G1**–**G3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** 05 landed (`bc8bffc`) and reviewed (`dbc59f9`). `git status` clean. Never in one
checkout while another prompt's change to `ShardedPool.py` or `Datastore.py` is uncommitted.
**Closes:** `[07-drop-actions-list-a-parent-table-before-its-children]` (`datastore-integrity`,
assigned to this prompt on 2026-10-05).
**Narrows:** `[00-the-generic-datastore-layer-hard-codes-project-schema-facts]` (audit C1 rows 9,
10, 11 and 14 go; 07–09 keep the rest).
**Opens:** nothing expected; any issue the work finds.
**Recommended model:** **Opus**. The change touches every constructor of the pool and the actor,
every reader of a closed store, and the drop at open, and it must change nothing a store holds.

**Read first:**

1. [`README.md`](README.md): §0.1, §1, §4, §5 (rules 4, 7–11), §6.2 **U2**, and the user's
   decisions of 2026-10-05 recorded there.
2. [`docs/datastore-generic-audit.md`](../../docs/datastore-generic-audit.md): **C1** (rows 9, 10,
   11 and 14), **C2**, **D**.
3. `Datastore/SQL/Datastore.py`: the imports (`:13-81`), `_factories` (`:89-129`), `_drop_actions`
   and `_drop_order` (`:135-171`), the constructor (`:175-270`), `_open_read_only` (`:301-…`),
   `register_factories` (`:479`), `_build_schema`, `_ensure_tables` and `_drop_actions` (`:538-572`).
4. `Datastore/SQL/ClientPool.py`: `_default_serial_batch_size` (`:9-48`) and `SerialPoolManager`
   (`:142-170`).
5. `Datastore/SQL/ShardedPool.py`: the imports (`:1-36`), the constructor (`:95-330`), the comment
   block "A READ-ONLY POOL" and `_open_read_only` (`:366-500`), and every use of `_factories`
   (`:539`, `:1047`, `:1843`, `:2005`, `:2037`, `:2154`).
6. `Datastore/SQL/schema.py`, all of it.
7. `Datastore/store_reader.py` and `Datastore/store_inventory.py`'s `read_inventory` (`:837-…`).
8. `config/sharding.py`, all of it.
9. The callers: `main.py` (`:79-85`, `:115-126`, `:288-305`, `:317-318`, `:4168-4182`),
   `extract_common.available_run_labels` (`:152-…`), the six `extract_*.py` scripts'
   `ShardedPool(` calls, `RunRegistry/stores.py`'s `fingerprint_store` (`:1440-1500`) and
   `tools/inventory_report.py`'s `_schema_tables` (`:163-167`).
10. `Datastore/tests/standin_pool.py`, `Datastore/tests/schema_description.py`,
    `Datastore/tests/real_store_fixtures.py`, and the test modules named in §2's test plan.

Line numbers are at `dbc59f9`.

---

## 1. What is wanted

The datastore layer (audit §C) imports its client's registries:
- **the registry of storable classes**, `Datastore.SQL.Datastore._factories`, which imports every
  factory and so every compute target. `ShardedPool` imports it as a private name, and so do the
  reader and the inventory (C1 row 10);
- **the drop groups and their order**, `_drop_actions` and `_drop_order`, beside it (C1 row 11);
- **the serial lease batch sizes**, `ClientPool._default_serial_batch_size` (C1 row 9);
- **the replicated set**, which the inventory imports from `config.sharding` (C1 row 14).

Measured on 2026-10-05: importing `ShardedPool`, `Datastore`, `ClientPool` or `store_reader` loads
22 factory modules, 21 `ComputeTargets` modules, 3 `CosmologyConcepts`, 11 `CosmologyModels` and
6 `MetadataConcepts`.

Under D1 and U2 (README §6), the client owns these, and the layer is given them:

- **G1 — the pool and the actor are given the registry.**
  - `ShardedPool` takes the registry as a required keyword `factories`, and gives it to every
    actor. It imports no registry.
  - The `Datastore` actor takes `factories` and registers it. `Datastore.py` imports no factory
    and holds none of the three registries.
  - The pool and the actor take the per-table serial lease batch sizes as `serial_batch_sizes`.
    `ClientPool` holds no table of them; a table they do not name leases 500 at a time, as now.
- **G2 — the reader and the inventory are given the registry.**
  - `open_read_only(primary, factories)` and `read_inventory(primary, factories)`.
  - **The replicated set is read from the store** (the user, 2026-10-05). Every primary records
    the classes its pool replicates, in its `replicated_tables` table, and a pool already refuses
    a store whose record differs from what it is given (`_read_shard_data`). The reader reads that
    record, and the inventory compares across shards exactly the classes the store says it
    replicates. Neither imports `config.sharding`.
- **G3 — the registries move to `config/`, and the drop order is derived.**
  - A new `config/datastore.py` holds `factories`, `drop_groups` (the action → tables mapping) and
    `serial_batch_sizes`, and the helper `tables_to_drop(actions)`. `main.py` passes them in (U2).
  - **The layer is given the tables to drop, not the CLI's action names** (the user,
    2026-10-05). `drop_actions` becomes `drop_tables` on the pool and the actor. `main.py`
    resolves `--drop` through `tables_to_drop`.
  - **The drop order is derived from foreign keys**, by one function in `schema.py`. `_drop_order`
    is deleted, not moved.

**What the derived order changes.** The hand-kept order was right *between* groups (prompt 07 of
`datastore-integrity`) and wrong *within* every group: each group lists its parent table first.
Measured on 2026-10-05, on a `build_full_store` shard holding rows in every dropped table, with
`PRAGMA foreign_keys = ON`:
- the old order violates 15 within-group foreign keys and fails at the first drop on both shards;
- the derived order drops all 25 tables on both shards, and `foreign_key_check` is empty after;
- dropped alone, `1loop-integral`, `quad-source-integral`, `quad-source` and `tk-wkb` now succeed
  on both shards, where the old order failed each of them on shard 0.

Production never turns enforcement on, so no store changes: a read-write stand-in pool dropping
every group from a full store gave byte-identical schemas, row counts and foreign-key checks
before and after. That closes `[07-drop-actions-list-a-parent-table-before-its-children]`.

**What it does not change.** A group dropped alone still fails under enforcement when rows of a
group *not* dropped reference it (`gk-numeric`, `gk-wkb`, `gk-source`, `gk-source-policy`,
`gk-source-policy-records`, `tk-numeric`). No order can fix that, and in production, with
enforcement off, such a drop leaves those rows naming rows that are gone. That is
`[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]`, opened when this prompt was
written. **It is not 06's to fix**: deciding whether a drop cascades or refuses changes what
`--drop` does.

**Nothing the code writes changes** (README §5 rule 9). Measured on 2026-10-05 with a rough version
of this change:
- the fingerprint of `build_full_store`, with `taken` removed, was byte-identical (digest
  `6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759`, problems `{}`);
- so were `build_real_store`'s inventory records and problems;
- `main.py --inventory` gave byte-identical stdout and stderr, and `store fingerprint` the same
  digest;
- every primary the current code writes records exactly `config.sharding.replicated_tables`, in
  order (`build_real_store`, `build_full_store`, a store the real `ShardedPool` created on stand-in
  shards).

---

## 2. What to change

**G3 — `config/datastore.py` (new).**
- **`factories`**: `Datastore.py`'s `_factories`, moved with the factory imports it needs. Same
  keys, same order, same comment.
- **`drop_groups`**: `Datastore.py`'s `_drop_actions`, moved with each group's tables unchanged,
  **with its keys in the order of `main.py`'s `allowed_drop_actions`** (`:115-126`). That order is
  what `--help` and argparse's invalid-choice message print (measured: `list()` of the old dict
  order changes both). The order of the tables within a group no longer matters, and the module
  says so.
- **`serial_batch_sizes`**: `ClientPool.py`'s `_default_serial_batch_size`, moved unchanged,
  comment included.
- **`tables_to_drop(actions) -> List[str]`**: the tables of the named groups, each once, in the
  order the groups and their lists give. An unknown action raises `ValueError` naming it.
- **`_drop_order` is deleted.** The module docstring says where the order now comes from
  (`schema.drop_order`), and why the groups are a client fact: they are the CLI's vocabulary.
- Importing it must not import `ray` beyond what the factories already import. Measured: no import
  cycle; `import config.datastore` works first in a fresh interpreter.

**G3 — `Datastore/SQL/schema.py`.**
- **`drop_order(tables) -> List[sqla.Table]`**: the given tables in an order in which each can be
  dropped while the others are present. A table that declares a foreign key into another of the
  given tables comes before it. Derived from the tables' foreign keys alone (for instance, the
  reverse of `sqlalchemy.schema.sort_tables`). A cycle among the given tables raises: measured,
  there is none, for any group or for all together.
- The module docstring names it. It still imports only `sqlalchemy`, and names no project table.

**G1, G3 — `Datastore/SQL/Datastore.py` (the actor).**
- Delete the factory imports (`:14-76`, keeping `ObjectFactories.base`'s `SQLAFactoryBase` for
  `_FactoryMappingType`), `_factories`, `_drop_actions` and `_drop_order`.
- The constructor takes `factories` (required) and `serial_batch_sizes` (optional). It registers
  `factories` with `register_factories`, and passes `serial_batch_sizes` to `SerialPoolManager`.
- `drop_actions` becomes `drop_tables: Optional[List[str]]`, in the constructor and in
  `_open_read_only`'s refusal. That refusal's text becomes `drop_tables=[…]`, with the detail
  "dropping a table is a write".
- `_drop_actions` becomes `_drop_tables(tables)`:
  - a name the actor's registry does not declare as a table raises `RuntimeError` naming it,
    before anything is dropped;
  - the declared tables are dropped in `drop_order`'s order;
  - a table the file lacks is skipped, as now (`has_table`; B10, U1);
  - it prints one line per table dropped (`Datastore: dropping table "<name>"`), where it printed
    one per action;
  - the inspector is regenerated after, as now.
- Nothing else in the actor changes.

**G1 — `Datastore/SQL/ClientPool.py`.**
- Delete `_default_serial_batch_size`.
- `SerialPoolManager` takes `batch_sizes: Optional[Mapping[str, int]] = None`, and a table it does
  not name leases `500` at a time (`ClientPool`'s own default, unchanged).

**G1 — `Datastore/SQL/ShardedPool.py`.**
- Delete the `_factories` import (`:14`).
- **The constructor** takes `factories` as a **required keyword-only** argument, and
  `serial_batch_sizes` as an optional one, after `read_only`. Measured: no caller passes an
  argument by position. It keeps a copy of each.
- `drop_actions` becomes `drop_tables`. The docstring says what each of the three is.
- Every `_factories` becomes `self._factories` (`:539`, `:1047`, `:1843`, `:2005`, `:2037`,
  `:2154`). `_recompute_validated`'s `self._factories["BackgroundModel"]` keeps the name: that is
  C1 row 6, prompt 07's.
- Both actor constructions pass `factories=` and `serial_batch_sizes=`, and the read-write one
  passes `drop_tables=`.
- **`factories`, a read-only property**, returning a copy. `extract_common` reads it.
- **An undeclared drop table.**
  - **Read-only:** step 1 is unchanged in place. Any non-empty `drop_tables` is refused with
    `ReadOnlyWrite` ("drop_tables=[…]", "dropping a table is a write"), whatever the names.
  - **Read-write:** a name the registry does not declare as a table is refused with
    `RuntimeError`, before the primary is created or opened. One "Cannot open sharded datastore …"
    prefix, the names, and "Nothing was opened".
- `_open_read_only` calls `open_read_only(self._primary_file, self._factories)`. Its comment's
  step 1 says `drop_tables`.

**G2 — `Datastore/store_reader.py`.**
- Delete the `_factories` import (`:42`). `open_read_only(primary, factories)` builds the schema
  from `factories`.
- **`ReadOnlyStore.replicated_tables`**: a tuple of the class names the primary's
  `replicated_tables` table records, in serial order.
  - It is read `mode=ro` after the shard and journal checks and before `build_schema`.
  - A primary whose `replicated_tables` cannot be read is refused through `_refuse` ("Cannot read
    sharded datastore …"), naming the table and the error.
  - So is one that names a class `factories` does not declare.
  - An empty record is not refused: it is what the store says, and the inventory then reports what
    it finds.
- **The read-only pool** meets this refusal at its step 2, before its own primary check at step 3.
  So a primary without `replicated_tables` now meets the reader's `RuntimeError` there, where
  after 05 it met the pool's `StoreSchemaMismatch`. No test pinned either. The comment block "A
  READ-ONLY POOL" says so at step 2.
- The module docstring and `open_read_only`'s refusal list say all of this.

**G2 — `Datastore/store_inventory.py`.**
- `read_inventory(primary, factories)`. It imports neither `_factories` nor `config.sharding`.
- The replicated set is `store.replicated_tables`.
- The module docstring says that which classes are replicated is what the store's primary records.

**The callers** (each a plumbing change; behaviour unchanged):
- **`main.py`**:
  - import `drop_groups`, `factories`, `serial_batch_sizes` and `tables_to_drop` from
    `config.datastore`;
  - `allowed_drop_actions = list(drop_groups)`;
  - `drop_tables = tables_to_drop(drop_actions)`, after `drop_actions` is formed (`:318`);
  - the `ShardedPool(` call passes `drop_tables=`, `factories=` and `serial_batch_sizes=`;
  - `--inventory` calls `read_inventory(args.database, factories)`.
- **The six `extract_*.py` scripts**: each `ShardedPool(` call passes `factories=factories`, imported
  from `config.datastore`. These are `extract_GkSource_data.py:931`, `extract_GkWKB_data.py:504`,
  `extract_Gk_data.py:452`, `extract_QuadSourceIntegral_data.py:1293`, `extract_TkWKB_data.py:492`
  and `extract_tensor_source_data.py:420`. Each is a read-only pool, so none passes batch sizes or
  drop tables.
- **`extract_common.available_run_labels`**: `read_inventory(pool.primary, pool.factories)`, and its
  docstring.
- **`RunRegistry/stores.py`'s `fingerprint_store`**: imports `factories` from `config.datastore`
  inside the function, beside its import of `read_inventory`, so that `import RunRegistry` still
  loads neither ray nor sqlalchemy (measured). It passes the registry, and its docstring step 3
  says so.
- **`tools/inventory_report.py`'s `_schema_tables`**: builds from `config.datastore.factories`. Without
  this, `main.py --inventory` and the report tests fail (measured).

**The tests.** Measured on 2026-10-05 at `dbc59f9`, with the code change alone and no test edited:
- Datastore ran 197, with 93 errors;
- RunRegistry ran 199, with 3 errors;
- ComputeTargets ran 638, with 9 errors.

Almost all of them are import errors (`_factories`, `_drop_actions`) and the `TypeError` of a pool
or reader built without the registry. README §5 rule 7 applies to every test not named here.

- **Test helpers** (not tests; each change is the one named):
  - `Datastore/tests/standin_pool.py`: `open_pool` and `open_pool_output` default `factories` to
    `config.datastore.factories` (`kwargs.setdefault`). They must default it:
    `docs/a3-v2-readiness/reader_writes_probe.py`, which `test_read_only_pool` imports, calls
    `open_pool` and is not edited (§6).
  - `Datastore/tests/schema_description.py:190`: `datastore_module._factories` becomes
    `config.datastore.factories`. An import grep does not find it, because it is an attribute.
  - `Datastore/tests/real_store_fixtures.py`: its `_factories` import comes from
    `config.datastore`.
- **Import only.** `from Datastore.SQL.Datastore import _factories` becomes `from config.datastore
  import factories as _factories`, and nothing else changes, in:
  - `Datastore/tests/`: `test_exact_identity_lookups`, `test_factory_select_columns`,
    `test_integral_read_batch`, `test_inventory_retired`, `test_prune_at_open`,
    `test_read_path_mechanics`, `test_schema_builder` and `test_value_reads_parent_key`;
  - `ComputeTargets/tests/`: `test_gksource_parent_set_main` and
    `test_quadsource_integral_parent_main`.
- **The registry passed to `open_read_only` or `read_inventory`**, with its import, at each call. A
  call inside a code string run in a child interpreter gets the import in the string. Nothing else
  changes.

  | Module | Calls |
  |---|---|
  | `Datastore/tests/test_store_schema` | 2 |
  | `Datastore/tests/test_store_inventory` | 3, and the child-interpreter string at `:1044` |
  | `Datastore/tests/test_store_reader` | 10, and the `_CHILD` string |
  | `Datastore/tests/test_absolute_shard_record_refused` | 1 |
  | `Datastore/tests/test_gksource_parent_set` | 1 |
  | `Datastore/tests/test_inventory_report` | 2 |
  | `Datastore/tests/test_inventory_report_labels` | 1 |
  | `Datastore/tests/test_quadsource_integral_parent_key` | 1 |
  | `Datastore/tests/test_quadsource_parent_key` | 1 |
  | `Datastore/tests/test_quadsource_policy_key` | 1 |
  | `RunRegistry/tests/test_store_fingerprint` | 8 |
  | `RunRegistry/tests/test_store_retire` | 1 |
  | `ComputeTargets/tests/test_qcd_cosmology_inventory_record` | 2 |

- **The registry passed to a pool or an actor:**
  - `test_read_only_pool`'s `open()` passes `factories=`;
  - `test_version_row_at_open`'s `open()` and `bare_actor`'s direct
    `Datastore.options(…).remote(…)` (`:491`) each pass `factories=`, and its `_factories` import
    comes from `config.datastore`.
- **`drop_actions` → `drop_tables`** (named, with the reason):
  - `test_read_only_pool.TestOtherWritesRaiseReadOnlyWrite`, the two methods that pass
    `{"drop_actions": ["gk-source-policy"]}`: it becomes `{"drop_tables": ["GkSourcePolicyData"]}`,
    and the asserted text becomes `"drop_tables"`. The method names stay;
  - `test_reconcile_at_open.TestPruningAfterRepair` (`:1106`): `drop_actions=[…]` becomes
    `drop_tables=tables_to_drop([…])`, with the same two groups;
  - `ComputeTargets/tests/test_reader_read_only.TestTheSixReaders.test_each_opens_its_pool_read_only`:
    `assertNotIn("drop_actions", keywords)` becomes `"drop_tables"`. It passes unchanged, but
    would then assert nothing.
- **`test_inventory_consumers.TestRunLabels`**: the fake pool of its three tests,
  `test_the_run_labels_from_the_structured_inventory`, `test_a_problem_in_store_tag_is_a_refusal`
  and `test_a_problem_elsewhere_does_not_matter`, becomes
  `SimpleNamespace(primary=…, factories=factories)`.
- **`test_wkb_numeric_parent_key.TestDropOrder`.** Its subject, `_drop_order`, is gone. Each test
  keeps its purpose against `drop_order`:
  - the `_drop` helper calls the actor's `_drop_tables` with `tables_to_drop(actions)`, and its fake
    gains `_my_name`;
  - `_OLD_DROP_ORDER` is deleted, and the module's imports come from `config.datastore`
    (`drop_groups`) and `schema` (`drop_order`). The module docstring's item 7 is corrected.

  | Method | Fate |
  |---|---|
  | `test_the_corrected_order_drops_both` | **converted**: `test_the_derived_order_drops_both`. Both orders of the two actions, as sub-tests, drop both tables under enforcement |
  | `test_the_old_order_is_refused_by_enforcement` | **converted**: `test_an_unsorted_order_is_refused_by_enforcement`. With the actor module's `drop_order` patched to the identity, `["tk-wkb", "tk-numeric"]` raises `IntegrityError`. It pins that the actor sorts |
  | `test_tk_wkb_drops_before_tk_numeric_and_gk_wkb_before_gk_numeric` | **converted**: the same, at table level, through `drop_order`. `TkWKBIntegration` comes before `TkNumericIntegration`, and `GkWKBIntegration` before `GkNumericIntegration` |
  | `test_every_declared_foreign_key_between_actions_is_respected` | **converted**: `test_every_declared_foreign_key_between_dropped_tables_is_respected`. Checked for each group alone and for all of them: every foreign key among the dropped tables puts the referencing table first, with at least 3 pairs |

- **No test is deleted.**
- **Must pass unchanged:**
  - every other test, including the U1 recovery tests of `test_store_schema`;
  - `test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_store`;
  - every user of `standin_pool`.
- **New tests: `Datastore/tests/test_layer_registry.py`.** The measurement's sketch had 15, all
  passing. The log names each.
  - **`config.datastore`:**
    - `tables_to_drop` gives each table once, and raises `ValueError` naming an unknown action;
    - every table of every group is declared by `factories`.
  - **`schema.drop_order`:**
    - a referencing table comes before what it references;
    - a cycle raises.
  - **The drop on a full store.** Under `PRAGMA foreign_keys = ON`, the real `_drop_tables` drops
    every group from both shards of `build_full_store`, which holds rows in every dropped table.
    It leaves `foreign_key_check` empty. Unsorted, it raises `IntegrityError`.
  - **An undeclared drop table:**
    - read-write, on a missing path and on an existing store, it is refused before anything
      opens: no actor call, nothing created, checksums unchanged;
    - the actor's own `_drop_tables` refuses one before dropping anything;
    - read-only, `ReadOnlyWrite` comes first.
  - **Serial batch sizes:** `SerialPoolManager` uses the sizes it is given and 500 otherwise.
  - **The recorded replicated set:**
    - the reader returns the primary's record in serial order (reversed serials give a reversed
      tuple);
    - a primary without `replicated_tables` is refused by the reader and by `read_inventory`, with
      one "Cannot read sharded datastore" prefix and the files unchanged;
    - so is a record naming an undeclared class;
    - the inventory uses the record: with `tolerance` left out of it, `tolerance` is combined as a
      sharded class.
  - **The pool uses its registry:** a registry without the `OneLoopIntegral` classes builds shards
    without their tables.
  - **The layer imports no client.**
    - In a fresh interpreter, importing `ShardedPool`, `Datastore`, `ClientPool`, `store_reader`
      and `store_inventory` loads none of the following:
      - `config.datastore` or `config.sharding`;
      - any `Datastore.SQL.ObjectFactories` module but the package and `base`;
      - anything under `ComputeTargets`, `CosmologyConcepts` or `CosmologyModels`.
    - `MetadataConcepts` and `utilities` are allowed, and the test says why: `store_tag` is
      prompt 07's (U4).
    - An `ast` check that none of the five files imports `config.datastore`, `config.sharding` or
      `_factories`.
    - 09's guard generalises this; this test pins only what 06 removes.

---

## 3. Verification

1. **Every suite**, with counts before and after, and each difference accounted for:
   - Datastore: 644 + the new tests;
   - RunRegistry: 280;
   - ComputeTargets: 693;
   - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped).
2. **Greps.** Each of these prints nothing; show the output:
   - `grep -rn '_factories\b' Datastore/SQL/ShardedPool.py Datastore/store_reader.py Datastore/store_inventory.py | grep -v 'self\._factories'`;
   - `grep -rnE '^_factories|^_drop_actions|^_drop_order|_default_serial_batch_size' Datastore/ --include='*.py'`;
   - `grep -rn 'config.sharding\|config.datastore' Datastore/SQL/ Datastore/store_reader.py Datastore/store_inventory.py`;
   - `grep -rnE 'from Datastore.SQL.ObjectFactories\.[A-Za-z_]+ import' Datastore/SQL/Datastore.py | grep -v '\.base import'`;
   - `grep -rn 'drop_actions' Datastore/ main.py extract_*.py --include='*.py'`.
3. **The fingerprint**, from a scratch script, not committed. The fingerprint of
   `build_full_store`, with `taken` removed, is byte-identical at this commit and at an export of
   its parent. Give the digest.
4. **`main.py`**, at this commit and at an export of its parent, on a `build_full_store` primary in
   the scratchpad:
   - `--inventory` gives byte-identical stdout and stderr, with exit 0;
   - `--inventory --drop gk-source` exits 2 with the same message;
   - `--drop nonsense` gives byte-identical stderr (the usage line and the choice order).
5. **The drop, on stand-in shards.** A read-write stand-in pool given every group's tables
   (`tables_to_drop(list(drop_groups))`), on a full store, leaves the same schema, row counts and
   `foreign_key_check` at this commit as `drop_actions` naming every group does at the parent.
6. `black --check` is clean on every touched Python file.
7. **Deliberate breakage.** Each mutation must make a named test fail; then restore it.
   - (i) `drop_order` returns its input unchanged. The full-store drop test and
     `test_the_derived_order_drops_both` fail.
   - (ii) `read_inventory` takes its replicated set from the registry's keys, not the store. The
     inventory's recorded-set test fails.
   - (iii) The pool's undeclared-drop-table check is skipped. The read-write refusal tests fail.
   - (iv) `SerialPoolManager` ignores `batch_sizes`. The batch-size test fails.
   - (v) The reader stops refusing a recorded class the registry does not declare. Its refusal test
     fails.

   Record each in the log as a diff, exactly as applied, with the failing tests.

---

## 4. Acceptance

1. **G1, G2.**
   - `ShardedPool`, the actor, `open_read_only` and `read_inventory` take the registry.
   - None of `Datastore/SQL/` or the two reader modules imports `_factories`, `config.datastore` or
     `config.sharding`.
   - The inventory's replicated set is the store's record.
2. **G3.**
   - `config/datastore.py` holds `factories`, `drop_groups`, `serial_batch_sizes` and
     `tables_to_drop`.
   - `_drop_order` is gone, and the drop order is `schema.drop_order`'s.
   - The layer receives `drop_tables`.
3. **Nothing the code writes changes**: §3.3–§3.5 hold.
4. Every test of §2's plan has the fate listed, and the log lists each by module, class and method.
   The new tests exist, and the deliberate-breakage record (i)–(v) is in the log.
5. **The records:**
   - `[07-drop-actions-list-a-parent-table-before-its-children]` is closed on the
     `datastore-integrity` board (its §3.3 entry moves to §4, with a dated "Closed (prompt 06 of
     datastore-generic)" line) and its index row deleted;
   - `[00-the-generic-datastore-layer-hard-codes-project-schema-facts]` gets a dated "Narrowed"
     line naming rows 9, 10, 11 and 14, and its index hook says so;
   - the count is corrected;
   - on this board: the §1 row, items G1–G3, §3/§4 and §5.

   All in the same commit, with `black --check` clean.

---

## 5. Stop conditions — stop and ask the user

- A test not named in §2's test plan fails after the change. That is current behaviour moving.
  Never edit it to pass.
- The fingerprint, the `--inventory` output or the drop's result differs from its parent's.
- Any primary the current code writes records a replicated set other than
  `config.sharding.replicated_tables`.
- `drop_order` finds a cycle among any group's tables.
- The change would write, open or touch any store under `var/` (README §3), or change what any
  code writes (README §5 rule 9).
- Removing an import from the layer needs a project name added to it (README §5 rule 11).

---

## 6. What this prompt does not do

- **Files it changes:**
  - new: `config/datastore.py` and `Datastore/tests/test_layer_registry.py`;
  - `Datastore/SQL/Datastore.py`, `Datastore/SQL/ClientPool.py`, `Datastore/SQL/ShardedPool.py`,
    `Datastore/SQL/schema.py`, `Datastore/store_reader.py`, `Datastore/store_inventory.py`;
  - `main.py`, the six `extract_*.py` scripts, `extract_common.py`, `RunRegistry/stores.py` and
    `tools/inventory_report.py`, each only as §2 says;
  - the test helpers and test modules of §2's plan;
  - the log, this board, the `datastore-integrity` board (the closed issue only) and
    `docs/OPEN_ISSUES.md`.

  Nothing else.
- **It does not edit `docs/` scripts.** These scripts import `_factories`, `_drop_actions` or
  `config.sharding`'s set for the inventory, or build a `ShardedPool` without the registry. They
  are records of past measurements, run at the SHA that wrote them (CLAUDE.md rule 6), and will not
  run at this commit:
  - `tolerance-convergence/{inventory,order_audit}.py`;
  - `datastore-integrity-audit/{fk_sweep_probe,tq_fk_probe,replicated_write_fault_probe,replicated_divergence_probe}.py`
    and `lookup_keys/{common,probe_03_column_existence}.py`;
  - `a3-v2-readiness/{readonly_open_probe,version_window_probe}.py` and
    `rehearsal/store_soundness.py`;
  - `source-remediation-verification/{run_quadsource_integrals,analyse_greens_and_source}.py`.

  The live drivers the tests load (both `scoped_pipeline_run.py` and
  `handover/quadsource_atol_sweep.py`) are unaffected (measured), and
  `a3-v2-readiness/reader_writes_probe.py` works through `standin_pool`'s default.
- **It removes no other project knowledge.**
  - `ShardedPool` still imports `MetadataConcepts.store_tag` and names `BackgroundModel` and the
    flag columns (C1 rows 1–8, prompt 07).
  - The inventory still has its class list and cosmology reference (rows 12–13, prompt 08).
  - Comments naming project tables stay unless a line above moves them.
- **It does not change what `--drop` does**: the same tables are dropped. Only their order within
  a group changes, and only an enforcing connection could see that.
  `[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]` and
  `[00-a-drop-action-drops-a-replicated-table-outside-the-in-flight-record]` stay open.
- It does not touch `ProfileAgent`, `SerialPoolBroker`, `replication.py`, `shard_paths.py`, the
  closed-store operations, or any factory.

---

## 7. The log and the board

`logs/06-give-the-layer-its-registry.md`, using README §5.1. In addition, it gives:
- every test edited, converted and added, each by module, class and method, grouped as §2 groups
  them;
- the module lists of §3's import test, before and after;
- the measured messages of the new refusals: an undeclared drop table (pool and actor), a primary
  without `replicated_tables`, and an undeclared recorded class;
- the fingerprint digest of §3.3.

`IMPLEMENTATION_STATE.md`:
- the §1 row for 06, and items G1–G3;
- the narrowed generic-layer issue;
- §5.

The `datastore-integrity` board: the closed issue. `docs/OPEN_ISSUES.md`: its row deleted, the
generic-layer hook narrowed, the count and the date.
