# Prompt 07 — declare the replication facts in the factories

**Campaign:** [`README.md`](README.md) · **Board items:** **F1**–**F4** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** 06 landed (`19f07ee`) and reviewed (`c6bddf8`). `git status` clean. Never in one
checkout while another prompt's change to `ShardedPool.py`, `Datastore.py`, `schema.py` or a
factory is uncommitted.
**Closes:** nothing.
**Narrows:** `[00-the-generic-datastore-layer-hard-codes-project-schema-facts]` (audit C1 rows 1,
2, 4, 5, 6, 7 and 8 go; 08 keeps rows 12 and 13, and 09 adds the guard).
**Opens:** nothing expected; any issue the work finds.
**Recommended model:** **Opus**. The change rewrites the licence of the check at open, the prune
unit and the version row's construction. It must change nothing a store holds, and no repair,
refusal or printed line.

**Read first:**

1. [`README.md`](README.md):
   - §0.1, §1, §4 and §5 (rules 4 and 7–11);
   - §6.2: **U4**, and the user's choice of method for 07, taken on 2026-10-06.
2. [`docs/datastore-generic-audit.md`](../../docs/datastore-generic-audit.md): **C1** (rows 1, 2,
   4–8, and "What is infrastructure"), **C2**, **D**.
3. The board's orchestrator review of 06 (§1), and its "Handed on" list.
4. `Datastore/SQL/ShardedPool.py`:
   - the imports (`:22`, `:35`);
   - the version row: the get in `__init__` (`:342-346`), `_find_version_row` (`:354-393`), the
     read-only `ReadOnlyMiss` (`:486-495`) and `_version_labels_present` (`:608-624`);
   - the comment "THE REPLICATED TABLES, RECONCILED AT OPEN" (`:1021-1060`);
   - `_replicated_table_specs` (`:1257-1294`), `_replicated_differences`' shard-key check
     (`:1439-1457`) and `_describe_differences`' `shard_key` entry (`:1565-1572`);
   - `_plan_replicated_repair` (`:1578-1811`), `_apply_replicated_repair` (`:1830-1887`),
     `_recompute_validated` (`:1889-1951`) and `_print_reconciliation` (`:1953-1993`);
   - the comment "THE PRUNE AT OPEN …" (`:1995-2025`), `_prune_unit` (`:2039-2054`) and its caller
     `_prune_shard` (`:2081-2140`; the refusal at `:2121-2125`);
   - `_value_serials` (`:3259-3265`) and `_store_impl_replicated_table` (`:3267-3305`);
   - the read-only comment in `_assign_shard_keys` (`:3436-3438`).
5. `Datastore/SQL/Datastore.py`: `_refuse_insert`'s docstring (`:269-276`) and `_insert`'s serial
   lease (`:729-738`).
6. `Datastore/SQL/schema.py`, all of it, and `Datastore/SQL/ObjectFactories/base.py`.
7. The factories:
   - `version.py` and `store_tag.py`, all of them;
   - `redshift.py` (`register`, `:17-26`) and `wavenumber.py` (`register`, `:24-33`);
   - `BackgroundModel.py`: `register` (`:159-200`), `validate` (`:729-765`), `inventory_records`
     (`:860-876`) and `sqla_BackgroundModelValue_factory.register` (`:884-…`).
8. `Datastore/store_inventory.py`'s `read_records` (`:432-560`).
9. `Datastore/tests/test_schema_builder.py`, `Datastore/tests/schema_description.py`,
   `Datastore/tests/test_layer_registry.py` (`:591-613`), and the tests named in §2.

Line numbers are at `c6bddf8`.

---

## 1. What is wanted

After 06 the layer is given its registry. It still knows seven facts about the project's tables
(audit C1):

| Row | Fact | Where |
|---|---|---|
| 1 | The version table: named `version`, keyed by `label`, its object a `MetadataConcepts.store_tag`, its serials not leased | `ShardedPool` (the import, the get, `_find_version_row`, `_version_labels_present`, the `ReadOnlyMiss`); the actor's lease; `schema.py`'s foreign key |
| 2 | The tables copied with a parent as a unit | `_plan_replicated_repair`'s hard-coded `BackgroundModel` unit; `_prune_unit` |
| 4 | The serials of an object's child rows, read from its `_values` | `_value_serials` |
| 5 | The flags a get may turn on | `flag_columns = {"redshift": …, "wavenumber": …}` |
| 6 | The `validated` rule, recomputed from stored rows | `_recompute_validated` (`_factories["BackgroundModel"]`, `_StoredModel`, the `label` column); the inventory's `table.c.validated` |
| 7 | The tag convention: `store_tag`, its `label`, the association column `tag_serial` | `_replicated_table_specs`; the inventory's `read_records` |
| 8 | The shard key as "wavenumber" | the comments and messages of `ShardedPool` |

Under D1 and U4 (README §6), and the user's choice of 2026-10-06:

- **F1 — the unit is read from foreign keys, through the one an owned table declares.**
  - A table whose rows belong to another's declares, in its factory's `register()`, the column
    whose foreign key names the owner.
  - A table with a foreign key to the tag table and every other foreign key to one class is that
    class's tag table, with no declaration (U4).
  - One function gives the copy unit of the check at open and the prune unit.
- **F2 — monotone flags are declared** by the factories of the tables that hold them.
- **F3 — the `validated` rule is declared.** The factory names its flag column and recomputes it
  for a stored row through a hook. `_StoredModel` is deleted.
- **F4 — the version and tag tables are the layer's contract (U4)**, named once. The pool builds
  the version object through the version table's factory and stops importing `MetadataConcepts`.
  No project name is left in `ShardedPool.py`'s code, comments or messages, nor in the actor's
  docstring. The serials of an object's owned rows come through a factory hook (row 4). Messages
  name the shard key through `self._ShardKeyType_name` (row 8).

**Measured on 2026-10-06 with a rough version of this change** (orchestrator, at `c6bddf8`):
- **Stores.** The fingerprint of `build_full_store`, with `taken` removed, was byte-identical
  (digest `6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759`, problems `{}`). So
  were `build_real_store`'s fingerprint and both inventories, and `sqlite_master` of every file.
- **The check at open.** Every test of `test_reconcile_at_open`, `test_prune_at_open`,
  `test_version_row_at_open`, `test_replicated_write` and `test_read_only_pool` was run with the
  pool's constructor recorded, with `random` seeded per test. That is 111 tests and 328
  constructions. Every exception message, every printed line, `reconciliation`, `pruned` and the
  version object were byte-identical at both trees.
- **Messages.** Every message whose source text changes was produced by a scenario at both trees,
  16 in all, and each was byte-identical. That needed one ordering rule (§2, the prune refusal).
- **`revalidate` and `validate`** gave the same result, wrote the same flag and printed the same
  warning, on a `build_full_store` shard as stored and with a value row deleted.
- **Imports.** Importing the five layer modules loads no `MetadataConcepts` module, where it loaded
  six.

**What it does not change.**
- Nothing a store holds: no table, column, key or constraint (README §5 rule 9). The DDL is
  unchanged.
- No repair, refusal, printed line or record of the check at open, the prune or the version row,
  for any table today. The unit of every replicated class is what it was (§2, F1).
- What is new:
  - `build_schema` refuses a factory whose declaration does not fit its table (§2), which the
    current registry never does;
  - every schema record of a class with a table carries three more keys. The schema witness is
    replaced for that reason alone (§2, the tests).

---

## 2. What to change

**U4 — `Datastore/contract.py` (new).** The tables and columns the layer owns as its contract,
each named once:
- `VERSION_TABLE = "version"` and `VERSION_LABEL = "label"`;
- `TAG_TABLE = "store_tag"`, `TAG_LABEL = "label"` and `TAG_SERIAL = "tag_serial"`.

It imports nothing outside the standard library, so that any module of the layer, including
`store_inventory` at module scope, can import it. Its docstring says why these are the layer's and
not the project's (U4).

**F4 — `Datastore/SQL/schema.py`.**
- The prepended `version` column's foreign key is `f"{VERSION_TABLE}.serial"`. The DDL is
  unchanged.
- The module docstring says what it now imports (`sqlalchemy` and `Datastore.contract`) and names
  the three keys below.

**F1–F3 — `register()` keys, carried by `build_schema` into each record of a class with a
table.** A record of a class with no table (`register()` returns `None`) is unchanged, so readers
use `.get`. Each key is checked against the table, and a misfit raises `ValueError` naming the
class, the key and what is wrong:

| Key | Meaning | Absent | Refused when |
|---|---|---|---|
| `owner_column` | the column whose one foreign key names the row this table's rows belong to | `None` | not a column; a column with no foreign key or with more than one |
| `monotone_flags` | Boolean columns a get may turn on and nothing turns off | `()` | given as a `str`; a name that is not a column; a column that is not `Boolean` |
| `validated_column` | the Boolean column of the class's validated flag, which the factory's `revalidate` recomputes from stored rows | `None` | not a column; not `Boolean` |

The declarations, and no others:
- `sqla_BackgroundModelValue_factory`: `"owner_column": "model_serial"`;
- `sqla_redshift_factory` and `sqla_wavenumber_factory`: `"monotone_flags": ("source", "response")`;
- `sqla_BackgroundModelFactory`: `"validated_column": "validated"`.

Each carries a one-line comment saying what the pool does with it.

**F3, row 4 — `Datastore/SQL/ObjectFactories/base.py`.** Two hooks, **not abstract**, so that no
factory without them changes:
- `revalidate(serial, conn, table, tables) -> bool` raises `NotImplementedError`. A factory that
  declares `validated_column` implements it: it recomputes the flag of the stored row `serial`
  from stored rows, writes it, and returns it, exactly as its `validate` does for an object.
- `owned_serials(obj) -> Optional[List[Optional[int]]]` returns `None`. It gives the serials of the
  rows `obj` owns, which a replicated store's controlling shard assigns and every replica must
  answer with.

**F3, row 4 — `sqla_BackgroundModelFactory`.**
- The rule moves into one static helper that `validate` and `revalidate` both call:
  `validate(obj, …)` keeps its `obj.available` check and passes `obj.store_id` and `obj.label`;
  `revalidate(serial, …)` reads the label from the row.
- The helper prints the same warning and writes the flag the same way.
- `owned_serials(obj)` returns what `ShardedPool._value_serials` returns now:
  `[getattr(v, "_my_id", None) for v in obj._values]`, or `None` when the object has no `_values`.

**F3 — `store_inventory.read_records`.**
- `validated: bool = False` becomes `validated: Optional[str] = None`, the name of the class's flag
  column. It is selected labelled `"validated"`, so the record is unchanged.
- The seven factories that pass `validated=True` pass `validated="validated"`:
  - `BackgroundModel.py:875`, `GkNumericIntegration.py:656`, `GkSource.py:874`;
  - `GkWKBIntegration.py:986`, `QuadSource.py:566`;
  - `TkNumericIntegration.py:658`, `TkWKBIntegration.py:981`.
- **F4:** the tag lookup uses `TAG_TABLE`, `TAG_LABEL` and `TAG_SERIAL`
  (`context.keys.get("store_tag")`, `key["label"]`, `tag_table.c.tag_serial`, and the "orphan-tag"
  message, whose text is unchanged).
- Nothing else in `store_inventory.py` changes: the class list, the cosmology reference and the
  module docstring are prompt 08's.

**F4 — `Datastore/SQL/Datastore.py` (the actor).**
- `_insert`'s `cls_name != "version"` becomes `VERSION_TABLE`, and its comment says "the version
  table".
- `_refuse_insert`'s docstring names no class.
- Nothing else changes.

**F1–F4 — `Datastore/SQL/ShardedPool.py`.**
- **Imports.** Delete `from MetadataConcepts import store_tag` and the now-unused `DatastoreObject`.
  Import the contract's constants.
- **The version row (U4).**
  - The get in `__init__`, `_version_labels_present` and the read-only `ReadOnlyMiss` use
    `VERSION_TABLE` and `VERSION_LABEL`.
  - `_find_version_row` keeps its raw `mode=ro` read of every shard, its duplicate check and its
    disagreement check, with the constants in its SQL. It then builds the object it returns:
    - by `self._factories[VERSION_TABLE].build(payload={VERSION_LABEL: label}, conn=…,
      table=…, inserter=…, tables=…, inserters={})`, as an actor's get calls `build`;
    - on a `mode=ro` SQLAlchemy connection (`store_reader.read_only_url`, imported in the
      function) to the lowest-serial shard that holds the row;
    - with an inserter that raises `RuntimeError`, which is never reached because the row exists;
    - and disposing of the engine after.

    Nothing else is added: no new refusal. The object must be what it is now, a `store_tag` whose
    `vars()` are exactly `_my_id`, `_label` and `_deserialized` (True).
- **The specs.** `_replicated_table_specs` excludes `TAG_TABLE` where it excluded `"store_tag"`.
  Each spec also carries the record's `owner_column`, `monotone_flags` and `validated_column`.
  `_plan_replicated_repair` keeps its signature: tests call it.
- **F1 — `_unit_tables(cls_name, specs) -> List[str]`, replacing `_prune_unit`.**
  - It returns the class, then the compared tables that are its tag tables, then the compared
    tables whose declared `owner_column` references it, each group in the specs' order.
  - A tag table of a class has a foreign key to `TAG_TABLE`, and every other foreign key to that
    class.
  - For `BackgroundModel` it gives `[BackgroundModel, BackgroundModel_tags,
    BackgroundModelValue]`. For every other compared table it gives the table alone (measured).
  - It is the unit of the repair licence and of the prune. A helper gives a member's column naming
    the class: its `owner_column`, or the tag table's foreign key to the class.
- **F1 — the licence.**
  - `unit = self._unit_tables(cls_name, specs)` for every class.
  - The as-unit check runs for every class whose unit has more than one table, where it ran for
    `BackgroundModel`. Its refusals name `cls_name`.
  - Its two messages keep their text: "…holds its model row, so this is not part of a {cls_name}
    it lacks" and "…already holds part of this model (…), and a {cls_name} is copied only as a
    unit". "Model row" and "value rows" are not registry names; they stay so that no message
    changes.
  - The variables `models` and `model_column` are renamed, which changes no text.
- **F2.**
  - `flag_columns` comes from the specs' `monotone_flags`.
  - The refusal "only a source or response flag after a get, or validated after a validate" is
    formed from the declared flags (their union over the specs, in order, joined by " or ") and
    the declared validated columns. The text is the same today.
- **F3.**
  - The `differs` branch licenses a recompute when `name == cls_name`, the class declares a
    `validated_column`, and `differing == [that column]`, together with the existing conditions.
    It was `name == "BackgroundModel"` and `["validated"]`.
  - `_recompute_validated` is generic over the record's class:
    - it calls the factory's `revalidate(serial, conn, table, tables)` inside the same savepoint;
    - it reads the stored flag from the declared column;
    - its refusal says `f"{column}=True"`;
    - `"class_name"` is `cls_name`.
  - `_StoredModel` is deleted. The action record keeps `"action": "validated recomputed"` and
    `"validated": True`: that is the layer's vocabulary, and tests read it.
- **The prune.** `_prune_shard` uses `_unit_tables`. Its refusal "…deletes only from [...]" names
  the unit as it always has: the class, then the rest of its unit **in the specs' order**. That is
  `[BackgroundModel, BackgroundModelValue, BackgroundModel_tags]`, not `_unit_tables`' order.
  Measured: a single order cannot keep both this text and the as-unit refusals' order.
- **Row 4.** `_value_serials` is deleted. `_store_impl_replicated_table` calls
  `self._factories[cls_name].owned_serials(…)` on the controller's object and on each answer. The
  `ReplicationMismatch` detail is unchanged.
- **Row 8.** The `shard_key` difference's detail is
  `f"the primary assigns {self._ShardKeyType_name} serial …"`, with the same text. Comments say
  "the shard key" or "the shard-key class".
- **Comments and docstrings.** No registry table, project column (`source`, `response`,
  `model_serial`) or project package is named. The contract's constants are named through the
  constant, or as "the version table" and "the tag table".
- **Nothing else in the pool changes.** In particular, nothing in the closed-store operations
  changes; their "source" is a source store.

**The tests.** Measured on 2026-10-06 with the code change alone and no test edited:
- Datastore ran 664 with 78 failures, all in the two witness tests of `test_schema_builder` (38
  per-class sub-tests each, plus each test's whole-witness assertion);
- every other suite passed. ComputeTargets also showed its known wall-clock flake.

README §5 rule 7 applies to every test not named here.

- **The schema witness** (`test_schema_builder`, both
  `TestSchemaIsUnchanged.test_build_schema_reproduces_the_witness` and
  `.test_actor_build_schema_reproduces_the_witness`).
  - A new witness, `Datastore/tests/data/schema_at_datastore-generic-07.json`, is captured with
    `schema_description.py`'s `__main__`. `WITNESS` names it. The old witness file stays, as a
    record (CLAUDE.md rule 6).
  - The module docstring's list of witnesses gains it, with the reason.
  - The log gives the diff between the two witnesses. Measured: the only difference is the three
    keys on all 38 records, with every table description, the class order and the metadata
    identical. Non-default values appear on exactly `redshift`, `wavenumber`, `BackgroundModel`
    and `BackgroundModelValue`.
- **`test_layer_registry`:**
  - `FORBIDDEN_PACKAGES` gains `"MetadataConcepts"`;
  - the comment above it says that `utilities` is the one package still allowed, and why
    `MetadataConcepts` is no longer loaded (06's review handed this on).
- **No test is deleted, and no other test is edited.** Everything else passes unchanged,
  including:
  - `test_reconcile_at_open`, `test_prune_at_open`, `test_version_row_at_open`,
    `test_replicated_write` and `test_read_only_pool`;
  - `test_store_inventory` and `test_layer_registry`'s other tests;
  - the call of `_plan_replicated_repair` at `test_version_row_at_open.py:605` and `:620`.
- **New tests: `Datastore/tests/test_declared_facts.py`.** The measurement's sketch had 11, all
  passing. The log names each.
  - **The records.**
    - On the real registry, every record of a class with a table carries the three keys.
    - Exactly the four declarations above are non-default.
    - A record with no table is unchanged.
  - **The refusals of `build_schema`**, one test each, asserting the class and key named:
    - an `owner_column` that is absent, has no foreign key, or has two;
    - `monotone_flags` given as a `str`, naming an absent column, or naming a non-`Boolean` one;
    - a `validated_column` that is absent or not `Boolean`.
  - **The unit.**
    - `_unit_tables` for every compared table of the real registry: `BackgroundModel`'s three, in
      order, and every other table alone.
    - On a hand-built registry, a table declaring an owner joins its owner's unit. A table with a
      foreign key to the class and no declaration does not.
  - **The flags** reach the specs, and `flag_columns`, from the declarations.
  - **`revalidate` against `validate`**, on a `build_full_store` shard copy, as stored and with one
    value row deleted, each in a rolled-back savepoint. The same return value, flag written and
    printed warning. The base `revalidate` raises `NotImplementedError`.
  - **`owned_serials`**: the base gives `None`; `BackgroundModel`'s gives its value serials.
  - **The version object** built by `_find_version_row`, read-write and read-only: built by the
    registry's version factory (a recording subclass), with `vars()` exactly `_my_id`, `_label`
    and `_deserialized`.
  - **The prune refusal's text**: a prune that deletes from a table outside its unit (an
    `AFTER DELETE` trigger, as M7 built it) names the unit in the specs' order.
  - **No project name in `ShardedPool.py`.**
    - An `ast` check: no string or name in the file is a table name of the real registry other
      than the contract's, and no import names a project package.
    - A text check: the comments and docstrings name none either.
    - "model" and "value rows" are allowed, and the test says why.

---

## 3. Verification

1. **Every suite**, with counts before and after, each difference accounted for:
   - Datastore: 664 + the new tests;
   - RunRegistry: 280; ComputeTargets: 693;
   - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped).
2. **Greps.** Show each one's output:
   - `grep -nE 'MetadataConcepts|store_tag\b|BackgroundModel|redshift|wavenumber|_StoredModel|_value_serials|_prune_unit' Datastore/SQL/ShardedPool.py`
     prints nothing (measured on the rough change);
   - `grep -nE '"(version|store_tag|label|tag_serial)"' Datastore/SQL/ShardedPool.py Datastore/SQL/Datastore.py Datastore/SQL/schema.py Datastore/store_inventory.py | grep -v 'label("validated")'`;
     at `c6bddf8` it prints 14 lines. Afterwards exactly five may remain, and the log lists each
     with its reason:
     - the layer's own prepended `version` column and its registration key (`schema.py:98`,
       `:102`, and `Datastore.py:745`'s payload key);
     - the inventory's class list (`store_inventory.py:114-115`), which is prompt 08's;
   - `grep -nF 'table.c.validated' Datastore/store_inventory.py` and
     `grep -nF 'validated=True' Datastore/SQL/ObjectFactories/*.py`. Each prints nothing. At
     `c6bddf8` the second prints the seven lines of §2. The factories' own queries of their
     `validated` columns stay, and neither grep matches them.
3. **The fingerprint**, from a scratch script, not committed. The fingerprint of
   `build_full_store`, with `taken` removed, is byte-identical at this commit and at an export of
   its parent, and so is `build_real_store`'s inventory. Give the digest.
4. **The check at open, recorded.** At this commit and at the export, run the five modules of §1
   with the pool's constructor recorded, as the measurement did:
   - `random` seeded per test, and `PYTHONHASHSEED=0`;
   - paths and datetimes normalised;
   - every exception message, printed line, `reconciliation`, `pruned` and the version object
     recorded.

   The two records are byte-identical. `orchestrator/measure-07/scripts/m2_open_behaviour.py` is
   the measurement's; the agent may use its own.
5. **The messages.** Each message whose source text changed is produced at both trees and is
   byte-identical:
   - the as-unit refusals, the flag-or-validated refusal and the recompute refusal;
   - the `shard_key` detail and the `ReplicationMismatch` detail;
   - the inventory's "orphan-tag" problem;
   - the version `ReadOnlyMiss`;
   - the prune refusal.
6. **Imports.** Importing each of the five layer modules in a fresh interpreter loads no
   `MetadataConcepts` module. Give the lists before and after.
7. `black --check` is clean on every touched Python file.
8. **Deliberate breakage.** Each mutation must make a named test fail; then restore it.
   - (i) `_unit_tables` returns `[cls_name]`. The new unit test fails, and so do
     `test_reconcile_at_open`'s tests of a `BackgroundModel` copied as a unit.
   - (ii) `BackgroundModelValue`'s `owner_column` declaration is removed. The new records and unit
     tests fail, and so does at least one `test_reconcile_at_open` or `test_prune_at_open` test of
     a value row.
   - (iii) `flag_columns` is empty. `test_reconcile_at_open`'s flag-repair tests fail.
   - (iv) `BackgroundModel`'s `revalidate` returns `True` without counting. The new `revalidate`
     test fails, and so does `test_reconcile_at_open`'s recompute that disagrees.
   - (v) `owned_serials` on `BackgroundModel` returns `None`. The new `owned_serials` test fails.
     Report whether any `test_replicated_write` test fails too. If none does, the log says so, and
     why.
   - (vi) The prune refusal names the unit in `_unit_tables`' order. The new prune-text test
     fails.
   - (vii) `build_schema` accepts a non-`Boolean` `monotone_flags` column. Its refusal test fails.

   Record each in the log as a diff, exactly as applied, with the failing tests.

---

## 4. Acceptance

1. **F1–F3.**
   - `owner_column`, `monotone_flags` and `validated_column` are declared by exactly the four
     factories of §2, carried by `build_schema`, and refused on a misfit.
   - `revalidate` and `owned_serials` are hooks on the base, implemented by `BackgroundModel`.
   - `_unit_tables` replaces `_prune_unit` and the hard-coded unit.
   - `_StoredModel` and `_value_serials` are gone.
2. **F4.**
   - `Datastore/contract.py` names the version and tag tables and their columns once.
   - `ShardedPool` imports no `MetadataConcepts` and builds the version object through the
     version factory.
   - No project name is in `ShardedPool.py`'s code, comments or messages.
   - The inventory's tag and validated reads name no project column.
3. **Nothing changes**: §3.3–§3.5 hold.
4. Every test of §2's plan has the fate listed, and the log lists each by module, class and method.
   The new tests exist, and the breakage record (i)–(vii) is in the log.
5. **The records:**
   - `[00-the-generic-datastore-layer-hard-codes-project-schema-facts]` gets a dated "Narrowed"
     line naming rows 1, 2, 4–8 and the two `register()` hooks and three keys. Its index hook
     says so. The count is unchanged unless an issue is opened.
   - On this board: the §1 row, items F1–F4, §3 and §5.
   - README §4's "07 ships" names the keys, the hooks and `Datastore/contract.py` as shipped.

   All in the same commit, with `black --check` clean.

---

## 5. Stop conditions — stop and ask the user

- A test not named in §2's plan fails after the change. That is current behaviour moving; never
  edit it to pass.
- The fingerprint, the recorded check at open or any message of §3.5 differs from its parent's.
- `_unit_tables` gives any compared table of the real registry a unit other than the old one.
- The change would write, open or touch any store under `var/` (README §3), or change what any
  code writes (README §5 rule 9).
- Removing a name from the layer needs a project name added to it (README §5 rule 11), or a
  generic form cannot keep a message's text.

---

## 6. What this prompt does not do

- **Files it changes:**
  - new: `Datastore/contract.py`, `Datastore/tests/test_declared_facts.py` and
    `Datastore/tests/data/schema_at_datastore-generic-07.json`;
  - `Datastore/SQL/ShardedPool.py`, `Datastore/SQL/Datastore.py`, `Datastore/SQL/schema.py`,
    `Datastore/SQL/ObjectFactories/base.py` and `Datastore/store_inventory.py`, each only as §2
    says;
  - the factories `BackgroundModel.py`, `redshift.py` and `wavenumber.py`, for the declarations
    and hooks;
  - `GkNumericIntegration.py`, `GkSource.py`, `GkWKBIntegration.py`, `QuadSource.py`,
    `TkNumericIntegration.py` and `TkWKBIntegration.py`, for the one `validated=` argument each;
  - `Datastore/tests/test_schema_builder.py` and `Datastore/tests/test_layer_registry.py`, as §2
    says;
  - the log, this board, the README (§4) and `docs/OPEN_ISSUES.md`.

  Nothing else.
- **It does not touch `store_inventory.py`'s class list, cosmology reference or docstring**
  (prompt 08), nor add 09's guard.
- **It does not edit `docs/` scripts.** None imports a name this prompt removes (measured).
- **It changes no factory's behaviour**, only what it declares and the two hooks.
  `BackgroundModel`'s `validate` keeps its rule, its warning and its write.
- It does not touch `ProfileAgent`, `SerialPoolBroker`, `replication.py`, `shard_paths.py`,
  `store_reader.py`, `ClientPool.py`, `config/`, or the closed-store operations.

---

## 7. The log and the board

`logs/07-declare-the-replication-facts-in-the-factories.md`, using README §5.1. In addition, it
gives:
- every test edited and added, by module, class and method;
- the diff between the two schema witnesses;
- `build_schema`'s new refusal messages, as measured;
- the import lists before and after;
- the fingerprint digest of §3.3, and the size of the record of §3.4 (tests, constructions,
  printed lines);
- every project name left in the five layer files after the change, with its classification:
  contract, layer vocabulary (`validated`, "model row"), or generic English.

`IMPLEMENTATION_STATE.md`:
- the §1 row for 07, and items F1–F4;
- the narrowed generic-layer issue;
- §5.

`docs/OPEN_ISSUES.md`: the generic-layer hook narrowed, and the date.
