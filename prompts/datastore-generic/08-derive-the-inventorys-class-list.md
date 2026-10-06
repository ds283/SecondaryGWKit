# Prompt 08 — derive the inventory's class list

**Campaign:** [`README.md`](README.md) · **Board items:** **I1**–**I2** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** 07 landed (`3f51939`) and reviewed (`5fe0ae6`). `git status` clean. Never in one
checkout while another prompt's change to `store_inventory.py`, a factory or
`tools/inventory_report.py` is uncommitted.
**Closes:** nothing.
**Narrows:** `[00-the-generic-datastore-layer-hard-codes-project-schema-facts]` (audit C1 rows 12
and 13 go; with them every row of C1 is gone, and 09 adds the guard).
**Opens:** one issue, `[08-resolve-shows-an-absent-nullable-parent-as-unresolved]` (§5 below), and
any other the work finds.
**Recommended model:** **Sonnet**. The change is wide (21 factories, the inventory and its report)
but mechanical. Its test plan and every output it must keep were measured with a rough version of
the change, and the fingerprint, the inventory and the report are pinned byte for byte.

**Read first:**

1. [`README.md`](README.md):
   - §0.1, §1, §4 and §5 (rules 4, 7, 9 and 11);
   - §6.2: **U2**, and the user's choices of method for 08, taken on 2026-10-06.
2. [`docs/datastore-generic-audit.md`](../../docs/datastore-generic-audit.md): **C1** (rows 12 and
   13, and the comment list for `SI`), **C2**, **D**.
3. The board's orchestrator review of 07 (§1), and its "Handed on" list.
4. `Datastore/store_inventory.py`, all of it. In particular:
   - the module docstring (`:1-73`);
   - `COMPUTE_TARGETS` and `INVENTORY_CLASSES` (`:100-128`), `COSMOLOGY` (`:130-132`) and
     `cosmology_classes` (`:138-142`);
   - `Parent` (`:237-261`), `ParentSet` (`:263-278`), `ShardRead` (`:297-313`) and
     `ClassInventory` (`:316-342`);
   - `StoreInventory.resolve` (`:373-410`);
   - `read_records`: its docstring (`:448-461`), the dependency check (`:475-491`), `_reference`
     (`:620-634`) and `types` (`:636`);
   - `_combine` (`:754-840`) and `read_inventory` (`:843-900`).
5. `Datastore/SQL/ObjectFactories/base.py`, and every factory's `inventory_records`: the 21 listed
   in §6, two of them in `wavenumber.py`.
6. `tools/inventory_report.py`: its imports (`:82-88`), `_Renderer.__init__` (`:173-190`),
   `parent_class` (`:232-236`) and `labelled` (`:375-389`).
7. The tests named in §2, at the lines given there.

Line numbers are at `5fe0ae6`.

---

## 1. What is wanted

After 07 the layer is given its registry and the factories declare the replication facts. Two
facts about the project's tables are left, both in `store_inventory.py` (audit C1):

| Row | Fact | Where |
|---|---|---|
| 12 | The inventory's class list and dependency order | `INVENTORY_CLASSES`, `COMPUTE_TARGETS` |
| 13 | The polymorphic cosmology reference: `cosmology_type` names `LambdaCDM` or `QCD_Cosmology`, from `CosmologyModels.model_ids` | `COSMOLOGY`, `cosmology_classes`; `resolve`, `read_records`; `tools/inventory_report.py` |

The module's docstrings and comments also name project tables, as examples (audit C1's comment
list, `SI:11-21, 59-68, 103-105, 133-134, 142, 247-256, 268-270`).

Under D1 and the user's three choices of 2026-10-06 (README §6.2):

- **I1 — the inventory's classes and their order are derived from what the factories declare.**
  - Each factory's `inventory_records(conn, table, tables, context)` is one `read_records` call
    today. It becomes a declaration, a static `inventory_spec()` returning the keywords it passes,
    as an `InventorySpec`.
  - The classes of the inventory are those whose factory declares a spec.
  - Their order is derived from the parents the specs declare: **roots first**, then a dependency
    sort, each in registry order. Foreign keys cannot give it: nine of the parents the inventory
    reads have none (measured, §1 below).
- **I2 — the polymorphic reference is declared by the factory that holds it.**
  - `Parent` takes a type map: `Parent(column, type_column=…, types={type value: class})`.
  - `wavenumber_exit_time`'s and `BackgroundModel`'s factories declare their cosmology reference
    through one client constant.
  - The layer loses `COSMOLOGY` and `cosmology_classes`, and no longer imports `CosmologyModels`.
- **No project name is left in `store_inventory.py`'s code, comments or docstrings.** The tag
  table's names come from `Datastore.contract`.

**Why foreign keys cannot order the inventory (measured at `5fe0ae6`).** Among the 21 classes,
these declared parents have no foreign key:
- `wavenumber_exit_time` and `BackgroundModel` → `LambdaCDM`, `QCD_Cosmology` (the polymorphic
  reference);
- `GkSource` → `GkNumericIntegration`, `GkWKBIntegration` (its parent set's members);
- `QuadSourceIntegral` → `QuadSource`, `TkNumericIntegration`, `TkWKBIntegration` (cross-shard).

The reversed `schema.drop_order` over the foreign keys violates all nine. The user therefore took
the parents the inventory actually reads as the source of the order (README §6.2).

**Measured on 2026-10-06 with a rough version of this change** (orchestrator, at `5fe0ae6`; the
diff, the scripts and the outputs are in `orchestrator/measure-08/`):
- **The order.** The derived order equals `INVENTORY_CLASSES` exactly. A plain dependency sort
  differs from index 7 on: `wavenumber_exit_time` before `IntegrationSolver`, and `GkSourcePolicy`
  and `QuadSourcePolicy` later.
- **The fingerprint** of `build_full_store`, with `taken` removed, is byte-identical (digest
  `6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759`). So is `build_real_store`'s
  (digest `d932853819d07ce40fbb9cce544dabe4bbe968e4a7c908bb33e80eecbca3264f`), and so are both
  whole inventories: class order, records, problems, parent sets and members.
- **The report.** `main.py --inventory`, plain and with `--inventory-verbose`, on one
  `build_full_store` primary: stdout, stderr and exit code are byte-identical.
- **`resolve()`.** It gives byte-identical output for every record of both fixture stores (55 and
  18 records).
- **Imports.** Importing `store_inventory` loads `Datastore.contract` and nothing else, as before.
  `read_records` alone, given no registry, loaded `CosmologyModels` at `5fe0ae6`, because every call
  ran `cosmology_classes()`. Now it loads nothing.

**What it does not change.**
- Nothing a store holds (README §5 rule 9), and nothing the inventory reads or reports: the
  fingerprint, every record, every problem, the report's text and `resolve()`'s output.
- No refusal that exists today. `read_records`' "has not been built … classes must be built in
  dependency order" keeps its text. A registry that does not fit its store is refused by the
  reader exactly as now (§2, the order is derived after the store is opened).
- What is new:
  - `ClassInventory.parents[field]` is `None` for a polymorphic field, where it was `"cosmology"`,
    and `ClassInventory.parent_types` carries the field's type column and map;
  - refusals of a malformed declaration (§2), which the current registry meets none of.

---

## 2. What to change

**I1 — `InventorySpec` and the hook.**
- `store_inventory.InventorySpec`, a frozen dataclass whose fields are `read_records`' six
  keywords, with their defaults: `leaves`, `parents`, `tags`, `values`, `validated` and
  `parent_sets`.
  - `keywords()` returns them as a dict, for `read_records(conn, table, tables, context,
    **spec.keywords())`. Not `dataclasses.asdict`, which would turn each nested `Parent` into a
    dict.
  - `dependencies()` returns, each once, every class that must be built first:
    - every class a parent names, and every class a polymorphic parent's map names;
    - every class a parent set's members name;
    - `TAG_TABLE`, when `tags` is given. A class's tags are read through the tag table's records
      (`context.keys[TAG_TABLE]`), so a tag table built after it would turn every tag row into an
      `orphan-tag`. No root class is tagged, so the order does not move.
- `SQLAFactoryBase.inventory_spec()`, a static method returning `None`, **not abstract**. Its
  docstring says what a spec is, and that `None` means the class is not in the inventory.
- **The 21 factories of §6.** Each `inventory_records` becomes `inventory_spec()`, returning
  `InventorySpec(…)` with exactly the keywords it passed.
  - Each keeps its comment, and imports from `store_inventory` inside the method, as now.
  - Every one is a single `read_records` call today (measured: `measure-08/out/
    base_inventory_records.txt`), so the edit is mechanical. A factory that turns out to do more
    is a stop condition (§5).
  - `inventory_records` is deleted from every factory.

**I1 — `inventory_specs(factories)` and `inventory_classes(factories)`** in `store_inventory`.
- `inventory_specs` maps each class of the registry whose factory gives a spec (not `None`) to it,
  in registry order. A factory with no `inventory_spec` attribute gives none.
- `inventory_classes` gives the classes of `inventory_specs`, ordered:
  1. **the roots**: the classes whose `dependencies()` are empty, in registry order;
  2. then, one at a time, the earliest remaining class in registry order whose dependencies are
     all placed.
- It raises `ValueError`, naming the classes, for:
  - a dependency that is not a class of the inventory ("… references A, which declares no
    inventory_spec");
  - a polymorphic parent whose `type_column` is not one of its class's `leaves` (`resolve()` and
    the report read the type from the key);
  - a class that references itself;
  - a cycle ("the references of A, B, C form a cycle, or depend on one; …").
- The docstring states the rule, and that every class comes after every class it references.

**I1 — `read_inventory`.**
- It reads the classes of `inventory_classes(factories)`, in that order, each with
  `read_records(…, **specs[name].keywords())`.
- **The order is derived inside the `with open_read_only(…)` block, before any class is read, not
  before the store is opened.** Today a registry that lacks a class the store records (for example
  `BackgroundModel` or `store_tag`) is refused by the reader's `RuntimeError`. It must still be;
  the measurement's rough version derived first and raised `ValueError` instead (M7b).
- `INVENTORY_CLASSES` and `COMPUTE_TARGETS` are deleted, and `read_inventory`'s `KeyError` on a
  registry lacking a listed class goes with them (07's handed-on list).

**I2 — the polymorphic `Parent`.**
- `Parent`'s fields: `column`, `of: Optional[str] = None`, `type_column: Optional[str] = None`,
  `types: Optional[Mapping[Any, str]] = field(default=None, hash=False)`, `nullable`,
  `cross_shard`.
  - Positional `Parent(column, of)` keeps working.
  - `types` is not hashed, so that a `Parent` stays hashable.
- `__post_init__` raises `ValueError` naming the column when:
  - both or neither of `of` and `types` are given;
  - `type_column` is given without `types`, or `types` without `type_column`;
  - `types` is empty.
- `classes()` gives every class the reference can name. `class_of(mapping)` gives the class it
  names in a row or key: `of`, or `types.get(mapping[type_column])`. `read_records`' dependency
  check and `_reference` use them in place of the `COSMOLOGY` branches.
- `ParentSet.__post_init__` refuses a polymorphic member, naming the table and fields: neither
  `resolve()` nor the report could name its class.
- The docstrings say this in generic terms: no table is named.

**I2 — what the inventory carries.**
- `ShardRead` and `ClassInventory` gain `parent_types: field -> (type_column, {type: class})`, for
  each polymorphic parent. `_combine` carries it from the first shard's read, as it carries
  `parent_sets`.
- `parents[field]` is `None` for a polymorphic field, so the field is still in `parents`.
- `ClassInventory.parent_class(field, key)` gives the class a key field references:
  - the declared class;
  - for a polymorphic parent, the class its map gives the key's type column;
  - `None` for a type the map does not hold.
- `StoreInventory.resolve` uses `parent_class`, tests `field not in cls.parents` where it tested
  `of is None`, and no longer calls `cosmology_classes()`. Its output does not change (§3.3).

**I2 — the client's map: `Datastore/SQL/ObjectFactories/cosmology_types.py` (new).**
- `COSMOLOGY_TYPES = {LAMBDACDM_IDENTIFIER: "LambdaCDM", QCD_EOS_IDENTIFIER: "QCD_Cosmology"}`,
  importing `CosmologyModels.model_ids`. Its docstring says which factories use it and why it is
  on the client's side.
- `sqla_wavenumber_exit_time_factory.inventory_spec` and `sqla_BackgroundModelFactory.inventory_spec`
  declare `"cosmology": Parent("cosmology_serial", type_column="cosmology_type",
  types=COSMOLOGY_TYPES)`.

**`tools/inventory_report.py`.**
- It stops importing `COSMOLOGY` and `cosmology_classes`.
- `_Renderer.__init__` loses `self.types`, and `_Renderer.parent_class` returns
  `self.cls(name).parent_class(field, record.key)`.
- The two comments naming `inventory_records` name `inventory_spec`.
- `labelled()` is unchanged: it already drops a `None`, as it dropped `"cosmology"`.
- Nothing else in the tool changes: `INVENTORY_CATEGORIES` and `VALUE_TABLES` are the client
  tool's own (§5, observations).

**`store_inventory.py`'s text.**
- The module docstring describes the spec, the derived order and the polymorphic parent
  generically.
- No table of the registry, project package or project column is named in a comment or docstring:
  `version`, `store_tag`, `IntegrationSolver`, `GkSource`, `GkSource_parents`, `GkWKBIntegration`,
  `QuadSource`, `CosmologyModels`, `cosmology_type`, `*Value`, `*_tags`, `Run_<label>` and the
  rest go. The tag table is named through the contract, or as "the tag table".
- The module docstring says what it imports: the standard library, `sqlalchemy` and
  `Datastore.contract`.

**Nothing else changes.** Not `ShardedPool`, the actor, `schema.py`, `store_reader`, `ClientPool`,
`config/` or `RunRegistry/stores.py`, nor any factory method but `inventory_records`.

**The tests.** Measured on 2026-10-06 with the code change alone and no test edited:
- `Datastore.tests.test_inventory_report` and `Datastore.tests.test_store_inventory` fail to import
  `INVENTORY_CLASSES`. The `[g-o]` group ran 134 of 151, and `[s-z]` 150 of 191.
- `test_wkb_numeric_parent_key.TestZInitIsPayload.test_the_inventory_key_has_no_z_init` errors in
  both sub-tests: the factory has no `inventory_records`.
- `RunRegistry.tests.test_store_fingerprint` fails to import `COMPUTE_TARGETS`. RunRegistry ran 248
  of 280.
- Every other suite passed: Datastore `[a-f]` 150 and `[p-r]` 200, ComputeTargets 693,
  AdaptiveLevin 32, CosmologyModels 39 and LiouvilleGreen 148 (1 skipped). `test_layer_registry`
  passed throughout.

README §5 rule 7 applies to every test not named here. The edits, each only as stated:

- **`Datastore/tests/test_store_inventory.py`**:
  - The import of `INVENTORY_CLASSES` (`:40`) is replaced by a module-level literal
    `INVENTORY_CLASSES`: the 21 names of today's list, in today's order. A comment says that the
    layer derives this order and that this literal pins the derivation. A derived tuple would make
    `test_every_class_has_a_record` (`:338`) compare the derivation with itself.
  - `TestFloats.test_only_canonical_formats_a_float`:
    - its scan (`:592-615`) looks for functions named `inventory_spec` where it looked for
      `inventory_records`;
    - it skips `base.py`, whose hook declares no class, so that the count stays 21;
    - its docstring (`:566`) says `inventory_spec`.
- **`Datastore/tests/test_inventory_report.py`**: the import of `INVENTORY_CLASSES` (`:26`) is
  replaced by the same literal, with the same comment.
- **`Datastore/tests/test_wkb_numeric_parent_key.py`**,
  `TestZInitIsPayload.test_the_inventory_key_has_no_z_init` (`:506-517`):
  - `captured` is `factory.inventory_spec().keywords()`, in place of the `mock.patch` of
    `read_records` and the `inventory_records(None, None, None, None)` call;
  - its assertions are unchanged;
  - `mock` stays imported: `:918` and `:936` use it.
- **`RunRegistry/tests/test_store_fingerprint.py`**: the import of `COMPUTE_TARGETS` (`:49`) is
  replaced by a module-level literal of the nine compute targets, in today's order, with a comment
  that their labels are not identity. A derived set would widen `:301`'s test to the two policy
  tables.
- **Text only**: `Datastore/tests/real_store_fixtures.py:403`'s comment and
  `ComputeTargets/tests/test_qcd_cosmology_inventory_record.py:15`'s docstring name
  `inventory_classes` and `inventory_spec`.
- **No test is deleted, and no other test is edited.** Everything else passes unchanged.
- **New tests: `Datastore/tests/test_inventory_declarations.py`.** The measurement's sketch had 24,
  all passing; this list adds the refusal-order test. The log names each.
  - **The order.**
    - `inventory_classes(factories)` equals a literal of today's 21 names.
    - The classes are exactly those whose factory declares a spec.
    - No `inventory_spec` takes an argument.
    - Roots first is distinguished from a plain dependency sort, on the real registry and on a
      hand-built `R1, C(→R1), R2`.
    - Every class comes after every class it depends on, and a tagged class after the tag table.
  - **The refusals**, one test each, asserting the exception and the names:
    - a cycle; a self-reference;
    - a parent whose class declares no spec, or is absent from the registry;
    - a tagged class whose tag table declares no spec;
    - a polymorphic type column that is not a leaf;
    - each of `Parent`'s five malformed forms;
    - a polymorphic `ParentSet` member.
    - `Parent` stays hashable.
  - **The refusal order.** A registry lacking `BackgroundModel`, given to `read_inventory` on a
    `build_full_store` primary, raises the reader's `RuntimeError` naming the primary's
    `replicated_tables`, not `inventory_classes`' `ValueError`.
  - **The polymorphic parent.**
    - `resolve()` goes through a declared map, on a hand-built `StoreInventory` (two types and an
      unknown one) and on `build_full_store`.
    - On the full store, `parents["cosmology"] is None` and `parent_types` is as declared.
    - Both factories declare the same `COSMOLOGY_TYPES` object (`assertIs`).
  - **No project knowledge in `store_inventory.py`.**
    - An `ast` check: it imports no project package.
    - In a fresh interpreter, importing it and calling `read_records` on a hand-built table loads
      no project module.
    - A token check: no registry table, project package or project column is named in its code,
      comments or docstrings. `validated`, the record's own field and keyword, is allowed, and the
      test says why.

---

## 3. Verification

1. **Every suite**, with counts before and after, each difference accounted for:
   - Datastore: 692 + the new tests;
   - RunRegistry: 280; ComputeTargets: 693;
   - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped).
2. **Greps.** Show each one's output.
   - `git grep -nE 'INVENTORY_CLASSES|COMPUTE_TARGETS|cosmology_classes|\bCOSMOLOGY\b|inventory_records' -- '*.py'`
     prints only:
     - the two test-local `INVENTORY_CLASSES` literals and their uses;
     - `test_store_fingerprint.py`'s `COMPUTE_TARGETS` literal and its use;
     - the forbidden names listed in the new test;
     - `main.py:1207`'s comment ("BAKE THE BACKGROUND COSMOLOGY …"), which is unrelated.

     No `docs/` script uses any of these names (measured).
   - `grep -nE 'CosmologyModels|model_ids' Datastore/store_inventory.py tools/inventory_report.py`
     prints nothing.
3. **The inventory, from a scratch script, not committed.** At this commit and at an export of its
   parent:
   - the fingerprint of `build_full_store` and of `build_real_store`, with `taken` removed, is
     byte-identical. Give both digests;
   - the whole inventory of each is byte-identical, with `parents` normalised for the polymorphic
     field: class order, records, problems, timestamps, parent sets and members;
   - `resolve()` of every record of both stores is byte-identical;
   - `main.py --database <primary> --inventory`, plain and with `--inventory-verbose`, on one
     `build_full_store` primary that both trees read: stdout, stderr and exit code are
     byte-identical, and the store's checksums are unchanged.

   `orchestrator/measure-08/scripts/` holds the measurement's scripts (`m1_fingerprint.py`,
   `m2_report.sh`, `m3_resolve.py`); the agent may use its own.
4. **The refusals that exist today.** `read_records`' "has not been built" `RuntimeError` is
   produced at both trees and is byte-identical for a plain parent, a cross-shard parent, a
   polymorphic parent and a parent-set member. `read_inventory` with a registry lacking
   `OneLoopIntegral`, `BackgroundModel` and `store_tag` in turn gives byte-identical exceptions at
   both trees (M7b, with this prompt's order).
5. **Imports.** In a fresh interpreter at both trees:
   - importing `Datastore.store_inventory` loads only `Datastore.contract` of the project's
     modules;
   - `read_records` alone loads no `CosmologyModels` module after the change, where it loaded
     three.

   Give the lists.
6. **Names left.** The log lists every hit of a token scan of `store_inventory.py` (registry
   tables, project packages, identifier columns), before and after, each classified.
7. `black --check` is clean on every touched Python file.
8. **Deliberate breakage.** Each mutation must make a named test fail; then restore it.
   - (i) `inventory_classes` drops "roots first" and sorts plainly. The new order tests fail, and so
     does `test_store_inventory`'s `test_every_class_has_a_record`.
   - (ii) `InventorySpec.dependencies` leaves out a polymorphic parent's classes. The new order
     tests that place it fail.
   - (iii) `dependencies` leaves out `TAG_TABLE`. The new tagged-class test fails.
   - (iv) `ClassInventory.parent_class` ignores `parent_types` and returns `parents.get(field)`. The
     new `resolve()` tests fail. Report whether `test_inventory_report` fails too, and if not, why.
   - (v) The cycle check is removed. The new cycle test fails (an error, not a hang: show that it
     ends).
   - (vi) `read_inventory` derives the order before `open_read_only`. The new refusal-order test
     fails.
   - (vii) A project table's name is added to a comment in `store_inventory.py`. The new token
     test fails.

   Record each in the log as a diff, exactly as applied, with the failing tests.

---

## 4. Acceptance

1. **I1.**
   - `InventorySpec`, `SQLAFactoryBase.inventory_spec()` (not abstract), `inventory_specs` and
     `inventory_classes` exist as §2 says.
   - All 21 factories declare a spec, and none has `inventory_records`.
   - `INVENTORY_CLASSES` and `COMPUTE_TARGETS` are gone from the layer.
   - The derived order equals today's list.
2. **I2.**
   - `Parent` takes `types` and `type_column`.
   - `COSMOLOGY_TYPES` is declared once, in `ObjectFactories/cosmology_types.py`, and used by both
     factories.
   - `COSMOLOGY` and `cosmology_classes` are gone. Neither `store_inventory.py` nor
     `tools/inventory_report.py` imports `CosmologyModels`.
3. **No project name** is in `store_inventory.py`'s code, comments or docstrings.
4. **Nothing changes**: §3.3–§3.5 hold.
5. Every test of §2's plan has the fate listed, and the log lists each by module, class and method.
   The new tests exist, and the breakage record (i)–(vii) is in the log.
6. **The records:**
   - `[00-the-generic-datastore-layer-hard-codes-project-schema-facts]` gets a dated "Narrowed"
     line naming rows 12 and 13, the hook, `InventorySpec`, `inventory_classes` and `Parent`'s
     type map. It says every row of C1 is now gone, and that 09's guard remains. Its index hook
     says so.
   - `[08-resolve-shows-an-absent-nullable-parent-as-unresolved]` is opened on this board's §3 and
     in the index (§5). The index goes from 107 to 108.
   - On this board: the §1 row, items I1–I2, §3 and §5.
   - README §4's "08 ships" names what is shipped.

   All in the same commit, with `black --check` clean.

---

## 5. Stop conditions — stop and ask the user

- A test not named in §2's plan fails after the change. That is current behaviour moving; never
  edit it to pass.
- The fingerprint, the inventory, `resolve()`, the report or any refusal of §3.4 differs from its
  parent's.
- `inventory_classes(factories)` differs from today's list.
- A factory's `inventory_records` does anything but a single `read_records` call.
- The change would write, open or touch any store under `var/` (README §3), or change what any
  code writes (README §5 rule 9).
- Removing a name from the layer needs a project name added to it (README §5 rule 11).

**The issue to open:** `[08-resolve-shows-an-absent-nullable-parent-as-unresolved]`. This board's §3
gets the measurement, the impact and the next step; the index gets one row.
- **What.** `StoreInventory.resolve` renders a nullable key parent whose value is `None` as
  `{"class": …, "digest": None, "unresolved": 0}`. That is a parent it failed to resolve, where
  the row has no such parent. On `build_full_store`, 11 entries are of this form, all the
  `numeric` field of a pure-WKB `GkWKBIntegration` reached through `GkSource`'s parent set. By
  contrast, `_resolve_reference`, for a parent-set member, returns `None`.
- **Its age.** It is the same at `5fe0ae6`; 08 keeps it, so that `resolve()` stays byte-identical.
- **Its impact.** `resolve()` is called only by tests, for display and checking. The report renders
  through its own path.
- **Next step.** A decision for the user: render the absent parent as `None`, as members are. Not
  assigned.

**Observations to record, not acted on:**
- `tools/inventory_report.py`'s `labelled()` ignores a polymorphic parent's classes. That is
  harmless today, because both cosmology classes are replicated, and it is unchanged from
  `5fe0ae6`.
- The tool's `INVENTORY_CATEGORIES` and `VALUE_TABLES` name project tables. It is a client tool,
  outside the layer.

---

## 6. What this prompt does not do

- **Files it changes:**
  - new: `Datastore/SQL/ObjectFactories/cosmology_types.py` and
    `Datastore/tests/test_inventory_declarations.py`;
  - `Datastore/store_inventory.py` and `Datastore/SQL/ObjectFactories/base.py`, as §2 says;
  - the factories' `inventory_records` → `inventory_spec`, in:
    - `version.py`, `store_tag.py`, `redshift.py`, `wavenumber.py` (both factories),
      `tolerance.py`, `LambdaCDM.py`, `QCD_Cosmology.py`, `integration_metadata.py`;
    - `GkSourcePolicy.py`, `QuadSourcePolicy.py`, `BackgroundModel.py`;
    - `TkNumericIntegration.py`, `TkWKBIntegration.py`, `GkNumericIntegration.py`,
      `GkWKBIntegration.py`;
    - `GkSource.py`, `GkSourcePolicyData.py`, `QuadSource.py`, `QuadSourceIntegral.py`,
      `OneLoopIntegral.py`;
  - `tools/inventory_report.py`, as §2 says;
  - the tests §2 names: `test_store_inventory.py`, `test_inventory_report.py`,
    `test_wkb_numeric_parent_key.py`, `real_store_fixtures.py` (comment),
    `ComputeTargets/tests/test_qcd_cosmology_inventory_record.py` (docstring),
    `RunRegistry/tests/test_store_fingerprint.py`;
  - the log, this board, the README (§4) and `docs/OPEN_ISSUES.md`.

  Nothing else.
- **It does not edit `docs/` scripts.** None uses a name this prompt removes (measured).
- **It changes no factory's behaviour**: only `inventory_records` becomes `inventory_spec`.
- It does not touch `ShardedPool`, the actor, `schema.py`, `store_reader.py`, `ClientPool.py`,
  `config/`, `RunRegistry/stores.py` or `main.py`, nor add 09's guard.

---

## 7. The log and the board

`logs/08-derive-the-inventorys-class-list.md`, using README §5.1. In addition, it gives:
- every test edited and added, by module, class and method;
- the derived order, and what a plain dependency sort would give;
- the new refusal messages, as measured;
- the import lists before and after;
- the fingerprint digests and the sizes of §3.3's comparisons;
- every name left in `store_inventory.py` after the change, with its classification.

`IMPLEMENTATION_STATE.md`:
- the §1 row for 08, and items I1–I2;
- the narrowed generic-layer issue, and the opened `resolve()` issue;
- §5.

`docs/OPEN_ISSUES.md`: the generic-layer hook narrowed, the new row, the count (108) and the
date.
