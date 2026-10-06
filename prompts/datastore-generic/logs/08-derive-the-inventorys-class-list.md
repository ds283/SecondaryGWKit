# Log — prompt 08: derive the inventory's class list

**Subject:** *Derive the inventory's class list from the factories' declarations* · **Model:**
Claude Sonnet 5.5 · **Date:** 2026-10-06 · **Base:** `6ad58bb` (code as at `5fe0ae6`), clean ·
**Result:** landed.
- **I1:** every factory declares `inventory_spec()` in place of `inventory_records`; the layer
  derives the inventory's classes and their order from the parents the specs declare
  (`inventory_classes`), roots first. `INVENTORY_CLASSES` and `COMPUTE_TARGETS` are gone.
- **I2:** `Parent` takes a type map (`type_column=`, `types=`); `wavenumber_exit_time`'s and
  `BackgroundModel`'s factories declare their cosmology reference through one client constant,
  `COSMOLOGY_TYPES`. `COSMOLOGY` and `cosmology_classes` are gone, and the layer no longer imports
  `CosmologyModels`.

No stop condition fired. Closes nothing. Narrows
`[00-the-generic-datastore-layer-hard-codes-project-schema-facts]` (audit C1 rows 12 and 13: every
row of C1 is now gone; 09's guard remains). Opens
`[08-resolve-shows-an-absent-nullable-parent-as-unresolved]`. The index goes from 107 to 108. The
commit is `6646b06`.

## What shipped

**`Datastore/store_inventory.py`.**
- `InventorySpec`, a frozen dataclass whose fields are `read_records`' six keywords (`leaves`,
  `parents`, `tags`, `values`, `validated`, `parent_sets`) with their defaults. `keywords()` returns
  them as a dict (not `dataclasses.asdict`); `dependencies()` returns, each once, every class a
  parent names (every class of a polymorphic parent's map), every class a parent set's members
  name, and `TAG_TABLE` when `tags` is given.
- `Parent(column, of=None, type_column=None, types=None, nullable=False, cross_shard=False)`.
  `types` is `field(default=None, hash=False)`, so a `Parent` stays hashable and positional
  `Parent(column, of)` still works. `__post_init__` refuses: neither or both of `of` and `types`;
  `type_column` without `types` or `types` without `type_column`; empty `types`. `classes()` and
  `class_of(mapping)` replace the two `COSMOLOGY` branches of `read_records` (the dependency check
  and `_reference`).
- `ParentSet.__post_init__` refuses a polymorphic member, naming the table and the fields.
- `ShardRead` and `ClassInventory` gain `parent_types: field -> (type column, {type: class})`;
  `_combine` carries it from the first shard's read, as it carries `parent_sets`.
  `parents[field]` is `None` for a polymorphic field. `ClassInventory.parent_class(field, key)`
  gives the declared class, or the class the map gives the key's type column, or `None`.
- `StoreInventory.resolve` tests `field not in cls.parents` where it tested `of is None`, calls
  `parent_class`, and no longer calls `cosmology_classes()`.
- `inventory_specs(factories)` and `inventory_classes(factories)`. The order is: the roots (no
  dependencies) in registry order; then, one at a time, the earliest remaining class in registry
  order whose dependencies are all placed. `ValueError`, naming the classes, for a dependency that
  declares no spec, a polymorphic type column that is not a leaf, a self-reference and a cycle.
- `read_inventory` derives the order **inside** the `with open_read_only(...)` block, before any
  class is read, so a registry that does not fit its store is refused by the reader, as now.
  It calls `read_records(..., **specs[name].keywords())`.
- Deleted: `COMPUTE_TARGETS`, `INVENTORY_CLASSES`, `COSMOLOGY`, `cosmology_classes`, and
  `read_inventory`'s `KeyError` on a registry lacking a listed class.
- The module docstring and the `Parent`, `ParentSet`, `ShardContext`, `ClassInventory` and
  `read_records` docstrings are rewritten in generic terms (the spec, the derived order, the
  polymorphic parent). The module docstring says it imports the standard library, `sqlalchemy` and
  `Datastore.contract`.

**`Datastore/SQL/ObjectFactories/base.py`.** `SQLAFactoryBase.inventory_spec()`, static, not
abstract, returning `None`; its docstring says what a spec is and that `None` means the class is
not in the inventory.

**`Datastore/SQL/ObjectFactories/cosmology_types.py` (new).** `COSMOLOGY_TYPES =
{LAMBDACDM_IDENTIFIER: "LambdaCDM", QCD_EOS_IDENTIFIER: "QCD_Cosmology"}`, importing
`CosmologyModels.model_ids`. Its docstring says which factories use it and why it is on the
client's side.

**The 21 factories** (`version`, `store_tag`, `redshift`, `wavenumber` and `wavenumber_exit_time`
in `wavenumber.py`, `tolerance`, `LambdaCDM`, `QCD_Cosmology`, `IntegrationSolver` in
`integration_metadata.py`, `GkSourcePolicy`, `QuadSourcePolicy`, `BackgroundModel`,
`TkNumericIntegration`, `TkWKBIntegration`, `GkNumericIntegration`, `GkWKBIntegration`, `GkSource`,
`GkSourcePolicyData`, `QuadSource`, `QuadSourceIntegral`, `OneLoopIntegral`): `inventory_records`
becomes `inventory_spec()`, returning `InventorySpec(...)` with exactly the keywords it passed. The
decorator, the comment and the in-method import (now `InventorySpec` for `read_records`) are kept.
Every method was read after the edit. `wavenumber_exit_time` and `BackgroundModel` declare
`"cosmology": Parent("cosmology_serial", type_column="cosmology_type", types=COSMOLOGY_TYPES)`.
No factory method but `inventory_records` changes.

**`tools/inventory_report.py`.** It no longer imports `COSMOLOGY` or `cosmology_classes`;
`_Renderer.__init__` loses `self.types`; `_Renderer.parent_class` returns
`self.cls(name).parent_class(field, record.key)`; the two comments naming `inventory_records` name
`inventory_spec`. `labelled()`, `INVENTORY_CATEGORIES` and `VALUE_TABLES` are unchanged.

## The tests, test by test

### Edited (README §5 rule 7: each only as the prompt says)

- **`Datastore/tests/test_store_inventory.py`**
  - module level: the import of `INVENTORY_CLASSES` is replaced by a module-level literal
    `INVENTORY_CLASSES` of today's 21 names, in today's order, with a comment that the layer
    derives this order and the literal pins the derivation;
  - `TestFloats.test_only_canonical_formats_a_float`: its scan looks for functions named
    `inventory_spec`; it skips `base.py`, whose hook declares no class, so the count stays 21; its
    docstring says `inventory_spec`.
- **`Datastore/tests/test_inventory_report.py`**: module level only, the same literal.
- **`Datastore/tests/test_wkb_numeric_parent_key.py`**,
  `TestZInitIsPayload.test_the_inventory_key_has_no_z_init`: `captured` is
  `factory.inventory_spec().keywords()`, in place of the `mock.patch` of `read_records` and the
  `inventory_records(None, None, None, None)` call; its assertions are unchanged. `mock` stays
  imported (used at `:918` and `:936`).
- **`RunRegistry/tests/test_store_fingerprint.py`**: module level only: the import of
  `COMPUTE_TARGETS` is replaced by a literal of the nine compute targets, in today's order, with a
  comment that their labels are not identity.
- **Text only:** `Datastore/tests/real_store_fixtures.py` (a comment names `inventory_classes`) and
  `ComputeTargets/tests/test_qcd_cosmology_inventory_record.py` (a docstring names
  `inventory_spec`).
- No test is deleted. No other test is edited.

### Added: `Datastore/tests/test_inventory_declarations.py` (new; 25 tests)

`TestTheDerivedOrder`
- `test_it_is_the_order_the_inventory_was_built_in`: `inventory_classes(config.datastore.factories)`
  equals a literal of today's 21 names;
- `test_the_classes_are_those_that_declare_a_spec`;
- `test_a_spec_takes_no_connection`: no `inventory_spec` has a parameter, and each returns an
  `InventorySpec`;
- `test_roots_come_first_unlike_a_plain_sort`: on the real registry;
- `test_roots_first_on_a_hand_built_registry`: `R1, C(->R1), R2` gives `R1, R2, C`, where a plain
  sort gives `R1, C, R2`;
- `test_the_earliest_ready_class_in_registry_order`;
- `test_every_class_follows_every_class_it_references`;
- `test_a_tagged_class_follows_the_tag_table`.

`TestRefusals`
- `test_a_cycle_is_refused_by_name`; `test_a_self_reference_is_refused`;
- `test_a_parent_with_no_spec_is_refused_by_name` (two sub-tests: the parent declares `None`, and
  the parent is absent from the registry);
- `test_a_tagged_class_whose_tag_table_declares_no_spec_is_refused` (the same two sub-tests);
- `test_a_polymorphic_parents_classes_come_before_it` (an order test, not a refusal);
- `test_a_type_column_that_is_not_a_leaf_is_refused`;
- `test_a_malformed_parent_is_refused` (the five malformed forms, as sub-tests);
- `test_a_polymorphic_member_of_a_parent_set_is_refused`;
- `test_a_parent_stays_hashable`;
- `test_the_reader_refuses_a_registry_before_the_order_is`: a registry lacking `BackgroundModel`
  and its two tables, on a `build_full_store` primary: `inventory_classes` alone raises
  `ValueError`, and `read_inventory` raises the reader's `RuntimeError`, naming
  `replicated_tables` and `BackgroundModel`.

`TestResolve`
- `test_a_hand_built_inventory`: two types and an unknown one, through a declared map;
- `test_the_full_store_resolves_both_cosmology_types`: on `build_full_store`, `parents["cosmology"]
  is None` and `parent_types` is as declared;
- `test_both_referencing_factories_declare_the_one_map`: `assertIs` on `COSMOLOGY_TYPES`.
- *Added at the orchestrator's review, in its follow-up commit:*
  `test_the_report_renders_both_cosmology_types`, which renders `build_full_store`'s report and
  checks that every cosmology field is the key of the class its `cosmology_type` names. It pins
  the gap of breakage (iv) below. The same commit makes `test_a_malformed_parent_is_refused`
  assert that each message names the column. The module then holds 26 tests.

`TestTheLayerKnowsNoProject`
- `test_it_imports_no_project_package` (`ast`);
- `test_reading_records_loads_no_project_module` (a fresh interpreter, a hand-built table);
- `test_read_inventory_loads_only_what_the_registry_loads` (a fresh interpreter);
- `test_no_project_name_in_its_code_or_text`: a token check over `store_inventory.py` for every
  registry table, project package and the cosmology columns, in code names and in comments,
  docstrings and strings, and for the fragments `_tags`, `*Value` and `Run_<`. `validated`, the
  record's own field and `read_records`' keyword, is not in the list, and the test says why.

## The derived order, and a plain dependency sort

`inventory_classes(config.datastore.factories)` equals today's `INVENTORY_CLASSES`:

`version, store_tag, redshift, wavenumber, tolerance, LambdaCDM, QCD_Cosmology, IntegrationSolver,
GkSourcePolicy, QuadSourcePolicy, wavenumber_exit_time, BackgroundModel, TkNumericIntegration,
TkWKBIntegration, GkNumericIntegration, GkWKBIntegration, GkSource, GkSourcePolicyData, QuadSource,
QuadSourceIntegral, OneLoopIntegral`

The roots are the first ten. A plain dependency sort (the earliest ready class in registry order,
with no roots-first step) gives:

`version, store_tag, redshift, wavenumber, tolerance, LambdaCDM, QCD_Cosmology,
wavenumber_exit_time, IntegrationSolver, BackgroundModel, TkNumericIntegration, TkWKBIntegration,
GkNumericIntegration, GkWKBIntegration, GkSourcePolicy, GkSource, GkSourcePolicyData,
QuadSourcePolicy, QuadSource, QuadSourceIntegral, OneLoopIntegral`

It first differs at index 7 (`wavenumber_exit_time` where `IntegrationSolver` stands), and
`GkSourcePolicy` and `QuadSourcePolicy` come later. Every class follows its dependencies in both
orders. The registry order of the 21 spec'd classes is `version, store_tag, redshift, wavenumber,
wavenumber_exit_time, tolerance, ...`. (Measured by a scratch script, not committed; a test pins
it.)

## The new refusal messages, as measured

Scratch script, not committed; each is a `ValueError`.
- `Parent('c'): give exactly one of 'of' (the parent class) and 'types' (a polymorphic reference's
  type map)` (neither, and both);
- `Parent('c'): 'type_column' and 'types' are given together or not at all`;
- `Parent('c'): 'types' is empty`;
- `ParentSet('T'): member field(s) b are polymorphic; a member of a parent set names its class
  ('of')`;
- `inventory_classes(): B references A, which declares no inventory_spec` (two: `B references A,
  C, which declare no inventory_spec`; a tagged class with no tag-table spec: `B references
  store_tag, which declares no inventory_spec`);
- `inventory_classes(): B.a is a polymorphic parent whose type column t is not one of B's leaves`;
- `inventory_classes(): A references itself`;
- `inventory_classes(): the references of A, B, C form a cycle, or depend on one; no order builds
  every parent before its child`.

The refusals that exist today are byte-identical at both trees (§3.4 below).

## Imports (§3.5)

`Datastore.store_inventory`'s own import statements, before: `contextlib, hashlib, json, os`,
`collections.Counter`, `dataclasses.dataclass, field`, `datetime.datetime`, `pathlib.Path`,
`typing.(Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union)`, `sqlalchemy`,
`Datastore.contract.(TAG_LABEL, TAG_SERIAL, TAG_TABLE)`; function-local:
`CosmologyModels.model_ids` (in `cosmology_classes`) and `Datastore.store_reader.open_read_only`
(in `read_inventory`). After: the same, with the `CosmologyModels` import gone.

`tools/inventory_report.py`, from `Datastore.store_inventory`, before: `COSMOLOGY, ClassInventory,
Record, StoreInventory, cosmology_classes`; after: `ClassInventory, Record, StoreInventory`.

In a fresh interpreter, at the parent export and at this tree (`m5_imports.py`):
- importing `Datastore.store_inventory` loads, of the project's modules, `Datastore.store_inventory`
  and `Datastore.contract`, at both trees, as before;
- `read_inventory` adds `Datastore.store_reader` at both, and at this tree also
  `Datastore.SQL.ObjectFactories.cosmology_types` (a factory's `inventory_spec` imports it when
  `read_inventory` calls the hook; it is the client's own module);
- `read_records` alone, on a hand-built table and no registry (`m5b_read_records_alone.py`): at the
  parent it loaded `CosmologyModels`, `CosmologyModels.base` and `CosmologyModels.model_ids`
  (three, through `cosmology_classes()`, called on every call). At this tree it loads none.

## Names left in `store_inventory.py` (§3.6)

`m6_names_left.py` (a `tokenize` scan for every registry table, project package, identifier
column and the naming patterns), at the parent and at this tree: **105 hits before, 23 after.**
Before: the class lists, `GkSource`, `QuadSource`, `CosmologyModels`, `model_ids`, `cosmology*`,
`COSMOLOGY`, `*Value`, `*_tags`, `Run_<label>` and the rest. After, all of:
- `validated` (17: the record's field, `read_records`' keyword and local, the column label and
  `mapping["validated"]`): **layer vocabulary**, the record's own field (not a column it assumes);
- `tag_serial` (4: locals of `read_records`' tag loop): **generic local names**, unchanged. The
  contract's `TAG_SERIAL` is what names the column;
- `label` (2: a docstring's "a label or name", and SQLAlchemy's `.label("validated")`):
  **generic English / the library's method**.

No registry table, project package, project column or naming pattern is left. A test pins it.
`tools/inventory_report.py` goes from 86 hits to 81; the five fewer are `COSMOLOGY` (twice),
`cosmology_classes` (twice) and the `cosmology_type` string of the deleted branch. What remains is the client tool's own tables (`INVENTORY_CATEGORIES`,
`VALUE_TABLES`), outside the layer.

## Deviations from the prompt

1. **The rough diff was the starting point** (the prompt says it governs; where they differ, the
   prompt was followed): the order is derived inside `open_read_only`; the two existing test
   modules keep a literal; the refusal-order test is added; the issue and the records are written.
   *Not a deviation: the list of what differs is the prompt's.*
2. **`test_a_tagged_class_whose_tag_table_declares_no_spec_is_refused` replaces the sketch's
   `test_a_polymorphic_class_with_no_spec_is_refused`**, so that the refusals are the prompt's
   list and the count stays 25. The polymorphic parent's classes are still placed by
   `test_a_polymorphic_parents_classes_come_before_it`, which breakage (ii) fails.
   **IMPLEMENTATION CHOICE.**
3. **The token test also checks the fragments `_tags`, `*Value` and `Run_<`** in strings and
   comments, beside the table, package and column names, because the prompt names these patterns
   (§2, "`*Value`, `*_tags`, `Run_<label>`"). **IMPLEMENTATION CHOICE.**
4. **`InventorySpec.dependencies()` includes `TAG_TABLE` for a tagged class** (the prompt says so),
   and `inventory_classes` refuses a tagged class whose tag table declares no spec by the same
   message as any other missing dependency. **Not a deviation; recorded for the test above.**
5. **No dead helper and no unused import** is left by the change (a scan of every touched file's
   top-level imports found none unused; `mock` stays used in `test_wkb_numeric_parent_key.py`).

No `UNINTENDED DRIFT`.

## Verification performed

Suites, serially, in the foreground, output to files, verdicts grepped from them.

| Suite | Before (`6ad58bb`) | After |
|---|---|---|
| Datastore `[a-f]` | 150 | 150 OK |
| Datastore `[g-o]` | 151 | 176 OK (+25, `test_inventory_declarations`) |
| Datastore `[p-r]` | 200 | 200 OK |
| Datastore `[s-z]` | 191 | 191 OK |
| **Datastore** | **692** | **717 OK** |
| RunRegistry | 280 OK | 280 OK |
| ComputeTargets | 693 OK | 693 OK |
| AdaptiveLevin | 32 OK | 32 OK |
| CosmologyModels | 39 OK | 39 OK |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) |

No test outside the plan failed. The four known flakes did not fail.

**§3.2, the greps.**
- `git grep -nE 'INVENTORY_CLASSES|COMPUTE_TARGETS|cosmology_classes|\bCOSMOLOGY\b|inventory_records' -- '*.py'`
  prints, in tracked files: the two test-local `INVENTORY_CLASSES` literals and their uses
  (`test_inventory_report.py:49, 106, 108, 119, 161, 175, 194, 207`; `test_store_inventory.py:56,
  363, 364, 489, 640`); `test_store_fingerprint.py`'s `COMPUTE_TARGETS` literal and its use (`:61,
  :314`); and the scripts under `prompts/datastore-generic/orchestrator/measure-08/scripts/`,
  which the orchestrator committed and which name the removed names as the measurement of them.
  The forbidden names of the new test are in the untracked-until-commit
  `test_inventory_declarations.py`, whose token test lists `COSMOLOGY`. `main.py:1207`'s comment did
  not match `\bCOSMOLOGY\b` in this `git grep`.
  No `docs/` script uses any of them.
- `grep -nE 'CosmologyModels|model_ids' Datastore/store_inventory.py tools/inventory_report.py`
  prints nothing.

**§3.3, the inventory, from scratch scripts** (`m1_fingerprint.py`, `m3_resolve.py`, `m2_report.sh`
of the orchestrator's `measure-08/scripts/`, with their paths set to the two trees; not committed),
at an export of `6ad58bb` and at this tree:
- the fingerprint of `build_full_store`, with `taken` removed: digest
  `6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759`, **byte-identical** (7507
  bytes of JSON); of `build_real_store`: digest
  `d932853819d07ce40fbb9cce544dabe4bbe968e4a7c908bb33e80eecbca3264f`, **byte-identical** (4484
  bytes);
- the whole inventory of each, with `parents` normalised for the polymorphic field (class order,
  records, problems, timestamps, parent sets and members), **byte-identical** (29950 and 12297
  bytes); 21 classes, 0 problems each;
- `resolve()` of every record: 55 records of `build_full_store` (209794 bytes of JSON; 11 entries
  `"unresolved"`, the issue below) and 18 of `build_real_store` (7442 bytes; none),
  **byte-identical**;
- `main.py --database <primary> --inventory`, plain and with `--inventory-verbose`, on one
  `build_full_store` primary that both trees read: stdout (14844 and 14956 bytes), stderr (0 bytes)
  and exit code (0) **byte-identical**, and the store's checksums unchanged before and after every
  run;
- the raw `parents` of the two polymorphic classes differ on purpose: `"cosmology": null` at this
  tree where the parent had `"cosmology"`.

**§3.4, the refusals that exist today** (`m7_messages.py`, `m7b_reduced_registry.py`), at both
trees, **byte-identical**:
- `read_records`' `RuntimeError` "has not been built on shard #3; classes must be built in
  dependency order", for a plain parent, a cross-shard parent, the second class of a polymorphic
  parent (`QCD_Cosmology`) and a parent-set member;
- `read_inventory` on a `build_full_store` primary with a registry lacking `OneLoopIntegral` (and
  its tag table): the reader's `StoreSchemaMismatch`; lacking `BackgroundModel` (and its two
  tables), or `store_tag`: the reader's `RuntimeError` naming `replicated_tables`. The order is
  derived after the reader has checked, so each is the reader's, as at the parent.

**§3.5, §3.6:** above. **§3.7:** `black --check` on the 31 touched Python files: clean.

## The deliberate-breakage record

**How each was run.** The tree was the finished change, uncommitted. A scratch script made each
mutation in `Datastore/store_inventory.py` and took its diff against the unmutated file, so that
`git apply` replays it from a clean tree at `6646b06`. Each was run, then the file restored with
`cp`; after all seven the file was `cmp`-identical to the unmutated one. After the commit each
diff was checked in `bash` with `git apply --check`, and `git apply -R --check` after applying
it; `git status` was clean after. None is committed. Unmutated, the modules run below pass (67
tests: `test_inventory_declarations` and `test_store_inventory`; 62 for (iv)).

**(i) `inventory_classes` drops "roots first" and sorts plainly (`order` starts empty).**
`test_inventory_declarations` and `test_store_inventory`: Ran 67, **FAILED (failures=4)**:
- `test_inventory_declarations.TestTheDerivedOrder.test_it_is_the_order_the_inventory_was_built_in`;
- `…test_roots_come_first_unlike_a_plain_sort`;
- `…test_roots_first_on_a_hand_built_registry`;
- `test_store_inventory.TestTheFullStore.test_every_class_has_a_record` (it compares `tuple(baseline.classes)` with the literal).

```diff
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -953,7 +953,7 @@
         if name in spec.dependencies():
             raise ValueError(f"inventory_classes(): {name} references itself")
 
-    order: List[str] = [name for name in specs if len(depends[name]) == 0]
+    order: List[str] = []
     placed = set(order)
     while len(order) < len(specs):
         ready = [
```

**(ii) `InventorySpec.dependencies` leaves out a polymorphic parent's classes.**
Ran 67, **FAILED (failures=1)**: `test_inventory_declarations.TestRefusals.test_a_polymorphic_parents_classes_come_before_it`.
The real registry's order does not move (the cosmology classes are roots placed before the classes that reference them, in registry order), so the order test on the real registry passes; the hand-built registry places `C` before `A` and `B` without the dependency, and the test fails.

```diff
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -324,7 +324,8 @@
         records."""
         named: List[str] = []
         for parent in self.parents.values():
-            named.extend(parent.classes())
+            if parent.types is None:
+                named.extend(parent.classes())
         for spec in self.parent_sets.values():
             for parent in spec.members.values():
                 named.extend(parent.classes())
```

**(iii) `dependencies` leaves out `TAG_TABLE`.**
Ran 67, **FAILED (failures=3)**:
- `test_inventory_declarations.TestRefusals.test_a_tagged_class_whose_tag_table_declares_no_spec_is_refused`, both sub-tests (`tag_table=None` and `'absent'`);
- `test_inventory_declarations.TestTheDerivedOrder.test_a_tagged_class_follows_the_tag_table`.
No root class is tagged, so the real registry's order does not move and `test_it_is_the_order_the_inventory_was_built_in` passes, as the prompt says.

```diff
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -328,8 +328,6 @@
         for spec in self.parent_sets.values():
             for parent in spec.members.values():
                 named.extend(parent.classes())
-        if self.tags is not None:
-            named.append(TAG_TABLE)
         return tuple(dict.fromkeys(named))
 
 
```

**(iv) `ClassInventory.parent_class` ignores `parent_types` and returns `parents.get(field)`.**
`test_inventory_declarations`, `test_inventory_report` and `test_inventory_report_labels`: Ran 62, **FAILED (failures=6)**, all in `test_inventory_declarations.TestResolve`:
- `test_a_hand_built_inventory`;
- `test_the_full_store_resolves_both_cosmology_types` (five failing sub-tests on `wavenumber_exit_time` and the method's own);
`test_inventory_report` and `test_inventory_report_labels` **do not fail**. The mutation does change the report: on the `build_full_store` primary, `main.py --inventory` differs from the unmutated output in 118 lines of diff, every polymorphic `cosmology={omega_m=…}` becoming `cosmology=<unresolved reference>`. No test of those two modules pins how the cosmology field of a `wavenumber_exit_time` or `BackgroundModel` record renders: they check structure (sections, ordering, labelled classes), and the report's text is pinned only by the byte-identity comparison of §3.3, which is a scratch check. See "Observations not acted on".

```diff
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -408,9 +408,6 @@
         declared class, or for a polymorphic parent the class its type map gives the key's type
         column (``None`` for a type the map does not hold, or a field that is not a parent).
         """
-        if field in self.parent_types:
-            type_column, types = self.parent_types[field]
-            return types.get(key.get(type_column))
         return self.parents.get(field)
 
 
```

**(v) The cycle check is removed (the `if len(ready) == 0: raise ...` block).**
Ran 67, **FAILED (errors=1)**: `test_inventory_declarations.TestRefusals.test_a_cycle_is_refused_by_name`, an `IndexError: list index out of range` from `ready[0]`. It ends: the loop appends one class per pass, and an empty `ready` raises at once, so there is no hang (the run took the same ~20 s as the others, almost all of it `test_store_inventory`).

```diff
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -961,12 +961,6 @@
             for name in specs
             if name not in placed and all(d in placed for d in depends[name])
         ]
-        if len(ready) == 0:
-            stuck = [name for name in specs if name not in placed]
-            raise ValueError(
-                f"inventory_classes(): the references of {', '.join(stuck)} form a cycle, "
-                f"or depend on one; no order builds every parent before its child"
-            )
         order.append(ready[0])
         placed.add(ready[0])
     return order
```

**(vi) `read_inventory` derives the order before `open_read_only`.**
Ran 67, **FAILED (errors=1)**: `test_inventory_declarations.TestRefusals.test_the_reader_refuses_a_registry_before_the_order_is`: the `ValueError` "TkNumericIntegration references BackgroundModel, which declares no inventory_spec" propagates where the reader's `RuntimeError` is expected.

```diff
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -983,13 +983,11 @@
     which classes are replicated is read from the store's primary.
     """
     from Datastore.store_reader import open_read_only
+
+    names = inventory_classes(factories)
+    specs = inventory_specs(factories)
 
     with open_read_only(primary, factories) as store, contextlib.ExitStack() as stack:
-        # derived inside the block, after the reader has checked the registry against the store: a
-        # registry that does not fit its store is refused by the reader, as it always was, and not
-        # by a refusal of the order
-        names = inventory_classes(factories)
-        specs = inventory_specs(factories)
         replicated = set(store.replicated_tables)
         conns = {
             shard.serial: stack.enter_context(shard.engine.connect())
```

**(vii) A project table's name is added to a comment in `store_inventory.py`.**
Ran 67, **FAILED (failures=1)**: `test_inventory_declarations.TestTheLayerKnowsNoProject.test_no_project_name_in_its_code_or_text`.

```diff
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -109,7 +109,7 @@
 
 PathType = Union[str, os.PathLike]
 
-# up to this many examples are named in a problem
+# up to this many examples are named in a problem (as wavenumber_exit_time's)
 _EXAMPLES = 5
 
 
```

## Observations not acted on

- **The report's rendering of the polymorphic reference is pinned by no unit test**
  (breakage (iv)). `test_inventory_report` and `test_inventory_report_labels` pass with
  `parent_class` ignoring the type map, while the report prints `<unresolved reference>` for every
  cosmology field. 08's own `resolve()` tests fail, and the byte-identical comparison of §3.3
  catches it, but that is a scratch check. A test that renders a `build_full_store` report and
  looks for a resolved `cosmology={` would pin it. Not added: the prompt's test list does not name
  one, and the file `test_inventory_report.py` may be edited only as stated. Not opened as an
  issue: a decision for the orchestrator.
  *Added at review:* the orchestrator added that test to `test_inventory_declarations.py`
  (`TestResolve.test_the_report_renders_both_cosmology_types`) in its follow-up commit. Under
  breakage (iv) it fails too, so the module fails 7 tests where it failed 6.
- **`tools/inventory_report.py`'s `labelled()` ignores a polymorphic parent's classes.** Harmless
  today, because both cosmology classes are replicated; unchanged from `5fe0ae6` (the prompt's §5).
- **`INVENTORY_CATEGORIES` and `VALUE_TABLES`** in the tool name project tables. It is a client
  tool, outside the layer.
- **`resolve()` shows an absent nullable parent as unresolved.** Opened as
  `[08-resolve-shows-an-absent-nullable-parent-as-unresolved]`, on this board's §3.1 and in the
  index. Measured here: 11 entries of the form `{"class": …, "digest": None, "unresolved": 0}` in
  the `resolve()` of `build_full_store`'s 55 records, none in `build_real_store`'s 18; byte-identical
  at both trees. `read_records` itself already gives such a field `None`; it is `resolve` that shows
  a non-`None` reference for it.
- **`orchestrator/measure-08/scripts/` are tracked and name the removed names** (the orchestrator's
  committed measurement). They are scratch measurements of the old tree, not code that runs, and
  they are not touched here.

## State handed to the next prompt

- Audit C1 has no row left. 09's guard (a test that the generic layer names no registry table and
  imports no client package) is the one remaining item; its token test can reuse the shape of
  `test_no_project_name_in_its_code_or_text`, which already covers `store_inventory.py`.
- `inventory_specs(factories)` and `inventory_classes(factories)` are the layer's way to ask which
  classes the inventory reads; a new factory joins the inventory by declaring `inventory_spec()`,
  and a class that should not be in it declares nothing.
- `Parent` can declare a polymorphic reference; a new client with one declares its own type map, as
  `cosmology_types.py` does. `ClassInventory.parent_types` carries it to a display.
- The index stands at 108.
