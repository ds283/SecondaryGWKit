# Log — prompt 01: refuse a drop that leaves references

**Subject:** *Refuse a drop that leaves other tables naming dropped rows* · **Model:** Claude
Opus 5.5 · **Date:** 2026-10-07 · **Base:** `3e1e7f0`, clean (code as at `6de692a`) ·
**Result:** landed.
- **D1:** `Datastore.SQL.schema.dependent_tables(dropped, factories) -> List[str]`, beside
  `drop_order`.
- **D2:** `ShardedPool._refuse_drop_that_leaves_references()`, called in the constructor
  immediately after `_refuse_undeclared_drop_tables()`.
- **D3:** `Datastore/tests/test_drop_refuses_dangling_references.py` (11 tests), and
  `test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_store` re-fixtured.

No stop condition fired. Closes `[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]`
(`datastore-generic`). Narrows nothing. Opens nothing. Index 107 → 106. The commit is `fee1fbf`.

## What shipped

**`Datastore/SQL/schema.py`.**
- `dependent_tables(dropped, factories)`: the tables of `build_schema(sqla.MetaData(),
  factories).tables`, other than those in `dropped`, that name a table in `dropped` or in the
  result, transitively, in the registry's table order, each once; `[]` when nothing must go.
  - Table `A` names `B` (`B` not `A`) when `A` declares a foreign key into `B`, or when `A`'s
    factory declares an `inventory_spec` whose `dependencies()` hold `B`
    (`Datastore.store_inventory.inventory_specs(factories)`).
  - A name in `dropped` that the registry does not declare as a table raises `ValueError`, naming
    it: `dependent_tables(): ['NotATable'] is not declared as tables by the registry`.
  - Its docstring states the rule, that it is read from the registry and never from a store's rows
    (an empty dependent is still named), and why both kinds of reference count, without naming a
    project table.
- The import `from Datastore.store_inventory import inventory_specs`. `store_inventory` imports
  only the standard library, `sqlalchemy` and `Datastore.contract` at module scope, so nothing
  cycles: in fresh interpreters, `import Datastore.SQL.schema`, `import Datastore.store_inventory`
  and `import Datastore.SQL.ShardedPool; import Datastore.store_reader` each succeed.
- The module docstring gains a paragraph on `dependent_tables`, and its sentence on imports names
  the new one.

**`Datastore/SQL/ShardedPool.py`.**
- `dependent_tables` joins the `schema` import.
- `_refuse_drop_that_leaves_references()`: nothing when `self._drop_tables` is empty or `None`;
  otherwise `RuntimeError` when any table of `dependent_tables(self._drop_tables,
  self._factories)` is not among them. The message has the prefix once, the drop tables sorted,
  the dependents in `dependent_tables`' order, and ends `Nothing was opened`. It names only tables
  given or derived, no `--drop` group.
- Called in the constructor immediately after `self._refuse_undeclared_drop_tables()`, so for a
  read-write pool only (the read-only branch returns earlier), before the primary is created or
  opened and before any actor or engine exists.
- The constructor's `:param drop_tables:` now says that a read-write pool refuses, after an
  undeclared name, a drop that would leave another table naming rows of a dropped one, and that the
  rule is `schema.dependent_tables`'.

By `ast`, docstrings stripped, against `3e1e7f0`: `ShardedPool.py` adds
`_refuse_drop_that_leaves_references` and changes only `__init__`; `schema.py` adds
`dependent_tables` and changes no function.

**`dependent_tables`' ten lists**, equal to README §0.2's table and to
`measure/out/f2_closure.txt`'s full lists (a `diff` of the two is empty):

| Group | `dependent_tables(drop_groups[group], factories)` |
|---|---|
| `tk-numeric` | `['TkWKBIntegration', 'TkWKB_tags', 'TkWKBValue', 'QuadSource', 'QuadSource_tags', 'QuadSourceValue', 'QuadSourceIntegral', 'QuadSourceIntegral_tags']` |
| `tk-wkb` | `['QuadSourceIntegral', 'QuadSourceIntegral_tags']` |
| `quad-source` | `['QuadSourceIntegral', 'QuadSourceIntegral_tags']` |
| `gk-numeric` | `['GkWKBIntegration', 'GkWKB_tags', 'GkWKBValue', 'GkSourcePolicyData', 'GkSource', 'GkSource_tags', 'GkSourceValue', 'GkSource_parents', 'QuadSourceIntegral', 'QuadSourceIntegral_tags']` |
| `gk-wkb` | `['GkSourcePolicyData', 'GkSource', 'GkSource_tags', 'GkSourceValue', 'GkSource_parents', 'QuadSourceIntegral', 'QuadSourceIntegral_tags']` |
| `gk-source` | `['GkSourcePolicyData', 'QuadSourceIntegral', 'QuadSourceIntegral_tags']` |
| `gk-source-policy-records` | `['GkSourcePolicyData', 'QuadSourceIntegral', 'QuadSourceIntegral_tags']` |
| `gk-source-policy` | `['QuadSourceIntegral', 'QuadSourceIntegral_tags']` |
| `quad-source-integral` | `[]` |
| `1loop-integral` | `[]` |

`tables_to_drop(list(drop_groups))` gives `[]`. Every dependent lies in some group.

**The refusal's message**, as raised for `tables_to_drop(["gk-source-policy-records"])` on an
absent primary (`<primary>` is the resolved path; its directory was not created):

```
Cannot open sharded datastore "<primary>": cannot drop ['GkSourcePolicy'] alone: the rows of ['GkSourcePolicyData', 'QuadSourceIntegral', 'QuadSourceIntegral_tags'] would name rows that are gone, so they must be dropped too. Nothing was opened
```

**Tests added**, `Datastore/tests/test_drop_refuses_dangling_references.py`, 11. No Ray, nothing
under `var/`.
- `TestDependentTables`, on `config.datastore.factories` and `drop_groups`:
  1. `test_each_group_alone_needs_its_measured_dependents`: each group's list equals README §0.2's,
     as a literal (`MEASURED`), in registry order; `MEASURED`'s keys are `drop_groups`' keys;
  2. `test_each_group_with_its_dependents_needs_nothing_more`: `[]` for each;
  3. `test_every_group_together_needs_nothing`: `[]`;
  4. `test_a_declared_parent_without_a_foreign_key_counts`: `QuadSourceIntegral` has no foreign key
     into `TkWKBIntegration` and its spec's `dependencies()` hold it; `tk-wkb`'s dependents hold
     `QuadSourceIntegral`;
  5. `test_a_foreign_key_without_a_declared_parent_counts`: `GkSource_parents`' factory's
     `inventory_spec()` is `None` (and it is not in `inventory_specs`), it has a foreign key into
     `GkNumericIntegration`; `gk-numeric`'s dependents hold it;
  6. `test_dependents_are_followed_transitively`: `QuadSourceIntegral_tags` has no foreign key into
     a `tk-numeric` table and declares no spec; `tk-numeric`'s dependents hold it;
  7. `test_an_undeclared_name_is_refused`: `ValueError` naming `'NotATable'`, not `'GkSource'`.
- `TestThePoolRefuses`, through the stand-in pool as `test_layer_registry` opens one, counting
  stand-in actors and engines:
  8. `test_a_drop_that_leaves_references_is_refused_before_anything_is_opened`: a
     `build_full_store`, opened read-write with `tables_to_drop(["gk-source-policy-records"])`:
     `RuntimeError` (not `ReadOnlyWrite`), the message equal to the one above with its prefix once,
     no group named, no actor, no engine, no actor call, and `store_checksums` unchanged;
  9. `test_nothing_is_created_for_an_absent_store`: the same drop on an absent primary in an empty
     directory: the same refusal and message, and the directory stays empty;
  10. `test_an_undeclared_name_is_refused_first`: `["GkSourcePolicy", "NotATable"]` (where
      `GkSourcePolicy` alone has dependents) meets the undeclared-table refusal, not this one;
  11. `test_a_drop_with_its_dependents_is_accepted`: `tables_to_drop(["gk-source-policy-records",
      "gk-source-policy", "quad-source-integral"])` on a `build_full_store` (asserted first to hold
      rows in those tables: shard 0 holds 2, 2, 2, 3 and shard 1 2, 1, 1, 1). The pool opens and
      closes; afterwards every dropped table is present and holds no rows on either shard.

**Re-fixtured**, not deleted (README §5 rule 7):
`test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_store`. Its drop became
`tables_to_drop(["gk-source", "gk-source-policy-records", "gk-source-policy",
"quad-source-integral"])`, because `["gk-source", "gk-source-policy-records"]` alone is now
refused (it leaves `GkSourcePolicyData`, `QuadSourceIntegral` and `QuadSourceIntegral_tags`
naming dropped rows). Its point, that the dropped replicated `GkSourcePolicy` goes on every shard
alike, is unchanged; its docstring says why the drop names the dependents. Nothing else in it
changed. `test_one_timestamp_per_write.TestRepairThenPruneAtOpen` inherits it, unedited.

**Records.** `datastore-generic`'s board: the issue moved from §3.1 to §4 with a dated "Closed"
line, and the header's date. `docs/OPEN_ISSUES.md`: its row left §1.19, one sentence in §1.19's
prose says prompt 01 closed it, the count is 106 and the date 2026-10-07. This board: header, the
§1 row, D1–D3, §3, §4, §5.2. README §4: "01 ships" says it shipped, as named.

## Deviations from the prompt

1. **`schema.py`'s module docstring** gained a paragraph on `dependent_tables`, and its sentence on
   what the module imports now names `Datastore.store_inventory`. The prompt asks only for the
   function's docstring; the module docstring would otherwise state a false import list.
   `IMPLEMENTATION CHOICE`.
2. **One comment line** above the new call in the constructor. `IMPLEMENTATION CHOICE`.
3. **`dependent_tables` takes the spec's class as a node even when it has no table**
   (`named.setdefault`), and returns only tables. Every spec class is a table today (the
   orchestrator's check), so nothing differs. `IMPLEMENTATION CHOICE`.
4. **The refusal names `sorted(set(drop_tables))`**, each table once if a caller repeats one.
   `IMPLEMENTATION CHOICE`.
5. **Tests 8–10 also assert that no stand-in actor and no engine was made and no actor was
   called**, as `test_layer_registry`'s undeclared-table test does, and test 8 asserts the whole
   message as well as its parts. Tests 9 and 10 use an absent primary in an existing, empty
   directory; the prompt does not say which for test 10. Test 11 opens with `shards=2`, the
   store's own count, so the pool prints no mismatch warning. `IMPLEMENTATION CHOICE`.
6. **`black` moved the re-fixtured docstring's closing quotes onto their own line**, because its
   last line grew past the limit. `STRUCTURALLY REQUIRED` (README rules: `black` on every touched
   file).
7. **Records beyond the prompt's list of edits**: the `datastore-generic` board's "Last updated"
   date, and one sentence in `OPEN_ISSUES.md` §1.19's prose; this board's §3 and §4 text. All in
   files §6 lists. `IMPLEMENTATION CHOICE`.

No `UNINTENDED DRIFT`.

## Verification performed

**Suites**, README §7's command, each in the foreground, serially, output to a file and the
verdict grepped. Load 4.7–11.5. Baselines are the brief's, at `33308fe` (same code as `3e1e7f0`).

| Suite | Before (`33308fe`) | After | Difference |
|---|---|---|---|
| Datastore `[a-f]` | 147 OK | 158 OK | +11: `test_drop_refuses_dangling_references`, tests 1–11 |
| Datastore `[g-o]` | 183 OK | 183 OK | — |
| Datastore `[p-r]` | 200 OK | 200 OK | — (the re-fixtured test passes) |
| Datastore `[s-z]` | 191 OK | 191 OK | — (its inherited copy passes) |
| **Datastore** | **721** | **732** | +11 |
| RunRegistry | 280 OK | 280 OK | — |
| ComputeTargets | 693 OK | 693 OK | — |
| AdaptiveLevin | 32 OK | 32 OK | — |
| CosmologyModels | 39 OK | 39 OK | — |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | — |

Every suite OK on its first run; none of the four known flakes fired.

**The guard.** `test_layer_is_generic` passes (8 tests, in the `[g-o]` group and run alone):
`schema.py`'s and `ShardedPool.py`'s new docstrings, comments and message name no project table,
class or group, and `schema.py`'s new import is allowed.

**The measurement reproduced.** The ten lists above; `diff` against `f2_closure.txt`'s unindented
lines is empty.

**No accepted drop changes** (README §5 rule 9). A probe outside the repository computed
`drop_order` for `tables_to_drop(list(drop_groups))` and for each group's tables plus its §0.2
dependents, at `3e1e7f0` and on the changed tree: the eleven orders are identical. `main.py`,
`config/`, `Datastore/SQL/Datastore.py` and `Datastore/store_inventory.py` are unchanged against
`3e1e7f0`. Test 11 shows an accepted drop emptying exactly the tables it names.

**`black --check`** is clean on `Datastore/SQL/schema.py`, `Datastore/SQL/ShardedPool.py`,
`Datastore/tests/test_drop_refuses_dangling_references.py` and
`Datastore/tests/test_reconcile_at_open.py`.

No Ray was started, and nothing under `var/` was opened. `python -m RunRegistry list` at the start
showed nothing `running`.

## The deliberate-breakage record

Each diff is exactly as applied, against `fee1fbf`'s tree. Each was checked in `bash` with
`git apply --check` before and `git apply -R --check` after applying, then reversed with
`git apply -R`; `git status` afterwards showed only this prompt's changes. The tests named in the
prompt are in bold.

**(i) `dependent_tables` ignores declared parents.**

```diff
--- a/Datastore/SQL/schema.py
+++ b/Datastore/SQL/schema.py
@@ -363,8 +363,6 @@
         name: {fk.column.table.name for fk in table.foreign_keys} - {name}
         for name, table in tables.items()
     }
-    for name, spec in inventory_specs(factories).items():
-        named.setdefault(name, set()).update(set(spec.dependencies()) - {name})
 
     gone = set(names)
     found: set = set()
```

Run: `test_drop_refuses_dangling_references`. 7 failures: **test 1** (`tk-numeric`, `tk-wkb`, `quad-source`, `gk-numeric`, `gk-wkb`), **test 4**, and test 6.

**(ii) `dependent_tables` ignores foreign keys.**

```diff
--- a/Datastore/SQL/schema.py
+++ b/Datastore/SQL/schema.py
@@ -360,7 +360,7 @@
 
     # for each table, the other tables it names: by foreign key, then by declared parent
     named: Dict[str, set] = {
-        name: {fk.column.table.name for fk in table.foreign_keys} - {name}
+        name: set()
         for name, table in tables.items()
     }
     for name, spec in inventory_specs(factories).items():
```

Run: `test_drop_refuses_dangling_references`. 12 failures: **test 1** (every group but the two with no dependents), **test 5**, test 6, and tests 8 and 9 (the message lists `['QuadSourceIntegral']` only).

**(iii) `dependent_tables` follows one level only.**

```diff
--- a/Datastore/SQL/schema.py
+++ b/Datastore/SQL/schema.py
@@ -378,6 +378,7 @@
             break
         found |= more
         gone |= more
+        break
 
     return [name for name in tables if name in found]
 
```

Run: `test_drop_refuses_dangling_references`. 11 failures: **test 1** (all eight refused groups), **test 6**, and tests 8 and 9 (the message lacks `QuadSourceIntegral_tags`).

**(iv) The constructor's call of `_refuse_drop_that_leaves_references` removed.**

```diff
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -212,7 +212,6 @@
         # opened
         self._refuse_undeclared_drop_tables()
         # and so is a drop that would leave another table naming rows of a dropped one
-        self._refuse_drop_that_leaves_references()
 
         # if primary file is absent, all shard databases should be likewise absent
         if self._primary_file.is_dir():
```

Run: `test_drop_refuses_dangling_references`. 2 fail: **test 8** and **test 9** (`RuntimeError not raised`).

**(v) The call moved before `_refuse_undeclared_drop_tables()`.**

```diff
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -210,9 +210,9 @@
 
         # a drop table the registry does not declare is refused before the primary is created or
         # opened
+        self._refuse_drop_that_leaves_references()
         self._refuse_undeclared_drop_tables()
         # and so is a drop that would leave another table naming rows of a dropped one
-        self._refuse_drop_that_leaves_references()
 
         # if primary file is absent, all shard databases should be likewise absent
         if self._primary_file.is_dir():
```

Run: `test_drop_refuses_dangling_references` and `test_layer_registry.TestAnUndeclaredDropTable`. 4 errors: **test 10**, and `test_read_write_the_pool_refuses_it_before_anything_is_opened` (its `missing` and `existing` subtests and the method), each `ValueError: dependent_tables(): ['NotATable'] is not declared as tables by the registry` where the undeclared-table refusal was expected.

**(vi) The re-fixtured test's drop restored to `["gk-source", "gk-source-policy-records"]`.**

```diff
--- a/Datastore/tests/test_reconcile_at_open.py
+++ b/Datastore/tests/test_reconcile_at_open.py
@@ -1108,12 +1108,7 @@
                     self.reopen(
                         prune_unvalidated=prune,
                         drop_tables=tables_to_drop(
-                            [
-                                "gk-source",
-                                "gk-source-policy-records",
-                                "gk-source-policy",
-                                "quad-source-integral",
-                            ]
+                            ["gk-source", "gk-source-policy-records"]
                         ),
                     )
                 self.assertNoRecord()
```

Run: `test_reconcile_at_open.TestPruningAfterRepair` and `test_one_timestamp_per_write.TestRepairThenPruneAtOpen`. 4 errors: **`test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_store`** and **its inherited copy `test_one_timestamp_per_write.TestRepairThenPruneAtOpen.test_an_interrupted_store`**, each for `prune=True` and `prune=False`, with the new refusal: `cannot drop ['GkSource', 'GkSourcePolicy', 'GkSourceValue', 'GkSource_parents', 'GkSource_tags'] alone: the rows of ['GkSourcePolicyData', 'QuadSourceIntegral', 'QuadSourceIntegral_tags'] would name rows that are gone`.

## Observations not acted on

- `datastore-generic`'s board, §1, prompt 09's closing narrative still lists the drop issue under
  "Left open on this board". It is a record of that date, and stays (verification documents are
  additive); §3.1 and §4 carry the closure.
- README §2's table still says 01 is "written". The prompt names README §4 only; the orchestrator
  keeps §2's status column.
- `test_layer_registry.TestAnUndeclaredDropTable` drops `["GkSource", "NotATable"]`. `GkSource`
  alone is now refused by this prompt's refusal too, so that test's read-write case passes only
  because the undeclared-table refusal comes first. It needs no change; test 10 pins the order
  explicitly, and breakage (v) fails both.
- A user who runs `main.py --drop` with a refused group now meets a message naming tables, not
  groups (README §6.2's choice); `config/datastore.py` lists which group holds each. No document
  outside the verification records tells anyone to run a refused group alone (README §0.2).

None is a defect, so no §3 issue is opened.

## State handed to the next prompt

- `ShardedPool.py` changed only in the constructor's call and docstring, the `schema` import and
  the new method. Prompt 03 edits `_read_closed_store` and its callers, which this prompt did not
  touch.
- Datastore 732 (158 + 183 + 200 + 191); every other suite as §5.1. Expected after 02: 734; after
  03: 739.
- `docs/OPEN_ISSUES.md` counts 106; §1.19 holds the two issues 02 and 03 close.
