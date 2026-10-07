# Log — prompt 02: resolve an absent parent as `None`

**Subject:** *Resolve an absent nullable parent as None* · **Model:** Claude Sonnet 5.5 ·
**Date:** 2026-10-07 · **Base:** `a2465d2`, clean (code as at `fee1fbf`) · **Result:** landed.
- **V1:** `StoreInventory.resolve` gives `None` for a key parent whose value is `None`; two tests
  in `test_inventory_declarations.TestResolve`.

No stop condition fired. Closes `[08-resolve-shows-an-absent-nullable-parent-as-unresolved]`
(`datastore-generic`). Narrows nothing. Opens nothing. Index 106 → 105. The commit is `4c34f13`.

## What shipped

**`Datastore/store_inventory.py`, `StoreInventory.resolve` only.**
- After the `field not in cls.parents` branch, and before `parent_class` and `find` are called:

  ```python
  if value is None:
      out[field] = None
      continue
  ```

  A value that is not `None` and does not resolve to exactly one record goes on to today's
  `{"class", "digest", "unresolved"}`. The `parent_sets` branch and the leaf branch are untouched.
- The docstring gains a paragraph: a `None` parent is a row with no such parent, which only a
  nullable parent can hold (`read_records` drops a row whose parent does not resolve); it resolves
  to `None`, as a parent-set member does; a value that is not `None` and does not resolve to one
  record keeps the unresolved form. It names no project class or field.
- By `ast`, docstrings stripped, against `a2465d2`: no function or class is added or removed in
  the module, and the only one whose body differs is `StoreInventory.resolve`.

**`Datastore/tests/test_inventory_declarations.py`, two new methods in `TestResolve`.** No import
was needed, and no existing test changed. By `ast`, the module's only additions are the two
methods (and the local `walk` of the second).
- `test_an_absent_nullable_parent_resolves_to_none`: a hand-built inventory, class `C` with key
  parents `p` (of `A`) and `q` (of `B`). One record has `q: None` and `p` a real digest: `q`
  resolves to `None`, `p` to `{"class": "A", "key": {"x": 1}, "tags": []}`. A second record has `q`
  a digest naming no record: `q` keeps `{"class": "B", "digest": <it>, "unresolved": 0}`.
- `test_the_full_store_resolves_no_absent_parent_as_unresolved`: on `build_full_store`, walks
  `resolve()` of every record of every class, and asserts
  - the walk visited at least one `GkWKBIntegration` record (so it cannot pass on an empty
    inventory);
  - no mapping holding `"unresolved"` has `"digest": None`;
  - the `numeric` field of a `GkWKBIntegration` key that is `None` is counted **11** times (the key
    of a record of that class, or the `"key"` of a mapping whose `"class"` is that class);
  - the `numeric` field of a parent-set member that is `None` is counted apart, **9** times.

**Records.** `datastore-generic`'s board: the issue moved from §3.1 to §4 with a dated "Closed"
line that corrects the hook, and the header's date. `docs/OPEN_ISSUES.md`: the row left §1.19, one
sentence in §1.19's prose says prompt 02 closed it, the count is 105 and the date 2026-10-07. This
board: header, the §1 row, V1, §3, §4, §5.3.

## Deviations from the prompt

1. **"Exactly 11 `numeric` fields are `None`" is read as the orchestrator clarified.** The 11 are
   the `numeric` field of a `GkWKBIntegration` key, 0 before the change and 11 after.
   `GkSource`'s parent-set members have a field also named `numeric`; 9 are `None` both before and
   after, through `_resolve_reference`. A count of every `numeric: None` is 9 before and 20 after,
   not 11. The test counts the two apart and asserts 11 and 9. `STRUCTURALLY REQUIRED`.
2. **The first test's inventory** declares `parents={"p": "A", "q": "B"}`, two plain parents,
   where `test_a_hand_built_inventory` declares one polymorphic parent. The prompt asks for "a
   second key parent in the same record that does resolve"; the plain form needs no type map. The
   test adds a second record, with a `q` that names no record, to show that the unresolved form
   survives for a value that is not `None`. `IMPLEMENTATION CHOICE`.
3. **The docstring's second sentence about a value that is not `None`** goes beyond the prompt's
   "says that an absent parent gives `None`". `IMPLEMENTATION CHOICE`.
4. **`black`** wrapped one record line in the first new test. `STRUCTURALLY REQUIRED` (`black` on
   every touched file).
5. **Records beyond the prompt's list of edits**: the `datastore-generic` board's "Last updated"
   date, one sentence in `OPEN_ISSUES.md` §1.19's prose, and this board's §3 and §4 text. All in
   files §6 lists. `IMPLEMENTATION CHOICE`.
6. **Breakage (ii)** keeps the new branch and changes the unresolved form to `None` (the
   alternative, removing the branch and widening the other, is equivalent). `IMPLEMENTATION CHOICE`.

No `UNINTENDED DRIFT`.

## Verification performed

**Suites**, README §7's command, each in the foreground, serially, output to a file and the
verdict grepped. Load 8–33. Baselines are the brief's, at `a2465d2`.

| Suite | Before (`a2465d2`) | After | Difference |
|---|---|---|---|
| Datastore `[a-f]` | 158 OK | 158 OK | — |
| Datastore `[g-o]` | 183 OK | 185 OK | +2: `test_inventory_declarations.TestResolve.test_an_absent_nullable_parent_resolves_to_none` and `.test_the_full_store_resolves_no_absent_parent_as_unresolved` |
| Datastore `[p-r]` | 200 OK | 200 OK | — |
| Datastore `[s-z]` | 191 OK | 191 OK | — |
| **Datastore** | **732** | **734** | +2 |
| RunRegistry | 280 OK | 280 OK | — |
| ComputeTargets | 693 OK | 693 OK | — |
| AdaptiveLevin | 32 OK | 32 OK | — |
| CosmologyModels | 39 OK | 39 OK | — |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | — |

Every suite OK on its first run; none of the four known flakes fired.

**The measurement reproduced** (prompt §3.2). `r2_resolve.py` was run, from the export's own
directory, at an export of `a2465d2` (`git archive HEAD`) and at an export of the changed tree
(that export with the two changed files copied over). `r2_count.py` on each:

```
== parent
GkWKBIntegration key `numeric` None: 0
parent-set member `numeric` None:   9
unresolved mappings: 11, of which digest None: 11
== changed tree
GkWKBIntegration key `numeric` None: 11
parent-set member `numeric` None:   9
unresolved mappings: 0, of which digest None: 0
```

`r2_compare.py` on the two outputs, the 11 entries of §1 and nothing else (every entry of
`build_real_store` is unchanged); `diff` against the orchestrator's `r2_compare.txt` is empty:

```
11 entries differ
3 ('full', 'GkSource', 'numeric', '{"class": "GkNumericIntegration", "digest": null, "unresolved": 0}', 'null')
3 ('full', 'GkSourcePolicyData', 'numeric', '{"class": "GkNumericIntegration", "digest": null, "unresolved": 0}', 'null')
2 ('full', 'GkWKBIntegration', 'numeric', '{"class": "GkNumericIntegration", "digest": null, "unresolved": 0}', 'null')
3 ('full', 'QuadSourceIntegral', 'numeric', '{"class": "GkNumericIntegration", "digest": null, "unresolved": 0}', 'null')
```

The paths of the 11 (before → after, each `{"class": "GkNumericIntegration", "digest": null,
"unresolved": 0}` → `null`): `GkWKBIntegration[0..1].numeric`;
`GkSource[0..2].parents.set[0].wkb.key.numeric`;
`GkSourcePolicyData[0..2].source.key.parents.set[0].wkb.key.numeric`;
`QuadSourceIntegral[0..2].data.key.source.key.parents.set[0].wkb.key.numeric`.
So the 9 are the key of the `wkb` member in a `GkSource` record's parent set, and the issue's
hook ("all 11 reached through `GkSource`'s parent set") was wrong only by the 2 direct records.

**No `None` key parent on a non-nullable parent.** The full-store walk finds `"unresolved"`
mappings with a `None` digest in none of the records after the change, and the 11 `None` key
parents are all `GkWKBIntegration.numeric`.

**The guard.** `test_layer_is_generic` passes (8 tests, run alone). `black --check` is clean on
`Datastore/store_inventory.py` and `Datastore/tests/test_inventory_declarations.py`.

No Ray was started, and nothing under `var/` was opened.

## The deliberate-breakage record

Each diff is exactly as applied, against `4c34f13`'s tree. Each was checked in `bash` with
`git apply --check` before and `git apply -R --check` after applying, then reversed with
`git apply -R`, and the file compared with the original (identical). The tests run were
`test_inventory_declarations` and `test_store_inventory` (68 tests). The tests the prompt names
are in bold.

**(i) The new branch removed.**

```diff
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -470,9 +470,6 @@
             if field not in cls.parents:
                 out[field] = value
                 continue
-            if value is None:
-                out[field] = None
-                continue
             of = cls.parent_class(field, record.key)
             found = self.find(of, value) if of is not None else []
             if len(found) != 1:
```

2 failures: **`TestResolve.test_an_absent_nullable_parent_resolves_to_none`**
(`AssertionError: {'class': 'B', 'digest': None, 'unresolved': 0} is not None`) and
**`TestResolve.test_the_full_store_resolves_no_absent_parent_as_unresolved`**
(`AssertionError: 11 != 0`, the mappings with a `None` digest).

**(ii) The branch widened: any reference that does not resolve to one record gives `None`.**

```diff
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -476,7 +476,7 @@
             of = cls.parent_class(field, record.key)
             found = self.find(of, value) if of is not None else []
             if len(found) != 1:
-                out[field] = {"class": of, "digest": value, "unresolved": len(found)}
+                out[field] = None
                 continue
             parent = found[0]
             out[field] = {
```

2 failures: **`TestResolve.test_a_hand_built_inventory`** (its third child, of a type the map does
not hold: `AssertionError: None != {'class': None, 'digest': '5041bf1f…', 'unresolved': 0}`) and
`TestResolve.test_an_absent_nullable_parent_resolves_to_none` (its second record, whose `q` names
no record: `None != {'class': 'B', 'digest': '0ee219ee…', 'unresolved': 0}`). The full-store test
passes, because the full store holds no non-`None` reference that fails to resolve.

**(iii) The branch moved above `field not in cls.parents`.**

```diff
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -466,13 +466,13 @@
                         for member in cls.members.get(field, {}).get(value, ())
                     ]
                 }
-                continue
-            if field not in cls.parents:
-                out[field] = value
                 continue
             if value is None:
                 out[field] = None
                 continue
+            if field not in cls.parents:
+                out[field] = value
+                continue
             of = cls.parent_class(field, record.key)
             found = self.find(of, value) if of is not None else []
             if len(found) != 1:
```

Nothing fails (68 tests, OK), as expected. A `None` value gives `None` whether the branch precedes
the leaf branch or follows it, so this records that the order does not matter. It is not a pin.

## Observations not acted on

- `datastore-generic`'s board, §1, prompt 08's and 09's narratives still say the resolve issue is
  open, and its "Left open on this board" list (end of prompt 09's review) still names it. They are
  records of those dates and stay (verification documents are additive); §3.1 and §4 carry the
  closure.
- README §2's status column for 02 still says "written". The prompt does not list the README
  (§6); the orchestrator keeps it, as it did for 01.
- Only one key parent is declared nullable in the registry (`GkWKBIntegration.numeric`, by
  `inventory_specs(factories)`), and over `build_full_store` 2 records hold a `None` key parent,
  both on it (the 2 direct records of the 11). The prompt's stop condition, a `None` on a parent
  that is not nullable, did not fire on the fixtures. `resolve()` renders a `None` key parent as
  `None` whether or not the parent is nullable; it does not check.

None is a defect, so no §3 issue is opened.

## State handed to the next prompt

- `store_inventory.py` changed only in `resolve` and its docstring. Prompt 03 edits
  `ShardedPool.py` and `store_reader.py`, which this prompt did not touch.
- Datastore 734 (158 + 185 + 200 + 191); every other suite as §5.1. Expected after 03: 739.
- `docs/OPEN_ISSUES.md` counts 105; §1.19 holds the one issue 03 closes.
