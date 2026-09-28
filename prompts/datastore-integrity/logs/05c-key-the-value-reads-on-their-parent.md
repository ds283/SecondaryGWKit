# Log 05c — Key the value reads on their parent, and refuse a second parent

**Prompt:** [`prompts/datastore-integrity/05c-key-the-value-reads-on-their-parent.md`](../05c-key-the-value-reads-on-their-parent.md)
**Commit:** *(this commit)* — "Key the value reads on their parent and refuse a second parent"
**Model:** Claude Opus 5.5
**Date:** 2026-09-28
**Result:** COMPLETE, after one §6 stop, which the user decided. P3, P4 and P5 shipped. P9 closes
with this prompt's share: round trips for the three value `read_batch`es, and a pin on every
`read_batch` under `ObjectFactories/`. **No guard pin moved.** `main.py` is unchanged. Datastore
rises from 267 to 287, the 20 tests of the new `test_value_reads_parent_key.py`. The other five
suites match board §5.5.
`[00-numeric-value-parent-lookup-omits-break-point-kind]` closes on the store-fingerprint board's
§4. One issue is opened: `[05c-read-back-guard-cannot-see-reads-in-a-comprehension]`.

**The §6 stop, and its decision.** With the factory changes in place, and before any test edit, the
Datastore suite gave `errors=4`. Three were the `TestTagsOmittedDefaultToEmpty` tests the
orchestrator had already named (`KeyError: 'atol'`, below). The fourth was an existing test the
brief did not name, `test_read_path_mechanics.py`'s
`TestTkWKBValueFriction.test_read_batch_untagged_returns_every_row_with_its_own_friction`:

```
ERROR: test_read_batch_untagged_returns_every_row_with_its_own_friction (Datastore.tests.test_read_path_mechanics.TestTkWKBValueFriction.test_read_batch_untagged_returns_every_row_with_its_own_friction)
  File "/Users/ds283/Documents/Code/SecondaryGWKit/Datastore/SQL/ObjectFactories/TkWKBIntegration.py", line 1284, in read_batch
    raise MultipleResultsFound(
sqlalchemy.exc.MultipleResultsFound: TkWKBValue.read_batch(): 2 TkWKBIntegration parents match one key (model serial=1, k serial=11): parent serials [1, 2]. Refusing to return values from more than one parent.
```

Its fixture stores two `TkWKBIntegration` parents under one `(model, k)`, tagged `{T}` and
untagged, with the same `z_init`. It then asserted that an untagged read returns both parents'
`friction`, which is the mixture D1 forbids. That is §6's "an existing test fails", so I stopped
with nothing committed and reported it. **The user decided (2026-09-28, in chat, relayed by the
orchestrator) that the fixture is wrong, not the refusal.** The orchestrator named this one test as
a second permitted edit under README §5 rule 7: the untagged read must run against a store holding
one parent, and must still check that the stored `friction` comes back. That decision is the
authority for the edit recorded below.

## What shipped

**P3 — the numeric value reads** (`TkNumericIntegration.py`, `GkNumericIntegration.py`).
- `TkNumericValue._build_impl_model`, `GkNumericValue._build_impl_model` and
  `GkNumericValue.read_batch` filter the parent on
  `integration_table.c.break_point_kind == <Target>.BREAK_POINT_KIND`. The value is read from
  `ComputeTargets.TkNumericIntegration.BREAK_POINT_KIND` and
  `ComputeTargets.GkNumericIntegration.BREAK_POINT_KIND`. Those are the same class attributes the
  parents' `build()` filter on (`"all"` and `"discontinuity"`), already imported by each factory
  module.
- `atol` and `rtol` are required in all three reads (`payload["atol"]`, `payload["rtol"]`). The
  `if atol is not None:` / `if rtol is not None:` conditional filters became unconditional filters
  in the same `.filter(...)` call.
- A store without the column is refused with a `RuntimeError` naming `"break_point_kind"`. It comes
  from an `except SQLAlchemyError` branch that tests `"break_point_kind" in str(e)`, as the parents'
  `build()` do. The original `OperationalError` is kept as `__cause__`. In both `_build_impl_model`s
  it is a second `except` after the existing `except MultipleResultsFound`. In
  `GkNumericValue.read_batch` it wraps the row query's `execute`, which is where SQLite raises.
- `TkNumericValue`: one key, `z_init`. `_build_impl_model` now reads `payload["z_init"]`, which is
  what `build()` already dispatched on. `build()`'s two messages now say `TkNumericValue`, not
  `GkNumericValue` (deviation 5 covers their `z_init` wording).

**P4 — the WKB value reads** (`GkWKBIntegration.py`, `TkWKBIntegration.py`).
`GkWKBValue._build_impl_model`, `GkWKBValue.read_batch`, `TkWKBValue._build_impl_model` and
`TkWKBValue.read_batch` take an optional `z_init: Optional[float] = payload.get("z_init", None)`.
When it is given, they filter the parent with its `build()`'s own expression,
`sqla.func.abs(wkb_table.c.z_init - z_init) < DEFAULT_FLOAT_PRECISION`, unimproved (07 removes
`z_init` from the key).

**The refusal**, in the three `read_batch`es:
- **The parent serial is selected into the row query.** The value table's own foreign key is
  selected: `table.c.integration_serial` in `GkNumericValue.read_batch`, `table.c.wkb_serial` in
  the two WKB reads. The query is then executed once, `row_data = conn.execute(row_query).all()`,
  and the refusal is decided on those rows before any object is built. There is no second query.
- **Gk (both).** Parents are grouped by `z_source`, and the read refuses if any group holds more
  than one:
  ```python
  parents_by_z_source = {}
  for row in row_data:
      parents_by_z_source.setdefault(row.z_source_serial, set()).add(
          row.integration_serial          # row.wkb_serial in GkWKBValue.read_batch
      )
  for z_source_serial, parents in parents_by_z_source.items():
      if len(parents) > 1:
          raise MultipleResultsFound(...)
  ```
  This is `GkNumericIntegration.py:1067-1080` and `GkWKBIntegration.py:1238-1250`. It fires whether
  or not `z_source` was given.
- **Tk WKB.** It has no `z_source`, and every row of one call shares `(model, k)`:
  ```python
  parents = set()
  for row in row_data:
      parents.add(row.wkb_serial)
  if len(parents) > 1:
      raise MultipleResultsFound(...)
  ```
  This is `TkWKBIntegration.py:1286-1294`. It fires whether or not `z_init` was given.
- **Type and message.** `sqlalchemy.exc.MultipleResultsFound`, the type the `build` paths already
  raise on the same condition. The message names the value class, `k`'s serial, the `z_source`
  serial for the Gk reads (the model serial for Tk), and every matching parent serial, sorted. For
  example:
  `TkWKBValue.read_batch(): 2 TkWKBIntegration parents match one key (model serial=1, k serial=11): parent serials [7, 9]. Refusing to return values from more than one parent.`
- **Unchanged cases.** A read that matches no parent returns `[]`. One parent per key returns
  exactly what it returned before.

**P5 — the labels.** `GkNumericIntegration.build`'s `z_source_is_response`
(`GkNumericIntegration.py:226`) and `GkNumericValue.read_batch`'s `z_response_is_response`
(`:1039`) now label `redshift_table.c.response`, not `.source`.

**P9 — this prompt's share, and the close.**
- **The new module.** `Datastore/tests/test_value_reads_parent_key.py`, 20 tests, round-trips the
  three value `read_batch`es: tests 3–5 of §3, plus the refusal cases.
- **The pin.** `READ_BATCH_ROUND_TRIPS` in that module holds every `read_batch` under
  `ObjectFactories/`, as `"module.py:class"`, each naming the module that round-trips it.
  `test_every_read_batch_has_a_round_trip` finds them by walking the sources with the guard's own
  `FACTORY_DIR` and `iter_functions`, imported from `test_factory_select_columns.py`, and asserts
  the set equals the pin's keys. It holds five entries:
  - `GkNumericValue`, `GkWKBValue` and `TkWKBValue` → this module;
  - `QuadSourceIntegral` and `OneLoopIntegral` → `test_integral_read_batch` (05b).
- **The pin's second test.** `test_each_named_module_calls_its_read_batch` parses each named
  module with `ast`. It checks that the module really calls that factory's `read_batch`, either
  directly or through this module's `_read_batch(store, <factory>, …)` helper.

**Tests edited under the two permissions.** Each edit follows a failing run shown above.
- **The 05a review's finding 1 permission.** `test_read_path_mechanics.py`'s three
  `TestTagsOmittedDefaultToEmpty` payloads gained `"atol": _Ref(1), "rtol": _Ref(1)`. That matches
  the stored `atol_serial=1` and `rtol_serial=1` of `_gk_numeric_parent`/`_tk_numeric_parent`.
  Forced by, from the run with only the factories changed:
  ```
  ERROR: test_gk_numeric_value_build_impl_model (...TestTagsOmittedDefaultToEmpty...)
    File ".../GkNumericIntegration.py", line 854, in _build_impl_model
      atol: tolerance = payload["atol"]
  KeyError: 'atol'
  ERROR: test_gk_numeric_value_read_batch (...TestTagsOmittedDefaultToEmpty...)
    File ".../GkNumericIntegration.py", line 962, in read_batch
      atol: tolerance = payload["atol"]
  KeyError: 'atol'
  ERROR: test_tk_numeric_value_build_impl_model (...TestTagsOmittedDefaultToEmpty...)
    File ".../TkNumericIntegration.py", line 859, in _build_impl_model
      atol: tolerance = payload["atol"]
  KeyError: 'atol'
  ```
- **The user's decision of 2026-09-28.** `TestTkWKBValueFriction.test_read_batch_untagged_returns_every_row_with_its_own_friction`
  now builds its own store, holding one untagged parent (serial 2, `friction=0.77`), and asserts
  `[0.77]`. It was forced by the `MultipleResultsFound` quoted above. The test keeps its name,
  since "every row" is still true of one parent's rows. `setUp` and the tagged sibling are
  unchanged, and the sibling still passes. The two-parent shape is now
  `TestSecondParentIsRefused.test_tk_wkb_value_tags` in the new module.
- **Nothing else** in that module changed.

**The guard's pins: none moved.** Recomputed with the guard's own functions on the final tree:
- `EXPECTED_COVERAGE` 44, equal to the pin;
- `KNOWN_BLIND` 1, equal;
- `KNOWN_UNRESOLVED_TABLES` 9, equal;
- `KNOWN_UNCHECKED_INSERTS` 13, equal;
- `KNOWN_BLIND_COLUMNS` 4, equal;
- no shortfall and no column problem, so `KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS` are still
  empty.

The refusal loops bind `row` in each `read_batch`. That is the same coverage key as the inlined
`make_obj(row)` loop (`…Value_factory.read_batch.row`), so each adds a decidable binding under an
existing key. The analyser confirms each such binding reads only selected columns:
`['integration_serial', 'z_source_serial']`, `['wkb_serial', 'z_source_serial']` and
`['wkb_serial']`. The new filters name `integration_table.c.break_point_kind` and
`wkb_table.c.z_init`, both through a `tables["X"]` local that G2 resolves. `test_factory_select_columns.py`
was not edited.

## Deviations from the prompt

1. **STRUCTURALLY REQUIRED — two edits to an existing test module,
   `test_read_path_mechanics.py`.** The three payloads (permitted by the orchestrator, 05a review
   finding 1) and the one friction test (permitted by the user's decision after the §6 stop).
   Neither could pass against §2 unedited. The failing assertion behind each is quoted above.
2. **IMPLEMENTATION CHOICE — the friction test's shape: its own single-parent store, inside the
   test.** The alternative was to split the class's `setUp`. The chosen shape changes one method,
   leaves `setUp` and the tagged sibling byte-identical, and keeps the test's name.
3. **IMPLEMENTATION CHOICE — no shared helper; the refusal is inline in each `read_batch`.** The
   three files have no common module in this prompt's file list. A module-level helper would sit
   outside the functions §2 names. And a second nested function in a `read_batch` would break the
   guard's `inline_nested_row_helper`, which inlines only the first. The missing-column message is
   likewise a literal in each of the three functions.
4. **IMPLEMENTATION CHOICE — the parent serial is the value table's own foreign key**
   (`table.c.integration_serial` / `table.c.wkb_serial`), not `subquery.c.serial`. It is the same
   value, joined on equality. It keeps the new reference resolvable by G2, rather than adding
   another `subquery.c.*` reference.
5. **IMPLEMENTATION CHOICE — `TkNumericValue.build`'s two messages also say `z_init` where they
   said `z_source`.** §2 asked for the class name to be corrected. After P3's one-key change, the
   old tuple `(model_proxy, wavenumber, z_source)` would name a key the method no longer reads.
6. **IMPLEMENTATION CHOICE — `TkWKBValue.read_batch` collects its parents in a `for` loop, not a
   set comprehension.** The first draft was `{row.wkb_serial for row in row_data}`. The guard
   produced no finding for that read at all, because its analyser walks a comprehension's element
   before its target is bound. So the parent read would have been unguarded. A `for` loop is a
   shape the analyser decides (§2: "Prefer a shape the analyser decides"). The hole this exposed is
   the issue opened below.
7. **IMPLEMENTATION CHOICE — a test for §2's missing-column requirement.** §3 does not list one, so
   it was added as `TestStoreWithoutBreakPointKindIsRefused`. It builds the in-memory schema, then
   runs `ALTER TABLE … DROP COLUMN break_point_kind`, which SQLite 3.53.4 supports. It asserts a
   `RuntimeError` naming the column, whose `__cause__` is an `OperationalError`. It covers
   `GkNumericValue.build`, `GkNumericValue.read_batch` and `TkNumericValue.build`.
8. **IMPLEMENTATION CHOICE — the new module's store helpers** (`Store`, `_ModelProxy`, `_Ref`,
   `_z`, `_exit_time_obj`) are local copies, as 05a's and 05b's were. Neither earlier module exports
   them, and `test_integral_read_batch.py` is not in this prompt's file list.
   `_exit_time_obj` builds a real `wavenumber_exit_time`, because the real `GkNumericIntegration`
   constructor, reached in test 6, reads `k.k.units`.

No `UNINTENDED DRIFT`. Every changed code line in the four factories is inside a value factory's
`build`, `_build_impl_model` or `read_batch`, or is `GkNumericIntegration.build`'s P5 label.
Checked by listing every `+`/`-` line of `git diff --cached -- Datastore/SQL/ObjectFactories/`.
Nothing was removed except the lines P3 replaces: the four `if atol/rtol is not None` blocks and
the four optional `payload.get` lines.

## Verification performed

**The new module**, 20 tests, all on the real factory code, schema in memory, no Ray, nothing
under `var/`:

| Class (tests) | §3 item | What it checks |
|---|---|---|
| `TestBreakPointKindSelects` (3) | 1 | Two parents differing only in kind. `GkNumericValue` `_build_impl_model` and `read_batch`, and `TkNumericValue.build` with `z_init` alone, serve only the target's kind; nothing raises |
| `TestStoreWithoutBreakPointKindIsRefused` (2) | §2 | Column dropped: `RuntimeError`, not `OperationalError` |
| `TestTolerancesRequired` (1, 6 subtests) | 2 | Each of the three numeric reads, missing `atol` or `rtol`, raises `KeyError` naming the key; the full payload is served |
| `TestLivePayloadShape` (2) | 3 | `main.py`'s payload: tags `[T, U]`, `z`, no `z_source`, tolerances for numeric only. Three parents at three `z_source`s, values at two response `z`s. Exactly three values come back, one per `z_source`, each at the requested `z`, each from the parent at its own `_z_source` |
| `TestSecondParentIsRefused` (6) | 4 | Tags `{T}`/`{T,U}` queried with `[T]`, in all three `read_batch`es, with `z_source` given and without (Tk: with `z` and without). Two `z_init`s with none given (both WKB). Each raises `MultipleResultsFound` naming `[7, 9]`, `k serial=11`, and `z_source serial=1` for Gk. `[T, U]` selects one. Parents at different `z_source`s return both, unrefused |
| `TestZInitSelects` (2) | 5 | A given `z_init` serves only its parent, through `read_batch` and `_build_impl_model`, for `GkWKBValue` and `TkWKBValue` |
| `TestResponseFlagsAreRead` (2) | 6 | Redshifts stored `(source, response)` = `(True, False)` and `(False, True)`, each used once as `z_source` and once as `z`. `GkNumericIntegration.build`'s `z_source` and `GkNumericValue.read_batch`'s `_z_source` and `z` report what was stored |
| `TestEveryReadBatchHasARoundTrip` (2) | 7 | P9's pin, and that each named module calls its `read_batch` |

Against the unchanged factories (the fix stashed, this module alone), the module gave
`failures=23, errors=4`. Every class except `TestLivePayloadShape` and the pin failed; those two
pass on either tree, as they should, since the live path was already correct with one parent per
key.

**Callers of the six changed reads**, found by `grep -rn` across `main.py`, `ComputeTargets/`,
`extract_*.py`, `docs/` and `tools/`. The patterns were the four value class names, and
`read_batch(` / `object_read_batch`.

| Caller | Reads | Behaviour | Evidence |
|---|---|---|---|
| `main.py` `build_missing_GkSource` | `GkNumericValue.read_batch`, `GkWKBValue.read_batch`, via `pool.object_read_batch` | **Unchanged on a store with one parent per `(k, z_source)`**: the refusal needs two. It already passes `atol`/`rtol` for `GkNumericValue`. It passes no `z_init`, so the new filter is inert. `k` arrives through `shard_key` (`ShardedPool.object_read_batch`: `payload.update(shard_key)`). The corrected `z_response_is_response` flag moves no behaviour: nothing in `main.py` reads `.is_response` off a returned object | `main.py:2391-2429`: `"z": z_response`, `**({"atol": Gk_numeric_atol, "rtol": Gk_numeric_rtol} if cls_name == "GkNumericValue" else {})`, the `for cls_name in ["GkNumericValue", "GkWKBValue"]` loop; `grep '\.is_response\b'` outside tests finds only `probe_05` |
| `main.py`, anything else | none | — | `grep` for the four class names in `main.py` matches only lines 2412 and 2429 |
| `ComputeTargets/` | none | unchanged | Value classes appear only as types and constructors (`GkSource.py`, `WKB_Gk.py`, `TkSourceFunctions.py`, …), never through an `object_get`/`object_read_batch`. `test_main_plumbing.py:596` is a comment |
| `extract_Gk_data.py`, `extract_GkWKB_data.py`, `extract_TkWKB_data.py` | none of the six | unchanged | They read values off the parent's `.values` (`extract_Gk_data.py:119`, `extract_GkWKB_data.py:149,186`, `extract_TkWKB_data.py:141,200`), which the parents' `build()` populate. `extract_Gk_data.py:422` calls `GkNumericIntegration.build`, whose only change is P5's flag, and it reads no flag |
| `extract_QuadSourceIntegral_data.py:1087` | `QuadSourceIntegral.read_batch` | unchanged, not one of the six | — |
| `docs/datastore-integrity-audit/lookup_keys/probe_05_value_lookups_and_read_batch.py` | all six | **Its two-parent cases now refuse.** Its no-tolerance calls raise `KeyError`, and its flag line now prints `is_response=True`. That is exactly the defect record the probe exists to show, and its `.out` stays correct for the audit's tree | `probe_05:229-363`. Not edited (§1 "Do not edit them") |
| `docs/gktk-remedial/analyse_scoped_run.py:307,313`, `docs/source-remediation-verification/analyse_greens_and_source.py:161`, `tools/inventory_report.py:100-104` | none | unchanged | Raw SQL or a table-name map, not the factories |

No caller outside the tests newly refuses on a store with one parent per key.

**`var/`.** Nothing under `var/` was opened or read, apart from `python -m RunRegistry list`,
which lists `var/runs/`, at session start: nothing `running`. No store was opened to ask whether a
real store would refuse.

**Suites**, README §7's command, run serially on the final tree:

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 552 OK |
| CosmologyModels | 39 OK |
| Datastore | **287 OK** (267 + the 20 of `test_value_reads_parent_key.py`) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

`./venv/bin/python -m black --check` is clean on all six Python files this commit touches.

## The deliberate-breakage record

The fix (the four factories and both test modules) was staged with `git add`. Each mutation was
then applied unstaged. Each diff below is `git diff`, taken while that mutation was applied and
pasted unaltered. The whole Datastore suite was run for each; every failing test and subtest is
listed. Each mutation was reverted with `git checkout -- <file>`, after which the unstaged diff
was empty and the staged set unchanged.

**(i) remove the `break_point_kind` filter from `GkNumericValue.read_batch`:**

```diff
diff --git a/Datastore/SQL/ObjectFactories/GkNumericIntegration.py b/Datastore/SQL/ObjectFactories/GkNumericIntegration.py
index 6699e20..0a8b22d 100644
--- a/Datastore/SQL/ObjectFactories/GkNumericIntegration.py
+++ b/Datastore/SQL/ObjectFactories/GkNumericIntegration.py
@@ -983,10 +983,6 @@ class sqla_GkNumericValue_factory(SQLAFactoryBase):
                 integration_table.c.validated == True,
                 integration_table.c.atol_serial == atol.store_id,
                 integration_table.c.rtol_serial == rtol.store_id,
-                # the parent's key includes its break-point policy, read from the compute target's
-                # single declaration exactly as the parent's build() reads it (prompt 05c, P3)
-                integration_table.c.break_point_kind
-                == GkNumericIntegration.BREAK_POINT_KIND,
             )
             .select_from(
                 integration_table.join(
```

Result: `Ran 287`, `failures=1, errors=1`.
- `TestBreakPointKindSelects.test_gk_numeric_value_read_batch_serves_only_the_kind_asked_for`
  errors with the new refusal, because both kinds now match:
  `MultipleResultsFound: GkNumericValue.read_batch(): 2 GkNumericIntegration parents match one key (k serial=11, z_source serial=1): parent serials [1, 2]. …`
- `TestStoreWithoutBreakPointKindIsRefused.test_gk_numeric_value_reads` fails with
  `AssertionError: RuntimeError not raised`: a `read_batch` that no longer names the column cannot
  detect a store without it.

**(ii) remove `GkWKBValue.read_batch`'s refusal:**

```diff
diff --git a/Datastore/SQL/ObjectFactories/GkWKBIntegration.py b/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
index b85cade..70ca0d4 100644
--- a/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
+++ b/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
@@ -1240,14 +1240,6 @@ class sqla_GkWKBValue_factory(SQLAFactoryBase):
             parents_by_z_source.setdefault(row.z_source_serial, set()).add(
                 row.wkb_serial
             )
-        for z_source_serial, parents in parents_by_z_source.items():
-            if len(parents) > 1:
-                raise MultipleResultsFound(
-                    f"GkWKBValue.read_batch(): {len(parents)} GkWKBIntegration parents match one "
-                    f"key (k serial={k.store_id}, z_source serial={z_source_serial}): parent "
-                    f"serials {sorted(parents)}. Refusing to return values from more than one "
-                    f"parent."
-                )
 
         def make_obj(row):
             obj = GkWKBValue(
```

Result: `Ran 287`, `failures=4`, all `AssertionError: MultipleResultsFound not raised`:
- `TestSecondParentIsRefused.test_gk_wkb_value_tags`, subtests `z_source=False` and
  `z_source=True`;
- `TestSecondParentIsRefused.test_gk_wkb_value_two_z_inits`, subtests `z_source=False` and
  `z_source=True`.

**(iii) remove the `z_init` filter from `TkWKBValue.read_batch`:**

```diff
diff --git a/Datastore/SQL/ObjectFactories/TkWKBIntegration.py b/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
index 3605f0d..26583cc 100644
--- a/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
+++ b/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
@@ -1218,14 +1218,6 @@ class sqla_TkWKBValue_factory(SQLAFactoryBase):
             wkb_table.c.rho_gauss_order == phase_residual.RHO_GAUSS_ORDER
         )
 
-        # z_init is in the parent's key when it is given: filtered with the parent build()'s own
-        # expression, the absolute DEFAULT_FLOAT_PRECISION match (prompt 05c, P4; prompt 07 removes
-        # z_init from the key)
-        if z_init is not None:
-            wkb_query = wkb_query.filter(
-                sqla.func.abs(wkb_table.c.z_init - z_init) < DEFAULT_FLOAT_PRECISION
-            )
-
         count = 0
         for tag in tags:
             tag: store_tag
```

Result: `Ran 287`, `errors=2`. `TestZInitSelects.test_tk_wkb_value`, subtests `z_init=10000.0` and
`z_init=20000.0`, both error with
`MultipleResultsFound: TkWKBValue.read_batch(): 2 TkWKBIntegration parents match one key (model serial=1, k serial=11): parent serials [7, 9]. …`.
The given `z_init` no longer selects, and the refusal catches the mixture.

**(iv) put `.source` back in `GkNumericValue.read_batch`'s `z_response_is_response` label:**

```diff
diff --git a/Datastore/SQL/ObjectFactories/GkNumericIntegration.py b/Datastore/SQL/ObjectFactories/GkNumericIntegration.py
index 6699e20..b3cf512 100644
--- a/Datastore/SQL/ObjectFactories/GkNumericIntegration.py
+++ b/Datastore/SQL/ObjectFactories/GkNumericIntegration.py
@@ -1036,7 +1036,7 @@ class sqla_GkNumericValue_factory(SQLAFactoryBase):
             redshift_table.c.z.label("z_response"),
             table.c.z_serial.label("z_response_serial"),
             redshift_table.c.source.label("z_response_is_source"),
-            redshift_table.c.response.label("z_response_is_response"),
+            redshift_table.c.source.label("z_response_is_response"),
         ).select_from(
             subquery.join(table, table.c.integration_serial == subquery.c.serial).join(
                 redshift_table, redshift_table.c.serial == table.c.z_serial
```

Result: `Ran 287`, `failures=2`, both in `TestResponseFlagsAreRead.test_gk_numeric_value_read_batch`:
- subtest `parent=1.0`: `AssertionError: Tuples differ: (False, True) != (False, False)`;
- subtest `parent=2.0`: `AssertionError: Tuples differ: (True, False) != (True, True)`.

**(v) make `atol` optional again in `TkNumericValue._build_impl_model`:**

```diff
diff --git a/Datastore/SQL/ObjectFactories/TkNumericIntegration.py b/Datastore/SQL/ObjectFactories/TkNumericIntegration.py
index 7bc6411..5f0ee3e 100644
--- a/Datastore/SQL/ObjectFactories/TkNumericIntegration.py
+++ b/Datastore/SQL/ObjectFactories/TkNumericIntegration.py
@@ -856,7 +856,7 @@ class sqla_TkNumericValue_factory(SQLAFactoryBase):
 
         # the parent's own lookup (sqla_TkNumericIntegration_factory.build) requires the tolerance
         # pair, so this read does too (prompt 05c, P3)
-        atol: tolerance = payload["atol"]
+        atol: Optional[tolerance] = payload.get("atol", None)
         rtol: tolerance = payload["rtol"]
         tags: List[store_tag] = payload.get("tags", [])
 
@@ -868,7 +868,6 @@ class sqla_TkNumericValue_factory(SQLAFactoryBase):
                 integration_table.c.wavenumber_exit_serial == k.store_id,
                 integration_table.c.z_init_serial == z_init.store_id,
                 integration_table.c.validated == True,
-                integration_table.c.atol_serial == atol.store_id,
                 integration_table.c.rtol_serial == rtol.store_id,
                 # the parent's key includes its break-point policy, read from the compute target's
                 # single declaration exactly as the parent's build() reads it (prompt 05c, P3)
@@ -876,6 +875,11 @@ class sqla_TkNumericValue_factory(SQLAFactoryBase):
                 == TkNumericIntegration.BREAK_POINT_KIND,
             )
 
+            if atol is not None:
+                integration_query = integration_query.filter(
+                    integration_table.c.atol_serial == atol.store_id
+                )
+
             count = 0
             for tag in tags:
                 tag: store_tag
```

Result: `Ran 287`, `failures=1`. Only
`TestTolerancesRequired.test_each_numeric_read_raises_without_each_tolerance`, subtest
`read='TkNumericValue._build_impl_model', omitted='atol'`, fails, with
`AssertionError: KeyError not raised`. The other five subtests still pass.

**(vi) remove one entry from the `read_batch` pin:**

```diff
diff --git a/Datastore/tests/test_value_reads_parent_key.py b/Datastore/tests/test_value_reads_parent_key.py
index 9ca7463..2c44507 100644
--- a/Datastore/tests/test_value_reads_parent_key.py
+++ b/Datastore/tests/test_value_reads_parent_key.py
@@ -803,7 +803,6 @@ class TestResponseFlagsAreRead(_StoreCase):
 READ_BATCH_ROUND_TRIPS = {
     "GkNumericIntegration.py:sqla_GkNumericValue_factory": "Datastore.tests.test_value_reads_parent_key",
     "GkWKBIntegration.py:sqla_GkWKBValue_factory": "Datastore.tests.test_value_reads_parent_key",
-    "TkWKBIntegration.py:sqla_TkWKBValue_factory": "Datastore.tests.test_value_reads_parent_key",
     "QuadSourceIntegral.py:sqla_QuadSourceIntegral_factory": "Datastore.tests.test_integral_read_batch",
     "OneLoopIntegral.py:sqla_OneLoopIntegral_factory": "Datastore.tests.test_integral_read_batch",
 }
```

Result: `Ran 287`, `failures=1`. Only
`TestEveryReadBatchHasARoundTrip.test_every_read_batch_has_a_round_trip` fails:
`AssertionError: Items in the second set but not the first: 'TkWKBIntegration.py:sqla_TkWKBValue_factory' : the set of read_batch functions under ObjectFactories/ has changed. …`.

**Replay from this file, before committing.** This was done with every fix staged and every other
section of this file final. The six fenced `diff` blocks were extracted from this file into a
scratch directory outside the repository. Each was byte-identical (`cmp`) to the `git diff`
captured while its mutation was applied. For each: `git apply --check` (rc 0), `git apply`
(rc 0), the whole Datastore suite, `git apply -R` (rc 0). Each reproduced exactly the result
recorded above: (i) `failures=1, errors=1`, (ii) `failures=4`, (iii) `errors=2`, (iv) `failures=2`,
(v) `failures=1`, (vi) `failures=1`, with the same tests and subtests. The unstaged diff was empty
after every revert. Only this paragraph was written after that replay. The diff blocks were
re-extracted from the file with this paragraph in place and are byte-identical to those replayed.
The replay from the committed file, after the commit, is reported to the orchestrator, not
described here.

## Observations not acted on

1. **The refusal is decided on the rows returned** (for 09's K6). §2 says to decide it by selecting
   the parent serial into the row query, and that is what is done. When `z` is given, as
   `main.py` always gives it, a second matching parent that has **no value row at that `z`** does
   not appear in the rows, so it is not detected. The read returns the one parent that does, with
   no refusal. 09's K6 plans to read the `GkSource` parent set "by the same `read_batch` the stage
   already uses", relying on this refusal. If 09 needs "every parent matching the key", not "every
   parent contributing a value at this `z`", it must decide that on the parent query. Carried to
   the board for 07/09.
2. **`TkNumericValue.build`'s dispatch** (and `GkNumericValue.build`'s) decides the model path on
   `model`, `k` and `z_init` (`z_source` for Gk) only. The tolerances are now required too, so a
   payload with the three but no `atol` passes the dispatch and fails inside `_build_impl_model`
   with `KeyError: 'atol'`, not with the dispatch's own `RuntimeError`. Loud either way, and no
   caller outside the tests reaches either `build`'s model path. Separately, the comment in
   `test_read_path_mechanics.py`'s `test_tk_numeric_value_build_impl_model` still says
   `_build_impl_model` "reads `z_source`" (at `TkNumericIntegration.py:853`), and that payload
   still carries a now-unused `z_source`. Both are outside the permitted edits (only the payload's
   tolerances were), so they are left. The test is correct: `z_init` is present. Carried to the
   board for 07/09.
3. **The read-back guard cannot see a read in a comprehension's element** →
   `[05c-read-back-guard-cannot-see-reads-in-a-comprehension]`, opened on this campaign's board
   §3.2.
   - `analyse_build` walks nodes in pre-order. For `[f(row.x) for row in rows]` it visits the
     element `row.x` before the `comprehension` node that binds `row`. So the read is charged to
     an earlier binding of the same name, or, with none, dropped. Found through deviation 6.
   - Measured on this tree (an `ast` walk for comprehensions whose element reads an attribute of
     their own target), comprehension row reads the guard does not check:
     - `redshift.py:113` and `wavenumber.py:112` (`read_table`);
     - `BackgroundModel.py:401` (`build`, inside `KNOWN_BLIND`'s function);
     - the seven `invalid_serials = [nv.serial for nv in not_validated]` in `validate_on_startup`.
   - I checked the two `read_table`s by hand: each selects every column it reads. The seven
     `validate_on_startup` reads take `serial`, which their queries select; their `for` loops over
     the same rows read `serial` and are in `EXPECTED_COVERAGE`. **So nothing is missed today.**
     The guard's docstring does not declare the hole, and no pin would catch a future defect in
     one of these shapes.
   - The fix is the guard's, and not this prompt's.
4. **`probe_05`'s script now behaves differently on this tree** (callers table). Its `.out` is the
   record of the audit's tree, and both are left untouched, per §1.

## State handed to the next prompt

- **Prompt 05 as a whole** (05a–05c) is the orchestrator's to mark landed in README §2 after
  review. P9 is closed on this board in this commit.
- **The guard's pins, unchanged by 05c:** `EXPECTED_COVERAGE` 44, `KNOWN_BLIND` 1,
  `KNOWN_UNRESOLVED_TABLES` 9, `KNOWN_UNCHECKED_INSERTS` 13, `KNOWN_BLIND_COLUMNS` 4.
  `KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS` are empty.
- **For 07.** The WKB value reads take an optional `z_init` matched by the parents' absolute
  `DEFAULT_FLOAT_PRECISION` expression, in four places: `GkWKBValue._build_impl_model` /
  `read_batch` and `TkWKBValue._build_impl_model` / `read_batch`. When 07 replaces `z_init` in the
  key by `numeric_serial`, those four filters are what it replaces. The refusal keys on the WKB
  parent serial (`table.c.wkb_serial`), and stays valid under 07's key.
  `TestZInitSelects` and `TestSecondParentIsRefused.test_*_two_z_inits` in the new module will
  need to follow 07's key.
- **For 09.** Observation 1.
- **Names.** Nothing new is shared. `READ_BATCH_ROUND_TRIPS` is the P9 pin a new `read_batch` must
  join. The module's store helpers are local.
- **Baselines** after this commit: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39,
  Datastore **287**, LiouvilleGreen 148 (1 skipped), RunRegistry 199.
