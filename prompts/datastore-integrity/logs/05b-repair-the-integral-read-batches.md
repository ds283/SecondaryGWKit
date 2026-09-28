# Log 05b — Repair the integral `read_batch`es: the tagged join, the exit times' serials, `OneLoopIntegral`

**Prompt:** [`prompts/datastore-integrity/05b-repair-the-integral-read-batches.md`](../05b-repair-the-integral-read-batches.md)
**Commit:** *(this commit)* — "Repair the integral read_batch join, exit-time serials and OneLoopIntegral"
**Model:** Claude Sonnet 5
**Date:** 2026-09-28
**Result:** COMPLETE. P1, P2 and P8 shipped, and P9's share of this prompt (round-trip tests for
the two `read_batch`es it touches). No §6 stop condition was met: every guard pin movement is
exactly the set §2 lists, each shown forced by the failing assertion it produced before the pin
was edited; no existing test failed; `source_serial` was left untouched everywhere, as D3 requires.
`KNOWN_UNFIXED_COLUMNS` is empty. `Datastore` rises from 257 to 267 (the 10 tests of the new
`test_integral_read_batch.py`; no other suite's count moves).

## What shipped

**P1 — the tagged join.** `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py`'s
`read_batch` joined `tab.c.parent_serial == query.c.serial`; `query.c` makes SQLAlchemy name an
anonymous subquery outside the compiled `FROM` clause, and SQLite raises
`OperationalError: no such column: anon_1.serial` on every tagged call (`probe_01_tagged_read_batch.py`,
confirmed again by mutation (i) below). Fixed to `tab.c.parent_serial == table.c.serial`, as
`probe_01b_fix_shape.py` verified in memory. The same join in `OneLoopIntegral.py`'s `read_batch`
is P8's, below — it moved from a module-level function into the class in the same edit, carrying
the same fix.

**P2 — the exit times' own serials.** `QuadSourceIntegral.read_batch` selected
`q_wavenumber_tab.c.serial.label("q_serial")` / `r_wavenumber_tab.c.serial.label("r_serial")` —
the *wavenumber* rows' serials — and used them correctly for the `q`/`r` wavenumbers, but also,
wrongly, as the `q_exit`/`r_exit` `wavenumber_exit_time` objects' `store_id` (audit §3.2). Fixed by
adding two selected columns, `q_exit_tab.c.serial.label("q_exit_serial")` and
`r_exit_tab.c.serial.label("r_exit_serial")`, and using those instead in `q_exit_payload`/
`r_exit_payload`'s `"store_id"` keys. The wavenumbers keep `q_serial`/`r_serial` unchanged. **The
rest of `make_object` was checked for any other value read off the wrong table's serial** (the
prompt's own instruction): every other `store_id` — `q`/`r` wavenumbers (`row.q_serial`/
`row.r_serial`, from `q_wavenumber_tab`/`r_wavenumber_tab`), the exit times' `atol`/`rtol`
(`row.q_atol_serial`/`row.q_rtol_serial` etc., which are the *exit time* row's own foreign-key
values into `tolerance`, not a wavenumber's), `z_response`/`z_source_max` (`row.z_response_serial`/
`row.z_source_max_serial`, the QSI row's own foreign keys), and the `QuadSourceIntegral` object
itself (`row.serial`) — reads from the table whose serial it names. **No second instance of P2's
defect was found.**

**P8 — `OneLoopIntegral`'s factory, bounded by D3's list and nothing more.**
- `build()` now joins `tables["OneLoopIntegral_tags"]` (was `tables["QuadSourceIntegral_tags"]`,
  audit §3.12, `probe_06_oneloop_tags_and_redshift_zero.py`).
- `read_batch` is now a `@staticmethod` of `sqla_OneLoopIntegral_factory`, carrying P1's join fix,
  tags defaulting to `[]` (was `None`), and `return [make_object(row) for row in row_data]`. The
  `# TODO: not yet complete` comment is removed — the thing it named (no return statement) is
  fixed; the deferred `source_serial` defect is unrelated to that comment and is recorded on the
  board instead (D3; test 5 below keeps it live). The module-level function of the same name is
  deleted outright, not left behind: `Datastore.object_read_batch` calls `factory.read_batch`
  where `factory` is the class object itself (`Datastore.SQL.Datastore._factories["OneLoopIntegral"]
  = sqla_OneLoopIntegral_factory`, the class, not an instance), so only a class attribute is ever
  reachable — the module-level function was permanently dead code, confirmed by
  `test_read_path_mechanics.py`'s antecedent probe (`hasattr(sqla_OneLoopIntegral_factory,
  "read_batch")` was `False` before this commit).
- `"validated": False` is removed from `store()`'s insert (the table has no such column;
  SQLAlchemy silently dropped it, which is what `KNOWN_UNFIXED_COLUMNS` was carrying).
- `inventory_records`'s comment, which called the tag join "not fixed here"
  (`[00-oneloop-lookup-joins-the-wrong-tag-table]`), is corrected to say it is fixed.

**Left alone, exactly as D3 requires:** `source_serial` does not appear anywhere in this commit —
not in the compute target, not in `build()`'s or `read_batch()`'s payload, not in the table's
columns. `[00-oneloop-factory-cannot-round-trip-a-row]` stays open for its `source_serial` half;
test 5 of the new module constructs the *real* `ComputeTargets.OneLoopIntegral` from what
`read_batch()` passes and asserts it still raises `KeyError: 'source_serial'`, so the deferred
defect cannot silently disappear under a future edit without that test forcing a decision.

**The guard's pins**, each forced by the assertion `test_factory_select_columns.py` gave *before*
the pin was edited (captured by running the guard's nine tests against the tree with only the
three factory files fixed, before touching the test module at all):

- `KNOWN_UNFIXED_COLUMNS` struck to `set()` (was one entry, `OneLoopIntegral.py`'s `validated`
  insert key). Forced by:
  ```
  FAIL: test_known_unfixed_columns_are_still_present
  AssertionError: ('OneLoopIntegral.py', 'sqla_OneLoopIntegral_factory', 'store',
  'OneLoopIntegral [insert key]', 'validated') not found in set() : ... no longer names a column
  absent from its table: if it has been fixed, remove its KNOWN_UNFIXED_COLUMNS entry
  ```
  (`test_column_references_resolve_against_schema` also failed, confirming the column problem no
  longer exists — see "Verification performed".)
- `KNOWN_BLIND`: the `OneLoopIntegral.py:<module-level>.read_batch.read_batch` entry is removed.
  Forced by:
  ```
  FAIL: test_blind_spots_are_the_declared_ones
  AssertionError: Items in the second set but not the first:
  'OneLoopIntegral.py:<module-level>.read_batch.read_batch' : the analyser cannot decide a row
  read it is not declared blind to. ...
  ```
- `EXPECTED_COVERAGE` gains exactly one entry,
  `"OneLoopIntegral.py:sqla_OneLoopIntegral_factory.read_batch.row_data"` (`make_object` kept its
  parameter name `row_data`). Forced by:
  ```
  FAIL: test_analyser_coverage_is_undiminished
  AssertionError: Items in the first set but not the second:
  'OneLoopIntegral.py:sqla_OneLoopIntegral_factory.read_batch.row_data' : the set of row reads
  this guard can decide has changed. ...
  ```
- `KNOWN_UNRESOLVED_TABLES` loses exactly three pairs: `QuadSourceIntegral.py:sqla_QuadSourceIntegral_factory.read_batch`
  base `query`, and the two `OneLoopIntegral.py:<module-level>.read_batch` pairs, bases `query` and
  `table` (the second gone because the function is now a method, and its own `table` parameter
  resolves through `cls_to_table` the way every other method's does). Forced by:
  ```
  FAIL: test_unresolved_table_sites_are_the_declared_ones
  AssertionError: Items in the second set but not the first:
  ('OneLoopIntegral.py:<module-level>.read_batch', 'query')
  ('OneLoopIntegral.py:<module-level>.read_batch', 'table')
  ('QuadSourceIntegral.py:sqla_QuadSourceIntegral_factory.read_batch', 'query') : a new (function,
  base) pair cannot be resolved to a schema table. ...
  ```

**No other pin moved.** `KNOWN_UNFIXED` (already empty since 05a), `KNOWN_BLIND_COLUMNS` and
`KNOWN_UNCHECKED_INSERTS` are byte-identical to prompt 05a's committed values — the full guard
suite (9 tests) passes unchanged against them after the four edits above, exactly the movements
the board's simulation (§1, "05a's and 05b's pin movements were simulated ... 05b adds one
coverage entry, removes one blind entry and three unresolved-table pairs") predicted.

## Deviations from the prompt

No `STRUCTURALLY REQUIRED` or `UNINTENDED DRIFT` deviations.

1. **IMPLEMENTATION CHOICE — the new test module's schema helper (`Store`) is written from scratch
   in `Datastore/tests/test_integral_read_batch.py` rather than importing
   `docs/datastore-integrity-audit/lookup_keys/common.py`.** Same reasoning 05a's log gave for its
   own module: the prompt's §3 names the *construction* ("as `common.py` does"), `common.py` is a
   script-running module the prompt's own §2 forbids editing and is not meant to be imported by
   `Datastore/tests/`, and every existing round-trip test module in that directory
   (`test_read_path_mechanics.py`, `test_backgroundmodelvalue_roundtrip.py`) already follows this
   pattern. The row-shape helpers (`_exit_time_row`, `_exit_time_obj`) match
   `probe_01_tagged_read_batch.py`'s `populate`/`exit_time` field-for-field, which that probe's own
   `.out` already verifies against the real schema.
2. **IMPLEMENTATION CHOICE — `test_integral_read_batch.py`'s `_ModelProxy`/`_Ref`/`_z` helpers
   duplicate `test_read_path_mechanics.py`'s rather than importing them.** Neither module exports
   these names as a shared interface (05a's own log: "`Store`, `_ModelProxy`, `_Ref`, `_z` are
   local ... and not part of any campaign interface"), and this prompt's file list does not include
   `test_read_path_mechanics.py`, so duplication rather than a shared import.
3. **IMPLEMENTATION CHOICE — the deferred-defect test (§3 item 5) is its own test class**
   (`TestOneLoopIntegralSourceSerialStaysBroken`) **rather than a method inside
   `TestOneLoopIntegralReadBatch`.** It deliberately does *not* patch `OneLoopIntegral` — every
   other test in the module does — so keeping it in its own class makes that asymmetry visible in
   the test list rather than requiring a reader to notice one method's absent `mock.patch`.

## Verification performed

**The factory fixes**, by direct inspection and `grep`: `query.c.serial` no longer appears in
either file; `q_exit_serial`/`r_exit_serial` are each selected exactly once and read exactly once;
`OneLoopIntegral.py` has exactly one `read_batch` (the class method) and no module-level function
of that name; `"validated"` does not appear in `store()`'s insert dict; `source_serial` does not
appear anywhere in either file's diff.

**The guard**, run in full (9 tests) both before any pin edit (four failures/errors, shown above,
each pasted from the actual run) and after (all nine pass) — `KNOWN_UNFIXED_COLUMNS` is empty,
`KNOWN_BLIND`/`EXPECTED_COVERAGE`/`KNOWN_UNRESOLVED_TABLES` match exactly what the guard's own
analyser computes from the fixed tree, and `KNOWN_UNFIXED`/`KNOWN_BLIND_COLUMNS`/
`KNOWN_UNCHECKED_INSERTS` are unmoved.

**The new module**, 10 tests, all against the real, unmodified `build()`/`read_batch()` code, no
Ray, no store, schema built in memory:
- `TestQuadSourceIntegralTaggedReadBatch` (3): probe_01's design — three `QuadSourceIntegral` rows
  tagged `{A,B}`, `{A}`, none; `tags=[]`/`[A]`/`[A,B]` return `{1,2,3}`/`{1,2}`/`{1}`, none raising.
- `TestQuadSourceIntegralExitTimeSerials` (1): the stored exit-time rows (serials 12, 13) are
  deliberately different from the wavenumber rows they point at (serials 2, 3) — stated in the
  test's own docstring, per the prompt's instruction, since without that separation P2's defect and
  its fix are indistinguishable. The returned object's `_q_exit.store_id`/`_r_exit.store_id` equal
  12/13 (the exit rows' own serials) and `.q.store_id`/`.r.store_id` equal 2/3 (the wavenumbers').
- `TestOneLoopIntegralBuildTags` (2): probe_06's two cases, via a `unittest.mock.patch`-scoped
  stand-in for `OneLoopIntegral` (README §5 rule 9) that only records constructor keyword
  arguments — a row tagged in `OneLoopIntegral_tags` is found; an untagged row whose serial (5)
  coincides with a `QuadSourceIntegral_tags.parent_serial` row is not matched.
- `TestOneLoopIntegralReadBatch` (3): reached as `sqla_OneLoopIntegral_factory.read_batch`, the
  exact attribute `Datastore.object_read_batch` looks up; `setUp` asserts that attribute exists
  before running anything, so a regression back to a module-level function fails loudly rather than
  masquerading as an `AttributeError` deep in a test body. Untagged returns both stored rows;
  tagged returns only the tagged one; omitting `tags` from the payload behaves exactly as
  `tags=[]`.
- `TestOneLoopIntegralSourceSerialStaysBroken` (1): constructing the **real**
  `ComputeTargets.OneLoopIntegral` (no patch) from what `read_batch()` passes raises
  `KeyError: 'source_serial'` — `[00-oneloop-factory-cannot-round-trip-a-row]`'s remaining half,
  deliberately left visible.

**Full suites**, README §7's command, serially (ComputeTargets alone):

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 552 OK |
| CosmologyModels | 39 OK |
| Datastore | 267 OK (257 + 10) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

All six match board §5.4 except Datastore, which rises by exactly the 10 tests added.
`./venv/bin/python -m black --check` is clean on every file this commit touches.

`git status --short` and `git diff --cached --stat` were clean before starting (`HEAD` at
`7ce8c66c...`, the 05a commit reviewed at `76e813b`). Before the deliberate-breakage record, this
commit's fix was staged with `git add`; each mutation below was applied unstaged, so `git diff`
shows exactly that one mutation, run against the full `Datastore` suite (not only the test the
mutation targets, per the board's review of 05a), then reverted with `git checkout --` before the
next. After the sixth revert, `git diff` was empty and `git diff --cached --stat` matched the
originally staged fix exactly, confirmed by re-running the full `Datastore` suite once more (267
OK) immediately before committing.

## The deliberate-breakage record

Each diff below is exactly as applied — `git diff` (unstaged, against the index this commit's fix
was staged into) taken **while the mutation was applied**, pasted unaltered — with the complete
list of failing/erroring tests the full `Datastore` suite reported (not only the named test), then
reverted (`git checkout --`) before the next.

**Before committing**, each of the six diffs below was extracted verbatim from **this log file, as
it stands here, about to be committed** into its own file under a scratch directory outside the
repository, and for each: `git apply --check`, `git apply`, the tests the mutation's own section
names, `git apply -R`. All six applied cleanly, reproduced the recorded result exactly, and
reverted the tree to clean (`git status --short` empty) afterwards. This replay is from the log
file's content before the commit exists, not a description of anything that happened after it —
see "State handed to the next prompt" for the second, post-commit replay the board's review of 05a
asked every subsequent prompt to perform and report, rather than describe here.

**(i) put `query.c.serial` back in `QuadSourceIntegral.read_batch`'s tag join:**

```diff
diff --git a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
index 927c836..2b9bfac 100644
--- a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
+++ b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
@@ -678,7 +678,7 @@ class sqla_QuadSourceIntegral_factory(SQLAFactoryBase):
             query = query.join(
                 tab,
                 and_(
-                    tab.c.parent_serial == table.c.serial,
+                    tab.c.parent_serial == query.c.serial,
                     tab.c.tag_serial == tag.store_id,
                 ),
             )
```

Result: `failures=1, errors=2`. `test_one_tag_returns_two` and `test_two_tags_returns_one`
(`TestQuadSourceIntegralTaggedReadBatch`) both error with
`sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) no such column: anon_1.serial` — the
`tags=[]` case in the same class does not error, because with no tag the join is never built.
`test_unresolved_table_sites_are_the_declared_ones` (G6, `TestColumnExistence`) also fails,
reporting `('QuadSourceIntegral.py:sqla_QuadSourceIntegral_factory.read_batch', 'query')` as a new
unresolved-table pair the pin does not declare. Reverted; tree returned to the staged-only state.

**(ii) use `row.q_serial` as the `q` exit time's `store_id` again:**

```diff
diff --git a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
index 927c836..dff6675 100644
--- a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
+++ b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
@@ -702,7 +702,7 @@ class sqla_QuadSourceIntegral_factory(SQLAFactoryBase):
             )
 
             q_exit_payload = {
-                "store_id": row.q_exit_serial,
+                "store_id": row.q_serial,
                 "z_exit": row.q_z_exit,
                 "compute_time": row.q_compute_time,
                 "stepping": row.q_stepping,
```

Result: `failures=1`. Only `test_exit_times_and_wavenumbers_carry_their_own_serials`
(`TestQuadSourceIntegralExitTimeSerials`) fails:
`AssertionError: 2 != 12` (`obj._q_exit.store_id` came back as the wavenumber's serial, 2, instead
of the exit row's own serial, 12). No collateral failure. Reverted.

**(iii) point `OneLoopIntegral.build` back at `QuadSourceIntegral_tags`:**

```diff
diff --git a/Datastore/SQL/ObjectFactories/OneLoopIntegral.py b/Datastore/SQL/ObjectFactories/OneLoopIntegral.py
index d57ca88..ba74133 100644
--- a/Datastore/SQL/ObjectFactories/OneLoopIntegral.py
+++ b/Datastore/SQL/ObjectFactories/OneLoopIntegral.py
@@ -138,7 +138,7 @@ class sqla_OneLoopIntegral_factory(SQLAFactoryBase):
         atol: tolerance = payload["atol"]
         rtol: tolerance = payload["rtol"]
 
-        tag_table = tables["OneLoopIntegral_tags"]
+        tag_table = tables["QuadSourceIntegral_tags"]
 
         query = sqla.select(
             table.c.serial,
```

Result: `failures=2`. Both `TestOneLoopIntegralBuildTags` tests fail as their own logic predicts:
`test_row_tagged_in_oneloop_tags_is_found` fails because the row tagged only in
`OneLoopIntegral_tags` is no longer found (`obj.store_id` is `None`, not `5`);
`test_untagged_row_coinciding_with_a_quadsourceintegral_tag_serial_is_not_matched` fails because
the untagged row is now falsely matched by the coincidental `QuadSourceIntegral_tags` row
(`obj.store_id` is `5`, not `None`) — probe_06's false hit, reproduced. No collateral failure.
Reverted.

**(iv) delete `read_batch`'s `return`:**

```diff
diff --git a/Datastore/SQL/ObjectFactories/OneLoopIntegral.py b/Datastore/SQL/ObjectFactories/OneLoopIntegral.py
index d57ca88..8f434df 100644
--- a/Datastore/SQL/ObjectFactories/OneLoopIntegral.py
+++ b/Datastore/SQL/ObjectFactories/OneLoopIntegral.py
@@ -335,7 +335,7 @@ class sqla_OneLoopIntegral_factory(SQLAFactoryBase):
             obj._deserialized = True
             return obj
 
-        return [make_object(row) for row in row_data]
+        [make_object(row) for row in row_data]
 
     @staticmethod
     def inventory_records(conn, table, tables, context):
```

Result: `errors=3`. All three `TestOneLoopIntegralReadBatch` tests error with
`TypeError: 'NoneType' object is not iterable` at the call site iterating `read_batch`'s return
value, since the function now falls off the end and returns `None`. No collateral failure.
Reverted.

**(v) put `"validated": False` back into `store()`:**

```diff
diff --git a/Datastore/SQL/ObjectFactories/OneLoopIntegral.py b/Datastore/SQL/ObjectFactories/OneLoopIntegral.py
index d57ca88..0a93e68 100644
--- a/Datastore/SQL/ObjectFactories/OneLoopIntegral.py
+++ b/Datastore/SQL/ObjectFactories/OneLoopIntegral.py
@@ -232,6 +232,7 @@ class sqla_OneLoopIntegral_factory(SQLAFactoryBase):
                 "metadata": (
                     json.dumps(obj.metadata) if obj._metadata is not None else None
                 ),
+                "validated": False,
             },
         )
 
```

Result: `failures=1`. G2, `test_column_references_resolve_against_schema`, fails as the prompt's
§4(v) requires: `('OneLoopIntegral.py', 'sqla_OneLoopIntegral_factory', 'store',
'OneLoopIntegral [insert key]', 'validated')` is reported as a column the schema does not have. No
test that actually runs `store()` exists in this tree (production never calls it in a test), so no
other test is affected. Reverted.

**(vi) put `None` back as `OneLoopIntegral.read_batch`'s tags default:**

```diff
diff --git a/Datastore/SQL/ObjectFactories/OneLoopIntegral.py b/Datastore/SQL/ObjectFactories/OneLoopIntegral.py
index d57ca88..ded0098 100644
--- a/Datastore/SQL/ObjectFactories/OneLoopIntegral.py
+++ b/Datastore/SQL/ObjectFactories/OneLoopIntegral.py
@@ -257,7 +257,7 @@ class sqla_OneLoopIntegral_factory(SQLAFactoryBase):
         atol: tolerance = payload["atol"]
         rtol: tolerance = payload["rtol"]
 
-        tags: List[store_tag] = payload.get("tags", [])
+        tags: Optional[List[store_tag]] = payload.get("tags", None)
 
         k_exit_table = tables["wavenumber_exit_time"].alias("k_exit_tab")
         k_table = tables["wavenumber"].alias("k_tab")
```

Result: `errors=1`. Only `test_tags_omitted_behaves_as_tags_equals_empty_list`
(`TestOneLoopIntegralReadBatch`) errors: `TypeError: 'NoneType' object is not iterable` at
`for tag in tags`, when the payload omits the key entirely. The other two tests in that class both
pass `tags=` explicitly, so they are unaffected. No collateral failure. Reverted.

Each mutation was applied alone, against the tree with this commit's fix already staged
(`git add`); `git diff` (unstaged) shows exactly that one mutation, `git diff --cached` was
unaffected throughout, and `git status --short` returned to the same staged-only state after every
`git checkout --`.

## Observations not acted on

1. **`OneLoopIntegralTagAssociation_factory.build` raises `NotImplementedError` unconditionally**,
   the same shape as every other tag-association factory in this tree (`QuadSourceIntegralTagAssociation_factory.build`
   is identical). Not a defect — nothing calls a tag-association factory's `build()`, only
   `add_tag`/`remove_tag` — and out of this prompt's D3-bounded scope in any case.
2. **`OneLoopIntegral.build`'s tag join reuses the same `count`/`tab` naming pattern as
   `QuadSourceIntegral.build`'s**, both walking `for tag in tags: tab = tag_table.alias(f"tag_{count}")`.
   Harmless duplication between two factories, not touched here (D3 bounds `OneLoopIntegral`'s
   factory to the mechanical defects §2 lists, and `QuadSourceIntegral.build` is explicitly out of
   scope, §2: "Do not change `QuadSourceIntegral.build`").
3. **`sqla_OneLoopIntegral_factory.read_batch`'s `k_exit_table`/`k_table` joins are built but never
   read from** in the `SELECT` list (only used to constrain the `FROM` via the wavenumber-exit
   chain, matching the pre-existing module-level function's shape exactly). Carried over unchanged
   from the original code, not a defect this prompt's charter (P8, D3) touches.

## State handed to the next prompt

- **For 05c.** The guard's pins after this commit: `KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS` are
  both empty (README §2 acceptance item 2). `EXPECTED_COVERAGE` (44 entries), `KNOWN_BLIND` (1
  entry, `BackgroundModelFactory.build.row_data`), `KNOWN_UNRESOLVED_TABLES` (9 pairs, all
  `subquery`), `KNOWN_UNCHECKED_INSERTS` (13 sites) and `KNOWN_BLIND_COLUMNS` (3 entries) are the
  values 05c's own pin movements (P3–P5, closing P9) start from.
- **`[00-oneloop-factory-cannot-round-trip-a-row]`** stays open, narrowed to its `source_serial`
  half only (this board's §3.2, updated in this commit); the `validated` half is closed. Whoever
  builds `OneLoopIntegral` must change `TestOneLoopIntegralSourceSerialStaysBroken` in this new
  module — its own docstring says so.
- **`[00-tagged-read-batch-joins-an-unselected-alias]`** and
  **`[00-oneloop-lookup-joins-the-wrong-tag-table]`** close on the store-fingerprint board's §4 in
  this commit, naming it by subject; their index rows are deleted.
- **Names.** No new shared name is introduced; `Store`, `_ModelProxy`, `_Ref`, `_z`,
  `_exit_time_row`, `_exit_time_obj`, `_OneLoopStandIn` are local to `test_integral_read_batch.py`
  and not part of any campaign interface.
- **Baselines** after this commit: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39,
  Datastore **267** (257 + 10), LiouvilleGreen 148 (1 skipped), RunRegistry 199.
