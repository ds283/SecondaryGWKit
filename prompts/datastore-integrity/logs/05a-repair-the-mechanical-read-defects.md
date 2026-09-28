# Log 05a — Repair the mechanical read-path defects: names, an argument, tags, raises

**Prompt:** [`prompts/datastore-integrity/05a-repair-the-mechanical-read-defects.md`](../05a-repair-the-mechanical-read-defects.md)
**Commit:** *(this commit)* — "Repair the mechanical read-path defects: names, an argument, tags, raises"
**Model:** Claude Sonnet 5
**Date:** 2026-09-28
**Result:** COMPLETE. P6 and P7 shipped, and P9's share of this prompt (round-trip tests for the
sites P6/P7 touch). No §6 stop condition was met: no fix moved a guard pin beyond the two struck
entries the prompt names, no existing test failed, and every fix was exactly the rename, default
or wrapped exception it looked like. `KNOWN_UNFIXED` is empty; `KNOWN_UNFIXED_COLUMNS` holds only
`OneLoopIntegral`'s `validated` entry (P8, 05b). `Datastore` rises from 237 to 257 (the 20 tests of
the new `test_read_path_mechanics.py`; no other suite's count moves).

## What shipped

**P6 — names and the argument.**
- `WKB_criterio` → `WKB_criterion` in `sqla_GkWKBValue_factory._build_impl_serial`
  (`GkWKBIntegration.py:947`) and `sqla_TkWKBValue_factory._build_impl_serial`
  (`TkWKBIntegration.py:1001`).
- `table.c.z_serial` → `table.c.z_source_serial` in `sqla_GkSourceValue_factory._build_impl_serial`'s
  filter (`GkSource.py:766`); the insert two lines below it already wrote `z_source_serial`, so this
  was the query and the write disagreeing about their own table's column name.
- `quadsource_table.c.model_id` → `quadsource_table.c.model_serial` in
  `sqla_QuadSourceValue_factory._build_impl_model` (`QuadSource.py:672`).
- `friction=row.friction` added to `TkWKBValue.read_batch`'s `make_obj` (`TkWKBIntegration.py`),
  beside the other four required constructor arguments the row already supplied. Checked, per the
  prompt: `TkWKBValue.__init__`'s only parameters with no default are `store_id`, `z`, `H_ratio`,
  `theta_mod_2pi`, `theta_div_2pi` and `friction` — the first four `make_obj` already passed, so
  `friction` was the only omission; no other required argument was missing.

**P7 — tags and raises.**
- Nine sites' `tags: Optional[List[store_tag]] = payload.get("tags", None)` became
  `tags: List[store_tag] = payload.get("tags", [])`:
  `GkNumericIntegration.py` (`GkNumericValue._build_impl_model`, `.read_batch`),
  `GkSource.py` (`GkSourceValue._build_impl_model`),
  `GkWKBIntegration.py` (`GkWKBValue._build_impl_model`, `.read_batch`),
  `TkNumericIntegration.py` (`TkNumericValue._build_impl_model`),
  `QuadSource.py` (`QuadSourceValue._build_impl_model`),
  `TkWKBIntegration.py` (`TkWKBValue._build_impl_model`, `.read_batch`).
- `raise ("…")` → `raise RuntimeError("…")`, message unchanged, at `GkSource.py`
  (`GkSourceValue._build_impl_serial`), `GkWKBIntegration.py` (`GkWKBValue._build_impl_serial`) and
  `TkWKBIntegration.py` (`TkWKBValue._build_impl_serial`). A sweep of `Datastore/SQL/ObjectFactories/`
  for any other `raise (` or `raise "` (`grep -rn 'raise (\|raise "'`) found only these three, both
  before and after the fix.

**The recount, and why it differs from the audit's eleven.** The prompt's own §2 already recounted
to nine (excluding `OneLoopIntegral.py`, 05b's under D3) when it was written, and this prompt's
recount confirms it: `grep -rn 'payload\.get("tags", None)' Datastore/SQL/ObjectFactories/` and,
independently, `grep -rn "Optional\[List\[store_tag\]\]" Datastore/SQL/ObjectFactories/` both return
exactly the same ten lines, one per site — the nine above plus `OneLoopIntegral.py:284`. No
eleventh site exists in this tree under either search. The audit's §3.11 count of eleven cannot be
reconciled from the current tree by any variant of this grep tried (case, quoting, or matching the
type annotation instead of the default); the likely explanation is that the audit counted two sites
since fixed by an unrelated change, or miscounted, but nothing in the git history between the
audit's writing (2026-09-25) and this commit touches a `payload.get("tags", ...)` site, so no
specific prior count of eleven could be reconstructed. Recorded here rather than chased further,
per the prompt's own instruction to recount and explain the difference, not to reconcile it.

**The guard's pins.** `Datastore/tests/test_factory_select_columns.py`:
- `KNOWN_UNFIXED` struck to `{}` (both `WKB_criterio` entries), with the comment above it rewritten
  to say the fix landed here and the dict is empty.
- `KNOWN_UNFIXED_COLUMNS` struck from three entries to one: the `z_serial` (`GkSource.py`) and
  `model_id` (`QuadSource.py`) entries removed; `OneLoopIntegral`'s `validated` insert-key entry
  (P8) is untouched. The comment above the dict rewritten to say what remains and who owns it.
- **No other pin moved.** `EXPECTED_COVERAGE`, `KNOWN_BLIND`, `KNOWN_UNRESOLVED_TABLES`,
  `KNOWN_UNCHECKED_INSERTS` and `KNOWN_BLIND_COLUMNS` are byte-identical to prompt 04b's committed
  values — checked by running the full `TestFactorySelectColumns` / `TestColumnExistence` suite
  (all nine tests) against the fixed tree: every test that pins one of those five structures passed
  unchanged. This is what the board's simulation (§1, "05a's and 05b's pin movements were simulated
  against the guard on a scratch copy of the factories … 05a moves nothing but the struck entries")
  predicted, and this commit confirms it against the real fix rather than a simulated one.

## Deviations from the prompt

No `STRUCTURALLY REQUIRED` or `UNINTENDED DRIFT` deviations.

1. **IMPLEMENTATION CHOICE — the new test module's schema helper is written from scratch in
   `Datastore/tests/test_read_path_mechanics.py` rather than importing
   `docs/datastore-integrity-audit/lookup_keys/common.py`.** The prompt's §3 says to build the
   schema "as `common.py` does" — read as naming the construction
   (`build_schema(MetaData(), Datastore.SQL.Datastore._factories)` over an in-memory SQLite engine),
   not as an instruction to import that module. `common.py` is written for the audit's own probes
   (a `sys.path.insert` script-running convention, not a package-importable module) and the prompt's
   own §2 forbids editing it; every other round-trip test module in `Datastore/tests/`
   (`test_backgroundmodelvalue_roundtrip.py`) likewise builds its own schema helper rather than
   importing from `docs/`. The helper here (`Store`) is a direct, independent re-implementation of
   `common.py`'s `Store` class, and the row-shape helpers (`_gk_wkb_parent`, `_tk_wkb_parent`,
   `_gk_numeric_parent`, `_tk_numeric_parent`) follow the field names `probe_05_value_lookups_and_read_batch.py`'s
   equivalent helpers use, which are already verified against the real schema by that probe's own
   `.out`.
2. **IMPLEMENTATION CHOICE — the "tags omitted" round trip (§3 item 5) uses a lighter-weight store
   than the other tests', for the two numeric-value sites.** `TestTagsOmittedDefaultToEmpty` reuses
   the `GkWKBValue`/`TkWKBValue`/`QuadSourceValue`/`GkSourceValue` parent rows the other test classes
   already establish are correct, but for `GkNumericValue`/`TkNumericValue` (not otherwise exercised
   by this prompt's other items) it inserts a minimal parent+value row rather than a full production
   shape, since the equivalence being checked — omitted `tags` behaves exactly as `tags=[]` — holds
   regardless of what the row contains, and the assertion compares the two calls' own results to
   each other rather than to an expected value.
3. **IMPLEMENTATION CHOICE — `TkNumericValue`'s tags-omission test supplies both `z_init` and
   `z_source` in its payload.** Not a defect this prompt fixes: `sqla_TkNumericValue_factory.build()`
   dispatches to the model path on `payload.get("z_init", None)`, but `_build_impl_model` then reads
   `payload["z_source"]` internally (`TkNumericIntegration.py:853`) — the same confusing key
   mismatch `probe_05_value_lookups_and_read_batch.py`'s `tk_payload2` works around. Both keys carry
   the same `redshift` object in the test so the model path is reached at all; noted in the test's
   own comment rather than silently worked around.

## Verification performed

**The nine sites and the three raises**, by direct `grep` before and after the fix (shown above).

**The guard**, run in full both before mutation and after: `TestFactorySelectColumns` (4 tests) and
`TestColumnExistence` (5 tests), all pass against the fixed tree — `KNOWN_UNFIXED` and
`KNOWN_UNFIXED_COLUMNS` match exactly what the guard reports, `EXPECTED_COVERAGE`/`KNOWN_BLIND`/
`KNOWN_UNRESOLVED_TABLES`/`KNOWN_UNCHECKED_INSERTS` are unmoved.

**The new module**, 20 tests, all against the real, unmodified `build()` / `read_batch` /
`_build_impl_*` code, no Ray, no store, schema built in memory:
- `TestWKBCriterionSpelling` (2): `GkWKBValue.build()` and `TkWKBValue.build()` by `wkb_serial`
  against a stored row return that row's `WKB_criterion`, a value (0.123456789 / 0.987654321)
  distinct from every other numeric column on the row. Only `_build_impl_serial`'s dispatch path is
  exercised (payload carries `wkb_serial`, no `H_ratio`/`theta_mod_2pi`/`theta_div_2pi`/`friction`,
  so the consistency checks below the lookup are skipped rather than needing a matching payload —
  `WKB_criterion` itself is never compared against a payload value, only overwritten from the stored
  row, in both factories).
- `TestGkSourceValueZSourceSerial` (2): two `GkSourceValue` rows under the same `parent_serial` at
  different `z_source_serial`; `build()` at one `z_source` returns that row's `G` (11.0) and at the
  other returns the other row's `G` (22.0) — proving the filter distinguishes by `z_source_serial`
  rather than either failing outright (the pre-fix behaviour: `table.c.z_serial` raised
  `AttributeError` at expression-build time, before any row was even looked up) or accidentally
  matching the wrong row. `G` is read via `obj.numeric.G` (`GkSourceValue`'s numeric fields are
  grouped under a `numeric` property, unlike the other value classes' flat attributes).
- `TestQuadSourceValueModelSerial` (2): two `QuadSource` parents under different `model_serial`,
  each with one `QuadSourceValue` row; `build()` under model 1 returns model 1's row
  (`source`/`undiff`/`diff` = 1.0/2.0/3.0) and under an unrelated model 3 returns the "not found"
  empty object (`obj.available` is `False`) rather than raising or hitting the wrong row.
- `TestTkWKBValueFriction` (2): two `TkWKBIntegration` parents (one tagged, one not), each with one
  `TkWKBValue` row at a distinct `friction`. `read_batch` with `tags=[]` returns both rows, with
  frictions `[0.25, 0.77]` matching the stored values; `read_batch` with the tag returns exactly the
  tagged row, with its own friction (0.25). This is P9's round-trip for this `read_batch`.
- `TestTagsOmittedDefaultToEmpty` (9): each of the nine sites called once with no `tags` key and
  once with `tags=[]`, against the same store, and the two calls' results compared —
  `store_id`/`available` for the `_build_impl_model` sites, plus the value each site's earlier test
  already establishes matters (`G`, `T`, `WKB_criterion`, `friction`, `source`) for `read_batch`.
  All nine pairs agree.
- `TestStringRaisesBecomeRuntimeError` (3): each of the three sites, called with no stored row and
  no data payload, raises `RuntimeError` (not `TypeError`) carrying its message (`assertIn` on both
  the factory name and "result was not found").

**Full suites**, README §7's command, serially (ComputeTargets alone):

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 552 OK |
| CosmologyModels | 39 OK |
| Datastore | 257 OK (237 + 20) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

All six match board §5.3 except Datastore, which rises by exactly the 20 tests added.
`./venv/bin/python -m black --check` is clean on every file this commit touches.

`git status --short` and `git diff --cached` were clean before starting (`HEAD` at
`fd6205ec355956c6b4ba3e57f89059a8ae4772d9`) and, after the deliberate-breakage record below (each
mutation applied and reverted in turn), the working tree matched the staged fix exactly before this
commit was made.

## The deliberate-breakage record

Each diff below is exactly as applied — `git diff` (unstaged, against the index this commit's fix
was staged into with `git add`) taken **while the mutation was applied**, pasted unaltered — with
the resulting test outcome, then reverted (`git checkout --`) before the next. All six were
re-checked with `git apply --check` from files extracted verbatim from this **committed** log (see
below), and each applies, reproduces the recorded failure, and reverts (`git apply -R`) exactly as
recorded here.

**(i) put `WKB_criterio` back in `TkWKBValue._build_impl_serial`:**

```diff
diff --git a/Datastore/SQL/ObjectFactories/TkWKBIntegration.py b/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
index b91bae4..aee65b1 100644
--- a/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
+++ b/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
@@ -998,7 +998,7 @@ class sqla_TkWKBValue_factory(SQLAFactoryBase):
         else:
             store_id = row_data.serial
             omega_WKB_sq = row_data.omega_WKB_sq
-            WKB_criterion = row_data.WKB_criterion
+            WKB_criterion = row_data.WKB_criterio
 
             T_WKB = row_data.T_WKB
 
```

Result: both test 1 and the guard proper fail, as the prompt's §4(i) requires.
`test_tk_wkb_value_build_by_serial_returns_WKB_criterion` errors with
`AttributeError: Could not locate column in row for column 'WKB_criterio'`;
`test_build_reads_only_columns_its_select_requests` fails, reporting
`{'TkWKBIntegration.py:sqla_TkWKBValue_factory': ['WKB_criterio']} != {}`. Reverted;
`git status --short` returned to the pre-mutation staged state.

**(ii) put `z_serial` back in `GkSourceValue._build_impl_serial`:**

```diff
diff --git a/Datastore/SQL/ObjectFactories/GkSource.py b/Datastore/SQL/ObjectFactories/GkSource.py
index 84c3fbe..be51291 100644
--- a/Datastore/SQL/ObjectFactories/GkSource.py
+++ b/Datastore/SQL/ObjectFactories/GkSource.py
@@ -763,7 +763,7 @@ class sqla_GkSourceValue_factory(SQLAFactoryBase):
                     table.c.analytic_Gprime_rad,
                 ).filter(
                     table.c.parent_serial == parent_serial,
-                    table.c.z_source_serial == z_source.store_id,
+                    table.c.z_serial == z_source.store_id,
                 )
             ).one_or_none()
         except MultipleResultsFound as e:
```

Result: both test 2 and G2 fail, as the prompt's §4(ii) requires. Both
`TestGkSourceValueZSourceSerial` tests error with `AttributeError: z_serial` (raised at
expression-build time, before any row is looked up);
`test_column_references_resolve_against_schema` fails, reporting
`('GkSource.py', 'sqla_GkSourceValue_factory', '_build_impl_serial', 'GkSourceValue', 'z_serial')`
as a column the schema does not have. Reverted.

**(iii) remove `friction=row.friction` from `TkWKBValue.read_batch`'s `make_obj`:**

```diff
diff --git a/Datastore/SQL/ObjectFactories/TkWKBIntegration.py b/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
index b91bae4..b7537ed 100644
--- a/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
+++ b/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
@@ -1266,7 +1266,6 @@ class sqla_TkWKBValue_factory(SQLAFactoryBase):
                 H_ratio=row.H_ratio,
                 theta_mod_2pi=row.theta_mod_2pi,
                 theta_div_2pi=row.theta_div_2pi,
-                friction=row.friction,
                 omega_WKB_sq=row.omega_WKB_sq,
                 WKB_criterion=row.WKB_criterion,
                 T_WKB=row.T_WKB,
```

Result: both `TestTkWKBValueFriction` tests error with
`TypeError: TkWKBValue.__init__() missing 1 required positional argument: 'friction'`. Reverted.

**(iv) put `None` back as one tags default** (`QuadSourceValue._build_impl_model`):

```diff
diff --git a/Datastore/SQL/ObjectFactories/QuadSource.py b/Datastore/SQL/ObjectFactories/QuadSource.py
index 00db7ca..08e08fc 100644
--- a/Datastore/SQL/ObjectFactories/QuadSource.py
+++ b/Datastore/SQL/ObjectFactories/QuadSource.py
@@ -663,7 +663,7 @@ class sqla_QuadSourceValue_factory(SQLAFactoryBase):
         q: wavenumber_exit_time = payload["q"]
         r: wavenumber_exit_time = payload["r"]
 
-        tags: List[store_tag] = payload.get("tags", [])
+        tags: Optional[List[store_tag]] = payload.get("tags", None)
 
         quadsource_table = tables["QuadSource"]
 
```

Result: `test_quad_source_value_build_impl_model` errors with
`TypeError: 'NoneType' object is not iterable` at `for tag in tags`. Reverted.

**(v) put the string raise back at one site** (`GkWKBValue._build_impl_serial`):

```diff
diff --git a/Datastore/SQL/ObjectFactories/GkWKBIntegration.py b/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
index 800283d..f0788ed 100644
--- a/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
+++ b/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
@@ -918,7 +918,7 @@ class sqla_GkWKBValue_factory(SQLAFactoryBase):
 
         if row_data is None:
             if not has_data:
-                raise RuntimeError(
+                raise (
                     "GkWKBValue().build(): result was not found in datastore, but a data payload was not provided"
                 )
 
```

Result: `test_gk_wkb_value_build_impl_serial` errors with
`TypeError: exceptions must derive from BaseException` (`assertRaises(RuntimeError)` does not catch
it). Reverted.

**(vi) restore one struck `KNOWN_UNFIXED_COLUMNS` entry without undoing its fix** (`GkSource.py`'s
`z_serial` entry, `GkSource.py` itself left fixed):

```diff
diff --git a/Datastore/tests/test_factory_select_columns.py b/Datastore/tests/test_factory_select_columns.py
index 4bcc6b5..edafea5 100644
--- a/Datastore/tests/test_factory_select_columns.py
+++ b/Datastore/tests/test_factory_select_columns.py
@@ -745,6 +745,15 @@ from Datastore.SQL.schema import build_schema  # noqa: E402
 ColumnProblem = Tuple[str, str, str, str, str]
 
 KNOWN_UNFIXED_COLUMNS: Set[ColumnProblem] = {
+    # mutation (vi): restoring this struck entry without undoing GkSource.py's fix must fail
+    # test_known_unfixed_columns_are_still_present -- the column problem it names no longer exists.
+    (
+        "GkSource.py",
+        "sqla_GkSourceValue_factory",
+        "_build_impl_serial",
+        "GkSourceValue",
+        "z_serial",
+    ),
     # P8: sqla_OneLoopIntegral_factory.store writes an insert key "validated"
     # (OneLoopIntegral.py:222) to a table with no such column -- D3: the mechanical defects only,
     # the rest logged for when OneLoopIntegral is actually built.
```

Result: `test_known_unfixed_columns_are_still_present` fails, reporting that
`('GkSource.py', 'sqla_GkSourceValue_factory', '_build_impl_serial', 'GkSourceValue', 'z_serial')`
is "not found" among the column problems the guard currently reports (since the fix is in place,
the guard no longer reports it) — exactly the "must fail" the prompt's §4(vi) asks for. Reverted.

Each mutation was applied alone, against the tree with this commit's fix already staged (`git add`),
so `git diff` (unstaged) shows exactly that one mutation; `git diff --cached` was unaffected
throughout, and `git status --short` returned to the same staged-only state after every
`git checkout --`.

**Replay from the committed log.** After committing, each of the six diffs above was extracted
verbatim from this file (the committed log) into its own file under a scratch directory outside the
repository, and for each: `git apply --check`, `git apply`, the relevant test(s), and `git apply -R`.
All six applied cleanly on the first try, reproduced the failure recorded above exactly, and
reverted the tree to clean (`git status --short` empty) afterwards.

## Observations not acted on

1. **The audit's §3.11 count of eleven `tags` sites cannot be reconciled with the current tree's
   ten** (nine here, one `OneLoopIntegral.py`'s, 05b) — see "The recount" above. Not chased further:
   this prompt's job was to recount and explain the difference, not to reconstruct an audit-time
   count that predates every commit since.
2. **`GkSourceValue` and `QuadSourceValue`'s "not found" branches return an empty object
   (`store_id=None`) rather than raising**, unlike the two WKB value factories and `GkSourceValue`'s
   own `_build_impl_serial`, which raise when no data payload is given either. This asymmetry is a
   design property of each factory's dispatch (serial-keyed lookups raise; model-keyed lookups
   return an unavailable object for the caller to check with `.available`), not a defect this
   prompt's charter (P6/P7) touches, and it predates this commit.
3. **`GkSourceValue`'s numeric fields are grouped under a `.numeric` property** (`.numeric.G`,
   `.numeric.Gprime`) while every other value class here exposes flat attributes (`.G`, `.friction`,
   `.WKB_criterion`, …). Noted because it cost a round of test failures while writing
   `TestGkSourceValueZSourceSerial`; not a defect, just an inconsistency in the ComputeTargets
   classes' own shape, out of this prompt's scope.

## State handed to the next prompt

- **For 05b.** `KNOWN_UNFIXED_COLUMNS` now holds only `OneLoopIntegral`'s `validated` entry (P8),
  which 05b empties. `KNOWN_UNFIXED` is empty and stays that way. The coverage pins
  (`EXPECTED_COVERAGE`, `KNOWN_BLIND`, `KNOWN_UNRESOLVED_TABLES`, `KNOWN_UNCHECKED_INSERTS`) are
  unmoved by this prompt, exactly as the board's simulation predicted; 05b's own pin movements
  (one coverage entry added, one blind entry and three unresolved-table pairs removed, per board §1)
  start from the values this commit leaves.
- **Names.** No new shared name is introduced; `Store`, `_ModelProxy`, `_Ref`, `_z` are local to
  `test_read_path_mechanics.py` and not part of any campaign interface.
- **Baselines** after this commit: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39,
  Datastore **257** (237 + 20), LiouvilleGreen 148 (1 skipped), RunRegistry 199.
