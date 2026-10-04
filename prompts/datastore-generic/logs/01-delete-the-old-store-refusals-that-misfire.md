# Log — prompt 01: delete the old-store refusals that misfire

**Subject:** *Delete the old-store refusals that misfire* · **Model:** Claude Sonnet 5.5 · **Date:**
2026-10-04 · **Base:** `6ea366e`, clean · **Result:** landed. The old-store handlers are gone from
`TkNumericIntegration`, `GkNumericIntegration`, `QCD_Cosmology` and `BackgroundModel`; a database
error reaches the caller as itself, and `BackgroundModel` always selects its grid identity and never
re-queries. `extract_common` refuses a store with no runs and names no older store. Eleven tests
deleted, thirteen added. None of the prompt's §5 stop conditions fired. Closes
`[00-old-store-refusals-misfire-on-any-database-error]`; narrows
`[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]`. Opens nothing. Index 108 → 107.
The commit is `8636491`.

## What shipped

- **`Datastore/SQL/ObjectFactories/TkNumericIntegration.py`** (M1): the two `except SQLAlchemyError`
  handlers that tested `"break_point_kind" in str(e)`, in `build` and in
  `TkNumericValue._build_impl_model`, deleted. Both `try`s also guard `except MultipleResultsFound`,
  so they stay.
- **`GkNumericIntegration.py`** (M1): the same two, and the third in `GkNumericValue.read_batch`,
  whose `try` guarded nothing else and is removed with it.
- **`QCD_Cosmology.py`** (M1): the handler (`"T_z_representation" in str(e)`) and its `try`
  deleted. Nothing there handled `MultipleResultsFound` separately (the old handler only re-raised
  it), so it simply propagates; the comment says so. The module docstring keeps why the column is in
  the key and loses the old-store paragraph. The `SQLAlchemyError` import, unused now, is removed.
- **`BackgroundModel.py`** (M1), `sqla_BackgroundModelFactory.build`: the whole `except` (the
  Gauss-order refusal, the grid-identity compute-path refusal and the read-path re-query) deleted;
  `_build_query()` lost its `with_grid_identity` parameter and always selects the two grid-identity
  columns; the `hasattr(row, "source_grid_digest")` test in the "spanning N generations" refusal and
  the one where the identity was set are gone, with the "unknown generation (this datastore records
  none)" wording; `_source_grid_identity` is now set from the found row at the end of `build`
  (`= None` on a miss and the read path's `z_sample=None` selection are kept); the value-table
  handler (`tau_lo_Mpc`, `cs_tau_Mpc`, `cs_tau_lo_Mpc`, `friction_F`) deleted; the module-docstring
  paragraphs and inline comments on pre-prompt-03/04, pre-05 and pre-14 stores removed. The three
  prune blocks (`TkNumericIntegration.py` `validate_on_startup`, `GkNumericIntegration.py`
  `validate_on_startup`, `BackgroundModel.py` `validate_on_startup`) are untouched, and
  `SQLAlchemyError` is still imported in the three files that use it.
- **`extract_common.py`** (M2):
  - `choose_run_label([], None)` raises `RuntimeError`: "extract: this datastore records no runs: its
    store_tag table holds no label beginning "Run_", which main.py writes with everything it
    computes. There is nothing to read." The requested-label message for an empty store ("none (this
    datastore records no runs)") is unchanged.
  - `RunSelection.label` is typed `str`; its docstring, the module docstring's bullet and
    `choose_run_label`'s docstring are corrected; `resolve_run_selection`'s
    `if selection.label is None` is gone.
  - **`describe_background_generation` (the prompt's choice): it says so, it does not raise.** A
    model with `_source_grid_identity is None` is one that was not found, and the function returns
    "no grid generation (the datastore holds no background model for it)". Reason: it is a print
    helper, and a raise there could abort an extraction at a print statement. All six `extract_*.py`
    scripts call `require_background(model, …)` two lines earlier, which raises `NoBackgroundModel`
    on a miss, so the branch is unreachable from every script and the choice cannot change a script's
    output. Its docstring says this.
- **Tests deleted (M3), eleven**:
  - `ComputeTargets/tests/test_run_identity.py`
    - `TestAStoreWithNoGridIdentity`:
      `test_the_read_path_still_reads_it_and_labels_it_unknown`,
      `test_the_compute_path_demands_regeneration`,
      `test_an_unrelated_database_error_is_not_disguised`;
    - `TestChoosingARun.test_a_store_with_no_runs_reads_as_unknown`;
  - `ComputeTargets/tests/test_cosmology_representation_key.py`, `TestOldSchemaFailsLoudly`:
    `test_a_table_without_the_column_demands_regeneration`,
    `test_the_error_it_raises_is_not_the_opaque_one`, `test_an_unrelated_database_error_is_not_disguised`;
  - `ComputeTargets/tests/test_numeric_break_point_key.py`, `TestOldSchemaFailsLoudly`:
    `test_a_missing_column_names_the_prompt_and_demands_regeneration`,
    `test_an_unrelated_database_error_is_not_disguised`;
  - `Datastore/tests/test_value_reads_parent_key.py`, `TestStoreWithoutBreakPointKindIsRefused`:
    `test_gk_numeric_value_reads`, `test_tk_numeric_value_build`.
  - Nine from ComputeTargets, two from Datastore. The docstring lines describing them are corrected
    in `test_run_identity` (the bullet on a store with no run label), `test_cosmology_representation_key`
    and `test_numeric_break_point_key` (item 3 of each, marked as removed by this prompt).
  - **The prompt's search for others** ("predates", "no migration", "unknown generation", "records no
    run", and tables built without one of the columns, by `DROP COLUMN`, `omit=` and
    `missing_columns`) found only matches that are not 01's: `predates` in
    `test_wkb_numeric_parent_key.py`, `test_quadsource_integral_parent_key.py`,
    `test_gksource_parent_set.py` and `test_quadsource_policy_key.py` (prompt 02);
    `test_prune_at_open.py` (prompt 03); `RunRegistry/tests/test_run_registry.py` (a run directory);
    "unknown generation" in `wkb_reference.py` and `convergence_reference.py` (a source-grid
    specification); the fixture machinery of `real_store_fixtures.py` and one use of
    `missing_columns` in `test_store_inventory.py` (a column outside this prompt's). None is deleted.
    `test_background_cs_tau_friction.py`'s `test_a_value_without_the_new_limbs_is_refused` tests an
    in-memory guard and is left.
  - **Helpers that served only the eleven**, removed (grepped for other uses first):
    `_append_table`'s `omit`, `_Schema(omit_grid_identity=…)`, the `OperationalError` import and the
    stale docstring text in `test_run_identity`; `_RaisingConnection`, `_missing_column_error`,
    `_build_table`'s `omit` and the `OperationalError` import in
    `test_cosmology_representation_key`; `_RaisingConnection`, `_missing_column_error`,
    `_run_build_against` and the `OperationalError` import in `test_numeric_break_point_key`; the two
    `_QueryCaptured` docstrings no longer speak of a missing-column handler. An AST check finds no
    unused import in any file changed.
- **Tests added**, thirteen:
  - `ComputeTargets/tests/test_run_identity.py`, `TestChoosingARun` (M2):
    - `test_a_store_with_no_runs_is_refused_saying_it_records_none`: `choose_run_label([], None)`
      raises, the message says "records no runs", names the `Run_` prefix, and contains neither
      "predates" nor "unknown";
    - `test_a_label_requested_of_a_store_with_no_runs_is_refused_naming_none`: the requested-label
      refusal names the label and "this datastore records no runs".
  - `test_run_identity.TestDescribingABackgroundGeneration` (M2): `test_a_found_model_is_described_by_its_construction_and_digest`
    and `test_a_model_that_was_not_found_is_said_to_have_no_generation` (also for an object with no
    `_source_grid_identity` attribute; no "predates", "prompt 14" or "unknown").
  - **New module `Datastore/tests/test_database_errors_reach_the_caller.py`** (M4), nine tests. The
    statement is **never written in the module**: `_StatementRaisingConnection.execute` is handed the
    query the factory built, compiles it with SQLite's dialect, and raises `OperationalError(<that
    text>, {}, Exception("database is locked"))`. `assert_passes_through` first checks that the
    error's `str()` names the columns the old handler looked for (so that an old-style handler would
    have matched it), then that the exception the caller sees **is** the raised object
    (`assertIs`) with no `__cause__`; each test also asserts the number of `execute` calls. Real tables
    from `build_schema(MetaData(), _factories)`.
    - `TestTheNumericFactories`: `test_tk_numeric_integration_build`,
      `test_gk_numeric_integration_build`, `test_tk_numeric_value_build`,
      `test_gk_numeric_value_build`, `test_gk_numeric_value_read_batch` (each: `break_point_kind`,
      executed once);
    - `TestQCDCosmology.test_qcd_cosmology_build` (`T_z_representation`, executed once);
    - `TestBackgroundModel`:
      `test_the_read_path_is_executed_once_and_the_error_reaches_the_caller` (`z_sample=None`; grid
      identity and Gauss-order columns; once), `test_the_compute_path_is_executed_once_and_the_error_reaches_the_caller`
      (a real `redshift_array`; once), `test_the_value_table_read_error_reaches_the_caller` (a real
      validated `BackgroundModel` row; the first `execute` is the real connection's and succeeds, the
      second raises; the statement names `tau_lo_Mpc`, `cs_tau_Mpc`, `cs_tau_lo_Mpc`, `friction_F`;
      executed twice).
    - The two value `build`s are covered in addition to the five the prompt lists, because their
      handlers were deleted too.
- **The board and the index**: board header, §1 row, M1–M4, §3 (the fallbacks issue narrowed), §4 (the
  misfire issue, with its closing note), §5.2. `docs/OPEN_ISSUES.md`: the misfire row deleted, the
  fallbacks row annotated, count 108 → 107 by grep, matching the header.

## Deviations from the prompt

1. **The dead test helpers were removed** (listed above). `IMPLEMENTATION CHOICE`, instructed by the
   orchestrator: they served only the deleted tests, and removing them is part of M3's deletion.
2. **`QCD_Cosmology`'s `SQLAlchemyError` import removed** with the handler that was its only use.
   `STRUCTURALLY REQUIRED` (an unused import otherwise).
3. **M4 covers `TkNumericValue.build` and `GkNumericValue.build` as well** as the paths the prompt
   lists. `IMPLEMENTATION CHOICE`: their handlers were deleted in M1, so their pass-through is pinned
   too.
4. **The M4 module is in `Datastore/tests/`, not beside the old tests.** `IMPLEMENTATION CHOICE`: it
   needs the real schema (`build_schema`) and one module serves every factory. The prompt allows a new
   module.
5. **Breakage (iii) as the prompt words it fails; a variant passes.** Recorded below.
   `IMPLEMENTATION CHOICE`.
6. **A `QCD_Cosmology.build` comment** ("Any database error, that one included, reaches the caller as
   itself") replaces the comment on the removed handler, so the `MultipleResultsFound` behaviour stays
   stated. `IMPLEMENTATION CHOICE`.

## Verification performed

- **Counts.** Serially, on the staged tree, `PYTHONPATH=. ./venv/bin/python -m unittest discover -s
  <package>/tests -t .`, verdicts grepped from the output files:

  | Suite | Before (orchestrator, `6ea366e`) | After | Difference |
  |---|---|---|---|
  | AdaptiveLevin | 32 OK | 32 OK | 0 |
  | ComputeTargets | 698, 1 failure | 693, 1 failure (run twice) | −9 deleted, +4 added |
  | CosmologyModels | 39 OK | 39 OK | 0 |
  | Datastore | 598 OK | 605 OK | −2 deleted, +9 added |
  | LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 0 |
  | RunRegistry | 281 OK | 281 OK | 0 |

  ComputeTargets was run once before the helper cleanup and once after it, both 693. In both, the one
  failure is `test_tk_wkb_phase.TestCost.test_wall_time_per_object` (the baseline's was
  `test_background_tau.test_build_cost`, which passed here: a wall-clock test that depends on load).
  **Re-run alone**, at load average about 25, it fails three times in this checkout (0.0856 s against
  a 0.06 limit) **and twice on an export of `6ea366e`** (`git archive HEAD`, 0.0506 s and 0.0687 s
  against 0.05): it fails without this prompt's changes, and nothing here touches `Tk WKB`. It is a
  wall-clock assertion that does not hold under this load.
- **`grep -n 'in str(e)' Datastore/SQL/ObjectFactories/*.py`**: no output (exit 1).
- **`black --check`** on the ten files changed: clean. An AST check finds no unused import in any of
  them.
- **Staging.** By explicit path; `git diff --cached --name-only` is shown in the commit session.

## The deliberate-breakage record

Each diff is taken against the staged tree (the tree `8636491` holds) and replays with `git apply`
from a clean tree at `8636491`, in `bash`. All were `git apply --check`ed, applied, run, and
reversed with `git apply -R`; the tree was clean against the index afterwards. None is committed.

**(i) Restore `TkNumericIntegration.build`'s handler, with its `str(e)` test.** Fails
`Datastore.tests.test_database_errors_reach_the_caller` (`Ran 9`, `FAILED (errors=1)`):
`TestTheNumericFactories.test_tk_numeric_integration_build` (the caller gets a `RuntimeError`).

```diff
diff --git a/Datastore/SQL/ObjectFactories/TkNumericIntegration.py b/Datastore/SQL/ObjectFactories/TkNumericIntegration.py
index 22e93e9..cbcd304 100644
--- a/Datastore/SQL/ObjectFactories/TkNumericIntegration.py
+++ b/Datastore/SQL/ObjectFactories/TkNumericIntegration.py
@@ -280,6 +280,15 @@ class sqla_TkNumericIntegration_factory(SQLAFactoryBase):
                 f"!! TkNumericIntegration.build(): multiple results found when querying for TkNumericIntegration"
             )
             raise e
+        except SQLAlchemyError as e:
+            if "break_point_kind" in str(e):
+                raise RuntimeError(
+                    "TkNumericIntegration.build(): the TkNumericIntegration table has no "
+                    '"break_point_kind" column. This datastore predates the numeric break-point '
+                    "policy becoming part of the lookup key and must be regenerated; there is "
+                    "no migration."
+                ) from e
+            raise
 
         if row_data is None:
             # build and return an unpopulated object
```

**(ii) Restore `BackgroundModel.build`'s read-path re-query alone** (the `with_grid_identity`
parameter and the handler, without the compute-path refusal or the others). Fails the same module
(`Ran 9`, `FAILED (failures=1)`):
`TestBackgroundModel.test_the_read_path_is_executed_once_and_the_error_reaches_the_caller` (the
connection is executed twice).

```diff
diff --git a/Datastore/SQL/ObjectFactories/BackgroundModel.py b/Datastore/SQL/ObjectFactories/BackgroundModel.py
index 66d0447..e66c033 100644
--- a/Datastore/SQL/ObjectFactories/BackgroundModel.py
+++ b/Datastore/SQL/ObjectFactories/BackgroundModel.py
@@ -240,7 +240,7 @@ class sqla_BackgroundModelFactory(SQLAFactoryBase):
         source_grid_construction = SOURCE_GRID_CONSTRUCTION_VERSION
         source_grid_digest = z_sample.digest() if z_sample is not None else None
 
-        def _build_query():
+        def _build_query(with_grid_identity: bool = True):
             """
             The lookup. It selects the two grid-identity columns on both paths, and on the
             compute path, where z_sample is given, also filters on them.
@@ -264,9 +264,11 @@ class sqla_BackgroundModelFactory(SQLAFactoryBase):
                 table.c.friction_F_gauss_order,
                 solver_table.c.label.label("solver_label"),
                 solver_table.c.stepping.label("solver_stepping"),
-                table.c.source_grid_digest,
-                table.c.source_grid_construction,
             ]
+            if with_grid_identity:
+                columns.extend(
+                    [table.c.source_grid_digest, table.c.source_grid_construction]
+                )
 
             # notice that we query only for validated data
             q = (
@@ -291,7 +293,7 @@ class sqla_BackgroundModelFactory(SQLAFactoryBase):
                 )
             )
 
-            if source_grid_digest is not None:
+            if with_grid_identity and source_grid_digest is not None:
                 # the compute path: a model tabulated on a different grid, or built by a
                 # different construction, is a different background and must miss
                 q = q.filter(
@@ -320,7 +322,15 @@ class sqla_BackgroundModelFactory(SQLAFactoryBase):
 
             return q
 
-        rows = list(conn.execute(_build_query()))
+        try:
+            rows = list(conn.execute(_build_query()))
+        except SQLAlchemyError as e:
+            if z_sample is not None or not any(
+                column in str(e)
+                for column in ("source_grid_digest", "source_grid_construction")
+            ):
+                raise
+            rows = list(conn.execute(_build_query(False)))
 
         if len(rows) > 1:
             generations = sorted(
```

**(iii) Replace the captured statement with the literal `"SELECT ..."` in one M4 test.**

```diff
diff --git a/Datastore/tests/test_database_errors_reach_the_caller.py b/Datastore/tests/test_database_errors_reach_the_caller.py
index f916652..4acb28f 100644
--- a/Datastore/tests/test_database_errors_reach_the_caller.py
+++ b/Datastore/tests/test_database_errors_reach_the_caller.py
@@ -166,6 +166,7 @@ class TestTheNumericFactories(_PassThroughCase):
 
     def test_tk_numeric_integration_build(self):
         conn = _StatementRaisingConnection()
+        conn._statement = lambda query: "SELECT ..."
         self.assert_passes_through(
             lambda: sqla_TkNumericIntegration_factory.build(
                 self._parent_payload("z_init"),
```

Alone, it fails `test_tk_numeric_integration_build` on the precondition `assert_passes_through` checks
first: the error's `str()` does not name `break_point_kind`. That is the module's guard working, and
it makes the prompt's demonstration impossible as worded: with (i) restored as well (`(i)` then
`(iii)`, `Ran 9`, `FAILED (failures=1)`) the test still fails, on the same guard. To show what the
captured statement buys, the variant below also drops that guard (an empty column list):

```diff
diff --git a/Datastore/tests/test_database_errors_reach_the_caller.py b/Datastore/tests/test_database_errors_reach_the_caller.py
index f916652..515f79c 100644
--- a/Datastore/tests/test_database_errors_reach_the_caller.py
+++ b/Datastore/tests/test_database_errors_reach_the_caller.py
@@ -166,6 +166,7 @@ class TestTheNumericFactories(_PassThroughCase):
 
     def test_tk_numeric_integration_build(self):
         conn = _StatementRaisingConnection()
+        conn._statement = lambda query: "SELECT ..."
         self.assert_passes_through(
             lambda: sqla_TkNumericIntegration_factory.build(
                 self._parent_payload("z_init"),
@@ -176,7 +177,7 @@ class TestTheNumericFactories(_PassThroughCase):
                 None,
             ),
             conn,
-            (BREAK_POINT_KIND,),
+            (),
         )
         self.assertEqual(1, conn.executed)
 
```

**(i) and the variant together: `Ran 9 tests`, `OK`.** The M4 test passes with the old handler back
in place. With the captured statement (i) alone fails it. That is the reason the statement is
captured, and the reason the module checks the statement names the columns first.

**(iv) Make `choose_run_label` return a selection for a store with no runs.** Fails
`ComputeTargets.tests.test_run_identity.TestChoosingARun`
(`Ran 7`, `FAILED (failures=1)`): `test_a_store_with_no_runs_is_refused_saying_it_records_none`.

```diff
diff --git a/extract_common.py b/extract_common.py
index 593043f..f42d688 100644
--- a/extract_common.py
+++ b/extract_common.py
@@ -220,11 +220,7 @@ def choose_run_label(
         )
 
     if len(available) == 0:
-        raise RuntimeError(
-            "extract: this datastore records no runs: its store_tag table holds no label "
-            f'beginning "{RUN_LABEL_TAG_PREFIX}", which main.py writes with everything it '
-            "computes. There is nothing to read."
-        )
+        return RunSelection(label="", tags=[], description="an unnamed run")
 
     if len(available) == 1:
         return RunSelection(
```

## Observations not acted on

- **Six `extract_*.py` scripts carry a stale comment**: "…and why a datastore that records no runs at
  all is read rather than refused" (`extract_Gk_data.py:468`, `extract_TkWKB_data.py:508`,
  `extract_QuadSourceIntegral_data.py:1309`, `extract_tensor_source_data.py:436`,
  `extract_GkWKB_data.py:520`, `extract_GkSource_data.py:947`). It is now false. Outside this prompt's
  files; no issue opened, per the orchestrator, who corrects them. No script reads `RunSelection.label`
  (the third stop condition does not fire); they print only `.description`.
- **`BackgroundModel.py`'s module docstring still says** that when prompt 11 changed the grid "a
  pre-prompt-11 datastore went on serving its 1,732-node background" for the new grid. Kept, as the
  rationale for the grid-identity key, not an accommodation; the prompt did not list it.
- **The three prune blocks** (`except SQLAlchemyError:` in `validate_on_startup`) swallow a database
  error and continue. Unchanged, not 01's.
- **The wall-clock tests** `test_tk_wkb_phase.TestCost.test_wall_time_per_object` and
  `test_background_tau.test_build_cost` fail or pass by load; see above.

## State handed to the next prompt

- No factory in `TkNumericIntegration`, `GkNumericIntegration`, `QCD_Cosmology` or `BackgroundModel`
  catches a `SQLAlchemyError` to name an older schema. Prompt 02 owns `TkWKBIntegration`,
  `GkWKBIntegration`, `GkSource` and `QuadSourceIntegral`, whose tests match `e.orig` and do not
  misfire.
- `Datastore/tests/test_database_errors_reach_the_caller.py` has the harness
  (`_StatementRaisingConnection`, `assert_passes_through`) for 02's N3: it needs only a table from
  `build_schema` and the factory's payload.
- `ComputeTargets` 693, `Datastore` 605; the others unchanged.
