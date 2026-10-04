# Log — prompt 02: delete the remaining factory refusals

**Subject:** *Delete the remaining factory refusals* · **Model:** Claude Sonnet 5.5 · **Date:**
2026-10-04 · **Base:** `bd5ad19`, clean · **Result:** landed. The old-store handlers, helpers and
comments are gone from `TkWKBIntegration`, `GkWKBIntegration`, `GkSource` and `QuadSourceIntegral`;
a database error from any of their statements reaches the caller as itself. Seventeen tests
deleted, two kept (one moved unchanged, one rewritten), twelve added. Neither of the prompt's §5
stop conditions fired. Closes nothing; narrows
`[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]`. Opens nothing. Index 107 → 107.
The commit is this commit.

## What shipped

- **N1, the four factories** (all handlers, helpers, docstring text and comments; the `try`s that
  also guard `MultipleResultsFound` stay):
  - `TkWKBIntegration.py`: the `except SQLAlchemyError` of `sqla_TkWKBIntegration_factory.build`
    (the `numeric_serial` refusal and the `rho_gauss_order` refusal); the one in
    `sqla_TkWKBValue_factory._build_impl_model` (which `TkWKBValue.build` reaches); in
    `read_batch`, the handler and its `try` (it guarded only `conn.execute(row_query).all()`); the
    module docstring's "There is no migration and no default … build() raises naming it" paragraph
    and its last sentence on "a store written before this change".
  - `GkWKBIntegration.py`: the same three handlers, `read_batch`'s `try` with them, and the same two
    docstring passages.
  - `GkSource.py`: `_refuse_a_store_that_predates_the_parent_set` and its two calls (in
    `sqla_GkSource_factory.build` and `sqla_GkSourceValue_factory._build_impl_model`, each now an
    `except MultipleResultsFound` only); the docstring's "A store written before this change has no
    parents_digest column …" sentence; and the paragraph "Nothing is added, so a datastore written
    before this change is readable by this factory without a migration" (audit B7: false at `HEAD`).
  - `QuadSourceIntegral.py`: `_refuse_a_store_that_predates_the_policy_key`,
    `_TRANSFER_FUNCTION_COLUMNS` (used only by the next),
    `_refuse_a_store_that_predates_the_transfer_function_key`, their calls in `build` (the `try`
    stays, for `MultipleResultsFound`) and in `read_batch` (the `try` guarded only `conn.execute`
    and goes), and the comment on the column order. The order itself stays (the prompt's choice;
    see Deviations 2). `SQLAlchemyError`, unused in this module now, is no longer imported.
  - **The four prune blocks are untouched** (`except SQLAlchemyError:` in `validate_on_startup`:
    `TkWKBIntegration.py:946`, `GkWKBIntegration.py:947`, `GkSource.py:826`, and `QuadSource.py:529`,
    which is not 02's file). They are the only `except SQLAlchemyError` left in these four files and
    in `QuadSource.py`.
  - **Read with `ast`** (functions whose body changed): `sqla_TkWKBIntegration_factory.build`,
    `sqla_TkWKBValue_factory._build_impl_model`, `sqla_TkWKBValue_factory.read_batch`, the same
    three in `GkWKBIntegration.py`, `sqla_GkSource_factory.build`,
    `sqla_GkSourceValue_factory._build_impl_model`, `sqla_QuadSourceIntegral_factory.build` and
    `read_batch`; functions removed: the three `_refuse_a_store_that_predates_*` helpers. No other
    function in the four files differs from `bd5ad19`.
- **The first stop condition does not fire** (read; the orchestrator's briefing agrees). Each
  deleted handler converted a matching error into a `RuntimeError` or re-raised it; none recovered
  from anything. The one match a current store could produce is `GkSource`'s `"GkSource_parents" in
  driver_message` for an absent `GkSource_parents` table (an interrupted first open, U1, which is
  05's): it raised before and raises after, as an `OperationalError` where it was a `RuntimeError`.
  The second stop condition does not fire either: no test outside N2's list asserts a deleted
  message (see N2).
- **The `numeric_serial` selects stay.** In both WKB factories `_build_impl_model` and `read_batch`
  select `wkb_table.c.numeric_serial` whether or not a parent is named (four sites). I checked, with
  `grep` and `ast`, that the selected value is **not read**: in all four the column goes into the
  inner query that becomes a subquery, and the outer `select` takes only `subquery.c.sin_coeff`,
  `cos_coeff` and `z_init`; `numeric_serial` otherwise appears in those functions only in the select
  and in `_numeric_parent_filter`, which is applied only when a parent is named. The four comments
  are reworded: "always selected, whether or not a parent is given, so that a table without the
  column raises rather than being read". They name no prompt's message and do not say "predates" or
  "written before". See Deviations 1 and Observations.
- **N2, the tests of what was deleted: seventeen methods deleted, two kept.** All in `Datastore`.
  - `test_gksource_parent_set.py`, **deleted**:
    - `TestAnOldStoreIsRefused` as a class: `test_a_keyed_lookup`, `test_the_reader_mode`,
      `test_the_value_model_read`;
    - `TestTheWKBValueReadsRefuseAPre07Store`: `test_gk_wkb_value_read_batch`,
      `test_gk_wkb_value_build_impl_model`, `test_tk_wkb_value_read_batch`,
      `test_tk_wkb_value_build_impl_model`; the class and its `setUp` (which rebuilt the WKB tables
      from the 06 witness) go.
    - **Kept, moved:** `test_a_current_table_is_still_read` is now
      `TestTheWKBValueReadsOnACurrentStore.test_a_current_table_is_still_read`, a class of its own.
      README §5 rule 7: it tests current behaviour. Its source segment is **byte-identical** to its
      pre-image (compared with `ast.get_source_segment` against `bd5ad19`); it builds its own
      current store and does not use the old class's `setUp`. It is counted as a deleted and an
      added method by an `ast` diff of names, but is neither: the suite count does not move for it.
    - `test_a_store_without_the_table_is_named_absent` (the inventory's, audit B5) is **left** for
      05.
  - `test_wkb_numeric_parent_key.py`: `TestAnOldStoreIsRefused.test_tk`, `.test_gk` (the class).
  - `test_quadsource_integral_parent_key.py`: `TestAnOldStoreIsRefused`:
    `test_a_table_from_before_this_prompt`, `test_a_table_from_before_prompt_08_keeps_its_message`,
    `test_another_database_error_is_not_turned_into_it` (the class). Its property is covered by the
    kept test below and by N3.
  - `test_quadsource_policy_key.py`: `TestAnOldStoreIsRefused`: `test_build`,
    `test_read_batch_untagged_and_tagged`. **Kept, rewritten:**
    `test_another_database_error_is_not_turned_into_it` is now
    `TestAnotherDatabaseErrorPassesThrough.test_another_database_error_is_not_turned_into_it`. Its
    fixture **was** the old table (the class's `setUp` recreated `QuadSourceIntegral` from the 07
    witness's DDL) and raised `OperationalError("SELECT 1", …)` through `mock.patch.object(conn,
    "execute")`. It is now on the current table that `_Case.setUp` builds, and raises through 01's
    harness (`_StatementRaisingConnection`, `_PassThroughCase.assert_passes_through`), so the
    statement is the compiled text of the one `build` or `read_batch` issued, the check that it names
    `quad_policy_serial` runs first, and the exception that reaches the caller is asserted to be
    the raised object (`assertIs`, no `__cause__`). **This is the same property as N3's
    `TestQuadSourceIntegral`, in the module of the key it concerns; I kept both** (N3's covers
    `build`, `read_batch` untagged and tagged, and names the four transfer-function columns as well;
    this one covers `build` and `read_batch` under the policy fixture, naming `quad_policy_serial`).
    The log's "which is which": the rewritten test is **the prompt's kept test**; N3's
    `QuadSourceIntegral` test is the one in `test_database_errors_reach_the_caller.py`.
  - `test_quadsource_parent_key.py`: `TestAnOldStoreIsRead`:
    `test_the_old_table_has_no_index_on_tr_and_both_serials`, `test_a_keyed_lookup_finds_its_row`,
    `test_the_reader_mode_finds_it_too` (the class; a compatibility property with no code behind it).
  - **Counted with `ast`** against `bd5ad19`, over the six test modules: 19 method names disappear
    (17 deletions plus the two kept ones, which reappear under a new class name) and 14 appear (the
    two kept plus 12 new N3 tests).
  - **Helpers that served only the deleted tests**, removed in the same commit (each grepped first;
    an `ast` check finds no unused import in any file changed that was not unused already):
    - `test_gksource_parent_set`: `WITNESS_06`, `WITNESS_09A`, `DATA`, `D4`, the `json` import and
      the `sqla_TkWKBValue_factory` import (used by the deleted `tk_*` helpers only);
    - `test_wkb_numeric_parent_key`: `WITNESS_06`, the `json` import (`_make_old`, `assertD4` went
      with the class). Its `_OLD_DROP_ORDER` and `mock` stay: they are `TestDropOrder`'s;
    - `test_quadsource_integral_parent_key`: `WITNESS_07`, `WITNESS_09B`, `DATA`, `D4`, and the
      `json`, `mock` and `OperationalError` imports (`recreate` and `refused` went with the class);
    - `test_quadsource_policy_key`: `WITNESS_07`, and the `json`, `mock` and `OperationalError`
      imports (`assertD4` and `setUp` went with the class);
    - `test_quadsource_parent_key`: `WITNESS_08` and the `json` import.
    - Left alone, because it was unused before: `TkNumericIntegration` in
      `test_gksource_parent_set` and `sqla_TkWKBValue_factory` in `test_wkb_numeric_parent_key`.
    - The witness JSON files under `Datastore/tests/data/` stay (`test_schema_builder.py` reads all
      of them, `test_store_inventory.py` the 06 one). `real_store_fixtures.py` is untouched.
  - **The module docstrings** were corrected: `test_gksource_parent_set` (items 6 and 7),
    `test_wkb_numeric_parent_key` (item 6), `test_quadsource_integral_parent_key` (item 6),
    `test_quadsource_policy_key` (the bullet on the old store, now on the pass-through) and
    `test_quadsource_parent_key` (item 7).
  - **The prompt's search for any other test asserting a deleted message** (`predates`,
    `no migration`, `readable at the git SHA`, `sidecar records`, `older store`, `written before`,
    the deleted messages' `no "<column>"` forms, `D4`, and `OperationalError` across
    `Datastore/tests`, `ComputeTargets/tests`, `RunRegistry/tests`) finds nothing outside N2's list.
    The remaining hits: `test_prune_at_open.py:26` and `:620` (a prune record, 03's);
    `test_database_errors_reach_the_caller.py:6` (01's docstring); `test_run_identity.py:722–791`
    (01's `assertNotIn("predates", …)`); `test_numeric_break_points.py:1341` (an unrelated
    "written before this prompt"); `RunRegistry/tests` (run directories and a sidecar); unrelated
    `D4`s. None is deleted.
- **N3, twelve new tests**, in `Datastore/tests/test_database_errors_reach_the_caller.py`, **new
  classes only** (01's nine tests and its harness are unchanged; the module docstring gained a
  paragraph, and the imports and column constants that the new classes need were added). Each uses
  01's `_StatementRaisingConnection` and `_PassThroughCase.assert_passes_through`: the error is
  raised with the compiled text of the query the factory built, `columns=` names the column the
  deleted handler looked for and is checked against that text first, the caller then gets the same
  object with no `__cause__`, and the number of `execute` calls is asserted. The schema is the real
  one (`wkb_keys._base_store()`, `gk_source_keys._gk_store()`, `qsi_keys._WithAStore`).
  - `TestTheWKBIntegrations`: `test_tk_wkb_integration_build`, `test_gk_wkb_integration_build` (a
    named parent; `numeric_serial` and `rho_gauss_order`; one `execute`);
  - `TestTheWKBValueReads` (`numeric_serial`; one `execute`; each as sub-tests with and without a
    parent named, `parents_only` for the Gk `read_batch`, and `None` for a pure-WKB Gk build):
    `test_tk_wkb_value_build`, `test_gk_wkb_value_build`, `test_tk_wkb_value_read_batch`,
    `test_gk_wkb_value_read_batch`;
  - `TestGkSource`: `test_gk_source_build` (`parents_digest`; one `execute`),
    `test_gk_source_build_in_the_reader_mode` (the same; the labels-only count of matches is the
    first `execute`, and it raises), `test_the_read_of_the_parent_set_reaches_the_caller` (a real,
    stored `GkSource`; the first two `execute`s are the real connection's and succeed; the third, the
    read of the `GkSource_parents` rows, raises, and its statement names `GkSource_parents`: the other
    thing the deleted handler matched), `test_gk_source_value_build` (`parents_digest`);
  - `TestQuadSourceIntegral`: `test_quad_source_integral_build`,
    `test_quad_source_integral_read_batch` (untagged and tagged); `quad_policy_serial` and the four
    transfer-function columns; one `execute` each.
- **The board and the index.** `IMPLEMENTATION_STATE.md`: header, §1 row, N1–N3, §3 (the fallbacks
  issue narrowed), §5.3. `docs/OPEN_ISSUES.md`: the fallbacks row names 02's deletions; the count is
  107 by grep, matching the header, and the header date is unchanged (2026-10-04).

## Deviations from the prompt

1. **The four `numeric_serial` selects are kept, and only their comments reworded.**
   `IMPLEMENTATION CHOICE`, directed by the orchestrator (the audit missed them; the user confirmed
   on 2026-10-04 that the campaign removes them later, at 05, item S3). Deleting them would let a
   labels-only read of a pre-07 table succeed with values from rows whose parent is unrecorded,
   where the prompt's §1 says such a store meets SQLAlchemy's own `OperationalError`. I checked that
   the selected value is not read (above), so the select is kept as the reason a table without the
   column raises, as current code.
2. **`QuadSourceIntegral`'s column order is kept** (the prompt's choice), and the comment that
   explained it is deleted. `IMPLEMENTATION CHOICE` (the prompt's).
3. **Helpers, constants and imports that only the deleted tests used are deleted in the same
   commit** (listed above). `IMPLEMENTATION CHOICE`, directed by the orchestrator and by 01's review.
4. **`QuadSourceIntegral.py`'s `SQLAlchemyError` import removed**, with the handlers that were its
   only use. `STRUCTURALLY REQUIRED` (an unused import otherwise).
5. **Two N3 tests beyond the prompt's one per factory**, and a `QuadSourceIntegral` property tested
   twice (the kept test and N3's). `IMPLEMENTATION CHOICE`: the value reads' handlers and
   `GkSourceValue.build`'s were deleted too, and the second `GkSource` statement is the one the
   handler's table-name test matched. The two `QuadSourceIntegral` tests live in different modules,
   one for each reason (the prompt's kept test; N3).
6. **The kept `QuadSourceIntegral` test imports 01's harness from
   `test_database_errors_reach_the_caller`** rather than only living in it. `IMPLEMENTATION CHOICE`:
   the rewrite is the prompt's, N3's own `QuadSourceIntegral` test is in the harness's module, and
   the briefing allows either.

## Verification performed

- **Counts.** Serially, on the staged tree (`bd5ad19` plus this change), `PYTHONPATH=. ./venv/bin/python
  -m unittest discover -s <package>/tests -t .`, output to a file under the scratchpad, verdicts
  grepped from the file:

  | Suite | Before (orchestrator, `bd5ad19`) | After | Difference |
  |---|---|---|---|
  | AdaptiveLevin | 32 OK | 32 OK | 0 |
  | ComputeTargets | 693, 1 failure (`test_tk_wkb_phase.TestCost.test_wall_time_per_object`, wall clock) | 693 OK | 0 (the flake passed) |
  | CosmologyModels | 39 OK | 39 OK | 0 |
  | Datastore | 605 OK | 600 OK | −17 deleted, +12 added |
  | LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 0 |
  | RunRegistry | 281 OK | 281 OK | 0 |

  The two kept tests do not move the count (each exists before and after). The Datastore run took
  161 s at a load average of 6–23. Every test outside N2's list passes untouched, including
  `test_factory_select_columns.py`, whose read-back guard walks the factories' selects.
- **`grep -nE 'predate|older store|written before' Datastore/SQL/ObjectFactories/`**: no output
  (exit 1). Before 02 it found the four files' sites, and the four select comments.
- **`black --check`** on the ten Python files changed: clean.
- **An `ast` check** finds no unused import or constant left by the deletions in any test module
  changed (two unused names that were unused at `bd5ad19` remain, above).
- **Staging.** By explicit path; `git diff --cached --name-only` is shown in the commit session.

## The deliberate-breakage record

Each diff is taken against the staged tree (the tree this commit holds), so `git apply` replays it
from a clean tree at this commit, in `bash`. Each was `git apply --check`ed (forward from clean, and
`-R` while applied), applied, run, and reversed with `git apply -R`; the tree was clean against
the index afterwards. None is committed.

**(i) Restore `_refuse_a_store_that_predates_the_parent_set` and its call in `GkSource.build`,
changed to raise for every database error.** A faithful restoration would **not** fail N3, which
pins pass-through, because the original matched the driver's own message and never misfired on a
lock. This is the changed one, which does. Fails
`Datastore.tests.test_database_errors_reach_the_caller` (`Ran 60` with `test_gksource_parent_set`,
`FAILED (errors=3)`), all in `TestGkSource`: `test_gk_source_build`,
`test_gk_source_build_in_the_reader_mode`, `test_the_read_of_the_parent_set_reaches_the_caller`
(a `RuntimeError` reaches the caller). `GkSourceValue.build` is not restored by this diff, so its
test passes.

```diff
diff --git a/Datastore/SQL/ObjectFactories/GkSource.py b/Datastore/SQL/ObjectFactories/GkSource.py
index df026c0..de59ff0 100644
--- a/Datastore/SQL/ObjectFactories/GkSource.py
+++ b/Datastore/SQL/ObjectFactories/GkSource.py
@@ -63,6 +63,14 @@ def GkSource_parents_digest(members) -> str:
     ).hexdigest()
 
 
+def _refuse_a_store_that_predates_the_parent_set(e: SQLAlchemyError, where: str):
+    """Restored, and changed to raise for every database error."""
+    raise RuntimeError(
+        f'{where}: the GkSource table has no "parents_digest" column, or the store has no '
+        "GkSource_parents table."
+    ) from e
+
+
 def _read_parent_sets(conn, tables, serials) -> dict:
     """GkSource serial -> the parent set its GkSource_parents rows record, for ``serials``."""
     parents_table = tables["GkSource_parents"]
@@ -422,6 +430,8 @@ class sqla_GkSource_factory(SQLAFactoryBase):
                 f"!! GkSource.build(): multiple results found when querying for GkSource"
             )
             raise e
+        except SQLAlchemyError as e:
+            _refuse_a_store_that_predates_the_parent_set(e, "GkSource.build()")
 
         if row_data is None:
             # build and return an unpopulated object
```

**(ii) In `QuadSourceIntegral`'s lookup (`build`), catch `OperationalError` and return a miss.**
Fails `Ran 40`, `FAILED (failures=2)`:
`test_database_errors_reach_the_caller.TestQuadSourceIntegral.test_quad_source_integral_build` and
`test_quadsource_policy_key.TestAnotherDatabaseErrorPassesThrough.test_another_database_error_is_not_turned_into_it`
(sub-test `function='build'`: no exception reaches the caller).

```diff
diff --git a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
index e2bd77f..890f01a 100644
--- a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
+++ b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
@@ -3,7 +3,7 @@ from typing import Optional, List
 
 import sqlalchemy as sqla
 from sqlalchemy import and_
-from sqlalchemy.exc import MultipleResultsFound
+from sqlalchemy.exc import MultipleResultsFound, OperationalError
 
 from ComputeTargets import (
     QuadSourceIntegral,
@@ -389,6 +389,8 @@ class sqla_QuadSourceIntegral_factory(SQLAFactoryBase):
                 f"!! QuadSourceIntegral.build(): multiple results found when querying for QuadSourceIntegral (k={k.k.k_inv_Mpc:.5g}/Mpc, store_id={k.store_id} | q={q.k.k_inv_Mpc:.5g}/Mpc, store_id={q.store_id} | r={r.k.k_inv_Mpc:.5g}/Mpc, store_id={r.store_id})"
             )
             raise e
+        except OperationalError:
+            row_data = None
 
         if row_data is None:
             return QuadSourceIntegral(
```

**(iii) Break the WKB value read the kept `test_a_current_table_is_still_read` covers
(`GkWKBValue.read_batch`), by filtering on a column that does not exist.** The kept test fails, so it
still guards the read: `Datastore.tests.test_gksource_parent_set.TestTheWKBValueReadsOnACurrentStore.test_a_current_table_is_still_read`
errors, with five more tests of the same read in `TestTheValueReadsHandBackTheirParent`
(`test_each_wkb_value_carries_its_integration`, `test_the_serial_only_form_still_refuses_a_second_parent`
(two sub-tests), `test_the_serial_only_forms_return_the_pairs_and_no_values`,
`test_the_two_forms_agree`): `Ran 60`, `FAILED (errors=6)`. The N3 tests pass here, as they should,
since the error is raised before any statement runs.

```diff
diff --git a/Datastore/SQL/ObjectFactories/GkWKBIntegration.py b/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
index b650cf9..6aab5e0 100644
--- a/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
+++ b/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
@@ -1373,6 +1373,7 @@ class sqla_GkWKBValue_factory(SQLAFactoryBase):
                 wkb_table.c.model_serial == model_proxy.store_id,
                 wkb_table.c.wavenumber_exit_serial == k.store_id,
                 wkb_table.c.validated == True,
+                sqla.column("no_such_column") == 1,
             )
         )
 
```

## Observations not acted on

- **The `numeric_serial` selects** in `TkWKBIntegration` and `GkWKBIntegration` (`_build_impl_model`
  and `read_batch`, four sites): the selected value is read by nothing. They are kept as current
  code that makes a table without the column raise, with a neutral comment. **For 05** (item S3):
  once the reader refuses a store whose columns differ from the registry's (README §4), the select
  has no purpose of its own and goes. Already assigned on the board; no new issue.
- **`GkSource.build`, with `labels_only`,** reads the matched row's `GkSource_parents` rows inside
  the same `try` as the main query. An absent `GkSource_parents` table (an interrupted first open,
  U1; 05's) is now an `OperationalError` from SQLAlchemy where it was a `RuntimeError` naming the
  table. Nothing in the tests depends on which.
- **The four prune blocks** (`except SQLAlchemyError:` in `validate_on_startup`) swallow a database
  error and continue. Unchanged; known (`datastore-integrity` 02, `a3-v2-readiness` 02).
- **`TkNumericIntegration` is unused in `test_gksource_parent_set.py`** and
  `sqla_TkWKBValue_factory` in `test_wkb_numeric_parent_key.py`, both before 02. Not 02's.
- **`test_database_errors_reach_the_caller.py`'s docstring** still opens with 01's account of
  `str(e)` and now carries 02's paragraph after it.

## State handed to the next prompt

- No factory in the eight modules 01 and 02 cover catches a `SQLAlchemyError` to name an older
  schema. After 01–02, a store written before one of these columns meets SQLAlchemy's own
  `OperationalError` naming it (README §4).
- `Datastore` 600; every other suite unchanged. The index holds 107.
- 05 owns, beside the reader, the inventory and `real_store_fixtures.py`'s old-store machinery, the
  four WKB `numeric_serial` selects (S3).
