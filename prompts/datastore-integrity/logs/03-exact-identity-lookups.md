# Log 03 — Exact identity lookups

**Prompt:** [`prompts/datastore-integrity/03-exact-identity-lookups.md`](../03-exact-identity-lookups.md)
**Commit:** *(this commit)* — "Match the exit-time and identity lookups exactly, and raise on a
duplicate"
**Model:** Claude Sonnet 5
**Date:** 2026-09-29
**Result:** COMPLETE. L1–L5 shipped. No §5 stop condition was met:
- no production caller (`main.py`, an `extract_*.py` script) misses after L1 or L2 — each is
  measured below, in the L5 table;
- no `.scalar()` change makes an existing test fail;
- no existing test, including the read-back guard, failed at any point; nothing had to be
  resolved by changing anything but this prompt's own new code;
- no existing test or schema witness had to change;
- the prompt's acceptance was met inside the permitted files.

`Datastore` rises from 325 to 343 (the 18 tests of the new
`Datastore/tests/test_exact_identity_lookups.py`); every other suite is unchanged. The index
`docs/OPEN_ISSUES.md` goes from 101 to **98** rows (three deleted, none opened).

## What shipped

**L1 — the exit time.** `sqla_wavenumber_exit_time_factory.build`
(`Datastore/SQL/ObjectFactories/wavenumber.py`):
- the filter now matches `table.c.atol_serial == target_atol.store_id`,
  `table.c.rtol_serial == target_rtol.store_id` and `table.c.stepping == target_stepping` —
  serial equality, exactly the form every other keyed type uses (`tolerance.build` has already
  mapped the requested value to one serial), in place of the inequality
  `atol_table.c.log10_tol - target_atol.log10_tol <= DEFAULT_FLOAT_PRECISION` /
  `rtol_table.c.log10_tol - ... <= DEFAULT_FLOAT_PRECISION` / `table.c.stepping >= target_stepping`;
- the descending `.order_by(atol_table.c.log10_tol.desc(), rtol_table.c.log10_tol.desc())` is
  deleted: `one_or_none()` already made it dead, whatever order the rows came back in;
- `row_data = conn.execute(query).one_or_none()` is dedented out of the
  `for z_offset in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:` loop, so the query now executes once
  per `build()` instead of once per sub-horizon column (six times before);
- **the joins to the tolerance table are kept.** `atol_table`/`rtol_table` still alias
  `tolerance` and are still joined and selected (`log10_atol`, `log10_rtol`), because `build()`
  still uses those two columns to construct the returned `atol`/`rtol` objects
  (`atol = tolerance(store_id=row_data.atol_serial, log10_tol=row_data.log10_atol)`). Dropping the
  joins would mean building those objects from `target_atol`/`target_rtol` directly instead, which
  is also correct once the match is exact, but is a larger, less local change to the same lines the
  read-back guard pins (`KNOWN_BLIND_COLUMNS`'s `wavenumber.py:sqla_wavenumber_exit_time_factory.build`
  entry names the dynamic `z_exit_*` columns, not this pair) — see "Deviations", item 2.

**L2 — `IntegrationSolver`.** `sqla_IntegrationSolver_factory.build`
(`Datastore/SQL/ObjectFactories/integration_metadata.py`): the filter is
`sqla.and_(table.c.label == label, table.c.stepping == stepping)` in place of `stepping >=`, and
the query now ends `.scalar_one_or_none()` in place of `.scalar()`.

**L3 — duplicates raise.** `.scalar()` → `.scalar_one_or_none()`, filter unchanged, at the eight
sites audit §3.8 lists:
- `tolerance.py:35` (now `:38`, after the added comment);
- `store_tag.py:26` (now `:29`);
- `version.py:26` (now `:29`);
- `integration_metadata.py:34` (L2, above);
- `GkSourcePolicy.py:39` (now `:42`);
- `QuadSourcePolicy.py:39` (now `:42`);
- `LambdaCDM.py:55` (now `:58`);
- `QCD_Cosmology.py:99` (now `:103`, inside the `try … except SQLAlchemyError` block described
  below).

**L4 — `redshift` at `z = 0`.** `sqla_redshift_factory.build`
(`Datastore/SQL/ObjectFactories/redshift.py`): the filter is now built as
```python
z_filter = (
    table.c.z == 0.0
    if z == 0.0
    else sqla.func.abs((table.c.z - z) / z) < DEFAULT_REDSHIFT_RELATIVE_PRECISION
)
```
and passed to a single `.filter(z_filter)` call — one filter expression, as the orchestrator note's
guard-neutral sketch described, with the branch decided in Python (at `z == 0` the SQL-level
divide-by-zero never happens at all) rather than inside the SQL with a `CASE`. Every other `z`
keeps the unchanged relative match.

**L5 — who moves.** The full table is below. Every production caller (`main.py`, the six
`extract_*.py` scripts) is unchanged; the two verification scripts under
`docs/source-remediation-verification/` now miss, and are not edited.

## The L5 table

Every caller `grep` finds of the exit-time and `IntegrationSolver` lookups, each marked
**unchanged** or **now misses**. An "unchanged" rests on the requested pair and stepping against
what is stored, not on assertion; a "now misses" says what the miss then does, found by reading the
script (none of them was run: each needs Ray and a store under `var/`).

| Caller | Lookup | Requested | Stored | Verdict |
|---|---|---|---|---|
| `main.py:992-994`, `build_k_exit_work` | exit time | `(hexit_atol, hexit_rtol)` = `DEFAULT_HEXIT_ABS_TOLERANCE`/`_REL_TOLERANCE` = `(1e-10, 1e-9)`, `stepping` defaulted to 0 | every computed exit time stores `stepping = 0` (`CosmologyConcepts/wavenumber.py:738`) at the same pair | **unchanged** — exact match on the pair and stepping production writes |
| `extract_Gk_data.py:347-351` | exit time | same `DEFAULT_HEXIT_*` pair, `stepping` defaulted to 0 | same | **unchanged** |
| `extract_GkWKB_data.py:391-395` | exit time | same | same | **unchanged** |
| `extract_QuadSourceIntegral_data.py:996-1000` | exit time | same | same | **unchanged** |
| `extract_tensor_source_data.py:339-343` | exit time | same | same | **unchanged** |
| `extract_TkWKB_data.py:410-414` | exit time | same | same | **unchanged** |
| `extract_GkSource_data.py:800-804` | exit time | same | same | **unchanged** |
| `main.py:3621-3625`, five `solve_ivp+*` labels | `IntegrationSolver` | `stepping=0` | production inserts `stepping=0` for these five labels | **unchanged** |
| `main.py:3627-3633`, the τ-table Gauss order | `IntegrationSolver` | `stepping=BackgroundModel.TAU_GAUSS_ORDER` | production inserts that same stepping under `BackgroundModel.TAU_SOLVER_LABEL_BASE` | **unchanged** |
| `main.py:3635-3639`, the WKB phase Gauss order | `IntegrationSolver` | `stepping=GkWKBIntegration.PHASE_SOLVER_STEPPING` | production inserts that same stepping under `GkWKBIntegration.PHASE_SOLVER_LABEL_BASE` | **unchanged** |
| `docs/source-remediation-verification/analyse_greens_and_source.py:447-448`, exit time at `:247-253` | exit time | `atol`/`rtol` from `DEFAULT_ABS_TOLERANCE`/`DEFAULT_REL_TOLERANCE` = `(1e-10, 1e-8)` | `(1e-10, 1e-9)` | **now misses** — `rtol_serial` no longer matches. The returned `k_exit` is an unpopulated `wavenumber_exit_time` (`available=False`); it is still passed as `k=k_exit` into the `GkSource` lookup at `:277`, whose filter `table.c.wavenumber_exit_serial == k_exit.store_id` becomes `IS NULL` (SQLAlchemy's translation of `== None`) against a `nullable=False` column, so that lookup misses too. The script already checks `if not source.available:` (`:284`) and does what it does for any unavailable source: appends `(k_val, z_val, "GkSource not available")` to `failures` and `continue`s. It does not compute, store or crash |
| `docs/source-remediation-verification/run_quadsource_integrals.py:140-141`, `k_exit_of` at `:186-195` | exit time | same `(1e-10, 1e-8)` pair | `(1e-10, 1e-9)` | **now misses**, for every source and response `k`. `distinct_k` (`:265`) and `pairs` (`:266`) key on `.store_id`, now absent for every entry, so both dicts collapse to **one** entry apiece instead of one per distinct wavenumber (`triples` itself is unaffected: `closes_triangle`, `:63-69`, compares physical `k.k.k` values). No `KeyError` follows — the one collapsed key is still valid — but every `object_get` downstream (`Tk_numeric_cache`/`Tk_WKB_cache`/`Gk_cache`, `:284-345`) is itself built from an unavailable `k`/`q`/`r`, so it too misses. `payload_ok` (`:352-358`) is `False` for every `(k, q, r)` triple, each is counted under `failures["ingredient unavailable"]` (`:361`) and `continue`d past; `compute_QuadSource_integral.remote` (`:365`) is never called. The script finishes having computed and stored nothing |

Both verification scripts are unedited, per the prompt's own charter (§2, §6): they are documents of
the tree they ran on. `[05a-two-verification-scripts-still-query-the-exit-time-under-the-shared-pair]`
is narrowed, not closed, to say so (tolerance-convergence board §4, and the index).

## The 21 `.scalar()` calls judged not to be identity lookups

`grep -n "\.scalar()" Datastore/SQL/ObjectFactories/*.py` gives 29 sites today; L3's eight are
above. The other 21 are all inside a `validate()` method or the `not_validated` reporting loop
beside it, in seven files, three sites each, all the same shape:

| File | Lines | What each selects |
|---|---|---|
| `GkNumericIntegration.py` | `:526`, `:533`, `:617` | `:526` — `table.c.z_samples` filtered `table.c.serial == obj.store_id`: a primary-key lookup, unique by construction. `:533` — `sqla.func.count(value_table.c.serial)` filtered on `integration_serial == obj.store_id`: a count aggregate, always exactly one row. `:617` — the same count, inside the `not_validated` report loop |
| `GkSource.py` | `:491`, `:498`, `:571` | same three-part shape (`z_samples` by `serial == obj.store_id`; `count(...)` by `parent_serial`; the report loop's count) |
| `BackgroundModel.py` | `:848`, `:855`, `:919` | same shape (`z_samples` by `serial == obj.store_id`; `count(...)` by `model_serial`; the report loop's count) |
| `GkWKBIntegration.py` | `:641`, `:648`, `:727` | same shape (`z_samples`; `count(...)` by `wkb_serial`; the report loop's count) |
| `QuadSource.py` | `:367`, `:374`, `:435` | same shape (`z_samples`; `count(...)` by `parent_serial`; the report loop's count) |
| `TkNumericIntegration.py` | `:529`, `:536`, `:619` | same shape (`z_samples`; `count(...)` by `integration_serial`; the report loop's count) |
| `TkWKBIntegration.py` | `:690`, `:697`, `:779` | same shape (`z_samples`; `count(...)` by `wkb_serial`; the report loop's count) |

None of the 21 is an identity lookup: the first of each triple filters on the row's own primary
key (`obj.store_id`, already unique), and the other two select a `sqla.func.count(...)`, which
SQL always returns as exactly one row whatever it counts. `.scalar()` cannot hide a duplicate at
either shape, so L3 leaves all 21 alone.

## The exit time's tolerance joins

Kept, feeding the returned `log10_tol` values (`obj._atol.log10_tol`/`obj._rtol.log10_tol`), as
"What shipped" above states. The alternative — drop the joins and build the returned `atol`/`rtol`
from `target_atol`/`target_rtol` directly, since an exact match guarantees they equal the stored
row's own tolerances — is also correct, but touches more of the function than the prompt's fix
needs and was not required by the read-back guard (the joined columns are not in
`KNOWN_BLIND_COLUMNS` for this site; only the dynamic `z_exit_*` loop columns are). Kept, as an
`IMPLEMENTATION CHOICE` (see "Deviations", item 2).

## L1's key, after this prompt

`sqla_wavenumber_exit_time_factory.store`'s replica path (prompt 01, W3) already keys on exactly
`("wavenumber_serial", "cosmology_type", "cosmology_serial", "atol_serial", "rtol_serial",
"stepping")` (`wavenumber.py`, `store`, `key_columns`, read but not edited by this prompt). The
inventory's identity (`inventory_records`, also read but not edited) is
`leaves=("cosmology_type", "stepping")` plus `parents={"k": Parent("wavenumber_serial",
"wavenumber"), "cosmology": Parent("cosmology_serial", COSMOLOGY, "cosmology_type"), "atol":
Parent("atol_serial", "tolerance"), "rtol": Parent("rtol_serial", "tolerance")}` —
`Datastore/store_inventory.py`'s `read_records` resolves `leaves` and `parents` together into one
canonical key, which is the same six columns. Before this prompt `build()`'s own match was an
inequality on two of those six (the tolerance columns, compared as `log10_tol`, not as serial) and
an inequality on a third (`stepping`), so the lookup, the replica check and the fingerprint could
each treat "the same row" differently. After L1 all three now compare exactly the same six columns
by exact equality, so they agree on one key.

## The within-shard duplicate, under 01/02

A serial-keyed row held twice under two serials, identically on every shard, is not a cross-shard
difference, so prompt 02's `_reconcile_replicated_tables` cannot see it (it compares shards against
each other, and both hold the same duplicate). L3 is the guard for exactly this case. After L3, the
next lookup that reaches such a duplicate raises `MultipleResultsFound` inside the shard's own
`build()`. Under prompt 01, that lookup runs as the controller's call inside
`ShardedPool._replicated_write`, whose `replication_in_flight` record is committed **before** that
call and deleted only after every replica has answered and been checked
(`prompts/datastore-integrity` log 01). The controller's call raising before it returns anything
leaves the record set, with the disk state unchanged (the duplicate was already identical on every
shard, and nothing about a `get` that raises writes anything). That is log 02's reopen-table row
18(a): "any path, a failure the driver sees after `P1`… the shards are the same (a controller that
raised before committing): **cleared**." So the next open clears the stale record and opens
cleanly — and the very next lookup at the same key raises `MultipleResultsFound` again, because
nothing in 01 or 02 repairs a within-shard duplicate; only a person editing the store can. The
result is loud (it raises, every time) and persistent (nothing heals it), which is the intended
outcome: a duplicate is a defect to be noticed and fixed by hand, not something either prompt
should paper over. A stand-in-pool test of this path is the orchestrator note's stated option, not
required by this prompt; none is added here.

## Deviations from the prompt

1. **`redshift`'s zero branch is a Python-level conditional expression bound to one local
   (`z_filter`), passed to a single `.filter(...)` call** — `IMPLEMENTATION CHOICE`. The prompt's
   text says "an explicit branch" and "the one filter expression"; this satisfies both without a
   SQL-level `CASE`, and the divide-by-zero the defect is about never reaches SQLite at `z == 0`
   (the Python branch decides the expression before `conn.execute` runs). The read-back guard is
   unaffected either way, since it does not parse filter expressions.
2. **The exit time's tolerance joins are kept, not dropped** — `IMPLEMENTATION CHOICE`, explained
   above under "The exit time's tolerance joins". Both shapes are correct after L1; keeping the
   joins is the smaller diff and the one the orchestrator's guard-neutral sketch measured.
3. **A one-to-three-line comment above each L3 site**, naming the prompt and board item and, where
   relevant (`integration_metadata.py`, `QCD_Cosmology.py`), the reasoning — `IMPLEMENTATION
   CHOICE`. No filter, column or control flow changed by any of them; `black --check` is clean.
4. **`QCD_Cosmology.py`'s `try … except SQLAlchemyError` handler is read, not changed** —
   `STRUCTURALLY REQUIRED` by the prompt's own instruction ("for `QCD_Cosmology`, read the
   `try … except SQLAlchemyError` handler … and say in the log that you read it"). Read at
   `Datastore/SQL/ObjectFactories/QCD_Cosmology.py:98-117`: it re-raises anything whose message
   does not name `"T_z_representation"`, and `MultipleResultsFound`'s message ("Multiple rows were
   found when one or none was required") never does, so it passes through unchanged.
   `TestDuplicatesRaise.test_qcd_cosmology` asserts `MultipleResultsFound` itself, not the
   handler's `RuntimeError`.
5. **The new test module's `Store` class is its own, not an import of
   `docs/datastore-integrity-audit/lookup_keys/common.py`** — `IMPLEMENTATION CHOICE`, following
   `Datastore/tests/test_integral_read_batch.py`'s precedent (its own `Store` class, same
   construction). The permitted-files list names "one new test module under `Datastore/tests/`",
   not a dependency on `docs/`; both build the schema the same way
   (`build_schema(MetaData(), Datastore.SQL.Datastore._factories)`).
6. **`Store.insert`/`Store.inserter` take the table name as `table_name`, not `name`** —
   `IMPLEMENTATION CHOICE`. `LambdaCDM` and `QCD_Cosmology` both have a `name` column, so a keyword
   argument named `name` collides with a `**row`/`**data` entry of that name; caught by running the
   two duplicate tests once with `name` and fixed before this commit.
7. **`Planck2018()` supplies the seven `LambdaCDM`/`QCD_Cosmology` parameters in the new tests,
   real values rather than invented ones** — `IMPLEMENTATION CHOICE`, following
   `ComputeTargets/tests/test_cosmology_representation_key.py`'s precedent; constructing a real
   `QCD_Cosmology` costs root solves and a banner, but `MultipleResultsFound` is raised by
   `.scalar_one_or_none()` before that construction is reached, so the duplicate test pays none of
   that cost (confirmed: the whole 18-test module runs in well under a second).
8. **The exit-time and `IntegrationSolver` registrations' `"stepping": "minimum"` are untouched** —
   `STRUCTURALLY REQUIRED` by README §5 rule 7 / D5 (06 is the first prompt to amend the schema
   witness). Recorded under "Observations not acted on".

No `UNINTENDED DRIFT` was found. The diff of every touched factory file is confined to the lines
"What shipped" names, plus the comments item 3 lists; nothing in any `register()`, `store()`, or
any `if "serial" in payload` branch changed. `git diff --cached -U0` was read in full to confirm
this before the commit.

## Verification performed

- **The starting tree.** `git status` was clean at `ae4d8f9`.
- **The read-back guard**, run alone first, then with the new module beside it:
  `PYTHONPATH=. ./venv/bin/python -m unittest Datastore.tests.test_factory_select_columns` — 9 OK,
  unmodified, both before and after every code change. No coverage set, blind-spot declaration or
  unresolved-table/insert declaration moved.
- **The new module alone:**
  `PYTHONPATH=. ./venv/bin/python -m unittest Datastore.tests.test_exact_identity_lookups -v` —
  18 OK.
- **The whole `Datastore` suite**, several times over the course of the work (after the fix, after
  each mutation, after each revert, and once more on the final tree):
  `PYTHONPATH=. ./venv/bin/python -m unittest discover -s Datastore/tests -t .` — **343 OK** on the
  final tree.
- **The six suites**, with README §7's command, serially (`ComputeTargets` run alone, so the known
  wall-clock flake has no parallel load to fire under — it did not fire):

  | Suite | Result |
  |---|---|
  | AdaptiveLevin | 32 OK |
  | ComputeTargets | 552 OK |
  | CosmologyModels | 39 OK |
  | Datastore | **343 OK** (325 + the 18 of the new `test_exact_identity_lookups.py`) |
  | LiouvilleGreen | 148 OK (1 skipped) |
  | RunRegistry | 199 OK |

- **Formatting.** `./venv/bin/python -m black --check` on every file this prompt touched (the ten
  factory files and the new test module) is clean.
- **Scope**, checked with `git diff --cached -U0` before drafting this log, and re-checked after
  finishing it: the staged hunks are exactly the ones "What shipped" and "Deviations" describe. No
  `register()`, `store()`, or `if "serial" in payload` branch is touched; no existing test file is
  touched; `Datastore/tests/test_factory_select_columns.py` and
  `Datastore/tests/data/schema_at_base.json` are untouched (`git status` lists neither).
  `Datastore/store_inventory.py` and `wavenumber.py`'s `store()` were read, for the "L1's key"
  section above, and are unedited.
- **Every production caller** the L5 table names was read, not run — none of them was executed, and
  nothing under `var/` was opened by anything in this session.
- **No Ray.** `ray` is imported nowhere in the new test module or in any edited factory file;
  `tearDownModule` in the new module raises if `ray.is_initialized()` is ever `True`, and every run
  above completed with it unset.

## The deliberate-breakage record

Each diff below was made by editing the working tree **after staging the fix** (`git add` on the
ten factory files and the new test module), so that `git diff` shows only the mutation, not the
mutation plus the fix. Applying it with `git apply` from the repository root, atop this commit,
reproduces exactly the tree the failure counts below were measured on. Each was applied, the whole
`Datastore` suite run, the failures recorded, and the file restored with `git checkout --
<file>` (which, at the time, restored the staged fix, not the pre-prompt code) before the next
mutation. None is committed.

- **Counts** are `unittest`'s, which counts each failing sub-test (`subTest`) separately.

**(i) Put `<=` back on `rtol` in the exit-time lookup**, leaving `atol_serial` and `stepping` exact.

```diff
diff --git a/Datastore/SQL/ObjectFactories/wavenumber.py b/Datastore/SQL/ObjectFactories/wavenumber.py
index baa7d0f..cf967f7 100644
--- a/Datastore/SQL/ObjectFactories/wavenumber.py
+++ b/Datastore/SQL/ObjectFactories/wavenumber.py
@@ -224,7 +224,8 @@ class sqla_wavenumber_exit_time_factory(SQLAFactoryBase):
                         # other keyed type uses, not the "at least as tight" inequality this
                         # lookup used to apply
                         table.c.atol_serial == target_atol.store_id,
-                        table.c.rtol_serial == target_rtol.store_id,
+                        rtol_table.c.log10_tol - target_rtol.log10_tol
+                        <= DEFAULT_FLOAT_PRECISION,
                         table.c.stepping == target_stepping,
                     )
                 )
```

`failures=1`:
- FAIL `test_exact_identity_lookups.TestExitTimeExactMatch.test_looser_and_tighter_requests_miss`
  (atol=-10.0, rtol=-8.0)

The other two looser/tighter cases in that sub-tested method (`atol=-8.0, rtol=-8.0` and
`atol=-12.0, rtol=-9.0`) still miss, because `atol_serial` is still exact; only the request whose
`rtol` alone is looser than the stored row (`-8.0` against stored `-9.0`) is now wrongly served.

**(ii) Put the query back inside the loop.**

```diff
diff --git a/Datastore/SQL/ObjectFactories/wavenumber.py b/Datastore/SQL/ObjectFactories/wavenumber.py
index baa7d0f..70b0d6c 100644
--- a/Datastore/SQL/ObjectFactories/wavenumber.py
+++ b/Datastore/SQL/ObjectFactories/wavenumber.py
@@ -234,7 +234,7 @@ class sqla_wavenumber_exit_time_factory(SQLAFactoryBase):
             for z_offset in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
                 query = query.add_columns(table.c[f"z_exit_subh_e{z_offset}"])
 
-            row_data = conn.execute(query).one_or_none()
+                row_data = conn.execute(query).one_or_none()
         except MultipleResultsFound as e:
             print(
                 f"!! wavenumber_exit_time.build(): multiple results found when querying for wavenumber_exit_time"
```

`failures=1`:
- FAIL `test_exact_identity_lookups.TestExitTimeOneStatement.test_build_executes_exactly_one_select`

The exact-match tests are unaffected: the last iteration's result is still complete and correct, so
only the statement count moves (to six), not any answer.

**(iii) Put `>=` back in `IntegrationSolver`, leaving `.scalar_one_or_none()` in place.**

```diff
diff --git a/Datastore/SQL/ObjectFactories/integration_metadata.py b/Datastore/SQL/ObjectFactories/integration_metadata.py
index 9f045b3..13ace76 100644
--- a/Datastore/SQL/ObjectFactories/integration_metadata.py
+++ b/Datastore/SQL/ObjectFactories/integration_metadata.py
@@ -33,7 +33,7 @@ class sqla_IntegrationSolver_factory(SQLAFactoryBase):
         # silently serving the first row scalar() would have returned
         store_id = conn.execute(
             sqla.select(table.c.serial).filter(
-                sqla.and_(table.c.label == label, table.c.stepping == stepping)
+                sqla.and_(table.c.label == label, table.c.stepping >= stepping)
             )
         ).scalar_one_or_none()
 
```

`failures=1, errors=1`:
- ERROR `test_exact_identity_lookups.TestIntegrationSolverExactMatch.test_request_4_is_served_serial_2`
- FAIL `test_exact_identity_lookups.TestIntegrationSolverExactMatch.test_request_6_inserts_a_new_row`

With `>=` restored, request 4 matches **both** stored rows (stepping 8 and stepping 4), so
`scalar_one_or_none()` raises `MultipleResultsFound` — an error, not the expected `store_id == 2` —
and request 6 matches the stepping-8 row (`8 >= 6`), so it is served that row instead of inserting
one, which fails the assertion that a new row was inserted.

**(iv) Put `.scalar()` back on `tolerance`.**

```diff
diff --git a/Datastore/SQL/ObjectFactories/tolerance.py b/Datastore/SQL/ObjectFactories/tolerance.py
index 30d8796..bedffa3 100644
--- a/Datastore/SQL/ObjectFactories/tolerance.py
+++ b/Datastore/SQL/ObjectFactories/tolerance.py
@@ -35,7 +35,7 @@ class sqla_tolerance_factory(SQLAFactoryBase):
             sqla.select(table.c.serial).filter(
                 sqla.func.abs(table.c.log10_tol - log10_tol) < DEFAULT_FLOAT_PRECISION
             )
-        ).scalar_one_or_none()
+        ).scalar()
 
         if store_id is None:
             insert_data = {"log10_tol": log10_tol}
```

`failures=1`:
- FAIL `test_exact_identity_lookups.TestDuplicatesRaise.test_tolerance`

`.scalar()` silently serves the first row instead of raising, so the test's
`assertRaises(MultipleResultsFound)` fails because nothing was raised.

**(v) Remove the zero branch.**

```diff
diff --git a/Datastore/SQL/ObjectFactories/redshift.py b/Datastore/SQL/ObjectFactories/redshift.py
index 8a29c7d..345a3c2 100644
--- a/Datastore/SQL/ObjectFactories/redshift.py
+++ b/Datastore/SQL/ObjectFactories/redshift.py
@@ -35,12 +35,7 @@ class sqla_redshift_factory(SQLAFactoryBase):
         # divides by z, and SQLite's 0/0 is NULL, which matches nothing, so every call at z = 0
         # would insert a fresh row (prompts/datastore-integrity, prompt 03, L4, audit §3.9).
         # z == 0 is matched by equality instead; every other z keeps the relative match unchanged.
-        z_filter = (
-            table.c.z == 0.0
-            if z == 0.0
-            else sqla.func.abs((table.c.z - z) / z)
-            < DEFAULT_REDSHIFT_RELATIVE_PRECISION
-        )
+        z_filter = sqla.func.abs((table.c.z - z) / z) < DEFAULT_REDSHIFT_RELATIVE_PRECISION
         query = sqla.select(
             table.c.serial,
             table.c.source,
```

`failures=1`:
- FAIL `test_exact_identity_lookups.TestRedshiftZero.test_three_builds_at_zero_give_one_row`

With the branch gone, `z = 0` divides by zero; SQLite's `0/0` is `NULL`, which never satisfies `<`,
so every call misses and inserts, and three calls leave three rows rather than one.
`test_z_0pt1_is_unchanged` is unaffected, since `z = 0.1` never takes the removed branch.

**The replay, before the commit.** All five diffs above were extracted from this file, in its
final form, to five files under this agent's scratch directory
(`/private/tmp/claude-35086/-Users-ds283-Documents-Code-SecondaryGWKit/99606255-686d-4741-8d31-0da68d9e2ce0/scratchpad/agent03/`),
and each applied in turn with plain `git apply <file>` from the repository root, atop the staged
fix. Each gave exactly the failure counts and test names above, and `git apply -R <file>` (followed
by `git diff --cached --stat` showing no residual change) restored the tree between mutations. This
is reported below, under "Verification performed (the replay)".

## Observations not acted on

1. **The exit-time and `IntegrationSolver` registrations' `"stepping": "minimum"` flags.** Neither
   `register()` was touched, per README §5 rule 7 and D5: 06 is the first schema-changing prompt,
   and its README §2 row says it makes these two flags "say what prompt 03 made true". Nothing
   reads `"stepping"` in `register()`'s return value except `Datastore/SQL/schema.py`'s
   `use_stepping`/`stepping_mode` bookkeeping (read, not changed), so this changes no behaviour
   today; it is a label now describing the wrong match (`"minimum"` when `build()` matches
   `==`), left for prompt 06 to correct. Already recorded on the campaign board (README §2's own
   table row for 06) and in the four owning boards' closing notes above; no new issue opened, since
   the campaign's own §2 already tracks it.
2. **The `QCD_Cosmology` duplicate test's cost.** Constructing a real `QCD_Cosmology` costs root
   solves (`ComputeTargets/tests/test_cosmology_representation_key.py`'s docstring: "500 root
   solves and prints a banner"), but `TestDuplicatesRaise.test_qcd_cosmology` never reaches that
   construction: `.scalar_one_or_none()` raises before `build()` gets to
   `obj = QCD_Cosmology(...)`. Measured: the whole 18-test module runs in well under a second.
   Nothing to act on.
3. **`[02-the-version-row-is-written-outside-the-in-flight-record]`** is open and is not this
   prompt's. `version.build` is one of L3's eight sites; making it `.scalar_one_or_none()` neither
   fixes nor worsens that issue, which is about where the row is written, not how it is looked up.
4. **Nothing here has run under Ray or on a real store.** Every test builds its own in-memory
   schema; the two verification scripts' new behaviour (the L5 table) is read, not measured by
   running them.

## State handed to the next prompt

- **Names.** No new public name: `sqla_wavenumber_exit_time_factory.build`,
  `sqla_IntegrationSolver_factory.build`, and the six other `build()`s keep their signatures.
- **For prompt 06.** The two registrations' `"stepping": "minimum"` flags are the first thing its
  README §2 row names; this log's "Observations not acted on" item 1 is the pointer.
- **For any later reader of the exit time or `IntegrationSolver`.** Both now match exactly one row
  or none, and raise `MultipleResultsFound` on a duplicate — the same contract every other keyed
  type has always had.
- **Baselines** after this commit: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39,
  Datastore **343**, LiouvilleGreen 148 (1 skipped), RunRegistry 199.
