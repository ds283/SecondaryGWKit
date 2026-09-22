# Log 01 — The `QuadSourceIntegral` read-back, and the guard that should have caught it

**Prompt:** [`prompts/datastore-readback/01-quadsourceintegral-readback.md`](../01-quadsourceintegral-readback.md)
**Commit:** *(this commit)* — "Add the missing numeric_quad column to the QuadSourceIntegral read"
**Model:** Claude Opus 5
**Date:** 2026-09-22
**Result:** COMPLETE

## What shipped

**One line of production code.** `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py:246` —
`table.c.numeric_quad,` added to the `SELECT` that `build()` executes, in the position the sibling
`SELECT` in `read_batch()` puts it (after `total_phase_limited`, before `WKB_Levin`). The diff of
that file is one insertion and nothing else: the query is not reorganised, the two `SELECT`s are
not factored together, the schema is untouched and `build()` is unchanged.

**A new test root, `Datastore/tests/`**, with `__init__.py` and one module:

**`Datastore/tests/test_factory_select_columns.py`** (516 lines, 4 test methods) — the static
guard. It parses every module under `Datastore/SQL/ObjectFactories/` with `ast`, following
`ComputeTargets/tests/test_main_plumbing.py`'s precedent of reading source statically rather than
executing it, and for each factory's `build()` collects

- the column names its `sqla.select(...)` — plus any `.add_columns(...)` — requests, resolving
  `table.c.NAME`, `table.c["NAME"]` and `<expr>.label("NAME")`, and
- the attribute names read off the variable the executed query was assigned to,

then asserts the second is a subset of the first. Public surface: `FACTORY_DIR`, `_column_name`,
`_QuerySpec`, `Finding`, `analyse_build`, `analyse_factories`, `KNOWN_UNFIXED`,
`EXPECTED_COVERAGE`, `KNOWN_BLIND`, `TestFactorySelectColumns`.

No Ray, no datastore, no SQLAlchemy engine: the test never runs a query, and so cannot be defeated
by the absence of one. It runs in 0.3 s.

The four methods are:

| Test | What it holds |
|---|---|
| `test_build_reads_only_columns_its_select_requests` | the guard proper: decidable shortfalls equal `KNOWN_UNFIXED` exactly |
| `test_known_unfixed_entries_are_still_present` | a `KNOWN_UNFIXED` entry that has been fixed must be deleted, not left to rot |
| `test_analyser_coverage_is_undiminished` | the 20 row-variable bindings the analyser can decide are pinned by name |
| `test_blind_spots_are_the_declared_ones` | the one binding it cannot decide is declared, with its reason |

## Which guard shape, and why

**Shape (a), static consistency**, as the prompt recommended, and it did not have to be weakened to
pass. Shape (b) — an in-memory round trip — would have been unambiguous for `QuadSourceIntegral`
and would have said nothing about the other thirteen factories that read rows; the defect being
guarded against is a *class*, and the class is what a static check can see. The round-trip fallback
was not needed and was not written.

Two things had to be got right for (a) to be honest rather than approximately honest, and both were
found by running it rather than by reasoning about it:

1. **Dynamically named columns cannot convict.** `sqla_wavenumber_exit_time_factory.build` extends
   its query in a loop with `table.c[f"z_exit_suph_e{z_offset}"]`. Such a column can only *enlarge*
   the requested set, so a binding with no shortfall is still soundly cleared; but a binding *with*
   a shortfall may be one the loop supplies, so it is reported as undecidable rather than as a
   defect. `Finding.decidable` encodes exactly that asymmetry.

2. **A rebound variable is two bindings, not one.** The first version of the analyser collected
   reads per variable *name* over the whole `build()`. `sqla_BackgroundModelFactory.build` uses
   `row` first in a set comprehension over its lookup rows and then in a loop over its sample rows,
   and merging the two produced a **false positive**: the grid-identity reads of the first `row`
   charged against the second `row`'s `SELECT`. That is precisely the "false positive you would
   have to special-case away" the prompt warned about, and special-casing it away would have been
   the wrong answer. The analyser now emits one `Finding` per binding and charges a read to
   whichever binding is live above it in the source text.

### What it cannot see

Stated in the module docstring in full; in summary, six holes, three of which exist in this tree
today:

1. **Dynamic attribute access.** `getattr(row, name)` and `row._mapping[name]` are not
   `ast.Attribute` nodes and are not collected. **Live:**
   `sqla_wavenumber_exit_time_factory.build` reads all of its `z_exit_suph_e*` / `z_exit_subh_e*`
   columns through `row_data._mapping[f"..."]`, and this guard says nothing about them.
2. **Dynamically named columns**, per (1) above — cannot convict, only clear. **Live:** same
   factory.
3. **Only `build()`.** `read_batch()` reads rows too, through a nested `make_object(row)` whose
   `row` is a function parameter the analyser cannot link to a query, and whose `SELECT` is
   extended by `add_columns` in a loop. `read_batch` is **not checked at all**. Opened as
   `[01-read-batch-is-outside-the-guard]`.
4. **Rows whose query cannot be resolved.** **Live:**
   `sqla_BackgroundModelFactory.build`'s lookup row comes from `rows[0]` where `rows` came from a
   nested `_build_query(with_grid_identity)` whose column list is assembled conditionally, with the
   reads of the two grid-identity columns guarded by `hasattr`. The shortfall there is genuinely
   branch-dependent and no column-set comparison can decide it; declared in `KNOWN_BLIND`. Its 17
   reads are unguarded.
5. **Control flow.** Reads are charged by source order, not by reachability. Every `build()` here
   has the straight-line `look up, then unpack` shape, for which the two agree.
6. **Column existence.** A `SELECT` naming a column the *table* does not have is caught by
   SQLAlchemy at query-build time, not here. This guard compares reads against requests, not
   requests against the schema.

Holes 3, 4 and 5 are why the guard's own reach is pinned. A static approximation that stops seeing
a factory goes on passing, which is the failure mode that costs everything to detect, so
`EXPECTED_COVERAGE` and `KNOWN_BLIND` are asserted as exact sets: a refactor that moves a read out
of sight fails the suite instead of silently reducing coverage.

## The deliberate-breakage record

Required by prompt §6 item 3: the guard must be shown to fail on the bug it was written for.

**Direction 1 — unfixed code, guard fails.** With the one-line fix stashed
(`git stash push -- Datastore/SQL/ObjectFactories/QuadSourceIntegral.py`),
`test_build_reads_only_columns_its_select_requests` **FAILED**, naming the defect:

```
AssertionError: {'BackgroundModel.py:sqla_BackgroundModelValue_factory': ['Hubble'],
                 'QuadSourceIntegral.py:sqla_QuadSourceIntegral_factory': ['numeric_quad']}
             != {'BackgroundModel.py:sqla_BackgroundModelValue_factory': ['Hubble']}
```

`Ran 4 tests ... FAILED (failures=1)`. The other three passed, as they should: coverage and blind
spots are unchanged by the defect.

**Direction 2 — fixed code, guard passes.** With the fix restored, `Ran 4 tests in 0.373s / OK`.

The same breakage was applied to the **real store** as well, not only to the static check — see the
next section.

## Verification performed

### 1. The read-back, against the real datastore, read-only

A throwaway script under the session scratch directory (not committed) instantiated the real
`Datastore` class — `Datastore.__ray_metadata__.modified_class`, the class behind the `@ray.remote`
decorator — via `object.__new__`, bypassing `__init__` because `__init__` calls `_ensure_tables()`
and `_validate_on_startup()`, which write. Its engine was opened on
`sqlite:///file:<shard>?mode=ro&uri=true`, so a write would have raised rather than succeeded. The
schema was built by the production `_build_schema()` from the production `_factories`, and the row
was fetched by calling **`Datastore.object_get` itself**, not a re-implementation of it.

Row `serial = 20806` of `handover-A3-baseline-lambdacdm-shard0000.sqlite`, chosen as the largest
`|numeric_quad|` in that shard so the value is unmistakable:

```
stored row serial=20806 in handover-A3-baseline-lambdacdm-shard0000.sqlite
  numeric_quad (raw sqlite) = -1.2033078530407802e-11
  object_get returned    = QuadSourceIntegral, store_id=20806, available=True
  obj.numeric_quad       = -1.2033078530407802e-11
  obj.total              = -1.2033078530407802e-11
  match numeric_quad     = True
  match total            = True
```

**The stored `numeric_quad` is `-1.2033078530407802e-11`, and `object_get` returns it exactly** —
equal as a float, not merely close.

The same script against the **unfixed** tree, same row, same store:

```
AttributeError: Could not locate column in row for column 'numeric_quad'
```

which is the failure of the manifest's `run_history[1]`, reproduced on demand in two seconds
instead of two minutes of replay.

The store was read and not written: the four shard files' mtimes are unchanged at
`2026-09-21 04:32`, before the failed resume. `var/datastores/backup-pre-resume-20260921T091011`
was not touched. The pipeline was not started.

### 2. The suites

Baselines are campaign README §4's, measured at `ab7079c`. `AdaptiveLevin` had never been
baselined; **it is baselined here at 32, OK**, measured on the pre-change tree.

| Suite | Baseline | This tree | Verdict |
|---|---|---|---|
| `ComputeTargets` | 552 | **552**, `FAILED (failures=1)` | the known flake, see below |
| `CosmologyModels` | 39 | **39**, OK | unchanged |
| `LiouvilleGreen` | 148 (skipped=1) | **148**, OK (skipped=1) | unchanged |
| `AdaptiveLevin` | **32 (new baseline)** | **32**, OK | unchanged |
| `Datastore` | — (did not exist) | **4**, OK | +4, exactly the tests added |

The single `ComputeTargets` failure is
`ComputeTargets.tests.test_tk_wkb_phase.TestCost.test_wall_time_per_object`, the wall-clock flake
campaign README §4 and `CLAUDE.md` both name. Re-run alone on this tree:
`Ran 17 tests in 1.913s / OK`. Nothing in this commit is in that module's reach — the diff is one
column in a `SELECT` and a new test root — and the count is unchanged at 552.

`black --check` is clean over the whole tree.

## The audit — all factories, findings opened not fixed

`Datastore/SQL/ObjectFactories/` holds **38 classes that define `build()`** across 21 modules.
(The campaign README's "22 factories, 12 of them reading `row_data`" counts the 37 entries of
`Datastore/SQL/Datastore.py`'s `_factories` registry differently; the numbers below are what the
guard measures, class by class, and are the ones to trust.)

**24 of the 38 read nothing off a query result** — they insert, or reduce with `.scalar()`. Checked
independently of the guard by scanning every one of them for an attribute read, an `_mapping`
subscript or a `getattr` on any row-named variable: none has one. They are the 12 tag-association
factories and value factories plus `sqla_GkSourcePolicy_factory`, `sqla_LambdaCDM_factory`,
`sqla_QCDCosmology_factory`, `sqla_QuadSourcePolicy_factory`, `sqla_IntegrationSolver_factory`,
`sqla_store_tag_factory`, `sqla_tolerance_factory`, `sqla_version_factory` and the `SQLAFactoryBase`
stub.

**14 read rows**, across **21 row-variable bindings**. Twenty are decidable and one is declared
blind:

| Factory | binding | reads | verdict |
|---|---|---|---|
| `BackgroundModel.py:sqla_BackgroundModelFactory` | `row_data` | 17 | **BLIND** — `rows[0]` of a conditionally-built query in a nested helper |
| `BackgroundModel.py:sqla_BackgroundModelFactory` | `row` | 20 | clean |
| `BackgroundModel.py:sqla_BackgroundModelValue_factory` | `row_data` | 16 | **DEFECT — `Hubble`**, opened below, not fixed |
| `GkNumericIntegration.py:sqla_GkNumericIntegration_factory` | `row_data`, `row` | 23, 13 | clean |
| `GkSource.py:sqla_GkSource_factory` | `row_data`, `row` | 14, 22 | clean |
| `GkSourcePolicyData.py:sqla_GkSourcePolicyData_factory` | `row_data` | 6 | clean |
| `GkWKBIntegration.py:sqla_GkWKBIntegration_factory` | `row_data`, `row` | 33, 15 | clean |
| `OneLoopIntegral.py:sqla_OneLoopIntegral_factory` | `row_data` | 5 | clean |
| `QuadSource.py:sqla_QuadSource_factory` | `row_data`, `row` | 6, 14 | clean |
| `QuadSourceIntegral.py:sqla_QuadSourceIntegral_factory` | `row_data` | 32 | **clean after the fix**, `numeric_quad` before it |
| `TkNumericIntegration.py:sqla_TkNumericIntegration_factory` | `row_data`, `row` | 23, 13 | clean |
| `TkWKBIntegration.py:sqla_TkWKBIntegration_factory` | `row_data`, `row` | 38, 16 | clean |
| `redshift.py:sqla_redshift_factory` | `row_data` | 3 | clean |
| `wavenumber.py:sqla_wavenumber_factory` | `row_data` | 3 | clean |
| `wavenumber.py:sqla_wavenumber_exit_time_factory` | `row_data` | 8 | clean; 2 of its columns are dynamically named and 6 of its reads go through `_mapping`, so the clearance is partial |

**One additional defect of the same class was found, and was not fixed**, per prompt §4 and campaign
README §5 rule 4. It is carried in the guard's `KNOWN_UNFIXED` with the issue that owns it, so the
suite passes on this tree and will fail the moment the entry is stale.

### `[01-backgroundmodelvalue-hubble]`

`Datastore/SQL/ObjectFactories/BackgroundModel.py:1003` and `:1005`, in
`sqla_BackgroundModelValue_factory.build()`, read **`row_data.Hubble`** on the branch taken when the
row already exists — the consistency check that the stored Hubble rate matches the one being
written, and the `ValueError` message that reports the mismatch. The column is **`Hubble_GeV`**;
the `SELECT` at `:926` requests `Hubble_GeV` and the table has no `Hubble` column at all. Any
execution of that branch raises `NoSuchColumnError` — and the check it defeats is a data-integrity
check, so the failure mode is a hard error, not a silent wrong answer.

**Reachability: not currently reachable from a running pipeline, nor from a resume.** Prompt §7's
third stop condition is therefore **not** met, and the user did not need to be interrupted.
`BackgroundModel.build()` does not go through this factory — it reads its sample rows with its own
`SELECT` over `tables["BackgroundModelValue"]` (`:454-486`) and constructs the
`BackgroundModelValue` objects itself. Nothing in the tree calls
`object_get("BackgroundModelValue", ...)`; the factory is registered, and its `register()` and
`store()` are used, but its `build()` is not called by anything. Independent corroboration that it
is dead: the `row_data is None` branch inserts with the key **`"wkb_serial"`**, which is not a
column of this table either (`register()` declares `model_serial`), so that branch cannot ever have
run successfully. The factory's `build()` has two defects and neither has ever fired.

**Impact:** prospective. Any future caller of `object_get("BackgroundModelValue", ...)` — a
diagnostic, an extract script, a per-sample lookup — hits it on the first stored row.

**Next step:** `row_data.Hubble` → `row_data.Hubble_GeV * GeV` in both places (the comparison is
against `Hubble`, which has already been converted to internal units above), and separately decide
whether the `"wkb_serial"` insert key is a typo for `"model_serial"` or evidence the whole `build()`
should be deleted. One prompt, in this campaign or the next that touches `Datastore/`.

### `[01-read-batch-is-outside-the-guard]`

The guard covers `build()` only. `read_batch()` reads rows too —
`sqla_QuadSourceIntegral_factory.read_batch` at `:530` alone reads some sixty attributes off a
`row` that is a parameter of a nested `make_object`, from a query extended by `add_columns` in two
loops — and is checked by nothing. The defect class this campaign exists for is equally possible
there, and `read_batch` is on the **live** path (`Datastore.object_read_batch`), not only on
resume; it has been exercised, which is weak evidence that it is currently clean, but it is not a
guarantee for the columns a given call does not touch.

**Impact:** the guard's stated coverage is `build()`, and a reader could take "the factories are
guarded" to mean more than it does.

**Next step:** either extend the analyser to link a nested helper's row parameter to the query it is
called with, or write the round-trip test of prompt §3 shape (b) for `read_batch`. Not attempted
here: prompt §3 scopes the guard to what `build()` reads, and widening it is a change with its own
risk that would obscure the one-line fix this commit exists to prove.

## Deviations from the prompt

1. **The guard is pinned in three places, not one.** Beyond the subset assertion the prompt asked
   for, the module pins `EXPECTED_COVERAGE`, `KNOWN_BLIND` and the currency of `KNOWN_UNFIXED`, in
   three further test methods. — **IMPLEMENTATION CHOICE.** Prompt §3 requires the guard to say
   what it cannot see and warns that "a guard that oversells its coverage is worse than none". A
   docstring says it to a reader; these three assertions say it to the next refactor. Without them
   the guard silently degrades to nothing the first time a `build()` is rewritten.

2. **The second defect is carried in `KNOWN_UNFIXED` rather than left failing.** — **STRUCTURALLY
   REQUIRED.** Prompt §4 forbids fixing it and prompt §6 item 7 requires the suites to be green;
   the two can only both hold if the guard records the known defect explicitly. The entry names the
   issue and `test_known_unfixed_entries_are_still_present` deletes the loophole when the fix lands.

3. **The audit reports 38 `build()` classes and 14 row-readers, where the prompt says 22 and 12.**
   — **UNINTENDED DRIFT**, in the prompt's figures rather than in the work. Both numbers are stated
   above with how they were counted; nothing was skipped, and the discrepancy is a counting
   convention (registry entries vs. classes defining `build()`), not a gap in the sweep.

4. **The read-back demonstration builds a `Datastore` instance through `object.__new__` rather than
   its constructor.** — **STRUCTURALLY REQUIRED.** Prompt §2 forbids writing to the store and §7
   makes any write a stop condition; `Datastore.__init__` calls `_ensure_tables()` and
   `_validate_on_startup()`. The path actually exercised — `object_get` → `factory.build` → the
   `SELECT` — is the production one, unmodified, and the engine was read-only so that the
   constraint was enforced rather than trusted.

## Observations not acted on

1. **`sqla_BackgroundModelValue_factory.build` inserts with the key `"wkb_serial"`**
   (`BackgroundModel.py:964`), which `register()` does not declare — the column is `model_serial`.
   Recorded inside `[01-backgroundmodelvalue-hubble]` above rather than as its own issue, because it
   is the same dead `build()` and the same decision: repair it or delete it.

2. **The two `SELECT`s in `QuadSourceIntegral.py` remain duplicated**, now with identical column
   sets modulo ordering and `read_batch`'s joins. Prompt §5 forbids factoring them together and is
   right to: it would obscure the one-line diff this commit exists to prove. It stays a candidate
   for a later prompt, and the guard makes the duplication safe to leave, not safe to forget.

3. **`sqla_wavenumber_exit_time_factory`'s `_mapping` reads are the guard's largest live hole.**
   Six of its eight decidable reads are ordinary attributes; its 20-odd horizon-crossing columns are
   read by name at run time and are outside the check entirely. Not opened as a separate issue: it
   is hole 1 of the docstring's list and belongs with `[01-read-batch-is-outside-the-guard]` if the
   analyser is ever extended.

## State handed to the next prompt

- `QuadSourceIntegral` rows read back. The interrupted run is resumable; finishing it is the user's,
  with the manifest's `restart` command. `var/datastores/backup-pre-resume-20260921T091011` is
  intact and should stay until a resume completes cleanly.
- `Datastore/tests/` exists as a test root with 4 tests, and must be added to whatever runs the
  suites:
  `PYTHONPATH=. ./venv/bin/python -m unittest discover -s Datastore/tests -t .`
- `AdaptiveLevin` now has a baseline: **32**.
- Two issues open in §3 of the board, neither fixed here.
