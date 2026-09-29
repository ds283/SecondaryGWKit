# Log 07 — Key each WKB row on the numeric row it starts from

**Prompt:** [`prompts/datastore-integrity/07-key-wkb-rows-on-their-numeric-row.md`](../07-key-wkb-rows-on-their-numeric-row.md)
**Commit:** *(this commit)* — "Key each WKB row on the numeric row it starts from"
**Model:** Claude Opus 5.5
**Date:** 2026-09-29
**Result:** COMPLETE. K1 and K2 shipped, on the design stated to the orchestrator and approved by
the user on 2026-09-29, under the four decisions of the orchestrator note's §0.1–§0.4, the
structurally required grant of §0.5, and two further grants the user made the same day after this
prompt stopped once (below):
- **Grant A.** `ComputeTargets/tests/test_gauss_order_key.py` joins §0.1's list, limited to
  following the new key;
- **Grant B.** The read-back guard's pins gain exactly the new sites 07 introduces;
- **Three `main.py` behaviour changes** accepted, each STRUCTURALLY REQUIRED (Deviations 1–3).

**The stop, and what released it.** The first attempt stopped before any code was kept. A throwaway
patch (the column, `build()` refusing a payload without a parent, `store()` writing the parent,
and no consistency check) made `test_gauss_order_key` report `FAILED (errors=16)` of 24 tests
across 13 methods. That module drives both WKB factories' `build()` with no parent and their
`store()` with `conn=None`. It was not in §0.1's list. The patch was reverted with `git checkout`,
and the question went to the orchestrator. It also named the guard's `EXPECTED_COVERAGE`, which a
store-time read of the parent row must move. The user granted A and B as worded, and accepted the
three behaviour changes.

Suites after this commit, serially:
- ComputeTargets **552 → 566** (the 14 tests of the new `test_wkb_numeric_parent_main.py`);
- Datastore **351 → 397** (the 41 of `test_wkb_numeric_parent_key.py` and the 5 of
  `test_wkb_parent_foreign_key.py`);
- AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped) and RunRegistry 199,
  unchanged.

The index `docs/OPEN_ISSUES.md` goes from 97 to **98** rows: one deleted and two opened, both on
this campaign's board §3.3.

## What shipped

**K1 — the key** (`Datastore/SQL/ObjectFactories/TkWKBIntegration.py`, `GkWKBIntegration.py`).
- **`numeric_serial`**, indexed, with a foreign key to `TkNumericIntegration.serial` and
  `GkNumericIntegration.serial` respectively. It sits after `model_serial` in `register()`.
  - **Tk: NOT NULL.**
  - **Gk: nullable, and NULL means pure WKB.**
  - Each module's docstring gains a schema note for prompt 07.
- **`build()`** takes the parent, or the explicit reader mode, never neither and never both:
  - `numeric_serial`: a serial filters `numeric_serial == serial`. For Gk,
    `numeric_serial=GkWKBIntegration.PURE_WKB` filters `numeric_serial IS NULL`
    (`_numeric_parent_filter`, one module-level helper per factory).
  - `labels_only=True`: the physical labels only (D1). It first selects the one column
    `numeric_serial` from the lookup's own query. If more than one row matches, it raises
    `MultipleResultsFound` naming their values, a NULL as `pure WKB`, and otherwise serves the one
    row or reports a miss.
  - Neither given: `RuntimeError` (Tk and Gk).
  - Both given: `ValueError`.
  - A Tk lookup given `PURE_WKB`: `TypeError`.
  - `build()` now selects `numeric_serial` and hands it to the constructor.
- **`z_init` and the initial values left the key.** The `if z_init is not None:` absolute
  `DEFAULT_FLOAT_PRECISION` filter is removed from both `build()`s. The columns stay stored, as
  payload, and are still selected and reported.
- **The consistency check**, `_check_numeric_parent(obj, conn, tables)`, a new staticmethod of each
  integration factory. It is called as the first statement of `store()`, before any insert, so a
  refused object writes nothing (tested).
  - **Tk, and Gk seeded.** It reads the parent row on this shard, outer-joined to its
    `wavenumber_exit_time` row (replicated, so present). It refuses:
    - a parent that is not there;
    - a parent for another `k`, another model or, for Gk, another `z_source`;
    - a parent whose stop point cannot be recomputed (a NULL `z_exit` or `stop_deltaz_subh`);
    - any of `z_init`, `T_init`/`G_init` and `Tprime_init`/`Gprime_init` that is not bit for bit
      (`struct.pack("<d", ...)`) `z_exit - stop_deltaz_subh`, `stop_T`/`stop_G` and
      `stop_Tprime`/`stop_Gprime`.
  - **Gk pure WKB.** It reads the `z_source` redshift row and refuses `G_init != 0.0`,
    `Gprime_init != 1.0`, or a `z_init` that is not the row's `z` bit for bit.
  - Every value refusal is a `ValueError` naming both values.
- **`store()`** writes `numeric_serial`, NULL for a pure-WKB row.
- **`inventory_records`**:
  - leaves `("rho_gauss_order",)`, with `z_init` out;
  - a new key parent `"numeric": Parent("numeric_serial", "TkNumericIntegration")`;
  - for Gk, `Parent("numeric_serial", "GkNumericIntegration", nullable=True)`.
- **The value reads** (§0.3): `TkWKBValue`/`GkWKBValue` `_build_impl_model` and `read_batch` take
  an optional `numeric_serial` in place of 05c's optional `z_init`, through the same
  `_numeric_parent_filter`, so the Gk ones also take `PURE_WKB`. With none given they span every
  parent, and 05c's refusal of a second `wkb_serial` is unchanged.
- **A store that predates the column (D4).** Both `build()`s' `SQLAlchemyError` handler now branches
  on the driver's message (`str(getattr(e, "orig", e))`). If it names `numeric_serial`, it raises a
  `RuntimeError`: *"the … table has no "numeric_serial" column. This datastore predates WKB rows
  being keyed on the numeric integration they start from (prompts/datastore-integrity, prompt 07).
  This tree does not read it; it remains readable at the git SHA its sidecar records."* It does not
  say "regenerated". The `rho_gauss_order` branch is unchanged after it.

**K1 — the compute targets** (`ComputeTargets/TkWKBIntegration.py`, `GkWKBIntegration.py`).
- **Tk.** `numeric_serial` and `labels_only` keywords, and a `numeric_serial` property.
  Constructing without a parent raises `ValueError`, unless `labels_only=True`: the object
  `build()` returns for a labels-only miss, whose `compute()` raises.
- **Gk.** The same keywords. Without a parent the object is pure WKB, and its `pure_WKB` property
  says so. The constructor accepts `PURE_WKB` as meaning None, and a labels-only object is neither
  seeded nor pure, and cannot be computed. `GkWKBIntegration.PURE_WKB = "pure-WKB"` is the marker:
  a string, so it survives the trip to a shard and compares by value.
- The parent travels as a serial from construction to the factory's `store()`, the prompt's
  choice.

**K1 — `main.py`** (the table below; README §5 rule 12 shown under Verification):
- **One new module-level helper**, `Gk_WKB_numeric_parent(k_exit, z_source, Gk)`: the Gk stage's
  seeded-or-pure choice, moved unchanged. It includes `z_source_limit`, and its long explanatory
  comment becomes the docstring.
- **Tk WKB stage.** The numeric lookup now runs first, over the whole batch, serial-only as
  before. Any `k` without a numeric row raises, with the same messages (Deviation 1). The existence
  query then passes `numeric_serial`, and the construction passes `numeric_serial=Tk.store_id`.
  The query's `"z_init": None` is gone.
- **Gk WKB stage.** The numeric lookup now runs first, over the whole batch, and gains
  `_do_not_populate` (Deviation 2). The choice is made for every `(k, z_source)` through the helper
  and turned into the key, the parent's serial or `PURE_WKB`. The vectorized existence query
  passes it. The seeded construction passes `numeric_serial=Gk.store_id` and the pure one
  `numeric_serial=GkWKBIntegration.PURE_WKB`. The branch condition becomes `if Gk is not None:`,
  on the helper's result.
- **QSI stage.** After the Tk numeric lookup, a mode whose numeric row is unavailable prints a
  `MISSING DATA WARNING` and the stage raises (Deviation 3). The Tk WKB lookup passes
  `"numeric_serial": Tk_numeric.store_id` for the numeric row `build_QuadSourceIntegral_payload`
  consumes beside it. Its `"z_init": None` is gone.
- `test_main_plumbing.EXPECTED_SITES` did not move: `test_main_plumbing` passes unmodified, with
  TkWKB 3, GkWKB 3, TkNumeric 5 and GkNumeric 3.

**K1 — readers.**
- **`extract_TkWKB_data.py`, `extract_GkWKB_data.py`.** The WKB lookup passes `labels_only=True`,
  the explicit reader mode. It serves a single row, refuses more than one with their
  `numeric_serial` values, and reports a miss as before. For Tk this is the prompt's choice between
  "resolve the numeric first" and "the reader mode". The reader mode keeps the script's shape, a
  task builder that holds only unresolved refs.
- **`docs/source-remediation-verification/run_quadsource_integrals.py`.** The Tk WKB lookup is
  keyed on the numeric row it has just looked up (`numeric_serial=Tk_numeric.store_id`, and
  `z_init=None` removed). A mode with no numeric row is recorded as unavailable
  (`SimpleNamespace(available=False)`, with `from types import SimpleNamespace` added) rather than
  looked up. That is how the driver already counts and skips a missing WKB row, so it does not
  crash (§0.6's grant).

**K1 — the inventory** (`Datastore/store_inventory.py`, §0.5, STRUCTURALLY REQUIRED as granted).
`Parent` gains `nullable: bool = False`. In `read_records`, a nullable parent whose column is NULL
gives the key field `None` and skips resolution. Nothing else in the module changed. A
non-nullable parent whose column is NULL is still an `unresolved-parent` problem
(`TestNullableParent.test_not_nullable`).

**K1 — the fingerprint format** (`RunRegistry/stores.py`, §0.2). `FINGERPRINT_FORMAT = 1` → `2`,
the one line.

**K2 — `_drop_order`** (`Datastore/SQL/Datastore.py`). `"tk-wkb"` now precedes `"tk-numeric"`.
`"gk-wkb"` already preceded `"gk-numeric"`, and `"quad-source"` already preceded `"tk-numeric"`.
The comment now says the rule: an action whose tables declare a foreign key into another action's
tables drops first. It names the WKB→numeric and `QuadSource.Tq_serial` pairs.

**D5.**
- A new witness, `Datastore/tests/data/schema_at_datastore-integrity-07.json`, captured with
  `PYTHONPATH=. ./venv/bin/python Datastore/tests/schema_description.py
  Datastore/tests/data/schema_at_datastore-integrity-07.json`.
- `test_schema_builder.WITNESS` repointed at it; `TestSchemaIsUnchanged` keeps its name.
- A new golden, `RunRegistry/tests/data/full_store_fingerprint_datastore-integrity-07.json`, with
  `test_store_fingerprint.GOLDEN` repointed at it.
- `schema_at_base.json`, `schema_at_datastore-integrity-06.json` and `full_store_fingerprint.json`
  are untouched (`git diff --cached --stat` does not list them).

**Tests.**
- `Datastore/tests/test_wkb_numeric_parent_key.py`, 41 tests: §3 tests 1–7, the nullable parent,
  and `z_init`-as-payload.
- `Datastore/tests/test_wkb_parent_foreign_key.py`, 5 tests: the new keys under `PRAGMA
  foreign_key_check`, on rows of the test's own.
- `ComputeTargets/tests/test_wkb_numeric_parent_main.py`, 14 tests: §3 test 8, the helper through
  `load_main_py_functions`, and the readers.
- Each module's `tearDownModule` asserts `ray.is_initialized()` is false.

## The shard evidence (prompt §2, §5 second stop)

`config/sharding.py:20-29`, the `sharded_tables` entries:

```
20:    "TkNumericIntegration": "k",
21:    "TkNumericValue": "k",
22:    "TkWKBIntegration": "k",
23:    "TkWKBValue": "k",
24:    "QuadSource": "q",
25:    "QuadSourceValue": "q",
26:    "GkNumericIntegration": "k",
27:    "GkNumericValue": "k",
28:    "GkWKBIntegration": "k",
29:    "GkWKBValue": "k",
```

A WKB row and its numeric parent are for the same `k`: `store()` now refuses any other
(`TestTheConsistencyCheck.test_a_parent_for_another_k_is_refused`). Both classes shard on `"k"`
through the one `shard_key_wavenumber_store_id`, so the parent is always on the WKB row's shard,
and a declared foreign key is possible. `test_wkb_parent_foreign_key.py` checks the declaration
on a real multi-shard store.

## The evidence that pure WKB is the only parentless Gk branch (§0.4, prompt §5 first stop)

`main.py` at `794728d`, the tree before this commit (`git show 794728d:main.py`):

```
2152:            z_source_limit = sqrt(k_exit.z_exit_subh_e3 * k_exit.z_exit_subh_e4)
2182:                if z_source.z >= z_source_limit and Gk.available:
```

- **The `if` branch (`:2183-2224`)** seeds from `Gk`'s stop point, so it always has a parent.
- **The `else` branch (`:2225-2268`)** is the only other construction of a `GkWKBIntegration` in
  `main.py`. It passes `G_init=0.0`, `Gprime_init=1.0` and no `z_init`: the pure-WKB branch. It is
  reached when no numeric row exists, and also when one exists but `z_source` is below
  `z_source_limit`.
- **Nothing else constructs one.** `git grep -n '"GkWKBIntegration"' -- '*.py'` at `794728d`
  finds, outside tests, only `main.py`'s three sites (the existence query and these two), the
  extraction script's lookup, `config/sharding.py`, `tools/inventory_report.py` and the
  `ClientPool` batch sizes.

So prompt §5's first stop does not trigger. The choice moved **unchanged** into
`Gk_WKB_numeric_parent`: the same two lines, now returning `Gk` or `None`.
`TestTheSeededOrPureChoice` exercises main.py's own function on both sides of the limit, at it,
and with no numeric row.

## Every `main.py` site changed (§0.6's table)

"Before" is `794728d`, and "after" is this commit.

| §0.6 site | Before | After |
|---|---|---|
| **Tk existence query.** Before: no parent, `"z_init": None`, over the batch, *then* the numeric lookup over the missing subset. After: the numeric lookup over the whole batch, the missing-numeric raise over the whole batch, then the query with `"numeric_serial": numeric_parents[k_exit.store_id].store_id` and no `z_init`. | `:1416-1452` query, `:1466-1513` numeric lookup | `:1459-1504` numeric lookup, `:1509-1526` raise, `:1531-1567` query |
| **Tk construction.** Adds `numeric_serial=Tk.store_id`, where `Tk` is `numeric_parents[k_exit.store_id]`. | `:1539-1560` | `:1591-1613` |
| **Gk existence query.** Before: vectorized, no parent, *then* the numeric lookup over the missing subset, populated. After: the numeric lookup over the whole batch with `_do_not_populate`; the choice through `Gk_WKB_numeric_parent`, as `numeric_parents` and `numeric_keys`; then the query with `"numeric_serial": numeric_keys[(k_exit.store_id, z_source.store_id)]`. | `:2025-2069` query, `:2088-2137` numeric lookup | `:2084-2134` numeric lookup, `:2138-2158` choice, `:2164-2211` query |
| **Gk seeded construction.** Adds `numeric_serial=Gk.store_id`. The branch condition becomes `if Gk is not None:`, on the helper's result. | `:2182`, `:2200-2224` | `:2247`, `:2265-2289` |
| **Gk pure construction.** Adds `numeric_serial=GkWKBIntegration.PURE_WKB`. | `:2248-2268` | `:2314-2335` |
| **QSI-stage Tk WKB lookup.** After: the unavailable-numeric raise, then the lookup with `"numeric_serial": Tk_numeric.store_id` over `zip(missing_Tk, Tk_numeric_lookup_queue.results)`, and no `z_init`. | `:3342-3377` | `:3414-3426` raise, `:3428-3463` lookup |
| **The helper** (new). | — | `:424-462` |

`git diff -U0 main.py` has hunks only inside these regions: the helper at `:424`,
`build_Tk_WKB_work`, `build_Gk_WKB_work` and `build_QuadSourceIntegral_batch`'s lookup. No
`EXPECTED_SITES` count moved.

**What the Gk stage reads off the numeric object** (Deviation 2's confirmation). In
`build_Gk_WKB_work` after this commit, the numeric object `Gk` is read for:
- `Gk._k_exit` and `Gk._z_source`, the provenance asserts;
- `Gk.available`, inside `Gk_WKB_numeric_parent`;
- `Gk.store_id`, the key and the seeded construction's parent;
- `Gk.stop_G`, `Gk.stop_Gprime` and `Gk.stop_deltaz_subh`, the seed.

That is every `Gk.` attribute read in the function, by `grep -o 'Gk\.[A-Za-z_]*'` over it. None is
a sample value. The `stop_*` properties do not depend on `_values`
(`ComputeTargets/GkNumericIntegration.py:277-295`), and the numeric factory returns them under
`_do_not_populate`. `test_the_pipeline_derivation_from_the_factory_read_numeric_row` reads a
`GkNumericIntegration` that way and seeds a stored row from it.

## The witness diff, in full (D5)

Pasted from `diff Datastore/tests/data/schema_at_datastore-integrity-06.json
Datastore/tests/data/schema_at_datastore-integrity-07.json`, in the default "normal" format. It
has **12 hunks and 86 changed lines** (`grep -c '^[0-9]'` = 12, `grep -c '^[<>]'` = 86). Counted
by the six kinds of §0.6, each hunk located by the JSON key that encloses it:

| Kind | `GkWKBIntegration` | `TkWKBIntegration` |
|---|---|---|
| 1. schema record's column list (`record.columns`) | `3671a3672,3675` | `9117a9159,9162` |
| 2. the column's description (`table.columns`) | `3906a3911,3929` | `9376a9422,9440` |
| 3. its `ForeignKeyConstraint` (`table.constraints`) | `4299a4323,4332` | `9857a9922,9931` |
| 4. the `CREATE TABLE` DDL (`table.ddl`) | `4356c4389` | `9904c9978` |
| 5. the index DDL (`table.index_ddl`) | `4358a4392` | `9907a9982` |
| 6. the index description (`table.indexes`) | `4371a4406,4412` | `9926a10002,10008` |

**2 classes × 6 kinds = 12 hunks, and no hunk outside them.** The only difference between the
classes' hunks is the nullability (`"nullable": true` for Gk, `false` for Tk), which shows in kinds
2 and 4 (`numeric_serial INTEGER` against `numeric_serial INTEGER NOT NULL`).

```
3671a3672,3675
>       "column": "numeric_serial",
>       "of_table": "GkWKBIntegration"
>      },
>      {
3906a3911,3929
>       "foreign_keys": [
>        {
>         "ondelete": null,
>         "onupdate": null,
>         "target": "GkNumericIntegration.serial"
>        }
>       ],
>       "index": true,
>       "name": "numeric_serial",
>       "nullable": true,
>       "primary_key": false,
>       "server_default": null,
>       "sqlite_type": "INTEGER",
>       "type": "Integer()",
>       "unique": null
>      },
>      {
>       "autoincrement": "'auto'",
>       "default": null,
4299a4323,4332
>        "numeric_serial"
>       ],
>       "kind": "ForeignKeyConstraint",
>       "name": null,
>       "targets": [
>        "GkNumericIntegration.serial"
>       ]
>      },
>      {
>       "columns": [
4356c4389
<     "ddl": "CREATE TABLE \"GkWKBIntegration\" (\n\tserial INTEGER NOT NULL, \n\tversion INTEGER, \n\ttimestamp DATETIME, \n\tlabel VARCHAR(256), \n\twavenumber_exit_serial INTEGER NOT NULL, \n\tmodel_serial INTEGER NOT NULL, \n\trho_gauss_order INTEGER NOT NULL, \n\tsolver_serial INTEGER NOT NULL, \n\tz_source_serial INTEGER NOT NULL, \n\tz_min_serial INTEGER NOT NULL, \n\tz_samples INTEGER NOT NULL, \n\tz_init FLOAT NOT NULL, \n\tsin_coeff FLOAT NOT NULL, \n\tcos_coeff FLOAT NOT NULL, \n\t\"G_init\" FLOAT NOT NULL, \n\t\"Gprime_init\" FLOAT NOT NULL, \n\tstage_1_compute_time FLOAT, \n\tstage_1_compute_steps INTEGER, \n\t\"stage_1_RHS_evaluations\" INTEGER, \n\t\"stage_1_mean_RHS_time\" FLOAT, \n\t\"stage_1_max_RHS_time\" FLOAT, \n\t\"stage_1_min_RHS_time\" FLOAT, \n\tstage_2_compute_time FLOAT, \n\tstage_2_compute_steps INTEGER, \n\t\"stage_2_RHS_evaluations\" INTEGER, \n\t\"stage_2_mean_RHS_time\" FLOAT, \n\t\"stage_2_max_RHS_time\" FLOAT, \n\t\"stage_2_min_RHS_time\" FLOAT, \n\t\"has_WKB_violation\" BOOLEAN, \n\t\"WKB_violation_z\" FLOAT, \n\t\"WKB_violation_efolds_subh\" FLOAT, \n\tinit_efolds_subh FLOAT, \n\tvalidated BOOLEAN NOT NULL, \n\tmetadata VARCHAR(256), \n\tPRIMARY KEY (serial), \n\tFOREIGN KEY(version) REFERENCES version (serial), \n\tFOREIGN KEY(wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(model_serial) REFERENCES \"BackgroundModel\" (serial), \n\tFOREIGN KEY(solver_serial) REFERENCES \"IntegrationSolver\" (serial), \n\tFOREIGN KEY(z_source_serial) REFERENCES redshift (serial), \n\tFOREIGN KEY(z_min_serial) REFERENCES redshift (serial)\n)",
---
>     "ddl": "CREATE TABLE \"GkWKBIntegration\" (\n\tserial INTEGER NOT NULL, \n\tversion INTEGER, \n\ttimestamp DATETIME, \n\tlabel VARCHAR(256), \n\twavenumber_exit_serial INTEGER NOT NULL, \n\tmodel_serial INTEGER NOT NULL, \n\tnumeric_serial INTEGER, \n\trho_gauss_order INTEGER NOT NULL, \n\tsolver_serial INTEGER NOT NULL, \n\tz_source_serial INTEGER NOT NULL, \n\tz_min_serial INTEGER NOT NULL, \n\tz_samples INTEGER NOT NULL, \n\tz_init FLOAT NOT NULL, \n\tsin_coeff FLOAT NOT NULL, \n\tcos_coeff FLOAT NOT NULL, \n\t\"G_init\" FLOAT NOT NULL, \n\t\"Gprime_init\" FLOAT NOT NULL, \n\tstage_1_compute_time FLOAT, \n\tstage_1_compute_steps INTEGER, \n\t\"stage_1_RHS_evaluations\" INTEGER, \n\t\"stage_1_mean_RHS_time\" FLOAT, \n\t\"stage_1_max_RHS_time\" FLOAT, \n\t\"stage_1_min_RHS_time\" FLOAT, \n\tstage_2_compute_time FLOAT, \n\tstage_2_compute_steps INTEGER, \n\t\"stage_2_RHS_evaluations\" INTEGER, \n\t\"stage_2_mean_RHS_time\" FLOAT, \n\t\"stage_2_max_RHS_time\" FLOAT, \n\t\"stage_2_min_RHS_time\" FLOAT, \n\t\"has_WKB_violation\" BOOLEAN, \n\t\"WKB_violation_z\" FLOAT, \n\t\"WKB_violation_efolds_subh\" FLOAT, \n\tinit_efolds_subh FLOAT, \n\tvalidated BOOLEAN NOT NULL, \n\tmetadata VARCHAR(256), \n\tPRIMARY KEY (serial), \n\tFOREIGN KEY(version) REFERENCES version (serial), \n\tFOREIGN KEY(wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(model_serial) REFERENCES \"BackgroundModel\" (serial), \n\tFOREIGN KEY(numeric_serial) REFERENCES \"GkNumericIntegration\" (serial), \n\tFOREIGN KEY(solver_serial) REFERENCES \"IntegrationSolver\" (serial), \n\tFOREIGN KEY(z_source_serial) REFERENCES redshift (serial), \n\tFOREIGN KEY(z_min_serial) REFERENCES redshift (serial)\n)",
4358a4392
>      "CREATE INDEX \"ix_GkWKBIntegration_numeric_serial\" ON \"GkWKBIntegration\" (numeric_serial)",
4371a4406,4412
>       "unique": false
>      },
>      {
>       "columns": [
>        "numeric_serial"
>       ],
>       "name": "ix_GkWKBIntegration_numeric_serial",
9117a9159,9162
>       "column": "numeric_serial",
>       "of_table": "TkWKBIntegration"
>      },
>      {
9376a9422,9440
>       "foreign_keys": [
>        {
>         "ondelete": null,
>         "onupdate": null,
>         "target": "TkNumericIntegration.serial"
>        }
>       ],
>       "index": true,
>       "name": "numeric_serial",
>       "nullable": false,
>       "primary_key": false,
>       "server_default": null,
>       "sqlite_type": "INTEGER",
>       "type": "Integer()",
>       "unique": null
>      },
>      {
>       "autoincrement": "'auto'",
>       "default": null,
9857a9922,9931
>        "numeric_serial"
>       ],
>       "kind": "ForeignKeyConstraint",
>       "name": null,
>       "targets": [
>        "TkNumericIntegration.serial"
>       ]
>      },
>      {
>       "columns": [
9904c9978
<     "ddl": "CREATE TABLE \"TkWKBIntegration\" (\n\tserial INTEGER NOT NULL, \n\tversion INTEGER, \n\ttimestamp DATETIME, \n\tlabel VARCHAR(256), \n\twavenumber_exit_serial INTEGER NOT NULL, \n\tmodel_serial INTEGER NOT NULL, \n\trho_gauss_order INTEGER NOT NULL, \n\tphase_solver_serial INTEGER NOT NULL, \n\tfriction_solver_serial INTEGER NOT NULL, \n\tz_min_serial INTEGER NOT NULL, \n\tz_samples INTEGER NOT NULL, \n\tz_init FLOAT NOT NULL, \n\tsin_coeff FLOAT NOT NULL, \n\tcos_coeff FLOAT NOT NULL, \n\t\"T_init\" FLOAT NOT NULL, \n\t\"Tprime_init\" FLOAT NOT NULL, \n\tstage_1_compute_time FLOAT, \n\tstage_1_compute_steps INTEGER, \n\t\"stage_1_RHS_evaluations\" INTEGER, \n\t\"stage_1_mean_RHS_time\" FLOAT, \n\t\"stage_1_max_RHS_time\" FLOAT, \n\t\"stage_1_min_RHS_time\" FLOAT, \n\tstage_2_compute_time FLOAT, \n\tstage_2_compute_steps INTEGER, \n\t\"stage_2_RHS_evaluations\" INTEGER, \n\t\"stage_2_mean_RHS_time\" FLOAT, \n\t\"stage_2_max_RHS_time\" FLOAT, \n\t\"stage_2_min_RHS_time\" FLOAT, \n\tfriction_compute_time FLOAT, \n\tfriction_compute_steps INTEGER, \n\t\"friction_RHS_evaluations\" INTEGER, \n\t\"friction_mean_RHS_time\" FLOAT, \n\t\"friction_max_RHS_time\" FLOAT, \n\t\"friction_min_RHS_time\" FLOAT, \n\t\"has_WKB_violation\" BOOLEAN, \n\t\"WKB_violation_z\" FLOAT, \n\t\"WKB_violation_efolds_subh\" FLOAT, \n\tinit_efolds_subh FLOAT, \n\tvalidated BOOLEAN NOT NULL, \n\tmetadata VARCHAR(256), \n\tPRIMARY KEY (serial), \n\tFOREIGN KEY(version) REFERENCES version (serial), \n\tFOREIGN KEY(wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(model_serial) REFERENCES \"BackgroundModel\" (serial), \n\tFOREIGN KEY(phase_solver_serial) REFERENCES \"IntegrationSolver\" (serial), \n\tFOREIGN KEY(friction_solver_serial) REFERENCES \"IntegrationSolver\" (serial), \n\tFOREIGN KEY(z_min_serial) REFERENCES redshift (serial)\n)",
---
>     "ddl": "CREATE TABLE \"TkWKBIntegration\" (\n\tserial INTEGER NOT NULL, \n\tversion INTEGER, \n\ttimestamp DATETIME, \n\tlabel VARCHAR(256), \n\twavenumber_exit_serial INTEGER NOT NULL, \n\tmodel_serial INTEGER NOT NULL, \n\tnumeric_serial INTEGER NOT NULL, \n\trho_gauss_order INTEGER NOT NULL, \n\tphase_solver_serial INTEGER NOT NULL, \n\tfriction_solver_serial INTEGER NOT NULL, \n\tz_min_serial INTEGER NOT NULL, \n\tz_samples INTEGER NOT NULL, \n\tz_init FLOAT NOT NULL, \n\tsin_coeff FLOAT NOT NULL, \n\tcos_coeff FLOAT NOT NULL, \n\t\"T_init\" FLOAT NOT NULL, \n\t\"Tprime_init\" FLOAT NOT NULL, \n\tstage_1_compute_time FLOAT, \n\tstage_1_compute_steps INTEGER, \n\t\"stage_1_RHS_evaluations\" INTEGER, \n\t\"stage_1_mean_RHS_time\" FLOAT, \n\t\"stage_1_max_RHS_time\" FLOAT, \n\t\"stage_1_min_RHS_time\" FLOAT, \n\tstage_2_compute_time FLOAT, \n\tstage_2_compute_steps INTEGER, \n\t\"stage_2_RHS_evaluations\" INTEGER, \n\t\"stage_2_mean_RHS_time\" FLOAT, \n\t\"stage_2_max_RHS_time\" FLOAT, \n\t\"stage_2_min_RHS_time\" FLOAT, \n\tfriction_compute_time FLOAT, \n\tfriction_compute_steps INTEGER, \n\t\"friction_RHS_evaluations\" INTEGER, \n\t\"friction_mean_RHS_time\" FLOAT, \n\t\"friction_max_RHS_time\" FLOAT, \n\t\"friction_min_RHS_time\" FLOAT, \n\t\"has_WKB_violation\" BOOLEAN, \n\t\"WKB_violation_z\" FLOAT, \n\t\"WKB_violation_efolds_subh\" FLOAT, \n\tinit_efolds_subh FLOAT, \n\tvalidated BOOLEAN NOT NULL, \n\tmetadata VARCHAR(256), \n\tPRIMARY KEY (serial), \n\tFOREIGN KEY(version) REFERENCES version (serial), \n\tFOREIGN KEY(wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(model_serial) REFERENCES \"BackgroundModel\" (serial), \n\tFOREIGN KEY(numeric_serial) REFERENCES \"TkNumericIntegration\" (serial), \n\tFOREIGN KEY(phase_solver_serial) REFERENCES \"IntegrationSolver\" (serial), \n\tFOREIGN KEY(friction_solver_serial) REFERENCES \"IntegrationSolver\" (serial), \n\tFOREIGN KEY(z_min_serial) REFERENCES redshift (serial)\n)",
9907a9982
>      "CREATE INDEX \"ix_TkWKBIntegration_numeric_serial\" ON \"TkWKBIntegration\" (numeric_serial)",
9926a10002,10008
>       "unique": false
>      },
>      {
>       "columns": [
>        "numeric_serial"
>       ],
>       "name": "ix_TkWKBIntegration_numeric_serial",
```

## The fingerprint re-capture (D5, §0.2)

`FINGERPRINT_FORMAT` goes from 1 to 2. The new golden was captured the way
`test_store_fingerprint._FullStores` builds its baseline: `build_full_store` into a temporary
directory, then `read_inventory(...).primary`, then `fingerprint_of(...)` with `taken` None. It was
serialised as `json.dumps(indent=2, sort_keys=True) + "\n"`, checked first to reproduce the old
golden byte for byte from its own content. The script is in this prompt's scratch directory and is
not part of the tree.

- **Moved:** `TkWKBIntegration` (count 3 → 3) and `GkWKBIntegration` (count 2 → 2).
  - Their tag-set counts are unchanged: Tk [2, 1] → [2, 1], Gk [1, 1] → [1, 1].
  - Tk digest `6a2e48d6a60aaa4f…` → `923ade1abcba9a7b…`.
  - Gk digest `d537bf41ab8d1bcd…` → `19b1567d6f076c5c…`.
  - Every tag set of both moved.
- **Why.** Each record's key lost the leaf `z_init` and gained the key parent `numeric`: for Tk
  the digest of `TkNumericIntegration` 1 or 2, and for the two pure-WKB Gk rows `None`. Nothing
  keys on a WKB row's digest, so no downstream class moved.
- **Not moved:** the other 19 classes, all identical: `BackgroundModel`, `GkNumericIntegration`,
  `GkSource`, `GkSourcePolicy`, `GkSourcePolicyData`, `IntegrationSolver`, `LambdaCDM`,
  `OneLoopIntegral`, `QCD_Cosmology`, `QuadSource`, `QuadSourceIntegral`, `QuadSourcePolicy`,
  `TkNumericIntegration`, `redshift`, `store_tag`, `tolerance`, `version`, `wavenumber` and
  `wavenumber_exit_time`.
- `problems` is `{}` before and after.
- The overall digest: `1fd51d93facee815…` → `cd04f246e59f1233…`.
- `test_store_fingerprint` passes unmodified apart from `GOLDEN`. It asserts
  `golden["fingerprint_format"] == FINGERPRINT_FORMAT`, which the old golden (format 1) can no
  longer satisfy.

## Readers (§0.6's table), and what now refuses a store it read before (README §5 rule 11)

| Reader | Before | After |
|---|---|---|
| `main.py`, all WKB sites | no parent | keyed (the table above); never the reader mode (`test_main_py_never_asks_for_the_reader_mode`) |
| `extract_TkWKB_data.py:459` | labels, beside an unresolved `TkNumeric_ref` | `labels_only=True` |
| `extract_GkWKB_data.py:469` | labels and `z_source` | `labels_only=True` |
| `run_quadsource_integrals.py:318-328` | labels, `z_init=None` | keyed on the numeric row it just looked up; unavailable if that is missing |
| `analyse_greens_and_source.py`, `docs/gktk-remedial/analyse_scoped_run.py`, `tools/inventory_report.py`, `docs/tolerance-convergence/order_audit.py` | raw SQL or table names | unchanged, not touched |

**What now refuses a store it read before:**
- **Every WKB `build()`, and so every script above that looks a WKB row up, on every store written
  before this commit** (the three under `var/` included, not opened here). This covers `main.py`'s
  Tk and Gk WKB stages and its QSI stage, both extraction scripts and the verification driver.
  They get the D4 message: the store predates the key and is readable at the SHA its sidecar
  records.
- **`store retire` of an existing store must be run at a pre-07 SHA** (§0.2). Its recorded
  fingerprint is format 1. At this tree `compare_fingerprints` answers *"fingerprints of different
  formats are not compared"* (`RunRegistry/stores.py:1258-1270`) rather than a match, and an
  inventory recomputed here would find the WKB tables `incomplete`, for want of `numeric_serial`.
  `store fingerprint` and `store amend` at this tree likewise record format 2.
- **The GkSource stage will refuse loudly on a second Gk WKB row for one `(k, z_source)` until
  prompt 09** (§0.3). That stage (`build_missing_GkSource`) reads `GkWKBValue` for every
  `z_source` with no parent. A resume under a changed numeric row now computes a second WKB row,
  where the old existence query served the stale one. 05c's refusal then raises
  `MultipleResultsFound` naming both WKB parents. That is correct until 09 keys `GkSource` on its
  Gk rows.
- **Still reads a pre-07 store, and should not:** the WKB value reads with no parent. Measured
  below, and opened as `[07-wkb-value-reads-do-not-refuse-a-pre-07-store]`.

## Deviations from the prompt

1. **The Tk stage raises for any `k` in the batch without a numeric row.** `STRUCTURALLY REQUIRED`,
   accepted by the user 2026-09-29. The existence query needs the parent, so the numeric lookup and
   its "no initial condition" raise run over the whole batch. Before, the raise covered only the
   `k`s missing a WKB row. A `k` whose WKB row is stored but whose numeric row is not now raises,
   where it was silently served. The messages are unchanged.
2. **The Gk stage's numeric lookup gains `_do_not_populate`.** `STRUCTURALLY REQUIRED`, accepted
   2026-09-29. It now runs over the whole batch (§0.4), and only the serial and the `stop_*`
   columns are read (confirmed above), so reading every sample value would be pure cost. Pinned by
   `test_the_gk_numeric_lookup_is_serial_only`.
3. **The QSI stage raises before the WKB lookup** when a mode has no numeric row. `STRUCTURALLY
   REQUIRED`, accepted 2026-09-29. Such a mode's WKB row cannot be keyed. Before, the raise came
   later, in `build_QuadSourceIntegral_payload`, after its warnings. Now there is one `MISSING DATA
   WARNING` line per mode and a `RuntimeError` naming the count.
4. **`test_store_inventory.test_a_missing_key_column_makes_the_class_incomplete` makes the column
   missing by recreating shard 1's table, not by `missing_columns`.** `STRUCTURALLY REQUIRED`.
   - §0.1 grants "It becomes `numeric_serial`". `real_store_fixtures._write_shard` removes a
     column with `ALTER TABLE … DROP COLUMN`, which SQLite refuses for a foreign-key column:
     `sqlite3.OperationalError: error in table TkWKBIntegration after drop column: unknown column
     "numeric_serial" in foreign key definition`.
   - None of the class's key columns can be dropped this way. The others are indexed or keyed
     too; `z_init` was the only droppable one, and it is no longer a key column.
   - So the test passes `extra_sql` for shard 1: `DROP TABLE "TkWKBIntegration"`, then that
     table's `CREATE TABLE` as the 06 witness records it. That is a pre-07 store's table exactly.
     The fixture module is unchanged.
   - The assertions are unchanged except the column name, which is the grant.
5. **The two 05c tests and their class are renamed.** `IMPLEMENTATION CHOICE`, within §0.1's grant
   ("each becomes the same test with two numeric parents in place of two `z_init`s").
   `TestZInitSelects` becomes `TestNumericParentSelects`, and `test_gk_wkb_value_two_z_inits` /
   `test_tk_wkb_value_two_z_inits` become `…_two_numeric_parents`. Names that said `z_init` would
   describe a filter that no longer exists. Each test asserts what it did, with a numeric parent in
   place of `z_init`, and the class still checks each value's own `z_init`.
6. **`test_value_reads_parent_key._gk_wkb` gains a `numeric_serial` parameter**, default `None`.
   `IMPLEMENTATION CHOICE`, needed to express item 5's Gk tests. §0.1 names only `_tk_wkb`. The
   default reproduces what the helper already wrote (a NULL, pure WKB), so no other caller's rows
   changed.
7. **The consistency check also compares the parent's model, and takes `z_exit` from the parent's
   exit-time row.** `IMPLEMENTATION CHOICE`. The model is part of the parent's identity as much as
   `k`. Reading `z_exit` from the store, not from `obj._k_exit`, recomputes main.py's expression
   from stored rows alone. The two are the same float: the driver's `k_exit` was read from that
   row.
8. **The labels-only mode issues a second query**, `SELECT numeric_serial` from the lookup's own
   query as a subquery. `IMPLEMENTATION CHOICE`. It names the parents without changing the keyed
   path's `.one_or_none()`. The guard sees it as a new unresolved-table site (Grant B, below).
9. **The Gk choice's long comment became the helper's docstring**, reflowed with its wording kept,
   and a one-line pointer is left at the branch. `IMPLEMENTATION CHOICE`.
10. **The Gk fixture rows write `"numeric_serial": None` explicitly**, with a comment saying why
    they are pure WKB. `IMPLEMENTATION CHOICE`. The column is nullable, so omitting it would write
    the same NULL. The explicit value records §0.1's reading of the data.

No `UNINTENDED DRIFT` was found. `git diff --cached -U0` before the commit shows only the hunks
listed here, and every `main.py` hunk is inside the regions of the site table.

## Existing-test edits, by §0.1's list (and Grants A and B)

| File | Edit | Reason |
|---|---|---|
| `Datastore/tests/real_store_fixtures.py` | the five default WKB rows gain `numeric_serial`: Tk 1, 2 → Tk numeric 1; Tk 3 → Tk numeric 2; Gk 1, 2 → `None` | NOT NULL on Tk; the Gk rows are at `z_source` 2, where no Gk numeric row exists, so they are pure WKB (§0.1). Their other values are not "corrected" |
| `Datastore/tests/test_store_inventory.py` | `IDENTITY`'s two WKB entries: `z_init` out, `"numeric"` in (`TkWKBIntegration` 2 → `numeric_serial` 3; `GkWKBIntegration` 1 → `numeric_serial` 1) | the key's fields now are `rho_gauss_order`, `model`, `k`, (`z_source`,) `numeric` |
| same | `NON_IDENTITY` gains `("TkWKBIntegration", 1, "z_init", 300.0)` beside `T_init` | `z_init` is payload now |
| same | `test_a_missing_key_column_makes_the_class_incomplete`: `numeric_serial`, by recreating the table (Deviation 4) | the missing key column follows the key |
| `Datastore/tests/test_read_path_mechanics.py` | `_tk_wkb_parent` writes `numeric_serial=1` | the new NOT NULL column; 1 is the `_tk_numeric_parent` serial where a test writes one |
| `Datastore/tests/test_value_reads_parent_key.py` | `_tk_wkb` gains `numeric_serial` (default 1); `_gk_wkb` gains it (default `None`, Deviation 6) | the new column |
| same | `TestSecondParentIsRefused.test_{gk,tk}_wkb_value_two_z_inits` → `…_two_numeric_parents`; `TestZInitSelects` → `TestNumericParentSelects` (Deviation 5) | 05c's hand-off: they follow 07's key |
| `ComputeTargets/tests/test_tk_wkb_phase.py` | `build_and_store` constructs with `numeric_serial=1` | a Tk construction without a parent is an error; nothing about the phase or the wall time changes |
| `ComputeTargets/tests/test_gauss_order_key.py` (Grant A) | `_NUMERIC_PARENT = 18`; `numeric_serial` in `TARGETS`' two WKB payloads and `_rehydrate`'s WKB payload; `numeric_serial` on `_WKBRowConnection`'s row and on `_WKBStandIn`; `_ParentRowConnection` and `_parent_tables` passed to the WKB `store()`s by `_store_payload` and by the direct `store(` call | a WKB lookup names its parent, and `store()`'s check reads a parent row, here one consistent with the stand-in (k 13, model 14, z_source 17, `z_exit - stop_deltaz_subh = 2.0 - 1.0 = z_init`, stop values equal to the stand-in's). No assertion changed |
| `Datastore/tests/test_factory_select_columns.py` (Grant B) | `EXPECTED_COVERAGE` **44 → 47**: `TkWKBIntegration.py:sqla_TkWKBIntegration_factory._check_numeric_parent.parent`, `GkWKBIntegration.py:sqla_GkWKBIntegration_factory._check_numeric_parent.parent`, `GkWKBIntegration.py:sqla_GkWKBIntegration_factory._check_numeric_parent.z_source_row`. `KNOWN_UNRESOLVED_TABLES` **9 → 11**: `(TkWKBIntegration.py:sqla_TkWKBIntegration_factory.build, "query.subquery()")` and the Gk twin, each with its reason | exactly the new row reads and the labels-only subquery; nothing removed, and no other pin moved |
| `Datastore/tests/test_schema_builder.py` | `WITNESS` → `schema_at_datastore-integrity-07.json` | D5 |
| `RunRegistry/tests/test_store_fingerprint.py` | `GOLDEN` → `full_store_fingerprint_datastore-integrity-07.json` | D5 |

The guard's other pins (`KNOWN_BLIND` 1, `KNOWN_UNCHECKED_INSERTS` 13, `KNOWN_BLIND_COLUMNS` 4,
empty `KNOWN_UNFIXED` / `KNOWN_UNFIXED_COLUMNS`) did not move. Before the two pins were extended,
`test_factory_select_columns` failed exactly those two assertions, naming these five entries and
no others.

## Verification performed

- **The starting tree.** `git status` clean at `794728d`, and `python -m RunRegistry list` showed
  nothing `running`. Another session committed `ccb3078` ("Write up the one-loop SIGW derivation as
  a LaTeX document", `docs/derivation/` only) on this branch while this prompt was in progress.
  This commit sits on it and touches none of its files.
- **`substitute_main_py()`** after the edits, called from `docs/handover/quadsource_atol_sweep.py`
  (loaded with `importlib`, not edited). The five `** quadsource_atol_sweep:` lines are
  `substitute_main_py()`'s own, and it raises unless each literal occurs exactly once. The last
  five lines are the orchestrator's counter script (`orch07/subst_count.py`), run on the same tree,
  which gives each literal's line after this commit:

  ```
  ** quadsource_atol_sweep: replaced 1 occurrence of 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_SOURCE_K_VALUES)' in main.py
  ** quadsource_atol_sweep: replaced 1 occurrence of 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_RESPONSE_K_VALUES)' in main.py
  ** quadsource_atol_sweep: replaced 1 occurrence of 'tol=DEFAULT_QUADRATURE_ATOL' in main.py
  ** quadsource_atol_sweep: replaced 1 occurrence of 'tol=DEFAULT_QUADRATURE_RTOL' in main.py
  ** quadsource_atol_sweep: replaced 1 occurrence of '        qsi_work_batches = list(grouper(qsi_work_items, n=750, incomplete="fill"))' in main.py
  substituted source: 161854 characters
  count=1 line=3791 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_SOURCE_K_VALUES)'
  count=1 line=3803 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_RESPONSE_K_VALUES)'
  count=1 line=3689 'tol=DEFAULT_QUADRATURE_ATOL'
  count=1 line=3690 'tol=DEFAULT_QUADRATURE_RTOL'
  count=1 line=3582 '        qsi_work_batches = list(grouper(qsi_work_items, n=750, incomplete="fill"))'
  ```

- **The six suites**, README §7's command, serially, on the final tree:

  | Suite | Result |
  |---|---|
  | AdaptiveLevin | 32 OK |
  | ComputeTargets | **566 OK** (552 + the 14 of `test_wkb_numeric_parent_main.py`; the known wall-clock flake did not fire) |
  | CosmologyModels | 39 OK |
  | Datastore | **397 OK** (351 + 41 + 5) |
  | LiouvilleGreen | 148 OK (1 skipped) |
  | RunRegistry | 199 OK |

- **The new modules, `-v`, unfiltered:**

  ```
  test_a_gk_construction_without_a_parent_is_pure_wkb_and_says_so (Datastore.tests.test_wkb_numeric_parent_key.TestAParentIsRequired.test_a_gk_construction_without_a_parent_is_pure_wkb_and_says_so) ... ok
  test_a_gk_lookup_given_neither_is_an_error (Datastore.tests.test_wkb_numeric_parent_key.TestAParentIsRequired.test_a_gk_lookup_given_neither_is_an_error) ... ok
  test_a_parent_and_labels_only_together_are_an_error (Datastore.tests.test_wkb_numeric_parent_key.TestAParentIsRequired.test_a_parent_and_labels_only_together_are_an_error) ... ok
  test_a_tk_construction_without_a_parent_is_an_error (Datastore.tests.test_wkb_numeric_parent_key.TestAParentIsRequired.test_a_tk_construction_without_a_parent_is_an_error) ... ok
  test_a_tk_lookup_has_no_pure_wkb_form (Datastore.tests.test_wkb_numeric_parent_key.TestAParentIsRequired.test_a_tk_lookup_has_no_pure_wkb_form) ... ok
  test_a_tk_lookup_without_a_parent_is_an_error (Datastore.tests.test_wkb_numeric_parent_key.TestAParentIsRequired.test_a_tk_lookup_without_a_parent_is_an_error) ... ok
  test_gk (Datastore.tests.test_wkb_numeric_parent_key.TestAnOldStoreIsRefused.test_gk) ... ok
  test_tk (Datastore.tests.test_wkb_numeric_parent_key.TestAnOldStoreIsRefused.test_tk) ... ok
  test_every_declared_foreign_key_between_actions_is_respected (Datastore.tests.test_wkb_numeric_parent_key.TestDropOrder.test_every_declared_foreign_key_between_actions_is_respected)
  Derived from the schema: an action whose tables declare a foreign key into another ... ok
  test_the_corrected_order_drops_both (Datastore.tests.test_wkb_numeric_parent_key.TestDropOrder.test_the_corrected_order_drops_both) ... ok
  test_the_old_order_is_refused_by_enforcement (Datastore.tests.test_wkb_numeric_parent_key.TestDropOrder.test_the_old_order_is_refused_by_enforcement) ... ok
  test_tk_wkb_drops_before_tk_numeric_and_gk_wkb_before_gk_numeric (Datastore.tests.test_wkb_numeric_parent_key.TestDropOrder.test_tk_wkb_drops_before_tk_numeric_and_gk_wkb_before_gk_numeric) ... ok
  test_gk_the_first_parent_finds_it_and_the_second_misses (Datastore.tests.test_wkb_numeric_parent_key.TestKeyedOnTheParent.test_gk_the_first_parent_finds_it_and_the_second_misses) ... ok
  test_the_stored_row_records_its_parent (Datastore.tests.test_wkb_numeric_parent_key.TestKeyedOnTheParent.test_the_stored_row_records_its_parent) ... ok
  test_tk_the_first_parent_finds_it_and_the_second_misses (Datastore.tests.test_wkb_numeric_parent_key.TestKeyedOnTheParent.test_tk_the_first_parent_finds_it_and_the_second_misses) ... ok
  test_tk_the_numeric_parents_differ_only_in_their_tolerance (Datastore.tests.test_wkb_numeric_parent_key.TestKeyedOnTheParent.test_tk_the_numeric_parents_differ_only_in_their_tolerance)
  What makes the test above the case the prompt names: one k, two numeric rows. ... ok
  test_not_nullable (Datastore.tests.test_wkb_numeric_parent_key.TestNullableParent.test_not_nullable) ... ok
  test_nullable (Datastore.tests.test_wkb_numeric_parent_key.TestNullableParent.test_nullable) ... ok
  test_both_rows_coexist_at_one_k_and_z_source (Datastore.tests.test_wkb_numeric_parent_key.TestPureWKBIsItsOwnKey.test_both_rows_coexist_at_one_k_and_z_source) ... ok
  test_each_lookup_finds_its_own (Datastore.tests.test_wkb_numeric_parent_key.TestPureWKBIsItsOwnKey.test_each_lookup_finds_its_own) ... ok
  test_the_value_reads_select_by_parent_or_marker (Datastore.tests.test_wkb_numeric_parent_key.TestPureWKBIsItsOwnKey.test_the_value_reads_select_by_parent_or_marker) ... ok
  test_without_a_parent_the_value_read_refuses_the_second (Datastore.tests.test_wkb_numeric_parent_key.TestPureWKBIsItsOwnKey.test_without_a_parent_the_value_read_refuses_the_second)
  05c's refusal stays: a value read naming no parent spans every parent, and two parents ... ok
  test_a_miss_reports_absence_and_cannot_be_computed (Datastore.tests.test_wkb_numeric_parent_key.TestReadersRefuseAmbiguity.test_a_miss_reports_absence_and_cannot_be_computed) ... ok
  test_gk_a_seeded_and_a_pure_row_are_refused_and_named (Datastore.tests.test_wkb_numeric_parent_key.TestReadersRefuseAmbiguity.test_gk_a_seeded_and_a_pure_row_are_refused_and_named) ... ok
  test_gk_two_seeded_rows_are_refused_and_named (Datastore.tests.test_wkb_numeric_parent_key.TestReadersRefuseAmbiguity.test_gk_two_seeded_rows_are_refused_and_named) ... ok
  test_one_row_is_served (Datastore.tests.test_wkb_numeric_parent_key.TestReadersRefuseAmbiguity.test_one_row_is_served) ... ok
  test_tk_two_parents_are_refused_and_named (Datastore.tests.test_wkb_numeric_parent_key.TestReadersRefuseAmbiguity.test_tk_two_parents_are_refused_and_named) ... ok
  test_a_parent_for_another_k_is_refused (Datastore.tests.test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_a_parent_for_another_k_is_refused) ... ok
  test_a_parent_for_another_z_source_is_refused (Datastore.tests.test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_a_parent_for_another_z_source_is_refused) ... ok
  test_a_parent_not_on_this_shard_is_refused (Datastore.tests.test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_a_parent_not_on_this_shard_is_refused) ... ok
  test_gk_one_ulp_off_is_refused (Datastore.tests.test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_gk_one_ulp_off_is_refused) ... ok
  test_nothing_is_written_when_refused (Datastore.tests.test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_nothing_is_written_when_refused) ... ok
  test_pure_wkb_one_ulp_off_is_refused (Datastore.tests.test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_pure_wkb_one_ulp_off_is_refused) ... ok
  test_rows_seeded_as_main_py_seeds_them_store (Datastore.tests.test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_rows_seeded_as_main_py_seeds_them_store) ... ok
  test_the_other_parents_stop_values_are_refused (Datastore.tests.test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_the_other_parents_stop_values_are_refused)
  A row carrying parent 2's initial data but naming parent 1. ... ok
  test_the_pipeline_derivation_from_the_factory_read_numeric_row (Datastore.tests.test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_the_pipeline_derivation_from_the_factory_read_numeric_row)
  The numeric row read back serial-only through its own factory, as main.py's Tk and Gk ... ok
  test_the_seed_rounds (Datastore.tests.test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_the_seed_rounds)
  z_init is a subtraction that rounds at this size, so a check that did not recompute it ... ok
  test_tk_one_ulp_off_is_refused (Datastore.tests.test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_tk_one_ulp_off_is_refused) ... ok
  test_gk (Datastore.tests.test_wkb_numeric_parent_key.TestZInitIsPayload.test_gk) ... ok
  test_the_inventory_key_has_no_z_init (Datastore.tests.test_wkb_numeric_parent_key.TestZInitIsPayload.test_the_inventory_key_has_no_z_init) ... ok
  test_tk (Datastore.tests.test_wkb_numeric_parent_key.TestZInitIsPayload.test_tk) ... ok
  test_a_dangling_gk_parent_is_reported (Datastore.tests.test_wkb_parent_foreign_key.TestTheNewKeys.test_a_dangling_gk_parent_is_reported) ... ok
  test_a_dangling_tk_parent_is_reported (Datastore.tests.test_wkb_parent_foreign_key.TestTheNewKeys.test_a_dangling_tk_parent_is_reported)
  Tk numeric 2 is on shard 1, not shard 0: the check reads the declared key. ... ok
  test_every_shard_passes_foreign_key_check (Datastore.tests.test_wkb_parent_foreign_key.TestTheNewKeys.test_every_shard_passes_foreign_key_check) ... ok
  test_the_added_rows_are_not_satisfied_by_coincidence (Datastore.tests.test_wkb_parent_foreign_key.TestTheNewKeys.test_the_added_rows_are_not_satisfied_by_coincidence) ... ok
  test_the_default_stores_pass_too (Datastore.tests.test_wkb_parent_foreign_key.TestTheNewKeys.test_the_default_stores_pass_too) ... ok
  test_every_construction_passes_the_parent (ComputeTargets.tests.test_wkb_numeric_parent_main.TestMainPyKeysEveryWKBSite.test_every_construction_passes_the_parent) ... ok
  test_every_query_passes_the_parent_and_no_z_init (ComputeTargets.tests.test_wkb_numeric_parent_main.TestMainPyKeysEveryWKBSite.test_every_query_passes_the_parent_and_no_z_init) ... ok
  test_main_py_never_asks_for_the_reader_mode (ComputeTargets.tests.test_wkb_numeric_parent_main.TestMainPyKeysEveryWKBSite.test_main_py_never_asks_for_the_reader_mode) ... ok
  test_the_gk_numeric_lookup_is_serial_only (ComputeTargets.tests.test_wkb_numeric_parent_main.TestMainPyKeysEveryWKBSite.test_the_gk_numeric_lookup_is_serial_only)
  Only the serial and the stop_* columns of the numeric row are read in the Gk stage, so ... ok
  test_the_gk_stage_decides_with_the_module_level_choice (ComputeTargets.tests.test_wkb_numeric_parent_main.TestMainPyKeysEveryWKBSite.test_the_gk_stage_decides_with_the_module_level_choice) ... ok
  test_the_numeric_row_is_looked_up_before_the_wkb_query (ComputeTargets.tests.test_wkb_numeric_parent_main.TestMainPyKeysEveryWKBSite.test_the_numeric_row_is_looked_up_before_the_wkb_query) ... ok
  test_the_seeded_constructions_name_the_numeric_row_and_the_pure_one_the_marker (ComputeTargets.tests.test_wkb_numeric_parent_main.TestMainPyKeysEveryWKBSite.test_the_seeded_constructions_name_the_numeric_row_and_the_pure_one_the_marker) ... ok
  test_the_sites_are_where_they_were (ComputeTargets.tests.test_wkb_numeric_parent_main.TestMainPyKeysEveryWKBSite.test_the_sites_are_where_they_were)
  Three per class, as test_main_plumbing.EXPECTED_SITES pins them: a finder that found ... ok
  test_the_extraction_scripts_ask_for_labels_only (ComputeTargets.tests.test_wkb_numeric_parent_main.TestTheReaders.test_the_extraction_scripts_ask_for_labels_only) ... ok
  test_the_verification_driver_keys_on_its_numeric_row (ComputeTargets.tests.test_wkb_numeric_parent_main.TestTheReaders.test_the_verification_driver_keys_on_its_numeric_row) ... ok
  test_above_the_limit_with_a_numeric_row_is_seeded (ComputeTargets.tests.test_wkb_numeric_parent_main.TestTheSeededOrPureChoice.test_above_the_limit_with_a_numeric_row_is_seeded) ... ok
  test_at_the_limit_is_seeded (ComputeTargets.tests.test_wkb_numeric_parent_main.TestTheSeededOrPureChoice.test_at_the_limit_is_seeded) ... ok
  test_below_the_limit_is_pure_even_with_a_numeric_row (ComputeTargets.tests.test_wkb_numeric_parent_main.TestTheSeededOrPureChoice.test_below_the_limit_is_pure_even_with_a_numeric_row) ... ok
  test_without_a_numeric_row_is_pure (ComputeTargets.tests.test_wkb_numeric_parent_main.TestTheSeededOrPureChoice.test_without_a_numeric_row_is_pure) ... ok
  
  ----------------------------------------------------------------------
  Ran 60 tests in 1.912s
  
  OK
  !! GkWKBIntegration.build(): multiple results found when querying for GkWKBIntegration
  !! GkWKBIntegration.build(): multiple results found when querying for GkWKBIntegration
  !! TkWKBIntegration.build(): multiple results found when querying for TkWKBIntegration
  !! Warning (TkNumericIntegration) k=1e+05/Mpc, log10_atol=-10.0, log10_rtol=-9.0
  |    Initial redshift z_init=5e+09 is later than the 3-efold superhorizon time z_e3=1.2346e+12.
  |    Setting initial conditions at this time may lead to meaningless results, because the initial values T_k(z) = 1, Tprime_k(z) = 0.
  |    used for the matter transfer function integration apply only on sufficiently superhorizon scales.
  ```

- **The prompt's §5 fifth stop.** Rows seeded exactly as `main.py` seeds them store cleanly
  (`test_rows_seeded_as_main_py_seeds_them_store`,
  `test_the_pipeline_derivation_from_the_factory_read_numeric_row`), and the check does not trigger
  on them. The pipeline-derivation test reads the numeric row back through its own factory with
  `_do_not_populate`, as main.py does, and applies main.py's expression. Its store holds
  production-sized redshifts, `z_exit = 1.2345678901234567e11` and `stop_deltaz_subh =
  2.3456789012345678e10`, where the subtraction rounds and one ulp of `z_init` exceeds `1e-7`
  (`test_the_seed_rounds`). So bit equality is doing work there.
- **The foreign keys (06's S4, §0.6).** `test_wkb_parent_foreign_key.py` runs `PRAGMA
  foreign_key_check` over every shard of a `build_full_store` store on the test's own rows. These
  are a Tk WKB row whose `numeric_serial` (3) is a Tk numeric serial on shard 0 and no Tk WKB
  serial there, and a seeded Gk WKB row with a Gk numeric (5) at its own `z_source`, 5 being no Gk
  WKB serial there. The docstring says so, and `test_the_added_rows_are_not_satisfied_by_coincidence`
  asserts it. Both shards report nothing; a parent on the other shard is reported, for each class.
  The in-memory stores of `test_wkb_numeric_parent_key.py` are clean under `foreign_key_check`
  after storing seeded and pure rows, `numeric_serial` included.
- **Formatting.** `./venv/bin/python -m black --check` on the 23 Python files this commit touches:
  "23 files would be left unchanged".
- **Scope.** `git diff --cached --name-only` before the commit is listed in the final report and
  checked against the brief's list. Staging was by explicit path only; `docs/derivation/` was never
  staged.
- **No Ray, nothing under `var/`.** No test module starts Ray, and each new one asserts it is not
  initialised at teardown. No store under `var/` was opened. Every store here is in memory or
  under `tempfile`, and the measurements below used this prompt's scratch directory.

## The deliberate-breakage record

Each mutation was applied with `git apply` to the staged tree of this commit, so that `git diff`
showed the mutation alone. Then the `Datastore`, `ComputeTargets` and `RunRegistry` suites were run
serially, and the mutation reverted with `git apply -R`, checking afterwards that `git diff` was
empty. AdaptiveLevin, CosmologyModels and LiouvilleGreen were not run under the mutations: none
imports a mutated module (`grep -rlE 'WKBIntegration|Datastore\.SQL|store_inventory|main\.py'`
finds only comments there).

The diffs are pasted from `git diff`, exactly as applied. None ends in blank context, so none
needed `-U1`. Each failing entry below is one `FAIL:` or `ERROR:` line of the unittest output; a
failing sub-test is its own entry, and the sub-test's parameters are given. The counts are
unittest's own `failures=` and `errors=`.

**Replays.** Before the commit, each of the five diff blocks below was extracted from this file
and applied with plain `git apply`, the same three suites were run, and it was reverted. **Every failing set and every count was identical to the one listed under its mutation**, compared line by line: Datastore `failures=12`, `8`, `14`, `2`, and `failures=2, errors=1`; ComputeTargets `OK` except (iv), `failures=2`; RunRegistry `OK` in all five. `git diff` was empty after each revert.
After the commit they are replayed again from
`git show HEAD:prompts/datastore-integrity/logs/07-key-wkb-rows-on-their-numeric-row.md`. That
result goes to the orchestrator and is not written here, since this file cannot record a run that
follows its own commit.

### Mutation (i) — `numeric_serial` dropped from both `build()`s' filter

```diff
diff --git a/Datastore/SQL/ObjectFactories/GkWKBIntegration.py b/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
index 9ccdb5e..bf917d6 100644
--- a/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
+++ b/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
@@ -355,10 +355,6 @@ class sqla_GkWKBIntegration_factory(SQLAFactoryBase):
         # the numeric parent, or NULL for pure WKB (prompt 07). z_init is not filtered on: it is a
         # function of the parent, and the absolute DEFAULT_FLOAT_PRECISION match it used to have
         # was bit equality at z ~ 1e9-1e12 (docs/datastore-integrity-audit.md §3.13)
-        if not labels_only:
-            query = query.filter(
-                _numeric_parent_filter(table.c.numeric_serial, numeric_serial)
-            )
 
         # require that the integration we search for has the specified list of tags
         count = 0
diff --git a/Datastore/SQL/ObjectFactories/TkWKBIntegration.py b/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
index db581bb..47f2aee 100644
--- a/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
+++ b/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
@@ -349,10 +349,6 @@ class sqla_TkWKBIntegration_factory(SQLAFactoryBase):
         # the numeric parent (prompt 07). z_init is not filtered on: it is a function of the
         # parent, and the absolute DEFAULT_FLOAT_PRECISION match it used to have was bit equality
         # at z ~ 1e9-1e12 (docs/datastore-integrity-audit.md §3.13)
-        if not labels_only:
-            query = query.filter(
-                _numeric_parent_filter(table.c.numeric_serial, numeric_serial)
-            )
 
         # require that the integration we search for has the specified list of tags
         count = 0
```

`Datastore`: `Ran 397 tests in 125.322s`, `FAILED (failures=12)`; 12 failing entries, none a sub-test:

- `FAIL` `test_wkb_numeric_parent_key.TestAParentIsRequired.test_a_tk_lookup_has_no_pure_wkb_form` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestKeyedOnTheParent.test_gk_the_first_parent_finds_it_and_the_second_misses` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestKeyedOnTheParent.test_the_stored_row_records_its_parent` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestKeyedOnTheParent.test_tk_the_first_parent_finds_it_and_the_second_misses` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestPureWKBIsItsOwnKey.test_both_rows_coexist_at_one_k_and_z_source` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestPureWKBIsItsOwnKey.test_each_lookup_finds_its_own` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestPureWKBIsItsOwnKey.test_the_value_reads_select_by_parent_or_marker` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestPureWKBIsItsOwnKey.test_without_a_parent_the_value_read_refuses_the_second` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestReadersRefuseAmbiguity.test_gk_a_seeded_and_a_pure_row_are_refused_and_named` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestReadersRefuseAmbiguity.test_gk_two_seeded_rows_are_refused_and_named` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestReadersRefuseAmbiguity.test_tk_two_parents_are_refused_and_named` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_rows_seeded_as_main_py_seeds_them_store` — the test itself (no sub-tests)

`ComputeTargets`: `Ran 566 tests in 137.060s`, `OK`.

`RunRegistry`: `Ran 199 tests in 67.844s`, `OK`.

### Mutation (ii) — the pure-WKB marker matches any parent instead of `IS NULL` (§0.3's reading)

```diff
diff --git a/Datastore/SQL/ObjectFactories/GkWKBIntegration.py b/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
index 9ccdb5e..b104008 100644
--- a/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
+++ b/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
@@ -85,7 +85,7 @@ def _numeric_parent_filter(column, numeric_serial):
     """The key's parent term: a GkNumericIntegration serial, or the explicit pure-WKB marker,
     which matches the rows that have no numeric parent (NULL) and nothing else."""
     if numeric_serial == GkWKBIntegration.PURE_WKB:
-        return column.is_(None)
+        return sqla.true()
     if isinstance(numeric_serial, bool) or not isinstance(numeric_serial, int):
         raise TypeError(
             f"GkWKBIntegration: a numeric parent is a GkNumericIntegration serial or "
```

`Datastore`: `Ran 397 tests in 111.018s`, `FAILED (failures=8)`; 8 failing entries, none a sub-test:

- `FAIL` `test_wkb_numeric_parent_key.TestKeyedOnTheParent.test_gk_the_first_parent_finds_it_and_the_second_misses` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestKeyedOnTheParent.test_the_stored_row_records_its_parent` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestPureWKBIsItsOwnKey.test_both_rows_coexist_at_one_k_and_z_source` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestPureWKBIsItsOwnKey.test_each_lookup_finds_its_own` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestPureWKBIsItsOwnKey.test_the_value_reads_select_by_parent_or_marker` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestPureWKBIsItsOwnKey.test_without_a_parent_the_value_read_refuses_the_second` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestReadersRefuseAmbiguity.test_gk_a_seeded_and_a_pure_row_are_refused_and_named` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_rows_seeded_as_main_py_seeds_them_store` — the test itself (no sub-tests)

`ComputeTargets`: `Ran 566 tests in 136.813s`, `OK`.

`RunRegistry`: `Ran 199 tests in 60.332s`, `OK`.

### Mutation (iii) — the consistency check removed from both `store()`s

```diff
diff --git a/Datastore/SQL/ObjectFactories/GkWKBIntegration.py b/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
index 9ccdb5e..7eeb578 100644
--- a/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
+++ b/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
@@ -618,7 +618,6 @@ class sqla_GkWKBIntegration_factory(SQLAFactoryBase):
     ):
         # no row is written whose initial data are not what its parent, or the pure-WKB branch,
         # gives
-        sqla_GkWKBIntegration_factory._check_numeric_parent(obj, conn, tables)
 
         store_id = inserter(
             conn,
diff --git a/Datastore/SQL/ObjectFactories/TkWKBIntegration.py b/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
index db581bb..a3c09f3 100644
--- a/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
+++ b/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
@@ -613,7 +613,6 @@ class sqla_TkWKBIntegration_factory(SQLAFactoryBase):
         inserters,
     ):
         # no row is written whose initial data are not its numeric parent's stop values
-        sqla_TkWKBIntegration_factory._check_numeric_parent(obj, conn, tables)
 
         store_id = inserter(
             conn,
```

`Datastore`: `Ran 397 tests in 109.554s`, `FAILED (failures=14)`; 14 failing entries, 9 of them sub-tests:

- `FAIL` `test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_a_parent_for_another_k_is_refused` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_a_parent_for_another_z_source_is_refused` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_a_parent_not_on_this_shard_is_refused` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_gk_one_ulp_off_is_refused` — sub-test `name='z_init'`
- `FAIL` `test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_gk_one_ulp_off_is_refused` — sub-test `name='G_init'`
- `FAIL` `test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_gk_one_ulp_off_is_refused` — sub-test `name='Gprime_init'`
- `FAIL` `test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_nothing_is_written_when_refused` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_pure_wkb_one_ulp_off_is_refused` — sub-test `name='G_init'`
- `FAIL` `test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_pure_wkb_one_ulp_off_is_refused` — sub-test `name='Gprime_init'`
- `FAIL` `test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_pure_wkb_one_ulp_off_is_refused` — sub-test `name='z_init'`
- `FAIL` `test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_the_other_parents_stop_values_are_refused` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_tk_one_ulp_off_is_refused` — sub-test `name='z_init'`
- `FAIL` `test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_tk_one_ulp_off_is_refused` — sub-test `name='T_init'`
- `FAIL` `test_wkb_numeric_parent_key.TestTheConsistencyCheck.test_tk_one_ulp_off_is_refused` — sub-test `name='Tprime_init'`

`ComputeTargets`: `Ran 566 tests in 138.111s`, `OK`.

`RunRegistry`: `Ran 199 tests in 64.716s`, `OK`.

### Mutation (iv) — `z_init` back in both keys with the absolute `1e-7`, and the parent removed from `main.py`'s Tk and Gk existence queries

```diff
diff --git a/Datastore/SQL/ObjectFactories/GkWKBIntegration.py b/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
index 9ccdb5e..a58124e 100644
--- a/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
+++ b/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
@@ -359,6 +359,10 @@ class sqla_GkWKBIntegration_factory(SQLAFactoryBase):
             query = query.filter(
                 _numeric_parent_filter(table.c.numeric_serial, numeric_serial)
             )
+        if z_init is not None:
+            query = query.filter(
+                sqla.func.abs(table.c.z_init - z_init) < DEFAULT_FLOAT_PRECISION
+            )
 
         # require that the integration we search for has the specified list of tags
         count = 0
diff --git a/Datastore/SQL/ObjectFactories/TkWKBIntegration.py b/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
index db581bb..89bb658 100644
--- a/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
+++ b/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
@@ -353,6 +353,10 @@ class sqla_TkWKBIntegration_factory(SQLAFactoryBase):
             query = query.filter(
                 _numeric_parent_filter(table.c.numeric_serial, numeric_serial)
             )
+        if z_init is not None:
+            query = query.filter(
+                sqla.func.abs(table.c.z_init - z_init) < DEFAULT_FLOAT_PRECISION
+            )
 
         # require that the integration we search for has the specified list of tags
         count = 0
diff --git a/main.py b/main.py
index 3ad98bc..c23bacd 100644
--- a/main.py
+++ b/main.py
@@ -1533,7 +1533,6 @@ def run_pipeline(
                 "solver_labels": [],
                 "model": model_proxy,
                 "k": k_exit,
-                "numeric_serial": numeric_parents[k_exit.store_id].store_id,
                 "z_sample": None,
                 "tags": [
                     TkProductionTag,
@@ -2169,9 +2168,6 @@ def run_pipeline(
                         "solver_labels": [],
                         "model": model_proxy,
                         "z_source": z_source,
-                        "numeric_serial": numeric_keys[
-                            (k_exit.store_id, z_source.store_id)
-                        ],
                         "z_sample": None,
                         "tags": [
                             GkProductionTag,
```

`Datastore`: `Ran 397 tests in 109.430s`, `FAILED (failures=2)`; 2 failing entries, none a sub-test:

- `FAIL` `test_wkb_numeric_parent_key.TestZInitIsPayload.test_gk` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestZInitIsPayload.test_tk` — the test itself (no sub-tests)

`ComputeTargets`: `Ran 566 tests in 135.712s`, `FAILED (failures=2)`; 2 failing entries, 2 of them sub-tests:

- `FAIL` `test_wkb_numeric_parent_main.TestMainPyKeysEveryWKBSite.test_every_query_passes_the_parent_and_no_z_init` — sub-test `function='build_Tk_WKB_work', line=1554`
- `FAIL` `test_wkb_numeric_parent_main.TestMainPyKeysEveryWKBSite.test_every_query_passes_the_parent_and_no_z_init` — sub-test `function='build_Gk_WKB_work', line=2193`

`RunRegistry`: `Ran 199 tests in 59.000s`, `OK`.

### Mutation (v) — `_drop_order` reverted

```diff
diff --git a/Datastore/SQL/Datastore.py b/Datastore/SQL/Datastore.py
index 9c889af..dda2dcb 100644
--- a/Datastore/SQL/Datastore.py
+++ b/Datastore/SQL/Datastore.py
@@ -160,8 +160,8 @@ _drop_order = [
     "gk-wkb",
     "gk-numeric",
     "quad-source",
-    "tk-wkb",
     "tk-numeric",
+    "tk-wkb",
 ]
 
 # read table configuration should be a Dict with the mapping
```

`Datastore`: `Ran 397 tests in 107.272s`, `FAILED (failures=2, errors=1)`; 3 failing entries, 1 of them sub-tests:

- `ERROR` `test_wkb_numeric_parent_key.TestDropOrder.test_the_corrected_order_drops_both` — the test itself (no sub-tests)
- `FAIL` `test_wkb_numeric_parent_key.TestDropOrder.test_every_declared_foreign_key_between_actions_is_respected` — sub-test `table='TkWKBIntegration', references='TkNumericIntegration'`
- `FAIL` `test_wkb_numeric_parent_key.TestDropOrder.test_tk_wkb_drops_before_tk_numeric_and_gk_wkb_before_gk_numeric` — the test itself (no sub-tests)

`ComputeTargets`: `Ran 566 tests in 141.269s`, `OK`.

`RunRegistry`: `Ran 199 tests in 60.110s`, `OK`.

## Observations not acted on

1. **`_drop_actions` lists each parent table before its children** → opened
   `[07-drop-actions-list-a-parent-table-before-its-children]` (board §3.3).
   - Measured on this prompt's test store, clean under `foreign_key_check`, with one Tk WKB row and
     its two `TkWKBValue` rows, under `PRAGMA foreign_keys = ON`: the real
     `_drop_actions(["tk-wkb"])`, the action alone, raised `IntegrityError: FOREIGN KEY constraint
     failed`.
   - `_drop_order` is right *between* actions, and that is all K2 licenses. `TestDropOrder`
     therefore uses a store with no value rows.
   - SQLite counts violations net: a dropped row whose own references already dangle offsets the
     one it creates. That is why `TestDropOrder` first asserts its store is clean under
     `foreign_key_check`. An earlier version of the test, on an unclean store, passed under the old
     order.
2. **The WKB value reads do not refuse a pre-07 store** → opened
   `[07-wkb-value-reads-do-not-refuse-a-pre-07-store]` (board §3.3). Measured on a
   `GkWKBIntegration` table recreated from the 06 witness's DDL, holding one pure-WKB row:
   - a parentless `GkWKBValue.read_batch` returned its 2 values;
   - a keyed one raised `OperationalError: no such column: GkWKBIntegration.numeric_serial`.

   `build()` alone gives the D4 message. The precedent, `rho_gauss_order`, has no friendly message
   in the value reads either.
3. **`test_schema_builder.py`'s docstring still names `schema_at_datastore-integrity-06.json` as
   the current witness.** §0.1 grants `WITNESS` only. Residue, for the orchestrator's follow-up; no
   issue opened.
4. **`RunRegistry/stores.py`'s comment above `FINGERPRINT_FORMAT` still names
   `full_store_fingerprint.json` as the golden that a bump regenerates.** §0.2 grants the one line.
   Residue; no issue opened.
5. **`GkWKBIntegration.build()` treats `z_source` as optional** (`payload.get("z_source", None)`,
   filtered only if given). This is pre-existing. `main.py` always passes it, and the labels-only
   reader's refusal is what would catch a lookup that spans `z_source`s. Not in the key question
   07 answers; no issue opened.
6. **The fixture's two pure-WKB Gk rows have `z_init` 50.0, not their `z_source`'s `z` (10.0).**
   They are raw rows, which `store()`'s check never sees (§0.1: "do not correct"). A store written
   through the factories could not hold them now.
7. **Nothing here has run under Ray or on a real store.** Prompt 10's end-to-end store is the first
   place `main.py`'s keyed stages, the check at `store()` and the labels-only readers meet
   production data.

## State handed to the next prompt

- **Names.** Every lookup and construction of a WKB row names its parent:
  - the column `numeric_serial` (README §4);
  - the marker `GkWKBIntegration.PURE_WKB`;
  - the reader mode `labels_only=True`;
  - the properties `numeric_serial` (Tk, Gk) and `pure_WKB` (Gk);
  - the factories' `_check_numeric_parent`;
  - `store_inventory.Parent(..., nullable=True)`;
  - `main.py`'s `Gk_WKB_numeric_parent`.
- **For 08.** `FINGERPRINT_FORMAT` is 2 and the golden is
  `full_store_fingerprint_datastore-integrity-07.json`. 08's re-capture adds its own file beside it
  and decides whether its identity change bumps the format again, by `stores.py`'s own rule. The
  witness is `schema_at_datastore-integrity-07.json`.
- **For 09.** The GkSource stage refuses a second Gk WKB row per `(k, z_source)` until `GkSource`
  is keyed on its Gk rows. `[07-wkb-value-reads-do-not-refuse-a-pre-07-store]` sits naturally with
  that work. The drop-order test's schema-derived check will hold 08's and 09's new foreign keys to
  `_drop_order` too.
- **The guard's pins:**
  - `EXPECTED_COVERAGE` 47;
  - `KNOWN_UNRESOLVED_TABLES` 11;
  - `KNOWN_BLIND` 1, `KNOWN_UNCHECKED_INSERTS` 13 and `KNOWN_BLIND_COLUMNS` 4;
  - `KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS` empty.
- **Baselines** after this commit: AdaptiveLevin 32, ComputeTargets **566**, CosmologyModels 39,
  Datastore **397**, LiouvilleGreen 148 (1 skipped), RunRegistry 199.
