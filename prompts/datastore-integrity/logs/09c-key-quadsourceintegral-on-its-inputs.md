# Log 09c — Key QuadSourceIntegral on the six rows it is computed from

**Prompt:** [`prompts/datastore-integrity/09c-key-quadsourceintegral-on-its-inputs.md`](../09c-key-quadsourceintegral-on-its-inputs.md)
**Commit:** *(this commit)* — "Key QuadSourceIntegral on the six rows it is computed from"
**Model:** Claude Opus 5.5
**Date:** 2026-09-30
**Result:** COMPLETE. K5 shipped, and with it K7 is complete, under D1, D4, D5 and D6, the user's
decisions of the orchestrator note's §0.1–§0.4, and the user's four decisions on the sketch (§0.8).
It closes `[09a-quadsourcevalue-model-read-joins-every-source-under-its-labels]` (§0.7, decision 3)
and opens no issue. No stop condition triggered:
- the six rows are `QuadSourceIntegral`'s only stored inputs (`compute()` receives `source`,
  `GkPolicy` and the four Tk rows; `b`, `Bessel_0pt5` and `Bessel_2pt5` are not stored rows), and in
  a store written as the pipeline writes it the relations hold for the rows `main.py`'s own
  resolution finds (`TestThePipelinesOwnLogic`, below);
- the stage resolves every work item's parents before its existence query without changing what
  work it schedules (Deviation 3), and all five `substitute_main_py()` literals still match once;
- no existing test outside §0.5's "Granted for 09c" changed, and each granted one only follows the
  key; in particular 09a's and 09b's `main.py` tests pass unmodified;
- `EXPECTED_SITES` grew by exactly this prompt's five parent-resolution sites, and the read-back
  guard's `EXPECTED_COVERAGE` by exactly its one new row-read site (§0.4); nothing fell;
- only `QuadSourceIntegral` moved in the fingerprint, and every witness hunk is in
  `QuadSourceIntegral`.

Suites after this commit, serially, on the final staged tree:
- ComputeTargets **635 → 665** (the 30 of the new `test_quadsource_integral_parent_main.py`); the
  known wall-clock flake did not fire in the final run.
- Datastore **505 → 532** (the 27 of the new `test_quadsource_integral_parent_key.py`);
- AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped) and RunRegistry 199,
  unchanged.

```
AdaptiveLevin: Ran 32 tests in 0.312s OK
CosmologyModels: Ran 39 tests in 3.095s OK
LiouvilleGreen: Ran 148 tests in 61.390s OK (skipped=1)
RunRegistry: Ran 199 tests in 90.311s OK
Datastore: Ran 532 tests in 185.401s OK
ComputeTargets: Ran 665 tests in 201.291s OK
```

The index `docs/OPEN_ISSUES.md` goes from **100 to 99** rows, counted
(`grep -c '^| \`\[' docs/OPEN_ISSUES.md`): one deleted (the 09a issue), none opened.

## What shipped

**K5 — the key** (`Datastore/SQL/ObjectFactories/QuadSourceIntegral.py`).
- **Four new columns**, after `data_serial` in `register()`: **`Tq_numeric_serial`**,
  **`Tq_WKB_serial`**, **`Tr_numeric_serial`** and **`Tr_WKB_serial`** (the prompt's choice of
  names: the compute payload's keys `Tq_numeric`, `Tq_WKB`, `Tr_numeric`, `Tr_WKB`, with `_serial`).
  Each NOT NULL, indexed, and with **no** foreign key; one comment above them says why (a Tk row is
  on its own mode's shard, q's or r's, not necessarily the integral's, which is k's; serials are
  unique across shards). README §4 names them.
- **`build()`** takes the six parents by serial in its payload, under the columns' names
  (`source_serial`, `data_serial` and the four), and filters on all six after the labels. It names
  all six or raises `RuntimeError` naming the missing ones (there is no reader mode: readers use
  `read_batch()`, D1); a parent that is not an `int` (a `bool`, a float, an object) is `TypeError`
  (one module-level helper, `_parent_serials`). A miss and a hit both carry the six they were asked
  under.
- **`store()`** refuses (`ValueError`) an object that does not name all six, before inserting, and
  writes the four new columns beside `source_serial` and `data_serial`.
- **`read_batch()`** selects `quad_policy_serial` and then the four new columns (the user's decision
  3), hands all six parents back on every object, and filters on each parent its payload names —
  any of them, or none. A reader holding only physical labels names none, and is handed every
  match, each with the parents that distinguish it.
- **`inventory_records`** declares `"data": Parent("data_serial", "GkSourcePolicyData")`, same-shard,
  and `"source"`, `"Tq_numeric"`, `"Tq_WKB"`, `"Tr_numeric"` and `"Tr_WKB"` with 09a's
  `cross_shard=True`. Its comment, which called `source_serial` and `data_serial` "provenance", now
  says what they are.
- **Old stores (D4).** `_refuse_a_store_that_predates_the_transfer_function_key`, beside 08's helper
  and in its pattern: it matches the driver's message for any of the four names and raises 07's form
  of refusal. `build()` and `read_batch()` call 08's helper first and then it.

**K5 — the compute target** (`ComputeTargets/QuadSourceIntegral.py`).
- `QuadSourceIntegral.PARENTS`, the six names. The constructor takes them as keyword arguments. An
  object to be computed (`payload=None`) **requires all six** (`ValueError` naming the missing);
  a stored object takes them from its payload. A `parents` property (and one for each Tk serial)
  reports them whether or not the object is populated.
- **`compute()` asserts**, after its existing checks of policy, `z_response`, `z_source_max` and the
  modes and before anything is dispatched (`_check_the_rows_it_is_keyed_on`), raising
  `RuntimeError` naming both sides:
  - that the six rows handed in are the six it is keyed on;
  - that the `Tq` numeric row is its source's `Tq_serial`, and the `Tr` numeric row its `Tr_serial`
    (audit §2.4: nothing checked this before);
  - that each WKB row's `numeric_serial` is its own numeric row.
- `store()` used to **overwrite** `_source_serial` and `_data_serial` with the serials the
  computation reported. Those two lines are removed (Deviation 2).

**`QuadSourceValue._build_impl_model`** (`Datastore/SQL/ObjectFactories/QuadSource.py`; the user's
decision 3 of §0.7). It selects its candidate `QuadSource`s' `serial`, `Tq_serial` and `Tr_serial`,
refuses more than one with `MultipleResultsFound` naming each serial and pair, and serves one match
as before. This closes the 09a issue.

**`main.py`** (table below): one new nested helper, `resolve_QuadSourceIntegral_parents`, and the
`QuadSourceIntegral` stage restructured to resolve every work item's six parents before its
existence query.

**The reader (D1)** (`extract_QuadSourceIntegral_data.py`): a module-level
`refuse_ambiguous_QuadSourceIntegrals(integral_data)`, called first thing in
`plot_QuadSourceIntegral`. It groups the rows `read_batch` returned by `z_response`, and where one
`(k, q, r, z_response)` has more than one it raises `RuntimeError` naming each row's serial and the
parents that differ among them (and `z_source_max`, if that differs); otherwise it returns the rows
unchanged. The `read_batch` call names no parent, and has a comment saying so.

**K7 — complete.** The six parents are declared (above); no change to `store_inventory.py` was
needed: 09a's cross-shard `Parent` resolves `source` and the four Tk rows, and `data` resolves on the
integral's own shard, each to the parent's own reference digest, never its serial.

**D5.** `Datastore/tests/data/schema_at_datastore-integrity-09c.json`, captured with
`PYTHONPATH=. ./venv/bin/python Datastore/tests/schema_description.py
Datastore/tests/data/schema_at_datastore-integrity-09c.json`; `RunRegistry/tests/data/
full_store_fingerprint_datastore-integrity-09c.json`; `FINGERPRINT_FORMAT` 5 → 6 with its "Format 6"
sentence. No older witness or golden is touched.

**Tests** (new):
- `Datastore/tests/test_quadsource_integral_parent_key.py`, 27 tests: §3 tests 1, 3, 5 and 6, the
  four columns' declaration, and `QuadSourceValue`'s model read.
- `ComputeTargets/tests/test_quadsource_integral_parent_main.py`, 30 tests: §3 tests 2, 4 and 7, the
  constructor, and the pipeline's own logic on a store.
- Each module's `tearDownModule` asserts `ray.is_initialized()` is false.

## The parent table

Shards from `config/sharding.py`: `"QuadSourceIntegral": "k"`, `"QuadSource": "q"`,
`"GkSourcePolicyData": "k"`, `"TkNumericIntegration": "k"`, `"TkWKBIntegration": "k"` (so a Tk row
for a mode is on that mode's shard).

| Parent | Column | Its shard | Foreign key | Declared to the inventory |
|---|---|---|---|---|
| the `QuadSource` | `source_serial` (existing) | `q`'s | no, and it cannot; indexed (existing) | `Parent("source_serial", "QuadSource", cross_shard=True)` |
| the `GkSourcePolicyData` | `data_serial` (existing) | `k`'s, the integral's own | yes, `GkSourcePolicyData.serial` (existing) | `Parent("data_serial", "GkSourcePolicyData")`, same shard |
| the `TkNumericIntegration` for q | `Tq_numeric_serial` (new) | `q`'s | no; indexed | `Parent("Tq_numeric_serial", "TkNumericIntegration", cross_shard=True)` |
| the `TkWKBIntegration` for q | `Tq_WKB_serial` (new) | `q`'s | no; indexed | `Parent("Tq_WKB_serial", "TkWKBIntegration", cross_shard=True)` |
| the `TkNumericIntegration` for r | `Tr_numeric_serial` (new) | `r`'s | no; indexed | `Parent("Tr_numeric_serial", "TkNumericIntegration", cross_shard=True)` |
| the `TkWKBIntegration` for r | `Tr_WKB_serial` (new) | `r`'s | no; indexed | `Parent("Tr_WKB_serial", "TkWKBIntegration", cross_shard=True)` |

A mode's shard may be `k`'s; the declaration does not change (the prompt). The relations asserted
between them: `Tq_numeric_serial` is the source's `Tq_serial` and `Tr_numeric_serial` its
`Tr_serial`; each WKB row's `numeric_serial` is its numeric row.

## Every `main.py` site changed

Line numbers from `git show HEAD:main.py` (before) and `git show :main.py` of the staged tree
(after). `git diff --cached -U0 main.py` has twelve hunks: `@@ -3294,0 +3295,313 @@`,
`@@ -3312 +3625,29 @@`, `@@ -3327,0 +3669,3 @@`, `@@ -3340 +3684 @@`, `@@ -3374 +3718,8 @@`,
`@@ -3377,3 +3728,4 @@`, `@@ -3390,2 +3742,2 @@`, `@@ -3394 +3746 @@`, `@@ -3396,6 +3748 @@`,
`@@ -3410,6 +3757,3 @@`, `@@ -3721,0 +4066,11 @@`, `@@ -3746,0 +4102 @@`.

| Site | Before | After |
|---|---|---|
| **The helper** `resolve_QuadSourceIntegral_parents(batch, GkSource_parents)`, nested in `run_pipeline` beside the stage: for every item of the batch, the six parents' serials, serial-only | — | def `:3295` |
| its `TkNumericIntegration` lookup, one per distinct mode, `_do_not_populate`, the stage's own tolerances and tags | — | batch `:3333`, `object_get` `:3357` |
| its `TkWKBIntegration` lookup, one per mode with a numeric row, keyed on it (`numeric_serial`), `_do_not_populate` | — | batch `:3382`, `object_get` `:3405` |
| its `QuadSource` lookup, vectorized by q, keyed on the two numeric rows (`Tq_serial`, `Tr_serial`), `_do_not_populate` | — | batch `:3438`, `object_get_vectorized` `:3467` |
| its `GkSource` lookup, vectorized by k, keyed on the parent set it is handed (`parents`), `_do_not_populate` | — | batch `:3491`, `object_get_vectorized` `:3522` |
| its `GkSourcePolicyData` lookup, vectorized by k, keyed on that `GkSource` (`source`) | — | batch `:3547`, `object_get_vectorized` `:3564` |
| `build_QuadSourceIntegral_batch` | def `:3295` | def `:3608` |
| the Gk parent sets, resolved once for **every** `(k, z_response)` of the batch, **before** the existence query (it was for the missing ones, after it) | `resolve_GkSource_parents` `:3413` | `:3631` |
| the six parents, resolved for every item | — | `resolve_QuadSourceIntegral_parents` `:3637` |
| the items whose six parents are all in the store, which alone are queried | — | `queried` `:3643` |
| the existence query, keyed on the six (`**QuadSourceIntegral_parents[...]`) and over `queried[k]` | payload `:3314`, `object_get_vectorized` `:3350` | payload `:3655`, parents `:3669`, `queried[k]` `:3684`, call `:3694` |
| the missing items: those not found, plus those with a parent missing, in the batch's order | `missing_labels` `:3375` | `found_labels` `:3720`, `missing_labels` `:3726` |
| `missing_Gk` and `missing_source_a`, from `missing_labels` (they re-zipped the query results) | `:3386`, `:3396` | `:3738`, `:3748` |
| the full `GkSource` lookup of the missing items, keyed on the set resolved above (only its comment changed) | `:3417` | `:3761` |
| the object to be computed: the six parents its existence query named, and a stop if an item's parents were not all found (unreachable, Deviation 4) | `object_get("QuadSourceIntegral", …)` `:3725` | `parents` `:4069`, `"QuadSourceIntegral"` `:4080`, `**parents` `:4102` |

Unchanged: `build_QuadSourceIntegral_payload` (`:350`), the full Tk numeric, Tk WKB and
`QuadSource` lookups of the missing items and their stops, and everything from the construction of
`qsi_work_items` to the `qsi_work_batches` line (`:4148`).

**`test_main_plumbing.EXPECTED_SITES`** (§0.4): five counts rose by one, each by one site of the new
helper, which is the parent resolution this prompt adds; no count fell and none other moved.

| Class | Before | After | The site |
|---|---|---|---|
| `TkNumericIntegration` | 6 | 7 | `resolve_QuadSourceIntegral_parents: RayWorkPool(Tk_numeric_serial_batch)`, `main.py:3357` |
| `TkWKBIntegration` | 3 | 4 | `RayWorkPool(Tk_WKB_serial_batch)`, `:3405` |
| `QuadSource` | 3 | 4 | `RayWorkPool(QuadSource_serial_batch)`, `:3467` |
| `GkSource` | 6 | 7 | `RayWorkPool(GkSource_serial_batch)`, `:3522` |
| `GkSourcePolicyData` | 4 | 5 | `RayWorkPool(GkSourcePolicy_serial_batch)`, `:3564` |

The rest are unchanged: `wavenumber_exit_time` 1, `GkNumericIntegration` 3, `QuadSourceIntegral`
2, `GkWKBIntegration` 3, `BackgroundModel` 1, `GkSourcePolicy` 2, `QuadSourcePolicy` 2,
`IntegrationSolver` 7, `redshift` 1, `store_tag` 10, `tolerance` 8, `wavenumber` 1. The new
Tk-numeric site carries `Tk_numeric_atol`/`Tk_numeric_rtol`, and the other four carry no tolerance,
as `test_every_site_carries_its_own_target_tolerance` and
`test_the_tolerance_free_targets_carry_no_tolerance_at_all` check; they pass. The comment above
`EXPECTED_SITES` names the helper; this is the one edit to `test_main_plumbing.py`.

**The read-back guard's pins** (§0.4), `python -c` over `test_factory_select_columns` at `HEAD` and
on the staged tree: `EXPECTED_COVERAGE` **50 → 51**, the one new row-read binding of this prompt,
named in the test with a comment:
`QuadSource.py:sqla_QuadSourceValue_factory._build_impl_model.candidate` (the model read's
candidate `QuadSource`s, reading `serial`, `Tq_serial` and `Tr_serial`). `KNOWN_UNRESOLVED_TABLES`
11, `KNOWN_BLIND` 1, `KNOWN_UNCHECKED_INSERTS` 13 and `KNOWN_BLIND_COLUMNS` 4 did not move;
`KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS` are empty. Every column name in the factory is literal
(the §0.8 finding on `table.c[name]`); `test_dynamic_column_sites_are_the_declared_ones` passes.

**Why the helper is a function of its own.** 09a's `test_quadsource_parent_main` pins the order of
the `QuadSource` and `TkNumericIntegration` lookups inside `build_QuadSourceIntegral_batch` (one of
each, Tk first), and 09b's `test_gksource_parent_set_main` pins one `GkSource` lookup there. Neither
is granted. Serial-only lookups written into the stage itself would have failed both, so they are in
a sibling function the stage calls; both modules pass unmodified. It also let
`TestThePipelinesOwnLogic` compile and run exactly that code against the real factories (Deviation
1).

## The measured cost of the extra lookups

On a stand-in pool (`Datastore/tests/standin_pool.py`: the real `ShardedPool`, `Datastore` actor
code and factories, results pickled through, no Ray), two shards, holding the parent rows of a
production-sized grid: 50 modes (the source and response k-grids are the same 50 wavenumbers,
`1e5`–`3e8`), a `TkNumericIntegration` and a `TkWKBIntegration` for each, a `QuadSource` for each of
the 1,275 `(q, r)` pairs, and a `GkSource` and `GkSourcePolicyData` for each of 5,000 `(k,
z_response)`. The batch is `main.py`'s own first work batch: the production ordering, triangle
filtered, first 750 items. `main.py`'s own `resolve_QuadSourceIntegral_parents`, compiled from its
source with `ast`, was run on it five times (`measure_cost.py`, in this prompt's scratch directory,
not in the tree), at a load average of 12–21, with this prompt's own suite run going on
beside it; so these are upper figures:

```
rows: {'wavenumber_exit_time': 100, 'TkNumericIntegration': 50, 'TkNumeric_tags': 50, 'TkWKBIntegration': 50, 'TkWKB_tags': 50, 'QuadSource': 1275, 'QuadSource_tags': 1275, 'GkSource': 5000, 'GkSource_tags': 5000, 'GkSourcePolicyData': 5000}
batch: 750 items; 1 z_response, 335 (q, r) pairs, 50 modes, 50 (k, z_response) pairs
resolved: 750 of 750 items have all six parents
actor calls per resolution: {('object_get', 'TkNumericIntegration'): 50, ('object_get', 'TkWKBIntegration'): 50, ('object_get', 'QuadSource'): 8, ('object_get', 'GkSource'): 50, ('object_get', 'GkSourcePolicyData'): 50}
resolve_QuadSourceIntegral_parents, 5 runs: median 1.116 s (min 0.957, max 1.777)
per item: 1.49 ms
```

So the new helper costs about **1.1 s per batch of 750** (≈1.5 ms an item), in 208 actor calls: 50
Tk numeric, 50 Tk WKB, 8 `QuadSource` (vectorized by q), 50 `GkSource` and 50
`GkSourcePolicyData` (vectorized by k). To that is added the Gk parent-set resolution, which the
stage now makes for every distinct `(k, z_response)` of a batch instead of only the missing ones:
50 here, at 09b's measured ≈18 ms each (log 09b), ≈0.9 s. So a batch whose integrals all exist,
which before this prompt cost one vectorized existence query per k, now costs about 2 s more. A
full production pass is about 5,133 triangle-closing items per `z_response` (6.8 batches), so of
order 23 minutes of serial-only lookups over 100 `z_response`s, before any parallelism across shards.
README §5 rule 11: recorded, not argued against.

## The `substitute_main_py()` output

`docs/handover/quadsource_atol_sweep.py` loaded by path with `importlib.util.spec_from_file_location`
and `substitute_main_py()` called on the staged tree (it only reads `main.py`). The five
`** quadsource_atol_sweep:` lines are its own, and it raises unless each literal occurs exactly
once; the last five lines count each literal in `main.py` and give its line:

```
** quadsource_atol_sweep: replaced 1 occurrence of 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_SOURCE_K_VALUES)' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_RESPONSE_K_VALUES)' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'tol=DEFAULT_QUADRATURE_ATOL' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'tol=DEFAULT_QUADRATURE_RTOL' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of '        qsi_work_batches = list(grouper(qsi_work_items, n=750, incomplete="fill"))' in main.py
substituted source: 186374 characters
count=1 line=4354 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_SOURCE_K_VALUES)'
count=1 line=4366 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_RESPONSE_K_VALUES)'
count=1 line=4255 'tol=DEFAULT_QUADRATURE_ATOL'
count=1 line=4256 'tol=DEFAULT_QUADRATURE_RTOL'
count=1 line=4148 'qsi_work_batches = list(grouper(qsi_work_items, n=750, incomplete="fill"))'
```

All five still match exactly once. Each moved by the 356 lines this commit adds above them
(`:3792` → `:4148`, `:3899` → `:4255`, `:3900` → `:4256`, `:3998` → `:4354`, `:4010` → `:4366`). The
sweep script is not edited. `TestMainPy.test_the_work_list_lines_are_unchanged` holds the work list's
construction and the `qsi_work_batches` line as they stood at `9a18063`, and finds each once.

## The witness diff, in full (D5)

Pasted from `diff Datastore/tests/data/schema_at_datastore-integrity-09b.json
Datastore/tests/data/schema_at_datastore-integrity-09c.json`, default format, unedited. **5 hunks,
102 changed lines, 108 lines of output** (`grep -c '^[0-9]'` = 5, `grep -c '^[<>]'` = 102, `wc -l` =
108), the sketch's counts exactly. Each hunk is located by the JSON key that encloses it (scanned
upward from its first line in the new witness):

| Kind | Hunk | Enclosing key |
|---|---|---|
| 1. the four columns in the schema record | `6590a6591,6606` | `classes.QuadSourceIntegral.record.columns` |
| 2. the four columns' descriptions | `6972a6989,7040` | `classes.QuadSourceIntegral.table.columns` |
| 3. the `CREATE TABLE` DDL | `7512c7580` | `classes.QuadSourceIntegral.table.ddl` |
| 4. the four index DDL lines | `7513a7582,7585` | `classes.QuadSourceIntegral.table.index_ddl` |
| 5. the four index descriptions | `7530a7603,7630` | `classes.QuadSourceIntegral.table.indexes` |

**5 kinds, all in `QuadSourceIntegral`. No hunk is outside it**, and none names a foreign key.

```
6590a6591,6606
>       "of_table": "QuadSourceIntegral"
>      },
>      {
>       "column": "Tq_numeric_serial",
>       "of_table": "QuadSourceIntegral"
>      },
>      {
>       "column": "Tq_WKB_serial",
>       "of_table": "QuadSourceIntegral"
>      },
>      {
>       "column": "Tr_numeric_serial",
>       "of_table": "QuadSourceIntegral"
>      },
>      {
>       "column": "Tr_WKB_serial",
6972a6989,7040
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
>       "foreign_keys": [],
>       "index": true,
>       "name": "Tq_numeric_serial",
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
>       "foreign_keys": [],
>       "index": true,
>       "name": "Tq_WKB_serial",
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
>       "foreign_keys": [],
>       "index": true,
>       "name": "Tr_numeric_serial",
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
>       "foreign_keys": [],
>       "index": true,
>       "name": "Tr_WKB_serial",
7512c7580
<     "ddl": "CREATE TABLE \"QuadSourceIntegral\" (\n\tserial INTEGER NOT NULL, \n\tversion INTEGER, \n\ttimestamp DATETIME, \n\tlabel VARCHAR(256), \n\tmodel_serial INTEGER NOT NULL, \n\tk_wavenumber_exit_serial INTEGER NOT NULL, \n\tq_wavenumber_exit_serial INTEGER NOT NULL, \n\tr_wavenumber_exit_serial INTEGER NOT NULL, \n\tatol_serial INTEGER NOT NULL, \n\trtol_serial INTEGER NOT NULL, \n\tpolicy_serial INTEGER NOT NULL, \n\tquad_policy_serial INTEGER NOT NULL, \n\tsource_serial INTEGER NOT NULL, \n\tdata_serial INTEGER NOT NULL, \n\tz_response_serial INTEGER NOT NULL, \n\tz_source_max_serial INTEGER NOT NULL, \n\tb FLOAT NOT NULL, \n\ttotal FLOAT NOT NULL, \n\ttotal_abserr FLOAT, \n\ttotal_converged BOOLEAN, \n\ttotal_phase_limited BOOLEAN, \n\tnumeric_quad FLOAT NOT NULL, \n\t\"WKB_Levin\" FLOAT NOT NULL, \n\tanalytic_rad FLOAT, \n\teta_source_max FLOAT, \n\teta_response FLOAT, \n\tnumeric_quad_compute_time FLOAT, \n\tnumeric_quad_compute_steps INTEGER, \n\t\"numeric_quad_RHS_evaluations\" INTEGER, \n\t\"numeric_quad_mean_RHS_time\" FLOAT, \n\t\"numeric_quad_max_RHS_time\" FLOAT, \n\t\"numeric_quad_min_RHS_time\" FLOAT, \n\t\"WKB_Levin_num_regions\" INTEGER, \n\t\"WKB_Levin_evaluations\" INTEGER, \n\t\"WKB_Levin_simple_regions\" INTEGER, \n\t\"WKB_Levin_SVD_errors\" INTEGER, \n\t\"WKB_Levin_order_changes\" INTEGER, \n\t\"WKB_Levin_chebyshev_min_order\" INTEGER, \n\t\"WKB_Levin_max_depth\" INTEGER, \n\t\"WKB_Levin_elapsed\" FLOAT, \n\t\"WKB_phase_spline_chunks\" INTEGER, \n\tcompute_time FLOAT, \n\tanalytic_compute_time FLOAT, \n\tmetadata VARCHAR(256), \n\tPRIMARY KEY (serial), \n\tFOREIGN KEY(version) REFERENCES version (serial), \n\tFOREIGN KEY(model_serial) REFERENCES \"BackgroundModel\" (serial), \n\tFOREIGN KEY(k_wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(q_wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(r_wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(atol_serial) REFERENCES tolerance (serial), \n\tFOREIGN KEY(rtol_serial) REFERENCES tolerance (serial), \n\tFOREIGN KEY(policy_serial) REFERENCES \"GkSourcePolicy\" (serial), \n\tFOREIGN KEY(quad_policy_serial) REFERENCES \"QuadSourcePolicy\" (serial), \n\tFOREIGN KEY(data_serial) REFERENCES \"GkSourcePolicyData\" (serial), \n\tFOREIGN KEY(z_response_serial) REFERENCES redshift (serial), \n\tFOREIGN KEY(z_source_max_serial) REFERENCES redshift (serial)\n)",
---
>     "ddl": "CREATE TABLE \"QuadSourceIntegral\" (\n\tserial INTEGER NOT NULL, \n\tversion INTEGER, \n\ttimestamp DATETIME, \n\tlabel VARCHAR(256), \n\tmodel_serial INTEGER NOT NULL, \n\tk_wavenumber_exit_serial INTEGER NOT NULL, \n\tq_wavenumber_exit_serial INTEGER NOT NULL, \n\tr_wavenumber_exit_serial INTEGER NOT NULL, \n\tatol_serial INTEGER NOT NULL, \n\trtol_serial INTEGER NOT NULL, \n\tpolicy_serial INTEGER NOT NULL, \n\tquad_policy_serial INTEGER NOT NULL, \n\tsource_serial INTEGER NOT NULL, \n\tdata_serial INTEGER NOT NULL, \n\t\"Tq_numeric_serial\" INTEGER NOT NULL, \n\t\"Tq_WKB_serial\" INTEGER NOT NULL, \n\t\"Tr_numeric_serial\" INTEGER NOT NULL, \n\t\"Tr_WKB_serial\" INTEGER NOT NULL, \n\tz_response_serial INTEGER NOT NULL, \n\tz_source_max_serial INTEGER NOT NULL, \n\tb FLOAT NOT NULL, \n\ttotal FLOAT NOT NULL, \n\ttotal_abserr FLOAT, \n\ttotal_converged BOOLEAN, \n\ttotal_phase_limited BOOLEAN, \n\tnumeric_quad FLOAT NOT NULL, \n\t\"WKB_Levin\" FLOAT NOT NULL, \n\tanalytic_rad FLOAT, \n\teta_source_max FLOAT, \n\teta_response FLOAT, \n\tnumeric_quad_compute_time FLOAT, \n\tnumeric_quad_compute_steps INTEGER, \n\t\"numeric_quad_RHS_evaluations\" INTEGER, \n\t\"numeric_quad_mean_RHS_time\" FLOAT, \n\t\"numeric_quad_max_RHS_time\" FLOAT, \n\t\"numeric_quad_min_RHS_time\" FLOAT, \n\t\"WKB_Levin_num_regions\" INTEGER, \n\t\"WKB_Levin_evaluations\" INTEGER, \n\t\"WKB_Levin_simple_regions\" INTEGER, \n\t\"WKB_Levin_SVD_errors\" INTEGER, \n\t\"WKB_Levin_order_changes\" INTEGER, \n\t\"WKB_Levin_chebyshev_min_order\" INTEGER, \n\t\"WKB_Levin_max_depth\" INTEGER, \n\t\"WKB_Levin_elapsed\" FLOAT, \n\t\"WKB_phase_spline_chunks\" INTEGER, \n\tcompute_time FLOAT, \n\tanalytic_compute_time FLOAT, \n\tmetadata VARCHAR(256), \n\tPRIMARY KEY (serial), \n\tFOREIGN KEY(version) REFERENCES version (serial), \n\tFOREIGN KEY(model_serial) REFERENCES \"BackgroundModel\" (serial), \n\tFOREIGN KEY(k_wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(q_wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(r_wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(atol_serial) REFERENCES tolerance (serial), \n\tFOREIGN KEY(rtol_serial) REFERENCES tolerance (serial), \n\tFOREIGN KEY(policy_serial) REFERENCES \"GkSourcePolicy\" (serial), \n\tFOREIGN KEY(quad_policy_serial) REFERENCES \"QuadSourcePolicy\" (serial), \n\tFOREIGN KEY(data_serial) REFERENCES \"GkSourcePolicyData\" (serial), \n\tFOREIGN KEY(z_response_serial) REFERENCES redshift (serial), \n\tFOREIGN KEY(z_source_max_serial) REFERENCES redshift (serial)\n)",
7513a7582,7585
>      "CREATE INDEX \"ix_QuadSourceIntegral_Tq_WKB_serial\" ON \"QuadSourceIntegral\" (\"Tq_WKB_serial\")",
>      "CREATE INDEX \"ix_QuadSourceIntegral_Tq_numeric_serial\" ON \"QuadSourceIntegral\" (\"Tq_numeric_serial\")",
>      "CREATE INDEX \"ix_QuadSourceIntegral_Tr_WKB_serial\" ON \"QuadSourceIntegral\" (\"Tr_WKB_serial\")",
>      "CREATE INDEX \"ix_QuadSourceIntegral_Tr_numeric_serial\" ON \"QuadSourceIntegral\" (\"Tr_numeric_serial\")",
7530a7603,7630
>        "Tq_WKB_serial"
>       ],
>       "name": "ix_QuadSourceIntegral_Tq_WKB_serial",
>       "unique": false
>      },
>      {
>       "columns": [
>        "Tq_numeric_serial"
>       ],
>       "name": "ix_QuadSourceIntegral_Tq_numeric_serial",
>       "unique": false
>      },
>      {
>       "columns": [
>        "Tr_WKB_serial"
>       ],
>       "name": "ix_QuadSourceIntegral_Tr_WKB_serial",
>       "unique": false
>      },
>      {
>       "columns": [
>        "Tr_numeric_serial"
>       ],
>       "name": "ix_QuadSourceIntegral_Tr_numeric_serial",
>       "unique": false
>      },
>      {
>       "columns": [
```

## The fingerprint re-capture (D5)

`FINGERPRINT_FORMAT` goes from 5 to 6. The golden was captured the way
`test_store_fingerprint._FullStores` builds its baseline: `build_full_store` into a temporary
directory, `read_inventory(...)`, `fingerprint_of(..., taken=None)`, serialised as
`json.dumps(indent=2, sort_keys=True) + "\n"` (`capture_golden.py`, in this prompt's scratch
directory, not in the tree). **The script was first run on a clean `git archive` of `HEAD` and
reproduced `full_store_fingerprint_datastore-integrity-09b.json` byte for byte** (`cmp`).

- **Moved**, checked class by class (a script comparing `classes[name]` of the two goldens; both list
  the same 21 classes in the same order):

  | Class | Count | Tag-set counts | Digest | Why |
  |---|---|---|---|---|
  | `QuadSourceIntegral` | 3 → 3 | [2, 1] → [2, 1] | `817af21962788bc4…` → `cc4367bbf0482046…` | its key gains the six parents, each resolved to its row's own identity |

- **Not moved: the other 20 classes**, identical in count, digest and tag sets. Nothing keys on a
  `QuadSourceIntegral`'s identity.
- `problems` is `{}` before and after; `taken` is null in both. The overall digest
  `a80e53c2dc30e97f…` → `6576b85d72e8acb5…`.
- The diff of the two goldens, in full (`diff`, unedited): three digests of `QuadSourceIntegral`
  (the class and its two tag sets), and the overall digest and format:

```
179c179
<       "digest": "817af21962788bc42697fe87e8eae95c38e815fb0e549bb4e30b46f2263dc829",
---
>       "digest": "cc4367bbf04820464edb6b3f137609d8d1ea317cf93ab1c84b2a5527af86e725",
183c183
<           "digest": "afd7dce3e59053d58451ac0baae1834f457c0b81f00772706ad01d462a36a399",
---
>           "digest": "99e793cb6c94b005db5284f8008e8b3878805813c548fd6a2c95623eb15551bd",
190c190
<           "digest": "5e71fbce637d4274aff3c02e224c92bacf3af07e5ef64daa6d9ff91a2f7c98be",
---
>           "digest": "7a0672b255ade31043be032f4437ffeca8dd1cdedb541a4aa6d7ad584d361ea1",
319,320c319,320
<   "digest": "a80e53c2dc30e97ff194a1c2f2fd40ab61a3463bc172e1f36d6ffb4aff7f5b94",
<   "fingerprint_format": 5,
---
>   "digest": "6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759",
>   "fingerprint_format": 6,
```

- **`store retire` of a store written before this prompt must be run at a pre-09c SHA (D4).** Its
  recorded fingerprint is format 5 or older; at this tree `compare_fingerprints` says the formats
  differ and does not compare, and an inventory recomputed here finds `QuadSourceIntegral`
  `incomplete` for want of the four columns. `store fingerprint` and `store amend` at this tree
  record format 6.

## Old stores (D4)

- **A pre-09c table is refused by both readers.** `TestAnOldStoreIsRefused` recreates
  `QuadSourceIntegral` from `schema_at_datastore-integrity-09b.json`'s own DDL (it has
  `quad_policy_serial` and none of the four): `build()`, `read_batch()` untagged, tagged and given
  the six parents each raise `RuntimeError` from the `OperationalError`: *"QuadSourceIntegral.build():
  the QuadSourceIntegral table has no "Tq_numeric_serial", "Tq_WKB_serial", "Tr_numeric_serial" or
  "Tr_WKB_serial" column. This datastore predates QuadSourceIntegral rows being keyed on the
  TkNumericIntegration and TkWKBIntegration rows for q and r they are computed from
  (prompts/datastore-integrity, prompt 09c). This tree does not read it; it remains readable at the
  git SHA its sidecar records."* It does not say "regenerated". Another database error is re-raised
  unchanged.
- **A pre-08 table keeps 08's message** (the user's decision 3). From
  `schema_at_datastore-integrity-07.json`'s DDL, which lacks all five columns, `build()` and
  `read_batch()` (untagged and tagged) raise 08's *"… no "quad_policy_serial" column …"*. `build()`
  filters on `quad_policy_serial` before the six parents, and `read_batch()` selects it before the
  four, so SQLite names it first; each also asks 08's helper first. 08's own
  `TestAnOldStoreIsRefused` passes with no edit to its assertions.
- **`store retire`** of a store written before this prompt must be run at a pre-09c SHA (above).

## Readers (README §5 rule 11), and what now refuses

| Reader | Before | After |
|---|---|---|
| `main.py`, the `QuadSourceIntegral` stage | existence query by labels; parents looked up for the missing items only | the six parents resolved serial-only for every item first; the existence query and the object to be computed keyed on them; full parents for the missing items only, as before |
| `extract_QuadSourceIntegral_data.py` (`object_read_batch`, `:1137` at `HEAD`) | `read_batch` by labels, every row used | still by labels, naming no parent; more than one row at one `(k, q, r, z_response)` refused, naming the parents that distinguish them |
| `QuadSourceValue._build_impl_model` (no caller in the tree; `probe_05` uses the class) | joined every matching `QuadSource`, raising `MultipleResultsFound` naming nothing | refuses more than one, naming each serial and `(Tq_serial, Tr_serial)`; serves one |
| `docs/handover/quadsource_atol_sweep.py`, `docs/gktk-remedial/scoped_pipeline_run.py` | `exec` `main.py` | keyed through `main.py`; not edited |
| `docs/source-remediation-verification/run_quadsource_integrals.py` | calls `compute_QuadSource_integral` directly | unchanged (it does not construct the target) |
| `tools/inventory_report.py` | a `QuadSourceIntegral`'s labels | also its six parents, rendered by physical labels (it already renders parent references; not edited) |
| `store retire`, `store fingerprint` comparisons | format 5 | format 6 |

**What now refuses a store it read before:**
- **Every store written before this commit** (the three under `var/` included, not opened here), in
  `main.py`'s `QuadSourceIntegral` stage (its existence query) and in
  `extract_QuadSourceIntegral_data.py`, and so in the sweep script and the scoped driver, which
  `exec` `main.py`: the D4 message. A pre-08 store gets 08's.
- **The extraction script, on a store holding two integrals at one `(k, q, r, z_response)`**, which
  it used to plot both of; a store built once holds one.
- **`QuadSourceValue`'s model read, on two sources under one set of labels** (it raised before too,
  naming nothing).
- **`main.py`'s stage now stops for an item any of whose six parents is missing even when a row
  under its labels exists** (Deviation 3).
- **`store retire` and `store fingerprint` comparisons**, as above.
- **The audit probes**, below.

### The fourteen audit probes, before and after

Run at `HEAD` (`9a18063`, a `git archive`, `venv` symlinked) and on an archive of the final staged
tree (`git write-tree`, then `git archive`), each probe in a subprocess with `PYTHONPATH` at its root,
both under the scratch directory: **the probes were not run in the checkout**, since
`replicated_write_fault_probe.py` writes a store beside itself. No probe opens anything under
`var/`: each builds its schema in memory or a store in a temporary directory (`lookup_keys/common.py`
mentions `var/` in a docstring only), and `replicated_serial_census.py` reads only the store paths
given as arguments and was run with none. Outputs compared after normalising the root path. **No
probe edited.**

| Probe | Exit before → after | Output |
|---|---|---|
| `fk_sweep_probe.py` | 0 → 0 | **changed**: four new lines, `Tq_numeric_serial`, `Tq_WKB_serial`, `Tr_numeric_serial`, `Tr_WKB_serial`, each `FK=- nullable=False` |
| `hot_journal_probe.py` | 0 → 0 | identical |
| `probe_01_tagged_read_batch.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `probe_01b_fix_shape.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `probe_02_readback_guard_by_hand.py` | 0 → 0 | **changed**: `QuadSourceValue._build_impl_model` gains its `candidate` row (3 reads, clean) and its `row_data` binding moves (772 → 798); `QuadSourceIntegral.read_batch` moves (`@546` → `@681`) and its row reads 68 columns, not 64 (line 737 → 905); still clean |
| `probe_03_column_existence.py` | 0 → 0 | **changed**: "checked 1268" → "checked 1287 resolvable column references" |
| `probe_04_exit_time_and_solver.py` | 0 → 0 | identical |
| `probe_04b_exact_match_fix_shape.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `probe_05_value_lookups_and_read_batch.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `probe_06_oneloop_tags_and_redshift_zero.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `replicated_divergence_probe.py` | 0 → 0 | **changed**: one new line, `unresolved-parent: QuadSourceIntegral: 2 row(s) on shard #0 … (field(s) source) … serials 1, 2`: the probe's own fixture's integrals name a `QuadSource` its store does not resolve |
| `replicated_serial_census.py` | 0 → 0 | identical |
| `replicated_write_fault_probe.py` | 1 → 1 | identical but for the temporary directory name and a timestamp in one line (already failing at `HEAD`) |
| `tq_fk_probe.py` | 0 → 0 | identical |

None newly fails; five change output. The six that fail at `HEAD` are unchanged. The sketch
measured four changes and `probe_03` at 1284; this tree has 1287 because it adds three references
the sketch did not: `read_batch`'s select of `quad_policy_serial` (the user's decision 3) and
`QuadSourceValue`'s `Tq_serial` and `Tr_serial` (decision 3 of §0.7). The 19 new references are the
six parents in `build()`'s filter, the four columns in `read_batch`'s select and its six filters,
and those three. The five are named in the rule-11 list: `fk_sweep_probe`, `probe_02`, `probe_03`,
`replicated_divergence_probe` (whose fixture's integrals are now `unresolved-parent`), and the
timestamp line of `replicated_write_fault_probe`.

## Deviations from the prompt

1. **The parent resolution is a nested helper of its own, `resolve_QuadSourceIntegral_parents`,
   called by the stage.** `STRUCTURALLY REQUIRED`: written into `build_QuadSourceIntegral_batch`
   itself, its lookups fail 09a's and 09b's `ast` tests of that function, which pin its
   `QuadSource`, Tk numeric and `GkSource` lookups and are not granted (above). The stage still
   resolves before its existence query (`TestMainPy`). A benefit: the helper compiles alone, so
   `TestThePipelinesOwnLogic` runs `main.py`'s own code on the real factories.
2. **The target's `store()` no longer overwrites `_source_serial` and `_data_serial` with what the
   computation reported.** `STRUCTURALLY REQUIRED`, as 09a's Deviation 3: they are the key, fixed
   at construction, and `compute()` now asserts that the rows it computes from are these, so the
   overwrite would be a no-op or the silent replacement of the key. `evaluate_QuadSource_integral`
   still returns `GkPolicy_serial` and `source_serial` (Observation 3).
3. **An item any of whose six parents is not in the store is missing without an existence query.**
   `IMPLEMENTATION CHOICE`. No row can be keyed on a parent that does not exist, so such an item
   was always going to be a missing item; it then reaches the full lookups of its parents, whose
   stops (the `Tk_numeric_unavailable` stop, `build_QuadSourceIntegral_payload`'s "MISSING DATA"
   raise) report it as they always have. So the work scheduled is unchanged. What changes is a store
   in which an integral exists under its labels but one of its parents does not: before, that item
   was skipped; now the stage stops for it. The prompt's §5 asks the stage not to change what work it
   schedules; this is the key's own consequence, as 09a's Deviation 2 was for the `QuadSource` stage.
4. **A stop at the construction if an item's parents were not found before its existence query.**
   `IMPLEMENTATION CHOICE`: unreachable, since `build_QuadSourceIntegral_payload` has already raised
   for such an item; it turns `**None` into a message rather than a `TypeError`.
5. **The Gk parent sets are resolved once per batch, for every `(k, z_response)`, and the full
   `GkSource` lookup of the missing items reuses them.** `IMPLEMENTATION CHOICE`: resolving again
   for the missing ones would read the same rows twice. The call moved from after the existence
   query to before it; `resolve_GkSource_parents` itself is unchanged.
6. **`build()` takes the parents by serial, under the columns' names, and has no reader mode.**
   `IMPLEMENTATION CHOICE`, as 09a's lookups name parents by serial. `build()` decides whether to
   compute (D1), and the one reader uses `read_batch()`, which takes any of the six or none.
7. **`read_batch()` filters on each parent its payload names.** `IMPLEMENTATION CHOICE`, the shape
   the sketch had and that mutation (v) presumes; the reader names none.
8. **The reader's refusal groups by `z_response` and names `z_source_max` too if it differs.**
   `IMPLEMENTATION CHOICE`: the prompt's grouping is `(k, q, r, z_response)`, and the script reads
   one `(k, q, r)` at a time with no `z_source_max`; two rows differing only in `z_source_max` are
   refused and named rather than both plotted.
9. **`QuadSourceIntegral.PARENTS` and the `parents` property.** `IMPLEMENTATION CHOICE`: one list of
   the six names, used by the factory's `_parent_serials`, the constructor and the reader. The
   existing `source_serial` and `data_serial` properties are unchanged (Observation 2).
No `UNINTENDED DRIFT` remains in the tree. `git diff --cached -U0` shows only the hunks this log
lists.

## Existing-test edits, by §0.5's "Granted for 09c"

| File | Edit | Reason |
|---|---|---|
| `Datastore/tests/real_store_fixtures.py` | the three `QuadSourceIntegral` rows gain the four columns, naming real Tk rows: 1 → Tk numeric 1, Tk WKB 1, Tk numeric 2, Tk WKB 3; 2 → 1, 2, 2, 3; 3 → 2, 3, 3, 1. Integral 3's `Tr` WKB row, 1, cannot satisfy the relation (the fixture has no Tk WKB row starting from Tk numeric 3), and a comment says so and why it is kept (the user's decision 2) | the columns are NOT NULL, and the six are identity. Each integral's numeric rows are its source's `Tq_serial` and `Tr_serial`; every other WKB row starts from its numeric row (agent's choice of which) |
| same | `_UNDECLARED_REFERENCES["QuadSourceIntegral"]` gains the four columns → `TkNumericIntegration` / `TkWKBIntegration` | they have no foreign key, so `relabel_serials` must be told what they reference |
| `Datastore/tests/test_store_inventory.py` | `IDENTITY["QuadSourceIntegral"]` gains the six parents, varied on integral 2 to real, resolvable rows: `source` 1 → 2 (shard 1, cross-shard), `data` 2 → 1 (its own shard), `Tq_numeric` 1 → 3, `Tq_WKB` 2 → 1, `Tr_numeric` 2 → 3, `Tr_WKB` 3 → 2; a comment | the key's fields now include them |
| same | `NON_IDENTITY` loses `source_serial` and `data_serial` | they are identity |
| same | `test_a_tag_on_the_background_model_reaches_its_children` expects `QuadSourceIntegral` too, and `test_a_tagged_parents_digest_covers_its_tags` too, each with a sentence | integral 3's source and `Tr` numeric row are model 2's; integral 2's data is `GkSource` 2's policy data |
| same | `TestDuplicates.test_across_shards` copies integral 1's same-shard chain to shard 1 (`GkSourcePolicyData` 1, `GkSource` 1 with its tags and `GkSource_parents`, and Gk numeric 1 and Gk WKB 1 with their tags) before the copy of the integral, with a comment; the duplicate is made inline rather than by `_duplicate` | `data` is same-shard, so the copy on shard 1 resolves only against that chain there (the user's decision 1) |
| same | `TestOldStoresAndOrphans.check()` gains `cascade`, with a docstring; `test_a_missing_key_column_makes_the_class_incomplete` passes `cascade={"QuadSourceIntegral": 3}` with a comment | taking Tk WKB 3 away leaves all three integrals naming it (integrals 1 and 2 as their `Tr` WKB, 3 as its `Tq` WKB): their only new problems are `unresolved-parent`, and their records are the baseline's less those three, both asserted |
| `Datastore/tests/test_integral_read_batch.py` | both raw inserts gain the four columns | NOT NULL |
| `Datastore/tests/test_quadsource_policy_key.py` (08's) | a module constant `PARENTS`; its `payload()` names the six, and `store_row`'s object carries them | `build()` requires the six; `store()` writes them; nothing asserted changed |
| `ComputeTargets/tests/test_quadsource_integral.py` | `TestComputePayload`'s construction passes the six; `TestPersistedSchema`'s payload gains the four and its `common` arguments the six | the constructor requires them (§0.8 decision 4) |
| `ComputeTargets/tests/test_quadsource_policy_main.py` | `TestTheConstructorRequiresIt.arguments()` passes the six | the same |
| `Datastore/tests/test_schema_builder.py` | `WITNESS` → `schema_at_datastore-integrity-09c.json`; the docstring's "current one" sentence, with 09b's moved to "before it" | D5 |
| `RunRegistry/tests/test_store_fingerprint.py` | `GOLDEN` → `full_store_fingerprint_datastore-integrity-09c.json` | D5 |
| `ComputeTargets/tests/test_main_plumbing.py` | `EXPECTED_SITES`: five counts +1, and a comment naming the helper | §0.4, the five sites above |
| `Datastore/tests/test_factory_select_columns.py` | `EXPECTED_COVERAGE` gains `QuadSourceValue._build_impl_model.candidate`, with a comment | §0.4, the one new row-read site |

Not needed: `test_quadsource_policy_foreign_key.py`, `test_inventory_report.py`,
`test_inventory_consumers.py`, `test_foreign_key_check.py`, and 09a's and 09b's modules.

## Verification performed

- **The starting tree.** `git status` clean at `9a18063`; `python -m RunRegistry list` showed nothing
  `running` (of seven runs, five finished and two `unknown`, all days old). No other session's change
  appeared in this checkout.
- **Before the existing-test edits** (the new witness and golden already captured and pointed at,
  so the two D5 tests do not appear), Datastore ran 505 with 23 failing entries: 11 of
  `test_quadsource_policy_key` and 4 of `test_integral_read_batch`, which the new `build()`,
  `read_batch()` and constructor make follow the key; 7 of `test_store_inventory` (the key fields,
  the two `NON_IDENTITY` sub-tests, the two `TestTags` tests, `test_across_shards` and the
  missing-key-column test); and the guard's `test_analyser_coverage_is_undiminished`. Every one is
  on §0.5's list or is the pin §0.4 grants. Of ComputeTargets, `test_main_plumbing` and 08's, 09a's
  and 09b's `main.py` modules were run then: they failed only `EXPECTED_SITES`'s counter and
  `test_quadsource_policy_main`'s `test_it_is_carried`, both granted; 09a's and 09b's passed.
- **The six suites**, README §7's command, serially, on the final staged tree:

```
AdaptiveLevin: Ran 32 tests in 0.312s OK
CosmologyModels: Ran 39 tests in 3.095s OK
LiouvilleGreen: Ran 148 tests in 61.390s OK (skipped=1)
RunRegistry: Ran 199 tests in 90.311s OK
Datastore: Ran 532 tests in 185.401s OK
ComputeTargets: Ran 665 tests in 201.291s OK
```

- **The new modules, `-v`, unfiltered** (57 tests; the `!!` and `|` lines are the factories' and the
  Tk target's own prints):

```
test_a_table_from_before_prompt_08_keeps_its_message (Datastore.tests.test_quadsource_integral_parent_key.TestAnOldStoreIsRefused.test_a_table_from_before_prompt_08_keeps_its_message)
The user's decision (2026-09-30): read_batch selects quad_policy_serial before the four ... ok
test_a_table_from_before_this_prompt (Datastore.tests.test_quadsource_integral_parent_key.TestAnOldStoreIsRefused.test_a_table_from_before_this_prompt) ... ok
test_another_database_error_is_not_turned_into_it (Datastore.tests.test_quadsource_integral_parent_key.TestAnOldStoreIsRefused.test_another_database_error_is_not_turned_into_it) ... ok
test_a_parent_is_an_integer_serial (Datastore.tests.test_quadsource_integral_parent_key.TestKeyedOnEachOfItsSixParents.test_a_parent_is_an_integer_serial) ... ok
test_build_names_all_six (Datastore.tests.test_quadsource_integral_parent_key.TestKeyedOnEachOfItsSixParents.test_build_names_all_six) ... ok
test_changing_any_one_misses (Datastore.tests.test_quadsource_integral_parent_key.TestKeyedOnEachOfItsSixParents.test_changing_any_one_misses) ... ok
test_found_by_the_rows_it_was_computed_from (Datastore.tests.test_quadsource_integral_parent_key.TestKeyedOnEachOfItsSixParents.test_found_by_the_rows_it_was_computed_from) ... ok
test_otherwise_identical_rows_under_another_parent_coexist (Datastore.tests.test_quadsource_integral_parent_key.TestKeyedOnEachOfItsSixParents.test_otherwise_identical_rows_under_another_parent_coexist) ... ok
test_store_refuses_an_object_that_does_not_name_them (Datastore.tests.test_quadsource_integral_parent_key.TestKeyedOnEachOfItsSixParents.test_store_refuses_an_object_that_does_not_name_them) ... ok
test_the_stored_row_records_the_six (Datastore.tests.test_quadsource_integral_parent_key.TestKeyedOnEachOfItsSixParents.test_the_stored_row_records_the_six) ... ok
test_the_tags_are_still_part_of_the_key (Datastore.tests.test_quadsource_integral_parent_key.TestKeyedOnEachOfItsSixParents.test_the_tags_are_still_part_of_the_key) ... ok
test_a_parent_given_is_an_integer_serial (Datastore.tests.test_quadsource_integral_parent_key.TestReadBatchHandsBackTheParents.test_a_parent_given_is_an_integer_serial) ... ok
test_each_parent_given_filters_the_rows (Datastore.tests.test_quadsource_integral_parent_key.TestReadBatchHandsBackTheParents.test_each_parent_given_filters_the_rows) ... ok
test_tagged (Datastore.tests.test_quadsource_integral_parent_key.TestReadBatchHandsBackTheParents.test_tagged) ... ok
test_the_labels_alone_select_every_match (Datastore.tests.test_quadsource_integral_parent_key.TestReadBatchHandsBackTheParents.test_the_labels_alone_select_every_match)
D1: a reader holding only physical labels is given every stored row, each with the ... ok
test_untagged (Datastore.tests.test_quadsource_integral_parent_key.TestReadBatchHandsBackTheParents.test_untagged) ... ok
test_a_tk_parent_on_no_shard_is_a_named_unresolved_parent (Datastore.tests.test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_a_tk_parent_on_no_shard_is_a_named_unresolved_parent) ... ok
test_a_tq_parent_on_another_shard_resolves_to_that_rows_identity (Datastore.tests.test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_a_tq_parent_on_another_shard_resolves_to_that_rows_identity) ... ok
test_data_resolves_on_the_integrals_own_shard_only (Datastore.tests.test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_data_resolves_on_the_integrals_own_shard_only) ... ok
test_the_fixture_has_every_parent_resolved (Datastore.tests.test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_the_fixture_has_every_parent_resolved) ... ok
test_the_same_parents_are_a_duplicate (Datastore.tests.test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_the_same_parents_are_a_duplicate) ... ok
test_two_rows_differing_only_in_one_cross_shard_parent (Datastore.tests.test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_two_rows_differing_only_in_one_cross_shard_parent) ... ok
test_the_declarations (Datastore.tests.test_quadsource_integral_parent_key.TestTheNewColumns.test_the_declarations) ... ok
test_no_match_is_a_miss (Datastore.tests.test_quadsource_integral_parent_key.TestTheQuadSourceValueModelRead.test_no_match_is_a_miss) ... ok
test_one_match_serves_its_value (Datastore.tests.test_quadsource_integral_parent_key.TestTheQuadSourceValueModelRead.test_one_match_serves_its_value) ... ok
test_two_matches_are_refused_even_with_one_value (Datastore.tests.test_quadsource_integral_parent_key.TestTheQuadSourceValueModelRead.test_two_matches_are_refused_even_with_one_value)
The refusal is of the sources, not of the joined rows: it does not depend on which of ... ok
test_two_matches_are_refused_naming_their_parents (Datastore.tests.test_quadsource_integral_parent_key.TestTheQuadSourceValueModelRead.test_two_matches_are_refused_naming_their_parents) ... ok
test_the_existence_query_is_keyed_on_them (ComputeTargets.tests.test_quadsource_integral_parent_main.TestMainPy.test_the_existence_query_is_keyed_on_them) ... ok
test_the_helper_resolves_each_parent_serial_only_by_its_own_parents (ComputeTargets.tests.test_quadsource_integral_parent_main.TestMainPy.test_the_helper_resolves_each_parent_serial_only_by_its_own_parents) ... ok
test_the_object_to_be_computed_is_keyed_on_the_same_parents (ComputeTargets.tests.test_quadsource_integral_parent_main.TestMainPy.test_the_object_to_be_computed_is_keyed_on_the_same_parents) ... ok
test_the_stage_does_not_ask_for_a_reader_mode (ComputeTargets.tests.test_quadsource_integral_parent_main.TestMainPy.test_the_stage_does_not_ask_for_a_reader_mode) ... ok
test_the_stage_resolves_the_parents_before_its_existence_query (ComputeTargets.tests.test_quadsource_integral_parent_main.TestMainPy.test_the_stage_resolves_the_parents_before_its_existence_query) ... ok
test_the_work_list_lines_are_unchanged (ComputeTargets.tests.test_quadsource_integral_parent_main.TestMainPy.test_the_work_list_lines_are_unchanged) ... ok
test_a_tq_numeric_row_that_is_not_its_sources_tq_serial_raises (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheAssertion.test_a_tq_numeric_row_that_is_not_its_sources_tq_serial_raises) ... ok
test_a_tq_wkb_row_that_does_not_start_from_the_tq_numeric_row_raises (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheAssertion.test_a_tq_wkb_row_that_does_not_start_from_the_tq_numeric_row_raises) ... ok
test_a_tr_numeric_row_that_is_not_its_sources_tr_serial_raises (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheAssertion.test_a_tr_numeric_row_that_is_not_its_sources_tr_serial_raises) ... ok
test_a_tr_wkb_row_that_does_not_start_from_the_tr_numeric_row_raises (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheAssertion.test_a_tr_wkb_row_that_does_not_start_from_the_tr_numeric_row_raises) ... ok
test_another_row_for_any_parent_raises_naming_both (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheAssertion.test_another_row_for_any_parent_raises_naming_both) ... ok
test_nothing_is_computed_when_it_raises (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheAssertion.test_nothing_is_computed_when_it_raises) ... ok
test_the_checks_that_were_there_still_run_first (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheAssertion.test_the_checks_that_were_there_still_run_first) ... ok
test_the_diagonal_hands_one_row_twice (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheAssertion.test_the_diagonal_hands_one_row_twice)
q == r: the Tk rows for q and r are one pair of rows, handed twice. ... ok
test_the_rows_it_is_keyed_on_are_accepted (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheAssertion.test_the_rows_it_is_keyed_on_are_accepted) ... ok
test_a_stored_object_takes_them_from_its_row (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheConstructor.test_a_stored_object_takes_them_from_its_row) ... ok
test_an_object_to_be_computed_names_all_six (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheConstructor.test_an_object_to_be_computed_names_all_six) ... ok
test_it_carries_them_from_construction (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheConstructor.test_it_carries_them_from_construction) ... ok
test_store_does_not_replace_them (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheConstructor.test_store_does_not_replace_them)
The target's store() used to overwrite the source and data serials with what the ... ok
test_every_differing_parent_is_named (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheExtractionScript.test_every_differing_parent_is_named) ... ok
test_no_rows_is_nothing_to_refuse (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheExtractionScript.test_no_rows_is_nothing_to_refuse) ... ok
test_one_row_per_z_response_is_used_as_it_is (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheExtractionScript.test_one_row_per_z_response_is_used_as_it_is) ... ok
test_the_plot_refuses_before_it_uses_the_rows (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheExtractionScript.test_the_plot_refuses_before_it_uses_the_rows) ... ok
test_the_read_names_no_parent (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheExtractionScript.test_the_read_names_no_parent) ... ok
test_two_rows_at_one_z_response_are_refused_naming_what_distinguishes_them (ComputeTargets.tests.test_quadsource_integral_parent_main.TestTheExtractionScript.test_two_rows_at_one_z_response_are_refused_naming_what_distinguishes_them) ... ok
test_an_item_with_a_parent_missing_has_none (ComputeTargets.tests.test_quadsource_integral_parent_main.TestThePipelinesOwnLogic.test_an_item_with_a_parent_missing_has_none) ... ok
test_each_mode_and_pair_is_looked_up_once (ComputeTargets.tests.test_quadsource_integral_parent_main.TestThePipelinesOwnLogic.test_each_mode_and_pair_is_looked_up_once) ... ok
test_nothing_wanted_reads_nothing (ComputeTargets.tests.test_quadsource_integral_parent_main.TestThePipelinesOwnLogic.test_nothing_wanted_reads_nothing) ... ok
test_the_rows_found_satisfy_the_relations_compute_asserts (ComputeTargets.tests.test_quadsource_integral_parent_main.TestThePipelinesOwnLogic.test_the_rows_found_satisfy_the_relations_compute_asserts) ... ok
test_the_six_are_resolved_serial_only (ComputeTargets.tests.test_quadsource_integral_parent_main.TestThePipelinesOwnLogic.test_the_six_are_resolved_serial_only) ... ok

----------------------------------------------------------------------
Ran 57 tests in 3.476s

OK
!! QuadSourceValue.build(): multiple results found when querying for QuadSourceValue
!! QuadSourceValue.build(): multiple results found when querying for QuadSourceValue
!! Warning (TkNumericIntegration) k=1e+05/Mpc, log10_atol=-10.0, log10_rtol=-9.0
|    Initial redshift z_init=5e+09 is later than the 3-efold superhorizon time z_e3=1.2346e+12.
|    Setting initial conditions at this time may lead to meaningless results, because the initial values T_k(z) = 1, Tprime_k(z) = 0.
|    used for the matter transfer function integration apply only on sufficiently superhorizon scales.
!! Warning (TkNumericIntegration) k=1e+05/Mpc, log10_atol=-10.0, log10_rtol=-9.0
|    Initial redshift z_init=5e+09 is later than the 3-efold superhorizon time z_e3=2.4691e+12.
|    Setting initial conditions at this time may lead to meaningless results, because the initial values T_k(z) = 1, Tprime_k(z) = 0.
|    used for the matter transfer function integration apply only on sufficiently superhorizon scales.
!! Warning (TkNumericIntegration) k=1e+05/Mpc, log10_atol=-10.0, log10_rtol=-9.0
|    Initial redshift z_init=5e+09 is later than the 3-efold superhorizon time z_e3=1.2346e+12.
|    Setting initial conditions at this time may lead to meaningless results, because the initial values T_k(z) = 1, Tprime_k(z) = 0.
|    used for the matter transfer function integration apply only on sufficiently superhorizon scales.
!! Warning (TkNumericIntegration) k=1e+05/Mpc, log10_atol=-10.0, log10_rtol=-9.0
|    Initial redshift z_init=5e+09 is later than the 3-efold superhorizon time z_e3=2.4691e+12.
|    Setting initial conditions at this time may lead to meaningless results, because the initial values T_k(z) = 1, Tprime_k(z) = 0.
|    used for the matter transfer function integration apply only on sufficiently superhorizon scales.
!! Warning (TkNumericIntegration) k=1e+05/Mpc, log10_atol=-10.0, log10_rtol=-9.0
|    Initial redshift z_init=5e+09 is later than the 3-efold superhorizon time z_e3=1.2346e+12.
|    Setting initial conditions at this time may lead to meaningless results, because the initial values T_k(z) = 1, Tprime_k(z) = 0.
|    used for the matter transfer function integration apply only on sufficiently superhorizon scales.
!! Warning (TkNumericIntegration) k=1e+05/Mpc, log10_atol=-10.0, log10_rtol=-9.0
|    Initial redshift z_init=5e+09 is later than the 3-efold superhorizon time z_e3=2.4691e+12.
|    Setting initial conditions at this time may lead to meaningless results, because the initial values T_k(z) = 1, Tprime_k(z) = 0.
|    used for the matter transfer function integration apply only on sufficiently superhorizon scales.
!! Warning (TkNumericIntegration) k=1e+05/Mpc, log10_atol=-10.0, log10_rtol=-9.0
|    Initial redshift z_init=5e+09 is later than the 3-efold superhorizon time z_e3=1.2346e+12.
|    Setting initial conditions at this time may lead to meaningless results, because the initial values T_k(z) = 1, Tprime_k(z) = 0.
|    used for the matter transfer function integration apply only on sufficiently superhorizon scales.
!! Warning (TkNumericIntegration) k=1e+05/Mpc, log10_atol=-10.0, log10_rtol=-9.0
|    Initial redshift z_init=5e+09 is later than the 3-efold superhorizon time z_e3=2.4691e+12.
|    Setting initial conditions at this time may lead to meaningless results, because the initial values T_k(z) = 1, Tprime_k(z) = 0.
|    used for the matter transfer function integration apply only on sufficiently superhorizon scales.
!! Warning (TkNumericIntegration) k=1e+05/Mpc, log10_atol=-10.0, log10_rtol=-9.0
|    Initial redshift z_init=5e+09 is later than the 3-efold superhorizon time z_e3=1.2346e+12.
|    Setting initial conditions at this time may lead to meaningless results, because the initial values T_k(z) = 1, Tprime_k(z) = 0.
|    used for the matter transfer function integration apply only on sufficiently superhorizon scales.
!! Warning (TkNumericIntegration) k=1e+05/Mpc, log10_atol=-10.0, log10_rtol=-9.0
|    Initial redshift z_init=5e+09 is later than the 3-efold superhorizon time z_e3=2.4691e+12.
|    Setting initial conditions at this time may lead to meaningless results, because the initial values T_k(z) = 1, Tprime_k(z) = 0.
|    used for the matter transfer function integration apply only on sufficiently superhorizon scales.
```

- **Formatting.** `./venv/bin/python -m black --check` on the 17 Python files this commit touches:
  "18 files would be left unchanged".
- **Scope.** Staging by explicit path only; `git diff --cached --name-only` is in the final report.
- **No Ray, nothing under `var/`.** No module starts Ray, and each new one asserts it is not
  initialised at teardown. Stores are in memory or in temporary directories; the probes, the cost
  measurement, the golden capture and the mutation runs ran under the scratch directory.

## The deliberate-breakage record

Each mutation was applied with `git apply` in a `git archive` export of the staged tree
(`git write-tree`, then `git archive`) made into its own git repository, so that `git diff` there
showed the mutation alone and the tests that call `git` pass; the `Datastore`, `ComputeTargets` and
`RunRegistry` suites were run serially; and it was reverted with `git apply -R`, after which
`git diff --name-only -- '*.py'` was empty (`run_mutations.sh`, in the scratch directory).
AdaptiveLevin, CosmologyModels and LiouvilleGreen were not run: none imports a mutated module. Each
failing entry is one `FAIL:` or `ERROR:` line; a sub-test is its own entry, with its parameters.
The diffs are pasted from `git diff`, exactly as applied. Mutation (iv) is done twice, for a `Tr`
parent and for a `Tq` parent (§0.8: the fixture's `Tq` rows are all on their integral's shard, so
only a test that varies a `Tq` parent to another shard's row catches it). **In `ComputeTargets`,
`test_wall_time_per_object` is the known flake; it failed in the runs of (i)–(iv) under a load
average of 30–70 from other work, and is never a mutation's.** The export has no `var/`, so one
test in each of ComputeTargets and RunRegistry skips there that runs in the checkout (the
hand-written A3 manifest's), which is the `skipped=1`.

**Replays.** Before the commit, each of the six blocks below was extracted from this file and
applied with plain `git apply`, and the three suites were run (the result is in the final report,
since this file cannot record a run that follows itself). After the commit they are replayed from
`git show HEAD:prompts/datastore-integrity/logs/09c-key-quadsourceintegral-on-its-inputs.md`; that
result goes to the orchestrator.

### Mutation (i) — `Tr_WKB_serial` dropped from `build()`'s filter (the charter's (ii))

```diff
diff --git a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
index 99f64a7..d10678a 100644
--- a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
+++ b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
@@ -425,7 +425,6 @@ class sqla_QuadSourceIntegral_factory(SQLAFactoryBase):
             table.c.Tq_numeric_serial == parents["Tq_numeric_serial"],
             table.c.Tq_WKB_serial == parents["Tq_WKB_serial"],
             table.c.Tr_numeric_serial == parents["Tr_numeric_serial"],
-            table.c.Tr_WKB_serial == parents["Tr_WKB_serial"],
         )
 
         # require that the queried calculation has any specified tags
```

`Datastore`: `Ran 532 tests in 427.106s`, `FAILED (failures=1, errors=1)`; 2 failing entries, 2 of them sub-tests:

- `FAIL` `test_quadsource_integral_parent_key.TestKeyedOnEachOfItsSixParents.test_changing_any_one_misses` — sub-test `parent='Tr_WKB_serial'`
- `ERROR` `test_quadsource_integral_parent_key.TestKeyedOnEachOfItsSixParents.test_otherwise_identical_rows_under_another_parent_coexist` — sub-test `parent='Tr_WKB_serial'`

`ComputeTargets`: `Ran 665 tests in 281.205s`, `FAILED (failures=1, skipped=1)`; 1 failing entries, 0 of them sub-tests (one is the known flake `test_wall_time_per_object`, which fails under this load and is not the mutation's):

- `FAIL` `test_tk_wkb_phase.TestCost.test_wall_time_per_object` — the test itself (no sub-tests)

`RunRegistry`: `Ran 199 tests in 133.536s`, `OK (skipped=1)`.


### Mutation (ii) — the `Tq`-numeric/source assertion removed (the charter's (iii))

```diff
diff --git a/ComputeTargets/QuadSourceIntegral.py b/ComputeTargets/QuadSourceIntegral.py
index 1242c74..a99ba6f 100644
--- a/ComputeTargets/QuadSourceIntegral.py
+++ b/ComputeTargets/QuadSourceIntegral.py
@@ -2234,7 +2234,6 @@ class QuadSourceIntegral(DatastoreObject):
             )
 
         for label, numeric, source_serial in (
-            ("Tq", Tq_numeric, source.Tq_serial),
             ("Tr", Tr_numeric, source.Tr_serial),
         ):
             if numeric.store_id != source_serial:
```

`Datastore`: `Ran 532 tests in 361.921s`, `OK`.

`ComputeTargets`: `Ran 665 tests in 230.481s`, `FAILED (failures=3, skipped=1)`; 3 failing entries, 0 of them sub-tests (one is the known flake `test_wall_time_per_object`, which fails under this load and is not the mutation's):

- `FAIL` `test_quadsource_integral_parent_main.TestTheAssertion.test_a_tq_numeric_row_that_is_not_its_sources_tq_serial_raises` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_integral_parent_main.TestTheAssertion.test_nothing_is_computed_when_it_raises` — the test itself (no sub-tests)
- `FAIL` `test_tk_wkb_phase.TestCost.test_wall_time_per_object` — the test itself (no sub-tests)

`RunRegistry`: `Ran 199 tests in 98.585s`, `OK (skipped=1)`.


### Mutation (iii) — the reader takes the first of two matches (the charter's (vi))

```diff
diff --git a/extract_QuadSourceIntegral_data.py b/extract_QuadSourceIntegral_data.py
index 1a85f3c..f7fbed1 100644
--- a/extract_QuadSourceIntegral_data.py
+++ b/extract_QuadSourceIntegral_data.py
@@ -154,8 +154,10 @@ def refuse_ambiguous_QuadSourceIntegrals(integral_data):
         by_z_response.setdefault(obj.z_response.store_id, []).append(obj)
 
     for matches in by_z_response.values():
-        if len(matches) < 2:
-            continue
+        integral_data = [
+            obj for obj in integral_data if obj is matches[0] or obj not in matches
+        ]
+        continue
 
         first = matches[0]
         differing = [
```

`Datastore`: `Ran 532 tests in 347.593s`, `OK`.

`ComputeTargets`: `Ran 665 tests in 525.063s`, `FAILED (failures=4, skipped=1)`; 4 failing entries, 0 of them sub-tests (one is the known flake `test_wall_time_per_object`, which fails under this load and is not the mutation's):

- `FAIL` `test_quadsource_integral_parent_main.TestTheExtractionScript.test_every_differing_parent_is_named` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_integral_parent_main.TestTheExtractionScript.test_one_row_per_z_response_is_used_as_it_is` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_integral_parent_main.TestTheExtractionScript.test_two_rows_at_one_z_response_are_refused_naming_what_distinguishes_them` — the test itself (no sub-tests)
- `FAIL` `test_tk_wkb_phase.TestCost.test_wall_time_per_object` — the test itself (no sub-tests)

`RunRegistry`: `Ran 199 tests in 139.006s`, `OK (skipped=1)`.


### Mutation (iv), for a `Tr` parent — `Tr_WKB` declared a same-shard `Parent`

```diff
diff --git a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
index 99f64a7..6f26311 100644
--- a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
+++ b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
@@ -1069,7 +1069,7 @@ class sqla_QuadSourceIntegral_factory(SQLAFactoryBase):
                 "Tr_numeric": Parent(
                     "Tr_numeric_serial", "TkNumericIntegration", cross_shard=True
                 ),
-                "Tr_WKB": Parent("Tr_WKB_serial", "TkWKBIntegration", cross_shard=True),
+                "Tr_WKB": Parent("Tr_WKB_serial", "TkWKBIntegration"),
             },
             tags=("QuadSourceIntegral_tags", "parent_serial"),
         )
```

`Datastore`: `Ran 532 tests in 292.608s`, `FAILED (failures=60, errors=2)`; 62 failing entries, 27 of them sub-tests:

- `FAIL` `test_inventory_consumers.TestMainInventoryIsReadOnly.test_it_prints_the_report_and_writes_nothing` — the test itself (no sub-tests)
- `FAIL` `test_inventory_report.TestTheDisplaySaysWhatTheInventoryHolds.test_the_title_and_the_problem_count` — the test itself (no sub-tests)
- `FAIL` `test_inventory_report.TestTheDisplaySaysWhatTheInventoryHolds.test_the_value_tables_named_are_the_ones_counted` — sub-test `cls='QuadSourceIntegral'`
- `ERROR` `test_inventory_report.TestTheDisplaySaysWhatTheInventoryHolds.test_verbose_shows_every_key_field_of_every_class` — sub-test `cls='QuadSourceIntegral'`
- `FAIL` `test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_a_tk_parent_on_no_shard_is_a_named_unresolved_parent` — the test itself (no sub-tests)
- `ERROR` `test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_a_tq_parent_on_another_shard_resolves_to_that_rows_identity` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_data_resolves_on_the_integrals_own_shard_only` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_the_fixture_has_every_parent_resolved` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_the_same_parents_are_a_duplicate` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_two_rows_differing_only_in_one_cross_shard_parent` — sub-test `field='Tq_WKB'`
- `FAIL` `test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_two_rows_differing_only_in_one_cross_shard_parent` — sub-test `field='Tq_numeric'`
- `FAIL` `test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_two_rows_differing_only_in_one_cross_shard_parent` — sub-test `field='Tr_WKB'`
- `FAIL` `test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_two_rows_differing_only_in_one_cross_shard_parent` — sub-test `field='Tr_numeric'`
- `FAIL` `test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_two_rows_differing_only_in_one_cross_shard_parent` — sub-test `field='source'`
- `FAIL` `test_quadsource_parent_key.TestTheInventoryResolvesAcrossShards.test_the_fixture_source_resolves_its_cross_shard_tr` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_policy_key.TestTheInventoryIdentityIncludesIt.test_the_fingerprint_moves_with_the_policy` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_policy_key.TestTheInventoryIdentityIncludesIt.test_the_same_policy_would_be_a_duplicate` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_policy_key.TestTheInventoryIdentityIncludesIt.test_two_otherwise_identical_rows_under_two_policies_are_two_records` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestDuplicates.test_across_shards` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestDuplicates.test_on_one_shard` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestDuplicates.test_other_tags_are_not_a_duplicate` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_each_identity_column_changes_the_records` — sub-test `cls='QuadSourceIntegral', field='Tq_WKB'`
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_each_identity_column_changes_the_records` — sub-test `cls='QuadSourceIntegral', field='Tq_numeric'`
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_each_identity_column_changes_the_records` — sub-test `cls='QuadSourceIntegral', field='Tr_numeric'`
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_each_identity_column_changes_the_records` — sub-test `cls='QuadSourceIntegral', field='atol'`
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_each_identity_column_changes_the_records` — sub-test `cls='QuadSourceIntegral', field='data'`
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_each_identity_column_changes_the_records` — sub-test `cls='QuadSourceIntegral', field='k'`
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_each_identity_column_changes_the_records` — sub-test `cls='QuadSourceIntegral', field='model'`
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_each_identity_column_changes_the_records` — sub-test `cls='QuadSourceIntegral', field='policy'`
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_each_identity_column_changes_the_records` — sub-test `cls='QuadSourceIntegral', field='q'`
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_each_identity_column_changes_the_records` — sub-test `cls='QuadSourceIntegral', field='quad_policy'`
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_each_identity_column_changes_the_records` — sub-test `cls='QuadSourceIntegral', field='r'`
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_each_identity_column_changes_the_records` — sub-test `cls='QuadSourceIntegral', field='rtol'`
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_each_identity_column_changes_the_records` — sub-test `cls='QuadSourceIntegral', field='source'`
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_each_identity_column_changes_the_records` — sub-test `cls='QuadSourceIntegral', field='z_response'`
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_each_identity_column_changes_the_records` — sub-test `cls='QuadSourceIntegral', field='z_source_max'`
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_the_key_fields_are_the_lookup_columns` — sub-test `cls='QuadSourceIntegral'`
- `FAIL` `test_store_inventory.TestKeysArePhysical.test_different_serial_assignments` — sub-test `cls='QuadSourceIntegral'`
- `FAIL` `test_store_inventory.TestKeysArePhysical.test_replicated_serials_permuted_and_rows_moved` — sub-test `cls='QuadSourceIntegral'`
- `FAIL` `test_store_inventory.TestKeysArePhysical.test_sharded_rows_on_different_shards` — sub-test `cls='QuadSourceIntegral'`
- `FAIL` `test_store_inventory.TestNonIdentityDoesNotMatter.test_timestamps` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestOldStoresAndOrphans.test_a_missing_column_the_key_does_not_need` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestOldStoresAndOrphans.test_a_missing_key_column_makes_the_class_incomplete` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestOldStoresAndOrphans.test_a_reference_that_cannot_be_resolved` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestOldStoresAndOrphans.test_a_tag_row_whose_parent_is_missing` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestOldStoresAndOrphans.test_a_tag_row_whose_tag_is_missing` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestOldStoresAndOrphans.test_an_absent_association_table` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestOldStoresAndOrphans.test_an_absent_replicated_association_table` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestOldStoresAndOrphans.test_an_absent_replicated_table` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestOldStoresAndOrphans.test_an_absent_table` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestOldStoresAndOrphans.test_an_absent_value_table` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestOldStoresAndOrphans.test_an_orphan_value_row` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestReadOnlyAndNoRay.test_no_ray` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestReadOnlyAndNoRay.test_read_inventory_writes_nothing` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestReplicatedDivergence.test_a_changed_row` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestReplicatedDivergence.test_a_changed_value_count` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestTags.test_a_tag_on_the_background_model_reaches_its_children` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestTags.test_a_tagged_parents_digest_covers_its_tags` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestTags.test_adding_a_tag_changes_that_record_and_nothing_else` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestTags.test_oneloop_tags_come_from_its_own_table` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestTheFullStore.test_every_class_has_a_record` — sub-test `cls='QuadSourceIntegral'`
- `FAIL` `test_store_inventory.TestTheFullStore.test_the_full_store_has_no_problem` — the test itself (no sub-tests)

`ComputeTargets`: `Ran 665 tests in 321.752s`, `FAILED (failures=1, skipped=1)`; 1 failing entries, 0 of them sub-tests (one is the known flake `test_wall_time_per_object`, which fails under this load and is not the mutation's):

- `FAIL` `test_tk_wkb_phase.TestCost.test_wall_time_per_object` — the test itself (no sub-tests)

`RunRegistry`: `Ran 199 tests in 107.935s`, `FAILED (failures=9, skipped=1)`; 9 failing entries, 0 of them sub-tests:

- `FAIL` `test_store_fingerprint.TestFingerprintCommand.test_a_difference_exits_one` — the test itself (no sub-tests)
- `FAIL` `test_store_fingerprint.TestFingerprintCommand.test_read_only_then_write_then_matches` — the test itself (no sub-tests)
- `FAIL` `test_store_fingerprint.TestGolden.test_the_golden_fingerprint` — the test itself (no sub-tests)
- `FAIL` `test_store_fingerprint.TestGolden.test_the_shape` — the test itself (no sub-tests)
- `FAIL` `test_store_fingerprint.TestLocalised.test_adding_a_tag_to_one_record` — the test itself (no sub-tests)
- `FAIL` `test_store_fingerprint.TestLocalised.test_deleting_one_quadsourceintegral_row` — the test itself (no sub-tests)
- `FAIL` `test_store_fingerprint.TestLocalised.test_deleting_the_row_alone_is_also_a_problem_entry` — the test itself (no sub-tests)
- `FAIL` `test_store_fingerprint.TestNotContent.test_timestamps_labels_and_solver_serials` — the test itself (no sub-tests)
- `FAIL` `test_store_retire.TestRefusals.test_a_row_changed_after_fingerprinting` — the test itself (no sub-tests)


### Mutation (iv), for a `Tq` parent — `Tq_numeric` declared a same-shard `Parent`

```diff
diff --git a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
index 99f64a7..4edd6bd 100644
--- a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
+++ b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
@@ -1062,9 +1062,7 @@ class sqla_QuadSourceIntegral_factory(SQLAFactoryBase):
                 "rtol": Parent("rtol_serial", "tolerance"),
                 "source": Parent("source_serial", "QuadSource", cross_shard=True),
                 "data": Parent("data_serial", "GkSourcePolicyData"),
-                "Tq_numeric": Parent(
-                    "Tq_numeric_serial", "TkNumericIntegration", cross_shard=True
-                ),
+                "Tq_numeric": Parent("Tq_numeric_serial", "TkNumericIntegration"),
                 "Tq_WKB": Parent("Tq_WKB_serial", "TkWKBIntegration", cross_shard=True),
                 "Tr_numeric": Parent(
                     "Tr_numeric_serial", "TkNumericIntegration", cross_shard=True
```

`Datastore`: `Ran 532 tests in 215.130s`, `FAILED (failures=2, errors=1)`; 3 failing entries, 1 of them sub-tests:

- `ERROR` `test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_a_tq_parent_on_another_shard_resolves_to_that_rows_identity` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_integral_parent_key.TestTheInventoryResolvesAllSix.test_two_rows_differing_only_in_one_cross_shard_parent` — sub-test `field='Tq_numeric'`
- `FAIL` `test_store_inventory.TestDuplicates.test_across_shards` — the test itself (no sub-tests)

`ComputeTargets`: `Ran 665 tests in 283.521s`, `FAILED (failures=1, skipped=1)`; 1 failing entries, 0 of them sub-tests (one is the known flake `test_wall_time_per_object`, which fails under this load and is not the mutation's):

- `FAIL` `test_tk_wkb_phase.TestCost.test_wall_time_per_object` — the test itself (no sub-tests)

`RunRegistry`: `Ran 199 tests in 85.922s`, `OK (skipped=1)`.


### Mutation (v) — `source_serial` dropped from `read_batch`'s filter

```diff
diff --git a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
index 99f64a7..b1e48d7 100644
--- a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
+++ b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
@@ -865,8 +865,6 @@ class sqla_QuadSourceIntegral_factory(SQLAFactoryBase):
                 table.c.z_source_max_serial == z_source_max.store_id,
             )
 
-        if "source_serial" in parents:
-            query = query.filter(table.c.source_serial == parents["source_serial"])
         if "data_serial" in parents:
             query = query.filter(table.c.data_serial == parents["data_serial"])
         if "Tq_numeric_serial" in parents:
```

`Datastore`: `Ran 532 tests in 177.688s`, `FAILED (errors=1)`; 1 failing entries, 1 of them sub-tests:

- `ERROR` `test_quadsource_integral_parent_key.TestReadBatchHandsBackTheParents.test_each_parent_given_filters_the_rows` — sub-test `parent='source_serial'`

`ComputeTargets`: `Ran 665 tests in 182.530s`, `OK (skipped=1)`.

`RunRegistry`: `Ran 199 tests in 64.961s`, `OK (skipped=1)`.

Mutation (i) is caught by the keyed-lookup sub-tests of the new module, and only by them: the
fixture-level tests never look an integral up. Mutation (ii) is caught by the assertion tests, and
by nothing in Datastore, which never calls `compute()`. Mutation (iii) is caught by the extraction
script's tests. Mutation (iv) for `Tr` is caught across the inventory, since the fixture's
integrals 1 and 2 have a cross-shard `Tr` WKB row, and by the golden; for `Tq` it is caught only by
the two tests that vary a `Tq` parent to another shard, and by the duplicate test whose copy on
shard 1 names shard-0 `Tq` rows, and not by the golden. Mutation (v) is caught by the reader's
filter sub-test for `source_serial` (the read returns two rows where one is unpacked).

## Observations not acted on

1. **The inventory report renders each integral's parents in full.** `tools/inventory_report.py`
   renders a parent reference by its physical labels, recursively, so a `QuadSourceIntegral` record
   now carries its `QuadSource` (with that source's `Tq` and `Tr`), its four Tk rows and its
   `GkSourcePolicyData` — and through it its `GkSource`'s whole parent set. On `build_full_store` an
   integral's line is about 4,300 characters (about 450 at `HEAD`). In production every one of
   ~500,000 integrals repeats its `GkSource`'s ~1,200 members. That is the defect of
   `[09b-inventory-report-renders-every-member-of-a-parent-set]`, whose impact now reaches every
   integral line as well as every `GkSourcePolicyData` line; not re-opened, and its board entry not
   edited (outside this prompt's board edits).
2. **`QuadSourceIntegral.source_serial` and `.data_serial` still raise "value has not yet been
   populated"** on an unpopulated object, though both are now known from construction; `parents`
   does not. Left as they were.
3. **`evaluate_QuadSource_integral` still returns `GkPolicy_serial` and `source_serial`**, which the
   target's `store()` no longer reads (Deviation 2). Outside the target's granted parts.
4. **A comment in 09b's `test_gksource_parent_set_main.py`** says its six functions are "the six
   test_main_plumbing.EXPECTED_SITES counts"; `EXPECTED_SITES["GkSource"]` is now 7, the seventh
   being the new helper's lookup, which that module's `GKSOURCE_LOOKUPS` does not list. The module
   passes, and is not granted. Residue; no issue. (`TestMainPy` here checks that lookup is keyed on
   the set it is handed.)
5. **`build_QuadSourceIntegral_payload`'s messages say "GkSource for for"** (and "QuadSource for
   for"), pre-existing. Not in the brief's parts of that function.
6. **The helper puts each serial-only `GkSource` in the Ray object store**, through `GkSourceProxy`,
   to key its `GkSourcePolicyData` lookup, as `build_GkSourcePolicy_batch` already does: up to 50 small
   objects a batch.
7. **Nothing here has run under Ray or on a real store.** Prompt 10's end-to-end store is the first
   place the keyed stage, the reader's refusal and the fingerprint's six parents meet production
   data.

## State handed to the next prompt

- **Names.** The columns `Tq_numeric_serial`, `Tq_WKB_serial`, `Tr_numeric_serial`,
  `Tr_WKB_serial` (README §4); the same names, with `source_serial` and `data_serial`, as the payload
  keys of `QuadSourceIntegral`'s `build()` and `read_batch()` and the constructor's keywords;
  `QuadSourceIntegral.PARENTS`, `.parents`; the inventory key fields `source`, `data`,
  `Tq_numeric`, `Tq_WKB`, `Tr_numeric`, `Tr_WKB`; `main.py`'s `resolve_QuadSourceIntegral_parents`;
  `extract_QuadSourceIntegral_data.refuse_ambiguous_QuadSourceIntegrals`;
  `_refuse_a_store_that_predates_the_transfer_function_key`.
- **Prompt 09 is complete** (09a, 09b, 09c): every derived row's key reaches the rows it was
  computed from, transitively, and the inventory resolves them all, across shards and as sets.
- **For 10.** The throwaway store is the first to exercise all of this: the keyed stages, the
  readers' `labels_only` and this reader's refusal, and the fingerprint at format 6 (the golden is
  `full_store_fingerprint_datastore-integrity-09c.json`, the witness
  `schema_at_datastore-integrity-09c.json`). Its `QuadSourceIntegral` stage pays the lookups
  measured above on a resume.
- **The guard's pins:** `EXPECTED_COVERAGE` 51; `KNOWN_UNRESOLVED_TABLES` 11; `KNOWN_BLIND` 1,
  `KNOWN_UNCHECKED_INSERTS` 13 and `KNOWN_BLIND_COLUMNS` 4; `KNOWN_UNFIXED` and
  `KNOWN_UNFIXED_COLUMNS` empty. **`EXPECTED_SITES`:** `TkNumericIntegration` 7, `TkWKBIntegration`
  4, `QuadSource` 4, `GkSource` 7, `GkSourcePolicyData` 5, and the rest as in 09b.
- **Baselines** after this commit: AdaptiveLevin 32, ComputeTargets **665**, CosmologyModels 39,
  Datastore **532**, LiouvilleGreen 148 (1 skipped), RunRegistry 199.
