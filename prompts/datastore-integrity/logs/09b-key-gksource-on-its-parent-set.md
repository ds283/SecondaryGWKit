# Log 09b — Key GkSource on the set of Gk rows it is assembled from

**Prompt:** [`prompts/datastore-integrity/09b-key-gksource-on-its-parent-set.md`](../09b-key-gksource-on-its-parent-set.md)
**Commit:** *(this commit)* — "Key GkSource on the set of Gk rows it is assembled from"
**Model:** Claude Opus 5.5
**Date:** 2026-09-30
**Result:** COMPLETE. K6 and K7's set-valued half shipped, in the charter's recommended shape, under
D1, D4 and D5, the user's decisions of the orchestrator note's §0.1–§0.4, and the user's four
decisions on the sketch (§0.7). It closes `[07-wkb-value-reads-do-not-refuse-a-pre-07-store]` and
opens two issues. No stop condition triggered:
- the Gk rows are `GkSource`'s only direct inputs (`compute()` hands `assemble_GkSource_values` the
  `numeric` and `WKB` maps, and nothing else from the store), and in a store written by the
  pipeline's own logic in a test the set recorded is the set assembled
  (`TestThePipelinesOwnLogic`, below);
- `main.py` resolves the set before its existence query without changing what work it schedules,
  and all five `substitute_main_py()` literals still match once;
- the inventory resolves the set in its one read-only pass (the member table is read once per
  shard, as a tag table is);
- no existing test outside §0.5's "Granted for 09b" changed, and each granted one only follows the
  key;
- `EXPECTED_SITES` did not move, and the read-back guard's `EXPECTED_COVERAGE` grew by exactly this
  prompt's three new row-read sites (§0.4);
- only `GkSource` and `GkSourcePolicyData` moved in the fingerprint.

Suites after this commit, serially, on the final staged tree:
- ComputeTargets **610 → 635** (the 25 of the new `test_gksource_parent_set_main.py`). The known
  wall-clock flake did not fire in the final run.
- Datastore **459 → 505** (the 46 of the new `test_gksource_parent_set.py`);
- AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped) and RunRegistry 199,
  unchanged.

```
AdaptiveLevin: Ran 32 tests in 0.054s OK
CosmologyModels: Ran 39 tests in 0.701s OK
LiouvilleGreen: Ran 148 tests in 23.597s OK (skipped=1)
RunRegistry: Ran 199 tests in 67.130s OK
Datastore: Ran 505 tests in 131.856s OK
ComputeTargets: Ran 635 tests in 146.947s OK
```

The index `docs/OPEN_ISSUES.md` goes from **99 to 100** rows, counted
(`grep -c '^| \`\[' docs/OPEN_ISSUES.md`): one deleted (the 07 issue), two opened.

## What shipped

**K6 — the record and the key** (`Datastore/SQL/ObjectFactories/GkSource.py`).
- **`GkSource_parents`**, a new table (factory `sqla_GkSourceParents_factory`, registered in
  `Datastore.py`'s factory map): `serial`, `gk_source_serial` → `GkSource.serial`,
  `z_source_serial` → `redshift.serial`, `numeric_serial` → `GkNumericIntegration.serial`
  (nullable) and `wkb_serial` → `GkWKBIntegration.serial` (nullable), each an indexed foreign key.
  One row per `z_source`. It has no lookup of its own (`build()` raises `NotImplementedError`, as the
  tag factories' do).
- **`parents_digest`** on `GkSource`: `String(64)`, NOT NULL, indexed, after `z_response_serial`.
  `GkSource_parents_digest(members)` is SHA-256 (hex) of the compact JSON of the canonical set,
  `[[z_source, numeric|null, wkb|null], ...]` sorted by `z_source` serial. It is the one function
  that forms the digest; `build()` and `store()` both call it.
- **`build()`** takes `payload["parents"]` (the set) or `payload["labels_only"]`, never neither
  (`RuntimeError`) and never both (`ValueError`), as 07 and 09a do. A keyed lookup filters
  `parents_digest == GkSource_parents_digest(parents)`. The object it returns carries the set it
  was asked under (a hit or a miss).
- **The D1 reader mode, `labels_only=True`.** It reads the candidates' `(serial, parents_digest)` from
  the lookup's own query; more than one is `MultipleResultsFound` naming each serial and digest and
  the `z_source` serials at which their sets differ (read from `GkSource_parents`). One match is
  served, carrying the set its `GkSource_parents` rows record; none is a labels-only miss, which
  cannot be computed.
- **`store()`** writes the set **assembled** — `obj.assembled_parents`, what
  `assemble_GkSource_values` reports it read — never the set the lookup asked for. It builds the
  member rows first, computes `parents_digest` from exactly those rows, inserts the `GkSource` row
  with it and then one `GkSource_parents` row per member, in the one transaction. **An empty or
  missing set is refused** (`ValueError`) before anything is written (the user's decision 4).
- **`validate_on_startup`**'s prune deletes a pruned `GkSource`'s `GkSource_parents` rows with its
  values and tags (Deviation 5).
- **`GkSourceValue._build_impl_model`** (the user's decision 3) selects its candidate `GkSource`s'
  `parents_digest` too, refuses more than one naming their digests and differing `z_source`s, and
  serves one match as before.
- **Old stores (D4).** `_refuse_a_store_that_predates_the_parent_set` matches the driver's own
  message for `parents_digest` or `GkSource_parents` and raises 07's form of refusal; `build()`
  (keyed and reader mode) and `GkSourceValue._build_impl_model` call it. Any other database error is
  re-raised unchanged.
- **`inventory_records`** declares the set (below). Its comment says `parents_digest` is not read.

**K6 — the compute target** (`ComputeTargets/GkSource.py`).
- **One canonical form**, `GkSource_parent_set(members)`: a tuple of `(z_source serial, numeric
  serial | None, WKB serial | None)` sorted by `z_source`. A member naming neither parent, a
  `z_source` named twice or a non-integer serial (a `bool` included) is `ValueError`.
  `GkSource_parent_set_of_values(z_sample, numeric_data, WKB_data)` forms it from value objects,
  `GkSource_parent_set_of_serials(...)` from the serial-only reads' pairs, and
  `GkSource_parents_differ(a, b)` names the `z_source`s at which two differ. Each reads only the
  `z_source`s of the sample, as `assemble_GkSource_values` does.
- **The constructor** takes `parents` and `labels_only`. An object to be computed (`payload=None`,
  not labels-only) must name its set (`ValueError`); both together are `ValueError`. Properties
  `parents`, `assembled_parents` and `labels_only`.
- **`compute()` asserts** (Deviation 4) that the values it is handed were read from exactly the set
  it is keyed on, before anything is dispatched: a difference raises `RuntimeError` naming the
  `k` and `z_response` serials and the `z_source`s that differ. A labels-only object, or one with an
  empty set, is refused.
- **`assemble_GkSource_values`** records, for each `z_source` it assembles, the integration each of
  its two values was read from (`_parent_serial`, below), and returns that set as `"parents"`. The
  target's `store()` keeps it as `assembled_parents`, which the factory writes.

**The value reads hand back their parent** (`GkNumericIntegration.py`, `GkWKBIntegration.py`).
- `GkNumericValue.read_batch` and `GkWKBValue.read_batch` attach the parent serial they already
  select to each value, as `obj._parent_serial` (a factory-set attribute, like `_k_exit` and
  `_z_source`; the value classes are not in this prompt's files).
- **The serial-only form, `parents_only=True`**, is the prompt's choice of shape: the same parent
  query, filters and tag joins, and a `SELECT DISTINCT z_source_serial, <parent serial>` over the
  value rows at `z`, returning the sorted `(z_source_serial, parent_serial)` pairs and no value. The
  refusal of a second parent per `z_source` (05c) runs in both forms.

**`[07-wkb-value-reads-do-not-refuse-a-pre-07-store]`.** `GkWKBValue`'s and `TkWKBValue`'s
`_build_impl_model` and `read_batch` always select `numeric_serial` in their parent query, keyed or
not, and catch `SQLAlchemyError`, raising 07's D4 message when the driver's message names
`numeric_serial` (Deviation 13 for why the message is written in each of the four).

**`main.py`** (table below): one new nested helper, `resolve_GkSource_parents`, and all six
`GkSource` lookups keyed on what it returns.

**Readers (D1).** `extract_GkSource_data.py`, `extract_QuadSourceIntegral_data.py`,
`docs/source-remediation-verification/analyse_greens_and_source.py` and `run_quadsource_integrals.py`
pass `labels_only=True` to their one `GkSource` lookup each, with a comment.

**K7 — the set-valued parent** (`Datastore/store_inventory.py`).
- **`ParentSet(table, owner, members)`**, passed to `read_records` as `parent_sets={field: ...}`:
  `members` maps each member field to a `Parent`, resolved as a key parent is (within the shard, or
  on every shard for a `cross_shard` one).
- `read_records` reads the member table **once per shard**, grouping its rows by owner. For each
  row of the class the field's value is `digest({"fields": [...], "members": [...]})` of its member
  fields' names and its resolved members, sorted: each member is the tuple of its rows' reference
  digests (`None` for a nullable field that names none). It is formed from the parents' identities
  and never from a serial or `parents_digest`.
- **Named problems:** a member that cannot be resolved makes its row an `unresolved-parent` (field
  `parents`); an owner with no member rows is the new `empty-parent-set` (the user's decision 4); a
  member row whose owner is not on its shard is the new `orphan-member`; an absent member table is
  `absent-table`, and a missing member column `incomplete`, and then none of the class's records
  are read from that shard. None of these rows is a record.
- The members are kept beside the records, in `ClassInventory.parent_sets` (field → member field →
  class) and `ClassInventory.members` (field → set digest → members, as tuples; Deviation 10), and
  `StoreInventory.resolve` resolves them. Neither is in a `Record`, so neither is in a fingerprint
  except through the key's digest.
- `GkSource.inventory_records` declares `"parents": ParentSet("GkSource_parents",
  "gk_source_serial", {"z_source": Parent("z_source_serial", "redshift"), "numeric":
  Parent("numeric_serial", "GkNumericIntegration", nullable=True), "wkb": Parent("wkb_serial",
  "GkWKBIntegration", nullable=True)})`.

**`tools/inventory_report.py`** (the user's decision 2): a parent-set field is rendered as
`[{z_source=..., numeric=..., wkb=...}, ...]`, each member field by the report's existing parent
rendering or `none`, in physical order. The module docstring says so.
`test_inventory_report.py` is unmodified and passes, `test_no_digest_appears` included.

**`_drop_actions`.** `GkSource_parents` joins `"gk-source"`, last. `TestDropOrder` (07) passes
unmodified. The order within the action is `[07-drop-actions-list-a-parent-table-before-its-children]`,
not fixed here.

**`Datastore/SQL/ClientPool.py`**: `"GkSource_parents": 7500` in the serial batch-size map
(Deviation 2). `config/sharding.py` is unchanged.

**D5.** `Datastore/tests/data/schema_at_datastore-integrity-09b.json`, captured with
`PYTHONPATH=. ./venv/bin/python Datastore/tests/schema_description.py
Datastore/tests/data/schema_at_datastore-integrity-09b.json`; `RunRegistry/tests/data/
full_store_fingerprint_datastore-integrity-09b.json`; `FINGERPRINT_FORMAT` 4 → 5 with its "Format 5"
sentence. No older witness or golden is touched.

**Tests** (new):
- `Datastore/tests/test_gksource_parent_set.py`, 46 tests: §3 tests 1–7 and 9, the digest, and the
  user's decisions 3 and 4.
- `ComputeTargets/tests/test_gksource_parent_set_main.py`, 25 tests: §3 test 8, the readers, the
  constructor, the `compute()` assertion, and the pipeline's own logic end to end.
- Each module's `tearDownModule` asserts `ray.is_initialized()` is false.

## The parent table, and the design

Shards from `config/sharding.py` (`"GkSource": "k"`, `"GkNumericIntegration": "k"`,
`"GkWKBIntegration": "k"`; `redshift` is replicated).

| Parent of a `GkSource` | Recorded in | Its shard | Foreign key |
|---|---|---|---|
| per `z_source`: the `GkNumericIntegration` whose value at `z_response` is assembled, or none | `GkSource_parents.numeric_serial`, nullable | `k`'s, the source's own | yes, `GkNumericIntegration.serial`, indexed |
| per `z_source`: the `GkWKBIntegration` whose value at `z_response` is assembled, or none | `GkSource_parents.wkb_serial`, nullable | `k`'s | yes, `GkWKBIntegration.serial`, indexed |
| the member's `z_source` | `GkSource_parents.z_source_serial` | replicated | yes, `redshift.serial`, indexed |
| the owning `GkSource` | `GkSource_parents.gk_source_serial` | `k`'s | yes, `GkSource.serial`, indexed |
| the set, digested | `GkSource.parents_digest` | — | none (a digest, not a reference) |

**Which parents: the integration rows, not the value rows** (the prompt's recommendation). A value
row is identified by its integration row and its `z`, and `GkSource` already keys on `z_response`;
the value reads' refusal of a second parent per `z_source` (05c) makes the integration row
determine the value at `z_response`.

**The design and why.** The recommended shape is kept: a table that records the set and a digest
that keys on it in one comparison. What this prompt adds to it:
- **Two sets on the object, the key and the record.** `parents` is what the lookup was asked
  under; `assembled_parents` is what `assemble_GkSource_values` reports it assembled. The factory
  writes the second and digests it, so the row is the record of exactly what was assembled and is
  found by what it is. `compute()` asserts that the first is the set of the values it is handed, so
  in the pipeline the two are one set. This is what gives mutation (i), "the requested set instead
  of the rows written", a meaning.
- **One canonical form and one digest function** (in the target and the factory respectively),
  and in `main.py` one helper that every stage keys on, so no two places form a set differently.
- **The inventory's identity is the resolved set**, not `parents_digest` (§0.6: the inventory never
  uses a serial as identity).

## Every `main.py` site changed

Line numbers from `git show HEAD:main.py` (before) and `git show :main.py` of the staged tree
(after). `git diff --cached -U0 main.py` has seventeen hunks: `@@ -31,0 +32 @@`,
`@@ -2454,0 +2456,80 @@`, `@@ -2465,0 +2547,7 @@`, `@@ -2476,0 +2565,3 @@`, `@@ -2528 +2619 @@`,
`@@ -2532 +2623 @@`, `@@ -2662,0 +2754,5 @@`, `@@ -2804 +2900 @@`, `@@ -2812,0 +2909,4 @@`,
`@@ -2901,0 +3002,7 @@`, `@@ -2909,0 +3017,3 @@`, `@@ -3021 +3131 @@`, `@@ -3023 +3133 @@`,
`@@ -3027 +3137 @@`, `@@ -3035,0 +3146,4 @@`, `@@ -3295,0 +3410,7 @@`, `@@ -3303,0 +3425 @@`.

| Site | Before | After |
|---|---|---|
| Import of `GkSource_parent_set_of_serials` | — | `:32` |
| **The shared helper** `resolve_GkSource_parents(wanted)`, nested in `run_pipeline`: for each `(k_exit, z_response)`, `GkNumericValue` and `GkWKBValue` `object_read_batch` with `"parents_only": True` and `build_missing_GkSource`'s payload (the numeric tolerance pair, the eight tags), then the set over `z_source_sample.truncate(z_response, keep="higher")` | — | def `:2456`, `"parents_only"` `:2485`, the set `:2528` |
| GkSource stage, `build_GkSource_batch`: the set resolved for every `(k, z_response)` of the batch **before** the existence query | def `:2455` | def `:2536`, `resolve_GkSource_parents` `:2550` |
| its existence query, keyed on the set | payload `:2469`, `object_get_vectorized` `:2501` | payload `:2557`, `"parents"` `:2565`, call `:2592` |
| the call of `build_missing_GkSource`, handing it the set | `:2528` | `:2619` |
| `build_missing_GkSource`: gains the parameter; its construction passes `parents=` | def `:2532`, `"GkSource"` `:2658` | def `:2623`, `"GkSource"` `:2749`, `parents=` `:2756` |
| `build_incomplete_GkSourcePolicy`: gains the parameter; its lookup keyed | def `:2804`, lookup `:2805`, call `:2834` | def `:2900`, lookup `:2901`, `"parents"` `:2910`, call `:2934` |
| `build_GkSourcePolicy_batch`: resolves the set once, keys its lookup, and hands it to the two below | def `:2898`, lookup `:2902`, call `:2932` | def `:2998`, resolve `:3005`, lookup `:3009`, `"parents"` `:3017`, call `:3042` |
| its calls of the two below | `:3021`, `:3023` | `:3131`, `:3133` |
| `build_missing_GkSourcePolicy`: gains the parameter; its lookup keyed | def `:3027`, lookup `:3028`, call `:3057` | def `:3137`, lookup `:3138`, `"parents"` `:3147`, call `:3171` |
| `build_QuadSourceIntegral_batch`: resolves the set for its missing `(k, z_response)`s, then keys its `GkSource` lookup | def `:3181`, lookup `:3296`, call `:3325` | def `:3295`, resolve `:3413`, lookup `:3417`, `"parents"` `:3425`, call `:3447` |

**`test_main_plumbing.EXPECTED_SITES` did not move** (before and after: `wavenumber_exit_time` 1,
`TkNumericIntegration` 6, `GkNumericIntegration` 3, `QuadSourceIntegral` 2, `GkWKBIntegration` 3,
`TkWKBIntegration` 3, `GkSource` 6, `QuadSource` 3, `BackgroundModel` 1, `GkSourcePolicy` 2,
`GkSourcePolicyData` 4, `QuadSourcePolicy` 2, `IntegrationSolver` 7, `redshift` 1, `store_tag` 10,
`tolerance` 8, `wavenumber` 1). The helper's reads go through `object_read_batch`, which the finder
does not count (as `build_missing_GkSource`'s value reads never were), so it adds no site;
`test_main_plumbing.py` is not edited.

**The read-back guard's pins** (`python -c` over `test_factory_select_columns`, before and after):
`EXPECTED_COVERAGE` **47 → 50**, the three new row-read bindings of this prompt, each named in the
test with a comment:
- `GkSource.py:sqla_GkSource_factory.build.candidate` (the reader mode's candidates);
- `GkSource.py:sqla_GkSourceValue_factory._build_impl_model.candidate` (the same, decision 3);
- `GkSource.py:<module-level>._read_parent_sets.member` (the `GkSource_parents` rows a refusal or a
  labels-only hit reads).

`KNOWN_UNRESOLVED_TABLES` 11, `KNOWN_BLIND` 1, `KNOWN_UNCHECKED_INSERTS` 13 and
`KNOWN_BLIND_COLUMNS` 4 did not move; `KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS` are empty. (The new
`GkSource_parents` insert is a dict literal, so the guard checks its keys against the schema.)

**Are the values read once or twice in the GkSource stage? Once.** The stage now reads the parent
set serial-only for every `(k, z_response)` of its batch, runs the existence query keyed on it, and
reads the values — full `read_batch`, as before — for the missing items only. So a value is read
once, and a parent serial twice for a missing item (once in each form). Why: the existence query
needs the set, and on a resume most `GkSource`s exist, so reading their values in the pre-existence
read would be pure cost (the full read is about twice the serial-only one, below). The second read
is the one `compute()`'s assertion checks against the first.

## The measured cost of the pre-existence read

On a stand-in pool (`Datastore/tests/standin_pool.py`: the real `ShardedPool`, `Datastore` actor
code and factories, results pickled through, no Ray), two shards, the Gk rows written into `k`'s
shard at production density: 1,200 `z_source`s (100 per decade), 100 `z_response`s (sparseness
12), numeric integrations at the upper 600 `z_source`s and WKB integrations at all 1,200, each with
a value at every `z_response` below its `z_source`, under eight tags as `main.py` asks (45,250
`GkNumericValue` and 60,500 `GkWKBValue` rows). Each read repeated 7 times; medians, at a load
average of 11–19. The script (`measure_cost.py`) is in this prompt's scratch directory, not in the
tree.

```
rows: {'GkNumericIntegration': 600, 'GkNumericValue': 45250, 'GkWKBIntegration': 1200, 'GkWKBValue': 60500}

low z_response (most z_sources above it): z=102.3
  GkNumericValue  parents_only=True :   600 items, median    7.19 ms (min    7.06, max    9.89)
  GkNumericValue  parents_only=False:   600 items, median   15.37 ms (min   14.16, max   15.92)
  GkWKBValue      parents_only=True :  1199 items, median   10.98 ms (min   10.57, max   12.87)
  GkWKBValue      parents_only=False:  1199 items, median   27.85 ms (min   27.72, max   56.95)

mid z_response: z=7.762e+07
  GkNumericValue  parents_only=True :   600 items, median    7.13 ms (min    7.03, max    7.70)
  GkNumericValue  parents_only=False:   600 items, median   14.11 ms (min   13.83, max   40.60)
  GkWKBValue      parents_only=True :   611 items, median    9.28 ms (min    9.18, max   11.34)
  GkWKBValue      parents_only=False:   611 items, median   18.79 ms (min   17.91, max   22.76)
```

So one `(k, z_response)`'s parent set costs about **18 ms** (7.2 + 11.0 ms) at a low `z_response`,
where most `z_source`s lie above it, and about 16 ms at a middling one, against about 43 ms and
33 ms for the two full reads. The GkSource stage pays it for every item of its batch, missing or
not; the `GkSourcePolicyData` stage once per item (its three lookups share one resolution); the
`QuadSourceIntegral` stage once per distinct missing `(k, z_response)`. At production size (50
`k` × about 100 `z_response`s, 5,000 items) that is of order 90 s of serial-only reads per stage
before any parallelism across shards. README §5 rule 11: it is recorded, not argued against.

## The `substitute_main_py()` output

`docs/handover/quadsource_atol_sweep.py` loaded by path with `importlib.util.spec_from_file_location`
and `substitute_main_py()` called on this tree (it only reads `main.py`). The five
`** quadsource_atol_sweep:` lines are its own, and it raises unless each literal occurs exactly
once; the last five lines count each literal in `main.py` and give its line:

```
** quadsource_atol_sweep: replaced 1 occurrence of 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_SOURCE_K_VALUES)' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_RESPONSE_K_VALUES)' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'tol=DEFAULT_QUADRATURE_ATOL' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'tol=DEFAULT_QUADRATURE_RTOL' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of '        qsi_work_batches = list(grouper(qsi_work_items, n=750, incomplete="fill"))' in main.py
substituted source: 171537 characters
count=1 line=3998 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_SOURCE_K_VALUES)'
count=1 line=4010 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_RESPONSE_K_VALUES)'
count=1 line=3899 'tol=DEFAULT_QUADRATURE_ATOL'
count=1 line=3900 'tol=DEFAULT_QUADRATURE_RTOL'
count=1 line=3792 'qsi_work_batches = list(grouper(qsi_work_items, n=750, incomplete="fill"))'
```

All five still match exactly once. Each moved by the 122 lines this commit adds above them
(`:3670` → `:3792`, `:3777` → `:3899`, `:3778` → `:3900`, `:3876` → `:3998`, `:3888` → `:4010`). The
sweep script is not edited.

## The witness diff, in full (D5)

Pasted from `diff Datastore/tests/data/schema_at_datastore-integrity-09a.json
Datastore/tests/data/schema_at_datastore-integrity-09b.json`, default format, unedited. **9 hunks,
246 changed lines, 256 lines of output** (`grep -c '^[0-9]'` = 9, `grep -c '^[<>]'` = 246, `wc -l` =
256), the sketch's counts exactly. Each hunk is located by the JSON key that encloses it (scanned
upward from its first line in the new witness; the seventh opens by closing `GkSourceValue`'s entry
and is `GkSource_parents`' whole class entry):

| Kind | Hunk | Class | Enclosing key |
|---|---|---|---|
| 1. its name in the class order | `31a32` | `GkSource_parents` | `class_order` |
| 2. the column in the schema record | `2214a2216,2219` | `GkSource` | `classes.GkSource.record.columns` |
| 3. the column's description | `2375a2381,2393` | `GkSource` | `classes.GkSource.table.columns` |
| 4. the `CREATE TABLE` DDL | `2554c2572` | `GkSource` | `classes.GkSource.table.ddl` |
| 5. the index DDL | `2557a2576` | `GkSource` | `classes.GkSource.table.index_ddl` |
| 6. the index description | `2576a2596,2602` | `GkSource` | `classes.GkSource.table.indexes` |
| 7. the new class's whole entry | `3517a3544,3759` | `GkSource_parents` | `classes.GkSource_parents` |
| 8. its name in the metadata's tables | `11570a11813` | `GkSource_parents` | `metadata_tables` |
| 9. its name in the table order | `11628a11872` | `GkSource_parents` | `table_order` |

**Six kinds for `GkSource` (hunks 2–6, the one column and its index) and four for
`GkSource_parents` (hunks 1, 7, 8, 9). No hunk is outside the two classes.**

```
31a32
>   "GkSource_parents",
2214a2216,2219
>       "column": "parents_digest",
>       "of_table": "GkSource"
>      },
>      {
2375a2381,2393
>       "unique": null
>      },
>      {
>       "autoincrement": "'auto'",
>       "default": null,
>       "foreign_keys": [],
>       "index": true,
>       "name": "parents_digest",
>       "nullable": false,
>       "primary_key": false,
>       "server_default": null,
>       "sqlite_type": "VARCHAR(64)",
>       "type": "String(length=64)",
2554c2572
<     "ddl": "CREATE TABLE \"GkSource\" (\n\tserial INTEGER NOT NULL, \n\tversion INTEGER, \n\ttimestamp DATETIME, \n\tlabel VARCHAR(256), \n\twavenumber_exit_serial INTEGER NOT NULL, \n\tmodel_serial INTEGER NOT NULL, \n\tz_response_serial INTEGER NOT NULL, \n\tz_max_serial INTEGER NOT NULL, \n\tz_samples INTEGER NOT NULL, \n\tnumeric_smallest_z_serial INTEGER, \n\t\"primary_WKB_largest_z_serial\" INTEGER, \n\tvalidated BOOLEAN NOT NULL, \n\tmetadata VARCHAR(256), \n\tPRIMARY KEY (serial), \n\tFOREIGN KEY(version) REFERENCES version (serial), \n\tFOREIGN KEY(wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(model_serial) REFERENCES \"BackgroundModel\" (serial), \n\tFOREIGN KEY(z_response_serial) REFERENCES redshift (serial), \n\tFOREIGN KEY(z_max_serial) REFERENCES redshift (serial), \n\tFOREIGN KEY(numeric_smallest_z_serial) REFERENCES redshift (serial), \n\tFOREIGN KEY(\"primary_WKB_largest_z_serial\") REFERENCES redshift (serial)\n)",
---
>     "ddl": "CREATE TABLE \"GkSource\" (\n\tserial INTEGER NOT NULL, \n\tversion INTEGER, \n\ttimestamp DATETIME, \n\tlabel VARCHAR(256), \n\twavenumber_exit_serial INTEGER NOT NULL, \n\tmodel_serial INTEGER NOT NULL, \n\tz_response_serial INTEGER NOT NULL, \n\tparents_digest VARCHAR(64) NOT NULL, \n\tz_max_serial INTEGER NOT NULL, \n\tz_samples INTEGER NOT NULL, \n\tnumeric_smallest_z_serial INTEGER, \n\t\"primary_WKB_largest_z_serial\" INTEGER, \n\tvalidated BOOLEAN NOT NULL, \n\tmetadata VARCHAR(256), \n\tPRIMARY KEY (serial), \n\tFOREIGN KEY(version) REFERENCES version (serial), \n\tFOREIGN KEY(wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(model_serial) REFERENCES \"BackgroundModel\" (serial), \n\tFOREIGN KEY(z_response_serial) REFERENCES redshift (serial), \n\tFOREIGN KEY(z_max_serial) REFERENCES redshift (serial), \n\tFOREIGN KEY(numeric_smallest_z_serial) REFERENCES redshift (serial), \n\tFOREIGN KEY(\"primary_WKB_largest_z_serial\") REFERENCES redshift (serial)\n)",
2557a2576
>      "CREATE INDEX \"ix_GkSource_parents_digest\" ON \"GkSource\" (parents_digest)",
2576a2596,2602
>       "unique": false
>      },
>      {
>       "columns": [
>        "parents_digest"
>       ],
>       "name": "ix_GkSource_parents_digest",
3517a3544,3759
>    }
>   },
>   "GkSource_parents": {
>    "has_table": true,
>    "record": {
>     "columns": [
>      {
>       "column": "gk_source_serial",
>       "of_table": "GkSource_parents"
>      },
>      {
>       "column": "z_source_serial",
>       "of_table": "GkSource_parents"
>      },
>      {
>       "column": "numeric_serial",
>       "of_table": "GkSource_parents"
>      },
>      {
>       "column": "wkb_serial",
>       "of_table": "GkSource_parents"
>      }
>     ],
>     "name": "GkSource_parents",
>     "serial_col": {
>      "column": "serial",
>      "of_table": "GkSource_parents"
>     },
>     "table": {
>      "table": "GkSource_parents"
>     },
>     "use_serial": true,
>     "use_stepping": false,
>     "use_timestamp": false,
>     "use_version": false,
>     "validate_on_startup": false
>    },
>    "table": {
>     "columns": [
>      {
>       "autoincrement": "'auto'",
>       "default": null,
>       "foreign_keys": [],
>       "index": null,
>       "name": "serial",
>       "nullable": false,
>       "primary_key": true,
>       "server_default": null,
>       "sqlite_type": "INTEGER",
>       "type": "Integer()",
>       "unique": null
>      },
>      {
>       "autoincrement": "'auto'",
>       "default": null,
>       "foreign_keys": [
>        {
>         "ondelete": null,
>         "onupdate": null,
>         "target": "GkSource.serial"
>        }
>       ],
>       "index": true,
>       "name": "gk_source_serial",
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
>       "foreign_keys": [
>        {
>         "ondelete": null,
>         "onupdate": null,
>         "target": "redshift.serial"
>        }
>       ],
>       "index": true,
>       "name": "z_source_serial",
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
>       "foreign_keys": [
>        {
>         "ondelete": null,
>         "onupdate": null,
>         "target": "GkWKBIntegration.serial"
>        }
>       ],
>       "index": true,
>       "name": "wkb_serial",
>       "nullable": true,
>       "primary_key": false,
>       "server_default": null,
>       "sqlite_type": "INTEGER",
>       "type": "Integer()",
>       "unique": null
>      }
>     ],
>     "constraints": [
>      {
>       "columns": [
>        "gk_source_serial"
>       ],
>       "kind": "ForeignKeyConstraint",
>       "name": null,
>       "targets": [
>        "GkSource.serial"
>       ]
>      },
>      {
>       "columns": [
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
>        "serial"
>       ],
>       "kind": "PrimaryKeyConstraint",
>       "name": null
>      },
>      {
>       "columns": [
>        "wkb_serial"
>       ],
>       "kind": "ForeignKeyConstraint",
>       "name": null,
>       "targets": [
>        "GkWKBIntegration.serial"
>       ]
>      },
>      {
>       "columns": [
>        "z_source_serial"
>       ],
>       "kind": "ForeignKeyConstraint",
>       "name": null,
>       "targets": [
>        "redshift.serial"
>       ]
>      }
>     ],
>     "ddl": "CREATE TABLE \"GkSource_parents\" (\n\tserial INTEGER NOT NULL, \n\tgk_source_serial INTEGER NOT NULL, \n\tz_source_serial INTEGER NOT NULL, \n\tnumeric_serial INTEGER, \n\twkb_serial INTEGER, \n\tPRIMARY KEY (serial), \n\tFOREIGN KEY(gk_source_serial) REFERENCES \"GkSource\" (serial), \n\tFOREIGN KEY(z_source_serial) REFERENCES redshift (serial), \n\tFOREIGN KEY(numeric_serial) REFERENCES \"GkNumericIntegration\" (serial), \n\tFOREIGN KEY(wkb_serial) REFERENCES \"GkWKBIntegration\" (serial)\n)",
>     "index_ddl": [
>      "CREATE INDEX \"ix_GkSource_parents_gk_source_serial\" ON \"GkSource_parents\" (gk_source_serial)",
>      "CREATE INDEX \"ix_GkSource_parents_numeric_serial\" ON \"GkSource_parents\" (numeric_serial)",
>      "CREATE INDEX \"ix_GkSource_parents_wkb_serial\" ON \"GkSource_parents\" (wkb_serial)",
>      "CREATE INDEX \"ix_GkSource_parents_z_source_serial\" ON \"GkSource_parents\" (z_source_serial)"
>     ],
>     "indexes": [
>      {
>       "columns": [
>        "gk_source_serial"
>       ],
>       "name": "ix_GkSource_parents_gk_source_serial",
>       "unique": false
>      },
>      {
>       "columns": [
>        "numeric_serial"
>       ],
>       "name": "ix_GkSource_parents_numeric_serial",
>       "unique": false
>      },
>      {
>       "columns": [
>        "wkb_serial"
>       ],
>       "name": "ix_GkSource_parents_wkb_serial",
>       "unique": false
>      },
>      {
>       "columns": [
>        "z_source_serial"
>       ],
>       "name": "ix_GkSource_parents_z_source_serial",
>       "unique": false
>      }
>     ],
>     "name": "GkSource_parents"
11570a11813
>   "GkSource_parents",
11628a11872
>   "GkSource_parents",
```

## The fingerprint re-capture (D5)

`FINGERPRINT_FORMAT` goes from 4 to 5. The golden was captured the way
`test_store_fingerprint._FullStores` builds its baseline: `build_full_store` into a temporary
directory, `read_inventory(...)`, `fingerprint_of(..., taken=None)`, serialised as
`json.dumps(indent=2, sort_keys=True) + "\n"` (`capture_golden.py`, in this prompt's scratch
directory, not in the tree). **The script was first run on a clean `git archive` of `HEAD` and
reproduced `full_store_fingerprint_datastore-integrity-09a.json` byte for byte** (`cmp`).

- **Moved**, checked class by class (a script comparing `classes[name]` of the two goldens; both
  list the same 21 classes in the same order):

  | Class | Count | Tag-set counts | Digest | Why |
  |---|---|---|---|---|
  | `GkSource` | 3 → 3 | [1, 2] → [1, 2] | `5a3db324f91e2afd…` → `c256dbd77a23f6ff…` | its key gains the resolved parent set |
  | `GkSourcePolicyData` | 3 → 3 | [3] → [3] | `995c3260997b3e59…` → `2169607c6132f98e…` | its key's `source` is its `GkSource`'s reference digest, which moved |

- **Not moved: the other 19 classes**, identical in count, digest and tag sets. `QuadSourceIntegral`
  does not key on a `GkSourcePolicyData` identity yet (09c).
- `problems` is `{}` before and after. The overall digest `efdaf6ef5faa1764…` → `a80e53c2dc30e97f…`.
- The diff of the two goldens, in full (`diff`, unedited): three digests of `GkSource`, two of
  `GkSourcePolicyData`, and the overall digest and format:

```
47c47
<       "digest": "5a3db324f91e2afd087c0a1709e0e2390617ed630055fe3bbac3dfba902bc524",
---
>       "digest": "c256dbd77a23f6ffa548d76d3ac1242a3426ba3aeeda69a00b3d5c013166bb7e",
51c51
<           "digest": "95f6dd67442c8f5f1d5ef807716307f8a9958d6df373b1de8e8bbecbeefa5803",
---
>           "digest": "a42b3f668dc8fc4ed7ef8099209f5e9425db50f7c0c02e9754dc85d35039b014",
58c58
<           "digest": "6c35f3eae7c0469269d5475c567d0dceebb764be67054c60514aa2b4d298d702",
---
>           "digest": "db6807fce88d3bdb1a42d1fbe570a6b5d3237b4e33f74f2947cf5f8edad503bd",
79c79
<       "digest": "995c3260997b3e597478e4e7ec3730dbcdac18448778ed362466b576b2d8794d",
---
>       "digest": "2169607c6132f98e85df19a738e97e44ccff24b014b4bfa374df99f6f5e4228b",
83c83
<           "digest": "995c3260997b3e597478e4e7ec3730dbcdac18448778ed362466b576b2d8794d",
---
>           "digest": "2169607c6132f98e85df19a738e97e44ccff24b014b4bfa374df99f6f5e4228b",
319,320c319,320
<   "digest": "efdaf6ef5faa176419837276e0a3847f4f4dae267de791bfc09976137662e069",
<   "fingerprint_format": 4,
---
>   "digest": "a80e53c2dc30e97ff194a1c2f2fd40ab61a3463bc172e1f36d6ffb4aff7f5b94",
>   "fingerprint_format": 5,
```

- **`store retire` of a store written before this prompt must be run at a pre-09b SHA (D4).** Its
  recorded fingerprint is format 4 or older; at this tree `compare_fingerprints` says the formats
  differ and does not compare, and an inventory recomputed here finds `GkSource_parents` absent
  (`absent-table` for `GkSource`, whose records are then none, and `unresolved-parent` for every
  `GkSourcePolicyData`). `store fingerprint` and `store amend` at this tree record format 5.

## Old stores (D4)

- **A pre-09b `GkSource` table is refused.** `TestAnOldStoreIsRefused` recreates `GkSource` from
  `schema_at_datastore-integrity-09a.json`'s own DDL (no `parents_digest`) and drops
  `GkSource_parents`, as a pre-09b store lacks it: a keyed `build()`, the reader mode, and
  `GkSourceValue._build_impl_model` each raise `RuntimeError`: *"GkSource.build(): the GkSource table
  has no "parents_digest" column, or the store has no GkSource_parents table. This datastore predates
  GkSource rows being keyed on the Gk numeric and WKB rows they are assembled from
  (prompts/datastore-integrity, prompt 09b). This tree does not read it; it remains readable at the
  git SHA its sidecar records."* It does not say "regenerated".
- **The four WKB value reads on a pre-07 table** (the 07 issue): `TestTheWKBValueReadsRefuseAPre07Store`
  recreates both WKB tables from `schema_at_datastore-integrity-06.json`'s DDL with one validated row
  and one value each. `GkWKBValue.read_batch` (parentless, a numeric parent, `PURE_WKB`,
  `parents_only`), `GkWKBValue._build_impl_model` (parentless, a parent, `PURE_WKB`),
  `TkWKBValue.read_batch` and `TkWKBValue._build_impl_model` (parentless, a parent) all raise 07's
  D4 message: 11 sub-tests. A control reads the current tables.
- **`store retire`** of a store written before this prompt must be run at a pre-09b SHA (above).

## Readers (README §5 rule 11), and what now refuses

| Reader | Before | After |
|---|---|---|
| `main.py`, the GkSource stage | existence query and construction by labels; values read for missing items | the set resolved serial-only first; both keyed on it; values still read for missing items only |
| `main.py`, the `GkSourcePolicyData` stage (three lookups) | labels | keyed on the set, resolved once per batch (§0.2) |
| `main.py`, the `QuadSourceIntegral` stage's `GkSource` lookup | labels | keyed on the set of each missing `(k, z_response)` (§0.2) |
| `extract_GkSource_data.py` (`:879` at `HEAD`) | labels | `labels_only=True` |
| `extract_QuadSourceIntegral_data.py`, its `GkSource` lookup (`:1154`) | labels | `labels_only=True` |
| `docs/source-remediation-verification/analyse_greens_and_source.py` (`:282`, no tags) | labels | `labels_only=True` |
| `docs/source-remediation-verification/run_quadsource_integrals.py` (`:364`) | labels | `labels_only=True` |
| `docs/handover/quadsource_atol_sweep.py`, `docs/gktk-remedial/scoped_pipeline_run.py` | `exec` `main.py` | keyed through `main.py`; not edited |
| `tools/inventory_report.py` | a `GkSource`'s labels | also its parent set, by physical labels |
| any `GkWKBValue` / `TkWKBValue` read (`main.py`'s GkSource stage; `probe_05`) | a parentless read served a pre-07 store | refused (D4) |
| `store retire`, `store fingerprint` comparisons | format 4 | format 5 |

**What now refuses a store it read before:**
- **Every store written before this commit** (the three under `var/` included, not opened here), in
  `main.py`'s GkSource, `GkSourcePolicyData` and `QuadSourceIntegral` stages, in the four readers
  above, in `GkSourceValue`'s model read, and in the sweep script and the scoped driver, which
  `exec` `main.py`: the D4 message. The sweep store under `var/` is such a store.
- **A pre-07 store in the GkSource stage's value reads**, which a parentless read used to serve.
- **The four readers, on a store holding two `GkSource`s under the same labels**, which they used to
  serve one of; a store built once holds one.
- **`store retire` and `store fingerprint` comparisons**, as above.
- **The audit probes**, below.

### The fourteen audit probes, before and after

Run at `HEAD` (`e080895`, a `git archive`, `venv` symlinked) and on an archive of the final staged
tree (`git write-tree`, then `git archive`), each probe in a subprocess with `PYTHONPATH` at its root,
both under the scratch directory: **the probes were not run in the checkout**, since
`replicated_write_fault_probe.py` writes a store beside itself. No probe opens anything under
`var/`: each builds its schema in memory or a store in a temporary directory (`lookup_keys/common.py`
mentions `var/` in a docstring only), and `replicated_serial_census.py` was run with no argument, so
it opened no store. Outputs compared after normalising the root path. **No probe edited.**

| Probe | Exit before → after | Output |
|---|---|---|
| `fk_sweep_probe.py` | 0 → 0 | **changed**: gains `GkSource_parents`' block (its four foreign keys, "UNLISTED (written by parent's Datastore actor)") |
| `hot_journal_probe.py` | 0 → 0 | identical |
| `probe_01_tagged_read_batch.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `probe_01b_fix_shape.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `probe_02_readback_guard_by_hand.py` | 0 → 0 | **changed**: the moved line numbers of the `GkSource`, Gk and Tk WKB value factories' functions, and the new row bindings (the two serial-only `row`s, `_build_impl_model`'s `candidate`); still clean |
| `probe_03_column_existence.py` | 0 → 0 | **changed**: "checked 1256" → "checked 1268 resolvable column references" |
| `probe_04_exit_time_and_solver.py` | 0 → 0 | identical |
| `probe_04b_exact_match_fix_shape.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `probe_05_value_lookups_and_read_batch.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `probe_06_oneloop_tags_and_redshift_zero.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `replicated_divergence_probe.py` | 0 → 0 | **changed**: two new lines, `unresolved-parent: GkSource` (field `parents`, serial 3) and `unresolved-parent: GkSourcePolicyData` (field `source`, serial 3), both on shard #1: the probe's own fixture's `GkSource` 3 names a Gk row its shard does not hold, so its `GkSourcePolicyData` cannot resolve either (the sketch's cascade) |
| `replicated_serial_census.py` | 0 → 0 | identical |
| `replicated_write_fault_probe.py` | 1 → 1 | identical but for the temporary directory name and a timestamp in one line (already failing at `HEAD`) |
| `tq_fk_probe.py` | 0 → 0 | identical |

None newly fails; five change output. The six that fail at `HEAD` (`probe_01`, `01b`, `04b`, `05`,
`06`, `replicated_write_fault_probe`) are unchanged. The sketch measured four changes; the fifth is
`replicated_write_fault_probe`'s timestamp line, which changes on every run. The five are named in
the rule-11 list: `fk_sweep_probe`, `probe_02`, `probe_03`, `replicated_divergence_probe` (whose
fixture's `GkSource` 3 is now `unresolved-parent`), and the timestamp line of
`replicated_write_fault_probe`.

## Deviations from the prompt

1. **`GkSource_parents` has its own `serial`.** `IMPLEMENTATION CHOICE`. The recommended shape
   lists four columns. A serial gives each member row an identity, as every value table's row has,
   and it is what §0.5's granted `IDENTITY` edit needs: `vary_row` finds a row by its serial. The
   sketch had one too.
2. **`"GkSource_parents": 7500` in `ClientPool`'s serial batch-size map; nothing in
   `config/sharding.py`.** `IMPLEMENTATION CHOICE`, within §0.7's "only if the new table's writes
   need them". A serial is leased from the broker per row, and a `GkSource` writes one row per
   `z_source` (up to ~1,200), so the default batch of 500 would cost two or three broker round
   trips per `GkSource`; 7,500 is `GkSourceValue`'s, which has one row per `z_source` too. The table
   is written only inside `GkSource.store()`, on the source's own shard, like `GkSource_tags`, which
   `sharded_tables` does not list either; listing it would also change what a primary records of its
   sharded tables, so an existing pool would refuse to open.
3. **The object carries the key set and the assembled set, and `store()` writes the assembled
   one.** `IMPLEMENTATION CHOICE` (the design, above).
4. **`compute()` asserts that its values were read from the set it is keyed on.** `IMPLEMENTATION
   CHOICE`, the charter's §2 ("a derived row is keyed on the rows its `compute()` receives") and
   09a's precedent. It also refuses a labels-only object and an empty set.
5. **`validate_on_startup`'s prune deletes a pruned `GkSource`'s `GkSource_parents` rows.**
   `STRUCTURALLY REQUIRED`. The dispatch lists the factory's `register`, `build`, `store`,
   `inventory_records` and reader mode; without this one statement a prune would leave member rows
   naming a deleted row (an `orphan-member`, and a foreign-key violation).
6. **The reader mode's one match carries the set its `GkSource_parents` rows record** (one more
   query). `IMPLEMENTATION CHOICE`: a stored object always reports its parents.
7. **The serial-only form, `parents_only=True`, on both Gk value reads.** `IMPLEMENTATION CHOICE`
   (the prompt's "your choice of shape"); measured above.
8. **`main.py` imports `GkSource_parent_set_of_serials`** (one line, `:32`). `IMPLEMENTATION CHOICE`:
   the helper forms the set with the target's own function rather than a copy of it.
9. **Three nested functions gain a `GkSource_parents` parameter** (`build_missing_GkSource`,
   `build_incomplete_GkSourcePolicy`, `build_missing_GkSourcePolicy`), handed by their one caller.
   `IMPLEMENTATION CHOICE`: the `GkSourcePolicyData` stage resolves once for its three lookups, and
   the construction passes the very set its existence query used.
10. **The inventory keeps members as tuples of digests, and the set's digest covers its field
    names.** `IMPLEMENTATION CHOICE`, made after a first draft kept each member as a mapping: a
    production store holds of order 3 million members (5,000 `GkSource`s × ~600), and a mapping
    per member would cost of order 0.7 GB in `read_inventory`.
11. **Two new problem names, `empty-parent-set` and `orphan-member`.** `IMPLEMENTATION CHOICE`; the
    module docstring lists them. An unresolvable member reuses `unresolved-parent`, and an absent
    member table `absent-table`.
12. **With the member table absent, no record of the class comes from that shard.**
    `IMPLEMENTATION CHOICE`: the set cannot be formed there, as with a missing key column. The
    sketch did the same.
13. **The WKB refusal is written in each of the four value reads.** `STRUCTURALLY REQUIRED`: the
    dispatch grants only those functions in `GkWKBIntegration.py` and `TkWKBIntegration.py`, so no
    module-level helper was added there.
14. **The two Gk `read_batch`es' full value query moved under an `else:`.** `IMPLEMENTATION CHOICE`:
    the read-back guard charges a row's reads to the last binding above them, so the full query,
    which `make_obj` reads, stays textually last. The diff shows it re-indented; its columns are
    unchanged.
15. **The new Datastore test module's helper is `read`, not `inventory`.** `IMPLEMENTATION CHOICE`,
    forced by `test_inventory_retired`'s `git grep`, which matched `cls.inventory(` only once the
    module was staged (log 08, deviation 8): the first staged run failed that one test, and it was
    renamed.

No `UNINTENDED DRIFT`. `git diff --cached -U0` shows only the hunks this log lists.

## Existing-test edits, by §0.5's "Granted for 09b"

| File | Edit | Reason |
|---|---|---|
| `Datastore/tests/real_store_fixtures.py` | the three `GkSource` rows gain `parents_digest` (computed by `GkSource_parents_digest`, imported); six `GkSource_parents` rows, 601–606 (`_GK_SOURCE_PARENT_ROWS`, written by `_gk_source_parents`): `GkSource` 1 and 2 (shard 0, exit 1) name Gk numeric 1 at `z_source` 1 and the pure-WKB Gk WKB 1 at `z_source` 2; `GkSource` 3 (shard 1, exit 2) Gk numeric 2 at 1 and Gk WKB 2 at 2 | the column is NOT NULL and the set is identity. Those are the only Gk rows of each shard, each at its own `z_source` (agent's choice) |
| `Datastore/tests/test_store_inventory.py` | `IDENTITY["GkSource"]` gains `"parents": ("GkSource_parents", 604, "numeric_serial", 1)`, with a comment | the key's fields now include it; row 604 is `GkSource` 2's member at `z_source` 2, and Gk numeric 1 is real and on the same shard, so the variation tests identity, not resolution |
| `ComputeTargets/tests/test_gauss_order_key.py` (granted on its condition) | `TARGETS["GkSource"]`'s payload gains a stand-in `"parents"`; `_GkSourceStandIn` gains `parents` and `assembled_parents`; `_store_payload` names a `GkSource_parents` inserter for `GkSource` | `build()` now requires the set, and `store()` writes it. No assertion changed |
| `Datastore/tests/test_schema_builder.py` | `WITNESS` → `schema_at_datastore-integrity-09b.json`; the docstring's "current one" sentence | D5 |
| `RunRegistry/tests/test_store_fingerprint.py` | `GOLDEN` → `full_store_fingerprint_datastore-integrity-09b.json` | D5 |
| `Datastore/tests/test_factory_select_columns.py` | `EXPECTED_COVERAGE` gains the three sites above, with a comment | §0.4: this prompt's own new sites |

No other existing test changed; `test_main_plumbing.py` and `test_inventory_report.py` are not
touched.

## Verification performed

- **The starting tree.** `git status` clean at `e080895`; `python -m RunRegistry list` showed nothing
  `running` (of seven runs, four `done`, one `failed` and two `unknown`, all days old). No other session's change
  appeared in this checkout.
- **The six suites** on the staged tree, above.
- **The new modules, `-v`, unfiltered** (71 tests; the `!!` and `|` lines are the factories' and
  `assemble_GkSource_values`' own prints):

```
test_a_keyed_lookup (Datastore.tests.test_gksource_parent_set.TestAnOldStoreIsRefused.test_a_keyed_lookup) ... ok
test_the_reader_mode (Datastore.tests.test_gksource_parent_set.TestAnOldStoreIsRefused.test_the_reader_mode) ... ok
test_the_value_model_read (Datastore.tests.test_gksource_parent_set.TestAnOldStoreIsRefused.test_the_value_model_read) ... ok
test_a_lookup_naming_neither_the_set_nor_the_mode_raises (Datastore.tests.test_gksource_parent_set.TestKeyedOnItsSet.test_a_lookup_naming_neither_the_set_nor_the_mode_raises) ... ok
test_a_miss_carries_the_set_it_was_asked_under (Datastore.tests.test_gksource_parent_set.TestKeyedOnItsSet.test_a_miss_carries_the_set_it_was_asked_under) ... ok
test_a_subset_is_not_the_set (Datastore.tests.test_gksource_parent_set.TestKeyedOnItsSet.test_a_subset_is_not_the_set) ... ok
test_found_by_its_own_set_and_missed_by_another (Datastore.tests.test_gksource_parent_set.TestKeyedOnItsSet.test_found_by_its_own_set_and_missed_by_another) ... ok
test_the_labels_and_tags_are_still_part_of_the_key (Datastore.tests.test_gksource_parent_set.TestKeyedOnItsSet.test_the_labels_and_tags_are_still_part_of_the_key) ... ok
test_the_set_and_the_mode_together_are_an_error (Datastore.tests.test_gksource_parent_set.TestKeyedOnItsSet.test_the_set_and_the_mode_together_are_an_error) ... ok
test_two_sets_differing_in_one_row_give_two_rows_each_found_by_its_own (Datastore.tests.test_gksource_parent_set.TestKeyedOnItsSet.test_two_sets_differing_in_one_row_give_two_rows_each_found_by_its_own) ... ok
test_a_malformed_member_is_refused (Datastore.tests.test_gksource_parent_set.TestTheDigest.test_a_malformed_member_is_refused) ... ok
test_it_is_the_sha256_of_the_sorted_members (Datastore.tests.test_gksource_parent_set.TestTheDigest.test_it_is_the_sha256_of_the_sorted_members) ... ok
test_a_dangling_reference_is_reported_for_each_key (Datastore.tests.test_gksource_parent_set.TestTheForeignKeys.test_a_dangling_reference_is_reported_for_each_key) ... ok
test_a_store_written_by_the_factory_passes_too (Datastore.tests.test_gksource_parent_set.TestTheForeignKeys.test_a_store_written_by_the_factory_passes_too) ... ok
test_the_keys_are_declared (Datastore.tests.test_gksource_parent_set.TestTheForeignKeys.test_the_keys_are_declared) ... ok
test_the_rows_are_not_satisfied_by_coincidence (Datastore.tests.test_gksource_parent_set.TestTheForeignKeys.test_the_rows_are_not_satisfied_by_coincidence) ... ok
test_the_store_passes_foreign_key_check (Datastore.tests.test_gksource_parent_set.TestTheForeignKeys.test_the_store_passes_foreign_key_check) ... ok
test_a_member_is_the_parents_identity_and_not_its_serial (Datastore.tests.test_gksource_parent_set.TestTheInventoryResolvesTheSet.test_a_member_is_the_parents_identity_and_not_its_serial)
Each member names the reference digests of its rows, as a key parent does. ... ok
test_a_store_without_the_table_is_named_absent (Datastore.tests.test_gksource_parent_set.TestTheInventoryResolvesTheSet.test_a_store_without_the_table_is_named_absent) ... ok
test_an_empty_set_is_a_named_problem (Datastore.tests.test_gksource_parent_set.TestTheInventoryResolvesTheSet.test_an_empty_set_is_a_named_problem) ... ok
test_an_orphan_member_row_is_a_named_problem (Datastore.tests.test_gksource_parent_set.TestTheInventoryResolvesTheSet.test_an_orphan_member_row_is_a_named_problem) ... ok
test_an_unresolvable_member_is_a_named_problem (Datastore.tests.test_gksource_parent_set.TestTheInventoryResolvesTheSet.test_an_unresolvable_member_is_a_named_problem) ... ok
test_parents_digest_is_not_identity (Datastore.tests.test_gksource_parent_set.TestTheInventoryResolvesTheSet.test_parents_digest_is_not_identity) ... ok
test_resolve_names_every_member (Datastore.tests.test_gksource_parent_set.TestTheInventoryResolvesTheSet.test_resolve_names_every_member) ... ok
test_the_fixture_has_every_set_resolved (Datastore.tests.test_gksource_parent_set.TestTheInventoryResolvesTheSet.test_the_fixture_has_every_set_resolved) ... ok
test_two_sources_differing_only_in_one_parent_are_two_records (Datastore.tests.test_gksource_parent_set.TestTheInventoryResolvesTheSet.test_two_sources_differing_only_in_one_parent_are_two_records)
GkSource 7 is GkSource 1 in every column and tag, but for one member: at z_source 2 it ... ok
test_no_match_is_a_labels_only_miss_that_cannot_be_computed (Datastore.tests.test_gksource_parent_set.TestTheReaderMode.test_no_match_is_a_labels_only_miss_that_cannot_be_computed) ... ok
test_one_match_is_returned_without_being_given_the_set (Datastore.tests.test_gksource_parent_set.TestTheReaderMode.test_one_match_is_returned_without_being_given_the_set) ... ok
test_the_tags_still_select_under_the_mode (Datastore.tests.test_gksource_parent_set.TestTheReaderMode.test_the_tags_still_select_under_the_mode) ... ok
test_two_matches_are_refused_naming_their_digests_and_where_they_differ (Datastore.tests.test_gksource_parent_set.TestTheReaderMode.test_two_matches_are_refused_naming_their_digests_and_where_they_differ) ... ok
test_an_empty_set_is_refused_and_nothing_is_written (Datastore.tests.test_gksource_parent_set.TestTheRecordAgreesWithTheKey.test_an_empty_set_is_refused_and_nothing_is_written) ... ok
test_parents_digest_is_the_digest_of_the_rows_written (Datastore.tests.test_gksource_parent_set.TestTheRecordAgreesWithTheKey.test_parents_digest_is_the_digest_of_the_rows_written) ... ok
test_the_rows_written_are_the_set_assembled_not_the_set_asked_for (Datastore.tests.test_gksource_parent_set.TestTheRecordAgreesWithTheKey.test_the_rows_written_are_the_set_assembled_not_the_set_asked_for)
An object asked for under SET_A whose values were assembled from SET_B: the row records ... ok
test_no_match_is_a_miss (Datastore.tests.test_gksource_parent_set.TestTheValueModelRead.test_no_match_is_a_miss) ... ok
test_one_match_serves_its_value (Datastore.tests.test_gksource_parent_set.TestTheValueModelRead.test_one_match_serves_its_value) ... ok
test_two_matches_are_refused_naming_what_distinguishes_them (Datastore.tests.test_gksource_parent_set.TestTheValueModelRead.test_two_matches_are_refused_naming_what_distinguishes_them) ... ok
test_each_numeric_value_carries_its_integration (Datastore.tests.test_gksource_parent_set.TestTheValueReadsHandBackTheirParent.test_each_numeric_value_carries_its_integration) ... ok
test_each_wkb_value_carries_its_integration (Datastore.tests.test_gksource_parent_set.TestTheValueReadsHandBackTheirParent.test_each_wkb_value_carries_its_integration) ... ok
test_the_serial_only_form_still_refuses_a_second_parent (Datastore.tests.test_gksource_parent_set.TestTheValueReadsHandBackTheirParent.test_the_serial_only_form_still_refuses_a_second_parent) ... ok
test_the_serial_only_forms_return_the_pairs_and_no_values (Datastore.tests.test_gksource_parent_set.TestTheValueReadsHandBackTheirParent.test_the_serial_only_forms_return_the_pairs_and_no_values) ... ok
test_the_two_forms_agree (Datastore.tests.test_gksource_parent_set.TestTheValueReadsHandBackTheirParent.test_the_two_forms_agree) ... ok
test_a_current_table_is_still_read (Datastore.tests.test_gksource_parent_set.TestTheWKBValueReadsRefuseAPre07Store.test_a_current_table_is_still_read)
The control: the same four reads on the current schema's tables find their rows. ... ok
test_gk_wkb_value_build_impl_model (Datastore.tests.test_gksource_parent_set.TestTheWKBValueReadsRefuseAPre07Store.test_gk_wkb_value_build_impl_model) ... ok
test_gk_wkb_value_read_batch (Datastore.tests.test_gksource_parent_set.TestTheWKBValueReadsRefuseAPre07Store.test_gk_wkb_value_read_batch) ... ok
test_tk_wkb_value_build_impl_model (Datastore.tests.test_gksource_parent_set.TestTheWKBValueReadsRefuseAPre07Store.test_tk_wkb_value_build_impl_model) ... ok
test_tk_wkb_value_read_batch (Datastore.tests.test_gksource_parent_set.TestTheWKBValueReadsRefuseAPre07Store.test_tk_wkb_value_read_batch) ... ok
test_each_function_looks_a_gk_source_up_once (ComputeTargets.tests.test_gksource_parent_set_main.TestMainPyKeysEveryGkSourceLookup.test_each_function_looks_a_gk_source_up_once) ... ok
test_every_lookup_is_keyed_on_a_resolved_set (ComputeTargets.tests.test_gksource_parent_set_main.TestMainPyKeysEveryGkSourceLookup.test_every_lookup_is_keyed_on_a_resolved_set) ... ok
test_every_set_comes_from_the_one_helper (ComputeTargets.tests.test_gksource_parent_set_main.TestMainPyKeysEveryGkSourceLookup.test_every_set_comes_from_the_one_helper)
Each function either calls the helper itself or is handed its result by the one that ... ok
test_no_stage_asks_for_the_reader_mode (ComputeTargets.tests.test_gksource_parent_set_main.TestMainPyKeysEveryGkSourceLookup.test_no_stage_asks_for_the_reader_mode) ... ok
test_the_construction_passes_the_set_its_existence_query_used (ComputeTargets.tests.test_gksource_parent_set_main.TestMainPyKeysEveryGkSourceLookup.test_the_construction_passes_the_set_its_existence_query_used) ... ok
test_the_downstream_stages_resolve_before_their_lookups (ComputeTargets.tests.test_gksource_parent_set_main.TestMainPyKeysEveryGkSourceLookup.test_the_downstream_stages_resolve_before_their_lookups) ... ok
test_the_gk_source_stage_resolves_the_set_before_its_existence_query (ComputeTargets.tests.test_gksource_parent_set_main.TestMainPyKeysEveryGkSourceLookup.test_the_gk_source_stage_resolves_the_set_before_its_existence_query) ... ok
test_the_helper_forms_the_set_over_the_grid_the_gk_source_stage_assembles_over (ComputeTargets.tests.test_gksource_parent_set_main.TestMainPyKeysEveryGkSourceLookup.test_the_helper_forms_the_set_over_the_grid_the_gk_source_stage_assembles_over) ... ok
test_the_helper_reads_both_classes_serial_only (ComputeTargets.tests.test_gksource_parent_set_main.TestMainPyKeysEveryGkSourceLookup.test_the_helper_reads_both_classes_serial_only) ... ok
test_a_missing_or_an_extra_parent_raises (ComputeTargets.tests.test_gksource_parent_set_main.TestTheAssertion.test_a_missing_or_an_extra_parent_raises) ... ok
test_a_value_outside_the_sample_is_not_a_member (ComputeTargets.tests.test_gksource_parent_set_main.TestTheAssertion.test_a_value_outside_the_sample_is_not_a_member)
assemble_GkSource_values reads only the z_sources of its sample, so a value at another ... ok
test_a_value_that_does_not_say_its_parent_raises (ComputeTargets.tests.test_gksource_parent_set_main.TestTheAssertion.test_a_value_that_does_not_say_its_parent_raises) ... ok
test_an_empty_set_is_refused (ComputeTargets.tests.test_gksource_parent_set_main.TestTheAssertion.test_an_empty_set_is_refused) ... ok
test_another_row_at_one_z_source_raises_naming_it (ComputeTargets.tests.test_gksource_parent_set_main.TestTheAssertion.test_another_row_at_one_z_source_raises_naming_it) ... ok
test_nothing_is_computed_when_it_raises (ComputeTargets.tests.test_gksource_parent_set_main.TestTheAssertion.test_nothing_is_computed_when_it_raises) ... ok
test_the_rows_it_is_keyed_on_are_accepted (ComputeTargets.tests.test_gksource_parent_set_main.TestTheAssertion.test_the_rows_it_is_keyed_on_are_accepted) ... ok
test_a_labels_only_miss_names_none_and_cannot_be_computed (ComputeTargets.tests.test_gksource_parent_set_main.TestTheConstructor.test_a_labels_only_miss_names_none_and_cannot_be_computed) ... ok
test_a_labels_only_object_given_a_set_is_an_error (ComputeTargets.tests.test_gksource_parent_set_main.TestTheConstructor.test_a_labels_only_object_given_a_set_is_an_error) ... ok
test_a_malformed_set_is_refused (ComputeTargets.tests.test_gksource_parent_set_main.TestTheConstructor.test_a_malformed_set_is_refused) ... ok
test_an_object_to_be_computed_names_its_set (ComputeTargets.tests.test_gksource_parent_set_main.TestTheConstructor.test_an_object_to_be_computed_names_its_set) ... ok
test_it_carries_the_set_it_is_keyed_on_in_canonical_form (ComputeTargets.tests.test_gksource_parent_set_main.TestTheConstructor.test_it_carries_the_set_it_is_keyed_on_in_canonical_form) ... ok
test_nothing_wanted_reads_nothing (ComputeTargets.tests.test_gksource_parent_set_main.TestThePipelinesOwnLogic.test_nothing_wanted_reads_nothing) ... ok
test_the_helper_resolves_the_set_serial_only (ComputeTargets.tests.test_gksource_parent_set_main.TestThePipelinesOwnLogic.test_the_helper_resolves_the_set_serial_only) ... ok
test_the_row_records_the_set_assembled_which_is_the_set_it_was_keyed_on (ComputeTargets.tests.test_gksource_parent_set_main.TestThePipelinesOwnLogic.test_the_row_records_the_set_assembled_which_is_the_set_it_was_keyed_on) ... ok
test_each_gk_source_lookup_asks_for_labels_only (ComputeTargets.tests.test_gksource_parent_set_main.TestTheReaders.test_each_gk_source_lookup_asks_for_labels_only) ... ok

----------------------------------------------------------------------
Ran 71 tests in 2.443s

OK
!! GkWKBIntegration.build(): multiple results found when querying for GkWKBIntegration
!! GkSourceValue.build(): multiple results found when querying for GkSourceValue
```

- **Formatting.** `./venv/bin/python -m black --check` on the 23 Python files this commit touches:
  "23 files would be left unchanged".
- **Scope.** Staging by explicit path only (`git add --pathspec-from-file`); `git diff --cached
  --name-only` is in the final report.
- **No Ray, nothing under `var/`.** No module starts Ray, and each new one asserts it is not
  initialised at teardown. Stores are in memory or in temporary directories; the probes, the cost
  measurement and the golden capture ran under the scratch directory.

## The deliberate-breakage record

Each mutation was applied with `git apply` to the staged tree, so that `git diff` showed the
mutation alone; the `Datastore`, `ComputeTargets` and `RunRegistry` suites were run serially; and it
was reverted with `git apply -R`, after which `git diff --name-only -- '*.py'` was empty
(`run_mutations.sh`, in the scratch directory). AdaptiveLevin, CosmologyModels and LiouvilleGreen
were not run: none imports a mutated module. Each failing entry is one `FAIL:` or `ERROR:` line; a
sub-test is its own entry, with its parameters. The diffs are pasted from `git diff`, exactly as
applied. **In `ComputeTargets`, `test_wall_time_per_object` is the known flake; it failed in the
run of (ii) only, passed 3 of 3 alone afterwards on the unmutated tree, and is never a mutation's.**

**Replays.** Before the commit, each of the five blocks below was extracted from this file and
applied with plain `git apply`, and the three suites were run (the result is in the final report,
since this file cannot record a run that follows itself). After the commit they are replayed from
`git show HEAD:prompts/datastore-integrity/logs/09b-key-gksource-on-its-parent-set.md`; that result
goes to the orchestrator.

### Mutation (i) — `parents_digest` computed from the requested set, not the rows written (the charter's (iv))

```diff
diff --git a/Datastore/SQL/ObjectFactories/GkSource.py b/Datastore/SQL/ObjectFactories/GkSource.py
index ad3be98..46f9059 100644
--- a/Datastore/SQL/ObjectFactories/GkSource.py
+++ b/Datastore/SQL/ObjectFactories/GkSource.py
@@ -632,14 +632,7 @@ class sqla_GkSource_factory(SQLAFactoryBase):
         ]
 
         # the key column is computed from the rows written below, and from nothing else
-        parents_digest = GkSource_parents_digest(
-            (
-                member["z_source_serial"],
-                member["numeric_serial"],
-                member["wkb_serial"],
-            )
-            for member in parent_rows
-        )
+        parents_digest = GkSource_parents_digest(obj.parents)
 
         store_id = inserter(
             conn,
```

`Datastore`: `Ran 505 tests in 139.310s`, `FAILED (failures=1)`; 1 failing entry, 0 of them sub-tests:

- `FAIL` test_the_rows_written_are_the_set_assembled_not_the_set_asked_for (test_gksource_parent_set.TestTheRecordAgreesWithTheKey)

`ComputeTargets`: `Ran 635 tests in 147.658s`, `OK`.

`RunRegistry`: `Ran 199 tests in 70.435s`, `OK`.


### Mutation (ii) — `parents_digest` dropped from `build()`'s filter

```diff
diff --git a/Datastore/SQL/ObjectFactories/GkSource.py b/Datastore/SQL/ObjectFactories/GkSource.py
index ad3be98..bc30809 100644
--- a/Datastore/SQL/ObjectFactories/GkSource.py
+++ b/Datastore/SQL/ObjectFactories/GkSource.py
@@ -396,10 +396,6 @@ class sqla_GkSource_factory(SQLAFactoryBase):
             )
 
         # the parent set, through its digest (prompt 09b)
-        if not labels_only:
-            query = query.filter(
-                table.c.parents_digest == GkSource_parents_digest(parents)
-            )
 
         # require that the integration we search for has the specified list of tags
         count = 0
```

`Datastore`: `Ran 505 tests in 138.894s`, `FAILED (failures=3, errors=1)`; 4 failing entries, 0 of them sub-tests:

- `ERROR` test_two_sets_differing_in_one_row_give_two_rows_each_found_by_its_own (test_gksource_parent_set.TestKeyedOnItsSet)
- `FAIL` test_a_subset_is_not_the_set (test_gksource_parent_set.TestKeyedOnItsSet)
- `FAIL` test_found_by_its_own_set_and_missed_by_another (test_gksource_parent_set.TestKeyedOnItsSet)
- `FAIL` test_the_rows_written_are_the_set_assembled_not_the_set_asked_for (test_gksource_parent_set.TestTheRecordAgreesWithTheKey)

`ComputeTargets`: `Ran 635 tests in 158.221s`, `FAILED (failures=1)`; 1 failing entry, 0 of them sub-tests (one is the known wall-clock flake `test_wall_time_per_object`, which is not this mutation's: it passed in the other four mutations' runs and alone afterwards):

- `FAIL` test_wall_time_per_object (test_tk_wkb_phase.TestCost)

`RunRegistry`: `Ran 199 tests in 77.609s`, `OK`.


### Mutation (iii) — the reader mode takes the first of two matches (the charter's (vi))

```diff
diff --git a/Datastore/SQL/ObjectFactories/GkSource.py b/Datastore/SQL/ObjectFactories/GkSource.py
index ad3be98..608b96f 100644
--- a/Datastore/SQL/ObjectFactories/GkSource.py
+++ b/Datastore/SQL/ObjectFactories/GkSource.py
@@ -422,7 +422,7 @@ class sqla_GkSource_factory(SQLAFactoryBase):
                 candidates = []
                 for candidate in conn.execute(query):
                     candidates.append((candidate.serial, candidate.parents_digest))
-                if len(candidates) > 1:
+                if False:
                     raise MultipleResultsFound(
                         f"GkSource.build(): labels_only lookup matched {len(candidates)} stored "
                         f"rows (model serial={model_proxy.store_id}, k serial={k_exit.store_id}, "
@@ -432,7 +432,7 @@ class sqla_GkSource_factory(SQLAFactoryBase):
                         f"name the parent set."
                     )
 
-            row_data = conn.execute(query).one_or_none()
+            row_data = conn.execute(query).first()
 
             # the reader mode's one match carries the parent set its row records
             if labels_only and row_data is not None:
```

`Datastore`: `Ran 505 tests in 125.124s`, `FAILED (failures=1)`; 1 failing entry, 0 of them sub-tests:

- `FAIL` test_two_matches_are_refused_naming_their_digests_and_where_they_differ (test_gksource_parent_set.TestTheReaderMode)

`ComputeTargets`: `Ran 635 tests in 144.842s`, `OK`.

`RunRegistry`: `Ran 199 tests in 72.470s`, `OK`.


### Mutation (iv) — the parent set left out of `GkSource`'s inventory identity

```diff
diff --git a/Datastore/SQL/ObjectFactories/GkSource.py b/Datastore/SQL/ObjectFactories/GkSource.py
index ad3be98..d7740e9 100644
--- a/Datastore/SQL/ObjectFactories/GkSource.py
+++ b/Datastore/SQL/ObjectFactories/GkSource.py
@@ -880,19 +880,6 @@ class sqla_GkSource_factory(SQLAFactoryBase):
                 "k": Parent("wavenumber_exit_serial", "wavenumber_exit_time"),
                 "z_response": Parent("z_response_serial", "redshift"),
             },
-            parent_sets={
-                "parents": ParentSet(
-                    table="GkSource_parents",
-                    owner="gk_source_serial",
-                    members={
-                        "z_source": Parent("z_source_serial", "redshift"),
-                        "numeric": Parent(
-                            "numeric_serial", "GkNumericIntegration", nullable=True
-                        ),
-                        "wkb": Parent("wkb_serial", "GkWKBIntegration", nullable=True),
-                    },
-                )
-            },
             tags=("GkSource_tags", "parent_serial"),
             values=("GkSourceValue", "parent_serial"),
             validated=True,
```

`Datastore`: `Ran 505 tests in 127.160s`, `FAILED (failures=8, errors=2)`; 10 failing entries, 2 of them sub-tests:

- `ERROR` test_a_member_is_the_parents_identity_and_not_its_serial (test_gksource_parent_set.TestTheInventoryResolvesTheSet)
- `ERROR` test_resolve_names_every_member (test_gksource_parent_set.TestTheInventoryResolvesTheSet)
- `FAIL` test_a_store_without_the_table_is_named_absent (test_gksource_parent_set.TestTheInventoryResolvesTheSet)
- `FAIL` test_an_empty_set_is_a_named_problem (test_gksource_parent_set.TestTheInventoryResolvesTheSet)
- `FAIL` test_an_orphan_member_row_is_a_named_problem (test_gksource_parent_set.TestTheInventoryResolvesTheSet)
- `FAIL` test_an_unresolvable_member_is_a_named_problem (test_gksource_parent_set.TestTheInventoryResolvesTheSet)
- `FAIL` test_the_fixture_has_every_set_resolved (test_gksource_parent_set.TestTheInventoryResolvesTheSet)
- `FAIL` test_two_sources_differing_only_in_one_parent_are_two_records (test_gksource_parent_set.TestTheInventoryResolvesTheSet)
- `FAIL` test_each_identity_column_changes_the_records (test_store_inventory.TestIdentityColumnsMatter) (cls='GkSource', field='parents') — sub-test
- `FAIL` test_the_key_fields_are_the_lookup_columns (test_store_inventory.TestIdentityColumnsMatter) (cls='GkSource') — sub-test

`ComputeTargets`: `Ran 635 tests in 140.879s`, `OK`.

`RunRegistry`: `Ran 199 tests in 60.804s`, `FAILED (failures=1)`; 1 failing entry, 0 of them sub-tests:

- `FAIL` test_the_golden_fingerprint (test_store_fingerprint.TestGolden)


### Mutation (v) — a parentless `GkWKBValue.read_batch` succeeds on a pre-07 table (`numeric_serial` no longer selected)

```diff
diff --git a/Datastore/SQL/ObjectFactories/GkWKBIntegration.py b/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
index 91cd29d..dceb249 100644
--- a/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
+++ b/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
@@ -1402,10 +1402,6 @@ class sqla_GkWKBValue_factory(SQLAFactoryBase):
                 wkb_table.c.sin_coeff,
                 wkb_table.c.cos_coeff,
                 wkb_table.c.z_init,
-                # always selected, whether or not a parent is given, so that a store from before
-                # the column (prompt 07) is refused with the D4 message rather than read (prompt
-                # 09b)
-                wkb_table.c.numeric_serial,
                 wkb_table.c.z_source_serial,
                 redshift_table.c.z.label("z_source"),
                 redshift_table.c.source.label("z_source_is_source"),
```

`Datastore`: `Ran 505 tests in 121.310s`, `FAILED (failures=2)`; 2 failing entries, 2 of them sub-tests:

- `FAIL` test_gk_wkb_value_read_batch (test_gksource_parent_set.TestTheWKBValueReadsRefuseAPre07Store) (no parameters: the parentless read) — sub-test
- `FAIL` test_gk_wkb_value_read_batch (test_gksource_parent_set.TestTheWKBValueReadsRefuseAPre07Store) (parents_only='True') — sub-test

`ComputeTargets`: `Ran 635 tests in 142.804s`, `OK`.

`RunRegistry`: `Ran 199 tests in 62.642s`, `OK`.

Mutation (i) is caught by the one test that stores an object whose assembled set differs from its
key: in the pipeline `compute()`'s assertion keeps the two one set, so only that test can tell them
apart. Mutation (ii) is caught by the keyed-lookup tests (and the record test, whose lookup under
the other set now finds the row). Mutation (iii) is caught by the reader-mode refusal. Mutation
(iv) is caught across the inventory and by the golden. Mutation (v) is caught by the two
parentless forms of `GkWKBValue.read_batch` on a pre-07 table; the keyed forms still refuse,
because their filter names the column.

## Observations not acted on

1. **`GkSource.store()` does not check its members against their rows** → opened
   `[09b-gksource-members-are-not-checked-against-their-rows]` (board §3.3). Measured on the
   in-memory store of `test_gksource_parent_set.py`: a source stored with members
   `((21, 42, None), (22, 41, 52))` — Gk numeric 42 is at `z_source` 22, 41 at 21, Gk WKB 52 at
   23 — was written as given. Inert in the pipeline (`compute()`'s assertion).
2. **The report renders every member of a set** → opened
   `[09b-inventory-report-renders-every-member-of-a-parent-set]` (board §3.3). Measured on
   `build_full_store`: about 200 characters a member, verbose or not; a production `GkSource` has up
   to ~1,200.
3. **`GkSource.build()` prints `!! GkWKBIntegration.build(): multiple results found when querying for
   GkWKBIntegration`** on a `MultipleResultsFound`: the wrong class's name, pre-existing. Left as
   it is (a first draft corrected it; that was not this prompt's to do). Residue; no issue.
4. **`GkSource_parents` has no uniqueness constraint on `(gk_source_serial, z_source_serial)`.** The
   factory's own `register()` form has no way to declare one; `store()` writes one row per
   `z_source` of a canonical set, so only a hand-written row could repeat one, and the inventory
   would then record both members. No issue.
5. **`relabel_serials` leaves the fixture's `parents_digest` stale** (the sketch's finding). Harmless:
   the inventory never reads it (`test_parents_digest_is_not_identity`).
6. **An index named `ix_GkSource_parents_digest`** (on `GkSource.parents_digest`) reads like one of
   `GkSource_parents`' indexes. SQLAlchemy's naming; no collision (that table has no `digest`
   column).
7. **Nothing here has run under Ray or on a real store.** Prompt 10's end-to-end store is the first
   place `main.py`'s keyed stages and the readers' `labels_only` meet production data.

## State handed to the next prompt

- **Names.** The table `GkSource_parents` and the column `parents_digest` (README §4); the payload
  keys `parents` and `labels_only` of `GkSource`'s `build()`; the value reads' `parents_only` form
  and their `_parent_serial` attribute; `GkSource.parents` / `.assembled_parents` / `.labels_only`;
  `GkSource_parent_set`, `GkSource_parent_set_of_values`, `GkSource_parent_set_of_serials`,
  `GkSource_parents_differ` (target) and `GkSource_parents_digest` (factory); `main.py`'s
  `resolve_GkSource_parents`; `store_inventory.ParentSet`, `ClassInventory.parent_sets` and
  `.members`; the problems `empty-parent-set` and `orphan-member`.
- **For 09c.** `GkSourcePolicyData`'s identity now reaches its `GkSource`'s parent set. The
  `QuadSourceIntegral` stage resolves its missing items' Gk sets before its `GkSource` lookup
  (`:3413`); 09c's restructuring around the work list keeps that. `QuadSourceValue._build_impl_model`
  is 09c's (decision 3), in the shape `GkSourceValue._build_impl_model` now has. The witness to diff
  from is `schema_at_datastore-integrity-09b.json`, the golden
  `full_store_fingerprint_datastore-integrity-09b.json`, and `FINGERPRINT_FORMAT` goes 5 → 6.
- **The refusal pattern for a missing column** (07, 08) now also has
  `_refuse_a_store_that_predates_the_parent_set`, which matches a missing table too.
- **The guard's pins:** `EXPECTED_COVERAGE` 50; `KNOWN_UNRESOLVED_TABLES` 11; `KNOWN_BLIND` 1,
  `KNOWN_UNCHECKED_INSERTS` 13 and `KNOWN_BLIND_COLUMNS` 4; `KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS`
  empty. **`EXPECTED_SITES`** as in 09a (`TkNumericIntegration` 6, `GkSource` 6, `QuadSource` 3,
  `GkNumericIntegration` 3, `GkWKBIntegration` 3, `TkWKBIntegration` 3, `GkSourcePolicyData` 4,
  `QuadSourceIntegral` 2).
- **Baselines** after this commit: AdaptiveLevin 32, ComputeTargets **635**, CosmologyModels 39, Datastore **505**, LiouvilleGreen
  148 (1 skipped), RunRegistry 199.
