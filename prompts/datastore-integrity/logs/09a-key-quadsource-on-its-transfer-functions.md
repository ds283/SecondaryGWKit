# Log 09a — Key QuadSource on the transfer functions it is computed from

**Prompt:** [`prompts/datastore-integrity/09a-key-quadsource-on-its-transfer-functions.md`](../09a-key-quadsource-on-its-transfer-functions.md)
**Commit:** *(this commit)* — "Key QuadSource on the transfer functions it is computed from"
**Model:** Claude Sonnet 5.5
**Date:** 2026-09-30
**Result:** COMPLETE. K4 and K7's cross-shard half shipped, under D1, D4 and D5 and the four decisions
of the orchestrator note's §0.1–§0.4 (2026-09-30). No stop condition triggered: the Tk numeric rows
are `QuadSource`'s only direct inputs (`compute()` receives `Tq` and `Tr`, both
`TkNumericIntegration`), `main.py` resolves them before the existence query without changing what
work is scheduled, the inventory resolves the cross-shard parent in its one read-only pass, no
existing test outside §0.5's list changed, `EXPECTED_SITES` and the guard's pins moved only as §0.4
allows, and only the `QuadSource` fingerprint class moved.

Suites after this commit, serially, on the final staged tree:
- ComputeTargets **585 → 610** (the 25 of the new `test_quadsource_parent_main.py`). The known
  wall-clock flake `test_tk_wkb_phase.TestCost.test_wall_time_per_object` failed in that run. It
  fails 3 of 3 alone on this tree and 2 of 2 alone on a clean archive of `HEAD` at the same load
  (average 24-29; 0.0603 s and 0.0646 s against its 0.06 s limit), so it is the flake and not this
  commit's.
- Datastore **422 → 459** (the 37 of the new `test_quadsource_parent_key.py`);
- AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped) and RunRegistry 199,
  unchanged.

The index `docs/OPEN_ISSUES.md` goes from **98 to 99** rows, counted (`grep -c '^| \`\[' docs/OPEN_ISSUES.md`):
one issue opened, none closed.

## What shipped

**K4 — the key** (`Datastore/SQL/ObjectFactories/QuadSource.py`).
- **`Tr_serial` is indexed** (`index=True`); its comment says why it has no foreign key (it is on
  `r`'s shard, and a foreign key cannot cross a shard; serials are unique across shards). Neither
  column is new.
- **`build()` filters on `Tq_serial` and `Tr_serial`.** **A lookup names the parents by serial,
  not by object** (the prompt's choice). Why: the query needs only the serial; the compute target
  already carries serials (`_Tq_serial`, `_Tr_serial`) and `store()` writes them; `Tr` is on
  another shard, so an object would carry a row the source's shard cannot join to; and it matches
  07's `numeric_serial`. `main.py` resolves the Tk rows with `_do_not_populate` and passes
  `Tk.store_id`. A parent that is not an `int` (a `bool`, a float, an object) raises `TypeError`.
- **The D1 reader mode is `labels_only=True`**, named as 07 named it. It selects the pairs
  `(Tq_serial, Tr_serial)` of the lookup's own query; if more than one row matches it raises
  `MultipleResultsFound` naming **each pair** (`Tq_serial=1, Tr_serial=5; Tq_serial=2, Tr_serial=5`),
  and otherwise serves the one row or reports a miss. Neither the parents nor the mode: `RuntimeError`.
  One parent alone: the same. The parents and the mode together: `ValueError`.
- **`store()`** refuses (`ValueError`) an object carrying no `Tq_serial`/`Tr_serial`; otherwise
  unchanged.
- **`inventory_records`** declares `"Tq": Parent("Tq_serial", "TkNumericIntegration")` and
  `"Tr": Parent("Tr_serial", "TkNumericIntegration", cross_shard=True)`. Its comment, which called
  the two serials "provenance, not identity", now says what they are.

**K4 — the compute target** (`ComputeTargets/QuadSource.py`).
- The constructor takes `Tq_serial`, `Tr_serial` and `labels_only`, and exposes `Tq_serial` and
  `Tr_serial` properties. An object to be computed (`payload=None`, not labels-only) **requires both**
  (`ValueError`). `labels_only=True` is the reader's parentless miss: no serials, and it cannot be
  computed. A stored object takes them from its payload, as before.
- **`compute()` asserts** that the `Tq` and `Tr` it receives are the rows the source is keyed under,
  after the availability checks and before anything is computed. A mismatch raises `RuntimeError`
  naming both pairs: `Tq store_id=7 but Tq_serial=5, and Tr store_id=6 but Tr_serial=6`.
- `compute()` used to end by **overwriting** `_Tq_serial` and `_Tr_serial` with the rows it had been
  handed, so what `store()` wrote was whatever it was given. Those two lines are removed (Deviation 3).

**K7 — the cross-shard parent** (`Datastore/store_inventory.py`).
- `Parent` gains `cross_shard: bool = False`. `ShardContext` gains `all_digests`, `{class: {serial:
  reference digest}}` over **every** shard, default empty. `read_records` resolves a cross-shard
  parent against it and a same-shard one against `digests`, exactly as before. The resolved identity
  is the parent's reference digest, never its serial. A parent class not yet built is a
  `RuntimeError`, as for a same-shard one.
- `read_inventory` builds `all_digests[class]` after each class is built on every shard, through
  `_merge_digests`: the union by serial. A serial that two shards hold with **different** digests is
  left out, so it resolves to nothing (`unresolved-parent`) instead of to one of them (Deviation 4).
- A cross-shard parent that resolves on no shard is an `unresolved-parent` problem, named as a
  same-shard one is. The store is read once and read-only.

**`main.py`** (table below). **Readers** and the **fingerprint format** (`FINGERPRINT_FORMAT` 3 → 4,
with a "Format 4" sentence) as below. **D5**: `Datastore/tests/data/schema_at_datastore-integrity-09a.json`,
captured with `PYTHONPATH=. ./venv/bin/python Datastore/tests/schema_description.py …`, and
`RunRegistry/tests/data/full_store_fingerprint_datastore-integrity-09a.json`. No older witness or
golden is touched.

**Tests** (new):
- `Datastore/tests/test_quadsource_parent_key.py`, 37 tests: §3 tests 1, 2, 3, 5 (lookup and
  inventory), 6 and 7, and unit tests of the cross-shard `Parent` and `_merge_digests`.
- `ComputeTargets/tests/test_quadsource_parent_main.py`, 25 tests: §3 tests 4, 5 (assertion) and 8, the
  constructor, and the three readers.
- Each module's `tearDownModule` asserts `ray.is_initialized()` is false.

## The parent table

Shards from `config/sharding.py` (`"QuadSource": "q"`, `"TkNumericIntegration": "k"`), as the note's
§0.6 measured.

| Parent | Its shard | Foreign key | Declared to the inventory |
|---|---|---|---|
| `Tq` (`Tq_serial`): Tk numeric at `k = q` | `q`'s, the source's own | yes (since 06), `TkNumericIntegration.serial` | `Parent("Tq_serial", "TkNumericIntegration")`, same shard |
| `Tr` (`Tr_serial`): Tk numeric at `k = r` | `r`'s, not necessarily the source's | no, and it cannot; **now indexed** | `Parent("Tr_serial", "TkNumericIntegration", cross_shard=True)` |

## Every `main.py` site changed

Line numbers from `git show HEAD:main.py` (before) and `git show :main.py` of the staged tree
(after). `git diff --cached -U0 main.py` has six hunks: `@@ -1668,0 +1669,69 @@`, `@@ -1675,0 +1745,2 @@`,
`@@ -1821,0 +1893,2 @@`, `@@ -3308,43 +3380,0 @@`, `@@ -3415 +3445,2 @@`, `@@ -3429,0 +3461,54 @@`.

| Site | Before | After |
|---|---|---|
| `QuadSource` stage (`build_tensor_source_work`, def at `:1656`): the distinct `k` of the batch, and a serial-only `TkNumericIntegration` lookup for each, `_do_not_populate: True`, then a stop if any is unavailable | — | `parent_k` `:1673`, `Tk_serial_batch` `:1679`, its `object_get` `:1707`, `Tk_serials` `:1721` |
| The existence query's payload names `Tq_serial` and `Tr_serial` | `query_batch` `:1669`, existence `object_get_vectorized` at `:1703` | payload `:1745-1746`, vectorized call `:1774` |
| The full Tk lookup, missing items only (unchanged) | `Tk_lookup_batch` `:1732`, `object_get` `:1759` | `:1803`, `:1830` |
| The object to be computed: `Tq_serial=Tq.store_id`, `Tr_serial=Tr.store_id` | `object_get("QuadSource", …)` `:1817` | `:1888`, keywords `:1893-1894` |
| `QuadSourceIntegral` stage (def `:3108` → `:3181`): the `QuadSource` lookup, keyed on the Tk rows, **moved after** the Tk numeric lookup | `QuadSource_lookup_batch` `:3308`, `object_get_vectorized` `:3336` | `:3470`, payload `:3478-3479`, call `:3500` |
| The Tk numeric lookup, moved ahead unchanged, and `Tk_numeric_serials` built from its results | `Tk_numeric_lookup_batch` `:3370`, `object_get` `:3397` | `:3400`, `:3427`; `Tk_numeric_serials` `:3465` |
| The unavailable-Tk stop, moved with it (Deviation 1); only its comment changed | `Tk_numeric_unavailable` `:3416` | `:3447` |
| The `TkWKBIntegration` lookup after them | `Tk_WKB_lookup_batch` `:3430` | `:3515` |

`test_main_plumbing.EXPECTED_SITES`: **`TkNumericIntegration` 5 → 6**, the one added site being the
`QuadSource` stage's serial-only parent lookup (`main.py:1707`, class `TkNumericIntegration`, §0.4's
"parent-resolution site"). The `QuadSourceIntegral` stage adds none: it moved its existing Tk lookup.
`GkSource` 6, `QuadSource` 3, `GkNumericIntegration` 3, `GkWKBIntegration` 3, `TkWKBIntegration` 3,
`GkSourcePolicyData` 4, `QuadSourceIntegral` 2 and the rest are unchanged; no count fell.

**The read-back guard's pins did not move** (`python -c` over `test_factory_select_columns`, before and
after): `EXPECTED_COVERAGE` 47 → 47, `KNOWN_UNRESOLVED_TABLES` 11, `KNOWN_BLIND` 1,
`KNOWN_UNCHECKED_INSERTS` 13, `KNOWN_BLIND_COLUMNS` 4, `KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS`
empty. (The first draft of the labels-only branch selected from a `.subquery()` bound to a
variable, which added a `KNOWN_UNRESOLVED_TABLES` pair; it now iterates `conn.execute(query)`
directly, Deviation 5.)

## The `substitute_main_py()` output

`docs/handover/quadsource_atol_sweep.py` loaded by path with `importlib.util.spec_from_file_location`
and `substitute_main_py()` called on this tree (it only reads `main.py`). The five `** quadsource_atol_sweep:`
lines are its own, and it raises unless each literal occurs exactly once; the last five lines count each
literal in `main.py` and give its line:

```
** quadsource_atol_sweep: replaced 1 occurrence of 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_SOURCE_K_VALUES)' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_RESPONSE_K_VALUES)' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'tol=DEFAULT_QUADRATURE_ATOL' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'tol=DEFAULT_QUADRATURE_RTOL' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of '        qsi_work_batches = list(grouper(qsi_work_items, n=750, incomplete="fill"))' in main.py
substituted source: 165813 characters
count=1 line=3876 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_SOURCE_K_VALUES)'
count=1 line=3888 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_RESPONSE_K_VALUES)'
count=1 line=3777 'tol=DEFAULT_QUADRATURE_ATOL'
count=1 line=3778 'tol=DEFAULT_QUADRATURE_RTOL'
count=1 line=3670 'qsi_work_batches = list(grouper(qsi_work_items, n=750, incomplete="fill"))'
```

All five still match exactly once. `:3670` (the `qsi_work_batches` line), `:3777` and `:3778` moved by the
85 lines this commit adds above them (from `:3585`, `:3692`, `:3693`); the two k-grid literals, below
them, moved from `:3791` and `:3803` to `:3876` and `:3888`. The sweep script is not edited.

## The witness diff, in full (D5)

Pasted from `diff Datastore/tests/data/schema_at_datastore-integrity-08.json
Datastore/tests/data/schema_at_datastore-integrity-09a.json`, default format, unedited. **3 hunks, 10 changed
lines, 14 lines of output** (`grep -c '^[0-9]'` = 3, `grep -c '^[<>]'` = 10, `wc -l` = 14), all in
`classes.QuadSource` (which spans lines 5936-6302 of the new witness), each hunk located by the JSON
key that encloses it (scanned upward from each hunk):

| Kind | Hunk | Enclosing key |
|---|---|---|
| 1. the column's description: `"index": null` → `true` | `6139c6139` | `table.columns`, the entry named `Tr_serial` |
| 2. the index DDL | `6249a6250` | `table.index_ddl` |
| 3. the index description | `6260a6262,6268` | `table.indexes` |

No hunk is outside `QuadSource`'s `Tr_serial` index.

```
6139c6139
<       "index": null,
---
>       "index": true,
6249a6250
>      "CREATE INDEX \"ix_QuadSource_Tr_serial\" ON \"QuadSource\" (\"Tr_serial\")",
6260a6262,6268
>       "unique": false
>      },
>      {
>       "columns": [
>        "Tr_serial"
>       ],
>       "name": "ix_QuadSource_Tr_serial",
```

## The fingerprint re-capture (D5)

`FINGERPRINT_FORMAT` goes from 3 to 4. The golden was captured the way `test_store_fingerprint._FullStores`
builds its baseline: `build_full_store` into a temporary directory, `read_inventory(...)`,
`fingerprint_of(...)`, `taken` set to null, serialised as `json.dumps(indent=2, sort_keys=True) + "\n"`.
**The script was first run on a clean `git archive` of `HEAD` and reproduced
`full_store_fingerprint_datastore-integrity-08.json` byte for byte** (`cmp`). It lives in this prompt's
scratch directory and is not part of the tree.

- **Moved**, checked class by class from the two goldens (a script comparing `classes[name]`):

  | Class | Count | Tag-set counts | Digest | Why |
  |---|---|---|---|---|
  | `QuadSource` | 2 → 2 | [1, 1] → [1, 1] | `949a9bc8ff26437a…` → `e5f060c27009b97b…` | its key gains the `Tq` and `Tr` parents; both tag-set digests moved with it |

- **Not moved: the other 20 classes**, identical in count, digest and tag sets. Nothing keys on a
  `QuadSource`'s digest yet (`QuadSourceIntegral.source_serial` joins the key in 09c).
- `problems` is `{}` before and after. The overall digest `46685ed2201f7bcd…` → `efdaf6ef5faa1764…`.
- The diff of the two goldens, in full (`diff`, unedited): three digests of `QuadSource` and the
  overall digest and format:

```
158c158
<       "digest": "949a9bc8ff26437a9b5ea749fcb47d60b5ed2178bab33a8b2da54432700414a5",
---
>       "digest": "e5f060c27009b97b831df8ff19d8963adc73701532b8dddcef1a3a665d3e111f",
162c162
<           "digest": "1a6cbc696961514fd49d3e318ea98ceccb4c87b96a726c07cbdcd3b8a523cd3a",
---
>           "digest": "138f617fdb5ad22998c8b51e6a36542466cf5a3fe2467f40f94e41ef1fd70a92",
169c169
<           "digest": "db4de99a6f074cecbd235291df6038515f51d2c818356193b9c9931483630a33",
---
>           "digest": "4d4e7f2651a2edd65bdc94503619545a0572bca071dd5101dd3878206452b35f",
319,320c319,320
<   "digest": "46685ed2201f7bcdcc71e2c6fc1134d2ba6e4e6b956f1fd74424e192a2fa67c7",
<   "fingerprint_format": 3,
---
>   "digest": "efdaf6ef5faa176419837276e0a3847f4f4dae267de791bfc09976137662e069",
>   "fingerprint_format": 4,
```

- **`store retire` of a store written before this prompt must be run at a pre-09a SHA (D4).** Its
  recorded fingerprint is format 3 (or older); at this tree `compare_fingerprints` says the formats
  differ and does not compare, and an inventory recomputed here gives its `QuadSource` rows a
  different identity. `store fingerprint` and `store amend` at this tree record format 4.

## Old stores (D4)

`QuadSource` gains **no column**, so nothing here refuses an old store's `QuadSource` table.
`TestAnOldStoreIsRead` recreates the table from `schema_at_datastore-integrity-08.json`'s own DDL and
indexes (which has no index on `Tr_serial`): a keyed lookup finds its row, a keyed lookup for
another `Tq` misses, and the reader mode finds it. What changes for an old store is its
fingerprint's identity, above. A resume of `main.py` against an old store now finds its sources by
the serials `store()` recorded, so it finds them exactly when the Tk rows it resolves are the rows
they were computed from.

## `q == r`

`itertools.combinations_with_replacement` gives the diagonal, where `Tq` and `Tr` are one row.
- **The lookup** names that serial twice (`Tq_serial = Tr_serial = s`); nothing in the schema or the
  filter needs them to differ, and a diagonal source and an off-diagonal one under the same `q` are
  different rows (`TestQEqualsR`, Datastore).
- **The assertion** accepts the one row handed twice, refuses a diagonal source given a different
  row for `r`, and refuses an off-diagonal source given the diagonal (`TestQEqualsR`, ComputeTargets).
  `main.py` hands `compute()` the one cached object for both.
- **The inventory** resolves both parents to the one record's digest, so `key["Tq"] == key["Tr"]`, with
  no problem (`test_the_inventory_resolves_both_parents_to_the_one_record`).

## Readers (README §5 rule 11), and what now refuses

| Reader | Before | After |
|---|---|---|
| `main.py`, the `QuadSource` stage | looked up by labels | keyed on the Tk serials it resolves first |
| `main.py`, the `QuadSourceIntegral` stage's `QuadSource` lookup | labels | keyed on the Tk rows it looks up ahead of it (§0.2) |
| `extract_tensor_source_data.py` (`:372` at `HEAD`) | labels | `labels_only=True`: one match served, more refused, naming both pairs |
| `extract_QuadSourceIntegral_data.py`, its `QuadSource` lookup only (`:1172`) | labels | `labels_only=True` |
| `docs/source-remediation-verification/run_quadsource_integrals.py`, its `QuadSource` lookup only (`:291`) | labels | `labels_only=True`; it looks the `QuadSource` up before the Tk rows, so the mode rather than the keys |
| `docs/handover/quadsource_atol_sweep.py`, `docs/gktk-remedial/scoped_pipeline_run.py` | `exec` `main.py` | keyed through `main.py`; not edited. The sweep store under `var/` (not opened) has both serials, so it is not refused |
| `docs/source-remediation-verification/analyse_greens_and_source.py`, `tools/inventory_report.py`, `docs/tolerance-convergence/*` | raw SQL or table names | unchanged, not touched |
| `store retire`, `store fingerprint` comparisons | format 3 | format 4: an older recorded fingerprint is not compared |

**What now refuses:**
- **`extract_tensor_source_data.py`, `extract_QuadSourceIntegral_data.py` and
  `run_quadsource_integrals.py`, on a store holding two `QuadSource` rows under the same labels**,
  which they previously served one of. A store built once holds one, so they keep working.
- **`main.py`'s `QuadSource` stage now stops for any transfer function missing from its batch**, before
  its existence query (Deviation 2).
- **`store retire` and `store fingerprint` comparisons**, as above.
- **No audit probe newly fails** (below).

### The fourteen audit probes, before and after

Run at `HEAD` (a `git archive`, `venv` symlinked) and on a copy of the final staged tree, each in a
subprocess with `PYTHONPATH` at its root, both under the scratch directory: **the probes were not run
in the checkout**, since `replicated_write_fault_probe.py` writes a store beside itself. No probe opens
anything under `var/`: each builds its schema in memory or a store in a temporary directory
(`lookup_keys/common.py` mentions `var/` in a docstring only), and `replicated_serial_census.py` was run
with no argument, so it opened no store. Outputs compared after normalising the root path. **No probe
edited.**

| Probe | Exit before → after | Output |
|---|---|---|
| `fk_sweep_probe.py` | 0 → 0 | identical |
| `hot_journal_probe.py` | 0 → 0 | identical |
| `probe_01_tagged_read_batch.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `probe_01b_fix_shape.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `probe_02_readback_guard_by_hand.py` | 0 → 0 | **changed**: `QuadSourceValue`'s two `_build_impl_*` moved (`@559` → `@638`, `@659` → `@738`) and their row bindings (`576` → `655`, `693` → `772`); still clean |
| `probe_03_column_existence.py` | 0 → 0 | **changed**: "checked 1254" → "checked 1256 resolvable column references" |
| `probe_04_exit_time_and_solver.py` | 0 → 0 | identical |
| `probe_04b_exact_match_fix_shape.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `probe_05_value_lookups_and_read_batch.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `probe_06_oneloop_tags_and_redshift_zero.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `replicated_divergence_probe.py` | 0 → 0 | **changed**: two new lines, `unresolved-parent: QuadSource` on shard #0 (field Tr) and shard #1 (field Tq): the probe's own fixture's `QuadSource` rows name a `Tq` and `Tr` its store does not hold |
| `replicated_serial_census.py` | 0 → 0 | identical |
| `replicated_write_fault_probe.py` | 1 → 1 | identical but for the temporary directory name and a timestamp in one line (already failing at `HEAD`) |
| `tq_fk_probe.py` | 0 → 0 | identical |

None newly fails; four change output. The six that fail at `HEAD` (`probe_01`, `01b`, `04b`, `05`,
`06`, `replicated_write_fault_probe`) are unchanged. The four are named in the rule-11 list: `probe_02`,
`probe_03`, `replicated_divergence_probe` (whose fixture's `QuadSource` rows are now
`unresolved-parent`), and the timestamp line of `replicated_write_fault_probe`.

## Deviations from the prompt

1. **The unavailable-Tk stop in the `QuadSourceIntegral` stage moved with the Tk lookup, ahead of the
   `QuadSource` lookup.** `STRUCTURALLY REQUIRED`. The prompt moves the lookup; an unavailable row has
   no `store_id`, and the `QuadSource` lookup is now keyed on it, so the check that stops the stage
   for it has to come first. Its condition, message and effect are unchanged; only its comment says
   why it now precedes the `QuadSource` lookup.
2. **The `QuadSource` stage stops for any unavailable Tk row in its batch before the existence query**,
   with the message the same stage already used. `IMPLEMENTATION CHOICE`. An item whose Tk row is
   absent cannot have a `QuadSource` keyed on it, so it was always a missing item and always reached
   that stop; only the order changed. The later `.available` checks in the stage are now
   unreachable and are left as they were.
3. **`compute()` no longer overwrites `_Tq_serial`/`_Tr_serial` with the rows it is handed.**
   `STRUCTURALLY REQUIRED`: with the assertion they would be no-ops, and without it they were the
   silent replacement of the key that the prompt is closing.
4. **`ShardContext.all_digests` and `_merge_digests`, with the ambiguity rule** (a serial two shards
   hold differently resolves to nothing). `IMPLEMENTATION CHOICE`: the prompt asks for resolution
   against every shard; a serial split must not resolve to a chosen shard. Tested. It cannot arise
   for `TkNumericIntegration`, which is sharded, so it changes no record today.
5. **The labels-only candidates are read from `conn.execute(query)` directly**, not a `.subquery()`.
   `IMPLEMENTATION CHOICE`, made so that the read-back guard's pins did not move.
6. **`store()` refuses an object with no serials.** `IMPLEMENTATION CHOICE`, a guard for a labels-only
   miss handed to `store()` (`store()` is in the prompt's grant "if needed").
7. **Two tests of `test_quadsource.py`, not one:** the prompt names two `QuadSource(...)` constructions
   (`:372`, `:435`) "if the constructor comes to require them". Only `:372` (`payload=None`) needs
   them; `:435` has a payload that already carries both. See the edits table.

No `UNINTENDED DRIFT`. `git diff --cached -U0` shows only the hunks this log lists.

## Existing-test edits, by §0.5's 09a list

| File | Edit | Reason |
|---|---|---|
| `Datastore/tests/test_store_inventory.py` | `IDENTITY["QuadSource"]` gains `"Tq": ("QuadSource", 1, "Tq_serial", 3)` and `"Tr": ("QuadSource", 1, "Tr_serial", 3)`. Tk numeric 3 is real and on shard 0: the same shard as `QuadSource` 1's `Tq` (1), another shard from its `Tr` (2, shard 1), so each variation changes an identity and does not make the parent unresolved | the key's fields now include them (note's own choice of serial 3) |
| same | `NON_IDENTITY` loses `("QuadSource", 1, "Tq_serial", 9)` | `Tq_serial` is now identity |
| same | `test_a_tag_on_the_background_model_reaches_its_children` expects `QuadSource` too, and a comment says why | model 2's Tk numeric 3 is `QuadSource` 2's `Tr`; "model 2 is the parent of TkNumericIntegration 3 only" stays true |
| `ComputeTargets/tests/test_quadsource.py` | the `payload=None` construction (`test_store_truncates_z_sample_and_the_spline_range`) gains `Tq_serial=1, Tr_serial=2`; the line `obj._Tq_serial, obj._Tr_serial = 1, 2` after it goes | the constructor requires them; the manual assignment named the same serials |
| `Datastore/tests/test_schema_builder.py` | `WITNESS` → `schema_at_datastore-integrity-09a.json`; the docstring's "current one" sentence | D5 |
| `RunRegistry/tests/test_store_fingerprint.py` | `GOLDEN` → `full_store_fingerprint_datastore-integrity-09a.json` | D5 |
| `ComputeTargets/tests/test_main_plumbing.py` | `EXPECTED_SITES["TkNumericIntegration"]` 5 → 6, with a comment | §0.4: the one parent-resolution site |

No other existing test changed. `test_factory_select_columns.py` is untouched (no pin moved).

## Verification performed

- **The starting tree.** `git status` clean at `2252b8c`. `RunRegistry list` was not run: no long job
  was launched and nothing under `var/` was opened.
- **The six suites**, README §7's command, serially, on the staged tree: AdaptiveLevin 32 OK; CosmologyModels
  39 OK; LiouvilleGreen 148 OK (skipped=1); RunRegistry 199 OK; Datastore **459 OK**; ComputeTargets
  **610 run, 1 failure, the known flake** (above: it fails alone at `HEAD` too at this load).
- **The new modules, `-v`, unfiltered** (62 tests; the three `!!` lines are `build()`'s own print
  before it re-raises a refusal, and are expected):

```
test_a_keyed_lookup_finds_its_row (Datastore.tests.test_quadsource_parent_key.TestAnOldStoreIsRead.test_a_keyed_lookup_finds_its_row) ... ok
test_the_old_table_has_no_index_on_tr_and_both_serials (Datastore.tests.test_quadsource_parent_key.TestAnOldStoreIsRead.test_the_old_table_has_no_index_on_tr_and_both_serials) ... ok
test_the_reader_mode_finds_it_too (Datastore.tests.test_quadsource_parent_key.TestAnOldStoreIsRead.test_the_reader_mode_finds_it_too) ... ok
test_a_class_not_yet_built_is_an_error_not_a_silent_miss (Datastore.tests.test_quadsource_parent_key.TestCrossShardParentUnit.test_a_class_not_yet_built_is_an_error_not_a_silent_miss) ... ok
test_a_same_shard_parent_does_not (Datastore.tests.test_quadsource_parent_key.TestCrossShardParentUnit.test_a_same_shard_parent_does_not) ... ok
test_a_serial_no_shard_holds_is_unresolved (Datastore.tests.test_quadsource_parent_key.TestCrossShardParentUnit.test_a_serial_no_shard_holds_is_unresolved) ... ok
test_it_resolves_against_another_shards_record (Datastore.tests.test_quadsource_parent_key.TestCrossShardParentUnit.test_it_resolves_against_another_shards_record) ... ok
test_the_identity_is_the_digest_and_not_the_serial (Datastore.tests.test_quadsource_parent_key.TestCrossShardParentUnit.test_the_identity_is_the_digest_and_not_the_serial) ... ok
test_a_miss_carries_the_parents_it_was_asked_under (Datastore.tests.test_quadsource_parent_key.TestKeyedOnTq.test_a_miss_carries_the_parents_it_was_asked_under) ... ok
test_found_by_the_row_it_was_computed_from_and_missed_by_another (Datastore.tests.test_quadsource_parent_key.TestKeyedOnTq.test_found_by_the_row_it_was_computed_from_and_missed_by_another) ... ok
test_the_same_labels_under_two_parents_coexist_and_each_is_found (Datastore.tests.test_quadsource_parent_key.TestKeyedOnTq.test_the_same_labels_under_two_parents_coexist_and_each_is_found) ... ok
test_the_stored_row_records_the_serials_it_was_keyed_on (Datastore.tests.test_quadsource_parent_key.TestKeyedOnTq.test_the_stored_row_records_the_serials_it_was_keyed_on) ... ok
test_the_tags_are_still_part_of_the_key (Datastore.tests.test_quadsource_parent_key.TestKeyedOnTq.test_the_tags_are_still_part_of_the_key) ... ok
test_found_by_r_and_missed_by_another_row_for_r (Datastore.tests.test_quadsource_parent_key.TestKeyedOnTrAcrossShards.test_found_by_r_and_missed_by_another_row_for_r) ... ok
test_the_row_is_on_no_table_of_this_shard (Datastore.tests.test_quadsource_parent_key.TestKeyedOnTrAcrossShards.test_the_row_is_on_no_table_of_this_shard) ... ok
test_there_is_no_foreign_key_to_hold_it (Datastore.tests.test_quadsource_parent_key.TestKeyedOnTrAcrossShards.test_there_is_no_foreign_key_to_hold_it) ... ok
test_tr_is_indexed (Datastore.tests.test_quadsource_parent_key.TestKeyedOnTrAcrossShards.test_tr_is_indexed) ... ok
test_a_replicated_serial_with_one_digest_on_every_shard_is_kept (Datastore.tests.test_quadsource_parent_key.TestMergeDigests.test_a_replicated_serial_with_one_digest_on_every_shard_is_kept) ... ok
test_a_serial_two_shards_hold_differently_resolves_to_neither (Datastore.tests.test_quadsource_parent_key.TestMergeDigests.test_a_serial_two_shards_hold_differently_resolves_to_neither) ... ok
test_the_union_across_shards (Datastore.tests.test_quadsource_parent_key.TestMergeDigests.test_the_union_across_shards) ... ok
test_the_diagonal_and_an_off_diagonal_source_are_different_rows (Datastore.tests.test_quadsource_parent_key.TestQEqualsR.test_the_diagonal_and_an_off_diagonal_source_are_different_rows) ... ok
test_the_inventory_resolves_both_parents_to_the_one_record (Datastore.tests.test_quadsource_parent_key.TestQEqualsR.test_the_inventory_resolves_both_parents_to_the_one_record) ... ok
test_the_lookup_names_the_same_serial_twice (Datastore.tests.test_quadsource_parent_key.TestQEqualsR.test_the_lookup_names_the_same_serial_twice) ... ok
test_a_tr_on_no_shard_is_a_named_unresolved_parent (Datastore.tests.test_quadsource_parent_key.TestTheInventoryResolvesAcrossShards.test_a_tr_on_no_shard_is_a_named_unresolved_parent) ... ok
test_the_fixture_source_resolves_its_cross_shard_tr (Datastore.tests.test_quadsource_parent_key.TestTheInventoryResolvesAcrossShards.test_the_fixture_source_resolves_its_cross_shard_tr) ... ok
test_the_same_rows_with_the_same_tr_are_a_duplicate (Datastore.tests.test_quadsource_parent_key.TestTheInventoryResolvesAcrossShards.test_the_same_rows_with_the_same_tr_are_a_duplicate) ... ok
test_tq_resolves_on_its_own_shard_only (Datastore.tests.test_quadsource_parent_key.TestTheInventoryResolvesAcrossShards.test_tq_resolves_on_its_own_shard_only) ... ok
test_two_rows_differing_only_in_tr_are_two_records_with_different_identities (Datastore.tests.test_quadsource_parent_key.TestTheInventoryResolvesAcrossShards.test_two_rows_differing_only_in_tr_are_two_records_with_different_identities) ... ok
test_a_lookup_naming_neither_the_parents_nor_the_mode_raises (Datastore.tests.test_quadsource_parent_key.TestTheReaderMode.test_a_lookup_naming_neither_the_parents_nor_the_mode_raises) ... ok
test_a_parent_is_a_serial (Datastore.tests.test_quadsource_parent_key.TestTheReaderMode.test_a_parent_is_a_serial) ... ok
test_no_match_is_a_labels_only_miss (Datastore.tests.test_quadsource_parent_key.TestTheReaderMode.test_no_match_is_a_labels_only_miss) ... ok
test_one_match_is_returned_without_being_given_any_parent (Datastore.tests.test_quadsource_parent_key.TestTheReaderMode.test_one_match_is_returned_without_being_given_any_parent) ... ok
test_one_parent_alone_is_not_the_key (Datastore.tests.test_quadsource_parent_key.TestTheReaderMode.test_one_parent_alone_is_not_the_key) ... ok
test_the_parents_and_the_mode_together_are_an_error (Datastore.tests.test_quadsource_parent_key.TestTheReaderMode.test_the_parents_and_the_mode_together_are_an_error) ... ok
test_the_tags_still_select_under_the_mode (Datastore.tests.test_quadsource_parent_key.TestTheReaderMode.test_the_tags_still_select_under_the_mode) ... ok
test_two_matches_are_refused_naming_both_pairs (Datastore.tests.test_quadsource_parent_key.TestTheReaderMode.test_two_matches_are_refused_naming_both_pairs) ... ok
test_two_matches_differing_only_in_tr_are_refused_naming_them (Datastore.tests.test_quadsource_parent_key.TestTheReaderMode.test_two_matches_differing_only_in_tr_are_refused_naming_them) ... ok
test_the_construction_names_both_parents (ComputeTargets.tests.test_quadsource_parent_main.TestMainPyResolvesTheParents.test_the_construction_names_both_parents) ... ok
test_the_existence_query_names_both_parents (ComputeTargets.tests.test_quadsource_parent_main.TestMainPyResolvesTheParents.test_the_existence_query_names_both_parents) ... ok
test_the_first_tk_lookup_is_serial_only_and_the_second_is_not (ComputeTargets.tests.test_quadsource_parent_main.TestMainPyResolvesTheParents.test_the_first_tk_lookup_is_serial_only_and_the_second_is_not) ... ok
test_the_quadsource_stage_resolves_the_tk_rows_before_its_existence_query (ComputeTargets.tests.test_quadsource_parent_main.TestMainPyResolvesTheParents.test_the_quadsource_stage_resolves_the_tk_rows_before_its_existence_query) ... ok
test_the_source_integral_stage_keys_its_quadsource_lookup_on_them (ComputeTargets.tests.test_quadsource_parent_main.TestMainPyResolvesTheParents.test_the_source_integral_stage_keys_its_quadsource_lookup_on_them) ... ok
test_the_source_integral_stage_looks_the_tk_rows_up_before_its_quadsource_lookup (ComputeTargets.tests.test_quadsource_parent_main.TestMainPyResolvesTheParents.test_the_source_integral_stage_looks_the_tk_rows_up_before_its_quadsource_lookup) ... ok
test_the_source_integral_stage_resolves_the_serials_after_checking_availability (ComputeTargets.tests.test_quadsource_parent_main.TestMainPyResolvesTheParents.test_the_source_integral_stage_resolves_the_serials_after_checking_availability) ... ok
test_the_source_integral_stage_still_loads_full_tk_rows (ComputeTargets.tests.test_quadsource_parent_main.TestMainPyResolvesTheParents.test_the_source_integral_stage_still_loads_full_tk_rows) ... ok
test_the_stage_does_not_ask_for_the_reader_mode (ComputeTargets.tests.test_quadsource_parent_main.TestMainPyResolvesTheParents.test_the_stage_does_not_ask_for_the_reader_mode) ... ok
test_a_diagonal_source_refuses_a_different_row_for_r (ComputeTargets.tests.test_quadsource_parent_main.TestQEqualsR.test_a_diagonal_source_refuses_a_different_row_for_r) ... ok
test_an_off_diagonal_source_refuses_the_diagonal (ComputeTargets.tests.test_quadsource_parent_main.TestQEqualsR.test_an_off_diagonal_source_refuses_the_diagonal) ... ok
test_the_one_row_handed_twice_is_accepted (ComputeTargets.tests.test_quadsource_parent_main.TestQEqualsR.test_the_one_row_handed_twice_is_accepted) ... ok
test_an_unavailable_row_is_still_its_own_error (ComputeTargets.tests.test_quadsource_parent_main.TestTheAssertion.test_an_unavailable_row_is_still_its_own_error) ... ok
test_another_tq_raises_naming_both_serials (ComputeTargets.tests.test_quadsource_parent_main.TestTheAssertion.test_another_tq_raises_naming_both_serials) ... ok
test_another_tr_raises_naming_both_serials (ComputeTargets.tests.test_quadsource_parent_main.TestTheAssertion.test_another_tr_raises_naming_both_serials) ... ok
test_nothing_is_computed_when_it_raises (ComputeTargets.tests.test_quadsource_parent_main.TestTheAssertion.test_nothing_is_computed_when_it_raises) ... ok
test_swapped_rows_raise (ComputeTargets.tests.test_quadsource_parent_main.TestTheAssertion.test_swapped_rows_raise) ... ok
test_the_rows_it_is_keyed_on_are_accepted (ComputeTargets.tests.test_quadsource_parent_main.TestTheAssertion.test_the_rows_it_is_keyed_on_are_accepted) ... ok
test_the_serials_are_not_replaced_by_the_rows_it_is_handed (ComputeTargets.tests.test_quadsource_parent_main.TestTheAssertion.test_the_serials_are_not_replaced_by_the_rows_it_is_handed) ... ok
test_a_labels_only_miss_given_a_parent_is_an_error (ComputeTargets.tests.test_quadsource_parent_main.TestTheConstructor.test_a_labels_only_miss_given_a_parent_is_an_error) ... ok
test_a_labels_only_miss_names_neither_and_cannot_be_computed (ComputeTargets.tests.test_quadsource_parent_main.TestTheConstructor.test_a_labels_only_miss_names_neither_and_cannot_be_computed) ... ok
test_a_stored_object_carries_the_serials_the_row_records (ComputeTargets.tests.test_quadsource_parent_main.TestTheConstructor.test_a_stored_object_carries_the_serials_the_row_records) ... ok
test_an_object_to_be_computed_names_both (ComputeTargets.tests.test_quadsource_parent_main.TestTheConstructor.test_an_object_to_be_computed_names_both) ... ok
test_it_carries_the_serials_it_is_keyed_under (ComputeTargets.tests.test_quadsource_parent_main.TestTheConstructor.test_it_carries_the_serials_it_is_keyed_under) ... ok
test_each_quadsource_lookup_asks_for_labels_only (ComputeTargets.tests.test_quadsource_parent_main.TestTheReaders.test_each_quadsource_lookup_asks_for_labels_only) ... ok

----------------------------------------------------------------------
Ran 62 tests in 1.792s

OK
!! QuadSource.build(): multiple results found when querying for QuadSource (q=1e+05/Mpc, store_id=11 | r=1e+05/Mpc, store_id=12)
!! QuadSource.build(): multiple results found when querying for QuadSource (q=1e+05/Mpc, store_id=11 | r=1e+05/Mpc, store_id=12)
!! QuadSource.build(): multiple results found when querying for QuadSource (q=1e+05/Mpc, store_id=11 | r=1e+05/Mpc, store_id=12)
```

- **Formatting.** `./venv/bin/python -m black --check` on the 15 Python files this commit touches
  (the two new test modules included): "15 files would be left unchanged".
- **Scope.** Staging was by explicit path only; `git diff --cached --name-only` is in the final report.
- **No Ray, nothing under `var/`.** No module starts Ray and each new one asserts it is not
  initialised at teardown. Stores are in temporary directories or in memory. The probes ran in the
  scratch directory.

## The deliberate-breakage record

Each mutation was applied with `git apply` to the staged tree, so that `git diff` showed the
mutation alone, the `Datastore`, `ComputeTargets` and `RunRegistry` suites were run serially, and it
was reverted with `git apply -R` (afterwards `git diff --name-only` was empty). AdaptiveLevin,
CosmologyModels and LiouvilleGreen were not run: none imports a mutated module. Each failing entry is
one `FAIL:` or `ERROR:` line; a sub-test is its own entry. The diffs are pasted from `git diff`,
exactly as applied. **In `ComputeTargets`, `test_wall_time_per_object` is the known flake; it failed in the runs of (i), (ii) and (iii) and not in (iv), and is never the mutation's.**

**Replays.** Before the commit, each of the four blocks below was extracted from this file and applied
with plain `git apply`, and the suites were run (result in the final report, since this file cannot
record a run that follows itself). After the commit they are replayed from
`git show HEAD:prompts/datastore-integrity/logs/09a-key-quadsource-on-its-transfer-functions.md`;
that result goes to the orchestrator.

### Mutation (i) — `Tr_serial` dropped from `build()`'s filter

```diff
diff --git a/Datastore/SQL/ObjectFactories/QuadSource.py b/Datastore/SQL/ObjectFactories/QuadSource.py
index 2b0bd22..b593ec9 100644
--- a/Datastore/SQL/ObjectFactories/QuadSource.py
+++ b/Datastore/SQL/ObjectFactories/QuadSource.py
@@ -206,7 +206,6 @@ class sqla_QuadSource_factory(SQLAFactoryBase):
         if not labels_only:
             query = query.filter(
                 table.c.Tq_serial == Tq_serial,
-                table.c.Tr_serial == Tr_serial,
             )
 
         # require that the tensor source calculation we search for has the specified list of tags
```

`Datastore`: `Ran 459 tests in 172.496s`, `FAILED (failures=2)`; 2 failing entries, 0 of them sub-tests:

- `FAIL` test_found_by_r_and_missed_by_another_row_for_r (test_quadsource_parent_key.TestKeyedOnTrAcrossShards)
- `FAIL` test_the_lookup_names_the_same_serial_twice (test_quadsource_parent_key.TestQEqualsR)

`ComputeTargets`: `Ran 610 tests in 194.951s`, `FAILED (failures=1)`; 1 failing entries, 0 of them sub-tests (one is the known flake `test_wall_time_per_object`, which fails at `HEAD` under this load and is not this mutation's):

- `FAIL` test_wall_time_per_object (test_tk_wkb_phase.TestCost)

`RunRegistry`: `Ran 199 tests in 89.037s`, `OK`:


### Mutation (ii) — `Tr` declared a same-shard `Parent` in `inventory_records` (the charter's mutation (v))

```diff
diff --git a/Datastore/SQL/ObjectFactories/QuadSource.py b/Datastore/SQL/ObjectFactories/QuadSource.py
index 2b0bd22..b1f0f94 100644
--- a/Datastore/SQL/ObjectFactories/QuadSource.py
+++ b/Datastore/SQL/ObjectFactories/QuadSource.py
@@ -559,7 +559,7 @@ class sqla_QuadSource_factory(SQLAFactoryBase):
                 "q": Parent("q_wavenumber_exit_serial", "wavenumber_exit_time"),
                 "r": Parent("r_wavenumber_exit_serial", "wavenumber_exit_time"),
                 "Tq": Parent("Tq_serial", "TkNumericIntegration"),
-                "Tr": Parent("Tr_serial", "TkNumericIntegration", cross_shard=True),
+                "Tr": Parent("Tr_serial", "TkNumericIntegration"),
             },
             tags=("QuadSource_tags", "parent_serial"),
             values=("QuadSourceValue", "parent_serial"),
```

`Datastore`: `Ran 459 tests in 165.123s`, `FAILED (failures=38, errors=1)`; 39 failing entries, 15 of them sub-tests:

- `ERROR` test_verbose_shows_every_key_field_of_every_class (test_inventory_report.TestTheDisplaySaysWhatTheInventoryHolds) (cls='QuadSource') — sub-test
- `FAIL` test_every_class_with_its_count_validated_split_and_value_sum (test_inventory_report.TestTheDisplaySaysWhatTheInventoryHolds) (cls='QuadSource', verbose=False) — sub-test
- `FAIL` test_every_class_with_its_count_validated_split_and_value_sum (test_inventory_report.TestTheDisplaySaysWhatTheInventoryHolds) (cls='QuadSource', verbose=True) — sub-test
- `FAIL` test_the_title_and_the_problem_count (test_inventory_report.TestTheDisplaySaysWhatTheInventoryHolds)
- `FAIL` test_the_value_tables_named_are_the_ones_counted (test_inventory_report.TestTheDisplaySaysWhatTheInventoryHolds) (cls='QuadSource') — sub-test
- `FAIL` test_the_inventory_resolves_both_parents_to_the_one_record (test_quadsource_parent_key.TestQEqualsR)
- `FAIL` test_a_tr_on_no_shard_is_a_named_unresolved_parent (test_quadsource_parent_key.TestTheInventoryResolvesAcrossShards)
- `FAIL` test_the_fixture_source_resolves_its_cross_shard_tr (test_quadsource_parent_key.TestTheInventoryResolvesAcrossShards)
- `FAIL` test_the_same_rows_with_the_same_tr_are_a_duplicate (test_quadsource_parent_key.TestTheInventoryResolvesAcrossShards)
- `FAIL` test_tq_resolves_on_its_own_shard_only (test_quadsource_parent_key.TestTheInventoryResolvesAcrossShards)
- `FAIL` test_two_rows_differing_only_in_tr_are_two_records_with_different_identities (test_quadsource_parent_key.TestTheInventoryResolvesAcrossShards)
- `FAIL` test_other_tags_are_not_a_duplicate (test_store_inventory.TestDuplicates)
- `FAIL` test_each_identity_column_changes_the_records (test_store_inventory.TestIdentityColumnsMatter) (cls='QuadSource', field='model') — sub-test
- `FAIL` test_each_identity_column_changes_the_records (test_store_inventory.TestIdentityColumnsMatter) (cls='QuadSource', field='q') — sub-test
- `FAIL` test_each_identity_column_changes_the_records (test_store_inventory.TestIdentityColumnsMatter) (cls='QuadSource', field='r') — sub-test
- `FAIL` test_each_identity_column_changes_the_records (test_store_inventory.TestIdentityColumnsMatter) (cls='QuadSource', field='Tq') — sub-test
- `FAIL` test_the_key_fields_are_the_lookup_columns (test_store_inventory.TestIdentityColumnsMatter) (cls='QuadSource') — sub-test
- `FAIL` test_different_serial_assignments (test_store_inventory.TestKeysArePhysical) (cls='QuadSource') — sub-test
- `FAIL` test_replicated_serials_permuted_and_rows_moved (test_store_inventory.TestKeysArePhysical) (cls='QuadSource') — sub-test
- `FAIL` test_sharded_rows_on_different_shards (test_store_inventory.TestKeysArePhysical) (cls='QuadSource') — sub-test
- `FAIL` test_a_missing_column_the_key_does_not_need (test_store_inventory.TestOldStoresAndOrphans)
- `FAIL` test_a_missing_key_column_makes_the_class_incomplete (test_store_inventory.TestOldStoresAndOrphans)
- `FAIL` test_a_reference_that_cannot_be_resolved (test_store_inventory.TestOldStoresAndOrphans)
- `FAIL` test_a_tag_row_whose_parent_is_missing (test_store_inventory.TestOldStoresAndOrphans)
- `FAIL` test_a_tag_row_whose_tag_is_missing (test_store_inventory.TestOldStoresAndOrphans)
- `FAIL` test_an_absent_association_table (test_store_inventory.TestOldStoresAndOrphans)
- `FAIL` test_an_absent_replicated_association_table (test_store_inventory.TestOldStoresAndOrphans)
- `FAIL` test_an_absent_replicated_table (test_store_inventory.TestOldStoresAndOrphans)
- `FAIL` test_an_absent_table (test_store_inventory.TestOldStoresAndOrphans)
- `FAIL` test_an_absent_value_table (test_store_inventory.TestOldStoresAndOrphans)
- `FAIL` test_an_orphan_value_row (test_store_inventory.TestOldStoresAndOrphans)
- `FAIL` test_a_changed_row (test_store_inventory.TestReplicatedDivergence)
- `FAIL` test_a_changed_value_count (test_store_inventory.TestReplicatedDivergence)
- `FAIL` test_a_tag_on_the_background_model_reaches_its_children (test_store_inventory.TestTags)
- `FAIL` test_classes_with_tags_validated_and_values (test_store_inventory.TestTheFullStore)
- `FAIL` test_every_class_has_a_record (test_store_inventory.TestTheFullStore) (cls='QuadSource') — sub-test
- `FAIL` test_the_full_store_has_no_problem (test_store_inventory.TestTheFullStore)
- `FAIL` test_deleting_a_value_row_lowers_one_count_by_one (test_store_inventory.TestValueCounts) (table='QuadSourceValue') — sub-test
- `FAIL` test_the_counts_are_per_parent (test_store_inventory.TestValueCounts) (cls='QuadSource') — sub-test

`ComputeTargets`: `Ran 610 tests in 171.737s`, `FAILED (failures=1)`; 1 failing entries, 0 of them sub-tests (one is the known flake `test_wall_time_per_object`, which fails at `HEAD` under this load and is not this mutation's):

- `FAIL` test_wall_time_per_object (test_tk_wkb_phase.TestCost)

`RunRegistry`: `Ran 199 tests in 68.552s`, `FAILED (failures=4)`; 4 failing entries, 0 of them sub-tests:

- `FAIL` test_read_only_then_write_then_matches (test_store_fingerprint.TestFingerprintCommand)
- `FAIL` test_the_golden_fingerprint (test_store_fingerprint.TestGolden)
- `FAIL` test_the_shape (test_store_fingerprint.TestGolden)
- `FAIL` test_deleting_the_row_alone_is_also_a_problem_entry (test_store_fingerprint.TestLocalised)

### Mutation (iii) — the reader mode takes the first of two matches (the charter's (vi))

```diff
diff --git a/Datastore/SQL/ObjectFactories/QuadSource.py b/Datastore/SQL/ObjectFactories/QuadSource.py
index 2b0bd22..dcc7fa8 100644
--- a/Datastore/SQL/ObjectFactories/QuadSource.py
+++ b/Datastore/SQL/ObjectFactories/QuadSource.py
@@ -230,7 +230,7 @@ class sqla_QuadSource_factory(SQLAFactoryBase):
                 candidates = sorted(
                     (row.Tq_serial, row.Tr_serial) for row in conn.execute(query)
                 )
-                if len(candidates) > 1:
+                if False:
                     named = "; ".join(
                         f"Tq_serial={Tq}, Tr_serial={Tr}" for Tq, Tr in candidates
                     )
@@ -240,7 +240,7 @@ class sqla_QuadSource_factory(SQLAFactoryBase):
                         f"serial={r.store_id}): {named}. Refusing to choose one; name the parents."
                     )
 
-            row_data = conn.execute(query).one_or_none()
+            row_data = conn.execute(query).first()
         except MultipleResultsFound as e:
             print(
                 f"!! QuadSource.build(): multiple results found when querying for QuadSource (q={q.k.k_inv_Mpc:.5g}/Mpc, store_id={q.store_id} | r={r.k.k_inv_Mpc:.5g}/Mpc, store_id={r.store_id})"
```

`Datastore`: `Ran 459 tests in 154.831s`, `FAILED (failures=3)`; 3 failing entries, 0 of them sub-tests:

- `FAIL` test_the_tags_still_select_under_the_mode (test_quadsource_parent_key.TestTheReaderMode)
- `FAIL` test_two_matches_are_refused_naming_both_pairs (test_quadsource_parent_key.TestTheReaderMode)
- `FAIL` test_two_matches_differing_only_in_tr_are_refused_naming_them (test_quadsource_parent_key.TestTheReaderMode)

`ComputeTargets`: `Ran 610 tests in 175.067s`, `FAILED (failures=1)`; 1 failing entries, 0 of them sub-tests (one is the known flake `test_wall_time_per_object`, which fails at `HEAD` under this load and is not this mutation's):

- `FAIL` test_wall_time_per_object (test_tk_wkb_phase.TestCost)

`RunRegistry`: `Ran 199 tests in 78.397s`, `OK`:


### Mutation (iv) — `compute()`'s assertion removed

```diff
diff --git a/ComputeTargets/QuadSource.py b/ComputeTargets/QuadSource.py
index 8e42f06..b608444 100644
--- a/ComputeTargets/QuadSource.py
+++ b/ComputeTargets/QuadSource.py
@@ -500,14 +500,6 @@ class QuadSource(DatastoreObject):
 
         # a derived row is keyed on the rows its compute() receives, so they must be the rows it
         # was made to be keyed under (prompts/datastore-integrity, prompt 09a)
-        if Tq.store_id != self._Tq_serial or Tr.store_id != self._Tr_serial:
-            raise RuntimeError(
-                "QuadSource: compute() was handed TkNumericIntegration rows other than the ones "
-                f"this source is keyed on: Tq store_id={Tq.store_id} but Tq_serial="
-                f"{self._Tq_serial}, and Tr store_id={Tr.store_id} but Tr_serial="
-                f"{self._Tr_serial}"
-            )
-
         # ensure that data ingredients are compatible with our specification
         check_cosmology(Tq.model_proxy, self._model_proxy)
         check_cosmology(Tr.model_proxy, self._model_proxy)
```

`Datastore`: `Ran 459 tests in 144.547s`, `OK`:


`ComputeTargets`: `Ran 610 tests in 164.934s`, `FAILED (failures=6)`; 6 failing entries, 0 of them sub-tests:

- `FAIL` test_a_diagonal_source_refuses_a_different_row_for_r (test_quadsource_parent_main.TestQEqualsR)
- `FAIL` test_an_off_diagonal_source_refuses_the_diagonal (test_quadsource_parent_main.TestQEqualsR)
- `FAIL` test_another_tq_raises_naming_both_serials (test_quadsource_parent_main.TestTheAssertion)
- `FAIL` test_another_tr_raises_naming_both_serials (test_quadsource_parent_main.TestTheAssertion)
- `FAIL` test_nothing_is_computed_when_it_raises (test_quadsource_parent_main.TestTheAssertion)
- `FAIL` test_swapped_rows_raise (test_quadsource_parent_main.TestTheAssertion)

`RunRegistry`: `Ran 199 tests in 72.597s`, `OK`:


Mutation (i) is caught by two tests of the new module (the `Tr` lookup and the diagonal), and only by
those: the fixture-level tests never look a source up. Mutation (ii) is caught across the inventory, since
both fixture rows' `Tr` are cross-shard, and by the golden. Mutation (iii) is caught by the reader-mode
tests. Mutation (iv) is caught by the assertion tests, and by nothing in Datastore, which never calls
`compute()`.

## Observations not acted on

1. **`QuadSourceValue._build_impl_model` joins every `QuadSource` under the same labels** → opened
   `[09a-quadsourcevalue-model-read-joins-every-source-under-its-labels]` (board §3.3). Measured: with two
   validated sources under the same labels each holding a value at one `z`, it raises
   `MultipleResultsFound`. No caller in the tree. Outside 09a's files.
2. **`Tk_numeric_unavailable`'s message says "so their TkWKBIntegration cannot be looked up"**, and the
   `QuadSource` lookup is now also unreachable for that mode. The wording is in the stage's text that the
   prompt does not grant.
3. **The `QuadSource` stage's later `.available` checks are unreachable** (Deviation 2). Left as they were.
4. **`_UNDECLARED_REFERENCES["QuadSource"]` in `real_store_fixtures.py` still names `Tr_serial`.** It is
   still true: `Tr_serial` has no foreign key.
5. **`extract_QuadSourceIntegral_data.py` reads `QuadSource` values through `source.values`** and so
   depends on the refusal in `labels_only`; nothing else there looks a source up.
6. **Nothing here has run under Ray or on a real store.** Prompt 10's end-to-end store is the first
   place `main.py`'s keyed `QuadSource` stage and the readers' `labels_only` meet production data.

## State handed to the next prompt

- **Names.** The payload keys `Tq_serial`, `Tr_serial` and `labels_only` of `QuadSource`'s `build()`
  (the same spelling as 07's `labels_only`); `QuadSource.Tq_serial` / `.Tr_serial` properties;
  `Parent(cross_shard=True)`; `ShardContext.all_digests`; `_merge_digests`.
- **For 09b:** the cross-shard `Parent` exists; the set-valued form does not. 09b's Gk rows are all on
  `k`'s shard, so it may not need `cross_shard`. The witness to diff from is
  `schema_at_datastore-integrity-09a.json`, the golden `full_store_fingerprint_datastore-integrity-09a.json`,
  and `FINGERPRINT_FORMAT` goes 4 → 5.
- **The refusal pattern for a missing column** (07, 08) is unchanged; 09a needed none.
- **The guard's pins:** `EXPECTED_COVERAGE` 47; `KNOWN_UNRESOLVED_TABLES` 11; `KNOWN_BLIND` 1,
  `KNOWN_UNCHECKED_INSERTS` 13 and `KNOWN_BLIND_COLUMNS` 4; `KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS`
  empty. **`EXPECTED_SITES`:** `TkNumericIntegration` 6, the rest as in 08 (`GkSource` 6, `QuadSource` 3,
  `GkNumericIntegration` 3, `GkWKBIntegration` 3, `TkWKBIntegration` 3, `GkSourcePolicyData` 4,
  `QuadSourceIntegral` 2).
- **Baselines** after this commit: AdaptiveLevin 32, ComputeTargets **610**, CosmologyModels 39,
  Datastore **459**, LiouvilleGreen 148 (1 skipped), RunRegistry 199.
