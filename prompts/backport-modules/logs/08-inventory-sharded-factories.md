# Log 08 — `inventory()` on the sharded-table factories, and the merge policies (F2c)

**Prompt:** prompts/backport-modules/08-inventory-sharded-factories.md
**Commit:** *(SHA intentionally not embedded — see `IMPLEMENTATION_STATE.md` §5 note 11)*
**Date:** 2026-09-04
**Result:** COMPLETE

## What shipped

All 15 sharded classes now have `@staticmethod def inventory(conn, table, tables, *args,
**kwargs)`, matching the contract prompts 06/07 established, and `config/sharding.py` now has an
`inventory_config` entry for every one of them.

### Group A — compute targets with a `validated` flag (6 classes, labelled shape)

`TkNumericIntegration`, `TkWKBIntegration`, `GkNumericIntegration`, `GkWKBIntegration`, `GkSource`
(all in their respective `*.py` files alongside their `*Value` sibling), and `QuadSource`
(`QuadSource.py`).

Each returns the two-bucket shape prompt 07 established for `BackgroundModel`:

```python
{
    "validated": {"labels": [...], "earliest_timestamp": ..., "latest_timestamp": ...},
    "unvalidated": {"labels": [...], "earliest_timestamp": ..., "latest_timestamp": ...},
}
```

via a `_bucket(validated_value)` closure identical in shape to `BackgroundModel.inventory`'s,
condition `table.c.validated == validated_value`, `func.min`/`func.max` on `table.c.timestamp` per
bucket. **No `NULL`-as-unvalidated handling was added** — unlike `BackgroundModel`, every one of
these six classes' `validated` column is registered `nullable=False, default=False` (confirmed by
reading each `register()` before writing), so there is no legacy-`NULL` case to guard against.

**Labels, one raw list entry per row (not deduplicated — see Deviations), built from the
foreign-key serials each `register()` actually has:**

| Class | Label fields |
|---|---|
| `TkNumericIntegration`, `TkWKBIntegration`, `GkNumericIntegration`, `GkWKBIntegration`, `GkSource` | `wavenumber_exit_serial`, `model_serial`, `atol_serial`, `rtol_serial` |
| `QuadSource` | `model_serial`, `q_wavenumber_exit_serial`, `r_wavenumber_exit_serial` (no atol/rtol — `QuadSource` has none) |

Format: `f"wavenumber_exit={v}, model={v}, atol={v}, rtol={v}"` (or the `QuadSource` equivalent
with `q_wavenumber_exit`/`r_wavenumber_exit`). Serials are left unresolved to actual values, per the
prompt's explicit steer — resolving would cost a query per row in an interactive reporting path;
prompt 09 can resolve them if it wants.

`GkSource` has no `solver_serial` (it is a source table, not an integration against a specific
`IntegrationSolver`), so its label omits it — confirmed by reading `GkSource.py`'s `register()`
before writing.

Config (`config/sharding.py`, shared dict `_compute_target_merge`, reused across all 6 classes,
mirroring upstream's single shared `_instanton_merge`):

```python
_compute_target_merge = {
    "validated": {"labels": "extend", "earliest_timestamp": "earliest", "latest_timestamp": "latest"},
    "unvalidated": {"labels": "extend", "earliest_timestamp": "earliest", "latest_timestamp": "latest"},
}
```

### Group B — compute targets with no `validated` column (3 classes, flat shape, no labels)

`GkSourcePolicyData`, `QuadSourceIntegral`, `OneLoopIntegral`.

All three confirmed (by reading `register()` directly) to have **no `validated` column at all** —
see "What I found checking `OneLoopIntegral`'s `validated` occurrence" below. All three return:

```python
{"count": <row count>, "earliest_timestamp": ..., "latest_timestamp": ...}
```

**Labels dropped for all three, not just `QuadSourceIntegral`/`OneLoopIntegral`.** The prompt names
only those two as "can be numerous" when discussing whether to drop labels, but `GkSourcePolicyData`
is one row per `(source, policy, wavenumber)` triple — bounded by the number of `GkSource` rows
times the number of registered policies, the same order of magnitude as the other two rather than a
small configuration table — so the same judgement was applied to it for consistency. Recorded here
as the prompt's own explicit ask ("any class where a label list was judged too expensive and
dropped").

Config (shared dict `_no_validated_merge`, reused across all 3):

```python
_no_validated_merge = {"count": "sum", "earliest_timestamp": "earliest", "latest_timestamp": "latest"}
```

### Group C — value tables (6 classes, count only)

`TkNumericValue`, `TkWKBValue`, `QuadSourceValue`, `GkNumericValue`, `GkWKBValue`, `GkSourceValue`
(each the sibling of a Group A class, in the same file).

Verbatim in shape to `BackgroundModelValue.inventory` from prompt 07:

```python
count = conn.execute(sqla.select(sqla.func.count()).select_from(table)).scalar()
return {"count": count}
```

No row iteration, no `GROUP BY`. Config (shared dict `_value_table_merge`, reused across all 6):
`{"count": "sum"}`.

## What I found checking `OneLoopIntegral`'s `validated` occurrence

The prompt flagged this as needing a check rather than an assumption. `OneLoopIntegral.py`'s
`register()` (line 78) has **no `validated` column** — confirmed by reading the full column list.
The one occurrence of the string `"validated"` in the file is at `OneLoopIntegral.py:235`, inside
`store()`'s insert payload dict: `"validated": False,` alongside `"label"`, `"model_serial"`, etc.
This is a **stray, dead dict key with no matching column** — not a reference to another table, and
not a live validated/unvalidated split hiding somewhere else in the file.

To confirm this is genuinely harmless rather than a live bug, I ran a throwaway check (not part of
the committed harness) against a real SQLAlchemy Core `insert()`: a dict key with no matching table
column is **silently dropped**, not an error —

```python
>>> conn.execute(sqla.insert(t), {'value': 1.0, 'validated': False})
# no error
```

So `OneLoopIntegral.store()`'s `"validated": False` has always been silently ignored by SQLAlchemy;
it does not raise, and it does not write anything. `OneLoopIntegral` is therefore correctly placed
in Group B (no validated/unvalidated split), and this finding is recorded under "Observations not
acted on" rather than fixed, per the prompt's explicit "do not change... any storage method" and the
campaign's own rule against fixing things a prompt did not ask for.

## Deviations from the prompt

### Group A labels are a raw per-row list, not deduplicated — IMPLEMENTATION CHOICE

Prompt 07's `BackgroundModel.inventory` deduplicates its `labels` list
(`sorted({row.label for row in label_rows})`), because there the label is a human-chosen name where
exact repeats represent the same named configuration re-validated, and collapsing them is the
readable behaviour. Group A's labels are different in kind: they are *synthesized* per-row
identifiers built from foreign-key serials that the prompt itself does not claim are unique per row
(a `(wavenumber_exit, model, atol, rtol)` combination can have multiple integration rows that differ
only in `z_init`/`z_source`, which is deliberately left out of the label per the prompt's "typically
the wavenumber and the model/policy/tolerance serials" guidance). Deduplicating here would silently
collapse genuinely distinct rows and make `len(labels)` an undercount with no way to recover the true
row count from the returned shape (Group A carries no separate `count` field, per the prompt's own
sketch). Chose a raw list, one entry per row, so the label list's length always equals the row count
in that bucket — verified directly in the cross-check harness (see Verification item 3: two rows
inserted with different `wavenumber_exit_serial` produced two distinct label strings, one per
bucket). This is an open choice the prompt explicitly left to the implementer ("A label should
identify the row..."; no dedup instruction either way) — flagged here in case prompt 09's report
finds a very long, duplicate-heavy label list unreadable for some class, in which case
deduplication (with a separate `count` field added to both the return shape and
`_compute_target_merge`) is the fix, at the cost of losing the current recover-count-for-free
property.

### `GkSourcePolicyData`'s labels dropped despite not being named in the prompt's "can be numerous" list — IMPLEMENTATION CHOICE

Covered above under Group B. The prompt names only `QuadSourceIntegral`/`OneLoopIntegral` when it
raises the numerous-rows concern, but does not say `GkSourcePolicyData` should keep a label list —
it only says "for a class with no validated split, use the flat shape" for all three, and separately
invites the implementer to judge dropping labels "if not appropriate" without restricting that
judgement to the two named classes. Chose to apply the same reasoning to all three for consistency,
on the volume grounds given above.

### No other deviations

Every other class follows its group's default shape with no structural surprises. Every `register()`
was read directly from the file before writing the corresponding `inventory` method and its
`inventory_config` entry, per the prompt's own warning that this is "the delicate one."

## Verification performed

1. **All 9 touched factory files plus `config/sharding.py` parse; `black --check` clean.**
   `./venv/bin/python3 -m py_compile` on all 10 files — exit 0 (`COMPILE OK`).
   `./venv/bin/python3 -m black` on the same 10 files — "10 files left unchanged" (no reformatting
   needed on first write).
2. **No `inventory` method takes `self`.**
   `grep -n "def inventory(self" Datastore/SQL/ObjectFactories/*.py` — empty.
   `grep -c "def inventory"` across all `ObjectFactories/*.py` sums to **28** (13 from prompt 07 +
   15 from this prompt), and a `grep -B1 "def inventory" ... | grep -c staticmethod` count also
   returns **28** — every one of the 28 factory `inventory` methods in the tree (not just this
   prompt's 15) is immediately preceded by `@staticmethod`.
3. **The config/factory cross-check that matters — actually run, not reasoned about.** Throwaway
   harness `scratchpad/verify_inventory_08.py` (not committed):
   - Builds a minimal in-memory SQLite table for each of the 15 sharded classes from that class's
     own `register()["columns"]` (plus `serial`/`version`/`timestamp` exactly as
     `Datastore._ensure_registered_schema` adds them; column `nullable` is relaxed to `True` in the
     test schema only, since the harness supplies just the columns each `inventory()` method
     actually reads, not a full row satisfying every `NOT NULL` constraint).
   - Calls `factory.inventory(conn, table, {})` against the **empty** table, then inserts 2–3
     synthetic rows and calls it again — both results printed for all 15 classes (full output
     captured below).
   - Asserts every class has an `inventory_config` entry (`config/sharding.py`), that the entry's
     shape (flat vs. labelled) matches the factory's actual return shape in **both** the empty and
     populated case, that the returned field set and the configured field set agree **in both
     directions** (an orphaned policy or an unconfigured returned field both fail the check), and
     that every policy string used is one of `_merge_queue`'s six (`extend`, `earliest`, `latest`,
     `sum`, `min`, `max`) applied to a value of a compatible Python type.
   - Result: **`ALL CHECKS PASSED`**, 0 problems across all 15 classes in both the empty and
     populated case. Full printed output:
     ```
     === TkNumericIntegration ===
       empty:      {'validated': {'labels': [], 'earliest_timestamp': None, 'latest_timestamp': None}, 'unvalidated': {'labels': [], 'earliest_timestamp': None, 'latest_timestamp': None}}
       populated:  {'validated': {'labels': ['wavenumber_exit=1, model=1, atol=1, rtol=1'], 'earliest_timestamp': ..., 'latest_timestamp': ...}, 'unvalidated': {'labels': ['wavenumber_exit=2, model=1, atol=1, rtol=1'], 'earliest_timestamp': ..., 'latest_timestamp': ...}}
     === TkNumericValue ===          empty: {'count': 0}   populated: {'count': 3}
     === TkWKBIntegration ===        (same shape as TkNumericIntegration)
     === TkWKBValue ===              empty: {'count': 0}   populated: {'count': 3}
     === QuadSource ===              (labelled shape, labels use model/q_wavenumber_exit/r_wavenumber_exit)
     === QuadSourceValue ===         empty: {'count': 0}   populated: {'count': 3}
     === GkNumericIntegration ===    (same shape as TkNumericIntegration)
     === GkNumericValue ===          empty: {'count': 0}   populated: {'count': 3}
     === GkWKBIntegration ===        (same shape as TkNumericIntegration)
     === GkWKBValue ===              empty: {'count': 0}   populated: {'count': 3}
     === GkSource ===                (same shape as TkNumericIntegration)
     === GkSourceValue ===           empty: {'count': 0}   populated: {'count': 3}
     === GkSourcePolicyData ===      empty: {'count': 0, 'earliest_timestamp': None, 'latest_timestamp': None}   populated: {'count': 2, ...}
     === QuadSourceIntegral ===      (same flat shape as GkSourcePolicyData)
     === OneLoopIntegral ===         (same flat shape as GkSourcePolicyData)

     ALL CHECKS PASSED
     ```
4. **Class-count parity, checked programmatically, not just by eye:**
   `len(sharded_tables) == len(inventory_config) == 15` and
   `set(sharded_tables.keys()) == set(inventory_config.keys())` — both `True`.
5. **The real cross-shard merge, exercised directly against `ShardedPool._merge_queue` (not
   reimplemented) for one class from each group — no live Ray cluster available, so this reproduces
   `ShardedPool.inventory()`'s own dispatch logic (label sniff, per-label merge) by hand against a
   `data_queue` built from two independent in-memory SQLite "shards," rather than `ray.get([shard...
   .remote(...) ...])`.** Throwaway harness `scratchpad/verify_inventory_08_merge.py` (not
   committed):
   - **`TkNumericIntegration`** (Group A): shard 1 has one `validated` row
     (`wavenumber_exit=1`, timestamp day 1), shard 2 has one `validated` row (`wavenumber_exit=2`,
     timestamp day 2). Merged result: `labels` contains both rows' label strings (set-equality
     checked, since `_merge_queue`'s pop-based accumulator does not guarantee shard order — see
     note below), `earliest_timestamp` = day 1, `latest_timestamp` = day 2, `unvalidated` bucket
     empty on both shards and stays empty merged. **Confirms**: `"extend"` merges the label lists
     across shards (not just returning one shard's), and `"earliest"`/`"latest"` pick the correct
     bound across shards, not within one.
   - **`GkSourcePolicyData`** (Group B): shard 1 has 3 rows (timestamp day 1), shard 2 has 2 rows
     (timestamp day 2). Merged `count` = **5** (confirmed **summed**, not one shard's value — the
     exact failure mode the prompt's verification item 5 warns a `"latest"`-style bug would produce
     and "look entirely plausible"), `earliest_timestamp`/`latest_timestamp` correct across shards.
   - **`TkNumericValue`** (Group C): shard 1 has 7 rows, shard 2 has 4 rows. Merged `count` = **11**
     (again confirmed summed, not `7` or `4` alone).
   - All three printed their per-shard and merged results and asserted the expected values;
     script exits with `ALL MERGE CHECKS PASSED`. This exercises `ShardedPool._merge_queue` and the
     labelled/flat dispatch sniff for real, against real per-shard `inventory()` output — the one
     thing check 3 above does not exercise (check 3 calls a single class's `inventory()` once per
     database, never merges across two).
   - **Not exercised**: `ShardedPool.inventory()`'s own method body (the `ray.get([shard.inventory
     .remote(...) ...])` fan-out, the replicated-vs-sharded dispatch branch, and the
     `RuntimeError`s for missing config) — that machinery was already verified by prompt 06's own
     harness (`scratchpad/verify_inventory_dispatch.py`, per its log) against a `ShardedPool`
     instance with `_inventory_config` populated by hand; this prompt adds no new sharded factory
     that exercises a *new* code path in `ShardedPool.inventory()` itself, only new entries in the
     `inventory_config` dict and new per-shard data for `_merge_queue` to combine, both covered
     directly by this check.
6. **Not run: a real multi-shard `ShardedPool` under live Ray.** No Ray cluster is available in
   this environment (same constraint every prompt in this campaign has recorded). Per the prompt's
   verification item 5 ("If you cannot, open a §3 issue for prompt 10"), a §3 issue is opened below.

## Observations not acted on

- **`OneLoopIntegral.store()`'s dead `"validated": False` insert key** (see "What I found" section
  above) — harmless (SQLAlchemy Core silently drops unmatched dict keys on `insert()`, confirmed
  directly), but it is dead code suggesting either a partially-reverted feature or a copy-paste
  leftover from a class that does have a `validated` column (e.g. `QuadSource`, whose `store()` in
  the same campaign sets `"validated": False` for a column that actually exists). Not fixed:
  `OneLoopIntegral.py`'s `store()` is a storage method, and the prompt explicitly says "do not change
  ... any storage method to make an inventory nicer" — this isn't even for the inventory's sake, but
  the same restraint applies since it is unrelated to this prompt's task. Worth a standalone
  cleanup commit if the maintainer wants it removed.
- **Group A's un-deduplicated label lists could grow large for a heavily-populated shard** (see
  Deviations above) — no cap or truncation was added, since no real populated datastore exists in
  this tree to judge actual scale against (per `IMPLEMENTATION_STATE.md` §5 note 1). If prompt 09's
  report turns out to render an unreadably long label list for some class, the fix is documented
  above (deduplicate and add a `count` field, updating `_compute_target_merge` to match).
- **No cross-shard check that every shard returns the same top-level key set** for a Group A/B
  class (e.g. if a future edit made one shard return `"validated"`/`"unvalidated"` and another
  returned a third bucket) — this is the same gap prompt 06's log already flagged as inherited
  behaviour (`data_queue[0]` is the sole reference for the label set), not something newly
  introduced here. All 15 factories in this prompt return a fixed, caller-independent key set (no
  `wavenumber`-style caller-dependent growth), so the gap is not actually reachable for any class
  this prompt added — recorded for completeness only.

## State handed to the next prompt

- **`config/sharding.py` now exports `inventory_config`** with all 15 sharded classes, built from
  three shared dicts (`_compute_target_merge`, `_no_validated_merge`, `_value_table_merge`) rather
  than 15 independent literals — prompt 09 can rely on `inventory_config[class_name]` resolving for
  every class in `sharded_tables`, and does not need its own fallback for an unconfigured sharded
  class (though it must still handle the *replicated*-class case, which never touches
  `inventory_config` at all, per prompt 06's log).
- **Every Group A/Group B class's label content is unresolved foreign-key serials, not values** —
  if prompt 09's report wants to show, say, an actual wavenumber in 1/Mpc instead of a bare serial
  number, it will need to resolve those serials itself (a `wavenumber_exit_time`/`wavenumber` lookup
  per distinct serial, batched to avoid one query per row) or accept the serial form as-is. This
  prompt deliberately did not do that resolution, per its own instructions.
- **Group A's `labels` lists are per-row, not deduplicated** (see Deviations) — prompt 09 should not
  assume `len(labels)` after a merge is "number of distinct configurations"; it is "number of rows
  across all shards in that bucket."
- **`OneLoopIntegral.py`'s dead `"validated"` insert key** is now formally documented (see
  Observations) rather than silently present with no record — if prompt 10 or a future maintainer
  investigates why `OneLoopIntegral` has no validated/unvalidated split despite the string
  `"validated"` appearing in the file, this log (and the code comment left in
  `OneLoopIntegral.inventory`) explains why.
- `IMPLEMENTATION_STATE.md` §5 note 11's no-self-referential-SHA convention followed: no SHA
  embedded in this log's header or in the status board's prompt-08 row.
