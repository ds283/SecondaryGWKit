# Prompt 08 — `inventory()` on the sharded-table factories, and the merge policies (F2, part 3 of 4)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit section:** §4 F2
**Depends on:** prompts 06 (the dispatch and merge contract) and 07 (the return-shape conventions)
**Files you may touch:** the sharded-table factory modules under `Datastore/SQL/ObjectFactories/`,
and `config/sharding.py`, plus the log and status board.

**Read the logs from prompts 06 and 07 before starting.** They fix the factory signature, the
merge-policy vocabulary and the field names. This prompt's whole risk is drift between what a
factory returns and what the merge config expects, and those two logs are the authority.

---

## Why this is the delicate one

Everything here is merged across shards, so a factory's return shape and its `inventory_config`
entry have to agree **field by field**. A mismatch produces either a `RuntimeError` naming the
missing field (if prompt 06's guards work) or a wrong number (if they don't). There is no type
system holding these in step — only care, and the cross-check in the verification section below.

Write the config and the factory methods together, class by class, rather than doing all the
factories and then all the config.

---

## The 15 classes, in three groups

From `config/sharding.py:sharded_tables`, with `register()` flags verified against each factory.

### Group A — compute targets with a `validated` flag (6 classes)

`TkNumericIntegration`, `TkWKBIntegration`, `QuadSource`, `GkNumericIntegration`,
`GkWKBIntegration`, `GkSource`

All registered `"timestamp": True, "version": True` with a `validated` column. Use the **labelled**
shape that prompt 07 established for `BackgroundModel` — read its log for the exact field names and
match them:

```python
        return {
            "validated": {
                "labels": [...],
                "earliest_timestamp": ...,
                "latest_timestamp": ...,
            },
            "unvalidated": { ... },
        }
```

Reference: `SI/Datastore/SQL/ObjectFactories/FullInstanton.py:inventory`. Adapt, do not copy — the
`self` parameter must become a `@staticmethod`, and the label construction is `SGWK`-specific.

A label should identify the row to a human reading a report. The natural content is the foreign-key
serials that distinguish it — for these classes typically the wavenumber and the model/policy/
tolerance serials. Check each factory's `register()` for what is actually there. Serial numbers are
not especially readable, but resolving them to values means extra queries per row; **prefer serials
and let prompt 09 resolve them if it wants to**, and say so in the log.

Corresponding `inventory_config` entry — one shared dict, as upstream does with `_instanton_merge`:

```python
_compute_target_merge = {
    "validated": {
        "labels": "extend",
        "earliest_timestamp": "earliest",
        "latest_timestamp": "latest",
    },
    "unvalidated": {
        "labels": "extend",
        "earliest_timestamp": "earliest",
        "latest_timestamp": "latest",
    },
}
```

Every field the factory returns needs an entry. If you add a count alongside the labels, it needs
`"sum"`.

### Group B — compute targets without a `validated` flag (3 classes)

`GkSourcePolicyData`, `QuadSourceIntegral`, `OneLoopIntegral`

`GkSourcePolicyData` and `QuadSourceIntegral` have no `validated` column at all; `OneLoopIntegral`
mentions `validated` once — **check what that occurrence actually is** before assuming it has the
same validated/unvalidated split as Group A. It may be a reference to another table rather than its
own column.

For a class with no validated split, use the **flat** shape:

```python
        return {
            "count": ...,
            "labels": [...],
            "earliest_timestamp": ...,
            "latest_timestamp": ...,
        }
```

with a flat config to match:

```python
    "QuadSourceIntegral": {
        "count": "sum",
        "labels": "extend",
        "earliest_timestamp": "earliest",
        "latest_timestamp": "latest",
    },
```

Prompt 06's shape sniff distinguishes flat from labelled by testing whether a top-level value is a
`dict`, so mixing the two across classes is fine — but **do not mix them within one class**.

`QuadSourceIntegral` and `OneLoopIntegral` can be numerous. Judge whether a label list is
appropriate; if not, return counts and timestamps only and drop `labels` from both the return and
the config.

### Group C — value tables (6 classes)

`TkNumericValue`, `TkWKBValue`, `QuadSourceValue`, `GkNumericValue`, `GkWKBValue`, `GkSourceValue`

All registered `"timestamp": False`. **They have no timestamp column**, and they are the
highest-volume tables in the datastore — `GkSourceValue` holds one row per (source, z) sample.

**Count only.** No labels, no timestamps:

```python
        return {"count": <row count>}
```

```python
_value_table_merge = {"count": "sum"}
```

This is exactly why prompt 06 added the numeric `"sum"` policy: upstream's `_merge_queue` supports
only `extend` / `earliest` / `latest` and would raise `RuntimeError` on an `int`.

Use `select(func.count()).select_from(table)`. Do **not** fetch rows and take `len`. If a per-parent
breakdown seems appealing, resist it — that is a `GROUP BY` over millions of rows on every shard,
for a report.

---

## Task — `config/sharding.py`

Add `inventory_config` after `read_table_config` (which prompt 04 re-keyed by class name; keep the
two visually parallel). Include the explanatory comment upstream carries, extended for the numeric
policies prompt 06 added:

```python
# Merge policies for pool.inventory() calls on sharded tables.
# Each field in the factory's inventory() return value needs a merge policy:
#   lists/sets → "extend"
#   datetimes  → "earliest" or "latest"
#   numbers    → "sum" (also "min"/"max")
```

Only **sharded** classes go in `inventory_config`. Replicated classes are served from a single shard
and are never merged; adding them would be harmless but misleading, so don't.

---

## Do not

- Do not modify `Datastore/SQL/ObjectFactories/base.py`.
- Do not add `inventory()` to the tag-association factories.
- Do not revisit prompt 07's replicated factories. If you find a genuine inconsistency between what
  07 shipped and what 08 needs, **do not silently change 07's work** — record it in the log, open a
  §3 issue, and adapt on this side if you can. If you cannot, say so.
- Do not add a `GROUP BY`-per-parent breakdown to any value table.
- Do not change `register()` or any storage method to make an inventory nicer.

---

## Verification

1. Every touched file parses; `black --check` clean if available.
2. Every new `inventory` is a `@staticmethod` and none takes `self`:
   ```bash
   grep -n "def inventory(self" Datastore/SQL/ObjectFactories/*.py   # must be empty
   ```
3. **The cross-check that matters.** Write a throwaway script (scratchpad, not committed) that
   imports `config.sharding.inventory_config`, and for every sharded class asserts:
   - the class has an entry in `inventory_config`;
   - the entry's shape (flat versus labelled) matches what the factory's `inventory` returns;
   - every field the factory returns has a policy, and every policy names a field the factory
     returns — **in both directions**, since an orphaned policy is a sign the factory drifted;
   - every policy string is one `_merge_queue` actually implements.

   The cleanest way to get the factory's real return shape is to run each `inventory()` against an
   empty in-memory table built from its own `register()` columns. That also exercises the empty case,
   which is the one most likely to be wrong. Record the output.
4. Every sharded class in `config/sharding.py:sharded_tables` (15 of them) has both an `inventory()`
   and an `inventory_config` entry. Count both; they should agree.
5. If you can build a real multi-shard datastore, call `pool.inventory(...)` for one class from each
   of the three groups and confirm the merge produces sensible values — in particular that a `count`
   is the **sum** across shards and not one shard's value. If you cannot, open a §3 issue for
   prompt 10.

---

## Finish

1. Write `prompts/backport-modules/logs/08-inventory-sharded-factories.md` per `README.md` §5.1.
   Record:
   - the shape chosen per class, and any class that departs from its group's default;
   - what a label contains, and whether serials are left unresolved;
   - any class where a label list was judged too expensive and dropped;
   - what you found when you checked `OneLoopIntegral`'s `validated` occurrence;
   - the full output of the config/factory cross-check in verification step 3.
2. Update `IMPLEMENTATION_STATE.md`: the prompt 08 row, the F2c item row, progress, "last updated",
   any §3 issues.
3. Commit. Suggested message:

```
Add inventory() to the sharded-table factories and merge policies

Third of four commits adding datastore-contents reporting. Implements the
factory side for the 15 sharded classes and the per-field merge policies that
ShardedPool.inventory needs to combine per-shard results.

The compute targets with a validated flag (TkNumericIntegration,
TkWKBIntegration, QuadSource, GkNumericIntegration, GkWKBIntegration,
GkSource) return the labelled validated/unvalidated shape introduced for
BackgroundModel, merged by extending the label lists and taking the earliest
and latest timestamps across shards. Labels carry the distinguishing foreign
key serials rather than resolving them, which would cost a query per row in
what is an interactive reporting path.

The compute targets without a validated flag (GkSourcePolicyData,
QuadSourceIntegral, OneLoopIntegral) return the flat shape.

The six value tables are registered with "timestamp": False and are the
highest-volume tables in the datastore, so they return a row count and
nothing else, summed across shards. Counts use a SQL aggregate rather than
fetching rows.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```
