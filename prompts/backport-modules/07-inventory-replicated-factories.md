# Prompt 07 — `inventory()` on the replicated-table factories (F2, part 2 of 4)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit section:** §4 F2
**Depends on:** prompt 06 (hard — the dispatch contract this implements against)
**Files you may touch:** the factory modules under `Datastore/SQL/ObjectFactories/` listed below,
plus the log and status board.

**Read `prompts/backport-modules/logs/06-inventory-plumbing.md` before starting.** Prompt 06 fixed
the factory signature and the merge-policy vocabulary; its log is the authority if anything here
disagrees with what actually shipped.

---

## The contract

Every method you write is:

```python
    @staticmethod
    def inventory(conn, table, tables, *args, **kwargs):
        ...
        return { ... }
```

`@staticmethod`, because `SGWK` registers factory *classes* rather than instances — unlike upstream,
whose inventory methods take `self`. Copying an `SI` method verbatim will leave a stray `self` in
the signature and break the call. `tables` is always passed, with no `tables_arg` switch.

**Replicated tables need no merge configuration.** `ShardedPool.inventory` queries a single shard for
a replicated class and returns its dict unchanged, so nothing you write here is merged and no entry
in `inventory_config` is required. That is prompt 08's problem, not yours.

---

## The 13 classes, and the shape each one wants

From `config/sharding.py:replicated_tables`, with the `register()` flags verified against each
factory:

### Group A — scalar value types (8 classes)

`redshift`, `wavenumber`, `tolerance`, `store_tag`, `version`, `LambdaCDM`, `QCD_Cosmology`,
`IntegrationSolver`

Flat return: the list of values held, plus the timestamp range. Model on
`SI/Datastore/SQL/ObjectFactories/redshift.py`:

```python
        return {
            "earliest_timestamp": earliest_timestamp,
            "latest_timestamp": latest_timestamp,
            "values": values,
        }
```

Per-class notes:

- **`redshift`** — `values` is the list of `z`. Upstream has this method almost verbatim; it is the
  cleanest starting point.
- **`wavenumber`** — values are `k_inv_Mpc`. Note `wavenumber.read_table` takes a `units` argument;
  if you want inventory values in physical units, accept `units` through `*args` the way
  `SI`'s `_inventory_dimensionful` caller does (`SI/main.py:1266`, `pool.inventory(type_name, units)`)
  and return the unit name alongside, so the caller can format. Otherwise return raw stored values
  and say so. Your call — but prompt 09 has to format these, so record the decision clearly.
- **`version`** — registered `"timestamp": False`. **It has no timestamp column.** Return values
  only, and omit the timestamp fields rather than returning `None` for them.
- **`store_tag`** — values are tag names. Useful as a plain sorted list.
- **`tolerance`**, **`LambdaCDM`**, **`QCD_Cosmology`**, **`IntegrationSolver`** — small
  configuration tables. A label per row (name, or the distinguishing parameters) plus the timestamp
  range is more useful than a raw value list. Use your judgement on what identifies a row; check
  each `register()` for the available columns first.

### Group B — `wavenumber_exit_time` (1 class)

Registered `"timestamp": True, "version": True`. It is a derived/computed table keyed on a
wavenumber. A count plus timestamp range plus, if cheap, the number of distinct wavenumbers covered.
Do not build a per-row label list unless the table is small — check how many rows a typical run
produces before deciding.

### Group C — `BackgroundModel` (1 class)

Registered `"timestamp": True, "version": True, "validated"` present. This is a compute target and
takes the **labelled** shape, matching what prompt 08 will use for the sharded compute targets:

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

Keep the field names identical to the ones prompt 08 will use for the sharded compute targets. They
are not merged here, but a caller that formats both should not have to special-case this one.
Whatever you choose, **write it down in the log** — prompt 08 must match it, and prompt 09 formats
both.

Upstream `SI/ObjectFactories/FullInstanton.py:inventory` is the reference for this shape. Note it
also returns a `"versions": []` field that it never populates; do not carry that across unless you
populate it.

### Group D — `BackgroundModelValue` (1 class)

Registered `"timestamp": False` — no timestamp column. **Count only.** This is a high-volume child
table; do not build a label list.

```python
        return {"count": <row count>}
```

Use `select(func.count()).select_from(table)`, not `len(conn.execute(select(table)).fetchall())`.
The distinction matters — these tables can hold millions of rows and the naive form loads all of
them into the driver.

---

## Cost discipline

This is the thing most likely to go wrong, and it is worth more attention than getting the field
names elegant.

An inventory is a **reporting** query that a user runs interactively. It must not load a large table
into memory. Concretely:

- Use SQL aggregates — `func.count()`, `func.min(table.c.timestamp)`, `func.max(table.c.timestamp)` —
  in preference to iterating rows in Python, wherever the result is a scalar. Upstream's `redshift`
  method iterates every row to compute min/max timestamps; that is fine for a few dozen redshifts and
  wrong for anything larger.
- Only build a list when the list is genuinely small and genuinely wanted. A `values` list for ~50
  wavenumbers is useful; a label list for a million value rows is not.
- Select only the columns you need. Several of these tables are wide.

If you are unsure whether a table is small, look at how it is populated — a table written once per
configuration item is small; one written per (model, k, z) triple is not.

---

## Do not

- Do not modify `Datastore/SQL/ObjectFactories/base.py`. `inventory` stays optional and discovered
  via `hasattr`; making it abstract would oblige every factory in the tree to implement one.
- Do not add `inventory()` to the tag-association factories
  (`sqla_*TagAssociation_factory` / `sqla_QuadSourceTagAssocation_factory` — note upstream's spelling
  of the latter). They are join tables and an inventory of them is meaningless.
- Do not touch sharded-table factories — prompt 08.
- Do not add anything to `config/sharding.py` — prompt 08.
- Do not change `register()`, `build()`, `store()`, `validate()` or `validate_on_startup()` on any
  factory. If an inventory needs a column that is not currently stored, **do not add the column**;
  note it in the log and inventory what is there.

---

## Verification

1. Every touched file parses; `black --check` clean if available.
2. `grep -c "def inventory" Datastore/SQL/ObjectFactories/*.py` accounts for exactly the 13 classes
   you intended, and every one is preceded by `@staticmethod`. A missing decorator will not fail
   until called, so check it explicitly:
   ```bash
   grep -B1 "def inventory" Datastore/SQL/ObjectFactories/*.py | grep -c staticmethod
   ```
3. No `def inventory(self,` anywhere — that is the upstream-copy failure mode.
4. **Run what you can against a real database.** The shard files in the working tree are an old
   schema (`IMPLEMENTATION_STATE.md` §5 note 1), so they will not serve directly. But these are
   plain SQLAlchemy queries against a table object: if you can construct the schema and point a
   query at an empty table, do it — the empty case is the one most likely to be wrong (`None`
   timestamps, empty lists, zero counts) and the one prompt 08's merge has to cope with.
5. Confirm every method returns sensibly for an **empty table**. Prompt 06's `_merge_queue` handles
   `None` and empty lists, but only if you return them rather than crashing.

---

## Finish

1. Write `prompts/backport-modules/logs/07-inventory-replicated-factories.md` per `README.md` §5.1.
   This log carries a contract prompt 08 depends on, so it must state:
   - the **exact field names** used for the labelled compute-target shape (Group C);
   - the **exact field name** used for counts (Group D) — `"count"` or otherwise;
   - whether `wavenumber` takes `units` through `*args`, and what it returns if so;
   - any class where you departed from the group's default shape, and why;
   - any table you judged too large for a label list, with the reasoning.
2. Update `IMPLEMENTATION_STATE.md`: the prompt 07 row, the F2b item row, progress, "last updated",
   any §3 issues.
3. Commit. Suggested message:

```
Add inventory() to the replicated-table object factories

Second of four commits adding datastore-contents reporting. Implements the
factory side for the 13 replicated classes; the sharded classes follow in the
next commit.

Scalar value types (redshift, wavenumber, tolerance, store_tag, version,
LambdaCDM, QCD_Cosmology, IntegrationSolver) return their held values with a
timestamp range. BackgroundModel is a compute target and returns the labelled
validated/unvalidated shape that the sharded compute targets will also use.
BackgroundModelValue has no timestamp column and is a high-volume child
table, so it returns a row count only. version likewise has no timestamp
column and omits those fields rather than returning None.

Each method is a staticmethod taking (conn, table, tables, *args, **kwargs),
matching this tree's convention of registering factory classes rather than
instances -- upstream's equivalents are instance methods and cannot be copied
across unaltered.

Counts and timestamp ranges are computed with SQL aggregates rather than by
iterating rows, since an inventory is an interactive reporting query and
several of these tables are large.

No merge configuration is needed: ShardedPool.inventory serves a replicated
class from a single shard and returns its result unchanged.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```
