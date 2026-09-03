# Prompt 04 — Replace the generated read-table methods with a `read_table()` service (B3)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit sections:** §3 B3, §5 X3, §5 X4, §8 checklist items 5–6
**Depends on:** prompt 02 — you cannot reopen an existing sharded datastore to verify anything until
B2 is fixed. Do not run this prompt before 02 has landed.
**Files you may touch:** `Datastore/SQL/Datastore.py`, `Datastore/SQL/ShardedPool.py`,
`config/sharding.py`, and the six `extract_*.py` scripts listed below, plus the log and status board.

**This is the largest change in the campaign.** It replaces a broken API with a different one and
touches 17 call sites. Land it on its own.

---

## What is wrong

The `read_table_config` machinery generates read-table methods at construction time. It is broken in
**four** independent ways.

The audit lists three, at `Datastore/SQL/Datastore.py:226-237` and
`Datastore/SQL/ShardedPool.py:188-201`:

1. `for method_name, method_config in read_table_config:` iterates a **dict**, which yields keys
   (strings). Unpacking `"read_wavenumber_table"` into two names raises
   `ValueError: too many values to unpack (expected 2)`. This fires during construction whenever
   `read_table_config` is not `None` — which is every `extract_*.py` script.
2. `setattr(self, method_name, wrapper)` binds a plain function to the *instance*, so `self` is
   never passed. `pool.read_wavenumber_table(units=…)` would bind `units` to the `self` parameter.
3. `wrapper` closes over the loop variables `method_name` / `method_config` by reference, so every
   generated method would resolve to the last config entry.

The planning pass found a fourth, in the consumer:

4. `Datastore._generic_read_table` already passes `tables=self._tables` unconditionally
   (`Datastore.py:761`), while the generated wrapper *also* injects `kwargs["tables"] = self._tables`
   when `tables_arg` is set (`Datastore.py:230-232`). Even with faults 1–3 repaired, a
   `tables_arg: True` class would raise
   `TypeError: read_table() got multiple values for keyword argument 'tables'`.

The whole feature is therefore dead. Do not try to repair it in place — replace it with the upstream
design, which removes all four faults by construction.

---

## Task 1 — `Datastore.read_table`

Replace the constructor block at `Datastore/SQL/Datastore.py:226-237` with a single stored config:

```python
        self._read_table_config: Optional[ReadTableConfigType] = read_table_config
```

and replace `_generic_read_table` (`Datastore.py:740-763`) with an explicit `read_table` method
modelled on `SI/Datastore/SQL/Datastore.py:720-762`:

```python
    def read_table(self, cls, *args, **kwargs):
        if self._read_table_config is None:
            raise RuntimeError("Datastore: the read_table service is not configured")

        if isinstance(cls, str):
            class_name = cls
        else:
            class_name = cls.__name__

        if class_name not in self._read_table_config:
            raise RuntimeError(
                f'Datastore: the read_table service is not available for objects of class "{class_name}"'
            )

        with ProfileBatchManager(
            self._profile_batcher, f"read_table[{class_name}]"
        ) as mgr:
            self._ensure_registered_schema(class_name)
            record = self._schema[class_name]

            tab = record["table"]
            factory = self._factories[class_name]

            if not hasattr(factory, "read_table"):
                raise RuntimeError(
                    f'Datastore: the object factory for "{class_name}" does not provide a read_table service'
                )

            config = self._read_table_config[class_name]
            if config.get("tables_arg", False):
                kwargs["tables"] = self._tables

            with self._engine.begin() as conn:
                objects = factory.read_table(conn, tab, *args, **kwargs)

            return objects
```

Points to get right:

- **`tables` is injected only when `tables_arg` is set**, and only via `kwargs`. The unconditional
  `tables=self._tables` in the old `_generic_read_table` must go — that is fault 4.
- **Add the `ReadTableConfigType` alias** (`SI/Datastore.py:151`, `ReadTableConfigType = Dict[str, Any]`)
  alongside the other type aliases in `Datastore.py`, and use it on the constructor parameter's
  `Optional[...]` annotation.
- Check whether `SGWK`'s `ProfileBatchManager` is used the same way as upstream's before copying the
  `with` block; if `SGWK`'s signature or the `mgr` usage differs, follow `SGWK`'s existing pattern in
  the surrounding methods rather than upstream's.
- Note upstream's `return` sits *inside* the `with ProfileBatchManager(...)` block, whereas `SGWK`'s
  current `_generic_read_table` returns outside it. That difference changes when the profile record
  is closed relative to the return. Pick one deliberately, and say which and why in the log.

## Task 2 — `ShardedPool.read_table`

Replace the constructor block at `Datastore/SQL/ShardedPool.py:188-201` with config storage plus
validation, and replace `_generic_read_table` (`ShardedPool.py:815-832`) with an explicit
`read_table`, modelled on `SI/Datastore/SQL/ShardedPool.py:833-880`.

Constructor:

```python
        self._read_table_config: Optional[ReadTableConfigType] = read_table_config
        if read_table_config is not None:
            for class_name, config in read_table_config.items():
                if class_name not in self._replicated_tables:
                    raise RuntimeError(
                        f'It is only possible to configure a read-table method for a replicated table (class name="{class_name}")'
                    )
```

> **X3 — keep the `None` guard.** `SI/ShardedPool.py:206` iterates `read_table_config.items()`
> unconditionally, despite the parameter defaulting to `None` and being typed `Optional`. That is a
> regression. `SGWK`'s existing `if read_table_config is not None:` is correct — **keep it**. Do not
> copy upstream's line verbatim. Note also that the existing `SGWK` error message has an unformatted
> f-string bug (`'...(class id="{class_specifier}")'` with no `f` prefix, `ShardedPool.py:194-196`);
> the replacement above fixes that in passing.

Method:

```python
    def read_table(self, cls, *args, **kwargs):
        if self._read_table_config is None:
            raise RuntimeError("ShardedPool: the read_table service is not configured")

        if isinstance(cls, str):
            class_name = cls
        else:
            class_name = cls.__name__

        if class_name in self._sharded_tables:
            raise RuntimeError(
                f'ShardedPool: the read_table service is only available for replicated tables, but "{class_name}" is configured as a sharded table'
            )

        if class_name not in self._read_table_config:
            raise RuntimeError(
                f'ShardedPool: the read_table service is not available for objects of class "{class_name}"'
            )

        # we only need to read the table from a single shard, so pick one at random
        shard_ids = list(self._shards.keys())
        i = random.randrange(len(shard_ids))

        # swap this entry with the last element, then pop it
        shard_ids[i], shard_ids[-1] = shard_ids[-1], shard_ids[i]
        shard_key = shard_ids.pop()

        shard = self._shards[shard_key]

        return shard.read_table.remote(class_name, *args, **kwargs)
```

The random-shard-selection block is unchanged from `SGWK`'s existing `_generic_read_table`; keep it
as it is rather than rewriting it.

## Task 3 — re-key `config/sharding.py`

`config/sharding.py:37-40` is keyed by *method name* and carries a `"class"` entry:

```python
read_table_config = {
    "read_wavenumber_table": {"class": "wavenumber", "tables_arg": False},
    "read_redshift_table": {"class": "redshift", "tables_arg": True},
}
```

Re-key by **class name**; the `"class"` entry disappears:

```python
read_table_config = {
    "wavenumber": {"tables_arg": False},
    "redshift": {"tables_arg": True},
}
```

The `tables_arg` values are already correct and must not be changed. Confirmed against the factory
signatures:

- `ObjectFactories/wavenumber.py:90` — `read_table(conn, table, units, is_source=None, is_response=None)`
  → takes no `tables` → `tables_arg: False`. ✅
- `ObjectFactories/redshift.py:76` — `read_table(conn, table, tables, is_source=None, is_response=None, model_proxy=None)`
  → takes `tables` positionally-or-by-keyword → `tables_arg: True`. ✅

Both are `@staticmethod`s on the factory class, and both `hasattr(factory, "read_table")` and
`factory.read_table(conn, tab, *args, **kwargs)` work unchanged against a class object.

> **X4 — do not convert factories to instances.** `CPBH`/`SI` register factory *instances*;
> `SGWK` registers *classes* with `@staticmethod`s (`ObjectFactories/base.py`). Both work with every
> call site. Converting is a whole-tree refactor of ~25 modules with no functional benefit, and it is
> **not** a prerequisite for this change. Leave the factory registration alone.

## Task 4 — update the 17 call sites

All calls move from a generated method to the service. The return remains a `ray.get`-able
`ObjectRef`, so **the surrounding code does not change** — only the call expression.

```
pool.read_wavenumber_table(units=…, …)   →   pool.read_table("wavenumber", units=…, …)
pool.read_redshift_table(…)              →   pool.read_table("redshift", …)
```

The 17 sites, verified at `79f0360`:

| File | Lines |
|---|---|
| `extract_Gk_data.py` | 318, 320, 368, 373 |
| `extract_GkSource_data.py` | 775, 777, 821, 826 |
| `extract_GkWKB_data.py` | 362, 364, 411 |
| `extract_QuadSourceIntegral_data.py` | 993, 995, 1045, 1050 |
| `extract_TkWKB_data.py` | 386 |
| `extract_tensor_source_data.py` | 317 |

Line numbers will drift as you edit; work by grep, not by line number:

```bash
grep -rn "read_wavenumber_table\|read_redshift_table" --include="*.py" .
```

Expect 19 hits before you start (17 calls + 2 `config/sharding.py` keys) and **0 after**. Note
`extract_common.py` has none — check anyway rather than assuming.

---

## Do not

- Do not drop the `read_table_config is not None` guard (**X3**).
- Do not convert factories to instances (**X4**).
- Do not touch `object_get_vectorized` or `object_read_batch` (**X1**, **X2**).
- Do not add `inventory_config` or any `inventory()` plumbing. **That is prompt 06**, which depends
  on this one. Upstream's constructor has an `inventory_config` parameter sitting right next to
  `read_table_config`, and its `inventory` method sits right next to `read_table`, so both are easy
  to pick up by accident while copying. Leave them; prompt 06 adds them deliberately, with changes
  `SGWK` needs that upstream does not have.

---

## Verification

1. `grep -rn "read_wavenumber_table\|read_redshift_table" --include="*.py" .` returns **0** hits.
2. `grep -rn "_generic_read_table" --include="*.py" .` returns **0** hits.
3. All eight touched files parse; `black --check` clean if available.
4. Import each of the six `extract_*.py` scripts (or at minimum byte-compile them:
   `python -m py_compile extract_*.py`) to catch syntax slips across a wide mechanical edit.
5. **Behavioural, from audit §8:**
   - Each `extract_*.py` constructs its `ShardedPool` and returns the same wavenumber/redshift
     arrays as before the change. Since the pre-change code *could not construct the pool at all*
     (fault 1 fires in the constructor), "the same as before" is not directly comparable — the
     honest check is against a **direct SQL query** of the `wavenumber` / `redshift` tables in a
     shard database. Say in the log which comparison you actually made.
   - `pool.read_table("GkSource", …)` — a sharded class — raises the intended `RuntimeError`.
   - `pool.read_table("LambdaCDM", …)` — replicated but not in `read_table_config` — likewise.

   The two negative cases are cheap and should be exercised. If constructing a real `ShardedPool` is
   too expensive, they can be checked against a minimally-constructed object, but say so.

   If you cannot run these, record it plainly in the log and open a §3 issue for prompt 10. Do not
   report untested code as verified.

---

## Finish

1. Write `prompts/backport-modules/logs/04-read-table-service.md` using the template in `README.md`
   §5.1. This is the highest-blast-radius change in the campaign, so the log needs to be
   correspondingly thorough: record the `ProfileBatchManager` return-placement decision, anything
   that differed from upstream's shape and why, and the full before/after call-site count. If any
   call site needed more than the mechanical substitution, call it out individually.
2. Update `IMPLEMENTATION_STATE.md`: the prompt 04 row, the B3 item row, the progress count, "last
   updated", and any §3 issues.
3. Commit in one commit. Suggested message:

```
Replace generated read-table methods with a read_table() service

The read_table_config machinery was broken in four independent ways and the
feature was dead: iterating a dict yielded keys, so unpacking each entry into
two names raised ValueError during construction whenever read_table_config
was supplied -- which every extract_*.py script does. Had that been repaired,
setattr bound a plain function to the instance so self was never passed, the
wrapper closed over the loop variables by reference so every generated method
resolved to the last entry, and _generic_read_table passed tables=self._tables
unconditionally while the wrapper also injected it, giving a duplicate
keyword argument.

Adopt the upstream redesign instead: a single explicit read_table(cls, *args,
**kwargs) on each of Datastore and ShardedPool. Datastore validates that the
service is configured, that the class is in the config, and that the factory
exposes read_table; it injects tables only when tables_arg is set. ShardedPool
rejects sharded and unconfigured classes, picks a shard at random, and
forwards. read_table_config is re-keyed by class name and the "class" entry
drops out.

The 17 call sites across six extract_*.py scripts move from
pool.read_wavenumber_table(units=...) to pool.read_table("wavenumber",
units=...). Both still return a ray.get-able ObjectRef, so no surrounding
code changes.

Unlike StochasticInstantons, the read_table_config is not None guard is kept
-- the parameter is Optional and defaults to None. The object factories are
left registered as classes with staticmethods; converting them to instances,
as ChamPBH and StochasticInstantons did, is not a prerequisite.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Adjust to match what you actually did.
