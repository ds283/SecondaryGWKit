# Prompt 02 — Fix the shard-key config reader (B2 + D2)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit sections:** §3 B2, §6 D2, §8 checklist item 3
**Depends on:** prompt 01 (ordering only — no code overlap; 01 edits `_assign_shard_keys`, this
edits `_read_shard_data`)
**Files you may touch:** `Datastore/SQL/ShardedPool.py`, plus the log and status board.

---

## Why this matters

`ShardedPool._read_shard_data` runs on **every reopen of an existing sharded pool**. The
shard-key-config block cannot succeed as written — it reads a column that was never selected — so
this path raises `AttributeError` unconditionally. Combined with prompt 04's finding, it is one of
two reasons the current tree cannot open a datastore it created itself.

Introduced by `a2bd966` ("Abstract out configuration of sharded and replicated tables from
ShardedPool", 2025-12-15), which renamed the column but not this reader.

---

## Task 1 — B2: read the column that was actually selected

`Datastore/SQL/ShardedPool.py:333-350`. The `select` requests exactly one column:

```python
            shard_key_configs = conn.execute(
                sqla.select(
                    self._shard_key_config_table.c.key_type,
                )
            )
```

and the table is defined with exactly one column, `key_type` (`ShardedPool.py:238-247`). But the
loop body tests and interpolates `row.key_attr`. A SQLAlchemy `Row` raises `AttributeError` for a
name it did not select.

There are **three** occurrences, all wrong:

| Line | Context |
|---|---|
| 342 | `if row.key_attr != self._ShardKeyType_name:` — the comparison |
| 344 | the `RuntimeError` f-string in that branch |
| 349 | the f-string in the `elif num_config > 1:` branch |

Line 342 is in the `num_config == 1` branch, which is taken on the *first row of every reopen*, so
this is not a rare edge case.

**Fix all three, not just the comparison.** Upstream (`SI/ShardedPool.py:357`) fixed only line 342
and left both f-strings interpolating `row.key_attr` — meaning `SI`'s error paths still raise
`AttributeError` from inside the diagnostic. Do not reproduce that. (Line 349 may instead be deleted
outright under task 2; if you delete it, only two occurrences remain to fix.)

**Do not confuse this with the legitimate `key_attr` column.** `key_attr` is a real column on the
*`sharded_tables`* table (`ShardedPool.py:271`), correctly written at line 305 and correctly read at
lines 391–431. Those uses are fine. Only the `shard_key_config` reader at 342/344/349 is wrong. A
blanket `key_attr` → `key_type` substitution across the file would break working code.

## Task 2 — D2: the `raise print(...)` branch

`ShardedPool.py:347-350`:

```python
                elif num_config > 1:
                    raise print(
                        f'ShardedPool has unexpected multiple shard key types: {num_config}="{row.key_attr}"'
                    )
```

Two faults. `print` returns `None`, so `raise print(...)` raises
`TypeError: exceptions must derive from BaseException` after printing — never the intended
diagnostic. And the branch appears to be unreachable by design: the block immediately below the loop
already handles the same condition correctly:

```python
            if num_config == 0:
                raise RuntimeError(f"No configured shard key type was found")
            elif num_config > 1:
                raise RuntimeError(f"Multiple configured shard key types were found")
```

**Confirm that reachability claim yourself before acting on it** — read the whole loop, check that
nothing between the `elif` and the post-loop check can consume or `break` out of the iteration, and
check that `key_type` really is the table's primary key (so more than one row is possible at all,
just handled later). Then:

- If the branch is genuinely unreachable, **delete it**. The post-loop `RuntimeError` is the correct
  and reachable handler, and keeping a dead `elif` that raises the wrong exception type is worse
  than not having one.
- If your reading says it *is* reachable, do not delete it — convert it to a proper
  `raise RuntimeError(...)` with `row.key_type`, and **say in the log why you disagreed with the
  planning pass**, with the specific control-flow reason.

Either way this is a judgement call the log must record explicitly, with the reasoning, so a later
reader can evaluate it without re-deriving the control flow.

---

## Do not

- Do not touch the `sharded_tables` `key_attr` reader at `ShardedPool.py:389-431`. It is correct.
- Do not touch `object_get_vectorized` (`ShardedPool.py:569-594`) or `object_read_batch`
  (`ShardedPool.py:596-618`). Audit §5 (**X1**, **X2**) says upstream's changes to both are broken
  or regressive.
- Do not restructure `_read_shard_data` beyond these two items, however tempting. It is long, but
  scope creep here destroys the revert boundary.

---

## Verification

1. `grep -n "key_attr" Datastore/SQL/ShardedPool.py` — every remaining hit must be in the
   `sharded_tables` region (the table definition around line 271, the write around line 305, and the
   reader block around lines 389–431). No hit may be inside the `shard_key_config` block.
2. The file parses and, if `black` is available, passes `black --check`.
3. **Behavioural — the real test.** Audit §8 asks that reopening an existing sharded datastore
   completes `_read_shard_data` without `AttributeError`, and that a deliberately mismatched
   `shard_key_type` produces the intended `RuntimeError` message rather than a second
   `AttributeError`.

   Note from `IMPLEMENTATION_STATE.md` §5 note 1: **`test-qcd-db.sqlite` cannot serve as a fixture**
   — it uses the pre-`a2bd966` schema and has no `shard_key_config` table at all. You need a
   datastore created by the current code. If you can create one cheaply and reopen it, do so and
   record exactly what happened. The mismatch path can be exercised by reopening with a different
   `shard_key_type` configured, or by constructing the objects directly if a full pipeline run is
   too expensive.

   If you cannot run this, **say so plainly in the log** and open an entry in
   `IMPLEMENTATION_STATE.md` §3 for prompt 10. Do not report a behavioural test as passing on the
   strength of reading the code.

---

## Finish

1. Write `prompts/backport-modules/logs/02-shard-config-reader.md` using the template in
   `README.md` §5.1. The D2 decision (delete versus repair) must appear under deviations as an
   **implementation choice** with its reasoning, even if you took the option the prompt recommended.
2. Update `IMPLEMENTATION_STATE.md`: the prompt 02 row, the B2 and D2 item rows, the progress count,
   the "last updated" line, and any §3 issues.
3. Commit in one commit. Suggested message:

```
Fix shard-key config reader in ShardedPool._read_shard_data

The select requests only shard_key_config.key_type, but the loop body tested
and interpolated row.key_attr, which SQLAlchemy raises AttributeError for.
The comparison sits in the num_config == 1 branch, taken on the first row of
every reopen of an existing sharded pool, so this path could not succeed as
written. Introduced by a2bd966, which renamed the column but not this reader.

Fix all three occurrences. StochasticInstantons fixed only the comparison and
left both f-strings interpolating row.key_attr, so its error paths still
raise AttributeError from inside the diagnostic.

The key_attr column on the sharded_tables table is unrelated and correct; its
reader is untouched.

Also remove the `raise print(f"...")` branch for multiple shard key types.
print returns None, so it raised TypeError rather than the intended
diagnostic, and the branch is unreachable -- the check immediately below the
loop already raises the correct RuntimeError.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Adjust to match what you actually did — in particular if you kept and repaired the D2 branch rather
than deleting it.
