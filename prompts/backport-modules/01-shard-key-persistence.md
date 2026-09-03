# Prompt 01 — Fix shard-key persistence (B1 + B5)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit sections:** §3 B1, §3 B5, §8 checklist items 1–2
**Depends on:** nothing (this is the first commit)
**Files you may touch:** `Datastore/SQL/ShardedPool.py`, a new `tools/shard_key_audit.py`, plus the
log and status board.

---

## Why this is first

This is the only Critical-severity item in the campaign. It causes **silent, permanent data
misrouting** that a single clean run cannot detect: the in-memory shard map is correct, so run 1
behaves perfectly, and the corruption only surfaces on the *next* run when the map is rebuilt from
disk. `StochasticInstantons` observed ~5,000 orphaned records from this.

Read `/Users/ds283/Documents/Code/StochasticInstantons/shard-key-assignment-bug.md` before you
start. It is the user's own write-up, with a reproduction snippet and the failure mode in full.

---

## Background

`ShardedPool` keeps a shard-key → shard-id map in two places:

- **in memory**, `self._shard_keys: Dict[int, int]`, keyed by shard-key object `store_id`;
- **on disk**, the `shard_keys` table in the primary SQLite file.

`ShardedPool.py:248-259` defines that table with primary key column **`key_serial`**.
`_read_shard_data` reads it back correctly (`ShardedPool.py:437-444`, `self._shard_keys[key.key_serial] = key.shard_id`).

But `_assign_shard_keys` writes `{"key_id": item.store_id, "shard_id": new_shard}`
(`ShardedPool.py:800-803`). There is no `key_id` column. SQLAlchemy silently discards the unknown
key, the INSERT binds only `shard_id`, and SQLite autoassigns `key_serial` from its own rowid
counter. Whenever shard-key objects are processed out of `store_id` order, the on-disk map diverges
from the in-memory one, and every record written under a displaced key becomes unreachable on the
next run.

`SGWK`'s shard key is `wavenumber` (`config/sharding.py:44`), created through the same vectorized
replicated-table path, so the exposure is identical to the one `SI` observed.

---

## Task 1 — B1: write the correct primary key, and detect mismatches

In `ShardedPool._assign_shard_keys` (`Datastore/SQL/ShardedPool.py:764-818`), change the INSERT to
bind `key_serial`, capture the primary key the database actually assigned, and warn loudly if it
differs from what was intended.

Current code (`ShardedPool.py:799-803`):

```python
                # insert a new record for this key
                conn.execute(
                    sqla.insert(self._shard_key_table),
                    {"key_id": item.store_id, "shard_id": new_shard},
                )
```

Target shape, matching upstream `SI/Datastore/SQL/ShardedPool.py:816-846`:

```python
                # insert a new record for this key
                result = conn.execute(
                    sqla.insert(self._shard_key_table),
                    {"key_serial": item.store_id, "shard_id": new_shard},
                )
                assigned_serial = result.inserted_primary_key[0]

                if assigned_serial != item.store_id:
                    print(
                        f"!! _assign_shard_keys MISMATCH: "
                        f"store_id={item.store_id}, "
                        f"assigned key_serial={assigned_serial}, "
                        f"shard={new_shard}"
                    )
```

Notes:

- **Port the mismatch check as well, not just the column name.** It is the only cheap ongoing
  detector for a recurrence, and it costs nothing when things are correct.
- Leave the existing commented-out `print(f">> assigned shard #…")` diagnostic alone, or replace it
  with upstream's commented-out equivalent — your choice, but say which in the log. Do not turn
  either into a live print; the per-key volume would be large.
- This fix is behaviour-preserving in the common sequential case (SQLite would have assigned the
  same value anyway), so it is safe to apply unconditionally with no migration flag.

## Task 2 — B5: deduplicate within a batch

Still in `_assign_shard_keys`, the collection loop (`ShardedPool.py:772-781`) appends to
`missing_keys` on the sole test `item.store_id not in self._shard_keys`. Nothing is written to
`self._shard_keys` until the *write* loop further down, so the same shard-key object appearing twice
in one `obj` list is appended twice — producing two INSERTs for one `store_id` and double-counting
the load balancer.

**This is a hard prerequisite for B1, not an independent nicety.** Before B1 the duplicate INSERT
was silently absorbed (both rows got autoassigned distinct rowids). After B1 the second INSERT binds
the same explicit `key_serial` and violates the primary-key constraint. The two changes must land
together.

Add a local `seen_store_ids` set, as upstream does (`SI/ShardedPool.py:788-799`):

```python
        seen_store_ids = set()
        missing_keys = []
        for item in data:
            if not isinstance(item, self._ShardKeyType):
                raise RuntimeError(
                    f'shard keys should be of type "{self._ShardKeyType_name}"'
                )

            if (
                item.store_id not in self._shard_keys
                and item.store_id not in seen_store_ids
            ):
                missing_keys.append(item)
                seen_store_ids.add(item.store_id)
```

(Upstream writes the condition on one long line; wrap it to match this repository's `black`
formatting.)

## Task 3 — the data auditor

Write a new **read-only** standalone script `tools/shard_key_audit.py` that inspects an existing
sharded datastore's primary database and reports whether its `shard_keys` table is self-consistent.

**Scope it correctly.** The planning pass established (see `IMPLEMENTATION_STATE.md` §5, note 1)
that the datastore checked into the tree, `test-qcd-db.sqlite`, uses the *pre-refactor* schema —
its shard-key PK column is `wavenumber_serial` and the `shard_key_config` / `replicated_tables` /
`sharded_tables` tables do not exist at all. So there is **no known-corrupt `SGWK` datastore to
rescue**. This tool is a forward-looking detector, and a way for the user to check datastores held
outside the repository. Do not write it as, or describe it as, a repair of known damage.

Requirements:

- Takes the path to a primary database file as an argument. Opens it **read-only** (use a
  `file:…?mode=ro` URI, or open with `sqlite3` and never issue a write). It must be impossible for
  this script to modify a datastore.
- Detects and reports the schema generation it is looking at: if the `shard_keys` PK column is
  `wavenumber_serial` rather than `key_serial`, say clearly that this is a pre-`a2bd966` datastore,
  that it is not readable by the current code, and that it must be rebuilt — then exit without
  pretending to audit it.
- For a current-schema datastore, report:
  - the row count of `shard_keys` versus the row count of the shard-key table (`wavenumber`);
  - any `shard_keys.key_serial` with no corresponding `wavenumber` row (orphaned key);
  - any `wavenumber` row with no `shard_keys` entry (unassigned key — not necessarily an error, but
    worth reporting);
  - the per-shard distribution of assigned keys, so a wildly unbalanced spread is visible.
- Exits non-zero if any inconsistency is found, zero otherwise, and prints a one-line verdict.
- **State plainly in the script's own output and docstring that there is no safe automated repair.**
  Reconstructing the correct map requires knowing the original insertion order, which is not
  recorded. The remedy for a corrupted store is to rebuild it. Do not write a fix-up path.

The `wavenumber` table lives in the *shard* databases, not the primary one, if the query needs it —
check where it actually is before writing the join, and if a cross-file join is needed either attach
one shard or take the shard-file paths from the primary `shards` table. If a clean single-file
audit is not possible, say so in the log and implement the strongest single-file check available
(count and monotonicity of `key_serial`, referential integrity of `shard_id` against `shards`).

---

## Do not

- Do not touch `object_get_vectorized` or `object_read_batch`. They sit near this code and the audit
  is explicit that upstream's changes to both (**X1**, **X2**) are broken or regressive and must not
  be backported.
- Do not touch `_read_shard_data` — that is prompt 02.
- Do not change the `shard_keys` table definition. `key_serial` is correct; the writer was wrong.
- Do not add a schema migration. See `IMPLEMENTATION_STATE.md` §5 note 1: old datastores are
  unreadable for independent reasons and must be rebuilt.

---

## Verification

Static, and doable now:

1. `grep -n "key_id" Datastore/SQL/ShardedPool.py` returns nothing.
2. `python -c "import ast,sys; ast.parse(open('Datastore/SQL/ShardedPool.py').read())"` — parses.
3. `black --check Datastore/SQL/ShardedPool.py tools/shard_key_audit.py` if `black` is available
   (this repository's formatting is `black`-shaped; match it either way).
4. Run `tools/shard_key_audit.py test-qcd-db.sqlite`. Expected: it identifies the pre-refactor
   schema and reports that the store must be rebuilt. This is the tool's negative path and it is
   the one case you can exercise today — do exercise it.

Behavioural, and probably **not** doable inside this prompt:

5. Audit §8 asks that on a fresh datastore every `shard_keys.key_serial` equals the `store_id` of
   the corresponding `wavenumber` row, with no `MISMATCH` lines; and that a stopped-and-resumed run
   finds all previously-written records. Both need a real Ray pipeline run. If you can create a
   fresh datastore cheaply, do it and record the result. If not, **say so explicitly in the log**
   and open an entry in `IMPLEMENTATION_STATE.md` §3 so prompt 10 picks it up. Do not claim these
   passed on the basis of reading the code.

---

## Finish

1. Write `prompts/backport-modules/logs/01-shard-key-persistence.md` using the template in
   `README.md` §5.1. Classify every deviation as **structurally required**, **an implementation
   choice** (with enough justification that a later reader can agree or disagree on the merits), or
   **unintended drift**. Be specific about what you actually ran versus what you reasoned about.
2. Update `IMPLEMENTATION_STATE.md`: prompt 01 row, the B1 and B5 item rows, the progress count,
   the "last updated" line, and any §3 issues you are opening.
3. Commit everything in one commit. Suggested message:

```
Fix shard-key persistence in ShardedPool

_assign_shard_keys inserted {"key_id": ...} into the shard_keys table, whose
primary key column is key_serial. SQLAlchemy silently discarded the unknown
key, so the INSERT bound only shard_id and SQLite autoassigned the primary
key from its rowid counter. The in-memory map self._shard_keys was correct,
so any single run behaved perfectly; on the next run _read_shard_data
rebuilt a different map whenever shard-key objects were processed out of
store_id order, and every record written under a displaced key became
permanently unfindable. StochasticInstantons observed ~5,000 orphaned
records from this.

Bind key_serial explicitly, capture result.inserted_primary_key[0], and print
a MISMATCH warning when it differs from the intended store_id -- the only
cheap ongoing detector for a recurrence.

Also deduplicate shard-key objects within a single batch. Nothing is written
to self._shard_keys until the write loop, so a repeated object was appended
to missing_keys twice. That was previously absorbed silently; with an
explicit primary key it would violate the constraint, so the two changes have
to land together.

Adds tools/shard_key_audit.py, a read-only consistency check for an existing
datastore's shard_keys table. There is no safe automated repair -- recovering
the correct map needs the original insertion order, which is not recorded --
so a corrupted store has to be rebuilt.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Adjust the body to match what you actually did.
