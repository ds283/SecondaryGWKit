# Prompt 03 — Small robustness fixes (F6, B4, D1, D3, D4, F3)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit sections:** §3 B4, §4 F3, §4 F6 (listed in §2.3), §6 D1/D3/D4, §8 checklist item 4
**Depends on:** prompts 01 and 02 (ordering only; no overlapping code regions)
**Files you may touch:** `Datastore/SQL/ShardedPool.py`, `RayTools/RayWorkPool.py`,
`Datastore/SQL/ClientPool.py`, `Datastore/SQL/Datastore.py`, plus the log and status board.

---

## Character of this commit

Six small, mutually independent fixes. Two are backports from upstream (F6, B4); three are latent
defects present in all three trees with no upstream fix to take (D1, D3, D4); one is an optional
decoupling tidy-up (F3).

They are bundled because each is a few lines and none carries behavioural risk. **If any one of them
turns out to be larger than described, drop it from this commit, note it in the log, and open a §3
issue** rather than letting it swell the change.

Work through them in order. Each is self-contained.

---

## F6 — guard the empty `sharded_tables` insert

**File:** `Datastore/SQL/ShardedPool.py:303-308`
**Present in:** upstream `SI` only

```python
            sharded_table_values = [
                {"serial": n, "table": t, "key_attr": k}
                for n, (t, k) in enumerate(self._sharded_tables.items())
            ]
            conn.execute(self._sharded_tables_table.insert(), sharded_table_values)
```

SQLAlchemy 2.x executes `INSERT ... DEFAULT VALUES` when handed an empty list, inserting a spurious
row. Take upstream's guard (`SI/ShardedPool.py:316-323`), comment included:

```python
            # SQLAlchemy 2.x executes DEFAULT VALUES when given an empty list;
            # guard to avoid that when no sharded tables are configured.
            if sharded_table_values:
                conn.execute(self._sharded_tables_table.insert(), sharded_table_values)
```

The `replicated_table_values` insert three lines above (`ShardedPool.py:299-301`) has the same
shape. Upstream did not guard it. `SGWK`'s `config/sharding.py` always configures 14 replicated
tables, so it is not reachable today — **but the same reasoning applies**. Guard it too if you judge
that right, and record the decision in the log either way as an implementation choice. Do not guard
it silently and do not leave it unguarded silently.

## B4 — tolerate a task builder returning `None`

**File:** `RayTools/RayWorkPool.py:283-297`
**Present in:** upstream `CPBH` and `SI`

The dispatch handles `list`/`tuple`/`set` and otherwise falls through to
`store_ref(ref_data, allow_store=True)`, which raises
`RuntimeError: could not interpret output from task builder (object type="NoneType" …)`.

Add an explicit `None` branch that **skips the item**, but raises when `store_results=True` —
because a skipped item there would leave a hole in `self.results` and silently misalign every
subsequent index. Match upstream's behaviour; check `SI/RayTools/RayWorkPool.py` for the exact shape
and error text before writing it.

Sketch of the intent (adapt to the real surrounding code, do not paste blind):

```python
                    if ref_data is None:
                        if self._store_results:
                            raise RuntimeError(
                                "a task builder returned None, which is not compatible with store_results=True"
                            )
                        # nothing to enqueue for this item
                    elif (
                        isinstance(ref_data, list)
                        or isinstance(ref_data, tuple)
                        or isinstance(ref_data, set)
                    ):
                        ...
```

Be careful with the bookkeeping around it: `count += 1` immediately follows the dispatch, and the
`_todo`/queue accounting must stay consistent for a skipped item. Read the surrounding loop
properly and make sure a skipped item does not desynchronise the progress counters or leave the pool
waiting on a task that was never submitted.

This is additive — no existing caller returns `None` today, so no behaviour changes.

## D1 — wrong attribute in the `ShardedPool` constructor error path

**File:** `Datastore/SQL/ShardedPool.py:85-89`

```python
        if self._primary_file.is_dir():
            raise RuntimeError(
                f'Specified database file "{str(self._db_file)}" is a directory'
            )
```

`ShardedPool` has no `_db_file` attribute — the field is `_primary_file`. So passing a directory
produces `AttributeError` instead of the intended message. Change `self._db_file` to
`self._primary_file`. Present in `SI` too (`SI:105`); there is no upstream fix to take.

## D3 — `KeyError` on an unlisted storable class

**File:** `Datastore/SQL/ClientPool.py:158-164`

```python
                default_batch_size=_default_serial_batch_size[table],
```

`_default_serial_batch_size` is a hand-maintained dict. Any storable class missing from it raises
`KeyError` at first insert — a failure mode that is both easy to hit and confusing when it does.

Change to a `.get(table, <default>)`. **Find the right default rather than inventing one:** look at
how `ClientPool`'s own `default_batch_size` parameter is declared and what the module already uses
as a fallback constant. If there is an existing module-level default, use it. If there genuinely
is not one, pick a conservative value, say why in the log, and treat it as an implementation choice.

Present in all three trees (`SI:172`, `CPBH:167`); no upstream fix.

## D4 — copy-paste in the notification bookkeeping

**File:** `RayTools/RayWorkPool.py:543`

```python
                    self._last_num_available_complete = self._num_store_complete
```

Should be `self._num_available_complete`. As written it corrupts the reported "available" rate
whenever an `available_handler` is in use. Check the surrounding block (lines ~540–544) to confirm
the intended pattern — each `_last_num_X_complete` is assigned from the matching `_num_X_complete` —
and that `self._num_available_complete` is the correct attribute name. Present in `SI` too
(`SI:575`).

## F3 — decouple `Datastore` from `MetadataConcepts`

**File:** `Datastore/SQL/Datastore.py:76` (the import) and `:246` (the use)
**Present in:** upstream `CPBH` and `SI`

`Datastore.py:246` does `self.object_get(version, **version_payload)` with `version` imported from
`MetadataConcepts` at line 76. `object_get` already accepts either a class or a string
(`Datastore.py:482-485`), so upstream passes `"version"` and drops the import from the actor module.
Purely a decoupling tidy-up.

**The audit's warning applies and is the reason this is in the prompt at all:** `ShardedPool.py:12`
also imports `version` and uses it at `ShardedPool.py:155`. Upstream did *not* change that one. Do
not create the inconsistency:

- **Recommended:** change both — `Datastore.py:246` and `ShardedPool.py:155` — and drop both imports
  if nothing else in each file needs them. Check before deleting an import; `grep -n "version"` in
  each file and confirm every remaining hit is `version_label`, `version_serial`, `self._version`
  or similar rather than the imported class.
- **Acceptable:** skip F3 entirely. It is marked optional in the audit and delivers no functional
  benefit.
- **Not acceptable:** change one and leave the other.

Record which you did and why.

---

## Do not

- Do not touch `object_get_vectorized` or `object_read_batch` (**X1**, **X2** — audit §5).
- Do not touch `read_table_config` / `_generic_read_table` in either file. That is prompt 04, and
  the two changes would conflict.
- Do not touch `store_handler` in `RayWorkPool`. That is prompt 05.
- Do not "fix" anything else you notice. Record it in the log's *Observations not acted on*.

---

## Verification

1. All four touched files parse; `black --check` clean if `black` is available.
2. `grep -n "_db_file" Datastore/SQL/ShardedPool.py` returns nothing.
3. `grep -n "_last_num_available_complete" RayTools/RayWorkPool.py` — the assignment reads from
   `_num_available_complete`.
4. If F3 was taken: `grep -n "MetadataConcepts" Datastore/SQL/Datastore.py Datastore/SQL/ShardedPool.py`
   is consistent with what you decided, and neither file references a name it no longer imports.
5. **B4 behavioural check** (audit §8): a task builder returning `None` for some items completes
   without error when `store_results=False`, and raises when `store_results=True`. This one is
   genuinely testable in isolation — `RayWorkPool` needs a `pool`, but a task builder returning
   `None` never reaches the pool. Write a small throwaway harness under the scratchpad (not
   committed) if that gets you a real result. If you cannot exercise it, say so and open a §3 issue.

---

## Finish

1. Write `prompts/backport-modules/logs/03-robustness-fixes.md` using the template in `README.md`
   §5.1. This prompt contains **three explicit judgement calls** — the `replicated_table_values`
   guard, the D3 default value, and whether to take F3 — and each must appear under deviations as an
   implementation choice with its reasoning, even where you took the recommended option. Also list
   any item you dropped from the commit and why.
2. Update `IMPLEMENTATION_STATE.md`: the prompt 03 row, the F6/B4/D1/D3/D4/F3 item rows (mark F3
   explicitly if you skipped it), the progress count, "last updated", and any §3 issues.
3. Commit in one commit. Suggested message:

```
Tidy up latent faults in ShardedPool, RayWorkPool and ClientPool

Six small independent fixes, none of which changes behaviour for any current
caller.

Backported from StochasticInstantons:

  * Guard the sharded_tables insert against an empty value list. SQLAlchemy
    2.x executes INSERT ... DEFAULT VALUES for an empty list, inserting a
    spurious row.

  * Let a RayWorkPool task builder return None to mean "nothing to enqueue
    for this item". Previously this fell through to the catch-all and raised
    "could not interpret output from task builder". Still raises under
    store_results=True, where a skipped item would leave a hole in the result
    list and misalign every subsequent index.

Latent faults present in all three trees, with no upstream fix to take:

  * ShardedPool.__init__ reported a directory argument using self._db_file,
    which does not exist on ShardedPool, turning a clear message into an
    AttributeError. The field is _primary_file.

  * SerialPoolManager.lease_serial indexed the hand-maintained
    _default_serial_batch_size dict directly, so any storable class missing
    from it raised KeyError at first insert. Fall back instead.

  * RayWorkPool assigned _last_num_available_complete from
    _num_store_complete, corrupting the reported "available" rate whenever an
    available_handler was in use.

Datastore.object_get is also now given the version class by name, dropping
the MetadataConcepts import from the actor module. ShardedPool is changed to
match, so the two do not disagree.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Adjust to match what you actually did — drop the F3 paragraph if you skipped it, and drop any item
you deferred.
