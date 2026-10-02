# Prompt 02 — write the version row, and the prune at open, through the recorded write

**Campaign:** [`README.md`](README.md) · **Board items:** **W1**–**W3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** 00 landed and reviewed (`cfd9791`, `e320338`). `git status` clean. One prompt at a time in
this checkout. 03 is written against this prompt's result and is not dispatched until it lands.
**Closes:** `[02-the-version-row-is-written-outside-the-in-flight-record]`, on the
`datastore-integrity` board's §4, and `[00-the-prune-at-open-deletes-a-replicated-row-outside-the-in-flight-record]`,
on this board's §4 (the user's decision U1, README §6.2). Each index row is deleted.
**Opens:** whatever it finds that this prompt does not fix.
**Recommended model:** **Opus**. It changes the order of events in `ShardedPool.__init__` and the
check at open's repair licence, and both are where a store is lost if they are wrong.

**Read first:**

1. [`README.md`](README.md): §0 item 6, §1, §4, §5, and §6.2 (U1 and U2).
2. [`docs/a3-v2-readiness-audit.md`](../../docs/a3-v2-readiness-audit.md): **V1**, **V2**, **M1**, and
   the closing section's *02*. This prompt is form A of V2. The audit's findings are the facts it
   rests on. Do not re-derive them.
3. The two board entries this prompt closes: `prompts/datastore-integrity/IMPLEMENTATION_STATE.md`
   §3.3, `[02-the-version-row-…]`, and this board's §3.2, `[00-the-prune-at-open-…]`.
4. `Datastore/SQL/ShardedPool.py`: `__init__`; the comment block above `_reconcile_replicated_tables`,
   and that function; `_plan_replicated_repair`; `_apply_replicated_repair`; `_replicated_write`;
   `_get_impl_replicated_table`. `Datastore/SQL/Datastore.py`: `__init__`, `_insert`,
   `_validate_on_startup`. `Datastore/SQL/ObjectFactories/BackgroundModel.py`:
   `sqla_BackgroundModelFactory.validate_on_startup`. `Datastore/SQL/ObjectFactories/version.py`.
   Cite both pool files by function name, never by line.
5. `Datastore/tests/standin_pool.py` and `Datastore/tests/test_reconcile_at_open.py`: the instrument,
   and how an interrupted write is made and reopened.
6. `docs/a3-v2-readiness/version_window_probe.py` and its `.txt`: cases 3 and 4 are the two defects
   this prompt fixes.

---

## 1. What is wanted

Two replicated writes are made inside actors' constructors, one shard at a time, outside the
`replication_in_flight` record:
- the `version` row, inserted by the first actor and then by every other actor under its serial;
- under `prune_unvalidated=True`, the deletion of an unvalidated `BackgroundModel` with its tags
  and values. Each actor runs it on its own shard in `_validate_on_startup`.

Either one, interrupted between actors, leaves shards that differ with no record to say which is
right, and the check at open refuses the store (audit V1, cases 3 and 4, measured). The second
matters for A3 v2: prompt 05 makes its resume prune at open.

After this prompt, both are written by the pool, under a record, and the check at open completes
either one when it is interrupted. The comment above `_reconcile_replicated_tables` says nothing in
the write path deletes a replicated row. After this prompt that is true again, with the prune named
as the one recorded exception.

---

## 2. What to change

**W1 — the version row (form A of audit V2).**
- `Datastore.__init__` no longer looks the version row up or inserts it.
  - It drops `version_serial`. Whether it keeps `version_label` for messages is the *prompt's
    choice*.
  - The actor gains `set_version(serial)`.
  - `_insert` raises before it writes, naming the class, on any insert into a class whose schema
    has `use_version`, until `set_version` has been called. No such insert can then carry a
    missing serial.
- `ShardedPool.__init__`, once every actor exists:
  - writes the row once, through `_get_impl_replicated_table("version", {"label": version_label})`
    and so through `_replicated_write`;
  - keeps the answer as `self._version`;
  - calls `set_version` on every actor, with `ray.get` over all of them;
  - only then returns. Where this sits relative to `read_largest_store_ids` and the broker's
    notification is the *prompt's choice*; state the reason in the log. `version` takes no lease
    (`_insert`), so either order is possible.
- Do not change `sqla_version_factory.build`'s return type (log 00, observation 2) unless W1 cannot
  work without it. If it must change, record the change as a deviation.

**W2 — the check at open treats the version row like any other replicated row.**
- With form A, an interrupted version write leaves a `get`/`version` record, and the generic `get`
  repair copies the controller's row (audit V2, read). Show by test that it does.
- If `_plan_replicated_repair`'s licence refuses the version row for some reason, make the smallest
  change that admits it. Show by test that it admits nothing new for any other class. The log says
  what the reason was. If the change would widen what the licence admits for other classes, stop
  (§5).
- With no record, a `version` table that differs across shards is still refused, exactly as today.
- A store whose shards all lack the label being opened is consistent. Its open writes the row
  through W1's path, as it would a new label.

**W3 — the prune of a replicated class, by the pool, under a record (U1).**
- **Who prunes.** Under `prune_unvalidated=True`, the pool prunes every replicated class whose
  factory prunes at startup. Today that is `BackgroundModel` with `BackgroundModel_tags` and
  `BackgroundModelValue`; find the set from the factories, not from a literal. The pool prunes:
  - **after** the check at open and **before** the broker or any actor exists, with no Ray, on
    plain connections to the shard files, as `_reconcile_replicated_tables` makes them;
  - with the factory's own `validate_on_startup(conn, table, tables, prune=True)`, so that the
    rule for "unvalidated" stays in one place;
  - under a record. A `prune` record naming the class is committed on the primary before the first
    shard is touched. Each shard's prune is its own transaction. The record is cleared last, in
    its own commit. A prune that finds nothing to delete on any shard may skip the record; that is
    the *prompt's choice*, and the log states it.
- **Actors.** Actors still run `_validate_on_startup`, and still prune sharded classes in it. They
  never delete a row of a replicated class: for those classes they report only, as with
  `prune=False`. How the actor knows which classes are replicated is the *prompt's choice*.
- **The check at open completes an interrupted prune.** When the primary holds a `prune` record:
  1. it re-runs that class's prune on every shard, which is idempotent: it deletes every
     unvalidated row of the class, and only those, so a shard already pruned loses nothing more;
  2. it then compares every replicated table as always;
  3. it refuses on any difference that remains;
  4. it clears the record, last.

  It does this whatever the current open's `prune_unvalidated` is. Completing an interrupted write
  is repair, not a choice of this open.
- **The record's schema.** `replication_in_flight.controller_shard` is `NOT NULL`, and a prune has
  no controlling shard. Change the schema as the prune needs. A schema change is free (README §5;
  no store is migrated). A `prune` record must never reach `_plan_replicated_repair`'s
  `get`/`store`/`validate` licence, and a `get`/`store`/`validate` record must never be completed
  as a prune.
- **The comment** above `_reconcile_replicated_tables`, and the docstring of
  `_reconcile_replicated_tables` and of any function whose account of the write path this changes,
  are made true. "Nothing in the write path deletes a replicated row" now names the recorded prune
  as its one exception.

---

## 3. Verification

A test module (or modules) under `Datastore/tests/`, on stand-in shards, with no Ray (README §5
rule 9). Each case below is one test or more.

1. **A new store.** The `version` row is identical on every shard. It was written once, through
   `_replicated_write`: a hook sees a `get` of `version` under a record, and sees no actor
   constructor insert it. Every versioned row stored afterwards carries that serial, on every
   shard.
2. **A new label on an existing store** (audit V1 case 3):
   - a fault **after** the controlling shard's version `get` commits. The next open repairs the
     other shards from the controller and clears the record. Every shard holds the row.
   - a fault **before** it commits. The next open clears a record over identical shards, then
     writes the row.
   - a fault on one replica only. Repaired as in the first case.
3. **A new store's first open, interrupted at the version write**, after every shard file exists.
   The next open succeeds and writes the row.
4. **`_insert` before `set_version`** raises, names the class, and writes nothing.
5. **The prune** (audit V1 case 4), on a store holding an unvalidated `BackgroundModel` with tags and
   values:
   - an open with `prune_unvalidated=True`, faulted after shard *k*'s prune for each *k*. The next
     open, with `prune_unvalidated` **false**, completes the prune on every shard, clears the
     record, and opens. `BackgroundModel`, its tags and its values are absent and identical on
     every shard.
   - the same, faulted before the first shard is touched. The next open re-runs the prune and
     clears the record.
   - an uninterrupted prune. Every shard identical, no record left. A validated model, and every
     sharded row, untouched.
6. **The two kinds of record stay apart.** A `prune` record over a difference in another replicated
   class refuses. A `get` record of `BackgroundModel` over shards that differ is repaired as today,
   not pruned.
7. **No actor deletes a replicated row.** A hook on the actor's prune sees no delete from a
   replicated table, under `prune_unvalidated=True`.
8. **Existing tests unchanged** (README §5 rule 7), with one named exception.
   `Datastore/tests/test_reconcile_at_open.py` `TestPruningAfterRepair` keeps every assertion: a
   model pruned on every shard or kept on every shard. Its class docstring says the prune runs
   "inside each actor". This prompt may correct that docstring, and nothing else in the file.
9. **The audit's probe.** Run `docs/a3-v2-readiness/version_window_probe.py` and paste its output in
   the log. Do not change the probe or its committed output. Cases 3 and 4 should now heal. If the
   probe no longer runs because it reaches into what this prompt removed (it passes
   `version_serial`), say so and paste the error. Cases 2 and 5 above are then the measurement.
10. All six suites at `HEAD` after the change, serially, with README §7's command. Every count
    unchanged except for the tests this prompt adds.

**Deliberate breakage** (README §5 rule 8). Each must make a test fail. Restore it afterwards, and
record each in the log as a diff, exactly as applied:
- (i) the actor's constructor inserts the version row again;
- (ii) `set_version` is not called on one actor;
- (iii) the pool prunes without committing the `prune` record;
- (iv) the check at open clears a `prune` record without re-running the prune;
- (v) the actor prunes `BackgroundModel` again under `prune_unvalidated=True`;
- (vi) `_plan_replicated_repair` accepts a `prune` record as a `get`.

---

## 4. Acceptance

1. W1–W3 as §2 states them, with each *prompt's choice* stated and justified in the log.
2. The tests of §3, and the deliberate-breakage record (i)–(vi).
3. `[02-the-version-row-…]` moved to the `datastore-integrity` board's §4, and
   `[00-the-prune-at-open-…]` to this board's §4. Each gets a closing note naming this commit and
   the tests that show it healed. Both index rows deleted; the count and date corrected.
4. `black --check` clean. On this board: the §1 row, W1–W3, §3/§4, §5. All in one commit.

---

## 5. Stop conditions — stop and ask the user

- The version row cannot be admitted by the `get` repair without widening what the licence admits
  for any other class.
- An existing test other than the named docstring would have to change.
- Any lookup a pipeline or a reader makes turns out to need the version serial (audit V2 read that
  none does).
- Completing an interrupted prune would delete a row that the interrupted prune would not have
  deleted: a row validated since, or a row of another class.
- Anything here needs a store under `var/`.

---

## 6. What this prompt does not do

- No read-only pool, and no change to the six readers. Those are 03's. But 03 will find the version
  row by label on a `mode=ro` select and refuse when it is absent (audit V2). Leave the row
  findable that way.
- It does not touch `_drop_actions`. A drop action of a replicated table is the same defect, opened
  as `[00-a-drop-action-drops-a-replicated-table-outside-the-in-flight-record]`, and is not this
  prompt's.
- It does not close `[00-a-new-stores-first-open-can-leave-a-primary-without-its-shards]`. The user
  decided to record it only (U2). Form A moves the version row after the shard files exist, which
  narrows that window but does not close it. The log says whether it narrowed, in one line.
- No `main.py`, no driver, no extraction script, no run.

---

## 7. The log and the board

`logs/02-write-the-version-row-through-the-recorded-write.md`, using README §5.1. In addition:
- the order of events in `ShardedPool.__init__` before and after, for a new store and for an
  existing one, as two numbered lists;
- the record's schema before and after.

`IMPLEMENTATION_STATE.md`: the §1 row for 02, W1–W3, §3/§4, §5.
