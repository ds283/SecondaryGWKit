# Prompt 01 — make the replicated write checked, idempotent and recorded

**Campaign:** [`README.md`](README.md) · **Board items:** **W1**–**W4** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** none of the §6.2 decisions. **Do not start** while any other campaign has an uncommitted
change to `Datastore/SQL/ShardedPool.py` in the tree (README §5 rule 13); `git status` must be
clean.
**Closes:** nothing. `[00-replicated-writes-can-diverge-across-shards]` (store-fingerprint board)
is closed by prompt 02, which repairs what this prompt records. Narrow it on that board: say what
this prompt made loud. **Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Opus**. The code is short. The judgement is in leaving no point between
two commits at which a crash leaves the shards different **and** nothing on disk says so.

**Read first:**

1. [`README.md`](README.md): §0, §1, §4, §5 and §6.1 in full. §6.1 records the user's decisions.
   Do not reopen them.
2. [`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md) §1 and §4 in full.
3. `Datastore/SQL/ShardedPool.py`: the constructor, `_create_engine`, `_write_shard_data`,
   `_read_shard_data`; then `object_get`, `_get_impl_replicated_table`, `object_store`,
   `_store_impl_replicated_table`, `object_validate`, `_validate_impl_replicated_table` and
   `_assign_shard_keys`. **Cite this file by function name**: another campaign edits it.
4. `Datastore/SQL/Datastore.py`: `object_get`, `object_store`, `object_validate`, `_insert`, and the
   version check in the constructor, which is the one replicated write that is already checked.
5. The replica-side `store()` of the two classes written through `object_store`:
   `Datastore/SQL/ObjectFactories/wavenumber.py` (`sqla_wavenumber_exit_time_factory.store`) and
   `Datastore/SQL/ObjectFactories/BackgroundModel.py` (`store`, and how value rows are inserted).
6. `docs/datastore-integrity-audit/replicated_write_fault_probe.py`: the no-Ray pattern your tests
   extend, and the serial split it demonstrates.
7. `CLAUDE.md`, the repository mechanics section.

---

## 1. What is wanted

A replicated object is written on a controlling shard, chosen at random per call, and then copied
to every other shard in separate transactions. Three things are wrong with that today (audit §4.1–
§4.3):
- **nothing records that a replication is in progress**, so after a crash between commits nothing
  on disk says the shards differ, or which of them is right;
- **the replicas' answers are discarded**, so a replica that already holds the key under a
  different serial returns it and nothing notices. That is how a missing row becomes a serial
  split;
- **the replica stores do not look before they insert**, and each shard stamps its own timestamp
  and draws its own value-row serials, so correct copies are not row-identical and cannot be
  compared whole.

This prompt fixes all three. It does **not** repair a store that has already diverged, and it does
not look at a store when it opens. That is prompt 02, which reads what this prompt records.

**The invariant this prompt establishes.** After it, at every instant of every replicated write,
either every shard holds the same replicated rows, or the primary's `replication_in_flight` table
names the operation, the class and the controlling shard whose rows are right.

---

## 2. What to change

**W1 — the in-flight record.**
- A table `replication_in_flight` in the **primary** file (README §4), created by
  `_write_shard_data` for a new store and created if missing when an existing store is opened. It
  holds at most one row: `operation` (`get`, `store` or `validate`), `class_name`,
  `controller_shard`, `store_id` (nullable until known), `started` (a timestamp).
- The row is **committed before the controller's call is submitted**, and deleted after the last
  replica's call has returned. Commit it through the pool's own engine on the primary, in its own
  transaction. A crash anywhere between those two commits leaves it set.
- **One helper** (README §4: `_replicated_write`, or an equivalently named one) through which the
  get, store and validate paths all go, so that the record is written and cleared in one place.
  The three `_…_impl_replicated_table` methods keep their names and call it.
- A get that inserts nothing needs no replication. *Prompt's choice:* you may write the record
  only when a replication will follow, **provided** no crash window is opened by doing so. A
  controller commit that precedes the record is such a window. If you cannot avoid it, write the
  record before every replicated get and record the cost per call in the log; a primary commit per
  replicated call is acceptable.
- A second record while one is set is a programming error: raise. Replicated calls never overlap
  in the driver (audit §4.1); the check makes that an assertion rather than an assumption.
- If the record is found set **at the start** of a replicated write, raise, naming it. That state
  belongs to prompt 02's check at open, and a write must never proceed over it.

**W2 — checked replica answers.**
- In the get path, both forms: for every replica and every replicated item, the `store_id` the
  replica returns must equal the controller's. In the store path: the same, on the object each
  replica returns. In the validate path: every replica's outcome must equal the controller's, as
  today, but checked before the record is cleared.
- A mismatch raises **`ReplicationMismatch`** (README §4), naming the shard, the class, the
  controller's serial and the replica's. **The record is left set** when it raises, so that prompt
  02 finds the store in the state the failure left it.
- Place the exception where both `ShardedPool` and tests can import it without Ray (*prompt's
  choice*; say where in the log).

**W3 — replica stores that look before they insert.**
- `wavenumber_exit_time` and `BackgroundModel` replicas receive the controller's object with its
  serial. Make each replica's store **insert-or-verify by serial**:
  - no row with that serial and no row with that key: insert it, as today;
  - a row with that serial whose identifying columns equal the object's: return it, write nothing;
  - a row with that serial that differs, or the key held under **another** serial: raise
    `ReplicationMismatch`.
- `BackgroundModel` replicas pass each value row's serial through, taken from the controller's
  stored object, so that every copy is row-identical. If the controller's object does not carry its
  value rows' serials, carrying them is part of this item; if it cannot, stop (§5).
- The controller's own store is unchanged in what it writes.
- Do not change `store()` for any sharded class.

**W4 — one timestamp.**
- The driver chooses one timestamp per replicated write and passes it to every shard, the
  controller included, as a per-call override that `Datastore._insert` uses in place of
  `datetime.now()`. A call with no override behaves exactly as today, so every sharded write is
  unchanged.
- Carry the override through the actor methods' arguments, not through actor state. An actor runs
  one call at a time, but state that outlives a call is how one call's timestamp leaks into the
  next.

`_assign_shard_keys` writes the primary's `shard_keys` in its own transaction after the replicas.
Leave it: a missing shard key is re-assigned the next time the wavenumber is got (read it and
confirm; record the reasoning in the log). If that is not so, it is an observation, not a fix.

---

## 3. Tests — a new module in `Datastore/tests/`, no Ray, nothing under `var/`

Build pools with stand-in shards (README §5 rule 9), in temporary directories, following
`replicated_write_fault_probe.py`: the real `ShardedPool`, `Datastore`, factories and broker, with
the actor classes unwrapped, `.remote()` a direct call returning a stand-in reference, and the
patched `ray.get` raising only after every submitted call has run, as Ray does. Scope every patch
with `unittest.mock.patch`. At minimum:

1. **A clean write.** For a get-inserted class (`tolerance`), a stored class
   (`wavenumber_exit_time`) and `BackgroundModel` with its tags and values: afterwards every shard
   holds row-identical copies, **timestamps and value serials included**, and
   `replication_in_flight` is empty.
2. **The record is set across the window.** Inject a failure after the controller's commit and
   before each replica, for each path. Afterwards the record names the operation, class and
   controller, and the controller holds the row.
3. **A replica's different serial is loud.** Build the split of the probe's step 4 by hand (the key
   under another serial on one shard); a get whose replica returns it raises `ReplicationMismatch`
   with both serials, and the record stays set.
4. **Idempotent replica stores.** Replaying a replica's store of `wavenumber_exit_time` and of
   `BackgroundModel` writes nothing and returns the row; a replica holding a differing row, or the
   key under another serial, raises.
5. **A write over a set record refuses**, and writes nothing to any shard.
6. **Sharded writes are unchanged.** A sharded store still stamps `datetime.now()`, and no record
   is written for it.
7. **No Ray.** `ray.is_initialized()` is false after the module.

**Deliberate breakage.** Show that each of these mutations makes a test fail, then restore it:
- (i) discard the replicas' answers again;
- (ii) write the record after the controller's call;
- (iii) clear the record before the last replica returns;
- (iv) let each shard stamp its own timestamp;
- (v) let a replica store insert without verifying.

Put each in the log as a diff, **exactly as applied**, and name the tests that failed. Never commit
a mutation.

---

## 4. Acceptance

1. §3's tests exist in a new module under `Datastore/tests/`, need no Ray and open nothing under
   `var/`. Every existing test passes unmodified.
2. The deliberate-breakage record, (i)–(v), each with its diff and failing tests.
3. The log has a table of every commit point of each replicated path, before and after, and for
   each point between two commits, what a crash there leaves on disk. **No row may say "shards
   differ, no record".**
4. All three replicated paths go through one helper, which alone writes and clears the record.
5. Every existing suite unchanged; `Datastore/tests` rises by exactly the tests you add.
6. `black --check` clean. On the board: the §1 row for 01, items W1–W4, §5. On the store-fingerprint
   board, `[00-replicated-writes-can-diverge-across-shards]` narrowed, and its index row with it.
   All in the same commit.

---

## 5. Stop conditions — stop and ask the user

- Another campaign's change to `ShardedPool.py` is uncommitted in the tree.
- A crash window with "shards differ, no record" cannot be closed without holding a transaction open
  across actor calls.
- The controller's `BackgroundModel` object cannot carry its value rows' serials to the replicas.
- An existing test would have to change.
- Any step would open anything under `var/`.

---

## 6. What this prompt does not do

- It does not read or repair a store at open, and does not look for existing divergence. That is
  prompt 02.
- It does not change any factory's `build()`, any lookup, any sharded write, or the `.scalar()`
  lookups (prompt 03).
- It does not touch `delete_store`, `copy_store`, `move_store` or the closed-store block.
- It does not touch the store inventory or the fingerprint.

---

## 7. The log and the board

`logs/01-checked-replicated-write.md`, using the template in README §5.1. In addition:
- the commit-point table of §4 item 3;
- where `ReplicationMismatch` lives, and why;
- the cost of the in-flight record per replicated call, measured on a stand-in pool;
- whether `_assign_shard_keys` self-heals, with the reasoning.

`IMPLEMENTATION_STATE.md`: the §1 row for 01, items W1–W4, and §5. Leave the held rows as they
are.
