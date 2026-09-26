# Prompt 02 — at open, repair an interrupted replication, and refuse any other divergence

**Campaign:** [`README.md`](README.md) · **Board items:** **W5**–**W7** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** prompt 01 has landed; `git status` is clean, and no other campaign has an uncommitted
change to `Datastore/SQL/ShardedPool.py` (README §5 rule 13).
**Closes:** `[00-replicated-writes-can-diverge-across-shards]`, on the store-fingerprint board's §4,
with its index row deleted. **Opens:** anything out of scope that you find (§6), **without fixing
it**.
**Recommended model:** **Opus**. The judgement is in what may be repaired and what must be refused:
a repair that copies the wrong shard's rows forward is worse than the divergence it removes.

**Read first:**

1. [`README.md`](README.md): §0, §1, §4, §5 and §6.1.
2. [`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md) §4 in full.
3. [`logs/01-checked-replicated-write.md`](logs/01-checked-replicated-write.md): what prompt 01
   ships, its commit-point table, and where `ReplicationMismatch` and `replication_in_flight` live.
4. `Datastore/SQL/ShardedPool.py`: the constructor, `_check_shard_files`, `_read_shard_data`,
   `_assign_shard_keys`, and prompt 01's replicated-write helper. **Cite it by function name.**
5. `Datastore/SQL/Datastore.py`: the constructor's `prune_unvalidated` and `drop_actions` handling
   and `_drop_order`, which run inside the actors after the point where your step runs.
6. `Datastore/SQL/schema.py` `build_schema`, which gives you every table without an actor.
7. `Datastore/SQL/ObjectFactories/BackgroundModel.py`: `validate` and how `validated` is decided
   from the value-row count.
8. `docs/datastore-integrity-audit/replicated_write_fault_probe.py`,
   `replicated_divergence_probe.py`, `hot_journal_probe.py` and `replicated_serial_census.py`.

---

## 1. What is wanted

After prompt 01, a crash during a replicated write leaves the shards different **and** leaves a
record in the primary saying which operation, which class and which controlling shard. This prompt
reads that record when the store is next opened, and puts the store right before anything else
touches it.

**Why repair is safe here, and only here.** Nothing in the write path deletes a replicated row, the
controller always commits first, and replicas only copy what it committed; the flag updates only
turn flags on (audit §4.5). So after an interrupted replication the right state is exactly "every
shard holds what the controller holds", and copying forward from the controller is a complete
repair. **Any other difference has no such guarantee**, and the store must not be opened:
- a difference with no record: a crash before prompt 01, a hand edit, an interrupted `--drop`;
- a row on some shard that the controller lacks;
- the same key under different serials — a serial split;
- a difference in any column but those a replication sets.

---

## 2. What to change

**W5 — the comparison.** `ShardedPool._reconcile_replicated_tables()` (README §4), called by the
constructor for an **existing** store after `_check_shard_files` and **before the broker or any
actor is created**, needing no Ray.
- **Hot journals first.** Open each shard file once, read-write, so that SQLite rolls back any hot
  journal, and close it. Record in `reconciliation` which files had one. This is what any open
  would do; doing it here means the comparison reads committed state.
- **Read every replicated table from every shard**, through tables built by `build_schema`, with
  plain connections. Compare full rows, keyed by serial, **excluding `timestamp`** (every store
  written before prompt 01 stamps it per shard). Compare `BackgroundModelValue` keyed by
  `(model_serial, z_serial)`, excluding its `serial`, which was drawn per shard before prompt 01.
  Compare tag tables by their composite key. *Prompt's choice:* how you compare, provided the
  classes, keys and exclusions are exactly these, and are stated in the log.
- Also check the primary's `shard_keys`: every `key_serial` is a wavenumber serial present on
  every shard.
- A store with no difference and no record opens as today. Measure the added open time on a
  stand-in store of production size (1740 redshifts, a `BackgroundModel` with 1740 values, four
  and twenty shards) and record it.

**W6 — the repair**, only when `replication_in_flight` holds a row.
- Its class's rows that the controller holds and another shard lacks are copied to that shard,
  **verbatim**, serial and timestamp included. A `BackgroundModel` is copied as a unit: row, tags
  and values.
- A flag the controller holds set and another shard unset (`redshift`/`wavenumber` `source` and
  `response`) is set on that shard.
- For a `validate` record, `validated` is **recomputed** on each shard by the factory's own rule
  from its own value rows, not copied; if the shards then disagree, refuse (W7).
- Then compare again, as W5. Only if every shard now agrees is the record cleared, in its own
  commit, last.
- A record whose controller holds nothing the others lack (the crash came after every replica) is
  simply cleared.
- Every row copied is listed in `ShardedPool.reconciliation` and printed, one line per class and
  shard, so that a resumed build's log says what was repaired.

**W7 — the refusal.** Any difference W6 does not cover raises **`ReplicatedDivergence`** (README
§4), before any actor exists, naming for each difference the class, the shard and the serial (or
key), and what differs. Say in the message that nothing was repaired, and that a store in this
state cannot be put right by the tree: it needs a person, or regeneration. In particular:
- a serial split (the key of the fault probe's step 4) is refused, whether or not a record is set;
- a row on a shard that the record's controller lacks is refused;
- a record naming a class other than the one that differs is refused.

Do **not** refuse or repair because of anything in a sharded table. That is not replicated, and
nothing here can say what it should be.

**What runs after.** `prune_unvalidated` and `drop_actions` run inside each actor after this step.
Because every shard is identical by then, they act identically. Say so in the log, and add a test
(§3 item 6).

---

## 3. Tests — a new module in `Datastore/tests/`, no Ray, nothing under `var/`

Stand-in shards as prompt 01's tests build them (README §5 rule 9). At minimum:

1. **Kill and reopen, at every point.** For each replicated path (get, store, validate) and each
   point of prompt 01's commit-point table, inject the failure, then reopen: every shard is
   identical afterwards, the record is empty, and `reconciliation` lists what was copied. Then a
   second reopen repairs nothing.
2. **Serial split refuses.** The probe's step 4 state, with and without a record: `ReplicatedDivergence`
   names both serials, and no shard file changed (`tree_state`, or checksums).
3. **No record, any difference, refuses.** A missing row, an extra row, a flag, a `validated`
   difference and a value-count difference, each with no record: each refused, nothing changed.
4. **A row the controller lacks refuses**, with the record set.
5. **A hot journal** on one shard, made with `os.fork` and `os._exit` inside `BEGIN IMMEDIATE` as
   `hot_journal_probe.py` makes it: rolled back, then compared, then opened.
6. **Pruning after repair.** An interrupted `BackgroundModel` validate, reopened with
   `prune_unvalidated` true and false: both end with every shard identical.
7. **A clean store is untouched.** Opening a store with no record and no difference writes nothing
   to any shard file.
8. **No Ray.** `ray.is_initialized()` is false after the module.

**Deliberate breakage.** Show that each makes a test fail, then restore it:
- (i) repair with no record;
- (ii) copy forward from a shard other than the record's controller;
- (iii) compare by key alone, not by serial, so that a split looks like agreement;
- (iv) copy `validated` instead of recomputing it;
- (v) clear the record before the second comparison.

Each in the log as a diff, exactly as applied, with the failing tests. Never commit one.

---

## 4. Acceptance

1. §3's tests exist in a new module under `Datastore/tests/`, need no Ray and open nothing under
   `var/`. Every existing test passes unmodified.
2. The deliberate-breakage record, (i)–(v).
3. The log's table: for each row of prompt 01's commit-point table, what a reopen does — repaired
   (and from where), cleared, or refused.
4. The added open time, measured (W5).
5. `[00-replicated-writes-can-diverge-across-shards]` moved to the store-fingerprint board's §4,
   with a closing note naming this commit and prompt 01's; its row deleted from
   `docs/OPEN_ISSUES.md`, with the count and date corrected.
6. `black --check` clean. On the board: the §1 row for 02, items W5–W7, §3.1, §5. All in the same
   commit.

---

## 5. Stop conditions — stop and ask the user

- The comparison cannot run before the actors exist, or needs Ray.
- A repair would need to delete a row from any shard.
- Any difference you meet in a test built only through the real write path, without injection, is
  not one this prompt can classify.
- An existing test would have to change.
- Any step would open anything under `var/`. The three real stores were measured clean by the audit
  (§4.4); this prompt does not re-measure them.

---

## 6. What this prompt does not do

- It does not change the write path; that is prompt 01's, and landed.
- It does not touch the store inventory or the fingerprint. Their blind spots are
  `[00-the-inventory-cannot-see-a-serial-split]`, recorded, not fixed here.
- It does not repair or refuse on sharded tables.
- It does not change `prune_unvalidated`, `drop_actions` or `_drop_order`.

---

## 7. The log and the board

`logs/02-repair-or-refuse-at-open.md`, using the template in README §5.1. In addition: the
comparison's classes, keys and exclusions; the reopen table of §4 item 3; the measured open time;
the example output of a repair and of a refusal, as a user would see them.

`IMPLEMENTATION_STATE.md`: the §1 row for 02, items W5–W7, §3.1 (the issue closed), and §5.
