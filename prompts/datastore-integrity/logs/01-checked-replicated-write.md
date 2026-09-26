# Log 01 — Make the replicated write checked, idempotent and recorded

**Prompt:** [`prompts/datastore-integrity/01-checked-replicated-write.md`](../01-checked-replicated-write.md)
**Commit:** *(this commit)* — "Make the replicated write checked, idempotent and recorded"
**Model:** Claude Opus 5.5
**Date:** 2026-09-26
**Result:** COMPLETE. W1–W4 shipped. No §5 stop condition was met: no crash window with "shards
differ, no record" remains (the commit-point table below has none), no transaction is held open
across actor calls, the controller's `BackgroundModel` object already carries its value rows'
serials, no existing test changed, and nothing under `var/` was opened. The new test module adds
26 tests; `Datastore` rises from 206 to 232 and every other suite is unchanged. Nothing is closed;
`[00-replicated-writes-can-diverge-across-shards]` is narrowed on the `store-fingerprint` board.
Nothing is opened.

## What shipped

**W1 — the in-flight record** (`Datastore/SQL/ShardedPool.py`).
- `_create_engine` declares the table `replication_in_flight` in the primary: `operation`,
  `class_name`, `controller_shard` (all NOT NULL), `store_id` (nullable), `started` (DateTime, NOT
  NULL). `_write_shard_data` creates it for a new store. For an existing store the constructor
  creates it (`checkfirst=True`) after `_read_shard_data` and `_check_shard_files`, so a store that
  fails its checks is not written.
- **`ShardedPool._replicated_write(operation, cls_name, submit_controller, replication,
  store_id=None)`** is the one helper. `_get_impl_replicated_table`,
  `_store_impl_replicated_table` and `_validate_impl_replicated_table` keep their names and each
  calls it, supplying two closures: how to submit the controller's call, and — given the
  controller's answer — whether a replication follows, how to submit it to one replica and how to
  check one replica's answer (the `_Replication` named tuple). The helper alone reads, writes,
  updates and deletes `replication_in_flight`; outside it the table name appears only in its
  declaration and its two `create` calls. In order, it:
  1. refuses if another replicated write is in progress in this pool (an in-memory flag,
     `_replication_active`: a `RuntimeError`, "Replicated writes must not overlap");
  2. draws the controlling shard exactly as before (`random.randrange`, swap, pop) and one
     timestamp, `datetime.now()`, in the driver;
  3. in **one transaction on the pool's own engine on the primary**, reads the table; if a row is
     there it raises **`ReplicationInFlight`** naming it, having written nothing to any shard or
     to the primary; otherwise it inserts the record (`store_id` set only for a validate, whose
     serial is known) and commits;
  4. only then submits the controller's call and reads its answer;
  5. asks the path whether a replication follows. If one does, it fills in `store_id` (scalar get
     and store) in its own primary transaction, submits **every** replica's call, reads every
     answer with one `ray.get`, and checks every answer against the controller's;
  6. only after all of that deletes the record, in its own transaction, requiring exactly one row.

  Any exception after step 3 leaves the record set: a replica that fails, a
  `ReplicationMismatch`, and also a controller that fails — see the table and deviation 3.
- The record is written before **every** replicated call, including a get that turns out to insert
  nothing, which is then cleared without touching a replica (deviation 2; cost below).

**W2 — checked replica answers** (`ShardedPool.py`).
- Scalar get: every replica's answer must carry the controller's `store_id`. Vectorized get:
  every replica must return one object per replicated item, each with the controller's serial for
  that item. Store: every replica's returned object must carry the controller's serial, and for a
  `BackgroundModel` the same list of value serials. Validate: every replica's outcome must equal
  the controller's, as before, now checked before the record is cleared.
- A difference raises **`ReplicationMismatch(shard, class_name, controller_serial,
  replica_serial, detail=None)`**, a `RuntimeError` subclass. The validate path's disagreement,
  which raised a bare `RuntimeError`, now raises it too (still a `RuntimeError`), with the same
  "!! Validation outcomes did not agree" print.

**W3 — replica stores that look before they insert** (`ObjectFactories/wavenumber.py`,
`ObjectFactories/BackgroundModel.py`, `store()` only).
- A replica is recognised as the existing comments already did: the object carries a serial
  (`_my_id`), which only the controller's stored object does. The two lines that put it into the
  payload are unchanged; the new block below them runs when `"serial"` is in the payload.
- **`sqla_wavenumber_exit_time_factory.store`**: a row under the serial whose every column
  `store()` writes equals the object's → return the object, write nothing; one that differs →
  `ReplicationMismatch` naming the columns; no row under the serial but a row with the same key
  (`wavenumber_serial`, `cosmology_type`, `cosmology_serial`, `atol_serial`, `rtol_serial`,
  `stepping`, the inventory's identity, compared exactly) → `ReplicationMismatch` naming the other
  serial; otherwise insert, as before.
- **`sqla_BackgroundModelFactory.store`**: the same three cases, where "equal" means every column
  `store()` writes except `validated` (which `validate()` sets afterwards), the same tag set, and
  the same value rows — the same serials, every column equal. The key is the inventory's identity
  columns (`cosmology_type`, `cosmology_serial`, the three Gauss orders, `source_grid_digest`,
  `source_grid_construction`, `z_init_serial`) among **validated** rows with **exactly** these tags
  (deviation 9). A replica also refuses if the controller's object lacks a value serial or if any
  of its value serials is already taken on the shard (looked for by range, then intersected, so no
  statement grows with the sample count).
- **Value rows carry the controller's serials.** The controller's `store()` already sets each
  value's `_my_id` from its inserter, and the driver receives that object, so no change to the
  object was needed (§5's third stop condition does not arise). A replica now inserts each value
  row with `serial=value._my_id`; before, it drew its own from the broker. The controller's own
  store writes exactly what it wrote before: its value payload is the same dict, now built by a
  local `value_payload` function shared with the verification.
- `Datastore.object_store` catches a `ReplicationMismatch` from `store()` and re-raises it naming
  the actor (`with_shard(self._my_name)`), because a factory does not know its shard.

**W4 — one timestamp** (`Datastore/SQL/Datastore.py`).
- `Datastore._insert(schema, table, conn, payload, insert_timestamp=None)` stamps
  `insert_timestamp` when given and `datetime.now()` otherwise.
- `Datastore.object_get(ObjectClass, *, insert_timestamp=None, **kwargs)` and
  `Datastore.object_store(objects, *, insert_timestamp=None)` take the override as a keyword-only
  call argument. `_inserters_for(insert_timestamp)` returns the actor's own inserters when there
  is none — the same objects as before — and otherwise a dict of `functools.partial(self._insert,
  schema, table, insert_timestamp=...)` built for that call alone, which is passed to `build()` /
  `store()` as `inserter` and `inserters`. Nothing is stored on the actor.
- The driver passes the helper's one timestamp to the controller and every replica on the get and
  store paths. The validate path inserts nothing and passes none. No sharded path passes one, so
  every sharded row still takes its shard's `datetime.now()`.

**`ReplicationMismatch` lives in `Datastore/replication.py`**, with `ReplicationInFlight` and two
small comparison helpers (`same_value`, `differing_columns`). Why there: the factories raise it
inside a Datastore actor, `ShardedPool` raises it in the driver, and the tests catch it. The
module is at the top of the `Datastore` package and imports nothing from it, so importing it pulls
in neither Ray's actor classes nor `Datastore.SQL` (whose `__init__` imports the `Datastore` actor
module, which imports every factory — a factory importing from there would be a cycle). The
factories import it inside `store()`, as they already import `store_inventory` inside
`inventory_records`. Both exceptions define `__reduce__`, so they cross a pickle — as they do
under Ray — with their fields intact; a test checks it.

**Tests.** `Datastore/tests/standin_pool.py` (the reusable stand-in pool) and
`Datastore/tests/test_replicated_write.py` (26 tests):

| § of the prompt | Tests |
|---|---|
| 1. a clean write | `TestCleanWrite.test_every_copy_is_row_identical` compares **every column of every row** of all 13 replicated tables plus `BackgroundModel_tags` across the three shards, after a scalar get (`tolerance`), a vectorized get (`redshift`), a stored `wavenumber_exit_time` and a `BackgroundModel` with two tags and three values, stored and validated. The Datastore module's clock is patched to tick on every call, so only a driver-chosen timestamp can agree; value serials are checked equal to the controller's; the record is empty. Also: a get that inserts nothing calls no replica; an existing store without the table gains it, empty, when opened. |
| 2. the record across the window | `TestRecordAcrossTheWindow`: a replica killed before it runs, for each replica in turn, on the scalar get, the vectorized get, the store of an exit time, the store of a `BackgroundModel` and its validate — the record names the operation, class and controller (and the serial where there is one), the controller holds the row and the killed replica does not. A controller killed **after** its commit, on get and both stores — the record is there. A hook reading the primary during every shard's call, on all five paths — the record is on disk throughout. |
| 3. a different serial is loud | `TestReplicaSerialMismatch`: the probe's step-4 split built by hand through one shard's actor, for a scalar and a vectorized get — `ReplicationMismatch` with shard, class and both serials, the record set. The same split for a store (the replica's `store()` refuses). The exception pickles. |
| 4. idempotent replica stores | `TestIdempotentReplicaStores`: a replay on every shard writes nothing (snapshot of every replicated table unchanged) and returns the row, for both classes; a differing row raises (exit time: `z_exit`; `BackgroundModel`: a value row, the model row, the tags); the key under another serial raises, for both. And an unvalidated row of the same key does not block a later store (deviation 9). |
| 5. a write over a set record | `TestWriteOverASetRecord`: with the record left by an interrupted get, every path (scalar get, vectorized get, both stores, validate) raises `ReplicationInFlight`, makes **no** actor call, changes no row of any shard and leaves the record as it was; the same after closing and reopening the pool; the in-memory overlap check raises and writes nothing. |
| 6. sharded writes unchanged | `TestShardedWritesUnchanged`: a `GkSourcePolicyData` store (sharded) makes one call, to the wavenumber's shard, with no `insert_timestamp`, sees no record before or after, and its row carries the shard's patched `datetime.now()`; a replicated row does not take the shard's clock. |
| 7. no Ray | `tearDownModule` raises if `ray.is_initialized()`, and every test's `tearDown` asserts it is false. |

## The commit points of each replicated path

`P` is a commit on the primary, `C` the controlling shard's commit, `R1`…`Rn` the replicas'
(each its own transaction), `K` `_assign_shard_keys`'s commit (wavenumber only, when a new key is
assigned). "Same" means every shard holds the same replicated rows. SQLite runs in rollback-journal
mode, so a crash **inside** a transaction rolls it back at the next read-write open, and lands in
the row before it.

**Before (at `65cd4b7`).**

| Path | Commit points | Crash between | On disk |
|---|---|---|---|
| get (scalar or vectorized), inserts | `C`, `R1`…`Rn`, `K` | `C`–`R1`, `Ri`–`Ri+1` | **shards differ, no record** |
| | | `Rn`–`K` | same; a new shard key missing |
| get, inserts nothing | `C` (writes nothing) | — | same |
| store | `C`, `R1`…`Rn` | `C`–`R1`, `Ri`–`Ri+1` | **shards differ, no record** |
| validate, controller `True` | `C`, `R1`…`Rn` | `C`–`R1`, `Ri`–`Ri+1` | **shards differ (`validated`), no record** |

and, on a completed write, a replica that already held the key under another serial kept it,
unreported (audit §4.3), and every copy carried its own timestamp and value serials. The rows in
bold are the defect this prompt removes; none of them survives in the table below.

**After (this commit).**

| Path | Commit points | Crash between | On disk |
|---|---|---|---|
| every path | `P1` record inserted, then … | before `P1` (or inside it) | same; no record; nothing was written |
| get, inserts or updates (scalar) | `P1`, `C`, `P2` (`store_id`), `R1`…`Rn`, check, `P3` (record deleted), `K` | `P1`–`C` | same; record set, `store_id` NULL |
| | | `C`–`P2` | shards differ (the controller holds the row); record set, `store_id` NULL |
| | | `P2`–`R1` | shards differ; record set with the controller's serial |
| | | `Ri`–`Ri+1` | shards differ (the controller and `i` replicas hold it); record set |
| | | `Rn`–`P3`, including a `ReplicationMismatch` raised by the check | same, or differing exactly as the mismatch reports; record set |
| | | `P3`–`K` | same; no record; the new wavenumber's shard key missing, re-assigned on its next get (below) |
| get, inserts or updates (vectorized) | `P1`, `C`, `R1`…`Rn`, check, `P3`, `K` | as the scalar get | as the scalar get, `store_id` NULL throughout |
| get, inserts nothing | `P1`, `C` (writes nothing), `P3` | `P1`–`C`, `C`–`P3` | same; record set |
| store (`wavenumber_exit_time`, `BackgroundModel`) | `P1`, `C`, `P2`, `R1`…`Rn`, check, `P3` | `P1`–`C` | same; record set, `store_id` NULL |
| | | `C`–`P2`, `P2`–`R1`, `Ri`–`Ri+1` | shards differ; record set (`store_id` from `P2`) |
| | | `Rn`–`P3` | same, or as the mismatch reports; record set |
| validate, controller `True` | `P1` (`store_id` set), `C`, `R1`…`Rn`, check, `P3` | `P1`–`C` | same; record set |
| | | `C`–`R1`, `Ri`–`Ri+1` | shards differ (`validated`); record set |
| | | `Rn`–`P3` | same, or as the mismatch reports; record set |
| validate, controller `False`/`None` | `P1`, `C`, `P3` | `P1`–`C`, `C`–`P3` | same; record set |
| | | after `P3` | same: the controller wrote `validated = False`, and every replica holds the `False` that `store()` wrote, because only `validate()` sets it `True` and a `False` from the controller means its value count is short, which row-identical copies written in one transaction cannot be (observation 1) |
| any path, a failure the driver sees after `P1` (the controller's call, the check, `P2`, an exception in the path's closure) | — | — | the record stays set whether or not the shards differ |
| a write that finds the record set at `P1` | nothing | — | nothing written anywhere; `ReplicationInFlight` |

**No row reads "shards differ, no record".** Every point at which the shards can differ lies
strictly between `P1` and `P3`, and `P1` commits before the controller's call is submitted. No
transaction is held across an actor call: each of `P1`, `P2`, `P3` is a separate, short
transaction on the pool's own engine.

## Cost of the in-flight record per replicated call

Measured on a stand-in pool (three shards, temporary directory under `/tmp`, macOS, SQLite
`synchronous=FULL` in rollback-journal mode), 300 repetitions each, by a scratchpad script that
runs the helper's record statements on `pool._engine` and times whole calls:

| What | Median | Mean |
|---|---|---|
| record insert + delete — a replicated call after which nothing replicates (2 primary commits) | 0.81 ms | 0.92 ms |
| record insert + `store_id` update + delete — a call that replicates (3 primary commits) | 1.24 ms | 1.30 ms |
| a whole replicated get that inserts nothing, stand-in actors (no Ray overhead) | 0.95 ms | 0.98 ms |
| a whole replicated get that inserts, on three shards (100 calls) | 2.86 ms | — |

So the record costs about 0.8 ms per replicated call that replicates nothing and 1.2 ms per call
that does. On stand-ins it dominates a no-insert get, because a stand-in call costs microseconds;
under Ray a remote call costs of order a millisecond already. The prompt accepts a primary commit
per replicated call; this is two to three.

## Does `_assign_shard_keys` self-heal?

**Yes.** `_assign_shard_keys` runs after the replication, in its own transaction on the primary,
and is not covered by the record (the prompt says to leave it). A crash after the record is cleared
and before its commit leaves every shard holding the new wavenumber and the primary lacking its
shard key. That heals, for three reasons read in the code:
1. `_get_impl_replicated_table` calls `_assign_shard_keys(objects)` on **every** get of the shard
   key class, whether or not anything was inserted, and `_assign_shard_keys` assigns every
   `store_id` absent from `self._shard_keys`, which the constructor loads from the primary.
2. Every run of `main.py` gets its whole wavenumber grid through the pool before any sharded work
   (`convert_to_wavenumbers`, a vectorized `object_get("wavenumber", payload_data=...)`, which is
   how every `k` gets a `store_id`), so the missing key is assigned at startup.
3. No sharded row can exist under the lost assignment. The in-memory `_shard_keys` entry is made
   inside the same `object_get` call, just before the commit, and the call returns only after the
   commit; a crash before the commit kills the driver before any sharded work for that `k` is
   submitted. So a re-assignment to a different shard orphans nothing.

## Deviations from the prompt

1. **`ReplicationInFlight`, a second exception** — `IMPLEMENTATION CHOICE`. The prompt says "raise,
   naming it" for a record found set, and names no class. A class of its own (a `RuntimeError`
   subclass carrying the record as a dict) lets prompt 02 and tests tell it apart from a mismatch.
   It lives beside `ReplicationMismatch`.
2. **The record is written before every replicated call, not only when a replication follows** —
   `IMPLEMENTATION CHOICE`, the one the prompt offers when the other opens a window. Whether a get
   inserts is known only from the controller's answer, after its commit; writing the record then
   would be the window the prompt forbids. Cost measured above.
3. **Any exception after the record is committed leaves it set, including a controller failure** —
   `STRUCTURALLY REQUIRED`. The driver cannot tell a controller that raised before committing from
   one that committed and then died (a `RayActorError` after the commit, which the test "a
   controller that dies after its commit" models). Clearing the record on a controller failure
   would reopen "shards differ, no record". The cost is that an ordinary exception inside a
   controller's `build()` or `store()` also leaves the record set, so the pool refuses further
   replicated writes until the store is reconciled (prompt 02). Such an exception already stops
   `main.py`.
4. **A vectorized get that inserts nothing no longer calls the replicas with an empty payload** —
   `IMPLEMENTATION CHOICE`. The old code sent every replica `object_get(cls, payload_data=[])`
   whatever the controller found. The prompt: "a get that inserts nothing needs no replication".
   No result changes.
5. **`store_id`** — `IMPLEMENTATION CHOICE`. It is written at `P1` for a validate (the serial is
   known), filled in at `P2` for a scalar get and a store, once the controller has answered and
   before any replica is called, and left NULL for a vectorized get, which has several.
6. **`started` is the write's own timestamp** — `IMPLEMENTATION CHOICE`. The record's start time is
   the timestamp every row of the write carries, so prompt 02 can recognise the in-flight rows of
   a timestamped class by it.
7. **The validate path passes no timestamp** — `IMPLEMENTATION CHOICE`. `validate()` inserts
   nothing; `object_validate` is unchanged.
8. **A controller validation outcome of `False` is not replicated, as before** — `IMPLEMENTATION
   CHOICE`, following W2's "as today". Why no window results is in the table's last validate row.
9. **`BackgroundModel`'s "key held under another serial" counts validated rows with exactly the
   same tags** — `IMPLEMENTATION CHOICE`. `build()` serves only validated rows and matches tags,
   and a store interrupted after its replication but before its validation leaves an unvalidated
   row on **every** shard; the next run's store then legitimately writes a second row beside it on
   every shard (the controller's own store does not look, and must not change). Counting
   unvalidated rows would make every replica refuse that resume. Ignoring tags would refuse two
   models that differ only in run label. So the check is for a row a replica's `build()` would
   serve in place of the controller's. A test pins both halves. The exit time has no `validated`
   column and its exact key cannot be stored twice through `build()`, so every row counts.
10. **What a replica compares** — `IMPLEMENTATION CHOICE`. Every column `store()` writes, exactly
    (floats bit for bit, two NaNs equal: the copies come from one pickled object). Not the
    `timestamp`, which a verified row keeps from its own insert, nor `version`, which each shard
    sets from its own version row (the constructor already checks those agree).
11. **`Datastore.object_store` names the shard on a mismatch** — `IMPLEMENTATION CHOICE`, so that a
    replica's refusal names the shard as W2 asks, although the factory raising it cannot.
12. **The stand-in pool is a reusable helper module, `Datastore/tests/standin_pool.py`, separate
    from the test module** — `IMPLEMENTATION CHOICE` within the prompt's latitude, so that prompts
    02 and 06 import it rather than copy it. It follows
    `docs/datastore-integrity-audit/replicated_write_fault_probe.py` (which is not edited), with
    every patch scoped by `unittest.mock.patch` through `StandinCluster.active()`.
13. **Stand-in faults are delivered by `ray.get`, not raised by `.remote()`** —
    `STRUCTURALLY REQUIRED` by §3 ("the patched `ray.get` raising only after every submitted call
    has run, as Ray does"). The probe raised from `.remote()`, which aborts the submission of later
    replicas. The helper also offers an `"after"` fault (run, commit, then fail) and call hooks.
14. **The table is created on an existing store after `_check_shard_files`** — `IMPLEMENTATION
    CHOICE` of placement, so a store the constructor refuses is not written. Prompt 02 adds its
    step after `_check_shard_files` too, and decides the order.
15. **Overlap is detected by an in-memory flag as well as the table** — `IMPLEMENTATION CHOICE`. The
    table alone cannot tell an overlapping call in this driver from a record left by a crash; the
    flag makes the former a programming error with its own message, as W1 asks.
16. **`BackgroundModel.store`'s unit constants moved above the insert, and its value payload into a
    local function** — `IMPLEMENTATION CHOICE`, so the replica's verification builds the same dict
    the insert writes. The controller's rows are unchanged; the clean-write test compares them
    column by column with the replicas'.

No `UNINTENDED DRIFT` was found. No factory's `build()`, no lookup, no sharded write, no `.scalar()`
lookup, nothing in the closed-store block of `ShardedPool.py`, `Datastore/store_inventory.py`,
`RunRegistry/` or any existing test is changed. In the two factories every hunk lies in `store()`.

## Verification performed

- `git status --short` and `git diff --cached` empty at `65cd4b7` before starting.
- `PYTHONPATH=. ./venv/bin/python -m unittest Datastore.tests.test_replicated_write`: 26 tests OK,
  about 15 s. Run repeatedly while developing.
- The six suites, after the change, with README §7's command:

  | Suite | Result |
  |---|---|
  | AdaptiveLevin | 32 OK |
  | ComputeTargets | 552 OK (the known wall-clock flake did not fire) |
  | CosmologyModels | 39 OK |
  | Datastore | **232 OK** (206 + 26) |
  | LiouvilleGreen | 148 OK (1 skipped) |
  | RunRegistry | 199 OK (the golden fingerprint did not move) |

- `./venv/bin/python -m black --check .` clean.
- The audit's probe `replicated_write_fault_probe.py` was run once, unmodified, before any change
  (it writes only to a temporary directory under its own directory, which it deletes).
- No test, script or manual check opened anything under `var/`. The stand-in stores and the cost
  measurement ran in temporary directories. `python -m RunRegistry list` was not needed: nothing
  long was launched and no store was touched.
- Scope, checked with `git diff --cached -U0`: in `wavenumber.py` one hunk and in
  `BackgroundModel.py` three, all inside `store()`; in `ShardedPool.py` the hunks are the imports,
  `_Replication`, the constructor's two additions, `_create_engine`, `_write_shard_data` and the
  replicated get/store/validate paths — none in `closed_store_files`, `delete_store`, `copy_store`,
  `move_store` or their helpers.

## The deliberate-breakage record

Each diff below applies with `git apply` from the repository root to this commit, and was checked
with `git apply --check` against it. Each was applied, the new module run, and the file restored
with `git checkout`. None is committed.

**(i) Discard the replicas' answers again.**

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 25dda3d..8891b4c 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -1299,7 +1299,7 @@ class ShardedPool:
 
                 # 4. every answer checked against the controller's, before the record is cleared
                 for (key, _), replica_answer in zip(replica_refs, replica_answers):
-                    plan.check(key, replica_answer)
+                    pass
 
             # 5. every shard holds what the controller holds: clear the record
             with self._engine.begin() as conn:
```

2 failures: `TestReplicaSerialMismatch.test_scalar_get`, `TestReplicaSerialMismatch.test_vectorized_get`.
(The store-path split still raises, from the replica's own `store()`, which is W3's check.)

**(ii) Write the record after the controller's call.**

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 25dda3d..b2e170d 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -1256,6 +1256,10 @@ class ShardedPool:
 
             record_table = self._replication_in_flight_table
 
+            # 2. the controlling shard
+            ref = submit_controller(self._shards[controller], timestamp)
+            answer = ray.get(ref)
+
             # 1. the record, committed before the controlling shard's call is submitted
             with self._engine.begin() as conn:
                 existing = conn.execute(sqla.select(record_table)).mappings().first()
@@ -1272,10 +1276,6 @@ class ShardedPool:
                     },
                 )
 
-            # 2. the controlling shard
-            ref = submit_controller(self._shards[controller], timestamp)
-            answer = ray.get(ref)
-
             # 3. the replicas, if anything is to be replicated
             plan = replication(controller, answer)
             if plan is not None:
```

6 failures in 4 tests: `TestRecordAcrossTheWindow.test_a_controller_that_dies_after_its_commit_leaves_the_record`
(all three sub-tests: `tolerance`, `wavenumber_exit_time`, `BackgroundModel`),
`TestRecordAcrossTheWindow.test_the_record_is_on_disk_during_every_shards_call`,
`TestWriteOverASetRecord.test_every_path_refuses`, `TestWriteOverASetRecord.test_a_reopened_pool_refuses`
(the refused write reaches the controller before it is refused).

**(iii) Clear the record before the last replica returns.** The record is deleted as soon as the
replicas are submitted, before their answers are read and checked; the final delete then tolerates
finding nothing.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 25dda3d..4af155f 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -1295,6 +1295,9 @@ class ShardedPool:
                     (key, plan.submit(self._shards[key], timestamp))
                     for key in shard_ids
                 ]
+                # the record cleared as soon as the replicas are submitted
+                with self._engine.begin() as conn:
+                    conn.execute(sqla.delete(record_table))
                 replica_answers = ray.get([r for _, r in replica_refs])
 
                 # 4. every answer checked against the controller's, before the record is cleared
@@ -1304,7 +1307,7 @@ class ShardedPool:
             # 5. every shard holds what the controller holds: clear the record
             with self._engine.begin() as conn:
                 result = conn.execute(sqla.delete(record_table))
-                if result.rowcount != 1:
+                if result.rowcount != 1 and plan is None:
                     raise RuntimeError(
                         f"ShardedPool: the replication_in_flight record of a replicated "
                         f'{operation} of "{cls_name}" has {result.rowcount} rows, expected 1'
```

16 failures in 11 tests: `TestRecordAcrossTheWindow.test_scalar_get`, `test_vectorized_get`,
`test_store_of_an_exit_time`, `test_store_of_a_background_model`,
`test_validate_of_a_background_model` (each for both replicas); `TestReplicaSerialMismatch.test_scalar_get`,
`test_vectorized_get`, `test_store_over_a_replica_holding_the_key_under_another_serial`;
`TestWriteOverASetRecord.test_every_path_refuses`, `test_a_reopened_pool_refuses`,
`test_overlapping_writes_are_a_programming_error` (their `setUp` finds no record left by the
interrupted write).

**(iv) Let each shard stamp its own timestamp.**

```diff
diff --git a/Datastore/SQL/Datastore.py b/Datastore/SQL/Datastore.py
index 307b224..44bc9b6 100644
--- a/Datastore/SQL/Datastore.py
+++ b/Datastore/SQL/Datastore.py
@@ -636,7 +636,7 @@ class Datastore:
                 # a replicated write stamps every shard with the driver's one timestamp
                 # (object_get / object_store); every other insert stamps its own time
                 timestamp = (
-                    insert_timestamp if insert_timestamp is not None else datetime.now()
+                    datetime.now()
                 )
                 payload = payload | {"timestamp": timestamp}
             if uses_stepping:
```

2 failures: `TestCleanWrite.test_every_copy_is_row_identical` (the copies' `timestamp` columns
differ), `TestShardedWritesUnchanged.test_a_replicated_row_does_not_take_the_shards_time`.

**(v) Let a replica store insert without verifying.**

```diff
diff --git a/Datastore/SQL/ObjectFactories/BackgroundModel.py b/Datastore/SQL/ObjectFactories/BackgroundModel.py
index 07f82de..6c5d74f 100644
--- a/Datastore/SQL/ObjectFactories/BackgroundModel.py
+++ b/Datastore/SQL/ObjectFactories/BackgroundModel.py
@@ -678,7 +678,7 @@ class sqla_BackgroundModelFactory(SQLAFactoryBase):
         #     it, write nothing;
         #   - a row with this serial that differs, or the key held under another serial: raise.
         # The controlling shard's own store (no serial yet) is unchanged.
-        if replica:
+        if False:
             from Datastore.replication import ReplicationMismatch, differing_columns
 
             serial = payload["serial"]
diff --git a/Datastore/SQL/ObjectFactories/wavenumber.py b/Datastore/SQL/ObjectFactories/wavenumber.py
index 2a82376..412bfb7 100644
--- a/Datastore/SQL/ObjectFactories/wavenumber.py
+++ b/Datastore/SQL/ObjectFactories/wavenumber.py
@@ -320,7 +320,7 @@ class sqla_wavenumber_exit_time_factory(SQLAFactoryBase):
         #   - a row with this serial whose columns equal the object's: return it, write nothing;
         #   - a row with this serial that differs, or the key held under another serial: raise.
         # The controlling shard's own store (no serial yet) is unchanged.
-        if "serial" in payload:
+        if False:
             from Datastore.replication import ReplicationMismatch, differing_columns
 
             serial = payload["serial"]
```

3 failures and 6 errors, in 7 tests: in `TestIdempotentReplicaStores`, `test_exit_time_replay_writes_nothing_and_returns_the_row`,
`test_exit_time_differing_row_raises`, `test_background_replay_writes_nothing_and_returns_the_row`
and the three sub-tests of `test_background_differing_row_raises` (each an `IntegrityError`, the
serial being taken), `test_exit_time_key_under_another_serial_raises` and
`test_background_key_under_another_serial_raises`; and
`TestReplicaSerialMismatch.test_store_over_a_replica_holding_the_key_under_another_serial`.

## Observations not acted on

1. **A controller validation of `False` is not replicated** (deviation 8). If a model were ever
   re-validated after being validated everywhere, and the controller alone counted short, the
   controller would hold `validated = False` and the replicas `True`, with no record. No caller
   re-validates a stored model (`main.py:1190-1191` validates once, right after storing), and a
   short count on one shard needs value rows deleted on that shard alone, which nothing in the
   write path does (only `--prune-unvalidated` deletes, and only unvalidated rows). Not opened as an
   issue: it is not reachable. Recorded so that a later change adding re-validation replicates a
   `False` too.
2. **Until prompt 02 lands, a store whose record is left set refuses every replicated write**, and
   nothing but removing the row by hand clears it. That is the intended hand-over (W1: "that state
   belongs to prompt 02's check at open"). No store under `var/` is written before prompt 10.
3. **Opening an existing store with this tree adds the empty `replication_in_flight` table to its
   primary.** The read-only readers (`Datastore/store_reader.py`, the inventory, the fingerprint)
   open stores `mode=ro` and do not; they read shard tables, not the primary's, so the fingerprint
   is unaffected (RunRegistry's golden fingerprint did not move).
4. **Nothing here has run under Ray.** Ray 2.43 re-raises an actor's exception as a class deriving
   from both `RayTaskError` and the cause, carrying the cause's attributes, so
   `except ReplicationMismatch` and its fields should work in the driver; keyword-only actor-method
   arguments bind through `inspect.Signature.bind` in Ray's `validate_args`. Prompt 10 is the first
   real run.

## State handed to the next prompt

- **For prompt 02.** `replication_in_flight` in the primary holds at most one row: `operation`
  (`get`, `store`, `validate`), `class_name`, `controller_shard`, `store_id` (NULL for a
  vectorized get, and for a get or store interrupted before `P2`), `started`. `started` equals the
  `timestamp` of every timestamped row the interrupted write inserted, on every shard it reached.
  The commit-point table above is the reopen table's input: every state with the record set is one
  of "same" or "some shards lack what the controller holds" (or, after a `ReplicationMismatch`,
  the difference it names). A pool with the record set opens today and refuses every replicated
  write with `ReplicationInFlight`; 02 decides at open.
- **Stand-in pool.** `Datastore/tests/standin_pool.py`: `StandinCluster` (`active()`,
  `open_pool`, `close_pool`, `controller`, `replica_ids()`, `fault(shard, method, class, "before"
  | "after")`, `hooks`, `calls`), `shard_rows`, `shard_snapshot`, `in_flight_records`, and object
  builders (`make_exit_time`, `make_background_model`, `make_policy_data`, `get_tolerance`,
  `get_wavenumber`, `get_redshifts`, `StandinCosmology`, `StandinSerial`).
- **Names.** `ShardedPool._replicated_write`; `Datastore.replication.ReplicationMismatch` and
  `ReplicationInFlight`; the override `insert_timestamp` on `Datastore.object_get`,
  `object_store` and `_insert`.
- **Baselines** after this commit: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39,
  Datastore **232**, LiouvilleGreen 148 (1 skipped), RunRegistry 199.
