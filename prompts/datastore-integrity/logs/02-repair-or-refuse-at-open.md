# Log 02 — Repair or refuse divergent replicated tables when a store opens

**Prompt:** [`prompts/datastore-integrity/02-repair-or-refuse-at-open.md`](../02-repair-or-refuse-at-open.md)
**Commit:** *(this commit)* — "Repair or refuse divergent replicated tables when a store opens"
**Model:** Claude Opus 5.5
**Date:** 2026-09-28
**Result:** COMPLETE. W5–W7 shipped. No §5 stop condition was met:
- the check runs before the broker or any actor exists, and needs no Ray;
- no repair deletes a row;
- every difference met in a test built only through the real write path, with no injection, was
  classified (there were none: every clean write left the shards identical);
- no existing test changed except the one the orchestrator note names, rewritten by the user's
  decision of 2026-09-28;
- nothing under `var/` was opened.

The new module `Datastore/tests/test_reconcile_at_open.py` adds 38 tests, so `Datastore` rises from
287 to 325, and every other suite is unchanged.
- **Closed:** `[00-replicated-writes-can-diverge-across-shards]`, on the `store-fingerprint`
  board's §4.
- **Opened:** `[02-the-version-row-is-written-outside-the-in-flight-record]`, on this campaign's
  board §3.3 (observation 1). The index is at **101** (one row deleted, one added).

## What shipped

All in `Datastore/SQL/ShardedPool.py` unless named otherwise. Every function is cited by name.

**The constructor.**
- `self.reconciliation` is `None` until the check runs, so a store the constructor creates has
  none.
- For an existing store, whether a journal lies beside the primary is noted before `_create_engine`.
  The primary's first read-write open, in `_read_shard_data`, rolls such a journal back, so after
  it nothing would show one had been there.
- After `_check_shard_files`, the constructor calls
  `self.reconciliation = self._reconcile_replicated_tables(primary_journal)`. That is before
  `replication_in_flight.create(checkfirst=True)`, before the broker and before any actor.
- Nothing else in the constructor moved.

**W5 — the comparison.** `ShardedPool._reconcile_replicated_tables` and the new methods it calls:
`_replicated_table_specs`, `_reconcile_engine`, `_read_in_flight_record`,
`_read_replicated_tables`, `_same_row`, `_replicated_differences`, `_key_text`,
`_describe_differences`, `_plan_replicated_repair`, `_read_full_rows`, `_apply_replicated_repair`,
`_recompute_validated` and `_print_reconciliation`. In order, the check does four things.
1. **Hot journals.** Each shard file is opened once with plain `sqlite3`, read-write, and
   `SELECT count(*) FROM sqlite_master` is run, so SQLite rolls back a hot journal. The file is
   then closed. `reconciliation["hot_journals"]` lists every file that had a `-journal` beside it
   when the store was opened (the primary, from the constructor's note, then the shards), and one
   line is printed for each. On a clean shard this writes nothing (test 7).
2. **The record.** `_read_in_flight_record` reads the primary's `replication_in_flight` through the
   pool's own engine.
   - A primary without the table, a store written before prompt 01, reads as "no record".
   - More than one row refuses.
3. **The tables.** Every replicated table is read from every shard, through tables built by
   `build_schema(sqla.MetaData(), _factories)`. Each shard is read on its own engine
   (`_reconcile_engine`: `NullPool`, the driver's implicit `BEGIN` off, an explicit `BEGIN` per
   transaction), and nothing goes through an actor.
4. **The comparison.** `_replicated_differences` compares the shards and returns every difference.

**The comparison's classes, keys and exclusions, exactly.** The tables compared are the
constructor's `replicated_tables` that have a table, in their order, then the tag table of every
replicated class. `_replicated_table_specs` finds those as the tables that are neither replicated
nor sharded and whose foreign keys, other than `store_tag`'s, all name replicated tables. On this
tree that is exactly `BackgroundModel_tags`, and `TestCleanStoreUntouched` pins the list.

| Table | Key | Not compared |
|---|---|---|
| `version` | `serial` | — (it has no `timestamp`) |
| `store_tag`, `redshift`, `wavenumber`, `wavenumber_exit_time`, `tolerance`, `LambdaCDM`, `QCD_Cosmology`, `IntegrationSolver`, `BackgroundModel`, `GkSourcePolicy`, `QuadSourcePolicy` | `serial` | `timestamp` |
| `BackgroundModelValue` | `(model_serial, z_serial)` | `serial` (it has no `timestamp`) |
| `BackgroundModel_tags` | `(model_serial, tag_serial)`, its primary key | `timestamp` |

Every other column that `build_schema` declares for the table is compared, exactly: floats bit
for bit, with two NaNs the same value (`same_value`, from `Datastore/replication.py`).
- **Rows missing on some shards.** A key some shards hold and others lack is a `missing`
  difference.
- **Rows that differ.** A key whose compared values differ between the shards that hold it is a
  `differs` difference.
- **Columns.** A shard that lacks a table reads it as empty, as its actor would create it. Shards
  whose tables have different compared columns are a `columns` difference.
- **Duplicates.** A key held by two rows on one shard is a `duplicate` difference. Only
  `BackgroundModelValue`'s key can repeat.
- **The primary's `shard_keys`.** Every `key_serial` must be a `wavenumber` serial that every shard
  holds; one that is not is a `shard_key` difference.
- **Sharded tables.** None is read. The tag tables of sharded classes are not read either.
- **Speed.** A table whose shards hold identical row dicts is passed without a row-by-row
  comparison.

**What follows the comparison.**
- **No record, no difference.** Nothing is written, `reconciliation` records what was compared,
  and the store opens as before (test 7, by checksum of every file).
- **No record, any difference.** `ReplicatedDivergence` (W7).
- **A record.** `_plan_replicated_repair` decides, before anything is written, whether the record's
  interrupted write explains every difference (W6). If it does not, `ReplicatedDivergence`.

**W6 — the repair.** The licence, as `_plan_replicated_repair` applies it, holds for exactly
these repairs.
- **The controller, and the class.** Rows come only from the record's `controller_shard`, and only
  in the tables of the record's class. For `BackgroundModel` those are `BackgroundModel`,
  `BackgroundModel_tags` and `BackgroundModelValue`; for any other class, its own table.
- **Missing rows.** A key the controller holds and another shard lacks is copied to that shard,
  after a `get` or a `store` record only. The copy is verbatim: `_read_full_rows` reads every
  column the controller's table has, `serial` and `timestamp` included, and the row is inserted as
  read. A copied row with a `timestamp` column must carry the record's `started`, which prompt 01
  stamps on every row the write inserts (log 01, deviation 6). Under a `store_id`, the class's own
  row must be that serial.
- **`BackgroundModel` as a unit.** A shard lacking the model row receives every one of its tag and
  value rows. A shard holding the model row but lacking one of its tag or value rows, or holding
  part of a model it lacks, refuses. A value row's serial, the controller's, must be free on the
  shard.
- **Flags.** A `source` or `response` flag of `redshift` or `wavenumber` is set on another shard
  when the controller holds it `True` and the shard `False`, after a `get` record only (under a
  `store_id`, on that serial only). No flag is ever written `False`.
- **`validated`.** After a `validate` record, a `validated` difference on the record's model is
  admitted only when the controller holds `True` and the other shard `False`.
  `_recompute_validated` then runs the factory's own rule, `sqla_BackgroundModelFactory.validate`
  (`_factories["BackgroundModel"]`), on each shard holding the model, from that shard's own value
  rows, each call inside a savepoint.
  - A result that turns the flag on is kept.
  - Any other result is rolled back to its savepoint, so no `False` is ever left written.
  - A shard holding `True` whose rule gives `False` refuses.
  - Nothing is copied.
- **Anything else refuses:** a row the controller lacks; a difference in a table outside the
  record's class; a row the recorded write did not insert; any other column; a `columns` or
  `duplicate` difference; a record naming no operation, class or shard of this store.

**Writing the repair.**
- Each shard's writes are made inside one open transaction per shard, in the order of the specs,
  so a model row precedes its tags and values.
- Then the second comparison: every table re-read from every shard, through those same
  transactions, and compared as in W5, the `shard_keys` check included. Any difference rolls
  every shard back and refuses (`after_repair=True`).
- Only when they all agree is each shard committed, and then the record is deleted, in its own
  transaction on the primary, requiring exactly one row, last.
- A record over identical shards, one whose controller holds nothing the others lack, writes
  nothing to any shard and is simply cleared.
- `reconciliation["repaired"]` lists every action: `copied`, `flags set` or
  `validated recomputed`, with its class, shard, source shard and keys. One line is printed per
  class and shard, between a line naming the record and a line saying it was cleared.

**W7 — the refusal.** `ReplicatedDivergence` is in `Datastore/replication.py`, beside
`ReplicationMismatch` and `ReplicationInFlight`. It is a `RuntimeError` carrying `primary`,
`differences`, `record` (or `None`) and `after_repair`, and it pickles with its fields.
- **The differences.** Each is a dict naming `class_name`, `shard` (the shards that deviate),
  `key` (serial or key) and `detail` (what differs, and under a record why the record does not
  license it).
- **A serial split.** Rows missing under different serials whose other compared values are equal
  are named as a serial split, with both serials, whether or not a record is set.
- **The message** says nothing was repaired, and that a store in this state needs a person, or
  regeneration.
- **Where it is raised.** Always before any actor exists. When the store has a record, the record
  is left as it was.

**What runs after.**
- `prune_unvalidated` and `drop_actions` run inside each `Datastore` actor, in its constructor,
  after this step (`Datastore.__init__`: `_drop_actions`, `_ensure_tables`,
  `_validate_on_startup`).
- By then every shard's replicated tables are identical, or the open was refused. So pruning, which
  deletes a `BackgroundModel` that is not validated, with its tags and values, acts identically
  on every shard.
- `drop_actions` drop whole tables on each shard, and `_ensure_tables` recreates them empty. One
  action, `gk-source-policy-records`, drops a compared table, the replicated `GkSourcePolicy`.
  Each actor drops the same tables on its own shard, so that too acts identically. The check runs
  first, so a divergence in a table about to be dropped still refuses; that is fail-closed.
- `TestPruningAfterRepair` reopens an interrupted validate and an interrupted store with
  `prune_unvalidated` true and false. The second also runs
  `drop_actions=["gk-source", "gk-source-policy-records"]` over a stored `GkSourcePolicy` row.
  Every shard ends identical. The store's copied model is pruned from every shard, or kept on
  every shard, and `GkSourcePolicy` is empty on every shard.

**Every `ShardedPool(` caller now runs the check at open** (`git grep -n "ShardedPool(" -- '*.py'`):
- `main.py:3516`;
- `extract_GkSource_data.py:924`, `extract_GkWKB_data.py:497`, `extract_Gk_data.py:449`,
  `extract_QuadSourceIntegral_data.py:1175`, `extract_TkWKB_data.py:484`,
  `extract_tensor_source_data.py:406`;
- `docs/source-remediation-verification/analyse_greens_and_source.py:430`,
  `docs/source-remediation-verification/run_quadsource_integrals.py:123`;
- `docs/datastore-integrity-audit/replicated_write_fault_probe.py:120`;
- `Datastore/tests/standin_pool.py`: `open_pool` and the new `open_pool_output`.

`docs/handover/quadsource_atol_sweep.py`, `docs/source-remediation-verification/scoped_pipeline_run.py`
and `docs/gktk-remedial/scoped_pipeline_run.py` `exec` `main.py`, so they reach it through
`main.py:3516`. `Datastore/tests/test_inventory_consumers.py` names the statement in an `ast` test
and constructs nothing.

**Tests.** The new module `Datastore/tests/test_reconcile_at_open.py` has 38 tests.

| § of the prompt | Tests |
|---|---|
| 1. kill and reopen, every point | `TestKillAndReopen`, 12 tests, one or more sub-tests per commit point (below). `run_case` checks each reopen, then a second reopen. |
| 2. a serial split refuses | `TestSerialSplit`: the probe's step-4 state with the record set and with it deleted. Each refusal names both serials and says "serial split", no actor is constructed (`_Options.remote` counted), and every file of the store has the same checksum after. The exception pickles. `TestMismatchLeftByTheCheck`: the other states a `ReplicationMismatch` leaves with the record set (a vectorized get's split, a store's split, a validate that counts short), each refused. |
| 3. no record, any difference, refuses | `TestNoRecordRefuses`: a missing row, an extra row, a flag, `validated`, a value count, each refused with nothing changed. A store without `replication_in_flight` is refused, and the table is not created. A hand edit in a sharded table is neither refused nor repaired. |
| 4. a row the controller lacks refuses | `TestRecordDoesNotExplain`, all with the record left set and nothing changed: a row the controller lacks; another class; a row whose timestamp is not the record's start; a flag the controller lacks; part of a `BackgroundModel`; a `validated` the factory's rule does not support; a `shard_keys` row the repair does not supply, refused by the second comparison and rolled back. |
| 5. a hot journal | `TestHotJournal`: `os.fork`, `BEGIN IMMEDIATE`, `os._exit`, as `hot_journal_probe.py` does it, on one shard (rolled back, compared, opened) and on the primary (rolled back before the record is read). |
| 6. pruning after repair | `TestPruningAfterRepair`, above. |
| 7. a clean store is untouched | `TestCleanStoreUntouched`: a store holding every kind of write, a sharded row included, opens with every file's SHA-256 unchanged, nothing repaired or printed, and exactly the compared list above. A new store has `reconciliation is None`. |
| 8. no Ray | `tearDownModule` raises if `ray.is_initialized()`, and every test's `tearDown` asserts it is false. |

For each case, `run_case` checks:
- every column of every row of the 14 tables is identical on every shard;
- the controller's rows are unchanged, unless a flag was recomputed;
- the record is empty;
- `reconciliation["cleared"]` is set, with the record and the controller;
- the set of `(action, class, replica)` equals the case's expectation, and every copy came from
  the controller;
- each copy's keys are exactly the keys the shard lacked;
- one printed line per action.

A second reopen repairs nothing and changes no file's checksum.

**The one rewritten test** (README §5 rule 7; the user's decision, 2026-09-28, named in the
orchestrator note). `test_replicated_write.py`'s `TestWriteOverASetRecord.test_a_reopened_pool_refuses`
becomes `test_a_reopened_pool_repairs_from_the_controller_and_the_write_succeeds`, on the same
fixture: the tolerance get whose replica 0 died before writing. Reopened, the pool:
- copies the tolerance row from the record's controller to replica 0;
- lists that copy, and only it, in `reconciliation["repaired"]`;
- clears the record, and leaves every shard's `tolerance` rows identical.

The write refused before, `get_tolerance(1e-4)`, then succeeds on every shard. The class's other
three tests are unchanged, and they still refuse without a reopen. No other existing test reopens a
pool over a set record or over diverged shards. Among the existing test modules,
`git grep open_pool` finds only `test_replicated_write.py`, whose other reopen, `test_an_existing_store_gains_an_empty_record_table_when_opened`,
reopens a clean store.

**`Datastore/tests/standin_pool.py`: two additions, no existing function changed.**
- `StandinCluster.open_pool_output(primary, shards=3, **kwargs)` returns `(pool, printed)`. It
  makes the same constructor call as `open_pool`, which discards what the constructor prints.
- `store_checksums(primary)` gives the SHA-256 of every file in the primary's directory: the
  primary, its shards and any journal.

## The reopen table

One row for each row of log 01's "After" table, in its order.
- **Outcomes.** Each state gets one outcome: **repaired** (and from where), **cleared** (the record
  deleted, nothing else written), or **refused**. A state with no record has nothing to repair or
  clear, and **opens clean**, writing nothing.
- **Rows with more than one state.** The rows are numbered here, in log 01's order. Rows 6, 12, 15
  and 18 are log 01 rows that name more than one disk state ("same, or differing exactly as the
  mismatch reports"; "whether or not the shards differ"), and row 8 is log 01's "as the scalar
  get". There each state, or each commit point, is given its own single outcome.
- **Tests.** Test names are in `test_reconcile_at_open.py` unless marked. `TKR` is
  `TestKillAndReopen`, and a quoted point is its sub-test.

| # | Log 01 row: path — crash between | On disk (log 01) | Outcome at reopen | Exercised by |
|---|---|---|---|---|
| 1 | every path — before `P1` (or inside it) | same; no record | **opens clean.** A crash inside `P1` leaves a journal on the primary, rolled back by the constructor's first read of it, and recorded | `TestCleanStoreUntouched.test_opening_a_clean_store_writes_nothing`; `TestHotJournal.test_a_hot_journal_on_the_primary_is_rolled_back_before_the_record_is_read` |
| 2 | scalar get — `P1`–`C` | same; record set, `store_id` NULL | **cleared** | `TKR.test_scalar_get` "P1-C" |
| 3 | scalar get — `C`–`P2` | the controller holds the row; record, `store_id` NULL | **repaired** from the controller to every replica | `TKR.test_scalar_get` "C-P2" |
| 4 | scalar get — `P2`–`R1` | shards differ; record with the serial | **repaired** from the controller to every replica | `TKR.test_scalar_get` "P2-R1" |
| 5 | scalar get — `Ri`–`Ri+1` | the controller and some replicas hold it | **repaired** from the controller to the replicas lacking it (a row copied, or a flag set) | `TKR.test_scalar_get` "Ri-Ri+1, r0 lacks" / "r1 lacks"; `TKR.test_scalar_get_of_the_shard_key`; `TKR.test_scalar_get_that_turns_a_flag_on`; `test_replicated_write.TestWriteOverASetRecord.test_a_reopened_pool_repairs_from_the_controller_and_the_write_succeeds` |
| 6 | scalar get — `Rn`–`P3`, including a `ReplicationMismatch` | (a) same; record set | **cleared** | `TKR.test_scalar_get` "Rn-P3" |
| | | (b) differing as the mismatch reports (a split) | **refused** | `TestSerialSplit.test_with_the_record_set` |
| 7 | scalar get — `P3`–`K` | same; no record; a shard key missing | **opens clean**; the next get assigns the key | `TKR.test_between_the_record_cleared_and_the_shard_key` |
| 8 | vectorized get — as the scalar get | as the scalar get, `store_id` NULL | `P1`–`C` **cleared**; `C`–`R1`, `Ri`–`Ri+1` **repaired** from the controller (rows copied, flags set); `Rn`–`P3` **cleared** if same, **refused** if the mismatch's split; `P3`–`K` **opens clean** | `TKR.test_vectorized_get` (all points); `TKR.test_vectorized_get_that_inserts_and_turns_a_flag_on`; `TestMismatchLeftByTheCheck.test_a_vectorized_get`; `P3`–`K` by row 7 (the same state) |
| 9 | get, inserts nothing — `P1`–`C`, `C`–`P3` | same; record set | **cleared** | `TKR.test_get_that_inserts_nothing` "P1-C", "C-P3" |
| 10 | store — `P1`–`C` | same; record set, `store_id` NULL | **cleared** | `TKR.test_store_of_an_exit_time`, `TKR.test_store_of_a_background_model`, each "P1-C" |
| 11 | store — `C`–`P2`, `P2`–`R1`, `Ri`–`Ri+1` | shards differ; record set | **repaired** from the controller to every shard lacking the row. A `BackgroundModel` is copied as a unit: row, tags and values | both store tests, "C-P2", "P2-R1", "Ri-Ri+1, r0 lacks", "r1 lacks"; `TKR.test_store_of_a_second_background_model_beside_the_first`; `TestPruningAfterRepair.test_an_interrupted_store` |
| 12 | store — `Rn`–`P3` | (a) same; record set | **cleared** | both store tests, "Rn-P3" |
| | | (b) as the mismatch reports | **refused** | `TestMismatchLeftByTheCheck.test_a_store` |
| 13 | validate, controller `True` — `P1`–`C` | same (all `False`); record set | **cleared**: the controller holds `False`, so nothing is turned on | `TKR.test_validate_of_a_background_model` "P1-C" |
| 14 | validate, controller `True` — `C`–`R1`, `Ri`–`Ri+1` | shards differ (`validated`); record set | **repaired**: `validated` recomputed by the factory's rule on each replica lacking it. The value is the shard's own; the controller's `True` is where it may be turned on | `TKR.test_validate_of_a_background_model` "C-R1", "Ri-Ri+1, r0 lacks" / "r1 lacks"; `TestPruningAfterRepair.test_an_interrupted_validate` |
| 15 | validate, controller `True` — `Rn`–`P3` | (a) same; record set | **cleared** | `TKR.test_validate_of_a_background_model` "Rn-P3" |
| | | (b) as the mismatch reports (a replica counting short) | **refused** | `TestMismatchLeftByTheCheck.test_a_validate` |
| 16 | validate, controller `False`/`None` — `P1`–`C`, `C`–`P3` | same; record set | **cleared** | `TKR.test_validate_whose_controller_counts_short` "P1-C", "C-P3" |
| 17 | validate, controller `False`/`None` — after `P3` | same; no record | **opens clean** | the second reopen inside `TKR.test_validate_whose_controller_counts_short` (that state exactly), and `TestCleanStoreUntouched` |
| 18 | any path, a failure the driver sees after `P1` | the record stays set, whether or not the shards differ | (a) the shards are the same (a controller that raised before committing): **cleared**. (b) the controller committed and some shards lack it: **repaired** from the controller. (c) a failed check left the difference it reports: **refused** | (a) every "P1-C" sub-test (controller killed before it ran). (b) every "C-P2"/"C-R1" sub-test (controller killed after its commit). (c) `TestSerialSplit.test_with_the_record_set`, `TestMismatchLeftByTheCheck` |
| 19 | a write that finds the record set at `P1` | nothing written; `ReplicationInFlight` | the state is the earlier record's, one of rows 2–18. On log 01's fixture, a tolerance get whose replica died: **repaired** from the controller, and the refused write then succeeds | `test_replicated_write.TestWriteOverASetRecord.test_a_reopened_pool_repairs_from_the_controller_and_the_write_succeeds` |

Every state with the record set is one of: identical (cleared), the controller holding what some
shards lack (repaired from the controller), or the difference a failed check reports (refused). No
row is repaired from any shard but the record's controller.

## The added open time

**Method.** `measure_open.py`, a script in the agent's own scratch subdirectory. It builds a
stand-in store of production size in a temporary directory: two tolerances, a `store_tag`, 1 740
redshifts (`geomspace(1e-2, 1e12, 1740)`) and one `BackgroundModel` with 1 740 value rows,
validated. It does so at 4 and at 20 shards, then opens each store 7 times through
`StandinCluster.open_pool`.
- Each open timed `_reconcile_replicated_tables` alone, by a wrapping patch.
- Each open was timed whole.
- The same 7 opens were timed with the method patched to a no-op.

The machine was this macOS host: SQLite in rollback-journal mode, stand-in actors, no Ray. Every
open found no record and no difference.

| Shards | Store on disk | The check (median; min–max) | Whole open with the check (median) | Whole open, check a no-op (median) |
|---|---|---|---|---|
| 4 | 4.6 MB | **94 ms** (55–98 ms) | 180 ms | 119 ms |
| 20 | 23.1 MB | **252 ms** (229–411 ms) | 1 037 ms | 777 ms |

So a clean store of production size costs about 0.1 s more to open at 4 shards and 0.25 s at 20.
The time goes on reading every replicated row of every shard, of which the 1 740 value rows are
most. It is spent once per `ShardedPool(...)`, before Ray actors, which cost seconds to start.

## What a user sees

**A repair.** A three-shard stand-in store whose `BackgroundModel` store was killed before replica
shard 2 ran, reopened with the constructor's output shown. Pasted as printed, with the
`!! INTEGRITY WARNING` lines the actors print afterwards: the model was stored and never
validated, on every shard alike.

```
>> Sharded datastore "/private/tmp/claude-35086/-Users-ds283-Documents-Code-SecondaryGWKit/b173fde2-7910-4580-995f-b2ee721fe27e/scratchpad/agent02/outputs-0n5fp_iw/repair/store.sqlite" was opened with a replicated write in flight (operation="store", class="BackgroundModel", controlling shard=0, store_id=1, started=2026-09-28 22:22:44.168230)
>>   copied 1 row(s) of "BackgroundModel" from shard 0 to shard 2: 1
>>   copied 3 row(s) of "BackgroundModelValue" from shard 0 to shard 2: (1, 1), (1, 2), (1, 3)
>>   copied 1 row(s) of "BackgroundModel_tags" from shard 0 to shard 2: (1, 1)
>>   the replication_in_flight record was cleared
>> Opened existing sharded datastore "/private/tmp/claude-35086/-Users-ds283-Documents-Code-SecondaryGWKit/b173fde2-7910-4580-995f-b2ee721fe27e/scratchpad/agent02/outputs-0n5fp_iw/repair/store.sqlite" with 3 shards
!! INTEGRITY WARNING (2026-09-28T22:22:44): datastore "shard0002-store" (physical file /private/tmp/claude-35086/-Users-ds283-Documents-Code-SecondaryGWKit/b173fde2-7910-4580-995f-b2ee721fe27e/scratchpad/agent02/outputs-0n5fp_iw/repair/store-shard0002.sqlite)
>> Background models
     The following unvalidated models were detected in the datastore:
       -- "standin-background" (store_id=1, N_tau=4, N_cs_tau=4, N_F=4)
          contains 3 z-sample values | expected=3
!! INTEGRITY WARNING (2026-09-28T22:22:44): datastore "shard0000-store" (physical file /private/tmp/claude-35086/-Users-ds283-Documents-Code-SecondaryGWKit/b173fde2-7910-4580-995f-b2ee721fe27e/scratchpad/agent02/outputs-0n5fp_iw/repair/store-shard0000.sqlite)
>> Background models
     The following unvalidated models were detected in the datastore:
       -- "standin-background" (store_id=1, N_tau=4, N_cs_tau=4, N_F=4)
          contains 3 z-sample values | expected=3
!! INTEGRITY WARNING (2026-09-28T22:22:44): datastore "shard0001-store" (physical file /private/tmp/claude-35086/-Users-ds283-Documents-Code-SecondaryGWKit/b173fde2-7910-4580-995f-b2ee721fe27e/scratchpad/agent02/outputs-0n5fp_iw/repair/store-shard0001.sqlite)
>> Background models
     The following unvalidated models were detected in the datastore:
       -- "standin-background" (store_id=1, N_tau=4, N_cs_tau=4, N_F=4)
          contains 3 z-sample values | expected=3
```

**A refusal.** The probe's step-4 split, left by a `ReplicationMismatch` with the record set,
reopened. The constructor prints nothing before it raises. What Python prints, with the traceback
frames elided:

```
Traceback (most recent call last):
  ...
Datastore.replication.ReplicatedDivergence: Cannot open sharded datastore "/private/tmp/claude-35086/-Users-ds283-Documents-Code-SecondaryGWKit/b173fde2-7910-4580-995f-b2ee721fe27e/scratchpad/agent02/outputs-0n5fp_iw/refusal/store.sqlite": its replicated tables differ across shards, and the replication_in_flight record (operation="get", class="tolerance", controlling shard=0, store_id=6, started=2026-09-28 22:22:44.766290) does not explain them. 1 difference(s):
  - class "tolerance", shard [0, 1], serial 1: held by shard(s) [2], missing on shard(s) [0, 1]; a serial split: the same row is held under serial 1 on shard(s) [2] and under serial 6 on shard(s) [0, 1]; the record's controlling shard 0 lacks this row, and a shard can only lack what the controller committed
Nothing was repaired: no shard and not the primary was written, and the replication_in_flight record, if any, was left as it was. A store in this state cannot be put right by this tree: it needs a person, or regeneration.
```

## Deviations from the prompt

1. **The licence is narrower than W6's text in five ways** — `IMPLEMENTATION CHOICE`. Each only
   refuses more; none repairs anything W6 would refuse.
   - (a) A copied row with a `timestamp` column must carry the record's `started`. Log 01
     (deviation 6) made `started` the timestamp of every row the write inserts, so that 02 could
     tell the write's own rows from an older difference in the same class.
   - (b) Under a `store_id`, only that serial of the class's own table is copied.
   - (c) A `validate` record licenses no copied row, and a flag is set only after a `get` record.
   - (d) A `BackgroundModel` is copied only as a whole unit: a shard holding part of a model
     refuses.
   - (e) The controller's value serials must be free on the shard they are copied to. A store
     written before prompt 01 drew them per shard.
   `TestRecordDoesNotExplain` pins (a) and (d).
2. **What "recomputed on each shard" does** — `IMPLEMENTATION CHOICE`. A recompute runs only when
   there is a `validated` difference on the record's model under a `validate` record, with the
   controller holding `True`. It then runs on every shard holding the model, the controller
   included.
   - Where the controller holds `False` (`P1`–`C`), nothing is turned on: turning it on would put
     on every shard a flag the controller never committed.
   - A shard holding `True` whose rule gives `False` refuses, and its flag is not turned off.
   - The rule runs inside a savepoint, and any result that does not turn the flag on is rolled
     back to it. So the factory's own `UPDATE … validated = False` never survives, even inside the
     repair's transaction.
3. **The factory's rule is called as it stands, driver-side** — `IMPLEMENTATION CHOICE`, within the
   orchestrator note's rule (no refactor, no re-implementation).
   - The call is `sqla_BackgroundModelFactory.validate(obj, conn=…, table=…, tables=…)`, with the
     shard's SQLAlchemy connection and `build_schema`'s tables.
   - `obj` is a local `_StoredModel(DatastoreObject)` holding the serial and label, which is all
     `validate` reads (`available`, `store_id`, `label`).
   - No actor is needed, so there was nothing to stop and ask about.
4. **The repair is one open transaction per shard, committed only after the second comparison** —
   `IMPLEMENTATION CHOICE`.
   - A refusal rolls every shard back, which makes "nothing was repaired" true of every refusal,
     including one the second comparison makes.
   - The shards' commits are not atomic across files. A crash between two of them leaves the
     record set and some shards repaired, which the next open repairs further: the same licence,
     now with fewer shards lacking.
5. **Engines with SQLite's own transactions** (`_reconcile_engine`) — `IMPLEMENTATION CHOICE`.
   pysqlite's implicit transaction handling does not make `SAVEPOINT` reliable, so the driver's
   `BEGIN` is turned off and one is emitted per transaction. `NullPool`, so each engine's
   `dispose()` closes its file.
6. **`shard_keys` under a record is checked by the second comparison** — `IMPLEMENTATION CHOICE`. A
   copied `wavenumber` can supply a key the first comparison finds missing. A `shard_keys`
   difference the repair does not remove refuses after rollback
   (`test_a_shard_key_the_repair_does_not_supply`).
7. **The primary's journal** — `IMPLEMENTATION CHOICE`. The prompt names the shards. The primary's
   journal is rolled back by `_read_shard_data`'s first read, before this step can open anything,
   so the constructor notes whether one was there. That is one line before `_create_engine`,
   within the constructor. `reconciliation["hot_journals"]` holds file paths, the primary's
   first, and one line is printed per file.
8. **The check runs before `replication_in_flight.create(checkfirst=True)`** — `IMPLEMENTATION
   CHOICE` of order (log 01, deviation 14, left it to 02). A refused store written before prompt 01
   is not given the table, and a missing table reads as no record. A store without the table and
   with no difference still gains it, as before.
9. **Imports at the top of `ShardedPool.py`** — `IMPLEMENTATION CHOICE`. These lines are outside
   the constructor and the new methods, and are listed so the scope check sees them:
   - `_factories`, joined to the existing `from Datastore.SQL.Datastore import …` line;
   - `build_schema`, `DatastoreObject`, and `ReplicatedDivergence` and `same_value` beside the
     existing `ReplicationInFlight`/`ReplicationMismatch` import.

   Local imports inside the methods were the alternative. `store_reader.py` imports `_factories`
   the same way.
10. **How a serial split is named, and what refuses it** — `IMPLEMENTATION CHOICE`.
    - **Naming.** `_describe_differences` pairs rows missing under different serials whose other
      compared values are equal, and names them "a serial split" with both serials. That is
      description only.
    - **Refusing.** A split is refused because it always contains a row that some shard holds and
      the record's controller lacks. Where there is no record, any difference refuses.
    - **A check removed during development.** A draft also refused a copy when the receiving shard
      held a row equal but for its serial. That refused a legitimate case: a second
      `BackgroundModel` whose row equals the first's and differs only in tags and values. It was
      redundant with the rule above. It was removed before the commit, and
      `test_store_of_a_second_background_model_beside_the_first` pins the case.
11. **Refusals the prompt does not list** — `IMPLEMENTATION CHOICE`. Each refuses a state that no
    write path leaves:
    - a `columns` difference;
    - a `duplicate` key;
    - more than one record row;
    - a record naming no operation, replicated class or shard of this store.
12. **The tag tables are found by a rule** — `IMPLEMENTATION CHOICE`. The rule is stated above; on
    this tree it gives exactly `BackgroundModel_tags`, and the clean-store test pins the list.
13. **The printed keys** — `IMPLEMENTATION CHOICE`. One line per class and shard gives the count and
    the keys, abbreviated to the first and last ten beyond 20; `reconciliation` holds every key.
14. **The stand-in additions** — `IMPLEMENTATION CHOICE`. These are `open_pool_output` and
    `store_checksums`. No existing function of `standin_pool.py` changed, so `open_pool_output`
    repeats `open_pool`'s constructor call rather than refactoring it. `store_checksums` hashes the
    directory, not `closed_store_files`, which refuses a store with a journal (test 5 needs one).
15. **Test 5 silences `os.fork`'s `DeprecationWarning`** ("this process is multi-threaded") —
    `IMPLEMENTATION CHOICE`. The child only opens SQLite, writes and calls `os._exit`, as
    `hot_journal_probe.py` does.
16. **More tests than §3's minimum** — `IMPLEMENTATION CHOICE`. There are 38; the reopen table
    shows which state each covers.

No `UNINTENDED DRIFT` was found.
- **`ShardedPool.py`.** The diff is the imports, three constructor hunks, and one contiguous block
  of new methods after `_check_shard_files`. There is no hunk in `_replicated_write`, the three
  `_…_impl_replicated_table` methods, `_assign_shard_keys` or the closed-store block.
- **Other files.** `replication.py` gains `ReplicatedDivergence` and two docstring sentences, and its
  docstring's "Both exceptions pickle" becomes "The exceptions pickle". `test_replicated_write.py` changes in the one named test only.
- **Untouched:** `Datastore.py`, the factories, `main.py`, the `extract_*.py` scripts,
  `store_inventory.py`, `RunRegistry/`, the audit probes and `orchestrator/`.

## Verification performed

- **The starting tree.** `git status` was clean at `65ee1d3`.
- **The new module.** `PYTHONPATH=. ./venv/bin/python -m unittest Datastore.tests.test_reconcile_at_open`
  gave 38 OK, in about 45 s. It was run repeatedly while developing.
- **The six suites**, with README §7's command, serially:

  | Suite | Result |
  |---|---|
  | AdaptiveLevin | 32 OK |
  | ComputeTargets | 552 OK (the known wall-clock flake did not fire) |
  | CosmologyModels | 39 OK |
  | Datastore | **325 OK** (287 + the 38 of `test_reconcile_at_open.py`; the rewritten test is not an addition) |
  | LiouvilleGreen | 148 OK (1 skipped) |
  | RunRegistry | 199 OK |

- **Formatting.** `./venv/bin/python -m black --check .` is clean (334 files).
- **The refusals' reasons.** Every refusal test's message was printed and read, once, by a scratch
  script, to check that each refuses for the reason its test names. That is how deviation 10's
  false refusal was found.
- **The audit probe.** `replicated_write_fault_probe.py` was run once, unmodified, after the change
  (observation 2). It writes only to a temporary directory under its own directory, which it
  deleted.
- **The version window.** The version-row window of observation 1 was demonstrated on stand-in
  shards by a scratch script.
- **Nothing under `var/`** was opened by any test, script or check. `python -m RunRegistry list` was
  not run: it reads `var/runs/`, which the orchestrator note puts out of bounds, and nothing long
  was launched.
- **Scope**, checked with `git diff --cached -U0`: the hunks are exactly those listed at the end of
  the deviations.

## The deliberate-breakage record

Each diff was made by editing the working tree over the staged change and taking `git diff`. It
applies with plain `git apply` from the repository root to this commit. Each was applied, the whole
`Datastore` suite run (325 tests), and `Datastore/SQL/ShardedPool.py` restored with
`git checkout`. None is committed.
- **Counts.** They are unittest's, which counts each failing sub-test separately.
- **The lists.** Every failing test and sub-test is listed.
- **The replay.** Before the commit, the five diffs were extracted from this file, in its final
  form, and replayed with plain `git apply`. Each gave exactly the counts below, and the tree was
  clean after each revert. The replay after the commit, from
  `git show HEAD:prompts/datastore-integrity/logs/02-repair-or-refuse-at-open.md`, is reported to
  the orchestrator; the commit is not amended to record it.

**(i) Repair with no record.** A difference with no record is repaired from the first shard that
holds the row, under a record made up from it, and the clearing accepts that there is no row to
delete.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 979e0f7..bc65850 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -698,7 +698,18 @@ class ShardedPool:
                 conn.rollback()
             differences = self._replicated_differences(specs, snapshots)
 
-            # 3. no record: any difference refuses
+            # 3. no record: a difference is repaired from the first shard holding the row
+            if record is None and len(differences) > 0 and differences[0]["kind"] == "missing":
+                d = differences[0]
+                spec = {s["name"]: s for s in specs}[d["table"]]
+                row = self._read_full_rows(conns[d["holders"][0]], spec, {d["key"]})
+                record = {
+                    "operation": "get",
+                    "class_name": d["table"],
+                    "controller_shard": d["holders"][0],
+                    "store_id": None,
+                    "started": row[d["key"]].get("timestamp"),
+                }
             if record is None:
                 if len(differences) > 0:
                     raise ReplicatedDivergence(
@@ -761,7 +772,7 @@ class ShardedPool:
             # 5. every shard holds what the controller holds: the record is cleared, last
             with self._engine.begin() as conn:
                 result = conn.execute(sqla.delete(self._replication_in_flight_table))
-                if result.rowcount != 1:
+                if result.rowcount > 1:
                     raise RuntimeError(
                         f"ShardedPool: the replication_in_flight record has {result.rowcount} "
                         "rows, expected 1"
```

`failures=4, errors=1`:
- ERROR `test_reconcile_at_open.TestNoRecordRefuses.test_a_store_written_before_the_record_existed`
- FAIL `test_reconcile_at_open.TestNoRecordRefuses.test_a_missing_row`
- FAIL `test_reconcile_at_open.TestNoRecordRefuses.test_a_value_count_difference`
- FAIL `test_reconcile_at_open.TestNoRecordRefuses.test_an_extra_row`
- FAIL `test_reconcile_at_open.TestSerialSplit.test_with_no_record`

`test_with_no_record` still refuses, but with the made-up record, not with none. The error is the
clearing's `DELETE` on a primary without the table.

**(ii) Copy forward from a shard other than the record's controller.**

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 979e0f7..87ea582 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -1157,7 +1157,7 @@ class ShardedPool:
         by_name = {spec["name"]: spec for spec in specs}
         operation = record.get("operation")
         cls_name = record.get("class_name")
-        controller = record.get("controller_shard")
+        controller = min(s for s in snapshots if s != record.get("controller_shard"))
         store_id = record.get("store_id")
         started = record.get("started")
         refusals = []
```

`failures=1, errors=16`:
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_scalar_get` (point='C-P2')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_scalar_get` (point='P2-R1')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_scalar_get` (point='Ri-Ri+1, r1 lacks')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_scalar_get_that_turns_a_flag_on`
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_store_of_a_background_model` (point='C-P2')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_store_of_a_background_model` (point='P2-R1')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_store_of_a_background_model` (point='Ri-Ri+1, r1 lacks')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_store_of_an_exit_time` (point='C-P2')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_store_of_an_exit_time` (point='P2-R1')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_store_of_an_exit_time` (point='Ri-Ri+1, r1 lacks')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_validate_of_a_background_model` (point='C-R1')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_validate_of_a_background_model` (point='Ri-Ri+1, r1 lacks')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_vectorized_get` (point='C-R1')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_vectorized_get` (point='Ri-Ri+1, r1 lacks')
- ERROR `test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_store` (prune=True)
- ERROR `test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_store` (prune=False)
- FAIL `test_reconcile_at_open.TestRecordDoesNotExplain.test_a_row_the_controller_lacks`

Where the substitute shard holds the rows the controller holds ("Ri-Ri+1, r0 lacks"), the copy is
the same and passes. Wherever only the controller holds them, the substitute lacks them and the
open is refused.

**(iii) Compare by key alone, not by serial.** Serial-keyed tables are keyed by every column but
`serial` and `timestamp`, and `serial` is not compared, so a split looks like agreement.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 979e0f7..734c503 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -810,8 +810,8 @@ class ShardedPool:
                 key = ("model_serial", "z_serial")
                 excluded = {"serial", "timestamp"}
             else:
-                key = tuple(c.name for c in table.primary_key.columns)
-                excluded = {"timestamp"}
+                excluded = {"timestamp", "serial"}
+                key = tuple(c.name for c in table.columns if c.name not in excluded)
             compared = [c.name for c in table.columns if c.name not in excluded]
             specs.append(
                 {
```

`failures=16, errors=26`:
- ERROR `test_reconcile_at_open.TestCleanStoreUntouched.test_opening_a_clean_store_writes_nothing`
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_scalar_get` (point='P2-R1')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_scalar_get` (point='Ri-Ri+1, r0 lacks')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_scalar_get` (point='Ri-Ri+1, r1 lacks')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_scalar_get_of_the_shard_key`
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_scalar_get_that_turns_a_flag_on`
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_store_of_a_background_model` (point='P2-R1')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_store_of_a_background_model` (point='Ri-Ri+1, r0 lacks')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_store_of_a_background_model` (point='Ri-Ri+1, r1 lacks')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_store_of_a_second_background_model_beside_the_first`
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_store_of_an_exit_time` (point='P1-C')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_store_of_an_exit_time` (point='C-P2')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_store_of_an_exit_time` (point='P2-R1')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_store_of_an_exit_time` (point='Ri-Ri+1, r0 lacks')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_store_of_an_exit_time` (point='Ri-Ri+1, r1 lacks')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_store_of_an_exit_time` (point='Rn-P3')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_validate_of_a_background_model` (point='C-R1')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_validate_of_a_background_model` (point='Ri-Ri+1, r0 lacks')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_validate_of_a_background_model` (point='Ri-Ri+1, r1 lacks')
- ERROR `test_reconcile_at_open.TestKillAndReopen.test_vectorized_get_that_inserts_and_turns_a_flag_on`
- ERROR `test_reconcile_at_open.TestNoRecordRefuses.test_a_difference_in_a_sharded_table_is_not_looked_at`
- ERROR `test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_store` (prune=True)
- ERROR `test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_store` (prune=False)
- ERROR `test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_validate` (prune=True)
- ERROR `test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_validate` (prune=False)
- ERROR `test_replicated_write.TestWriteOverASetRecord.test_a_reopened_pool_repairs_from_the_controller_and_the_write_succeeds`
- FAIL `test_reconcile_at_open.TestKillAndReopen.test_scalar_get` (point='C-P2')
- FAIL `test_reconcile_at_open.TestKillAndReopen.test_store_of_a_background_model` (point='C-P2')
- FAIL `test_reconcile_at_open.TestKillAndReopen.test_vectorized_get` (point='C-R1')
- FAIL `test_reconcile_at_open.TestKillAndReopen.test_vectorized_get` (point='Ri-Ri+1, r0 lacks')
- FAIL `test_reconcile_at_open.TestKillAndReopen.test_vectorized_get` (point='Ri-Ri+1, r1 lacks')
- FAIL `test_reconcile_at_open.TestMismatchLeftByTheCheck.test_a_store`
- FAIL `test_reconcile_at_open.TestMismatchLeftByTheCheck.test_a_vectorized_get`
- FAIL `test_reconcile_at_open.TestNoRecordRefuses.test_a_flag`
- FAIL `test_reconcile_at_open.TestNoRecordRefuses.test_a_missing_row`
- FAIL `test_reconcile_at_open.TestNoRecordRefuses.test_a_value_count_difference`
- FAIL `test_reconcile_at_open.TestNoRecordRefuses.test_an_extra_row`
- FAIL `test_reconcile_at_open.TestRecordDoesNotExplain.test_a_row_the_controller_lacks`
- FAIL `test_reconcile_at_open.TestRecordDoesNotExplain.test_a_shard_key_the_repair_does_not_supply`
- FAIL `test_reconcile_at_open.TestSerialSplit.test_the_refusal_pickles_with_its_fields`
- FAIL `test_reconcile_at_open.TestSerialSplit.test_with_no_record`
- FAIL `test_reconcile_at_open.TestSerialSplit.test_with_the_record_set`

`TestSerialSplit.test_with_the_record_set` and `test_with_no_record` fail with
`ReplicatedDivergence not raised`: the split opens.

**(iv) Copy `validated` instead of recomputing it.** The controller's `True` is written on each
shard in place of the factory's rule.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 979e0f7..8f36a4e 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -1497,11 +1497,10 @@ class ShardedPool:
             label = row[columns.index("label")] if "label" in columns else None
 
             savepoint = conns[sid].begin_nested()
-            result = factory.validate(
-                _StoredModel(serial, label),
-                conn=conns[sid],
-                table=table,
-                tables=built.tables,
+            # the controller's flag, copied
+            result = True
+            conns[sid].execute(
+                sqla.update(table).where(table.c.serial == serial).values(validated=True)
             )
             if result is True and not stored:
                 savepoint.commit()
```

`failures=1`:
- FAIL `test_reconcile_at_open.TestRecordDoesNotExplain.test_a_recomputed_validated_that_disagrees`

The failure is `ReplicatedDivergence not raised`. The controller's `True` over value rows that are
short on every shard is copied to every replica and agrees.

**(v) Clear the record before the second comparison.** It is cleared as soon as the repair is
written, and the final clearing accepts that it is gone.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 979e0f7..aac4d6c 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -737,6 +737,10 @@ class ShardedPool:
                             self._primary_file, refusals, record, after_repair=True
                         )
 
+                    # the record cleared first
+                    with self._engine.begin() as conn:
+                        conn.execute(sqla.delete(self._replication_in_flight_table))
+
                     # the second comparison, of what the repair's transactions hold
                     after = {
                         sid: self._read_replicated_tables(conns[sid], specs)
@@ -761,7 +765,7 @@ class ShardedPool:
             # 5. every shard holds what the controller holds: the record is cleared, last
             with self._engine.begin() as conn:
                 result = conn.execute(sqla.delete(self._replication_in_flight_table))
-                if result.rowcount != 1:
+                if result.rowcount > 1:
                     raise RuntimeError(
                         f"ShardedPool: the replication_in_flight record has {result.rowcount} "
                         "rows, expected 1"
```

`failures=1`:
- FAIL `test_reconcile_at_open.TestRecordDoesNotExplain.test_a_shard_key_the_repair_does_not_supply`

The second comparison refuses and rolls the shards back. The record is already gone: the primary's
checksum changed, and the store would open next time with nothing saying which shard was right.

## Observations not acted on

1. **The `version` row is replicated outside the in-flight record, and a crash there now refuses the
   next open.** Opened as `[02-the-version-row-is-written-outside-the-in-flight-record]` (board
   §3.3).
   - **The window.** `ShardedPool.__init__` builds the first actor, whose `Datastore.__init__`
     inserts the version label's row. Each later actor then inserts it under that serial
     (`version_serial`), each in its own constructor. Nothing records this, since it does not go
     through `_replicated_write`.
   - **Demonstrated** on stand-in shards (a scratch script): a store opened under label `L1`, then
     opened under `L2` with every actor after the first killed. The second open under `L2` is
     refused: `class "version", shard [0, 1], serial 2: held by shard(s) [2], missing on shard(s)
     [0, 1]`.
   - **At `HEAD`** the same second open would have healed: the first actor finds `L2`, and the
     others insert it under its serial. That is what the prompt's "no record, any difference,
     refuses" forbids here.
   - **How rarely.** Only an open under a label the store has never seen writes these rows.
     `VERSION_LABEL` is the constant `"2025.1.1"` in `main.py` and the six extraction scripts, so
     the window is a new store's first open, or the first open after the label changes.
   - **Not fixed.** The actors' constructors are `Datastore.py`, outside this prompt.
2. **The audit's fault probe now stops at its step 5**, with `ReplicationInFlight`.
   - **Step 3.** Its reopen now repairs the row step 2's interrupted get left missing, so step 4
     finds every shard holding it and no longer makes a split.
   - **Step 5.** It kills a replica, then writes again through the same pool without reopening.
     Since prompt 01 that refuses over the set record.
   - **Status.** The probe is the audit's measurement record, and is not edited. Not an issue.
3. **The check refuses two things the audit's census could not see.** Nothing under `var/` was
   opened, so this is untested on the real stores.
   - **Duplicates.** The census compared rows as sets, so it would not notice two
     `BackgroundModelValue` rows of one model at one redshift on a shard. The check refuses that.
   - **Shard keys.** The census did not read the primary's `shard_keys`. The check refuses a
     `key_serial` that is not a `wavenumber` on every shard.
   - **The consequence.** `main.py` resuming the live A3 store, or prompt 10's first real run,
     would refuse on either before any actor starts.
4. **A refusal leaves the pool's own primary engine undisposed**, as every existing refusal in the
   constructor does (`_check_shard_files`, `_read_shard_data`). The check's own shard engines are
   disposed in a `finally`. Not an issue.
5. **Columns a shard's table has that `build_schema` does not declare are not compared.** The
   comparison reads through the built tables, as the prompt asks. No such column exists on this
   tree.
6. **Nothing here has run under Ray or on a real store.** Prompt 10 is the first real run.

## State handed to the next prompt

- **Names.**
  - `ShardedPool._reconcile_replicated_tables(primary_journal)`, run by the constructor;
  - `ShardedPool.reconciliation`: `None` for a new store; otherwise a dict with `compared`
    (tables), `shards`, `hot_journals` (paths), `record` (the in-flight row or `None`),
    `controller`, `repaired` (actions: `action`, `class_name`, `shard`, `from`, `keys`, and
    `columns` or `validated`) and `cleared`;
  - `Datastore.replication.ReplicatedDivergence(primary, differences, record, after_repair)`.
- **For prompt 03.** Its lookups change on the classes compared here. A store the tree opens now
  has identical replicated tables on every shard, so a lookup's answer no longer depends on which
  shard serves it. A duplicate within one shard (what `.scalar()` hides) is not a cross-shard
  difference: the check sees it only for `BackgroundModelValue`'s key. For every other class,
  `one_or_none()` is still the guard.
- **For prompt 10.** The check will run on the first real open of every store, with the two
  refusals of observation 3 untested on real data.
- **Stand-in pool.** It gains `StandinCluster.open_pool_output` and `store_checksums`.
- **Baselines** after this commit: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39,
  Datastore **325**, LiouvilleGreen 148 (1 skipped), RunRegistry 199.
