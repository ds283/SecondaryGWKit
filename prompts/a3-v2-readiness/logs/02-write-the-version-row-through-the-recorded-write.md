# Log — prompt 02: write the version row, and the prune at open, through the recorded write

**Subject:** *Write the version row and the replicated prune through a record* · **Model:** Claude
Opus 5.5 · **Date:** 2026-10-03 · **Base:** `56d1988`, clean · **Result:** landed. The version row is
written once, by the pool, through `_replicated_write`, and only when no shard holds the label; the
prune at open of a replicated class is the pool's, under a `prune` record that the check at open
completes. `[02-the-version-row-is-written-outside-the-in-flight-record]` (on `datastore-integrity`)
and `[00-the-prune-at-open-deletes-a-replicated-row-outside-the-in-flight-record]` (on this board)
are closed. No stop condition of the prompt's §5 fired. The commit is the one that adds this file
("this commit" below, since its SHA does not exist yet).

The orchestrator's facts for this prompt (its note, checked at `1c458d3`) are cited below as
"the orchestrator's", where this prompt relies on them without re-deriving them.

## What shipped

- **`Datastore/SQL/Datastore.py`** (60 added, 19 deleted):
  - `Datastore.__init__` neither looks up nor inserts the version row. It drops `version_serial`.
    It keeps `version_label`, to name the label in the refusal below (*prompt's choice* 1). It
    gains a required `replicated_tables` argument (*prompt's choice* 5).
  - `set_version(serial)`: sets the serial. The same serial again is harmless; a different one,
    or a non-`int`, raises.
  - `_insert` raises `RuntimeError`, naming the class and `set_version`, on any insert into a
    class whose schema has `use_version`, until `set_version` has been called. The check comes
    before the `SerialLeaseManager` `with`, so a refused insert takes no serial (the
    orchestrator's: the lease is taken inside that `with`).
  - `_validate_on_startup` passes `prune=False` to the factory of every class in
    `replicated_tables`, whatever `prune_unvalidated` is. Sharded classes are pruned as before.
  - The `IntegrityError` import, used only by the removed serial check, is gone.
- **`Datastore/SQL/ShardedPool.py`** (514 added, 78 deleted):
  - `__init__`: every actor is submitted at once, in the order the pool always built them (the
    last shard id first), with no version work. After `read_largest_store_ids`, the broker's
    notification and the `read_table_config` check, `_find_version_row` reads the label's row;
    only if no shard holds it is it written by
    `ray.get(self._get_impl_replicated_table("version", {"label": version_label}))`. The row is
    kept as `self._version`, and `set_version` is called on every actor with one `ray.get`.
  - `_find_version_row`: `SELECT serial, label FROM version WHERE label = ?` on every shard file,
    opened `mode=ro` with `sqlite3`. A label held twice on a shard, or shards that disagree, raise.
    Returns a `store_tag`, as the factory does, or None.
  - Under `prune_unvalidated`, after the check at open and `replication_in_flight.create`, and
    before the broker, `_prune_replicated_tables` prunes each class `_replicated_prune_classes`
    returns. That set is read from the factories' registrations (`validate_on_startup` true)
    intersected with the pool's `replicated_tables`; today it is `["BackgroundModel"]`.
    - `_prune_needed`: reads each shard with the factory's `validate_on_startup(prune=False)`.
      If no shard reports anything, there is no record and nothing is written.
    - `_commit_prune_record`: `operation="prune"`, the class, `controller_shard` and `store_id`
      NULL, in its own commit, before the first shard. On a table whose `controller_shard` is NOT
      NULL the insert fails, and it raises `RuntimeError` before any shard is touched.
    - `_prune_shard_and_commit` → `_prune_shard`, each shard in ascending id in its own
      transaction: the factory's `validate_on_startup(prune=True)`; then the factory again with
      `prune=False`, which must report nothing; then every compared table read before and after,
      which must have lost rows only, and only in the class's own table and the compared tables
      with a foreign key to it (`_prune_unit`). Otherwise that shard rolls back and the open raises
      with the record set (*prompt's choice* 6).
    - `_clear_in_flight_record`, in its own commit, last. `self.pruned` lists what was deleted.
  - `_reconcile_replicated_tables`: a record whose operation is `prune` goes to
    `_complete_interrupted_prune` and nowhere else. That function refuses a class not in
    `_replicated_prune_classes`. Otherwise it opens a transaction on every shard, runs
    `_prune_shard` on each, compares every replicated table inside those transactions, and on any
    difference or failed prune rolls every shard back and raises (`ReplicatedDivergence` with
    `after_repair=True`, or the `RuntimeError`). Only then does it commit every shard and clear the
    record, last. This is done whatever the open's `prune_unvalidated` is.
  - `replication_in_flight.controller_shard` is now `nullable=True` (the record's schema, below).
  - The comment above `_reconcile_replicated_tables`, its docstring, `_plan_replicated_repair`'s
    docstring, the comment above `_replicated_write` and the comment on the record's table are
    made true. The first now names the recorded prune as the one exception to "Nothing in the
    write path deletes a replicated row", and says the version row is written only through
    `_replicated_write`. Its numbered steps, and the step comments inside the function, are
    renumbered for the new step 2.
  - `_plan_replicated_repair`'s code is unchanged. Its licence already refused any operation but
    `get`, `store` and `validate`; it admits the version row as it stands (W2, below).
- **`Datastore/tests/standin_pool.py`** (40 added, 0 deleted): additions only, below.
- **`Datastore/tests/test_version_row_at_open.py`** (new, 12 tests) and
  **`Datastore/tests/test_prune_at_open.py`** (new, 11 tests), below.
- **`Datastore/tests/test_reconcile_at_open.py`** (4 added, 2 deleted): `TestPruningAfterRepair`'s
  class docstring only. It said the prune runs "inside each actor"; it now says the replicated
  prune runs in the pool, under a record, before any actor exists, and the sharded one in each
  actor. No assertion and no other line changed.
- `Datastore/SQL/ObjectFactories/version.py` and `BackgroundModel.py` are not touched.
  `sqla_version_factory.build`'s return type is unchanged.

### The order of events in `ShardedPool.__init__`

**Before (`56d1988`), a new store:**
1. The primary's absence, and every shard file's, checked; the directory made.
2. `_create_engine`; `_write_shard_data` (the primary's six tables, then their rows).
3. The broker.
4. The first actor, for the last shard id. Its constructor creates its file and every table, then
   inserts the version row (`object_get("version", label=…)`, serial from its engine).
5. `ray.get(shard0.object_get.remote("version", label=…))` for the serial.
6. Every other actor, each given `version_serial`. Each constructor creates its file and tables,
   and inserts the row under that serial.
7. `read_largest_store_ids`; the broker's notification.
8. The `read_table_config` check.

**Before (`56d1988`), an existing store:**
1. The primary's journal noted.
2. `_create_engine`; `_read_shard_data`.
3. `_check_shard_files`.
4. `_reconcile_replicated_tables`.
5. `replication_in_flight.create(checkfirst=True)`.
6. The broker.
7. The first actor: `_drop_actions`, `_ensure_tables`, `_validate_on_startup`, which under
   `prune_unvalidated` prunes every class including `BackgroundModel`; then the version get, which
   inserts under a label the store has not seen.
8. The pool's `ray.get` of that actor's version get.
9. Every other actor, the same, with `version_serial`.
10. `read_largest_store_ids`; the broker's notification.
11. The `read_table_config` check.

**After, a new store:**
1. The absence checks; the directory made.
2. `_create_engine`; `_write_shard_data`.
3. The broker.
4. Every actor, submitted at once, the last shard id first. Each creates its file and every table.
   None writes the version row.
5. `read_largest_store_ids`; the broker's notification.
6. The `read_table_config` check.
7. `_find_version_row`: a `mode=ro` select on every shard file. The label is absent.
8. The row is written once through `_get_impl_replicated_table("version", …)`, so through
   `_replicated_write`. The record is committed, the controller inserts, `store_id` is filled in,
   the replicas insert under that serial, their answers are checked, and the record is cleared.
9. `set_version` on every actor.

**After, an existing store:**
1. The primary's journal noted.
2. `_create_engine`; `_read_shard_data`.
3. `_check_shard_files`.
4. `_reconcile_replicated_tables`. Hot journals are rolled back. A `prune` record is completed:
   re-run on every shard, compared, committed, cleared. Any other record goes to the
   compare/repair/refuse path as before.
5. `replication_in_flight.create(checkfirst=True)`.
6. Under `prune_unvalidated` only, `_prune_replicated_tables`. For each replicated class whose
   factory prunes at startup: read; if anything is to be pruned, the record, then each shard
   (pruned, checked, committed), then the record cleared.
7. The broker.
8. Every actor, submitted at once: `_drop_actions`, `_ensure_tables`, `_validate_on_startup`, which
   prunes sharded classes only and reports replicated ones.
9. `read_largest_store_ids`; the broker's notification.
10. The `read_table_config` check.
11. `_find_version_row`. If the label is found, nothing is written. If it is absent, the row is
    written through `_replicated_write`, as in the new store's step 8.
12. `set_version` on every actor.

### The record's schema

**Before:**

| Column | Type | Null |
|---|---|---|
| `operation` | `String(DEFAULT_STRING_LENGTH)` | NOT NULL |
| `class_name` | `String(DEFAULT_STRING_LENGTH)` | NOT NULL |
| `controller_shard` | `Integer` | **NOT NULL** |
| `store_id` | `Integer` | NULL |
| `started` | `DateTime` | NOT NULL |

**After:** the same, except that `controller_shard` is **NULL**able. A `get`/`store`/`validate`
record fills it, exactly as before. A `prune` record leaves it, and `store_id`, NULL.

**An existing store's table** keeps `controller_shard NOT NULL`, because
`create(checkfirst=True)` does not alter a table (the orchestrator's). No store is migrated
(README §5). On such a store:
- every open that does not prune a replicated class behaves as before;
- a pruning open with a replicated row to prune raises in `_commit_prune_record`, before any shard
  is touched, with nothing written: "cannot record the prune … Nothing was pruned and no shard was
  touched. Open it with prune_unvalidated=False, or at the tree that wrote it";
- with `prune_unvalidated=False` it opens and keeps the row. Pinned by
  `test_prune_at_open.TestRecordTableBeforePrune`, which compares every file's SHA-256 before and
  after the refused open.

### The prompt's choices

1. **The actor keeps `version_label`, and drops `version_serial`.** The label names the store's
   label in `_insert`'s refusal and in `set_version`'s, and costs nothing. Nothing else on the
   actor reads it.
2. **How the pool reads the label's row:** a plain `SELECT serial, label FROM version WHERE label =
   ?` on every shard file, opened `mode=ro`. That is the read audit V2 measured, and the one 03
   will make. The prompt allows one shard. Every shard is read so that a disagreement raises,
   instead of being settled by whichever shard happened to be read. A new store's shards have had
   no check at open, and the read costs one select per shard.
3. **Where the version step sits:** last, after `read_largest_store_ids`, the broker's
   notification and the `read_table_config` check. Two reasons.
   - The version write is then made with the pool in the state every later replicated get finds
     it in, the broker told the largest serials.
   - An open refused by the `read_table_config` check, which raises, writes no version row.
   
   `version` takes no lease (`_insert`), so the broker's order does not matter to the write itself.
4. **A prune that finds nothing to delete on any shard skips the record.** Otherwise every pruning
   open would write the primary, the defect W1's read-first design avoids for the version row.
   `test_prune_at_open.TestUninterruptedPrune.test_a_prune_with_nothing_to_prune_writes_nothing`
   pins it by SHA-256.
5. **How the actor knows which classes are replicated:** a required `replicated_tables` constructor
   argument, which the pool fills from its own `self._replicated_tables`. That is the list the
   check at open compares and the pool's prune derives from. An actor that imported
   `config.sharding` would disagree with a pool given other lists (the orchestrator's). The
   argument has no default, so an actor cannot be built without saying.
6. **How the pool knows a shard's prune worked** (the factory catches its own `SQLAlchemyError`
   and returns normally; the orchestrator's). Inside the shard's transaction, before it commits:
   - the factory is asked again with `prune=False` and must report nothing. That is the
     re-selection the prompt suggests, made by the factory's own rule, not a copy of it;
   - every compared table, read before and after, must have lost rows only, and only in the
     class's own table and those with a foreign key to it. So the prune is shown to delete no row
     of another class.

### W2, the licence and the version row

`_plan_replicated_repair`'s licence admits the version row unchanged. No change was needed, so none
was made. As the orchestrator's note says, `version` is the only replicated class with no
`timestamp` column, so the "written by the recorded write" timestamp test does not apply to it. A
fault after the controller commits and before `_replicated_write`'s step 3 leaves `store_id` null.
The licence then copies any `version` row the controller holds and another shard lacks. After this
prompt nothing else writes `version`, so a second, unrecorded difference in `version` cannot sit
beside a recorded one. A store damaged by the old defect has no record, and is refused before it
can reach a recorded write (`TestVersionDifferenceWithNoRecord`).

The stand-in's `"after"` fault on the controller raises at step 2's `ray.get`, before step 3, so
`TestNewLabelInterrupted.test_a_fault_after_the_controller_commits` goes through the
null-`store_id` path. The test asserts the record `("get", "version", controller, None)`. The
one-replica case goes through the `store_id` path: `("get", "version", controller, 2)`.

### The stand-in's additions (`standin_pool.py`, additions only)

1. `StandinCluster.__init__`: two attributes, `open_controller = None` and `_writing_pool = None`.
2. `StandinCluster.pin_controller(shard_id)`, a context manager. Inside it,
   `ShardedPool._replicated_write` is wrapped (`mock.patch.object`) to note the pool making the
   write, and the controlling shard of every replicated write is `shard_id`, resolved against
   that pool.
   - The reason: `_StandinRandom` resolves `controller` against `cluster.pool`, which
     `open_pool` sets only after `ShardedPool(...)` returns. So the version write W1 makes inside
     `__init__` could not be pinned (the orchestrator's).
3. `_StandinRandom.randrange`: three lines at the top, which apply only when `open_controller` is
   set and a write is running inside `pin_controller`.

No existing caller sets `open_controller` or enters `pin_controller`. `controller`'s behaviour,
and every existing method, are unchanged. `git diff` of the file shows no deleted line.

### The tests

`test_version_row_at_open.py`, 12 tests (the prompt's §3 cases 1–4, 6, and W2):
- `TestNewStore`:
  - `test_the_row_is_written_once_through_the_recorded_write`: a hook sees three version gets,
    one per shard, each made while the primary held `("get", "version", controller)`. Every shard
    holds `[(1, 'L1')]`, no record is left, and `set_version` was called once on each of 0, 1, 2;
  - `test_no_actor_constructor_inserts_the_row`: the pool's version get is faulted `"before"` on
    every shard. All three files exist, every `version` table is empty, and the get's record is
    left. This is shown from the shard files, because a hook sees method calls, not constructors
    (the orchestrator's);
  - `test_versioned_rows_carry_the_serial_on_every_shard`: `wavenumber_exit_time`,
    `GkSourcePolicy` and `BackgroundModel` on every shard, and the one sharded
    `GkSourcePolicyData` row, carry serial 1. After a reopen under `L2`, a new `GkSourcePolicy`
    carries 2 on every shard.
- `TestNewLabelInterrupted` (audit V1 case 3): a fault after the controller commits, before it
  commits, and on one replica only, each followed by a clean reopen. Each case asserts the record,
  the copies made, every shard `[(1, 'L1'), (2, 'L2')]`, and that a second reopen leaves every
  file's SHA-256 unchanged. A fourth test opens a store whose shards all lack `L2`: it writes the
  row through three version gets, and the reopen under `L1` makes no version get and changes no
  file.
- `TestNewStoreInterrupted.test_interrupted_at_the_version_write`: sub-tests `before` and `after`
  on a new store's first open, after all three files exist. The next open succeeds, clears the
  record, and every shard holds `[(1, 'L1')]`.
- `TestInsertBeforeSetVersion`, on an actor built directly with no `set_version`:
  - an insert of `GkSourcePolicy` (versioned) raises `RuntimeError` naming `"GkSourcePolicy"` and
    `set_version`. Every file's SHA-256 is unchanged, and the only stand-in call made is the
    actor's own `object_get`: the broker was never called;
  - an unversioned `tolerance` insert is not refused. After `set_version(1)` the policy is
    inserted with `version = 1`. `set_version(1)` again passes; `set_version(2)` and
    `set_version("1")` raise.
- `TestVersionDifferenceWithNoRecord`: a hand-inserted `version` row on one shard, with no record,
  is refused naming `class "version"`, and no file changes.
- `TestTheLicenceAndTheVersionRow`: the state an interrupted version get leaves, planned by
  `_plan_replicated_repair`.
  - Under its own `get` record the plan copies `version` to both replicas, with no refusal.
  - Under the same record as a `prune`, naming the controller and naming none, it is refused
    ("does not name an operation…").

`test_prune_at_open.py`, 11 tests (the prompt's §3 cases 5–7). Each store holds a validated model A,
an unvalidated model B (its own tag, values scaled), an exit time and a sharded
`GkSourcePolicyData`:
- `TestUninterruptedPrune`:
  - `test_every_shard_identical_and_no_record_left`: a wrapper on `_prune_shard_and_commit` sees
    `[("prune", "BackgroundModel", None, None)]` on the primary before each shard. Afterwards there
    is no record. B is gone as a unit from every shard. A's rows are unchanged on every shard.
    Every sharded table, and every other replicated table, is unchanged row for row. `pool.pruned`
    names the three unit tables on each shard;
  - `test_a_prune_with_nothing_to_prune_writes_nothing`: SHA-256 unchanged;
  - `test_the_set_of_classes_is_read_from_the_factories`: `["BackgroundModel"]` for the production
    list. A list that also named `GkSource` would include it.
- `TestInterruptedPrune`:
  - `test_faulted_after_each_shard`, sub-tests k = 0, 1, 2: the record is left, and shards up to k
    are pruned and the rest not. The next open, with `prune_unvalidated=False`, completes the
    prune on exactly the shards not reached, with controller None, and clears the record. B is
    gone and A is unchanged everywhere. A second reopen changes no file;
  - `test_faulted_before_the_first_shard`: the record is left and no shard file changed. The next
    open re-runs the prune on all three and clears the record.
- `TestPruneFailsInTheFactory.test_the_record_stays_and_the_next_open_completes_it`: a SQLite
  trigger makes the `BackgroundModel` delete raise on the middle shard. The factory swallows it.
  - The open raises "did not take … shard 1". The record is left. Shard 0 is pruned. Shard 1 still
    holds all of B's rows: its tags and values had been deleted before the model's delete failed,
    and were rolled back. Shard 2 was not reached.
  - A completion with the trigger still there raises and changes no file.
  - With the trigger dropped, the next open completes the prune on shards 1 and 2 and clears the
    record.
- `TestRecordsStayApart`:
  - a `prune` record over a deleted `tolerance` row on one shard refuses, with `after_repair` true
    and only `tolerance` named, every file unchanged and the record kept;
  - a `prune` record naming `tolerance`, which does not prune at startup, refuses with nothing
    written;
  - an interrupted store of B whose replica never ran, with its record then read as `get`:
    repaired as today, by three copies to that replica, and B is kept, unvalidated, on every
    shard.
- `TestNoActorDeletesAReplicatedRow.test_under_prune_unvalidated`: the pool's prune is stood down
  (`mock.patch`), so the actors start with B present under `prune_unvalidated=True`. An engine
  listener on each actor's `_validate_on_startup` sees no `DELETE` on a replicated table. The
  factories were asked `("BackgroundModel", False)` and `("TkNumericIntegration", True)` on each of
  the three shards. B is still on every shard.
- `TestRecordTableBeforePrune`: above, under *The record's schema*.

Every test runs on stand-in shards in a temporary directory, with no Ray. Each module's
`tearDownModule` asserts `ray.is_initialized()` is false.

## Deviations from the prompt

1. **The check at open completes a prune inside open transactions, all or nothing.** The prompt's
   four steps are kept: re-run, compare, refuse on a difference, clear last. Every shard's re-run is
   held open until the comparison passes, and is then committed. The open's own prune, by
   contrast, commits each shard in its own transaction, as the prompt requires. The reason:
   `ReplicatedDivergence`'s message says "no shard and not the primary was written", and
   `Datastore/replication.py` is outside this prompt's files. A refusal after committed re-runs
   would make it false. The existing repair holds its writes open in the same way. **IMPLEMENTATION
   CHOICE.**
2. **The pool's prune checks more than the re-selection the prompt names.** It also checks that only
   the class's own rows, and its dependents', were deleted (*prompt's choice* 6). It reads every
   compared table twice per pruned shard, and only on an open that has something to prune.
   **IMPLEMENTATION CHOICE.**
3. **A `prune` record naming a class that is not a replicated class pruned at startup is refused**,
   with nothing written. The prompt does not say what such a record does. **IMPLEMENTATION CHOICE.**
4. **`set_version` refuses a second, different serial and a non-`int`.** The prompt asks only for
   the method. A second serial would leave rows already inserted under the first. **IMPLEMENTATION
   CHOICE.**
5. **The unused `IntegrityError` import is removed from `Datastore.py`.** It was used only by the
   constructor's serial check, which W1 removes. **STRUCTURALLY REQUIRED.**
6. **The drop actions now run after the prune of replicated classes.** They used to run before it,
   inside each actor. The prune must run before any actor exists (W3), and the drops stay in the
   actor (§6: `_drop_actions` is not this prompt's). No drop action names a table of a replicated
   class that prunes at startup (`_drop_actions`: `BackgroundModel`, its tags, values and
   `IntegrationSolver` appear in none), so the two do not interact. **STRUCTURALLY REQUIRED.**
7. **Breakage (vi) is caught by a direct test of `_plan_replicated_repair`**, not through an open.
   The check at open sends a `prune` record to `_complete_interrupted_prune` before the licence is
   reached, so a licence that accepted a `prune` record could not be seen through an open. The
   licence is tested as a function. **STRUCTURALLY REQUIRED.**
8. **The `get` record of `BackgroundModel` in §3 case 6 is made by editing a real record.** An
   interrupted `object_store` of B leaves a `store` record, whose `operation` is then set to `get`.
   No replicated `get` of `BackgroundModel` inserts a model, so no write leaves that record as it
   stands. The licence copies under `get` and `store` alike, and the test shows the record repaired
   and B kept. **IMPLEMENTATION CHOICE.**
9. **Two test modules, not one** (`test_version_row_at_open.py`, `test_prune_at_open.py`). The prompt
   allows "a test module (or modules)". Not a deviation; noted for the file list.

No `UNINTENDED DRIFT`.

## Verification performed

### The audit's probe

`PYTHONPATH=. ./venv/bin/python docs/a3-v2-readiness/version_window_probe.py`, with its output written
to this prompt's scratch directory. The probe and its committed `.txt` are not changed. It exits 0
and ends `ray.is_initialized() = False`. Each case's own lines, quoted exactly:

- **Case 1** (`start`): `  open under 'L1' returned (no crash happened)`, then
  `  next open under 'L1': opened; version serial 1`. As the orchestrator's note predicts, `start`
  fires only when the constructor is given `version_serial`, which W1 removes, so nothing is killed.
  Every shard holds `[(1, 'L1')]`.
- **Case 2** (`version`, new store): `  open under 'L1': crashed: constructor of shard0002-store killed
  before inserting the version row`, after which `store-shard0001.sqlite: version rows [(1, 'L1')]`,
  and shards 0 and 2 `[]`; then `  next open under 'L1': opened; version serial 1`, with every shard
  `[(1, 'L1')]`. **Healed.** The probe's "constructor of" wording is its own: what it killed is the
  pool's replica call, `object_get("version", serial=…)`.
- **Case 3** (`version`, `L1` store opened under `L2`): `  open under 'L2': crashed: constructor of
  shard0002-store killed before inserting the version row`, leaving shard 0 `[(1, 'L1'), (2, 'L2')]`
  and shards 1 and 2 `[(1, 'L1')]`; then `  next open under 'L2': opened; version serial 2`, every shard
  `[(1, 'L1'), (2, 'L2')]`. **Healed** (it was `ReplicatedDivergence` at `56d1988`).
- **Case 4** (`start`, the prune): `  open returned (no crash happened)`, every shard then
  `BackgroundModel (serial, validated) []`; `  next open with prune_unvalidated=True: opened`. Its
  `start` kill fires on `version_serial` too, so nothing was killed. **This case is no evidence about
  W3** (the orchestrator's). §3 case 5's tests are the measurement.

### Deliberate breakage

Below.

### Suites

All six, serially, with README §7's command, each suite's output to a file in the scratch
directory and its verdict grepped from it, at the tree of this commit:

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 669 OK |
| CosmologyModels | 39 OK |
| Datastore | **574 OK** |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

Measured 00:35–00:46 on 2026-10-03, at load averages of 10–22 (1-minute) on a machine shared with an
unrelated Ray job. Datastore is the brief's 551 plus this prompt's 23 (12 + 11). Every other count is
unchanged from the baselines at dispatch. No test failed, so no wall-clock flake needed re-running.

### Formatting and staging

- `black` on the six touched Python files, then `black --check` on them: clean.
- Staged by explicit path. `git diff --cached --name-only` was shown before the commit.

## The deliberate-breakage record

Each diff was made against the staged tree of this commit and recorded with `git diff`. Each was run
against both new modules (and, for (vii), `test_reconcile_at_open.TestCleanStoreUntouched`), and then
restored with `git checkout`. Before finishing, each was re-checked with `git apply --check` against
the committed tree.

**(i) The actor's constructor inserts the version row again.** 10 failures, in 9 of the 12 tests
of `test_version_row_at_open`. Among them:
- `TestNewStore.test_the_row_is_written_once_through_the_recorded_write`: the pool finds the row its
  actors inserted, so no version get is made;
- `TestNewStore.test_no_actor_constructor_inserts_the_row`: for the same reason the faulted get never
  runs, and the open is not killed;
- `TestInsertBeforeSetVersion.test_a_versioned_insert_raises_names_the_class_and_writes_nothing`:
  the constructor has set the serial.

```diff
diff --git a/Datastore/SQL/Datastore.py b/Datastore/SQL/Datastore.py
index 1604782..c981d1f 100644
--- a/Datastore/SQL/Datastore.py
+++ b/Datastore/SQL/Datastore.py
@@ -252,6 +252,8 @@ class Datastore:
 
         self._read_table_config: Optional[ReadTableConfigType] = read_table_config
 
+        self._version_serial = self.object_get("version", label=version_label).store_id
+
     def set_version(self, serial: int):
         """
         Set the serial of the version row, which ShardedPool has found or written on every shard
```

**(ii) `set_version` is not called on one actor.** Fails
`TestNewStore.test_the_row_is_written_once_through_the_recorded_write` (`set_version` on two shards
of three). Errors in `TestNewStore.test_versioned_rows_carry_the_serial_on_every_shard` and in every
`test_prune_at_open` test but one, because their first versioned replicated write is refused on the
actor that was skipped: 1 failure, 13 errors of 23.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index c1d30c4..dc5af32 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -301,7 +301,7 @@ class ShardedPool:
         ray.get(
             [
                 shard.set_version.remote(self._version.store_id)
-                for shard in self._shards.values()
+                for shard in list(self._shards.values())[1:]
             ]
         )
 
```

**(iii) The pool prunes without committing the `prune` record.** The diff also removes the record's
clearing, which would otherwise fail on the missing row. Fails
`TestUninterruptedPrune.test_every_shard_identical_and_no_record_left` (no record seen while
pruning), `TestInterruptedPrune.test_faulted_after_each_shard` (all three sub-tests: no record left
to complete), `test_faulted_before_the_first_shard`,
`TestPruneFailsInTheFactory.test_the_record_stays_and_the_next_open_completes_it` and
`TestRecordTableBeforePrune.test_a_pruning_open_refuses_before_any_shard_is_touched`. It errors in
`TestRecordsStayApart.test_a_prune_record_over_a_difference_in_another_class_refuses`: 7 failures,
1 error.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index c1d30c4..ee4cbc7 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -1892,7 +1892,6 @@ class ShardedPool:
                 continue
 
             # 2. the record, before the first shard is touched
-            self._commit_prune_record(cls_name)
 
             # 3. each shard in its own transaction
             actions = []
@@ -1902,7 +1901,6 @@ class ShardedPool:
                 )
 
             # 4. the record, cleared last
-            self._clear_in_flight_record("prune", cls_name)
             print(
                 f'>> Pruned the unvalidated rows of "{cls_name}" from every shard of '
                 f'"{str(self._primary_file)}" under a "prune" record, which was cleared'
```

**(iv) The check at open clears a `prune` record without re-running the prune.** Fails
`TestInterruptedPrune.test_faulted_after_each_shard` (`after=0` and `after=1`: refused, with the
unpruned shards different) and `test_faulted_before_the_first_shard`. Also
`TestPruneFailsInTheFactory.test_the_record_stays_and_the_next_open_completes_it` and
`TestRecordsStayApart.test_a_prune_record_over_a_difference_in_another_class_refuses`: 2 failures,
3 errors. `after=2` passes under it: every shard was already pruned, so there is nothing left to
re-run.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index c1d30c4..60f974a 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -1951,7 +1951,7 @@ class ShardedPool:
             try:
                 # 1. the prune, again, on every shard
                 actions = []
-                for sid in shard_ids:
+                for sid in []:
                     done, _ = self._prune_shard(cls_name, sid, conns[sid], built, specs)
                     actions.extend(done)
 
```

**(v) The actor prunes `BackgroundModel` again under `prune_unvalidated=True`.** Fails
`TestNoActorDeletesAReplicatedRow.test_under_prune_unvalidated`.

```diff
diff --git a/Datastore/SQL/Datastore.py b/Datastore/SQL/Datastore.py
index 1604782..7d31b85 100644
--- a/Datastore/SQL/Datastore.py
+++ b/Datastore/SQL/Datastore.py
@@ -409,9 +409,7 @@ class Datastore:
 
                 tab = record["table"]
 
-                prune = (
-                    self._prune_unvalidated and cls_name not in self._replicated_tables
-                )
+                prune = self._prune_unvalidated
 
                 with self._engine.begin() as conn:
                     msgs = factory.validate_on_startup(conn, tab, self._tables, prune)
```

**(vi) `_plan_replicated_repair` accepts a `prune` record as a `get`.** Fails
`TestTheLicenceAndTheVersionRow.test_a_prune_record_is_refused_where_a_get_record_is_admitted`: the
prune record naming the controller is planned as the get's copy.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index c1d30c4..24886f7 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -1256,7 +1256,7 @@ class ShardedPool:
             )
 
         if (
-            operation not in ("get", "store", "validate")
+            operation not in ("get", "store", "validate", "prune")
             or cls_name not in by_name
             or controller not in snapshots
         ):
```

**(vii) The pool writes the version row through `_get_impl_replicated_table` when the label is already
present.** Fails `test_reconcile_at_open.TestCleanStoreUntouched.test_opening_a_clean_store_writes_nothing`.
Also:
- `TestUninterruptedPrune.test_a_prune_with_nothing_to_prune_writes_nothing`;
- three tests of `TestNewLabelInterrupted`, in each of which a reopen under a label the store holds
  changed a file or made a version get;
- `TestInterruptedPrune`'s second-reopen checksum, in four (sub-)tests.

9 failures of 25.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index c1d30c4..6b6f608 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -294,7 +294,7 @@ class ShardedPool:
         # already holds the label. Only a label no shard holds is written. Then every actor is
         # given the serial; until then an actor refuses every insert that carries it
         self._version = self._find_version_row(version_label)
-        if self._version is None:
+        if True:
             self._version = ray.get(
                 self._get_impl_replicated_table("version", {"label": version_label})
             )
```

## Observations not acted on

1. **U2's window** (`[00-a-new-stores-first-open-can-leave-a-primary-without-its-shards]`, record
   only) **narrowed and is not closed.** The shard files after the first no longer wait on the
   first actor's constructor, its version insert and a `ray.get`. Every constructor is submitted at
   once, and the version row comes after the last file exists. One line is added to that entry on
   the board.
2. **A store whose `replication_in_flight` predates this commit cannot take a prune record.** It
   refuses a pruning open that has a replicated row to prune (*The record's schema*). Every store
   built before this commit is such a store; A3 v2 will be built after it. No store is migrated
   (README §5, and the user: old stores are read at their own SHA). Not a defect.
3. **The probe's case 4 no longer measures anything about the prune.** Its `start` kill is keyed on
   `version_serial`. The probe and its output are the audit's and are not changed (prompt §3 item 9).
4. **`docs/a3-v2-readiness/readonly_open_probe.txt` and `reader_writes_probe.txt` describe the tree they
   were taken on.** For example, M1's "a version label the store has not seen: `version +1` on every
   shard" now also commits and clears a record on the primary. They are records of `5707851`, not
   claims about this tree (CLAUDE.md rule 6). 03 re-measures what it relies on. Both probes still
   run to the end on this tree (exit 0, `ray.is_initialized() = False`), with their output written
   to the scratch directory and not committed.
5. **`[00-a-drop-action-drops-a-replicated-table-outside-the-in-flight-record]` is untouched** (§6).
   `_drop_actions` still runs in each actor's constructor. After this prompt it is the one
   replicated write a constructor makes. Its board entry's "the third replicated write a
   constructor makes, beside the version row and the prune" is left as written: that entry is not
   this prompt's to edit.
6. **Actors are built concurrently under Ray now.** Each `Datastore.__init__` used to wait for the
   first actor's version get; now none waits. Each actor opens only its own file, so nothing is
   shared between them. Noted because the order of their prints in a real run's log will
   interleave.

7. **`docs/datastore-integrity-audit/replicated_write_fault_probe.py` stops at its step 5**, with
   `ReplicationInFlight` (a get after a killed replica, with no reopen between). It does the same,
   with the same traceback, on an export of `56d1988` (`git archive HEAD`, in the scratch
   directory), so this prompt did not cause it. That probe predates the in-flight record
   (`datastore-integrity` prompt 01), and its committed output is a record of that earlier tree. It
   creates its store in a directory beside itself and removes it; none was left behind.

No new issue is opened. The index count is therefore 104: 106 less the two rows this prompt closes.

## State handed to the next prompt

- **For 03.** The actor method is **`set_version`**, and the record's operation for a prune is
  **`prune`** (the orchestrator's: both names fixed).
  - The version row is findable by `SELECT serial, label FROM version WHERE label = ?` on a
    `mode=ro` shard file. `ShardedPool._find_version_row` is that read, and it writes nothing.
  - A read-only pool can call `_find_version_row` and refuse on None, without
    `_get_impl_replicated_table`.
  - Whether it calls `set_version` is 03's choice. Without it, every insert of a versioned class
    is already refused by `_insert`.
  - A `prune` record left on the primary is a write in flight, which 03 refuses like any other.
  - Under `read_only=True`, `_prune_replicated_tables` and `_complete_interrupted_prune` are both
    writes.
- **The actor's constructor signature changed.** `version_serial` is gone and `replicated_tables`
  is required. Only `ShardedPool` constructs actors (the orchestrator's, and a search of the tree).
- **`ShardedPool.pruned`** lists what the open's prune deleted, one entry per table and shard.
- **Baselines** after this commit: §5.4 of the board. Datastore is 551 + 23.
- **Index:** 104 open.
