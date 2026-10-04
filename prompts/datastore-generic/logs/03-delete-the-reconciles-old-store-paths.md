# Log — prompt 03: delete the reconcile's old-store paths

**Subject:** *Delete the reconcile's old-store paths* · **Model:** Claude Opus 5.5 · **Date:**
2026-10-04 · **Base:** `65be64f`, clean · **Result:** landed.
- **R1:** `e53f323` reviewed. Its argument holds and nothing in it was changed.
- **R2:** first measured that every replicated write path stamps one timestamp; then
  `timestamp` was made a compared column.
- **R3:** `replication_in_flight` is assumed at both sites.
- **R4:** the `NOT NULL` prune refusal and B9's conditional `label` are gone.

**Stop condition.** The brief's fourth stop condition fired: four existing tests failed under R2. The
user decided it on 2026-10-04 (Deviations 1). Three tests deleted, 32 added (4 new in
`test_reconcile_at_open`, 28 in a new measurement module). Closes nothing; narrows
`[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]`. Opens nothing. Index 107 → 107.
The commit is this commit.

## What shipped

**`Datastore/SQL/ShardedPool.py`** (line numbers in this commit):
- **R2.**
  - `_replicated_table_specs` (`:1131`) compares every column, `timestamp` included:
    `compared = [c.name for c in table.columns]`.
  - Its docstring loses "(every store written before prompt 01 stamps it per shard)" and now says
    why every column is compared.
  - The comment block "THE REPLICATED TABLES, RECONCILED AT OPEN" never said that `timestamp` is not
    compared. Its step 3 already said "compared by full row", which is now true, so it is unchanged.
- **R3.**
  - The constructor's `self._replication_in_flight_table.create(self._engine, checkfirst=True)`
    and its comment are deleted (they were after `:227`).
  - `_read_in_flight_record` (`:1196`) loses its `has_table` test and the docstring sentence on a
    store written before prompt 01.
  - A primary without the table now raises SQLAlchemy's `OperationalError` ("no such table:
    replication_in_flight"), from both the read-write and the read-only open, before any actor
    exists.
- **R4.**
  - `_commit_prune_record` (`:2085`): the `try`/`except sqla.exc.IntegrityError` and its message are
    deleted; the insert stays.
  - In the comment above the table definition, the three lines on a table created before the prune
    record could be recorded are deleted, leaving "… so controller_shard is NULL" (`:709`).
- **B9.** `_recompute_validated` reads `label = row[columns.index("label")]` unconditionally
  (`:1829`).
- **The approved fix** (Deviations 1), in `_describe_differences`.
  - The serial-split grouping ignores `timestamp` as well as `serial`:
    `if c not in ("serial", "timestamp")` (`:1392`).
  - Its comment is now "the same row, but for its serial and timestamp (its two copies come from
    two writes)".
  - The docstring says "whose other compared values but timestamp are the same".
- **Untouched.**
  - The repair licence's `row["timestamp"] != started` test and its docstring. These are current
    code.
  - The project names in the unit lists and in `_recompute_validated` (07's).
  - The absent-table and absent-column tolerance in `_read_replicated_tables` (05's).
- **The prompt's grep.** `grep -nE 'before prompt 01|written before|predates'
  Datastore/SQL/ShardedPool.py` prints nothing (exit 1), and so does the same grep widened with
  `created before`. Before 03 it found `:229`, `:1146` and `:1204` (and `:713` with
  `created before`), and the message deleted with the handler. Nothing remains to account for.

**Tests deleted** (README §5 rule 7; each existed only for a deleted path). All are in `Datastore`:
- `test_reconcile_at_open.py`: `TestNoRecordRefuses.test_a_store_written_before_the_record_existed`
  (B3). The class keeps its other six tests and gains three (below).
- `test_replicated_write.py`:
  `TestCleanWrite.test_an_existing_store_gains_an_empty_record_table_when_opened` (B3). The class
  keeps two tests. `sqlite3` stays imported, because `test_overlapping_writes_are_a_programming_error`
  uses it.
- `test_prune_at_open.py`:
  `TestRecordTableBeforePrune.test_a_pruning_open_refuses_before_any_shard_is_touched` (B4). It was
  the class's only test, so the class and its section header go. So does the module docstring's
  item 6; the old item 7 becomes item 6.

**Other checks on the tests.**
- **Counted with `ast`** against `65be64f`: exactly these three names disappear, and no others.
- **No helper or import was left dead.** `execute`, `quiet`, `sqlite3` and `mock` are still used in
  each module (`ast` check: no unused import in any of the four test modules).
- **No other test or fixture builds a primary without `replication_in_flight` and then opens it.**
  - `grep` for `replication_in_flight` with `DROP|CREATE TABLE|has_table` over `Datastore/tests/`
    found only the three deleted tests.
  - `shard_store_fixtures.write_legacy_primary` creates none, but its users never reach the check at
    open: they read it with `read_pool` or with the copy, move and delete operations. It is 04's.
  - So nothing is re-fixtured.

**Tests added.**
- **`Datastore/tests/test_reconcile_at_open.py`** (the module docstring's items 3 and 4 are
  extended to name them):
  - `TestNoRecordRefuses.test_a_timestamp_difference` (R2's refusal). r1's copy of the tolerance row
    is given another timestamp. The open is refused with the record absent, naming class `tolerance`
    and shard r1. The detail is "the shards hold different values: timestamp: …" and names no other
    column.
  - `TestNoRecordRefuses.test_a_value_row_held_under_another_serial` (R1 a). r0's last
    `BackgroundModelValue` row is moved to serial + 1000. The open is refused with two
    `BackgroundModelValue` differences, each detail saying "a serial split", and the text naming both
    serials.
  - `TestNoRecordRefuses.test_a_primary_without_the_record_table_is_refused` (R3). Its sub-tests
    `read_only=False` and `True` each check four things:
    - the open raises `OperationalError` naming `replication_in_flight`;
    - no actor is constructed (`_Options.remote` counted);
    - `store_checksums` is unchanged;
    - the primary's `sqlite_master` still has no such table.
  - `TestRecordDoesNotExplain.test_part_of_the_recorded_background_model` (R1 c).
    - A store of model M is interrupted at Rn-P3 (r1 faulted "after"). Every shard holds M, and the
      record names it (`store_id` = M).
    - r0 then loses one of M's value rows.
    - The open is refused with the record, by "shard r0 holds its model row". That is the unit
      check's `got - held` branch, with the record naming the same model.
    - `e.record["store_id"]` is M, and the only difference is in `BackgroundModelValue`.
- **`Datastore/tests/test_one_timestamp_per_write.py`** (new; R2's measurement; 28 tests). Every test
  runs with `sp.ds_mod.datetime` patched to `test_replicated_write._TickingDatetime`, from before
  the store is built (mixin `_TickingClock.setUp`).
  - **What it adds to each test's own assertion.** Each inherited test already asserts that every
    shard holds identical full rows (`SELECT *`, `timestamp` included). `assertIdentical` is
    extended to require, too, that no replicated row on any shard carries a 2030 tick.
  - **The test classes it inherits:**
    - `TestRepairAtOpen` ← `test_reconcile_at_open.TestKillAndReopen` (12);
    - `TestRepairThenPruneAtOpen` ← `TestPruningAfterRepair` (2);
    - `TestCompletionOfAnInterruptedPrune` ← `test_prune_at_open.TestInterruptedPrune` (2);
    - `TestPruneAtOpen` ← `TestUninterruptedPrune` (3);
    - `TestVersionRowOfANewStore` ← `test_version_row_at_open.TestNewStore` (3), plus one new test,
      `test_the_version_row_is_its_serial_and_label`, which asserts that `version`'s columns are
      `["serial", "label"]` on every shard;
    - `TestVersionRowOfANewLabel` ← `TestNewLabelInterrupted` (4);
    - `TestVersionRowOfAnInterruptedNewStore` ← `TestNewStoreInterrupted` (1).
  - **The version classes.** Their tests compare the version rows only. So each version class's
    `tearDown` also compares every replicated table of the last store opened, whole rows, on every
    shard, and checks it for ticks.
  - **Imports.** The parent classes are reached through module attributes
    (`reconcile_at_open.TestKillAndReopen`), so discovery does not run them twice from this module.

## R2's measurement, path by path

Measured **before** `_replicated_table_specs` was changed: the new module was run on `65be64f`'s
`ShardedPool` and `Datastore`, giving **28 tests, OK, 97 s**
(`<scratch>/agent03/r2-measure-head.txt`). It was run again on this commit's tree, inside the
Datastore suite.

| Path | Driven by | Result |
|---|---|---|
| Clean write: get (scalar and vectorized), store, validate, tags and values | `test_replicated_write.TestCleanWrite.test_every_copy_is_row_identical` (already under the ticking clock; not repeated) | every copy identical, timestamp included; no row stamped by a shard's clock |
| Repair at open after an interrupted get, store or validate, at every commit point; flags turned on; a second `BackgroundModel` | `TestRepairAtOpen` (12 tests, 35 sub-cases) | identical on every shard, no tick |
| The same repair followed by the prune at open, with `prune_unvalidated` true and false, and a drop action | `TestRepairThenPruneAtOpen` (2) | identical, no tick |
| Completion of an interrupted prune, faulted after each shard and before the first | `TestCompletionOfAnInterruptedPrune` (2) | identical, no tick |
| The prune at open | `TestPruneAtOpen` (3) | identical, no tick |
| The version row: new store, new label (fault after the controller commits, before it, on one replica), a label no shard holds, a new store interrupted at the version write | `TestVersionRowOfANewStore`, `…OfANewLabel`, `…OfAnInterruptedNewStore` (8) | `version` has no `timestamp` column (asserted); every replicated table identical at the end, no tick |

**No path stamps per shard, so the stop condition did not fire.** What each path does with
`timestamp`, as read and then measured:
- **The repair** copies the controller's rows verbatim. `_read_full_rows` reads every column, and
  `_apply_replicated_repair` inserts the mappings.
- **The prune and its completion** only delete.
- **The version row** has no timestamp, and neither has `BackgroundModelValue`.
- **Every timestamped replicated table** (and `BackgroundModel_tags`) is written only through
  `Datastore._insert` with the driver's `insert_timestamp`.

## The review of `e53f323` (R1)

I read the diff against `_replicated_table_specs`, `_plan_replicated_repair`, `_read_full_rows`,
`_describe_differences` and `_apply_replicated_repair` at `65be64f`. No file it touches has changed
since (`git log e53f323..65be64f` on them is empty).

- **The specs.** One key rule: the primary key.
  - `BackgroundModelValue` is keyed by `(serial,)`, and `serial` is compared. So it is
    `serial_keyed`, and `_describe_differences` names a split in it as in any serial-keyed table
    (pinned by (a)).
  - Before this prompt, `timestamp` was excluded for every table. `BackgroundModelValue` has no
    timestamp column, so the exclusion did nothing for it.
- **The unit check finds a child row's model through its foreign key, for both children.** Read
  from `build_schema`:
  - `BackgroundModel_tags`: primary key `(model_serial, tag_serial)`; foreign key `model_serial` →
    `BackgroundModel`.
  - `BackgroundModelValue`: primary key `(serial,)`; foreign keys `model_serial` →
    `BackgroundModel` and `z_serial` → `redshift`.

  `next(fk.parent.name for fk in … if fk.column.table.name == "BackgroundModel")` gives
  `model_serial` for each. `snap["columns"]` holds the compared columns, which include
  `model_serial`, so `model_column` is that column's index in the controller's snapshot.
  - `models` comes from the `BackgroundModel` copies, keyed `(serial,)`, so `k[0]` is the model
    serial.
  - `model_of` maps each controller row of the child table whose model is in `models` to that
    model.
  - `got - held` refuses a child row copied to a shard that is not receiving its model ("holds its
    model row").
  - `held - got` refuses a model whose part the shard already holds, and names the model as
    `(model_of[key],)`.

  The old code's `key[0]` was the model only because the value key began with `model_serial`. That
  is exactly what breakage (iii) restores, and it fails eight tests.
- **The deleted serial-free check, and its argument.** The commit message says it cannot fire under
  serial keys. **That holds.**
  - A copy of key `(S,)` to shard `sid` is planned only from a `missing` difference whose lackers
    include `sid`, so `sid` holds no row under serial S at all. The insert cannot meet a row
    already there, which is the case the deleted check looked for.
  - The case it existed to refuse (a shard holding another row under S, while the controller's row
    under S is "missing" under the old `(model_serial, z_serial)` key) is now a `differs` on key S.
    The licence admits only flag columns of `redshift`/`wavenumber`, and `validated` on
    `BackgroundModel`, so it is refused, as before.
  - If the shard holds the controller's `(model, z)` row under another serial, that is a row the
    controller lacks, refused by "the record's controlling shard … lacks this row", as before.
- **What the repair licenses for a current store is unchanged** (stop condition 2 did not fire).
  Since prompt 01, value serials agree across shards: `TestCleanWrite` asserts it, and the clean
  write's measurement above does too. So keying by `serial` and keying by
  `(model_serial, z_serial)` find the same differences, and the same copies are planned and
  admitted. Every `TestKillAndReopen` case passes untouched.
- **`_read_full_rows`** filters on `spec["key"][0]`, now `serial`; correct.
- **`_apply_replicated_repair`** inserts by spec order: model, tags, values. Unchanged.
- **The test's `KEYS`** now lists only `BackgroundModel_tags`. `keys_of` falls back to
  `("serial",)` for `BackgroundModelValue`, which matches `reconciliation["repaired"]`; correct.
- **README §5 rule 11.** The foreign-key lookup names `"BackgroundModel"`, which is project
  knowledge already present in the code around it (the unit list), and 07's to remove. Not new;
  not fixed here.
- **Verdict: nothing wrong; nothing changed in it.**

**(a)–(c), as the prompt asks:**
- **(a)** was missing and is added: `test_a_value_row_held_under_another_serial`.
- **(b)** exists:
  - `TestKillAndReopen.test_store_of_a_background_model`, with six commit points, four of which
    copy;
  - `test_store_of_a_second_background_model_beside_the_first`.

  These reach the unit check with `got == held` for each lacking shard, so neither refusal fires.
  `run_case`'s `assertIdentical` compares full rows on every shard, `BackgroundModelValue.serial`
  included, so the values arrive under the controller's serials. Under (iii) all of them error.
- **(c)** is added.
  - `TestRecordDoesNotExplain.test_part_of_a_background_model` reaches the "holds its model row"
    branch, but with the record naming a **different** model (the second one).
  - `TestNoRecordRefuses.test_a_value_count_difference` has no record, so it is refused at step 4
    and never reaches the unit check.
  - So `test_part_of_the_recorded_background_model` is new.

## Deviations from the prompt

1. **`_describe_differences` groups a serial split by every compared column but `serial` *and
   `timestamp`*.** `STRUCTURALLY REQUIRED`.
   - **The stop condition.** With R2 alone, four existing tests failed, each every time it ran (not a
     wall-clock flake). The brief's stop condition, "any existing test outside the three deleted
     ones fails", fired. I stopped and reported, and **the user approved this one-line fix on
     2026-10-04**. The four tests are not edited:
     - `test_reconcile_at_open.TestSerialSplit.test_with_the_record_set`
     - `TestSerialSplit.test_with_no_record`
     - `TestMismatchLeftByTheCheck.test_a_vectorized_get`
     - `TestMismatchLeftByTheCheck.test_a_store`
   - **Why it is required.** A real serial split always comes from two separate writes: a replica
     that already held the row under its own serial, and the controller's later write under
     another. So its copies always differ in `timestamp`. Once R2 compares `timestamp`, a grouping by
     "every compared column but `serial`" never matches, and a split is never named.
   - **What it changes.** Only the wording of a refusal: "a serial split …", and with the record set
     the other serial. Whether a store opens never changes: every one of the four was refused
     before and after. Before R2 the grouping never saw `timestamp`, so this restores exactly what it
     compared.
   - **Pinned** by breakage (v).
   - **Where.** `_describe_differences` is outside the brief's list of allowed `ShardedPool` sites,
     which is why it went to the user. Its comment and docstring each gained one short clause
     saying why.
2. **The prompt's breakage (iv), as written, cannot fail any test, so (iv) restores both B3
   sites.** `STRUCTURALLY REQUIRED` (from the orchestrator's note).
   - The constructor reads the record, inside `_reconcile_replicated_tables`, before the creation
     would run. So with the `has_table` test gone, restoring only the creation still fails the open
     on "no such table", and R3's test still passes.
   - **Measured.** The creation-only diff (the first hunk of (iv) below) gives
     `TestNoRecordRefuses`: Ran 9, **OK**.
   - The corrected (iv) restores the creation and the `has_table` test, and fails R3's test.
3. **R2's measurement is a new module that reruns existing test classes under the ticking clock**,
   rather than new tests in `TestCleanWrite`. `IMPLEMENTATION CHOICE` (the brief allows a new
   module).
   - The existing classes already drive every commit point and assert full-row identity, so
     inheriting them measures every path with no new driving code.
   - Cost: about 100 s added to the Datastore suite.
4. **The R3, R1 (a) and R2 refusal tests are in `TestNoRecordRefuses`**, whose `setUp` store holds
   every replicated class. R3's test replaces the deleted test in place, and also covers the
   read-only open (welcome, not required). `IMPLEMENTATION CHOICE`.
5. **Module docstrings.** Edited only where they listed what changed:
   - `test_reconcile_at_open`: items 3–4;
   - `test_prune_at_open`: item 6 deleted, item 7 renumbered.

   `IMPLEMENTATION CHOICE`.

No `UNINTENDED DRIFT`.

## Verification performed

- **Suites**, serially on the staged tree, README §7's command, output under
  `<scratch>/agent03/suite-*.txt`, verdict grepped from the file:

  | Suite | Before (orchestrator, `65be64f`) | After | Difference |
  |---|---|---|---|
  | AdaptiveLevin | 32 OK | 32 OK | 0 |
  | ComputeTargets | 693, 1 failure (`test_tk_wkb_phase.TestCost.test_wall_time_per_object`, wall clock) | 693, 2 failures (`test_background_tau.test_build_cost`, which passed alone 3 times of 3; `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, which failed alone 3 times of 3 and also 3 times of 3 on an export of `65be64f`: load) | 0 |
  | CosmologyModels | 39 OK | 39 OK | 0 |
  | Datastore | 600 OK | 629 OK (716 s) | −3 deleted, +4 (`test_reconcile_at_open`), +28 (`test_one_timestamp_per_write`) |
  | LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 0 |
  | RunRegistry | 281 OK | 281 OK | 0 |

  - **Before the approved fix**, the Datastore suite ran **629, FAILED (failures=4)**: exactly the
    four tests of Deviations 1. Everything else passed, including every `TestKillAndReopen` case,
    `test_read_only_pool`, `test_version_row_at_open` and `TestCleanStoreUntouched`.
  - `ComputeTargets/tests/test_quadsource_policy_main.py` opens a stand-in pool; it is in the
    ComputeTargets run.
  - The Datastore run (load 11–30) preceded the docstring clause in `_describe_differences`. That
    clause is docstring-only; after it, `test_reconcile_at_open` was run again: Ran 41, OK.
  - **ComputeTargets' two failures are wall-clock tests.**
    - `test_background_tau.test_build_cost` passed alone 3 times of 3.
    - `test_tk_wkb_phase.TestCost.test_wall_time_per_object` failed alone 3 times of 3 (0.064–0.082
      s against 0.06 s). It failed 3 times of 3 (0.073–0.081 s) on a `git archive 65be64f` export
      too, at load 25–40. It is load, not 03.
- **`black --check`** on the five Python files: clean.
- **Staging** by explicit path; `git diff --cached --name-only` shown before the commit.

## The deliberate-breakage record

**How each was run.** Each diff was taken with `git diff` against the staged final tree (the tree
this commit holds), so `git apply` replays it from a clean checkout of this commit. For each:
1. `git apply -R --check` while applied;
2. the tests run;
3. reversed with `git apply -R`;
4. `git apply --check` (forward) on the clean tree.

All passed, and the tree was clean against the index after each. None is committed.

**(i) Stop comparing `timestamp`.** Run on `test_reconcile_at_open`: Ran 41, FAILED (failures=1):
`TestNoRecordRefuses.test_a_timestamp_difference` (the refusal it requires never happens).

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index d766848..a9fa202 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -1155,7 +1155,7 @@ class ShardedPool:
         for name in replicated + tag_tables:
             table = built.tables[name]
             key = tuple(c.name for c in table.primary_key.columns)
-            compared = [c.name for c in table.columns]
+            compared = [c.name for c in table.columns if c.name != "timestamp"]
             specs.append(
                 {
                     "name": name,
```

**(ii) In `Datastore._insert`, stamp every row with `datetime.now()` instead of the driver's
timestamp.** Run on the whole Datastore suite: Ran 629, **FAILED (failures=11, errors=118)**.
- **The measurement fails on the clean write and on the repair**, and on every other path:
  - `test_replicated_write.TestCleanWrite.test_every_copy_is_row_identical`;
  - in `test_one_timestamp_per_write`: `TestRepairAtOpen`, every one of its 35 sub-cases (the
    repair's licence now refuses the controller's own row, "is not the recorded write's start");
  - `TestRepairThenPruneAtOpen` (4 sub-cases), `TestCompletionOfAnInterruptedPrune` (4),
    `TestPruneAtOpen` (2), `TestVersionRowOfANewStore.test_versioned_rows_carry_the_serial_on_every_shard`.
- **The rest.** Every other open of a store holding replicated rows is now refused, because its
  copies differ in `timestamp`. These are in `test_reconcile_at_open`, `test_prune_at_open`,
  `test_read_only_pool`, `test_version_row_at_open` and `test_replicated_write`.

```diff
diff --git a/Datastore/SQL/Datastore.py b/Datastore/SQL/Datastore.py
index 4cb3c24..d6a775a 100644
--- a/Datastore/SQL/Datastore.py
+++ b/Datastore/SQL/Datastore.py
@@ -897,9 +897,7 @@ class Datastore:
             if uses_timestamp:
                 # a replicated write stamps every shard with the driver's one timestamp
                 # (object_get / object_store); every other insert stamps its own time
-                timestamp = (
-                    insert_timestamp if insert_timestamp is not None else datetime.now()
-                )
+                timestamp = datetime.now()
                 payload = payload | {"timestamp": timestamp}
             if uses_stepping:
                 if "stepping" not in payload:
```

**(iii) In the unit check, find a child row's model from the first element of its key again.** Run
on `test_reconcile_at_open` and `test_prune_at_open`: Ran 51, **FAILED (errors=8)**. Each is a
`ReplicatedDivergence` from a repair the unit check now refuses ("holds its model row"):
- `TestKillAndReopen.test_store_of_a_background_model`, at four commit points (C-P2, P2-R1, both
  Ri-Ri+1);
- `test_store_of_a_second_background_model_beside_the_first`;
- `TestPruningAfterRepair.test_an_interrupted_store` (both sub-cases);
- `test_prune_at_open.TestRecordsStayApart.test_a_get_record_of_a_background_model_is_repaired_not_pruned`.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index d766848..3abde06 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -1676,9 +1676,7 @@ class ShardedPool:
                         )
                     )
                     model_of = {
-                        k: row[model_column]
-                        for k, row in snap["rows"].items()
-                        if row[model_column] in models
+                        k: k[0] for k, row in snap["rows"].items() if k[0] in models
                     }
                     got = copies.get((name, sid), set())
                     held = set(model_of.keys())
```

**(iv), corrected (Deviations 2): restore the constructor's creation of `replication_in_flight`
*and* `_read_in_flight_record`'s `has_table` test.** R3's test drops the table from a test primary
itself.
- Run on `TestNoRecordRefuses`: Ran 9, **FAILED (failures=2)**. Both sub-tests of
  `test_a_primary_without_the_record_table_is_refused` (`read_only=False` and `True`) fail with
  "OperationalError not raised": the store opens.
- **The first hunk alone** (the prompt's (iv) as written) gives Ran 9, **OK**.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index d766848..f00e367 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -226,6 +226,10 @@ class ShardedPool:
             # record's controlling shard, and refuse to open on any other difference
             self.reconciliation = self._reconcile_replicated_tables(primary_journal)
 
+            # a store written before the replicated write was recorded has no in-flight table;
+            # create it, empty, only once the store has been checked
+            self._replication_in_flight_table.create(self._engine, checkfirst=True)
+
             # the prune at open of a replicated class is made here, by the pool, under a record:
             # after the check at open (so that an interrupted write is repaired before anything
             # is judged unvalidated) and before the broker or any actor exists. The actors still
@@ -1199,6 +1203,8 @@ class ShardedPool:
         something the write path leaves, and refuses.
         """
         with self._engine.connect() as conn:
+            if not sqla.inspect(conn).has_table("replication_in_flight"):
+                return None
             rows = [
                 dict(row)
                 for row in conn.execute(
```

**(v) (added at the user's instruction): group a serial split by every column but `serial` alone
again.** Run on `test_reconcile_at_open`: Ran 41, **FAILED (failures=4)**, exactly the four tests
of Deviations 1:
- `TestSerialSplit.test_with_the_record_set`
- `TestSerialSplit.test_with_no_record`
- `TestMismatchLeftByTheCheck.test_a_vectorized_get`
- `TestMismatchLeftByTheCheck.test_a_store`

The open is still refused in each; the message no longer names the split.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index d766848..e1f9e17 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -1389,7 +1389,7 @@ class ShardedPool:
             content = tuple(
                 v
                 for c, v in zip(snap["columns"], snap["rows"][d["key"]])
-                if c not in ("serial", "timestamp")
+                if c != "serial"
             )
             held.setdefault((d["table"], content), []).append(d)
         for group in held.values():
```

## Observations not acted on

- **Stop condition 3 does not fire** (read). `_write_shard_data` creates `replication_in_flight` with
  the other primary tables, and inserts every row in one transaction after. A first open interrupted
  before that commit has no `shard_key_config` row (or no table), and `_read_shard_data` raises
  before the check at open. Every primary that reaches `_read_in_flight_record` has the table. No
  current-code recovery test (audit B10) failed.
- **The prune section's comment** ("after the check at open and after replication_in_flight exists")
  is still true: the table always exists now. Unchanged.
- **The read-only open's comment** ("No replication_in_flight table is created") is still true.
  Unchanged.
- **A primary without the table now surfaces as SQLAlchemy's `OperationalError`**, not a named
  refusal. That is the README §4 contract for 03. Whether 05's single place for refusing a store
  whose schema differs should also cover the primary's own tables is 05's question. No issue is
  opened.
- **The unit check's foreign-key lookup names `"BackgroundModel"`** (README §5 rule 11). This is
  existing project knowledge, 07's (F1). No issue.

## State handed to the next prompt

- `_replicated_table_specs` compares every column of every replicated table under its primary key.
  `ShardedPool` assumes `replication_in_flight` on every primary it opens (README §4).
- The serial-split naming ignores `timestamp` and `serial`.
- `Datastore` 629; every other suite unchanged. The index holds 107.
- 04 (`write_legacy_primary`, B6) and 05 (B5, the four WKB selects) are unchanged by this prompt.
