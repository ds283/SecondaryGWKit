# Orchestrator — prompt 03, delete the reconcile's old-store paths

Read [`../README.md`](../README.md) §0.1, §1, §4, §5 and §6.1 (D3, D4). Then the audit's B1–B4, B9
and B10 ([`docs/datastore-generic-audit.md`](../../../docs/datastore-generic-audit.md)), `git show
e53f323`, and the board's reviews of 01 and 02 (§1). The conventions are
[`prompt-01.md`](prompt-01.md)'s, with what the reviews of 01 and 02 added (§0, last list).

**You do not write code.** You may:
- run the suites and the tests the log names;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process check from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`03-delete-the-reconciles-old-store-paths.md`](../03-delete-the-reconciles-old-store-paths.md)
**Board items:** R1–R4 · **Closes:** nothing · **Narrows:**
`[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]` · **Model:** Opus, as the prompt
recommends.
**Gate:** 02 landed (`9d3a42c`) and reviewed (`587a9ae`); `e53f323` an ancestor of `HEAD`; the tree
clean. `HEAD` is the commit that adds this note, or a later one that touches no code. If a later
commit touches `Datastore/`, re-check every fact below that it could move before dispatch, and hand
the agent the corrections as an addendum. One prompt at a time in this checkout.

## 0. What makes this prompt unusual

**It changes what decides whether a store opens.** 01 and 02 deleted refusals: an old store got a
worse message, and a current store was untouched. 03 changes `_replicated_table_specs`, which both
the read-write and the read-only open compare with, and the comparison that a repair must pass
before it commits. A mistake here either refuses a good store or opens a divergent one. So the
review leans on the *measurement* (R2) and on the existing repair tests passing untouched, more
than on the deletions.

**It reviews a commit it did not write.** `e53f323` was written outside the campaign and merged
into `handover-remedial` on 2026-10-04. R1 is a review of it, not a re-implementation. The agent
reads it against the code at `HEAD`, says in the log whether its argument holds, and fixes in 03's
commit only what the review finds wrong. Nothing has touched `ShardedPool.py`, `Datastore.py`,
`test_reconcile_at_open.py`, `test_replicated_write.py`, `test_prune_at_open.py` or
`shard_store_fixtures.py` since `e53f323` (`git log e53f323..HEAD` on them is empty), so its diff
applies to the code as it stands.

**The prompt's breakage (iv) cannot fail as written; this note corrects it.** (iv) restores "the
constructor's creation of `replication_in_flight` on an existing store". But the constructor reads
the record **before** it would create the table:
- `self.reconciliation = self._reconcile_replicated_tables(primary_journal)` is at `:227`;
- `self._replication_in_flight_table.create(self._engine, checkfirst=True)` is at `:231`;
- `_reconcile_replicated_tables` calls `_read_in_flight_record` at `:1029`.

So with R3's `has_table` test gone, restoring only the creation changes nothing: the open still
fails on `no such table` at `:1029`, and any test of that failure still passes. B3 is **two** sites,
and the mutation must restore both:
- the creation at `:229-231`;
- `_read_in_flight_record`'s `has_table` test at `:1209-1210`.

With both restored, a primary without the table opens again, and the test that says it must not
fails. R3 deletes the two tests that asserted the old behaviour and adds **one** that asserts the
new: a read-write open of a primary whose `replication_in_flight` table was dropped
- raises, naming `replication_in_flight`;
- writes nothing (`sp.store_checksums` unchanged);
- leaves the primary without the table.

The read-only open reaches `_read_in_flight_record` too (`:525`). A second assertion or sub-test
for it is welcome, not required. The log records the corrected (iv) as a deviation, STRUCTURALLY
REQUIRED: the prompt's mutation cannot fail any test.

**R2 is a measurement first, and its result is a stop condition.** D4 lets `timestamp` be compared
only after it is shown that every replicated write path stamps one timestamp. If any path stamps
per shard, that is a defect in current code, and the agent **stops and asks**. The orchestrator's
reading (2026-10-04, at `587a9ae`) finds no such path [read]:
- **`Datastore._insert`** (`Datastore.py:848`) is the only place a replicated row is stamped. It
  uses `insert_timestamp` when given, and `datetime.now()` otherwise (`:897-903`).
  `_inserters_for` (`:825`) binds every table's inserter to the call's one timestamp, tag tables
  included. `_replicated_write` passes one `timestamp = datetime.now()` (`ShardedPool.py:2932`) to
  the controller and to every replica (`:2953`, `:2972`).
- **The repair at open** copies the controller's rows verbatim. `_read_full_rows` (`:1729`) reads
  every column the table has, `timestamp` included, and `_apply_replicated_repair` inserts those
  mappings unchanged (`:1766-1769`). Its flag updates (`:1785`) set only the flag columns.
- **The prune at open** and **the completion of an interrupted prune** only delete.
- **The version row** has no `timestamp` column. Neither has `BackgroundModelValue`.
- **The replicated tables with a `timestamp` column:** `store_tag`, `redshift`, `wavenumber`,
  `wavenumber_exit_time`, `tolerance`, `LambdaCDM`, `QCD_Cosmology`, `IntegrationSolver`,
  `BackgroundModel`, `GkSourcePolicy` and `QuadSourcePolicy`, plus the tag table
  `BackgroundModel_tags`.

The agent's measurement replaces this reading; it does not cite it. Two tools exist already:
- **`test_replicated_write._TickingDatetime`** patches `sp.ds_mod.datetime` so that every shard's
  clock ticks on every call. Under it, only a timestamp the driver chose can agree.
  `TestCleanWrite.test_every_copy_is_row_identical` uses it for a clean get, store and validate.
- **`test_reconcile_at_open`'s `run_case`** ends in `assertIdentical()`, which compares every
  column of every row through `sp.shard_rows` (`SELECT *`), `timestamp` included, after every
  repair `TestKillAndReopen` makes. It does not tick the clock.

The cheapest complete measurement runs the repair cases, an interrupted prune's completion and the
version row's write under the ticking clock, and asserts `shard_rows` identity afterwards. The
design is the agent's.

**R1's three tests: one is new, two probably exist.** The prompt asks the agent to name existing
ones rather than add.
- **(a) A `BackgroundModelValue` serial split.** Not present: `TestSerialSplit` splits a
  `tolerance`. Add it. Under `e53f323`, `BackgroundModelValue` is `serial_keyed`, so
  `_describe_differences` (`:1395`) names a split for it as it does for any serial-keyed table.
- **(b) An interrupted `BackgroundModel` store repaired, with the values under the controller's
  serials.** Probably `TestKillAndReopen.test_store_of_a_background_model` and
  `test_store_of_a_second_background_model_beside_the_first`: `run_case`'s `assertIdentical`
  compares full rows, serial included. The agent confirms this and names them.
- **(c) A shard holding the model row but lacking some of its value rows is refused.** Probably
  `TestRecordDoesNotExplain.test_part_of_a_background_model`, under a record, which reaches the
  unit check's "holds its model row" branch. Without a record, it is
  `TestNoRecordRefuses.test_a_value_count_difference`. The agent confirms which branch each reaches.
  If no existing test reaches that branch with the record naming **the same** model, it adds one.

**The facts, checked by the orchestrator on 2026-10-04 at `587a9ae`.** Pass them on.

- **`ShardedPool.py`'s sites** [read]. The audit's line numbers are from `dbd8a51`, before
  `e53f323`; these are at `HEAD`.
  - **The comment block** the prompt calls "THE CHECK AT OPEN" is headed **"THE REPLICATED TABLES,
    RECONCILED AT OPEN"** (`:944-985`). **It nowhere says that `timestamp` is not compared**. Its
    step 3 says that rows are "compared by full row" (`:969`), which becomes true after R2. There is
    nothing in it for R2 to correct unless the agent finds otherwise.
  - **`_replicated_table_specs`** (`:1137`). The sentence "(every store written before prompt 01
    stamps it per shard)" is at `:1146`, and `c.name != "timestamp"` at `:1164`.
  - **R3, the constructor:** `:229-231`. **`_read_in_flight_record`:** the docstring sentence at
    `:1204-1205` and the `has_table` test at `:1209-1210`.
  - **R4.**
    - The comment at `:712-715`, from "A store whose table was created before that prompt" to
      "(_commit_prune_record)". The rest of the comment, down to "controller_shard is NULL", is
      current and stays.
    - `_commit_prune_record`'s `try`/`except sqla.exc.IntegrityError` at `:2102-2121`. The insert
      stays.
    - **B9:** `label = row[columns.index("label")] if "label" in columns else None` at `:1838`.
  - **Not 03's, and staying:**
    - the repair licence's `timestamp` test, `row["timestamp"] != started` (`:1659-1666`), and its
      docstring (`:1499-1512`). This is current code: it is how the repair knows that the recorded
      write inserted a row;
    - the project names in the unit lists (`:1544-1552`, `:1964`) and `_recompute_validated`'s
      `_factories["BackgroundModel"]`, which are 07's;
    - the absent-table and absent-column tolerance in `_read_replicated_tables` (`:1233`), which is
      05's.
- **The verification grep**, `grep -nE 'before prompt 01|written before|predates'
  Datastore/SQL/ShardedPool.py`, finds today `:229`, `:1146` and `:1204`. Widened with `created
  before` it also finds `:713`. After 03 it must find none of these. `_commit_prune_record`'s
  message ("was created before a prune could be recorded", `:2116-2117`) goes with its handler.
- **`e53f323`, for R1** [read].
  - It changes `_replicated_table_specs`: one key rule, `timestamp` alone excluded.
  - It changes the unit check in `_plan_replicated_repair` (`:1668-1717`). A child row's model is
    now the column that the child's foreign key to `BackgroundModel` names. Both children have one:
    `BackgroundModel_tags.model_serial` (primary key `(model_serial, tag_serial)`) and
    `BackgroundModelValue.model_serial` (primary key `(serial,)`). `snap["columns"]` holds the
    compared columns, and `model_serial` is one.
  - It deletes the check that copied value serials are free on the target shard.
  - It updates `test_reconcile_at_open.KEYS`.
  - **The orchestrator's reading of its argument**, for the agent to test rather than copy.
    - For a current store, value serials agree across shards (since prompt 01). So keying
      `BackgroundModelValue` by `serial` rather than `(model_serial, z_serial)` finds the same
      differences.
    - A copy of serial S onto a shard holding a different row under S is now a `differs` on key S.
      The licence turns on only flags, so it refuses that difference. The deleted check refused the
      same case. So the repair's licence for a current store is unchanged, and stop condition 2
      does not fire. **If the agent finds a case where it does, it stops.**
  - **README §5 rule 11.** The unit check now names `"BackgroundModel"` once more, in the
    foreign-key lookup. The code around it already names that table (`:1544-1552`). This is
    existing project knowledge in `ShardedPool`, 07's to remove, not new. The log may say so; it
    does not need fixing.
- **The tests R3 and R4 delete, exactly** [read, counted with `ast`]. Three methods, all in
  `Datastore`:
  - `test_reconcile_at_open.TestNoRecordRefuses.test_a_store_written_before_the_record_existed`
    (`:723`). The class keeps its six other tests.
  - `test_replicated_write.TestCleanWrite.test_an_existing_store_gains_an_empty_record_table_when_opened`
    (`:238`). The class keeps two. The module's `sqlite3` import stays, because `:669` uses it.
  - `test_prune_at_open.TestRecordTableBeforePrune.test_a_pruning_open_refuses_before_any_shard_is_touched`
    (`:625`). It is the class's only test, so the class goes.

  These are the only tests that drop or recreate `replication_in_flight` (`grep` for
  `replication_in_flight` with `DROP|CREATE TABLE|has_table` over `Datastore/tests/`).
- **`write_legacy_primary` needs no re-fixture** [read]. It builds a primary without
  `replication_in_flight`. Its users are `test_copy_move_store`, `test_delete_store`,
  `test_shard_key_audit_copy` and `test_shardedpool_shard_paths`. They read it with
  `shard_store_fixtures.read_pool`, which stops before the check at open, or with the copy, move and
  delete operations, none of which reads the record. No code outside `ShardedPool.py` reads
  `replication_in_flight`, except `Datastore/replication.py`'s messages and three `docs/`
  verification scripts, which are not run. So nothing re-fixtures for R3. If the agent finds a user
  that does open the store, the prompt's rule applies: it lists it.
- **Stop condition 3 does not fire** [read]. `_write_shard_data` creates `replication_in_flight`
  (`:736`) with the other five primary tables. It then inserts every row in one transaction
  (`:758-787`). A first open interrupted before that commit has no `shard_key_config` row, and
  `_read_shard_data` raises "No configured shard key type was found" (`:828`) before the check at
  open. So every primary that reaches `_read_in_flight_record` has the table.
- **Comparing `timestamp` breaks no fixture** [read].
  - `real_store_fixtures` and the in-memory stores stamp `FIXED_TIMESTAMP`.
  - The stand-in pools stamp through the driver.
  - Hand-inserted rows in `test_reconcile_at_open` (`:700`, `:780`, `:906`) are differences or
    rolled back anyway.
  - `test_store_inventory.test_timestamps` does not open a pool.

  **If an existing test outside the three now fails, that is current behaviour moving**: stop and
  report it, never edit the test.
- **Other suites.** `ComputeTargets/tests/test_quadsource_policy_main.py` opens a stand-in
  `ShardedPool`, so the check at open runs there too. It is expected to pass unchanged. No other
  suite opens a pool.
- **Expected counts after 03.**
  - **Datastore:** 600 − 3, plus 03's new tests: R1 (a), and any (b)/(c) it adds; R2's measurement
    and its timestamp refusal; R3's absent-table test.
  - **ComputeTargets** 693, and every other suite, unchanged.
- **The index.** **107** at `587a9ae` (`grep -c '^| \`\[' docs/OPEN_ISSUES.md`, matching the
  header). 03 closes nothing, so it is **107** after, plus one for each issue it opens. The fallbacks
  row gains 03's part of its narrowing, naming `e53f323` as reviewed.

**What the review exists to establish.**
- **(E1) The old-store paths are gone, and only they.**
  - B1: `timestamp` is compared.
  - B3: `replication_in_flight` is assumed, at both sites.
  - B4: the `NOT NULL` handler and its comment are gone.
  - B9: `label` is read unconditionally.

  Nothing else in `ShardedPool.py` changes, beyond what the review of `e53f323` found and recorded.
- **(E2) Every replicated write path stamps one timestamp, measured** before the comparison relies
  on it. The measurement names each path: the clean write, the repair, the prune completion, the
  version row and the prune at open.
- **(E3) Nothing a current store does changes.**
  - Every existing test outside the three deleted ones passes untouched, including every
    `TestKillAndReopen` repair, `test_read_only_pool` and `test_version_row_at_open`.
  - A clean store opens and writes nothing (`TestCleanStoreUntouched`).
- **(E4) `e53f323` is reviewed.** The log says whether its argument holds and names the tests for
  (a)–(c). Breakage (iii) fails a test.
- **(E5) The pins.** Breakage (i) fails R2's timestamp-refusal test. (ii) fails the measurement on
  at least the clean write and the repair. (iii) fails a unit-check test. The corrected (iv) fails
  R3's absent-table test.
- **(E6) The records.** The fallbacks issue is narrowed with a dated line naming 03's deletions and
  `e53f323`; the index is 107 (or more, by issues opened); the board has the §1 row, R1–R4 and §5.

**Conventions, carried from 01 and 02.** All of these bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only. Show `git diff --cached --name-only` before
  committing. Count, then paste. A log states only what the diff and the tests show.
- **The SHA.** The agent writes "this commit"; the orchestrator substitutes the SHA in its
  follow-up.
- **Suites.** Use README §7's command, write the output to a file, and grep the verdict from it.
  **Run each suite in the foreground and wait for it within the turn.** Run them serially.
  - 01's agent ended its turn waiting on background suites three times; 02's, told this, did not.
  - Under load, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`,
    `test_background_tau.test_build_cost` and
    `test_numeric_phase_cut.test_the_print_is_gated_both_ways` fail.
  - Re-run a failure alone before calling it real. If it persists alone, settle it on an export of
    the tree before the prompt.
- **Breakage diffs** are recorded so that `git apply` replays them from a clean tree at the agent's
  commit. Check each with `git apply --check` and `-R`, and replay in `bash`. Under (ii) many tests
  fail. The log names the measurement's failures and gives the total, not every name.
- **Dead helpers** that only deleted tests used go in the same commit, after a grep
  (IMPLEMENTATION CHOICE). This includes an import left unused.
- **Scratch.** The agent uses its own subdirectory of the session scratchpad, never `/tmp`.
- **Ray.** 03 starts no Ray. The stand-in pools need none. ChamPBH's cluster (`gcs_server` 62525,
  `raylet` 62574) runs from its own venv and is left alone.
- **Other sessions** may land commits on this branch. Review against the agent's commit, and check
  `git log` before measuring at `HEAD`.
- **Usage limits.** Resume a stopped agent with `SendMessage`, saying exactly where it stopped.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`. `git status --short` and `git diff --cached` must be
   empty, and `git worktree list` must show this checkout alone. `git log --oneline 587a9ae..HEAD`
   must touch no code (otherwise see the gate). `git merge-base --is-ancestor e53f323 HEAD`
   must succeed.
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`. At
   `587a9ae` it held 16 runs: 14 finished, 2 `unknown`.
3. **Ray.** Record `pgrep -lf 'gcs_server|raylet|ray::'`. No listed process may come from this
   repository's `venv`.
4. **Baselines.** Last measured at `9d3a42c` (02's review, check 10): AdaptiveLevin 32,
   ComputeTargets 693, CosmologyModels 39, Datastore 600, LiouvilleGreen 148 (1 skipped), RunRegistry
   281. `587a9ae` changed one test docstring. Re-measure all six at `HEAD`, serially, with the load
   average, and give the agent the numbers.
5. **The index.** 107 now, and 107 after unless 03 opens an issue. Tell the agent both.
6. **The pre-images**, for checks 1–3. Record `git rev-parse HEAD:<path>` for:
   - `Datastore/SQL/ShardedPool.py` and `Datastore/SQL/Datastore.py`;
   - `test_reconcile_at_open.py`, `test_replicated_write.py`, `test_prune_at_open.py`,
     `test_read_only_pool.py`, `test_version_row_at_open.py`, `standin_pool.py` and
     `shard_store_fixtures.py`.

## 2. Dispatch

Dispatch one fresh-context subagent, on **Opus**. Give it:
- the prompt;
- the campaign README, the board (including the reviews of 01 and 02), 02's log, and the audit's
  B1–B4, B9 and B10;
- `HEAD`, the baselines and the index counts;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad. The
  agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/03-delete-the-reconciles-old-store-paths.md`, in README
  §5.1's form, with the prompt's §7 additions: the review of `e53f323`, R2's measurement path by
  path, and every test deleted.
- **The files it may add or change:**
  - `Datastore/SQL/ShardedPool.py`. The changes are limited to:
    - R2's `_replicated_table_specs`;
    - R3's two sites;
    - R4's handler and comment, and B9's line;
    - whatever the review of `e53f323` finds wrong, each one recorded as a deviation.
  - `Datastore/tests/test_reconcile_at_open.py`, `test_replicated_write.py` and
    `test_prune_at_open.py`: the three deletions, their dead helpers, and new tests;
  - a new test module, if the agent prefers one for R2's measurement;
  - `Datastore/tests/standin_pool.py`, **additions only**, if the measurement needs a helper;
  - the log, this board and `docs/OPEN_ISSUES.md`.

  **Nothing else.** In particular, it must not touch:
  - `Datastore/SQL/Datastore.py` (breakage (ii) is a mutation, never committed);
  - the factories, the reader, the inventory or `real_store_fixtures.py`;
  - `shard_store_fixtures.py`, whose `write_legacy_primary` is 04's;
  - any other existing test;
  - the absent-table and absent-column tolerance (05);
  - any project name in `ShardedPool` (07);
  - anything under `var/`, `docs/` (other than the index) or `orchestrator/`.
- **Stop and ask** on any of the prompt's §5 conditions. This note's reading answers all three as
  "does not fire" at `587a9ae`. The agent's own measurement and review decide. An existing test
  failing outside the three deletions is a sign that current behaviour moved, not a reason to edit
  the test.
- **The corrected breakage (iv)** (§0): restore both B3 sites, and see R3's new test fail.

## 3. The review — ten checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only §2's list. The diff over
   `Datastore/SQL/Datastore.py`, `Datastore/SQL/ObjectFactories/`, `Datastore/store_*.py`,
   `Datastore/tests/{real_store_fixtures,shard_store_fixtures}.py`, `config/`, `main.py`, `docs/`
   (other than `OPEN_ISSUES.md`) and `var/` is empty. Under `*/tests/`, only the three modules and
   any new module or `standin_pool.py` addition change.
2. **(E1) `ShardedPool.py`.**
   - **By grep.** The verification grep, widened with `created before`, prints nothing that
     describes an older store. `has_table("replication_in_flight")`, `checkfirst=True` on that
     table and `IntegrityError` are gone. `"label" in columns` is gone.
   - **By `ast`, against the pre-image.** The changed functions are:
     - `__init__`;
     - `_replicated_table_specs`;
     - `_read_in_flight_record`;
     - `_commit_prune_record`;
     - `_recompute_validated`;
     - anything the review of `e53f323` fixed, each matched to a deviation in the log.

     The `:712-715` comment sits in `_create_engine`. A comment does not change the `ast`, so read
     the diff for it, and check that `_create_engine` dumps identically. The repair licence's
     timestamp test (`:1659-1666`) is unchanged.
3. **(E3) The tests, by `ast`.** List the test methods of the changed modules before and after.
   - Exactly the three of §0 are gone.
   - Every new method is R1's, R2's or R3's, and the log names its purpose.
   - Under `*/tests/`, no existing method body changes; compare each surviving method's dump with
     its pre-image.
4. **(E4) `e53f323`, by reading the log.**
   - The log states whether the argument holds, and why.
   - It names the tests for (a)–(c), and says which it added.
   - It records anything found and fixed as a deviation.

   Compare its argument with §0's reading. Where they differ, read the code and decide.
5. **(E2) The measurement, by reading and running.**
   - Each of the five paths is driven under a clock that ticks per shard (or an equivalent the log
     justifies).
   - Each asserts row identity, `timestamp` included, on every shard.
   - Run the measurement's tests by name and see them pass.
6. **The new refusals, by running.**
   - R2's test: two shards differing only in `timestamp` are refused, and the difference names
     `timestamp`.
   - R3's test: an absent `replication_in_flight` is refused at open, nothing is written, and no
     table is created.
   - R1 (a): a value-row serial split is named as a split.
7. **(E5) The pins.** In `bash`, replay each breakage diff (i)–(iv) with `git apply`, run the
   named tests and see them fail.
   - For (iv), also apply **the prompt's (iv) as written**: restore only the creation at
     `:229-231`. See R3's test **pass**. That is §0's point, and the review records it as expected.
   - Revert each with `git apply -R`, and check that `git status` is clean.
8. **The probe, against the fix.** From the scratchpad, at the agent's commit and at an export of
   `HEAD~1`, through `standin_pool`:
   - **A clean store, written and reopened**, opens at both, and writes nothing.
   - **An interrupted `BackgroundModel` store, reopened**, is repaired at both, and the shards are
     row-identical afterwards, `timestamp` included.
   - **A store whose copies differ only in `timestamp`** opens at `HEAD~1`, and is refused at the
     agent's commit, naming `timestamp`.
   - **A primary with `replication_in_flight` dropped** opens at `HEAD~1`, which creates the table,
     and is refused at the agent's commit.
   - **The read-only open** of the timestamp-divergent store is refused at the agent's commit too.
9. **(E6) The records.**
   - The fallbacks issue carries a dated line naming 03's deletions and `e53f323` as reviewed.
   - The index is 107 (or more, by issues opened), by counting, matching the header.
   - This board has the §1 row, R1–R4 and §5.
10. **Suites.** All six, serially at `HEAD`:
    - Datastore at §0's expected count, with each new test named in the log;
    - ComputeTargets 693, with `test_quadsource_policy_main` passing;
    - every other count unchanged.

    `black --check` is clean on every touched Python file.

## 4. After the review

- **Record it on this board** as a §1 *Orchestrator review of prompt 03* paragraph, in the form
  of 01's and 02's. Fix small residue in a follow-up commit of its own, including "this commit" →
  the SHA, and this note marked "used" in the board's list.
- **Report to the user:**
  - what R2 measured, path by path, and that every one stamps one timestamp;
  - the verdict on `e53f323`;
  - what a current store sees (check 8: nothing changes);
  - what an old store now meets;
  - the three tests deleted and the tests added;
  - what remains of the old-store fallbacks (04, 05).
- **What 03 hands on.**
  - **04** (not yet written) deletes the legacy shard records and decides what becomes of
    `write_legacy_primary`. Its note must say that 03 left it untouched and that it builds a
    primary without `replication_in_flight`, which no longer opens.
  - **05**'s note must carry the absent-table and absent-column tolerance in
    `_read_replicated_tables` (`:1233`), which 03 left alone, beside 02's four `numeric_serial`
    selects.
