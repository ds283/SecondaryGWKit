# Orchestrator — prompt 02, the version row and the prune at open

Read [`../README.md`](../README.md) first: §0, §1, §4, §5 and §6.2. Then read
[`prompt-01.md`](prompt-01.md) §0 and this board's *Orchestrator review of prompt 01*. Together
they are the conventions this note does not repeat.

**You do not write code.** You may:
- run the suites, the stand-in tests and `version_window_probe.py`;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process patch from the session scratchpad, as §0's measurement was taken, to check a
  claim. Never commit one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`02-write-the-version-row-through-the-recorded-write.md`](../02-write-the-version-row-through-the-recorded-write.md)
**Board items:** W1–W3 · **Closes:** `[02-the-version-row-is-written-outside-the-in-flight-record]`,
on the `datastore-integrity` board's §4, and
`[00-the-prune-at-open-deletes-a-replicated-row-outside-the-in-flight-record]`, on this board's §4 ·
**Leaves open:** `[00-a-new-stores-first-open-can-leave-a-primary-without-its-shards]` (U2) and
`[00-a-drop-action-drops-a-replicated-table-outside-the-in-flight-record]` · **Model:** Opus.
**Gate:** 00 and 01 reviewed. `HEAD` is the commit that adds this note, or a later one that touches
no code. One prompt at a time in this checkout: 05 and 06 may land before or after 02, but not
while it runs. 03 waits for 02's review.

## 0. What makes this prompt unusual

**It changes the order of events in `ShardedPool.__init__`, and what the check at open may write.**
Those are the two places where a wrong change loses a store. Everything 01 did was a comment and a
constant; here the review has to show behaviour, on stand-in shards, for each interrupted write.

**The prompt was amended with this note** (W1, §3 cases 1 and 5, breakage (vii)), after the
orchestrator's reading found two things the prompt as first written did not know. Both are below.
The review checks the agent against the amended prompt.

**The facts, checked by the orchestrator on 2026-10-02 at `1c458d3`.** Pass them on; the agent need
not rediscover them. Each is marked as audit 00 marks its findings.

- **A version `get` on every open writes the primary on every open** [measured]. Every replicated
  `get` goes through `_replicated_write`, which commits a `replication_in_flight` row on the
  primary and deletes it, even on a hit (audit R2). SQLite rewrites the file either way: a scratch
  database's SHA-256 changed after one insert and one delete. The orchestrator then patched
  `ShardedPool.__init__` in-process to end with
  `ray.get(self._get_impl_replicated_table("version", {"label": self._version_label}))` and ran
  `Datastore/tests`. Of 551, **39 failed**:
  - 38 because `store.sqlite` changed on a reopen that should write nothing. That is 36 in
    `test_reconcile_at_open.TestKillAndReopen` (its second-reopen check), plus
    `TestCleanStoreUntouched.test_opening_a_clean_store_writes_nothing` and
    `TestNoRecordRefuses.test_a_difference_in_a_sharded_table_is_not_looked_at`;
  - 1 an artefact of the patch: `test_inventory_retired` inspects `ShardedPool.__init__`'s
    signature.

  So W1 now reads the label's row first, and writes through `_get_impl_replicated_table` only when
  it is absent. After the check at open, every shard's `version` table is identical, so one read
  answers for all. The existing checksum tests are W1's "writes nothing" test, unchanged.
- **The factory's prune swallows its own failure** [read].
  `sqla_BackgroundModelFactory.validate_on_startup` wraps its three deletes in
  `try: … except SQLAlchemyError:` and appends "DATABASE ERROR encountered when pruning". It
  returns normally. A pool that treats the return as success would clear the `prune` record over
  shards that differ, which is the very store this prompt exists to prevent. §3 case 5 now has a
  case for it. The factory is not to be edited; the pool checks the effect.
- **The prune must come after the check at open's repair** [read, and pinned by an existing test].
  An interrupted `validate` of a `BackgroundModel` leaves `validated` set on the controller only.
  Pruned before the repair, the model would be deleted from the other shards and kept on the
  controller. That is stop condition 4. `TestPruningAfterRepair.test_an_interrupted_validate`
  (both `prune` values) is the test that holds this, and must pass unchanged.
- **The prune must also come after `replication_in_flight.create(checkfirst=True)`** [read]. A
  store written before `datastore-integrity` prompt 01 has no such table. Today the constructor
  creates it after the check at open; the `prune` record needs it to exist.
- **Only one replicated class prunes at startup** [read]. Seven factories set
  `"validate_on_startup": True`: `BackgroundModel`, `GkWKBIntegration`, `GkNumericIntegration`,
  `TkWKBIntegration`, `TkNumericIntegration`, `QuadSource` and `GkSource`. Of those, only
  `BackgroundModel` is in `config/sharding.py`'s `replicated_tables`. Its prune deletes
  `BackgroundModelValue`, then `BackgroundModel_tags`, then `BackgroundModel` rows with
  `validated` false or null. The W3 set must be derived (prompt §2), but its value today is that
  one class.
- **The actor does not know which classes are replicated** [read]. `Datastore.__init__` takes no
  table lists. The pool has them as `self._replicated_tables`, from its caller. An actor that
  imported `config.sharding` instead would disagree with any pool given other lists. This is the
  *prompt's choice*, but the log says why the source it chose is the pool's.
- **The version row is admitted by the generic `get` repair as it stands** [read]. `version` is
  the only replicated class with no `timestamp` column (`register()` gives `"timestamp": False`).
  So the licence's "written by the recorded write" test, the row's timestamp against the record's
  `started`, does not apply to it. The record's `store_id` is filled in only at
  `_replicated_write` step 3, after the controller has answered. A fault after the controller
  commits and before step 3 leaves `store_id` null, and the licence then copies any `version` row
  the controller holds and another shard lacks.
  - After W1, nothing else writes `version`, so a second, unrecorded difference in `version` cannot
    sit beside a recorded one. A store damaged by today's defect has no record, and is refused
    before it can reach a recorded write.
  - **The log states this.** It is not expected to change the licence; if it does, the change is
    for `version` alone, and a test shows nothing new admitted for any other class.
  - The stand-in's `"after"` fault on the controller raises at step 2's `ray.get`, before step 3.
    So W2's "fault after the controlling shard commits" case goes through the null-`store_id`
    path.
- **`_insert` leases a serial before it writes** [read]. The leasing is
  `with SerialLeaseManager(…)`; `version` is exempt, other classes are not. W1's refusal of a
  versioned insert before `set_version` must come **before** that `with`. If it comes inside, a
  refused insert still takes a serial from the broker.
- **What the stand-in can and cannot do** (`Datastore/tests/standin_pool.py`) [read].
  - Faults and hooks see **actor method calls**, keyed `(shard, method, class)`, not
    constructors. "No actor constructor inserts the version row" is therefore shown from the
    shard files, e.g. every shard's `version` table empty when the pool's write is faulted
    `"before"` on every shard. A hook cannot show it.
  - `_StandinRandom` pins the controller through `cluster.pool._shards`, and `cluster.pool` is set
    only **after** `ShardedPool(...)` returns. A replicated write made inside `__init__`, which W1
    now makes, cannot be pinned by `cluster.controller` as it stands. The draw is random when
    `cluster.pool` is `None`, and resolved against a stale pool otherwise. §3 case 2's "a fault on
    one replica only" needs a known controller. The record left on the primary names it, and so
    does the call log.
  - The pool's prune runs on plain connections, with no actor, so no stand-in fault reaches it. A
    fault inside it is the test module's own `mock.patch`.
  - `standin_pool.py` may gain **additions** for either purpose, e.g. a way to see the pool under
    construction. No existing caller's behaviour may change; the log lists each addition. It is
    not a test module (its docstring), but every Datastore stand-in test imports it.
- **What W1 does to the audit's probe** [read]. `version_window_probe.py` kills an actor in one of
  two ways. At `start`, it raises when the constructor is given `version_serial`, which W1
  removes, so nothing is killed. At `version`, it raises when `object_get("version", serial=…)` is
  called, which after W1 is exactly the pool's replica call, so that one does fire.
  - **Cases 2 and 3 (`version`) do exercise the new path.** They should now heal.
  - **Cases 1 and 4 (`start`) kill nothing.** They print `returned (no crash happened)`. Case 4 is
    the interrupted prune, so its output is no evidence about W3. §3 case 5 is the measurement.
  - The probe and its `.txt` are not changed (prompt §3 item 9). The log quotes each case's own
    line, and does not summarise four cases as "healed".
- **What 03 depends on** [read: prompt 03]. The actor method is named **`set_version`**, and the
  record's operation for a prune is **`prune`**. 03 refuses on a record "of any operation including
  02's `prune`", and decides whether a read-only pool calls `set_version`. Both names are fixed
  (README §4 does not list them, prompt 02 §2 does).
- **The record's schema on an existing store** [read]. `replication_in_flight` is created with
  `checkfirst=True`, so a store's existing table keeps `controller_shard NOT NULL` whatever the
  new schema says. No store is migrated (README §5; the user: schema changes are free, and old
  stores are read at their own SHA). A `prune` record that cannot be inserted on such a store must
  fail **before** the first shard is touched. The log says what such a store does.
- **No existing test calls the actor constructor directly, or passes `version_serial`** [read]. A
  search of `*/tests/` finds only the stand-in, which constructs through `ShardedPool`. The
  factory-level tests (`test_exact_identity_lookups`, `test_integral_read_batch`, …) build their
  own inserters. So stop condition 2 is not expected to fire on W1. If it does, it is real.

**What the review exists to establish.**
- **(C1) A clean open writes nothing.** The existing checksum tests pass unchanged, and a new
  store's row is written once, through `_replicated_write`.
- **(C2) Every interrupted write heals, and nothing else is admitted.** The version row through
  the generic `get` repair; the prune through its own record. The licence admits nothing new for
  any other class, and the two kinds of record never cross.
- **(C3) The prune deletes exactly what the interrupted prune would have.** It runs after the
  repair, never deletes a validated model or another class's row, and never clears its record over
  a shard whose prune failed.
- **(C4) The pins catch what they claim.** Breakages (i)–(vii) replay with `git apply`, and each
  fails a test the log names.
- **(C5) The records.** Both issues closed on their owning boards with the healing tests named, the
  two that stay open untouched except for one line on U2's window, and the index count.

**Carried from the reviews of 00 and 01.** All bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, never `git add -A`, `.` or `commit -a`.
  Show `git diff --cached --name-only` before committing. Count, then paste. A log states only what
  the diff and the tests show; cite both pool files by function name, never by line.
- **Suites.** README §7's command, output to a file and the verdict grepped from it. Under heavy
  load `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, `test_background_tau.test_build_cost`
  and `test_numeric_phase_cut.test_the_print_is_gated_both_ways` fail. Re-run a failure alone
  before calling it real.
- **Breakage diffs** are recorded so that `git apply` replays them from a clean tree at the
  agent's commit (01's log did this well).
- **Residue** found in review is the orchestrator's, fixed in a follow-up commit (the user,
  2026-09-29).
- **Scratch.** The agent works in its own subdirectory of the session scratchpad, never its top
  level.
- **Usage limits.** Resume a stopped agent with `SendMessage`, saying exactly where it stopped. If
  it commits but sends no report, review from the commit and its log.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` are both empty, and `git worktree list` shows
     this checkout alone.
   - `git log --oneline 1c458d3..HEAD` touches no code. If it does, re-check §0's facts that it
     touches.
2. **The registry.** `./venv/bin/python -m RunRegistry list`: nothing `running`. At `1c458d3` there
   were 12 runs, 10 finished and 2 `unknown` (`a3-pilot`, `a3-pilot-resume`). Report any change,
   without acting on it. 02 launches nothing and opens nothing under `var/`.
3. **Baselines.** Measured 16:01–16:07 on 2026-10-02, at `2a44f7e`'s code (`1caecf8` and `1c458d3`
   change comments and records only): AdaptiveLevin 32, ComputeTargets 669, CosmologyModels 39,
   Datastore 551, LiouvilleGreen 148 (1 skipped), RunRegistry 199. Re-measure all six on `HEAD`,
   serially, and record the load average.
4. **The index.** `grep -c '^| \`\[' docs/OPEN_ISSUES.md`. It is **106** at `1c458d3`. After 02
   it is 104. Tell the agent both numbers.
5. **The pre-images**, for check 2: `git rev-parse HEAD:Datastore/SQL/ShardedPool.py`,
   `HEAD:Datastore/SQL/Datastore.py`, `HEAD:Datastore/tests/standin_pool.py` and
   `HEAD:Datastore/tests/test_reconcile_at_open.py`.

## 2. Dispatch

One fresh-context subagent, on **Opus**. Give it:
- the prompt, as amended;
- the campaign README, the audit, and this board;
- `HEAD`, the baselines and both index counts;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/02-write-the-version-row-through-the-recorded-write.md`,
  in README §5.1's form, with the prompt's §7 additions: the order of events in
  `ShardedPool.__init__` before and after, for a new store and an existing one, and the record's
  schema before and after.
- **The files it may add or change:**
  - `Datastore/SQL/ShardedPool.py` and `Datastore/SQL/Datastore.py`;
  - `Datastore/SQL/ObjectFactories/version.py`, only if W1 cannot work without it, as a deviation;
  - new test modules under `Datastore/tests/`;
  - `Datastore/tests/standin_pool.py`, **additions only**, each listed in the log;
  - `Datastore/tests/test_reconcile_at_open.py`: `TestPruningAfterRepair`'s class docstring, and
    nothing else;
  - the log; this board; the `datastore-integrity` board (`[02-the-version-row-…]` to its §4);
    `docs/OPEN_ISSUES.md`.

  **Nothing else.** No factory but `version.py` (in particular not `BackgroundModel.py`), no
  `main.py`, no driver, no `extract_*.py`, no probe and no committed probe output, nothing under
  `docs/` or `var/`, nothing under `orchestrator/`.
- **Stop and ask** on any of the prompt's §5 conditions. Say that §0's measurement is why W1 reads
  first, so that 38 checksum failures are not mistaken for a reason to stop.

## 3. The review — nine checks

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only §2's list.
   - `git diff HEAD~1 HEAD -- main.py 'extract_*.py' docs/ var/ Datastore/SQL/ObjectFactories/ ':!Datastore/SQL/ObjectFactories/version.py'`
     is empty.
   - `test_reconcile_at_open.py`'s diff is the one class docstring. `standin_pool.py`'s numstat
     shows no deletion, or each changed line is shown to leave every existing caller as it was.
2. **(C1) A clean open writes nothing; a new store writes once.**
   - Grep `Datastore.__init__`: no `object_get("version"` and no `version_serial`.
   - Read `_insert`: the refusal comes before `SerialLeaseManager`.
   - The 38 tests of §0 pass unchanged. Breakage (vii) is the converse: it puts the write back
     on a hit, and must fail `TestCleanStoreUntouched` (check 6).
   - The new store's test shows one `get` of `version` under a record, and an empty `version`
     table on every shard when that write is faulted `"before"` everywhere.
3. **(C2) The version row's repair.**
   - `_plan_replicated_repair`'s diff: empty, or a change for `version` alone with the test the
     prompt asks for.
   - The log states §0's timestamp-and-`store_id` fact for `version`, and how case 2's
     replica-only fault knew its controller.
4. **(C3) The prune.**
   - Read `ShardedPool.__init__`'s new order: the check at open, then
     `replication_in_flight.create`, then the prune, then the broker. Match it against the log's
     numbered list.
   - The record is committed before any shard is touched, each shard is its own transaction, and
     the record is cleared last.
   - The swallowed-failure case of §3 case 5: the record stays and the open raises.
   - `TestPruningAfterRepair` passes, both subtests of each test.
   - A store whose primary has the old `replication_in_flight` schema: the log says what happens,
     and nothing is pruned if the record cannot be written.
   - The comment above `_reconcile_replicated_tables` names the recorded prune as the one
     exception.
5. **The two kinds of record.** A `prune` record never reaches the `get`/`store`/`validate`
   licence, and a `get` record of `BackgroundModel` is repaired and not pruned (§3 case 6). Read
   both branches in the diff.
6. **(C4) The pins.** Replay each of the log's breakage diffs (i)–(vii) with `git apply`, and run
   the named test. It fails. Revert with `git apply -R`, and check `git status` is clean after each.
7. **The probe.** Run `version_window_probe.py` into the scratchpad.
   - Cases 2 and 3 open on the reopen.
   - Cases 1 and 4 print `returned (no crash happened)`.
   - The log says exactly that, and the committed `.txt` is unchanged.
8. **(C5) The records.**
   - `[02-the-version-row-…]` is in the `datastore-integrity` board's §4, and
     `[00-the-prune-at-open-…]` in this board's §4, each with a closing note naming the commit and
     the tests that show it healed.
   - Both index rows are gone. The count is 104 by counting, and matches the header.
   - `[00-a-new-stores-first-open-…]` has one line saying whether its window narrowed, and stays
     open. `[00-a-drop-action-…]` is untouched.
   - The log's *Observations* name the issues opened, if any.
9. **Suites.** All six serially, at `HEAD`.
   - Datastore is 551 plus the new tests; every other count is unchanged.
   - `black --check` is clean on every touched Python file.

## 4. After the review

- **Record it on this board**, after the prompt-01 review and in its form, with `HEAD` at
  dispatch, the checks and any remnants. Fix small residue in a follow-up commit of its own.
- **Report to the user:**
  - the order of events at open, before and after;
  - that both interrupted writes now heal, by which tests;
  - what the probe now shows, and that cases 1 and 4 are no longer a measurement;
  - which issues closed, and which stay open and why.
- **What 02 hands on.** 03 is unblocked: 01 and 02 have both landed. Its note must carry
  `set_version`, the `prune` operation, the read-first version lookup (a read-only pool finds the
  row the same way, and refuses when it is absent), and the old-schema behaviour of
  `replication_in_flight`.
