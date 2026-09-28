# Orchestrator — prompt 02, repair or refuse at open

Read [`../README.md`](../README.md) first: §0.1, §2, §4 and §5, and §6.1. **You do not write
code.** You may *replay* a mutation the log records, with `git apply`, and then revert it.

**The prompt:** [`02-repair-or-refuse-at-open.md`](../02-repair-or-refuse-at-open.md)
**Board items:** W5–W7 · **Closes:** `[00-replicated-writes-can-diverge-across-shards]`
(store-fingerprint board) · **Gate:** 01 landed and reviewed (board §1). No other campaign has an
uncommitted change to `Datastore/SQL/ShardedPool.py` (README §5 rule 13). Never dispatched
concurrently with 03–09 in this checkout.

## 0. What makes this prompt unusual

- **It runs on every open of every store.** `main.py` and all six `extract_*.py` scripts open
  stores through `ShardedPool(...)`, so the check runs before every build, every resume and every
  extraction. Two things follow.
  - **A clean store must open exactly as before, writing nothing.** Test 7 is the main defence.
  - **Every repair writes to a store.** A repair that copies the wrong shard's rows forward is
    worse than the divergence it removes. The review reads the repair and refusal logic against
    the code, not against the log's prose.
- **Repair has one licence: a set record, and only its controller.** Audit §4.5's argument holds
  only for an interrupted replication. With a record set, the replicas can only lack what the
  controller committed. Anything else refuses: no record, a row the controller lacks, a serial
  split, a record naming a class other than the one that differs, or a sharded table. And nothing
  may delete a row (§5).
- **Log 01's commit-point table is the input.** Every state it lists with the record set is
  either "same" or "some shards lack what the controller holds". The one exception is after a
  `ReplicationMismatch`, where the state is whatever difference it names. 02's reopen table
  (acceptance item 3) must give each row of log 01's "After" table one outcome: repaired (from
  where), cleared, or refused.
- **One existing test changes, by the user's decision (2026-09-28).**
  `Datastore/tests/test_replicated_write.py`'s `TestWriteOverASetRecord.test_a_reopened_pool_refuses`
  pins 01's behaviour. Its fixture is a tolerance get whose replica died before writing, which
  leaves the controller holding a row one replica lacks, with the record set. Reopened, the pool
  refuses every replicated write. Log 01 handed that state to 02, which repairs it at open, so the
  test fails by design.
  - Under README §5 rule 7, this note names it: **rewrite it to 02's rule on the same fixture.**
    Reopened, the pool copies the tolerance row forward from the record's controller and clears
    the record, and the write that was refused now succeeds.
  - Rename it to say so, and keep `TestWriteOverASetRecord`'s other tests unchanged. They refuse
    without reopening, and still must.
  - This is the one permitted edit to an existing test. Any other existing test that fails is a
    §5 stop. The orchestrator found no other test that reopens a pool over a set record or
    diverged shards; the agent confirms that.

**Carried from this session (the 05 reviews).**
- The stand-in pool is `Datastore/tests/standin_pool.py` (`StandinCluster`, `open_pool`,
  `close_pool`, `fault(…, "before" | "after")`, `shard_snapshot`, `in_flight_records`). The
  exceptions live in `Datastore/replication.py`. 02 imports both and copies neither.
  - Adding to `standin_pool.py` is allowed.
  - Changing an existing function of it is allowed only if every existing test still passes
    unmodified, and the change is logged.
- `ReplicatedDivergence` belongs in `Datastore/replication.py`, beside `ReplicationMismatch` and
  `ReplicationInFlight`. `TestReplicatedDivergence` in `test_store_inventory.py` is an unrelated
  class of that name.
- A mutation record lists **every** failing test, with counts, not only the ones the prompt
  names.
- The replay is done twice:
  - **before the commit**, from the final log file, and said to be so;
  - **after the commit**, from `git show HEAD:<log>`, and reported, with no amend.
- Every change is in the log, including a removal of unused code.
- The agent works in **its own scratch subdirectory**, not the session scratchpad's top level,
  where the orchestrator's `orch_*` scripts live. (05c's agent overwrote one.)

## 1. Before you dispatch

1. Record the branch and `HEAD`. It must be the commit that adds this note.
2. `python -m RunRegistry list`: nothing `running`.
3. Snapshot `var/datastores/`, `var/runs/` and the root `physics-test-n20-lambdacdm-zend0p1*.sqlite`
   read-only, with `orch_04_snapshot.py` in the session scratchpad. Write it again if it is
   missing; never put it in the tree. The 05 session's snapshot held 46 files and 10 directories.
4. Baselines, board §5.6: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39, Datastore 287,
   LiouvilleGreen 148 (1 skipped), RunRegistry 199. Re-measure all six on `HEAD`, serially.
5. `git status --short` and `git diff --cached` are both empty.

## 2. Dispatch

One fresh-context subagent (**Opus**, as the prompt recommends). Give it:
- the prompt;
- the README;
- the audit, `docs/datastore-integrity-audit.md`, §4 in full;
- log 01;
- the board's review of 01 and its "Carried to prompt 02's orchestrator notes";
- `HEAD` and the baselines;
- §0 of this note.

Tell it plainly:
- **one commit**, with its log at `logs/02-repair-or-refuse-at-open.md`, in README §5.1's form,
  with prompt 02 §7's additions:
  - the comparison's classes, keys and exclusions;
  - the reopen table;
  - the measured open time;
  - a repair's and a refusal's output, as a user would see them;
- in the same commit:
  - the board: its §1 row, W5–W7, §3.1 and a new §5 subsection;
  - the store-fingerprint board, where the issue moves to §4 with a closing note naming this
    commit and 01's by subject;
  - `docs/OPEN_ISSUES.md`, with the row deleted, the count 101 → 100 and the date;
- **the files it may change:**
  - `Datastore/SQL/ShardedPool.py`: the constructor, and the new methods it calls;
  - `Datastore/replication.py`;
  - the new test module, and additions to `standin_pool.py`;
  - the one named test in `test_replicated_write.py`;
  - the log, the two boards and the index.

  **Nothing else:**
  - not the write path (`_replicated_write` and the three `_…_impl_replicated_table` methods);
  - not the closed-store block (`closed_store_files`, `delete_store`, `copy_store`, `move_store`
    and their helpers);
  - not `Datastore.py`'s `prune_unvalidated`, `drop_actions` or `_drop_order`;
  - no factory, no `main.py`, no `extract_*.py`, no `store_inventory.py`, no `RunRegistry/`;
  - no audit probe;
  - nothing in `orchestrator/`.

  If `validated`'s recomputation needs the factory's own rule and the agent cannot call it as it
  stands, **stop and ask**. Do not refactor the factory;
- **no Ray, and nothing opens anything under `var/`**, or a copy of anything under it. Every store
  is a stand-in in a temporary directory;
- every mutation diff is pasted from `git diff`, and replayed twice as §0 says;
- run `black`;
- the prompt's §5 stop conditions mean *stop and ask*, and so does any existing test other than
  the named one failing.

## 3. The review

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only the permitted files.
   - `git diff HEAD~1 HEAD -- main.py extract_*.py Datastore/SQL/ObjectFactories/ Datastore/SQL/Datastore.py Datastore/store_inventory.py RunRegistry/ prompts/datastore-integrity/orchestrator/`
     is empty. `docs/` changes only in `docs/OPEN_ISSUES.md`.
   - `--diff-filter=M` over `*/tests/*` shows only `test_replicated_write.py`, where one test
     changes, and `standin_pool.py`.
   - No `ShardedPool.py` hunk falls in the write path or the closed-store block. Read every hunk
     and say where it is.
2. **Before any actor, no Ray.**
   - Read the constructor. `_reconcile_replicated_tables()` runs for an existing store after
     `_check_shard_files` and before the broker or any actor is created, and needs no Ray.
   - A new store skips it.
   - README §4's three names exist as fixed: `_reconcile_replicated_tables`,
     `ReplicatedDivergence` and `ShardedPool.reconciliation`.
3. **The comparison is W5's, exactly.**
   - The classes come from `config.sharding.replicated_tables`, not a hand-written list. Tag
     tables are included.
   - Rows are keyed by serial, with full rows compared and only `timestamp` excluded.
     `BackgroundModelValue` is keyed by `(model_serial, z_serial)`, with `serial` excluded. Tag
     tables are keyed by their composite key.
   - Every primary `shard_keys.key_serial` is checked to be a wavenumber serial present on every
     shard.
   - No sharded table is read.
4. **Repair only under the licence.** Read the code, then check each condition against a test:
   - it happens only with a record set;
   - it copies only from the record's `controller_shard`, and only the record's class;
   - rows are copied verbatim, serial and timestamp included, and a `BackgroundModel` as a unit:
     row, tags and values;
   - flags are only ever turned on;
   - after a `validate` record, `validated` is recomputed by the factory's own rule on each
     shard, and a disagreement then refuses;
   - there is a second full comparison before the record is cleared, in its own commit, last;
   - the new code deletes nothing except the record's own row in `replication_in_flight`, and
     sets no flag to `0`/`False`. `grep` it for `DELETE`, `.delete(` and `UPDATE`.
5. **Refusal otherwise.** Tests 2–4 cover:
   - a serial split, with and without a record;
   - each no-record difference;
   - a row the controller lacks.

   Each asserts no shard file changed (checksums), and the message names the class, the shard and
   the serial, and says nothing was repaired. A record naming another class refuses.
6. **The reopen table.** Walk each row of log 01's "After" table against 02's table and against
   test 1's cases: every point, every path. Rows the table gives as "same, record set" are
   cleared. "Controller holds more" is repaired from the controller. A `ReplicationMismatch`
   state that is a serial split is refused. A second reopen repairs nothing.
7. **The named test.**
   - The diff of `test_replicated_write.py` changes that one test only.
   - Its fixture is unchanged.
   - It now asserts the repair, the cleared record and a write that succeeds.
   - Its siblings are byte-identical.
8. **A clean store is untouched, and the open is cheap.**
   - Test 7 checks every shard file and the primary, by checksum, before and after opening a clean
     store.
   - The measured open time for 1740 redshifts, a 1740-value model, and 4 and 20 shards is in the
     log. Say whether it is small beside a pool open under Ray. If it is not, raise it with the
     user; it is paid on every extraction.
9. **Callers.** List every `ShardedPool(` caller by `grep`: `main.py`, the six `extract_*.py`,
   and the `docs/` scripts. For each, state that a clean store opens as before, and that a
   divergent one now refuses or repairs at open. Say on the board that extraction now runs the
   check. Do **not** open a store under `var/` to see what it would do. The audit's census (§4.4)
   found all three clean serial by serial. If you think a real store would refuse, raise it with
   the user.
10. **Replay the mutations** (i)–(v) from the committed log with plain `git apply`. Each gives its
    logged failures. Revert each, leaving the tree clean.
11. **`var/`, suites, boards.**
    - The snapshot is identical.
    - Run all six suites serially. Datastore rises by exactly the tests added, and the others
      match §1.4.
    - `black --check` is clean.
    - `ray.is_initialized()` is false after the new module.
    - The issue is on the store-fingerprint board's §4, with a closing note naming both commits.
    - The index row is deleted, the count is 100 (rows counted), and the date is right.
    - The board's §1 row, W5–W7, §3.1 and the new §5 subsection are present.

## 4. After the review

Record it on the board, after the review of 05c. Then:
- **03 is next**, with its own orchestrator note carrying this review's findings. 03 changes
  lookups on the classes 02 reconciles.
- Log 01's observation 2 ("until 02 lands, a set record refuses every replicated write") is
  discharged. Say so on the board.
- Prompt 10 remains the first real run. Nothing in 02 is exercised under Ray until then.
