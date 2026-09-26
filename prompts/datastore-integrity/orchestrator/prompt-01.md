# Orchestrator — prompt 01, the checked replicated write

Read [`../README.md`](../README.md) first: §0, §4, §5 and §6.1. **You do not write code.** You may
*replay* a mutation the log records, with `git apply`, and then revert it.

**The prompt:** [`01-checked-replicated-write.md`](../01-checked-replicated-write.md)
**Board items:** W1–W4 · **Closes:** nothing; **narrows**
`[00-replicated-writes-can-diverge-across-shards]` (store-fingerprint board) · **Gate:** none of the
§6.2 decisions; no other campaign's change to `ShardedPool.py` uncommitted.

## 0. What makes this prompt unusual

It is the first change this campaign makes to code production runs: every replicated object the
pipeline writes goes through it, and it is tested only on stand-in shards, never under Ray (README
§5 rule 9; prompt 10 is the first real run). The review turns on four things.
- **No silent window.** The log's commit-point table (acceptance item 3) must have no row reading
  "shards differ, no record". Read every row against the code, not against the log's prose. The
  one window the prompt warns about is a controller commit that precedes the record (W1); if the
  agent took the "write the record only when a replication will follow" choice, check that the
  record is still committed **before** the controller's call is submitted.
- **Loud, and left for prompt 02.** A `ReplicationMismatch` leaves the record set. A write that
  finds a record already set refuses and writes nothing. Mutations (i) and (iii) must fail.
- **Row-identical copies.** After a clean write, the tests compare every column of every copy,
  timestamps and `BackgroundModelValue` serials included. A test that compares only keys would pass
  mutation (iv); check that (iv) fails.
- **Sharded writes unchanged.** The timestamp override is per call, through arguments, and absent
  for every sharded write. A diff that stamps sharded rows from the driver is out of scope.

**Carried from this campaign's writing, and from `store-retirement`.**
- The stand-in shard pattern is `docs/datastore-integrity-audit/replicated_write_fault_probe.py`.
  Prompts 02 and 06 reuse it. Tell the agent to put the stand-in pool in a **reusable test helper
  module** under `Datastore/tests/`, not inside its test module, so 02 and 06 need not copy it.
  That is an implementation choice within the prompt's latitude; it must say so in the log.
- The probes under `docs/datastore-integrity-audit/` are the audit's record. The agent may run and
  read them, and must not edit them.
- While this campaign was written, another session had changes **staged** in this checkout.
  `git status --short` shows staged changes in its first column; check `git diff --cached` is
  empty as well as the working tree.

## 1. Before you dispatch

1. Record the branch and `HEAD`. `HEAD` must be this campaign's latest commit, the one that added
   these notes.
2. `python -m RunRegistry list`: nothing `running`.
3. **Snapshot, read-only,** everything under `var/datastores/` and `var/runs/`, and the repository
   root's `physics-test-n20-lambdacdm-zend0p1*.sqlite`: size, mtime and SHA-256 for stores and
   sidecars, size and mtime for run files, and the listings. The live A3 store and its sidecar
   matter most; the sweep and backup names now hold tombstones. Write the script in the session
   scratchpad as `orch_01_snapshot.py`, never in the tree.
4. The baselines, at `cdf097f` (no code has changed since `store-retirement` closed): AdaptiveLevin
   32, ComputeTargets 552 (`test_tk_wkb_phase.TestCost.test_wall_time_per_object` is a known
   wall-clock flake), CosmologyModels 39, Datastore 206, LiouvilleGreen 148 (1 skipped),
   RunRegistry 199. Re-measure all six on `HEAD`, in the checkout itself, with README §7's command.
5. `git status --short` is empty and `git diff --cached` is empty.

## 2. Dispatch

One fresh-context subagent (**Opus**, as the prompt recommends). Give it:
- the prompt;
- the campaign README;
- the audit, `docs/datastore-integrity-audit.md`;
- `HEAD` and the baselines.

Tell it plainly:
- **one commit**, with its log at `logs/01-checked-replicated-write.md`;
- in the same commit: this board (§1 row, W1–W4, §5), the store-fingerprint board's entry for
  `[00-replicated-writes-can-diverge-across-shards]` **narrowed** (not closed), and that issue's
  row in `docs/OPEN_ISSUES.md` §1.15 with the date;
- run `black`;
- **no test starts Ray, and nothing opens anything under `var/`**, or a copy of anything under it.
  The new `replication_in_flight` table is created when an existing store is opened: that is one
  more reason never to open a real store here;
- the stand-in pool goes in a reusable helper module (§0);
- **these are not changed:** any factory's `build()`; any lookup; any sharded write; the `.scalar()`
  lookups (prompt 03); `delete_store`, `copy_store`, `move_store` and the closed-store block;
  `Datastore/store_inventory.py`; `RunRegistry/`; every existing test;
- mutations (i)–(v) are recorded as diffs **that apply with plain `git apply` from the repository
  root**: fenced ```` ```diff ```` blocks, not indented, with correct hunk headers, each checked
  with `git apply --check` against the final commit. Mutations are never committed;
- do not touch `orch_*` files;
- the prompt's §5 stop conditions mean *stop and ask*.

## 3. The review — nine checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only:
   - `Datastore/SQL/ShardedPool.py` and `Datastore/SQL/Datastore.py`;
   - `Datastore/SQL/ObjectFactories/wavenumber.py` and `BackgroundModel.py`, in their `store()`
     paths only;
   - the new test module and helper module under `Datastore/tests/`;
   - wherever `ReplicationMismatch` lives, if that is a new file;
   - the log, this board, the store-fingerprint board and the index.

   `git diff HEAD~1 HEAD --diff-filter=M -- '*/tests/*'` is empty. In the two factories no hunk
   falls in `build()`, `read_batch` or `inventory_records`. In `ShardedPool.py` no hunk falls in the
   closed-store block (`closed_store_files`, `delete_store`, `copy_store`, `move_store` and their
   helpers).
2. **One helper.** The three `_…_impl_replicated_table` methods call one helper; only it writes and
   clears `replication_in_flight`. `grep` for the table name outside it and its creation.
3. **The commit-point table.** Walk each replicated path in the code and confirm each row. No row
   says "shards differ, no record". The record is committed before the controller's call is
   submitted, in its own transaction on the pool's engine.
4. **Checked answers.** Scalar and vector get, store and validate each compare every replica's
   answer with the controller's before the record is cleared. A mismatch raises
   `ReplicationMismatch` naming shard, class and both serials, and leaves the record set.
5. **Replica stores.** `wavenumber_exit_time` and `BackgroundModel` replicas insert-or-verify by
   serial, with the three cases of W3 each tested. `BackgroundModel` value rows carry the
   controller's serials. The controller's own store writes what it wrote before.
6. **One timestamp.** `Datastore._insert`'s override is a call argument, not actor state; with no
   override it calls `datetime.now()` as before. A sharded store's timestamp is still its own.
7. **Tests.** Run the new module twice. `ray.is_initialized()` is false after it. Grep: no test
   path reaches `var/`.
8. **Replay mutations (i)–(v)** with plain `git apply`. Each gives its recorded failures. Revert
   each, leaving the tree clean.
9. **`var/`, suites and the boards.**
   - Re-take §1.3's snapshot. It must be identical, including the live A3 primary's SHA-256.
   - Re-run all six suites. Datastore rises by exactly the tests added; the others match §1.4,
     RunRegistry included (its golden fingerprint must not move). `black --check` is clean.
   - This board: the §1 row for 01, W1–W4, §5. The store-fingerprint entry is narrowed, not
     closed, and says what this prompt made loud. The index row changed with it, and the count is
     unchanged.

**After the review.** Record it on this board, in the same style as `store-retirement`'s reviews.
Prompt 02 is next; its orchestrator notes are written then, carrying anything this review found —
in particular where the helper module and `ReplicationMismatch` live, and the commit-point table,
which 02's reopen table is built on.
