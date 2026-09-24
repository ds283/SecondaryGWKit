# Orchestrator — prompt 02, copy and move a store

Read [`../README.md`](../README.md) first, especially **§6**, which records the user's decisions
that this prompt implements. **You do not write code.** That includes the change, the tests and the
mutations. You may *replay* a mutation the log records, with `git apply`, and revert it.

**The prompt:** [`02-copy-and-move-a-store.md`](../02-copy-and-move-a-store.md)
**Board items:** P5–P8 · **Closes:** `[01-whole-store-rename-is-unsupported]`
**Leaves alone:** `[store-sidecar-manifests-have-no-owner]` and prompt 03, which is held.

## 0. What makes this prompt unusual

Copying or moving five files and running one `UPDATE` is easy to write, and a version that works
on the happy path is easy to get. The review is about four things:

- **Interruption.** The prompt's P6 property says every state a crash can leave behind either opens
  against the right files or is refused. Read the log's interruption table against the code, row by
  row. A row whose claim is not backed by a test is a stop.
- **The source is never written.** In copy mode the source is opened `mode=ro` and only copied
  from. In either mode, **nothing is ever deleted and nothing is ever overwritten**, including a
  stale `-journal` at a destination name.
- **One definition each.** The shard naming rule exists once, used by the creator and the
  interface. Reading and checking a store's shard records exists once, shared with the constructor.
  A second copy of either is a stop.
- **Layering.** Nothing in `ShardedPool` or `tools/sharded_store.py` knows about sidecars or the
  registry (README §6.3). Code that moves, copies, refuses because of or warns about a
  `.manifest.json` is a stop, however helpful it looks.

## 1. Before you dispatch

1. **Confirm the branch and `HEAD`**, and record the SHA. HEAD must be the commit that adds this
   prompt, with prompt 01 (`b04671f`) beneath it. Tell the agent the tree is not its own.
2. `python -m RunRegistry list`: nothing `running` against `var/datastores/`.
3. **Baseline every suite.** At `b04671f` the counts were AdaptiveLevin 32, ComputeTargets 552
   (the one failure is the `test_wall_time_per_object` flake), CosmologyModels 39, Datastore 37,
   LiouvilleGreen 148 (1 skipped), RunRegistry 38. Re-measure; do not copy these.
4. **Snapshot the three stores** read-only (`sqlite3` `mode=ro`): mtimes, sizes, per-table row
   counts per shard, and the three primaries' SHA-256. **Keep the snapshot script and its output
   under a name the agent will not reuse.** The prompt 01 agent shared this session's scratchpad
   and overwrote the orchestrator's `store_state.py`. Prefix yours `orch_`, and tell the agent not
   to touch `orch_*` files.
5. **Free disk space**: at least 2 GB on the volume holding `var/` (`df -h .`). The demonstration
   needs about 0.7 GB, and the volume was 99% full on 2026-09-24.
6. `git status` clean, and nothing under `var/` except `datastores/`, `runs/` and the one log file.

## 2. Dispatch

One fresh-context subagent. Give it the prompt file, the campaign README, the SHA, the baselines,
and the files the prompt's "Read first" names. Nothing else.

Tell it plainly:

- **one commit**; the log goes at `logs/02-copy-and-move-a-store.md`;
- update the board per the prompt's §8, and `docs/OPEN_ISSUES.md` in the same commit;
- run `black` before committing;
- **never point `tools/sharded_store.py`, `ShardedPool` or `main.py` at an original store or the
  backup**; the script runs only on the hand-made copy and on what the script produces from it;
- **delete `var/portability-check-02/` afterwards**;
- the mutations are recorded in the log as diffs, exactly as applied, and never committed;
- do not touch `orch_*` files in the scratchpad;
- the stop conditions in §7 mean *stop and ask*.

The demonstration takes minutes, not hours, so it does not need the run registry.

## 3. The review — twelve checks

1. **Interruption table.** Every row names a test; every such test exists and passes. Check the
   cases the prompt requires: same directory with a new stem, new directory with the same stem, and
   new directory with a new stem, each with a legacy source whose records name a populated other
   directory.
2. **No overwrite, no delete, no source write.** Read the implementation for `os.remove`,
   `unlink`, `rmtree` and any write opened on a source path. There must be none. Every destination
   name and its three journal names are checked before the first write.
3. **The prescribed order**, or a logged justification for a different one that you find at least
   as strong. Copy writes the primary under a temporary name and `os.replace`s it last.
4. **One naming rule.** `git grep -n -e '-shard{' -- Datastore/ tools/` finds the pattern only in
   `Datastore/shard_paths.py` and in test modules' literal expected names. Before this prompt it
   also matched `ShardedPool.py:103` and `shard_store_fixtures.py:67`; both must now call the
   function. (A bare `'shard{'` also matches the actor names `shard{key:04d}-store`, which are not
   file names.) The two copies in `docs/handover/quadsource_atol_sweep.py` stay, and are listed in
   the log's observations.
5. **One read-and-check.** The resolver, `shard_file_problem` and the duplicate check are each
   called from one place for the constructor and the interface, not re-implemented. Prompt 01's 27
   tests are **unmodified**: `git diff HEAD~1 HEAD -- Datastore/tests/test_shard_paths.py
   Datastore/tests/test_shardedpool_shard_paths.py Datastore/tests/test_shard_key_audit_copy.py` is
   empty.
6. **Mutations reproduce.** Replay mutation (i) and one other from the log with `git apply`, run
   the named tests, confirm they fail, and revert. `git status` is clean afterwards.
7. **No Ray, no datastore in the tests.** Nothing opens anything under `var/`, calls the
   constructor, or initialises Ray. The script test checks that Ray is not initialised.
8. **Layering.** `git grep -n -i "manifest\|RunRegistry" -- Datastore/ tools/sharded_store.py`
   returns only the docstring and `--help` statements the prompt requires.
9. **The schema check.** The log says which tables in a primary and in a shard were inspected for
   a stored path, name or stem, and what it found. If anything records one, the agent should have
   stopped.
10. **The stores are untouched.** Re-take §1.4. Row counts, mtimes, sizes and the three primaries'
   hashes are identical. `var/portability-check-02/` is gone.
11. **The demonstration discriminates.** The inventory of `pcopy` and of `pmoved` is the
   original's counts with exactly one table one lower, the table the agent edited in the hand-made
   source. The audit tool attached `pcopy-shard0000.sqlite`.
12. **Suites** match their baselines, `Datastore/tests` is up by exactly the tests added, `black
   --check` is clean, the rename issue has moved to the board's §4, its row is gone from
   `docs/OPEN_ISSUES.md` §1.12, the count and date are corrected, and prompt 03's held row and the
   sidecar issue are unchanged. `git diff HEAD~1 HEAD -- docs/handover/ Datastore/SQL/Datastore.py
   main.py RunRegistry/` is empty.

## 4. Stop and ask the user

Relay verbatim; do not adjudicate.

- Any of §7 of the prompt.
- Any check in §3 fails. **Report it; do not repair it.**
- The agent proposes handling the sidecar, consulting the registry, deleting or cleaning up files,
  or rewriting rows on open.

## 5. After it lands

Report:

- the commit;
- where the naming rule and the shared read-and-check live;
- the interruption table, summarised, and anything in it you could not verify;
- the mutation record, and which you replayed;
- the demonstration's numbers, including the one-row discriminator;
- the store state before and after;
- the issues opened and closed;
- the suite counts.

Then stop. Prompt 03 stays held until the user has decided the registry's charter and the
sidecar's owner and format (README §6.4).
