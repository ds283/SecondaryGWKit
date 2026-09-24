# Orchestrator — prompt 03, the registry owns the store sidecar

Read [`../README.md`](../README.md) first, especially **§6.5**. It records the user's decisions,
which this prompt implements. **You do not write code.** That includes the change, the tests and
the mutations. You may *replay* a mutation the log records, with `git apply`, and revert it.

**The prompt:** [`03-the-registry-owns-the-store-sidecar.md`](../03-the-registry-owns-the-store-sidecar.md)
**Board items:** P9–P13 · **Closes:** `[store-sidecar-manifests-have-no-owner]`
**Leaves alone:** `[01-atol-sweep-check-expects-absolute-shard-records]`, every existing sidecar,
and every existing run manifest.

## 0. What makes this prompt unusual

A sidecar reader and writer is easy to write, and so is a copy that also writes a JSON file. The
review is about five things:

- **Unknown fields.** Every field the registry does not define survives every write,
  value-identical. A writer that rebuilds the object from known fields passes every test
  that only looks at known fields. Look for the test that uses the A3 sidecar's shape, and for
  the demonstration's probe of the real A3 sidecar.
- **Identity.** A copy gets a new `store_id`; a move keeps it; the source's sidecar is never
  written by a copy. The one-row discriminator shows which *store files* were read. The ids and
  histories show which *sidecar* went where.
- **In use.** The refusal covers **every** `running` run, stale ones included, and it matches by
  `results_store_id` as well as by path. A check that looks only at `alive` runs, or only at
  paths, is a stop.
- **Layering.** Nothing in `Datastore/` or `tools/` changes. `import RunRegistry` and
  `python -m RunRegistry list` stay free of `ray` and `sqlalchemy`.
- **Nothing existing is rewritten.** No original sidecar, no backup sidecar, no run manifest
  under `var/runs/`. The only adopt of real content is on copies under
  `var/portability-check-03/`.

## 1. Before you dispatch

1. **Confirm the branch and `HEAD`**, and record the SHA. `HEAD` must be the commit that adds
   this prompt. Beneath it must be `eca9a0b` (the decisions), `7a995b3` (prompt 02's review) and
   `2d1aabe` (prompt 02). Tell the agent the tree is not its own.
2. `python -m RunRegistry list`: nothing `running`.
3. **Baseline every suite.** At `2d1aabe` the counts were AdaptiveLevin 32, ComputeTargets 552
   (the one failure is the `test_wall_time_per_object` flake), CosmologyModels 39, Datastore 70,
   LiouvilleGreen 148 (1 skipped) and RunRegistry 38. Re-measure them; do not copy these.
4. **Snapshot**, read-only:
   - the three stores: mtimes, sizes, per-table row counts per shard, and the three primaries'
     SHA-256;
   - **the three sidecars**: SHA-256, mtime, size;
   - **`var/runs/`**: every entry, and the SHA-256 of every `manifest.json` and `status.json`.

   Keep the script and its output under an `orch_` prefix in the scratchpad, and tell the agent
   not to touch `orch_*` files. Prompt 02's `orch_store_state.py` already covers the stores and
   the sidecars. Extend it or add a second script for `var/runs/`.
5. **Free disk space**: at least 2 GB on the volume holding `var/` (`df -h .`).
6. `git status` clean. Nothing under `var/` except `datastores/`, `runs/`, `.DS_Store` and the one
   log file.

## 2. Dispatch

One fresh-context subagent. Give it the prompt file, the campaign README, the SHA, the baselines,
and the files the prompt's "Read first" names. Nothing else.

Tell it plainly:

- **one commit**; the log goes at `logs/03-the-registry-owns-the-store-sidecar.md`;
- update the board per the prompt's §8, and `docs/OPEN_ISSUES.md` in the same commit;
- run `black` before committing;
- **never point new code, `ShardedPool` or `main.py` at an original store, the backup, or any of
  their sidecars.** Never write under `var/runs/` or `var/datastores/`. Every `store copy` /
  `store move` and every throwaway `begin()` uses a root under `var/portability-check-03/`;
- **delete `var/portability-check-03/` afterwards**;
- the mutations are recorded in the log as diffs, exactly as applied, and never committed;
- do not touch `orch_*` files in the scratchpad;
- the stop conditions in §7 mean *stop and ask*.

The demonstration takes minutes, not hours, so it does not need the run registry's launch rules.

## 3. The review — thirteen checks

1. **Interruption table.** Every row names a test, and every such test exists and passes. It
   covers copy and move, and the move covers all three layouts. Check the property's two
   exclusions against the code:
   - no problem-free sidecar beside the wrong store, or with a missing history entry;
   - no two problem-free sidecars with one `store_id`.
2. **No delete, no overwrite, no source write.** Read `RunRegistry/stores.py` for `os.remove`,
   `unlink`, `rmtree` and `rmdir`; there must be none. A copy never opens the source's sidecar for
   writing. Every destination name is checked before the first write: the sidecar name, its
   `.tmp` and the `.incomplete-move` name. The only in-place updates are adopt's and the move's
   temporary sidecar.
3. **The prescribed order**, or a logged justification for another order that you find at least
   as strong. The store goes first and the sidecar last. The move updates its sidecar under the
   temporary name.
4. **One definition.** `git grep -n -F '.manifest.json' -- RunRegistry/ ':!RunRegistry/tests'`
   finds the pattern only in `sidecar_path` and in docstrings. The reader and the writer each
   exist once, and `begin()` calls the reader rather than parsing a sidecar itself.
5. **Layering.** `git diff HEAD~1 HEAD -- Datastore/ tools/ main.py docs/handover/ docs/gktk-remedial/ CLAUDE.md`
   is empty. In a child interpreter, `import RunRegistry` and `python -m RunRegistry list` leave
   neither `ray` nor `sqlalchemy` in `sys.modules`. Check this yourself.
6. **Existing tests unmodified.** `git diff HEAD~1 HEAD -- RunRegistry/tests/test_run_registry.py RunRegistry/tests/test_pipeline_adoption.py Datastore/tests/`
   is empty.
7. **Mutations reproduce.** Replay mutation (iii) (known fields only) and (iv) (alive runs only)
   from the log with `git apply`. Run the named tests, confirm that they fail, and revert.
   `git status` is clean afterwards.
8. **No Ray, no `var/` in the tests.** Nothing opens anything under `var/`, calls the
   `ShardedPool` constructor, or initialises Ray. The runs root and every store are in temporary
   directories.
9. **Unknown fields.** A test takes an A3-shaped fixture through adopt, copy and move, and
   compares every unknown field after each step. In the demonstration, the probe's unknown fields
   equal the real A3 sidecar's; re-derive this from the log's record, not from its summary.
10. **In use.** The tests cover four cases: a stale `running` run refuses; a match by
    `results_store_id` alone refuses; a finished run does not refuse; and a manifest without
    `results_store_id` is tolerated. In the demonstration, the throwaway run's manifest carried
    the source's `store_id`, and the refusal named it as stale.
11. **The stores, sidecars and runs are untouched.** Re-take §1.4. Row counts, mtimes, sizes and
    the three primaries' hashes are identical, the three sidecars are identical, and `var/runs/`
    is identical, entry for entry and hash for hash. `var/portability-check-03/` is gone.
12. **The demonstration discriminates.** The inventories of `pcopy` and `pmoved` are the
    original's counts with exactly one table one lower. `pcopy`'s `store_id` differs from the
    adopted source's; `pmoved`'s equals `pcopy`'s. The histories have two entries and then three.
    `dst/` is empty after the move.
13. **Suites and the board.** Every suite matches its baseline. `RunRegistry/tests` is up by
    exactly the tests added, and `black --check` is clean. On the board:
    - the sidecar issue has moved to §4, and its §1.12 row in `docs/OPEN_ISSUES.md` is gone;
    - the `prepare()` issue is open in §3, with its index row;
    - the count is net unchanged and the date is correct;
    - `[01-atol-sweep-check-expects-absolute-shard-records]` is unchanged.

## 4. Stop and ask the user

Relay verbatim; do not adjudicate.

- Any of §7 of the prompt.
- Any check in §3 fails. **Report it; do not repair it.**
- The agent proposes adopting or rewriting an original sidecar, the backup's, or a run manifest;
  having a driver write sidecars; or having the registry delete, clean up, lock or finish runs.

## 5. After it lands

Report:

- the commit;
- the sidecar format as shipped, and every *prompt's choice* the agent changed;
- the interruption table, summarised, and anything in it you could not verify;
- the mutation record, and which mutations you replayed;
- the demonstration's numbers: the discriminator, the ids and the histories;
- the state of the stores, sidecars and runs, before and after;
- the issues opened and closed;
- the suite counts.

Then stop. That closes the campaign's written prompts. Whether anything follows, such as making
the pipeline drivers create sidecars, is the user's call.
