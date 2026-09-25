# Orchestrator — prompt 02, the sweep prepares through the registry

Read [`../README.md`](../README.md) first. **You do not write code.** You may *replay* a mutation the
log records, with `git apply`, and then revert it.

**The prompt:** [`02-the-sweep-prepares-through-the-registry.md`](../02-the-sweep-prepares-through-the-registry.md)
**Board items:** R4–R5 · **Closes:** `datastore-portability`'s
`[03-atol-sweep-prepare-writes-its-store-sidecar-by-hand]` and
`[01-atol-sweep-check-expects-absolute-shard-records]`

## 0. What makes this prompt unusual

The script is the record of a measurement, so the review turns on how small the edit is, and on
two behaviours.
- **The tombstone survives.** A sidecar alone at the sweep name makes `prepare()` refuse, and
  leaves that sidecar byte-identical, with and without `force`. This is test 2, and mutation (i)
  must fail it.
- **Modern stores pass the check.** A bare-name store passes `assert_store_is_self_consistent`,
  so `--build --resume` works. The three real primaries, all legacy-absolute, still pass it.

## 1. Before you dispatch

1. Record the branch and `HEAD`. `HEAD` must be this campaign's latest commit.
2. `python -m RunRegistry list`: nothing `running`.
3. **Snapshot, read-only,** the three primaries and their sidecars (size, mtime, SHA-256), and the
   listings of `var/datastores/`, its backup directory and `var/runs/`. Put it in the session
   scratchpad under an `orch_` prefix.
4. The baselines: README §7. No code has changed since `42d4910`.
5. `git status` is clean.

## 2. Dispatch

One fresh-context subagent. Give it the prompt, the campaign README, the audit, `HEAD` and the
baselines. Tell it plainly:
- **one commit**, with its log at `logs/02-the-sweep-prepares-through-the-registry.md`;
- update this board, `datastore-portability`'s board and `docs/OPEN_ISSUES.md` in the same commit;
- run `black`;
- **never call `prepare()`, `copy_store`, `--build` or any writer against a real store or name.**
  The one permitted contact with `var/` is the read-only check of §3 of the prompt;
- **never begin a run under `var/runs/`**;
- mutations are recorded as diffs, exactly as applied, and never committed;
- do not touch `orch_*` files;
- the prompt's §5 stop conditions mean *stop and ask*.

## 3. The review — eight checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only the script, the new test module, the log,
   the two boards and the index. In the script's diff, every hunk is in `prepare()`,
   `assert_store_is_self_consistent`, `--force`'s handling or help, or their docstrings and
   messages.
2. **One `copy_store` call.** `prepare()` has no `shutil.copy2`, no `UPDATE` and no `write_text`,
   and passes the old purpose text verbatim.
3. **The check.** It compares serial by serial through `resolve_shard_path`, opens the primary
   `mode=ro`, and requires exactly the serials `0 … SHARDS-1`. Its refusal gives no advice that
   is wrong for any of its three callers.
4. **Tests.** Run the new module, twice. Confirm that test 2 asserts byte-identity of the sidecar,
   with and without `force`, and that nothing in the module resolves a path under `var/`.
5. **Replay mutation (i) and (ii)** from the log with `git apply`. Get the recorded failures, then
   revert, leaving the tree clean.
6. **The real primaries.** Re-take §1.3's snapshot. It must be identical: every size, mtime,
   SHA-256 and listing, `var/runs/` included.
7. **Suites.** Re-run every suite. Each matches its baseline, and RunRegistry rises by exactly the
   tests added. `test_store_fingerprint` test 12 passes.
8. **Boards and index.** Both issues are in `datastore-portability`'s §4 with closing notes naming
   the commit, and `[01-…]`'s note records the understated impact. Their rows are gone from
   `docs/OPEN_ISSUES.md` §1.14, and the count (102 → 100) and date are right. This board's row
   for 02, R4–R5 and §5 are updated.
