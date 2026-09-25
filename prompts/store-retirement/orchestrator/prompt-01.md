# Orchestrator — prompt 01, delete a closed store

Read [`../README.md`](../README.md) first. **You do not write code.** You may *replay* a mutation the
log records, with `git apply`, and then revert it.

**The prompt:** [`01-delete-a-closed-store.md`](../01-delete-a-closed-store.md)
**Board items:** R1–R3 · **Closes:** nothing

## 0. What makes this prompt unusual

It is the first code in the tree that deletes a file, and it is one resolver defect away from
deleting the live A3 store (audit §2.7). The review turns on four things.
- **Only the store's own files.** The legacy test, in which *B*'s records name *A*'s files by
  absolute path, must show *A* byte-identical. Mutation (i) must fail it, and its variant with the
  directory assertion removed must show *A*'s files deleted, on a fixture.
- **The primary last.** Mutation (ii) must fail the interruption tests.
- **`resume` relaxes one refusal.** Only the missing shard, and for both methods.
- **The charter text.** `CLAUDE.md:52` must read exactly as the board records D0, and no other
  comment, docstring or message may change except the `ShardedPool` block comment.

## 1. Before you dispatch

1. Record the branch and `HEAD`. `HEAD` must be this campaign's latest commit.
2. `python -m RunRegistry list`: nothing `running`.
3. **Snapshot, read-only,** everything under `var/datastores/` and `var/runs/` (size, mtime and
   SHA-256 of the stores and sidecars; size and mtime of run files; the listings). Put it in the
   session scratchpad under an `orch_` prefix. Reuse `orch_02_snapshot.py`.
4. The baselines: this board's §5, plus the prompt 02 review (RunRegistry 128).
5. `git status` is clean, apart from any untracked path that is not this campaign's. Tell the
   agent to leave such paths alone.

## 2. Dispatch

One fresh-context subagent (Opus). Give it the prompt, the campaign README, the audit, `HEAD` and
the baselines. Tell it plainly:
- **one commit**, with its log at `logs/01-delete-a-closed-store.md`;
- it holds `CLAUDE.md`'s amendment in D0's words, exactly, and this board's updates;
- run `black`;
- **never point `closed_store_files`, `delete_store`, a test or a script at anything under
  `var/`, or at a copy of anything under it.** This prompt has no real-store demonstration;
- mutations are recorded as diffs **that apply with plain `git apply` from the repository root**:
  fenced ```` ```diff ```` blocks, not indented, with correct hunk headers. Prompt 02's did not,
  and had to be replayed with `--recount --ignore-whitespace`. Mutations are never committed;
- do not touch `orch_*` files, or untracked paths that are not its own;
- the prompt's §5 stop conditions mean *stop and ask*.

## 3. The review — eight checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only `Datastore/SQL/ShardedPool.py`,
   `CLAUDE.md`, the new test module and any fixture extension, the log, the board, and the index
   only if an issue was opened. `git diff HEAD~1 HEAD -- RunRegistry/ tools/ docs/` is empty.
2. **The charter.** `CLAUDE.md:52` equals D0's wording on the board, character for character. The
   `ShardedPool` diff outside the two new methods and their helpers is the block comment alone.
   `_failure_message` is unchanged.
3. **One planning step.** Both methods go through one function that calls `_read_closed_store`.
   Nothing opens a stored record as a path (`Path(stored)`, `open(stored)` and the like). There is a
   separate check that every file is in the primary's directory. Every `os.unlink` is preceded by a
   regular-file, not-a-link check.
4. **Tests.** Run the new module twice. Test 2 asserts *A* byte-identical, with mtimes.
5. **Replay mutations (i), both variants, and (ii)** with plain `git apply`. Get the recorded
   results, then revert, leaving the tree clean.
6. **`var/`.** Re-take §1.3's snapshot. It must be identical.
7. **Suites.** Re-run all six. Datastore rises by exactly the tests added, and the others match.
8. **The board.** The §1 row for 01, R1–R3 and §5 are updated. `docs/OPEN_ISSUES.md` changed
   only if an issue was opened.
