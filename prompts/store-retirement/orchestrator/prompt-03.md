# Orchestrator — prompt 03, retire a store

Read [`../README.md`](../README.md) first, §6.2 in full. **You do not write code.** You may
*replay* a mutation the log records, with `git apply`, and then revert it.

**The prompt:** [`03-retire-a-store.md`](../03-retire-a-store.md)
**Board items:** R6–R8 · **Closes:** nothing · **Gate:** prompt 01 has landed (`ff65f9f`,
reviewed in `39fd66f`)

## 0. What makes this prompt unusual

It is the campaign's one irreversible operation, and the first caller of prompt 01's deletion. The
tombstone is the only record left when the files go, so the review turns on the order of the
writes and on what the sidecar says at every point.
- **The tombstone before the deletion.** `retired`, with `files` and `references`, is on disk in
  state `retiring` before any `os.unlink`. Mutation (i) must fail the tests.
- **Complete only when nothing is left.** State `retired` is written after the check that no
  listed file exists. Mutation (ix) must fail the tests.
- **The fingerprint condition is exact.** A fresh read-only fingerprint must compare empty,
  `problems` included, and `--without-fingerprint` must refuse a store that *can* be
  fingerprinted. Mutations (ii), (iii) and (vi).
- **`ok` is false for a tombstone, and no existing caller changes.** Existing tests pass
  unmodified, and every non-test use of `SidecarReading.ok` refuses a tombstone.
- **Nothing else is written.** The references search reads other sidecars and run manifests, and
  never writes one. `ShardedPool` is not changed.

## 1. Before you dispatch

1. Record the branch and `HEAD`. `HEAD` must be this campaign's latest commit.
2. `python -m RunRegistry list`: nothing `running`.
3. **Snapshot, read-only,** everything under `var/datastores/` and `var/runs/` (size, mtime and
   SHA-256 of the stores and sidecars; size and mtime of run files; the listings), **and** the
   repository root's pre-registry store, `physics-test-n20-lambdacdm-zend0p1*.sqlite`, which the
   prompt 01 review found. Put it in the session scratchpad under an `orch_` prefix. Reuse
   `orch_02_snapshot.py` if this session still has it; otherwise write it again there, never in
   the tree.
4. The baselines, from the board's §5 after prompt 01: AdaptiveLevin 32, ComputeTargets 552,
   CosmologyModels 39, Datastore 206, LiouvilleGreen 148 (1 skipped), RunRegistry 128. Re-measure
   them on `HEAD` in the checkout itself, **not** in an exported copy: the git-dependent tests fail
   spuriously outside a repository, which cost prompt 01's first dispatch its baselines.
5. `git status` is clean, apart from untracked paths that are not this campaign's (today
   `docs/datastore-integrity-audit*` and `prompts/datastore-integrity/`). Tell the agent to leave
   them alone.

## 2. Dispatch

One fresh-context subagent (**Opus**). Give it the prompt, the campaign README, the audit, log 01,
`HEAD` and the baselines. Tell it plainly:
- **one commit**, with its log at `logs/03-retire-a-store.md`;
- update this board, and `docs/OPEN_ISSUES.md` only if it opens an issue, in the same commit;
- run `black`;
- **never call `retire_store`, `store retire` (even `--dry-run`), `fingerprint_store` or any
  writer against anything under `var/`, or a copy of it.** This prompt has no real-store contact
  at all; prompt 05 has it;
- **every test passes `runs_root` and `stores_root` explicitly**, and every command-line test
  passes `--runs-root` and `--stores-root`. The defaults are `var/runs/` and `var/datastores/`,
  so a call that omits one reads the real sidecars and manifests;
- `ShardedPool`, `tools/sharded_store.py`, `CLAUDE.md` and every existing test are not changed.
  If §3 cannot be served by prompt 01's methods as shipped, that is a §7 stop, not a fix;
- mutations (i)–(ix) are recorded as diffs **that apply with plain `git apply` from the repository
  root**: fenced ```` ```diff ```` blocks, not indented, with correct hunk headers. Mutations are
  never committed;
- do not touch `orch_*` files, or untracked paths that are not its own;
- the prompt's §7 stop conditions mean *stop and ask*.

## 3. The review — ten checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only `RunRegistry/stores.py`,
   `RunRegistry/__init__.py`, `RunRegistry/__main__.py`, the new test module, any addition to
   `RunRegistry/tests/store_fixtures.py`, the log, the board, and the index only if an issue was
   opened. `git diff HEAD~1 HEAD -- Datastore/ tools/ docs/ CLAUDE.md` is empty. In
   `__init__.py`, every hunk is inside `begin`. `git diff HEAD~1 HEAD --diff-filter=M --
   RunRegistry/tests/` is empty, or is `store_fixtures.py` with additions only.
2. **The order of the writes.** Read `retire_store`. Step 1 is one `_update_sidecar` call adding
   `retired` in state `retiring`, with `files` and `references`, and appending the one `retire`
   entry. Then `ShardedPool.delete_store`, with `resume=True` only on the completion path or for
   a missing shard under the flag. Then the check that no listed file exists. Then the second
   `_update_sidecar`, which appends no history. Nothing between them catches and retries or cleans
   up. `retired.files` is `closed_store_files(primary, resume=…)` with the same `resume` as the
   deletion. `_update_sidecar`'s docstring, which counts its callers, is still true.
3. **The fingerprint.** Without the flag, a fresh `fingerprint_store(…, write=False)` is compared
   with the recorded one, and any difference, `problems` included, refuses. With the flag, a
   fingerprint that succeeds refuses, and one that fails is kept verbatim. A hot journal and an
   unreadable `shards` table are refused under both. Nothing falls back to `shard_file_name` or
   any naming rule: grep the diff.
4. **The reader and the history.** A completed tombstone has empty `problems`, `ok` false and
   `store_id` `None`. `retiring` and a reappeared primary are problems, and name what remains and
   the remedy. A legacy sidecar with a `retired` key reads as before. `_history_problems` accepts
   `retire`'s `to` rule for `retire` only: confirm that a test gives a copy or move entry
   `retire`'s `to` value and finds a problem. If none does, record it as a finding. List every
   non-test use of `.ok` (today `RunRegistry/__init__.py:355` and `RunRegistry/stores.py:173`,
   `:649`, `:1156`, before the diff moves them) and confirm each does the right thing for a
   tombstone.
5. **The guards.** In `begin`, the retired-store check is above `os.makedirs`. Copy and move from
   and to, fingerprint, adopt and create each refuse a tombstone with the tombstone message, and a
   test covers each. The completion path re-checks running runs by the recorded `store_id` field,
   not `SidecarReading.store_id`. `retire_store` has no caller outside `stores.py`, `__main__.py`
   and the tests: D0 says a person runs it.
6. **The references and the docstrings.** The sidecar search walks `stores_root` recursively,
   skips the store's own, lists an unreadable one without stopping, and opens nothing for writing.
   The report names the roots searched and what was not. The `stores.py` and `__main__.py`
   docstrings are quoted before and after in the log, say that `retire` alone deletes, and only a
   store's own files, through `ShardedPool.delete_store`, and carry `retire` in both operation
   lists and the format table.
7. **Tests.** Run the new module twice. Confirm test 1 checks the fingerprint field
   value-identical and the other store, sidecars and runs root unchanged by `tree_state`. Confirm
   every row of the log's interruption table names the test behind it. Check that the table says
   what a `.tmp` sidecar left by a killed write does. `_update_sidecar` refuses while one exists,
   and a monkeypatched failure leaves none. If the table is silent on it, record that as a finding,
   not a failure. Run `python -c "import RunRegistry, sys; print('ray' in sys.modules,
   'sqlalchemy' in sys.modules)"` yourself: it must print `False False`.
8. **Replay mutations (i)–(ix)** with plain `git apply`, from an empty working directory in the
   scratchpad, with the repository on `PYTHONPATH`. Each gives its recorded failures. Revert each,
   leaving the tree clean and the working directory empty.
9. **`var/` and the repository root.** Re-take §1.3's snapshot. It must be identical, the
   `physics-test-n20-*` store's mtimes included.
10. **Suites and the board.** Re-run all six. RunRegistry rises by exactly the tests added, and
    the others match §1.4. `black --check` is clean. The §1 row for 03, R6–R8 and §5 are updated.
    `docs/OPEN_ISSUES.md` changed only if an issue was opened, with the count and date right.
