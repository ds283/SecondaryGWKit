# Orchestrator — prompt 01, one schema builder and a read-only reader

Read [`../README.md`](../README.md) first. **You do not write code.** That covers the change, the
tests and the mutations. You may *replay* a mutation the log records, with `git apply`, and then
revert it.

**The prompt:** [`01-a-read-only-store-reader.md`](../01-a-read-only-store-reader.md)
**Board items:** F1–F3 · **Closes:** `[00-build-schema-reads-registration-before-its-none-check]`

## 0. What makes this prompt unusual

The code is small. The review turns on two proofs.

- **The refactor changed nothing.** The witness is `Datastore/tests/data/schema_at_base.json`,
  captured from the **unchanged** code. Check two things:
  - that it was captured before the change, at the SHA the log names. Re-capture it yourself at
    `HEAD~1` and compare;
  - that both `build_schema` and the actor's `_build_schema` are tested against it. A witness
    regenerated from the new code proves nothing.
- **The reader never writes.** The test must compare hashes, sizes, mtimes **and the directory
  listing**, because a `-journal` appearing is a write. It must also show that a write through the
  reader's engine raises.

## 1. Before you dispatch

1. **Confirm the branch and `HEAD`**, and record the SHA. `HEAD` must be the commit that adds this
   campaign. Tell the agent the tree is not its own.
2. `python -m RunRegistry list`: nothing `running`.
3. **Baseline every suite** (README §7). Re-measure; do not copy the README's numbers.
4. **Snapshot, read-only,** the three stores and their sidecars: each file's size, `st_mtime_ns`
   and per-table row counts, and the SHA-256 of the primaries and sidecars. Also every entry under
   `var/runs/`, with the SHA-256 of each `manifest.json` and `status.json`. Keep the script and its
   output under an `orch_` prefix in the scratchpad, and tell the agent not to touch `orch_*`
   files.
5. **Free disk space:** at least 1 GB (`df -h .`).
6. `git status` is clean, and `var/` holds nothing new.

## 2. Dispatch

One fresh-context subagent. Give it:
- the prompt, the campaign README and the audit;
- the SHA and the baselines;
- the files the prompt's "Read first" names.

Nothing else. Tell it plainly:
- **one commit.** The log goes at `logs/01-a-read-only-store-reader.md`;
- update the board and `docs/OPEN_ISSUES.md` in the same commit;
- run `black` before committing;
- **never point the reader, or any new code, at an original store or the backup.** Never write
  under `var/datastores/` or `var/runs/`. The demonstration works only on a copy under
  `var/store-fingerprint-check-01/`, which it deletes afterwards;
- mutations are recorded in the log as diffs, exactly as applied, and never committed;
- do not touch `orch_*` files;
- the prompt's §7 stop conditions mean *stop and ask*.

## 3. The review — ten checks

1. **The witness.** Re-capture `schema_at_base.json` from `HEAD~1` the way the log describes, and
   compare it with the committed file. Both must be identical. Both `build_schema` and the actor's
   `_build_schema` are tested against it.
2. **One builder.** `Datastore._build_schema` calls `build_schema` and adds only the inserters.
   Nothing else in `Datastore/` rebuilds tables from `register()`. Check this with
   `git grep -n 'register()' -- Datastore/ ':!Datastore/tests'`.
3. **The fix.** `registration_data` is checked for `None` before `.get` is called on it, and a test
   covers a `None` registration.
4. **No write path.** `Datastore/store_reader.py` holds no `create`, `create_all`, `_ensure_tables`,
   `execute` of DDL or DML, or `immutable=1`, and every URI carries `mode=ro`. Shards come from
   `_read_closed_store`, and journals from `_journal_paths`, not from copies of either.
5. **The never-writes test** compares SHA-256, size, `st_mtime_ns` and the directory listing, and
   shows that a write through the reader's engine raises.
6. **Layering.**
   `git diff HEAD~1 HEAD -- Datastore/SQL/ShardedPool.py Datastore/SQL/ObjectFactories/ main.py RunRegistry/ tools/ config/`
   is empty. `git diff HEAD~1 HEAD --stat -- '*/tests/*'` shows only added files, besides the new
   test data.
7. **Mutations reproduce.** Replay (ii) (a prepended column omitted) and (v) (the journal refusal
   removed) from the log with `git apply`. Run the named tests, confirm that they fail, and revert.
   `git status` is clean afterwards.
8. **No Ray, no `var/`, in the tests.** Nothing initialises Ray, calls the `ShardedPool`
   constructor, or opens anything under `var/`.
9. **The originals are untouched.** Re-take §1.4. Everything must be identical, and
   `var/store-fingerprint-check-01/` must be gone. Re-derive the demonstration's per-table counts
   for one shard from the log's record, and check them against your own snapshot of the original
   sweep store; a `cp -p` copy has the same counts.
10. **Suites and the board.** Every suite matches its baseline. `Datastore/tests` rises by exactly
    the tests added, and `black --check` is clean. On the board, F1–F3 are done and the issue has
    moved to §4. In `docs/OPEN_ISSUES.md`, its row is gone, and the count and date are right.

## 4. Stop and ask the user

Relay verbatim; do not adjudicate.

- Any of the prompt's §7.
- Any check in §3 fails. **Report it; do not repair it.**
- The agent proposes changing `ShardedPool`, a factory's registration, or what an actor writes on
  open.

## 5. After it lands

Report:
- the commit;
- how the witness was captured, and your re-capture;
- what importing the reader loads;
- the demonstration's numbers, including every absent or extra column on the sweep copy;
- the mutations you replayed;
- the state of the originals, before and after;
- the suite counts.

Then stop. Prompt 02 is dispatched separately, once README §6.2 D1 is decided.
