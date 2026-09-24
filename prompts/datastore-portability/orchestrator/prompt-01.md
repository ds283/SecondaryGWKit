# Orchestrator — prompt 01, relative shard paths

Read [`../README.md`](../README.md) first, especially §0 and §0.1. **You do not write code.** That
includes the change, the tests, and the deliberate breakage that shows the tests work.

**The prompt:** [`01-relative-shard-paths.md`](../01-relative-shard-paths.md)
**Board items:** P0–P4 · **Closes:** `run-registry`'s
`[04-sharded-store-paths-are-absolute-and-so-stores-are-not-portable]`

## 0. What makes this prompt unusual

The code change is a few lines, and a version of it that "works" is easy to write. The review is
about three things:

- **Is there any fallback to the stored absolute path?** A resolver that tries the sibling and then
  "helpfully" falls back to the old absolute path passes every move test and reintroduces the
  copy failure. Read the resolver line by line for this.
- **Does opening a store change it?** Legacy rows are to be interpreted when read, not rewritten.
  The backup has to stay byte-identical.
- **Was a real store ever opened?** Only a copy may be, and only after the change. On the unfixed
  tree, opening a copy writes to the original's shards.

## 1. Before you dispatch

1. **Confirm the branch and `HEAD`**, and record the SHA. Tell the agent the tree is not its own.
2. `python -m RunRegistry list`, and confirm nothing is `running` against `var/datastores/`.
3. **Baseline every suite**, `Datastore/tests` included, and record the counts.
4. **Record the state of the three stores**: mtimes and per-shard row counts for
   `handover-atol-sweep-shard*`, `handover-A3-baseline-lambdacdm-shard*` and the backup's shards,
   plus a `sha256sum` of each of the three primaries. Take all of it read-only with `sqlite3`
   `mode=ro`. You will compare against it afterwards.
5. **See §0.1 for yourself:**
   ```bash
   ./venv/bin/python -c "import sqlite3; c=sqlite3.connect('file:var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite?mode=ro', uri=True); print(list(c.execute('select * from shards')))"
   ```
   The paths printed point into `var/datastores/`, not into the backup directory.
6. `git status` clean.

## 2. Dispatch

One fresh-context subagent. Give it the prompt file, the campaign README, the SHA, the baselines,
and the files the prompt's "Read first" names. Nothing else.

Tell it plainly:

- **one commit**;
- the log goes at `logs/01-relative-shard-paths.md`;
- create the board per §8, and update `docs/OPEN_ISSUES.md` in the same commit;
- run `black` before committing;
- **never open an original store or the backup through `ShardedPool` or `main.py`**; only a copy,
  and only after the change;
- **delete the copy afterwards**;
- the stop conditions in §7 mean *stop and ask*.

Running the §4 inventory on a copy takes minutes, not hours, so it does not need the run registry.

## 3. The review — ten checks

1. **No fallback.** Read the resolver. An absolute stored value resolves to
   `primary.parent / name`, full stop. Any branch that returns the stored absolute path, under any
   condition, is a stop.
2. **Fail closed, before any actor exists.** The missing-shard check runs in the existing-store
   branch, before the first `Datastore.options(...).remote(...)`.
3. **The tests fail on the unfixed code.** The log must show it for the legacy, copy and
   fail-closed tests. Check it yourself: revert the two production files to `HEAD~1`, run those
   tests, confirm they fail, then restore.
4. **No Ray, no datastore in the tests.** If a test opens anything under `var/`, or calls the
   constructor, that is a stop.
5. **One resolver.** `grep -n "filename" Datastore/SQL/ShardedPool.py tools/shard_key_audit.py`:
   every read goes through it. The audit tool still runs standalone and read-only.
6. **The stores are untouched.** Re-take §1.4. Row counts and mtimes are identical, the three
   primaries' hashes are identical, and no copy is left under `var/`.
7. **The whole-store rename is opened, not decided.** It appears as a §3 issue with both options
   (a rename tool, or names derived from the stem). No implementation of either.
8. **P0 is recorded.** The log quotes what a moved store did on the unfixed tree: raised, or
   opened with empty shards. The throwaway store is gone. The `run-registry` §4 note says which
   account in README §0 was right.
9. **The sweep script is untouched.** `git diff HEAD~1 HEAD -- docs/handover/` is empty.
10. **Suites** match their baselines, `Datastore/tests` is up by exactly the tests added, `black
   --check` is clean, the board is in the §8 shape, and `docs/OPEN_ISSUES.md` has its count and
   date corrected. The `run-registry` issue has moved from that board's §3 to its §4, and its
   row is gone from `docs/OPEN_ISSUES.md` §1.11.

## 4. Stop and ask the user

Relay verbatim; do not adjudicate.

- Any of §7 of the prompt.
- Any check in §3 fails. **Report it; do not repair it.**
- The agent proposes rewriting the rows of an existing store, or "fixing" the backup.

## 5. After it lands

Report:

- the commit;
- what P0 found;
- the resolver, and where it lives;
- the deliberate-breakage record, and whether you reproduced it;
- the §4 inventory against the original's row counts;
- the store state before and after;
- what opening the backup would now do;
- the issues opened;
- the suite counts.

Then stop. Whether to support whole-store renames, and how, is the user's decision.
