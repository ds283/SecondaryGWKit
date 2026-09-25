# Orchestrator — prompt 04, amend an unknown field

Read [`../README.md`](../README.md) first, §6.2 D5 and D6 in full. **You do not write code.** You
may *replay* a mutation the log records, with `git apply`, and then revert it.

**The prompt:** [`04-amend-an-unknown-field.md`](../04-amend-an-unknown-field.md)
**Board item:** R9 · **Closes:** nothing · **Gate:** prompt 03 has landed (`854e2ae`, reviewed in
`d9a511c`)

## 0. What makes this prompt unusual

The operation is small. The review turns on two properties, and on how the refusals are layered.
- **It can touch nothing the registry interprets.** Every field in `KNOWN_FIELDS`, `retired`
  included, is refused before anything is written, and `history` changes only by one appended
  entry. Mutation (i) must fail the tests.
- **It loses nothing it replaces.** `before` is the old value verbatim, nested structure included,
  and the "absent" and "removed" markers cannot be mistaken for any JSON value. A bare sentinel
  string such as `"<absent>"` fails this, because a field could hold that string. Mutation (ii)
  must fail the tests.
- **The refusals are layered.** Prompt 03's validator already rejects any entry after `retire`,
  and `_check_before_writing` runs on every write. So an amend of a tombstone, or of a known field
  with a malformed value, may be caught by the validator even when the refusal the prompt asks for
  is missing. Mutations (i) and (v) then fail only on the *message*. That is acceptable only if
  the tests assert the message: the tombstone text for (v), and the name of the owning operation
  for (i). Check that they do.

**Carried from prompt 03's review.**
- `SidecarReading.retired` and `_tombstone_text(reading)` are how every other operation refuses a
  tombstone. Amend should use them, and check for a tombstone before `ok`, so the message says
  "tombstone" and not the generic "not a problem-free registry sidecar".
- The running-run check reads the sidecar's recorded `store_id` field, not
  `SidecarReading.store_id`. The latter is `None` for any sidecar that is not `ok`.
- A missing required flag exits 2, which is argparse's code and the `--purpose`/`--reason`
  precedent. A refusal inside the operation exits 1. Prompt 04's "exits 1 on a refusal" is read
  that way, and is not a finding.
- `[03-the-package-docstring-still-says-the-registry-deletes-nothing]` stays open. It is **not**
  this prompt's: `RunRegistry/__init__.py` is out of scope here.

## 1. Before you dispatch

1. Record the branch and `HEAD`. `HEAD` must be this campaign's latest commit, the one that added
   these notes.
2. `python -m RunRegistry list`: nothing `running`.
3. **Snapshot, read-only,** as for prompt 03: everything under `var/datastores/` and `var/runs/`,
   and the repository root's `physics-test-n20-lambdacdm-zend0p1*.sqlite`. Record size, mtime and
   SHA-256 for stores and sidecars, and size and mtime for run files. The live A3 sidecar, which
   prompt 05 will amend, is the one that matters most. Reuse `orch_03_snapshot.py` if this session
   still has it; otherwise write it again in the session scratchpad under an `orch_` prefix, never
   in the tree.
4. The baselines, from the board's §5 after prompt 03: AdaptiveLevin 32, ComputeTargets 552,
   CosmologyModels 39, Datastore 206, LiouvilleGreen 148 (1 skipped), RunRegistry 169. Re-measure
   them on `HEAD`, in the checkout itself, never in an exported copy.
5. `git status` is clean, apart from untracked paths that are not this campaign's (today
   `docs/datastore-integrity-audit*` and `prompts/datastore-integrity/`). Tell the agent to leave
   them alone.

## 2. Dispatch

One fresh-context subagent (**Sonnet**, as the prompt recommends). Give it:
- the prompt;
- the campaign README;
- the audit;
- log 03;
- `HEAD` and the baselines.

Tell it plainly:
- **one commit**, with its log at `logs/04-amend-an-unknown-field.md`;
- update this board in the same commit, and `docs/OPEN_ISSUES.md` only if it opens an issue;
- run `black`;
- **never call `amend_sidecar` or `store amend` against anything under `var/`, or a copy of
  anything under `var/`.** That includes the live A3 sidecar, which prompt 05 amends by the user's
  command. This prompt has no real-sidecar contact at all;
- **every test passes `runs_root` explicitly**, and every command-line test passes
  `--runs-root`. The default is `var/runs/`, so a call that omits it reads the real manifests;
- `a3_shaped`'s fields hold repository-relative paths such as `var/datastores/backup-…`. They are
  data. No test opens, resolves or creates anything at them;
- these are **not changed**:
  - `ShardedPool`;
  - `tools/sharded_store.py`;
  - `CLAUDE.md`;
  - `RunRegistry/__init__.py`;
  - the behaviour of `retire_store`, `copy_store`, `move_store` and `fingerprint_store`;
  - every existing test.

  `RunRegistry/tests/store_fixtures.py` may receive additions only;
- mutations (i)–(v) are recorded as diffs **that apply with plain `git apply` from the repository
  root**: fenced ```` ```diff ```` blocks, not indented, with correct hunk headers, each checked
  with `git apply --check` against the final commit. Mutations are never committed;
- do not touch `orch_*` files, or untracked paths that are not its own;
- the prompt's §5 stop conditions mean *stop and ask*.

## 3. The review — ten checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only:
   - `RunRegistry/stores.py` and `RunRegistry/__main__.py`;
   - the new test module, and any addition to `store_fixtures.py`;
   - the log and the board;
   - the index, only if an issue was opened.

   `git diff HEAD~1 HEAD -- Datastore/ tools/ CLAUDE.md RunRegistry/__init__.py` is empty.
   `git diff HEAD~1 HEAD --diff-filter=M -- RunRegistry/tests/` is empty, or is `store_fixtures.py`
   with additions only. In `stores.py`, no hunk falls inside `retire_store`, `copy_store`,
   `move_store`, `_prepare`, `fingerprint_store` or the reader's tombstone logic, apart from
   docstrings.
2. **Only unknown fields.** The refusal tests membership of `KNOWN_FIELDS` itself, not a
   hand-written list, and comes before any write. The message names the owning operation. A test
   iterates over `KNOWN_FIELDS`, and `retired` is among them. After an amendment, `history[:-1]`
   equals the history before, and every field but the amended one and `history` is value-identical.
   The fingerprint is not retaken.
3. **Nothing lost.**
   - Read the markers. No JSON value can equal one. A test amends, or removes, a field whose value
     has the marker's own shape, and reads it back unambiguously. If no test does, record that as
     a finding, not a failure.
   - `before` is value-identical to the old value, nested structure included.
   - Two amendments of one field give the whole sequence of values, in order.
4. **The refusals.** Each is refused, and a test covers each:
   - a blank reason;
   - both `value` and `remove`, or neither;
   - `remove` of an absent field;
   - an identical `value`, by JSON round trip;
   - a value that is not JSON-serialisable, and bad `--json`;
   - an absent, legacy, unreadable or problem sidecar;
   - a tombstone, **complete and incomplete**, with the tombstone message;
   - an alive running run, and a stale one.

   The running-run check is `_running_runs_naming("amend", …)` with the recorded `store_id`. Amend
   never imports `ShardedPool` and never opens the store's files. Confirm that `tree_state` of the
   store's directory is unchanged apart from the sidecar.
5. **The history rule.** `amend` is added to what may stand after index 0. It may not stand at
   index 0, or after `retire`, which stays terminal. Its extra keys (`field`, `reason`, `before`,
   `after`) are required on `amend` and are a problem on any other operation. `amend`'s
   `from`/`to` rule is exact, and the rules for `copy`, `move` and `retire` are unchanged.
   `test_store_retire.TestHistoryRule` passes unmodified. `_registry_problems`' rule that
   `retired` stands if and only if the history ends in `retire` still holds.
6. **Copy and move** carry an amended field and its `amend` entry verbatim, and a test shows it.
   The copy's own history appends its `copy` entry after the `amend`.
7. **The docstrings.**
   - `_update_sidecar`'s docstring counts five uses, amend included, and is true.
   - The `stores.py` module docstring lists amend among the in-place updates, among the
     operations, and in the format table, with the `amend` keys, the markers and the `from`/`to`
     rule as shipped.
   - `__main__.py`'s docstring adds `amend`, and says it is the one way to change an unknown field
     and never touches a known one.
   - "Every operation but a retirement's completion refuses a tombstone" is still true.
8. **Tests.** Run the new module twice. Grep it: every `amend_sidecar`, `copy_store`, `move_store`
   and `begin` call passes `runs_root`, and every `store` command passes `--runs-root`. Run
   `python -c "import RunRegistry, sys; print('ray' in sys.modules, 'sqlalchemy' in sys.modules)"`
   yourself; it must print `False False`.
9. **Replay mutations (i)–(v)** with plain `git apply`, from an empty working directory in the
   scratchpad, with the repository on `PYTHONPATH`. Each gives its recorded failures. For (i) and
   (v), note whether the failure is on the refusal itself or on its message (§0). Revert each,
   leaving the tree clean and the working directory empty.
10. **`var/`, suites and the board.**
    - Re-take §1.3's snapshot. It must be identical, including the live A3 sidecar's SHA-256 and
      the `physics-test-n20-*` store's mtimes.
    - Re-run all six suites. RunRegistry rises by exactly the tests added, and the others match
      §1.4. `black --check` is clean.
    - On the board, the §1 row for 04, R9 and §5 are updated, and 05's row is left held.
    - `docs/OPEN_ISSUES.md` changed only if an issue was opened, with the count and date right.

**After the review.** With 04 landed, 01–04 are done, and README §2 releases prompt 05 to be
written against what 01–04 shipped. Writing it is a separate task. Log 03's two points for 05's
orchestrator (the board's review of prompt 03) go into it.
