# Orchestrator — prompt 05, repair the read paths (05a → 05b → 05c)

Read [`../README.md`](../README.md) first: §0.1, §2, §4 and §5, with D1–D3. **You do not write
code.** You may *replay* a mutation the log records, with `git apply`, and then revert it.

**The prompts**, dispatched in this order within one session:

| # | Prompt | Items | Model | Closes |
|---|---|---|---|---|
| 05a | [`05a-repair-the-mechanical-read-defects.md`](../05a-repair-the-mechanical-read-defects.md) | P6, P7 | Sonnet | — (empties `KNOWN_UNFIXED`) |
| 05b | [`05b-repair-the-integral-read-batches.md`](../05b-repair-the-integral-read-batches.md) | P1, P2, P8 | Sonnet | `[00-tagged-read-batch-joins-an-unselected-alias]`, `[00-oneloop-lookup-joins-the-wrong-tag-table]` |
| 05c | [`05c-key-the-value-reads-on-their-parent.md`](../05c-key-the-value-reads-on-their-parent.md) | P3, P4, P5, P9 | **Opus** | `[00-numeric-value-parent-lookup-omits-break-point-kind]` |

**Gate:** 04 and 04b landed and reviewed (board §1). Independent of 02 and 03, which touch disjoint
factory files (checked while writing 05), but **never dispatched concurrently with them in this
checkout**.

## 0. What makes this session unusual

- **Three prompts, one charter.** README §2's 05 charter is split three ways so that each commit
  stays revertible, and so that the mechanical fixes land before the round trips that need them.
  **Review each prompt fully, and record the review, before dispatching the next.** Each review's
  findings go into the next dispatch's brief. A finding that needs the user's decision ends the
  session there. Report it, and do not dispatch the next prompt.
- **The guard is the scorer, and each prompt moves it.** Pin edits in
  `test_factory_select_columns.py` are allowed only as each prompt's §2 lists them, each shown
  forced by a failing assertion. The orchestrator simulated 05a's and 05b's edits against the
  guard while writing the prompts:
  - **05a:** no movement except the struck entries;
  - **05b:** exactly one coverage entry added (`OneLoopIntegral.py:sqla_OneLoopIntegral_factory.read_batch.row_data`),
    one `KNOWN_BLIND` entry removed, three `KNOWN_UNRESOLVED_TABLES` pairs removed, and
    `KNOWN_UNFIXED_COLUMNS` emptied;
  - **05c:** none expected.

  Anything else is a finding.
- **05c changes two live paths.** `GkNumericValue.read_batch` and `GkWKBValue.read_batch` are
  called by `main.py`'s `build_missing_GkSource`. The new refusal must fire only where two parents
  match one key.
- **Mutation diffs come from `git diff`,** and apply with plain `git apply` from the committed log.
  04's (v) did not (review of 04, finding 4).

## 1. Before each dispatch

1. Record the branch and `HEAD`. It must be the previous review's commit, or this note's commit for
   05a.
2. `python -m RunRegistry list`: nothing `running`.
3. Snapshot `var/datastores/`, `var/runs/` and the root `physics-test-n20-lambdacdm-zend0p1*.sqlite`
   read-only, with `orch_04_snapshot.py` in the session scratchpad. Write it again if it is
   missing; never put it in the tree.
4. Baselines: the board's latest §5 subsection. Before 05a that is §5.3: AdaptiveLevin 32,
   ComputeTargets 552, CosmologyModels 39, Datastore 237, LiouvilleGreen 148 (1 skipped),
   RunRegistry 199. Re-measure all six on `HEAD`, serially.
5. `git status --short` and `git diff --cached` are both empty.

## 2. Dispatch

One fresh-context subagent per prompt, on the model the table names. Give it:
- the prompt;
- the README;
- the audit;
- the board's reviews of 04 onwards;
- the previous log in this session, if any;
- `HEAD` and the baselines;
- the previous review's findings, if any.

Tell it plainly:
- **one commit**, with its log at the path its §5 names, in README §5.1's form;
- in the same commit: the board (its §1 row, its items, §3.1 if an issue closes, and its new §5
  subsection); the owning board of any issue it closes or narrows; and `docs/OPEN_ISSUES.md`,
  per CLAUDE.md's index rule;
- which files it may change. That is the factories its §2 names; its new test module;
  `Datastore/tests/test_factory_select_columns.py`, only as its §2 allows; the log; and the boards
  and index. Nothing else: no `main.py`, no `extract_*.py`, no other test, no probe, nothing in
  `orchestrator/`;
- no Ray, no engine on a real store, nothing under `var/`;
- every mutation diff is pasted from `git diff`, then replayed from the committed log;
- run `black`;
- its §6 stop conditions mean *stop and ask*.

## 3. The review — for each prompt

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only the permitted files.
   `git diff HEAD~1 HEAD -- main.py extract_*.py docs/ prompts/datastore-integrity/orchestrator/`
   is empty, and so is `--diff-filter=M` over every `*/tests/*` path except the guard module.
   Read every factory hunk, and say for each whether it is one of its §2's changes.
2. **The guard.** Diff the guard module and list every pin that moved.
   - Compare the list with §0's predictions.
   - For each movement, confirm the log shows the assertion that forced it.
   - Recompute the pins from the new tree with the guard's own functions: they must equal what is
     pinned.
   - `KNOWN_UNFIXED` is empty after 05a, and both unfixed lists are empty after 05b.
3. **The fixes do what they say.** For each change, find the test that goes through it and would
   fail without it. The mutations are one check; also read the test.
   - **05a:** each rename names a column that exists, and the tags recount is explained.
   - **05b:**
     - the P2 test's exit-time serials really differ from their wavenumbers'; read the fixture;
     - the `OneLoopIntegral` stand-in is scoped by `mock.patch`;
     - `source_serial` is untouched (`git diff HEAD~1 HEAD | grep source_serial` shows nothing
       removed or added);
     - the `KeyError` pin test exists and names the issue.
   - **05c:**
     - `break_point_kind` comes from the compute targets' constants;
     - the refusal fires on two parents and not on one. Run the prompt's test 3, the live payload
       shape, yourself, and read the refusal's condition in the code;
     - `main.py` is unchanged;
     - the `read_batch` pin equals the set a walk of `ObjectFactories/` finds.
4. **Live paths (05c).** Verify the log's caller table independently, by `grep` for every caller of
   the six reads. Do **not** open a store under `var/` to find out whether a real store would now
   refuse. If you think one would, raise it with the user.
5. **Replay the mutations** from the committed log with plain `git apply`. Each gives its logged
   failures. Revert each, leaving the tree clean.
6. **`var/`, suites, boards.**
   - The snapshot is identical.
   - All six suites, run serially: Datastore rises by exactly the tests added, and the others match
     §1.4.
   - `black --check` is clean.
   - Issues closed on their owning boards' §4 with closing notes, index rows deleted, and the count
     and date right.

**Record each review** on the board, after the previous one, as the reviews of 04 and 04b are
recorded, and commit it before the next dispatch.

## 4. After 05c's review

- Mark P9 closed and prompt 05 landed on the board and in README §2.
- 07's gate is 05 **and** 06, and 06 follows 03, which follows 02. So after this session the next
  dispatches are 02 → 03 → 06, each with its own orchestrator note. Say so on the board.
- Anything 05c records for 07, such as how the WKB reads now take `z_init` and refuse, goes into the
  board's "carried to 07" note.
