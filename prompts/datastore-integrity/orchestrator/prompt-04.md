# Orchestrator — prompt 04, extend the read-back guard

Read [`../README.md`](../README.md) first: §0, §2 (why the instrument comes before the remedy), §4
and §5. **You do not write code.** You may *replay* a mutation the log records, with `git apply`,
and then revert it.

**The prompt:** [`04-extend-the-read-back-guard.md`](../04-extend-the-read-back-guard.md)
**Board items:** G1–G3 · **Closes:** `[01-read-batch-is-outside-the-guard]` (datastore-readback
board) · **Gate:** none; independent of 01–03.

## 0. What makes this prompt unusual

It is the instrument prompt 05 is scored against, and it **modifies an existing test module**,
which README §5 rule 7 allows only because the prompt names it. So the review turns on the guard
getting stronger and never weaker, and on its findings being recorded rather than fixed.
- **Nothing weakened.** Every assertion the module made before still stands. `EXPECTED_COVERAGE`
  only grows: no entry that was there is removed. `KNOWN_BLIND` grows only with a reason per
  entry. The four existing tests keep their names and their assertions. Diff the module against
  `HEAD~1` and read every removed line.
- **Nothing fixed.** `git diff HEAD~1 HEAD -- Datastore/SQL/` is empty. Every defect found is in
  `KNOWN_UNFIXED`, naming its P-item. The probes predicted five (two `WKB_criterio`, `z_serial`,
  `model_id`, the `validated` insert key); anything else is **new**, and the log must say so.
- **It cannot be undone silently.** Mutation (v), the analyser skipping `read_batch` again, must
  fail `test_analyser_coverage_is_undiminished`. If it fails some other test instead, or none, the
  coverage pin does not cover the extension.
- **A live-path defect is a stop.** If the guard finds a defect on a path `main.py` reaches during
  a build, the prompt's §5 says stop and ask, because prompt 05 is held and the user may want the
  fix moved ahead of it. Check the log for any such finding, and bring it to the user before
  recording the review.

**Carried from this campaign's writing.**
- The method is `docs/datastore-integrity-audit/lookup_keys/probe_02_readback_guard_by_hand.py` and
  `probe_03_column_existence.py`, with their `.out` files. They are the audit's record: the agent
  runs and reads them and must not edit them. `probe_02.out` lists the 16 bindings the extension is
  expected to add; a different count is acceptable with an explanation.
- **Order against 01.** 01 and 04 touch different files, but 04's guard reads every factory's
  source and 01 edits `wavenumber.py`'s and `BackgroundModel.py`'s `store()` paths. **Dispatch them
  one at a time, never concurrently in this checkout.** Either order is right. If 01 has landed
  first, this dispatch's `HEAD` includes it and the baselines are 01's.
- Check `git diff --cached` is empty as well as the working tree: during this campaign's writing
  another session had changes staged in this checkout.

## 1. Before you dispatch

1. Record the branch and `HEAD`. `HEAD` must be this campaign's latest commit.
2. `python -m RunRegistry list`: nothing `running`.
3. **Snapshot, read-only,** as for prompt 01: `var/datastores/`, `var/runs/` and the repository
   root's `physics-test-n20-lambdacdm-zend0p1*.sqlite`. Reuse `orch_01_snapshot.py` if this session
   has it; otherwise write it again in the session scratchpad as `orch_04_snapshot.py`, never in the
   tree. This prompt opens no store, so the snapshot is a check that it did not.
4. The baselines. If 01 has not landed: at `cdf097f`, AdaptiveLevin 32, ComputeTargets 552 (the
   `test_wall_time_per_object` flake is known), CosmologyModels 39, Datastore 206, LiouvilleGreen
   148 (1 skipped), RunRegistry 199. If 01 has landed: the figures its review recorded on this
   board's §5. Either way, re-measure all six on `HEAD`, in the checkout, with README §7's command.
5. `git status --short` is empty and `git diff --cached` is empty.

## 2. Dispatch

One fresh-context subagent (**Sonnet**, as the prompt recommends). Give it:
- the prompt;
- the campaign README;
- the audit, `docs/datastore-integrity-audit.md`;
- the datastore-readback board's entry for `[01-read-batch-is-outside-the-guard]`;
- `HEAD` and the baselines.

Tell it plainly:
- **one commit**, with its log at `logs/04-extend-the-read-back-guard.md`;
- in the same commit: this board (§1 row, G1–G3, §3.1, §5); the datastore-readback board, with the
  issue moved to its §4 and a closing note naming this commit; and `docs/OPEN_ISSUES.md`, with the
  row deleted from §1.15 and the count and date corrected;
- run `black`;
- **the only existing test file it may change is `Datastore/tests/test_factory_select_columns.py`.**
  The column-existence test may go there or in a new module beside it. No factory under
  `Datastore/SQL/` changes; every finding goes into `KNOWN_UNFIXED`, naming its P-item from the
  board's §2;
- the guard stays static: no Ray, no engine, no store. The column-existence test may call
  `build_schema(MetaData(), _factories)` and nothing more;
- mutations (i)–(v) are recorded as diffs **that apply with plain `git apply` from the repository
  root**: fenced ```` ```diff ```` blocks, not indented, with correct hunk headers, each checked
  with `git apply --check` against the final commit. Mutations are never committed;
- do not touch `orch_*` files, or the audit's probes;
- the prompt's §5 stop conditions mean *stop and ask* — above all, a defect on a live build path.

## 3. The review — eight checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only
   `Datastore/tests/test_factory_select_columns.py`, a new test module if the agent chose one, the
   log, this board, the datastore-readback board and the index. `git diff HEAD~1 HEAD --
   Datastore/SQL/ docs/datastore-integrity-audit/` is empty, and so is `--diff-filter=M` over every
   other `*/tests/*` path.
2. **Nothing weakened.** In the module's diff, every removed line is a docstring line of points 3
   or 6, or is replaced by a stronger form of itself; list any other in the review. Every
   `EXPECTED_COVERAGE` entry at `HEAD~1` is still present. Every new `KNOWN_BLIND` entry has a
   reason.
3. **Reach.** The analyser now visits every function of every factory, `read_batch` and the
   `_build_impl_*` helpers included, and `OneLoopIntegral`'s module-level `read_batch` is either
   decided or declared blind. Compare the new `EXPECTED_COVERAGE` entries with `probe_02.out`'s 16;
   the log explains each difference.
4. **`KNOWN_UNFIXED`.** Each entry names a P-item. The five the probes predicted are there, or the
   log says why one is not. Every other entry is marked new in the log. For each new one, look at
   the factory line yourself: a real defect, or an analyser false positive? A false positive is a
   review finding, not a pass.
5. **Live paths.** For each `KNOWN_UNFIXED` entry, say whether `main.py` reaches it during a build,
   on resume only, or only from an `extract_*.py` script. If any is on a build path and the agent
   did not stop, that is a finding: bring it to the user.
6. **Column existence.** The test resolves every `<table>.c.<name>` and every insert key, and the
   three names `probe_03.out` found are in `KNOWN_UNFIXED`.
7. **Replay mutations (i)–(v)** with plain `git apply`. Each gives its recorded failures; (v) fails
   `test_analyser_coverage_is_undiminished`. Revert each, leaving the tree clean.
8. **`var/`, suites and the boards.**
   - Re-take §1.3's snapshot. It must be identical.
   - Re-run all six suites. Datastore rises by exactly the tests added; the others match §1.4.
     `black --check` is clean.
   - This board: the §1 row for 04, G1–G3, §3.1, §5. The datastore-readback board has the issue in
     its §4 with a closing note, the index row is gone from §1.15, and the index count fell by one.

**After the review.** Record it on this board. Prompt 05 is then released to be written, against the
`KNOWN_UNFIXED` list as shipped and this review's findings; its charter is fixed in README §2 and
its method follows D2 and D3. Writing it is a separate task.
