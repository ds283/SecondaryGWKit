# Orchestrator — prompt 04b, tighten the read-back guard

Read [`../README.md`](../README.md) first: §0.1, §2, §4 and §5. **You do not write code.** You may
*replay* a mutation the log records, with `git apply`, and then revert it.

**The prompt:** [`04b-tighten-the-read-back-guard.md`](../04b-tighten-the-read-back-guard.md)
**Board items:** G4–G6 · **Closes:** nothing · **Gate:** after 04 (`6a97d8a`), its review
(`85cbe28`) and the prompt (`9d53d5a`).

## 0. What makes this prompt unusual

It repairs the instrument of the prompt before it, on findings 1–3 of that prompt's review. So the
review asks one question three times: **can each of the holes the 04 review measured still be
opened without a test failing?** The measurement is
[`review_04_measurements.py`](review_04_measurements.py) and its `.out`, and the agent must not edit
either.
- **Re-keying is the one permitted rewrite.** `EXPECTED_COVERAGE` and `KNOWN_BLIND` change key form,
  so "every entry still present" becomes "every one of 04's 30 entries is the projection, with the
  function dropped, of a new one". Check that by computing it yourself, not by trusting the log.
- **The unfixed lists do not move.** `KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS` are prompt 05's
  input (README §4). Their keys and contents are the same as at `6a97d8a`, unless the log records
  a **new** defect.
- **`validate_on_startup` is a live path.** It runs on every pool open. A new defect there means
  stop and ask, and the agent should not have committed. The measurement says all seven are clean.
- **Mutation diffs come from `git diff`.** 04's (v) did not apply (review finding 4). Every diff
  here must apply with plain `git apply` from the committed log.

## 1. Before you dispatch

1. Record the branch and `HEAD`, which must be `9d53d5a` or the commit that writes this note.
2. `python -m RunRegistry list`: nothing `running`.
3. Snapshot `var/datastores/`, `var/runs/` and the root `physics-test-n20-lambdacdm-zend0p1*.sqlite`
   read-only, with `orch_04_snapshot.py` in the session scratchpad. Write it again if the session
   lacks it; never put it in the tree.
4. Baselines, from board §5.2: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39,
   Datastore 235, LiouvilleGreen 148 (1 skipped), RunRegistry 199. Re-measure all six on `HEAD`,
   serially.
5. `git status --short` and `git diff --cached` are both empty.

## 2. Dispatch

One fresh-context subagent (**Sonnet**). Give it the prompt, the README, the board's review of 04,
log 04, the measurement script and its `.out`, `HEAD` and the baselines. Tell it plainly:
- **one commit**, with its log at `logs/04b-tighten-the-read-back-guard.md` in README §5.1's form;
- in the same commit, the board: the §1 row for 04b, G4–G6, and a new §5.3. `docs/OPEN_ISSUES.md`
  changes only if an issue is opened;
- the only file it may change besides the log and the board is
  `Datastore/tests/test_factory_select_columns.py`. `Datastore/SQL/`, the audit's probes, log 04,
  `orchestrator/` and every other test are untouched;
- the guard stays static: no Ray, no engine, no store;
- every mutation diff is pasted from `git diff`, then replayed from the committed log with
  `git apply --check`, `git apply` and `git apply -R`;
- run `black`;
- §5's stop conditions mean *stop and ask*.

## 3. The review — seven checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches the test module, the log and the board, and
   nothing else. `git diff HEAD~1 HEAD -- Datastore/SQL/ docs/ prompts/datastore-integrity/orchestrator/`
   is empty, and `--diff-filter=M` over every other `*/tests/*` path is empty.
2. **Nothing weakened.**
   - Every removed line of the module is one of three things: an `EXPECTED_COVERAGE` or
     `KNOWN_BLIND` key rewritten to its function-qualified form, `_ROW_HELPERS` and its gate, or
     docstring point 6. List any other removed line in the review.
   - Compute the projection of the new `EXPECTED_COVERAGE` and check it contains all 30 entries at
     `6a97d8a`.
   - `KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS` are byte-identical to `6a97d8a`'s, or the log
     marks the difference **new**.
3. **Reach (finding 1).** Re-run §A of the measurement against the new module's own findings. Every
   function with a row binding is visited, and no name list decides it. The figures are 43
   decidable and 2 blind, or the log explains the difference.
4. **The pin (finding 2).** Repeat the 04 review's experiment: remove the analyser's visit to each
   function name that contributes coverage, one at a time: `build`, `_build_impl_serial`,
   `_build_impl_model`, `read_batch` and `validate_on_startup`. Each must fail
   `test_analyser_coverage_is_undiminished` on its own, whatever `KNOWN_UNFIXED` holds. Also test
   `_build_impl_serial` with the two `WKB_criterio` entries imagined struck: that must still fail
   the coverage test, and not only the unfixed-list tests.
5. **G2 (finding 3).**
   - The unresolved-table pin holds 12 pairs, or fewer where the agent resolved some.
   - Look at each resolution rule yourself. It must be sound, and it must be able to find a
     defect: rename one column at the resolved site, and G2 must fail.
   - The insert pin holds 13 sites, less any the agent resolved; resolved payload keys are checked.
   - `OneLoopIntegral`'s module-level `read_batch` is walked.
   - An unknown `tables["X"]` fails.
6. **Replay mutations (i)–(vi)** from the committed log with plain `git apply`. Each gives its
   recorded failures. Revert each, leaving the tree clean.
7. **`var/`, suites and the board.**
   - Re-take the snapshot; it must be identical.
   - Re-run all six suites. Datastore rises by exactly the tests added, and the others match §1.4.
     `black --check` is clean.
   - The board has the §1 row for 04b, G4–G6 and §5.3, and 05 is still held.

**After the review.** Record it on the board. Prompt 05 is then released to be written, against the
two unfixed lists as they stand and 04b's pins, with its method under D2 and D3. Writing 05 is a
separate task.
