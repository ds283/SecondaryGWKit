# Orchestrator — prompt 06, the foreign keys and the first witness change

Read [`../README.md`](../README.md) first: §0.1, §2, §4, §5 and §6.2 **D5** in full. **You do not
write code.** You may *replay* a mutation the log records, with `git apply`, and then revert it.

**The prompt:** [`06-correct-the-foreign-keys.md`](../06-correct-the-foreign-keys.md)
**Board items:** S1–S5 · **Closes:** `[00-quadsource-tq-serial-has-the-wrong-foreign-key]`
(store-fingerprint) · **Gate:** 03 landed and reviewed (board §1, `7ad6ab7`). Two follow-ups have
landed since, at the user's request: `3520b9f`, a comment in the exit-time `build()`, and
`e224e55`, the two verification drivers. Never dispatched concurrently with 07–09 in this checkout.

## 0. What makes this prompt unusual

The prompt was written on 2026-09-26. Its task is unchanged. What moved under it is below,
measured at `e224e55`. The orchestrator sketched S1–S3 and the two flags in a throwaway worktree,
since removed. The sketch is not the agent's to copy, and was not given to it.

- **This is the first deliberate schema change.** D5's procedure is what later prompts copy:
  07, 08 and 09 each add a witness beside this one. The procedure matters more than the edits.
  - The witness test is `Datastore/tests/test_schema_builder.py`, and its one constant is
    `WITNESS` (`:31`). Two tests compare against it, `test_build_schema_reproduces_the_witness` and
    `test_actor_build_schema_reproduces_the_witness`, class by class and then byte for byte.
  - `Datastore/tests/schema_description.py` captures a witness from the current tree through its
    `__main__` (`:199`). It is also what the tests call.
  - **Measured.** In the sketch those two tests failed, and nothing else did: sub-tests for
    `GkSourcePolicyData`, `IntegrationSolver`, `QuadSource` and `wavenumber_exit_time`, plus the
    whole-file assertion.
    - Datastore gave `failures=10`, all in `test_schema_builder`.
    - RunRegistry, including `test_store_fingerprint` (33 OK), ComputeTargets 552 and
      CosmologyModels 39 were unchanged.
    - RunRegistry shows `skipped=1` in a worktree only, because the hand-written A3 manifest under
      `var/` is absent there. In this checkout it is 199 OK.
  - **So the golden fingerprint does not move**, as the prompt expects. If it moves under the
    agent's change, that is §5's stop, not something to re-capture.
- **The witness diff is larger than "S1, S2 and two flags" suggests.** `describe_schema` records
  each foreign key three times:
  - in the column's `foreign_keys`;
  - in the table's `ForeignKeyConstraint` list;
  - in the table's `CREATE TABLE` DDL string.

  In the sketch, `diff` between `schema_at_base.json` and a fresh capture gave **9 hunks and 30
  changed lines**:
  - `GkSourcePolicyData`: 3 hunks (a new column foreign key, a new constraint, the DDL);
  - `QuadSource`: 3 hunks (the target in each of those three places);
  - one `stepping_mode` hunk per class for the two flags;
  - one more hunk, `GkSourcePolicyData`'s constraint list growing.

  All 30 lines are S1, S2 or a flag. The prompt's rule, "exactly S1, S2 and the two flags, and
  nothing else", holds for that diff. The agent should expect those three appearances per key
  and not stop on them. A line outside those four classes is the real stop.
- **The fixture cannot show S4's baseline as it stands. This is the finding that shapes the test.**
  `build_full_store`'s `QuadSource` rows have `Tq_serial` equal to a `QuadSource` serial on their
  own shard:
  - shard 0: `QuadSource` serial 1, with `Tq_serial` 1 and `Tr_serial` 2;
  - shard 1: serial 2, with `Tq_serial` 2 and `Tr_serial` 3.

  So at `e224e55`, `PRAGMA foreign_key_check` finds **no violation** on any file of the full store,
  or of `build_real_store`'s default store. The wrong key is satisfied by coincidence. On that
  fixture, prompt §3.1 ("fails on the tree before S1") and mutation (i) could not fail.
  - `relabel_serials` does not help. It renumbers each reference by the table `references()`
    names, which is the declared foreign key. So pointing `Tq_serial` back at `QuadSource` makes
    the relabelled data follow the wrong key too. After the sketch's S1 and S3, the relabelled
    store's `Tq_serial` values (706, 705) are `TkNumericIntegration` serials and pass, but under
    mutation (i) they would be `QuadSource` serials again.
  - **What the test needs:** a `QuadSource` row, as fixed data, whose `Tq_serial` is a
    `TkNumericIntegration` serial on its shard and **not** a `QuadSource` serial there. Build it by
    passing the test's own `sharded` rows to `build_full_store` or `build_real_store`. Do not change
    the fixture module's default rows, which the inventory and golden-fingerprint tests read. The
    log should say which rows were chosen and why.
  - A store written through stand-in shards would probably have the same coincidence, since
    each class's serials start at 1. If the agent uses one, it must show that the serials differ.
- **Measured after the sketch's S1–S3.** There are no violations on the full store, the default
  store or the relabelled store:
  - every `Tq_serial` resolves to a `TkNumericIntegration` row on its shard;
  - every `GkSourcePolicyData.wavenumber_exit_serial` resolves to a replicated exit time.
- **Mutation (ii) fires on the full store as it is.** With `Tr_serial` given a foreign key to
  `TkNumericIntegration`, the check reports `('QuadSource', 1, 'TkNumericIntegration', 0)` on shard
  0 and `('QuadSource', 2, 'TkNumericIntegration', 0)` on shard 1. `Tr` 2 and 3 are on other
  shards, or nowhere. So the fixture already holds the cross-shard pair the prompt asks about.
- **`Tq` is on `QuadSource`'s shard.** This is prompt §5's first stop, and it does not trigger. The
  log must quote this evidence:
  - `main.py:1705` builds the cache from `pool.object_get("TkNumericIntegration", …)`.
  - `main.py:1732-1733` takes `Tq`/`Tr` from that cache by `q`/`r`.
  - `ComputeTargets/QuadSource.py:451-452` and `:473-474` record their `store_id`s.
  - `config/sharding.py:20,24` shards `TkNumericIntegration` on `k` and `QuadSource` on `q`,
    through the same wavenumber key (`shard_key_wavenumber_store_id`).
- **`stepping_mode` is read nowhere else.** This is prompt §5's second stop, and it does not
  trigger.
  - It is set at `schema.py:105-121`, which accepts `"minimum"` and `"exact"`, and read only by
    `schema_description.py`.
  - `Datastore.py:610` reads `use_stepping`, not the mode.
  - Log 03's "Observations not acted on" item 1 hands the two flags to 06.
- **The fixture's map changes what `relabel_serials` does.** `_UNDECLARED_REFERENCES`
  (`real_store_fixtures.py:1182-1186`) and `references()` feed `relabel_serials`. After S1 and
  S3, `Tq_serial` and `Tr_serial` are renumbered as `TkNumericIntegration` serials, where today
  they are renumbered as `QuadSource` serials.
  - `QuadSource`'s `inventory_records` treats both as provenance, not identity
    (`QuadSource.py:472`).
  - So the inventory and fingerprint relabel tests do not move, and that was measured. The log
    should say it.
- **The two docstrings that name the base witness.** Two docstrings describe `schema_at_base.json`
  as the one witness: the module docstring of `test_schema_builder.py`, and that of
  `schema_description.py`.
  - The prompt grants the first, under D5.
  - The second becomes false after 06. It says the tests compare against "that file". **This note
    grants `schema_description.py`'s module docstring, and only its docstring**, so that the two
    agree. Its code is not touched.
  - This follows the user's standing rule (2026-09-29) that residue is fixed rather than left.
- **Line numbers, re-checked at `e224e55`.**
  - `QuadSource.py:113-125` holds `Tq_serial` (its `ForeignKey("QuadSource.serial")` at
    `:115-117`) and `Tr_serial` (the commented-out key at `:124`).
  - `GkSourcePolicyData.py:68-70` holds `wavenumber_exit_serial`.
  - The flags are at `wavenumber.py:178` and `integration_metadata.py:16`.
- **The index.** 97 rows now. 06 closes one, so it goes to **96**, plus any issue it opens. Rows
  are counted, not computed.

**Carried from the 03 session.**
- **A claim about code that is not run is checked against the code it depends on.** 03's L5 said
  that a missed exit time's `store_id` becomes `IS NULL`. `DatastoreObject.store_id` raises, and
  the review had to correct two boards.
  - For 06, the claims are that a foreign key is not fingerprint identity, that nothing reads
    `stepping_mode`, and that `Tq` shares its shard.
  - Each rests on a test run or a quoted line, not on inference.
- **A log names only sections it contains.** Log 03 pointed at a "Verification performed (the
  replay)" section that does not exist.
- **Mutations.**
  - A mutation record lists every failing test, with counts, and says where a failing test has
    sub-tests.
  - The replay is done twice: before the commit from the final log file, and after it from
    `git show HEAD:<log>`, reported with no amend.
- **Residue.** The agent still follows README §5 rule 4, recording and not fixing. The user's
  rule (2026-09-29) is that small residue found in review is fixed afterwards, in a follow-up
  commit of its own. That is the orchestrator's job, not the agent's.
- **Scratch files.** The agent works in its own scratch subdirectory, not the session
  scratchpad's top level.

## 1. Before you dispatch

1. Record the branch and `HEAD`. It must be the commit that adds this note.
2. `python -m RunRegistry list`: nothing `running`.
3. Snapshot `var/datastores/`, `var/runs/` and the root `physics-test-n20-lambdacdm-zend0p1*.sqlite`
   read-only, with `orch_04_snapshot.py` in the session scratchpad. Write it again if it is
   missing (SHA-256, size and mtime of every file; the directory list), and never put it in the
   tree. The 03 session's snapshot held 46 files and 10 directories.
4. Baselines, board §5.8: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39, Datastore 343,
   LiouvilleGreen 148 (1 skipped), RunRegistry 199. `e224e55` added sub-tests to
   `ReaderToleranceAgreementTestCase` and no test methods, so ComputeTargets stays at 552.
   Re-measure all six on `HEAD`, serially.
5. `git status --short` and `git diff --cached` are both empty, and `git worktree list` shows this
   checkout alone.

## 2. Dispatch

One fresh-context subagent (**Sonnet**, as the prompt recommends). Give it:
- the prompt;
- the README, with D5;
- the audit, `docs/datastore-integrity-audit.md` §1, §2.1, §2.2 and §2.6;
- `docs/datastore-integrity-audit/fk_sweep_probe.py` and `tq_fk_probe.py`;
- the issue's entry on the store-fingerprint board;
- log 03's "Observations not acted on" and "State handed to the next prompt";
- `HEAD` and the baselines;
- §0 of this note.

Tell it plainly:
- **one commit**, with its log at `logs/06-correct-the-foreign-keys.md`, in README §5.1's form, with
  prompt 06 §7's additions:
  - the evidence that `Tq` shares `QuadSource`'s shard, quoted;
  - the witness diff **in full**, pasted from `diff`;
  - the fingerprint test run, quoted;
  - which store or stores S4 checks, the rows it relies on for the "before S1" failure, and what it
    cannot see;
- in the same commit:
  - the board: its §1 row, S1–S5, §3.1, and a new §5 subsection `5.9 After prompt 06`;
  - the store-fingerprint board: the issue moves to its §4, with a closing note naming this commit
    by subject;
  - `docs/OPEN_ISSUES.md`: its row deleted, and the count and date corrected;
- **the files it may change:**
  - `Datastore/SQL/ObjectFactories/QuadSource.py`: the `Tq_serial` and `Tr_serial` column
    declarations in `register()`, and the comments beside them;
  - `Datastore/SQL/ObjectFactories/GkSourcePolicyData.py`: the `wavenumber_exit_serial` column
    declaration in `register()`;
  - `Datastore/SQL/ObjectFactories/wavenumber.py` and `integration_metadata.py`: the
    `"stepping"` value in `register()` only;
  - `Datastore/tests/real_store_fixtures.py`: `_UNDECLARED_REFERENCES` only (S3);
  - `Datastore/tests/test_schema_builder.py`: the `WITNESS` constant and the module docstring
    (S5, D5);
  - `Datastore/tests/schema_description.py`: its module docstring only (this note, §0);
  - one new witness, `Datastore/tests/data/schema_at_datastore-integrity-06.json`, captured with
    `schema_description.py`'s `__main__`;
  - one new test module under `Datastore/tests/`;
  - the log, the two boards and the index.

  **Nothing else:**
  - not `schema_at_base.json`, which is never overwritten;
  - no `build()`, `store()`, `read_batch` or `inventory_records`;
  - no other existing test, and not the golden fingerprint `RunRegistry/tests/data/full_store_fingerprint.json`;
  - not `schema.py`, `ShardedPool.py`, `Datastore.py` or `store_inventory.py`;
  - not `_drop_order`, `main.py` or any `extract_*.py`;
  - no audit probe, and nothing in `orchestrator/`;
- **no Ray, and nothing opens anything under `var/`**, or a copy of anything under it. Every store is
  in a temporary directory;
- S4's "before S1" failure needs the fixed rows §0 describes, and must not rely on
  `relabel_serials`;
- every mutation diff is pasted from `git diff`, and replayed twice as §0 says;
- run `black`;
- prompt 06 §5's stop conditions mean *stop and ask*. So does any existing test failing other than
  `test_schema_builder`'s two witness tests before the constant is repointed, and so does a witness
  diff line outside the four classes.

## 3. The review

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only the permitted files, plus the one new witness and
     the one new test module.
   - `git diff HEAD~1 HEAD -- main.py extract_*.py Datastore/SQL/schema.py Datastore/SQL/ShardedPool.py Datastore/SQL/Datastore.py Datastore/store_inventory.py Datastore/tests/data/schema_at_base.json RunRegistry/ docs/ prompts/datastore-integrity/orchestrator/`
     is empty, except `docs/OPEN_ISSUES.md`.
   - `--diff-filter=M` over `*/tests/*` shows exactly `real_store_fixtures.py`,
     `test_schema_builder.py` and `schema_description.py`. Read each hunk: the map, the constant
     and docstring, and a docstring, and nothing else.
   - Every factory hunk is inside `register()`, on the four named columns or flags.
2. **S1, S2.** `Tq_serial` targets `TkNumericIntegration.serial`. `Tr_serial` has no foreign key,
   and its comment names `TkNumericIntegration` and the cross-shard reason.
   `GkSourcePolicyData.wavenumber_exit_serial` targets `wavenumber_exit_time.serial`, with `index`
   and `nullable` unchanged.
3. **S3.** The map has `"Tr_serial": "TkNumericIntegration"` and no `GkSourcePolicyData` entry.
   The `QuadSourceIntegral.source_serial` entry is unchanged.
4. **S4.** The new test runs `PRAGMA foreign_key_check` on every shard.
   - Its rows put a `Tq_serial` that is not a `QuadSource` serial on its shard.
   - Its docstring names the three unchecked cross-shard references: `Tr_serial`,
     `QuadSourceIntegral.source_serial` and `shard_keys`.
   - Check out the parent's `QuadSource.py` over the new test, and confirm it fails. Then restore.
5. **S5, D5.**
   - The witness file is new, and `schema_at_base.json` is byte-identical at both commits.
   - `WITNESS` names the new file, and both docstrings state D5's rule.
   - Re-capture a witness from `HEAD` with `schema_description.py` into the scratchpad. It must be
     byte-identical to the committed one.
   - Diff the two witnesses yourself. It should be the 9 hunks and 30 lines §0 lists, and it must
     match the log's paste.
6. **The fingerprint is unmoved.** `full_store_fingerprint.json` is unchanged, and
   `test_store_fingerprint` passes.
7. **Replay the mutations** (i)–(iv) from the committed log with plain `git apply`. Each gives its
   logged failures, and (i) fails S4 on the rows §0 asks for. Revert each, leaving the tree clean.
8. **Boards.** The issue is on the store-fingerprint board's §4, with a closing note naming this
   commit. The index count is 96, plus any opened, counted as rows, with the date. The campaign
   board has its §1 row, S1–S5, §3.1 and §5.9.
9. **`var/`, suites.**
   - The snapshot is identical.
   - Run all six suites serially. Datastore rises by exactly the tests added, and the others match
     §1.4.
   - `black --check` is clean.
   - `ray.is_initialized()` is false after the new module.

## 4. After the review

Record it on the board, after the review of 03. Any small residue the review finds is fixed in a
follow-up commit of its own, closing its issue in that commit (the user, 2026-09-29).

Then:
- **07 is next** (README §2: 07 follows 05 and 06), with its own orchestrator note carrying this
  review's findings. 07 is the first prompt to add a column and change a lookup key. It adds its
  witness beside 06's, by the procedure 06 set. It is also the first whose foreign key constrains
  a drop, so `_drop_order` is its concern.
- Prompt 10 remains the first real run.
