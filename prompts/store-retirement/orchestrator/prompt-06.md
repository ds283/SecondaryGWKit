# Orchestrator — prompt 06, fix the three residuals, and close the campaign

Read [`../README.md`](../README.md) first: §5, §6.2 D0 and D5, and §6.4. **You do not write
code.** You may *replay* a mutation the log records, with `git apply`, and then revert it.

**The prompt:** [`06-fix-the-residuals-and-close.md`](../06-fix-the-residuals-and-close.md)
**Board items:** R13–R16 · **Closes:** `[03-the-package-docstring-still-says-the-registry-deletes-nothing]`,
`[04-amend-calls-true-1-and-1-0-identical]`, `[04-no-test-amends-a-value-shaped-like-the-marker]` ·
**Gate:** prompts 01–05 have landed and been reviewed (05 in `0075f72`)

## 0. What makes this prompt unusual

It is the campaign's last prompt, and it does two jobs. It fixes the three issues that are this
campaign's own code, which the user decided on 2026-09-26 should not be left behind (README §6.4).
Then, once you have reviewed it, **you** record the closure, in the commit that records your
review, as `store-fingerprint`'s was (`42d4910`). The prompt's agent does not write the closure.

The review turns on three properties.
- **Only the refusal decision moves.** The canonical-text comparison changes *whether*
  `amend_sidecar` refuses, and nothing else. What it writes when it does not refuse is what it
  wrote before. Check this with a probe (§3 check 2), not by reading.
- **The marker test would catch a flattening.** Mutation (iii) returns the bare value from
  `_amend_slot`. The review of 04 saw that such a change could pass every committed test. After
  this prompt it must not. The case that proves it is the one where the bare value is itself a
  well-formed marker, `{"present": false}`, which the validator accepts. There, only an
  `assertEqual` on the whole marker fails. Check that the test asserts it that way.
- **The end state of `var/` still holds.** Nothing in this prompt touches `var/`. The closure
  says that the campaign's two retirements and one amendment are what is on disk, so you check
  that on the real files, read-only, before you write it.

**Carried from the review of prompt 05.** None of the three user outputs showed an exit code.
Nothing here is run by the user, so that does not recur.

## 1. Before you dispatch

1. Record the branch and `HEAD`. `HEAD` must be the commit that added the prompt and these notes.
2. `python -m RunRegistry list`: nothing `running`.
3. **Snapshot, read-only,** as for prompt 05: everything under `var/datastores/` (size, mtime and
   SHA-256), everything under `var/runs/` (size and mtime), and the repository root's
   `physics-test-n20-lambdacdm-zend0p1*.sqlite`. Reuse `orch_05_snapshot.py` if this session has
   it; otherwise write it again in the session scratchpad under an `orch_` prefix, never in the
   tree. Then check the snapshot against the campaign's end state:
   - the three sidecars equal `logs/05-tombstones.json`, by `json.load`;
   - the live A3 store's five `.sqlite` files have the SHA-256, size and mtime in log 05's C2
     table;
   - `var/datastores/` holds eight files: the live store's five and its sidecar, and the two
     tombstones, the backup's alone in its directory.

   **A difference is a stop.** Bring it to the user before dispatch. Something wrote under `var/`
   after 05, and the closure cannot say what 05 left until that is explained.
4. **The baselines.** From the board's §5 after prompt 04: AdaptiveLevin 32, ComputeTargets 552,
   CosmologyModels 39, Datastore 206, LiouvilleGreen 148 (1 skipped), RunRegistry 190. No code has
   changed since `66617c9`. Re-measure all six on `HEAD`, in the checkout itself, with README §7's
   command.
5. **The shipped behaviour, as a probe.** In the scratchpad, on a store built in a temporary
   directory, confirm the defect on `HEAD`: `1` cannot be amended to `true`. Also confirm that a
   marker-shaped value reads back unambiguously, as the review of 04 found. Then amend one field
   `"a"` → `"b"`, on a store built at a fixed scratchpad path, and keep the sidecar it writes.
   Mask what differs from one fresh store to the next: `store_id`, `created`, and every history
   entry's `when`, `git_head` and `git_dirty`.
   Keep the probe too; §3 check 2 reruns it.
6. `git status` is clean, apart from untracked paths that are not this campaign's (today
   `docs/datastore-integrity-audit*` and `prompts/datastore-integrity/`). Tell the agent to leave
   them alone.

## 2. Dispatch

One fresh-context subagent (**Sonnet**, as the prompt recommends), in the foreground. Give it:
- the prompt;
- the campaign README;
- log 04;
- the board's review of prompt 04;
- `HEAD` and the baselines.

Tell it plainly:
- **one commit**, with its log at `logs/06-fix-the-residuals-and-close.md`. The board, the index
  and the code go in the same commit;
- run `black`;
- **nothing under `var/` is opened**, read-only included. Every test builds in a temporary
  directory, passes `runs_root`, and passes `--runs-root` on the command line;
- **`RunRegistry/tests/test_store_amend.py` is not changed at all**, and no other existing test is
  either. The new tests go in `test_store_amend_residuals.py`;
- the docstring paragraph is the prompt's §2.1 text **exactly**, re-wrapped only;
- in `stores.py`, the only hunks are in `amend_sidecar`: the comparison, the message if it adds a
  clause, and item 9 of its docstring. They also include a private helper, if the agent makes one;
- mutations (i)–(iii) are recorded as diffs **that apply with plain `git apply` from the
  repository root**: fenced ```` ```diff ```` blocks, not indented, with correct hunk headers,
  each checked with `git apply --check` against the final commit. They are never committed;
- **it does not write the closure.** The board's header says the work is done and that the user
  closes it after review. The README and the index's closure sentence are yours;
- do not touch `orch_*` files, or untracked paths that are not its own;
- the prompt's §5 stop conditions mean *stop and ask*.

## 3. The review — nine checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only:
   - `RunRegistry/__init__.py` and `RunRegistry/stores.py`;
   - the new `RunRegistry/tests/test_store_amend_residuals.py`;
   - the log, the board and `docs/OPEN_ISSUES.md`.

   `git diff HEAD~1 HEAD --diff-filter=M -- RunRegistry/tests/` is empty.
   `git diff HEAD~1 HEAD -- RunRegistry/__main__.py Datastore/ tools/ CLAUDE.md docs/handover/` is
   empty. In `__init__.py`, the one hunk is the docstring paragraph. In `stores.py`, every hunk is
   inside `amend_sidecar`, or is the new helper.
2. **Only the decision moved.** Re-run §1.5's probe on the commit:
   - `1` → `true`, `1` → `1.0`, `true` → `1` and `{"k": 1}` → `{"k": true}` are accepted;
   - `1` → `1` and a key-reordered object are refused, with the tree unchanged;
   - **the `"a"` → `"b"` sidecar written on the commit, at the same path and masked the same way,
     equals the one kept in §1.5**, by `json.load` and in its canonical text. That shows the
     accepted path writes what it wrote before.

   The message still contains `identical, after a JSON round trip`, and docstring item 9 says how
   the comparison is made. The log records the `NaN` change.
3. **The docstring.** The paragraph is the prompt's §2.1 text, word for word, compared after
   joining lines. `grep -n "deletes nothing" RunRegistry/__init__.py` is empty. `list_runs`'s
   docstring is unchanged. The first paragraph's claims now agree with `CLAUDE.md`'s run-registry
   paragraph, and with the `stores.py` and `__main__.py` docstrings.
4. **The tests.**
   - Run the new module twice.
   - The comparison tests assert types with `assertIs(type(…), …)`, not only `assertEqual`.
   - The marker tests assert each whole marker with `assertEqual`, and walk the history back to
     the four-value sequence.
   - Every `amend_sidecar` call passes `runs_root`, and every command passes `--runs-root`.
   - `python -c "import RunRegistry, sys; print('ray' in sys.modules, 'sqlalchemy' in sys.modules)"`
     prints `False False`.
5. **Replay mutations (i)–(iii)** with plain `git apply`, from an empty working directory in the
   scratchpad, with the repository on `PYTHONPATH`. Each gives its recorded failures.
   - For (iii), confirm that the `{"present": false}` step fails **on the marker assertion**, and
     not only because the validator refuses a malformed marker elsewhere.
   - Revert each, leaving the tree clean and the working directory empty.
6. **The suites.** Re-run all six. `RunRegistry` rises by exactly the new tests, and the other five
   match §1.4. `black --check` is clean on the touched files.
7. **The records.**
   - **The board.** The three entries are in §4, each keeping its text and its `Assigned` line,
     with a `Closed (date) by prompt 06` paragraph. §3 holds four, and its opening count is right.
     The §1 row for 06, R13–R16 and §5 are updated. The header says the work is done, **not** that
     the campaign is closed.
   - **The index.** Its §1.14 has four rows and a sentence about prompt 06. The count is 101, and
     **Last updated** is the commit's date.
8. **`var/`.** Re-take §1.3's snapshot. It is identical to §1.3's, every mtime included.
9. **The commit.** One commit, in `CLAUDE.md`'s form. The tree is clean afterwards, apart from the
   untracked paths that are not this campaign's.

## 4. After the review — the closure

If all nine checks pass and the review opens no issue, the closure proceeds as the user decided on
2026-09-26 (README §6.4). **If the review opens anything, bring it to the user first.** An issue
found in the last prompt's code is the user's call: the campaign can close with it open, or fix it
before closing.

The closure is **one commit**, holding your review and the closure together, as `42d4910` did:
- **The board.** Your review, after the §1 table, as for 01–05. Then the header's status becomes
  **"COMPLETE — 6 of 6 prompts landed. Closed by the user on <date>."** Add a dated paragraph
  saying that the charter is met:
  - a store is retired through the registry, and its sidecar stays behind as the record;
  - the two stores are retired, and the live A3 store is byte-identical;
  - the one present-tense claim is corrected;
  - the campaign's own residuals are fixed;
  - four issues remain open in §3, unassigned, each in the index for its owner.
- **The README.** One line in its header, beside the written dates: **"Closed** by the user on
  <date>, with all six prompts landed; see the board." Update §2's row for 06 to landed.
- **`docs/OPEN_ISSUES.md` §1.14.** One sentence: **"The user closed the campaign on <date> at
  6 / 6."** Its four rows stay open, each recorded there for its owner. No row changes, so the
  count stays at 101. Set **Last updated** if the date has moved.

Subject: "Close the store retirement campaign".
