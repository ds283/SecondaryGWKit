# Orchestrator — prompt 03, state each closed-store prefix once

Read [`../README.md`](../README.md) §0.2 (item 3), §1, §4, §5 and §6.1 (decision 3). Then read:
- `datastore-generic`'s board, §3.1, the entry `[04-closed-store-refusals-repeat-their-prefix]`;
- this board's review of 02 (§1), with its "Handed on" line.

The conventions are [`prompt-01.md`](prompt-01.md)'s, carried below.

**You do not write code.** You may:
- run the suites and the tests the log names;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process check or a probe from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`03-state-each-closed-store-prefix-once.md`](../03-state-each-closed-store-prefix-once.md)
**Board items:** P1–P2 · **Closes:** `[04-closed-store-refusals-repeat-their-prefix]`, on
`datastore-generic`'s board · **Narrows:** nothing · **Opens:** nothing expected · **Model:**
Sonnet, as the prompt recommends.

**Gate:**
- 02 landed (`4c34f13`) and was reviewed (`f8fafbf`).
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means `Datastore/`, `RayTools/`, `config/`, `utilities.py`, `main.py`,
    `extract_*.py`, `RunRegistry/`, `ComputeTargets/`, `tools/` or `docs/handover/`.
  - If a later commit touches code, re-check every fact below that it could move, and hand the
    agent the corrections as an addendum.
- One prompt at a time in this checkout.

## 0. What makes this prompt unusual

**It is the campaign's last prompt, and it closes the campaign.** Its change is small: three
`raise`s lose a prefix, one parameter goes, and the reader gains a `try`. The records are larger
than the code: the issue closes, and with it the last of `datastore-generic`'s issues, so this
board, the README header, `prompts/INDEX.md` and the index's §1.18 and §1.19 all say so.

**The orchestrator adds one fixture and one breakage to the prompt's test plan.** This is the
orchestrator's addition, not a reading of the prompt, and the log records it as an
IMPLEMENTATION CHOICE made at the orchestrator's direction. Pass it on as such.
- `_read_closed_store` has three `raise`s: the `shards` table cannot be read; the resolver refuses
  a record; and the file problems (a missing shard, a symbolic link, a non-regular file, two
  serials sharing a file). The prompt's five tests use two fixtures, an unusable record and a
  dropped `shards` table. Neither reaches the third `raise`.
- **Measured:** with the rough change, the file-problem `raise` prefixed again with
  `Cannot open sharded datastore "<primary>": ` fails **no test in any suite**
  (`measure-03/out/suites.txt` §C, `measure-03/out/gap_problems_branch.diff`). The reader's
  missing-shard test counts only `Cannot read sharded datastore`, so a second prefix of another
  verb passes it.
- **So:** test 5, `test_read_closed_store_states_no_prefix`, covers **three** fixtures: an unusable
  record, a dropped `shards` table and a **missing shard file**. Add breakage **(v)**, the
  file-problem `raise` prefixed again with the same literal as (i) and (ii), which must fail
  test 5. The count of new tests stays 5.

**What moves on purpose:**
- `ShardedPool._read_closed_store`: its signature (no `verb`), its three `raise`s, its docstring.
- Its three callers in `ShardedPool.py` stop passing a verb, and nothing else in them changes.
- `store_reader.open_read_only`: the call is wrapped in `try`/`except RuntimeError`, raising
  `_refuse(primary_path, str(e)) from e`.
- One test changes, named by the prompt: `test_store_reader.TestRefusals.
  test_missing_shard_is_refused_by_read_closed_store`.
- One module is added, `Datastore/tests/test_closed_store_refusals.py`, with 5 tests.
- The issue closes; the index goes from 105 to 104; the campaign closes.

**The messages, measured** (`measure-03/out/m3_compare.txt`, from `scripts/m3_messages.py` at an
export of `f8fafbf` with and without `measure-03/rough_closed_store.diff`). For each of the three
fixtures, the messages of `_read_closed_store`, `closed_store_files`, `delete_store`,
`copy_store`, `move_store` and `open_read_only`: all 18 are as expected.
- `closed_store_files`, `delete_store`, `copy_store` and `move_store` say `sharded datastore` twice
  before and once after. Only the repeated `Cannot <verb> sharded datastore "<path>": ` goes.
- `open_read_only` says it once before and once after, and gains `. Nothing was read or repaired`.
- `_read_closed_store`'s own message loses its prefix and says it zero times.

**Two things the measurement shows that are not stops:**
- **The resolved path leaves some messages.** The stores are built in an unresolved temporary
  directory (`/var/…` on macOS, which resolves to `/private/var/…`).
  - The wrapper's prefix names the primary as given, `<T>`. The inner prefix, now gone, named it
    resolved, `<R>`.
  - So for a dropped `shards` table, `delete_store`, `copy_store` and `move_store` no longer name
    the resolved path anywhere: their reason, `its shards table could not be read (…)`, names no
    file.
  - The resolver's reason names the primary resolved (`Shard #0 of primary database "<R>/…"`),
    so `test_absolute_shard_record_refused`'s check of `primary.resolve()` still passes.

  This is the removed repetition, not a changed reason.
- **`_read_back`'s refusal is never reached by a test.** `_read_back` reads the copy back through
  `_read_closed_store` mid-operation, and `_failure_message` wraps it as
  `copy of sharded datastore "<src>" to "<dst>" failed at step "read back the destination":
  RuntimeError: <reason>`. Today `<reason>` starts with `Cannot copy sharded datastore "<file read
  back>": `. After the change it does not. The step names which file was read back, and the
  message lists the files at the destination. The prompt does not change `_failure_message`, and
  the agent must not. Record it as an observation.

**The suites, measured with the rough change** (`measure-03/out/suites.txt`, at an export of
`f8fafbf` made a git repository). Code alone: one test errors,
`test_missing_shard_is_refused_by_read_closed_store`, on `TypeError: … takes 1 positional argument
but 2 were given`. Every other test passes, RunRegistry's two named tests included. With the
prompt's change to that test, `test_store_reader` runs 15, OK.

**What each breakage also fails**, beyond the tests the prompt names. A breakage that fails more
is not wrong; the log lists every failure.
- **(iii)**, the reader's `try` removed: also the existing
  `test_store_reader.TestRefusals.test_primary_with_no_shards_table`, because the bare
  `shards`-table reason names no file. The new test 4 will fail too, since it reads through the
  reader.
- **(iv)**, `_plan_deletion` not wrapping: also 54 subtests of `test_delete_store` (§E), and the new
  test 4.
- **(v)**: no existing test fails (§C). (i) and (ii) were not probed against the existing tests;
  the new tests are what pin them.

**The facts, checked by the orchestrator on 2026-10-07 at `f8fafbf`** (the code is `4c34f13`'s:
`git diff 4c34f13 f8fafbf` touches no code). Pass them on.
- **The prompt's line numbers have moved by +27**, because 01 added `ShardedPool` code above them.
  Find each by name. At `f8fafbf`:
  - `copy_store` `:2535`, `move_store` `:2556`, and the docstrings of `closed_store_files` and
    `delete_store` `:2574-2646`;
  - `_read_closed_store` `:2648-2688`, its three `raise`s at `:2663`, `:2671` and `:2680`;
  - `_plan_relocation` `:2690-`, the call `:2720`;
  - `_read_back` `:2826-2835`, the call `:2830`, called from `_relocate_store` at `:2868` and
    `:2888`; `_failure_message` `:2894-2926`;
  - `_plan_deletion` `:2928-`, the call `:2971-2973`.

  `store_reader.py` has not moved: `_VERB` `:61`, `_refuse` `:97-100`, `open_read_only` `:176-`,
  the call `:206`. Neither has `test_store_reader.py` (`:304-312`; `assert_refused_naming`
  `:272-279`). `RunRegistry/stores.py:1567-1568` holds `_SHARDS_TABLE_UNREADABLE`, used at `:1818`
  and `:1863`. `RunRegistry/tests/test_store_copy_move.py:328` and
  `RunRegistry/tests/test_store_retire.py:790` are as the prompt says.
- **Exactly four callers.** `git grep -n '_read_closed_store(' -- '*.py'` gives the three in
  `ShardedPool.py` and `store_reader.py:206`. The `.md` files that mention it are records and are
  not touched.
- **The rough diff applies at `HEAD`.** `measure-03/rough_closed_store.diff` is the 03 hunks of
  `measure/rough_change.diff`, regenerated against `f8fafbf`. Its resolver branch is
  `raise RuntimeError(str(e)) from e`, a re-raise that adds nothing. Letting the resolver's
  `RuntimeError` propagate is equally right. Either is an IMPLEMENTATION CHOICE; breakage (i) then
  adds the wrap back.
- **The prompt's fixture.** "the absolute-record fixture of `test_absolute_shard_record_refused.py`
  (a shard record `../outside.sqlite`)" names two fixtures. That module's store B records absolute
  paths. A `../outside.sqlite` record is the issue's measurement, built as
  `test_copy_move_store.py:249` builds it (`write_hand_built_primary(src, {0:
  "../outside.sqlite"})`). Both reach the resolver's refusal; either will do.
- **`missing_ok`** stays keyword-only, and only `_plan_deletion` passes it.
- **The guard.** `ShardedPool.py` and `store_reader.py` are layer files. The new docstring names
  no project table. Tests are outside the guard.
- **Not touched, and staying:** `_failure_message`, `_deletion_failure_message`, the resolver
  (`_resolve_shard_rows`, `resolve_shard_path`), `_shard_file_problems`, the constructor's own
  `_check_shard_files` refusal (whose `Cannot open sharded datastore` prefix is its own),
  `RunRegistry/`, `tools/`, every other test and every store under `var/`.
- **Expected counts after 03:**
  - Datastore 739 = 163 + 185 + 200 + 191 (`test_closed_store_refusals` sorts into `[a-f]`);
  - RunRegistry 280; ComputeTargets 693; AdaptiveLevin 32; CosmologyModels 39;
    LiouvilleGreen 148 (1 skipped).
- **The index:** 105 now, 104 after. §1.19's table is then empty, and §1.18 and §1.19 say both
  campaigns' issues are closed.
- **`prompts/INDEX.md`'s row** for this campaign still reads "**written** — 3 of 3 written, none
  landed", with 3 issues. Its rule is that a status changes when the board closes a campaign. 03
  rewrites the row: closed 3 / 3 with the date, the last-touched date, and 0 open issues.

**What the review exists to establish.**
- **(E1) The change.** `_read_closed_store` raises each reason bare and takes no `verb`. The
  three callers in `ShardedPool.py` keep their wrapping exactly. The reader wraps with `_refuse`.
- **(E2) Nothing else moves**, by `ast`, docstrings stripped: in `ShardedPool.py`,
  `_read_closed_store` and the three callers, each by its call; in `store_reader.py`,
  `open_read_only` by the `try`; in `test_store_reader.py`, the one test.
- **(E3) The messages.** `m3_messages.py` at the agent's commit gives 18 messages as expected
  against the parent, and the agent's messages equal the rough change's
  (`messages-rough.json`) byte for byte.
- **(E4) The pins.** (i)–(v) each fail the named tests.
- **(E5) The records.** The issue is in `datastore-generic`'s §4 with a dated "Closed" line naming
  the fourth caller. It is gone from §3.1 and the index, which counts 104. The campaign is closed
  on this board, the README header, `prompts/INDEX.md` and the index's §1.18–§1.19.

**Conventions.** All of these bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, and show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the diff and the tests show.
- **The SHA.** The agent writes "this commit", and the orchestrator substitutes the SHA in its
  follow-up.
- **Suites.**
  - Use README §7's command, write the output to a file, and grep the verdict from it.
  - Run each suite **in the foreground**, serially, within the turn.
  - Run the Datastore suite in module groups (`-p 'test_[a-f]*.py'`, `[g-o]`, `[p-r]`, `[s-z]`),
    and add the counts. A Bash call is capped at ten minutes.
  - Under load, four tests are known to fail on wall clock or chance:
    - `test_tk_wkb_phase.TestCost.test_wall_time_per_object`;
    - `test_background_tau.test_build_cost`;
    - `test_numeric_phase_cut.test_the_print_is_gated_both_ways`;
    - LiouvilleGreen's `test_3bessel_analytic.test_JJJ`.

    Re-run a failure alone before calling it real. If it persists alone, settle it on an export of
    the tree before the prompt.
  - In a git worktree, RunRegistry reports one skip (`test_pipeline_adoption`, whose manifest is
    under the gitignored `var/`). In this checkout it does not.
  - **An export is not a repository.** `test_inventory_retired`'s two grep tests run `git grep`,
    and fail in a plain `git archive` export with "not a git repository". To run the Datastore
    suite at an export, `git init` it and commit everything first.
- **Breakage diffs** are recorded so that `git apply` replays them from a clean tree at the agent's
  commit. Check each with `git apply --check` and `-R --check`, and replay in `bash`, not zsh.
  zsh does not word-split an unquoted `$VAR`, so a list of test modules in one variable runs as
  one bad name.
- **Dead helpers** that only the changed code or tests used go in the same commit, after a grep
  (IMPLEMENTATION CHOICE). This includes an import left unused. `_VERB` stays: `_refuse` uses it.
- **Scratch.** The agent uses its own subdirectory of the session scratchpad, never `/tmp`.
  Python's `tempfile` in tests is fine.
  - A scratch script never names a helper `inventory`: a repo-wide grep test forbids
    `def inventory(` in any tracked file.
  - **No scratch `.py` file goes under `Datastore/` or `RayTools/`**, even for a moment: the guard
    globs the filesystem and would scan it.
- **Ray.** The agent starts no Ray. ChamPBH's cluster, if running, is left alone.
- **Stop conditions are real.** A stop goes to the user through the orchestrator, and the agent
  resumes with `SendMessage`.
- **Other sessions** may land commits on this branch. Review against the agent's commit, and check
  `git log` before measuring at `HEAD`.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` must be empty.
   - `git worktree list` must show this checkout alone.
   - `git log --oneline f8fafbf..HEAD` must touch no code; otherwise see the gate.
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`.
3. **Ray.** Record `pgrep -lf 'gcs_server|raylet|ray::'`. No listed process may come from this
   repository's `venv`.
4. **Baselines.**
   - First check that `./venv/bin/python -c 'import scipy.sparse.linalg'` succeeds.
   - The last measurement was at `4c34f13`, on 2026-10-07, at 02's review (board §5.3 and the
     review's check 9): AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped),
     ComputeTargets 693, RunRegistry 280, Datastore 734 (158 + 185 + 200 + 191), all OK.
   - Re-measure all six at `HEAD`, serially, with the load average, and give the agent the
     numbers.
   - **Run the full Datastore suite before dispatch**, so that a committed measurement file
     tripping a repo-wide grep test (this note's `measure-03/` is new, with two `.py` scripts)
     shows at once.
5. **The index.** It is 105 now, and 104 after. Tell the agent both.
6. **The pre-images**, for check 3. Record `git rev-parse HEAD:<path>` for
   `Datastore/SQL/ShardedPool.py` (`49d6aca` at `f8fafbf`), `Datastore/store_reader.py`
   (`dd7e4a1`) and `Datastore/tests/test_store_reader.py` (`e16c472`).
7. **The parent side of check 4.** Export `HEAD` to the scratchpad (`git archive HEAD | tar -x -C
   …`, with the venv symlinked). Run `measure-03/scripts/m3_messages.py <export> <out>/parent.json`
   from the export. It must equal `measure-03/out/messages-parent.json`; if it does not, the tree
   has moved, and the gate applies.

## 2. Dispatch

Dispatch one fresh-context subagent, on **Sonnet**. Give it:
- the prompt;
- the campaign README, this board (including the reviews of 01 and 02), `datastore-generic`'s
  board §3.1 entry and its §4 (for the form of a closed entry), and 02's log as an example of the
  log's form;
- `measure-03/rough_closed_store.diff`, `measure-03/scripts/` and `measure-03/out/`, and the
  `test_store_reader.py` hunk of `measure/rough_change.diff`;
- `HEAD`, the baselines and the index counts;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/` beyond `measure-03/`.

Tell it that the prompt governs, with §0's addition of a third fixture to test 5 and breakage (v),
and that the agent need not reproduce the rough diff exactly.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/03-state-each-closed-store-prefix-once.md`, in
  README §5.1's form, with the prompt's §7 additions. For §3.2's messages it may run
  `m3_messages.py` and `m3_compare.py`, and paste their output.
- **The files it may add or change are the prompt's §6 list, and nothing else.** That includes the
  README header and `prompts/INDEX.md`. In particular, it must not touch:
  - anything under `docs/` other than `OPEN_ISSUES.md`;
  - `orchestrator/`, `measure/` or `var/`;
  - in `ShardedPool.py`, anything but `_read_closed_store` and its three calls;
    `_failure_message` is not touched;
  - in `store_reader.py`, anything but the one call;
  - any test but the one named and the new module.
- **It runs no build and starts no Ray.**
- **Stop and ask** on any of the prompt's §5 conditions. A message whose reason changes beyond the
  removed repetition is a stop. The resolved path leaving the shards-table messages is not (§0). A
  guard hit is a stop, never an allow-list entry. A test outside the plan that fails is current
  behaviour moving, never a reason to edit it.
- **Run the Datastore suite in four groups**, in the foreground.

## 3. The review — nine checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only the prompt's §6 list.
   - The diff over `docs/` (but `OPEN_ISSUES.md`), `config/`, `utilities.py`, `main.py`,
     `extract_*.py`, `RunRegistry/`, `tools/`, `var/`, `orchestrator/`, `measure/`, every factory,
     and every module of the layer but `ShardedPool.py` and `store_reader.py`, is empty.
2. **E1, by reading.** Read `_read_closed_store`, its three callers' calls and `open_read_only`'s.
   - Each `raise` gives the reason alone, and the resolver's and the file problems' text is
     unchanged.
   - No `verb` remains; `missing_ok` is keyword-only.
   - The reader's `except` catches `RuntimeError` only, and raises `_refuse(primary_path, …)`
     `from e`.
   - The docstring says that it raises `RuntimeError` naming the problem, and that each caller
     states the operation. It names no project table.
3. **E2, by `ast` against the pre-images, docstrings stripped.**
   - `ShardedPool.py`: only `_read_closed_store`, `_plan_relocation`, `_read_back` and
     `_plan_deletion` (and the class that holds them) differ. Each caller differs only by its call.
   - `store_reader.py`: only `open_read_only`, by the `try`.
   - `test_store_reader.py`: only `test_missing_shard_is_refused_by_read_closed_store`.
4. **E3, by running.** At an export of the agent's commit, `m3_messages.py`; then `m3_compare.py`
   against the parent side gives 18 as expected and 0 unexpected, and the output equals
   `messages-rough.json` byte for byte. `git grep -n '_read_closed_store(' -- '*.py'` shows four
   calls, none with a verb.
5. **The tests, by reading.** Read `test_closed_store_refusals.py` in full.
   - "Once" is checked as the prompt says: the message begins with the operation's prefix, and
     holds `sharded datastore` exactly once.
   - Test 2 names the destination. Test 3 ends `Nothing was read or repaired`. Test 4 covers
     `delete_store`, `copy_store` and `open_read_only`, each holding "its shards table could not
     be read". Test 5 covers all three fixtures, including the missing shard.
   - No test passes on a call that does not raise.
   - Run the module, `test_store_reader`, `test_absolute_shard_record_refused`, `test_delete_store`
     and `test_copy_move_store`.
6. **The changed test, by reading.** It calls `_read_closed_store(self.store.primary)`. It asserts
   that the reader's message contains that message, and that `Cannot read sharded datastore`
   appears once. Its comment says the refusal is `_read_closed_store`'s under the reader's one
   prefix.
7. **E4, the pins.** In `bash`, replay each breakage diff with `git apply`, and run the named
   tests:
   - (i) fails tests 1, 2, 3 and 5; (ii) fails tests 4 and 5;
   - (iii) fails test 3 and the changed test, and also test 4 and `test_primary_with_no_shards_table`;
   - (iv) fails test 1, and also test 4 and `test_delete_store`;
   - (v) fails test 5.

   Each log entry lists every failure, not only the named ones. Revert each with `git apply -R`,
   and check that `git status` is clean.
8. **E5, the records.**
   - The issue is in `datastore-generic`'s §4 with a dated "Closed" line naming this campaign, the
     change and the fourth caller. It is gone from §3.1 and from the index.
   - The index is 104 by counting, matching its header. §1.19's table is empty, and §1.18 and
     §1.19 say that both campaigns' issues are closed.
   - This board has the §1 row, P1–P2 ✅, §5, and the header CLOSED — 3 of 3. The README header
     says the campaign closed. `prompts/INDEX.md`'s row says closed 3 / 3, with 0 open issues.
9. **Suites.** Run all six, serially, at `HEAD`.
   - Datastore must be 739 (163 + 185 + 200 + 191), and RunRegistry 280.
   - Every other count is unchanged.
   - `black --check` is clean on every touched Python file.

## 4. After the review

- **Record it on this board** as a §1 *Orchestrator review of prompt 03* paragraph, in the form of
  02's. Fix small residue in a follow-up commit of its own. That includes:
  - "this commit" → the SHA, on both boards, in the log, the README and `prompts/INDEX.md`;
  - README §2's status column for 03;
  - this note added to the board's list of notes, marked "used for 03".
- Anything the review finds that is not residue is opened on this board's §3 and in the index. The
  campaign then is not closed; say so on the board, in the README and in `prompts/INDEX.md`.
- **Report to the user:**
  - the change, and the four callers;
  - one refusal's message before and after, for each operation;
  - the orchestrator's addition (the missing-shard fixture and breakage (v)), and why;
  - the tests added and the one changed;
  - `_read_back`'s message, which no test reaches;
  - the issue closed, the index count, and that the campaign and `datastore-generic`'s issues are
    all closed.
