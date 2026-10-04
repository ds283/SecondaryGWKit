# Orchestrator — prompt 01, delete the old-store refusals that misfire

Read [`../README.md`](../README.md) first: §0, §0.1, §1, §4 and §5. Then the audit's §0, F0, B8 and
B10 ([`docs/datastore-generic-audit.md`](../../../docs/datastore-generic-audit.md)). This is the
campaign's first note, so it states the conventions in full; later notes refer back to §0's last
list.

**You do not write code.** You may:
- run the suites and the tests the log names;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process check from the session scratchpad. Never commit one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`01-delete-the-old-store-refusals-that-misfire.md`](../01-delete-the-old-store-refusals-that-misfire.md)
**Board items:** M1–M4 · **Closes:** `[00-old-store-refusals-misfire-on-any-database-error]`, on this
board's §4 · **Narrows:** `[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]` ·
**Model:** Sonnet, as the prompt recommends.
**Gate:** none beyond a clean tree. `HEAD` is the commit that adds this note, or a later one that
touches no code. One prompt at a time in this checkout.

## 0. What makes this prompt unusual

**It is a deletion that fixes a defect, so the review has two halves.** The first half is the
ordinary one: the old-store code is gone, and only that code. The second is that the defect stays
fixed. The handlers being deleted were meant to be guarded by tests that an unrelated error is "not
disguised", and those tests passed for years while the handlers disguised every error. They built
the error as `OperationalError("SELECT ...", …)`, a statement naming no column, so `str(e)` never
contained the column the handler looked for. **M4's tests are only worth having if their statement
is the factory's real one.** Breakage (iii) exists to prove that: with the handler restored, a test
using `"SELECT ..."` passes and a test using the captured statement fails. The review replays it and
reads the M4 tests for where their statement comes from.

**The second thing to watch is over-deletion.** Three other `except SQLAlchemyError:` blocks sit in
the same files and are **not** old-store code (below). They swallow errors from the factories' own
prune deletes, which is a separate matter. M1 must leave them alone.

**The facts, checked by the orchestrator on 2026-10-04 at `8dab816`.** Pass them on; the agent need
not rediscover them. None of the files 01 touches has changed since the audit's tree, `dbd8a51`
(`git diff dbd8a51 HEAD` over them is empty), so the audit's line numbers hold.

- **The handlers to delete** [read].
  - `TkNumericIntegration.py`: `:283` in `build` (test `:290`) and `:920` in `TkNumericValue.build`
    (test `:923`), each `if "break_point_kind" in str(e)`.
  - `GkNumericIntegration.py`: `:279` (`:286`), `:919` (`:922`), `:1068` in `GkNumericValue.read_batch`
    (`:1071`).
  - `QCD_Cosmology.py`: `:105` (`:112`), `"T_z_representation" in str(e)`; module docstring `:10-18`.
  - `BackgroundModel.py`, `sqla_BackgroundModelFactory.build`: one `except` at `:347` holding the
    Gauss-order refusal (`:351`, tests `e.orig`) **and** the grid-identity branch (`:375`, tests
    `str(e)`), the compute-path refusal (`:380-393`) and the read-path re-query (`:397`,
    `_build_query(False)`); `_build_query(with_grid_identity)` at `:258`, used at `:284` and `:312`;
    the value-table handler at `:490` (`:499`). The Gauss-order refusal **has no test** at `HEAD`.
- **The three blocks to leave** [read]. `TkNumericIntegration.py:640`, `GkNumericIntegration.py:638`
  and `BackgroundModel.py:940` are `except SQLAlchemyError:` in `validate_on_startup`'s prune: they
  append "DATABASE ERROR encountered when pruning these values" and continue. That is current
  behaviour (`datastore-integrity` prompt 02 and `a3-v2-readiness` prompt 02 both note that a
  factory's prune returns normally on its own error). Not 01's; if the agent judges it worth an
  issue, it records it under *Observations*, and does not touch it.
- **`_source_grid_identity`** [read]. Set at `BackgroundModel.py:443` (`None`, on a miss: keep) and
  `:595` (the found row's identity). Read only by `extract_common.describe_background_generation`.
  **All six `extract_*.py` scripts call `require_background(model, …)` two lines before
  `describe_background_generation(model)`** (`extract_Gk_data.py` 335/337, `extract_GkWKB_data.py`
  377/379, `extract_TkWKB_data.py` 398/400, `extract_GkSource_data.py` 786/788,
  `extract_QuadSourceIntegral_data.py` 1080/1082, `extract_tensor_source_data.py` 329/331), and
  `require_background` raises `NoBackgroundModel` on a miss. So after 01 the "unknown" branch is
  unreachable from every script: M2's choice between saying so and raising cannot change a script's
  output. Either is acceptable; the log says which and why.
- **`RunSelection.label is None`** [read]. Read in one place, `resolve_run_selection`
  (`extract_common.py:257`). `choose_run_label`'s only caller is `resolve_run_selection` (`:255`).
  No extraction script reads `RunSelection.label`. **The prompt's third stop condition does not
  fire.** `available_run_labels` is tested by `Datastore/tests/test_inventory_consumers.py` and is
  not changed by 01.
- **The tests M3 deletes, exactly** [read, counted with `ast`]. Eleven methods:
  - `ComputeTargets/tests/test_run_identity.py`: `TestAStoreWithNoGridIdentity` (`:711`) —
    `test_the_read_path_still_reads_it_and_labels_it_unknown`,
    `test_the_compute_path_demands_regeneration`, `test_an_unrelated_database_error_is_not_disguised`;
    and `TestChoosingARun.test_a_store_with_no_runs_reads_as_unknown` (`:866`). The module docstring's
    bullet at `:33` ("a store that records no run label at all still **reads**…") is corrected with
    them;
  - `ComputeTargets/tests/test_cosmology_representation_key.py`: `TestOldSchemaFailsLoudly` (`:465`)
    — `test_a_table_without_the_column_demands_regeneration`,
    `test_the_error_it_raises_is_not_the_opaque_one`, `test_an_unrelated_database_error_is_not_disguised`;
  - `ComputeTargets/tests/test_numeric_break_point_key.py`: `TestOldSchemaFailsLoudly` (`:414`) —
    `test_a_missing_column_names_the_prompt_and_demands_regeneration`,
    `test_an_unrelated_database_error_is_not_disguised`;
  - `Datastore/tests/test_value_reads_parent_key.py`: `TestStoreWithoutBreakPointKindIsRefused`
    (`:402`) — `test_gk_numeric_value_reads`, `test_tk_numeric_value_build`.

  That is **9** from ComputeTargets and **2** from Datastore. The prompt's search for others finds,
  at `HEAD`, only matches that are **not** 01's: `predates` in `test_wkb_numeric_parent_key.py`,
  `test_quadsource_integral_parent_key.py`, `test_gksource_parent_set.py` and
  `test_quadsource_policy_key.py` (prompt 02); `test_prune_at_open.py` (prompt 03);
  `RunRegistry/tests/test_run_registry.py:541` (a run directory, not a store); and "unknown
  generation" in `ComputeTargets/tests/wkb_reference.py:381` and `convergence_reference.py:698`
  (a source-grid specification, not a store). None is deleted by 01.
- **A harness for M4 already exists** [read]. `_RaisingConnection` in
  `test_numeric_break_point_key.py:92` and `test_cosmology_representation_key.py:144` is handed the
  query object in `execute(self, query)`, and `_run_build_against` (`test_numeric_break_point_key.py`)
  drives the production `build` with no datastore. Compiling that query (`query.compile(...)`, with
  literal binds or not) and raising `OperationalError(<compiled statement>, {}, Exception("database is
  locked"))` from `execute` gives an error whose `str()` names every column, as the probe's does.
  `BackgroundModel` has no module-level harness: the one at `test_run_identity.py:829` is inside a
  class M3 deletes. The agent writes its own, and may borrow that class's setup. The value-table
  read needs a connection whose **first** `execute` succeeds and whose second raises.
- **Expected counts after 01.** ComputeTargets **686 − 9**, Datastore **598 − 2**, each plus 01's
  new tests (M2's refusal test, M4's). Every other suite unchanged.
- **The index.** **108** at `8dab816` (`grep -c '^| \`\[' docs/OPEN_ISSUES.md`, matching the header).
  After 01: **107**, plus one for each issue 01 opens.

**What the review exists to establish.**
- **(E1) The old-store handlers are gone** from the four factories, and only they: the three prune
  blocks are untouched, and no other `except` changed.
- **(E2) `BackgroundModel` always selects its grid identity** and never re-queries; what a current
  store returns (hit, miss, value, identity) is unchanged — every existing `BackgroundModel` and
  run-identity test outside M3's list passes untouched.
- **(E3) `extract_common`** refuses a store with no runs, and names no older store anywhere.
- **(E4) The misfire is pinned** by M4 tests whose statement is captured from the factory's own
  query, on every path the prompt lists, including `BackgroundModel`'s read path executing once.
- **(E5) The deletions are exactly M3's eleven**, plus the docstring lines that described them.
- **(E6) The records.** The issue in this board's §4; the fallbacks issue narrowed; the index 107.

**Conventions, carried from `a3-v2-readiness`'s reviews.** All bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, never `git add -A`, `.` or `commit -a`.
  Show `git diff --cached --name-only` before committing. Count, then paste. A log states only what
  the diff and the tests show.
- **The SHA, not "this commit".** The agent cannot know its own SHA. Fixing that is the
  orchestrator's residue.
- **Suites.** README §7's command, output to a file, the verdict grepped from it — never `| tail`.
  Under load, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`,
  `test_background_tau.test_build_cost` and `test_numeric_phase_cut.test_the_print_is_gated_both_ways`
  fail. Re-run a failure alone before calling it real; if it persists alone, settle it on an export
  of the tree before the prompt.
- **Breakage diffs** are recorded so that `git apply` replays them from a clean tree at the agent's
  commit. Replay them in `bash`, not `zsh`.
- **Scratch.** The agent works in its own subdirectory of the session scratchpad, never its top
  level, and never `/tmp`.
- **Ray.** 01 starts no Ray. The user's ChamPBH cluster runs on this machine (`gcs_server` 62525,
  `raylet` 62574, dashboard agent 62594 at `8dab816`), from its own venv. It is left alone.
- **Other sessions.** Another of the user's sessions may land commits on this branch while a prompt
  runs (it happened during `a3-v2-readiness` 07). Review against the agent's commit, and check
  `git log` for anything after it before measuring at `HEAD`.
- **Usage limits.** Resume a stopped agent with `SendMessage`, saying exactly where it stopped. If it
  commits but sends no report, review from the commit and its log.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`. `git status --short` and `git diff --cached` are both
   empty; `git worktree list` shows this checkout alone; `git log --oneline 8dab816..HEAD` touches no
   code.
2. **The registry.** `./venv/bin/python -m RunRegistry list`: nothing `running`. At `8dab816` it held
   16 runs: 14 finished, 2 `unknown` (`a3-pilot`, `a3-pilot-resume`).
3. **Ray.** Record `pgrep -lf 'gcs_server|raylet|ray::'`. No listed process may come from this
   repository's `venv`; ChamPBH's are expected.
4. **Baselines.** Last measured at `a6c8f39` (`a3-v2-readiness` board §5.17): AdaptiveLevin 32,
   ComputeTargets 686, CosmologyModels 39, Datastore 598, LiouvilleGreen 148 (1 skipped), RunRegistry
   281. Since then the only code change is `e53f323` (`ShardedPool.py` and one test module), on which
   the plan measured Datastore 598. Re-measure all six at `HEAD`, serially, with the load average,
   and record them as this board's §5.2.
5. **The index.** 108; after 01, 107. Tell the agent both, and that it rises by one per issue opened.
6. **The pre-images**, for checks 1 and 3: `git rev-parse HEAD:<path>` for the four factories,
   `extract_common.py` and the four test modules of M3.

## 2. Dispatch

One fresh-context subagent, on **Sonnet**. Give it:
- the prompt;
- the campaign README, this board, and the audit's §0, F0, B8 and B10;
- `HEAD`, the baselines and both index counts;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad. The
  agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/01-delete-the-old-store-refusals-that-misfire.md`, in
  README §5.1's form, with the prompt's §7 additions.
- **The files it may add or change:**
  - `Datastore/SQL/ObjectFactories/TkNumericIntegration.py`, `GkNumericIntegration.py`,
    `QCD_Cosmology.py`, `BackgroundModel.py`: the handlers, the `_build_query` parameter, the
    `hasattr` tests and the comments and docstrings M1 names. Nothing else in them;
  - `extract_common.py`: `choose_run_label`, `resolve_run_selection`,
    `describe_background_generation`, `RunSelection`'s docstring, the module docstring's bullet;
  - the four test modules of M3: the eleven deletions and their docstring lines, and new tests beside
    them if the agent puts M2's and M4's there;
  - new test modules for M2 and M4, if it prefers;
  - the log; this board; `docs/OPEN_ISSUES.md`.

  **Nothing else.** In particular: no other factory (02's), no `ShardedPool`, reader or inventory
  (03–05), no `extract_*.py` script, no `ComputeTargets/BackgroundModel.py`, no existing test outside
  M3's list, nothing under `var/` or `orchestrator/`, and not the three prune blocks.
- **Stop and ask** on any of the prompt's §5 conditions. Say that this note's facts answer the third
  as "does not fire" at `8dab816`, and that an existing test failing outside M3's list is a sign that
  current behaviour moved, not a reason to edit the test.

## 3. The review — ten checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only §2's list. `git diff HEAD~1 HEAD --
   Datastore/SQL/ShardedPool.py Datastore/store_*.py 'extract_[A-Za-z]*_data.py' main.py config/ var/
   ComputeTargets/BackgroundModel.py` is empty. Under `*/tests/`, only M3's four modules change, plus
   any new modules.
2. **(E1) The handlers, by reading and grep.**
   - `grep -n 'in str(e)' Datastore/SQL/ObjectFactories/` names no column (the prompt's verification
     2); `grep -n 'except SQLAlchemyError' ` over the four files shows the three prune blocks, at
     their lines or shifted only by the deletions above them, and nothing else.
   - With `ast`, compare each factory against its pre-image: the only changed functions are the
     `build`/`read_batch` methods named in §0, and `validate_on_startup` dumps identically.
3. **(E5) The deletions.** With `ast`, list the test methods of M3's four modules before and after:
   exactly the eleven of §0 are gone, nothing else is gone, and every new method is M2's or M4's.
4. **(E2) `BackgroundModel`, by reading.** `_build_query` takes no grid-identity flag and always
   selects `source_grid_digest` and `source_grid_construction`; the compute path still **filters** on
   them when `z_sample` is given (`:312`'s condition, less the flag); no `hasattr(row,
   "source_grid_digest")` is left; `_source_grid_identity = None` on a miss is kept; the read path's
   `z_sample=None` selection is unchanged.
5. **(E4) The M4 tests, by reading.** For each path the prompt lists (`TkNumericIntegration.build`,
   `GkNumericIntegration.build`, `GkNumericValue.read_batch`, `QCD_Cosmology.build`, and
   `BackgroundModel.build` on the compute path, the read path and the value-table read): the error's
   statement is compiled from the query the factory passed to `execute`, not a literal; the test
   asserts that the raised exception itself reaches the caller (identity or type and message, not a
   `RuntimeError`); and for `BackgroundModel`'s read path, that `execute` was called once.
6. **(E3) `extract_common`, by reading and running.** `choose_run_label([], None)` raises with a
   message saying the store records no runs; `choose_run_label([], "x")` still raises as before; no
   text in the module mentions prompt 14, an older store or an unknown generation;
   `describe_background_generation`'s behaviour for a model with no identity is what the log says.
7. **(E4) The pins.** In `bash`, replay each breakage diff (i)–(iv) with `git apply`; run the named
   tests and see them fail; for (iii), apply (i) and (iii) together and see the `"SELECT ..."` test
   **pass** while the captured-statement test of another path still fails. Revert each with `git
   apply -R` and check `git status` is clean.
8. **The probe, against the fix.** From the scratchpad, drive `TkNumericIntegration.build` and
   `BackgroundModel.build` (read path) through the new M4 harness with a `database is locked` error
   carrying the real statement, at the agent's commit: the `OperationalError` reaches the caller, and
   the read path executes once. Then at the pre-image (an export of `HEAD~1`): a `RuntimeError`
   saying "predates", and two executes. This is the defect, shown before and after.
9. **(E6) The records.** The issue is in this board's §4, with a closing note naming the commit (as
   the orchestrator will correct it), the M4 tests and the probe; its index row is gone, and the count
   is 107 by counting, matching the header. The fallbacks issue carries a dated line naming what 01
   deleted. This board has the §1 row, M1–M4 and §5.
10. **Suites.** All six serially at `HEAD`: ComputeTargets and Datastore as §0's expected counts plus
    the new tests, each new test named in the log; every other count unchanged. `black --check` clean
    on every touched Python file.

## 4. After the review

- **Record it on this board**, as a §1 *Orchestrator review of prompt 01* paragraph in the form
  `a3-v2-readiness`'s board uses: `HEAD` at dispatch, the checks, any remnants. Fix small residue in
  a follow-up commit of its own, including "this commit" → the SHA, and this note marked "used" in the
  board's list of orchestrator notes.
- **Report to the user:** what a database error on a current store now does, against what it did
  (check 8's before and after); the eleven tests deleted and the tests that replace their purpose;
  what remains of the old-store fallbacks (02–05).
- **What 01 hands on.** 02 follows 01's pattern for four more factories. Its note must carry: how
  01's M4 harness captures the statement (so that 02's pass-through tests reuse it rather than
  re-invent it); the review's findings; and that `real_store_fixtures.py`'s old-store machinery is
  still used by the reader and inventory tests, and stays until 05.
