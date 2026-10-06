# Orchestrator — prompt 01, refuse a drop that leaves references

Read [`../README.md`](../README.md) first: §0.2 (item 1), §1, §3, §4, §5 and §6. Then
`datastore-generic`'s board, §3.1, the entry
`[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]`. This is the campaign's first
note, so it states the conventions in full. They are `datastore-generic`'s
([`../../datastore-generic/orchestrator/prompt-09.md`](../../datastore-generic/orchestrator/prompt-09.md)
§0's last list), carried over.

**You do not write code.** You may:
- run the suites and the tests the log names;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process check or a probe from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`01-refuse-a-drop-that-leaves-references.md`](../01-refuse-a-drop-that-leaves-references.md)
**Board items:** D1–D3 · **Closes:**
`[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]`, on `datastore-generic`'s board ·
**Narrows:** nothing · **Opens:** nothing expected · **Model:** Opus, as the prompt recommends.

**Gate:**
- The campaign is written (`33308fe`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means `Datastore/`, `RayTools/`, `config/`, `utilities.py`, `main.py`,
    `extract_*.py`, `RunRegistry/`, `ComputeTargets/`, `tools/` or `docs/handover/`.
  - If a later commit touches code, re-check every fact below that it could move, and hand the
    agent the corrections as an addendum.
- One prompt at a time in this checkout. 01 and 03 both edit `ShardedPool.py`.

## 0. What makes this prompt unusual

**It is the campaign's one behavioural change.** Something that opens today is refused after it:
a read-write pool given `drop_tables` that leave references. 02 and 03 change a rendering and a
message. So the review has two halves:
- **the refusal is right and complete**: the rule is the user's (README §6.1, decision 1), every
  table it names is §0.2's, and it fires where §2 puts it, before anything is opened;
- **nothing else moves** (README §5 rule 9). A drop the refusal accepts drops exactly what it
  dropped before, in the same order, and an open with no drop is untouched.

**Both kinds of reference matter, and each half of the rule is pinned by its own test.** Foreign
keys alone miss `QuadSourceIntegral`'s cross-shard parents and `GkSource`'s parent set; declared
parents alone miss every value table, tag table and `GkSource_parents`; one level alone misses the
tables behind the first. `measure/out/f2_closure.txt` shows that each of the three changes some
group's list. Breakages (i)–(iii) are the review's evidence that the tests know it.

**The guard is the easy thing to trip.** `test_layer_is_generic` scans every layer module's code,
strings, comments and docstrings for project names. `dependent_tables`' docstring must say why
both kinds of reference count **without naming a project table as its example**, and the refusal's
message names only tables it was given or derived. Tests are outside the guard, so the new test
module may name tables freely.

**What moves on purpose:**
- `ShardedPool`'s constructor gains one call, after `_refuse_undeclared_drop_tables()`, and the
  class one method. `schema.py` gains `dependent_tables` and an import.
- `drop_tables`' `:param` in the constructor's docstring.
- One test is re-fixtured: `test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_store`,
  its drop and its docstring. It runs twice (`test_one_timestamp_per_write.TestRepairThenPruneAtOpen`
  inherits it).
- Eleven tests are added, in `test_drop_refuses_dangling_references.py`.
- The drop issue closes. The index goes from 107 to 106.

**The test plan was measured, not read** (README §0.2, [`../measure/`](../measure/)). The rough
change of all three prompts was made in a scratch export of `6de692a` and every suite run. With
the rough refusal, the only test that moved was the one re-fixtured.

**The prompt departs from the rough diff in four places**, so the agent's diff will not match
`rough_change.diff`, and need not:
- **The derivation lives in `schema.py`** as `dependent_tables`, beside `drop_order`, not as a
  static method of the pool (README §6.2).
- **The message lists the dependents in registry order**, not sorted, and the dropped tables
  sorted.
- **An undeclared name raises `ValueError`** in `dependent_tables`. The rough version ignored it.
- **The eleven tests and the records.** The rough version made neither.

**A test outside the plan that fails is therefore a real signal.** The agent stops and reports.

**The facts, checked by the orchestrator on 2026-10-06 at `33308fe`** (the code is `6de692a`'s:
`git diff 6de692a 33308fe` touches only `docs/OPEN_ISSUES.md` and `prompts/`). Pass them on.
- **The prompt's line numbers hold.** Checked: `ShardedPool.py:107`, `:122-125`, `:144-145`,
  `:198-208`, `:698-711`; `Datastore.py:140-141`, `:410-436`; `schema.py:279-319`;
  `store_inventory.py:82-85`, `:912-922`; `test_reconcile_at_open.py:1084`;
  `test_one_timestamp_per_write.py:94-97`; `test_layer_registry.py:315`;
  `test_wkb_numeric_parent_key.py:911`.
- **Class names are table names.** `config.datastore.factories` has 38 entries and the registry's
  `build_schema(...).tables` 38, in the same order. Every class `inventory_specs` returns and every
  name any spec's `dependencies()` gives is a table of the registry. So "registry order" is the
  same whichever of the two is read, and the spec's class names can be compared with table names
  directly.
- **`dependencies()` names more than parents.** It includes `TAG_TABLE` for a tagged class, and
  every class of a polymorphic type map. Neither changes any group's list: `store_tag` and the
  cosmology tables are in no group. The 13 tables in no group are `version`, `store_tag`,
  `redshift`, `wavenumber`, `wavenumber_exit_time`, `tolerance`, `LambdaCDM`, `QCD_Cosmology`,
  `IntegrationSolver`, `BackgroundModel`, `BackgroundModel_tags`, `BackgroundModelValue` and
  `QuadSourcePolicy`. None is any group's dependent.
- **The import.** `schema.py` imports only `typing`, `sqlalchemy` and `Datastore.contract` today.
  `store_inventory.py` imports only the standard library, `sqlalchemy` and `Datastore.contract` at
  module scope. `ShardedPool.py` imports `schema` and does not import `store_inventory`.
  `Datastore/__init__.py` imports `.object` only. The guard's `import_allowed` accepts any
  `Datastore.*` module outside the factories directory. The agent must still show that
  `python -c 'import Datastore.SQL.schema'` and `import Datastore.store_inventory` each succeed in
  a fresh interpreter.
- **Every caller that passes drop tables** (`git grep`, tracked `.py`):
  - `main.py:314` (`tables_to_drop(drop_actions)`) and `:4176`, the only production caller;
  - `test_reconcile_at_open.py:1107`, the one that moves;
  - `test_layer_registry.py:356` (`DROP = ["GkSource", "NotATable"]`, read-write: the undeclared
    refusal fires first) and `:404` (read-only: `ReadOnlyWrite` fires first);
  - `test_read_only_pool.py:623`, `:638` (read-only, `["GkSourcePolicyData"]`: `ReadOnlyWrite`
    first, since `_open_read_only` returns before the read-write refusals);
  - `test_layer_registry.py:284`, `:304`, `:383` and `test_wkb_numeric_parent_key.py:911`, which
    call the actor's `_drop_tables` directly, and so never meet the pool's refusal.
- **`standin_pool.store_checksums`** is at `standin_pool.py:394`; `build_full_store` at
  `real_store_fixtures.py:1188`.
- **Not touched, and staying:** `main.py`, `config/`, `Datastore/SQL/Datastore.py`,
  `store_inventory.py`, every factory, every `docs/` script, every other test, and every store
  under `var/`.
- **Expected counts after 01:**
  - Datastore 732 = 158 + 183 + 200 + 191 (the new module sorts into `[a-f]`);
  - RunRegistry 280; ComputeTargets 693; AdaptiveLevin 32; CosmologyModels 39;
    LiouvilleGreen 148 (1 skipped).
- **The index:** 107 now, 106 after.

**What the review exists to establish.**
- **(E1) The derivation.** `dependent_tables` for each of the ten groups equals
  `f2_closure.txt`'s list, in registry order. Both kinds of reference count, transitively. An
  undeclared name raises `ValueError`. It reads no store.
- **(E2) The refusal.** It is called immediately after `_refuse_undeclared_drop_tables()`, for a
  read-write pool only, before the primary is created or opened. Its message is §2's, with the
  prefix once. It names no group and no table by literal.
- **(E3) Nothing else in the layer changes.** Every function of `ShardedPool.py` but `__init__`
  dumps identically by `ast`, docstrings stripped; `__init__` differs by the one call.
  `schema.py` differs by the import and the new function.
- **(E4) Accepted drops and drop-free opens are unchanged** (rule 9): §3 check 7's probe.
- **(E5) The pins.** (i)–(vi) each fail the named tests.
- **(E6) The records.** The issue is in `datastore-generic`'s §4 with a dated "Closed" line, gone
  from its §3.1 and from the index, which counts 106. This board has the §1 row, D1–D3 and §5.
  README §4's "01 ships" says it shipped.

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
- **Breakage diffs** are recorded so that `git apply` replays them from a clean tree at the agent's
  commit. Check each with `git apply --check` and `-R --check`, and replay in `bash`, not zsh.
- **Dead helpers** that only the changed code or tests used go in the same commit, after a grep
  (IMPLEMENTATION CHOICE). This includes an import left unused.
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
   - `git log --oneline 33308fe..HEAD` must touch no code; otherwise see the gate.
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`.
3. **Ray.** Record `pgrep -lf 'gcs_server|raylet|ray::'`. No listed process may come from this
   repository's `venv`.
4. **Baselines.**
   - First check that `./venv/bin/python -c 'import scipy.sparse.linalg'` succeeds.
   - The last measurement was at `33308fe`, on 2026-10-06 at load 4–20, while this note was
     written: AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped),
     ComputeTargets 693, RunRegistry 280, Datastore 721 (147 + 183 + 200 + 191), all OK.
   - If `HEAD` has moved past the note's commit, re-measure all six, serially, with the load
     average. Give the agent the numbers.
5. **The index.** It is 107 now, and 106 after. Tell the agent both.
6. **The pre-images**, for check 4. Record `git rev-parse HEAD:<path>` for
   `Datastore/SQL/ShardedPool.py`, `Datastore/SQL/schema.py` and
   `Datastore/tests/test_reconcile_at_open.py`.
7. **The parent side of check 7.** Export `HEAD` to the scratchpad (`git archive HEAD | tar -x -C
   …`, with the venv symlinked). At the export:
   - build `build_full_store` with the export's code
     (`../../datastore-generic/orchestrator/measure-09/scripts/m4_build.py`), keep it, and record
     its checksums;
   - make three copies, and open each read-write through the export's stand-in pool, as
     `m4c_open_rw.py` does, with `drop_tables`:
     - (a) none;
     - (b) `tables_to_drop(list(drop_groups))`;
     - (c) `tables_to_drop(["gk-source-policy-records", "gk-source-policy",
       "quad-source-integral"])`, test 11's accepted drop;
   - for each, record what the constructor prints, its reconciliation and prune records, and the
     `sqlite_master` and row dump of the store after it closes (`m4_sqlite_dump.py`).

## 2. Dispatch

Dispatch one fresh-context subagent, on **Opus**. Give it:
- the prompt;
- the campaign README, this board, `datastore-generic`'s board §3.1 entry, and
  `measure/` (the rough diff's `ShardedPool.py` and `test_reconcile_at_open.py` hunks,
  `f2_closure.py` and its output);
- `HEAD`, the baselines and the index counts;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it that the prompt, not the rough diff, governs, and where they differ (§0 above).

Tell it plainly:
- **One commit, at the end.** Its log is `logs/01-refuse-a-drop-that-leaves-references.md`, in
  README §5.1's form, with the prompt's §7 additions.
- **The files it may add or change are the prompt's §6 list, and nothing else.** In particular, it
  must not touch:
  - anything under `docs/` other than `OPEN_ISSUES.md`;
  - `orchestrator/`, `measure/` or `var/`;
  - `main.py`, `config/`, `Datastore/SQL/Datastore.py` or `store_inventory.py`;
  - any test but the re-fixture and the new module.
- **It runs no build and starts no Ray.**
- **Stop and ask** on any of the prompt's §5 conditions. A list that differs from §0.2's is a
  stop, never a reason to change the expected list. A guard hit is a stop, never an allow-list
  entry. A test outside the plan that fails is current behaviour moving, never a reason to edit
  it.
- **Run the Datastore suite in four groups**, in the foreground.

## 3. The review — nine checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only the prompt's §6 list.
   - The diff over `docs/` (but `OPEN_ISSUES.md`), `config/`, `utilities.py`, `main.py`,
     `extract_*.py`, `RunRegistry/`, `tools/`, `var/`, `orchestrator/`, `measure/`, every factory,
     and every module of the layer but `ShardedPool.py` and `schema.py`, is empty.
2. **E1, by running.**
   - At the agent's commit, print `dependent_tables(tables_to_drop([g]), factories)` for each of
     the ten groups, and compare each with `f2_closure.txt`'s full list, in order.
   - Each group plus its list gives `[]`; all ten together give `[]`.
   - An undeclared name raises `ValueError`, naming it.
   - Re-run `f2_closure.py`'s three mutated closures against the agent's function by monkeypatch
     or by reading: the function uses foreign keys, declared parents and transitivity.
3. **E1 and E2, by reading.** Read `dependent_tables` and `_refuse_drop_that_leaves_references`
   in full.
   - The derivation reads the registry only, opens nothing and takes no store path.
   - The docstring states the rule, that it is structural (README §6.2) and why both kinds count,
     naming no project table.
   - The refusal is called immediately after `_refuse_undeclared_drop_tables()`, and nothing in
     the read-only branch reaches it.
   - Raise it for `gk-source-policy-records` alone. The message begins
     `Cannot open sharded datastore "<primary>": ` once. It names the one dropped table,
     `GkSourcePolicy`, and its three dependents in registry order: `GkSourcePolicyData`,
     `QuadSourceIntegral`, `QuadSourceIntegral_tags`. It says they must be dropped too, and ends
     `Nothing was opened`.
4. **E3, by `ast` against the pre-images, docstrings stripped.**
   - `ShardedPool.py`: every function and class statement but `__init__` and the new method dumps
     identically. `__init__` differs only by the one call.
   - `schema.py`: every existing function dumps identically; the module differs by the import and
     `dependent_tables`.
   - `test_reconcile_at_open.py`: only `test_an_interrupted_store` differs, by its drop and its
     docstring.
5. **The tests, by reading.** Read `test_drop_refuses_dangling_references.py` in full.
   - Test 1's expected lists are literals, equal to §0.2's, not computed.
   - Tests 4–6 make their structural assertion before the dependent one, so that each can fail
     only for its own reason.
   - Test 8 compares checksums before and after; test 9 checks the directory; test 11 asserts rows
     existed before it asserts they are gone.
   - No test passes on an empty input. Run the module, `test_reconcile_at_open` and
     `test_one_timestamp_per_write`.
6. **E5, the pins.** In `bash`, replay each breakage diff (i)–(vi) with `git apply`, run the named
   tests and see them fail, and nothing pass that should fail. Revert each with `git apply -R`, and
   check that `git status` is clean.
7. **E4, the probe**, from the scratchpad, at the agent's commit against the parent side taken
   before dispatch (§1 step 7). From a fresh copy of the same parent-built store each time:
   - (a), (b) and (c) open at the agent's commit as they did at the parent: the same printed
     lines, reconciliation and prune records, and nothing raised;
   - the post-close dumps are identical to the parent's, or differ only in columns the open itself
     stamps with the time. Name any such column, and show that it differs between two parent-side
     opens too;
   - (d) `tables_to_drop(["gk-source-policy-records"])` raises at the agent's commit, where the
     parent opened, and the store's checksums are unchanged. That is the intended change.
8. **E6, the records.**
   - The drop issue is in `datastore-generic`'s §4 with a dated "Closed" line naming
     `dependent_tables`, the refusal, this campaign and §0.2's undercount; gone from its §3.1 and
     from the index.
   - The index is 106 by counting, matching the header.
   - README §4's "01 ships" says it shipped as named, or names what differs.
   - This board has the §1 row, D1–D3 ✅ and §5.
9. **Suites.** Run all six, serially, at `HEAD`.
   - Datastore must be 732 (158 + 183 + 200 + 191), and RunRegistry 280.
   - Every other count is unchanged.
   - `black --check` is clean on every touched Python file.

## 4. After the review

- **Record it on this board** as a §1 *Orchestrator review of prompt 01* paragraph: what each
  check found, the probe's outcome, and anything handed on to 02 or 03. Fix small residue in a
  follow-up commit of its own. That includes:
  - "this commit" → the SHA;
  - a list of the orchestrator notes on the board, with this one marked "used for 01".
- Anything the review finds that is not residue is opened on this board's §3 and in the index.
- **Report to the user:**
  - the rule, and where it lives;
  - the ten groups: which are refused alone and what each must take with it;
  - the refusal's message for `gk-source-policy-records` alone;
  - the tests added, and the one re-fixtured;
  - that accepted drops and drop-free opens are unchanged, with the probe's evidence;
  - the issue closed; the index count.
- **Then 02.** It touches no file 01 touches. Its note is written after this review.
