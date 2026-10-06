# Orchestrator — prompt 08, derive the inventory's class list

Read [`../README.md`](../README.md) §0.1, §1, §4, §5 and §6.2 (the user's three choices of method
for 08, taken on 2026-10-06). Then read:
- the audit's C1 (rows 12 and 13, and the comment list for `SI`), C2 and D
  ([`docs/datastore-generic-audit.md`](../../../docs/datastore-generic-audit.md));
- the board's review of 07 (§1), with its "Handed on" list.

The conventions are [`prompt-07.md`](prompt-07.md)'s, carried below.

**You do not write code.** You may:
- run the suites and the tests the log names;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process check or a probe from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:**
[`08-derive-the-inventorys-class-list.md`](../08-derive-the-inventorys-class-list.md)
**Board items:** I1–I2 · **Closes:** nothing ·
**Narrows:** `[00-the-generic-datastore-layer-hard-codes-project-schema-facts]` (rows 12 and 13;
every row of C1 is then gone) ·
**Opens:** `[08-resolve-shows-an-absent-nullable-parent-as-unresolved]` · **Model:** Sonnet, as
the prompt recommends.

**Gate:**
- 07 landed (`3f51939`) and was reviewed (`5fe0ae6`).
- 08 is written (`b28ceb8`), in the commit before this note's. The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means `Datastore/`, `config/`, `main.py`, `extract_*.py`, `RunRegistry/`,
    `ComputeTargets/` or `tools/`.
  - If a later commit touches code, re-check every fact below that it could move, and hand the
    agent the corrections as an addendum.
- One prompt at a time in this checkout.

## 0. What makes this prompt unusual

**It is wide and mechanical, and everything it touches is pinned byte for byte.**
- All 21 factories' `inventory_records` become `inventory_spec()` declarations. The layer calls
  `read_records` itself.
- The class list and its order are derived. The polymorphic reference is declared by the client.
- None of this may move the fingerprint, any record or problem, the report's text, `resolve()`'s
  output or any refusal that exists today. The measurement found all of them byte-identical, and
  the review repeats that (check 8).

**Three choices of method were the user's.** On 2026-10-06, each taken as recommended (README
§6.2):
- **a declarative spec**: the inventory's dependencies are the parents each factory declares.
  Foreign keys cannot give them: nine of the parents the inventory reads have none (measured), and
  the reversed foreign-key order violates all nine;
- **roots first**: the classes that reference nothing come first, in registry order, then a
  dependency sort in registry order. It reproduces today's 21-class order exactly. A plain sort
  would move `wavenumber_exit_time`, `GkSourcePolicy` and `QuadSourcePolicy`;
- **the type map in the referencing `Parent`**, through one client constant,
  `ObjectFactories/cosmology_types.COSMOLOGY_TYPES`.

The README's charter for 08 ("derived from the registry and foreign keys") is amended to match.

**What moves on purpose:**
- `ClassInventory.parents[field]` is `None` for the polymorphic field, where it was `"cosmology"`.
  `ClassInventory.parent_types` carries the type column and map. No output reads the old marker
  but the report's `parent_class`, which the prompt rewrites, and `labelled()`, which drops both.
- New `ValueError` refusals of a malformed declaration. The real registry meets none.
- `store_inventory.py`'s docstrings and comments are rewritten generically.
- One issue is opened, `[08-resolve-shows-an-absent-nullable-parent-as-unresolved]`. The index goes
  from 107 to 108.

**What 08 does not change, and why the prompt says so.**
- **The refusal order.** The order is derived inside `open_read_only`, so that a registry lacking
  a class the store records is still refused by the reader's `RuntimeError`. The measurement's
  rough version derived it before opening, and raised `ValueError` instead (M7b). The prompt pins
  the current order with a new test and breakage (vi).
- **`resolve()`'s quirk**: an absent nullable parent shows as `"unresolved": 0`. 08 opens an issue
  rather than fixing it, so that `resolve()` stays byte-identical.
- **The report tool's own tables** (`INVENTORY_CATEGORIES`, `VALUE_TABLES`) and `labelled()`. The
  tool is a client, outside the layer.

**The test plan was measured, not read.** On 2026-10-06 at `5fe0ae6`, an Opus subagent made a
rough version of the change in a scratch worktree, ran every suite and took M1–M8. It then removed
the worktree. The rough diff, the scripts and the outputs are committed beside this note in
[`measure-08/`](measure-08/), with scratch paths shown as `<scratch>` and the SQLite stores left
out.
- **Phase A, code only:**
  - `test_inventory_report` and `test_store_inventory` failed to import, so Datastore ran
    150 + 134 + 200 + 150;
  - `test_wkb_numeric_parent_key`'s `test_the_inventory_key_has_no_z_init` errored in both
    sub-tests;
  - RunRegistry's `test_store_fingerprint` failed to import, so RunRegistry ran 248;
  - every other suite passed.
- **Phase B, the tests edited to the new names:** every suite back at its baseline. Datastore 692,
  RunRegistry 280 (1 skipped in the worktree), ComputeTargets 693.
- **Phase C, the 24-test sketch of `test_inventory_declarations.py`:**
  - Datastore 716 (150 + 175 + 200 + 191);
  - RunRegistry 280; ComputeTargets 693;
  - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped).

**The prompt departs from the rough diff in six places**, so the agent's diff will not match
`rough_change.diff`, and need not:
- **The order is derived inside `open_read_only`**, not before it (above).
- **The two existing test modules keep a literal `INVENTORY_CLASSES`** of today's 21 names. The
  rough version derived it, which made `test_every_class_has_a_record` compare the derivation with
  itself.
- **A new refusal-order test**, so the plan has 25 new tests where the sketch had 24.
- **Breakage** (i)–(vii) is the prompt's list. The sketch's five mutations overlap it: (iii) the
  tag table, (vi) the refusal order and (vii) the token test are new.
- **The issue is opened**, with its board entry and index row.
- **The records** (log, board, README §4, index): the rough version made none.

**A test outside the plan that fails is therefore a real signal.** The agent stops and reports.

**The facts, checked by the orchestrator on 2026-10-06 at `b28ceb8`** (the code is `5fe0ae6`'s).
Pass them on.
- **The prompt's line numbers** were checked against the tree.
- **Callers.**
  - Of `inventory_records`: only `read_inventory` (`store_inventory.py:882`),
    `test_wkb_numeric_parent_key.py:517` and `test_store_inventory.py`'s `ast` scan (`:596`).
  - Of `INVENTORY_CLASSES`, `COMPUTE_TARGETS`, `cosmology_classes` and `COSMOLOGY`: `tools/
    inventory_report.py` (`:83`, `:87`, `:177`, `:234`), `test_inventory_report.py`,
    `test_store_inventory.py`, `RunRegistry/tests/test_store_fingerprint.py` and the factories of
    `wavenumber_exit_time` and `BackgroundModel`.
  - **No `docs/` or `prompts/` script** uses any of them.
- **`read_records` is called directly** by `test_quadsource_parent_key.py:453` and
  `test_wkb_numeric_parent_key.py:999`, with positional `Parent(column, of, …)`. Both pass
  unchanged (Phase B), and `read_records`' signature stays.
- **`ClassInventory.parents` is read** by `tools/inventory_report.py:233` and
  `test_inventory_report_labels.py:549-552`. The latter tests membership in the sharded classes,
  and `None` is no more a member than `"cosmology"` was, so it passes unchanged.
- **`StoreInventory.resolve`** is called only by tests (`test_store_inventory.py:402`,
  `test_gksource_parent_set.py:915`, `test_quadsource_integral_parent_key.py:400`).
- **`mock`** stays used in `test_wkb_numeric_parent_key.py` (`:918`, `:936`) after the edit.
- **Not touched, and staying:** `ShardedPool`, the actor, `schema.py`, `store_reader.py`,
  `ClientPool.py`, `config/`, `RunRegistry/stores.py`, `main.py`, every `docs/` script and every
  store under `var/`.
- **Expected counts after 08:**
  - Datastore: 692 plus the new tests, which the log names; the plan has 25;
  - RunRegistry: 280; ComputeTargets: 693; every other suite unchanged.
- **The index:** 107 now, 108 after (the one issue 08 opens), plus one for each other issue it
  opens.

**What the review exists to establish.**
- **(E1) `store_inventory.py` names no project table, column or package**, in code, comments or
  docstrings, and loads no `CosmologyModels` module. `tools/inventory_report.py` imports neither
  `CosmologyModels` nor the deleted names.
- **(E2) The facts are declared, once each.**
  - All 21 factories declare an `inventory_spec`; none has `inventory_records`.
  - `COSMOLOGY_TYPES` is declared once and used by both factories.
  - `inventory_classes` derives today's order. The refusals the prompt lists raise.
- **(E3) Nothing the inventory reads or reports changes.**
  - Both fingerprints, both whole inventories and every `resolve()` are byte-identical.
  - `main.py --inventory`, plain and verbose, is byte-identical.
  - The refusals that exist today are byte-identical, the reader's coming first.
- **(E4) The test plan was followed.**
  - Only the six test files the prompt names are edited, each only as it says.
  - No test is deleted.
  - Every new method is in `test_inventory_declarations.py`.
- **(E5) The pins.** (i)–(vii) each fail a named test.
- **(E6) The records.**
  - The generic-layer issue is narrowed (rows 12 and 13).
  - The `resolve()` issue is opened on the board and in the index, and the index is 108.
  - README §4 names what 08 shipped.
  - This board has the §1 row, I1–I2, §3 and §5.

**Conventions, carried from 01–07.** All of these bind the agent, and the brief says so.
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
  commit. Check each with `git apply --check` and `-R`, and replay in `bash`, not zsh.
- **Dead helpers** that only the changed code or tests used go in the same commit, after a grep
  (IMPLEMENTATION CHOICE). This includes an import left unused.
- **Scratch.** The agent uses its own subdirectory of the session scratchpad, never `/tmp`.
  Python's `tempfile` in tests is fine.
  - A scratch script never names a helper `inventory`: a repo-wide grep test forbids
    `def inventory(` in any tracked file (prompt 06's review).
- **Ray.** 08 starts no Ray. ChamPBH's cluster, if running, is left alone.
- **Stop conditions are real.** A stop goes to the user through the orchestrator, and the agent
  resumes with `SendMessage`.
- **Other sessions** may land commits on this branch. Review against the agent's commit, and check
  `git log` before measuring at `HEAD`.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` must be empty.
   - `git worktree list` must show this checkout alone.
   - `git log --oneline 5fe0ae6..HEAD` must touch no code; otherwise see the gate.
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`.
3. **Ray.** Record `pgrep -lf 'gcs_server|raylet|ray::'`. No listed process may come from this
   repository's `venv`.
4. **Baselines.**
   - First check that `./venv/bin/python -c 'import scipy.sparse.linalg'` succeeds.
   - The last measurement was at `3f51939` (07's review):
     - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped);
     - ComputeTargets 693; RunRegistry 280;
     - Datastore 692 (150 + 151 + 200 + 191).
   - Re-measure all six at `HEAD`, serially, with the load average, and give the agent the
     numbers.
   - **Run the full Datastore suite before dispatch**, so that a committed measurement file
     tripping a repo-wide grep test shows at once. One did at 06 (`154d868`).
5. **The index.** It is 107 now, and 108 after, plus any other issue 08 opens. Tell the agent
   both.
6. **The pre-images**, for checks 1–4. Record `git rev-parse HEAD:<path>` for:
   - `Datastore/store_inventory.py`, `Datastore/SQL/ObjectFactories/base.py` and
     `tools/inventory_report.py`;
   - the 20 factory modules of the prompt's §6;
   - the six test files of the prompt's §2.

   Export `HEAD` to the scratchpad (`git archive HEAD | tar -x -C …`, with the venv symlinked).
   There, before dispatch, take the parent side of check 8:
   - the fingerprints and whole inventories of `build_full_store` and `build_real_store`;
   - `resolve()` of every record of both;
   - `main.py --inventory`, plain and `--inventory-verbose`, on one `build_full_store` primary,
     copied once and kept for both trees, with its checksums;
   - the refusals of `measure-08/scripts/m7_messages.py` and `m7b_reduced_registry.py`.

## 2. Dispatch

Dispatch one fresh-context subagent, on **Sonnet**. Give it:
- the prompt;
- the campaign README, the board (including the review of 07), 07's log, and the audit's C1, C2
  and D;
- `HEAD`, the baselines and the index counts;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

It may also be given copies of `measure-08/rough_change.diff` and `measure-08/scripts/`, as a
worked example. Tell it that the prompt, not the rough diff, governs, and where they differ (§0
above). `phaseA_edit_factories.py` rewrites the 21 factories mechanically, and the agent may adapt
it, but it must read each result.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/08-derive-the-inventorys-class-list.md`, in README
  §5.1's form, with the prompt's §7 additions.
- **The files it may add or change are the prompt's §6 list, and nothing else.** In particular, it
  must not touch:
  - anything under `docs/` other than `OPEN_ISSUES.md`;
  - `orchestrator/` or `var/`;
  - any test but the six the prompt names and the new module;
  - any factory method but `inventory_records`;
  - in `tools/inventory_report.py`, anything but the imports, `self.types`, `parent_class` and the
    two comments.
- **Stop and ask** on any of the prompt's §5 conditions. A test outside the plan that fails is
  current behaviour moving, never a reason to edit it.
- **Run the Datastore suite in four groups**, in the foreground.

## 3. The review — ten checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only the prompt's §6 list.
   - The diff over these is empty: `docs/` (but `OPEN_ISSUES.md`), `Datastore/SQL/ShardedPool.py`,
     `Datastore.py`, `schema.py`, `ClientPool.py`, `store_reader.py`, `replication.py`,
     `shard_paths.py`, `config/`, `main.py`, `extract_*.py`, `RunRegistry/stores.py`,
     `RunRegistry/` but its one test, `var/` and `orchestrator/`.
2. **E1, by grep and by running.**
   - The prompt's §3.2 greps print what it says, and nothing more.
   - Run `measure-08/scripts/m6_names_left.py` (adapted) at the agent's commit. Every remaining hit
     is the contract, the record's own `validated`, SQLAlchemy's `.label()` or a local name, as the
     log classifies it.
   - Run `m5_imports.py` and `m5b_read_records_alone.py`: no `CosmologyModels` module is loaded by
     `store_inventory` or by `read_records` alone.
3. **E2, by `ast` and in-process, against the pre-images**, comparing each function with its
   docstring stripped.
   - **The factories:** in each, `inventory_records` is gone and `inventory_spec` is new, and every
     other function dumps identically. For each of the 21, the spec's `keywords()` equals the
     keywords the pre-image passed to `read_records`. Capture them at the export by patching
     `store_inventory.read_records` with a recorder and calling each `inventory_records(None,
     table, tables, None)`. `measure-08/scripts/show_inventory_records.py` lists the pre-images'
     sources (`measure-08/out/base_inventory_records.txt`).
   - **`store_inventory.py`:**
     - gone: `cosmology_classes`, `COSMOLOGY`, `INVENTORY_CLASSES` and `COMPUTE_TARGETS`;
     - new: `InventorySpec`, `inventory_specs`, `inventory_classes`, and `Parent`'s and
       `ParentSet`'s checks and helpers;
     - changed: `Parent`, `ParentSet`, `ShardRead`, `ClassInventory`, `StoreInventory.resolve`,
       `read_records`, `_combine` and `read_inventory`;
     - every other function dumps identically: `canonical`, the digests, `Record`, `find`,
       `_merge_digests` and the rest.
   - **`tools/inventory_report.py`:** only `_Renderer.__init__` and `_Renderer.parent_class` change,
     with the imports and the two comments.
   - **In-process, on the real registry:**
     - `inventory_classes(factories)` equals today's 21 names in order;
     - the classes are exactly those with a spec;
     - each refusal of the prompt's §2 raises with the names;
     - `COSMOLOGY_TYPES` `is` the map both factories declare.
4. **E4, the tests, by `ast`.**
   - List the test methods of the six edited test files before and after: none gone, and only the
     methods and module lines the prompt names change.
   - Every new method is in `test_inventory_declarations.py`, and the log names its purpose.
5. **E2, by reading and running.** Read `test_inventory_declarations.py`.
   - Each refusal test asserts the exception and the names.
   - The order test pins a literal.
   - The refusal-order test uses a real `build_full_store` primary and asserts the reader's
     `RuntimeError`.
   - Run the module and the six edited test files.
6. **E3, by running.** Run by name:
   - `test_store_inventory`, `test_inventory_report`, `test_inventory_report_labels`,
     `test_inventory_consumers`, `test_gksource_parent_set`, `test_quadsource_parent_key`,
     `test_quadsource_integral_parent_key` and `test_wkb_numeric_parent_key`;
   - RunRegistry's `test_store_fingerprint` and `test_store_retire`;
   - ComputeTargets' `test_qcd_cosmology_inventory_record`.
7. **E5, the pins.** In `bash`, replay each breakage diff (i)–(vii) with `git apply`, run the named
   tests and see them fail. Revert each with `git apply -R`, and check that `git status` is clean.
   For (iv), read what the log says about `test_inventory_report`. For (v), check that the run
   ends.
8. **The probe, against the fix.** From the scratchpad, at the agent's commit and at the export of
   `HEAD~1`:
   - **the fingerprints** of `build_full_store` and `build_real_store`, with `taken` removed, are
     byte-identical. Give the digests; they should be `6576b85d…` and `d9328538…`;
   - **the whole inventories** are byte-identical, with `parents` normalised for the polymorphic
     field (`measure-08/scripts/m1_fingerprint.py`, adapted);
   - **`resolve()`** of every record is byte-identical (`m3_resolve.py`);
   - **`main.py --inventory`**, plain and verbose, on the kept primary: byte-identical stdout and
     stderr, exit 0, and the checksums unchanged;
   - **the refusals** (`m7_messages.py`, `m7b_reduced_registry.py`): the "has not been built" texts
     and the three reduced registries' exceptions are byte-identical. The reader's `RuntimeError`
     still comes first for a registry lacking `BackgroundModel` or `store_tag`.
9. **E6, the records.**
   - The generic-layer issue has a dated "Narrowed" line naming rows 12 and 13 and saying every C1
     row is gone. Its index hook says so.
   - `[08-resolve-shows-an-absent-nullable-parent-as-unresolved]` is on the board's §3, with its
     measurement, and in the index.
   - The index is 108 (or more) by counting, matching the header.
   - README §4 names `InventorySpec`, `inventory_spec`, `inventory_classes`, `Parent`'s type map
     and `COSMOLOGY_TYPES`.
   - This board has the §1 row, I1–I2 and §5.
10. **Suites.** Run all six, serially, at `HEAD`.
    - Datastore must be 692 plus the log's new tests, and RunRegistry 280.
    - Every other count is unchanged.
    - `black --check` is clean on every touched Python file.

## 4. After the review

- **Record it on this board** as a §1 *Orchestrator review of prompt 08* paragraph, in the form
  of 01's–07's. Fix small residue in a follow-up commit of its own. That includes:
  - "this commit" → the SHA;
  - this note marked "used" in the board's list;
  - the board header's count.
- **Report to the user:**
  - what the factories now declare, and how the order is derived;
  - where the cosmology map lives, and that the layer no longer loads `CosmologyModels`;
  - that the fingerprints, the inventories, the report and `resolve()` are byte-identical (check
    8), with the digests;
  - the issue opened;
  - the tests edited and added.
- **What 08 hands on.** 09's note must carry:
  - every C1 row is gone. What remains for the guard is the comments of the audit's list outside
    `ShardedPool.py` and `store_inventory.py`: `Datastore.py`, `ClientPool.py:37-38`,
    `replication.py:200, 204` and `shard_paths.py:14`, at the audit's lines. 09 must re-measure
    them;
  - the vocabulary the guard must allow:
    - `ShardedPool`'s "model row", "value rows" and "part of this model" (07);
    - the record's `validated`;
    - the contract's names through `Datastore.contract`;
  - the two existing name tests, `test_declared_facts.TestNoProjectNameInThePool` and
    `test_inventory_declarations`' token test, which 09's guard generalises;
  - `tools/inventory_report.py` is a client tool that names project tables, and is outside the
    guard;
  - whatever 08's log hands on.
