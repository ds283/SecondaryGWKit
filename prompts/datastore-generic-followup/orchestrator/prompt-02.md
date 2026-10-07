# Orchestrator — prompt 02, resolve an absent parent as `None`

Read [`../README.md`](../README.md) §0.2 (item 2), §1, §4, §5 and §6.1 (decision 2). Then read:
- `datastore-generic`'s board, §3.1, the entry
  `[08-resolve-shows-an-absent-nullable-parent-as-unresolved]`;
- this board's review of 01 (§1), with its "Handed on" line.

The conventions are [`prompt-01.md`](prompt-01.md)'s, carried below.

**You do not write code.** You may:
- run the suites and the tests the log names;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process check or a probe from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`02-resolve-an-absent-parent-as-none.md`](../02-resolve-an-absent-parent-as-none.md)
**Board item:** V1 · **Closes:**
`[08-resolve-shows-an-absent-nullable-parent-as-unresolved]`, on `datastore-generic`'s board ·
**Narrows:** nothing · **Opens:** nothing expected · **Model:** Sonnet, as the prompt recommends.

**Gate:**
- 01 landed (`fee1fbf`) and was reviewed (`56bbd59`).
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means `Datastore/`, `RayTools/`, `config/`, `utilities.py`, `main.py`,
    `extract_*.py`, `RunRegistry/`, `ComputeTargets/`, `tools/` or `docs/handover/`.
  - If a later commit touches code, re-check every fact below that it could move, and hand the
    agent the corrections as an addendum.
- One prompt at a time in this checkout.

## 0. What makes this prompt unusual

**It is three lines, and the risk is in its test, not its change.** The branch is the rough
diff's (`measure-02/rough_store_inventory.diff`): in `resolve()`, after the leaf branch, a key
parent whose value is `None` gives `None`. The second new test is where the prompt is loose.

**The prompt's "exactly 11 `numeric` fields are `None`" needs reading carefully.** This is the
orchestrator's clarification of the prompt, not a change to it. Pass it on as such.
- `GkSource`'s parent set has a member field that is **also** named `numeric`. Its absent members
  already resolve to `None`, through `_resolve_reference`. On `build_full_store`, 9 of them are
  reached: 3 each in `GkSource`, `GkSourcePolicyData` and `QuadSourceIntegral` records.
- So a walk that counts every `numeric: None` finds **9 before the change and 20 after**, not 11.
- **The 11 are the `numeric` field of a `GkWKBIntegration` key**: a record of that class, or the
  `"key"` of a mapping whose `"class"` is `GkWKBIntegration`. Counted that way, 0 before and 11
  after. The parent-set members stay 9 on both sides.
- The test counts the 11 that way. It also asserts that the 9 members are still `None`, so that it
  shows that the members are untouched as well as that the key parents moved.
- A count of 20 or 9 is a miscount, not a stop. A count of the key-parent field other than 11 is a
  stop (prompt §5).

**What moves on purpose:**
- `StoreInventory.resolve`: one branch, and its docstring.
- Two tests are added, both in `test_inventory_declarations.TestResolve`.
- `resolve()` of every record of the two fixture stores differs in exactly 11 entries, all in
  `build_full_store` (`measure-02/out/r2_compare.txt`).
- The resolve issue closes. The index goes from 106 to 105.

**Nothing else moves.** `resolve()` is called by `store_inventory.py` itself, recursively, and
by tests. It is not called by the fingerprint, the inventory, `RunRegistry` or `tools/`'s
`format_inventory_report`, which renders parents through its own path. So the fingerprint, the
inventory and the report are unchanged, and the review checks that by running them (§3 check 6).

**The facts, checked by the orchestrator on 2026-10-07 at `56bbd59`** (the code is `fee1fbf`'s).
Pass them on.
- **The prompt's line numbers hold.** 01 did not touch `store_inventory.py` or
  `test_inventory_declarations.py`: `git diff 6de692a HEAD` over them is empty. `resolve` is
  `:445-479`, the leaf branch `:465-467`, `_resolve_reference` `:481-492`, `_reference`
  `:691-703`, and the row-dropping loop `:705-720`. `TestResolve` starts at `:309`;
  `test_a_hand_built_inventory` is `:310-340`.
- **The measurement reproduces at `HEAD`.** `measure/scripts/r2_resolve.py` at an export of
  `HEAD`, and at the same export with the rough hunk applied, gives exactly the 11 entries of
  README §0.2. Each `numeric` goes from `{"class": "GkNumericIntegration", "digest": null,
  "unresolved": 0}` to `null`. There are 2 in `GkWKBIntegration` records, and 3 each in `GkSource`,
  `GkSourcePolicyData` and `QuadSourceIntegral`. `build_real_store` holds no `GkWKBIntegration`
  record, and none of its entries moves.
- **Before the change, the full store holds 11 `"unresolved"` mappings. All 11 have `"digest":
  None`.** After it, there are none (`measure-02/out/r2_count.txt`). So the new test's "no
  unresolved mapping has a `None` digest" fails before the change and passes after it.
- **Nullability.** In the registry, exactly one key parent is declared nullable:
  `GkWKBIntegration.numeric`. Over both fixture stores, 2 records hold a `None` key parent, both on
  it. So the stop condition on a non-nullable `None` does not fire on the fixtures.
- **`test_a_hand_built_inventory`'s third child** has `kind: 7`, which the type map does not hold.
  It resolves to `{"class": None, "digest": da, "unresolved": 0}`, with a digest that is not `None`.
  Breakage (ii) turns it into `None`, and the test fails.
- **The other tests that call `resolve()`** must not move, and do not reach the 11:
  - `test_gksource_parent_set.py:912` asserts that each `GkSource` member's `numeric` and `wkb`
    are `None` or carry a `"key"`. It does not look inside a `GkWKBIntegration` key;
  - `test_quadsource_integral_parent_key.py:395` and `test_store_inventory.py:424` assert that
    `QuadSourceIntegral`'s own parent fields carry a `"key"`. They do not descend;
  - `test_inventory_declarations.py:342` resolves only the `cosmology` field.

  With the rough change, every suite passed (README §0.2).
- **The guard.** `store_inventory.py` is a layer file. The docstring sentence must name no project
  class: not `GkWKBIntegration` and not `numeric`. Tests are outside the guard.
- **Not touched, and staying:** `_resolve_reference`, `read_records`, `Parent`, every other
  module, every other test, `tools/`, `RunRegistry/` and every store under `var/`.
- **Expected counts after 02:**
  - Datastore 734 = 158 + 185 + 200 + 191 (`test_inventory_declarations` sorts into `[g-o]`);
  - RunRegistry 280; ComputeTargets 693; AdaptiveLevin 32; CosmologyModels 39;
    LiouvilleGreen 148 (1 skipped).
- **The index:** 106 now, 105 after.

**What the review exists to establish.**
- **(E1) The branch.** It is after the leaf branch. It gives `None` for a `None` value and for
  nothing else. A reference that does not resolve keeps today's form.
- **(E2) Only `resolve` changes**, by `ast`, docstrings stripped. In the test module, only the two
  methods are added.
- **(E3) The measurement reproduces.** `r2_resolve.py` at the parent and at the agent's commit
  differs in exactly the 11 entries, and `r2_count.py` gives 11 and 9.
- **(E4) The fingerprint, inventory and report are unchanged** for both fixture stores.
- **(E5) The pins.** (i) and (ii) fail the named tests. (iii) is recorded as a check that the
  order does not matter.
- **(E6) The records.** The issue is in `datastore-generic`'s §4 with a dated "Closed" line that
  corrects its hook. It is gone from §3.1 and from the index, which counts 105. This board has the
  §1 row, V1 and §5.

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
   - `git log --oneline 56bbd59..HEAD` must touch no code; otherwise see the gate.
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`.
3. **Ray.** Record `pgrep -lf 'gcs_server|raylet|ray::'`. No listed process may come from this
   repository's `venv`.
4. **Baselines.**
   - First check that `./venv/bin/python -c 'import scipy.sparse.linalg'` succeeds.
   - The last measurement was at `fee1fbf`, on 2026-10-07, at 01's review (board §5.2):
     AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped), ComputeTargets 693,
     RunRegistry 280, Datastore 732 (158 + 183 + 200 + 191), all OK.
   - Re-measure all six at `HEAD`, serially, with the load average, and give the agent the
     numbers.
   - **Run the full Datastore suite before dispatch**, so that a committed measurement file
     tripping a repo-wide grep test (this note's `measure-02/` is new) shows at once.
5. **The index.** It is 106 now, and 105 after. Tell the agent both.
6. **The pre-images**, for check 3. Record `git rev-parse HEAD:<path>` for
   `Datastore/store_inventory.py` and `Datastore/tests/test_inventory_declarations.py`.
7. **The parent side of checks 4 and 6.** Export `HEAD` to the scratchpad (`git archive HEAD |
   tar -x -C …`, with the venv symlinked). At the export:
   - run `../measure/scripts/r2_resolve.py <export> <out>/parent.json` from the export, and
     `measure-02/scripts/r2_count.py` on it. It should give 0, 9, and 11 of 11;
   - build both fixture stores, keep them, and record their checksums. Use
     `../../datastore-generic/orchestrator/measure-09/scripts/m4_build.py`;
   - read their fingerprints (`taken` removed) and whole inventories with `m4_inventory.py`;
   - render `format_inventory_report(read_inventory(primary, factories), "the-store.sqlite")` for
     each, and keep the text.

## 2. Dispatch

Dispatch one fresh-context subagent, on **Sonnet**. Give it:
- the prompt;
- the campaign README, this board (including the review of 01), `datastore-generic`'s board §3.1
  entry, and 01's log as an example of the log's form;
- `measure-02/rough_store_inventory.diff`, `measure/scripts/r2_resolve.py`,
  `measure-02/scripts/r2_compare.py` and `r2_count.py`, with `measure-02/out/`;
- `HEAD`, the baselines and the index counts;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it that the prompt governs, with §0's reading of "exactly 11", and that the agent need not
reproduce the rough diff exactly.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/02-resolve-an-absent-parent-as-none.md`, in README
  §5.1's form, with the prompt's §7 additions.
- **The files it may add or change are the prompt's §6 list, and nothing else.** In particular, it
  must not touch:
  - anything under `docs/` other than `OPEN_ISSUES.md`;
  - `orchestrator/`, `measure/` or `var/`;
  - in `store_inventory.py`, anything but `resolve` and its docstring;
  - any test but the two new methods.
- **It runs no build and starts no Ray.**
- **Stop and ask** on any of the prompt's §5 conditions. A count of the key-parent field other
  than 11 is a stop, never a reason to change the expected count. A guard hit is a stop, never an
  allow-list entry. A test outside the plan that fails is current behaviour moving, never a
  reason to edit it.
- **Run the Datastore suite in four groups**, in the foreground.

## 3. The review — nine checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only the prompt's §6 list.
   - The diff over `docs/` (but `OPEN_ISSUES.md`), `config/`, `utilities.py`, `main.py`,
     `extract_*.py`, `RunRegistry/`, `tools/`, `var/`, `orchestrator/`, `measure/`, every factory,
     and every module of the layer but `store_inventory.py`, is empty.
2. **E1, by reading.** Read `resolve` in full.
   - The branch tests `value is None` and nothing wider. It sits after the leaf branch and before
     `parent_class`.
   - The docstring says what a `None` parent gives, why only a nullable parent can hold one, and
     that a member gives the same. It names no project class.
3. **E2, by `ast` against the pre-images, docstrings stripped.**
   - `store_inventory.py`: every function and class statement but `resolve` (and the class that
     holds it) dumps identically. `resolve` differs only by the branch.
   - `test_inventory_declarations.py`: the only differences are the two added methods, and any
     import they need.
4. **E3, by running.** Run `r2_resolve.py` at an export of the agent's commit. `r2_compare.py`
   against the parent side gives exactly the 11 entries of `measure-02/out/r2_compare.txt`.
   `r2_count.py` gives 11, 9, and 0 of 0.
5. **The tests, by reading.**
   - The hand-built test has a `None` key parent and a second key parent that resolves, in one
     record, and asserts both.
   - The full-store test asserts that it visited a `GkWKBIntegration` record first. It counts the
     key-parent `numeric` field as §0 says, 11, and the members, 9. It asserts that no unresolved
     mapping has a `None` digest.
   - Run `test_inventory_declarations`.
6. **E4, the probe**, at the agent's commit against the parent side taken before dispatch (§1
   step 7). For the parent-built stores read at the agent's commit:
   - the fingerprints are byte-identical, and so are the whole inventories;
   - the rendered reports are identical text;
   - the stores' checksums are unchanged.
7. **E5, the pins.** In `bash`, replay each breakage diff with `git apply`, and run the named tests.
   (i) fails both new tests, and (ii) fails `test_a_hand_built_inventory`. (iii) fails nothing, as
   the prompt expects; the log must say so and must not call it a pin. Revert each with
   `git apply -R`, and check that `git status` is clean.
8. **E6, the records.**
   - The resolve issue is in `datastore-generic`'s §4, with a dated "Closed" line that names this
     campaign, the change and the 11 entries. It corrects the hook: 2 direct entries, and 3 each
     through `GkSource`, `GkSourcePolicyData` and `QuadSourceIntegral`. It is gone from §3.1 and
     from the index.
   - The index is 105 by counting, matching its header.
   - This board has the §1 row, V1 ✅ and §5.
9. **Suites.** Run all six, serially, at `HEAD`.
   - Datastore must be 734 (158 + 185 + 200 + 191), and RunRegistry 280.
   - Every other count is unchanged.
   - `black --check` is clean on every touched Python file.

## 4. After the review

- **Record it on this board** as a §1 *Orchestrator review of prompt 02* paragraph, in the form of
  01's. Fix small residue in a follow-up commit of its own. That includes:
  - "this commit" → the SHA;
  - README §2's status column for 02;
  - this note added to the board's list of notes, marked "used for 02".
- Anything the review finds that is not residue is opened on this board's §3 and in the index.
- **Report to the user:**
  - the change, and the 11 entries it moves;
  - the reading of "exactly 11" that the test uses, and why;
  - the two tests added;
  - that the fingerprint, inventory and report are unchanged, with the probe's evidence;
  - the issue closed; the index count.
- **Then 03.** It edits `ShardedPool.py` and `store_reader.py`, neither of which 02 touches. Its
  note is written after this review.
