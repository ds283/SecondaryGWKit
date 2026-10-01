# Orchestrator — prompt 09d, derived parents by label in the inventory report

Read [`../README.md`](../README.md) first: §0.1, §2, §4 and §5. **You do not write code.** You may
*replay* a mutation the log records, with `git apply`, and then revert it. You may render the
report of a fixture store in a throwaway export to measure it (§1), and then remove the export.

**The prompt:** [`09d-refer-to-derived-parents-by-label-in-the-inventory-report.md`](../09d-refer-to-derived-parents-by-label-in-the-inventory-report.md)
**Board items:** none · **Closes:** `[09b-inventory-report-renders-every-member-of-a-parent-set]` ·
**Model:** Sonnet
**Gate:** 09c landed and reviewed. The relevant commits:
- `014be95`: 09c;
- `b5fd484`: its remnants;
- `1e8364d`: its review;
- `fef6f79` and `a761914`: the prompt, then its widening.

**Never dispatched concurrently with anything else in this checkout.**

## 0. What makes this prompt unusual

**It changes a display, not a key.** One module changes, `tools/inventory_report.py`.
- Nothing stored changes, nor the inventory, the fingerprint, the schema or `main.py`.
- So there is no witness, no golden, no format bump and no D4 refusal. There is no `main.py` edit,
  so no `substitute_main_py()` check beyond showing `main.py` untouched.
- **Every existing test must pass unmodified.** So there is no grant, and no sketch-and-ask step
  before dispatch.

**The user's decisions of 2026-10-01**, on the review of 09c. They bind the agent as the prompt
does.
1. **Its own prompt.** The issue is fixed by a small prompt of its own, in the pattern of 04b, and
   not inside prompt 10.
2. **The wider scope.** It covers every reference to a **sharded** (derived) row, not only the
   parent sets. The sets are only part of why a `QuadSourceIntegral` line grew from ~400
   characters to ~4,300 on `build_full_store()`. Its `QuadSource`, `GkSourcePolicyData` and four Tk
   rows, each rendered in full, account for ~3,500.
3. **Not a summary.** A count-and-range summary of a set is not allowed. It would let two distinct
   records print as one line, which `test_verbose_renders_every_distinct_record_as_a_distinct_line`
   forbids. This was the orchestrator's finding while writing the prompt, and the prompt's §2 says
   so.

**The three things the review exists to establish.** All else is mechanics.
- **(P1) Distinct stays distinct.** Two distinct records never render as the same line. That
  includes pairs differing only in one member of a `GkSource`'s set, and only in one derived parent
  of a `QuadSourceIntegral`.
- **(P2) Every label is defined.** Every `<Class>#<n>` and `S<n>` printed has exactly one defining
  line in the same report, verbose or not. In a report without `verbose`, that includes rows that
  are referenced but not among the five shown.
- **(P3) Lines are bounded.** A record line does not grow with the size of a parent set or the depth
  of a chain of derived parents.

**What is measured already** (the prompt's §1 has the full tables): fixture record lines at
`1e8364d` are ~557 characters for `GkSource`, ~760 for `GkSourcePolicyData` and ~4,300 for
`QuadSourceIntegral`. At `2252b8c` they were ~165, ~370 and ~400. The production estimates are
derived from the grid sizes, not measured.

**Carried from the reviews of 08 and 09a–09c** (board §1). All bind the agent, and the brief says so.
- **Staging and reporting.**
  - Stage by explicit path only, never `git add -A`, `.` or `commit -a`. Show
    `git diff --cached --name-only` before committing.
  - Another session may write in this checkout; stop and ask if one modifies a file the agent needs.
  - **Count, then paste.** Paste `-v` output unfiltered.
  - A log states only what the diff shows, and its line numbers come from `git show`, not a
    working-tree view.
- **Mutations.**
  - Each is pasted as `git diff`, and replayed twice: before the commit from the final log file,
    and after it from `git show HEAD:<log>`, reported with no amend.
  - Replays run in a `git archive` export **made into its own git repository**, so that the tests
    that call `git` pass there. In a plain export, four tests fail every run: `test_inventory_retired`'s
    two, and one each in `test_run_registry` and `test_store_sidecar`.
- **Load-sensitive tests.** The machine may be heavily loaded by an unrelated job in another of the
  user's workspaces (load averages of 35–290 were seen during 09). Then these fail:
  - the known flake `test_tk_wkb_phase.TestCost.test_wall_time_per_object`;
  - `test_background_tau.test_build_cost`;
  - `test_numeric_phase_cut.test_the_print_is_gated_both_ways`.

  Re-run a failure alone, and on an export of the parent commit, before calling it real. Never
  touch a process you did not start.
- **New test files.** `test_inventory_retired`'s `git grep` sees only tracked files, so a new
  module's name clash shows only once it is staged (log 08, deviation 8; log 09b, deviation 15).
  Run the suites on the staged tree.
- **Adjacent prose and attribution.**
  - A docstring or comment beside a changed line that the change makes false must be fixed. Here
    that includes the report module's docstring, which describes how a parent and a set are
    rendered.
  - A comment citing a decision names it as the user's (the 09a review had to fix one that said
    "the orchestrator's").
- **Residue and scratch.**
  - Small residue found in review is the orchestrator's, in a follow-up commit (the user,
    2026-09-29).
  - The agent works in its own scratch subdirectory, never the session scratchpad's top level.
- **Usage limits.** Agents of 09b and 09c stopped on API usage limits mid-task. Resume one with
  `SendMessage`, saying exactly where it stopped and that its staged work is intact. If an agent
  commits but sends no report, review from the commit, its log and its scratch outputs.

## 1. Before you dispatch

1. Record the branch and `HEAD`. It is `a761914`, or a later commit that touches no code; say which.
2. `git status --short` and `git diff --cached` are both empty, and `git worktree list` shows this
   checkout alone. If another session has uncommitted work here, do not dispatch; ask the user.
3. `python -m RunRegistry list`: nothing `running`. Report any stale or unknown runs, without acting
   on them.
4. Snapshot `var/datastores/`, `var/runs/` and the root `physics-test-n20-lambdacdm-zend0p1*.sqlite`
   read-only, with `orch_04_snapshot.py` in the session scratchpad. Write it again if it is
   missing: the SHA-256, size and mtime of every file, and the directory list. Open files only as
   bytes, never as databases, and never put the script in the tree. The 09 session's snapshot held
   46 files and 10 directories.
5. Baselines: board §5.14. AdaptiveLevin 32, ComputeTargets 665, CosmologyModels 39, Datastore 532,
   LiouvilleGreen 148 (1 skipped), RunRegistry 199. Re-measure all six on `HEAD`, serially, with
   README §7's command.
6. Count the index rows (`grep -c '^| \`\[' docs/OPEN_ISSUES.md`), and tell the agent the number.
   It is **99** at `a761914`.
7. **Render the fixture report at `HEAD`** for the review's before-and-after.
   - Work in a `git archive` export of `HEAD` under the session scratchpad, with `venv` symlinked.
   - Call `read_inventory(build_full_store(<tmpdir>).primary)` and
     `tools.inventory_report.format_inventory_report(inventory, "build_full_store", verbose=…)`,
     once verbose and once not, and save both texts.
   - Record each class's record-line lengths.
   - This needs no Ray and opens nothing under `var/`.

## 2. Dispatch

One fresh-context subagent, on Sonnet. Give it:
- the prompt;
- the README, §5 and §5.1;
- logs 09b and 09c;
- `HEAD`, the baselines and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad. The
  agent does not read `orchestrator/`.

Tell it plainly:
- **One commit**, with its log at
  `logs/09d-refer-to-derived-parents-by-label-in-the-inventory-report.md`, in README §5.1's form,
  with the prompt's §7 additions.
- **In the same commit:**
  - this board: its header status line, its §1 row for 09d, the issue moved from §3.3 to §4 with a
    closing note naming the commit by subject, a §3.3 entry for anything opened, and
    `### 5.15 After prompt 09d (<date>)`;
  - `docs/OPEN_ISSUES.md`: the row deleted, any opened rows added, the count and date corrected by
    counting.
- **The files it may change:**
  - `tools/inventory_report.py`, including its module docstring;
  - new test modules under `Datastore/tests/`;
  - the log, the board and the index.

  **Nothing else.** No existing test, `inventory_report_parsing.py` included; not
  `Datastore/store_inventory.py`; no factory, witness, golden or `main.py`; nothing in
  `orchestrator/`.
- **Constraints.**
  - No Ray, and nothing opens anything under `var/`; every store is in a temporary directory.
  - Every mutation diff is pasted from `git diff` and replayed twice, as §0 says.
  - Run `black`, and the six suites serially on the staged tree.
- **Stop and ask** if any of the prompt's §5 stop conditions triggers, and in particular if:
  - an existing test fails;
  - a class with no sharded parent, and that nothing references, renders differently;
  - the fix seems to need `store_inventory.py`.

## 3. The review — ten checks

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only §2's files.
   - `--diff-filter=M` over `*/tests/*` is **empty**.
   - The diff over `Datastore/store_inventory.py`, every factory, every witness and golden,
     `main.py` and `orchestrator/` is empty.
2. **(P1) Distinct stays distinct.**
   - Read the new tests for the pairs the prompt's §3 test 1 names.
   - Then build one more pair yourself, in a scratch script against the committed code: two
     `GkSource`s with equal member counts and `z_source` ranges, differing in the Gk row at one
     `z_source`.
   - Render the verbose report, and confirm two distinct lines and two distinct set labels. Do the
     same for two `QuadSourceIntegral`s that differ in one Tk parent only.
3. **(P2) Every label is defined.** On both fixture reports, at `HEAD`, with your own regex rather
   than the agent's:
   - every `<Class>#<n>` and `S<n>` printed has exactly one defining line;
   - in the report without `verbose`, the referenced-but-not-shown listing is closed under
     reference;
   - no defining line uses a record line's prefix. Check this with `sections()`: each section's
     record count is still at most five per tag set, and `more` is right.
4. **(P3) Lines are bounded.**
   - Compare §1 step 7's measurements with a re-render at `HEAD`.
   - The `QuadSourceIntegral` line should be close to its pre-09 ~400 characters, and no record
     line may depend on the set size. Check with the agent's stand-in at two set sizes: the record
     lines must be equal in length.
   - Match the log's numbers.
5. **Nothing else changed.**
   - Diff the two fixture reports, `HEAD~1` against `HEAD`, class section by class section. Only the
     classes the log names may differ: those with a sharded parent, and those referenced, which gain
     a label.
   - Every replicated parent's text (`k`, `q`, `r`, the model, the tolerances, the policies) is
     byte-identical where it appears.
6. **The property's own tests.** `test_inventory_report.py` and `test_inventory_consumers.py` pass
   unmodified, and `test_no_digest_appears` covers the new lines. `HEX64` finds nothing in either
   report.
7. **Labels are stable.** Re-render after `relabel_serials`, and confirm every label maps to the same
   physical row.
8. **Replay the mutations** from `git show HEAD:<log>`, with plain `git apply`, in a git-initialised
   export of `HEAD`.
   - Each gives exactly its logged failing entries, sub-test parameters included.
   - Compare entry by entry, after first checking the log's list format; 09c's differed from 09b's.
   - Revert each, leaving the export's tracked diff empty.
9. **Boards.** The §1 row, the issue on §4 with its closing note, §5.15, and §3.3 for anything
   opened. Count the index rows (expect 98 plus any opened), and check the date.
10. **`var/`, suites, `black`, Ray.**
    - The snapshot is identical.
    - All six suites run serially in the checkout, and each rises by exactly the tests added. Treat
      load failures as §0 says.
    - `black --check` is clean.
    - `ray.is_initialized()` is false after each new module.
    - `main.py` is byte-identical to `HEAD~1`.

**Audit probes are not run.** None imports `tools/inventory_report.py`; check that again with
`git grep -l inventory_report -- docs/`. If any now does, run all fourteen as the 09 note's review
did.

## 4. After the review

Record it on the board, after the 09c review, in that review's form. Fix any small residue in a
follow-up commit of its own.

**Then prompt 10 is next.** It is the first real run: a throwaway store, built through the
registry, then resumed, then extracted and fingerprinted. Its own orchestrator note carries the
reviews of 09a, 09b, 09c and 09d. With 09d landed, that store's `main.py --inventory` is readable,
and prompt 10 may use it as evidence.
