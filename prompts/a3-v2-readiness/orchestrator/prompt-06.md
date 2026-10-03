# Orchestrator — prompt 06, count only real references

Read [`../README.md`](../README.md) first: §0, §1, §4 and §5. Then read [`prompt-05.md`](prompt-05.md)
§0 and this board's *Orchestrator review of prompt 05*. Together they are the conventions this
note does not repeat.

**You do not write code.** You may:
- run the suites and the tests the log names;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process check from the session scratchpad, or a check on an export of the tree there.
  Never commit one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`06-count-only-real-references.md`](../06-count-only-real-references.md)
**Board items:** F1–F2 · **Closes:** `[10-store-retire-reports-every-tombstone-as-a-reference]`, on
**this** board's §4 (it is held in §3.2) · **Model:** Sonnet, as the prompt recommends.
**Gate:** the prompt has none, beyond a clean tree. 05 has landed and been reviewed (`dd7f272`;
review `c372bee`). `HEAD` is the commit that adds this note, or a later one that touches no code.
One prompt at a time in this checkout.

## 0. What makes this prompt unusual

**It narrows a report that exists to find real references, so the review has to show two things.**
The false matches must go, and **no real one may go with them.** A report that lost the backup's
directory match would be quieter and wrong, and nothing would refuse. So the review rests on:
- the existing `test_a_retirement`, unchanged, which already requires the subdirectory match;
- breakage (ii), which shows that test fails when the directory is dropped outright;
- a reading of the four tombstones already under `var/datastores/`, which shows that none of the
  matches F1 removes was real (§0 below, measured by the orchestrator).

**The prompt needs no amendment.** Reading the code at `c372bee`, and running F1 on an export of the
tree, turned up nothing it could not pass and nothing it had wrong. Neither stop condition fires.

**The facts, checked by the orchestrator on 2026-10-03 at `c372bee`.** Pass them on; the agent
need not rediscover them. Each is marked as audit 00 marks its findings.

- **The code today** [read]. `RunRegistry/stores.py`:
  - `_find_references(primary, planned, store_id, runs_root, stores_root, own)` builds
    `targets = {os.path.realpath(p) for p in [primary, *planned, primary.parent]}`. It then walks
    `os.walk(stores_root)`, sorted, over every `*.manifest.json` but the store's own.
  - A sidecar matches by `$.copied_from.store_id`, or by any string anywhere in it that
    `_names_one_of` reads as one of the targets. `_names_one_of` resolves the string with `_resolve`
    (an absolute path as it is, a relative one against `REPO_ROOT`) and then `os.path.realpath`.
  - `retire_store` takes `stores_root = str(stores_root or DEFAULT_STORES_ROOT)`, where
    `DEFAULT_STORES_ROOT` is the absolute `<repo>/var/datastores`. `store retire --stores-root`
    passes the CLI's value through as given.
  - The tombstone records `references.stores_root` as `_repo_path(stores_root)`. That is
    `var/datastores` in the repository, and an absolute path for a temporary root.
- **The F1 comparison.** `os.path.realpath(primary.parent) != os.path.realpath(stores_root)`.
  That is the same cwd-relative resolution `os.walk(stores_root)` uses, so the root that is
  compared is the root that is searched. A textual or `abspath` comparison misses a symlinked
  root (breakage (iii)).
- **Measured on an export of `c372bee`** (orchestrator, 2026-10-03, scratchpad, not committed).
  F1 was applied as the three-line form above to `RunRegistry/stores.py` alone.
  - **The RunRegistry suite, 243 tests.** 241 pass, including `test_a_retirement`. The other two,
    `test_run_registry.TestManifest.test_manifest_carries_the_provenance_triple` and
    `test_store_sidecar.TestCreate.test_the_fields_and_one_create_entry`, fail on `git_head`
    because an export is not a git repository. Neither touches `_find_references`. **No existing
    test has to change** (stop condition 1).
  - **Three stores directly in the fixture's stores root, retired in turn.** `a_full_store("",
    stem=…)` puts a store directly in the root.
    - At `c372bee` the reports list 0, then 1, then 2 earlier tombstones. Each match is
      `$.retired.references.stores_root`.
    - Two more stores, retired through a **symlink** to the same root, list 3, then 4.
    - With F1, all five list none.
    - So breakage (iii) is just this: a textual comparison, and a test that retires through a
      symlinked root. `StoreTestCase.top` is already `.resolve()`d, so the stores root and every
      `primary.parent` are real paths, and only the symlink differs textually.
- **The tombstones under `var/datastores/`** [read, never written]. These answer stop condition 2.
  - `datastore-integrity-10-reference` (2026-10-02, the first of the two) lists 2 sidecars:
    `handover-atol-sweep` and the backup's. Both match by `$.retired.references.stores_root` alone.
  - `datastore-integrity-10-resumed` (the second) lists 3: `datastore-integrity-10-reference`,
    `handover-atol-sweep` and the backup's. All three match by `$.retired.references.stores_root`
    alone.
  - `handover-atol-sweep` (2026-09-25) lists none.
  - The backup, `backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm` (2026-09-25),
    lists the live A3 sidecar by `$.backup.path`. That match is **real**, and it is the
    subdirectory shape, which F1 keeps.
  - **None of the five matches F1 removes is a real reference**, so stop condition 2 does not fire.
    No tombstone is rewritten (§6).
- **What `test_a_retirement` pins** [read, `RunRegistry/tests/test_store_retire.py`].
  - The store is in `<stores root>/S`, beside a second store `other`. A sidecar under
    `datastores/D` names `S` by `$.backup.path` (absolute) and `$.backup.relative` (relative to
    `REPO_ROOT`).
  - The expected sidecar set is exactly {the copy's, the backup's, the junk file}, and the
    backup's fields are exactly `["$.backup.path", "$.backup.relative"]`.
  - So it pins the subdirectory match (breakage (ii)), and that nothing else in that directory is
    reported.
- **§7's reconstruction**, of the second 2026-10-02 retirement. At the time the stores root held:
  - three tombstones directly in the root (`handover-atol-sweep`, `datastore-integrity-10-reference`
    and the store being retired), each with `references.stores_root` naming that root;
  - the live A3 sidecar and its store files, whose `backup.path` names the backup's subdirectory;
  - the backup's tombstone in its subdirectory.

  The rebuild goes in a temporary stores root, never in `var/`. The expected result is three
  sidecars before, each by `$.retired.references.stores_root`, and none after. Retiring real
  stores in sequence, or writing tombstone-shaped sidecars, is the *prompt's choice*.
- **The optional tombstone skip** (prompt F1, *prompt's choice*). The tombstone's shape check,
  `_retired_problems`, requires only `runs`, `sidecars`, `runs_root` and `stores_root` in
  `references`. F1 changes none of them. If the agent also skips a tombstone's `$.retired.references`
  subtree, the log must say what it would hide. One example: a tombstone whose `references.sidecars`
  names a sidecar file is not a reference to a store anyway.
- **`not_searched`** names what is not walked. F1 changes what is a target, not what is walked. So
  the text need not change. The docstring must say what a directory match means: a sidecar that
  names the directory the store sits in, when that directory is not the stores root. It may be
  shared with other stores, as `test_a_retirement`'s is.
- **Who reads the report.** `RunRegistry/__main__.py` prints `sidecars naming it, under …: N`
  from it, and nothing else consumes it. `__main__.py` does not change.
- **The issue and the index.** `[10-store-retire-…]` is in **this** board's §3.2 and closes to this
  board's §4. The index is **103** at `c372bee`. After 06 it is **102**, plus one for each issue 06
  opens.

**What the review exists to establish.**
- **(E1) The false matches are gone.** Sequential retirements directly in the root list no earlier
  tombstone, through a symlinked root too.
- **(E2) No real match is lost.** `test_a_retirement` passes unchanged, and breakage (ii) shows it
  fails without the directory. The new subdirectory test reports a match by the subdirectory.
- **(E3) The comparison is by `realpath`**, shown by breakage (iii).
- **(E4) Nothing else changes.** Refusals, deletions and the tombstone's other fields are as they
  were, and every existing RunRegistry test passes unchanged.
- **(E5) The pins.** Breakages (i)–(iii) each fail a named test when replayed with `git apply`.
- **(E6) The records.** The issue is in this board's §4, and the index is 102.

**Carried from the reviews of 00–05.** All bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, never `git add -A`, `.` or `commit -a`.
  Show `git diff --cached --name-only` before committing. Count, then paste. A log states only
  what the diff and the tests show.
- **The SHA, not "this commit".** The agent cannot know its own SHA. Fixing that stays the
  orchestrator's residue.
- **Suites.** Use README §7's command, with output to a file and the verdict grepped from it.
  Under load, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`,
  `test_background_tau.test_build_cost` and `test_numeric_phase_cut.test_the_print_is_gated_both_ways`
  fail. Re-run a failure alone before calling it real.
- **Breakage diffs** are recorded so that `git apply` replays them from a clean tree at the
  agent's commit. Take them with `git diff -- <the code files>` only, not the whole tree, so that
  the Markdown edits do not stop them applying (log 05's first attempt). Replay them in `bash`,
  not `zsh`.
- **A breakage must fail for its own reason.** A stand-in that raises `AttributeError` under a
  mutation catches it for the wrong reason (log 05, breakage (ii)'s first pass). Read each failure.
- **`var/`.** Read the tombstones there. Write nothing, and run no `store retire` against
  `var/datastores/`.
- **Scratch.** The agent works in its own subdirectory of the session scratchpad, never its top
  level.
- **Ray.** 06 needs none. Another project's Ray cluster (ChamPBH's) may be running on this machine.
  It is the user's, and it is left alone.
- **Usage limits.** Resume a stopped agent with `SendMessage`, saying exactly where it stopped. If
  it commits but sends no report, review from the commit and its log.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` are both empty, and `git worktree list` shows
     this checkout alone.
   - `git log --oneline c372bee..HEAD` touches no code.
2. **The registry.** `./venv/bin/python -m RunRegistry list`: nothing `running`. At `c372bee` it
   held 12 runs: 10 finished and 2 `unknown` (`a3-pilot`, `a3-pilot-resume`).
3. **`var/datastores/`.** Record a listing with sizes and modification times. At `c372bee` it held
   10 entries. The listing is re-taken in check 8.
4. **Baselines.** §5.11: AdaptiveLevin 32, ComputeTargets 686, CosmologyModels 39, Datastore 598,
   LiouvilleGreen 148 (1 skipped), RunRegistry 243. Re-measure all six at `HEAD`, serially, and
   record the load average.
5. **The index.** `grep -c '^| \`\[' docs/OPEN_ISSUES.md` gives **103**. After 06 it is 102. Tell
   the agent both numbers, and that it rises by one for each issue it opens.
6. **The pre-images**, for check 1. Record `git rev-parse HEAD:<path>` for `RunRegistry/stores.py`
   (`9f5105f` at `c372bee`), `RunRegistry/tests/test_store_retire.py` (`2d9afd9`) and
   `RunRegistry/__main__.py` (`88cf6dd`).

## 2. Dispatch

One fresh-context subagent, on **Sonnet**. Give it:
- the prompt;
- the campaign README, this board, and `prompts/store-retirement/03-retire-a-store.md` §2–§3;
- `HEAD`, the baselines and both index counts;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/06-count-only-real-references.md`, in README §5.1's
  form, with the prompt's §7 addition: `_find_references` before and after for the second
  2026-10-02 retirement, rebuilt in a temporary stores root.
- **The files it may add or change:**
  - `RunRegistry/stores.py`: `_find_references`, its docstring, and its `not_searched` text if what
    is searched changes. Nothing else in the file;
  - a new test module under `RunRegistry/tests/`, or a new class appended to `test_store_retire.py`
    (*prompt's choice*). If it appends, no existing line of that file changes;
  - the log; this board (§1 row, F1–F2, §3.2 → §4, a new §5.12); `docs/OPEN_ISSUES.md`.

  **Nothing else.** In particular:
  - no `RunRegistry/__main__.py` and no other module;
  - no existing test;
  - nothing written under `var/`, no tombstone rewritten, and no `store retire` run against the
    real stores root;
  - nothing under `orchestrator/`.
- **Stop and ask** on either of the prompt's §5 conditions. Say that this note's facts answer both
  as "does not fire" at `c372bee`. An existing test that fails is a sign that F1 went wider than
  the stores root, not a reason to edit the test.

## 3. The review — ten checks

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only §2's list.
   - `git diff HEAD~1 HEAD -- main.py Datastore/ config/ docs/ RunRegistry/__init__.py RunRegistry/__main__.py`
     is empty, apart from `docs/OPEN_ISSUES.md`.
   - Compared by `ast` with the parent, `stores.py` changes only `_find_references`.
   - If `test_store_retire.py` changed, the diff only appends.
2. **(E1, E3) F1, by reading.**
   - The root is dropped from the targets only when `realpath(primary.parent) ==
     realpath(stores_root)`. `primary` and `planned` stay targets.
   - The docstring says what a directory match means.
   - If the optional tombstone skip was taken, read the log's justification. It must be *in
     addition to* F1, and the non-tombstone test must still exercise F1 alone.
3. **(E2) The subdirectory match.** `test_a_retirement` is byte-identical to the parent and passes.
   The new subdirectory test reports a sidecar that names the store's subdirectory.
4. **(E1) The tests, by reading and running.**
   - Run the new tests and `test_store_retire`.
   - Read that the sequential test retires at least three stores directly in the root and asserts
     the **empty** list each time, not merely a shorter one.
   - Read that the non-tombstone sidecar names the stores root and is not reported.
   - Read that a test retires through a symlinked stores root.
5. **(E5) The pins.** In `bash`, replay each of the log's breakage diffs (i)–(iii) with
   `git apply`.
   - Run the named tests and see each fail for its own reason: an assertion on the report, not an
     error in the fixture.
   - (ii) must fail `test_a_retirement`.
   - Revert with `git apply -R`, and check that `git status` is clean after each.
6. **§7's reconstruction.** Read the log's before and after for the second 2026-10-02 retirement.
   They should be the three `$.retired.references.stores_root` matches of §0, and none. Re-run the
   reconstruction from the log, if it gives a script, or rebuild it from the scratchpad.
7. **(E4) Nothing else changed.**
   - All of `test_store_retire`, `test_store_command_line`, `test_store_amend` and
     `test_store_sidecar` pass unchanged.
   - The tombstone's keys are as before: run one retirement in a temporary root and compare
     `sorted(retired)` and `sorted(retired["references"])` with the parent's.
8. **`var/`.** The listing from §1 item 3 is unchanged (sizes and modification times).
9. **(E6) The records.**
   - The issue is in this board's §4, with a closing note that names the tests and cites the
     2026-10-02 tombstones.
   - Its §3.2 entry is gone, and its index row is gone. The count is 102 by counting, matching the
     header.
   - This board has the §1 row, F1–F2 and §5.12.
   - *State handed to the next prompt* says that 07 can retire its throwaway stores, and that their
     reports will list only real references.
10. **Suites.** All six serially, at `HEAD`.
    - RunRegistry is 243 plus 06's new tests.
    - Every other count is unchanged.
    - `black --check` is clean on every touched Python file.

## 4. After the review

- **Record it on this board**, after the prompt-05 review and in its form, with `HEAD` at
  dispatch, the checks and any remnants. Fix small residue in a follow-up commit of its own:
  - "this commit" → the SHA;
  - README §2's status cell for 06 ("landed …, `<sha>`");
  - the orchestrator notes list (`prompt-06.md` "used"), and the board header.
- **Report to the user:**
  - what `store retire` now reports for a store directly in the stores root, against what it did;
  - that the backup's subdirectory match still holds;
  - which issue closed, and that 01–06 have all landed, so 07 is released.
- **What 06 hands on.** 07 builds throwaway stores, and its last step may retire them. Its note
  must carry:
  - that a retirement's report now lists only sidecars that name the store, its files, or its own
    subdirectory;
  - that the four existing tombstones keep the false matches they recorded, which are correct as a
    record of what that code found (§6), so a reader comparing old and new reports will see the
    difference.
