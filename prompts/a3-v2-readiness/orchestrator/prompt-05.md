# Orchestrator — prompt 05, prune on resume

Read [`../README.md`](../README.md) first: §0, §1, §4 and §5. Then read [`prompt-04.md`](prompt-04.md)
§0 and this board's *Orchestrator review of prompt 04*. Together they are the conventions this
note does not repeat.

**You do not write code.** You may:
- run the suites and the tests the log names;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process check from the session scratchpad. Never commit one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`05-prune-on-resume.md`](../05-prune-on-resume.md)
**Board items:** P1–P2 · **Closes:** `[10-a3-v2-launch-resumes-without-pruning-unvalidated-rows]`,
on the `datastore-integrity` board's §4 · **Model:** Sonnet, as the prompt recommends.
**Gate:** the prompt has none, beyond a clean tree. 04 has landed and been reviewed (`79ab21a`;
review `d2db3a2`), so its change to `run_build` is committed. `HEAD` is the commit that adds this
note, or a later one that touches no code. One prompt at a time in this checkout.

## 0. What makes this prompt unusual

**The fix is one token removed from two lists. The review rests on proving that nothing else moved.**
The flag goes from two `sys.argv` literals, which have no test today. To test them, P1 moves both
lists into a function. A refactor like that can drop, reorder or retype an element without
breaking anything that runs, because nothing that runs reads the list except `main.py`. So P2's
element-by-element comparison carries most of the weight. The review also checks it by hand
against the pre-images.

**The prompt was amended with this note** (P2's first bullet, breakage (v), acceptance 3). Each
amended line is dated 2026-10-03.
- **An abbreviation also disables pruning.** `main.py`'s parser is `argparse.ArgumentParser()`
  (`main.py:128`), with `allow_abbrev` at its default. So `--no-prune` resolves to
  `--no-prune-unvalidated` as surely as the full spelling. No other option of `main.py` starts with
  `--no-p`. A test that looks only for the full token would pass with `--no-prune` in the list,
  and "nothing else in them disables pruning" has to cover that. Breakage (v) pins it.
- **`resumed-d` ran before prompt 02.** It ran on 2026-10-01 at 23:54. Prompt 02 (`b647c0b`)
  later moved the prune of the **replicated** classes (`BackgroundModel` and its tags and values)
  from the actors into the pool, under a `prune` record. The 45 `GkSource` rows that `resumed-d`
  removed are a **sharded** class, and an actor still prunes those
  (`Datastore._validate_on_startup`; `prune = self._prune_unvalidated and cls_name not in
  self._replicated_tables`). So the measurement still stands for the duplicates it found. The
  closing note says so, and names 07's L2 as the measurement at the tree A3 v2 will run.

**The facts, checked by the orchestrator on 2026-10-03 at `7746871`.** Pass them on; the agent
need not rediscover them. Each is marked as audit 00 marks its findings.

- **The two lists today** [read].
  - `run_build`:
    ```python
    sys.argv = ["main.py", "--database", str(database), "--job-name", args.job_name,
                "--shards", str(SHARDS), "--zend", "0.1", "--source-samples-log10z", "100",
                "--no-prune-unvalidated"]
    ```
    Here `database` is `Path(args.database).resolve()`, computed at the top of `run_build`. The
    list does not depend on `--resume`, which only decides the pre-check.
  - `run_child`:
    ```python
    main_args = ["--database", str(SWEEP_STORE), "--job-name",
                 f"handover-atol-sweep-a{args.atol:g}-r{args.rtol:g}", "--shards", str(SHARDS),
                 "--zend", "0.1", "--source-samples-log10z", "100", "--no-prune-unvalidated"]
    sys.argv = ["main.py"] + main_args
    ```
  - Each driver then prints `main.py argv = {sys.argv[1:]}`. That line stays, and so does what it
    prints, less the flag.
- **The function must live at module scope** [read]. `run_build` begins `import ray`, and
  `run_child` imports it after its store check. A list built inside either cannot be reached
  without Ray (README §5 rule 9). The module's own imports are light: the standard library and
  `numpy`. `test_quadsource_atol_sweep_prepare._load_script` already imports it by path.
- **What the parsed arguments look like** [measured, from the scratchpad]. `main()` builds its
  parser inline, so there is no parser to import. Replace `run_build` and `run_child` on a loaded
  copy of the module with functions that keep their argument, set `sys.argv`, and call `main()`.
  That gives the exact namespace each path receives:
  - `--build --database <p> --atol 1e-32 --rtol 1e-7` gives `job_name='handover-A3-baseline-v2'`,
    `resume=False`, `cpus=6`, `phase='main'`, `register=None`;
  - `--child --atol 1e-32 --rtol 1e-7` gives the same defaults, with `database=None`.

  This is one way to do §3 item 3 without running a build, through the real parser and its real
  defaults. A test may use it too, or build an `argparse.Namespace`. That is the *prompt's
  choice*. **`main()` itself is not refactored.**
- **`/tmp` is `/private/tmp` on macOS.** `Path.resolve()` turns one into the other, so a test that
  builds a path under `tempfile` and compares `--database` must expect the resolved form, or pass
  a path that is already resolved.
- **What pruning does at a resume, at `HEAD`** [read]. This is the first stop condition's
  question.
  - `main.py` reads `args.prune_unvalidated` once: `ShardedPool(..., prune_unvalidated=...)`
    (`main.py:4179`). It has no other use.
  - In the pool (since 02), after the check at open and before any actor exists,
    `_prune_replicated_tables` prunes unvalidated `BackgroundModel` rows under a `prune` record.
  - In each actor, `_validate_on_startup` prunes the sharded classes. It reports the replicated
    ones and never prunes them.
  - Both are the designed recovery (U1, and the user's decision of 2026-10-01). A store whose
    `replication_in_flight` table was created before `b647c0b` keeps `controller_shard NOT NULL`
    and refuses a pruning open that has a replicated row to prune, with a message saying so
    (`ShardedPool.py`, near its line 2137).
    - A3 v2's store does not exist yet. It will be created at this tree or later, so that refusal
      cannot reach it.
    - The sweep store `run_child` writes into is a copy of the A3 baseline. It has been retired:
      only its sidecar is left in `var/datastores/`.
    - Neither fires the stop condition. The agent records the second in *Observations* if it
      judges it bears on `run_child`.
- **No existing test pins the old lists** [read]. This is the second stop condition's question.
  - Nothing under any `tests/` directory contains `no-prune` or a `sys.argv` of either driver.
  - `test_killed_handler.TestTheDrivers` reads `run_build` with `ast`. It finds `_local_ray_init`,
    one `exec`, one `try`, and `finish_after_exec` after it.
  - `test_store_fingerprint.TestDrivers` counts `run.finish(` calls: (3, 2) for this file.
  - `docs/a3-v2-readiness/sigterm_handler_probe.py` extracts `run_build`'s `_local_ray_init` by
    `ast`.

  All four still hold if the list moves out and nothing else in `run_build` moves. If one fails,
  the agent moved something else.
- **`main.py`'s default** [read and parsed]. `main.py:235`:
  `parser.add_argument('--prune-unvalidated', action=argparse.BooleanOptionalAction, default=True,
  help=...)`. It is the only `add_argument` for that name.
- **`substitute_main_py(filter_cases=False)`** [measured at `7746871`]. It replaced each of its four
  literals exactly once: the two `np.logspace` grid lines, `tol=DEFAULT_QUADRATURE_ATOL` and
  `tol=DEFAULT_QUADRATURE_RTOL`.
- **The usage lines** [read]. `docs/gktk-remedial/scoped_pipeline_run.py:40` and
  `docs/source-remediation-verification/scoped_pipeline_run.py:25` both end
  `--zend 1e4 --source-samples-log10z 100 --no-prune-unvalidated`. No test reads either module
  docstring.
- **Out of scope in `run_child`.** `[04-the-sweeps-run-child-shares-the-sigterm-and-cluster-address-defects]`
  is open on this board: no `address="local"`, and no handler. 05 changes `run_child`'s list and
  nothing else. It does not fix that issue, and it does not touch `sweep`.
- **The issue and the index.** `[10-a3-v2-launch-…]` is in the `datastore-integrity` board's
  §3.3, with two *Assigned* lines, and closes to that board's §4 with both kept. The index is
  **104** at `7746871`. After 05 it is **103**, plus one for each issue 05 opens.

**What the review exists to establish.**
- **(E1) The flag is gone** from both lists and both usage lines, and nothing in either list
  disables pruning, abbreviations included.
- **(E2) One function builds each list** (one, or one per mode). `run_build` and `run_child` call
  it, and it needs no Ray.
- **(E3) Pruning is the default.** The test reads `main.py`'s default from source with `ast`.
- **(E4) Nothing else moved.** Each list, less the flag, equals its pre-image element by element.
  The test pins this, and the orchestrator checks it by hand.
- **(E5) The pins.** Breakages (i)–(v) each fail a named test when replayed with `git apply`.
- **(E6) The records.** The issue is closed on the `datastore-integrity` board, citing `resumed-d`
  and the 02 caveat. The index is 103.

**Carried from the reviews of 00–04.** All bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, never `git add -A`, `.` or `commit -a`.
  Show `git diff --cached --name-only` before committing. Count, then paste. A log states only
  what the diff and the tests show.
- **The SHA, not "this commit".** The agent cannot know its own SHA. Fixing that stays the
  orchestrator's residue.
- **Suites.** Use README §7's command, with output to a file and the verdict grepped from it.
  Under load, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`,
  `test_background_tau.test_build_cost` and `test_numeric_phase_cut.test_the_print_is_gated_both_ways`
  fail. Re-run a failure alone before calling it real. If it persists alone, settle it on an export
  of the tree before the prompt.
- **Breakage diffs** are recorded so that `git apply` replays them from a clean tree at the
  agent's commit. Replay them in `bash`, not `zsh`. Breakage (iii) is a diff to `main.py`, applied
  and reverted. `main.py` is not in the commit.
- **Scratch.** The agent works in its own subdirectory of the session scratchpad, never its top
  level.
- **Ray.** 05 starts no Ray. Another project's Ray cluster (ChamPBH's) is running on this machine,
  and it is the user's. It is left alone.
- **Usage limits.** Resume a stopped agent with `SendMessage`, saying exactly where it stopped. If
  it commits but sends no report, review from the commit and its log.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` are both empty, and `git worktree list` shows
     this checkout alone.
   - `git log --oneline 7746871..HEAD` touches no code.
2. **The registry.** `./venv/bin/python -m RunRegistry list`: nothing `running`. At `7746871` it
   held 12 runs: 10 finished and 2 `unknown` (`a3-pilot`, `a3-pilot-resume`).
3. **Ray.** Record `pgrep -lf 'gcs_server|raylet|ray::'`. No listed process may come from this
   repository's `venv`. ChamPBH's, from its own venv, are expected. Note them and leave them alone.
4. **Baselines.** §5.9: AdaptiveLevin 32, ComputeTargets 686, CosmologyModels 39, Datastore 598,
   LiouvilleGreen 148 (1 skipped), RunRegistry 225. Re-measure all six at `HEAD`, serially, and
   record the load average.
5. **The index.** `grep -c '^| \`\[' docs/OPEN_ISSUES.md` gives **104**. After 05 it is 103. Tell
   the agent both numbers, and that it rises by one for each issue it opens.
6. **The pre-images**, for checks 1 and 4. Record `git rev-parse HEAD:<path>` for
   `docs/handover/quadsource_atol_sweep.py`, `docs/gktk-remedial/scoped_pipeline_run.py`,
   `docs/source-remediation-verification/scoped_pipeline_run.py` and `main.py`.

## 2. Dispatch

One fresh-context subagent, on **Sonnet**. Give it:
- the prompt, as amended;
- the campaign README, this board, and the `datastore-integrity` entry;
- `HEAD`, the baselines and both index counts;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/05-prune-on-resume.md`, in README §5.1's form,
  with the prompt's §7 addition: both lists before and after.
- **The files it may add or change:**
  - `docs/handover/quadsource_atol_sweep.py`: `run_build`, `run_child`, and the new module-level
    function or functions. Nothing else in it: not `main()`, `sweep`, `substitute_main_py` or the
    module docstring;
  - `docs/gktk-remedial/scoped_pipeline_run.py` and
    `docs/source-remediation-verification/scoped_pipeline_run.py`: the one token in the usage line
    of each module docstring;
  - one new test module under `RunRegistry/tests/`;
  - the log; this board; the `datastore-integrity` board (the issue to its §4);
    `docs/OPEN_ISSUES.md`.

  **Nothing else.** In particular:
  - no `main.py` in the commit, no pool and no `RunRegistry/*.py`;
  - no existing test and no existing probe;
  - nothing under `var/` and nothing under `orchestrator/`;
  - in `run_child`, no `address="local"` and no handler. That is
    `[04-the-sweeps-run-child-…]`'s, and stays open.
- **Stop and ask** on either of the prompt's §5 conditions. Say that this note's facts answer
  both as "does not fire" at `7746871`, and that a test which fails after the move is a sign that
  something else moved, not a reason to edit it.

## 3. The review — ten checks

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only §2's list.
   - `git diff HEAD~1 HEAD -- main.py Datastore/ config/ var/ 'RunRegistry/*.py' docs/a3-v2-readiness/`
     is empty.
   - Compare against the parent with `ast`:
     - in `quadsource_atol_sweep.py`, only `run_build` and `run_child` changed, and the only
       additions are the new module-level functions;
     - in each `scoped_pipeline_run.py`, the module with its docstring removed dumps identically,
       and the docstring differs by the one token.
   - Under `*/tests/`, the diff adds one file and changes none.
2. **(E1) The flag.**
   - `grep -n 'no-prune\|prune' ` over the four touched code and docstring files: nothing left in
     either list or usage line.
   - The printed `main.py argv = …` line is still there in both drivers.
3. **(E2) The function, by reading.**
   - It is at module scope, imports nothing from `ray`, and takes the parsed arguments and the
     database path.
   - `run_build` and `run_child` each call it once, and assign its result to `sys.argv`.
   - Nothing else in `run_build` moved: `TestTheDrivers` and `TestDrivers` pass untouched.
4. **(E4) The lists, by hand.**
   - Rebuild the two pre-image lists from `git show HEAD~1:docs/handover/quadsource_atol_sweep.py`,
     less the flag.
   - Run the function from the scratchpad, through `main()`'s parser as in §0, for `--build`, for
     `--build --resume` and for `--child`. Compare element by element, with types (every element
     a `str`).
   - Compare with the log's before and after lists.
5. **(E3) The tests, by reading and running.**
   - Run the new module, `test_quadsource_atol_sweep_prepare`, `test_killed_handler` and
     `test_store_fingerprint`.
   - Read that it covers the build with and without `--resume`, and the child.
   - Read that it rejects an abbreviation of `--no-prune-unvalidated`, not only the full token.
   - Read that `main.py`'s default is read with `ast`, not imported, and fails if the default is
     not the constant `True`.
   - Read that the expected lists are literals or the pre-image's, not computed from the function
     under test.
6. **(E5) The pins.** In `bash`, replay each of the log's breakage diffs (i)–(v) with
   `git apply`.
   - Run the named test, and see it fail.
   - Revert with `git apply -R`, and check that `git status` is clean after each.
   - (iii) mutates `main.py`. Check `git diff -- main.py` is empty after the revert.
7. **Verification 2 and 3.** `substitute_main_py(filter_cases=False)` matches each of its four
   substitutions once. The log shows the list for verification 3's command line, and it matches
   check 4.
8. **The stop conditions.**
   - The log says what pruning does on a resume at `HEAD`, as §0 does, and why neither stop
     condition fired.
   - The log does not say a build or resume was run, because none was.
9. **(E6) The records.**
   - The issue is in the `datastore-integrity` board's §4, with both *Assigned* lines kept.
   - Its closing note cites `resumed-d` (verification §2.1.4), names the new tests, and carries
     the 02 caveat.
   - Its index row is gone, and the count is 103 by counting, matching the header.
   - This board has the §1 row, P1–P2 and §5.10.
   - *State handed to the next prompt* says that 07's `--build --resume` now prunes at startup,
     and what L2 should see.
10. **Suites.** All six serially, at `HEAD`.
    - RunRegistry is 225 plus 05's new tests.
    - Every other count is unchanged.
    - `black --check` is clean on every touched Python file.

## 4. After the review

- **Record it on this board**, after the prompt-04 review and in its form, with `HEAD` at
  dispatch, the checks and any remnants. Fix small residue in a follow-up commit of its own,
  including "this commit" → the SHA, and the orchestrator notes list (`prompt-05.md` "used").
- **Report to the user:**
  - what `--build --resume` now does at startup, against what it did;
  - the two lists before and after;
  - which issue closed, and what remains between the campaign and 07 (06).
- **What 05 hands on.** 07's L1–L2 interrupt a registered `--build` and resume it. Its note must
  carry:
  - that the resume now prunes at startup. Each actor prints its unvalidated rows and
    `** Note: these values have been pruned from the datastore.` (`resumed-d`'s
    `resumed_d_pruning.txt`), and Ray may fold the lines as `[repeated Nx across cluster]`;
  - that an unvalidated `BackgroundModel` is pruned by the pool, under a `prune` record, and that
    `resumed-d` never exercised this, since it ran before 02;
  - that the inventory of the resumed store should report no `duplicate`, and its fingerprint
    should equal the uninterrupted reference's.
