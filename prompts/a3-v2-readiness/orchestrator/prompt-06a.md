# Orchestrator — prompt 06a, give `--build` a rehearsal scope

Read [`../README.md`](../README.md) first: §0, §1, §2, §4 and §5, and §6.1's D5. Then read
[`prompt-06.md`](prompt-06.md) §0 and this board's *Orchestrator review of prompt 06*. Together they
are the conventions this note does not repeat.

**You do not write code.** You may:
- run the suites and the tests the log names;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process check from the session scratchpad, or a check on an export of the tree there.
  Never commit one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`06a-give-the-build-a-rehearsal-scope.md`](../06a-give-the-build-a-rehearsal-scope.md)
**Board items:** B1–B3 · **Closes:** nothing · **Model:** Sonnet, as the prompt recommends.
**Gate:** 06 has landed and been reviewed (`938be2b`; review `118ca94`). `HEAD` is the commit that
adds this note and the prompt, or a later one that touches no code. One prompt at a time in this
checkout.

## 0. What makes this prompt unusual

**It edits A3 v2's own launcher, to add a mode that A3 v2 must never use.** The flag is for 07's
rehearsal. A3 v2 is launched without it, so the review rests on proving two things:
- **without the flag, nothing changes**: the argv list, the k sample, the manifest's `scope` text,
  and every function but the ones the prompt names;
- **with it, all three move together**, and only these change: the grid and `--zend` in what
  `main.py` runs, and the text of the record that describes the run.

A scope that changed `--zend` but not the grid, or the grid but not the manifest, would run without
error and be recorded wrongly. That is why the k sample, the argv and the scope text come from one
function.

**B3 rides along for the same reason as B1.** 07 interrupts its builds with `datastore-integrity`'s
`interrupt_when.py`, which cannot fire on a `--build` today. If 07 changed it, each of 07's launches
would run with a tracked file modified. Landing it here leaves 07 nothing to change but new files.

**The prompt needs no amendment.** It was written on 2026-10-03 at `118ca94`, from the code read
below.

**The facts, checked by the orchestrator on 2026-10-03 at `118ca94`.** Pass them on; the agent
need not rediscover them. Each is marked as audit 00 marks its findings.

- **Where the scope lives today** [read]. `docs/handover/quadsource_atol_sweep.py`:
  - `K_MIN, K_MAX, K_COUNT = 1e5, 3e8, 8`; `k_sample()` is
    `np.logspace(np.log10(K_MIN), np.log10(K_MAX), K_COUNT).tolist()`. `run_build`, `run_child` and
    `sweep` all reach it.
  - `run_build`'s `exec` namespace sets `K_GRID_NAME: k_sample()`, where `K_GRID_NAME` is
    `"SWEEP_K_SAMPLE"`. `substitute_main_py` puts it in place of both of `main.py`'s grid literals.
  - `--zend` is the literal `"0.1"` in `_main_py_argv`, which `build_main_argv` and
    `child_main_argv` share.
  - The manifest's `scope=` in `run_build`:
    `f"full build: main.py's own 9280-item QuadSourceIntegral work list at atol={args.atol:g}
    rtol={args.rtol:g}, {args.cpus} cpus, LambdaCDM, 8 log-spaced k over {K_MIN:g}-{K_MAX:g} /Mpc,
    into {database.name}"`, as three adjacent f-strings.
- **The rehearsal scope is D10-1's exactly** [measured]. `np.logspace(5, 6, 3).tolist()` and
  `np.logspace(np.log10(1e5), np.log10(1e6), 3).tolist()` are both
  `[100000.0, 316227.7660168379, 1000000.0]`, equal as floats. `datastore-integrity` 10's
  `reference` manifest records `--k-min 1e5 --k-max 1e6 --k-count 3` and `--zend 250
  --source-samples-log10z 100 --shards 4`, through `scoped_pipeline_run.py`, which builds its grid
  the second way. So 07's stores should hold the same rows by class as that store, except for the
  tolerance pair (`docs/datastore-integrity-verification.md` §2.1.5).
- **What pins `run_build` today** [read]. All must pass after 06a. Only the one `Namespace` the
  prompt names may change.
  - `test_prune_on_resume`, 18 tests:
    - the production list element by element;
    - `ast.unparse` of `run_build`'s one `sys.argv` assignment equals
      `"build_main_argv(args, database)"`;
    - no string constant beginning `--` without a space in `run_build` or `run_child`. So
      `run_build` must not spell `"--rehearsal"` or `"--zend"`;
    - `_main_py_argv`, `build_main_argv` and `child_main_argv` import nothing.
  - `test_the_job_name_is_the_namespaces` passes `argparse.Namespace(job_name=…, resume=False)` to
    `build_main_argv`. If that function reads `args.rehearsal`, the test raises `AttributeError`.
    Hence the prompt's one permitted change.
  - `test_killed_handler.TestTheDrivers` finds `_local_ray_init`, one `exec`, one `try`, and
    `finish_after_exec` after it, in `run_build`.
  - `test_store_fingerprint.TestDrivers` counts `run.finish(` calls per driver: (3, 2) for this
    file.
  - `docs/a3-v2-readiness/sigterm_handler_probe.py` extracts `run_build`'s `_local_ray_init` with
    `ast`.
  - `test_quadsource_atol_sweep_prepare` loads the module by path.
- **`substitute_main_py(filter_cases=False)`** [measured at `118ca94`]. It replaced each of its four
  literals exactly once: the two `np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_*_K_VALUES)`
  lines, `tol=DEFAULT_QUADRATURE_ATOL` and `tol=DEFAULT_QUADRATURE_RTOL`. 06a does not touch it.
- **`main()`'s dispatch order** [read]. `--prepare`, then `--report`, then `--build`, then
  `--child`, then the sweep. `--prepare` copies the A3 baseline's shards, and the sweep launches a
  child build per tolerance pair. So the refusal of `--rehearsal` must come first, and a test of a
  refusal stubs every mode.
- **The interrupter** [read]. `docs/datastore-integrity-verification/interrupt_when.py` (blob
  `4caa9b2`):
  - `DRIVER = "scoped_pipeline_run.py"`;
  - `names_the_run(command, slug)` requires `DRIVER in command` and the tokens
    `--register <slug>`;
  - `run_slug(run_dir)` reads the slug from the manifest's `argv`;
  - the refusal's log line names `{DRIVER}`.

  `RunRegistry.begin` records `"argv": list(sys.argv)` (`RunRegistry/__init__.py`, near line 574).
  `run_build` calls `begin` before it replaces `sys.argv` with `main.py`'s, so a `--build`
  manifest's `argv[0]` is `docs/handover/quadsource_atol_sweep.py`.
  `datastore-integrity` 10's `reference` manifest has `argv[0]`
  `docs/gktk-remedial/scoped_pipeline_run.py`. `interrupter_dummy_test.sh` writes manifests with
  that `argv[0]`, and runs its children as `exec -a "python docs/gktk-remedial/scoped_pipeline_run.py
  --register dummy-…" sleep …`. So reading the driver from `argv[0]` leaves the dummy test's five
  cases as they are.
- **The issue and the index.** 06a closes nothing. The index is **102** at `118ca94`, and stays 102
  unless 06a opens an issue.

**What the review exists to establish.**
- **(E1) Production is unchanged.** Without the flag the argv list, the k sample and the scope text
  are exactly today's, by test and by the orchestrator's own check from the parent's `ast`.
- **(E2) The rehearsal moves all three together.** With the flag the k sample is
  `[1e5, 316227.7660168379, 1e6]`, `--zend` is `250`, and the scope text names both and does not say
  `9280`. One function decides all three, and `run_build` reads the grid and the scope from it.
- **(E3) It is refused outside `--build`**, before any mode runs.
- **(E4) Nothing else moved.** By `ast` against the parent, only `run_build`, `build_main_argv`,
  `_main_py_argv` and `main` change, plus the new function and constants. The existing tests pass,
  with at most the one `Namespace` changed.
- **(E5) The pins.** Breakages (i)–(vi) each fail a named test, or B3's dummy case, when replayed
  with `git apply`.
- **(E6) The interrupter** takes the driver from the run's manifest. The existing dummy test passes
  unchanged, and the new case signals a `quadsource_atol_sweep.py` child and refuses a
  `scoped_pipeline_run.py` one under the same manifest.

**Carried from the reviews of 00–06.** All bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, never `git add -A`, `.` or `commit -a`.
  Show `git diff --cached --name-only` before committing. Count, then paste. A log states only
  what the diff and the tests show.
- **The SHA, not "this commit".** The agent cannot know its own SHA. Fixing that stays the
  orchestrator's residue.
- **Suites.** Use README §7's command, with output to a file and the verdict grepped from it.
  - Under load, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`,
    `test_background_tau.test_build_cost` and
    `test_numeric_phase_cut.test_the_print_is_gated_both_ways` fail. Re-run a failure alone before
    calling it real.
  - **At 06's review, `test_wall_time_per_object` failed at every run**, alone too, at loads of
    15–34. Another session was running ChamPBH's own test suite. It failed the same way on an export
    of `dd7f272`, where it had passed, so it is load. If it fails again, say so and settle it the
    same way. It is not a reason to stop.
- **Breakage diffs** are recorded so that `git apply` replays them from a clean tree at the agent's
  commit. Take each with `git diff --` the one code file it changes, never the whole tree. Replay them in
  `bash`, not `zsh`. A breakage must fail for its own reason: an assertion, not an
  `AttributeError` in a stand-in.
- **`var/`.** Nothing is read or written there. 06a launches nothing.
- **Scratch.** The agent works in its own subdirectory of the session scratchpad, never its top
  level.
- **Ray.** 06a starts none. ChamPBH's Ray cluster may be running on this machine. It is the user's,
  and it is left alone.
- **Usage limits.** Resume a stopped agent with `SendMessage`, saying exactly where it stopped. If
  it commits but sends no report, review from the commit and its log.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` are both empty, and `git worktree list` shows
     this checkout alone.
   - `git log --oneline 118ca94..HEAD` touches no code.
2. **The registry.** `./venv/bin/python -m RunRegistry list`: nothing `running`. At `118ca94` it
   held 12 runs: 10 finished and 2 `unknown` (`a3-pilot`, `a3-pilot-resume`).
3. **Baselines.** §5.13: AdaptiveLevin 32, ComputeTargets 686, CosmologyModels 39, Datastore 598,
   LiouvilleGreen 148 (1 skipped), RunRegistry 253. Re-measure all six at `HEAD`, serially, and
   record the load average.
4. **The index.** `grep -c '^| \`\[' docs/OPEN_ISSUES.md` gives **102**.
5. **The pre-images**, for checks 1 and 3. Record `git rev-parse HEAD:<path>` for:
   - `docs/handover/quadsource_atol_sweep.py` (`41f7c72` at `118ca94`);
   - `RunRegistry/tests/test_prune_on_resume.py` (`281f432`);
   - `test_killed_handler.py` (`213f2a8`), `test_store_fingerprint.py` (`ed5bd65`) and
     `test_quadsource_atol_sweep_prepare.py` (`dab75fe`);
   - `docs/datastore-integrity-verification/interrupt_when.py` (`4caa9b2`) and
     `interrupter_dummy_test.sh`;
   - `main.py` (`2c43521`).

## 2. Dispatch

One fresh-context subagent, on **Sonnet**. Give it:
- the prompt;
- the campaign README and this board;
- `HEAD`, the baselines and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/06a-give-the-build-a-rehearsal-scope.md`, in README
  §5.1's form, with the prompt's §7 additions.
- **The files it may add or change** are the prompt's §7 list, and **nothing else**. In particular:
  - in `quadsource_atol_sweep.py`, nothing outside §2's list: not `run_child`, not `sweep`, not
    `substitute_main_py`, not the module docstring;
  - in `test_prune_on_resume.py`, the one `Namespace` at most;
  - in `docs/datastore-integrity-verification/`, `interrupt_when.py` alone;
  - no other test, no `main.py`, no `RunRegistry/*.py`, nothing under `var/` and nothing under
    `orchestrator/`.
- **It launches nothing.** Verification 3 prints from the parser and stops.
- **Stop and ask** on any of the prompt's §5 conditions. A test that fails is a sign that the change
  went wider than §2, not a reason to edit the test.

## 3. The review — ten checks

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only §2's list.
   - `git diff HEAD~1 HEAD -- main.py Datastore/ config/ var/ 'RunRegistry/*.py' docs/gktk-remedial/`
     is empty.
   - Compared with the parent by `ast`, in `quadsource_atol_sweep.py`: only `run_build`,
     `build_main_argv`, `_main_py_argv` and `main` differ. The additions are the scope function and
     its constants. Every other top-level node dumps identically, the module docstring included.
   - Under `*/tests/`, the diff adds one file. If it changes `test_prune_on_resume.py`, the diff is
     the one `Namespace`, adding `rehearsal=False`.
   - In `docs/datastore-integrity-verification/`, only `interrupt_when.py` changes. By `ast`, only
     `names_the_run`, `run_slug` (or a new reader beside it), `watch` and the module docstring
     differ, and `DRIVER` may go.
2. **(E1) Production, by hand.** From the scratchpad, through the script's own parser:
   - `build_main_argv` for `--build` equals the parent's list, evaluated from `HEAD~1`'s `ast` on
     the same `Namespace`, element by element;
   - the k sample equals `k_sample()`;
   - the scope text equals the parent's f-string, evaluated with the same arguments and database.
3. **(E2) The rehearsal, by hand.** For `--build --rehearsal` and `--build --rehearsal --resume`:
   - the k sample is `[100000.0, 316227.7660168379, 1000000.0]`;
   - the argv is the production list with `"250"` in place of `"0.1"`, and nothing else differs;
   - the scope text names the grid and `zend 250`, and has no `9280`.

   Read that `run_build` takes `K_GRID_NAME` and `scope=` from the function's one result, and that
   it holds no `k_sample()` call.
4. **(E3) The refusals.** With `run_build`, `run_child`, `prepare`, `report` and `sweep` all stubbed
   on a loaded copy, `--rehearsal` with `--child`, `--prepare`, `--report` and no mode each exits 2,
   and no stub is called.
5. **(E4) The tests.** Run the new module, `test_prune_on_resume`, `test_killed_handler`,
   `test_store_fingerprint` and `test_quadsource_atol_sweep_prepare`. Read that the new module's
   expectations are literals, not values computed by the function under test.
6. **(E5) The pins.** In `bash`, replay each of (i)–(vi) from the log with `git apply`.
   - Run the named tests and see each fail for its own reason. (i) and (v) must fail
     `test_prune_on_resume`.
   - Revert with `git apply -R`, and check that `git status` is clean after each.
7. **(E6) The interrupter.** Run `interrupter_dummy_test.sh` at `HEAD` into a new directory in the
   scratchpad. Its five cases behave as its own record says
   (`docs/datastore-integrity-verification/interrupter_dummy_test.txt`). Then run B3's new case:
   it signals the `quadsource_atol_sweep.py` child and refuses the other. Neither leaves a `sleep`
   child of yours alive.
8. **`substitute_main_py`.** `substitute_main_py(filter_cases=False)` matches each of its four
   literals once. The log's verification 2 shows the same.
9. **The records.**
   - This board has the §1 row, B1–B3 and §5.14.
   - The log's observations record that `--resume` does not check the scope a store was built with
     (prompt §6).
   - The index is 102 by counting, plus any issue opened.
10. **Suites.** All six serially, at `HEAD`.
    - RunRegistry is 253 plus 06a's new tests.
    - Every other count is unchanged.
    - `black --check` is clean on every touched Python file.

## 4. After the review

- **Record it on this board**, after the prompt-06 review and in its form, with `HEAD` at dispatch,
  the checks and any remnants. Fix small residue in a follow-up commit of its own:
  - "this commit" → the SHA;
  - README §2's status cell for 06a ("landed …, `<sha>`");
  - the orchestrator notes list (`prompt-06a.md` "used"), and the board header.
- **Then complete [`prompt-07.md`](prompt-07.md) §0's "Carried from the review of 06a"**, in the same
  follow-up commit. It must carry:
  - the name of the scope function and what it returns;
  - the exact scope text a rehearsal manifest will hold, as the orchestrator printed it in check 3;
  - the line `run_build` prints for the rehearsal scope;
  - that `--resume` does not check a store's scope, so every launch of 07 passes `--rehearsal`;
  - that the interrupter now reads the driver from the manifest, and the output of B3's dummy case.
- **Report to the user:**
  - what `--build --rehearsal` now runs, against `--build`;
  - that production is unchanged, and how that was shown;
  - that 07 is released.
