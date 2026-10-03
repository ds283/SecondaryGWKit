# Orchestrator — prompt 04, a `SIGTERM` recorded as killed

Read [`../README.md`](../README.md) first: §0, §1, §4 and §5. Then read [`prompt-03.md`](prompt-03.md)
§0 and this board's *Orchestrator review of prompt 03*. Together they are the conventions this
note does not repeat.

**You do not write code.** You may:
- run the suites, the tests the log names and the probe;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process check from the session scratchpad. Never commit one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`04-record-a-sigterm-as-killed.md`](../04-record-a-sigterm-as-killed.md)
**Board items:** K1–K4 · **Closes:** `[10-a-sigterm-to-the-scoped-driver-is-recorded-as-failed-or-not-at-all]`,
on the `datastore-integrity` board's §4 · **Model:** Opus.
**Gate:** 00 landed and reviewed (`cfd9791`; review `e320338`). 01–03 have landed and been
reviewed too (reviews `1caecf8`, `f1999b1`, `74270dd`). 05 has not landed, and its change to
`run_build` is not in the tree. `HEAD` is the commit that adds this note, or a later one that
touches no code. One prompt at a time in this checkout.

## 0. What makes this prompt unusual

**It is the campaign's only prompt that runs real Ray, and it changes how two drivers die.** A
handler that is wrong here fails only when a run is interrupted, which no test exercises end to
end: in `ray.wait`, under Ray's own signal machinery. So the review rests on two things:
- **unit tests that call the handler and the post-`exec` function directly**, with a stand-in run
  and a stand-in shutdown;
- **a probe on real Ray** whose children go through the real wrapper shape, the real installer
  and the real post-`exec` function.

Both must be shown to catch the breakages before their passes count.

**The prompt was amended with this note, by the user's decisions of 2026-10-03.** Reading the code
at `74270dd` turned up two things the prompt could not pass, and three it had wrong. Each amended
line is dated 2026-10-03.
- **User decision: a pinned test is named.** `test_store_fingerprint.TestDrivers` pins the
  drivers' `run.finish(` calls (§3 item 5). That is the prompt's own fourth stop condition. The
  user chose to name the test under README §5 rule 7, with its two assertions kept.
- **User decision: the gate, and K4.** Another project's Ray cluster was running. A bare
  `ray.init` from either driver would resolve to that cluster instead of starting its own. The user
  chose to fix this in 04, as K4 (`address="local"`), and to let the probe run beside the other
  cluster.
- **Three amendments found by reading:**
  - the handler remembers the signal **first**;
  - the remembered signal decides **whatever the exception's type**;
  - the unregistered `exec` goes through the same function.

The review checks the agent against the amended prompt.

**The facts, checked by the orchestrator on 2026-10-03 at `74270dd`.** Pass them on; the agent
need not rediscover them. Each is marked as audit 00 marks its findings.

- **The drivers today** [read].
  - `scoped_pipeline_run.main`:
    - `register()` installs `terminal` for `SIGTERM` and `SIGINT`, only when `--register` is
      given, before `ray.init`. `terminal` writes `killed`, restores `SIG_DFL` and re-kills itself;
    - `_local_ray_init` is `return _real_ray_init(num_cpus=…, include_dashboard=False,
      ignore_reinit_error=True)`;
    - with `run is None` the `exec` has **no `try`**. With a run it has `except SystemExit` →
      `terminal_state(exc.code)`, `except BaseException` → `failed`, then `done`, each with
      `fingerprint=True`.
  - `quadsource_atol_sweep.run_build` has the same shape inline. It defines its own `terminal`
    inside `if args.register:` and calls `scoped.terminal_state`, with `scoped` coming from
    `_load_scoped_driver()`.
  - **`run_child` and `sweep` are left unchanged** (prompt K1).
    - `sweep` is the parent, and never calls `ray.init`, so its `terminal` is never replaced and
      works as written.
    - `run_child` is unregistered and has no handler. Its bare `ray.init` gets Ray's handler: a
      `SIGTERM` reaches it as `SystemExit(1)`, and the parent then raises its own `RuntimeError`.
      It also shares K4's defect.
    - The log records both, and neither is fixed.
- **`main.py` calls `ray.init` once** (`ray.init(address=args.ray_address)`, its line 313). The
  wrapper ignores the arguments. Because of `ignore_reinit_error=True`, a second call would reach
  the installer again. An installer that resets the remembered signal on a second call is a bug,
  even though `main.py` does not make one.
- **What `ray.init` installs on the driver** [read: `ray/_private/worker.py`
  `set_sigterm_handler`]. It installs `SIGTERM` only. `SIGINT` keeps Python's default, so a Ctrl-C
  in `ray.wait` today is a `KeyboardInterrupt`, which the drivers record `failed`. `DeferSigint`
  is used by workers, not the driver. Audit S1 measured `SIGTERM` only, so the probe's variants
  are `SIGTERM`. `SIGINT`'s 130 is pinned by the unit tests.
- **`Run.finish`** [read: `RunRegistry/__init__.py`].
  - It may be called twice, and the second call's fields overwrite the first's.
    `_update` reads `status.json`, updates it, and writes it through `write_json_atomic`
    (`status.json.tmp`, then `os.replace`).
  - With `fingerprint=True` it catches `Exception` from the fingerprint as `fingerprint_error`.
    It catches **`BaseException`** by writing `state=<its own argument>` and
    `fingerprint_error="interrupted: …"`, then re-raising.
  - So a signal during a not-signalled `finish(..., fingerprint=True)` puts the caller's state over
    the handler's `killed`. Amended K2 makes what happens then the *prompt's choice*, logged and
    pinned. `RunRegistry` is not changed.
  - Its docstring already says a signal handler passes no fingerprint, because the pool is open.
- **Two re-entrancy windows** [read; neither measured].
  - **The print.** Python signal handlers run on the main thread between bytecodes. A handler
    that writes to `sys.stderr` while the main thread is inside a write to the same buffered
    stream raises `RuntimeError: reentrant call inside <_io.BufferedWriter …>`. Hence amended K1:
    the signal is remembered first, and the print is guarded.
  - **`status.json.tmp`.** The stdout `RegisteredStream` heartbeats at most every 30 s through
    `_update`. If the handler's `finish` interrupts that `_update` inside its
    `with open(tmp, "w")`, both handles share the tmp's inode. The handler's write is renamed to
    `status.json`. The interrupted handle's buffered bytes can then be flushed into it on unwind,
    leaving JSON that does not parse.
    - The window is microseconds every 30 s.
    - K2's second `finish` rewrites the file from what it reads. A file that does not parse reads
      as `{}`, which loses `pid` and `stage`, but the file is valid again.
    - **Record it in the log's *Observations*, and open an issue if the agent agrees it is real.
      Do not fix it** (prompt §6).
- **What reaches the driver** [measured, audit S1].
  - A `SystemExit(128 + signum)` raised by the handler while the main thread is in `ray.wait` or
    `ray.get` arrives as `SystemExit(1)`.
  - Raised anywhere else, it arrives as itself.
  - The remembered signal is the only reliable record.
- **K4, measured by the orchestrator at 15:33**, from the scratchpad and not committed.
  - `/tmp/ray/ray_current_cluster` held `127.0.0.1:6379`, the GCS of a `ray start` cluster in the
    user's ChamPBH project (gcs_server pid 62525, since 2026-10-02 23:27).
  - `ray._private.services.canonicalize_bootstrap_address(None)` returned `127.0.0.1:6379`. With
    `"local"` it returned `None`, which means start a fresh instance.
  - `ray.init(num_cpus=1, include_dashboard=False, ignore_reinit_error=True)` raised
    `ValueError: When connecting to an existing cluster, num_cpus and num_gpus must not be
    provided.` It started no process.
  - The same call with `address="local"` started its own instance in 2.30 s, with its GCS at
    `127.0.0.1:65387` and 9 descendant processes. `ray.shutdown()` took 1.57 s, and 3 s later none
    remained.
  - ChamPBH's three processes were unchanged throughout.
- **The pinned test** [read: `RunRegistry/tests/test_store_fingerprint.py`].
  - `_finish_calls` walks the file's AST. It finds every call `run.finish(...)` whose receiver is
    the bare name `run`, and marks it *inside* when an enclosing function is named `terminal`.
  - `TestDrivers` expects (3 outside, 1 inside) for `scoped_pipeline_run.py` and (6, 2) for
    `quadsource_atol_sweep.py`. It asserts that every *outside* call passes `fingerprint=True` and
    no *inside* call does.
  - After K1–K2, `run_build`'s three calls move into `scoped_pipeline_run.py` behind a call through
    `scoped.`. The handler is no longer named `terminal`, and its no-fingerprint `run.finish` would
    count as *outside*.
  - Amended §3 item 5 allows exactly two changes: the handler's name, and the counts.
- **The other pins** [read].
  - `test_pipeline_adoption.TestTerminalState` pins `terminal_state(15) == "killed"` and the
    ordinary exits. They stay, unchanged (K3).
  - `TestTheDatastoreIsNotALedger.register` calls `driver.register(...)` and puts back the
    `SIGTERM`/`SIGINT` handlers. It passes whether or not `register()` still installs `terminal`.
  - `load_driver` imports `scoped_pipeline_run.py` by path. Its docstring says the module imports
    numpy and argparse at module scope and nothing else, with `ray` reached only inside `main()`.
    **The new functions must not import `ray` at module scope**, and the unit tests must not need
    it (README §5 rule 9). `_load_scoped_driver` imports the same module the same way.
  - `TestOptIn` runs the driver as a subprocess that exits before `import ray`.
- **The probe's raw material** [read: `docs/a3-v2-readiness/sigterm_probe.py`].
  - Its helpers can be imported by path, behind a `__main__` guard: `ray_procs`, `all_procs`,
    `descendants`, `exists`, `fmt` and `short` (which imports `ray`).
  - `run_variant` is not reusable: it spawns `[sys.executable, "-u", __file__, "--child", variant]`
    for that probe's own children, and keys on its `READY`/`SENDING` lines. The new probe writes
    its own runner, and **`sigterm_probe.py` is not edited**.
  - The children signal themselves from a `threading.Timer` with `os.kill(os.getpid(), SIGTERM)`.
  - **With ChamPBH's cluster up**, the before and after listings include its processes. Acceptance
    concerns the child's own descendants. The runner signals only those, re-checking each command
    line before it signals, as `sigterm_probe.py` does.
- **The store and the run for the probe** [read].
  - `Datastore/tests/real_store_fixtures.build_real_store` builds a small store in a directory.
  - `RunRegistry.begin(..., results=<it>, root=<a temporary directory>)` finds no sidecar, so
    `finish(fingerprint=True)` fingerprints with `write=False` and records
    `fingerprint_sidecar: "not written: …"`. That is expected: the fingerprint field is present.
- **The issue.** `[10-a-sigterm-…]` is in the `datastore-integrity` board's §3.3, with its
  *Assigned* line, and closes to that board's §4. The index is **102** at `74270dd`. After 04 it is
  **101**, plus one for each issue 04 opens.

**What the review exists to establish.**
- **(E1) The record.** A `SIGTERM` in `ray.wait` is recorded `killed`/143 within a second, then
  gains its fingerprint. The process exits 143 with no abort. This is shown on real Ray, 3 runs of
  each of 3 variants.
- **(E2) Nothing left behind.** No `gcs_server` or other own descendant survives any probe run.
- **(E3) Never `failed` once signalled.** That holds whatever exception reaches the driver, and on
  both the registered and the unregistered path.
- **(E4) The not-signalled path is unchanged.** The existing outcomes are pinned, and
  `test_pipeline_adoption` passes untouched.
- **(E5) K4.** Both drivers' real `ray.init` carry `address="local"`, and `run_child`'s is
  recorded, not changed.
- **(E6) The pins.** Breakages (i)–(vii) replay with `git apply`, and each fails what the log
  names. (i) and (iv) are probe results, and the log records what the probe showed.
- **(E7) The one changed test** changes only as §3 item 5 allows.
- **(E8) The records.** The issue is closed on the `datastore-integrity` board, naming the tests
  and the probe output. The index is 101.

**Carried from the reviews of 00–03.** All bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, never `git add -A`, `.` or `commit -a`.
  Show `git diff --cached --name-only` before committing. Count, then paste. A log states only
  what the diff, the tests and the probe show.
- **The SHA, not "this commit".** This stays the orchestrator's residue to fix.
- **Suites.** Use README §7's command, with output to a file and the verdict grepped from it.
  Under load, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`,
  `test_background_tau.test_build_cost` and `test_numeric_phase_cut.test_the_print_is_gated_both_ways`
  fail. Re-run a failure alone before calling it real. If it persists alone, settle it on an export
  of the tree before the prompt.
- **Breakage diffs** are recorded so that `git apply` replays them from a clean tree at the
  agent's commit. Replay them in `bash`, not `zsh`.
- **Scratch.** The agent works in its own subdirectory of the session scratchpad, never its top
  level. The probe's stores and run roots are temporary directories, never `var/`.
- **Ray processes.** The agent touches no Ray process it did not start. ChamPBH's cluster is the
  user's, and is left alone.
- **Usage limits.** Resume a stopped agent with `SendMessage`, saying exactly where it stopped. If
  it commits but sends no report, review from the commit and its log.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` are both empty, and `git worktree list` shows
     this checkout alone.
   - `git log --oneline 74270dd..HEAD` touches no code.
2. **The registry.** `./venv/bin/python -m RunRegistry list`: nothing `running`. At `74270dd` it
   held 12 runs: 10 finished and 2 `unknown` (`a3-pilot`, `a3-pilot-resume`).
3. **Ray.** Record `pgrep -lf 'gcs_server|raylet|ray::'` and `/tmp/ray/ray_current_cluster`.
   - Under the amended gate, no listed process may come from this repository's `venv`.
   - ChamPBH's processes, from its own venv, are expected. Note them and leave them alone.
4. **Baselines.** §5.7: AdaptiveLevin 32, ComputeTargets 686, CosmologyModels 39, Datastore 598,
   LiouvilleGreen 148 (1 skipped), RunRegistry 199. Re-measure all six on `HEAD`, serially, and
   record the load average.
5. **The index.** `grep -c '^| \`\[' docs/OPEN_ISSUES.md` gives **102**. After 04 it is 101. Tell
   the agent both numbers, and that it rises by one for each issue it opens.
6. **The pre-images**, for check 1. Record `git rev-parse HEAD:<path>` for
   `docs/gktk-remedial/scoped_pipeline_run.py`, `docs/handover/quadsource_atol_sweep.py`,
   `RunRegistry/tests/test_store_fingerprint.py`, `RunRegistry/tests/test_pipeline_adoption.py`
   and `docs/a3-v2-readiness/sigterm_probe.py`.

## 2. Dispatch

One fresh-context subagent, on **Opus**. Give it:
- the prompt, as amended;
- the campaign README, the audit (S1–S4 and the closing section's *04*), this board, and the
  `datastore-integrity` entry;
- `HEAD`, the baselines and both index counts;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/04-record-a-sigterm-as-killed.md`, in README §5.1's
  form, with the prompt's §7 additions. These are the Ray version and machine, the probe's wall
  time and load, and the event sequence before and after, as two numbered lists.
- **The files it may add or change:**
  - `docs/gktk-remedial/scoped_pipeline_run.py` and `docs/handover/quadsource_atol_sweep.py`.
    In the latter, only `run_build`;
  - new test modules under `RunRegistry/tests/`;
  - `RunRegistry/tests/test_store_fingerprint.py`, **only** `_finish_calls`' notion of a
    handler and `TestDrivers`' expected counts;
  - the probe under `docs/a3-v2-readiness/` and its committed `.txt`;
  - the log; this board; the `datastore-integrity` board (the issue to its §4);
    `docs/OPEN_ISSUES.md`.

  **Nothing else.** In particular:
  - no `RunRegistry/*.py`, no `main.py` and no pool;
  - no `sigterm_probe.py` and no other existing probe;
  - nothing under `var/`, and nothing under `orchestrator/`;
  - no `run_child` and no `sweep`;
  - no other existing test.
- **Stop and ask** on any of the prompt's §5 conditions.
  - Say that the not-signalled path's outcomes must not change. `TestTerminalState` and the rest of
    `test_pipeline_adoption` must pass untouched.
  - Say that ChamPBH's Ray cluster is running, is the user's, and must not be signalled.

## 3. The review — ten checks

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only §2's list.
   - `git diff HEAD~1 HEAD -- RunRegistry/*.py main.py Datastore/ config/ var/ docs/a3-v2-readiness/sigterm_probe.py` is empty.
   - In `quadsource_atol_sweep.py`, an `ast` comparison against the parent shows only `run_build`
     changed.
   - In `test_store_fingerprint.py`, only `_finish_calls` and `TestDrivers` changed.
2. **(E7) The changed test.**
   - Both assertions are still there: `all(outside)` and `not any(inside)`.
   - The handler's name is recognised, and the counts are updated.
   - The log gives the old (3, 1)/(6, 2) and the new counts, with the reason for each change.
   - Hand-check the new counts against `_finish_calls` run on both files.
3. **(E3) The handler and the post-`exec` function, by reading.**
   - The handler remembers first, writes `killed` with no fingerprint, guards its print and
     raises `SystemExit(128 + signum)`. On a later call it prints and returns.
   - The post-`exec` function keys on the remembered signal for **any** `BaseException`. Both
     drivers route both paths, registered and not, through it.
   - The installer imports nothing from `ray` at module scope, and a second install keeps the
     remembered signal.
4. **(E1, E4) The unit tests.**
   - Run the new modules and `test_pipeline_adoption`.
   - Read that the not-signalled cases cover every existing `terminal_state` outcome.
   - Read the case for a signal during the not-signalled fingerprint: the outcome is pinned and
     agrees with the log's *prompt's choice*.
5. **(E5) K4.**
   - Grep both drivers: one `address="local"` in each real `ray.init` call of `run_build` and
     `main`, none in `run_child`.
   - The `ast` test pins it, and breakage (vi) fails it.
6. **(E6) The pins.** In `bash`, replay each of the log's breakage diffs (i)–(vii) with
   `git apply`.
   - For a unit-test breakage, run the named test and see it fail.
   - For (i) and (iv), read the probe output the log gives for them. Whether (iv) left a
     `gcs_server` is the log's to say: S3 says it can, not always.
   - Revert with `git apply -R`, and check that `git status` is clean after each.
7. **(E1, E2) The probe output, by reading.** Nine runs: 3 variants × 3.
   - Each run shows `killed`/143 within 1 s of the signal, the fingerprint field present, exit
     status 143, no abort and no banner.
   - No own descendant is alive at +10 s, and the probe killed nothing.
   - The output ends with the processes before and after, and the pids killed.
   - In the second-`SIGTERM` variant, the second signal printed one line and changed nothing.
8. **(E1, E2) The probe, re-run once by the orchestrator.**
   - Record the ChamPBH processes before and after the run; they are unchanged.
   - The result matches the committed output's verdicts run for run, though times may differ.
   - A survivor or a `failed` is a finding, not noise.
9. **(E8) The records.**
   - The issue is in the `datastore-integrity` board's §4, with a closing note naming the tests and
     the probe output.
   - Its index row is gone, and the count is 101 by counting, matching the header.
   - The log's *Observations* carry `run_child`/`sweep`, the `status.json.tmp` window, and any issue
     opened.
   - *State handed to the next prompt* says that 07 and A3 v2 inherit K4.
10. **Suites.** All six serially, at `HEAD`.
    - RunRegistry is 199 plus 04's new tests. `TestDrivers` stays one test.
    - Every other count is unchanged.
    - `black --check` is clean on every touched Python file.

## 4. After the review

- **Record it on this board**, after the prompt-03 review and in its form, with `HEAD` at
  dispatch, the checks and any remnants. Fix small residue in a follow-up commit of its own,
  including "this commit" → the SHA.
- **Report to the user:**
  - what now happens after a `SIGTERM` or a Ctrl-C, in order, against what happened before;
  - what the probe showed, run by run, and that nothing of ChamPBH's was touched;
  - what K4 changes for a launch on a machine with another Ray cluster;
  - which issue closed, and what remains between the campaign and 07.
- **What 04 hands on.** 07's L1–L2 send a registered `--build` a `SIGTERM` and resume it. Its note
  must carry:
  - the handler's stderr line, which replaces Ray's banner, as the sign of an interruption;
  - that the record reads `killed`/143 and gains its fingerprint after the pool closes;
  - that the drivers now start their own Ray instance (`address="local"`), so the rehearsal can run
    beside another cluster;
  - the not-signalled-fingerprint choice, if it bears on how the rehearsal ends a run.
