# Log — rrk-v020-adoption prompt 02, adopt the hard-kill route

- **Subject:** [`02-adopt-the-hard-kill-route.md`](../02-adopt-the-hard-kill-route.md), board items A1, A2 and A3.
- **Commit:** this commit (parent `c435666`).
- **Result:** landed. Acceptance 1–5 pass; no stop condition of the prompt's §4 fired.
- **Model:** Sonnet 5.5.
- **Date:** 2026-10-18.

The orchestrator's brief for this prompt (its note's §0) was read first and is cited as "the note".

## What shipped

- **`CLAUDE.md`**, "Long-running jobs — the run registry": rule 2 gains host and boot; a new rule 5, the route after a hard
  kill (Durability and Provenance become 6 and 7); the pointer sentence gains "Ending a run whose end was never recorded";
  and `end` joins the command list in the layer section (deviation 1). Quoted below.
- **`docs/datastore-integrity-verification/interrupt_when.py`**: `--signal {TERM,KILL}`, default `TERM`. `watch` takes
  `sig=signal.SIGTERM`, sends it where it sent `SIGTERM`, and builds the log line's name from it. The docstring's first line,
  usage and one new paragraph name the option. The `watching` line is unchanged.
- **`docs/a3-v2-readiness/interrupter_build_dummy_test.sh`**: a case D after case C, and the header's case list gains D.
- **`RunDrivers/tests/test_hard_killed_predecessor.py`** (new): four tests (deviation 2).
- **`measure/out/`**: `02-roots.txt`, `02-dummy-build.txt`, `02-dummy-scoped.txt`.
- This log, and the board: header, the §1 row for 02, items A1–A3, `### 5.2 After prompt 02`.

### `CLAUDE.md`, as changed

- Layer section: "(`list`, `end`, and `store show | create | fingerprint | copy | move | retire | amend`)."
- Pointer: "RunRegistryKit's `README.md`, "The rules", tells each rule with the failure it answers, and its section "Ending a run
  whose end was never recorded" tells rule 5 below."
- Rule 2, added after the existing text: "The test is made **on the host and boot the run recorded**: a pid names a process only
  there. A `running` run recorded on another host is `foreign`, and is not tested here. A `running` run from an earlier boot of
  this host is `stale` whatever its pid answers. A run with no host recorded, which is every run written before `v0.2.0`, is
  judged as before."
- Rule 5 (new): "**The route after a hard kill.** A run killed by `SIGKILL`, a power loss, a reboot or the OOM killer never records
  its end, and its status says `running` for ever. `list` shows it `stale`, `!!`, and a resume's `begin` on its store is refused,
  with nothing created: the registry does not decide that a run is dead. A person checks that it is over, then runs
  `./venv/bin/python -m runregistrykit end RUN_ID --reason TEXT` **on the host it ran on**. `end` refuses a pid that answers
  `kill -0`, with no override; `--unverified` is for a run on another host, or one whose status does not say, once checked
  there. The run is then `lost`, and the resume is begun. **`Run.finish` is never called by hand on a run that is not the
  caller's**: it checks nothing. The one case `end` refuses that is not a live run is a killed process its parent has not yet
  reaped, which still answers `kill -0` (RunRegistryKit's `[04-a-killed-process-not-yet-reaped-answers-kill-0]`); a detached
  launch through `nohup sh -c` is reaped by its `sh`."

Checked against RunRegistryKit's README at `v0.2.0` ("A run", "Ending a run whose end was never recorded"): `foreign`, `stale`
by boot, the no-host rule, `end`'s refusals, `--unverified`, `lost` and the reaped-parent caveat are as it says. The source
for rule 5 is that section; "The rules" has no hard-kill rule, and the pointer says so.

## Deviations from the prompt

1. **`end` is added to the command list in the layer section** (`CLAUDE.md:74-75`), which the prompt does not name.
   **IMPLEMENTATION CHOICE**, sanctioned by the note under README §6.2 C4 (the pin made the list incomplete).
2. **The test module has four tests, not three.** Beyond the prompt's three it adds
   `test_a_second_end_on_the_lost_run_is_refused` (the note's fact 5, which 03 needs), and test 3 also tries `end --unverified`
   on the live run and asserts the status is unchanged. Test 1's restoration observation is taken inside the register helper,
   before its `finally`, as the note requires; test 2's "every other key is kept" excludes `state`, `exit_code`, `ended` and
   `heartbeat`. **IMPLEMENTATION CHOICE** (the note fixes both points; the extra test and `--unverified` check are additions).
3. **The dummy tests' header lines** give the command and `HEAD` `c435666` with "prompt 02's changes uncommitted", since the
   commit did not yet exist. **IMPLEMENTATION CHOICE.**
4. **`end` is run with `sys.executable`**, not `./venv/bin/python`, in the test: the same interpreter, robust in a worktree.
   **IMPLEMENTATION CHOICE.**
5. **No other deviation.**

## Verification performed

- **The test module**, alone: 4 tests OK. Timings (each includes a child's start): `test_register_meets_the_refusal_and_takes_nothing_over`
  0.31 s; `test_end_is_refused_on_a_run_whose_pid_answers` 0.50 s; `test_a_second_end_on_the_lost_run_is_refused` 0.50 s;
  `test_end_records_the_run_lost_and_the_second_registration_is_accepted` 0.56 s.
- **The seven roots**, serially, foreground (`measure/out/02-roots.txt`, with the load average before each):

  | Root | Expected | Result |
  |---|---|---|
  | `AdaptiveLevin/tests` | 32 | 32 OK |
  | `CosmologyModels/tests` | 39 | 39 OK |
  | `ObjectFactories/tests` | 279 | 279 OK |
  | `tools/tests` | 91 | 91 OK |
  | `LiouvilleGreen/tests` | 148 | 148 OK (1 skipped) |
  | `RunDrivers/tests` | 99 + 4 | 103 OK |
  | `ComputeTargets/tests` | 700 | 700 OK (140 s, load about 21) |

  No known flake fired; none was re-run.
- **Both dummy tests**, whole (`measure/out/02-dummy-build.txt`, `02-dummy-scoped.txt`). Compared with `runregistry-extraction` 08's
  recorded outputs on the `===`, `exit=` and "still alive" lines, the only differences are the pids. Build: A `interrupter exit=0`,
  `child exit=143`; B `exit=2`, child alive; C `exit=2`, child alive; **D `interrupter exit=0`, `child exit=137`, the log
  "sent SIGKILL to pid …"**. Scoped, unchanged: A 0/143; B child 0, interrupter 0; C 2, child alive; D 0/143; E child 0,
  interrupter 0. The default's log lines read as before: "sent SIGTERM to pid … at stage … (45% >= 40%); command …; exiting".
- **`black --check`** is clean on the new module and on `interrupt_when.py`.
- **`ps`** (`hard-killed`, `sleep 120`, `sleep 30`, `sleep(600`) after all runs, mutations included: nothing found.
- **`var/`:** `find var -mmin -60` is empty; `list` reads 45 runs, nothing running or stale, as before.

## The deliberate-breakage record

Each applied to the working tree, run, and reverted; `git diff` shows none left. Mutations 1–3 are in
`docs/gktk-remedial/scoped_pipeline_run.py`, run with `unittest RunDrivers.tests.test_hard_killed_predecessor`; 4 and 5 are in
`interrupt_when.py`, run on the build dummy test.

1. **`results=None`** in `register`'s `begin`:
   ```diff
diff --git a/docs/gktk-remedial/scoped_pipeline_run.py b/docs/gktk-remedial/scoped_pipeline_run.py
index 0343b61..c79d3ae 100644
--- a/docs/gktk-remedial/scoped_pipeline_run.py
+++ b/docs/gktk-remedial/scoped_pipeline_run.py
@@ -552,7 +552,7 @@ def register(args, main_args, k_sample, database, runs_root=None):
         # *not* `checkpoint`: that field is the unit ledger record()/known() own, this run has no
         # ledger and records no units, and a run that named its datastore there would be one
         # record() call away from appending JSON-Lines to a SQLite file.
-        results=str(database) if database is not None else None,
+        results=None,
         scope=scope,
         heartbeat_means=HEARTBEAT_MEANS.format(interval=BEAT_MIN_INTERVAL),
         root=runs_root,
   ```
   Two failures: `TestTheRefusal.test_register_meets_the_refusal_and_takes_nothing_over`, at
   `self.assertIsInstance(outcome, RuntimeError, "begin was not refused")` ("<Run object> is not an instance of RuntimeError"),
   and `TestEndThenTheResume.test_end_records_the_run_lost_and_the_second_registration_is_accepted`, at its first
   `assertIsInstance(outcome, RuntimeError)`.
2. **`begin` wrapped in `try … except RuntimeError: return None`** (the whole call indented under the `try`; diff in full):
   ```diff
diff --git a/docs/gktk-remedial/scoped_pipeline_run.py b/docs/gktk-remedial/scoped_pipeline_run.py
index 0343b61..9a4f721 100644
--- a/docs/gktk-remedial/scoped_pipeline_run.py
+++ b/docs/gktk-remedial/scoped_pipeline_run.py
@@ -541,22 +541,25 @@ def register(args, main_args, k_sample, database, runs_root=None):
         f"{len(k_sample)} log-spaced k over {args.k_min:g}-{args.k_max:g} /Mpc, {models}, "
         f"{args.cpus} cpus; main.py args: {' '.join(main_args)}"
     )
-    run = runregistrykit.begin(
-        campaign=args.campaign,
-        prompt=args.prompt,
-        slug=args.register,
-        purpose=args.purpose,
-        script=__file__,
-        # README §0.2: this job's results live in its datastore, and `results` is the field that
-        # names the durable thing they live in -- which is what a resume needs. It is deliberately
-        # *not* `checkpoint`: that field is the unit ledger record()/known() own, this run has no
-        # ledger and records no units, and a run that named its datastore there would be one
-        # record() call away from appending JSON-Lines to a SQLite file.
-        results=str(database) if database is not None else None,
-        scope=scope,
-        heartbeat_means=HEARTBEAT_MEANS.format(interval=BEAT_MIN_INTERVAL),
-        root=runs_root,
-    )
+    try:
+        run = runregistrykit.begin(
+            campaign=args.campaign,
+            prompt=args.prompt,
+            slug=args.register,
+            purpose=args.purpose,
+            script=__file__,
+            # README §0.2: this job's results live in its datastore, and `results` is the field that
+            # names the durable thing they live in -- which is what a resume needs. It is deliberately
+            # *not* `checkpoint`: that field is the unit ledger record()/known() own, this run has no
+            # ledger and records no units, and a run that named its datastore there would be one
+            # record() call away from appending JSON-Lines to a SQLite file.
+            results=str(database) if database is not None else None,
+            scope=scope,
+            heartbeat_means=HEARTBEAT_MEANS.format(interval=BEAT_MIN_INTERVAL),
+            root=runs_root,
+        )
+    except RuntimeError:
+        return None
 
     def terminal(signum, _frame):
         """Record the death, then die exactly as the default handler would have."""
   ```
   The same two tests fail at the same assertions ("None is not an instance of RuntimeError").
3. **`register` takes over `sys.stdout` before `begin`**:
   ```diff
diff --git a/docs/gktk-remedial/scoped_pipeline_run.py b/docs/gktk-remedial/scoped_pipeline_run.py
index 0343b61..1458386 100644
--- a/docs/gktk-remedial/scoped_pipeline_run.py
+++ b/docs/gktk-remedial/scoped_pipeline_run.py
@@ -541,6 +541,7 @@ def register(args, main_args, k_sample, database, runs_root=None):
         f"{len(k_sample)} log-spaced k over {args.k_min:g}-{args.k_max:g} /Mpc, {models}, "
         f"{args.cpus} cpus; main.py args: {' '.join(main_args)}"
     )
+    sys.stdout = RegisteredStream(sys.stdout, None, StageTracker())
     run = runregistrykit.begin(
         campaign=args.campaign,
         prompt=args.prompt,
   ```
   One failure: `test_register_meets_the_refusal_and_takes_nothing_over`, at its `assertEqual(taken, {"stdout": False, …})`:
   `{'stdout': True, 'stderr': False, 'handlers': False} != {'stdout': False, …}`. The observation is taken inside the helper,
   so the `finally` does not hide it.
4. **The interrupter sends `signal.SIGTERM` whatever `--signal` says** (`diff` of `interrupt_when.py`):
   ```diff
217c217
<                 os.kill(int(pid), sig)
---
>                 os.kill(int(pid), signal.SIGTERM)
   ```
   Build dummy test, case D: `interrupter exit=0`, **`child exit=143`** (not 137), and the log still says "sent SIGKILL", the
   disagreement being what the case catches. Cases A–C unchanged.
5. **The interrupter's `--signal` default becomes `KILL`**:
   ```diff
246c246
<         default="TERM",
---
>         default="KILL",
   ```
   Build dummy test, case A: **`child exit=137`** (recorded: 143), its log "sent SIGKILL to pid …" (bash also prints
   "Killed: 9" for the job).

## Observations not acted on

- `register` takes no `client_root`, so `begin` finds `runregistrykit.toml` by walking up from `os.getcwd()`. The new test (as
  `test_pipeline_adoption`'s) therefore needs the suite run from the repository root, as `CLAUDE.md` already says. No issue.
- The interrupter's argparse description is the docstring's first line, so `--help` now reads longer. Harmless.
- A `KILL`ed build leaves its `interrupter.log` as the only record of why; 03 should read it before `end`.

## State handed to the next prompt

- `CLAUDE.md` states the route after a hard kill (rule 5) and liveness by host and boot (rule 2); `end` is in the command list.
- `interrupt_when.py --signal KILL` sends `SIGKILL` and logs it; the default is unchanged. The build dummy test's case D shows
  child exit 137.
- `RunDrivers/tests/test_hard_killed_predecessor.py` pins the refusal at `register`, `end`'s acceptance, its refusal on a live
  run and on a `lost` one, and the second registration. `RunDrivers` is 103; every other root at its count.
- The registry is 45 runs, nothing running or stale; `docs/OPEN_ISSUES.md` stays at 100. Neither driver, `main.py`, any reader nor
  any existing test changed.
