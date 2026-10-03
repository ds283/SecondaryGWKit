# Log — prompt 06a: give `--build` a rehearsal scope

**Subject:** *Give `--build` a rehearsal scope* · **Model:** Claude Sonnet 5.5 · **Date:**
2026-10-03 · **Base:** `3cdb52a`, clean · **Result:** landed. `quadsource_atol_sweep.py --build`
takes `--rehearsal`, which selects `datastore-integrity` prompt 10's scope (three log-spaced k over
1e5-1e6 /Mpc, `--zend 250`) in place of the production grid and `--zend 0.1`; without the flag the
k sample, the argv list and the manifest's `scope` text are what they were. `interrupt_when.py`
takes the driver's name from the run's manifest `argv[0]`, so it can interrupt a `--build`. None of
the prompt's four §5 stop conditions fired. Nothing was launched, no Ray was started, nothing was
written under `var/`. Nothing was closed or opened: `docs/OPEN_ISSUES.md` is unchanged (102). The
commit is `7bd202f`.

## What shipped

- **`docs/handover/quadsource_atol_sweep.py`.**
  - New constants: `PRODUCTION_ZEND = "0.1"`; `REHEARSAL_K_MIN, REHEARSAL_K_MAX, REHEARSAL_K_COUNT
    = 1e5, 1e6, 3`; `REHEARSAL_ZEND = "250"`; `PRODUCTION_SCOPE_TEXT` and `REHEARSAL_SCOPE_TEXT`,
    the two manifest `scope` texts as `str.format` templates over `atol`, `rtol`, `cpus` and
    `database` (the store's file name).
  - **`build_scope(args)`**, the one function that decides the scope. It reads `args.rehearsal` and
    returns `(k sample, zend, scope text)`. Without the flag: `k_sample()`, `"0.1"`,
    `PRODUCTION_SCOPE_TEXT`. With it: `np.logspace(np.log10(1e5), np.log10(1e6), 3).tolist()`, which
    is `[100000.0, 316227.7660168379, 1000000.0]`, `"250"`, `REHEARSAL_SCOPE_TEXT`. It imports
    nothing.
  - **`run_build`** unpacks `k_values, zend, scope_text = build_scope(args)` first, gives
    `K_GRID_NAME` `k_values`, and writes `scope=scope_text.format(atol=..., rtol=..., cpus=...,
    database=database.name)`. It prints one line when the flag is given (below). Its
    `sys.argv = build_main_argv(args, database)` line is untouched. Its docstring says what
    `--rehearsal` does.
  - **`build_main_argv`** takes `--zend` from `build_scope(args)`. `_main_py_argv` gained
    `zend=PRODUCTION_ZEND`; `child_main_argv` is unchanged and still produces `0.1`.
  - **`main()`** gained `--rehearsal` (`store_true`; the help says it is a rehearsal scope, not A3
    v2, and valid only with `--build`) and, straight after `parse_args()` and before any mode
    runs, `parser.error(...)` if `args.rehearsal and not args.build`.
  - **The line `run_build` prints** for the rehearsal, after the `main.py argv` line:
    `** quadsource_atol_sweep --build: rehearsal scope, NOT A3 v2: k = [100000.0, 316227.7660168379, 1000000.0], zend 250`
  - **By `ast` against `3cdb52a`** (script in the scratch directory, not committed): the top-level
    definitions and assignments that changed are `_main_py_argv`, `build_main_argv`, `main` and
    `run_build`; added are the six names above and `build_scope`; none was removed; the module
    docstring, `run_child`, `sweep`, `substitute_main_py`, `k_sample`, `make_case_selector`,
    `child_main_argv`, `prepare`, `report`, `assert_store_is_self_consistent`, `shard_paths` and every
    existing constant are identical.
- **`RunRegistry/tests/test_build_rehearsal_scope.py`** (new, 28 tests, no Ray), below.
- **`RunRegistry/tests/test_prune_on_resume.py`**: **the named `Namespace` changed.** In
  `test_the_job_name_is_the_namespaces`, `argparse.Namespace(job_name="some-other-label",
  resume=False)` became `argparse.Namespace(job_name="some-other-label", resume=False,
  rehearsal=False)` (black wraps it over three lines). Without the change `build_main_argv` raises
  `AttributeError` on it, as the prompt foresaw. Nothing else in the file changed; all 18 tests pass.
- **`docs/datastore-integrity-verification/interrupt_when.py`.** The constant `DRIVER` is gone.
  `run_driver(run_dir)` returns the basename of the manifest's `argv[0]` (`None` if `argv` is empty,
  its first element is not a string, or its basename is empty), read beside the slug.
  `names_the_run(command, slug, driver)` requires `driver in command` and the tokens `--register
  <slug>`, as before. `watch` logs `the run's manifest names no argv[0]; exiting without
  signalling` and returns 2 when there is none, as it does for a missing slug; the refusal's message
  names the driver that was required. The docstring says so. Nothing else in the file changed; the
  header log line is as it was.
- **`docs/a3-v2-readiness/interrupter_build_dummy_test.sh`** and **`.txt`** (new): three cases,
  below. `interrupter_dummy_test.sh` and its output are untouched.
- **This log and the board** (§1 row, B1–B3, a new §5.14).
- Not touched: `main.py`, every `RunRegistry/*.py`, every other test, every other file of
  `docs/datastore-integrity-verification/`, `docs/OPEN_ISSUES.md`, `var/`, and
  `prompts/a3-v2-readiness/orchestrator/`.

### The tests (`RunRegistry/tests/test_build_rehearsal_scope.py`, 28)

The parser is the script's own: `parsed_arguments` is imported from `test_prune_on_resume`, not
copied. Every expectation is a literal, except the prompt's own "the k sample equals `k_sample()`".
The scope text is filled with `.format(atol, rtol, cpus, database)` from the parsed `Namespace`, as
`run_build` fills it, and compared with a literal.

- **`TestProduction`** (6): the flag defaults off; the k sample equals `k_sample()`, and (so that
  `k_sample()` cannot drift unseen) has 8 points from `1e5` to `3e8`; the argv list is the
  production list, with and without `--resume`; the scope text is the 9280-item string of `3cdb52a`
  as a literal.
- **`TestRehearsal`** (8): the flag is on; the k sample is the literal `[1e5, 316227.7660168379,
  1e6]`, also with `--resume`; the argv list is the literal with `"250"` after `--zend`, with and
  without `--resume`; the text names `REHEARSAL build, not A3 v2`, `3 log-spaced k over
  100000-1e+06 /Mpc` and `zend 250`, and lacks `9280`, `8 log-spaced` and `3e+08`; the two argv lists
  differ in one place, the token after `--zend`.
- **`TestRefusals`** (7): `main()` on a fresh copy of the module with `run_build`, `run_child`,
  `prepare`, `report` and `sweep` all replaced by recorders. `--rehearsal` with `--child`,
  `--prepare`, `--report`, no mode, and no mode with `--register`, each exits 2 with a message
  naming both options and reaches no mode. Two controls keep the instrument honest: the same stubs
  reach `run_build` for `--build --rehearsal`, and reach each of `prepare`, `report`, `run_child`
  and `sweep` without the flag.
- **`TestTheDriversReadTheScope`** (7), with `ast`: `run_build` unpacks `build_scope(args)`; the
  value its namespace dict gives `K_GRID_NAME` is the unpacked k name; `run_build` holds no call of
  `k_sample`; the `scope=` of its `RunRegistry.begin` call is `<unpacked text name>.format(...)`; its
  one `sys.argv` assignment is still `build_main_argv(args, database)`; `run_child`'s value for
  `K_GRID_NAME` is `k_sample()` and it neither names `build_scope` nor mentions `rehearsal`;
  `build_scope` imports nothing.

## Deviations from the prompt

1. **The scope function returns the scope text as a `str.format` template, which `run_build`
   fills.** The prompt has the function return "the text for the manifest's `scope`", and
   `build_main_argv` read it too. But `test_the_job_name_is_the_namespaces` hands `build_main_argv`
   a `Namespace` that may gain `rehearsal=False` and nothing else, and today's text needs `atol`,
   `rtol`, `cpus` and the store's name. A function that built the finished text from the `Namespace`
   would raise `AttributeError` there on `atol`. So the function reads `args.rehearsal` only, and
   returns the text with those four fields open. One function still decides all three, and the test
   compares the filled text with today's string as a literal. **STRUCTURALLY REQUIRED.**
2. **`build_scope(args)`, not `build_scope(args, database)`**, for the same reason. **IMPLEMENTATION
   CHOICE** (prompt: the name is the prompt's choice).
3. **`_main_py_argv` takes `zend` with a default of `PRODUCTION_ZEND`**, so `child_main_argv` is not
   touched, which the prompt prefers. **IMPLEMENTATION CHOICE.**
4. **`DRIVER` is removed from `interrupt_when.py`**, where the prompt said "the driver's name is the
   basename of the manifest's `argv[0]`". The constant has no remaining use, and `names_the_run`
   gained a parameter. **STRUCTURALLY REQUIRED.**
5. **A third dummy case**, an empty `argv[0]`, beside the two the prompt names, to exercise the
   "no `argv[0]`" refusal; and 28 tests, beyond the prompt's list (the two scopes' `--resume`
   variants, the controls on the stubs, the `scope=` and `sys.argv` reads). **IMPLEMENTATION CHOICE.**

No `UNINTENDED DRIFT` was found.

## Verification performed

### 1. Every existing test passes

`test_prune_on_resume` (18), `test_killed_handler`, `test_store_fingerprint` and
`test_quadsource_atol_sweep_prepare` pass, with the one `Namespace` changed and nothing else. The
suites table below is the whole of it.

### 2. `substitute_main_py(filter_cases=False)`

Each of its four literals is replaced exactly once. Its output, verbatim:

```
** quadsource_atol_sweep: replaced 1 occurrence of 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_SOURCE_K_VALUES)' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_RESPONSE_K_VALUES)' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'tol=DEFAULT_QUADRATURE_ATOL' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'tol=DEFAULT_QUADRATURE_RTOL' in main.py
```

### 3. The two scopes, from the parser

For `--build`, `--build --rehearsal` and `--build --rehearsal --resume`, with `--database`,
`--atol 1e-32` and `--rtol 1e-7`, parsed by the script's own parser, nothing launched. The script
is in the scratch directory; the text is filled with `cpus` 6 (the parser's default) and the
store's file name. Verbatim:

```
$ quadsource_atol_sweep.py --build --database /private/tmp/a3-v2-readiness-scope-check/store.sqlite --atol 1e-32 --rtol 1e-7
  k sample : [100000.0, 313856.84721559234, 985061.2054411147, 3091682.0425413917, 9703455.784651041, 30454960.396664713, 95584978.52172911, 300000000.0000001]
  argv     : ['main.py', '--database', '/private/tmp/a3-v2-readiness-scope-check/store.sqlite', '--job-name', 'handover-A3-baseline-v2', '--shards', '4', '--zend', '0.1', '--source-samples-log10z', '100']
  scope    : full build: main.py's own 9280-item QuadSourceIntegral work list at atol=1e-32 rtol=1e-07, 6 cpus, LambdaCDM, 8 log-spaced k over 100000-3e+08 /Mpc, into store.sqlite

$ quadsource_atol_sweep.py --build --rehearsal --database /private/tmp/a3-v2-readiness-scope-check/store.sqlite --atol 1e-32 --rtol 1e-7
  k sample : [100000.0, 316227.7660168379, 1000000.0]
  argv     : ['main.py', '--database', '/private/tmp/a3-v2-readiness-scope-check/store.sqlite', '--job-name', 'handover-A3-baseline-v2', '--shards', '4', '--zend', '250', '--source-samples-log10z', '100']
  scope    : REHEARSAL build, not A3 v2: main.py's own unfiltered QuadSourceIntegral work list at atol=1e-32 rtol=1e-07, 6 cpus, LambdaCDM, 3 log-spaced k over 100000-1e+06 /Mpc, zend 250, into store.sqlite

$ quadsource_atol_sweep.py --build --rehearsal --resume --database /private/tmp/a3-v2-readiness-scope-check/store.sqlite --atol 1e-32 --rtol 1e-7
  k sample : [100000.0, 316227.7660168379, 1000000.0]
  argv     : ['main.py', '--database', '/private/tmp/a3-v2-readiness-scope-check/store.sqlite', '--job-name', 'handover-A3-baseline-v2', '--shards', '4', '--zend', '250', '--source-samples-log10z', '100']
  scope    : REHEARSAL build, not A3 v2: main.py's own unfiltered QuadSourceIntegral work list at atol=1e-32 rtol=1e-07, 6 cpus, LambdaCDM, 3 log-spaced k over 100000-1e+06 /Mpc, zend 250, into store.sqlite
```

### 4. The interrupter

`interrupter_dummy_test.sh`, unchanged, run into a new directory in the session scratchpad. The
work directory is replaced by `<work dir>`, as in the committed `interrupter_dummy_test.txt`; every
case behaves as in that file (A and D signal, B and E exit without signalling, C refuses). Verbatim:

```
=== case A: signals when the stage matches
interrupter exit=0
child exit=143 (143 = killed by SIGTERM)
--- <work dir>/case-a/interrupter.log
2026-10-03T23:01:59+01:00 interrupter pid 98003 watching <work dir>/case-a (slug 'dummy-a'); stage regex '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, ' at >= 40.0%; every 1 s; stale after 900 s; hold 0 s
2026-10-03T23:01:59+01:00 liveness stale; pid 98001; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, 45.00% (11 of 20 work items remaining)'
2026-10-03T23:02:03+01:00 liveness alive; pid 98001; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, 10.00% (18 of 20 work items remaining)'
2026-10-03T23:02:06+01:00 liveness alive; pid 98001; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, 45.00% (11 of 20 work items remaining)'
2026-10-03T23:02:06+01:00 sent SIGTERM to pid 98001 at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, 45.00% (11 of 20 work items remaining)' (45% >= 40%); command 'python docs/gktk-remedial/scoped_pipeline_run.py --register dummy-a 120'; exiting
=== case B: exits silently when the child dies first
child exit=0 (0 = sleep ended by itself)
interrupter exit=0
--- <work dir>/case-b/interrupter.log
2026-10-03T23:02:06+01:00 interrupter pid 98099 watching <work dir>/case-b (slug 'dummy-b'); stage regex '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, ' at >= 40.0%; every 1 s; stale after 900 s; hold 0 s
2026-10-03T23:02:06+01:00 liveness alive; pid 98097; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, 10.00% (18 of 20 work items remaining)'
2026-10-03T23:02:10+01:00 run's pid 98097 no longer answers kill -0; exiting without signalling
=== case C: refuses a pid whose command line does not name the run
interrupter exit=2
child 98416 still alive (not signalled)
--- <work dir>/case-c/interrupter.log
2026-10-03T23:02:10+01:00 interrupter pid 98418 watching <work dir>/case-c (slug 'dummy-c'); stage regex '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, ' at >= 40.0%; every 1 s; stale after 900 s; hold 0 s
2026-10-03T23:02:10+01:00 liveness alive; pid 98416; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, 45.00% (11 of 20 work items remaining)'
2026-10-03T23:02:10+01:00 pid 98416's command line does not name scoped_pipeline_run.py (the run's own driver, its manifest's argv[0]) and --register dummy-c: 'sleep 30'; exiting without signalling
=== case D: --hold 4 at a bare banner (0%): holds, reads again, then signals
interrupter exit=0
child exit=143 (143 = killed by SIGTERM)
--- <work dir>/case-d/interrupter.log
2026-10-03T23:02:10+01:00 interrupter pid 98424 watching <work dir>/case-d (slug 'dummy-d'); stage regex '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' at >= 0.0%; every 1 s; stale after 900 s; hold 4 s
2026-10-03T23:02:10+01:00 liveness alive; pid 98422; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'
2026-10-03T23:02:10+01:00 eligible at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'; holding 4 s, then reading again
2026-10-03T23:02:14+01:00 sent SIGTERM to pid 98422 at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT' (0% >= 0%); command 'python docs/gktk-remedial/scoped_pipeline_run.py --register dummy-d 120'; exiting
=== case E: --hold 4, but the stage completes during the hold: no signal; exits when the child dies
child exit=0 (0 = sleep ended by itself)
interrupter exit=0
--- <work dir>/case-e/interrupter.log
2026-10-03T23:02:14+01:00 interrupter pid 98481 watching <work dir>/case-e (slug 'dummy-e'); stage regex '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT' at >= 0.0%; every 1 s; stale after 900 s; hold 4 s
2026-10-03T23:02:14+01:00 liveness alive; pid 98479; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'
2026-10-03T23:02:14+01:00 eligible at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'; holding 4 s, then reading again
2026-10-03T23:02:18+01:00 liveness alive; pid 98479; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, all work items complete'
2026-10-03T23:02:23+01:00 run's pid 98479 no longer answers kill -0; exiting without signalling
```

B3's new dummy case, `docs/a3-v2-readiness/interrupter_build_dummy_test.sh`, run the same way.
Case A: a manifest whose `argv[0]` is `docs/handover/quadsource_atol_sweep.py` and a child named for
it: signalled. Case B: the same manifest and a child named for `scoped_pipeline_run.py`: refused,
the child still alive. Case C: an empty `argv[0]`: exits without signalling. Verbatim (also
committed as `interrupter_build_dummy_test.txt`):

```
=== case A: a --build manifest (argv[0] the sweep script), a child named for it: signals
interrupter exit=0
child exit=143 (143 = killed by SIGTERM)
--- <work dir>/case-a/interrupter.log
2026-10-03T23:02:41+01:00 interrupter pid 98926 watching <work dir>/case-a (slug 'dummy-build-a'); stage regex '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, ' at >= 40.0%; every 1 s; stale after 900 s; hold 0 s
2026-10-03T23:02:41+01:00 liveness alive; pid 98924; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, 45.00% (11 of 20 work items remaining)'
2026-10-03T23:02:41+01:00 sent SIGTERM to pid 98924 at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, 45.00% (11 of 20 work items remaining)' (45% >= 40%); command 'python docs/handover/quadsource_atol_sweep.py --build --register dummy-build-a 120'; exiting
=== case B: the same manifest, a child named for scoped_pipeline_run.py: refuses
interrupter exit=2
child 98930 still alive (not signalled)
--- <work dir>/case-b/interrupter.log
2026-10-03T23:02:41+01:00 interrupter pid 98932 watching <work dir>/case-b (slug 'dummy-build-b'); stage regex '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, ' at >= 40.0%; every 1 s; stale after 900 s; hold 0 s
2026-10-03T23:02:41+01:00 liveness alive; pid 98930; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, 45.00% (11 of 20 work items remaining)'
2026-10-03T23:02:41+01:00 pid 98930's command line does not name quadsource_atol_sweep.py (the run's own driver, its manifest's argv[0]) and --register dummy-build-b: 'python docs/gktk-remedial/scoped_pipeline_run.py --build --register dummy-build-b 30'; exiting without signalling
=== case C: a manifest whose argv[0] is empty: exits without signalling
interrupter exit=2
child 98937 still alive (not signalled)
--- <work dir>/case-c/interrupter.log
2026-10-03T23:02:41+01:00 interrupter pid 98939 watching <work dir>/case-c (slug 'dummy-build-c'); stage regex '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, ' at >= 40.0%; every 1 s; stale after 900 s; hold 0 s
2026-10-03T23:02:41+01:00 the run's manifest names no argv[0]; exiting without signalling
```

### Formatting and staging

`./venv/bin/python -m black --check` on the four Python files that changed or are new reports each
"would be left unchanged". The files were staged by explicit path and `git diff --cached
--name-only` was shown before the commit (see the report).

### Suites

Measured serially at the tree of `7bd202f`'s code, between 23:10 on 2026-10-03 and 00:15 on
2026-10-04, with README §7's command, each suite's output written to a file and its verdict
grepped from it:

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 686 run; 2 failures, both load-sensitive tests the brief names: `test_background_tau.TestBackgroundTau.test_build_cost` and `test_tk_wkb_phase.TestCost.test_wall_time_per_object`; the other 684 pass |
| CosmologyModels | 39 OK |
| Datastore | 598 OK |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 281 OK |

RunRegistry is 253 + the 28 tests of `test_build_rehearsal_scope.py`; every other count is as at
`3cdb52a`. Nothing under `ComputeTargets/` changed.

**The two ComputeTargets failures are load.** The 1-minute load average on this machine was 440-725
for the whole run (another project's Ray workers, `ray::compute_BBN_data`, each at ~95% CPU; the
ComputeTargets run took 1448 s against a few minutes unloaded). Re-run alone afterwards, at a load of
~544: `test_build_cost` **passed** (2.4 s); `test_wall_time_per_object` **failed** again, `0.1118 s`
against its `0.06 s` bound. The brief records it failing at every recent run for this reason. Neither
test imports anything this prompt changed, and no file in `ComputeTargets/` changed. I call them the
known wall-clock flakes, not regressions; I did not see `test_wall_time_per_object` pass in this
session.

## The deliberate-breakage record

Each diff is exactly as applied, taken with `git diff -- <the one file>` against the staged file,
and undone with `git checkout -- <the file>` afterwards. They replay with `git apply` in `bash`
from a clean tree at `7bd202f`. Each failure was read: those of (i)–(iv) are `AssertionError`s on
a list, a text or an exit code. Tests are in `RunRegistry/tests/test_build_rehearsal_scope.py`
(`B`) or `test_prune_on_resume.py` (`P`); the run was those two modules with `test_killed_handler`,
`test_store_fingerprint` and `test_quadsource_atol_sweep_prepare` (116 tests).

**(i) the rehearsal's `zend` used without the flag.** 4 failures.

```diff
diff --git a/docs/handover/quadsource_atol_sweep.py b/docs/handover/quadsource_atol_sweep.py
index 820d215..64bd202 100644
--- a/docs/handover/quadsource_atol_sweep.py
+++ b/docs/handover/quadsource_atol_sweep.py
@@ -440,7 +440,7 @@ def build_main_argv(args, database):
     is the scope's (`build_scope`): production's `0.1`, or the rehearsal's `250` with
     `--rehearsal`. Needs no Ray, so a test can call it.
     """
-    _, zend, _ = build_scope(args)
+    zend = REHEARSAL_ZEND
     return _main_py_argv(database, args.job_name, zend)
 
 
```

- `B.TestProduction.test_the_argv_list_is_the_production_list`
- `B.TestProduction.test_the_resumed_argv_list_is_the_same_list`
- `B.TestRehearsal.test_the_two_scopes_differ_in_the_three_places_and_no_others`
- `P.TestTheListsAreWhatTheyWere.test_the_build_list_is_what_run_build_built_less_the_flag`
  (`test_prune_on_resume` fails, as the prompt requires: at element 8 the list has `'250'` where it expects `'0.1'`)

**(ii) `run_build` gives `K_GRID_NAME` `k_sample()` again, the argv still the scope's.** 2 failures.

```diff
diff --git a/docs/handover/quadsource_atol_sweep.py b/docs/handover/quadsource_atol_sweep.py
index 820d215..869e620 100644
--- a/docs/handover/quadsource_atol_sweep.py
+++ b/docs/handover/quadsource_atol_sweep.py
@@ -642,7 +642,7 @@ def run_build(args):
     namespace = {
         "__name__": "__main__",
         "__file__": str(MAIN_PY),
-        K_GRID_NAME: k_values,
+        K_GRID_NAME: k_sample(),
         "SWEEP_QUAD_ATOL": args.atol,
         "SWEEP_QUAD_RTOL": args.rtol,
     }
```

- `B.TestTheDriversReadTheScope.test_run_builds_namespace_takes_the_grid_from_the_scope`
- `B.TestTheDriversReadTheScope.test_run_build_holds_no_call_of_k_sample`

Behaviour cannot see this one without Ray (the argv and the text are still the rehearsal's); `ast`
does.

**(iii) the rehearsal's scope text is the production string.** 3 failures.

```diff
diff --git a/docs/handover/quadsource_atol_sweep.py b/docs/handover/quadsource_atol_sweep.py
index 820d215..3278b11 100644
--- a/docs/handover/quadsource_atol_sweep.py
+++ b/docs/handover/quadsource_atol_sweep.py
@@ -327,7 +327,7 @@ def build_scope(args):
         k_values = np.logspace(
             np.log10(REHEARSAL_K_MIN), np.log10(REHEARSAL_K_MAX), REHEARSAL_K_COUNT
         ).tolist()
-        return k_values, REHEARSAL_ZEND, REHEARSAL_SCOPE_TEXT
+        return k_values, REHEARSAL_ZEND, PRODUCTION_SCOPE_TEXT
     return k_sample(), PRODUCTION_ZEND, PRODUCTION_SCOPE_TEXT
 
 
```

- `B.TestRehearsal.test_the_scope_text_says_what_ran`
- `B.TestRehearsal.test_the_scope_text_does_not_say_the_production_size`
- `B.TestRehearsal.test_the_two_scopes_differ_in_the_three_places_and_no_others`

**(iv) `main()` accepts `--rehearsal` with `--child`.** 1 failure.

```diff
diff --git a/docs/handover/quadsource_atol_sweep.py b/docs/handover/quadsource_atol_sweep.py
index 820d215..88040eb 100644
--- a/docs/handover/quadsource_atol_sweep.py
+++ b/docs/handover/quadsource_atol_sweep.py
@@ -1007,7 +1007,7 @@ def main():
     args = parser.parse_args()
 
     # before any mode runs: --prepare copies a store, and the sweep launches a build per pair
-    if args.rehearsal and not args.build:
+    if args.rehearsal and not (args.build or args.child):
         parser.error(
             "--rehearsal is valid only with --build: it selects the rehearsal scope of a "
             "build, which is not A3 v2"
```

- `B.TestRefusals.test_with_child_is_refused` (`None != 2`: `main()` did not exit)

**(v) `--rehearsal`'s default made `True`.** 11 failures and 10 errors.

```diff
diff --git a/docs/handover/quadsource_atol_sweep.py b/docs/handover/quadsource_atol_sweep.py
index 820d215..a9e8933 100644
--- a/docs/handover/quadsource_atol_sweep.py
+++ b/docs/handover/quadsource_atol_sweep.py
@@ -979,6 +979,7 @@ def main():
     parser.add_argument(
         "--rehearsal",
         action="store_true",
+        default=True,
         help="with --build only: a rehearsal scope (3 log-spaced k over 1e5-1e6 /Mpc, --zend "
         "250) in place of the production one. It is not A3 v2",
     )
```

- the 11 failures are all in `B`: `TestProduction.test_the_flag_defaults_to_off`,
  `test_the_k_sample_is_k_sample`, `test_the_k_sample_is_the_eight_point_production_grid`,
  `test_the_argv_list_is_the_production_list`, `test_the_resumed_argv_list_is_the_same_list`,
  `test_the_scope_text_is_what_run_build_wrote_before`,
  `TestRehearsal.test_the_two_scopes_differ_in_the_three_places_and_no_others`, and
  `TestRefusals.test_the_stubs_reach_each_mode_without_the_flag` for its four modes (`--prepare`,
  `--report`, `--child` and no mode are each refused by the flag that is now on).
- the 10 errors are in `P`: every test whose `setUp` parses a `--child` line gets `SystemExit(2)` from
  `main()`'s own refusal, which is (v)'s consequence reaching the older module (and so is a refusal,
  not a defect of a stand-in).

**(vi) `interrupt_when.py` compares the command line with the constant `"scoped_pipeline_run.py"`
again.** B3's new dummy case fails to signal, and the existing dummy test still passes.

```diff
diff --git a/docs/datastore-integrity-verification/interrupt_when.py b/docs/datastore-integrity-verification/interrupt_when.py
index bceb3ba..4ae3a38 100644
--- a/docs/datastore-integrity-verification/interrupt_when.py
+++ b/docs/datastore-integrity-verification/interrupt_when.py
@@ -193,7 +193,7 @@ def watch(run_dir: Path, pattern, minimum, interval, stale_after, log, hold=0.0)
                     )
                     return 0
                 command = command_of(pid)
-                if not names_the_run(command, slug, driver):
+                if not names_the_run(command, slug, "scoped_pipeline_run.py"):
                     log(
                         f"pid {pid}'s command line does not name {driver} (the run's own "
                         f"driver, its manifest's argv[0]) and --register {slug}: "
```

- `interrupter_build_dummy_test.sh` case A: the interrupter exits 2 with `command line does not name
  quadsource_atol_sweep.py ...` and the child it should have signalled exits 0 after its full 120 s
  `sleep`, not 143. Case B, the refusal, now **signals** the `scoped_pipeline_run.py` child (a
  `sent SIGTERM` line for `dummy-build-b`), so it fails too. Case C is unaffected.
- `interrupter_dummy_test.sh` run under the same mutation: the same two `sent SIGTERM` lines (cases
  A and D) and the same five cases as with the change; it passes.

## Observations not acted on

1. **`--resume` does not check the scope a store was built with** (prompt §6). A store built with
   `--rehearsal` and resumed without it, or the reverse, would mix the two scopes: the production
   grid's rows beside the rehearsal's, `--zend` 0.1 against 250. Nothing records a store's scope
   except its runs' manifests (the `scope` of each). 07 launches every build with the flag, so the
   reference and the resumed one cannot differ in it, but a person resuming by hand could. Not
   fixed here.
2. **`--prepare` and `--report` run before `--build` in `main()`**, so `--prepare --build
   --rehearsal` is not refused (`--build` is given) and `prepare()` runs. This is the existing
   dispatch order, which the prompt said to keep ("valid only with `--build`" is checked by
   `--build` being present). Not changed.
3. **No test runs `run_build` itself**, which needs Ray. What the rehearsal prints, and the exact
   `scope=` it hands `RunRegistry.begin`, are seen only through `ast` and through `build_scope`'s
   result. 07 is the first run to exercise them.
4. **The board's header ("7 of 9 prompts landed") and README §2's 06a row ("**written**") are
   stale after `7bd202f`.** Outside this prompt's file list ("the §1 row, items B1–B3, and §5").
5. **A refused child in a dummy case costs its whole `sleep`.** Breakage (vi) took about two
   minutes because the case-A child, correctly not signalled, sleeps 120 s and the script waits for
   it. It is how the existing test is built too.
6. **The existing `interrupter_dummy_test.txt` records the old refusal wording**
   (`does not name scoped_pipeline_run.py and --register dummy-c`), which is now
   `does not name scoped_pipeline_run.py (the run's own driver, its manifest's argv[0]) and --register
   dummy-c`. The committed file belongs to `datastore-integrity`'s verification and is left; this
   log's verification 4 is the current wording.

## State handed to the next prompt

- **To launch 07's builds:** `PYTHONPATH=. ./venv/bin/python docs/handover/quadsource_atol_sweep.py
  --build --rehearsal --database <store> --atol 1e-32 --rtol 1e-7 --register <slug> --purpose ...`
  from a tree at `7bd202f`, with nothing modified. Resume with the same line and `--resume`.
- **The scope it runs:** `main.py`'s own unfiltered work list, LambdaCDM, `--shards 4`,
  `--source-samples-log10z 100`, `--zend 250`, k = `[1e5, 316227.7660168379, 1e6]`, the pair from
  `--atol`/`--rtol`. The manifest's `scope` begins `REHEARSAL build, not A3 v2:`.
- **The interrupter** works on such a run: `python docs/datastore-integrity-verification/interrupt_when.py
  var/runs/<id> '<stage regex>' <percent>`. It signals only a pid whose command line contains
  `quadsource_atol_sweep.py` and `--register <slug>`, the driver and slug of the run's own manifest.
- **Open:** observation 1 (a resume does not check the scope); nothing was opened on the board.
- **Suites** as in §5.14 of the board. The index is **102**.
