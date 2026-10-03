# Log — prompt 05: prune unvalidated rows when a build resumes

**Subject:** *Prune unvalidated rows when an A3 v2 build resumes* · **Model:** Claude Sonnet 5.5 ·
**Date:** 2026-10-03 · **Base:** `633c3e9`, clean · **Result:** landed. `--no-prune-unvalidated` is
out of `run_build`'s and `run_child`'s argument lists for `main.py` and out of the usage line of
both `scoped_pipeline_run.py` module docstrings. Both lists are built at module scope by
functions that need no Ray, and a new test module keeps the flag, and its abbreviations, out. No
stop condition of the prompt's §5 fired. No build was run, no Ray was started, and nothing under
`var/` was read or written. `[10-a3-v2-launch-resumes-without-pruning-unvalidated-rows]` is
closed on the `datastore-integrity` board, with its index row deleted (104 to 103). Nothing was
opened. The commit's SHA is not written here: it is the orchestrator's to fill in on the board's
§1 row and in the closing note.

## What shipped

- **`docs/handover/quadsource_atol_sweep.py`** — three new module-level functions, placed before
  `run_child`, and two call sites:
  - `_main_py_argv(database, job_name)` returns the list (program name first): `main.py`,
    `--database`, `str(database)`, `--job-name`, `job_name`, `--shards`, `str(SHARDS)`, `--zend`,
    `0.1`, `--source-samples-log10z`, `100`. No flag.
  - `build_main_argv(args, database)` is `_main_py_argv(database, args.job_name)`.
  - `child_main_argv(args, database)` is `_main_py_argv(database,
    f"handover-atol-sweep-a{args.atol:g}-r{args.rtol:g}")`.
  - `run_build` now reads `sys.argv = build_main_argv(args, database)`, where `database` is its
    own `Path(args.database).resolve()`. `run_child` reads `sys.argv = child_main_argv(args,
    SWEEP_STORE)`. The local `main_args` is gone; nothing else read it. Each driver's
    `print(... main.py argv = {sys.argv[1:]})` is untouched.
  - By `ast`, against `HEAD`'s file: the top-level definitions that changed are `run_build` and
    `run_child` (the argv assignment only); the only ones added are the three above; none was
    removed. `main`, `sweep`, `substitute_main_py` and the module docstring are unchanged.
  - Numstat of the file: 42 added, 28 deleted. The test module is 385 lines added.
- **`docs/gktk-remedial/scoped_pipeline_run.py`** and
  **`docs/source-remediation-verification/scoped_pipeline_run.py`** (1 line each): the
  `--no-prune-unvalidated` token is gone from the usage line of the module docstring. By `ast`,
  everything after the docstring is identical to `HEAD`, and the docstring differs in that one
  line.
- **`RunRegistry/tests/test_prune_on_resume.py`** (new, 18 tests), below.
- **This log**, the two boards, and `docs/OPEN_ISSUES.md` (the row deleted; count and date).
- Not touched: `main.py` (breakage (iii) mutated it and `git checkout -- main.py` restored it;
  `git diff HEAD --stat -- main.py` is empty), any `RunRegistry/*.py`, the pool, every existing
  test and probe, `var/`, `prompts/a3-v2-readiness/orchestrator/`, and `run_child`'s `ray.init`
  and signal handling.

### The tests (`RunRegistry/tests/test_prune_on_resume.py`, 18)

The `Namespace` each list is built from is the one the script's own `main()` gives: `main()` runs
on a fresh import with `run_build` and `run_child` replaced by functions that keep their argument
(the orchestrator's method). So the defaults are the real ones, and `main()` is not refactored.
The database path in the tests is already resolved (`/private/tmp/...`), because `run_build`
resolves what it is given.

- **`TestTheListsAreWhatTheyWere`** (5), E4. The build's list is compared, by `assertEqual` on
  the list, with the pre-image less the flag; so is the child's; the `--resume` build equals the
  plain build; every element is a `str`; the job name is the `Namespace`'s (build) and
  `handover-atol-sweep-a{atol:g}-r{rtol:g}` (child). The expected lists are literal, with `"4"`
  for `--shards`.
- **`TestNothingDisablesPruning`** (5), E1.
  - No list contains `--no-prune-unvalidated`.
  - No list contains a token that begins `--no-p`, or a prefix of the flag that begins `--no`.
  - **`main.py`'s own option table, read with `ast`, parses each list as `prune_unvalidated` true.**
    The test rebuilds an `argparse.ArgumentParser()` with `allow_abbrev` at its default from
    every `parser.add_argument(...)` call in `main.py`: option strings, and each keyword whose
    value is a literal; `argparse.BooleanOptionalAction` is kept by name. A non-literal keyword
    (`type=str`, `default=DEFAULT_ZEND`, `choices=allowed_drop_actions`) is left out.
  - So that the instrument is not blind, a test shows the rebuilt parser reads `--no-prune-unvalidated`,
    `--no-prune`, `--no-prune-unvalid` and `--no-p` as `prune_unvalidated` false, and a list
    with the flag, or with `--no-prune`, put back as false.
- **`TestMainPyDefault`** (3), E3. The one `add_argument` call for `--prune-unvalidated` has
  `default=True`, is a `BooleanOptionalAction`, and has the option string `--prune-unvalidated`;
  and, of every option string main.py's calls spell, together with the `--no-` spelling of each
  `BooleanOptionalAction`, the only one beginning `--no-p` is the flag. That is the prompt's
  claim that no other option does.
- **`TestTheDriversCallThem`** (4), E2. By `ast`, `run_build` has one assignment to `sys.argv`,
  which is `build_main_argv(args, database)`; `run_child` has one, `child_main_argv(args,
  SWEEP_STORE)`; neither function holds a string constant that begins `--` with no space in it
  (an option literal); and the three builders import nothing.
- **`TestTheUsageLinesAreClean`** (1). Both module docstrings still hold
  `--source-samples-log10z 100` and no `--no-p`.

## Deviations from the prompt

1. **Two public functions over one private helper**, not one function or two independent ones.
   The prompt leaves it to the prompt. The two lists differ only in the job name, so
   `_main_py_argv(database, job_name)` holds the rest once, and `build_main_argv` and
   `child_main_argv` supply the job name. **IMPLEMENTATION CHOICE.**
2. **The functions return the whole `sys.argv`, program name included.** `run_build` assigned a
   list with `main.py` first; `run_child` assigned `["main.py"] + main_args`. Returning the full
   list makes each driver's assignment a single call and the two `print(... argv[1:])` lines
   unchanged. **IMPLEMENTATION CHOICE.**
3. **`run_child`'s local `main_args` is removed.** It existed only to be prefixed with `"main.py"`
   on the next line; by `ast` nothing else in `run_child` read it. **STRUCTURALLY REQUIRED** by
   moving the list out.
4. **Docstrings on the three functions** say why the list carries no pruning option, cite the
   issue ID and name the test module. **IMPLEMENTATION CHOICE.**
5. **The test also rebuilds `main.py`'s parser and parses each list with it**, and pins that the
   rebuilt parser sees the abbreviations. The prompt asked for the token and the abbreviation to
   be absent. A list of absent spellings cannot say that nothing else reads as the flag; the
   parse can, and it fails on any spelling `argparse` resolves to the flag. **IMPLEMENTATION
   CHOICE.**
6. **A test on the two usage lines** (`TestTheUsageLinesAreClean`), which P2's three bullets do
   not list. P1 removes the flag from the two docstrings "which teach it to anyone launching a
   pipeline by copying them", and nothing else keeps it removed. Breakages (i)–(v) do not
   exercise it; I did not add a sixth. **IMPLEMENTATION CHOICE.**
7. **The tests do not call `run_build` or `run_child`.** Both begin with Ray or a store check, and
   README §5 rule 9 forbids a test that needs either. The tests reach the argv through
   the functions, and the call sites through `ast` (`TestTheDriversCallThem`). What `run_build`
   resolves into `database` is therefore not run by any test; the test passes a path that is
   already resolved. **STRUCTURALLY REQUIRED** by rule 9.

No `UNINTENDED DRIFT` was found.

## Verification performed

### Both argument lists, before and after

`HEAD`'s lists are evaluated from `HEAD`'s source (`git show HEAD:docs/handover/quadsource_atol_sweep.py`,
the list literals of `run_build` and `run_child` read with `ast` and evaluated) beside the new
functions' output, on the `Namespace` that `main()` gives for each command line, with `<db>` for
the resolved path of a file in a scratch directory. The script is in the scratchpad, not the
repository.

**`--build --database <db> --atol 1e-32 --rtol 1e-7`**, with or without `--resume`. The
`Namespace` is `job_name='handover-A3-baseline-v2'`, `resume=False` (or `True`), `cpus=6`,
`phase='main'`, `register=None`, `atol=1e-32`, `rtol=1e-07`. `run_build`'s list, before:

```
['main.py', '--database', '<db>', '--job-name', 'handover-A3-baseline-v2', '--shards', '4',
 '--zend', '0.1', '--source-samples-log10z', '100', '--no-prune-unvalidated']
```

After, `build_main_argv(args, Path(args.database).resolve())`; this is the argv that §3 item 3
asks to be shown in place of the build:

```
['main.py', '--database', '<db>', '--job-name', 'handover-A3-baseline-v2', '--shards', '4',
 '--zend', '0.1', '--source-samples-log10z', '100']
```

**`--child --atol 1e-32 --rtol 1e-7`** (`database=None`, the other defaults the same), before:

```
['main.py', '--database', '/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/handover-atol-sweep.sqlite',
 '--job-name', 'handover-atol-sweep-a1e-32-r1e-07', '--shards', '4', '--zend', '0.1',
 '--source-samples-log10z', '100', '--no-prune-unvalidated']
```

After, `child_main_argv(args, SWEEP_STORE)`:

```
['main.py', '--database', '/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/handover-atol-sweep.sqlite',
 '--job-name', 'handover-atol-sweep-a1e-32-r1e-07', '--shards', '4', '--zend', '0.1',
 '--source-samples-log10z', '100']
```

For each of the three, the script printed `after == before less the flag: True`, `flag in before:
True`, `flag in after: False`. This is §3 item 3: the three were obtained through the real
parser, and no build was run (§3 item 3's command was not run).

### The other checks

- **§3 item 1.** `test_prune_on_resume` passes (18 tests). The RunRegistry suite is 243 OK:
  the 225 of `79ab21a` plus the 18, with no existing test changed.
- **§3 item 2.** `substitute_main_py(filter_cases=False)` still matches each of its four
  substitutions once: the two `np.logspace` grid lines (`NUMBER_SOURCE_K_VALUES`,
  `NUMBER_RESPONSE_K_VALUES`), `tol=DEFAULT_QUADRATURE_ATOL` and `tol=DEFAULT_QUADRATURE_RTOL`,
  each "replaced 1 occurrence". `main.py` is unchanged, so this was as expected.
- **The first stop condition** (the flag's removal changes something else `main.py` does at
  startup that a resume relies on): not fired. `grep -n prune main.py` shows `args.prune_unvalidated`
  read once, at `ShardedPool(..., prune_unvalidated=...)`; I did not re-read the pool, and rely
  for what pruning does there on the orchestrator's facts.
- **The second** (an existing test pins the old argv): not fired. `grep -rn 'no-prune'` over
  every file outside `var/`, `venv/`, `.git/` and the orchestrator directory finds only
  Markdown records, one probe output text (`docs/datastore-integrity-verification/interrupter_firings.txt`)
  and this commit's own files; no test in any `tests/` directory held either list before, and
  all existing tests pass.
- **A first pass of breakage (ii)** raised `AttributeError` in `test_the_job_name_is_the_namespaces`,
  whose stand-in `Namespace` had no `resume`. That was an error, not a failure, so the test was
  caught for the wrong reason. It was fixed by giving the stand-in `resume=False`, and the
  breakages were re-run. The record below is of the re-run.
- **The first diffs** were taken with `git diff`, which also held the unstaged Markdown edits;
  they did not apply. They were retaken with `git diff -- main.py
  docs/handover/quadsource_atol_sweep.py`, after the code files were staged, and `git apply --check`
  of each succeeds against the tree.

### Suites

Measured 17:34–17:43 at the tree of this commit's code, serially, each suite's output written to
a file and its verdict grepped from it (README §7). The tree held the Markdown edits of this
commit unstaged and the four code files staged:

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 686 OK |
| CosmologyModels | 39 OK |
| Datastore | 598 OK |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 243 OK |

The 1-minute load average on 10 CPUs was 5.9 at the start, 22 after ComputeTargets, 77 after
Datastore and 32 at the end, on a machine beside another project's Ray cluster. No flake fired.

### Formatting and staging

`./venv/bin/python -m black --check` on the four Python files: each "would be left unchanged".
Files were staged by explicit path, and `git diff --cached --name-only` was shown before the
commit (see the commit's staging in the report).

## The deliberate-breakage record

Each diff is exactly as applied, was taken against the staged files, and was reverted afterwards.
`git apply` replays each from a clean tree at the commit: each was checked with `git apply --check`
against the staged tree before the commit. After each, `git status` was clean for
`main.py` and the sweep script. The tests are in `RunRegistry/tests/test_prune_on_resume.py`; the
module's name is dropped below. A failure count is `unittest`'s, so a failing sub-test counts as
one.

**(i) the flag back into the build's list.** 7 failures.

```diff
diff --git a/docs/handover/quadsource_atol_sweep.py b/docs/handover/quadsource_atol_sweep.py
--- a/docs/handover/quadsource_atol_sweep.py
+++ b/docs/handover/quadsource_atol_sweep.py
@@ -392,7 +392,7 @@ def build_main_argv(args, database):
     It does not depend on `--resume`, which only decides the pre-check in `run_build`. Needs no
     Ray, so a test can call it.
     """
-    return _main_py_argv(database, args.job_name)
+    return _main_py_argv(database, args.job_name) + ["--no-prune-unvalidated"]
 
 
 def child_main_argv(args, database):
```

- `TestTheListsAreWhatTheyWere.test_the_build_list_is_what_run_build_built_less_the_flag`
- `TestNothingDisablesPruning.test_no_list_contains_the_flag` (`build`, `build --resume`)
- `TestNothingDisablesPruning.test_no_list_contains_an_abbreviation_of_it` (both)
- `TestNothingDisablesPruning.test_main_pys_own_parser_reads_every_list_as_pruning` (both)

**(ii) the flag back only when `--resume` is given.** 4 failures. `build` and `child` are
unaffected; only the `--resume` list shows it.

```diff
diff --git a/docs/handover/quadsource_atol_sweep.py b/docs/handover/quadsource_atol_sweep.py
--- a/docs/handover/quadsource_atol_sweep.py
+++ b/docs/handover/quadsource_atol_sweep.py
@@ -392,7 +392,10 @@ def build_main_argv(args, database):
     It does not depend on `--resume`, which only decides the pre-check in `run_build`. Needs no
     Ray, so a test can call it.
     """
-    return _main_py_argv(database, args.job_name)
+    argv = _main_py_argv(database, args.job_name)
+    if args.resume:
+        argv.append("--no-prune-unvalidated")
+    return argv
 
 
 def child_main_argv(args, database):
```

- `TestTheListsAreWhatTheyWere.test_the_resumed_build_list_is_the_same_list`
- `TestNothingDisablesPruning.test_no_list_contains_the_flag` (`build --resume`)
- `TestNothingDisablesPruning.test_no_list_contains_an_abbreviation_of_it` (`build --resume`)
- `TestNothingDisablesPruning.test_main_pys_own_parser_reads_every_list_as_pruning`
  (`build --resume`)

**(iii) `main.py`'s default to `False`**, a mutation of the file, restored afterwards with
`git checkout -- main.py`. 5 failures.

```diff
diff --git a/main.py b/main.py
--- a/main.py
+++ b/main.py
@@ -235,7 +235,7 @@ parser.add_argument(
 parser.add_argument(
     "--prune-unvalidated",
     action=argparse.BooleanOptionalAction,
-    default=True,
+    default=False,
     help="prune unvalidated data from the datastore during startup",
 )
 parser.add_argument(
```

- `TestMainPyDefault.test_prune_unvalidated_defaults_to_true_in_main_pys_source`
- `TestNothingDisablesPruning.test_main_pys_own_parser_reads_every_list_as_pruning` (`build`,
  `build --resume`, `child`)
- `TestNothingDisablesPruning.test_the_parser_rebuilt_from_main_py_sees_what_main_pys_does`

**(iv) `--source-samples-log10z 100` dropped from the build's list.** 1 failure.

```diff
diff --git a/docs/handover/quadsource_atol_sweep.py b/docs/handover/quadsource_atol_sweep.py
--- a/docs/handover/quadsource_atol_sweep.py
+++ b/docs/handover/quadsource_atol_sweep.py
@@ -392,7 +392,7 @@ def build_main_argv(args, database):
     It does not depend on `--resume`, which only decides the pre-check in `run_build`. Needs no
     Ray, so a test can call it.
     """
-    return _main_py_argv(database, args.job_name)
+    return _main_py_argv(database, args.job_name)[:-2]
 
 
 def child_main_argv(args, database):
```

- `TestTheListsAreWhatTheyWere.test_the_build_list_is_what_run_build_built_less_the_flag`

Only the element-by-element comparison sees this; as expected, the other tests are about pruning.

**(v) `--no-prune` put into the child's list.** 3 failures. The abbreviation is not the full
token, so the token test passes; the parse and the element comparison fail.

```diff
diff --git a/docs/handover/quadsource_atol_sweep.py b/docs/handover/quadsource_atol_sweep.py
--- a/docs/handover/quadsource_atol_sweep.py
+++ b/docs/handover/quadsource_atol_sweep.py
@@ -400,7 +400,7 @@ def child_main_argv(args, database):
 
     Needs no Ray, so a test can call it.
     """
-    return _main_py_argv(database, f"handover-atol-sweep-a{args.atol:g}-r{args.rtol:g}")
+    return _main_py_argv(database, f"handover-atol-sweep-a{args.atol:g}-r{args.rtol:g}") + ["--no-prune"]
 
 
 def run_child(args):
```

- `TestTheListsAreWhatTheyWere.test_the_child_list_is_what_run_child_built_less_the_flag`
- `TestNothingDisablesPruning.test_no_list_contains_an_abbreviation_of_it` (`child`)
- `TestNothingDisablesPruning.test_main_pys_own_parser_reads_every_list_as_pruning` (`child`)

`test_no_list_contains_the_flag` passes under (v), as it should: the full token is not there.
That is the case the 2026-10-03 amendment is about.

## Observations not acted on

1. **`run_child` still calls `ray.init` with no `address` and installs no handler.** That is
   `[04-the-sweeps-run-child-shares-the-sigterm-and-cluster-address-defects]`, which stays open.
   This commit changed `run_child`'s argument list and nothing else.
2. **The sweep store `run_child` writes into is retired**, so the pruning refusal the
   orchestrator's facts describe (a `replication_in_flight` table created before `b647c0b`
   keeps `controller_shard NOT NULL` and refuses a pruning open that has a replicated row to
   prune) cannot reach `run_child` or A3 v2's launch as things stand. That is the
   orchestrator's finding, not mine: I did not read `ShardedPool.py` or look under `var/`. I
   judge it does not bear on this prompt, because `run_child` cannot run against a retired
   store.
3. **Records that still call the issue open or the flag current**, none of which this prompt may
   edit and none of which is wrong for the tree it was written on:
   - `docs/datastore-integrity-verification.md` (§2.1.4, §2.6) and prompt 10's log;
   - `prompts/a3-v2-readiness/README.md` §0 item 1 and §2's row for 05 ("**written**;
     independent"), which is stale now that 05 has landed. The README is outside this prompt's
     file list.
4. **The test's rebuilt parser is only as good as its reading of `main.py`.** It keeps literal
   keywords and `argparse.BooleanOptionalAction` by name and drops the rest. A future option
   whose `action` is something else non-literal would be built as a plain `store`. Today's 22
   calls are all of a kind it reads; an option added with a `nargs` or `action` it cannot read
   would not raise.
5. **`docs/OPEN_ISSUES.md`'s §1.17 intro prose gives no count of open issues**, so only the
   header count changed. The header date was already 2026-10-03.
6. **The extraction scripts' `prune_unvalidated=False`** is right for a reader and is prompt 03's
   to make structural; untouched.
7. **`TestTheUsageLinesAreClean` is a docstring check only.** The two `scoped_pipeline_run.py`
   drivers take their `main.py` options after `--`, verbatim, so what they pass is the
   launcher's own list. No code path of either file passes `--no-prune-unvalidated`.

## State handed to the next prompt

- **06** (`count only real references`) is independent and does not touch any file here.
- **07** (`rehearse the A3 v2 launch`) is held on 01–06. Its **L2** is the measurement of a
  pruned resume at the tree A3 v2 will run: `resumed-d` ran on 2026-10-01 at 23:54, before
  prompt 02 (`b647c0b`) moved the prune of the replicated classes into the pool. The 45
  `GkSource` rows it removed are a sharded class, which an actor still prunes, so the measurement
  stands for them. After this commit `--build --resume` of A3 v2 prunes unvalidated rows at
  startup, by `main.py`'s default.
- **Baselines** (board §5.10): AdaptiveLevin 32, ComputeTargets 686, CosmologyModels 39,
  Datastore 598, LiouvilleGreen 148 (1 skipped), RunRegistry 243.
- **Index:** 103 open, from 104. Nothing opened.
- **The orchestrator's residue:** this commit's SHA in the board's §1 row for 05 and in the
  closing note of
  `[10-a3-v2-launch-resumes-without-pruning-unvalidated-rows]` on the `datastore-integrity`
  board's §4.
