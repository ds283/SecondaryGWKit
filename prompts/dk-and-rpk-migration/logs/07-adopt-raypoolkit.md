# Log — prompt 07, adopt `raypoolkit`

**Prompt:** [`../07-adopt-raypoolkit.md`](../07-adopt-raypoolkit.md) ·
**Board items:** B1–B3 · **Date:** 2026-10-11 · **Model:** Sonnet 5.5 ·
**Commit:** this commit · **Result:** landed, no stop. B1–B3 as the prompt says. `RayTools/` is gone,
seven import lines read `from raypoolkit import RayWorkPool`, `CLAUDE.md`'s section covers both
packages, and one comment sentence of `test_pipeline_adoption.py` is corrected. RayPoolKit's script
finds 0 imports and 0 `import_module` calls naming `RayTools`, and all 54 pool constructions
`constructs`. All six roots at their counts. Nothing opened (index stays 108).

Dispatched at `789b72d` (code outside `prompts/` is `b6cc119`'s), tree clean, registry 25 runs (23
finished, 2 `unknown`), none `running`, no Ray process started or needed. Governed by the prompt and
the orchestrator's brief for 07 (its §0), cited below as "the brief". Saves are in
`var/dk-and-rpk-migration-07/` (gitignored): `measure-head.md`, `before/` and `after6/` (the six
named modules, `-v`), `after/` (the six roots, `-v`), `s33.txt`, `scratch-sha.txt`, and `breakage/`.
No scratch file was written under the repository's tracked trees, and nothing under `RayTools/`
remains.

## What shipped

- **B1.** `git rm RayTools/RayWorkPool.py RayTools/__init__.py`. The 575-line copy (blob `3c1d9bd5…`,
  `b4666fa…` plus 04's U12 line) and the empty `__init__.py` go; so does the one line 04 rewrote
  (`:8`). `utilities.format_time` stays. The directory's ignored `__pycache__` was removed too
  (deviation 4).
- **B2.** Seven lines, one each, `from RayTools.RayWorkPool import RayWorkPool` to
  `from raypoolkit import RayWorkPool` (the `-U0` hunks are below).
- **B3.1.** `CLAUDE.md`'s section is retitled "The datastore layer is `datastorekit`; the work queue
  is `raypoolkit`" and covers both packages (the first paragraph names RayPoolKit, its path and URL,
  `v0.2.0`; "Neither is this repository's code" reports each to its own repository; "Imports name
  the package" adds `from raypoolkit import RayWorkPool` and says the name stays bare because five
  `ComputeTargets/tests` modules find constructions by `ast.Name`). The `RayTools/` paragraph goes.
  The section is **27 lines** (`CLAUDE.md:47-73`, heading to the last line, counted as the brief
  counts it) against 26 before and the brief's limit of 29. Nothing else in `CLAUDE.md` changed.
- **B3.2.** `RunRegistry/tests/test_pipeline_adoption.py:32-33`, the first sentence of the comment
  only, still two lines (so `A3_OUTPUT` stays at `:38-44`).
- **New:** `measure/out/07-raypoolkit-measure_client_pools.md` (§3.1) and this log.
- **Records:** the board (§1 row, B1–B3, `### 5.7`), README §4's "07 ships" line.
  `docs/OPEN_ISSUES.md` is untouched.

### The seven `-U0` hunks (`git diff -U0 -- main.py 'extract_*.py'`)

```
extract_GkSource_data.py        @@ -37 +37 @@   -from RayTools.RayWorkPool import RayWorkPool  +from raypoolkit import RayWorkPool
extract_GkWKB_data.py           @@ -35 +35 @@   (the same two lines)
extract_Gk_data.py              @@ -32 +32 @@
extract_QuadSourceIntegral_data.py @@ -42 +42 @@
extract_TkWKB_data.py           @@ -30 +30 @@
extract_tensor_source_data.py   @@ -31 +31 @@
main.py                         @@ -63 +63 @@
```

Each hunk is that one line, `-1 +1`, at the prompt's line, and nothing else in the file changed.
`black --check` leaves all eight touched Python files unchanged. The test file's `-U0` hunk:

```
@@ -32,2 +32,2 @@
-# Six lines in the vocabulary `RayWorkPool` prints, in the order it prints them: the banner from
-# its constructor (:166), two progress notifications (:548) and the completion pair (:566). They
+# Five lines in the vocabulary `RayWorkPool` prints: the banner (`raypoolkit/RayWorkPool.py:206`)
+# and two progress notifications of two lines (`print(msg)` :588); no completion line (:606). They
```

The three places were checked against the installed `raypoolkit/RayWorkPool.py`: `:206` is
`print(f"\n** {title}")`, `:588` is `print(msg)` (the second line, `:592`), `:606` is the completion
`print(`.

## Deviations from the prompt

1. **The comment is `:32-37`, not `:31-37`; B3.2 rewrites `:32-33`** (prompt "Read first" 6 and B3.2).
   The brief, correction 1, gives `:32-37`, `A3_OUTPUT` `:38-44`, its banner `:39`. **IMPLEMENTATION
   CHOICE** (the brief, correction 1).
2. **Checklist item 5's trace is checked in the installed `datastorekit`, not in SGK's deleted
   copies** (the brief, correction 2). **IMPLEMENTATION CHOICE.** See "Checklist item 5" below.
3. **The `prompts/` string literal** is
   `prompts/datastore-generic/orchestrator/measure-09/scripts/m3_make_mutations.py:33` (the brief,
   correction 3). **IMPLEMENTATION CHOICE.** The measurement agrees.
4. **`RayTools/__pycache__/` was removed after `git rm`.** `git rm` leaves the directory, because it
   holds ignored bytecode, and a directory with no `__init__.py` imports as a namespace package, so
   `import RayTools` would not raise `ModuleNotFoundError` (§3.3 requires it to). The bytecode is
   generated and ignored. **STRUCTURALLY REQUIRED.**
5. **The corrected comment says what the block holds in two lines**, banner (`:206`) and progress
   notifications (`:588`), and that there is no completion line (`:606`); it drops "in the order it
   prints them", "from its constructor" and the completion pair. The prompt asked for the places and
   for the sentence to say what the block holds; the wording is mine. **IMPLEMENTATION CHOICE.**
6. **Breakages (ii) and (iii) were made on the scratch commit `38b1462`, taken before one reflow of
   `CLAUDE.md`** (an inline code span had broken across a line); the final scratch commit is
   `5624f4c`. The two trees differ in that file only. **IMPLEMENTATION CHOICE.**

No `UNINTENDED DRIFT`.

## Verification performed

### Order of work

1. Script at `HEAD` (`789b72d`); before test ids of the six named modules (`-v`).
2. `git rm` (B1); the seven lines (B2); `RayTools/__pycache__` removed; B3.2; B3.1.
3. `black --check`; `py_compile`; stage by explicit path; scratch commit; script at it.
4. Six roots; ids after; §3.3; the four breakages; records; commit.

### §3.1 — RayPoolKit's script

From `/Users/ds283/Documents/Code/RayPoolKit` (clean, `main`), with its venv, output redirected into
SGK by absolute path. `measure-head.md` is at `789b72d`; the staged tree is the scratch commit
`5624f4c` (`git commit-tree $(git write-tree) -p HEAD`), saved as
[`../measure/out/07-raypoolkit-measure_client_pools.md`](../measure/out/07-raypoolkit-measure_client_pools.md).

| What | At `789b72d` (HEAD) | At `5624f4c` (staged) | Brief |
|---|---|---|---|
| `git ls-tree <c> RayTools` | 2 blobs (`3c1d9bd5…`, `e69de29…`) | **empty** | empty |
| files tracked | 1473 | 1471 | records excepted |
| imports of a `RayTools` module | 7 (module 7, nested 0, private 0) | **0** (`None.`) | 0 |
| `import_module` / `__import__` calls naming it | 0 | **0** | 0 |
| string literals naming `RayTools` | 2 | **2** (`docs/tolerance-convergence/inventory.py:382`, `prompts/datastore-generic/orchestrator/measure-09/scripts/m3_make_mutations.py:33`) | 2 |
| `RayWorkPool(...)` calls | 54 | **54** | 54 |
| verdicts | `constructs` 54 | **`constructs` 54** | 54 |
| no compute handler / given / default | 44 / 6 / 4 | 44 / 6 / 4 | |
| `store_results=True`; `title` set | 39; 26 | 39; 26 | |
| test modules naming either | 6 | 6 | 6 |
| non-`.py` lines naming either | 474 | 475 | |

The handler-keyword table is byte-identical at both commits. Everything else in the two Markdown
files differs only by the commit, the file count, the seven import rows and the two lines in
`CLAUDE.md` that now name `RayWorkPool`/`raypoolkit` (the `:71` line that named `RayTools/` is gone),
and the test comment. `git diff --stat 5624f4c HEAD` lists the 11 files of B1–B3 and nothing else.
The landed commit adds only its records to that tree (the log, the board, README §4's line, and
the measurement).

### §3.2 — the suites

README §7's command with `-v`, each in the foreground, serially, Datastore in four groups, load 4–9.

| Root | At dispatch (`cd03ec4`) | After | Expected |
|---|---|---|---|
| `AdaptiveLevin/tests` | 32 OK | **32** OK | 32 |
| `CosmologyModels/tests` | 39 OK | **39** OK | 39 |
| `LiouvilleGreen/tests` | 148 OK (1 skipped) | **148** OK (1 skipped) | 148 |
| `RunRegistry/tests` | 286 OK | **286** OK | 286 |
| `ComputeTargets/tests` | 693, 1 failure (wall clock) | **693** OK | 693 |
| `Datastore/tests` a–f / g–o / p–r / s–z | 54 / 103 / 101 / 62 OK | **54 / 103 / 101 / 62** OK = 320 | 320 |

ComputeTargets passed whole this time, the wall-clock test included, so no re-run alone was needed.

**The six named modules**, `unittest -v` at `HEAD` before any edit and after, ids with outcomes
compared (`var/dk-and-rpk-migration-07/before/` and `after6/`):

| Module | Tests before / after | Ids and outcomes |
|---|---|---|
| `ComputeTargets.tests.test_main_plumbing` | 27 / 27 | equal, OK |
| `ComputeTargets.tests.test_quadsource_policy_main` | 19 / 19 | equal, OK |
| `ComputeTargets.tests.test_wkb_numeric_parent_main` | 14 / 14 | equal, OK |
| `ComputeTargets.tests.test_gksource_parent_set_main` | 25 / 25 | equal, OK |
| `ComputeTargets.tests.test_quadsource_integral_parent_main` | 30 / 30 | equal, OK |
| `RunRegistry.tests.test_pipeline_adoption` | 15 / 15 | equal, OK |

`black --check` is clean on the eight touched Python files (`main.py`, the six `extract_*.py`,
`test_pipeline_adoption.py`; black 25.1.0).

### §3.3 — through the package, without Ray

1. `py_compile` succeeded on `main.py`, the six `extract_*.py` and `extract_common.py` (bytecode
   written to the scratch directory). None was imported.
2. `./venv/bin/python -I` with the repository on `sys.path` (`s33.txt`):
   ```
   raypoolkit.RayWorkPool True
   RayTools ModuleNotFoundError: No module named 'RayTools'
   RayTools.RayWorkPool ModuleNotFoundError: No module named 'RayTools'
   ray.is_initialized() False
   ```
3. `PYTHONPATH=. ./venv/bin/python -m RunRegistry list`: exit 0, "25 run(s) … 23 finished, 2
   unknown".

### Checklist item 5, row by row, against the tree

| Change | At SGK now |
|---|---|
| Constructor checks raise `ValueError` | no site affected: all 54 construct under `raypoolkit 0.2.0`'s own constructor (§3.1). The three `except` clauses the checklist names are `main.py:297` (`except RuntimeError` round `read_inventory`), `:505` and `:520` (round `obj.has_unresolved_osc` and `obj.unresolved_efolds_subh`); none wraps a `RayWorkPool(...)` |
| A pool with no compute handler needs only `compute_handler=None` | nothing: the 44 lookup-only sites keep their spelling (C7) |
| **Post handler after a persist, no validation handler, gets the persisted object** | one site, `main.py:3247` (below) |
| `store_results=True` accepts a generator | nothing: 39 sites pass `True`, none a generator |
| In-flight work keyed on each ref's value | every site, no visible change |
| "Available" rate from the available count | already so in SGK's copy |
| Messages say "handler" | exception text only; no site raises one |

The other post-handler sites: `main.py:1427` and `:2122` (`record_unresolved_osc` lambdas) follow a
validation handler (the table's `set | — | — | — | set | set | set` rows, 2 calls); `:3235` passes
`post_handler=None`; `:2985` (`dump_Gk_handler`) is a lookup-only pool with no compute or persist
handler, so there is no persist whose result it could be given. The checklist's `:1420`, `:2115`
and `:3229` are these at `b510bc9`.

**`main.py:3247` and the trace, checked in the installed packages** (the brief, correction 2):
- `GkSourcePolicy_queue` is at `main.py:3247`, `compute_handler=compute_GkSourcePolicy_work`,
  `validation_handler=None`, `post_handler=postprocess_GkSourcePolicy` (`:3253`); the statistics
  banner is `:3265`. `postprocess_GkSourcePolicy` is `main.py:2871`; it reads `obj.policy`,
  `policy.label` (or `policy.store_id`), `obj.type` and `obj.quality`.
- `config/sharding.py:32`: `"GkSourcePolicyData": "k"`.
- `datastorekit/SQL/ShardedPool.py:3441` is `object_store`; its sharded branch is
  `self._shards[shard_id].object_store.remote(item)` at `:3531`; the actor's `object_store` is
  `datastorekit/SQL/Datastore.py:661` and returns `output_objects[0]` for one object (`:733`).
- The factory's `store`, `Datastore/SQL/ObjectFactories/GkSourcePolicyData.py:150-174`, sets
  `obj._my_id` at `:172` and returns `obj` at `:174`.
- `ComputeTargets/GkSourcePolicyData.py`: `_policy` at `:519`; `_type` at `:574` and `_quality` at
  `:575` (the checklist's `:574-575`; the brief's `:573-574` is one line early). The properties are
  `:585` (`policy`), `:589` (`type`), `:596` (`quality`).
- `git grep` for `__getstate__|__setstate__|__reduce__` in installed `datastorekit` finds them only
  in `replication.py` and `SQL/schema.py`; `datastorekit/object.py` has none.
- In the package, the default handlers are `raypoolkit/RayWorkPool.py:79-84` and the post handler is
  called with the persisted object at `:529-530`.

By reading, nothing the handler records changes. 07 does not measure it; 08 does (C8).

## The deliberate-breakage record

Each diff is exactly as applied, made against the staged tree (index against working tree), checked
with `git apply -R --check` while applied and `git apply --check` after the revert. Saves are in
`var/dk-and-rpk-migration-07/breakage/`. After each, the working tree equalled the staged tree. (The
first attempt at (i) saved a `-U0` diff, which `git apply` refuses without `--unidiff-zero`; the diffs
here are the default context.) Replay in `bash` from a clean tree at this commit.

### (i) a call spelled `raypoolkit.RayWorkPool(...)`, at `main.py:3686`

The site is the `query_queue` of `build_QuadSourceIntegral_batch`, one the brief tried at `HEAD`.

```diff
diff --git a/main.py b/main.py
index 7d17e8c..e3a58e0 100644
--- a/main.py
+++ b/main.py
@@ -61,6 +61,7 @@ from LiouvilleGreen.bessel_phase import bessel_phase
 from MetadataConcepts import GkSourcePolicy, QuadSourcePolicy
 from Quadrature.integration_metadata import IntegrationSolver
 from raypoolkit import RayWorkPool
+import raypoolkit
 from Units import Mpc_units
 from config.defaults import (
     DEFAULT_FLOAT_PRECISION,
@@ -3683,7 +3684,7 @@ def run_pipeline(
             for k in k_keys
         ]
 
-        query_queue = RayWorkPool(
+        query_queue = raypoolkit.RayWorkPool(
             pool,
             query_batch,
             task_builder=lambda x: pool.object_get_vectorized(
```

Six tests fail, none errors, as the brief found:
- `test_quadsource_policy_main.TestMainPyKeysOnQuadSourcePolicy1pt5`: `test_the_two_sites_are_found`
  and `test_the_GkSourcePolicy_is_still_passed_beside_it` (`['construction', 'construction'] !=
  ['construction', 'query']`), and `test_each_site_passes_QuadSource_policy_1pt5` (`[] !=
  ['QuadSource_policy_1pt5']`, site `'construction'`).
- `test_main_plumbing.ObjectGetToleranceWiringTestCase`: `test_the_expected_sites_are_all_found`,
  `test_every_object_get_has_a_classified_tolerance` (`[('QuadSourceIntegral', [], 'build_QuadSou…`
  `!= []`) and `test_the_already_decoupled_target_is_guarded` (`1 != 2`).
- `test_wkb_numeric_parent_main`, `test_gksource_parent_set_main` and
  `test_quadsource_integral_parent_main` still pass: this site is not one they inspect.

The two modules that find the call by `ast.Name` see the attribute call as no construction.

### (ii) one `extract_*.py` import reverted

```diff
diff --git a/extract_Gk_data.py b/extract_Gk_data.py
index 323f3c3..56e0c60 100644
--- a/extract_Gk_data.py
+++ b/extract_Gk_data.py
@@ -29,7 +29,7 @@ from CosmologyConcepts.wavenumber import (
 from datastorekit.SQL.ProfileAgent import ProfileAgent
 from datastorekit.SQL.ShardedPool import ShardedPool
 from MetadataConcepts import tolerance
-from raypoolkit import RayWorkPool
+from RayTools.RayWorkPool import RayWorkPool
 from Units import Mpc_units
 from config.defaults import (
     DEFAULT_GK_NUMERIC_ABS_TOLERANCE,
```

Scratch commit `694446f` (a temporary index, `GIT_INDEX_FILE=<scratch>/idx git read-tree 38b1462`, `git
update-index --cacheinfo` the mutated blob, `write-tree`, `commit-tree -p 38b1462`; no ref names it,
and the checkout's index was never mutated). RayPoolKit's script gives **1 import**,
`extract_Gk_data.py:32`, and still `constructs` 54. No test imports an `extract_*.py`, so the script
is the check.

### (iii) `main.py:63` reverted

```diff
diff --git a/main.py b/main.py
index 7d17e8c..e636e92 100644
--- a/main.py
+++ b/main.py
@@ -60,7 +60,7 @@ from datastorekit.SQL.ShardedPool import ShardedPool
 from LiouvilleGreen.bessel_phase import bessel_phase
 from MetadataConcepts import GkSourcePolicy, QuadSourcePolicy
 from Quadrature.integration_metadata import IntegrationSolver
-from raypoolkit import RayWorkPool
+from RayTools.RayWorkPool import RayWorkPool
 from Units import Mpc_units
 from config.defaults import (
     DEFAULT_FLOAT_PRECISION,
```

Scratch commit `7c722d1` from a temporary index, as (ii). The script gives **1 import**,
`main.py:63`. With the mutation applied in the working tree, `test_main_plumbing` gave `Ran 27
tests … OK`: it reads `main.py` by `ast` and never runs its imports. So the script, not the suite, is
B2's proof.

### (iv) `A3_OUTPUT`'s banner changed

```diff
diff --git a/RunRegistry/tests/test_pipeline_adoption.py b/RunRegistry/tests/test_pipeline_adoption.py
index ebf0a99..dfd1069 100644
--- a/RunRegistry/tests/test_pipeline_adoption.py
+++ b/RunRegistry/tests/test_pipeline_adoption.py
@@ -36,7 +36,7 @@ DRIVER = os.path.join(REPO_ROOT, "docs", "gktk-remedial", "scoped_pipeline_run.p
 # run on a fresh clone. The whole of that file, fed through `StageTracker`, ends at the last
 # stage asserted below.
 A3_OUTPUT = """
-** CALCULATE QUADRATIC SOURCE INTEGRALS
+* CALCULATE QUADRATIC SOURCE INTEGRALS
    -- 2026-09-21 02:14:03+0100 (5h 11m running): 4/13 work items remaining = 69.23% complete
       inflight: 0 lookup, 4 compute, 0 store | completed: 9 lookup, 9 compute, 9 store
    -- 2026-09-21 07:36:58+0100 (10h 33m running): 1/13 work items remaining = 92.31% complete
```

`test_pipeline_adoption`: `Ran 15 tests`, `FAILED (failures=1, errors=2)`, as the brief found:
`TestStage.test_the_stage_is_the_line_that_had_to_be_reconstructed_by_hand` (`None != 'CALCULATE
QUADRATIC SOURCE INTEGRALS, 92.31% (1 of 13 work items remaining)'`), and the errors
`TestStage.test_the_stage_is_the_hand_written_manifests_own_field` (`TypeError: argument of type
'NoneType' is not iterable`) and
`TestHeartbeatHonesty.test_the_stream_passes_everything_through_and_beats_on_a_new_stage`
(`KeyError: 'stage'`). The pinned lines still bite after the swap.

## Observations not acted on

- **`test_pipeline_adoption.py:33-37`'s second sentence is not verbatim true** (the brief, §0). It
  says the block is "transcribed from `var/runs/a3-pilot/run.out`". That file's progress lines carry
  no UTC offset, the pool's `datetime.now()` is naive so `%z` prints nothing, and neither stamp of
  the block (`02:14:03+0100`, `07:36:58+0100`) is in `a3-pilot/run.out` or `a3-pilot-resume/run.out`.
  The block still parses as the pool's vocabulary, which is what the test needs. B3.2 rewrites the
  first sentence only, so `:33-37`'s remainder and `A3_OUTPUT` are byte for byte; **no issue
  opened**, as the brief directs. The review decides what it is.
- **The brief's `_type`/`_quality` lines** are one early (`:573-574`); measured `:574`, `:575`.
- **Prose outside `docs/` and `prompts/` naming `RayTools`:** none remains. Inside them it is
  untouched, as the prompt says: `docs/tolerance-convergence/inventory.py:382` (`"RayTools"` in
  `PRODUCTION_TREES`, C10) and the dated `docs/` records (the checklist's item 9 lists 39 lines in
  10 files).
- **`main.py`'s line numbers moved since the checklist** (`:1420`, `:2115`, `:3229` are now `:1427`,
  `:2122`, `:3235`), because of 04's rewrite and earlier commits. Nothing depends on them.

## State handed to the next prompt

- **The commit:** this commit. Its code is the tree measured at scratch commit `5624f4c`.
- **The venv**, unchanged by 07. `RunRegistry.package_provenance()`:
  ```json
  {"datastorekit": {"version": "0.2.2", "url": "https://github.com/ds283/DatastoreKit", "commit": "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "requested_revision": "v0.2.2", "editable": false}, "raypoolkit": {"version": "0.2.0", "url": "https://github.com/ds283/RayPoolKit", "commit": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "requested_revision": "v0.2.0", "editable": false}}
  ```
- **What R2's launches should expect to differ from R1's:**
  - Every manifest's `packages` holds both records above (R1's held `raypoolkit: null`).
  - The pool is the package's `v0.2.0`, not SGK's `RayTools` copy (`v0.1.0` plus its fixes).
    Constructor checks raise `ValueError` rather than warn, and no site is affected.
  - **`main.py:3247`**, the `APPLY GKSOURCE POLICIES` queue: its `post_handler` now gets the object
    the persist returned, not the local copy. By reading the statistics are the same; 08 measures it
    from the `** GkSourcePolicyData SUMMARY STATISTICS` block (`main.py:3265`). The existing
    `[02-the-policy-statistics-block-orders-its-lines-by-completion]` still applies to L4.
  - The printed lines (banner `:206`, progress `:588` and `:592`, completion `:606`, and the `Queue
    summary` line `:609`) are the same strings as the copy's, by `ast` (checklist item 5), and
    `test_pipeline_adoption` passes on them. Their places in the file moved by 40 lines
    (`:166`/`:548`/`:566` to `:206`/`:588`/`:606`), which only matters to prose.
  - No log line should differ otherwise; the registry's `StageTracker` fields are the run-level test.
- **Rollback (README §4.2):** `git revert` of this commit alone. An unimported package is inert.
- **Registry:** 25 runs, none `running`. 07 launched nothing and started no Ray process.
- **`docs/OPEN_ISSUES.md`:** 108, unchanged.
