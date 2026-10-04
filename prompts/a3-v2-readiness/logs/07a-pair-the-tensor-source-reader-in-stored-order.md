# Log — prompt 07a: pair the tensor-source reader's k in stored order

**Subject:** *Pair the tensor-source reader's k in stored order* · **Model:** Claude Opus 5.5 (the
prompt recommends Sonnet) · **Date:** 2026-10-04 · **Base:** `e923b3c`, clean · **Result:**
landed. `extract_tensor_source_data.py` pairs its k through `extract_common.quad_source_pairs`,
which orders them by `wavenumber_exit_time_array`, the class `main.py`'s stored pairs come from, so
it asks for q ≤ r only (E1). A labels-only `QuadSource` miss is refused by
`extract_common.require_quad_source`, naming q and r (E2). Run three times on prompt 07's resumed
store, read-only, the reader exited 0 each time and read all six pairs, and the store stayed
byte-identical (E3). None of the prompt's §5 stop conditions fired. Closes
`[07-the-tensor-source-reader-pairs-k-in-random-order-and-crashes-on-q-gt-r]`; opens
`[07a-the-quadsourceintegral-extractor-reads-a-quadsource-miss-unchecked]` (§6 of the prompt, confirmed
by reading). Index 108 → 108. The commit is this prompt's, parent `e923b3c`.

## What shipped

- **`extract_common.py`**, a new section after `read_models`:
  - **`quad_source_pairs(k_exits)`** returns
    `list(itertools.combinations_with_replacement(wavenumber_exit_time_array(k_exits), 2))`. Its
    docstring says why: a store holds `QuadSource` for q ≤ r only (`main.py`'s
    `--quad-source-queue` pairs a `wavenumber_exit_time_array`), a labels-only lookup of q > r
    matches nothing, and ordering through the same class makes the reader's order and the build's
    one order by construction.
  - **`require_quad_source(source)`** returns if `source.available`; otherwise raises
    `RuntimeError`. The message, for a miss with q = 3.1623e5 /Mpc (exit time 13) and r = 1e5 /Mpc
    (exit time 12):
    ```
    extract: no stored QuadSource matched q = 316227.76601683797 /Mpc (wavenumber_exit_time store_id 13), r = 100000.0 /Mpc (wavenumber_exit_time store_id 12) under this run's tags. A store holds QuadSource rows for q <= r only.
    ```
    Its docstring says a miss is an error, not a skip.
  - Imports added: `itertools`, `Iterable` and `Tuple` from `typing`, and `wavenumber_exit_time_array`
    from `CosmologyConcepts.wavenumber`.
- **`extract_tensor_source_data.py`**, three changes and nothing else:
  - `run_pipeline`: `work_grid = quad_source_pairs(k_subsample)`, with a two-line comment, in
    place of `list(itertools.combinations_with_replacement(k_subsample, 2))`. The `sample` draw is
    unchanged.
  - `plot_tensor_source`: `require_quad_source(source)` after `sns.set_theme()`, before the first
    read of `source.values`, with a one-line comment.
  - The two helpers imported from `extract_common`; `import itertools` removed, since nothing else
    in the script used it.
  - `build_tensor_source_work`, the one `object_get("QuadSource", …, labels_only=True, …)`, the
    `ShardedPool(… read_only=True …)`, `--models` and the run selection are untouched.
- **`ComputeTargets/tests/test_tensor_source_pairs.py`**, new, 12 tests, no Ray, no datastore,
  nothing under `var/`:
  - `TestQuadSourcePairs` (E1, run): every permutation of three and of four k gives one list,
    equal to `list(combinations_with_replacement(wavenumber_exit_time_array(items), 2))`; every pair
    has q ≤ r and there are n(n+1)/2; the pairs of three k reversed are the six ascending pairs, as
    literals; a duplicate (the same object twice, and a second object for the same `store_id`)
    appears once; a generator works.
  - `TestRequireQuadSource` (E2, run): a labels-only miss built as
    `test_quadsource_parent_main.test_a_labels_only_miss_names_neither_and_cannot_be_computed`
    builds one raises `RuntimeError` holding both `k` values, both `store_id`s and "no stored
    QuadSource"; a diagonal miss names its k; a stored `QuadSource` (`available` true) raises
    nothing.
  - `TestTheReader` (read with `ast`): `run_pipeline` assigns `work_grid` once, from
    `quad_source_pairs(k_subsample)`, and calls no `combinations_with_replacement`;
    `plot_tensor_source`'s first `require_quad_source(source)` precedes its first `source.values`
    or `source.functions` by (line, column); both helpers are imported from `extract_common`.
  - **The stand-ins (the prompt's choice).** The exit times are **real** `wavenumber_exit_time`s,
    built from a payload as the factory builds them, with real `wavenumber`s in `Mpc_units`, so they
    order by `k` and hash and compare by `store_id` exactly as the store's do. Only the cosmology is
    a stand-in: a class carrying the `units` that `check_units` compares. The k are chosen out of
    `store_id` order (11 → 1e6, 12 → 1e5, 13 → 3.16e5, 14 → 2e5), so that an order by `store_id`
    cannot pass for an order by k.
- **`docs/a3-v2-readiness-verification.md`**: a new §2.3.4 only (E3). §2.3.2 is unchanged.
- **`docs/a3-v2-readiness/e3/`**, new: the two scripts E3 ran (`e3_run.sh`, `snapshot.sh`, run
  from `var/a3-v2-readiness-07a/`), each run's log (ANSI codes stripped) and its before/after
  state, the final snapshot, the Ray listing afterwards, the fingerprint summaries before and after,
  and the list of files the runs wrote.
- **The board**: header, §1 row, E1–E3, the issue moved to §4 with its closing note, the new §3.2
  issue, §5.18. **`docs/OPEN_ISSUES.md`**: the closed row replaced by the new one, the §1.17 prose
  extended; 108 rows by count, as the header already said.

## Deviations from the prompt

1. **The pre-change baseline was measured on an export of `e923b3c`, not in the checkout.**
   `IMPLEMENTATION CHOICE`. It let the suites run while the code was written. Four tests fail on an
   export because each needs a git repository (`git grep`, a 40-character `git_head`); they pass in
   the checkout on the staged tree. §5.18 lists them. The verification run the prompt asks for
   (§3.1, staged tree) was in the checkout.
2. **Breakage (iii) also restores `import itertools`.** `IMPLEMENTATION CHOICE`. The prompt names
   only the `work_grid` line; without the import the mutated script would not run, and the
   mutation should be the old code as it was.
3. **Five tests beyond the prompt's list** (the ascending literals, the generator, the diagonal
   miss, the `store_id`s in the message, the imports). `IMPLEMENTATION CHOICE`. Each pins a stated
   property of E1 or E2.
4. **The board's §1 row for 07a.** `UNINTENDED DRIFT`, corrected before staging. I first added a
   second 07a row beside the one the prompt's writer had put there; I removed it and updated the
   existing row. Its Model cell reads "Sonnet (run by Opus 5.5)".
5. **The board's §1 row and §4 note say "this prompt's commit (parent `e923b3c`)"**, not a SHA,
   which a commit cannot hold of itself. `STRUCTURALLY REQUIRED`.

## Verification performed

### 1. Every existing test passes

The four modules of "Read first" item 8, after the change and before the suites:
`test_quadsource_parent_main`, `test_reader_read_only`, `test_run_identity`, `test_main_plumbing`:
99 tests, OK. None was modified.

The six suites, serially, in the checkout on the staged tree, 10:34–10:44 at a load of 17–25, README
§7's command, each to a file and its verdict grepped (`run_suites.sh` in the scratch directory):

| Suite | Staged tree (07a) | Export of `e923b3c`, 10:23–10:31, load 9–16 | §5.17 |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 32 |
| ComputeTargets | 698 run; 1 failure, `test_tk_wkb_phase.TestCost.test_wall_time_per_object` (0.077 s against 0.06 s) | 686 OK (1 skipped) | 686 |
| CosmologyModels | 39 OK | 39 OK | 39 |
| Datastore | 598 OK | 598, 2 failures (no git repository) | 598 |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 148 |
| RunRegistry | 281 OK | 281, 2 failures (no git repository) | 281 |

698 = 686 + 12. `test_wall_time_per_object` re-run alone at a load of 17: failed in the checkout
(0.103 s) and on the export of `e923b3c` (0.080 s). It is the known load flake, and fails without
this change.

### 2. Deliberate breakage

Below.

### 3. E3 — the reader on 07's resumed store, three times

Verbatim in verification §2.3.4, with the hashes and each run's summary lines. In short:

- **Before** (10:23–10:31): `RunRegistry list` 16 runs, none `running`. No process naming
  `SecondaryGWKit/venv`; every Ray process the ChamPBH cluster's (`gcs_server` 62525, `raylet`
  62574, started 2026-10-02 23:27–23:28, and its agents and four idle workers). `RAY_ADDRESS`
  unset. `store show`: the registry sidecar, no `retired`. `store fingerprint`: digest
  `39809dca…`, "matches the fingerprint taken 2026-10-04T02:09:15+01:00 by run
  a3-v2-readiness-07-resumed-c-20261004T020421".
- **Each run** (10:31:37, 10:32:18, 10:32:47): exit 0 in 25, 27, 26 s; `reading run "default"`;
  `>> RUNNING PIPELINE FOR MODEL LambdaCDM`; `QUERY K_EXIT VALUES` 3 lookup; `EXTRACT TENSOR SOURCE
  DATA` 6 lookup, "no compute items were evaluated"; 6 csv, and plots `full` 6, `parts` 6,
  `q-zoom` 6, `r-zoom` 3 (drawn only for q ≠ r); no `Traceback`, `ReadOnlyMiss`, `ReadOnlyWrite`
  or `RuntimeError`. The three csv directories are identical, and equal to 07's sorted-draw
  wrapper's.
- **Around each run**, `shasum -a 256` and `/usr/bin/stat -f '%m %z'` of all five files: identical
  before and after every run, no `-journal`, `-wal` or `-shm`.
- **After the last run** (10:33:32), the listing:
  ```
  processes naming SecondaryGWKit/venv:
  every ray process (gcs_server, raylet, ray::, ray/ in command):
  62525     1 Fri Oct  2 23:27:58 2026     /Users/ds283/Documents/Code/ChamPBH/venv/lib/python3.13/site-packages/ray/core/src/ray/gcs/gcs_server --log_dir=/tmp/ray/session_
  62529     1 Fri Oct  2 23:27:59 2026     /opt/local/Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python -u /Users/ds283/Documents
  62531     1 Fri Oct  2 23:27:59 2026     /opt/local/Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python /Users/ds283/Documents/Co
  62573     1 Fri Oct  2 23:28:01 2026     /opt/local/Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python -u /Users/ds283/Documents
  62574     1 Fri Oct  2 23:28:01 2026     /Users/ds283/Documents/Code/ChamPBH/venv/lib/python3.13/site-packages/ray/core/src/ray/raylet/raylet --raylet_socket_name=/tmp/ra
  62594 62574 Fri Oct  2 23:28:02 2026     /opt/local/Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python -u /Users/ds283/Documents
  62595 62574 Fri Oct  2 23:28:02 2026     /opt/local/Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python -u /Users/ds283/Documents
  96937 62574 Sat Oct  3 04:16:30 2026     ray::IDLE_SpillWorker
  96961 62574 Sat Oct  3 04:16:31 2026     ray::IDLE_SpillWorker
  97272 62574 Sat Oct  3 04:16:40 2026     ray::IDLE_RestoreWorker
  97276 62574 Sat Oct  3 04:16:41 2026     ray::IDLE_RestoreWorker
  ```
  All ChamPBH's, as before. No process of this checkout's venv was alive after any run, so none
  was waited for.
- **`store fingerprint`** afterwards (no `--write`): the same output, line for line, as before.
- **What three passes mean.** The unit tests are the proof that every order is paired correctly.
  The runs show the shipped reader works on a real store. Three passes of the old reader would have
  had probability 1/216; that bounds luck, it does not prove anything. The reader does not print
  its draw, so which orders these runs drew is not recorded.

### Formatting and staging

`black --check` is clean on the three Python files changed or added. Staged by explicit path, after
`git diff --cached --name-only`.

## The deliberate-breakage record

Each diff was taken with `git diff -- <file>` against the index, which holds this commit's version,
so `git apply` replays it at this commit (each checked with `git apply --check`). Each was restored
with `git checkout -- <file>`. Tests run: `ComputeTargets.tests.test_tensor_source_pairs`.

**(i) `quad_source_pairs` pairs its input in the order given.** Caught: 60 failures in 5 tests
(`test_every_order_gives_mains_pairs` ×28 and `test_every_pair_has_q_le_r_and_there_are_n_n_plus_1_over_2`
×28, every non-ascending permutation; `test_a_duplicated_member_appears_once` ×2;
`test_any_iterable`; `test_the_pairs_are_ascending_in_k`).

```diff
diff --git a/extract_common.py b/extract_common.py
index 48881c9..7063f09 100644
--- a/extract_common.py
+++ b/extract_common.py
@@ -364,7 +364,7 @@ def quad_source_pairs(
     (``random.sample`` returns its draw in random order).
     """
     return list(
-        itertools.combinations_with_replacement(wavenumber_exit_time_array(k_exits), 2)
+        itertools.combinations_with_replacement(k_exits, 2)
     )
 
 
```

**(ii) `quad_source_pairs` orders by descending k.** Caught: 61 failures in 3 tests
(`test_every_order_gives_mains_pairs` ×30, `test_every_pair_has_q_le_r_and_there_are_n_n_plus_1_over_2`
×30, `test_the_pairs_are_ascending_in_k`).

```diff
diff --git a/extract_common.py b/extract_common.py
index 48881c9..0f41bad 100644
--- a/extract_common.py
+++ b/extract_common.py
@@ -364,7 +364,9 @@ def quad_source_pairs(
     (``random.sample`` returns its draw in random order).
     """
     return list(
-        itertools.combinations_with_replacement(wavenumber_exit_time_array(k_exits), 2)
+        itertools.combinations_with_replacement(
+            reversed(list(wavenumber_exit_time_array(k_exits))), 2
+        )
     )
 
 
```

**(iii) `run_pipeline` builds `work_grid` from `combinations_with_replacement(k_subsample, 2)`
again.** Caught: `test_run_pipeline_builds_work_grid_from_quad_source_pairs`,
`test_run_pipeline_calls_no_combinations_with_replacement`.

```diff
diff --git a/extract_tensor_source_data.py b/extract_tensor_source_data.py
index 2f0e1a5..23f10fc 100644
--- a/extract_tensor_source_data.py
+++ b/extract_tensor_source_data.py
@@ -1,4 +1,5 @@
 import argparse
+import itertools
 import sys
 from datetime import datetime
 from math import fabs
@@ -393,7 +394,7 @@ def run_pipeline(model_data):
 
     # the store holds QuadSource for q <= r only, and sample() returns k_subsample in random
     # order: pair it in the stored order (prompts/a3-v2-readiness, prompt 07a)
-    work_grid = quad_source_pairs(k_subsample)
+    work_grid = list(itertools.combinations_with_replacement(k_subsample, 2))
 
     work_queue = RayWorkPool(
         pool,
```

**(iv) `plot_tensor_source` reads `source.values` before calling `require_quad_source`.** Caught:
`test_plot_tensor_source_requires_the_source_before_reading_it`.

```diff
diff --git a/extract_tensor_source_data.py b/extract_tensor_source_data.py
index 2f0e1a5..71af848 100644
--- a/extract_tensor_source_data.py
+++ b/extract_tensor_source_data.py
@@ -133,10 +133,10 @@ def plot_tensor_source(model_label: str, source: QuadSource):
 
     sns.set_theme()
 
+    values: List[QuadSourceValue] = source.values
+
     # a labels-only miss is an unpopulated QuadSource: refuse it, naming q and r
     require_quad_source(source)
-
-    values: List[QuadSourceValue] = source.values
     functions: QuadSourceFunctions = source.functions
 
     abs_source_points = [(value.z.z, safe_fabs(value.source)) for value in values]
```

**(v) `require_quad_source` returns silently when `source.available` is false.** Caught:
`test_a_miss_names_both_k_and_says_no_stored_quadsource`, `test_a_miss_on_the_diagonal_names_its_k`.

```diff
diff --git a/extract_common.py b/extract_common.py
index 48881c9..8e2dfdc 100644
--- a/extract_common.py
+++ b/extract_common.py
@@ -378,7 +378,7 @@ def require_quad_source(source) -> None:
     reader asks only for pairs of k the store itself holds, so a miss means the store lacks a row
     it should have.
     """
-    if source.available:
+    if source.available or not source.available:
         return
 
     q_exit: wavenumber_exit_time = source._q_exit
```

## Observations not acted on

1. **The `QuadSourceIntegral` extractor reads a `QuadSource` miss unchecked** (the prompt's §6,
   confirmed by reading `extract_QuadSourceIntegral_data.py`). It looks `QuadSource` up
   labels-only with `z_sample=None` (line 1237) and hands it to `plot_QuadSourceIntegrand` (line
   1251) and to its own `plot_tensor_source` (line 1264). `plot_QuadSourceIntegrand` checks
   `GkPolicy.available` and `Gk.available`, then reads `source.z_sample.max.z` (line 365); the
   miss's `z_sample` is `None`, so that is an `AttributeError` naming no pair.
   `plot_tensor_source` (line 853) reads `source.values` with no check. Its pairs are in stored
   order, so only a row truly missing from a store reaches either. Opened as
   `[07a-the-quadsourceintegral-extractor-reads-a-quadsource-miss-unchecked]`. Not fixed: §6.
2. **README §2's status cell for 07a still reads "written".** README is not on §7's list of files
   this prompt may change. Left for the orchestrator's review, as 07's was.
3. **An export run without `git init` fails four tests** (two in Datastore, two in RunRegistry),
   which need a git repository. §5.17's export was made into a repository and did not see them.
   Not a defect; a note for whoever measures on an export.
4. **The reader does not print the order `sample` drew**, so E3 cannot say which draws it
   exercised. Not a defect; the unit test covers every order.

## State handed to the next prompt

- No further prompt. The campaign's work is done; it is closed by the user after the orchestrator's
  review of 07a.
- 07's two stores are unchanged and keep their sidecars (D7-2); retiring them is the user's.
  `var/a3-v2-readiness-07a/` holds E3's outputs, logs and scripts.
- Index: 108 open.
- Baselines: AdaptiveLevin 32, ComputeTargets 698, CosmologyModels 39, Datastore 598,
  LiouvilleGreen 148 (1 skipped), RunRegistry 281.
