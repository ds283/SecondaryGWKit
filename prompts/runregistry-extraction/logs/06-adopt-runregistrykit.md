# Log — runregistry-extraction prompt 06, adopt `runregistrykit`

- **Subject:** [`06-adopt-runregistrykit.md`](../06-adopt-runregistrykit.md), board items A1–A5.
- **Commit:** this commit (parent `85f273d`). **Probe:** `ec781383730ff477c63430f0ea8fc10e106e0efc`, tree
  `f2e20c2ad7ff0165d69dbc3051066b7a9d42a30b`, parent `85f273d`, named by no branch (§7.2).
- **Result:** landed. `runregistrykit` is the registry; `RunRegistry/` is gone from git and from disk. All seven roots pass at
  §7.5's counts with no failure. Acceptance 1–8 hold; one breakage (4) fails its named test at a different assertion
  than the prompt states (deviation 9). No stop condition of §10 fired. Closes three issues; opens none.
- **Model:** Opus 5.5.
- **Date:** 2026-10-15.

The orchestrator's brief for this prompt (its §0) was read and is cited as "the brief"; its nine rulings are cited as
"ruling N".

## What shipped

- **A1, the client's two files.**
  - `runregistrykit.toml`: the checklist's item 3, byte for byte, under one comment line naming RunRegistryKit's README.
  - `config/fingerprint.py`: a module docstring in `config/datastore.py`'s register; the format history of
    `RunRegistry/stores.py:194-205` at `a45aefc`, verbatim (`diff` of the two 12-line blocks is empty); one dated
    paragraph (C17); `FINGERPRINT_FORMAT = 6`; `fingerprinter = DatastoreKitFingerprinter(factories,
    FINGERPRINT_FORMAT)`, with the class imported by name into the module (the brief's note on breakage 4). Room is left
    after the format for 07's pin; nothing is added for it.
- **A2, the rewrites in live code** (§2), by file:

  | File | What changed |
  |---|---|
  | `ObjectFactories/tests/test_quadsource_policy_key.py` | `:411` `from RunRegistry.stores import fingerprint_of` → `from datastorekit.store_fingerprint import fingerprint_of` and `from config.fingerprint import FINGERPRINT_FORMAT`; the three calls (`:415`, `:418`, `:421` now) pass `FINGERPRINT_FORMAT` second |
  | `docs/a3-v2-readiness/fingerprint_time_probe.py` | docstring `RunRegistry.stores.fingerprint_store`, `RunRegistry.begin` → `runregistrykit.…`; the import → `runregistrykit.stores.fingerprint_store` and `config.fingerprint.FINGERPRINT_FORMAT`; the timing label `import runregistrykit.stores` (deviation 6) |
  | `docs/a3-v2-readiness/sigterm_handler_probe.py` | docstring `RunRegistry.begin`; `import runregistrykit`, `import runregistrykit.stores as registry_stores`; `runregistrykit.begin(`. The patch by assignment at `:210` is left (observation 1) |
  | `docs/a3-v2-readiness/interrupter_build_dummy_test.sh` | `:6`'s comment, `runregistrykit.begin` |
  | `docs/datastore-integrity-verification/compare_stores.py` | the import: `compare_fingerprints` from `datastorekit.store_fingerprint`, `fingerprint_store` from `runregistrykit.stores`, `FINGERPRINT_FORMAT` from `config.fingerprint` |
  | `docs/datastore-integrity-verification/interrupt_when.py` | every `RunRegistry` → `runregistrykit` (12 lines: the import, `now_iso`, `read_json` ×3, `TERMINAL_STATES`, `pid_alive` ×2, `liveness`, `DEFAULT_STALE_AFTER`, two docstrings) |
  | `docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py` | as `compare_stores.py`, and its docstring |
  | `docs/dk-and-rpk-migration/rehearsal/stage_counts.py` | `import runregistrykit`; `RunRegistry._slug` → `runregistrykit.slugify` (`:37`, `:38`); the docstring's `runregistrykit.run_id` |
  | `docs/dk-and-rpk-migration/rehearsal/store_soundness.py` | the docstring's `runregistrykit.stores.fingerprint_store` |
  | `docs/gktk-remedial/scoped_pipeline_run.py` | `register(..., runs_root=None)` passing `root=runs_root` (§2.2); `import runregistrykit` and its call; the refusal reworded (the package missing from the venv; install `requirements.txt`), its `try` kept; the printed command `./venv/bin/python -m runregistrykit list`; `--register`'s help; the module docstring's manifest and `:116`'s comment |
  | `docs/handover/quadsource_atol_sweep.py` | `prepare(force=False, runs_root=None)` passing `runs_root=runs_root` to `copy_store` (§2.2); the three imports and their calls; both printed commands; `:419`'s docstring → `RunDrivers/tests/test_prune_on_resume.py`; `prepare`'s docstring and `--force`'s message (`runregistrykit.stores.copy_store`, `runregistrykit store retire`; "`store-retirement` prompt 03" kept) |
  | `docs/runregistry-extraction/fingerprint_both_ways.py` | rewritten whole (§5) |

- **A3, the 101 tests** (§3). Counted by discovery before anything moved: 28, 26, 15, 18, 10, `TestGolden` 3, `TestDrivers` 1.
  - `git mv` of the five modules to `RunDrivers/tests/`, and of the six golden files to `ObjectFactories/tests/data/`
    (all six `R100`). `RunDrivers/__init__.py` and `RunDrivers/tests/__init__.py`, empty.
  - `RunDrivers/tests/registry_fixtures.py` (C14; ruling 3): `RegistryTestCase` (`setUp`, `begin`) from
    `test_run_registry.py:113-126` and `StoreTestCase` (`setUp`, `state`) from `store_fixtures.py:114-124`, `:155-156`,
    at `a45aefc`, and nothing else.
  - `RunDrivers/tests/test_finish_calls.py`: `TestDrivers`, `_HANDLERS`, `_finish_calls`, `DRIVERS`, and its own
    `REPO_ROOT = Path(__file__).resolve().parents[2]`.
  - `ObjectFactories/tests/test_store_fingerprint_golden.py`: `_FullStores`, `without_taken`, `QSI`, `TestGolden` (3) and
    `TestConfigFingerprint` (3, C18).
  - `ObjectFactories/tests/data/full_store_fingerprint_runregistry-extraction-06.json` (C13), captured through
    `config.fingerprint.fingerprinter.take(build_full_store().primary, None)[0]`: `diff` against `09c` is one added line,
    `  "encoding_format": 1,`; digest `6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759`.
- **`RunRegistry/` deleted** (§4): `git rm -r`, then its `__pycache__/` `.pyc` files and empty directories removed;
  `ls RunRegistry` fails and `import RunRegistry` raises `ModuleNotFoundError` (ruling 7).
- **§5:** `docs/runregistry-extraction/fingerprint_both_ways.py <primary>... --against <run dir>... [--expect-digest D]`.
- **A4, `CLAUDE.md`** (§6): the package section names the third package, its pin with `[stores]`, its imports and CLI, and
  "None of the three is this repository's code", with `datastorekit` at `v0.4.0` (ruling 8), and one sentence naming
  `RunRegistry/` as the registry's former home; the registry section's opening rewritten, the six rules unchanged but for
  rule 1's command, `./venv/bin/python -m runregistrykit list`; the freeze section removed; the roots list
  `RunDrivers/tests` in place of `RunRegistry/tests` (still seven). The rehearsal-lock section is unchanged.
- **A5:** `docs/runregistry-extraction/worktree_run_check.py` (§7.4), and its run.
- **`measure/out/`:** `06-before.txt`, `06-after.txt`, `06-roots.txt`, `06-measure.txt`, `06-worktree.txt`,
  `06-both-ways.txt`, and `06-commands.txt` (§7.8).
- **Under `var/`**, exactly two run directories, both records: `runregistry-extraction-06-worktree-check-20261015T224836`
  (`done`/0) and `runregistry-extraction-06-worktree-check-breakage-20261015T225237` (`failed`/1). Nothing else under `var/`
  was written (`find var -newer <marker>`, the marker set before the first command, lists `var/runs` and those two
  directories' files alone).

### The test changes of §3.2 and §3.3, each classified

Every change is one §3.2 or §3.3 names (**STRUCTURALLY REQUIRED**, README §5 rule 7), unless marked otherwise.

- **`test_build_rehearsal_scope.py`:** `:39` `from RunDrivers.tests.test_prune_on_resume import parsed_arguments`; `:317`
  `"runregistrykit.begin"`. Nothing else.
- **`test_killed_handler.py`:** `:26` `import runregistrykit`; `:27` `from RunDrivers.tests.test_pipeline_adoption import …`;
  `:90` `runregistrykit.begin(`; `:186`, `:336`, `:435` `runregistrykit.read_json(`. `:333` and `:432` (the patches of
  `Run._fingerprint_results`) unchanged.
- **`test_pipeline_adoption.py`:** `import runregistrykit`; `begin`, `list_runs`, `read_json` (×3), `TERMINAL_STATES` (which
  `black` rewrapped onto three lines); `:146`, `:158` `"-m", "runregistrykit"`; `:278` `root =
  runregistrykit.find_client_root(client_root=REPO_ROOT).runs_root`; `:330` the patch removed and
  `self.driver.register(..., runs_root=self.root)`; the new assertion, `os.path.dirname(run.path) == self.root`, in
  `test_the_datastore_is_registered_as_results_and_not_as_a_ledger`. **`from unittest import mock` removed**, its one use
  being the patch removed: **IMPLEMENTATION CHOICE** (deviation 3).
- **`test_prune_on_resume.py`:** unchanged (`R100`).
- **`test_quadsource_atol_sweep_prepare.py`:** `from runregistrykit import stores`; `StoreTestCase` from
  `RunDrivers.tests.registry_fixtures`; the `stores.DEFAULT_ROOT` patch removed, and with it `from unittest import mock`;
  the five `prepare(` calls pass `runs_root=self.root` (measured five: `:118`, `:151`, `:168`, `:182`, `:187` at `14bcac0`);
  `:134` expects `os.path.abspath(baseline)` (below); the module and class docstrings say what the test now does; and
  the new test `test_a_running_run_in_the_given_runs_root_blocks_the_copy`. `import os` added for the two.
  - **`:134`'s expectation.** At `v0.1.0`, `copy_store` records `copied_from.datastore = _repo_path(src, client.root)`,
    where `_prepare` has made `src` `Path(os.path.abspath(src))`, and `_repo_path` returns `os.path.abspath(path)` for a
    path outside the record root. The baseline is in a resolved temporary directory, so the test states
    `os.path.abspath(baseline)` and calls no private function. **IMPLEMENTATION CHOICE**, the checklist's advice.
  - **The new test** begins a run in `self.root` through `registry_fixtures.RegistryTestCase.begin`, with `results` the
    baseline, asserts the run directory's parent is `self.root`, takes `self.state()`, then requires `prepare(runs_root=
    self.root)` to raise naming the run (`run <id> is running`, `names this store`), the tree unchanged and no sweep store.
- **`TestDrivers`:** nothing but its module's `REPO_ROOT`.
- **`TestGolden`:** `GOLDEN` names the new golden file; `test_the_golden_fingerprint` also asserts `golden["encoding_format"]
  == ENCODING_FORMAT`; `test_the_shape`'s key set gains `"encoding_format"` and its `_fingerprint_problems` call is
  `runregistrykit.stores._fingerprint_problems(…)`; `test_taken_is_outside_every_digest` calls `fingerprint_of(
  self.baseline_inventory, FINGERPRINT_FORMAT, taken)`, with `compare_fingerprints` from DatastoreKit. `_FullStores`'s two
  `fingerprint_of` calls pass `FINGERPRINT_FORMAT`. **`_FullStores` gains `maxDiff = None`** so that a failure prints the
  differing class rather than unittest's truncated diff: **STRUCTURALLY REQUIRED** by §8 breakage 3's "naming the class"
  (deviation 4).
- **`TestConfigFingerprint`**, new: the three tests §3.3 names, on `_FullStores` (for its `store()`).

## Deviations from the prompt

1. **§7.6 prints two lines, not one** (ruling 1): `CLAUDE.md:82`, the one sentence, and `config/fingerprint.py:31`, the
   history's own `RunRegistry/tests/data/full_store_fingerprint*.json`, moved verbatim. **STRUCTURALLY REQUIRED**, by
   ruling 1. The dated paragraph and `registry_fixtures.py`'s docstring name the old package without the literal
   ("the registry package's `stores.py`", "its `tests/test_run_registry.py`"), so that nothing else matches:
   **IMPLEMENTATION CHOICE**.
2. **The golden file is written with sorted keys**, `json.dumps(indent=2, sort_keys=True) + "\n"`, not "key order as
   `fingerprint_of` returns it". `09c` itself was written sorted (its capture, recorded in `014be95`'s message, and its
   keys and classes are in sorted order), so §3.4's other requirement, the two files' bytes equal but for the added key,
   is met only this way: `diff` is the one line `"encoding_format": 1,`. **STRUCTURALLY REQUIRED** (§3.4 contradicts itself;
   the byte comparison wins). The tests compare parsed JSON, so nothing depends on it.
3. **`from unittest import mock` removed** from `test_pipeline_adoption.py` and `test_quadsource_atol_sweep_prepare.py`,
   where the only use was the patch §3.2 removes; `import os` added to the second for `os.path.abspath` and
   `os.path.dirname`. **IMPLEMENTATION CHOICE.**
4. **`_FullStores.maxDiff = None`**, above. **STRUCTURALLY REQUIRED** (§8 breakage 3).
5. **`registry_fixtures.RegistryTestCase.begin` passes `client_root=REPO_ROOT` by default**, so that its runs find
   this repository's config whatever the working directory (§3.2: "with `script` or `client_root` such that it finds SGK's
   config"); the frozen copy had no such keyword. **IMPLEMENTATION CHOICE.**
6. **`fingerprint_time_probe.py` imports `FINGERPRINT_FORMAT` after printing the import's time**, so that the label
   `import runregistrykit.stores: … s` times that import alone and not `config.datastore` too. **IMPLEMENTATION CHOICE.**
7. **The two seams' docstrings gain one sentence each** saying what `runs_root` is and that `None` is the config's runs
   root. **STRUCTURALLY REQUIRED**, with the signature changes (§2.2), which are themselves **STRUCTURALLY REQUIRED**.
8. **`fingerprint_both_ways.py` compares key order only where a record keeps it.** §5 asks for "the same keys in the same
   order … and for every class in the same order". Every record is written with `sort_keys=True` (`write_json_atomic`,
   `runregistrykit/__init__.py:247`, and `RunRegistry/__init__.py:203` at `a45aefc`): a recorded fingerprint's keys and
   classes are in sorted order, a fresh one's classes in registry order (`version`, `store_tag`, … against
   `BackgroundModel`, …). An order check across the two would fail on every pair, so the script checks the same keys at
   every level, every value, and the order of each `tag_sets` list (which a record keeps), and its docstring says why.
   **STRUCTURALLY REQUIRED.** 04's form compared order between two in-memory fingerprints, where it meant something.
9. **Breakage 4 fails `TestConfigFingerprint` test 1 at its first assertion, the config's string, not with the loader's
   refusal.** Test 1 asserts `client.stores == {"fingerprinter": "config.fingerprint:fingerprinter"}` before it calls
   `load_fingerprinter`, so the mutated config fails it there (`{'fingerprinter':
   'config.fingerprint:DatastoreKitFingerprinter'} != …`). The named test fails, as §8 requires, but not at the line §8
   names. The loader's refusal itself was shown directly on the same mutated config: `load_fingerprinter` raises
   `ConfigError: … is the class DatastoreKitFingerprinter, not an instance of it …` (exit 1), and `store fingerprint` on the
   scratch store prints it and exits 1. **UNINTENDED DRIFT.** It was found after the probe and the worktree run; reordering
   the test would change the tree they ran at, and the worktree run cannot be retaken without a third run under `var/runs/`.
   Not a stop by ruling 4 (the named test fails); the orchestrator may want test 1 reordered in a follow-up.
10. **Git pairs `RunRegistry/tests/__init__.py` with `RunDrivers/__init__.py`** (both empty) as `R100`, so
    `--name-status` shows 13 `D` lines and 12 `R` lines, not 14 and 11; the pair is a deletion and an addition. Not a
    deviation in the work; noted for the review.
11. **`06-commands.txt`** is an extra `measure/out/06-*` file, holding §7.8's runs. **IMPLEMENTATION CHOICE.**
12. **No other deviation.** Nothing outside §12's list was changed. `README.md`, `prompts/INDEX.md`, `requirements.txt`,
    the venv and `orchestrator/` were not edited; the orchestrator's directory was not read. RunRegistryKit, DatastoreKit
    and RayPoolKit were read only (RunRegistryKit's checkout is at `caad864`, clean, before and after).

## Verification performed

- **The gate.** `HEAD` `85f273d`, clean; one worktree; `python -m RunRegistry list` 37 runs (20 done, 13 killed, 2 failed,
  2 unknown), none `running`; 18 sidecars, all tombstones, no `.sqlite` under `var/datastores/`. No `git log` movement
  during the work (checked before measuring and before the commit).
- **§7.1, before and after** (`06-before.txt`, `06-after.txt`). "Before" was the first command of the prompt, with
  `RunRegistry/` present. "After" at the staged tree, before the probe's runs existed. Both: `list` exit 0 and 18 `store
  show` blocks, each exit 0.
  - With the command lines, the header and the `AGE` column removed, the two files are **identical** (`diff` empty): every
    run's id, state, progress, purpose and stage, and every `store show` block byte for byte, "runs naming this store"
    (by results and by `store_id`) included. The only lines that differ are the 19 command lines (the tool's name), the
    header, and 5 `AGE` values that rolled over between the readings.
  - The 37 runs are 01's 29 and R0's and R1's 8. Each of 01's 29 has the same id, state and purpose-and-stage text as
    `01-list.txt`.
  - Every one of the 18 blocks is `kind: registry`, `problems: none`, with a `retirement` block `state: retired`.
- **§7.5, the roots** (`06-roots.txt`), at the staged tree, serially, load 8–13: AdaptiveLevin 32 OK, ComputeTargets 693
  OK, CosmologyModels 39 OK, LiouvilleGreen 148 OK (skipped=1), **ObjectFactories 277** OK, **RunDrivers 99** OK, tools 91
  OK. No failure, so no known flake needed a re-run. `RunRegistry/tests` is gone.
- **`black --check`** on every staged `.py`: 23 files unchanged. (`black --check .` reports 49 files, all under
  `prompts/datastore-generic/`; observation 6.)
- **`config.fingerprint` imports nothing heavy of its own:** after `import config.datastore`, importing it adds
  `config.fingerprint`, `runregistrykit`, `runregistrykit.client`, `runregistrykit.fingerprint` and `tomllib`'s four
  modules, and nothing of `ray`, `sqlalchemy` or `datastorekit` that `config.datastore` had not loaded.
- **§7.2, the probe.** `git write-tree` = `f2e20c2ad7ff0165d69dbc3051066b7a9d42a30b`; `git commit-tree "$tree" -p HEAD`
  = `ec781383730ff477c63430f0ea8fc10e106e0efc`; `git branch --contains` names nothing.
- **§7.3, the measure** (`06-measure.txt`): RunRegistryKit's `docs/adoption/measure_client_registry.py sgk --commit
  ec781383…`, from its root with its venv, exit 0. 421 tracked `.py` files parsed. **Import statements 0** in live code
  and 0 in the staying tests; names used 0; calls bound 0; string literals naming it 0; `-m RunRegistry` 0. 48 name-alike
  calls, none the registry's (the `runregistrykit.begin`, DatastoreKit `fingerprint_of`, `engine.begin` and other
  `copy_store` calls). Campaign records: 6 `.py` files import it, 7 name it, all under `prompts/`.
- **§7.4, the worktree check** (`06-worktree.txt`). `git worktree add --detach <scratch>/sgk-06-worktree ec781383…`; from
  inside it, the main checkout's venv by absolute path ran the script: exit 0, 4 of 4 checks. The run
  `runregistry-extraction-06-worktree-check-20261015T224836` lies in `/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/`;
  its manifest's `git_head` is the probe, `ec781383730ff477c63430f0ea8fc10e106e0efc`, with `git_dirty` **false**, while
  the main checkout's `HEAD` was `85f273d` and dirty (51 `git status` lines); its `script` and `cwd` are the worktree's
  paths; `packages` names `datastorekit` 0.4.0, `raypoolkit` 0.2.0 and `runregistrykit` 0.1.0 (`9f8ebe43…`, `v0.1.0`),
  none editable. `git worktree remove` (no `--force`) exit 0, `git worktree prune`, `git worktree list` the main checkout
  alone, the directory gone; then `./venv/bin/python -m runregistrykit list` shows the run `done`, 38 runs.
- **§7.6** at the staged tree:
  ```text
  $ git grep -nP 'RunRegistry(?!Kit)' -- ':!prompts' ':!docs/*.md' ':!docs/*.txt'
  CLAUDE.md:82:2026-10-15 the registry was this repository's own package `RunRegistry/`; campaign records from
  config/fingerprint.py:31:# (`RunRegistry/tests/data/full_store_fingerprint*.json`; `test_store_fingerprint.GOLDEN` names the
  ```
  and `PYTHONPATH=. ./venv/bin/python -c 'import RunRegistry'` gives `ModuleNotFoundError: No module named 'RunRegistry'`.
- **§5's script** (`06-both-ways.txt`): on the scratch set (a `build_full_store()` store with a sidecar from `store create`
  and a fingerprint from `store fingerprint --write`, both exit 0; a scratch run directory holding the `09c` golden),
  `--expect-digest 6576b85d…`: **exit 0**, 8 of 8 checks. `compare_fingerprints(recorded, current)` is one `encoding`
  entry (recorded `None`, current `1`); the sidecar compares empty. Every file of the set has the same SHA-256, size and
  `mtime_ns` before and after.
- **Nothing stored changes.** The new golden's digest is `09c`'s; format 6; `TestConfigFingerprint` test 2 shows the
  fingerprinter's fingerprint equals `fingerprint_of(read_inventory(primary, factories), 6)` and the golden with `taken`
  aside (acceptance 5).
- **After §8**, `git diff` (working tree against the index) is empty and `git write-tree` is
  `f2e20c2ad7ff0165d69dbc3051066b7a9d42a30b`, the probe's tree. `git worktree list` shows the main checkout alone.

### §7.8: the commands for README §4.1, from now on (for 08)

Each was run once, read-only, on a tombstone, the scratch store or R1's run directories; `06-commands.txt` holds the output.
- **Discovery:** `./venv/bin/python -m runregistrykit list` (`--root DIR`, `--stale-after S`; `--client-root DIR` before
  the command). Exit 0.
- **Stores:** `./venv/bin/python -m runregistrykit store show | create --purpose TEXT | fingerprint [--write] [--listing
  PATH] <primary>`. `store show` on all 18 tombstones exit 0 (§7.1); `store fingerprint` on a tombstone refuses, exit 1
  ("is a tombstone … Nothing was read or written"); `create` and `fingerprint --write` on the scratch store exit 0, and a
  later `store fingerprint` there gives `recorded: matches`. Every `store` command loads the fingerprinter from
  `runregistrykit.toml`.
- **Provenance:** `runregistrykit.package_provenance(names)` takes the names, e.g.
  `runregistrykit.package_provenance(["runregistrykit", "datastorekit", "raypoolkit"])`; it gave each at its tag, none
  editable.
- **The rehearsal tooling** under `docs/dk-and-rpk-migration/rehearsal/`, as rewritten:
  - `PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py <primary>... --against
    <run dir>...` — run on the scratch store against R1's `reference` run directory: exit 1, as it must be, with "against
    its sidecar: 0 difference(s)" and one `encoding` difference against the recorded fingerprint (README §4: at 08 it is run
    on R2's own stores and sidecars only). Run first against the scratch `09c` run directory, it raised `AttributeError`
    on that record's `taken: null` (observation 3).
  - `PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/stage_counts.py <run dir>...` — on R1's four run
    directories: exit 0, the table byte-identical to `docs/runregistry-extraction/rehearsal/stage_counts_04.txt`
    (`runregistrykit.slugify` labels the columns as `_slug` did).
  - `PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/store_soundness.py <primary>...` — on the scratch
    store: exit 0.
  - `policy_statistics.py` and `prune_check.py` are unchanged and import no registry; **`run_readers.sh` was not run**: it
    runs the readers, which start Ray.
- **The interrupter**, `docs/datastore-integrity-verification/interrupt_when.py`, through its two dummy tests with work
  directories in the scratch directory: `interrupter_dummy_test.sh` and `interrupter_build_dummy_test.sh`, each exit 0,
  every case as its header states; no process of theirs left.
- **`fingerprint_both_ways.py`**, for 08: `./venv/bin/python docs/runregistry-extraction/fingerprint_both_ways.py
  <R2 primary>... --against <R0 and R1 reference and resumed-c run dirs>...`, `--expect-digest` left at its default,
  `39809dca…`.

## The deliberate-breakage record

Each was applied as shown, run, and reverted (`git checkout -- <file>` from the index, or in breakage 7b a scratch copy of
the data); none was committed. Every test named by §8 fails; per ruling 4 every failing test is listed.

1. **`FINGERPRINT_FORMAT = 5`.** ObjectFactories root: exit 1, `FAILED (failures=3)`, 277 run.
   ```diff
   -FINGERPRINT_FORMAT = 6
   +FINGERPRINT_FORMAT = 5
   ```
   - `TestConfigFingerprint.test_the_config_names_the_fingerprinter`: `AssertionError: 5 != 6`;
   - `TestConfigFingerprint.test_the_fingerprinter_takes_the_golden_fingerprint`: `{'fingerprint_format': 5, … != …`;
   - `TestGolden.test_the_golden_fingerprint`: `AssertionError: 6 != 5`.
2. **The fingerprinter over the registry less `OneLoopIntegral`.** ObjectFactories root: exit 1, `FAILED (failures=1,
   errors=1)`.
   ```diff
   -fingerprinter = DatastoreKitFingerprinter(factories, FINGERPRINT_FORMAT)
   +fingerprinter = DatastoreKitFingerprinter(
   +    {k: v for k, v in factories.items() if k != "OneLoopIntegral"}, FINGERPRINT_FORMAT
   +)
   ```
   - `TestConfigFingerprint.test_the_fingerprinter_takes_the_golden_fingerprint`: **ERROR**,
     `datastorekit.SQL.schema.StoreSchemaMismatch: Cannot read sharded datastore "…/full-store.sqlite": shard #0 … differs
     from …` — the reader refuses a store whose schema the reduced registry does not describe, before any fingerprint is
     compared;
   - `TestConfigFingerprint.test_the_config_names_the_fingerprinter`: `AssertionError: {'version': <class …> … is not …`
     (the `factories` identity).
3. **One class's `digest` altered in the new golden file** (`QuadSourceIntegral`, first hex digit). ObjectFactories root:
   exit 1, `FAILED (failures=3)`; each failure's diff names `'QuadSourceIntegral': {'count': 3, - 'digest': 'cc4367bb…'
   + 'digest': '0c4367bb…'`.
   ```diff
        "QuadSourceIntegral": {
          "count": 3,
   -      "digest": "cc4367bbf04820464edb6b3f137609d8d1ea317cf93ab1c84b2a5527af86e725",
   +      "digest": "0c4367bbf04820464edb6b3f137609d8d1ea317cf93ab1c84b2a5527af86e725",
   ```
   - `TestConfigFingerprint.test_the_new_golden_is_the_old_one_with_its_encoding`;
   - `TestGolden.test_the_golden_fingerprint`;
   - `TestConfigFingerprint.test_the_fingerprinter_takes_the_golden_fingerprint`.
4. **`runregistrykit.toml` names the class.** ObjectFactories root: exit 1, `FAILED (failures=1)`.
   ```diff
   -fingerprinter = "config.fingerprint:fingerprinter"
   +fingerprinter = "config.fingerprint:DatastoreKitFingerprinter"
   ```
   - `TestConfigFingerprint.test_the_config_names_the_fingerprinter`: `AssertionError: {'fingerprinter':
     'config.fingerprint:DatastoreKitFingerprinter'} != {'fingerprinter': 'config.fingerprint:fingerprinter'}` — at the
     config assertion, before the loader (deviation 9). On the same mutated config, directly: `load_fingerprinter(
     find_client_root(client_root="."))` exit 1, `runregistrykit.client.ConfigError: …/runregistrykit.toml: [stores]
     fingerprinter 'config.fingerprint:DatastoreKitFingerprinter' is the class DatastoreKitFingerprinter, not an instance
     of it: …`; `./venv/bin/python -m runregistrykit store fingerprint <scratch primary>` exit 1, printing the same.
5. **`scoped_pipeline_run.register` forwards another root**, inside the temporary directory. RunDrivers root: exit 1,
   `FAILED (failures=1)`, 99 run; `var/runs/` unchanged (72 entries before and after).
   ```diff
   -        root=runs_root,
   +        root=os.path.join(runs_root, "elsewhere") if runs_root else None,
   ```
   - `TestTheDatastoreIsNotALedger.test_the_datastore_is_registered_as_results_and_not_as_a_ledger`: `AssertionError:
     '/var/folders/…/tmp7zy2d3e1/elsewhere' != '/var/folders/…/tmp7zy2d3e1'` (the new assertion).
6. **`quadsource_atol_sweep.prepare` drops `runs_root`.** RunDrivers root: exit 1, `FAILED (failures=1)`.
   ```diff
                    "it. The comparator is handover-A3-baseline-lambdacdm, not this."
                ),
   -            runs_root=runs_root,
            )
   ```
   - `SweepPrepareTestCase.test_a_running_run_in_the_given_runs_root_blocks_the_copy`: `AssertionError: RuntimeError not
     raised` (`copy_store` read the real `var/runs/`, read-only, found no running run naming the temporary baseline, and
     copied it inside the temporary directory).
7. **`fingerprint_both_ways.py`**, on the scratch set, `--expect-digest 6576b85d…`:
   - (a) `encoding_format` not set aside: exit 1, 7 of 8 checks, `[FAIL] field by field, …: 1 differing path(s)` /
     `encoding_format: current 1, recorded '<absent>'`.
     ```diff
     -SET_ASIDE = ("encoding_format", "taken")
     +SET_ASIDE = ("taken",)
     ```
   - (b) one tag set's `count` in the recorded fingerprint incremented, in a scratch copy of the run directory
     (`rundir-09c-countplus1`): exit 1, 7 of 8 checks, `classes.QuadSourceIntegral.tag_sets[0] tags [Run_fixture].count:
     current 2, recorded 3`. The package's `compare_fingerprints(recorded, current)` still gave its one `encoding` entry,
     and nothing about the count: the reason the comparison is made field by field.
     ```diff
     @@ -180,7 +180,7 @@
              "digest": "cc4367bbf04820464edb6b3f137609d8d1ea317cf93ab1c84b2a5527af86e725",
              "tag_sets": [
                {
     -            "count": 2,
     +            "count": 3,
                  "digest": "99e793cb6c94b005db5284f8008e8b3878805813c548fd6a2c95623eb15551bd",
     ```
8. **`worktree_run_check.py` expects the worktree's own `var/runs/`**, applied in a second scratch worktree at the probe
   (ruling 6). `--slug worktree-check-breakage`: exit 1, 3 of 4 checks, `[FAIL] the run directory lies directly under
   the main checkout's var/runs/: /Users/ds283/Documents/Code/SecondaryGWKit/var/runs and
   <scratch>/sgk-06-worktree-breakage/var/runs`. Its run, `runregistry-extraction-06-worktree-check-breakage-20261015T225237`,
   is in the main checkout's `var/runs/`, `git_dirty` **true**, and **ended `failed`/1** (ruling 5). The mutation was
   reverted in the worktree (`git -C <wt> checkout -- …`, then `status --short` empty), and the worktree removed without
   `--force` and pruned. The transcript is appended to `06-worktree.txt`.
   ```diff
   -    expected_root = os.path.join(main_checkout, "var", "runs")
   +    expected_root = os.path.join(worktree_top, "var", "runs")
   ```

## Observations not acted on

1. **Private names and patch targets the staying tests and probes still reach**, each present at `v0.1.0` with the same
   parameters; the package promises none of them ("The public names"):
   - `docs/a3-v2-readiness/sigterm_handler_probe.py:210` assigns `registry_stores.fingerprint_store` (§2.1). It still
     takes effect: `Run._fingerprint_results` imports `fingerprint_store` from `runregistrykit.stores` when it runs.
   - `RunDrivers/tests/test_killed_handler.py:333` and `:432` patch `Run._fingerprint_results` on a `Run` (§3.2).
   - `ObjectFactories/tests/test_store_fingerprint_golden.py`'s `test_the_shape` calls
     `runregistrykit.stores._fingerprint_problems` (§3.3).
   A release that moved any of them would fail these tests or silently disarm the probe; a pin prompt would see it.
2. **Breakage 4 and test 1's order** (deviation 9): test 1 does catch a class named in the config, but at its string
   assertion. A reordering, `load_fingerprinter` first, would make the loader's refusal the failure. Not done here (it would
   move the probe's tree).
3. **`compare_with_recorded.py:59` raises `AttributeError` on a recorded fingerprint whose `taken` is `null`**
   (`fp.get("taken", {})` returns `None`). No run directory records one (each run's `taken` is an object); only a
   synthetic record such as the golden files has. Pre-existing, unchanged by this prompt, and outside 08's use.
4. **`quadsource_atol_sweep.py`'s `--force` message** now reads "once `runregistrykit store retire` exists
   (store-retirement prompt 03)", which has long existed. §2.3 rewrote the name and kept the history; the sentence was
   already stale before.
5. **The checklist's measure counts "staying tests" by their old location**, so its staying-tests column is 0 by
   construction once they move; the live-code column, which now covers `RunDrivers/tests/` and
   `ObjectFactories/tests/`, is the one that shows they import nothing of the old package.
6. **`black --check .` is not clean**: 49 files under `prompts/datastore-generic/` (orchestrator measure scripts, campaign
   records) would be reformatted. `CLAUDE.md` says the tree is clean under `--check`. Pre-existing; every file this prompt
   touched is clean.
None is a defect that needs an issue: 1 is the package's stated policy, 2 is this prompt's own test (for the review), 3–6
are minor and pre-existing or cosmetic.

## State handed to the next prompt

- **`runregistrykit` is the registry.** `runregistrykit.toml` at the root and `config/fingerprint.py` (`FINGERPRINT_FORMAT =
  6`, `fingerprinter`) are what it reads; `RunRegistry/` is gone. 07 adds its pin beside `FINGERPRINT_FORMAT`.
- **The roots:** AdaptiveLevin 32, ComputeTargets 693, CosmologyModels 39, LiouvilleGreen 148 (1 skipped), ObjectFactories
  277, **RunDrivers 99**, tools 91.
- **The registry:** 39 runs, 37 finished and 2 `unknown`, none `running`: 01's 29, R0's and R1's 8, and this prompt's two,
  `runregistry-extraction-06-worktree-check-20261015T224836` (`done`) and
  `runregistry-extraction-06-worktree-check-breakage-20261015T225237` (`failed`). 18 sidecars, all tombstones.
- **For 08:** §7.8's commands; `fingerprint_both_ways.py` takes R2's primaries and `--against` R0's and R1's run
  directories, at its default digest. A manifest's `packages` now names `runregistrykit` 0.1.0 beside the two.
- **The probe** `ec781383…` is a commit no branch names; `git diff --stat ec781383 <this commit>` is the records alone.
- **Rollback:** `git revert` of this commit restores `RunRegistry/` and the old tests; nothing in the venv changes (README
  §4.2). The two run directories under `var/runs/` stay as records either way.
