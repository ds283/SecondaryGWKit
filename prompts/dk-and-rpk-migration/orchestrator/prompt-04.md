# Orchestrator — prompt 04, adopt `datastorekit`

Read [`../README.md`](../README.md) first: §0.2, §1, §2.2 (04's charter), §4, §4.2, §4.3, §5, §6
(U3, C2–C6, C10) and §7. Then read [`prompt-01.md`](prompt-01.md) §0's conventions, which carry over
except where §0 below says otherwise, and this board's *Orchestrator review of prompt 03*.

**You do not write code.** You may:
- run DatastoreKit's measuring script from its root, with its venv, at any SGK commit;
- make scratch commit objects with `git commit-tree` from a **temporary index** in the session
  scratchpad, never the checkout's index, and never move a ref;
- run the suites, replay the log's breakage diffs with `git apply`, and revert them;
- read R0's reference store read-only (`store fingerprint` without `--write`, `shard_key_audit`);
- run in-process checks from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`04-adopt-datastorekit.md`](../04-adopt-datastorekit.md)
**Board items:** A1–A5 · **Closes:** nothing · **Narrows:** nothing · **Opens:** A5's
`[04-six-docs-probes-import-factories-from-the-layer]`, and §0's prose issue · **Model:** Opus,
as the prompt recommends.

**Gate:**
- 03 landed (`9ec603a`) and was reviewed (`41d9bc0`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means anything outside `prompts/`.
  - If a later commit touches code, re-measure §0's figures at the new `HEAD` and hand the agent
    the corrections as an addendum.
- `datastorekit` 0.2.2 is installed from the pin at `6d8ee9ae…`, not editable; `pip check` clean.
- Nothing is `running` in the registry, and no Ray process of this repository is up.
- R0's stores exist and are as 02 left them.

## 0. What makes this prompt unusual

**It is the swap.** After it, SGK runs no layer code of its own: every `ShardedPool`, `Datastore`
actor, `read_inventory` and `open_read_only` is the package's. It is wide (88 files rewritten, 42
deleted) and mechanical, and it runs no build. So the review has three halves:
- **the rewrite is exact**: every changed line outside the hand edits is a module-path
  substitution at a line DatastoreKit's script named, and nothing else in those files moves;
- **nothing was missed**: the script, re-run at the commit, finds 0 layer imports, strings and
  `import_module` calls outside `prompts/`, and the stale layer cannot be imported;
- **the package reads what SGK wrote**: R0's reference store fingerprints `39809dca…` through the
  package, and its files do not change.

**Corrections to the prompt.** The orchestrator checked the prompt against the tree at `41d9bc0`.
Five statements in it are wrong or stale. Each correction below governs, and the agent logs each as
an `IMPLEMENTATION CHOICE` citing this note, or `STRUCTURALLY REQUIRED` where the prompt cannot be
followed as written.

1. **The counts** (prompt §0, §2.5, README §0.2). The prompt's "about 180 import statements in some
   100 files" and the README's "17 `docs/` statements in 13 files" are `b510bc9`'s. At `41d9bc0`:
   - 234 statements in 139 files; **18** in `docs/` (14 files), the extra one being
     `docs/dk-and-rpk-migration/rehearsal/store_soundness.py:30` (02's, as its review said);
   - after A1's and A3's deletions, the rewrite has **134 statements** (35 nested), **7
     `import_module` calls** and **0 string literals** to change, in **88 files**, all outside
     `prompts/`. Every statement is a `from` import naming exactly one layer module.
2. **A5 is six statements, not ten** (prompt §2.5 and §2.2's last bullet; DatastoreKit's checklist
   item 3 says ten). Of the ten `_factories` imports at `HEAD`, **six are outside `prompts/`, one
   in each of the six files**; the other four are in
   `prompts/datastore-generic/orchestrator/measure-05/scripts/`, which A2 does not touch. The six
   failures reproduce at `HEAD` (§0's facts). Five fail at import when run with no arguments.
   **`docs/tolerance-convergence/order_audit.py` does not**: its import is nested in `d3_tables()`
   (`:1252`), which `main` reaches only at stage 6, after minutes of computation. For it, load the
   module by path (`importlib.util.spec_from_file_location`) and call `d3_tables()`, which raises
   the `ImportError` at once. Do the same before and after, and never let the full script run.
3. **`shard_key_audit` has no `--help`** (prompt §3.3.1). `main(argv)` takes `argv[1]` as the
   primary, so `--help` prints `!! No such file: …/--help` and exits 2 — **SGK's own
   `tools/shard_key_audit.py` does exactly the same at `HEAD`**. So §3.3.1 becomes:
   - `python -m datastorekit.tools.sharded_store --help` exits 0;
   - `python -m datastorekit.tools.shard_key_audit --help` exits 2 with that line, as SGK's copy
     did (show both, SGK's before the deletion);
   - `python -m datastorekit.tools.shard_key_audit var/datastores/dk-and-rpk-migration-02-reference.sqlite`
     exits 0 with `VERDICT: OK`, inside §3.3.3's before-and-after hashes. This is a second read of
     R0's reference through the package, read-only, and is allowed by the prompt's §5.
4. **The index has the section** (prompt §4.6). 02 opened `docs/OPEN_ISSUES.md` §1.20 for this
   campaign. A5's row goes there; there is no new section. The count is 106 before 04.
5. **`store_soundness.py` changes** (README §4, "02 ships": "05 and 08 run them unchanged"). A2
   rewrites `docs/dk-and-rpk-migration/rehearsal/store_soundness.py:30`, the one S1-family script
   with a layer import. That is required (it would fail otherwise) and is the only change to it.
   The agent logs it as `STRUCTURALLY REQUIRED`, says so in README §4's "04 ships" line, and adds
   it to §3.3.4's load-by-path list.

**The rewrite script.** The prompt's §2.2 governs. Three details it leaves open:
- **Match the module as a token**, not a substring: `from Datastore import …` becomes
  `from datastorekit import …`, and `Datastore.SQL.ObjectFactories.cosmology_types` on another line
  is never touched. The JSON's `written` field is the text to find on that line; the script refuses
  the line if `written` is not there as a whole dotted name after `from` (or as the
  `import_module` argument in quotes).
- **Its own literals count.** The measuring script counts string literals naming a layer module in
  every tracked `.py`, `prompts/` included. The rewrite script's map (`"Datastore"`,
  `"Datastore.SQL.ObjectFactories.base"`, `"tools.sharded_store"`, …) will be counted under
  `prompts/`. That is fine. §3.1 then reads: `prompts/`'s three literals of `b510bc9`
  (`prompts/datastore-generic/orchestrator/measure-08/scripts/m5_imports.py:14-16`) unchanged, plus
  the rewrite script's, each listed by line. Likewise for `import_module` calls: one in `prompts/`
  (`measure-07/scripts/m2_open_behaviour.py:30`), plus any of the script's own.
- **It prints** what it changed, as `file:line old → new`, and a count. The log gives the count and
  says where the full listing is saved (`var/dk-and-rpk-migration-04/`).

**No line will reflow.** Applying the map at the JSON's 141 sites leaves no line over 88 columns,
and all 88 files, with `test_inventory_report.py`, `test_inventory_retired.py`,
`RunRegistry/stores.py`, `config/datastore.py` and `real_store_fixtures.py`, are `black`-clean at
`HEAD`. So `black` should change nothing after the rewrite. If it changes anything, the log says
what.

**Prose the rewrite leaves stale — record, do not fix.** Beyond `CLAUDE.md` and
`RunRegistry/stores.py:5`, about two dozen comments and docstrings in staying code name a layer
module by its old path. The orchestrator found these (2026-10-11, at `41d9bc0`):
- `Datastore.SQL.Datastore._build_schema` / `._factories` in docstrings:
  `ComputeTargets/tests/test_cosmology_representation_key.py:32`, `test_gauss_order_key.py:60`,
  `test_numeric_break_point_key.py:31`, `test_run_identity.py:41`;
  `Datastore/tests/test_database_errors_reach_the_caller.py:37`, `test_exact_identity_lookups.py:29`,
  `test_integral_read_batch.py:26`, `test_quadsource_policy_key.py:28`,
  `test_read_path_mechanics.py:25`, `test_value_reads_parent_key.py:27`,
  `test_wkb_numeric_parent_key.py:35`. (`._factories` has been wrong since `19f07ee` already.)
- `Datastore.store_inventory…`, `Datastore.shard_paths…`, `Datastore.SQL.schema…`:
  `RunRegistry/stores.py:107`, `:114`, `:1155`; `RunRegistry/tests/test_quadsource_atol_sweep_prepare.py:7`;
  `config/datastore.py:18`; `extract_common.py:157`; `main.py:283`; `tools/inventory_report.py:5`.
- File paths: `config/datastore.py:4-5` (``Datastore/SQL/``, `store_reader.py`,
  `store_inventory.py`); `RunRegistry/stores.py:209` (`Datastore/shard_paths.py`);
  `Datastore/tests/real_store_fixtures.py:403` (`Datastore/store_inventory.py`).

The prompt's §2.4.3 says "Nothing else", and README §2.2 gives 09 the acceptance "nothing outside
`prompts/` and dated `docs/` names a deleted file as a current path". So the agent **changes none
of these**. It re-derives the list itself (a regex over every tracked file outside `prompts/` and
`docs/`, excluding import lines and `import_module` calls, given in the log), and opens one §3 issue,
**`[04-comments-name-the-layers-old-module-paths]`**, whose next step is 09's acceptance 2, with
its index row under §1.20. `test_wkb_numeric_parent_key.py:86-87` is not on this list: the prompt
rewrites that comment (§2.3.5).

**What moves on purpose:**
- **Deleted (42):** the 15 layer files of A1; the 26 test modules of A3; `schema_description.py`.
- **Emptied (2):** `Datastore/__init__.py`, `Datastore/SQL/__init__.py`, to 0 bytes (`git
  cat-file -s` gives 0).
- **Rewritten by the script (88 files):** the 134 statements and 7 `import_module` calls. No other
  line in a file the script alone touches.
- **Hand edits, after the script:** `test_inventory_report.py` (the moved test, `import re` and any
  missing name), `test_inventory_retired.py` (§2.3.3), `test_wkb_numeric_parent_key.py:86-87`'s
  comment, `CLAUDE.md` (§2.4.1), `RunRegistry/stores.py:5` (§2.4.2).
  `standin_pool.py` and `test_wkb_numeric_parent_key.py:88` change by the script alone: they are
  among the 7 `import_module` calls.
- **New:** `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py`;
  `measure/out/04-datastorekit-measure_client_imports.md`.
- **Records:** the log, this board, README §4's "04 ships" line, `docs/OPEN_ISSUES.md` (two rows,
  106 → 108).
- **Not touched:** `RayTools/` but `:8`; every factory but its imports; `config/` but its one
  import (`config/datastore.py`, a `from` statement A2 rewrites); `main.py` and the `extract_*.py`
  but their imports; every other test; `prompts/` but this campaign's records and the new script.

**The facts, checked by the orchestrator on 2026-10-11 at `41d9bc0`.** The code outside `prompts/`
is `9ec603a`'s (`git diff --stat 9ec603a 41d9bc0 -- . ':!prompts'` is empty). Pass them on.
- **The measurement at `HEAD`.** `measure_client_imports.py sgk --commit 41d9bc0` differs from
  `measure/out/b510bc9-datastorekit-measure_client_imports.md` only by the commit, 451 files read
  (445), and the one `docs/` statement above: 234 statements (233), `docs/` 18 (17),
  `read_inventory` 30 (29). Strings 13, `import_module` 11, factory imports 100 (98 outside the
  factory package), all unchanged.
- **After the deletions alone** (a scratch tree with A1 and A3 applied and nothing rewritten,
  measured by the same script): 15 layer files absent; 185 statements (51 in `prompts/`); 3 string
  literals, all in `prompts/`; 8 `import_module` calls (1 in `prompts/`); **factory imports 99 (97
  outside the factory package)**, `Datastore/tests/` 46 → 45. The one factory import lost is
  `test_inventory_declarations.py:474` (`cosmology_types`), in
  `test_both_referencing_factories_declare_the_one_map`, not in the moved test. The moved test
  brings no layer or factory import that `test_inventory_report.py` lacks: that module already
  imports `read_inventory`, `build_full_store`, `_factories` (from `config.datastore`, not a
  factory module), `tempfile` and `Path`. It lacks `re`, and `format_inventory_report` if its
  `tools.inventory_report` import does not already name it. The nested
  `from CosmologyModels.model_ids import …` travels inside the body.
- **So the expected §3.1 figures at the commit:** 15 absent, 2 present at 0 bytes; outside
  `prompts/` 0 statements, 0 strings, 0 `import_module` calls; `prompts/` 51 statements in 31
  files, 3 strings plus the script's own, 1 `import_module` call plus the script's own; factory
  imports **99 (97)**.
- **Every name resolves.** For each of the 134 statements, every imported name was looked up with
  `getattr` on the mapped package module, in a fresh `python -I`. 156 names resolve, including
  `_merge_digests` (`test_quadsource_parent_key.py:47`). The only misses are the six `_factories`.
  `ray.is_initialized()` was false throughout.
- **No staying file depends on a deleted test module.** The only mentions of a deleted module's name
  in a staying file are docstrings (`test_quadsource_policy_foreign_key.py:5`,
  `test_wkb_parent_foreign_key.py:6`) and `test_inventory_retired.py`'s `HELPER_MODULE`, which
  §2.3.3 removes. `standin_pool.py`'s one staying user is
  `ComputeTargets/tests/test_quadsource_policy_main.py`. No staying `.py` uses a relative import into
  the layer. `Datastore/SQL/ObjectFactories/__init__.py` is 0 bytes and imports nothing.
- **Nothing staying reads a layer file by path** but `test_inventory_retired.py:75`, which §2.3.3
  repoints. `test_factory_select_columns.py:514` globs the factory folder, `base.py` included, but
  `base.py` contributes no finding to either analyser today (checked), so its deletion changes no
  result.
- **`test_inventory_retired`'s staying assertions hold against the package**:
  `find_spec("datastorekit.SQL.Datastore").origin` is the venv's file; its `Datastore` class has
  `read_table` and no `inventory`, and assigns no `InventoryConfigType`; `ShardedPool` from
  `datastorekit.SQL.ShardedPool` has neither method, no `inventory_config` parameter, a
  `read_table_config` parameter and a `primary` property.
- **The line numbers the prompt cites hold at `HEAD`:** `CLAUDE.md:47-73` (the section, 27 lines);
  `RunRegistry/stores.py:5`; `standin_pool.py:43-45`; `test_wkb_numeric_parent_key.py:86-88` and
  `:909`; `test_inventory_declarations.py:445-471`; `RayTools/RayWorkPool.py:8`; `main.py:93`;
  `replicated_write_fault_probe.py:37-39`.
- **A5 at `HEAD`.** Each of the six raises
  `ImportError: cannot import name '_factories' from 'Datastore.SQL.Datastore'` — five on a
  no-argument run, `order_audit.py` through `d3_tables()` (§0, correction 2). The six probes that
  import `lookup_keys/common.py` are `probe_01_tagged_read_batch`, `probe_01b_fix_shape`,
  `probe_04_exit_time_and_solver`, `probe_04b_exact_match_fix_shape`,
  `probe_05_value_lookups_and_read_batch` and `probe_06_oneloop_tags_and_redshift_zero`.
- **Entry points.** The six `extract_*.py` parse arguments and call `ray.init` at module scope
  (`extract_Gk_data.py:93`, `:100`): **byte-compile them; never import them.** The five `docs/`
  scripts §3.3.4 loads by path each guard their entry point with `if __name__ == "__main__"`:
  `docs/handover/quadsource_atol_sweep.py`, the two `docs/source-remediation-verification/`
  scripts, `docs/a3-v2-readiness/rehearsal/store_soundness.py`, and
  `docs/dk-and-rpk-migration/rehearsal/store_soundness.py` (added to the list, correction 5).
- **The package's tools.** `python -m datastorekit.tools.sharded_store --help` exits 0. For
  `shard_key_audit`, see correction 3. Both copies of `shard_key_audit` on R0's reference give the
  same output, `VERDICT: OK`, exit 0, and R0's files kept their SHA-256 and mtimes.
- **The string default.** `config.defaults.DEFAULT_STRING_LENGTH` and
  `datastorekit.defaults.DEFAULT_STRING_LENGTH` are both 256.
- **R0's reference store**: `var/datastores/dk-and-rpk-migration-02-reference.sqlite`, four shards
  (`-shard0000` … `-shard0003`) and `.manifest.json`; primary and shard 2 last modified 14:30:17,
  shards 0, 1, 3 at 14:38:00. No `-journal`, `-wal` or `-shm` file under `var/datastores/`.
- **On this Mac `stat` is GNU coreutils** (`/opt/local/libexec/gnubin/stat`), so `stat -f` means
  "file system". Use `/usr/bin/stat -f '%m %N'`, or Python's `os.stat`, for mtimes.
- **`grep` in the agent's shell may be a function wrapping `ugrep`**, and `xargs grep` reaches BSD
  grep, which has no `-P`. Use `git grep` or Python for any regex the log relies on.
- **The registry.** 21 runs: 19 finished, 2 `unknown` (the pre-registry `a3-pilot` pair), none
  `running`. No Ray process is up.
- **Expected counts after 04:**
  - **Datastore 320 = 54 + 103 + 101 + 62** in the four groups: a–f 54 (6 + 21 + 18 + 9); g–o 103
    (38 + 10 + 11 + 19 + 19 + 6, with `test_inventory_report` 18 + 1 and `test_inventory_retired`
    7 − 1); p–r 101 (24 + 34 + 4 + 19 + 20); s–z 62 (18 + 39 + 5). 18 modules.
  - RunRegistry 286, ComputeTargets 693, AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148
    (1 skipped): unchanged.
- **The index:** 106 before; 108 after (A5's row and the prose issue's), both under §1.20.

**What the review exists to establish.**
- **(E1) A1.** The 15 are gone, the two `__init__.py` are 0 bytes, `tools/__init__.py` and
  `tools/inventory_report.py` stay, and `import Datastore.SQL.ShardedPool` (or any of the 15) fails
  with `ModuleNotFoundError` in a fresh process.
- **(E2) A2 is exact.** Every line the commit changes in a file outside the hand-edit list is one of
  the JSON's 141 sites, changed by the map and nothing else.
- **(E3) A2 is complete.** The script at the commit gives §0's expected figures.
- **(E4) A3.** The 27 files are gone; the moved test's body dumps identically by `ast`;
  `test_inventory_retired` changes as §2.3.3 says and its four kept tests are otherwise unchanged;
  320 tests in 18 modules, and every test id that ran before and still exists passes.
- **(E5) The package reads R0's store.** `39809dca…`, format 6, 21 classes equal to the sidecar's,
  through the package, and no file of the store changes.
- **(E6) The breakages.** (i)–(v) each fail what the prompt says.
- **(E7) A4 and A5.** `CLAUDE.md`'s section says §2.4.1's six things in no more than 27 lines;
  `stores.py:5` names the package; the six A5 files fail before and after on `_factories`, now on
  `datastorekit.SQL.Datastore`.
- **(E8) The records.** The board, README's line, the two issues and the index agree with the tree.

**Conventions.** Those of [`prompt-01.md`](prompt-01.md) §0 bind the agent, with these changes:
- **The venv.** The agent installs and uninstalls nothing. `datastorekit` is as 03 left it.
- **DatastoreKit.** The agent runs `docs/extraction/measure_client_imports.py` from that root with
  that venv, redirecting output to SGK's `var/dk-and-rpk-migration-04/` by absolute path. It changes,
  fetches into, checks out and commits nothing there (README §5 rule 12).
- **Scratch commits.** The prompt's `git commit-tree` recipe is used once, on the final staged tree
  (§3.1). For breakage (i)'s measurement, make the commit from a **temporary index**
  (`GIT_INDEX_FILE=<scratch>/idx git read-tree HEAD`, then `git update-index --add` the mutated
  file), so that the checkout's index is never touched by a mutation. No ref moves; nothing is
  amended.
- **Test ids before.** Before any edit, the agent saves `unittest -v` output for the six roots
  (Datastore in its four groups) at `HEAD`, under `var/dk-and-rpk-migration-04/`. The after-lists
  come from the same commands. The comparison is of sorted ids with outcomes.
- **Breakage (v) is self-contained.** The diff edits only `test_inventory_retired.py`: for example,
  the actor test parses the package's source with `"\n    def " + _INVENTORY + "(self):\n        pass\n"`
  inserted into the `Datastore` class text before `ast.parse`. It names no file outside the
  repository, so `git apply` replays it, and it spells the retired name the module's way, so it does
  not also trip `test_the_grep_prints_nothing`. Say in the log what else, if anything, failed.
- **R0's store.** Only `store fingerprint` without `--write` and `shard_key_audit`, each inside one
  before-and-after record of SHA-256 and mtime of all six files (primary, four shards, sidecar).
- **Ray.** The agent starts none. No test, import or check may initialise it; §3.3.6's checks are
  recorded.
- **Stop conditions are the prompt's §5, unchanged.** In particular, a staying test that fails for
  any reason but a missed import is a stop, and so is any fingerprint other than `39809dca…`.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` must be empty.
   - `git worktree list` must show this checkout alone.
   - `git log --oneline 41d9bc0..HEAD` must touch no code; otherwise see the gate.
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`.
3. **Ray.** Record `pgrep -lf 'gcs_server|raylet|ray::'`. No listed process may come from this
   repository's `venv`.
4. **The venv.** `pip show datastorekit` gives 0.2.2 in this venv's `site-packages`, not editable;
   `raypoolkit` is absent; `pip check` is clean.
5. **Baselines.** Re-measure all six roots at `HEAD`, serially, with the load average (README §7).
   The last measurement was at `9ec603a` (03's review): AdaptiveLevin 32, CosmologyModels 39,
   LiouvilleGreen 148 (1 skipped), RunRegistry 286, ComputeTargets 693, Datastore 739
   (163 + 185 + 200 + 191). Give the agent the numbers.
6. **R0's store.** Record the SHA-256 and mtime of its six files, for the review's check 9.

## 2. Dispatch

Dispatch one fresh-context subagent, on **Opus**. Give it:
- the prompt;
- the campaign README and this board;
- `HEAD`, the baselines and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/04-adopt-datastorekit.md`, in README §5.1's form,
  with the prompt's §7 additions.
- **The files it may add, change or delete** are §0's "What moves on purpose", and nothing else. In
  particular, not the stale prose §0 lists, not `orchestrator/`, not `measure/` but the one saved
  measurement, and nothing in DatastoreKit.
- **Order:** save the before-ids and A5's before-failures; delete (A1, A3, after moving the test
  out); run the script; hand edits; `black`; measure the staged tree; suites; §3.3; breakages; A5's
  after-failures; records; commit.
- **Stop and ask** on any of the prompt's §5 conditions.
- **Run the Datastore suite in four groups**, in the foreground.

## 3. The review — fourteen checks

1. **Scope.** `git diff --stat HEAD~1 HEAD` and `git diff --name-status HEAD~1 HEAD`.
   - `D` for exactly the 15 + 26 + 1 files; `M` to 0 bytes for the two `__init__.py`; `A` for the
     script, the measurement and the log.
   - Every `M` file is one of the 88 or a hand-edit file of §0, or a record. Nothing under
     `RayTools/` but `RayWorkPool.py`, whose diff is its `:8`.
2. **E2, the rewrite is exact.** From `HEAD~1`'s JSON (made fresh here, with the deletions applied
   in a temporary index), apply the map at each of the 141 sites to `HEAD~1`'s text, in the
   scratchpad. For every file not in the hand-edit list, the result equals the committed blob byte
   for byte. For each hand-edit file, the diff between the result and the committed file is the
   hand edit alone.
3. **E3, by the script at the commit.** Run `measure_client_imports.py sgk --commit <commit>`. It
   gives §0's expected figures, 99 (97) factory imports included, and equals the saved
   `measure/out/04-…md` but for the commit lines. `git diff --stat <scratch> <commit>` lists only
   the records.
4. **E1, the layer is gone.** `git cat-file -s` gives 0 for both `__init__.py`. In a fresh
   `python -I` with the repository on its path, each of the 15 modules fails with
   `ModuleNotFoundError`, and `import config.datastore` succeeds with every factory's
   `SQLAFactoryBase` being `datastorekit.SQL.factory_base.SQLAFactoryBase`, and `ray.is_initialized()`
   false.
5. **The rewrite script.** Read it. It reads the JSON, refuses a line whose text lacks the module,
   matches by token, writes nothing under `prompts/`, and prints its changes. Run it again on the
   committed tree with the old JSON: it must refuse every line (the text no longer names the old
   module), and change nothing.
6. **E4, the moved test.** `ast.dump` of `TestTheReportRendersBothCosmologyTypes`'s method body,
   docstring stripped, equals `HEAD~1`'s `TestResolve.test_the_report_renders_both_cosmology_types`.
   The class docstring says where it came from and why it stayed (U16).
7. **E4, `test_inventory_retired`.** By `ast` against `HEAD~1`: `test_no_factory_has_the_old_method`
   and `test_config_sharding_has_no_merge_policies` dump identically; the actor test differs only
   in the path it parses; the pool test only by its import; the grep test only by the pathspec;
   `HELPER_MODULE`, `HELPER_MATCH` and the excluded-file test are gone. The module docstring says
   what changed and why.
8. **The suites.** All six roots, serially, at the commit: Datastore 320 (54 + 103 + 101 + 62),
   RunRegistry 286, ComputeTargets 693, AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148
   (1 skipped). Compare the agent's before- and after-id lists: the differences are exactly the 419
   deleted ids, the moved id's new name, and the one removed `test_inventory_retired` test.
   `black --check` is clean on every touched `.py`.
9. **E5, R0's store.** `PYTHONPATH=. ./venv/bin/python -m RunRegistry store fingerprint
   var/datastores/dk-and-rpk-migration-02-reference.sqlite`, without `--write`: format 6,
   `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, 21 classes equal to the
   sidecar. The six files' SHA-256 and mtimes equal §1 step 6's, before and after both the agent's
   reads and this one, and no journal appears. Confirm by `sys.modules` in-process that
   `read_inventory` came from `datastorekit.store_inventory`.
10. **E6, the breakages.** Replay (i)–(v) from the log in `bash`, each from a clean tree: `git apply
    --check`, apply, run the named tests (and for (i), measure from a temporary index), see the
    failure, `git apply -R`, and `git status` clean.
11. **E7, A5.** Re-run the six as §0 says. Each raises the `ImportError` on `_factories`, now from
    `datastorekit.SQL.Datastore`. `order_audit.py` through `d3_tables()`.
12. **E7, `CLAUDE.md`.** The section is titled "The datastore layer is `datastorekit`", is at most
    27 lines, and says the six things of §2.4.1. Nothing else in `CLAUDE.md` moves; the test command
    is unchanged (09's). `RunRegistry/stores.py`'s diff is `:5` and nothing else in its docstring,
    plus the script's import lines.
13. **Entry points.** `py_compile` on `main.py`, the six `extract_*.py`, `extract_common.py` and the
    14 rewritten `docs/` files. Load the five `docs/` scripts by path; Ray stays down.
    `ComputeTargets/tests/test_main_plumbing` passes (it is in check 8's count).
14. **E8, the records.**
    - The log: the prompt's §7 list, each correction of §0 classified, the before-and-after counts
      by group, §3.1 against `b510bc9`'s appendix and §0's figures, the fingerprint in full, the
      breakages as diffs, and the stale-prose list with its regex.
    - The board: the §1 row, A1–A5 ✅, the two §3 entries, and `### 5.4 After prompt 04 (<date>)`
      naming the rollback (`git revert` alone; README §4.2).
    - README §4's "04 ships" line names `store_soundness.py`'s rewrite.
    - `docs/OPEN_ISSUES.md`: two rows under §1.20, 108 by counting, the date.

## 4. After the review

- **Record it on this board** as a §5.4 *Orchestrator review of prompt 04* paragraph: what each
  check found, the fingerprint, and anything handed on to 05. Fix small residue in a follow-up
  commit of its own. That includes:
  - "this commit" → the SHA, on the board and in the log's header;
  - `orchestrator/prompt-04.md` added to the board's list of notes, marked "used for 04".
- Anything the review finds that is not residue is opened on this board's §3 and in the index.
- **Report to the user:**
  - what went and what stayed, with the counts;
  - the measurement at the commit;
  - the fingerprint of R0's store through the package;
  - the six roots' counts;
  - the two issues opened, and the corrections this note made to the prompt.
- **Then 05–08 are written** (README §2.1), from 04's log: R1 at 04's commit, with every manifest's
  `packages.datastorekit` at `6d8ee9ae…`.
