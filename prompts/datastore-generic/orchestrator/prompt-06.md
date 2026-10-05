# Orchestrator — prompt 06, give the layer its registry

Read [`../README.md`](../README.md) §0.1, §1, §4, §5, and §6.2 (U2 and the two choices the user
took on 2026-10-05). Then the audit's C1, C2 and D
([`docs/datastore-generic-audit.md`](../../../docs/datastore-generic-audit.md)), and the board's
review of 05 (§1). The conventions are [`prompt-05.md`](prompt-05.md)'s, carried below.

**You do not write code.** You may:
- run the suites and the tests the log names;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process check or a CLI from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`06-give-the-layer-its-registry.md`](../06-give-the-layer-its-registry.md)
**Board items:** G1–G3 · **Closes:** `[07-drop-actions-list-a-parent-table-before-its-children]`
(`datastore-integrity`) · **Narrows:** `[00-the-generic-datastore-layer-hard-codes-project-schema-facts]`
· **Opens:** nothing expected · **Model:** Opus, as the prompt recommends.
**Gate:**
- 05 landed (`bc8bffc`) and reviewed (`dbc59f9`); 06 written (`7117f2a`); the tree clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code. If a later commit
  touches `Datastore/`, `config/`, `main.py`, `extract_*.py`, `RunRegistry/` or `tools/`, re-check
  every fact below that it could move before dispatch, and hand the agent the corrections as an
  addendum.
- One prompt at a time in this checkout.

## 0. What makes this prompt unusual

**It changes every constructor and every reader's signature, and for a store nothing.** The
registry stops being something the layer imports and becomes something it is given. That touches:
- `ShardedPool`, the `Datastore` actor, `open_read_only` and `read_inventory`;
- each of their callers: `main.py`, the six extract scripts, `extract_common`, `RunRegistry`'s
  fingerprint and `tools/inventory_report.py`;
- about thirty test modules, almost all by one import line or one argument.

What a store holds and what any reader reports must not move. Measured with a rough version of the
change (below):
- the fingerprint of `build_full_store` was byte-identical (digest
  `6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759`, problems `{}`);
- so were `build_real_store`'s inventory records and problems;
- `main.py --inventory` gave byte-identical stdout (14856 bytes) and stderr;
- `python -m RunRegistry store fingerprint` gave the same digest;
- a read-write stand-in pool dropping every group from a full store gave the same schema, row
  counts and `foreign_key_check`.

**Two choices of method were the user's** (2026-10-05, README §6.2 under U2), both taken as
recommended:
- the layer is given the **tables** to drop, and orders them by foreign key;
- the inventory reads the replicated set **from the store's primary**.

Every primary the current code writes records exactly `config.sharding.replicated_tables`, in
order. That was measured on `build_real_store`, `build_full_store` and a store the real
`ShardedPool` created on stand-in shards, so the second choice changes no current inventory. The
hand-built primary of `shard_store_fixtures` records only `['version', 'wavenumber']`, and none of
its users reaches the inventory.

**What moves on purpose:**
- **The drop order within a group.** The old lists put the parent first and violated 15
  within-group foreign keys. Under `PRAGMA foreign_keys = ON` on a full-store shard, the old order
  failed at the first drop; the derived order dropped all 25 tables and left `foreign_key_check`
  empty. Production never enforces, so no store changes. This is what closes
  `[07-drop-actions-list-a-parent-table-before-its-children]`.
- **The actor's console lines**: one per table dropped, where there was one per group (25 where
  there were 10 for a full drop).
- **The read-only pool on a primary without `replicated_tables`.** It now meets the reader's
  `RuntimeError` at its step 2, where after 05 it met the pool's own `StoreSchemaMismatch` at
  step 3. No test pinned either. The prompt has the agent say so in the pool's comment.

**What 06 does not fix, and why it is recorded.** Measuring the drop found that a group dropped
alone, with enforcement off as in production, leaves other groups' rows naming rows it dropped.
Six groups do this, with 1–4 dangling references per shard on the full fixture store
(`out/m11.txt`, measured by the orchestrator at `dbc59f9` with
`scripts/m11_orphans.py`). That is
`[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]`, opened with the prompt (`7117f2a`)
for the user to decide. 06 must not touch it: deciding whether a drop cascades or refuses changes
what `--drop` does.

**The test plan was measured, not read.** On 2026-10-05 at `dbc59f9`, a read-only Opus subagent
made a rough version of the change in a scratch worktree, ran every suite, and removed the
worktree. The rough diff, its scripts and their outputs are committed beside this note in
[`measure-06/`](measure-06/), with scratch paths as `<scratch>`.
- **Phase A, code only:**
  - Datastore ran 197, with 93 errors (out of 644);
  - RunRegistry ran 199, with 3 errors;
  - ComputeTargets ran 638, with 9 errors.

  These were almost all module import errors (`_factories`, `_drop_actions`), plus the `TypeError`
  of a pool or a reader built without the registry (`out/A_*.txt`).
- **Phase B, mechanical edits** (imports, the registry passed, `drop_tables`): 7 methods still
  failed, the three `TestRunLabels` fakes and the four `TestDropOrder` methods (`out/B_*.txt`).
- **Phase C**, those seven changed and a 15-test sketch of the new module added:
  - Datastore 659 (122 + 146 + 200 + 191), RunRegistry 280, ComputeTargets 693;
  - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped).

  Every one of these is in the prompt's plan, with its fate.

**The prompt departs from the rough diff in four places**, so the agent's diff will not match
`rough_change.diff`, and need not:
- **`schema.drop_order`.** The rough version sorted inside the actor (`reversed(sort_tables(…))`),
  and its tests read the order through a recording inspector. The prompt puts the order in
  `schema.py`, so that it can be tested directly.
- **Precedence on a read-only pool.** The rough version checked undeclared drop tables before the
  read-only branch, so a read-only pool given one met `RuntimeError` instead of `ReadOnlyWrite`.
  The prompt keeps `ReadOnlyWrite` first.
- **`drop_groups`' key order.** The rough version kept the old dict order, and `list(drop_groups)`
  moved the `--drop` choices in `--help` and in argparse's invalid-choice message
  (`out/m6.txt`). The prompt orders the keys as `main.py`'s `allowed_drop_actions`, so both are
  byte-identical.
- **The actor's own refusal of an undeclared drop table.** It is in the prompt's plan as a test;
  the sketch did not test it.

**A test outside the plan failing is therefore a real signal.** The agent stops and reports.

**The facts, checked by the orchestrator on 2026-10-05 at `dbc59f9`.** Pass them on. The prompt's
line numbers were taken from the tree. These need care:
- **`Datastore/SQL/Datastore.py`** [read]:
  - the factory imports run `:14-76`, and `ObjectFactories.base` (`:65`) stays for
    `_FactoryMappingType`;
  - `_factories` is `:89-129`, and `_drop_actions` and `_drop_order` are `:135-171`;
  - `register_factories(_factories)` is at `:239`;
  - the read-only refusal is at `:305-310`, and `_drop_actions` at `:538-572`.
- **`Datastore/SQL/ShardedPool.py`** [read]: `_factories` is imported at `:14` and used at `:539`,
  `:1047`, `:1843`, `:2005`, `:2037` and `:2154`. The two actor constructions are at `:272-286`
  and `:468-483`.
- **`store_reader.py:42`, `:167`; `store_inventory.py:846-848`, `:869`** [read].
- **The callers** [grep, M9]:
  - `ShardedPool(` is called at `main.py:4168` and in the six extract scripts at the lines the
    prompt gives;
  - the non-test callers of the reader and the inventory are `main.py:301`,
    `extract_common.py:174`, `RunRegistry/stores.py:1494` and `tools/inventory_report.py:164`;
  - the last one was found by the measurement, not the design.
- **`docs/a3-v2-readiness/reader_writes_probe.py` is imported live** by `test_read_only_pool` and
  calls `standin_pool.open_pool`. So `standin_pool` must default `factories`, as the prompt says.
- **No call of `ShardedPool(` passes an argument by position** [M9], so a keyword-only
  `factories` breaks no caller.
- **Pickling** [M4, no `ray.init`]: `ray.cloudpickle.dumps(factories)` is 2936 bytes for 38
  entries, and round-trips to the identical classes, pickled by reference.
- **Imports** [M5]:
  - before, importing `ShardedPool`, `Datastore`, `ClientPool` or `store_reader` loads 22
    `ObjectFactories` modules, 21 `ComputeTargets`, 3 `CosmologyConcepts`, 11 `CosmologyModels`,
    6 `MetadataConcepts` and `config.defaults`;
  - after, only the `ObjectFactories` package and `base`, and `config` with `config.defaults`.
    `MetadataConcepts` stays (`store_tag`, prompt 07's).
- **No cycle** [M3]: none in any drop group, nor in all of them together. No table outside the
  groups has a foreign key into a dropped table.
- **Not touched, and staying:**
  - the `docs/` scripts the prompt's §6 lists, which will not run at 06's commit;
  - `ProfileAgent`, `SerialPoolBroker`, `replication.py` and `shard_paths.py`;
  - the closed-store operations;
  - every factory;
  - every store under `var/`.
- **Expected counts after 06:**
  - Datastore: 644 plus the new tests, which the log names; the sketch had 15;
  - RunRegistry: 280; ComputeTargets: 693; every other suite unchanged.
- **The index.** **108** at `7117f2a`, matching the header. 06 closes one issue and opens none
  expected, so it is **107** after, plus one for each issue it opens.

**What the review exists to establish.**
- **(E1) The layer is given, and imports, no registry.**
  - None of `Datastore/SQL/`, `store_reader.py` or `store_inventory.py` imports `_factories`,
    `config.datastore`, `config.sharding` or a factory other than `base`.
  - `Datastore.py` holds none of the three registries, and `ClientPool` holds no batch sizes.
  - The inventory's replicated set is the store's record.
  - Nothing else in those files changes, docstrings aside.
- **(E2) `config/datastore.py` holds exactly what moved.**
  - `factories` is the old `_factories`: the same keys in the same order, and the same classes.
  - `serial_batch_sizes` is the old table.
  - `drop_groups` has the old groups' tables, its keys in `main.py`'s old choice order.
  - `_drop_order` is gone, and `schema.drop_order` replaces it.
- **(E3) Nothing a store holds or a reader reports changes.**
  - The fingerprint is byte-identical.
  - `main.py --inventory`, `--inventory --drop …`, `--drop nonsense` and `--help` are
    byte-identical.
  - A full drop leaves the same store.
  - Every test outside the plan passes untouched.
- **(E4) The test plan was followed:**
  - no test deleted;
  - each edit is of the kind the plan names, in the module it names;
  - the four `TestDropOrder` conversions keep their purpose;
  - every new method is in `test_layer_registry.py`.
- **(E5) The pins.** (i)–(v) each fail a named test.
- **(E6) The records.**
  - `[07-drop-actions-…]` is closed into `datastore-integrity`'s §4, and its index row deleted.
  - The generic-layer issue is narrowed.
  - The index is 107 (or more).
  - This board has the §1 row, G1–G3 and §5.

**Conventions, carried from 01–05.** All of these bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only. Show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the diff and the tests show.
- **The SHA.** The agent writes "this commit"; the orchestrator substitutes the SHA in its
  follow-up.
- **Suites.** Use README §7's command, write the output to a file, and grep the verdict from it.
  Run each suite **in the foreground**, serially, within the turn.
  - Run the Datastore suite in module groups (`-p 'test_[a-f]*.py'`, `[g-o]`, `[p-r]`, `[s-z]`) and
    add the counts. A Bash call is capped at ten minutes.
  - Under load, four tests are known to fail on wall clock or chance:
    - `test_tk_wkb_phase.TestCost.test_wall_time_per_object`;
    - `test_background_tau.test_build_cost`;
    - `test_numeric_phase_cut.test_the_print_is_gated_both_ways`;
    - `LiouvilleGreen`'s `test_3bessel_analytic.test_JJJ`. Its draws are unseeded
      `uniform(0.1, 5)`. It failed once in the measurement, then passed three times alone in each
      tree.

    Re-run a failure alone before calling it real. If it persists alone, settle it on an export of
    the tree before the prompt.
  - In a git worktree, RunRegistry reports one skip (`test_pipeline_adoption`, whose manifest is
    under the gitignored `var/`). In this checkout it does not.
- **Breakage diffs** are recorded so that `git apply` replays them from a clean tree at the agent's
  commit. Check each with `git apply --check` and `-R`, and replay in `bash`, not zsh.
- **Dead helpers** that only changed tests used go in the same commit, after a grep
  (IMPLEMENTATION CHOICE). This includes an import left unused.
- **Scratch.** The agent uses its own subdirectory of the session scratchpad, never `/tmp`. Python's
  `tempfile` in tests is fine.
- **Ray.** 06 starts no Ray. ChamPBH's cluster, if running, is left alone. `standin_pool` needs no
  Ray.
- **Stop conditions are real.** A stop goes to the user through the orchestrator, and the agent
  resumes with `SendMessage`.
- **Other sessions** may land commits on this branch. Review against the agent's commit, and check
  `git log` before measuring at `HEAD`.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` must be empty.
   - `git worktree list` must show this checkout alone.
   - `git log --oneline 7117f2a..HEAD` must touch no code (otherwise see the gate).
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`.
3. **Ray.** Record `pgrep -lf 'gcs_server|raylet|ray::'`. No listed process may come from this
   repository's `venv`.
4. **Baselines.** Check first that `./venv/bin/python -c 'import scipy.sparse.linalg'` succeeds.
   The last measurement was at `bc8bffc` (05's review, check 10):
   - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped);
   - ComputeTargets 693; RunRegistry 280;
   - Datastore 644 (122 + 131 + 200 + 191).

   Re-measure all six at `HEAD`, serially, with the load average, and give the agent the numbers.
5. **The index.** 108 now; 107 after, plus any issue 06 opens. Tell the agent both.
6. **The pre-images**, for checks 1–4. Record `git rev-parse HEAD:<path>` for:
   - `Datastore/SQL/Datastore.py`, `ClientPool.py`, `ShardedPool.py` and `schema.py`, and
     `Datastore/store_reader.py` and `store_inventory.py`;
   - `main.py`, `extract_common.py`, the six `extract_*.py` scripts, `RunRegistry/stores.py` and
     `tools/inventory_report.py`;
   - `Datastore/tests/standin_pool.py`, `schema_description.py` and `real_store_fixtures.py`;
   - every test module of the prompt's plan.

   Export `HEAD` to the scratchpad (`git archive HEAD | tar -x -C …`) for check 8, and build a
   `build_full_store` there for the `main.py` and CLI probes.

## 2. Dispatch

Dispatch one fresh-context subagent, on **Opus**. Give it:
- the prompt;
- the campaign README, the board (including the review of 05), 05's log, and the audit's C1, C2
  and D;
- `HEAD`, the baselines and the index counts;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad. The
  agent does not read `orchestrator/`. It may be given copies of `measure-06/rough_change.diff`
  and `measure-06/scripts/`, as a worked example. Tell it that the prompt, not the rough diff,
  governs, and where they differ (§0 above).

Tell it plainly:
- **One commit, at the end.** Its log is `logs/06-give-the-layer-its-registry.md`, in README
  §5.1's form, with the prompt's §7 additions.
- **The files it may add or change are the prompt's §6 list, and nothing else.** In particular, it
  must not touch:
  - anything under `docs/` other than `OPEN_ISSUES.md`;
  - `orchestrator/`, any factory, or `var/`;
  - any test not named in the prompt's plan;
  - in `main.py`, the extract scripts, `extract_common.py`, `RunRegistry/stores.py` and
    `tools/inventory_report.py`, anything but the lines the prompt names.
- **Stop and ask** on any of the prompt's §5 conditions. A test outside the plan failing is current
  behaviour moving, never a reason to edit it.
- **The Datastore suite in four groups**, in the foreground.

## 3. The review — ten checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only the prompt's §6 list.
   - The diff over `docs/` (but `OPEN_ISSUES.md`), `Datastore/SQL/ObjectFactories/`,
     `Datastore/replication.py`, `Datastore/shard_paths.py`, `Datastore/SQL/ProfileAgent.py`,
     `Datastore/SQL/SerialPoolBroker.py`, `config/sharding.py`, `var/` and `orchestrator/` is
     empty.
   - On the `datastore-integrity` board, only the closed issue moves.
2. **E1, by grep and by running.** The prompt's §3.2 greps print nothing.
   `test_layer_registry`'s import test passes; run it alone and read its allow-list. It must allow
   no more than the prompt's: `MetadataConcepts` and `utilities`.
3. **E1, E2, by `ast`, against the pre-images.** Compare each function with its docstring
   stripped.
   - `Datastore.py`: only `__init__`, `_open_read_only` and `_drop_actions` (now `_drop_tables`)
     change, and the module-level registries and factory imports are gone. Every other method dumps
     identically.
   - `ShardedPool.py`:
     - changed: `__init__`, `_open_read_only`, `_check_replicated_tables_read_only`,
       `_reconcile_replicated_tables`, `_recompute_validated`, `_prune_needed`, `_prune_shard`
       and `_prune_replicated_tables`;
     - each differs only by `self._factories`, the actor arguments and `drop_tables`;
     - new: the `factories` property and the undeclared-drop check;
     - every other function dumps identically, the closed-store operations especially.
   - `schema.py`: `build_schema`, `schema_differences`, `SchemaDifferences` and
     `StoreSchemaMismatch` dump identically, and `drop_order` is new.
   - `store_reader.py`: `open_read_only` and `ReadOnlyStore` change, plus the new reading helper.
     `store_inventory.py`: only `read_inventory` changes.
   - `ClientPool.py`: only `SerialPoolManager.__init__` and `lease_serial` change.
   - `config/datastore.py`, in-process:
     - `factories` equals the pre-image's `_factories`: the same keys in the same order, and `is`
       the same classes;
     - `serial_batch_sizes` equals `_default_serial_batch_size`;
     - `drop_groups` has the same tables per key as `_drop_actions`, with its keys in the
       pre-image `main.py`'s `allowed_drop_actions` order (extract that with `ast`).
   - The callers: `main.py`, the extract scripts, `extract_common.py`, `RunRegistry/stores.py` and
     `tools/inventory_report.py` differ only in the lines the prompt names.
4. **E4, the tests, by `ast`.** List the test methods of every module of the plan, before and
   after, and of the new module.
   - None is gone except the four `TestDropOrder` renames.
   - Each changed method is one the plan names, changed as the plan says. The import-only modules
     differ only in imports.
   - Every new method belongs to `test_layer_registry.py`, and the log names its purpose.
5. **E2, by reading and running.** Read `test_layer_registry.py`.
   - Each refusal asserts the exception, the names, unchanged files, and for a pool that no actor
     was constructed.
   - The full-store drop runs with enforcement on and checks `foreign_key_check`.
   - Run the module, the changed tests and `TestDropOrder` by name.
6. **E3, by running.** Run by name:
   - `test_store_schema`, which includes U1's recovery tests;
   - `test_reconcile_at_open`, `test_version_row_at_open`, `test_prune_at_open`,
     `test_replicated_write` and `test_read_only_pool`;
   - `test_store_inventory`, `test_inventory_report`, `test_inventory_consumers` and
     `test_store_reader`;
   - RunRegistry's `test_store_fingerprint`, and ComputeTargets' `test_reader_read_only`.
7. **E5, the pins.** In `bash`, replay each breakage diff (i)–(v) with `git apply`, run the named
   tests and see them fail. Revert each with `git apply -R`, and check that `git status` is clean.
8. **The probe, against the fix.** From the scratchpad, at the agent's commit and at the export of
   `HEAD~1`:
   - **the fingerprint** of `build_full_store`, with `taken` removed, is byte-identical. Give the
     digest; it should be `6576b85d…`;
   - **`main.py`**, on a `build_full_store` primary, byte-identical at both trees:
     - `--inventory` (stdout, stderr, exit 0);
     - `--inventory --drop gk-source` (exit 2);
     - `--drop nonsense` (stderr, exit 2);
     - `--help`;
   - **`python -m RunRegistry store fingerprint … --runs-root <scratch>`**: the same digest and exit
     0, with the runs root left empty;
   - **a full drop on stand-in shards**: a read-write pool on a full store, at the agent's commit
     with `drop_tables=tables_to_drop(list(drop_groups))`, and at `HEAD~1` with every action.
     Compare the schema, the row counts and `foreign_key_check`
     (`measure-06/scripts/m3b_pool_drop.py`, adapted);
   - **imports**: `measure-06/scripts/m5_imports.py` at the agent's commit gives the after-list of
     §0;
   - **pickling**: `ray.cloudpickle` round-trips `config.datastore.factories` without `ray.init`.
9. **E6, the records.**
   - `[07-drop-actions-…]` is in `datastore-integrity`'s §4, with a dated "Closed (… prompt 06 of
     datastore-generic)" line, and its index row is gone.
   - The generic-layer issue has a dated "Narrowed" line naming rows 9, 10, 11 and 14, and its
     index hook says so.
   - The index is 107 (or more) by counting, matching the header.
   - This board has the §1 row, G1–G3 and §5.
10. **Suites.** All six, serially at `HEAD`:
    - Datastore at 644 plus the log's new tests, and RunRegistry at 280;
    - every other count unchanged.

    `black --check` is clean on every touched Python file.

## 4. After the review

- **Record it on this board** as a §1 *Orchestrator review of prompt 06* paragraph, in the form
  of 01's–05's. Fix small residue in a follow-up commit of its own. That includes:
  - "this commit" → the SHA;
  - this note marked "used" in the board's list;
  - the board header's count.
- **Report to the user:**
  - what the layer now imports, before and after (§0's module counts);
  - what moved to `config/datastore.py`, and that `main.py`'s CLI text is unchanged;
  - what a store sees (check 8: nothing changes; the fingerprint digest);
  - the drop order within groups, and the issue it closes;
  - `[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]`, still the user's to decide;
  - the tests edited, converted and added.
- **What 06 hands on.** 07's note must carry:
  - `ShardedPool._recompute_validated` still names `BackgroundModel` through
    `self._factories["BackgroundModel"]` (C1 row 6);
  - `ShardedPool` and `store_reader` still load `MetadataConcepts` through `store_tag` (row 1, U4).
    `test_layer_registry`'s allow-list names it, and 07 should remove it from the list once the
    import is gone;
  - the registry now arrives through `ShardedPool(…, factories=…)`. A fact 07 declares in a
    factory's `register()` reaches the pool through `build_schema(…, self._factories)`, and the
    reader and the inventory through their `factories` arguments;
  - the `docs/` scripts that no longer run at `HEAD` (the prompt's §6);
  - whatever 06's log hands on.
