# Orchestrator — prompt 07, declare the replication facts in the factories

Read [`../README.md`](../README.md) §0.1, §1, §4, §5 and §6.2 (U4, and the user's choice of
method for 07, taken on 2026-10-06). Then read:
- the audit's C1 (rows 1, 2, 4–8, and "What is infrastructure"), C2 and D
  ([`docs/datastore-generic-audit.md`](../../../docs/datastore-generic-audit.md));
- the board's review of 06 (§1), with its "Handed on" list.

The conventions are [`prompt-06.md`](prompt-06.md)'s, carried below.

**You do not write code.** You may:
- run the suites and the tests the log names;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process check or a probe from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:**
[`07-declare-the-replication-facts-in-the-factories.md`](../07-declare-the-replication-facts-in-the-factories.md)
**Board items:** F1–F4 · **Closes:** nothing ·
**Narrows:** `[00-the-generic-datastore-layer-hard-codes-project-schema-facts]` (rows 1, 2, 4–8) ·
**Opens:** nothing expected · **Model:** Opus, as the prompt recommends.

**Gate:**
- 06 landed (`19f07ee`) and was reviewed (`c6bddf8`).
- 07 is written, in the commit that adds this note's sibling prompt. The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means `Datastore/`, `config/`, `main.py`, `extract_*.py`, `RunRegistry/` or
    `tools/`.
  - If a later commit touches code, re-check every fact below that it could move, and hand the
    agent the corrections as an addendum.
- One prompt at a time in this checkout.

## 0. What makes this prompt unusual

**It rewrites the licence of the check at open, and every repair it makes must stay the same.**
- The hard-coded `BackgroundModel` unit, the `redshift`/`wavenumber` flags and the
  `BackgroundModel` validated rule become facts that factories declare.
- `_prune_unit` and the copy unit become one function.
- The version object is built by the version table's factory instead of by importing
  `MetadataConcepts`.

None of this may move a repair, a refusal, a printed line or a record of the check at open. The
measurement recorded all of them and found them byte-identical. The review repeats that (check 8).

**One choice of method was the user's.** On 2026-10-06, taken as recommended (README §6.2):
- **the child declares its owner**: `"owner_column": "model_serial"` on `BackgroundModelValue`;
- tag tables are recognised through the contract's tag table, without a declaration.

`_prune_unit`'s literal rule was rejected. It gives `BackgroundModel` its right unit, but it puts
`BackgroundModel` in `redshift`'s unit and `wavenumber_exit_time` in `wavenumber`'s (M3).

**What moves on purpose:**
- **Every schema record of a class with a table carries three more keys**: `owner_column`,
  `monotone_flags` and `validated_column`. That fails both witness tests of `test_schema_builder`
  (78 failures), so 07 adds a new witness. Its diff from the 09c witness is those keys alone
  (`measure-07/out/C_witness_diff.txt`).
- **`build_schema` gains refusals** of a declaration that does not fit its table. The current
  registry meets none.
- **`MetadataConcepts` is no longer loaded** by the layer (M5).

**What 07 does not change, and why the prompt says so.**
- **Messages that use BackgroundModel's vocabulary keep it**: "holds its model row", "part of this
  model", "from its own value rows" and "its value rows have serials". These are not registry
  names, and changing them would change a message.
- **The prune refusal keeps its old unit order.** No single order keeps both it and the as-unit
  refusals (M7); the prompt says so.

**The test plan was measured, not read.** On 2026-10-06 at `c6bddf8`, an Opus subagent made a
rough version of the change in a scratch worktree, ran every suite and took M1–M9. It then removed
the worktree. Its first run was cut short by a usage limit and resumed with its state intact. The
rough diff, the scripts and the outputs are committed beside this note in
[`measure-07/`](measure-07/), with scratch paths shown as `<scratch>`.
- **Phase A, code only.** Datastore ran 664, with 78 failures, all in the two witness tests. Every
  other suite passed, apart from ComputeTargets' known wall-clock flake.
- **Phase B.** `test_layer_registry`'s `FORBIDDEN_PACKAGES` gains `"MetadataConcepts"`.
- **Phase C.** The new witness was added, and the 11-test sketch of `test_declared_facts.py`. The
  final counts:
  - Datastore 675 (133 + 151 + 200 + 191);
  - RunRegistry 280 (1 skipped in the worktree; see below);
  - ComputeTargets 693;
  - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped).

**The prompt departs from the rough diff in seven places**, so the agent's diff will not match
`rough_change.diff`, and need not:
- **`_find_version_row` adds no refusal.** The rough version raised if the factory's `store_id`
  differed from the serial read. That cannot be reached, and it would be a new refusal.
- **`build_schema` refuses more.** It refuses a `monotone_flags` given as a `str`, which the rough
  version split into characters. It also refuses a flag or validated column that is not
  `Boolean`. The rough version checked only that the column exists.
- **The variables `models` and `model_column` are renamed.** The rough version kept them. No text
  changes.
- **The schema witness's docstring** gains the new witness. The rough version did not update it.
- **`schema.py`'s module docstring** says it imports `Datastore.contract`. The rough version left
  "imports only `sqlalchemy`".
- **The new tests go beyond the sketch**:
  - the `str` and non-`Boolean` refusals;
  - a hand-built registry showing that a declared owner joins a unit and an undeclared foreign key
    does not;
  - the prune refusal's text, through M7's `AFTER DELETE` trigger;
  - a text check of `ShardedPool.py`'s comments.
- **Breakage (v)** asks whether any `test_replicated_write` test catches `owned_serials` returning
  `None`. The measurement did not run it.

**A test outside the plan that fails is therefore a real signal.** The agent stops and reports.

**The facts, checked by the orchestrator on 2026-10-06 at `c6bddf8`.** Pass them on.
- **The prompt's line numbers** were checked against the tree.
- **Callers** [grep, M-facts]. Only `ShardedPool.py` calls `_prune_unit` (`:2094`),
  `_value_serials` (`:3281`, `:3290`), `_recompute_validated` (`:1881`) and `_find_version_row`
  (`:342`, `:487`). No test or `docs/` script does.
  - `test_version_row_at_open.py:605` and `:620` call `_plan_replicated_repair(record, specs,
    snapshots, differences, conns)`. Its signature stays, so the unit's facts travel in the specs.
- **`read_records(…, validated=…)`.** No test passes it. The seven factories at the prompt's lines
  do.
- **Stand-in factories** (`test_schema_builder.py:62`, `:70`;
  `docs/datastore-integrity-audit/tq_fk_probe.py:42`) have `register()` only. They need no hook,
  because the hooks are not abstract.
- **The compared tables** are the 13 replicated classes and `BackgroundModel_tags`. Only
  `BackgroundModel` prunes at open (M3).
- **The declared columns** are `sqla.Boolean`: `BackgroundModel.validated`, and `source` and
  `response` on `redshift` and `wavenumber`.
- **§3.2's greps.**
  - At `c6bddf8` the second grep prints 14 lines.
  - On the rough change it prints five: `schema.py`'s registration key and prepended column,
    `Datastore.py`'s payload key, and the inventory's class list.
  - The first and third print nothing on the rough change (orchestrator, applied to an export).
- **Not touched, and staying:**
  - `store_reader.py`, `ClientPool.py`, `config/`, `ProfileAgent`, `SerialPoolBroker`,
    `replication.py`, `shard_paths.py` and the closed-store operations;
  - the inventory's class list, cosmology reference and docstring (08's);
  - every `docs/` script;
  - every store under `var/`.
- **Expected counts after 07:**
  - Datastore: 664 plus the new tests, which the log names; the sketch had 11;
  - RunRegistry: 280; ComputeTargets: 693; every other suite unchanged.
- **The index:** 107 now. 07 closes nothing and opens nothing expected, so 107 after, plus one for
  each issue it opens.

**What the review exists to establish.**
- **(E1) The layer names no project table, column or package outside its contract.**
  - `ShardedPool.py` names none, in code, comments or messages.
  - `MetadataConcepts` is not imported or loaded.
  - The actor, `schema.py` and the inventory's tag and validated reads name only the contract.
- **(E2) The facts are declared, once each.**
  - Exactly four factories declare the three keys, and `build_schema` carries and checks them.
  - `revalidate` and `owned_serials` are hooks, and `BackgroundModel` implements them by its
    existing rule.
  - `_unit_tables` replaces both units.
- **(E3) Nothing a store holds, and nothing the check at open does, changes.**
  - The fingerprint is byte-identical.
  - The recorded check at open is byte-identical.
  - Every changed message is byte-identical.
  - `main.py --inventory` is byte-identical.
  - The witness differs by the three keys alone.
- **(E4) The test plan was followed.**
  - Only `test_schema_builder` and `test_layer_registry` are edited, as the plan names.
  - No test is deleted.
  - Every new method is in `test_declared_facts.py`.
- **(E5) The pins.** (i)–(vii) each fail a named test.
- **(E6) The records.**
  - The generic-layer issue is narrowed (rows 1, 2, 4–8), and the index is still 107 (or more).
  - README §4 names what 07 shipped.
  - This board has the §1 row, F1–F4 and §5.

**Conventions, carried from 01–06.** All of these bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, and show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the diff and the tests show.
- **The SHA.** The agent writes "this commit", and the orchestrator substitutes the SHA in its
  follow-up.
- **Suites.**
  - Use README §7's command, write the output to a file, and grep the verdict from it.
  - Run each suite **in the foreground**, serially, within the turn.
  - Run the Datastore suite in module groups (`-p 'test_[a-f]*.py'`, `[g-o]`, `[p-r]`, `[s-z]`),
    and add the counts. A Bash call is capped at ten minutes.
  - Under load, four tests are known to fail on wall clock or chance:
    - `test_tk_wkb_phase.TestCost.test_wall_time_per_object`;
    - `test_background_tau.test_build_cost`;
    - `test_numeric_phase_cut.test_the_print_is_gated_both_ways`;
    - LiouvilleGreen's `test_3bessel_analytic.test_JJJ`.

    Re-run a failure alone before calling it real. If it persists alone, settle it on an export of
    the tree before the prompt.
  - In a git worktree, RunRegistry reports one skip (`test_pipeline_adoption`, whose manifest is
    under the gitignored `var/`). In this checkout it does not.
- **Breakage diffs** are recorded so that `git apply` replays them from a clean tree at the agent's
  commit. Check each with `git apply --check` and `-R`, and replay in `bash`, not zsh.
- **Dead helpers** that only the changed code or tests used go in the same commit, after a grep
  (IMPLEMENTATION CHOICE). This includes an import left unused.
- **Scratch.** The agent uses its own subdirectory of the session scratchpad, never `/tmp`.
  Python's `tempfile` in tests is fine.
  - A scratch script never names a helper `inventory`: a repo-wide grep test forbids
    `def inventory(` in any tracked file (prompt 06's review).
- **Ray.** 07 starts no Ray. ChamPBH's cluster, if running, is left alone.
- **Stop conditions are real.** A stop goes to the user through the orchestrator, and the agent
  resumes with `SendMessage`.
- **Other sessions** may land commits on this branch. Review against the agent's commit, and check
  `git log` before measuring at `HEAD`.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` must be empty.
   - `git worktree list` must show this checkout alone.
   - `git log --oneline c6bddf8..HEAD` must touch no code; otherwise see the gate.
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`.
3. **Ray.** Record `pgrep -lf 'gcs_server|raylet|ray::'`. No listed process may come from this
   repository's `venv`.
4. **Baselines.**
   - First check that `./venv/bin/python -c 'import scipy.sparse.linalg'` succeeds.
   - The last measurement was at `c6bddf8` (06's review, its follow-up):
     - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped);
     - ComputeTargets 693; RunRegistry 280;
     - Datastore 664 (122 + 151 + 200 + 191).
   - Re-measure all six at `HEAD`, serially, with the load average, and give the agent the
     numbers.
   - **Run the full Datastore suite before dispatch**, so that a committed measurement file
     tripping a repo-wide grep test shows at once. One did at 06 (`154d868`).
5. **The index.** It is 107 now, and 107 after, plus any issue 07 opens. Tell the agent both.
6. **The pre-images**, for checks 1–4. Record `git rev-parse HEAD:<path>` for:
   - `Datastore/SQL/ShardedPool.py`, `Datastore.py`, `schema.py` and `ObjectFactories/base.py`, and
     `Datastore/store_inventory.py`;
   - the nine factories of the prompt's §6;
   - `Datastore/tests/test_schema_builder.py` and `test_layer_registry.py`.

   Export `HEAD` to the scratchpad (`git archive HEAD | tar -x -C …`, with the venv symlinked).
   There, before dispatch, take the parent side of check 8:
   - the M1 fingerprint;
   - the M2 record;
   - `main.py --inventory` on a `build_full_store` primary.

## 2. Dispatch

Dispatch one fresh-context subagent, on **Opus**. Give it:
- the prompt;
- the campaign README, the board (including the review of 06), 06's log, and the audit's C1, C2
  and D;
- `HEAD`, the baselines and the index counts;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

It may also be given copies of `measure-07/rough_change.diff` and `measure-07/scripts/`, as a
worked example. Tell it that the prompt, not the rough diff, governs, and where they differ (§0
above).

Tell it plainly:
- **One commit, at the end.** Its log is `logs/07-declare-the-replication-facts-in-the-factories.md`,
  in README §5.1's form, with the prompt's §7 additions.
- **The files it may add or change are the prompt's §6 list, and nothing else.** In particular, it
  must not touch:
  - anything under `docs/` other than `OPEN_ISSUES.md`;
  - `orchestrator/` or `var/`;
  - any factory not in §6;
  - any test but `test_schema_builder`, `test_layer_registry` and the new module;
  - in `store_inventory.py`, anything but `read_records`' tag and validated reads and its import.
- **Stop and ask** on any of the prompt's §5 conditions. A test outside the plan that fails is
  current behaviour moving, never a reason to edit it.
- **Run the Datastore suite in four groups**, in the foreground.

## 3. The review — ten checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only the prompt's §6 list.
   - The diff over these is empty: `docs/` (but `OPEN_ISSUES.md`), `store_reader.py`,
     `ClientPool.py`, `replication.py`, `shard_paths.py`, `ProfileAgent.py`, `SerialPoolBroker.py`,
     `config/`, `main.py`, `extract_*.py`, `RunRegistry/`, `tools/`, `var/` and `orchestrator/`.
   - The old witness file is unchanged.
2. **E1, by grep and by running.**
   - The prompt's §3.2 greps print what it says, and nothing more.
   - `test_layer_registry`'s import test passes with `MetadataConcepts` forbidden.
   - Run `measure-07/scripts/m6_names_left.py` (adapted) at the agent's commit. Every remaining
     hit is contract, layer vocabulary or generic English, as the log classifies it.
3. **E1, E2, by `ast`, against the pre-images**, comparing each function with its docstring
   stripped.
   - **`ShardedPool.py`:**
     - changed: `__init__`, `_find_version_row`, `_open_read_only`, `_version_labels_present`,
       `_replicated_table_specs`, `_describe_differences`, `_plan_replicated_repair`,
       `_recompute_validated`, `_prune_shard` and `_store_impl_replicated_table`;
     - gone: `_prune_unit` and `_value_serials`;
     - new: `_unit_tables` and its helpers;
     - every other function dumps identically, and the closed-store operations especially.
   - **`Datastore.py`:** only `_insert` changes; `_refuse_insert` changes in its docstring only.
   - **`schema.py`:** `build_schema` changes by the foreign key's text and the three keys, and its
     checking helpers are new. `drop_order`, `schema_differences` and the rest dump identically.
   - **`store_inventory.py`:** only `read_records` changes.
   - **The factories:**
     - in the six with one argument changed, only `inventory_records`;
     - in `redshift` and `wavenumber`, only `register`;
     - in `BackgroundModel.py`:
       - `register` (both factories) changes;
       - `validate` keeps its rule, through the shared helper;
       - `revalidate`, `owned_serials` and the helper are new;
       - `inventory_records` changes by its one argument;
       - everything else dumps identically.
   - **In-process**, on the real registry:
     - exactly the four declarations are non-default;
     - `_unit_tables` equals M3's table for every compared table;
     - each `build_schema` refusal of the prompt raises.
4. **E4, the tests, by `ast`.**
   - List the test methods of `test_schema_builder` and `test_layer_registry` before and after:
     none gone, and only `WITNESS`, the docstring, the comment and `FORBIDDEN_PACKAGES` change.
   - Every new method is in `test_declared_facts.py`, and the log names its purpose.
5. **E2, by reading and running.** Read `test_declared_facts.py`.
   - Each refusal test asserts the class and key named.
   - The `revalidate` test compares return, flag and printed warning.
   - The version test checks the factory was used and the object's `vars()`.
   - Run the module, `test_schema_builder` and `test_layer_registry`.
6. **E3, by running.** Run by name:
   - `test_reconcile_at_open`, `test_prune_at_open`, `test_version_row_at_open`,
     `test_replicated_write` and `test_read_only_pool`;
   - `test_store_inventory`, `test_inventory_report`, `test_store_schema` and `test_store_reader`;
   - RunRegistry's `test_store_fingerprint`.
7. **E5, the pins.** In `bash`, replay each breakage diff (i)–(vii) with `git apply`, run the named
   tests and see them fail. Revert each with `git apply -R`, and check that `git status` is clean.
   For (v), read what the log says about `test_replicated_write`.
8. **The probe, against the fix.** From the scratchpad, at the agent's commit and at the export of
   `HEAD~1`:
   - **the fingerprint** of `build_full_store`, with `taken` removed, is byte-identical. Give the
     digest; it should be `6576b85d…`. `build_real_store`'s inventory is byte-identical too;
   - **the check at open**: `measure-07/scripts/m2_open_behaviour.py` (adapted) gives
     byte-identical records at both trees. Give the counts of tests, constructions and printed
     lines;
   - **the messages**: `measure-07/scripts/m7_messages.py` (adapted) gives byte-identical texts
     for every scenario, the prune refusal included;
   - **`main.py --inventory`** on a `build_full_store` primary: byte-identical stdout and stderr,
     exit 0;
   - **`revalidate`**: `measure-07/scripts/m9_revalidate.py` (adapted) agrees with `validate`;
   - **imports**: `m5_imports.py` at the agent's commit loads no `MetadataConcepts` module from any
     of the five layer modules;
   - **the witness**: `measure-07/scripts/witness_diff.py` shows the three keys alone.
9. **E6, the records.**
   - The generic-layer issue has a dated "Narrowed" line naming rows 1, 2, 4–8, and its index hook
     says so.
   - The index is 107 (or more) by counting, matching the header.
   - README §4 names the keys, the hooks and `Datastore/contract.py`.
   - This board has the §1 row, F1–F4 and §5.
10. **Suites.** Run all six, serially, at `HEAD`.
    - Datastore must be 664 plus the log's new tests, and RunRegistry 280.
    - Every other count is unchanged.
    - `black --check` is clean on every touched Python file.

## 4. After the review

- **Record it on this board** as a §1 *Orchestrator review of prompt 07* paragraph, in the form
  of 01's–06's. Fix small residue in a follow-up commit of its own. That includes:
  - "this commit" → the SHA;
  - this note marked "used" in the board's list;
  - the board header's count.
- **Report to the user:**
  - the facts now declared, and by which factories;
  - what the layer imports now (`MetadataConcepts` gone);
  - that the check at open, the prune and the version row behave identically (check 8), and the
    fingerprint digest;
  - the new witness and why;
  - the tests edited and added.
- **What 07 hands on.** 08's note must carry:
  - what is left in `store_inventory.py` for 08: rows 12 and 13 (the class list and its order, and
    the cosmology reference);
  - `read_records` now reads the tag table and the validated column through the contract and the
    factory's argument;
  - `schema.drop_order` and the foreign keys are available to derive the class order;
  - the BackgroundModel vocabulary kept in `ShardedPool`'s messages ("model row", "value rows"),
    which 09's guard must allow;
  - whatever 07's log hands on.
