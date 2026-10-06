# Orchestrator — prompt 09, keep the layer generic

Read [`../README.md`](../README.md) §0.1, §1, §3 (its one exception), §4, §5 and §6.2 (the
user's four choices of method for 09, taken on 2026-10-06). Then read:
- the audit's C and C2 ([`docs/datastore-generic-audit.md`](../../../docs/datastore-generic-audit.md));
- the board's review of 08 (§1), with its "Handed on" list.

The conventions are [`prompt-08.md`](prompt-08.md)'s, carried below.

**You do not write code.** You may:
- run the suites and the tests the log names;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process check or a probe from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4);
- **run the one rehearsal build of README §3's exception**, after the review (§4.2).

**The prompt:** [`09-keep-the-layer-generic.md`](../09-keep-the-layer-generic.md)
**Board items:** K1–K2 · **Closes:**
`[00-the-generic-datastore-layer-hard-codes-project-schema-facts]` · **Narrows:** nothing ·
**Opens:** nothing expected · **Model:** Sonnet, as the prompt recommends.

**Gate:**
- 08 landed (`6646b06`) and was reviewed (`3ff1c80`).
- 09 is written (`c1d1bd3`), in the commit before this note's. The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means `Datastore/`, `RayTools/`, `config/`, `utilities.py`, `main.py`,
    `extract_*.py`, `RunRegistry/`, `ComputeTargets/`, `tools/` or `docs/handover/`.
  - If a later commit touches code, re-check every fact below that it could move, and hand the
    agent the corrections as an addendum.
- One prompt at a time in this checkout.

## 0. What makes this prompt unusual

**It is small, and it closes the campaign.**
- The agent adds one test module, deletes five tests, and makes three edits in the layer: one
  local renamed and two docstring sentences. No behaviour of any code changes.
- The closing verification of README §5 rule 9 has two halves. The **fixture half** is the
  agent's. The **rebuild** is yours, after the review: README §3's one exception (§4.2 below).
- The campaign is closed only when the rebuild's fingerprint matches. Until then the board says
  09 landed and K2 is half done.

**Four choices of method were the user's** (2026-10-06, README §6.2). One was not the
recommendation:
- **`config.defaults` is allow-listed.** The recommendation was a constant of the layer's own in
  `Datastore/contract.py`; the user chose the allow-list. Do not reopen it.
- **`utilities` is allow-listed**, as recommended.
- **07's `TestNoProjectNameInThePool` and 08's two per-file tests are folded into the guard and
  deleted**, as recommended. The union of their vocabularies is checked in every file.
- **Rule 9 is closed on fixtures and by one rebuild.** The user first asked what the rebuild adds
  and whether it was worth 6 CPUs and 13 minutes. The answer recorded in §6.2: the fixtures write
  with SQL, so only a build exercises the write path, which 01–03 and 07 touched. The user then
  took it.

**What moves on purpose:**
- Five tests are deleted, by name:
  - `test_declared_facts.TestNoProjectNameInThePool`'s three methods, with the helpers and imports
    only they used;
  - `test_inventory_declarations.TestTheLayerKnowsNoProject.test_it_imports_no_project_package`
    and `test_no_project_name_in_its_code_or_text`, with their unused imports.
- Eight tests are added, all in `test_layer_is_generic.py`.
- `store_inventory.read_records`' local `parent_serial` becomes `parent`.
- `replication.same_value`'s docstring and `shard_paths.py`'s module docstring change one
  sentence each.
- The generic-layer issue closes. The index goes from 108 to 107.

**The test plan was measured, not read.** On 2026-10-06 at `3ff1c80`, an Opus subagent made a
rough version of the change in a scratch worktree, ran every suite, replayed eight mutations, and
took the fixture half of rule 9. It then removed the worktree. The rough diff, the scripts and the
outputs are committed beside this note in [`measure-09/`](measure-09/). Scratch paths are shown as
`<scratch>`; the SQLite stores, and two copies of `main.py` the dry probe made, are left out (the
latter as their diff).
- **The guard on today's layer: 10 hits at three sites**, none of them project knowledge:
  - `store_inventory.py`'s local `parent_serial`, 8 lines;
  - `replication.py:289`'s "tolerance";
  - `shard_paths.py:12`'s `tools/`.

  On the `dbd8a51` layer it finds 64 code, 209 string, 23 comment and 23 import hits.
- **`layer_files()`: 17 files**: the audit's 12, `contract.py`, `ObjectFactories/base.py` and
  three `__init__.py`. The vocabulary has 234 words.
- **Suites with the rough change:**
  - Datastore 720 = 147 + 182 + 200 + 191;
  - RunRegistry 280 (1 skipped, in a worktree);
  - ComputeTargets 693, AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped).
- **Mutations (i)–(viii)** each failed the guard (`measure-09/out/m3_mutation_results.txt`).
- **Rule 9, fixtures, `dbd8a51` against the rough tree:**
  - (a) 6 files, 720 schema entries, 164 tables and 286 rows: 0 differences;
  - (b) the fingerprints of a `dbd8a51`-written store at both trees are byte-identical,
    `6576b85d…` (full) and `d9328538…` (real), and so are the inventories;
  - (c) a read-write open through each tree's stand-in pool is identical, and the pool's
    constructor arguments differ as 06 made them (`kwargs DIFFERS` in `m4c_compare.txt` is that).
- **M5:**
  - the sweep script's `--help` is byte-identical at both trees;
  - its diff touches only `assert_store_is_self_consistent`, which a fresh `--build` never calls;
  - the scope, the `main.py` argv and the four substitutions are identical;
  - `--job-name` reaches no stored row;
  - the fingerprint format is 6 at both trees.

**The prompt departs from the rough diff in four places**, so the agent's diff will not match
`rough_change.diff`, and need not:
- **A new `test_the_allow_list`**, so the module has 8 tests and Datastore is **721**
  (147 + 183 + 200 + 191), not 720.
- **`shard_paths.py` names no client file.** The rough version still named `shard_key_audit.py`.
- **Breakage (ix)**, for the new test.
- **The records**: the rough version made none.

**A test outside the plan that fails is therefore a real signal.** The agent stops and reports.

**The facts, checked by the orchestrator on 2026-10-06 at `c1d1bd3`** (the code is `3ff1c80`'s).
Pass them on.
- **The prompt's line numbers** were checked against the tree.
- **The deleted names** (`TestNoProjectNameInThePool`, `_registry_names`, `_FLAG_PHRASES`,
  `_LAYER_WORDS`, `_named`, `_WORD`, `_PROJECT_PACKAGES`, the two `test_inventory_declarations`
  methods) are used only in their own modules, outside `prompts/`.
- **`TestTheLayerKnowsNoProject` stays**, with its two load tests, `PROJECT_PACKAGES` and `_run`.
- **The repo-wide grep test** (`test_inventory_retired`) does not match anything in
  `measure-09/`.
- **The layer's imports outside it** are exactly `config.defaults` (`ShardedPool.py:35`,
  `ProfileAgent.py:10`) and `utilities` (`Datastore.py:20`, `ProfileAgent.py:11`,
  `RayWorkPool.py:9`).
- **Not touched, and staying:** every module of the layer except the three edits; `config/`,
  `utilities.py`, every factory, `main.py`, `tools/`, every `docs/` script, and every store under
  `var/`.
- **Expected counts after 09:**
  - Datastore 721 (147 + 183 + 200 + 191);
  - RunRegistry 280; ComputeTargets 693; every other suite unchanged.
- **The index:** 108 now, 107 after.

**What the review exists to establish.**
- **(E1) The guard covers the layer and passes.**
  - Its files are derived (17), and a new module is checked.
  - Its vocabulary is the union of the two deleted tests'.
  - Its allow-list is §2's and no more, each entry with a reason.
  - On the parent's layer it finds exactly the 10 hits.
- **(E2) The deletions are exactly the five tests**, with only their own helpers and imports.
  Nothing else in the two modules changes but their docstrings.
- **(E3) No behaviour changes.**
  - `read_records` differs from its pre-image only by the rename.
  - `replication.py` and `shard_paths.py` differ only in docstrings.
- **(E4) Rule 9 holds on fixtures**, (a)–(c), at the agent's commit against `dbd8a51`.
- **(E5) The pins.** (i)–(ix) each fail a named test.
- **(E6) The records.** The issue is closed in §4 and gone from the index, which counts 107. README
  §4 names what 09 shipped. This board has the §1 row, K1–K2 and §5.
- **(E7) Rule 9 holds end to end**: the rebuild's fingerprint equals `39809dca…` (§4.2).

**Conventions, carried from 01–08.** All of these bind the agent, and the brief says so.
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
  commit. Check each with `git apply --check` and `-R`, and replay in `bash`, not zsh. Mutation
  (vii) adds a file: its diff creates it, and `-R` deletes it.
- **Dead helpers** that only the changed code or tests used go in the same commit, after a grep
  (IMPLEMENTATION CHOICE). This includes an import left unused.
- **Scratch.** The agent uses its own subdirectory of the session scratchpad, never `/tmp`.
  Python's `tempfile` in tests is fine.
  - A scratch script never names a helper `inventory`: a repo-wide grep test forbids
    `def inventory(` in any tracked file (prompt 06's review).
  - **No scratch `.py` file goes under `Datastore/` or `RayTools/`**, even for a moment, except
    mutation (vii): the guard globs the filesystem and would scan it.
- **Ray.** The agent starts no Ray. ChamPBH's cluster, if running, is left alone. The rebuild
  (§4.2) is yours.
- **Stop conditions are real.** A stop goes to the user through the orchestrator, and the agent
  resumes with `SendMessage`.
- **Other sessions** may land commits on this branch. Review against the agent's commit, and check
  `git log` before measuring at `HEAD`.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` must be empty.
   - `git worktree list` must show this checkout alone.
   - `git log --oneline 3ff1c80..HEAD` must touch no code; otherwise see the gate.
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`.
3. **Ray.** Record `pgrep -lf 'gcs_server|raylet|ray::'`. No listed process may come from this
   repository's `venv`.
4. **Baselines.**
   - First check that `./venv/bin/python -c 'import scipy.sparse.linalg'` succeeds.
   - The last measurement was at `6646b06` (08's review), and `3ff1c80` added one test:
     - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped);
     - ComputeTargets 693; RunRegistry 280;
     - Datastore 718 (150 + 177 + 200 + 191).
   - Re-measure all six at `HEAD`, serially, with the load average, and give the agent the
     numbers.
   - **Run the full Datastore suite before dispatch**, so that a committed measurement file
     tripping a repo-wide grep test shows at once.
5. **The index.** It is 108 now, and 107 after. Tell the agent both.
6. **The pre-images**, for checks 3–4. Record `git rev-parse HEAD:<path>` for
   `Datastore/store_inventory.py`, `Datastore/replication.py`, `Datastore/shard_paths.py`,
   `Datastore/tests/test_declared_facts.py` and `Datastore/tests/test_inventory_declarations.py`.

   Export `HEAD` and `dbd8a51` to the scratchpad (`git archive <sha> | tar -x -C …`, with the venv
   symlinked). At the `dbd8a51` export, take the parent side of check 8 before dispatch:
   - build `build_full_store` and `build_real_store` with `dbd8a51`'s code
     (`measure-09/scripts/m4_build.py`), keep the stores, and record their checksums;
   - their fingerprints (`taken` removed) and whole inventories, read at `dbd8a51`. The digests
     should be `6576b85d…` and `d9328538…`;
   - the `sqlite_master` and row dumps of both, built at `dbd8a51` (`m4_sqlite_dump.py`);
   - a read-write open of a copy of the full store through `dbd8a51`'s stand-in pool
     (`m4c_open_rw.py`).

## 2. Dispatch

Dispatch one fresh-context subagent, on **Sonnet**. Give it:
- the prompt;
- the campaign README, the board (including the review of 08), 08's log, and the audit's C and
  C2;
- `HEAD`, the baselines and the index counts;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

It may also be given copies of `measure-09/rough_change.diff`, `measure-09/scripts/` and
`measure-09/out/test_layer_is_generic.py.txt`, as a worked example. Tell it that the prompt, not
the rough diff, governs, and where they differ (§0 above).

Tell it plainly:
- **One commit, at the end.** Its log is `logs/09-keep-the-layer-generic.md`, in README §5.1's
  form, with the prompt's §7 additions.
- **The files it may add or change are the prompt's §6 list, and nothing else.** In particular, it
  must not touch:
  - anything under `docs/` other than `OPEN_ISSUES.md`;
  - `orchestrator/` or `var/`;
  - any test but the two the prompt names and the new module;
  - in the layer, anything but `read_records`' one local and the two docstring sentences.
- **It runs no build and starts no Ray.** The rebuild is the orchestrator's.
- **Stop and ask** on any of the prompt's §5 conditions. A new guard hit is a stop, never an
  allow-list entry. A test outside the plan that fails is current behaviour moving, never a reason
  to edit it.
- **Run the Datastore suite in four groups**, in the foreground.

## 3. The review — ten checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only the prompt's §6 list.
   - The diff over `docs/` (but `OPEN_ISSUES.md`), `config/`, `utilities.py`, `main.py`,
     `extract_*.py`, `RunRegistry/`, `tools/`, `var/`, `orchestrator/`, every factory, and every
     module of the layer but `store_inventory.py`, `replication.py` and `shard_paths.py`, is empty.
2. **E1, by running.**
   - The guard passes at the agent's commit, and `layer_files()` gives the 17 files.
   - Copy the new module into the export of `HEAD~1` and run it there. It reports exactly the 10
     hits of §0, and nothing else.
   - Read the allow-list: the four groups of the prompt's §2, each with its reason.
   - Its vocabulary contains every word of the deleted tests' (compute both from the pre-images).
3. **E3, by `ast` against the pre-images, docstrings stripped.**
   - `store_inventory.py`: only `read_records` changes. With `parent` renamed back to
     `parent_serial`, it dumps identically.
   - `replication.py` and `shard_paths.py`: every function and module statement dumps identically.
4. **E2, the tests, by `ast`.**
   - In `test_declared_facts.py`: gone are exactly `TestNoProjectNameInThePool` and the module
     helpers and imports it alone used. Every other function and class dumps identically.
   - In `test_inventory_declarations.py`: gone are exactly the two methods, and their imports.
     Every other function, `PROJECT_PACKAGES` and `_run` dump identically.
   - Every new method is in `test_layer_is_generic.py`, and the log names it.
5. **E1, by reading.** Read `test_layer_is_generic.py` in full.
   - Each failure message lists `file:line word`.
   - `test_the_allow_list` asserts both acceptances and refusals.
   - No test passes on an empty input.
   - Run the module, `test_declared_facts` and `test_inventory_declarations`.
6. **The deleted tests' coverage.** For each of the five, name the guard test that now catches
   what it caught. Then, at the agent's commit, make one mutation per deleted test and see the
   guard fail. 07's log records no breakage of its name test; 08's is its (vii).
   - `BackgroundModel` in a `ShardedPool.py` message string, and "source or response" in a
     `ShardedPool.py` comment (07's three methods);
   - `from MetadataConcepts import store_tag` in `ShardedPool.py` (07's import method);
   - 08's (vii): a project table in a `store_inventory.py` comment;
   - `import CosmologyModels` in `store_inventory.py` (08's import method).
7. **E5, the pins.** In `bash`, replay each breakage diff (i)–(ix) with `git apply`, run the named
   tests and see them fail. Revert each with `git apply -R`, and check that `git status` is clean,
   with no `Datastore/scratch_module.py` left.
8. **E4, the probe**, from the scratchpad, at the agent's commit against the `dbd8a51` side taken
   before dispatch:
   - (a) `sqlite_master` and rows of both fixture stores, built at the agent's commit: 0
     differences from `dbd8a51`'s;
   - (b) the `dbd8a51`-built stores read at the agent's commit: the fingerprints are byte-identical
     (`6576b85d…`, `d9328538…`), so are the inventories, and the checksums are unchanged;
   - (c) a read-write open of another copy through the agent's commit's stand-in pool: the same
     reconciliation, prune, version object and printed lines as at `dbd8a51`. Nothing raised.
9. **E6, the records.**
   - The generic-layer issue is in §4, with a dated "Closed" line naming the guard, and gone from
     §3 and from the index.
   - The index is 107 by counting, matching the header.
   - README §4 names the guard module, `layer_files()`, `import_allowed` and the allow-list.
   - This board has the §1 row, K1 ✅, K2 "fixture half; the rebuild at the review", and §5.
10. **Suites.** Run all six, serially, at `HEAD`.
    - Datastore must be 721 (147 + 183 + 200 + 191), and RunRegistry 280.
    - Every other count is unchanged.
    - `black --check` is clean on every touched Python file.

## 4. After the review

### 4.1 Record the review

- **Record it on this board** as a §1 *Orchestrator review of prompt 09* paragraph, in the form
  of 01's–08's. Fix small residue in a follow-up commit of its own. That includes:
  - "this commit" → the SHA;
  - this note marked "used" in the board's list.
- The campaign is **not** closed in that commit: K2 waits on the rebuild.

### 4.2 The rebuild — README §3's one exception

This is a long job under `CLAUDE.md`'s run-registry rules, with nothing relaxed.

1. **Before the launch:**
   - `./venv/bin/python -m RunRegistry list`: nothing `running`, and nothing stale that this
     repository owns;
   - `./venv/bin/python -m RunRegistry store show var/datastores/datastore-generic-09-rehearsal.sqlite`:
     kind `absent`, and no runs. If anything exists by that name, stop and ask;
   - `pgrep -lf 'gcs_server|raylet|ray::'`: no Ray process from this repository's venv;
   - `git status` clean, and `HEAD` is the review's follow-up, with no code changed since the
     agent's commit.
2. **The launch**, from the repository root. It is the reference's command (`var/runs/a3-v2-readiness-07-reference-20261004T012909/manifest.json`)
   with a new store, job name, slug, purpose and campaign. `measure-09/out/m5/` shows the scope
   unchanged.
   ```bash
   nohup sh -c 'PYTHONPATH=. ./venv/bin/python -u docs/handover/quadsource_atol_sweep.py \
       --build --rehearsal \
       --database var/datastores/datastore-generic-09-rehearsal.sqlite \
       --job-name datastore-generic-09 --atol 1e-32 --rtol 1e-7 --cpus 6 \
       --register rehearsal \
       --purpose "datastore-generic prompt 09: README section 5 rule 9 closing rehearsal build (3 k over 1e5-1e6/Mpc, zend 250, atol 1e-32 rtol 1e-7) at the campaign final tree; its fingerprint must equal a3-v2-readiness-07-reference (dbd8a51) 39809dca" \
       --campaign datastore-generic --prompt 09; \
     echo "** driver exit status $?"' \
     > var/runs/datastore-generic-09-rehearsal.launch.log 2>&1 &
   disown
   ```
   The launch log sits beside the run directory, not in it, as a3-v2-readiness-07's did
   (`[00-a-launch-log-lives-outside-its-run-directory]`). Do not work round that.
3. **One check that it started.** `RunRegistry list` shows it `running`, with a fresh heartbeat.
   Then **end the turn**. Do not poll. The reference took about 13 minutes at `dbd8a51`.
4. **When it is done:**
   - the launch log ends `** driver exit status 0`, and the run is `done`;
   - `var/runs/datastore-generic-09-rehearsal-<timestamp>/status.json`'s `fingerprint.digest` is
     compared with `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`. The
     per-class counts and digests are compared with the reference's `status.json`;
   - `fingerprint_sidecar` reads "not written: … (there is none)", as the reference's did. Then
     run `store create --purpose …` on the closed primary, so that the user can retire it;
   - after the driver has exited, `pgrep -lf 'gcs_server|raylet|ray::'` shows none of its Ray
     processes. A killed build's agents outlive it by 1–2 minutes
     (`[07-a-killed-builds-ray-agents-outlive-the-driver]`); a `done` one should not. Record what
     you see.
5. **If the digest matches:** K2 is ✅, and the campaign is **closed**. In a commit of its own:
   - this board: the review's record of the rebuild (run id, duration, digest, per-class
     agreement), K2 ✅, and the header "CLOSED — 9 of 9 prompts landed";
   - the README header says so;
   - `docs/OPEN_ISSUES.md`: the campaign's remaining open issues stay. They are
     `[04-closed-store-refusals-repeat-their-prefix]`, `[06-dropping-one-group-…]` and
     `[08-resolve-shows-…]`. Each is a decision for the user, or a cleanup the user will have
     done. Their board is the record.
6. **If it differs:** stop. Name the classes whose digest or count differs, and report to the user
   before anything else. Open an issue on this board's §3, and add it to the index. The campaign
   stays open. Do not rebuild to see whether it changes.
7. **The store.** Tell the user it exists and is theirs to retire:
   `./venv/bin/python -m RunRegistry store retire var/datastores/datastore-generic-09-rehearsal.sqlite --reason "…"`.
   `store retire` is run by a person (`CLAUDE.md`).

### 4.3 Report to the user

- what the guard checks, over which files, and its allow-list;
- the five tests deleted and the eight added;
- the three edits in the layer, and that no behaviour changed;
- rule 9:
  - the fixture half, with its digests;
  - the rebuild's digest against `39809dca…`, or that it is still running;
- the issue closed. Whether the campaign is closed. What remains open on its board, and the store
  to retire.

**Nothing is handed on.** 09 is the campaign's last prompt. Anything the review finds that is
not residue is opened on this board and in the index.
