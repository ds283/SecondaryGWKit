# Orchestrator — prompt 05, refuse a store whose schema differs

Read [`../README.md`](../README.md) §0.1, §1, §3, §4, §5, §6.1 (D2) and §6.2 (U1). Then the audit's
B5 and B10 ([`docs/datastore-generic-audit.md`](../../../docs/datastore-generic-audit.md)), and the
board's review of 04 (§1). The conventions are [`prompt-04.md`](prompt-04.md)'s, carried below.

**You do not write code.** You may:
- run the suites and the tests the log names;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process check from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`05-refuse-a-store-whose-schema-differs.md`](../05-refuse-a-store-whose-schema-differs.md)
**Board items:** S1–S3 · **Closes:** `[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]`
· **Widens:** `[04-the-shard-key-audit-crashes-where-it-should-refuse]` (a "Found" line) ·
**Opens:** nothing expected · **Model:** Opus, as the prompt recommends.
**Gate:**
- 04 landed (`bbdba04`) and reviewed (`bc3e944`); 05 written (`f24f180`); the tree clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code. If a later commit
  touches `Datastore/`, `RunRegistry/`, `tools/` or the extraction scripts, re-check every fact
  below that it could move before dispatch, and hand the agent the corrections as an addendum.
- One prompt at a time in this checkout.

## 0. What makes this prompt unusual

**It changes what opens, on every path, and for a current store nothing.** The reader, and so the
inventory, the fingerprint and `extract_common`, as well as the read-only pool and the read-write
check at open, all start refusing a store whose files differ from what the code declares. Every
store the current code writes has no difference:
- measured on every shard and the primary of `build_real_store` and `build_full_store`;
- measured on a store the real `ShardedPool` created on stand-in shards.

The fingerprint of `build_full_store` was byte-identical before and after a rough version of the
change (digest `6576b85d…`, problems `{}`).

**One behaviour moves on purpose, for a shape the current code can reach.** A store left by an
interrupted drop has a shard that lacks a table. Today the read-only pool opens it and reports
the gap (`test_read_only_pool` 829, 850); after 05 it refuses it by name. Under U1 only a
read-write open recovers such a store, and that recovery is unchanged.

**U1 had no test.** No existing test opens a store read-write while a shard lacks a table. The
measurement ran the three cases the prompt names, through `standin_pool`, and all three opened:
- the `gk-source` tables dropped from shard 0;
- the replicated `GkSourcePolicy` dropped from shard 1 and empty elsewhere;
- one shard lacking half its tables, with no version row anywhere.

Afterwards every shard matched the schema and held the version row. The prompt's new tests are
the first coverage of U1, and breakage (ii) pins them.

**The primary check goes beyond the board's original S1–S3 wording.** Prompt 03's review handed
05 the question whether its one refusal should cover the primary's own tables, because 03 left a
primary without `replication_in_flight` meeting a bare `OperationalError`. Writing 05 answered it
as follows (board item S3, widened):
- both constructors compare the primary with `_create_engine`'s six tables;
- `_read_closed_store` and the closed-store operations do not. They read only the `shards` table,
  and `write_hand_built_primary` builds a primary without `replication_in_flight` for them on
  purpose (04).

So the reader and the fingerprint read such a primary, and the pool refuses it. That is
intended. **If the user objects to the primary check, strike it from the prompt before dispatch**:
it is S3's last clause, one changed test (`test_a_primary_without_the_record_table_is_refused`
then keeps `OperationalError`), the new primary tests and breakage (iv).

**The test plan was measured, not read.** On 2026-10-05 at `bc3e944`, a read-only Opus subagent
made a rough version of the change in a scratch worktree, ran Datastore, RunRegistry and
ComputeTargets, and removed the worktree. The rough diff, its scripts and their outputs are
committed beside this note in [`measure-05/`](measure-05/). The scripts import
`schema_differences` and so run only with `rough_change.diff` applied; `m6_u1.py` writes into
`measure-05/tmp`.
- **Datastore:** Ran 622, **failures=6, errors=23** (636 − 15 + 1: `test_store_reader`'s 15
  tests collapse into one import error, because `build_old_store` is gone). Twenty-four distinct
  methods in eight modules failed, plus that module. Every one is in the prompt's test plan with
  its fate:
  - nine deletions;
  - three conversions to a refusal;
  - two re-fixtures, both measured (`scripts/m5_refixtures.py`);
  - the changed assertions and constructions.
- **`test_store_reader`, run against a shim** that supplied the deleted fixtures
  (`scripts/run_store_reader_shim.py`): 7 of its 15 tests still failed, and the prompt names the
  edit each needs.
- **RunRegistry:** 280, OK. **ComputeTargets:** 693, OK.
- **No read-write test moved.** Every user of `standin_pool` passed, apart from the named tests.
  So did the interrupted-first-open test (`test_version_row_at_open.TestNewStoreInterrupted`), the
  interrupted-drop test (`test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_store`)
  and `test_wkb_numeric_parent_key.TestDropOrder`.
- **The measured messages** (`out/m2.txt`) were the rough version's wording. The prompt fixes only
  what they must name.
- **`python -m RunRegistry store fingerprint`** on a store missing a table (`out/m3_*.txt`):
  - before: exit 0 and a full fingerprint, with `problems: … absent-table 1`;
  - after: exit 1, with the refusal on stderr through the CLI's `except RuntimeError`
    (`RunRegistry/__main__.py:297`). No traceback, and nothing written.

**A test outside the plan failing is therefore a real signal.** It means current behaviour moved.
The agent stops and reports.

**The facts, checked by the orchestrator on 2026-10-05 at `bc3e944`.** Code is unchanged since;
`f24f180` and this note add only prompts and the measurement record. Pass them on. Every line
number in the prompt was re-read against the tree; these need care:
- **`Datastore/store_reader.py`** [read]:
  - the "Old stores." paragraph is `:25-30`;
  - `_describe_shard` is `:112-150`, and `open_read_only`'s `try` begins at `:184`.
- **`Datastore/SQL/ShardedPool.py`** [read]:
  - the read-write constructor calls `_create_engine()` at `:216` and `_read_shard_data()` at
    `:217`;
  - the read-only constructor makes its mode=ro engine at `:430` and calls `_read_shard_data()`
    at `:433`;
  - the reconcile's step 1 is `:1000-1016`, `built` is made at `:1018`, and the record is read at
    `:1022`;
  - the `columns` kind is at `:1294`, `:1412` and `:1560`;
  - the prune's kept `has_table` skips are at `:1977` and `:1998`.
- **`Datastore/SQL/Datastore.py`** [read]:
  - the read-only actor's absent-table block is at `:348-357`, `_skipped_validation` at `:359`,
    and the skip at `:375-382`;
  - `read_only_state` is at `:461-469`;
  - `self._inspector` (`:336`) is on the read-only path only.
- **`Datastore/store_inventory.py`** [read]: the branches at `:512-520`, `:522`, `:530-540`,
  `:542-573`, `:591-603` and `:644-656`; `comparable` at `:310`, `:499`, `:505`, `:585`, `:819`
  and `:849-855`.
- **The four selects** [read]: `TkWKBIntegration.py:1243-1245` and `:1363-1365`, and
  `GkWKBIntegration.py:1236-1238` and `:1358-1360`. Nothing reads the selected value.
  `test_factory_select_columns`' `KNOWN_BLIND` entries name `query.subquery().c.numeric_serial`
  in the integration factories' `build`s (prompt 07 of `datastore-integrity`). That is a
  different site, which stays.
- **The tests** [the test names were checked with `ast` at the cited lines]:
  - `test_store_inventory` `:874`–`:952`;
  - `test_gksource_parent_set:902`;
  - `test_read_only_pool:829`, `:850`;
  - `test_store_reader:200`, `:233`;
  - `test_inventory_consumers:271`, `test_inventory_report:236`;
  - `test_database_errors_reach_the_caller:416`, `:444`, `:460`, `:476`, `:492`;
  - `test_reconcile_at_open:780` and its only `OperationalError` import at `:46`.
- **`tools/shard_key_audit.py:178-215`** [measured by the subagent]: when shard 0 lacks the
  key-type table it skips the cross-file check and can end "VERDICT: OK". The prompt does not
  touch it, and has the agent add it to `[04-the-shard-key-audit-crashes-where-it-should-refuse]`.
- **Not touched, and staying:**
  - `Datastore._ensure_tables` and `_drop_actions`' `has_table` skip (B10, U1);
  - `ProfileAgent.py:122-125`;
  - the closed-store operations;
  - `RunRegistry/` code;
  - every store under `var/`.
- **Expected counts after 05.**
  - **Datastore:** 636 − 9, plus the new tests. The log names each new test.
  - **RunRegistry:** 280, unchanged, with no RunRegistry test edited.
  - **ComputeTargets:** 693, and every other suite, unchanged.
- **The index.** **109** at `bc3e944`, matching the header. 05 closes one issue, so it is **108**
  after, plus one for each issue 05 opens.

**What the review exists to establish.**
- **(E1) One comparison, one exception, and the tolerances gone.**
  - `schema_differences` and `StoreSchemaMismatch(RuntimeError)` are in `schema.py`, which
    imports only `sqlalchemy` and names no project table.
  - The reader calls them, and so do the reconcile and both constructors' primary check.
  - The reader's four fields, the inventory's two problems and `comparable`, the read-only pool's
    and actor's absent-table reporting, `unkeyed`, the `columns` kind, `_read_full_rows`' filter,
    the four selects and `build_old_store` are gone.
  - Nothing else in those files changes, docstrings aside.
- **(E2) The refusals hold at every entry point.**
  - The reader, `read_inventory`, the read-only pool and the read-write open each refuse an
    absent column, an extra column and an extra table, by name.
  - All but the read-write open refuse an absent table.
  - Both constructors refuse a primary that differs.
  - No file changes on a refusal, and no actor is built.
- **(E3) U1 holds, and nothing a current store does changes.**
  - The three U1 cases open read-write and leave every shard matching the schema.
  - Every test outside the plan passes untouched.
  - The fingerprint of `build_full_store` is byte-identical.
- **(E4) The test plan was followed:**
  - exactly the nine deletions;
  - each conversion keeps its shape and asserts the refusal;
  - each re-fixture changes only the rows it builds;
  - each changed assertion or construction is the named one.
- **(E5) The pins.** (i)–(v) each fail a named test.
- **(E6) The records.**
  - The fallbacks issue is closed into §4 with a dated line, and its index row is deleted.
  - The shard-key-audit issue carries a "Found (prompt 05)" line, with its hook updated.
  - The index is 108 (or more).
  - The board has the §1 row, S1–S3 and §5.

**Conventions, carried from 01–04.** All of these bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only. Show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the diff and the tests show.
- **The SHA.** The agent writes "this commit"; the orchestrator substitutes the SHA in its
  follow-up.
- **Suites.** Use README §7's command, write the output to a file, and grep the verdict from it.
  Run each suite **in the foreground**, serially, within the turn.
  - The Datastore suite takes 16–30 minutes under load. A Bash call is capped at ten minutes, so
    run it in module groups (`-p 'test_[a-f]*.py'`, `[g-o]`, `[p-r]`, `[s-z]`) and add the counts.
    The four groups cover every module.
  - Under load, three tests fail on wall clock:
    - `test_tk_wkb_phase.TestCost.test_wall_time_per_object`;
    - `test_background_tau.test_build_cost`;
    - `test_numeric_phase_cut.test_the_print_is_gated_both_ways`.

    Re-run a failure alone before calling it real. If it persists alone, settle it on an export of
    the tree before the prompt. At 04's review `test_wall_time_per_object` failed alone at load
    23–36, both before and after.
- **Breakage diffs** are recorded so that `git apply` replays them from a clean tree at the agent's
  commit. Check each with `git apply --check` and `-R`, and replay in `bash`, not zsh: zsh does not
  split an unquoted test list, as 04's review found.
- **Dead helpers** that only deleted tests used go in the same commit, after a grep (IMPLEMENTATION
  CHOICE). This includes an import left unused.
- **Scratch.** The agent uses its own subdirectory of the session scratchpad, never `/tmp`. Python's
  `tempfile` in tests is fine.
- **Ray.** 05 starts no Ray. ChamPBH's cluster (`gcs_server` 62525, `raylet` 62574, from its own
  venv) is left alone. `standin_pool` needs no Ray.
- **Stop conditions are real.** A stop goes to the user through the orchestrator, and the agent
  resumes with `SendMessage`.
- **Other sessions** may land commits on this branch. Review against the agent's commit, and check
  `git log` before measuring at `HEAD`.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` must be empty.
   - `git worktree list` must show this checkout alone.
   - `git log --oneline bc3e944..HEAD` must touch no code (otherwise see the gate).
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`. At
   `bc3e944` it held 16 runs: 14 finished, 2 `unknown`.
3. **Ray.** Record `pgrep -lf 'gcs_server|raylet|ray::'`. No listed process may come from this
   repository's `venv`.
4. **Baselines.** Last measured at `bbdba04` (04's review, check 10):
   - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped);
   - ComputeTargets 693 (2 wall-clock failures);
   - Datastore 636, RunRegistry 280.

   Re-measure all six at `HEAD`, serially, with the load average, and give the agent the numbers.
5. **The index.** 109 now; 108 after, plus any issue 05 opens. Tell the agent both.
6. **The pre-images**, for checks 1–4. Record `git rev-parse HEAD:<path>` for:
   - `Datastore/SQL/schema.py`, `Datastore/store_reader.py`, `Datastore/store_inventory.py`,
     `Datastore/SQL/ShardedPool.py`, `Datastore/SQL/Datastore.py`;
   - the two WKB factories and `Datastore/tests/real_store_fixtures.py`;
   - every test module of the prompt's test plan.

   Export `HEAD` to the scratchpad (`git archive HEAD | tar -x -C …`) for check 8.
7. **The primary check.** If the user has not seen it (§0), say so in one line before dispatch.

## 2. Dispatch

Dispatch one fresh-context subagent, on **Opus**. Give it:
- the prompt;
- the campaign README, the board (including the review of 04), 04's log, and the audit's B5 and
  B10;
- `HEAD`, the baselines and the index counts;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad. The
  agent does not read `orchestrator/`. It may be given `measure-05/scripts/m6_u1.py` and
  `m5_refixtures.py` as copies, as worked examples of the U1 cases and the two re-fixtures.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/05-refuse-a-store-whose-schema-differs.md`, in
  README §5.1's form, with the prompt's §7 additions.
- **The files it may add or change are the prompt's §6 list, and nothing else.** In particular, it
  must not touch:
  - `RunRegistry/`, `tools/`, `main.py` or the extraction scripts;
  - in `Datastore.py`, anything outside the read-only actor;
  - in the WKB factories, anything but the four selects;
  - any test not named in the prompt's test plan;
  - anything under `var/`, `docs/` (other than the index) or `orchestrator/`.
- **Stop and ask** on any of the prompt's §5 conditions. A test outside the plan failing is
  current behaviour moving, never a reason to edit it.
- **The Datastore suite in four groups**, in the foreground.

## 3. The review — ten checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only the prompt's §6 list.
   - The diff over these is empty: `RunRegistry/`, `tools/`, `main.py`, `extract_*.py`,
     `config/`, `Datastore/shard_paths.py`, `Datastore/replication.py`, `var/` and
     `orchestrator/`.
   - In `ObjectFactories/`, only the two WKB files change. Under `docs/`, only `OPEN_ISSUES.md`.
2. **E1, by grep.** The prompt's §3.2 greps print nothing. `ReadOnlyShard`'s fields are exactly
   `serial`, `path`, `engine` and `tables`.
3. **E1, by `ast`, against the pre-images.** Compare each function with its docstring stripped.
   - `schema.py`: `build_schema` dumps identically; `schema_differences` and the exception are
     new. Read the exception: one prefix, the verb, the file, each difference by name.
   - `store_reader.py`: `_describe_shard` and `open_read_only` change.
   - `store_inventory.py`: `read_records`, `_combine` and `read_inventory` change, plus
     `ShardContext` and `ShardRead`.
   - `ShardedPool.py`: `__init__` (the primary check), `_open_read_only`,
     `_version_labels_present`, `_reconcile_replicated_tables` (the check after step 1),
     `_read_replicated_tables`, `_replicated_differences`, `_describe_differences`,
     `_plan_replicated_repair`, `_read_full_rows`, and any helper the checks are factored into.
     Every other function dumps identically: `_prune_needed`, `_prune_shard` and the
     closed-store operations especially.
   - `Datastore.py`: only the read-only branch of `__init__`, `_validate_on_startup_read_only` and
     `read_only_state` change.
   - The WKB factories: each of the four functions differs only by the one selected column.
4. **E4, the tests, by `ast`.** List the test methods of every module of the plan, before and
   after, and of the new module.
   - Exactly the nine of the plan are gone.
   - The three conversions keep their store shape and assert `StoreSchemaMismatch`. The log names
     any rename.
   - The two re-fixtures differ only in the rows they build.
   - Every other changed method is one the plan names, changed as the plan says.
   - Every new method belongs to `test_store_schema.py`, and the log names its purpose.
5. **E2, by reading and running.** Read `test_store_schema.py`. Each refusal asserts the
   exception, the names, an unchanged file state, and for a pool that no actor was constructed.
   The U1 tests assert that the store opens and that every shard matches the schema afterwards.
   Run the module, the conversions and the changed tests by name.
6. **E3, by running.** Run by name: the U1 tests the plan lists as must-pass, `test_reconcile_at_open`,
   `test_version_row_at_open`, `test_prune_at_open`, `test_replicated_write`, the rest of
   `test_read_only_pool`, `test_copy_move_store`, `test_delete_store` and
   `test_absolute_shard_record_refused`.
7. **E5, the pins.** In `bash`, replay each breakage diff (i)–(v) with `git apply`, run the named
   tests and see them fail. Revert each with `git apply -R`, and check that `git status` is clean.
8. **The probe, against the fix.** From the scratchpad, at the agent's commit and at the export of
   `HEAD~1`:
   - **the fingerprint** of `build_full_store`, with `taken` removed, is byte-identical. Give the
     digest; it should be `6576b85d…`;
   - **a current store** (`build_real_store`) is read by `open_read_only` and `read_inventory`
     with the same rows and problems at both trees;
   - **a store missing a table** (`missing_tables={1: ["OneLoopIntegral_tags"]}`):
     - at `HEAD~1`, read, with the problem `absent-table`;
     - at the agent's commit, refused by `open_read_only`, `read_inventory` and
       `python -m RunRegistry store fingerprint … --runs-root <scratch>` (exit 1, message on
       stderr, no traceback, nothing written);
   - **a store with an extra column**: read at `HEAD~1` (reported in `extra_columns`), refused
     at the agent's commit;
   - **the read-write U1 case**: run `measure-05/scripts/m6_u1.py`'s three cases against the
     agent's commit, adapted to its names. All three open.
9. **E6, the records.**
   - The fallbacks issue is in §4 with a dated "Closed (…, prompt 05)" line naming 05's
     deletions, and its index row is gone.
   - The shard-key-audit issue has its "Found (prompt 05)" line, and its index hook is updated.
   - The index is 108 (or more) by counting, matching the header.
   - The board has the §1 row, S1–S3 and §5.
10. **Suites.** All six, serially at `HEAD`:
    - Datastore and RunRegistry at §0's expected counts, with each new test named in the log;
    - every other count unchanged.

    `black --check` is clean on every touched Python file.

## 4. After the review

- **Record it on this board** as a §1 *Orchestrator review of prompt 05* paragraph, in the form
  of 01's–04's. Fix small residue in a follow-up commit of its own. That includes:
  - "this commit" → the SHA;
  - this note marked "used" in the board's list;
  - the board header's count.
- **Report to the user:**
  - what an old store now meets, entry point by entry point;
  - what a current store sees (check 8: nothing changes; the fingerprint digest);
  - the one deliberate move (a read-only open of an interrupted drop);
  - U1's first tests;
  - the tests deleted, converted, re-fixtured, changed and added;
  - the fallbacks issue closed: the end of D2's deletions, 01–05.
- **What 05 hands on.** 06's note must carry:
  - `schema_differences`' callers take `built.tables` from `build_schema(…, _factories)`, which
    the reader and `ShardedPool` import. That is exactly what 06 replaces with an injected
    registry, so every call site of the check is one of 06's;
  - the primary's six tables are `ShardedPool`'s own, the layer's contract, and not the
    registry's;
  - whatever 05's log hands on.
