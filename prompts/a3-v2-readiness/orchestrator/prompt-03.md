# Orchestrator — prompt 03, the read-only pool and the readers

Read [`../README.md`](../README.md) first: §0, §1, §4, §5 and §6.1 D3. Then read
[`prompt-02.md`](prompt-02.md) §0 and this board's *Orchestrator review of prompt 02*. Together
they are the conventions this note does not repeat.

**You do not write code.** You may:
- run the suites and the tests the log names;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process patch from the session scratchpad to check a claim. Never commit one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`03-open-readers-read-only.md`](../03-open-readers-read-only.md)
**Board items:** O1–O4 · **Closes:** `[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]`
and `[10-extraction-scripts-exit-1-on-a-store-without-every-model]`, both on the
`datastore-integrity` board's §4 · **Model:** Opus.
**Gate:** 01 and 02 landed and reviewed (`2a44f7e`, `b647c0b`; reviews `1caecf8`, `f1999b1`).
`HEAD` is the commit that adds this note, or a later one that touches no code. One prompt at a
time in this checkout: 04, 05 and 06 may land before or after 03, but not while it runs.

## 0. What makes this prompt unusual

**It adds a second mode to `ShardedPool` and `Datastore`, and a mode like that is only worth
having if nothing escapes it.** A read-only pool that writes once, anywhere (at open, in a
lookup, in a check at open that was not told), is worse than none: it says the reader did not
write. The review therefore has to see **every file's SHA-256 unchanged** across the readers'
whole lookup sequence, with the instrument shown first to count a write. It also has to see the
read-write path **unchanged**, by the existing suite.

**The prompt was amended with this note** (read-first item 4, O1 twice, O4, breakage (v)). The
orchestrator's reading at `f1999b1` found four places where 02's result, or the code, changes
what the prompt as first written assumed. All four are below. The review checks the agent
against the amended prompt.

**The facts, checked by the orchestrator on 2026-10-03 at `f1999b1`.** Pass them on; the agent
need not rediscover them. Each is marked as audit 00 marks its findings.

- **What 02 changed at open** [read]. `ShardedPool.__init__` on an existing store now runs, in
  order:
  1. `_read_shard_data`;
  2. `_check_shard_files`;
  3. `_reconcile_replicated_tables`;
  4. `replication_in_flight.create(checkfirst=True)`;
  5. `_prune_replicated_tables` under `prune_unvalidated`;
  6. the broker, then every actor, then `read_largest_store_ids` and the broker notification;
  7. the `read_table_config` check;
  8. `_find_version_row`, writing through `_get_impl_replicated_table` only when the label is
     absent;
  9. `set_version` on every actor.

  M1's table predates this. Its row "the version insert in `Datastore.__init__`" is now
  steps 8–9. The prune row is now step 5, and 03 refuses `prune_unvalidated=True` at construction
  before step 1.
- **The `prune` record is completed before the comparison** [read: `_reconcile_replicated_tables`].
  - After step 1's read-write `sqlite3.connect`, the function reads the record. A `prune` record
    returns to `_complete_interrupted_prune`, which deletes rows, before any table is compared.
  - So a read-only refusal of "a record of any operation" must come **before that branch**.
    Otherwise a `prune` record is met by the `mode=ro` file's own `attempt to write a readonly
    database`, which does not name the record.
  - The prompt now says so (O1, amended).
- **The engines the check at open uses** [read].
  - `_reconcile_engine` builds `sqlite:///{path}`, read-write. Under `read_only=True` every engine
    it makes must be `mode=ro`.
  - `_read_in_flight_record` uses the pool's own engine (`_create_engine`, also read-write
    today). It already reads a missing `replication_in_flight` table as no record, so no DDL is
    needed.
  - `_read_shard_data` reads the primary inside `self._engine.begin()`. That works on a `mode=ro`
    engine; it is the read-write open that would roll a hot journal back.
- **`_find_version_row` is already a read-only read** [read]. It opens each shard with
  `sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)` and `SELECT serial, label FROM version
  WHERE label = ?`. It raises if the shards disagree, and returns `None` when no shard holds the
  label.
  - A read-only pool can call it and refuse on `None`, never reaching
    `_get_impl_replicated_table`. It needs only the shard files, so it may run before any actor
    exists. Where it runs is the *prompt's choice*.
  - **Every reader and `main.py` open under `VERSION_LABEL = "2025.1.1"`**, so a read-only reader
    finds the label on any store `main.py` built. The stand-in opens under `"standin"`.
- **`set_version` is not needed by a read-only actor, but is harmless** [read]. Without it, `_insert` refuses any
  versioned insert with a `RuntimeError`, ahead of the serial lease. But a read-only actor's
  inserters raise `ReadOnlyMiss` first, so `_insert` is never reached from a lookup. Whether to
  call it is the *prompt's choice*.
- **`store_reader` cannot be imported at module scope by either pool file** [read]. Its first
  lines import `ShardedPool` from `Datastore.SQL.ShardedPool` and `_factories` from
  `Datastore.SQL.Datastore`, so a module-level import back is circular.
  - The journal *rule* already lives in the pool: `ShardedPool._journal_paths`, which
    `store_reader._refuse_journals` calls. `read_only_url` is a one-line f-string.
  - A function-scope import of `Datastore.store_reader` reuses both with no edit to it.
  - `store_reader.py` is **not** on 03's file list (§2 below). If the agent finds it must edit it,
    that is a stop-and-ask (prompt §5, fourth condition), not a deviation to log.
- **The actor's engine and DDL** [read: `Datastore.__init__`, `_create_engine`].
  - On a missing file the constructor `mkdir`s and creates tables. On an existing one it runs
    `_drop_actions`, `_ensure_tables` and `_validate_on_startup`. `_create_engine` builds
    `sqlite:///{file}`, read-write.
  - `object_get` hands `factory.build` **both** `inserter=` and `inserters=`, the whole map,
    which a factory uses for nested rows (a `BackgroundModel`'s tags and values). A read-only
    actor must replace **every** entry, not only the class's own.
- **`_validate_on_startup` would fail on a missing table** [read]. Seven factories validate at
  startup: `BackgroundModel`, `GkNumericIntegration`, `GkWKBIntegration`, `TkNumericIntegration`,
  `TkWKBIntegration`, `QuadSource` and `GkSource`.
  - Each `validate_on_startup` selects from its table, and deletes only under `prune=True`. The
    report-only path writes nothing (read, all seven).
  - With no `_ensure_tables`, a shard lacking one of those tables raises `no such table` at open.
  - M1's measured missing-table case was `OneLoopIntegral`, which does not validate, so M1 did not
    see this. The prompt now says to skip and report such a class (O1, amended).
- **`_assign_shard_keys` is on the replicated `get` path** [read: `_get_impl_replicated_table`].
  After `_replicated_write`, a `get` of the shard-key class (`wavenumber`) calls
  `_assign_shard_keys`, which inserts into the primary's `shard_keys` for an unseen `store_id`.
  - On a hit against a checked store it returns early. The check at open compares `shard_keys`
    against the wavenumbers every shard holds.
  - A read-only replicated `get` that bypasses `_replicated_write` must keep that call, or its
    equivalent, raising `ReadOnlyWrite` where it would insert (prompt O2). It must not drop it.
  - The readers `read_table` `wavenumber` and do not `object_get` it (audit R1). The stand-in's
    `get_wavenumber` does `object_get` it.
- **`read_table` already picks one shard** [read]. It swaps a `random.randrange` draw to the end
  and pops it. "Chosen as `read_table` chooses" means that draw. Under the stand-in, the draw goes
  through `_StandinRandom`, so `cluster.controller` pins which shard answers a read-only `get`. A
  test can then make one shard's copy differ and see that shard's answer.
- **The backstop arrives as SQLAlchemy's exception** [read; measured, audit M2c]. A write
  through a `mode=ro` engine raises `sqlalchemy.exc.OperationalError` whose `.orig` is
  `sqlite3.OperationalError: attempt to write a readonly database`.
  - `Datastore.object_get` catches `SQLAlchemyError`, prints and re-raises.
  - The translation to `ReadOnlyWrite` must match on the SQLAlchemy wrapper (or its `.orig`), and
    must not swallow other `OperationalError`s, such as `no such table`.
- **A hot journal fails a `mode=ro` read** with `SQLITE_READONLY_ROLLBACK` (M2d). That is why it
  is refused by presence first. An empty journal is not refused by SQLite at all.
- **Breakage (v) is invisible to a checksum** [read]. Journals are refused by presence before
  step 1. A read-write `sqlite3.connect` and `SELECT` on a shard with no journal writes nothing.
  So the test that pins (v) must observe the connection: for example, patch
  `Datastore.SQL.ShardedPool.sqlite3.connect` and assert that every call under `read_only=True`
  carries `mode=ro`. The prompt now says so.
  - (iii) is different. A read-write actor engine lets the backstop test's write succeed, so that
    test fails without `ReadOnlyWrite`.
- **`--models`** [read: `docs/gktk-remedial/scoped_pipeline_run.py`].
  - `scoped_pipeline_run.py` filters `build_model_list`'s result to the given labels, raises only
    when none match, and drops an unknown label silently.
  - The prompt's rule, that each unknown label is an error, is stricter and governs; the prompt
    now says so. The reader filter is a pure function in `extract_common.py`, beside
    `add_run_selection_argument`.
- **The six scripts** [read]. Each script:
  - calls `ray.init(address=args.ray_address)` at module scope, so it cannot be imported;
  - constructs `ShardedPool(version_label=VERSION_LABEL, …, prune_unvalidated=False,
    read_table_config=read_table_config)` with no `drop_actions`;
  - then calls `resolve_run_selection(pool, …)` and `build_model_list(pool, units)`, and loops
    `for model_data in model_list:`;
  - raises `RuntimeError("Could not locate suitable background model instance in the
    datastore")` from a per-model function: `extract_GkSource_data.py` near its line 782,
    `extract_QuadSourceIntegral_data.py` near 1076, and similarly in the other four.

  `available_run_labels` reads the store through `read_inventory`, so through `store_reader`,
  while the pool is open. Under a read-only pool that is still sound.
- **Two existing tests read the six scripts with `ast`, and must pass unchanged** [read]. 01's and
  02's reviews carried both.
  - `ComputeTargets/tests/test_main_plumbing.py` (`EXTRACT_SCRIPTS`,
    `ReaderToleranceAgreementTestCase`) compares each reader's tolerance lookups with `main.py`.
  - `ComputeTargets/tests/test_quadrature_tolerance_pin.py`'s second class reads
    `extract_QuadSourceIntegral_data.py`'s `quad_atol`/`quad_rtol` bindings and its
    `object_read_batch` call.

  Moving a lookup into a helper can break either. Leaving each script's lookups where they are
  breaks neither.
- **The instrument** [read: `docs/a3-v2-readiness/reader_writes_probe.py`]. `qsi_sequence(pool,
  rec, units)` and `build_full_store(primary, cluster)` are module-level functions behind a
  `__main__` guard.
  - Importing the probe by `importlib` path runs nothing.
  - Whether the test imports them or transcribes them is the *prompt's choice*. Importing them
    makes a probe under `docs/` a test dependency. Transcribing does not, so the log says which
    and why.
- **The stand-in** [read]. `StandinCluster.open_pool(primary, shards=3, **kwargs)` passes
  `read_only=True` through to `ShardedPool`, and `store_checksums(primary)` is the SHA-256
  instrument. `standin_pool.py` may gain additions only, each listed in the log, as in 02.
- **Nothing in the tree breaks on a new keyword** [read].
  - `test_inventory_retired` inspects `ShardedPool.__init__`'s parameters for two names, and a
    `read_only` keyword does not touch either.
  - `SerialPoolManager` is inert with `broker=None` (`ClientPool.py`). A read-only pool that
    skips the broker can pass `None`.
- **The two issues** sit in the `datastore-integrity` board's §3.3, `[10-extraction-scripts-…]`
  and `[08-extraction-script-…]`, each with its *Widened* line. Both close to that board's §4.
  The index is **104** at `f1999b1`; after 03 it is **102**.

**What the review exists to establish.**
- **(D1) A read-only pool writes nothing.** QSI's whole sequence leaves every file byte-identical,
  with the instrument shown first to count a read-write pool's commits.
- **(D2) At open, M1's table and nothing else.** Each row is implemented where the log says. A
  record of any operation, `prune` included, is refused by name and before 02's completion
  branch. Journals are refused by presence. A missing primary creates nothing. An absent label
  refuses. A missing table, including a validating class's, opens.
- **(D3) Every write is stopped, before SQLite where it can be.** `ReadOnlyMiss` for the five
  insert-on-miss classes. `ReadOnlyWrite` for every other write, and for the backstop.
- **(D4) The read-write path is unchanged.** The Datastore suite's 574 pass untouched, and so do
  `test_main_plumbing` and `test_quadrature_tolerance_pin`.
- **(D5) The readers.** Each opens read-only and takes `--models`. A model without a background is
  skipped, and if every model is skipped the script exits non-zero.
- **(D6) The pins.** Breakages (i)–(vii) replay with `git apply`, and each fails a test the log
  names. (v) is caught by a test that observes the connection.
- **(D7) The records.** Both issues are closed on the `datastore-integrity` board with the tests
  named, and the index is 102.

**Carried from the reviews of 00–02.** All bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, never `git add -A`, `.` or `commit -a`.
  Show `git diff --cached --name-only` before committing. Count, then paste. A log states only
  what the diff and the tests show. Cite the pool files by function name, never by line.
- **The SHA, not "this commit".** 02's records said "this commit" in five places, and the review
  replaced them with the SHA. The agent cannot know its own SHA before committing, so this stays
  the orchestrator's residue to fix. The brief need not ask the impossible.
- **Suites.** README §7's command, output to a file and the verdict grepped from it. Under load
  `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, `test_background_tau.test_build_cost`
  and `test_numeric_phase_cut.test_the_print_is_gated_both_ways` fail.
  - Re-run a failure alone before calling it real.
  - On 2026-10-03 `test_wall_time_per_object` also failed alone, at a load of 11–17. It failed the
    same way on an export of the tree before 02, which settled it. A failure that persists alone is
    settled the same way: run it on an export of the tree before the prompt.
- **Breakage diffs** are recorded so that `git apply` replays them from a clean tree at the agent's
  commit. 02's log did this well.
- **Replay them in `bash`, not `zsh`.** The review of 02 first paired each breakage with the wrong
  test, because zsh arrays start at 1.
- **Residue** found in review is the orchestrator's, fixed in a follow-up commit (the user,
  2026-09-29).
- **Scratch.** The agent works in its own subdirectory of the session scratchpad, never its top
  level.
- **Usage limits.** Resume a stopped agent with `SendMessage`, saying exactly where it stopped. If
  it commits but sends no report, review from the commit and its log.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` are both empty, and `git worktree list` shows
     this checkout alone.
   - `git log --oneline f1999b1..HEAD` touches no code. If it does, re-check §0's facts that it
     touches. 04, 05 and 06 edit `quadsource_atol_sweep.py`, `scoped_pipeline_run.py` and
     `RunRegistry/stores.py`. None of those is on 03's list, but `scoped_pipeline_run.py` is §0's
     source for `--models`.
2. **The registry.** `./venv/bin/python -m RunRegistry list`: nothing `running`. At `f1999b1`
   there were 12 runs, 10 finished and 2 `unknown` (`a3-pilot`, `a3-pilot-resume`). Report any
   change, without acting on it. 03 launches nothing and opens nothing under `var/`.
3. **Baselines.** Measured 00:59–01:11 on 2026-10-03 at `b647c0b`'s code (`f1999b1` changes
   records only), board §5.5:

   | Suite | Count |
   |---|---|
   | AdaptiveLevin | 32 |
   | ComputeTargets | 669 (`test_wall_time_per_object` flaking) |
   | CosmologyModels | 39 |
   | Datastore | 574 |
   | LiouvilleGreen | 148 (1 skipped) |
   | RunRegistry | 199 |

   Re-measure all six on `HEAD`, serially, and record the load average. Check what is loading the
   machine before you start. On 2026-10-02 it was an unrelated Ray job in another of the user's
   projects. Leave such a job alone, and tell the agent not to touch any Ray process.
4. **The index.** `grep -c '^| \`\[' docs/OPEN_ISSUES.md`. It is **104** at `f1999b1`. After 03
   it is 102. Tell the agent both numbers, and that it is one more for each issue it opens.
5. **The pre-images**, for check 1: `git rev-parse` at `HEAD` of `Datastore/SQL/ShardedPool.py`,
   `Datastore/SQL/Datastore.py`, `Datastore/replication.py`, `Datastore/store_reader.py`,
   `Datastore/tests/standin_pool.py`, `extract_common.py` and the six `extract_*.py`.

## 2. Dispatch

One fresh-context subagent, on **Opus**. Give it:
- the prompt, as amended;
- the campaign README, the audit, this board and prompt 02's log;
- `HEAD`, the baselines and both index counts;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/03-open-readers-read-only.md`, in README §5.1's
  form, with the prompt's §7 additions: M1's table with a column naming the function that
  implements each row, and the full text of each new exception's message as raised in a test.
  M1's table predates 02. The log's copy gains a row for each of 02's writes at open
  (`_prune_replicated_tables`, `_complete_interrupted_prune`, the version write), and the log
  says so.
- **The files it may add or change:**
  - `Datastore/SQL/ShardedPool.py`, `Datastore/SQL/Datastore.py` and `Datastore/replication.py`;
  - the six `extract_*.py` scripts and `extract_common.py`;
  - new test modules under `Datastore/tests/` and `ComputeTargets/tests/`;
  - `Datastore/tests/standin_pool.py`, **additions only**, each listed in the log;
  - the log; this board; the `datastore-integrity` board (both issues to its §4);
    `docs/OPEN_ISSUES.md`.

  **Nothing else.** In particular:
  - no `Datastore/store_reader.py` (§0: a function-scope import reuses it; if that cannot work,
    stop and ask);
  - no `config/model_list.py`, `config/sharding.py`, `main.py`, driver or factory;
  - no probe and no committed probe output;
  - nothing else under `docs/`, nothing under `var/`, nothing under `orchestrator/`;
  - no existing test.
- **Stop and ask** on any of the prompt's §5 conditions.
  - Say that the read-write path must not change at all. A diff to a read-write branch that is
    "equivalent" is still a change, and the Datastore suite's 574 must pass untouched.
  - Say that 02's version row and prune are not to be reworked. 03 only adds the read-only branch
    beside them.

## 3. The review — ten checks

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only §2's list.
   - `git diff HEAD~1 HEAD -- Datastore/store_reader.py config/ main.py docs/ var/ Datastore/SQL/ObjectFactories/ ':!docs/OPEN_ISSUES.md'`
     is empty.
   - No existing test module is in the diff. `standin_pool.py`'s numstat shows no deletion.
2. **(D4) The read-write path, by reading.**
   - Read the pool's and the actor's diffs. Every change is either new code reached only under
     `read_only` or a new parameter defaulting to `False`.
   - `_replicated_write`, `_plan_replicated_repair`, `_prune_replicated_tables`,
     `_complete_interrupted_prune` and `_find_version_row` have empty diffs. If one does not, read
     it and say why it changed.
3. **(D1) Nothing written.**
   - The full-store test compares `store_checksums` before and after QSI's whole sequence.
   - Before reporting the read-only zero, the same sequence on a read-write pool changes the
     primary.
   - Run that test yourself.
4. **(D2) At open.**
   - Walk the log's M1 table against the diff, row by row. Each row names its function, and that
     function does what the row says.
   - Read `_reconcile_replicated_tables`: under `read_only`, a record of any operation refuses
     **before** the `prune` branch.
   - Read the actor: `_validate_on_startup` skips and reports a validating class whose table the
     shard lacks.
   - The test for a missing table uses a **validating** class (`GkSource`, say), not only one like
     `OneLoopIntegral`. If it uses only a non-validating class, the amended O1 is untested. That is
     residue to fix: a test of your own in a follow-up.
5. **(D3) Every write stopped.**
   - `ReadOnlyMiss` is raised for each of `store_tag`, `LambdaCDM`, `QCD_Cosmology`, `tolerance`
     and `GkSourcePolicy`, with no file changed.
   - Every entry of `inserters=` is replaced, not only `inserter=`. A miss on a nested row shows it.
   - `ReadOnlyWrite` is raised for each of store, validate, a new shard key, `drop_actions`,
     `prune_unvalidated=True` and the backstop.
   - The backstop matches `attempt to write a readonly database` alone. A `no such table` error
     is not turned into `ReadOnlyWrite`.
6. **(D5) The readers.**
   - Grep the six: `read_only=True`, no `prune_unvalidated`, one `--models` argument added through
     `extract_common`, and one call to the skip helper per model loop.
   - The `RuntimeError("Could not locate suitable background model…")` text is gone, or is no
     longer raised for an absent background.
   - `test_main_plumbing` and `test_quadrature_tolerance_pin` pass unchanged. Run them.
7. **(D6) The pins.** In `bash`, replay each of the log's breakage diffs (i)–(vii) with
   `git apply`, run the named test, and see it fail. Revert with `git apply -R`, and check that
   `git status` is clean after each. For (v), read the test and see that it observes the
   connection, not a checksum.
8. **An end-to-end look, in-process.**
   - From the scratchpad, open a stand-in read-only pool on a store from `build_full_store`, and
     run `qsi_sequence` against it.
   - Then delete the `GkSourcePolicy` row from every shard (`delete_rows`), and run it again.
   - The first run leaves every file unchanged. The second raises `ReadOnlyMiss` naming
     `GkSourcePolicy`, with every file unchanged. Never commit this.
9. **(D7) The records.**
   - Both issues are in the `datastore-integrity` board's §4, each with a closing note naming the
     tests.
   - Both index rows are gone, and the count is 102 by counting, matching the header.
   - The log's *Observations* name the issues opened, if any.
10. **Suites.** All six serially, at `HEAD`.
    - Datastore is 574 plus 03's new Datastore tests, and ComputeTargets is 669 plus its new ones.
      Every other count is unchanged.
    - `black --check` is clean on every touched Python file.

## 4. After the review

- **Record it on this board**, after the prompt-02 review and in its form, with `HEAD` at
  dispatch, the checks and any remnants. Fix small residue in a follow-up commit of its own,
  including "this commit" → the SHA.
- **Report to the user:**
  - what a read-only open does, row by row against M1, and what it refuses;
  - that a reader's whole sequence now leaves every file byte-identical, by which test;
  - what `--models` and the skip do on a LambdaCDM-only store, which is A3 v2's shape;
  - which issues closed, and what remains between the campaign and 07.
- **What 03 hands on.** 07's L3 runs every reader read-only against the rehearsal's resumed store.
  Its note must carry:
  - the exceptions' names and messages, and the `--models` invocation the rehearsal uses;
  - that a reader on a store with a journal beside it is refused: run readers only after the build
    has exited;
  - that the readers open under `VERSION_LABEL = "2025.1.1"`, the build's.
