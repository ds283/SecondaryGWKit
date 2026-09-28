# Orchestrator — prompt 03, exact identity lookups

Read [`../README.md`](../README.md) first: §0.1, §2, §4 and §5, and §6.1 (decision 3 is the user's:
the exit-time lookup becomes an **exact match**). **You do not write code.** You may *replay* a
mutation the log records, with `git apply`, and then revert it.

**The prompt:** [`03-exact-identity-lookups.md`](../03-exact-identity-lookups.md)
**Board items:** L1–L5 · **Closes:** `[02-wavenumber-exit-time-tolerance-is-an-inequality-key]`
(tolerance-convergence), `[02-exit-time-lookup-runs-inside-the-subhorizon-loop]`
(store-fingerprint), `[03-integrationsolver-stepping-minimum-lookup]` (GkTk-remedial) ·
**Narrows:** `[05a-two-verification-scripts-still-query-the-exit-time-under-the-shared-pair]`
(tolerance-convergence) · **Gate:** 02 landed and reviewed (board §1, `631c3f3`). Never dispatched
concurrently with 04–09 in this checkout.

## 0. What makes this prompt unusual

The prompt was written on 2026-09-25, before 01, 02, 04, 04b and 05a–05c landed. Its task is
unchanged. What moved under it is below, measured at `631c3f3`.

- **Every build and every extraction goes through these lookups.** They are the identity lookups
  of the replicated classes, and the exit time is looked up for every `k`. The care is in changing
  the match and nothing else. Production is measured unchanged:
  - `main.py` (`build_k_exit_work`, `:993`) and the six `extract_*.py` scripts request the exit
    time at `DEFAULT_HEXIT_ABS_TOLERANCE`/`DEFAULT_HEXIT_REL_TOLERANCE` = (1e-10, 1e-9), with no
    `stepping`, so `build` asks for 0. Every computed exit time stores `stepping = 0`
    (`CosmologyConcepts/wavenumber.py:738`). So `==` on serials and on stepping hits every row
    production stores.
  - `main.py:3621-3640` requests each `IntegrationSolver` at exactly the stepping it inserts: 0
    for the five `solve_ivp` labels, and `BackgroundModel.TAU_GAUSS_ORDER` and
    `GkWKBIntegration.PHASE_SOLVER_STEPPING` for the two Gauss-order labels.
  - If the agent finds a production caller that would miss, that is prompt §5's first stop.
- **The read-back guard pins the shape of the functions 03 edits.** `test_factory_select_columns.py`
  (04, 04b) asserts by **exact set equality**:
  - `EXPECTED_COVERAGE`, which includes `wavenumber.py:sqla_wavenumber_exit_time_factory.build.row_data`
    and `redshift.py:sqla_redshift_factory.build.row_data`;
  - `KNOWN_BLIND_COLUMNS`, which declares the exit-time build's `table.c[f"z_exit_…"]` loops;
  - `KNOWN_UNCHECKED_INSERTS`, which declares the `inserter` call of every L3 `build`.

  A rewrite that binds a second row variable, drops the f-string loops or reshapes an insert moves
  one of those sets, and the guard's own message then says to edit the test. **03 may not edit it**
  (README §5 rule 7), so that is a §5 stop, not a fix.
  - **A guard-neutral shape exists.** The orchestrator sketched L1–L4 in a throwaway worktree
    and ran Datastore, ComputeTargets and CosmologyModels unmodified: 325, 552 and 39 OK. The
    sketch kept the joins, bound `row_data` once after both loops, and put L4's zero branch
    inside the one filter expression. The sketch is not the agent's to copy, and was not given to
    it. What it shows is that the prompt can be met without touching a test, so a guard failure
    means the shape should change, not the guard.
- **01 put a replica path inside each of these `build()`s.** Every one of them now inserts under
  `payload["serial"]` when the driver sends the controlling shard's serial. Each replica runs the
  same lookup and answers with its own match, and `_replicated_write` checks that answer against
  the controller's.
  - So L1–L4 change what every shard matches, not only the controller. That is as it should be;
    the `if "serial" in payload` branches themselves are not 03's to touch.
  - 01's replica `store()` for the exit time already keys on exactly `atol_serial`, `rtol_serial`
    and `stepping` (`wavenumber.py`, `store`, `key_columns`), as does the inventory's identity.
    After L1, the lookup, the replica check and the fingerprint agree on one key. The log should
    say so.
- **02 makes the shards identical at open, but it cannot see a duplicate inside a shard.** A
  serial-keyed row held twice under two serials, identically on every shard, is not a cross-shard
  difference. L3 is the guard for exactly that case (log 02, "For prompt 03").
  - After L3 such a duplicate raises `MultipleResultsFound` inside each shard's `build()`. Under
    01, that is a failure after `P1`, so the record stays set. Log 02's reopen table (row 18a)
    then clears it over identical shards, and the next lookup raises again.
  - So the result is loud and persistent, and nothing repairs it. That is the intended result. The
    log states this path; a stand-in-pool test of it is the agent's choice, not required.
- **`QCD_Cosmology.build`'s `.scalar()` is inside `try … except SQLAlchemyError`**, and
  `MultipleResultsFound` is a `SQLAlchemyError`. The handler converts only an error naming
  `T_z_representation` and re-raises everything else. So test 4 for `QCD_Cosmology` should see
  `MultipleResultsFound` itself, and the log should say the handler was read.
- **The prompt says D5 is undecided.** It has been decided since (2026-09-26). The restriction
  stands anyway: prompt 06 is the first to change the schema witness. The two registrations'
  `"stepping": "minimum"` are 06's, whose README §2 row says 06 makes them "say what prompt 03
  made true". So 03's log hands them to 06 under "Observations not acted on".
- **Line numbers, re-checked at `631c3f3`.**
  - The eight L3 sites are still on audit §3.8's lines: `tolerance.py:35`, `store_tag.py:26`,
    `version.py:26`, `integration_metadata.py:34`, `GkSourcePolicy.py:39`, `QuadSourcePolicy.py:39`,
    `LambdaCDM.py:55`, `QCD_Cosmology.py:99`.
  - The exit time's filter is at `wavenumber.py:216-228` and the in-loop execute at `:235`.
    `IntegrationSolver`'s `>=` is at `integration_metadata.py:30-34`, and `redshift.py:40`
    divides by `z`.
  - There are 21 other `.scalar()` calls in `ObjectFactories/`. L3 leaves them, and the log
    lists each one judged.
  - None of the eight identity tables, nor `redshift`, has a UNIQUE constraint, so each of test
    4's duplicates can be built.
- **The two verification scripts request the shared pair.** `analyse_greens_and_source.py:447-448`
  and `run_quadsource_integrals.py:140-141` build it from `DEFAULT_ABS_TOLERANCE`/`DEFAULT_REL_TOLERANCE`
  = (1e-10, 1e-8), and look the exit time up at `:247` and `:188`.
  - Today they are served the stored (1e-10, 1e-9) row and report it as their own pair. After L1
    they miss.
  - The L5 table must say what a miss **then does** in each script: an unpopulated object, and
    whether the script goes on to compute, store or fail. That is found by reading them. Do not run
    them: they need Ray and a store under `var/`.
  - They are not edited, and `[05a-…]` is narrowed on its board and in the index to say so.
- **The index.** 101 rows now. 03 closes three and narrows one, so it goes to **98**, plus any
  issue it opens. Rows are counted, not computed.

**Carried from the 02 session.**
- A mutation record lists **every** failing test, with counts. Where a failing test has
  sub-tests, the log says so, so that a replay can compare distinct tests.
- The replay is done twice:
  - **before the commit**, from the final log file, and said to be so;
  - **after the commit**, from `git show HEAD:<log>`, and reported, with no amend.
- Every change is in the log, including a removal of unused code (the dead `ORDER BY`).
- The agent works in **its own scratch subdirectory**, not the session scratchpad's top level,
  where the orchestrator's `orch_*` files live.
- `[02-the-version-row-is-written-outside-the-in-flight-record]` is open and is **not** 03's.
  `version.build` is one of L3's eight, and making it `one_or_none()` neither fixes nor worsens
  that issue.

## 1. Before you dispatch

1. Record the branch and `HEAD`. It must be the commit that adds this note.
2. `python -m RunRegistry list`: nothing `running`.
3. Snapshot `var/datastores/`, `var/runs/` and the root `physics-test-n20-lambdacdm-zend0p1*.sqlite`
   read-only, with `orch_04_snapshot.py` in the session scratchpad. Write it again if it is
   missing; never put it in the tree. The 02 session's snapshot held 46 files and 10 directories.
4. Baselines, board §5.7: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39, Datastore 325,
   LiouvilleGreen 148 (1 skipped), RunRegistry 199. Re-measure all six on `HEAD`, serially.
5. `git status --short` and `git diff --cached` are both empty, and `git worktree list` shows this
   checkout alone.

## 2. Dispatch

One fresh-context subagent (**Sonnet**, as the prompt recommends). Give it:
- the prompt;
- the README;
- the audit, `docs/datastore-integrity-audit.md`, §3.5–§3.9;
- the three probes and their `.out` files, which the prompt names;
- the four issues' entries on their own boards;
- log 02's "State handed to the next prompt";
- `HEAD` and the baselines;
- §0 of this note.

Tell it plainly:
- **one commit**, with its log at `logs/03-exact-identity-lookups.md`, in README §5.1's form, with
  prompt 03 §7's additions:
  - the L5 table: every caller, each "unchanged" or "now misses", the evidence, and what a miss
    then does;
  - the `.scalar()` calls judged not to be identity lookups, each with its reason;
  - which way it took on the exit time's tolerance joins, and why;
- in the same commit:
  - the board: its §1 row, L1–L5, §3.1 and a new §5 subsection `5.8 After prompt 03`;
  - the three owning boards: each issue moves to its §4, with a closing note naming this commit by
    subject;
  - `[05a-…]` narrowed on the tolerance-convergence board;
  - `docs/OPEN_ISSUES.md`: three rows deleted, `[05a-…]`'s hook narrowed, and the count and date
    corrected;
- **the files it may change:**
  - the `build()` of each of the ten factories L1–L4 name, and nothing else in those files:
    `wavenumber.py` (`sqla_wavenumber_exit_time_factory.build` only), `integration_metadata.py`,
    `tolerance.py`, `store_tag.py`, `version.py`, `GkSourcePolicy.py`, `QuadSourcePolicy.py`,
    `LambdaCDM.py`, `QCD_Cosmology.py` and `redshift.py`;
  - one new test module under `Datastore/tests/`;
  - the log, the four boards and the index.

  **Nothing else:**
  - no `register()`, and so neither `"stepping": "minimum"`;
  - no `store()`, and no `if "serial" in payload` branch;
  - no existing test, including `test_factory_select_columns.py`, and not the schema witness;
  - not `ShardedPool.py`, `Datastore.py` or `store_inventory.py`;
  - not `main.py`, any `extract_*.py`, or the two verification scripts;
  - no audit probe, and nothing in `orchestrator/`;
- **no Ray, and nothing opens anything under `var/`**, or a copy of anything under it. Every store is
  in memory or in a temporary directory, built with `build_schema(MetaData(), _factories)` and the
  real factory `build()`s, as the probes do;
- every mutation diff is pasted from `git diff`, and replayed twice as §0 says;
- run `black`;
- prompt 03 §5's stop conditions mean *stop and ask*. So does any existing test failing, the
  read-back guard included.

## 3. The review

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only the permitted files.
   - `git diff HEAD~1 HEAD -- main.py extract_*.py docs/source-remediation-verification/ Datastore/SQL/ShardedPool.py Datastore/SQL/Datastore.py Datastore/store_inventory.py Datastore/tests/data/ RunRegistry/ docs/datastore-integrity-audit/ prompts/datastore-integrity/orchestrator/`
     is empty. `docs/` changes only in `docs/OPEN_ISSUES.md`.
   - `--diff-filter=M` over `*/tests/*` is empty.
   - Every factory hunk lies inside a `build()`, and none in a `register()`, a `store()` or an
     `if "serial" in payload` branch. Read every hunk and say where it is.
2. **L1, the exit time.**
   - The filter is `atol_serial == target_atol.store_id`, `rtol_serial == target_rtol.store_id` and
     `stepping == target_stepping`, with no `log10_tol` comparison left in it.
   - The `order_by` is gone.
   - `conn.execute(query)` runs once, after both `add_columns` loops.
   - The returned `atol`/`rtol` are the stored row's, which are now the requested serials.
   - Compare the shape with the orchestrator's sketch (`orch_03_sketch.diff` in the session
     scratchpad, if it survives). A different shape is fine if the guard is unmoved.
3. **L2.** `stepping == stepping` and `.scalar_one_or_none()`. The `stepping < 0` clamp above it is
   unchanged.
4. **L3.**
   - Exactly the eight sites change, and no filter changes with them.
   - `grep -n "\.scalar()" Datastore/SQL/ObjectFactories/*.py` leaves the 21 other calls, and the
     log's list of calls judged not to be identity lookups accounts for each of them.
   - `QCD_Cosmology`'s handler was read, and its test sees `MultipleResultsFound`.
5. **L4.** A branch for `z == 0` only. For every other `z` the filter expression is unchanged.
6. **The tests.**
   - Tests 1–5 exist with the prompt's cases: the three misses, the two-row case, stepping 0 against
     1, the solver's 4/8/6, a duplicate for **each** of the eight classes, and three `z = 0` builds
     giving one row.
   - Test 2 counts `SELECT`s with a `before_cursor_execute` listener and asserts exactly one.
   - `ray.is_initialized()` is false after the module, and nothing is under `var/`.
7. **The guard is unmoved.** `test_factory_select_columns.py` is byte-identical, and passes.
8. **L5.**
   - The table lists every caller `grep` finds: `main.py:993`, the six `extract_*.py` exit-time
     lookups, `main.py`'s solver lookups, and the two verification scripts.
   - Each "unchanged" rests on the pair and the stepping, not on assertion.
   - Each "now misses" says what the script then does, and that was found by reading, not by
     running.
9. **Replay the mutations** (i)–(v) from the committed log with plain `git apply`. Each gives its
   logged failures. Revert each, leaving the tree clean.
10. **Boards.**
    - The three issues are on their own boards' §4, each with a closing note naming this commit.
    - `[05a-…]` is narrowed on its board and in the index.
    - The index count is 98, plus any opened, counted as rows, with the date.
    - The campaign board has its §1 row, L1–L5, §3.1 and §5.8.
    - The `"stepping": "minimum"` observation is handed to 06.
11. **`var/`, suites.**
    - The snapshot is identical.
    - Run all six suites serially. Datastore rises by exactly the tests added, and the others match
      §1.4.
    - `black --check` is clean.

## 4. After the review

Record it on the board, after the review of 02. Then:
- **06 is next** (README §2: 06 follows 03), with its own orchestrator note carrying this review's
  findings. It corrects the two `"stepping": "minimum"` flags to say what 03 made true, and is the
  first schema change, under D5.
- Prompt 10 remains the first real run. Nothing in 03 is exercised under Ray until then.
