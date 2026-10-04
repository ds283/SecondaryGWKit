# Orchestrator — prompt 04, delete the legacy shard records

Read [`../README.md`](../README.md) §0.1, §1, §3, §4, §5 and §6.1 (D2). Then the audit's B6, B10
and G4 ([`docs/datastore-generic-audit.md`](../../../docs/datastore-generic-audit.md)), and the
board's review of 03 (§1). The conventions are [`prompt-03.md`](prompt-03.md)'s, with what the
review of 03 added (§0, last list).

**You do not write code.** You may:
- run the suites and the tests the log names;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process check from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`04-delete-the-legacy-shard-records.md`](../04-delete-the-legacy-shard-records.md)
**Board items:** L1–L3 · **Closes:** nothing · **Narrows:**
`[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]` · **Opens:**
`[04-closed-store-refusals-repeat-their-prefix]` · **Model:** Sonnet, as the prompt recommends.
**Gate:**
- 03 landed (`d8d4afe`) and reviewed (`4ca36c4`); 04 written (`ddc266d`); the tree clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code. If a later commit
  touches `Datastore/`, `tools/`, `RunRegistry/` or the sweep script, re-check every fact below
  that it could move before dispatch, and hand the agent the corrections as an addendum.
- One prompt at a time in this checkout.

## 0. What makes this prompt unusual

**It changes what every reader accepts, and for a current store nothing.** Every primary written
since `b04671f` holds bare names, so the deletion moves no current store. The review leans on two
things:
- the new refusal module, which pins the safety property the legacy tests guarded (a primary whose
  records name another store's files never touches them);
- every test outside §2 L3's list passing untouched.

**It reaches beyond `Datastore/`.** The first prompt of the campaign to do so.
- It edits `tools/shard_key_audit.py` and one function of `docs/handover/quadsource_atol_sweep.py`.
- It edits two `RunRegistry/tests/` modules, `test_quadsource_atol_sweep_prepare.py` and
  `test_store_retire.py`. The README's §1 scope covers them as "the tests of all of the above":
  they test the sweep check and `ShardedPool.delete_store` through the registry.
- `RunRegistry/` code is not touched (U3).

**The test plan was measured, not read.** On 2026-10-04 at `4ca36c4`, a read-only subagent made
the deletion in a scratch worktree. It removed the absolute branch of `resolve_shard_path` and the
notice, and kept `is_legacy_record` so that `test_shard_paths` still imported. It then ran every
affected module and both suites, and the worktree was removed:
- `discover -s Datastore/tests`: Ran 629, **failures=25, errors=64**. That is exactly the sum over
  the five Datastore modules below; no other module moved.
  - `test_shard_paths`: 10 tests, 5 errors.
  - `test_shardedpool_shard_paths`: 14 tests, 8 errors.
  - `test_copy_move_store`: 24 tests, 5 failures and 43 errors (subtests).
  - `test_delete_store`: 29 tests, 18 failures and 8 errors.
  - `test_shard_key_audit_copy`: 3 tests, 2 failures.
- `discover -s RunRegistry/tests`: Ran 281, **errors=7** (skipped=1): the six sweep tests that
  write absolute records, and `TestLegacyShape`.
- `test_store_reader`, `test_sharded_store_script`, `test_shard_file_name` and
  `test_store_copy_move` pass unchanged.
- With `is_legacy_record` deleted as well, `test_shard_paths` fails to import until its import goes.
- **Re-fixtures verified by that subagent**, each in the scratch worktree:
  - the bare-name `{0: n0, 1: n0}` cases (`test_copy_move_store`, `test_delete_store` under both
    `resume` values and with the file missing, `test_shardedpool_shard_paths`);
  - the moved-primary case;
  - `test_file_outside_the_primary_directory`, with two `write_new_store` stores and the resolver
    mocked to `lambda primary, stored: a_dir / stored`;
  - the audit `setUp` with bare records;
  - the four sweep refusals with bare names.

  So the prompt's §5 stop condition "a re-fixtured test cannot pass" is not expected to fire.
- Under that mutation, an absolute record is refused as unusable by `read_pool`, the closed-store
  operations, the audit tool (which today still ends "VERDICT: OK") and the sweep check (which
  raises a raw `ValueError`).

**A test outside the list failing is therefore a real signal.** It means current behaviour moved,
not that the inventory was incomplete. The agent stops and reports.

**Two behaviour changes beyond deletion, both deliberate.**
- **The audit tool's verdict on an unusable record.** Today it prints "record is unusable" and
  exits 0 with "VERDICT: OK", for any unusable record, `../outside.sqlite` included. After 04 every
  store that was legacy takes that path, so the prompt makes it exit 2. A missing shard *file* is
  left as it is (cross-file check skipped).
- **The sweep check's `ValueError`.** The prompt has the check report a refused record among its
  mismatches and raise its own `RuntimeError`.

The review checks that these are the only behaviour changes outside the resolver and the notice.

**Breakage (ii) moves real files.** (ii) restores the 2026-09-23 failure: the resolver returns an
absolute record's path when the file exists. Under it, `move_store` on B renames **A's** shard
files, and `copy_store` copies them. `delete_store` is still refused, by `_plan_deletion`'s
directory guard, but with a different message. The new module's stores live in temporary
directories, so the damage stays there. Replay (ii) only from the agent's commit, in `bash`.

**The A3 baseline, which the user may retire before 04 runs.** `var/datastores/handover-A3-baseline-lambdacdm.sqlite`
records its four shards by absolute path. The orchestrator read it on 2026-10-04, `mode=ro` and
through the registry's read-only paths. Its files' SHA-256 prefixes were the same before and after
(primary `fdbe93a541083e8f`, G4's).
- **At `HEAD`, `store retire` is already refused.** The fresh fingerprint is format 6 against the
  sidecar's format 1. Its reader also reads `GkSource` as an absent table and `QuadSourceIntegral`
  as incomplete, so a format-6 fingerprint would misdescribe the store. `--without-fingerprint`
  does not apply, because the fingerprint succeeds. `store copy` and `store move` still work at
  `HEAD`; after 04 they refuse.
- **At `81c0a9a`** (the tree that retired the A3 backup on 2026-09-25), `store retire --dry-run`
  matched the recorded fingerprint, digest `433b7fc3…`. It found no run and no sidecar naming the
  store, and listed exactly its five files. That is the clean route, and it does not depend on 04.
- **Before dispatch (§1 step 6),** check whether the store still exists. If it was retired, 04's §6
  cost no longer applies: tell the agent, and the log says so instead.

**The facts, checked by the orchestrator on 2026-10-04 at `4ca36c4`** (code unchanged since; `ddc266d`
adds only the prompt). Pass them on.

- **`Datastore/shard_paths.py`** [read].
  - The module docstring's two forms are at `:6-14`, with the stale `prepare()` sentence at
    `:13-14`. The legacy paragraph is at `:24-29`, and "Old rows are interpreted when read" at
    `:31-32`.
  - `resolve_shard_path` is at `:78`; its docstring bullets are at `:85-89`, and its absolute
    branch at `:108-113`.
  - `is_legacy_record` is at `:120-122`. `PurePath` is used only by the branch and by
    `is_legacy_record`, so its import goes; `Path` stays.
- **`Datastore/SQL/ShardedPool.py`** [read].
  - The import is at `:26-31`.
  - `_resolve_shard_rows` is at `:2250`: its docstring at `:2261-2263`, `relocated_legacy` at
    `:2265`, `:2282-2283` and `:2285-2293`.
  - `copy_store`'s docstring sentence on a legacy source is at `:2364-2365`, and `delete_store`'s
    at `:2409-2412`.
  - **Not 04's, and staying:**
    - `_shard_file_problems`' "both resolve" check (`:2320-2328`);
    - `_plan_deletion`'s directory guard;
    - `_write_shard_data`'s round-trip check (`:745`).
- **The doubled prefix** [measured]. `closed_store_files` on a hand-built primary with
  `{0: "../outside.sqlite"}` raises
  `Cannot delete sharded datastore "<p>": Cannot delete sharded datastore "<p>": Shard #0 of primary database …`.
  The two `<p>`s differ: one is the path as given (`/var/folders/…`), the other resolved
  (`/private/var/folders/…`). `copy_store` gives the same doubling with "Cannot copy … to …". The
  agent measures it again, records it in the issue, and does not fix it.
- **`tools/shard_key_audit.py`** [read].
  - The docstring's legacy sentence is at `:18-22`.
  - The `wavenumber_serial` branch is at `:90-97`, inside `if "key_serial" not in …` (`:89-107`).
    The "Unrecognised shard_keys schema" refusal at `:98-107` stays.
  - The `shard_key_config` tolerance is at `:146-149`.
  - The unusable-record message is at `:211-212`, and the verdict at `:223-231`.
  - No test exercises the `wavenumber_serial` branch or a missing `shard_key_config` today; the new
    tests are the first.
- **`docs/handover/quadsource_atol_sweep.py`** [read]. `assert_store_is_self_consistent` is at
  `:669`: the docstring at `:670-692`, and the call to `resolve_shard_path` at `:708`, outside any
  `try`. Its callers (`run_child()`, `prepare()`, `run_build()`'s `--resume`) are unchanged.
- **`shard_store_fixtures.py`** [read].
  - The module docstring's old-code sentence is at `:4-6`.
  - `read_pool` is at `:47`, with `read_stdout` at `:49` and `:58`. Its only readers are
    `TestLegacyRecords` (`test_shardedpool_shard_paths.py:132-133`, `:149`), which 04 deletes.
  - `write_legacy_primary` is at `:88`.
  - **Users after 04** (each writes bare or malformed non-absolute records):
    - `test_shardedpool_shard_paths`: `:295`;
    - `test_copy_move_store`: `:314`, `:448`, `:450`;
    - `test_delete_store`: `:357`, `:369`;
    - the re-fixtured sites.
  - No test opens a hand-built primary through the `ShardedPool` constructor, so its missing
    `replication_in_flight` (prompt 03) touches nobody.
- **The tests, with lines at `4ca36c4`** [counted with `ast`; classified by the subagent and the
  orchestrator].
  - **The fifteen deletions:**
    - `test_shard_paths`: `:41`, `:50`, `:112`;
    - `test_shardedpool_shard_paths`: `:119`, `:138`, `:231`, `:308`;
    - `test_copy_move_store`: `:209` (with `TestCopyOfALegacySource`) and `:267`;
    - `test_delete_store`: `:230` and `:247` (with `TestLegacyRecordsNamingAnotherStore`), and
      `:543`;
    - `test_quadsource_atol_sweep_prepare`: `:198`, `:208`;
    - `test_store_retire`: `:330` (with `TestLegacyShape`).
  - **The re-fixtures:**
    - `test_shardedpool_shard_paths`: `TestCopiedStore._original` (serving `:161` and `:188`),
      `:251` and `:275`;
    - `test_copy_move_store`: `:326`, its shared case at `:346-362`;
    - `test_delete_store`: `:391` and `:416`;
    - `test_shard_key_audit_copy.setUp` at `:59`;
    - the sweep tests at `:230`, `:246`, `:261` and `:277`.
  - **The splits:**
    - `test_shard_paths`: `:65`;
    - `test_copy_move_store.TestInterruption`: `:504`, `:568`, `:587`, `:653`;
    - `test_delete_store`: `:469` (`SHAPES` at `:467`).
  - **Kept with the fixture renamed:** `test_shardedpool_shard_paths:292`,
    `test_copy_move_store:312` and `:445`, and `test_delete_store:355` and `:360`.
  - **`test_shard_key_audit_copy`'s second test (`:108`)** keeps its assertions under the
    re-fixtured `setUp`. Its premise, that there is no fall-back to the original's absolute path,
    moves to the new audit refusal test with B's records. So it is re-fixtured, not "weaker"; the
    agent need not stop on it.
  - **Dead after the deletions:**
    - `_legacy_store` (`test_shardedpool_shard_paths:59`);
    - `legacy_source` (`test_copy_move_store:68`);
    - `legacy_pair` (`test_delete_store:92`);
    - `marker` in `test_delete_store` (used only at `:273`) and `stored_records` there (`:235`),
      if the deletions leave them unused;
    - in `test_store_retire`, `stored_records` (`:343`) and the `contextlib` redirect at `:354`,
      which exists only for the notice.

    Each after a grep.
- **Not touched, and staying:**
  - `RunRegistry/stores.py` and its legacy **sidecar** tests (U3). `stores.py:81-82` mentions
    "prompt 01's rule for legacy shard records"; that is an observation for the log, not a change;
  - `RunRegistry/tests/test_store_copy_move.py:175`, a legacy *sidecar* on a bare-name store;
  - every store under `var/`.
- **Expected counts after 04.**
  - **Datastore:** 629 − 12, plus the new tests. The new module has at least one test per entry
    point, or one test with sub-tests; the log says which. The audit refusal tests are four cases.
  - **RunRegistry:** 281 − 3, plus the sweep refusal and the retire refusal.
  - **ComputeTargets** 693, and every other suite, unchanged.
- **The index.** **107** at `4ca36c4`, matching the header. 04 opens one issue, so it is **108**
  after, plus one for each other issue it opens. The fallbacks row gains 04's part.

**What the review exists to establish.**
- **(E1) The legacy reading is gone, and only it.**
  - The resolver has one form; `is_legacy_record` and the notice are gone.
  - The audit tool loses the `wavenumber_serial` branch and the `shard_key_config` tolerance, and
    refuses an unusable record.
  - The sweep check raises its own error.
  - Nothing else in those files changes, docstrings aside.
- **(E2) The safety property is pinned at every entry point.**
  - A primary whose records are the absolute paths of another store's existing shards is refused by
    the resolver, `read_pool`, `copy_store`, `move_store`, `closed_store_files`, `delete_store`
    (both `resume` values), `open_read_only`, the audit tool, the sweep check and `retire_store`.
  - Neither store changes.
- **(E3) Nothing a current store does changes.** Every test outside §2 L3 passes untouched, and a
  bare-name store opens, copies, moves, deletes and audits identically before and after.
- **(E4) The test plan was followed.**
  - Exactly the fifteen deletions.
  - Each re-fixture changes only the records it builds.
  - Each split loses only its legacy case.
  - Each kept test differs only in the fixture's name.
- **(E5) The pins.** (i)–(v) each fail a named test.
- **(E6) The records.**
  - The fallbacks issue is narrowed with a dated line naming 04's deletions, and the A3 cost if the store still exists.
  - `[04-closed-store-refusals-repeat-their-prefix]` is opened with the measured message.
  - The index is 108 (or more).
  - The board has the §1 row, L1–L3, §3 and §5.

**Conventions, carried from 01–03.** All of these bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only. Show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the diff and the tests show.
- **The SHA.** The agent writes "this commit"; the orchestrator substitutes the SHA in its
  follow-up.
- **Suites.** Use README §7's command, write the output to a file, and grep the verdict from it.
  Run each suite **in the foreground**, serially, within the turn.
  - Under load, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`,
    `test_background_tau.test_build_cost` and
    `test_numeric_phase_cut.test_the_print_is_gated_both_ways` fail on wall clock.
  - Re-run a failure alone before calling it real. If it persists alone, settle it on an export of
    the tree before the prompt.
  - At 03's review, `test_wall_time_per_object` failed alone at load 42–89 at both `65be64f` and
    `d8d4afe`.
- **Breakage diffs** are recorded so that `git apply` replays them from a clean tree at the agent's
  commit. Check each with `git apply --check` and `-R`, and replay in `bash`.
- **Dead helpers** that only deleted tests used go in the same commit, after a grep (IMPLEMENTATION
  CHOICE). This includes an import left unused.
- **Scratch.** The agent uses its own subdirectory of the session scratchpad, never `/tmp`. Python's
  `tempfile` in tests is fine.
- **Ray.** 04 starts no Ray. ChamPBH's cluster (`gcs_server` 62525, `raylet` 62574, from its own
  venv) is left alone.
- **Stop conditions are real.** At 03 the note's own condition fired: an existing test failed. The
  agent stopped and the user decided. Here, too, a stop goes to the user through the orchestrator,
  and the agent resumes with `SendMessage`.
- **Other sessions** may land commits on this branch. Review against the agent's commit, and check
  `git log` before measuring at `HEAD`.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` must be empty.
   - `git worktree list` must show this checkout alone.
   - `git log --oneline 4ca36c4..HEAD` must touch no code (otherwise see the gate).
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`. At
   `4ca36c4` it held 16 runs: 14 finished, 2 `unknown`.
3. **Ray.** Record `pgrep -lf 'gcs_server|raylet|ray::'`. No listed process may come from this
   repository's `venv`.
4. **Baselines.** Last measured at `d8d4afe` (03's review, check 10): AdaptiveLevin 32,
   ComputeTargets 693 (1 wall-clock failure), CosmologyModels 39, Datastore 629, LiouvilleGreen 148
   (1 skipped), RunRegistry 281. Re-measure all six at `HEAD`, serially, with the load average, and
   give the agent the numbers.
5. **The index.** 107 now; 108 after, plus any other issue 04 opens. Tell the agent both.
6. **The A3 baseline.** `ls var/datastores/handover-A3-baseline-lambdacdm*`, and read its
   sidecar's `retired` field. Tell the agent whether the store still exists (§0).
7. **The pre-images**, for checks 1–4. Record `git rev-parse HEAD:<path>` for:
   - `Datastore/shard_paths.py`, `Datastore/SQL/ShardedPool.py`, `tools/shard_key_audit.py`,
     `docs/handover/quadsource_atol_sweep.py` and `Datastore/tests/shard_store_fixtures.py`;
   - the seven test modules of §0.

## 2. Dispatch

Dispatch one fresh-context subagent, on **Sonnet**. Give it:
- the prompt;
- the campaign README, the board (including the review of 03), 03's log, and the audit's B6, B10
  and G4;
- `HEAD`, the baselines and the index counts;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad. The
  agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/04-delete-the-legacy-shard-records.md`, in README
  §5.1's form, with the prompt's §7 additions.
- **The files it may add or change are the prompt's §6 list, and nothing else.** In particular, it
  must not touch:
  - `RunRegistry/` code;
  - `Datastore/store_reader.py`, `store_inventory.py`, `real_store_fixtures.py` or `Datastore.py`;
  - any part of the sweep script outside `assert_store_is_self_consistent`;
  - any test not named in the prompt's §2 L3;
  - anything under `var/`, `docs/` (other than the index and that one function) or `orchestrator/`.
- **Stop and ask** on any of the prompt's §5 conditions. A test outside §2 L3 failing is current
  behaviour moving, never a reason to edit it.
- **Breakage (ii)** is applied only to the agent's own tree. It moves files only inside the tests'
  temporary directories.

## 3. The review — ten checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only the prompt's §6 list.
   - The diff over these is empty: `RunRegistry/*.py` (not tests), `Datastore/SQL/Datastore.py`,
     `ObjectFactories/`, `Datastore/store_*.py`, `real_store_fixtures.py`, `config/`, `main.py`,
     `var/` and `orchestrator/`.
   - Under `docs/`, only `OPEN_ISSUES.md` and the sweep script change.
2. **E1, by grep.**
   - The prompt's §3.2 greps print nothing.
   - `PurePath(stored).is_absolute()` appears nowhere.
   - `grep -rn is_legacy_record` over the repository prints nothing outside `prompts/` and
     `docs/`.
3. **E1, by `ast`, against the pre-images.** Compare each function with its docstring stripped.
   - In `shard_paths.py`, only `resolve_shard_path` changes, and `is_legacy_record` is gone.
   - In `ShardedPool.py`, only `_resolve_shard_rows`' body changes, plus the import; `copy_store`
     and `delete_store` dump identically.
   - In the sweep script, only `assert_store_is_self_consistent` changes. Read its new handling of
     `ValueError`.
   - In the audit tool, only `main` changes. Read it against §0: the branch and the tolerance are
     gone, an unusable record exits 2, and a missing file behaves as before.
4. **E4, the tests, by `ast`.** List the test methods of the seven modules before and after, and
   the new modules.
   - Exactly the fifteen of §0 are gone.
   - Every new method belongs to the new module, the audit refusals, the sweep refusal or the
     retire refusal, and the log names its purpose.
   - Compare each surviving method with its pre-image:
     - a kept test differs only in the fixture's name;
     - a re-fixture only in the records it builds (and its name, if the log says so);
     - a split only in its case list.

     Read every diff that is not one of these.
5. **E2, the new module, by reading and running.**
   - Each entry point is driven against B, the store whose records name A's existing shards.
   - Each asserts the refusal's text and `tree_state` of both directories.
   - Run it, and the audit, sweep and retire refusal tests, by name.
6. **E3, by running.** Every test of §0's "pass unchanged" list, and the kept tests, pass. Run
   `test_shard_file_name`, `test_store_reader`, `test_sharded_store_script` and
   `test_store_copy_move` by name.
7. **E5, the pins.** In `bash`, replay each breakage diff (i)–(v) with `git apply`, run the named
   tests and see them fail. Under (ii), check that nothing outside the tests' temporary directories
   changed (`git status` clean, `var/` untouched). Revert each with `git apply -R`, and check that
   `git status` is clean.
8. **The probe, against the fix.** From the scratchpad, at the agent's commit and at an export of
   `HEAD~1`, with `shard_store_fixtures`:
   - **A bare-name store** (`write_new_store`):
     - is read by `read_pool`;
     - is copied, moved and deleted to the same `tree_state` at both trees;
     - is audited by the tool with the same output and exit code at both.
   - **B** (records naming A's shards):
     - at `HEAD~1` it is read as its own siblings, with the notice;
     - at the agent's commit every entry point refuses it;
     - A is unchanged at both.
   - **A primary whose absolute records name its own siblings** (every legacy store's shape) opens
     at `HEAD~1` and is refused at the agent's commit.
   - **The audit tool:**
     - on a primary without `shard_key_config`: at `HEAD~1` it exits 0 ("Cross-file check … was
       not possible"); at the agent's commit it exits 2;
     - on `{0: "../outside.sqlite"}`: at `HEAD~1` it exits 0 with "VERDICT: OK"; at the agent's
       commit it exits 2.
   - **The A3 baseline is not opened**, whether or not it still exists.
9. **E6, the records.**
   - The fallbacks issue carries a dated "Narrowed (…, prompt 04)" line, with the A3 cost if the store still exists.
   - `[04-closed-store-refusals-repeat-their-prefix]` is in §3 with the measured message, and its
     row is in the index.
   - The index is 108 (or more) by counting, matching the header.
   - The board has the §1 row, L1–L3 and §5.
10. **Suites.** All six, serially at `HEAD`:
    - Datastore and RunRegistry at §0's expected counts, with each new test named in the log;
    - every other count unchanged.

    `black --check` is clean on every touched Python file.

## 4. After the review

- **Record it on this board** as a §1 *Orchestrator review of prompt 04* paragraph, in the form
  of 01's–03's. Fix small residue in a follow-up commit of its own, including "this commit" → the
  SHA, and this note marked "used" in the board's list.
- **Report to the user:**
  - what an old store now meets, entry point by entry point;
  - what a current store sees (check 8: nothing changes);
  - the audit tool's and the sweep check's new refusals;
  - the tests deleted, re-fixtured, split and added;
  - the issue opened;
  - the A3 baseline's cost, if it still exists;
  - what remains of the old-store fallbacks (05).
- **What 04 hands on.** **05**'s note must carry:
  - the absent-table and absent-column tolerance in `_read_replicated_tables`;
  - 02's four `numeric_serial` selects;
  - the bare `OperationalError` for a primary without `replication_in_flight` (03);
  - the split grouping's exclusion of `timestamp`, which is current behaviour (03);
  - whatever 04's log hands on.
