# Orchestrator — prompt 02, delete the remaining factory refusals

Read [`../README.md`](../README.md) §0.1, §1, §4 and §5. Then the audit's B7 and B10
([`docs/datastore-generic-audit.md`](../../../docs/datastore-generic-audit.md)), 01's log, and 01's
review on the board (§1, *Orchestrator review of prompt 01*). The conventions are
[`prompt-01.md`](prompt-01.md)'s, carried forward with the changes 01's review taught (§0, last
list).

**You do not write code.** You may:
- run the suites and the tests the log names;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process check from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`02-delete-the-remaining-factory-refusals.md`](../02-delete-the-remaining-factory-refusals.md)
**Board items:** N1–N3 · **Closes:** nothing · **Narrows:**
`[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]` · **Model:** Sonnet, as the prompt
recommends.
**Gate:** 01 landed (`8636491`) and reviewed (`c647ee2`), and the tree is clean. `HEAD` is the
commit that adds this note, or a later one that touches no code. If a later commit touches code,
re-check every fact below that it could move before dispatch, and hand the agent the corrections as
an addendum, as 01's dispatch did. One prompt at a time in this checkout.

## 0. What makes this prompt unusual

**It is 01's deletion without 01's defect, so N3 guards less than M4 did.** 02's four factories
match the driver's own message (`str(getattr(e, "orig", e))`), not `str(e)`, so they never misfired.
A faithful restoration of any of them would therefore **pass** N3's tests. An injected `database is
locked` names no column in `e.orig`, so the restored handler re-raises it unchanged. That is
correct: N3 pins pass-through, and a handler that does not misfire does not break pass-through. It
follows that:
- **The deletion is established by reading**, with grep and `ast`, not by a test.
- **Breakage (i) is the pin.** The prompt makes (i) a restoration "changed to raise for every
  `OperationalError`", and that is a misfire, which N3 must catch. The review replays it.

Do not ask N3 to catch a faithful restoration; it cannot.

**Two tests are kept, and one of them is rewritten.** README §5 rule 7 lets an existing test change
only where the prompt names it.
- **`test_gksource_parent_set.TestTheWKBValueReadsRefuseAPre07Store.test_a_current_table_is_still_read`
  is moved, unchanged, to a class of its own.** It builds its own current store (`_gk_store()`) and
  does not use the class's `setUp`, which rebuilds the WKB tables from the 06 witness. Its body must
  not change.
- **`test_quadsource_policy_key.TestAnOldStoreIsRefused.test_another_database_error_is_not_turned_into_it`
  is rewritten.** Its fixture **was** the old table, because the class's `setUp` recreates
  `QuadSourceIntegral` from the 07 witness's DDL. It raises `OperationalError("SELECT 1", …)`,
  01's weak literal, through `mock.patch.object(conn, "execute")`. The rewrite puts it on a current
  table and raises the error through 01's harness, with the captured statement. It may then *be*
  N3's `QuadSourceIntegral` test; if so, the log says which is which.
- **The parallel test in `test_quadsource_integral_parent_key` is deleted with its class**, as the
  prompt says. The kept one and N3 cover its property.

**A fallback the audit missed: four selects of `numeric_serial`** [read]. In both WKB factories,
`_build_impl_model` and `read_batch` select `wkb_table.c.numeric_serial` "always … whether or not a
parent is given, so that a store from before the column (prompt 07) is refused with the D4 message
rather than read":
- `TkWKBIntegration.py:1276-1279` and `:1409-1411`;
- `GkWKBIntegration.py:1270-1273` and `:1405-1408`.

No code reads the selected value: `numeric_serial` appears in those functions only in the select and
in `_numeric_parent_filter`, which is applied only when a parent is given. **The select stays; the
comment is reworded.** Deleting it would change the outcome on a pre-07 table: a labels-only read
would succeed and return values from rows whose parent is unrecorded. The docstring of the deleted
`TestTheWKBValueReadsRefuseAPre07Store` says that is what happened before 09b. The prompt's §1
promises the opposite: such a store "meets SQLAlchemy's own `OperationalError` naming the missing
column". So:
- The comment says the column is selected so that a table without it raises rather than being read,
  names no prompt's message, and does not say "predates" or "written before".
- The log records it as a deviation (IMPLEMENTATION CHOICE, directed by the orchestrator) and under
  *Observations*, for 05: once the reader refuses a store whose columns differ from the registry's
  (README §4), the select has no purpose of its own.
- This parallels the prompt's leave to keep `QuadSourceIntegral`'s column order.
- **If the agent finds that the selected value is read** after all, it says so and keeps the select
  as current code.

**The over-deletion to watch.** Four `except SQLAlchemyError:` blocks in these files are **not**
old-store code. They are prunes in `validate_on_startup` that append "DATABASE ERROR encountered when
pruning these values" and continue:
- `TkWKBIntegration.py:979`;
- `GkWKBIntegration.py:981`;
- `GkSource.py:850`;
- `QuadSource.py:529`, which is not 02's file at all.

`QuadSourceIntegral.py` has none. Not 02's to touch.

**The facts, checked by the orchestrator on 2026-10-04 at `c647ee2`.** Pass them on. None of the
four factories, the five test modules or `real_store_fixtures.py` has changed since the audit's tree
`dbd8a51`, so the audit's line numbers hold.

- **The sites to delete** [read]. *Method* is where the site sits.
  - `TkWKBIntegration.py`:
    - module docstring `:19-20` ("There is no migration … build() raises naming it") and `:44-45`
      ("A store written before this change … sidecar records");
    - `sqla_TkWKBIntegration_factory.build`, `except SQLAlchemyError` at `:394-422`: both the
      `numeric_serial` refusal and the `rho_gauss_order` refusal. The `try` stays, because it still
      guards `MultipleResultsFound` (`:389`);
    - `sqla_TkWKBValue_factory._build_impl_model`, which `TkWKBValue.build` reaches, `:1343-1354`.
      The `try` stays;
    - `read_batch`, `:1476-1488`. Here the `try` guards only `conn.execute(row_query).all()`, so it
      goes too.
  - `GkWKBIntegration.py`: the same.
    - Docstring `:19-20` and `:48-50`.
    - `build` `:404-432` (the `try` stays).
    - `_build_impl_model` `:1336-1347` (the `try` stays).
    - `read_batch` `:1504-1516` (the `try` goes).
  - `GkSource.py`:
    - `_refuse_a_store_that_predates_the_parent_set` (`:71-84`);
    - its calls in `sqla_GkSource_factory.build` (`:447-448`) and in
      `sqla_GkSourceValue_factory._build_impl_model` (`:1224-1225`). Both `try`s stay
      (`MultipleResultsFound`);
    - the docstring's `:32-34` ("A store written before this change has no parents_digest column …");
    - the stale sentence `:14-15` ("so a datastore written before this change is readable by this
      factory without a migration"), which the audit marks false.
  - `QuadSourceIntegral.py`:
    - `_refuse_a_store_that_predates_the_policy_key` (`:80-101`);
    - `_TRANSFER_FUNCTION_COLUMNS` (`:103-109`), used only by the next;
    - `_refuse_a_store_that_predates_the_transfer_function_key` (`:112-138`);
    - their calls in `build` (`:452-455`; the `try` stays) and in `read_batch` (`:898-903`; the `try`
      guards only `conn.execute(query)` and goes);
    - the column-order comment `:724-727`. The order may stay (*prompt's choice*).

    With the helpers gone, **`SQLAlchemyError` is unused in this module** and its import goes, as in
    01's `QCD_Cosmology` (STRUCTURALLY REQUIRED). The other three modules still use it, for their
    prunes.
- **The verification grep** (prompt §3.2), `grep -nE 'predate|older store|written before'
  Datastore/SQL/ObjectFactories/`, finds today **only** these four files: every hit above, plus
  `:1276`, `:1409`, `:1270` and `:1405` (the select comments). After 02 it must find nothing that
  refers to a store's schema. The other factories already have none, after 01.
- **The first stop condition does not fire** [read]. No deleted handler recovers from anything. Each
  either converts a matching error into a `RuntimeError` or re-raises it. The only match a current
  store could produce is `GkSource`'s `"GkSource_parents" in driver_message` for an absent
  `GkSource_parents` table, which is U1's interrupted first open (05's). It raised before and it
  raises after; only the exception's type changes.
- **The tests N2 deletes, exactly** [read, counted with `ast`]. Seventeen methods, all in
  `Datastore`:
  - `test_gksource_parent_set.py`:
    - `TestAnOldStoreIsRefused` (`:611`): `test_a_keyed_lookup`, `test_the_reader_mode`,
      `test_the_value_model_read`;
    - `TestTheWKBValueReadsRefuseAPre07Store` (`:665`): `test_gk_wkb_value_read_batch`,
      `test_gk_wkb_value_build_impl_model`, `test_tk_wkb_value_read_batch`,
      `test_tk_wkb_value_build_impl_model`;
  - `test_wkb_numeric_parent_key.py`: `TestAnOldStoreIsRefused` (`:871`): `test_tk`, `test_gk`;
  - `test_quadsource_integral_parent_key.py`: `TestAnOldStoreIsRefused` (`:488`):
    `test_a_table_from_before_this_prompt`, `test_a_table_from_before_prompt_08_keeps_its_message`,
    `test_another_database_error_is_not_turned_into_it`;
  - `test_quadsource_policy_key.py`: `TestAnOldStoreIsRefused` (`:312`): `test_build`,
    `test_read_batch_untagged_and_tagged`;
  - `test_quadsource_parent_key.py`: `TestAnOldStoreIsRead` (`:521`):
    `test_the_old_table_has_no_index_on_tr_and_both_serials`, `test_a_keyed_lookup_finds_its_row`,
    `test_the_reader_mode_finds_it_too`. The `QuadSource` factory has no code behind them: it catches
    only `MultipleResultsFound`.

  **Kept: 2** (above). Their classes' helpers (`setUp`, `assertD4`, `_make_old`, `recreate`,
  `refused`, and the `gk_*`/`tk_*` readers) go with the classes, unless something kept uses them.
- **What becomes dead, and goes in 02's commit** (01's review made this the rule: helpers, constants
  and imports that only the deleted tests used are part of the deletion; IMPLEMENTATION CHOICE). Grep
  each before removing it.
  - **The module-level witness constants:**
    - `WITNESS_06` and `WITNESS_09A` in `test_gksource_parent_set` (`:79-80`);
    - `WITNESS_06` in `test_wkb_numeric_parent_key` (`:96`);
    - `WITNESS_07` and `WITNESS_09B` in `test_quadsource_integral_parent_key` (`:69-70`);
    - `WITNESS_07` in `test_quadsource_policy_key` (`:61`);
    - `WITNESS_08` in `test_quadsource_parent_key` (`:72`).
  - **The `D4` message constants** in `test_gksource_parent_set:110` and
    `test_quadsource_integral_parent_key:94`.
  - **Any `json` or `mock` import left unused.** `test_wkb_numeric_parent_key` still uses `mock` at
    `:521`, `:972` and `:984`. Its `_OLD_DROP_ORDER` at `:984` is a drop-order test, not 02's.
  - **The witness JSON files under `Datastore/tests/data/` stay.** `test_schema_builder.py` reads all
    of them, and `test_store_inventory.py` reads the 06 one.
  - **The module docstrings' bullets** describing the deleted tests are corrected:
    `test_gksource_parent_set` ("§3 test 6/7"), `test_wkb_numeric_parent_key:29`,
    `test_quadsource_integral_parent_key:22`, `test_quadsource_policy_key:21`, and
    `test_quadsource_parent_key`'s "§3 test 7".
- **The prompt's search** (any other test asserting a deleted message) finds, at `c647ee2`, nothing
  outside N2's list. I searched for "predates", "readable at the git SHA", "does not read it",
  "sidecar records" and `D4`. The remaining hits:
  - `test_prune_at_open.py:26`, `:620`: 03's;
  - `test_database_errors_reach_the_caller.py:6`: 01's docstring;
  - `test_run_identity.py:723`, `:791`: 01's `assertNotIn("predates", …)`;
  - `RunRegistry/tests/test_run_registry.py:541`: a run directory;
  - unrelated `D4`s in `test_store_retire`, `test_delete_store`, `test_qcd_cosmology_inventory_record`
    and `test_T_z_representation`.

  `real_store_fixtures.build_old_store` drops `OneLoopIntegral_tags` and
  `TkNumericIntegration.stop_Tprime`, which none of 02's factories refuse. Its users
  (`test_store_reader`, `test_store_inventory`, `test_inventory_*`) are 05's.
  `test_gksource_parent_set.py:1087`'s `test_a_store_without_the_table_is_named_absent` (inventory)
  stays.
- **01's harness, for N3** [read, `Datastore/tests/test_database_errors_reach_the_caller.py`].
  - `_StatementRaisingConnection(inner=None, succeed=0)` passes its first `succeed` calls to `inner`
    and raises every later one as `OperationalError(<compiled statement>, {}, Exception("database is
    locked"))`. `executed` counts the calls and `raised` holds the error.
  - `_PassThroughCase.assert_passes_through(call, conn, columns)` checks that `str(conn.raised)`
    names every one of `columns`, then that the caller gets the same object with no `__cause__`.
  - **N3 adds classes to that module** (preferred), or imports the harness from it. It does not
    write a second harness.
  - The payload helpers for these factories already exist: `_gk_store`, `source_lookup` and `_call`
    in `test_gksource_parent_set`, `_WithAStore.build`/`read_batch` in
    `test_quadsource_integral_parent_key`, and the store helpers of `test_wkb_numeric_parent_key`. A
    path that executes something before its main query uses `succeed=`.
  - Each N3 test names the column the deleted handler looked for in `columns=`. That is
    `numeric_serial` for the WKB reads, `parents_digest` for `GkSource`, and `quad_policy_serial`
    for `QuadSourceIntegral`.
- **Expected counts after 02.** Datastore **605 − 17**, plus 02's new tests (N3). The two kept tests
  are still counted. **ComputeTargets 693 and every other suite unchanged**: no ComputeTargets test
  touches 02's refusals.
- **The index.** **107** at `c647ee2` (`grep -c '^| \`\[' docs/OPEN_ISSUES.md`, matching the header).
  02 closes nothing: **107** after, plus one for each issue it opens. The fallbacks row gains 02's
  part of its narrowing.

**What the review exists to establish.**
- **(E1) The old-store handlers, helpers and comments are gone** from the four factories, and only
  they. The four prunes are untouched. The `numeric_serial` selects stay, with neutral comments. No
  other `except` changed.
- **(E2) Nothing a current store returns changes.** Every existing test outside N2's list passes
  untouched. The moved test's body is byte-identical to its pre-image.
- **(E3) The deletions are exactly N2's seventeen, plus their now-dead helpers and docstring
  lines.** The two kept tests are present, one moved and one rewritten as the prompt says.
- **(E4) N3 is 01's pattern.** Each factory has a pass-through test through `_StatementRaisingConnection`,
  with a captured statement and `columns=`. Breakage (i) and (ii) fail it; (iii) fails the kept
  test.
- **(E5) The records.** The fallbacks issue is narrowed with a dated line; the index is 107 (or
  more, if 02 opens an issue); and the board has the §1 row, N1–N3 and §5.

**Conventions, carried from 01, with what 01 taught.** All of these bind the agent, and the brief
says so.
- **Staging and reporting.** Stage by explicit path only. Show `git diff --cached --name-only` before
  committing. Count, then paste. A log states only what the diff and the tests show.
- **The SHA.** The agent writes "this commit"; the orchestrator substitutes the SHA in its follow-up.
- **Suites.** Use README §7's command, write the output to a file, and grep the verdict from it.
  **Run each suite in the foreground and wait for it within the turn.** 01's agent ran its suites
  in the background and ended its turn three times waiting on them, which cost two resumes. Under
  load, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, `test_background_tau.test_build_cost`
  and `test_numeric_phase_cut.test_the_print_is_gated_both_ways` fail. Re-run a failure alone before
  calling it real. If it persists alone, settle it on an export of the tree before the prompt. In
  01's review, `test_wall_time_per_object` failed alone at both `8636491` and `6ea366e`.
- **Breakage diffs** are recorded so that `git apply` replays them from a clean tree at the agent's
  commit. Check each with `git apply --check` and `-R`, and replay in `bash`.
- **Dead helpers** that only deleted tests used go in the same commit (above).
- **Scratch.** The agent uses its own subdirectory of the session scratchpad, never `/tmp`.
- **Ray.** 02 starts no Ray. ChamPBH's cluster (`gcs_server` 62525, `raylet` 62574) runs from its own
  venv and is left alone.
- **Other sessions** may land commits on this branch. Review against the agent's commit, and check
  `git log` before measuring at `HEAD`.
- **Usage limits.** Resume a stopped agent with `SendMessage`, saying exactly where it stopped.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`. `git status --short` and `git diff --cached` must be
   empty, `git worktree list` must show this checkout alone, and `git log --oneline c647ee2..HEAD`
   must touch no code (otherwise see the gate).
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`. At
   `c647ee2` it held 16 runs: 14 finished, 2 `unknown`.
3. **Ray.** Record `pgrep -lf 'gcs_server|raylet|ray::'`. No listed process may come from this
   repository's `venv`.
4. **Baselines.** Last measured at `8636491` (01's review, check 10): AdaptiveLevin 32,
   ComputeTargets 693, CosmologyModels 39, Datastore 605, LiouvilleGreen 148 (1 skipped), RunRegistry
   281. `c647ee2` changed only comments in the `extract_*_data.py` scripts, and their `ast`s are
   unchanged. Re-measure all six at `HEAD`, serially, with the load average, and give the agent the
   numbers.
5. **The index.** 107 now, and 107 after unless 02 opens an issue. Tell the agent both.
6. **The pre-images**, for checks 1–3. Record `git rev-parse HEAD:<path>` for:
   - the four factories;
   - `QuadSource.py`;
   - the five test modules of N2;
   - `test_database_errors_reach_the_caller.py`;
   - `real_store_fixtures.py`.

## 2. Dispatch

Dispatch one fresh-context subagent, on **Sonnet**. Give it:
- the prompt;
- the campaign README, the board (including 01's review), 01's log, and the audit's B7 and B10;
- `HEAD`, the baselines and the index counts;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad. The
  agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/02-delete-the-remaining-factory-refusals.md`, in README
  §5.1's form, with the prompt's §7 additions.
- **The files it may add or change:**
  - `Datastore/SQL/ObjectFactories/TkWKBIntegration.py`, `GkWKBIntegration.py`, `GkSource.py`,
    `QuadSourceIntegral.py`. The changes are limited to the handlers, helpers, docstrings and
    comments N1 and §0 name, the four select comments, and `QuadSourceIntegral`'s unused import.
    Nothing else in them;
  - the five test modules of N2: the seventeen deletions, the two keeps, their dead helpers and
    docstring lines;
  - `Datastore/tests/test_database_errors_reach_the_caller.py`, **new classes only**. 01's nine
    tests and its harness are not modified;
  - the log, this board and `docs/OPEN_ISSUES.md`.

  **Nothing else.** In particular, it must not touch:
  - 01's factories, `QuadSource.py`, `ShardedPool`, the reader or the inventory;
  - `real_store_fixtures.py`, or any test outside the six modules above;
  - the four prune blocks;
  - anything under `var/` or `orchestrator/`.
- **Stop and ask** on either of the prompt's §5 conditions. This note's facts answer the first as
  "does not fire" at `c647ee2`. An existing test failing outside N2's list is a sign that current
  behaviour moved, not a reason to edit the test.

## 3. The review — ten checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only §2's list. The diff over
   `Datastore/SQL/ObjectFactories/{TkNumericIntegration,GkNumericIntegration,QCD_Cosmology,BackgroundModel,QuadSource}.py`,
   `Datastore/SQL/ShardedPool.py`, `Datastore/store_*.py`, `Datastore/tests/real_store_fixtures.py`,
   `extract_*.py`, `main.py`, `config/` and `var/` is empty.
2. **(E1) The handlers.**
   - **By grep.** `grep -nE 'predate|older store|written before'
     Datastore/SQL/ObjectFactories/` shows nothing that refers to a store's schema.
     `_refuse_a_store_that_predates` and `_TRANSFER_FUNCTION_COLUMNS` are gone everywhere.
     `except SQLAlchemyError` in the four files survives only in the three prunes, shifted only by
     the deletions above them.
   - **By `ast`, against the pre-image.** The only changed functions are:
     - the two WKB `build`s, `_build_impl_model`s and `read_batch`es;
     - `sqla_GkSource_factory.build` and `sqla_GkSourceValue_factory._build_impl_model`;
     - `sqla_QuadSourceIntegral_factory.build` and `read_batch`;
     - the removed module-level helpers.

     `validate_on_startup` dumps identically in all four files.
   - **The `numeric_serial` selects.** In the four functions, each select's column list is unchanged
     and only its comment differs. If the agent changed a column list, read why.
3. **(E3) The tests.** With `ast`, list the test methods of the six modules before and after:
   - exactly the seventeen of §0 are gone;
   - the two kept ones are present;
   - every new method is N3's, or the rewrite of the kept `QuadSourceIntegral` test.

   Show that the moved `test_a_current_table_is_still_read`'s function body dumps identically to
   its pre-image.
4. **(E2) Nothing a current store sees changes.** Every existing test outside N2's list passes, and
   none is modified. Check that with the diff: under `*/tests/`, only the six modules change.
5. **(E4) N3, by reading.**
   - One test per factory, through `_StatementRaisingConnection`, with the statement compiled from
     the query the factory executes.
   - `columns=` names the deleted handler's column.
   - The test asserts `assertIs` on the raised error.
   - The kept `QuadSourceIntegral` test no longer uses `"SELECT 1"` or `mock.patch` on `execute`.
6. **The kept tests, by running.** The moved test and the rewritten one pass. Then replay breakage
   (iii) and see the moved test fail.
7. **(E4) The pins.** In `bash`, replay each breakage diff (i)–(iii) with `git apply`, run the
   named tests and see them fail.
   - **For (i), also replay a faithful restoration**: the old helper with its `driver_message`
     test, unchanged. See N3's `GkSource` test **pass**. That is §0's point, and the review records
     it as expected, not as a gap.
   - Revert each with `git apply -R`, and check `git status` is clean.
8. **The probe, against the fix.** From the scratchpad, drive `GkSource.build` and
   `QuadSourceIntegral.read_batch` through the N3 harness with a `database is locked` error carrying
   the real statement, at the agent's commit and at an export of `HEAD~1`. Both raise the same
   `OperationalError`. The defect 01 fixed is absent here, and the probe shows it: this deletion
   changes no current behaviour. Then build one pre-07 `TkWKBIntegration` table from the 06 witness
   in a scratch store, and do a labels-only `TkWKBValue.read_batch`:
   - **at the pre-image:** a `RuntimeError` saying "predates";
   - **at the agent's commit:** SQLAlchemy's `OperationalError` naming `numeric_serial`, not a
     silent read. That is the select doing its job.
9. **(E5) The records.**
   - The fallbacks issue carries a dated line naming what 02 deleted, and an *Observations* note on
     the selects, for 05.
   - The index is 107 (or more, by issues opened), by counting, matching the header.
   - This board has the §1 row, N1–N3 and §5.
10. **Suites.** All six, serially at `HEAD`: Datastore at §0's expected count, plus N3, with each new
    test named in the log, and every other count unchanged. `black --check` is clean on every
    touched Python file.

## 4. After the review

- **Record it on this board** as a §1 *Orchestrator review of prompt 02* paragraph, in 01's form.
  Fix small residue in a follow-up commit of its own, including "this commit" → the SHA, and this
  note marked "used" in the board's list.
- **Report to the user:**
  - that 02 changed no current behaviour (check 8);
  - the seventeen tests deleted and the two kept;
  - the `numeric_serial` selects: kept, and why, and that 05 inherits them;
  - what remains of the old-store fallbacks (03–05).
- **What 02 hands on.**
  - **03** is independent of 01–02 and edits `ShardedPool`. Its note must carry 01's and 02's
    conventions.
  - **05**'s note, when written, must carry the four `numeric_serial` selects and the decision to
    keep them until the reader refuses a store whose columns differ.
