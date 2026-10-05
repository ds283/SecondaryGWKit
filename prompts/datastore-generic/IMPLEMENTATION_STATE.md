# datastore-generic campaign — implementation state

**Last updated:** 2026-10-05 · **Status: IN PROGRESS — 4 of 9 prompts landed (01, 02, 03, 04).** The audit ([`docs/datastore-generic-audit.md`](../../docs/datastore-generic-audit.md)) is
written. Prompts 01–05 are written; 06–09 are held on the prompts before them. The user took U1–U4
as recommended on 2026-10-04 (README §6.2), and had the plan and `e53f323` merged into
`handover-remedial` the same day.

The campaign was opened on 2026-10-04, when the user found that `ShardedPool` names project tables
and decided that the datastore layer is made generic again and that every fallback for a store
written by older code is deleted (README §0, §6.1). One commit, `e53f323` on branch
`drop-valuerow-key-exception`, was made before the campaign at the user's request; prompt 03
reviews it (D3).

**Campaign:** [`README.md`](README.md) ·
**Code:** `Datastore/` (the generic layer and `SQL/ObjectFactories/`), `extract_common.py`,
`tools/shard_key_audit.py`, `config/` ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.18

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**:
> the row is added, moved or deleted, and the count and date in its header are corrected. The index
> is an index: one line per issue, pointing here. Where the two disagree, this board is right. See
> `CLAUDE.md`.

### Decisions

**Made by the user, 2026-10-04 (README §6.1):**

| Decision | Decided |
|---|---|
| **D1** — the datastore layer | generic: it names none of the project's tables; the facts it needs are declared by the factories or in `config/` |
| **D2** — old-store fallbacks | every one deleted, **through this campaign**, not on sight; reflects the build phase, and reopens once a store backs a published paper |
| **D3** — the `BackgroundModelValue` key exception | deleted early in `e53f323` (side branch); prompt 03 reviews it. *Amended 2026-10-04:* merged into `handover-remedial` at the user's instruction, so 03 reviews it in place |
| **D4** — the `timestamp` exception | goes, in prompt 03, after a measurement that every replicated write stamps one timestamp |

**Put to the user by the audit (README §6.2), and taken by the user on 2026-10-04, each as
recommended:**

| Decision | Taken |
|---|---|
| **U1** — an absent table (05) | a read-write open keeps reading it as empty (crash recovery); the reader, inventory and read-only pool refuse it by name |
| **U2** — the older registries (06) | `_factories`, `_drop_actions`/`_drop_order` and `ClientPool`'s batch sizes move to `config/`, passed in by `main.py` |
| **U3** — the registry's legacy records | out of scope |
| **U4** — the `version` and tag tables (07) | the layer owns both as part of its contract, named once, and builds their objects through their factories |

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Delete the old-store refusals that misfire](01-delete-the-old-store-refusals-that-misfire.md) | **M1**–**M4** | Sonnet | ✍️ yes, 2026-10-04 | ✅ 2026-10-04 | `8636491` | [log](logs/01-delete-the-old-store-refusals-that-misfire.md) |
| 02 | [Delete the remaining factory refusals](02-delete-the-remaining-factory-refusals.md) | **N1**–**N3** | Sonnet | ✍️ yes, 2026-10-04; after 01 | ✅ 2026-10-04 | `9d3a42c` | [log](logs/02-delete-the-remaining-factory-refusals.md) |
| 03 | [Delete the reconcile's old-store paths](03-delete-the-reconciles-old-store-paths.md) | **R1**–**R4** | Opus | ✍️ yes, 2026-10-04 | ✅ 2026-10-04 | `d8d4afe` | [log](logs/03-delete-the-reconciles-old-store-paths.md) |
| 04 | [Delete the legacy shard records](04-delete-the-legacy-shard-records.md) | **L1**–**L3** | Sonnet | ✍️ yes, 2026-10-04; after 03 | ✅ 2026-10-04 | `bbdba04` | [log](logs/04-delete-the-legacy-shard-records.md) |
| 05 | [Refuse a store whose schema differs](05-refuse-a-store-whose-schema-differs.md) | **S1**–**S3** | Opus | ✍️ yes, 2026-10-05; after 04 | ⬜ | — | — |
| 06 | Give the layer its registry | **G1**–**G3** | Opus | ⏸ held on 01–05 (U2 taken) | ⬜ | — | — |
| 07 | Declare the replication facts in the factories | **F1**–**F4** | Opus | ⏸ held on 06 (U4 taken) | ⬜ | — | — |
| 08 | Derive the inventory's class list | **I1**–**I2** | Sonnet | ⏸ held on 06 | ⬜ | — | — |
| 09 | Keep the layer generic | **K1**–**K2** | Sonnet | ⏸ held on 01–08 | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not landed · ✅ landed.

**Orchestrator notes** are written one prompt at a time, just before its dispatch, so that each
carries the previous review's findings: [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md)
(2026-10-04, used for 01); [`orchestrator/prompt-02.md`](orchestrator/prompt-02.md) (2026-10-04,
used for 02); [`orchestrator/prompt-03.md`](orchestrator/prompt-03.md) (2026-10-04, used for 03);
[`orchestrator/prompt-04.md`](orchestrator/prompt-04.md) (2026-10-04, used for 04);
[`orchestrator/prompt-05.md`](orchestrator/prompt-05.md) (2026-10-05, not yet used), with the
measurement of 05's test plan in [`orchestrator/measure-05/`](orchestrator/measure-05/).

**Orchestrator review of prompt 01 (2026-10-04).** Dispatched from `6ea366e` to one Sonnet
subagent; reviewed against its commit `8636491`, with nothing landed after it. **Every check of the
note's §3 passed**; acceptance 1–5 are met, and no §5 stop condition fired.

*At dispatch.* The tree was clean, the only worktree was this checkout, and the registry held 16
runs: 14 finished, 2 `unknown`, none `running`. The only Ray processes were ChamPBH's, from its own
venv. **The note's gate did not hold as written.** `3b96ba1` (`a3-v2-readiness` 07a) landed after the
note and touches `extract_common.py`. It adds `quad_source_pairs` and `require_quad_source` and two
imports. The facts it could move were re-checked at `6ea366e` and handed to the agent as an addendum.
`extract_common`'s line numbers moved by two, and `extract_tensor_source_data.py`'s
`require_background` call is now at `:333`. M2's functions were unchanged. ComputeTargets is now
**698**, which includes `test_tensor_source_pairs`' 12. The baselines, serially at load 13–64:
AdaptiveLevin 32, ComputeTargets 698 (1 failure, `test_background_tau.test_build_cost`, which passed
alone in 2 of 3 re-runs), CosmologyModels 39, Datastore 598 (449 s), LiouvilleGreen 148 (1 skipped),
RunRegistry 281. The index held 108.

*The agent.* It stopped twice, waiting on its own suite run, and was resumed with `SendMessage`. On
resuming it was told three things. Remove the test helpers that only the eleven deleted tests used
(the log's IMPLEMENTATION CHOICE). Open no issue for the six `extract_*.py` comments; the residue
below fixes them. Its M2 choice, that `describe_background_generation` says so rather than raising,
is accepted.

The checks:
1. **Scope.** 13 files, all on the note's §2 list. The diff over `ShardedPool.py`,
   `Datastore/store_*.py`, the `extract_*_data.py` scripts, `main.py`, `config/`, `var/` and
   `ComputeTargets/BackgroundModel.py` is empty. Under `*/tests/`, M3's four modules changed and one
   module was added, `Datastore/tests/test_database_errors_reach_the_caller.py`.
2. **E1.** `grep -n 'in str(e)'` over `ObjectFactories/` prints nothing. `except SQLAlchemyError`
   survives only in the three prune blocks, now at `TkNumericIntegration.py:623`,
   `GkNumericIntegration.py:621` and `BackgroundModel.py:835`. They are shifted only by the deletions
   above them. Compared with `ast`, the only changed functions are:
   - the two `build`s and the value factories' `_build_impl_model`, which is what
     `TkNumericValue.build` and `GkNumericValue.build` delegate to;
   - `GkNumericValue.read_batch`;
   - `QCD_Cosmology.build`, whose module also lost the unused `SQLAlchemyError` import;
   - `BackgroundModel.build` and its `_build_query`.

   `validate_on_startup` is unchanged in all three.
3. **E5.** With `ast`, the test methods of the five modules before and after differ by exactly the
   eleven of the note's §0, and 13 new methods. Four are M2's (`TestChoosingARun` ×2,
   `TestDescribingABackgroundGeneration` ×2). Nine are M4's.
4. **E2, read.** `_build_query()` takes no flag and always selects `source_grid_digest` and
   `source_grid_construction`. It filters on them iff `source_grid_digest is not None`. No `hasattr`
   is left, `_source_grid_identity = None` on a miss is kept (`:356`), and a hit sets it from the
   row.
5. **E4, read.** `_StatementRaisingConnection.execute` compiles the query it is handed (SQLite
   dialect) and raises `OperationalError(<that text>, {}, Exception("database is locked"))`. No
   statement is a literal. Each test asserts that the error's `str()` names the old handler's
   columns, that the caller gets the same object (`assertIs`, no `__cause__`), and how many times
   `execute` ran: once on every path except the value-table read, where a real row is read first.
   That precondition check makes the module stronger than the note's (iii) assumed: a literal
   statement fails the test outright instead of letting it pass.
6. **E3, run.**
   - `choose_run_label([], None)` raises `RuntimeError`: "this datastore records no runs …".
   - `choose_run_label([], "x")` raises as before.
   - A model with no identity is described as "no grid generation (the datastore holds no
     background model for it)".
   - "prompt 14" survives only as the provenance of the run-label mechanism (`:2`, `:11`, `:47`),
     never as an older store.
7. **E4, the pins.** The five diffs were extracted from the log and replayed with `git apply` in
   `bash`:
   - (i): `test_tk_numeric_integration_build` errors.
   - (ii): the read-path test fails, because the connection was executed twice.
   - (iii) alone, and (i)+(iii): the Tk test fails on the precondition.
   - (i)+(iii)'s variant, which also empties the column list: `Ran 9`, **OK**. A test with a
     literal statement passes with the handler restored.
   - (iv): `test_a_store_with_no_runs_is_refused_saying_it_records_none` fails.

   Each was reverted with `git apply -R`, and the tree was clean after each.
8. **The probe**, from the scratchpad, through the M4 harness:
   - **At `6ea366e`** (an export, with the new module copied in): `TkNumericIntegration.build`
     raises `RuntimeError` saying "has no "break_point_kind" column … predates". `BackgroundModel`'s
     read path executes **twice**.
   - **At `8636491`:** both raise the injected `OperationalError` itself, after one `execute`.
9. **E6.**
   - The issue is in §4, with its closing note.
   - Its index row is gone. The index is 107 by counting, matching the header.
   - The fallbacks issue carries a dated "Narrowed" line.
   - §1, M1–M4 and §5.2 are written. §5.2 holds 01's after-counts, with the orchestrator's baseline
     beside them.
10. **Suites at `8636491`, serially, load 14–60:** AdaptiveLevin 32, ComputeTargets 693
    (698 − 9 + 4), CosmologyModels 39, Datastore 605 (598 − 2 + 9; 283 s), LiouvilleGreen 148
    (1 skipped), RunRegistry 281. ComputeTargets had 2 failures, both wall-clock tests:
    - `test_background_tau.test_build_cost` passed alone 3 times of 3.
    - `test_tk_wkb_phase.TestCost.test_wall_time_per_object` failed alone 3 times of 3 (0.072–0.110 s
      against 0.06 s), and failed 3 times of 3 on an export of `6ea366e` (0.065–0.107 s). It is
      load, not 01: 01 touches nothing it exercises.

    `black --check` is clean on the ten Python files.

*Residue, fixed in the follow-up commit that records this review.*
- "this commit" → `8636491` in §1, §4, §5.2 and the log.
- The six `extract_*_data.py` scripts said that `resolve_run_selection` explains "why a datastore
  that records no runs at all is read rather than refused". They now say that it is refused. The
  change is comment-only, and each script's `ast` is unchanged.

*Kept as is.* The agent's observations not acted on:
- the "pre-prompt-11" sentence in `BackgroundModel.py`'s docstring, which is the rationale for the
  grid-identity key;
- the three prune blocks that swallow their own error, which are known (`datastore-integrity` 02,
  `a3-v2-readiness` 02).

Neither opens an issue.

*Handed to 02.* Its note must carry three things:
- **The harness.** `_StatementRaisingConnection` in `test_database_errors_reach_the_caller.py`, with
  its precondition check, is the harness 02's pass-through tests reuse. It builds the schema in
  memory from `test_value_reads_parent_key._base_store`, and its `succeed=` handles a path whose
  first query must succeed.
- **The review's findings above.**
- **`real_store_fixtures.py`.** Its old-store machinery is still used by the reader and inventory
  tests, and stays until 05.

**Orchestrator review of prompt 02 (2026-10-04).** Dispatched from `bd5ad19` to one Sonnet
subagent; reviewed against its commit `9d3a42c`, with nothing landed after it. **Every check of the
note's §3 passed**; acceptance 1–4 are met, and no §5 stop condition fired.

*At dispatch.* The note's gate held: `cc4c1e9` and `bd5ad19`, after `c647ee2`, touch only this board
and the note. The tree was clean, the only worktree was this checkout, and the registry held 16
runs: 14 finished, 2 `unknown`, none `running`. The only Ray processes were ChamPBH's, from its own
venv. The baselines, serially at `bd5ad19`, load 14–76: AdaptiveLevin 32, ComputeTargets 693 (1
failure, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`), CosmologyModels 39, Datastore 605
(248 s), LiouvilleGreen 148 (1 skipped), RunRegistry 281. The index held 107.

*The agent.* It ran to its commit in one turn: no stop, no resume. It ran its suites in the
foreground, as the note asked after 01.

The checks:
1. **Scope.** 13 files, all on the note's §2 list. The diff over 01's factories, `QuadSource.py`,
   `ShardedPool.py`, `Datastore/store_*.py`, `real_store_fixtures.py`, the `extract_*.py` scripts,
   `main.py`, `config/` and `var/` is empty.
2. **E1.**
   - **Grep.** `grep -nE 'predate|older store|written before' Datastore/SQL/ObjectFactories/`
     prints nothing. `_refuse_a_store_that_predates_*` and `_TRANSFER_FUNCTION_COLUMNS` appear
     nowhere in the tree. `except SQLAlchemyError` in the four files survives only in the three
     prunes, now at `TkWKBIntegration.py:946`, `GkWKBIntegration.py:947` and `GkSource.py:826`.
   - **`ast`.** Against the pre-image, the changed functions are exactly the note's list: the two
     WKB `build`s, `_build_impl_model`s and `read_batch`es; `GkSource.build` and
     `GkSourceValue._build_impl_model`; `QuadSourceIntegral.build` and `read_batch`; and the three
     removed helpers. **Each changed function dumps identically to its pre-image once its
     `except SQLAlchemyError` handlers are removed and a `try` left with none is unwrapped.** So
     no other line moved, and the four `numeric_serial` selects keep their column lists; only their
     comments differ. `validate_on_startup` dumps identically in the three files that have one. At
     module level only the docstrings changed, plus `QuadSourceIntegral`'s import and its
     `_TRANSFER_FUNCTION_COLUMNS`.
3. **E3.** With `ast`, over the six test modules:
   - 19 method names go: the seventeen of the note's §0, plus the two kept tests under their old
     class names.
   - 14 appear: the two kept tests under new classes, and N3's twelve.
   - The moved `test_a_current_table_is_still_read` dumps identically to its pre-image.
   - No import is left unused that was not unused at `bd5ad19`.
4. **E2.** Under `*/tests/`, only the six modules changed.
5. **E4, read.**
   - Four new classes in `test_database_errors_reach_the_caller.py`: `TestTheWKBIntegrations`,
     `TestTheWKBValueReads`, `TestGkSource` and `TestQuadSourceIntegral`. Each goes through
     `_StatementRaisingConnection`, so the statement is the one the factory compiled. Each names its
     deleted handler's columns in `columns=`, asserts identity through `assert_passes_through`, and
     counts `execute`. `TestGkSource.test_the_read_of_the_parent_set_reaches_the_caller` uses
     `succeed=2` to reach the `GkSource_parents` read, the deleted handler's second match.
   - 01's harness and nine tests dump identically.
   - The rewritten kept test, `test_quadsource_policy_key.TestAnotherDatabaseErrorPassesThrough`,
     uses the current table and the harness. It has no `"SELECT 1"` and no `mock.patch`.
   - Keeping both it and N3's `QuadSourceIntegral` test is accepted (the log's deviation 5).
6. **The kept tests, run.** Both pass (23 tests with the harness module, OK). With (iii) applied,
   the moved test errors.
7. **E4, the pins.** The three diffs were extracted from the log and replayed with `git apply` in
   `bash`. Each failed exactly as the log says:
   - (i): `TestGkSource`'s three `GkSource.build` tests error.
   - (ii): `test_quad_source_integral_build` fails, and so does the rewritten kept test's `build`
     sub-test.
   - (iii): `test_a_current_table_is_still_read` errors.

   **A faithful restoration** was also applied: the pre-image's helper, unchanged, with its call in
   `GkSource.build`. `TestGkSource` then ran 4 tests, **OK**, as the note's §0 expected; it is not a
   gap. Each diff was reverted with `git apply -R`, and the tree was clean after each.
8. **The probe**, from the scratchpad, through the N3 harness, at `9d3a42c` and at an export of
   `bd5ad19`:
   - `GkSource.build` (keyed and reader mode) and `QuadSourceIntegral.read_batch` (untagged and
     tagged), given `database is locked` with the real statement, raise the injected
     `OperationalError` itself after one `execute`, **identically at both trees**. 02 changes no
     current behaviour.
   - A pre-07 `TkWKBIntegration` table, built from the 06 witness with one validated row and its
     `TkWKBValue`, was read labels-only with `TkWKBValue.read_batch`. At `bd5ad19` that raises
     `RuntimeError`: "… has no "numeric_serial" column. This datastore predates …". At `9d3a42c` it
     raises SQLAlchemy's `OperationalError: no such column: TkWKBIntegration.numeric_serial`, not a
     silent read. That is the kept select doing its job.
9. **E5.**
   - The fallbacks issue carries a dated "Narrowed (prompt 02)" line. The log's *Observations*
     hand the four selects to 05.
   - The index is 107 by counting, matching the header. Its row names 02's factories.
   - §1, N1–N3 and §5.3 are written.
10. **Suites**, serially at `9d3a42c` plus this review's docstring residue, load 8–13:
    AdaptiveLevin 32, ComputeTargets 693 (all passing; the wall-clock test passed this time),
    CosmologyModels 39, Datastore 600 (605 − 17 + 12; 165 s), LiouvilleGreen 148 (1 skipped),
    RunRegistry 281. Every suite is OK. `black --check` is clean on the ten Python files.

*Residue, fixed in the follow-up commit that records this review.*
- "this commit" → `9d3a42c` in §1, §5.3 and the log.
- **`test_database_errors_reach_the_caller.py`'s docstring.** It said 02's classes "pin that those
  stay deleted", which its next sentence contradicts. It now says that they pin that an error from
  the statements reaches the caller unchanged. The change is docstring-only.
- **The log's (iii).** It explained why N3 passes under (iii) by saying "the error is raised before
  any statement runs". It now says that the harness raises at `execute` whatever the statement,
  and that the broken statement still names `numeric_serial`.
- This board's note list marks `orchestrator/prompt-02.md` used.

*Kept as is.* The agent's observations not acted on:
- the four `numeric_serial` selects, assigned to 05 (S3);
- an absent `GkSource_parents` table now surfaces as an `OperationalError`, which is U1's case
  (05);
- the known prunes;
- two imports that were already unused at `bd5ad19`: `TkNumericIntegration` in
  `test_gksource_parent_set` and `sqla_TkWKBValue_factory` in `test_wkb_numeric_parent_key`.

None opens an issue.

*Handed on.*
- **03** is independent of 01–02 and edits `ShardedPool`. Its note carries 01's and 02's
  conventions, including foreground suites.
- **05**'s note must carry the four `numeric_serial` selects
  (`TkWKBIntegration`/`GkWKBIntegration`, `_build_impl_model` and `read_batch`), and the decision
  to keep them until the reader refuses a store whose columns differ. It must also carry the
  absent-`GkSource_parents` observation.

**Orchestrator review of prompt 03 (2026-10-04).** Dispatched from `65be64f` to one Opus subagent;
reviewed against its commit `d8d4afe`, with nothing landed after it. **Every check of the note's §3
passed**; acceptance 1–5 are met. **One stop condition fired**, the note's own: an existing test
outside the three deletions failed. The user decided it (below).

*At dispatch.* The note's gate held: `65be64f`, after `587a9ae`, touches only this board and the
note. The tree was clean, the only worktree was this checkout, `e53f323` is an ancestor of `HEAD`,
and the registry held 16 runs: 14 finished, 2 `unknown`, none `running`. The only Ray processes
were ChamPBH's, from its own venv. The baselines, serially at `65be64f`, load 14–149: AdaptiveLevin
32, ComputeTargets 693 (1 failure, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, which
failed alone twice at load 63–89, 0.091 s and 0.113 s against 0.06 s), CosmologyModels 39, Datastore
600 (765 s), LiouvilleGreen 148 (1 skipped), RunRegistry 281. The index held 107.

*The agent.* It stopped once, on the note's condition. With R2 alone, four existing tests in
`test_reconcile_at_open` failed every time they ran:
- `TestSerialSplit.test_with_the_record_set` and `test_with_no_record`;
- `TestMismatchLeftByTheCheck.test_a_vectorized_get` and `test_a_store`.

The cause is `_describe_differences`, which names a serial split by grouping rows on every compared
column but `serial`. A real split always comes from two separate writes, so its copies always differ
in `timestamp` too. Once `timestamp` is compared, a split is never named. Each store was still
refused and nothing was written; only the message changed. **The user approved, on 2026-10-04, the
agent's one-line fix**: the grouping ignores `timestamp` as well as `serial` (the log's deviation
1, STRUCTURALLY REQUIRED), pinned by a fifth breakage, (v). The agent was resumed with
`SendMessage` and finished in one more turn. It ran its suites in the foreground.

The checks:
1. **Scope.** 8 files, all on the note's §2 list. The diff over `Datastore.py`, `ObjectFactories/`,
   `Datastore/store_*.py`, `real_store_fixtures.py`, `shard_store_fixtures.py`, `config/`, `main.py`,
   `var/` and `orchestrator/` is empty; under `docs/` only `OPEN_ISSUES.md` changed. Under
   `*/tests/`, the three modules changed and one module was added,
   `Datastore/tests/test_one_timestamp_per_write.py`. `standin_pool.py` is untouched.
2. **E1, `ShardedPool.py`.**
   - **Grep.** `before prompt 01|written before|predates|created before` prints nothing.
     `has_table("replication_in_flight")`, `IntegrityError` and `"label" in columns` are gone. The
     one `replication_in_flight` creation left is `_write_shard_data`'s (`:730`), which builds a new
     primary.
   - **`ast`, against the pre-image.** The changed functions are `__init__`,
     `_replicated_table_specs`, `_read_in_flight_record`, `_commit_prune_record`,
     `_recompute_validated` and `_describe_differences`. The last is deviation 1. `_create_engine`
     dumps identically; its comment lost only the three `NOT NULL` lines. The repair licence's
     `row["timestamp"] != started` test is unchanged.
3. **E3, the tests, by `ast`.**
   - Exactly the note's three methods are gone, with `TestRecordTableBeforePrune`'s class body.
   - Four methods are new in `test_reconcile_at_open`: `test_a_timestamp_difference` (R2),
     `test_a_value_row_held_under_another_serial` (R1 a),
     `test_a_primary_without_the_record_table_is_refused` (R3) and
     `TestRecordDoesNotExplain.test_part_of_the_recorded_background_model` (R1 c).
   - No surviving method changed in any of the three modules. Their module-level changes are
     docstrings and one import (`OperationalError`).
4. **E4, `e53f323`, by reading the log.** The log says the argument holds, for the note's reasons:
   - a copy of key `(S,)` is planned only to a shard that holds nothing under S;
   - the case the deleted serial-free check refused is now a `differs` on key S, which the licence
     refuses;
   - value serials agree across a current store's shards, so the licence is unchanged.

   It names (b) as `TestKillAndReopen.test_store_of_a_background_model` and
   `test_store_of_a_second_background_model_beside_the_first`, and adds (a) and (c). For (c), no
   existing test reached the "holds its model row" branch with the record naming the same model:
   `test_part_of_a_background_model`'s record names the second model, and
   `test_a_value_count_difference` has no record. That agrees with the note's reading, and nothing
   was changed in `e53f323`.
5. **E2, the measurement.** `test_one_timestamp_per_write` reruns `TestKillAndReopen`,
   `TestPruningAfterRepair`, `TestInterruptedPrune`, `TestUninterruptedPrune` and three
   `test_version_row_at_open` classes with `sp.ds_mod.datetime` ticking from 2030. The driver's
   clock, in `ShardedPool`, is not patched, so only a shard's own stamp can carry a 2030 tick. Each
   inherited `assertIdentical` (full rows, `timestamp` included) also asserts that no replicated row
   on any shard carries a tick. The prune classes reach it through `assertPruned` and
   `assertCompleted`. The version classes compare every replicated table in `tearDown`. The clean
   write is `TestCleanWrite.test_every_copy_is_row_identical`, which already ticked. Run by name, the
   module and that test: 29 tests, OK (318 s at load 193).
6. **The new refusals, run.** The four new tests in `test_reconcile_at_open`: 4 tests, OK.
7. **E5, the pins.** The five diffs were extracted from the log and replayed with `git apply` in
   `bash`:
   - (i): `test_a_timestamp_difference` fails.
   - (ii), on the measurement module and the clean write (29 tests): 2 failures, 46 errors.
     `TestCleanWrite.test_every_copy_is_row_identical` fails; every `TestRepairAtOpen` and
     `TestRepairThenPruneAtOpen` case errors, refused at open; so do both prune classes and a version
     test.
   - (iii): 8 errors, exactly the log's: four commit points of
     `test_store_of_a_background_model`, the second-model test, both `TestPruningAfterRepair`
     sub-cases and `TestRecordsStayApart`'s background-model test.
   - (iv), corrected: both sub-tests of R3's test fail. **The prompt's (iv) as written** (the
     creation alone) gives `TestNoRecordRefuses` 9 tests, **OK**, as the note expected.
   - (v): exactly the four tests of deviation 1 fail.

   Each was reverted with `git apply -R`, and the tree was clean after each.
8. **The probe**, from the scratchpad, through `standin_pool`, at exports of `65be64f` and
   `d8d4afe`:
   - **A clean store, written and reopened**, opens at both, repairs nothing, and its checksums
     are unchanged.
   - **A `BackgroundModel` store interrupted on one replica**, reopened, is repaired identically at
     both (the model, its tag and its three values copied from the controller), and every
     replicated table is row-identical on every shard, `timestamp` included.
   - **A store whose copies of one `tolerance` row differ only in `timestamp`** opens at
     `65be64f`, read-write and read-only. At `d8d4afe` both opens are refused with
     `ReplicatedDivergence`, naming `tolerance`, the shard and `timestamp`.
   - **A primary with `replication_in_flight` dropped** opens at `65be64f`, which recreates the
     table. At `d8d4afe` it is refused with `OperationalError: no such table:
     replication_in_flight`; the primary is unchanged and the table is not created.
9. **E6.**
   - The fallbacks issue carries a dated "Narrowed (2026-10-04, prompt 03)" line naming 03's
     deletions and `e53f323` as reviewed. Its index row names 03's part.
   - The index is 107 by counting, matching the header.
   - §1, R1–R4 and §5.4 are written.
10. **Suites**, serially at `d8d4afe`, load 8–63:
    - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped), RunRegistry 281;
    - Datastore 629 (600 − 3 + 4 + 28; 417 s), OK;
    - ComputeTargets 693, with `test_quadsource_policy_main` passing (19 tests run alone) and 1
      failure, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`. It failed alone 3 times of 3
      at load 42–56 (0.081–0.108 s against 0.06 s), as it did at `65be64f`: load, not 03.

    `black --check` is clean on the five Python files.

*Residue, fixed in the follow-up commit that records this review.*
- "this commit" → `d8d4afe` in §1, §5.4 and the log.
- `_describe_differences`' docstring read "whose other compared values but timestamp are the same".
  It now reads "whose compared values other than serial and timestamp are the same". The change is
  docstring-only: the function's `ast`, without its docstring, is unchanged.
- This board's note list marks `orchestrator/prompt-03.md` used.

*Kept as is.* The agent's observations not acted on:
- a primary without `replication_in_flight` now surfaces as SQLAlchemy's `OperationalError`, not a
  named refusal. That is README §4's contract for 03, and 05's question if its single refusal
  should cover the primary's own tables;
- the unit check's foreign-key lookup names `"BackgroundModel"` (rule 11; 07's, F1).

Neither opens an issue.

*Handed on.*
- **04** is not yet written. Its note must say that 03 left `write_legacy_primary` untouched, and
  that it builds a primary without `replication_in_flight`, which no longer opens. Its current users
  never open it through the check at open.
- **05**'s note must carry three things beside 02's four `numeric_serial` selects:
  - the absent-table and absent-column tolerance in `_read_replicated_tables` (`:1224`), which 03
    left alone;
  - the `OperationalError` for a primary without its record table;
  - the split grouping's exclusion of `timestamp`, which is current behaviour, not a fallback.

**Orchestrator review of prompt 04 (2026-10-05).** Dispatched from `2ed3469` to one Sonnet
subagent. Reviewed against its commit `bbdba04`, with nothing landed after it. **Every check of the
note's §3 passed**, and acceptance 1–5 are met. No stop condition fired, and the agent ran its
suites in the foreground.

*At dispatch.* The note's gate held: the commits after `4ca36c4` (`ddc266d`, `52a5d39`, `d1e7a50`,
`2ed3469`) touch only prompts, this board, the README, other boards and the index. The tree was
clean, and the only worktree was this checkout. The registry held 16 runs (14 finished, 2
`unknown`, none `running`). The only Ray processes were ChamPBH's. The A3 baseline's directory
listing showed only its sidecar, with `retired.state` `retired`. The baselines, serially at
`2ed3469`, load 25–251:
- AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped), RunRegistry 281,
  Datastore 629 (1784 s);
- ComputeTargets 693, with 2 failures, `test_background_tau.test_build_cost` and
  `test_tk_wkb_phase.TestCost.test_wall_time_per_object`. Both failed alone at load 23 (0.596 s
  against 0.5 s, and 0.109 s against 0.06 s).

The index held 107.

The checks:
1. **Scope.** 17 files, all on the prompt's §6 list. The diff over `RunRegistry/*.py` (tests aside),
   `Datastore.py`, `ObjectFactories/`, `Datastore/store_*.py`, `real_store_fixtures.py`, `config/`,
   `main.py`, `var/` and `orchestrator/` is empty. Under `docs/` only `OPEN_ISSUES.md` and the sweep
   script changed.
2. **E1, by grep.** The prompt's §3.2 greps print nothing. `PurePath(stored).is_absolute()` appears
   nowhere, and `is_legacy_record` appears nowhere outside `prompts/` and `docs/`.
3. **E1, by `ast`, against the pre-images, docstrings stripped.**
   - `shard_paths.py`: only `resolve_shard_path` changes (one `_require_bare_name` call), and
     `is_legacy_record` and the `PurePath` import are gone.
   - `ShardedPool.py`: only `_resolve_shard_rows` and the import change. `copy_store` and
     `delete_store` dump identically.
   - The sweep script: only `assert_store_is_self_consistent` changes. Its `ValueError` is caught
     per serial and reported among the mismatches, so the check raises its own `RuntimeError`.
   - The audit tool: only `main` changes.
     - `shard_key_config` is required, and so is a row in it.
     - The `wavenumber_serial` branch is gone.
     - Every record is resolved before the audit (log deviation 1), and an unusable one exits 2.
     - A missing shard file still skips the cross-file check.
4. **E4, the tests, by `ast`.**
   - Exactly the note's fifteen methods are gone, with `TestLegacyRecords`,
     `TestCopyOfALegacySource`, `TestLegacyRecordsNamingAnotherStore` and `TestLegacyShape`.
   - One method was renamed, as the prompt allows: `test_moved_store_with_no_shards_is_refused`.
     Its body differs only in building a bare-name store.
   - Every re-fixture differs only in the records it builds.
     `TestCopiedStore.test_copied_directory_reads_its_own_shards_not_the_originals` now asserts
     that B's copied records are the bare names, where it asserted that they named A's files (log
     deviation 3). Its precondition needed absolute records, and the property (B's pool reads B's
     files, and A is byte-identical) is unchanged.
   - Every split loses only its legacy case. The collapsed helpers (`source(kind)` → `new_source`,
     `_cases` → `LAYOUTS`, `store(shape)` → `new_store()`, `SHAPES`, the `other` parameter) gave
     the surviving cases exactly as before.
   - Every kept test differs only in the fixture's name. So does `write_hand_built_primary`'s
     body.
   - The new methods are the two new modules (12 + 7), the sweep refusal and the retire refusal.
5. **E2, the new module.** It was read, then run by name with the audit, sweep and retire refusal
   tests: 21 tests, OK. Each entry point is driven against B, and each asserts "Shard #0",
   "unusable record", the record, and `tree_state` of the whole temporary root, both stores
   included.
6. **E3, by running.** `test_shard_file_name`, `test_store_reader`, `test_sharded_store_script`,
   `test_store_copy_move` and the five kept tests: 56 tests, OK.
7. **E5, the pins.** The five diffs were extracted from the log and replayed with `git apply` in
   `bash`:
   - (i): 20 failures and 1 error over the 21 tests of check 5. That is every refusal test, and
     neither control.
   - (ii): the same 21. Only `shard_paths.py` showed in `git status` under it, and `var/` was
     unchanged (names, mtimes and sizes).
   - (iii): `test_a_primary_without_shard_key_config` fails.
   - (iv): the three unusable-record audit tests fail.
   - (v): both sub-tests of `test_the_check_refuses_an_absolute_record` error.

   The tree was clean after each `git apply -R`. (My first pass at (i) and (ii) passed the test
   list unsplit under zsh and loaded nothing. The figures are from the rerun in `bash`.)
8. **The probe**, from the scratchpad, at `bbdba04` and at an export of `2ed3469`, with
   `shard_store_fixtures`:
   - **A bare-name store**, at both trees:
     - `read_pool` reads it;
     - `copy_store`, `move_store` and `delete_store` leave identical trees;
     - the audit gives identical stdout and exit code.

     On the hand-built current store that is `VERDICT: OK`, exit 0. On `write_new_store`'s
     placeholders it is a traceback, exit 1, and only the traceback's line number differs.
   - **B**:
     - at `2ed3469`, `read_pool`, `copy_store`, `move_store`, `closed_store_files` and
       `delete_store` (both `resume` values) read B's own siblings, with the relocation notice;
     - at `bbdba04`, every entry point, `open_read_only` included, refuses with "has an unusable
       record … is not a bare file name";
     - A is unchanged at both.
   - **C**, whose absolute records name its own siblings: read at `2ed3469`, refused at `bbdba04`.
   - **The audit tool:**
     - without `shard_key_config`: exit 0 and "Cross-file check … was not possible" at
       `2ed3469`, exit 2 at `bbdba04`;
     - on `{0: "../outside.sqlite"}`: exit 0 and "VERDICT: OK" at `2ed3469`, exit 2 at
       `bbdba04`.
9. **E6.**
   - The fallbacks issue carries "Narrowed (2026-10-04, prompt 04)", and its index row names 04's
     part.
   - `[04-closed-store-refusals-repeat-their-prefix]` is in §3 with the measured message, and in
     the index.
   - The index held 108 by counting, matching its header.
   - §1, L1–L3 and §5.5 are written.
10. **Suites**, serially at `bbdba04`, load 14–40:
    - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped);
    - RunRegistry 280 (281 − 3 + 2), OK;
    - Datastore 636 (629 − 12 + 19, 959 s), OK;
    - ComputeTargets 693, with the same 2 failures as at `2ed3469` and
      `test_quadsource_policy_main` passing (19 tests).

    Run alone, `test_build_cost` passed and `test_wall_time_per_object` failed (0.066 s against
    0.05 s), as it did alone before the change. That is load, not 04, which touches nothing in
    `ComputeTargets`. `black --check` is clean on the 14 Python files.

*Residue, fixed in the follow-up commit that records this review.*
- "this commit" → `bbdba04` in §1, §5.5 and the log.
- The audit tool's module docstring had one line of 118 characters, which `black` does not
  reflow. It is rewrapped, and the change is docstring-only.
- This board's note list marks `orchestrator/prompt-04.md` used.
- **Opened** `[04-the-shard-key-audit-crashes-where-it-should-refuse]` (§3, and the index, 109),
  from the log's third observation. It was measured at `bbdba04`, and is current behaviour that
  04 did not change.

*Kept as is.* The log's other observations:
- The verification scripts under `docs/` that read `shards.filename` directly
  (`a3-v2-readiness/rehearsal/closed_store_state.py` reads the final component) are measurements of
  past runs, and none goes through the resolver.
- `RunRegistry/stores.py:81-82` is U3's.
- `quiet` in `test_delete_store` is now a pass-through, kept to leave its call sites alone.

None opens an issue.

*Handed on.* **05**'s note must carry:
- the absent-table and absent-column tolerance in `_read_replicated_tables`;
- 02's four `numeric_serial` selects;
- the bare `OperationalError` for a primary without `replication_in_flight` (03);
- the split grouping's exclusion of `timestamp`, which is current behaviour (03).

04's log adds that `write_hand_built_primary` builds a primary without `replication_in_flight`, so
the constructor refuses it. 05's refusal of a store whose tables differ must not be pinned with that
fixture through the constructor, unless the test means the absent table.

---

## 2. Items

| Item | Prompt | What | Audit | Status |
|---|---|---|---|---|
| **M1** | 01 | the old-store handlers in `TkNumericIntegration`, `GkNumericIntegration`, `QCD_Cosmology`, `BackgroundModel` (incl. the read-path re-query) | F0, B9 | ✅ |
| **M2** | 01 | `extract_common`: no "unknown generation" run; a store with no runs refused | B8 | ✅ |
| **M3** | 01 | the tests of M1–M2's deleted paths | F0, B8 | ✅ |
| **M4** | 01 | pass-through tests with the factories' real statement text | F0 | ✅ |
| **N1** | 02 | the old-store refusals in `TkWKBIntegration`, `GkWKBIntegration`, `GkSource`, `QuadSourceIntegral` | B7 | ✅ |
| **N2** | 02 | their tests, with the two current-behaviour tests kept | B7 | ✅ |
| **N3** | 02 | pass-through tests | B7 | ✅ |
| **R1** | 03 | review `e53f323` (merged 2026-10-04): `BackgroundModelValue` under its serial | B2 | ✅ |
| **R2** | 03 | measure one timestamp on every replicated write path; compare `timestamp` | B1 | ✅ |
| **R3** | 03 | `replication_in_flight` assumed present | B3 | ✅ |
| **R4** | 03 | the `controller_shard NOT NULL` refusal; `label` read unconditionally | B4, B9 | ✅ |
| **L1** | 04 | absolute shard records out of `shard_paths.py` and `ShardedPool` | B6 | ✅ |
| **L2** | 04 | `tools/shard_key_audit.py`'s old-primary paths, and an unusable record refused; the sweep script's check and docstring | B6 | ✅ |
| **L3** | 04 | `write_legacy_primary` renamed for the malformed-record tests; legacy tests deleted or re-fixtured; an absolute record pinned as refused by every reader | B6 | ✅ |
| **S1** | 05 | the reader and the inventory refuse absent or extra tables and columns | B5 | ⬜ |
| **S2** | 05 | the read-only pool likewise; the reconcile's column-subset, `unkeyed` and `columns` logic deleted | B5 | ⬜ |
| **S3** | 05 | an absent table on a read-write open, per U1, and every other difference refused there; the primary refused by both constructors if it differs from `_create_engine`'s tables (03's hand-on; confirmed by the user, 2026-10-05); `real_store_fixtures.py`'s old-store machinery; the four WKB value reads' `numeric_serial` selects (found by the 02 note) | B5 | ⬜ |
| **G1** | 06 | `ShardedPool` given the registry; no import of `Datastore._factories` | C1 #10 | ⬜ |
| **G2** | 06 | `store_reader` and `store_inventory` given the registry and the replicated set | C1 #10, #14 | ⬜ |
| **G3** | 06 | the older registries, per U2 | C1 #9–#11 | ⬜ |
| **F1** | 07 | the copy-as-a-unit rule derived from foreign keys | C1 #2 | ⬜ |
| **F2** | 07 | monotone flags declared by the factories | C1 #5 | ⬜ |
| **F3** | 07 | the `validated` rule declared by the factory; `_StoredModel` gone | C1 #6 | ⬜ |
| **F4** | 07 | the version and tag tables per U4; project names out of `ShardedPool`'s comments and messages | C1 #1, #7, #8 | ⬜ |
| **I1** | 08 | the inventory's class list and order derived | C1 #12 | ⬜ |
| **I2** | 08 | the polymorphic cosmology reference declared by the client | C1 #13 | ⬜ |
| **K1** | 09 | a test that the generic layer names no registry table and imports no client package | §C | ⬜ |
| **K2** | 09 | every suite; README §5 rule 9's closing verification | — | ⬜ |

---

## 3. Active and unresolved issues

### 3.1 Opened by this campaign's audit (2026-10-04)

- **[00-the-generic-datastore-layer-hard-codes-project-schema-facts]** *(opened 2026-10-04 by the
  audit, §C; read)*
  - **The defect.** `ShardedPool`, `store_reader`, `store_inventory`, `Datastore` and `ClientPool`
    hard-code fourteen facts about the project's tables (audit C1): `BackgroundModel`'s copy unit and
    `validated` rule, the `redshift`/`wavenumber` flags, the version and tag tables, the factory
    registry imported as a private name, the inventory's class list, and more. `ShardedPool` had
    one project literal before `f4b2c91`.
  - **Impact.** The layer cannot be reused without editing it, and every new table that needs
    replication-time treatment is another special case in the pool. No wrong result.
  - **Next step.** Prompts 06–08, then 09's guard.
  - **Assigned (2026-10-04):** prompts 06–09.
- **[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]** *(opened 2026-10-04 by the
  audit, §B; read, and G4 measured)*
  - **The defect.** About thirty sites exist only to open, read, repair or refuse a store written by
    older code (audit B1–B9), with their tests and fixtures. No store needs them: the only store
    under `var/` that predates them, the A3 baseline, is already readable only at the SHA that wrote
    it (G4).
  - **Impact.** Maintenance, and code the refactor of 06–08 would otherwise have to preserve; one
    group of them is also a defect (`[00-old-store-refusals-misfire-on-any-database-error]`).
  - **Next step.** Prompts 01–05 delete them, each narrowing this entry; 05 closes it.
  - **2026-10-04, while 04 was written:** the A3 baseline was retired by the user on 2026-10-04 (completed 21:31:59) with `store retire` run at `81c0a9a`.
    The registry at `HEAD` refused it already, because the recorded fingerprint is format 1 and
    `HEAD` computes format 6. At `81c0a9a` a fresh fingerprint matched, digest `433b7fc3…`, and
    exactly its five files were deleted; the sidecar is the tombstone. No store under `var/` now
    holds an old-store shape, so 04 costs no store anything.
  - **Narrowed (2026-10-04, prompt 01):** deleted, with their tests: the old-store refusals in
    `TkNumericIntegration` (`build`, `TkNumericValue.build`), `GkNumericIntegration` (`build`,
    `GkNumericValue.build`, `GkNumericValue.read_batch`), `QCD_Cosmology.build` and
    `BackgroundModel.build` (the Gauss-order refusal, the grid-identity refusal and the read-path
    re-query, the value-table refusal), and `extract_common`'s "no run labels" and "unknown
    generation" branches (audit F0, B8, B9's comment); the eleven tests that exercised them. Left
    for 02: `TkWKBIntegration`, `GkWKBIntegration`, `GkSource`, `QuadSourceIntegral` (B7); for
    03–05: B1–B6.
  - **Narrowed (2026-10-04, prompt 02):** deleted, with their tests: the old-store refusals in
    `TkWKBIntegration` (`build`, `TkWKBValue.build`, `TkWKBValue.read_batch`), `GkWKBIntegration` (the
    same three), `GkSource` (`_refuse_a_store_that_predates_the_parent_set`, in `GkSource.build` and
    `GkSourceValue.build`; and the false "readable by this factory without a migration" sentence)
    and `QuadSourceIntegral` (the policy-key and transfer-function-key refusals, in `build` and
    `read_batch`) (audit B7); seventeen tests. The four WKB `numeric_serial` selects below are kept.
    Left for 03–05: B1–B6, the four selects (05, S3).
  - **Narrowed (2026-10-04, prompt 03):** deleted from `ShardedPool`: the exclusion of
    `timestamp` from the check at open (B1; every replicated write path was first measured to
    stamp one timestamp, `Datastore/tests/test_one_timestamp_per_write.py`); the constructor's
    creation of `replication_in_flight` on an existing store and `_read_in_flight_record`'s
    `has_table` test (B3); `_commit_prune_record`'s `controller_shard NOT NULL` handler and its
    comment (B4); `_recompute_validated`'s conditional read of `label` (B9). With them, three
    tests: `test_reconcile_at_open.TestNoRecordRefuses.test_a_store_written_before_the_record_existed`,
    `test_replicated_write.TestCleanWrite.test_an_existing_store_gains_an_empty_record_table_when_opened`,
    `test_prune_at_open.TestRecordTableBeforePrune.test_a_pruning_open_refuses_before_any_shard_is_touched`.
    `e53f323` (B2, deleted outside the campaign) **reviewed**: its argument holds, and nothing
    in it was changed. The serial-split naming now ignores `timestamp` as well as `serial` (a
    stop condition, decided by the user). Left for 04–05: B5, B6, the four selects (S3).
  - **Narrowed (2026-10-04, prompt 04):** deleted: the reading of an absolute shard record as a
    sibling by its final component, with `is_legacy_record` and the relocation notice
    (`Datastore/shard_paths.py`, `ShardedPool._resolve_shard_rows`; B6), so that every reader
    refuses an absolute record as unusable; `tools/shard_key_audit.py`'s pre-`a2bd966` branch and
    its tolerance of a primary without `shard_key_config` (the tool now also refuses an unusable
    record, exit 2, where it said OK); fifteen tests, and the fixture `write_legacy_primary`
    renamed `write_hand_built_primary`. The sweep check raises its own `RuntimeError` on an
    absolute record. Left for 05: B5, the four selects (S3). `RunRegistry/stores.py:81-82` still
    describes a sidecar rule as "prompt 01's rule for legacy shard records"; that is U3's.
  - **Found (2026-10-04, writing 02's orchestrator note):** a site the audit missed. In
    `TkWKBIntegration` and `GkWKBIntegration`, `_build_impl_model` and `read_batch` select
    `numeric_serial` whether or not a parent is given, so that a store without the column fails
    rather than being read; nothing reads the value. Deleting the selects before the reader refuses
    a store whose columns differ would let such a store be read silently, so 02 keeps them and
    rewords their comments, and **05 deletes them** (item S3). Confirmed by the user, 2026-10-04.
  - **Assigned (2026-10-04):** prompts 01–05.
- **[04-closed-store-refusals-repeat-their-prefix]** *(opened 2026-10-04 by prompt 04; measured)*
  - **The defect.** A refusal of a closed store's operation prints its prefix twice. On a hand-built
    primary with `{0: "../outside.sqlite"}`, `ShardedPool.closed_store_files` and `delete_store`
    raise
    `Cannot delete sharded datastore "<p>": Cannot delete sharded datastore "<p>": Shard #0 of primary database "<p>" has an unusable record: shard record '../outside.sqlite' is not a bare file name; ShardedPool only ever records shards that are siblings of the primary, so this row was written by something else. Nothing was deleted`;
    `copy_store` and `move_store` raise
    `Cannot copy sharded datastore "<p>" to "<d>": Cannot copy sharded datastore "<p>": Shard #0 … Nothing was written`
    (and "Cannot move …"). In the delete case the two `<p>`s can differ, because one is the path
    as given and the other resolved (`/var/folders/…` and `/private/var/folders/…`).
    `store_reader.open_read_only`'s message is not doubled.
  - **Cause.** `_read_closed_store` wraps the message of the refusal with the verb's prefix, and
    its caller (`_plan_deletion`, `_relocate_store`) wraps it again. Current code, not 04's.
  - **Impact.** The text is longer than it need be. Nothing is read or written wrongly. Prompt 04
    makes this the message an absolute shard record meets in those four operations.
  - **Next step.** The user will have it cleaned up (2026-10-04). The new
    `Datastore/tests/test_absolute_shard_record_refused.py` asserts fragments of the message, not
    its prefix, so a cleanup does not break it.

---

## 4. Resolved issues

- **[04-the-shard-key-audit-crashes-where-it-should-refuse]** *(opened 2026-10-05 by the
  orchestrator's review of prompt 04, from the log's observations; measured at `bbdba04`)*
  - **The defect.** `tools/shard_key_audit.py` ends in a Python traceback and exit code 1, with no
    verdict, in two cases:
    - shard 0's file exists but is not a database: `ATTACH` raises
      `sqlite3.DatabaseError: file is not a database`, and the tool catches only
      `OperationalError` there;
    - `shard_key_config` exists without a `key_type` column:
      `sqlite3.OperationalError: no such column: key_type`, from the `SELECT` that prompt 04 moved
      ahead of the audit (the same `SELECT` raised the same error before 04, inside its
      `if "shard_key_config" in tables`).
  - **Impact.** Exit code 1 is also the tool's `VERDICT: INCONSISTENT` code, so a caller that
    reads only the exit code is told "rebuild this store" where the tool has not audited it. Nothing
    is written: the tool opens everything `mode=ro`. No current store has either shape; the first is
    what the shard fixtures' placeholder files look like to it.
  - **Next step.** Both should be refusals with exit 2, like the tool's other refusals. Not 04's
    (unchanged by it, README §5 rule 4); for the user to assign.
  - **Found (2026-10-05, measuring prompt 05's test plan):** a third shape, read rather than a
    traceback. When shard 0 exists but lacks the key-type table, the tool printed "skipping
    cross-file checks" and could still end `VERDICT: OK`; an `OperationalError` on the `ATTACH`
    was skipped the same way.
  - **Closed (2026-10-05, outside the prompts, at the user's request; this commit).** Each shape
    is now a refusal with exit code 2 and no verdict: a primary without a column the tool reads
    (`shards.serial`/`.filename`, `shard_keys.shard_id`, `shard_key_config.key_type`); a shard 0
    file that cannot be read as a database (any `sqlite3.DatabaseError` from the `ATTACH` or the
    reads after it); a shard 0 without the key-type table. A missing shard 0 *file* still skips
    the cross-file check and gives a verdict, as prompt 04 kept it. The module docstring now
    states the three exit codes. Pinned by four new tests in
    `Datastore/tests/test_shard_key_audit_refusals.py`
    (`test_a_shard_key_config_without_key_type`, `test_a_shards_table_without_filename`,
    `test_a_shard_0_that_is_not_a_database`, `test_a_shard_0_without_the_shard_key_table`), each
    failing against the unfixed tool; the module's seven other tests and
    `test_shard_key_audit_copy`'s three pass unchanged. Run with `ShardedPool` stubbed out of the
    fixtures' imports, because the venv's scipy cannot be loaded on macOS 27.0.1 (its
    `_spropack` extension is rejected by dyld); the tests use only the fixtures' stdlib parts.
- **[00-old-store-refusals-misfire-on-any-database-error]** *(opened 2026-10-04 by the audit, F0;
  measured)*
  - **The defect.** `TkNumericIntegration`, `GkNumericIntegration`, `QCD_Cosmology` and
    `BackgroundModel` decide that a store predates a column by testing `"<column>" in str(e)`.
    `str(e)` carries the SQL statement, which names the column, so every database error matches.
  - **Measured** (`docs/datastore-generic-audit/str_e_misfire_probe.py`): on a current schema, a
    `database is locked` and an unrelated `no such table` both match all four tests.
  - **Impact.** No stored value is wrong. A transient lock during a build is reported as a stale
    store, with advice to regenerate it; on `BackgroundModel`'s read path the query is silently
    re-run without grid identity, and the extraction scripts report the background as of unknown
    generation. The tests meant to catch this build the error with the statement `"SELECT ..."`.
  - **Next step.** Prompt 01 deletes the four refusals (D2) and adds pass-through tests with the
    real statement text.
  - **Closed (2026-10-04, prompt 01, `8636491`).** The four old-store refusals are deleted, so a
    database error reaches the caller as itself, and `BackgroundModel` always selects its grid
    identity and never re-queries. Pinned by
    `Datastore/tests/test_database_errors_reach_the_caller.py` (nine tests: `TkNumericIntegration`
    and `GkNumericIntegration` `build`, both value `build`s, `GkNumericValue.read_batch`,
    `QCD_Cosmology.build`, and `BackgroundModel` on the read path, the compute path and the
    value-table read), each raising the error with the compiled statement of the query the factory
    really issued, so that `str(e)` names the columns as in the probe, and asserting that the
    caller gets the same object and that `BackgroundModel`'s read path executes once. Deliberate
    breakage (log): restoring `TkNumericIntegration.build`'s handler, and restoring the read-path
    re-query, each fail them; with the statement replaced by the literal `"SELECT ..."` and the
    column check dropped, they pass with the handler restored, which is why the statement is
    captured.

---

## 5. Baselines

### 5.1 At the plan (2026-10-04, `dbd8a51`)

| Suite | Result |
|---|---|
| Datastore | 598 OK (428 s) |
| Datastore, on `e53f323` | 598 OK (769 s, under load) |

The other suites: see `a3-v2-readiness`'s board §5; re-measured before 01's dispatch.

### 5.2 After prompt 01 (2026-10-04, `8636491`)

Measured serially on the staged tree, under load (another project's Ray cluster running).

| Suite | Before (orchestrator, `6ea366e`) | After | Difference |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 0 |
| ComputeTargets | 698 (1 failure, `test_background_tau.test_build_cost`, wall clock) | 693 (1 failure, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, wall clock) | −9 deleted, +4 added |
| CosmologyModels | 39 OK | 39 OK | 0 |
| Datastore | 598 OK | 605 OK (294 s) | −2 deleted, +9 added |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 0 |
| RunRegistry | 281 OK | 281 OK | 0 |

The one ComputeTargets failure fails alone under this load and also on an export of `6ea366e`; see
the log.

### 5.3 After prompt 02 (2026-10-04, `9d3a42c`)

Measured serially on the staged tree.

| Suite | Before (orchestrator, `bd5ad19`) | After | Difference |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 0 |
| ComputeTargets | 693 (1 failure, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, wall clock) | 693 OK | 0 (the wall-clock test passed) |
| CosmologyModels | 39 OK | 39 OK | 0 |
| Datastore | 605 OK | 600 OK (161 s) | −17 deleted, +12 added |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 0 |
| RunRegistry | 281 OK | 281 OK | 0 |

The two kept tests (one moved unchanged, one rewritten) do not move the count.

### 5.4 After prompt 03 (2026-10-04, `d8d4afe`)

Measured serially on the staged tree.

| Suite | Before (orchestrator, `65be64f`) | After | Difference |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 0 |
| ComputeTargets | 693 (1 failure, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, wall clock) | 693, 2 failures (`test_background_tau.test_build_cost`, which passed alone 3 times of 3; `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, which failed alone 3 times of 3 and also 3 times of 3 on an export of `65be64f`: load) | 0 |
| CosmologyModels | 39 OK | 39 OK | 0 |
| Datastore | 600 OK (765 s) | 629 OK (716 s) | −3 deleted, +4 added in `test_reconcile_at_open`, +28 in `test_one_timestamp_per_write` |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 0 |
| RunRegistry | 281 OK | 281 OK | 0 |

`test_one_timestamp_per_write`'s 28 are 27 existing tests rerun under a ticking clock plus one
new; they add about 100 s to the Datastore suite.

### 5.5 After prompt 04 (2026-10-04, `bbdba04`)

Measured serially on the staged tree, at load 21–77. The Datastore suite was run in four groups of
modules (`discover -p 'test_[a-f]*.py'`, `[g-o]`, `[p-r]`, `[s-z]`), because a single Bash call
cannot run for more than ten minutes; the four counts are 122, 132, 201 and 181.

| Suite | Before (orchestrator, `2ed3469`) | After | Difference |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 0 |
| ComputeTargets | 693 (2 failures, `test_background_tau.test_build_cost` and `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, wall clock) | 693 (the same two failures, wall clock: both fail alone at load 77, 0.68 s against 0.5 s and 0.085 s against 0.05 s; `test_quadsource_policy_main` passes, 19 tests) | 0 |
| CosmologyModels | 39 OK | 39 OK | 0 |
| Datastore | 629 OK | 636 OK | −12 deleted, +12 in `test_absolute_shard_record_refused`, +7 in `test_shard_key_audit_refusals` |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 0 |
| RunRegistry | 281 OK | 280 OK | −3 deleted, +1 sweep refusal, +1 retire refusal |
