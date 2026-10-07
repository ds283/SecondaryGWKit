# datastore-generic campaign — implementation state

**Last updated:** 2026-10-07 (§4: the closed-store prefix issue, closed by `datastore-generic-followup` 03, the last of this campaign's issues; before it, the resolve issue by 02 and the drop issue by 01) · **Status: CLOSED — 9 of 9 prompts landed (01–09).** The closing rehearsal rebuild's fingerprint equals `dbd8a51`'s (K2, 2026-10-06). The audit ([`docs/datastore-generic-audit.md`](../../docs/datastore-generic-audit.md)) is
written. Prompts 01–09 are written. The user took U1–U4
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
| **U2** — the older registries (06) | `_factories`, `_drop_actions`/`_drop_order` and `ClientPool`'s batch sizes move to `config/`, passed in by `main.py`. *2026-10-05, while 06 was written:* the layer is given the tables to drop, in an order derived from foreign keys, and the inventory reads the replicated set from the store (README §6.2) |
| **U3** — the registry's legacy records | out of scope |
| **U4** — the `version` and tag tables (07) | the layer owns both as part of its contract, named once, and builds their objects through their factories |

**Choices of method put to the user while prompts were written (README §6.2), each taken as
recommended:** for 07 (2026-10-06), a table whose rows belong to another's declares the column
naming its owner; for 08 (2026-10-06), the inventory's order is derived from a declarative
`inventory_spec` per factory, roots first, and the polymorphic cosmology reference is declared in
the referencing `Parent`. For 09 (2026-10-06), four choices:
- `config.defaults` is allow-listed, where a constant of the layer's own was recommended;
- `utilities` is allow-listed;
- 07's and 08's per-file name tests are folded into the guard;
- rule 9 is closed on fixtures and by one rehearsal rebuild at the review, README §3's one
  exception.

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Delete the old-store refusals that misfire](01-delete-the-old-store-refusals-that-misfire.md) | **M1**–**M4** | Sonnet | ✍️ yes, 2026-10-04 | ✅ 2026-10-04 | `8636491` | [log](logs/01-delete-the-old-store-refusals-that-misfire.md) |
| 02 | [Delete the remaining factory refusals](02-delete-the-remaining-factory-refusals.md) | **N1**–**N3** | Sonnet | ✍️ yes, 2026-10-04; after 01 | ✅ 2026-10-04 | `9d3a42c` | [log](logs/02-delete-the-remaining-factory-refusals.md) |
| 03 | [Delete the reconcile's old-store paths](03-delete-the-reconciles-old-store-paths.md) | **R1**–**R4** | Opus | ✍️ yes, 2026-10-04 | ✅ 2026-10-04 | `d8d4afe` | [log](logs/03-delete-the-reconciles-old-store-paths.md) |
| 04 | [Delete the legacy shard records](04-delete-the-legacy-shard-records.md) | **L1**–**L3** | Sonnet | ✍️ yes, 2026-10-04; after 03 | ✅ 2026-10-04 | `bbdba04` | [log](logs/04-delete-the-legacy-shard-records.md) |
| 05 | [Refuse a store whose schema differs](05-refuse-a-store-whose-schema-differs.md) | **S1**–**S3** | Opus | ✍️ yes, 2026-10-05; after 04 | ✅ 2026-10-05 | `bc8bffc` | [log](logs/05-refuse-a-store-whose-schema-differs.md) |
| 06 | [Give the layer its registry](06-give-the-layer-its-registry.md) | **G1**–**G3** | Opus | ✍️ yes, 2026-10-05; after 05 | ✅ 2026-10-06 | `19f07ee` | [log](logs/06-give-the-layer-its-registry.md) |
| 07 | [Declare the replication facts in the factories](07-declare-the-replication-facts-in-the-factories.md) | **F1**–**F4** | Opus | ✍️ yes, 2026-10-06; after 06 | ✅ 2026-10-06 | `3f51939` | [log](logs/07-declare-the-replication-facts-in-the-factories.md) |
| 08 | [Derive the inventory's class list](08-derive-the-inventorys-class-list.md) | **I1**–**I2** | Sonnet | ✍️ yes, 2026-10-06; after 07 | ✅ 2026-10-06 | `6646b06` | [log](logs/08-derive-the-inventorys-class-list.md) |
| 09 | [Keep the layer generic](09-keep-the-layer-generic.md) | **K1**–**K2** | Sonnet | ✍️ yes, 2026-10-06; after 08 | ✅ 2026-10-06 | `e980b6e` | [log](logs/09-keep-the-layer-generic.md) |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not landed · ✅ landed.

**Orchestrator notes** are written one prompt at a time, just before its dispatch, so that each
carries the previous review's findings: [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md)
(2026-10-04, used for 01); [`orchestrator/prompt-02.md`](orchestrator/prompt-02.md) (2026-10-04,
used for 02); [`orchestrator/prompt-03.md`](orchestrator/prompt-03.md) (2026-10-04, used for 03);
[`orchestrator/prompt-04.md`](orchestrator/prompt-04.md) (2026-10-04, used for 04);
[`orchestrator/prompt-05.md`](orchestrator/prompt-05.md) (2026-10-05, used for 05), with the
measurement of 05's test plan in [`orchestrator/measure-05/`](orchestrator/measure-05/);
[`orchestrator/prompt-06.md`](orchestrator/prompt-06.md) (2026-10-05, used for 06), with the
measurement of 06's test plan in [`orchestrator/measure-06/`](orchestrator/measure-06/);
[`orchestrator/prompt-07.md`](orchestrator/prompt-07.md) (2026-10-06, used for 07), with the
measurement of 07's test plan and of the check at open in
[`orchestrator/measure-07/`](orchestrator/measure-07/);
[`orchestrator/prompt-08.md`](orchestrator/prompt-08.md) (2026-10-06, used for 08), with the
measurement of 08's test plan and of the inventory, the report and `resolve()` in
[`orchestrator/measure-08/`](orchestrator/measure-08/);
[`orchestrator/prompt-09.md`](orchestrator/prompt-09.md) (2026-10-06, used for 09), with the
measurement of 09's test plan, of the guard on the layer and of rule 9 on fixtures from
`dbd8a51` in [`orchestrator/measure-09/`](orchestrator/measure-09/).

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

**Orchestrator review of prompt 05 (2026-10-05).** Dispatched from `6ba15c7` to one Opus
subagent. Reviewed against its commit `bc8bffc`, with nothing landed after it. **Every check of the
note's §3 passed**, and acceptance 1–5 are met. No stop condition fired, and the agent ran its
suites in the foreground.

*At dispatch.* The note's gate held:
- the commits after `bc3e944` touch code only in `b97cf06` (the audit tool and its test) and
  `8abb3da` (`requirements.txt`), both already allowed for in the note;
- the tree was clean, and the only worktree was this checkout;
- the registry held 16 runs (14 finished, 2 `unknown`, none `running`), and no Ray process came
  from this repository's venv;
- the venv imported `scipy.sparse.linalg` (Python 3.12.15, scipy 1.16.3).

The baselines, serially at `6ba15c7`, load 13–36, were the first taken under scipy 1.16.3:
- AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped), RunRegistry 280;
- ComputeTargets 693, OK, with no wall-clock failure;
- Datastore 640 (122 + 132 + 201 + 185).

The index held 108. The pre-images of the eighteen files of the note's §1.6 were recorded, and
`6ba15c7` was exported to the scratchpad.

The checks:
1. **Scope.** 22 files, all on the prompt's §6 list.
   - The diff over `RunRegistry/`, `tools/`, `main.py`, `extract_*.py`, `config/`,
     `shard_paths.py`, `replication.py`, `var/` and `orchestrator/` is empty.
   - In `ObjectFactories/` only the two WKB files changed. Under `docs/` only `OPEN_ISSUES.md`
     changed.
2. **E1, by grep.** The §3.2 greps print nothing. The one exception: `build_old_store|OLD_MISSING`
   matches `orchestrator/measure-05/scripts/`. Those are the committed shims of the measurement at
   `bc3e944`, a record of the tree before 05. `ReadOnlyShard`'s fields are exactly `serial`,
   `path`, `engine` and `tables`.
3. **E1, by `ast`, against the pre-images, docstrings stripped.**
   - `schema.py`: `build_schema` dumps identically. `SchemaDifferences`, `schema_differences` and
     `StoreSchemaMismatch` are new, and the module still imports only `sqlalchemy`. The message
     has one prefix ("Cannot read …" or "Cannot open …"), then the file, then each difference by
     kind and name.
   - `store_reader.py`: only `_describe_shard`, `open_read_only` and `ReadOnlyShard` change.
   - `store_inventory.py`: only `read_records`, `_combine`, `read_inventory`, `ShardContext` and
     `ShardRead` change. The tag and value branches are dedented and otherwise unchanged.
   - `ShardedPool.py`:
     - changed: `__init__`, `_open_read_only`, `_version_labels_present`,
       `_reconcile_replicated_tables` (one call, after `built`), `_read_replicated_tables`,
       `_replicated_differences`, `_describe_differences`, `_plan_replicated_repair` and
       `_read_full_rows`;
     - new: `_refuse_a_primary_that_differs` and `_refuse_shards_that_differ`;
     - every other function dumps identically, including `_prune_needed`, `_prune_shard` and the
       closed-store operations.
   - `Datastore.py`: only `_open_read_only` (the read-only branch of the constructor),
     `_validate_on_startup_read_only` and `read_only_state` change.
   - The WKB factories: each of the four functions loses only the `numeric_serial` select and its
     comment.
4. **E4, the tests, by `ast`.**
   - Exactly the plan's nine methods are gone.
   - The three conversions are renamed (log), keep their store shapes and assert
     `StoreSchemaMismatch`.
   - The two re-fixtures change only the rows they build (orphan tag rows).
   - Every other changed method is one the plan names: the four WKB value reads, the two
     hand-built `ShardContext`s, `test_store_reader`'s helpers, `TestNoRay` and the reconcile
     test's exception.
   - The thirteen new methods are all in `test_store_schema.py`, and the log names each one's
     purpose.
5. **E2, by reading and running.** `test_store_schema.py` was read in full.
   - Each refusal asserts:
     - the exception and its `differences`;
     - the file by name;
     - unchanged checksums;
     - for a pool, that no stand-in actor was constructed and that the in-flight record was not
       read.
   - The U1 tests assert that every shard then has no difference and holds the version row.
   - The module, the conversions, the re-fixtures and the changed tests: 254 tests, OK.
6. **E3, by running.** 148 tests, OK: `test_reconcile_at_open`, `test_version_row_at_open`,
   `test_prune_at_open`, `test_replicated_write`, `test_copy_move_store`, `test_delete_store` and
   `test_absolute_shard_record_refused`.
7. **E5, the pins.** The five diffs were extracted from the log and replayed with `git apply` in
   `bash`, over the log's six modules (143 tests):
   - (i): failures=3, errors=2;
   - (ii): errors=3, the three U1 tests and nothing else;
   - (iii): failures=5;
   - (iv): failures=2, errors=4;
   - (v): failures=3.

   Each named exactly the tests the log names. The tree was clean after each `git apply -R`.
8. **The probe**, from the scratchpad, at `bc8bffc` and at the export of `6ba15c7`:
   - **The fingerprint** of `build_full_store`, with `taken` removed, is byte-identical (`cmp`):
     digest `6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759`, problems `{}`.
   - **`build_real_store`**: every table of every shard read through `open_read_only`, and
     `read_inventory`'s problems, are identical at both trees.
   - **A missing table, a missing column, an extra column and an extra table**, each on one shard:
     - at `6ba15c7`, all four were read. The missing table gave `absent-table`; the other three
       gave no problem, and only the reader's fields recorded them;
     - at `bc8bffc`, `open_read_only` and `read_inventory` refuse all four, each naming the
       shard, its path and the table or column.
   - **`python -m RunRegistry store fingerprint`** on the missing-table store at `bc8bffc`, with a
     scratch `--runs-root`: exit 1, the message on stderr through `!!`, no traceback. The store's
     checksums and listing were unchanged, and the runs root stayed empty.
   - **The pool, on stand-in shards**, for the same four shapes on shard 1:
     - the read-only pool refuses all four, through the reader's "Cannot read …";
     - a read-write open refuses all but the absent table ("Cannot open …"), and opens that one;
     - every refusal built no actor and changed no file.

     The tests pin only the absent table on the read-only pool. The other three reach it through
     the reader, and this probe measures them directly.
   - **U1**: `measure-05/scripts/m6_u1.py`'s three cases ran unchanged against `bc8bffc`. All
     three opened, and afterwards every shard matched the schema and held the version row.
9. **E6.**
   - The fallbacks issue is in §4 with a dated "Closed (2026-10-05, prompt 05)" line naming 05's
     deletions, and its index row is gone.
   - The index holds 107 by counting, matching its header.
   - §1, S1–S3 and §5.6 are written.
10. **Suites**, serially at `bc8bffc`, load 7–38:
    - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped);
    - RunRegistry 280, OK;
    - ComputeTargets 693, OK;
    - Datastore 644 (640 − 9 + 13; 122 + 131 + 200 + 191), OK.

    `black --check` is clean on the 19 Python files.

*Residue, fixed in the follow-up commit that records this review.*
- "this commit" → `bc8bffc` in §1, §4, §5.6 and the log.
- §4's closing line for `[04-the-shard-key-audit-crashes-where-it-should-refuse]` still said
  "this commit" from `b97cf06`. It now names `b97cf06`.
- `_combine` kept a local named `comparable` and an `if len(comparable) > 0` guard. The local now
  holds every shard, and `reads` is never empty: `min(reads)` precedes the guard. The local is
  renamed `shards`, the always-true guard goes, and a comment says why every shard is compared.
  By `ast`, only `_combine` changes.
- The three unused imports the log names (`TkNumericIntegration` in `test_gksource_parent_set`,
  `sqla_TkWKBValue_factory` in `test_wkb_numeric_parent_key`, `Record` in `test_store_inventory`)
  are removed. Each was unused before 05 too, in a module 05 edits.
- This board's note list marks `orchestrator/prompt-05.md` used.

These were verified by Datastore and RunRegistry at the follow-up (counts unchanged), and by the
fingerprint digest again.

*Kept as is.*
- `_read_replicated_tables` keeps one `PRAGMA table_info` per table, only to detect an absent
  table (U1).
- `_refuse_shards_that_differ` imports `read_only_url` at function scope, because `store_reader`
  imports `ShardedPool` at module scope.
- Both messages end "so nothing was read", for the reason the log gives: a read-write open may
  already have rolled back a journal.

*Handed on.* **06**'s note must carry:
- every caller of `schema_differences` (the reader's `_describe_shard`, `_refuse_shards_that_differ`)
  takes `built.tables` from `build_schema(…, _factories)`, which `store_reader` and `ShardedPool`
  import. That is exactly what 06 replaces with an injected registry, so every call site of the
  check is one of 06's;
- the primary's six tables, checked by `_refuse_a_primary_that_differs`, are `ShardedPool`'s own:
  the layer's contract, not the registry's;
- U1's only pins are `test_store_schema`'s three recovery tests. Under breakage (ii),
  `TestNewStoreInterrupted`, `TestPruningAfterRepair.test_an_interrupted_store` and
  `TestDropOrder` still pass, because none reaches an absent table;
- the split grouping's exclusion of `timestamp` (03) remains current behaviour.

**Orchestrator review of prompt 06 (2026-10-06).** Dispatched from `154d868` to one Opus
subagent. Reviewed against its commit `19f07ee`, with nothing landed after it. **Every check of the
note's §3 passed**, and acceptance 1–5 are met. No stop condition fired, and the agent ran its
suites in the foreground.

*At dispatch.* The note's gate held at `f87df17`:
- the tree was clean, and the only worktree was this checkout;
- the registry held 16 runs (14 finished, 2 `unknown`, none `running`), and no Ray process was up;
- the venv imported `scipy.sparse.linalg`.

The first baseline did not match 05's review. At `f87df17` the Datastore `[g-o]` group failed one
test, `test_inventory_retired.TestNoReferenceRemains.test_the_grep_prints_nothing`. Its repo-wide
grep for the retired inventory service's names matched `def inventory(` in
`orchestrator/measure-06/scripts/m1_fingerprint.py`, which the note's own commit added. The local
helper was renamed `take_inventory` in `154d868`, which touches no code, before dispatch. The agent
would otherwise have met a failure outside its plan.

The baselines, serially at `154d868`, at load 4–10, match 05's review:
- AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped);
- RunRegistry 280, OK; ComputeTargets 693, OK;
- Datastore 644 (122 + 131 + 200 + 191).

The index held 108. The pre-images of the 48 files of the note's §1.6 were recorded, and `154d868`
was exported to the scratchpad. The parent side of check 8 was measured before dispatch, on a
`build_full_store` built there.

The checks:
1. **Scope.** 54 files, all on the prompt's §6 list.
   - The diff over `docs/` (but `OPEN_ISSUES.md`), `ObjectFactories/`, `replication.py`,
     `shard_paths.py`, `ProfileAgent.py`, `SerialPoolBroker.py`, `config/sharding.py`, `var/` and
     `orchestrator/` is empty.
   - On the `datastore-integrity` board only the closed issue moves, from §3 to §4.
2. **E1, by grep.**
   - Greps 1, 2 and 4 print nothing.
   - Grep 3 prints three lines of prose in factory docstrings (`GkSource.py:18`,
     `GkWKBIntegration.py:35`, `TkWKBIntegration.py:33`), where the pattern's `.` matches the `/`
     of `config/sharding.py`. No layer file imports `config` beyond `config.defaults`.
   - Grep 5 prints `main.py`'s `allowed_drop_actions`, `specified_drop_actions` and `drop_actions`,
     which §2 keeps, and the method name `test_drop_actions_and_prune_refuse_…`, which §2 also
     keeps. The log records this as deviation 12.
   - `TestTheLayerImportsNoClient` passes. It is a deny-list, as the prompt specifies. It forbids
     `config.datastore`, `config.sharding`, every `ObjectFactories` module but the package and
     `base`, and `ComputeTargets`, `CosmologyConcepts` and `CosmologyModels`. Its comment names
     `MetadataConcepts` and `utilities` as the two allowed, and why.
3. **E1, E2, by `ast`, against the pre-images, docstrings stripped.**
   - `Datastore.py`: only `__init__` and `_open_read_only` change, and `_drop_actions` becomes
     `_drop_tables`. The factory imports, `_factories`, `_drop_actions` and `_drop_order` are gone.
   - `ShardedPool.py`:
     - changed: exactly the note's eight functions, each by `self._factories`, the actor
       arguments and `drop_tables` only;
     - new: `factories` and `_refuse_undeclared_drop_tables`;
     - every other function dumps identically.
   - `schema.py`: only `drop_order` is new.
   - `store_reader.py`: `open_read_only` and `ReadOnlyStore` change, and `_read_replicated_tables`
     is new. `store_inventory.py`: only `read_inventory` changes.
   - `ClientPool.py`: only `SerialPoolManager.__init__` and `lease_serial` change.
   - `config/datastore.py`, in-process against the pre-images:
     - `factories` has the 38 keys of `_factories` in the same order, each `is` the same class;
     - `serial_batch_sizes` equals `_default_serial_batch_size`, in order;
     - `drop_groups` has each group's tables unchanged, and its keys in the pre-image `main.py`'s
       `allowed_drop_actions` order;
     - `ray.cloudpickle` round-trips `factories` (2936 bytes) to the same classes, without
       `ray.init`.
   - The callers change only the lines the prompt names.
4. **E4, the tests, by `ast`.**
   - No method is gone but the three `TestDropOrder` renames. The fourth conversion keeps its name.
   - The ten import-only modules differ only in the import.
   - Every other changed method is one the plan names, changed only by the registry argument,
     `drop_tables` or the `SimpleNamespace`.
   - The twenty new methods are all in `test_layer_registry.py`, and the log names each.
   - Deviation 1 is right. With `drop_order` patched to the identity, `["tk-wkb", "tk-numeric"]`
     is a valid drop on `TestDropOrder`'s store, which holds no value rows. Only the other order
     pins that the actor sorts.
5. **E2, by reading and running.** `test_layer_registry.py` was read in full.
   - Each refusal asserts the exception, the names, and unchanged checksums or file state.
   - The pool refusals also assert that no stand-in actor was constructed and no engine made.
   - The full-store drop runs with `PRAGMA foreign_keys = ON` and asserts `foreign_key_check` is
     empty.
   - The module (20 tests), and the changed tests with `TestDropOrder` (25 tests): OK.
6. **E3, by running.** 260 tests, OK: `test_store_schema`, `test_reconcile_at_open`,
   `test_version_row_at_open`, `test_prune_at_open`, `test_replicated_write`, `test_read_only_pool`,
   `test_store_inventory`, `test_inventory_report`, `test_inventory_consumers`, `test_store_reader`,
   RunRegistry's `test_store_fingerprint` and ComputeTargets' `test_reader_read_only`.
7. **E5, the pins.** The five diffs were extracted from the log and replayed with `git apply` in
   `bash`, each against its named tests:
   - (i): errors=3, the full-store drop on both shards and `test_the_derived_order_drops_both`;
   - (ii): failures=1, `test_the_inventory_compares_exactly_the_recorded_classes`;
   - (iii): failures=2, both sub-tests of the read-write refusal;
   - (iv): failures=1, the batch-size test;
   - (v): failures=2, the reader and inventory sub-tests.

   The tree was clean after each `git apply -R`.
8. **The probe**, from the scratchpad, at `19f07ee` and at the export of `154d868`:
   - **The fingerprint** of `build_full_store`, with `taken` removed, is byte-identical (`cmp`):
     digest `6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759`, problems `{}`.
     So are the full inventory and `build_real_store`'s fingerprint and inventory.
   - **`main.py`** on a `build_full_store` primary gives byte-identical stdout and stderr at both
     trees:
     - `--inventory`: 14835 bytes, exit 0;
     - `--inventory --drop gk-source`: exit 2;
     - `--drop nonsense`: exit 2;
     - `--help`: the same.

     The store's checksums were unchanged throughout.
   - **`python -m RunRegistry store fingerprint`**: the same digest and exit 0 at both trees. Its
     output is identical apart from the `taken` and git lines, and the runs root stayed empty.
   - **A full drop on stand-in shards** (`measure-06/scripts/m3b_pool_drop.py`): `sqlite_master`,
     the row counts and `foreign_key_check` are identical (`cmp`). `19f07ee` prints 25 drop lines
     per shard, where `154d868` printed 10.
   - **Imports** (`m5_imports.py`, `ShardedPool`):
     - at `154d868`: 22 `ObjectFactories` modules, 21 `ComputeTargets`, 3 `CosmologyConcepts`,
       11 `CosmologyModels` and 6 `MetadataConcepts`;
     - at `19f07ee`: the package and `base`, no compute target, 6 `MetadataConcepts`, and
       `config` with `config.defaults`.
9. **E6.**
   - `[07-drop-actions-list-a-parent-table-before-its-children]` is in `datastore-integrity`'s §4,
     with a dated "Closed (2026-10-06, prompt 06 of `prompts/datastore-generic`)" line, and its
     index row is gone.
   - The generic-layer issue has a dated "Narrowed" line naming rows 9, 10, 11 and 14, and its
     index hook says so.
   - The index holds 107 by counting, matching its header.
   - §1, G1–G3 and §5.7 are written.
10. **Suites**, serially at `19f07ee`, load 4–6:
    - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped);
    - RunRegistry 280, OK;
    - ComputeTargets 693, OK;
    - Datastore 664 (644 + 20; 122 + 151 + 200 + 191), OK.

    The agent's one ComputeTargets failure was `test_wall_time_per_object`, at load ~23; it did
    not recur here. `black --check` is clean on the 50 Python files.

*Residue, fixed in the follow-up commit that records this review.*
- "this commit" → `19f07ee` in §1, §3, §5.7, the log and the `datastore-integrity` board's
  closing line.
- Five imports were unused before 06, in modules 06 edits, and are removed:
  - `redshift` and `wavenumber_exit_time` in `test_exact_identity_lookups`;
  - `ClassInventory` in `test_inventory_report_labels`;
  - `Optional` in `test_read_path_mechanics`;
  - `tolerance` in `main.py`.
- This board's note list marks `orchestrator/prompt-06.md` used.

These were verified at the follow-up: Datastore 664, RunRegistry 280 and ComputeTargets 693, all
OK; `black --check` clean; and `main.py --inventory` and `--help` byte-identical to `154d868`'s
again.

*Kept as is.*
- Grep 3's and grep 5's matches, explained above.
- The import test is a deny-list. It does not pin that `LiouvilleGreen`, `AdaptiveLevin` and
  `Units` are no longer loaded, though they are not (the log's lists). 09's guard generalises it.
- Both reader refusals name the primary twice, as the reader's existing "records no shards" does.

*Handed on.* **07**'s note must carry:
- `ShardedPool._recompute_validated` still names `BackgroundModel`, through
  `self._factories["BackgroundModel"]` (C1 row 6).
- `ShardedPool` and `store_reader` still load `MetadataConcepts`, through `store_tag` (C1 row 1,
  U4). `test_layer_registry`'s import test allows it only by not forbidding it; its comment names
  it. 07 should add `MetadataConcepts` to `FORBIDDEN_PACKAGES` once the import is gone.
- The registry now arrives through `ShardedPool(…, *, factories, serial_batch_sizes=None)`.
  - A fact 07 declares in a factory's `register()` reaches the pool through
    `build_schema(…, self._factories).records`.
  - It reaches the reader and the inventory through their `factories` arguments.
- `schema.drop_order` orders any set of tables by their foreign keys. F1's copy-as-a-unit rule can
  derive from the same keys.
- The `docs/` scripts of the prompt's §6 no longer run at `HEAD`. They are records, run at the SHA
  that wrote them.
- `read_inventory` raises `KeyError` if its registry lacks a name in `INVENTORY_CLASSES`. That list
  is C1 row 12, prompt 08's.
- The pool's undeclared-drop check is the only guard before a store opens. Without it, a new store
  given an unknown drop table is created silently (log, breakage (iii)).

**Orchestrator review of prompt 07 (2026-10-06).** Dispatched from `f1b8d02` to one Opus
subagent. Reviewed against its commit `3f51939`, with nothing landed after it. **Every check of the
note's §3 passed**, and acceptance 1–5 are met. No stop condition fired, and the agent ran its
suites in the foreground.

*At dispatch.* The note's gate held at `f1b8d02`:
- the tree was clean, and the only worktree was this checkout;
- `c6bddf8..f1b8d02` touches only `prompts/`;
- the registry held 16 runs (14 finished, 2 `unknown`, none `running`), and no Ray process was up;
- the venv imported `scipy.sparse.linalg`.

The baselines, serially at `f1b8d02`, at load 7–35 (a transient spike from system daemons), match
06's review:
- AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped);
- RunRegistry 280, OK; ComputeTargets 693, OK;
- Datastore 664 (122 + 151 + 200 + 191), OK. No committed measurement file tripped a grep test.

The index held 107. The pre-images of the 16 files of the note's §1.6 were recorded, and
`f1b8d02` was exported to the scratchpad. The parent side of check 8 was measured there before
dispatch:
- the fingerprint, digest `6576b85d…`;
- the M2 record, 111 tests and 328 constructions, byte-identical over two runs;
- `main.py --inventory` on a copy of the `build_full_store` primary: 14839 bytes, exit 0.

The checks:
1. **Scope.** 23 files, all on the prompt's §6 list.
   - The diff over `docs/` (but `OPEN_ISSUES.md`), `store_reader.py`, `ClientPool.py`,
     `replication.py`, `shard_paths.py`, `ProfileAgent.py`, `SerialPoolBroker.py`, `config/`,
     `main.py`, `extract_*.py`, `RunRegistry/`, `tools/`, `var/` and `orchestrator/` is empty.
   - The only change under `Datastore/tests/data/` is the new witness; the 09c witness is
     unchanged.
2. **E1, by grep and by running.**
   - Greps 1 and 3 print nothing.
   - Grep 2 prints the five lines §3.2 allows: `Datastore.py:745`, `schema.py:117` and `:121`, and
     `store_inventory.py:116-117`.
   - `test_layer_registry` passes with `MetadataConcepts` in `FORBIDDEN_PACKAGES`.
   - `m6_names_left.py`, extended to the five layer files, at both trees:
     - `ShardedPool.py` goes from 107 hits to 42, `Datastore.py` from 19 to 11 and
       `store_inventory.py` from 58 to 57.
     - What remains is the contract (`version`, `label`), the layer's vocabulary (`validated`),
       generic English (a "source" store, a `config` local) and the inventory's class list
       (08's).
     - The log's list omitted `schema.py` (six hits: `version` and `validated`). It was added at
       this review (below).
3. **E1, E2, by `ast`, against the pre-images, docstrings stripped.**
   - **`ShardedPool.py`:**
     - changed: exactly the note's ten functions, plus `replication` and `check` nested in
       `_store_impl_replicated_table`;
     - gone: `_prune_unit`, `_value_serials` and `_StoredModel`;
     - new: `_unit_tables`, `_is_tag_table_of`, `_declares_owner`, `_column_naming`, and
       `_find_version_row`'s nested `refuse_insert`;
     - every other function dumps identically.
   - **The other layer files:**
     - `Datastore.py`: only `_insert` changes; `_refuse_insert` changes in its docstring only;
     - `schema.py`: only `build_schema` changes, and its four checking helpers are new;
     - `store_inventory.py`: only `read_records`;
     - `base.py`: the two hooks are new.

     Every module-level change is an import.
   - **The factories:**
     - the six with one argument: `inventory_records` only;
     - `redshift` and `wavenumber`: `register` only;
     - `BackgroundModel.py`:
       - both `register`s, `validate` and `inventory_records` change;
       - `_validate_row`, `revalidate` and `owned_serials` are new;
       - everything else dumps identically.
   - **In-process, on the real registry:**
     - exactly the four declarations are non-default;
     - `_unit_tables` equals the old copy unit for all 14 compared tables, and `BackgroundModel`'s
       equals `_prune_unit`'s as a set;
     - every refusal of §2 raises on a hand-built registry, naming the class and the key. A
       fitting declaration is accepted.
4. **E4, the tests, by `ast`.**
   - In `test_schema_builder` and `test_layer_registry` no function changes. The module-level
     changes are `WITNESS`, the docstring, the comment and `FORBIDDEN_PACKAGES`.
   - The 28 new methods are all in `test_declared_facts.py`, and the log names each.
5. **E2, by reading and running.** `test_declared_facts.py` was read in full.
   - Each refusal test asserts `"Member"` and the key.
   - The `revalidate` test compares (return, flag written, printed) as one tuple, as stored and
     with a value row deleted, in rolled-back savepoints, and the copy's checksum is unchanged.
   - The version test records the factory's calls through a subclass, read-write and read-only:
     one `mode=ro` build each, an inserter that raises, and `vars()` exactly `_my_id`, `_label`,
     `_deserialized`.
   - The module with `test_schema_builder` and `test_layer_registry`: 55 tests, OK.
6. **E3, by running.** 232 tests, OK: the five check-at-open modules, `test_store_inventory`,
   `test_inventory_report`, `test_store_schema`, `test_store_reader` and RunRegistry's
   `test_store_fingerprint`.
7. **E5, the pins.** The eight diffs ((iv) in two forms) were extracted from the log and replayed
   with `git apply` in `bash`. Each failed exactly the tests and counts the log gives:
   - (i): failures=6, errors=14;
   - (ii): failures=7, errors=14;
   - (iii): failures=1, errors=2;
   - (iv-a): failures=3, errors=5;
   - (iv-b): failures=3, `test_a_recomputed_validated_that_disagrees` among them;
   - (v): failures=2, both in `test_declared_facts`; no `test_replicated_write` test;
   - (vi): failures=1; (vii): failures=1.

   The tree was clean after each `git apply -R`.
   - On (v), the log's account is right. "its value rows have serials" appears in no test at
     `f1b8d02`. The pool's owned-serials check was untested until 07's hook test.
   - Deviation 9, which runs (iv) in two forms, is right: an unwritten `True` leaves the shards
     differing, and the second comparison refuses.
8. **The probe**, from the scratchpad, at `3f51939` and at the export of `f1b8d02`:
   - **The fingerprint** of `build_full_store`, with `taken` removed, is byte-identical (`cmp`).
     The digest is `6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759`, problems
     `{}`. The full inventory, and `build_real_store`'s fingerprint (`d9328538…`) and inventory,
     are byte-identical too. `sqlite_master`'s type, name, table and SQL are identical in all six
     files.
   - **The check at open** (`m2_open_behaviour.py`) is byte-identical: 111 tests (110 with a
     construction), 328 constructions (56 raised), 982 printed lines.
   - **The messages** (`m7_messages.py`): all 16 scenarios are byte-identical, the prune refusal
     included.
   - **`main.py --inventory`** on the same copied primary: byte-identical stdout (14839 bytes) and
     stderr, exit 0. The store's checksums were unchanged.
   - **`revalidate`** (`m9_revalidate.py`) agrees with `validate` in return, flag and warning,
     for both models, as stored and with a value row deleted. The checksum was unchanged.
   - **Imports** (`m5_imports.py`): at `f1b8d02`, `ShardedPool` and `store_reader` each loaded 6
     `MetadataConcepts` modules. At `3f51939`, none of the five layer modules loads any, and none
     initialises Ray.
   - **The witness** (`witness_diff.py`): only the three keys are added, on all 38 classes, with
     non-default values on exactly `redshift`, `wavenumber`, `BackgroundModel` and
     `BackgroundModelValue`.
9. **E6.**
   - The generic-layer issue has a dated "Narrowed" line naming rows 1, 2, 4–8, the three keys
     and the two hooks. Its index hook says so and assigns it to 08–09.
   - The index holds 107 by counting, matching its header.
   - README §4 names the keys, the hooks, `Datastore/contract.py` and `_unit_tables`.
   - §1, F1–F4 and §5.8 are written.
10. **Suites**, serially at `3f51939`, load 12–48:
    - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped);
    - RunRegistry 280, OK;
    - ComputeTargets 693, OK;
    - Datastore 692 (664 + 28; 150 + 151 + 200 + 191), OK.

    No flake appeared. `black --check` is clean on the 18 Python files.

*Residue, fixed in the follow-up commit that records this review.*
- "this commit" → `3f51939` in §1, §3, §5.8, README §4 and the log.
- The log's "Project names left in the five layer files" gains a `schema.py` entry, marked as
  added at review. The agent's scan listed `ClientPool.py` in its place.
- This board's note list marks `orchestrator/prompt-07.md` used.

*Kept as is.*
- Grep 2's five lines, which §3.2 allows.
- **Index root pages.** `sqlite_master.rootpage` of the indexes differs between two builds of
  `build_full_store` on one tree (`f1b8d02` against itself): SQLAlchemy keeps a table's indexes in
  a set ordered by object identity. The parent-against-child comparison therefore compares the
  schema text, not root pages. Nothing 07 changed, and nothing the fingerprint reads.
- `_find_version_row` builds the whole schema to reach the version table, once per open of an
  existing store (the log's observation). It writes nothing.
- The generic-layer issue's `**Assigned (2026-10-04):** prompts 06–09` line is the original
  assignment. The dated "Narrowed" lines record what each prompt took.

*Handed on.* **08**'s note must carry:
- What is left in `store_inventory.py` for 08: C1 rows 12 and 13, the class list and its order
  (`INVENTORY_CLASSES`, the two contract names at `:116-117` among them), and the cosmology
  reference.
- `read_records` now reads the tag table through `Datastore.contract` (`TAG_TABLE`, `TAG_LABEL`,
  `TAG_SERIAL`). It reads the validated column through the factory's `validated=<column>`
  argument.
- `schema.drop_order` and the foreign keys are there to derive the class order.
  `build_schema`'s records now also carry `owner_column`, which names a table's owner.
- `ShardedPool`'s messages keep BackgroundModel's vocabulary ("holds its model row", "part of
  this model", "from its own value rows", "its value rows have serials"), which 09's guard must
  allow. `test_declared_facts.TestNoProjectNameInThePool` already allows it, and says why.
- `store_inventory.py`'s `Parent` and `ParentSet` docstrings name project classes as examples
  (`:239`, `:265`). They are in the audit's comment list and outside `read_records`, so 09's
  guard will meet them (the log).
- The pool's owned-serials comparison is pinned only on a stand-in pool
  (`test_the_pool_checks_each_answer_through_the_hook`). No test reaches it through a real
  replicated write.

**Orchestrator review of prompt 08 (2026-10-06).** Dispatched from `6ad58bb` to one Sonnet
subagent. Reviewed against its commit `6646b06`, with nothing landed after it. **Every check of the
note's §3 passed**, and acceptance 1–6 are met. No stop condition fired, and the agent ran its
suites in the foreground.

*At dispatch.* The note's gate held at `6ad58bb`:
- the tree was clean, and the only worktree was this checkout;
- `5fe0ae6..6ad58bb` touches only `prompts/`;
- the registry held 16 runs (14 finished, 2 `unknown`, none `running`), and no Ray process was up;
- the venv imported `scipy.sparse.linalg`.

The baselines, serially at `6ad58bb`, at load 7–24 (a transient spike from system daemons), match
07's review:
- AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped);
- RunRegistry 280, OK; ComputeTargets 693, OK;
- Datastore 692 (150 + 151 + 200 + 191), OK. No committed measurement file tripped a grep test.

The index held 107. The pre-images of the 29 files of the note's §1.6 were recorded, and `6ad58bb`
was exported to the scratchpad. The parent side of check 8 was measured there before dispatch:
- the fingerprints, digests `6576b85d…` and `d9328538…`, the whole inventories, and `resolve()` of
  all 55 + 18 records (11 entries `"unresolved"`);
- `main.py --inventory` on a copy of one `build_full_store` primary: 14837 and 14949 bytes (plain,
  verbose), exit 0, checksums unchanged;
- the four "has not been built" messages and the three reduced registries' refusals;
- each factory's `read_records` keywords, recorded by patching `read_records` and calling
  `inventory_records(None, "T", {}, None)`, normalised so that a `Parent` compares across trees.

The checks:
1. **Scope.** 35 files, all on the prompt's §6 list. The diff over `docs/` (but `OPEN_ISSUES.md`),
   `ShardedPool.py`, `Datastore.py`, `schema.py`, `ClientPool.py`, `store_reader.py`,
   `replication.py`, `shard_paths.py`, `config/`, `main.py`, `extract_*.py`, `RunRegistry/` (but
   `test_store_fingerprint.py`), `var/` and `orchestrator/` is empty.
2. **E1, by grep and by running.**
   - The first §3.2 grep, outside `orchestrator/`, prints the two test literals and their uses and
     `test_store_fingerprint.py`'s literal and its use, nothing more. The second prints nothing.
   - `main.py:1207`'s comment is not printed: this `git grep -E` does not honour `\b`. The log
     says so. The orchestrator's own `measure-08/scripts/` name the removed names, as measurements
     of the old tree.
   - `m6_names_left.py` on `store_inventory.py`: 23 hits, from 105. All are the record's
     `validated` (17), SQLAlchemy's `.label()` and the word "label" in the docstring (2), and the
     local `tag_serial` (4), as the log classifies them.
   - `m5_imports.py`: importing `store_inventory` loads `Datastore.contract` alone, as before.
     `read_inventory` now also loads `ObjectFactories.cosmology_types`, the client's module,
     through the two factories' hooks. `m5b_read_records_alone.py`: `read_records` alone loaded
     `CosmologyModels`, `.base` and `.model_ids` at `6ad58bb`, and loads nothing at `6646b06`.
3. **E2, by `ast` against the pre-images, docstrings stripped.**
   - **The 21 factories:** in each, `inventory_records` is gone, `inventory_spec` is new, and
     every other function and module-level statement dumps identically. `base.py` gains only
     `inventory_spec`.
   - **`store_inventory.py`:**
     - gone: `cosmology_classes`, and the module-level `COMPUTE_TARGETS`, `INVENTORY_CLASSES` and
       `COSMOLOGY`;
     - new: `InventorySpec` (`keywords`, `dependencies`), `inventory_specs`, `inventory_classes`,
       `Parent.__post_init__`, `classes` and `class_of`, `ParentSet.__post_init__` and
       `ClassInventory.parent_class`;
     - changed: the fields of `Parent`, `ShardRead` and `ClassInventory`, `StoreInventory.resolve`,
       `read_records`, `_combine` and `read_inventory`;
     - every other function dumps identically.
   - **`tools/inventory_report.py`:** only `_Renderer.__init__`, `_Renderer.parent_class`, the
     import and the two comments.
   - **In-process, on the real registry:**
     - the 21 specs' `keywords()` equal, normalised, the keywords the pre-images passed to
       `read_records` (`cmp`);
     - `inventory_classes(factories)` equals today's 21 names in order, and the classes are
       exactly the 21 of the 38 with a spec;
     - no factory has `inventory_records`, and `COSMOLOGY_TYPES` `is` both factories' map;
     - each refusal of the prompt's §2 raises `ValueError` with the names: `Parent`'s five
       malformed forms, a polymorphic `ParentSet` member, a parent or tag table with no spec, a
       class absent from the registry, a type column that is not a leaf, a self-reference and a
       cycle. `Parent` stays hashable.
4. **E4, the tests, by `ast`.** In the six edited files no test method is gone or new. Only
   `test_only_canonical_formats_a_float` and `test_the_inventory_key_has_no_z_init` change, and
   the module-level lines are the two `INVENTORY_CLASSES` literals and the `COMPUTE_TARGETS`
   literal, with their imports. The comment and docstring edits are text only. The 25 new methods
   are all in `test_inventory_declarations.py`, and the log names each.
5. **E2, by reading and running.** `test_inventory_declarations.py` was read in full.
   - The order test pins a literal. The refusal-order test builds a real `build_full_store`
     primary and asserts the reader's `RuntimeError`, naming `replicated_tables` and
     `BackgroundModel`.
   - Each refusal test asserts the exception and the names, but one:
     `test_a_malformed_parent_is_refused` asserted only `ValueError`. It was tightened at this
     review (below).
6. **E3, by running.** The new module, the eight inventory modules the note names, RunRegistry's
   `test_store_fingerprint` and `test_store_retire`, and `test_qcd_cosmology_inventory_record`:
   327 tests, OK.
7. **E5, the pins.** The seven diffs were extracted from the log and replayed with `git apply` in
   `bash`, `--check` and `-R --check` first. Each failed exactly the tests the log gives:
   - (i): failures=4, `test_every_class_has_a_record` among them;
   - (ii): failures=1; (iii): failures=3; (iv): failures=6, all in `TestResolve`;
   - (v): errors=1, an `IndexError`, in 1.9 s, so the run ends; (vi): errors=1; (vii): failures=1.

   The tree was clean after each `git apply -R`. On (iv) the log is right: `test_inventory_report`
   and `test_inventory_report_labels` pass while the report prints `<unresolved reference>` for
   every cosmology field. A pin was added at this review (below).
8. **The probe**, from the scratchpad, at `6646b06` and at the export of `6ad58bb`:
   - **the fingerprints**, with `taken` removed, are byte-identical (`cmp`): `build_full_store`'s
     digest `6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759` (7507 bytes) and
     `build_real_store`'s `d932853819d07ce40fbb9cce544dabe4bbe968e4a7c908bb33e80eecbca3264f` (4484
     bytes);
   - **the whole inventories**, with `parents` normalised for the polymorphic field, are
     byte-identical (29950 and 12297 bytes). The raw `parents["cosmology"]` is `null` where it was
     `"cosmology"`, as intended;
   - **`resolve()`** of all 55 + 18 records is byte-identical (209794 and 7442 bytes);
   - **`main.py --inventory`** on the kept primary, plain and verbose: stdout (14837, 14949 bytes),
     stderr (empty) and exit (0) byte-identical, and the store's checksums unchanged. The log's
     sizes (14844, 14956) are of its own primary, whose path is seven characters longer;
   - **the refusals**: the four "has not been built" messages and the three reduced registries'
     exceptions are byte-identical. A registry lacking `BackgroundModel` or `store_tag` still meets
     the reader's `RuntimeError` first.
9. **E6.**
   - The generic-layer issue has a dated "Narrowed" line naming rows 12 and 13, the hook,
     `InventorySpec`, `inventory_classes` and `Parent`'s type map. It says every C1 row is gone and
     09's guard remains, and its index hook says so.
   - `[08-resolve-shows-an-absent-nullable-parent-as-unresolved]` is on §3.1, with its measurement,
     and in the index.
   - The index holds 108 by counting, matching its header.
   - README §4 names `InventorySpec`, `inventory_spec`, `inventory_classes`, `Parent`'s type map
     and `COSMOLOGY_TYPES`. §1, I1–I2 and §5.9 are written.
10. **Suites**, serially at `6646b06`, load 7–18:
    - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped);
    - RunRegistry 280, OK; ComputeTargets 693, OK;
    - Datastore 717 (692 + 25; 150 + 176 + 200 + 191), OK.

    No flake appeared. `black --check` is clean on the 31 Python files.

*Residue, fixed in the follow-up commit that records this review.*
- "this commit" → `6646b06` in §1, §3, §5.9, README §4 and the log.
- README §4's line "with one addition to `Parent`" named an addition the bullet under it does not
  describe. It now says the last bullet gives the refusals and where the order is derived.
- **The report's rendering of the polymorphic reference is pinned.** New:
  `TestResolve.test_the_report_renders_both_cosmology_types`, which renders `build_full_store`'s
  report and checks that every cosmology field is the key its `cosmology_type` names: a
  `T_z_representation` for a `QCD_Cosmology`, none for a `LambdaCDM`, and no
  `<unresolved reference>`. Under breakage (iv) the module now fails 7 tests where it failed 6.
- `test_a_malformed_parent_is_refused` asserts that each message names the column
  (`^Parent\('c'\): `).
- The log notes both additions, marked as made at review. §5.9 records the new count.
- This board's note list marks `orchestrator/prompt-08.md` used.

*Kept as is.*
- An absent dependency is named once, by the first class in registry order that references it
  (`BackgroundModel references redshift`). The class it lacks is named, which is what a reader of
  the refusal needs.
- `read_inventory` loading `cosmology_types` through the hooks. It is the client's module, loaded
  by the client's factories; the layer imports nothing of the project.
- The orchestrator's `measure-08/scripts/` keep the removed names. They measure the old tree.

*Handed on.* **09**'s note must carry:
- Every C1 row is gone. What remains for the guard is the comments of the audit's list outside
  `ShardedPool.py` and `store_inventory.py`: `Datastore.py`, `ClientPool.py:37-38`,
  `replication.py:200, 204` and `shard_paths.py:14`, at the audit's lines. 09 must re-measure them.
- The vocabulary the guard must allow:
  - `ShardedPool`'s "model row", "value rows" and "part of this model" (07);
  - the record's `validated`, and the generic word "label";
  - the contract's names through `Datastore.contract`.
- The two existing name tests, `test_declared_facts.TestNoProjectNameInThePool` and
  `test_inventory_declarations.TestTheLayerKnowsNoProject.test_no_project_name_in_its_code_or_text`,
  which 09's guard generalises.
- `tools/inventory_report.py` is a client tool that names project tables
  (`INVENTORY_CATEGORIES`, `VALUE_TABLES`), and is outside the guard.
- A guard written as `git grep -E` must not rely on `\b`, which git 2.56 does not match there
  (`-P` does). A token scan in Python, as both existing name tests use, does not have the
  problem. A repository-wide scan must exclude `prompts/`, whose measurement scripts name the
  removed names.
- 08's log: a factory joins the inventory by declaring `inventory_spec()`, and a client with a
  polymorphic reference declares its own type map, as `cosmology_types.py` does.

**Orchestrator review of prompt 09 (2026-10-06).** Dispatched from `c96e56d` to one Sonnet
subagent. Reviewed against its commit `e980b6e`, with nothing landed after it. **Every check of the
note's §3 passed but E7**, and acceptance 1–3 are met. No stop condition fired, and the agent ran
its suites in the foreground. E7, the rehearsal rebuild of README §3's exception, runs after this
record (note §4.2), so K2 stays at its fixture half and the campaign is not yet closed.

*At dispatch.* The note's gate held at `c96e56d`:
- the tree was clean, and the only worktree was this checkout;
- `3ff1c80..c96e56d` touches only `prompts/`;
- the registry held 16 runs (14 finished, 2 `unknown`, none `running`), and no Ray process was up;
- the venv imported `scipy.sparse.linalg`.

The baselines, serially at `c96e56d`, at load 6–23, match 08's review:
- AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped);
- RunRegistry 280, OK; ComputeTargets 693, OK;
- Datastore 718 (150 + 177 + 200 + 191), OK. No committed measurement file tripped a grep test.

The index held 108. The pre-images of the note's §1.6 five files were recorded, and `c96e56d` and
`dbd8a51` were exported to the scratchpad. The `dbd8a51` side of check 8 was measured before
dispatch:
- both fixture stores built with `dbd8a51`'s code: 6 files, 164 tables, 286 rows, kept with
  their checksums;
- their fingerprints read at `dbd8a51`, digests `6576b85d…` and `d9328538…`, and their whole
  inventories; the reads left the checksums unchanged;
- a read-write open of a copy of the full store through `dbd8a51`'s stand-in pool, under the
  store's own label (`2025.1.1`) and a new one (`standin`). Neither raised, and each printed 15
  lines. The own label changed no file; the new one wrote version row 3.

The checks:
1. **Scope.** 10 files, all on the prompt's §6 list. The diff over `docs/` (but `OPEN_ISSUES.md`),
   `config/`, `utilities.py`, `main.py`, `extract_*.py`, `RunRegistry/`, `tools/`, `var/`,
   `orchestrator/`, every factory, `RayTools/`, and every module of the layer but
   `store_inventory.py`, `replication.py` and `shard_paths.py`, is empty.
   - `OPEN_ISSUES.md` also gains one sentence in §1.18's introduction, saying that 09 closed the
     issue, as 06's closing did there. It is in an allowed file.
2. **E1, by running.**
   - The guard passes at `e980b6e`, and `layer_files()` gives the 17 files.
   - Copied into the export of `c96e56d`, it fails `test_no_code_name` and
     `test_no_string_or_docstring` with exactly the 10 hits, and nothing else.
   - The allow-list is the prompt's four groups, each with its reason. No other entry exists.
   - **The vocabulary.** Computed from the two pre-images, the deleted tests' union is 238 words
     (07's 232, 08's 48); the guard's 234 words all lie inside it. The four words of the union not
     in the guard are the prompt's own exclusions:
     - `RayTools` and `config`, the packages §2 removes (the layer, and the import check);
     - `store_tag` and `version`, which 08's class-name list held, and which `LAYER_WORDS` removes
       as the contract's two tables.

     The 07 pre-image's `REPO_ROOT` resolves against its own file, so its package set was
     recomputed against the repository root.
3. **E3, by `ast` against the pre-images, docstrings stripped.**
   - `store_inventory.py`: only `read_records` changes. With the `parent` names on the eight
     lines renamed back, the whole module dumps identically, docstrings included.
     - A rename-back of every `parent` does not dump identically. `read_records` already used
       `parent` in its `for field, parent in declared` loop (`:555`); the nested `_reference`
       takes it as a parameter (`:691`), and the loop at `:712` rebinds it.
     - Nothing reads the `:555` binding after `:607`, so the rename shadows nothing, as the log
       says.
   - `replication.py` (29 units) and `shard_paths.py` (7): every function and module statement
     dumps identically.
4. **E2, the tests, by `ast`.**
   - In `test_declared_facts.py`, the only things gone are:
     - `TestNoProjectNameInThePool` (its `setUpClass` and three methods);
     - `_named` and `_registry_names`;
     - the module statements `REPO_ROOT`, `POOL_FILE`, `_WORD`, `_LAYER_WORDS`,
       `_PROJECT_PACKAGES` and `_FLAG_PHRASES`;
     - the imports `ast`, `re`, `sys` and `tokenize`.

     Every other unit dumps identically.
   - In `test_inventory_declarations.py`, the only things gone are the two methods,
     `STORE_INVENTORY`, and the imports `ast`, `io` and `tokenize`. Every other unit, among them
     `PROJECT_PACKAGES`, `_run` and the two load tests, dumps identically.
   - No module-level name is left unused in either module or in the guard.
   - All eight new methods are in `test_layer_is_generic.py`, and the log names each.
5. **E1, by reading.** `test_layer_is_generic.py` was read in full.
   - Each failure lists `file:line word`, for every file at once.
   - `test_the_allow_list` asserts five acceptances and five refusals.
   - Empty input is pinned at module level. The derived files must hold audit §C's list as a
     literal, and the vocabulary must hold known tables, columns and packages and no `LAYER_WORDS`.
   - The module, `test_declared_facts` and `test_inventory_declarations`: 57 tests, OK.
6. **The deleted tests' coverage.** One mutation per deleted test, at `e980b6e`, each against the
   guard:
   - `BackgroundModel` in a `ShardedPool.py` `RuntimeError` message (07's string-and-name method):
     `test_no_string_or_docstring`, `ShardedPool.py:213`;
   - "source or response flag" in a `ShardedPool.py` comment (07's comment method):
     `test_no_comment`, `ShardedPool.py:65`;
   - `from MetadataConcepts import store_tag` in `ShardedPool.py` (07's import method):
     `test_every_import_is_allowed`;
   - `GkSource` in a `store_inventory.py` comment (08's (vii)): `test_no_comment`,
     `store_inventory.py:236`;
   - `import CosmologyModels` in `store_inventory.py` (08's import method):
     `test_every_import_is_allowed` and `test_no_code_name`.
7. **E5, the pins.** The nine diffs were extracted from the log and replayed with `git apply` in
   `bash`, with `--check` and `-R --check` first. Each failed exactly the tests the log gives:
   - (i), (vi): `test_no_comment`; (ii): `test_no_code_name`;
   - (iii): `test_every_import_is_allowed` and `test_no_code_name`; (iv):
     `test_every_import_is_allowed`;
   - (v), (vii): `test_no_string_or_docstring`; (viii): `test_the_derived_files_hold_the_audits_list`;
   - (ix): `test_the_allow_list` alone, with two failing subtests.

   The tree was clean after each `git apply -R`, with no `Datastore/scratch_module.py` left.
8. **E4, the probe**, from the scratchpad, at `e980b6e` against the `dbd8a51` side taken before
   dispatch:
   - **(a)** both fixture stores built at `e980b6e`: 6 files, 164 identical tables, 286
     identical rows, `sqlite_master` the same, 0 differences;
   - **(b)** the `dbd8a51`-built stores read at `e980b6e`:
     - the fingerprints, with `taken` removed, are `cmp`-identical to those read at `dbd8a51`:
       `6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759` (7507 bytes) and
       `d932853819d07ce40fbb9cce544dabe4bbe968e4a7c908bb33e80eecbca3264f` (4484 bytes);
     - the whole inventories are `cmp`-identical (29954 and 12301 bytes);
     - the store checksums are unchanged;
   - **(c)** a read-write open of fresh copies through `e980b6e`'s stand-in pool, under both labels:
     - nothing raised;
     - the reconciliation, the prune (empty), the version object and the 15 printed lines are
       `SAME`;
     - under the store's own label no file changed;
     - under the new label the files written are byte-identical to `dbd8a51`'s;
     - only `kwargs` differs, as 06 made it.
9. **E6.**
   - The generic-layer issue is in §4, with a dated "Closed" line naming the guard, and gone from
     §3 and the index.
   - The index holds 107 by counting, matching its header.
   - README §4 names the guard module, `layer_files()`, `import_allowed` and the allow-list.
   - This board has the §1 row, K1 ✅, K2 "✅ fixture half; the rebuild at the review", and §5.10.
10. **Suites**, serially at `e980b6e`, load 6–9:
    - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped);
    - RunRegistry 280, OK; ComputeTargets 693, OK;
    - Datastore 721 (147 + 183 + 200 + 191), OK.

    No flake appeared. `black --check` is clean on the six Python files.

*Residue, fixed in the follow-up commit that records this review.*
- "this commit" → `e980b6e` in §1, §4, §5.10, README §4 and the log.
- The log did not record that mutation (vii) names `BackgroundModel` in a module constant, where
  the prompt says the docstring. It is now an IMPLEMENTATION CHOICE in the log's deviations,
  marked as added at review. The same test scans both.
- This board's note list marks `orchestrator/prompt-09.md` used.

*Kept as is.*
- **The import check reads a `from` import's module, not the names it imports.** So
  `from Datastore.SQL import ObjectFactories` and `from Datastore import tests` pass it.
  - The deleted 07 test had the same rule, so no coverage was lost.
  - `ObjectFactories/__init__.py` is empty, so importing the package loads no factory.
  - Any use of a factory module through it names a registered class, which `test_no_code_name`
    catches.
  - The check also refuses more than it must: `from .ObjectFactories import base` is refused,
    though `import Datastore.SQL.ObjectFactories.base` is allowed. Nothing in the layer uses the
    first form.
- `store_tag` and `version` are allowed anywhere in the layer, where 08's test forbade both in
  `store_inventory.py`. That is §2's "less `LAYER_WORDS`", which defines the vocabulary once.

*Handed on.* Nothing: 09 is the campaign's last prompt. E7 follows (note §4.2).

**The rehearsal rebuild, E7 (2026-10-06): the fingerprint matches, and the campaign is closed.**
This is README §3's one exception, run under `CLAUDE.md`'s run-registry rules with nothing relaxed.

*Before the launch.*
- `RunRegistry list` showed nothing `running` and nothing stale (16 runs).
- `store show var/datastores/datastore-generic-09-rehearsal.sqlite` gave kind `absent`, with no runs.
- No Ray process was up.
- The tree was clean at `53d4e18`, the review's follow-up, with no code changed since `e980b6e`.

*The launch.* It used the reference's command, with a new store, job name, slug, purpose and
campaign, detached under `nohup` with `disown`. The launch log is
`var/runs/datastore-generic-09-rehearsal.launch.log`, beside the run directory, as
a3-v2-readiness-07's was. One check showed it `running` with a fresh heartbeat, and the turn
ended there.

*The run,* `datastore-generic-09-rehearsal-20261006T152452`:
- Created 15:24:52 at `53d4e18`, `git_dirty` false; last heartbeat 15:29:02. That is about
  4 minutes, where the reference took 13 at `dbd8a51`.
- `done`, exit code 0, and the launch log ends `** driver exit status 0`.
- **The digest is `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, equal to
  a3-v2-readiness-07-reference's.**
- All 21 classes' per-class entries (count and digest) in `status.json` equal the reference's.
- `fingerprint_sidecar` reads "not written: … (there is none)", as the reference's did.
- `store create --purpose …` then wrote the closed primary's registry sidecar, problems none.
  `store fingerprint` on the closed store gives the same digest.
- After the driver exited, `pgrep -lf 'gcs_server|raylet|ray::'` showed no Ray process.

So rule 9 holds end to end. A store built through the write path at the campaign's final tree is
the store `dbd8a51` built, class by class. K2 is ✅.

*Left open on this board,* each a decision for the user or a cleanup the user will have done:
- `[04-closed-store-refusals-repeat-their-prefix]`;
- `[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]`;
- `[08-resolve-shows-an-absent-nullable-parent-as-unresolved]`.

They stay in §3 and in the index.

*The store* `var/datastores/datastore-generic-09-rehearsal.sqlite` is throwaway, and the user
retires it with `store retire`.
- *Retired by the user, 2026-10-06 (completed 20:21:22), at `f58c785`.* The first attempt was
  refused, and nothing was written or deleted. A sidecar made by `store create` records no
  fingerprint, and the note's §4.2 named only `store create`.
- The orchestrator then ran `store fingerprint --write`, which recorded
  `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`.
- The retirement's fresh read-only fingerprint matched it, and the primary and its four shards
  were deleted. The sidecar is the tombstone.
- A future note that has a store created for retirement should name both steps.

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
| **S1** | 05 | the reader and the inventory refuse absent or extra tables and columns | B5 | ✅ |
| **S2** | 05 | the read-only pool likewise; the reconcile's column-subset, `unkeyed` and `columns` logic deleted | B5 | ✅ |
| **S3** | 05 | an absent table on a read-write open, per U1, and every other difference refused there; the primary refused by both constructors if it differs from `_create_engine`'s tables (03's hand-on; confirmed by the user, 2026-10-05); `real_store_fixtures.py`'s old-store machinery; the four WKB value reads' `numeric_serial` selects (found by the 02 note) | B5 | ✅ |
| **G1** | 06 | `ShardedPool` and the actor given the registry and the serial batch sizes; no import of `Datastore._factories` | C1 #9, #10 | ✅ |
| **G2** | 06 | `store_reader` and `store_inventory` given the registry; the replicated set read from the store's primary (the user, 2026-10-05) | C1 #10, #14 | ✅ |
| **G3** | 06 | the older registries moved to `config/datastore.py`, per U2; the layer given the tables to drop, in an order derived from foreign keys (`_drop_order` deleted) | C1 #9–#11 | ✅ |
| **F1** | 07 | the copy-as-a-unit rule derived from foreign keys, through the owner an owned table declares (the user, 2026-10-06): `owner_column` on `BackgroundModelValue`; `ShardedPool._unit_tables` replaces the hard-coded unit and `_prune_unit` | C1 #2 | ✅ |
| **F2** | 07 | monotone flags declared by the factories: `monotone_flags` on `redshift` and `wavenumber` | C1 #5 | ✅ |
| **F3** | 07 | the `validated` rule declared by the factory: `validated_column` and the `revalidate` hook on `BackgroundModel`; `_StoredModel` gone; `read_records(validated=<column>)` | C1 #6 | ✅ |
| **F4** | 07 | the version and tag tables per U4 (`Datastore/contract.py`; the version object built by the version factory; `MetadataConcepts` no longer imported); the `owned_serials` hook (`_value_serials` gone); project names out of `ShardedPool`'s code, comments and messages | C1 #1, #4, #7, #8 | ✅ |
| **I1** | 08 | the inventory's class list and order derived, from the parents each factory declares in an `inventory_spec`, roots first (the user, 2026-10-06; foreign keys miss nine of them) | C1 #12 | ✅ |
| **I2** | 08 | the polymorphic cosmology reference declared by the client, in the referencing `Parent`'s type map (the user, 2026-10-06) | C1 #13 | ✅ |
| **K1** | 09 | a test that the generic layer names no registry table and imports no client package: `test_layer_is_generic`, over every file of the layer, derived; `config.defaults` and `utilities` allow-listed, and 07's and 08's per-file tests folded in (the user, 2026-10-06) | §C | ✅ |
| **K2** | 09 | every suite; README §5 rule 9's closing verification: fixtures in the prompt, and one rehearsal rebuild at the review compared with `dbd8a51`'s `39809dca…` (the user, 2026-10-06; README §3's exception) | — | ✅ (fixtures at `e980b6e`; the rebuild at the review, 2026-10-06) |

---

## 3. Active and unresolved issues

### 3.1 Opened by this campaign's audit (2026-10-04)

None open.

---

## 4. Resolved issues

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
  - **Decided (2026-10-06):** the user chose to remove the prefix in `_read_closed_store`, so that
    each caller states its own once.
  - **Assigned (2026-10-06):** `prompts/datastore-generic-followup` prompt 03, which carries out
    the decision. Its measurement found a fourth caller, `store_reader.open_read_only`, which
    relies on the inner prefix and so must wrap the message itself (that campaign's README §0.2).
  - **Closed (2026-10-07, `prompts/datastore-generic-followup` prompt 03; `086c81a`).**
    `ShardedPool._read_closed_store(primary, *, missing_ok=False)` raises the bare reason for each
    of its three refusals (the `shards` table could not be read; the resolver's message; the joined
    file problems) and takes no `verb`. Its three callers in `ShardedPool.py` (`_plan_relocation`,
    `_read_back`, `_plan_deletion`) keep their wrapping and stop passing a verb, so
    `closed_store_files`, `delete_store`, `copy_store` and `move_store` state their prefix once.
    The fourth caller, `store_reader.open_read_only`, which relied on the inner prefix, now wraps
    the reason with its own `_refuse`, so its message ends ". Nothing was read or repaired", which
    it lacked. Which stores are refused, the reason text and the phrase "its shards table could not
    be read" are unchanged. For a dropped `shards` table, `delete_store`, `copy_store` and
    `move_store` no longer name the resolved path anywhere (the removed prefix was the one that
    did); the reason names no file. Pinned by `test_closed_store_refusals` (5 tests) and the one
    changed `test_store_reader` test; deliberate breakage (i)-(v) in that prompt's log.

- **[08-resolve-shows-an-absent-nullable-parent-as-unresolved]** *(opened 2026-10-06 by prompt
  08; measured at `5fe0ae6` and at 08's tree)*
  - **The defect.** `StoreInventory.resolve` renders a nullable key parent whose value is `None` as
    `{"class": …, "digest": None, "unresolved": 0}`. That is a parent it failed to resolve, where
    the row has no such parent. On `build_full_store`, 11 entries are of this form, all the
    `numeric` field of a pure-WKB `GkWKBIntegration` reached through `GkSource`'s parent set. By
    contrast, `_resolve_reference`, for a parent-set member, returns `None`.
  - **Its age.** It is the same at `5fe0ae6`; 08 keeps it, so that `resolve()` stays
    byte-identical.
  - **Impact.** `resolve()` is called only by tests, for display and checking. The report renders
    through its own path.
  - **Next step.** A decision for the user: render the absent parent as `None`, as members are.
    Not assigned.
  - **Decided (2026-10-06):** the user chose to render it as `None`.
  - **Assigned (2026-10-06):** `prompts/datastore-generic-followup` prompt 02, which carries out
    the decision. Measured for it, the 11 entries are 2 direct `GkWKBIntegration` records and 9
    reached through `GkSource`, `GkSourcePolicyData` and `QuadSourceIntegral`, 3 each. The hook
    above says all 11 come through `GkSource`'s parent set; prompt 02's closing line corrects it.
  - **Closed (2026-10-07, `prompts/datastore-generic-followup` prompt 02; `4c34f13`).**
    `StoreInventory.resolve` gives `None` for a key parent whose value is `None`: one branch, after
    the `field not in cls.parents` branch, before the parent is looked up. A value that is not
    `None` and does not resolve to exactly one record keeps `{"class", "digest", "unresolved"}`,
    and `_resolve_reference`, `read_records`, the fingerprint, the inventory and the report are
    unchanged. The docstring says that only a nullable parent can hold `None` (`read_records` drops
    a row whose parent does not resolve). Over every record of `build_full_store` and
    `build_real_store`, `resolve()` differs from its parent's in exactly **11 entries**, each the
    `numeric` field of a `GkWKBIntegration` key going from
    `{"class": "GkNumericIntegration", "digest": None, "unresolved": 0}` to `None`: 2 in direct
    `GkWKBIntegration` records, and 3 each reached through `GkSource`, `GkSourcePolicyData` and
    `QuadSourceIntegral` (the 9 are the key of the `wkb` member in a `GkSource` record's parent set,
    which `GkSourcePolicyData` and `QuadSourceIntegral` reach through their own parents). **This
    corrects the hook above**, which said all 11 came through `GkSource`'s parent set: 2 are
    direct `GkWKBIntegration` records. `GkSource`'s parent set also has a member field named
    `numeric`; its 9 absent members were already `None` (through `_resolve_reference`) and are
    unchanged. Before, `build_full_store` held 11 `"unresolved"` mappings, all with a `None`
    digest; after, none. Pinned by two tests in `test_inventory_declarations.TestResolve`
    (`test_an_absent_nullable_parent_resolves_to_none` and
    `test_the_full_store_resolves_no_absent_parent_as_unresolved`); deliberate breakage (i)–(iii)
    in that prompt's log.
- **[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]** *(opened 2026-10-05 while
  prompt 06 was written; measured at `dbc59f9`)*
  - **The defect.** `--drop` drops exactly the tables of the groups named, and nothing that
    references them. Six groups have tables that other groups' rows reference by foreign key:
    `gk-source-policy`, `gk-source-policy-records`, `gk-source`, `gk-wkb`, `gk-numeric` and
    `tk-numeric`. Production never turns enforcement on, so SQLite accepts the drop and leaves
    those rows naming rows that are gone.
  - **Measured** on copies of both shards of `build_full_store`, clean under `PRAGMA
    foreign_key_check` before. Each group was dropped alone with enforcement off. The six left 1–4
    dangling references per shard, and the other four groups none:
    - `tk-numeric`: `TkWKBIntegration` and `QuadSource` → `TkNumericIntegration`;
    - `gk-numeric` and `gk-wkb`: `GkSource_parents` → the dropped table;
    - `gk-source`: `GkSourcePolicyData` → `GkSource`;
    - `gk-source-policy`: `QuadSourceIntegral` → `GkSourcePolicyData`;
    - `gk-source-policy-records`: `GkSourcePolicyData` and `QuadSourceIntegral` →
      `GkSourcePolicy`.

    Under enforcement the same six drops fail in any order (prompt 06's measurement, M3).
  - **Impact [inferred, not measured].**
    - A re-created table starts empty, and the broker leases serials from each table's largest
      stored serial. A recomputed row can therefore take a serial that a stale row still names,
      and the stale row then names a different, unrelated row.
    - Whether any lookup reaches such a row has not been traced. The datastore-integrity checks on
      parents (`_check_numeric_parent` and its kin) guard writes, not stale rows.
  - **Next step.** A decision for the user: a drop could cascade to every table whose foreign keys
    reach the dropped ones (derivable from the schema, as `schema.drop_order` will be), refuse a
    drop that would leave references, or stay as it is with the hazard documented. Prompt 06
    changes only the order within a drop (README §5 rule 9) and does not fix this.
  - **Decided (2026-10-06):** the user chose to refuse such a drop. Measuring for that campaign
    found that `PRAGMA foreign_key_check` undercounts. `QuadSourceIntegral` names
    `TkWKBIntegration` and `QuadSource` rows, and `GkSource` names `GkNumericIntegration` and
    `GkWKBIntegration` rows, through parents its factory's `inventory_spec` declares, with no
    foreign key. So `--drop tk-wkb` and `--drop quad-source` alone have the same hazard. The user
    then chose that a reference is a foreign key **or** a declared parent, followed transitively:
    8 of the 10 groups are refused alone.
  - **Assigned (2026-10-06):** `prompts/datastore-generic-followup` prompt 01, which carries out
    the decision.
  - **Closed (2026-10-07, `prompts/datastore-generic-followup` prompt 01; `fee1fbf`).** A
    read-write `ShardedPool` refuses `drop_tables` that would leave another table naming rows of
    a dropped one, before the primary is created or opened and before any actor exists, naming the
    tables given (sorted) and every table that must go with them (registry order). The rule is
    `Datastore.SQL.schema.dependent_tables(dropped, factories)`, read from the registry alone: a
    table names another by a foreign key **or** by a parent its factory's `inventory_spec`
    declares, followed transitively. Both halves are needed: as that campaign's README §0.2 found,
    foreign keys alone undercount (they miss `QuadSourceIntegral`'s cross-shard parents and
    `GkSource`'s parent set, so `--drop tk-wkb` and `--drop quad-source` alone would pass), and
    declared parents alone miss every value, tag and member table. 8 of the 10 groups are refused
    alone, each needing exactly §0.2's tables; every group together, and each group with its
    dependents, is accepted and drops what it dropped before, in the same order. The refusal is
    called immediately after `_refuse_undeclared_drop_tables()` (method
    `_refuse_drop_that_leaves_references`), and names no `--drop` group. Pinned by
    `Datastore/tests/test_drop_refuses_dangling_references.py` (11 tests); one test re-fixtured
    (`test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_store`, which now names
    `gk-source-policy-records`' dependents); deliberate breakage (i)–(vi) in that prompt's log.
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
  - **Closed (2026-10-05, prompt 05; `bc8bffc`).** Deleted (audit B5, and the selects found by
    02's note): the reader's four fields (`absent_tables`, `absent_columns`, `extra_tables`,
    `extra_columns`) and its reading of a shard that differs; the inventory's `absent-table` and
    `incomplete` problems with their branches, `ShardContext.absent_tables`/`absent_columns` and
    `ShardRead.comparable`; the read-only pool's `absent_tables` report and
    `_version_labels_present`'s branch for a shard without `version`; the read-only actor's
    absent-table report, `_skipped_validation`, its validation skip and `_inspector`; the
    reconcile's column subset, `unkeyed` and the `columns` difference kind, and `_read_full_rows`'
    column filter; the four WKB value reads' `numeric_serial` selects; `real_store_fixtures`'
    `build_old_store` and `OLD_MISSING_*`; nine tests (log). In their place one comparison and one
    refusal in `Datastore/SQL/schema.py`, `schema_differences` and
    `StoreSchemaMismatch(RuntimeError)`, used by the reader (so by the inventory, the fingerprint
    and the read-only pool), by the read-write check at open, which allows only an absent table
    (U1: an interrupted first open or drop, which the actors complete), and by both constructors'
    check of the primary against `_create_engine`'s six tables. A store written by older code now
    meets `StoreSchemaMismatch`, naming the file and each table and column. Pinned by
    `Datastore/tests/test_store_schema.py` (13 tests, the first coverage of U1) and three tests
    converted to refusals; deliberate breakage (i)–(v) in the log. The fingerprint of
    `build_full_store` is byte-identical before and after (digest `6576b85d…`).
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
  - **Closed (2026-10-05, outside the prompts, at the user's request; `b97cf06`).** Each shape
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

- **[00-the-generic-datastore-layer-hard-codes-project-schema-facts]** *(opened 2026-10-04 by the
  audit, §C; read)*
  - **The defect.** `ShardedPool`, `store_reader`, `store_inventory`, `Datastore` and `ClientPool`
    hard-code fourteen facts about the project's tables (audit C1): `BackgroundModel`'s copy unit and
    `validated` rule, the `redshift`/`wavenumber` flags, the version and tag tables, the factory
    registry imported as a private name, the inventory's class list, and more. `ShardedPool` had
    one project literal before `f4b2c91`.
  - **Impact.** The layer cannot be reused without editing it, and every new table that needs
    replication-time treatment is another special case in the pool. No wrong result.
  - **Next step.** Prompts 06–08, then 09's guard. After 08, 09's guard only.
  - **Assigned (2026-10-04):** prompts 06–09.
  - **Narrowed (2026-10-06, prompt 06; `19f07ee`):** audit C1 rows 9, 10, 11 and 14 are gone.
    The layer imports no registry: `ShardedPool`, the `Datastore` actor, `open_read_only` and
    `read_inventory` are given `factories`; the pool and the actor are given `serial_batch_sizes`
    (`ClientPool` holds no table of them) and the tables to drop (`drop_tables`), which the actor
    drops in the order `schema.drop_order` derives from foreign keys (`_drop_order` deleted). The
    three registries, the drop groups and `tables_to_drop` live in `config/datastore.py`, and
    `main.py` passes them in. The inventory compares exactly the classes the store's primary
    records in `replicated_tables` (`ReadOnlyStore.replicated_tables`), and imports neither
    `_factories` nor `config.sharding`. Importing the five layer modules now loads only the
    `ObjectFactories` package and `base`, `config` and `config.defaults`, `MetadataConcepts` and
    `utilities`, where it loaded 22 factory modules and 21 `ComputeTargets`, 3 `CosmologyConcepts`
    and 11 `CosmologyModels` modules (`Datastore/tests/test_layer_registry.py`,
    `TestTheLayerImportsNoClient`). Left for 07–09: rows 1, 2, 4–8, 12 and 13, and the guard.
  - **Narrowed (2026-10-06, prompt 07; `3f51939`):** audit C1 rows 1, 2, 4, 5, 6, 7 and 8 are
    gone. Three `register()` keys, carried by `build_schema` into every record of a class with a
    table and refused on a misfit: `owner_column` (declared by `BackgroundModelValue`),
    `monotone_flags` (`redshift`, `wavenumber`) and `validated_column` (`BackgroundModel`). Two
    factory hooks on `SQLAFactoryBase`, not abstract: `revalidate(serial, conn, table, tables)`
    and `owned_serials(obj)`, implemented by `BackgroundModel`'s factory. The pool's copy unit and
    prune unit are one function, `ShardedPool._unit_tables`, read from foreign keys through the
    declared owner and the tag table; `_prune_unit`, `_value_serials` and `_StoredModel` are
    deleted. The version and tag tables and their columns are named once, in
    `Datastore/contract.py` (U4); the pool builds the version object through the version table's
    factory and no longer imports `MetadataConcepts`; the actor, `schema.py` and
    `read_records`' tag and validated reads name no project table or column, only the contract
    and the layer's own prepended `version` column. `ShardedPool.py` names no
    registry table but the contract's, in code, comments or messages, pinned by
    `Datastore/tests/test_declared_facts.py`. Every repair, refusal, printed line and record of
    the check at open, and the fingerprint, are byte-identical to `f1b8d02`'s (the log). Left for
    08–09: rows 12 and 13, and the guard.
  - **Narrowed (2026-10-06, prompt 08; `6646b06`):** audit C1 rows 12 and 13 are gone, and with
    them **every row of C1**. 09's guard remains. The inventory's classes and their order are
    derived: each factory declares a static `inventory_spec()` (not abstract on
    `SQLAFactoryBase`, default `None`) returning a `store_inventory.InventorySpec`, which is
    `read_records`' six keywords and gives `keywords()` and `dependencies()`; `inventory_records`
    is gone from all 21 factories. `inventory_specs(factories)` and `inventory_classes(factories)`
    give the classes that declare a spec, roots first and then a dependency sort, each in registry
    order, refusing (`ValueError`, by name) a dependency with no spec, a polymorphic type column
    that is not a leaf, a self-reference and a cycle; `read_inventory` derives the order after the
    reader has opened the store, so a registry that does not fit its store is refused by the
    reader as before. `INVENTORY_CLASSES` and `COMPUTE_TARGETS` are deleted. The polymorphic
    cosmology reference is a `Parent` with a type map, `Parent(column, type_column=…,
    types={type value: class})`, declared by `wavenumber_exit_time`'s and `BackgroundModel`'s
    factories through the one client constant `ObjectFactories/cosmology_types.COSMOLOGY_TYPES`;
    `COSMOLOGY` and `cosmology_classes` are deleted, and `store_inventory.py` and
    `tools/inventory_report.py` import no `CosmologyModels`. `ClassInventory.parents[field]` is
    `None` for a polymorphic field, and `parent_types` and `parent_class(field, key)` carry and
    use the map. `store_inventory.py` names no registry table, project package or project column
    in code, comments or docstrings (pinned by `Datastore/tests/test_inventory_declarations.py`).
    Both fingerprints, both whole inventories, every `resolve()`, `main.py --inventory` and every
    refusal that exists today are byte-identical to `6ad58bb`'s (the log).
  - **Closed (2026-10-06, prompt 09; `e980b6e`).** Every C1 row is gone (06–08), and the guard
    keeps it so. `Datastore/tests/test_layer_is_generic.py` derives the layer's files from the
    filesystem (`layer_files()`: 17 today), checks each one's code names, strings and docstrings,
    and comments against the registry's tables, class names and identifier columns, the project's
    packages, 08's cosmology names and 07's flag phrases and 08's fragments (234 words, less
    `LAYER_WORDS`), and allows only the standard library, `ray`, `sqlalchemy`, the layer's own
    modules (of the factories only `base`), `config.defaults` and `utilities` (`import_allowed`;
    the user's choices, 2026-10-06). On today's layer it found ten hits at three sites, none of
    them project knowledge (a local `parent_serial`, the word "tolerance" in one docstring and
    the path `tools/` in another), and those three sites were fixed in `e980b6e`. Nine
    deliberate breakages each fail a named test (the log). Rule 9 on fixtures holds, byte for
    byte, against `dbd8a51` (the log); the rehearsal rebuild is the review's.

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

### 5.6 After prompt 05 (2026-10-05, `bc8bffc`)

Measured serially on the staged tree, at load 11–40. The Datastore suite was run in four groups of
modules (`discover -p 'test_[a-f]*.py'`, `[g-o]`, `[p-r]`, `[s-z]`); the four counts are 122, 131,
200 and 191 (before: 122, 132, 201, 185).

| Suite | Before (orchestrator, `6ba15c7`) | After | Difference |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 0 |
| ComputeTargets | 693 OK | 693 OK | 0 |
| CosmologyModels | 39 OK | 39 OK | 0 |
| Datastore | 640 OK | 644 OK | −9 deleted, +13 in `test_store_schema`; three tests converted (renamed) |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 0 |
| RunRegistry | 280 OK | 280 OK | 0 |

### 5.7 After prompt 06 (2026-10-06, `19f07ee`)

Measured serially on the staged tree, at load 7–27. The Datastore suite was run in four groups of
modules (`discover -p 'test_[a-f]*.py'`, `[g-o]`, `[p-r]`, `[s-z]`); the four counts are 122, 151,
200 and 191 (before: 122, 131, 200, 191).

| Suite | Before (orchestrator, `154d868`) | After | Difference |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 0 |
| ComputeTargets | 693 OK | 693 (1 failure, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, wall clock: it failed alone 3 times of 3 at load ~23 and passed alone a fourth time, and failed alone 2 times of 3 on an export of `154d868`) | 0 |
| CosmologyModels | 39 OK | 39 OK | 0 |
| Datastore | 644 OK | 664 OK | +20 in `test_layer_registry`; four `TestDropOrder` methods converted (three renamed) |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 0 |
| RunRegistry | 280 OK | 280 OK | 0 |

### 5.8 After prompt 07 (2026-10-06, `3f51939`)

Measured serially on the staged tree, at load 7–30. The Datastore suite was run in four groups of
modules (`discover -p 'test_[a-f]*.py'`, `[g-o]`, `[p-r]`, `[s-z]`); the four counts are 150, 151,
200 and 191 (before: 122, 151, 200, 191).

| Suite | Before (orchestrator, `f1b8d02`) | After | Difference |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 0 |
| ComputeTargets | 693 OK | 693 OK | 0 |
| CosmologyModels | 39 OK | 39 OK | 0 |
| Datastore | 664 OK | 692 OK | +28 in `test_declared_facts`; the two witness tests of `test_schema_builder` read the new witness |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 0 |
| RunRegistry | 280 OK | 280 OK | 0 |

### 5.9 After prompt 08 (2026-10-06, `6646b06`)

Measured serially on the staged tree. The Datastore suite was run in four groups of modules
(`discover -p 'test_[a-f]*.py'`, `[g-o]`, `[p-r]`, `[s-z]`); the four counts are 150, 176, 200 and
191 (before: 150, 151, 200, 191).

| Suite | Before (orchestrator, `6ad58bb`) | After | Difference |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 0 |
| ComputeTargets | 693 OK | 693 OK | 0 |
| CosmologyModels | 39 OK | 39 OK | 0 |
| Datastore | 692 OK | 717 OK | +25 in `test_inventory_declarations`; the two inventory test modules, `test_wkb_numeric_parent_key` and `test_store_fingerprint` edited at their imports and one method as the prompt says |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 0 |
| RunRegistry | 280 OK | 280 OK | 0 |

*Added at the orchestrator's review (2026-10-06).* The follow-up commit that records the review
adds one test to `test_inventory_declarations` (`test_the_report_renders_both_cosmology_types`),
and tightens one assertion. Datastore `[g-o]` then runs 177, OK, and the suite totals **718**.

### 5.10 After prompt 09 (2026-10-06, `e980b6e`)

Measured serially on the working tree before the commit, which holds only `e980b6e`'s change. The
Datastore suite was run in four groups of modules (`discover -p 'test_[a-f]*.py'`, `[g-o]`,
`[p-r]`, `[s-z]`); the four counts are 147, 183, 200 and 191 (before: 150, 177, 200, 191).

| Suite | Before (orchestrator, `c96e56d`) | After | Difference |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 0 |
| ComputeTargets | 693 OK | 693 OK | 0 |
| CosmologyModels | 39 OK | 39 OK | 0 |
| Datastore | 718 OK | 721 OK | −5 deleted (`test_declared_facts.TestNoProjectNameInThePool`, three; `test_inventory_declarations.TestTheLayerKnowsNoProject`, two), +8 in `test_layer_is_generic` |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | 0 |
| RunRegistry | 280 OK | 280 OK | 0 |

No flake appeared. K2's fixture half is in the log; the rehearsal rebuild is the review's.
