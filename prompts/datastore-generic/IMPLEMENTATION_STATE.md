# datastore-generic campaign — implementation state

**Last updated:** 2026-10-04 · **Status: IN PROGRESS — 1 of 9 prompts landed (01).** The audit ([`docs/datastore-generic-audit.md`](../../docs/datastore-generic-audit.md)) is
written. Prompts 01, 02 and 03 are written. 04 is not yet written, though its task is fixed; 05–09
are held on the prompts before them. The user took U1–U4 as recommended on 2026-10-04 (README
§6.2), and had the plan and `e53f323` merged into `handover-remedial` the same day.

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
| 02 | [Delete the remaining factory refusals](02-delete-the-remaining-factory-refusals.md) | **N1**–**N3** | Sonnet | ✍️ yes, 2026-10-04; after 01 | ⬜ | — | — |
| 03 | [Delete the reconcile's old-store paths](03-delete-the-reconciles-old-store-paths.md) | **R1**–**R4** | Opus | ✍️ yes, 2026-10-04 | ⬜ | — | — |
| 04 | Delete the legacy shard records | **L1**–**L3** | Sonnet | not yet; task fixed; after 03 | ⬜ | — | — |
| 05 | Refuse a store whose schema differs | **S1**–**S3** | Opus | ⏸ held on 03–04 (U1 taken) | ⬜ | — | — |
| 06 | Give the layer its registry | **G1**–**G3** | Opus | ⏸ held on 01–05 (U2 taken) | ⬜ | — | — |
| 07 | Declare the replication facts in the factories | **F1**–**F4** | Opus | ⏸ held on 06 (U4 taken) | ⬜ | — | — |
| 08 | Derive the inventory's class list | **I1**–**I2** | Sonnet | ⏸ held on 06 | ⬜ | — | — |
| 09 | Keep the layer generic | **K1**–**K2** | Sonnet | ⏸ held on 01–08 | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not landed · ✅ landed.

**Orchestrator notes** are written one prompt at a time, just before its dispatch, so that each
carries the previous review's findings: [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md)
(2026-10-04, used for 01); [`orchestrator/prompt-02.md`](orchestrator/prompt-02.md) (2026-10-04,
not yet used).

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

---

## 2. Items

| Item | Prompt | What | Audit | Status |
|---|---|---|---|---|
| **M1** | 01 | the old-store handlers in `TkNumericIntegration`, `GkNumericIntegration`, `QCD_Cosmology`, `BackgroundModel` (incl. the read-path re-query) | F0, B9 | ✅ |
| **M2** | 01 | `extract_common`: no "unknown generation" run; a store with no runs refused | B8 | ✅ |
| **M3** | 01 | the tests of M1–M2's deleted paths | F0, B8 | ✅ |
| **M4** | 01 | pass-through tests with the factories' real statement text | F0 | ✅ |
| **N1** | 02 | the old-store refusals in `TkWKBIntegration`, `GkWKBIntegration`, `GkSource`, `QuadSourceIntegral` | B7 | ⬜ |
| **N2** | 02 | their tests, with the two current-behaviour tests kept | B7 | ⬜ |
| **N3** | 02 | pass-through tests | B7 | ⬜ |
| **R1** | 03 | review `e53f323` (merged 2026-10-04): `BackgroundModelValue` under its serial | B2 | ⬜ |
| **R2** | 03 | measure one timestamp on every replicated write path; compare `timestamp` | B1 | ⬜ |
| **R3** | 03 | `replication_in_flight` assumed present | B3 | ⬜ |
| **R4** | 03 | the `controller_shard NOT NULL` refusal; `label` read unconditionally | B4, B9 | ⬜ |
| **L1** | 04 | absolute shard records out of `shard_paths.py` and `ShardedPool` | B6 | ⬜ |
| **L2** | 04 | `tools/shard_key_audit.py`'s old-primary paths; the sweep script's docstring | B6 | ⬜ |
| **L3** | 04 | `write_legacy_primary` renamed for the malformed-record tests; legacy tests deleted or re-fixtured | B6 | ⬜ |
| **S1** | 05 | the reader and the inventory refuse absent or extra tables and columns | B5 | ⬜ |
| **S2** | 05 | the read-only pool likewise; the reconcile's column-subset, `unkeyed` and `columns` logic deleted | B5 | ⬜ |
| **S3** | 05 | an absent table on a read-write open, per U1; `real_store_fixtures.py`'s old-store machinery | B5 | ⬜ |
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
  - **Narrowed (2026-10-04, prompt 01):** deleted, with their tests: the old-store refusals in
    `TkNumericIntegration` (`build`, `TkNumericValue.build`), `GkNumericIntegration` (`build`,
    `GkNumericValue.build`, `GkNumericValue.read_batch`), `QCD_Cosmology.build` and
    `BackgroundModel.build` (the Gauss-order refusal, the grid-identity refusal and the read-path
    re-query, the value-table refusal), and `extract_common`'s "no run labels" and "unknown
    generation" branches (audit F0, B8, B9's comment); the eleven tests that exercised them. Left
    for 02: `TkWKBIntegration`, `GkWKBIntegration`, `GkSource`, `QuadSourceIntegral` (B7); for
    03–05: B1–B6.
  - **Assigned (2026-10-04):** prompts 01–05.

---

## 4. Resolved issues

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
