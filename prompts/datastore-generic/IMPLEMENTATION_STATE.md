# datastore-generic campaign — implementation state

**Last updated:** 2026-10-04 · **Status: PLANNED — reviewed; 0 of 9 prompts landed.** The audit ([`docs/datastore-generic-audit.md`](../../docs/datastore-generic-audit.md)) is
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
| 01 | [Delete the old-store refusals that misfire](01-delete-the-old-store-refusals-that-misfire.md) | **M1**–**M4** | Sonnet | ✍️ yes, 2026-10-04 | ⬜ | — | — |
| 02 | [Delete the remaining factory refusals](02-delete-the-remaining-factory-refusals.md) | **N1**–**N3** | Sonnet | ✍️ yes, 2026-10-04; after 01 | ⬜ | — | — |
| 03 | [Delete the reconcile's old-store paths](03-delete-the-reconciles-old-store-paths.md) | **R1**–**R4** | Opus | ✍️ yes, 2026-10-04 | ⬜ | — | — |
| 04 | Delete the legacy shard records | **L1**–**L3** | Sonnet | not yet; task fixed; after 03 | ⬜ | — | — |
| 05 | Refuse a store whose schema differs | **S1**–**S3** | Opus | ⏸ held on 03–04 (U1 taken) | ⬜ | — | — |
| 06 | Give the layer its registry | **G1**–**G3** | Opus | ⏸ held on 01–05 (U2 taken) | ⬜ | — | — |
| 07 | Declare the replication facts in the factories | **F1**–**F4** | Opus | ⏸ held on 06 (U4 taken) | ⬜ | — | — |
| 08 | Derive the inventory's class list | **I1**–**I2** | Sonnet | ⏸ held on 06 | ⬜ | — | — |
| 09 | Keep the layer generic | **K1**–**K2** | Sonnet | ⏸ held on 01–08 | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not landed · ✅ landed.

---

## 2. Items

| Item | Prompt | What | Audit | Status |
|---|---|---|---|---|
| **M1** | 01 | the old-store handlers in `TkNumericIntegration`, `GkNumericIntegration`, `QCD_Cosmology`, `BackgroundModel` (incl. the read-path re-query) | F0, B9 | ⬜ |
| **M2** | 01 | `extract_common`: no "unknown generation" run; a store with no runs refused | B8 | ⬜ |
| **M3** | 01 | the tests of M1–M2's deleted paths | F0, B8 | ⬜ |
| **M4** | 01 | pass-through tests with the factories' real statement text | F0 | ⬜ |
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
  - **Assigned (2026-10-04):** prompt 01.
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
  - **Assigned (2026-10-04):** prompts 01–05.

---

## 4. Resolved issues

None yet.

---

## 5. Baselines

### 5.1 At the plan (2026-10-04, `dbd8a51`)

| Suite | Result |
|---|---|
| Datastore | 598 OK (428 s) |
| Datastore, on `e53f323` | 598 OK (769 s, under load) |

The other suites: see `a3-v2-readiness`'s board §5; re-measured before 01's dispatch.
