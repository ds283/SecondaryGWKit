# Datastore integrity campaign — implementation state

**Last updated:** 2026-09-26 · **Status: 0 of 10 prompts landed. 01–04 are written; 06–08 are
released, to be written.** The user decided D2–D6 on 2026-09-26 (README §6.2); D1 is open, since the
user asked what it means, and holds 09. 05 waits on 04's list, and 10 on 01–09. `store-retirement`
closed on 2026-09-26 at 6 / 6, which discharges both cross-campaign constraints (README §2), so 01
and 04 are ready.

The campaign was opened on 2026-09-25, when the user asked that every known datastore-integrity
defect be fixed before the A3 v2 store is built, the first store meant to have archival value. The
audit ([`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md)) re-verified
the eleven open issues in that class, all still present, and found each class wider than recorded:
- the replicated write can diverge, and a resume turns a missing row into a serial split the
  fingerprint cannot see; the three real stores are clean serial by serial (§4);
- the exit-time and `IntegrationSolver` lookups serve the wrong row, and identity lookups hide
  duplicates (§3.5–§3.9);
- every tagged `QuadSourceIntegral` `read_batch` fails, so extraction from A3 v2 would fail
  (§3.1); value reads omit parts of their parent's key (§3.3–§3.4);
- one foreign key names the wrong table and another is missing (§2.1–§2.2);
- derived rows are not keyed on what they were computed from (§2.3–§2.5).

**Campaign:** [`README.md`](README.md) ·
**Code:** `Datastore/SQL/ShardedPool.py`, `Datastore/SQL/Datastore.py`, `Datastore/SQL/ObjectFactories/`, `Datastore/store_inventory.py`, `Datastore/tests/`, the WKB / `QuadSource` / `QuadSourceIntegral` / `GkSource` compute targets, `main.py`, `extract_*.py` ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.15

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**:
> the row is added, moved or deleted, and the count and date in its header are corrected. An
> issue owned by another board is moved to **that** board's §4, and its row is deleted from the
> index. The index is an index: one line per issue, pointing here. Where the two disagree, this
> board is right. See `CLAUDE.md`.

### Decisions

**Made by the user, 2026-09-25 (README §6.1):** the replicated write is made atomic;
`QuadSourcePolicy` is kept and added to `QuadSourceIntegral`'s key; the exit-time tolerance lookup
becomes an exact match; the adjacent issues are out of scope.

| Decision | Decided (README §6.2) | Blocked, until decided |
|---|---|---|
| **D1** — key derived rows on their parents | **open**; the user asked what it means (2026-09-26), answered in README §6.2. Recommended: yes | 09 |
| **D2** — fix latent read-path defects no caller reaches | **yes** (2026-09-26) | — |
| **D3** — how much of `OneLoopIntegral`'s factory to repair | **the mechanical defects only**; the rest when `OneLoopIntegral` is built (2026-09-26) | — |
| **D4** — existing stores | **readers are not made version-aware** (2026-09-26); the ordering against `store-retirement` was discharged when it closed | — |
| **D5** — deliberately changing the schema witness and golden fingerprint | **as recommended** (2026-09-26): a new witness per schema-changing prompt, never overwritten, the diff in the log | — |
| **D6** — which `QuadSourcePolicy` each `QuadSourceIntegral` gets | **`QuadSource_policy_1pt5` only, for now** (2026-09-26) | — |

---|---|---|
| **D1** — key derived rows on their parents | yes: same-shard parents with foreign keys, cross-shard as indexed columns; the inventory resolves parents across shards | 09 |
| **D2** — fix latent read-path defects no caller reaches | yes | 05 |
| **D3** — how much of `OneLoopIntegral`'s factory to repair | the mechanical defects only; its round-trip defects recorded as one issue | 05 |
| **D4** — existing stores, and order against `store-retirement` | accept that they become unreadable by the tree for the changed targets; hold 07–09 until `store-retirement` 05 has landed | 07, 08, 09 |
| **D5** — deliberately changing the schema witness and golden fingerprint | a new witness per schema-changing prompt beside the old, never overwritten, the diff in the log; the fingerprint re-captured the same way | 06, 07, 08, 09 |
| **D6** — which `QuadSourcePolicy` each `QuadSourceIntegral` gets | `QuadSource_policy_1pt5`, paired with `GkSource_policy_1pt5` | 08 |

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Checked replicated write](01-checked-replicated-write.md) | **W1**–**W4** | Opus | ✍️ yes, 2026-09-25 | ⬜ ready | — | — |
| 02 | [Repair or refuse at open](02-repair-or-refuse-at-open.md) | **W5**–**W7** | Opus | ✍️ yes, 2026-09-25 | ⬜ after 01 | — | — |
| 03 | [Exact identity lookups](03-exact-identity-lookups.md) | **L1**–**L5** | Sonnet | ✍️ yes, 2026-09-25 | ⬜ after 02 | — | — |
| 04 | [Extend the read-back guard](04-extend-the-read-back-guard.md) | **G1**–**G3** | Sonnet | ✍️ yes, 2026-09-25 | ⬜ ready | — | — |
| 05 | Repair the read paths | **P1**–**P9** | Sonnet | ⏸️ held on 04 | — | — | — |
| 06 | Correct the foreign keys | **S1**–**S5** | Sonnet | 🔓 released 2026-09-26 | — | — | — |
| 07 | Key WKB rows on their numeric row | **K1**–**K2** | Opus | 🔓 released 2026-09-26 | — | — | — |
| 08 | Key `QuadSourceIntegral` on its policy | **K3** | Sonnet | 🔓 released 2026-09-26 | — | — | — |
| 09 | Key derived rows on their parents | **K4**–**K7** | Opus | ⏸️ held on D1 | — | — | — |
| 10 | Build a throwaway store end to end | **V1**–**V3** | Opus | ⏸️ held until 01–09 land | — | — | — |

**05–10 are held, not unplanned.** Their charters are fixed in README §2 and cannot drift to fit
what 01–04 find; what waits is their method.

---

## 2. Items

| Item | Kind | Description | Prompt | Status |
|---|---|---|---|---|
| W1 | **FACILITY** | `replication_in_flight` in the primary: one row, committed before the controller's call, deleted after the last replica returns. One helper through which the get, store and validate paths all go. | 01 | ⬜ |
| W2 | **GUARD** | Every replica's answer checked against the controller's serial, scalar and vector forms; `ReplicationMismatch` names the shard, class and both serials. | 01 | ⬜ |
| W3 | **REMEDY** | Replica stores of `wavenumber_exit_time` and `BackgroundModel` insert-or-verify by serial; `BackgroundModel` replicas pass each value's serial, so copies are row-identical. | 01 | ⬜ |
| W4 | **REMEDY** | One timestamp per replicated write, chosen by the driver and passed to every shard through `Datastore._insert`. | 01 | ⬜ |
| W5 | **FACILITY** | `_reconcile_replicated_tables()` in the constructor, before any actor, no Ray: hot journals rolled back, every replicated table compared across shards by full row (timestamp exempt for old rows). | 02 | ⬜ |
| W6 | **REMEDY** | With an in-flight record set, the missing rows of its class copied forward from its controller, `BackgroundModel` as a unit with `validated` recomputed; the record then cleared. | 02 | ⬜ |
| W7 | **GUARD** | Any other difference refuses the open with `ReplicatedDivergence`, including a serial split and a row the controller lacks. Closes `[00-replicated-writes-can-diverge-across-shards]`. | 02 | ⬜ |
| L1 | **REMEDY** | `wavenumber_exit_time.build` by serial equality on atol, rtol and `==` on stepping; the dead `ORDER BY` removed; the query out of the loop. Closes `[02-wavenumber-exit-time-tolerance-is-an-inequality-key]` and `[02-exit-time-lookup-runs-inside-the-subhorizon-loop]`. | 03 | ⬜ |
| L2 | **REMEDY** | `IntegrationSolver.build` by `stepping ==` and `one_or_none()`. Closes `[03-integrationsolver-stepping-minimum-lookup]`. | 03 | ⬜ |
| L3 | **GUARD** | `.scalar()` → `one_or_none()` on the eight identity lookups (audit §3.8), so a duplicate raises. | 03 | ⬜ |
| L4 | **REMEDY** | `redshift.build` matches `z = 0` (audit §3.9). | 03 | ⬜ |
| L5 | **RECORD** | `[05a-two-verification-scripts-still-query-the-exit-time-under-the-shared-pair]` narrowed: after L1 the two scripts miss rather than silently reuse. | 03 | ⬜ |
| G1 | **GUARD** | The read-back guard visits every function of every factory, `read_batch` and `_build_impl_*` included, linking a nested helper's parameter to its row. | 04 | ⬜ |
| G2 | **GUARD** | A column-existence test: every `<table>.c.<name>` and every insert key resolves against the built schema. | 04 | ⬜ |
| G3 | **RECORD** | `KNOWN_UNFIXED`: what G1–G2 find today, for prompt 05 to empty. Closes `[01-read-batch-is-outside-the-guard]`. | 04 | ⬜ |
| P1 | **REMEDY** | The tagged join in `QuadSourceIntegral` / `OneLoopIntegral` `read_batch` on `table.c.serial`. Closes `[00-tagged-read-batch-joins-an-unselected-alias]`. | 05 | ⏸️ |
| P2 | **REMEDY** | `QuadSourceIntegral.read_batch` gives its exit times their own serials (audit §3.2). | 05 | ⏸️ |
| P3 | **REMEDY** | `break_point_kind` in the Tk/Gk numeric value reads and `GkNumericValue.read_batch`; one key name for `z_init`. Closes `[00-numeric-value-parent-lookup-omits-break-point-kind]`. | 05 | ⏸️ |
| P4 | **REMEDY** | `z_init` in the `GkWKBValue` reads; more than one parent per `z_source` refused (audit §3.4). | 05 | ⏸️ |
| P5 | **REMEDY** | The two `…_is_response` flags read `response` (audit §3.11). | 05 | ⏸️ |
| P6 | **REMEDY** | `WKB_criterio`, `z_serial`, `model_id`, `friction` (audit §3.10–§3.11), under D2. | 05 | ⏸️ |
| P7 | **REMEDY** | Tags default to `[]` at every value helper and `read_batch`; the three `raise ("…")`. | 05 | ⏸️ |
| P8 | **REMEDY** | `OneLoopIntegral`: its own tag table, `read_batch` a method with a return value, as D3 bounds. Closes `[00-oneloop-lookup-joins-the-wrong-tag-table]`. | 05 | ⏸️ |
| P9 | **GUARD** | Round-trip tests of every `read_batch`, tagged and untagged; `KNOWN_UNFIXED` empty. | 05 | ⏸️ |
| S1 | **REMEDY** | `QuadSource.Tq_serial` → `TkNumericIntegration.serial`; the `Tr_serial` comment. Closes `[00-quadsource-tq-serial-has-the-wrong-foreign-key]`. | 06 | ⏸️ |
| S2 | **REMEDY** | `GkSourcePolicyData.wavenumber_exit_serial` gains its foreign key. | 06 | ⏸️ |
| S3 | **REMEDY** | `real_store_fixtures.py`'s undeclared-reference map corrected. | 06 | ⏸️ |
| S4 | **GUARD** | A test that turns `PRAGMA foreign_keys` on over a fixture store written by the real factories and finds no violation. | 06 | ⏸️ |
| S5 | **RECORD** | The first deliberate schema-witness change, under D5, including the exit-time and `IntegrationSolver` `stepping_mode` flags made exact. | 06 | ⏸️ |
| K1 | **REMEDY** | `numeric_serial` on both WKB tables, in the lookup; `z_init` and the initial values leave the key; `main.py` passes the parent. Closes `[20-wkb-rows-consume-numeric-initial-data]`. | 07 | ⏸️ |
| K2 | **REMEDY** | `_drop_order` respects the new key. | 07 | ⏸️ |
| K3 | **REMEDY** | `quad_policy_serial` on `QuadSourceIntegral`, in the key, through target, factory, `main.py` and the extraction script. Closes `[00-quadsourcepolicy-rows-are-referenced-by-nothing]`. | 08 | ⏸️ |
| K4 | **REMEDY** | `QuadSource` keyed on `Tq_serial` and `Tr_serial`. | 09 | ⏸️ |
| K5 | **REMEDY** | `QuadSourceIntegral` keyed on `source_serial`, `data_serial` and its Tk rows. | 09 | ⏸️ |
| K6 | **REMEDY** | `GkSource` keyed on its Gk numeric and WKB rows. | 09 | ⏸️ |
| K7 | **FACILITY** | The store inventory resolves key parents across shards, so cross-shard parents are fingerprint identity. | 09 | ⏸️ |
| V1 | **VERIFY** | A scoped build through the registry into a new throwaway store, interrupted and resumed. | 10 | ⏸️ |
| V2 | **VERIFY** | Every target extracted from it; its fingerprint; the census of audit §4.4. | 10 | ⏸️ |
| V3 | **RECORD** | The campaign closed; what A3 v2's launch must now pass. | 10 | ⏸️ |

---

## 3. Active and unresolved issues

### 3.1 Assigned to this campaign (2026-09-26)

Each stays on its owning board, which carries an **Assigned** line, and closes on that board's §4.

| Issue | Owning board | Prompt |
|---|---|---|
| `[00-replicated-writes-can-diverge-across-shards]` | store-fingerprint | 01, closed by 02 |
| `[02-wavenumber-exit-time-tolerance-is-an-inequality-key]` | tolerance-convergence | 03 |
| `[02-exit-time-lookup-runs-inside-the-subhorizon-loop]` | store-fingerprint | 03 |
| `[03-integrationsolver-stepping-minimum-lookup]` | GkTk-remedial | 03 |
| `[01-read-batch-is-outside-the-guard]` | datastore-readback | 04 |
| `[00-tagged-read-batch-joins-an-unselected-alias]` | store-fingerprint | 05 |
| `[00-numeric-value-parent-lookup-omits-break-point-kind]` | store-fingerprint | 05 |
| `[00-oneloop-lookup-joins-the-wrong-tag-table]` | store-fingerprint | 05 |
| `[00-quadsource-tq-serial-has-the-wrong-foreign-key]` | store-fingerprint | 06 |
| `[20-wkb-rows-consume-numeric-initial-data]` | GkTk-remedial | 07 |
| `[00-quadsourcepolicy-rows-are-referenced-by-nothing]` | store-fingerprint | 08 |

### 3.2 Opened by the audit (2026-09-25), recorded for their owners

None is assigned to a prompt of this campaign.

- **[00-the-inventory-cannot-see-a-serial-split]** *(opened 2026-09-25 by the audit)*
  - **The defect.** The structured inventory compares replicated classes across shards by key,
    tags, `validated` and value count. It does not compare serials, the `source`/`response` flags,
    timestamps or value contents (audit §4.3, `replicated_divergence_probe.py`). A store holding a
    serial split, or shards that disagree on a flag, fingerprints with no problem.
  - **Impact.** The fingerprint is not evidence that a store's replicated tables agree. After
    prompt 02, the check at open is that evidence for any store the tree can open. The fingerprint
    reads stores the tree may not open.
  - **Next step.** Decide whether the fingerprint should carry replicated agreement, or say plainly
    that it does not. Not this campaign's: README §1.
- **[00-oneloop-factory-cannot-round-trip-a-row]** *(opened 2026-09-25 by the audit)*
  - **The defect.** `sqla_OneLoopIntegral_factory.build` puts no `source_serial` in the payload
    and the table has no such column, while `ComputeTargets/OneLoopIntegral.py:43` reads
    `payload["source_serial"]`, so every hit raises `KeyError`. `store()` writes a `validated` key
    to a table with no such column (audit §3.12, `probe_06`).
  - **Impact.** None while nothing builds the target. The first one-loop build cannot read back a
    row it stored.
  - **Next step.** Part of how `OneLoopIntegral` is keyed, which the user put downstream beside
    `[02-oneloopintegral-is-a-ninth-keyed-object-type]` (README §6.1.4, D3).
- **[00-the-5pt0-policy-pair-is-created-and-never-used]** *(opened 2026-09-25 by the audit)*
  - **The defect.** `main.py` creates `GkSource_policy_5pt0` and `QuadSource_policy_5pt0` and
    passes them into `run_pipeline`, which uses neither (audit §2.5).
  - **Impact.** Two dead rows in every store, which a reader may take for configuration that
    applied.
  - **Next step.** An author's decision: remove them, or say what they are for. Not a key defect,
    so not this campaign's.

---

## 4. Resolved issues

None yet.

---

## 5. Baselines

README §7. Re-measured at every dispatch.
