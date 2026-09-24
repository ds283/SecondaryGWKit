# Store fingerprint campaign — implementation state

**Last updated:** 2026-09-24 · **Status: 0 of 5 prompts landed. 01 and 02 are written; 02's
dispatch waits on decision D1. 03–05 are held (README §2).**

The campaign was opened on 2026-09-24. It owns `run-registry`'s
`[04-a-runs-product-is-named-but-never-fingerprinted]`, as amended at `218ca74`: a store's content
fingerprint, as digests in its sidecar, computed read-only from a structured inventory that names
work items by physical labels and tag sets.
[`docs/store-fingerprint-audit.md`](../../docs/store-fingerprint-audit.md) established what that
inventory must read, and that a read-only reader with no Ray is possible. It also opened eight
issues (§3). Two are assigned to this campaign's prompts; the other six are recorded for their
owners.

**Campaign:** [`README.md`](README.md) · **Audit:** [`docs/store-fingerprint-audit.md`](../../docs/store-fingerprint-audit.md) ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.13

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**: the
> row added, moved or deleted, and the count and date in its header corrected. The index is an
> index: one line per issue, pointing here. Where the two disagree, this board is right. See
> `CLAUDE.md`.

**Legend:** ✍️ written · ⏳ not dispatched · ⏸️ held · ✅ landed

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [One schema builder, and a read-only reader](01-a-read-only-store-reader.md) | **F1**–**F3** | Opus | ✍️ yes, 2026-09-24 | ⏳ not dispatched | — | — |
| 02 | [A structured inventory](02-a-structured-inventory.md) | **F4**–**F7** | Opus | ✍️ yes, 2026-09-24 | ⏳ waits on **D1** | — | — |
| 03 | *One inventory service* | **F8**–**F9** | — | ⏸️ held until 02 lands | — | — | — |
| 04 | *The fingerprint* | **F10**–**F12** | — | ⏸️ held until 02 lands; **D2** | — | — | — |
| 05 | *Fingerprint the real stores* | **F13** | — | ⏸️ held until 04 lands; **D3** | — | — | — |

The charters of 03–05 are fixed in README §2. Only their methods wait on 02's structure and on the
user's decisions. They are held by design, not missing.

---

## 2. Items

| Item | Kind | Description | Prompt | Status |
|---|---|---|---|---|
| F1 | **REMEDY** | One schema builder, `Datastore/SQL/schema.py` `build_schema`, called by the actor. Witnessed unchanged by a schema captured from the code before the change. | 01 | ⏳ written |
| F2 | **FACILITY** | `Datastore/store_reader.py` `open_read_only`: a closed store's shards opened `mode=ro`, tables from `build_schema`, absent tables and columns reported, journals refused, no Ray, no write path. | 01 | ⏳ written |
| F3 | **FACILITY** | A real multi-shard store for tests, built with no Ray, including an "old store" variant. | 01 | ⏳ written |
| F4 | **FACILITY** | `Datastore/store_inventory.py` `read_inventory`: records built by each factory in dependency order. Each record has a canonical key, a tag set, `validated` and `value_count`. Parents are referenced by the digest of their key; there is one `canonical`. | 02 | ⏳ written; D1 |
| F5 | **REMEDY** | The key of every class, from its lookup: physical leaves, optional filters included, nothing store-local. Real `QuadSourceIntegral`, `OneLoopIntegral` and `GkSourcePolicyData` records. | 02 | ⏳ written |
| F6 | **REMEDY** | Shards and what goes wrong: sharded classes are a union; replicated classes are read from every shard, and a divergence is named. Duplicates, old stores and orphans are named problems. | 02 | ⏳ written |
| F7 | **MEASUREMENT** | Records only, never `*Value` rows. Size, time and memory measured on a copy of the sweep store, with a one-row discriminator. | 02 | ⏳ written |
| F8 | **REMEDY** | The display (`main.py --inventory`, read-only and needing no Ray) and `available_run_labels` consume the structured inventory. Closes `[00-inventory-run-prunes-unvalidated-rows-by-default]` and the `BackgroundModel` half of `[03-qcd-inventory-does-not-report-the-representation]`. | 03 | ⏸️ held |
| F9 | **REMEDY** | Retire the old three shapes, the old `inventory()` methods, `ShardedPool.inventory`, `_merge_queue` and `inventory_config`, so that there is one inventory service. | 03 | ⏸️ held |
| F10 | **FACILITY** | The fingerprint, a pure function of the structured inventory. It holds a format version, and per class and per tag set a count and a digest over the sorted canonical records, plus an overall digest. No timestamps. | 04 | ⏸️ held |
| F11 | **FACILITY** | The `fingerprint` sidecar field, a known field of `RunRegistry.stores` that copy and move carry. `python -m RunRegistry store fingerprint`: read-only, refuses a store a `running` run names, compares against the recorded value and writes only when asked. | 04 | ⏸️ held |
| F12 | **FACILITY** | The digest in the run record at finish, and `scoped_pipeline_run.py` taking one when a registered run finishes. Closes `run-registry`'s `[04-a-runs-product-is-named-but-never-fingerprinted]`. | 04 | ⏸️ held; D2 |
| F13 | **MEASUREMENT** | The three real stores fingerprinted, plus a registry copy of one; the digests localise the known differences. They are written into the real sidecars only if D3 says so. | 05 | ⏸️ held; D3 |

---

## 3. Active and unresolved issues

All eight were opened on 2026-09-24 by the audit
([`docs/store-fingerprint-audit.md`](../../docs/store-fingerprint-audit.md)), which is not a prompt.

- **[00-inventory-run-prunes-unvalidated-rows-by-default]** *(opened 2026-09-24 by the audit;
  **assigned to prompt 03**)*
  - **The defect.** `main.py --inventory` builds the full read-write pool before it reaches its
    `--inventory` branch (`main.py:3484-3504`). `--prune-unvalidated` is a
    `BooleanOptionalAction` with `default=True` (`main.py:233-238`), so each actor `DELETE`s its
    shard's unvalidated rows and their values and tags on open (`Datastore.py:447-472`). **An
    inventory run that does not pass `--no-prune-unvalidated` deletes data.** It also runs the
    `--drop` actions, creates missing tables, and needs Ray.
  - **Impact.** Every inventory of a store that holds unvalidated rows changes the store it is
    reporting on. Every `datastore-portability` demonstration passed the flag, so nothing
    measured there was affected.
  - **Next step.** Prompt 03 moves `--inventory` onto the read-only reader, before any pool is
    built.
  - **Assigned (2026-09-24):** prompt 03 of this campaign. It is the prompt that owns the
    `--inventory` branch, and the read-only reader it needs is prompt 01's.
- **[00-oneloop-lookup-joins-the-wrong-tag-table]** *(opened 2026-09-24 by the audit)*
  - **The defect.** `sqla_OneLoopIntegral_factory.build` filters requested tags against
    `tables["QuadSourceIntegral_tags"]` (`Datastore/SQL/ObjectFactories/OneLoopIntegral.py:141`),
    while `store()` writes `OneLoopIntegral_tags` (`:243`). So a tagged `OneLoopIntegral` lookup
    joins `QuadSourceIntegral` tag rows against `OneLoopIntegral` serials.
  - **Impact.** The table is empty in the sweep store (0 rows in the 2026-09-24 inventories), so
    there is no effect today. The first
    tagged lookup would miss rows that exist, or match rows by coincidence of serials.
  - **Next step.** A one-line fix in the factory, in whichever campaign next has
    `OneLoopIntegral.py`'s lookup in scope. This campaign's inventory reads
    `OneLoopIntegral_tags`, and does not fix the lookup.
- **[00-tagged-read-batch-joins-an-unselected-alias]** *(opened 2026-09-24 by the audit;
  **suspected, not run**)*
  - **The defect.** The tag joins in `read_batch` use `query.c.serial`
    (`QuadSourceIntegral.py:679`; `OneLoopIntegral.py:332`). Compiled under SQLAlchemy 2.0.39, the
    pattern gives a deprecation warning and SQL that names `anon_1.serial`, which is not in the
    FROM clause. `extract_QuadSourceIntegral_data.py:1087-1099` passes non-empty tags.
  - **Impact.** If the compiled SQL fails when run, tagged batch reads of `QuadSourceIntegral`
    fail, and every extraction from a store written since the run tags existed fails with them.
    This has not been run against a store.
  - **Next step.** Run a tagged `read_batch` against a copy of a store, to confirm or refute. Then
    fix it in the campaign that owns `read_batch`; `datastore-readback`'s
    `[01-read-batch-is-outside-the-guard]` is the neighbouring record.
- **[00-quadsource-tq-serial-has-the-wrong-foreign-key]** *(opened 2026-09-24 by the audit)*
  - **The defect.** `QuadSource.Tq_serial` declares `ForeignKey("QuadSource.serial")`
    (`QuadSource.py:113-116`), but it holds a transfer-function store id
    (`ComputeTargets/QuadSource.py:473`). `Tr_serial` has no foreign key.
  - **Impact.** SQLite does not enforce foreign keys here, so nothing fails. The schema states a
    relation that is false.
  - **Next step.** Correct or remove the constraint in whichever campaign next touches
    `QuadSource`'s schema. Schema changes cost nothing.
- **[00-numeric-value-parent-lookup-omits-break-point-kind]** *(opened 2026-09-24 by the audit)*
  - **The defect.** The parent query in `TkNumericValue`'s build (`TkNumericIntegration.py:873-878`),
    and its `GkNumeric` counterpart, filter on model, wavenumber exit, `z_init` / `z_source`,
    validated and the tolerances. They do not filter on `break_point_kind`, which is in the
    parent's own lookup.
  - **Impact.** A store holding two parents that differ only in `break_point_kind` would make the
    value read ambiguous. `one_or_none()` would then raise, or pick the wrong parent's values.
    Every store today uses one `BREAK_POINT_KIND`.
  - **Next step.** Add the filter, in the campaign that owns these factories' reads.
- **[00-quadsourcepolicy-rows-are-referenced-by-nothing]** *(opened 2026-09-24 by the audit)*
  - **The defect.** `main.py:3651-3657` creates `QuadSourcePolicy` rows, but no factory table
    references the class. `QuadSourceIntegral` keys on `GkSourcePolicy`
    (`QuadSourceIntegral.py:135-141`).
  - **Impact.** The rows are dead configuration, and a reader might take them for the policy that
    governed the integrals.
  - **Next step.** Decide whether `QuadSourcePolicy` is vestigial, and remove it or wire it up. That
    is an author's decision about the physics, not this campaign's.
- **[00-replicated-writes-can-diverge-across-shards]** *(opened 2026-09-24 by the audit)*
  - **The defect.** A replicated write commits on the controlling shard first, and on the other
    shards afterwards, in separate transactions (`ShardedPool.py:975-1030`, `:1158-1167`). A crash
    in between leaves the copies different for good. Each shard also stamps its own `timestamp`
    (`Datastore.py:665`). `ShardedPool.inventory` reads a replicated class from one shard chosen at
    random (`:1469-1483`), so it hides any divergence.
  - **Impact.** Unknown: no store has been checked. After such a crash, a lookup's answer depends on
    which shard serves it.
  - **Next step.** Prompt 02's inventory reads every shard and names any divergence, which measures
    it. Making the write atomic belongs to the datastore code and is not this campaign's.
- **[00-build-schema-reads-registration-before-its-none-check]** *(opened 2026-09-24 by the
  audit; **assigned to prompt 01**)*
  - **The defect.** `Datastore._build_schema` calls `registration_data.get(...)`
    (`Datastore.py:314`) before its `is not None` check (`:320`).
  - **Impact.** It is latent: no factory returns `None`.
  - **Next step.** Prompt 01 moves this code into `build_schema`, and fixes the order as it does.
  - **Assigned (2026-09-24):** prompt 01 of this campaign, which is moving the code anyway.

---

## 4. Resolved issues

None yet.

---

## 5. Baselines

At `f53598f` (README §7): AdaptiveLevin 32, ComputeTargets 552 (the known wall-clock flake
aside), CosmologyModels 39, Datastore 70, LiouvilleGreen 148 (1 skipped), RunRegistry 84.
