# Store fingerprint campaign — implementation state

**Last updated:** 2026-09-25 · **Status: 2 of 5 prompts landed (01, 02). 03 and 04 were written on
2026-09-25 against 02's structure, and are ready to dispatch in either order. 05 is held until 04
lands. Decisions D1–D3 were made on 2026-09-24, and D4 on 2026-09-25 (README §6.2).**

The campaign was opened on 2026-09-24. It owns `run-registry`'s
`[04-a-runs-product-is-named-but-never-fingerprinted]`, as amended at `218ca74`: a store's content
fingerprint, as digests in its sidecar, computed read-only from a structured inventory that names
work items by physical labels and tag sets.
[`docs/store-fingerprint-audit.md`](../../docs/store-fingerprint-audit.md) established what that
inventory must read, and that a read-only reader with no Ray is possible. It also opened eight
issues (§3). Two are assigned to this campaign's prompts; the other six are recorded for their
owners.

**Prompt 01 landed 2026-09-25.** There is one schema builder, `Datastore/SQL/schema.py`
`build_schema`. The actor's `_build_schema` calls it, and adds only its inserters. Both reproduce,
byte for byte, a description of the schema captured from the unchanged code at `417c647`
(`Datastore/tests/data/schema_at_base.json`). `Datastore/store_reader.py` `open_read_only` reads a
closed store's shards through `ShardedPool._read_closed_store`. It opens every file `mode=ro`,
refuses a journal by name, and reports absent and extra tables and columns. It has no write path
and needs no Ray. A copy of the sweep store read through it with every per-table count equal to an
independent `sqlite3` count. The copy and the originals were unchanged, and the copy has no absent
or extra schema. This closes `[00-build-schema-reads-registration-before-its-none-check]` (§4).

**Prompt 02 landed 2026-09-25.** `Datastore/store_inventory.py` `read_inventory` builds, through
the read-only reader, a structured inventory of all 21 classes in dependency order. Each factory
gained one static method, `inventory_records`, which states its class's key from its `build()`
lookup, the optional filters included, and calls one shared reader. A record has four fields:
- a `key` of canonical leaves (`canonical`: `float.hex` of the stored value, D1) and parent
  digests (SHA-256 of the parent's canonical key, and of its tags if it has them);
- `tags`, from the record's own association table;
- `validated`;
- `value_count`, from one `GROUP BY` per value table per shard.

Replicated classes are compared across every shard. Absent tables, incomplete columns, orphans,
unresolved parents, duplicates and divergences are named problems. The one difference from audit
§4 is `GkSourcePolicyData`'s `k`, which its lookup filters on.

It was read on a copy of the sweep store, for 30 341 records:
- every class's record count equals its row count;
- every value table's total equals its parents' summed counts;
- there is **no problem at all**: no orphan, no duplicate, and **no replicated divergence**;
- the read took 6.8 s, with a peak RSS of 228 MB;
- deleting one `QuadSourceIntegral` row removed exactly that record, named by its physical labels,
  and changed nothing else.

It opened `[02-exit-time-lookup-runs-inside-the-subhorizon-loop]` (§3).

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
| 01 | [One schema builder, and a read-only reader](01-a-read-only-store-reader.md) | **F1**–**F3** | Opus 5.5 | ✍️ yes, 2026-09-24 | ✅ 2026-09-25 | *"Add one schema builder and a read-only store reader"* | [`logs/01-…`](logs/01-a-read-only-store-reader.md) |
| 02 | [A structured inventory](02-a-structured-inventory.md) | **F4**–**F7** | Opus 5.5 | ✍️ yes, 2026-09-24 | ✅ 2026-09-25 | *"Add a structured store inventory keyed by physical labels"* | [`logs/02-…`](logs/02-a-structured-inventory.md) |
| 03 | [One inventory service](03-one-inventory-service.md) | **F8**–**F9** | Opus | ✍️ yes, 2026-09-25 | ⏳ not dispatched | — | — |
| 04 | [The fingerprint](04-the-fingerprint.md) | **F10**–**F12** | Opus | ✍️ yes, 2026-09-25 | ⏳ not dispatched | — | — |
| 05 | *Fingerprint the real stores* | **F13** | — | ⏸️ held until 04 lands | — | — | — |

The charters of 03–05 are fixed in README §2. 03 and 04 were written once 02's structure had
landed; 05's method waits on 04's format. It is held by design, not missing. Orchestration prompts:
[`orchestrator/prompt-03.md`](orchestrator/prompt-03.md) and
[`orchestrator/prompt-04.md`](orchestrator/prompt-04.md).

---

## 2. Items

| Item | Kind | Description | Prompt | Status |
|---|---|---|---|---|
| F1 | **REMEDY** | One schema builder, `Datastore/SQL/schema.py` `build_schema`, called by the actor. Witnessed unchanged by a schema captured from the code before the change. | 01 | ✅ **Done, 2026-09-25.** `build_schema(metadata, factories) -> BuiltSchema(tables, records)`: the old loop minus the inserters, with the `None` check first. `_build_schema` calls it, adds the inserters, and fills `_tables` / `_inserters` / `_schema` as before. The witness was captured at `417c647`, from a `git archive` of the base and from the unedited tree (identical). It covers all 37 classes: columns, types, keys, indexes, constraints, DDL and the record fields. Both the function and the actor reproduce it byte for byte. The factory map did not move. **Deliberate breakage (ii), (iii), (vi)** each fail the tests written against them, and (iii) fails only the actor's. |
| F2 | **FACILITY** | `Datastore/store_reader.py` `open_read_only`: a closed store's shards opened `mode=ro`, tables from `build_schema`, absent tables and columns reported, journals refused, no Ray, no write path. | 01 | ✅ **Done, 2026-09-25.** Shards come from `_read_closed_store`. Every file is opened `sqlite:///file:{path}?mode=ro&uri=true`. `-journal` / `-wal` / `-shm` beside the primary or any shard is refused, naming both files. The report per shard is absent tables, absent and extra columns, and extra tables. Every engine is disposed on exit. File hashes, sizes, mtimes and the listing are unchanged by a full read, and INSERT and DDL raise "readonly". In a child interpreter `ray.is_initialized()` stays false. On a copy of the sweep store, all 37 tables on all 4 shards count equal to an independent `sqlite3` count, with no absent or extra schema, and the copy and originals were byte-identical. **Deliberate breakage (i), (iv), (v)** each fail the tests written against them. |
| F3 | **FACILITY** | A real multi-shard store for tests, built with no Ray, including an "old store" variant. | 01 | ✅ **Done, 2026-09-25.** `Datastore/tests/real_store_fixtures.py` `build_real_store`. Its primary is written by `ShardedPool._write_shard_data`, with `config/sharding.py`'s lists and shard-key rows. It has two shards with every table. Replicated rows (ten tables) are in both, with the same serials. Each shard has a `TkNumericIntegration` with `TkNumeric_tags` and `TkNumericValue` rows. Row sets are plain data, extensible with `with_rows`. `build_old_store` gives shard 1 without `OneLoopIntegral_tags` and shard 0's `TkNumericIntegration` without `stop_Tprime`. `missing_tables` / `missing_columns` / `extra_sql` give other cases. |
| F4 | **FACILITY** | `Datastore/store_inventory.py` `read_inventory`: records built by each factory in dependency order. Each record has a canonical key, a tag set, `validated` and `value_count`. Parents are referenced by the digest of their key; there is one `canonical`, which uses `float.hex` (D1). | 02 | ✅ **Done, 2026-09-25.** `read_inventory(primary) -> StoreInventory` (per class: `records`, `count`, timestamp range, `problems`, plus `replicated`, `tagged`, `parents`). It builds `INVENTORY_CLASSES` in dependency order on every shard, and refuses to resolve a parent not yet built. `Record(key, tags, validated, value_count)` is frozen and JSON-safe, and a values field can be added beside the four. `canonical` is the only float formatter, and an AST test holds that. `canonical_json` sorts keys and has no whitespace. `reference_digest` is SHA-256 of the key, or of the key and tags for a tagged class. Each of 21 factories gained one static `inventory_records` (382 lines added, 0 removed) calling `read_records`. `resolve` expands parents for display. **Deliberate breakage (i), (ii), (iii), (vii)** each fail their tests. |
| F5 | **REMEDY** | The key of every class, from its lookup: physical leaves, optional filters included, nothing store-local. Real `QuadSourceIntegral`, `OneLoopIntegral` and `GkSourcePolicyData` records. | 02 | ✅ **Done, 2026-09-25.** Every lookup was read. The shipped key table, with its lookup lines, is in the log. The one difference from audit §4 is **`GkSourcePolicyData`'s `k`** (`GkSourcePolicyData.py:99` filters on `wavenumber_exit_serial`). `z_init` / `z_source` / `z_response` are always in the key. The cosmology is resolved through `cosmology_type`. `OneLoopIntegral` tags come from `OneLoopIntegral_tags`. No solver, label, name, `version` key or timestamp is in any key. Test 1: the same content under relabelled serials, replicated ones included, and on other shards gives equal inventories. Test 2: each of 83 identity columns alone changes the records. Test 3: 42 non-identity variations change nothing. **Deliberate breakage (i), (v), (vi)** each fail their tests. |
| F6 | **REMEDY** | Shards and what goes wrong: sharded classes are a union; replicated classes are read from every shard, and a divergence is named. Duplicates, old stores and orphans are named problems. | 02 | ✅ **Done, 2026-09-25.** A sharded class is the union of its shards. A replicated class is compared across every comparable shard, and takes the lowest's records. A shard whose class, tag or value table is absent, or whose key columns are incomplete, is named and left out. The named problems are `absent-table`, `incomplete`, `replicated-divergence` (shard, both-way counts, up to five keys), `duplicate` (all kept), `orphan-value`, `orphan-tag` and `unresolved-parent`, each with a count and up to five examples. Tests 7–9 show each case named, on the right shard, with nothing else changed. **Deliberate breakage (iv), (viii)** each fail their tests. |
| F7 | **MEASUREMENT** | Records only, never `*Value` rows. Size, time and memory measured on a copy of the sweep store, with a one-row discriminator. | 02 | ✅ **Done, 2026-09-25.** Every statement naming a value table is a `count(*) … GROUP BY` (test 6). The sweep copy has 30 341 records, and every class's record count equals its row count. Every value table's total equals its parents' summed `value_count`. There is no problem of any kind, and the 12 replicated classes agree on all four shards. `read_inventory` took 6.8 s, with a peak RSS of 228 MB (175 MB of it the factories' imports). The full listing is 19.3 MB of JSON. Deleting `QuadSourceIntegral` serial 329386 on shard 0 took the class from 7 706 to 7 705 records, removing exactly the record with $k=q=r=3.05\times10^7$, $z_{\rm response}=0.1$, atol $10^{-32}$, rtol $10^{-8}$. The other 20 classes were identical, record for record. Its 9 tag rows were then named `orphan-tag`. The copy and the originals were unchanged by every read, and the copy was deleted. |
| F8 | **REMEDY** | The display (`main.py --inventory`, read-only and needing no Ray) and `available_run_labels` consume the structured inventory. Closes `[00-inventory-run-prunes-unvalidated-rows-by-default]` and the `BackgroundModel` half of `[03-qcd-inventory-does-not-report-the-representation]`. | 03 | ✍️ written |
| F9 | **REMEDY** | Retire the old three shapes, the old `inventory()` methods, `ShardedPool.inventory`, `_merge_queue` and `inventory_config`, so that there is one inventory service. | 03 | ✍️ written |
| F10 | **FACILITY** | The fingerprint, a pure function of the structured inventory. It holds a format version, and per class and per tag set a count and a digest over the sorted canonical records, plus an overall digest. No timestamps. | 04 | ✍️ written |
| F11 | **FACILITY** | The `fingerprint` sidecar field, a known field of `RunRegistry.stores` that copy and move carry. `python -m RunRegistry store fingerprint`: read-only, refuses a store a `running` run names, compares against the recorded value and writes only when asked. | 04 | ✍️ written |
| F12 | **FACILITY** | The digest in the run record at finish. `scoped_pipeline_run.py` and `quadsource_atol_sweep.py`, including `--build` for the A3 v2 store (D2), take one when a registered run finishes. Closes `run-registry`'s `[04-a-runs-product-is-named-but-never-fingerprinted]`. | 04 | ✍️ written |
| F13 | **REMEDY** | Remedial: the three existing stores fingerprinted read-only, and each fingerprint written into its sidecar (D3), as is any store built here before 04 landed, such as the A3 v2 store. A registry copy of one is fingerprinted, and the digests localise the known differences. | 05 | ⏸️ held |

---

## 3. Active and unresolved issues

Eight were opened on 2026-09-24 by the audit
([`docs/store-fingerprint-audit.md`](../../docs/store-fingerprint-audit.md)), which is not a prompt.
Prompt 01 closed one of them on 2026-09-25 (§4). Prompt 02 opened one on 2026-09-25, and measured
`[00-replicated-writes-can-diverge-across-shards]` on the sweep store. Eight remain.

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
  - **Measured (2026-09-25) by prompt 02, on a copy of the sweep store: no divergence.** All 12
    replicated classes, including `BackgroundModel` with its tags and its 1 740
    `BackgroundModelValue` rows, hold the same multiset of key, tags, validated flag and value count
    on all four shards. The A3 store and the backup have not been read; prompt 05 reads them. The
    defect in the write path stands.
- **[02-exit-time-lookup-runs-inside-the-subhorizon-loop]** *(opened 2026-09-25 by prompt 02)*
  - **The defect.** In `sqla_wavenumber_exit_time_factory.build`, the line
    `row_data = conn.execute(query).one_or_none()` (`Datastore/SQL/ObjectFactories/wavenumber.py:270`)
    is indented inside `for z_offset in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:` (`:267`), not
    after it.
  - **Impact.** Every exit-time lookup runs its query six times, and the first five lack some of the
    sub-horizon columns. The last iteration's result is complete, so the answer is right today. If
    the list were ever empty, `row_data` would be unbound and the lookup would fail with an
    `UnboundLocalError`.
  - **Next step.** Dedent the line by one level, in whichever campaign next has
    `wavenumber_exit_time.build` in scope. This campaign's scope excludes every factory's `build`
    (README §1).

---

## 4. Resolved issues

- **[00-build-schema-reads-registration-before-its-none-check]** *(opened 2026-09-24 by the
  audit; **assigned to prompt 01**)*
  - **The defect.** `Datastore._build_schema` calls `registration_data.get(...)`
    (`Datastore.py:314`) before its `is not None` check (`:320`).
  - **Impact.** It is latent: no factory returns `None`.
  - **Next step.** Prompt 01 moves this code into `build_schema`, and fixes the order as it does.
  - **Assigned (2026-09-24):** prompt 01 of this campaign, which is moving the code anyway.
  - **Closed (2026-09-25) by prompt 01.** `build_schema` (`Datastore/SQL/schema.py`) checks
    `registration_data` for `None` before reading it. Such a class gets
    `{"name", "validate_on_startup": False, "table": None}`, and the actor adds `"insert": None`.
    `test_schema_builder.TestNoneRegistration` covers the function and the actor. Reverting the
    fix (deliberate breakage (vi)) makes both raise `AttributeError`. No factory returns `None`,
    so behaviour on every real store is unchanged. The schema witness shows that nothing else
    changed. Log: [`logs/01-a-read-only-store-reader.md`](logs/01-a-read-only-store-reader.md).

---

## 5. Baselines

At `f53598f` (README §7): AdaptiveLevin 32, ComputeTargets 552 (the known wall-clock flake
aside), CosmologyModels 39, Datastore 70, LiouvilleGreen 148 (1 skipped), RunRegistry 84.

| Suite | At `417c647` (before prompt 01) | After prompt 01 | After prompt 02 |
|---|---|---|---|
| `AdaptiveLevin` | 32 OK | 32 OK | 32 OK |
| `ComputeTargets` | 552 OK (the flake is known) | 552 OK (the flake passed) | 552 OK (the flake passed) |
| `CosmologyModels` | 39 OK | 39 OK | 39 OK |
| `Datastore` | 70 OK | **92 OK** (+22) | **141 OK** (+49) |
| `LiouvilleGreen` | 148 OK (skipped=1) | 148 OK (skipped=1) | 148 OK (skipped=1) |
| `RunRegistry` | 84 OK | 84 OK | 84 OK |
