# Datastore-generic audit — project knowledge and old-store fallbacks in the datastore layer

**Date:** 2026-10-04 · **Tree:** `handover-remedial` at `dbd8a51`, clean (read in a detached
worktree) · **Author:** Claude Opus 5.5 · **Campaign:** [`prompts/datastore-generic`](../prompts/datastore-generic/README.md)
· **Status:** read-only, **draft for the user's review**. No production file or test was changed.
One probe and its output are under [`docs/datastore-generic-audit/`](datastore-generic-audit/).
Under `var/`, one file was read: the A3 baseline's primary, `mode=ro`, its SHA-256 the same before
and after (G4).

**Why.** On 2026-10-04 the user, auditing recent changes to `ShardedPool` and `Datastore`, found that
`ShardedPool` names `BackgroundModel` and `wavenumber`: a generic component tied to its client's
tables. The user had made it generic once already, in December 2025 (`11fa287`, `a2bd966`,
`29cb2a3`: every project-specific sharding choice moved to `config/sharding.py`). The user asked how
it happened, and decided that the datastore layer is made generic again and that **every fallback
for a store written by older code is deleted**: stores are regenerated, not accommodated. That
decision reflects the build phase. Once a store holds curated results behind a published paper,
keeping access to it may matter, and the question reopens (README §6.1, D2).

**How each finding is marked.**
- **[measured]** — a committed probe reproduces it; the output file is named.
- **[read]** — read in the source at `dbd8a51`; file and line are named.
- **[inferred]** — neither; what it was inferred from is named.

**Abbreviations.** SP = `Datastore/SQL/ShardedPool.py`, DS = `Datastore/SQL/Datastore.py`, CP =
`Datastore/SQL/ClientPool.py`, SI = `Datastore/store_inventory.py`, SR = `Datastore/store_reader.py`,
OF = `Datastore/SQL/ObjectFactories/`. Line numbers are at `dbd8a51`.

**The inventory was taken by two read-only agents** working in a worktree at `dbd8a51`, and checked
by the author: every citation used by a prompt below was re-read, and the one finding that bears on
current stores (F0) was measured.

---

## 0. Summary

1. **A defect in current code [measured].** Four factories decide that a store predates a column by
   testing `"<column>" in str(e)`. SQLAlchemy's `str()` of a database error carries the statement,
   and the statement names the column, so **any** database error on a current store matches —
   "database is locked" included. On the compute path the error is replaced by a false "this
   datastore predates prompt N, regenerate it". On `BackgroundModel`'s read path the query is
   **silently re-run without grid identity**, and the extraction scripts report the background as
   an "unknown generation". The tests meant to catch this build the error with the statement text
   `"SELECT ..."`, which names no column (F0).
2. **Project knowledge in the generic layer [read].** Fourteen distinct schema facts are hard-coded
   in files meant to be generic (§C). Before `f4b2c91` (2026-09-28), `ShardedPool`'s only project
   literal was `"version"`. The replication and integrity prompts added the rest
   (`f4b2c91`, `b647c0b`, `20977cb`); the fingerprint prompts brought the project's factory
   registry into the reader and the inventory (`f52a7df`, `8de5a40`). Three older registries date
   from 2024–2025 (`_factories`, `_drop_actions`/`_drop_order`, `ClientPool`'s batch sizes).
3. **Old-store fallbacks [read].** About thirty sites exist only to open, read, repair or refuse a
   store written by older code (§B). None is needed by a store the current code writes. Four need
   care: one is entangled with repair logic (B2, already resolved on a side branch); "a missing
   table reads as empty" is also reached by a current store after an interrupted open or drop
   (B5); four factory refusals are F0; and `Datastore._ensure_tables` must stay (B10).
4. **No store needs the fallbacks [measured, G4].** The A3 baseline, the only store under `var/`
   with a Keep sidecar that predates them, already cannot be read at `HEAD`: its rows predate the
   key changes of `a171ced`, `3d7b606` and `b18bb0b`, whose factories refuse it. It is read at the
   SHA that wrote it.
5. **The pattern to extend already exists (§D).** `_prune_unit` derives a parent's child tables
   from foreign keys; `_replicated_prune_classes` reads `validate_on_startup` from the factories'
   registrations; each factory's `inventory_records` already declares its key, parents, tags,
   values and `validated` flag.

---

## A. A defect in current code

### F0. Old-store refusals that match on `str(e)` fire on any database error [measured]

**The probe.** [`str_e_misfire_probe.py`](datastore-generic-audit/str_e_misfire_probe.py)
([output](datastore-generic-audit/str_e_misfire_probe.txt)) creates a table holding **every**
column the factories test for, then raises two errors that have nothing to do with a missing
column: a `JOIN` on a table that does not exist, and a read while a second connection holds
`BEGIN EXCLUSIVE`. For both, each factory's test matches:

```
--- database is locked (schema is current)
    str(e) first line: (sqlite3.OperationalError) database is locked
    TkNumericIntegration / GkNumericIntegration: test matches -> True
    QCD_Cosmology: test matches -> True
    BackgroundModel (read path re-queries without grid identity): test matches -> True
```

**The sites [read].**

| Site | Test | What happens on a current store |
|---|---|---|
| OF `TkNumericIntegration.py:283-299`, `:920-930` | `"break_point_kind" in str(e)` | the error is replaced by "predates prompt 20 … must be regenerated" |
| OF `GkNumericIntegration.py:279-295`, `:919-929`, `:1068-1078` | the same | the same |
| OF `QCD_Cosmology.py:105-121` | `"T_z_representation" in str(e)` | "its QCD half must be regenerated" |
| OF `BackgroundModel.py:374-397` | `"source_grid_digest"`/`"source_grid_construction" in str(e)` | compute path (`z_sample` given): "predates prompt 14". **Read path: re-runs the query without the grid-identity columns** (`_build_query(False)`, `:397`); if that succeeds, `_source_grid_identity = None` and `extract_common.describe_background_generation` (`extract_common.py:265-277`) prints "unknown (this datastore predates prompt 14)" |
| OF `BackgroundModel.py:488-509` | `tau_lo`/`cs_tau*`/`friction_F` in `str(e)` | "predates prompts 03 and 04" |

The refusals in `TkWKBIntegration`, `GkWKBIntegration`, `GkSource` and `QuadSourceIntegral` test
`e.orig` (the DBAPI message alone) and do not misfire [read; B7].

**Why the tests miss it [read].** `ComputeTargets/tests/test_numeric_break_point_key.py:436`,
`test_cosmology_representation_key.py:521` and `test_run_identity.py:828` build their "unrelated"
error as `OperationalError("SELECT ...", {}, Exception("database is locked"))`. The real statements
name the column; `"SELECT ..."` does not.

**Impact.** No stored value is wrong. A transient lock during a build is reported as a stale store
and the build stops with advice to regenerate it; during extraction, a background is read without
the identity of the grid it was tabulated on and reported as of unknown generation. The remedy the
user has decided on — delete the old-store refusals — removes the defect with them.

---

## B. Old-store fallbacks

Every row here exists only for a store written by older code. **Deleting it changes nothing for a
store the current code writes** unless the row says otherwise. "Old store if deleted" says what a
store written by the older code would then meet; in every case it is loud.

### B1. The reconcile does not compare `timestamp` — SP:1137-1181 [read]

`_replicated_table_specs` excludes `timestamp` from every replicated table's comparison (`:1170`),
because "every store written before prompt 01 stamps it per shard" (`:1145-1146`; `f4b2c91`).
Since prompt 01 (`c08a9d0`) every replicated insert is stamped with the pool's one timestamp:
`_replicated_write` passes it to every shard, `Datastore._inserters_for` binds every table's
inserter to it, and DS:897-903 stamps it. No factory writes `timestamp` itself (no match for
`timestamp` in OF other than the `register()` flags), repair copies rows verbatim, and `version`
has no timestamp column. **Old store if deleted:** `ReplicatedDivergence`, "differs: timestamp".
**[inferred]** that no current path stamps a replicated row per shard: from the reading above. The
prompt that deletes it measures it (prompt 03, R2).

### B2. `BackgroundModelValue` compared under `(model_serial, z_serial)` — SP:1146-1148, :1165-1167 [read]

Its `serial` is not compared because "a store written before prompt 01 drew it per shard". **Not
deletable alone:** the copy-as-a-unit check assumes a child row's model is the first element of its
key (`:1681`, `:1687`, `:1706`), and a check that copied value serials are free on the target shard
(`:1714-1740`) exists only because the key is not the serial.

**Resolved on a side branch (2026-10-04).** The user asked for this exception to be deleted, and it
was, outside any campaign, on branch `drop-valuerow-key-exception`, commit **`e53f323`**: the key
exception goes, the unit check finds a row's model through its foreign key to `BackgroundModel`, and
the serial-free check is deleted (under serial keys, a row missing under serial S means the shard
holds nothing under S). Datastore suite 598 OK, as at `dbd8a51`. **Prompt 03 reviews it and takes
it into its commit** (README §6.1, D3).

*Added 2026-10-04, after the user's review:* the branch was merged into `handover-remedial` at the
user's instruction, so `e53f323` is on the branch before prompt 03 runs; 03 reviews it in place.

### B3. A store with no `replication_in_flight` table — SP:229-231, :1209-1217 [read]

The constructor creates the table on an existing store "written before the replicated write was
recorded" (`:229-231`), and `_read_in_flight_record` returns `None` when it is absent
(`:1209-1217`; `c08a9d0`, `f4b2c91`). A current primary always has it: `_write_shard_data` creates
it (`:736`) before any shard row, and a primary interrupted earlier has no `shards` rows and is
refused. **Old store if deleted:** `no such table` at open. **Tests:**
`test_reconcile_at_open.py:724`, `test_replicated_write.py:238`.

### B4. A `replication_in_flight` table with `controller_shard NOT NULL` — SP:710-715, :2125-2144 [read]

A table created before `b647c0b` cannot hold a `prune` record; `_commit_prune_record` catches the
`IntegrityError` and refuses with an explanation. Current tables declare the column nullable
(`:725`). **Old store if deleted:** a raw `IntegrityError`, before any shard is touched. **Test:**
`test_prune_at_open.py:629-645`.

### B5. Tables or columns a shard lacks — mixed [read]

| Site | Behaviour | Old-store only? |
|---|---|---|
| SP:1240-1278 `_read_replicated_tables`; SP:1751-1771 `_read_full_rows` | reads only the columns `present` on a shard; an absent table "is read as empty (an actor would create it empty)"; an absent key column marks the table `unkeyed` | **columns and `unkeyed`: yes.** **Absent table: no** — a current store reaches it when an actor dies partway through `Datastore._ensure_tables` on a first open (each `create` commits on its own), or when an open with `--drop gk-source-policy-records` (a replicated table) is interrupted between an actor's drop and its re-create (see `[00-a-drop-action-drops-a-replicated-table-outside-the-in-flight-record]`, `a3-v2-readiness`) |
| SP:1302-1313, :1429-1440 | the difference kind `columns`: shards whose tables have different column sets | yes; no test covers it |
| SP:2010-2011, :2031-2033 | the prune at open skips a shard without the class's table | no, the same two cases |
| SP:416-426, :573-593; DS:348-357, :359-382, :461-469 (`20977cb`) | a read-only pool reports tables a shard lacks and opens anyway; validation skips them | mostly; the same two cases |
| SR:25-30, :55-69, :112-150 (`f52a7df`) | the reader reports `absent_tables`, `absent_columns`, `extra_tables`, `extra_columns` | yes; all four are empty for a current store |
| SI:44-48, :70-72, :295-296, :512-602, :642-655, :967-968 (`8de5a40`) | the inventory turns an absent table or column into a named problem (`absent-table`, `incomplete`) | yes |

**This needs a decision (README §6.2, U1):** the column half is plainly old-store; the table half
is also crash recovery for current code.

### B6. Legacy absolute shard records — SP:26-31, :2291-2334, :2406-2407, :2451-2454; `Datastore/shard_paths.py`:6-14, :24-32, :85-87, :108-113, :120-122 [read]

A primary written before `b04671f` (2026-09-24) records its shards by absolute path; they are read
as siblings of the primary with a notice. `_write_shard_data` (SP:741-756) writes only bare names.
`copy_store`/`move_store` are today the only tool that turns a legacy primary's records into bare
names. **Old store if deleted:** "Shard #N … has an unusable record", from open, copy, move, delete
and the reader. **Also:** `tools/shard_key_audit.py:18-22`, `:88-107` (a pre-`a2bd966` primary),
`:146`; `docs/handover/quadsource_atol_sweep.py:671-691` (`assert_store_is_self_consistent` accepts
a legacy store). `shard_paths.py:13-14` is stale: `prepare()` now copies through
`RunRegistry.stores.copy_store`. **Keep** `_require_bare_name` and the "both resolve to one file"
check (SP:2362-2369). **Tests:** listed in prompt 04 when it is written.

### B7. Factory refusals for an older schema — `e.orig` tests, no misfire [read]

Each leaves a raw `OperationalError` for an old store if deleted, and changes nothing for a current
one.

| Factory | Sites | Accommodates |
|---|---|---|
| `BackgroundModel.py` | :347-372 | no Gauss-order columns (`90d0114`) |
| `TkWKBIntegration.py` | :20-21, :44-45, :394-422, :1343-1354, :1478-1488 | no `numeric_serial` (`a171ced`), no `rho_gauss_order` (`90d0114`) |
| `GkWKBIntegration.py` | :20-21, :48-50, :404-432, :1336-1347, :1506-1516 | the same |
| `GkSource.py` | :14-15 (stale comment), :32-34, :71-84, :447-448, :1224-1225 | no `parents_digest` / `GkSource_parents` (`3d7b606`) |
| `QuadSourceIntegral.py` | :80-101, :112-138, :452-455, :724-727 (comment), :900-903 | no `quad_policy_serial` (`b18bb0b`), no transfer-function columns (`014be95`) |

### B8. Run labels and grid identity in the extraction scripts — `extract_common.py` [read]

"No run labels" is read as a store from before prompt 14 and read without tags
(`:24-26`, `:78-83`, `:221-231`, `:257-258`; `96abdfe`). `describe_background_generation`'s
"unknown generation" branch (`:265-277`) is reached only through F0's read-path re-query. `main.py`
always writes `Run_<label>` (`DEFAULT_RUN_LABEL`, `main.py:109`, `:1184`).

### B9. Small items [read]

SP:1861 (`label` read only if present on a `BackgroundModel` table); `BackgroundModel.py:593-595`
(comment).

### B10. Not fallbacks — keep [read]

These look similar and must not be deleted:
- DS:263-268, :551-554 `_ensure_tables` on an existing store. It also re-creates the tables a drop
  action just dropped, and completes an interrupted first open.
- DS:576 (`_drop_actions` skips a table already dropped); `ProfileAgent.py:122-125` (the profile
  database is created lazily).
- Recovery of interrupted writes by the current code: the hot-journal rollback (SP:1007-1023), the
  record-driven repair (SP:1067-1129), `_complete_interrupted_prune` (SP:2191-2282), the copy and
  move interruption refusals, `delete_store(resume=True)`.
- SP:1668 (`version` has no timestamp column); the `getattr(self, "_read_only", False)` defaults
  (pools built with `object.__new__` in tests).
- `BackgroundModel.py:443` (`_source_grid_identity = None` on a miss) and the read path's
  `z_sample=None` selection: the read path legitimately supplies no grid; only the missing-column
  re-query is old-store.
- `extract_common`'s `NoBackgroundModel` skip; `T_Z_REPRESENTATION_VERSION`, a current key.

### B11. Registry records — out of scope, recorded

`RunRegistry/` has fallbacks for legacy store **sidecars** (no `sidecar_format`; `adopt_sidecar` is
their only upgrade path), run directories with no manifest, manifests without `results_store_id`
(docstrings only; no separate path) and ledger records without `script_sha256`. These are records,
not stores; the registry never deletes a record, and `python -m RunRegistry list` lists the old run
directories today. **Recommended out of scope** (README §6.2, U3).

---

## C. Project knowledge in the generic layer

The generic layer is SP, DS, CP, SI, SR, `Datastore/SQL/schema.py`, `Datastore/SQL/SerialPoolBroker.py`,
`Datastore/SQL/ProfileAgent.py`, `Datastore/object.py`, `Datastore/replication.py`,
`Datastore/shard_paths.py` and `RayTools/RayWorkPool.py`. The factories under OF are the client.
`SerialPoolBroker.py`, `object.py`, `schema.py` (but for the `version.serial` foreign key, an
infrastructure contract), `RayWorkPool.py` and `ProfileAgent.py` hold no project knowledge.

### C1. The distinct schema facts hard-coded, and where [read]

| # | Fact | Sites (executable) | Introduced |
|---|---|---|---|
| 1 | **Version table identity**: named `version`, natural key `label`, object type `store_tag`, serials not leased from the broker | SP:32 (`from MetadataConcepts import store_tag`), :311, :335, :357, :456-458, :581, :584; DS:888; schema.py:92 | DS:888 2024; SP `b647c0b`, `20977cb` |
| 2 | **Child tables that travel with a parent** | SP:1552-1561, :1679-1712 (and the positional `k[0]`, resolved in `e53f323`); DS:135-152 `_drop_actions` (the same grouping, for drops) | SP `f4b2c91`; DS `4ee5d61` (2024) |
| 3 | **A natural comparison key unlike the primary key** | SP:1165-1167 | `f4b2c91`; **deleted in `e53f323`** |
| 4 | **Child rows whose serials the controller assigns** | SP:1714-1740 (deleted in `e53f323`); SP:3227-3233, :3249, :3258-3266 (the `_values` attribute) | `f4b2c91`, `c08a9d0` |
| 5 | **Monotone flags a get may turn on** (`redshift`/`wavenumber` `source`, `response`) | SP:1562-1565, :1616-1628 | `f4b2c91` |
| 6 | **A `validated` flag with a rule recomputable from stored rows** | SP:1629-1636, :1832-1894 (`_factories["BackgroundModel"]`, a `_StoredModel` stand-in that knows the rule reads `.label`); SI:576-578, :785 (the column name) | `f4b2c91`; `8de5a40` |
| 7 | **Tag-association convention**: tag class `store_tag`, column `label`, association column `tag_serial` | SP:1156-1158; SI:606-612 | `f4b2c91`; `8de5a40` |
| 8 | **The shard key's type** — generic through `_ShardKeyType_name`, but comments and messages say "wavenumber" | SP:970, :1294, :1349, :1492, :1575, :3405 (text only) | `f4b2c91`, `20977cb` |
| 9 | **Per-table serial lease batch size** | CP:9-48, consumed at CP:166 | `5b459bd` (2024) onwards |
| 10 | **The registry of storable classes** | DS:14-76, :89-129 `_factories`; reached into as a private name by SP:14 (used :539, :1025, :1841, :2003, :2035, :2162), SR:41/:182, SI:939/:962 | DS 2024; SP `f4b2c91`; SR `f52a7df`; SI `8de5a40` |
| 11 | **Drop groups and their foreign-key order** | DS:135-170 | `4ee5d61`, `9f05f62` (2024) |
| 12 | **The inventory's class list and dependency order** | SI:106-131 | `8de5a40` |
| 13 | **The polymorphic cosmology reference** (`cosmology_type` → class), importing `CosmologyModels.model_ids` | SI:133-145, :405, :487-489, :713-714, :724 | `8de5a40` |
| 14 | **Which classes are replicated**, read from `config.sharding` instead of the store or a parameter | SI:941, :943, :985 | `8de5a40` |

Comments, docstrings and messages naming project tables, beyond the executable sites: SP:323, 970,
974-976, 1141-1142, 1146-1148, 1294, 1349, 1492-1493, 1507-1511, 1575, 1640-1642, 1677-1678,
1698-1699, 1710-1711, 1737-1738, 1834-1836, 1846, 1890-1891, 1941, 1985, 3229, 3405-3406; DS:119,
153-158, 196-201, 423-427, 884; CP:37-38; SI:11-21, 59-68, 103-105, 133-134, 142, 247-256, 268-270;
`replication.py`:200, 204; `shard_paths.py`:14.

**What is infrastructure, not a finding.** The pool's own tables (`shards`, `shard_key_config`,
`shard_keys`, `replicated_tables`, `sharded_tables`, `replication_in_flight`), the columns the
layer prepends (`serial`, `version`, `timestamp`, `stepping`), and ProfileAgent's own tables. The
`version` table and the tag table are a harder case: every storable row carries a foreign key to
`version.serial` (schema.py:92), and every tagged class associates with `store_tag`. The layer can
reasonably own those two as part of its contract, provided it builds their objects through their
factories rather than importing `MetadataConcepts` (README §6.2, U4).

### C2. Imports from project packages [read]

SP:14 (`_factories`, `f4b2c91`), SP:32 (`MetadataConcepts.store_tag`, `b647c0b`), SP:33 and
ProfileAgent.py:10 (`config.defaults.DEFAULT_STRING_LENGTH`, `a2bd966`, a generic constant), DS:14-76
(every factory), SR:41, SI:939, SI:941 (`config.sharding`), SI:143 (`CosmologyModels.model_ids`).

---

## D. What already derives per-table facts generically — the pattern to extend [read]

| Site | Derives | From |
|---|---|---|
| SP:126-133 | the shard-key class and table, the key getter, replicated and sharded membership | the constructor's arguments, which `main.py` takes from `config/sharding.py` (`11fa287`, `a2bd966`) |
| SP:1150-1160 | the tag tables to compare | foreign keys (only the `store_tag` exclusion is hard-coded) |
| SP:1169-1171 | the default comparison key | the primary key |
| SP:1970-1980 `_replicated_prune_classes` | the replicated classes pruned at open | `built.records[name]["validate_on_startup"]`, the factories' registrations |
| SP:1982-1997 `_prune_unit` | the tables a prune may delete from | the class and every compared table with a foreign key to it |
| SP:1999-2089 `_prune_needed`, `_prune_shard` | what "unvalidated" means | the factory's own `validate_on_startup` |
| SI:441-466, :973 `read_records`, `inventory_records` | each class's key, parents, tag and value tables, `validated` flag | **declared by each factory** — the closest existing model for declaring per-table facts |
| `Datastore/SQL/schema.py:30-135` `build_schema` | every table | each factory's `register()`; it reads `validate_on_startup`, `serial`, `version`, `timestamp`, `stepping` and `columns`, and no other key |

The remedy for C1 is therefore not new machinery: the facts in rows 2, 5 and 6 become keys of a
factory's `register()` (or hooks on the factory), read through `built.records`, and derived from
foreign keys where they can be — as `_prune_unit` already does for row 2.

---

## G. Measurements and checks

**G1.** The probe in F0, at `dbd8a51` with the repository's venv: output committed beside it.

**G2.** Datastore suite at `dbd8a51`: **598 OK** (428 s). On `e53f323`: **598 OK** (769 s; the
machine was running prompt 07's rehearsal build).

**G3.** `black --check` clean on the files `e53f323` changes.

**G4. The A3 baseline [measured, read-only].** `var/datastores/handover-A3-baseline-lambdacdm.sqlite`,
opened `file:…?mode=ro` with `sqlite3`, its `shards` table and table list read; SHA-256 prefix
`fdbe93a541083e8f` before and after. It records all four shards **by absolute path** (B6), and its
primary has **no `replication_in_flight` table** (B3): it predates both `b04671f` and prompt 01. Its
sidecar records it created on 2026-09-20, before the key changes of `b18bb0b` and `a171ced`
(2026-09-29), `3d7b606` and `014be95` (2026-09-30), whose factories refuse a store without their
columns [inferred from the dates and B7; its shards were not opened]. It is readable only at the SHA
that wrote it already, and remains the comparator through the fingerprint in its sidecar (taken
2026-09-25 at `50a24ac`). Deleting B3 and B6 costs it nothing further.
`quadsource_atol_sweep.py prepare()`, which copies it, is affected in the same way and already.

---

## H. Decisions this audit puts to the user

Collected in README §6.2: U1 (absent tables), U2 (the older registries — scope), U3 (registry
records — scope), U4 (the `version` and tag tables — the layer's or the client's).

*Added 2026-10-04:* the user took all four as recommended (README §6.2).
