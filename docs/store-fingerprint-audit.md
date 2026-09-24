# Store fingerprint audit — what the inventory can say about a datastore

**Date:** 2026-09-24 · **Tree:** `handover-remedial` at `218ca74`, clean · **Author:** Claude Opus
5.5, orchestrating two read-only research agents (Opus 5.5) whose reports this document
consolidates · **Status:** read-only. No file in the repository was changed to write it, and
nothing under `var/` was opened.

**Why it exists.** `run-registry`'s `[04-a-runs-product-is-named-but-never-fingerprinted]`, as
amended on 2026-09-24 (`218ca74`), decided that a store's content fingerprint is a set of digests
in its sidecar. The digests are computed by a read-only registry `store fingerprint` from a
structured inventory service, which names work items by physical labels and tag sets. This
audit establishes what that service has to read, class by class, and whether a read-only reader
with no Ray is possible. The campaign that builds it is
[`prompts/store-fingerprint/`](../prompts/store-fingerprint/README.md).

**How to read the citations.** Paths without a directory are under
`Datastore/SQL/ObjectFactories/`. **[checked]** marks a claim the orchestrator re-read in the source
while writing this document. Every other claim comes from an agent's report, with its citation, and
was not independently re-read.

---

## 1. How the inventory works today

- **The service.** Each factory has a static `inventory(conn, table, tables, ...)`.
  `Datastore.inventory` (`Datastore/SQL/Datastore.py:777-804`) calls it inside
  `self._engine.begin()`, and takes nothing from the actor except the engine and the `Table`
  objects. **[checked]**
- **The pool.** `ShardedPool.inventory` (`Datastore/SQL/ShardedPool.py:1453-1536`) behaves in one
  of two ways:
  - **Replicated class:** it asks **one shard chosen at random** and merges nothing (`:1469-1483`).
  - **Sharded class:** it fans out to every shard. It guesses the result's shape from shard 0
    alone, then merges with `_merge_queue` (`:1365-1451`), driven by `inventory_config`
    (`config/sharding.py:43-102`). Lists are concatenated, not de-duplicated, and not sorted.
    Datetimes take a min or max; numbers take a sum, min or max.
- **The three shapes** (`tools/inventory_report.py:9-23`): labelled buckets split by `validated`,
  a flat `values` list, or a flat `count`.
- **What the labels are.** A compute target's label is a string of **store-local serials**, e.g.
  `wavenumber_exit=12, model=1, atol=3, rtol=5` (`TkNumericIntegration.py:653-687`). **[checked,
  for `GkNumericIntegration.py:651-686` and `QuadSource.py:470-484`]**
- **What is only counted.** `QuadSourceIntegral`, `GkSourcePolicyData` and `OneLoopIntegral` report
  a count and a time range (`QuadSourceIntegral.py:813-830`). **[checked]** So do all seven `*Value`
  tables.
- **What is invisible.** The nine `*_tags` association tables have **no `inventory()`**, and appear
  in neither the replicated nor the sharded list, so `ShardedPool.inventory` cannot dispatch them.
  `tools/inventory_report.py:34-38` leaves them out deliberately. **Nothing today says which tags a
  product carries.**
- **Consumers.** There are two:
  - `tools/inventory_report.py:271`, reached from `main.py --inventory` (`main.py:3501-3504`)
    **[checked]**;
  - `extract_common.available_run_labels(pool)` (`extract_common.py:105-121`), which reads
    `pool.inventory("store_tag")["values"]` and is used by every `extract_*.py` **[checked]**.
- **Tests.** The only test of `inventory()` is
  `ComputeTargets/tests/test_qcd_cosmology_inventory.py`. It has three methods, each calling one
  factory directly. Nothing tests `ShardedPool.inventory`, `_merge_queue`, `Datastore.inventory`
  or the report. `backport-modules` logs 06–09 used scratch harnesses that were never committed.
- **Constraints recorded in `backport-modules` logs 06–09:**
  - log 06: the shape is guessed from shard 0, so labelled factories must return the same keys
    from an empty shard;
  - log 07: a shape whose meaning depends on an argument (`wavenumber`'s `units`) was rejected;
  - log 08: labels were deliberately left as unresolved serials, to avoid one query per row.

## 2. What opening a store writes

- **The pool's own open.** `ShardedPool`'s existing-store branch (`ShardedPool.py:143-165`) only
  builds `Table` metadata, runs `SELECT`s on the primary, and checks files. The primary is written
  later, by `_assign_shard_keys` (`:1261-1324`), when a new `wavenumber` is first seen.
- **Each `Datastore` actor, for an existing file** (`Datastore.py:223-244`), in this order:
  1. `_drop_actions`: `DROP TABLE` for the `--drop` groups (`:415-445`).
  2. `_ensure_tables`: `CREATE TABLE` for any registered table that is missing (`:404-407`).
  3. `_validate_on_startup`: when `prune_unvalidated` is set, it **`DELETE`s unvalidated rows** and
     their values and tags (`:447-472`). Every delete in the seven `validate_on_startup`
     implementations is inside `if prune:`.
  4. The `version` row, inserted **only when its label is new** (`version.py:21-41`). The label is
     the constant `"2025.1.1"` (`main.py:281`), so the row is written **once per store per
     release, not once per open**.
- **A correction to the record.** The amendment of `218ca74` says "opening a store through the
  `ShardedPool` constructor inserts a `version` row into each shard". That is true only the first
  time, per release. Its conclusion still holds, because steps 1–3 write. The correction is
  recorded on the `run-registry` board beside the amendment.
- **Journals.** A read-write open of a file with a hot rollback journal replays the journal, which
  is itself a write.
- **`main.py --inventory` is not read-only, and by default it deletes data.** It builds the full
  pool before reaching the `--inventory` branch (`main.py:3484-3504`). `--prune-unvalidated` is an
  `argparse.BooleanOptionalAction` with **`default=True`** (`main.py:233-238`) **[checked]**. So
  every inventory run that does not pass `--no-prune-unvalidated` deletes the store's unvalidated
  rows, and inventory runs also need Ray. Every demonstration in `datastore-portability` passed
  the flag. **This is not recorded anywhere; it is opened as
  `[00-inventory-run-prunes-unvalidated-rows-by-default]`.**

## 3. A read-only reader with no Ray is possible

- **Where tables live.** The primary holds only the five configuration tables (`shards`,
  `shard_key_config`, `shard_keys`, `replicated_tables`, `sharded_tables`; `ShardedPool.py:271-311`).
  Every factory table exists in **every shard**:
  - replicated rows are copied into each shard (`:959-1038`, `:1146-1169`);
  - a sharded row lives in one shard;
  - its `*_tags` rows are written in the same transaction as the row itself (e.g.
    `TkNumericIntegration.py:485-487`), so they sit in the same shard.

  So every join a physical key needs, from a work item to its replicated parents and tags, **stays
  inside one shard file**.
- **The schema can be built without writing.** `Datastore._build_schema` (`Datastore.py:300-402`)
  builds each `sqla.Table` from `factory.register()` into a `MetaData` object. It prepends the
  `serial`, `version` foreign key, `timestamp` and `stepping` columns. Creating the tables is a
  separate step (`_ensure_tables`). `register()` returns fresh `Column` objects on every call.
- **An end-to-end check.** One agent re-implemented `_build_schema` over `_factories` in the
  session scratchpad and opened a throwaway store with `sqlite:///file:{db}?mode=ro&uri=true`. All
  28 `inventory()` calls ran inside one `engine.begin()`. The file's SHA-256 and mtime were
  unchanged, and an `INSERT` raised "attempt to write a readonly database". The store was a
  scratch file, not a real store; the campaign repeats this check on a copy of a real one.
- **No shared schema builder exists.** `_build_schema` is an actor method, and its logic has been
  copied by hand into four test helpers:
  - `Datastore/tests/test_backgroundmodelvalue_roundtrip.py:55-72`;
  - `ComputeTargets/tests/test_numeric_break_point_key.py:115`;
  - `ComputeTargets/tests/test_run_identity.py:117`;
  - `ComputeTargets/tests/test_qcd_cosmology_inventory.py:30-44`.
- **What can be reused.** `ShardedPool._read_closed_store(primary, verb)` (`:667-704`) reads
  `shards` with `mode=ro`, then resolves and checks every shard; it is static and needs no Ray.
  `Datastore/shard_paths.py` uses only the standard library.
- **Imports.** Importing `_factories` imports `ray` transitively but never initialises it. The
  registry's promise that `import RunRegistry` loads neither `ray` nor `sqlalchemy`
  (`RunRegistry/stores.py:45-46`) therefore needs the reader to be imported inside the operation,
  as `copy_store` already imports `ShardedPool`.
- **Stores older than the code.** A read-write open would have created any missing tables; a
  read-only reader cannot. It must treat a missing table as empty, and detect a missing column
  itself (`PRAGMA table_info`).
- **A latent bug in the code to be factored.** `_build_schema` calls
  `registration_data.get(...)` (`:314`) before its `is not None` check (`:320`). No factory returns
  `None` today. **[checked]**

## 4. What identifies each class

- **Replicated or sharded** is set in `config/sharding.py:3-35` **[checked]**. The shard key is the
  `wavenumber`: `QuadSource` and `QuadSourceValue` shard by `q`, everything else by `k`.
- **How tags filter.** A tagged lookup joins the tag table once per requested tag (e.g.
  `QuadSourceIntegral.py:279-291`) **[checked]**. A row therefore matches if it carries **at least**
  the requested tags. The fingerprint must record each row's **full** tag set, read from its
  association table.

In the table, `kx` is `wavenumber_exit_time`, `M` is `BackgroundModel`, `tol` is
`tolerance.log10_tol` and `z(·)` is a `redshift` row. "Identity" means the columns the factory's
lookup filters on.

| Class | Kind | Where | Identity → physical leaves | validated | Tags |
|---|---|---|---|---|---|
| `version` | config | repl | `label` | – | – |
| `store_tag` | config | repl | `label` | – | – |
| `redshift` | grid | repl | `z` (REAL; lookup relative 1e-7) | – | – |
| `wavenumber` | grid | repl | `k_inv_Mpc` (REAL; absolute 1e-7) | – | – |
| `wavenumber_exit_time` | grid | repl | k; cosmology (`cosmology_type` + serial, no foreign key); atol, rtol → tol; `stepping`. The lookup matches `<=` / `>=` best-first; `store()` writes exact values (`wavenumber.py:222-255`, `:382-394`, `:460-468`) | – | – |
| `tolerance` | config | repl | `log10_tol` (absolute 1e-7) | – | – |
| `LambdaCDM` | config | repl | omega_m, omega_cc, h, f_baryon, T_CMB_Kelvin, Neff | – | – |
| `QCD_Cosmology` | config | repl | the same six, plus `log10_max_z` and `T_z_representation` | – | – |
| `IntegrationSolver` | config | repl | `label`; lookup `stepping >=` | – | – |
| `GkSourcePolicy`, `QuadSourcePolicy` | config | repl | Levin_threshold, numeric_policy | – | – |
| `BackgroundModel` | target | repl | cosmology; three Gauss orders; `source_grid_digest`, `source_grid_construction`; optional z_init (`BackgroundModel.py:297-323`) | yes | yes |
| `TkNumericIntegration` | target | k | kx, M, atol, rtol, `break_point_kind`; optional z_init (`:242-261`) | yes | yes |
| `TkWKBIntegration` | target | k | kx, M, `rho_gauss_order`; optional `z_init` (a REAL column) | yes | yes |
| `GkNumericIntegration` | target | k | kx, M, atol, rtol, `break_point_kind`; optional z_source | yes | yes |
| `GkWKBIntegration` | target | k | kx, M, `rho_gauss_order`; optional z_source, `z_init` (REAL) | yes | yes |
| `GkSource` | target | k | kx, M; optional z_response | yes | yes |
| `GkSourcePolicyData` | target | k | parent `GkSource`, `GkSourcePolicy` (`GkSourcePolicyData.py:96-100`) | no | no |
| `QuadSource` | target | q | M, kx(q), kx(r) | yes | yes |
| `QuadSourceIntegral` | target | k | M, `GkSourcePolicy` (**not** `QuadSourcePolicy`), kx(k), kx(q), kx(r), z_response, z_source_max, atol, rtol (`QuadSourceIntegral.py:267-277`) **[checked]** | **no** | yes |
| `OneLoopIntegral` | target | k | M, kx, z_response, atol, rtol | no | yes (see §6) |
| `*Value` (7 tables) | value | parent's | (parent serial, z serial) | – | – |

**Notes on the table:**
- **Optional lookup filters.** `z_init`, `z_source` and `z_response` are filtered only when the
  caller supplies them. The fingerprint always includes them.
- **Identity from code constants.** The Gauss orders, `RHO_GAUSS_ORDER`, `BREAK_POINT_KIND` and
  `T_Z_REPRESENTATION_VERSION` are module constants in the lookup, but each is also a stored
  column. The fingerprint reads them from the row.
- **Not identity:**
  - payload and provenance columns: solver serials, `z_min_serial`, `z_samples`, `Tq_serial` /
    `Tr_serial`, `source_serial` / `data_serial`, results, timings, `metadata`, the `version` foreign
    key and `timestamp`;
  - free-text compute-target `label`s, which are `{job_name}-…-{datetime.now()}`
    (e.g. `main.py:1324`);
  - cosmology `name` and policy `label`;
  - the `source` / `response` flags on `wavenumber` and `redshift`, which accumulate by OR.
- **`IntegrationSolver.label` is the exception**: it *is* that class's identity.
- **Serials are not deterministic.** They are leased in batches, popped from a set and recycled
  (`ClientPool.py:9-45`, `SerialPoolBroker.py:46-50`). The controlling shard for a replicated
  write is random, and shards are assigned by minimum load. **Two stores built independently need
  not share a single serial**, so only physical labels can compare them.
- **`store_tag` can hold tags no product carries**, because tags are created on `object_get`. Tag
  sets must come from the association rows, not from `store_tag`.

## 5. Floats

Every physical leaf is a REAL, and the lookups match it within a tolerance: `k` absolute 1e-7, `z`
relative 1e-7, `log10_tol` absolute 1e-7, and cosmology parameters and `Levin_threshold`
absolute 1e-7. A digest has to be exact, so the fingerprint needs a canonical form. The two
candidates:

- **The stored bits** (`float.hex`). A copy always matches its original, and two stores match
  only if they hold identical values. Two independent builds on different hardware could differ in
  the last bit of a grid value (compare `run-registry`'s `[04-runs-do-not-say-which-machine-produced-them]`)
  and would then fingerprint differently, although their lookups would match.
- **A rounded form**, e.g. about 12 significant figures, much finer than the lookup tolerances and
  much coarser than last-bit noise. This absorbs last-bit differences, but a value lying close to a
  rounding boundary can still split two identical stores, and "the same data" becomes "data equal
  to 12 figures".

The choice is recorded as decision **D1** in the campaign README.

## 6. Defects found

Each is opened on the campaign board's §3, with an index row. The campaign fixes only those it is
assigned.

- **`[00-inventory-run-prunes-unvalidated-rows-by-default]`**: §2. The campaign fixes it by moving
  `--inventory` onto the read-only reader (prompt 03).
- **`[00-oneloop-lookup-joins-the-wrong-tag-table]`**: `OneLoopIntegral.build` filters tags
  against `tables["QuadSourceIntegral_tags"]` (`OneLoopIntegral.py:141`), while `store()` writes
  `OneLoopIntegral_tags` (`:243`) **[checked]**. A tagged lookup joins QSI tag rows against
  `OneLoopIntegral` serials. The fingerprint reads `OneLoopIntegral_tags`, which is what is stored.
- **`[00-tagged-read-batch-joins-an-unselected-alias]`** *(suspected, not run)*: the tag joins in
  `read_batch` use `query.c.serial` (`QuadSourceIntegral.py:679` **[checked]**;
  `OneLoopIntegral.py:332`). Compiled under SQLAlchemy 2.0.39, the pattern gives a deprecation
  warning and SQL that names `anon_1.serial`, which is not in the FROM clause.
  `extract_QuadSourceIntegral_data.py:1087-1099` passes non-empty tags. The query has not been
  run against a store.
- **`[00-quadsource-tq-serial-has-the-wrong-foreign-key]`**: `Tq_serial` declares
  `ForeignKey("QuadSource.serial")` (`QuadSource.py:113-116`), but it holds a transfer-function
  store id (`ComputeTargets/QuadSource.py:473`) **[checked]**. `Tr_serial` has no foreign key.
- **`[00-numeric-value-parent-lookup-omits-break-point-kind]`**: the parent query in
  `TkNumericValue`'s build (`TkNumericIntegration.py:873-878`) **[checked]**, and its
  `GkNumeric` counterpart, filter on model, wavenumber exit, `z_init` / `z_source`, validated and the
  tolerances, but not on `break_point_kind`, which is in the parent's own lookup.
- **`[00-quadsourcepolicy-rows-are-referenced-by-nothing]`**: `QuadSourcePolicy` rows are created
  (`main.py:3651-3657`), but no factory table references the class **[checked, by grep]**.
  `QuadSourceIntegral` keys on `GkSourcePolicy`.
- **`[00-replicated-writes-can-diverge-across-shards]`**: a replicated write commits on the
  controlling shard first and on the others afterwards (`ShardedPool.py:975-1030`, `:1158-1167`).
  A crash in between leaves the copies different for good. Each shard also stamps its own
  `timestamp` (`Datastore.py:665`). The inventory's random choice of shard hides this; the
  fingerprint reads every shard and reports a divergence.
- **`[00-build-schema-reads-registration-before-its-none-check]`**: §3. It is latent, and it is
  fixed where the code is factored out (prompt 01).

An existing issue closes as a side effect. `qcd-background-audit`'s
`[03-qcd-inventory-does-not-report-the-representation]` has its `BackgroundModel` half open: the
report omits `source_grid_digest` and `source_grid_construction`. The structured record carries
both (prompt 02), and the display renders them (prompt 03).

## 7. Concurrency

- **Journal mode.** The stores use SQLite's default rollback journal; no PRAGMA is set anywhere.
  pysqlite issues no `BEGIN` before a `SELECT`, so each statement sees its own snapshot.
- **No cross-shard consistency.** Shards are separate databases with no shared transaction. A
  reader of a store that a pipeline is writing sees no consistent state across shards, and can see
  an inconsistent one between two statements on one shard.
- **Holding a snapshot costs the writer.** An explicit per-shard `BEGIN` would give a consistent
  snapshot, but its SHARED lock blocks the writer's commit, which then waits `--db-timeout`
  (60 s by default).
- **Crashed stores.** `mode=ro` cannot roll back a hot journal, so a crashed store fails to open
  with `SQLITE_READONLY_ROLLBACK`. That is the right answer: the reader must say so, not repair it.
- **Consequence.** A fingerprint is taken of a **closed** store. The registry refuses one when a
  `running` run names the store, using the check `store copy` / `move` already make.

## 8. What this means for the design

1. **One schema builder**, used by the actor and by the reader, with the latent bug fixed while
   moving it.
2. **A read-only reader** over a closed store's shards (`mode=ro`), with no actors and no Ray. It
   treats missing tables as empty, reports missing columns, and refuses a hot journal by name.
3. **A structured inventory** per class, built on that reader:
   - work items keyed by physical leaves, with parents referenced by their own canonical key and
     never by serial;
   - each item's full tag set, as sorted labels from its association table;
   - its `validated` flag where the class has one;
   - its per-parent `*Value` count;
   - replicated classes read from every shard and compared;
   - a record shape with room for computed values later, outside the key.
4. **The consumers move onto it.** The report (`main.py --inventory`, which becomes read-only and
   needs no Ray) and `available_run_labels`. Then the old three shapes, `ShardedPool.inventory`,
   `_merge_queue` and `inventory_config` retire, so there is one inventory service.
5. **The fingerprint** is a pure function of the structured inventory:
   - a format version;
   - per class and per tag set, a count and a digest over the sorted canonical records;
   - one overall digest;
   - no timestamps.

   The registry writes it to the sidecar and the run record, as the amendment decided.
