# Campaign — datastore-generic

**Written:** 2026-10-04 on branch `datastore-generic-plan`, from `handover-remedial` at `dbd8a51`
(the audit's tree), committed on `a6c8f39`, which changes no file the audit cites, by Claude Opus 5.5, from the audit [`docs/datastore-generic-audit.md`](../../docs/datastore-generic-audit.md)
and the user's decisions in §6.1. **Draft for the user's review.** **Prompts 01–09 are
written**; their tasks are fixed by §6.1 and the audit alone (04 was written on 2026-10-04, after
03 landed and was reviewed; 05 and 06 on 2026-10-05, and 07, 08 and 09 on 2026-10-06, each after
the prompt before it landed and was reviewed).

**2026-10-04, after the user's review:** the user took **U1–U4 as recommended** (§6.2). 05 is no
longer held on a decision, only on 03–04 landing; 06 and 07 wait only on the prompts before them.
The branches `datastore-generic-plan` and `drop-valuerow-key-exception` were merged into
`handover-remedial` at the user's instruction, so **`e53f323` is on `handover-remedial`** before
any prompt has run: prompt 03 reviews it there rather than cherry-picking it (§6.1 D3, amended).

## 0. Why this campaign exists

On 2026-10-04 the user, auditing recent changes to `ShardedPool` and `Datastore`, found that
`ShardedPool` names `BackgroundModel` and `wavenumber`. It is meant to be a generic component, and
the user had made it one in December 2025 (`11fa287`, `a2bd966`, `29cb2a3`). The replication and
integrity prompts of September–October 2026 re-coupled it: their prompts described the behaviour
they wanted table by table, and the implementations wrote those tables into the pool (audit §C).
The user asked how it happened and decided two things (§6.1):

1. **The datastore layer is generic again.** It names none of the project's tables; the facts it
   needs about them are declared by the factories or in `config/`, and derived generically where
   they can be (audit §D).
2. **Every fallback for a store written by older code is deleted.** Stores are regenerated, not
   accommodated. About thirty such sites exist (audit §B), and they are maintenance carried for
   stores nobody keeps.

The audit found, beside these, **a defect in current code** (audit F0): four factories' old-store
refusals match a column name against `str(e)`, which contains the statement, so any database error
on a current store — a lock included — is reported as a stale store, and `BackgroundModel`'s read
path silently re-queries without grid identity. Deleting those refusals removes it.

**0.1 Correctness is the only objective.** Work is sequenced by epistemic dependency:
1. **The defect first** (01). It touches current stores; the rest is maintenance.
2. **Delete the fallbacks before making the layer generic** (01–05 before 06–08). Every fallback
   deleted is code the refactor does not have to preserve, and several fallbacks sit in exactly the
   functions the refactor rewrites (audit B2, B5).
3. **Inject the registry before declaring facts in it** (06 before 07–08): a fact declared in a
   factory's `register()` is read through the registry the layer is given, not the one it imports.
4. **Last, a guard that the layer stays generic** (09), so that this does not recur.

## 1. Scope

**In scope:**
- `Datastore/` — `SQL/ShardedPool.py`, `SQL/Datastore.py`, `SQL/ClientPool.py`, `SQL/schema.py`,
  `shard_paths.py`, `store_reader.py`, `store_inventory.py`, `replication.py`;
- `Datastore/SQL/ObjectFactories/*` — their old-store refusals (01, 02), and the `register()` keys
  or hooks that 07 adds;
- `extract_common.py` (01), `tools/shard_key_audit.py` (04), the docstring of
  `docs/handover/quadsource_atol_sweep.py`'s `assert_store_is_self_consistent` (04);
- `config/` — where the registries the layer stops owning are moved (U2; 06: `config/datastore.py`);
- the plumbing that passes them in (06): `main.py`'s `--drop` resolution, its `ShardedPool` call and
  `--inventory`; the six `extract_*.py` scripts' `ShardedPool` calls; `extract_common`'s run labels;
  `RunRegistry/stores.py`'s fingerprint; `tools/inventory_report.py`'s schema. Their behaviour does
  not change;
- the tests of all of the above, under §5 rule 7.

**Out of scope:**
- **any change to what the code writes**: no table, column, key, constraint or stored value
  changes (§5 rule 9). A store written at `dbd8a51` opens, reads and fingerprints the same after
  every prompt;
- `RunRegistry/`'s fallbacks for legacy sidecars, manifests and ledger records (audit B11), unless
  U3 says otherwise;
- `main.py`'s and the drivers' behaviour (06 changes only how they pass the registries in), and
  every compute target;
- every store under `var/` (§3);
- `a3-v2-readiness`'s open issues, including
  `[00-a-drop-action-drops-a-replicated-table-outside-the-in-flight-record]`, which U1 bears on
  but does not fix.

## 2. Prompts

| # | Prompt | Covers | Status |
|---|---|---|---|
| 01 | [`01-delete-the-old-store-refusals-that-misfire.md`](01-delete-the-old-store-refusals-that-misfire.md) | Audit F0, B8, B9's comment. The old-store refusals and the read-path re-query in `TkNumericIntegration`, `GkNumericIntegration`, `QCD_Cosmology` and `BackgroundModel`; `extract_common`'s "no run labels" and "unknown generation" branches; their tests. A test, with real statement text, that an unrelated database error reaches the caller unchanged. Closes `[00-old-store-refusals-misfire-on-any-database-error]`. | **written** 2026-10-04 |
| 02 | [`02-delete-the-remaining-factory-refusals.md`](02-delete-the-remaining-factory-refusals.md) | Audit B7. The old-store refusals in `TkWKBIntegration`, `GkWKBIntegration`, `GkSource` and `QuadSourceIntegral`, and their tests. | **written** 2026-10-04; after 01 |
| 03 | [`03-delete-the-reconciles-old-store-paths.md`](03-delete-the-reconciles-old-store-paths.md) | Audit B1–B4, B9. Reviews and takes `e53f323` (the `BackgroundModelValue` key, D3); compares `timestamp` (D4); deletes the missing-`replication_in_flight` path and the `controller_shard NOT NULL` refusal. | **written** 2026-10-04; independent of 01–02 |
| 04 | [`04-delete-the-legacy-shard-records.md`](04-delete-the-legacy-shard-records.md) | Audit B6. Absolute shard records in `shard_paths.py` and `ShardedPool`; `tools/shard_key_audit.py`'s pre-`a2bd966` and missing-`shard_key_config` paths, and its OK verdict on an unusable record; the sweep script's check and docstring; `write_legacy_primary` renamed for the malformed-record tests that use it; the legacy tests deleted, and an absolute record pinned as refused by every reader. | **written** 2026-10-04; after 03 |
| 05 | `05-refuse-a-store-whose-schema-differs.md` | Audit B5. The reader, the inventory and the read-only pool stop tolerating absent or extra tables and columns; the reconcile's column-subset and `unkeyed` logic and the `columns` difference go. An **absent table** on a read-write open per U1. | **written** 2026-10-05; after 04 |
| 06 | [`06-give-the-layer-its-registry.md`](06-give-the-layer-its-registry.md) | Audit C1 rows 9, 10, 11, 14 and the `config.sharding` import. `ShardedPool`, the actor, `store_reader` and `store_inventory` are given the factory registry instead of importing `Datastore._factories`; the inventory reads the replicated set from the store's primary instead of `config.sharding` (*amended 2026-10-05*, the user's decision, §6.2). `_factories`, the drop groups and `ClientPool`'s batch sizes move to `config/datastore.py` (U2); the layer is given the tables to drop, and `_drop_order` is replaced by an order derived from foreign keys. | **written** 2026-10-05; after 05 |
| 07 | [`07-declare-the-replication-facts-in-the-factories.md`](07-declare-the-replication-facts-in-the-factories.md) | Audit C1 rows 1, 2, 4–8. The copy-as-a-unit rule derived from foreign keys (as `_prune_unit` is; *amended 2026-10-06*, the user's decision, §6.2: through the foreign key an owned table declares); monotone flags and the `validated` rule declared by the factories; the version and tag tables per U4; every project name out of `ShardedPool`'s code, comments and messages. | **written** 2026-10-06; after 06 |
| 08 | [`08-derive-the-inventorys-class-list.md`](08-derive-the-inventorys-class-list.md) | Audit C1 rows 12, 13. The inventory's class list and order derived from the registry and foreign keys (*amended 2026-10-06*, the user's decision, §6.2: from the parents each factory declares in an `inventory_spec`, roots first, since foreign keys miss nine of them); the polymorphic cosmology reference declared by the client (the user, 2026-10-06: in the referencing `Parent`). | **written** 2026-10-06; after 07 |
| 09 | [`09-keep-the-layer-generic.md`](09-keep-the-layer-generic.md) | A test that the generic layer (audit §C) names no table of the registry and imports no client package, outside a stated allow-list; every suite; the closing verification of §5 rule 9 (*2026-10-06*, the user's choices, §6.2: `config.defaults` and `utilities` allow-listed; 07's and 08's per-file name tests folded into the guard; rule 9 checked on fixtures in the prompt and by one rehearsal rebuild at the review, §3). Closes the campaign. | **written** 2026-10-06; after 08 |

**Why 04 was written after 03.** Its task was fixed: the audit's G4 found the only store that
would meet it, the A3 baseline, already unreadable at `HEAD` for other reasons. It edits
`ShardedPool` after 03, so it was written once 03 had landed and been reviewed, against the tree 03
left (`4ca36c4`). Writing it found that the A3 baseline could not be retired at `HEAD` (its fingerprint
formats differ), and that after 04 it could not be copied or moved there either. The user retired
it the same day at `81c0a9a` (§3), so 04 costs no store anything.

**Why 05 was written after 04.** It edits `ShardedPool` after 04, and its test plan was measured
against the tree 04 left (`bc3e944`): a rough version of the change was made in a scratch worktree
and every suite run, so that each test it moves is named with its fate before dispatch.

**Why 06 was written after 05.** It edits `ShardedPool`, the reader and the inventory after 05, and
its test plan and facts were measured the same way, against the tree 05 left (`dbc59f9`). Writing
it put two choices of method to the user, recorded in §6.2 under U2.

**Why 07 was written after 06.** It edits `ShardedPool`, the actor, `schema.py` and
`read_records` after 06, and declares facts through the registry that 06 made the layer's input.
Its test plan and facts were measured the same way, against the tree 06 left (`c6bddf8`). The
measurement also recorded every repair, refusal and printed line of the check at open, so that the
prompt could require them unchanged. Writing it put one choice of method to the user, recorded in
§6.2.

**Why 08 was written after 07.** It rewrites `store_inventory.py` after 07 had changed its tag
and validated reads, and declares facts in the factories as 07 did. Its test plan and facts were
measured the same way, against the tree 07 left (`5fe0ae6`); the measurement also compared the
fingerprint, the whole inventory, the report and every `resolve()` at both trees. Writing it found
that foreign keys cannot order the inventory, and put three choices of method to the user,
recorded in §6.2.

**Why 09 was written after 08.** Its guard checks the layer 01–08 left, so what it would find
could not be known before 08 landed. Its test plan was measured the same way, against the tree 08
left (`3ff1c80`): the guard found ten hits at three sites, none of them project knowledge. The
measurement also checked rule 9 on fixtures, from a store written at `dbd8a51`. Writing it put
four choices of method to the user, recorded in §6.2. One of them adds the exception in §3.

**Order.**
- 01 → 02: both edit factory refusals and their test modules overlap.
- 03 is independent of 01–02, but never in one checkout concurrently with another prompt that
  edits `ShardedPool.py`.
- 03 → 04 → 05: each edits `ShardedPool.py`.
- 06 after 01–05; 07 after 06; 08 after 07 (both edit `store_inventory.py` and the factories);
  09 last.

## 3. Datastores

**No prompt opens, writes, copies or retires a store under `var/`.** Tests build stores in temporary
directories, with the stand-in pool (`Datastore/tests/standin_pool.py`) or the real-store fixtures.
The A3 baseline (`var/datastores/handover-A3-baseline-lambdacdm`) was readable only at the SHA
that wrote it (audit G4). It was retired by the user on 2026-10-04 (completed 21:31:59) with `store retire` run at `81c0a9a`, the tree whose fingerprint
format matched its sidecar's. Its sidecar is the tombstone. No store under `var/` now holds an
old-store shape.

**The one exception (the user, 2026-10-06, for 09's closing verification; §6.2).** The
orchestrator's review of 09 runs one rehearsal build:
- the same command as a3-v2-readiness-07's reference, which ran at `dbd8a51`;
- through the run registry, detached;
- into a **new** store, `var/datastores/datastore-generic-09-rehearsal.sqlite`.

Its fingerprint is compared with the reference's, `39809dca…`. It opens no existing store, and
the user retires the new one afterwards (`store retire` is run by a person). Prompts open nothing
under `var/`, as before.

**Before touching anything,** run `python -m RunRegistry list` and report anything `running`.

## 4. The interfaces between prompts

The **names** below are fixed so that each prompt can be written against the one before it. The
internals are the implementing prompt's to design.

- **After 01–02**, no factory catches a `SQLAlchemyError` in order to name an older schema. A store
  written before a column existed meets SQLAlchemy's own `OperationalError`.
- **After 03**, `_replicated_table_specs` compares every column of every replicated table under its
  primary key; `ShardedPool` assumes `replication_in_flight` exists on every primary it opens.
- **After 04**, a `shards` record is a bare file name, or every reader refuses it as unusable;
  `Datastore.shard_paths` has no `is_legacy_record`.
- **After 05**, a store whose tables or columns differ from the registry's is refused by name before
  anything reads it, in one place: `schema_differences` and `StoreSchemaMismatch(RuntimeError)`,
  both in `Datastore/SQL/schema.py`. A read-write open still reads an absent table as empty (U1);
  both constructors also refuse a primary that differs from `_create_engine`'s tables.
- **06 ships** the constructor parameter by which `ShardedPool` receives the registry, and the
  parameter by which `store_reader.open_read_only` and `store_inventory.read_inventory` receive it.
  06 names them; 07–09 use them. *Named 2026-10-05 (prompt 06):*
  - `ShardedPool(…, *, factories, serial_batch_sizes=None)`, and `drop_tables` in place of
    `drop_actions`; the `Datastore` actor takes the same three; `ShardedPool.factories` returns
    the registry;
  - `open_read_only(primary, factories)` and `read_inventory(primary, factories)`;
    `ReadOnlyStore.replicated_tables` is the store's recorded replicated set;
  - `Datastore.SQL.schema.drop_order(tables)`;
  - `config/datastore.py`: `factories`, `drop_groups`, `serial_batch_sizes`, `tables_to_drop`.
- **07 ships** the `register()` keys (or factory hooks) for the facts of audit C1 rows 5 and 6. 07
  names them; `schema.build_schema` carries them into `built.records`. *Named 2026-10-06 (prompt
  07, written); shipped 2026-10-06 (prompt 07, `3f51939`), as named:*
  - the `register()` keys `owner_column`, `monotone_flags` and `validated_column`, carried into
    every record of a class with a table and checked against it (a misfit raises `ValueError`
    naming the class and the key); declared today by `BackgroundModelValue` (`owner_column`),
    `redshift` and `wavenumber` (`monotone_flags`) and `BackgroundModel` (`validated_column`);
  - the factory hooks `revalidate(serial, conn, table, tables)` and `owned_serials(obj)`, on
    `SQLAFactoryBase` and not abstract, implemented by `BackgroundModel`'s factory;
  - `Datastore/contract.py`: `VERSION_TABLE`, `VERSION_LABEL`, `TAG_TABLE`, `TAG_LABEL` and
    `TAG_SERIAL`, the tables and columns the layer owns (U4);
  - `ShardedPool._unit_tables(cls_name, specs)`, the one unit of the repair licence and the prune;
  - `store_inventory.read_records(…, validated=<column name or None>)`, where it took a bool.
- **08 ships** the declarations from which the inventory derives its classes and their order, and
  the client's declaration of the polymorphic reference. *Named 2026-10-06 (prompt 08, written);
  shipped 2026-10-06 (prompt 08, `6646b06`), as named; the last bullet, added at landing, gives
  its refusals and where the order is derived:*
  - `SQLAFactoryBase.inventory_spec()`, static and not abstract, returning `None` or a
    `store_inventory.InventorySpec` (`read_records`' six keywords; `keywords()`,
    `dependencies()`); `inventory_records` is gone from every factory;
  - `store_inventory.inventory_specs(factories)` and `inventory_classes(factories)`, the classes
    of the inventory in their derived order; `INVENTORY_CLASSES`, `COMPUTE_TARGETS`, `COSMOLOGY`
    and `cosmology_classes` are gone;
  - `Parent(column, type_column=…, types={type value: class})`, a polymorphic reference, with
    `classes()` and `class_of(mapping)`; `ClassInventory.parent_types` and
    `ClassInventory.parent_class(field, key)`;
  - `Datastore/SQL/ObjectFactories/cosmology_types.py`: `COSMOLOGY_TYPES`, the client's map.
  - `InventorySpec.dependencies()` includes the tag table for a tagged class; `inventory_classes`
    refuses a dependency that declares no spec, a polymorphic type column that is not a leaf, a
    self-reference and a cycle, each `ValueError` naming the classes; `ParentSet` refuses a
    polymorphic member; `read_inventory` derives the order after `open_read_only` has checked the
    registry against the store.
- **09 ships** the guard that keeps the layer generic. *Named 2026-10-06 (prompt 09, written):*
  - `Datastore/tests/test_layer_is_generic.py`: `layer_files()`, the layer's modules, derived (every
    module under `Datastore/` but its tests and the client's factories, with
    `ObjectFactories/base.py`, and `RayTools/`); the forbidden vocabulary (the registry's tables,
    class names and identifier columns, less the layer's own words; the project's packages; 07's
    flag phrases and 08's cosmology names and fragments); `import_allowed`, the allow-list (the
    standard library, `ray`, `sqlalchemy`, the layer's own modules, `config.defaults` and
    `utilities`);
  - 07's `TestNoProjectNameInThePool` and 08's two per-file name and import tests are gone, folded
    into it.

## 5. The rules this campaign runs under

The project-wide ones in `CLAUDE.md`, plus:

1. **One commit per prompt.** The commit boundary is the rollback boundary.
2. **Every prompt writes a log** to `logs/NN-<name>.md`, using §5.1, classifying every deviation as
   `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.
3. **Every prompt updates `IMPLEMENTATION_STATE.md` in its own commit**, plus `docs/OPEN_ISSUES.md`
   (§1.18), plus the owning board of any issue it closes.
4. **Do not fix things the prompt did not ask for.** Record them, and open a §3 issue. In particular,
   a fallback the audit missed is recorded, not deleted, unless the prompt's charter covers it.
5. **Commit messages** in `CLAUDE.md`'s form, ending with `Co-Authored-By:` naming the model.
6. **Verification documents are additive.**
7. **Existing tests are not modified**, except where a prompt names the test and the reason. A test
   that exists only to exercise a deleted fallback is **deleted**, and the log lists each by module,
   class and method. A test that uses an old-store fixture only as a convenient way to cause some
   other problem is **re-fixtured**, not deleted, and the log says which.
8. **Deliberate breakage.** Each prompt names mutations its tests must catch, and the log records
   each as a diff, exactly as applied, so the orchestrator can replay it with `git apply`.
   Mutations are never committed.
9. **Nothing the code writes changes.** No table, column, key, constraint or stored value. A prompt
   that finds it cannot do its task without such a change stops and asks.
10. **No test needs Ray, and no test opens anything under `var/`.**
11. **The generic layer gains no project knowledge.** From 01 on, no prompt adds a project table,
    column or package name to a file of the generic layer (audit §C). 09 makes this a test.

### 5.1 The log template

- the subject, commit and result;
- **What shipped**;
- **Deviations from the prompt**, each classified;
- **Verification performed**;
- **The deliberate-breakage record**;
- **Observations not acted on**;
- **State handed to the next prompt**.

## 6. Decisions

### 6.1 Made by the user (2026-10-04), and not reopened here

- **D1 — the datastore layer is generic.** `ShardedPool`, `Datastore`, the reader and the
  inventory name none of the project's tables; the facts they need are declared by the factories or
  in `config/`. The user had made `ShardedPool` generic in December 2025 and did not intend it to
  stop being so: "Why does `ShardedPool` need to be tightly coupled to its client code? This is bad
  design."
- **D2 — every old-store fallback is deleted, through this campaign.** Old stores are regenerated.
  Not "on sight": each deletion goes through a prompt, a log and a review, because deleting code that
  current stores also reach is a risk (audit B5, B10). **This reflects the build phase, not a
  permanent rule.** Once a store holds curated results behind a published paper, keeping access to
  it may matter, and the question reopens; that is not now.
- **D3 — the `BackgroundModelValue` key exception was deleted early, and is reviewed here.** The
  user asked for it to go; it was deleted on 2026-10-04 on branch `drop-valuerow-key-exception`,
  commit `e53f323`, outside any campaign, and kept there so as not to be lost. Prompt 03 reviews that
  commit against its own acceptance tests and takes it into its commit (`git cherry-pick
  --no-commit`), so that it lands under a log like everything else.
  *Amended 2026-10-04:* the user had the branch merged into `handover-remedial` with the plan, so
  `e53f323` is on the branch before 03 runs. 03 reviews it in place; anything the review finds is
  fixed in 03's commit and recorded there.
- **D4 — the `timestamp` exception goes too**, in a controlled way: prompt 03.

### 6.2 Decisions this campaign needs

Put by the audit (its §H). **Taken by the user on 2026-10-04, each as recommended**; the
recommendation below is the decision.

- **U1 (taken, as recommended) — an absent table** (audit B5; prompt 05). The column half of B5 is plainly old-store and
  goes. But "a table a shard lacks reads as empty" is also how the reconcile and the prune at open
  survive two current-code interruptions: an actor that dies partway through its first
  `_ensure_tables`, and an interrupted `--drop gk-source-policy-records`. **Recommended:** a
  read-write open keeps reading an absent table as empty (the actors then create it), because that
  is crash recovery, not compatibility; the reader, the inventory and the read-only pool refuse an
  absent table by name, because they must not create it and have nothing to recover. **Rejected:**
  refusing everywhere (an interrupted first open would leave a store only a person can remove,
  `a3-v2-readiness`'s U2 again); tolerating everywhere (keeps the old-store paths).
- **U2 (taken, as recommended) — the older registries** (audit C1 rows 9–11; prompt 06). `Datastore._factories` (the
  composition root), `_drop_actions`/`_drop_order` and `ClientPool._default_serial_batch_size` date
  from 2024–2025. **Recommended:** move all three to `config/`, as the sharding configuration was
  moved, with `main.py` passing them in; derive `_drop_order` from foreign keys if 06 can do so in
  scope. **Rejected:** leaving them (the layer keeps importing every factory).
  *Two choices of method, taken by the user on 2026-10-05 while 06 was written, each as
  recommended:*
  - **The layer is given the tables to drop**, not the CLI's action names: `config/` holds the
    action → tables groups, `main.py` resolves `--drop`, and the actor drops in an order derived
    from foreign keys, so `_drop_order` is deleted. **Rejected:** giving the layer the names and the
    groups (it would carry the client's vocabulary); moving both registries unchanged (the
    hand-kept order stays, and it was wrong within every group).
  - **The inventory reads the replicated set from the store**: every primary records it, and a
    pool already refuses a store whose record differs from what it is given. This amends §2's
    "given … (and the replicated set)". **Rejected:** a parameter, which could disagree with the
    store it describes.
- **U3 (taken, as recommended) — the registry's legacy records** (audit B11). **Recommended:** out of scope. They are
  records, not stores, the registry never deletes a record, and the old run directories are listed
  today.
- **U4 (taken, as recommended) — the `version` and tag tables** (audit §C, "what is infrastructure"; prompt 07). Every row
  carries a foreign key to `version.serial` and every tagged class associates with `store_tag`.
  **Recommended:** the layer owns both as part of its contract (named once, as constants of the
  layer), and builds their objects through their factories; `ShardedPool` stops importing
  `MetadataConcepts`. **Rejected:** declaring them per project (they are not a project's choice).

*A choice of method for 07, taken by the user on 2026-10-06 as recommended* (audit C1 row 2):
- **A table whose rows belong to another's declares the foreign-key column that names its owner**,
  as a key of its factory's `register()`; the owner is the table that foreign key references. Tag
  tables need no declaration: under U4 the tag table is the layer's own, so a table with a foreign
  key to it and to one class is that class's tag table. One function gives both the copy unit of
  the check at open and the prune unit. Today only `BackgroundModelValue` declares an owner.
  - **Why a declaration.** No foreign key says that a value row belongs to its model while a model
    does not belong to its redshift: `BackgroundModelValue → BackgroundModel` and
    `BackgroundModel → redshift` have the same shape. Measured on 2026-10-06, `_prune_unit`'s rule
    (every compared table with a foreign key to the class) gives `BackgroundModel` its right unit.
    It also puts `BackgroundModel` and `BackgroundModelValue` in `redshift`'s unit, `BackgroundModel`
    in `IntegrationSolver`'s, and `wavenumber_exit_time` in `wavenumber`'s and `tolerance`'s.
  - **Rejected:** that rule as it is. Its outcomes are the same in every state an interrupted write
    can leave, but its licence names tables the write never touches.
  - **Rejected:** a structural rule with no declaration, under which a table nothing references
    belongs to the most derived compared table it references. It matches today's unit, but it
    infers an ownership the schema never states, and a future table would join a unit unasked.

  This amends §2's "derived from foreign keys (as `_prune_unit` is)": the unit is still read from
  foreign keys, through the one the owning table names.

*Three choices of method for 08, taken by the user on 2026-10-06, each as recommended* (audit C1
rows 12 and 13):
- **The inventory's dependencies are the parents each factory declares, in a declarative spec.**
  Each factory's `inventory_records`, today one `read_records` call, becomes a static
  `inventory_spec()` returning the keywords it passes; the layer calls `read_records` itself. The
  classes are those whose factory declares a spec, and the order is a dependency sort over the
  parents the specs declare.
  - **Why not foreign keys.** Measured on 2026-10-06 at `5fe0ae6`: nine of the parents the
    inventory reads have no foreign key (the polymorphic cosmology reference of
    `wavenumber_exit_time` and `BackgroundModel`, `GkSource`'s parent-set members, and
    `QuadSourceIntegral`'s cross-shard references), and the reversed foreign-key order violates
    all nine.
  - **Rejected:** declaring the parents a second time beside `inventory_records` (two statements
    of one fact, which could disagree until a read failed); foreign keys plus the nine edges
    declared separately (the same, and a new non-key parent would be misordered silently); the
    list moved to `config/` (a hand-kept order, which U2 rejected for the drop order).
- **Roots first.** The classes that reference no parent come first, in registry order, then a
  dependency sort in registry order. Measured: this reproduces today's order exactly, so the
  report, the fingerprint's class order and the tests that pin the order do not move.
  **Rejected:** a plain dependency sort, which moves `wavenumber_exit_time`, `GkSourcePolicy` and
  `QuadSourcePolicy` (the fingerprint's digest would not change).
- **The polymorphic reference is declared in the referencing `Parent`**, through one client
  constant that both referencing factories use. **Rejected:** each cosmology factory declaring the
  type value it answers to (one more declaration mechanism, for the same fact); the map in
  `config/`, passed to `read_inventory` (another parameter, which could disagree with what the
  factories read).

  This amends §2's "derived from the registry and foreign keys": the order is derived from the
  registry and the parents its factories declare.

*Four choices of method for 09, taken by the user on 2026-10-06.* Three were taken as
recommended. The first was not, and the record says so.
- **`config.defaults` is allow-listed** (not as recommended). `ShardedPool` and `ProfileAgent`
  size the layer's own text columns with `DEFAULT_STRING_LENGTH` (256), imported from the client's
  configuration. The guard allows that import and no other module of `config`.
  - **Recommended, not taken:** a constant of the layer's own in `Datastore/contract.py`, with the
    same value, so that the `CREATE TABLE` text is unchanged and the layer imports nothing from
    `config/`.
- **`utilities` is allow-listed** (as recommended). `Datastore.py`, `ProfileAgent` and
  `RayWorkPool` import `WallclockTimer` and `format_time` from it. It is a shared helper
  importing only the standard library, and it knows no table.
  - **Rejected:** moving the two helpers into the layer and re-exporting them from `utilities`.
    The layer would then import only the standard library, `ray` and `sqlalchemy`, but at the cost
    of moving code for no behaviour.
- **07's and 08's per-file name tests are folded into the guard and deleted** (as recommended),
  so that what the layer may not name is defined once. Their vocabulary, the union of the two,
  is checked in every file of the layer.
  - **Rejected:** keeping them beside the guard, which would define the vocabulary three times,
    free to drift apart.
- **Rule 9's closing check is done on fixtures in the prompt, and by one rehearsal rebuild at the
  review** (as recommended, after the user asked what the rebuild adds). The fixtures write their
  rows with SQL, not through the factories, so they show that the schema and the read path are
  unchanged, but not what the code writes.
  - The prompts that touched the write path are 01–02 (handlers in factory build paths), 03 (the
    check at open) and 07 (the replicated write's version row and owned serials). Nothing has run
    that path end to end since `dbd8a51`.
  - a3-v2-readiness-07's reference rehearsal ran at exactly `dbd8a51`, and its fingerprint
    (`39809dca…`) is on record. No compute code has changed since. The same command at the final
    tree should reproduce that fingerprint: about 13 minutes on 6 CPUs, run once and detached.
  - This adds the one exception in §3.
  - **Rejected:** fixtures only, which leaves the write path checked only by the stand-in pool's
    tests.

## 7. Baselines

Measured at `dbd8a51` by the author of this plan:

| Suite | Result |
|---|---|
| Datastore | 598 OK (428 s) |

The other suites were last measured by `a3-v2-readiness` (its README §7 and board §5); re-measure
every suite at `HEAD`, clean, before every dispatch, with:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t . > <file> 2>&1
grep -E '^(Ran|OK|FAILED)|^(FAIL|ERROR):' <file>
```

Read the verdict from the file, never from `| tail -N`: the suites print banners and warnings after
unittest's summary. `ComputeTargets`' `test_tk_wkb_phase.TestCost.test_wall_time_per_object` is a
known wall-clock flake.
