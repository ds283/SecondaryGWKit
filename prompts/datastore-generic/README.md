# Campaign — datastore-generic

**Written:** 2026-10-04 on branch `datastore-generic-plan`, from `handover-remedial` at `dbd8a51`
(the audit's tree), committed on `a6c8f39`, which changes no file the audit cites, by Claude Opus 5.5, from the audit [`docs/datastore-generic-audit.md`](../../docs/datastore-generic-audit.md)
and the user's decisions in §6.1. **Draft for the user's review.** **Prompts 01, 02 and 03 are
written**; their tasks are fixed by §6.1 and the audit alone. **04 is not yet written**, though its
task is fixed (the audit's G4 settled the one question it had). **05 is held** on U1; **06–09 are
held** on 01–05 and on U2–U4 (§2).

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
- `config/` — where the registries the layer stops owning are moved, if U2 says so (06);
- the tests of all of the above, under §5 rule 7.

**Out of scope:**
- **any change to what the code writes**: no table, column, key, constraint or stored value
  changes (§5 rule 9). A store written at `dbd8a51` opens, reads and fingerprints the same after
  every prompt;
- `RunRegistry/`'s fallbacks for legacy sidecars, manifests and ledger records (audit B11), unless
  U3 says otherwise;
- `main.py`, the drivers' behaviour, and every compute target;
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
| 04 | `04-delete-the-legacy-shard-records.md` | Audit B6. Absolute shard records in `shard_paths.py` and `ShardedPool`; `tools/shard_key_audit.py`'s pre-`a2bd966` and missing-`shard_key_config` paths; the sweep script's docstring; `write_legacy_primary` renamed for the malformed-record tests that use it; the legacy tests deleted. | **not yet written**; task fixed; after 03 |
| 05 | `05-refuse-a-store-whose-schema-differs.md` | Audit B5. The reader, the inventory and the read-only pool stop tolerating absent or extra tables and columns; the reconcile's column-subset and `unkeyed` logic and the `columns` difference go. What happens to an **absent table** is U1. | **held** on U1; after 03 |
| 06 | `06-give-the-layer-its-registry.md` | Audit C1 rows 10, 14 and the `config.sharding` import. `ShardedPool`, `store_reader` and `store_inventory` are given the factory registry (and the replicated set) instead of importing `Datastore._factories` and `config.sharding`. Whether `_factories`, `_drop_actions`/`_drop_order` and `ClientPool`'s batch sizes move to `config/` is U2. | **held** on 01–05 and U2 |
| 07 | `07-declare-the-replication-facts-in-the-factories.md` | Audit C1 rows 1, 2, 4–8. The copy-as-a-unit rule derived from foreign keys (as `_prune_unit` is); monotone flags and the `validated` rule declared by the factories; the version and tag tables per U4; every project name out of `ShardedPool`'s code, comments and messages. | **held** on 06 and U4 |
| 08 | `08-derive-the-inventorys-class-list.md` | Audit C1 rows 12, 13. The inventory's class list and order derived from the registry and foreign keys; the polymorphic cosmology reference declared by the client. | **held** on 06 |
| 09 | `09-keep-the-layer-generic.md` | A test that the generic layer (audit §C) names no table of the registry and imports no client package, outside a stated allow-list; every suite; the closing verification of §5 rule 9. Closes the campaign. | **held** on 01–08 |

**Why 04 is not yet written.** Its task is fixed: the audit's G4 found the only store that would
meet it, the A3 baseline, already unreadable at `HEAD` for other reasons. It edits `ShardedPool`
after 03 and is written once 03 has landed and been reviewed, against the tree 03 leaves.

**Why 05–09 are held.** Their charters above are fixed and cannot drift to fit what 01–04 find.
What waits is their method, and for 05–07 a decision of the user's (§6.2).

**Order.**
- 01 → 02: both edit factory refusals and their test modules overlap.
- 03 is independent of 01–02, but never in one checkout concurrently with another prompt that
  edits `ShardedPool.py`.
- 03 → 04 → 05: each edits `ShardedPool.py`.
- 06 after 01–05; 07 and 08 after 06; 09 last.

## 3. Datastores

**No prompt opens, writes, copies or retires a store under `var/`.** Tests build stores in temporary
directories, with the stand-in pool (`Datastore/tests/standin_pool.py`) or the real-store fixtures.
The A3 baseline (`var/datastores/handover-A3-baseline-lambdacdm`, Keep) is already readable only at
the SHA that wrote it (audit G4); nothing here changes that, and nothing opens it.

**Before touching anything,** run `python -m RunRegistry list` and report anything `running`.

## 4. The interfaces between prompts

The **names** below are fixed so that each prompt can be written against the one before it. The
internals are the implementing prompt's to design.

- **After 01–02**, no factory catches a `SQLAlchemyError` in order to name an older schema. A store
  written before a column existed meets SQLAlchemy's own `OperationalError`.
- **After 03**, `_replicated_table_specs` compares every column of every replicated table under its
  primary key; `ShardedPool` assumes `replication_in_flight` exists on every primary it opens.
- **After 05**, a store whose tables or columns differ from the registry's is refused by name before
  anything reads it, in one place. 05 names the exception.
- **06 ships** the constructor parameter by which `ShardedPool` receives the registry, and the
  parameter by which `store_reader.open_read_only` and `store_inventory.read_inventory` receive it.
  06 names them; 07–09 use them.
- **07 ships** the `register()` keys (or factory hooks) for the facts of audit C1 rows 5 and 6. 07
  names them; `schema.build_schema` carries them into `built.records`.

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
- **D4 — the `timestamp` exception goes too**, in a controlled way: prompt 03.

### 6.2 Decisions this campaign needs

Put by the audit (its §H). Each has a recommendation; none is taken.

- **U1 — an absent table** (audit B5; prompt 05). The column half of B5 is plainly old-store and
  goes. But "a table a shard lacks reads as empty" is also how the reconcile and the prune at open
  survive two current-code interruptions: an actor that dies partway through its first
  `_ensure_tables`, and an interrupted `--drop gk-source-policy-records`. **Recommended:** a
  read-write open keeps reading an absent table as empty (the actors then create it), because that
  is crash recovery, not compatibility; the reader, the inventory and the read-only pool refuse an
  absent table by name, because they must not create it and have nothing to recover. **Rejected:**
  refusing everywhere (an interrupted first open would leave a store only a person can remove,
  `a3-v2-readiness`'s U2 again); tolerating everywhere (keeps the old-store paths).
- **U2 — the older registries** (audit C1 rows 9–11; prompt 06). `Datastore._factories` (the
  composition root), `_drop_actions`/`_drop_order` and `ClientPool._default_serial_batch_size` date
  from 2024–2025. **Recommended:** move all three to `config/`, as the sharding configuration was
  moved, with `main.py` passing them in; derive `_drop_order` from foreign keys if 06 can do so in
  scope. **Rejected:** leaving them (the layer keeps importing every factory).
- **U3 — the registry's legacy records** (audit B11). **Recommended:** out of scope. They are
  records, not stores, the registry never deletes a record, and the old run directories are listed
  today.
- **U4 — the `version` and tag tables** (audit §C, "what is infrastructure"; prompt 07). Every row
  carries a foreign key to `version.serial` and every tagged class associates with `store_tag`.
  **Recommended:** the layer owns both as part of its contract (named once, as constants of the
  layer), and builds their objects through their factories; `ShardedPool` stops importing
  `MetadataConcepts`. **Rejected:** declaring them per project (they are not a project's choice).

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
