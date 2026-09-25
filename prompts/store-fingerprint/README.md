# Campaign — store fingerprint

**Written:** 2026-09-24 at `218ca74` on `handover-remedial`, by Claude Opus 5.5, from
[`docs/store-fingerprint-audit.md`](../../docs/store-fingerprint-audit.md) and the user's
decisions recorded in §6. **Prompts 03 and 04 written** 2026-09-25 at `8de5a40`, against the
structure prompt 02 shipped.

## 0. Why this campaign exists

A store is copied between machines: it is built on `macstudio-tunnel`, then rsync'd here. After
that, nothing can say whether the copy holds what the original held. A file hash is the wrong
instrument, because SQLite files are not byte-stable: VACUUM, page reuse and journal replay change
the bytes without changing the content. `run-registry`'s
`[04-a-runs-product-is-named-but-never-fingerprinted]` opened the question. Its amendment of
2026-09-24 (`218ca74`, on that board's §3) decided the answer, and **this campaign owns it**:

- a **content fingerprint**, made of digests only, never a full listing;
- kept in the store's **sidecar**, and copied into the run record when a run finishes;
- computed by a **read-only registry operation**, `store fingerprint`;
- from a **structured inventory service** that names each work item by its **physical labels** and
  its **tag set**, gives `QuadSourceIntegral` a real record, and keeps the `*Value` tables as counts.

The audit found the present inventory far from that.

- **Its labels are store-local serials.** Two stores built independently need not share one serial
  (audit §4).
- **It counts rather than lists** the headline product, `QuadSourceIntegral`.
- **It cannot see tags at all**, although tags are part of every compute target's lookup identity.
- **It samples replicated tables from one shard chosen at random** (§1).
- **It is reached only through a full read-write pool open**, which needs Ray and, by default,
  **deletes the store's unvalidated rows** (§2).

**0.1 Correctness is the only objective.** Work is sequenced by epistemic dependency, never by
urgency or cost.

1. The schema builder and the read-only reader come first, because every later step reads through
   them. A reader that could write would make every later measurement suspect.
2. The structured inventory comes next, because both the display and the fingerprint consume it.
3. The fingerprint is written only once the inventory it digests exists and has been checked
   against the store.

## 1. Scope

**In scope:**
- `Datastore/SQL/Datastore.py`: `_build_schema` (prompt 01), and `inventory` and
  `InventoryConfigType` (prompt 03, which retires them). Also a new schema module beside it, and the
  new read-only reader and structured inventory modules under `Datastore/`;
- every factory in `Datastore/SQL/ObjectFactories/`, for its inventory methods only;
- `Datastore/SQL/ShardedPool.py`, for `inventory`, `_merge_queue`, the constructor's
  `inventory_config` parameter, and one read-only `primary` property (prompt 03, which retires the
  first three and adds the fourth for `available_run_labels`);
- `config/sharding.py` (`inventory_config` only, prompt 03);
- `tools/inventory_report.py`, `main.py`'s `--inventory` branch and its `inventory_config`
  import, and `extract_common.available_run_labels` (prompt 03);
- the two scripts that pass `inventory_config` to a pool,
  `docs/source-remediation-verification/analyse_greens_and_source.py` and
  `run_quadsource_integrals.py`, for that argument only (prompt 03). Found when prompt 03 was
  written: retiring the parameter breaks them otherwise;
- `ComputeTargets/tests/test_qcd_cosmology_inventory.py`, the only test of an `inventory()`, which
  prompt 03 retires and re-expresses against the new service (D4, §6.2);
- `RunRegistry/` (prompt 04);
- `docs/gktk-remedial/scoped_pipeline_run.py` and `docs/handover/quadsource_atol_sweep.py`, for
  taking the fingerprint when a registered run finishes (prompt 04; D2 for the second);
- the three real sidecars, for their `fingerprint` field only (prompt 05; D3);
- the tests of all of these;
- the boards of `qcd-background-audit` (prompt 03) and `run-registry` (prompt 04), for the closure
  of the one issue each prompt closes there.

**Out of scope:**
- every factory's `build`, `store`, `read_batch` and `validate_on_startup`;
- every lookup key;
- any physics;
- `ShardedPool`'s open, copy, move and routing paths;
- every other part of `docs/handover/quadsource_atol_sweep.py`;
- every existing run manifest, and every field of an existing sidecar except `fingerprint`;
- the defects the audit found outside the inventory. They are opened on the board (§3 there), not
  fixed here. The exceptions are the two assigned to prompts below.

**The layering is unchanged.** `ShardedPool` and `tools/sharded_store.py` still know nothing about
sidecars (`datastore-portability` README §6.3). The reader and the inventory live in `Datastore/`,
because they read `Datastore` tables. The fingerprint's format and its place in the sidecar and
run record live in `RunRegistry/`, because the registry owns those files.

## 2. Prompts

| # | Prompt | Covers | Status |
|---|---|---|---|
| 01 | [`01-a-read-only-store-reader.md`](01-a-read-only-store-reader.md) | One schema builder shared by the actor and the reader. A read-only, no-Ray reader over a closed store's shards. A real multi-shard store fixture for tests. | **written** |
| 02 | [`02-a-structured-inventory.md`](02-a-structured-inventory.md) | Per-class structured records on the reader: physical keys, parents by canonical key, full tag sets, `validated`, per-parent value counts, real `QuadSourceIntegral` / `OneLoopIntegral` / `GkSourcePolicyData` records, replicated classes compared across shards. | **written** (D1 decided) |
| 03 | [`03-one-inventory-service.md`](03-one-inventory-service.md) | The display (`main.py --inventory`, which becomes read-only and needs no Ray) and `available_run_labels` consume the structured inventory. The old three shapes, `ShardedPool.inventory`, `_merge_queue` and `inventory_config` retire. Closes `[00-inventory-run-prunes-unvalidated-rows-by-default]` and the `BackgroundModel` half of `qcd-background-audit`'s `[03-qcd-inventory-does-not-report-the-representation]`. | **written** 2026-09-25 (D4 open) |
| 04 | [`04-the-fingerprint.md`](04-the-fingerprint.md) | A pure function from the structured inventory to digests. A known `fingerprint` field in `RunRegistry.stores`. `python -m RunRegistry store fingerprint`, read-only, refusing a store a `running` run names. The digest in the run record at finish. `scoped_pipeline_run.py` and `quadsource_atol_sweep.py` (including `--build`, which builds the A3 v2 store) take one when their registered run finishes. Closes `run-registry`'s `[04-a-runs-product-is-named-but-never-fingerprinted]`. | **written** 2026-09-25 |
| 05 | *Fingerprint the real stores* | Remedial: fingerprint the three existing stores, read-only, and record each fingerprint in its sidecar (D3). Also any store built here before 04 landed, such as the A3 v2 store, which therefore had no fingerprint taken at finish. Fingerprint a registry copy of one, and show that the digests localise the known differences. | **held** until 04 lands |

**Why 03–05 were held.** Each consumes the structure prompt 02 ships. Their **charters** are fixed
above and cannot drift to fit what 02 finds; only their **methods** waited. 02 landed on 2026-09-25
at `8de5a40`, and 03 and 04 were written the same day against it. They are independent of each
other, and either may go first. 03 is dispatched only once D4 is recorded. 05 is held until 04
lands, because it writes 04's format.

## 3. Datastores

Three stores live under `var/datastores/`: `handover-A3-baseline-lambdacdm`, `handover-atol-sweep`
and the backup in `backup-pre-resume-20260921T091011/`. Each is about 350 MB over four shards.
Since 2026-09-24 each has a registry sidecar.

**All three are read-only to prompts 01–04.** Every demonstration works on a `cp -p` copy in
`var/store-fingerprint-check-NN/`, which is deleted afterwards. A read-only reader that is
correct would not write the originals. But whether it is correct is what the demonstration tests,
so it is not pointed at them. Only prompt 05 reads the originals, and only once 01's reader has been
shown not to write. Prompt 05 then writes one thing: the `fingerprint` field of each original's
sidecar (D3). It never writes a store.

**Before touching any store,** run `python -m RunRegistry list` and confirm nothing is `running`.
About 0.4 GB of free space is needed for a copy. The volume had about 15 GB free on 2026-09-24.

## 4. The interfaces between prompts

The **names** below are fixed here, so that each prompt can be written against the one before it.
The internals are the implementing prompt's to design.

- **Prompt 01** ships `Datastore/SQL/schema.py` with **`build_schema(metadata, factories)`**. It
  returns the `Table` objects and the per-class schema records that `Datastore._build_schema`
  builds today. `_build_schema` calls it, so there is one definition.
- **Prompt 01** ships `Datastore/store_reader.py` with **`open_read_only(primary)`**, a context
  manager that yields a read-only store. For each shard it gives the shard's serial and path, a
  read-only engine, the `Table` objects, the tables absent from that shard, and the columns absent
  from tables that are present. There is no write path.
- **Prompt 02** ships `Datastore/store_inventory.py` with **`read_inventory(primary)`**. It returns
  the structured inventory: per class, its records, its count, its time range and its problems.
  Each record has:
  - a canonical `key` built only from JSON-safe physical leaves and parent keys;
  - `tags`, a sorted tuple of labels;
  - `validated` (`None` where the class has no such column);
  - `value_count` (`None` where the class has no `*Value` table).

  There is room for a computed-values field later, outside the key. One function, **`canonical`**,
  turns a leaf into its canonical form (D1).
- **Prompt 03** ships `tools/inventory_report.py` **`format_inventory_report(inventory, db_name,
  verbose=False)`**, which renders a `StoreInventory`, and the read-only property
  **`ShardedPool.primary`**. `extract_common.available_run_labels(pool)` keeps its signature.
- **Prompt 04** ships `RunRegistry/stores.py` **`fingerprint_store(primary, …)`** and the sidecar
  field **`fingerprint`**. Also the pure **`fingerprint_of(inventory, …)`** and
  **`compare_fingerprints(recorded, current)`**, **`FINGERPRINT_FORMAT`**,
  **`Run.finish(state, exit_code=None, *, fingerprint=False)`**, and
  `python -m RunRegistry store fingerprint`.

## 5. The rules this campaign runs under

The project-wide ones in `CLAUDE.md`, unchanged, plus:

1. **One commit per prompt.** The commit boundary is the rollback boundary.
2. **Every prompt writes a log** to `logs/NN-<name>.md`, classifying every deviation as
   `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.
3. **Every prompt updates `IMPLEMENTATION_STATE.md` in its own commit**, plus `docs/OPEN_ISSUES.md`.
4. **Do not fix things the prompt did not ask for.** Record them and open a §3 issue.
5. **Commit messages** in `CLAUDE.md`'s form, ending with `Co-Authored-By:` naming the model.
6. **Verification documents are additive.**
7. **Existing tests are not modified.** The four hand-copies of `_build_schema` in existing tests
   (audit §3) stay as they are. A prompt may add tests beside them. **The one exception** is D4
   (§6.2): prompt 03 retires `ComputeTargets/tests/test_qcd_cosmology_inventory.py`, the test of an
   `inventory()` it deletes, and re-expresses its claims against the new service.
8. **Deliberate breakage.** Each prompt names mutations that its tests must catch. Each is recorded
   in the log as a diff, exactly as applied, so that the orchestrator can replay it with
   `git apply`. Mutations are never committed.
9. **No test needs Ray, and no test opens anything under `var/`.** Stores for tests are built in
   temporary directories.

### 5.1 The log template

- the subject, commit and result;
- **What shipped**;
- **Deviations from the prompt**, each classified;
- **Verification performed**;
- **The deliberate-breakage record**;
- **Observations not acted on**;
- **State handed to the next prompt**.

## 6. Decisions

### 6.1 Made by the user (2026-09-24), and not reopened here

Recorded in full on the `run-registry` board, under the amendment to
`[04-a-runs-product-is-named-but-never-fingerprinted]`:

1. **The fingerprint lives in the store sidecar.** It describes a store, and a store gathers several
   runs' products. It is also copied into the run record when a run finishes.
2. **The inventory service returns structured output.** Display and fingerprints are two
   consumers of it. It may later carry computed values, which each consumer uses or ignores.
3. **The inventory is concise.** It must not become another serialisation of the store. The
   `*Value` tables, which hold nearly all rows, are counted and never listed.
4. **Work items are named by physical labels**, not row ids.
5. **Tags are part of each work item's record and of the fingerprint.** They label the grid a
   product was computed on.
6. **`QuadSourceIntegral` gets a real inventory record.**
7. **Digests only.** The sidecar holds, per class and per tag set, a count and a digest, plus an
   overall digest, a format version, when it was taken and by which run. Timestamps are excluded.
   A full listing is generated on demand, where the store is.
8. **The accepted loss.** Rows removed after a fingerprint is taken, by `--prune-unvalidated` or by
   hand, cannot be identified afterwards.
9. **The registry computes the fingerprint.** "Does not open results" was one docstring's
   boundary, not a rule. `CLAUDE.md`'s limits stand: no scheduling, supervising, restarting,
   locking or deleting, and no growing into a project of its own.

### 6.2 Decisions D1–D4

D1–D3 were asked for when the campaign was written at `066057d`, and the user made them the same
day, 2026-09-24. D4 was found when prompt 03 was written, on 2026-09-25, and is open.

- **D1 — the canonical form of a float: the stored bits, as `float.hex`.** The recommendation was
  taken. Prompt 02 was written against it, and needs no amendment. The alternative, rounding to
  about 12 significant figures, is recorded in audit §5. It would absorb last-bit differences
  between machines, but at the cost of making "the same data" approximate.
- **D2 — `docs/handover/quadsource_atol_sweep.py` takes a fingerprint when its registered run
  finishes. Yes.** The user wants the A3 v2 store fingerprinted, and `--build` is what builds it.
  Prompt 04 changes that script for this purpose only. Nothing else in it changes, and its
  measurement code and results are untouched. If the A3 v2 store is built here before prompt 04
  lands, prompt 05 fingerprints it.
- **D3 — prompt 05 writes fingerprints into the three existing sidecars. Yes.** Prompt 05 is the
  remedial step for the stores that existed before the campaign: the live A3 store, the sweep store
  and the backup. It writes only the `fingerprint` field, through `RunRegistry.stores`' writer.
  Every other field, and every store file, stays as it is. This is the explicit request that
  `datastore-portability` README §6.5 point 7 requires before an existing sidecar changes.
- **D4 — the test of a retired method. Open; to be decided before prompt 03 is dispatched.**
  Prompt 03 deletes every factory's `inventory()`, which is F9's charter. One existing test module
  calls one of them: `ComputeTargets/tests/test_qcd_cosmology_inventory.py`, three tests that
  `sqla_QCDCosmology_factory.inventory()` reports `T_z_representation`. It is also one of the four
  hand-copies of `_build_schema` that rule 7 (§5) keeps. So F9 and rule 7 cannot both hold for it.
  - **Recommended: retire the module, and re-express its claims against the new service** in a
    new test: two `QCD_Cosmology` rows differing only in `T_z_representation` are two records and
    render differently; the representation is shown beside `log10_max_z`. The third claim changes
    meaning, and the prompt says so. Two rows that differ only in `name` were two labels; under
    the new service they share a key, and the inventory names them a `duplicate`. What the test
    protected, that the representation is visible, is kept. Prompt 03 is written against this.
  - *Alternative:* keep `sqla_QCDCosmology_factory.inventory()` alive for the test alone. Then
    there are two inventory services for one class, which is what F9 exists to end, and the test
    guards a method nothing calls.
  - *Alternative:* keep the module, and rewrite its body against the new service. That modifies
    an existing test, which is what rule 7 forbids, with no gain over the recommendation.

### 6.3 Choices the prompts make, where §6.1 leaves the method open

These are marked *prompt's choice* in the prompts, each with its reason, and may be overridden by a
logged deviation that is at least as strong:

- the full identity, including the optional filters `z_init`, `z_source` and `z_response`, is in
  every key;
- solver identity is not in a work item's key, because it is not in any lookup;
- unvalidated rows are recorded, with their flag;
- the `Run_<label>` tag stays in the tag set, because it is part of the lookup;
- a missing table in an old store reads as empty, and a missing column is a named problem;
- a replicated class is read from every shard, and any disagreement is a named problem;
- (prompt 03) the display types a float leaf by the schema, never by the shape of a string, and
  renders it to six figures, or as `repr` when verbose;
- (prompt 03) `--inventory` never creates a store, and refuses `--drop`;
- (prompt 03) `available_run_labels` returns what it returned before, every `Run_` label in
  `store_tag`, and refuses a problem in that class;
- (prompt 04) a digest covers the canonical record, `validated` and `value_count` included, and is
  formed so that a listing's lines hash to it;
- (prompt 04) problems are counted beside the digests, never in them, because their text is
  store-local;
- (prompt 04) a copy keeps the source's fingerprint, `taken` included;
- (prompt 04) a registered run's finish writes the sidecar as well as the run record, and takes a
  fingerprint on every terminal state it reaches with the store closed; a failure is recorded and
  never changes the state.

## 7. Baselines

At `f53598f`, measured by the `datastore-portability` prompt 03 orchestrator:

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 |
| ComputeTargets | 552 (`test_tk_wkb_phase.TestCost.test_wall_time_per_object` is a known wall-clock flake) |
| CosmologyModels | 39 |
| Datastore | 70 |
| LiouvilleGreen | 148 (1 skipped) |
| RunRegistry | 84 |

Re-measure before every dispatch, with:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t . 2>&1 | tail -40
```

The suites print banners, so `| tail -5` will not show the verdict. Do not set
`THREE_BESSEL_DIAGNOSTIC_PLOTS`.
