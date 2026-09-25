# Log 02 — A structured inventory: work items by physical label, with their tags

**Prompt:** [`prompts/store-fingerprint/02-a-structured-inventory.md`](../02-a-structured-inventory.md)
**Commit:** *(this commit)*, "Add a structured store inventory keyed by physical labels"
**Base:** `f52a7df52405cc20b1b334b43f81974afe2113fd` ("Add one schema builder and a read-only store
reader"), clean
**Model:** Claude Opus 5.5
**Date:** 2026-09-25
**Result:** COMPLETE. F4–F7 shipped. No §7 stop condition arose. The §4 copy of the sweep store has
**no duplicate and no replicated divergence**, and no key leaf column holds two storage classes.
All eight deliberate-breakage mutations were caught. One issue was opened
(`[02-exit-time-lookup-runs-inside-the-subhorizon-loop]`), and
`[00-replicated-writes-can-diverge-across-shards]` gained a measurement.

## What shipped

**`Datastore/store_inventory.py`** (new): **F4**, **F6**, **F7**.

- `read_inventory(primary) -> StoreInventory`. It opens the store with prompt 01's `open_read_only`
  and holds one connection per shard. It builds every class of `INVENTORY_CLASSES` in dependency
  order on every shard, calling the class's factory's `inventory_records`, and then combines the
  shards (below). `StoreInventory` has `primary`, `shards`, `classes` (name → `ClassInventory`, in
  dependency order) and `problems`. It also has three helpers: `digest_of(cls, record)`,
  `find(cls, digest)`, and `resolve(cls, record)`, which replaces every parent reference,
  recursively, by the parent's key and tags. The helpers are for display and checking.
- `ClassInventory`: `name`, `replicated`, `tagged`, `parents` (the key field → parent class map),
  `records` (sorted by canonical JSON), `count`, `earliest_timestamp`, `latest_timestamp` and
  `problems`.
- `Record(key, tags, validated, value_count)`: a frozen, JSON-safe dataclass, with `as_json()`. A
  computed-values field can be added beside the four without touching them.
- **`canonical(value)`**, the one leaf canonicaliser (D1):
  - a float becomes `float.hex(value)` of the stored value;
  - an `int`, `str`, `bool` or `None` is kept as it is;
  - anything else raises `TypeError`.
- **`canonical_json(obj)`**, the one canonical JSON: every leaf through `canonical`, `sort_keys`,
  separators `(",", ":")`, ASCII. `digest(obj)` is the SHA-256 hex of it.
  `reference_digest(key, tags, tagged)` is `digest(key)` for an untagged class, and
  `digest({"key": key, "tags": [...]})` for a tagged one.
- **`read_records(conn, table, tables, context, *, leaves, parents, tags, values, validated)`**: the
  one reader that every factory's builder calls with its identity spec. On one shard it:
  - selects `serial`, the key columns, `validated` and `timestamp`;
  - reads the association table `(parent column, tag_serial)` and maps each tag to its label
    through the shard's `store_tag` keys;
  - counts the value table with **one `SELECT parent, count(*) … GROUP BY parent`**;
  - resolves each parent reference through `context.digests`. The polymorphic cosmology reference
    goes through `cosmology_type`, using `CosmologyModels/model_ids.py`: 0 → `LambdaCDM`,
    1 → `QCD_Cosmology`;
  - names every problem.

  It returns a `ShardRead` of `(serial, Record, timestamp)` rows, the problems, and whether the
  shard is `comparable`, which is false when its class, association or value table is absent or
  its key columns are incomplete. It raises if a parent class has not been built yet on that
  shard, so dependency order is enforced, not assumed.
- **Combining the shards** (`_combine`):
  - a **sharded** class is the union of every shard's rows;
  - a **replicated** class is compared across its comparable shards. The comparison is a multiset
    of each record's canonical JSON, which covers key, tags, validated and value count. The
    records are the lowest-serial comparable shard's, and every other shard that differs is a
    `replicated-divergence` problem. The problem names the shard, the number of records that
    differ, both ways, and up to five example keys.
  - **Duplicates** are records of one class that share a key and a tag set, across all
    contributing rows. They are a `duplicate` problem, naming up to five groups by shard and
    serial, and every one is kept.
  - The timestamp range covers the contributing rows only. It is for display.
- **Named problems.** Each is a string `"<name>: <class>: <text>"`. The names are:
  - `absent-table`;
  - `incomplete` (a key column absent);
  - `replicated-divergence`;
  - `duplicate`;
  - `orphan-value` (the count, and up to five parent serials);
  - `orphan-tag` (a missing parent or a missing tag, the count, and up to five serials);
  - `unresolved-parent` (the count, the fields, and up to five row serials; such rows are not
    records).
- At module scope it imports only the standard library and `sqlalchemy`. The reader, the factory
  map and `config.sharding` are imported inside `read_inventory`, and `CosmologyModels.model_ids`
  inside `cosmology_classes()`. So a factory can import it from inside its method without a cycle.

**Twenty-one factories gain one static method each, `inventory_records(conn, table, tables,
context)`**: **F5**. Each goes directly after that factory's existing `inventory()`. Each is a
comment naming the identity, a function-local import of `read_records` (and `Parent` /
`COSMOLOGY`), and one `read_records(...)` call stating the key. **The diff under
`Datastore/SQL/ObjectFactories/` is 382 insertions and 0 deletions.** No `inventory()`, `build`,
`store`, `read_batch`, `validate_on_startup`, lookup or schema changed. The file-level import lists
are unchanged too, because every import is inside the new method.

**`Datastore/tests/real_store_fixtures.py`** (+874, 0 lines removed): the full store. Prompt 01's
`REPLICATED_ROWS`, `SHARDED_ROWS`, `build_real_store`, `build_old_store` and helpers are untouched.
Only the module docstring gained a paragraph.

- `FULL_REPLICATED_ROWS` / `FULL_SHARDED_ROWS` / `FULL_SHARD_KEYS`: at least one row of every one
  of the 21 classes. They include:
  - tags on every tagged class;
  - unvalidated `BackgroundModel`, `TkNumericIntegration` and `QuadSource` rows;
  - value rows in all seven value tables, with different counts per parent;
  - a `QCD_Cosmology` row, with a `wavenumber_exit_time` and a `BackgroundModel` that reference it
    through `cosmology_type` 1;
  - two rows of every parent class, so that a reference can be re-pointed;
  - a `OneLoopIntegral` that shares serial 1 with a `QuadSourceIntegral` and carries other tags;
  - a `store_tag` that no row carries.
- `build_full_store(...)`: like `build_real_store`, with the same primary writer
  (`_write_primary`). Each shard is a byte copy of **one empty shard made once per process**,
  because `create_all` of 37 tables commits once per statement and was 0.45 s of every build. The
  copy is then filled with rows in one transaction. A table in `missing_tables[n]` is dropped after
  the rows are written, and its rows are skipped on that shard.
- `fill_required(rows)`: gives every non-nullable column a row leaves out a placeholder of its
  type, and makes a nullable column that another row of the same table states `None`.
- `references()`: the schema's foreign keys, plus the undeclared references:
  - `GkSourcePolicyData.wavenumber_exit_serial`;
  - `QuadSourceIntegral.source_serial`;
  - `QuadSource.Tr_serial`;
  - the polymorphic `cosmology_serial`.
- `relabel_serials(...)`: the same content under other serials. Every table's serials are reversed
  and offset, and every reference follows. Replicated rows get the same new serials on every
  shard. `find_row` and `vary_row` locate a row and change one of its columns.

**`Datastore/tests/test_store_inventory.py`** (new): **49 test methods** in 11 classes, with
subtests. None needs Ray, and none opens anything under `var/`.

| §3 | Test class | What it shows |
|---|---|---|
| — | `TestTheFullStore` (6) | Every one of the 21 classes has records. The full store has no problem. Records are JSON-safe. Tags, validated and value count are present exactly where the class has them, and the `Run_` tag is kept. Unvalidated rows are recorded. Replicated and sharded classes follow `config/sharding.py`. `resolve` names every `QuadSourceIntegral` parent. |
| 1 | `TestKeysArePhysical` (3) | An **equal inventory**, class for class and record for record, under three changes: every serial relabelled, replicated ones included (asserted disjoint from the originals); the sharded rows moved to other shards of a 3-shard store; and both together, with the shards swapped. |
| 2 | `TestIdentityColumnsMatter` (2) | The key fields of every class are exactly the `IDENTITY` list read from its lookup. For each of the **83 identity columns**, a store differing in that column alone gives different records, and a different multiset of that field's values. |
| 3 | `TestNonIdentityDoesNotMatter` (2) | **42** single-column variations give equal records for every class: nine compute-target labels, payload and provenance, six solver serials, cosmology names, policy labels, five `version` foreign keys, and the `source`/`response` flags. So does moving every timestamp. |
| 4 | `TestFloats` (6) | `canonical` and `canonical_json` directly. The last bit of one `k` changes the `wavenumber` and `wavenumber_exit_time` records. `log10_tol` is used as stored. With `canonical` patched, every float leaf carries the patch's mark. An AST check: no function of `store_inventory.py` except `canonical` uses `.hex`, `round` or `format`, and none of the 21 `inventory_records` bodies contains an f-string, `.hex`, `.format`, `float`, `round`, `repr`, `str` or `format`. |
| 5 | `TestTags` (5) | Adding a tag row changes that record's tags and nothing else. A tag on a `GkSource` changes exactly one `GkSourcePolicyData` record, in its `source` field only. A tag on `BackgroundModel` 2 reaches its child. `OneLoopIntegral` tags come from `OneLoopIntegral_tags`, and are unaffected by a `QuadSourceIntegral_tags` row with the same parent serial. |
| 6 | `TestValueCounts` (3) | The per-parent counts are exact. Deleting one row from each of the seven value tables lowers exactly one record's `value_count` by one. **Every SQL statement that names a `*Value` table is a `count(*) … GROUP BY`** (an engine event listener, F7). |
| 7 | `TestReplicatedDivergence` (4) | A changed row, a changed tag set, or a changed value count on shard 1 only gives a `replicated-divergence` naming shard #1. The records stay shard 0's. On a 3-shard store the problem names shard #2 and not #1. |
| 8 | `TestOldStoresAndOrphans` (12) | On a 3-shard store, each case gives exactly one problem, of the right name and naming the right shard, and changes nothing else. The cases: an absent class table, association table, value table, replicated table and replicated association table; a missing key column (the class loses exactly that shard's record); a missing non-key column (no problem at all); an orphan value row; a tag row with a missing parent or a missing tag; an unresolved `model_serial`; an unknown `cosmology_type` (named once per shard). |
| 9 | `TestDuplicates` (4) | A duplicate on one shard, one across shards, and one in a replicated class are each named, and the count rises by one. The same key with other tags is not a duplicate. |
| 10 | `TestReadOnlyAndNoRay` (2) | A full `read_inventory` leaves every file's SHA-256, size and `st_mtime_ns`, and the listing, unchanged, with no journal. In a child interpreter, `ray.is_initialized()` is false after a full `read_inventory`. |

## The key of every class, as shipped

"By digest" means the parent's `reference_digest`. For a tagged parent that covers its key **and**
its tag set. The lookup lines are in the committed tree.

| Class | Where | Key fields → columns | tags (association table) | validated | value_count | Lookup (`Datastore/SQL/ObjectFactories/`) |
|---|---|---|---|---|---|---|
| `version` | repl | `label` | – | – | – | `version.py:25` |
| `store_tag` | repl | `label` | – | – | – | `store_tag.py:25` |
| `redshift` | repl | `z` | – | – | – | `redshift.py:40` |
| `wavenumber` | repl | `k_inv_Mpc` | – | – | – | `wavenumber.py:47` |
| `tolerance` | repl | `log10_tol` | – | – | – | `tolerance.py:33` |
| `LambdaCDM` | repl | `omega_m`, `omega_cc`, `h`, `f_baryon`, `T_CMB_Kelvin`, `Neff` | – | – | – | `LambdaCDM.py:42-53` |
| `QCD_Cosmology` | repl | the six, `log10_max_z`, `T_z_representation` | – | – | – | `QCD_Cosmology.py:80-95` |
| `IntegrationSolver` | repl | `label`, `stepping` | – | – | – | `integration_metadata.py:32` |
| `GkSourcePolicy` | repl | `Levin_threshold`, `numeric_policy` | – | – | – | `GkSourcePolicy.py:33-38` |
| `QuadSourcePolicy` | repl | `Levin_threshold`, `numeric_policy` | – | – | – | `QuadSourcePolicy.py:33-38` |
| `wavenumber_exit_time` | repl | `k` → `wavenumber_serial` (digest); `cosmology_type`; `cosmology` → `cosmology_serial` (digest of the row that type names); `atol`, `rtol` → `atol_serial`, `rtol_serial` (digest); `stepping` | – | – | – | `wavenumber.py:252-261` |
| `BackgroundModel` | repl | `cosmology_type`; `cosmology` (digest, by type); `tau_gauss_order`, `cs_tau_gauss_order`, `friction_F_gauss_order`; `source_grid_construction`, `source_grid_digest`; `z_init` → `z_init_serial` (digest) | `BackgroundModel_tags` (`model_serial`) | yes | `BackgroundModelValue` | `BackgroundModel.py:297-323` |
| `TkNumericIntegration` | k | `model` (digest); `k` → `wavenumber_exit_serial` (digest); `atol`, `rtol` (digest); `break_point_kind`; `z_init` → `z_init_serial` (digest) | `TkNumeric_tags` (`integration_serial`) | yes | `TkNumericValue` | `TkNumericIntegration.py:242-261` |
| `TkWKBIntegration` | k | `model`, `k` (digest); `rho_gauss_order`; `z_init` (the REAL column, a leaf) | `TkWKB_tags` (`wkb_serial`) | yes | `TkWKBValue` | `TkWKBIntegration.py:273-288` |
| `GkNumericIntegration` | k | `model`, `k`, `atol`, `rtol` (digest); `break_point_kind`; `z_source` → `z_source_serial` (digest) | `GkNumeric_tags` (`integration_serial`) | yes | `GkNumericValue` | `GkNumericIntegration.py:238-257` |
| `GkWKBIntegration` | k | `model`, `k` (digest); `rho_gauss_order`; `z_source` (digest); `z_init` (REAL, a leaf) | `GkWKB_tags` (`wkb_serial`) | yes | `GkWKBValue` | `GkWKBIntegration.py:261-281` |
| `GkSource` | k | `model`, `k` (digest); `z_response` → `z_response_serial` (digest) | `GkSource_tags` (`parent_serial`) | yes | `GkSourceValue` | `GkSource.py:209-219` |
| `GkSourcePolicyData` | k | `source` → `source_serial` (digest of the `GkSource`, **tags included**); `policy` → `policy_serial` (digest); **`k` → `wavenumber_exit_serial` (digest)** | – | – | – | `GkSourcePolicyData.py:96-100` |
| `QuadSource` | q | `model` (digest); `q`, `r` → `q_/r_wavenumber_exit_serial` (digest) | `QuadSource_tags` (`parent_serial`) | yes | `QuadSourceValue` | `QuadSource.py:159-164` |
| `QuadSourceIntegral` | k | `model`; `policy` → `GkSourcePolicy`; `k`, `q`, `r`; `z_response`; `z_source_max`; `atol`; `rtol` (all digest) | `QuadSourceIntegral_tags` (`parent_serial`) | – | – | `QuadSourceIntegral.py:267-277` |
| `OneLoopIntegral` | k | `model`, `k`, `z_response`, `atol`, `rtol` (all digest) | **`OneLoopIntegral_tags`** (`parent_serial`), which `store()` writes (`:243`), not the `QuadSourceIntegral_tags` its `build()` joins (`:141`) | – | – | `OneLoopIntegral.py:149-155` |

**Not in any key**, each checked against its lookup:
- the free-text labels of compute targets and of `BackgroundModel`;
- the cosmology `name` and the policy `label`;
- `version` foreign keys, and timestamps;
- every solver serial (`solver_serial`, `phase_solver_serial`, `friction_solver_serial`);
- `z_min_serial`, `z_max_serial`, `z_samples`, the `numeric_smallest_z` / `primary_WKB_largest_z`
  serials, `Tq_serial`, `Tr_serial`, `source_serial` / `data_serial` on `QuadSourceIntegral`,
  `metadata`, and every result, timing and payload column;
- the `source` / `response` flags on `wavenumber` and `redshift`.

**The one difference from audit §4** is `GkSourcePolicyData`'s `k`: see deviation 1.

**Points the prompt asked to settle:**
- **A tagged parent's digest covers its tags.** Settled so, and tested: test 5 and mutation (vii).
- **`OneLoopIntegral` tags come from `OneLoopIntegral_tags`.** Test 5 and mutation (v). The lookup
  is not fixed; `[00-oneloop-lookup-joins-the-wrong-tag-table]` stands.
- **`wavenumber_exit_time`'s and `BackgroundModel`'s cosmology** is resolved through
  `cosmology_type`. An unknown type is an `unresolved-parent` problem (test 8).
- **The cosmology `name` and the policy `label`** stay out of the key. Test 3 varies all four.
- **Solver identity** is in no compute target's key. Test 3 varies all six solver columns.

## Each *prompt's choice*, and whether it was kept

| Choice | Kept? |
|---|---|
| Each factory owns its record builder, a new static method beside its existing `inventory()` | **Kept.** Each of the 21 builders states its class's identity. The reading mechanics (tags, values, orphans, parent resolution) live once in `read_records`, which every builder calls (deviation 2). |
| The `Run_<label>` tag stays in the tag set (README §6.3) | **Kept.** On the sweep copy every compute target's tag set includes `Run_default`. |
| Unvalidated rows are recorded, with their flag | **Kept.** `validated` is `True`, `False`, or `None` for a legacy NULL. Test: `test_classes_with_tags_validated_and_values`. |
| A parent is referenced by the SHA-256 of its key's canonical JSON, never its serial | **Kept**, for every reference, including those to the grid leaves `redshift`, `wavenumber` and `tolerance`. Test 1, and mutation (i). |
| `z_init`, `z_source` and `z_response` are always in the key | **Kept**, in all seven classes that filter on them optionally. Test 2, and mutation (vi). |
| The digest of a tagged parent covers its key and its tag set | **Kept.** Test 5, and mutation (vii). |
| Solver identity is not in any compute target's key | **Kept.** Test 3. |
| A replicated class's records are the lowest-serial shard's | **Kept, refined.** The lowest-serial **comparable** shard's, where a shard missing the class's table, association table, value table or a key column is not comparable (deviation 4). |
| (README §6.3) A missing table reads as empty; a missing column is a named problem | **Kept, as F6 words it.** A missing column is a problem only if the key needs it (`incomplete`). A missing non-key column is not a problem (test 8). |

## Deviations from the prompt

1. **`GkSourcePolicyData`'s key has a third field, `k`** (its `wavenumber_exit_serial`, by digest),
   which audit §4 and the prompt's F5 outline do not list. *STRUCTURALLY REQUIRED.* The lookup
   filters on it: `GkSourcePolicyData.py:99`,
   `table.c.wavenumber_exit_serial == k_exit.store_id`, beside `source_serial` (`:97`) and
   `policy_serial` (`:98`). F5 says every column a lookup filters on is in the key. The column has
   no foreign key; it names a `wavenumber_exit_time` row, which is replicated, so it resolves
   inside the shard. For a consistent row it repeats the `GkSource`'s own `k`, but the lookup
   treats it as independent, so the key does too. Test 2 varies it alone.
2. **The builders share one reader.** *IMPLEMENTATION CHOICE.* Each factory's `inventory_records`
   is a declaration (leaf columns, `Parent` references, association table, value table,
   `validated`) and a call to `store_inventory.read_records`. Twenty-one hand copies of the
   tag-join, value-count, orphan and resolution logic would each have been a place for one class
   to differ. The identity is still defined in the factory, beside its lookup.
3. **The builder's `context` carries both maps**, serial → canonical key and serial → reference
   digest, per class already built on that shard, plus the shard's absent tables and columns.
   *IMPLEMENTATION CHOICE.* The prompt names the key map. Children need the digest, and tags need
   the `store_tag` keys' labels.
4. **Which shards a replicated class is compared across.** *IMPLEMENTATION CHOICE.* A shard on
   which the class's table, association table or value table is absent, or whose key columns are
   incomplete, has that absence named, and is left out of the comparison and of the choice of
   records. Otherwise one absent table on the lowest shard would have emptied the class, and every
   absence would have been reported twice, once as itself and once as a derived "divergence". Test
   8 (absent replicated table, absent replicated association table) shows exactly one problem, and
   unchanged records.
5. **Per-shard problems on replicated classes are named once per shard.** *IMPLEMENTATION
   CHOICE.* An unknown `cosmology_type` in a replicated row is named on each shard that holds it,
   three times on a 3-shard store (test 8). This is faithful to what was read, and each names its
   shard.
6. **A tag row naming a missing `store_tag`** is an `orphan-tag` problem, and the unknown tag is
   **not** in the record's tag set. *IMPLEMENTATION CHOICE.* There is no label to put there. The
   test shows the record unchanged and the problem named.
7. **An absent `validated` column** gives `validated = None`, with no problem. *IMPLEMENTATION
   CHOICE.* The flag is not in the key, and F6 says a missing column the key does not need is not a
   problem.
8. **The full-store fixture is new code beside prompt 01's, not a change to its defaults.**
   *IMPLEMENTATION CHOICE.* The prompt says "extend `real_store_fixtures.py` so that its store
   holds at least one row of every class". Changing `REPLICATED_ROWS` / `SHARDED_ROWS` would have
   changed the store prompt 01's tests were written against. It would also have broken
   `build_old_store`, whose shard 1 has no `OneLoopIntegral_tags` table and must receive no rows
   for it. The extension is `build_full_store` and its row sets. Its shard writer copies one empty
   shard, which takes a store build from 0.45 s to about 0.05 s, and it skips a missing table's
   rows and drops the table afterwards. `build_real_store` is unchanged.
9. **Additions to the interface**, all in new code, none a write path: `INVENTORY_CLASSES`,
   `COMPUTE_TARGETS`, `COSMOLOGY`, `cosmology_classes()`, `digest`, `reference_digest`,
   `Record.identity` / `canonical_json`, `StoreInventory.digest_of` / `find` / `resolve` /
   `problems`, and `ClassInventory.replicated` / `tagged` / `parents`. *IMPLEMENTATION CHOICE.*
   Prompt 03's display needs the parents resolved, and prompt 04 needs the per-class digest rule.
10. **§4 step 4 deleted the `QuadSourceIntegral` row only**, as the prompt and `datastore-portability`
    prompts 02–03 did. Its **9 `QuadSourceIntegral_tags` rows** therefore became orphans. The second
    inventory names them in one `orphan-tag` problem on `QuadSourceIntegral`, shard #0, parent serial
    329386. Not a deviation: it is the orphan detection working on real data. It is recorded so
    that the one problem in the "after" inventory is not a surprise.

No UNINTENDED DRIFT. The script that inserted the 21 methods first ended one method early in
`wavenumber.py`, where `sqla_wavenumber_factory.inventory`'s signature spans three lines. `black`
refused to parse the result. The file was restored with `git checkout` and the insertion redone.
The shipped diff there is 30 insertions and 0 deletions, like every other factory's.

## Verification performed

- **Baselines at `f52a7df`** (from the orchestrator at dispatch): AdaptiveLevin 32, ComputeTargets
  552, CosmologyModels 39, Datastore 92, LiouvilleGreen 148 (1 skipped), RunRegistry 84.
- **After the change.** All six suites were run concurrently, each with
  `PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t .`:
  - AdaptiveLevin 32 OK;
  - ComputeTargets 552 OK (the wall-clock flake did not fire);
  - CosmologyModels 39 OK;
  - **Datastore 141 OK**, which is 92 + 49 new;
  - LiouvilleGreen 148 OK (skipped=1);
  - RunRegistry 84 OK.

  The new module takes about 43 s on its own, most of it in test 2's 83 variant stores. Each store
  costs about 0.2 s to build and read, mostly SQLAlchemy compiling statements anew for each fresh
  engine.
- **Scope.** `git diff --cached -- tools/ main.py extract_common.py config/ Datastore/SQL/ShardedPool.py RunRegistry/`
  is empty. Under `Datastore/SQL/ObjectFactories/`, the diff has 0 deletions: 382 lines added, all
  inside the 21 new `inventory_records` methods. No existing test module was modified.
- **`black --check`** is clean on all 23 Python files added or changed.

### §4 — the inventory of a copy of the sweep store

1. **Before.** `python -m RunRegistry list`: 7 runs, **none `running`** (5 finished, 2 `unknown`
   pre-registry). 72 GB free. The originals were snapshotted read-only: every file under
   `var/datastores/` and `backup-pre-resume-20260921T091011/`, 18 files. For each it recorded the
   listing, `lstat` size and `st_mtime_ns`, the SHA-256 of the bytes, and, per `.sqlite`, every
   table's `COUNT(*)` through stdlib `sqlite3` `mode=ro`. The primaries' and sidecars' prefixes
   are as prompt 01 recorded them: primaries A3 `fdbe93a5` (live and backup) and sweep
   `c3cda49d`; sidecars A3 `f7220408`, sweep `a595b7c9`, backup `ce7b476a`.
2. `cp -p` of `handover-atol-sweep.sqlite` and its four shards into
   `var/store-fingerprint-check-02/`, then the same snapshot of the copy.
3. **A census of every key leaf and reference column** (`typeof()`, per shard, stdlib `sqlite3`
   `mode=ro`), taken before the inventory, for the §7 float stop condition:
   - every REAL leaf holds only `real`, and every integer leaf and reference only `integer`;
   - every text leaf holds only `text`;
   - there are no NULLs;
   - **no column holds two storage classes.** `float.hex` is unambiguous on this store.
4. **`read_inventory` on the copy.** `inventory.problems == ()`: **no problem of any kind.**

| Class | Where | Records | Table rows (per shard) | Tag sets (count) | validated | Σ value_count |
|---|---|---:|---|---|---|---:|
| `version` | repl | 1 | 1 (1/1/1/1) | – | – | – |
| `store_tag` | repl | 10 | 10 (10/10/10/10) | – | – | – |
| `redshift` | repl | 1 740 | 1 740 (×4) | – | – | – |
| `wavenumber` | repl | 8 | 8 (×4) | – | – | – |
| `tolerance` | repl | 13 | 13 (×4) | – | – | – |
| `LambdaCDM` | repl | 1 | 1 (×4) | – | – | – |
| `QCD_Cosmology` | repl | 1 | 1 (×4) | – | – | – |
| `IntegrationSolver` | repl | 7 | 7 (×4) | – | – | – |
| `GkSourcePolicy` | repl | 2 | 2 (×4) | – | – | – |
| `QuadSourcePolicy` | repl | 2 | 2 (×4) | – | – | – |
| `wavenumber_exit_time` | repl | 8 | 8 (×4) | – | – | – |
| `BackgroundModel` | repl | 1 | 1 (×4) | **B** (1) | True 1 | 1 740 |
| `TkNumericIntegration` | k | 8 | 8 (2/2/2/2) | **T** (8) | True 8 | 3 865 |
| `TkWKBIntegration` | k | 8 | 8 (2/2/2/2) | **T** (8) | True 8 | 9 627 |
| `GkNumericIntegration` | k | 4 549 | 4 549 (1 188/1 089/985/1 287) | **G** (4 549) | True 4 549 | 146 445 |
| `GkWKBIntegration` | k | 13 920 | 13 920 (3 480 ×4) | **G** (13 920) | True 13 920 | 916 937 |
| `GkSource` | k | 1 160 | 1 160 (290 ×4) | **G** (1 160) | True 1 160 | 1 016 160 |
| `GkSourcePolicyData` | k | 1 160 | 1 160 (290 ×4) | – | – | – |
| `QuadSource` | q | 36 | 36 (10/8/6/12) | **T** (36) | True 36 | 17 178 |
| `QuadSourceIntegral` | k | 7 706 | 7 706 (1 955/1 892/1 920/1 939) | **Q** (7 706) | – | – |
| `OneLoopIntegral` | k | 0 | 0 (×4) | – | – | – |

   Every class's record count equals its table's row count: one shard's count for a replicated
   class (identical on all four), and the sum for a sharded one. Every class that has tags has
   **exactly one tag set**:
   - **B** = `LargestSourceRedshift_2.0636e+16`, `Run_default`, `SmallestSourceRedshift_0.1`,
     `SourceGridConstruction_2`;
   - **T** = **B** + `OutsideHorizonEfolds_e3`, `SourceRedshiftGrid_1740_8d1d43b6`,
     `TkOneLoopDensity`;
   - **G** = **B** + `GkOneLoopDensity`, `ResponseRedshiftGrid_145_c51d43ac`,
     `ResponseSparsenessZ_12`, `SourceRedshiftGrid_1740_8d1d43b6`;
   - **Q** = **G** + `TkOneLoopDensity`.

   All 10 `store_tag` labels are carried by some record. Every validated flag is `True`: the store
   holds no unvalidated row.

   **The value counts.** Each value table's total equals the sum of its parents' `value_count`,
   and **there are no orphans**:
   - `BackgroundModelValue`: 1 740 per shard, replicated; Σ 1 740;
   - `TkNumericValue`: 3 865 (967/963/963/972);
   - `TkWKBValue`: 9 627 (2 354/2 457/2 561/2 255);
   - `GkNumericValue`: 146 445 (39 179/33 876/28 525/44 865);
   - `GkWKBValue`: 916 937 (227 083/231 942/235 866/222 046);
   - `GkSourceValue`: 1 016 160 (254 040 ×4);
   - `QuadSourceValue`: 17 178 (4 807/3 681/2 658/6 032).

   **The replicated classes agree across all four shards.** For all 12 of them, each shard's
   records are the same multiset of key, tags, validated and value count. That includes
   `BackgroundModel` with its tags and its 1 740 `BackgroundModelValue` count.
   `[00-replicated-writes-can-diverge-across-shards]` shows **no divergence on this store**.

   **Three `QuadSourceIntegral` records**, in full, with their parents resolved from the listing.
   Each record's key has nine fields, every one a 64-hex digest, and the nine tags **Q**, with
   `validated` and `value_count` both `null`. The first is:
   ```json
   {"key": {"atol": "19cba87dbdc955cc07646fbfec0de3966e7e11ef2c7e746f9f3ce84b52e74ff3",
            "k": "266b1f8f5fd724a698c898bec6333f03a0778fdac0f583c7fd987f5bfce1e262",
            "model": "ea6c118b1a40e9bf83ad0fea6e3e14470e3455b7918eb0b0f0be6aa79350811e",
            "policy": "638ce4c02948538352efe782dc3159569f434682a8cf94a5f2a08d65831847ae",
            "q": "14e4d17ec2eca7e5e89807504a257c335bd6dc174bd324595f45517cf588523d",
            "r": "14e4d17ec2eca7e5e89807504a257c335bd6dc174bd324595f45517cf588523d",
            "rtol": "52d4126bfe13aa7eabbf3eeb690aef5a386ba0089a3f3244b06ff0e6f5c0153c",
            "z_response": "0eac33cf1377ecf7b793bb997eecf309270d24ae824bd704871c7f1add71abf0",
            "z_source_max": "6a551231cf053ce0a41188de1c074ba1985da3eabdabf7a026d089fb7f807a6f"},
    "tags": ["GkOneLoopDensity", "LargestSourceRedshift_2.0636e+16", "ResponseRedshiftGrid_145_c51d43ac",
             "ResponseSparsenessZ_12", "Run_default", "SmallestSourceRedshift_0.1",
             "SourceGridConstruction_2", "SourceRedshiftGrid_1740_8d1d43b6", "TkOneLoopDensity"],
    "validated": null, "value_count": null}
   ```
   Resolved through `StoreInventory.resolve`. Decimals are `float.fromhex` of the stored hex, and
   $k$, $q$, $r$ are in 1/Mpc:

   | | $k$ | $q$ | $r$ | $z_{\rm response}$ | $z_{\rm source,max}$ | log10 atol | log10 rtol | key digests (k / q / r) |
   |---|---|---|---|---|---|---|---|---|
   | 1 | 3.0455e7 (`0x1.d0b4b0658bd19p+24`) | 3e8 (`0x1.1e1a300000002p+28`) | 3e8 | 4 800.13 (`0x1.2c020ca92109dp+12`) | 1.63911e16 (`0x1.d1dcab63d0a80p+53`) | −22 | −8 | `266b1f8f…` / `14e4d17e…` / `14e4d17e…` |
   | 2 | 9.5585e7 (`0x1.6ca0b4a164028p+26`) | 9.85061e5 (`0x1.e0fca692f93eap+19`) | 9.5585e7 | 7.62741e7 (`0x1.22f66fea09b70p+26`) | 1.63911e16 | −32 | −8 | `4123eec0…` / `851ef8bd…` / `4123eec0…` |
   | 3 | 1e5 (`0x1.86a0000000000p+16`) | 9.5585e7 | 9.5585e7 | 4.02202e15 (`0x1.c94011e0b013cp+51`) | 1.63911e16 | −32 | −8 | `904ac962…` / `4123eec0…` / `4123eec0…` |

   The three share their other parents:
   - `model` `ea6c118b…`: `BackgroundModel` on `LambdaCDM` (`cosmology_type` 0), Gauss orders
     4/4/4, `source_grid_construction` 2, `source_grid_digest` `8d1d43b6`, `z_init` 0.1, tags
     **B**;
   - `policy` `638ce4c0…`: `GkSourcePolicy` (`Levin_threshold` 1.5, `maximize-WKB`);
   - each `wavenumber_exit_time` parent has `stepping` 0, `LambdaCDM`, and log10 atol / rtol −10 / −9.

   Where two records reference the same physical parent, they carry the same digest: record 2's
   $k$ and $r$, and record 3's $q$ and $r$, are all `4123eec0…`.

   **F7 — size, time, memory.**
   - **30 341 records** in all.
   - The full listing, as compact JSON of every record, is **19.3 MB**. That is the on-demand
     listing; prompt 04 digests it and keeps no records.
   - **`read_inventory` wall time: 6.8 s**, and 7.3 s on the second run. The whole process,
     imports included, took 10.3 s (`/usr/bin/time`).
   - **Peak resident set: 228 MB** (`ru_maxrss`; `/usr/bin/time -l` reports 260 MB for the whole
     process). For comparison, a process that only imports the factory map, the reader and the
     inventory peaks at **175 MB**, because the factories import numpy, scipy and ray. So the
     inventory itself adds about 50 MB. No `*Value` row is ever fetched: the statements that name
     a value table are `GROUP BY` counts only (test 6), one per value table per shard.
5. **One discriminating deletion.** On the copy's shard 0 only, with `sqlite3`:
   `DELETE FROM QuadSourceIntegral WHERE serial = 329386`. The row's label is
   `handover-atol-sweep-a1e-32-r1e-08-QuadSourceIntegral-k3.05e+07-q3.05e+07-r3.05e+07-zresponse0.1-2026-09-24T00:17:34`.
   The shard's `QuadSourceIntegral` count went from **1 955 to 1 954**. After the deletion, the copy
   was snapshotted again. The second `read_inventory` left all five files byte-identical to that
   snapshot. Shards 1–3 and the primary were identical to the pre-inventory snapshot of the copy
   throughout, and shard 0's only count change is `QuadSourceIntegral` 1 955 → 1 954.
   - **`QuadSourceIntegral`: 7 706 → 7 705 records. Exactly one is gone, and none is new.**
   - **Its key**, resolved from the "before" listing:
     - $k = q = r = 3.0454960396664713\times10^{7}$ /Mpc, all three the same
       `wavenumber_exit_time` digest `266b1f8f…`;
     - $z_{\rm response} = 0.1$;
     - $z_{\rm source,max} = 1.6391061605324032\times10^{16}$;
     - log10 atol $= -32$ and log10 rtol $= -8$;
     - policy `GkSourcePolicy` (1.5, `maximize-WKB`), and model `ea6c118b…` as above.

     These are the label's `k3.05e+07-q3.05e+07-r3.05e+07-zresponse0.1` and `a1e-32-r1e-08`.
   - **Its tag set** is **Q**.
   - **Every other class is unchanged, record for record**: the canonical JSON of every record of
     the other 20 classes is identical before and after.
   - **The one problem** is `orphan-tag: QuadSourceIntegral: 9 QuadSourceIntegral_tags row(s) on
     shard #0 name a QuadSourceIntegral row that is not there; e.g. parent serials 329386`. These
     are the deleted row's own tag rows (deviation 10).
6. **The originals were re-snapshotted: identical** (`cmp` of the two JSON snapshots).
7. **`var/store-fingerprint-check-02/` was deleted.** `var/` holds `bootstrap-a3-resume.log`,
   `datastores` and `runs`, as before. `RunRegistry list` is unchanged: 7 runs, none `running`.
   Nothing from §4 is committed. The scripts are in the session scratchpad: `impl02_snapshot.py`,
   `impl02_typeof.py` and `impl02_demo.py`.

## The deliberate-breakage record

Each mutation was applied to the working tree with every change of this prompt staged, so each
diff is `git diff` (working tree against the index) and applies with `git apply` to this commit.
`Datastore.tests.test_store_inventory` was run under each, and the file was then restored with
`git checkout -- <file>`, after which `git diff --quiet` was true. No mutation is committed. The
other Datastore test modules do not call the inventory.

### (i) One compute target references its model by serial instead of by digest

`TkNumericIntegration`'s `model_serial` becomes a leaf.

```diff
diff --git a/Datastore/SQL/ObjectFactories/TkNumericIntegration.py b/Datastore/SQL/ObjectFactories/TkNumericIntegration.py
index aae6aa8..8da5f7d 100644
--- a/Datastore/SQL/ObjectFactories/TkNumericIntegration.py
+++ b/Datastore/SQL/ObjectFactories/TkNumericIntegration.py
@@ -699,9 +699,8 @@ class sqla_TkNumericIntegration_factory(SQLAFactoryBase):
             table,
             tables,
             context,
-            leaves=("break_point_kind",),
+            leaves=("break_point_kind", "model_serial"),
             parents={
-                "model": Parent("model_serial", "BackgroundModel"),
                 "k": Parent("wavenumber_exit_serial", "wavenumber_exit_time"),
                 "atol": Parent("atol_serial", "tolerance"),
                 "rtol": Parent("rtol_serial", "tolerance"),
```
**Failed** (5 failures, 1 error, 49 run):
- `TestKeysArePhysical.test_different_serial_assignments` and
  `test_replicated_serials_permuted_and_rows_moved`, for `TkNumericIntegration`;
- `TestIdentityColumnsMatter.test_the_key_fields_are_the_lookup_columns` (`TkNumericIntegration`),
  and ERROR `test_each_identity_column_changes_the_records` (`TkNumericIntegration`, `model`:
  no `model` field);
- `TestTags.test_a_tag_on_the_background_model_reaches_its_children`, because a tag on the
  parent no longer reaches the child;
- `TestOldStoresAndOrphans.test_a_reference_that_cannot_be_resolved`, because serial 999 is now
  just a number.

### (ii) Tags are dropped from records

```diff
diff --git a/Datastore/store_inventory.py b/Datastore/store_inventory.py
index 3edb821..0d84b52 100644
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -554,7 +554,7 @@ def read_records(
         record = Record(
             key=key,
             tags=(
-                tuple(sorted(tag_sets.get(row.serial, ()))) if tags is not None else ()
+                ()
             ),
             validated=(
                 (None if mapping["validated"] is None else bool(mapping["validated"]))
```
**Failed** (15 failures):
- `TestTags`: `test_adding_a_tag_changes_that_record_and_nothing_else`,
  `test_a_tagged_parents_digest_covers_its_tags`,
  `test_a_tag_on_the_background_model_reaches_its_children` and
  `test_oneloop_tags_come_from_its_own_table`;
- `TestTheFullStore.test_classes_with_tags_validated_and_values`, for all nine tagged classes;
- `TestReplicatedDivergence.test_a_changed_tag_set`;
- `TestDuplicates.test_other_tags_are_not_a_duplicate`.

### (iii) `canonical` rounds floats to 12 significant figures

```diff
diff --git a/Datastore/store_inventory.py b/Datastore/store_inventory.py
index 3edb821..79745dc 100644
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -146,7 +146,7 @@ def canonical(value: Any) -> Any:
     if value is None or isinstance(value, (bool, int, str)):
         return value
     if isinstance(value, float):
-        return float.hex(value)
+        return float.hex(float(f"{value:.12g}"))
     raise TypeError(
         f"canonical(): no canonical form for a leaf of type {type(value).__name__!r} ({value!r})"
     )
```
**Failed** (2 failures):
- `TestFloats.test_canonical`, where `0.1` and its next float map alike;
- `TestFloats.test_the_last_bit_of_k_changes_the_records`.

### (iv) Replicated classes are read from one shard only

Every shard is still read, because each shard's sharded rows resolve against that shard's
replicated rows. Only the lowest shard's records are kept, and no other shard is looked at.

```diff
diff --git a/Datastore/store_inventory.py b/Datastore/store_inventory.py
index 3edb821..35344cf 100644
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -594,7 +594,7 @@ def _combine(
     first = reads[min(reads)]
 
     if replicated:
-        comparable = [s for s in sorted(reads) if reads[s].comparable]
+        comparable = [s for s in sorted(reads) if reads[s].comparable][:1]
         contributing = comparable[:1]
         if len(comparable) > 0:
             base = comparable[0]
```
**Failed** (3 failures, 1 error): every `TestReplicatedDivergence` test.
- `test_a_changed_row`, `test_a_changed_tag_set` and `test_a_changed_value_count` fail, because no
  divergence is named.
- `test_the_right_shard_of_three` errors, because there is no problem to unpack.

### (v) `OneLoopIntegral` tags are read from `QuadSourceIntegral_tags`

```diff
diff --git a/Datastore/SQL/ObjectFactories/OneLoopIntegral.py b/Datastore/SQL/ObjectFactories/OneLoopIntegral.py
index b145bbc..f68fab4 100644
--- a/Datastore/SQL/ObjectFactories/OneLoopIntegral.py
+++ b/Datastore/SQL/ObjectFactories/OneLoopIntegral.py
@@ -291,7 +291,7 @@ class sqla_OneLoopIntegral_factory(SQLAFactoryBase):
                 "atol": Parent("atol_serial", "tolerance"),
                 "rtol": Parent("rtol_serial", "tolerance"),
             },
-            tags=("OneLoopIntegral_tags", "parent_serial"),
+            tags=("QuadSourceIntegral_tags", "parent_serial"),
         )
 
 
```
**Failed** (19 failures):
- `TestTags.test_oneloop_tags_come_from_its_own_table`;
- `TestKeysArePhysical`, all three, for `OneLoopIntegral`, because the joined tags follow
  `QuadSourceIntegral` serials;
- `TestTheFullStore.test_the_full_store_has_no_problem`, where `QuadSourceIntegral_tags` rows whose
  parent is no `OneLoopIntegral` are named `orphan-tag`;
- through that same stray problem, the tests that count every problem in the store: 11 of the 12
  in `TestOldStoresAndOrphans` (all but `test_an_unknown_cosmology_type`, which reads only its own
  class's problems), `test_a_changed_row` and `test_a_changed_value_count` in
  `TestReplicatedDivergence`, and `test_other_tags_are_not_a_duplicate`.

### (vi) `z_response` is left out of `GkSource`'s key

```diff
diff --git a/Datastore/SQL/ObjectFactories/GkSource.py b/Datastore/SQL/ObjectFactories/GkSource.py
index 831209d..1f2b28e 100644
--- a/Datastore/SQL/ObjectFactories/GkSource.py
+++ b/Datastore/SQL/ObjectFactories/GkSource.py
@@ -650,7 +650,6 @@ class sqla_GkSource_factory(SQLAFactoryBase):
             parents={
                 "model": Parent("model_serial", "BackgroundModel"),
                 "k": Parent("wavenumber_exit_serial", "wavenumber_exit_time"),
-                "z_response": Parent("z_response_serial", "redshift"),
             },
             tags=("GkSource_tags", "parent_serial"),
             values=("GkSourceValue", "parent_serial"),
```
**Failed** (2 failures):
- `TestIdentityColumnsMatter.test_the_key_fields_are_the_lookup_columns` (`GkSource`);
- `test_each_identity_column_changes_the_records` (`GkSource`, `z_response`).

### (vii) The digest of a tagged parent leaves out its tags

```diff
diff --git a/Datastore/store_inventory.py b/Datastore/store_inventory.py
index 3edb821..da64837 100644
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -184,7 +184,7 @@ def reference_digest(key: Mapping[str, Any], tags: Sequence[str], tagged: bool)
     because two rows of a tagged class can differ only in their tags.
     """
     if tagged:
-        return digest({"key": key, "tags": list(tags)})
+        return digest(key)
     return digest(key)
 
 
```
**Failed** (2 failures):
- `TestTags.test_a_tagged_parents_digest_covers_its_tags`, where `GkSourcePolicyData` no longer
  changes;
- `TestTags.test_a_tag_on_the_background_model_reaches_its_children`.

### (viii) `value_count` counts every value row, not the parent's own

```diff
diff --git a/Datastore/store_inventory.py b/Datastore/store_inventory.py
index 3edb821..75cf6a5 100644
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -561,7 +561,7 @@ def read_records(
                 if has_validated
                 else None
             ),
-            value_count=counts.get(row.serial, 0) if values is not None else None,
+            value_count=sum(counts.values()) if values is not None else None,
         )
         out.append(
             (row.serial, record, mapping["timestamp"] if has_timestamp else None)
```
**Failed** (8 failures):
- `TestValueCounts.test_the_counts_are_per_parent`, for `BackgroundModel`, `TkNumericIntegration`,
  `TkWKBIntegration` and `GkSource`;
- `test_deleting_a_value_row_lowers_one_count_by_one`, for `BackgroundModelValue`,
  `TkNumericValue`, `TkWKBValue` and `GkSourceValue`.

The three classes that passed hold one parent per shard in the fixture, so a total and a
per-parent count coincide there. Each failing class holds two parents on one shard.

## Observations not acted on

- **`[02-exit-time-lookup-runs-inside-the-subhorizon-loop]`** (opened on the board's §3). In
  `sqla_wavenumber_exit_time_factory.build`, the line
  `row_data = conn.execute(query).one_or_none()` (`wavenumber.py:270` in this tree, `:261` at
  `f52a7df`) is indented inside `for z_offset in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:` (`:267`).
  - Each lookup therefore runs the query six times, and the first five lack some of the
    sub-horizon columns. The last iteration's result is complete, so the answer is right today.
  - If that list were ever empty, `row_data` would be unbound.
  - `build` is out of this campaign's scope (README §1).
- **The reference digest does not cover `validated`.** An unvalidated row and a validated row with
  the same key and tags have one digest, and are a `duplicate`. That follows the prompt's
  definition, and the sweep copy has no unvalidated row. It is noted for prompt 04, which digests
  records and so does see the flag.
- **`TkWKBIntegration.T_init` / `Tprime_init` and `GkWKBIntegration.G_init` / `Gprime_init`** are
  stored, non-nullable initial conditions that `build()` takes in its payload and never filters on.
  So they are not in the key, as the lookup rule requires. They are fixed by the model, $k$ and
  $z_{\rm init}$ in every caller read, so no issue is opened. `QuadSourceIntegral.b` is the same
  case.
- **The full listing of the sweep store is 19.3 MB**, almost all of it 64-hex parent digests
  (about 9 per `QuadSourceIntegral`). It is generated on demand and never stored (README §6.1
  point 7). If prompt 03's display wants it smaller, it can resolve digests to labels as it
  renders.
- **Suite time.** `Datastore/tests` goes from 16.7 s (the 92 prior tests, measured on this tree
  without the new module) to 53.7 s. The new module is about
  43 s of that, most of it test 2's 83 variant stores. Each store is cheap to build now. What
  remains is SQLAlchemy compiling each statement anew for every fresh engine.
- **The stale sentence in `docs/OPEN_ISSUES.md`'s preamble** ("Of the 90 above, 87 …"), which
  prompt 01's log recorded, is still there. It predates this campaign and is prose, not a row, so
  it was left alone.

## State handed to the next prompt

- `from Datastore.store_inventory import read_inventory`. `inv = read_inventory(primary)`.
  - `inv.classes` is ordered by `INVENTORY_CLASSES`. `inv[name]` has `.records`, `.count`,
    `.earliest_timestamp`, `.latest_timestamp`, `.problems`, `.replicated`, `.tagged` and
    `.parents`.
  - A `Record` has `.key`, `.tags`, `.validated`, `.value_count` and `.as_json()`.
  - `inv.resolve(name, record)` expands the parents, for display. `inv.problems` gathers every
    named problem.
- **For prompt 03 (the display):**
  - `BackgroundModel` records carry `source_grid_digest` and `source_grid_construction`, the
    other half of `[03-qcd-inventory-does-not-report-the-representation]`.
  - The `store_tag` class lists every label, which is what `available_run_labels` needs.
  - The old `inventory()` methods, `ShardedPool.inventory`, `_merge_queue` and `inventory_config`
    are untouched.
- **For prompt 04 (the fingerprint):**
  - `canonical_json(record.as_json())` is the canonical record. Records are already sorted by it
    within each class.
  - Tag sets are `record.tags`.
  - No timestamp is in any record.
  - `reference_digest` is the one digest rule a parent reference uses.
- `Datastore/tests/real_store_fixtures.py`: `build_full_store`, `full_rows`, `relabel_serials`,
  `vary_row`, `find_row` and `fill_required`.
- The `Datastore` baseline is now **141**.
