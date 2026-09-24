# Prompt 02 — a structured inventory: work items by physical label, with their tags

**Campaign:** [`README.md`](README.md) · **Board items:** **F4**–**F7** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** nothing. The consumers switch over in prompt 03, and the fingerprint is prompt 04.
**Opens:** anything out of scope that you find (§6), **without fixing it**.
**Decision D1** (README §6.2, the user, 2026-09-24): a float's canonical form is its stored bits,
as `float.hex`.
**Recommended model:** **Opus**. The judgement is in the key of each of about twenty classes: every
column that decides identity must be in it, and nothing that is store-local may be.

**Read first:**

1. [`README.md`](README.md) in full, especially §4, §6.1 and §6.3.
2. [`docs/store-fingerprint-audit.md`](../../docs/store-fingerprint-audit.md) in full. **§4 is the
   starting point for every key below. Verify it against the code; do not copy it.**
3. Prompt 01 and its log, [`logs/01-a-read-only-store-reader.md`](logs/01-a-read-only-store-reader.md):
   `open_read_only`, `build_schema` and `Datastore/tests/real_store_fixtures.py`.
4. Every factory in `Datastore/SQL/ObjectFactories/`, for three things: its `register()` columns,
   the lookup its `build()` runs (the columns it filters on and the tags it joins), and its current
   `inventory()`. Also `config/sharding.py`, and `CosmologyModels/model_ids.py` (the
   `cosmology_type` values).
5. `tools/inventory_report.py` and `extract_common.available_run_labels`. These are the consumers
   prompt 03 will switch. **You do not change them.**
6. `CLAUDE.md`: "Repository mechanics", and the author conventions, which are not defects.

---

## 1. What is wanted

A structured inventory of a closed store, read through prompt 01's reader, that says **what the
store holds** in terms that do not depend on how it was built:

- **Every work item is named by physical labels**: wavenumbers, redshifts, tolerances, cosmology
  parameters, orders, kinds, digests. It is never named by a serial, because serials are leased,
  pooled and recycled, and two stores built independently need not share one (audit §4).
- **Every work item carries its full tag set**, because tags are part of lookup identity and label
  the grid a product was computed on (README §6.1 point 5).
- **The `*Value` tables are counted, never listed** (§6.1 point 3). Each parent carries the number
  of value rows that hang off it.
- **`QuadSourceIntegral`, `OneLoopIntegral` and `GkSourcePolicyData` have real records** (§6.1
  point 6).

This prompt adds the service. It changes no consumer and retires nothing. The existing
`inventory()` methods, `ShardedPool.inventory` and `inventory_config` stay exactly as they are
until prompt 03, so the tree works at this commit.

---

## 2. What to change

### F4 — `Datastore/store_inventory.py`, `read_inventory(primary)`, and one record builder per class

**The driver.** `read_inventory(primary)` opens the store with `open_read_only` and builds every
class in **dependency order**. Leaves come first: `version`, `store_tag`, `redshift`, `wavenumber`,
`tolerance`, the cosmologies, `IntegrationSolver` and the policies. Then `wavenumber_exit_time` and
`BackgroundModel`. Then the compute targets. It returns the structured inventory, and for each
class it holds:
- `records`;
- `count`;
- the earliest and latest `timestamp`, kept for display only (§6.1 point 7 keeps timestamps out of
  any digest);
- `problems`, a list of named strings.

**Who builds a record.** *Prompt's choice:* **each factory owns its record builder**, as a new
static method beside its existing `inventory()`. The factories are the inventory service layer,
and a factory is where its table's identity is defined. The driver supplies each builder with the
connection, the tables, and a map from serial to canonical key for every class already built on
that shard. The old `inventory()` is not touched.

**A record** is a small JSON-safe object:

- **`key`**: the class's physical identity. It is built only from canonical leaves (below) and
  references to parents. **No serial, label, name or timestamp appears in it**, except where the
  label *is* the identity (`version`, `store_tag`, `IntegrationSolver`).
- **`tags`**: the sorted tuple of `store_tag` labels from the class's `*_tags` association rows,
  joined inside the shard. It is empty where the class has no association table. *Prompt's
  choice:* the `Run_<label>` tag stays in (README §6.3), because it is part of the lookup.
- **`validated`**: the row's flag, or `None` where the class has no such column. *Prompt's choice:*
  unvalidated rows are recorded, because they are content.
- **`value_count`**: the number of rows in the class's `*Value` table whose parent is this row, or
  `None` where the class has none.

Design the record so that a computed-values field can be added later **outside** the key, without
changing what the key, tags, validated flag or value count are (§6.1 point 2). Do not add values
now.

**References to parents.** *Prompt's choice:* **a parent is referenced by the digest of its own
canonical key** (SHA-256 of the key's canonical JSON), never by its serial. The parent itself is
listed in its own class. A child record stays small, and a listing can still be joined by digest.
The canonical JSON is **one function**: sorted keys, no whitespace, and leaves through
`canonical`.

**Floats (D1).** **One function, `canonical(value)`**, turns every leaf into its canonical form:
- a float becomes `float.hex(value)`, **of the value as stored**. For example `log10_tol` is used
  as stored, never `10**log10_tol`;
- an integer, string, boolean or `None` is kept as it is.

The lookups match floats within 1e-7, but the key records the stored bits (README §6.2 D1, decided). No
other code converts a float for a key.

### F5 — the key of every class

Start from audit §4. **For every class, read its `build()` lookup and confirm that every column it
filters on is in the key.** That includes the optional filters `z_init`, `z_source` and
`z_response`: *prompt's choice*, always in the key (README §6.3). Also confirm that nothing
store-local is in it. Record the shipped key of every class in the log as a table. **Any
difference from audit §4 is a deviation**, with the lookup line that justifies it.

In outline (verify every entry):

| Class | Key (references to parents by digest) | tags | validated | value_count |
|---|---|---|---|---|
| `version`, `store_tag` | `label` | – | – | – |
| `redshift` | `z` | – | – | – |
| `wavenumber` | `k_inv_Mpc` | – | – | – |
| `tolerance` | `log10_tol` | – | – | – |
| `LambdaCDM` | the six parameters | – | – | – |
| `QCD_Cosmology` | the six, plus `log10_max_z` and `T_z_representation` | – | – | – |
| `IntegrationSolver` | `label`, `stepping` | – | – | – |
| `GkSourcePolicy`, `QuadSourcePolicy` | `Levin_threshold`, `numeric_policy` | – | – | – |
| `wavenumber_exit_time` | k, the cosmology (by type and digest), atol, rtol, `stepping` | – | – | – |
| `BackgroundModel` | the cosmology, the three Gauss orders, `source_grid_construction`, `source_grid_digest`, z_init | yes | yes | `BackgroundModelValue` |
| `TkNumericIntegration` | M, kx, atol, rtol, `break_point_kind`, z_init | yes | yes | `TkNumericValue` |
| `TkWKBIntegration` | M, kx, `rho_gauss_order`, `z_init` (REAL) | yes | yes | `TkWKBValue` |
| `GkNumericIntegration` | M, kx, atol, rtol, `break_point_kind`, z_source | yes | yes | `GkNumericValue` |
| `GkWKBIntegration` | M, kx, `rho_gauss_order`, z_source, `z_init` (REAL) | yes | yes | `GkWKBValue` |
| `GkSource` | M, kx, z_response | yes | yes | `GkSourceValue` |
| `GkSourcePolicyData` | its `GkSource` (by digest, tags included), its `GkSourcePolicy` | – | – | – |
| `QuadSource` | M, kx(q), kx(r) | yes | yes | `QuadSourceValue` |
| `QuadSourceIntegral` | M, `GkSourcePolicy`, kx(k), kx(q), kx(r), z_response, z_source_max, atol, rtol | yes | – | – |
| `OneLoopIntegral` | M, kx, z_response, atol, rtol | yes, from **`OneLoopIntegral_tags`** | – | – |

Points to settle and record:

- **A `GkSourcePolicyData` parent's digest must cover the `GkSource`'s tags**, because two
  `GkSource` rows can differ only in their tags. The same holds for any reference to a tagged
  parent. *Prompt's choice:* **the digest of a tagged parent covers its key and its tag set.**
- **`OneLoopIntegral` tags are read from `OneLoopIntegral_tags`**, which is what `store()` writes,
  whatever its `build()` joins (audit §6, `[00-oneloop-lookup-joins-the-wrong-tag-table]`, which
  you do not fix).
- **`wavenumber_exit_time`'s cosmology** is `cosmology_type` plus a serial with no foreign key.
  Resolve it through the type (`CosmologyModels/model_ids.py`).
- **The cosmology `name` and the policy `label`** are descriptive, and stay out of the key.
- **Solver identity is not in any compute target's key.** *Prompt's choice* (README §6.3): it is
  not in any lookup.

### F6 — shards, and what goes wrong

- **Sharded classes:** the union of the records from every shard.
- **Replicated classes:** read from **every** shard and compared on key, tags, validated flag and
  value count. *Prompt's choice:* the class's records are the lowest-serial shard's. **Any shard
  that differs is a named problem**, giving the shard, the number of differing records, and up to
  five example keys. This is how `[00-replicated-writes-can-diverge-across-shards]` becomes
  visible. The inventory reports the divergence and repairs nothing.
- **Duplicates.** Two records in one class with the same key and the same tags are a named problem,
  and both are kept. They should not exist, because the lookups' `one_or_none()` would fail on
  them.
- **Old stores.** A table absent from a shard reads as empty on that shard, and the absence is a
  named problem. A missing column that the key needs makes that class **incomplete** on that shard:
  a named problem, with none of that class's records from that shard. A missing column the key
  does not need is not a problem.
- **Orphans.** A `*Value` row whose parent is not in the store, a tag row whose parent or tag is
  missing, and a reference to a parent that cannot be resolved are each a named problem, counted,
  with up to five example serials. They are not records.

### F7 — the size stays small

The inventory holds one record per work item and per configuration row, and **no `*Value` row
ever**. The value counts come from one `GROUP BY` per value table, per shard. Record in the log,
for the §4 copy:
- the number of records per class;
- the wall time;
- the peak resident memory of the process.

---

## 3. Tests — in `Datastore/tests/`, no Ray, nothing under `var/`

Extend `real_store_fixtures.py` so that its store holds at least one row of **every** class, with
tags where the class has them, unvalidated rows, and value rows. Then:

1. **The keys are physical.** Build the same content twice:
   - with different serial assignments;
   - with the sharded rows on different shards;
   - with the replicated rows' serials permuted.

   The two inventories are **equal**, class for class and record for record. This is the test the
   whole campaign rests on.
2. **Every identity column matters.** For each class, build two stores that differ only in one
   identity column, once for each column in that class's key, including each optional filter. The
   records differ.
3. **Non-identity does not matter.** Two stores that differ only in these have equal records:
   - compute-target labels, timestamps and payload columns;
   - solver serials;
   - the cosmology `name`;
   - `version` foreign keys.
4. **Floats.** Two stores differing only in the last bit of one `k` give different records, and
   `canonical` is the only code that formats a float for a key.
5. **Tags.** Adding a tag row to one work item changes that record's `tags` and nothing else. The
   `OneLoopIntegral` tags come from its own association table.
6. **Value counts.** Deleting one value row lowers exactly one parent's `value_count` by one.
7. **Replicated divergence.** Changing a replicated row on one shard only produces a named problem
   that names that shard.
8. **Old stores and orphans.** Each case in F6 produces the named problem, and nothing else changes.
9. **Duplicates.** Two rows with the same key and tags produce the named problem.
10. **Read-only and no Ray.** Prompt 01's never-writes check holds across a full `read_inventory`.
    In a child interpreter, `ray.is_initialized()` stays false.

**Deliberate breakage.** Each of these mutations must fail the tests written against it. Record
each diff exactly as applied, with the tests that failed.

- (i) one compute target references its model by serial instead of by digest;
- (ii) tags are dropped from records;
- (iii) `canonical` rounds floats to 12 significant figures;
- (iv) replicated classes are read from one shard only;
- (v) `OneLoopIntegral` tags are read from `QuadSourceIntegral_tags`;
- (vi) `z_response` is left out of `GkSource`'s key;
- (vii) the digest of a tagged parent leaves out its tags;
- (viii) `value_count` counts every value row rather than the parent's own.

---

## 4. The demonstration — on a copy of the sweep store

As in prompt 01 §4: never an original, `cp -p` into `var/store-fingerprint-check-02/`, snapshot
the originals before and after, and delete the directory at the end.

1. Before: `python -m RunRegistry list` shows nothing `running`. Snapshot the originals.
2. `cp -p` the sweep store's five files. Snapshot the copy.
3. Run `read_inventory` on the copy. Record:
   - **per class:** the record count set against the table's row count (`sqlite3` `mode=ro`);
     the problems; and the number of distinct tag sets, with each tag set's labels and count;
   - **the value counts:** that each `*Value` table's total equals the sum of its parents'
     `value_count` (report any orphans);
   - **the replicated classes:** that they agree across all four shards, or the problems if not;
   - **three example records** from `QuadSourceIntegral`, in full, with their parents resolved
     from the listing;
   - **F7's size, time and memory.**
4. Delete **one** `QuadSourceIntegral` row from the copy's shard 0 with `sqlite3`: serial 329386,
   the row prompts 02 and 03 of `datastore-portability` used. Record its label, and the table's
   count before and after. Run `read_inventory` again and show:
   - that exactly one `QuadSourceIntegral` record is gone;
   - its key, named by physical labels ($k$, $q$, $r$, $z_{\rm response}$, tolerances) and its tag
     set;
   - that every other class is unchanged, record for record.
5. Re-take the snapshot of the originals. It must be identical.
6. Delete `var/store-fingerprint-check-02/`.

---

## 5. Acceptance

1. §3's tests exist and pass, with no Ray and nothing under `var/`. Every existing test module is
   unmodified.
2. The key table of every class is in the log, with every deviation from audit §4 justified by a
   lookup line.
3. The deliberate-breakage record, (i)–(viii).
4. §4 is done, with its numbers, and the working directory is deleted.
5. No consumer changed:
   `git diff HEAD~1 HEAD -- tools/ main.py extract_common.py config/ Datastore/SQL/ShardedPool.py RunRegistry/`
   is empty. No factory's `inventory()`, `build`, `store`, `read_batch` or
   `validate_on_startup` changed.
6. Every suite matches its baseline, except that `Datastore/tests` rises by exactly the tests you
   add. `black --check` is clean.
7. On the board: the §1 row for 02, items F4–F7, any issue opened, and §5's baselines. Also update
   `docs/OPEN_ISSUES.md` in the same commit.

---

## 6. What this prompt does not do

- It does not change `tools/inventory_report.py`, `main.py`, `extract_common.py`, `config/`,
  `ShardedPool` or any factory's existing `inventory()`. That is prompt 03.
- It does not compute a fingerprint, and it writes nothing into a sidecar or a run record. That is
  prompt 04.
- It does not add computed values to records.
- It does not fix the audit's §6 defects. It makes the replicated divergence and the orphans
  visible, and reads `OneLoopIntegral`'s tags from where they are stored.
- It does not change any lookup, key column or schema.

---

## 7. Stop conditions — stop and ask the user

- Anything in the store makes `float.hex` of a stored value ambiguous, for example a column that
  holds both floats and strings.
- A class's identity cannot be written in physical leaves without a serial, a store-local label or
  a timestamp.
- The §4 copy has two records in one class with the same key and tags (F6 "Duplicates"). Report
  them; do not choose between them.
- The §4 copy shows replicated divergence across its shards. Report it; that is a finding about
  the store, not about the code.
- Any key would need a change to a lookup, a schema or a factory's existing method.
- `read_inventory` writes anything, or needs Ray.

---

## 8. The log and the board

`logs/02-a-structured-inventory.md`, using README §5.1's template. In addition, record:
- the key table of every class as shipped;
- each *prompt's choice*, and whether it was kept;
- the §4 numbers.

On `IMPLEMENTATION_STATE.md`: the §1 row for 02; items F4–F7; any new §3 issues; §5's baselines.
Also update `docs/OPEN_ISSUES.md` in the same commit.
