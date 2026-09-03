# Backport audit: shared Ray/Datastore infrastructure modules

**Date:** 2026-09-03
**Auditor:** Claude (read-only analysis session; no code was written or modified)
**Purpose:** Input to a planning agent that will produce a backport implementation plan for
`SecondaryGWKit`.

---

## 1. Scope

Six reusable modules exist in three sibling codebases:

| Module | SecondaryGWKit | ChamPBH | StochasticInstantons |
|---|---|---|---|
| `RayTools/RayWorkPool.py` | 558 lines | 580 | 590 |
| `Datastore/SQL/Datastore.py` | 788 | 810 | 816 |
| `Datastore/SQL/ShardedPool.py` | 832 | 982 | 1001 |
| `Datastore/SQL/ClientPool.py` | 252 | 256 | 261 |
| `Datastore/SQL/ProfileAgent.py` | 340 | 355 | 355 |
| `Datastore/SQL/SerialPoolBroker.py` | 149 | 164 | 164 |

Repository paths:
- `SGWK` = `/Users/ds283/Documents/Code/SecondaryGWKit` (backport target, git branch `main`, clean at `68cff5d`)
- `CPBH` = `/Users/ds283/Documents/Code/ChamPBH`
- `SI`   = `/Users/ds283/Documents/Code/StochasticInstantons`

Method: full textual diffs of each module across all three trees, plus structural diffs of class
bodies only (to separate genuine logic drift from project-specific registry/config content), plus
inspection of call sites in each tree and of `SGWK` git history.

---

## 2. Executive summary

### 2.1 Is `StochasticInstantons` a superset of `ChamPBH`?

**Yes — confirmed, for all six modules.** There is no fix or enhancement in `ChamPBH` that is
absent from `StochasticInstantons`.

Evidence:
- `ProfileAgent.py` and `SerialPoolBroker.py` are **byte-identical** between `CPBH` and `SI`.
- `ClientPool.py` differs only in the project-specific `_default_serial_batch_size` table.
- `Datastore.py`: the class body from `@ray.remote` onwards is **byte-identical**; all differences
  are the project-specific factory imports, `_factories` registry, and `_drop_actions`/`_drop_order`.
- `ShardedPool.py`: the `class ShardedPool` body differs by exactly three changes, all of them
  present **only in `SI`** (items **F1**, **F5**, **F6** below).
- `RayWorkPool.py`: differs by exactly one change, present **only in `SI`** (item **E1**).

**Conclusion:** treat `StochasticInstantons` as the single upstream reference. `ChamPBH` needs to be
consulted only as evidence of provenance (most of the generic improvements originated there and were
carried forward unchanged).

### 2.2 State of `SecondaryGWKit`

`SGWK` is the *oldest* lineage of all three. It carries **four confirmed defects**, two of which
appear to make currently-committed code paths unrunnable:

- **B1** — silent, permanent data misrouting after restart (shard-key persistence). Highest severity.
- **B2** — `AttributeError` on every reopen of an existing sharded pool.
- **B3** — `ValueError` on `ShardedPool`/`Datastore` construction whenever `read_table_config` is
  supplied, which all seven `extract_*.py` scripts do.
- **B4/B5** — smaller latent faults.

`SGWK`'s git history dates B2 and B3 to two commits on 2025-12-15 (`a2bd966`, `53dc75d`). Both look
like refactors that were never exercised end-to-end. **A planning agent should ask the user to
confirm whether `main.py` / `extract_*.py` have actually been run against an existing sharded
datastore since 2025-12-15**, because if they have, one of these two analyses is wrong and the
finding needs re-derivation before any code is changed.

### 2.3 Recommendation summary

| ID | Item | Origin | Recommend | Effort |
|---|---|---|---|---|
| **B1** | `_assign_shard_keys` writes wrong PK column | SI | **Backport — urgent** | XS (+ data audit) |
| **B2** | `_read_shard_data` reads `row.key_attr` instead of `row.key_type` | CPBH→SI | **Backport** | XS |
| **B3** | `read_table_config` machinery is broken; replace with `read_table()` service | CPBH→SI | **Backport** | M (touches 7 scripts) |
| **B4** | `RayWorkPool` task builder returning `None` | CPBH→SI | **Backport** | XS |
| **B5** | `_assign_shard_keys` duplicate-`store_id` dedup | SI | **Backport** | XS |
| **F6** | Empty-list guard on `sharded_tables` insert | SI | **Backport** | XS |
| **E1** | `store_handler` / `persist_handler` split in `RayWorkPool` | SI | **Backport with care** | M (35 call sites) |
| **F2** | `inventory()` service (Datastore + ShardedPool + merge policies) | CPBH→SI | **Defer** | L (needs per-factory work) |
| **F3** | `object_get("version", …)` by name rather than class | CPBH→SI | Optional | XS |
| **F4** | Apache-2.0 licence headers | CPBH→SI | Optional, separate change | XS |
| **X1** | `object_read_batch` object-semantics guard | CPBH→SI | **Do not backport** (broken upstream) | — |
| **X2** | `object_get_vectorized` shard-key handling changes | SI | **Do not backport** (breaks SGWK callers) | — |
| **X3** | Dropping the `read_table_config is not None` guard | SI | **Do not backport** (regression) | — |
| **X4** | Registering factory *instances* rather than classes | CPBH→SI | **Do not backport** (project-level design) | — |
| **D1–D4** | Latent defects present in all three trees | — | Fix opportunistically | XS each |

---

## 3. Findings — bug fixes to backport

### B1. `ShardedPool._assign_shard_keys` inserts into a non-existent column

- **Module:** `Datastore/SQL/ShardedPool.py`
- **SGWK location:** table definition at `ShardedPool.py:248-259` (PK column is `key_serial`);
  offending insert at `ShardedPool.py:801-804` (`{"key_id": item.store_id, …}`)
- **Present in:** SGWK ✗ (bug), CPBH ✗ (bug), SI ✓ (fixed)
- **Category:** correctness — silent data loss
- **Severity:** **Critical**

SQLAlchemy silently discards the unknown key `key_id`; the INSERT binds only `shard_id`, and SQLite
auto-assigns the primary key. The in-memory map `self._shard_keys[item.store_id] = new_shard` is
correct, so a single run behaves perfectly. On the *next* run `_read_shard_data` (`ShardedPool.py:437-444`,
which correctly reads `key.key_serial`) reconstructs a *different* map whenever any shard key object
was processed out of `store_id` order inside `_assign_shard_keys`. Every record written under a
displaced key is then queried from the wrong shard and is permanently unfindable.

`StochasticInstantons` documents this in `shard-key-assignment-bug.md` (worth reading in full;
it includes a reproduction snippet and an observed case of ~5,000 orphaned records). `SGWK`'s shard
key is `wavenumber` (`config/sharding.py`), created through the same vectorized replicated-table
path, so the exposure is identical.

**Upstream fix (SI `ShardedPool.py:816-846`):** use `{"key_serial": …}`, capture
`result.inserted_primary_key[0]`, and print a `!! _assign_shard_keys MISMATCH` warning if it differs
from `item.store_id`.

**Backport recommendation: YES, first, as a standalone commit.** Additional work the plan must
include:

1. Port the mismatch assertion as well — it is the only cheap ongoing detector.
2. **Data audit step.** Existing `SGWK` sharded datastores may already contain a corrupt `shard_keys`
   table. Add a one-off read-only check that joins `shard_keys.key_serial` against the `wavenumber`
   table's serials and reports any row where the recorded `key_serial` cannot be a real wavenumber
   `store_id`, or where the count of `shard_keys` rows differs from the count of `wavenumber` rows.
   There is no safe repair without knowing the original insertion order, so the remedy for a
   corrupted store is to rebuild it — the plan should say so explicitly rather than attempting a fix-up.
3. The fix is behaviour-preserving in the common sequential case, so it is safe to apply
   unconditionally.

### B2. `ShardedPool._read_shard_data` reads a column it did not select

- **SGWK location:** `ShardedPool.py:333-350`; specifically `row.key_attr` at line 342, and again in
  the message strings at lines 344 and 349
- **Present in:** SGWK ✗ (bug), CPBH ✓ (fixed), SI ✓ (fixed)
- **Category:** correctness — hard failure
- **Severity:** High

The `select` at lines 334-337 requests only `self._shard_key_config_table.c.key_type`, but line 342
tests `row.key_attr`. SQLAlchemy `Row` raises `AttributeError` for an unselected name. Line 342 sits
in the `num_config == 1` branch, which is taken on the first row of every reopen of an existing
sharded pool — so this path cannot succeed as written.

Introduced by `a2bd966` ("Abstract out configuration of sharded and replicated tables from
ShardedPool", 2025-12-15), which renamed the column but not this reader.

**Upstream fix:** `row.key_type` (SI `ShardedPool.py:357`).

**Backport recommendation: YES.** Note that upstream fixed only the comparison — the two f-strings
at SI lines 359 and 365 still interpolate `row.key_attr` and will raise `AttributeError` from inside
the error path. **Fix all three occurrences in `SGWK`, not just the one upstream fixed.**

### B3. The `read_table_config` method-generation machinery is broken in both `Datastore` and `ShardedPool`

- **SGWK locations:**
  - `Datastore/SQL/Datastore.py:226-237`
  - `Datastore/SQL/ShardedPool.py:188-201`
  - consumer: `Datastore/SQL/Datastore.py:740-763` (`_generic_read_table`),
    `Datastore/SQL/ShardedPool.py:815-832` (`_generic_read_table`)
  - config: `config/sharding.py:37-40`
- **Present in:** SGWK ✗ (bug), CPBH ✓ (redesigned), SI ✓ (redesigned)
- **Category:** correctness — hard failure, plus API design
- **Severity:** High

Three independent faults in the same block:

1. `for method_name, method_config in read_table_config:` iterates a `dict`, which yields **keys**
   (strings). Unpacking `"read_wavenumber_table"` into two names raises
   `ValueError: too many values to unpack (expected 2)`. **Verified experimentally.** This fires
   during construction whenever `read_table_config` is not `None`.
2. `setattr(self, method_name, wrapper)` binds a plain function to the *instance*, so `self` is never
   passed; `pool.read_wavenumber_table(units=…)` would bind `units` to the `self` parameter.
3. `wrapper` closes over the loop variables `method_name` / `method_config` by reference, so all
   generated methods would resolve to the last entry.

All seven `extract_*.py` scripts pass `read_table_config=read_table_config` and then call
`pool.read_wavenumber_table(...)` / `pool.read_redshift_table(...)`
(`extract_Gk_data.py:318,320,368,373`, `extract_GkSource_data.py:775,777,821,826`,
`extract_GkWKB_data.py:362,364,411`, `extract_QuadSourceIntegral_data.py:993,995,1045,1050`,
`extract_TkWKB_data.py:386`, `extract_tensor_source_data.py:317`). The whole feature is therefore
currently dead in `SGWK`.

**Upstream design (recommended replacement).** `CPBH`/`SI` deleted the generated-method approach and
replaced it with a single explicit method on each class:

- `Datastore.read_table(cls, *args, **kwargs)` (SI `Datastore.py:720-762`): validates that the
  service is configured, that `class_name` is in the config, and that the factory actually exposes
  `read_table`; honours `config["tables_arg"]` by injecting `tables=self._tables`; profiles under
  `read_table[<class>]`.
- `ShardedPool.read_table(cls, *args, **kwargs)` (SI `ShardedPool.py:833-880`): rejects sharded
  classes and unconfigured classes, picks a shard at random, and forwards to
  `shard.read_table.remote(class_name, *args, **kwargs)`.
- `read_table_config` is re-keyed **by class name** rather than by method name, and the `"class"`
  entry disappears: `{"redshift": {"tables_arg": True}, "wavenumber": {"tables_arg": False}}`.

**Backport recommendation: YES.** This is the largest of the recommended items. The plan must cover:

1. Replace both constructor blocks and both `_generic_read_table` implementations.
2. Rewrite `config/sharding.py:37-40` to the class-keyed form
   (`"wavenumber": {"tables_arg": False}`, `"redshift": {"tables_arg": True}`).
3. Update the ~15 call sites listed above from `pool.read_wavenumber_table(units=…)` to
   `pool.read_table("wavenumber", units=…)` and `pool.read_redshift_table(…)` to
   `pool.read_table("redshift", …)`. Both remain `ray.get`-able `ObjectRef`s, so the surrounding code
   does not change.
4. **Keep `SGWK`'s existing `None` guard** (`if read_table_config is not None`) — see **X3**.
5. No factory changes are needed. `SGWK`'s factories expose `read_table` as a `@staticmethod` on the
   class (`ObjectFactories/redshift.py:76`, `ObjectFactories/wavenumber.py:90`), and both
   `hasattr(factory, "read_table")` and `factory.read_table(conn, tab, *args, **kwargs)` work
   unchanged against class objects. **Do not** couple this to **X4**.
6. Preserve `SGWK`'s existing positional/keyword calling convention. Upstream calls
   `factory.read_table(conn, tab, *args, **kwargs)` and injects `tables` only when `tables_arg` is
   set; `SGWK`'s current `_generic_read_table` always passes `tables=self._tables`. `SGWK`'s
   `redshift.read_table` signature (`conn, table, tables, is_source, is_response, model_proxy`)
   takes `tables` positionally-or-by-keyword and its config entry already sets `tables_arg: True`,
   so the upstream convention is compatible — but this should be checked per factory during
   implementation, not assumed.

### B4. `RayWorkPool` does not tolerate a task builder returning `None`

- **SGWK location:** `RayTools/RayWorkPool.py:283-297` (the `list`/`tuple`/`set` vs. scalar dispatch)
- **Present in:** SGWK ✗, CPBH ✓, SI ✓
- **Category:** robustness
- **Severity:** Low

`SGWK` falls through to `store_ref(ref_data, allow_store=True)`, which raises
`RuntimeError: could not interpret output from task builder (object type="NoneType" …)`. Upstream
adds an explicit `elif ref_data is None:` branch that skips the item, and raises only if
`store_results=True` (where a skipped item would leave a hole in the result list).

**Backport recommendation: YES.** Additive, no behaviour change for existing callers, enables task
builders that legitimately have nothing to enqueue for some items.

### B5. `_assign_shard_keys` does not deduplicate within a single batch

- **SGWK location:** `ShardedPool.py:772-781`
- **Present in:** SGWK ✗, CPBH ✗, SI ✓
- **Category:** correctness
- **Severity:** Low–Medium

If the same shard key object appears twice in one `obj` list, it is appended to `missing_keys` twice
(nothing is written to `self._shard_keys` until the write loop), producing two INSERTs for the same
`store_id` and double-counting the load balancer. With **B1** applied the second INSERT would violate
the primary-key constraint rather than silently duplicating — so B5 is a genuine prerequisite for B1
being safe in the presence of duplicates.

**Upstream fix (SI `ShardedPool.py:788-799`):** a local `seen_store_ids` set.

**Backport recommendation: YES — bundle with B1.**

---

## 4. Findings — features and API changes

### E1. `RayWorkPool`: `store_handler` / `persist_handler` split

- **SGWK locations:** `RayTools/RayWorkPool.py:64-65` (`_default_store_handler`), `89-97`
  (constructor validation), `164-176` and `201-213` (status messages), `409-430` (the `"compute"`
  branch, in particular the hardcoded `obj.store()` at line 414 and `self._store_handler(obj, self._pool)`
  at line 424)
- **Present in:** SGWK ✗, CPBH ✗, SI ✓ (`SI` only — the single generic change `SI` made beyond `CPBH`
  in this module)
- **Category:** API extension
- **Severity:** n/a (capability)

Upstream splits one hook into two:

- `store_handler(obj, pool) -> None` — runs **locally in the driver** after compute completes.
  Default is `obj.store()`, exactly reproducing today's hardcoded line 414. Overridable so a caller
  can mint associated datastore objects before serialization.
- `persist_handler(obj, pool) -> ObjectRef` — the datastore round-trip. Default is
  `pool.object_store(obj)`, exactly today's `_default_store_handler`.

Constructor validation and the two status-message builders switch from testing `store_handler` to
testing `persist_handler`.

**Backport recommendation: YES, but this is a source-incompatible change and must be a separate,
carefully-sequenced commit.**

Migration hazard the plan must handle explicitly: `SGWK` has **35 call sites across 7 files** that
pass `store_handler=None` to mean *"do not persist"* (`main.py`, `extract_Gk_data.py`,
`extract_GkSource_data.py`, `extract_GkWKB_data.py`, `extract_QuadSourceIntegral_data.py`,
`extract_TkWKB_data.py`, `extract_tensor_source_data.py`; 45 `RayWorkPool(...)` constructions in
total). Under the new semantics those calls leave `persist_handler` at its default, and the
constructor branch `compute_handler is None and persist_handler is not None` fires
`raise RuntimeWarning(...)` — which does raise, so the failure is loud and immediate rather than
silent. Nevertheless **every one of those 35 sites must be updated to pass
`persist_handler=None` alongside `store_handler=None`**, matching how `SI`'s own drivers do it
(`SI/main.py:232-233`, `SI/plot_GradientCoupledSolutions.py:943-944`).

Call sites that use the default compute/store path are unaffected: `compute_handler=default`,
`store_handler=default` (`obj.store()`), `persist_handler=default` (`pool.object_store`) reproduces
current behaviour exactly.

Value: without it, any `SGWK` pipeline that needs to create dependent datastore objects between
compute and persist has to do so inside `compute()`. Worth having, but there is no *current* `SGWK`
consumer, so this is discretionary — if the user wants a minimal-risk backport, defer E1 and take
B1–B5 only.

### F2. `inventory()` service

- **Present in:** SGWK ✗, CPBH ✓, SI ✓
- **Upstream locations:** `Datastore.inventory()` (SI `Datastore.py:789-816`);
  `ShardedPool.inventory()` and `ShardedPool._merge_queue()` (SI `ShardedPool.py:882-1001`);
  `InventoryConfigType` (SI `Datastore.py:161-162`); `inventory_config` constructor parameter
  (SI `ShardedPool.py:54`, `ShardedPool.py:211`)
- **Category:** new feature

Produces a human-readable summary of datastore contents per object class. For replicated tables it
queries one shard; for sharded tables it fans out to every shard and merges the per-shard dicts using
a declarative per-field merge policy (`"extend"` for lists/sets, `"earliest"`/`"latest"` for
datetimes, `None`-fill otherwise) supplied in `config/sharding.py:inventory_config`.

**Backport recommendation: DEFER.** The pool/datastore plumbing is ~150 lines and mechanical, but it
is inert without an `inventory()` method on each factory. `SGWK` has **zero** factories exposing
`inventory` today (`grep "def inventory" Datastore/SQL/ObjectFactories/*.py` → no matches), whereas
`SI` has ~20. Backporting the plumbing alone delivers nothing; backporting it usefully means writing
per-factory inventory queries and merge configuration for `SGWK`'s ~15 sharded and ~14 replicated
classes. That is a feature project, not a reconciliation. Recommend the planning agent record it as a
follow-on and ask the user whether they actually want the reporting capability.

If it *is* wanted, note that `SI`'s `ShardedPool.inventory` calls `self._inventory_config[class_name]`
without first checking `self._inventory_config is not None` in the label-merge branch, and the
`_merge_queue` policy dispatch falls through to a `RuntimeError` for any type not in
`{list, set, datetime, None}` (e.g. an `int` count field) — both worth tightening on the way in.

### F3. `Datastore.object_get("version", …)` by name

- **SGWK location:** `Datastore.py:246` (`self.object_get(version, **version_payload)`), with
  `from MetadataConcepts import version` at `Datastore.py:76`
- **Present in:** SGWK ✗, CPBH ✓, SI ✓

Upstream passes the string `"version"`, allowing the `MetadataConcepts` import to be dropped from the
actor module. Purely a decoupling tidy-up — `object_get` already accepts either form
(`Datastore.py:482-485`).

**Recommendation: OPTIONAL.** Harmless; fold into whichever commit touches `Datastore.py` if
convenient. Note `ShardedPool.py:12` also imports `version` and uses it at `ShardedPool.py:155`,
which upstream did *not* change — do not create an inconsistency by fixing only one.

### F4. Apache-2.0 licence headers

Every `CPBH`/`SI` module carries a 14-line `(c) University of Sussex 2026 / Apache 2.0` header;
`SGWK` carries none anywhere. This accounts for the bulk of the raw line-count difference in
`ProfileAgent.py` and `SerialPoolBroker.py` (which are otherwise **byte-identical** to `SGWK`).

**Recommendation: OPTIONAL, and out of scope for the backport.** If the user wants headers, apply
them repo-wide in a separate commit rather than to six files only.

---

## 5. Findings — do NOT backport

### X1. `object_read_batch` object-semantics guard

- **SGWK location:** `ShardedPool.py:596-618`
- **Upstream:** `CPBH`/`SI` changed the guard from `if shard_key_field not in shard_key:` to
  `if not hasattr(shard_key, shard_key_field):`, and the error message from `shard_key.keys()` to
  `shard_key.__attrs__`.

**This change is broken upstream and must not be carried over.** The guard was converted to object
semantics but the two lines that follow it were left on dict semantics:
`shard_key[shard_key_field]` and `payload.update(shard_key)` (SI `ShardedPool.py:628-632`). A
`shard_key` that passes the new `hasattr` guard will fail at `shard_key[...]`; a plain dict fails the
guard. `SGWK`'s version is internally coherent (dict throughout) and is actively used at
`extract_QuadSourceIntegral_data.py:1089`.

**Recommendation: leave `SGWK` as-is.** Worth reporting back to the `StochasticInstantons` tree as a
latent break.

### X2. `object_get_vectorized` shard-key handling

- **SGWK location:** `ShardedPool.py:569-594`
- **Upstream (SI only):** accepts a bare `ShardKeyType` instance in addition to a dict, and **removes**
  `for value in payload_data: value.update(shard_key)`.

Two problems for `SGWK`:

1. `SGWK`'s callers depend on the removed line. All ten `object_get_vectorized` call sites in
   `SGWK/main.py` pass `shard_key` as a dict (e.g. `{"k": k_exit}` at `main.py:1193`, with the
   comment at `main.py:1192` stating the shard key is *deliberately* supplied separately so the pool
   can merge it into each payload). Removing the merge silently drops the `k` field from every
   payload.
2. `SI` implements the new branch using a **module-level** `from config.sharding import ShardKeyType`
   (SI `ShardedPool.py:25`, used at line 595) rather than the constructor-injected
   `self._ShardKeyType` (line 71). That re-couples a deliberately generic module to one project's
   configuration and shadows the constructor parameter of the same name.

**Recommendation: do not backport.** If the "accept a bare shard-key object" convenience is ever
wanted in `SGWK`, implement it against `self._ShardKeyType` **and keep the `value.update(shard_key)`
merge** for the dict branch.

### X3. Removal of the `read_table_config is not None` guard

`SI/ShardedPool.py:206` does `for class_name, config in read_table_config.items():` unconditionally,
despite the parameter defaulting to `None` and being typed `Optional`. `SGWK`'s
`if read_table_config is not None:` at `ShardedPool.py:189` is correct.

**Recommendation: when implementing B3, keep `SGWK`'s guard.** Do not copy this line verbatim.

### X4. Registering factory instances rather than classes

`CPBH`/`SI` register `sqla_version_factory()` (instances, with instance methods);
`SGWK` registers `sqla_version_factory` (classes, with `@staticmethod`s, per
`ObjectFactories/base.py`). Both work with every call site in `Datastore.py`.

**Recommendation: do not backport.** This is a whole-tree refactor of ~25 factory modules with no
functional benefit for `SGWK`, and it is not a prerequisite for any recommended item — in particular
**not** for B3 (see B3, note 5).

---

## 6. Latent defects common to all three trees

None of these are backports (there is no upstream fix to take), but a planning agent should fold the
cheap ones into whichever commit already touches the file.

- **D1.** `ShardedPool.__init__` error path references `self._db_file`, which never exists on
  `ShardedPool` — `SGWK/ShardedPool.py:87` (also `SI:105`). Should be `self._primary_file`. Turns a
  clear "you passed a directory" message into an `AttributeError`.
- **D2.** `raise print(f"…")` in `_read_shard_data` — `SGWK/ShardedPool.py:348-350` (also `SI:363-365`).
  `print` returns `None`, so this raises `TypeError` instead of the intended diagnostic. The branch is
  also unreachable-by-design given the `num_config > 1` check below it; simplest fix is to delete it.
- **D3.** `SerialPoolManager.lease_serial` does `_default_serial_batch_size[table]`
  (`SGWK/ClientPool.py:163`, `SI:172`, `CPBH:167`). Any storable class missing from that hand-maintained
  dict raises `KeyError` at first insert rather than falling back. `.get(table, default_batch_size)`
  would be safer.
- **D4.** `RayWorkPool` notification bookkeeping: `self._last_num_available_complete = self._num_store_complete`
  (`SGWK/RayWorkPool.py:543`, `SI:575`) — copy-paste; should be `self._num_available_complete`.
  Corrupts the reported "available" rate whenever an `available_handler` is in use.

Two things that are **already correct in `SGWK`** and need no action, noted so the planning agent does
not chase them:

- The `validate_on_startup` cascade-delete issue described in the closing section of
  `StochasticInstantons/shard-key-assignment-bug.md` **does not apply**. All six `SGWK` factories with
  a `validate_on_startup` prune path (`BackgroundModel`, `GkNumericIntegration`, `GkSource`,
  `GkWKBIntegration`, `QuadSource`, `TkNumericIntegration`, `TkWKBIntegration`) already delete child
  value rows and tag rows before the parent row.
- `ProfileAgent.py` and `SerialPoolBroker.py` are fully in sync with upstream (byte-identical modulo
  the licence header). **No work required on these two modules.**

---

## 7. Suggested sequencing

The items are largely independent; the ordering below front-loads risk reduction and keeps each
commit independently revertible.

1. **Commit 1 — shard-key persistence (B1 + B5).** `ShardedPool` only. Include the mismatch warning.
   Do the read-only data audit described in B1 *before* landing, and record the outcome.
2. **Commit 2 — shard-config reader (B2).** `ShardedPool` only. Fix all three `row.key_attr`
   occurrences. Verify by reopening an existing sharded datastore.
3. **Commit 3 — small robustness fixes (F6, B4, and optionally D1–D4).** `ShardedPool` +
   `RayWorkPool` + `ClientPool`.
4. **Commit 4 — `read_table` service (B3).** `Datastore` + `ShardedPool` + `config/sharding.py` +
   the seven `extract_*.py` scripts. Largest blast radius; land it on its own.
5. **Commit 5 — `persist_handler` split (E1), if wanted.** `RayWorkPool` + all 35 `store_handler=None`
   call sites in one commit, since the change is source-incompatible.
6. **Deferred / needs user decision:** F2 (`inventory()`), F3, F4.

---

## 8. Verification checklist for the implementer

- [ ] **B1:** on a fresh datastore, after the first shard-key assignment, every row of the primary
      database's `shard_keys` table has `key_serial` equal to the `store_id` of the corresponding
      `wavenumber` row; no `!! _assign_shard_keys MISMATCH` lines appear.
- [ ] **B1:** a run stopped and resumed against the same datastore finds all previously-written
      records (no unexpected recomputation of already-stored objects).
- [ ] **B2:** reopening an existing sharded datastore completes `_read_shard_data` without
      `AttributeError`; a deliberately mismatched `shard_key_type` produces the intended
      `RuntimeError` message rather than a second `AttributeError`.
- [ ] **B3:** each of the seven `extract_*.py` scripts constructs its `ShardedPool` and returns the
      same wavenumber/redshift arrays as before the change (compare against a pre-change run, or
      against a direct SQL query of the `wavenumber`/`redshift` tables).
- [ ] **B3:** `pool.read_table("GkSource", …)` (a sharded class) raises the intended `RuntimeError`;
      `pool.read_table("LambdaCDM", …)` (replicated but unconfigured) likewise.
- [ ] **B4:** a task builder returning `None` for some items completes without error when
      `store_results=False`, and raises when `store_results=True`.
- [ ] **E1:** `grep -rn "store_handler=None" --include="*.py" .` returns 35 hits and every one is
      accompanied by `persist_handler=None`; a work pool using the default handlers still stores
      results exactly as before.
- [ ] Full pipeline smoke test: `main.py` through at least one compute→store→validate cycle on a
      fresh datastore, then a restart against the same datastore.

---

## 9. Reference material

- `/Users/ds283/Documents/Code/StochasticInstantons/shard-key-assignment-bug.md` — the user's own
  write-up of B1, with reproduction snippet, an observed failure case, and instrumentation
  suggestions. Read before implementing B1.
- `SGWK` git commits `a2bd966` and `53dc75d` (both 2025-12-15) — introduced B2 and B3 respectively.
