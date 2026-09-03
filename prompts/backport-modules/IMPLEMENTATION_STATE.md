# Implementation state — backport of shared Ray/Datastore modules

**Campaign:** [`README.md`](README.md) · **Source audit:** [`docs/backport-modules-audit.md`](../../docs/backport-modules-audit.md)
**Baseline commit:** `79f0360` (`main`, clean)
**Last updated:** 2026-09-03 — by the planning pass (no code written yet)

> **Maintenance rule.** Every prompt updates this file *in its own commit*, before committing.
> Set your row's status, fill in the commit SHA and the log link, and add or clear entries in
> §3 (Active issues). Do not edit rows other than your own except to close an issue you resolved.

---

## 1. Status board

Legend: ⬜ not started · 🟡 in flight · ✅ complete · ⚠️ complete with deviations · ⛔ blocked

### Backport proper (audit items)

| # | Prompt | Items | Status | Commit | Log |
|---|---|---|---|---|---|
| 01 | [Shard-key persistence](01-shard-key-persistence.md) | B1, B5 | ⬜ | — | — |
| 02 | [Shard-config reader](02-shard-config-reader.md) | B2, D2 | ⬜ | — | — |
| 03 | [Robustness fixes](03-robustness-fixes.md) | F6, B4, D1, D3, D4, F3 | ⬜ | — | — |
| 04 | [`read_table` service](04-read-table-service.md) | B3 | ⬜ | — | — |
| 05 | [`persist_handler` split](05-persist-handler-split.md) | E1 | ⬜ | — | — |

### `inventory()` sub-campaign (F2)

| # | Prompt | Items | Status | Commit | Log |
|---|---|---|---|---|---|
| 06 | [Inventory plumbing](06-inventory-plumbing.md) | F2a | ⬜ | — | — |
| 07 | [Replicated factories](07-inventory-replicated-factories.md) | F2b | ⬜ | — | — |
| 08 | [Sharded factories + merge config](08-inventory-sharded-factories.md) | F2c | ⬜ | — | — |
| 09 | [Inventory reporting](09-inventory-reporting.md) | F2d | ⬜ | — | — |

### Close-out

| # | Prompt | Items | Status | Commit | Log |
|---|---|---|---|---|---|
| 10 | [Verification pass](10-verification.md) | audit §8 + F2 | ⬜ | — | — |

**Progress:** 0 / 10 complete.

---

## 2. Item-level tracking

Traceability from the audit's finding IDs to the prompt that discharges them.

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| B1 | **Critical** | `_assign_shard_keys` inserts `key_id` into a `key_serial` PK column → silent shard misrouting after restart | 01 | ⬜ |
| B5 | Low–Medium | `_assign_shard_keys` does not dedup within a batch (hard prerequisite for B1) | 01 | ⬜ |
| B2 | High | `_read_shard_data` reads `row.key_attr` from a `key_type`-only select → `AttributeError` on every reopen | 02 | ⬜ |
| D2 | Latent | `raise print(f"…")` raises `TypeError`, and the branch is unreachable | 02 | ⬜ |
| F6 | Low | Unguarded empty-list insert into `sharded_tables` | 03 | ⬜ |
| B4 | Low | `RayWorkPool` rejects a task builder returning `None` | 03 | ⬜ |
| D1 | Latent | `ShardedPool.__init__` error path references non-existent `self._db_file` | 03 | ⬜ |
| D3 | Latent | `_default_serial_batch_size[table]` raises `KeyError` for unlisted classes | 03 | ⬜ |
| D4 | Latent | `_last_num_available_complete` assigned from `_num_store_complete` | 03 | ⬜ |
| F3 | Optional | `object_get("version", …)` by name, dropping the `MetadataConcepts` coupling | 03 | ⬜ |
| B3 | High | `read_table_config` method generation is broken in four ways; replace with `read_table()` | 04 | ⬜ |
| E1 | Feature | `store_handler` / `persist_handler` split in `RayWorkPool` — **confirmed wanted** | 05 | ⬜ |
| F2a | Feature | `inventory()` plumbing: `Datastore`, `ShardedPool`, `_merge_queue`, numeric merge policies | 06 | ⬜ |
| F2b | Feature | `inventory()` on the 13 replicated-table factories | 07 | ⬜ |
| F2c | Feature | `inventory()` on the 15 sharded-table factories + `inventory_config` | 08 | ⬜ |
| F2d | Feature | Inventory reporting entry point | 09 | ⬜ |

**Explicitly not backported** (audit §5 — do not schedule these): X1, X2, X3, X4.
**Out of scope:** F4 (licence headers — repo-wide or not at all; see `README.md` §6).

---

## 3. Active and unresolved issues

*Nothing yet — the campaign has not started.*

> Add an entry here whenever a prompt finishes with something unresolved: a verification step that
> could not be run, an assumption that could not be confirmed, a deviation that a later prompt has
> to work around. Format:
>
> - **[NN-shortname]** *(opened by prompt NN, YYYY-MM-DD)* — description. **Impact:** who is
>   affected. **Next step:** what would close it.
>
> Move closed entries to §4 rather than deleting them.

---

## 4. Resolved issues

*None yet.*

---

## 5. Standing notes for all implementers

Established by the planning pass against the working tree at `79f0360`. These do not need
re-deriving.

1. **No `SGWK` datastore in the tree uses the current schema.** `test-qcd-db.sqlite` has
   `shard_keys.wavenumber_serial` and lacks `shard_key_config`, `replicated_tables` and
   `sharded_tables` — it predates commit `a2bd966` (2025-12-15). Consequences: (a) it corroborates
   that the B2/B3 code paths have never been exercised, so those findings need no re-derivation;
   (b) it cannot be used as a fixture for "reopen an existing pool" tests — a fresh datastore must
   be created first; (c) any pre-existing datastore must be rebuilt regardless of B1, purely on
   schema grounds.
2. **There is no test infrastructure for `Datastore`, `ShardedPool` or `RayWorkPool`.** The only
   pytest suites are `AdaptiveLevin/tests/` and `LiouvilleGreen/tests/`. Verification in this
   campaign is by static check plus purpose-written scripts; be explicit in your log about what you
   actually executed versus what you only reasoned about.
3. **Only two factories expose `read_table`:** `ObjectFactories/redshift.py:76`
   (`conn, table, tables, is_source, is_response, model_proxy` → `tables_arg: True`) and
   `ObjectFactories/wavenumber.py:90` (`conn, table, units, is_source, is_response` →
   `tables_arg: False`). Both are `@staticmethod`s on the factory class.
4. **Verified call-site counts** at `79f0360`: 35 `store_handler=None` across 7 files; 45
   `RayWorkPool(` constructions; 17 `read_wavenumber_table`/`read_redshift_table` calls across 6
   `extract_*.py` scripts. If your grep disagrees, something landed between then and now — stop and
   reconcile before editing.

### Notes specific to the F2 sub-campaign (prompts 06–09)

5. **Factories are classes with `@staticmethod`s, not instances.** `SGWK` registers
   `sqla_x_factory` (the class); `CPBH`/`SI` register `sqla_x_factory()` (an instance). So every
   upstream `inventory` method takes `self` and **cannot be copied across unaltered** — `SGWK`'s
   signature is `@staticmethod def inventory(conn, table, tables, *args, **kwargs)`. This is the
   most likely silent failure in prompts 07 and 08: a stray `self` binds `conn` and the error
   surfaces far from its cause.
6. **`inventory` must stay optional.** `ObjectFactories/base.py` is an ABC whose members are all
   `@staticmethod` **and** `@abstractmethod`. Making `inventory` abstract would oblige all ~37
   factory classes to implement one, including the tag-association factories where it is
   meaningless. Discover it with `hasattr`, as upstream does. **Do not modify `base.py`.**
7. **Class counts:** 13 replicated (`version`, `store_tag`, `redshift`, `wavenumber`,
   `wavenumber_exit_time`, `tolerance`, `LambdaCDM`, `QCD_Cosmology`, `IntegrationSolver`,
   `BackgroundModel`, `BackgroundModelValue`, `GkSourcePolicy`, `QuadSourcePolicy`) and 15 sharded
   (`TkNumericIntegration`, `TkNumericValue`, `TkWKBIntegration`, `TkWKBValue`, `QuadSource`,
   `QuadSourceValue`, `GkNumericIntegration`, `GkNumericValue`, `GkWKBIntegration`, `GkWKBValue`,
   `GkSource`, `GkSourceValue`, `GkSourcePolicyData`, `QuadSourceIntegral`, `OneLoopIntegral`).
   Note the audit says 14 replicated; the config lists 13.
8. **Tables with no `timestamp` column** (registered `"timestamp": False`): `version`,
   `BackgroundModelValue`, and all six sharded value tables (`TkNumericValue`, `TkWKBValue`,
   `QuadSourceValue`, `GkNumericValue`, `GkWKBValue`, `GkSourceValue`). These can only be counted,
   which is why prompt 06 must add a numeric `"sum"` merge policy that upstream lacks.
9. **Tables with a `validated` column:** `BackgroundModel`, `GkNumericIntegration`, `GkSource`,
   `GkWKBIntegration`, `QuadSource`, `TkNumericIntegration`, `TkWKBIntegration`. These take the
   labelled validated/unvalidated inventory shape. `OneLoopIntegral` mentions `validated` once —
   prompt 08 must check what that occurrence actually is rather than assuming.
10. **Zero factories expose `inventory` today.** Prompts 07 and 08 write all ~28 from scratch;
    there is nothing to reconcile against in this tree.
