# Imports of the layer: SecondaryGWKit (SGK) at `b95bf19`

Written by `docs/extraction/measure_client_imports.py`, reading through `git ls-tree` and `git show` only.

- **Repository:** `/Users/ds283/Documents/Code/SecondaryGWKit`
- **Commit:** `b95bf19203b98afa2abcec3c949326f6f439385a` (given as `b95bf19`), 2026-10-12 Write the orchestrator note for dk-and-rpk-migration prompt 09
- **Layer files** (17): `Datastore/__init__.py`, `Datastore/object.py`, `Datastore/contract.py`, `Datastore/replication.py`, `Datastore/shard_paths.py`, `Datastore/store_reader.py`, `Datastore/store_inventory.py`, `Datastore/SQL/__init__.py`, `Datastore/SQL/schema.py`, `Datastore/SQL/ShardedPool.py`, `Datastore/SQL/Datastore.py`, `Datastore/SQL/ClientPool.py`, `Datastore/SQL/SerialPoolBroker.py`, `Datastore/SQL/ProfileAgent.py`, `Datastore/SQL/ObjectFactories/base.py`, `tools/sharded_store.py`, `tools/shard_key_audit.py`
- **Layer files absent at this commit:** `Datastore/object.py`, `Datastore/contract.py`, `Datastore/replication.py`, `Datastore/shard_paths.py`, `Datastore/store_reader.py`, `Datastore/store_inventory.py`, `Datastore/SQL/schema.py`, `Datastore/SQL/ShardedPool.py`, `Datastore/SQL/Datastore.py`, `Datastore/SQL/ClientPool.py`, `Datastore/SQL/SerialPoolBroker.py`, `Datastore/SQL/ProfileAgent.py`, `Datastore/SQL/ObjectFactories/base.py`, `tools/sharded_store.py`, `tools/shard_key_audit.py`
- **Factory package:** `Datastore/SQL/ObjectFactories/` (its `base.py` is the layer's)
- **Files read:** 423 tracked `.py` files outside the layer; 0 do not parse

## Totals

| What | Count |
|---|---|
| Import statements naming a layer module | **51** in 31 files |
| of which nested (not at module level) | 3 |
| String literals naming a layer module | 20 (20 exactly its name) |
| `import_module` / `__import__` calls naming a layer module | 1 |
| Import statements naming a factory module (not layer imports) | 99 (97 outside `Datastore/SQL/ObjectFactories/`) |

## Layer imports, by group

| Group | Statements | Nested | Files |
|---|---|---|---|
| `prompts/` | 51 | 3 | 31 |
| **total** | **51** | **3** | **31** |

## Layer imports, by module

A statement naming two layer modules is counted under each.

| Module | Statements | Names imported by `from` (statements) |
|---|---|---|
| `Datastore` | 2 | — |
| `Datastore.SQL` | 1 | `ShardedPool as _unused` 1 |
| `Datastore.SQL.Datastore` | 6 | `_drop_actions` 2, `_factories` 4 |
| `Datastore.SQL.ShardedPool` | 4 | `ShardedPool` 2, `ShardedPool as _unused` 1 |
| `Datastore.SQL.schema` | 11 | `StoreSchemaMismatch` 2, `build_schema` 10, `schema_differences` 3 |
| `Datastore.object` | 1 | `DatastoreObject` 1 |
| `Datastore.replication` | 1 | `ReadOnlyMiss` 1, `ReplicatedDivergence` 1, `ReplicationMismatch` 1 |
| `Datastore.shard_paths` | 1 | `shard_file_name` 1 |
| `Datastore.store_inventory` | 22 | `InventorySpec` 1, `Parent` 1, `ParentSet` 1, `ShardContext` 2, `inventory_classes` 2, `inventory_specs` 3, `read_inventory` 10, `read_records` 2 |
| `Datastore.store_reader` | 3 | `open_read_only` 3 |

## Every layer import

### `prompts/` (51)

| File:line | Nested | Module | Names |
|---|---|---|---|
| `prompts/datastore-generic-followup/measure/scripts/f1_groups.py:3` | — | `Datastore.SQL.schema` | `build_schema` |
| `prompts/datastore-generic-followup/measure/scripts/f1_groups.py:4` | — | `Datastore.store_inventory` | `inventory_specs` |
| `prompts/datastore-generic-followup/measure/scripts/f2_closure.py:6` | — | `Datastore.SQL.schema` | `build_schema` |
| `prompts/datastore-generic-followup/measure/scripts/f2_closure.py:7` | — | `Datastore.store_inventory` | `inventory_specs` |
| `prompts/datastore-generic-followup/measure/scripts/r2_resolve.py:6` | — | `Datastore.store_inventory` | `read_inventory` |
| `prompts/datastore-generic-followup/orchestrator/measure-03/scripts/m3_messages.py:13` | — | `Datastore` | — |
| `prompts/datastore-generic-followup/orchestrator/measure-03/scripts/m3_messages.py:14` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `prompts/datastore-generic-followup/orchestrator/measure-03/scripts/m3_messages.py:15` | — | `Datastore.shard_paths` | `shard_file_name` |
| `prompts/datastore-generic-followup/orchestrator/measure-03/scripts/m3_messages.py:16` | — | `Datastore.store_reader` | `open_read_only` |
| `prompts/datastore-generic/orchestrator/measure-05/scripts/m1_clean_stores.py:8` | — | `Datastore.SQL.Datastore` | `_factories` |
| `prompts/datastore-generic/orchestrator/measure-05/scripts/m1_clean_stores.py:9` | — | `Datastore.SQL.schema` | `build_schema`, `schema_differences` |
| `prompts/datastore-generic/orchestrator/measure-05/scripts/m2_messages.py:11` | — | `Datastore.SQL.Datastore` | `_factories` |
| `prompts/datastore-generic/orchestrator/measure-05/scripts/m2_messages.py:12` | — | `Datastore.SQL.schema` | `build_schema`, `schema_differences`, `StoreSchemaMismatch` |
| `prompts/datastore-generic/orchestrator/measure-05/scripts/m2_messages.py:13` | — | `Datastore.store_reader` | `open_read_only` |
| `prompts/datastore-generic/orchestrator/measure-05/scripts/m4_fingerprint.py:15` | — | `Datastore.store_inventory` | `read_inventory` |
| `prompts/datastore-generic/orchestrator/measure-05/scripts/m5_refixtures.py:9` | — | `Datastore.SQL.Datastore` | `_factories` |
| `prompts/datastore-generic/orchestrator/measure-05/scripts/m5_refixtures.py:10` | — | `Datastore.SQL.schema` | `build_schema` |
| `prompts/datastore-generic/orchestrator/measure-05/scripts/m5_refixtures.py:11` | — | `Datastore.store_inventory` | `read_inventory` |
| `prompts/datastore-generic/orchestrator/measure-05/scripts/m6_u1.py:9` | — | `Datastore.SQL.Datastore` | `_factories` |
| `prompts/datastore-generic/orchestrator/measure-05/scripts/m6_u1.py:10` | — | `Datastore.SQL.schema` | `build_schema`, `schema_differences` |
| `prompts/datastore-generic/orchestrator/measure-05/scripts/run_shims.py:10` | — | `Datastore.SQL.schema` | `StoreSchemaMismatch` |
| `prompts/datastore-generic/orchestrator/measure-05/scripts/run_shims.py:11` | — | `Datastore.store_inventory as inv` | — |
| `prompts/datastore-generic/orchestrator/measure-06/scripts/m11_orphans.py:7` | — | `Datastore.SQL.Datastore` | `_drop_actions` |
| `prompts/datastore-generic/orchestrator/measure-06/scripts/m1_fingerprint.py:14` | — | `Datastore.store_inventory` | `read_inventory` |
| `prompts/datastore-generic/orchestrator/measure-06/scripts/m3_drop_order.py:12` | — | `Datastore.SQL.schema` | `build_schema` |
| `prompts/datastore-generic/orchestrator/measure-06/scripts/m3b_pool_drop.py:16` | — | `Datastore.SQL.ShardedPool as spm` | — |
| `prompts/datastore-generic/orchestrator/measure-06/scripts/m3b_pool_drop.py:24` | if | `Datastore.SQL.Datastore` | `_drop_actions` |
| `prompts/datastore-generic/orchestrator/measure-06/scripts/m8_messages.py:15` | — | `Datastore.store_reader` | `open_read_only` |
| `prompts/datastore-generic/orchestrator/measure-06/scripts/m8_messages.py:16` | — | `Datastore.store_inventory` | `read_inventory` |
| `prompts/datastore-generic/orchestrator/measure-07/scripts/m1_fingerprint.py:14` | — | `Datastore.store_inventory` | `read_inventory` |
| `prompts/datastore-generic/orchestrator/measure-07/scripts/m2_open_behaviour.py:27` | — | `Datastore.SQL` | `ShardedPool as _unused` |
| `prompts/datastore-generic/orchestrator/measure-07/scripts/m3_units.py:15` | — | `Datastore.SQL.schema` | `build_schema` |
| `prompts/datastore-generic/orchestrator/measure-07/scripts/m3_units.py:16` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `prompts/datastore-generic/orchestrator/measure-07/scripts/m7_messages.py:23` | — | `Datastore.replication` | `ReadOnlyMiss`, `ReplicatedDivergence`, `ReplicationMismatch` |
| `prompts/datastore-generic/orchestrator/measure-07/scripts/m7_messages.py:263` | def test_orphan_tag | `Datastore.store_inventory` | `read_inventory` |
| `prompts/datastore-generic/orchestrator/measure-07/scripts/m8_build_schema_keys.py:8` | — | `Datastore.SQL.schema` | `build_schema` |
| `prompts/datastore-generic/orchestrator/measure-07/scripts/m9_revalidate.py:19` | — | `Datastore.object` | `DatastoreObject` |
| `prompts/datastore-generic/orchestrator/measure-07/scripts/m9_revalidate.py:20` | — | `Datastore.SQL.schema` | `build_schema` |
| `prompts/datastore-generic/orchestrator/measure-08/scripts/m1_fingerprint.py:19` | — | `Datastore.store_inventory as si` | — |
| `prompts/datastore-generic/orchestrator/measure-08/scripts/m1_fingerprint.py:20` | — | `Datastore.store_inventory` | `read_inventory` |
| `prompts/datastore-generic/orchestrator/measure-08/scripts/m3_resolve.py:13` | — | `Datastore.store_inventory` | `read_inventory` |
| `prompts/datastore-generic/orchestrator/measure-08/scripts/m4_order.py:18` | — | `Datastore.store_inventory` | `inventory_classes`, `inventory_specs` |
| `prompts/datastore-generic/orchestrator/measure-08/scripts/m4b_order_matters.py:15` | — | `Datastore.store_inventory as si` | — |
| `prompts/datastore-generic/orchestrator/measure-08/scripts/m5_imports.py:37` | — | `Datastore.store_inventory as si` | — |
| `prompts/datastore-generic/orchestrator/measure-08/scripts/m5b_read_records_alone.py:11` | — | `Datastore.store_inventory` | `ShardContext`, `read_records` |
| `prompts/datastore-generic/orchestrator/measure-08/scripts/m7_messages.py:12` | — | `Datastore.store_inventory as si` | — |
| `prompts/datastore-generic/orchestrator/measure-08/scripts/m7_messages.py:13` | — | `Datastore.store_inventory` | `Parent`, `ParentSet`, `ShardContext`, `read_records` |
| `prompts/datastore-generic/orchestrator/measure-08/scripts/m7_messages.py:48` | if | `Datastore.store_inventory` | `InventorySpec`, `inventory_classes` |
| `prompts/datastore-generic/orchestrator/measure-08/scripts/m7b_reduced_registry.py:10` | — | `Datastore.store_inventory` | `read_inventory` |
| `prompts/datastore-generic/orchestrator/measure-09/scripts/m4_build.py:8` | — | `Datastore` | — |
| `prompts/datastore-generic/orchestrator/measure-09/scripts/m4_inventory.py:11` | — | `Datastore.store_inventory as si` | — |

## String literals naming a layer module

| File:line | Literal | Exact |
|---|---|---|
| `prompts/datastore-generic/orchestrator/measure-08/scripts/m5_imports.py:14` | `Datastore.store_inventory` | yes |
| `prompts/datastore-generic/orchestrator/measure-08/scripts/m5_imports.py:15` | `Datastore.store_reader` | yes |
| `prompts/datastore-generic/orchestrator/measure-08/scripts/m5_imports.py:16` | `Datastore.contract` | yes |
| `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py:50` | `Datastore.object` | yes |
| `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py:51` | `Datastore.contract` | yes |
| `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py:52` | `Datastore.replication` | yes |
| `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py:53` | `Datastore.shard_paths` | yes |
| `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py:54` | `Datastore.store_reader` | yes |
| `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py:55` | `Datastore.store_inventory` | yes |
| `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py:56` | `Datastore.SQL` | yes |
| `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py:57` | `Datastore.SQL.schema` | yes |
| `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py:58` | `Datastore.SQL.ShardedPool` | yes |
| `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py:59` | `Datastore.SQL.Datastore` | yes |
| `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py:60` | `Datastore.SQL.ClientPool` | yes |
| `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py:61` | `Datastore.SQL.SerialPoolBroker` | yes |
| `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py:62` | `Datastore.SQL.ProfileAgent` | yes |
| `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py:63` | `Datastore.SQL.ObjectFactories.base` | yes |
| `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py:64` | `tools.sharded_store` | yes |
| `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py:65` | `tools.shard_key_audit` | yes |
| `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py:70` | `Datastore.SQL.Datastore` | yes |

## `import_module` calls naming a layer module

| File:line | Call | Argument |
|---|---|---|
| `prompts/datastore-generic/orchestrator/measure-07/scripts/m2_open_behaviour.py:30` | `import_module` | `Datastore.SQL.ShardedPool` |

## Imports of factory modules, by group (count only)

| Group | Statements |
|---|---|
| `ComputeTargets/` | 17 |
| `Datastore/SQL/ObjectFactories/` | 2 |
| `Datastore/tests/` | 45 |
| `config/` | 20 |
| `docs/` | 15 |
| **total** | **99** |

## Files that do not parse

None.
