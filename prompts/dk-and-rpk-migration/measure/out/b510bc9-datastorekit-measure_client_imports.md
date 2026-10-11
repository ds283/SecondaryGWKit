# Imports of the layer: SecondaryGWKit (SGK) at `b510bc9`

Written by `docs/extraction/measure_client_imports.py`, reading through `git ls-tree` and `git show` only.

- **Repository:** `/Users/ds283/Documents/Code/SecondaryGWKit`
- **Commit:** `b510bc90b66985532d28996ab4e5b13b0ff71709` (given as `b510bc9`), 2026-10-07 Freeze the datastore layer while it moves to DatastoreKit
- **Layer files** (17): `Datastore/__init__.py`, `Datastore/object.py`, `Datastore/contract.py`, `Datastore/replication.py`, `Datastore/shard_paths.py`, `Datastore/store_reader.py`, `Datastore/store_inventory.py`, `Datastore/SQL/__init__.py`, `Datastore/SQL/schema.py`, `Datastore/SQL/ShardedPool.py`, `Datastore/SQL/Datastore.py`, `Datastore/SQL/ClientPool.py`, `Datastore/SQL/SerialPoolBroker.py`, `Datastore/SQL/ProfileAgent.py`, `Datastore/SQL/ObjectFactories/base.py`, `tools/sharded_store.py`, `tools/shard_key_audit.py`
- **Layer files absent at this commit:** none
- **Factory package:** `Datastore/SQL/ObjectFactories/` (its `base.py` is the layer's)
- **Files read:** 445 tracked `.py` files outside the layer; 0 do not parse

## Totals

| What | Count |
|---|---|
| Import statements naming a layer module | **233** in 138 files |
| of which nested (not at module level) | 41 |
| String literals naming a layer module | 13 (10 exactly its name) |
| `import_module` / `__import__` calls naming a layer module | 11 |
| Import statements naming a factory module (not layer imports) | 100 (98 outside `Datastore/SQL/ObjectFactories/`) |

## Layer imports, by group

| Group | Statements | Nested | Files |
|---|---|---|---|
| `(top level)` | 16 | 0 | 8 |
| `ComputeTargets/` | 11 | 0 | 11 |
| `CosmologyConcepts/` | 2 | 0 | 2 |
| `CosmologyModels/` | 1 | 0 | 1 |
| `Datastore/SQL/ObjectFactories/` | 43 | 23 | 20 |
| `Datastore/tests/` | 68 | 4 | 37 |
| `MetadataConcepts/` | 5 | 0 | 5 |
| `Quadrature/` | 1 | 0 | 1 |
| `RayTools/` | 1 | 0 | 1 |
| `RunRegistry/` | 14 | 6 | 6 |
| `config/` | 1 | 0 | 1 |
| `docs/` | 17 | 4 | 13 |
| `prompts/` | 51 | 3 | 31 |
| `tools/` | 2 | 1 | 1 |
| **total** | **233** | **41** | **138** |

## Layer imports, by module

A statement naming two layer modules is counted under each.

| Module | Statements | Names imported by `from` (statements) |
|---|---|---|
| `Datastore` | 22 | `DatastoreObject` 19, `contract` 1 |
| `Datastore.SQL` | 1 | `ShardedPool as _unused` 1 |
| `Datastore.SQL.ClientPool` | 1 | `SerialPoolManager` 1 |
| `Datastore.SQL.Datastore` | 12 | `_drop_actions` 2, `_factories` 10 |
| `Datastore.SQL.ObjectFactories.base` | 21 | `SQLAFactoryBase` 21 |
| `Datastore.SQL.ProfileAgent` | 7 | `ProfileAgent` 7 |
| `Datastore.SQL.ShardedPool` | 28 | `ShardedPool` 26, `ShardedPool as _unused` 1 |
| `Datastore.SQL.schema` | 35 | `SchemaDifferences` 1, `StoreSchemaMismatch` 6, `build_schema` 31, `dependent_tables` 1, `drop_order` 2, `schema_differences` 4 |
| `Datastore.contract` | 3 | `TAG_LABEL` 1, `TAG_SERIAL` 1, `TAG_TABLE` 2, `VERSION_LABEL` 1, `VERSION_TABLE` 1, `contract` 1 |
| `Datastore.object` | 2 | `DatastoreObject` 2 |
| `Datastore.replication` | 10 | `ReadOnlyMiss` 1, `ReadOnlyWrite` 2, `ReplicatedDivergence` 4, `ReplicationInFlight` 1, `ReplicationMismatch` 6, `differing_columns` 2 |
| `Datastore.shard_paths` | 15 | `resolve_shard_path` 3, `shard_file_name` 13, `shard_file_problem` 1 |
| `Datastore.store_inventory` | 68 | `ClassInventory` 2, `InventorySpec` 23, `Parent` 15, `ParentSet` 3, `Record` 3, `ShardContext` 4, `StoreInventory` 3, `_merge_digests` 1, `canonical` 1, `canonical_json` 3, `digest` 1, `inventory_classes` 3, `inventory_specs` 5, `read_inventory` 29, `read_records` 4, `reference_digest` 2 |
| `Datastore.store_reader` | 10 | `open_read_only` 9, `read_only_url` 2 |

## Every layer import

### `(top level)` (16)

| File:line | Nested | Module | Names |
|---|---|---|---|
| `extract_GkSource_data.py:34` | — | `Datastore.SQL.ProfileAgent` | `ProfileAgent` |
| `extract_GkSource_data.py:35` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `extract_GkWKB_data.py:32` | — | `Datastore.SQL.ProfileAgent` | `ProfileAgent` |
| `extract_GkWKB_data.py:33` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `extract_Gk_data.py:29` | — | `Datastore.SQL.ProfileAgent` | `ProfileAgent` |
| `extract_Gk_data.py:30` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `extract_QuadSourceIntegral_data.py:40` | — | `Datastore.SQL.ProfileAgent` | `ProfileAgent` |
| `extract_QuadSourceIntegral_data.py:41` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `extract_TkWKB_data.py:28` | — | `Datastore.SQL.ProfileAgent` | `ProfileAgent` |
| `extract_TkWKB_data.py:29` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `extract_common.py:44` | — | `Datastore.store_inventory` | `read_inventory` |
| `extract_tensor_source_data.py:28` | — | `Datastore.SQL.ProfileAgent` | `ProfileAgent` |
| `extract_tensor_source_data.py:29` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `main.py:58` | — | `Datastore.SQL.ProfileAgent` | `ProfileAgent` |
| `main.py:59` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `main.py:92` | — | `Datastore.store_inventory` | `read_inventory` |

### `ComputeTargets/` (11)

| File:line | Nested | Module | Names |
|---|---|---|---|
| `ComputeTargets/BackgroundModel.py:24` | — | `Datastore` | `DatastoreObject` |
| `ComputeTargets/GkNumericIntegration.py:14` | — | `Datastore` | `DatastoreObject` |
| `ComputeTargets/GkSource.py:13` | — | `Datastore` | `DatastoreObject` |
| `ComputeTargets/GkSourcePolicyData.py:12` | — | `Datastore` | `DatastoreObject` |
| `ComputeTargets/GkWKBIntegration.py:13` | — | `Datastore` | `DatastoreObject` |
| `ComputeTargets/OneLoopIntegral.py:7` | — | `Datastore` | `DatastoreObject` |
| `ComputeTargets/QuadSource.py:16` | — | `Datastore` | `DatastoreObject` |
| `ComputeTargets/QuadSourceIntegral.py:17` | — | `Datastore` | `DatastoreObject` |
| `ComputeTargets/TkNumericIntegration.py:14` | — | `Datastore` | `DatastoreObject` |
| `ComputeTargets/TkWKBIntegration.py:10` | — | `Datastore` | `DatastoreObject` |
| `ComputeTargets/tests/test_qcd_cosmology_inventory_record.py:33` | — | `Datastore.store_inventory` | `read_inventory` |

### `CosmologyConcepts/` (2)

| File:line | Nested | Module | Names |
|---|---|---|---|
| `CosmologyConcepts/redshift.py:6` | — | `Datastore` | `DatastoreObject` |
| `CosmologyConcepts/wavenumber.py:11` | — | `Datastore` | `DatastoreObject` |

### `CosmologyModels/` (1)

| File:line | Nested | Module | Names |
|---|---|---|---|
| `CosmologyModels/base.py:3` | — | `Datastore` | `DatastoreObject` |

### `Datastore/SQL/ObjectFactories/` (43)

| File:line | Nested | Module | Names |
|---|---|---|---|
| `Datastore/SQL/ObjectFactories/BackgroundModel.py:90` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/BackgroundModel.py:579` | if | `Datastore.replication` | `ReplicationMismatch`, `differing_columns` |
| `Datastore/SQL/ObjectFactories/BackgroundModel.py:888` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec`, `Parent` |
| `Datastore/SQL/ObjectFactories/GkNumericIntegration.py:14` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/GkNumericIntegration.py:639` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec`, `Parent` |
| `Datastore/SQL/ObjectFactories/GkSource.py:48` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/GkSource.py:847` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec`, `Parent`, `ParentSet` |
| `Datastore/SQL/ObjectFactories/GkSourcePolicy.py:3` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/GkSourcePolicy.py:72` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec` |
| `Datastore/SQL/ObjectFactories/GkSourcePolicyData.py:9` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/GkSourcePolicyData.py:181` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec`, `Parent` |
| `Datastore/SQL/ObjectFactories/GkWKBIntegration.py:64` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/GkWKBIntegration.py:968` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec`, `Parent` |
| `Datastore/SQL/ObjectFactories/LambdaCDM.py:4` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/LambdaCDM.py:96` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec` |
| `Datastore/SQL/ObjectFactories/OneLoopIntegral.py:11` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/OneLoopIntegral.py:346` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec`, `Parent` |
| `Datastore/SQL/ObjectFactories/QCD_Cosmology.py:21` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/QCD_Cosmology.py:132` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec` |
| `Datastore/SQL/ObjectFactories/QuadSource.py:14` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/QuadSource.py:550` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec`, `Parent` |
| `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py:17` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py:973` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec`, `Parent` |
| `Datastore/SQL/ObjectFactories/QuadSourcePolicy.py:3` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/QuadSourcePolicy.py:109` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec` |
| `Datastore/SQL/ObjectFactories/TkNumericIntegration.py:16` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/TkNumericIntegration.py:641` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec`, `Parent` |
| `Datastore/SQL/ObjectFactories/TkWKBIntegration.py:60` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/TkWKBIntegration.py:966` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec`, `Parent` |
| `Datastore/SQL/ObjectFactories/integration_metadata.py:3` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/integration_metadata.py:61` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec` |
| `Datastore/SQL/ObjectFactories/redshift.py:8` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/redshift.py:137` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec` |
| `Datastore/SQL/ObjectFactories/store_tag.py:3` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/store_tag.py:51` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec` |
| `Datastore/SQL/ObjectFactories/tolerance.py:5` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/tolerance.py:59` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec` |
| `Datastore/SQL/ObjectFactories/version.py:3` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/version.py:50` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec` |
| `Datastore/SQL/ObjectFactories/wavenumber.py:11` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/SQL/ObjectFactories/wavenumber.py:130` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec` |
| `Datastore/SQL/ObjectFactories/wavenumber.py:329` | if | `Datastore.replication` | `ReplicationMismatch`, `differing_columns` |
| `Datastore/SQL/ObjectFactories/wavenumber.py:390` | def inventory_spec | `Datastore.store_inventory` | `InventorySpec`, `Parent` |

### `Datastore/tests/` (68)

| File:line | Nested | Module | Names |
|---|---|---|---|
| `Datastore/tests/real_store_fixtures.py:52` | — | `Datastore.SQL.schema` | `build_schema` |
| `Datastore/tests/real_store_fixtures.py:53` | — | `Datastore.shard_paths` | `shard_file_name` |
| `Datastore/tests/shard_store_fixtures.py:22` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `Datastore/tests/shard_store_fixtures.py:23` | — | `Datastore.shard_paths` | `shard_file_name` |
| `Datastore/tests/test_absolute_shard_record_refused.py:35` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `Datastore/tests/test_absolute_shard_record_refused.py:36` | — | `Datastore.shard_paths` | `resolve_shard_path`, `shard_file_name` |
| `Datastore/tests/test_absolute_shard_record_refused.py:37` | — | `Datastore.store_reader` | `open_read_only` |
| `Datastore/tests/test_closed_store_refusals.py:26` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `Datastore/tests/test_closed_store_refusals.py:27` | — | `Datastore.shard_paths` | `shard_file_name` |
| `Datastore/tests/test_closed_store_refusals.py:28` | — | `Datastore.store_reader` | `open_read_only` |
| `Datastore/tests/test_copy_move_store.py:29` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `Datastore/tests/test_copy_move_store.py:30` | — | `Datastore.shard_paths` | `shard_file_name` |
| `Datastore/tests/test_declared_facts.py:43` | — | `Datastore.contract` | `TAG_LABEL`, `TAG_SERIAL`, `TAG_TABLE`, `VERSION_LABEL`, `VERSION_TABLE` |
| `Datastore/tests/test_declared_facts.py:50` | — | `Datastore.object` | `DatastoreObject` |
| `Datastore/tests/test_declared_facts.py:51` | — | `Datastore.replication` | `ReplicationMismatch` |
| `Datastore/tests/test_declared_facts.py:52` | — | `Datastore.SQL.ObjectFactories.base` | `SQLAFactoryBase` |
| `Datastore/tests/test_declared_facts.py:53` | — | `Datastore.SQL.schema` | `build_schema` |
| `Datastore/tests/test_delete_store.py:35` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `Datastore/tests/test_delete_store.py:36` | — | `Datastore.shard_paths` | `shard_file_name` |
| `Datastore/tests/test_drop_refuses_dangling_references.py:35` | — | `Datastore.SQL.schema` | `build_schema`, `dependent_tables` |
| `Datastore/tests/test_drop_refuses_dangling_references.py:36` | — | `Datastore.replication` | `ReadOnlyWrite` |
| `Datastore/tests/test_drop_refuses_dangling_references.py:37` | — | `Datastore.store_inventory` | `inventory_specs` |
| `Datastore/tests/test_exact_identity_lookups.py:62` | — | `Datastore.SQL.schema` | `build_schema` |
| `Datastore/tests/test_factory_select_columns.py:740` | — | `Datastore.SQL.schema` | `build_schema` |
| `Datastore/tests/test_gksource_parent_set.py:58` | — | `Datastore.store_inventory` | `read_inventory` |
| `Datastore/tests/test_integral_read_batch.py:62` | — | `Datastore.SQL.schema` | `build_schema` |
| `Datastore/tests/test_inventory_consumers.py:33` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `Datastore/tests/test_inventory_declarations.py:29` | — | `Datastore.contract` | `TAG_TABLE` |
| `Datastore/tests/test_inventory_declarations.py:30` | — | `Datastore.store_inventory` | `ClassInventory`, `InventorySpec`, `Parent`, `ParentSet`, `Record`, `StoreInventory`, `inventory_classes`, `inventory_specs`, `read_inventory`, `reference_digest` |
| `Datastore/tests/test_inventory_report.py:26` | — | `Datastore.store_inventory` | `read_inventory` |
| `Datastore/tests/test_inventory_report_labels.py:39` | — | `Datastore.store_inventory` | `Record`, `StoreInventory`, `digest`, `read_inventory`, `reference_digest` |
| `Datastore/tests/test_inventory_retired.py:30` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `Datastore/tests/test_layer_is_generic.py:49` | — | `Datastore` | `contract` |
| `Datastore/tests/test_layer_is_generic.py:50` | — | `Datastore.SQL.schema` | `build_schema` |
| `Datastore/tests/test_layer_registry.py:52` | — | `Datastore.SQL.ClientPool` | `SerialPoolManager` |
| `Datastore/tests/test_layer_registry.py:53` | — | `Datastore.SQL.schema` | `build_schema`, `drop_order` |
| `Datastore/tests/test_layer_registry.py:54` | — | `Datastore.replication` | `ReadOnlyWrite` |
| `Datastore/tests/test_layer_registry.py:55` | — | `Datastore.store_inventory` | `read_inventory` |
| `Datastore/tests/test_layer_registry.py:56` | — | `Datastore.store_reader` | `open_read_only` |
| `Datastore/tests/test_prune_at_open.py:44` | — | `Datastore.replication` | `ReplicatedDivergence` |
| `Datastore/tests/test_prune_at_open.py:302` | def test_the_set_of_classes_is_read_from_the_factories | `Datastore.SQL.schema` | `build_schema` |
| `Datastore/tests/test_quadsource_integral_parent_key.py:52` | — | `Datastore.store_inventory` | `read_inventory` |
| `Datastore/tests/test_quadsource_parent_key.py:47` | — | `Datastore.store_inventory` | `Parent`, `ShardContext`, `_merge_digests`, `read_inventory`, `read_records` |
| `Datastore/tests/test_quadsource_policy_key.py:357` | def read_the_store | `Datastore.store_inventory` | `read_inventory` |
| `Datastore/tests/test_read_only_pool.py:52` | — | `Datastore.SQL.schema` | `StoreSchemaMismatch` |
| `Datastore/tests/test_read_path_mechanics.py:56` | — | `Datastore.SQL.schema` | `build_schema` |
| `Datastore/tests/test_reconcile_at_open.py:46` | — | `Datastore.SQL.schema` | `StoreSchemaMismatch` |
| `Datastore/tests/test_reconcile_at_open.py:47` | — | `Datastore.replication` | `ReplicatedDivergence`, `ReplicationMismatch` |
| `Datastore/tests/test_replicated_write.py:37` | — | `Datastore.replication` | `ReplicationInFlight`, `ReplicationMismatch` |
| `Datastore/tests/test_schema_builder.py:50` | — | `Datastore.SQL.schema` | `build_schema` |
| `Datastore/tests/test_shard_file_name.py:20` | — | `Datastore.shard_paths` | `shard_file_name` |
| `Datastore/tests/test_shard_paths.py:24` | — | `Datastore.shard_paths` | `resolve_shard_path`, `shard_file_problem` |
| `Datastore/tests/test_sharded_store_script.py:21` | — | `Datastore.shard_paths` | `shard_file_name` |
| `Datastore/tests/test_store_inventory.py:38` | — | `Datastore.store_inventory as store_inventory` | — |
| `Datastore/tests/test_store_inventory.py:39` | — | `Datastore.store_inventory` | `canonical`, `canonical_json`, `read_inventory` |
| `Datastore/tests/test_store_inventory.py:532` | def store_inventory_tables | `Datastore.SQL.schema` | `build_schema` |
| `Datastore/tests/test_store_reader.py:34` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `Datastore/tests/test_store_reader.py:35` | — | `Datastore.SQL.schema` | `StoreSchemaMismatch` |
| `Datastore/tests/test_store_reader.py:36` | — | `Datastore.store_reader` | `open_read_only` |
| `Datastore/tests/test_store_schema.py:43` | — | `Datastore.SQL.schema` | `SchemaDifferences`, `StoreSchemaMismatch`, `build_schema`, `schema_differences` |
| `Datastore/tests/test_store_schema.py:49` | — | `Datastore.store_inventory` | `read_inventory` |
| `Datastore/tests/test_store_schema.py:50` | — | `Datastore.store_reader` | `open_read_only`, `read_only_url` |
| `Datastore/tests/test_store_schema.py:233` | def test_a_refusal_yields_nothing_and_disposes_every_engine | `Datastore.store_reader as store_reader` | — |
| `Datastore/tests/test_value_reads_parent_key.py:58` | — | `Datastore.SQL.schema` | `build_schema` |
| `Datastore/tests/test_version_row_at_open.py:41` | — | `Datastore.SQL.schema` | `build_schema` |
| `Datastore/tests/test_version_row_at_open.py:42` | — | `Datastore.replication` | `ReplicatedDivergence` |
| `Datastore/tests/test_wkb_numeric_parent_key.py:81` | — | `Datastore.SQL.schema` | `build_schema`, `drop_order` |
| `Datastore/tests/test_wkb_numeric_parent_key.py:82` | — | `Datastore.store_inventory` | `Parent`, `ShardContext`, `read_records` |

### `MetadataConcepts/` (5)

| File:line | Nested | Module | Names |
|---|---|---|---|
| `MetadataConcepts/GkSourcePolicy.py:3` | — | `Datastore` | `DatastoreObject` |
| `MetadataConcepts/QuadSourcePolicy.py:3` | — | `Datastore` | `DatastoreObject` |
| `MetadataConcepts/store_tag.py:1` | — | `Datastore` | `DatastoreObject` |
| `MetadataConcepts/tolerance.py:3` | — | `Datastore` | `DatastoreObject` |
| `MetadataConcepts/version.py:1` | — | `Datastore` | `DatastoreObject` |

### `Quadrature/` (1)

| File:line | Nested | Module | Names |
|---|---|---|---|
| `Quadrature/integration_metadata.py:3` | — | `Datastore` | `DatastoreObject` |

### `RayTools/` (1)

| File:line | Nested | Module | Names |
|---|---|---|---|
| `RayTools/RayWorkPool.py:8` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |

### `RunRegistry/` (14)

| File:line | Nested | Module | Names |
|---|---|---|---|
| `RunRegistry/stores.py:1084` | def copy_store | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `RunRegistry/stores.py:1121` | def move_store | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `RunRegistry/stores.py:1197` | def fingerprint_of | `Datastore.store_inventory` | `canonical_json` |
| `RunRegistry/stores.py:1383` | def listing_lines | `Datastore.store_inventory` | `canonical_json` |
| `RunRegistry/stores.py:1493` | def fingerprint_store | `Datastore.store_inventory` | `read_inventory` |
| `RunRegistry/stores.py:1814` | def retire_store | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `RunRegistry/tests/store_fixtures.py:19` | — | `Datastore.shard_paths` | `shard_file_name` |
| `RunRegistry/tests/test_quadsource_atol_sweep_prepare.py:29` | — | `Datastore.shard_paths` | `shard_file_name` |
| `RunRegistry/tests/test_store_copy_move.py:30` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `RunRegistry/tests/test_store_copy_move.py:31` | — | `Datastore.shard_paths` | `shard_file_name` |
| `RunRegistry/tests/test_store_fingerprint.py:49` | — | `Datastore.store_inventory` | `read_inventory` |
| `RunRegistry/tests/test_store_retire.py:45` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `RunRegistry/tests/test_store_retire.py:46` | — | `Datastore.shard_paths` | `shard_file_name` |
| `RunRegistry/tests/test_store_retire.py:47` | — | `Datastore.store_inventory` | `read_inventory` |

### `config/` (1)

| File:line | Nested | Module | Names |
|---|---|---|---|
| `config/model_list.py:4` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |

### `docs/` (17)

| File:line | Nested | Module | Names |
|---|---|---|---|
| `docs/a3-v2-readiness/readonly_open_probe.py:53` | — | `Datastore.store_reader` | `open_read_only`, `read_only_url` |
| `docs/a3-v2-readiness/rehearsal/store_soundness.py:19` | — | `Datastore.store_inventory` | `read_inventory` |
| `docs/datastore-integrity-audit/fk_sweep_probe.py:8` | — | `Datastore.SQL.Datastore` | `_factories` |
| `docs/datastore-integrity-audit/fk_sweep_probe.py:9` | — | `Datastore.SQL.schema` | `build_schema` |
| `docs/datastore-integrity-audit/lookup_keys/common.py:18` | — | `Datastore.SQL.Datastore` | `_factories` |
| `docs/datastore-integrity-audit/lookup_keys/common.py:19` | — | `Datastore.SQL.schema` | `build_schema` |
| `docs/datastore-integrity-audit/lookup_keys/probe_03_column_existence.py:25` | — | `Datastore.SQL.Datastore` | `_factories` |
| `docs/datastore-integrity-audit/lookup_keys/probe_03_column_existence.py:26` | — | `Datastore.SQL.schema` | `build_schema` |
| `docs/datastore-integrity-audit/replicated_divergence_probe.py:16` | — | `Datastore.store_inventory` | `read_inventory` |
| `docs/datastore-integrity-audit/replicated_write_fault_probe.py:159` | def main | `Datastore.store_inventory` | `read_inventory` |
| `docs/datastore-integrity-audit/tq_fk_probe.py:8` | — | `Datastore.SQL.Datastore` | `_factories` |
| `docs/datastore-integrity-audit/tq_fk_probe.py:9` | — | `Datastore.SQL.schema` | `build_schema` |
| `docs/handover/quadsource_atol_sweep.py:693` | def assert_store_is_self_consistent | `Datastore.shard_paths` | `resolve_shard_path` |
| `docs/source-remediation-verification/analyse_greens_and_source.py:35` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `docs/source-remediation-verification/run_quadsource_integrals.py:41` | — | `Datastore.SQL.ShardedPool` | `ShardedPool` |
| `docs/tolerance-convergence/inventory.py:196` | def keyed_object_types | `Datastore.SQL.Datastore` | `_factories` |
| `docs/tolerance-convergence/order_audit.py:1253` | def d3_tables | `Datastore.SQL.Datastore` | `_factories` |

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

### `tools/` (2)

| File:line | Nested | Module | Names |
|---|---|---|---|
| `tools/inventory_report.py:82` | — | `Datastore.store_inventory` | `ClassInventory`, `Record`, `StoreInventory` |
| `tools/inventory_report.py:162` | def _schema_tables | `Datastore.SQL.schema` | `build_schema` |

## String literals naming a layer module

| File:line | Literal | Exact |
|---|---|---|
| `Datastore/tests/test_delete_store.py:339` | `Datastore.SQL.ShardedPool.resolve_shard_path` | no |
| `Datastore/tests/test_layer_is_generic.py:356` | `Datastore.SQL.ObjectFactories.base` | yes |
| `Datastore/tests/test_layer_registry.py:432` | `Datastore.SQL.ClientPool.ClientPool` | no |
| `Datastore/tests/test_layer_registry.py:592` | `Datastore.SQL.ShardedPool` | yes |
| `Datastore/tests/test_layer_registry.py:593` | `Datastore.SQL.Datastore` | yes |
| `Datastore/tests/test_layer_registry.py:594` | `Datastore.SQL.ClientPool` | yes |
| `Datastore/tests/test_layer_registry.py:595` | `Datastore.store_reader` | yes |
| `Datastore/tests/test_layer_registry.py:596` | `Datastore.store_inventory` | yes |
| `Datastore/tests/test_layer_registry.py:618` | `Datastore.SQL.ObjectFactories.base` | yes |
| `Datastore/tests/test_read_only_pool.py:882` | `Datastore.SQL.ShardedPool.sqlite3.connect` | no |
| `prompts/datastore-generic/orchestrator/measure-08/scripts/m5_imports.py:14` | `Datastore.store_inventory` | yes |
| `prompts/datastore-generic/orchestrator/measure-08/scripts/m5_imports.py:15` | `Datastore.store_reader` | yes |
| `prompts/datastore-generic/orchestrator/measure-08/scripts/m5_imports.py:16` | `Datastore.contract` | yes |

## `import_module` calls naming a layer module

| File:line | Call | Argument |
|---|---|---|
| `Datastore/tests/schema_description.py:185` | `import_module` | `Datastore.SQL.Datastore` |
| `Datastore/tests/standin_pool.py:43` | `import_module` | `Datastore.SQL.ShardedPool` |
| `Datastore/tests/standin_pool.py:44` | `import_module` | `Datastore.SQL.Datastore` |
| `Datastore/tests/standin_pool.py:45` | `import_module` | `Datastore.SQL.SerialPoolBroker` |
| `Datastore/tests/test_layer_registry.py:66` | `import_module` | `Datastore.SQL.Datastore` |
| `Datastore/tests/test_schema_builder.py:160` | `import_module` | `Datastore.SQL.Datastore` |
| `Datastore/tests/test_wkb_numeric_parent_key.py:88` | `import_module` | `Datastore.SQL.Datastore` |
| `docs/datastore-integrity-audit/replicated_write_fault_probe.py:37` | `import_module` | `Datastore.SQL.ShardedPool` |
| `docs/datastore-integrity-audit/replicated_write_fault_probe.py:38` | `import_module` | `Datastore.SQL.Datastore` |
| `docs/datastore-integrity-audit/replicated_write_fault_probe.py:39` | `import_module` | `Datastore.SQL.SerialPoolBroker` |
| `prompts/datastore-generic/orchestrator/measure-07/scripts/m2_open_behaviour.py:30` | `import_module` | `Datastore.SQL.ShardedPool` |

## Imports of factory modules, by group (count only)

| Group | Statements |
|---|---|
| `ComputeTargets/` | 17 |
| `Datastore/SQL/ObjectFactories/` | 2 |
| `Datastore/tests/` | 46 |
| `config/` | 20 |
| `docs/` | 15 |
| **total** | **100** |

## Files that do not parse

None.
