"""M3: the copy unit and the prune unit of every compared table, at HEAD's rule and the new one.

usage: python m3_units.py <tree>   (run from the tree, PYTHONPATH=.)
At HEAD the old units are computed by the pool's own code where it has a function
(_prune_unit), and by the hard-coded copy-unit rule of _plan_replicated_repair (copied
below verbatim); in the worktree the new unit is _unit_tables.
"""
import json
import sys

import sqlalchemy as sqla

from config.datastore import factories
from config.sharding import replicated_tables, sharded_tables
from Datastore.SQL.schema import build_schema
from Datastore.SQL.ShardedPool import ShardedPool

built = build_schema(sqla.MetaData(), factories)
pool = object.__new__(ShardedPool)
pool._replicated_tables = list(replicated_tables)
pool._sharded_tables = dict(sharded_tables)
pool._factories = dict(factories)
specs = pool._replicated_table_specs(built)
by_name = {s["name"]: s for s in specs}


def old_copy_unit(cls_name):
    # verbatim from HEAD's _plan_replicated_repair
    unit = [cls_name]
    if cls_name == "BackgroundModel":
        unit = [
            t
            for t in ("BackgroundModel", "BackgroundModel_tags", "BackgroundModelValue")
            if t in by_name
        ]
    return unit


def old_prune_unit(cls_name):
    # verbatim from HEAD's _prune_unit
    unit = [cls_name]
    for spec in specs:
        name = spec["name"]
        if name == cls_name:
            continue
        if any(fk.column.table.name == cls_name for fk in built.tables[name].foreign_keys):
            unit.append(name)
    return unit


out = {"compared": [s["name"] for s in specs], "units": {}}
prune_classes = ShardedPool._replicated_prune_classes(pool._replicated_tables, built)
out["prune_classes"] = prune_classes
for spec in specs:
    name = spec["name"]
    row = {"old_copy_unit": old_copy_unit(name), "old_prune_unit": old_prune_unit(name)}
    if hasattr(ShardedPool, "_prune_unit"):
        row["pool_prune_unit"] = ShardedPool._prune_unit(name, built, specs)
        assert row["pool_prune_unit"] == row["old_prune_unit"]
    if hasattr(ShardedPool, "_unit_tables"):
        row["new_unit"] = ShardedPool._unit_tables(name, specs)
        row["new_equals_old_copy"] = row["new_unit"] == row["old_copy_unit"]
        row["new_equals_old_prune_as_set"] = set(row["new_unit"]) == set(row["old_prune_unit"])
        row["new_equals_old_prune_in_order"] = row["new_unit"] == row["old_prune_unit"]
    out["units"][name] = row
print(json.dumps(out, indent=1))
