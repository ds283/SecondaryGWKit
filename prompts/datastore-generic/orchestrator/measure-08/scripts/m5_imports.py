"""M5: in a fresh interpreter, import Datastore.store_inventory and list the project modules loaded;
then import the registry (config.datastore) and list what that adds; then call read_inventory on a
store built beforehand and list what that adds. Also: inspect.getsource-free check of which modules
name CosmologyModels among those store_inventory itself imports.

usage: python m5_imports.py <tree> <primary of a store built beforehand>
"""
import json, sys

tree, primary = sys.argv[1], sys.argv[2]
sys.path.insert(0, tree)
PREFIXES = (
    "Datastore.SQL.ObjectFactories",
    "Datastore.store_inventory",
    "Datastore.store_reader",
    "Datastore.contract",
    "config",
    "ComputeTargets",
    "CosmologyConcepts",
    "CosmologyModels",
    "MetadataConcepts",
)


def snapshot():
    return {
        p: sorted(m for m in sys.modules if m == p or m.startswith(p + "."))
        for p in PREFIXES
    }


def diff(a, b):
    return {p: sorted(set(b[p]) - set(a[p])) for p in PREFIXES if set(b[p]) - set(a[p])}


s0 = snapshot()
import Datastore.store_inventory as si

s1 = snapshot()
from config.datastore import factories

s2 = snapshot()
inv = si.read_inventory(primary, factories)
s3 = snapshot()
import ray

print(json.dumps({
    "after import Datastore.store_inventory": diff(s0, s1),
    "after import config.datastore": {p: len(v) for p, v in diff(s1, s2).items()},
    "added by read_inventory": diff(s2, s3),
    "store_inventory module globals naming a project package": sorted(
        n for n, v in vars(si).items()
        if getattr(v, "__module__", "").split(".")[0] in ("CosmologyModels", "ComputeTargets", "CosmologyConcepts", "MetadataConcepts", "config")
    ),
    "classes read": len(inv.classes),
    "ray_initialized": ray.is_initialized(),
}, indent=1))
