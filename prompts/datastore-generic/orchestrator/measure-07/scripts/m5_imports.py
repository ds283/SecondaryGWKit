"""M5: in a fresh interpreter, import one layer module and list the project modules loaded.

usage: python m5_imports.py <tree> <module>
"""
import json, sys

tree, module = sys.argv[1], sys.argv[2]
sys.path.insert(0, tree)
import importlib

importlib.import_module(module)
prefixes = (
    "Datastore.SQL.ObjectFactories",
    "config",
    "ComputeTargets",
    "CosmologyConcepts",
    "CosmologyModels",
    "MetadataConcepts",
)
loaded = {
    p: sorted(m for m in sys.modules if m == p or m.startswith(p + "."))
    for p in prefixes
}
import ray

print(json.dumps({"module": module, "loaded": loaded, "ray_initialized": ray.is_initialized()}))
