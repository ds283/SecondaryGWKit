"""M4: the pool's version object, on a fresh and an existing store, read-write and read-only.

usage (from the tree, PYTHONPATH=.): python m4_version_object.py
"""
import json
import tempfile
from pathlib import Path

from Datastore.tests.standin_pool import StandinCluster


def describe(v):
    attrs = {k: repr(val) for k, val in sorted(vars(v).items())}
    return {
        "type": f"{type(v).__module__}.{type(v).__qualname__}",
        "store_id": v.store_id,
        "label": v.label,
        "_deserialized": getattr(v, "_deserialized", "<absent>"),
        "_new_insert": getattr(v, "_new_insert", "<absent>"),
        "attributes": attrs,
    }


out = {}
with tempfile.TemporaryDirectory() as d:
    primary = Path(d) / "store.sqlite"
    cluster = StandinCluster()
    with cluster.active():
        pool = cluster.open_pool(primary)
        out["fresh, read-write"] = describe(pool._version)
        cluster.close_pool(pool)
        pool = cluster.open_pool(primary)
        out["existing, read-write"] = describe(pool._version)
        cluster.close_pool(pool)
        pool = cluster.open_pool(primary, read_only=True)
        out["existing, read-only"] = describe(pool._version)
        cluster.close_pool(pool)
print(json.dumps(out, indent=1))
