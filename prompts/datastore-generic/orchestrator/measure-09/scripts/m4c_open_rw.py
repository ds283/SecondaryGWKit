"""M4(c): open an existing store read-write through <tree>'s stand-in pool (no Ray), record what
the constructor prints, its reconciliation and prune records, or what it raises; close it; and
give the store's checksums before and after.
Usage: python m4c_open_rw.py <tree> <primary> <version_label> <out.json>"""
import contextlib, datetime, hashlib, inspect, io, json, random, sys
from pathlib import Path
tree, primary, label, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
sys.path.insert(0, tree)
from Datastore.tests import standin_pool as sp
from config.sharding import replicated_tables, sharded_tables, shard_key_type, shard_key_wavenumber_store_id
assert sp.__file__.startswith(tree), sp.__file__

def checksums():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(primary.parent.iterdir()) if p.is_file()}

def safe(x):
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        items = [safe(v) for v in x]
        return sorted(items, key=repr) if isinstance(x, (set, frozenset)) else items
    if isinstance(x, (datetime.datetime, datetime.date)):
        return "<datetime>"
    if isinstance(x, (int, float, bool)) or x is None:
        return x
    return str(x)

kwargs = {}
if "factories" in inspect.signature(sp.sp_mod.ShardedPool.__init__).parameters:
    from config.datastore import factories
    kwargs["factories"] = factories
before = checksums()
random.seed(20261006)
cluster = sp.StandinCluster()
result = {"tree": tree, "label": label, "kwargs": sorted(kwargs)}
printed = io.StringIO()
with cluster.active():
    try:
        with contextlib.redirect_stdout(printed):
            pool = sp.sp_mod.ShardedPool(
                version_label=label, db_name=primary, ShardKeyType=shard_key_type,
                ShardKeyStoreIdGetter=shard_key_wavenumber_store_id,
                replicated_tables=replicated_tables, sharded_tables=sharded_tables, shards=2, **kwargs)
    except BaseException as e:
        result["raised"] = type(e).__name__
        result["message"] = str(e)
    else:
        result["reconciliation"] = safe(getattr(pool, "reconciliation", None))
        result["pruned"] = safe(getattr(pool, "pruned", None))
        v = getattr(pool, "_version", None)
        result["version"] = None if v is None else {"type": type(v).__name__, "store_id": v.store_id, "label": getattr(v, "label", None)}
        cluster.pool = pool
        cluster.close_pool(pool)
result["printed"] = printed.getvalue().splitlines()
after = checksums()
result["checksums_before"] = before
result["checksums_after"] = after
result["changed_files"] = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
out.write_text(json.dumps(result, indent=1, sort_keys=True))
print(json.dumps({k: result[k] for k in result if not k.startswith("checksums")}, indent=1, sort_keys=True))
