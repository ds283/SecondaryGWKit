"""M3b: a read-write pool on stand-in shards drops every group from a full store; the result.

usage: python m3b_pool_drop.py <tree> <label> <scratch-dir>
"""
import inspect, json, shutil, sqlite3, sys
from pathlib import Path

tree, label, scratch = sys.argv[1], sys.argv[2], Path(sys.argv[3]) / "m3b"
sys.path.insert(0, tree)
d = scratch / label
if d.exists():
    shutil.rmtree(d)
d.mkdir(parents=True)
from Datastore.tests.real_store_fixtures import build_full_store
from Datastore.tests import standin_pool as sp
import Datastore.SQL.ShardedPool as spm

store = build_full_store(d)
params = inspect.signature(spm.ShardedPool.__init__).parameters
if "drop_tables" in params:
    from config.datastore import drop_groups, tables_to_drop
    kwargs = {"drop_tables": tables_to_drop(list(drop_groups))}
else:
    from Datastore.SQL.Datastore import _drop_actions
    kwargs = {"drop_actions": list(_drop_actions)}
cluster = sp.StandinCluster()
with cluster.active():
    pool, printed = cluster.open_pool_output(store.primary, shards=2, **kwargs)
    cluster.close_pool(pool)
drops = [l for l in printed.splitlines() if "dropping" in l]
out = {}
for sid, path in sorted(store.shard_files.items()):
    c = sqlite3.connect(path)
    master = sorted(c.execute("SELECT type, name, tbl_name, sql FROM sqlite_master").fetchall(), key=lambda r: (r[0], r[1]))
    tables = [r[1] for r in master if r[0] == "table"]
    counts = {t: c.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in tables}
    fk = c.execute("PRAGMA foreign_key_check").fetchall()
    c.close()
    out[sid] = {"master": master, "counts": counts, "fk_check": fk}
(scratch / f"result-{label}.json").write_text(json.dumps(out, indent=1, sort_keys=True))
print(label, "drop lines printed per shard:", len(drops) // 2, "first:", drops[:2])
for sid in out:
    nonzero = {t: n for t, n in out[sid]["counts"].items() if n}
    print(label, f"shard {sid}: {len(out[sid]['master'])} schema objects, non-empty tables {nonzero}, foreign_key_check {out[sid]['fk_check'][:3]}")
