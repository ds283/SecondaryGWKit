"""M2: the replicated set each kind of test store's primary records, against config.sharding.

usage: python m2_recorded_replicated.py <tree> <scratch-dir>
"""
import contextlib, io, shutil, sqlite3, sys
from pathlib import Path

tree, scratch = sys.argv[1], Path(sys.argv[2]) / "m2"
sys.path.insert(0, tree)
if scratch.exists():
    shutil.rmtree(scratch)
scratch.mkdir(parents=True)

from config.sharding import replicated_tables as configured
from Datastore.tests.real_store_fixtures import build_full_store, build_real_store
from Datastore.tests import shard_store_fixtures as ssf
from Datastore.tests import standin_pool as sp


def recorded(primary):
    c = sqlite3.connect(f"{Path(primary).as_uri()}?mode=ro", uri=True)
    try:
        return [r[1] for r in c.execute('SELECT serial, "table" FROM replicated_tables ORDER BY serial')]
    finally:
        c.close()


def compare(label, rows):
    print(f"{label}: {len(rows)} rows {rows}")
    print(f"   same set as config.sharding: {set(rows) == set(configured)}; same order: {rows == list(configured)}")
    if set(rows) != set(configured):
        print(f"   only recorded: {sorted(set(rows) - set(configured))}; only configured: {sorted(set(configured) - set(rows))}")


print(f"config.sharding.replicated_tables: {len(configured)} {list(configured)}")
(scratch / "real").mkdir()
compare("build_real_store", recorded(build_real_store(scratch / "real").primary))
(scratch / "full").mkdir()
compare("build_full_store", recorded(build_full_store(scratch / "full").primary))

cluster = sp.StandinCluster()
with cluster.active():
    pool = cluster.open_pool(scratch / "standin" / "store.sqlite")
    cluster.close_pool(pool)
compare("ShardedPool on stand-in shards (standin_pool.open_pool)", recorded(scratch / "standin" / "store.sqlite"))

ssf.write_hand_built_primary(scratch / "hand" / "store.sqlite", {0: "store-shard0000.sqlite"})
compare("shard_store_fixtures.write_hand_built_primary", recorded(scratch / "hand" / "store.sqlite"))
