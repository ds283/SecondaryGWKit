"""M8: the refusals the change adds or renames, with their text (after tree only).

usage: python m8_messages.py <tree> <scratch-dir>
"""
import shutil, sqlite3, sys, types
from pathlib import Path

tree, scratch = sys.argv[1], Path(sys.argv[2]) / "m8"
sys.path.insert(0, tree)
if scratch.exists():
    shutil.rmtree(scratch)
scratch.mkdir(parents=True)

from config.datastore import factories
from Datastore.store_reader import open_read_only
from Datastore.store_inventory import read_inventory
from Datastore.tests import standin_pool as sp
from Datastore.tests.real_store_fixtures import build_full_store, file_state


def show(label, fn):
    try:
        fn()
        print(f"[{label}] no exception")
    except Exception as e:
        print(f"[{label}] {type(e).__module__}.{type(e).__name__}: {e}")


def full(name):
    d = scratch / name
    d.mkdir()
    return build_full_store(d)


cluster = sp.StandinCluster()
with cluster.active():
    # a store built by the pool, for the pool cases
    built = scratch / "pool" / "store.sqlite"
    cluster.close_pool(cluster.open_pool(built))
    before = sp.store_checksums(built)

    missing = scratch / "nowhere" / "store.sqlite"
    cluster.calls.clear()
    show("read-only pool, drop_tables=['GkSourcePolicyData'], missing path",
         lambda: cluster.open_pool(missing, read_only=True, drop_tables=["GkSourcePolicyData"]))
    print("   created anything:", missing.parent.exists(), "actor calls:", len(cluster.calls))
    show("read-only pool, drop_tables=['GkSourcePolicyData'], existing store",
         lambda: cluster.open_pool(built, read_only=True, drop_tables=["GkSourcePolicyData"]))
    for ro in (False, True):
        cluster.calls.clear()
        show(f"read_only={ro}, drop_tables=['NotATable', 'GkSource'], missing path",
             lambda: cluster.open_pool(missing, read_only=ro, drop_tables=["NotATable", "GkSource"]))
        print("   created anything:", missing.parent.exists(), "actor calls:", len(cluster.calls))
        show(f"read_only={ro}, drop_tables=['NotATable'], existing store",
             lambda: cluster.open_pool(built, read_only=ro, drop_tables=["NotATable"]))
        print("   store unchanged:", sp.store_checksums(built) == before, "actor calls:", len(cluster.calls))
    # a drop table that is a registered class with no table? every registered class has a table
    # the actor's own refusal (reached only if the pool's is bypassed)
    raw = sp.DatastoreClass
    fake = types.SimpleNamespace(_tables={"GkSource": object()}, _my_name="shard0000-store")
    show("Datastore._drop_tables, undeclared", lambda: raw._drop_tables(fake, ["NotATable"]))

# the reader: a primary with no replicated_tables table
s = full("no-replicated-table")
c = sqlite3.connect(s.primary); c.execute("DROP TABLE replicated_tables"); c.commit(); c.close()
st = file_state(s.directory)


def enter():
    with open_read_only(s.primary, factories):
        pass


show("open_read_only, primary without replicated_tables", enter)
show("read_inventory, primary without replicated_tables", lambda: read_inventory(s.primary, factories))
print("   files unchanged:", file_state(s.directory) == st)
with sp.StandinCluster().active() as _:
    pass
cl = sp.StandinCluster()
with cl.active():
    show("read-only pool, primary without replicated_tables", lambda: cl.open_pool(s.primary, shards=2, read_only=True))
    show("read-write pool, primary without replicated_tables", lambda: cl.open_pool(s.primary, shards=2))
print("   files unchanged:", file_state(s.directory) == st)

# the reader: a recorded replicated class the registry does not declare
s = full("undeclared-replicated")
c = sqlite3.connect(s.primary); c.execute('INSERT INTO replicated_tables (serial, "table") VALUES (99, \'NotAClass\')'); c.commit(); c.close()
show("open_read_only, replicated_tables names NotAClass", lambda: (lambda cm: (cm.__enter__(), cm.__exit__(None, None, None)))(open_read_only(s.primary, factories)))

# the reader: a registry that lacks a class the store records as replicated
s = full("registry-lacks")
small = {k: v for k, v in factories.items() if k != "QuadSourcePolicy"}
show("open_read_only, registry without QuadSourcePolicy", lambda: (lambda cm: (cm.__enter__(), cm.__exit__(None, None, None)))(open_read_only(s.primary, small)))

# what the reader yields for a sound store
s = full("sound")
with open_read_only(s.primary, factories) as store:
    print("ReadOnlyStore.replicated_tables:", store.replicated_tables)

# the empty replicated set: a primary that records none
s = full("none-replicated")
c = sqlite3.connect(s.primary); c.execute("DELETE FROM replicated_tables"); c.commit(); c.close()
inv = read_inventory(s.primary, factories)
print("records none: replicated classes", sorted(n for n, x in inv.classes.items() if x.replicated),
      "; problems:", {n: list(x.problems) for n, x in inv.classes.items() if x.problems})
