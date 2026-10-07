"""Every closed-store refusal's message, with <tree>'s code, for three broken stores: shard
records that are absolute paths (the resolver's refusal), a dropped `shards` table, and a missing
shard file (the file-problem refusal). The stores are built in an unresolved temporary directory,
so that a message naming the path as given and one naming it resolved can be told apart: the
resolved root is normalised to <R>, and the root as given to <T>. Writes {case: {operation: message}} as JSON.
Usage: python m3_messages.py <tree> <out.json>"""

import inspect, json, sqlite3, sys, tempfile
from pathlib import Path

tree, out = sys.argv[1], Path(sys.argv[2])
sys.path.insert(0, tree)
import Datastore
from Datastore.SQL.ShardedPool import ShardedPool
from Datastore.shard_paths import shard_file_name
from Datastore.store_reader import open_read_only
from Datastore.tests.shard_store_fixtures import (
    write_hand_built_primary,
    write_new_store,
    write_placeholder,
)
from config.datastore import factories

assert Datastore.__file__.startswith(tree), Datastore.__file__
takes_verb = "verb" in inspect.signature(ShardedPool._read_closed_store).parameters
print("tree", tree, "_read_closed_store takes verb:", takes_verb)


def shard_paths(primary, n=3):
    return {i: primary.parent / shard_file_name(primary, i) for i in range(n)}


def absolute(root):
    a = root / "A" / "store.sqlite"
    write_new_store(a, shards=3)
    b = root / "B" / "store.sqlite"
    write_hand_built_primary(
        b, {i: str(p) for i, p in shard_paths(a).items()}, shard_keys=[(7, 2)]
    )
    for p in shard_paths(b).values():
        write_placeholder(p)
    return b


def no_shards_table(root):
    p = root / "N" / "store.sqlite"
    write_new_store(p, shards=3)
    conn = sqlite3.connect(p)
    with conn:
        conn.execute("DROP TABLE shards")
    conn.close()
    return p


def missing_shard(root):
    p = root / "M" / "store.sqlite"
    write_new_store(p, shards=3)
    shard_paths(p)[1].unlink()
    return p


def message(call):
    try:
        call()
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return "<no exception>"


def read_it(p):
    with open_read_only(p, factories):
        pass


result = {}
with tempfile.TemporaryDirectory() as t:
    root = Path(t)
    assert str(root) != str(root.resolve()), "the root must be unresolved"
    for case, build in (
        ("absolute", absolute),
        ("no_shards_table", no_shards_table),
        ("missing_shard", missing_shard),
    ):
        p = build(root)
        dst = root / "D" / f"{case}.sqlite"
        args = (p.resolve(), "read") if takes_verb else (p.resolve(),)
        ops = {
            "_read_closed_store": lambda: ShardedPool._read_closed_store(*args),
            "closed_store_files": lambda: ShardedPool.closed_store_files(p),
            "delete_store": lambda: ShardedPool.delete_store(p),
            "copy_store": lambda: ShardedPool.copy_store(p, dst),
            "move_store": lambda: ShardedPool.move_store(p, dst),
            "open_read_only": lambda: read_it(p),
        }
        result[case] = {}
        for op, call in ops.items():
            m = message(call)
            m = m.replace(str(root.resolve()), "<R>").replace(str(root), "<T>")
            result[case][op] = m
        assert p.exists() and not dst.exists(), (case, "something was written")
out.write_text(json.dumps(result, indent=1, sort_keys=True))
