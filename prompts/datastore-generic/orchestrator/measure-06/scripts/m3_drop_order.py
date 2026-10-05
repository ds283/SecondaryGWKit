"""M3: the foreign-key-derived drop order, against the old _drop_order.

usage (from the worktree): python m3_drop_order.py <after-tree> <before-tree> <scratch-dir>
"""
import ast, shutil, sqlite3, sys, warnings
from pathlib import Path

after, before, scratch = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, after)
import sqlalchemy as sqla
from config.datastore import drop_groups, factories, tables_to_drop
from Datastore.SQL.schema import build_schema

# the old registries, read from the before tree's source (never imported beside the new tree)
src = (before / "Datastore/SQL/Datastore.py").read_text()
old = {}
for node in ast.parse(src).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id in ("_drop_actions", "_drop_order"):
            old[node.targets[0].id] = ast.literal_eval(node.value)
assert old["_drop_actions"] == drop_groups, "drop groups moved verbatim"
tables = build_schema(sqla.MetaData(), factories).tables


def derived(names):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        order = [t.name for t in reversed(sqla.schema.sort_tables([tables[n] for n in names]))]
    return order, [str(x.message) for x in w]


def old_order(actions):
    out = []
    for action in old["_drop_order"]:
        if action in actions:
            out += old["_drop_actions"][action]
    return out


def fk_pairs(names):
    pairs = []
    for n in names:
        for fk in tables[n].foreign_keys:
            t = fk.column.table.name
            if t in names and t != n:
                pairs.append((n, fk.parent.name, t))
            if t == n:
                pairs.append((n, fk.parent.name, t + " (self)"))
    return pairs


def violations(order, pairs):
    return [p for p in pairs if not p[2].endswith("(self)") and order.index(p[0]) > order.index(p[2])]


cases = {a: [a] for a in drop_groups}
cases["ALL"] = list(drop_groups)
for case, actions in cases.items():
    names = tables_to_drop(actions)
    order, warns = derived(names)
    pairs = fk_pairs(names)
    print(f"== {case}: {names}")
    print(f"   derived: {order}")
    print(f"   sort_tables warnings: {warns}")
    print(f"   FK pairs among dropped (child.col -> parent): {[f'{a}.{c}->{b}' for a,c,b in pairs]}")
    print(f"   derived violates: {violations(order, pairs)}")
    oo = old_order(actions)
    print(f"   old order: {oo}")
    print(f"   old violates: {[f'{a}.{c}->{b}' for a,c,b in violations(oo, pairs)]}")

# FKs from a dropped table to one not dropped, and from a kept table into a dropped one
all_names = tables_to_drop(list(drop_groups))
into = sorted({(t.name, fk.parent.name, fk.column.table.name) for t in tables.values() for fk in t.foreign_keys
               if t.name not in all_names and fk.column.table.name in all_names})
print("== FKs from a table no group drops into a dropped table:", into)

# enforcement on a copy of a build_full_store shard holding rows
from Datastore.tests.real_store_fixtures import build_full_store
d = scratch / "m3"
if d.exists():
    shutil.rmtree(d)
d.mkdir(parents=True)
store = build_full_store(d / "src")  if False else None
(d / "src").mkdir()
store = build_full_store(d / "src")


def counts(path, names):
    c = sqlite3.connect(path)
    try:
        return {n: c.execute(f'SELECT count(*) FROM "{n}"').fetchone()[0] for n in names}
    finally:
        c.close()


for sid, shard in sorted(store.shard_files.items()):
    print(f"== shard {sid} rows in dropped tables:", {k: v for k, v in counts(shard, all_names).items() if v})


def try_drop(order, label, sid):
    path = d / f"{label}-shard{sid}.sqlite"
    shutil.copy(store.shard_files[sid], path)
    c = sqlite3.connect(path, isolation_level=None)
    c.execute("PRAGMA foreign_keys = ON")
    assert c.execute("PRAGMA foreign_key_check").fetchall() == [], "clean before"
    done = []
    try:
        for n in order:
            c.execute(f'DROP TABLE "{n}"')
            done.append(n)
        print(f"   {label} shard {sid}: every drop succeeded ({len(done)} tables); foreign_key_check after: {c.execute('PRAGMA foreign_key_check').fetchall()[:3]}")
    except sqlite3.Error as e:
        print(f"   {label} shard {sid}: failed at DROP \"{n}\" after {len(done)}: {type(e).__name__}: {e}")
    finally:
        c.close()


print("== PRAGMA foreign_keys=ON, every group dropped:")
for sid in sorted(store.shard_files):
    try_drop(derived(all_names)[0], "derived", sid)
    try_drop(old_order(list(drop_groups)), "old", sid)
    try_drop(all_names, "group-listed-order", sid)

# per group alone, both orders, on each shard
print("== per group, PRAGMA foreign_keys=ON:")
for a in drop_groups:
    for sid in sorted(store.shard_files):
        names = tables_to_drop([a])
        for label, order in (("derived", derived(names)[0]), ("old", old_order([a]))):
            path = d / f"g-{a}-{label}-{sid}.sqlite"
            shutil.copy(store.shard_files[sid], path)
            c = sqlite3.connect(path, isolation_level=None)
            c.execute("PRAGMA foreign_keys = ON")
            try:
                for n in order:
                    c.execute(f'DROP TABLE "{n}"')
                res = "ok"
            except sqlite3.Error as e:
                res = f"FAILED at {n}: {e}"
            finally:
                c.close()
            print(f"   {a} {label} shard {sid}: {res}")
