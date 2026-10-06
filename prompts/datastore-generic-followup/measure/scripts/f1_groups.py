import sqlalchemy as sqla
from config.datastore import factories, drop_groups
from Datastore.SQL.schema import build_schema
from Datastore.store_inventory import inventory_specs

built = build_schema(sqla.MetaData(), factories)
T = built.tables
fk = {}  # referencing table -> set of referenced tables
for name, t in T.items():
    fk[name] = {
        f.column.table.name for f in t.foreign_keys if f.column.table.name != name
    }
specs = inventory_specs(factories)
dep = {}
for cls, spec in specs.items():
    dep[cls] = set(spec.dependencies()) - {cls}
nonfk = {c: d - fk.get(c, set()) for c, d in dep.items()}
print("non-FK inventory dependencies:")
for c, d in nonfk.items():
    if d:
        print(" ", c, "->", sorted(d))
print()
for g, tabs in drop_groups.items():
    s = set(tabs)
    by_fk = sorted((r, sorted(fk[r] & s)) for r in T if r not in s and fk[r] & s)
    by_dep = sorted(
        (r, sorted(nonfk.get(r, set()) & s))
        for r in dep
        if r not in s and nonfk.get(r, set()) & s
    )
    print(f"{g}: {tabs}")
    print("   FK from outside:", by_fk)
    print("   non-FK parents from outside:", by_dep)
