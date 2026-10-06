"""Per drop group: the tables that must also be dropped, under the full rule (FK + declared
parents, transitive) and under each mutation of it."""

import sqlalchemy as sqla
from config.datastore import factories, drop_groups
from Datastore.SQL.schema import build_schema
from Datastore.store_inventory import inventory_specs

T = build_schema(sqla.MetaData(), factories).tables
fk = {n: {f.column.table.name for f in t.foreign_keys} - {n} for n, t in T.items()}
dep = {c: set(s.dependencies()) - {c} for c, s in inventory_specs(factories).items()}


def closure(dropped, use_fk=True, use_dep=True, transitive=True):
    names = {n: set() for n in T}
    for n in T:
        if use_fk:
            names[n] |= fk[n]
        if use_dep:
            names[n] |= dep.get(n, set())
    gone, need = set(dropped), set()
    while True:
        more = {n for n, r in names.items() if n not in gone and r & gone}
        if not more:
            return need
        need |= more
        if not transitive:
            return need
        gone |= more


order = list(factories)
srt = lambda s: [n for n in order if n in s]
for g, tabs in drop_groups.items():
    full = closure(tabs)
    print(f"{g}: {srt(full)}")
    for lab, kw in (
        ("fk-only", dict(use_dep=False)),
        ("declared-only", dict(use_fk=False)),
        ("direct-only", dict(transitive=False)),
    ):
        alt = closure(tabs, **kw)
        if alt != full:
            print(f"    {lab}: missing {srt(full - alt)}")
