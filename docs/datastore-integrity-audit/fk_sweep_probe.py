"""Read-only probe: build every factory's table in an in-memory SQLite database and list
integer reference-like columns with their declared foreign keys, nullability and whether
they are in the factory's lookup key (unique constraint / index). Writes nothing to disk.
"""

import sqlalchemy as sqla

from datastorekit.SQL.Datastore import _factories
from datastorekit.SQL.schema import build_schema
from config.sharding import replicated_tables, sharded_tables

md = sqla.MetaData()
built = build_schema(md, _factories)
eng = sqla.create_engine("sqlite://")
md.create_all(eng)

with eng.connect() as conn:
    print(
        "PRAGMA foreign_keys (default engine) =",
        conn.exec_driver_sql("PRAGMA foreign_keys").scalar(),
    )

for name, tab in built.tables.items():
    where = (
        "replicated"
        if name in replicated_tables
        else (
            f"sharded by {sharded_tables[name]}"
            if name in sharded_tables
            else "UNLISTED (written by parent's Datastore actor)"
        )
    )
    print(f"\n== {name}  [{where}]")
    uniques = [
        tuple(c.name for c in con.columns)
        for con in tab.constraints
        if isinstance(con, sqla.UniqueConstraint)
    ]
    idx = [(i.name, tuple(c.name for c in i.columns), i.unique) for i in tab.indexes]
    for col in tab.columns:
        if not isinstance(col.type, sqla.Integer):
            continue
        if col.name in ("stepping",) or col.primary_key:
            continue
        fks = [fk.target_fullname for fk in col.foreign_keys]
        print(f"   {col.name:32s} FK={fks or '-'} nullable={col.nullable}")
    for u in uniques:
        print("   UNIQUE", u)
    for i in idx:
        if i[2]:
            print("   UNIQUE INDEX", i)
