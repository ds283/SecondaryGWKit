"""M5b: in a fresh interpreter with no registry imported, call store_inventory.read_records on a
one-table in-memory SQLite database (a generic stand-in class, no project table) and list the
project modules loaded. The base tree's read_records calls cosmology_classes() on every call.

usage: python m5b_read_records_alone.py <tree>
"""
import json, sys

sys.path.insert(0, sys.argv[1])
import sqlalchemy as sqla
from Datastore.store_inventory import ShardContext, read_records

md = sqla.MetaData()
t = sqla.Table("thing", md, sqla.Column("serial", sqla.Integer, primary_key=True), sqla.Column("x", sqla.Float))
engine = sqla.create_engine("sqlite://")
md.create_all(engine)
with engine.begin() as conn:
    conn.execute(t.insert(), [{"serial": 1, "x": 0.5}])
    read = read_records(conn, t, {"thing": t}, ShardContext(shard=0, keys={}, digests={}), leaves=("x",))
loaded = sorted(m for m in sys.modules if m.split(".")[0] in
                ("CosmologyModels", "ComputeTargets", "CosmologyConcepts", "MetadataConcepts", "config")
                or m.startswith("Datastore.SQL.ObjectFactories"))
print(json.dumps({"records": [r.as_json() for _, r, _ in read.rows], "project modules loaded": loaded}))
