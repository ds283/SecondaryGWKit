"""M9: the factory's revalidate against validate, on a copy of a build_full_store shard.

usage (from the tree, PYTHONPATH=.): python m9_revalidate.py <shard copy>
For each BackgroundModel row, as stored and with one value row deleted, inside a savepoint that
is rolled back: validate (through the stand-in object HEAD's _recompute_validated built) and, where
the factory has it, revalidate. Records the return value, the flag the call wrote and every line
it printed. The copy is opened read-write and every savepoint rolled back; its checksum is printed
before and after.
"""
import contextlib
import hashlib
import io
import json
import sys

import sqlalchemy as sqla

from config.datastore import factories
from Datastore.object import DatastoreObject
from Datastore.SQL.schema import build_schema

path = sys.argv[1]


def checksum():
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


class _StoredModel(DatastoreObject):
    """HEAD's stand-in, verbatim: what the factory's validate reads of a model."""

    def __init__(self, store_id: int, label):
        super().__init__(store_id)
        self.label = label


before = checksum()
built = build_schema(sqla.MetaData(), factories)
factory = factories["BackgroundModel"]
table = built.tables["BackgroundModel"]
values = built.tables["BackgroundModelValue"]
engine = sqla.create_engine(f"sqlite:///{path}", future=True)
out = []
with engine.connect() as conn:
    models = conn.execute(sqla.select(table.c.serial, table.c.label).order_by(table.c.serial)).all()
    for serial, label in models:
        for variant in ("as stored", "one value row deleted"):
            for how in ("validate", "revalidate"):
                if how == "revalidate" and not hasattr(factory, "revalidate"):
                    continue
                if how == "revalidate" and "NotImplementedError" in repr(factory.revalidate):
                    continue
                sp = conn.begin_nested()
                try:
                    if variant != "as stored":
                        victim = conn.execute(
                            sqla.select(sqla.func.max(values.c.serial)).filter(
                                values.c.model_serial == serial
                            )
                        ).scalar()
                        conn.execute(sqla.delete(values).where(values.c.serial == victim))
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        if how == "validate":
                            result = factory.validate(
                                _StoredModel(serial, label), conn=conn, table=table, tables=built.tables
                            )
                        else:
                            result = factory.revalidate(serial, conn, table, built.tables)
                    written = conn.execute(
                        sqla.select(table.c.validated).filter(table.c.serial == serial)
                    ).scalar()
                finally:
                    sp.rollback()
                out.append(
                    {
                        "serial": serial,
                        "variant": variant,
                        "call": how,
                        "returned": result,
                        "written": written,
                        "printed": buf.getvalue().splitlines(),
                    }
                )
    conn.rollback()
engine.dispose()
after = checksum()
print(json.dumps({"checksum_before": before, "checksum_after": after, "calls": out}, indent=1))
