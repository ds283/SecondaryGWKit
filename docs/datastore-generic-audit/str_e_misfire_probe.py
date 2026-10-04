"""
Does an old-store refusal that tests ``"<column>" in str(e)`` fire on an error that has nothing to
do with a missing column?

datastore-generic audit, finding F0. Several factories (TkNumericIntegration, GkNumericIntegration,
QCD_Cosmology, BackgroundModel) decide that a datastore predates a column by looking for the
column's name in ``str(e)`` of a SQLAlchemyError. SQLAlchemy's ``str()`` of a DBAPI error carries
the statement text, and the statement names the column. This probe raises two errors on a table
whose schema is current -- one that is not about the column at all ("no such table" for a table
the statement joins, and "database is locked" from a second writer) -- and prints whether each
factory's test would match.

Needs no Ray and no datastore under var/; everything is in a temporary directory.
"""

import os
import sqlite3
import tempfile

import sqlalchemy as sqla
from sqlalchemy.exc import SQLAlchemyError

TESTS = {
    "TkNumericIntegration / GkNumericIntegration": ("break_point_kind",),
    "QCD_Cosmology": ("T_z_representation",),
    "BackgroundModel (read path re-queries without grid identity)": (
        "source_grid_digest",
        "source_grid_construction",
    ),
}


def report(label, e):
    text = str(e)
    print(f"--- {label}")
    print(f"    str(e) first line: {text.splitlines()[0]}")
    for factory, columns in TESTS.items():
        hit = any(c in text for c in columns)
        print(f"    {factory}: test matches -> {hit}")


with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "probe.sqlite")
    engine = sqla.create_engine(f"sqlite:///{path}", connect_args={"timeout": 0.1})
    with engine.begin() as conn:
        # every column the factories' queries name is present: this is a current schema
        conn.exec_driver_sql(
            "CREATE TABLE t (serial INTEGER PRIMARY KEY, break_point_kind TEXT, "
            "T_z_representation TEXT, source_grid_digest TEXT, source_grid_construction TEXT)"
        )

    # 1. an error not about the column: the statement joins a table that does not exist
    statement = sqla.text(
        "SELECT t.break_point_kind, t.T_z_representation, t.source_grid_digest, "
        "t.source_grid_construction FROM t JOIN absent_table a ON a.serial = t.serial"
    )
    try:
        with engine.connect() as conn:
            conn.execute(statement)
    except SQLAlchemyError as e:
        report("no such table (every queried column exists)", e)

    # 2. an error not about the schema at all: another connection holds the write lock
    blocker = sqlite3.connect(path)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        with engine.connect() as conn:
            conn.execute(
                sqla.text(
                    "SELECT break_point_kind, T_z_representation, source_grid_digest, "
                    "source_grid_construction FROM t"
                )
            )
    except SQLAlchemyError as e:
        report("database is locked (schema is current)", e)
    finally:
        blocker.rollback()
        blocker.close()
    engine.dispose()
