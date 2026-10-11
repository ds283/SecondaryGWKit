"""
Shared scaffolding for the lookup-key / read-path probes of the datastore-integrity audit.

Builds every table the factories declare (``Datastore.SQL.schema.build_schema`` over
``Datastore.SQL.Datastore._factories``, the same definitions production uses) in an **in-memory**
SQLite database. No Ray is initialised, no file is opened, nothing under ``var/`` is touched.

Run from the repository root:

    PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-audit/lookup_keys/<probe>.py
"""

import warnings
from datetime import datetime

import sqlalchemy as sqla

from datastorekit.SQL.Datastore import _factories
from datastorekit.SQL.schema import build_schema
from Units import Mpc_units

UNITS = Mpc_units()


class Store:
    def __init__(self):
        self.metadata = sqla.MetaData()
        built = build_schema(self.metadata, _factories)
        self.tables = built.tables
        self.records = built.records
        self.engine = sqla.create_engine("sqlite://", future=True)
        self.metadata.create_all(self.engine)
        self.conn = self.engine.connect()
        self.inserters = {n: self._inserter(n) for n in self.tables}

    def _inserter(self, name):
        table = self.tables[name]
        rec = self.records[name]

        def insert(conn, data):
            payload = dict(data)
            if rec.get("use_timestamp", False):
                payload["timestamp"] = datetime(2026, 9, 25)
            if rec.get("use_version", False):
                payload.setdefault("version", 1)
            return conn.execute(sqla.insert(table), payload).lastrowid

        return insert

    def insert(self, name, **row):
        return self.inserters[name](self.conn, row)

    def compile(self, query):
        return str(
            query.compile(
                dialect=self.engine.dialect, compile_kwargs={"literal_binds": True}
            )
        )


class RecordingConn:
    """Wraps a Connection: records each executed statement (compiled) and any warnings."""

    def __init__(self, store: Store):
        self.store = store
        self.statements = []
        self.warnings = []

    def execute(self, stmt, *args, **kwargs):
        try:
            sql = self.store.compile(stmt)
        except Exception as e:  # noqa: BLE001 -- recorded, not swallowed silently
            sql = f"<could not compile: {type(e).__name__}: {e}>"
        self.statements.append(sql)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self.store.conn.execute(stmt, *args, **kwargs)
        self.warnings.extend(f"{w.category.__name__}: {w.message}" for w in caught)
        return result


class StandInCosmology:
    type_id = 0

    def __init__(self, store_id=1):
        self.store_id = store_id
        self.units = UNITS


class StandInModelProxy:
    def __init__(self, store_id=1):
        self.store_id = store_id
        self.units = UNITS
        self.cosmology = StandInCosmology()


class Ref:
    """Anything a factory only reads ``.store_id`` off."""

    def __init__(self, store_id, **extra):
        self.store_id = store_id
        for k, v in extra.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"Ref({self.store_id})"
