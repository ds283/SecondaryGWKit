"""Measured fact 1: a fixture store has no schema difference on any shard or the primary."""

import tempfile
from pathlib import Path

import sqlalchemy as sqla

from Datastore.SQL.Datastore import _factories
from Datastore.SQL.schema import build_schema, schema_differences
from Datastore.tests.real_store_fixtures import build_real_store, build_full_store
from Datastore.tests.shard_store_fixtures import bare_pool

SCRATCH = Path(__file__).resolve().parents[1] / "tmp"
SCRATCH.mkdir(exist_ok=True)

built = build_schema(sqla.MetaData(), _factories)
p = bare_pool(Path("/nonexistent/x.sqlite"))
p._create_engine()
primary_tables = dict(p._metadata.tables)
p._engine.dispose()
print("primary tables declared:", sorted(primary_tables))

for builder in (build_real_store, build_full_store):
    with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
        store = builder(Path(d))
        for serial, path in sorted(store.shard_files.items()):
            eng = sqla.create_engine(
                f"sqlite:///file:{path}?mode=ro&uri=true", future=True
            )
            with eng.connect() as conn:
                diffs = schema_differences(conn, built.tables)
            eng.dispose()
            print(
                builder.__name__,
                f"shard {serial}:",
                tuple(diffs),
                "empty" if diffs.empty else "DIFFERS",
            )
        eng = sqla.create_engine(
            f"sqlite:///file:{store.primary}?mode=ro&uri=true", future=True
        )
        with eng.connect() as conn:
            diffs = schema_differences(conn, primary_tables)
        eng.dispose()
        print(
            builder.__name__,
            "primary:",
            tuple(diffs),
            "empty" if diffs.empty else "DIFFERS",
        )
