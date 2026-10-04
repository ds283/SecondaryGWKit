"""U1: an absent table on a read-write open is allowed and created by the actors."""

import sqlite3
import tempfile
from pathlib import Path

import sqlalchemy as sqla

from Datastore.SQL.Datastore import _factories
from Datastore.SQL.schema import build_schema, schema_differences
from Datastore.tests import standin_pool as sp

SCRATCH = Path(__file__).resolve().parents[1] / "tmp"
SCRATCH.mkdir(exist_ok=True)
built = build_schema(sqla.MetaData(), _factories)


def execute(path, *stmts):
    c = sqlite3.connect(path)
    try:
        for s in stmts:
            c.execute(s)
        c.commit()
    finally:
        c.close()


def case(label, mutate):
    with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
        cluster = sp.StandinCluster()
        with cluster.active():
            primary = Path(d) / "store.sqlite"
            pool = cluster.open_pool(primary)
            files = dict(pool._shard_db_files)
            cluster.close_pool(pool)
            mutate(files)
            try:
                p, out = cluster.open_pool_output(primary)
                diffs = {}
                for sid, path in files.items():
                    eng = sqla.create_engine(
                        f"sqlite:///file:{path}?mode=ro&uri=true", future=True
                    )
                    with eng.connect() as conn:
                        diffs[sid] = schema_differences(conn, built.tables).empty
                    eng.dispose()
                labels = {
                    sid: sqlite3.connect(path)
                    .execute("SELECT label FROM version")
                    .fetchall()
                    for sid, path in files.items()
                }
                print(
                    f"{label}: OPENED; every shard matches the schema afterwards: {diffs}; version rows {labels}"
                )
                cluster.close_pool(p)
            except Exception as e:
                print(
                    f"{label}: {type(e).__name__}: {str(e).replace(d, '<tmp>')[:600]}"
                )


case(
    "interrupted drop, sharded tables (gk-source) gone from shard 0",
    lambda f: execute(
        f[0],
        *[
            f'DROP TABLE "{t}"'
            for t in ["GkSourceValue", "GkSource_tags", "GkSource_parents", "GkSource"]
        ],
    ),
)
case(
    "interrupted drop, replicated GkSourcePolicy gone from shard 1 (empty elsewhere)",
    lambda f: execute(f[1], 'DROP TABLE "GkSourcePolicy"'),
)
case(
    "interrupted first open: shard 2 lacks half its tables, no version row anywhere",
    lambda f: (
        [execute(p, "DELETE FROM version") for p in f.values()],
        execute(
            f[2],
            *[
                f'DROP TABLE "{t}"'
                for t in list(built.tables)[len(built.tables) // 2 :][::-1]
            ],
        ),
    ),
)
