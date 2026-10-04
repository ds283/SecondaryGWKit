"""Measured fact 2: the StoreSchemaMismatch messages."""

import contextlib
import io
import sqlite3
import tempfile
from pathlib import Path

import sqlalchemy as sqla

from Datastore.SQL.Datastore import _factories
from Datastore.SQL.schema import build_schema, schema_differences, StoreSchemaMismatch
from Datastore.store_reader import open_read_only
from Datastore.tests.real_store_fixtures import build_real_store
from Datastore.tests import standin_pool as sp

SCRATCH = Path(__file__).resolve().parents[1] / "tmp"
SCRATCH.mkdir(exist_ok=True)


def reader(label, **kw):
    with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
        store = build_real_store(Path(d), **kw)
        try:
            with open_read_only(store.primary):
                print(f"[reader] {label}: OPENED (no refusal)")
        except StoreSchemaMismatch as e:
            print(f"[reader] {label}: {type(e).__name__}: {str(e).replace(d, '<tmp>')}")
        except Exception as e:
            print(f"[reader] {label}: OTHER {type(e).__name__}: {e}")


reader(
    "shard 1 missing OneLoopIntegral_tags", missing_tables={1: ["OneLoopIntegral_tags"]}
)
reader(
    "shard 0 TkNumericIntegration missing stop_Tprime",
    missing_columns={0: {"TkNumericIntegration": ["stop_Tprime"]}},
)
reader(
    "shard 1 extra table retired_table",
    extra_sql={1: ["CREATE TABLE retired_table (x INTEGER)"]},
)
reader(
    "shard 1 extra column redshift.legacy",
    extra_sql={1: ['ALTER TABLE "redshift" ADD COLUMN legacy INTEGER']},
)
reader(
    "shard 0 several at once",
    missing_tables={0: ["OneLoopIntegral_tags"]},
    missing_columns={0: {"TkNumericIntegration": ["stop_Tprime"]}},
    extra_sql={
        0: [
            "CREATE TABLE retired_table (x INTEGER)",
            'ALTER TABLE "redshift" ADD COLUMN legacy INTEGER',
        ]
    },
)


# read-write (and read-only pool) through the real ShardedPool on stand-in shards
def execute(path, *stmts):
    c = sqlite3.connect(path)
    try:
        for s in stmts:
            c.execute(s)
        c.commit()
    finally:
        c.close()


def pool_case(label, mutate, read_only=False):
    with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
        cluster = sp.StandinCluster()
        with cluster.active():
            primary = Path(d) / "store.sqlite"
            pool = cluster.open_pool(primary)
            files = dict(pool._shard_db_files)
            cluster.close_pool(pool)
            # a current store written by the current code passes the comparison
            built = build_schema(sqla.MetaData(), _factories)
            for sid, path in sorted(files.items()):
                eng = sqla.create_engine(
                    f"sqlite:///file:{path}?mode=ro&uri=true", future=True
                )
                with eng.connect() as conn:
                    diffs = schema_differences(conn, built.tables)
                eng.dispose()
                assert diffs.empty, (sid, diffs)
            mutate(primary, files)
            try:
                p, out = cluster.open_pool_output(primary, read_only=read_only)
                print(f"[pool read_only={read_only}] {label}: OPENED")
                if not read_only:
                    tabs = (
                        sqlite3.connect(files[1])
                        .execute(
                            "SELECT name FROM sqlite_master WHERE name='OneLoopIntegral_tags'"
                        )
                        .fetchall()
                    )
                    print(f"    shard 1 OneLoopIntegral_tags after open: {tabs}")
                cluster.close_pool(p)
            except StoreSchemaMismatch as e:
                print(
                    f"[pool read_only={read_only}] {label}: {type(e).__name__}: {str(e).replace(d, '<tmp>')}"
                )
            except Exception as e:
                print(
                    f"[pool read_only={read_only}] {label}: OTHER {type(e).__name__}: {str(e).replace(d, '<tmp>')}"
                )


cases = [
    (
        "shard 1 missing OneLoopIntegral_tags",
        lambda pr, f: execute(f[1], 'DROP TABLE "OneLoopIntegral_tags"'),
    ),
    (
        "shard 0 TkNumericIntegration missing stop_Tprime",
        lambda pr, f: execute(
            f[0], 'ALTER TABLE "TkNumericIntegration" DROP COLUMN "stop_Tprime"'
        ),
    ),
    (
        "shard 1 extra table retired_table",
        lambda pr, f: execute(f[1], "CREATE TABLE retired_table (x INTEGER)"),
    ),
    (
        "shard 1 extra column redshift.legacy",
        lambda pr, f: execute(f[1], 'ALTER TABLE "redshift" ADD COLUMN legacy INTEGER'),
    ),
    (
        "primary without replication_in_flight",
        lambda pr, f: execute(pr, 'DROP TABLE "replication_in_flight"'),
    ),
    (
        "primary with an extra column on shards",
        lambda pr, f: execute(pr, 'ALTER TABLE "shards" ADD COLUMN legacy INTEGER'),
    ),
]
for ro in (False, True):
    for label, m in cases:
        pool_case(label, m, read_only=ro)
