"""
A store whose files differ from what the code declares is refused (prompts/datastore-generic,
prompt 05, S1–S3): one comparison, ``Datastore.SQL.schema.schema_differences``, and one refusal,
``StoreSchemaMismatch``.

1. ``schema_differences``: every shard and the primary of ``build_real_store`` and
   ``build_full_store``, and of a store a fresh ``ShardedPool`` created, have no difference; each of
   the four kinds (a table the file lacks, a table it has that is not declared, a declared column a
   table lacks, a column that is not declared) is found on the right shard, by name, in
   declaration or file order; the comparison issues no write.
2. The reader: a refusal yields nothing and disposes every engine it made. (Each kind is refused
   by name in ``test_store_reader.TestADifferingSchemaIsRefused``.)
3. ``read_inventory`` refuses a store missing a table, as ``store fingerprint`` does through it.
4. The read-write open, through ``standin_pool``: an absent column, an extra column and an extra
   table are each refused, naming the shard, before any actor is constructed and before the
   ``replication_in_flight`` record is read, with every file unchanged; a prune record left on the
   primary is not completed.
5. U1: a read-write open of a shard that lacks tables opens, and the actors create them. The
   ``gk-source`` tables dropped from one shard (an interrupted drop of a sharded class), the
   replicated ``GkSourcePolicy`` dropped from one shard (an interrupted
   ``--drop gk-source-policy-records``), and one shard lacking half its tables with no version row
   anywhere (an interrupted first open). Afterwards every shard has no difference and holds the
   version row.
6. The primary, read-write and read-only: an extra table and a missing column are each refused,
   naming "the primary", before any actor is constructed, with every file unchanged. (The missing
   ``replication_in_flight`` table is
   ``test_reconcile_at_open.TestNoRecordRefuses.test_a_primary_without_the_record_table_is_refused``.)

No Ray; every store is built in a temporary directory, and nothing under ``var/`` is opened.
"""

import contextlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ray
import sqlalchemy as sqla

from config.datastore import factories as _factories
from Datastore.SQL.schema import (
    SchemaDifferences,
    StoreSchemaMismatch,
    build_schema,
    schema_differences,
)
from Datastore.store_inventory import read_inventory
from Datastore.store_reader import open_read_only, read_only_url
from Datastore.tests import standin_pool as sp
from Datastore.tests.real_store_fixtures import (
    build_full_store,
    build_real_store,
    file_state,
)
from Datastore.tests.shard_store_fixtures import bare_pool

BUILT = build_schema(sqla.MetaData(), _factories)

# the gk-source drop action's tables, in an order SQLite accepts
GK_SOURCE_TABLES = ("GkSourceValue", "GkSource_tags", "GkSource_parents", "GkSource")


def tearDownModule():
    if ray.is_initialized():
        raise AssertionError("Ray was initialised by test_store_schema")


def execute(path: Path, *statements):
    conn = sqlite3.connect(path)
    try:
        for statement in statements:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


def differences_of(path: Path, tables=None) -> SchemaDifferences:
    """``schema_differences`` of one file, on a mode=ro connection."""
    engine = sqla.create_engine(read_only_url(Path(path)), future=True)
    try:
        with engine.connect() as conn:
            return schema_differences(conn, BUILT.tables if tables is None else tables)
    finally:
        engine.dispose()


def primary_tables():
    """The six tables ``ShardedPool._create_engine`` declares for a primary."""
    pool = bare_pool(Path("unused.sqlite"))
    pool._create_engine()
    pool._engine.dispose()
    return {t.name: t for t in pool._metadata.tables.values()}


NONE = SchemaDifferences(
    absent_tables=(), extra_tables=(), absent_columns={}, extra_columns={}
)


class _TempDir(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()

    def directory(self, name: str) -> Path:
        d = self.root / name
        d.mkdir()
        return d


# ------------------------------------------------------------------------------------------------
# 1. the comparison
# ------------------------------------------------------------------------------------------------


class TestSchemaDifferences(_TempDir):
    def test_every_file_of_a_current_store_has_no_difference(self):
        primary = primary_tables()
        self.assertEqual(
            sorted(primary),
            [
                "replicated_tables",
                "replication_in_flight",
                "shard_key_config",
                "shard_keys",
                "sharded_tables",
                "shards",
            ],
        )
        for name, builder in (
            ("build_real_store", build_real_store),
            ("build_full_store", build_full_store),
        ):
            store = builder(self.directory(name))
            for serial, path in sorted(store.shard_files.items()):
                with self.subTest(store=name, shard=serial):
                    differences = differences_of(path)
                    self.assertEqual(differences, NONE)
                    self.assertTrue(differences.empty)
            with self.subTest(store=name, file="primary"):
                self.assertEqual(differences_of(store.primary, primary), NONE)

    def test_each_kind_is_found_on_the_right_shard_by_name(self):
        store = build_real_store(
            self.directory("store"),
            shards=3,
            missing_tables={1: ["OneLoopIntegral_tags", "GkSource_parents"]},
            missing_columns={0: {"TkNumericIntegration": ["stop_Tprime"]}},
            extra_sql={
                2: [
                    "CREATE TABLE zz_retired (x INTEGER)",
                    'ALTER TABLE "GkSource" ADD COLUMN legacy_note VARCHAR(64)',
                    "CREATE TABLE aa_retired (x INTEGER)",
                ]
            },
        )
        found = {s: differences_of(p) for s, p in store.shard_files.items()}
        self.assertEqual(
            found[0],
            NONE._replace(absent_columns={"TkNumericIntegration": ("stop_Tprime",)}),
        )
        # declaration order, not the order the fixture was given them in
        declared = [
            n for n in BUILT.tables if n in ("OneLoopIntegral_tags", "GkSource_parents")
        ]
        self.assertEqual(found[1], NONE._replace(absent_tables=tuple(declared)))
        # file order: the table created first comes first
        self.assertEqual(
            found[2],
            NONE._replace(
                extra_tables=("zz_retired", "aa_retired"),
                extra_columns={"GkSource": ("legacy_note",)},
            ),
        )
        for serial, differences in found.items():
            with self.subTest(shard=serial):
                self.assertFalse(differences.empty)

        self.assertEqual(
            found[2].describe(),
            'it has the table(s) "zz_retired", "aa_retired", which this code does not declare; '
            'its table "GkSource" has the column(s) "legacy_note", which this code does not '
            "declare",
        )
        self.assertEqual(
            found[0].describe(),
            'its table "TkNumericIntegration" lacks the column(s) "stop_Tprime"',
        )

    def test_it_issues_no_write(self):
        store = build_real_store(self.directory("store"))
        path = store.shard_files[0]
        before = file_state(store.directory)
        seen = []

        def before_execute(conn, cursor, statement, parameters, context, executemany):
            seen.append(statement)

        # a read-write engine, so that a write would not be refused by the file
        engine = sqla.create_engine(f"sqlite:///{path}", future=True)
        sqla.event.listen(engine, "before_cursor_execute", before_execute)
        try:
            with engine.connect() as conn:
                self.assertEqual(schema_differences(conn, BUILT.tables), NONE)
                conn.rollback()
        finally:
            engine.dispose()
        self.assertGreater(len(seen), 1)
        for statement in seen:
            self.assertTrue(
                statement.startswith("SELECT name FROM sqlite_master")
                or statement.startswith("PRAGMA table_info("),
                statement,
            )
        self.assertEqual(before, file_state(store.directory))


# ------------------------------------------------------------------------------------------------
# 2-3. the reader and the inventory
# ------------------------------------------------------------------------------------------------


class TestTheReader(_TempDir):
    def test_a_refusal_yields_nothing_and_disposes_every_engine(self):
        store = build_real_store(
            self.directory("store"), shards=3, missing_tables={1: ["OneLoopIntegral"]}
        )
        before = file_state(store.directory)
        import Datastore.store_reader as store_reader

        made = []
        original = store_reader._read_only_engine

        def recording(path):
            engine = original(path)
            made.append((path, engine))
            return engine

        with mock.patch.object(store_reader, "_read_only_engine", recording):
            with self.assertRaises(StoreSchemaMismatch) as cm:
                with open_read_only(store.primary, _factories):
                    self.fail("the store was opened")
        # shard 0 was opened and compared, shard 1 refused, and shard 2 never opened
        self.assertEqual(
            [p for p, _ in made], [store.shard_files[0], store.shard_files[1]]
        )
        for _, engine in made:
            self.assertEqual(engine.pool.checkedin(), 0)
        self.assertIn(f'shard #1 "{store.shard_files[1]}"', str(cm.exception))
        self.assertEqual(before, file_state(store.directory))


class TestTheInventory(_TempDir):
    def test_read_inventory_refuses_a_store_missing_a_table(self):
        store = build_full_store(
            self.directory("store"), missing_tables={1: ["QuadSourceIntegral_tags"]}
        )
        before = file_state(store.directory)
        with self.assertRaises(StoreSchemaMismatch) as cm:
            read_inventory(store.primary, _factories)
        message = str(cm.exception)
        self.assertTrue(
            message.startswith(f'Cannot read sharded datastore "{store.primary}"')
        )
        self.assertIn(f'shard #1 "{store.shard_files[1]}"', message)
        self.assertIn('it lacks the table(s) "QuadSourceIntegral_tags"', message)
        self.assertEqual(before, file_state(store.directory))


# ------------------------------------------------------------------------------------------------
# 4-6. the pool's opens, on stand-in shards
# ------------------------------------------------------------------------------------------------


class _PoolTestCase(_TempDir):
    """A fresh three-shard store, created by a read-write pool and closed, per test."""

    def setUp(self):
        super().setUp()
        self.cluster = sp.StandinCluster()
        active = self.cluster.active()
        active.__enter__()
        self.addCleanup(active.__exit__, None, None, None)
        self._stores = 0

    def tearDown(self):
        self.assertFalse(ray.is_initialized())

    def new_store(self):
        """Create a store in its own directory, close it, and return (primary, shard files)."""
        self._stores += 1
        primary = self.root / f"store-{self._stores}" / "store.sqlite"
        pool = self.cluster.open_pool(primary)
        files = dict(pool._shard_db_files)
        self.cluster.close_pool(pool)
        return primary.resolve(), files

    @contextlib.contextmanager
    def counting(self):
        """Record every stand-in actor constructed, and every read of the in-flight record."""
        constructed = []
        records = []
        original_remote = sp._Options.remote
        original_record = sp.sp_mod.ShardedPool._read_in_flight_record

        def remote(options, *args, **kw):
            constructed.append(options._name)
            return original_remote(options, *args, **kw)

        def record(pool):
            records.append(pool)
            return original_record(pool)

        with mock.patch.object(sp._Options, "remote", remote), mock.patch.object(
            sp.sp_mod.ShardedPool, "_read_in_flight_record", record
        ):
            yield constructed, records

    def assert_refused(self, primary, read_only=False):
        """Open ``primary``, expect StoreSchemaMismatch before any actor exists and before the
        in-flight record is read, with every file unchanged; return the exception."""
        before = sp.store_checksums(primary)
        with self.counting() as (constructed, records):
            with self.assertRaises(StoreSchemaMismatch) as cm:
                self.cluster.open_pool_output(primary, read_only=read_only)
        self.assertEqual([], constructed, "an actor was created")
        self.assertEqual([], records, "the replication_in_flight record was read")
        self.assertEqual(before, sp.store_checksums(primary))
        return cm.exception


class TestTheReadWriteOpenRefuses(_PoolTestCase):
    def test_an_absent_column_an_extra_column_and_an_extra_table(self):
        cases = (
            (
                "an absent column",
                'ALTER TABLE "TkNumericIntegration" DROP COLUMN "stop_Tprime"',
                NONE._replace(
                    absent_columns={"TkNumericIntegration": ("stop_Tprime",)}
                ),
                'its table "TkNumericIntegration" lacks the column(s) "stop_Tprime"',
            ),
            (
                "an extra column",
                'ALTER TABLE "tolerance" ADD COLUMN legacy_note VARCHAR(64)',
                NONE._replace(extra_columns={"tolerance": ("legacy_note",)}),
                'its table "tolerance" has the column(s) "legacy_note", which this code does '
                "not declare",
            ),
            (
                "an extra table",
                "CREATE TABLE retired_table (x INTEGER)",
                NONE._replace(extra_tables=("retired_table",)),
                'it has the table(s) "retired_table", which this code does not declare',
            ),
        )
        for label, statement, expected, fragment in cases:
            with self.subTest(case=label):
                primary, files = self.new_store()
                execute(files[2], statement)
                e = self.assert_refused(primary)
                self.assertIsInstance(e, RuntimeError)
                self.assertEqual(e.verb, "open")
                self.assertEqual(e.differences, expected)
                message = str(e)
                self.assertTrue(
                    message.startswith(f'Cannot open sharded datastore "{primary}": ')
                )
                self.assertIn(f'shard #2 "{files[2]}"', message)
                self.assertIn(fragment, message)
                self.assertTrue(message.endswith("so nothing was read"))

    def test_an_absent_table_beside_another_difference_is_still_refused(self):
        primary, files = self.new_store()
        execute(
            files[1],
            'DROP TABLE "GkSourcePolicy"',
            "CREATE TABLE retired_table (x INTEGER)",
        )
        e = self.assert_refused(primary)
        self.assertEqual(
            e.differences,
            NONE._replace(
                absent_tables=("GkSourcePolicy",), extra_tables=("retired_table",)
            ),
        )

    def test_a_prune_record_is_not_completed(self):
        """The check comes before the record is read, so it covers an interrupted prune too: a
        prune record of the one replicated class that prunes at open is left on the primary, and
        not completed or cleared."""
        primary, files = self.new_store()
        execute(
            primary,
            "INSERT INTO replication_in_flight (operation, class_name, controller_shard, "
            "store_id, started) VALUES ('prune', 'BackgroundModel', NULL, NULL, "
            "'2026-10-05 00:00:00.000000')",
        )
        execute(files[0], "CREATE TABLE retired_table (x INTEGER)")
        self.assert_refused(primary)
        self.assertEqual(
            sp._read(
                primary, "SELECT operation, class_name FROM replication_in_flight"
            ),
            [("prune", "BackgroundModel")],
        )


class TestTheReadWriteOpenRecoversAnAbsentTable(_PoolTestCase):
    """U1 (README §6.2 of prompts/datastore-generic): a read-write open reads a table a shard
    lacks as empty, and the actors create it."""

    def assert_opens_and_completes(self, primary, files):
        pool, _ = self.cluster.open_pool_output(primary)
        self.cluster.close_pool(pool)
        for serial, path in sorted(files.items()):
            with self.subTest(shard=serial):
                self.assertEqual(differences_of(path), NONE)
                self.assertEqual(
                    sp._read(path, "SELECT label FROM version"), [("standin",)]
                )

    def test_a_fresh_store_has_no_difference(self):
        primary, files = self.new_store()
        for serial, path in sorted(files.items()):
            with self.subTest(shard=serial):
                self.assertEqual(differences_of(path), NONE)
        self.assertEqual(differences_of(primary, primary_tables()), NONE)

    def test_an_interrupted_drop_of_a_sharded_class(self):
        primary, files = self.new_store()
        execute(files[0], *[f'DROP TABLE "{t}"' for t in GK_SOURCE_TABLES])
        self.assertEqual(
            differences_of(files[0]).absent_tables,
            tuple(n for n in BUILT.tables if n in GK_SOURCE_TABLES),
        )
        self.assert_opens_and_completes(primary, files)

    def test_an_interrupted_drop_of_a_replicated_class(self):
        primary, files = self.new_store()
        self.assertEqual(
            sp._read(files[0], 'SELECT count(*) FROM "GkSourcePolicy"'), [(0,)]
        )
        execute(files[1], 'DROP TABLE "GkSourcePolicy"')
        self.assert_opens_and_completes(primary, files)

    def test_an_interrupted_first_open(self):
        primary, files = self.new_store()
        for path in files.values():
            execute(path, "DELETE FROM version")
        names = list(BUILT.tables)
        dropped = names[len(names) // 2 :][::-1]
        execute(files[2], *[f'DROP TABLE "{t}"' for t in dropped])
        self.assertEqual(len(differences_of(files[2]).absent_tables), len(dropped))
        self.assert_opens_and_completes(primary, files)


class TestThePrimaryIsRefused(_PoolTestCase):
    def test_an_extra_table_and_a_missing_column(self):
        cases = (
            (
                "an extra table",
                "CREATE TABLE retired_table (x INTEGER)",
                NONE._replace(extra_tables=("retired_table",)),
            ),
            (
                "a missing column",
                'ALTER TABLE "sharded_tables" DROP COLUMN "key_attr"',
                NONE._replace(absent_columns={"sharded_tables": ("key_attr",)}),
            ),
        )
        for label, statement, expected in cases:
            for read_only in (False, True):
                with self.subTest(case=label, read_only=read_only):
                    primary, _ = self.new_store()
                    execute(primary, statement)
                    e = self.assert_refused(primary, read_only=read_only)
                    self.assertEqual(e.verb, "open")
                    self.assertEqual(e.differences, expected)
                    self.assertIn(
                        f'Cannot open sharded datastore "{primary}": the primary "{primary}" '
                        "differs",
                        str(e),
                    )


if __name__ == "__main__":
    unittest.main()
