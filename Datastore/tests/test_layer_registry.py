"""
The datastore layer is given its registries (prompts/datastore-generic, prompt 06).

The generic layer -- ``ShardedPool``, the ``Datastore`` actor, ``ClientPool``, ``store_reader`` and
``store_inventory`` -- imports none of the project's registries. The client keeps them in
``config/datastore.py`` and gives them: the registry of storable classes (``factories``), the
serial lease batch sizes, and the tables to drop. The replicated set is read from the store.

1. ``TestTheClientRegistries`` -- ``config.datastore``: ``tables_to_drop`` gives each table once
   and names an unknown action; every table of every group is declared by ``factories``; the
   groups are in the order the command line offers them.
2. ``TestDropOrder`` -- ``schema.drop_order``: a referencing table comes before what it references;
   a cycle raises.
3. ``TestTheDropOnAFullStore`` -- under ``PRAGMA foreign_keys = ON``, the real
   ``Datastore._drop_tables`` drops every group from both shards of ``build_full_store``, which
   holds rows in every dropped table, one printed line per table, and leaves ``foreign_key_check``
   empty. Unsorted, the same drop raises ``IntegrityError``.
4. ``TestAnUndeclaredDropTable`` -- read-write, refused by the pool before anything is opened, on a
   missing path and on an existing store; refused by the actor's own ``_drop_tables`` before
   anything is dropped; read-only, ``ReadOnlyWrite`` comes first.
5. ``TestSerialBatchSizes`` -- ``SerialPoolManager`` leases the sizes it is given, and 500 for a
   table it is not given; the pool gives every actor its sizes and its registry.
6. ``TestTheRecordedReplicatedSet`` -- the reader yields the primary's record in serial order; a
   primary whose record cannot be read, or names an undeclared class, is refused by the reader and
   by ``read_inventory``; the inventory compares exactly the classes the store records.
7. ``TestThePoolUsesItsRegistry`` -- a registry without the ``OneLoopIntegral`` classes builds
   shards without their tables.
8. ``TestTheLayerImportsNoClient`` -- importing the layer in a fresh interpreter loads no client
   registry, no factory but the base, and no compute target; none of its five files imports one.

No Ray, nothing under ``var/``.
"""

import ast
import contextlib
import importlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import sqlalchemy as sqla
from sqlalchemy.exc import IntegrityError

from Datastore.SQL.ClientPool import SerialPoolManager
from Datastore.SQL.schema import build_schema, drop_order
from Datastore.replication import ReadOnlyWrite
from Datastore.store_inventory import read_inventory
from Datastore.store_reader import open_read_only
from Datastore.tests import standin_pool as sp
from Datastore.tests.real_store_fixtures import build_full_store, file_state
from config.datastore import drop_groups, factories, tables_to_drop
from config.sharding import replicated_tables as configured_replicated

REPO_ROOT = Path(__file__).resolve().parents[2]

# the module itself: `import Datastore.SQL.Datastore as ...` binds the package's re-exported actor
# class of the same name, not the module whose drop_order these tests patch
datastore_module = importlib.import_module("Datastore.SQL.Datastore")

# the measured messages of the refusals, printed when the module is run as a script
MESSAGES = {}

# main.py's --drop choices, in the order its --help and argparse's invalid-choice message print them
COMMAND_LINE_ORDER = [
    "tk-numeric",
    "tk-wkb",
    "quad-source",
    "gk-numeric",
    "gk-wkb",
    "gk-source",
    "gk-source-policy-records",
    "gk-source-policy",
    "quad-source-integral",
    "1loop-integral",
]


class _TempDir(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()

    def full_store(self, name: str):
        """``build_full_store`` in its own directory under the test's temporary directory."""
        directory = self.root / name
        directory.mkdir()
        return build_full_store(directory)


def _actor_on(path: Path, enforce: bool):
    """A stand-in for the actor's state that ``_drop_tables`` reads, on the SQLite file ``path``,
    with ``PRAGMA foreign_keys = ON`` on every connection if ``enforce``. Returns (fake, engine).
    """
    engine = sqla.create_engine(f"sqlite:///{path}", future=True)
    if enforce:

        @sqla.event.listens_for(engine, "connect")
        def _enforce(dbapi_connection, connection_record):
            dbapi_connection.execute("PRAGMA foreign_keys = ON")

    metadata = sqla.MetaData()
    tables = build_schema(metadata, factories).tables
    fake = types.SimpleNamespace(
        _inspector=sqla.inspect(engine),
        _tables=tables,
        _engine=engine,
        _metadata=metadata,
        _my_name="registry-test",
    )
    return fake, engine


def _drop_tables(fake, tables) -> str:
    """Run the real actor's ``_drop_tables`` on ``fake``; return what it printed."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        sp.DatastoreClass._drop_tables(fake, tables)
    return out.getvalue()


def _table_names(path: Path):
    with contextlib.closing(sqlite3.connect(path)) as conn:
        return {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }


# ------------------------------------------------------------------------------------------------
# 1. config.datastore
# ------------------------------------------------------------------------------------------------


class TestTheClientRegistries(unittest.TestCase):
    def test_tables_to_drop_gives_each_table_once_and_names_an_unknown_action(self):
        self.assertEqual(
            tables_to_drop(["tk-wkb", "tk-wkb"]),
            ["TkWKBIntegration", "TkWKB_tags", "TkWKBValue"],
        )
        self.assertEqual(
            tables_to_drop(["gk-source-policy", "tk-numeric"]),
            [
                "GkSourcePolicyData",
                "TkNumericIntegration",
                "TkNumeric_tags",
                "TkNumericValue",
            ],
        )
        every = tables_to_drop(list(drop_groups))
        self.assertEqual(len(every), len(set(every)))
        self.assertEqual(
            set(every), {t for group in drop_groups.values() for t in group}
        )
        self.assertEqual(tables_to_drop([]), [])

        with self.assertRaises(ValueError) as cm:
            tables_to_drop(["tk-wkb", "no-such-group"])
        self.assertIn('"no-such-group"', str(cm.exception))

    def test_every_table_of_every_group_is_declared_by_the_registry(self):
        declared = build_schema(sqla.MetaData(), factories).tables
        for action, tables in drop_groups.items():
            for table in tables:
                with self.subTest(action=action, table=table):
                    self.assertIn(table, declared)

    def test_the_groups_are_in_the_command_line_order(self):
        # main.py offers list(drop_groups) as --drop's choices: this order is what --help prints
        self.assertEqual(list(drop_groups), COMMAND_LINE_ORDER)


# ------------------------------------------------------------------------------------------------
# 2. schema.drop_order
# ------------------------------------------------------------------------------------------------


class TestDropOrder(unittest.TestCase):
    def test_a_referencing_table_comes_before_what_it_references(self):
        m = sqla.MetaData()
        outside = sqla.Table(
            "outside", m, sqla.Column("id", sqla.Integer, primary_key=True)
        )
        a = sqla.Table(
            "a",
            m,
            sqla.Column("id", sqla.Integer, primary_key=True),
            sqla.Column("outside_id", sqla.ForeignKey("outside.id")),
            # a self-reference does not constrain the order
            sqla.Column("a_id", sqla.ForeignKey("a.id")),
        )
        b = sqla.Table(
            "b",
            m,
            sqla.Column("id", sqla.Integer, primary_key=True),
            sqla.Column("a_id", sqla.ForeignKey("a.id")),
        )
        c = sqla.Table(
            "c",
            m,
            sqla.Column("id", sqla.Integer, primary_key=True),
            sqla.Column("b_id", sqla.ForeignKey("b.id")),
            sqla.Column("a_id", sqla.ForeignKey("a.id")),
        )
        free = sqla.Table("free", m, sqla.Column("id", sqla.Integer, primary_key=True))

        self.assertEqual([t.name for t in drop_order([a, b, c])], ["c", "b", "a"])
        self.assertEqual(
            [t.name for t in drop_order([free, a, c, b, a])], ["free", "c", "b", "a"]
        )
        self.assertEqual(drop_order([]), [])
        # the table referenced but not given is not returned
        self.assertNotIn(outside, drop_order([a, b]))

    def test_a_cycle_raises(self):
        m = sqla.MetaData()
        x = sqla.Table(
            "x",
            m,
            sqla.Column("id", sqla.Integer, primary_key=True),
            sqla.Column("y_id", sqla.ForeignKey("y.id")),
        )
        y = sqla.Table(
            "y",
            m,
            sqla.Column("id", sqla.Integer, primary_key=True),
            sqla.Column("x_id", sqla.ForeignKey("x.id")),
        )
        z = sqla.Table(
            "z",
            m,
            sqla.Column("id", sqla.Integer, primary_key=True),
            sqla.Column("x_id", sqla.ForeignKey("x.id")),
        )
        with self.assertRaises(ValueError) as cm:
            drop_order([x, y, z])
        message = str(cm.exception)
        self.assertIn("cycle", message)
        self.assertIn("'x'", message)
        self.assertIn("'y'", message)


# ------------------------------------------------------------------------------------------------
# 3. the drop on a full store
# ------------------------------------------------------------------------------------------------


class TestTheDropOnAFullStore(_TempDir):
    def setUp(self):
        super().setUp()
        self.store = self.full_store("full")
        self.dropped = tables_to_drop(list(drop_groups))

    def test_every_group_drops_from_both_shards_and_leaves_no_violation(self):
        # the full store holds rows in every dropped table, on one shard or the other
        held = {}
        for sid, path in sorted(self.store.shard_files.items()):
            with contextlib.closing(sqlite3.connect(path)) as conn:
                self.assertEqual(
                    conn.execute("PRAGMA foreign_key_check").fetchall(), []
                )
                for table in self.dropped:
                    n = conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
                    held[table] = held.get(table, 0) + n
        self.assertEqual([t for t, n in held.items() if n == 0], [])

        tables = build_schema(sqla.MetaData(), factories).tables
        expected = [
            f'Datastore: dropping table "{t.name}"'
            for t in drop_order([tables[n] for n in self.dropped])
        ]
        for sid, path in sorted(self.store.shard_files.items()):
            with self.subTest(shard=sid):
                fake, engine = _actor_on(path, enforce=True)
                try:
                    printed = _drop_tables(fake, self.dropped)
                    with engine.connect() as conn:
                        self.assertEqual(
                            conn.execute(sqla.text("PRAGMA foreign_keys")).scalar(), 1
                        )
                        violations = conn.execute(
                            sqla.text("PRAGMA foreign_key_check")
                        ).all()
                finally:
                    engine.dispose()
                self.assertEqual(printed.splitlines(), expected)
                self.assertEqual(violations, [])
                self.assertEqual(_table_names(path) & set(self.dropped), set())

    def test_unsorted_the_same_drop_is_refused_by_enforcement(self):
        path = self.store.shard_files[0]
        fake, engine = _actor_on(path, enforce=True)
        try:
            with mock.patch.object(datastore_module, "drop_order", lambda t: list(t)):
                with self.assertRaises(IntegrityError):
                    _drop_tables(fake, self.dropped)
        finally:
            engine.dispose()


# ------------------------------------------------------------------------------------------------
# 4. an undeclared drop table
# ------------------------------------------------------------------------------------------------


class TestAnUndeclaredDropTable(_TempDir):
    DROP = ["GkSource", "NotATable"]

    def setUp(self):
        super().setUp()
        self.cluster = sp.StandinCluster()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(self.cluster.active())
        self.existing = self.root / "store" / "store.sqlite"
        self.cluster.close_pool(self.cluster.open_pool(self.existing))
        self.existing = self.existing.resolve()

    @contextlib.contextmanager
    def counting(self):
        """Record every stand-in actor constructed, and every engine the pool makes."""
        constructed = []
        engines = []
        original_remote = sp._Options.remote
        original_engine = sp.sp_mod.ShardedPool._create_engine

        def remote(options, *args, **kw):
            constructed.append(options._name)
            return original_remote(options, *args, **kw)

        def create_engine(pool):
            engines.append(pool)
            return original_engine(pool)

        with mock.patch.object(sp._Options, "remote", remote), mock.patch.object(
            sp.sp_mod.ShardedPool, "_create_engine", create_engine
        ):
            yield constructed, engines

    def test_read_write_the_pool_refuses_it_before_anything_is_opened(self):
        missing = self.root / "nowhere" / "store.sqlite"
        before = sp.store_checksums(self.existing)
        for where, primary in (("missing", missing), ("existing", self.existing)):
            with self.subTest(where=where):
                self.cluster.calls.clear()
                with self.counting() as (constructed, engines):
                    with self.assertRaises(RuntimeError) as cm:
                        self.cluster.open_pool(primary, drop_tables=self.DROP)
                self.assertNotIsInstance(cm.exception, ReadOnlyWrite)
                message = str(cm.exception)
                self.assertTrue(
                    message.startswith(
                        f'Cannot open sharded datastore "{primary.resolve()}"'
                    ),
                    message,
                )
                self.assertEqual(message.count("Cannot open sharded datastore"), 1)
                self.assertIn("'NotATable'", message)
                self.assertNotIn("'GkSource'", message)
                self.assertTrue(message.endswith("Nothing was opened"), message)
                self.assertEqual(constructed, [], "an actor was created")
                self.assertEqual(engines, [], "an engine was made")
                self.assertEqual(self.cluster.calls, [])
                self.assertFalse(missing.parent.exists())
                self.assertEqual(sp.store_checksums(self.existing), before)
        MESSAGES["pool"] = message

    def test_the_actor_refuses_it_before_dropping_anything(self):
        store = self.full_store("full")
        path = store.shard_files[0]
        before = file_state(store.directory)
        fake, engine = _actor_on(path, enforce=False)
        try:
            with self.assertRaises(RuntimeError) as cm:
                _drop_tables(fake, self.DROP)
        finally:
            engine.dispose()
        message = str(cm.exception)
        self.assertIn('"registry-test"', message)
        self.assertIn("'NotATable'", message)
        self.assertNotIn("'GkSource'", message)
        self.assertIn("Nothing was dropped", message)
        self.assertIn("GkSource", _table_names(path))
        self.assertEqual(file_state(store.directory), before)
        MESSAGES["actor"] = message

    def test_read_only_the_refusal_is_read_only_write(self):
        missing = self.root / "nowhere" / "store.sqlite"
        before = sp.store_checksums(self.existing)
        for where, primary in (("missing", missing), ("existing", self.existing)):
            with self.subTest(where=where):
                self.cluster.calls.clear()
                with self.counting() as (constructed, engines):
                    with self.assertRaises(ReadOnlyWrite) as cm:
                        self.cluster.open_pool(
                            primary, read_only=True, drop_tables=self.DROP
                        )
                self.assertIn("drop_tables=", str(cm.exception))
                self.assertNotIn("does not declare", str(cm.exception))
                self.assertEqual(constructed, [])
                self.assertEqual(engines, [])
                self.assertEqual(self.cluster.calls, [])
                self.assertFalse(missing.parent.exists())
                self.assertEqual(sp.store_checksums(self.existing), before)


# ------------------------------------------------------------------------------------------------
# 5. serial batch sizes
# ------------------------------------------------------------------------------------------------


class TestSerialBatchSizes(_TempDir):
    @staticmethod
    def batch_size(manager, table):
        seen = {}

        class Recording:
            def __init__(self, table, profiler, broker, default_batch_size):
                seen[table] = default_batch_size

            def lease_serial(self):
                return 1

        with mock.patch("Datastore.SQL.ClientPool.ClientPool", Recording):
            manager.lease_serial(table)
        return seen[table]

    def test_the_manager_uses_the_sizes_it_is_given_and_500_otherwise(self):
        given = SerialPoolManager(
            profiler=None, broker=object(), batch_sizes={"TkNumericValue": 7}
        )
        self.assertEqual(self.batch_size(given, "TkNumericValue"), 7)
        self.assertEqual(self.batch_size(given, "redshift"), 500)
        bare = SerialPoolManager(profiler=None, broker=object())
        self.assertEqual(self.batch_size(bare, "TkNumericValue"), 500)

    def test_the_pool_gives_every_actor_its_sizes_and_its_registry(self):
        sizes = {"TkNumericValue": 7, "redshift": 11}
        cluster = sp.StandinCluster()
        with cluster.active():
            pool = cluster.open_pool(
                self.root / "store.sqlite", serial_batch_sizes=sizes
            )
            actors = [handle.obj for handle in pool._shards.values()]
            cluster.close_pool(pool)
        self.assertEqual(len(actors), 3)
        for actor in actors:
            self.assertEqual(actor._serial_manager._batch_sizes, sizes)
            self.assertEqual(list(actor._factories), list(factories))


# ------------------------------------------------------------------------------------------------
# 6. the recorded replicated set
# ------------------------------------------------------------------------------------------------


class TestTheRecordedReplicatedSet(_TempDir):
    def primary_sql(self, store, *statements):
        with contextlib.closing(sqlite3.connect(store.primary)) as conn:
            for statement in statements:
                conn.execute(statement)
            conn.commit()

    def test_the_reader_yields_the_record_in_serial_order(self):
        store = self.full_store("as-written")
        with open_read_only(store.primary, factories) as opened:
            self.assertEqual(opened.replicated_tables, tuple(configured_replicated))

        reversed_store = self.full_store("reversed")
        self.primary_sql(
            reversed_store, "UPDATE replicated_tables SET serial = 100 - serial"
        )
        with open_read_only(reversed_store.primary, factories) as opened:
            self.assertEqual(
                opened.replicated_tables, tuple(reversed(configured_replicated))
            )

    def test_a_store_the_pool_wrote_records_the_configured_set(self):
        cluster = sp.StandinCluster()
        with cluster.active():
            primary = self.root / "pool" / "store.sqlite"
            cluster.close_pool(cluster.open_pool(primary))
        with open_read_only(primary, factories) as opened:
            self.assertEqual(opened.replicated_tables, tuple(configured_replicated))

    def assert_refused(self, store, *fragments):
        before = file_state(store.directory)
        messages = []
        for what, read in (
            ("reader", lambda: open_read_only(store.primary, factories).__enter__()),
            ("inventory", lambda: read_inventory(store.primary, factories)),
        ):
            with self.subTest(what=what):
                with self.assertRaises(RuntimeError) as cm:
                    read()
                message = str(cm.exception)
                self.assertTrue(
                    message.startswith(
                        f'Cannot read sharded datastore "{store.primary}": '
                    ),
                    message,
                )
                self.assertEqual(message.count("Cannot read sharded datastore"), 1)
                for fragment in fragments:
                    self.assertIn(fragment, message)
                messages.append(message)
        self.assertEqual(file_state(store.directory), before)
        # the reader's message, kept for the record (none if a sub-test failed)
        return messages[0] if len(messages) > 0 else None

    def test_a_primary_without_the_record_is_refused(self):
        store = self.full_store("full")
        self.primary_sql(store, "DROP TABLE replicated_tables")
        MESSAGES["no record"] = self.assert_refused(
            store, "replicated_tables", "no such table"
        )

    def test_a_record_naming_an_undeclared_class_is_refused(self):
        store = self.full_store("full")
        self.primary_sql(
            store,
            "INSERT INTO replicated_tables (serial, \"table\") VALUES (99, 'NotAClass')",
        )
        MESSAGES["undeclared class"] = self.assert_refused(
            store, "replicated_tables", "'NotAClass'"
        )

    def test_the_inventory_compares_exactly_the_recorded_classes(self):
        recorded = read_inventory(self.full_store("as-written").primary, factories)
        self.assertTrue(recorded["tolerance"].replicated)
        self.assertEqual(recorded["tolerance"].problems, ())

        store = self.full_store("without-tolerance")
        self.primary_sql(
            store, "DELETE FROM replicated_tables WHERE \"table\" = 'tolerance'"
        )
        inventory = read_inventory(store.primary, factories)
        # tolerance is combined as a sharded class: the union of both shards' copies, each a
        # duplicate of the other
        tolerance = inventory["tolerance"]
        self.assertFalse(tolerance.replicated)
        self.assertEqual(tolerance.count, 2 * recorded["tolerance"].count)
        self.assertEqual([p.split(":")[0] for p in tolerance.problems], ["duplicate"])
        # and the classes the store still records are compared as before
        self.assertTrue(inventory["redshift"].replicated)
        self.assertEqual(inventory["redshift"].records, recorded["redshift"].records)


# ------------------------------------------------------------------------------------------------
# 7. the pool uses its registry
# ------------------------------------------------------------------------------------------------


class TestThePoolUsesItsRegistry(_TempDir):
    def test_a_registry_without_a_class_builds_shards_without_its_tables(self):
        small = {
            name: factory
            for name, factory in factories.items()
            if name not in ("OneLoopIntegral", "OneLoopIntegral_tags")
        }
        cluster = sp.StandinCluster()
        with cluster.active():
            pool = cluster.open_pool(self.root / "store.sqlite", factories=small)
            given = pool.factories
            given["extra"] = None
            self.assertEqual(pool.factories, small, "the property is not a copy")
            files = dict(pool._shard_db_files)
            cluster.close_pool(pool)
        self.assertEqual(len(files), 3)
        for sid, path in sorted(files.items()):
            with self.subTest(shard=sid):
                names = _table_names(path)
                self.assertNotIn("OneLoopIntegral", names)
                self.assertNotIn("OneLoopIntegral_tags", names)
                self.assertIn("QuadSourceIntegral", names)
                self.assertIn("QuadSourceIntegral_tags", names)


# ------------------------------------------------------------------------------------------------
# 8. the layer imports no client
# ------------------------------------------------------------------------------------------------

LAYER_MODULES = [
    "Datastore.SQL.ShardedPool",
    "Datastore.SQL.Datastore",
    "Datastore.SQL.ClientPool",
    "Datastore.store_reader",
    "Datastore.store_inventory",
]
LAYER_FILES = [
    "Datastore/SQL/ShardedPool.py",
    "Datastore/SQL/Datastore.py",
    "Datastore/SQL/ClientPool.py",
    "Datastore/store_reader.py",
    "Datastore/store_inventory.py",
]
# utilities is the one project package still allowed: the actor imports utilities.WallclockTimer,
# a generic helper. MetadataConcepts is no longer loaded: since prompt 07 the pool builds the
# version object through the version table's factory, and imports no store_tag (README §6.2,
# U4). Prompt 09's guard generalises this test; this one pins only what prompts 06 and 07 removed.
FORBIDDEN_MODULES = {"config.datastore", "config.sharding"}
FORBIDDEN_PACKAGES = (
    "ComputeTargets",
    "CosmologyConcepts",
    "CosmologyModels",
    "MetadataConcepts",
)
ALLOWED_FACTORY_MODULES = {
    "Datastore.SQL.ObjectFactories",
    "Datastore.SQL.ObjectFactories.base",
}


def layer_imports():
    """The modules loaded by importing the layer in a fresh interpreter, sorted."""
    code = (
        "import importlib, json, sys\n"
        f"for m in {LAYER_MODULES!r}:\n"
        "    importlib.import_module(m)\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestTheLayerImportsNoClient(unittest.TestCase):
    def test_a_fresh_interpreter_loads_no_client_registry(self):
        loaded = layer_imports()
        for name in LAYER_MODULES:
            self.assertIn(name, loaded)
        self.assertEqual(sorted(FORBIDDEN_MODULES & set(loaded)), [])
        self.assertEqual(
            [
                m
                for m in loaded
                if m.startswith("Datastore.SQL.ObjectFactories")
                and m not in ALLOWED_FACTORY_MODULES
            ],
            [],
        )
        self.assertEqual(
            [m for m in loaded if m.split(".")[0] in FORBIDDEN_PACKAGES], []
        )

    def test_no_layer_file_imports_a_registry(self):
        for path in LAYER_FILES:
            with self.subTest(path=path):
                tree = ast.parse((REPO_ROOT / path).read_text())
                found = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        found += [
                            a.name for a in node.names if a.name in FORBIDDEN_MODULES
                        ]
                    elif isinstance(node, ast.ImportFrom):
                        if node.module in FORBIDDEN_MODULES:
                            found.append(node.module)
                        if node.module == "config":
                            found += [
                                f"config.{a.name}"
                                for a in node.names
                                if f"config.{a.name}" in FORBIDDEN_MODULES
                            ]
                        found += [
                            f"{node.module}.{a.name}"
                            for a in node.names
                            if a.name == "_factories"
                        ]
                self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main(exit=False)
    for key, message in MESSAGES.items():
        print(f"{key}: {message}")
