"""
The replication facts the factories declare, and the layer's contract (prompts/datastore-generic,
prompt 07; README §6.2, U4 and the user's choice for 07).

1. **The records.** ``build_schema`` carries ``owner_column``, ``monotone_flags`` and
   ``validated_column`` into the record of every class with a table, with their defaults where a
   factory declares nothing; exactly four declarations are not default; a record with no table is
   unchanged.
2. **The refusals of ``build_schema``**: a declaration that does not fit its table raises
   ``ValueError`` naming the class and the key, one test per misfit.
3. **The unit** (``ShardedPool._unit_tables``): ``BackgroundModel``'s three tables, in order, and
   every other compared table of the registry alone; on a hand-built registry a table that declares
   an owner joins its unit, and a table with a foreign key to the class and no declaration does
   not.
4. **The flags**: the declared flags reach the specs, and the repair licence's flags, its
   recompute and its refusal text come from the declarations.
5. **``revalidate`` against ``validate``**, on a copy of a ``build_full_store`` shard; the base
   hook raises.
6. **``owned_serials``**: the base owns nothing; ``BackgroundModel``'s gives its value serials, and
   the pool checks each replica's answer through it.
7. **The version object**, built by the registry's version factory on a read-write and a read-only
   open, with the attributes it had.
8. **The prune refusal's text** names the unit in the specs' order.
9. **No project name in ``ShardedPool.py``**, by ``ast`` and by its comments and docstrings.

No Ray, nothing under ``var/``.
"""

import ast
import contextlib
import hashlib
import io
import re
import shutil
import sys
import tempfile
import tokenize
import unittest
from datetime import datetime
from pathlib import Path

import sqlalchemy as sqla

from config.datastore import factories
from config.sharding import replicated_tables, sharded_tables
from Datastore.contract import (
    TAG_LABEL,
    TAG_SERIAL,
    TAG_TABLE,
    VERSION_LABEL,
    VERSION_TABLE,
)
from Datastore.object import DatastoreObject
from Datastore.replication import ReplicationMismatch
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from Datastore.SQL.schema import build_schema
from Datastore.tests import standin_pool as sp
from Datastore.tests.test_prune_at_open import _PruneTestCase, execute, quiet

REPO_ROOT = Path(__file__).resolve().parents[2]
POOL_FILE = REPO_ROOT / "Datastore" / "SQL" / "ShardedPool.py"
ShardedPool = sp.sp_mod.ShardedPool

# the declarations of the real registry, and no others: (owner_column, monotone_flags,
# validated_column)
DEFAULT = (None, (), None)
DECLARED = {
    "redshift": (None, ("source", "response"), None),
    "wavenumber": (None, ("source", "response"), None),
    "BackgroundModel": (None, (), "validated"),
    "BackgroundModelValue": ("model_serial", (), None),
}


def declared(record):
    return (
        record["owner_column"],
        record["monotone_flags"],
        record["validated_column"],
    )


def bare_pool(replicated, sharded=None, registry=None):
    """A ``ShardedPool`` with only what the check at open's planning reads: no file, no actor."""
    pool = object.__new__(ShardedPool)
    pool._replicated_tables = list(replicated)
    pool._sharded_tables = dict(sharded or {})
    pool._factories = dict(registry if registry is not None else factories)
    pool._ShardKeyType_name = "NoShardKeyClass"
    pool._shard_keys = {}
    pool._timeout = None
    return pool


def registry_specs():
    """The real registry's schema and the specs of the tables the check at open compares."""
    built = build_schema(sqla.MetaData(), factories)
    pool = bare_pool(replicated_tables, sharded_tables)
    return built, pool._replicated_table_specs(built)


def factory(columns=lambda: [], **keys):
    """A stand-in factory with no version or timestamp column, the columns ``columns()`` makes
    (a column belongs to one table, so each registration makes its own), and ``keys`` in its
    registration."""

    class _Factory:
        @staticmethod
        def register():
            return {"version": False, "timestamp": False, "columns": columns(), **keys}

    return _Factory


# ------------------------------------------------------------------------------------------------
# 1. the records
# ------------------------------------------------------------------------------------------------


class TestTheRecords(unittest.TestCase):
    def test_every_record_of_a_class_with_a_table_carries_the_three_keys(self):
        built = build_schema(sqla.MetaData(), factories)
        self.assertEqual(len(built.records), len(built.tables))
        for name, record in built.records.items():
            with self.subTest(cls=name):
                self.assertIn("owner_column", record)
                self.assertIn("monotone_flags", record)
                self.assertIn("validated_column", record)
                self.assertIsInstance(record["monotone_flags"], tuple)

    def test_exactly_the_four_declarations_are_not_default(self):
        built = build_schema(sqla.MetaData(), factories)
        found = {
            name: declared(record)
            for name, record in built.records.items()
            if declared(record) != DEFAULT
        }
        self.assertEqual(DECLARED, found)

    def test_a_record_with_no_table_is_unchanged(self):
        class _NoTable:
            @staticmethod
            def register():
                return None

        built = build_schema(sqla.MetaData(), {"NoTable": _NoTable})
        self.assertEqual(
            {"name": "NoTable", "validate_on_startup": False, "table": None},
            built.records["NoTable"],
        )


# ------------------------------------------------------------------------------------------------
# 2. the refusals of build_schema
# ------------------------------------------------------------------------------------------------


class TestBuildSchemaRefusesAMisfit(unittest.TestCase):
    """Each misfit on one class, ``Member``, beside two tables its columns reference."""

    @staticmethod
    def member_columns():
        return [
            sqla.Column("plain", sqla.Integer),
            sqla.Column("owner_serial", sqla.Integer, sqla.ForeignKey("Owner.serial")),
            sqla.Column(
                "two_keys",
                sqla.Integer,
                sqla.ForeignKey("Owner.serial"),
                sqla.ForeignKey("Other.serial"),
            ),
            sqla.Column("flag", sqla.Boolean),
        ]

    def assert_refused(self, key, value, *fragments):
        registry = {
            "Owner": factory(),
            "Other": factory(),
            "Member": factory(self.member_columns, **{key: value}),
        }
        with self.assertRaises(ValueError) as raised:
            build_schema(sqla.MetaData(), registry)
        message = str(raised.exception)
        self.assertIn('"Member"', message)
        self.assertIn(key, message)
        for fragment in fragments:
            self.assertIn(fragment, message)
        return message

    def test_the_fitting_declarations_are_accepted(self):
        registry = {
            "Owner": factory(),
            "Other": factory(),
            "Member": factory(
                self.member_columns,
                owner_column="owner_serial",
                monotone_flags=["flag"],
                validated_column="flag",
            ),
        }
        record = build_schema(sqla.MetaData(), registry).records["Member"]
        self.assertEqual(("owner_serial", ("flag",), "flag"), declared(record))

    def test_an_owner_column_that_is_not_a_column(self):
        self.assert_refused("owner_column", "nonesuch", "'nonesuch'", "not a column")

    def test_an_owner_column_with_no_foreign_key(self):
        self.assert_refused("owner_column", "plain", "'plain'", "0 foreign keys")

    def test_an_owner_column_with_two_foreign_keys(self):
        self.assert_refused("owner_column", "two_keys", "'two_keys'", "2 foreign keys")

    def test_monotone_flags_given_as_a_string(self):
        self.assert_refused("monotone_flags", "flag", "'flag'", "a string")

    def test_monotone_flags_naming_a_column_that_is_absent(self):
        self.assert_refused(
            "monotone_flags", ("flag", "nonesuch"), "['nonesuch']", "not columns"
        )

    def test_monotone_flags_naming_a_column_that_is_not_boolean(self):
        self.assert_refused(
            "monotone_flags", ("flag", "plain"), "['plain']", "not Boolean"
        )

    def test_a_validated_column_that_is_not_a_column(self):
        self.assert_refused(
            "validated_column", "nonesuch", "'nonesuch'", "not a column"
        )

    def test_a_validated_column_that_is_not_boolean(self):
        self.assert_refused("validated_column", "plain", "'plain'", "not Boolean")


# ------------------------------------------------------------------------------------------------
# 3. the unit
# ------------------------------------------------------------------------------------------------


class TestTheUnit(unittest.TestCase):
    def test_every_compared_table_of_the_registry(self):
        _, specs = registry_specs()
        names = [spec["name"] for spec in specs]
        self.assertIn("BackgroundModel_tags", names)
        for name in names:
            with self.subTest(cls=name):
                expected = (
                    ["BackgroundModel", "BackgroundModel_tags", "BackgroundModelValue"]
                    if name == "BackgroundModel"
                    else [name]
                )
                self.assertEqual(expected, ShardedPool._unit_tables(name, specs))

    def test_a_declared_owner_joins_and_an_undeclared_reference_does_not(self):
        integer = sqla.Integer
        registry = {
            TAG_TABLE: factory(lambda: [sqla.Column(TAG_LABEL, sqla.String(64))]),
            "Owner": factory(lambda: [sqla.Column("n", integer)]),
            # a foreign key to Owner and no declaration: not Owner's
            "Referrer": factory(
                lambda: [
                    sqla.Column(
                        "owner_serial", integer, sqla.ForeignKey("Owner.serial")
                    )
                ]
            ),
            # declares Owner its owner
            "Owned": factory(
                lambda: [
                    sqla.Column("n", integer),
                    sqla.Column("owner_ref", integer, sqla.ForeignKey("Owner.serial")),
                ],
                owner_column="owner_ref",
            ),
            # neither replicated nor sharded: Owner's tag table
            "Owner_tags": factory(
                lambda: [
                    sqla.Column(
                        "owner_id",
                        integer,
                        sqla.ForeignKey("Owner.serial"),
                        primary_key=True,
                    ),
                    sqla.Column(
                        TAG_SERIAL,
                        integer,
                        sqla.ForeignKey(f"{TAG_TABLE}.serial"),
                        primary_key=True,
                    ),
                ],
                serial=False,
            ),
        }
        built = build_schema(sqla.MetaData(), registry)
        pool = bare_pool([TAG_TABLE, "Owner", "Referrer", "Owned"], registry=registry)
        specs = pool._replicated_table_specs(built)
        by_name = {spec["name"]: spec for spec in specs}
        self.assertEqual(
            [TAG_TABLE, "Owner", "Referrer", "Owned", "Owner_tags"], list(by_name)
        )

        self.assertEqual(
            ["Owner", "Owner_tags", "Owned"], ShardedPool._unit_tables("Owner", specs)
        )
        for alone in (TAG_TABLE, "Referrer", "Owned", "Owner_tags"):
            with self.subTest(cls=alone):
                self.assertEqual([alone], ShardedPool._unit_tables(alone, specs))

        # the column of each member that names the class's row
        self.assertEqual(
            "owner_ref", ShardedPool._column_naming(by_name["Owned"], "Owner")
        )
        self.assertEqual(
            "owner_id", ShardedPool._column_naming(by_name["Owner_tags"], "Owner")
        )


# ------------------------------------------------------------------------------------------------
# 4. the flags
# ------------------------------------------------------------------------------------------------


class TestTheFlags(unittest.TestCase):
    def test_the_declared_flags_reach_the_specs(self):
        _, specs = registry_specs()
        self.assertEqual(
            {"redshift": ("source", "response"), "wavenumber": ("source", "response")},
            {s["name"]: s["monotone_flags"] for s in specs if s["monotone_flags"]},
        )
        self.assertEqual(
            {"BackgroundModel": "validated"},
            {
                s["name"]: s["validated_column"]
                for s in specs
                if s["validated_column"] is not None
            },
        )
        self.assertEqual(
            {"BackgroundModelValue": "model_serial"},
            {s["name"]: s["owner_column"] for s in specs if s["owner_column"]},
        )

    REGISTRY = {
        "Flagged": factory(
            lambda: [
                sqla.Column("on", sqla.Boolean),
                sqla.Column("lit", sqla.Boolean),
                sqla.Column("off", sqla.Boolean),
            ],
            monotone_flags=("on", "lit"),
        ),
        "Checked": factory(
            lambda: [sqla.Column("ok", sqla.Boolean)], validated_column="ok"
        ),
    }

    def plan(self, operation, cls_name, rows):
        """The repair plan of a record of ``operation`` on ``cls_name`` (controlling shard 0,
        serial 1) over two shards holding ``rows``: shard -> {table: [row mappings]}."""
        with tempfile.TemporaryDirectory() as d:
            metadata = sqla.MetaData()
            built = build_schema(metadata, self.REGISTRY)
            pool = bare_pool(list(self.REGISTRY), registry=self.REGISTRY)
            specs = pool._replicated_table_specs(built)
            engines = {}
            try:
                for sid in (0, 1):
                    engines[sid] = sqla.create_engine(
                        f"sqlite:///{Path(d) / f'shard{sid}.sqlite'}",
                        future=True,
                        poolclass=sqla.pool.NullPool,
                    )
                    metadata.create_all(engines[sid])
                    with engines[sid].begin() as conn:
                        for table, table_rows in rows[sid].items():
                            conn.execute(sqla.insert(built.tables[table]), table_rows)
                conns = {sid: engines[sid].connect() for sid in engines}
                try:
                    snapshots = {
                        sid: pool._read_replicated_tables(conn, specs)
                        for sid, conn in conns.items()
                    }
                    differences = pool._replicated_differences(specs, snapshots)
                    record = {
                        "operation": operation,
                        "class_name": cls_name,
                        "controller_shard": 0,
                        "store_id": 1,
                        "started": datetime(2026, 10, 6),
                    }
                    return pool._plan_replicated_repair(
                        record, specs, snapshots, differences, conns
                    )
                finally:
                    for conn in conns.values():
                        conn.close()
            finally:
                for engine in engines.values():
                    engine.dispose()

    def flagged(self, on, lit, off):
        return {
            "Flagged": [{"serial": 1, "on": on, "lit": lit, "off": off}],
            "Checked": [{"serial": 1, "ok": False}],
        }

    def test_a_declared_flag_is_set_after_a_get(self):
        plan, refusals = self.plan(
            "get",
            "Flagged",
            {0: self.flagged(True, True, False), 1: self.flagged(False, False, False)},
        )
        self.assertEqual([], refusals)
        self.assertEqual({("Flagged", 1): {(1,): ["on", "lit"]}}, plan["flags"])
        self.assertFalse(plan["recompute"])

    def test_an_undeclared_flag_is_refused_in_the_declarations_words(self):
        plan, refusals = self.plan(
            "get",
            "Flagged",
            {0: self.flagged(True, False, True), 1: self.flagged(True, False, False)},
        )
        self.assertEqual(1, len(refusals))
        self.assertIn(
            "only a on or lit flag after a get, or ok after a validate, can be turned on "
            "from the controlling shard 0; shard 1 differs in ['off']",
            refusals[0]["detail"],
        )

    def test_the_declared_validated_column_is_recomputed_after_a_validate(self):
        rows = {sid: self.flagged(False, False, False) for sid in (0, 1)}
        rows[0]["Checked"] = [{"serial": 1, "ok": True}]
        plan, refusals = self.plan("validate", "Checked", rows)
        self.assertEqual([], refusals)
        self.assertTrue(plan["recompute"])
        self.assertEqual({}, plan["flags"])


# ------------------------------------------------------------------------------------------------
# 5. revalidate against validate
# ------------------------------------------------------------------------------------------------


class _Model(DatastoreObject):
    """What ``validate`` reads of a model: its serial and its label."""

    def __init__(self, store_id, label):
        super().__init__(store_id)
        self.label = label


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestRevalidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from Datastore.tests.real_store_fixtures import build_full_store

        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        (root / "store").mkdir()
        with quiet():
            build_full_store(root / "store")
        shard = sorted((root / "store").glob("*shard0000*.sqlite"))[0]
        cls.copy = root / "copy.sqlite"
        shutil.copyfile(shard, cls.copy)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def call(self, conn, built, how, serial, label, short):
        """One call of ``how`` on the row ``serial``, with one of its value rows deleted if
        ``short``, inside a savepoint rolled back after: (returned, flag written, printed).
        """
        table = built.tables["BackgroundModel"]
        values = built.tables["BackgroundModelValue"]
        bg = factories["BackgroundModel"]
        savepoint = conn.begin_nested()
        try:
            if short:
                victim = conn.execute(
                    sqla.select(sqla.func.max(values.c.serial)).filter(
                        values.c.model_serial == serial
                    )
                ).scalar()
                conn.execute(sqla.delete(values).where(values.c.serial == victim))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                if how == "validate":
                    result = bg.validate(
                        _Model(serial, label),
                        conn=conn,
                        table=table,
                        tables=built.tables,
                    )
                else:
                    result = bg.revalidate(serial, conn, table, built.tables)
            written = conn.execute(
                sqla.select(table.c.validated).filter(table.c.serial == serial)
            ).scalar()
            return result, written, out.getvalue()
        finally:
            savepoint.rollback()

    def test_the_same_result_flag_and_warning_as_stored_and_with_a_value_row_deleted(
        self,
    ):
        before = _checksum(self.copy)
        built = build_schema(sqla.MetaData(), factories)
        table = built.tables["BackgroundModel"]
        # the engine the check at open recomputes on, with SQLite's own transactions
        engine = bare_pool([])._reconcile_engine(self.copy)
        try:
            with engine.connect() as conn:
                conn.begin()
                models = conn.execute(
                    sqla.select(table.c.serial, table.c.label).order_by(table.c.serial)
                ).all()
                self.assertGreater(len(models), 0)
                for serial, label in models:
                    for short in (False, True):
                        with self.subTest(serial=serial, short=short):
                            a = self.call(conn, built, "validate", serial, label, short)
                            b = self.call(
                                conn, built, "revalidate", serial, label, short
                            )
                            self.assertEqual(a, b)
                            self.assertEqual(not short, a[0])
                            self.assertEqual(not short, a[1])
                            self.assertEqual(
                                short, "did not validate after serialization" in a[2]
                            )
                conn.rollback()
        finally:
            engine.dispose()
        self.assertEqual(before, _checksum(self.copy))

    def test_the_base_hook_raises(self):
        with self.assertRaises(NotImplementedError):
            SQLAFactoryBase.revalidate(1, None, None, None)


# ------------------------------------------------------------------------------------------------
# 6. owned_serials
# ------------------------------------------------------------------------------------------------


class _Row:
    def __init__(self, serial):
        self._my_id = serial


class _Owner:
    def __init__(self, serial, owned):
        self._my_id = serial
        self._values = [_Row(s) for s in owned]


class TestOwnedSerials(unittest.TestCase):
    def test_the_base_owns_nothing(self):
        self.assertIsNone(SQLAFactoryBase.owned_serials(_Owner(1, [2, 3])))

    def test_a_background_model_names_its_value_rows(self):
        bg = factories["BackgroundModel"]
        self.assertEqual([3, None, 5], bg.owned_serials(_Owner(1, [3, None, 5])))
        self.assertEqual([], bg.owned_serials(_Owner(1, [])))
        self.assertIsNone(bg.owned_serials(DatastoreObject(1)))

    def test_the_pool_checks_each_answer_through_the_hook(self):
        pool = bare_pool([])
        captured = {}

        def fake_write(operation, cls_name, submit_controller, replication):
            captured["replication"] = replication
            return None, None

        pool._replicated_write = fake_write
        pool._store_impl_replicated_table("BackgroundModel", _Owner(7, [1, 2]))
        check = captured["replication"](0, _Owner(7, [1, 2])).check

        check(2, _Owner(7, [1, 2]))
        with self.assertRaises(ReplicationMismatch) as raised:
            check(2, _Owner(7, [1, 3]))
        self.assertIn(
            "its value rows have serials [1, 3], the controller's [1, 2]",
            str(raised.exception),
        )


# ------------------------------------------------------------------------------------------------
# 7. the version object
# ------------------------------------------------------------------------------------------------


class TestTheVersionObject(unittest.TestCase):
    def test_built_by_the_registry_version_factory_with_the_attributes_it_had(self):
        version_factory = factories[VERSION_TABLE]
        calls = []

        class _Recording(version_factory):
            @staticmethod
            def build(payload, conn, table, inserter, tables, inserters):
                calls.append(
                    {
                        "payload": dict(payload),
                        "url": str(conn.engine.url),
                        "inserter": inserter,
                        "inserters": inserters,
                    }
                )
                return version_factory.build(
                    payload, conn, table, inserter, tables, inserters
                )

        registry = dict(factories)
        registry[VERSION_TABLE] = _Recording
        with tempfile.TemporaryDirectory() as d:
            primary = Path(d) / "store.sqlite"
            cluster = sp.StandinCluster()
            with cluster.active():
                pool = cluster.open_pool(primary, factories=registry)
                fresh = pool._version
                cluster.close_pool(pool)
                for read_only in (False, True):
                    calls.clear()
                    pool = cluster.open_pool(
                        primary, factories=registry, read_only=read_only
                    )
                    version = pool._version
                    cluster.close_pool(pool)
                    with self.subTest(read_only=read_only):
                        # the pool's own build, mode=ro on one shard: exactly one
                        ours = [c for c in calls if "mode=ro" in c["url"]]
                        self.assertEqual(1, len(ours))
                        self.assertEqual({VERSION_LABEL: "standin"}, ours[0]["payload"])
                        self.assertEqual({}, ours[0]["inserters"])
                        with self.assertRaises(RuntimeError):
                            ours[0]["inserter"](None, {VERSION_LABEL: "standin"})

                        self.assertIs(type(fresh), type(version))
                        self.assertEqual(
                            {"_my_id": 1, "_label": "standin", "_deserialized": True},
                            vars(version),
                        )


# ------------------------------------------------------------------------------------------------
# 8. the prune refusal's text
# ------------------------------------------------------------------------------------------------


class TestThePruneRefusalNamesItsUnit(_PruneTestCase):
    def test_a_prune_that_deletes_outside_its_unit(self):
        self.build()
        failing = self.shard_ids[1]
        execute(
            self.files[failing],
            'CREATE TRIGGER standin_spill AFTER DELETE ON "BackgroundModel" BEGIN '
            "DELETE FROM tolerance WHERE serial = (SELECT min(serial) FROM tolerance); END",
        )
        with quiet(), self.assertRaises(RuntimeError) as raised:
            self.open(prune_unvalidated=True)
        self.cluster.pool = None
        # the class, then the rest of its unit in the specs' order, as it has always read
        self.assertIn(
            '"tolerance" lost rows [(1,)], and a prune of "BackgroundModel" deletes only from '
            "['BackgroundModel', 'BackgroundModelValue', 'BackgroundModel_tags']",
            str(raised.exception),
        )


# ------------------------------------------------------------------------------------------------
# 9. no project name in ShardedPool.py
# ------------------------------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# the layer's own tables and columns (Datastore.contract), the columns schema.py prepends, and
# the layer's vocabulary: "validated" is the action record's key and the flag the check at open
# recomputes (prompt 07 keeps both)
_LAYER_WORDS = {
    VERSION_TABLE,
    VERSION_LABEL,
    TAG_TABLE,
    TAG_LABEL,
    TAG_SERIAL,
    "serial",
    "version",
    "timestamp",
    "stepping",
    "validated",
}

# the project's packages: none may be imported or named
_PROJECT_PACKAGES = {
    path.name
    for path in REPO_ROOT.iterdir()
    if (path / "__init__.py").is_file() and path.name not in ("Datastore", "config")
}


def _registry_names():
    """(tables, identifier columns) of the real registry, less the layer's own. A column is an
    identifier if it has an underscore or a capital: ``model_serial``, ``k_inv_Mpc``. A column
    that is a plain word, such as ``source``, ``response``, ``type`` or ``value``, cannot be told
    from English prose, and a word check would forbid the closed-store operations' "source
    store"; ``_FLAG_PHRASES`` checks the two declared flags instead. "model" and "value rows"
    are allowed: they are not names of any table, and the two as-unit refusals and the
    recompute's messages keep them so that no message's text changes (prompt 07, §2)."""
    built = build_schema(sqla.MetaData(), factories)
    tables = set(built.tables) - _LAYER_WORDS
    columns = {
        c.name
        for t in built.tables.values()
        for c in t.columns
        if "_" in c.name or c.name.lower() != c.name
    } - _LAYER_WORDS
    return tables, columns


_FLAG_PHRASES = re.compile(
    r"\b(source|response)\s+(or|and)\s+(source|response)\b|\b(source|response)\s+flags?\b"
)


def _named(text: str, forbidden) -> list:
    return sorted({w for w in _WORD.findall(text) if w in forbidden})


class TestNoProjectNameInThePool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = POOL_FILE.read_text()
        cls.tree = ast.parse(cls.source)
        cls.tables, cls.columns = _registry_names()

    def test_no_string_or_name_is_a_registry_table(self):
        forbidden = self.tables | self.columns | _PROJECT_PACKAGES
        found = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found += [(node.lineno, w) for w in _named(node.value, forbidden)]
                if _FLAG_PHRASES.search(node.value):
                    found.append((node.lineno, node.value))
            names = []
            if isinstance(node, ast.Name):
                names = [node.id]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.arg):
                names = [node.arg]
            elif isinstance(node, ast.keyword) and node.arg is not None:
                names = [node.arg]
            for name in names:
                parts = {name} | set(name.split("_"))
                found += [(node.lineno, name) for p in parts if p in forbidden]
        self.assertEqual([], found)

    def test_no_import_names_a_project_package(self):
        # the standard library, ray, sqlalchemy, the layer's own modules (no factory but the
        # base) and config.defaults, a generic constant
        allowed_roots = set(sys.stdlib_module_names) | {"ray", "sqlalchemy"}
        found = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module]
            else:
                continue
            for module in modules:
                root = module.split(".")[0]
                if root in allowed_roots or module == "config.defaults":
                    continue
                if root == "Datastore" and not (
                    module.startswith("Datastore.SQL.ObjectFactories")
                    and module != "Datastore.SQL.ObjectFactories.base"
                ):
                    continue
                found.append((node.lineno, module))
        self.assertEqual([], found)

    def test_no_comment_or_docstring_names_one(self):
        forbidden = self.tables | self.columns | _PROJECT_PACKAGES
        found = []
        for token in tokenize.generate_tokens(io.StringIO(self.source).readline):
            if token.type == tokenize.COMMENT:
                found += [(token.start[0], w) for w in _named(token.string, forbidden)]
                if _FLAG_PHRASES.search(token.string):
                    found.append((token.start[0], token.string))
        for node in ast.walk(self.tree):
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                doc = ast.get_docstring(node, clean=False)
                if doc is None:
                    continue
                found += [(node.__class__.__name__, w) for w in _named(doc, forbidden)]
                if _FLAG_PHRASES.search(doc):
                    found.append((node.__class__.__name__, doc))
        self.assertEqual([], found)


if __name__ == "__main__":
    unittest.main()
