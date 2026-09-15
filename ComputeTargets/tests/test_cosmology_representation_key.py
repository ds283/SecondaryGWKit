"""
Tests for prompt 03 of ``prompts/qcd-background-audit``: how ``T(z)`` is represented is part of the
QCD cosmology's datastore lookup key.

Before this commit ``sqla_QCDCosmology_factory.build()`` matched a cosmology on seven parameter
values and ``log10_max_z``. Every one of those is an *input* to the model; none of them can see how
the background is built from those inputs -- the tolerance each ``T(z)`` node is root-solved to,
the quantity that is splined, the node count, the spline order, the segmentation, or the
break-point set. So when prompts 04, 05, 06 and 07 change the representation, a pre-existing
datastore returns the same cosmology row under the same serial, every ``BackgroundModel`` keyed on
that serial deserialises, and its ``tau`` / ``cs_tau`` / ``friction_F`` limbs are the *old*
background's. There is no exception, no warning, and no column that differs.

Four things are checked, in the order prompt 03 §3 states them:

1. **Two representations, two serials.** The same parameter block under
   ``T_Z_REPRESENTATION_VERSION`` 1 and 2 produces two distinct rows; the same version twice reuses
   one. This runs the production ``build()`` against a real in-memory SQLite database built from
   the factory's own ``register()``, so the assertion is about what SQL actually does.
2. **The filter reads the constant, not a literal.** Read with ``ast`` (following
   ``test_numeric_break_point_key.test_the_integrator_call_site_reads_the_class_constant``), and
   then demonstrated dynamically: re-point the single declaration and both the query's ``WHERE``
   clause and the value written by the insert follow it together.
3. **A table without the column raises the campaign's message**, not an opaque ``OperationalError``
   a reader cannot interpret. ``Datastore._ensure_tables()`` creates absent tables but never alters
   an existing one, so an old datastore keeps its old table and the mismatch surfaces as SQLite's
   "no such column" when ``build()``'s select executes.
4. **``LambdaCDM`` is untouched.** It computes ``T = T_CMB (1+z)`` in closed form, has no ``T(z)``
   spline and no representation to key on, and must not acquire a column.

Nothing here needs Ray or the project's ``Datastore`` machinery. The tables are built from the
factories' own ``register()`` output in the shape ``Datastore.SQL.Datastore._build_schema`` builds
them, and the inserter is the shape of ``Datastore._insert`` reduced to what this table uses
(a serial from the engine, and a timestamp).
"""

import ast
import contextlib
import io
import unittest
from datetime import datetime
from pathlib import Path

import sqlalchemy as sqla
from sqlalchemy.exc import OperationalError

from CosmologyModels.GenericEOS.LambdaCDM_GenericEOS import LambdaCDM_GenericEOS
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.LambdaCDM import Planck2018
from Datastore.SQL.ObjectFactories.LambdaCDM import sqla_LambdaCDM_factory
from Datastore.SQL.ObjectFactories.QCD_Cosmology import sqla_QCDCosmology_factory
from Units import Mpc_units

REPO_ROOT = Path(__file__).parents[2]

FACTORY_SOURCE = (
    REPO_ROOT / "Datastore" / "SQL" / "ObjectFactories" / "QCD_Cosmology.py"
)

# the production configuration (config/model_list.py)
PRODUCTION_MAX_Z = 1e20

KEY_COLUMN = "T_z_representation"
CONSTANT_NAME = "T_Z_REPRESENTATION_VERSION"

# the seven parameter columns LambdaCDM keys on. It has no log10_max_z either, because it
# tabulates nothing.
LAMBDACDM_COLUMNS = [
    "name",
    "omega_m",
    "omega_cc",
    "h",
    "f_baryon",
    "T_CMB_Kelvin",
    "Neff",
]


# ---------------------------------------------------------------------------------------------
# schema and connection helpers
# ---------------------------------------------------------------------------------------------


def _build_table(
    name: str, factory, metadata: sqla.MetaData, omit: str = None
) -> sqla.Table:
    """
    Reproduce the table Datastore._build_schema builds for this factory, from the factory's own
    register() output -- so that the schema under test is the production schema, not a copy of it.

    ``omit`` drops one column, which is how the pre-prompt-03 schema is reproduced: there is no
    other record of it, and hand-writing the old column list would not stay in step with the
    factory.
    """
    registration = factory.register()

    tab = sqla.Table(name, metadata)
    if registration.get("serial", True):
        tab.append_column(sqla.Column("serial", sqla.Integer, primary_key=True))
    if registration.get("version", False):
        tab.append_column(sqla.Column("version", sqla.Integer))
    if registration.get("timestamp", False):
        tab.append_column(sqla.Column("timestamp", sqla.DateTime()))
    if registration.get("stepping", False):
        tab.append_column(sqla.Column("stepping", sqla.Integer))

    for column in registration.get("columns", []):
        if omit is not None and column.name == omit:
            continue
        tab.append_column(column)

    return tab


def _inserter_for(table: sqla.Table):
    """
    What Datastore._insert does for this table: no serial broker, no version, a timestamp, and the
    serial the engine assigns.
    """

    def inserter(conn, data):
        payload = dict(data) | {"timestamp": datetime.now()}
        return conn.execute(sqla.insert(table), payload).lastrowid

    return inserter


class _QueryCaptured(Exception):
    """
    raised by the stand-in connection in place of executing. Deliberately *not* a SQLAlchemyError,
    so that it passes through the factory's missing-column handler untouched.
    """

    def __init__(self, query):
        super().__init__("query captured")
        self.query = query


class _CapturingConnection:
    def execute(self, query):
        raise _QueryCaptured(query)


class _RaisingConnection:
    def __init__(self, exc):
        self._exc = exc

    def execute(self, query):
        raise self._exc


def _missing_column_error() -> OperationalError:
    """
    what SQLite raises when a query names a column an existing table does not have. Only used by
    the "an unrelated error is not disguised" companion test; the positive case below runs against
    a genuine old-schema table and lets SQLite raise it for real.
    """
    return OperationalError(
        f"SELECT QCD_Cosmology.serial ... WHERE QCD_Cosmology.{KEY_COLUMN} = ?",
        {},
        Exception(f"no such column: QCD_Cosmology.{KEY_COLUMN}"),
    )


def _payload(max_z: float = PRODUCTION_MAX_Z) -> dict:
    return {"params": Planck2018(), "units": Mpc_units(), "max_z": max_z}


def _build(conn, table, inserter):
    """
    Run the production build(). Constructing a QCD_Cosmology costs 500 root solves and prints a
    banner; the banner is swallowed so the test output stays readable.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return sqla_QCDCosmology_factory.build(
            _payload(), conn, table, inserter, None, None
        )


def _capture_build_query():
    """Run build() far enough to form its query, and hand the query back."""
    metadata = sqla.MetaData()
    table = _build_table("QCD_Cosmology", sqla_QCDCosmology_factory, metadata)

    try:
        sqla_QCDCosmology_factory.build(
            _payload(), _CapturingConnection(), table, None, None, None
        )
    except _QueryCaptured as e:
        return e.query

    raise AssertionError("QCD_Cosmology.build() did not issue a query")


def _equality_criteria(query) -> dict:
    """
    column name -> the value the WHERE clause requires it to equal, for the top-level equality
    criteria of a compiled select. The other criteria are the parameter block's
    abs(column - value) < DEFAULT_FLOAT_PRECISION comparisons, which are not equalities and do not
    appear here.
    """
    criteria = {}
    for clause in query.whereclause.clauses:
        left = getattr(clause, "left", None)
        right = getattr(clause, "right", None)
        if left is None or not hasattr(left, "name"):
            continue
        if not hasattr(right, "value"):
            continue
        criteria[left.name] = right.value

    return criteria


# ---------------------------------------------------------------------------------------------
# ast helpers
# ---------------------------------------------------------------------------------------------


def _build_function() -> ast.FunctionDef:
    tree = ast.parse(FACTORY_SOURCE.read_text(), filename=str(FACTORY_SOURCE))
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "build"
    ]
    assert (
        len(functions) == 1
    ), "expected exactly one build() in the QCD cosmology factory"
    return functions[0]


def _is_key_column(node: ast.AST) -> bool:
    """``table.c.T_z_representation``"""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == KEY_COLUMN
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "c"
    )


# ---------------------------------------------------------------------------------------------


class TestTheDeclaration(unittest.TestCase):
    """Prompt 03 §2 item 1: one declaration, on the generic-EOS base, inherited by QCD."""

    def test_the_constant_lives_on_the_base_class_and_qcd_inherits_it(self):
        self.assertIn(CONSTANT_NAME, LambdaCDM_GenericEOS.__dict__)

        # not shadowed: a second copy on the subclass would be a second thing to bump
        self.assertNotIn(CONSTANT_NAME, QCD_Cosmology.__dict__)
        self.assertIs(
            QCD_Cosmology.T_Z_REPRESENTATION_VERSION,
            LambdaCDM_GenericEOS.T_Z_REPRESENTATION_VERSION,
        )

    def test_it_is_an_integer_identifier(self):
        version = QCD_Cosmology.T_Z_REPRESENTATION_VERSION
        self.assertIsInstance(version, int)
        self.assertNotIsInstance(version, bool)
        self.assertGreaterEqual(version, 1)

    def test_it_is_readable_from_the_class_without_an_instance(self):
        # the factory filters on it before any model exists, so this is load-bearing
        self.assertEqual(
            getattr(QCD_Cosmology, CONSTANT_NAME),
            QCD_Cosmology.T_Z_REPRESENTATION_VERSION,
        )


class TestTheKeyField(unittest.TestCase):
    """Prompt 03 §2 item 2."""

    def test_the_column_is_registered_as_a_non_nullable_integer(self):
        columns = {c.name: c for c in sqla_QCDCosmology_factory.register()["columns"]}
        self.assertIn(KEY_COLUMN, columns)

        column = columns[KEY_COLUMN]
        self.assertIsInstance(column.type, sqla.Integer)
        self.assertFalse(column.nullable)

        # an identifier, not a measured quantity: it must not be a float, because the parameter
        # block's DEFAULT_FLOAT_PRECISION comparison has no business near it
        self.assertNotIsInstance(column.type, sqla.Float)

    def test_build_filters_on_it_beside_the_parameter_key(self):
        criteria = _equality_criteria(_capture_build_query())

        self.assertIn(KEY_COLUMN, criteria)
        self.assertEqual(criteria[KEY_COLUMN], QCD_Cosmology.T_Z_REPRESENTATION_VERSION)

        # the parameter block is still matched, by the tolerance comparisons it always used --
        # the representation is an addition to the key, not a replacement for it
        compiled = str(
            _capture_build_query().whereclause.compile(
                compile_kwargs={"literal_binds": True}
            )
        )
        for column in (
            "omega_m",
            "omega_cc",
            "h",
            "f_baryon",
            "T_CMB_Kelvin",
            "Neff",
            "log10_max_z",
        ):
            self.assertIn(column, compiled)
        self.assertIn(KEY_COLUMN, compiled)


class TestTheFilterReadsTheConstant(unittest.TestCase):
    """
    Prompt 03 §3 item 2. A literal in build() would silently stop tracking the constant the moment
    prompt 04 bumps it, and that is the exact failure this prompt exists to prevent.
    """

    def test_the_filter_is_not_a_literal(self):
        comparisons = [
            node
            for node in ast.walk(_build_function())
            if isinstance(node, ast.Compare) and _is_key_column(node.left)
        ]
        self.assertEqual(len(comparisons), 1)

        comparison = comparisons[0]
        self.assertEqual(len(comparison.comparators), 1)

        operand = comparison.comparators[0]
        self.assertNotIsInstance(operand, ast.Constant)
        self.assertIsInstance(operand, ast.Name)

        # ... and the name it reads is bound, in build(), to the class constant and to nothing else
        bindings = [
            node
            for node in ast.walk(_build_function())
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == operand.id
                for target in node.targets
            )
        ]
        self.assertEqual(len(bindings), 1)

        value = bindings[0].value
        self.assertIsInstance(value, ast.Attribute)
        self.assertEqual(value.attr, CONSTANT_NAME)
        self.assertIsInstance(value.value, ast.Name)
        self.assertEqual(value.value.id, "QCD_Cosmology")

    def test_the_insert_writes_the_same_name_the_filter_reads(self):
        """the stored value and the queried value cannot disagree"""
        comparison = [
            node
            for node in ast.walk(_build_function())
            if isinstance(node, ast.Compare) and _is_key_column(node.left)
        ][0]
        queried = comparison.comparators[0].id

        written = [
            value
            for node in ast.walk(_build_function())
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == KEY_COLUMN
        ]
        self.assertEqual(len(written), 1)
        self.assertIsInstance(written[0], ast.Name)
        self.assertEqual(written[0].id, queried)

    def test_repointing_the_constant_moves_the_query_with_it(self):
        """
        The direct demonstration: change the one declaration, and the key the factory queries on
        follows it. Nothing else in the factory mentions the representation.
        """
        original = LambdaCDM_GenericEOS.T_Z_REPRESENTATION_VERSION
        sentinel = original + 41
        LambdaCDM_GenericEOS.T_Z_REPRESENTATION_VERSION = sentinel
        try:
            self.assertEqual(
                _equality_criteria(_capture_build_query())[KEY_COLUMN], sentinel
            )
        finally:
            LambdaCDM_GenericEOS.T_Z_REPRESENTATION_VERSION = original

        self.assertEqual(
            _equality_criteria(_capture_build_query())[KEY_COLUMN], original
        )


class TestTwoRepresentationsTwoSerials(unittest.TestCase):
    """
    Prompt 03 §3 item 1, against a real database. The whole point of the prompt is what SQL does
    with a candidate row, so this runs the production build() against SQLite rather than reasoning
    about the compiled query.
    """

    def setUp(self):
        self.metadata = sqla.MetaData()
        self.table = _build_table(
            "QCD_Cosmology", sqla_QCDCosmology_factory, self.metadata
        )
        self.engine = sqla.create_engine("sqlite://", future=True)
        self.table.create(self.engine)
        self.conn = self.engine.connect()
        self.inserter = _inserter_for(self.table)
        self.addCleanup(self.engine.dispose)
        self.addCleanup(self.conn.close)

    def _rows(self):
        return list(
            self.conn.execute(
                sqla.select(self.table.c.serial, self.table.c.T_z_representation)
            )
        )

    def test_the_same_representation_reuses_a_row_and_a_different_one_does_not(self):
        original = LambdaCDM_GenericEOS.T_Z_REPRESENTATION_VERSION
        try:
            LambdaCDM_GenericEOS.T_Z_REPRESENTATION_VERSION = 1
            first = _build(self.conn, self.table, self.inserter)
            second = _build(self.conn, self.table, self.inserter)

            # the same parameter block under the same representation is the same cosmology
            self.assertEqual(first.store_id, second.store_id)
            self.assertTrue(getattr(first, "_new_insert", False))
            self.assertTrue(getattr(second, "_deserialized", False))
            self.assertEqual(len(self._rows()), 1)

            LambdaCDM_GenericEOS.T_Z_REPRESENTATION_VERSION = 2
            third = _build(self.conn, self.table, self.inserter)
        finally:
            LambdaCDM_GenericEOS.T_Z_REPRESENTATION_VERSION = original

        # ... and under a different representation it is a different background, so the pre-prompt
        # row must not be served
        self.assertNotEqual(third.store_id, first.store_id)
        self.assertTrue(getattr(third, "_new_insert", False))

        rows = self._rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row.T_z_representation for row in rows},
            {1, 2},
        )
        self.assertEqual(
            {row.serial: row.T_z_representation for row in rows},
            {first.store_id: 1, third.store_id: 2},
        )

    def test_the_row_written_carries_the_running_representation(self):
        built = _build(self.conn, self.table, self.inserter)

        stored = self.conn.execute(
            sqla.select(self.table.c.T_z_representation).filter(
                self.table.c.serial == built.store_id
            )
        ).scalar()

        self.assertEqual(stored, QCD_Cosmology.T_Z_REPRESENTATION_VERSION)


class TestOldSchemaFailsLoudly(unittest.TestCase):
    """
    Prompt 03 §2 item 3 and §3 item 3. There is no defensible representation to assume for a row
    that records none, so there is no migration and no silent hit.
    """

    def test_a_table_without_the_column_demands_regeneration(self):
        metadata = sqla.MetaData()
        # the pre-prompt-03 schema: the production registration with the new column dropped
        table = _build_table(
            "QCD_Cosmology", sqla_QCDCosmology_factory, metadata, omit=KEY_COLUMN
        )
        engine = sqla.create_engine("sqlite://", future=True)
        table.create(engine)
        self.addCleanup(engine.dispose)

        # build() must query the column it now keys on, so the table handed to it is the
        # production one -- exactly as Datastore._build_schema builds it from the code, while the
        # database underneath still holds the old table. _ensure_tables() creates absent tables
        # but never alters an existing one, which is why the two can differ.
        current = _build_table(
            "QCD_Cosmology", sqla_QCDCosmology_factory, sqla.MetaData()
        )

        with engine.connect() as conn:
            with self.assertRaises(RuntimeError) as caught:
                _build(conn, current, _inserter_for(current))

        message = str(caught.exception)
        self.assertIn(KEY_COLUMN, message)
        self.assertIn("prompt 03", message)
        self.assertIn("regenerated", message)
        self.assertIn("no migration", message)
        self.assertIn("QCD_Cosmology", message)

        # and the underlying database error is preserved for anyone who wants it
        self.assertIsInstance(caught.exception.__cause__, OperationalError)

    def test_the_error_it_raises_is_not_the_opaque_one(self):
        """the failure mode being avoided: an unhandled 'no such column' a reader cannot act on"""
        metadata = sqla.MetaData()
        table = _build_table(
            "QCD_Cosmology", sqla_QCDCosmology_factory, metadata, omit=KEY_COLUMN
        )
        engine = sqla.create_engine("sqlite://", future=True)
        table.create(engine)
        self.addCleanup(engine.dispose)

        current = _build_table(
            "QCD_Cosmology", sqla_QCDCosmology_factory, sqla.MetaData()
        )

        with engine.connect() as conn:
            with self.assertRaises(RuntimeError):
                _build(conn, current, _inserter_for(current))

    def test_an_unrelated_database_error_is_not_disguised(self):
        unrelated = OperationalError("SELECT ...", {}, Exception("database is locked"))

        metadata = sqla.MetaData()
        table = _build_table("QCD_Cosmology", sqla_QCDCosmology_factory, metadata)

        with self.assertRaises(OperationalError):
            _build(_RaisingConnection(unrelated), table, None)

        # ... and the missing-column error, arriving by the same route, still converts
        with self.assertRaises(RuntimeError):
            _build(_RaisingConnection(_missing_column_error()), table, None)


class TestLambdaCDMIsUntouched(unittest.TestCase):
    """
    Prompt 03 §3 item 4 and README §2 (g). LambdaCDM returns T_CMB (1+z) in closed form, tabulates
    nothing and declares no break points, so it has no representation to key on.
    """

    def test_the_lambdacdm_column_list_is_unchanged(self):
        columns = [c.name for c in sqla_LambdaCDM_factory.register()["columns"]]
        self.assertEqual(columns, LAMBDACDM_COLUMNS)
        self.assertNotIn(KEY_COLUMN, columns)

    def test_the_lambdacdm_factory_never_mentions_the_representation(self):
        source = (
            REPO_ROOT / "Datastore" / "SQL" / "ObjectFactories" / "LambdaCDM.py"
        ).read_text()
        self.assertNotIn(KEY_COLUMN, source)
        self.assertNotIn(CONSTANT_NAME, source)


if __name__ == "__main__":
    unittest.main()
