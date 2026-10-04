"""
A database error reaches the caller as itself: prompt 01 of ``prompts/datastore-generic``
(``docs/datastore-generic-audit.md`` F0, board item M4).

Four factories used to catch every ``SQLAlchemyError`` and ask whether a column's name was in
``str(e)``, in order to say that a datastore "predates" the column. ``str(e)`` of a SQLAlchemy
database error carries the SQL statement, and the statement names the column, so the question was
answered yes for *every* error -- a ``database is locked`` included -- and the caller was told to
regenerate a store that was fine; ``BackgroundModel``'s read path re-ran its query without the
grid-identity columns and carried on. The tests that were meant to catch it raised
``OperationalError("SELECT ...", ...)``, a statement that names no column, so the handler's test
never matched and the tests passed for as long as the defect lasted
(``docs/datastore-generic-audit/str_e_misfire_probe.py``).

The handlers are deleted. What is pinned here is that they stay deleted, and the statement is the
point: **it is never written out in this module**. ``_StatementRaisingConnection.execute`` is
handed the query the factory built, compiles it, and raises ``OperationalError`` with *that text*
as the error's statement. So the error's ``str()`` names every column the factory asked for, as
SQLite's does, and each test checks that it does (``columns=``) before checking anything else: a
handler of the old kind would match it. A test that raised ``"SELECT ..."`` instead would pass
with the handler restored, which is the whole failure being guarded against.

For every path the prompt names, the exception that reaches the caller is **the one that was
raised** (identity, not merely its type), and for ``BackgroundModel`` the connection is executed
the number of times the path has queries, which on the read path is once: no second query.

No Ray, no datastore file, nothing under ``var/``: the schema is
``build_schema(MetaData(), Datastore.SQL.Datastore._factories)`` in memory, as in
``test_value_reads_parent_key.py``, and every call is to the real factory code.
"""

import unittest
from importlib import import_module

from sqlalchemy.dialects import sqlite
from sqlalchemy.exc import OperationalError

from CosmologyConcepts import redshift, redshift_array
from CosmologyConcepts.wavenumber import SOURCE_GRID_CONSTRUCTION_VERSION
from CosmologyModels.LambdaCDM import Planck2018
from Datastore.SQL.ObjectFactories.BackgroundModel import sqla_BackgroundModelFactory
from Datastore.SQL.ObjectFactories.GkNumericIntegration import (
    sqla_GkNumericIntegration_factory,
    sqla_GkNumericValue_factory,
)
from Datastore.SQL.ObjectFactories.QCD_Cosmology import sqla_QCDCosmology_factory
from Datastore.SQL.ObjectFactories.TkNumericIntegration import (
    sqla_TkNumericIntegration_factory,
    sqla_TkNumericValue_factory,
)
from Datastore.tests.test_value_reads_parent_key import (
    ATOL,
    K_EXIT,
    MODEL,
    RTOL,
    _base_store,
    _gk_numeric_model_payload,
    _tk_numeric_model_payload,
)
from Units import Mpc_units

# the module, not the class: ComputeTargets/__init__.py rebinds the package attribute to the class,
# and the factory reaches the three Gauss orders the same way
BackgroundModelModule = import_module("ComputeTargets.BackgroundModel")

UNITS = Mpc_units()

# the columns the old handlers looked for, which every statement the factories issue names
BREAK_POINT_KIND = "break_point_kind"
T_Z_REPRESENTATION = "T_z_representation"
GRID_IDENTITY = ("source_grid_digest", "source_grid_construction")
GAUSS_ORDERS = ("tau_gauss_order", "cs_tau_gauss_order", "friction_F_gauss_order")
VALUE_TABLE_COLUMNS = ("tau_lo_Mpc", "cs_tau_Mpc", "cs_tau_lo_Mpc", "friction_F")


class _StatementRaisingConnection:
    """
    A stand-in for the connection a factory executes against. The first ``succeed`` calls are
    passed to ``inner``; every later call raises an ``OperationalError`` whose statement is the
    **compiled text of the query it was handed**, not a literal.

    ``executed`` counts every call; ``raised`` is the exception object, for an identity check.
    """

    def __init__(self, inner=None, succeed: int = 0):
        self.inner = inner
        self.succeed = succeed
        self.executed = 0
        self.raised = None

    @staticmethod
    def _statement(query) -> str:
        return str(query.compile(dialect=sqlite.dialect()))

    def execute(self, query):
        self.executed += 1
        if self.executed <= self.succeed:
            return self.inner.execute(query)

        self.raised = OperationalError(
            self._statement(query), {}, Exception("database is locked")
        )
        raise self.raised


class _Cosmology:
    """what BackgroundModel's factory reads off a cosmology: an identity and a unit system"""

    type_id = 4242
    store_id = 1
    units = UNITS


def _grid() -> redshift_array:
    return redshift_array(
        [
            redshift(store_id=serial, z=z, is_source=True)
            for serial, z in ((1, 100.0), (2, 10.0), (3, 1.0))
        ]
    )


def _background_payload(z_sample) -> dict:
    return {
        "solver_labels": {},
        "cosmology": _Cosmology(),
        "z_sample": z_sample,
        "tags": [],
    }


class _PassThroughCase(unittest.TestCase):
    def assert_passes_through(self, call, conn, columns):
        """
        ``call()`` raises exactly what ``conn`` raised: the same object, not a ``RuntimeError``
        and not another ``OperationalError``. The precondition is checked first: the error's
        ``str()`` names every one of ``columns``, so that a ``"<column>" in str(e)`` test would
        have matched it.
        """
        with self.assertRaises(OperationalError) as caught:
            call()

        self.assertIsNotNone(conn.raised)
        for column in columns:
            self.assertIn(column, str(conn.raised))
        self.assertIs(caught.exception, conn.raised)
        self.assertIsNone(caught.exception.__cause__)


class TestTheNumericFactories(_PassThroughCase):
    def setUp(self):
        self.s = _base_store()
        self.addCleanup(self.s.close)

    def _parent_payload(self, redshift_key):
        return {
            "solver_labels": {},
            "atol": ATOL,
            "rtol": RTOL,
            "k": K_EXIT,
            "model": MODEL,
            "z_sample": None,
            redshift_key: None,
            "tags": [],
        }

    def test_tk_numeric_integration_build(self):
        conn = _StatementRaisingConnection()
        self.assert_passes_through(
            lambda: sqla_TkNumericIntegration_factory.build(
                self._parent_payload("z_init"),
                conn,
                self.s.tables["TkNumericIntegration"],
                None,
                self.s.tables,
                None,
            ),
            conn,
            (BREAK_POINT_KIND,),
        )
        self.assertEqual(1, conn.executed)

    def test_gk_numeric_integration_build(self):
        conn = _StatementRaisingConnection()
        self.assert_passes_through(
            lambda: sqla_GkNumericIntegration_factory.build(
                self._parent_payload("z_source"),
                conn,
                self.s.tables["GkNumericIntegration"],
                None,
                self.s.tables,
                None,
            ),
            conn,
            (BREAK_POINT_KIND,),
        )
        self.assertEqual(1, conn.executed)

    def test_tk_numeric_value_build(self):
        conn = _StatementRaisingConnection()
        self.assert_passes_through(
            lambda: sqla_TkNumericValue_factory.build(
                _tk_numeric_model_payload(),
                conn,
                self.s.tables["TkNumericValue"],
                None,
                self.s.tables,
                None,
            ),
            conn,
            (BREAK_POINT_KIND,),
        )
        self.assertEqual(1, conn.executed)

    def test_gk_numeric_value_build(self):
        conn = _StatementRaisingConnection()
        self.assert_passes_through(
            lambda: sqla_GkNumericValue_factory.build(
                _gk_numeric_model_payload(),
                conn,
                self.s.tables["GkNumericValue"],
                None,
                self.s.tables,
                None,
            ),
            conn,
            (BREAK_POINT_KIND,),
        )
        self.assertEqual(1, conn.executed)

    def test_gk_numeric_value_read_batch(self):
        conn = _StatementRaisingConnection()
        self.assert_passes_through(
            lambda: list(
                sqla_GkNumericValue_factory.read_batch(
                    {"model": MODEL, "k": K_EXIT, "atol": ATOL, "rtol": RTOL},
                    conn,
                    self.s.tables["GkNumericValue"],
                    self.s.tables,
                )
            ),
            conn,
            (BREAK_POINT_KIND,),
        )
        self.assertEqual(1, conn.executed)


class TestQCDCosmology(_PassThroughCase):
    def test_qcd_cosmology_build(self):
        s = _base_store()
        self.addCleanup(s.close)

        conn = _StatementRaisingConnection()
        self.assert_passes_through(
            lambda: sqla_QCDCosmology_factory.build(
                {"params": Planck2018(), "units": UNITS, "max_z": 1.0e3},
                conn,
                s.tables["QCD_Cosmology"],
                None,
                s.tables,
                None,
            ),
            conn,
            (T_Z_REPRESENTATION,),
        )
        self.assertEqual(1, conn.executed)


class TestBackgroundModel(_PassThroughCase):
    def setUp(self):
        self.s = _base_store()
        self.addCleanup(self.s.close)

    def _build(self, conn, z_sample):
        return sqla_BackgroundModelFactory.build(
            _background_payload(z_sample),
            conn,
            self.s.tables["BackgroundModel"],
            None,
            self.s.tables,
            None,
        )

    def test_the_read_path_is_executed_once_and_the_error_reaches_the_caller(self):
        """z_sample=None: the path that used to re-run its query without the grid identity"""
        conn = _StatementRaisingConnection()
        self.assert_passes_through(
            lambda: self._build(conn, None), conn, GRID_IDENTITY + GAUSS_ORDERS
        )
        self.assertEqual(1, conn.executed)

    def test_the_compute_path_is_executed_once_and_the_error_reaches_the_caller(self):
        conn = _StatementRaisingConnection()
        self.assert_passes_through(
            lambda: self._build(conn, _grid()), conn, GRID_IDENTITY + GAUSS_ORDERS
        )
        self.assertEqual(1, conn.executed)

    def test_the_value_table_read_error_reaches_the_caller(self):
        """
        The first query, the model's own row, succeeds against a real row; the second, the
        value-table read, raises.
        """
        self.s.insert(
            "BackgroundModel",
            serial=1,
            label="found",
            cosmology_type=_Cosmology.type_id,
            cosmology_serial=_Cosmology.store_id,
            tau_gauss_order=BackgroundModelModule.TAU_GAUSS_ORDER,
            cs_tau_gauss_order=BackgroundModelModule.CS_TAU_GAUSS_ORDER,
            friction_F_gauss_order=BackgroundModelModule.FRICTION_F_GAUSS_ORDER,
            solver_serial=1,
            z_init_serial=1,
            z_samples=3,
            source_grid_digest="a-digest",
            source_grid_construction=SOURCE_GRID_CONSTRUCTION_VERSION,
            compute_time=1.0,
            compute_steps=1,
            RHS_evaluations=1,
            mean_RHS_time=1.0,
            max_RHS_time=1.0,
            min_RHS_time=1.0,
            validated=True,
        )
        self.s.conn.commit()

        conn = _StatementRaisingConnection(inner=self.s.conn, succeed=1)
        self.assert_passes_through(
            lambda: self._build(conn, None), conn, VALUE_TABLE_COLUMNS
        )
        self.assertEqual(2, conn.executed)


if __name__ == "__main__":
    unittest.main()
