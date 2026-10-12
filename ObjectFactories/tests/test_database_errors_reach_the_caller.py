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

Prompt 02 of the same campaign (board item N3) deleted the same kind of handler from
``TkWKBIntegration``, ``GkWKBIntegration``, ``GkSource`` and ``QuadSourceIntegral``, and the
classes from ``TestTheWKBIntegrations`` down pin that an error from their statements reaches the
caller unchanged. Their handlers matched the driver's own message (``e.orig``), not ``str(e)``, so
they never misfired on a lock, and a test of pass-through alone cannot tell a faithful restoration
of one from its absence. What these tests catch is a handler that *does* turn an error into
another (``raise`` for every ``OperationalError``, or a miss for one); each still names, in
``columns=``, the column the deleted handler looked for.

No Ray, no datastore file, nothing under ``var/``: the schema is
``build_schema(MetaData(), config.datastore.factories)`` in memory, as in
``test_value_reads_parent_key.py``, and every call is to the real factory code.
"""

import unittest
from importlib import import_module

from sqlalchemy.dialects import sqlite
from sqlalchemy.exc import OperationalError

from CosmologyConcepts import redshift, redshift_array
from CosmologyConcepts.wavenumber import SOURCE_GRID_CONSTRUCTION_VERSION
from CosmologyModels.LambdaCDM import Planck2018
from ObjectFactories.BackgroundModel import sqla_BackgroundModelFactory
from ObjectFactories.GkNumericIntegration import (
    sqla_GkNumericIntegration_factory,
    sqla_GkNumericValue_factory,
)
from ObjectFactories.QCD_Cosmology import sqla_QCDCosmology_factory
from ObjectFactories.TkNumericIntegration import (
    sqla_TkNumericIntegration_factory,
    sqla_TkNumericValue_factory,
)
from ObjectFactories.GkSource import (
    sqla_GkSource_factory,
    sqla_GkSourceValue_factory,
)
from ObjectFactories.GkWKBIntegration import (
    sqla_GkWKBIntegration_factory,
    sqla_GkWKBValue_factory,
)
from ObjectFactories.QuadSourceIntegral import (
    sqla_QuadSourceIntegral_factory,
)
from ObjectFactories.TkWKBIntegration import (
    sqla_TkWKBIntegration_factory,
    sqla_TkWKBValue_factory,
)
from ObjectFactories.tests import test_gksource_parent_set as gk_source_keys
from ObjectFactories.tests import test_quadsource_integral_parent_key as qsi_keys
from ObjectFactories.tests import test_wkb_numeric_parent_key as wkb_keys
from ObjectFactories.tests.test_value_reads_parent_key import (
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

# prompt 02: the columns its deleted handlers looked for
NUMERIC_SERIAL = "numeric_serial"
RHO_GAUSS_ORDER = "rho_gauss_order"
PARENTS_DIGEST = "parents_digest"
PARENTS_TABLE = "GkSource_parents"
QUAD_POLICY_SERIAL = "quad_policy_serial"
TRANSFER_FUNCTION_COLUMNS = (
    "Tq_numeric_serial",
    "Tq_WKB_serial",
    "Tr_numeric_serial",
    "Tr_WKB_serial",
)


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


class TestTheWKBIntegrations(_PassThroughCase):
    """``TkWKBIntegration.build`` and ``GkWKBIntegration.build``, which looked for
    ``numeric_serial`` and ``rho_gauss_order``."""

    def setUp(self):
        self.s = wkb_keys._base_store()
        self.addCleanup(self.s.close)

    def test_tk_wkb_integration_build(self):
        conn = _StatementRaisingConnection()
        self.assert_passes_through(
            lambda: sqla_TkWKBIntegration_factory.build(
                wkb_keys._tk_payload(numeric_serial=1),
                conn,
                self.s.tables["TkWKBIntegration"],
                None,
                self.s.tables,
                None,
            ),
            conn,
            (NUMERIC_SERIAL, RHO_GAUSS_ORDER),
        )
        self.assertEqual(1, conn.executed)

    def test_gk_wkb_integration_build(self):
        conn = _StatementRaisingConnection()
        self.assert_passes_through(
            lambda: sqla_GkWKBIntegration_factory.build(
                wkb_keys._gk_payload(numeric_serial=1),
                conn,
                self.s.tables["GkWKBIntegration"],
                None,
                self.s.tables,
                None,
            ),
            conn,
            (NUMERIC_SERIAL, RHO_GAUSS_ORDER),
        )
        self.assertEqual(1, conn.executed)


class TestTheWKBValueReads(_PassThroughCase):
    """The four WKB value reads, which looked for ``numeric_serial``. Since
    prompts/datastore-generic prompt 05 none selects it, and a read that names no parent does not
    filter on it, so the statement is recognised by ``rho_gauss_order``, which every one of them
    filters on.
    """

    def setUp(self):
        self.s = wkb_keys._base_store()
        self.addCleanup(self.s.close)

    def _payload(self, **extra):
        payload = {
            "model": gk_source_keys.MODEL,
            "k": gk_source_keys.K_EXIT,
            "z": gk_source_keys.Z_RESPONSE,
        }
        payload.update(extra)
        return payload

    def _build(self, factory, name, conn, payload):
        return factory.build(
            payload, conn, self.s.tables[name], None, self.s.tables, None
        )

    def _read_batch(self, factory, name, conn, payload):
        return list(
            factory.read_batch(payload, conn, self.s.tables[name], self.s.tables)
        )

    def test_tk_wkb_value_build(self):
        for extra in ({}, {"numeric_serial": 1}):
            with self.subTest(**extra):
                conn = _StatementRaisingConnection()
                self.assert_passes_through(
                    lambda: self._build(
                        sqla_TkWKBValue_factory,
                        "TkWKBValue",
                        conn,
                        self._payload(**extra),
                    ),
                    conn,
                    (RHO_GAUSS_ORDER,),
                )
                self.assertEqual(1, conn.executed)

    def test_gk_wkb_value_build(self):
        for extra in ({}, {"numeric_serial": 1}, {"numeric_serial": None}):
            with self.subTest(**{k: str(v) for k, v in extra.items()}):
                conn = _StatementRaisingConnection()
                self.assert_passes_through(
                    lambda: self._build(
                        sqla_GkWKBValue_factory,
                        "GkWKBValue",
                        conn,
                        self._payload(z_source=gk_source_keys.Z_SOURCES[23], **extra),
                    ),
                    conn,
                    (RHO_GAUSS_ORDER,),
                )
                self.assertEqual(1, conn.executed)

    def test_tk_wkb_value_read_batch(self):
        for extra in ({}, {"numeric_serial": 1}):
            with self.subTest(**extra):
                conn = _StatementRaisingConnection()
                self.assert_passes_through(
                    lambda: self._read_batch(
                        sqla_TkWKBValue_factory,
                        "TkWKBValue",
                        conn,
                        self._payload(**extra),
                    ),
                    conn,
                    (RHO_GAUSS_ORDER,),
                )
                self.assertEqual(1, conn.executed)

    def test_gk_wkb_value_read_batch(self):
        for extra in ({}, {"numeric_serial": 1}, {"parents_only": True}):
            with self.subTest(**{k: str(v) for k, v in extra.items()}):
                conn = _StatementRaisingConnection()
                self.assert_passes_through(
                    lambda: self._read_batch(
                        sqla_GkWKBValue_factory,
                        "GkWKBValue",
                        conn,
                        self._payload(**extra),
                    ),
                    conn,
                    (RHO_GAUSS_ORDER,),
                )
                self.assertEqual(1, conn.executed)


class TestGkSource(_PassThroughCase):
    """``GkSource.build`` and ``GkSourceValue.build``, which looked for ``parents_digest`` and for
    ``GkSource_parents``."""

    def setUp(self):
        self.s = gk_source_keys._gk_store()
        self.addCleanup(self.s.close)

    def _source_payload(self, **extra):
        payload = {
            "model": gk_source_keys.MODEL,
            "k": gk_source_keys.K_EXIT,
            "z_response": gk_source_keys.Z_RESPONSE,
            "z_sample": None,
            "tags": [],
            "_do_not_populate": True,
        }
        payload.update(extra)
        return payload

    def _build_source(self, conn, **extra):
        return sqla_GkSource_factory.build(
            self._source_payload(**extra),
            conn,
            self.s.tables["GkSource"],
            None,
            self.s.tables,
            None,
        )

    def test_gk_source_build(self):
        conn = _StatementRaisingConnection()
        self.assert_passes_through(
            lambda: self._build_source(conn, parents=gk_source_keys.SET_A),
            conn,
            (PARENTS_DIGEST,),
        )
        self.assertEqual(1, conn.executed)

    def test_gk_source_build_in_the_reader_mode(self):
        conn = _StatementRaisingConnection()
        self.assert_passes_through(
            lambda: self._build_source(conn, labels_only=True),
            conn,
            (PARENTS_DIGEST,),
        )
        # the labels-only mode counts the matches before it reads the one
        self.assertEqual(1, conn.executed)

    def test_the_read_of_the_parent_set_reaches_the_caller(self):
        """
        The reader mode's third statement, which reads the matched row's ``GkSource_parents``
        rows and which the deleted handler also matched, by that table's name. The first two (the
        count of matches, then the row) succeed against a real, stored GkSource.
        """
        gk_source_keys.store_source(self.s, gk_source_keys.SET_A, tags=())
        conn = _StatementRaisingConnection(inner=self.s.conn, succeed=2)
        self.assert_passes_through(
            lambda: self._build_source(conn, labels_only=True),
            conn,
            (PARENTS_TABLE,),
        )
        self.assertEqual(3, conn.executed)

    def test_gk_source_value_build(self):
        conn = _StatementRaisingConnection()
        self.assert_passes_through(
            lambda: sqla_GkSourceValue_factory.build(
                {
                    "model": gk_source_keys.MODEL,
                    "k": gk_source_keys.K_EXIT,
                    "z_response": gk_source_keys.Z_RESPONSE,
                    "z_source": gk_source_keys.Z_SOURCES[22],
                },
                conn,
                self.s.tables["GkSourceValue"],
                None,
                self.s.tables,
                None,
            ),
            conn,
            (PARENTS_DIGEST,),
        )
        self.assertEqual(1, conn.executed)


class TestQuadSourceIntegral(qsi_keys._WithAStore, _PassThroughCase):
    """``QuadSourceIntegral.build`` and ``read_batch``, which looked for ``quad_policy_serial`` and
    for the four transfer-function columns. (``test_quadsource_policy_key`` also holds a test of
    this property, the rewritten
    ``TestAnotherDatabaseErrorPassesThrough.test_another_database_error_is_not_turned_into_it``.)
    """

    def test_quad_source_integral_build(self):
        conn = _StatementRaisingConnection()
        payload = self.labels()
        payload.update(qsi_keys.PARENTS)
        self.assert_passes_through(
            lambda: sqla_QuadSourceIntegral_factory.build(
                payload, conn, self.table(), None, self.s.tables, None
            ),
            conn,
            (QUAD_POLICY_SERIAL,) + TRANSFER_FUNCTION_COLUMNS,
        )
        self.assertEqual(1, conn.executed)

    def test_quad_source_integral_read_batch(self):
        for tags in ((), ("A",)):
            with self.subTest(tags=tags):
                conn = _StatementRaisingConnection()
                self.assert_passes_through(
                    lambda: list(
                        sqla_QuadSourceIntegral_factory.read_batch(
                            self.labels(tags), conn, self.table(), self.s.tables
                        )
                    ),
                    conn,
                    (QUAD_POLICY_SERIAL,) + TRANSFER_FUNCTION_COLUMNS,
                )
                self.assertEqual(1, conn.executed)


if __name__ == "__main__":
    unittest.main()
