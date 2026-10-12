"""
Tests for prompt 03 of ``prompts/datastore-integrity`` (board items L1-L5): identity lookups match
exactly, and a duplicate raises (docs/datastore-integrity-audit.md §3.5-§3.9).

  * L1 -- ``sqla_wavenumber_exit_time_factory.build`` matched an inequality: any stored row at
    least as tight as the request, ordered loosest-first, so tightening the tolerance later left
    the *original* request served the tighter row (reporting the stored pair, not the requested
    one) and, once two qualifying rows existed, raised ``MultipleResultsFound`` on the original
    request (audit §3.5, ``probe_04_exit_time_and_solver.py``). Fixed to serial equality on
    ``atol_serial``/``rtol_serial``/``stepping`` -- the same match every other keyed type uses,
    since ``tolerance.build`` has already mapped a value to one serial (``probe_04b``'s verified
    shape).
  * L1 also -- the query ran once per sub-horizon column, six times total, with only the last
    (complete) execution's result kept (audit §3.6).
  * L2 -- ``sqla_IntegrationSolver_factory.build`` matched ``stepping >=`` and took ``.scalar()``,
    so a lower stepping could be served a higher-order row that happened to satisfy the inequality
    (audit §3.7). Fixed to ``stepping ==`` and ``.scalar_one_or_none()``.
  * L3 -- eight identity lookups took ``.scalar()``, which returns the first row and never notices
    a second (audit §3.8): ``tolerance``, ``store_tag``, ``version``, ``IntegrationSolver`` (L2),
    ``GkSourcePolicy``, ``QuadSourcePolicy``, ``LambdaCDM`` and ``QCD_Cosmology``. Fixed to
    ``.scalar_one_or_none()``, which raises ``MultipleResultsFound`` on a duplicate.
  * L4 -- ``sqla_redshift_factory.build`` matched ``abs((z - stored) / stored) < tol``, which
    divides by the *stored* value: at ``z = 0`` every candidate row divides by zero, SQLite's
    ``0/0`` is ``NULL``, nothing matches, and every call at ``z = 0`` inserted a fresh row (audit
    §3.9). Fixed with an explicit branch: ``z == 0`` matches by equality, every other ``z`` keeps
    the unchanged relative match.

No Ray, no datastore file, nothing under ``var/``: the schema is built in memory with
``build_schema(MetaData(), config.datastore.factories)``, exactly as
``docs/datastore-integrity-audit/lookup_keys/common.py`` and the probes do, and every call below is
to the real, unmodified ``build()`` of the factory named.
"""

import unittest
from datetime import datetime
from math import log10

import sqlalchemy as sqla
from sqlalchemy import event
from sqlalchemy.exc import MultipleResultsFound

from CosmologyConcepts import wavenumber
from CosmologyConcepts.wavenumber import (
    WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS,
    WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS,
)
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.LambdaCDM import Planck2018
from config.datastore import factories as _factories
from ObjectFactories.GkSourcePolicy import sqla_GkSourcePolicy_factory
from ObjectFactories.LambdaCDM import sqla_LambdaCDM_factory
from ObjectFactories.QCD_Cosmology import sqla_QCDCosmology_factory
from ObjectFactories.QuadSourcePolicy import sqla_QuadSourcePolicy_factory
from ObjectFactories.integration_metadata import (
    sqla_IntegrationSolver_factory,
)
from ObjectFactories.redshift import sqla_redshift_factory
from ObjectFactories.store_tag import sqla_store_tag_factory
from ObjectFactories.tolerance import sqla_tolerance_factory
from ObjectFactories.version import sqla_version_factory
from ObjectFactories.wavenumber import sqla_wavenumber_exit_time_factory
from datastorekit.SQL.schema import build_schema
from MetadataConcepts import tolerance
from Units import Mpc_units

UNITS = Mpc_units()
FIXED_TIMESTAMP = datetime(2026, 9, 29, 12, 0, 0)


class _Cosmology:
    """Anything a factory reads ``.store_id``, ``.type_id`` and ``.units`` off, standing in for a
    ``BaseCosmology``."""

    type_id = 0

    def __init__(self, store_id: int = 1):
        self.store_id = store_id
        self.units = UNITS


class Store:
    """
    Every table ``build_schema`` declares, in one in-memory SQLite database. No Ray, no file,
    nothing under ``var/``: ``build_schema`` only declares ``sqla.Table`` objects, and
    ``create_all`` makes an in-memory schema from them -- the same construction
    ``docs/datastore-integrity-audit/lookup_keys/common.py`` and
    ``ObjectFactories/tests/test_integral_read_batch.py`` use.
    """

    def __init__(self):
        self.metadata = sqla.MetaData()
        built = build_schema(self.metadata, _factories)
        self.tables = built.tables
        self.records = built.records
        self.engine = sqla.create_engine("sqlite://", future=True)
        self.metadata.create_all(self.engine)
        self.conn = self.engine.connect()

    def close(self):
        self.conn.close()
        self.engine.dispose()

    def _stamped(self, table_name: str, payload: dict) -> dict:
        rec = self.records[table_name]
        payload = dict(payload)
        if rec.get("use_timestamp", False):
            payload.setdefault("timestamp", FIXED_TIMESTAMP)
        if rec.get("use_version", False):
            payload.setdefault("version", 1)
        return payload

    def insert(self, table_name: str, **row):
        table = self.tables[table_name]
        return self.conn.execute(
            sqla.insert(table), self._stamped(table_name, row)
        ).lastrowid

    def inserter(self, table_name: str):
        """A callable of the shape ``build()`` expects for its ``inserter`` argument."""
        table = self.tables[table_name]

        def _insert(conn, data):
            return conn.execute(
                sqla.insert(table), self._stamped(table_name, data)
            ).lastrowid

        return _insert


def _exit_time_row(
    s: Store,
    serial: int,
    atol_serial: int,
    rtol_serial: int,
    stepping: int = 0,
    z_exit: float = 1.0e4,
) -> None:
    row = dict(
        serial=serial,
        stepping=stepping,
        wavenumber_serial=1,
        cosmology_type=0,
        cosmology_serial=1,
        atol_serial=atol_serial,
        rtol_serial=rtol_serial,
        compute_time=0.1,
        z_exit=z_exit,
    )
    for o in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
        row[f"z_exit_suph_e{o}"] = 1.0
    for o in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
        row[f"z_exit_subh_e{o}"] = 1.0
    s.insert("wavenumber_exit_time", **row)


# tolerance serials this module uses throughout: -12 -> 1, -10 -> 2, -9 -> 3, -8 -> 4
_TOL_SERIAL = {-12.0: 1, -10.0: 2, -9.0: 3, -8.0: 4}


def _tol(log10_tol: float) -> tolerance:
    return tolerance(store_id=_TOL_SERIAL[log10_tol], log10_tol=log10_tol)


class ExitTimeTestCase(unittest.TestCase):
    """Shared fixture for the exit-time tests: one wavenumber, the four tolerances above."""

    def setUp(self):
        self.s = s = Store()
        self.addCleanup(s.close)
        s.insert("version", serial=1, label="test_exact_identity_lookups")
        for log10_tol, serial in _TOL_SERIAL.items():
            s.insert("tolerance", serial=serial, log10_tol=log10_tol)
        s.insert("wavenumber", serial=1, k_inv_Mpc=1.0e5, source=True, response=True)
        s.conn.commit()

    def _lookup(self, atol: float, rtol: float, stepping=None):
        payload = dict(
            k=wavenumber(store_id=1, k_inv_Mpc=1.0e5, units=UNITS),
            cosmology=_Cosmology(),
            atol=_tol(atol),
            rtol=_tol(rtol),
        )
        if stepping is not None:
            payload["stepping"] = stepping
        return sqla_wavenumber_exit_time_factory.build(
            payload,
            self.s.conn,
            self.s.tables["wavenumber_exit_time"],
            None,
            self.s.tables,
            None,
        )


class TestExitTimeExactMatch(ExitTimeTestCase):
    """§3 item 1 of the prompt: stored (1e-10, 1e-9), the exact request hits and reports its own
    pair; (1e-10, 1e-8), (1e-8, 1e-8) and (1e-12, 1e-9) miss. With (1e-12, 1e-9) stored as well,
    each exact request is served its own row and nothing raises. Stepping 1 stored: request 0
    misses, request 1 hits."""

    def test_exact_request_hits_and_reports_its_own_pair(self):
        _exit_time_row(self.s, 1, _TOL_SERIAL[-10.0], _TOL_SERIAL[-9.0])
        self.s.conn.commit()

        obj = self._lookup(-10.0, -9.0)

        self.assertTrue(obj.available)
        self.assertEqual(obj.store_id, 1)
        # the object reports the pair it was asked for, which -- now that the match is exact --
        # is necessarily the stored row's own pair
        self.assertEqual(obj._atol.log10_tol, -10.0)
        self.assertEqual(obj._rtol.log10_tol, -9.0)

    def test_looser_and_tighter_requests_miss(self):
        _exit_time_row(self.s, 1, _TOL_SERIAL[-10.0], _TOL_SERIAL[-9.0])
        self.s.conn.commit()

        for atol, rtol in [(-10.0, -8.0), (-8.0, -8.0), (-12.0, -9.0)]:
            with self.subTest(atol=atol, rtol=rtol):
                obj = self._lookup(atol, rtol)
                self.assertFalse(obj.available)

    def test_two_stored_rows_each_served_its_own_and_nothing_raises(self):
        _exit_time_row(self.s, 1, _TOL_SERIAL[-10.0], _TOL_SERIAL[-9.0], z_exit=1.0e4)
        _exit_time_row(
            self.s, 2, _TOL_SERIAL[-12.0], _TOL_SERIAL[-9.0], z_exit=1.00001e4
        )
        self.s.conn.commit()

        first = self._lookup(-10.0, -9.0)
        self.assertTrue(first.available)
        self.assertEqual(first.store_id, 1)

        second = self._lookup(-12.0, -9.0)
        self.assertTrue(second.available)
        self.assertEqual(second.store_id, 2)

        # a request neither row satisfies exactly still just misses -- it does not raise, because
        # only one row (if any) can now equal a given (atol_serial, rtol_serial, stepping)
        third = self._lookup(-8.0, -8.0)
        self.assertFalse(third.available)

    def test_stepping_must_match_exactly(self):
        _exit_time_row(self.s, 1, _TOL_SERIAL[-10.0], _TOL_SERIAL[-9.0], stepping=1)
        self.s.conn.commit()

        miss = self._lookup(-10.0, -9.0, stepping=0)
        self.assertFalse(miss.available)

        hit = self._lookup(-10.0, -9.0, stepping=1)
        self.assertTrue(hit.available)
        self.assertEqual(hit.store_id, 1)


class TestExitTimeOneStatement(ExitTimeTestCase):
    """§3 item 2: a build() of the exit time executes exactly one SELECT, counted with an
    event.listen(engine, "before_cursor_execute", ...) listener -- before L1, the query ran once
    per sub-horizon column (audit §3.6, six statements measured by probe_04)."""

    def test_build_executes_exactly_one_select(self):
        _exit_time_row(self.s, 1, _TOL_SERIAL[-10.0], _TOL_SERIAL[-9.0])
        self.s.conn.commit()

        statements = []

        def _before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            statements.append(statement)

        event.listen(self.s.engine, "before_cursor_execute", _before_cursor_execute)
        try:
            obj = self._lookup(-10.0, -9.0)
        finally:
            event.remove(self.s.engine, "before_cursor_execute", _before_cursor_execute)

        self.assertTrue(obj.available)
        selects = [st for st in statements if st.strip().upper().startswith("SELECT")]
        self.assertEqual(len(selects), 1)


class TestIntegrationSolverExactMatch(unittest.TestCase):
    """§3 item 3: stored {serial 1: stepping 8, serial 2: stepping 4}, request 4 -> serial 2,
    request 8 -> serial 1, request 6 inserts a new row (audit §3.7, probe_04's measurement).
    """

    def setUp(self):
        self.s = s = Store()
        self.addCleanup(s.close)
        s.insert("IntegrationSolver", serial=1, label="cumulative-GL", stepping=8)
        s.insert("IntegrationSolver", serial=2, label="cumulative-GL", stepping=4)
        s.conn.commit()

    def _build(self, stepping: int):
        return sqla_IntegrationSolver_factory.build(
            {"label": "cumulative-GL", "stepping": stepping},
            self.s.conn,
            self.s.tables["IntegrationSolver"],
            self.s.inserter("IntegrationSolver"),
            self.s.tables,
            None,
        )

    def test_request_4_is_served_serial_2(self):
        obj = self._build(4)
        self.assertEqual(obj.store_id, 2)
        self.assertFalse(getattr(obj, "_new_insert", False))

    def test_request_8_is_served_serial_1(self):
        obj = self._build(8)
        self.assertEqual(obj.store_id, 1)
        self.assertFalse(getattr(obj, "_new_insert", False))

    def test_request_6_inserts_a_new_row(self):
        obj = self._build(6)
        self.assertTrue(getattr(obj, "_new_insert", False))
        self.assertNotIn(obj.store_id, (1, 2))
        self.assertEqual(obj.stepping, 6)


class TestDuplicatesRaise(unittest.TestCase):
    """§3 item 4: for each of L3's eight identity lookups, a duplicate makes build() raise
    MultipleResultsFound. One test method per class (audit §3.8's eight sites)."""

    def setUp(self):
        self.s = s = Store()
        self.addCleanup(s.close)
        s.insert("version", serial=1, label="test_exact_identity_lookups")
        s.conn.commit()

    def test_tolerance(self):
        s = self.s
        s.insert("tolerance", serial=2, log10_tol=-10.0)
        s.insert("tolerance", serial=3, log10_tol=-10.0)
        s.conn.commit()
        with self.assertRaises(MultipleResultsFound):
            sqla_tolerance_factory.build(
                {"log10_tol": -10.0},
                s.conn,
                s.tables["tolerance"],
                s.inserter("tolerance"),
                s.tables,
                None,
            )

    def test_store_tag(self):
        s = self.s
        s.insert("store_tag", serial=1, label="dup")
        s.insert("store_tag", serial=2, label="dup")
        s.conn.commit()
        with self.assertRaises(MultipleResultsFound):
            sqla_store_tag_factory.build(
                {"label": "dup"},
                s.conn,
                s.tables["store_tag"],
                s.inserter("store_tag"),
                s.tables,
                None,
            )

    def test_version(self):
        s = self.s
        # the version row inserted by setUp is serial 1, label "test_exact_identity_lookups";
        # add a second row under a duplicate label of its own so the two rows this test cares
        # about are not entangled with that fixture row
        s.insert("version", serial=2, label="dup")
        s.insert("version", serial=3, label="dup")
        s.conn.commit()
        with self.assertRaises(MultipleResultsFound):
            sqla_version_factory.build(
                {"label": "dup"},
                s.conn,
                s.tables["version"],
                s.inserter("version"),
                s.tables,
                None,
            )

    def test_integration_solver(self):
        s = self.s
        s.insert("IntegrationSolver", serial=1, label="dup", stepping=4)
        s.insert("IntegrationSolver", serial=2, label="dup", stepping=4)
        s.conn.commit()
        with self.assertRaises(MultipleResultsFound):
            sqla_IntegrationSolver_factory.build(
                {"label": "dup", "stepping": 4},
                s.conn,
                s.tables["IntegrationSolver"],
                s.inserter("IntegrationSolver"),
                s.tables,
                None,
            )

    def test_gk_source_policy(self):
        s = self.s
        s.insert(
            "GkSourcePolicy",
            serial=1,
            label="dup",
            Levin_threshold=1.5,
            numeric_policy="maximize-WKB",
        )
        s.insert(
            "GkSourcePolicy",
            serial=2,
            label="dup",
            Levin_threshold=1.5,
            numeric_policy="maximize-WKB",
        )
        s.conn.commit()
        with self.assertRaises(MultipleResultsFound):
            sqla_GkSourcePolicy_factory.build(
                {"Levin_threshold": 1.5, "numeric_policy": "maximize-WKB"},
                s.conn,
                s.tables["GkSourcePolicy"],
                s.inserter("GkSourcePolicy"),
                s.tables,
                None,
            )

    def test_quad_source_policy(self):
        s = self.s
        s.insert(
            "QuadSourcePolicy",
            serial=1,
            label="dup",
            Levin_threshold=1.5,
            numeric_policy="maximize-WKB",
        )
        s.insert(
            "QuadSourcePolicy",
            serial=2,
            label="dup",
            Levin_threshold=1.5,
            numeric_policy="maximize-WKB",
        )
        s.conn.commit()
        with self.assertRaises(MultipleResultsFound):
            sqla_QuadSourcePolicy_factory.build(
                {"Levin_threshold": 1.5, "numeric_policy": "maximize-WKB"},
                s.conn,
                s.tables["QuadSourcePolicy"],
                s.inserter("QuadSourcePolicy"),
                s.tables,
                None,
            )

    def test_lambda_cdm(self):
        s = self.s
        p = Planck2018()
        row = dict(
            name=p.name,
            omega_m=p.omega_m,
            omega_cc=p.omega_cc,
            h=p.h,
            f_baryon=p.f_baryon,
            T_CMB_Kelvin=p.T_CMB_Kelvin,
            Neff=p.Neff,
        )
        s.insert("LambdaCDM", serial=1, **row)
        s.insert("LambdaCDM", serial=2, **row)
        s.conn.commit()
        with self.assertRaises(MultipleResultsFound):
            sqla_LambdaCDM_factory.build(
                {"params": p, "units": UNITS},
                s.conn,
                s.tables["LambdaCDM"],
                s.inserter("LambdaCDM"),
                s.tables,
                None,
            )

    def test_qcd_cosmology(self):
        """QCD_Cosmology.build's .scalar() sits inside a try/except SQLAlchemyError whose handler
        (ObjectFactories/QCD_Cosmology.py:98-117, read for this test) re-raises
        everything except a message naming "T_z_representation", which it turns into a
        RuntimeError about a pre-representation datastore. MultipleResultsFound is a
        SQLAlchemyError, but its message ("Multiple rows were found when one or none was
        required") never names that column, so the handler's bare `raise` passes it through
        unchanged: this test must see MultipleResultsFound itself, not the RuntimeError.
        """
        s = self.s
        p = Planck2018()
        max_z = 1.0e6
        row = dict(
            name=p.name,
            omega_m=p.omega_m,
            omega_cc=p.omega_cc,
            h=p.h,
            f_baryon=p.f_baryon,
            T_CMB_Kelvin=p.T_CMB_Kelvin,
            Neff=p.Neff,
            log10_max_z=log10(1.0 + max_z),
            T_z_representation=QCD_Cosmology.T_Z_REPRESENTATION_VERSION,
        )
        s.insert("QCD_Cosmology", serial=1, **row)
        s.insert("QCD_Cosmology", serial=2, **row)
        s.conn.commit()
        with self.assertRaises(MultipleResultsFound):
            sqla_QCDCosmology_factory.build(
                {"params": p, "units": UNITS, "max_z": max_z},
                s.conn,
                s.tables["QCD_Cosmology"],
                s.inserter("QCD_Cosmology"),
                s.tables,
                None,
            )


class TestRedshiftZero(unittest.TestCase):
    """§3 item 5: three builds at z = 0 give one row; a build at z = 0.1 is unchanged (audit
    §3.9, probe_06's measurement: before L4, three calls at z = 0 inserted three rows).
    """

    def setUp(self):
        self.s = s = Store()
        self.addCleanup(s.close)

    def _build(self, z: float):
        return sqla_redshift_factory.build(
            dict(z=z, is_source=False, is_response=True),
            self.s.conn,
            self.s.tables["redshift"],
            self.s.inserter("redshift"),
            self.s.tables,
            None,
        )

    def test_three_builds_at_zero_give_one_row(self):
        first = self._build(0.0)
        self.s.conn.commit()
        second = self._build(0.0)
        self.s.conn.commit()
        third = self._build(0.0)
        self.s.conn.commit()

        self.assertEqual(first.store_id, second.store_id)
        self.assertEqual(first.store_id, third.store_id)

        rows = self.s.conn.execute(
            self.s.tables["redshift"]
            .select()
            .where(self.s.tables["redshift"].c.z == 0.0)
        ).fetchall()
        self.assertEqual(len(rows), 1)

    def test_z_0pt1_is_unchanged(self):
        first = self._build(0.1)
        self.s.conn.commit()
        second = self._build(0.1)
        self.s.conn.commit()

        self.assertEqual(first.store_id, second.store_id)

        rows = self.s.conn.execute(
            self.s.tables["redshift"]
            .select()
            .where(self.s.tables["redshift"].c.z == 0.1)
        ).fetchall()
        self.assertEqual(len(rows), 1)


def tearDownModule():
    import ray

    if ray.is_initialized():
        raise RuntimeError(
            "test_exact_identity_lookups: ray was initialised by this module's tests, which must "
            "not happen (README §5 rule 9)"
        )


if __name__ == "__main__":
    unittest.main()
