"""
Tests for prompt 07 of ``prompts/datastore-integrity`` (board items K1, K2): a WKB row is keyed on
the numeric row it starts from (docs/datastore-integrity-audit.md §2.3, §3.13).

A ``TkWKBIntegration`` starts from its ``TkNumericIntegration``'s stop point, and a
``GkWKBIntegration`` from its ``GkNumericIntegration``'s, or is *pure WKB* (``G_init = 0``,
``Gprime_init = 1`` at ``z_source``). Before this prompt the WKB row recorded nothing of the
numeric row, and its lookup filtered ``z_init`` only optionally, so once any WKB row existed for
``k`` it was served whatever numeric row existed now. Now each row carries ``numeric_serial``
(NOT NULL on Tk; nullable on Gk, NULL meaning pure WKB), ``build()`` filters on it, and
``store()`` refuses initial data that are not the parent's stop values bit for bit.

The prompt's §3 tests, by class:

  1. ``TestKeyedOnTheParent`` -- two Tk numeric rows for one ``k`` (different tolerances), a WKB
     row stored from the first: naming the first finds it, naming the second misses. The same for
     Gk, and ``z_init`` / the initial values are no longer in the key (``TestZInitIsPayload``).
  2. ``TestPureWKBIsItsOwnKey`` -- a seeded and a pure-WKB Gk row for one ``(k, z_source)``
     coexist, and each lookup finds its own, through ``build()`` and the value reads.
  3. ``TestAParentIsRequired`` -- a Tk lookup or construction without a parent is an error; so is
     a Gk lookup given neither a parent nor the pure-WKB marker.
  4. ``TestTheConsistencyCheck`` -- one ulp (``math.nextafter``) off the parent's stop value, in
     ``z_init`` or an initial value, and ``store()`` refuses, naming both values. Rows seeded the
     way ``main.py`` seeds them -- the numeric row read back serial-only through its own factory,
     and ``z_init = k_exit.z_exit - Tk.stop_deltaz_subh`` -- store cleanly.
  5. ``TestReadersRefuseAmbiguity`` -- the labels-only reader mode refuses two stored rows for the
     same physical labels, naming both ``numeric_serial`` values (a NULL as pure WKB).
  7. ``TestDropOrder`` -- under ``PRAGMA foreign_keys = ON``, dropping ``tk-numeric`` with a Tk WKB
     row present succeeds in the order derived from the foreign keys and not unsorted; and that
     order respects every declared foreign key between the dropped tables.

and, for the inventory's nullable parent (the orchestrator note's §0.5), ``TestNullableParent``.

No Ray, nothing under ``var/``: the schema is built in memory (or, for the drop order, in a file in
a temporary directory) with ``build_schema(MetaData(), config.datastore.factories)``, and
every call is to the real factory code. Parents that exist only as a serial -- a model, a tag, a
solver -- are stand-ins carrying ``.store_id``.
"""

import importlib
import tempfile
import types
import unittest
from datetime import datetime
from math import nextafter
from pathlib import Path
from unittest import mock

import ray
import sqlalchemy as sqla
from sqlalchemy.exc import IntegrityError, MultipleResultsFound

import ComputeTargets.phase_residual as phase_residual
from ComputeTargets import (
    GkNumericIntegration,
    GkWKBIntegration,
    GkWKBValue,
    TkNumericIntegration,
    TkWKBIntegration,
    TkWKBValue,
)
from CosmologyConcepts import redshift, redshift_array, wavenumber, wavenumber_exit_time
from CosmologyConcepts.wavenumber import (
    WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS,
    WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS,
)
from config.datastore import drop_groups, factories as _factories, tables_to_drop
from Datastore.SQL.ObjectFactories.GkNumericIntegration import (
    sqla_GkNumericIntegration_factory,
)
from Datastore.SQL.ObjectFactories.GkWKBIntegration import (
    sqla_GkWKBIntegration_factory,
    sqla_GkWKBValue_factory,
)
from Datastore.SQL.ObjectFactories.TkNumericIntegration import (
    sqla_TkNumericIntegration_factory,
)
from Datastore.SQL.ObjectFactories.TkWKBIntegration import (
    sqla_TkWKBIntegration_factory,
)
from datastorekit.SQL.schema import build_schema, drop_order
from datastorekit.store_inventory import Parent, ShardContext, read_records
from MetadataConcepts import tolerance
from Units import Mpc_units

# the module itself: `import datastorekit.SQL.Datastore as ...` binds the package's re-exported
# actor class of the same name, not the module whose drop_order TestDropOrder patches
datastore_module = importlib.import_module("datastorekit.SQL.Datastore")

UNITS = Mpc_units()
FIXED_TIMESTAMP = datetime(2026, 9, 29, 12, 0, 0)

# production-sized redshifts, where one ulp of z_init is ~1e-5 and DEFAULT_FLOAT_PRECISION is
# nowhere near it: the stop point is a subtraction that rounds
Z_EXIT = 1.2345678901234567e11
Z_SOURCE = 5.0e9
STOP_DELTAZ_SUBH = 2.3456789012345678e10

# numeric parents: serial -> (atol serial, the stop values). Tk rows 1 and 2 share a k and differ
# only in their tolerance, and so in their stop point
TK_NUMERIC = {
    1: (1, 0.123456789012345, -3.2109876543210e-12, STOP_DELTAZ_SUBH),
    2: (3, 0.123456789098765, -3.2109876500000e-12, STOP_DELTAZ_SUBH * 1.000001),
}
GK_NUMERIC = {
    1: (1, 1.2345678901e-6, -7.654321e-17, STOP_DELTAZ_SUBH),
    2: (3, 1.2345678999e-6, -7.654000e-17, STOP_DELTAZ_SUBH * 1.000001),
}


class _ModelProxy:
    """Anything a factory reads `.store_id` and `.units` off, standing in for `ModelProxy`."""

    def __init__(self, store_id=1):
        self.store_id = store_id
        self.units = UNITS


class _Ref:
    """Anything a factory only reads `.store_id` off: a tag, a solver."""

    def __init__(self, store_id, **extra):
        self.store_id = store_id
        for key, value in extra.items():
            setattr(self, key, value)


def _exit_time(serial: int, z_exit: float) -> wavenumber_exit_time:
    payload = {"store_id": serial, "z_exit": z_exit, "compute_time": 0.0, "stepping": 0}
    for o in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
        payload[f"z_exit_suph_e{o}"] = 10.0 * z_exit
    for o in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
        payload[f"z_exit_subh_e{o}"] = z_exit / 10.0
    return wavenumber_exit_time(
        payload=payload,
        k=wavenumber(store_id=1, k_inv_Mpc=1.0e5, units=UNITS),
        cosmology=_Ref(1, units=UNITS),
        atol=tolerance(store_id=1, log10_tol=-10.0),
        rtol=tolerance(store_id=2, log10_tol=-9.0),
    )


MODEL = _ModelProxy(store_id=1)
K_EXIT = _exit_time(serial=11, z_exit=Z_EXIT)
OTHER_K_EXIT = _exit_time(serial=12, z_exit=2.0 * Z_EXIT)
TOL = {
    1: tolerance(store_id=1, log10_tol=-10.0),
    3: tolerance(store_id=3, log10_tol=-12.0),
}
RTOL = tolerance(store_id=2, log10_tol=-9.0)
Z_SOURCE_OBJ = redshift(store_id=1, z=Z_SOURCE, is_source=True, is_response=False)
Z_RESPONSES = [
    redshift(store_id=2, z=10.0, is_source=False, is_response=True),
    redshift(store_id=3, z=5.0, is_source=False, is_response=True),
]
SOLVER = _Ref(1, label="solver", stepping=0)


class Store:
    """Every table `build_schema` declares, in one SQLite database (in memory unless `url`)."""

    def __init__(self, url="sqlite://"):
        self.metadata = sqla.MetaData()
        built = build_schema(self.metadata, _factories)
        self.tables = built.tables
        self.records = built.records
        self.engine = sqla.create_engine(url, future=True)
        self.metadata.create_all(self.engine)
        self.conn = self.engine.connect()
        self.inserters = {name: self._inserter(name) for name in self.tables}

    def close(self):
        self.conn.close()
        self.engine.dispose()

    def insert(self, name: str, **row):
        table = self.tables[name]
        rec = self.records[name]
        payload = dict(row)
        if rec.get("use_timestamp", False):
            payload.setdefault("timestamp", FIXED_TIMESTAMP)
        if rec.get("use_version", False):
            payload.setdefault("version", 1)
        return self.conn.execute(sqla.insert(table), payload).lastrowid

    def _inserter(self, name):
        def inserter(conn, data):
            return self.insert(name, **data)

        return inserter


def _base_store(url="sqlite://") -> Store:
    s = Store(url)
    s.insert("version", serial=1, label="test_wkb_numeric_parent_key")
    s.insert("redshift", serial=1, z=Z_SOURCE, source=True, response=False)
    for z in Z_RESPONSES:
        s.insert("redshift", serial=z.store_id, z=z.z, source=False, response=True)
    s.insert("store_tag", serial=1, label="T")
    s.insert("IntegrationSolver", serial=1, label="solver", stepping=0)
    s.insert("tolerance", serial=1, log10_tol=-10.0)
    s.insert("tolerance", serial=2, log10_tol=-9.0)
    s.insert("tolerance", serial=3, log10_tol=-12.0)
    # the rows every declared foreign key below names, so that the store is clean under
    # PRAGMA foreign_key_check (which TestDropOrder relies on)
    s.insert("wavenumber", serial=1, k_inv_Mpc=1.0e5)
    s.insert(
        "BackgroundModel",
        serial=MODEL.store_id,
        cosmology_type=1,
        cosmology_serial=1,
        tau_gauss_order=4,
        cs_tau_gauss_order=4,
        friction_F_gauss_order=4,
        solver_serial=1,
        z_init_serial=1,
        z_samples=0,
        source_grid_digest="fixture",
        source_grid_construction="fixture",
        validated=True,
    )
    for k_exit in (K_EXIT, OTHER_K_EXIT):
        s.insert(
            "wavenumber_exit_time",
            serial=k_exit.store_id,
            wavenumber_serial=1,
            cosmology_type=1,
            cosmology_serial=1,
            atol_serial=1,
            rtol_serial=2,
            z_exit=k_exit.z_exit,
        )
    for serial, (atol, stop_T, stop_Tprime, stop_dz) in TK_NUMERIC.items():
        s.insert(
            "TkNumericIntegration",
            serial=serial,
            label=f"tknum-{serial}",
            wavenumber_exit_serial=K_EXIT.store_id,
            model_serial=MODEL.store_id,
            atol_serial=atol,
            rtol_serial=RTOL.store_id,
            break_point_kind=TkNumericIntegration.BREAK_POINT_KIND,
            solver_serial=1,
            z_init_serial=1,
            z_min_serial=3,
            z_samples=1,
            stop_deltaz_subh=stop_dz,
            stop_T=stop_T,
            stop_Tprime=stop_Tprime,
            validated=True,
        )
    for serial, (atol, stop_G, stop_Gprime, stop_dz) in GK_NUMERIC.items():
        s.insert(
            "GkNumericIntegration",
            serial=serial,
            label=f"gknum-{serial}",
            wavenumber_exit_serial=K_EXIT.store_id,
            model_serial=MODEL.store_id,
            atol_serial=atol,
            rtol_serial=RTOL.store_id,
            break_point_kind=GkNumericIntegration.BREAK_POINT_KIND,
            solver_serial=1,
            z_source_serial=Z_SOURCE_OBJ.store_id,
            z_min_serial=3,
            z_samples=1,
            stop_deltaz_subh=stop_dz,
            stop_G=stop_G,
            stop_Gprime=stop_Gprime,
            validated=True,
        )
    s.conn.commit()
    return s


def _tk_seed(numeric_serial: int):
    """(z_init, T_init, Tprime_init) exactly as main.py derives them from the stop point."""
    _, stop_T, stop_Tprime, stop_dz = TK_NUMERIC[numeric_serial]
    return K_EXIT.z_exit - stop_dz, stop_T, stop_Tprime


def _gk_seed(numeric_serial: int):
    _, stop_G, stop_Gprime, stop_dz = GK_NUMERIC[numeric_serial]
    return K_EXIT.z_exit - stop_dz, stop_G, stop_Gprime


def _call(s: Store, factory, name: str, method: str, *args):
    fn = getattr(factory, method)
    table = s.tables[name]
    if method == "build":
        return fn(args[0], s.conn, table, s.inserters[name], s.tables, s.inserters)
    if method == "store":
        return fn(args[0], s.conn, table, s.inserters[name], s.tables, s.inserters)
    if method == "validate":
        return fn(args[0], s.conn, table, s.tables)
    raise AssertionError(method)


def _computed(obj, value_cls):
    """Fill in what compute() and the target's store() would have, so that the factory's store()
    can write the object: the phase fields, the solvers, one value per response redshift.
    """
    obj._rho_gauss_order = phase_residual.RHO_GAUSS_ORDER
    obj._sin_coeff = 1.0
    obj._cos_coeff = 0.0
    obj._init_efolds_subh = 1.0
    obj._has_WKB_violation = False
    if value_cls is TkWKBValue:
        obj._phase_solver = SOLVER
        obj._friction_solver = SOLVER
        obj._values = [
            TkWKBValue(None, z, 1.0, 0.5, 3, -0.25, T_WKB=0.1) for z in Z_RESPONSES
        ]
    else:
        obj._solver = SOLVER
        obj._values = [
            GkWKBValue(None, z, 1.0, 0.5, 3, omega_WKB_sq=2.0, WKB_criterion=0.1)
            for z in Z_RESPONSES
        ]
    return obj


def _tk_payload(**extra):
    payload = {
        "solver_labels": {},
        "model": MODEL,
        "k": K_EXIT,
        "z_sample": None,
        "tags": [],
    }
    payload.update(extra)
    return payload


def _gk_payload(**extra):
    return _tk_payload(z_source=Z_SOURCE_OBJ, **extra)


def store_tk(s: Store, numeric_serial: int, **override):
    """A TkWKBIntegration seeded from `numeric_serial` the way main.py seeds it, built through the
    factory's own build() (a miss), computed by stand-in, stored and validated through the
    factory. `override` replaces an initial datum, for the consistency check's tests."""
    if numeric_serial in TK_NUMERIC:
        z_init, T_init, Tprime_init = _tk_seed(numeric_serial)
        data = dict(z_init=z_init, T_init=T_init, Tprime_init=Tprime_init)
    else:
        data = {}
    data.update(override)
    obj = _call(
        s,
        sqla_TkWKBIntegration_factory,
        "TkWKBIntegration",
        "build",
        _tk_payload(
            numeric_serial=numeric_serial,
            z_sample=redshift_array(Z_RESPONSES),
            **data,
        ),
    )
    assert not obj.available
    _computed(obj, TkWKBValue)
    _call(s, sqla_TkWKBIntegration_factory, "TkWKBIntegration", "store", obj)
    _call(s, sqla_TkWKBIntegration_factory, "TkWKBIntegration", "validate", obj)
    s.conn.commit()
    return obj


def store_gk(s: Store, numeric_serial, **override):
    """As store_tk, for a GkWKBIntegration seeded from `numeric_serial`, or pure WKB when it is
    GkWKBIntegration.PURE_WKB (main.py passes no z_init there, and G_init=0, Gprime_init=1).
    """
    if numeric_serial == GkWKBIntegration.PURE_WKB:
        data = dict(G_init=0.0, Gprime_init=1.0)
    else:
        z_init, G_init, Gprime_init = _gk_seed(numeric_serial)
        data = dict(z_init=z_init, G_init=G_init, Gprime_init=Gprime_init)
    data.update(override)
    obj = _call(
        s,
        sqla_GkWKBIntegration_factory,
        "GkWKBIntegration",
        "build",
        _gk_payload(
            numeric_serial=numeric_serial,
            z_sample=redshift_array(Z_RESPONSES),
            **data,
        ),
    )
    assert not obj.available
    _computed(obj, GkWKBValue)
    _call(s, sqla_GkWKBIntegration_factory, "GkWKBIntegration", "store", obj)
    _call(s, sqla_GkWKBIntegration_factory, "GkWKBIntegration", "validate", obj)
    s.conn.commit()
    return obj


def tk_lookup(s: Store, **extra):
    return _call(
        s,
        sqla_TkWKBIntegration_factory,
        "TkWKBIntegration",
        "build",
        _tk_payload(**extra),
    )


def gk_lookup(s: Store, **extra):
    return _call(
        s,
        sqla_GkWKBIntegration_factory,
        "GkWKBIntegration",
        "build",
        _gk_payload(**extra),
    )


class _StoreCase(unittest.TestCase):
    def setUp(self):
        self.s = _base_store()
        self.addCleanup(self.s.close)


class TestKeyedOnTheParent(_StoreCase):
    """§3 test 1."""

    def test_tk_the_first_parent_finds_it_and_the_second_misses(self):
        stored = store_tk(self.s, numeric_serial=1)

        found = tk_lookup(self.s, numeric_serial=1)
        self.assertTrue(found.available)
        self.assertEqual(found.store_id, stored.store_id)
        self.assertEqual(found.numeric_serial, 1)

        self.assertFalse(tk_lookup(self.s, numeric_serial=2).available)

    def test_tk_the_numeric_parents_differ_only_in_their_tolerance(self):
        """What makes the test above the case the prompt names: one k, two numeric rows."""
        rows = self.s.conn.execute(
            sqla.select(
                self.s.tables["TkNumericIntegration"].c.serial,
                self.s.tables["TkNumericIntegration"].c.wavenumber_exit_serial,
                self.s.tables["TkNumericIntegration"].c.atol_serial,
            )
        ).all()
        self.assertEqual({r.wavenumber_exit_serial for r in rows}, {K_EXIT.store_id})
        self.assertEqual(len({r.atol_serial for r in rows}), 2)

    def test_gk_the_first_parent_finds_it_and_the_second_misses(self):
        stored = store_gk(self.s, numeric_serial=1)

        found = gk_lookup(self.s, numeric_serial=1)
        self.assertTrue(found.available)
        self.assertEqual(found.store_id, stored.store_id)
        self.assertFalse(gk_lookup(self.s, numeric_serial=2).available)
        # nor is a seeded row served to a pure-WKB lookup
        self.assertFalse(
            gk_lookup(self.s, numeric_serial=GkWKBIntegration.PURE_WKB).available
        )

    def test_the_stored_row_records_its_parent(self):
        store_tk(self.s, numeric_serial=2)
        store_gk(self.s, numeric_serial=1)
        store_gk(self.s, numeric_serial=GkWKBIntegration.PURE_WKB)
        tk = self.s.tables["TkWKBIntegration"]
        gk = self.s.tables["GkWKBIntegration"]
        self.assertEqual(
            [2],
            [
                r.numeric_serial
                for r in self.s.conn.execute(sqla.select(tk.c.numeric_serial))
            ],
        )
        self.assertEqual(
            [1, None],
            [
                r.numeric_serial
                for r in self.s.conn.execute(
                    sqla.select(gk.c.numeric_serial).order_by(gk.c.serial)
                )
            ],
        )


class TestZInitIsPayload(_StoreCase):
    """`z_init` and the initial values left the key: a lookup keyed on the parent finds the row
    whatever initial data its payload carries, and the row reports its own."""

    def test_tk(self):
        stored = store_tk(self.s, numeric_serial=1)
        found = tk_lookup(
            self.s, numeric_serial=1, z_init=1.0, T_init=-9.0, Tprime_init=9.0
        )
        self.assertTrue(found.available)
        self.assertEqual(found.store_id, stored.store_id)
        self.assertEqual(found.z_init, _tk_seed(1)[0])
        self.assertEqual(found.T_init, _tk_seed(1)[1])

    def test_gk(self):
        stored = store_gk(self.s, numeric_serial=1)
        found = gk_lookup(
            self.s, numeric_serial=1, z_init=1.0, G_init=-9.0, Gprime_init=9.0
        )
        self.assertTrue(found.available)
        self.assertEqual(found.store_id, stored.store_id)
        self.assertEqual(found.z_init, _gk_seed(1)[0])

    def test_the_inventory_key_has_no_z_init(self):
        for factory in (sqla_TkWKBIntegration_factory, sqla_GkWKBIntegration_factory):
            with self.subTest(factory=factory.__name__):
                captured = factory.inventory_spec().keywords()
                self.assertNotIn("z_init", captured["leaves"])
                self.assertEqual(
                    captured["parents"]["numeric"].column, "numeric_serial"
                )


class TestPureWKBIsItsOwnKey(_StoreCase):
    """§3 test 2."""

    def setUp(self):
        super().setUp()
        self.seeded = store_gk(self.s, numeric_serial=1)
        self.pure = store_gk(self.s, numeric_serial=GkWKBIntegration.PURE_WKB)

    def test_both_rows_coexist_at_one_k_and_z_source(self):
        gk = self.s.tables["GkWKBIntegration"]
        rows = self.s.conn.execute(
            sqla.select(gk.c.wavenumber_exit_serial, gk.c.z_source_serial)
        ).all()
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(set(rows)), 1)

    def test_each_lookup_finds_its_own(self):
        seeded = gk_lookup(self.s, numeric_serial=1)
        pure = gk_lookup(self.s, numeric_serial=GkWKBIntegration.PURE_WKB)
        self.assertEqual(seeded.store_id, self.seeded.store_id)
        self.assertEqual(pure.store_id, self.pure.store_id)
        self.assertNotEqual(seeded.store_id, pure.store_id)
        self.assertFalse(seeded.pure_WKB)
        self.assertTrue(pure.pure_WKB)
        self.assertIsNone(pure.numeric_serial)
        self.assertEqual(pure.z_init, Z_SOURCE)
        self.assertEqual((pure.G_init, pure.Gprime_init), (0.0, 1.0))

    def test_the_value_reads_select_by_parent_or_marker(self):
        for marker, expected in (
            (1, self.seeded.store_id),
            (GkWKBIntegration.PURE_WKB, self.pure.store_id),
        ):
            with self.subTest(numeric_serial=marker):
                objs = sqla_GkWKBValue_factory.read_batch(
                    {"model": MODEL, "k": K_EXIT, "numeric_serial": marker},
                    self.s.conn,
                    self.s.tables["GkWKBValue"],
                    self.s.tables,
                )
                values = self.s.tables["GkWKBValue"]
                expected_serials = {
                    r.serial
                    for r in self.s.conn.execute(
                        sqla.select(values.c.serial).filter(
                            values.c.wkb_serial == expected
                        )
                    )
                }
                self.assertEqual({o.store_id for o in objs}, expected_serials)

                obj = sqla_GkWKBValue_factory.build(
                    {
                        "z": Z_RESPONSES[0],
                        "model": MODEL,
                        "k": K_EXIT,
                        "z_source": Z_SOURCE_OBJ,
                        "numeric_serial": marker,
                    },
                    self.s.conn,
                    self.s.tables["GkWKBValue"],
                    None,
                    self.s.tables,
                    None,
                )
                self.assertIn(obj.store_id, expected_serials)

    def test_without_a_parent_the_value_read_refuses_the_second(self):
        """05c's refusal stays: a value read naming no parent spans every parent, and two parents
        for one z_source are refused rather than mixed."""
        with self.assertRaises(MultipleResultsFound):
            sqla_GkWKBValue_factory.read_batch(
                {"model": MODEL, "k": K_EXIT},
                self.s.conn,
                self.s.tables["GkWKBValue"],
                self.s.tables,
            )


class TestAParentIsRequired(_StoreCase):
    """§3 test 3, and the Gk half the orchestrator note's §0.3 adds."""

    def test_a_tk_lookup_without_a_parent_is_an_error(self):
        with self.assertRaises(RuntimeError) as caught:
            tk_lookup(self.s)
        self.assertIn("no numeric parent", str(caught.exception))

    def test_a_tk_construction_without_a_parent_is_an_error(self):
        with self.assertRaises(ValueError) as caught:
            TkWKBIntegration(payload=None, solver_labels={}, model=MODEL, k=K_EXIT)
        self.assertIn("without a numeric parent", str(caught.exception))

    def test_a_tk_lookup_has_no_pure_wkb_form(self):
        with self.assertRaises(TypeError):
            tk_lookup(self.s, numeric_serial=GkWKBIntegration.PURE_WKB)

    def test_a_gk_lookup_given_neither_is_an_error(self):
        with self.assertRaises(RuntimeError) as caught:
            gk_lookup(self.s)
        self.assertIn("PURE_WKB", str(caught.exception))

    def test_a_parent_and_labels_only_together_are_an_error(self):
        with self.assertRaises(ValueError):
            tk_lookup(self.s, numeric_serial=1, labels_only=True)
        with self.assertRaises(ValueError):
            gk_lookup(
                self.s, numeric_serial=GkWKBIntegration.PURE_WKB, labels_only=True
            )

    def test_a_gk_construction_without_a_parent_is_pure_wkb_and_says_so(self):
        obj = GkWKBIntegration(
            payload=None, solver_labels={}, model=MODEL, k=K_EXIT, z_source=Z_SOURCE_OBJ
        )
        self.assertTrue(obj.pure_WKB)
        self.assertIsNone(obj.numeric_serial)
        seeded = GkWKBIntegration(
            payload=None,
            solver_labels={},
            model=MODEL,
            k=K_EXIT,
            z_source=Z_SOURCE_OBJ,
            numeric_serial=1,
        )
        self.assertFalse(seeded.pure_WKB)


class TestTheConsistencyCheck(_StoreCase):
    """§3 test 4, and the prompt's §5 stop: rows seeded as main.py seeds them must store."""

    def assertRefused(self, store, numeric_serial, name, stored, expected):
        with self.assertRaises(ValueError) as caught:
            store(self.s, numeric_serial, **{name: stored})
        message = str(caught.exception)
        self.assertIn(f"{name}={stored!r}", message)
        self.assertIn(repr(expected), message)

    def test_rows_seeded_as_main_py_seeds_them_store(self):
        store_tk(self.s, numeric_serial=1)
        store_gk(self.s, numeric_serial=1)
        store_gk(self.s, numeric_serial=GkWKBIntegration.PURE_WKB)
        # and what they wrote satisfies every declared foreign key, numeric_serial's included
        self.assertEqual(
            self.s.conn.execute(sqla.text("PRAGMA foreign_key_check")).all(), []
        )

    def test_the_seed_rounds(self):
        """z_init is a subtraction that rounds at this size, so a check that did not recompute it
        bit for bit could not tell one ulp from agreement."""
        z_init = _tk_seed(1)[0]
        self.assertNotEqual(nextafter(z_init, 0.0), z_init)
        self.assertGreater(z_init - nextafter(z_init, 0.0), 1.0e-7)

    def test_the_pipeline_derivation_from_the_factory_read_numeric_row(self):
        """The numeric row read back serial-only through its own factory, as main.py's Tk and Gk
        WKB stages read it (`_do_not_populate`), then `z_init = k_exit.z_exit - Tk.stop_deltaz_subh`
        with `T_init = Tk.stop_T`, `Tprime_init = Tk.stop_Tprime` (main.py's build_Tk_WKB_work),
        and the Gk twin: both store."""
        Tk = sqla_TkNumericIntegration_factory.build(
            {
                "solver_labels": [],
                "model": MODEL,
                "k": K_EXIT,
                "z_sample": None,
                "z_init": None,
                "atol": TOL[1],
                "rtol": RTOL,
                "tags": [],
                "_do_not_populate": True,
            },
            self.s.conn,
            self.s.tables["TkNumericIntegration"],
            None,
            self.s.tables,
            None,
        )
        self.assertTrue(Tk.available)
        store_tk(
            self.s,
            Tk.store_id,
            z_init=K_EXIT.z_exit - Tk.stop_deltaz_subh,
            T_init=Tk.stop_T,
            Tprime_init=Tk.stop_Tprime,
        )

        Gk = sqla_GkNumericIntegration_factory.build(
            {
                "solver_labels": [],
                "model": MODEL,
                "k": K_EXIT,
                "z_source": Z_SOURCE_OBJ,
                "z_sample": None,
                "atol": TOL[1],
                "rtol": RTOL,
                "tags": [],
                "_do_not_populate": True,
            },
            self.s.conn,
            self.s.tables["GkNumericIntegration"],
            None,
            self.s.tables,
            None,
        )
        self.assertTrue(Gk.available)
        store_gk(
            self.s,
            Gk.store_id,
            z_init=K_EXIT.z_exit - Gk.stop_deltaz_subh,
            G_init=Gk.stop_G,
            Gprime_init=Gk.stop_Gprime,
        )

    def test_tk_one_ulp_off_is_refused(self):
        z_init, T_init, Tprime_init = _tk_seed(1)
        for name, value in (
            ("z_init", z_init),
            ("T_init", T_init),
            ("Tprime_init", Tprime_init),
        ):
            with self.subTest(name=name):
                self.assertRefused(store_tk, 1, name, nextafter(value, 1.0e300), value)

    def test_gk_one_ulp_off_is_refused(self):
        z_init, G_init, Gprime_init = _gk_seed(1)
        for name, value in (
            ("z_init", z_init),
            ("G_init", G_init),
            ("Gprime_init", Gprime_init),
        ):
            with self.subTest(name=name):
                self.assertRefused(store_gk, 1, name, nextafter(value, 1.0e300), value)

    def test_pure_wkb_one_ulp_off_is_refused(self):
        with self.subTest(name="G_init"):
            with self.assertRaises(ValueError):
                store_gk(self.s, GkWKBIntegration.PURE_WKB, G_init=nextafter(0.0, 1.0))
        with self.subTest(name="Gprime_init"):
            with self.assertRaises(ValueError):
                store_gk(
                    self.s, GkWKBIntegration.PURE_WKB, Gprime_init=nextafter(1.0, 2.0)
                )
        with self.subTest(name="z_init"):
            self.assertRefused(
                store_gk,
                GkWKBIntegration.PURE_WKB,
                "z_init",
                nextafter(Z_SOURCE, 0.0),
                Z_SOURCE,
            )

    def test_the_other_parents_stop_values_are_refused(self):
        """A row carrying parent 2's initial data but naming parent 1."""
        z_init, T_init, Tprime_init = _tk_seed(2)
        with self.assertRaises(ValueError):
            store_tk(self.s, 1, z_init=z_init, T_init=T_init, Tprime_init=Tprime_init)

    def test_a_parent_for_another_k_is_refused(self):
        self.s.conn.execute(
            sqla.update(self.s.tables["TkNumericIntegration"])
            .where(self.s.tables["TkNumericIntegration"].c.serial == 1)
            .values(wavenumber_exit_serial=OTHER_K_EXIT.store_id)
        )
        with self.assertRaises(RuntimeError) as caught:
            store_tk(self.s, 1)
        self.assertIn("wavenumber exit serial", str(caught.exception))

    def test_a_parent_for_another_z_source_is_refused(self):
        self.s.conn.execute(
            sqla.update(self.s.tables["GkNumericIntegration"])
            .where(self.s.tables["GkNumericIntegration"].c.serial == 1)
            .values(z_source_serial=2)
        )
        with self.assertRaises(RuntimeError) as caught:
            store_gk(self.s, 1)
        self.assertIn("z_source serial", str(caught.exception))

    def test_a_parent_not_on_this_shard_is_refused(self):
        with self.assertRaises(RuntimeError) as caught:
            z_init, T_init, Tprime_init = _tk_seed(1)
            store_tk(self.s, 99, z_init=z_init, T_init=T_init, Tprime_init=Tprime_init)
        self.assertIn(
            "no TkNumericIntegration with that serial is on this shard",
            str(caught.exception),
        )

    def test_nothing_is_written_when_refused(self):
        with self.assertRaises(ValueError):
            store_tk(self.s, 1, T_init=nextafter(_tk_seed(1)[1], 1.0))
        count = self.s.conn.execute(
            sqla.select(sqla.func.count()).select_from(
                self.s.tables["TkWKBIntegration"]
            )
        ).scalar()
        self.assertEqual(count, 0)


class TestReadersRefuseAmbiguity(_StoreCase):
    """§3 test 5: the labels-only mode the extraction scripts use."""

    def test_tk_two_parents_are_refused_and_named(self):
        store_tk(self.s, numeric_serial=1)
        store_tk(self.s, numeric_serial=2)
        with self.assertRaises(MultipleResultsFound) as caught:
            tk_lookup(self.s, labels_only=True)
        self.assertIn("numeric_serial values [1, 2]", str(caught.exception))

    def test_gk_a_seeded_and_a_pure_row_are_refused_and_named(self):
        store_gk(self.s, numeric_serial=1)
        store_gk(self.s, numeric_serial=GkWKBIntegration.PURE_WKB)
        with self.assertRaises(MultipleResultsFound) as caught:
            gk_lookup(self.s, labels_only=True)
        self.assertIn("numeric_serial values ['1', 'pure WKB']", str(caught.exception))

    def test_gk_two_seeded_rows_are_refused_and_named(self):
        store_gk(self.s, numeric_serial=1)
        store_gk(self.s, numeric_serial=2)
        with self.assertRaises(MultipleResultsFound) as caught:
            gk_lookup(self.s, labels_only=True)
        self.assertIn("numeric_serial values ['1', '2']", str(caught.exception))

    def test_one_row_is_served(self):
        tk = store_tk(self.s, numeric_serial=2)
        gk = store_gk(self.s, numeric_serial=GkWKBIntegration.PURE_WKB)
        found_tk = tk_lookup(self.s, labels_only=True)
        found_gk = gk_lookup(self.s, labels_only=True)
        self.assertEqual(found_tk.store_id, tk.store_id)
        self.assertEqual(found_tk.numeric_serial, 2)
        self.assertEqual(found_gk.store_id, gk.store_id)
        self.assertTrue(found_gk.pure_WKB)

    def test_a_miss_reports_absence_and_cannot_be_computed(self):
        tk = tk_lookup(self.s, labels_only=True)
        gk = gk_lookup(self.s, labels_only=True)
        self.assertFalse(tk.available)
        self.assertFalse(gk.available)
        self.assertIsNone(tk.numeric_serial)
        self.assertFalse(gk.pure_WKB)
        with self.assertRaises(RuntimeError):
            tk.compute()


class TestDropOrder(unittest.TestCase):
    """§3 test 7. The real `Datastore._drop_tables`, on a file store with foreign-key enforcement
    turned on for every connection its engine opens (production never turns it on; SQLite's DROP
    TABLE performs an implicit DELETE, which enforcement checks). The store holds the Tk numeric rows
    and one Tk WKB row naming the first, written raw with no value or tag rows, and is
    clean under `PRAGMA foreign_key_check` before the drop: SQLite counts violations net, so a
    row whose own references already dangled would hide the one the drop makes. The order is
    derived from the schema's foreign keys (prompts/datastore-generic, prompt 06)."""

    def _drop(self, actions):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        url = f"sqlite:///{Path(tmp.name) / 'drop.sqlite'}"
        s = _base_store(url)
        self.addCleanup(s.close)
        z_init, T_init, Tprime_init = _tk_seed(1)
        s.insert(
            "TkWKBIntegration",
            serial=1,
            label="tkwkb-1",
            wavenumber_exit_serial=K_EXIT.store_id,
            model_serial=MODEL.store_id,
            numeric_serial=1,
            rho_gauss_order=phase_residual.RHO_GAUSS_ORDER,
            phase_solver_serial=1,
            friction_solver_serial=1,
            z_min_serial=3,
            z_samples=0,
            z_init=z_init,
            sin_coeff=1.0,
            cos_coeff=0.0,
            T_init=T_init,
            Tprime_init=Tprime_init,
            validated=True,
        )
        s.conn.commit()
        self.assertEqual(
            s.conn.execute(sqla.text("PRAGMA foreign_key_check")).all(), []
        )
        s.conn.close()

        @sqla.event.listens_for(s.engine, "connect")
        def _enforce(dbapi_connection, connection_record):
            dbapi_connection.execute("PRAGMA foreign_keys = ON")

        s.engine.dispose()
        fake = types.SimpleNamespace(
            _inspector=sqla.inspect(s.engine),
            _tables=s.tables,
            _engine=s.engine,
            _metadata=s.metadata,
            _my_name="drop-order-test",
        )
        raw = datastore_module.Datastore.__ray_metadata__.modified_class
        with mock.patch("builtins.print"):
            raw._drop_tables(fake, tables_to_drop(actions))
        with s.engine.connect() as conn:
            self.assertEqual(conn.execute(sqla.text("PRAGMA foreign_keys")).scalar(), 1)
        return sqla.inspect(s.engine)

    def test_the_derived_order_drops_both(self):
        for actions in (["tk-numeric", "tk-wkb"], ["tk-wkb", "tk-numeric"]):
            with self.subTest(actions=actions):
                inspector = self._drop(actions)
                self.assertFalse(inspector.has_table("TkWKBIntegration"))
                self.assertFalse(inspector.has_table("TkNumericIntegration"))

    def test_an_unsorted_order_is_refused_by_enforcement(self):
        # pins that the actor sorts: with the actor module's drop_order the identity, the tables
        # drop in the order named, so tk-numeric's TkNumericIntegration goes while the
        # TkWKBIntegration row still references it. (Named the other way round, the identity
        # happens to be a valid order here: the store holds no value or tag rows.)
        with mock.patch.object(datastore_module, "drop_order", lambda tables: tables):
            with self.assertRaises(IntegrityError):
                self._drop(["tk-numeric", "tk-wkb"])

    def test_tk_wkb_drops_before_tk_numeric_and_gk_wkb_before_gk_numeric(self):
        tables = build_schema(sqla.MetaData(), _factories).tables
        order = [
            t.name
            for t in drop_order([tables[n] for n in tables_to_drop(list(drop_groups))])
        ]
        self.assertLess(
            order.index("TkWKBIntegration"), order.index("TkNumericIntegration")
        )
        self.assertLess(
            order.index("GkWKBIntegration"), order.index("GkNumericIntegration")
        )

    def test_every_declared_foreign_key_between_dropped_tables_is_respected(self):
        """Derived from the schema: a table with a foreign key into another dropped table drops
        first, for every drop group alone and for all of them together."""
        tables = build_schema(sqla.MetaData(), _factories).tables
        cases = {action: [action] for action in drop_groups}
        cases["all"] = list(drop_groups)
        for case, actions in cases.items():
            names = tables_to_drop(actions)
            order = [t.name for t in drop_order([tables[n] for n in names])]
            self.assertEqual(sorted(order), sorted(names))
            pairs = 0
            for name in names:
                for fk in tables[name].foreign_keys:
                    target = fk.column.table.name
                    if target == name or target not in names:
                        continue
                    pairs += 1
                    with self.subTest(case=case, table=name, references=target):
                        self.assertLess(order.index(name), order.index(target))
            if case == "all":
                # the WKB->numeric pairs and QuadSource.Tq_serial at least
                self.assertGreaterEqual(pairs, 3)


class TestNullableParent(unittest.TestCase):
    """The orchestrator note's §0.5: `Parent`'s nullable form gives a NULL column the key field
    None; a parent that is not nullable still reports a NULL as an unresolved parent."""

    def _read(self, parent):
        metadata = sqla.MetaData()
        table = sqla.Table(
            "Child",
            metadata,
            sqla.Column("serial", sqla.Integer, primary_key=True),
            sqla.Column("numeric_serial", sqla.Integer, nullable=True),
        )
        engine = sqla.create_engine("sqlite://", future=True)
        metadata.create_all(engine)
        with engine.begin() as conn:
            conn.execute(sqla.insert(table), [{"serial": 1, "numeric_serial": 7}])
            conn.execute(sqla.insert(table), [{"serial": 2, "numeric_serial": None}])
            context = ShardContext(
                shard=0,
                keys={},
                digests={"Parent": {7: "digest-of-7"}},
            )
            out = read_records(conn, table, {}, context, parents={"numeric": parent})
        engine.dispose()
        return out

    def test_nullable(self):
        read = self._read(Parent("numeric_serial", "Parent", nullable=True))
        self.assertEqual(read.problems, [])
        keys = {serial: record.key for serial, record, _ in read.rows}
        self.assertEqual(keys, {1: {"numeric": "digest-of-7"}, 2: {"numeric": None}})

    def test_not_nullable(self):
        read = self._read(Parent("numeric_serial", "Parent"))
        self.assertEqual([serial for serial, _, _ in read.rows], [1])
        self.assertEqual(len(read.problems), 1)
        self.assertIn("unresolved-parent", read.problems[0])


def tearDownModule():
    assert not ray.is_initialized()


if __name__ == "__main__":
    unittest.main()
