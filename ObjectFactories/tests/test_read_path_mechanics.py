"""
Round-trip guard for the mechanical read-path defects prompts/datastore-integrity prompt 05a
fixed (docs/datastore-integrity-audit.md §3.10-§3.11, board items P6/P7):

  * two wrong attribute names -- ``row_data.WKB_criterio`` (missing the trailing "n") in
    ``sqla_GkWKBValue_factory._build_impl_serial`` and ``sqla_TkWKBValue_factory._build_impl_serial``;
  * two wrong column names -- ``table.c.z_serial`` in ``sqla_GkSourceValue_factory._build_impl_serial``,
    where the column is ``z_source_serial``, and ``quadsource_table.c.model_id`` in
    ``sqla_QuadSourceValue_factory._build_impl_model``, where the column is ``model_serial``;
  * a missing argument -- ``sqla_TkWKBValue_factory.read_batch`` selects ``friction`` and never
    passes it to ``TkWKBValue(...)``, which requires it;
  * tags defaulting to ``None`` at nine sites, so an omitted ``tags`` key raised ``TypeError`` where
    ``for tag in tags`` ran;
  * three ``raise ("...")`` sites, which raised ``TypeError: exceptions must derive from
    BaseException`` instead of the message they carry.

``ObjectFactories/tests/test_factory_select_columns.py`` is a static guard: it can see the two wrong
names and the two wrong column names (they are read-back defects and column-existence defects
respectively), but by its own module docstring (point 3) it "cannot see ... a correctly selected
column never passed on to the object it builds" -- ``TkWKBValue.read_batch``'s missing ``friction``
-- nor whether an omitted key crashes a loop, nor what a ``raise`` actually raises. Those need a
round trip that executes the real factory code, which is what this module is.

No Ray, no datastore file, nothing under ``var/``: the schema is built in memory with
``build_schema(MetaData(), config.datastore.factories)`` -- the same construction
``docs/datastore-integrity-audit/lookup_keys/common.py`` uses for the audit's probes, and what
prompt 05a §3 names as the pattern to follow -- so every table the real factories declare exists,
and every call below is to the real, unmodified ``build()`` / ``read_batch`` / ``_build_impl_*``
code.

SQLite does not enforce foreign keys unless ``PRAGMA foreign_keys`` is turned on, and the
production ``Datastore`` does not turn it on (see ``test_backgroundmodelvalue_roundtrip.py``), so
parent rows that exist only as a bare integer serial -- a ``model``, a wavenumber exit time, a
tag -- are stand-ins carrying only the ``.store_id`` a factory reads off them, exactly as
``docs/datastore-integrity-audit/lookup_keys/common.py``'s ``Ref`` and ``StandInModelProxy`` do.
"""

import unittest
from datetime import datetime

import sqlalchemy as sqla

import ComputeTargets.phase_residual as phase_residual
from CosmologyConcepts import redshift
from config.datastore import factories as _factories
from ObjectFactories.GkNumericIntegration import (
    sqla_GkNumericValue_factory,
)
from ObjectFactories.GkSource import sqla_GkSourceValue_factory
from ObjectFactories.GkWKBIntegration import sqla_GkWKBValue_factory
from ObjectFactories.QuadSource import sqla_QuadSourceValue_factory
from ObjectFactories.TkNumericIntegration import (
    sqla_TkNumericValue_factory,
)
from ObjectFactories.TkWKBIntegration import sqla_TkWKBValue_factory
from datastorekit.SQL.schema import build_schema
from Units import Mpc_units

UNITS = Mpc_units()
FIXED_TIMESTAMP = datetime(2026, 9, 28, 12, 0, 0)


class _ModelProxy:
    """Anything a factory reads ``.store_id`` and ``.units`` off, standing in for ``ModelProxy``."""

    def __init__(self, store_id=1):
        self.store_id = store_id
        self.units = UNITS


class _Ref:
    """Anything a factory only reads ``.store_id`` off: a wavenumber exit time, a tolerance, a tag."""

    def __init__(self, store_id):
        self.store_id = store_id


def _z(serial, value, is_source=False, is_response=False):
    return redshift(
        store_id=serial, z=value, is_source=is_source, is_response=is_response
    )


class Store:
    """
    Every table ``build_schema`` declares, in one in-memory SQLite database. No Ray, no file,
    nothing under ``var/``: ``build_schema`` only declares ``sqla.Table`` objects, and
    ``create_all`` makes an in-memory schema from them.
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

    def insert(self, name: str, **row):
        table = self.tables[name]
        rec = self.records[name]
        payload = dict(row)
        if rec.get("use_timestamp", False):
            payload.setdefault("timestamp", FIXED_TIMESTAMP)
        if rec.get("use_version", False):
            payload.setdefault("version", 1)
        return self.conn.execute(sqla.insert(table), payload).lastrowid


def _base_store() -> Store:
    """
    The metadata rows every test below shares: a version row (required by nothing here directly,
    but present in every real store), two redshifts distinguishable by serial and value, and one
    tag.
    """
    s = Store()
    s.insert("version", serial=1, label="test_read_path_mechanics")
    s.insert("redshift", serial=1, z=1.0e5, source=True, response=False)
    s.insert("redshift", serial=2, z=10.0, source=False, response=True)
    s.insert("store_tag", serial=1, label="T")
    s.conn.commit()
    return s


MODEL = _ModelProxy(store_id=1)
K_EXIT = _Ref(store_id=11)
TAG = _Ref(store_id=1)


def _gk_wkb_parent(s: Store, serial: int, WKB_criterion: float, tagged: bool) -> None:
    """A validated GkWKBIntegration row (model=1, k=11, z_source=serial 1) and one value row at
    z=serial 2, carrying a WKB_criterion distinct from every other numeric column on the row.
    """
    s.insert(
        "GkWKBIntegration",
        serial=serial,
        label=f"gkwkb-{serial}",
        wavenumber_exit_serial=K_EXIT.store_id,
        model_serial=MODEL.store_id,
        rho_gauss_order=phase_residual.RHO_GAUSS_ORDER,
        solver_serial=1,
        z_source_serial=1,
        z_min_serial=2,
        z_samples=1,
        z_init=1.0e4,
        sin_coeff=0.0,
        cos_coeff=1.0,
        G_init=0.0,
        Gprime_init=1.0,
        validated=True,
    )
    if tagged:
        s.insert("GkWKB_tags", wkb_serial=serial, tag_serial=TAG.store_id)
    s.insert(
        "GkWKBValue",
        serial=10 * serial,
        wkb_serial=serial,
        z_serial=2,
        H_ratio=1.0,
        theta_mod_2pi=0.5,
        theta_div_2pi=3,
        omega_WKB_sq=2.0,
        WKB_criterion=WKB_criterion,
        G_WKB=3.0,
    )


def _tk_wkb_parent(
    s: Store, serial: int, friction: float, WKB_criterion: float, tagged: bool
) -> None:
    """A validated TkWKBIntegration row (model=1, k=11) and one value row at z=serial 2, carrying
    a friction and a WKB_criterion each distinct from every other numeric column on the row.
    """
    s.insert(
        "TkWKBIntegration",
        serial=serial,
        label=f"tkwkb-{serial}",
        wavenumber_exit_serial=K_EXIT.store_id,
        model_serial=MODEL.store_id,
        # the numeric parent (NOT NULL since prompts/datastore-integrity prompt 07): serial 1,
        # which _tk_numeric_parent writes where a test has one; foreign keys are not enforced here
        numeric_serial=1,
        rho_gauss_order=phase_residual.RHO_GAUSS_ORDER,
        phase_solver_serial=1,
        friction_solver_serial=1,
        z_min_serial=2,
        z_samples=1,
        z_init=1.0e4,
        sin_coeff=0.0,
        cos_coeff=1.0,
        T_init=1.0,
        Tprime_init=0.0,
        validated=True,
    )
    if tagged:
        s.insert("TkWKB_tags", wkb_serial=serial, tag_serial=TAG.store_id)
    s.insert(
        "TkWKBValue",
        serial=10 * serial,
        wkb_serial=serial,
        z_serial=2,
        H_ratio=1.0,
        theta_mod_2pi=0.5,
        theta_div_2pi=3,
        friction=friction,
        omega_WKB_sq=2.0,
        WKB_criterion=WKB_criterion,
        T_WKB=4.0,
    )


def _gk_numeric_parent(s: Store, serial: int, tagged: bool) -> None:
    s.insert(
        "GkNumericIntegration",
        serial=serial,
        label=f"gknum-{serial}",
        wavenumber_exit_serial=K_EXIT.store_id,
        model_serial=MODEL.store_id,
        atol_serial=1,
        rtol_serial=1,
        break_point_kind="discontinuity",
        solver_serial=1,
        z_source_serial=1,
        z_min_serial=2,
        z_samples=1,
        validated=True,
    )
    if tagged:
        s.insert("GkNumeric_tags", integration_serial=serial, tag_serial=TAG.store_id)
    s.insert(
        "GkNumericValue",
        serial=10 * serial,
        integration_serial=serial,
        z_serial=2,
        G=float(serial),
        Gprime=0.0,
    )


def _tk_numeric_parent(s: Store, serial: int, tagged: bool) -> None:
    s.insert(
        "TkNumericIntegration",
        serial=serial,
        label=f"tknum-{serial}",
        wavenumber_exit_serial=K_EXIT.store_id,
        model_serial=MODEL.store_id,
        atol_serial=1,
        rtol_serial=1,
        break_point_kind="all",
        solver_serial=1,
        z_init_serial=1,
        z_min_serial=2,
        z_samples=1,
        validated=True,
    )
    if tagged:
        s.insert("TkNumeric_tags", integration_serial=serial, tag_serial=TAG.store_id)
    s.insert(
        "TkNumericValue",
        serial=10 * serial,
        integration_serial=serial,
        z_serial=2,
        T=float(serial),
        Tprime=0.0,
    )


def _quad_source_parent(s: Store, serial: int, model_serial: int, tagged: bool) -> None:
    """A validated QuadSource row and one QuadSourceValue row at z=serial 2."""
    s.insert(
        "QuadSource",
        serial=serial,
        label=f"quad-{serial}",
        model_serial=model_serial,
        q_wavenumber_exit_serial=11,
        r_wavenumber_exit_serial=12,
        Tq_serial=serial,
        Tr_serial=serial,
        z_samples=1,
        validated=True,
    )
    if tagged:
        s.insert("QuadSource_tags", parent_serial=serial, tag_serial=TAG.store_id)
    s.insert(
        "QuadSourceValue",
        serial=10 * serial,
        parent_serial=serial,
        z_serial=2,
        source=1.0,
        undiff=2.0,
        diff=3.0,
    )


class TestWKBCriterionSpelling(unittest.TestCase):
    """
    §3 item 1. `WKB_criterio` (missing the trailing "n") raised AttributeError on every hit of
    `_build_impl_serial`, for both GkWKBValue and TkWKBValue -- the defect KNOWN_UNFIXED carried
    until this prompt struck it.
    """

    def setUp(self):
        self.s = _base_store()
        self.addCleanup(self.s.close)

    def test_gk_wkb_value_build_by_serial_returns_WKB_criterion(self):
        _gk_wkb_parent(self.s, serial=1, WKB_criterion=0.123456789, tagged=False)
        self.s.conn.commit()

        obj = sqla_GkWKBValue_factory.build(
            {"z": _z(2, 10.0), "wkb_serial": 1},
            self.s.conn,
            self.s.tables["GkWKBValue"],
            None,
            self.s.tables,
            None,
        )
        self.assertEqual(0.123456789, obj.WKB_criterion)

    def test_tk_wkb_value_build_by_serial_returns_WKB_criterion(self):
        _tk_wkb_parent(
            self.s, serial=1, friction=0.5, WKB_criterion=0.987654321, tagged=False
        )
        self.s.conn.commit()

        obj = sqla_TkWKBValue_factory.build(
            {"z": _z(2, 10.0), "wkb_serial": 1},
            self.s.conn,
            self.s.tables["TkWKBValue"],
            None,
            self.s.tables,
            None,
        )
        self.assertEqual(0.987654321, obj.WKB_criterion)


class TestGkSourceValueZSourceSerial(unittest.TestCase):
    """
    §3 item 2. `table.c.z_serial` -- a column GkSourceValue does not have -- raised AttributeError
    at expression-build time on every call, before any row was even looked up. The fix,
    `z_source_serial`, is exercised here by two rows that differ only in z_source: a query at one
    must return that row and not the other.
    """

    def setUp(self):
        self.s = _base_store()
        self.addCleanup(self.s.close)
        self.s.insert(
            "GkSourceValue",
            serial=101,
            parent_serial=5,
            z_source_serial=1,
            G=11.0,
            Gprime=0.0,
        )
        self.s.insert(
            "GkSourceValue",
            serial=102,
            parent_serial=5,
            z_source_serial=2,
            G=22.0,
            Gprime=0.0,
        )
        self.s.conn.commit()

    def test_build_hits_the_row_at_its_own_z_source(self):
        obj = sqla_GkSourceValue_factory.build(
            {"z_source": _z(1, 1.0e5), "parent_serial": 5},
            self.s.conn,
            self.s.tables["GkSourceValue"],
            None,
            self.s.tables,
            None,
        )
        self.assertEqual(11.0, obj.numeric.G)

    def test_build_at_the_other_z_source_misses_the_first_row(self):
        obj = sqla_GkSourceValue_factory.build(
            {"z_source": _z(2, 10.0), "parent_serial": 5},
            self.s.conn,
            self.s.tables["GkSourceValue"],
            None,
            self.s.tables,
            None,
        )
        self.assertEqual(22.0, obj.numeric.G)


class TestQuadSourceValueModelSerial(unittest.TestCase):
    """
    §3 item 3. `quadsource_table.c.model_id` -- QuadSource has `model_serial`, not `model_id` --
    raised AttributeError at expression-build time on every call to `_build_impl_model`. Fixed, a
    query by model hits the row stored under that model and misses one stored under another.
    """

    def setUp(self):
        self.s = _base_store()
        self.addCleanup(self.s.close)
        _quad_source_parent(self.s, serial=1, model_serial=1, tagged=False)
        _quad_source_parent(self.s, serial=2, model_serial=2, tagged=False)
        self.s.conn.commit()

    def test_build_hits_the_row_under_its_own_model(self):
        obj = sqla_QuadSourceValue_factory.build(
            {
                "z": _z(2, 10.0),
                "model": _ModelProxy(store_id=1),
                "q": _Ref(11),
                "r": _Ref(12),
                "tags": [],
            },
            self.s.conn,
            self.s.tables["QuadSourceValue"],
            None,
            self.s.tables,
            None,
        )
        self.assertEqual(1.0, obj.source)
        self.assertEqual(2.0, obj.undiff)
        self.assertEqual(3.0, obj.diff)

    def test_build_under_another_model_misses(self):
        obj = sqla_QuadSourceValue_factory.build(
            {
                "z": _z(2, 10.0),
                "model": _ModelProxy(store_id=3),
                "q": _Ref(11),
                "r": _Ref(12),
                "tags": [],
            },
            self.s.conn,
            self.s.tables["QuadSourceValue"],
            None,
            self.s.tables,
            None,
        )
        self.assertFalse(obj.available)


class TestTkWKBValueFriction(unittest.TestCase):
    """
    §3 item 4. `read_batch` selected `friction` but never passed it to `TkWKBValue(...)`, which
    requires it: every call raised `TypeError: TkWKBValue.__init__() missing 1 required positional
    argument: 'friction'`. This is P9's round trip for this read_batch (05c adds its parent-key
    cases). Checked, per the prompt: `TkWKBValue`'s constructor takes no other required argument
    that `make_obj` omits -- `store_id`, `z`, `H_ratio`, `theta_mod_2pi`, `theta_div_2pi` and
    `friction` are the five with no default, and the first four were already supplied.
    """

    def setUp(self):
        self.s = _base_store()
        self.addCleanup(self.s.close)
        _tk_wkb_parent(self.s, serial=1, friction=0.25, WKB_criterion=0.1, tagged=True)
        _tk_wkb_parent(self.s, serial=2, friction=0.77, WKB_criterion=0.2, tagged=False)
        self.s.conn.commit()

    def test_read_batch_untagged_returns_every_row_with_its_own_friction(self):
        # its own store, holding one parent: an untagged read over the class's two parents is a
        # two-parent mixture, which prompt 05c's refusal removes under D1 (edit named by the user's
        # decision of 2026-09-28; the two-parent case is test_value_reads_parent_key.py's)
        s = _base_store()
        self.addCleanup(s.close)
        _tk_wkb_parent(s, serial=2, friction=0.77, WKB_criterion=0.2, tagged=False)
        s.conn.commit()

        objs = sqla_TkWKBValue_factory.read_batch(
            {"model": MODEL, "k": K_EXIT, "tags": []},
            s.conn,
            s.tables["TkWKBValue"],
            s.tables,
        )
        frictions = sorted(obj.friction for obj in objs)
        self.assertEqual([0.77], frictions)

    def test_read_batch_tagged_returns_only_the_tagged_row_with_its_friction(self):
        objs = sqla_TkWKBValue_factory.read_batch(
            {"model": MODEL, "k": K_EXIT, "tags": [TAG]},
            self.s.conn,
            self.s.tables["TkWKBValue"],
            self.s.tables,
        )
        objs = list(objs)
        self.assertEqual(1, len(objs))
        self.assertEqual(0.25, objs[0].friction)


class TestTagsOmittedDefaultToEmpty(unittest.TestCase):
    """
    §3 item 5. `payload.get("tags", None)` followed by `for tag in tags` raised `TypeError:
    'NoneType' object is not iterable` at every one of nine sites the moment `tags` was left out
    of the payload. Fixed to default to `[]`, omitting `tags` must behave exactly as passing
    `tags=[]` -- checked here by calling each site both ways against the same store and comparing
    every attribute the round trip below already exercises.
    """

    def setUp(self):
        self.s = _base_store()
        self.addCleanup(self.s.close)
        _gk_wkb_parent(self.s, serial=1, WKB_criterion=0.11, tagged=False)
        _tk_wkb_parent(
            self.s, serial=1, friction=0.33, WKB_criterion=0.22, tagged=False
        )
        _gk_numeric_parent(self.s, serial=1, tagged=False)
        _tk_numeric_parent(self.s, serial=1, tagged=False)
        _quad_source_parent(self.s, serial=1, model_serial=1, tagged=False)
        self.s.insert(
            "GkSourceValue",
            serial=101,
            parent_serial=5,
            z_source_serial=1,
            G=11.0,
            Gprime=0.0,
        )
        self.s.conn.commit()

    def test_gk_numeric_value_build_impl_model(self):
        payload = {
            "z": _z(2, 10.0),
            "model": MODEL,
            "k": K_EXIT,
            "z_source": _z(1, 1.0e5),
            "atol": _Ref(1),
            "rtol": _Ref(1),
        }
        omitted = sqla_GkNumericValue_factory.build(
            payload,
            self.s.conn,
            self.s.tables["GkNumericValue"],
            None,
            self.s.tables,
            None,
        )
        explicit = sqla_GkNumericValue_factory.build(
            dict(payload, tags=[]),
            self.s.conn,
            self.s.tables["GkNumericValue"],
            None,
            self.s.tables,
            None,
        )
        self.assertEqual(explicit.G, omitted.G)
        self.assertEqual(explicit.store_id, omitted.store_id)

    def test_gk_numeric_value_read_batch(self):
        payload = {"model": MODEL, "k": K_EXIT, "atol": _Ref(1), "rtol": _Ref(1)}
        omitted = list(
            sqla_GkNumericValue_factory.read_batch(
                payload, self.s.conn, self.s.tables["GkNumericValue"], self.s.tables
            )
        )
        explicit = list(
            sqla_GkNumericValue_factory.read_batch(
                dict(payload, tags=[]),
                self.s.conn,
                self.s.tables["GkNumericValue"],
                self.s.tables,
            )
        )
        self.assertEqual([o.G for o in explicit], [o.G for o in omitted])

    def test_tk_numeric_value_build_impl_model(self):
        # build()'s own dispatch reads "z_init"; _build_impl_model then reads "z_source" off the
        # same payload (ObjectFactories/TkNumericIntegration.py:853) -- both keys
        # must carry the same redshift for the model path to be reached at all, independent of
        # anything this prompt fixes.
        z_source = _z(1, 1.0e5)
        payload = {
            "z": _z(2, 10.0),
            "model": MODEL,
            "k": K_EXIT,
            "z_init": z_source,
            "z_source": z_source,
            "atol": _Ref(1),
            "rtol": _Ref(1),
        }
        omitted = sqla_TkNumericValue_factory.build(
            payload,
            self.s.conn,
            self.s.tables["TkNumericValue"],
            None,
            self.s.tables,
            None,
        )
        explicit = sqla_TkNumericValue_factory.build(
            dict(payload, tags=[]),
            self.s.conn,
            self.s.tables["TkNumericValue"],
            None,
            self.s.tables,
            None,
        )
        self.assertEqual(explicit.T, omitted.T)
        self.assertEqual(explicit.store_id, omitted.store_id)

    def test_gk_source_value_build_impl_model(self):
        payload = {
            "z_source": _z(1, 1.0e5),
            "model": MODEL,
            "k": K_EXIT,
            "z_response": _z(2, 10.0),
        }
        omitted = sqla_GkSourceValue_factory.build(
            payload,
            self.s.conn,
            self.s.tables["GkSourceValue"],
            None,
            self.s.tables,
            None,
        )
        explicit = sqla_GkSourceValue_factory.build(
            dict(payload, tags=[]),
            self.s.conn,
            self.s.tables["GkSourceValue"],
            None,
            self.s.tables,
            None,
        )
        self.assertEqual(explicit.available, omitted.available)

    def test_gk_wkb_value_build_impl_model(self):
        payload = {
            "z": _z(2, 10.0),
            "model": MODEL,
            "k": K_EXIT,
            "z_source": _z(1, 1.0e5),
        }
        omitted = sqla_GkWKBValue_factory.build(
            payload,
            self.s.conn,
            self.s.tables["GkWKBValue"],
            None,
            self.s.tables,
            None,
        )
        explicit = sqla_GkWKBValue_factory.build(
            dict(payload, tags=[]),
            self.s.conn,
            self.s.tables["GkWKBValue"],
            None,
            self.s.tables,
            None,
        )
        self.assertEqual(explicit.store_id, omitted.store_id)
        self.assertEqual(explicit.WKB_criterion, omitted.WKB_criterion)

    def test_gk_wkb_value_read_batch(self):
        payload = {"model": MODEL, "k": K_EXIT}
        omitted = list(
            sqla_GkWKBValue_factory.read_batch(
                payload, self.s.conn, self.s.tables["GkWKBValue"], self.s.tables
            )
        )
        explicit = list(
            sqla_GkWKBValue_factory.read_batch(
                dict(payload, tags=[]),
                self.s.conn,
                self.s.tables["GkWKBValue"],
                self.s.tables,
            )
        )
        self.assertEqual(
            [o.WKB_criterion for o in explicit], [o.WKB_criterion for o in omitted]
        )

    def test_tk_wkb_value_build_impl_model(self):
        payload = {"z": _z(2, 10.0), "model": MODEL, "k": K_EXIT}
        omitted = sqla_TkWKBValue_factory.build(
            payload,
            self.s.conn,
            self.s.tables["TkWKBValue"],
            None,
            self.s.tables,
            None,
        )
        explicit = sqla_TkWKBValue_factory.build(
            dict(payload, tags=[]),
            self.s.conn,
            self.s.tables["TkWKBValue"],
            None,
            self.s.tables,
            None,
        )
        self.assertEqual(explicit.store_id, omitted.store_id)
        self.assertEqual(explicit.friction, omitted.friction)

    def test_tk_wkb_value_read_batch(self):
        payload = {"model": MODEL, "k": K_EXIT}
        omitted = list(
            sqla_TkWKBValue_factory.read_batch(
                payload, self.s.conn, self.s.tables["TkWKBValue"], self.s.tables
            )
        )
        explicit = list(
            sqla_TkWKBValue_factory.read_batch(
                dict(payload, tags=[]),
                self.s.conn,
                self.s.tables["TkWKBValue"],
                self.s.tables,
            )
        )
        self.assertEqual([o.friction for o in explicit], [o.friction for o in omitted])

    def test_quad_source_value_build_impl_model(self):
        payload = {
            "z": _z(2, 10.0),
            "model": _ModelProxy(store_id=1),
            "q": _Ref(11),
            "r": _Ref(12),
        }
        omitted = sqla_QuadSourceValue_factory.build(
            payload,
            self.s.conn,
            self.s.tables["QuadSourceValue"],
            None,
            self.s.tables,
            None,
        )
        explicit = sqla_QuadSourceValue_factory.build(
            dict(payload, tags=[]),
            self.s.conn,
            self.s.tables["QuadSourceValue"],
            None,
            self.s.tables,
            None,
        )
        self.assertEqual(explicit.store_id, omitted.store_id)
        self.assertEqual(explicit.source, omitted.source)


class TestStringRaisesBecomeRuntimeError(unittest.TestCase):
    """
    §3 item 6. `raise ("...")` is not a raise of an exception carrying that string as its message:
    it is `raise` of a parenthesised string literal, and Python refuses it --
    `TypeError: exceptions must derive from BaseException` -- discarding the message entirely.
    Fixed to `raise RuntimeError("...")`, each of the three sites now raises RuntimeError carrying
    its message, reached with no stored row and no data payload.
    """

    def setUp(self):
        self.s = _base_store()
        self.addCleanup(self.s.close)

    def test_gk_wkb_value_build_impl_serial(self):
        with self.assertRaises(RuntimeError) as caught:
            sqla_GkWKBValue_factory.build(
                {"z": _z(2, 10.0), "wkb_serial": 999},
                self.s.conn,
                self.s.tables["GkWKBValue"],
                None,
                self.s.tables,
                None,
            )
        self.assertIn("GkWKBValue", str(caught.exception))
        self.assertIn("result was not found", str(caught.exception))

    def test_tk_wkb_value_build_impl_serial(self):
        with self.assertRaises(RuntimeError) as caught:
            sqla_TkWKBValue_factory.build(
                {"z": _z(2, 10.0), "wkb_serial": 999},
                self.s.conn,
                self.s.tables["TkWKBValue"],
                None,
                self.s.tables,
                None,
            )
        self.assertIn("TkWKBValue", str(caught.exception))
        self.assertIn("result was not found", str(caught.exception))

    def test_gk_source_value_build_impl_serial(self):
        with self.assertRaises(RuntimeError) as caught:
            sqla_GkSourceValue_factory.build(
                {"z_source": _z(1, 1.0e5), "parent_serial": 999},
                self.s.conn,
                self.s.tables["GkSourceValue"],
                None,
                self.s.tables,
                None,
            )
        self.assertIn("GkSourceValue", str(caught.exception))
        self.assertIn("result was not found", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
