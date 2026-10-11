"""
Round-trip tests for the value reads prompts/datastore-integrity prompt 05c keys on their parent
(docs/datastore-integrity-audit.md §3.3, §3.4, §3.11; board items P3, P4, P5, and the close of
P9):

  * P3 -- `TkNumericValue._build_impl_model`, `GkNumericValue._build_impl_model` and
    `GkNumericValue.read_batch` omitted `break_point_kind`, which the parents' `build()` filter on,
    and took `atol`/`rtol` as optional where the parents require them. Given two parents that
    differ only in break-point kind, both `build` paths raised and `read_batch` returned both
    parents' values (`docs/datastore-integrity-audit/lookup_keys/probe_05_value_lookups_and_read_batch.py`).
    `TkNumericValue.build` dispatched on `payload["z_init"]` while its helper read
    `payload["z_source"]`.
  * P4 -- the Gk and Tk WKB value reads omitted `z_init`, which their parents' `build()` filter on
    when it is given.
  * D1 (README §6.2) -- a read holding only physical labels may select by them, but must refuse
    when more than one stored parent matches, naming the parents. `main.py`'s
    `build_missing_GkSource` asks for every `z_source` at once and files each value under its
    `_z_source.store_id`, so two parents for one `z_source` would silently overwrite each other.
    The Gk `read_batch`es refuse per `z_source`; `TkWKBValue.read_batch`, which has no `z_source`,
    per `(model, k)`.
  * P5 -- `GkNumericIntegration.build` and `GkNumericValue.read_batch` labelled `redshift.source` as
    `..._is_response`, so each reported a response flag equal to the source flag.
  * P9 -- a pin on every `read_batch` under `ObjectFactories/`, each naming the module that
    round-trips it.

No Ray, no datastore file, nothing under `var/`: the schema is built in memory with
`build_schema(MetaData(), config.datastore.factories)`, as in
`test_read_path_mechanics.py` and `test_integral_read_batch.py`, and every call is to the real
factory code. SQLite does not enforce foreign keys here, so parents that exist only as a serial --
a model, a tolerance, a tag, a solver -- are stand-ins carrying only `.store_id`.
"""

import ast
import unittest
from datetime import datetime
from pathlib import Path

import sqlalchemy as sqla
from sqlalchemy.exc import MultipleResultsFound

import ComputeTargets.phase_residual as phase_residual
from ComputeTargets import GkNumericIntegration, TkNumericIntegration
from CosmologyConcepts import redshift, wavenumber, wavenumber_exit_time
from CosmologyConcepts.wavenumber import (
    WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS,
    WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS,
)
from config.datastore import factories as _factories
from Datastore.SQL.ObjectFactories.GkNumericIntegration import (
    sqla_GkNumericIntegration_factory,
    sqla_GkNumericValue_factory,
)
from Datastore.SQL.ObjectFactories.GkWKBIntegration import sqla_GkWKBValue_factory
from Datastore.SQL.ObjectFactories.TkNumericIntegration import (
    sqla_TkNumericValue_factory,
)
from Datastore.SQL.ObjectFactories.TkWKBIntegration import sqla_TkWKBValue_factory
from datastorekit.SQL.schema import build_schema
from Datastore.tests.test_factory_select_columns import FACTORY_DIR, iter_functions
from MetadataConcepts import tolerance
from Units import Mpc_units

UNITS = Mpc_units()
FIXED_TIMESTAMP = datetime(2026, 9, 28, 12, 0, 0)

# the break-point kind each numeric sector's parent build() filters on, and the other one
GK_KIND = GkNumericIntegration.BREAK_POINT_KIND
TK_KIND = TkNumericIntegration.BREAK_POINT_KIND
_OTHER_KIND = {"all": "discontinuity", "discontinuity": "all"}


class _ModelProxy:
    """Anything a factory reads `.store_id` and `.units` off, standing in for `ModelProxy`."""

    def __init__(self, store_id=1):
        self.store_id = store_id
        self.units = UNITS


class _Ref:
    """Anything a factory only reads `.store_id` off: a tolerance, a tag, a cosmology."""

    def __init__(self, store_id, **extra):
        self.store_id = store_id
        for key, value in extra.items():
            setattr(self, key, value)


def _z(serial, value, is_source=False, is_response=False):
    return redshift(
        store_id=serial, z=value, is_source=is_source, is_response=is_response
    )


def _exit_time_obj(serial: int, wavenumber_serial: int, k_inv_Mpc: float):
    """A real `wavenumber_exit_time`, which `GkNumericIntegration`'s constructor needs (it reads
    `k.k.units`); the value reads only read its `.store_id`."""
    payload = {"store_id": serial, "z_exit": 1.0, "compute_time": 0.0, "stepping": 0}
    for o in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
        payload[f"z_exit_suph_e{o}"] = 1.0
    for o in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
        payload[f"z_exit_subh_e{o}"] = 1.0
    return wavenumber_exit_time(
        payload=payload,
        k=wavenumber(store_id=wavenumber_serial, k_inv_Mpc=k_inv_Mpc, units=UNITS),
        cosmology=_Ref(1, units=UNITS),
        atol=tolerance(store_id=1, log10_tol=-10.0),
        rtol=tolerance(store_id=2, log10_tol=-9.0),
    )


MODEL = _ModelProxy(store_id=1)
K_EXIT = _exit_time_obj(serial=11, wavenumber_serial=1, k_inv_Mpc=1.0e5)
ATOL = tolerance(store_id=1, log10_tol=-10.0)
RTOL = tolerance(store_id=2, log10_tol=-9.0)
TAG_T = _Ref(1)
TAG_U = _Ref(2)

# redshifts: three source redshifts, a response redshift at which main.py asks, and a second
# response redshift that must not come back when the first is asked for
Z_SOURCES = {1: 1.0e5, 3: 2.0e5, 4: 3.0e5}
Z_RESPONSE = _z(2, 10.0, is_source=False, is_response=True)
Z_OTHER_RESPONSE = _z(5, 5.0, is_source=False, is_response=True)


class Store:
    """Every table `build_schema` declares, in one in-memory SQLite database."""

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
    s = Store()
    s.insert("version", serial=1, label="test_value_reads_parent_key")
    for serial, value in Z_SOURCES.items():
        s.insert("redshift", serial=serial, z=value, source=True, response=False)
    s.insert("redshift", serial=2, z=10.0, source=False, response=True)
    s.insert("redshift", serial=5, z=5.0, source=False, response=True)
    s.insert("store_tag", serial=1, label="T")
    s.insert("store_tag", serial=2, label="U")
    s.insert("IntegrationSolver", serial=1, label="solver", stepping=0)
    s.conn.commit()
    return s


def _gk_numeric(
    s, serial, z_source_serial=1, kind=GK_KIND, tags=(), z_values=(2,), atol=1, rtol=2
):
    """A validated GkNumericIntegration parent with one value row per `z_values` serial. Each
    value's G is the parent serial, so a returned value names the parent it came from.
    """
    s.insert(
        "GkNumericIntegration",
        serial=serial,
        label=f"gknum-{serial}",
        wavenumber_exit_serial=K_EXIT.store_id,
        model_serial=MODEL.store_id,
        atol_serial=atol,
        rtol_serial=rtol,
        break_point_kind=kind,
        solver_serial=1,
        z_source_serial=z_source_serial,
        z_min_serial=2,
        z_samples=len(z_values),
        validated=True,
    )
    for tag in tags:
        s.insert("GkNumeric_tags", integration_serial=serial, tag_serial=tag.store_id)
    for z_serial in z_values:
        s.insert(
            "GkNumericValue",
            serial=100 * serial + z_serial,
            integration_serial=serial,
            z_serial=z_serial,
            G=float(serial),
            Gprime=0.0,
        )


def _tk_numeric(s, serial, kind=TK_KIND, z_init_serial=1, tags=(), z_values=(2,)):
    s.insert(
        "TkNumericIntegration",
        serial=serial,
        label=f"tknum-{serial}",
        wavenumber_exit_serial=K_EXIT.store_id,
        model_serial=MODEL.store_id,
        atol_serial=ATOL.store_id,
        rtol_serial=RTOL.store_id,
        break_point_kind=kind,
        solver_serial=1,
        z_init_serial=z_init_serial,
        z_min_serial=2,
        z_samples=len(z_values),
        validated=True,
    )
    for tag in tags:
        s.insert("TkNumeric_tags", integration_serial=serial, tag_serial=tag.store_id)
    for z_serial in z_values:
        s.insert(
            "TkNumericValue",
            serial=100 * serial + z_serial,
            integration_serial=serial,
            z_serial=z_serial,
            T=float(serial),
            Tprime=0.0,
        )


def _gk_wkb(
    s,
    serial,
    z_source_serial=1,
    z_init=1.0e4,
    tags=(),
    z_values=(2,),
    numeric_serial=None,
):
    """A validated GkWKBIntegration parent; each value's H_ratio is the parent serial. Pure WKB
    (no numeric parent) unless `numeric_serial` is given."""
    s.insert(
        "GkWKBIntegration",
        serial=serial,
        label=f"gkwkb-{serial}",
        wavenumber_exit_serial=K_EXIT.store_id,
        model_serial=MODEL.store_id,
        numeric_serial=numeric_serial,
        rho_gauss_order=phase_residual.RHO_GAUSS_ORDER,
        solver_serial=1,
        z_source_serial=z_source_serial,
        z_min_serial=2,
        z_samples=len(z_values),
        z_init=z_init,
        sin_coeff=0.0,
        cos_coeff=1.0,
        G_init=0.0,
        Gprime_init=1.0,
        validated=True,
    )
    for tag in tags:
        s.insert("GkWKB_tags", wkb_serial=serial, tag_serial=tag.store_id)
    for z_serial in z_values:
        s.insert(
            "GkWKBValue",
            serial=100 * serial + z_serial,
            wkb_serial=serial,
            z_serial=z_serial,
            H_ratio=float(serial),
            theta_mod_2pi=0.5,
            theta_div_2pi=3,
        )


def _tk_wkb(s, serial, z_init=1.0e4, tags=(), z_values=(2,), numeric_serial=1):
    """A validated TkWKBIntegration parent; each value's friction is the parent serial. Its
    numeric parent (NOT NULL since prompts/datastore-integrity prompt 07) is `numeric_serial`,
    a serial only: foreign keys are not enforced here."""
    s.insert(
        "TkWKBIntegration",
        serial=serial,
        label=f"tkwkb-{serial}",
        wavenumber_exit_serial=K_EXIT.store_id,
        model_serial=MODEL.store_id,
        numeric_serial=numeric_serial,
        rho_gauss_order=phase_residual.RHO_GAUSS_ORDER,
        phase_solver_serial=1,
        friction_solver_serial=1,
        z_min_serial=2,
        z_samples=len(z_values),
        z_init=z_init,
        sin_coeff=0.0,
        cos_coeff=1.0,
        T_init=1.0,
        Tprime_init=0.0,
        validated=True,
    )
    for tag in tags:
        s.insert("TkWKB_tags", wkb_serial=serial, tag_serial=tag.store_id)
    for z_serial in z_values:
        s.insert(
            "TkWKBValue",
            serial=100 * serial + z_serial,
            wkb_serial=serial,
            z_serial=z_serial,
            H_ratio=1.0,
            theta_mod_2pi=0.5,
            theta_div_2pi=3,
            friction=float(serial),
        )


def _read_batch(s, factory, name, payload):
    return list(factory.read_batch(payload, s.conn, s.tables[name], s.tables))


def _build(s, factory, name, payload):
    return factory.build(payload, s.conn, s.tables[name], None, s.tables, None)


def _gk_numeric_model_payload(**extra):
    payload = {
        "z": Z_RESPONSE,
        "model": MODEL,
        "k": K_EXIT,
        "z_source": _z(1, Z_SOURCES[1], is_source=True),
        "atol": ATOL,
        "rtol": RTOL,
    }
    payload.update(extra)
    return payload


def _tk_numeric_model_payload(**extra):
    payload = {
        "z": Z_RESPONSE,
        "model": MODEL,
        "k": K_EXIT,
        "z_init": _z(1, Z_SOURCES[1], is_source=True),
        "atol": ATOL,
        "rtol": RTOL,
    }
    payload.update(extra)
    return payload


class _StoreCase(unittest.TestCase):
    def setUp(self):
        self.s = _base_store()
        self.addCleanup(self.s.close)


class TestBreakPointKindSelects(_StoreCase):
    """
    §3 item 1 (probe_05's design). Two numeric parents that differ only in break-point kind: every
    numeric value read serves only the parent whose kind is the compute target's constant, and
    nothing raises.
    """

    def test_gk_numeric_value_build_impl_model_serves_only_the_kind_asked_for(self):
        _gk_numeric(self.s, serial=1, kind=GK_KIND)
        _gk_numeric(self.s, serial=2, kind=_OTHER_KIND[GK_KIND])
        self.s.conn.commit()

        obj = _build(
            self.s,
            sqla_GkNumericValue_factory,
            "GkNumericValue",
            _gk_numeric_model_payload(),
        )
        self.assertEqual(102, obj.store_id)
        self.assertEqual(1.0, obj.G)

    def test_gk_numeric_value_read_batch_serves_only_the_kind_asked_for(self):
        _gk_numeric(self.s, serial=1, kind=GK_KIND)
        _gk_numeric(self.s, serial=2, kind=_OTHER_KIND[GK_KIND])
        self.s.conn.commit()

        objs = _read_batch(
            self.s,
            sqla_GkNumericValue_factory,
            "GkNumericValue",
            {"model": MODEL, "k": K_EXIT, "atol": ATOL, "rtol": RTOL},
        )
        self.assertEqual([102], [o.store_id for o in objs])

    def test_tk_numeric_value_build_with_z_init_serves_only_the_kind_asked_for(self):
        _tk_numeric(self.s, serial=1, kind=TK_KIND)
        _tk_numeric(self.s, serial=2, kind=_OTHER_KIND[TK_KIND])
        self.s.conn.commit()

        # one key, z_init: no z_source in the payload at all
        obj = _build(
            self.s,
            sqla_TkNumericValue_factory,
            "TkNumericValue",
            _tk_numeric_model_payload(),
        )
        self.assertEqual(102, obj.store_id)
        self.assertEqual(1.0, obj.T)


class TestTolerancesRequired(_StoreCase):
    """§3 item 2. Each numeric value read, with `atol` or `rtol` omitted, raises."""

    def setUp(self):
        super().setUp()
        _gk_numeric(self.s, serial=1)
        _tk_numeric(self.s, serial=1)
        self.s.conn.commit()

    def _omitting(self, payload, key):
        return {k: v for k, v in payload.items() if k != key}

    def test_each_numeric_read_raises_without_each_tolerance(self):
        reads = {
            "GkNumericValue._build_impl_model": lambda p: _build(
                self.s, sqla_GkNumericValue_factory, "GkNumericValue", p
            ),
            "GkNumericValue.read_batch": lambda p: _read_batch(
                self.s, sqla_GkNumericValue_factory, "GkNumericValue", p
            ),
            "TkNumericValue._build_impl_model": lambda p: _build(
                self.s, sqla_TkNumericValue_factory, "TkNumericValue", p
            ),
        }
        payloads = {
            "GkNumericValue._build_impl_model": _gk_numeric_model_payload(),
            "GkNumericValue.read_batch": {
                "model": MODEL,
                "k": K_EXIT,
                "atol": ATOL,
                "rtol": RTOL,
            },
            "TkNumericValue._build_impl_model": _tk_numeric_model_payload(),
        }
        for name, read in reads.items():
            # the complete payload is served without raising, so what raises below is the
            # omission alone
            read(payloads[name])
            for key in ("atol", "rtol"):
                with self.subTest(read=name, omitted=key):
                    with self.assertRaises(KeyError) as caught:
                        read(self._omitting(payloads[name], key))
                    self.assertEqual(key, caught.exception.args[0])


class TestLivePayloadShape(_StoreCase):
    """
    §3 item 3. `main.py`'s `build_missing_GkSource` payload: tags, `z` (the response), no
    `z_source`, and the tolerance pair for `GkNumericValue` only. With one parent per `z_source`
    across three `z_source`s, every value comes back, each carrying its own `_z_source`. This is
    P9's untagged-`z_source` round trip for both live `read_batch`es.
    """

    def setUp(self):
        super().setUp()
        # parent serials 21, 23, 24 at z_source serials 1, 3, 4; a value at both response z's
        for z_source_serial in Z_SOURCES:
            _gk_numeric(
                self.s,
                serial=20 + z_source_serial,
                z_source_serial=z_source_serial,
                tags=(TAG_T, TAG_U),
                z_values=(2, 5),
            )
            _gk_wkb(
                self.s,
                serial=20 + z_source_serial,
                z_source_serial=z_source_serial,
                tags=(TAG_T, TAG_U),
                z_values=(2, 5),
            )
        self.s.conn.commit()

    def _check(self, objs, parent_of):
        self.assertEqual(3, len(objs))
        by_z_source = {o._z_source.store_id: o for o in objs}
        self.assertEqual(set(Z_SOURCES), set(by_z_source))
        for z_source_serial, obj in by_z_source.items():
            self.assertEqual(Z_RESPONSE.store_id, obj.z.store_id)
            self.assertEqual(K_EXIT.store_id, obj._k_exit.store_id)
            self.assertEqual(Z_SOURCES[z_source_serial], obj._z_source.z)
            # the value came from the parent stored at that z_source, not another
            self.assertEqual(float(20 + z_source_serial), parent_of(obj))

    def test_gk_numeric_value_read_batch(self):
        objs = _read_batch(
            self.s,
            sqla_GkNumericValue_factory,
            "GkNumericValue",
            {
                "model": MODEL,
                "z": Z_RESPONSE,
                "atol": ATOL,
                "rtol": RTOL,
                "tags": [TAG_T, TAG_U],
                "k": K_EXIT,
            },
        )
        self._check(objs, lambda o: o.G)

    def test_gk_wkb_value_read_batch(self):
        objs = _read_batch(
            self.s,
            sqla_GkWKBValue_factory,
            "GkWKBValue",
            {
                "model": MODEL,
                "z": Z_RESPONSE,
                "tags": [TAG_T, TAG_U],
                "k": K_EXIT,
            },
        )
        self._check(objs, lambda o: o.H_ratio)


class TestSecondParentIsRefused(_StoreCase):
    """
    §3 item 4. Two parents that both match -- tags {T} and {T, U} queried with [T], and for WKB
    also two `z_init`s with no `z_init` given -- make each `read_batch` raise
    `MultipleResultsFound` naming both parent serials, with `z_source` given and without. Two
    parents at *different* `z_source`s do not refuse.
    """

    def _assert_refuses(self, read, payload, cls_name, parents, z_source_serial=None):
        with self.assertRaises(MultipleResultsFound) as caught:
            read(payload)
        message = str(caught.exception)
        self.assertIn(cls_name, message)
        self.assertIn(f"k serial={K_EXIT.store_id}", message)
        self.assertIn(f"parent serials {sorted(parents)}", message)
        if z_source_serial is not None:
            self.assertIn(f"z_source serial={z_source_serial}", message)

    def _gk_numeric_read(self, payload):
        return _read_batch(
            self.s, sqla_GkNumericValue_factory, "GkNumericValue", payload
        )

    def _gk_wkb_read(self, payload):
        return _read_batch(self.s, sqla_GkWKBValue_factory, "GkWKBValue", payload)

    def _tk_wkb_read(self, payload):
        return _read_batch(self.s, sqla_TkWKBValue_factory, "TkWKBValue", payload)

    def _gk_payloads(self, base):
        """The same query without z_source, and with it."""
        return [
            dict(base),
            dict(base, z_source=_z(1, Z_SOURCES[1], is_source=True)),
        ]

    def test_gk_numeric_value_tags(self):
        _gk_numeric(self.s, serial=7, tags=(TAG_T,))
        _gk_numeric(self.s, serial=9, tags=(TAG_T, TAG_U))
        self.s.conn.commit()
        base = {"model": MODEL, "k": K_EXIT, "atol": ATOL, "rtol": RTOL}
        for payload in self._gk_payloads(dict(base, tags=[TAG_T])):
            with self.subTest(z_source="z_source" in payload):
                self._assert_refuses(
                    self._gk_numeric_read, payload, "GkNumericValue", [7, 9], 1
                )
        # the tag that distinguishes them selects one, and nothing refuses
        objs = self._gk_numeric_read(dict(base, tags=[TAG_T, TAG_U]))
        self.assertEqual([9.0], [o.G for o in objs])

    def test_gk_wkb_value_tags(self):
        _gk_wkb(self.s, serial=7, tags=(TAG_T,))
        _gk_wkb(self.s, serial=9, tags=(TAG_T, TAG_U))
        self.s.conn.commit()
        base = {"model": MODEL, "k": K_EXIT}
        for payload in self._gk_payloads(dict(base, tags=[TAG_T])):
            with self.subTest(z_source="z_source" in payload):
                self._assert_refuses(
                    self._gk_wkb_read, payload, "GkWKBValue", [7, 9], 1
                )
        objs = self._gk_wkb_read(dict(base, tags=[TAG_T, TAG_U]))
        self.assertEqual([9.0], [o.H_ratio for o in objs])

    def test_gk_wkb_value_two_numeric_parents(self):
        _gk_wkb(self.s, serial=7, numeric_serial=1)
        _gk_wkb(self.s, serial=9, numeric_serial=2)
        self.s.conn.commit()
        for payload in self._gk_payloads({"model": MODEL, "k": K_EXIT}):
            with self.subTest(z_source="z_source" in payload):
                self._assert_refuses(
                    self._gk_wkb_read, payload, "GkWKBValue", [7, 9], 1
                )

    def test_tk_wkb_value_tags(self):
        _tk_wkb(self.s, serial=7, tags=(TAG_T,))
        _tk_wkb(self.s, serial=9, tags=(TAG_T, TAG_U))
        self.s.conn.commit()
        base = {"model": MODEL, "k": K_EXIT}
        for payload in (
            dict(base, tags=[TAG_T]),
            dict(base, tags=[TAG_T], z=Z_RESPONSE),
        ):
            with self.subTest(z="z" in payload):
                self._assert_refuses(self._tk_wkb_read, payload, "TkWKBValue", [7, 9])
        objs = self._tk_wkb_read(dict(base, tags=[TAG_T, TAG_U]))
        self.assertEqual([9.0], [o.friction for o in objs])

    def test_tk_wkb_value_two_numeric_parents(self):
        _tk_wkb(self.s, serial=7, numeric_serial=1)
        _tk_wkb(self.s, serial=9, numeric_serial=2)
        self.s.conn.commit()
        self._assert_refuses(
            self._tk_wkb_read, {"model": MODEL, "k": K_EXIT}, "TkWKBValue", [7, 9]
        )

    def test_parents_at_different_z_sources_do_not_refuse(self):
        _gk_numeric(self.s, serial=7, z_source_serial=1)
        _gk_numeric(self.s, serial=9, z_source_serial=3)
        _gk_wkb(self.s, serial=7, z_source_serial=1, z_init=1.0e4)
        _gk_wkb(self.s, serial=9, z_source_serial=3, z_init=2.0e4)
        self.s.conn.commit()

        numeric = self._gk_numeric_read(
            {"model": MODEL, "k": K_EXIT, "atol": ATOL, "rtol": RTOL}
        )
        wkb = self._gk_wkb_read({"model": MODEL, "k": K_EXIT})
        self.assertEqual({1: 7.0, 3: 9.0}, {o._z_source.store_id: o.G for o in numeric})
        self.assertEqual(
            {1: 7.0, 3: 9.0}, {o._z_source.store_id: o.H_ratio for o in wkb}
        )


class TestNumericParentSelects(_StoreCase):
    """
    §3 item 5, on prompts/datastore-integrity prompt 07's key. With the two-numeric-parent store,
    giving one numeric parent serves only that WKB parent, through `read_batch` and through
    `_build_impl_model`, for both `GkWKBValue` and `TkWKBValue`. (Written by 05c on `z_init`,
    which prompt 07 took out of the key; each WKB parent keeps its own `z_init`, which the value
    still reports.)
    """

    PARENTS = ((7, 1, 1.0e4), (9, 2, 2.0e4))  # (WKB serial, numeric parent, z_init)

    def setUp(self):
        super().setUp()
        for serial, numeric_serial, z_init in self.PARENTS:
            _gk_wkb(self.s, serial=serial, z_init=z_init, numeric_serial=numeric_serial)
            _tk_wkb(self.s, serial=serial, z_init=z_init, numeric_serial=numeric_serial)
        self.s.conn.commit()

    def test_gk_wkb_value(self):
        for serial, numeric_serial, z_init in self.PARENTS:
            with self.subTest(numeric_serial=numeric_serial):
                objs = _read_batch(
                    self.s,
                    sqla_GkWKBValue_factory,
                    "GkWKBValue",
                    {"model": MODEL, "k": K_EXIT, "numeric_serial": numeric_serial},
                )
                self.assertEqual([float(serial)], [o.H_ratio for o in objs])
                self.assertEqual([z_init], [o.z_init for o in objs])

                obj = _build(
                    self.s,
                    sqla_GkWKBValue_factory,
                    "GkWKBValue",
                    {
                        "z": Z_RESPONSE,
                        "model": MODEL,
                        "k": K_EXIT,
                        "z_source": _z(1, Z_SOURCES[1], is_source=True),
                        "numeric_serial": numeric_serial,
                    },
                )
                self.assertEqual(100 * serial + 2, obj.store_id)
                self.assertEqual(z_init, obj.z_init)

    def test_tk_wkb_value(self):
        for serial, numeric_serial, z_init in self.PARENTS:
            with self.subTest(numeric_serial=numeric_serial):
                objs = _read_batch(
                    self.s,
                    sqla_TkWKBValue_factory,
                    "TkWKBValue",
                    {"model": MODEL, "k": K_EXIT, "numeric_serial": numeric_serial},
                )
                self.assertEqual([float(serial)], [o.friction for o in objs])
                self.assertEqual([z_init], [o.z_init for o in objs])

                obj = _build(
                    self.s,
                    sqla_TkWKBValue_factory,
                    "TkWKBValue",
                    {
                        "z": Z_RESPONSE,
                        "model": MODEL,
                        "k": K_EXIT,
                        "numeric_serial": numeric_serial,
                    },
                )
                self.assertEqual(100 * serial + 2, obj.store_id)
                self.assertEqual(z_init, obj.z_init)


class TestResponseFlagsAreRead(_StoreCase):
    """
    §3 item 6 (P5). A redshift row stored with `source=True, response=False` and one with the
    reverse, read back through `GkNumericIntegration.build` and `GkNumericValue.read_batch`: each
    flag equals what was stored. Parent 1 sits at the first as its z_source with a value at the
    second; parent 2 the other way round, so both combinations pass through both labels.
    """

    SOURCE_LIKE = 1  # redshift serial 1: source=True, response=False
    RESPONSE_LIKE = 2  # redshift serial 2: source=False, response=True
    STORED = {1: (True, False), 2: (False, True)}

    def setUp(self):
        super().setUp()
        _gk_numeric(
            self.s,
            serial=1,
            z_source_serial=self.SOURCE_LIKE,
            z_values=(self.RESPONSE_LIKE,),
        )
        _gk_numeric(
            self.s,
            serial=2,
            z_source_serial=self.RESPONSE_LIKE,
            z_values=(self.SOURCE_LIKE,),
        )
        self.s.conn.commit()

    def _flags(self, z: redshift):
        return (z.is_source, z.is_response)

    def test_gk_numeric_integration_build(self):
        for z_source_serial in (self.SOURCE_LIKE, self.RESPONSE_LIKE):
            with self.subTest(z_source=z_source_serial):
                obj = sqla_GkNumericIntegration_factory.build(
                    {
                        "solver_labels": {},
                        "atol": ATOL,
                        "rtol": RTOL,
                        "k": K_EXIT,
                        "model": MODEL,
                        "z_sample": None,
                        "z_source": _z(z_source_serial, 0.0),
                        "_do_not_populate": True,
                    },
                    self.s.conn,
                    self.s.tables["GkNumericIntegration"],
                    None,
                    self.s.tables,
                    None,
                )
                self.assertEqual(z_source_serial, obj.store_id)
                self.assertEqual(
                    self.STORED[z_source_serial], self._flags(obj.z_source)
                )

    def test_gk_numeric_value_read_batch(self):
        objs = _read_batch(
            self.s,
            sqla_GkNumericValue_factory,
            "GkNumericValue",
            {"model": MODEL, "k": K_EXIT, "atol": ATOL, "rtol": RTOL},
        )
        self.assertEqual(2, len(objs))
        for obj in objs:
            with self.subTest(parent=obj.G):
                self.assertEqual(
                    self.STORED[obj._z_source.store_id], self._flags(obj._z_source)
                )
                self.assertEqual(self.STORED[obj.z.store_id], self._flags(obj.z))


# P9's pin: every factory function named `read_batch` under ObjectFactories/, as
# "module.py:class", each naming the test module that round-trips it. A new read_batch fails
# test_every_read_batch_has_a_round_trip until someone writes its test and adds it here.
READ_BATCH_ROUND_TRIPS = {
    "GkNumericIntegration.py:sqla_GkNumericValue_factory": "Datastore.tests.test_value_reads_parent_key",
    "GkWKBIntegration.py:sqla_GkWKBValue_factory": "Datastore.tests.test_value_reads_parent_key",
    "TkWKBIntegration.py:sqla_TkWKBValue_factory": "Datastore.tests.test_value_reads_parent_key",
    "QuadSourceIntegral.py:sqla_QuadSourceIntegral_factory": "Datastore.tests.test_integral_read_batch",
    "OneLoopIntegral.py:sqla_OneLoopIntegral_factory": "Datastore.tests.test_integral_read_batch",
}


class TestEveryReadBatchHasARoundTrip(unittest.TestCase):
    """§3 item 7 (P9). The set of `read_batch`es, found by walking the sources as the guard does."""

    def test_every_read_batch_has_a_round_trip(self):
        found = set()
        for path in sorted(FACTORY_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for cls, fn in iter_functions(tree):
                if fn.name == "read_batch":
                    found.add(f"{path.name}:{cls}")
        self.assertEqual(
            set(READ_BATCH_ROUND_TRIPS),
            found,
            "the set of read_batch functions under ObjectFactories/ has changed. A read_batch "
            "is a read path the build never exercises: write a round-trip test that executes "
            "it, and add it to READ_BATCH_ROUND_TRIPS naming that test module.",
        )

    def test_each_named_module_calls_its_read_batch(self):
        """
        Each named module calls that factory's `read_batch`: either `<factory>.read_batch(...)`
        directly, or this module's `_read_batch(store, <factory>, ...)` helper, which does.
        """
        root = Path(__file__).parents[2]
        for site, test_module in READ_BATCH_ROUND_TRIPS.items():
            with self.subTest(site=site):
                cls = site.split(":")[1]
                path = root / (test_module.replace(".", "/") + ".py")
                calls = [
                    n
                    for n in ast.walk(ast.parse(path.read_text()))
                    if isinstance(n, ast.Call)
                ]
                direct = any(
                    isinstance(c.func, ast.Attribute)
                    and c.func.attr == "read_batch"
                    and isinstance(c.func.value, ast.Name)
                    and c.func.value.id == cls
                    for c in calls
                )
                through_helper = any(
                    isinstance(c.func, ast.Name)
                    and c.func.id == "_read_batch"
                    and any(isinstance(a, ast.Name) and a.id == cls for a in c.args)
                    for c in calls
                )
                self.assertTrue(
                    direct or through_helper,
                    f"{test_module} does not call {cls}.read_batch",
                )


if __name__ == "__main__":
    unittest.main()
