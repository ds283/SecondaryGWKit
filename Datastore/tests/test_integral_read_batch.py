"""
Round-trip tests for the integral `read_batch`es prompt 05b repairs
(docs/datastore-integrity-audit.md §3.1, §3.2, §3.12; board items P1, P2, P8):

  * P1 -- the tagged join in both `QuadSourceIntegral.read_batch` and
    `sqla_OneLoopIntegral_factory.read_batch` was written `tab.c.parent_serial == query.c.serial`.
    `query.c` makes SQLAlchemy name an anonymous subquery outside the FROM clause, so every tagged
    call raised `OperationalError: no such column: anon_1.serial` (audit §3.1,
    `docs/datastore-integrity-audit/lookup_keys/probe_01_tagged_read_batch.py`). Fixed to
    `table.c.serial`, as `probe_01b_fix_shape.py` verified.
  * P2 -- `QuadSourceIntegral.read_batch` selected the *wavenumber* serials as `q_serial`/
    `r_serial` and used them correctly for the wavenumbers, but also -- wrongly -- as the exit
    times' `store_id` (audit §3.2). Fixed by selecting each exit time's own serial
    (`q_exit_serial`/`r_exit_serial`) and using that instead.
  * P8 -- `sqla_OneLoopIntegral_factory`, bounded by README D3 to its mechanical defects:
    `build()` joined `QuadSourceIntegral_tags` while `store()` wrote `OneLoopIntegral_tags` (audit
    §3.12, `probe_06_oneloop_tags_and_redshift_zero.py`); `read_batch` was a module-level function
    `Datastore.object_read_batch` could never reach (it calls `factory.read_batch`, and `factory`
    is the class itself), had no `return` statement, and its tags defaulted to `None`. Left alone
    by D3: `build()`/`read_batch()` still put no `source_serial` in the payload, and the real
    `OneLoopIntegral` constructor still reads `payload["source_serial"]` unconditionally, so every
    hit still raises `KeyError` -- `[00-oneloop-factory-cannot-round-trip-a-row]`, which test 5
    below keeps visible rather than papering over.

No Ray, no datastore file, nothing under `var/`: the schema is built in memory with
`build_schema(MetaData(), config.datastore.factories)`, the same construction
`docs/datastore-integrity-audit/lookup_keys/common.py` and
`Datastore/tests/test_read_path_mechanics.py` use, and every call below is to the real,
unmodified `build()` / `read_batch()` code.

Tests 3 and 4 construct a real `OneLoopIntegral` from what `build()`/`read_batch()` pass it, which
-- as test 5 demonstrates, and D3 accepted -- raises `KeyError: 'source_serial'` on every hit. So
those two patch the name `OneLoopIntegral` inside `Datastore.SQL.ObjectFactories.OneLoopIntegral`
with a stand-in that only records its constructor's keyword arguments, scoped by
`unittest.mock.patch` (README §5 rule 9); the real class, and every other module, are untouched.

SQLite does not enforce foreign keys unless `PRAGMA foreign_keys` is turned on, and the production
`Datastore` does not turn it on (see `test_backgroundmodelvalue_roundtrip.py`), so parent rows that
exist only as a bare integer serial -- a model, a policy, a tag -- are stand-ins carrying only the
`.store_id` a factory reads off them, exactly as
`docs/datastore-integrity-audit/lookup_keys/common.py`'s `Ref` and `StandInModelProxy` do.
"""

import unittest
from datetime import datetime
from typing import List
from unittest import mock

import sqlalchemy as sqla

from CosmologyConcepts import redshift, wavenumber, wavenumber_exit_time
from CosmologyConcepts.wavenumber import (
    WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS,
    WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS,
)
from config.datastore import factories as _factories
from Datastore.SQL.ObjectFactories import OneLoopIntegral as OLI_module
from Datastore.SQL.ObjectFactories.OneLoopIntegral import sqla_OneLoopIntegral_factory
from Datastore.SQL.ObjectFactories.QuadSourceIntegral import (
    sqla_QuadSourceIntegral_factory,
)
from datastorekit.SQL.schema import build_schema
from MetadataConcepts import store_tag, tolerance
from Units import Mpc_units

UNITS = Mpc_units()
FIXED_TIMESTAMP = datetime(2026, 9, 28, 12, 0, 0)


class _ModelProxy:
    """Anything a factory reads `.store_id`, `.units` and `.cosmology` off, standing in for
    `ModelProxy`."""

    def __init__(self, store_id=1):
        self.store_id = store_id
        self.units = UNITS
        self.cosmology = _Ref(store_id, units=UNITS)


class _Ref:
    """Anything a factory only reads `.store_id` off: a policy, a tolerance, a tag, a cosmology."""

    def __init__(self, store_id, **extra):
        self.store_id = store_id
        for k, v in extra.items():
            setattr(self, k, v)


def _z(serial, value, is_source=False, is_response=False):
    return redshift(
        store_id=serial, z=value, is_source=is_source, is_response=is_response
    )


class Store:
    """
    Every table `build_schema` declares, in one in-memory SQLite database. No Ray, no file,
    nothing under `var/`: `build_schema` only declares `sqla.Table` objects, and `create_all`
    makes an in-memory schema from them.
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


def _exit_time_row(s: Store, exit_serial: int, wavenumber_serial: int) -> None:
    """
    A `wavenumber_exit_time` row whose own serial (11/12/13) is deliberately different from the
    `wavenumber` row it points at (1/2/3) -- exactly the distinct-serials precondition §3 item 2
    of the prompt asks the log to state: without it, P2's defect (the wavenumber's serial used as
    the exit time's `store_id`) and the fix are indistinguishable.
    """
    row = dict(
        serial=exit_serial,
        stepping=0,
        wavenumber_serial=wavenumber_serial,
        cosmology_type=0,
        cosmology_serial=1,
        atol_serial=1,
        rtol_serial=2,
        compute_time=0.1,
        z_exit=1.0e4 * wavenumber_serial,
    )
    for o in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
        row[f"z_exit_suph_e{o}"] = 1.0
    for o in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
        row[f"z_exit_subh_e{o}"] = 1.0
    s.insert("wavenumber_exit_time", **row)


def _exit_time_obj(serial: int, wavenumber_serial: int, k_inv_Mpc: float):
    """A real `wavenumber_exit_time` object matching `_exit_time_row`'s shape, for a `k`/`q`/`r`
    payload argument."""
    payload = {
        "store_id": serial,
        "z_exit": 1.0,
        "compute_time": 0.0,
        "stepping": 0,
    }
    for o in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
        payload[f"z_exit_suph_e{o}"] = 1.0
    for o in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
        payload[f"z_exit_subh_e{o}"] = 1.0
    return wavenumber_exit_time(
        payload=payload,
        k=wavenumber(store_id=wavenumber_serial, k_inv_Mpc=k_inv_Mpc, units=UNITS),
        cosmology=_ModelProxy().cosmology,
        atol=tolerance(store_id=1, log10_tol=-10.0),
        rtol=tolerance(store_id=2, log10_tol=-9.0),
    )


class TestQuadSourceIntegralTaggedReadBatch(unittest.TestCase):
    """
    §3 item 1 (P1, probe_01's design): three `QuadSourceIntegral` rows sharing one key, tagged
    {A, B}, {A} and untagged. `tags=[]` must return all three, `[A]` two, `[A, B]` one, and no
    call may raise -- before the fix every tagged call raised `OperationalError`.
    """

    def setUp(self):
        self.s = s = Store()
        self.addCleanup(s.close)

        s.insert("version", serial=1, label="test_integral_read_batch")
        s.insert("tolerance", serial=1, log10_tol=-10.0)
        s.insert("tolerance", serial=2, log10_tol=-9.0)
        s.insert("store_tag", serial=1, label="A")
        s.insert("store_tag", serial=2, label="B")
        s.insert("redshift", serial=1, z=10.0, source=False, response=True)
        s.insert("redshift", serial=2, z=1.0e5, source=True, response=False)

        # wavenumbers 1..3 (k, q, r); exit-time serials 11..13 are deliberately different
        for serial, k in ((1, 1.0e5), (2, 2.0e5), (3, 3.0e5)):
            s.insert(
                "wavenumber", serial=serial, k_inv_Mpc=k, source=True, response=True
            )
        for exit_serial, wn_serial in ((11, 1), (12, 2), (13, 3)):
            _exit_time_row(s, exit_serial, wn_serial)

        for serial in (1, 2, 3):
            s.insert(
                "QuadSourceIntegral",
                serial=serial,
                label=f"qsi-{serial}",
                model_serial=1,
                k_wavenumber_exit_serial=11,
                q_wavenumber_exit_serial=12,
                r_wavenumber_exit_serial=13,
                atol_serial=1,
                rtol_serial=2,
                policy_serial=1,
                quad_policy_serial=1,
                source_serial=100 + serial,
                data_serial=200 + serial,
                Tq_numeric_serial=31,
                Tq_WKB_serial=41,
                Tr_numeric_serial=32,
                Tr_WKB_serial=42,
                z_response_serial=1,
                z_source_max_serial=2,
                b=0.0,
                total=float(serial),
                numeric_quad=0.0,
                WKB_Levin=0.0,
            )
        # serial 1: {A, B}; serial 2: {A}; serial 3: no tags
        for parent, tag in ((1, 1), (1, 2), (2, 1)):
            s.insert("QuadSourceIntegral_tags", parent_serial=parent, tag_serial=tag)
        s.conn.commit()

        self.k_exit = _exit_time_obj(11, 1, 1.0e5)
        self.tags = {
            "A": store_tag(store_id=1, label="A"),
            "B": store_tag(store_id=2, label="B"),
        }

    def _read_batch(self, tag_names) -> List:
        payload = dict(
            model=_ModelProxy(),
            policy=_Ref(1),
            quad_policy=_Ref(1),
            k=self.k_exit,
            atol=tolerance(store_id=1, log10_tol=-10.0),
            rtol=tolerance(store_id=2, log10_tol=-9.0),
            tags=[self.tags[n] for n in tag_names],
        )
        return sqla_QuadSourceIntegral_factory.read_batch(
            payload, self.s.conn, self.s.tables["QuadSourceIntegral"], self.s.tables
        )

    def test_no_tags_returns_all_three(self):
        objs = self._read_batch(())
        self.assertEqual({1, 2, 3}, {o.store_id for o in objs})

    def test_one_tag_returns_two(self):
        objs = self._read_batch(("A",))
        self.assertEqual({1, 2}, {o.store_id for o in objs})

    def test_two_tags_returns_one(self):
        objs = self._read_batch(("A", "B"))
        self.assertEqual({1}, {o.store_id for o in objs})


class TestQuadSourceIntegralExitTimeSerials(unittest.TestCase):
    """
    §3 item 2 (P2). The stored `q`/`r` exit-time rows (serials 12, 13) are deliberately different
    from the wavenumber rows they point at (serials 2, 3) -- see `_exit_time_row`'s docstring for
    why that is required for this test to be able to fail. Each returned integral's `q`/`r` exit
    times must carry their own exit-time row's serial; their wavenumbers must carry the
    wavenumber's serial.
    """

    def setUp(self):
        self.s = s = Store()
        self.addCleanup(s.close)

        s.insert("version", serial=1, label="test_integral_read_batch")
        s.insert("tolerance", serial=1, log10_tol=-10.0)
        s.insert("tolerance", serial=2, log10_tol=-9.0)
        s.insert("redshift", serial=1, z=10.0, source=False, response=True)
        s.insert("redshift", serial=2, z=1.0e5, source=True, response=False)
        for serial, k in ((1, 1.0e5), (2, 2.0e5), (3, 3.0e5)):
            s.insert(
                "wavenumber", serial=serial, k_inv_Mpc=k, source=True, response=True
            )
        for exit_serial, wn_serial in ((11, 1), (12, 2), (13, 3)):
            _exit_time_row(s, exit_serial, wn_serial)
        s.insert(
            "QuadSourceIntegral",
            serial=1,
            label="qsi-1",
            model_serial=1,
            k_wavenumber_exit_serial=11,
            q_wavenumber_exit_serial=12,
            r_wavenumber_exit_serial=13,
            atol_serial=1,
            rtol_serial=2,
            policy_serial=1,
            quad_policy_serial=1,
            source_serial=101,
            data_serial=201,
            Tq_numeric_serial=31,
            Tq_WKB_serial=41,
            Tr_numeric_serial=32,
            Tr_WKB_serial=42,
            z_response_serial=1,
            z_source_max_serial=2,
            b=0.0,
            total=1.0,
            numeric_quad=0.0,
            WKB_Levin=0.0,
        )
        s.conn.commit()

    def test_exit_times_and_wavenumbers_carry_their_own_serials(self):
        payload = dict(
            model=_ModelProxy(),
            policy=_Ref(1),
            quad_policy=_Ref(1),
            k=_exit_time_obj(11, 1, 1.0e5),
            atol=tolerance(store_id=1, log10_tol=-10.0),
            rtol=tolerance(store_id=2, log10_tol=-9.0),
            tags=[],
        )
        objs = sqla_QuadSourceIntegral_factory.read_batch(
            payload, self.s.conn, self.s.tables["QuadSourceIntegral"], self.s.tables
        )
        self.assertEqual(1, len(objs))
        obj = objs[0]

        # the exit times' own serials (P2's fix)
        self.assertEqual(12, obj._q_exit.store_id)
        self.assertEqual(13, obj._r_exit.store_id)

        # the wavenumbers' own serials (unchanged, always correct)
        self.assertEqual(2, obj.q.store_id)
        self.assertEqual(3, obj.r.store_id)


class _OneLoopStandIn:
    """
    Stand-in for `ComputeTargets.OneLoopIntegral`, patched in for `Datastore.SQL.ObjectFactories.
    OneLoopIntegral.OneLoopIntegral` so that `build()`/`read_batch()` can be exercised without
    hitting the real constructor's `payload["source_serial"]`, which is `[00-oneloop-factory-
    cannot-round-trip-a-row]` -- D3's declared, deliberately unfixed hole (test 5 keeps it real).
    Records every constructor keyword argument it was given.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        payload = kwargs.get("payload")
        self.store_id = payload["store_id"] if payload is not None else None


class TestOneLoopIntegralBuildTags(unittest.TestCase):
    """
    §3 item 3 (P8, probe_06's design). Two cases:
      * a row tagged in `OneLoopIntegral_tags` is found by its tags;
      * an untagged row whose serial coincides with a `QuadSourceIntegral_tags.parent_serial` row
        is *not* matched by those tags -- before the fix, `build()` joined
        `QuadSourceIntegral_tags`, so this coincidence produced a false hit (probe_06).
    """

    def _store_with_row(self):
        s = Store()
        self.addCleanup(s.close)
        s.insert("version", serial=1, label="test_integral_read_batch")
        s.insert("tolerance", serial=1, log10_tol=-10.0)
        s.insert("tolerance", serial=2, log10_tol=-8.0)
        s.insert("store_tag", serial=1, label="T")
        s.insert("redshift", serial=1, z=10.0, source=False, response=True)
        s.insert("wavenumber", serial=1, k_inv_Mpc=1.0e5, source=True, response=True)
        _exit_time_row(s, exit_serial=11, wavenumber_serial=1)
        s.insert(
            "OneLoopIntegral",
            serial=5,
            label="oli-5",
            model_serial=1,
            wavenumber_exit_serial=11,
            atol_serial=1,
            rtol_serial=2,
            z_response_serial=1,
            value=1.25,
        )
        return s

    def _build(self, s: Store):
        payload = dict(
            model=_ModelProxy(),
            k=_exit_time_obj(11, 1, 1.0e5),
            z_response=_Ref(1),
            atol=tolerance(store_id=1, log10_tol=-10.0),
            rtol=tolerance(store_id=2, log10_tol=-8.0),
            tags=[store_tag(store_id=1, label="T")],
        )
        with mock.patch.object(OLI_module, "OneLoopIntegral", _OneLoopStandIn):
            return sqla_OneLoopIntegral_factory.build(
                payload,
                s.conn,
                s.tables["OneLoopIntegral"],
                None,
                s.tables,
                None,
            )

    def test_row_tagged_in_oneloop_tags_is_found(self):
        s = self._store_with_row()
        s.insert("OneLoopIntegral_tags", parent_serial=5, tag_serial=1)
        s.conn.commit()

        obj = self._build(s)
        self.assertEqual(5, obj.store_id)

    def test_untagged_row_coinciding_with_a_quadsourceintegral_tag_serial_is_not_matched(
        self,
    ):
        s = self._store_with_row()
        # no OneLoopIntegral_tags row for serial 5 -- only a QuadSourceIntegral_tags row whose
        # parent_serial happens to equal the OneLoopIntegral row's serial (probe_06's coincidence)
        s.insert("QuadSourceIntegral_tags", parent_serial=5, tag_serial=1)
        s.conn.commit()

        obj = self._build(s)
        self.assertIsNone(obj.store_id)


class TestOneLoopIntegralReadBatch(unittest.TestCase):
    """
    §3 item 4 (P8). `read_batch` is reached as `sqla_OneLoopIntegral_factory.read_batch`, exactly
    the attribute `Datastore.object_read_batch` looks up on `self._factories[cls_name]` -- before
    the fix that attribute did not exist (`read_batch` was a module-level function, and
    `hasattr(sqla_OneLoopIntegral_factory, "read_batch")` was `False`, as
    `test_read_path_mechanics.py`'s antecedent probe recorded). With the stand-in, it must return
    the right rows untagged, tagged, and with `tags` omitted from the payload entirely.
    """

    def setUp(self):
        self.s = s = Store()
        self.addCleanup(s.close)
        s.insert("version", serial=1, label="test_integral_read_batch")
        s.insert("tolerance", serial=1, log10_tol=-10.0)
        s.insert("tolerance", serial=2, log10_tol=-8.0)
        s.insert("store_tag", serial=1, label="T")
        s.insert("redshift", serial=1, z=10.0, source=False, response=True)
        s.insert("wavenumber", serial=1, k_inv_Mpc=1.0e5, source=True, response=True)
        _exit_time_row(s, exit_serial=11, wavenumber_serial=1)
        for serial in (1, 2):
            s.insert(
                "OneLoopIntegral",
                serial=serial,
                label=f"oli-{serial}",
                model_serial=1,
                wavenumber_exit_serial=11,
                atol_serial=1,
                rtol_serial=2,
                z_response_serial=1,
                value=float(serial),
            )
        s.insert("OneLoopIntegral_tags", parent_serial=1, tag_serial=1)
        s.conn.commit()

        self.assertTrue(
            hasattr(sqla_OneLoopIntegral_factory, "read_batch"),
            "sqla_OneLoopIntegral_factory has no read_batch attribute; "
            "Datastore.object_read_batch cannot reach it",
        )

    def _read_batch(self, payload):
        with mock.patch.object(OLI_module, "OneLoopIntegral", _OneLoopStandIn):
            return sqla_OneLoopIntegral_factory.read_batch(
                payload, self.s.conn, self.s.tables["OneLoopIntegral"], self.s.tables
            )

    def _payload(self, **extra):
        payload = dict(
            model=_ModelProxy(),
            atol=tolerance(store_id=1, log10_tol=-10.0),
            rtol=tolerance(store_id=2, log10_tol=-8.0),
        )
        payload.update(extra)
        return payload

    def test_untagged_returns_both_rows(self):
        objs = self._read_batch(self._payload(tags=[]))
        self.assertEqual({1, 2}, {o.store_id for o in objs})

    def test_tagged_returns_only_the_tagged_row(self):
        objs = self._read_batch(self._payload(tags=[store_tag(store_id=1, label="T")]))
        self.assertEqual([1], [o.store_id for o in objs])

    def test_tags_omitted_behaves_as_tags_equals_empty_list(self):
        omitted = self._read_batch(self._payload())
        explicit = self._read_batch(self._payload(tags=[]))
        self.assertEqual({o.store_id for o in explicit}, {o.store_id for o in omitted})


class TestOneLoopIntegralSourceSerialStaysBroken(unittest.TestCase):
    """
    §3 item 5. D3 (README §6.2) bounded prompt 05b to `OneLoopIntegral`'s mechanical defects: the
    join, the method conversion, the return, the tags default and the `validated` insert key --
    not `source_serial`. Neither `build()` nor `read_batch()` puts a `source_serial` key in the
    payload it constructs, and the table has no such column (`[00-oneloop-factory-cannot-round-
    trip-a-row]`, this board's §3.2). The real `ComputeTargets.OneLoopIntegral` constructor reads
    `payload["source_serial"]` unconditionally, so constructing the *real* object from what
    `read_batch()` passes must still raise `KeyError`. Whoever closes
    `[00-oneloop-factory-cannot-round-trip-a-row]` must change this test.
    """

    def test_read_batch_hit_raises_key_error_on_the_real_constructor(self):
        s = Store()
        self.addCleanup(s.close)
        s.insert("version", serial=1, label="test_integral_read_batch")
        s.insert("tolerance", serial=1, log10_tol=-10.0)
        s.insert("tolerance", serial=2, log10_tol=-8.0)
        s.insert("redshift", serial=1, z=10.0, source=False, response=True)
        s.insert("wavenumber", serial=1, k_inv_Mpc=1.0e5, source=True, response=True)
        _exit_time_row(s, exit_serial=11, wavenumber_serial=1)
        s.insert(
            "OneLoopIntegral",
            serial=5,
            label="oli-5",
            model_serial=1,
            wavenumber_exit_serial=11,
            atol_serial=1,
            rtol_serial=2,
            z_response_serial=1,
            value=1.25,
        )
        s.conn.commit()

        payload = dict(
            model=_ModelProxy(),
            atol=tolerance(store_id=1, log10_tol=-10.0),
            rtol=tolerance(store_id=2, log10_tol=-8.0),
            tags=[],
        )
        with self.assertRaises(KeyError) as caught:
            sqla_OneLoopIntegral_factory.read_batch(
                payload, s.conn, s.tables["OneLoopIntegral"], s.tables
            )
        self.assertEqual("source_serial", caught.exception.args[0])


if __name__ == "__main__":
    unittest.main()
