"""
Tests for prompt 08 of ``prompts/datastore-integrity`` (board item K3): a ``QuadSourceIntegral`` is
keyed on the ``QuadSourcePolicy`` it was made under (docs/datastore-integrity-audit.md §2.5;
decisions 6.1.2 and D6). The column is ``quad_policy_serial``, NOT NULL, indexed, with a foreign
key to the replicated ``QuadSourcePolicy`` table. No computation reads the policy's fields, so the
key changes the record and no number.

The prompt's §3 tests, by class:

  1. ``TestKeyedOnThePolicy`` -- two ``QuadSourcePolicy`` rows; an integral stored under the first
     is found under the first and missed under the second, all else equal.
  2. ``TestReadBatchFiltersOnIt`` -- ``read_batch`` filters on it, tagged and untagged.
  4. ``TestTheInventoryIdentityIncludesIt`` -- two otherwise identical rows under the two policies
     are two records (and, under one policy, the inventory reports a duplicate); the fingerprint
     moves with the policy. (Test 3, the constructor, and test 5, ``main.py``, are in
     ``ComputeTargets/tests/test_quadsource_policy_main.py``.)

and, from the orchestrator note:

  * ``TestAnotherDatabaseErrorPassesThrough`` -- a database error raised by ``build()``'s or
    ``read_batch()``'s own statement reaches the caller unchanged;
  * ``TestReadTable`` -- the ``read_table`` the extraction script finds its policy with (§0.1): it
    matches as ``build()`` does, returns every match, and writes nothing.

The foreign key itself is checked by ``test_quadsource_policy_foreign_key.py``.

No Ray, nothing under ``var/``: the schema is built in memory with
``build_schema(MetaData(), config.datastore.factories)``, and every call is to the real
factory code. Parents that exist only as a serial -- a model, a policy, a tag -- are stand-ins
carrying ``.store_id``.
"""

import unittest
from pathlib import Path

import ray
import sqlalchemy as sqla

from ComputeTargets import QuadSourceIntegral
from CosmologyConcepts import redshift
from ObjectFactories.QuadSourceIntegral import (
    sqla_QuadSourceIntegral_factory,
)
from ObjectFactories.QuadSourcePolicy import (
    sqla_QuadSourcePolicy_factory,
)
from StoreFixtures import real_store_fixtures as rsf
from ObjectFactories.tests import test_database_errors_reach_the_caller as harness
from ObjectFactories.tests.test_integral_read_batch import (
    Store,
    _exit_time_obj,
    _exit_time_row,
    _ModelProxy,
    _Ref,
)
from MetadataConcepts import QuadSourcePolicy, store_tag, tolerance
from config.datastore import factories as _factories

ATOL = tolerance(store_id=1, log10_tol=-10.0)
RTOL = tolerance(store_id=2, log10_tol=-9.0)
Z_RESPONSE = redshift(store_id=1, z=10.0, is_source=False, is_response=True)
Z_SOURCE_MAX = redshift(store_id=2, z=1.0e5, is_source=True, is_response=False)
TAGS = {
    "A": store_tag(store_id=1, label="A"),
    "B": store_tag(store_id=2, label="B"),
}
PARENTS = dict(
    source_serial=101,
    data_serial=201,
    Tq_numeric_serial=31,
    Tq_WKB_serial=41,
    Tr_numeric_serial=32,
    Tr_WKB_serial=42,
)


class _Case(unittest.TestCase):
    """A store with the parents a ``QuadSourceIntegral`` row needs, two ``QuadSourcePolicy``
    rows (serials 1 and 2), and the payload of a lookup under either."""

    def setUp(self):
        self.s = s = Store()
        self.addCleanup(s.close)

        s.insert("version", serial=1, label="test_quadsource_policy_key")
        s.insert("tolerance", serial=1, log10_tol=-10.0)
        s.insert("tolerance", serial=2, log10_tol=-9.0)
        s.insert("store_tag", serial=1, label="A")
        s.insert("store_tag", serial=2, label="B")
        s.insert("redshift", serial=1, z=10.0, source=False, response=True)
        s.insert("redshift", serial=2, z=1.0e5, source=True, response=False)
        for serial, k in ((1, 1.0e5), (2, 2.0e5), (3, 3.0e5)):
            s.insert(
                "wavenumber", serial=serial, k_inv_Mpc=k, source=True, response=True
            )
        for exit_serial, wn_serial in ((11, 1), (12, 2), (13, 3)):
            _exit_time_row(s, exit_serial, wn_serial)
        s.insert(
            "GkSourcePolicy",
            serial=1,
            label="gk-policy",
            Levin_threshold=1.5,
            numeric_policy="maximize-WKB",
        )
        for serial, threshold, label in ((1, 1.5, "1.5"), (2, 5.0, "5.0")):
            s.insert(
                "QuadSourcePolicy",
                serial=serial,
                label=f"quad-policy-{label}",
                Levin_threshold=threshold,
                numeric_policy="maximize-WKB",
            )
        s.conn.commit()

        self.model = _ModelProxy()
        self.k = _exit_time_obj(11, 1, 1.0e5)
        self.q = _exit_time_obj(12, 2, 2.0e5)
        self.r = _exit_time_obj(13, 3, 3.0e5)

    def table(self):
        return self.s.tables["QuadSourceIntegral"]

    def payload(self, quad_policy=1, tags=()):
        """A ``build()`` payload: every key the lookup names."""
        return dict(
            model=self.model,
            policy=_Ref(1),
            quad_policy=_Ref(quad_policy),
            k=self.k,
            q=self.q,
            r=self.r,
            z_response=Z_RESPONSE,
            z_source_max=Z_SOURCE_MAX,
            atol=ATOL,
            rtol=RTOL,
            tags=[TAGS[n] for n in tags],
            # the six rows the integral is computed from, which it is keyed on since prompt 09c:
            # the ones store_row's object carries
            **PARENTS,
        )

    def store_row(self, quad_policy=1, tags=(), total=1.0):
        """Store a ``QuadSourceIntegral`` under ``quad_policy`` through the factory's own
        ``store()``, and return it."""
        obj = QuadSourceIntegral(
            {
                "store_id": None,
                "b": 0.0,
                "total": total,
                "total_abserr": None,
                "total_converged": None,
                "total_phase_limited": None,
                "numeric_quad": 0.25,
                "WKB_Levin": 0.75,
                "analytic_rad": None,
                "eta_source_max": None,
                "eta_response": None,
                "numeric_quad_data": None,
                "WKB_Levin_data": None,
                "WKB_phase_spline_chunks": None,
                "compute_time": None,
                "analytic_compute_time": None,
                "source_serial": 101,
                "data_serial": 201,
                "Tq_numeric_serial": 31,
                "Tq_WKB_serial": 41,
                "Tr_numeric_serial": 32,
                "Tr_WKB_serial": 42,
                "metadata": None,
            },
            model=self.model,
            policy=_Ref(1),
            quad_policy=_Ref(quad_policy),
            z_response=Z_RESPONSE,
            z_source_max=Z_SOURCE_MAX,
            k=self.k,
            q=self.q,
            r=self.r,
            atol=ATOL,
            rtol=RTOL,
            tags=[TAGS[n] for n in tags],
        )
        inserters = {
            name: (lambda conn, data, name=name: self.s.insert(name, **data))
            for name in ("QuadSourceIntegral", "QuadSourceIntegral_tags")
        }
        stored = sqla_QuadSourceIntegral_factory.store(
            obj,
            self.s.conn,
            self.table(),
            inserters["QuadSourceIntegral"],
            self.s.tables,
            inserters,
        )
        self.s.conn.commit()
        return stored

    def build(self, **kwargs):
        return sqla_QuadSourceIntegral_factory.build(
            self.payload(**kwargs), self.s.conn, self.table(), None, self.s.tables, None
        )

    def read_batch(self, **kwargs):
        return sqla_QuadSourceIntegral_factory.read_batch(
            self.payload(**kwargs), self.s.conn, self.table(), self.s.tables
        )


class TestKeyedOnThePolicy(_Case):
    """§3 test 1."""

    def test_found_under_the_first_and_missed_under_the_second(self):
        stored = self.store_row(quad_policy=1)

        hit = self.build(quad_policy=1)
        self.assertTrue(hit.available)
        self.assertEqual(hit.store_id, stored.store_id)

        miss = self.build(quad_policy=2)
        self.assertFalse(miss.available)

    def test_the_stored_row_records_its_policy(self):
        first = self.store_row(quad_policy=1)
        second = self.store_row(quad_policy=2)
        table = self.table()
        rows = dict(
            self.s.conn.execute(
                sqla.select(table.c.serial, table.c.quad_policy_serial)
            ).all()
        )
        self.assertEqual(rows, {first.store_id: 1, second.store_id: 2})

    def test_the_same_computation_under_both_policies_coexists(self):
        """Otherwise identical rows are two rows, and each lookup finds its own: no
        ``MultipleResultsFound`` (which is what the key not being there would raise)."""
        first = self.store_row(quad_policy=1)
        second = self.store_row(quad_policy=2)
        self.assertNotEqual(first.store_id, second.store_id)
        self.assertEqual(self.build(quad_policy=1).store_id, first.store_id)
        self.assertEqual(self.build(quad_policy=2).store_id, second.store_id)

    def test_a_hit_and_a_miss_carry_the_policy_they_were_asked_under(self):
        self.store_row(quad_policy=1)
        self.assertEqual(self.build(quad_policy=1).quad_policy.store_id, 1)
        self.assertEqual(self.build(quad_policy=2).quad_policy.store_id, 2)

    def test_build_requires_the_policy(self):
        payload = self.payload()
        del payload["quad_policy"]
        with self.assertRaises(KeyError):
            sqla_QuadSourceIntegral_factory.build(
                payload, self.s.conn, self.table(), None, self.s.tables, None
            )


class TestReadBatchFiltersOnIt(_Case):
    """§3 test 2: tagged and untagged."""

    def setUp(self):
        super().setUp()
        # two rows for one computation under the two policies, both tagged {A}; a third under the
        # first policy only, tagged {A, B}
        self.first = self.store_row(quad_policy=1, tags=("A",), total=1.0)
        self.second = self.store_row(quad_policy=2, tags=("A",), total=2.0)
        self.first_ab = self.store_row(quad_policy=1, tags=("A", "B"), total=3.0)

    def serials(self, **kwargs):
        return {o.store_id for o in self.read_batch(**kwargs)}

    def test_untagged(self):
        self.assertEqual(
            self.serials(quad_policy=1), {self.first.store_id, self.first_ab.store_id}
        )
        self.assertEqual(self.serials(quad_policy=2), {self.second.store_id})

    def test_tagged(self):
        self.assertEqual(
            self.serials(quad_policy=1, tags=("A",)),
            {self.first.store_id, self.first_ab.store_id},
        )
        self.assertEqual(
            self.serials(quad_policy=2, tags=("A",)), {self.second.store_id}
        )
        self.assertEqual(
            self.serials(quad_policy=1, tags=("A", "B")), {self.first_ab.store_id}
        )
        self.assertEqual(self.serials(quad_policy=2, tags=("A", "B")), set())

    def test_a_policy_with_no_rows_finds_none(self):
        self.assertEqual(self.serials(quad_policy=3), set())

    def test_the_objects_carry_the_policy_they_were_read_under(self):
        for quad_policy in (1, 2):
            for obj in self.read_batch(quad_policy=quad_policy):
                self.assertEqual(obj.quad_policy.store_id, quad_policy)

    def test_read_batch_requires_the_policy(self):
        payload = self.payload()
        del payload["quad_policy"]
        with self.assertRaises(KeyError):
            sqla_QuadSourceIntegral_factory.read_batch(
                payload, self.s.conn, self.table(), self.s.tables
            )


class TestAnotherDatabaseErrorPassesThrough(_Case, harness._PassThroughCase):
    """
    A database error raised by either reader's real statement reaches the caller as itself. This
    was ``TestAnOldStoreIsRefused.test_another_database_error_is_not_turned_into_it``, on a table
    recreated from the 07 witness's DDL; the table is now the current one, and the error is raised
    by ``test_database_errors_reach_the_caller``'s connection, which hands back the compiled text
    of the query the factory issued (so the statement names ``quad_policy_serial``) rather than the
    literal ``"SELECT 1"``.
    """

    def test_another_database_error_is_not_turned_into_it(self):
        for label, call in (
            (
                "build",
                lambda conn: sqla_QuadSourceIntegral_factory.build(
                    self.payload(), conn, self.table(), None, self.s.tables, None
                ),
            ),
            (
                "read_batch",
                lambda conn: list(
                    sqla_QuadSourceIntegral_factory.read_batch(
                        self.payload(), conn, self.table(), self.s.tables
                    )
                ),
            ),
        ):
            with self.subTest(function=label):
                conn = harness._StatementRaisingConnection()
                self.assert_passes_through(
                    lambda: call(conn), conn, ("quad_policy_serial",)
                )
                self.assertEqual(1, conn.executed)


class TestTheInventoryIdentityIncludesIt(unittest.TestCase):
    """§3 test 4, on a store ``real_store_fixtures.build_full_store`` writes. The added row is the
    fixture's ``QuadSourceIntegral`` 1 again -- same model, policy, wavenumbers, redshifts, tolerances
    and tags -- under ``quad_policy_serial`` 1 or 2 (the fixture has two policies)."""

    def setUp(self):
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self._count = 0

    def read_the_store(self, added_quad_policy):
        import copy

        from datastorekit.store_inventory import read_inventory

        replicated, sharded, keys = rsf.full_rows()
        row = copy.deepcopy(sharded[0]["QuadSourceIntegral"][0])
        self.assertEqual(row["quad_policy_serial"], 1)
        row.update(serial=10, label="fixture-qsi-added")
        row["quad_policy_serial"] = added_quad_policy
        sharded[0] = rsf.with_rows(
            sharded[0],
            {
                "QuadSourceIntegral": [row],
                # the tags of QuadSourceIntegral 1
                "QuadSourceIntegral_tags": [
                    {"parent_serial": 10, "tag_serial": 1},
                    {"parent_serial": 10, "tag_serial": 2},
                ],
            },
        )
        self._count += 1
        directory = self.root / f"store-{self._count}"
        directory.mkdir()
        store = rsf.build_full_store(
            directory, replicated=replicated, sharded=sharded, shard_keys=keys
        )
        return read_inventory(store.primary, _factories)

    def test_the_key_carries_the_policy(self):
        records = self.read_the_store(2)["QuadSourceIntegral"].records
        self.assertTrue(all("quad_policy" in record.key for record in records))

    def test_two_otherwise_identical_rows_under_two_policies_are_two_records(self):
        cls = self.read_the_store(2)["QuadSourceIntegral"]
        self.assertEqual(cls.count, 4)
        self.assertEqual(cls.problems, ())
        # the two added to QuadSourceIntegral 1 differ in exactly one key field
        keys = [record.key for record in cls.records]
        differing = [
            (a, b)
            for i, a in enumerate(keys)
            for b in keys[i + 1 :]
            if {f for f in a if a[f] != b[f]} == {"quad_policy"}
        ]
        self.assertEqual(len(differing), 1)

    def test_the_same_policy_would_be_a_duplicate(self):
        """The control: the same two rows under one policy share an identity, which the
        inventory reports. So the test above passes because of the policy, not for want of a
        check."""
        cls = self.read_the_store(1)["QuadSourceIntegral"]
        self.assertEqual(cls.count, 4)
        self.assertEqual(len(cls.problems), 1)
        self.assertTrue(cls.problems[0].startswith("duplicate: QuadSourceIntegral"))

    def test_the_fingerprint_moves_with_the_policy(self):
        from RunRegistry.stores import fingerprint_of

        under_1 = fingerprint_of(self.read_the_store(1))["classes"][
            "QuadSourceIntegral"
        ]
        under_2 = fingerprint_of(self.read_the_store(2))["classes"][
            "QuadSourceIntegral"
        ]
        again_2 = fingerprint_of(self.read_the_store(2))["classes"][
            "QuadSourceIntegral"
        ]
        self.assertEqual(under_2, again_2)
        self.assertNotEqual(under_1["digest"], under_2["digest"])


class TestReadTable(_Case):
    """§0.1: the lookup the extraction script makes. It matches as ``build()`` does, returns every
    match in serial order, and inserts nothing."""

    def read(self, **kwargs):
        return sqla_QuadSourcePolicy_factory.read_table(
            self.s.conn, self.s.tables["QuadSourcePolicy"], **kwargs
        )

    def test_it_finds_the_policy_by_its_parameters(self):
        found = self.read(Levin_threshold=1.5, numeric_policy="maximize-WKB")
        self.assertEqual([p.store_id for p in found], [1])
        self.assertIsInstance(found[0], QuadSourcePolicy)
        self.assertEqual(found[0].Levin_threshold, 1.5)
        self.assertEqual(found[0].numeric_policy, "maximize-WKB")
        self.assertEqual(found[0].label, "quad-policy-1.5")

        found = self.read(Levin_threshold=5.0, numeric_policy="maximize-WKB")
        self.assertEqual([p.store_id for p in found], [2])

    def test_no_match_is_an_empty_list_and_inserts_nothing(self):
        table = self.s.tables["QuadSourcePolicy"]
        before = self.s.conn.execute(sqla.select(sqla.func.count()).select_from(table))
        before = before.scalar()
        self.assertEqual(
            self.read(Levin_threshold=2.5, numeric_policy="maximize-WKB"), []
        )
        self.assertEqual(
            self.read(Levin_threshold=1.5, numeric_policy="maximize_numeric"), []
        )
        after = self.s.conn.execute(sqla.select(sqla.func.count()).select_from(table))
        self.assertEqual(after.scalar(), before)

    def test_the_threshold_is_matched_as_build_matches_it(self):
        """Within ``DEFAULT_FLOAT_PRECISION``, not exactly; and not beyond it."""
        from config.defaults import DEFAULT_FLOAT_PRECISION

        inside = 1.5 + 0.5 * DEFAULT_FLOAT_PRECISION
        outside = 1.5 + 2.0 * DEFAULT_FLOAT_PRECISION
        self.assertEqual(
            [
                p.store_id
                for p in self.read(
                    Levin_threshold=inside, numeric_policy="maximize-WKB"
                )
            ],
            [1],
        )
        self.assertEqual(
            self.read(Levin_threshold=outside, numeric_policy="maximize-WKB"), []
        )

    def test_every_match_is_returned_in_serial_order(self):
        """Two rows build() would call the same policy. A reader must be able to refuse them."""
        self.s.insert(
            "QuadSourcePolicy",
            serial=7,
            label="duplicate",
            Levin_threshold=1.5,
            numeric_policy="maximize-WKB",
        )
        self.s.insert(
            "QuadSourcePolicy",
            serial=5,
            label="another",
            Levin_threshold=1.5,
            numeric_policy="maximize-WKB",
        )
        found = self.read(Levin_threshold=1.5, numeric_policy="maximize-WKB")
        self.assertEqual([p.store_id for p in found], [1, 5, 7])


def tearDownModule():
    assert not ray.is_initialized()


if __name__ == "__main__":
    unittest.main()
