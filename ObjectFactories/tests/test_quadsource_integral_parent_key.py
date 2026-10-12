"""
Tests for prompt 09c of ``prompts/datastore-integrity`` (board items K5, and K7 completed): a
``QuadSourceIntegral`` is keyed on the six stored rows it is computed from
(docs/datastore-integrity-audit.md §2.4) -- its ``QuadSource`` (``source_serial``), its
``GkSourcePolicyData`` (``data_serial``), and the ``TkNumericIntegration`` and
``TkWKBIntegration`` rows for q and r (``Tq_numeric_serial``, ``Tq_WKB_serial``,
``Tr_numeric_serial``, ``Tr_WKB_serial``). The integral is sharded on k; the policy data is on its
shard, with a foreign key, and the other five on q's or r's, as indexed columns with none.

The prompt's §3 tests, by class:

  1. ``TestKeyedOnEachOfItsSixParents`` -- an integral stored from six rows is found by them, and
     changing any one misses, each as its own sub-test.
  3. ``TestReadBatchHandsBackTheParents`` -- ``read_batch`` hands the six parents back on every
     object, tagged and untagged; selecting by labels alone returns every match; a parent it is
     given filters the rows.
  5. ``TestTheInventoryResolvesAllSix`` -- on the full-store fixture: two rows differing only in
     one cross-shard parent are two records with different identities, including a ``Tq`` parent
     varied to a row on another shard; a parent on no shard is a named problem; ``data`` resolves
     on the integral's own shard only.

and, by the user's decision of 2026-09-30:

  * ``TestTheQuadSourceValueModelRead`` -- ``QuadSourceValue._build_impl_model``, which joined
    every ``QuadSource`` under its labels, refuses more than one, naming their ``Tq_serial`` and
    ``Tr_serial``, and serves one. It closes
    ``[09a-quadsourcevalue-model-read-joins-every-source-under-its-labels]``.

``TestTheNewColumns`` checks the four columns' declaration. Tests 2 (the asserted relations), 4
(the extraction script's refusal) and 7 (``main.py``) are in
``ComputeTargets/tests/test_quadsource_integral_parent_main.py``.

No Ray, nothing under ``var/``: the schema is built in memory with the real factories, and the
inventory tests write stores in temporary directories with ``build_full_store``.
"""

import copy
import tempfile
import unittest
from pathlib import Path

import ray
import sqlalchemy as sqla
from sqlalchemy.exc import MultipleResultsFound

from ComputeTargets import QuadSourceIntegral
from CosmologyConcepts import redshift
from ObjectFactories.QuadSource import sqla_QuadSourceValue_factory
from ObjectFactories.QuadSourceIntegral import (
    sqla_QuadSourceIntegral_factory,
)
from datastorekit.store_inventory import read_inventory
from ObjectFactories.tests.real_store_fixtures import (
    build_full_store,
    full_rows,
    with_rows,
)
from ObjectFactories.tests.test_integral_read_batch import (
    Store,
    _exit_time_obj,
    _exit_time_row,
    _ModelProxy,
    _Ref,
)
from MetadataConcepts import store_tag, tolerance
from config.datastore import factories as _factories

ATOL = tolerance(store_id=1, log10_tol=-10.0)
RTOL = tolerance(store_id=2, log10_tol=-9.0)
Z_RESPONSE = redshift(store_id=1, z=10.0, is_source=False, is_response=True)
Z_SOURCE_MAX = redshift(store_id=2, z=1.0e5, is_source=True, is_response=False)
TAGS = {
    "A": store_tag(store_id=1, label="A"),
    "B": store_tag(store_id=2, label="B"),
}

# the six parents, by serial. None of these rows is in this store: the lookup names a parent by
# its serial, and the integral's own table holds only serials (five of the six are on other
# shards in production)
PARENTS = dict(
    source_serial=101,
    data_serial=201,
    Tq_numeric_serial=31,
    Tq_WKB_serial=41,
    Tr_numeric_serial=32,
    Tr_WKB_serial=42,
)
OTHER = {name: serial + 1000 for name, serial in PARENTS.items()}


class _WithAStore(unittest.TestCase):
    """A store with the labels a ``QuadSourceIntegral`` row needs, and its lookups."""

    def setUp(self):
        self.s = s = Store()
        self.addCleanup(s.close)

        s.insert("version", serial=1, label="test_quadsource_integral_parent_key")
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
        s.insert(
            "QuadSourcePolicy",
            serial=1,
            label="quad-policy",
            Levin_threshold=1.5,
            numeric_policy="maximize-WKB",
        )
        s.conn.commit()

        self.model = _ModelProxy()
        self.k = _exit_time_obj(11, 1, 1.0e5)
        self.q = _exit_time_obj(12, 2, 2.0e5)
        self.r = _exit_time_obj(13, 3, 3.0e5)

    def table(self):
        return self.s.tables["QuadSourceIntegral"]

    def labels(self, tags=()):
        """What a lookup names besides the parents: the physical labels and the tags."""
        return dict(
            model=self.model,
            policy=_Ref(1),
            quad_policy=_Ref(1),
            k=self.k,
            q=self.q,
            r=self.r,
            z_response=Z_RESPONSE,
            z_source_max=Z_SOURCE_MAX,
            atol=ATOL,
            rtol=RTOL,
            tags=[TAGS[n] for n in tags],
        )

    def store_row(self, parents=None, tags=(), total=1.0):
        """Store a ``QuadSourceIntegral`` computed from ``parents`` through the factory's own
        ``store()``, and return it."""
        parents = PARENTS if parents is None else parents
        labels = self.labels(tags)
        payload = {
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
            "metadata": None,
        }
        payload.update(parents)
        obj = QuadSourceIntegral(payload, **labels)
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

    def build(self, parents=None, tags=(), **changes):
        payload = self.labels(tags)
        payload.update(PARENTS if parents is None else parents)
        payload.update(changes)
        return sqla_QuadSourceIntegral_factory.build(
            payload, self.s.conn, self.table(), None, self.s.tables, None
        )

    def read_batch(self, tags=(), **parents):
        payload = self.labels(tags)
        # a reader holding only physical labels names neither z_response nor any parent
        del payload["q"], payload["r"], payload["z_response"], payload["z_source_max"]
        payload.update(q=self.q, r=self.r)
        payload.update(parents)
        return sqla_QuadSourceIntegral_factory.read_batch(
            payload, self.s.conn, self.table(), self.s.tables
        )


class TestKeyedOnEachOfItsSixParents(_WithAStore):
    """§3 test 1."""

    def test_found_by_the_rows_it_was_computed_from(self):
        stored = self.store_row()
        hit = self.build()
        self.assertTrue(hit.available)
        self.assertEqual(hit.store_id, stored.store_id)
        self.assertEqual(hit.parents, PARENTS)

    def test_changing_any_one_misses(self):
        self.store_row()
        for name in QuadSourceIntegral.PARENTS:
            with self.subTest(parent=name):
                miss = self.build(**{name: OTHER[name]})
                self.assertFalse(miss.available)
                # a miss is keyed on the parents it was asked under, and so can be computed
                self.assertEqual(miss.parents, dict(PARENTS, **{name: OTHER[name]}))

    def test_otherwise_identical_rows_under_another_parent_coexist(self):
        first = self.store_row()
        for name in QuadSourceIntegral.PARENTS:
            with self.subTest(parent=name):
                parents = dict(PARENTS, **{name: OTHER[name]})
                second = self.store_row(parents=parents)
                self.assertNotEqual(first.store_id, second.store_id)
                self.assertEqual(self.build().store_id, first.store_id)
                self.assertEqual(self.build(parents=parents).store_id, second.store_id)

    def test_the_stored_row_records_the_six(self):
        stored = self.store_row()
        table = self.table()
        row = self.s.conn.execute(
            sqla.select(
                table.c.source_serial,
                table.c.data_serial,
                table.c.Tq_numeric_serial,
                table.c.Tq_WKB_serial,
                table.c.Tr_numeric_serial,
                table.c.Tr_WKB_serial,
            ).where(table.c.serial == stored.store_id)
        ).one()
        self.assertEqual(dict(row._mapping), PARENTS)

    def test_the_tags_are_still_part_of_the_key(self):
        self.store_row(tags=("A",))
        self.assertTrue(self.build(tags=("A",)).available)
        self.assertFalse(self.build(tags=("A", "B")).available)

    def test_build_names_all_six(self):
        for name in QuadSourceIntegral.PARENTS:
            with self.subTest(parent=name):
                parents = dict(PARENTS)
                del parents[name]
                with self.assertRaises(RuntimeError) as caught:
                    self.build(parents=parents)
                self.assertIn(name, str(caught.exception))

    def test_a_parent_is_an_integer_serial(self):
        for value in (True, 31.0, _Ref(31)):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    self.build(Tq_numeric_serial=value)

    def test_store_refuses_an_object_that_does_not_name_them(self):
        with self.assertRaises(ValueError) as caught:
            self.store_row(parents=dict(PARENTS, Tr_WKB_serial=None))
        self.assertIn("Tr_WKB_serial", str(caught.exception))
        table = self.table()
        self.assertEqual(
            self.s.conn.execute(
                sqla.select(sqla.func.count()).select_from(table)
            ).scalar_one(),
            0,
        )


class TestTheNewColumns(_WithAStore):
    """The four Tk columns: NOT NULL, indexed, and no foreign key, since their rows are on q's and
    r's shards and a foreign key cannot cross a shard. ``data_serial`` keeps its foreign key, and
    ``source_serial`` has none."""

    def test_the_declarations(self):
        table = self.table()
        indexed = {c for index in table.indexes for c in index.columns.keys()}
        for name in (
            "Tq_numeric_serial",
            "Tq_WKB_serial",
            "Tr_numeric_serial",
            "Tr_WKB_serial",
            "source_serial",
        ):
            with self.subTest(column=name):
                column = table.c[name]
                self.assertFalse(column.nullable)
                self.assertEqual(column.foreign_keys, set())
                self.assertIn(name, indexed)
        (fk,) = table.c.data_serial.foreign_keys
        self.assertEqual(fk.target_fullname, "GkSourcePolicyData.serial")


class TestReadBatchHandsBackTheParents(_WithAStore):
    """§3 test 3: two rows under one set of labels, computed from different parents. One carries
    tag A and the other none."""

    def setUp(self):
        super().setUp()
        self.first = self.store_row(tags=("A",), total=1.0)
        self.second = self.store_row(parents=OTHER, total=2.0)

    def test_untagged(self):
        objs = self.read_batch()
        self.assertEqual(
            {o.store_id: o.parents for o in objs},
            {self.first.store_id: PARENTS, self.second.store_id: OTHER},
        )

    def test_tagged(self):
        (obj,) = self.read_batch(tags=("A",))
        self.assertEqual(obj.store_id, self.first.store_id)
        self.assertEqual(obj.parents, PARENTS)

    def test_the_labels_alone_select_every_match(self):
        """D1: a reader holding only physical labels is given every stored row, each with the
        parents that distinguish it, so that it can refuse more than one."""
        self.assertEqual(len(self.read_batch()), 2)

    def test_each_parent_given_filters_the_rows(self):
        for name in QuadSourceIntegral.PARENTS:
            with self.subTest(parent=name):
                (obj,) = self.read_batch(**{name: PARENTS[name]})
                self.assertEqual(obj.store_id, self.first.store_id)
                (obj,) = self.read_batch(**{name: OTHER[name]})
                self.assertEqual(obj.store_id, self.second.store_id)
                self.assertEqual(self.read_batch(**{name: 9999}), [])

    def test_a_parent_given_is_an_integer_serial(self):
        with self.assertRaises(TypeError):
            self.read_batch(source_serial=_Ref(101))


def read_the_store(replicated, sharded, keys):
    with tempfile.TemporaryDirectory() as tmp:
        return read_inventory(
            build_full_store(
                Path(tmp).resolve(),
                replicated=replicated,
                sharded=sharded,
                shard_keys=keys,
            ).primary,
            _factories,
        )


class TestTheInventoryResolvesAllSix(unittest.TestCase):
    """
    §3 test 5, on the full-store fixture. QuadSourceIntegral 1 is on shard 0 (k's). Its source,
    QuadSource 1, is on shard 0 (q's); its data, GkSourcePolicyData 1, on shard 0; its Tq rows, Tk
    numeric 1 and Tk WKB 1, on shard 0; and its Tr rows, Tk numeric 2 and Tk WKB 3, on shard 1.
    Shard 1 holds Tk numeric 2, Tk WKB 3 and QuadSource 2; shard 0 holds Tk numeric 1 and 3, Tk
    WKB 1 and 2 and QuadSource 1.

    The copy is QuadSourceIntegral 1 again, under serial 10 with its tags, with one parent changed.
    A ``Tq`` parent is varied to a row on **another** shard (the fixture's own ``Tq`` rows are all
    on their integral's shard, so only this can show that a ``Tq`` parent resolves across shards).
    """

    @classmethod
    def setUpClass(cls):
        cls.baseline = read_the_store(*full_rows())

    def with_a_copy(self, **changes):
        replicated, sharded, keys = full_rows()
        row = copy.deepcopy(sharded[0]["QuadSourceIntegral"][0])
        self.assertEqual(row["serial"], 1)
        row.update(serial=10, label="copy", **changes)
        sharded[0] = with_rows(
            sharded[0],
            {
                "QuadSourceIntegral": [row],
                "QuadSourceIntegral_tags": [
                    {"parent_serial": 10, "tag_serial": 1},
                    {"parent_serial": 10, "tag_serial": 2},
                ],
            },
        )
        return read_the_store(replicated, sharded, keys)

    def test_the_fixture_has_every_parent_resolved(self):
        qsi = self.baseline["QuadSourceIntegral"]
        self.assertEqual(qsi.problems, ())
        self.assertEqual(qsi.count, 3)
        for record in qsi.records:
            resolved = self.baseline.resolve("QuadSourceIntegral", record)
            for field in ("source", "data", "Tq_numeric", "Tq_WKB", "Tr_numeric"):
                self.assertIn(field, record.key)
                self.assertIn("key", resolved[field], field)
            self.assertIn("key", resolved["Tr_WKB"])

    def test_two_rows_differing_only_in_one_cross_shard_parent(self):
        for field, column, value in (
            # to rows on shard 1, another shard from the integral's
            ("Tq_numeric", "Tq_numeric_serial", 2),
            ("Tq_WKB", "Tq_WKB_serial", 3),
            ("source", "source_serial", 2),
            # QuadSourceIntegral 1's own Tr rows are on shard 1; these are on shard 0
            ("Tr_numeric", "Tr_numeric_serial", 1),
            ("Tr_WKB", "Tr_WKB_serial", 1),
        ):
            with self.subTest(field=field):
                inv = self.with_a_copy(**{column: value})
                qsi = inv["QuadSourceIntegral"]
                self.assertEqual(qsi.problems, ())
                self.assertEqual(qsi.count, 4)
                pairs = [
                    (a, b)
                    for i, a in enumerate(qsi.records)
                    for b in qsi.records[i + 1 :]
                    if {f for f in a.key if a.key[f] != b.key[f]} == {field}
                ]
                self.assertEqual(len(pairs), 1)
                ((one, two),) = pairs
                self.assertNotEqual(one.identity(), two.identity())
                self.assertNotEqual(
                    inv.digest_of("QuadSourceIntegral", one),
                    inv.digest_of("QuadSourceIntegral", two),
                )

    def test_a_tq_parent_on_another_shard_resolves_to_that_rows_identity(self):
        # QuadSourceIntegral 3 (shard 1) has Tk numeric 2, on its own shard, as its Tq numeric
        # row. The copy of QuadSourceIntegral 1 (shard 0) names the same row across shards, and
        # resolves to the same identity
        inv = self.with_a_copy(Tq_numeric_serial=2)
        by_q = {}
        for record in inv["QuadSourceIntegral"].records:
            by_q.setdefault(record.key["q"], []).append(record)
        # QuadSourceIntegral 3 is alone at its q; 1, 2 and the copy share theirs
        ((third,),) = [records for records in by_q.values() if len(records) == 1]
        (at_q,) = [records for records in by_q.values() if len(records) == 3]
        (copied,) = [
            r
            for r in at_q
            if [o.key["Tq_numeric"] for o in at_q].count(r.key["Tq_numeric"]) == 1
        ]
        self.assertEqual(copied.key["Tq_numeric"], third.key["Tq_numeric"])

    def test_a_tk_parent_on_no_shard_is_a_named_unresolved_parent(self):
        inv = self.with_a_copy(Tq_WKB_serial=99)
        qsi = inv["QuadSourceIntegral"]
        self.assertEqual(qsi.count, 3, "the row is not a record")
        (problem,) = qsi.problems
        self.assertTrue(problem.startswith("unresolved-parent: QuadSourceIntegral:"))
        self.assertIn("Tq_WKB", problem)

    def test_data_resolves_on_the_integrals_own_shard_only(self):
        # GkSourcePolicyData 3 exists, on shard 1, and is no parent for an integral on shard 0:
        # data is the same-shard form, with a foreign key
        inv = self.with_a_copy(data_serial=3)
        qsi = inv["QuadSourceIntegral"]
        self.assertEqual(qsi.count, 3)
        (problem,) = qsi.problems
        self.assertIn("unresolved-parent", problem)
        self.assertIn("data", problem)

    def test_the_same_parents_are_a_duplicate(self):
        # the control: the copy with nothing changed has the same identity as QuadSourceIntegral 1
        inv = self.with_a_copy()
        qsi = inv["QuadSourceIntegral"]
        self.assertEqual(qsi.count, 4)
        self.assertEqual([p.split(":")[0] for p in qsi.problems], ["duplicate"])


class TestTheQuadSourceValueModelRead(unittest.TestCase):
    """
    ``QuadSourceValue._build_impl_model`` reads one value by physical labels -- model, q, r,
    ``validated`` and tags -- and joined the value table to **every** matching QuadSource. Since
    prompt 09a two can share those labels, one per ``(Tq_serial, Tr_serial)``. By the user's
    decision (2026-09-30) it now refuses more than one, naming their ``Tq_serial`` and
    ``Tr_serial``, and serves one: ``[09a-quadsourcevalue-model-read-joins-every-source-under-its-
    labels]``.
    """

    def setUp(self):
        from ObjectFactories.tests.test_quadsource_parent_key import (
            Q_EXIT,
            R_EXIT,
            TAG,
            _base_store,
            store_source,
        )
        from ObjectFactories.tests.test_wkb_numeric_parent_key import MODEL

        self.s = _base_store()
        self.addCleanup(self.s.close)
        self.store_source = lambda Tq, Tr: store_source(self.s, Tq, Tr)
        self.payload = {
            "model": MODEL,
            "q": Q_EXIT,
            "r": R_EXIT,
            "z": redshift(store_id=2, z=10.0, is_source=False, is_response=True),
            "tags": [TAG],
        }

    def read(self):
        return sqla_QuadSourceValue_factory.build(
            dict(self.payload),
            self.s.conn,
            self.s.tables["QuadSourceValue"],
            self.s.inserters["QuadSourceValue"],
            self.s.tables,
            self.s.inserters,
        )

    def value_row(self, parent: int, serial: int, source: float):
        self.s.insert(
            "QuadSourceValue",
            serial=serial,
            parent_serial=parent,
            z_serial=2,
            source=source,
            undiff=0.0,
            diff=0.0,
        )
        self.s.conn.commit()

    def test_one_match_serves_its_value(self):
        first = self.store_source(1, 2)
        self.value_row(first, 801, 11.0)
        found = self.read()
        self.assertEqual(found.store_id, 801)
        self.assertEqual(found.source, 11.0)

    def test_two_matches_are_refused_naming_their_parents(self):
        first = self.store_source(1, 2)
        second = self.store_source(2, 1)
        self.value_row(first, 801, 11.0)
        self.value_row(second, 802, 22.0)
        with self.assertRaises(MultipleResultsFound) as caught:
            self.read()
        message = str(caught.exception)
        self.assertIn(f"serial={first}: Tq_serial=1, Tr_serial=2", message)
        self.assertIn(f"serial={second}: Tq_serial=2, Tr_serial=1", message)

    def test_two_matches_are_refused_even_with_one_value(self):
        """The refusal is of the sources, not of the joined rows: it does not depend on which of
        them happens to hold a value at z."""
        first = self.store_source(1, 2)
        self.store_source(2, 1)
        self.value_row(first, 801, 11.0)
        with self.assertRaises(MultipleResultsFound):
            self.read()

    def test_no_match_is_a_miss(self):
        self.assertFalse(self.read().available)


def tearDownModule():
    assert not ray.is_initialized()


if __name__ == "__main__":
    unittest.main()
