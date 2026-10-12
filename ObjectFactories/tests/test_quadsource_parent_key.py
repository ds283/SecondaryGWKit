"""
Tests for prompt 09a of ``prompts/datastore-integrity`` (board items K4, and K7's cross-shard
parent): a ``QuadSource`` is keyed on the two ``TkNumericIntegration`` rows it is computed from
(docs/datastore-integrity-audit.md §2.4), and the store inventory resolves a parent that is on
another shard.

A ``QuadSource`` is computed from exactly two rows, the Tk numeric rows for ``q`` and ``r``, and
``store()`` has always recorded their serials in ``Tq_serial`` and ``Tr_serial``. Its lookup named
neither, so a change to the Tk rows under the same tags served the old source as a match. Now
``build()`` filters on both. ``Tq`` is on the source's shard (a foreign key since prompt 06); ``Tr``
is on ``r``'s shard, so it is an indexed column with no foreign key, and the inventory resolves it
against the Tk records of every shard.

The prompt's §3 tests, by class:

  1. ``TestKeyedOnTq`` -- a source stored from one Tk numeric row for ``q`` is found by it and
     missed by another.
  2. ``TestKeyedOnTrAcrossShards`` -- the same for ``r``, whose row is on another shard: no local
     row, no foreign key.
  3. ``TestTheReaderMode`` -- ``labels_only`` refuses two matches, naming both pairs of serials;
     serves one without being given a parent; a lookup naming neither the parents nor the mode
     raises.
  5. ``TestQEqualsR`` -- the diagonal, where ``Tq`` and ``Tr`` are the same row, in the lookup and
     in the inventory (the assertion's half is in ``ComputeTargets/tests``).
  6. ``TestTheInventoryResolvesAcrossShards`` -- two otherwise identical rows differing only in
     ``Tr`` are two records with different identities; a ``Tr`` on no shard is a named
     ``unresolved-parent`` problem. ``TestCrossShardParentUnit`` and ``TestMergeDigests`` test
     the facility itself.

Tests 4 (the compute-time assertion) and 8 (``main.py``) are in
``ComputeTargets/tests/test_quadsource_parent_main.py``.

No Ray, nothing under ``var/``: the schema is built in memory with the real factories, and the
inventory tests write stores in temporary directories with ``build_full_store``.
"""

import tempfile
import unittest
from pathlib import Path

import ray
import sqlalchemy as sqla
from sqlalchemy.exc import IntegrityError, MultipleResultsFound

from ComputeTargets import QuadSource
from ObjectFactories.QuadSource import sqla_QuadSource_factory
from datastorekit.store_inventory import (
    Parent,
    ShardContext,
    _merge_digests,
    read_inventory,
    read_records,
)
from StoreFixtures.real_store_fixtures import (
    _tk_numeric,
    build_full_store,
    full_rows,
    with_rows,
)
from ObjectFactories.tests.test_wkb_numeric_parent_key import (
    MODEL,
    Store,
    _base_store,
    _exit_time,
    _Ref,
    Z_EXIT,
)
from config.datastore import factories as _factories

Q_EXIT = _exit_time(serial=11, z_exit=Z_EXIT)
R_EXIT = _exit_time(serial=12, z_exit=2.0 * Z_EXIT)
TAG = _Ref(1)
OTHER_TAG = _Ref(2)


def source_lookup(s: Store, q=Q_EXIT, r=R_EXIT, tags=(TAG,), **extra):
    """The lookup ``main.py`` makes: the labels, the tags and whichever parents ``extra`` names."""
    payload = {
        "model": MODEL,
        "q": q,
        "r": r,
        "z_sample": None,
        "tags": list(tags),
        "_do_not_populate": True,
    }
    payload.update(extra)
    return sqla_QuadSource_factory.build(
        payload,
        s.conn,
        s.tables["QuadSource"],
        s.inserters["QuadSource"],
        s.tables,
        s.inserters,
    )


def keyed(s: Store, Tq_serial, Tr_serial, **kwargs):
    return source_lookup(s, Tq_serial=Tq_serial, Tr_serial=Tr_serial, **kwargs)


def store_source(s: Store, Tq_serial, Tr_serial, q=Q_EXIT, r=R_EXIT, tags=(TAG,)):
    """A validated, empty QuadSource stored through the real factory. Returns its serial."""
    obj = QuadSource(
        payload=None,
        model=MODEL,
        z_sample=None,
        q=q,
        r=r,
        tags=list(tags),
        Tq_serial=Tq_serial,
        Tr_serial=Tr_serial,
    )
    obj._values = []
    obj._compute_time = 1.0
    sqla_QuadSource_factory.store(
        obj,
        s.conn,
        s.tables["QuadSource"],
        s.inserters["QuadSource"],
        s.tables,
        s.inserters,
    )
    table = s.tables["QuadSource"]
    s.conn.execute(
        sqla.update(table).where(table.c.serial == obj.store_id).values(validated=True)
    )
    s.conn.commit()
    return obj.store_id


class _WithAStore(unittest.TestCase):
    def setUp(self):
        self.s = _base_store()
        self.addCleanup(self.s.close)


class TestKeyedOnTq(_WithAStore):
    """§3 test 1: two Tk numeric rows for q (serials 1 and 2 of the fixture)."""

    def test_found_by_the_row_it_was_computed_from_and_missed_by_another(self):
        stored = store_source(self.s, Tq_serial=1, Tr_serial=5)

        found = keyed(self.s, 1, 5)
        self.assertTrue(found.available)
        self.assertEqual(found.store_id, stored)
        self.assertEqual((found.Tq_serial, found.Tr_serial), (1, 5))

        missed = keyed(self.s, 2, 5)
        self.assertFalse(missed.available)

    def test_a_miss_carries_the_parents_it_was_asked_under(self):
        missed = keyed(self.s, 2, 5)
        self.assertEqual((missed.Tq_serial, missed.Tr_serial), (2, 5))

    def test_the_same_labels_under_two_parents_coexist_and_each_is_found(self):
        first = store_source(self.s, Tq_serial=1, Tr_serial=5)
        second = store_source(self.s, Tq_serial=2, Tr_serial=5)
        self.assertNotEqual(first, second)
        self.assertEqual(keyed(self.s, 1, 5).store_id, first)
        self.assertEqual(keyed(self.s, 2, 5).store_id, second)

    def test_the_stored_row_records_the_serials_it_was_keyed_on(self):
        stored = store_source(self.s, Tq_serial=1, Tr_serial=5)
        table = self.s.tables["QuadSource"]
        row = self.s.conn.execute(
            sqla.select(table.c.Tq_serial, table.c.Tr_serial).where(
                table.c.serial == stored
            )
        ).one()
        self.assertEqual((row.Tq_serial, row.Tr_serial), (1, 5))

    def test_the_tags_are_still_part_of_the_key(self):
        store_source(self.s, Tq_serial=1, Tr_serial=5)
        self.assertFalse(keyed(self.s, 1, 5, tags=(TAG, OTHER_TAG)).available)


class TestKeyedOnTrAcrossShards(_WithAStore):
    """§3 test 2: r is on another shard. Its Tk row is not in this shard's table, and the column
    that names it has no foreign key, but serials are unique across shards, so it still keys.
    """

    def test_the_row_is_on_no_table_of_this_shard(self):
        local = self.s.conn.execute(
            sqla.text('SELECT serial FROM "TkNumericIntegration" ORDER BY serial')
        ).scalars()
        self.assertEqual(list(local), [1, 2])
        # Tk rows 5 and 6 are r's, on r's shard, which this store does not hold

    def test_found_by_r_and_missed_by_another_row_for_r(self):
        stored = store_source(self.s, Tq_serial=1, Tr_serial=5)
        self.assertEqual(keyed(self.s, 1, 5).store_id, stored)
        self.assertFalse(keyed(self.s, 1, 6).available)

    def test_there_is_no_foreign_key_to_hold_it(self):
        # the store enforces its foreign keys: Tr_serial=5 names no local row and is accepted, and
        # Tq_serial, which has a foreign key, is not
        self.s.conn.exec_driver_sql("PRAGMA foreign_keys = ON")
        store_source(self.s, Tq_serial=1, Tr_serial=5)
        with self.assertRaises(IntegrityError):
            store_source(self.s, Tq_serial=99, Tr_serial=5)
        self.s.conn.rollback()

    def test_tr_is_indexed(self):
        table = self.s.tables["QuadSource"]
        indexed = {c.name for index in table.indexes for c in index.columns}
        self.assertIn("Tr_serial", indexed)
        self.assertIn("Tq_serial", indexed)
        self.assertEqual(
            [fk.target_fullname for fk in table.c.Tr_serial.foreign_keys], []
        )
        self.assertEqual(
            [fk.target_fullname for fk in table.c.Tq_serial.foreign_keys],
            ["TkNumericIntegration.serial"],
        )


class TestTheReaderMode(_WithAStore):
    """§3 test 3: a reader that holds only physical labels may select by them, and is refused
    when more than one stored row matches."""

    def test_two_matches_are_refused_naming_both_pairs(self):
        store_source(self.s, Tq_serial=1, Tr_serial=5)
        store_source(self.s, Tq_serial=2, Tr_serial=5)
        with self.assertRaises(MultipleResultsFound) as caught:
            source_lookup(self.s, labels_only=True)
        message = str(caught.exception)
        self.assertIn("Tq_serial=1, Tr_serial=5", message)
        self.assertIn("Tq_serial=2, Tr_serial=5", message)

    def test_two_matches_differing_only_in_tr_are_refused_naming_them(self):
        store_source(self.s, Tq_serial=1, Tr_serial=5)
        store_source(self.s, Tq_serial=1, Tr_serial=6)
        with self.assertRaises(MultipleResultsFound) as caught:
            source_lookup(self.s, labels_only=True)
        self.assertIn("Tq_serial=1, Tr_serial=5", str(caught.exception))
        self.assertIn("Tq_serial=1, Tr_serial=6", str(caught.exception))

    def test_one_match_is_returned_without_being_given_any_parent(self):
        stored = store_source(self.s, Tq_serial=2, Tr_serial=6)
        found = source_lookup(self.s, labels_only=True)
        self.assertTrue(found.available)
        self.assertEqual(found.store_id, stored)
        self.assertEqual((found.Tq_serial, found.Tr_serial), (2, 6))

    def test_no_match_is_a_labels_only_miss(self):
        missed = source_lookup(self.s, labels_only=True)
        self.assertFalse(missed.available)
        self.assertIsNone(missed.Tq_serial)
        self.assertIsNone(missed.Tr_serial)

    def test_the_tags_still_select_under_the_mode(self):
        store_source(self.s, Tq_serial=1, Tr_serial=5, tags=(TAG,))
        store_source(self.s, Tq_serial=2, Tr_serial=5, tags=(TAG, OTHER_TAG))
        # each is found by its own tag set, so neither is ambiguous with the other under the
        # tags that only one of them carries; the tag join is a superset match, so TAG alone
        # matches both and is refused
        self.assertTrue(
            source_lookup(self.s, tags=(OTHER_TAG,), labels_only=True).available
        )
        with self.assertRaises(MultipleResultsFound):
            source_lookup(self.s, tags=(TAG,), labels_only=True)

    def test_a_lookup_naming_neither_the_parents_nor_the_mode_raises(self):
        with self.assertRaises(RuntimeError) as caught:
            source_lookup(self.s)
        self.assertIn("labels_only", str(caught.exception))

    def test_one_parent_alone_is_not_the_key(self):
        for extra in ({"Tq_serial": 1}, {"Tr_serial": 5}):
            with self.subTest(extra=extra):
                with self.assertRaises(RuntimeError):
                    source_lookup(self.s, **extra)

    def test_the_parents_and_the_mode_together_are_an_error(self):
        with self.assertRaises(ValueError):
            source_lookup(self.s, Tq_serial=1, Tr_serial=5, labels_only=True)
        with self.assertRaises(ValueError):
            source_lookup(self.s, Tq_serial=1, labels_only=True)

    def test_a_parent_is_a_serial(self):
        for bad in (True, 1.0, "1", _Ref(1)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    keyed(self.s, bad, 5)
                with self.assertRaises(TypeError):
                    keyed(self.s, 1, bad)


class TestQEqualsR(_WithAStore):
    """§3 test 5. ``itertools.combinations_with_replacement`` gives the diagonal, where ``Tq`` and
    ``Tr`` are the same row: the key names the one serial twice, and nothing in the schema or the
    lookup needs the two to differ."""

    def test_the_lookup_names_the_same_serial_twice(self):
        stored = store_source(self.s, Tq_serial=1, Tr_serial=1, r=Q_EXIT)
        found = keyed(self.s, 1, 1, r=Q_EXIT)
        self.assertEqual(found.store_id, stored)
        self.assertEqual((found.Tq_serial, found.Tr_serial), (1, 1))
        self.assertFalse(keyed(self.s, 1, 2, r=Q_EXIT).available)
        self.assertFalse(keyed(self.s, 2, 1, r=Q_EXIT).available)

    def test_the_diagonal_and_an_off_diagonal_source_are_different_rows(self):
        diagonal = store_source(self.s, Tq_serial=1, Tr_serial=1, r=Q_EXIT)
        off = store_source(self.s, Tq_serial=1, Tr_serial=2)
        self.assertNotEqual(diagonal, off)

    def test_the_inventory_resolves_both_parents_to_the_one_record(self):
        replicated, sharded, keys = full_rows()
        diagonal = dict(sharded[0]["QuadSource"][0])
        diagonal.update(
            serial=3,
            label="diagonal",
            r_wavenumber_exit_serial=diagonal["q_wavenumber_exit_serial"],
            Tq_serial=1,
            Tr_serial=1,
        )
        sharded[0] = with_rows(sharded[0], {"QuadSource": [diagonal]})
        sharded[0] = with_rows(
            sharded[0],
            {
                "QuadSource_tags": [
                    {"parent_serial": 3, "tag_serial": 1},
                    {"parent_serial": 3, "tag_serial": 2},
                ]
            },
        )
        inv = read_the_store(replicated, sharded, keys)
        self.assertEqual(inv["QuadSource"].problems, ())
        self.assertEqual(inv["QuadSource"].count, 3)
        (record,) = [r for r in inv["QuadSource"].records if r.key["Tq"] == r.key["Tr"]]
        self.assertEqual(record.key["q"], record.key["r"])


def read_the_store(replicated, sharded, keys):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp).resolve()
        return read_inventory(
            build_full_store(
                directory, replicated=replicated, sharded=sharded, shard_keys=keys
            ).primary,
            _factories,
        )


class TestTheInventoryResolvesAcrossShards(unittest.TestCase):
    """§3 test 6. In the full-store fixture QuadSource 1 is on shard 0 with Tq = Tk 1 (shard 0) and
    Tr = Tk 2 (shard 1): a cross-shard Tr."""

    def _rows(self):
        replicated, sharded, keys = full_rows()
        # another Tk row for the same k on shard 1, differing from Tk 2 only in its atol: a second
        # possible Tr for QuadSource 1, on the other shard
        other = _tk_numeric(serial=4, exit_serial=2)
        other["TkNumericIntegration"][0]["atol_serial"] = 3
        sharded[1] = with_rows(sharded[1], other)
        return replicated, sharded, keys

    @staticmethod
    def _copy_of_the_first_source(sharded, **changes):
        row = dict(sharded[0]["QuadSource"][0])
        row.update(serial=3, label="copy", **changes)
        sharded[0] = with_rows(sharded[0], {"QuadSource": [row]})
        sharded[0] = with_rows(
            sharded[0],
            {
                "QuadSource_tags": [
                    {"parent_serial": 3, "tag_serial": 1},
                    {"parent_serial": 3, "tag_serial": 2},
                ]
            },
        )

    def test_the_fixture_source_resolves_its_cross_shard_tr(self):
        replicated, sharded, keys = self._rows()
        inv = read_the_store(replicated, sharded, keys)
        self.assertEqual(inv["QuadSource"].problems, ())
        self.assertEqual(inv["QuadSource"].count, 2)
        self.assertEqual(inv.problems, ())

    def test_two_rows_differing_only_in_tr_are_two_records_with_different_identities(
        self,
    ):
        replicated, sharded, keys = self._rows()
        self._copy_of_the_first_source(sharded, Tr_serial=4)
        inv = read_the_store(replicated, sharded, keys)

        self.assertEqual(inv["QuadSource"].problems, ())
        # QuadSource 1 and its copy carry two tags; QuadSource 2 carries one
        records = [r for r in inv["QuadSource"].records if len(r.tags) == 2]
        self.assertEqual(len(records), 2)
        one, two = records
        differing = {f for f in one.key if one.key[f] != two.key[f]}
        self.assertEqual(differing, {"Tr"})
        self.assertNotEqual(one.identity(), two.identity())
        self.assertNotEqual(
            inv.digest_of("QuadSource", one), inv.digest_of("QuadSource", two)
        )

    def test_the_same_rows_with_the_same_tr_are_a_duplicate(self):
        # the control: what the two rows above would be if Tr were not in the key
        replicated, sharded, keys = self._rows()
        self._copy_of_the_first_source(sharded, Tr_serial=2)
        inv = read_the_store(replicated, sharded, keys)
        self.assertEqual(inv["QuadSource"].count, 3)
        self.assertEqual(
            [p.split(":")[0] for p in inv["QuadSource"].problems], ["duplicate"]
        )

    def test_a_tr_on_no_shard_is_a_named_unresolved_parent(self):
        replicated, sharded, keys = self._rows()
        self._copy_of_the_first_source(sharded, Tr_serial=99)
        inv = read_the_store(replicated, sharded, keys)
        self.assertEqual(inv["QuadSource"].count, 2, "the row is not a record")
        (problem,) = inv["QuadSource"].problems
        self.assertTrue(problem.startswith("unresolved-parent: QuadSource:"), problem)
        self.assertIn("field(s) Tr)", problem)
        self.assertIn("serials 3", problem)

    def test_tq_resolves_on_its_own_shard_only(self):
        # Tq is the same-shard form: Tk 2 exists, on shard 1, and is no parent for a source on
        # shard 0. (Tr = Tk 2 from shard 0 does resolve: it is the cross-shard form.)
        replicated, sharded, keys = self._rows()
        self._copy_of_the_first_source(sharded, Tq_serial=2)
        inv = read_the_store(replicated, sharded, keys)
        self.assertEqual(inv["QuadSource"].count, 2)
        (problem,) = inv["QuadSource"].problems
        self.assertIn("unresolved-parent", problem)
        self.assertIn("field(s) Tq)", problem)


class TestCrossShardParentUnit(unittest.TestCase):
    """``Parent(cross_shard=True)`` against ``read_records`` directly: it resolves against
    ``all_digests``, and never against this shard's ``digests`` alone."""

    def _read(self, parent, digests, all_digests):
        metadata = sqla.MetaData()
        table = sqla.Table(
            "Child",
            metadata,
            sqla.Column("serial", sqla.Integer, primary_key=True),
            sqla.Column("parent_serial", sqla.Integer),
        )
        engine = sqla.create_engine("sqlite://", future=True)
        metadata.create_all(engine)
        with engine.begin() as conn:
            conn.execute(sqla.insert(table), [{"serial": 1, "parent_serial": 7}])
            context = ShardContext(
                shard=0,
                keys={},
                digests=digests,
                all_digests=all_digests,
            )
            out = read_records(conn, table, {}, context, parents={"p": parent})
        engine.dispose()
        return out

    def test_it_resolves_against_another_shards_record(self):
        read = self._read(
            Parent("parent_serial", "Parent", cross_shard=True),
            digests={"Parent": {}},
            all_digests={"Parent": {7: "digest-of-7"}},
        )
        self.assertEqual(read.problems, [])
        self.assertEqual([r.key for _, r, _ in read.rows], [{"p": "digest-of-7"}])

    def test_the_identity_is_the_digest_and_not_the_serial(self):
        read = self._read(
            Parent("parent_serial", "Parent", cross_shard=True),
            digests={"Parent": {}},
            all_digests={"Parent": {7: "digest-of-7"}},
        )
        self.assertNotIn(7, [v for _, r, _ in read.rows for v in r.key.values()])

    def test_a_same_shard_parent_does_not(self):
        read = self._read(
            Parent("parent_serial", "Parent"),
            digests={"Parent": {}},
            all_digests={"Parent": {7: "digest-of-7"}},
        )
        self.assertEqual(read.rows, [])
        self.assertIn("unresolved-parent", read.problems[0])

    def test_a_serial_no_shard_holds_is_unresolved(self):
        read = self._read(
            Parent("parent_serial", "Parent", cross_shard=True),
            digests={"Parent": {}},
            all_digests={"Parent": {8: "digest-of-8"}},
        )
        self.assertEqual(read.rows, [])
        self.assertIn("unresolved-parent", read.problems[0])

    def test_a_class_not_yet_built_is_an_error_not_a_silent_miss(self):
        with self.assertRaises(RuntimeError):
            self._read(
                Parent("parent_serial", "Parent", cross_shard=True),
                digests={"Parent": {7: "digest-of-7"}},
                all_digests={},
            )


class TestMergeDigests(unittest.TestCase):
    def test_the_union_across_shards(self):
        self.assertEqual(
            _merge_digests([{1: "a", 2: "b"}, {3: "c"}]), {1: "a", 2: "b", 3: "c"}
        )

    def test_a_replicated_serial_with_one_digest_on_every_shard_is_kept(self):
        self.assertEqual(_merge_digests([{1: "a"}, {1: "a"}]), {1: "a"})

    def test_a_serial_two_shards_hold_differently_resolves_to_neither(self):
        self.assertEqual(_merge_digests([{1: "a", 2: "b"}, {1: "x"}]), {2: "b"})


def tearDownModule():
    assert not ray.is_initialized()


if __name__ == "__main__":
    unittest.main()
