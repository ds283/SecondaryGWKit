"""
Tests for prompt 06 of ``prompts/datastore-integrity`` (board item S4): every shard of a store
written the way production writes it passes SQLite's own ``PRAGMA foreign_key_check`` with no row
(docs/datastore-integrity-audit.md §1, §2.1, §2.2).

Nothing in production turns foreign-key enforcement on (``PRAGMA foreign_keys`` reads 0), so a
declared foreign key is documentation, and a wrong one is a false statement nothing checks.
``foreign_key_check`` works whether or not enforcement is on, so this test changes nothing about
production's connections. It would have caught ``[00-quadsource-tq-serial-has-the-wrong-foreign-key]``
(``QuadSource.Tq_serial`` declared ``QuadSource.serial`` but holds a ``TkNumericIntegration``
serial), given rows are ones the wrong key does not satisfy by coincidence (below).

What is checked, and on which stores:

  * ``TestFullStore`` -- ``real_store_fixtures.build_full_store`` on the test's own ``sharded`` rows,
    which are the fixture's rows plus one ``QuadSource`` whose ``Tq_serial`` is a
    ``TkNumericIntegration`` serial on its shard and **not** a ``QuadSource`` serial there. The
    fixture's own ``QuadSource`` rows have ``Tq_serial`` equal to a ``QuadSource`` serial on the
    same shard (shard 0: 1, shard 1: 2), so the wrong key passes on them by coincidence; the added
    row is what makes this test fail on the tree before S1. The fixture module's default rows are
    unchanged. The test asserts that property of the row rather than assuming it.
  * ``TestStandinStore`` -- a store written through ``ShardedPool`` on the stand-in shards of
    ``standin_pool``: a replicated exit time and a ``GkSourcePolicyData`` (a sharded class) that
    names it, so that the row written by the real ``store()`` resolves its exit time on the shard
    it is on, which is what S2's new key asserts. The stand-in policy data's ``source_serial`` is a
    placeholder that no ``GkSource`` row backs (writing a real ``GkSource`` needs a whole Gk
    pipeline), so that one reference is declared and expected to be reported; the test asserts
    that it is the *only* violation, and that none names ``wavenumber_exit_time``.
  * ``TestTheCheckSeesAViolation`` -- two dangling references, to show the check reports a row
    when there is one (a check that cannot fail proves nothing).

**What this cannot see.** ``foreign_key_check`` sees only a declared foreign key, and a key can be
declared only to a table in the same file. So three references are not checked, because each may
name a row on another shard:

  * ``QuadSource.Tr_serial`` -- the ``TkNumericIntegration`` at ``k = r``, on ``r``'s shard, while
    ``QuadSource`` shards on ``q``;
  * ``QuadSourceIntegral.source_serial`` -- a ``QuadSource``, sharded on ``q``, while
    ``QuadSourceIntegral`` shards on ``k``;
  * ``shard_keys`` -- the primary's table naming a shard for each wavenumber serial, which are
    replicated rows of the shards, in another file.

It also cannot see a *right-looking wrong* reference: a key that names a table that does hold the
serial on that shard by coincidence passes (the reason for the added row above).

No Ray, nothing under ``var/``: every store is in a temporary directory.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import ray

import Datastore.tests.standin_pool as sp
from Datastore.tests import real_store_fixtures as rsf

# a QuadSource on shard 0 (q = wavenumber 1, exits 1 and 4), whose Tq is TkNumericIntegration 3
# (shard 0: Tk 1 and 3; QuadSource 1). Its Tr is TkNumericIntegration 2, on shard 1
ADDED_QUADSOURCE_SERIAL = 10
ADDED_TQ_SERIAL = 3
ADDED_TR_SERIAL = 2


def violations(path: Path) -> List[tuple]:
    """``PRAGMA foreign_key_check`` on ``path``, opened ``mode=ro``: (table, rowid, parent, fk id)
    for each row that violates a declared foreign key."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        conn.close()


def all_violations(store: rsf.RealStore) -> Dict[int, List[tuple]]:
    return {serial: violations(path) for serial, path in store.shard_files.items()}


def own_rows(extra_shard0=None):
    """The full store's rows, plus the added ``QuadSource`` (and ``extra_shard0``'s rows)."""
    replicated, sharded, keys = rsf.full_rows()
    added = {
        "QuadSource": [
            {
                "serial": ADDED_QUADSOURCE_SERIAL,
                "version": 1,
                "label": "fixture-quadsource-added",
                "model_serial": 1,
                "q_wavenumber_exit_serial": 1,
                "r_wavenumber_exit_serial": 2,
                "Tq_serial": ADDED_TQ_SERIAL,
                "Tr_serial": ADDED_TR_SERIAL,
                "z_samples": 1,
                "validated": True,
            }
        ]
    }
    sharded[0] = rsf.with_rows(sharded[0], added)
    if extra_shard0 is not None:
        sharded[0] = rsf.with_rows(sharded[0], extra_shard0)
    return replicated, sharded, keys


def serials(rows: rsf.RowSet, table: str) -> List[int]:
    return [row["serial"] for row in rows.get(table, [])]


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)


class TestFullStore(_TempDirCase):
    def build(self, extra_shard0=None) -> rsf.RealStore:
        replicated, sharded, keys = own_rows(extra_shard0)
        return rsf.build_full_store(
            self.root, replicated=replicated, sharded=sharded, shard_keys=keys
        )

    def test_the_added_row_is_not_satisfied_by_coincidence(self):
        """The row's ``Tq_serial`` is a Tk serial on its shard and no ``QuadSource`` serial there:
        it is what lets this module fail on the wrong key."""
        _, sharded, _ = own_rows()
        rows = sharded[0]
        self.assertIn(ADDED_TQ_SERIAL, serials(rows, "TkNumericIntegration"))
        self.assertNotIn(ADDED_TQ_SERIAL, serials(rows, "QuadSource"))
        # ... whereas the fixture's own rows would satisfy the wrong key by coincidence
        for shard, default in rsf.full_rows()[1].items():
            for row in default["QuadSource"]:
                self.assertIn(row["Tq_serial"], serials(default, "QuadSource"))
                self.assertIn(
                    row["Tq_serial"], serials(default, "TkNumericIntegration")
                )

    def test_the_added_rows_are_in_the_written_shard(self):
        store = self.build()
        conn = sqlite3.connect(f"file:{store.shard_files[0]}?mode=ro", uri=True)
        try:
            self.assertEqual(
                [(ADDED_QUADSOURCE_SERIAL, ADDED_TQ_SERIAL, ADDED_TR_SERIAL)],
                conn.execute(
                    'SELECT serial, Tq_serial, Tr_serial FROM "QuadSource" WHERE serial = ?',
                    (ADDED_QUADSOURCE_SERIAL,),
                ).fetchall(),
            )
        finally:
            conn.close()

    def test_every_shard_passes_foreign_key_check(self):
        store = self.build()
        self.assertEqual(2, len(store.shard_files))
        self.assertEqual({0: [], 1: []}, all_violations(store))

    def test_the_default_stores_pass_too(self):
        replicated, sharded, keys = rsf.full_rows()
        full = rsf.build_full_store(self.root, stem="default-full")
        self.assertEqual({0: [], 1: []}, all_violations(full))
        real = rsf.build_real_store(self.root, stem="default-real")
        self.assertEqual({0: [], 1: []}, all_violations(real))

    def test_a_relabelled_store_passes(self):
        """``relabel_serials`` follows the schema's declared keys, so the relabelled data is
        consistent with them. (It cannot, by itself, tell a right key from a wrong one.)
        """
        replicated, sharded, keys = own_rows()
        replicated, sharded, keys = rsf.relabel_serials(replicated, sharded, keys)
        store = rsf.build_full_store(
            self.root, replicated=replicated, sharded=sharded, shard_keys=keys
        )
        self.assertEqual({0: [], 1: []}, all_violations(store))


class TestTheCheckSeesAViolation(_TempDirCase):
    def test_a_dangling_tq_is_reported(self):
        """``Tq_serial`` 2 is a Tk row, but on shard 1: not in this shard's ``TkNumericIntegration``."""
        replicated, sharded, keys = own_rows()
        for row in sharded[0]["QuadSource"]:
            if row["serial"] == ADDED_QUADSOURCE_SERIAL:
                row["Tq_serial"] = 2
        store = rsf.build_full_store(
            self.root, replicated=replicated, sharded=sharded, shard_keys=keys
        )
        result = all_violations(store)
        self.assertEqual([], result[1])
        self.assertEqual(1, len(result[0]))
        table, _rowid, parent, _fk = result[0][0]
        self.assertEqual(("QuadSource", "TkNumericIntegration"), (table, parent))

    def test_a_dangling_exit_time_in_the_policy_data_is_reported(self):
        replicated, sharded, keys = own_rows()
        self.assertGreater(len(sharded[0]["GkSourcePolicyData"]), 0)
        sharded[0]["GkSourcePolicyData"][0]["wavenumber_exit_serial"] = 999
        store = rsf.build_full_store(
            self.root, replicated=replicated, sharded=sharded, shard_keys=keys
        )
        result = all_violations(store)
        self.assertEqual([], result[1])
        self.assertEqual(1, len(result[0]))
        table, _rowid, parent, _fk = result[0][0]
        self.assertEqual(
            ("GkSourcePolicyData", "wavenumber_exit_time"), (table, parent)
        )


class TestStandinStore(_TempDirCase):
    def test_a_store_written_by_the_real_write_path_passes(self):
        cluster = sp.StandinCluster()
        with cluster.active():
            primary = self.root / "standin" / "store.sqlite"
            pool = cluster.open_pool(primary)
            try:
                cluster.controller = list(pool._shards.keys())[1]
                units = sp.make_units()
                cosmology = sp.StandinCosmology(units)
                atol = sp.get_tolerance(pool, 1.0e-10)
                rtol = sp.get_tolerance(pool, 1.0e-9)
                k = sp.get_wavenumber(pool, 0.5, units)
                exit_time = ray.get(
                    pool.object_store(sp.make_exit_time(k, cosmology, atol, rtol))
                )
                policy = ray.get(
                    pool.object_get(
                        "GkSourcePolicy",
                        label="standin-policy",
                        Levin_threshold=1.5,
                        numeric_policy="maximize-numeric",
                    )
                )
                stored = ray.get(
                    pool.object_store(
                        sp.make_policy_data(
                            exit_time, source_serial=1, policy_serial=policy.store_id
                        )
                    )
                )
                shard_files = {
                    sid: Path(pool._shard_db_files[sid]) for sid in pool._shards.keys()
                }
                holder_shard = pool._shard_keys[exit_time.k.store_id]
            finally:
                cluster.close_pool()

        self.assertGreaterEqual(len(shard_files), 2)
        for sid, path in shard_files.items():
            with self.subTest(shard=sid):
                found = violations(path)
                # the stand-in source is the one reference no row backs (module docstring)
                self.assertEqual(
                    [], [row for row in found if row[2] != "GkSource"], found
                )
                # (the last field of a row is the foreign key's index in the table, which moves
                # when a key is added, so it is not compared)
                self.assertEqual(
                    (
                        []
                        if sid != holder_shard
                        else [("GkSourcePolicyData", 1, "GkSource")]
                    ),
                    [row[:3] for row in found],
                )

        # the policy data is on one shard only, and the exit time it names is on every shard
        holders = []
        for sid, path in shard_files.items():
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                policy_rows = conn.execute(
                    'SELECT wavenumber_exit_serial FROM "GkSourcePolicyData"'
                ).fetchall()
                exit_rows = conn.execute(
                    'SELECT serial FROM "wavenumber_exit_time"'
                ).fetchall()
            finally:
                conn.close()
            self.assertEqual([(exit_time.store_id,)], exit_rows)
            if policy_rows:
                holders.append(sid)
                self.assertEqual([(exit_time.store_id,)], policy_rows)
        self.assertEqual(1, len(holders))
        self.assertIsNotNone(stored.store_id)


def tearDownModule():
    if ray.is_initialized():
        raise RuntimeError(
            "test_foreign_key_check: ray was initialised by this module's tests, which must not "
            "happen (README §5 rule 9)"
        )
