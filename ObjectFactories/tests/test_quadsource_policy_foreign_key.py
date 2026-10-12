"""
The foreign key prompt 08 of ``prompts/datastore-integrity`` declares --
``QuadSourceIntegral.quad_policy_serial`` to ``QuadSourcePolicy.serial`` -- checked with SQLite's own
``PRAGMA foreign_key_check`` over every shard of a store ``real_store_fixtures.build_full_store``
writes, as prompt 06's ``test_foreign_key_check.py`` and prompt 07's
``test_wkb_parent_foreign_key.py`` check the keys they declared. Those modules are not modified;
this one follows their method.

**The rows are the test's own, because the fixture's would pass a wrong key by coincidence.** The
fixture holds two ``QuadSourcePolicy`` rows (serials 1 and 2) and two ``GkSourcePolicy`` rows
(serials 1 and 2), so ``quad_policy_serial`` 1 or 2 is also a ``GkSourcePolicy`` serial. A key
that named ``GkSourcePolicy.serial`` -- the neighbouring ``policy_serial`` column's table, the
mistake most easily made here -- would pass on the fixture's rows, as ``QuadSource.Tq_serial``'s
wrong key did before prompt 06. So the store gains, in the test's own rows:

  * a ``QuadSourcePolicy`` (serial 3), a replicated row and so on every shard;
  * on shard 0 a ``QuadSourceIntegral`` (serial 10) whose ``quad_policy_serial`` is 3: a
    ``QuadSourcePolicy`` serial and **no** ``GkSourcePolicy`` serial (1 and 2).

``test_the_added_row_is_not_satisfied_by_coincidence`` asserts that property of the data. The
fixture module's default rows are not changed by this module (apart from the second policy and the
key the fixture's ``QuadSourceIntegral`` rows carry, which prompt 08 added to it).

No Ray, nothing under ``var/``: every store is in a temporary directory, opened ``mode=ro`` for the
check.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import ray

from StoreFixtures import real_store_fixtures as rsf

ADDED_POLICY = 3
ADDED_QSI = 10


def violations(path: Path) -> List[tuple]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        conn.close()


def own_rows(quad_policy_serial=ADDED_POLICY):
    """The full store's rows, plus a third ``QuadSourcePolicy`` and, on shard 0, a
    ``QuadSourceIntegral`` keyed on ``quad_policy_serial``."""
    replicated, sharded, keys = rsf.full_rows()
    replicated = rsf.with_rows(
        replicated,
        {
            "QuadSourcePolicy": [
                {
                    "serial": ADDED_POLICY,
                    "version": 1,
                    "label": "qpolicy-added",
                    "Levin_threshold": 2.5,
                    "numeric_policy": "maximize-WKB",
                }
            ]
        },
    )
    sharded[0] = rsf.with_rows(
        sharded[0],
        {
            "QuadSourceIntegral": [
                {
                    "serial": ADDED_QSI,
                    "version": 1,
                    "label": "fixture-qsi-added",
                    "model_serial": 1,
                    "k_wavenumber_exit_serial": 1,
                    "q_wavenumber_exit_serial": 1,
                    "r_wavenumber_exit_serial": 2,
                    "atol_serial": 1,
                    "rtol_serial": 2,
                    "policy_serial": 1,
                    "quad_policy_serial": quad_policy_serial,
                    "source_serial": 1,
                    "data_serial": 1,
                    "z_response_serial": 3,
                    "z_source_max_serial": 1,
                    "b": 0.0,
                    "total": 4.0,
                    "numeric_quad": 1.0,
                    "WKB_Levin": 3.0,
                }
            ]
        },
    )
    return replicated, sharded, keys


def serials(rows: rsf.RowSet, table: str) -> List[int]:
    return [row["serial"] for row in rows.get(table, [])]


class TestTheNewKey(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def build(self, **kwargs) -> Dict[int, List[tuple]]:
        replicated, sharded, keys = own_rows(**kwargs)
        store = rsf.build_full_store(
            self.root, replicated=replicated, sharded=sharded, shard_keys=keys
        )
        return {serial: violations(path) for serial, path in store.shard_files.items()}

    def test_the_added_row_is_not_satisfied_by_coincidence(self):
        replicated, sharded, _ = own_rows()
        self.assertIn(ADDED_POLICY, serials(replicated, "QuadSourcePolicy"))
        self.assertNotIn(ADDED_POLICY, serials(replicated, "GkSourcePolicy"))
        # ... whereas every quad_policy_serial the fixture's own rows carry is also a
        # GkSourcePolicy serial, so a key to the wrong policy table would pass on them
        default_replicated, default_sharded, _ = rsf.full_rows()
        for rows in default_sharded.values():
            for row in rows.get("QuadSourceIntegral", []):
                self.assertIn(
                    row["quad_policy_serial"],
                    serials(default_replicated, "GkSourcePolicy"),
                )
                self.assertIn(
                    row["quad_policy_serial"],
                    serials(default_replicated, "QuadSourcePolicy"),
                )

    def test_every_shard_passes_foreign_key_check(self):
        self.assertEqual(self.build(), {0: [], 1: []})

    def test_the_default_stores_pass_too(self):
        store = rsf.build_full_store(self.root)
        for serial, path in store.shard_files.items():
            with self.subTest(shard=serial):
                self.assertEqual(violations(path), [])

    def test_a_dangling_policy_is_reported(self):
        """Policy 4 exists on no shard: the check reads the declared key."""
        found = self.build(quad_policy_serial=4)
        self.assertEqual(
            [v[:3] for v in found[0]],
            [("QuadSourceIntegral", ADDED_QSI, "QuadSourcePolicy")],
        )
        self.assertEqual(found[1], [])


def tearDownModule():
    assert not ray.is_initialized()


if __name__ == "__main__":
    unittest.main()
