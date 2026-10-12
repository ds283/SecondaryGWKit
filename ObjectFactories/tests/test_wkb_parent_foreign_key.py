"""
The foreign keys prompt 07 of ``prompts/datastore-integrity`` declares -- ``TkWKBIntegration.
numeric_serial`` to ``TkNumericIntegration.serial`` and ``GkWKBIntegration.numeric_serial`` to
``GkNumericIntegration.serial`` -- checked with SQLite's own ``PRAGMA foreign_key_check`` over
every shard of a store ``real_store_fixtures.build_full_store`` writes, as prompt 06's
``test_foreign_key_check.py`` checks the keys it corrected. That module is 06's and is not
modified; this one follows its method.

**The rows are the test's own, because the fixture's would pass a wrong key by coincidence.** On
the fixture's shard 0, ``numeric_serial`` 1 (Tk WKB rows 1 and 2) is both a Tk numeric serial and a
Tk WKB serial, so a key naming ``TkWKBIntegration.serial`` would pass as ``QuadSource.Tq_serial``'s
wrong key did before 06. And the fixture's two Gk WKB rows are pure WKB (NULL), which no foreign key
check reads at all. So shard 0 gains, in the test's own ``sharded`` rows:

  * a ``TkWKBIntegration`` (serial 10) whose ``numeric_serial`` is 3: a ``TkNumericIntegration``
    serial on shard 0 (Tk numeric 1 and 3) and **no** ``TkWKBIntegration`` serial there (1, 2, 10);
  * a **seeded** ``GkWKBIntegration`` (serial 10) with a ``GkNumericIntegration`` (serial 5) at the
    WKB row's own ``z_source`` (2), which the fixture has none of; 5 is no ``GkWKBIntegration``
    serial on shard 0 (1, 10).

``test_the_added_rows_are_not_satisfied_by_coincidence`` asserts that property of the data. The
fixture module's default rows are not changed by this module.

No Ray, nothing under ``var/``: every store is in a temporary directory, opened ``mode=ro`` for the
check.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import ray

from ObjectFactories.tests import real_store_fixtures as rsf

ADDED_TK_WKB = 10
ADDED_TK_PARENT = 3
ADDED_GK_WKB = 10
ADDED_GK_PARENT = 5
ADDED_GK_Z_SOURCE = 2


def violations(path: Path) -> List[tuple]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        conn.close()


def added_rows(tk_parent=ADDED_TK_PARENT, gk_parent=ADDED_GK_PARENT) -> rsf.RowSet:
    return {
        "TkWKBIntegration": [
            {
                "serial": ADDED_TK_WKB,
                "version": 1,
                "label": "fixture-tkwkb-added",
                "wavenumber_exit_serial": 4,
                "model_serial": 2,
                "numeric_serial": tk_parent,
                "rho_gauss_order": 10,
                "phase_solver_serial": 1,
                "friction_solver_serial": 1,
                "z_min_serial": 3,
                "z_samples": 0,
                "z_init": 100.0,
                "T_init": 1.0,
                "Tprime_init": 0.0,
                "validated": True,
            }
        ],
        "GkNumericIntegration": [
            {
                "serial": ADDED_GK_PARENT,
                "version": 1,
                "label": "fixture-gknum-added",
                "wavenumber_exit_serial": 1,
                "model_serial": 1,
                "atol_serial": 1,
                "rtol_serial": 2,
                "break_point_kind": "discontinuity",
                "solver_serial": 1,
                "z_source_serial": ADDED_GK_Z_SOURCE,
                "z_min_serial": 3,
                "z_samples": 0,
                "validated": True,
            }
        ],
        "GkWKBIntegration": [
            {
                "serial": ADDED_GK_WKB,
                "version": 1,
                "label": "fixture-gkwkb-added",
                "wavenumber_exit_serial": 1,
                "model_serial": 1,
                "numeric_serial": gk_parent,
                "rho_gauss_order": 10,
                "solver_serial": 1,
                "z_source_serial": ADDED_GK_Z_SOURCE,
                "z_min_serial": 3,
                "z_samples": 0,
                "z_init": 50.0,
                "G_init": 0.0,
                "Gprime_init": 1.0,
                "validated": True,
            }
        ],
    }


def own_rows(**kwargs):
    replicated, sharded, keys = rsf.full_rows()
    sharded[0] = rsf.with_rows(sharded[0], added_rows(**kwargs))
    return replicated, sharded, keys


def serials(rows: rsf.RowSet, table: str) -> List[int]:
    return [row["serial"] for row in rows.get(table, [])]


class TestTheNewKeys(unittest.TestCase):
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

    def test_the_added_rows_are_not_satisfied_by_coincidence(self):
        rows = own_rows()[1][0]
        self.assertIn(ADDED_TK_PARENT, serials(rows, "TkNumericIntegration"))
        self.assertNotIn(ADDED_TK_PARENT, serials(rows, "TkWKBIntegration"))
        self.assertIn(ADDED_GK_PARENT, serials(rows, "GkNumericIntegration"))
        self.assertNotIn(ADDED_GK_PARENT, serials(rows, "GkWKBIntegration"))
        # the Gk parent is at the seeded row's own z_source
        parent = [
            r for r in rows["GkNumericIntegration"] if r["serial"] == ADDED_GK_PARENT
        ]
        self.assertEqual(parent[0]["z_source_serial"], ADDED_GK_Z_SOURCE)
        # ... whereas the fixture's own Tk rows would satisfy a key to the WKB table itself
        default = rsf.full_rows()[1][0]
        for row in default["TkWKBIntegration"]:
            self.assertIn(row["numeric_serial"], serials(default, "TkWKBIntegration"))

    def test_every_shard_passes_foreign_key_check(self):
        self.assertEqual(self.build(), {0: [], 1: []})

    def test_the_default_stores_pass_too(self):
        store = rsf.build_full_store(self.root)
        for serial, path in store.shard_files.items():
            with self.subTest(shard=serial):
                self.assertEqual(violations(path), [])

    def test_a_dangling_tk_parent_is_reported(self):
        """Tk numeric 2 is on shard 1, not shard 0: the check reads the declared key."""
        found = self.build(tk_parent=2)
        self.assertEqual(
            [v[:3] for v in found[0]],
            [("TkWKBIntegration", ADDED_TK_WKB, "TkNumericIntegration")],
        )
        self.assertEqual(found[1], [])

    def test_a_dangling_gk_parent_is_reported(self):
        found = self.build(gk_parent=2)
        self.assertEqual(
            [v[:3] for v in found[0]],
            [("GkWKBIntegration", ADDED_GK_WKB, "GkNumericIntegration")],
        )
        self.assertEqual(found[1], [])


def tearDownModule():
    assert not ray.is_initialized()


if __name__ == "__main__":
    unittest.main()
