"""
``tools/store_value_diff.py`` on two stores whose value tables differ by a column
(``prompts/levin-phase-floor`` prompt 06 §1.5).

Prompt 06 adds ``theta_residual`` to the value tables of ``GkWKBIntegration``,
``TkWKBIntegration`` and ``GkSource``. Its rehearsal pair, R13 at the code commit's parent and R14
at the code commit, is compared by this tool at the code commit, whose declared tables hold the
column; R13's files do not, and ``open_read_only`` refuses a store whose files lack a declared
column. The tool reads such a store without the columns it lacks, reports each as present in one
store only, and compares every other column as before.

The stores are ``StoreFixtures.real_store_fixtures.build_full_store`` stores in a temporary
directory; the "earlier" store is made by dropping the column from every shard file with SQLite's
``ALTER TABLE ... DROP COLUMN``. No Ray; nothing under ``var/``.

- **(t6a)** a store lacking ``GkWKBValue.theta_residual`` against a full one: exit 1, the column is
  reported as present in the full store only, every value row is still matched and compared on
  its other columns (all identical), and every other class is value-identical. Both stores' files
  are unchanged by the comparison. The same with the stores the other way round.
- **(t6b)** the same, with one ``theta_mod_2pi`` changed by one ulp in the store lacking the column: that
  difference is still reported, column and delta, beside the absent column.
- **(t6c)** a schema difference of any other kind -- here a column the code does not declare --
  is still a refusal, exit 2.
"""

import contextlib
import io
import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

from StoreFixtures.real_store_fixtures import (
    build_full_store,
    file_state,
    find_row,
    full_rows,
    vary_row,
)
from config.datastore import factories as _factories
from tools.store_value_diff import compare_stores, main

# a GkWKBValue row of the full store: GkWKBIntegration serial 1, at redshift serial 2
GK_WKB_VALUE_SERIAL = 301
VALUE_TABLE = "GkWKBValue"
OWNER = "GkWKBIntegration"
COLUMN = "theta_residual"


def _run(argv):
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main([str(a) for a in argv], factories=_factories)
    return code, out.getvalue(), err.getvalue()


def _alter_every_shard(store, statement: str):
    """Run ``statement`` on every shard file of ``store`` (the primary holds no value table)."""
    for path in store.shard_files.values():
        conn = sqlite3.connect(str(path))
        try:
            conn.execute(statement)
            conn.commit()
        finally:
            conn.close()


class _StoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._n = 0

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, rows=None):
        self._n += 1
        directory = self.root / f"store{self._n}"
        directory.mkdir()
        if rows is None:
            return build_full_store(directory)
        replicated, sharded, keys = rows
        return build_full_store(
            directory, replicated=replicated, sharded=sharded, shard_keys=keys
        )

    def without_the_column(self, rows=None):
        store = self.build(rows)
        _alter_every_shard(store, f'ALTER TABLE "{VALUE_TABLE}" DROP COLUMN {COLUMN}')
        return store

    def assert_only_the_owner_differs(self, report):
        for name, c in report["classes"].items():
            if name != OWNER:
                self.assertTrue(c["identical"], name)
        self.assertFalse(report["classes"][OWNER]["identical"])
        self.assertFalse(report["identical"])


class TestAbsentColumn(_StoreTest):
    def test_t6a_a_column_in_one_store_only_is_reported_and_the_rest_compared(self):
        full = self.build()
        earlier = self.without_the_column()
        before = (file_state(full.directory), file_state(earlier.directory))

        for a, b, side in ((full, earlier, "a"), (earlier, full, "b")):
            with self.subTest(full_store=side):
                code, text, err = _run([a.primary, b.primary, "--json"])
                self.assertEqual(code, 1, err)
                report = json.loads(text)
                self.assert_only_the_owner_differs(report)

                owner = report["classes"][OWNER]
                self.assertEqual(owner["rows"]["differ"], 0)
                self.assertEqual(owner["rows"]["columns_only_in_a"], [])
                self.assertEqual(owner["rows"]["columns_only_in_b"], [])

                values = owner["values"][VALUE_TABLE]
                other = "b" if side == "a" else "a"
                self.assertEqual(values[f"columns_only_in_{side}"], [COLUMN])
                self.assertEqual(values[f"columns_only_in_{other}"], [])
                # every value row is matched and compared on its other columns
                self.assertEqual(values["differ"], 0)
                self.assertEqual(values["unmatched"], [])
                self.assertEqual(values["only_in_a"], [])
                self.assertEqual(values["only_in_b"], [])
                self.assertGreater(values["identical"], 0)
                self.assertEqual(values["columns"], {})

                code, text, err = _run([a.primary, b.primary])
                self.assertEqual(code, 1, err)
                self.assertIn(
                    f"column {COLUMN}: present in {side.upper()} only, not compared",
                    text,
                )
                self.assertIn(f"{OWNER}: NOT value-identical", text)

        # the stores are read, never written
        self.assertEqual(
            (file_state(full.directory), file_state(earlier.directory)), before
        )

    def test_t6b_other_columns_of_that_table_are_still_compared(self):
        full = self.build()
        replicated, sharded, keys = full_rows()
        row = find_row(replicated, sharded, VALUE_TABLE, GK_WKB_VALUE_SERIAL)
        self.assertNotIn("theta_mod_2pi", row)
        # the fixture fills the non-nullable column with its placeholder
        old = 0.5
        new = math.nextafter(old, math.inf)
        vary_row(
            replicated, sharded, VALUE_TABLE, GK_WKB_VALUE_SERIAL, "theta_mod_2pi", new
        )
        earlier = self.without_the_column((replicated, sharded, keys))

        report = compare_stores(full.primary, earlier.primary, _factories)
        self.assert_only_the_owner_differs(report)
        values = report["classes"][OWNER]["values"][VALUE_TABLE]
        self.assertEqual(values["columns_only_in_a"], [COLUMN])
        self.assertEqual(values["differ"], 1)
        self.assertEqual(list(values["columns"]), ["theta_mod_2pi"])
        (entry,) = values["rows"]
        diff = entry["columns"]["theta_mod_2pi"]
        self.assertEqual(diff["a"], float.hex(old))
        self.assertEqual(diff["b"], float.hex(new))
        self.assertEqual(diff["delta"], new - old)

    def test_t6c_any_other_schema_difference_is_still_refused(self):
        full = self.build()
        other = self.build()
        _alter_every_shard(
            other, f'ALTER TABLE "{VALUE_TABLE}" ADD COLUMN not_declared FLOAT'
        )
        code, text, err = _run([full.primary, other.primary])
        self.assertEqual(code, 2)
        self.assertIn("not_declared", err)


if __name__ == "__main__":
    unittest.main()
