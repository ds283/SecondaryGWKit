"""
Tests for ``tools/store_value_diff.py`` (``prompts/levin-phase-floor`` prompt 05 §3A, tests
(t1)-(t5)): the comparison of two closed stores' values, matched by physical identity.

Every store is built by ``StoreFixtures.real_store_fixtures.build_full_store``, which holds at least
one row of every class of the inventory, with tags and value rows, in a temporary directory. The
expected identity of a row is found independently of the tool, from ``read_inventory``: the record
that disappears when the row is deleted, or the record ``StoreInventory.find`` gives for a digest,
resolved by ``StoreInventory.resolve``. No Ray; nothing under ``var/``.

- **(t1)** two builds of the same fixture are value-identical, exit 0, in every class, with no
  table left uncompared and no column that cannot be compared by identity; both stores' files are
  unchanged by the comparison;
- **(t2)** one float changed by one ulp in a copy -- in a row's own column, then in a value-table
  row -- is reported precisely: class, identity, column and delta;
- **(t3)** a row deleted from a copy is reported as present in one store only;
- **(t4)** a copy whose serials are all renumbered, with the same identities, is value-identical:
  the tool never matches by serial;
- **(t5)** a JSON column with the same content in another key order is identical, and a changed
  value in it is reported by key.
"""

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from datastorekit.store_inventory import read_inventory
from StoreFixtures.real_store_fixtures import (
    build_full_store,
    file_state,
    find_row,
    full_rows,
    relabel_serials,
    vary_row,
)
from config.datastore import factories as _factories
from tools.store_value_diff import compare_stores, main

# a QuadSourceIntegral row on shard 0 of the full store (total = 1.0), and a TkNumericValue row of
# TkNumericIntegration serial 1 (k = 0.1, LambdaCDM) at redshift serial 2 (z = 10), T = 0.5
QSI_SERIAL = 1
TK_VALUE_SERIAL = 12
TK_VALUE_PARENT_K = 0.1
TK_VALUE_Z = 10.0


def _run(argv):
    """``main(argv)``'s exit code and what it printed."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main([str(a) for a in argv], factories=_factories)
    return code, out.getvalue()


def _without_row(replicated, sharded, table, serial, tag_table=None):
    """The row sets with one row (and its tag association rows) removed, in place."""
    for rows in [replicated] + list(sharded.values()):
        if table in rows:
            rows[table] = [r for r in rows[table] if r.get("serial") != serial]
        if tag_table is not None and tag_table in rows:
            rows[tag_table] = [
                r for r in rows[tag_table] if r.get("parent_serial") != serial
            ]


def _digests(primary, name):
    inventory = read_inventory(primary, _factories)
    return {inventory.digest_of(name, r) for r in inventory[name].records}


class _StoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._n = 0

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, rows=None):
        """A full store in a directory of its own, from ``rows`` (or the defaults)."""
        self._n += 1
        directory = self.root / f"store{self._n}"
        directory.mkdir()
        if rows is None:
            return build_full_store(directory)
        replicated, sharded, keys = rows
        return build_full_store(
            directory, replicated=replicated, sharded=sharded, shard_keys=keys
        )

    def assert_only_class_differs(self, report, name):
        for other, c in report["classes"].items():
            if other != name:
                self.assertTrue(c["identical"], other)
        self.assertFalse(report["classes"][name]["identical"])
        self.assertFalse(report["identical"])


class TestIdentical(_StoreTest):
    def test_t1_two_builds_of_one_fixture_are_value_identical(self):
        a = self.build()
        b = self.build()
        before = (file_state(a.directory), file_state(b.directory))
        code, text = _run([a.primary, b.primary])
        self.assertEqual(code, 0, text)
        self.assertIn("RESULT: the stores are value-identical", text)
        # read-only: both stores' directories are unchanged, file by file (listing, SHA-256,
        # size and mtime), and nothing is left beside them
        self.assertEqual((file_state(a.directory), file_state(b.directory)), before)

        report = compare_stores(a.primary, b.primary, _factories)
        self.assertTrue(report["identical"])
        # every class of the inventory, every table accounted for, every reference resolvable:
        # the production schema's classes are all matchable by identity
        self.assertEqual(report["not_compared"], [])
        self.assertEqual(len(report["classes"]), 21)
        compared_values = 0
        for name, c in report["classes"].items():
            self.assertTrue(c["identical"], name)
            self.assertEqual(c["unresolvable_columns"], [], name)
            for table in [c["rows"], *c["values"].values()]:
                self.assertEqual(table["unmatched"], [], name)
                self.assertEqual(table["differ"], 0, name)
            self.assertGreater(c["rows"]["identical"], 0, name)
            compared_values += sum(t["identical"] for t in c["values"].values())
        self.assertGreater(compared_values, 0)

        # --json is the same report, and machine-readable
        code, text = _run([a.primary, b.primary, "--json"])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(text)["identical"])


class TestOneUlp(_StoreTest):
    def test_t2_one_ulp_in_a_row_s_own_column(self):
        a = self.build()
        replicated, sharded, keys = full_rows()
        old = find_row(replicated, sharded, "QuadSourceIntegral", QSI_SERIAL)["total"]
        new = math.nextafter(old, math.inf)
        vary_row(replicated, sharded, "QuadSourceIntegral", QSI_SERIAL, "total", new)
        b = self.build((replicated, sharded, keys))

        # the row's identity, independently: the record that goes when the row is deleted
        replicated, sharded, keys = full_rows()
        _without_row(
            replicated,
            sharded,
            "QuadSourceIntegral",
            QSI_SERIAL,
            "QuadSourceIntegral_tags",
        )
        c = self.build((replicated, sharded, keys))
        (expected,) = _digests(a.primary, "QuadSourceIntegral") - _digests(
            c.primary, "QuadSourceIntegral"
        )

        code, text = _run([a.primary, b.primary, "--json"])
        self.assertEqual(code, 1)
        report = json.loads(text)
        self.assert_only_class_differs(report, "QuadSourceIntegral")
        rows = report["classes"]["QuadSourceIntegral"]["rows"]
        self.assertEqual(rows["differ"], 1)
        self.assertEqual(rows["identical"], 2)
        self.assertEqual(list(rows["columns"]), ["total"])
        (entry,) = rows["rows"]
        self.assertEqual(entry["identity"], expected)
        self.assertEqual(list(entry["columns"]), ["total"])
        diff = entry["columns"]["total"]
        self.assertEqual(diff["kind"], "float")
        self.assertEqual(diff["a"], float.hex(old))
        self.assertEqual(diff["b"], float.hex(new))
        self.assertEqual(diff["delta"], new - old)
        self.assertEqual(rows["columns"]["total"]["max_abs_delta"], new - old)
        self.assertEqual(rows["columns"]["total"]["max_rel_delta"], (new - old) / old)

        # and the text names the class, the identity, the column and the delta
        code, text = _run([a.primary, b.primary])
        self.assertEqual(code, 1)
        self.assertIn("QuadSourceIntegral: NOT value-identical", text)
        self.assertIn(f"identity {expected[:16]}", text)
        self.assertIn(
            f"total: A {float.hex(old)} ({old:.6g}), B {float.hex(new)} ({new:.6g}), "
            f"delta {new - old:.6g}",
            text,
        )

    def test_t2_one_ulp_in_a_value_table_row(self):
        a = self.build()
        replicated, sharded, keys = full_rows()
        row = find_row(replicated, sharded, "TkNumericValue", TK_VALUE_SERIAL)
        old = row["T"]
        new = math.nextafter(old, -math.inf)
        vary_row(replicated, sharded, "TkNumericValue", TK_VALUE_SERIAL, "T", new)
        b = self.build((replicated, sharded, keys))

        report = compare_stores(a.primary, b.primary, _factories)
        self.assert_only_class_differs(report, "TkNumericIntegration")
        cls = report["classes"]["TkNumericIntegration"]
        self.assertEqual(cls["rows"]["differ"], 0)
        values = cls["values"]["TkNumericValue"]
        self.assertEqual(values["differ"], 1)
        self.assertEqual(values["identical"], 8)
        (entry,) = values["rows"]
        self.assertEqual(list(entry["columns"]), ["T"])
        self.assertEqual(entry["columns"]["T"]["a"], float.hex(old))
        self.assertEqual(entry["columns"]["T"]["b"], float.hex(new))
        self.assertEqual(entry["columns"]["T"]["delta"], new - old)

        # its identity: the parent's, resolved by the inventory itself, and the redshift it names
        inventory = read_inventory(a.primary, _factories)
        (parent,) = inventory.find("TkNumericIntegration", entry["parent"]["identity"])
        resolved = inventory.resolve("TkNumericIntegration", parent)
        exit_time = resolved["k"]["key"]
        self.assertEqual(
            float.fromhex(exit_time["k"]["key"]["k_inv_Mpc"]), TK_VALUE_PARENT_K
        )
        self.assertEqual(exit_time["cosmology"]["class"], "LambdaCDM")
        self.assertEqual(entry["z_serial"]["class"], "redshift")
        self.assertEqual(entry["z_serial"]["z"], TK_VALUE_Z)


class TestOnlyInOne(_StoreTest):
    def test_t3_a_deleted_row_is_in_one_store_only(self):
        a = self.build()
        replicated, sharded, keys = full_rows()
        _without_row(
            replicated,
            sharded,
            "QuadSourceIntegral",
            QSI_SERIAL,
            "QuadSourceIntegral_tags",
        )
        b = self.build((replicated, sharded, keys))
        (expected,) = _digests(a.primary, "QuadSourceIntegral") - _digests(
            b.primary, "QuadSourceIntegral"
        )

        code, text = _run([a.primary, b.primary])
        self.assertEqual(code, 1)
        self.assertIn(f"only in A: identity {expected[:16]}", text)

        report = compare_stores(a.primary, b.primary, _factories)
        self.assert_only_class_differs(report, "QuadSourceIntegral")
        rows = report["classes"]["QuadSourceIntegral"]["rows"]
        self.assertEqual([e["identity"] for e in rows["only_in_a"]], [expected])
        self.assertEqual(rows["only_in_b"], [])
        self.assertEqual(rows["differ"], 0)
        self.assertEqual(rows["identical"], 2)

        # and the other way round
        report = compare_stores(b.primary, a.primary, _factories)
        rows = report["classes"]["QuadSourceIntegral"]["rows"]
        self.assertEqual([e["identity"] for e in rows["only_in_b"]], [expected])
        self.assertEqual(rows["only_in_a"], [])


class TestNeverBySerial(_StoreTest):
    def test_t4_renumbered_serials_are_value_identical(self):
        a = self.build()
        replicated, sharded, keys = relabel_serials(*full_rows())
        b = self.build((replicated, sharded, keys))

        # the serials really are different: no QuadSourceIntegral or TkNumericValue serial of the
        # copy is one of the original's
        original_r, original_s, _ = full_rows()
        for table in ("QuadSourceIntegral", "TkNumericValue", "redshift"):
            before = {
                r["serial"]
                for rows in [original_r] + list(original_s.values())
                for r in rows.get(table, [])
            }
            after = {
                r["serial"]
                for rows in [replicated] + list(sharded.values())
                for r in rows.get(table, [])
            }
            self.assertEqual(len(before), len(after), table)
            self.assertEqual(before & after, set(), table)

        code, text = _run([a.primary, b.primary])
        self.assertEqual(code, 0, text)
        report = compare_stores(a.primary, b.primary, _factories)
        self.assertTrue(report["identical"])
        for name, c in report["classes"].items():
            self.assertTrue(c["identical"], name)


class TestJSONColumns(_StoreTest):
    METADATA = {"alpha": 1, "beta": {"x": 0.5, "y": [1, 2.5]}, "gamma": "s"}

    def _with_metadata(self, text):
        replicated, sharded, keys = full_rows()
        vary_row(
            replicated, sharded, "QuadSourceIntegral", QSI_SERIAL, "metadata", text
        )
        return self.build((replicated, sharded, keys))

    def test_t5_json_key_order_is_immaterial_and_a_changed_value_is_named(self):
        a = self._with_metadata(json.dumps(self.METADATA))
        reordered = '{"gamma": "s", "beta": {"y": [1, 2.5], "x": 0.5}, "alpha": 1}'
        self.assertNotEqual(reordered, json.dumps(self.METADATA))
        self.assertEqual(json.loads(reordered), self.METADATA)
        b = self._with_metadata(reordered)

        code, text = _run([a.primary, b.primary])
        self.assertEqual(code, 0, text)

        changed = json.loads(reordered)
        changed["beta"]["x"] = 0.75
        c = self._with_metadata(json.dumps(changed))
        code, text = _run([a.primary, c.primary, "--json"])
        self.assertEqual(code, 1)
        report = json.loads(text)
        self.assert_only_class_differs(report, "QuadSourceIntegral")
        rows = report["classes"]["QuadSourceIntegral"]["rows"]
        self.assertEqual(rows["differ"], 1)
        (entry,) = rows["rows"]
        self.assertEqual(
            entry["columns"]["metadata"], {"kind": "json", "keys": ["beta.x"]}
        )
        self.assertEqual(rows["columns"]["metadata"]["json_keys"], {"beta.x": 1})


if __name__ == "__main__":
    unittest.main()
