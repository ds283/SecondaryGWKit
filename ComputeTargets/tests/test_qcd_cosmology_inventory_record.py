"""
The claims of the retired ``ComputeTargets/tests/test_qcd_cosmology_inventory.py``
(``prompts/background-solver-robustness`` prompt 07), re-expressed against the structured
inventory and its display (store-fingerprint prompt 03, decision D4 of 2026-09-25).

Since ``prompts/qcd-background-audit`` prompt 03, ``T_z_representation`` is part of the QCD
cosmology's lookup key: two rows can share every physical parameter and ``log10_max_z`` and still
be different backgrounds. The retired module tested that ``sqla_QCDCosmology_factory``'s old
per-factory report showed the column ([03-qcd-inventory-does-not-report-the-representation]).
That report is gone. The same three claims, against the service that replaces it:

1. Two rows that differ only in ``T_z_representation`` are two records of the structured
   inventory, and render differently.
2. In the rendered record, ``T_z_representation`` sits beside ``log10_max_z``. The display follows
   the order of the class's key as its factory's ``inventory_records`` declares it, where the
   representation comes directly **after** ``log10_max_z``; the retired test required it directly
   before. Either way the two fields that say how the background was computed, and over what
   range, sit together.
3. **This claim changes meaning.** The retired test asserted that two rows with the same
   representation and different ``name`` were two labels. ``name`` is descriptive and not in the
   key, so under the structured inventory they are **one key held twice**: both are kept as
   records, and the inventory names them a ``duplicate`` problem, which the display prints.

Stores are built with ``Datastore/tests/real_store_fixtures.build_full_store`` in a temporary
directory. No Ray; nothing under ``var/``.
"""

import copy
import tempfile
import unittest
from pathlib import Path

from Datastore.store_inventory import read_inventory
from Datastore.tests.inventory_report_parsing import fields_of, sections
from Datastore.tests.real_store_fixtures import (
    build_full_store,
    full_rows,
    with_rows,
)
from tools.inventory_report import format_inventory_report

KEY_COLUMN = "T_z_representation"


class TestTheRepresentationInTheInventory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name).resolve()
        cls._count = 0

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def inventory_with_second_qcd_row(self, **changes):
        """The full store, with a second QCD_Cosmology row: a copy of the first with ``changes``."""
        replicated, sharded, keys = full_rows()
        (first,) = replicated["QCD_Cosmology"]
        second = copy.deepcopy(first)
        second["serial"] = 2
        second.update(changes)
        replicated = with_rows(replicated, {"QCD_Cosmology": [second]})
        type(self)._count += 1
        directory = self.root / f"store-{self._count}"
        directory.mkdir()
        store = build_full_store(
            directory, replicated=replicated, sharded=sharded, shard_keys=keys
        )
        return read_inventory(store.primary)

    def qcd_lines(self, inventory, verbose):
        section = sections(
            format_inventory_report(inventory, "store", verbose=verbose)
        )["QCD_Cosmology"]
        return section, [line for _, line in section.records]

    def test_two_rows_differing_only_in_representation_are_two_records(self):
        inventory = self.inventory_with_second_qcd_row(**{KEY_COLUMN: 4})
        records = inventory["QCD_Cosmology"].records
        self.assertEqual(len(records), 2)
        self.assertEqual({r.key[KEY_COLUMN] for r in records}, {3, 4})
        # every other key field is the same, so the representation is what tells them apart
        for field in records[0].key:
            if field != KEY_COLUMN:
                self.assertEqual(records[0].key[field], records[1].key[field])
        self.assertEqual(inventory["QCD_Cosmology"].problems, ())

        for verbose in (False, True):
            with self.subTest(verbose=verbose):
                section, lines = self.qcd_lines(inventory, verbose)
                self.assertEqual(len(lines), 2)
                self.assertNotEqual(lines[0], lines[1])
                self.assertEqual(
                    sorted(dict(fields_of(line))[KEY_COLUMN] for line in lines),
                    ["3", "4"],
                )
                # the fields the two share are stated once, and the parameters are among them
                self.assertIn("omega_m", dict(fields_of(section.common)))
                self.assertIn("log10_max_z", dict(fields_of(section.common)))

    def test_the_representation_sits_beside_log10_max_z(self):
        replicated, sharded, keys = full_rows()
        directory = self.root / "one-qcd"
        directory.mkdir()
        store = build_full_store(
            directory, replicated=replicated, sharded=sharded, shard_keys=keys
        )
        inventory = read_inventory(store.primary)
        for verbose in (False, True):
            with self.subTest(verbose=verbose):
                _, (line,) = self.qcd_lines(inventory, verbose)
                names = [name for name, _ in fields_of(line)]
                self.assertIn(KEY_COLUMN, names)
                self.assertEqual(
                    names.index(KEY_COLUMN), names.index("log10_max_z") + 1, names
                )

    def test_a_different_name_is_one_key_held_twice(self):
        # this claim changes meaning under the structured inventory (module docstring, item 3)
        inventory = self.inventory_with_second_qcd_row(name="another-name")
        cls = inventory["QCD_Cosmology"]
        self.assertEqual(cls.count, 2)
        self.assertEqual(cls.records[0].key, cls.records[1].key)
        (problem,) = cls.problems
        self.assertTrue(problem.startswith("duplicate: QCD_Cosmology: "), problem)
        self.assertIn("1 key-and-tag set(s) are held by more than one record", problem)

        for verbose in (False, True):
            with self.subTest(verbose=verbose):
                section, lines = self.qcd_lines(inventory, verbose)
                # one key, so one rendering, twice; the problem is printed in full
                self.assertEqual(len(lines), 2)
                self.assertEqual(lines[0], lines[1])
                self.assertEqual(section.problems, [problem])
                self.assertNotIn("another-name", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
