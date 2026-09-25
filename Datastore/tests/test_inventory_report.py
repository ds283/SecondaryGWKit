"""
Tests for store-fingerprint prompt 03, F8 items 1-3: ``tools/inventory_report.format_inventory_report``
renders a ``StoreInventory`` and says what it holds.

1. **The display says what the inventory holds.** Every class, with its count, tag sets,
   validated split and value-count sum; with ``verbose`` every two distinct records of a class
   are distinct lines, and every key field of every class is shown; no 64-hex digest appears;
   every problem appears, in full, when there are many.
2. **Floats** are typed by the schema: the last bit of one ``k`` shows in verbose mode and not at
   six figures, and a string that looks like a hex float is printed as the string it is.
3. **The representation and the grid identity**: a ``BackgroundModel`` record shows
   ``source_grid_digest`` and ``source_grid_construction``, and a ``QCD_Cosmology`` record shows
   ``T_z_representation`` ([03-qcd-inventory-does-not-report-the-representation]).

Stores are built with ``Datastore/tests/real_store_fixtures.build_full_store`` in a temporary
directory. No Ray; nothing under ``var/``.
"""

import math
import sqlite3
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from Datastore.store_inventory import INVENTORY_CLASSES, read_inventory
from Datastore.tests.inventory_report_parsing import (
    HEX64,
    fields_of,
    sections,
)
from Datastore.tests.real_store_fixtures import (
    build_full_store,
    full_rows,
    vary_row,
    with_rows,
)
from tools.inventory_report import (
    INVENTORY_CATEGORIES,
    MAX_SHOWN,
    VALUE_TABLES,
    format_inventory_report,
)

DB_NAME = "the-store.sqlite"


class _Stores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name).resolve()
        cls._count = 0
        cls.full = cls.store()
        cls.inventory = read_inventory(cls.full.primary)
        cls.report = format_inventory_report(cls.inventory, DB_NAME)
        cls.verbose = format_inventory_report(cls.inventory, DB_NAME, verbose=True)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def store(cls, **kwargs):
        cls._count += 1
        directory = cls.root / f"store-{cls._count}"
        directory.mkdir()
        return build_full_store(directory, **kwargs)

    @classmethod
    def read(cls, **kwargs):
        return read_inventory(cls.store(**kwargs).primary)


class TestTheDisplaySaysWhatTheInventoryHolds(_Stores):
    def test_the_categories_cover_every_class_once(self):
        named = [n for _, names in INVENTORY_CATEGORIES for n in names]
        self.assertEqual(sorted(named), sorted(INVENTORY_CLASSES))
        self.assertEqual(len(named), len(set(named)))
        self.assertEqual(set(sections(self.report)), set(INVENTORY_CLASSES))

    def test_the_title_and_the_problem_count(self):
        lines = self.report.split("\n")
        self.assertEqual(lines[0], f"== Datastore inventory: {DB_NAME} ==")
        self.assertEqual(self.inventory.problems, ())
        self.assertEqual(lines[1], "   problems: none")

    def test_every_class_with_its_count_validated_split_and_value_sum(self):
        for report in (self.report, self.verbose):
            found = sections(report)
            for name in INVENTORY_CLASSES:
                cls = self.inventory[name]
                section = found[name]
                with self.subTest(cls=name, verbose=report is self.verbose):
                    self.assertEqual(section.count, cls.count)
                    self.assertEqual(
                        section.where, "replicated" if cls.replicated else "sharded"
                    )
                    flags = [r.validated for r in cls.records]
                    if any(f is not None for f in flags):
                        self.assertIn(
                            f"{flags.count(True):,} validated, "
                            f"{flags.count(False):,} unvalidated",
                            section.header,
                        )
                    else:
                        self.assertNotIn("validated", section.header)
                    if name in VALUE_TABLES:
                        total = sum(r.value_count for r in cls.records)
                        self.assertIn(
                            f"{total:,} {VALUE_TABLES[name]} row", section.header
                        )
                    else:
                        self.assertNotIn("Value row", section.header)

    def test_the_value_tables_named_are_the_ones_counted(self):
        # value_count is present exactly on the classes VALUE_TABLES names, and each sum is the
        # value table's own row count, read with sqlite3 independently of the inventory (on one
        # shard for a replicated class, whose value rows every shard holds; summed otherwise)
        def rows_in(table, paths):
            total = 0
            for path in paths:
                conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                try:
                    total += conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[
                        0
                    ]
                finally:
                    conn.close()
            return total

        shards = [self.full.shard_files[s] for s in sorted(self.full.shard_files)]
        for name in INVENTORY_CLASSES:
            cls = self.inventory[name]
            with self.subTest(cls=name):
                has = {r.value_count is not None for r in cls.records}
                self.assertEqual(has, {name in VALUE_TABLES})
                if name in VALUE_TABLES:
                    paths = shards[:1] if cls.replicated else shards
                    self.assertEqual(
                        sum(r.value_count for r in cls.records),
                        rows_in(VALUE_TABLES[name], paths),
                    )

    def test_the_tag_sets_and_their_counts(self):
        found = sections(self.report)
        for name in INVENTORY_CLASSES:
            cls = self.inventory[name]
            with self.subTest(cls=name):
                if not cls.tagged:
                    self.assertEqual(found[name].tag_sets, [])
                    continue
                expected = Counter(r.tags for r in cls.records)
                shown = {
                    tuple(labels.split(", ")) if labels != "(no tags)" else (): n
                    for _, labels, n in found[name].tag_sets
                }
                self.assertEqual(shown, dict(expected))

    def test_verbose_renders_every_distinct_record_as_a_distinct_line(self):
        # the test that the display drops no key field: a display that left one out would make
        # two records that differ only in it collide. Lines are compared within their tag set
        found = sections(self.verbose)
        for name in INVENTORY_CLASSES:
            cls = self.inventory[name]
            with self.subTest(cls=name):
                section = found[name]
                self.assertEqual(len(section.records), cls.count)
                self.assertEqual(section.more, 0)
                self.assertEqual(len(set(section.records)), len(set(cls.records)))

    def test_verbose_shows_every_key_field_of_every_class(self):
        found = sections(self.verbose)
        for name in INVENTORY_CLASSES:
            cls = self.inventory[name]
            with self.subTest(cls=name):
                section = found[name]
                shown = set()
                if section.common is not None:
                    shown |= {f for f, _ in fields_of(section.common)}
                for _, line in section.records:
                    shown |= {f for f, _ in fields_of(line)}
                self.assertEqual(shown, set(cls.records[0].key))

    def test_no_digest_appears(self):
        for report in (self.report, self.verbose):
            self.assertIsNone(HEX64.search(report))

    def test_the_store_tag_class_lists_every_label_and_marks_the_uncarried(self):
        for report in (self.report, self.verbose):
            section = sections(report)["store_tag"]
            lines = dict((fields_of(line)[0][1], line) for _, line in section.records)
            labels = {r.key["label"] for r in self.inventory["store_tag"].records}
            self.assertEqual(set(lines), labels)
            self.assertIn("(carried by no record)", lines["unused-tag"])
            for label in labels - {"unused-tag"}:
                self.assertNotIn("carried by no record", lines[label])

    def test_records_are_ordered_numerically(self):
        section = sections(self.report)["tolerance"]
        self.assertEqual(
            [line for _, line in section.records],
            ["log10_tol=-9", "log10_tol=-7", "log10_tol=-5"],
        )

    def test_without_verbose_at_most_five_records_per_tag_set(self):
        replicated, sharded, keys = full_rows()
        extra = [
            {"serial": 10 + i, "z": 2.0 + i, "source": True, "response": False}
            for i in range(4)
        ]
        inventory = self.read(
            replicated=with_rows(replicated, {"redshift": extra}),
            sharded=sharded,
            shard_keys=keys,
        )
        self.assertGreater(inventory["redshift"].count, MAX_SHOWN)
        section = sections(format_inventory_report(inventory, DB_NAME))["redshift"]
        self.assertEqual(len(section.records), MAX_SHOWN)
        self.assertEqual(section.more, inventory["redshift"].count - MAX_SHOWN)
        # the five shown are the five smallest redshifts, in order
        shown = [float(fields_of(line)[0][1]) for _, line in section.records]
        self.assertEqual(
            shown,
            sorted(float.fromhex(r.key["z"]) for r in inventory["redshift"].records)[
                :MAX_SHOWN
            ],
        )

    def test_every_problem_appears_in_full_when_there_are_many(self):
        absent = [
            "TkNumeric_tags",
            "TkWKB_tags",
            "GkNumeric_tags",
            "GkWKB_tags",
            "GkSource_tags",
            "QuadSource_tags",
            "QuadSourceIntegral_tags",
        ]
        # seven shards: every shard but #0 lacks QuadSourceIntegral_tags, so that one class has
        # more than five problems, and shard #1 lacks six more association tables besides
        missing = {s: ["QuadSourceIntegral_tags"] for s in range(1, 7)}
        missing[1] = absent
        inventory = self.read(shards=7, missing_tables=missing)
        problems = inventory.problems
        self.assertGreater(len(problems), MAX_SHOWN)
        self.assertGreater(len(inventory["QuadSourceIntegral"].problems), MAX_SHOWN)
        for verbose in (False, True):
            report = format_inventory_report(inventory, DB_NAME, verbose=verbose)
            with self.subTest(verbose=verbose):
                self.assertEqual(
                    report.split("\n")[1],
                    f"   problems: {len(problems)}, each listed in full under its class",
                )
                shown = [p for s in sections(report).values() for p in s.problems]
                self.assertEqual(sorted(shown), sorted(problems))
                for problem in problems:
                    self.assertIn(f"!! {problem}\n", report + "\n")


class TestFloats(_Stores):
    def test_the_last_bit_of_k_shows_in_verbose_mode_only(self):
        replicated, sharded, keys = full_rows()
        vary_row(
            replicated, sharded, "wavenumber", 1, "k_inv_Mpc", math.nextafter(0.1, 1.0)
        )
        other = self.read(replicated=replicated, sharded=sharded, shard_keys=keys)
        self.assertNotEqual(
            format_inventory_report(other, DB_NAME, verbose=True), self.verbose
        )
        self.assertEqual(format_inventory_report(other, DB_NAME), self.report)
        line = sections(format_inventory_report(other, DB_NAME, verbose=True))[
            "wavenumber"
        ].records[0][1]
        self.assertEqual(line, f"k_inv_Mpc={math.nextafter(0.1, 1.0)!r}")

    def test_floats_are_numbers_at_six_figures_and_repr_in_verbose(self):
        replicated, sharded, keys = full_rows()
        vary_row(replicated, sharded, "wavenumber", 1, "k_inv_Mpc", 1.0 / 3.0)
        inventory = self.read(replicated=replicated, sharded=sharded, shard_keys=keys)
        short = sections(format_inventory_report(inventory, DB_NAME))["wavenumber"]
        full = sections(format_inventory_report(inventory, DB_NAME, verbose=True))[
            "wavenumber"
        ]
        self.assertIn((0, "k_inv_Mpc=0.333333"), short.records)
        self.assertIn((0, f"k_inv_Mpc={1.0 / 3.0!r}"), full.records)

    def test_a_string_that_looks_like_a_hex_float_is_printed_as_it_is(self):
        replicated, sharded, keys = full_rows()
        hexlike = float.hex(3.0)
        replicated = with_rows(
            replicated,
            {
                "store_tag": [{"serial": 20, "label": hexlike}],
                "IntegrationSolver": [
                    {"serial": 20, "stepping": 0, "label": float.hex(8.0)}
                ],
            },
        )
        inventory = self.read(replicated=replicated, sharded=sharded, shard_keys=keys)
        for verbose in (False, True):
            report = format_inventory_report(inventory, DB_NAME, verbose=verbose)
            found = sections(report)
            with self.subTest(verbose=verbose):
                self.assertIn(
                    f"label={hexlike}",
                    [line.split("  ")[0] for _, line in found["store_tag"].records],
                )
                self.assertIn(
                    (0, f"label={float.hex(8.0)}, stepping=0"),
                    found["IntegrationSolver"].records,
                )
                # a float leaf is never printed as its hex form: the only "0x" in the report are
                # these two labels
                self.assertEqual(report.count("0x"), 2)


class TestTheRepresentationAndTheGridIdentity(_Stores):
    def test_background_model_records_show_the_grid_identity(self):
        for report in (self.report, self.verbose):
            section = sections(report)["BackgroundModel"]
            shown = [dict(fields_of(line)) for _, line in section.records]
            common = dict(fields_of(section.common)) if section.common else {}
            with self.subTest(verbose=report is self.verbose):
                digests = sorted(
                    s.get("source_grid_digest", common.get("source_grid_digest"))
                    for s in shown
                )
                self.assertEqual(digests, ["fixture-digest", "fixture-digest-qcd"])
                for s in shown:
                    self.assertEqual(
                        s.get(
                            "source_grid_construction",
                            common.get("source_grid_construction"),
                        ),
                        "1",
                    )

    def test_a_lone_background_model_shows_every_field(self):
        replicated, sharded, keys = full_rows()
        # keep BackgroundModel 1 only: drop the QCD one and every row that refers to it
        replicated["BackgroundModel"] = [
            r for r in replicated["BackgroundModel"] if r["serial"] == 1
        ]
        replicated["BackgroundModel_tags"] = [
            r for r in replicated["BackgroundModel_tags"] if r["model_serial"] == 1
        ]
        replicated["BackgroundModelValue"] = [
            r for r in replicated["BackgroundModelValue"] if r["model_serial"] == 1
        ]
        for rows in sharded.values():
            for table, table_rows in rows.items():
                if table_rows and "model_serial" in table_rows[0]:
                    rows[table] = [r for r in table_rows if r["model_serial"] == 1]
        inventory = self.read(replicated=replicated, sharded=sharded, shard_keys=keys)
        self.assertEqual(inventory["BackgroundModel"].count, 1)
        section = sections(format_inventory_report(inventory, DB_NAME))[
            "BackgroundModel"
        ]
        self.assertIsNone(section.common)
        (line,) = [line for _, line in section.records]
        shown = dict(fields_of(line))
        self.assertEqual(shown["source_grid_digest"], "fixture-digest")
        self.assertEqual(shown["source_grid_construction"], "1")

    def test_qcd_cosmology_records_show_the_representation(self):
        for report in (self.report, self.verbose):
            section = sections(report)["QCD_Cosmology"]
            (line,) = [line for _, line in section.records]
            self.assertEqual(dict(fields_of(line))["T_z_representation"], "3")


if __name__ == "__main__":
    unittest.main()
