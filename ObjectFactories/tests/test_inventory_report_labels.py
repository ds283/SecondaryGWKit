"""
Tests for datastore-integrity prompt 09d: ``tools/inventory_report`` refers to a **derived** row,
which is a row of a sharded class that another class's key names, by a label ``<Class>#<n>``, and
prints that row once, in its own section. A parent set (a ``GkSource``'s) is the same, ``S<n>``.

What the report must keep, and what these tests pin:

1. **Distinct stays distinct** (P1). Two distinct records never render as one line, though a record
   no longer carries its parents' keys: pairs that differ only in one member of a ``GkSource``'s
   set, only through that set, in one Tk parent of an integral, or in its ``QuadSource``.
   A derived class's record line also carries the record's own label, which would make any two
   of its lines differ, so these tests take the label off first.
2. **Every label is defined, once** (P2), in every report, verbose or not: the ``Class#n`` and
   ``S<n>`` that a line names have exactly one defining line, and it is the row or set named. A
   reference is always a label where its class is sharded, and the rows a shown record names that
   the report did not show are listed under their own prefix.
3. **Labels are stable.** The same store reported twice, the same store under other serials, and the
   same inventory with every class's records in another order, give the same report: a label is
   the record's ordinal in the physical order, never its place in the order it was read.
4. **The size is bounded** (P3). A record line does not grow with the number of members of a
   parent set, and a set's line grows by a bounded number of characters a member.
5. **Nothing else changed.** A replicated class, and a class that has no sharded parent and that
   nothing names, prints no label and no listing.

The stand-in stores of 1, 2 and 4 are inventories built in memory from the fixture's own
replicated rows (``_Stand``): the renderer reads nothing from a store but the inventory it is
given, and a set of production size is not something the fixture's SQLite store should hold.
Tests 2 and 3 also run on the fixture store itself. No Ray; nothing under ``var/``.
"""

import dataclasses
import re
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from unittest import mock

from datastorekit.store_inventory import (
    Record,
    StoreInventory,
    digest,
    read_inventory,
    reference_digest,
)
from ObjectFactories.tests.inventory_report_parsing import HEX64, fields_of, sections
from ObjectFactories.tests.real_store_fixtures import (
    build_full_store,
    full_rows,
    relabel_serials,
)
from tools.inventory_report import format_inventory_report
from config.datastore import factories as _factories

DB_NAME = "the-store.sqlite"

DERIVED_CLASSES = (
    "TkNumericIntegration",
    "TkWKBIntegration",
    "GkNumericIntegration",
    "GkWKBIntegration",
    "GkSource",
    "GkSourcePolicyData",
    "QuadSource",
    "QuadSourceIntegral",
    "OneLoopIntegral",
)

# the label of a record, beyond its fields, and the ones that name a row or a set
OWN_LABEL = re.compile(r"  #(\d+)(?=  |$)")
ROW_LABEL = re.compile(r"\b([A-Za-z]\w*)#(\d+)\b")
SET_LABEL = re.compile(r"parents=S(\d+)\b")
SET_LINE = re.compile(r"^         parent set S(\d+): (.*)$")
LISTED = "            > "
SHOWN = "            - "


class _Stand:
    """An inventory built in memory over the fixture's replicated rows, with derived rows of its
    own. A derived row is the fixture's first row of its class with some key fields replaced; a
    parent is named by its reference digest, as the inventory names it."""

    def __init__(self, base: StoreInventory, z_values: Sequence[float] = ()):
        self.base = base
        self.tagged = {n: c.tagged for n, c in base.classes.items()}
        self.rows: Dict[str, List[Record]] = {n: [] for n in DERIVED_CLASSES}
        self.members: Dict[str, Tuple[Tuple[Optional[str], ...], ...]] = {}
        self.redshift = list(base["redshift"].records) + [
            Record(key={"z": float(z).hex()}, tags=(), validated=None, value_count=None)
            for z in z_values
        ]
        self.z = [self.d("redshift", r) for r in self.redshift]
        self.exit = [
            self.d("wavenumber_exit_time", r)
            for r in self.records("wavenumber_exit_time")
        ]
        self.tol = [self.d("tolerance", r) for r in self.records("tolerance")]
        self.model = self.d("BackgroundModel", self.records("BackgroundModel")[0])
        self.policy = self.d("GkSourcePolicy", self.records("GkSourcePolicy")[0])
        self.quad_policy = self.d(
            "QuadSourcePolicy", self.records("QuadSourcePolicy")[0]
        )

    def records(self, name: str) -> Sequence[Record]:
        return self.base[name].records

    def d(self, name: str, record: Record) -> str:
        return reference_digest(record.key, record.tags, self.tagged[name])

    def row(self, name: str, tags: Optional[Tuple[str, ...]] = None, **over) -> Record:
        template = self.base[name].records[0]
        key = {**template.key, **over}
        record = Record(
            key=key,
            tags=template.tags if tags is None else tags,
            validated=template.validated,
            value_count=template.value_count,
        )
        self.rows[name].append(record)
        return record

    def ref(self, name: str, record: Optional[Record]) -> Optional[str]:
        return None if record is None else self.d(name, record)

    def parent_set(
        self, members: Sequence[Tuple[int, Optional[Record], Optional[Record]]]
    ) -> str:
        """A set of (index into ``self.z``, Gk numeric row or None, Gk WKB row or None)."""
        resolved = tuple(
            (
                self.z[z],
                self.ref("GkNumericIntegration", numeric),
                self.ref("GkWKBIntegration", wkb),
            )
            for z, numeric, wkb in members
        )
        name = digest([list(m) for m in sorted(resolved, key=repr)])
        self.members[name] = resolved
        return name

    def as_inventory(self) -> StoreInventory:
        classes = dict(self.base.classes)
        classes["redshift"] = dataclasses.replace(
            classes["redshift"],
            records=tuple(self.redshift),
            count=len(self.redshift),
        )
        for name, rows in self.rows.items():
            classes[name] = dataclasses.replace(
                classes[name],
                records=tuple(rows),
                count=len(rows),
                problems=(),
                members={"parents": dict(self.members)} if name == "GkSource" else {},
            )
        return StoreInventory(
            primary=self.base.primary, shards=self.base.shards, classes=classes
        )


KIND = {0: "kind-0", 1: "kind-1"}
RHO = {0: 11, 1: 10}


def _chain(stand: _Stand, set_a: str, set_b: Optional[str] = None) -> Dict[str, Record]:
    """The rows of one integral and everything it names, over the parent set ``set_a``; and, with
    ``set_b``, a second ``GkSource`` over that set."""
    # each Tk row has a leaf of its own, which its line shows: ``break_point_kind=kind-<i>`` for a
    # numeric row and ``rho_gauss_order=<n>`` for a WKB one. The WKB rows order the other way from
    # the numeric rows they name, so that the order a label is met in is not the physical order
    t = {i: stand.row("TkNumericIntegration", break_point_kind=KIND[i]) for i in (0, 1)}
    w = {
        i: stand.row(
            "TkWKBIntegration",
            rho_gauss_order=RHO[i],
            numeric=stand.ref("TkNumericIntegration", t[i]),
        )
        for i in (0, 1)
    }
    source = stand.row(
        "QuadSource",
        Tq=stand.ref("TkNumericIntegration", t[0]),
        Tr=stand.ref("TkNumericIntegration", t[1]),
    )
    g_a = stand.row("GkSource", parents=set_a)
    data_a = stand.row("GkSourcePolicyData", source=stand.ref("GkSource", g_a))
    out = {"t0": t[0], "t1": t[1], "w0": w[0], "w1": w[1], "source": source}
    out.update({"g_a": g_a, "data_a": data_a})
    if set_b is not None:
        g_b = stand.row("GkSource", parents=set_b)
        data_b = stand.row("GkSourcePolicyData", source=stand.ref("GkSource", g_b))
        out.update({"g_b": g_b, "data_b": data_b})
    return out


def _integral(stand: _Stand, rows: Dict[str, Record], **over) -> Record:
    fields = dict(
        source=stand.ref("QuadSource", rows["source"]),
        data=stand.ref("GkSourcePolicyData", rows["data_a"]),
        Tq_numeric=stand.ref("TkNumericIntegration", rows["t0"]),
        Tq_WKB=stand.ref("TkWKBIntegration", rows["w0"]),
        Tr_numeric=stand.ref("TkNumericIntegration", rows["t1"]),
        Tr_WKB=stand.ref("TkWKBIntegration", rows["w1"]),
    )
    fields.update(over)
    return stand.row("QuadSourceIntegral", **fields)


class _Fixture(unittest.TestCase):
    """The fixture store, read once."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name).resolve()
        cls.base = cls.read(cls.root / "base", {})
        cls.stand_base = cls.base

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @staticmethod
    def read(directory: Path, kwargs) -> StoreInventory:
        directory.mkdir()
        return read_inventory(build_full_store(directory, **kwargs).primary, _factories)


def report(
    inventory: StoreInventory, verbose: bool, shown: Optional[int] = None
) -> str:
    """The report; with ``shown``, at most that many records per tag set without ``verbose``."""
    if shown is None:
        return format_inventory_report(inventory, DB_NAME, verbose=verbose)
    with mock.patch("tools.inventory_report.MAX_SHOWN", shown):
        return format_inventory_report(inventory, DB_NAME, verbose=verbose)


def record_lines(text: str, name: str) -> List[str]:
    return [line for _, line in sections(text)[name].records]


# --- 1. distinct stays distinct ---------------------------------------------------------------


class TestDistinctStaysDistinct(_Fixture):
    def assert_two_lines(self, inventory: StoreInventory, name: str):
        for verbose in (False, True):
            with self.subTest(cls=name, verbose=verbose):
                # a record's own label would make any two lines of a derived class differ, so
                # it is taken off: the fields alone must tell the two records apart
                lines = [
                    OWN_LABEL.sub("", line)
                    for line in record_lines(report(inventory, verbose), name)
                ]
                self.assertEqual(len(lines), 2)
                self.assertNotEqual(lines[0], lines[1])

    def test_gksource_differing_in_the_gk_row_at_one_z_source(self):
        # the case prompt 09b keys on, with the same member count and z_source range. The two
        # sets name the same z_source values, and differ in one Gk numeric row, or one Gk WKB row
        for which in ("numeric", "wkb"):
            with self.subTest(differing=which):
                stand = _Stand(self.base)
                n = [
                    stand.row("GkNumericIntegration", rtol=stand.tol[i]) for i in (0, 1)
                ]
                g = [
                    stand.row("GkWKBIntegration", rho_gauss_order=10 + i)
                    for i in (0, 1)
                ]
                a = stand.parent_set([(0, n[0], None), (1, None, g[0])])
                b = stand.parent_set(
                    [(0, n[1], None), (1, None, g[0])]
                    if which == "numeric"
                    else [(0, n[0], None), (1, None, g[1])]
                )
                stand.row("GkSource", parents=a)
                stand.row("GkSource", parents=b)
                self.assert_two_lines(stand.as_inventory(), "GkSource")

    def test_gksourcepolicydata_and_integral_differing_only_through_the_set(self):
        stand = _Stand(self.base)
        n = [stand.row("GkNumericIntegration", rtol=stand.tol[i]) for i in (0, 1)]
        g = stand.row("GkWKBIntegration")
        a = stand.parent_set([(0, n[0], None), (1, None, g)])
        b = stand.parent_set([(0, n[1], None), (1, None, g)])
        rows = _chain(stand, a, b)
        _integral(stand, rows, data=stand.ref("GkSourcePolicyData", rows["data_a"]))
        _integral(stand, rows, data=stand.ref("GkSourcePolicyData", rows["data_b"]))
        inventory = stand.as_inventory()
        self.assert_two_lines(inventory, "GkSourcePolicyData")
        self.assert_two_lines(inventory, "QuadSourceIntegral")

    def test_integrals_differing_only_in_one_tk_parent(self):
        for parent, other in (
            ("Tq_numeric", "t1"),
            ("Tr_numeric", "t0"),
            ("Tq_WKB", "w1"),
            ("Tr_WKB", "w0"),
        ):
            with self.subTest(parent=parent):
                stand = _Stand(self.base)
                a = stand.parent_set([(0, None, None)])
                rows = _chain(stand, a)
                _integral(stand, rows)
                name = "TkNumericIntegration" if other[0] == "t" else "TkWKBIntegration"
                _integral(stand, rows, **{parent: stand.ref(name, rows[other])})
                self.assert_two_lines(stand.as_inventory(), "QuadSourceIntegral")

    def test_integrals_differing_only_in_their_quadsource(self):
        stand = _Stand(self.base)
        a = stand.parent_set([(0, None, None)])
        rows = _chain(stand, a)
        other = stand.row(
            "QuadSource",
            Tq=stand.ref("TkNumericIntegration", rows["t1"]),
            Tr=stand.ref("TkNumericIntegration", rows["t0"]),
        )
        _integral(stand, rows)
        _integral(stand, rows, source=stand.ref("QuadSource", other))
        self.assert_two_lines(stand.as_inventory(), "QuadSourceIntegral")

    def test_two_parents_that_differ_only_in_their_tags_get_two_labels(self):
        stand = _Stand(self.base)
        for tags in (("Run_fixture",), ("Run_fixture", "grid-A")):
            stand.row("TkNumericIntegration", tags=tags)
        numerics = stand.rows["TkNumericIntegration"]
        stand.row(
            "TkWKBIntegration", numeric=stand.ref("TkNumericIntegration", numerics[0])
        )
        stand.row(
            "TkWKBIntegration", numeric=stand.ref("TkNumericIntegration", numerics[1])
        )
        self.assert_two_lines(stand.as_inventory(), "TkWKBIntegration")


# --- 2. every label is defined, once ----------------------------------------------------------


def definitions(text: str) -> Tuple[Dict[str, List[str]], Dict[int, List[str]]]:
    """(``Class#n`` -> the lines that define it, ``n`` of ``S<n>`` -> the lines that define it)."""
    rows: Dict[str, List[str]] = {}
    sets: Dict[int, List[str]] = {}
    for name, section in sections(text).items():
        for line in section.lines:
            body = line[len(SHOWN) :] if line.startswith((SHOWN, LISTED)) else None
            if body is not None:
                own = OWN_LABEL.search(body)
                if own is not None:
                    rows.setdefault(f"{name}#{own.group(1)}", []).append(line)
            found = SET_LINE.match(line)
            if found is not None:
                sets.setdefault(int(found.group(1)), []).append(line)
    return rows, sets


def references(text: str) -> Tuple[set, set]:
    """The ``Class#n`` and the ``n`` of ``S<n>`` that the report names, outside a label's own slot."""
    names = set()
    numbers = set()
    for line in text.split("\n"):
        names.update(f"{c}#{n}" for c, n in ROW_LABEL.findall(line))
        numbers.update(int(n) for n in SET_LABEL.findall(line))
        found = SET_LINE.match(line)
        if found is not None:
            numbers.add(int(found.group(1)))
    return names, numbers


def body(line: str) -> str:
    """A defining line without its prefix and without the tags a listed row carries."""
    line = line[len(SHOWN) :] if line.startswith((SHOWN, LISTED)) else line
    return re.sub(r"  tags: .*$", "", line)


class TestEveryLabelIsDefinedOnce(_Fixture):
    def stores(self):
        """(a name, the inventory) for the fixture and for stand-ins with several records."""
        yield "fixture", self.base
        stand = _Stand(self.base, z_values=(5.0, 50.0, 500.0))
        n = [stand.row("GkNumericIntegration", rtol=stand.tol[i]) for i in (0, 1)]
        g = stand.row("GkWKBIntegration")
        a = stand.parent_set([(0, n[0], None), (3, None, g), (4, n[1], None)])
        b = stand.parent_set([(0, n[1], None), (3, None, g), (4, n[0], None)])
        rows = _chain(stand, a, b)
        _integral(stand, rows)
        _integral(stand, rows, data=stand.ref("GkSourcePolicyData", rows["data_b"]))
        _integral(
            stand,
            rows,
            Tq_numeric=stand.ref("TkNumericIntegration", rows["t1"]),
        )
        yield "stand-in", stand.as_inventory()

    def settings(self):
        yield dict(verbose=True)
        yield dict(verbose=False)
        yield dict(verbose=False, shown=1)
        yield dict(verbose=False, shown=2)

    def test_every_label_has_exactly_one_defining_line(self):
        for name, inventory in self.stores():
            for setting in self.settings():
                with self.subTest(store=name, **setting):
                    text = report(inventory, **setting)
                    rows, sets = definitions(text)
                    named_rows, named_sets = references(text)
                    for label in named_rows:
                        self.assertEqual(len(rows.get(label, [])), 1, label)
                    for number in named_sets:
                        self.assertEqual(len(sets.get(number, [])), 1, f"S{number}")
                    self.assertTrue(all(len(v) == 1 for v in rows.values()))
                    self.assertTrue(all(len(v) == 1 for v in sets.values()))
                    self.assertIsNone(HEX64.search(text))

    def test_a_defining_line_is_the_row_or_set_it_names(self):
        # the report that lists everything, in the same float format, says what each label names;
        # a report that lists less must say the same of every label it prints
        for name, inventory in self.stores():
            complete = report(inventory, verbose=False, shown=10**6)
            full_rows_, full_sets = definitions(complete)
            for setting in self.settings():
                if setting["verbose"]:
                    continue  # its floats are repr; test_labels_are_the_ordinals covers it
                with self.subTest(store=name, **setting):
                    text = report(inventory, **setting)
                    rows, sets = definitions(text)
                    self.assertTrue(rows)
                    for label, lines in rows.items():
                        self.assertEqual(body(lines[0]), body(full_rows_[label][0]))
                    for number, lines in sets.items():
                        self.assertEqual(lines[0], full_sets[number][0])

    def test_labels_are_the_ordinals_of_the_listing(self):
        # with verbose, every record is listed: a class's labels run 1, 2, 3, ... down its section,
        # tag set by tag set, and the sets run S1, S2, ... in the physical order
        for name, inventory in self.stores():
            with self.subTest(store=name):
                text = report(inventory, verbose=True)
                labelled = set()
                for cls, section in sections(text).items():
                    labels = [
                        int(OWN_LABEL.search(line).group(1))
                        for _, line in section.records
                        if OWN_LABEL.search(line)
                    ]
                    if labels:
                        labelled.add(cls)
                        self.assertEqual(
                            labels, list(range(1, len(section.records) + 1))
                        )
                self.assertEqual(
                    labelled,
                    {
                        "TkNumericIntegration",
                        "TkWKBIntegration",
                        "GkNumericIntegration",
                        "GkWKBIntegration",
                        "GkSource",
                        "GkSourcePolicyData",
                        "QuadSource",
                    },
                )
                numbers = [
                    int(m.group(1))
                    for line in sections(text)["GkSource"].lines
                    for m in [SET_LINE.match(line)]
                    if m
                ]
                self.assertEqual(numbers, list(range(1, len(numbers) + 1)))

    def test_a_reference_names_the_row_it_points_to(self):
        # a label that is defined once may still be the wrong row's. Each Tk row of the chain has
        # a leaf of its own, so a reference is checked against the line that defines its label
        stand = _Stand(self.base)
        a = stand.parent_set([(0, None, None)])
        rows = _chain(stand, a)
        _integral(stand, rows)
        _integral(stand, rows, Tq_numeric=stand.ref("TkNumericIntegration", rows["t1"]))
        inventory = stand.as_inventory()
        for setting in self.settings():
            with self.subTest(**setting):
                text = report(inventory, **setting)
                defined, _ = definitions(text)

                def defining(label: str) -> str:
                    return body(defined[label][0])

                integral = sections(text)["QuadSourceIntegral"]
                common = dict(fields_of(integral.common))
                self.assertIn(
                    f"break_point_kind={KIND[1]}", defining(common["Tr_numeric"])
                )
                self.assertIn(f"rho_gauss_order={RHO[0]}", defining(common["Tq_WKB"]))
                self.assertIn(f"rho_gauss_order={RHO[1]}", defining(common["Tr_WKB"]))
                # the two integrals name the two numeric rows, one each
                kinds = sorted(
                    re.search(r"break_point_kind=(kind-\d)", defining(value)).group(1)
                    for _, line in integral.records
                    for field, value in fields_of(line)
                    if field == "Tq_numeric"
                )
                if "shown" in setting:
                    self.assertTrue(set(kinds) <= set(KIND.values()) and kinds)
                else:
                    self.assertEqual(kinds, sorted(KIND.values()))
                # a WKB row names the numeric row of its own chain
                wkb = sections(text)["TkWKBIntegration"]
                self.assertEqual(
                    len(wkb.records), 1 if setting.get("shown") == 1 else 2
                )
                for _, line in wkb.records:
                    value = dict(fields_of(line))
                    rho = int(value["rho_gauss_order"])
                    kind = KIND[0] if rho == RHO[0] else KIND[1]
                    self.assertIn(
                        f"break_point_kind={kind}", defining(value["numeric"])
                    )

    def test_without_verbose_the_rows_referred_to_but_not_shown_are_listed(self):
        for name, inventory in self.stores():
            with self.subTest(store=name):
                text = report(inventory, verbose=False, shown=1)
                listed = [line for line in text.split("\n") if line.startswith(LISTED)]
                self.assertTrue(listed)
                rows, _ = definitions(text)
                for line in listed:
                    self.assertIsNotNone(OWN_LABEL.search(line))
                # a listed row is not also a record line
                for lines in rows.values():
                    self.assertEqual(len(lines), 1)

    def test_every_reference_to_a_sharded_class_is_a_label(self):
        sharded = {n for n, c in self.base.classes.items() if not c.replicated}
        for name, inventory in self.stores():
            for verbose in (False, True):
                with self.subTest(store=name, verbose=verbose):
                    text = report(inventory, verbose=verbose)
                    for cls, section in sections(text).items():
                        info = inventory[cls]
                        lines = [line for _, line in section.records]
                        if section.common is not None:
                            lines.append(section.common)
                        for line in lines:
                            for field, value in fields_of(line):
                                if field in info.parent_sets:
                                    self.assertRegex(value, r"^S\d+$")
                                elif info.parents.get(field) in sharded:
                                    self.assertRegex(
                                        value,
                                        rf"^{info.parents[field]}#\d+$|^<unresolved ",
                                    )


# --- 3. labels are stable ---------------------------------------------------------------------


class TestLabelsAreStable(_Fixture):
    def test_the_same_store_reported_twice(self):
        for verbose in (False, True):
            self.assertEqual(report(self.base, verbose), report(self.base, verbose))

    def test_the_same_store_under_other_serials(self):
        replicated, sharded, keys = relabel_serials(*full_rows())
        again = self.read(
            self.root / "relabelled",
            dict(replicated=replicated, sharded=sharded, shard_keys=keys),
        )
        for verbose in (False, True):
            for shown in (None, 1):
                with self.subTest(verbose=verbose, shown=shown):
                    a = report(self.base, verbose, shown)
                    b = report(again, verbose, shown)
                    self.assertEqual(a, b)
                    self.assertTrue(definitions(a)[0])

    def test_the_same_inventory_with_every_class_read_in_another_order(self):
        # a label is an ordinal in the physical order, not a place in the order of reading, which
        # follows the serials
        flipped = dataclasses.replace(
            self.base,
            classes={
                n: dataclasses.replace(c, records=tuple(reversed(c.records)))
                for n, c in self.base.classes.items()
            },
        )
        for verbose in (False, True):
            for shown in (None, 1):
                with self.subTest(verbose=verbose, shown=shown):
                    self.assertEqual(
                        report(self.base, verbose, shown),
                        report(flipped, verbose, shown),
                    )


# --- 4. the size is bounded -------------------------------------------------------------------

# the most characters a parent set's line may grow by for each member. Measured, a member prints
# as ``{z_source=<z>, numeric=none, wkb=GkWKBIntegration#<n>}`` and a separator: about sixty. The
# bound is not the measurement, only a number far under what the sets cost before 09d (a member's
# whole parent chain, about 200)
MAX_CHARACTERS_PER_MEMBER = 100

# the record lines of every class that carries a derived parent
CARRIERS = (
    "TkWKBIntegration",
    "QuadSource",
    "GkSource",
    "GkSourcePolicyData",
    "QuadSourceIntegral",
)


def production_stand(base: StoreInventory, members: int) -> StoreInventory:
    """Integrals that reference a ``GkSource`` of ``members`` members, one per ``z_source``."""
    stand = _Stand(base, z_values=[1.0 + i for i in range(members)])
    n = stand.row("GkNumericIntegration")
    g = stand.row("GkWKBIntegration")
    # the first two z are the fixture's own; the others are the stand-in's
    z = range(len(stand.z))
    a = stand.parent_set([(i, n if i % 2 else None, None if i % 2 else g) for i in z])
    b = stand.parent_set([(i, None if i % 2 else n, g if i % 2 else None) for i in z])
    rows = _chain(stand, a, b)
    _integral(stand, rows)
    _integral(stand, rows, data=stand.ref("GkSourcePolicyData", rows["data_b"]))
    return stand.as_inventory()


class TestTheSizeIsBounded(_Fixture):
    SIZES = (100, 400, 1700)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.reports = {
            members: {
                verbose: format_inventory_report(
                    production_stand(cls.base, members), DB_NAME, verbose=verbose
                )
                for verbose in (False, True)
            }
            for members in cls.SIZES
        }

    def longest(self, members: int, name: str, verbose: bool = True) -> int:
        return max(
            len(line) for line in record_lines(self.reports[members][verbose], name)
        )

    def test_a_record_line_does_not_grow_with_the_members_of_a_set(self):
        for name in CARRIERS:
            with self.subTest(cls=name):
                lengths = {m: self.longest(m, name) for m in self.SIZES}
                self.assertEqual(len(set(lengths.values())), 1, lengths)

    def test_a_sets_line_grows_by_a_bounded_number_of_characters_a_member(self):
        def set_lines(members: int) -> List[str]:
            return [
                line
                for line in sections(self.reports[members][True])["GkSource"].lines
                if SET_LINE.match(line)
            ]

        small, large = self.SIZES[0], self.SIZES[-1]
        # each set has one member for each of the fixture's three redshifts and the stand-in's own
        extra = large - small
        a, b = set_lines(small), set_lines(large)
        self.assertEqual((len(a), len(b)), (2, 2))
        for before, after in zip(a, b):
            per_member = (len(after) - len(before)) / extra
            self.assertGreater(per_member, 0)
            self.assertLessEqual(per_member, MAX_CHARACTERS_PER_MEMBER)

    def test_the_whole_report_is_bounded_by_its_sets(self):
        # what the members add to the report is the two sets' lines, and the replicated
        # redshift rows that the members name (the stand-in's own z), which print as they always did
        for verbose in (False, True):
            rest = {}
            for members in self.SIZES:
                text = self.reports[members][verbose]
                found = sections(text)
                sets = [line for line in text.split("\n") if SET_LINE.match(line)]
                self.assertEqual(len(sets), 2)
                rest[members] = (
                    len(text)
                    - sum(len(line) + 1 for line in sets)
                    - sum(len(line) + 1 for line in found["redshift"].lines)
                )
            with self.subTest(verbose=verbose):
                # the redshift header's count is the one number in it that changes
                self.assertLessEqual(max(rest.values()) - min(rest.values()), 16, rest)


# --- 5. nothing else changed ------------------------------------------------------------------


class TestNothingElseChanged(_Fixture):
    def test_a_replicated_class_and_an_unreferenced_class_print_no_label(self):
        # OneLoopIntegral has no sharded parent and nothing names it
        for verbose in (False, True):
            text = report(self.base, verbose)
            found = sections(text)
            for name, info in self.base.classes.items():
                if info.replicated or name == "OneLoopIntegral":
                    with self.subTest(cls=name, verbose=verbose):
                        section = found[name]
                        self.assertNotRegex(section.text, r"#\d")
                        self.assertNotRegex(section.text, r"parent set")
                        self.assertFalse(
                            [line for line in section.lines if line.startswith(LISTED)]
                        )

    def test_a_class_that_nothing_names_prints_no_label_of_its_own(self):
        for verbose in (False, True):
            section = sections(report(self.base, verbose))["QuadSourceIntegral"]
            for _, line in section.records:
                self.assertIsNone(OWN_LABEL.search(line))


if __name__ == "__main__":
    unittest.main()
