"""The store fingerprint (store-fingerprint prompt 04): `RunRegistry.stores.fingerprint_of`,
`compare_fingerprints`, the sidecar's `fingerprint` field, `fingerprint_store`,
`python -m RunRegistry store fingerprint`, and `Run.finish(..., fingerprint=True)`.

1. the fingerprint is physical: other serials, other shards, the same fingerprint;
2. a difference is localised to the class and tag set that changed, and no other;
3. what is not content (timestamps, compute-target labels, solver serials) does not matter;
4. the format is pinned by a golden fingerprint of `build_full_store()`;
5. a listing's lines hash to the digests it lists;
6. fingerprints of different formats are not compared;
7. the sidecar field: its shape, and copy and move carry it verbatim;
8. `store fingerprint` without `--write` writes nothing and initialises no Ray;
9. `store fingerprint --write` changes only the `fingerprint` field, and is refused on a sidecar
   that is not a problem-free registry sidecar;
10. a `running` run naming the store makes it refuse, except the run passed as `taken_by`;
11. `Run.finish(..., fingerprint=True)` records a fingerprint, or why not, and never changes how
    the run ended;
12. the drivers pass `fingerprint=True` to every `run.finish(` outside a `terminal` handler, and
    to none inside one.

Stores are built with `Datastore.tests.real_store_fixtures` in temporary directories, runs in a
temporary runs root. No Ray is initialised and nothing under `var/` is opened.
"""

import ast
import hashlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import RunRegistry
from RunRegistry import stores
from RunRegistry.stores import (
    FINGERPRINT_FORMAT,
    compare_fingerprints,
    fingerprint_of,
    fingerprint_store,
)
from RunRegistry.tests.store_fixtures import (
    StoreTestCase,
    load,
    sweep_shaped,
    write_sidecar_json,
)
from RunRegistry.tests.test_store_command_line import CommandLineTestCase

from Datastore.store_inventory import COMPUTE_TARGETS, read_inventory
from Datastore.tests.real_store_fixtures import (
    FIXED_TIMESTAMP,
    _schema_tables,
    build_full_store,
    file_state,
    full_rows,
    relabel_serials,
)

REPO_ROOT = Path(RunRegistry.REPO_ROOT)
GOLDEN = Path(__file__).resolve().parent / "data" / "full_store_fingerprint.json"
DRIVERS = (
    REPO_ROOT / "docs" / "gktk-remedial" / "scoped_pipeline_run.py",
    REPO_ROOT / "docs" / "handover" / "quadsource_atol_sweep.py",
)
QSI = "QuadSourceIntegral"
TKN = "TkNumericIntegration"
SOLVER_COLUMNS = ("solver_serial", "phase_solver_serial", "friction_solver_serial")


def without_taken(fingerprint) -> dict:
    out = dict(fingerprint)
    out.pop("taken", None)
    return out


def entries(fingerprint) -> dict:
    """(class, None) -> (count, digest) for each class, and (class, tags) -> (count, digest) for
    each of its tag sets."""
    out = {}
    for name, entry in fingerprint["classes"].items():
        out[(name, None)] = (entry["count"], entry["digest"])
        for tag_set in entry["tag_sets"]:
            out[(name, tuple(tag_set["tags"]))] = (tag_set["count"], tag_set["digest"])
    return out


def changed_entries(before, after) -> set:
    a, b = entries(before), entries(after)
    return {key for key in set(a) | set(b) if a.get(key) != b.get(key)}


def sha256_lines(lines) -> str:
    return hashlib.sha256("".join(line + "\n" for line in lines).encode()).hexdigest()


def parse_listing(text: str):
    """{"digest", "classes": {name: {"count", "digest", "lines", "tag_sets": {tags: {"count",
    "digest", "lines"}}}}} from a listing's text."""
    out = {"classes": {}}
    cls = tag_set = None
    for line in text.splitlines():
        if line.startswith("# fingerprint_format "):
            words = line.split()
            out["format"], out["digest"] = int(words[2]), words[4]
        elif line.startswith("# class "):
            _, _, name, _, count, _, digest = line.split()
            cls = {"count": int(count), "digest": digest, "lines": [], "tag_sets": {}}
            out["classes"][name] = cls
        elif line.startswith("# tags "):
            body, _, rest = line[len("# tags ") :].rpartition(" count ")
            count, _, digest = rest.split()
            tag_set = {"count": int(count), "digest": digest, "lines": []}
            cls["tag_sets"][tuple(json.loads(body))] = tag_set
        else:
            cls["lines"].append(line)
            tag_set["lines"].append(line)
    return out


class _FullStores(unittest.TestCase):
    """A temporary directory, and full stores built in it, each in its own directory."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.top = Path(cls._tmp.name).resolve()
        cls._count = 0
        cls.baseline_inventory = read_inventory(cls.store().primary)
        cls.baseline = fingerprint_of(cls.baseline_inventory)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def store(cls, **kwargs):
        cls._count += 1
        directory = cls.top / f"store-{cls._count}"
        directory.mkdir()
        return build_full_store(directory, **kwargs)

    @classmethod
    def fingerprint(cls, **kwargs):
        return fingerprint_of(read_inventory(cls.store(**kwargs).primary))


class TestPhysical(_FullStores):
    """Test 1: the relabelled store, with its rows moved between shards (prompt 02's test 1)."""

    def test_other_serials_and_other_shards(self):
        replicated, sharded, keys = relabel_serials(*full_rows(), offset=37)
        moved = {0: sharded[1], 1: sharded[0]}
        moved_keys = {k: 1 - s for k, s in keys.items()}
        other = self.fingerprint(
            replicated=replicated, sharded=moved, shard_keys=moved_keys
        )
        self.assertEqual(without_taken(other), without_taken(self.baseline))
        self.assertEqual(compare_fingerprints(self.baseline, other), [])

    def test_sharded_rows_on_a_third_shard(self):
        replicated, sharded, keys = relabel_serials(*full_rows(), offset=100)
        moved = {0: {}, 1: sharded[1], 2: sharded[0]}
        moved_keys = {k: {0: 2, 1: 1}[s] for k, s in keys.items()}
        other = self.fingerprint(
            replicated=replicated, sharded=moved, shard_keys=moved_keys, shards=3
        )
        self.assertEqual(without_taken(other), without_taken(self.baseline))


class TestLocalised(_FullStores):
    """Test 2: each change moves exactly the digests it names, and the comparison names exactly
    those classes and tag sets."""

    def assertLocalised(self, cls, after_inventory, expected_counts=None):
        """Only ``cls`` has records that changed. The changed entries are ``cls`` itself and the
        tag set of every record that changed; the overall digest changed; problems did not.
        """
        after = fingerprint_of(after_inventory)
        for name in self.baseline_inventory.classes:
            if name != cls:
                self.assertEqual(
                    after_inventory[name].records,
                    self.baseline_inventory[name].records,
                    name,
                )
        gone = set(self.baseline_inventory[cls].records) - set(
            after_inventory[cls].records
        )
        new = set(after_inventory[cls].records) - set(
            self.baseline_inventory[cls].records
        )
        self.assertTrue(gone or new, "the change changed nothing")
        tag_sets = {r.tags for r in gone | new}
        self.assertEqual(
            changed_entries(self.baseline, after),
            {(cls, None)} | {(cls, tags) for tags in tag_sets},
        )
        self.assertNotEqual(after["digest"], self.baseline["digest"])
        self.assertEqual(after["problems"], self.baseline["problems"])

        comparison = compare_fingerprints(self.baseline, after)
        self.assertEqual(
            {(e["kind"], e["class"], tuple(e["tags"])) for e in comparison},
            {("tag_set", cls, tags) for tags in tag_sets},
        )
        self.assertEqual(len(comparison), len(tag_sets))
        if expected_counts is not None:
            self.assertEqual(
                {tuple(e["tags"]): (e["recorded"], e["current"]) for e in comparison},
                expected_counts,
            )
        return after

    def test_deleting_one_quadsourceintegral_row(self):
        inventory = read_inventory(
            self.store(
                extra_sql={
                    1: [
                        f"DELETE FROM {QSI} WHERE serial = 3",
                        f"DELETE FROM {QSI}_tags WHERE parent_serial = 3",
                    ]
                }
            ).primary
        )
        after = self.assertLocalised(
            QSI, inventory, expected_counts={("Run_fixture",): (2, 1)}
        )
        self.assertEqual(after["classes"][QSI]["count"], 2)
        self.assertEqual(self.baseline["classes"][QSI]["count"], 3)

    def test_deleting_the_row_alone_is_also_a_problem_entry(self):
        """The discriminator of prompt 02: the row goes and its tag rows stay, so the one
        content difference comes with one `orphan-tag` problem entry of its own."""
        after = self.fingerprint(extra_sql={1: [f"DELETE FROM {QSI} WHERE serial = 3"]})
        comparison = compare_fingerprints(self.baseline, after)
        self.assertEqual(
            [(e["kind"], e["class"]) for e in comparison],
            [("tag_set", QSI), ("problems", QSI)],
        )
        self.assertEqual((comparison[0]["recorded"], comparison[0]["current"]), (2, 1))
        self.assertEqual(after["problems"], {QSI: {"orphan-tag": 1}})
        self.assertIn("orphan-tag problems: 0 recorded, 1 now", comparison[1]["text"])

    def test_adding_a_tag_to_one_record(self):
        inventory = read_inventory(
            self.store(
                extra_sql={
                    1: [
                        f"INSERT INTO {QSI}_tags (timestamp, parent_serial, tag_serial) "
                        f"VALUES ('2026-09-24 12:00:00.000000', 3, 3)"
                    ]
                }
            ).primary
        )
        after = self.assertLocalised(
            QSI,
            inventory,
            expected_counts={
                ("Run_fixture",): (2, 1),
                ("Run_fixture", "unused-tag"): (None, 1),
            },
        )
        self.assertEqual(after["classes"][QSI]["count"], 3)

    def test_deleting_one_value_row(self):
        inventory = read_inventory(
            self.store(
                extra_sql={1: ["DELETE FROM TkNumericValue WHERE serial = 24"]}
            ).primary
        )
        after = self.assertLocalised(TKN, inventory)
        self.assertEqual(
            after["classes"][TKN]["count"], self.baseline["classes"][TKN]["count"]
        )

    def test_flipping_one_validated_flag(self):
        inventory = read_inventory(
            self.store(
                extra_sql={0: [f"UPDATE {TKN} SET validated = 1 WHERE serial = 3"]}
            ).primary
        )
        self.assertLocalised(TKN, inventory)


class TestNotContent(_FullStores):
    """Test 3: every timestamp, every compute-target label and every solver serial changed."""

    def test_timestamps_labels_and_solver_serials(self):
        replicated, sharded, keys = full_rows()
        tables = _schema_tables()
        stamp = datetime(2031, 2, 3, 4, 5, 6)
        labelled = set(COMPUTE_TARGETS) | {"BackgroundModel"}
        changed = {"timestamp": 0, "label": 0, "solver": 0}
        for rows in [replicated] + list(sharded.values()):
            for table, table_rows in rows.items():
                columns = tables[table].c
                for row in table_rows:
                    if "timestamp" in columns:
                        row["timestamp"] = stamp
                        changed["timestamp"] += 1
                    if table in labelled and "label" in columns:
                        row["label"] = f"relabelled-{table}-{row['serial']}"
                        changed["label"] += 1
                    for column in SOLVER_COLUMNS:
                        if column in columns:
                            row[column] = 2
                            changed["solver"] += 1
        self.assertTrue(all(n > 0 for n in changed.values()), changed)
        inventory = read_inventory(
            self.store(replicated=replicated, sharded=sharded, shard_keys=keys).primary
        )
        self.assertEqual(inventory[QSI].earliest_timestamp, stamp)
        after = fingerprint_of(inventory)
        self.assertEqual(without_taken(after), without_taken(self.baseline))
        self.assertEqual(compare_fingerprints(self.baseline, after), [])

    def test_no_timestamp_is_in_any_record(self):
        for text in (str(FIXED_TIMESTAMP), FIXED_TIMESTAMP.isoformat(), "2026-09-24"):
            for name, cls in self.baseline_inventory.classes.items():
                for record in cls.records:
                    self.assertNotIn(text, record.canonical_json(), name)


class TestGolden(_FullStores):
    """Test 4: the fingerprint of `build_full_store()` is the committed golden file."""

    def test_the_golden_fingerprint(self):
        golden = json.loads(GOLDEN.read_text())
        self.assertEqual(golden["fingerprint_format"], FINGERPRINT_FORMAT)
        self.assertIsNone(golden["taken"])
        self.assertEqual(without_taken(self.baseline), without_taken(golden))
        # and it is deterministic: a second build gives the same
        self.assertEqual(without_taken(self.fingerprint()), without_taken(golden))

    def test_the_shape(self):
        fp = self.baseline
        self.assertEqual(
            set(fp), {"fingerprint_format", "classes", "digest", "problems", "taken"}
        )
        self.assertEqual(list(fp["classes"]), list(self.baseline_inventory.classes))
        self.assertEqual(fp["problems"], {})
        for name, entry in fp["classes"].items():
            self.assertEqual(entry["count"], self.baseline_inventory[name].count)
            self.assertEqual(sum(t["count"] for t in entry["tag_sets"]), entry["count"])
            if not self.baseline_inventory[name].tagged:
                self.assertEqual([t["tags"] for t in entry["tag_sets"]], [[]])
        self.assertEqual(
            [t["tags"] for t in fp["classes"][QSI]["tag_sets"]],
            [["Run_fixture"], ["Run_fixture", "grid-A"]],
        )
        self.assertEqual(stores._fingerprint_problems(dict(fp, taken={})), [])

    def test_taken_is_outside_every_digest(self):
        taken = {"when": "now", "git_head": "x", "git_dirty": True, "run_id": "r"}
        with_taken = fingerprint_of(self.baseline_inventory, taken)
        self.assertEqual(with_taken["taken"], taken)
        self.assertEqual(without_taken(with_taken), without_taken(self.baseline))
        self.assertEqual(compare_fingerprints(self.baseline, with_taken), [])


class TestListing(_FullStores):
    """Test 5: the lines a listing writes for each class and tag set hash to its digests."""

    def test_the_listing_hashes_to_the_fingerprint(self):
        path = self.top / "listing.txt"
        stores.write_listing(
            self.baseline_inventory, self.baseline, path, self.store().primary
        )
        listing = parse_listing(path.read_text())
        self.assertEqual(listing["digest"], self.baseline["digest"])
        self.assertEqual(listing["format"], FINGERPRINT_FORMAT)
        self.assertEqual(list(listing["classes"]), list(self.baseline["classes"]))
        for name, entry in self.baseline["classes"].items():
            listed = listing["classes"][name]
            with self.subTest(cls=name):
                self.assertEqual(listed["count"], entry["count"])
                self.assertEqual(len(listed["lines"]), entry["count"])
                # the class digest is over its records in canonical order: its lines, sorted
                self.assertEqual(sha256_lines(sorted(listed["lines"])), entry["digest"])
                self.assertEqual(
                    set(listed["tag_sets"]),
                    {tuple(t["tags"]) for t in entry["tag_sets"]},
                )
                for tag_set in entry["tag_sets"]:
                    lines = listed["tag_sets"][tuple(tag_set["tags"])]["lines"]
                    self.assertEqual(len(lines), tag_set["count"])
                    self.assertEqual(sha256_lines(lines), tag_set["digest"])

    def test_a_listing_is_never_written_over_a_file_or_beside_the_store(self):
        primary = self.store().primary
        taken = self.top / "taken.txt"
        taken.write_text("mine\n")
        for path in (
            taken,
            primary.parent / "listing.txt",
            primary.parent / "sub" / "listing.txt",
            self.top / "no-such-directory" / "listing.txt",
        ):
            if path.parent.name == "sub":
                path.parent.mkdir()
            with self.subTest(path=str(path)):
                with self.assertRaises(RuntimeError):
                    stores.write_listing(
                        self.baseline_inventory, self.baseline, path, primary
                    )
        self.assertEqual(taken.read_text(), "mine\n")
        self.assertFalse((primary.parent / "listing.txt").exists())
        self.assertFalse((primary.parent / "sub" / "listing.txt").exists())


class TestFormats(unittest.TestCase):
    """Test 6: two fingerprints of different formats are not compared."""

    def fingerprint(self, fmt):
        return {
            "fingerprint_format": fmt,
            "classes": {"A": {"count": fmt, "digest": str(fmt) * 64, "tag_sets": []}},
            "digest": str(fmt) * 64,
            "problems": {"A": {"duplicate": fmt}},
            "taken": {"when": str(fmt)},
        }

    def test_different_formats(self):
        comparison = compare_fingerprints(self.fingerprint(1), self.fingerprint(2))
        self.assertEqual(len(comparison), 1)
        self.assertEqual(comparison[0]["kind"], "format")
        self.assertEqual((comparison[0]["recorded"], comparison[0]["current"]), (1, 2))
        self.assertIn("Recompute both", comparison[0]["text"])

    def test_the_same_format_is_compared(self):
        self.assertEqual(
            compare_fingerprints(self.fingerprint(1), self.fingerprint(1)), []
        )


class TestSidecarField(StoreTestCase):
    """Test 7: the `fingerprint` field's shape, and copy and move carry it verbatim."""

    FINGERPRINT = {
        "fingerprint_format": 1,
        "classes": {"version": {"count": 1, "digest": "a" * 64, "tag_sets": []}},
        "digest": "b" * 64,
        "problems": {},
        "taken": {
            "when": "2026-09-25T10:00:00+01:00",
            "git_head": "c" * 40,
            "git_dirty": False,
            "run_id": "store-fingerprint-04-test-20260925T100000",
        },
    }

    def with_fingerprint(self, primary, value):
        fields = load(stores.sidecar_path(primary))
        fields["fingerprint"] = value
        write_sidecar_json(primary, fields)
        return fields

    def test_a_registry_sidecar_without_one_is_problem_free(self):
        primary = self.a_registry_store()
        reading = stores.read_sidecar(primary)
        self.assertTrue(reading.ok, reading.problems)
        self.assertNotIn("fingerprint", reading.fields)
        self.assertIn("fingerprint", stores.KNOWN_FIELDS)

    def test_a_well_formed_fingerprint_is_not_a_problem(self):
        primary = self.a_registry_store()
        self.with_fingerprint(primary, self.FINGERPRINT)
        reading = stores.read_sidecar(primary)
        self.assertTrue(reading.ok, reading.problems)
        self.assertNotIn("fingerprint", reading.unknown_fields)

    def test_each_malformed_fingerprint_is_a_problem(self):
        good = self.FINGERPRINT
        variants = {
            "null": None,
            "a list": [good],
            "a string": "b" * 64,
            "no format": {k: v for k, v in good.items() if k != "fingerprint_format"},
            "a boolean format": dict(good, fingerprint_format=True),
            "a string format": dict(good, fingerprint_format="1"),
            "no classes": {k: v for k, v in good.items() if k != "classes"},
            "classes a list": dict(good, classes=[]),
            "no digest": {k: v for k, v in good.items() if k != "digest"},
            "a short digest": dict(good, digest="b" * 63),
            "an uppercase digest": dict(good, digest="B" * 64),
            "no taken": {k: v for k, v in good.items() if k != "taken"},
            "taken null": dict(good, taken=None),
            "taken a string": dict(good, taken="now"),
        }
        for index, (label, value) in enumerate(variants.items()):
            with self.subTest(variant=label):
                primary = self.a_registry_store(f"M{index}")
                self.with_fingerprint(primary, value)
                reading = stores.read_sidecar(primary)
                self.assertEqual(reading.kind, "registry")
                self.assertFalse(reading.ok)
                self.assertTrue(
                    any("fingerprint" in p for p in reading.problems), reading.problems
                )

    def test_copy_and_move_carry_it_verbatim(self):
        src = self.a_registry_store("S")
        self.with_fingerprint(src, self.FINGERPRINT)
        copied = self.top / "C" / "copied.sqlite"
        fields = stores.copy_store(src, copied, "a copy", runs_root=self.root)
        self.assertEqual(fields["fingerprint"], self.FINGERPRINT)
        self.assertEqual(
            load(stores.sidecar_path(copied))["fingerprint"], self.FINGERPRINT
        )
        self.assertTrue(stores.read_sidecar(copied).ok)
        # the source is unchanged
        self.assertEqual(
            load(stores.sidecar_path(src))["fingerprint"], self.FINGERPRINT
        )

        moved = self.top / "D" / "moved.sqlite"
        fields = stores.move_store(copied, moved, runs_root=self.root)
        self.assertEqual(fields["fingerprint"], self.FINGERPRINT)
        self.assertEqual(
            load(stores.sidecar_path(moved))["fingerprint"], self.FINGERPRINT
        )
        self.assertTrue(stores.read_sidecar(moved).ok)


class _FullStoreCase(CommandLineTestCase):
    """A full store with a registry sidecar in this test's temporary directory."""

    def a_full_store(self, directory="S", sidecar="registry"):
        (self.top / directory).mkdir()
        primary = build_full_store(self.top / directory, stem="store").primary
        if sidecar == "registry":
            stores.create_sidecar(primary, "a full store in a temporary directory")
        elif sidecar == "legacy":
            write_sidecar_json(primary, sweep_shaped(self.top / "elsewhere"))
        elif sidecar == "problem":
            stores.create_sidecar(primary, "a full store with a sidecar problem")
            fields = load(stores.sidecar_path(primary))
            fields["purpose"] = ""
            write_sidecar_json(primary, fields)
        return primary

    def expected(self, primary):
        return fingerprint_of(read_inventory(primary))

    def store_state(self, primary):
        return file_state(primary.parent)


class TestFingerprintCommand(_FullStoreCase):
    """Tests 8 and 9: `store fingerprint`, read-only and with `--write`."""

    def test_read_only_then_write_then_matches(self):
        primary = self.a_full_store()
        runs = ["--runs-root", self.root]
        fields = load(stores.sidecar_path(primary))
        fields["note"] = {"kept": ["verbatim", 1, None], "path": "var/elsewhere"}
        write_sidecar_json(primary, fields)
        expected = self.expected(primary)

        # 8: read-only, with a listing written away from the store
        before = self.store_state(primary)
        listing = self.top / "listing.txt"
        report, out, err = self.command(
            "store", "fingerprint", primary, "--listing", listing, *runs
        )
        self.assertEqual(report["code"], 0, err)
        self.assertEqual(self.store_state(primary), before)
        self.assertIn(f"digest:   {expected['digest']}", out)
        self.assertIn("recorded: none recorded (the sidecar is registry)", out)
        self.assertIn("problems: none", out)
        self.assertEqual(
            parse_listing(listing.read_text())["digest"], expected["digest"]
        )

        # 9: --write changes only the fingerprint field
        sidecar_before = load(stores.sidecar_path(primary))
        report, out, err = self.command(
            "store", "fingerprint", primary, "--write", *runs
        )
        self.assertEqual(report["code"], 0, err)
        self.assertIn("it replaced none", out)
        after = load(stores.sidecar_path(primary))
        self.assertEqual(
            {k: v for k, v in after.items() if k != "fingerprint"}, sidecar_before
        )
        self.assertEqual(without_taken(after["fingerprint"]), without_taken(expected))
        self.assertEqual(set(after["fingerprint"]["taken"]), set(stores.TAKEN_KEYS))
        self.assertIsNone(after["fingerprint"]["taken"]["run_id"])
        self.assertTrue(stores.read_sidecar(primary).ok)
        self.assertEqual(
            [p for p in self.store_state(primary)[0]], before[0]
        )  # no file added beside the store

        # and read-only again: it matches, naming when and by whom
        report, out, err = self.command("store", "fingerprint", primary, *runs)
        self.assertEqual(report["code"], 0, err)
        self.assertIn(
            f"recorded: matches the fingerprint taken {after['fingerprint']['taken']['when']} "
            f"by a person",
            out,
        )

        # store show prints it, and still loads neither ray nor sqlalchemy
        report, out, _ = self.command("store", "show", primary, *runs)
        self.assertEqual(report["code"], 0)
        self.assertIn(expected["digest"], out)
        self.assertFalse(report["ray"] or report["sqlalchemy"])

    def test_a_difference_exits_one(self):
        primary = self.a_full_store()
        fingerprint_store(primary, write=True, runs_root=self.root)
        shard = primary.parent / "store-shard0001.sqlite"
        import sqlite3

        conn = sqlite3.connect(shard)
        with conn:
            conn.execute(f"DELETE FROM {QSI} WHERE serial = 3")
        conn.close()
        report, out, _ = self.command(
            "store", "fingerprint", primary, "--runs-root", self.root
        )
        self.assertEqual(report["code"], 1)
        self.assertIn("recorded: 2 difference(s)", out)
        self.assertIn(f"{QSI}: tag set [Run_fixture] differs: 2 recorded, 1 now", out)
        self.assertIn(f"{QSI}: orphan-tag problems: 0 recorded, 1 now", out)

    def test_write_is_refused_on_an_absent_legacy_or_problem_sidecar(self):
        for kind in ("absent", "legacy", "problem"):
            with self.subTest(sidecar=kind):
                primary = self.a_full_store(f"W-{kind}", sidecar=kind)
                before = self.state()
                with self.assertRaises(RuntimeError) as caught:
                    fingerprint_store(primary, write=True, runs_root=self.root)
                self.assertIn("store create", str(caught.exception))
                self.assertIn("store adopt", str(caught.exception))
                self.assertEqual(self.state(), before, "a refused write wrote")
                # and read-only it is not refused
                result = fingerprint_store(primary, runs_root=self.root)
                self.assertFalse(result["wrote"])
                self.assertIsNone(result["comparison"])
        err = self.refused(
            "store",
            "fingerprint",
            self.top / "W-legacy" / "store.sqlite",
            "--write",
            "--runs-root",
            self.root,
        )
        self.assertIn("legacy", err)


class TestRunningRefusal(_FullStoreCase):
    """Test 10: a running run naming the store, alive or stale, by path or by store_id."""

    def by_store_id_only(self, run):
        """Point ``run``'s manifest at another path, so that only its results_store_id names
        the store. The manifest is test data here."""
        manifest = load(Path(run.manifest_path))
        self.assertIsNotNone(manifest["results_store_id"])
        manifest["results"] = str(self.top / "elsewhere" / "other.sqlite")
        Path(run.manifest_path).write_text(json.dumps(manifest))
        return run

    def test_refused_alive_or_stale_by_path_or_store_id(self):
        by_path = self.a_full_store("P", sidecar="absent")
        by_id = self.a_full_store("I")
        cases = {
            ("alive", "path"): (
                by_path,
                lambda: self.a_running_run("alive-path", by_path),
            ),
            ("stale", "path"): (
                by_path,
                lambda: self.a_stale_run("stale-path", by_path),
            ),
            ("alive", "store_id"): (
                by_id,
                lambda: self.by_store_id_only(self.a_running_run("alive-id", by_id)),
            ),
            ("stale", "store_id"): (
                by_id,
                lambda: self.by_store_id_only(self.a_stale_run("stale-id", by_id)),
            ),
        }
        for (liveness, matched), (primary, make) in cases.items():
            run = make()
            for write in (False, True):
                with self.subTest(liveness=liveness, matched=matched, write=write):
                    before = self.state()
                    with self.assertRaises(RuntimeError) as caught:
                        fingerprint_store(primary, write=write, runs_root=self.root)
                    message = str(caught.exception)
                    self.assertIn(run.id, message)
                    self.assertIn(f"({liveness},", message)
                    self.assertIn(
                        (
                            "by its results_store_id. "
                            if matched == "store_id"
                            else "by its results. "
                        ),
                        message,
                    )
                    self.assertEqual(self.state(), before)
            # the same run, as the caller, does not refuse itself
            result = fingerprint_store(primary, runs_root=self.root, taken_by=run)
            self.assertEqual(result["fingerprint"]["taken"]["run_id"], run.id)
            run.finish("killed")

    def test_the_command_refuses(self):
        primary = self.a_full_store()
        run = self.a_running_run("in-use", primary)
        for extra in ([], ["--write"]):
            err = self.refused(
                "store", "fingerprint", primary, *extra, "--runs-root", self.root
            )
            self.assertIn(run.id, err)


class TestFinish(_FullStoreCase):
    """Test 11: `Run.finish(..., fingerprint=True)`."""

    def test_done_with_the_fingerprint_in_the_status_and_the_sidecar(self):
        primary = self.a_full_store()
        run = self.begin("fingerprinted", results=str(primary))
        status = run.finish("done", exit_code=0, fingerprint=True)
        self.assertEqual((status["state"], status["exit_code"]), ("done", 0))
        self.assertEqual(run.status(), status)
        self.assertNotIn("fingerprint_error", status)
        self.assertEqual(status["fingerprint_sidecar"], "written")
        sidecar = load(stores.sidecar_path(primary))
        self.assertEqual(status["fingerprint"], sidecar["fingerprint"])
        self.assertEqual(status["fingerprint"]["taken"]["run_id"], run.id)
        self.assertEqual(
            without_taken(status["fingerprint"]), without_taken(self.expected(primary))
        )
        self.assertNotIn("fingerprint", load(Path(run.manifest_path)))

    def test_every_terminal_state_takes_one(self):
        primary = self.a_full_store()
        for state, code in (("failed", 1), ("killed", 15)):
            with self.subTest(state=state):
                run = self.begin(f"state-{state}", results=str(primary))
                status = run.finish(state, exit_code=code, fingerprint=True)
                self.assertEqual((status["state"], status["exit_code"]), (state, code))
                self.assertIn("fingerprint", status)

    def test_a_refused_store_is_recorded_and_the_run_still_ends(self):
        primary = self.a_full_store()
        journal = Path(str(primary.parent / "store-shard0000.sqlite") + "-journal")
        journal.write_bytes(b"")
        sidecar = stores.sidecar_path(primary).read_bytes()
        run = self.begin("refused", results=str(primary))
        status = run.finish("done", exit_code=0, fingerprint=True)
        self.assertEqual((status["state"], status["exit_code"]), ("done", 0))
        self.assertNotIn("fingerprint", status)
        self.assertIsInstance(status["fingerprint_error"], str)
        self.assertIn("-journal", status["fingerprint_error"])
        self.assertEqual(stores.sidecar_path(primary).read_bytes(), sidecar)

    def test_another_running_run_is_recorded_and_the_run_still_ends(self):
        primary = self.a_full_store()
        other = self.a_running_run("other", primary)
        run = self.begin("second", results=str(primary))
        status = run.finish("failed", exit_code=2, fingerprint=True)
        self.assertEqual((status["state"], status["exit_code"]), ("failed", 2))
        self.assertIn(other.id, status["fingerprint_error"])
        self.assertNotIn(run.id, status["fingerprint_error"])
        self.assertNotIn("fingerprint", status)

    def test_a_legacy_sidecar_is_untouched_and_says_why(self):
        primary = self.a_full_store(sidecar="legacy")
        sidecar = stores.sidecar_path(primary).read_bytes()
        run = self.begin("legacy", results=str(primary))
        status = run.finish("done", exit_code=0, fingerprint=True)
        self.assertEqual(status["state"], "done")
        self.assertIn("fingerprint", status)
        self.assertTrue(status["fingerprint_sidecar"].startswith("not written"))
        self.assertIn("legacy", status["fingerprint_sidecar"])
        self.assertEqual(stores.sidecar_path(primary).read_bytes(), sidecar)

    def test_no_results_store_is_recorded(self):
        run = self.begin("no-results")
        status = run.finish("done", exit_code=0, fingerprint=True)
        self.assertEqual(status["state"], "done")
        self.assertIn("names no results store", status["fingerprint_error"])

    def test_fingerprint_false_writes_neither(self):
        primary = self.a_full_store()
        sidecar = stores.sidecar_path(primary).read_bytes()
        run = self.begin("plain", results=str(primary))
        status = run.finish("done", exit_code=0)
        for key in ("fingerprint", "fingerprint_error", "fingerprint_sidecar"):
            self.assertNotIn(key, status)
        self.assertEqual(stores.sidecar_path(primary).read_bytes(), sidecar)


def _finish_calls(path: Path):
    """``[(inside a terminal handler, passes fingerprint=True)]`` for every ``run.finish(``."""
    tree = ast.parse(path.read_text())
    found = []

    def visit(node, inside):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inside = inside or node.name == "terminal"
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "finish"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "run"
        ):
            passes = any(
                k.arg == "fingerprint"
                and isinstance(k.value, ast.Constant)
                and k.value.value is True
                for k in node.keywords
            )
            found.append((inside, passes))
        for child in ast.iter_child_nodes(node):
            visit(child, inside)

    visit(tree, False)
    return found


class TestDrivers(unittest.TestCase):
    """Test 12: every `run.finish(` after the pipeline passes `fingerprint=True`; the `terminal`
    signal handlers, which run with the pool open, do not."""

    def test_the_finish_calls(self):
        expected = {
            "scoped_pipeline_run.py": (3, 1),
            "quadsource_atol_sweep.py": (6, 2),
        }
        for path in DRIVERS:
            with self.subTest(driver=path.name):
                calls = _finish_calls(path)
                outside = [passes for inside, passes in calls if not inside]
                inside = [passes for inside, passes in calls if inside]
                self.assertEqual((len(outside), len(inside)), expected[path.name])
                self.assertTrue(all(outside), outside)
                self.assertFalse(any(inside), inside)


if __name__ == "__main__":
    unittest.main()
