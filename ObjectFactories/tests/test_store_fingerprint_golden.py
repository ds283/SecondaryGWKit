"""This repository's golden store fingerprint, and the module that takes it (runregistry-extraction
prompt 06; README §6.2, C5, C13 and C18).

What goes into a fingerprint is this repository's: the factories' inventory specs, the registry in
`config.datastore`, and the format number in `config.fingerprint`. How it is encoded and digested
is DatastoreKit's (`datastorekit.store_fingerprint`), and when and where it is taken is
RunRegistryKit's, through the fingerprinter `config.fingerprint` holds and `runregistrykit.toml`
names.

- `TestGolden` (store-fingerprint prompt 04, test 4), moved here by prompt 06 from the registry's
  `tests/test_store_fingerprint.py` with `_FullStores` and the helpers it uses: the fingerprint of
  `build_full_store()` is the committed golden file `GOLDEN`.
- `TestConfigFingerprint`: `config/fingerprint.py` and the config's naming of it.

The golden files are under `data/`, one per format change and never overwritten. `GOLDEN` is the
current one, `datastore-integrity-09c`'s with DatastoreKit's `encoding_format` added; its digest has
not moved, so the format is still 6.

Stores are built with `StoreFixtures.real_store_fixtures` in temporary directories. No Ray is
initialised and nothing under `var/` is opened; `TestConfigFingerprint` reads
`runregistrykit.toml` at the repository root.
"""

import json
import tempfile
import unittest
from pathlib import Path

import runregistrykit
import runregistrykit.stores
from datastorekit.store_fingerprint import (
    ENCODING_FORMAT,
    compare_fingerprints,
    fingerprint_of,
)
from datastorekit.store_inventory import read_inventory
from runregistrykit.fingerprint import DatastoreKitFingerprinter, load_fingerprinter

import config.datastore
import config.fingerprint
from config.datastore import factories as _factories
from config.fingerprint import FINGERPRINT_FORMAT
from StoreFixtures.real_store_fixtures import build_full_store

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
GOLDEN = DATA / "full_store_fingerprint_runregistry-extraction-06.json"
GOLDEN_09C = DATA / "full_store_fingerprint_datastore-integrity-09c.json"
DIGEST_09C = "6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759"
QSI = "QuadSourceIntegral"


def without_taken(fingerprint) -> dict:
    out = dict(fingerprint)
    out.pop("taken", None)
    return out


class _FullStores(unittest.TestCase):
    """A temporary directory, and full stores built in it, each in its own directory."""

    # a failure names the class and the field that differ, rather than truncating the diff
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.top = Path(cls._tmp.name).resolve()
        cls._count = 0
        cls.baseline_inventory = read_inventory(cls.store().primary, _factories)
        cls.baseline = fingerprint_of(cls.baseline_inventory, FINGERPRINT_FORMAT)

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
        return fingerprint_of(
            read_inventory(cls.store(**kwargs).primary, _factories), FINGERPRINT_FORMAT
        )


class TestGolden(_FullStores):
    """Test 4: the fingerprint of `build_full_store()` is the committed golden file."""

    def test_the_golden_fingerprint(self):
        golden = json.loads(GOLDEN.read_text())
        self.assertEqual(golden["fingerprint_format"], FINGERPRINT_FORMAT)
        self.assertEqual(golden["encoding_format"], ENCODING_FORMAT)
        self.assertIsNone(golden["taken"])
        self.assertEqual(without_taken(self.baseline), without_taken(golden))
        # and it is deterministic: a second build gives the same
        self.assertEqual(without_taken(self.fingerprint()), without_taken(golden))

    def test_the_shape(self):
        fp = self.baseline
        self.assertEqual(
            set(fp),
            {
                "fingerprint_format",
                "encoding_format",
                "classes",
                "digest",
                "problems",
                "taken",
            },
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
        self.assertEqual(
            runregistrykit.stores._fingerprint_problems(dict(fp, taken={})), []
        )

    def test_taken_is_outside_every_digest(self):
        taken = {"when": "now", "git_head": "x", "git_dirty": True, "run_id": "r"}
        with_taken = fingerprint_of(self.baseline_inventory, FINGERPRINT_FORMAT, taken)
        self.assertEqual(with_taken["taken"], taken)
        self.assertEqual(without_taken(with_taken), without_taken(self.baseline))
        self.assertEqual(compare_fingerprints(self.baseline, with_taken), [])


class TestConfigFingerprint(_FullStores):
    """`config/fingerprint.py`, and `runregistrykit.toml`'s naming of its fingerprinter."""

    def test_the_config_names_the_fingerprinter(self):
        client = runregistrykit.find_client_root(client_root=REPO_ROOT)
        self.assertEqual(
            client.stores, {"fingerprinter": "config.fingerprint:fingerprinter"}
        )
        loaded = load_fingerprinter(client)
        self.assertIs(loaded, config.fingerprint.fingerprinter)
        self.assertIsInstance(loaded, DatastoreKitFingerprinter)
        self.assertEqual(loaded.fingerprint_format, FINGERPRINT_FORMAT)
        self.assertEqual(FINGERPRINT_FORMAT, 6)
        self.assertIs(loaded.factories, config.datastore.factories)

    def test_the_fingerprinter_takes_the_golden_fingerprint(self):
        primary = self.store().primary
        fp, _ = config.fingerprint.fingerprinter.take(primary, None)
        self.assertEqual(fp["encoding_format"], ENCODING_FORMAT)
        self.assertEqual(
            fp, fingerprint_of(read_inventory(primary, _factories), FINGERPRINT_FORMAT)
        )
        golden = json.loads(GOLDEN.read_text())
        self.assertEqual(without_taken(fp), without_taken(golden))

    def test_the_new_golden_is_the_old_one_with_its_encoding(self):
        golden = json.loads(GOLDEN.read_text())
        old = json.loads(GOLDEN_09C.read_text())
        self.assertIn("encoding_format", golden)
        self.assertNotIn("encoding_format", old)
        self.assertEqual(
            {k: v for k, v in golden.items() if k != "encoding_format"}, old
        )
        self.assertEqual(old["digest"], DIGEST_09C)
        self.assertEqual(golden["digest"], DIGEST_09C)


if __name__ == "__main__":
    unittest.main()
