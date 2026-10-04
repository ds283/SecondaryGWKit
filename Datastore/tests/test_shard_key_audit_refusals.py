"""
``tools/shard_key_audit.py`` refuses, with exit code 2 and no verdict, a primary that is not a
completely written current-generation ShardedPool primary, and one whose shard record is unusable.
`prompts/datastore-generic` prompt 04 §2 L2.

Until prompt 04 the tool had a branch for a pre-``a2bd966`` primary (a ``shard_keys`` table keyed
``wavenumber_serial``), tolerated a primary with no ``shard_key_config``, and ended an unusable
shard record in "VERDICT: OK" and exit code 0, however unusable it was. Now:

* ``shard_keys`` without ``key_serial`` meets the one "Unrecognised shard_keys schema" refusal,
  which does not mention ``a2bd966``;
* ``shard_key_config`` is a required table, and must hold a row;
* every ``shards`` record goes through the one resolver, and one that it refuses, an absolute
  path included, is a refusal naming the record and the resolver's reason.

A missing shard *file* is not a refusal: the cross-file check is skipped and said to be
(``test_shard_key_audit_copy`` pins it).

The tool is run as ``python tools/shard_key_audit.py <primary>`` from an unrelated working
directory with no ``PYTHONPATH``. Everything is in a temporary directory; no Ray, no datastore.
"""

import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from Datastore.tests.shard_store_fixtures import (
    tree_state,
    write_hand_built_primary,
    write_new_store,
    write_placeholder,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "shard_key_audit.py"
NAMES = [f"store-shard{i:04d}.sqlite" for i in range(3)]


def _execute(primary: Path, *statements: str) -> None:
    conn = sqlite3.connect(primary)
    try:
        with conn:
            for statement in statements:
                conn.execute(statement)
    finally:
        conn.close()


def _write_key_shard(path: Path) -> None:
    """Shard 0 as the tool's cross-file check reads it: a database with the key table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _execute(
        path,
        "CREATE TABLE wavenumber (serial INTEGER PRIMARY KEY)",
        "INSERT INTO wavenumber (serial) VALUES (1)",
    )


class TestTheAuditRefuses(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.cwd = self.root / "elsewhere"
        self.cwd.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def a_current_store(self, directory: str) -> Path:
        """A hand-built primary with bare-name records, a key in shard 0 and a shard_key_config
        row: the shape the tool audits to "VERDICT: OK"."""
        primary = self.root / directory / "store.sqlite"
        write_hand_built_primary(primary, dict(enumerate(NAMES)), shard_keys=[(1, 0)])
        for name in NAMES[1:]:
            write_placeholder(primary.parent / name)
        _write_key_shard(primary.parent / NAMES[0])
        return primary

    def audit(self, primary: Path):
        return subprocess.run(
            [sys.executable, str(TOOL), str(primary)],
            capture_output=True,
            text=True,
            cwd=self.cwd,
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        )

    def assertRefusesWithoutAVerdict(self, primary: Path):
        """The tool exits 2, prints no verdict, and changes nothing under the temporary root.
        Returns what it printed."""
        before = tree_state(self.root)
        result = self.audit(primary)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertNotIn("VERDICT", result.stdout)
        self.assertEqual(result.stderr, "")
        self.assertEqual(tree_state(self.root), before)
        return result.stdout

    def test_a_current_store_is_still_audited(self):
        """The control: the same fixture, untouched, is not refused."""
        result = self.audit(self.a_current_store("A"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("VERDICT: OK", result.stdout)

    def test_a_primary_naming_another_stores_shards_by_absolute_path(self):
        """B's records are the absolute paths of A's existing shard files: the 2026-09-23 shape.
        Before prompt 04 the tool read them as B's siblings and said OK."""
        a = self.a_current_store("A")
        b = self.root / "B" / "store.sqlite"
        records = {i: str(a.parent / n) for i, n in enumerate(NAMES)}
        write_hand_built_primary(b, records, shard_keys=[(1, 0)])
        for name in NAMES:
            write_placeholder(b.parent / name)

        out = self.assertRefusesWithoutAVerdict(b)

        self.assertIn("shard #0 record is unusable", out)
        self.assertIn("is not a bare file name", out)
        self.assertIn(records[0], out)
        self.assertNotIn("cross-file check against shard", out)

    def test_a_primary_naming_its_own_siblings_by_absolute_path(self):
        """The shape every store written before ``b04671f`` held."""
        b = self.root / "B" / "store.sqlite"
        records = {i: str(b.parent / n) for i, n in enumerate(NAMES)}
        write_hand_built_primary(b, records, shard_keys=[(1, 0)])
        for name in NAMES:
            write_placeholder(b.parent / name)

        out = self.assertRefusesWithoutAVerdict(b)

        self.assertIn("shard #0 record is unusable", out)
        self.assertIn(records[0], out)

    def test_a_record_that_is_unusable_in_a_later_shard_is_refused_too(self):
        primary = self.root / "B" / "store.sqlite"
        records = dict(enumerate(NAMES))
        records[2] = "../outside.sqlite"
        write_hand_built_primary(primary, records, shard_keys=[(1, 0)])
        for name in NAMES:
            write_placeholder(primary.parent / name)

        out = self.assertRefusesWithoutAVerdict(primary)

        self.assertIn("shard #2 record is unusable", out)
        self.assertIn("../outside.sqlite", out)

    def test_a_primary_without_shard_key_config(self):
        primary = self.a_current_store("A")
        _execute(primary, "DROP TABLE shard_key_config")

        out = self.assertRefusesWithoutAVerdict(primary)

        self.assertIn("shard_key_config", out)
        self.assertIn("completely written", out)

    def test_a_primary_whose_shard_key_config_is_empty(self):
        primary = self.a_current_store("A")
        _execute(primary, "DELETE FROM shard_key_config")

        out = self.assertRefusesWithoutAVerdict(primary)

        self.assertIn("shard_key_config", out)
        self.assertIn("holds no row", out)

    def test_a_shard_keys_table_without_key_serial(self):
        """The pre-``a2bd966`` shape. It meets the one refusal, which says nothing of that
        commit."""
        primary = self.a_current_store("A")
        _execute(
            primary,
            "DROP TABLE shard_keys",
            "CREATE TABLE shard_keys (wavenumber_serial INTEGER NOT NULL PRIMARY KEY, "
            "shard_id INTEGER NOT NULL REFERENCES shards (serial))",
            "INSERT INTO shard_keys (wavenumber_serial, shard_id) VALUES (1, 0)",
        )

        out = self.assertRefusesWithoutAVerdict(primary)

        self.assertIn("Unrecognised shard_keys schema", out)
        self.assertIn("wavenumber_serial", out)  # the columns it found, as data
        self.assertNotIn("a2bd966", out)
        self.assertNotIn("pre-", out)


if __name__ == "__main__":
    unittest.main()
