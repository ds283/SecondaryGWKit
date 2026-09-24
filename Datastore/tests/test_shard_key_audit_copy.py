"""
``tools/shard_key_audit.py`` finds the shard it attaches exactly as ``ShardedPool`` does, so
auditing a **copied** store checks the copy's shard 0 and not the original's.
`prompts/datastore-portability` prompt 01 §2 P4 and §6 item 5.

Before prompt 01 the tool attached ``Path(shards.filename)`` as it stood. On a copied primary that
is the original's shard, so the audit of the copy silently reported on the original. The tool now
resolves the record through ``Datastore/shard_paths.py``, the same resolver ``ShardedPool`` uses.

The tool must also stay a standalone, read-only script: it is run here as
``python tools/shard_key_audit.py <primary>`` from an unrelated working directory with no
``PYTHONPATH``, and must not import ray or sqlalchemy.

Everything is in a temporary directory; no Ray, no datastore. This module does not import
``Datastore.shard_paths``, so that it can be run against the unfixed tool for the
deliberate-breakage record.
"""

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from Datastore.tests.shard_store_fixtures import (
    sha256,
    write_legacy_primary,
    write_placeholder,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "shard_key_audit.py"
NAMES = [f"store-shard{i:04d}.sqlite" for i in range(3)]


def _write_key_table(path: Path, serials) -> None:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        with conn:
            conn.execute("CREATE TABLE wavenumber (serial INTEGER PRIMARY KEY)")
            conn.executemany(
                "INSERT INTO wavenumber (serial) VALUES (?)", [(s,) for s in serials]
            )
    finally:
        conn.close()


def _clean_env():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return env


class TestAuditOfACopiedStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name).resolve()

        # the original, A: legacy absolute records naming its own shards; shard 0 has 3 keys
        self.a = root / "A"
        write_legacy_primary(
            self.a / "store.sqlite",
            {i: str(self.a / n) for i, n in enumerate(NAMES)},
            shard_keys=[(1, 0), (2, 1)],
        )
        _write_key_table(self.a / NAMES[0], [1, 2, 3])
        for n in NAMES[1:]:
            write_placeholder(self.a / n)

        # the copy, B: same records (still naming A), but its own shard 0 has 2 keys
        self.b = root / "B"
        shutil.copytree(self.a, self.b)
        _write_key_table(self.b / NAMES[0], [1, 2])

        self.cwd = root / "elsewhere"
        self.cwd.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_audit_of_the_copy_attaches_the_copys_shard(self):
        a_before = {p.name: sha256(p) for p in self.a.iterdir()}
        b_before = {p.name: sha256(p) for p in self.b.iterdir()}

        result = subprocess.run(
            [sys.executable, str(TOOL), str(self.b / "store.sqlite")],
            capture_output=True,
            text=True,
            cwd=self.cwd,
            env=_clean_env(),
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(
            f"cross-file check against shard #0: {self.b / NAMES[0]}", result.stdout
        )
        self.assertIn("'wavenumber' table (shard #0) row count: 2", result.stdout)
        self.assertIn("VERDICT: OK", result.stdout)

        # read-only: neither store changed
        self.assertEqual({p.name: sha256(p) for p in self.a.iterdir()}, a_before)
        self.assertEqual({p.name: sha256(p) for p in self.b.iterdir()}, b_before)

    def test_audit_does_not_fall_back_to_the_original_when_the_copy_lacks_shard_0(self):
        (self.b / NAMES[0]).unlink()

        result = subprocess.run(
            [sys.executable, str(TOOL), str(self.b / "store.sqlite")],
            capture_output=True,
            text=True,
            cwd=self.cwd,
            env=_clean_env(),
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"{self.b / NAMES[0]} does not exist", result.stdout)
        self.assertIn(
            "Cross-file check against the shard-key table was not possible",
            result.stdout,
        )
        self.assertNotIn("row count: 3", result.stdout)

    def test_tool_imports_no_heavy_dependency(self):
        code = (
            "import runpy, sys\n"
            f"sys.argv = [{str(TOOL)!r}, {str(self.b / 'store.sqlite')!r}]\n"
            "try:\n"
            f"    runpy.run_path({str(TOOL)!r}, run_name='__main__')\n"
            "except SystemExit as e:\n"
            "    code = e.code\n"
            "print('EXIT', code)\n"
            "print('HEAVY', sorted(m for m in ('ray', 'sqlalchemy', 'Datastore.SQL') "
            "if m in sys.modules))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=self.cwd,
            env=_clean_env(),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("EXIT 0", result.stdout)
        self.assertIn("HEAVY []", result.stdout)


if __name__ == "__main__":
    unittest.main()
