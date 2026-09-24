"""
``tools/sharded_store.py``, the command-line client of ``ShardedPool.copy_store`` /
``move_store``. `prompts/datastore-portability` prompt 02 §2 P7 and §3 test 8.

It is run here the way a person runs it: as a subprocess, from a directory other than the
repository root, with ``PYTHONPATH`` removed from the environment. It imports ``ShardedPool`` and
so ``ray``, and must never initialise Ray. That is checked by running the script through
``runpy`` inside a child interpreter and asking that interpreter, after the script has finished,
whether ``ray`` was imported and whether ``ray.is_initialized()``.

Everything is in a temporary directory; no Ray, no datastore.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from Datastore.shard_paths import shard_file_name
from Datastore.tests.shard_store_fixtures import (
    stored_records,
    tree_state,
    write_new_store,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "sharded_store.py"


def _clean_env():
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


class TestShardedStoreScript(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.src = self.root / "A" / "store.sqlite"
        write_new_store(self.src, shards=2)
        self.cwd = self.root / "elsewhere"
        self.cwd.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def run_script(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, args)],
            capture_output=True,
            text=True,
            cwd=self.cwd,
            env=_clean_env(),
        )

    def test_copy_succeeds_from_another_directory_without_pythonpath(self):
        dst = self.root / "B" / "copy.sqlite"
        result = self.run_script("copy", self.src, dst)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected = {i: shard_file_name(dst, i) for i in range(2)}
        self.assertEqual(stored_records(dst), expected)
        for serial, name in expected.items():
            self.assertIn(f"shard #{serial}: {dst.parent / name}", result.stdout)
        self.assertTrue(self.src.exists())

    def test_move_succeeds(self):
        dst = self.root / "B" / "moved.sqlite"
        result = self.run_script("move", self.src, dst)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            stored_records(dst), {i: shard_file_name(dst, i) for i in range(2)}
        )
        self.assertFalse(self.src.exists())

    def test_refusal_exits_nonzero_and_writes_nothing(self):
        taken = self.root / "B" / shard_file_name("copy.sqlite", 1)
        taken.parent.mkdir()
        taken.write_bytes(b"something already here")
        before = tree_state(self.root)

        result = self.run_script("copy", self.src, self.root / "B" / "copy.sqlite")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(taken), result.stderr)
        self.assertIn("already exist", result.stderr)
        self.assertEqual(tree_state(self.root), before)

    def test_ray_is_imported_but_never_initialised(self):
        dst = self.root / "B" / "copy.sqlite"
        code = (
            "import runpy, sys\n"
            f"sys.argv = [{str(SCRIPT)!r}, 'copy', {str(self.src)!r}, {str(dst)!r}]\n"
            "try:\n"
            f"    runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
            "except SystemExit as e:\n"
            "    code = e.code\n"
            "import ray\n"
            "print('EXIT', code)\n"
            "print('RAY IMPORTED', 'ray' in sys.modules)\n"
            "print('RAY INITIALIZED', ray.is_initialized())\n"
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
        self.assertIn("RAY IMPORTED True", result.stdout)
        self.assertIn("RAY INITIALIZED False", result.stdout)
        self.assertTrue(dst.exists())

    def test_help_carries_the_three_statements(self):
        result = self.run_script("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        text = " ".join(result.stdout.split())
        # the primary and its shards only; a manifest is left where it is
        self.assertIn("handles the primary and its shards and nothing else", text)
        self.assertIn("<stem>.manifest.json is neither copied nor moved", text)
        # it cannot tell whether the store is open; that is the caller's job
        self.assertIn("cannot tell whether a process has the store open", text)
        self.assertIn("rollback journal", text)
        self.assertIn("the caller's job", text)
        # a registry-level tool is the place for both
        self.assertIn("A registry-level tool is the place that does both", text)
        self.assertIn("does not consult the registry", text)


if __name__ == "__main__":
    unittest.main()
