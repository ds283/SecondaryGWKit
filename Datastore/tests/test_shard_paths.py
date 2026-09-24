"""
The shard-path resolver, ``Datastore/shard_paths.py``, as a pure function.
`prompts/datastore-portability` prompt 01 §3, test 1.

``resolve_shard_path(primary, stored)`` is the one definition of where the shard that a
``shards.filename`` record names lives. It always returns a file in the primary's directory:

* a bare file name (the current form) resolves to that name beside the primary;
* an absolute path (the legacy form) resolves to its final component beside the primary, and
  **never to itself**, even when it exists and is somewhere else -- that fallback is exactly the
  copied-store failure the resolver exists to remove;
* anything else is refused.

No Ray, no datastore.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from Datastore.shard_paths import (
    is_legacy_record,
    resolve_shard_path,
    shard_file_problem,
)

PRIMARY = Path("/data/stores/B/store.sqlite")
SIBLING = Path("/data/stores/B/store-shard0000.sqlite")


class TestResolveShardPath(unittest.TestCase):
    def test_bare_name_resolves_to_the_sibling(self):
        self.assertEqual(resolve_shard_path(PRIMARY, "store-shard0000.sqlite"), SIBLING)
        self.assertEqual(
            resolve_shard_path(str(PRIMARY), "store-shard0000.sqlite"), SIBLING
        )

    def test_legacy_absolute_record_resolves_to_the_sibling_by_name(self):
        for stored in (
            "/data/stores/A/store-shard0000.sqlite",
            "/somewhere/else/entirely/store-shard0000.sqlite",
            str(SIBLING),
        ):
            with self.subTest(stored=stored):
                self.assertEqual(resolve_shard_path(PRIMARY, stored), SIBLING)

    def test_legacy_record_is_not_used_even_when_it_exists_and_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            original = root / "A" / "store-shard0000.sqlite"
            original.parent.mkdir()
            original.write_bytes(b"the original's shard")
            primary = root / "B" / "store.sqlite"  # B's sibling does not even exist

            resolved = resolve_shard_path(primary, str(original))

            self.assertEqual(resolved, root / "B" / "store-shard0000.sqlite")
            self.assertNotEqual(resolved, original)
            self.assertTrue(original.exists())
            self.assertFalse(resolved.exists())

    def test_result_is_absolute_and_in_the_primarys_directory(self):
        for stored in (
            "store-shard0003.sqlite",
            "/x/y/store-shard0003.sqlite",
            "a b.sqlite",
        ):
            with self.subTest(stored=stored):
                resolved = resolve_shard_path(PRIMARY, stored)
                self.assertTrue(resolved.is_absolute())
                self.assertEqual(resolved.parent, PRIMARY.parent)

    def test_refused_records(self):
        for stored in (
            "",
            ".",
            "..",
            "sub/store-shard0000.sqlite",
            "./store-shard0000.sqlite",
            "../store-shard0000.sqlite",
            "../../etc/passwd",
            "a\\b.sqlite",
            "C:\\stores\\store-shard0000.sqlite",
            "/",
            "/data/stores/A/..",
            "/data/stores/A/.",
            "/data/stores/A/",
            "store\x00shard.sqlite",
        ):
            with self.subTest(stored=stored):
                with self.assertRaises(ValueError):
                    resolve_shard_path(PRIMARY, stored)

    def test_non_string_record_is_refused(self):
        for stored in (
            None,
            b"store-shard0000.sqlite",
            0,
            Path("store-shard0000.sqlite"),
        ):
            with self.subTest(stored=stored):
                with self.assertRaises(ValueError):
                    resolve_shard_path(PRIMARY, stored)

    def test_relative_primary_is_refused(self):
        with self.assertRaises(ValueError):
            resolve_shard_path(Path("stores/store.sqlite"), "store-shard0000.sqlite")

    def test_is_legacy_record(self):
        self.assertTrue(is_legacy_record("/a/store-shard0000.sqlite"))
        self.assertFalse(is_legacy_record("store-shard0000.sqlite"))
        self.assertFalse(is_legacy_record(None))


class TestShardFileProblem(unittest.TestCase):
    def test_each_kind_of_unusable_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            good = root / "good.sqlite"
            good.write_bytes(b"x")
            directory = root / "dir.sqlite"
            directory.mkdir()
            link = root / "link.sqlite"
            link.symlink_to(good)
            dangling = root / "dangling.sqlite"
            dangling.symlink_to(root / "nowhere.sqlite")

            self.assertIsNone(shard_file_problem(good))
            self.assertIn("does not exist", shard_file_problem(root / "missing.sqlite"))
            self.assertIn("not a regular file", shard_file_problem(directory))
            self.assertIn("symbolic link", shard_file_problem(link))
            self.assertIn("symbolic link", shard_file_problem(dangling))


class TestModuleIsStandalone(unittest.TestCase):
    """Prompt §2 P4: the audit tool imports this module and must stay a standalone script, so
    importing it must not pull in ray, sqlalchemy or the Datastore.SQL package."""

    def test_import_pulls_in_no_heavy_dependency(self):
        repo_root = Path(__file__).resolve().parents[2]
        code = (
            "import sys; import Datastore.shard_paths; "
            "print(sorted(m for m in ('ray', 'sqlalchemy', 'Datastore.SQL') if m in sys.modules))"
        )
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(repo_root)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            cwd=tempfile.gettempdir(),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
