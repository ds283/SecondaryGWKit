"""
The one shard naming rule, ``Datastore/shard_paths.py`` ``shard_file_name``.
`prompts/datastore-portability` prompt 02 §2 P5 and §3 test 1.

Until prompt 02 the constructor named shard *i* of a new store inline, as
``primary.with_stem(f"{stem}-shard{i:04d}")``, and ``shard_store_fixtures.write_new_store``
repeated the pattern. Now both call ``shard_file_name``, and so does ``ShardedPool.copy_store`` /
``move_store`` when it names a destination's shards. New stores must be named exactly as before:
the expected names below are literals, spelled out on purpose, because they are what the function
is tested against.

No Ray, no datastore.
"""

import re
import tempfile
import unittest
from pathlib import Path

from Datastore.shard_paths import shard_file_name
from Datastore.tests.shard_store_fixtures import stored_records, write_new_store

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestShardFileName(unittest.TestCase):
    def test_reproduces_the_constructors_old_names(self):
        cases = [
            ("/data/store.sqlite", 0, "store-shard0000.sqlite"),
            ("/data/store.sqlite", 3, "store-shard0003.sqlite"),
            ("/data/store.sqlite", 42, "store-shard0042.sqlite"),
            ("/data/store.sqlite", 9999, "store-shard9999.sqlite"),
            ("/data/store.sqlite", 12345, "store-shard12345.sqlite"),
            (
                "/x/handover-atol-sweep.sqlite",
                1,
                "handover-atol-sweep-shard0001.sqlite",
            ),
            ("/x/pcopy.sqlite", 2, "pcopy-shard0002.sqlite"),
            ("/x/a.b.sqlite", 0, "a.b-shard0000.sqlite"),
            ("/x/store.db", 7, "store-shard0007.db"),
            ("/x/store", 5, "store-shard0005"),
            ("relative/store.sqlite", 1, "store-shard0001.sqlite"),
        ]
        for primary, serial, expected in cases:
            with self.subTest(primary=primary, serial=serial):
                self.assertEqual(shard_file_name(primary, serial), expected)
                self.assertEqual(shard_file_name(Path(primary), serial), expected)
                # and the expression the constructor used before prompt 02, verbatim
                p = Path(primary)
                self.assertEqual(
                    p.with_stem(f"{p.stem}-shard{serial:04d}").name, expected
                )

    def test_returns_a_bare_name_independent_of_the_directory(self):
        a = shard_file_name("/one/place/store.sqlite", 1)
        b = shard_file_name("/another/store.sqlite", 1)
        self.assertEqual(a, b)
        self.assertIsInstance(a, str)
        self.assertNotIn("/", a)

    def test_a_new_store_written_through_the_fixtures_is_named_by_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            primary = root / "named.sqlite"
            pool = write_new_store(primary, shards=3)

            expected = {i: shard_file_name(primary, i) for i in range(3)}
            self.assertEqual(
                expected,
                {
                    0: "named-shard0000.sqlite",
                    1: "named-shard0001.sqlite",
                    2: "named-shard0002.sqlite",
                },
            )
            self.assertEqual(stored_records(primary), expected)
            self.assertEqual(
                pool._shard_db_files, {i: root / n for i, n in expected.items()}
            )
            self.assertEqual(
                sorted(p.name for p in root.iterdir()),
                sorted(["named.sqlite", *expected.values()]),
            )

    def test_the_pattern_is_written_once_outside_the_tests(self):
        """The shard file name pattern appears in production code only in shard_paths.py. (The
        actor names ``shard{key:04d}-store`` are not file names and do not match.)"""
        pattern = re.compile(r"-shard\{")
        found = []
        for path in sorted(
            list((REPO_ROOT / "Datastore").rglob("*.py"))
            + list((REPO_ROOT / "tools").rglob("*.py"))
        ):
            if "tests" in path.relative_to(REPO_ROOT).parts:
                continue
            for n, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line):
                    found.append(f"{path.relative_to(REPO_ROOT)}:{n}")
        self.assertEqual(
            [f.split(":")[0] for f in found], ["Datastore/shard_paths.py"], found
        )


if __name__ == "__main__":
    unittest.main()
