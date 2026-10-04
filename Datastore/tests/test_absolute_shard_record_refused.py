"""
A primary whose ``shards`` records are **absolute paths** is refused by every reader, and neither
it nor the store it names is read, written, moved or deleted.
`prompts/datastore-generic` prompt 04 §2 L3.

This is the safety property that the legacy-record tests used to guard. A primary written before
``b04671f`` (2026-09-24) recorded its shards by absolute path, and a copy of such a primary still
named the **original's** shards. On 2026-09-23 a pool opened on such a copy read and wrote the
original (54 rows into the A3 baseline store). Before prompt 04 the property held because an
absolute record was read by its final component, as a sibling of the primary. Since prompt 04 it
holds because the record is refused: ``resolve_shard_path`` raises ``ValueError`` on anything but a
bare file name, and every reader reports "Shard #N ... has an unusable record".

Two stores are built in a temporary directory, each with stand-in shard files (small placeholders
holding a marker, as in ``shard_store_fixtures``; everything here is refused before any shard is
opened as a database):

* **A**, whose records are bare names;
* **B**, whose records are the absolute paths of A's existing shard files. B has shard files of its
  own beside its primary, under the same names. This is the shape of the 2026-09-23 failure.

For each entry point, B is refused, the error names shard #0 and "unusable record", and the whole
tree, both stores, is unchanged: every file's content, every mtime, every directory entry.

A third store, **C**, records absolute paths naming **its own** siblings, the shape every store
written before ``b04671f`` held. It was read by name before prompt 04; it is refused now, like B.

No Ray, no datastore, nothing under ``var/``.
"""

import tempfile
import unittest
from pathlib import Path

from Datastore.SQL.ShardedPool import ShardedPool
from Datastore.shard_paths import resolve_shard_path, shard_file_name
from Datastore.store_reader import open_read_only
from Datastore.tests.shard_store_fixtures import (
    read_pool,
    tree_state,
    write_hand_built_primary,
    write_new_store,
    write_placeholder,
)

N_SHARDS = 3


class _TwoStores(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # resolved: on macOS the temporary directory is reached through /var -> /private/var
        self.root = Path(self._tmp.name).resolve()

        self.a = self.root / "A" / "store.sqlite"
        write_new_store(self.a, shards=N_SHARDS)

        # B names A's files by absolute path, and has its own beside its primary
        self.b = self.root / "B" / "store.sqlite"
        self.b_records = {i: str(p) for i, p in self.shard_paths(self.a).items()}
        write_hand_built_primary(self.b, self.b_records, shard_keys=[(7, 2)])
        for path in self.shard_paths(self.b).values():
            write_placeholder(path)

        # C names its own siblings by absolute path
        self.c = self.root / "C" / "store.sqlite"
        self.c_records = {i: str(p) for i, p in self.shard_paths(self.c).items()}
        write_hand_built_primary(self.c, self.c_records, shard_keys=[(7, 2)])
        for path in self.shard_paths(self.c).values():
            write_placeholder(path)

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def shard_paths(primary: Path):
        return {
            i: primary.parent / shard_file_name(primary, i) for i in range(N_SHARDS)
        }

    def assertRefused(self, call, record, primary=None):
        """``call()`` raises ``RuntimeError`` naming shard #0, "unusable record" and the record,
        and nothing under the temporary root changed. Returns the message."""
        before = tree_state(self.root)
        with self.assertRaises(RuntimeError) as ctx:
            call()
        message = str(ctx.exception)
        self.assertIn("Shard #0", message)
        self.assertIn("unusable record", message)
        self.assertIn(record, message)
        self.assertIn("is not a bare file name", message)
        if primary is not None:
            self.assertIn(str(primary.resolve()), message)
        self.assertEqual(tree_state(self.root), before)
        return message


class TestTheResolver(_TwoStores):
    def test_resolve_shard_path_refuses_an_absolute_record(self):
        for record in (*self.b_records.values(), *self.c_records.values()):
            with self.subTest(record=record):
                with self.assertRaises(ValueError) as ctx:
                    resolve_shard_path(self.b, record)
                self.assertIn(record, str(ctx.exception))
                self.assertIn("is not a bare file name", str(ctx.exception))

    def test_the_bare_name_of_the_same_file_is_not_refused(self):
        """The control: it is the record's form that is refused, not the file it names."""
        for serial, record in self.b_records.items():
            with self.subTest(serial=serial):
                self.assertEqual(
                    resolve_shard_path(self.b, Path(record).name),
                    self.b.parent / Path(record).name,
                )


class TestEveryReader(_TwoStores):
    def test_the_constructors_read_of_the_shards_table(self):
        self.assertRefused(lambda: read_pool(self.b), self.b_records[0], self.b)

    def test_a_primary_naming_its_own_siblings_is_refused_the_same_way(self):
        """What every store written before ``b04671f`` held. One reader is enough: the others
        go through the same resolver."""
        self.assertRefused(lambda: read_pool(self.c), self.c_records[0], self.c)

    def test_copy_store(self):
        dst = self.root / "D" / "copy.sqlite"
        self.assertRefused(
            lambda: ShardedPool.copy_store(self.b, dst), self.b_records[0], self.b
        )
        self.assertFalse(dst.parent.exists())

    def test_move_store(self):
        dst = self.root / "D" / "moved.sqlite"
        self.assertRefused(
            lambda: ShardedPool.move_store(self.b, dst), self.b_records[0], self.b
        )
        self.assertFalse(dst.parent.exists())

    def test_closed_store_files(self):
        self.assertRefused(
            lambda: ShardedPool.closed_store_files(self.b), self.b_records[0], self.b
        )

    def test_delete_store(self):
        self.assertRefused(
            lambda: ShardedPool.delete_store(self.b), self.b_records[0], self.b
        )

    def test_delete_store_resuming(self):
        self.assertRefused(
            lambda: ShardedPool.delete_store(self.b, resume=True),
            self.b_records[0],
            self.b,
        )

    def test_closed_store_files_resuming(self):
        self.assertRefused(
            lambda: ShardedPool.closed_store_files(self.b, resume=True),
            self.b_records[0],
            self.b,
        )

    def test_the_read_only_reader(self):
        def open_it():
            with open_read_only(self.b):
                self.fail("the store was opened")

        self.assertRefused(open_it, self.b_records[0], self.b)

    def test_the_store_it_names_is_not_refused(self):
        """The control: A, whose records are bare names, is read, listed and unchanged, so the
        refusals above are the record's and not a defect of the fixtures."""
        before = tree_state(self.root)

        pool = read_pool(self.a)
        pool._check_shard_files()

        self.assertEqual(pool._shard_db_files, self.shard_paths(self.a))
        self.assertEqual(
            ShardedPool.closed_store_files(self.a),
            [*self.shard_paths(self.a).values(), self.a],
        )
        self.assertEqual(tree_state(self.root), before)


if __name__ == "__main__":
    unittest.main()
