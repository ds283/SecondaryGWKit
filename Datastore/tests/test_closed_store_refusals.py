"""
Each refusal of a closed-store operation states its prefix **once**.
`prompts/datastore-generic-followup` prompt 03, closing `datastore-generic`'s
`[04-closed-store-refusals-repeat-their-prefix]`.

``ShardedPool._read_closed_store`` raises the bare reason, and takes no verb. Each of its four
callers states the operation: ``closed_store_files`` and ``delete_store`` ("Cannot delete"),
``copy_store`` and ``move_store`` ("Cannot copy"/"Cannot move", naming the destination) and
``store_reader.open_read_only`` ("Cannot read", ending "Nothing was read or repaired"). Before
this, the first three said "Cannot <verb> sharded datastore ..." twice.

"Once" is tested as: the message begins with the operation's prefix and holds the words
``sharded datastore`` exactly once. No reason text holds them, so a second prefix of any verb is
caught.

Three broken stores: a shard record ``../outside.sqlite`` (the resolver's refusal), a dropped
``shards`` table, and a missing shard file (the file-problem refusal). No Ray, no datastore,
nothing under ``var/``.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from Datastore.SQL.ShardedPool import ShardedPool
from Datastore.shard_paths import shard_file_name
from Datastore.store_reader import open_read_only
from Datastore.tests.shard_store_fixtures import (
    write_hand_built_primary,
    write_new_store,
    write_placeholder,
)
from config.datastore import factories as _factories

WORDS = "sharded datastore"
UNREADABLE = "its shards table could not be read"


class _Fixtures(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def unusable_record(self) -> Path:
        """A primary whose shard #0 is recorded as ``../outside.sqlite``."""
        primary = self.root / "U" / "store.sqlite"
        write_hand_built_primary(primary, {0: "../outside.sqlite"})
        write_placeholder(self.root / "outside.sqlite")
        return primary

    def no_shards_table(self) -> Path:
        primary = self.root / "N" / "store.sqlite"
        write_new_store(primary, shards=3)
        conn = sqlite3.connect(primary)
        try:
            with conn:
                conn.execute("DROP TABLE shards")
        finally:
            conn.close()
        return primary

    def missing_shard(self) -> Path:
        """Returns the primary; shard #1's file is gone."""
        primary = self.root / "M" / "store.sqlite"
        write_new_store(primary, shards=3)
        (primary.parent / shard_file_name(primary, 1)).unlink()
        return primary

    @staticmethod
    def message_of(call) -> str:
        with unittest.TestCase().assertRaises(RuntimeError) as cm:
            call()
        return str(cm.exception)

    @staticmethod
    def read_it(primary: Path) -> None:
        with open_read_only(primary, _factories):
            pass

    def assert_once(self, message: str, prefix: str):
        self.assertTrue(message.startswith(prefix), message)
        self.assertEqual(message.count(WORDS), 1, message)


class TestEachPrefixOnce(_Fixtures):
    def test_delete_and_closed_store_files_state_their_prefix_once(self):
        primary = self.unusable_record()
        prefix = f'Cannot delete {WORDS} "{primary}"'
        for call in (
            lambda: ShardedPool.delete_store(primary),
            lambda: ShardedPool.closed_store_files(primary),
        ):
            message = self.message_of(call)
            self.assert_once(message, prefix)
            self.assertIn("unusable record", message)
            self.assertIn("../outside.sqlite", message)

    def test_copy_and_move_state_their_prefix_once(self):
        primary = self.unusable_record()
        dst = self.root / "D" / "copy.sqlite"
        for verb, operation in (
            ("copy", ShardedPool.copy_store),
            ("move", ShardedPool.move_store),
        ):
            with self.subTest(verb=verb):
                message = self.message_of(lambda: operation(primary, dst))
                self.assert_once(
                    message, f'Cannot {verb} {WORDS} "{primary}" to "{dst}": '
                )
                self.assertIn("unusable record", message)
                self.assertFalse(dst.exists())

    def test_the_reader_states_its_prefix_once(self):
        primary = self.unusable_record()
        message = self.message_of(lambda: self.read_it(primary))
        self.assert_once(message, f'Cannot read {WORDS} "{primary}"')
        self.assertIn("unusable record", message)
        self.assertTrue(message.endswith("Nothing was read or repaired"), message)

    def test_an_unreadable_shards_table_is_named_once_by_each(self):
        primary = self.no_shards_table()
        dst = self.root / "D" / "copy.sqlite"
        for verb, prefix, call in (
            (
                "delete_store",
                f'Cannot delete {WORDS} "{primary}"',
                lambda: ShardedPool.delete_store(primary),
            ),
            (
                "copy_store",
                f'Cannot copy {WORDS} "{primary}" to "{dst}"',
                lambda: ShardedPool.copy_store(primary, dst),
            ),
            (
                "open_read_only",
                f'Cannot read {WORDS} "{primary}"',
                lambda: self.read_it(primary),
            ),
        ):
            with self.subTest(operation=verb):
                message = self.message_of(call)
                self.assert_once(message, prefix)
                self.assertIn(UNREADABLE, message)
                self.assertEqual(message.count(UNREADABLE), 1, message)


class TestTheBareReason(_Fixtures):
    def test_read_closed_store_states_no_prefix(self):
        missing = self.missing_shard()
        for name, primary, fragment in (
            ("unusable record", self.unusable_record(), "unusable record"),
            ("dropped shards table", self.no_shards_table(), UNREADABLE),
            ("missing shard file", missing, "does not exist"),
        ):
            with self.subTest(fixture=name):
                message = self.message_of(
                    lambda: ShardedPool._read_closed_store(primary)
                )
                self.assertFalse(message.startswith("Cannot"), message)
                self.assertNotIn(WORDS, message)
                self.assertIn(fragment, message)


if __name__ == "__main__":
    unittest.main()
