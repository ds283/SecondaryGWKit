"""
Where ``ShardedPool`` finds its shards, read from the primary's ``shards`` table, and what it does
when one is missing. `prompts/datastore-portability` prompt 01 §3, tests 2-6.

Until prompt 01, ``_write_shard_data`` recorded every shard by **absolute** path and
``_read_shard_data`` used that path as it stood, so:

* a **copied** store opened against the *original's* shards, read them, wrote to them and put a
  ``version`` row into each. That happened on 2026-09-23, when the first run of the atol sweep put
  54 rows into the A3 baseline store its own docstring promised to leave alone;
* a **moved** store opened with the stale paths, and the ``Datastore`` actor behind each one,
  finding no file there, created an empty database at the old location (measured by prompt 01's
  P0 on the unfixed tree). The pool opened, and everything in it looked uncomputed.

Now shards are recorded by bare file name, every record is read through the one resolver in
``Datastore/shard_paths.py``, a legacy absolute record is read as the sibling of that name and
**never** as the absolute path, and ``_check_shard_files`` refuses to go on if any resolved shard
is missing, before any actor exists.

Everything here runs on an instance made with ``object.__new__(ShardedPool)`` in a temporary
directory, through the real ``_create_engine`` / ``_write_shard_data`` / ``_read_shard_data`` /
``_check_shard_files``. The constructor, which starts Ray actors, is never called. No Ray, no
datastore, per `CLAUDE.md`.

This module deliberately does **not** import ``Datastore.shard_paths``, so that it can be run
against the unfixed ``ShardedPool`` for the deliberate-breakage record (prompt §6 item 2).
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from Datastore.tests.shard_store_fixtures import (
    marker,
    read_pool,
    sha256,
    stored_records,
    write_legacy_primary,
    write_new_store,
    write_placeholder,
)

SHARD_NAMES = [f"store-shard{i:04d}.sqlite" for i in range(3)]


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # resolved: on macOS the temporary directory is reached through the /var -> /private/var
        # symlink, and ShardedPool resolves the primary
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()


def _legacy_store(directory: Path, records_dir: Path, stem: str = "store") -> Path:
    """A primary in ``directory`` whose records are absolute paths into ``records_dir``, as the
    old code wrote them, with its shard placeholders in ``directory``."""
    primary = directory / f"{stem}.sqlite"
    names = [f"{stem}-shard{i:04d}.sqlite" for i in range(3)]
    write_legacy_primary(
        primary, {i: str(records_dir / n) for i, n in enumerate(names)}
    )
    for n in names:
        write_placeholder(directory / n)
    return primary


class TestRoundTrip(_TempDirCase):
    """Test 2: a new store records bare names, reads back absolute siblings, and still reads its
    own shards after its directory is moved."""

    def test_new_store_records_bare_names_and_survives_a_move(self):
        a = self.root / "A"
        primary = a / "store.sqlite"
        write_new_store(primary)

        # what is on disk is the bare file name -- prompt §6 item 3
        self.assertEqual(stored_records(primary), dict(enumerate(SHARD_NAMES)))

        pool = read_pool(primary)
        self.assertEqual(
            pool._shard_db_files, {i: a / n for i, n in enumerate(SHARD_NAMES)}
        )
        for path in pool._shard_db_files.values():
            self.assertTrue(path.is_absolute())

        # move the whole directory, primary and shards together
        b = self.root / "B"
        os.rename(a, b)
        self.assertFalse(a.exists())

        pool = read_pool(b / "store.sqlite")
        self.assertEqual(
            pool._shard_db_files, {i: b / n for i, n in enumerate(SHARD_NAMES)}
        )
        pool._check_shard_files()

    def test_renaming_the_primary_alone_keeps_its_shards(self):
        """Prompt §5: once shards are recorded by name, renaming the primary on its own works."""
        a = self.root / "A"
        write_new_store(a / "store.sqlite")
        os.rename(a / "store.sqlite", a / "renamed.sqlite")

        pool = read_pool(a / "renamed.sqlite")
        self.assertEqual(
            pool._shard_db_files, {i: a / n for i, n in enumerate(SHARD_NAMES)}
        )
        pool._check_shard_files()


class TestLegacyRecords(_TempDirCase):
    """Test 3: absolute records pointing into directory A, with the store actually in B, resolve
    to B. The table is built by hand, to stand in for rows the old code wrote."""

    def test_legacy_absolute_records_resolve_to_the_primarys_directory(self):
        a = self.root / "A"  # never created: the records name a directory that is gone
        b = self.root / "B"
        primary = _legacy_store(b, records_dir=a)

        pool = read_pool(primary)

        self.assertEqual(
            pool._shard_db_files, {i: b / n for i, n in enumerate(SHARD_NAMES)}
        )
        pool._check_shard_files()

        # one line per store, not per shard, naming where the records pointed
        notices = [l for l in pool.read_stdout.splitlines() if "legacy absolute" in l]
        self.assertEqual(len(notices), 1, pool.read_stdout)
        self.assertTrue(notices[0].startswith("!!"))
        self.assertIn(str(a), notices[0])
        self.assertIn(str(b), notices[0])

    def test_legacy_records_that_name_their_own_directory_are_silent(self):
        """The two real stores hold absolute records naming their own shards; reading them is
        unchanged and prints nothing."""
        b = self.root / "B"
        primary = _legacy_store(b, records_dir=b)

        pool = read_pool(primary)

        self.assertEqual(
            pool._shard_db_files, {i: b / n for i, n in enumerate(SHARD_NAMES)}
        )
        self.assertNotIn("legacy absolute", pool.read_stdout)


class TestCopiedStore(_TempDirCase):
    """Test 4: with the original A still present and populated, the store copied to B reads B's
    shards. This is the failure that put 54 rows into the A3 baseline store, and the one that makes
    the retained backup of that store unsafe to open in place."""

    def _original(self) -> Path:
        a = self.root / "A"
        return _legacy_store(a, records_dir=a)

    def test_copied_directory_reads_its_own_shards_not_the_originals(self):
        a_primary = self._original()
        a = a_primary.parent
        a_bytes = {p.name: p.read_bytes() for p in a.iterdir()}

        b = self.root / "B"
        shutil.copytree(a, b)
        for n in SHARD_NAMES:
            write_placeholder(b / n)  # give B's copies B's own marker

        # B's primary still names A's shards, which exist
        for stored in stored_records(b / "store.sqlite").values():
            self.assertTrue(stored.startswith(str(a)))
            self.assertTrue(Path(stored).exists())

        pool = read_pool(b / "store.sqlite")

        self.assertEqual(
            pool._shard_db_files, {i: b / n for i, n in enumerate(SHARD_NAMES)}
        )
        for path in pool._shard_db_files.values():
            self.assertIn(str(b), marker(path))
        pool._check_shard_files()

        # and nothing of A's was touched
        self.assertEqual({p.name: p.read_bytes() for p in a.iterdir()}, a_bytes)

    def test_copied_with_a_renamed_primary_reads_its_own_shards(self):
        """The copy in a new directory with only the primary renamed (the §4 demonstration's
        shape, and README §0.1's backup is the same case without the rename)."""
        a = self._original().parent
        b = self.root / "B"
        b.mkdir()
        shutil.copy2(a / "store.sqlite", b / "copy.sqlite")
        for n in SHARD_NAMES:
            write_placeholder(b / n)

        pool = read_pool(b / "copy.sqlite")
        self.assertEqual(
            pool._shard_db_files, {i: b / n for i, n in enumerate(SHARD_NAMES)}
        )
        pool._check_shard_files()


class TestFailClosed(_TempDirCase):
    """Test 5: with a shard file missing, the check raises and names it, rather than letting a
    Datastore actor create an empty database in its place."""

    def assertRefusal(self, pool, *fragments):
        with self.assertRaises(RuntimeError) as ctx:
            pool._check_shard_files()
        message = str(ctx.exception)
        self.assertIn(str(pool._primary_file), message)
        for fragment in fragments:
            self.assertIn(fragment, message)
        return message

    def test_missing_shard_of_a_new_store_is_refused_by_name(self):
        a = self.root / "A"
        write_new_store(a / "store.sqlite")
        (a / SHARD_NAMES[1]).unlink()

        pool = read_pool(a / "store.sqlite")
        message = self.assertRefusal(
            pool, "#1", f'"{SHARD_NAMES[1]}"', str(a / SHARD_NAMES[1]), "does not exist"
        )
        self.assertNotIn("#0", message)
        self.assertNotIn("#2", message)
        self.assertFalse((a / SHARD_NAMES[1]).exists())

    def test_missing_sibling_is_refused_even_when_the_absolute_record_exists(self):
        """No fallback: B's shard 2 is missing, A's exists at the recorded absolute path, and the
        pool must refuse rather than use A's."""
        a = self.root / "A"
        _legacy_store(a, records_dir=a)
        b = self.root / "B"
        shutil.copytree(a, b)
        (b / SHARD_NAMES[2]).unlink()
        self.assertTrue((a / SHARD_NAMES[2]).exists())

        pool = read_pool(b / "store.sqlite")
        self.assertEqual(pool._shard_db_files[2], b / SHARD_NAMES[2])
        self.assertRefusal(
            pool,
            "#2",
            str(a / SHARD_NAMES[2]),
            str(b / SHARD_NAMES[2]),
            "does not exist",
        )

    def test_moved_store_with_legacy_records_and_no_shards_is_refused(self):
        """The P0 case after the fix: a legacy store moved without its shards."""
        a = self.root / "A"
        primary = _legacy_store(a, records_dir=a)
        b = self.root / "B"
        b.mkdir()
        shutil.move(str(primary), str(b / "store.sqlite"))

        pool = read_pool(b / "store.sqlite")
        message = self.assertRefusal(pool, "#0", "#1", "#2")
        self.assertIn(str(b / SHARD_NAMES[0]), message)

    def test_symlinked_shard_is_refused(self):
        """A shard that is a symlink would put the file the actor opens outside the primary's
        directory; the creator never makes one."""
        a = self.root / "A"
        write_new_store(a / "store.sqlite")
        elsewhere = self.root / "elsewhere.sqlite"
        (a / SHARD_NAMES[0]).rename(elsewhere)
        (a / SHARD_NAMES[0]).symlink_to(elsewhere)

        pool = read_pool(a / "store.sqlite")
        self.assertRefusal(pool, "#0", "symbolic link")

    def test_two_records_resolving_to_one_file_are_refused(self):
        """Reading legacy records by name can collapse two distinct absolute paths onto one
        sibling; two actors on one file must be refused."""
        b = self.root / "B"
        primary = b / "store.sqlite"
        write_legacy_primary(
            primary,
            {
                0: str(self.root / "X" / SHARD_NAMES[0]),
                1: str(self.root / "Y" / SHARD_NAMES[0]),
            },
        )
        write_placeholder(b / SHARD_NAMES[0])

        pool = read_pool(primary)
        self.assertRefusal(pool, "#0", "#1", "both resolve")

    def test_record_that_is_not_a_bare_name_is_refused_on_read(self):
        b = self.root / "B"
        primary = b / "store.sqlite"
        write_legacy_primary(primary, {0: "../outside.sqlite"})
        write_placeholder(self.root / "outside.sqlite")

        with self.assertRaises(RuntimeError) as ctx:
            read_pool(primary)
        self.assertIn("#0", str(ctx.exception))
        self.assertIn("../outside.sqlite", str(ctx.exception))


class TestNoSideEffect(_TempDirCase):
    """Test 6: reading a store does not change its primary. Legacy rows are interpreted, not
    rewritten -- the retained A3 backup must stay byte-identical when read."""

    def test_reading_a_relocated_legacy_store_leaves_the_primary_unchanged(self):
        a = self.root / "A"
        b = self.root / "B"
        primary = _legacy_store(b, records_dir=a)
        before = sha256(primary)
        records_before = stored_records(primary)

        read_pool(primary)

        self.assertEqual(sha256(primary), before)
        self.assertEqual(stored_records(primary), records_before)

    def test_reading_a_new_store_leaves_the_primary_unchanged(self):
        a = self.root / "A"
        write_new_store(a / "store.sqlite")
        before = sha256(a / "store.sqlite")

        read_pool(a / "store.sqlite")

        self.assertEqual(sha256(a / "store.sqlite"), before)


if __name__ == "__main__":
    unittest.main()
