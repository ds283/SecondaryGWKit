"""
Deleting a closed store's own files through ``ShardedPool.closed_store_files`` / ``delete_store``.
`prompts/store-retirement` prompt 01 §3 tests 1-6.

A store is its primary and the shards its primary's ``shards`` table names, read through the one
resolver (``_read_closed_store``). The case that matters is a primary whose legacy absolute
records name **another populated store's** shards, byte for byte, as the backup of the A3 store
does (``docs/store-retirement-audit.md`` §2.7): deleting it must remove its own siblings and leave
the other store untouched. The shards go first, in ascending serial, and the primary last, so an
interrupted deletion leaves a primary and a subset of its shards, which the constructor refuses to
open and ``delete_store(primary, resume=True)`` completes (the interruption table in
`logs/01-delete-a-closed-store.md`).

Stores are built with ``shard_store_fixtures`` in temporary directories. Every deletion in this
module happens inside a temporary directory the test itself built. Nothing under ``var/`` is
opened. The constructor is never called; its check is exercised through ``read_pool`` +
``_check_shard_files``. No Ray, no datastore.
"""

import contextlib
import errno
import io
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Tuple
from unittest import mock

import ray

from Datastore.SQL.ShardedPool import ShardedPool
from Datastore.shard_paths import shard_file_name
from Datastore.tests.shard_store_fixtures import (
    marker,
    read_pool,
    stored_records,
    tree_state,
    write_legacy_primary,
    write_new_store,
    write_placeholder,
)

N_SHARDS = 4
JOURNAL_SUFFIXES = ("-journal", "-wal", "-shm")

# the two public methods, by name, for the refusal tests: each refusal is made by both
METHODS = {
    "closed_store_files": ShardedPool.closed_store_files,
    "delete_store": ShardedPool.delete_store,
}


def quiet(fn, *args, **kwargs):
    """Call ``fn`` with stdout captured: reading a legacy primary prints a '!!' notice."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def present_section(message: str) -> str:
    """The bracketed list after 'Store files still present: ' in a failure message."""
    head = "Store files still present: ["
    return message.split(head, 1)[1].split("]", 1)[0]


def quoted(paths) -> str:
    return ", ".join(f'"{str(p)}"' for p in paths)


class _StoreCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # resolved: on macOS the temporary directory is reached through /var -> /private/var
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def fresh(self):
        self.tearDown()
        self.setUp()

    # --- building stores

    def new_store(self, directory: str = "B", stem: str = "store") -> Path:
        primary = self.root / directory / f"{stem}.sqlite"
        write_new_store(primary, shards=N_SHARDS)
        return primary

    def legacy_pair(self) -> Tuple[Path, Path]:
        """Directory A holds a complete, populated store whose records are absolute paths to its
        own shards (the form every real primary holds). Directory B holds a store under the
        **same file names**, whose primary records its shards by absolute paths **into A**, with
        its own shards beside it. Returns (A's primary, B's primary)."""
        a = self.root / "A"
        names = [shard_file_name("store.sqlite", i) for i in range(N_SHARDS)]
        into_a = {i: str(a / n) for i, n in enumerate(names)}
        write_legacy_primary(a / "store.sqlite", into_a, shard_keys=[(7, 2)])
        for n in names:
            write_placeholder(a / n)

        b_primary = self.root / "B" / "store.sqlite"
        write_legacy_primary(b_primary, into_a, shard_keys=[(7, 2)])
        for n in names:
            write_placeholder(b_primary.parent / n)
        return a / "store.sqlite", b_primary

    def store(self, shape: str) -> Tuple[Path, Path]:
        """(primary to delete, other populated directory or None) for a store shape."""
        if shape == "new":
            return self.new_store(), None
        a_primary, b_primary = self.legacy_pair()
        return b_primary, a_primary.parent

    # --- expectations

    @staticmethod
    def shard_paths(primary: Path) -> List[Path]:
        return [primary.parent / shard_file_name(primary, i) for i in range(N_SHARDS)]

    def store_files(self, primary: Path) -> List[Path]:
        """What a deletion of ``primary`` must name: its shards in ascending serial, then it."""
        return self.shard_paths(primary) + [primary]

    def rel(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    def assertOpensAgainst(self, primary: Path, expected: List[Path]):
        pool = quiet(read_pool, primary)
        pool._check_shard_files()
        self.assertEqual(
            pool._shard_db_files, {i: p for i, p in enumerate(expected)}, str(primary)
        )

    def assertConstructorRefuses(self, primary: Path):
        """``primary`` exists, and the constructor's read-and-check refuses it."""
        self.assertTrue(primary.is_file())
        with self.assertRaises(RuntimeError) as ctx:
            quiet(read_pool, primary)._check_shard_files()
        self.assertIn("Cannot open sharded datastore", str(ctx.exception))

    def assertCleanlyDeleted(self, primary: Path, before: Dict, other: Path = None):
        """The tree is what a clean delete of ``primary`` leaves: the store's files gone, and
        every other entry under the root as it was in ``before``, the store's own directory's
        mtime aside. ``other`` (the legacy shape's A) is untouched, its own mtime included.
        """
        after = tree_state(self.root)
        gone = {self.rel(p) for p in self.store_files(primary)}
        own_dir = self.rel(primary.parent)
        self.assertEqual(set(after), set(before) - gone)
        for key in after:
            if key != own_dir:
                self.assertEqual(after[key], before[key], key)
        if other is not None:
            for key in before:
                if key == self.rel(other) or key.startswith(self.rel(other) + os.sep):
                    self.assertEqual(after.get(key), before[key], key)


class TestNewStyleStore(_StoreCase):
    """Test 1: a store whose records are bare names."""

    def test_lists_its_shards_in_ascending_serial_then_its_primary(self):
        primary = self.new_store()
        before = tree_state(self.root)

        listed = ShardedPool.closed_store_files(primary)

        self.assertEqual(listed, self.store_files(primary))
        self.assertEqual(
            [p.name for p in listed],
            [
                "store-shard0000.sqlite",
                "store-shard0001.sqlite",
                "store-shard0002.sqlite",
                "store-shard0003.sqlite",
                "store.sqlite",
            ],
        )
        for path in listed:
            self.assertTrue(path.is_absolute())
        # listing never writes
        self.assertEqual(tree_state(self.root), before)

    def test_deletes_exactly_its_files_and_nothing_beside_them(self):
        primary = self.new_store()
        b = primary.parent
        manifest = b / "store.manifest.json"
        manifest.write_text('{"datastore": "%s"}\n' % primary)
        notes = b / "store-notes.txt"
        notes.write_text("not part of the store\n")
        other = self.new_store(stem="other")
        beside = [manifest, notes, *self.store_files(other)]
        # a directory mtime in the past, so that the change a deletion makes is always seen
        os.utime(b, ns=(10**18, 10**18))
        before = tree_state(self.root)

        listed = ShardedPool.closed_store_files(primary)
        deleted = ShardedPool.delete_store(primary)

        self.assertEqual(deleted, listed)
        self.assertEqual(deleted, self.store_files(primary))
        for path in deleted:
            self.assertFalse(os.path.lexists(path), str(path))
        after = tree_state(self.root)
        # the entries beside the store are untouched, by tree_state restricted to them
        self.assertEqual(
            {self.rel(p): after.get(self.rel(p)) for p in beside},
            {self.rel(p): before[self.rel(p)] for p in beside},
        )
        self.assertEqual(
            sorted(p.name for p in b.iterdir()), sorted(p.name for p in beside)
        )
        # the directory's own mtime changes, as any deletion must change it
        self.assertNotEqual(after["B"], before["B"])
        self.assertCleanlyDeleted(primary, before)
        # the other store in the same directory still opens against its own files
        self.assertOpensAgainst(other, self.shard_paths(other))


class TestLegacyRecordsNamingAnotherStore(_StoreCase):
    """Test 2: B's records are absolute paths into A, where a complete store of the same file
    names exists. Only B's five files go; A is byte-identical, mtimes included."""

    # a failure prints every path, so that it says which store's files were deleted
    maxDiff = None

    def test_every_listed_file_is_in_b(self):
        a_primary, b_primary = self.legacy_pair()
        a = a_primary.parent
        # the precondition that makes this the dangerous case: B's records name A's files, which
        # exist
        for stored in stored_records(b_primary).values():
            self.assertTrue(stored.startswith(str(a) + os.sep), stored)
            self.assertTrue(Path(stored).is_file(), stored)
        before = tree_state(self.root)

        listed = quiet(ShardedPool.closed_store_files, b_primary)

        self.assertEqual(listed, self.store_files(b_primary))
        for path in listed:
            self.assertEqual(path.parent, b_primary.parent, str(path))
        self.assertEqual(tree_state(self.root), before)

    def test_delete_removes_only_b_files_and_a_is_untouched(self):
        a_primary, b_primary = self.legacy_pair()
        a = a_primary.parent
        a_state = tree_state(a)
        before = tree_state(self.root)

        deleted = quiet(ShardedPool.delete_store, b_primary)

        # A's files first, one by one, so that a failure names every one of them that was
        # deleted, and B's files that were left
        a_files_gone = [
            str(p) for p in self.store_files(a_primary) if not os.path.lexists(p)
        ]
        b_files_left = [
            str(p) for p in self.store_files(b_primary) if os.path.lexists(p)
        ]
        self.assertEqual(
            {"A's files deleted": a_files_gone, "B's files left": b_files_left},
            {"A's files deleted": [], "B's files left": []},
        )
        self.assertEqual(deleted, self.store_files(b_primary))
        # every file in A is byte-identical, with an unchanged mtime, and so is A itself
        self.assertEqual(tree_state(a), a_state)
        self.assertEqual(tree_state(self.root)["A"], before["A"])
        self.assertEqual(list(b_primary.parent.iterdir()), [])
        self.assertCleanlyDeleted(b_primary, before, other=a)
        # A still opens against its own files, which still hold A's markers
        self.assertOpensAgainst(a_primary, self.shard_paths(a_primary))
        for path in self.shard_paths(a_primary):
            self.assertIn(str(a), marker(path))


class TestRefusals(_StoreCase):
    """Test 3, and the refusals test 5 requires under resume: every refusal of R1, before
    anything is deleted. Each is made by both methods and, except the missing shard, under both
    values of ``resume``. Each asserts that the error names the offending file and ends 'Nothing
    was deleted', and that nothing under the temporary root was created, changed or removed.
    """

    def assertRefusal(self, primary, *fragments, resumes=(False, True)):
        for name, method in METHODS.items():
            for resume in resumes:
                with self.subTest(
                    method=name, resume=resume, expect=str(fragments[-1])
                ):
                    before = tree_state(self.root)
                    with self.assertRaises(RuntimeError) as ctx:
                        quiet(method, primary, resume=resume)
                    message = str(ctx.exception)
                    for fragment in fragments:
                        self.assertIn(str(fragment), message)
                    self.assertTrue(message.endswith("Nothing was deleted"), message)
                    self.assertEqual(tree_state(self.root), before)

    def test_primary_missing(self):
        self.new_store()
        missing = self.root / "B" / "nothing.sqlite"
        self.assertRefusal(
            missing, missing, "does not exist", "no file can be identified"
        )

    def test_primary_is_a_directory(self):
        directory = self.root / "D.sqlite"
        directory.mkdir()
        self.assertRefusal(directory, directory, "not a regular file")

    def test_primary_is_a_symbolic_link(self):
        target = self.new_store()
        link = self.root / "B" / "link.sqlite"
        link.symlink_to(target)
        self.assertRefusal(link, link, "symbolic link")
        self.assertOpensAgainst(target, self.shard_paths(target))

    def test_journal_beside_the_primary(self):
        primary = self.new_store()
        for suffix in JOURNAL_SUFFIXES:
            journal = primary.with_name(primary.name + suffix)
            journal.write_bytes(b"hot")
            try:
                self.assertRefusal(primary, primary, journal, "not closed cleanly")
            finally:
                journal.unlink()

    def test_journal_beside_a_shard(self):
        primary = self.new_store()
        shard2 = self.shard_paths(primary)[2]
        for suffix in JOURNAL_SUFFIXES:
            journal = shard2.with_name(shard2.name + suffix)
            journal.write_bytes(b"hot")
            try:
                self.assertRefusal(primary, shard2, journal, "#2", "not closed cleanly")
            finally:
                journal.unlink()

    def test_shards_table_cannot_be_read(self):
        not_a_database = self.root / "N" / "store.sqlite"
        write_placeholder(not_a_database)
        write_placeholder(not_a_database.parent / shard_file_name(not_a_database, 0))
        self.assertRefusal(not_a_database, not_a_database, "could not be read")

        no_table = self.root / "T" / "store.sqlite"
        no_table.parent.mkdir()
        conn = sqlite3.connect(no_table)
        with conn:
            conn.execute("CREATE TABLE other (x INTEGER)")
        conn.close()
        self.assertRefusal(no_table, no_table, "could not be read")

    def test_no_shard_recorded(self):
        empty = self.root / "E" / "store.sqlite"
        write_legacy_primary(empty, {})
        self.assertRefusal(empty, empty, "records no shards")

    def test_unusable_record(self):
        # each names a file that exists, so a deletion that followed it would find something
        write_placeholder(self.root / "outside.sqlite")
        write_placeholder(self.root / "U" / "sub" / "store-shard0000.sqlite")
        for stored in ("../outside.sqlite", "sub/store-shard0000.sqlite", "", ".."):
            with self.subTest(stored=stored):
                primary = self.root / "U" / "store.sqlite"
                if primary.exists():
                    primary.unlink()
                write_legacy_primary(primary, {0: stored})
                self.assertRefusal(primary, primary, "#0", "unusable record")

    def test_record_resolves_to_a_symbolic_link(self):
        primary = self.new_store()
        shard1 = self.shard_paths(primary)[1]
        target = self.root / "elsewhere" / "real.sqlite"
        write_placeholder(target)
        target_bytes = target.read_bytes()
        shard1.unlink()
        shard1.symlink_to(target)
        self.assertRefusal(primary, shard1, "#1", "symbolic link")
        # the target survives, unchanged
        self.assertEqual(target.read_bytes(), target_bytes)

    def test_record_resolves_to_a_non_regular_file(self):
        primary = self.new_store()
        shard1 = self.shard_paths(primary)[1]
        shard1.unlink()
        shard1.mkdir()
        self.assertRefusal(primary, shard1, "#1", "not a regular file")

    def test_two_serials_share_a_file(self):
        shared = self.root / "S" / "store.sqlite"
        write_legacy_primary(
            shared,
            {
                0: str(self.root / "P" / "store-shard0000.sqlite"),
                1: str(self.root / "Q" / "store-shard0000.sqlite"),
            },
        )
        write_placeholder(shared.parent / "store-shard0000.sqlite")
        self.assertRefusal(
            shared, shared.parent / "store-shard0000.sqlite", "both resolve"
        )
        # the same when the shared file is missing, under resume too
        (shared.parent / "store-shard0000.sqlite").unlink()
        self.assertRefusal(
            shared, shared.parent / "store-shard0000.sqlite", "both resolve"
        )

    def test_missing_shard(self):
        primary = self.new_store()
        shard1 = self.shard_paths(primary)[1]
        shard1.unlink()
        self.assertRefusal(primary, shard1, "#1", "does not exist", resumes=(False,))

    def test_file_outside_the_primary_directory(self):
        """The resolver never returns a path outside the primary's directory. Stand in for a
        later change to it that did, by following a legacy absolute record as a path: the
        separate directory assertion refuses, naming the file and its directory, and A is
        untouched."""
        a_primary, b_primary = self.legacy_pair()
        a = a_primary.parent

        def follows_the_record(primary, stored):
            return (
                Path(stored) if Path(stored).is_absolute() else primary.parent / stored
            )

        with mock.patch(
            "Datastore.SQL.ShardedPool.resolve_shard_path", follows_the_record
        ):
            self.assertRefusal(
                b_primary,
                self.shard_paths(a_primary)[0],
                f'in the directory "{a}"',
                "not the primary's own directory",
            )
        self.assertOpensAgainst(a_primary, self.shard_paths(a_primary))
        self.assertOpensAgainst(b_primary, self.shard_paths(b_primary))


def _fail_on_unlink(n: int, after: bool = False):
    """A stand-in for os.unlink that raises on its n-th call, as if the process died there, and
    otherwise unlinks. With ``after``, the failing call unlinks first and then raises, as if the
    process died just after the unlink returned."""
    real = os.unlink
    calls = {"n": 0}

    def fake(path, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == n:
            if after:
                real(path, *args, **kwargs)
            raise OSError(errno.EIO, "injected failure", str(path))
        return real(path, *args, **kwargs)

    return fake


class TestInterruption(_StoreCase):
    """Test 4: the n-th os.unlink fails, for each unlink of each store shape, both before and
    after it takes effect. The state left behind is the interruption table's row: a primary and
    a subset of its shards, or everything, or nothing. The constructor's check refuses every
    partial state, delete_store refuses it, and delete_store(resume=True) completes it.
    """

    SHAPES = ("new", "legacy")

    def test_every_point_of_interruption(self):
        for shape in self.SHAPES:
            for n in range(1, N_SHARDS + 2):
                for after in (False, True):
                    with self.subTest(shape=shape, unlink=n, after=after):
                        self.fresh()
                        self._interrupt(shape, n, after)

    def _interrupt(self, shape: str, n: int, after: bool):
        primary, other = self.store(shape)
        files = self.store_files(primary)
        before = tree_state(self.root)

        with mock.patch("os.unlink", _fail_on_unlink(n, after)):
            with self.assertRaises(RuntimeError) as ctx:
                quiet(ShardedPool.delete_store, primary)
        message = str(ctx.exception)
        done = n if after else n - 1  # unlinks that took effect
        present, gone = files[done:], files[:done]

        # the error names the step and the files still present, and the remedy
        step = "delete the primary" if n == N_SHARDS + 1 else f"delete shard #{n - 1}"
        self.assertIn(f'failed at step "{step}"', message)
        self.assertEqual(present_section(message), quoted(present))
        self.assertIn("injected failure", message)
        self.assertIn(f'delete_store("{primary}", resume=True)', message)

        # the file state is the table's row, and nothing else changed
        for path in files:
            self.assertEqual(os.path.lexists(path), path in present, str(path))
        self._assertOthersUntouched(primary, before, other)

        if done == 0:
            # row D0: nothing was deleted. The store is whole and opens
            self.assertOpensAgainst(primary, self.shard_paths(primary))
        elif done <= N_SHARDS:
            # rows D1-D4: the primary and some, or none, of its shards
            self.assertConstructorRefuses(primary)
            state = tree_state(self.root)
            with self.assertRaises(RuntimeError) as ctx:
                quiet(ShardedPool.delete_store, primary)
            refusal = str(ctx.exception)
            for path in gone:
                self.assertIn(f'"{path}"', refusal)
            self.assertIn("does not exist", refusal)
            self.assertTrue(refusal.endswith("Nothing was deleted"))
            self.assertEqual(tree_state(self.root), state)
        else:
            # row D5: everything is gone. There is no primary, so nothing can be identified
            for resume in (False, True):
                state = tree_state(self.root)
                with self.assertRaises(RuntimeError) as ctx:
                    quiet(ShardedPool.delete_store, primary, resume=resume)
                self.assertIn("no file can be identified", str(ctx.exception))
                self.assertEqual(tree_state(self.root), state)
            self.assertCleanlyDeleted(primary, before, other)
            return

        listed = quiet(ShardedPool.closed_store_files, primary, resume=True)
        completed = quiet(ShardedPool.delete_store, primary, resume=True)
        self.assertEqual(completed, present)
        self.assertEqual(listed, present)
        self.assertCleanlyDeleted(primary, before, other)

    def _assertOthersUntouched(self, primary: Path, before: Dict, other: Path):
        after = tree_state(self.root)
        own = {self.rel(p) for p in self.store_files(primary)} | {
            self.rel(primary.parent)
        }
        for key, value in before.items():
            if key not in own:
                self.assertEqual(after.get(key), value, key)
        self.assertEqual(set(after) - set(before), set())

    def test_the_legacy_shape_other_directory_is_untouched_throughout(self):
        """A, whose files B's records name, is identical after every interrupted deletion of B,
        after the refusal of the plain retry, and after the resumed completion."""
        for n in range(1, N_SHARDS + 2):
            with self.subTest(unlink=n):
                self.fresh()
                primary, other = self.store("legacy")
                a_state = tree_state(other)
                with mock.patch("os.unlink", _fail_on_unlink(n)):
                    with self.assertRaises(RuntimeError):
                        quiet(ShardedPool.delete_store, primary)
                self.assertEqual(tree_state(other), a_state)
                if n > 1:
                    with self.assertRaises(RuntimeError):
                        quiet(ShardedPool.delete_store, primary)
                    self.assertEqual(tree_state(other), a_state)
                quiet(ShardedPool.delete_store, primary, resume=True)
                self.assertEqual(tree_state(other), a_state)
                self.assertOpensAgainst(
                    other / "store.sqlite", self.shard_paths(other / "store.sqlite")
                )


class TestResumeRelaxesOneThingOnly(_StoreCase):
    """Test 5: under resume=True only a missing shard is tolerated. A journal, a symbolic link and
    a missing primary are still refused by both methods, with the tree unchanged, even when a
    shard is also missing (so that resume has something to relax). The duplicate, the unreadable
    table, the unusable record and the file outside the directory are refused under resume by
    TestRefusals, which makes every refusal under both values."""

    def assertRefusedUnderResume(self, primary, *fragments):
        for name, method in METHODS.items():
            with self.subTest(method=name, expect=str(fragments[-1])):
                before = tree_state(self.root)
                with self.assertRaises(RuntimeError) as ctx:
                    quiet(method, primary, resume=True)
                message = str(ctx.exception)
                for fragment in fragments:
                    self.assertIn(str(fragment), message)
                self.assertTrue(message.endswith("Nothing was deleted"))
                self.assertEqual(tree_state(self.root), before)

    def test_a_journal_is_still_refused(self):
        primary = self.new_store()
        shards = self.shard_paths(primary)
        shards[0].unlink()
        for owner in (primary, shards[2], shards[0]):
            for suffix in JOURNAL_SUFFIXES:
                journal = owner.with_name(owner.name + suffix)
                journal.write_bytes(b"hot")
                try:
                    self.assertRefusedUnderResume(
                        primary, owner, journal, "not closed cleanly"
                    )
                finally:
                    journal.unlink()

    def test_a_symbolic_link_is_still_refused(self):
        primary = self.new_store()
        shards = self.shard_paths(primary)
        shards[0].unlink()
        target = self.root / "elsewhere" / "real.sqlite"
        write_placeholder(target)
        shards[2].unlink()
        shards[2].symlink_to(target)
        self.assertRefusedUnderResume(primary, shards[2], "#2", "symbolic link")
        self.assertTrue(target.is_file())
        # a dangling link is an entry, not a missing shard
        target.unlink()
        self.assertRefusedUnderResume(primary, shards[2], "#2", "symbolic link")

    def test_a_missing_primary_is_still_refused(self):
        primary = self.new_store()
        primary.unlink()
        self.assertRefusedUnderResume(
            primary, primary, "does not exist", "no file can be identified"
        )
        for path in self.shard_paths(primary):
            self.assertTrue(path.is_file(), str(path))

    def test_the_resumed_list_is_what_is_deleted(self):
        primary = self.new_store()
        shards = self.shard_paths(primary)
        shards[1].unlink()
        shards[3].unlink()
        before = tree_state(self.root)

        listed = ShardedPool.closed_store_files(primary, resume=True)
        self.assertEqual(listed, [shards[0], shards[2], primary])
        self.assertEqual(tree_state(self.root), before)
        with self.assertRaises(RuntimeError):
            ShardedPool.closed_store_files(primary)

        deleted = ShardedPool.delete_store(primary, resume=True)
        self.assertEqual(deleted, listed)
        for path in self.store_files(primary):
            self.assertFalse(os.path.lexists(path), str(path))

    def test_resume_on_a_whole_store_deletes_all_of_it(self):
        primary = self.new_store()
        self.assertEqual(
            ShardedPool.closed_store_files(primary, resume=True),
            ShardedPool.closed_store_files(primary),
        )
        self.assertEqual(
            ShardedPool.delete_store(primary, resume=True), self.store_files(primary)
        )


class TestRecheckBeforeEachUnlink(_StoreCase):
    """R1: a file that changed between the plan and its unlink is a refusal at that step, not a
    silent skip and not a deletion of whatever is there now. The change is made just after the
    real plan returns."""

    def _delete_with_change(self, primary: Path, change):
        real = ShardedPool._plan_deletion

        def plan_then_change(p, resume):
            plan = real(p, resume)
            change()
            return plan

        with mock.patch.object(ShardedPool, "_plan_deletion", plan_then_change):
            with self.assertRaises(RuntimeError) as ctx:
                ShardedPool.delete_store(primary)
        return str(ctx.exception)

    def test_a_shard_that_became_a_symbolic_link(self):
        primary = self.new_store()
        shards = self.shard_paths(primary)
        target = self.root / "elsewhere" / "real.sqlite"
        write_placeholder(target)

        def change():
            shards[2].unlink()
            shards[2].symlink_to(target)

        message = self._delete_with_change(primary, change)
        self.assertIn('failed at step "delete shard #2"', message)
        self.assertIn("symbolic link", message)
        self.assertIn("when the deletion was planned", message)
        self.assertEqual(
            present_section(message), quoted([shards[2], shards[3], primary])
        )
        self.assertTrue(shards[2].is_symlink())
        self.assertTrue(target.is_file())

    def test_a_shard_that_became_a_directory(self):
        primary = self.new_store()
        shards = self.shard_paths(primary)

        def change():
            shards[1].unlink()
            shards[1].mkdir()

        message = self._delete_with_change(primary, change)
        self.assertIn('failed at step "delete shard #1"', message)
        self.assertIn("not a regular file", message)
        self.assertIn("when the deletion was planned", message)
        self.assertTrue(shards[1].is_dir())

    def test_a_shard_that_vanished(self):
        primary = self.new_store()
        shards = self.shard_paths(primary)

        message = self._delete_with_change(primary, shards[3].unlink)
        self.assertIn('failed at step "delete shard #3"', message)
        self.assertIn("does not exist", message)
        self.assertIn("when the deletion was planned", message)
        self.assertTrue(primary.is_file())

    def test_a_primary_that_became_a_symbolic_link(self):
        primary = self.new_store()
        target = self.root / "elsewhere" / "real.sqlite"
        write_placeholder(target)

        def change():
            primary.unlink()
            primary.symlink_to(target)

        message = self._delete_with_change(primary, change)
        self.assertIn('failed at step "delete the primary"', message)
        self.assertIn("symbolic link", message)
        self.assertTrue(primary.is_symlink())
        self.assertTrue(target.is_file())


class TestNoRay(_StoreCase):
    """Test 6: the module imports ShardedPool, so ray is imported. It is never initialised, by
    this test's own listing, refusal and deletion, or by any other test here (tearDownModule).
    """

    def test_ray_is_imported_and_never_initialised(self):
        primary = self.new_store()
        ShardedPool.closed_store_files(primary)
        with self.assertRaises(RuntimeError):
            ShardedPool.delete_store(self.root / "B" / "nothing.sqlite")
        ShardedPool.delete_store(primary)
        self.assertIn("ray", sys.modules)
        self.assertFalse(ray.is_initialized())


def tearDownModule():
    if ray.is_initialized():
        raise AssertionError("ray was initialised by a test of the store deletion")


if __name__ == "__main__":
    unittest.main()
