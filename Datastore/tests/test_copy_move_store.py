"""
Copying and moving a closed store through ``ShardedPool.copy_store`` / ``move_store``.
`prompts/datastore-portability` prompt 02 §2 P6 and §3 tests 2-7.

Since prompt 01 a store's ``shards`` table holds bare file names (or legacy absolute paths, read by
their file name), so moving a store's directory or renaming its primary alone already works.
Renaming its shards as well needs the new names written into the table. That is what these two
static methods do: they copy or rename the primary and its shards to a new stem and rewrite the
destination's ``shards`` rows, in an order chosen so that a process that dies after any step leaves
nothing that opens wrongly (the interruption table in `logs/02-copy-and-move-a-store.md`).

Stores are built with ``shard_store_fixtures`` in temporary directories: a primary with the real
five tables, and placeholder shard files holding a marker. Every "does this store open, and
against which files" question is asked through ``_read_shard_data`` + ``_check_shard_files`` on an
``object.__new__`` instance, the constructor's own read-and-check. The constructor is never
called. No Ray, no datastore.
"""

import errno
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Dict
from unittest import mock

from Datastore.SQL.ShardedPool import ShardedPool
from Datastore.shard_paths import shard_file_name
from Datastore.tests.shard_store_fixtures import (
    marker,
    read_pool,
    sha256,
    stored_records,
    tree_state,
    write_legacy_primary,
    write_new_store,
    write_placeholder,
)

N_SHARDS = 3

# the three layouts prompt §2 P6 asks for: (destination directory, destination stem)
LAYOUTS = {
    "same directory, new stem": ("SRC", "renamed"),
    "new directory, same stem": ("C", "store"),
    "new directory, new stem": ("C", "renamed"),
}


class _StoreCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # resolved: on macOS the temporary directory is reached through /var -> /private/var
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    # --- building sources

    def new_source(self, directory: str = "B", stem: str = "store") -> Path:
        primary = self.root / directory / f"{stem}.sqlite"
        write_new_store(primary, shards=N_SHARDS)
        return primary

    def legacy_source(self, directory: str = "B", stem: str = "store") -> Path:
        """A primary in ``directory`` whose records are absolute paths into ``X``, where another
        store of the same stem **exists and is populated**, with this store's own shards beside
        its primary. This is prompt 01 §4.1's case and the hand-made source of prompt 02 §4.
        """
        x = self.root / "X"
        names = [shard_file_name(f"{stem}.sqlite", i) for i in range(N_SHARDS)]
        write_legacy_primary(
            x / f"{stem}.sqlite", {i: str(x / n) for i, n in enumerate(names)}
        )
        for n in names:
            write_placeholder(x / n)

        primary = self.root / directory / f"{stem}.sqlite"
        write_legacy_primary(primary, {i: str(x / n) for i, n in enumerate(names)})
        for n in names:
            write_placeholder(primary.parent / n)
        return primary

    def source(self, kind: str, directory: str = "B") -> Path:
        return (
            self.new_source(directory)
            if kind == "new"
            else self.legacy_source(directory)
        )

    # --- expectations

    @staticmethod
    def shard_paths(primary: Path) -> Dict[int, Path]:
        return {
            i: primary.parent / shard_file_name(primary, i) for i in range(N_SHARDS)
        }

    @staticmethod
    def bare_names(primary: Path) -> Dict[int, str]:
        return {i: shard_file_name(primary, i) for i in range(N_SHARDS)}

    def assertOpensAgainst(self, primary: Path, expected: Dict[int, Path], contents):
        """The constructor's read-and-check accepts ``primary`` and resolves exactly ``expected``,
        and shard *i* holds what the source's shard *i* held."""
        pool = read_pool(primary)
        pool._check_shard_files()
        self.assertEqual(pool._shard_db_files, expected)
        for serial, path in expected.items():
            self.assertEqual(path.read_bytes(), contents[serial], f"shard #{serial}")

    def assertRefused(self, primary: Path):
        """``primary`` exists, and the constructor's read-and-check refuses it."""
        self.assertTrue(primary.is_file())
        with self.assertRaises(RuntimeError):
            read_pool(primary)._check_shard_files()

    def assertGuarded(self, dst: Path):
        """No destination primary, and shard 0 at its destination name: the constructor's
        new-store branch raises 'Primary database is missing, but shard ... already exists'
        (ShardedPool.__init__) for any shard count, because it checks shard 0 first."""
        self.assertFalse(os.path.lexists(dst))
        self.assertTrue(os.path.lexists(dst.parent / shard_file_name(dst, 0)))

    def assertAbsent(self, primary: Path):
        """No primary and no shard under this store's names: nothing to open."""
        for path in [primary, *self.shard_paths(primary).values()]:
            self.assertFalse(os.path.lexists(path), str(path))

    def contents(self, primary: Path) -> Dict[int, bytes]:
        return {
            serial: path.read_bytes()
            for serial, path in read_pool(primary)._shard_db_files.items()
        }


class TestCopy(_StoreCase):
    """Test 2: copy to a new directory under a new stem."""

    def test_copy_to_a_new_directory_and_stem_reads_its_own_files(self):
        src = self.new_source()
        src_state = tree_state(src.parent)
        contents = self.contents(src)
        dst = self.root / "C" / "copy.sqlite"

        result = ShardedPool.copy_store(src, dst)

        # the destination's rows are its own bare names, and it reads its own files
        self.assertEqual(stored_records(dst), self.bare_names(dst))
        self.assertEqual(
            stored_records(dst),
            {
                0: "copy-shard0000.sqlite",
                1: "copy-shard0001.sqlite",
                2: "copy-shard0002.sqlite",
            },
        )
        self.assertEqual(result, self.shard_paths(dst))
        self.assertOpensAgainst(dst, self.shard_paths(dst), contents)
        self.assertEqual(
            sorted(p.name for p in dst.parent.iterdir()),
            sorted(["copy.sqlite", *self.bare_names(dst).values()]),
        )

        # they are the destination's files, not the source's: change them and the source's are
        # untouched, and the destination reads the change
        for path in self.shard_paths(dst).values():
            write_placeholder(path)
        for path in read_pool(dst)._shard_db_files.values():
            self.assertIn(str(dst.parent), marker(path))

        # every file of the source is byte-identical, with an unchanged mtime
        self.assertEqual(tree_state(src.parent), src_state)
        self.assertOpensAgainst(src, self.shard_paths(src), contents)

    def test_no_other_table_is_changed(self):
        """Only shards.filename differs between the source primary and the destination's."""
        src = self.new_source()
        conn = sqlite3.connect(src)
        with conn:
            conn.execute("INSERT INTO shard_keys (key_serial, shard_id) VALUES (7, 2)")
        conn.close()
        dst = self.root / "C" / "copy.sqlite"

        ShardedPool.copy_store(src, dst)

        def dump(path):
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                return [
                    line
                    for line in conn.iterdump()
                    if not line.startswith('INSERT INTO "shards"')
                ]
            finally:
                conn.close()

        self.assertEqual(dump(dst), dump(src))


class TestCopyOfALegacySource(_StoreCase):
    """Test 3: absolute records naming a populated directory X, files in B, copied to C under a
    new stem. C reads C, its rows are bare, and X and B are unchanged. Prompt 01 §4.1's case.
    """

    def test_legacy_source_copied_to_a_new_stem_reads_its_own_files(self):
        src = self.legacy_source("B")
        x = self.root / "X"
        x_state, b_state = tree_state(x), tree_state(src.parent)
        contents = self.contents(src)
        # the source's records name X, where files of those names exist
        for stored in stored_records(src).values():
            self.assertTrue(stored.startswith(str(x)))
            self.assertTrue(Path(stored).exists())
        dst = self.root / "C" / "copy.sqlite"

        result = ShardedPool.copy_store(src, dst)

        self.assertEqual(stored_records(dst), self.bare_names(dst))
        self.assertEqual(result, self.shard_paths(dst))
        self.assertOpensAgainst(dst, self.shard_paths(dst), contents)
        for path in result.values():
            self.assertIn(str(src.parent), marker(path))  # B's shards, copied; not X's
            self.assertNotIn(str(x), marker(path))
        self.assertEqual(tree_state(x), x_state)
        self.assertEqual(tree_state(src.parent), b_state)


class TestMove(_StoreCase):
    """Test 4: move in each of the three layouts, for a new-style and a legacy source. The source
    names are gone and the destination reads its own files."""

    def _move(self, kind: str, layout: str):
        src = self.source(kind, "SRC")
        contents = self.contents(src)
        x_state = tree_state(self.root / "X") if kind == "legacy" else None
        directory, stem = LAYOUTS[layout]
        dst = self.root / directory / f"{stem}.sqlite"

        result = ShardedPool.move_store(src, dst)

        self.assertEqual(stored_records(dst), self.bare_names(dst))
        self.assertEqual(result, self.shard_paths(dst))
        self.assertOpensAgainst(dst, self.shard_paths(dst), contents)
        if dst.parent != src.parent:
            self.assertAbsent(src)
        else:
            self.assertFalse(os.path.lexists(src))
            for path in self.shard_paths(src).values():
                self.assertFalse(os.path.lexists(path))
        if kind == "legacy":
            # the populated directory the legacy records name is untouched
            self.assertEqual(tree_state(self.root / "X"), x_state)

    def test_same_directory_new_stem(self):
        self._move("new", "same directory, new stem")

    def test_new_directory_same_stem(self):
        self._move("new", "new directory, same stem")

    def test_new_directory_new_stem(self):
        self._move("new", "new directory, new stem")

    def test_legacy_source_in_each_layout(self):
        for layout in LAYOUTS:
            with self.subTest(layout=layout):
                self.tearDown()
                self.setUp()
                self._move("legacy", layout)


class TestRefusals(_StoreCase):
    """Test 5: every refusal of P6, before anything is written. Each asserts that the error names
    the offending file, and that nothing under the temporary root was created, changed or
    removed (a full listing with hashes and mtimes, before and after)."""

    def assertRefusal(self, mode: str, src, dst, *fragments):
        """One refusal, as a subtest, so that every case of a test is reported."""
        with self.subTest(mode=mode, src=str(src), dst=str(dst), expect=fragments[-1]):
            before = tree_state(self.root)
            operation = (
                ShardedPool.copy_store if mode == "copy" else ShardedPool.move_store
            )
            with self.assertRaises(RuntimeError) as ctx:
                operation(src, dst)
            message = str(ctx.exception)
            for fragment in fragments:
                self.assertIn(str(fragment), message)
            self.assertIn("Nothing was written", message)
            self.assertEqual(tree_state(self.root), before)

    @staticmethod
    def both():
        return ("copy", "move")

    def test_source_primary_missing_or_not_a_regular_file(self):
        self.new_source("B")
        (self.root / "D.sqlite").mkdir()
        os.symlink(self.root / "B" / "store.sqlite", self.root / "link.sqlite")
        dst = self.root / "C" / "copy.sqlite"
        for mode in self.both():
            missing = self.root / "B" / "nothing.sqlite"
            self.assertRefusal(mode, missing, dst, missing, "does not exist")
            directory = self.root / "D.sqlite"
            self.assertRefusal(mode, directory, dst, directory, "not a regular file")
            link = self.root / "link.sqlite"
            self.assertRefusal(mode, link, dst, link, "symbolic link")

    def test_source_shard_record_unusable(self):
        src = self.root / "B" / "store.sqlite"
        write_legacy_primary(src, {0: "../outside.sqlite"})
        write_placeholder(self.root / "outside.sqlite")
        for mode in self.both():
            self.assertRefusal(
                mode,
                src,
                self.root / "C" / "copy.sqlite",
                src,
                "#0",
                "../outside.sqlite",
            )

    def test_source_shard_missing_symlinked_irregular_or_shared(self):
        dst = self.root / "C" / "copy.sqlite"
        src = self.new_source("B")
        shard1 = src.parent / shard_file_name(src, 1)

        shard1.unlink()
        for mode in self.both():
            self.assertRefusal(mode, src, dst, shard1, "#1", "does not exist")

        elsewhere = self.root / "elsewhere.sqlite"
        write_placeholder(elsewhere)
        shard1.symlink_to(elsewhere)
        for mode in self.both():
            self.assertRefusal(mode, src, dst, shard1, "#1", "symbolic link")

        shard1.unlink()
        shard1.mkdir()
        for mode in self.both():
            self.assertRefusal(mode, src, dst, shard1, "#1", "not a regular file")

        shared = self.root / "S" / "store.sqlite"
        write_legacy_primary(
            shared,
            {
                0: str(self.root / "P" / "store-shard0000.sqlite"),
                1: str(self.root / "Q" / "store-shard0000.sqlite"),
            },
        )
        write_placeholder(shared.parent / "store-shard0000.sqlite")
        for mode in self.both():
            self.assertRefusal(
                mode,
                shared,
                dst,
                shared.parent / "store-shard0000.sqlite",
                "both resolve",
            )

    def test_source_journal_files(self):
        dst = self.root / "C" / "copy.sqlite"
        src = self.new_source("B")
        shard2 = src.parent / shard_file_name(src, 2)
        for owner in (src, shard2):
            for suffix in ("-journal", "-wal", "-shm"):
                journal = owner.with_name(owner.name + suffix)
                journal.write_bytes(b"hot")
                try:
                    for mode in self.both():
                        self.assertRefusal(
                            mode, src, dst, owner, journal, "not closed cleanly"
                        )
                finally:
                    journal.unlink()

    def test_destination_is_the_source(self):
        src = self.new_source("B")
        for mode in self.both():
            for dst in (src, self.root / "B" / ".." / "B" / "store.sqlite"):
                self.assertRefusal(mode, src, dst, src, "same file as the source")

    def test_destination_is_an_existing_directory(self):
        src = self.new_source("B")
        (self.root / "C").mkdir()
        for mode in self.both():
            self.assertRefusal(
                mode, src, self.root / "C", self.root / "C", "is an existing directory"
            )

    def test_destination_names_or_their_journals_exist(self):
        src = self.new_source("B")
        dst = self.root / "C" / "copy.sqlite"
        shard1 = dst.parent / shard_file_name(dst, 1)
        taken = [
            dst,
            shard1,
            dst.with_name(dst.name + "-journal"),
            dst.with_name(dst.name + "-wal"),
            shard1.with_name(shard1.name + "-shm"),
            (dst.parent / shard_file_name(dst, 0)).with_name(
                shard_file_name(dst, 0) + "-journal"
            ),
        ]
        for path in taken:
            write_placeholder(path)
            try:
                for mode in self.both():
                    self.assertRefusal(mode, src, dst, path, "already exist")
            finally:
                path.unlink()
        # a dangling symlink at a destination name is also taken
        shard1.symlink_to(self.root / "nowhere")
        try:
            for mode in self.both():
                self.assertRefusal(mode, src, dst, shard1, "already exist")
        finally:
            shard1.unlink()

    def test_copy_temporary_name_or_its_journal_exists(self):
        src = self.new_source("B")
        dst = self.root / "C" / "copy.sqlite"
        temp = dst.with_name(dst.name + ".incomplete-copy")
        for path in (temp, temp.with_name(temp.name + "-journal")):
            write_placeholder(path)
            try:
                self.assertRefusal("copy", src, dst, path, "already exist")
            finally:
                path.unlink()

    def test_move_into_a_directory_holding_a_source_shard_name(self):
        """A move renames the primary before rewriting its rows; until then the destination
        primary reads the source's records by name in its new directory. A file of that name
        there would be read as this store's shard, so the move is refused."""
        src = self.new_source("B")
        other = self.root / "C" / shard_file_name(src, 2)
        write_placeholder(other)
        self.assertRefusal("move", src, self.root / "C" / "renamed.sqlite", other, "#2")
        # a copy never exposes the source's records at the destination name, so it proceeds
        ShardedPool.copy_store(src, self.root / "C" / "renamed.sqlite")

    def test_source_with_no_shards_or_no_shard_zero(self):
        dst = self.root / "C" / "copy.sqlite"
        empty = self.root / "E" / "store.sqlite"
        write_legacy_primary(empty, {})
        no_zero = self.root / "Z" / "store.sqlite"
        write_legacy_primary(no_zero, {1: "store-shard0001.sqlite"})
        write_placeholder(no_zero.parent / "store-shard0001.sqlite")
        for mode in self.both():
            self.assertRefusal(mode, empty, dst, empty, "records no shards")
            self.assertRefusal(mode, no_zero, dst, no_zero, "no shard #0")


def _fail_on_call(n: int, real, partial: bool = False):
    """A stand-in for shutil.copy2 / os.rename / os.replace that raises on its n-th call (as if
    the process died there) and otherwise calls the real function. With ``partial``, the failing
    call first writes half of the file, as a copy cut off part-way would."""
    calls = {"n": 0}

    def fake(a, b, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == n:
            if partial:
                data = Path(a).read_bytes()
                Path(b).write_bytes(data[: len(data) // 2])
            raise OSError(errno.EIO, "injected failure", str(b))
        return real(a, b, *args, **kwargs)

    return fake


class TestInterruption(_StoreCase):
    """Test 6: for every step of each operation, the step after it fails, and the state left
    behind is what the log's interruption table says. Every store a later opener could find
    either opens against the right files, is refused by the read-and-check, or is a destination
    with shards and no primary (the constructor's guard). Each layout, for a new-style source
    and for a legacy source whose records name a populated other directory."""

    def _run(self, mode, kind, layout, fail):
        """Build a source, run ``mode`` with the failure ``fail`` injected, and return
        (src, dst, contents, error)."""
        src = self.source(kind, "SRC")
        contents = self.contents(src)
        directory, stem = LAYOUTS[layout]
        dst = self.root / directory / f"{stem}.sqlite"
        operation = ShardedPool.copy_store if mode == "copy" else ShardedPool.move_store
        with fail():
            with self.assertRaises(RuntimeError) as ctx:
                operation(src, dst)
        return src, dst, contents, str(ctx.exception)

    def _cases(self):
        for kind in ("new", "legacy"):
            for layout in LAYOUTS:
                yield kind, layout

    def _fresh(self):
        self.tearDown()
        self.setUp()

    def test_copy(self):
        real_copy2, real_replace = shutil.copy2, os.replace
        rewrite = lambda: mock.patch.object(
            ShardedPool,
            "_write_shard_records",
            side_effect=OSError(errno.EIO, "injected failure"),
        )
        # (row of the table, the failure, what the destination must be afterwards)
        points = [
            (
                "C1 dies before shard #0 is copied",
                lambda: mock.patch("shutil.copy2", _fail_on_call(1, real_copy2)),
                "absent",
            ),
            (
                "C2 dies copying shard #0, part-written",
                lambda: mock.patch(
                    "shutil.copy2", _fail_on_call(1, real_copy2, partial=True)
                ),
                "guard",
            ),
            (
                "C3 dies after shard #0",
                lambda: mock.patch("shutil.copy2", _fail_on_call(2, real_copy2)),
                "guard",
            ),
            (
                "C3 dies after shard #1",
                lambda: mock.patch("shutil.copy2", _fail_on_call(3, real_copy2)),
                "guard",
            ),
            (
                "C4 dies after every shard, before the primary",
                lambda: mock.patch("shutil.copy2", _fail_on_call(4, real_copy2)),
                "guard",
            ),
            (
                "C5 dies after the temporary primary, before the rewrite",
                rewrite,
                "guard",
            ),
            (
                "C6 dies after the rewrite, before the rename",
                lambda: mock.patch("os.replace", _fail_on_call(1, real_replace)),
                "guard",
            ),
        ]
        for kind, layout in self._cases():
            for row, fail, expected in points:
                with self.subTest(kind=kind, layout=layout, row=row):
                    self._fresh()
                    src, dst, contents, error = self._run("copy", kind, layout, fail)

                    # the source is never written: it opens against its own files
                    self.assertOpensAgainst(src, self.shard_paths(src), contents)
                    if expected == "absent":
                        self.assertAbsent(dst)
                    else:
                        self.assertGuarded(dst)
                    # the failure names the step and lists what exists, and deletes nothing
                    self.assertIn("failed at step", error)
                    self.assertIn("Nothing has been deleted", error)
                    self.assertIn(str(src), error)

    def test_copy_rewrite_is_one_transaction(self):
        """C5': the rewrite fails part-way (a trigger aborts the second UPDATE). The temporary
        primary keeps every one of the source's rows, and there is still no destination
        primary."""
        for kind, layout in self._cases():
            with self.subTest(kind=kind, layout=layout):
                self._fresh()
                src = self.source(kind, "SRC")
                _abort_second_update(src)
                src_rows = stored_records(src)
                directory, stem = LAYOUTS[layout]
                dst = self.root / directory / f"{stem}.sqlite"
                with self.assertRaises(RuntimeError) as ctx:
                    ShardedPool.copy_store(src, dst)
                self.assertIn("rewrite", str(ctx.exception))
                temp = dst.with_name(dst.name + ".incomplete-copy")
                self.assertEqual(stored_records(temp), src_rows)
                self.assertGuarded(dst)

    def test_move(self):
        real_rename = os.rename
        rewrite = lambda: mock.patch.object(
            ShardedPool,
            "_write_shard_records",
            side_effect=OSError(errno.EIO, "injected failure"),
        )
        points = [
            (
                "M1 dies before shard #0 is renamed",
                lambda: mock.patch("os.rename", _fail_on_call(1, real_rename)),
                "opens",
                "absent",
            ),
            (
                "M2 dies after shard #0",
                lambda: mock.patch("os.rename", _fail_on_call(2, real_rename)),
                "refused",
                "guard",
            ),
            (
                "M2 dies after shard #1",
                lambda: mock.patch("os.rename", _fail_on_call(3, real_rename)),
                "refused",
                "guard",
            ),
            (
                "M3 dies after every shard, before the primary",
                lambda: mock.patch("os.rename", _fail_on_call(4, real_rename)),
                "refused",
                "guard",
            ),
            (
                "M4 dies after the primary, before the rewrite",
                rewrite,
                "absent",
                "old rows",
            ),
        ]
        for kind, layout in self._cases():
            for row, fail, at_src, at_dst in points:
                with self.subTest(kind=kind, layout=layout, row=row):
                    self._fresh()
                    src, dst, contents, error = self._run("move", kind, layout, fail)

                    if at_src == "opens":
                        self.assertOpensAgainst(src, self.shard_paths(src), contents)
                    elif at_src == "refused":
                        self.assertRefused(src)
                    else:
                        self.assertFalse(os.path.lexists(src))

                    if at_dst == "absent":
                        self.assertAbsent(dst)
                    elif at_dst == "guard":
                        self.assertGuarded(dst)
                    elif layout == "new directory, same stem":
                        # the old names are the new names, in the new directory: its own files
                        self.assertOpensAgainst(dst, self.shard_paths(dst), contents)
                    else:
                        # the old names, read in the destination directory, name nothing
                        self.assertRefused(dst)

                    self.assertIn("failed at step", error)
                    self.assertIn("Nothing has been deleted", error)

    def test_move_rewrite_is_one_transaction(self):
        """M4': the rewrite fails part-way. The destination primary keeps every one of the
        source's rows, and so is in state M4."""
        for kind, layout in self._cases():
            with self.subTest(kind=kind, layout=layout):
                self._fresh()
                src = self.source(kind, "SRC")
                contents = self.contents(src)
                _abort_second_update(src)
                src_rows = stored_records(src)
                directory, stem = LAYOUTS[layout]
                dst = self.root / directory / f"{stem}.sqlite"
                with self.assertRaises(RuntimeError):
                    ShardedPool.move_store(src, dst)
                self.assertEqual(stored_records(dst), src_rows)
                if layout == "new directory, same stem":
                    self.assertOpensAgainst(dst, self.shard_paths(dst), contents)
                else:
                    self.assertRefused(dst)

    def test_move_across_filesystems_fails_before_anything_moves(self):
        src = self.new_source("B")
        contents = self.contents(src)
        dst = self.root / "C" / "moved.sqlite"
        cross = OSError(errno.EXDEV, "Cross-device link")
        with mock.patch("os.rename", side_effect=cross):
            with self.assertRaises(RuntimeError) as ctx:
                ShardedPool.move_store(src, dst)
        message = str(ctx.exception)
        self.assertIn("different filesystems", message)
        self.assertIn("copy it instead, and then delete the source by hand", message)
        self.assertOpensAgainst(src, self.shard_paths(src), contents)
        self.assertAbsent(dst)


def _abort_second_update(primary: Path) -> None:
    """Make any UPDATE of shard #1's row abort, so that a rewrite fails after its first UPDATE.
    The trigger is copied with the primary."""
    conn = sqlite3.connect(primary)
    try:
        with conn:
            conn.execute(
                "CREATE TRIGGER abort_second BEFORE UPDATE ON shards WHEN OLD.serial = 1 "
                "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
            )
    finally:
        conn.close()


class TestNothingElseIsTouched(_StoreCase):
    """Test 7: a <stem>.manifest.json and an unrelated file beside the source stay where they are,
    unchanged, after a copy and after a move, and nothing of that name appears at the
    destination. The interface handles the primary and its shards and nothing else."""

    def _others(self, src: Path):
        manifest = src.with_name(f"{src.stem}.manifest.json")
        manifest.write_text('{"datastore": "%s"}\n' % src)
        unrelated = src.with_name("notes.txt")
        unrelated.write_text("not part of the store\n")
        return {p: (sha256(p), p.stat().st_mtime_ns) for p in (manifest, unrelated)}

    def _check(self, others, dst: Path):
        for path, (digest, mtime) in others.items():
            self.assertEqual((sha256(path), path.stat().st_mtime_ns), (digest, mtime))
        self.assertEqual(
            sorted(p.name for p in dst.parent.iterdir()),
            sorted([dst.name, *self.bare_names(dst).values()]),
        )
        self.assertFalse((dst.parent / "notes.txt").exists())
        self.assertFalse(dst.with_name(f"{dst.stem}.manifest.json").exists())

    def test_copy(self):
        src = self.new_source("B")
        others = self._others(src)
        dst = self.root / "C" / "copy.sqlite"
        ShardedPool.copy_store(src, dst)
        self._check(others, dst)

    def test_move(self):
        src = self.new_source("B")
        others = self._others(src)
        dst = self.root / "C" / "moved.sqlite"
        ShardedPool.move_store(src, dst)
        self._check(others, dst)


if __name__ == "__main__":
    unittest.main()
