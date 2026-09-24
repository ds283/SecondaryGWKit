"""
The read-only store reader, ``Datastore/store_reader.py`` ``open_read_only`` (store-fingerprint
prompt 01), on the real multi-shard stores of ``real_store_fixtures``.

- it reads: every shard, with its serial and path, and row counts equal to an independent stdlib
  ``sqlite3`` ``mode=ro`` count of each file;
- it never writes: every file's SHA-256, size and ``st_mtime_ns``, and the directory listing, are
  unchanged by opening the store and reading every table on every shard, and a write attempted
  through one of its engines raises;
- old stores: a missing table and a missing column are reported, by name, on the right shard,
  and every other table still reads;
- refusals, each naming the file and writing nothing: a journal beside the primary or a shard, a
  missing shard (the refusal coming from ``ShardedPool._read_closed_store``), a primary with no
  ``shards`` table;
- no Ray: in a child interpreter, opening a store and reading every table leaves
  ``ray.is_initialized()`` false.

No test here starts Ray or constructs a ``ShardedPool``, and nothing is opened under ``var/``.
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import sqlalchemy as sqla
from sqlalchemy.exc import OperationalError

from Datastore.SQL.ShardedPool import ShardedPool
from Datastore.store_reader import open_read_only
from Datastore.tests.real_store_fixtures import (
    OLD_MISSING_COLUMN,
    OLD_MISSING_COLUMN_SHARD,
    OLD_MISSING_COLUMN_TABLE,
    OLD_MISSING_TABLE,
    OLD_MISSING_TABLE_SHARD,
    build_old_store,
    build_real_store,
    expected_row_counts,
    file_state,
    independent_row_counts,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

JOURNAL_SUFFIXES = ("-journal", "-wal", "-shm")


def _readable_columns(shard, name):
    table = shard.tables[name]
    absent = set(shard.absent_columns.get(name, ()))
    return [c for c in table.columns if c.name not in absent]


def read_every_table(store):
    """Read every row of every table present on every shard, selecting the columns the file has.
    Returns serial -> table -> row count."""
    counts = {}
    for shard in store.shards:
        counts[shard.serial] = {}
        with shard.engine.connect() as conn:
            for name in shard.tables:
                if name in shard.absent_tables:
                    continue
                rows = conn.execute(
                    sqla.select(*_readable_columns(shard, name))
                ).fetchall()
                counts[shard.serial][name] = len(rows)
    return counts


class _TempStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.dir = self.root / "store"
        self.dir.mkdir()

    def tearDown(self):
        self._tmp.cleanup()


class TestReaderReads(_TempStore):
    def setUp(self):
        super().setUp()
        self.store = build_real_store(self.dir)

    def test_every_shard_with_its_serial_and_path(self):
        with open_read_only(self.store.primary) as store:
            self.assertEqual(store.primary, self.store.primary)
            self.assertEqual(
                [(s.serial, s.path) for s in store.shards],
                sorted(self.store.shard_files.items()),
            )
            for shard in store.shards:
                self.assertEqual(shard.absent_tables, frozenset())
                self.assertEqual(shard.extra_tables, frozenset())
                self.assertTrue(all(v == () for v in shard.absent_columns.values()))
                self.assertTrue(all(v == () for v in shard.extra_columns.values()))
                self.assertEqual(set(shard.absent_columns), set(shard.tables))

    def test_row_counts_equal_an_independent_count(self):
        with open_read_only(self.store.primary) as store:
            counts = read_every_table(store)
            for shard in store.shards:
                with shard.engine.connect() as conn:
                    by_count = {
                        name: conn.execute(
                            sqla.select(sqla.func.count()).select_from(table)
                        ).scalar()
                        for name, table in shard.tables.items()
                    }
                independent = independent_row_counts(shard.path)
                with self.subTest(shard=shard.serial):
                    self.assertEqual(by_count, independent)
                    self.assertEqual(counts[shard.serial], independent)
                    # and they are the rows the fixture wrote
                    expected = expected_row_counts(self.store, shard.serial)
                    self.assertEqual(
                        {n: c for n, c in independent.items() if c > 0}, expected
                    )

    def test_replicated_rows_have_the_same_serials_on_every_shard(self):
        with open_read_only(self.store.primary) as store:
            per_shard = []
            for shard in store.shards:
                with shard.engine.connect() as conn:
                    per_shard.append(
                        {
                            name: sorted(
                                conn.execute(
                                    sqla.select(shard.tables[name].c.serial)
                                ).scalars()
                            )
                            for name in self.store.replicated
                            if "serial" in shard.tables[name].c
                        }
                    )
            for other in per_shard[1:]:
                self.assertEqual(per_shard[0], other)

    def test_every_engine_is_disposed_on_exit(self):
        with open_read_only(self.store.primary) as store:
            read_every_table(store)
            engines = [s.engine for s in store.shards]
            self.assertTrue(all(e.pool.checkedin() > 0 for e in engines))
        self.assertTrue(all(e.pool.checkedin() == 0 for e in engines))

    def test_engines_open_mode_ro_and_never_immutable(self):
        with open_read_only(self.store.primary) as store:
            for shard in store.shards:
                url = shard.engine.url
                self.assertEqual(url.query.get("mode"), "ro")
                self.assertEqual(url.query.get("uri"), "true")
                self.assertNotIn("immutable", url.query)
                self.assertEqual(url.database, f"file:{shard.path}")


class TestReaderNeverWrites(_TempStore):
    def setUp(self):
        super().setUp()
        self.store = build_real_store(self.dir)

    def test_reading_every_table_changes_nothing(self):
        before = file_state(self.dir)
        with open_read_only(self.store.primary) as store:
            read_every_table(store)
            during = file_state(self.dir)
        after = file_state(self.dir)
        self.assertEqual(before, during)
        self.assertEqual(before, after)
        self.assertFalse(any(n.endswith(JOURNAL_SUFFIXES) for n in after[0]))

    def test_a_write_through_a_reader_engine_raises(self):
        before = file_state(self.dir)
        with open_read_only(self.store.primary) as store:
            for shard in store.shards:
                with self.subTest(shard=shard.serial, what="insert"):
                    with self.assertRaises(OperationalError) as cm:
                        with shard.engine.begin() as conn:
                            conn.execute(
                                sqla.insert(shard.tables["store_tag"]),
                                {"serial": 99, "label": "written"},
                            )
                    self.assertIn("readonly", str(cm.exception))
                with self.subTest(shard=shard.serial, what="ddl"):
                    with self.assertRaises(OperationalError):
                        with shard.engine.begin() as conn:
                            conn.exec_driver_sql("CREATE TABLE written (x INTEGER)")
        self.assertEqual(before, file_state(self.dir))
        for path in self.store.shard_files.values():
            self.assertNotIn("written", independent_row_counts(path))


class TestOldStores(_TempStore):
    def test_missing_table_and_column_are_reported_on_the_right_shard(self):
        old = build_old_store(self.dir)
        before = file_state(self.dir)
        with open_read_only(old.primary) as store:
            for shard in store.shards:
                with self.subTest(shard=shard.serial):
                    if shard.serial == OLD_MISSING_TABLE_SHARD:
                        self.assertEqual(
                            shard.absent_tables, frozenset({OLD_MISSING_TABLE})
                        )
                        self.assertNotIn(OLD_MISSING_TABLE, shard.absent_columns)
                    else:
                        self.assertEqual(shard.absent_tables, frozenset())

                    absent = {n: c for n, c in shard.absent_columns.items() if c}
                    if shard.serial == OLD_MISSING_COLUMN_SHARD:
                        self.assertEqual(
                            absent, {OLD_MISSING_COLUMN_TABLE: (OLD_MISSING_COLUMN,)}
                        )
                    else:
                        self.assertEqual(absent, {})
                    self.assertTrue(all(c == () for c in shard.extra_columns.values()))

            # every other table still reads, and the table missing a column reads without it
            counts = read_every_table(store)
            for shard in store.shards:
                independent = independent_row_counts(shard.path)
                self.assertEqual(counts[shard.serial], independent)
            self.assertEqual(
                counts[OLD_MISSING_COLUMN_SHARD][OLD_MISSING_COLUMN_TABLE], 1
            )
        self.assertEqual(before, file_state(self.dir))

    def test_extra_columns_and_tables_are_reported(self):
        store_files = build_real_store(
            self.dir,
            extra_sql={
                1: [
                    'ALTER TABLE "GkSource" ADD COLUMN legacy_note VARCHAR(64)',
                    "CREATE TABLE retired_table (x INTEGER)",
                ]
            },
        )
        with open_read_only(store_files.primary) as store:
            shard0, shard1 = store.shard(0), store.shard(1)
            self.assertEqual(
                {n: c for n, c in shard1.extra_columns.items() if c},
                {"GkSource": ("legacy_note",)},
            )
            self.assertEqual(shard1.extra_tables, frozenset({"retired_table"}))
            self.assertEqual({n: c for n, c in shard0.extra_columns.items() if c}, {})
            self.assertEqual(shard0.extra_tables, frozenset())
            self.assertTrue(all(c == () for c in shard1.absent_columns.values()))


class TestRefusals(_TempStore):
    def setUp(self):
        super().setUp()
        self.store = build_real_store(self.dir)

    def assert_refused_naming(self, fragment: str):
        before = file_state(self.dir)
        with self.assertRaises(RuntimeError) as cm:
            with open_read_only(self.store.primary):
                self.fail("the store was opened")
        self.assertIn(fragment, str(cm.exception))
        self.assertEqual(before, file_state(self.dir))
        return str(cm.exception)

    def test_journal_beside_the_primary(self):
        for suffix in JOURNAL_SUFFIXES:
            with self.subTest(suffix=suffix):
                journal = self.store.primary.with_name(self.store.primary.name + suffix)
                journal.write_bytes(b"")
                try:
                    message = self.assert_refused_naming(str(journal))
                    self.assertIn("the primary", message)
                finally:
                    journal.unlink()

    def test_journal_beside_a_shard(self):
        for serial, path in sorted(self.store.shard_files.items()):
            for suffix in JOURNAL_SUFFIXES:
                with self.subTest(shard=serial, suffix=suffix):
                    journal = path.with_name(path.name + suffix)
                    journal.write_bytes(b"")
                    try:
                        message = self.assert_refused_naming(str(journal))
                        self.assertIn(f"shard #{serial}", message)
                    finally:
                        journal.unlink()

    def test_missing_shard_is_refused_by_read_closed_store(self):
        missing = self.store.shard_files[1]
        missing.unlink()
        message = self.assert_refused_naming(str(missing))
        self.assertIn("does not exist", message)
        # the refusal is _read_closed_store's own, not a copy of it
        with self.assertRaises(RuntimeError) as cm:
            ShardedPool._read_closed_store(self.store.primary, "read")
        self.assertEqual(message, str(cm.exception))

    def test_primary_with_no_shards_table(self):
        self.store.primary.unlink()
        conn = sqlite3.connect(self.store.primary)
        try:
            with conn:
                conn.execute("CREATE TABLE unrelated (x INTEGER)")
        finally:
            conn.close()
        message = self.assert_refused_naming(str(self.store.primary))
        self.assertIn("shards table could not be read", message)

    def test_missing_primary(self):
        self.store.primary.unlink()
        message = self.assert_refused_naming(str(self.store.primary))
        self.assertIn("does not exist", message)
        self.assertFalse(self.store.primary.exists())


_CHILD = r"""
import json, sys
from pathlib import Path
import sqlalchemy as sqla
from Datastore.store_reader import open_read_only
import ray

counts = {}
with open_read_only(Path(sys.argv[1])) as store:
    for shard in store.shards:
        with shard.engine.connect() as conn:
            for name, table in shard.tables.items():
                if name in shard.absent_tables:
                    continue
                counts[f"{shard.serial}:{name}"] = len(conn.execute(sqla.select(table)).fetchall())
print(json.dumps({"ray_initialized": ray.is_initialized(), "tables_read": len(counts)}))
"""


class TestNoRay(_TempStore):
    def test_reading_in_a_child_interpreter_leaves_ray_uninitialised(self):
        store = build_real_store(self.dir)
        before = file_state(self.dir)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        result = subprocess.run(
            [sys.executable, "-c", _CHILD, str(store.primary)],
            cwd=str(self.root),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertFalse(report["ray_initialized"])
        with open_read_only(store.primary) as opened:
            expected = sum(len(set(s.tables) - s.absent_tables) for s in opened.shards)
        self.assertEqual(report["tables_read"], expected)
        self.assertEqual(before, file_state(self.dir))


if __name__ == "__main__":
    unittest.main()
