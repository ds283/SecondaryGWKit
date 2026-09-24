"""
Stand-ins for a ShardedPool store on disk, for the shard-path tests. No Ray, no datastore.

A primary is built **by hand** with stdlib ``sqlite3``, with the same five tables and columns as
``ShardedPool._create_engine`` declares, so that a test can stand in for rows the *old* code wrote
(absolute shard paths) without calling ``_write_shard_data``, which no longer writes that form.

Shard files are small placeholder files, not databases: everything these tests exercise --
``_read_shard_data``, the resolver and ``_check_shard_files`` -- looks only at the ``shards`` table
and at whether each shard file exists. Each placeholder holds a marker naming its own location,
so a test can tell which of two same-named files a path points at.

This module is not a test module (no ``test_`` prefix); the test modules import it.
"""

import contextlib
import hashlib
import io
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Tuple

from Datastore.SQL.ShardedPool import ShardedPool

KEY_TYPE = "wavenumber"
REPLICATED = ["version", "wavenumber"]
SHARDED = {"GkSource": "k"}


def bare_pool(primary: Path) -> ShardedPool:
    """A ShardedPool with the attributes the shard-table methods use, and no actors."""
    pool = object.__new__(ShardedPool)
    pool._db_name = primary
    pool._primary_file = Path(primary).resolve()
    pool._timeout = None
    pool._ShardKeyType_name = KEY_TYPE
    pool._replicated_tables = list(REPLICATED)
    pool._sharded_tables = dict(SHARDED)
    pool._shard_db_files = {}
    pool._shard_records = {}
    pool._shard_keys = {}
    pool._engine = None
    return pool


def read_pool(primary: Path) -> ShardedPool:
    """What the constructor's existing-store branch does, up to (not including) the check.
    What the read printed is kept as ``pool.read_stdout``."""
    pool = bare_pool(primary)
    out = io.StringIO()
    pool._create_engine()
    try:
        with contextlib.redirect_stdout(out):
            pool._read_shard_data()
    finally:
        pool._engine.dispose()
        pool.read_stdout = out.getvalue()
    return pool


def write_new_store(primary: Path, shards: int = 3) -> ShardedPool:
    """What the constructor's new-store branch does, minus the actors: create shard placeholders
    as siblings of the primary and write the primary's tables with the current code."""
    pool = bare_pool(primary)
    stem = pool._primary_file.stem
    for i in range(shards):
        shard_file = pool._primary_file.with_stem(f"{stem}-shard{i:04d}")
        write_placeholder(shard_file)
        pool._shard_db_files[i] = shard_file
    pool._create_engine()
    try:
        pool._write_shard_data()
    finally:
        pool._engine.dispose()
    return pool


def write_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"placeholder shard at {path}\n".encode())


def marker(path: Path) -> str:
    return path.read_bytes().decode()


def write_legacy_primary(
    primary: Path,
    records: Dict[int, str],
    shard_keys: Iterable[Tuple[int, int]] = (),
) -> None:
    """Build a primary by hand, holding exactly the given ``shards.filename`` records."""
    primary.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(primary)
    try:
        with conn:
            conn.execute(
                "CREATE TABLE shards (serial INTEGER NOT NULL PRIMARY KEY, "
                "filename VARCHAR(256) NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE shard_key_config (key_type VARCHAR(256) NOT NULL PRIMARY KEY)"
            )
            conn.execute(
                "CREATE TABLE shard_keys (key_serial INTEGER NOT NULL PRIMARY KEY, "
                "shard_id INTEGER NOT NULL REFERENCES shards (serial))"
            )
            conn.execute(
                'CREATE TABLE replicated_tables (serial INTEGER NOT NULL PRIMARY KEY, "table" '
                "VARCHAR(256) NOT NULL)"
            )
            conn.execute(
                'CREATE TABLE sharded_tables (serial INTEGER NOT NULL PRIMARY KEY, "table" '
                "VARCHAR(256) NOT NULL, key_attr VARCHAR(256) NOT NULL)"
            )
            conn.executemany(
                "INSERT INTO shards (serial, filename) VALUES (?, ?)",
                sorted(records.items()),
            )
            conn.execute(
                "INSERT INTO shard_key_config (key_type) VALUES (?)", (KEY_TYPE,)
            )
            conn.executemany(
                "INSERT INTO shard_keys (key_serial, shard_id) VALUES (?, ?)",
                list(shard_keys),
            )
            conn.executemany(
                'INSERT INTO replicated_tables (serial, "table") VALUES (?, ?)',
                list(enumerate(REPLICATED)),
            )
            conn.executemany(
                'INSERT INTO sharded_tables (serial, "table", key_attr) VALUES (?, ?, ?)',
                [(n, t, k) for n, (t, k) in enumerate(SHARDED.items())],
            )
    finally:
        conn.close()


def stored_records(primary: Path) -> Dict[int, str]:
    """The ``shards`` table as stored, read with sqlite3 mode=ro."""
    conn = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
    try:
        return dict(conn.execute("SELECT serial, filename FROM shards"))
    finally:
        conn.close()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
