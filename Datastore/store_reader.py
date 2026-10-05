"""
A read-only reader over a **closed** ShardedPool store.

``open_read_only(primary)`` is a context manager. It finds the store's shards through
``ShardedPool._read_closed_store`` (the one implementation of reading and checking a primary's
``shards`` table), opens every shard file read-only, and builds the tables once with the one schema
builder, ``Datastore.SQL.schema.build_schema``. It yields a ``ReadOnlyStore``; on exit it disposes
every engine it made.

**It cannot write.** It is not "careful"; it has no write path:

- every file is opened ``sqlite:///file:{path}?mode=ro&uri=true``. A read-write open would replay a
  hot rollback journal, which is a write. ``immutable=1`` is never used either: it would hide a
  concurrent writer instead of refusing it;
- a store with a journal file (``ShardedPool._journal_paths``: ``-journal``, ``-wal``, ``-shm``)
  beside its primary or any shard is refused, naming the file, before that file is opened. Such a
  store is open, or was not closed cleanly, and a reading of it describes no instant. The reader
  reports this and repairs nothing;
- there is no ``create_all``, no ``Datastore._ensure_tables``, no DDL and no DML.

**It needs no Ray.** It never calls ``ray.init`` and never constructs a ``Datastore`` actor or a
``ShardedPool``. Importing this module imports ``ray`` transitively, through the factory map in
``Datastore/SQL/Datastore.py`` and through ``ShardedPool``; that initialises nothing.

**A store whose files differ from the declared tables is refused.** Every shard file is compared
with the tables ``build_schema`` declares by ``Datastore.SQL.schema.schema_differences``, before
anything reads its tables. A table the file lacks, a table it has that the code does not declare,
a declared column a table lacks, or a column the code does not declare refuses the store with
``StoreSchemaMismatch``, naming the shard, its path and each table and column by kind
(prompts/datastore-generic, prompt 05). Nothing is created here, and the reader has nothing to
recover: a store written by older code is regenerated, not read.
"""

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Tuple, Union

import sqlalchemy as sqla

from Datastore.SQL.Datastore import _factories
from Datastore.SQL.ShardedPool import ShardedPool
from Datastore.SQL.schema import (
    StoreSchemaMismatch,
    build_schema,
    schema_differences,
)
from Datastore.shard_paths import shard_file_problem

PathType = Union[str, os.PathLike]

_VERB = "read"


@dataclass(frozen=True)
class ReadOnlyShard:
    """One shard of a store opened by ``open_read_only``.

    ``tables`` are the ``Table`` objects the code declares, shared by every shard of the store.
    The shard's file holds exactly these tables, each with exactly its declared columns: a shard
    that differs is refused before it is yielded.
    """

    serial: int
    path: Path
    engine: sqla.Engine
    tables: Mapping[str, sqla.Table]


@dataclass(frozen=True)
class ReadOnlyStore:
    """A closed store opened read-only: its primary, and its shards in serial order."""

    primary: Path
    shards: Tuple[ReadOnlyShard, ...]
    tables: Mapping[str, sqla.Table]
    records: Mapping[str, Mapping]

    def shard(self, serial: int) -> ReadOnlyShard:
        for shard in self.shards:
            if shard.serial == serial:
                return shard
        raise KeyError(f"store {str(self.primary)!r} has no shard #{serial}")


def _refuse(primary: Path, reason: str) -> RuntimeError:
    return RuntimeError(
        f'Cannot {_VERB} sharded datastore "{str(primary)}": {reason}. Nothing was read or repaired'
    )


def _refuse_journals(primary: Path, path: Path, what: str) -> None:
    for journal in ShardedPool._journal_paths(path):
        if os.path.lexists(journal):
            raise _refuse(
                primary,
                f'{what} "{str(path)}" has "{str(journal)}" beside it, so the store is open or was not closed cleanly',
            )


def read_only_url(path: Path) -> str:
    """The SQLAlchemy URL of ``path`` opened read-only: ``mode=ro``, never ``immutable=1``."""
    return f"sqlite:///file:{path}?mode=ro&uri=true"


def _read_only_engine(path: Path) -> sqla.Engine:
    return sqla.create_engine(read_only_url(path), future=True)


def _describe_shard(
    primary: Path,
    serial: int,
    path: Path,
    engine: sqla.Engine,
    tables: Mapping[str, sqla.Table],
) -> ReadOnlyShard:
    """The shard, once its file is shown to hold exactly ``tables``. Any difference, a table the
    file lacks included, raises ``StoreSchemaMismatch`` naming the shard and each difference.
    """
    with engine.connect() as conn:
        differences = schema_differences(conn, tables)
        conn.rollback()

    if not differences.empty:
        raise StoreSchemaMismatch(
            _VERB, primary, f'shard #{serial} "{str(path)}"', differences
        )

    return ReadOnlyShard(serial=serial, path=path, engine=engine, tables=tables)


@contextlib.contextmanager
def open_read_only(primary: PathType) -> Iterator[ReadOnlyStore]:
    """
    Open the closed store whose primary is ``primary``, read-only, and yield a ``ReadOnlyStore``.

    Refuses with ``RuntimeError``, naming the file, before any shard is opened, if: the primary is
    missing, not a regular file or a symbolic link; the primary or any shard has a journal file
    beside it; the primary's ``shards`` table cannot be read, or names a shard that is unusable
    (``ShardedPool._read_closed_store``); or the primary records no shards.

    Refuses with ``StoreSchemaMismatch`` (a ``RuntimeError``), naming the shard, before anything
    reads its tables and before anything is yielded, if a shard's file differs from the tables
    the code declares: a table it lacks, a table or column the code does not declare, or a
    declared column it lacks (``Datastore.SQL.schema.schema_differences``).

    Every engine is disposed on exit, and on a refusal after any was made.
    """
    given = Path(primary).absolute()
    problem = shard_file_problem(given)
    if problem is not None:
        raise _refuse(given, f'the primary "{str(given)}" {problem}')
    primary_path = given.resolve()

    _refuse_journals(primary_path, primary_path, "the primary")

    files, _records = ShardedPool._read_closed_store(primary_path, _VERB)
    if len(files) == 0:
        raise _refuse(
            primary_path, f'the primary "{str(primary_path)}" records no shards'
        )

    for serial, path in sorted(files.items()):
        _refuse_journals(primary_path, Path(path), f"shard #{serial}")

    built = build_schema(sqla.MetaData(), _factories)

    engines = []
    try:
        shards = []
        for serial, path in sorted(files.items()):
            engine = _read_only_engine(Path(path))
            engines.append(engine)
            shards.append(
                _describe_shard(primary_path, serial, Path(path), engine, built.tables)
            )

        yield ReadOnlyStore(
            primary=primary_path,
            shards=tuple(shards),
            tables=built.tables,
            records=built.records,
        )
    finally:
        for engine in engines:
            engine.dispose()
