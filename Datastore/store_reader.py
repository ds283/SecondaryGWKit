"""
A read-only reader over a **closed** ShardedPool store.

``open_read_only(primary, factories)`` is a context manager. It finds the store's shards through
``ShardedPool._read_closed_store`` (the one implementation of reading and checking a primary's
``shards`` table), opens every shard file read-only, and builds the tables once with the one schema
builder, ``Datastore.SQL.schema.build_schema``, from the registry ``factories`` it is given: the
reader imports no registry (prompts/datastore-generic, prompt 06). It yields a ``ReadOnlyStore``;
on exit it disposes every engine it made.

**The replicated set is the store's.** Every primary records the classes its pool replicates, in
its ``replicated_tables`` table, and a pool refuses to open a store whose record differs from what
it is given. The reader reads that record ``mode=ro`` and yields it as
``ReadOnlyStore.replicated_tables``, in serial order. A primary whose record cannot be read, or
names a class ``factories`` does not declare, is refused. An empty record is what the store says,
and is not refused.

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
``ShardedPool``. Importing this module imports ``ray`` transitively, through ``ShardedPool``; that
initialises nothing.

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
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Tuple, Union

import sqlalchemy as sqla

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
    """A closed store opened read-only: its primary, its shards in serial order, and the classes
    its primary records as replicated, in the order of their serials."""

    primary: Path
    shards: Tuple[ReadOnlyShard, ...]
    tables: Mapping[str, sqla.Table]
    records: Mapping[str, Mapping]
    replicated_tables: Tuple[str, ...]

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


def _read_replicated_tables(primary: Path, factories: Mapping) -> Tuple[str, ...]:
    """
    The class names the primary's ``replicated_tables`` table records, in serial order, read
    ``mode=ro``. Refuses (``_refuse``) a primary whose table cannot be read, naming the table and
    the error, and one that names a class ``factories`` does not declare. An empty record is
    returned as it is.
    """
    try:
        conn = sqlite3.connect(f"{primary.as_uri()}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                'SELECT serial, "table" FROM replicated_tables ORDER BY serial'
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as e:
        raise _refuse(
            primary,
            f'the replicated_tables table of the primary "{str(primary)}" could not be read '
            f"({e})",
        ) from e

    names = tuple(row[1] for row in rows)
    undeclared = [name for name in names if name not in factories]
    if len(undeclared) > 0:
        raise _refuse(
            primary,
            f'the replicated_tables table of the primary "{str(primary)}" names the class(es) '
            f"{undeclared}, which the registry it was given does not declare",
        )
    return names


@contextlib.contextmanager
def open_read_only(primary: PathType, factories: Mapping) -> Iterator[ReadOnlyStore]:
    """
    Open the closed store whose primary is ``primary``, read-only, and yield a ``ReadOnlyStore``.

    ``factories`` is the registry of storable classes (name -> factory) the tables are built from.
    ``ReadOnlyStore.replicated_tables`` is what the primary records, not anything ``factories``
    says.

    Refuses with ``RuntimeError``, naming the file, before any shard is opened, if: the primary is
    missing, not a regular file or a symbolic link; the primary or any shard has a journal file
    beside it; the primary's ``shards`` table cannot be read, or names a shard that is unusable
    (``ShardedPool._read_closed_store``); the primary records no shards; or its
    ``replicated_tables`` table cannot be read, or names a class ``factories`` does not declare.

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

    replicated = _read_replicated_tables(primary_path, factories)

    built = build_schema(sqla.MetaData(), factories)

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
            replicated_tables=replicated,
        )
    finally:
        for engine in engines:
            engine.dispose()
