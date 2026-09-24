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

**Old stores.** A store written before the code declared some table or column cannot have it
created here. Each shard reports the tables the code declares that its file lacks
(``absent_tables``, from ``sqlite_master``), and for each table present, the declared columns the
file lacks (``absent_columns``) and the columns in the file that the code does not declare
(``extra_columns``), both from ``PRAGMA table_info``. The reader reports these and does not refuse
the store: what an old store's missing table or column means is its consumer's decision.
"""

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterator, Mapping, Tuple, Union

import sqlalchemy as sqla

from Datastore.SQL.Datastore import _factories
from Datastore.SQL.ShardedPool import ShardedPool
from Datastore.SQL.schema import build_schema
from Datastore.shard_paths import shard_file_problem

PathType = Union[str, os.PathLike]

_VERB = "read"


@dataclass(frozen=True)
class ReadOnlyShard:
    """One shard of a store opened by ``open_read_only``.

    ``tables`` are the ``Table`` objects the code declares, shared by every shard of the store.
    ``absent_tables`` names the declared tables this shard's file lacks. ``absent_columns`` and
    ``extra_columns`` map each declared table the file holds to the declared columns the file
    lacks, and to the file's columns the code does not declare; a table with neither maps to an
    empty tuple. Column names are in declaration order and file order respectively.
    """

    serial: int
    path: Path
    engine: sqla.Engine
    tables: Mapping[str, sqla.Table]
    absent_tables: FrozenSet[str]
    absent_columns: Mapping[str, Tuple[str, ...]]
    extra_columns: Mapping[str, Tuple[str, ...]]
    extra_tables: FrozenSet[str]


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
    serial: int,
    path: Path,
    engine: sqla.Engine,
    tables: Mapping[str, sqla.Table],
) -> ReadOnlyShard:
    with engine.connect() as conn:
        in_file = {
            row[0]
            for row in conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

        absent_columns: Dict[str, Tuple[str, ...]] = {}
        extra_columns: Dict[str, Tuple[str, ...]] = {}
        for name, table in tables.items():
            if name not in in_file:
                continue
            quoted = name.replace('"', '""')
            file_columns = [
                row[1] for row in conn.exec_driver_sql(f'PRAGMA table_info("{quoted}")')
            ]
            declared = [c.name for c in table.columns]
            absent_columns[name] = tuple(c for c in declared if c not in file_columns)
            extra_columns[name] = tuple(c for c in file_columns if c not in declared)

    return ReadOnlyShard(
        serial=serial,
        path=path,
        engine=engine,
        tables=tables,
        absent_tables=frozenset(n for n in tables if n not in in_file),
        absent_columns=absent_columns,
        extra_columns=extra_columns,
        extra_tables=frozenset(
            n for n in in_file if n not in tables and not n.startswith("sqlite_")
        ),
    )


@contextlib.contextmanager
def open_read_only(primary: PathType) -> Iterator[ReadOnlyStore]:
    """
    Open the closed store whose primary is ``primary``, read-only, and yield a ``ReadOnlyStore``.

    Refuses with ``RuntimeError``, naming the file, before any shard is opened, if: the primary is
    missing, not a regular file or a symbolic link; the primary or any shard has a journal file
    beside it; the primary's ``shards`` table cannot be read, or names a shard that is unusable
    (``ShardedPool._read_closed_store``); or the primary records no shards.

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
            shards.append(_describe_shard(serial, Path(path), engine, built.tables))

        yield ReadOnlyStore(
            primary=primary_path,
            shards=tuple(shards),
            tables=built.tables,
            records=built.records,
        )
    finally:
        for engine in engines:
            engine.dispose()
