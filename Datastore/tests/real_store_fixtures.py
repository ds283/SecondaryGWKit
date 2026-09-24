"""
A small **real** ShardedPool store on disk, for tests of the read-only reader and what is built on
it. No Ray, no actor, nothing under ``var/``.

``shard_store_fixtures`` makes a primary with placeholder shards, which are text files and cannot
be read as databases. This module makes shards that are SQLite databases:

- the primary is written by ``ShardedPool._write_shard_data`` itself, on a pool made without its
  constructor (``shard_store_fixtures.bare_pool``), with the replicated and sharded table lists of
  ``config/sharding.py``. The shard-key rows are then added through the pool's own ``shard_keys``
  table;
- every shard holds every table ``build_schema`` declares, created on a temporary read-write
  engine that is disposed before the builder returns. No journal file is left behind;
- replicated rows (``REPLICATED_ROWS``) are copied into every shard, with the same serials, as
  ``ShardedPool`` replicates them;
- sharded rows (``SHARDED_ROWS``) go into the shard their key names. Shards 0 and 1 each hold a
  ``TkNumericIntegration`` with ``TkNumeric_tags`` rows and ``TkNumericValue`` rows.

**Adding rows.** The row sets are plain data: table name -> list of column -> value dicts, with
explicit serials. A test (or prompt 02's inventory tests) adds rows by passing its own
``replicated`` / ``sharded`` mappings to ``build_real_store``, usually built with ``with_rows``
from the defaults. A ``timestamp`` column that a row leaves out is filled with ``FIXED_TIMESTAMP``,
as ``Datastore._insert`` fills it with the time; every other value is the row's own.

**Old stores.** ``build_old_store`` makes a store as if written before the code declared one table
and one column: shard ``OLD_MISSING_TABLE_SHARD`` lacks ``OLD_MISSING_TABLE``, and on shard
``OLD_MISSING_COLUMN_SHARD`` the table ``OLD_MISSING_COLUMN_TABLE`` lacks
``OLD_MISSING_COLUMN``. ``build_real_store`` takes ``missing_tables`` and ``missing_columns`` for
any other combination.

This module is not a test module (no ``test_`` prefix); the test modules import it.
"""

import copy
import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import sqlalchemy as sqla

from Datastore.SQL.Datastore import _factories
from Datastore.SQL.schema import build_schema
from Datastore.shard_paths import shard_file_name
from Datastore.tests.shard_store_fixtures import bare_pool
from config.sharding import replicated_tables, sharded_tables

RowSet = Dict[str, List[Dict[str, object]]]

FIXED_TIMESTAMP = datetime(2026, 9, 24, 12, 0, 0)

# illustrative values: LambdaCDM's type id (CosmologyModels/model_ids.py) and TkNumericIntegration's
# break-point kind (CosmologyModels/GenericEOS/GenericEOS.py BREAK_POINT_ALL)
_LAMBDACDM_TYPE = 0
_BREAK_POINT_ALL = "all"

REPLICATED_ROWS: RowSet = {
    "version": [{"serial": 1, "label": "2025.1.1"}],
    "store_tag": [
        {"serial": 1, "label": "Run_fixture"},
        {"serial": 2, "label": "grid-A"},
        {"serial": 3, "label": "unused-tag"},
    ],
    "redshift": [
        {"serial": 1, "z": 1.0e5, "source": True, "response": False},
        {"serial": 2, "z": 10.0, "source": True, "response": True},
        {"serial": 3, "z": 0.0, "source": False, "response": True},
    ],
    "wavenumber": [
        {"serial": 1, "k_inv_Mpc": 0.1, "source": True, "response": True},
        {"serial": 2, "k_inv_Mpc": 1.0, "source": True, "response": True},
    ],
    "tolerance": [
        {"serial": 1, "log10_tol": -5.0},
        {"serial": 2, "log10_tol": -7.0},
    ],
    "LambdaCDM": [
        {
            "serial": 1,
            "name": "fixture-LambdaCDM",
            "omega_m": 0.31,
            "omega_cc": 0.69,
            "h": 0.67,
            "f_baryon": 0.16,
            "T_CMB_Kelvin": 2.7255,
            "Neff": 3.046,
        }
    ],
    "IntegrationSolver": [{"serial": 1, "stepping": 0, "label": "fixture-solver"}],
    "wavenumber_exit_time": [
        {
            "serial": 1,
            "version": 1,
            "stepping": 0,
            "wavenumber_serial": 1,
            "cosmology_type": _LAMBDACDM_TYPE,
            "cosmology_serial": 1,
            "atol_serial": 1,
            "rtol_serial": 2,
            "z_exit": 3.0e4,
        },
        {
            "serial": 2,
            "version": 1,
            "stepping": 0,
            "wavenumber_serial": 2,
            "cosmology_type": _LAMBDACDM_TYPE,
            "cosmology_serial": 1,
            "atol_serial": 1,
            "rtol_serial": 2,
            "z_exit": 3.0e5,
        },
    ],
    "BackgroundModel": [
        {
            "serial": 1,
            "version": 1,
            "label": "fixture-background",
            "cosmology_type": _LAMBDACDM_TYPE,
            "cosmology_serial": 1,
            "tau_gauss_order": 8,
            "cs_tau_gauss_order": 8,
            "friction_F_gauss_order": 8,
            "solver_serial": 1,
            "z_init_serial": 1,
            "z_samples": 3,
            "source_grid_digest": "fixture-digest",
            "source_grid_construction": 1,
            "validated": True,
        }
    ],
    "BackgroundModel_tags": [{"model_serial": 1, "tag_serial": 1}],
}


def _tk_numeric(serial: int, exit_serial: int) -> RowSet:
    """One TkNumericIntegration with two tags and three values, serials offset by ``serial``."""
    return {
        "TkNumericIntegration": [
            {
                "serial": serial,
                "version": 1,
                "label": f"fixture-tk-{serial}",
                "wavenumber_exit_serial": exit_serial,
                "model_serial": 1,
                "atol_serial": 1,
                "rtol_serial": 2,
                "break_point_kind": _BREAK_POINT_ALL,
                "solver_serial": 1,
                "z_init_serial": 1,
                "z_min_serial": 3,
                "z_samples": 3,
                "stop_T": 0.5,
                "stop_Tprime": -0.25,
                "validated": True,
            }
        ],
        "TkNumeric_tags": [
            {"integration_serial": serial, "tag_serial": 1},
            {"integration_serial": serial, "tag_serial": 2},
        ],
        "TkNumericValue": [
            {
                "serial": 10 * serial + z,
                "integration_serial": serial,
                "z_serial": z,
                "T": 1.0 / z,
                "Tprime": -0.5 / z,
            }
            for z in (1, 2, 3)
        ],
    }


# shard serial -> rows in that shard. The shard key is the wavenumber (config/sharding.py):
# wavenumber 1 lives on shard 0 and wavenumber 2 on shard 1 (SHARD_KEYS)
SHARDED_ROWS: Dict[int, RowSet] = {
    0: _tk_numeric(serial=1, exit_serial=1),
    1: _tk_numeric(serial=2, exit_serial=2),
}

# wavenumber serial -> shard serial, written to the primary's shard_keys table
SHARD_KEYS: Dict[int, int] = {1: 0, 2: 1}

OLD_MISSING_TABLE = "OneLoopIntegral_tags"
OLD_MISSING_TABLE_SHARD = 1
OLD_MISSING_COLUMN_TABLE = "TkNumericIntegration"
OLD_MISSING_COLUMN = "stop_Tprime"
OLD_MISSING_COLUMN_SHARD = 0


@dataclass(frozen=True)
class RealStore:
    """A store built by ``build_real_store``: its primary and its shard files, by serial."""

    primary: Path
    shard_files: Dict[int, Path]
    replicated: RowSet
    sharded: Dict[int, RowSet]

    @property
    def directory(self) -> Path:
        return self.primary.parent


def with_rows(base: RowSet, extra: RowSet) -> RowSet:
    """A copy of ``base`` with ``extra``'s rows appended, table by table."""
    out = copy.deepcopy(base)
    for table, rows in extra.items():
        out.setdefault(table, []).extend(copy.deepcopy(rows))
    return out


def _write_primary(
    primary: Path, shard_files: Dict[int, Path], shard_keys: Mapping[int, int]
) -> None:
    pool = bare_pool(primary)
    pool._ShardKeyType_name = "wavenumber"
    pool._replicated_tables = list(replicated_tables)
    pool._sharded_tables = dict(sharded_tables)
    pool._shard_db_files = dict(shard_files)
    pool._create_engine()
    try:
        pool._write_shard_data()
        if len(shard_keys) > 0:
            with pool._engine.begin() as conn:
                conn.execute(
                    sqla.insert(pool._shard_key_table),
                    [
                        {"key_serial": key, "shard_id": shard}
                        for key, shard in sorted(shard_keys.items())
                    ],
                )
    finally:
        pool._engine.dispose()


def _insert_rows(conn, tables: Mapping[str, sqla.Table], rows: RowSet) -> None:
    for name, table_rows in rows.items():
        if len(table_rows) == 0:
            continue
        table = tables[name]
        payload = []
        for row in table_rows:
            row = dict(row)
            if "timestamp" in table.c and "timestamp" not in row:
                row["timestamp"] = FIXED_TIMESTAMP
            payload.append(row)
        conn.execute(sqla.insert(table), payload)


def _write_shard(
    path: Path,
    rows: List[RowSet],
    missing_tables: Iterable[str],
    missing_columns: Mapping[str, Iterable[str]],
    extra_sql: Iterable[str] = (),
) -> None:
    metadata = sqla.MetaData()
    built = build_schema(metadata, _factories)
    missing_tables = set(missing_tables)

    engine = sqla.create_engine(f"sqlite:///{path}", future=True)
    try:
        created = [t for n, t in built.tables.items() if n not in missing_tables]
        metadata.create_all(engine, tables=created)
        with engine.begin() as conn:
            for row_set in rows:
                _insert_rows(conn, built.tables, row_set)
    finally:
        engine.dispose()

    # an old store's table lacks a column the code now declares. The column is dropped after the
    # rows are written, so that the rest of each row is still there (SQLite >= 3.35). Then any
    # extra statements (a column or a table the code does not declare) are run
    statements = [
        f'ALTER TABLE "{table}" DROP COLUMN "{column}"'
        for table, columns in missing_columns.items()
        for column in columns
    ] + list(extra_sql)
    if len(statements) > 0:
        conn = sqlite3.connect(path)
        try:
            with conn:
                for statement in statements:
                    conn.execute(statement)
        finally:
            conn.close()


def build_real_store(
    directory: Path,
    stem: str = "fixture-store",
    shards: int = 2,
    replicated: Optional[RowSet] = None,
    sharded: Optional[Dict[int, RowSet]] = None,
    shard_keys: Optional[Mapping[int, int]] = None,
    missing_tables: Optional[Mapping[int, Iterable[str]]] = None,
    missing_columns: Optional[Mapping[int, Mapping[str, Iterable[str]]]] = None,
    extra_sql: Optional[Mapping[int, Iterable[str]]] = None,
) -> RealStore:
    """
    Build a real store in ``directory`` (which must exist) and return it.

    ``replicated`` rows go into every shard with the same serials; ``sharded[n]`` rows go into
    shard ``n`` only. Both default to this module's row sets. ``missing_tables[n]`` lists tables
    shard ``n`` is created without; ``missing_columns[n][table]`` lists columns dropped from that
    table on shard ``n``. No row may be given for a missing table. ``extra_sql[n]`` lists SQL
    statements run on shard ``n`` last, e.g. to add a column or a table the code does not declare.
    """
    replicated = copy.deepcopy(REPLICATED_ROWS if replicated is None else replicated)
    sharded = copy.deepcopy(SHARDED_ROWS if sharded is None else sharded)
    shard_keys = dict(SHARD_KEYS if shard_keys is None else shard_keys)
    missing_tables = {} if missing_tables is None else missing_tables
    missing_columns = {} if missing_columns is None else missing_columns
    extra_sql = {} if extra_sql is None else extra_sql

    if shards < 2:
        raise ValueError("a real store fixture has at least two shards")
    unknown = set(sharded) - set(range(shards))
    if unknown:
        raise ValueError(
            f"sharded rows name shards {sorted(unknown)} that do not exist"
        )

    primary = (Path(directory) / f"{stem}.sqlite").resolve()
    shard_files = {
        serial: primary.parent / shard_file_name(primary, serial)
        for serial in range(shards)
    }

    for serial, path in shard_files.items():
        _write_shard(
            path,
            [replicated, sharded.get(serial, {})],
            missing_tables.get(serial, ()),
            missing_columns.get(serial, {}),
            extra_sql.get(serial, ()),
        )

    _write_primary(primary, shard_files, shard_keys)

    return RealStore(
        primary=primary,
        shard_files=shard_files,
        replicated=replicated,
        sharded=sharded,
    )


def build_old_store(directory: Path, stem: str = "old-store") -> RealStore:
    """A store with one shard missing a table and one table missing a column (module docstring)."""
    return build_real_store(
        directory,
        stem=stem,
        missing_tables={OLD_MISSING_TABLE_SHARD: [OLD_MISSING_TABLE]},
        missing_columns={
            OLD_MISSING_COLUMN_SHARD: {OLD_MISSING_COLUMN_TABLE: [OLD_MISSING_COLUMN]}
        },
    )


def expected_row_counts(store: RealStore, serial: int) -> Dict[str, int]:
    """Rows this fixture put in each table of shard ``serial`` (tables it left empty omitted)."""
    counts: Dict[str, int] = {}
    for row_set in (store.replicated, store.sharded.get(serial, {})):
        for table, rows in row_set.items():
            counts[table] = counts.get(table, 0) + len(rows)
    return counts


def independent_row_counts(path: Path) -> Dict[str, int]:
    """Every table's row count in the file ``path``, read with stdlib ``sqlite3`` ``mode=ro``,
    independently of the reader and of SQLAlchemy."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return {
            name: conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            for name in names
        }
    finally:
        conn.close()


def file_state(directory: Path) -> Tuple[List[str], Dict[str, Tuple[str, int, int]]]:
    """The directory listing, and each file's (SHA-256, size, st_mtime_ns)."""
    listing = sorted(p.name for p in Path(directory).iterdir())
    state = {}
    for name in listing:
        path = Path(directory) / name
        st = path.lstat()
        state[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            st.st_size,
            st.st_mtime_ns,
        )
    return listing, state
