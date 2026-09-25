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

**The full store** (store-fingerprint prompt 02). ``build_full_store`` builds a store holding at
least one row of every class of the structured inventory, with tags, unvalidated rows and value
rows (``FULL_REPLICATED_ROWS``, ``FULL_SHARDED_ROWS``). It leaves the defaults above as they are.
``relabel_serials`` gives the same content under other serials, and ``vary_row`` changes one column
of one row.

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


# ---------------------------------------------------------------------------------------------
# a full store: every inventory class (store-fingerprint prompt 02)
# ---------------------------------------------------------------------------------------------
#
# ``build_full_store`` builds a store holding at least one row of every class of the structured
# inventory (Datastore/store_inventory.py INVENTORY_CLASSES), with tags wherever the class has an
# association table, unvalidated rows, and value rows. It adds to the defaults above and leaves
# them as they are, so prompt 01's tests read the store they were written against.
#
# Every row states the columns of its key and its references explicitly. Any other non-nullable
# column a row leaves out is filled by ``fill_required`` with a placeholder of its type.
#
# ``relabel_serials`` gives the same content under different serials, and ``vary_row`` changes one
# column of one row, so that tests can build two stores that differ in exactly one respect.

_QCD_TYPE = 1

FULL_REPLICATED_ROWS: RowSet = with_rows(
    REPLICATED_ROWS,
    {
        "version": [{"serial": 2, "label": "2024.9.9"}],
        "store_tag": [
            {"serial": 4, "label": "grid-B"},
            {"serial": 5, "label": "oneloop-tag"},
        ],
        "redshift": [
            {"serial": 4, "z": 1000.0, "source": True, "response": False},
            {"serial": 5, "z": 1.0, "source": False, "response": True},
        ],
        "wavenumber": [
            {"serial": 3, "k_inv_Mpc": 0.5, "source": True, "response": True}
        ],
        "tolerance": [{"serial": 3, "log10_tol": -9.0}],
        "LambdaCDM": [
            {
                "serial": 2,
                "name": "fixture-LambdaCDM-2",
                "omega_m": 0.3,
                "omega_cc": 0.7,
                "h": 0.7,
                "f_baryon": 0.15,
                "T_CMB_Kelvin": 2.725,
                "Neff": 3.0,
            }
        ],
        "QCD_Cosmology": [
            {
                "serial": 1,
                "name": "fixture-QCD",
                "omega_m": 0.31,
                "omega_cc": 0.69,
                "h": 0.67,
                "f_baryon": 0.16,
                "T_CMB_Kelvin": 2.7255,
                "Neff": 3.046,
                "log10_max_z": 10.0,
                "T_z_representation": 3,
            }
        ],
        "IntegrationSolver": [{"serial": 2, "stepping": 1, "label": "other-solver"}],
        "GkSourcePolicy": [
            {
                "serial": 1,
                "version": 1,
                "label": "policy-a",
                "Levin_threshold": 0.1,
                "numeric_policy": "maximize-numeric",
            },
            {
                "serial": 2,
                "version": 1,
                "label": "policy-b",
                "Levin_threshold": 0.5,
                "numeric_policy": "maximize-WKB",
            },
        ],
        "QuadSourcePolicy": [
            {
                "serial": 1,
                "version": 1,
                "label": "qpolicy-a",
                "Levin_threshold": 0.1,
                "numeric_policy": "maximize-numeric",
            }
        ],
        "wavenumber_exit_time": [
            {
                "serial": 3,
                "version": 1,
                "stepping": 0,
                "wavenumber_serial": 3,
                "cosmology_type": _LAMBDACDM_TYPE,
                "cosmology_serial": 1,
                "atol_serial": 1,
                "rtol_serial": 2,
                "z_exit": 1.0e5,
            },
            {
                "serial": 4,
                "version": 1,
                "stepping": 0,
                "wavenumber_serial": 1,
                "cosmology_type": _QCD_TYPE,
                "cosmology_serial": 1,
                "atol_serial": 1,
                "rtol_serial": 2,
                "z_exit": 3.1e4,
            },
        ],
        "BackgroundModel": [
            {
                "serial": 2,
                "version": 1,
                "label": "fixture-background-qcd",
                "cosmology_type": _QCD_TYPE,
                "cosmology_serial": 1,
                "tau_gauss_order": 6,
                "cs_tau_gauss_order": 6,
                "friction_F_gauss_order": 6,
                "solver_serial": 2,
                "z_init_serial": 1,
                "z_samples": 2,
                "source_grid_digest": "fixture-digest-qcd",
                "source_grid_construction": 1,
                "validated": False,
            }
        ],
        "BackgroundModel_tags": [
            {"model_serial": 1, "tag_serial": 2},
            {"model_serial": 2, "tag_serial": 1},
        ],
        "BackgroundModelValue": [
            {"serial": s, "model_serial": m, "z_serial": z, "Hubble_GeV": 1.0 / s}
            for s, m, z in ((1, 1, 1), (2, 1, 2), (3, 1, 3), (4, 2, 1), (5, 2, 2))
        ],
    },
)


def _values(table: str, parent_column: str, parent: int, serials_and_z) -> RowSet:
    return {
        table: [
            {"serial": serial, parent_column: parent, "z_serial": z}
            for serial, z in serials_and_z
        ]
    }


def _tags(table: str, parent_column: str, parent: int, tags) -> RowSet:
    return {table: [{parent_column: parent, "tag_serial": t} for t in tags]}


def _merge(*row_sets: RowSet) -> RowSet:
    out: RowSet = {}
    for rows in row_sets:
        out = with_rows(out, rows)
    return out


def _full_shard0() -> RowSet:
    return _merge(
        SHARDED_ROWS[0],
        # an unvalidated TkNumericIntegration on the QCD background, with fewer values
        {
            "TkNumericIntegration": [
                {
                    "serial": 3,
                    "version": 1,
                    "label": "fixture-tk-3",
                    "wavenumber_exit_serial": 4,
                    "model_serial": 2,
                    "atol_serial": 1,
                    "rtol_serial": 1,
                    "break_point_kind": _BREAK_POINT_ALL,
                    "solver_serial": 1,
                    "z_init_serial": 1,
                    "z_min_serial": 3,
                    "z_samples": 2,
                    "validated": False,
                }
            ]
        },
        _tags("TkNumeric_tags", "integration_serial", 3, (1, 4)),
        {
            "TkNumericValue": [
                {"serial": 31, "integration_serial": 3, "z_serial": 1, "T": 1.0},
                {"serial": 32, "integration_serial": 3, "z_serial": 2, "T": 0.5},
            ]
        },
        {
            "TkWKBIntegration": [
                {
                    "serial": 1,
                    "version": 1,
                    "label": "fixture-tkwkb-1",
                    "wavenumber_exit_serial": 1,
                    "model_serial": 1,
                    "rho_gauss_order": 10,
                    "phase_solver_serial": 1,
                    "friction_solver_serial": 1,
                    "z_min_serial": 3,
                    "z_samples": 3,
                    "z_init": 100.0,
                    "T_init": 1.0,
                    "Tprime_init": 0.0,
                    "validated": True,
                },
                {
                    "serial": 2,
                    "version": 1,
                    "label": "fixture-tkwkb-2",
                    "wavenumber_exit_serial": 1,
                    "model_serial": 1,
                    "rho_gauss_order": 10,
                    "phase_solver_serial": 1,
                    "friction_solver_serial": 1,
                    "z_min_serial": 3,
                    "z_samples": 1,
                    "z_init": 200.0,
                    "T_init": 1.0,
                    "Tprime_init": 0.0,
                    "validated": True,
                },
            ]
        },
        _tags("TkWKB_tags", "wkb_serial", 1, (1, 2)),
        _tags("TkWKB_tags", "wkb_serial", 2, (1,)),
        _values("TkWKBValue", "wkb_serial", 1, ((101, 1), (102, 2), (103, 3))),
        _values("TkWKBValue", "wkb_serial", 2, ((104, 1),)),
        {
            "GkNumericIntegration": [
                {
                    "serial": 1,
                    "version": 1,
                    "label": "fixture-gknum-1",
                    "wavenumber_exit_serial": 1,
                    "model_serial": 1,
                    "atol_serial": 1,
                    "rtol_serial": 2,
                    "break_point_kind": _BREAK_POINT_ALL,
                    "solver_serial": 1,
                    "z_source_serial": 1,
                    "z_min_serial": 3,
                    "z_samples": 3,
                    "stop_G": 0.1,
                    "validated": True,
                }
            ]
        },
        _tags("GkNumeric_tags", "integration_serial", 1, (1, 2)),
        _values(
            "GkNumericValue", "integration_serial", 1, ((201, 1), (202, 2), (203, 3))
        ),
        {
            "GkWKBIntegration": [
                {
                    "serial": 1,
                    "version": 1,
                    "label": "fixture-gkwkb-1",
                    "wavenumber_exit_serial": 1,
                    "model_serial": 1,
                    "rho_gauss_order": 10,
                    "solver_serial": 1,
                    "z_source_serial": 2,
                    "z_min_serial": 3,
                    "z_samples": 2,
                    "z_init": 50.0,
                    "G_init": 0.0,
                    "Gprime_init": 1.0,
                    "validated": True,
                }
            ]
        },
        _tags("GkWKB_tags", "wkb_serial", 1, (1, 2)),
        _values("GkWKBValue", "wkb_serial", 1, ((301, 2), (302, 3))),
        {
            "GkSource": [
                {
                    "serial": 1,
                    "version": 1,
                    "label": "fixture-gksource-1",
                    "wavenumber_exit_serial": 1,
                    "model_serial": 1,
                    "z_response_serial": 3,
                    "z_max_serial": 1,
                    "z_samples": 3,
                    "validated": True,
                },
                {
                    "serial": 2,
                    "version": 1,
                    "label": "fixture-gksource-2",
                    "wavenumber_exit_serial": 1,
                    "model_serial": 1,
                    "z_response_serial": 2,
                    "z_max_serial": 1,
                    "z_samples": 2,
                    "validated": True,
                },
            ]
        },
        _tags("GkSource_tags", "parent_serial", 1, (1, 2)),
        _tags("GkSource_tags", "parent_serial", 2, (1,)),
        {
            "GkSourceValue": [
                {"serial": s, "parent_serial": p, "z_source_serial": z}
                for s, p, z in ((401, 1, 1), (402, 1, 2), (403, 1, 4))
            ]
            + [
                {"serial": s, "parent_serial": p, "z_source_serial": z}
                for s, p, z in ((404, 2, 1), (405, 2, 4))
            ]
        },
        {
            "GkSourcePolicyData": [
                {
                    "serial": 1,
                    "version": 1,
                    "source_serial": 1,
                    "policy_serial": 1,
                    "wavenumber_exit_serial": 1,
                    "type": 0,
                    "quality": 0,
                    "crossover_z": 20.0,
                },
                {
                    "serial": 2,
                    "version": 1,
                    "source_serial": 2,
                    "policy_serial": 2,
                    "wavenumber_exit_serial": 1,
                    "type": 1,
                    "quality": 1,
                },
            ]
        },
        {
            "QuadSource": [
                {
                    "serial": 1,
                    "version": 1,
                    "label": "fixture-quadsource-1",
                    "model_serial": 1,
                    "q_wavenumber_exit_serial": 1,
                    "r_wavenumber_exit_serial": 2,
                    "Tq_serial": 1,
                    "Tr_serial": 2,
                    "z_samples": 3,
                    "validated": True,
                }
            ]
        },
        _tags("QuadSource_tags", "parent_serial", 1, (1, 2)),
        {
            "QuadSourceValue": [
                {"serial": s, "parent_serial": 1, "z_serial": z}
                for s, z in ((501, 1), (502, 2), (503, 3))
            ]
        },
        {
            "QuadSourceIntegral": [
                {
                    "serial": 1,
                    "version": 1,
                    "label": "fixture-qsi-1",
                    "model_serial": 1,
                    "k_wavenumber_exit_serial": 1,
                    "q_wavenumber_exit_serial": 1,
                    "r_wavenumber_exit_serial": 2,
                    "atol_serial": 1,
                    "rtol_serial": 2,
                    "policy_serial": 1,
                    "source_serial": 1,
                    "data_serial": 1,
                    "z_response_serial": 3,
                    "z_source_max_serial": 1,
                    "b": 0.0,
                    "total": 1.0,
                    "numeric_quad": 0.25,
                    "WKB_Levin": 0.75,
                },
                {
                    "serial": 2,
                    "version": 1,
                    "label": "fixture-qsi-2",
                    "model_serial": 1,
                    "k_wavenumber_exit_serial": 1,
                    "q_wavenumber_exit_serial": 1,
                    "r_wavenumber_exit_serial": 2,
                    "atol_serial": 1,
                    "rtol_serial": 2,
                    "policy_serial": 1,
                    "source_serial": 1,
                    "data_serial": 2,
                    "z_response_serial": 2,
                    "z_source_max_serial": 1,
                    "b": 0.0,
                    "total": 2.0,
                    "numeric_quad": 0.5,
                    "WKB_Levin": 1.5,
                },
            ]
        },
        _tags("QuadSourceIntegral_tags", "parent_serial", 1, (1, 2)),
        _tags("QuadSourceIntegral_tags", "parent_serial", 2, (1,)),
        # OneLoopIntegral 1 shares its serial with QuadSourceIntegral 1, and carries different
        # tags, so reading its tags from the wrong association table is visible
        {
            "OneLoopIntegral": [
                {
                    "serial": 1,
                    "version": 1,
                    "label": "fixture-oneloop-1",
                    "model_serial": 1,
                    "wavenumber_exit_serial": 1,
                    "atol_serial": 1,
                    "rtol_serial": 2,
                    "z_response_serial": 3,
                    "value": 0.5,
                }
            ]
        },
        _tags("OneLoopIntegral_tags", "parent_serial", 1, (1, 5)),
    )


def _full_shard1() -> RowSet:
    return _merge(
        SHARDED_ROWS[1],
        {
            "TkNumericValue": [
                {"serial": 24, "integration_serial": 2, "z_serial": 4, "T": 0.1}
            ]
        },
        {
            "TkWKBIntegration": [
                {
                    "serial": 3,
                    "version": 1,
                    "label": "fixture-tkwkb-3",
                    "wavenumber_exit_serial": 2,
                    "model_serial": 1,
                    "rho_gauss_order": 10,
                    "phase_solver_serial": 1,
                    "friction_solver_serial": 1,
                    "z_min_serial": 3,
                    "z_samples": 2,
                    "z_init": 100.0,
                    "T_init": 1.0,
                    "Tprime_init": 0.0,
                    "validated": True,
                }
            ]
        },
        _tags("TkWKB_tags", "wkb_serial", 3, (1,)),
        _values("TkWKBValue", "wkb_serial", 3, ((105, 1), (106, 2))),
        {
            "GkNumericIntegration": [
                {
                    "serial": 2,
                    "version": 1,
                    "label": "fixture-gknum-2",
                    "wavenumber_exit_serial": 2,
                    "model_serial": 1,
                    "atol_serial": 1,
                    "rtol_serial": 2,
                    "break_point_kind": _BREAK_POINT_ALL,
                    "solver_serial": 1,
                    "z_source_serial": 1,
                    "z_min_serial": 3,
                    "z_samples": 1,
                    "validated": True,
                }
            ]
        },
        _tags("GkNumeric_tags", "integration_serial", 2, (1,)),
        _values("GkNumericValue", "integration_serial", 2, ((204, 1),)),
        {
            "GkWKBIntegration": [
                {
                    "serial": 2,
                    "version": 1,
                    "label": "fixture-gkwkb-2",
                    "wavenumber_exit_serial": 2,
                    "model_serial": 1,
                    "rho_gauss_order": 10,
                    "solver_serial": 1,
                    "z_source_serial": 2,
                    "z_min_serial": 3,
                    "z_samples": 1,
                    "z_init": 50.0,
                    "G_init": 0.0,
                    "Gprime_init": 1.0,
                    "validated": True,
                }
            ]
        },
        _tags("GkWKB_tags", "wkb_serial", 2, (1,)),
        _values("GkWKBValue", "wkb_serial", 2, ((303, 3),)),
        {
            "GkSource": [
                {
                    "serial": 3,
                    "version": 1,
                    "label": "fixture-gksource-3",
                    "wavenumber_exit_serial": 2,
                    "model_serial": 1,
                    "z_response_serial": 3,
                    "z_max_serial": 1,
                    "z_samples": 1,
                    "validated": True,
                }
            ]
        },
        _tags("GkSource_tags", "parent_serial", 3, (1, 2)),
        {"GkSourceValue": [{"serial": 406, "parent_serial": 3, "z_source_serial": 1}]},
        {
            "GkSourcePolicyData": [
                {
                    "serial": 3,
                    "version": 1,
                    "source_serial": 3,
                    "policy_serial": 1,
                    "wavenumber_exit_serial": 2,
                    "type": 0,
                    "quality": 0,
                }
            ]
        },
        # an unvalidated QuadSource
        {
            "QuadSource": [
                {
                    "serial": 2,
                    "version": 1,
                    "label": "fixture-quadsource-2",
                    "model_serial": 1,
                    "q_wavenumber_exit_serial": 2,
                    "r_wavenumber_exit_serial": 3,
                    "Tq_serial": 2,
                    "Tr_serial": 3,
                    "z_samples": 2,
                    "validated": False,
                }
            ]
        },
        _tags("QuadSource_tags", "parent_serial", 2, (1,)),
        {
            "QuadSourceValue": [
                {"serial": s, "parent_serial": 2, "z_serial": z}
                for s, z in ((504, 1), (505, 2))
            ]
        },
        {
            "QuadSourceIntegral": [
                {
                    "serial": 3,
                    "version": 1,
                    "label": "fixture-qsi-3",
                    "model_serial": 1,
                    "k_wavenumber_exit_serial": 2,
                    "q_wavenumber_exit_serial": 2,
                    "r_wavenumber_exit_serial": 3,
                    "atol_serial": 1,
                    "rtol_serial": 2,
                    "policy_serial": 1,
                    "source_serial": 2,
                    "data_serial": 3,
                    "z_response_serial": 3,
                    "z_source_max_serial": 1,
                    "b": 0.0,
                    "total": 3.0,
                    "numeric_quad": 1.0,
                    "WKB_Levin": 2.0,
                }
            ]
        },
        _tags("QuadSourceIntegral_tags", "parent_serial", 3, (1,)),
    )


# shard serial -> sharded rows. Wavenumber 1 (exits 1 and 4) is on shard 0; wavenumbers 2 and 3
# (exits 2 and 3) are on shard 1. QuadSource is sharded by q, the others by k
FULL_SHARDED_ROWS: Dict[int, RowSet] = {0: _full_shard0(), 1: _full_shard1()}

FULL_SHARD_KEYS: Dict[int, int] = {1: 0, 2: 1, 3: 1}


_SCHEMA: Dict[str, Mapping[str, sqla.Table]] = {}


def _schema_tables() -> Mapping[str, sqla.Table]:
    """The ``build_schema`` tables, built once per process. Read, never created from."""
    if "tables" not in _SCHEMA:
        _SCHEMA["tables"] = build_schema(sqla.MetaData(), _factories).tables
    return _SCHEMA["tables"]


def _placeholder(column: sqla.Column):
    kind = column.type
    if isinstance(kind, sqla.Boolean):
        return False
    if isinstance(kind, sqla.Integer):
        return 0
    if isinstance(kind, sqla.Float):
        return 0.5
    if isinstance(kind, sqla.String):
        return "fixture"
    if isinstance(kind, sqla.DateTime):
        return FIXED_TIMESTAMP
    raise TypeError(f"no placeholder for column {column!r}")


def fill_required(rows: RowSet) -> RowSet:
    """A copy of ``rows`` in which every non-nullable, non-key column a row leaves out holds a
    placeholder of its type, and a nullable column that another row of the same table states is
    ``None`` (one insert takes one set of columns). Columns a row states are never changed.
    """
    tables = _schema_tables()
    out = copy.deepcopy(rows)
    for name, table_rows in out.items():
        table = tables[name]
        stated = {c for row in table_rows for c in row}
        for row in table_rows:
            for column in table.columns:
                if column.name in row or column.primary_key:
                    continue
                if not column.nullable:
                    row[column.name] = _placeholder(column)
                elif column.name in stated and column.name != "timestamp":
                    row[column.name] = None
    return out


def full_rows() -> Tuple[RowSet, Dict[int, RowSet], Dict[int, int]]:
    """Fresh copies of the full store's replicated rows, sharded rows and shard keys."""
    return (
        copy.deepcopy(FULL_REPLICATED_ROWS),
        copy.deepcopy(FULL_SHARDED_ROWS),
        dict(FULL_SHARD_KEYS),
    )


_EMPTY_SHARD: Dict[str, bytes] = {}


def _empty_shard_bytes() -> bytes:
    """An empty shard file holding every ``build_schema`` table, made once per process. Creating
    37 tables commits once per statement, so a copy of this is much faster than ``create_all``.
    """
    if "bytes" not in _EMPTY_SHARD:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.sqlite"
            metadata = sqla.MetaData()
            build_schema(metadata, _factories)
            engine = sqla.create_engine(f"sqlite:///{path}", future=True)
            try:
                metadata.create_all(engine)
            finally:
                engine.dispose()
            _EMPTY_SHARD["bytes"] = path.read_bytes()
    return _EMPTY_SHARD["bytes"]


def _write_full_shard(
    path: Path,
    rows: List[RowSet],
    missing_tables: Iterable[str],
    missing_columns: Mapping[str, Iterable[str]],
    extra_sql: Iterable[str],
) -> None:
    """As ``_write_shard``, from a copy of the empty shard. Rows for a missing table are skipped
    (replicated rows are written to every shard), and the table is then dropped."""
    missing_tables = set(missing_tables)
    path.write_bytes(_empty_shard_bytes())
    tables = _schema_tables()
    engine = sqla.create_engine(f"sqlite:///{path}", future=True)
    try:
        with engine.begin() as conn:
            for row_set in rows:
                _insert_rows(
                    conn,
                    tables,
                    {t: r for t, r in row_set.items() if t not in missing_tables},
                )
    finally:
        engine.dispose()

    statements = (
        [f'DROP TABLE "{table}"' for table in sorted(missing_tables)]
        + [
            f'ALTER TABLE "{table}" DROP COLUMN "{column}"'
            for table, columns in missing_columns.items()
            for column in columns
        ]
        + list(extra_sql)
    )
    if len(statements) > 0:
        conn = sqlite3.connect(path)
        try:
            with conn:
                for statement in statements:
                    conn.execute(statement)
        finally:
            conn.close()


def build_full_store(
    directory: Path,
    stem: str = "full-store",
    shards: int = 2,
    replicated: Optional[RowSet] = None,
    sharded: Optional[Dict[int, RowSet]] = None,
    shard_keys: Optional[Mapping[int, int]] = None,
    missing_tables: Optional[Mapping[int, Iterable[str]]] = None,
    missing_columns: Optional[Mapping[int, Mapping[str, Iterable[str]]]] = None,
    extra_sql: Optional[Mapping[int, Iterable[str]]] = None,
) -> RealStore:
    """
    A store like ``build_real_store``'s, with the full store's rows (or the ones given), through
    ``fill_required``. Its primary is written the same way. Its shards are copies of one empty
    shard, filled with the rows; a table in ``missing_tables[n]`` is dropped from shard ``n``
    after the rows are written, and its rows are skipped there.
    """
    base_replicated, base_sharded, base_keys = full_rows()
    replicated = fill_required(base_replicated if replicated is None else replicated)
    sharded = {
        n: fill_required(rows)
        for n, rows in (base_sharded if sharded is None else sharded).items()
    }
    shard_keys = dict(base_keys if shard_keys is None else shard_keys)
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
        _write_full_shard(
            path,
            [replicated, sharded.get(serial, {})],
            missing_tables.get(serial, ()),
            missing_columns.get(serial, {}),
            extra_sql.get(serial, ()),
        )
    _write_primary(primary, shard_files, shard_keys)

    return RealStore(
        primary=primary, shard_files=shard_files, replicated=replicated, sharded=sharded
    )


# references that the schema does not declare as foreign keys: table -> column -> target table.
# The cosmology references are polymorphic, and are resolved through cosmology_type
_UNDECLARED_REFERENCES = {
    "GkSourcePolicyData": {"wavenumber_exit_serial": "wavenumber_exit_time"},
    "QuadSourceIntegral": {"source_serial": "QuadSource"},
    "QuadSource": {"Tr_serial": "QuadSource"},
}
_COSMOLOGY_TABLES = {_LAMBDACDM_TYPE: "LambdaCDM", _QCD_TYPE: "QCD_Cosmology"}


def references() -> Dict[str, Dict[str, str]]:
    """table -> column -> the table whose serial it holds: the schema's foreign keys, plus the
    references the schema does not declare. ``cosmology_serial`` maps to ``"cosmology"``.
    """
    out: Dict[str, Dict[str, str]] = {}
    for name, table in _schema_tables().items():
        refs = {}
        for column in table.columns:
            for fk in column.foreign_keys:
                refs[column.name] = fk.column.table.name
        if "cosmology_serial" in table.c:
            refs["cosmology_serial"] = "cosmology"
        refs.update(_UNDECLARED_REFERENCES.get(name, {}))
        out[name] = refs
    return out


def relabel_serials(
    replicated: RowSet,
    sharded: Dict[int, RowSet],
    shard_keys: Mapping[int, int],
    offset: int = 100,
) -> Tuple[RowSet, Dict[int, RowSet], Dict[int, int]]:
    """
    The same content under different serials. In every table the serials are reversed in order
    and moved by a per-table offset, and every reference follows them. Replicated rows get the
    same new serials on every shard, as ``ShardedPool`` replicates them.
    """
    replicated = copy.deepcopy(replicated)
    sharded = copy.deepcopy(sharded)
    row_sets = [replicated] + [sharded[n] for n in sorted(sharded)]

    olds: Dict[str, List[int]] = {}
    for rows in row_sets:
        for table, table_rows in rows.items():
            for row in table_rows:
                if "serial" in row:
                    olds.setdefault(table, []).append(row["serial"])

    mapping: Dict[str, Dict[int, int]] = {}
    for index, table in enumerate(sorted(olds)):
        serials = sorted(set(olds[table]))
        base = offset * (index + 1)
        mapping[table] = {
            old: base + len(serials) - position for position, old in enumerate(serials)
        }

    refs = references()
    for rows in row_sets:
        for table, table_rows in rows.items():
            for row in table_rows:
                if "serial" in row:
                    row["serial"] = mapping[table][row["serial"]]
                for column, target in refs.get(table, {}).items():
                    if column not in row:
                        continue
                    if target == "cosmology":
                        target = _COSMOLOGY_TABLES[row["cosmology_type"]]
                    row[column] = mapping.get(target, {}).get(row[column], row[column])

    new_keys = {
        mapping["wavenumber"].get(key, key): shard for key, shard in shard_keys.items()
    }
    return replicated, sharded, new_keys


def find_row(
    replicated: RowSet, sharded: Dict[int, RowSet], table: str, serial: int
) -> Dict[str, object]:
    """The one row of ``table`` with ``serial``, wherever it is."""
    found = [
        row
        for rows in [replicated] + list(sharded.values())
        for row in rows.get(table, [])
        if row.get("serial") == serial
    ]
    if len(found) != 1:
        raise KeyError(f"{len(found)} rows of {table} have serial {serial}")
    return found[0]


def vary_row(
    replicated: RowSet,
    sharded: Dict[int, RowSet],
    table: str,
    serial: int,
    column: str,
    value: object,
) -> None:
    """Set ``column`` of the row of ``table`` with ``serial`` to ``value``, in place."""
    find_row(replicated, sharded, table, serial)[column] = value
