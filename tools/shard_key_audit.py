#!/usr/bin/env python3
"""
Read-only consistency check for a ShardedPool datastore's shard_keys table.

ShardedPool keeps a shard-key -> shard-id map in two places: in memory
(self._shard_keys, rebuilt on every startup from disk) and on disk, in the
`shard_keys` table of the primary SQLite file, whose primary key column is
`key_serial`. If that column is ever populated with a value other than the
corresponding shard-key object's `store_id` (see the B1 fix in
ShardedPool._assign_shard_keys), the map rebuilt on the next startup diverges
from the one used when records were written, and every record filed under a
displaced key becomes permanently unreachable. This tool detects that
divergence after the fact; it does not prevent it.

This script only ever opens databases read-only (sqlite3 `mode=ro` URIs). It
is structurally incapable of modifying a datastore.

The shard file it attaches for the cross-file check is found exactly as
ShardedPool finds it, by the one resolver in Datastore/shard_paths.py: the
`shards` record's file name, in the primary's own directory. A record that is
not a bare file name, an absolute path included, is refused, so auditing a
copied store can never check the original's shard in place of the copy's.
That module imports only the standard library and is outside the
Datastore.SQL package, so importing it does not pull in ray or sqlalchemy;
the repository root is put on sys.path below so that the script still runs
standalone, from any directory, with no PYTHONPATH.

IMPORTANT -- there is no safe automated repair. Reconstructing the correct
map requires knowing the order in which shard-key objects were originally
assigned, which is not recorded anywhere. If this tool reports a problem,
the remedy is to rebuild the datastore from scratch, not to patch it in
place.

Exit status: 0 for "VERDICT: OK", 1 for "VERDICT: INCONSISTENT", and 2 when
the store cannot be audited at all (it is refused, and no verdict is given):
a primary that is not a completely written current-generation ShardedPool
primary, a shard record that is not a bare file name, or a shard 0 that
exists but cannot be read as a shard (not a database, or without the
shard-key table its primary names). A shard 0 file that is missing is not a
refusal: the cross-file check is skipped and said to be, and the single-file
checks still give a verdict.

Usage:
    python tools/shard_key_audit.py /path/to/primary.sqlite
"""

import sqlite3
import sys
from pathlib import Path
from typing import List

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Datastore.shard_paths import resolve_shard_path, shard_file_problem


def _open_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def _table_names(conn: sqlite3.Connection) -> set:
    return {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <path-to-primary-database>")
        return 2

    primary_path = Path(argv[1]).resolve()
    if not primary_path.exists():
        print(f"!! No such file: {primary_path}")
        return 2

    try:
        conn = _open_readonly(primary_path)
        tables = _table_names(conn)
    except sqlite3.OperationalError as e:
        print(f"!! Could not open {primary_path} read-only: {e}")
        return 2

    if not {"shard_keys", "shards", "shard_key_config"} <= tables:
        print(
            f"!! {primary_path} does not look like a completely written ShardedPool "
            f"primary database (expected tables 'shard_keys', 'shards' and "
            f"'shard_key_config'; found {sorted(tables)})"
        )
        return 2

    shard_keys_columns = _table_columns(conn, "shard_keys")

    if "key_serial" not in shard_keys_columns:
        print(
            f"!! Unrecognised shard_keys schema (columns: {shard_keys_columns}). "
            f"This does not look like a current-generation ShardedPool datastore."
        )
        print(
            "!! There is no safe automated upgrade path. This datastore must be "
            "rebuilt from scratch; this tool will not attempt to audit it further."
        )
        return 2

    # every other column this tool reads must be there too: a primary without one is refused,
    # not met with a traceback, whose exit code 1 would read as INCONSISTENT
    absent_columns = [
        f"{table}.{column}"
        for table, columns in (
            ("shards", ("serial", "filename")),
            ("shard_keys", ("shard_id",)),
            ("shard_key_config", ("key_type",)),
        )
        for column in columns
        if column not in _table_columns(conn, table)
    ]
    if absent_columns:
        print(
            f"!! {primary_path} lacks the column(s) {absent_columns}. This does not look "
            f"like a current-generation ShardedPool primary database; this tool will not "
            f"attempt to audit it."
        )
        return 2

    config_row = conn.execute("SELECT key_type FROM shard_key_config").fetchone()
    if config_row is None:
        print(
            f"!! The shard_key_config table of {primary_path} holds no row. This is "
            f"not a completely written ShardedPool primary database; this tool will "
            f"not attempt to audit it."
        )
        return 2
    key_type = config_row[0]

    # every shard record goes through the one resolver, which accepts a bare file name only. An
    # unusable record is a refusal, not a verdict: the store cannot be opened by ShardedPool.
    shard_files = list(
        conn.execute("SELECT serial, filename FROM shards ORDER BY serial")
    )
    shard_paths = {}
    for shard_serial, shard_filename in shard_files:
        try:
            shard_paths[shard_serial] = resolve_shard_path(primary_path, shard_filename)
        except ValueError as e:
            print(f"!! shard #{shard_serial} record is unusable: {e}")
            print(
                "!! A ShardedPool primary records each shard by its bare file name. "
                "This tool will not attempt to audit this datastore."
            )
            return 2

    print(f">> {primary_path}: current-generation schema (key_serial present)")

    problems: List[str] = []

    # Referential integrity of shard_id against the shards table.
    shard_ids = {row[0] for row in conn.execute("SELECT serial FROM shards")}
    bad_shard_refs = list(
        conn.execute(
            "SELECT key_serial, shard_id FROM shard_keys "
            f"WHERE shard_id NOT IN ({','.join('?' * len(shard_ids)) or 'NULL'})",
            tuple(shard_ids),
        )
    )
    if bad_shard_refs:
        problems.append(
            f"{len(bad_shard_refs)} shard_keys row(s) reference a non-existent shard_id: "
            f"{bad_shard_refs[:10]}{' ...' if len(bad_shard_refs) > 10 else ''}"
        )

    # Monotonicity / duplicate check on key_serial itself (cheap, single-file).
    key_serials = [row[0] for row in conn.execute("SELECT key_serial FROM shard_keys")]
    if len(key_serials) != len(set(key_serials)):
        problems.append("duplicate key_serial values found in shard_keys")

    # Per-shard distribution -- always reportable from the primary file alone.
    distribution = dict(
        conn.execute(
            "SELECT shard_id, COUNT(*) FROM shard_keys GROUP BY shard_id ORDER BY shard_id"
        )
    )
    print(f">> shard_keys row count: {len(key_serials)}")
    print(f">> per-shard key distribution: {distribution}")

    # Cross-file check against the actual shard-key table (e.g. "wavenumber"),
    # which lives in the replicated tables inside each shard database, not in
    # the primary file. Best-effort: attach one shard file read-only.
    cross_file_done = False
    if key_type is not None and shard_files:
        shard_serial, shard_filename = shard_files[0]
        shard_path = shard_paths[shard_serial]
        shard_problem = shard_file_problem(shard_path)
        if shard_problem is None:
            print(
                f">> cross-file check against shard #{shard_serial}: {shard_path} "
                f"(record {shard_filename!r})"
            )
            try:
                conn.execute(f"ATTACH DATABASE 'file:{shard_path}?mode=ro' AS shard0")
                shard_tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM shard0.sqlite_master WHERE type='table'"
                    )
                }
                if key_type in shard_tables:
                    key_table_serials = {
                        row[0]
                        for row in conn.execute(f"SELECT serial FROM shard0.{key_type}")
                    }
                    shard_key_serials = set(key_serials)

                    orphaned = sorted(shard_key_serials - key_table_serials)
                    unassigned = sorted(key_table_serials - shard_key_serials)

                    print(
                        f">> '{key_type}' table (shard #{shard_serial}) row count: "
                        f"{len(key_table_serials)}"
                    )
                    if orphaned:
                        problems.append(
                            f"{len(orphaned)} shard_keys.key_serial value(s) have no "
                            f"corresponding '{key_type}' row (orphaned): "
                            f"{orphaned[:10]}{' ...' if len(orphaned) > 10 else ''}"
                        )
                    if unassigned:
                        print(
                            f">> {len(unassigned)} '{key_type}' row(s) have no shard_keys "
                            f"entry (unassigned, not necessarily an error): "
                            f"{unassigned[:10]}{' ...' if len(unassigned) > 10 else ''}"
                        )
                    cross_file_done = True
                else:
                    # every shard holds the shard-key table, which is replicated: a shard 0
                    # without it is not a completely written shard, and cannot be audited
                    print(
                        f"!! shard #{shard_serial} database {shard_path} does not contain "
                        f"the '{key_type}' table that shard_key_config names. This tool "
                        f"will not attempt to audit this datastore."
                    )
                    return 2
            except sqlite3.DatabaseError as e:
                # OperationalError included: the file is there, but is not a readable shard
                print(
                    f"!! shard #{shard_serial} file {shard_path} cannot be read as a shard "
                    f"database: {e}. This tool will not attempt to audit this datastore."
                )
                return 2
        else:
            print(f"!! shard #{shard_serial} file {shard_path} {shard_problem}")

    if not cross_file_done:
        print(
            ">> Cross-file check against the shard-key table was not possible; "
            "only single-file checks (count, duplicates, shard_id referential "
            "integrity) were performed."
        )

    if problems:
        print("!! INCONSISTENT:")
        for p in problems:
            print(f"   - {p}")
        print(f"VERDICT: INCONSISTENT -- {primary_path} needs to be rebuilt.")
        return 1

    print(f"VERDICT: OK -- no inconsistency found in {primary_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
