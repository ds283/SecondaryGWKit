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

IMPORTANT -- there is no safe automated repair. Reconstructing the correct
map requires knowing the order in which shard-key objects were originally
assigned, which is not recorded anywhere. If this tool reports a problem,
the remedy is to rebuild the datastore from scratch, not to patch it in
place.

Usage:
    python tools/shard_key_audit.py /path/to/primary.sqlite
"""

import sqlite3
import sys
from pathlib import Path
from typing import List


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

    if "shard_keys" not in tables or "shards" not in tables:
        print(
            f"!! {primary_path} does not look like a ShardedPool primary database "
            f"(expected tables 'shard_keys' and 'shards'; found {sorted(tables)})"
        )
        return 2

    shard_keys_columns = _table_columns(conn, "shard_keys")

    if "key_serial" not in shard_keys_columns:
        if "wavenumber_serial" in shard_keys_columns:
            print(
                "!! This datastore uses the pre-a2bd966 schema: the shard_keys "
                "primary key column is 'wavenumber_serial', not 'key_serial'. It "
                "predates the shard_key_config / replicated_tables / sharded_tables "
                "refactor and is not readable by the current ShardedPool code."
            )
        else:
            print(
                f"!! Unrecognised shard_keys schema (columns: {shard_keys_columns}). "
                f"This does not look like a current-generation ShardedPool datastore."
            )
        print(
            "!! There is no safe automated upgrade path. This datastore must be "
            "rebuilt from scratch; this tool will not attempt to audit it further."
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
    key_type = None
    if "shard_key_config" in tables:
        row = conn.execute("SELECT key_type FROM shard_key_config").fetchone()
        if row is not None:
            key_type = row[0]

    shard_files = list(
        conn.execute("SELECT serial, filename FROM shards ORDER BY serial")
    )

    cross_file_done = False
    if key_type is not None and shard_files:
        shard_serial, shard_filename = shard_files[0]
        shard_path = Path(shard_filename)
        if shard_path.exists():
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
                    print(
                        f"!! shard #{shard_serial} database does not contain a "
                        f"'{key_type}' table; skipping cross-file checks"
                    )
            except sqlite3.OperationalError as e:
                print(f"!! could not attach shard #{shard_serial} ({shard_path}): {e}")
        else:
            print(f"!! shard file does not exist on disk: {shard_path}")

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
