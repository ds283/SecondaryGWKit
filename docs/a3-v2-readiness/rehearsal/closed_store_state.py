"""The state of a closed store between launches (a3-v2-readiness prompt 07, §1.5 item 4).

Read-only throughout: every file is opened with `sqlite3` `mode=ro`, and nothing is written. For
the primary and each shard it lists any `-journal`, `-wal` or `-shm` file beside it, the rows of the
primary's `replication_in_flight`, each file's `version` rows, and, per file, every table with a
`validated` column: its row count, the count of rows with `validated = 0`, and their serials.

    PYTHONPATH=. ./venv/bin/python closed_store_state.py var/datastores/<store>.sqlite
"""

import sqlite3
import sys
from pathlib import Path


def connect_ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def main(primary):
    primary = Path(primary).resolve()
    conn = connect_ro(primary)
    try:
        shards = sorted(conn.execute("SELECT serial, filename FROM shards"))
        in_flight_cols = [
            r[1] for r in conn.execute("PRAGMA table_info(replication_in_flight)")
        ]
        in_flight = list(conn.execute("SELECT * FROM replication_in_flight"))
    finally:
        conn.close()
    files = [("primary", primary)] + [
        (f"shard {serial}", primary.with_name(Path(name).name))
        for serial, name in shards
    ]

    print(f"store {primary}")
    print("-- journal, WAL and shared-memory files beside the store's files:")
    found = []
    for _, path in files:
        for suffix in ("-journal", "-wal", "-shm"):
            side = path.with_name(path.name + suffix)
            if side.exists():
                found.append(side.name)
    print("   " + (", ".join(found) if found else "none"))

    print(
        f"-- primary's replication_in_flight ({', '.join(in_flight_cols)}): {len(in_flight)} row(s)"
    )
    for row in in_flight:
        print(f"   {row}")

    for label, path in files:
        conn = connect_ro(path)
        try:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
                )
            ]
            version = (
                list(conn.execute("SELECT serial, label FROM version"))
                if "version" in tables
                else None
            )
            print(f"-- {label}: {path.name}; version rows {version}")
            for table in tables:
                cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
                if "validated" not in cols:
                    continue
                total = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                serials = [
                    r[0]
                    for r in conn.execute(
                        f'SELECT serial FROM "{table}" WHERE validated = 0 ORDER BY serial'
                    )
                ]
                print(
                    f"   {table}: {total} row(s), {len(serials)} with validated = 0"
                    + (f": serials {serials}" if serials else "")
                )
        finally:
            conn.close()


if __name__ == "__main__":
    main(sys.argv[1])
