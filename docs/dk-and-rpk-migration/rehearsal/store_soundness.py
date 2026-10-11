"""Integrity, foreign keys, version rows and the inventory of each closed store.

    PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/store_soundness.py \\
        var/datastores/<store>.sqlite [...]

For each primary given: on the primary and on every shard, opened `mode=ro`, `PRAGMA
integrity_check`, `PRAGMA foreign_key_check` and `SELECT serial, label FROM version`; then
`read_inventory(primary, factories)` (every file `mode=ro`, no Ray), given the registry of storable
classes from `config.datastore`, with each class's record count and problems, and whether any
problem is a `duplicate`. The shards are read from the primary's own `shards` table. It writes
nothing.

It is `docs/a3-v2-readiness/rehearsal/store_soundness.py` with two changes, and that record is left
as it is. That script calls `read_inventory(primary)`, which raises `TypeError` since `19f07ee`
gave the inventory its registry as an argument; this one passes `factories` exactly as
`RunRegistry.stores.fingerprint_store` does. And that one globs for shard files; this one reads the
`shards` table.

Exit status: 0 if every check is clean on every store; 1 otherwise.
"""

import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datastorekit.store_inventory import read_inventory  # noqa: E402
from config.datastore import factories  # noqa: E402


def pragmas(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        integrity = [r[0] for r in conn.execute("PRAGMA integrity_check")]
        foreign = list(conn.execute("PRAGMA foreign_key_check"))
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        version = (
            list(conn.execute("SELECT serial, label FROM version"))
            if "version" in tables
            else "no table"
        )
    finally:
        conn.close()
    return integrity, foreign, version


def shard_files(primary: Path):
    conn = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
    try:
        rows = sorted(conn.execute("SELECT serial, filename FROM shards"))
    finally:
        conn.close()
    return [primary.with_name(Path(name).name) for _, name in rows]


def main(primaries) -> int:
    if not primaries:
        print(__doc__, file=sys.stderr)
        return 1
    clean = True
    for primary in primaries:
        primary = Path(primary).resolve()
        print(f"== {primary.name}")
        for path in [primary] + shard_files(primary):
            integrity, foreign, version = pragmas(path)
            print(
                f"  {path.name}: integrity_check {integrity}; foreign_key_check "
                f"{len(foreign)} row(s){(' ' + repr(foreign[:5])) if foreign else ''}; version {version}"
            )
            clean = clean and integrity == ["ok"] and not foreign
        inventory = read_inventory(primary, factories)
        print(f"  read_inventory: shards {inventory.shards}")
        for name, cls in inventory.classes.items():
            print(
                f"    {name:24s} {'replicated' if cls.replicated else 'sharded   '} "
                f"count {cls.count:6d}  problems {list(cls.problems) or 'none'}"
            )
        duplicates = [p for p in inventory.problems if "duplicate" in p]
        print(
            f"  problems in any class: {len(inventory.problems)}; of them 'duplicate': {len(duplicates)}"
        )
        clean = clean and not inventory.problems
    print(f"=> {'clean' if clean else 'NOT CLEAN'}")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
