"""Prompt 07 §2.2: integrity, foreign keys, version rows and the inventory of each closed store.

For each primary given: on the primary and on every shard, opened `mode=ro`, `PRAGMA
integrity_check`, `PRAGMA foreign_key_check` and `SELECT serial, label FROM version`; then
`Datastore.store_inventory.read_inventory(primary)` (every file `mode=ro`, no Ray), with each
class's record count and problems, and whether any problem is a `duplicate`. It writes nothing.

    PYTHONPATH=. ./venv/bin/python docs/a3-v2-readiness/rehearsal/store_soundness.py <primary> ...
"""

import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Datastore.store_inventory import read_inventory  # noqa: E402


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


def main(primaries) -> int:
    clean = True
    for primary in primaries:
        primary = Path(primary).resolve()
        print(f"== {primary.name}")
        files = [primary] + sorted(
            primary.parent.glob(f"{primary.stem}-shard*{primary.suffix}")
        )
        for path in files:
            integrity, foreign, version = pragmas(path)
            print(
                f"  {path.name}: integrity_check {integrity}; foreign_key_check "
                f"{len(foreign)} row(s){(' ' + repr(foreign[:5])) if foreign else ''}; version {version}"
            )
            clean = clean and integrity == ["ok"] and not foreign
        inventory = read_inventory(primary)
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
