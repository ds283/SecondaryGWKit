"""Are the rows a killed launch left `validated = 0` gone after the next launch's open?

    ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/prune_check.py \\
        <closed_store_state output after the killed launch> var/datastores/<store>.sqlite

The first argument is the saved output of
`docs/a3-v2-readiness/rehearsal/closed_store_state.py` on the same store, taken after the killed
launch and before the next. From it this reads every `<class>: N row(s), M with validated = 0:
serials [...]` line, under its `-- shard <serial>:` heading. Nothing is hardcoded: 07's
`prune_serials.py` named its own serials, so it could check no other store.

It then opens the store's current shards `mode=ro` (sqlite3, read-only; nothing is written), read
from the primary's own `shards` table, and for every such serial prints the row now at that serial
in that class's table: `absent`, or `now (serial, validated, timestamp, label)`. A pruned row's
serial is either absent or reused by a row stored later, whose label carries the later time, and
which must then be validated. Per class it also prints the rows that are `validated = 0` now.

Exit status: 0 if no listed serial now holds a row with `validated = 0`; 1 if one does; 2 on a
usage error or an input with no shard heading.
"""

import re
import sqlite3
import sys
from pathlib import Path

SHARD = re.compile(r"^-- shard (\d+): ")
UNVALIDATED = re.compile(
    r"^\s+(\w+): \d+ row\(s\), (\d+) with validated = 0: serials \[([\d, ]*)\]\s*$"
)


def listed(state_text):
    """{shard serial: {class: [serials]}} from closed_store_state.py's output."""
    found, shard, headings = {}, None, 0
    for line in state_text.splitlines():
        heading = SHARD.match(line)
        if heading:
            shard = int(heading.group(1))
            headings += 1
            continue
        match = UNVALIDATED.match(line)
        if match and shard is not None:
            serials = [int(s) for s in match.group(3).split(",") if s.strip()]
            found.setdefault(shard, {})[match.group(1)] = serials
    return found, headings


def main(argv) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    state_file, primary = Path(argv[0]), Path(argv[1]).resolve()
    found, headings = listed(state_file.read_text())
    if headings == 0:
        print(f"{state_file}: no '-- shard N:' heading", file=sys.stderr)
        return 2
    conn = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
    try:
        shards = dict(conn.execute("SELECT serial, filename FROM shards"))
    finally:
        conn.close()

    print(f"store {primary}")
    print(f"serials listed unvalidated in {state_file}")
    if not found:
        print("   none: there is nothing to check")
        return 0
    checked = still_unvalidated = reused = 0
    for shard in sorted(found):
        path = primary.with_name(Path(shards[shard]).name)
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            for cls, serials in sorted(found[shard].items()):
                now = conn.execute(
                    f'SELECT COUNT(*) FROM "{cls}" WHERE validated = 0'
                ).fetchone()[0]
                print(
                    f"-- shard {shard}, {cls}: {len(serials)} serial(s) listed; "
                    f"{now} row(s) with validated = 0 now"
                )
                for serial in serials:
                    checked += 1
                    row = conn.execute(
                        f'SELECT serial, validated, timestamp, label FROM "{cls}" '
                        "WHERE serial = ?",
                        (serial,),
                    ).fetchone()
                    if row is None:
                        print(f"   serial {serial}: absent")
                        continue
                    if row[1]:
                        reused += 1
                    else:
                        still_unvalidated += 1
                    print(f"   serial {serial}: now {row}")
        finally:
            conn.close()
    print(
        f"{checked} serial(s) checked: {checked - reused - still_unvalidated} absent, "
        f"{reused} reused by a validated row, {still_unvalidated} still unvalidated"
    )
    return 1 if still_unvalidated else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
