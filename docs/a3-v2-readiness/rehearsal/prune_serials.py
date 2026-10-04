"""Prompt 07 §2.2: are the GkSource rows `resumed-a` left `validated = 0` gone after the next open?

Reads each shard `mode=ro` and, for every serial found unvalidated after `resumed-a`
(`var/a3-v2-readiness-07/state/after-resumed-a.txt`), prints the row now at that serial, if any,
with its timestamp and label. A pruned row's serial is either absent or reused by a row stored
later (its label carries the later timestamp). It also counts, per shard, the GkSource rows and the
distinct keys `(wavenumber_exit_serial, model_serial, z_response_serial, parents_digest)`, which
must be equal if nothing is duplicated.

    python docs/a3-v2-readiness/rehearsal/prune_serials.py var/datastores/<store>.sqlite
"""

import sqlite3
import sys
from pathlib import Path

UNVALIDATED_AFTER_RESUMED_A = {
    0: [11, 12, 13, 14, 15, 22, 28, 29, 30, 31, 33, 34],
    1: [214, 215, 216, 217, 218, 219, 228],
    3: [522, 523, 528, 529, 530, 531, 539, 540, 541, 542, 543, 544, 545, 547, 548, 549],
}
KEY = "wavenumber_exit_serial, model_serial, z_response_serial, parents_digest"


def main(primary):
    primary = Path(primary).resolve()
    total = 0
    for shard in range(4):
        path = primary.with_name(f"{primary.stem}-shard{shard:04d}{primary.suffix}")
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT COUNT(*) FROM GkSource").fetchone()[0]
            keys = conn.execute(
                f"SELECT COUNT(*) FROM (SELECT DISTINCT {KEY} FROM GkSource)"
            ).fetchone()[0]
            unvalidated = conn.execute(
                "SELECT COUNT(*) FROM GkSource WHERE validated = 0"
            ).fetchone()[0]
            print(
                f"shard {shard}: {rows} GkSource rows, {keys} distinct keys, {unvalidated} with validated = 0"
            )
            for serial in UNVALIDATED_AFTER_RESUMED_A.get(shard, []):
                total += 1
                row = conn.execute(
                    "SELECT serial, validated, timestamp, label FROM GkSource WHERE serial = ?",
                    (serial,),
                ).fetchone()
                print(
                    f"   serial {serial}: "
                    + ("absent" if row is None else f"now {row}")
                )
        finally:
            conn.close()
    print(f"{total} serials checked")


if __name__ == "__main__":
    main(sys.argv[1])
