"""
S4 (a3-v2-readiness audit 00): how long the fingerprint that ``Run.finish(..., fingerprint=True)``
takes lasts on a real-sized store.

``Run.finish(fingerprint=True)`` calls ``runregistrykit.stores.fingerprint_store(results,
write=reading.ok, ...)``, which writes the sidecar when the sidecar is a problem-free registry
sidecar. This probe calls ``fingerprint_store(primary, write=False)`` directly, on the A3 baseline,
``var/datastores/handover-A3-baseline-lambdacdm.sqlite``, and on nothing else. It never calls
``Run.finish``, ``runregistrykit.begin``, any ``store`` subcommand, or ``ShardedPool``.
``fingerprint_store`` reads the store through ``Datastore.store_inventory.read_inventory``, which
opens every file ``mode=ro`` and refuses a journal (``Datastore/store_reader.py``), and reads the
sidecar and ``var/runs/`` (for its running-run refusal). The README of the a3-v2-readiness campaign,
§3, permits this read and no other.

It records the SHA-256, size and mtime of the five store files and the sidecar before and after,
the load average, and the wall time of the import and of two calls, each timed alone. It prints
what the fingerprint says of the store, or the refusal, unabridged.

Run from the repository root:

    PYTHONPATH=. ./venv/bin/python docs/a3-v2-readiness/fingerprint_time_probe.py \
        > docs/a3-v2-readiness/fingerprint_time_probe.txt 2>&1
"""

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

STORE = Path("var/datastores/handover-A3-baseline-lambdacdm.sqlite")


def files():
    stem = STORE.with_suffix("")
    out = [STORE] + [Path(f"{stem}-shard{i:04d}.sqlite") for i in range(4)]
    out.append(Path(f"{stem}.manifest.json"))
    return out


def snapshot():
    snap = {}
    for p in files():
        st = p.stat()
        snap[str(p)] = (
            hashlib.sha256(p.read_bytes()).hexdigest(),
            st.st_size,
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        )
    return snap


def journals():
    found = []
    for p in files()[:5]:
        for suffix in ("-journal", "-wal", "-shm"):
            if os.path.lexists(str(p) + suffix):
                found.append(str(p) + suffix)
    return found


def main():
    print("# fingerprint_time_probe.py (a3-v2-readiness audit 00, S4)")
    print(f"python {sys.version.split()[0]}, {os.uname().sysname} {os.uname().machine}")
    print(
        f"started {time.strftime('%Y-%m-%d %H:%M:%S %Z')}, load average {os.getloadavg()}"
    )
    before = snapshot()
    print("\n## before")
    total = 0
    for path, (sha, size, mtime) in before.items():
        print(f"  {sha}  {size:>12,d}  {mtime}  {path}")
        if path.endswith(".sqlite"):
            total += size
    print(
        f"  store size (primary + 4 shards): {total:,d} bytes ({total / 2**20:.1f} MiB)"
    )
    print(f"  journals beside the store: {journals()}")

    t = time.perf_counter()
    from runregistrykit.stores import fingerprint_store

    print(f"\nimport runregistrykit.stores: {time.perf_counter() - t:.3f} s")
    from config.fingerprint import FINGERPRINT_FORMAT

    print(f"FINGERPRINT_FORMAT at HEAD: {FINGERPRINT_FORMAT}")

    for n in (1, 2):
        print(f"\n## call {n}: fingerprint_store(primary, write=False)")
        print(f"load average {os.getloadavg()}")
        t = time.perf_counter()
        try:
            taken = fingerprint_store(STORE, write=False)
            wall = time.perf_counter() - t
            print(f"wall time: {wall:.3f} s")
            fp = taken["fingerprint"]
            print(f"wrote: {taken['wrote']}; sidecar kind: {taken['sidecar']}")
            print(
                f"fingerprint_format {fp['fingerprint_format']}, digest {fp['digest']}, "
                f"problems {fp['problems']}"
            )
            print("classes (count):")
            for name, entry in fp["classes"].items():
                print(f"  {name}: {entry['count']}")
            recorded = taken["recorded"]
            print(
                "recorded fingerprint_format: "
                f"{recorded.get('fingerprint_format') if recorded else None}"
            )
            print(
                f"comparison with the recorded fingerprint: {json.dumps(taken['comparison'])}"
            )
        except Exception as e:
            wall = time.perf_counter() - t
            print(
                f"raised after {wall:.3f} s: {type(e).__module__}.{type(e).__name__}: {e}"
            )
            traceback.print_exc(file=sys.stdout)

    after = snapshot()
    print("\n## after")
    for path, (sha, size, mtime) in after.items():
        same = "unchanged" if before[path] == after[path] else "CHANGED"
        print(f"  {sha}  {size:>12,d}  {mtime}  {path}  {same}")
    print(f"  journals beside the store: {journals()}")
    print(
        f"finished {time.strftime('%Y-%m-%d %H:%M:%S %Z')}, load average {os.getloadavg()}"
    )


if __name__ == "__main__":
    main()
