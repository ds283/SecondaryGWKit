"""Read-only census: do the replicated tables of a sharded store agree, row by row and serial by
serial, across every shard?  Opens each shard with mode=ro URIs; writes nothing.

store-fingerprint prompt 05 compared key/tag/validated/value-count multisets and per-table row
counts. Neither sees a serial split (the same key under different serials on different shards),
which replicated_write_fault_probe.py shows a resume can create after an interrupted replication.
This compares full rows, excluding `timestamp` (stamped per shard by design) and, for
BackgroundModelValue, `serial` (drawn per shard by design); value rows are compared on the rest.
"""

import sqlite3
import sys
from pathlib import Path

REPLICATED = [
    "version",
    "store_tag",
    "redshift",
    "wavenumber",
    "wavenumber_exit_time",
    "tolerance",
    "LambdaCDM",
    "QCD_Cosmology",
    "IntegrationSolver",
    "BackgroundModel",
    "BackgroundModelValue",
    "GkSourcePolicy",
    "QuadSourcePolicy",
    "BackgroundModel_tags",
]


def rows(con, table):
    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]
    if not cols:
        return None
    drop = {"timestamp"} | ({"serial"} if table == "BackgroundModelValue" else set())
    keep = [c for c in cols if c not in drop]
    q = "SELECT " + ", ".join(f'"{c}"' for c in keep) + f' FROM "{table}"'
    return keep, sorted(con.execute(q).fetchall(), key=repr)


def census(primary: Path):
    shards = sorted(primary.parent.glob(primary.stem + "-shard*.sqlite"))
    print(f"== {primary}  ({len(shards)} shards)")
    cons = {s.name: sqlite3.connect(f"file:{s}?mode=ro", uri=True) for s in shards}
    bad = 0
    for t in REPLICATED:
        per = {n: rows(c, t) for n, c in cons.items()}
        present = {n: v for n, v in per.items() if v is not None}
        if not present:
            continue
        sets = {n: set(v[1]) for n, v in present.items()}
        union = set().union(*sets.values())
        counts = {n: len(v[1]) for n, v in present.items()}
        diff = {n: len(union - s) for n, s in sets.items()}
        ok = all(d == 0 for d in diff.values()) and len(present) == len(shards)
        bad += not ok
        print(
            f"  {t:24s} rows/shard={list(counts.values())}  missing-vs-union={list(diff.values())}  {'OK' if ok else 'DIVERGES'}"
        )
        if not ok and "serial" in present[next(iter(present))][0]:
            i = present[next(iter(present))][0].index("serial")
            for n, s in sets.items():
                lacking = sorted(r[i] for r in union - s)[:10]
                if lacking:
                    print(f"      {n} lacks serials {lacking}")
    for c in cons.values():
        c.close()
    print(f"  => {'no divergence' if bad == 0 else f'{bad} table(s) diverge'}")


for p in sys.argv[1:]:
    census(Path(p))
