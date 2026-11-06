"""Orchestrator's own whole-row comparison of QuadSourceIntegral between two rehearsal stores.

Usage: python -I -B qsi_full_compare.py A_PRIMARY B_PRIMARY

Every shard read with sqlite mode=ro. Rows keyed by (k, q, r, z_response, log10 atol, log10 rtol),
resolved through each shard's own tables. Every column is compared except store bookkeeping (serial,
version, timestamp, label and every *_serial) and the timing columns (named below). `metadata` is
parsed as JSON and compared leaf by leaf, dropping only leaves whose key is `elapsed` or
`compute_time` or ends in `_time`. Floats compare by float.hex; everything else by type and value.
Prints the differing fields by name with counts.
"""

import collections
import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path

SKIP = {"serial", "version", "timestamp", "label"}
TIMING = {
    "numeric_quad_compute_time",
    "numeric_quad_mean_RHS_time",
    "numeric_quad_max_RHS_time",
    "numeric_quad_min_RHS_time",
    "WKB_Levin_elapsed",
    "compute_time",
    "analytic_compute_time",
}


def read(primary):
    primary = Path(primary)
    c = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
    shards = [r[0] for r in c.execute("SELECT filename FROM shards ORDER BY serial")]
    c.close()
    out = {}
    for name in shards:
        c = sqlite3.connect(f"file:{primary.parent / Path(name).name}?mode=ro", uri=True)
        cols = [r[1] for r in c.execute("PRAGMA table_info(QuadSourceIntegral)")]
        keep = [x for x in cols if x not in SKIP and x not in TIMING and not x.endswith("_serial")]
        kw = (
            "(SELECT w.k_inv_Mpc FROM wavenumber_exit_time e JOIN wavenumber w "
            "ON w.serial = e.wavenumber_serial WHERE e.serial = q.{})"
        )
        sql = (
            f"SELECT {kw.format('k_wavenumber_exit_serial')}, {kw.format('q_wavenumber_exit_serial')}, "
            f"{kw.format('r_wavenumber_exit_serial')}, (SELECT z FROM redshift WHERE serial = q.z_response_serial), "
            "(SELECT log10_tol FROM tolerance WHERE serial = q.atol_serial), "
            "(SELECT log10_tol FROM tolerance WHERE serial = q.rtol_serial), "
            + ", ".join(f'q."{x}"' for x in keep)
            + " FROM QuadSourceIntegral q"
        )
        for row in c.execute(sql):
            key = tuple(row[:6])
            assert key not in out, key
            rec = {}
            for col, v in zip(keep, row[6:]):
                if col == "metadata":
                    flat(json.loads(v) if v else None, "metadata", rec)
                else:
                    rec[col] = v
            out[key] = rec
        c.close()
    return out


def flat(o, p, rec):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("elapsed", "compute_time") or k.endswith("_time"):
                continue
            flat(v, f"{p}.{k}", rec)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            flat(v, f"{p}[{i}]", rec)
    else:
        rec[p] = o


def same(x, y):
    if isinstance(x, float) and isinstance(y, float):
        return x.hex() == y.hex()
    return type(x) is type(y) and x == y


def generic(path):
    import re

    return re.sub(r"\[\d+\]", "[]", path)


def main():
    a_p, b_p = sys.argv[1:3]
    print(f"# qsi_full_compare.py sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}")
    print(f"# A {a_p}\n# B {b_p}")
    a, b = read(a_p), read(b_p)
    print(f"rows A {len(a)} B {len(b)}; in one only {len(set(a) ^ set(b))}")
    nf = nd = 0
    fieldset_diff = 0
    diffs = collections.Counter()
    rows_diff = 0
    for k in a:
        x, y = a[k], b[k]
        if set(x) != set(y):
            fieldset_diff += 1
        rd = False
        for f in set(x) | set(y):
            nf += 1
            if not same(x.get(f, "<absent>"), y.get(f, "<absent>")):
                nd += 1
                rd = True
                diffs[generic(f)] += 1
        rows_diff += rd
    print(f"fields compared {nf}; differing {nd}; rows with any difference {rows_diff}; rows whose field sets differ {fieldset_diff}")
    for f, n in sorted(diffs.items(), key=lambda t: -t[1]):
        print(f"  {n:7d}  {f}")


if __name__ == "__main__":
    main()
