"""
Orchestrator's own read-only re-score of prompt 06h §4.3 H0 on R21 and R22.

Reads every QuadSourceIntegral row of both stores (sqlite mode=ro), pairs them by physical key
(k, q, r, z_response, atol, rtol), and compares `total`, `total_abserr`, `total_converged` and
`analytic_rad` by bit pattern. Scores H0 as written on `total`, and the same rule on the analytic
oracle (`analytic_rad` against the metadata's `analytic.abserr`) as the second reading.

Usage: orch_h0.py R21_PRIMARY R22_PRIMARY
"""

import json
import math
import sqlite3
import sys
from pathlib import Path


def read(primary):
    primary = Path(primary)
    con = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
    shards = [r[0] for r in con.execute("SELECT filename FROM shards ORDER BY serial")]
    con.close()
    out = {}
    for name in shards:
        con = sqlite3.connect(f"file:{primary.parent / Path(name).name}?mode=ro", uri=True)
        wn = (
            "(SELECT w.k_inv_Mpc FROM wavenumber_exit_time e JOIN wavenumber w "
            "ON w.serial = e.wavenumber_serial WHERE e.serial = q.{})"
        )
        sql = (
            f"SELECT {wn.format('k_wavenumber_exit_serial')}, "
            f"{wn.format('q_wavenumber_exit_serial')}, {wn.format('r_wavenumber_exit_serial')}, "
            "(SELECT z FROM redshift WHERE serial = q.z_response_serial), "
            "(SELECT log10_tol FROM tolerance WHERE serial = q.atol_serial), "
            "(SELECT log10_tol FROM tolerance WHERE serial = q.rtol_serial), "
            "q.total, q.total_abserr, q.total_converged, q.analytic_rad, q.metadata "
            "FROM QuadSourceIntegral q"
        )
        for row in con.execute(sql):
            key = tuple(row[:6])
            assert key not in out, key
            meta = json.loads(row[10]) if row[10] is not None else {}
            an = meta.get("analytic", {}) if isinstance(meta, dict) else {}
            out[key] = dict(
                total=row[6],
                abserr=row[7],
                conv=row[8],
                arad=row[9],
                aerr=an.get("abserr") if isinstance(an, dict) else None,
            )
        con.close()
    return out


def hx(x):
    return None if x is None else float(x).hex()


def score(a, b, vk, ek, label):
    n = viol = nonfinite = skipped = ident = 0
    worst = (0.0, None)
    for key in a:
        va, vb, ea, eb = a[key][vk], b[key][vk], a[key][ek], b[key][ek]
        if va is None or vb is None or ea is None or eb is None:
            skipped += 1
            continue
        if not (math.isfinite(ea) and math.isfinite(eb)):
            nonfinite += 1
            continue
        n += 1
        if hx(va) == hx(vb):
            ident += 1
        d = abs(vb - va)
        r = d / (ea + eb) if ea + eb > 0 else (0.0 if d == 0 else math.inf)
        if r > worst[0]:
            worst = (r, key)
        if d > ea + eb:
            viol += 1
            print(f"  VIOLATION {label} {key}: {hx(va)} {hx(vb)} abserr {hx(ea)} {hx(eb)}")
    print(
        f"{label}: scored {n}, bit-identical {ident}, violations {viol}, non-finite {nonfinite},"
        f" skipped (missing) {skipped}, worst |d|/(e21+e22) {worst[0]:.6e} at {worst[1]}"
    )


def main():
    a, b = read(sys.argv[1]), read(sys.argv[2])
    print(f"rows R21 {len(a)}, R22 {len(b)}, keys equal {set(a) == set(b)}")
    assert set(a) == set(b)
    conv_change = sum(1 for k in a if a[k]["conv"] != b[k]["conv"])
    print(f"total_converged differing: {conv_change}; true in R21 {sum(1 for k in a if a[k]['conv'])}"
          f", in R22 {sum(1 for k in b if b[k]['conv'])}")
    for fld in ("total", "abserr", "arad", "aerr"):
        same = sum(1 for k in a if hx(a[k][fld]) == hx(b[k][fld]))
        print(f"field {fld}: bit-identical in {same} of {len(a)}")
    score(a, b, "total", "abserr", "H0 (total, as written)")
    score(a, b, "arad", "aerr", "H0 form on analytic_rad / analytic.abserr")


if __name__ == "__main__":
    main()
