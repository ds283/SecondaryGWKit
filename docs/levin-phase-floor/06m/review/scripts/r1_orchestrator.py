"""Orchestrator's own reader for prompt 06m §4's R1, from the R25 and R26 stores (read only).

Usage: python -I -B r1_orchestrator.py R25_PRIMARY R26_PRIMARY ROWS_JSON

Reads every shard's QuadSourceIntegral with sqlite mode=ro, keys each row by its physical key
(k, q, r, z_response; floats read from the wavenumber/redshift tables), and scores R1 in exact
rational arithmetic (fractions.Fraction of each double), so no rounding enters the comparison:
  |total_26 - total_25| <= total_abserr_25 + total_abserr_26, and the same for analytic_rad against
  metadata.analytic.abserr.
Also reports bit-identical counts (float.hex), total_converged changes in each direction, and the
WKB_Levin_num_regions / WKB_Levin_max_depth columns. A row is a stationary-point row only if it is
an A3 v2 row at all: the rehearsal keys are checked against ROWS_JSON (06a's rows.json, every A3 v2
QuadSourceIntegral row); if none matches, none of 06a's 720 can be among them.
"""

import hashlib
import json
import math
import sqlite3
import sys
from fractions import Fraction
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(primary):
    primary = Path(primary)
    c = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
    shards = [r[0] for r in c.execute("SELECT filename FROM shards ORDER BY serial")]
    c.close()
    out = {}
    for name in shards:
        c = sqlite3.connect(f"file:{primary.parent / Path(name).name}?mode=ro", uri=True)
        kw = (
            "(SELECT w.k_inv_Mpc FROM wavenumber_exit_time e JOIN wavenumber w "
            "ON w.serial = e.wavenumber_serial WHERE e.serial = q.{})"
        )
        sql = (
            f"SELECT {kw.format('k_wavenumber_exit_serial')}, {kw.format('q_wavenumber_exit_serial')}, "
            f"{kw.format('r_wavenumber_exit_serial')}, "
            "(SELECT z FROM redshift WHERE serial = q.z_response_serial), "
            "q.total, q.total_abserr, q.total_converged, q.analytic_rad, q.metadata, "
            "q.WKB_Levin_num_regions, q.WKB_Levin_max_depth FROM QuadSourceIntegral q"
        )
        for row in c.execute(sql):
            key = tuple(row[:4])
            assert key not in out, key
            md = json.loads(row[8]) if row[8] else {}
            an = (md.get("analytic") or {}).get("abserr")
            out[key] = dict(
                total=row[4],
                err=row[5],
                conv=row[6],
                an=row[7],
                an_err=an,
                nreg=row[9],
                depth=row[10],
                shard=name,
            )
        c.close()
    return out


def fin(x):
    return x is not None and math.isfinite(x)


def score(a, b, va, ea, vb, eb):
    scored = viol = nonfin = 0
    worst = (0.0, None)
    vlist = []
    ident = 0
    for k in a:
        x, y = a[k], b[k]
        if (
            x[va] is not None
            and y[va] is not None
            and float(x[va]).hex() == float(y[va]).hex()
        ):
            ident += 1
        if not (fin(x[va]) and fin(y[va]) and fin(x[ea]) and fin(y[ea])):
            nonfin += 1
            continue
        scored += 1
        d = abs(Fraction(y[va]) - Fraction(x[va]))
        s = Fraction(x[ea]) + Fraction(y[ea])
        if d > s:
            viol += 1
            vlist.append(k)
        if s > 0:
            r = float(d / s)
            if r > worst[0]:
                worst = (r, k)
        elif d > 0:
            worst = (math.inf, k)
    return dict(
        scored=scored,
        violations=viol,
        nonfinite=nonfin,
        worst=worst,
        bit_identical=ident,
        violators=vlist,
    )


def main():
    p25, p26, rows_json = map(Path, sys.argv[1:4])
    print(f"# r1_orchestrator.py sha256 {sha(__file__)}")
    print(f"# R25 {p25}\n# R26 {p26}\n# rows.json {rows_json} sha256 {sha(rows_json)}")
    a, b = read(p25), read(p26)
    print(f"rows R25 {len(a)}  R26 {len(b)}  in one only: {len(set(a) ^ set(b))}")
    assert set(a) == set(b)
    a3 = {
        (float(r["k"]), float(r["q"]), float(r["r"]), float(r["z_response"]))
        for r in json.loads(rows_json.read_text())["rows"]
    }
    print(
        f"rehearsal rows that are A3 v2 rows (so could be among 06a's 720): {sum(1 for k in a if k in a3)}"
    )
    for lab, va, ea in (("total", "total", "err"), ("oracle", "an", "an_err")):
        s = score(a, b, va, ea, va, ea)
        print(
            f"R1 {lab}: scored {s['scored']}, violations {s['violations']}, non-finite {s['nonfinite']}, "
            f"bit-identical value {s['bit_identical']}/{len(a)}, largest |d|/(e25+e26) {s['worst'][0]:.6g} at {s['worst'][1]}"
        )
        for k in s["violators"]:
            x, y = a[k], b[k]
            print(
                f"  VIOLATION {lab} {k}: v25 {float(x[va]).hex()} v26 {float(y[va]).hex()} "
                f"e25 {float(x[ea]).hex()} e26 {float(y[ea]).hex()}"
            )
    ei = sum(1 for k in a if float(a[k]["err"]).hex() == float(b[k]["err"]).hex())
    print(f"total_abserr bit-identical {ei}/{len(a)}")
    c_lost = sorted(k for k in a if a[k]["conv"] and not b[k]["conv"])
    c_gain = sorted(k for k in a if not a[k]["conv"] and b[k]["conv"])
    print(
        f"total_converged: R25 {sum(1 for k in a if a[k]['conv'])}, R26 {sum(1 for k in b if b[k]['conv'])}; "
        f"converged->unconverged {len(c_lost)}, unconverged->converged {len(c_gain)}"
    )
    for k in c_lost:
        print(f"  lost {k}")
    for k in c_gain:
        print(f"  gained {k}")
    for lab, f in (("WKB_Levin_num_regions", "nreg"), ("WKB_Levin_max_depth", "depth")):
        sa = [a[k][f] for k in a if a[k][f] is not None]
        sb = [b[k][f] for k in b if b[k][f] is not None]
        print(
            f"{lab}: rows with value R25 {len(sa)} R26 {len(sb)}; sum R25 {sum(sa)} R26 {sum(sb)}; "
            f"max R25 {max(sa)} R26 {max(sb)}; rows at 20 R25 {sa.count(20)} R26 {sb.count(20)}; "
            f"rows changed {sum(1 for k in a if a[k][f] != b[k][f])}"
        )


if __name__ == "__main__":
    main()
