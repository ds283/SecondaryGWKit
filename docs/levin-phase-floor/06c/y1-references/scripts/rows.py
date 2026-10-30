"""Y1 references, step 1: the 30 Y1-violating QuadSourceIntegral rows, from the store copies.

Adapted from the orchestrator's review/checks/y1_reader.py (read only): every shard of each copy
opened with sqlite3 ``mode=ro``; rows keyed by physical identity (k, q, r in 1/Mpc through
wavenumber_exit_time -> wavenumber.k_inv_Mpc; redshift.z of z_response; tolerance.log10_tol of
atol and rtol; all as float.hex); Y1 scored exactly in rationals. Added: each row's shard and
serial in each store, its total, total_abserr, numeric_quad, WKB_Levin, and the per-group records
of its metadata (``WKB_Levin.subintervals[*].groups``: value and abserr scaled by 1+z_response,
regions, converged, phase_limited), all floats as hex.

Usage: python -B rows.py OUT_JSON
"""

import glob
import json
import math
import sqlite3
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yprov import COPY_DIR, STORE_NAMES, header  # noqa: E402


def hx(x):
    return None if x is None else float(x).hex()


def read(store):
    rows = {}
    for path in sorted(glob.glob(str(COPY_DIR / (STORE_NAMES[store] + "-shard*.sqlite")))):
        shard = int(path[-11:-7])
        con = sqlite3.connect("file:" + path + "?mode=ro", uri=True)
        k = {s: x for s, x in con.execute(
            "select e.serial, w.k_inv_Mpc from wavenumber_exit_time e join wavenumber w on e.wavenumber_serial=w.serial")}
        z = dict(con.execute("select serial, z from redshift"))
        tol = dict(con.execute("select serial, log10_tol from tolerance"))
        for r in con.execute(
            "select k_wavenumber_exit_serial, q_wavenumber_exit_serial, r_wavenumber_exit_serial,"
            " z_response_serial, atol_serial, rtol_serial, total, total_abserr, total_converged,"
            " WKB_Levin_num_regions, WKB_Levin_max_depth, serial, numeric_quad, WKB_Levin, metadata,"
            " z_source_max_serial from QuadSourceIntegral"):
            key = (k[r[0]].hex(), k[r[1]].hex(), k[r[2]].hex(), z[r[3]].hex(), tol[r[4]].hex(), tol[r[5]].hex())
            assert key not in rows, key
            md = json.loads(r[14])
            subs = []
            for s in md["WKB_Levin"]["subintervals"]:
                subs.append({
                    "z_max": hx(s["z_max"]), "z_min": hx(s["z_min"]), "regime": s["regime"],
                    "value": hx(s["value"]), "abserr": hx(s["abserr"]),
                    "groups": [{"label": g["label"], "value": hx(g["value"]), "abserr": hx(g["abserr"]),
                                "regions": g["regions"], "converged": g["converged"],
                                "phase_limited": g["phase_limited"]} for g in s["groups"]],
                })
            rows[key] = dict(total=r[6], abserr=r[7], conv=r[8], nreg=r[9], depth=r[10], shard=shard,
                             serial=r[11], numeric_quad=r[12], WKB_Levin=r[13], subintervals=subs,
                             z_response=z[r[3]], z_source_max=z[r[15]],
                             numeric_quad_abserr=md["numeric_quad"]["abserr"])
        con.close()
    return rows


def main():
    out = Path(sys.argv[1])
    for l in header(Path(__file__).resolve(), "Y1 references: the violating rows, from the copies"):
        print(l)
    a, b = read("R17"), read("R18")
    print("rows R17", len(a), "R18", len(b), "same keys", set(a) == set(b))
    scored = 0
    viol = []
    for key in sorted(a):
        x, y = a[key], b[key]
        if not all(v["abserr"] is not None and math.isfinite(v["abserr"]) for v in (x, y)):
            continue
        scored += 1
        d = abs(Fraction(y["total"]) - Fraction(x["total"]))
        e = Fraction(x["abserr"]) + Fraction(y["abserr"])
        if d > e:
            viol.append((float(d / e), key, x, y))
    viol.sort(key=lambda t: -t[0])
    print("Y1 scored", scored, "violations", len(viol))
    recs = []
    for i, (ratio, key, x, y) in enumerate(viol):
        kk = [float.fromhex(h) for h in key]
        rid = f"Y{i + 1:02d}"
        rec = {"id": rid, "rank": i + 1, "ratio": ratio, "key": list(key),
               "k": kk[0], "q": kk[1], "r": kk[2], "z_response": kk[3], "log10_atol": kk[4], "log10_rtol": kk[5],
               "equilateral": kk[0] == kk[1] == kk[2]}
        for name, v in (("R17", x), ("R18", y)):
            rec[name] = {"shard": v["shard"], "serial": v["serial"], "total": hx(v["total"]), "total_abserr": hx(v["abserr"]),
                         "total_converged": v["conv"], "numeric_quad": hx(v["numeric_quad"]),
                         "numeric_quad_abserr": hx(v["numeric_quad_abserr"]), "WKB_Levin": hx(v["WKB_Levin"]),
                         "num_regions": v["nreg"], "max_depth": v["depth"], "subintervals": v["subintervals"],
                         "z_source_max": hx(v["z_source_max"])}
        recs.append(rec)
        print(f"{rid} {ratio:10.4g} k={kk[0]:.4g} q={kk[1]:.4g} r={kk[2]:.4g} z={kk[3]:.6g} "
              f"R17 shard {x['shard']} serial {x['serial']}  R18 shard {y['shard']} serial {y['serial']}  "
              f"T17={x['total'].hex()} T18={y['total'].hex()}")
    out.write_text(json.dumps(recs, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
