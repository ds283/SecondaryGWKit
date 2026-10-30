"""Orchestrator's own read-only Y1 reader (06c Part B review, check 4).

Opens every shard of R17 and R18 with sqlite3 mode=ro, keys each QuadSourceIntegral row by
physical identity (k, q, r in 1/Mpc; z_response; log10 atol, log10 rtol; all by float.hex),
and scores Y1: |T18 - T17| <= e17 + e18, with rows whose total_abserr is non-finite in either
store counted apart. The comparison is made exactly in rationals (fractions.Fraction).
"""

import glob
import math
import sqlite3
import sys
from fractions import Fraction

M = "/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/"


def read(store):
    rows = {}
    for path in sorted(glob.glob(M + store + "-shard*.sqlite")):
        con = sqlite3.connect("file:" + path + "?mode=ro", uri=True)
        k = {s: x for s, x in con.execute(
            "select e.serial, w.k_inv_Mpc from wavenumber_exit_time e join wavenumber w on e.wavenumber_serial=w.serial")}
        z = dict(con.execute("select serial, z from redshift"))
        tol = dict(con.execute("select serial, log10_tol from tolerance"))
        for r in con.execute(
            "select k_wavenumber_exit_serial, q_wavenumber_exit_serial, r_wavenumber_exit_serial,"
            " z_response_serial, atol_serial, rtol_serial, total, total_abserr, total_converged,"
            " WKB_Levin_num_regions, WKB_Levin_max_depth from QuadSourceIntegral"):
            key = (k[r[0]].hex(), k[r[1]].hex(), k[r[2]].hex(), z[r[3]].hex(), tol[r[4]].hex(), tol[r[5]].hex())
            assert key not in rows, key
            rows[key] = dict(total=r[6], abserr=r[7], conv=r[8], nreg=r[9], depth=r[10])
        con.close()
    return rows


a = read("levin-phase-floor-06c-r17")
b = read("levin-phase-floor-06c-r18")
print("rows R17", len(a), "R18", len(b), "same keys", set(a) == set(b))
bit_total = sum(1 for key in a if a[key]["total"].hex() == b[key]["total"].hex())
print("total bit-identical", bit_total)
print("total_converged true: R17", sum(1 for v in a.values() if v["conv"]), "R18", sum(1 for v in b.values() if v["conv"]))
print("max WKB_Levin_max_depth: R17", max((v["depth"] or 0) for v in a.values()), "R18", max((v["depth"] or 0) for v in b.values()))
print("sum regions: R17", sum((v["nreg"] or 0) for v in a.values()), "R18", sum((v["nreg"] or 0) for v in b.values()))
nonfin = {"R17 only": 0, "R18 only": 0, "both": 0}
scored = viol = 0
worst = []
for key in sorted(a):
    x, y = a[key], b[key]
    fa = x["abserr"] is not None and math.isfinite(x["abserr"])
    fb = y["abserr"] is not None and math.isfinite(y["abserr"])
    if not (fa and fb):
        nonfin["both" if not fa and not fb else ("R17 only" if not fa else "R18 only")] += 1
        continue
    scored += 1
    d = abs(Fraction(y["total"]) - Fraction(x["total"]))
    e = Fraction(x["abserr"]) + Fraction(y["abserr"])
    if d > e:
        viol += 1
        worst.append((float(d / e) if e else math.inf, key, x, y))
print("non-finite total_abserr", nonfin)
print("Y1 scored", scored, "violations", viol)
worst.sort(reverse=True)
for ratio, key, x, y in worst:
    kk = [float.fromhex(h) for h in key]
    print("  %.4g  k=%.4g q=%.4g r=%.4g z=%.6g rtol=1e%g  T17=%s T18=%s e17=%s e18=%s conv %s/%s" % (
        ratio, kk[0], kk[1], kk[2], kk[3], kk[5], x["total"].hex(), y["total"].hex(), x["abserr"].hex(),
        y["abserr"].hex(), x["conv"], y["conv"]))
