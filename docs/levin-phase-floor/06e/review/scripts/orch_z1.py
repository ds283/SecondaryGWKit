"""Orchestrator's own Z1 reader for 06e: every QuadSourceIntegral row of R19 and R20, opened
read-only (sqlite3 mode=ro, no package code), matched by physical key (k, q, r, z_response, atol,
rtol), compared exactly (Fraction of the stored doubles). Z1: |T20 - T19| <= e19 + e20."""

import math
import sqlite3
from fractions import Fraction

STORES = {
    "R19": "var/datastores/levin-phase-floor-06e-r19",
    "R20": "var/datastores/levin-phase-floor-06e-r20",
}
Q = """
select wk.k_inv_Mpc, wq.k_inv_Mpc, wr.k_inv_Mpc, z.z, ta.log10_tol, tr.log10_tol,
       i.total, i.total_abserr, i.total_converged, i.WKB_Levin_num_regions, i.WKB_Levin_max_depth
from QuadSourceIntegral i
join wavenumber_exit_time ek on ek.serial = i.k_wavenumber_exit_serial join wavenumber wk on wk.serial = ek.wavenumber_serial
join wavenumber_exit_time eq on eq.serial = i.q_wavenumber_exit_serial join wavenumber wq on wq.serial = eq.wavenumber_serial
join wavenumber_exit_time er on er.serial = i.r_wavenumber_exit_serial join wavenumber wr on wr.serial = er.wavenumber_serial
join redshift z on z.serial = i.z_response_serial
join tolerance ta on ta.serial = i.atol_serial join tolerance tr on tr.serial = i.rtol_serial
"""


def read(base):
    rows = {}
    for s in range(4):
        con = sqlite3.connect(f"file:{base}-shard{s:04d}.sqlite?mode=ro", uri=True)
        for r in con.execute(Q):
            key = tuple(float(x).hex() for x in r[:6])
            assert key not in rows, key
            rows[key] = r[6:]
        con.close()
    return rows


a, b = read(STORES["R19"]), read(STORES["R20"])
print("rows: R19", len(a), " R20", len(b), " same keys:", set(a) == set(b))
n = viol = nonfin = bit = 0
worst = (Fraction(0), None)
conv = [0, 0]
reg = [0, 0]
deep = [0, 0]
for k in sorted(a):
    t19, e19, c19, n19, d19 = a[k]
    t20, e20, c20, n20, d20 = b[k]
    conv[0] += bool(c19)
    conv[1] += bool(c20)
    reg[0] += n19 or 0
    reg[1] += n20 or 0
    deep[0] += (d19 or 0) >= 12
    deep[1] += (d20 or 0) >= 12
    bit += t19.hex() == t20.hex()
    if e19 is None or e20 is None or not (math.isfinite(e19) and math.isfinite(e20)):
        nonfin += 1
        continue
    n += 1
    d = abs(Fraction(t20) - Fraction(t19))
    s = Fraction(e19) + Fraction(e20)
    viol += d > s
    r = d / s if s else Fraction(0)
    if r > worst[0]:
        worst = (r, k, t19.hex(), t20.hex(), e19.hex(), e20.hex())
print(f"Z1: scored {n}, non-finite {nonfin}, violations {viol} ->", "HOLDS" if viol == 0 else "FAILS")
print(f"total bit-identical (float.hex): {bit}; moved: {len(a) - bit}")
print(f"largest |dT|/(e19+e20) = {float(worst[0]):.6g}; |dT|/e19 = {float(abs(Fraction(float.fromhex(worst[3])) - Fraction(float.fromhex(worst[2]))) / Fraction(float.fromhex(worst[4]))):.6g}")
print("  at key (k,q,r,z,log10 atol,log10 rtol) =", tuple(float.fromhex(x) for x in worst[1]))
print("  T19", worst[2], " T20", worst[3], " e19", worst[4], " e20", worst[5])
print(f"total_converged: R19 {conv[0]}, R20 {conv[1]}; WKB_Levin_num_regions summed: R19 {reg[0]}, R20 {reg[1]}; rows at depth >= 12: R19 {deep[0]}, R20 {deep[1]}")
