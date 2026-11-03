"""Orchestrator's own reader: V1 re-scored from R23 and R24, read-only (mode=ro).

Rows matched by physical identity: (k, q, r [k_inv_Mpc via wavenumber_exit_time], z_response,
z_source_max [redshift.z], log10 atol, log10 rtol), each looked up in the row's own shard.
V1 rule (prompt 06g §4): |t24 - t23| <= e23 + e24 on total/total_abserr, and on analytic_rad /
metadata analytic.abserr, for rows with finite errors in both stores.
"""
import json, math, sqlite3, sys, collections

ROOT = "/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/levin-phase-floor-06g-"


def load(tag):
    rows = {}
    for s in range(4):
        con = sqlite3.connect(f"file:{ROOT}{tag}-shard{s:04d}.sqlite?mode=ro", uri=True)
        k = dict(con.execute("select e.serial, w.k_inv_Mpc from wavenumber_exit_time e join wavenumber w on e.wavenumber_serial=w.serial"))
        z = dict(con.execute("select serial, z from redshift"))
        tol = dict(con.execute("select serial, log10_tol from tolerance"))
        for r in con.execute("select k_wavenumber_exit_serial,q_wavenumber_exit_serial,r_wavenumber_exit_serial,"
                             "z_response_serial,z_source_max_serial,atol_serial,rtol_serial,total,total_abserr,"
                             "total_converged,analytic_rad,metadata,WKB_Levin_num_regions,WKB_Levin_max_depth from QuadSourceIntegral"):
            key = (k[r[0]].hex(), k[r[1]].hex(), k[r[2]].hex(), z[r[3]].hex(), z[r[4]].hex(), tol[r[5]], tol[r[6]])
            assert key not in rows, key
            md = json.loads(r[11])
            ae = md.get("analytic", {}).get("abserr")
            rows[key] = dict(shard=s, total=r[7], err=r[8], conv=r[9], an=r[10],
                             an_err=(float(ae) if ae is not None else None), regions=r[12], depth=r[13])
        con.close()
    return rows


a, b = load("r23"), load("r24")
print("rows", len(a), len(b), "keys equal", set(a) == set(b))
res = collections.Counter()
worst = {}
viol = {"total": [], "oracle": []}
for key in sorted(a):
    x, y = a[key], b[key]
    for name, v, e in (("total", "total", "err"), ("oracle", "an", "an_err")):
        if x[v] is None or y[v] is None or x[e] is None or y[e] is None:
            res[(name, "missing")] += 1
            continue
        if not (math.isfinite(x[e]) and math.isfinite(y[e])):
            res[(name, "infinite err")] += 1
            continue
        if x[v].hex() == y[v].hex():
            res[(name, "bit-identical")] += 1
        d = abs(y[v] - x[v])
        s = x[e] + y[e]
        ratio = d / s if s > 0 else (0.0 if d == 0 else math.inf)
        res[(name, "scored")] += 1
        if ratio > 1:
            viol[name].append((ratio, key, x[v].hex(), y[v].hex(), x[e].hex(), y[e].hex()))
        if ratio > worst.get(name, (-1,))[0]:
            worst[name] = (ratio, key)
conv = collections.Counter((a[k]["conv"], b[k]["conv"]) for k in a)
print("counts", dict(res))
print("total_converged (R23, R24):", dict(conv))
print("rows at depth 20: R23", sum(1 for k in a if a[k]["depth"] == 20), "R24", sum(1 for k in b if b[k]["depth"] == 20))
print("regions sum: R23", sum(a[k]["regions"] or 0 for k in a), "R24", sum(b[k]["regions"] or 0 for k in b))
for name in ("total", "oracle"):
    print(f"V1 on {name}: {len(viol[name])} violations; worst ratio {worst[name][0]:.6g} at {worst[name][1]}")
    for t in sorted(viol[name], reverse=True):
        print("  ratio %.6g k=%s z_resp=%r t23 %s t24 %s e23 %s e24 %s" % (
            t[0], float.fromhex(t[1][0]), float.fromhex(t[1][3]), t[2], t[3], t[4], t[5]))
print("V1:", "holds" if not viol["total"] and not viol["oracle"] else "fails")
