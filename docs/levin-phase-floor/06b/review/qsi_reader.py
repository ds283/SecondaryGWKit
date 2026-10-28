"""Orchestrator's independent read-only reader for 06b check 4 (R15 vs R16 QuadSourceIntegral)."""

import json
import sqlite3
import sys

D = "/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/levin-phase-floor-06b-"
ID_COLS = [
    "model_serial", "k_wavenumber_exit_serial", "q_wavenumber_exit_serial",
    "r_wavenumber_exit_serial", "atol_serial", "rtol_serial", "policy_serial",
    "quad_policy_serial", "source_serial", "data_serial", "Tq_numeric_serial",
    "Tq_WKB_serial", "Tr_numeric_serial", "Tr_WKB_serial", "z_response_serial",
    "z_source_max_serial",
]


def con(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def hx(x):
    return float.hex(x) if isinstance(x, float) else repr(x)


def load(tag):
    rows = {}
    replicated = {}
    for i in range(4):
        c = con(f"{D}{tag}-shard{i:04d}.sqlite")
        c.row_factory = sqlite3.Row
        for t, cols in (("wavenumber_exit_time", "*"), ("redshift", "*"), ("tolerance", "*")):
            try:
                for r in c.execute(f"select {cols} from {t}"):
                    d = dict(r)
                    d.pop("timestamp", None)
                    d.pop("compute_time", None)
                    d.pop("label", None)
                    replicated.setdefault(t, {})[d["serial"]] = tuple(sorted((k, hx(v)) for k, v in d.items()))
            except sqlite3.OperationalError:
                pass
        kv = {r[0]: r[1] for r in c.execute("select serial, k_inv_Mpc from wavenumber")}
        ex = {r[0]: kv[r[1]] for r in c.execute("select serial, wavenumber_serial from wavenumber_exit_time")}
        tol = {r[0]: r[1] for r in c.execute("select serial, log10_tol from tolerance")}
        zz = {r[0]: r[1] for r in c.execute("select serial, z from redshift")}
        gp = {r[0]: r[1] for r in c.execute("select serial, Levin_threshold from GkSourcePolicy")}
        qp = {r[0]: r[1] for r in c.execute("select serial, Levin_threshold from QuadSourcePolicy")}
        for r in c.execute("select * from QuadSourceIntegral"):
            d = dict(r)
            key = tuple(hx(x) for x in (
                ex[d["k_wavenumber_exit_serial"]], ex[d["q_wavenumber_exit_serial"]],
                ex[d["r_wavenumber_exit_serial"]], tol[d["atol_serial"]], tol[d["rtol_serial"]],
                gp[d["policy_serial"]], qp[d["quad_policy_serial"]],
                zz[d["z_response_serial"]], zz[d["z_source_max_serial"]]))
            assert key not in rows, ("duplicate identity", tag, key)
            d["_shard"] = i
            rows[key] = d
        c.close()
    return rows, replicated


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(o, list):
        for j, v in enumerate(o):
            yield from walk(v, f"{path}[{j}]")
    else:
        yield path, o


COUNTER_LEAVES = {"regions", "simple_regions", "evaluations", "elapsed", "compute_time"}


def flags(d):
    m = json.loads(d["metadata"])
    a = bool(d["total_phase_limited"])
    b = []
    c = []
    for p, v in walk(m):
        if p.endswith("phase_limited") and v is True:
            if p.startswith("WKB_Levin.subintervals") and ".groups[" in p:
                b.append(p)
            elif p.startswith("analytic."):
                c.append(p)
    return a, b, c, m


def main():
    r15, rep15 = load("r15")
    r16, rep16 = load("r16")
    print("rows R15", len(r15), "R16", len(r16))
    print("identity sets equal:", set(r15) == set(r16))
    for t in rep15:
        print("replicated", t, "equal:", rep15[t] == rep16.get(t), len(rep15[t]))
    value_cols = ["b", "total", "total_abserr", "total_converged", "numeric_quad", "WKB_Levin",
                  "analytic_rad", "eta_source_max", "eta_response"]
    flagged = {}
    n_a = n_b = n_c = 0
    nb_flags = nc_flags = 0
    viol = []
    moved_total = []
    for key, a in r15.items():
        b = r16[key]
        fa, fb, fc, ma = flags(a)
        if fa: n_a += 1
        if fb: n_b += 1
        if fc: n_c += 1
        nb_flags += len(fb)
        nc_flags += len(fc)
        any_flag = fa or fb or fc
        # R16 flags
        ga, gb, gc, mb = flags(b)
        if ga or gb or gc:
            flagged.setdefault("r16", []).append(key)
        if any_flag:
            flagged.setdefault("r15", []).append(key)
        if hx(a["total"]) != hx(b["total"]):
            moved_total.append((key, a, b, fa, fb, fc))
        if not any_flag:
            for col in value_cols:
                if hx(a[col]) != hx(b[col]):
                    viol.append((key, col, hx(a[col]), hx(b[col])))
            # analytic block and every group record, counters/timings aside
            la = {p: v for p, v in walk(ma) if p.split(".")[-1].split("[")[0] not in COUNTER_LEAVES}
            lb = {p: v for p, v in walk(mb) if p.split(".")[-1].split("[")[0] not in COUNTER_LEAVES}
            for p in set(la) | set(lb):
                if not (p.startswith("analytic") or p.startswith("WKB_Levin")):
                    continue
                if hx(la.get(p)) != hx(lb.get(p)):
                    viol.append((key, p, hx(la.get(p)), hx(lb.get(p))))
    print("R15 flagged rows:", len(flagged.get("r15", [])), "(a)", n_a, "(b)", n_b, "flags", nb_flags, "(c)", n_c, "flags", nc_flags)
    print("R16 flagged rows:", len(flagged.get("r16", [])))
    print("violations among unflagged rows:", len(viol))
    for v in viol[:20]:
        print("  ", v)
    print("rows whose total moved:", len(moved_total))
    print("  all have (b) flag:", all(fb for _, _, _, _, fb, _ in moved_total))
    mx15 = mx16 = 0.0
    for key, a, b, fa, fb, fc in moved_total:
        d = abs(a["total"] - b["total"])
        mx15 = max(mx15, d / a["total_abserr"])
        mx16 = max(mx16, d / b["total_abserr"])
    print("  max |dT|/abserr15 = %.4g, /abserr16 = %.4g" % (mx15, mx16))
    if moved_total:
        key, a, b, *_ = sorted(moved_total, key=lambda t: -abs(t[1]["total"] - t[2]["total"]) / t[1]["total_abserr"])[0]
        print("  spot-check (largest): R15 total", hx(a["total"]), "abserr", hx(a["total_abserr"]))
        print("                        R16 total", hx(b["total"]), "abserr", hx(b["total_abserr"]))
        print("                        shard", a["_shard"], "serial R15", a["serial"], "R16", b["serial"])
    json.dump(sorted([list(k) for k in flagged.get("r15", [])]), open(sys.argv[1], "w"))


main()
