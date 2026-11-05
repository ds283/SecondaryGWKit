# Orchestrator's review §B.3 check 5: C1 and S1 (and C2 alongside) recomputed per route from the raw
# census checkpoints (not the agent's extracts), with my own reader.
# - The 720: rows (shard-serial) whose 1e-7 unit in 06a's census scan shows a sign change of theta'
#   in any call (06e/06h's rule); checked against v8_calls.json's stationary_point_row flags.
# - C1: v8_calls.json "lost" entries outside the 720, per rtol: the route's call (same unit, same
#   index) converged.
# - C2: calls outside the 720 (calls + other_calls) that 06g converges and the route does not.
# - S1: calls outside the 720 with |v_route - v_g| > e_route + e_g, exact (Fractions of the doubles).
import json, sys, collections
from fractions import Fraction
M = "/Users/ds283/Documents/Code/SecondaryGWKit/var/"
G = M + "runs/levin-phase-floor-06g-census-20261101T114315/checkpoint.jsonl"
A = M + "runs/levin-phase-floor-06a-census-20261027T023143/checkpoint.jsonl"
V8 = M + "levin-phase-floor-06g/census/out/v8_calls.json"
RUNS = {"route1": "levin-phase-floor-06k-census-route1-20261104T103009",
        "route2": "levin-phase-floor-06k-census-route2-20261104T151120",
        "route2lit": "levin-phase-floor-06k-census-route2lit-20261104T185927",
        "route3": "levin-phase-floor-06k-census-route3-20261104T225339"}
def row(uid):
    return "-".join(uid.rsplit("-", 2)[-2:])
sp = set()
for line in open(A):
    d = json.loads(line)["data"]
    if d["rtol"] == "1e-7" and any(s and s["n_sign_changes"] > 0 for s in (d.get("scan") or [])):
        sp.add(f"{d['shard']}-{d['serial']}")
v8 = json.load(open(V8))
flag_agree = all((c["row"] in sp) == c["stationary_point_row"] for c in v8["lost"])
print(f"720 from 06a's scan: {len(sp)} rows; v8 lost entries' stationary_point_row agree: {flag_agree}", flush=True)
def read(path):
    out = {}
    n = 0
    for line in open(path):
        r = json.loads(line); d = r["data"]; n += 1
        assert d["status"] == "done", (path, r["unit"])
        cs = {}
        for kind in ("calls", "other_calls"):
            for c in d[kind]:
                cs[c["index"]] = (c["value"]["hex"], c["abserr"]["hex"], bool(c["converged"]))
        out[r["unit"]] = cs
    return out, n
g, ng = read(G)
print(f"06g: {ng} records, {len(g)} units", flush=True)
def F(h):
    return Fraction(float.fromhex(h))
def fin(h):
    x = float.fromhex(h); return x == x and abs(x) != float("inf")
for name, run in RUNS.items():
    r, nr = read(M + "runs/" + run + "/checkpoint.jsonl")
    assert set(r) == set(g), name
    c1 = {}
    for rt in ("1e-7", "1e-8"):
        L = [c for c in v8["lost"] if c["rtol"] == rt and not c["stationary_point_row"]]
        conv = sum(1 for c in L if r[c["unit"]][c["g"]["index"]][2])
        c1[rt] = (conv, len(L), conv / len(L))
    c2 = []; s1 = []
    for u in g:
        if row(u) in sp:
            continue
        assert set(r[u]) == set(g[u]), (name, u)
        for i, (vg, eg, cg) in g[u].items():
            vr, er, cr = r[u][i]
            if cg and not cr:
                c2.append((u, i))
            if all(fin(x) for x in (vg, eg, vr, er)):
                d = abs(F(vr) - F(vg)); s = F(er) + F(eg)
                if d > s:
                    s1.append((u, i, float(d / s)))
    c1ok = all(x[2] >= 0.9 for x in c1.values())
    print(f"{name}: {nr} records; C1 {'holds' if c1ok else 'fails'} " +
          ", ".join(f"{rt} {a}/{b} ({100*f:.1f} %)" for rt, (a, b, f) in c1.items()) +
          f"; C2 {'holds' if not c2 else 'fails'}: {len(c2)} lost ({dict(collections.Counter(u.split('-')[1] + '-' + u.split('-')[2] for u, _ in c2))})"
          f"; S1 {'holds' if not s1 else 'fails'}: {len(s1)} calls in {len({row(u) for u, _, _ in s1})} rows"
          + (f", worst {max(x[2] for x in s1):.4f} ({max(s1, key=lambda x: x[2])[:2]})" if s1 else ""), flush=True)
