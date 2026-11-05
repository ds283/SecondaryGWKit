# Orchestrator's review §B.3 check 5 for 06l: C1, C2 (with each loss's cause) and S1 recomputed from
# the raw census checkpoints of route B and 06g, with my own reader (06k's review c5b reader adapted);
# also the calls route 1 converges and B does not, and B converges and route 1 does not, outside the
# 720, per rtol; and K1 from B's firing records (Levin parents, outside the 720).
# - The 720: rows (shard-serial) whose 1e-7 unit in 06a's census scan shows a sign change of theta'.
# - S1: |v_B - v_g| > e_B + e_g exactly (Fractions of the doubles), calls + other_calls.
import json, collections
from fractions import Fraction
M = "/Users/ds283/Documents/Code/SecondaryGWKit/var/"
G = M + "runs/levin-phase-floor-06g-census-20261101T114315/checkpoint.jsonl"
R1 = M + "runs/levin-phase-floor-06k-census-route1-20261104T103009/checkpoint.jsonl"
B = M + "runs/levin-phase-floor-06l-census-routeB-20261105T100126/checkpoint.jsonl"
A = M + "runs/levin-phase-floor-06a-census-20261027T023143/checkpoint.jsonl"
V8 = M + "levin-phase-floor-06g/census/out/v8_calls.json"
OUT = M + "levin-phase-floor-06l/review/checks/c5_lists.json"
def row(uid):
    return "-".join(uid.rsplit("-", 2)[-2:])
sp = set()
for line in open(A):
    d = json.loads(line)["data"]
    if d["rtol"] == "1e-7" and any(s and s["n_sign_changes"] > 0 for s in (d.get("scan") or [])):
        sp.add(f"{d['shard']}-{d['serial']}")
v8 = json.load(open(V8))
print(f"720 from 06a's scan: {len(sp)} rows; v8 flags agree: "
      f"{all((c['row'] in sp) == c['stationary_point_row'] for c in v8['lost'])}", flush=True)
k1 = collections.Counter(); k1_max = 0.0
def read(path, firings=False):
    global k1_max
    out = {}; n = 0
    for line in open(path):
        r = json.loads(line); d = r["data"]; n += 1
        assert d["status"] == "done", (path, r["unit"])
        cs = {}
        for kind in ("calls", "other_calls"):
            for c in d[kind]:
                cs[c["index"]] = (c["value"]["hex"], c["abserr"]["hex"], bool(c["converged"]),
                                  bool(c.get("node_limited")), int(c.get("n_at_depth_max") or 0),
                                  int(c.get("num_untrusted_regions") or 0))
                if firings and row(r["unit"]) not in sp and c.get("route_06l"):
                    for rec in c["route_06l"]["records"]:
                        if rec["kind"] == "fired":
                            par = "Levin" if not rec["is_direct"] else "CC"
                            q = rec["delta1"] / rec["phi_P"] if rec["phi_P"] else float("inf")
                            k1[(par, q >= 5.8)] += 1
                            if par == "Levin":
                                k1_max = max(k1_max, q)
        out[r["unit"]] = cs
    return out, n
g, ng = read(G); print(f"06g: {ng} records, {len(g)} units", flush=True)
b, nb = read(B, firings=True); print(f"B: {nb} records, {len(b)} units", flush=True)
r1, nr1 = read(R1); print(f"route 1: {nr1} records, {len(r1)} units", flush=True)
assert set(b) == set(g) == set(r1)
F = lambda h: Fraction(float.fromhex(h))
fin = lambda h: (lambda x: x == x and abs(x) != float("inf"))(float.fromhex(h))
c1 = {}
for rt in ("1e-7", "1e-8"):
    L = [c for c in v8["lost"] if c["rtol"] == rt and not c["stationary_point_row"]]
    conv = sum(1 for c in L if b[c["unit"]][c["g"]["index"]][2])
    c1[rt] = (conv, len(L), conv / len(L))
c2 = []; s1 = []; r1_not_b = collections.Counter(); b_not_r1 = collections.Counter(); lists = {"r1_not_b": [], "b_not_r1": []}
for u in g:
    if row(u) in sp:
        continue
    rt = u.split("-")[1] + "-" + u.split("-")[2]
    for i, (vg, eg, cg, *_g) in g[u].items():
        vb, eb, cb, nlb, dmb, unb = b[u][i]
        if cg and not cb:
            c2.append((u, i, "node_limited" if nlb else ("depth_max" if dmb else ("untrusted" if unb else "other"))))
        if all(fin(x) for x in (vg, eg, vb, eb)) and abs(F(vb) - F(vg)) > F(eb) + F(eg):
            s1.append((u, i, float(abs(F(vb) - F(vg)) / (F(eb) + F(eg)))))
        c_r1 = r1[u][i][2]
        if c_r1 and not cb:
            r1_not_b[rt] += 1; lists["r1_not_b"].append((u, i))
        if cb and not c_r1:
            b_not_r1[rt] += 1; lists["b_not_r1"].append((u, i))
print("C1 " + ("holds" if all(x[2] >= 0.9 for x in c1.values()) else "fails") + ": " +
      ", ".join(f"{rt} {a}/{n} ({100*f:.1f} %)" for rt, (a, n, f) in c1.items()))
print(f"C2 {'holds' if all(x[2] == 'node_limited' for x in c2) else 'fails'}: {len(c2)} converged under 06g and not B; causes "
      f"{dict(collections.Counter(x[2] for x in c2))}; by rtol {dict(collections.Counter(u.split('-')[1]+'-'+u.split('-')[2] for u,_,_ in c2))}")
print(f"S1 {'holds' if not s1 else 'fails'}: {len(s1)} calls in {len({(u.rsplit('-',2)[0], row(u)) for u,_,_ in s1})} census units, "
      f"{len({row(u) for u,_,_ in s1})} shard-serial rows" + (f", worst ratio {max(x[2] for x in s1):.4f}" if s1 else ""))
print(f"route 1 converges, B does not (outside the 720): {dict(r1_not_b)}; B converges, route 1 does not: {dict(b_not_r1)}")
print(f"K1 {'holds' if k1[('Levin', True)] == 0 else 'fails'}: firings outside the 720 by (parent, d1/phi_P >= 5.8): {dict(k1)}; "
      f"largest Levin-parent d1/phi_P {k1_max:.4g}")
lists["c2"] = c2; lists["s1"] = s1
json.dump(lists, open(OUT, "w"), indent=0)
