"""Orchestrator's independent recount of the depth20-tiling cheap pass (all 262 lines). Read-only.
Lineage: from the census call's x_span, halve at (b + a) / 2.0 towards the recorded leaf; at each
level, tiling of the computed end nodes (chebyshev_matrices at order 24) of parent and halves.
Levels from deep_lines_06c.json: [branch, residual, floor, truncation, e, ratio, ...] (hex)."""
import collections, hashlib, json, statistics, sys
import AdaptiveLevin.levin_quadrature as LQ
print("tree", LQ.__file__, "script sha256", hashlib.sha256(open(__file__,'rb').read()).hexdigest())
DL = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06c/census/out/deep_lines_06c.json"
CK = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-06c-census-20261029T130214/checkpoint.jsonl"
lines = json.load(open(DL)); want = {(l["unit"], l["index"]) for l in lines}
units = {u for u, _ in want}; spans = {}
with open(CK) as f:
    for raw in f:
        if not any(u in raw[:80] for u in units): continue
        dd = json.loads(raw).get("data") or {}
        for c in (dd.get("calls") or []) + (dd.get("other_calls") or []):
            if (dd.get("unit"), c.get("index")) in want and "x_span" in c:
                spans[(dd["unit"], c["index"])] = tuple(float.fromhex(v) if isinstance(v, str) else float.fromhex(v["hex"]) for v in c["x_span"])
print("spans found", len(spans), "of", len(want))
def ends(a, b):
    x, _ = LQ.chebyshev_matrices((a, b), 24); return float(x[-1]), float(x[0])
H = lambda s: float.fromhex(s) if isinstance(s, str) else float(s)
tally = collections.Counter(); ratios = []; leaf_ok = 0; nlev_all = collections.Counter()
for l in lines:
    a, b = spans[(l["unit"], l["index"])]
    ls, le = float.fromhex(l["leaf_start"]), float.fromhex(l["leaf_end"])
    path = []
    for lev in l["levels"]:
        path.append((a, b))
        m = (b + a) / 2.0
        if ls >= a and le <= m: b = m
        else: a = m
    # path[i] is the span at level i; the final level is the leaf itself
    leaf_ok += (path[-1] == (ls, le))
    n = len(l["levels"])
    for i, lev in enumerate(l["levels"]):
        A, B = path[i]; M = (B + A) / 2.0
        loP, hiP = ends(A, B); loL, topL = ends(A, M); botR, hiR = ends(M, B)
        tile = loL == loP and hiR == hiP and topL == botR
        res, flo = H(lev[1]), H(lev[2]); br = lev[0]
        flag = res > 10 * flo
        key = (br, "res>10floor" if flag else "res<=10floor", "tile" if tile else "no tile")
        nlev_all[key] += 1
        if i >= n - 5:
            tally[key] += 1
            if br == "Levin" and flo > 0: ratios.append(res / flo)
print("lineage reproduces recorded leaf:", leaf_ok, "of", len(lines))
print("last five levels:"); [print("  ", k, v) for k, v in sorted(tally.items())]
print("all levels:"); [print("  ", k, v) for k, v in sorted(nlev_all.items())]
ratios.sort(); print("Levin residual/floor last five: n", len(ratios), "median %.3g max %.3g" % (statistics.median(ratios), ratios[-1]))
