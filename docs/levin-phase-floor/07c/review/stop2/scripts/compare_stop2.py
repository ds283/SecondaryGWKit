"""Orchestrator's own stop-2 comparer (07c Part C). Compares units' Levin calls (group `calls` and analytic
`other_calls`) by float.hex in value and abserr, and converged, num_regions, max_depth; and unit totals.
usage: compare_stop2.py <A units.jsonl or checkpoint lines> <B units.jsonl> [<S replay dir>]"""
import json, sys, os, hashlib
print("script", hashlib.sha256(open(__file__,'rb').read()).hexdigest())
def load(p):
    out = {}
    for line in open(p):
        d = json.loads(line); data = d.get("data", d)
        out[d["unit"]] = data
    return out
def hx(x):
    if isinstance(x, dict): return x.get("hex")
    if isinstance(x, str): return x
    return float(x).hex() if x is not None else None
def calls(data):
    res = {}
    for kind in ("calls", "other_calls"):
        for c in data.get(kind) or data.get("result", {}).get(kind) or []:
            res[(kind, c["index"])] = (hx(c["value"]), hx(c["abserr"]), c["converged"], c["num_regions"], c["max_depth"], c.get("num_chase_stops"))
    return res
A, B = load(sys.argv[1]), load(sys.argv[2])
for u in sorted(B):
    ca, cb = calls(A[u]), calls(B[u])
    diff = [k for k in ca if ca[k][:5] != cb.get(k, (None,))[:5]]
    stops = {k: cb[k][5] for k in cb if cb[k][5]}
    ta = [hx(A[u].get("result", {}).get(f)) for f in ("total", "total_abserr")]
    tb = [hx(B[u].get("result", {}).get(f)) for f in ("total", "total_abserr")]
    print(u, "calls A", len(ca), "B", len(cb), "keys equal", set(ca) == set(cb), "differing", len(diff), "totals equal", ta == tb, ta != [None, None], "B chase stops", stops)
    for k in diff: print("   diff", k, ca[k], cb.get(k))
if len(sys.argv) > 3:
    for u in B:
        cb = calls(B[u])
        for (kind, i), t in cb.items():
            f = os.path.join(sys.argv[3], "c-%s-%d.json" % (u[2:], i))
            if os.path.exists(f):
                r = json.load(open(f))["replay"]
                s = (hx(r["value"]), hx(r["abserr"]), r["converged"], r["num_regions"], r["max_depth"])
                print("  vs (S) replay", os.path.basename(f), kind, "equal" if s == t[:5] else ("DIFFER", s, t))
