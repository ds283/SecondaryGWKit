# Orchestrator's review §B.3 check 5: the recovered call g-1e-8-3-825 #0 (V8-lost under 06g,
# converged under B) re-run by the orchestrator under route B (recovered_rerun/), compared with B's
# census record and 06g's: value and abserr by hex, num_regions, converged, n_firings; and the whole
# call record (but elapsed) as JSON against B's census.
import json
M = "/Users/ds283/Documents/Code/SecondaryGWKit/var/"
H = M + "levin-phase-floor-06l/review/checks/recovered_rerun/routeB/units.jsonl"
U, I = "g-1e-8-3-825", 0
def call(d):
    for kind in ("calls", "other_calls"):
        for c in d[kind]:
            if c["index"] == I: return c
def rec(path):
    for line in open(path):
        r = json.loads(line)
        if r["unit"] == U: return r["data"]
mine = call(rec(H))
b = call(rec(M + "runs/levin-phase-floor-06l-census-routeB-20261105T100126/checkpoint.jsonl"))
g = call(rec(M + "runs/levin-phase-floor-06g-census-20261101T114315/checkpoint.jsonl"))
f = lambda c: {"value": c["value"]["hex"], "abserr": c["abserr"]["hex"], "num_regions": c["num_regions"], "converged": c["converged"],
               "n_firings": (c.get("route_06l") or {}).get("n_firings"), "node_limited": c.get("node_limited"), "n_at_depth_max": c.get("n_at_depth_max")}
print("re-run  ", f(mine)); print("B census", f(b)); print("06g     ", f(g))
print("re-run equals B's census on these:", f(mine) == f(b))
strip = lambda c: json.dumps({k: v for k, v in c.items() if k != "elapsed"}, sort_keys=True)
print("whole call record (but elapsed) equal to B's census:", strip(mine) == strip(b))
print("|v_B - v_g| / (e_B + e_g):", abs(float.fromhex(b["value"]["hex"]) - float.fromhex(g["value"]["hex"])) / (float.fromhex(b["abserr"]["hex"]) + float.fromhex(g["abserr"]["hex"])))
