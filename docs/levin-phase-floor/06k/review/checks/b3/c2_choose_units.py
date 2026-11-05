# Orchestrator's review §B.3 check 2: choose two K0 units for re-running, where route 2 (as built)
# fired nowhere in K0 (so the route must equal 06g bit for bit), at different k and rtol, short wall.
import json
V = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06k/"
rows = []
for line in open(V + "state/k0/route2/units.jsonl"):
    r = json.loads(line); d = r["data"]
    fired = sum((c.get("route_06k") or {}).get("n_firings", 0) for k in ("calls", "other_calls") for c in d[k])
    rows.append((r["unit"], d["rtol"], d["k_key"], fired, d["wall_s"], len(d["calls"]) + len(d["other_calls"])))
nf = [x for x in rows if x[3] == 0 and x[0] != "g-1e-7-2-1449"]
print("route2 K0 units", len(rows), "with no firing", len(nf))
for x in sorted(nf, key=lambda x: x[4]): print(x)
