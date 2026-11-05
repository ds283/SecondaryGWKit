# Orchestrator's review §B.3 check 2: choose two K0 units where route B fired nowhere in K0 (so B must
# equal 06g bit for bit), at different k and rtol, short wall, not the development unit.
import json
V = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06l/"
rows = []
for line in open(V + "state/k0/routeB/units.jsonl"):
    r = json.loads(line); d = r["data"]
    fired = sum((c.get("route_06l") or {}).get("n_firings", 0) for k in ("calls", "other_calls") for c in d[k])
    rows.append((r["unit"], d["rtol"], d["k_key"], fired, round(d["wall_s"], 1), len(d["calls"]) + len(d["other_calls"])))
nf = [x for x in rows if x[3] == 0 and x[0] != "g-1e-7-2-1449"]
print("routeB K0 units", len(rows), "with no firing", len(nf))
for x in sorted(nf, key=lambda x: x[4])[:25]: print(x)
