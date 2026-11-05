# Orchestrator's review §B.3 check 5: choose one V8-lost call outside the 720 that B converges (06g
# does not), in a unit with a short wall under B, with at least one firing in the call.
import json
M = "/Users/ds283/Documents/Code/SecondaryGWKit/var/"
v8 = json.load(open(M + "levin-phase-floor-06g/census/out/v8_calls.json"))
lost = {(c["unit"], c["g"]["index"]) for c in v8["lost"] if not c["stationary_point_row"]}
units = {u for u, _ in lost}
cand = []
for line in open(M + "runs/levin-phase-floor-06l-census-routeB-20261105T100126/checkpoint.jsonl"):
    r = json.loads(line)
    if r["unit"] not in units: continue
    d = r["data"]
    for kind in ("calls", "other_calls"):
        for c in d[kind]:
            if (r["unit"], c["index"]) in lost and c["converged"] and (c.get("route_06l") or {}).get("n_firings", 0) > 0:
                cand.append((round(d["wall_s"], 1), r["unit"], c["index"], d["k_key"], c["route_06l"]["n_firings"], c["num_regions"]))
cand.sort()
print(len(cand), "candidates; shortest:")
for x in cand[:10]: print(x)
