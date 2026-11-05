# Orchestrator's own reader for Gate K0 (review stop 1): each K0 run's unit records against 06g's
# census checkpoint. Per call (calls + other_calls) every key but "elapsed" and "route_06k" must be
# equal as JSON; for a route, only calls with route_06k n_firings == 0 are compared (clause 2).
# The unit's "result" and the region-pickle record hash are compared where nothing fired.
import json, sys
V = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06k/"
C = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-06g-census-20261101T114315/checkpoint.jsonl"
units = json.load(open(V + "out/k0/units.json"))
uids = units["units"] if isinstance(units, dict) and "units" in units else units
uids = [u if isinstance(u, str) else u["uid"] for u in uids]
want = set(uids)
ref = {}
for line in open(C):
    r = json.loads(line)
    if r["unit"] in want:
        ref[r["unit"]] = r["data"]
print("K0 units", len(uids), "found in 06g checkpoint", len(ref))
SKIP = {"elapsed", "route_06k"}
for mode in ["identity", "route1", "route2", "route3"]:
    recs = {}
    for line in open(V + f"state/k0/{mode}/units.jsonl"):
        r = json.loads(line)
        recs[r["unit"]] = r["data"]
    n_calls = n_cmp = n_fired = n_bad = 0
    n_result_eq = n_result_cmp = n_pkl_eq = n_pkl_cmp = 0
    bad = []
    for u in uids:
        d, g = recs[u], ref[u]
        assert d["status"] == "done", (mode, u, d["status"])
        unit_fired = False
        for kind in ("calls", "other_calls"):
            assert len(d[kind]) == len(g[kind]), (mode, u, kind)
            for a, b in zip(d[kind], g[kind]):
                n_calls += 1
                rk = a.get("route_06k") or {}
                if rk.get("n_firings", 0):
                    n_fired += 1
                    unit_fired = True
                    continue
                n_cmp += 1
                ka = {k: v for k, v in a.items() if k not in SKIP}
                kb = {k: v for k, v in b.items() if k not in SKIP}
                if json.dumps(ka, sort_keys=True) != json.dumps(kb, sort_keys=True):
                    n_bad += 1
                    bad.append((u, kind, a["index"], sorted(k for k in set(ka) | set(kb) if ka.get(k) != kb.get(k))))
        if not unit_fired:
            n_result_cmp += 1
            n_result_eq += json.dumps(d["result"], sort_keys=True) == json.dumps(g["result"], sort_keys=True)
            n_pkl_cmp += 1
            n_pkl_eq += d.get("pickle_record_sha256") == g.get("pickle_record_sha256")
    print(f"{mode}: calls {n_calls}, fired {n_fired}, compared {n_cmp}, differing {n_bad}; "
          f"units with no firing {n_result_cmp}: result equal {n_result_eq}, region-pickle record hash equal {n_pkl_eq}")
    for b in bad[:10]:
        print("   DIFF", b)
