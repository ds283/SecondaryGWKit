# Orchestrator's review §B.3 check 5 (second part): one recovered call re-run by the orchestrator under its
# route (V8 lost call g-1e-7-3-825 #0, unconverged under 06g, converged under route 1), the whole unit re-run
# with scripts/run_k0.py route1 (recovered_rerun/); the call's value, abserr, num_regions, converged and
# firing count, and the whole call record but "elapsed", against route 1's census record; and the unit's
# other calls likewise.
import json
H = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06k/review/checks/b3/recovered_rerun/"
R = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-06k-census-route1-20261104T103009/checkpoint.jsonl"
G = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-06g-census-20261101T114315/checkpoint.jsonl"
U, I = "g-1e-7-3-825", 0
def rec(path):
    for line in open(path):
        if f'"{U}"' in line[:200]:
            r = json.loads(line)
            if r["unit"] == U:
                return r["data"]
mine = rec(H + "route1/units.jsonl"); cen = rec(R); g = rec(G)
def call(d, i):
    return [c for c in d["calls"] + d["other_calls"] if c["index"] == i][0]
a, b, c = call(mine, I), call(cen, I), call(g, I)
for k in ("value", "abserr"):
    print(f"{k}: re-run {a[k]['hex']}  census {b[k]['hex']}  equal {a[k]['hex'] == b[k]['hex']}  (06g {c[k]['hex']})")
for k in ("num_regions", "converged", "max_depth", "num_untrusted_regions"):
    print(f"{k}: re-run {a[k]}  census {b[k]}  equal {a[k] == b[k]}  (06g {c[k]})")
fa, fb = a["route_06k"]["n_firings"], b["route_06k"]["n_firings"]
print(f"n_firings: re-run {fa}  census {fb}  equal {fa == fb}; firing records equal {json.dumps(a['route_06k'], sort_keys=True) == json.dumps(b['route_06k'], sort_keys=True)}")
n = bad = 0
for kind in ("calls", "other_calls"):
    for x, y in zip(mine[kind], cen[kind]):
        n += 1
        kx = {k: v for k, v in x.items() if k != "elapsed"}; ky = {k: v for k, v in y.items() if k != "elapsed"}
        bad += json.dumps(kx, sort_keys=True) != json.dumps(ky, sort_keys=True)
print(f"unit {U}: {n} call records, differing from route 1's census record (elapsed aside) {bad}; "
      f"result equal (compute_time aside) {json.dumps({k: v for k, v in mine['result'].items() if k != 'compute_time'}, sort_keys=True) == json.dumps({k: v for k, v in cen['result'].items() if k != 'compute_time'}, sort_keys=True)}")
