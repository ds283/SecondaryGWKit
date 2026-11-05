# Orchestrator's review §B.3 check 6: for two referenced calls per route, |v_route - ref| / e_route
# and |v_g - ref| / e_g recomputed from the raw census checkpoints and the reference's own source
# (06g's scores.json "rows" ref_30; 06j's w_calls.csv ref_30), exactly (Fractions; ref from its
# 30-digit string), and compared with the agent's out/partC/h1_calls.csv.
import json, csv
from fractions import Fraction
from decimal import Decimal
M = "/Users/ds283/Documents/Code/SecondaryGWKit/"
T = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf06k-8937060/docs/levin-phase-floor/"
RUNS = {"g": "levin-phase-floor-06g-census-20261101T114315",
        "route1": "levin-phase-floor-06k-census-route1-20261104T103009",
        "route2": "levin-phase-floor-06k-census-route2-20261104T151120",
        "route2lit": "levin-phase-floor-06k-census-route2lit-20261104T185927",
        "route3": "levin-phase-floor-06k-census-route3-20261104T225339"}
PICK = {"route1": ["v03", "z01"], "route2": ["v03", "r56"], "route2lit": ["v03", "c01"], "route3": ["c01", "r56"]}
refs = {}
for r in json.load(open(T + "06g/references/out/scores_stage3/scores.json"))["rows"]:
    refs[r["sid"]] = (r["unit"], int(r["index"]), r["ref_30"], "06g scores.json rows")
for r in csv.DictReader(l for l in open(T + "06j/out/w_calls.csv") if not l.startswith("#")):
    if r["sid"] in ("c01",):
        refs[r["sid"]] = (r["unit"], int(r["index"]), r["ref_30"], "06j w_calls.csv")
need = {s for v in PICK.values() for s in v}
units = {refs[s][0] for s in need}
vals = {}
for name, run in RUNS.items():
    for line in open(M + "var/runs/" + run + "/checkpoint.jsonl"):
        if not any(f'"{u}"' in line[:200] for u in units):
            continue
        rec = json.loads(line)
        if rec["unit"] not in units:
            continue
        for kind in ("calls", "other_calls"):
            for c in rec["data"][kind]:
                vals[(name, rec["unit"], c["index"])] = (c["value"]["hex"], c["abserr"]["hex"])
agent = {r["rid"]: r for r in csv.DictReader(l for l in open(M + "var/levin-phase-floor-06k/out/partC/h1_calls.csv") if not l.startswith("#"))}
def ratio(name, u, i, ref):
    v, e = vals[(name, u, i)]
    return float(abs(Fraction(float.fromhex(v)) - ref) / Fraction(float.fromhex(e)))
allok = True
for route, sids in PICK.items():
    for s in sids:
        u, i, r30, src = refs[s]
        ref = Fraction(Decimal(r30))
        rg = ratio("g", u, i, ref); rr = ratio(route, u, i, ref)
        ag = float(agent[s]["ratio_g"]); ar = float(agent[s][f"{route}_ratio"])
        ok = abs(rg - ag) <= 1e-12 * max(1, ag) and abs(rr - ar) <= 1e-12 * max(1, ar)
        allok &= ok
        print(f"{route} {s} ({src}; {u} #{i}): |v_g-ref|/e_g {rg:.10g} (agent {ag:.10g}); "
              f"|v_route-ref|/e_route {rr:.10g} (agent {ar:.10g}); bound max(1, g) {max(1, rg):.6g}; "
              f"H1 on this call {'holds' if rr <= max(1, rg) else 'fails'}; agrees with agent {ok}")
print("ALL AGREE:", allok)
