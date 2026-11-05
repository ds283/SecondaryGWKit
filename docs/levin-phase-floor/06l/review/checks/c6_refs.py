# Orchestrator's review §B.3 check 6 (06l; 06k's review c6 adapted): for three referenced calls,
# |v_B - ref| / e_B, |v_g - ref| / e_g and |v_route1 - ref| / e_route1 recomputed from the raw census
# checkpoints and the reference's own source (06g's scores.json "rows" ref_30; 06j's w_calls.csv
# ref_30), exactly (Fractions; ref from its 30-digit string), against the agent's out/partC/h1_calls.csv.
import json, csv
from fractions import Fraction
from decimal import Decimal
M = "/Users/ds283/Documents/Code/SecondaryGWKit/"
T = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf06l-af4c4dc/docs/levin-phase-floor/"
RUNS = {"g": "levin-phase-floor-06g-census-20261101T114315",
        "route1": "levin-phase-floor-06k-census-route1-20261104T103009",
        "routeB": "levin-phase-floor-06l-census-routeB-20261105T100126"}
PICK = ["v03", "z01", "c01", "r17", "z15"]  # the last two added: calls whose value B moves (z01 and c01 are unchanged by every route)
refs = {}
for r in json.load(open(T + "06g/references/out/scores_stage3/scores.json"))["rows"]:
    refs[r["sid"]] = (r["unit"], int(r["index"]), r["ref_30"], "06g scores.json rows")
for r in csv.DictReader(l for l in open(T + "06j/out/w_calls.csv") if not l.startswith("#")):
    if r["sid"] in PICK:
        refs[r["sid"]] = (r["unit"], int(r["index"]), r["ref_30"], "06j w_calls.csv")
units = {refs[s][0] for s in PICK}
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
agent = {r["rid"]: r for r in csv.DictReader(l for l in open(M + "var/levin-phase-floor-06l/out/partC/h1_calls.csv") if not l.startswith("#"))}
def ratio(name, u, i, ref):
    v, e = vals[(name, u, i)]
    return float(abs(Fraction(float.fromhex(v)) - ref) / Fraction(float.fromhex(e)))
allok = True
for s in PICK:
    u, i, r30, src = refs[s]
    ref = Fraction(Decimal(r30))
    rg, r1, rb = (ratio(n, u, i, ref) for n in ("g", "route1", "routeB"))
    a = agent[s]
    ag, a1, ab = float(a["ratio_g"]), float(a["route1_ratio"]), float(a["routeB_ratio"])
    ok = all(abs(x - y) <= 1e-12 * max(1, y) for x, y in ((rg, ag), (r1, a1), (rb, ab)))
    allok &= ok
    print(f"{s} ({src}; {u} #{i}; in_720 {a['in_720']}): 06g {rg:.10g} (agent {ag:.10g}); route 1 {r1:.10g} (agent {a1:.10g}); "
          f"B {rb:.10g} (agent {ab:.10g}); H1 bound max(1, g) {max(1, rg):.6g}: {'holds' if rb <= max(1, rg) else 'fails'}; agrees {ok}")
print("ALL AGREE:", allok)
