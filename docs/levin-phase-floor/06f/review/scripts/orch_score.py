"""06f review, check 5: F1-F6 re-scored by the orchestrator from the agent's tables (m1_calls.csv,
m2_calls.csv, every M2 call's nodes.json.gz and result.json), against prompt section 4's rules."""
import csv, gzip, hashlib, json, statistics
from pathlib import Path
V = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06f")
print("# script sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
for n in ("m1_calls.csv", "m2_calls.csv"):
    print("#", n, hashlib.sha256((V / "out" / n).read_bytes()).hexdigest())
m1 = list(csv.DictReader(open(V / "out/m1_calls.csv")))
z2 = [r for r in m1 if r["unit"] == "g-1e-8-0-33995"][0]
r1 = float(z2["sum_absE"]) / float(z2["abs_sumE"])
print(f"F1: sum|E|/|sum E| = {r1:.4g} (rule >= 10) -> {'holds' if r1 >= 10 else 'fails'}")
sc, req = float(z2["signed_combination_error"]), float(z2["request"])
lin = float(z2["signed_combination_error_linear"])
print(f"F2: signed-combination {sc:.4g} (literal {lin:.4g}) vs request {req:.4g} -> {'holds' if max(sc, lin) <= req else 'fails'}")
big = [r for r in m1 if int(r["N_cc"]) >= 100]
rat = [float(r["sum_absE"]) / float(r["abs_sumE"]) for r in big]
rat_noz2 = [float(r["sum_absE"]) / float(r["abs_sumE"]) for r in big if r["unit"] != "g-1e-8-0-33995"]
print(f"F3: {len(big)} M1 calls with N_cc >= 100 (by rtol: {sorted(r['rtol'] for r in big)}); ratios {sorted(round(x, 3) for x in rat)}")
print(f"    median {statistics.median(rat):.4g}; without Z2's call ({len(rat_noz2)}) {statistics.median(rat_noz2):.4g} -> {'holds' if min(statistics.median(rat), statistics.median(rat_noz2)) >= 10 else 'see numbers'}")
m2 = list(csv.DictReader(open(V / "out/m2_calls.csv")))
nv = nn = 0; calls_v = []; worst = 0
for r in m2:
    tag = f"{r['unit']}_{r['index']}"
    rows = json.load(gzip.open(V / "out/replay" / tag / "nodes.json.gz", "rt"))
    v = [x for x in rows if abs(x["delta_theta"]) > x["theta_abserr"]]
    nn += len(rows); nv += len(v)
    if v:
        calls_v.append((tag, len(v), len(rows), max(abs(x["delta_theta"]) / x["theta_abserr"] for x in v)))
print(f"F4: {len(m2)} M2 calls, {nn} nodes, {nv} with |dtheta| > theta_abserr, on {len(calls_v)} calls: {calls_v} -> {'holds' if nv == 0 else 'fails'}")
f5, f6 = [], []
for r in m2:
    res = json.loads((V / "out/replay" / f"{r['unit']}_{r['index']}" / "result.json").read_text())["M2"]
    mx = max(abs(res["dI_plus"]), abs(res["dI_minus"]))
    f5.append(res["D"] / mx)
    f6.append(res["sum_abs_dI_leaf"] / mx)
print(f"F5: D/max|dI| median {statistics.median(f5):.4g} (range {min(f5):.3g}-{max(f5):.3g}) -> {'holds' if statistics.median(f5) >= 10 else 'fails'}")
print(f"F6: sum|dI_leaf|/max|dI| median {statistics.median(f6):.4g} (range {min(f6):.4g}-{max(f6):.4g}) -> {'holds' if statistics.median(f6) >= 10 else 'fails'}")
