"""Orchestrator's exact re-score of R32's S1, S2, S4 (07c Part C). Reads the rehearsal agent's score files for R32's
values, checks each reference string against its source, checks R32's values against 07b's R31 score files, and
recomputes |v - ref| / abserr in exact Fractions. usage: s124_exact.py"""
import glob, hashlib, json
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path
M = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var")
R = M / "levin-phase-floor-07c/rehearsal"; R31 = M / "levin-phase-floor-07b/rehearsal"
V1 = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-cc3368c/docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
hx = lambda s: F(float.fromhex(s))
print("script", sha(__file__))
for p in (R/"out/s1_scores.json", R/"out/s2_scores.json", R/"out/block_scores.json", R31/"out/s1_scores.json",
          R31/"out/s2_scores.json", R31/"out/block_scores.json", V1):
    print("input", p, sha(p))
# S1
v1 = {r["wid"]: r for r in json.load(open(V1))["V1"]}
s31 = {r["wid"]: r for r in json.load(open(R31/"out/s1_scores.json"))["rows"]}
worst = 0; ok = True
for r in json.load(open(R/"out/s1_scores.json"))["rows"]:
    src = v1[r["wid"]]["readings"]["fcc_r2"]
    assert src["value_30"] == r["ref_30"] and src["value_hex"] == r["ref_hex"], r["wid"]
    ratio = abs(hx(r["v32"]) - F(Decimal(r["ref_30"]))) / hx(r["abserr32"])
    eq31 = (r["v32"], r["abserr32"]) == (s31[r["wid"]]["v31"], s31[r["wid"]]["abserr31"])
    ok &= ratio <= 1 and r["converged32"] and r["location_ok"]
    worst = max(worst, ratio)
    print(f"S1 {r['wid']} ratio {float(ratio):.6g} (agent {r['ratio_v32']}) conv {r['converged32']} loc {r['location_ok']} "
          f"equal_R31 {eq31} (agent {r['equal_to_R31']}) chase {r['chase_07c'] if not isinstance(r['chase_07c'], dict) else r['chase_07c'].get('n_stopped', r['chase_07c'])}")
print("S1 holds", ok, "worst", float(worst))
# S2
s31 = {r["p1"]: r for r in json.load(open(R31/"out/s2_scores.json"))["rows"]}
ok = True
for r in json.load(open(R/"out/s2_scores.json"))["rows"]:
    o = s31[r["p1"]]
    assert o["ref_30"] == r["ref_30"] and o["ref_hex"] == r["ref_hex"]
    files = glob.glob(str(R/f"out/p1_references/p1_row{r['p1']}.json")) or glob.glob(str(R31/f"out/p1_references/p1_row{r['p1']}.json"))
    insrc = any(r["ref_30"] in open(f).read() for f in files)
    ratio = abs(hx(r["R32"]["value"]) - F(Decimal(r["ref_30"]))) / hx(r["R32"]["abserr"])
    eq31 = (r["R32"]["value"], r["R32"]["abserr"]) == (o["R31"]["value"], o["R31"]["abserr"]) if isinstance(o.get("R31"), dict) else None
    ok &= ratio <= 1 and r["location_ok"]
    print(f"S2 row{r['p1']} ratio {float(ratio):.6g} (agent {r['R32']['ratio']}) loc {r['location_ok']} ref same as R31's review {True} "
          f"ref found in p1_row file {insrc} ({files}) equal_R31 {eq31}")
print("S2 holds", ok)
# S4
b31 = {json.dumps(b["key"]) + b["block"]: b for b in json.load(open(R31/"out/block_scores.json"))["blocks"]}
ok = True
for b in json.load(open(R/"out/block_scores.json"))["blocks"]:
    o = b31[json.dumps(b["key"]) + b["block"]]
    oref = o.get("reference"); oval = (o.get("R31") or o.get("R30") or {})
    ratio = abs(hx(b["R32"]["value"]) - hx(b["reference"])) / hx(b["R32"]["abserr"])
    ok &= ratio <= 1 and b["location_ok"] and b["num_guarded_regions"] == 0 and b["R32"]["converged"]
    print(f"S4 {b['block']} {b['key'][3]:.6g} ratio {float(ratio):.6g} (agent {b['R32']['ratio']}) loc {b['location_ok']} guarded {b['num_guarded_regions']} "
          f"ref equal to R31 review's {oref == b['reference']} value/abserr equal R31 {(oval.get('value'), oval.get('abserr')) == (b['R32']['value'], b['R32']['abserr'])}")
print("S4 holds", ok)
