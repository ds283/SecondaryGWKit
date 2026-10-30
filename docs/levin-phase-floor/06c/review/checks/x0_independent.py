"""Orchestrator's independent X0 clause-2/3 check (06c Part B review, check 6).

Compares each X0 parent unit's per-call value and abserr (float.hex) with the same unit in 06b's
census checkpoint (read only), and the two g determinism runs with each other.
"""

import json
import sys

C = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06c/census/state/X0/"
B = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-06b-census-20261028T161212/checkpoint.jsonl"


def unit_data(path):
    out = {}
    for line in open(path):
        d = json.loads(line)
        dd = d.get("data")
        if isinstance(dd, dict) and "calls" in dd:
            out[dd["unit"]] = dd
    return out


def h(x):
    if isinstance(x, dict):
        return x.get("hex", x.get("repr"))
    if isinstance(x, float):
        return x.hex()
    return x


def calls(dd):
    return [(c["index"], h(c["value"]), h(c["abserr"]), c.get("converged"), c.get("num_regions"), c.get("max_depth"))
            for c in dd["calls"]]


want = {"P1": "b-1e-7-2-38302", "P2": "b-1e-7-2-2095", "P3": "b-1e-8-0-1971"}
b06 = {}
for line in open(B):
    d = json.loads(line)
    if d.get("unit") in want.values():
        b06[d["unit"]] = d["data"]
ok = True
for w, bunit in want.items():
    p = unit_data(C + w + "/emit.jsonl")
    (punit, pd), = p.items()
    bd = b06[bunit]
    a, b = calls(pd), calls(bd)
    same = a == b
    st = (h(pd["stored"]["total"]), h(bd["stored"]["total"]))
    print(w, punit, "vs 06b", bunit, "calls", len(a), len(b), "per-call (index,value,abserr,converged,regions,depth) equal:", same)
    for x, y in zip(a, b):
        print("   p  ", x)
        if x != y:
            print("   06b", y)
    ok &= same
g = [unit_data(C + w + "/emit.jsonl") for w in ("G1", "G2")]
(gu1, g1), = g[0].items()
(gu2, g2), = g[1].items()
same = calls(g1) == calls(g2) and json.dumps(g1["calls"], sort_keys=True, default=str) == json.dumps(
    g2["calls"], sort_keys=True, default=str) if False else calls(g1) == calls(g2)
print("G1", gu1, "G2", gu2, "per-call equal:", same, "n calls", len(calls(g1)))
for x in calls(g1):
    print("   g  ", x)
ok &= same
print("X0 clauses 2 and 3 (independent):", "HOLD" if ok else "FAIL")
