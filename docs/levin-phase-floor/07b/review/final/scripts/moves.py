"""Orchestrator's own descriptive check (levin-phase-floor 07b Part C, §C.2): calls whose value moved against 07a's
census beyond 07a's abserr, and beyond the sum of both abserrs; rows whose total_converged changed. Read only; prints.
usage: python -I -B moves.py
Both checkpoints at my own indices (review/stop3/index07{b,a}.json, file SHA-256 asserted, line SHA-256 checked). Exact."""
import hashlib, json
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

M = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
RB = M / "var/runs/levin-phase-floor-07b-census-20261117T010200"
RA = M / "var/runs/levin-phase-floor-07a-census-20261112T132744"
IX = M / "var/levin-phase-floor-07b/review/stop3"
fx = lambda v: F(float.fromhex(v["hex"]))
ib = json.load(open(IX / "index07b.json")); ia = json.load(open(IX / "index07a.json"))
assert ib["sha256"].startswith("396784bc") and ia["sha256"].startswith("461df2ec")


def rd(f, e):
    f.seek(e[0]); raw = f.read(e[1])
    assert hashlib.sha256(raw).hexdigest() == e[2]
    return json.loads(raw)["data"]


print("script", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
C = Counter(); beyond_both = []; rows_tc = Counter()
fb, fa = open(RB / "checkpoint.jsonl", "rb"), open(RA / "checkpoint.jsonl", "rb")
for key in sorted(ib["units"]):
    b, a = rd(fb, ib["units"][key]), rd(fa, ia["units"][key])
    rt = b["rtol"]
    tb = (b.get("result") or {}).get("total_converged"); ta = (a.get("result") or {}).get("total_converged")
    rows_tc[(rt, ta, tb)] += 1
    ca = {(k, c["index"]): c for k in ("calls", "other_calls") for c in a[k]}
    for k in ("calls", "other_calls"):
        for c in b[k]:
            o = ca[(k, c["index"])]
            dv = abs(fx(c["value"]) - fx(o["value"])); eb, ea = fx(c["abserr"]), fx(o["abserr"])
            kd = "group" if k == "calls" else "analytic"
            if dv > ea:
                C[("moved > abserr_07a", kd, rt)] += 1
            if dv > ea + eb:
                C[("moved > abserr_07a + abserr_07b", kd, rt)] += 1
                C[("  of which converged both", kd, rt, bool(c["converged"] and o["converged"]))] += 1
                beyond_both.append((key, kd, c["index"], c["notify_label"].split("group ")[-1], float(dv / ea), float(dv / eb),
                                    c["converged"], o["converged"], c["num_edge_rule_refusals"], c["num_range_rule_refusals"]))
for k, v in sorted(C.items(), key=str):
    print(k, v)
print("total moved beyond both:", len(beyond_both), " converged both:", sum(1 for x in beyond_both if x[6] and x[7]))
print("by label:", Counter(x[3] for x in beyond_both))
print("rows total_converged (rtol, 07a, 07b):", {str(k): v for k, v in sorted(rows_tc.items(), key=str)})
print("largest |dv|/abserr_07b among converged-both:", sorted((x for x in beyond_both if x[6] and x[7]), key=lambda x: -x[5])[:10])
