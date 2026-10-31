"""06f review, check 5 (sample): the orchestrator's own count of node- and declared-dominated calls in
06e's checkpoint (largest of t_value_rounding, t_argument, node_term, declared over the
floor-attributed CC leaves' sums, first of equals), and of node-dominated calls with n_cc_leaves >= 100,
with the sampled calls' strata re-derived."""
import collections, hashlib, json
from pathlib import Path
RUN = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-06e-census-20261030T081318")
V = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06f")
T = ("t_value_rounding", "t_argument", "node_term", "declared")
print("# script sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
cnt = collections.Counter(); big = collections.defaultdict(list); dom = {}
for line in open(RUN / "checkpoint.jsonl"):
    d = json.loads(line)["data"]
    for c in (d.get("calls") or []) + (d.get("other_calls") or []):
        fl = c.get("cc_floor")
        if not fl or not fl.get("n_floor_attributed"):
            continue
        s = fl["sums_floor_attributed_cc_leaves"]
        vals = [float.fromhex(s[t]["hex"]) for t in T]
        top = T[vals.index(max(vals))]
        cnt[(d["rtol"], top)] += 1
        dom[(d["unit"], c["index"])] = (top, fl["n_cc_leaves"], d["row"] if "row" in d else None)
        if top == "node_term" and fl["n_cc_leaves"] >= 100:
            big[d["rtol"]].append((d["unit"], c["index"], fl["n_cc_leaves"]))
for k in sorted(cnt):
    print(k, cnt[k])
for rt, L in big.items():
    print(f"node-dominated with n_cc >= 100 at {rt}: {len(L)}: {sorted(L)}")
sample = json.loads((V / "out/capture.json").read_text())
bad = []
for e in sample:
    top, ncc, _ = dom[(e["unit"], e["index"])]
    want = "node_term" if e["sample"] == "M1" else "declared"
    if top != want or ncc != e["n_cc_leaves"]:
        bad.append((e["unit"], e["index"], top, ncc))
print("sampled calls:", len(sample), "with the wrong stratum or n_cc:", bad)
