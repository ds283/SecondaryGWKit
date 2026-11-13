"""Orchestrator's own count of W5's forced node-partition solves (07a Part C §C.2). Read only; prints only.
Per call at h: route_07a.partition.forced, n_partition_nonforced, n_forced, and guard_07 points by kind; how many calls
carry forced partition sums, how many root-located calls there are, and whether the two sets coincide."""
import hashlib, json
from collections import Counter
from pathlib import Path
R = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-07a-census-20261112T132744/checkpoint.jsonl")
print("script", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
C = Counter(); fp, roots = set(), set(); per = Counter()
h = hashlib.sha256()
with open(R, "rb") as f:
    for raw in f:
        h.update(raw); d = json.loads(raw)["data"]
        for kind in ("calls", "other_calls"):
            for c in d[kind]:
                ra = c["route_07a"]; k = (d["unit"], kind, c["index"])
                fo = ra["partition"]["forced"]
                C["forced_sum"] += fo; C["nonforced_sum"] += ra["n_partition_nonforced"]
                if fo: fp.add(k); per[fo] += 1
                if any(p["kind"] == "root" for p in c["guard_07"]["points"]): roots.add(k)
                if c["guard_07"]["points"] and not any(p["kind"] == "root" for p in c["guard_07"]["points"]) and fo:
                    C["forced_in_non_root_located"] += 1
print("checkpoint", h.hexdigest())
print(dict(C), "calls with forced partition", len(fp), "root-located calls", len(roots), "sets equal", fp == roots)
print("forced partition count per call:", dict(sorted(per.items())))
# second pass: the root-located calls without a forced partition sum
print("forced subset of root-located:", fp <= roots)
diff = Counter()
with open(R, "rb") as f:
    for raw in f:
        d = json.loads(raw)["data"]
        for kind in ("calls", "other_calls"):
            for c in d[kind]:
                k = (d["unit"], kind, c["index"])
                if k in roots and k not in fp:
                    g = c["guard_07"]
                    diff[(kind, d["rtol"], g["n_windows"] > 0, g["num_window_pieces"] > 0, g["returned"]["n_window"] > 0,
                          tuple(sorted(Counter(p["kind"] for p in g["points"]).items())))] += 1
print("root-located without forced partition, by (kind, rtol, has window, has pieces, returned window leaves, point kinds):")
for k, v in sorted(diff.items(), key=lambda t: -t[1]):
    print("  ", v, k)
