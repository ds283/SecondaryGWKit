"""levin-phase-floor prompt 06k, route 3: the smallest eta that would accept each call's left leaves on its returned
partition (eta_min, recorded per call by the variant), and the same with any untrusted leaf at depth_max added
(the reading the user ruled against, Q2). New in 06k. Read only: state/extract/route3.pkl.
Usage: python -B eta_route3.py"""
import pickle, sys
from pathlib import Path
d = pickle.load(open(Path(__file__).resolve().parent.parent / "state" / "extract" / "route3.pkl", "rb"))
rows = []
for u in d["units"].values():
    for c in u["calls"]:
        r = c["route"]
        if r and r.get("n_left_returned"):
            rows.append((r["eta_min"], r["eta_min_with_depth_max"], r["n_untrusted_at_depth_max_not_left"], c["converged"]))
em = sorted(x[0] for x in rows)
print(f"route 3 calls returning left leaves: {len(rows)}; eta_min p10 {em[len(em)//10]:.3g} median {em[len(em)//2]:.3g} "
      f"p90 {em[int(0.9*len(em))]:.3g} p99 {em[int(0.99*len(em))]:.3g} max {em[-1]:.3g}")
for e in (1e-4, 1e-5, 1e-6, 1e-7):
    print(f"  calls with eta_min > {e:g} (whose left set would not be accepted at that eta on this partition): {sum(1 for x in em if x > e)}")
print(f"  calls where the depth_max reading differs: {sum(1 for a, b, n, _ in rows if a != b)}; calls with an untrusted leaf at depth_max not left: {sum(1 for a, b, n, _ in rows if n)}")
