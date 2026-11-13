"""Summary of R30's replay (out/replay_calls.csv, replay07a_r30.py check): located points by record kind and label,
point kind and found_as; window pieces; guarded regions. levin-phase-floor 07a, rehearsal agent, R30. New.

    ./venv/bin/python -I -B replay_summary_r30.py <replay_calls.csv>
"""
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


p = sys.argv[1]
print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
print(f"# input {p} sha256 {sha256(p)}")
rows = list(csv.DictReader(open(p)))
print(f"lines {len(rows)}; records {len({(r['k'], r['q'], r['r'], r['z_response'], r['record']) for r in rows})}; reproduce {sum(r['reproduces'] == 'True' for r in rows)} lines")
c = Counter()
recs = {}
for r in rows:
    key = (r["k"], r["q"], r["r"], r["z_response"], r["record"])
    recs[key] = r
    if r["point"]:
        c[(r["kind"], r["label"] if r["kind"] == "group" else r["record"].split(".", 1)[1], r["point_kind"], r["found_as"])] += 1
print("located points by (record kind, label/block, point kind, found_as):")
for k, n in sorted(c.items()):
    print(f"  {k}: {n}")
g = Counter()
for key, r in recs.items():
    sp = int(r["num_stationary_points"] or 0)
    g[(r["kind"], "located" if sp else "none", "guarded" if int(r["num_guarded_regions"] or 0) else "unguarded",
       "guard at depth_max" if int(r["num_guard_at_depth_max"] or 0) else "-")] += 1
print("records by (kind, located, guarded, guard at depth_max):")
for k, n in sorted(g.items()):
    print(f"  {k}: {n}")
print(f"records with more than 64 points: {sum(1 for r in recs.values() if int(r['num_stationary_points'] or 0) > 64)}")
print(f"window pieces summed: {sum(int(r['num_window_pieces'] or 0) for r in recs.values())}; points summed: {sum(int(r['num_stationary_points'] or 0) for r in recs.values())}")
