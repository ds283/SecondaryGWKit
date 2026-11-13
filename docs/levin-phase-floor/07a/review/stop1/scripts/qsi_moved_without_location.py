"""Orchestrator stop-1 reader, 07a Part C: of R29->R30 moved QSI rows, those with no located point and
no guarded region in either store, by which columns moved. Imports qsi_compare.py beside it.
usage: python -I -B qsi_moved_without_location.py <R29 primary> <R30 primary>"""
import sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import qsi_compare as q
print("script", hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "qsi_compare", q.sha(q.__file__))
A, B = q.read(sys.argv[1]), q.read(sys.argv[2])
from collections import Counter
cnt = Counter(); cols = Counter(); worst = (0, None)
for k in A:
    fa, fb = A[k][0], B[k][0]
    if fa == fb:
        continue
    loc = any(r.get("stationary_points", 0) or r.get("guarded_regions", 0)
              for R in (A, B) for _, r in q.levin_records(R[k][1]))
    cnt["moved, located/guarded in either" if loc else "moved, nothing located or guarded in either"] += 1
    if not loc:
        for p in set(fa) | set(fb):
            if fa.get(p) != fb.get(p) and p[0] == "col":
                cols[p[1]] += 1
        if A[k][3]:
            r = abs(B[k][2] - A[k][2]) / A[k][3]
            if r > worst[0]:
                worst = (r, k)
print(dict(cnt)); print("columns moved among unlocated rows:", dict(cols))
print("largest |dtotal|/abserr29 among unlocated rows: %.6g" % worst[0], worst[1] and (float.fromhex(worst[1][0]), float.fromhex(worst[1][3])))
