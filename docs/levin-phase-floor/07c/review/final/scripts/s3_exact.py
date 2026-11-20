"""Orchestrator's own S3 reader (07c Part C): every QuadSourceIntegral row, |t32-t31| <= e31+e32 in exact Fractions.
Reuses the stop-1 reader's read() (07b orchestrator's, copied) to key rows. usage: s3_exact.py <R31 primary> <R32 primary>"""
import sys, importlib.util
from fractions import Fraction as F
spec = importlib.util.spec_from_file_location("q", sys.argv[3]); q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
A, B = q.read(sys.argv[1]), q.read(sys.argv[2])
assert set(A) == set(B)
n = viol = finite = 0; worst = (F(0), None)
for k in A:
    t1, e1, t2, e2 = A[k][2], A[k][3], B[k][2], B[k][3]
    if any(x is None for x in (t1, e1, t2, e2)): continue
    import math
    if not all(math.isfinite(x) for x in (t1, e1, t2, e2)): continue
    finite += 1
    d = abs(F(t2) - F(t1)); bound = F(e1) + F(e2)
    if d > bound: viol += 1
    if bound and d / bound > worst[0]: worst = (d / bound, k)
    if d: n += 1
print("rows", len(A), "finite in both", finite, "total moved", n, "violations", viol)
k = worst[1]; print("largest ratio %.6g at k=%r q=%r r=%r z=%r rtol=%s" % (float(worst[0]), *(float.fromhex(x) for x in k[:4]), k[5]))
for k in A:
    if float.fromhex(k[0]) == 1e5 and float.fromhex(k[1]) == 1e6 and float.fromhex(k[2]) == 1e6 and abs(float.fromhex(k[3]) - 144627.7067) < 1e-3:
        d = abs(F(B[k][2]) - F(A[k][2])); print("named row rtol", k[5], "conv", A[k][4], "->", B[k][4], "ratio %.6g" % float(d / (F(A[k][3]) + F(B[k][3]))))
