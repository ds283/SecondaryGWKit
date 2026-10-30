"""Sanity checks of the mp same-scheme reproduction against 06d's double code."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, mpmath as mp
import common as C
lq = C.lq
print("lq from", lq.__file__)
# exact D vs double D
x, D = lq._chebyshev_base(16)
t, Dm = C.mp_base(16)
print("max|t_double - t_exact|", max(abs(float(t[i]) - x[i]) for i in range(16)))
print("max|D_double - D_exact|", max(abs(float(Dm[i, j]) - D[i, j]) for i in range(16) for j in range(16)), "max|D|", np.abs(D).max())
w = lq._cc_weights_base(31); wm = C.mp_cc_weights_base(31)
print("max|w_double - w_exact|", max(abs(float(wm[i]) - w[i]) for i in range(31)))
case = C.CASES[1]
for span in [(0.0, 1/64), (0.703125, 0.71875), (0.71875, 0.734375), (0.7236, 0.72362)]:
    t0 = time.perf_counter()
    r = C.analyse_leaf(case, span)
    dt = time.perf_counter() - t0
    print(span, r["kind"], "value", r["value"], "roundoff", r["roundoff"], "trunc", r.get("truncation"),
          "floor", r["floor_code"], "recon", r["floor_recomputed"], "sumterms", r["floor_sum_terms"], "terms", r["terms"], f"{dt:.2f}s")
# whole-interval closed form vs reference.closed_R
sys.path.insert(0, str(C.SCRIPTS_07))
import reference as R
for case in C.CASES:
    with mp.workdps(60):
        a = C.closed_leaf(case, 0.0, 1.0)
        parts = sum(C.closed_leaf(case, i/8, (i+1)/8) for i in range(8))
        b = R.closed_R(case["c"], case["x_s"], case["omega"])
        print(case["name"], mp.nstr(a, 25), "parts-whole", mp.nstr(parts - a, 5), "vs ref", mp.nstr(a - b, 5))
