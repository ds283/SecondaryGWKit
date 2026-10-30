"""Attribute the Levin solve round-off on leaves near x_r: the mp scheme at the exact nodes vs at
the double (rounded) nodes. If mp(double nodes) ~ the double value, node rounding is the cause.
The last leaf is a control far from x_r, whose round-off is the endpoint phase rounding."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, mpmath as mp, common as C
lq = C.lq
print(json.dumps(C.header(__file__)))
for ci, a, b in [(1, 0.722412109375, 0.72265625), (2, 0.725341796875, 0.7255859375), (1, 0.72265625, 0.7236328125), (2, 0.72265625, 0.7236328125), (1, 0.7236328125, 0.724609375), (1, 0.8359375, 0.8369140625)]:
    case = C.CASES[ci]
    d = C.solve_leaf(case, (a, b))
    p_use = [np.isfinite(r) and r > C.RTOL for r in d["p_ratios"]]
    grid, _ = lq.chebyshev_matrices((a, b), C.ORDER)
    with mp.workdps(C.MP_DPS):
        ex = C.mp_levin(case, a, b, p_use)
        dn = C.mp_levin(case, a, b, p_use, nodes=[mp.mpf(g) for g in grid])
        print(f"{case['name']} [{a}, {b}] floor {d['phase_err']:.2e} | double-mp(exact nodes) {float(mp.mpf(d['value']) - ex):+.3e} | "
              f"node part mp(double nodes)-mp(exact) {float(dn - ex):+.3e} | remainder double-mp(double nodes) {float(mp.mpf(d['value']) - dn):+.3e}")
