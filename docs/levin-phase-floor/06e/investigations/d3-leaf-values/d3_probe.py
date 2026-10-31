"""D3 probe: replay a captured call at this tree, recording every sub-solve's value, then for each
returned leaf print V_P, the halves and quarters sums, delta1, delta2, r and the declared error,
and compare the call's value (and the value with each checked leaf replaced by its quarters sum)
with the reference."""

import sys
from fractions import Fraction as F
from pathlib import Path

import ray.cloudpickle as cp

import AdaptiveLevin.levin_quadrature as LQ

pkl, ref_hex = sys.argv[1], sys.argv[2]
rec = cp.loads(Path(pkl).read_bytes())
ref = float.fromhex(ref_hex)
scale = F(float.fromhex(rec["scale"])) if rec.get("scale") else F(1)
print("scale", rec.get("scale"))

solves = {}
_orig = LQ._adaptive_levin_subregion


def _wrapped(span, *a, **k):
    d = _orig(span, *a, **k)
    solves[(float(span[0]), float(span[1]))] = d["value"]
    return d


LQ._adaptive_levin_subregion = _wrapped
rr = LQ.adaptive_levin_sincos(
    tuple(rec["x_span"]), rec["f"], theta=rec["theta"], **rec["kw"]
)
v, e = rr["value"], rr["abserr"]
print(
    f"value {v.hex()} abserr {e:.6e} regions {rr['num_regions']} converged {rr['converged']}"
)
print(
    f"|v-ref|/abserr {float(abs(scale * F(v) - F(ref)) / (abs(scale) * F(e))):.6f}  v-ref {v - ref:+.6e}"
)

corrected = F(v)
for R in rr["regions"]:
    a, b = R.start, R.end
    c = (a + b) / 2.0
    cL, cR = (c + a) / 2.0, (b + c) / 2.0
    VP = solves.get((a, b))
    H = [solves.get(s) for s in ((a, c), (c, b))]
    Q = [solves.get(s) for s in ((a, cL), (cL, c), (c, cR), (cR, b))]
    line = (
        f"  [{a:.6f},{b:.6f}] d{R.depth} {R.type:6s} abserr {R.abserr:.3e} "
        f"trusted {R.trusted} r {R.convergence_ratio} declared {R.declared_resolution}"
    )
    print(line)
    if VP is not None and None not in H and None not in Q:
        h, q = H[0] + H[1], (Q[0] + Q[1]) + (Q[2] + Q[3])
        print(
            f"      VP-H {VP - h:+.6e}  H-Q {h - q:+.6e}  VP-Q {VP - q:+.6e}"
            f"  delta1 {R.delta1}  delta2 {R.delta2}"
        )
        if R.trusted:
            corrected += F(q) - F(VP)
print(
    f"value with checked leaves -> quarters: |v'-ref|/abserr {float(abs(scale * corrected - F(ref)) / (abs(scale) * F(e))):.6f}"
)
