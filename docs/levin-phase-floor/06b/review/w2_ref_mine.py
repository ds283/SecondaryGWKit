"""Orchestrator's own recomputation of W2's reference on two selected calls (06b check 9).
Run with PYTHONPATH=<tree b>. Read-only: loads the census's saved input records, writes nothing."""

import hashlib
import json
import math
import pickle
import sys

import AdaptiveLevin.levin_quadrature as LQ

SEL = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06b/census/state/w2_selection.json"
WANT = [("b-1e-7-3-34455", 1), ("b-1e-7-2-39035", 2)]


def equal_pieces(span, depth):
    s, e = span
    out = [(s, e)]
    for _ in range(depth):
        out = [q for (a, b) in out for q in ((a, (b + a) / 2.0), ((b + a) / 2.0, b))]
    return out


print("levin_quadrature from", LQ.__file__)
sel = json.load(open(SEL))
for unit, idx in WANT:
    e = next(x for x in sel if x["unit"] == unit and x["index"] == idx)
    with open(f'{e["dir"]}/{e["pickle"]}', "rb") as f:
        f.seek(e["offset"])
        blob = f.read(e["length"])
    assert hashlib.sha256(blob).hexdigest() == e["sha256"]
    rec = pickle.loads(blob)
    assert rec["unit"] == unit and rec["index"] == idx
    span = tuple(rec["x_span"])
    kw = dict(rec["kw"])
    rr = LQ.adaptive_levin_sincos(span, rec["f"], theta=rec["theta"], **kw)
    print(f"\n== {unit} call {idx}  span {span[0].hex()} .. {span[1].hex()}")
    print("  re-run value", rr["value"].hex(), "abserr", rr["abserr"].hex(),
          "equal to census:", (rr["value"].hex(), rr["abserr"].hex()) == (e["value_b"], e["abserr_b"]))
    basis = LQ._Basis_SinCos(rec["theta"])
    order = kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER)
    S = {}
    for D in (9, 10, 11):
        vals, floors = [], []
        cache = {}
        for p in equal_pieces(span, D):
            d = LQ._adaptive_levin_subregion(p, rec["f"], basis, id_label=LQ._LazyUUID(),
                                             chebyshev_order=order, rtol=kw["rtol"],
                                             vectorize_cache=cache)
            vals.append(float(d["value"]))
            floors.append(float(d.get("phase_err") or 0.0))
        S[D] = math.fsum(vals)
        print(f"  S{D} = {S[D].hex()}  ({S[D]!r})  floor sum {math.fsum(floors):.3e}")
    vb, eb = float.fromhex(e["value_b"]), float.fromhex(e["abserr_b"])
    print(f"  |S9-S10| / abserr_b = {abs(S[9]-S[10])/eb:.4g};  |S10-S11| / abserr_b = {abs(S[10]-S[11])/eb:.4g}")
    print(f"  |value_b - S10| / abserr_b = {abs(vb-S[10])/eb:.4g}  -> holds: {abs(vb-S[10]) <= eb}")
    print(f"  |value_b - S11| / abserr_b = {abs(vb-S[11])/eb:.4g}")
    va, ea = float.fromhex(e["value_06a"]), float.fromhex(e["abserr_06a"])
    print(f"  06a: |value_06a - S10| / abserr_06a = {abs(va-S[10])/ea:.4g}")
