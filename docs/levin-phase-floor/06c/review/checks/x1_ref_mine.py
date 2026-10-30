"""Orchestrator's own recomputation of X1's reference on selected calls (06c Part B review, check 9).

Modelled on 06b's review/w2_ref_mine.py. Run with PYTHONPATH=<tree g>. Read-only: loads the census
agent's captured input records (checking each blob's SHA-256), writes nothing.
Reference (prompt 06c section 5, X1 = 06b's W2): Levin forced to 2^9 and 2^10 equal pieces, each
piece by _adaptive_levin_subregion, with each piece's floor (phase_err) summed. Agreement rule:
|S9 - S10| <= abserr_g / 100; ref = S10; holds iff |value_g - ref| <= abserr_g.
"""

import hashlib
import json
import math
import pickle
import sys

import AdaptiveLevin.levin_quadrature as LQ

C = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06c/census/"
SEL = C + "state/x1_captured.json"
WANT = {("g-1e-7-0-14127", 3): (9, 10, 11, 12), ("g-1e-7-3-38436", 5): (9, 10, 11), ("g-1e-7-3-39140", 2): (9, 10, 11)}


def equal_pieces(span, depth):
    out = [tuple(span)]
    for _ in range(depth):
        out = [q for (a, b) in out for q in ((a, (b + a) / 2.0), ((b + a) / 2.0, b))]
    return out


print("levin_quadrature from", LQ.__file__)
sel = json.load(open(SEL))
for (unit, idx), depths in WANT.items():
    e = next(x for x in sel if x["unit"] == unit and x["index"] == idx)
    with open(C + e["dir"] + "/" + e["pickle"], "rb") as f:
        f.seek(e["offset"])
        blob = f.read(e["length"])
    assert hashlib.sha256(blob).hexdigest() == e["sha256"]
    rec = pickle.loads(blob)
    assert rec["unit"] == unit and rec["index"] == idx
    span = tuple(rec["x_span"])
    kw = dict(rec["kw"])
    rr = LQ.adaptive_levin_sincos(span, rec["f"], theta=rec["theta"], **kw)
    vg, eg = float.fromhex(e["census_value"]), float.fromhex(e["census_abserr"])
    print(f"\n== {unit} call {idx}  span {span[0].hex()} .. {span[1].hex()}  rtol {kw['rtol']}")
    print("  census value", e["census_value"], "abserr", e["census_abserr"])
    print("  re-run value", rr["value"].hex(), "abserr", rr["abserr"].hex(),
          "equal to census:", (rr["value"].hex(), rr["abserr"].hex()) == (e["census_value"], e["census_abserr"]))
    basis = LQ._Basis_SinCos(rec["theta"])
    order = kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER)
    S = {}
    cache = {}
    for D in depths:
        vals, floors = [], []
        for p in equal_pieces(span, D):
            d = LQ._adaptive_levin_subregion(p, rec["f"], basis, id_label=LQ._LazyUUID(),
                                             chebyshev_order=order, rtol=kw["rtol"], vectorize_cache=cache)
            vals.append(float(d["value"]))
            floors.append(float(d.get("phase_err") or 0.0))
        S[D] = math.fsum(vals)
        print(f"  S{D} = {S[D].hex()}  ({S[D]!r})  floor sum {math.fsum(floors):.3e}  max {max(floors):.3e}")
    agree = abs(S[9] - S[10]) <= eg / 100
    print(f"  |S9-S10| / abserr_g = {abs(S[9]-S[10])/eg:.4g}  -> agree (<= 1/100): {agree}")
    for D in depths[2:]:
        print(f"  |S10-S{D}| / abserr_g = {abs(S[10]-S[D])/eg:.4g}")
    print(f"  |value_g - S10| / abserr_g = {abs(vg-S[10])/eg:.4g}  -> holds: {abs(vg-S[10]) <= eg}"
          + ("" if agree else "  (dropped from the rule: S9 and S10 disagree)"))
    vb, eb = float.fromhex(e["value_06b"]), float.fromhex(e["abserr_06b"])
    print(f"  06b: value {e['value_06b']} abserr {e['abserr_06b']}; |value_06b - S10| = {abs(vb-S[10]):.4g} "
          f"(/abserr_06b {abs(vb-S[10])/eb if math.isfinite(eb) else 'n/a'})")
    if unit == "g-1e-7-0-14127":
        kw2 = dict(kw)
        kw2["rtol"] = 1e-11
        r2 = LQ.adaptive_levin_sincos(span, rec["f"], theta=rec["theta"], **kw2)
        print(f"  driver at rtol 1e-11: value {r2['value'].hex()} abserr {r2['abserr'].hex()} regions {r2.get('num_regions')}"
              f"; |v - S10|/abserr_g {abs(r2['value']-S[10])/eg:.4g}; v - value_g = {r2['value']-vg:.4g}")
