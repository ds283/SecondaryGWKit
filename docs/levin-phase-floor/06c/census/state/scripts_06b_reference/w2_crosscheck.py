"""levin-phase-floor prompt 06b section 5, W2: a descriptive cross-check of the reference, after W2
was scored (it is not part of the rule). New in 06b.

For each listed call of state/w2_selection.json (1-based positions), reloaded as w2_reference.py
reloads it, at the tree b:
- S_11, the same uniform-piece sum at 2^11 pieces, against w2_reference.py's S_10;
- the adaptive driver itself, ``adaptive_levin_sincos`` on the reloaded inputs with ``rtol`` 1e-11
  (the call's other arguments unchanged), against S_10;
- the call's b value against S_10, as a fraction of |S_10| and of the call's own request
  max(atol, rtol |value|);
- b's regions for the call (depth, type, abserr, phase_err), from the census pickle.

Usage: PYTHONPATH=<tree b> python -B w2_crosscheck.py REF_JSON POS[,POS...]
"""

import json
import math
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header  # noqa: E402
from w2_reference import pieces  # noqa: E402

import AdaptiveLevin.levin_quadrature as LQ  # noqa: E402


def main():
    ref = json.loads(Path(sys.argv[1]).read_text())["results"]
    pos = [int(x) for x in sys.argv[2].split(",")]
    sel = json.loads((WIDER / "state" / "w2_selection.json").read_text())
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06b: W2's reference cross-checked (descriptive)", [Path(sys.argv[1])]):
        print(l)
    print(f"# levin_quadrature {LQ.__file__}")
    ck = {}
    for p in pos:
        e = sel[p - 1]
        r = ref[p - 1]
        assert (r["unit"], r["index"]) == (e["unit"], e["index"])
        with open(Path(e["dir"]) / e["pickle"], "rb") as f:
            f.seek(e["offset"])
            rec = pickle.loads(f.read(e["length"]))
        x_span, f_, theta, kw = tuple(rec["x_span"]), rec["f"], rec["theta"], dict(rec["kw"])
        basis = LQ._Basis_SinCos(theta)
        vals = []
        for pc in pieces(x_span, 11):
            d = LQ._adaptive_levin_subregion(pc, f_, basis, id_label=LQ._LazyUUID(), chebyshev_order=kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER),
                                             rtol=kw.get("rtol", LQ.DEFAULT_LEVIN_RELTOL), notify_label=None, build_p_sample=False,
                                             diagnostics_path=LQ.DEFAULT_LEVIN_DIAGNOSTICS_PATH, vectorize_cache={})
            vals.append(float(d["value"]))
        S11 = math.fsum(vals)
        kw2 = dict(kw)
        kw2["rtol"] = 1e-11
        tight = LQ.adaptive_levin_sincos(x_span, f_, theta=theta, **kw2)
        S10 = float.fromhex(r["ref"])
        vb = float.fromhex(r["value_b"])
        eb = float.fromhex(r["abserr_b"])
        va = float.fromhex(r["value_06a"])
        ea = float.fromhex(r["abserr_06a"])
        req = max(kw.get("atol", 0.0), kw.get("rtol", 0.0) * abs(vb))
        print(f"\n== #{p} {e['unit']} call {e['index']} ({e['band']}, {e['k_key']}) {rec['notify_label'].split('regime')[-1]}")
        print(f"   S10 {S10!r}; S11 {S11!r}: |S10-S11|/abserr_b {abs(S10 - S11) / eb:.3g}")
        print(f"   adaptive at rtol 1e-11: value {tight['value']!r} abserr {tight['abserr']:.3g} regions {tight['num_regions']} converged {tight['converged']}: "
              f"|tight-S10|/abserr_b {abs(tight['value'] - S10) / eb:.3g}")
        print(f"   b:   value {vb!r} abserr {eb:.3g}: |b-S10| {abs(vb - S10):.3g} = {abs(vb - S10) / eb:.3g} abserr_b = {abs(vb - S10) / abs(S10):.3g} relative = "
              f"{abs(vb - S10) / req:.3g} of the call's request max(atol, rtol|value|) {req:.3g} (rtol {kw.get('rtol')}, atol {kw.get('atol')})")
        print(f"   06a: value {va!r} abserr {ea:.3g}: |06a-S10| {abs(va - S10):.3g} = {abs(va - S10) / ea:.3g} abserr_06a = {abs(va - S10) / abs(S10):.3g} relative")
        ub = e["unit"]
        if ub not in ck:
            for line in open(Path(e["dir"]) / "checkpoint.jsonl"):
                if f'"unit": "{ub}"' in line:
                    ck[ub] = json.loads(line)["data"]
                    break
        d = ck[ub]
        with open(Path(e["dir"]) / d["pickle"], "rb") as fh:
            fh.seek(d["pickle_offset"])
            prec = pickle.loads(fh.read(d["pickle_length"]))
        arr = next(a for a in prec["regions"] if a["index"] == e["index"])
        print("   b's regions (depth, type 0 Levin/1 CC, abserr, phase_err):")
        for i in range(len(arr["start"])):
            print(f"      {int(arr['depth'][i]):2d} {int(arr['type'][i])} {arr['abserr'][i]:.3g} {arr['phase_err'][i]:.3g}")


if __name__ == "__main__":
    main()
