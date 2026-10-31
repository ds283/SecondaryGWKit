"""levin-phase-floor prompt 06c section 5, X1: a descriptive cross-check of the reference on named
calls, after X1 was scored (it is not part of the rule). Copied from 06b's w2_crosscheck.py (kept
beside it and in state/scripts_06b_reference/), changed (prompt 06c): the tree is g; calls are named
"<unit>:<index>" and read from state/x1_captured.json and out/x1_reference.json; the census's own
leaves are printed with the trust check's numbers; S_12 is added beside S_11.

For each named call, reloaded from its capture, at the tree g:
- S_11 and S_12, the uniform-piece sums at 2^11 and 2^12 pieces, against x1_reference.py's S_10;
- the adaptive driver itself, ``adaptive_levin_sincos`` on the reloaded inputs with ``rtol`` 1e-11
  (the call's other arguments unchanged), against S_10;
- the census value against S_10, as a fraction of abserr_g, of |S_10| and of the call's request;
- the census's leaves for the call, from the run's pickle.

Usage: PYTHONPATH=<tree g> python -B x1_crosscheck.py RUN_DIR UNIT:INDEX[,UNIT:INDEX...]
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


def S(x_span, f_, basis, kw, depth):
    vals = []
    for pc in pieces(x_span, depth):
        d = LQ._adaptive_levin_subregion(pc, f_, basis, id_label=LQ._LazyUUID(), chebyshev_order=kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER),
                                         rtol=kw.get("rtol", LQ.DEFAULT_LEVIN_RELTOL), notify_label=None, build_p_sample=False,
                                         diagnostics_path=LQ.DEFAULT_LEVIN_DIAGNOSTICS_PATH, vectorize_cache={})
        vals.append(float(d["value"]))
    return math.fsum(vals)


def main():
    run_dir = Path(sys.argv[1])
    names = [tuple(x.rsplit(":", 1)) for x in sys.argv[2].split(",")]
    cap = {(e["unit"], e["index"]): e for e in json.loads((WIDER / "state" / "x1_captured.json").read_text())}
    ref = {(r["unit"], r["index"]): r for r in json.loads((WIDER / "out" / "x1_reference.json").read_text())["results"]}
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06c: X1's reference cross-checked (descriptive)",
                    [WIDER / "state" / "x1_captured.json", WIDER / "out" / "x1_reference.json"]):
        print(l)
    print(f"# levin_quadrature {LQ.__file__}")
    for unit, idx in names:
        idx = int(idx)
        e, r = cap[(unit, idx)], ref[(unit, idx)]
        with open(Path(e["dir"]) / e["pickle"], "rb") as f:
            f.seek(e["offset"])
            rec = pickle.loads(f.read(e["length"]))
        x_span, f_, theta, kw = tuple(rec["x_span"]), rec["f"], rec["theta"], dict(rec["kw"])
        basis = LQ._Basis_SinCos(theta)
        S11 = S(x_span, f_, basis, kw, 11)
        S12 = S(x_span, f_, basis, kw, 12)
        kw2 = dict(kw)
        kw2["rtol"] = 1e-11
        tight = LQ.adaptive_levin_sincos(x_span, f_, theta=theta, **kw2)
        S10 = float.fromhex(r["ref"])
        vg = float.fromhex(r["value_g"])
        eg = float.fromhex(r["abserr_g"])
        req = max(kw.get("atol", 0.0), kw.get("rtol", 0.0) * abs(vg))
        print(f"\n== {unit} call {idx} ({e['band']}, {e['k_key']}, category {e['category']}) {rec['notify_label'].split('regime')[-1]}; span {x_span}")
        print(f"   S10 {S10!r}; S11 {S11!r}; S12 {S12!r}: |S10-S11|/abserr_g {abs(S10 - S11) / eg:.3g}, |S11-S12|/abserr_g {abs(S11 - S12) / eg:.3g}")
        print(f"   adaptive at rtol 1e-11: value {tight['value']!r} abserr {tight['abserr']:.3g} regions {tight['num_regions']} depth {tight['max_depth']} "
              f"converged {tight['converged']}: |tight-S10|/abserr_g {abs(tight['value'] - S10) / eg:.3g}, |tight-S12|/abserr_g {abs(tight['value'] - S12) / eg:.3g}")
        print(f"   g:   value {vg!r} abserr {eg:.3g}: |g-S10| {abs(vg - S10):.3g} = {abs(vg - S10) / eg:.3g} abserr_g = {abs(vg - S10) / abs(S10):.3g} relative = "
              f"{abs(vg - S10) / req:.3g} of the call's request {req:.3g} (rtol {kw.get('rtol')}, atol {kw.get('atol')}); |g-S12|/abserr_g {abs(vg - S12) / eg:.3g}")
        print(f"   06b: value {e['value_06b']} abserr {e['abserr_06b']}: |06b-S10|/|S10| {abs(float.fromhex(e['value_06b']) - S10) / abs(S10):.3g}")
        d = None
        for line in open(run_dir / "checkpoint.jsonl"):
            if f'"unit": "{unit}"' in line:
                d = json.loads(line)["data"]
                break
        cg = next(c for c in d["calls"] if c["index"] == idx)
        print(f"   g's call record: budget {cg['budget']}; E_share { {k: v['repr'] for k, v in cg['E_share'].items()} }; E_unwidened {cg['E_unwidened']['repr']}; leaves {cg['leaves']}")
        with open(run_dir / d["pickle"], "rb") as fh:
            fh.seek(d["pickle_offset"])
            prec = pickle.loads(fh.read(d["pickle_length"]))
        arr = next(a for a in prec["regions"] if a["index"] == idx)
        print("   g's leaves (start, end, depth, type 0 Levin/1 CC, residual, floor, total_err, delta1, delta2, r, trusted, declared_resolution):")
        for i in range(len(arr["start"])):
            print(f"      {arr['start'][i]:.6g} {arr['end'][i]:.6g} {int(arr['depth'][i]):2d} {int(arr['type'][i])} {arr['abserr'][i]:.3g} {arr['phase_err'][i]:.3g} "
                  f"{arr['total_err'][i]:.3g} {arr['delta1'][i]:.3g} {arr['delta2'][i]:.3g} {arr['convergence_ratio'][i]:.3g} {int(arr['trusted'][i])} "
                  f"{arr['declared_resolution'][i]:.3g}")


if __name__ == "__main__":
    main()
