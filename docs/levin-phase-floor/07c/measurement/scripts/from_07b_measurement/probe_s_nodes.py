"""levin-phase-floor prompt 06j, Part S, **descriptive, added after S1-S4 were scored** (out/cmd/42): is a slow leaf's
delta1 the node-displacement error 06e's node term models? Not a prediction; it scores nothing. New in 06j.

For every sampled leaf (out/sample/s_sample.json), the leaf rule's seven solves on its span (the tree's own
``_adaptive_levin_subregion``, as probe_s.py) are taken with each solve's ``node_error_signed`` E (06e's signed
first-order error of the computed nodes: the rule at the computed nodes minus the rule at the exact nodes is E to
first order; every sampled leaf is a Clenshaw-Curtis leaf, out/cmd/42). Reported per leaf:
- delta1, delta2, r as the rule forms them (they must equal the census's, as in probe_s.py);
- the same with every solve's value corrected by -E: delta1_c = |(V_P - E_P) - ((V_L - E_L) + (V_R - E_R))|,
  delta2_c likewise from the quarters, r_c = delta2_c / delta1_c;
- |E_P - (E_L + E_R)| against delta1, and E_P, phase_err, and the floor's node part.
Writes out/partS_nodes.json and out/partS_nodes.csv.

Usage: PYTHONPATH=<tree>:<this directory> python -B probe_s_nodes.py
"""

import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402
from probe_s import Probe, ratio_of, fx  # noqa: E402


def q(v, ps=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)):
    v = sorted(x for x in v if x is not None and math.isfinite(x))
    if not v:
        return "none"
    return ", ".join(f"p{int(p*100)} {v[min(len(v)-1, int(p*(len(v)-1)))]:.4g}" for p in ps)


def main():
    import ray.cloudpickle as cp

    import AdaptiveLevin.levin_quadrature as LQ

    ssam = json.loads((RC.ROOT / "out" / "sample" / "s_sample.json").read_text())["calls"]
    inputs = [RC.ROOT / "state" / "inputs" / f"{s['sid']}.pkl" for s in ssam]
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06j: Part S, descriptive: delta1 against the node errors E",
                 [RC.ROOT / "out" / "sample" / "s_sample.json", Path(LQ.__file__)] + inputs)
    for l in hdr:
        print(l)
    rows = []
    for s, pkl in zip(ssam, inputs):
        rec = cp.loads(pkl.read_bytes())
        pr = Probe(LQ, rec)
        for lv in s["leaves"]:
            a, b = lv["start"], lv["end"]
            c = (b + a) / 2.0
            cL, cR = (c + a) / 2.0, (b + c) / 2.0
            S = {k: pr.solve(x0, x1) for k, (x0, x1) in {"P": (a, b), "L": (a, c), "R": (c, b), "LL": (a, cL), "LR": (cL, c),
                                                          "RL": (c, cR), "RR": (cR, b)}.items()}
            V = {k: d["value"] for k, d in S.items()}
            E = {k: d.get("node_error_signed") for k, d in S.items()}
            if any(e is None for e in E.values()):
                rows.append({"sid": s["sid"], "start": a, "end": b, "note": "a solve took the Levin branch (no E)"})
                continue
            h = V["L"] + V["R"]
            qq = (V["LL"] + V["LR"]) + (V["RL"] + V["RR"])
            d1, d2 = abs(V["P"] - h), abs(h - qq)
            Vc = {k: V[k] - E[k] for k in V}
            hc = Vc["L"] + Vc["R"]
            qc = (Vc["LL"] + Vc["LR"]) + (Vc["RL"] + Vc["RR"])
            d1c, d2c = abs(Vc["P"] - hc), abs(hc - qc)
            eh = abs(E["P"] - (E["L"] + E["R"]))
            rows.append({"sid": s["sid"], "group": s["group"], "rtol": s["rtol"], "start": a, "end": b,
                         "delta1": d1, "delta2": d2, "r": ratio_of(d1, d2),
                         "equals_census": fx(d1) == lv["hex"]["delta1"] and fx(d2) == lv["hex"]["delta2"],
                         "delta1_corrected": d1c, "delta2_corrected": d2c, "r_corrected": ratio_of(d1c, d2c),
                         "E_P": E["P"], "abs_EP_minus_EL_ER": eh, "phi": S["P"]["phase_err"],
                         "phi_less_node": S["P"].get("phase_err_less_node"), "node_bound": S["P"].get("node_error_bound"),
                         "delta1_corr_over_delta1": d1c / d1 if d1 else math.nan,
                         "E_diff_over_delta1": eh / d1 if d1 else math.nan,
                         "delta1_corr_over_phi_less_node": d1c / S["P"]["phase_err_less_node"] if S["P"].get("phase_err_less_node") else math.nan})
    good = [r for r in rows if "delta1" in r]
    summ = {"header": hdr, "n": len(rows), "n_cc": len(good), "n_equal_census": sum(1 for r in good if r["equals_census"]),
            "delta1_corr_over_delta1": q([r["delta1_corr_over_delta1"] for r in good]),
            "E_diff_over_delta1": q([r["E_diff_over_delta1"] for r in good]),
            "r": q([r["r"] for r in good]), "r_corrected": q([r["r_corrected"] for r in good]),
            "delta1_over_phi": q([r["delta1"] / r["phi"] for r in good]),
            "delta1_corr_over_phi": q([r["delta1_corrected"] / r["phi"] for r in good]),
            "delta1_corr_over_phi_less_node": q([r["delta1_corr_over_phi_less_node"] for r in good]),
            "n_corrected_trusted_b": sum(1 for r in good if r["r_corrected"] < 0.25 or (r["r_corrected"] < 1 and r["delta1_corrected"] <= r["phi"])),
            "n_r_corrected_in_04_06": sum(1 for r in good if 0.4 <= r["r_corrected"] <= 0.6)}
    for k, v in summ.items():
        if k != "header":
            print(f"{k}: {v}")
    (RC.ROOT / "out" / "partS_nodes.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, default=str))
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(RC.ROOT / "out" / "partS_nodes.csv", "w", newline="") as f:
        f.write("".join(f"{l}\n" for l in hdr))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote out/partS_nodes.json sha256 {RC.sha256(RC.ROOT / 'out' / 'partS_nodes.json')}")


if __name__ == "__main__":
    main()
