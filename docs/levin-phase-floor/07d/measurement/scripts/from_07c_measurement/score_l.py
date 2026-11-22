"""levin-phase-floor prompt 06k, Part L: the probes tabulated, and L1 and L2 scored per route by the rules as
written (state/predictions_and_rules_20261104T0915Z.md section 1). New in 06k. Read only.

Reads out/partL/<sid>.json (probe_l.py) for every call of out/sample/selection.json. Writes
out/partL_leaves.csv (one row per leaf evaluation: its population, call, span, depth, branch, the
residual, delta1, delta2, r, phi_P, phi_L, phi_R, nu1, delta1/phi_P, delta1/nu1, E of the seven solves,
delta1c, delta2c, r_c, delta1c/phi_P, D, the request, D/(eta request) and each route's decision) and
out/partL_summary.json, and prints the tables.
- **L1** (per route): of the 06h carrying leaves (L-b; 22 evaluations, one per W call, on 17 distinct
  leaves: f05-f09 are f00-f04's integrands at 1e-8), the route trusts or leaves none. Scored on the 17
  distinct leaves (a leaf counts if the route trusts or leaves it at either of its calls) and reported
  on all 22.
- **L2** (per route): the route trusts or leaves at least 80 % of 06j's 142 sampled slow leaves (L-a).
- **L-c**: on every returned leaf of the 24 S calls that 06g trusted, whether the route's decision
  differs from "trusted" (route 2 both as built and literally).
Usage: python -B score_l.py
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

ROUTES = ("g", "route1", "route2", "route2_literal", "route3")
K1_LIMIT = 5.8


def main():
    sel = json.loads((RC.ROOT / "out" / "sample" / "selection.json").read_text())
    files = [RC.ROOT / "out" / "partL" / f"{e['sid']}.json" for e in sel]
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06k: Part L tabulated, L1 and L2 scored", files)
    for l in hdr:
        print(l)
    rows = []
    for e, f in zip(sel, files):
        d = json.loads(f.read_text())
        for L in d["leaves"]:
            E = L["E"]
            rows.append({
                "population": L["population"], "sid": d["sid"], "unit": d["unit"], "index": d["index"], "rtol": d["rtol"],
                "group": e.get("g_record", {}).get("group"), "start": L["start_hex"], "end": L["end_hex"],
                "depth": L["depth"], "P_type": L["types"]["P"], "residual": L["residual"], "delta1": L["delta1"],
                "delta2": L["delta2"], "r": L["r"], "phi_P": L["phi_P"], "phi_L": L["phi_L"], "phi_R": L["phi_R"],
                "nu1": L["nu1"], "d1_over_phiP": L["d1_over_phiP"], "d1_over_nu1": L["d1_over_nu1"],
                **{f"E_{k}": E[k] for k in ("P", "L", "R", "LL", "LR", "RL", "RR")},
                "delta1c": L["delta1c"], "delta2c": L["delta2c"], "r_c": L["r_c"], "d1c_over_phiP": L["d1c_over_phiP"],
                "D": L["D"], "request": L["request"], "D_over_eta_request": L["D_over_eta_request"],
                **{f"dec_{k}": L["decision"][k] for k in ROUTES},
                "reproduces_census": L["reproduces_census"],
                "h06_r": (L.get("h06_record") or {}).get("convergence_ratio"), "h06_d1_over_phi": L.get("h06_d1_over_phi"),
            })
    out_csv = RC.ROOT / "out" / "partL_leaves.csv"
    with open(out_csv, "w", newline="") as fo:
        for l in hdr:
            fo.write(l + "\n")
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    pop = {p: [r for r in rows if r["population"] == p] for p in ("L-a", "L-b", "L-c")}
    acc = lambda r, k: r[f"dec_{k}"] in ("trusted", "left")  # noqa: E731
    summ = {"n": {p: len(v) for p, v in pop.items()},
            "reproduces_census": {p: sum(1 for r in v if r["reproduces_census"]) for p, v in pop.items()}}
    # L-b distinct leaves: W-5's 1e-8 calls f05-f09 share f00-f04's integrands and leaves
    distinct = {}
    for r in pop["L-b"]:
        key = ("-".join(r["unit"].rsplit("-", 2)[-2:]), r["index"], r["start"], r["end"])  # the row, the call, the span
        distinct.setdefault(key, []).append(r)
    summ["L-b_distinct"] = len(distinct)
    verdict = {}
    for k in ROUTES:
        lb_hit = [r["sid"] for r in pop["L-b"] if acc(r, k)]
        lb_distinct_hit = sum(1 for v in distinct.values() if any(acc(r, k) for r in v))
        la_hit = sum(1 for r in pop["L-a"] if acc(r, k))
        lc_changed = [(r["sid"], r["start"], r[f"dec_{k}"]) for r in pop["L-c"] if r[f"dec_{k}"] != "trusted"]
        la_fire = [r for r in pop["L-a"] if acc(r, k) and r["dec_g"] == "untrusted"]
        verdict[k] = {
            "L1_distinct_trusted_or_left": lb_distinct_hit, "L1_evaluations_trusted_or_left": lb_hit,
            "L1": "holds" if lb_distinct_hit == 0 else "fails",
            "L2_trusted_or_left": la_hit, "L2_fraction": la_hit / len(pop["L-a"]),
            "L2": "holds" if la_hit / len(pop["L-a"]) >= 0.8 else "fails",
            "L2_by_d1_over_phiP": {
                "<2": [sum(1 for r in pop["L-a"] if r["d1_over_phiP"] < 2 and acc(r, k)), sum(1 for r in pop["L-a"] if r["d1_over_phiP"] < 2)],
                ">=2": [sum(1 for r in pop["L-a"] if r["d1_over_phiP"] >= 2 and acc(r, k)), sum(1 for r in pop["L-a"] if r["d1_over_phiP"] >= 2)]},
            "L-a_accepted_with_d1_over_phiP_ge_5.8": sum(1 for r in la_fire if r["d1_over_phiP"] >= K1_LIMIT),
            "L-a_accepted_max_d1_over_phiP": max((r["d1_over_phiP"] for r in la_fire), default=None),
            "L-c_changed": len(lc_changed), "L-c_changed_list": lc_changed[:50],
            "L-a_decisions": {x: sum(1 for r in pop["L-a"] if r[f"dec_{k}"] == x) for x in ("trusted", "left", "untrusted")},
        }
    summ["verdicts"] = verdict
    # L-b detail
    summ["L-b"] = [{k_: r[k_] for k_ in ("sid", "group", "depth", "P_type", "r", "d1_over_phiP", "d1_over_nu1", "r_c",
                                          "d1c_over_phiP", "D_over_eta_request", "h06_r", "h06_d1_over_phi")}
                   | {f"dec_{k}": r[f"dec_{k}"] for k in ROUTES} for r in pop["L-b"]]
    # L-a distributions
    def q(v, p):
        v = sorted(x for x in v if x is not None and math.isfinite(x))
        return v[min(len(v) - 1, int(p * len(v)))] if v else None
    summ["L-a_distributions"] = {name: {p: q([r[name] for r in pop["L-a"]], p) for p in (0.0, 0.1, 0.5, 0.9, 1.0)}
                                for name in ("r", "d1_over_phiP", "d1_over_nu1", "r_c", "d1c_over_phiP", "D_over_eta_request")}
    out_j = RC.ROOT / "out" / "partL_summary.json"
    out_j.write_text(json.dumps({"header": hdr, **summ}, indent=1, default=str))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("L-b",)}, indent=1, default=str))
    print("L-b:")
    for r in summ["L-b"]:
        print("  ", r)
    print(f"wrote {out_csv} sha256 {RC.sha256(out_csv)}; {out_j} sha256 {RC.sha256(out_j)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
