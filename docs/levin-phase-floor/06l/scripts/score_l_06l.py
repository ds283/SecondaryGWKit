"""levin-phase-floor prompt 06l, Part L: route B's probes tabulated beside 06k's columns, L1 and L2 scored by the rules
as written (prompt 06l section 4), and Gate L3 (section 2). New in 06l, after 06k's score_l.py. Read only.

Reads out/partL/<sid>.json (probe_l.py, 06l's copy) for every call of out/sample/selection.json, and 06k's committed
table docs/levin-phase-floor/06k/out/partL_leaves.csv (in the tree), whose shared columns are compared with this
run's leaf by leaf, as text (both written by Python's csv from the same floats). Writes out/partL_leaves.csv (one
row per leaf evaluation: 06k's columns, then route B's: each solve's kind, phi^- and E of P, L and R, nu_B as the
variant forms it and exactly, |E_L + E_R| / (|E_L| + |E_R|), delta1/nu_B, B's decision beside route 1's, and L3's
per-leaf checks) and out/partL_summary.json, and prints the tables.

- **L1:** of the 17 distinct 06h carrying leaves (L-b; 22 evaluations), B trusts none. A leaf counts if B trusts it
  at either of its calls.
- **L2:** B trusts all 75 of 06j's sampled slow leaves with delta1/phi_P < 2 (L-a).
- **Gate L3**, on every Part L leaf: nu_B <= nu1 (1 + 1e-12), and B trusts a leaf only if route 1 trusts it, unless
  delta1 lies within that rounding of nu1 (|delta1 - nu1| <= 1e-12 nu1), which is reported.
Usage: PYTHONPATH=<tree>:<scripts> python -B score_l_06l.py
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

ROUTES_06K = ("g", "route1", "route2", "route2_literal", "route3")
K1_LIMIT = 5.8
L3_REL = 1e-12
K06_CSV = RC.G_TREE / "docs" / "levin-phase-floor" / "06k" / "out" / "partL_leaves.csv"


def main():
    sel = json.loads((RC.ROOT / "out" / "sample" / "selection.json").read_text())
    files = [RC.ROOT / "out" / "partL" / f"{e['sid']}.json" for e in sel]
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06l: Part L tabulated for route B; L1, L2 and Gate L3",
                 files + [K06_CSV])
    for l in hdr:
        print(l)
    rows = []
    for e, f in zip(sel, files):
        d = json.loads(f.read_text())
        for L in d["leaves"]:
            E = L["E"]
            T = L["types"]
            nu1, nuB = L["nu1"], L["nu_B_variant"]
            d1 = L["delta1"]
            l3_nu = nuB <= nu1 * (1.0 + L3_REL)
            near = abs(d1 - nu1) <= L3_REL * nu1
            trB, tr1 = L["decision"]["routeB"] == "trusted", L["decision"]["route1"] == "trusted"
            l3_inc = (not trB) or tr1 or near
            sumabs = (abs(E["L"]) + abs(E["R"])) if (E["L"] is not None and E["R"] is not None) else None
            rows.append({
                # 06k's columns, as 06k's score_l.py writes them
                "population": L["population"], "sid": d["sid"], "unit": d["unit"], "index": d["index"], "rtol": d["rtol"],
                "group": e.get("g_record", {}).get("group"), "start": L["start_hex"], "end": L["end_hex"],
                "depth": L["depth"], "P_type": T["P"], "residual": L["residual"], "delta1": d1,
                "delta2": L["delta2"], "r": L["r"], "phi_P": L["phi_P"], "phi_L": L["phi_L"], "phi_R": L["phi_R"],
                "nu1": nu1, "d1_over_phiP": L["d1_over_phiP"], "d1_over_nu1": L["d1_over_nu1"],
                **{f"E_{k}": E[k] for k in ("P", "L", "R", "LL", "LR", "RL", "RR")},
                "delta1c": L["delta1c"], "delta2c": L["delta2c"], "r_c": L["r_c"], "d1c_over_phiP": L["d1c_over_phiP"],
                "D": L["D"], "request": L["request"], "D_over_eta_request": L["D_over_eta_request"],
                **{f"dec_{k}": L["decision"][k] for k in ROUTES_06K},
                "reproduces_census": L["reproduces_census"],
                "h06_r": (L.get("h06_record") or {}).get("convergence_ratio"), "h06_d1_over_phi": L.get("h06_d1_over_phi"),
                # route B's columns (prompt 06l)
                "L_type": T["L"], "R_type": T["R"],
                "phim_P": L["phim"]["P"], "phim_L": L["phim"]["L"], "phim_R": L["phim"]["R"],
                "nu_B": nuB, "nu_B_exact": L["nu_B_exact"],
                "E_LR_cancellation": L["cancellation_LR"],
                "E_LR_abs_sum": abs(E["L"] + E["R"]) if sumabs is not None else None,
                "E_L_abs_plus_E_R_abs": sumabs,
                "d1_over_nuB": L["d1_over_nuB"],
                "d1_over_EP_plus_ELR": (d1 / (abs(E["P"]) + abs(E["L"] + E["R"])) if (sumabs is not None and E["P"] is not None) else None),
                "dec_routeB": L["decision"]["routeB"],
                "nuB_over_nu1": nuB / nu1 if nu1 > 0 else None,
                "L3_nuB_le_nu1": l3_nu, "L3_inclusion": l3_inc, "L3_d1_within_rounding_of_nu1": near,
            })
    # 06k's table, leaf by leaf, on the shared columns
    k06 = []
    with open(K06_CSV, newline="") as fh:
        lines = [l for l in fh if not l.startswith("#")]
    for r in csv.DictReader(lines):
        k06.append(r)
    shared = [c for c in k06[0].keys()]
    key = lambda r: (r["population"], r["sid"], str(r["start"]), str(r["end"]))  # noqa: E731
    k06_by = {}
    for r in k06:
        k06_by.setdefault(key(r), []).append(r)
    out_csv = RC.ROOT / "out" / "partL_leaves.csv"
    with open(out_csv, "w", newline="") as fo:
        for l in hdr:
            fo.write(l + "\n")
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # re-read our own rows as text, to compare with 06k's text
    with open(out_csv, newline="") as fh:
        mine_txt = list(csv.DictReader([l for l in fh if not l.startswith("#")]))
    n_match, mism = 0, []
    for r in mine_txt:
        cands = k06_by.get(key(r), [])
        if len(cands) != 1:
            mism.append([r["sid"], r["start"], f"{len(cands)} rows of 06k at this key"])
            continue
        diffc = [c for c in shared if r[c] != cands[0][c]]
        if diffc:
            mism.append([r["sid"], r["start"], diffc])
        else:
            n_match += 1
    pop = {p: [r for r in rows if r["population"] == p] for p in ("L-a", "L-b", "L-c")}
    trB = lambda r: r["dec_routeB"] == "trusted"  # noqa: E731
    distinct = {}
    for r in pop["L-b"]:
        distinct.setdefault(("-".join(r["unit"].rsplit("-", 2)[-2:]), r["index"], r["start"], r["end"]), []).append(r)
    lb_hit = sum(1 for v in distinct.values() if any(trB(r) for r in v))
    first = [r for r in pop["L-a"] if r["d1_over_phiP"] < 2]
    second = [r for r in pop["L-a"] if r["d1_over_phiP"] >= 2]
    l2_hit = sum(1 for r in first if trB(r))
    summ = {
        "n": {p: len(v) for p, v in pop.items()},
        "reproduces_census": {p: sum(1 for r in v if r["reproduces_census"]) for p, v in pop.items()},
        "against_06k_table": {"rows": len(mine_txt), "rows_of_06k": len(k06), "shared_columns": len(shared),
                              "rows_equal_on_every_shared_column": n_match, "mismatches": mism[:50]},
        "L1": {"rule": "of the 17 06h carrying leaves (L-b), B trusts none", "distinct": len(distinct),
               "distinct_trusted": lb_hit, "evaluations_trusted": [r["sid"] for r in pop["L-b"] if trB(r)],
               "verdict": "holds" if lb_hit == 0 and len(distinct) == 17 else "fails"},
        "L2": {"rule": "B trusts all 75 of 06j's sampled slow leaves with delta1/phi_P < 2", "n_first_population": len(first),
               "trusted": l2_hit, "not_trusted": [{k: r[k] for k in ("sid", "start", "end", "r", "d1_over_phiP", "d1_over_nuB", "P_type", "L_type", "R_type")}
                                                  for r in first if not trB(r)],
               "verdict": "holds" if l2_hit == len(first) == 75 else "fails"},
        "L3": {"rule": "on every Part L leaf nu_B <= nu1 (1 + 1e-12), and B trusts only where route 1 does unless delta1 is within 1e-12 nu1 of nu1",
               "leaves": len(rows),
               "nuB_le_nu1_fails": [[r["population"], r["sid"], r["start"], r["nuB_over_nu1"]] for r in rows if not r["L3_nuB_le_nu1"]],
               "inclusion_fails": [[r["population"], r["sid"], r["start"]] for r in rows if not r["L3_inclusion"]],
               "B_trusts_route1_not_within_rounding": [[r["population"], r["sid"], r["start"], r["delta1"], r["nu1"]] for r in rows
                                                       if trB(r) and r["dec_route1"] != "trusted" and r["L3_d1_within_rounding_of_nu1"]],
               "nuB_gt_nu1_count": sum(1 for r in rows if r["nu1"] > 0 and r["nu_B"] > r["nu1"]),
               "nuB_over_nu1_max": max((r["nuB_over_nu1"] for r in rows if r["nuB_over_nu1"] is not None), default=None),
               "nuB_variant_vs_exact_rel_max": max((abs(r["nu_B"] - r["nu_B_exact"]) / r["nu_B_exact"] for r in rows if r["nu_B_exact"] > 0), default=None)},
        "L-a": {
            "trusted_B": sum(1 for r in pop["L-a"] if trB(r)), "trusted_route1": sum(1 for r in pop["L-a"] if r["dec_route1"] == "trusted"),
            "first_population": [l2_hit, len(first)],
            "second_population_trusted": [sum(1 for r in second if trB(r)), len(second)],
            "second_population_trusted_route1": sum(1 for r in second if r["dec_route1"] == "trusted"),
            "trusted_B_at_d1_over_phiP_ge_5.8": sum(1 for r in pop["L-a"] if trB(r) and r["d1_over_phiP"] >= K1_LIMIT),
            "trusted_route1_at_d1_over_phiP_ge_5.8": sum(1 for r in pop["L-a"] if r["dec_route1"] == "trusted" and r["d1_over_phiP"] >= K1_LIMIT),
            "kinds": sorted({(r["P_type"], r["L_type"], r["R_type"]) for r in pop["L-a"]}),
            "second_population_trusted_list": [{k: r[k] for k in ("sid", "start", "r", "d1_over_phiP", "d1_over_nuB", "d1_over_nu1",
                                                                 "E_LR_cancellation", "d1_over_EP_plus_ELR")} for r in second if trB(r)],
        },
        "L-b": [{k: r[k] for k in ("sid", "group", "depth", "P_type", "L_type", "R_type", "r", "d1_over_phiP", "d1_over_nu1",
                                   "d1_over_nuB", "dec_route1", "dec_routeB")} for r in pop["L-b"]],
        "L-c": {"changed_from_trusted_under_B": [[r["sid"], r["start"], r["dec_routeB"]] for r in pop["L-c"] if r["dec_routeB"] != "trusted"],
                "kinds": {f"{a}/{b}+{c}": sum(1 for r in pop["L-c"] if (r["P_type"], r["L_type"], r["R_type"]) == (a, b, c))
                          for a, b, c in sorted({(r["P_type"], r["L_type"], r["R_type"]) for r in pop["L-c"]})}},
    }
    summ["L3"]["verdict"] = "holds" if not summ["L3"]["nuB_le_nu1_fails"] and not summ["L3"]["inclusion_fails"] else "FAILS"

    def q(v, p):
        v = sorted(x for x in v if x is not None and math.isfinite(x))
        return v[min(len(v) - 1, int(p * len(v)))] if v else None
    summ["L-a_distributions"] = {
        f"{name} ({lab})": {p: q([r[name] for r in grp], p) for p in (0.0, 0.1, 0.5, 0.9, 1.0)}
        for lab, grp in (("first", first), ("second", second))
        for name in ("r", "d1_over_phiP", "d1_over_nu1", "d1_over_nuB", "E_LR_cancellation", "nuB_over_nu1", "d1_over_EP_plus_ELR")}
    out_j = RC.ROOT / "out" / "partL_summary.json"
    out_j.write_text(json.dumps({"header": hdr, **summ}, indent=1, default=str))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("L-b",)}, indent=1, default=str))
    print("L-b:")
    for r in summ["L-b"]:
        print("  ", r)
    print(f"wrote {out_csv} sha256 {RC.sha256(out_csv)}; {out_j} sha256 {RC.sha256(out_j)}")
    return 0 if summ["L3"]["verdict"] == "holds" else 1


if __name__ == "__main__":
    sys.exit(main())
