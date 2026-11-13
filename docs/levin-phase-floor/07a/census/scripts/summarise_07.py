"""levin-phase-floor prompt 07 section 6: the verdicts of V1, V2, V3 and V5 and the descriptive tables, from
score_07.py's output (read only). New in 07. Rules: state/predictions.md (parts 1, 2 and 4).
Writes OUT_JSON (every number and every listed call) and prints the summary.
Usage: python -B summarise_07.py SCORE_JSON OUT_JSON
"""

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402

TWO_PHI_W = 4.0 * math.pi  # theta'^2/|theta''| at a root's own window edge, 2 Phi_w with Phi_w = 2 pi


def H(x):
    return None if x is None else float.fromhex(x)


def hist(xs):
    """A histogram with None (no call) sorted last."""
    return {str(k): v for k, v in sorted(Counter(xs).items(), key=lambda kv: (kv[0] is None, kv[0] if kv[0] is not None else 0))}


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(math.floor(p * (len(xs) - 1) + 0.5)))] if xs else None


def main():
    src, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07: V1, V2, V3, V5 and the descriptive tables", [src])
    doc = json.loads(src.read_text())
    units = doc["units"]
    S = {"header": hdr, "source": str(src), "source_sha256": sha256(src)}
    S["statuses"] = dict(Counter(u["status"] for u in units))
    S["statuses_06m"] = dict(Counter(u["status_06m"] for u in units))
    S["scan_07_errors"] = sum(u.get("scan_07_error") or 0 for u in units)
    S["n_scanned_calls"] = {rt: {k: sum(u["n_scanned_by_kind"][k] for u in units if u["rtol"] == rt) for k in ("group", "analytic")}
                            for rt in ("1e-7", "1e-8")}
    # ---- V1
    S["V1"] = {"stop3": doc["stop3"], "by_rtol": doc["stop3_by_rtol"]}
    S["V1"]["holds"] = doc["stop3"]["n_units_differing"] == 0 and all(
        v["units"] == v["units_ok"] and v["units_all_clean"] == v["units_all_clean_total_equal"] for v in doc["stop3_by_rtol"].values())
    # ---- per-call walk
    V2_levin, listed_other = [], Counter()
    V3_viol, V3_set_aside, V3_end = [], [], []
    v3_scored = 0
    V5_viol = []
    located, guard_only = [], []
    for u in units:
        for c in u.get("calls") or []:
            key = {"unit": u["unit"], "rtol": u["rtol"], "kind": c["kind"], "index": c["index"], "label": c["label"]}
            for x in c.get("listed") or []:
                cls = ("Levin" if x["branch"] == "Levin" else
                       "window" if x["window"] == 1 else "guard_at_cap" if x["guard_at_cap"] == 1 else "CC")
                if cls == "Levin":
                    V2_levin.append(dict(key, **x))
                listed_other[(u["rtol"], c["kind"], cls)] += 1
            if c.get("locates") or c.get("guards"):
                if c.get("n_at_depth_max"):
                    V5_viol.append(dict(key, n_at_depth_max=c["n_at_depth_max"], max_depth=c["max_depth"],
                                        converged=c["converged"], points=[p["kind"] for p in c.get("points") or []]))
                (located if c.get("locates") else guard_only).append((u, c))
            if c.get("scanned") and "scan" in c:
                sc = c["scan"]
                if c["skipped"]:
                    V3_set_aside.append(dict(key, skipped=c["skipped"], n_sign_changes=sc["n_sign_changes"],
                                             sign_change_x=sc["sign_change_x"][:8], locates=c.get("locates"),
                                             guards=c.get("guards")))
                    continue
                v3_scored += 1
                h = H(sc["h"])
                S_x = sc["sign_change_x"]
                roots = [p["x"] for p in c.get("points") or [] if p["kind"] == "root"]
                ends = [p["x"] for p in c.get("points") or [] if p["kind"] == "end"]
                miss_change = [x for x in S_x if not any(abs(r - x) <= h for r in roots)]
                miss_root = [r for r in roots if not any(abs(r - x) <= h for x in S_x)]
                if ends:
                    alt = roots + ends
                    V3_end.append(dict(key, ends=ends, found_as=[p.get("found_as") for p in c.get("points") or [] if p["kind"] == "end"],
                                       n_sign_changes=len(S_x),
                                       violates_as_roots=bool([x for x in S_x if not any(abs(r - x) <= h for r in alt)]
                                                              or [r for r in alt if not any(abs(r - x) <= h for x in S_x)])))
                if miss_change or miss_root:
                    V3_viol.append(dict(key, h=h, sign_change_x=S_x[:8], roots=roots[:8], ends=ends,
                                        changes_without_root=miss_change[:8], roots_without_change=miss_root[:8]))
    S["V2"] = {"n_levin_leaves_listed": len(V2_levin), "levin_leaves": V2_levin[:200],
               "listed_by_rtol_kind_class": {"|".join(k): v for k, v in sorted(listed_other.items())},
               "holds": len(V2_levin) == 0}
    S["V3"] = {"n_calls_scored": v3_scored, "n_violating": len(V3_viol), "violations": V3_viol[:200],
               "n_set_aside": len(V3_set_aside),
               "set_aside_with_sign_change": [x for x in V3_set_aside if x["n_sign_changes"]],
               "set_aside_by": dict(Counter(f"{x['rtol']}|{x['kind']}|sign_changes={x['n_sign_changes'] > 0}|guards={x['guards']}" for x in V3_set_aside)),
               "set_aside_variation_range": [min(H(s["variation"]["hex"]) for x in V3_set_aside for s in x["skipped"]),
                                             max(H(s["variation"]["hex"]) for x in V3_set_aside for s in x["skipped"])] if V3_set_aside else None,
               "end_points": V3_end, "holds": len(V3_viol) == 0}
    S["V3"]["set_aside_list"] = V3_set_aside
    S["V5"] = {"n_violating_calls": len(V5_viol), "violations": V5_viol, "over_budget_units": S["statuses"].get("over budget", 0),
               "holds": len(V5_viol) == 0 and not S["statuses"].get("over budget")}
    # ---- descriptive: locations, windows, stopped windows
    loc_kinds, minima_l, end_pts, an_loc, pieces, widths, edge, stopped = Counter(), [], [], Counter(), [], [], [], []
    win_stop = defaultdict(lambda: {"windows": 0, "stopped": 0, "falling_hist": Counter(), "calls_with_stopped": 0,
                                    "calls_with_stopped_converged": 0, "calls_with_falling": 0, "calls_with_falling_converged": 0})
    conv_loc = Counter()
    unloc_min = []
    for u, c in located:
        for p in c["points"]:
            loc_kinds[(u["rtol"], c["kind"], p["kind"])] += 1
            if p["kind"] == "minimum":
                minima_l.append(p["lambda"])
            if p["kind"] == "end":
                end_pts.append({"unit": u["unit"], "kind": c["kind"], "index": c["index"], "x": p["x"], "found_as": p["found_as"]})
            for e in p["edge_ratio"] or []:
                edge.append(e)
        if c["kind"] == "analytic":
            an_loc[(u["rtol"], c["label"])] += 1
        pieces.append(c.get("num_window_pieces"))
        for w in c["windows"]:
            widths.append(w["hi"] - w["lo"])
            ws = win_stop[u["rtol"]]
            ws["windows"] += 1
            if w["stopped"]:
                ws["stopped"] += 1
                ws["falling_hist"][w["leaves_falling_at_stop"]] += 1
                stopped.append({"unit": u["unit"], "rtol": u["rtol"], "kind": c["kind"], "index": c["index"],
                                "label": c["label"], "iroff1_count": w["iroff1_count"],
                                "stopped_after_split": w["stopped_after_split"], "leaves_at_stop": w["leaves_at_stop"],
                                "leaves_falling_at_stop": w["leaves_falling_at_stop"], "max_depth_at_stop": w["max_depth_at_stop"],
                                "call_converged": c["converged"][0], "call_converged_06m": c["converged"][1]})
        st = [w for w in c["windows"] if w["stopped"]]
        if st:
            ws = win_stop[u["rtol"]]
            ws["calls_with_stopped"] += 1
            ws["calls_with_stopped_converged"] += bool(c["converged"][0])
            if any(w["leaves_falling_at_stop"] for w in st):
                ws["calls_with_falling"] += 1
                ws["calls_with_falling_converged"] += bool(c["converged"][0])
        conv_loc[(u["rtol"], c["kind"], f"g={c['converged'][0]}", f"06m={c['converged'][1]}")] += 1
    for u in units:
        for c in u.get("calls") or []:
            if c.get("unlocated_minima_lambda"):
                unloc_min += c["unlocated_minima_lambda"]
    S["locations"] = {
        "located_calls": len(located), "points_by_rtol_kind_pointkind": {"|".join(k): v for k, v in sorted(loc_kinds.items())},
        "minima_located": len(minima_l), "minima_lambda_range": [min(minima_l), max(minima_l)] if minima_l else None,
        "unlocated_minima": len(unloc_min), "unlocated_minima_lambda_range": [min(unloc_min), max(unloc_min)] if unloc_min else None,
        "end_points": end_pts, "analytic_calls_that_located": {"|".join(k): v for k, v in sorted(an_loc.items())},
        "guard_only_calls": [{"unit": u["unit"], "rtol": u["rtol"], "kind": c["kind"], "index": c["index"], "label": c["label"],
                              "num_guarded_regions": c["num_guarded_regions"], "skipped": c["skipped"], "converged": c["converged"]}
                             for u, c in guard_only],
        "num_guarded_regions_total": sum((c.get("num_guarded_regions") or 0) for _, c in located + guard_only),
        "num_guard_at_depth_max_total": sum((c.get("num_guard_at_depth_max") or 0) for _, c in located + guard_only),
        "located_calls_by_convergence": {"|".join(k): v for k, v in sorted(conv_loc.items())},
    }
    S["windows"] = {
        "window_pieces_per_call": dict(sorted(Counter(pieces).items(), key=lambda kv: (kv[0] is None, kv[0]))),
        "width_x": {"n": len(widths), "min": min(widths), "median": q(widths, 0.5), "max": max(widths)} if widths else None,
        "edge_theta_deriv_sq_over_second": {"n": len(edge), "min": min(edge), "median": q(edge, 0.5), "max": max(edge),
                                            "n_within_1e-3_of_4pi": sum(1 for e in edge if abs(e / TWO_PHI_W - 1) <= 1e-3)} if edge else None,
        "stopped_by_rtol": {rt: dict(v, falling_hist=dict(sorted(v["falling_hist"].items()))) for rt, v in win_stop.items()},
        "stopped_windows": stopped,
    }
    # ---- depth
    dep = {}
    for rt in ("1e-7", "1e-8"):
        us = [u for u in units if u["rtol"] == rt and u["status"] == "done"]
        dep[rt] = {"g": hist(u["row_depth"][0] for u in us),
                   "06m": hist(u["row_depth"][1] for u in us),
                   "g_in720": hist(u["row_depth"][0] for u in us if u["in720"]),
                   "06m_in720": hist(u["row_depth"][1] for u in us if u["in720"]),
                   "rows_at_20": [sum(1 for u in us if (u["row_depth"][0] or 0) >= 20), sum(1 for u in us if (u["row_depth"][1] or 0) >= 20)],
                   "rows_ge_18": [sum(1 for u in us if (u["row_depth"][0] or 0) >= 18), sum(1 for u in us if (u["row_depth"][1] or 0) >= 18)]}
    S["depth"] = dep
    # ---- totals in the 720, by band and k
    tot = defaultdict(list)
    for u in units:
        if not u["in720"] or u["status"] != "done":
            continue
        tg, t6, e6 = H(u["total"][0]), H(u["total"][1]), H(u["total_abserr"][1])
        eg = H(u["total_abserr"][0])
        tot[(u["rtol"], u["band"], u["k"])].append((abs(tg - t6) / abs(t6) if t6 else math.inf, abs(tg - t6) / e6 if e6 else math.inf,
                                                   eg / e6 if e6 else math.inf, u["unit"]))
    S["totals_720"] = {"|".join(k): {"n": len(v), "rel_max": max(x[0] for x in v), "rel_median": q([x[0] for x in v], 0.5),
                                     "over_abserr06m_max": max(x[1] for x in v), "over_abserr06m_median": q([x[1] for x in v], 0.5),
                                     "n_over_abserr06m_gt_1": sum(1 for x in v if x[1] > 1),
                                     "abserr_g_over_06m_median": q([x[2] for x in v], 0.5),
                                     "largest": max(v, key=lambda x: x[1])[3]}
                       for k, v in sorted(tot.items())}
    allv = [x for v in tot.values() for x in v]
    S["totals_720_all"] = {"n": len(allv), "n_over_abserr06m_gt_1": sum(1 for x in allv if x[1] > 1),
                           "worst": sorted(allv, key=lambda x: -x[1])[:10]}
    # ---- convergence and the flag (06m's table's measures), group and analytic
    conv = {}
    for side, fk, ci in (("g", "flag", 0), ("06m", "flag_06m", 1)):
        for where in (True, False):
            cs = [(u, c) for u in units for c in u.get("calls") or [] if u["in720"] == where and not c["converged"][ci]]
            for kind in ("group", "analytic"):
                ck = [(u, c) for u, c in cs if c["kind"] == kind]
                fl = [c[fk] for _, c in ck]
                shares = [f["share"] for f in fl if f["share"] is not None]
                conv[f"{side}|in720={where}|{kind}"] = {
                    "calls": len(ck), "units": len({u["unit"] for u, _ in ck}), "rows": len({u["row"] for u, _ in ck}),
                    "at_1e-7": sum(1 for u, _ in ck if u["rtol"] == "1e-7"), "at_1e-8": sum(1 for u, _ in ck if u["rtol"] == "1e-8"),
                    "within_request": sum(1 for f in fl if f["within_request"]),
                    "over_request": sum(1 for f in fl if not f["within_request"]),
                    "over_request_node_limited": sum(1 for f in fl if not f["within_request"] and f["node_limited"]),
                    "over_request_max_ratio": max([f["over_request_ratio"] for f in fl if not f["within_request"]], default=None),
                    "node_limited": sum(1 for f in fl if f["node_limited"]),
                    "holding_untrusted_leaf": sum(1 for f in fl if f["untrusted"] > 0),
                    "at_depth_max": sum(1 for f in fl if f["n_at_depth_max"]),
                    "share_max": max(shares, default=None),
                    "share_bins": {"<=1%": sum(1 for s in shares if s <= 0.01), "(1%,5%]": sum(1 for s in shares if 0.01 < s <= 0.05),
                                   "(5%,50%]": sum(1 for s in shares if 0.05 < s <= 0.5), ">50%": sum(1 for s in shares if s > 0.5)},
                }
    S["convergence"] = conv
    S["convergence_changes_located"] = {"to_converged": sum(1 for _, c in located if c["converged"][0] and not c["converged"][1]),
                                        "to_unconverged": sum(1 for _, c in located if not c["converged"][0] and c["converged"][1])}
    # ---- cost
    cost = {}
    for rt in ("1e-7", "1e-8"):
        us = [u for u in units if u["rtol"] == rt]
        cost[rt] = {"group_regions": [sum(u["group_regions"][0] for u in us), sum(u["group_regions"][1] for u in us)],
                    "unit_wall_s": [sum(u["wall_s"][0] for u in us), sum(u["wall_s"][1] for u in us)],
                    "unit_time_s": [sum(u["time_s"][0] for u in us), sum(u["time_s"][1] for u in us)]}
    ratios = sorted(((u["wall_s"][0] / u["wall_s"][1], u["unit"], u["wall_s"][0], u["wall_s"][1]) for u in units if u["wall_s"][1]), reverse=True)
    cost["unit_wall_ratio_g_over_06m"] = {"max10": ratios[:10], "min3": ratios[-3:], "median": q([r[0] for r in ratios], 0.5),
                                          "n_over_2": sum(1 for r in ratios if r[0] > 2)}
    cost["largest_unit_wall_g"] = sorted(((u["wall_s"][0], u["unit"], u["wall_s"][1]) for u in units), reverse=True)[:10]
    cost["over_300s"] = sum(1 for u in units if (u["wall_s"][0] or 0) > 300)
    loc_units = {u["unit"] for u, _ in located + guard_only}
    cost["located_units_wall"] = [sum(u["wall_s"][0] for u in units if u["unit"] in loc_units),
                                  sum(u["wall_s"][1] for u in units if u["unit"] in loc_units), len(loc_units)]
    S["cost"] = cost
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(S, indent=1, default=str))
    brief = {k: S[k] for k in ("statuses", "scan_07_errors", "n_scanned_calls")}
    print(json.dumps(brief))
    print("V1", S["V1"]["holds"], json.dumps(S["V1"]["by_rtol"]))
    print("V2", S["V2"]["holds"], S["V2"]["n_levin_leaves_listed"], json.dumps(S["V2"]["listed_by_rtol_kind_class"]))
    print("V3", S["V3"]["holds"], S["V3"]["n_calls_scored"], S["V3"]["n_violating"], "set aside", S["V3"]["n_set_aside"],
          json.dumps(S["V3"]["set_aside_by"]), "V range", S["V3"]["set_aside_variation_range"], "ends", len(S["V3"]["end_points"]))
    print("V5", S["V5"]["holds"], S["V5"]["n_violating_calls"])
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
