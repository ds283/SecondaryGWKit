"""levin-phase-floor prompt 07a section 6: W1, W2, W3, W5 and W7 scored, W4 and W6 gathered, and the descriptive tables.
New in 07a, after 07's summarise_07.py (a record, not edited), whose V2 and V3 rules it keeps. Read only.

The rules are state/predictions.md's, under the user's rulings at STOP 2 (state/ruling_stop2.md: Q1 reading A, Q2 the
call's outputs, with the ``location`` and ``harness`` categories set aside as phase_span is). Each W is also given under
reading B and strictly, for the record. Inputs: state/score_07a_units.json (score_07a.py), out/stop3/
compare_census_calls.json and its .calls.jsonl (compare_07a.py), out/scores/w4_scores.json (score_v4_07a.py), and the
cross-checks out/xcheck/h_vs_07_pop07/xcheck.json, out/xcheck/h_vs_07_poph/xcheck.json and
out/xcheck/h_vs_06m_pop07/disagreement.json (xcheck_07a.py, xcheck_06m_from07_07a.py). Floats compared exactly
(Fraction of their hex).

Usage: python -B summarise_07a.py OUT_JSON
"""

import json
import math
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header, sha256  # noqa: E402

UNITS = WIDER / "state" / "score_07a_units.json"
CMP = WIDER / "out" / "stop3" / "compare_census_calls.json"
CALLS = WIDER / "out" / "stop3" / "compare_census_calls.calls.jsonl"
W4 = WIDER / "out" / "scores" / "w4_scores.json"
X07 = WIDER / "out" / "xcheck" / "h_vs_07_pop07" / "xcheck.json"
XH = WIDER / "out" / "xcheck" / "h_vs_07_poph" / "xcheck.json"
X06 = WIDER / "out" / "xcheck" / "h_vs_06m_pop07" / "disagreement.json"
SIX = 6.0 * math.pi
LU = 20.0 * math.pi


def F(h):
    return Fraction(float.fromhex(h))


def H(x):
    return None if x is None else float.fromhex(x["hex"] if isinstance(x, dict) else x)


def q(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return {"n": len(xs), "min": xs[0], "p10": xs[len(xs) // 10], "median": xs[len(xs) // 2], "p90": xs[9 * len(xs) // 10],
            "p99": xs[min(len(xs) - 1, 99 * len(xs) // 100)], "max": xs[-1]}


def ulp(x):
    return math.ulp(x) if x is not None else None


def main():
    out_p = Path(sys.argv[1])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07a: W1-W7 and the descriptive tables",
                 [UNITS, CMP, CALLS, W4, X07, XH, X06])
    U = json.loads(UNITS.read_text())["units"]
    cmpd = json.loads(CMP.read_text())
    S = {"header": hdr, "statuses": dict(Counter(u["status"] for u in U)), "statuses_07": dict(Counter(u["status_07"] for u in U)),
         "scan_07_errors": sum(u.get("scan_07_error") or 0 for u in U),
         "n_scanned_calls": {rt: {k: sum(u["n_scanned_by_kind"][k] for u in U if u["rtol"] == rt) for k in ("group", "analytic")}
                             for rt in ("1e-7", "1e-8")},
         "route_errors": sum(u["agg"]["route_errors"] for u in U), "no_build_solves": sum(u["agg"]["no_build"] for u in U)}
    unit_info = {u["unit"]: u for u in U}
    calls = [json.loads(l) for l in open(CALLS)]
    # ---------------- W1
    w1 = {}
    for reading, cls in (("A", {"untouched"}), ("B", {"untouched", "aux_only"})):
        for rt in ("1e-7", "1e-8"):
            cs = [c for c in calls if c["rtol"] == rt and c["class"] in cls]
            w1[f"{reading}|{rt}"] = {"calls": len(cs), "strict_equal": sum(1 for c in cs if c["equal"]),
                                     "call_category_differing": sum(1 for c in cs if "call" in c["categories"]),
                                     "location_only": sum(1 for c in cs if c["categories"] == ["location"]),
                                     "harness_only": sum(1 for c in cs if c["categories"] == ["harness"]),
                                     "harness_and_location_only": sum(1 for c in cs if c["categories"] == ["harness", "location"]),
                                     "call_differing_list": [{k: c[k] for k in ("unit", "kind", "index", "class", "first_differing")}
                                                             for c in cs if "call" in c["categories"]][:50]}
            us = [u for u in cmpd["units"] if u["rtol"] == rt and (u.get("all_w1") or {}).get(reading)]
            w1[f"{reading}|{rt}"].update(rows_all_w1=len(us), rows_totals_equal=sum(1 for u in us if u["totals_equal"]),
                                         rows_whole_equal=sum(1 for u in us if (u.get("whole") or {}).get("equal")),
                                         rows_whole_categories=dict(Counter("+".join((u.get("whole") or {}).get("categories") or []) or "none" for u in us)),
                                         rows_totals_differing=[u["unit"] for u in us if not u["totals_equal"]][:50])
    S["W1"] = {"by_reading_rtol": w1,
               "holds_A_ruled": all(w1[f"A|{rt}"]["call_category_differing"] == 0 and w1[f"A|{rt}"]["rows_all_w1"] == w1[f"A|{rt}"]["rows_totals_equal"]
                                    for rt in ("1e-7", "1e-8")),
               "holds_A_strict": all(w1[f"A|{rt}"]["calls"] == w1[f"A|{rt}"]["strict_equal"] for rt in ("1e-7", "1e-8")),
               "holds_B_ruled": all(w1[f"B|{rt}"]["call_category_differing"] == 0 and w1[f"B|{rt}"]["rows_all_w1"] == w1[f"B|{rt}"]["rows_totals_equal"]
                                    for rt in ("1e-7", "1e-8"))}
    # ---------------- per-call walk of score_07a's units (W2, W3, W5, descriptive)
    V2_levin, listed_other = [], Counter()
    V3_viol, V3_set_aside, v3_scored = [], [], 0
    w5 = Counter()
    w5_calls = {"partition_nonforced": [], "partition_forced": [], "cc_lh": [], "guarded": []}
    changed_solves = []
    win = defaultdict(Counter)
    stopped_list = []
    loc_kinds, pieces_h, gt64 = Counter(), [], []
    newly = []
    unconv = defaultdict(Counter)
    for u in U:
        a = u["agg"]
        w5["partition_nonforced"] += a["partition_nonforced"]
        w5["partition_forced"] += a["partition_forced"]
        w5["cc_with_levin_half"] += a["cc_lh"]
        w5["cc_with_levin_half_07est"] += a["cc_lh_07est"]
        w5["num_guarded_regions"] += a["guarded"]
        w5["num_guard_at_depth_max"] += a["guard_at_depth_max"]
        for c in u.get("calls") or []:
            key = {"unit": u["unit"], "rtol": u["rtol"], "kind": c["kind"], "index": c["index"], "label": c["label"],
                   "band": u["band"], "k": u["k"], "in720": u["in720"]}
            r = c.get("route") or {}
            if r.get("n_partition_nonforced") or (r.get("partition") or {}).get("nonforced") and sum((r.get("partition") or {}).get("nonforced", {}).values()):
                w5_calls["partition_nonforced"].append(dict(key, partition=r.get("partition")))
            if (r.get("partition") or {}).get("forced"):
                w5_calls["partition_forced"].append(dict(key, partition=r.get("partition")))
            if r.get("n_cc_with_levin_half"):
                w5_calls["cc_lh"].append(dict(key, n=r["n_cc_with_levin_half"], detail=c.get("cc_with_levin_half")))
            if c.get("num_guarded_regions"):
                w5_calls["guarded"].append(dict(key, n=c["num_guarded_regions"]))
            for x in c.get("listed") or []:
                cls = ("Levin" if x["branch"] == "Levin" else "window" if x["window"] == 1 else
                       "guard_at_cap" if x["guard_at_cap"] == 1 else "CC")
                if cls == "Levin":
                    V2_levin.append(dict(key, **x))
                listed_other[(u["rtol"], c["kind"], cls)] += 1
            if c.get("scanned") and "scan" in c:
                sc = c["scan"]
                if c["skipped"][0]:
                    V3_set_aside.append(dict(key, skipped=c["skipped"][0], n_sign_changes=sc["n_sign_changes"],
                                             min_over_max=sc.get("min_over_max_abs_theta_deriv"), locates=c.get("locates")))
                else:
                    v3_scored += 1
                    h = H(sc["h"])
                    S_x = sc["sign_change_x"]
                    roots = [p["x"] for p in c.get("points") or [] if p["kind"] == "root"]
                    miss_change = [x for x in S_x if not any(abs(rr - x) <= h for rr in roots)]
                    miss_root = [rr for rr in roots if not any(abs(rr - x) <= h for x in S_x)]
                    if miss_change or miss_root:
                        V3_viol.append(dict(key, h=h, sign_change_x=S_x[:8], roots=roots[:8], changes_without_root=miss_change[:8],
                                            roots_without_change=miss_root[:8], ends=[p["x"] for p in c.get("points") or [] if p["kind"] == "end"]))
            for x in c.get("changed") or []:
                V, est = H(x["V"]), H(x["est_07"])
                if x["decision"] == 1:
                    dirn = "to CC (V < 6pi <= est)" if V < SIX else "to Levin (est < 6pi <= V)"
                else:
                    dirn = "to LU (est <= 20pi < V)" if V > LU else "to least squares (V <= 20pi < est)"
                changed_solves.append(dict(key, cls=x["class"], role=x["role"], dirn=dirn, V=V, est=est,
                                           ratio=(est / V) if V else None, returned=x["returned"]))
            # windows (both codes) for the calls with a window at h or at 07
            for w in c.get("windows") or []:
                win[u["rtol"]]["windows_h"] += 1
                if w["stopped"]:
                    win[u["rtol"]]["stopped_h"] += 1
                    if w["leaves_falling_at_stop"]:
                        win[u["rtol"]]["stopped_falling_h"] += 1
                    stopped_list.append(dict(key, iroff1=w["iroff1_count"], falling=w["leaves_falling_at_stop"],
                                             converged=c["converged"]))
            for w in c.get("windows_07") or []:
                win[u["rtol"]]["windows_07_recorded"] += 1
            if c.get("windows"):
                st = [w for w in c["windows"] if w["stopped"]]
                if st:
                    win[u["rtol"]]["calls_with_stopped_h"] += 1
                    win[u["rtol"]]["calls_with_stopped_h_unconverged"] += not c["converged"][0]
            for p in c.get("points") or []:
                loc_kinds[(u["rtol"], c["kind"], p["kind"])] += 1
            if c.get("points") is not None and c["n_points"][0]:
                pieces_h.append((c.get("num_window_pieces") or [None])[0])
                if c["n_points"][0] > 64:
                    gt64.append(dict(key, n_points=c["n_points"][0]))
            if c["n_windows"][0] and not c["n_windows"][1]:
                newly.append(dict(key, n_points=c["n_points"][0], kinds=dict(Counter(p["kind"] for p in c.get("points") or [])),
                                  pieces=(c.get("num_window_pieces") or [None])[0], skipped_07=c["skipped"][1],
                                  converged=c["converged"]))
            for j, code in ((0, "h"), (1, "07")):
                if c["converged"][j] is False:
                    unconv[code][(u["rtol"], c["kind"], "720" if u["in720"] else "outside")] += 1
    S["W2"] = {"n_levin_leaves_listed": len(V2_levin), "levin_leaves": V2_levin[:100],
               "listed_by_rtol_kind_class": {"|".join(k): v for k, v in sorted(listed_other.items())}, "holds": len(V2_levin) == 0}
    bad_skip = [x for x in V3_set_aside if any(s.get("reason") != "zero_rate" for s in x["skipped"]) or x["n_sign_changes"]]
    S["W3"] = {"n_calls_scored": v3_scored, "n_violating": len(V3_viol), "violations": V3_viol[:100],
               "n_set_aside": len(V3_set_aside),
               "set_aside_by": dict(Counter(f"{x['rtol']}|{x['kind']}|{','.join(s.get('reason') or 'NO REASON' for s in x['skipped'])}|"
                                            f"sign_changes={x['n_sign_changes']}|min_over_max={x['min_over_max']}" for x in V3_set_aside)),
               "set_aside_not_zero_rate_or_with_sign_change": bad_skip[:50], "set_aside_list": V3_set_aside,
               "holds": len(V3_viol) == 0 and not bad_skip}
    S["W5"] = {"counts": dict(w5), "calls": {k: v[:50] for k, v in w5_calls.items()},
               "n_calls": {k: len(v) for k, v in w5_calls.items()},
               "holds": w5["cc_with_levin_half"] == 0 and w5["partition_nonforced"] == 0 and w5["partition_forced"] == 0
               and w5["num_guarded_regions"] == 0,
               "holds_gated_partition_only": w5["cc_with_levin_half"] == 0 and w5["partition_nonforced"] == 0 and w5["num_guarded_regions"] == 0}
    # ---------------- W4, W6 (gathered)
    w4 = json.loads(W4.read_text())
    S["W4"] = {"V4": w4["V4"], "ratios": {x["sid"]: x["ratio_g"] for x in w4["calls"] + w4["z"]},
               "three_figures": {x["sid"]: float(f"{x['ratio_g']:.3g}") for x in w4["calls"] + w4["z"] if x["verdict"] != "honest"},
               "all_bit_identical_to_07": all(x["value_equal_07"] and x["abserr_equal_07"] for x in w4["calls"] + w4["z"])}
    x07 = json.loads(X07.read_text())
    S["W6"] = {"pairs": x07["A"]["pairs"], "r_g_le_1": x07["A"]["overall"]["r_g<=1"], "unpaired": len(x07["A"]["unpaired"]),
               "r_g_gt_1": [{k: p[k] for k in ("row", "kind", "index", "group", "r_g", "r_m", "converged")} for p in x07["A"]["r_g_gt_1"]],
               "dist_r_g": x07["A"]["dist"]["r_g"], "checkpoints_unchanged": x07["checkpoints_unchanged"]}
    S["W6"]["holds"] = S["W6"]["r_g_le_1"] == S["W6"]["pairs"]
    xh = json.loads(XH.read_text())
    S["W6"]["h_population_descriptive"] = {"pairs": xh["A"]["pairs"], "r_g_le_1": xh["A"]["overall"]["r_g<=1"],
                                          "r_g_gt_1": [(p["row"], p["kind"], p["index"], p["r_g"]) for p in xh["A"]["r_g_gt_1"]]}
    # ---------------- W7
    w7 = {}
    for reading in ("A", "B"):
        for rt in ("1e-7", "1e-8"):
            cov, viol = [], []
            for c in calls:
                if c["rtol"] != rt or c["class"] != "window_both":
                    continue
                r = c["route"] or {}
                if reading == "A" and (r.get("n_route_differs") or r.get("n_lu_differs")):
                    continue
                if reading == "B" and (r.get("n_route_differs_own") or r.get("n_lu_differs_own")):
                    continue
                newly_loc = bool(c["skipped_07"]) and not c["skipped_h"]
                if newly_loc:
                    continue
                vh, v7, e7 = F(c["value"][0]), F(c["value"][1]), F(c["abserr"][1])
                mv = abs(vh - v7)
                ok = mv <= Fraction(1, 100000) * e7 and c["converged"][0] == c["converged"][1]
                rec = {k: c[k] for k in ("unit", "kind", "index", "converged")} | {
                    "move_over_abserr07": float(mv / e7) if e7 else (0.0 if mv == 0 else math.inf)}
                cov.append(rec)
                if not ok:
                    viol.append(rec)
            w7[f"{reading}|{rt}"] = {"calls": len(cov), "violating": len(viol), "violations": viol[:50],
                                     "n_moved": sum(1 for x in cov if x["move_over_abserr07"] > 0),
                                     "max_move_over_abserr07": max((x["move_over_abserr07"] for x in cov), default=None),
                                     "converged_changes": sum(1 for x in cov if x["converged"][0] != x["converged"][1])}
    S["W7"] = {"by_reading_rtol": w7, "holds_A_ruled": all(w7[f"A|{rt}"]["violating"] == 0 for rt in ("1e-7", "1e-8")),
               "holds_B": all(w7[f"B|{rt}"]["violating"] == 0 for rt in ("1e-7", "1e-8"))}
    # ---------------- descriptive: route changes
    cs_by = Counter((x["rtol"], x["kind"], x["band"], x["k"], x["cls"], x["dirn"]) for x in changed_solves)
    S["route_changes"] = {
        "n_solves_changed": len(changed_solves),
        "by_direction_class": dict(Counter(f"{x['dirn']}|{x['cls']}" for x in changed_solves)),
        "by_rtol_kind_direction": dict(Counter(f"{x['rtol']}|{x['kind']}|{x['dirn']}" for x in changed_solves)),
        "by_band_k": dict(Counter(f"{x['band']}|{x['k']}" for x in changed_solves)),
        "by_all": {"|".join(map(str, k)): v for k, v in sorted(cs_by.items())},
        "est_over_V_by_direction": {d: q([x["ratio"] for x in changed_solves if x["dirn"] == d]) for d in sorted({x["dirn"] for x in changed_solves})},
        "V_by_direction": {d: q([x["V"] for x in changed_solves if x["dirn"] == d]) for d in sorted({x["dirn"] for x in changed_solves})},
        "calls_by_class": dict(Counter(f"{c['rtol']}|{c['kind']}|{c['class']}" for c in calls if c["class"] in ("own", "aux_only"))),
        "n_solves_route_lu_from_units": [sum(u["agg"]["n_route"] for u in U), sum(u["agg"]["n_lu"] for u in U)],
        "n_solves_total": sum(u.get("n_solves") or 0 for u in U)}
    # ---------------- descriptive: moves of totals
    mv = {"rows_total_differs": 0, "rel": [], "over_tae07": [], "by_band_k": Counter(), "converged_changes": []}
    for u in U:
        t, ta, tc = u.get("total"), u.get("total_abserr"), u.get("total_converged")
        if not t or t[0] is None:
            continue
        if t[0] != t[1] or ta[0] != ta[1]:
            mv["rows_total_differs"] += 1
            d = abs(F(t[0]) - F(t[1]))
            mv["rel"].append(float(d / abs(F(t[1]))) if F(t[1]) else None)
            mv["over_tae07"].append(float(d / F(ta[1])) if F(ta[1]) else None)
            mv["by_band_k"][f"{u['rtol']}|{u['band']}|{u['k']}"] += 1
        if tc[0] != tc[1]:
            mv["converged_changes"].append({"unit": u["unit"], "h_07": tc})
    S["moves"] = {"rows_total_or_abserr_differs": mv["rows_total_differs"], "rel": q(mv["rel"]), "over_total_abserr_07": q(mv["over_tae07"]),
                  "n_over_total_abserr_07_gt_1": sum(1 for x in mv["over_tae07"] if x and x > 1),
                  "by_rtol_band_k": dict(mv["by_band_k"]), "total_converged_changes": mv["converged_changes"]}
    cm = [c for c in calls if c["value"][0] != c["value"][1] or c["abserr"][0] != c["abserr"][1]]
    S["moves"]["calls_value_or_abserr_differs_by_class"] = dict(Counter(c["class"] for c in cm))
    S["moves"]["calls_value_move_over_abserr07_by_class"] = {
        k: q([float(abs(F(c["value"][0]) - F(c["value"][1])) / F(c["abserr"][1])) for c in cm if c["class"] == k and F(c["abserr"][1])])
        for k in sorted({c["class"] for c in cm})}
    S["moves"]["calls_converged_changes"] = [{k: c[k] for k in ("unit", "kind", "index", "class", "converged", "abserr")}
                                             for c in calls if c["converged"][0] != c["converged"][1]]
    # ---------------- descriptive: newly located, windows, convergence, depth
    S["newly_located"] = {"n_calls": len(newly), "by": dict(Counter(f"{x['rtol']}|{x['kind']}|{x['label'] if x['kind'] == 'analytic' else 'group'}" for x in newly)),
                          "points_kinds": dict(Counter(k for x in newly for k in x["kinds"])), "pieces": q([x["pieces"] for x in newly]),
                          "converged_h_07": dict(Counter(str(x["converged"]) for x in newly)),
                          "skipped_07_variation": q([H(s["variation"]) for x in newly for s in x["skipped_07"] if "variation" in s]),
                          "calls_over_64_points": gt64, "list": newly[:30]}
    S["locations"] = {"by_rtol_kind_point": {"|".join(k): v for k, v in sorted(loc_kinds.items())}, "window_pieces_per_call": q(pieces_h)}
    S["windows"] = {rt: dict(v) for rt, v in win.items()}
    S["windows"]["07_recorded"] = {"1e-7": {"stopped": 1157, "falling": 1153}, "1e-8": {"stopped": 1195, "falling": 1192}}
    S["windows"]["stopped_falling_hist"] = dict(Counter(x["falling"] for x in stopped_list))
    S["unconverged"] = {code: {"|".join(k): v for k, v in sorted(c.items())} for code, c in unconv.items()}
    S["row_depth"] = {rt: {"h": dict(Counter(u["row_depth"][0] for u in U if u["rtol"] == rt)),
                           "07": dict(Counter(u["row_depth"][1] for u in U if u["rtol"] == rt))} for rt in ("1e-7", "1e-8")}
    S["row_depth_20"] = {rt: [sum(1 for u in U if u["rtol"] == rt and u["row_depth"][j] == 20) for j in (0, 1)] for rt in ("1e-7", "1e-8")}
    # ---------------- descriptive: cost
    S["cost"] = {"unit_wall_s": {rt: [sum((u["wall_s"][j] or 0) for u in U if u["rtol"] == rt) for j in (0, 1)] for rt in ("1e-7", "1e-8")},
                 "unit_time_s": {rt: [sum(u["time_s"][j] for u in U if u["rtol"] == rt) for j in (0, 1)] for rt in ("1e-7", "1e-8")},
                 "group_regions": {rt: [sum(u["group_regions"][j] for u in U if u["rtol"] == rt) for j in (0, 1)] for rt in ("1e-7", "1e-8")},
                 "all_regions": {rt: [sum(u["all_regions"][j] for u in U if u["rtol"] == rt) for j in (0, 1)] for rt in ("1e-7", "1e-8")},
                 "census_wall_s": {"h": 23284.0, "07": 22708.0}}
    # ---------------- descriptive: section 10.4's item
    item104 = []
    cmap = {(u["unit"], c["kind"], c["index"]): c for u in U for c in u.get("calls") or []}
    for c in calls:
        if c["converged"][1] is True and c["converged"][0] is False and not c["win07"] and not c["winh"] and c["class"] in ("own", "aux_only"):
            sc = cmap.get((c["unit"], c["kind"], c["index"])) or {}
            item104.append({k: c[k] for k in ("unit", "kind", "index", "label", "class")} | {
                "abserr_h_07": c["abserr"], "abserr_h_07_repr": [repr(H(c["abserr"][0])), repr(H(c["abserr"][1]))],
                "regions_h_07": c["num_regions"],
                "changed": [{k: x.get(k) for k in ("a", "b", "role", "class", "V", "est_07", "decision")} for x in sc.get("changed") or []][:8]})
    S["item_10_4"] = {"n": len(item104), "list": item104}
    # ---------------- descriptive: |theta| at a gated end, margins
    thm = {}
    for u in U:
        for kind, v in u["agg"]["theta_max"].items():
            k = f"{u['rtol']}|{kind}"
            if k not in thm or v[0] > thm[k]["theta"]:
                thm[k] = {"theta": v[0], "ulp": ulp(v[0]), "unit": u["unit"], "index": v[1], "label": v[2]}
    S["theta_end_max"] = thm
    marg = {}
    for nm in ("m6", "m20"):
        for u in U:
            m = u["agg"][nm]
            if m is None:
                continue
            k = f"{nm}|{u['rtol']}|{m[1]}"
            if k not in marg or m[0] < marg[k]["margin"]:
                marg[k] = {"margin": m[0], "unit": u["unit"], "index": m[2], "detail": m[3]}
    S["margins"] = marg
    # ---------------- descriptive: the aux_only calls (the orchestrator's request)
    aux = [c for c in calls if c["class"] == "aux_only"]
    ratio = [float(F(c["abserr"][0]) / F(c["abserr"][1])) for c in aux if F(c["abserr"][1])]
    vmove = [float(abs(F(c["value"][0]) - F(c["value"][1])) / F(c["abserr"][1])) for c in aux if F(c["abserr"][1])]
    w4_by = {}
    for x in w4["calls"] + w4["z"]:
        w4_by[("h-" + x["unit"].split("-", 1)[1], x.get("kind", "group"), x["index"])] = x
    xp = {}
    for p in x07["pairs"]:
        for rt in ("1e-7", "1e-8"):
            xp[(f"h-{rt}-{p['row']}", p["kind"], p["index"])] = p
    ref_rows = []
    for c in aux:
        k = (c["unit"], c["kind"], c["index"])
        if k in w4_by:
            x = w4_by[k]
            ref_rows.append({"call": x["sid"], "unit": c["unit"], "ratio_h": x["ratio_g"], "ratio_07": x["score_07"]["ratio_g"],
                             "abserr_h_07": c["abserr"]})
        if k in xp:
            p = xp[k]
            ref_rows.append({"call": f"xcheck {p['row']} {p['kind']} {p['index']}", "unit": c["unit"], "r_g_h": p["r_g"],
                             "r_g_07": p["r_m"], "abserr_h_07": c["abserr"]})
    S["aux_only"] = {
        "n": len(aux), "by_rtol_kind": dict(Counter(f"{c['rtol']}|{c['kind']}" for c in aux)),
        "abserr_equal": sum(1 for c in aux if c["abserr"][0] == c["abserr"][1]),
        "abserr_smaller_h": sum(1 for x in ratio if x < 1), "abserr_larger_h": sum(1 for x in ratio if x > 1),
        "abserr_shrank_more_than_2x": sum(1 for x in ratio if x < 0.5),
        "abserr_shrank_more_than_2x_list": [{k: c[k] for k in ("unit", "kind", "index", "label", "abserr", "value", "converged")}
                                            for c in aux if F(c["abserr"][1]) and F(c["abserr"][0]) / F(c["abserr"][1]) < Fraction(1, 2)],
        "abserr_grew_more_than_2x": sum(1 for x in ratio if x > 2),
        "abserr_ratio_h_over_07": q(ratio),
        "abserr_ratio_bins": dict(Counter("<0.5" if x < 0.5 else "[0.5,1)" if x < 1 else "1" if x == 1 else "(1,2]" if x <= 2 else ">2" for x in ratio)),
        "value_equal": sum(1 for c in aux if c["value"][0] == c["value"][1]),
        "value_move_over_abserr07": q(vmove), "value_move_over_abserr07_gt_1": [
            {k: c[k] for k in ("unit", "kind", "index", "label", "value", "abserr")} for c in aux
            if F(c["abserr"][1]) and abs(F(c["value"][0]) - F(c["value"][1])) > F(c["abserr"][1])],
        "value_move_over_abserr_h_gt_1": sum(1 for c in aux if F(c["abserr"][0]) and abs(F(c["value"][0]) - F(c["value"][1])) > F(c["abserr"][0])),
        "converged_changes": [{k: c[k] for k in ("unit", "kind", "index", "converged")} for c in aux if c["converged"][0] != c["converged"][1]],
        "regions_changed": sum(1 for c in aux if c["num_regions"][0] != c["num_regions"][1]),
        "referenced": ref_rows,
        "differs_by_class": dict(Counter(k for c in aux for k, v in ((c["route"] or {}).get("differs_by_class") or {}).items() if v["route"] or v["lu"]))}
    S["disagreement"] = {"h_vs_07_pop07": json.loads(X07.read_text())["B"]["by_class"] | {"n": x07["B"]["n_listed_calls"]},
                         "h_vs_07_poph": xh["B"]["by_class"] | {"n": xh["B"]["n_listed_calls"]},
                         "h_vs_06m_pop07": {k: v for k, v in json.loads(X06.read_text()).items() if k not in ("calls", "header")}}
    S["post_unit_counterfactual"] = {"n_solves": sum((u.get("post_unit") or {}).get("n_solves") or 0 for u in U),
                                     "n_route": sum((u.get("post_unit") or {}).get("n_route_differs") or 0 for u in U),
                                     "n_lu": sum((u.get("post_unit") or {}).get("n_lu_differs") or 0 for u in U)}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(S, indent=1, default=str))
    brief = {k: S[k] for k in ("statuses", "route_errors", "no_build_solves")}
    brief["W1"] = {k: v for k, v in S["W1"].items() if k != "by_reading_rtol"} | {
        k: {kk: vv for kk, vv in v.items() if kk not in ("call_differing_list", "rows_totals_differing")} for k, v in S["W1"]["by_reading_rtol"].items()}
    brief["W2"] = {k: S["W2"][k] for k in ("n_levin_leaves_listed", "listed_by_rtol_kind_class", "holds")}
    brief["W3"] = {k: S["W3"][k] for k in ("n_calls_scored", "n_violating", "n_set_aside", "set_aside_by", "holds")}
    brief["W4"] = S["W4"]["V4"] | {"three_figures": S["W4"]["three_figures"], "bit_identical": S["W4"]["all_bit_identical_to_07"]}
    brief["W5"] = {k: S["W5"][k] for k in ("counts", "n_calls", "holds", "holds_gated_partition_only")}
    brief["W6"] = {k: S["W6"][k] for k in ("pairs", "r_g_le_1", "r_g_gt_1", "holds", "h_population_descriptive")}
    brief["W7"] = S["W7"]
    print(json.dumps(brief, indent=1, default=str))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
