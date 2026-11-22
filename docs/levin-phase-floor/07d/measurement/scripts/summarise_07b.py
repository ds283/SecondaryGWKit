"""levin-phase-floor prompt 07b section 6: U1-U8 gathered and scored against state/predictions.md, and the descriptive
tables. New in 07b, after 07a's summarise_07a.py (a record, not edited), whose W2, W3 and W5 rules it keeps for U2, U3 and
U5 (U5 on gated solves only, 07a's ruling at its W5 stop). Read only.

Inputs: state/score_07b_units.json (score_07b.py); out/stop3/compare_census_calls.json and .calls.jsonl (compare_07b.py:
U1, the classes, convergence, abserr, regions); out/stop3/u8.json (score_u8_07b.py); out/scores/u4_v4_scores.json
(score_v4_07b.py) and out/scores/u4_u6_scores.json (score_refs_07b.py); out/xcheck/u7_vs_07a_pop07/xcheck.json
(xcheck_07b.py); out/stop3/cost.json; state/rows.json (band, k); and 07a's stop-3 call list
(var/levin-phase-floor-07a/census/out/stop3/compare_census_calls.calls.jsonl), whose class of each call (07a's W1 class:
untouched, aux_only, own, window_both, window_h_only, window_07_only) is "the class" of the descriptive tables, beside the
call's kind and whether it has a window at 07b. Floats compared exactly (Fraction of their hex).

Usage: python -B summarise_07b.py OUT_JSON
"""

import json
import math
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import REC07A, WIDER, header, sha256  # noqa: E402

UNITS = WIDER / "state" / "score_07b_units.json"
CMP = WIDER / "out" / "stop3" / "compare_census_calls.json"
CALLS = WIDER / "out" / "stop3" / "compare_census_calls.calls.jsonl"
U8 = WIDER / "out" / "stop3" / "u8.json"
U4V4 = WIDER / "out" / "scores" / "u4_v4_scores.json"
U4U6 = WIDER / "out" / "scores" / "u4_u6_scores.json"
U7 = WIDER / "out" / "xcheck" / "u7_vs_07a_pop07" / "xcheck.json"
COST = WIDER / "out" / "stop3" / "cost.json"
ROWS = WIDER / "state" / "rows.json"
CALLS07A = REC07A / "out" / "stop3" / "compare_census_calls.calls.jsonl"


def F(h):
    return Fraction(float.fromhex(h))


def q(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return {"n": len(xs), "min": xs[0], "p01": xs[len(xs) // 100], "p10": xs[len(xs) // 10], "median": xs[len(xs) // 2],
            "p90": xs[9 * len(xs) // 10], "p99": xs[min(len(xs) - 1, 99 * len(xs) // 100)], "max": xs[-1]}


def main():
    out_p = Path(sys.argv[1])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07b: U1-U8 and the descriptive tables",
                 [UNITS, CMP, CALLS, U8, U4V4, U4U6, U7, COST, ROWS, CALLS07A])
    U = json.loads(UNITS.read_text())["units"]
    cmpd = json.loads(CMP.read_text())
    rows = {r["name"]: r for r in json.loads(ROWS.read_text())["rows"]}
    cls07a = {}
    for l in open(CALLS07A):
        c = json.loads(l)
        cls07a[(c["unit"], c["kind"], c["index"])] = c["class"]
    calls = [json.loads(l) for l in open(CALLS)]
    uinfo = {u["unit"]: u for u in U}
    S = {"header": hdr, "statuses": dict(Counter(u["status"] for u in U)), "n_calls": len(calls),
         "route_errors": sum(u["agg"]["route_errors"] for u in U), "scan_07_errors": sum(u.get("scan_07_error") or 0 for u in U)}
    # ---------------- U1
    u1 = {}
    for rt in ("1e-7", "1e-8"):
        cs = [c for c in calls if c["rtol"] == rt and c["class"] == "untouched"]
        us = [u for u in cmpd["units"] if u["rtol"] == rt and (u.get("all_w1") or {}).get("A")]
        u1[rt] = {"untouched_calls": len(cs), "by_kind": dict(Counter(c["kind"] for c in cs)),
                  "five_equal": sum(1 for c in cs if c["u1_equal"]), "five_differing": [
                      {k: c[k] for k in ("unit", "kind", "index", "u1_differing")} for c in cs if not c["u1_equal"]][:50],
                  "whole_record_equal": sum(1 for c in cs if c["equal"]),
                  "rows_all_untouched": len(us), "rows_totals_equal": sum(1 for u in us if u["totals_equal"]),
                  "rows_whole_categories": dict(Counter("+".join((u.get("whole") or {}).get("categories") or []) or "none" for u in us))}
    S["U1"] = {"by_rtol": u1, "no_counts_calls": sum(1 for c in calls if c["class"] == "no_counts"),
               "holds": all(u1[rt]["five_equal"] == u1[rt]["untouched_calls"] and u1[rt]["rows_totals_equal"] == u1[rt]["rows_all_untouched"]
                            for rt in u1) and not any(c["class"] == "no_counts" for c in calls),
               "fraction_untouched": sum(u1[rt]["untouched_calls"] for rt in u1) / len(calls)}
    # ---------------- U2, U3, U5 (07a's W2, W3, W5 extraction)
    V2_levin, listed_other = [], Counter()
    V3_viol, V3_set_aside, v3_scored = [], [], 0
    w5 = Counter()
    w5_calls = {"partition_nonforced": [], "cc_lh": [], "guarded": []}
    unconv = defaultdict(Counter)
    for u in U:
        a = u["agg"]
        w5["partition_nonforced"] += a["partition_nonforced"]
        w5["partition_forced"] += a["partition_forced"]
        w5["cc_with_levin_half"] += a["cc_lh"]
        w5["num_guarded_regions"] += a["guarded"]
        w5["num_guard_at_depth_max"] += a["guard_at_depth_max"]
        for c in u.get("calls") or []:
            key = {"unit": u["unit"], "rtol": u["rtol"], "kind": c["kind"], "index": c["index"], "band": u["band"], "k": u["k"]}
            r = c.get("route") or {}
            if (r.get("partition") or {}).get("nonforced") and sum((r.get("partition") or {}).get("nonforced", {}).values()):
                w5_calls["partition_nonforced"].append(dict(key, partition=r.get("partition")))
            if r.get("n_cc_with_levin_half"):
                w5_calls["cc_lh"].append(dict(key, n=r["n_cc_with_levin_half"]))
            if c.get("num_guarded_regions"):
                w5_calls["guarded"].append(dict(key, n=c["num_guarded_regions"]))
            for x in c.get("listed") or []:
                cl = ("Levin" if x["branch"] == "Levin" else "window" if x["window"] == 1 else
                      "guard_at_cap" if x["guard_at_cap"] == 1 else "CC")
                if cl == "Levin":
                    V2_levin.append(dict(key, **x))
                listed_other[(u["rtol"], c["kind"], cl)] += 1
            if c.get("scanned") and "scan" in c:
                sc = c["scan"]
                if c["skipped"][0]:
                    V3_set_aside.append(dict(key, skipped=c["skipped"][0], n_sign_changes=sc["n_sign_changes"]))
                else:
                    v3_scored += 1
                    h = float.fromhex(sc["h"])
                    S_x = sc["sign_change_x"]
                    roots = [p["x"] for p in c.get("points") or [] if p["kind"] == "root"]
                    mc = [x for x in S_x if not any(abs(rr - x) <= h for rr in roots)]
                    mr = [rr for rr in roots if not any(abs(rr - x) <= h for x in S_x)]
                    if mc or mr:
                        V3_viol.append(dict(key, changes_without_root=mc[:8], roots_without_change=mr[:8]))
            for j, code in ((0, "07b"), (1, "07a")):
                if c["converged"][j] is False:
                    unconv[code][(u["rtol"], c["kind"], "720" if u["in720"] else "outside")] += 1
    S["U2"] = {"n_levin_leaves_listed": len(V2_levin), "levin_leaves": V2_levin[:100],
               "listed_by_rtol_kind_class": {"|".join(k): v for k, v in sorted(listed_other.items())}, "holds": len(V2_levin) == 0}
    bad_skip = [x for x in V3_set_aside if any(s.get("reason") != "zero_rate" for s in x["skipped"]) or x["n_sign_changes"]]
    S["U3"] = {"n_calls_scored": v3_scored, "n_violating": len(V3_viol), "violations": V3_viol[:100], "n_set_aside": len(V3_set_aside),
               "set_aside_by": dict(Counter(f"{x['rtol']}|{x['kind']}|{','.join(s.get('reason') or 'NO REASON' for s in x['skipped'])}|"
                                            f"sign_changes={x['n_sign_changes']}" for x in V3_set_aside)),
               "set_aside_not_zero_rate_or_with_sign_change": bad_skip[:50], "holds": len(V3_viol) == 0 and not bad_skip}
    S["U5"] = {"counts": dict(w5), "n_calls": {k: len(v) for k, v in w5_calls.items()}, "calls": {k: v[:50] for k, v in w5_calls.items()},
               "holds_gated": w5["cc_with_levin_half"] == 0 and w5["partition_nonforced"] == 0 and w5["num_guarded_regions"] == 0,
               "forced_partition_descriptive": w5["partition_forced"]}
    # ---------------- U4
    v4 = json.loads(U4V4.read_text())
    ru = json.loads(U4U6.read_text())
    v4rows = v4["calls"] + v4["z"]
    u4rows = ([{"sid": x["sid"], "set": "V4", "verdict": ("unconverged" if not x["converged_g"] else x["verdict"]),
                "ratio": x.get("ratio_g"), "converged": x["converged_g"]} for x in v4rows]
              + [{"sid": k, "set": x["set"], "verdict": x["verdict"], "ratio": x.get("ratio"), "converged": x.get("converged")}
                 for k, x in ru["u4_partA"].items()]
              + [{"sid": k, "set": "07a rows", "verdict": x["verdict"], "ratio": x["ratio"], "converged": x["converged"]}
                 for k, x in ru["u4_07a_rows"].items()])
    bad = [x for x in u4rows if x["verdict"] in ("NOT honest", "converged and NOT honest")]
    unc = sorted(x["sid"] for x in u4rows if x["verdict"] == "unconverged" or x["converged"] is False)
    S["U4"] = {"n_calls": len(u4rows), "by_verdict": dict(Counter(x["verdict"] for x in u4rows)),
               "by_set_verdict": dict(Counter(f"{x['set']}|{x['verdict']}" for x in u4rows)),
               "converged_dishonest": bad, "unconverged": unc,
               "unconverged_ratios": {x["sid"]: x["ratio"] for x in u4rows if x["sid"] in unc},
               "not_scored": sorted(x["sid"] for x in u4rows if x["verdict"] in ("reported, not scored", "no reference",
                                                                                 "readings disagree (reported, not scored)")),
               "max_ratio_honest": max((x["ratio"] for x in u4rows if x["verdict"] == "honest"), default=None),
               "holds": not bad,
               "prediction_unconverged_exactly_v04_R2_28": unc == sorted(["v04", "R2-28", "R2-28~o"]),
               "prediction_R2_28_ratio_3fig": {k: float(f"{v:.4g}") for k, v in ru["summary"]["u4_partA_unconverged_ratios"].items()}}
    # ---------------- U6, U7, U8
    S["U6"] = {"by_verdict": ru["summary"]["u6"], "ratios_07b_07a_06b_converged": ru["summary"]["u6_ratios"],
               "dishonest_at_07a": ru["summary"]["u6_07a_dishonest"],
               "holds": all(v["verdict"] in ("honest", "unconverged") for v in ru["u6"].values())}
    x7 = json.loads(U7.read_text())
    pairs = x7["pairs"]
    gt = [p for p in pairs if not p["g_self"]]
    S["U7"] = {"pairs": len(pairs), "unpaired": len(x7["A"]["unpaired"]), "r_g_le_1": x7["A"]["overall"]["r_g<=1"],
               "r_g_gt_1": [{k: p[k] for k in ("row", "kind", "index", "r_g", "converged")} for p in gt],
               "r_g_gt_1_without_unconverged_member": [p["row"] for p in gt if p["converged"]["g7"] and p["converged"]["g8"]],
               "dist_r_g": x7["A"]["dist"]["r_g"], "r_g_07a": x7["A"]["overall"]["r_m<=1"],
               "the_two_pairs": {p["row"]: {"r_g": p["r_g"], "r_g_07a": p["r_m"]} for p in pairs
                                 if p["row"] in ("3-14501", "3-35006") and p["kind"] == "group" and p["index"] == 1},
               "pairs_with_unconverged_member": sum(1 for p in pairs if not (p["converged"]["g7"] and p["converged"]["g8"])),
               "checkpoints_unchanged": x7["checkpoints_unchanged"], "disagreement_vs_07a": x7["B"]["by_class"] | {"n": x7["B"]["n_listed_calls"]}}
    S["U7"]["holds"] = len(S["U7"]["r_g_gt_1_without_unconverged_member"]) == 0 and S["U7"]["unpaired"] == 0
    u8 = json.loads(U8.read_text())
    S["U8"] = {"n": u8["n_calls"], "n_equal": u8["n_equal"], "n_distinct": u8["n_distinct_census_calls"], "holds": u8["u8_holds"]}
    # ---------------- descriptive: where the fix acts
    def cinfo(c):
        p = c["unit"].split("-")
        R = rows[f"{p[3]}-{p[4]}"]
        return R["band"], R["k"], cls07a.get((c["unit"], c["kind"], c["index"]), "?"), uinfo[c["unit"]]["in720"]
    act = Counter()
    refus = Counter()
    for c in calls:
        band, k, c7, in720 = cinfo(c)
        e, r = c["num_edge_rule_refusals"] or 0, c["num_range_rule_refusals"] or 0
        for dim, val in (("rtol|kind", f"{c['rtol']}|{c['kind']}"), ("band", band), ("k", k), ("class_07a", c7),
                         ("window_07b", c["winh"]), ("in720", in720), ("rtol|kind|band|k", f"{c['rtol']}|{c['kind']}|{band}|{k}")):
            act[(dim, str(val), "calls")] += 1
            act[(dim, str(val), "acted")] += (e + r) > 0
            act[(dim, str(val), "edge_calls")] += e > 0
            act[(dim, str(val), "range_calls")] += r > 0
            refus[(dim, str(val), "edge")] += e
            refus[(dim, str(val), "range")] += r
    tab = defaultdict(dict)
    for (dim, val, what), n in act.items():
        tab[dim].setdefault(val, {})[what] = n
    for (dim, val, what), n in refus.items():
        tab[dim].setdefault(val, {})[f"refusals_{what}"] = n
    S["where_the_fix_acts"] = tab
    # the instrument's aggregates
    ev, rf, qr, qp, wd = Counter(), Counter(), Counter(), Counter(), Counter()
    edge_list, n_unrec, n_uneq = [], 0, 0
    for u in U:
        r7 = u.get("r07b") or {}
        for nm, C in (("evaluated", ev), ("refused", rf), ("q_refused", qr), ("q_passed", qp), ("width_refused", wd)):
            for k_, v_ in (r7.get(nm) or {}).items():
                C[f"{u['rtol']}|{k_}"] += v_
        edge_list += [dict(e, unit=u["unit"]) for e in r7.get("edge_refusals") or []]
        n_unrec += r7.get("n_unrecorded") or 0
        n_uneq += r7.get("n_counts_unequal") or 0
    S["refused_leaves"] = {"evaluated_by_rtol_kind_rule_depth": dict(ev), "refused_by_rtol_kind_rule_depth": dict(rf),
                           "q_refused_log10_bins": dict(qr), "q_passed_log10_bins": dict(qp), "width_refused_log10": dict(wd),
                           "n_edge_refusals": len(edge_list), "edge_refusals": edge_list,
                           "edge_q": q([e["q"] for e in edge_list]), "edge_depth": dict(Counter(e["depth"] for e in edge_list)),
                           "instrument_unrecorded": n_unrec, "calls_instrument_counts_unequal_result": n_uneq,
                           "totals": {"edge": sum(c["num_edge_rule_refusals"] or 0 for c in calls),
                                      "range": sum(c["num_range_rule_refusals"] or 0 for c in calls)}}
    # ---------------- descriptive: convergence, abserr, regions, totals
    conv = Counter((c["rtol"], c["kind"], str(c["converged"][1]), str(c["converged"][0])) for c in calls)
    S["converged_07a_to_07b"] = {"|".join(k): v for k, v in sorted(conv.items())}
    d20 = {(u["unit"], x["kind"], x["index"]): x for u in U for x in u.get("depth20") or []}
    newly = [c for c in calls if c["converged"][1] is True and c["converged"][0] is False]
    regained = [c for c in calls if c["converged"][1] is False and c["converged"][0] is True]
    S["newly_unconverged"] = {"n": len(newly), "list": [
        {**{k: c[k] for k in ("unit", "kind", "index", "label", "num_edge_rule_refusals", "num_range_rule_refusals", "num_regions", "max_depth")},
         **{k: (d20.get((c["unit"], c["kind"], c["index"])) or {}).get(k) for k in
            ("n_depth20", "routes", "descends_from_refused", "routes_descending", "refused_ancestor_depths", "untrusted")},
         "abserr_over_07a": float(F(c["abserr"][0]) / F(c["abserr"][1])) if F(c["abserr"][1]) else None,
         "value_move_over_abserr_07a": float(abs(F(c["value"][0]) - F(c["value"][1])) / F(c["abserr"][1])) if F(c["abserr"][1]) else None}
        for c in newly]}
    S["newly_converged"] = {"n": len(regained), "list": [{k: c[k] for k in ("unit", "kind", "index", "label", "num_regions", "max_depth")}
                                                          for c in regained]}
    S["unconverged_by_code"] = {code: {"|".join(k): v for k, v in sorted(cc.items())} for code, cc in unconv.items()}
    ratios = defaultdict(list)
    moves = defaultdict(list)
    for c in calls:
        if F(c["abserr"][1]) and F(c["abserr"][0]):
            a = "untouched" if c["class"] == "untouched" else "acted"
            ratios[f"{c['rtol']}|{c['kind']}|{a}"].append(float(F(c["abserr"][0]) / F(c["abserr"][1])))
            moves[f"{c['rtol']}|{c['kind']}|{a}"].append(float(abs(F(c["value"][0]) - F(c["value"][1])) / F(c["abserr"][1])))
    S["abserr_ratio_07b_over_07a"] = {k: q(v) for k, v in sorted(ratios.items())}
    S["abserr_ratio_bins"] = {k: dict(Counter("<0.5" if x < 0.5 else "[0.5,1)" if x < 1 else "1" if x == 1 else "(1,2]" if x <= 2 else ">2"
                                              for x in v)) for k, v in sorted(ratios.items())}
    S["value_move_over_abserr_07a"] = {k: q(v) for k, v in sorted(moves.items())}
    S["value_move_over_abserr_07a_gt_1"] = {k: sum(1 for x in v if x > 1) for k, v in sorted(moves.items())}
    big = [c for c in calls if F(c["abserr"][1]) and abs(F(c["value"][0]) - F(c["value"][1])) > F(c["abserr"][1])]
    S["calls_moved_beyond_abserr_07a"] = {"n": len(big), "by": dict(Counter(f"{c['rtol']}|{c['kind']}" for c in big)),
                                          "beyond_both": sum(1 for c in big if abs(F(c["value"][0]) - F(c["value"][1])) > F(c["abserr"][0])),
                                          "list": [{k: c[k] for k in ("unit", "kind", "index", "value", "abserr", "converged")} for c in big][:200]}
    mv = {"n": 0, "over_tae07a": [], "conv": []}
    for u in U:
        t, ta, tc = u.get("total"), u.get("total_abserr"), u.get("total_converged")
        if not t or t[0] is None:
            continue
        if t[0] != t[1] or ta[0] != ta[1]:
            mv["n"] += 1
            if F(ta[1]):
                mv["over_tae07a"].append(float(abs(F(t[0]) - F(t[1])) / F(ta[1])))
        if tc[0] != tc[1]:
            mv["conv"].append({"unit": u["unit"], "07b_07a": tc})
    S["row_totals"] = {"rows_differing": mv["n"], "move_over_total_abserr_07a": q(mv["over_tae07a"]),
                       "n_gt_1": sum(1 for x in mv["over_tae07a"] if x > 1), "total_converged_changes": mv["conv"]}
    S["row_depth_20"] = {rt: [sum(1 for u in U if u["rtol"] == rt and u.get("row_depth") and u["row_depth"][j] == 20) for j in (0, 1)]
                         for rt in ("1e-7", "1e-8")}
    S["cost"] = json.loads(COST.read_text()) | {
        "unit_time_s": {rt: [sum(u["time_s"][j] for u in U if u["rtol"] == rt) for j in (0, 1)] for rt in ("1e-7", "1e-8")},
        "unit_wall_s_max": [max((u["wall_s"][0] or 0) for u in U), max((u["wall_s"][1] or 0) for u in U)],
        "n_solves_07b": sum(u.get("n_solves") or 0 for u in U)}
    S["cost"].pop("header", None)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(S, indent=1, default=str))
    brief = {k: S[k] for k in ("statuses", "n_calls", "route_errors", "scan_07_errors")}
    brief["U1"] = S["U1"]
    for k in ("U2", "U3"):
        brief[k] = {kk: vv for kk, vv in S[k].items() if kk not in ("levin_leaves", "violations")}
    brief["U4"] = S["U4"]
    brief["U5"] = {kk: vv for kk, vv in S["U5"].items() if kk != "calls"}
    brief["U6"] = {kk: vv for kk, vv in S["U6"].items() if kk != "ratios_07b_07a_06b_converged"}
    brief["U7"] = S["U7"]
    brief["U8"] = S["U8"]
    brief["refusals_totals"] = S["refused_leaves"]["totals"]
    brief["edge"] = {k: S["refused_leaves"][k] for k in ("n_edge_refusals", "edge_q", "edge_depth")}
    brief["converged_07a_to_07b"] = S["converged_07a_to_07b"]
    brief["newly_unconverged_n"] = S["newly_unconverged"]["n"]
    brief["newly_converged_n"] = S["newly_converged"]["n"]
    brief["row_totals"] = {k: v for k, v in S["row_totals"].items()}
    print(json.dumps(brief, indent=1, default=str))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
