"""levin-phase-floor prompt 07c section 6: Y1-Y9 gathered and scored against state/predictions.md (part 5's readings),
and the descriptive tables. New in 07c, after 07b's summarise_07b.py (a record, not edited), whose U2, U3 and U5 rules it
keeps for Y2, Y3 and Y5 (Y5 on gated solves only, 07a's ruling at its W5 stop). Read only.

Inputs: state/score_07c_units.json (score_07c.py); out/stop3/compare_census_calls.json and .calls.jsonl (compare_07c.py:
Y1, the classes, convergence, abserr, regions); out/stop3/y8.json (score_y8_07c.py); out/scores/y4_v4_scores.json
(score_v4_07c.py), out/scores/y4a_y6_scores.json (score_refs_07c.py), out/scores/y4b_scores.json
(score_refs_partA_07c.py); out/xcheck/y7_vs_07b_pop07/xcheck.json (xcheck_07c.py); out/stop3/stop3.json (cost);
state/rows.json (band, k); 07b's census's stop-3 call list (var/levin-phase-floor-07b/census/out/stop3/
compare_census_calls.calls.jsonl: 07b's class of each call by the rate rules, and 07a's value, abserr and converged, its
second column) and 07b's census summary (out/scores/summary.json: the 198 newly unconverged calls, Y9 (ii)). Floats
compared exactly (Fraction of their hex).

Usage: python -B summarise_07c.py OUT_JSON
"""

import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import REC07B, WIDER, header, sha256  # noqa: E402

UNITS = WIDER / "state" / "score_07c_units.json"
CMP = WIDER / "out" / "stop3" / "compare_census_calls.json"
CALLS = WIDER / "out" / "stop3" / "compare_census_calls.calls.jsonl"
Y8 = WIDER / "out" / "stop3" / "y8.json"
Y4V4 = WIDER / "out" / "scores" / "y4_v4_scores.json"
Y4A = WIDER / "out" / "scores" / "y4a_y6_scores.json"
Y4B = WIDER / "out" / "scores" / "y4b_scores.json"
Y7 = WIDER / "out" / "xcheck" / "y7_vs_07b_pop07" / "xcheck.json"
COST = WIDER / "out" / "stop3" / "stop3.json"
ROWS = WIDER / "state" / "rows.json"
CALLS07B = REC07B / "out" / "stop3" / "compare_census_calls.calls.jsonl"
SUMM07B = REC07B / "out" / "scores" / "summary.json"
R2_28 = {"R2-28", "R2-28~o", "c-1e-7-0-13757-5", "c-1e-8-0-13757-5"}
FIVE = {"c-1e-8-1-2229-4", "c-1e-8-3-13490-2", "c-1e-8-3-13490-3", "c-1e-8-3-13615-2", "c-1e-8-3-13615-3"}


def F(h):
    return Fraction(float.fromhex(h))


def q(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return {"n": len(xs), "min": xs[0], "p01": xs[len(xs) // 100], "p10": xs[len(xs) // 10], "median": xs[len(xs) // 2],
            "p90": xs[9 * len(xs) // 10], "p99": xs[min(len(xs) - 1, 99 * len(xs) // 100)], "max": xs[-1]}


def sid_of(unit, kind, index):
    p = unit.split("-")  # h-1e-8-1-2229
    return f"c-{p[1]}-{p[2]}-{p[3]}-{p[4]}-{index}" if kind == "group" else f"a-{p[1]}-{p[2]}-{p[3]}-{p[4]}-{index}"


def main():
    out_p = Path(sys.argv[1])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07c: Y1-Y9 and the descriptive tables",
                 [UNITS, CMP, CALLS, Y8, Y4V4, Y4A, Y4B, Y7, COST, ROWS, CALLS07B, SUMM07B])
    U = json.loads(UNITS.read_text())["units"]
    cmpd = json.loads(CMP.read_text())
    rows = {r["name"]: r for r in json.loads(ROWS.read_text())["rows"]}
    calls = [json.loads(l) for l in open(CALLS)]
    keyset = {(c["unit"], c["kind"], c["index"]) for c in calls}
    b7 = {}
    for l in open(CALLS07B):
        c = json.loads(l)
        k = (c["unit"], c["kind"], c["index"])
        if k in keyset:
            b7[k] = {"class_07b": c["class"], "value_07a": c["value"][1], "abserr_07a": c["abserr"][1],
                     "converged_07a": c["converged"][1]}
    uinfo = {u["unit"]: u for u in U}
    S = {"header": hdr, "statuses": dict(Counter(u["status"] for u in U)), "n_calls": len(calls),
         "route_errors": sum(u["agg"]["route_errors"] for u in U if "agg" in u),
         "scan_07_errors": sum(u.get("scan_07_error") or 0 for u in U), "n_calls_matched_07b_list": len(b7)}
    # ---------------- Y1
    y1 = {}
    for rt in ("1e-7", "1e-8"):
        cs = [c for c in calls if c["rtol"] == rt and c["class"] == "untouched"]
        us = [u for u in cmpd["units"] if u["rtol"] == rt and (u.get("all_w1") or {}).get("A")]
        y1[rt] = {"untouched_calls": len(cs), "by_kind": dict(Counter(c["kind"] for c in cs)),
                  "touched_calls": sum(1 for c in calls if c["rtol"] == rt and c["class"] == "acted"),
                  "five_equal": sum(1 for c in cs if c["u1_equal"]), "five_differing": [
                      {k: c[k] for k in ("unit", "kind", "index", "u1_differing")} for c in cs if not c["u1_equal"]][:50],
                  "whole_record_equal": sum(1 for c in cs if c["equal"]),
                  "rows_all_untouched": len(us), "rows_totals_equal": sum(1 for u in us if u["totals_equal"]),
                  "rows_whole_categories": dict(Counter("+".join((u.get("whole") or {}).get("categories") or []) or "none" for u in us))}
    S["Y1"] = {"by_rtol": y1, "no_counts_calls": sum(1 for c in calls if c["class"] == "no_counts"),
               "holds": all(y1[rt]["five_equal"] == y1[rt]["untouched_calls"] and y1[rt]["rows_totals_equal"] == y1[rt]["rows_all_untouched"]
                            for rt in y1) and not any(c["class"] == "no_counts" for c in calls),
               "fraction_untouched": sum(y1[rt]["untouched_calls"] for rt in y1) / len(calls)}
    # ---------------- Y2, Y3, Y5 (07b's U2, U3, U5 extraction, unchanged)
    V2_levin, listed_other = [], Counter()
    V3_viol, V3_set_aside, v3_scored = [], [], 0
    w5 = Counter()
    w5_calls = {"partition_nonforced": [], "cc_lh": [], "guarded": []}
    unconv = defaultdict(Counter)
    for u in U:
        a = u.get("agg")
        if a is None:
            continue
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
            for j, code in ((0, "07c"), (1, "07b")):
                if c["converged"][j] is False:
                    unconv[code][(u["rtol"], c["kind"], "720" if u["in720"] else "outside")] += 1
    S["Y2"] = {"n_levin_leaves_listed": len(V2_levin), "levin_leaves": V2_levin[:100],
               "listed_by_rtol_kind_class": {"|".join(k): v for k, v in sorted(listed_other.items())}, "holds": len(V2_levin) == 0}
    bad_skip = [x for x in V3_set_aside if any(s.get("reason") != "zero_rate" for s in x["skipped"]) or x["n_sign_changes"]]
    S["Y3"] = {"n_calls_scored": v3_scored, "n_violating": len(V3_viol), "violations": V3_viol[:100], "n_set_aside": len(V3_set_aside),
               "set_aside_by": dict(Counter(f"{x['rtol']}|{x['kind']}|{','.join(s.get('reason') or 'NO REASON' for s in x['skipped'])}|"
                                            f"sign_changes={x['n_sign_changes']}" for x in V3_set_aside)),
               "set_aside_not_zero_rate_or_with_sign_change": bad_skip[:50], "holds": len(V3_viol) == 0 and not bad_skip}
    S["Y5"] = {"counts": dict(w5), "n_calls": {k: len(v) for k, v in w5_calls.items()}, "calls": {k: v[:50] for k, v in w5_calls.items()},
               "holds_gated": w5["cc_with_levin_half"] == 0 and w5["partition_nonforced"] == 0 and w5["num_guarded_regions"] == 0,
               "forced_partition_descriptive": w5["partition_forced"]}
    # ---------------- Y4
    v4 = json.loads(Y4V4.read_text())
    ra = json.loads(Y4A.read_text())
    rb = json.loads(Y4B.read_text())
    y4rows = ([{"sid": x["sid"], "set": "(a) V4", "verdict": ("unconverged" if not x["converged_g"] else x["verdict"]),
                "ratio": x.get("ratio_g"), "converged": x["converged_g"]} for x in v4["calls"] + v4["z"]]
              + [{"sid": k, "set": "(a) " + x["set"], "verdict": x["verdict"], "ratio": x.get("ratio"), "converged": x.get("converged")}
                 for k, x in ra["u4_partA"].items()]
              + [{"sid": k, "set": "(a) 07a rows", "verdict": x["verdict"], "ratio": x["ratio"], "converged": x["converged"]}
                 for k, x in ra["u4_07a_rows"].items()]
              + [{"sid": k, "set": "(b) " + x["set"], "verdict": x["verdict"], "ratio": x["ratio"], "converged": x["converged"]}
                 for k, x in rb["calls"].items()])
    bad = [x for x in y4rows if x["verdict"] in ("NOT honest", "converged and NOT honest")]
    S["Y4"] = {"n_scorings": len(y4rows), "n_a": sum(1 for x in y4rows if x["set"].startswith("(a)")),
               "n_b": sum(1 for x in y4rows if x["set"].startswith("(b)")),
               "by_verdict": dict(Counter(x["verdict"] for x in y4rows)),
               "by_set_verdict": dict(Counter(f"{x['set']}|{x['verdict']}" for x in y4rows)),
               "converged_dishonest": bad, "converged_dishonest_other_than_r2_28": [x for x in bad if x["sid"] not in R2_28],
               "unconverged": {x["sid"]: x["ratio"] for x in y4rows if x["verdict"] == "unconverged"},
               "not_scored": sorted(x["sid"] for x in y4rows if x["verdict"] in ("reported, not scored", "no reference")),
               "max_ratio_honest": max((x["ratio"] for x in y4rows if x["verdict"] == "honest"), default=None),
               "r2_28": {x["sid"]: x["ratio"] for x in y4rows if x["sid"] in R2_28},
               "holds": not [x for x in bad if x["sid"] not in R2_28]}
    # ---------------- Y6, Y7, Y8
    S["Y6"] = {"by_verdict": ra["summary"]["u6"], "ratios": {k: x["ratio"] for k, x in ra["u6"].items()},
               "ratio_min_max": [min(x["ratio"] for x in ra["u6"].values()), max(x["ratio"] for x in ra["u6"].values())],
               "n_converged": sum(1 for x in ra["u6"].values() if x["converged"]),
               "holds": all(v["verdict"] in ("honest", "unconverged") for v in ra["u6"].values()) and len(ra["u6"]) == 24}
    x7 = json.loads(Y7.read_text())
    pairs = x7["pairs"]
    gt = [p for p in pairs if not p["g_self"]]
    S["Y7"] = {"pairs": len(pairs), "unpaired": len(x7["A"]["unpaired"]), "r_g_le_1": x7["A"]["overall"]["r_g<=1"],
               "r_g_gt_1": [{k: p[k] for k in ("row", "kind", "index", "r_g", "converged")} for p in gt],
               "r_g_gt_1_without_unconverged_member": [p["row"] for p in gt if p["converged"]["g7"] and p["converged"]["g8"]],
               "max_r_g": max(p["r_g"] for p in pairs if p["r_g"] is not None), "r_g_07b_le_1": x7["A"]["overall"]["r_m<=1"],
               "the_two_pairs": {p["row"]: {"r_g": p["r_g"], "r_g_07b": p["r_m"]} for p in pairs
                                 if p["row"] in ("3-14501", "3-35006") and p["kind"] == "group" and p["index"] == 1},
               "pairs_with_unconverged_member": sum(1 for p in pairs if not (p["converged"]["g7"] and p["converged"]["g8"])),
               "checkpoints_unchanged": x7["checkpoints_unchanged"]}
    S["Y7"]["holds"] = (len(S["Y7"]["r_g_gt_1_without_unconverged_member"]) == 0 and S["Y7"]["unpaired"] == 0
                        and S["Y7"]["pairs"] == 2484)
    y8 = json.loads(Y8.read_text())
    S["Y8"] = {"n": y8["n_calls"], "n_equal": y8["n_equal"], "n_distinct": y8["n_distinct_census_calls"],
               "n_acted": y8["n_acted"], "holds": y8["y8_holds"]}
    # ---------------- Y9
    d20 = {(u["unit"], x["kind"], x["index"]): x for u in U for x in u.get("depth20") or []}
    cmap = {(c["unit"], c["kind"], c["index"]): c for c in calls}
    unconv_calls = [c for c in calls if c["converged"][0] is False]
    chase = []
    no_d20 = []
    for c in unconv_calls:
        x = d20.get((c["unit"], c["kind"], c["index"]))
        if x is None:
            raise SystemExit(f"no depth-20 record for unconverged call {c['unit']} {c['kind']} #{c['index']}")
        if x["n_depth20"] > 0 and x["descends_from_refused"] == x["n_depth20"]:
            chase.append({k: x[k] for k in ("kind", "index", "label", "n_depth20", "routes", "refused_ancestor_depths")} | {"unit": c["unit"]})
        if x["n_depth20"] == 0:
            no_d20.append(c)
    s7b = json.loads(SUMM07B.read_text())["newly_unconverged"]["list"]
    still = []
    for e in s7b:
        c = cmap[(e["unit"], e["kind"], e["index"])]
        if c["converged"][0] is False:
            still.append({"sid": sid_of(e["unit"], e["kind"], e["index"]), "unit": e["unit"], "kind": e["kind"],
                          "index": e["index"], "class": c["class"], "u1_equal": c["u1_equal"],
                          "num_chase_stops": c["num_chase_stops"],
                          "n_depth20": (d20.get((e["unit"], e["kind"], e["index"])) or {}).get("n_depth20")})
    ii = ({x["sid"] for x in still} == FIVE and len(still) == 5
          and all(x["num_chase_stops"] == 0 and x["class"] == "untouched" and x["u1_equal"] for x in still))
    S["Y9"] = {"i_chase_calls": chase, "i_n_chase_calls": len(chase), "i_holds": len(chase) == 0,
               "unconverged_calls": len(unconv_calls),
               "unconverged_by_rtol_kind": dict(Counter(f"{c['rtol']}|{c['kind']}" for c in unconv_calls)),
               "unconverged_with_no_depth20_leaf": len(no_d20),
               "unconverged_with_depth20_leaf_not_all_below_refused": len(unconv_calls) - len(chase) - len(no_d20),
               "ii_n_07b_newly_unconverged": len(s7b), "ii_still_unconverged": still, "ii_holds": ii,
               "holds": len(chase) == 0 and ii}
    # ---------------- descriptive: where the fix acts
    def cinfo(c):
        p = c["unit"].split("-")
        R = rows[f"{p[3]}-{p[4]}"]
        return R["band"], R["k"], (b7.get((c["unit"], c["kind"], c["index"])) or {}).get("class_07b", "?"), uinfo[c["unit"]]["in720"]
    act = Counter()
    for c in calls:
        band, k, cb, in720 = cinfo(c)
        ns, nl, nb = c["num_chase_stops"] or 0, c["num_chase_stopped_leaves"] or 0, c["num_chase_noise_bisections"] or 0
        for dim, val in (("rtol|kind", f"{c['rtol']}|{c['kind']}"), ("band", band), ("k", k), ("class_07b", cb),
                         ("window_07c", c["winh"]), ("in720", in720), ("rtol|kind|band", f"{c['rtol']}|{c['kind']}|{band}"),
                         ("rtol|kind|k", f"{c['rtol']}|{c['kind']}|{k}")):
            act[(dim, str(val), "calls")] += 1
            act[(dim, str(val), "acted")] += ns > 0
            act[(dim, str(val), "counted_calls")] += nb > 0
            act[(dim, str(val), "stops")] += ns
            act[(dim, str(val), "stopped_leaves")] += nl
            act[(dim, str(val), "noise_bisections")] += nb
    tab = defaultdict(dict)
    for (dim, val, what), n in act.items():
        tab[dim].setdefault(val, {})[what] = n
    S["where_the_fix_acts"] = tab
    S["fix_totals"] = {"calls_acted": sum(1 for c in calls if (c["num_chase_stops"] or 0) > 0),
                       "stops": sum(c["num_chase_stops"] or 0 for c in calls),
                       "stopped_leaves": sum(c["num_chase_stopped_leaves"] or 0 for c in calls),
                       "noise_bisections": sum(c["num_chase_noise_bisections"] or 0 for c in calls),
                       "calls_with_a_counted_bisection": sum(1 for c in calls if (c["num_chase_noise_bisections"] or 0) > 0)}
    lv, sb, rbn = Counter(), Counter(), Counter()
    n_uneq = n_err = n_nr = 0
    mx = None
    for u in U:
        r7 = u.get("r07c") or {}
        for nm, C in (("leaves", lv), ("share_bins", sb), ("r_bins", rbn)):
            for k_, v_ in (r7.get(nm) or {}).items():
                C[f"{u['rtol']}|{k_}"] += v_
        n_uneq += r7.get("n_counts_unequal") or 0
        n_err += r7.get("n_instrument_error") or 0
        n_nr += r7.get("n_not_returned") or 0
        if r7.get("max_share") and (mx is None or r7["max_share"][0] > mx[0]):
            mx = r7["max_share"] + [u["unit"]]
    by_route = Counter()
    by_depth = Counter()
    for k_, v_ in lv.items():
        rt, kind, route, trust, dep = k_.split("|")
        by_route[f"{rt}|{kind}|{route}|{trust}"] += v_
        by_depth[f"{route}|{dep}"] += v_
    S["stopped_leaves"] = {"by_rtol_kind_route_trust_depth": dict(lv), "by_rtol_kind_route_trust": dict(by_route),
                           "by_route_depth": dict(sorted(by_depth.items(), key=lambda kv: (kv[0].split("|")[0], int(kv[0].split("|")[1])))),
                           "share_bins": dict(sb), "r_bins": dict(rbn), "largest_share": mx,
                           "instrument_counts_unequal": n_uneq, "instrument_errors": n_err, "not_returned": n_nr,
                           "total": sum(lv.values())}
    # ---------------- descriptive: convergence against 07b and 07a
    conv = Counter((c["rtol"], c["kind"], str(c["converged"][1]), str(c["converged"][0])) for c in calls)
    S["converged_07b_to_07c"] = {"|".join(k): v for k, v in sorted(conv.items())}
    conv_a = Counter((c["rtol"], c["kind"], str((b7.get((c["unit"], c["kind"], c["index"])) or {}).get("converged_07a")),
                      str(c["converged"][0])) for c in calls)
    S["converged_07a_to_07c"] = {"|".join(k): v for k, v in sorted(conv_a.items())}
    newly = [c for c in calls if c["converged"][1] is True and c["converged"][0] is False]
    S["newly_unconverged_vs_07b"] = {"n": len(newly), "list": [
        {k: c[k] for k in ("unit", "kind", "index", "label", "num_chase_stops", "num_regions", "max_depth")}
        | {"carrier": (c.get("chase_07c") or {}).get("carrier")} for c in newly]}
    newly_a = [c for c in calls if (b7.get((c["unit"], c["kind"], c["index"])) or {}).get("converged_07a") is True
               and c["converged"][0] is False]
    S["newly_unconverged_vs_07a"] = {"n": len(newly_a), "by": dict(Counter(f"{c['rtol']}|{c['kind']}" for c in newly_a)),
                                     "list": [{k: c[k] for k in ("unit", "kind", "index", "class", "num_chase_stops", "num_regions",
                                                                 "max_depth")} for c in newly_a]}
    regained = [c for c in calls if c["converged"][1] is False and c["converged"][0] is True]
    S["newly_converged_vs_07b"] = {"n": len(regained), "by": dict(Counter(f"{c['rtol']}|{c['kind']}|{c['class']}" for c in regained))}
    S["unconverged_by_code"] = {code: {"|".join(k): v for k, v in sorted(cc.items())} for code, cc in unconv.items()}
    # ---------------- abserr, values past the sum, rows
    ratios = defaultdict(list)
    past = defaultdict(int)
    past_list = []
    past_a = defaultdict(int)
    for c in calls:
        a = "untouched" if c["class"] == "untouched" else "acted"
        k = f"{c['rtol']}|{c['kind']}|{a}"
        v, v7, e, e7 = F(c["value"][0]), F(c["value"][1]), F(c["abserr"][0]), F(c["abserr"][1])
        if e7 and e:
            ratios[k].append(float(e / e7))
        if abs(v - v7) > e + e7:
            past[k] += 1
            past_list.append({k_: c[k_] for k_ in ("unit", "kind", "index", "value", "abserr", "converged", "num_chase_stops")}
                             | {"move_over_sum": float(abs(v - v7) / (e + e7))})
        bb = b7.get((c["unit"], c["kind"], c["index"]))
        if bb and abs(v - F(bb["value_07a"])) > e + F(bb["abserr_07a"]):
            past_a[k] += 1
    S["abserr_ratio_07c_over_07b"] = {k: q(v) for k, v in sorted(ratios.items())}
    S["abserr_ratio_bins"] = {k: dict(Counter("<0.5" if x < 0.5 else "[0.5,1)" if x < 1 else "1" if x == 1 else "(1,2]" if x <= 2 else ">2"
                                              for x in v)) for k, v in sorted(ratios.items())}
    S["values_past_the_sum_vs_07b"] = {"n": len(past_list), "by": dict(past),
                                       "converged_both": sum(1 for x in past_list if x["converged"][0] and x["converged"][1]),
                                       "move_over_sum": q([x["move_over_sum"] for x in past_list]),
                                       "list": sorted(past_list, key=lambda x: -x["move_over_sum"])[:300]}
    S["values_past_the_sum_vs_07a"] = {"n": sum(past_a.values()), "by": dict(past_a)}
    mv = {"n": 0, "over_tae07b": [], "conv": []}
    for u in U:
        t, ta, tc = u.get("total"), u.get("total_abserr"), u.get("total_converged")
        if not t or t[0] is None:
            continue
        if t[0] != t[1] or ta[0] != ta[1]:
            mv["n"] += 1
            if F(ta[1]):
                mv["over_tae07b"].append(float(abs(F(t[0]) - F(t[1])) / F(ta[1])))
        if tc[0] != tc[1]:
            mv["conv"].append({"unit": u["unit"], "07c_07b": tc})
    S["row_totals"] = {"rows_differing": mv["n"], "move_over_total_abserr_07b": q(mv["over_tae07b"]),
                       "n_gt_1": sum(1 for x in mv["over_tae07b"] if x > 1),
                       "total_converged_changes": dict(Counter(f"{x['07c_07b'][1]}->{x['07c_07b'][0]}" for x in mv["conv"])),
                       "total_converged_change_units": mv["conv"]}
    S["row_depth_20"] = {rt: [sum(1 for u in U if u["rtol"] == rt and u.get("row_depth") and u["row_depth"][j] == 20) for j in (0, 1)]
                         for rt in ("1e-7", "1e-8")}
    cost = json.loads(COST.read_text())
    S["cost"] = {k: cost[k] for k in ("regions", "regions_total", "07c", "07b", "wall_against_projection")} | {
        "unit_time_s": {rt: [sum(u["time_s"][j] or 0 for u in U if u["rtol"] == rt) for j in (0, 1)] for rt in ("1e-7", "1e-8")},
        "unit_wall_s_max": [max((u["wall_s"][0] or 0) for u in U), max((u["wall_s"][1] or 0) for u in U)]}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(S, indent=1, default=str))
    brief = {k: S[k] for k in ("statuses", "n_calls", "route_errors", "scan_07_errors", "n_calls_matched_07b_list")}
    brief["Y1"] = S["Y1"]
    for k in ("Y2", "Y3"):
        brief[k] = {kk: vv for kk, vv in S[k].items() if kk not in ("levin_leaves", "violations")}
    brief["Y4"] = {kk: vv for kk, vv in S["Y4"].items()}
    brief["Y5"] = {kk: vv for kk, vv in S["Y5"].items() if kk != "calls"}
    brief["Y6"] = {kk: vv for kk, vv in S["Y6"].items() if kk != "ratios"}
    brief["Y7"] = S["Y7"]
    brief["Y8"] = S["Y8"]
    brief["Y9"] = S["Y9"]
    brief["fix_totals"] = S["fix_totals"]
    brief["where_rtol_kind"] = S["where_the_fix_acts"]["rtol|kind"]
    brief["stopped_leaves_by_route_trust"] = S["stopped_leaves"]["by_rtol_kind_route_trust"]
    brief["stopped_leaves_by_route_depth"] = S["stopped_leaves"]["by_route_depth"]
    brief["stopped_leaves_checks"] = {k: S["stopped_leaves"][k] for k in ("instrument_counts_unequal", "instrument_errors",
                                                                            "not_returned", "total", "largest_share")}
    for k in ("converged_07b_to_07c", "converged_07a_to_07c", "newly_converged_vs_07b", "unconverged_by_code", "row_depth_20"):
        brief[k] = S[k]
    brief["newly_unconverged_vs_07b_n"] = S["newly_unconverged_vs_07b"]["n"]
    brief["newly_unconverged_vs_07a"] = {k: v for k, v in S["newly_unconverged_vs_07a"].items() if k != "list"}
    brief["values_past_the_sum_vs_07b"] = {k: v for k, v in S["values_past_the_sum_vs_07b"].items() if k != "list"}
    brief["values_past_the_sum_vs_07a"] = S["values_past_the_sum_vs_07a"]
    brief["row_totals"] = {k: v for k, v in S["row_totals"].items() if k != "total_converged_change_units"}
    print(json.dumps(brief, indent=1, default=str))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
