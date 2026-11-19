"""levin-phase-floor prompt 06c section 5: the census's scoring and descriptive items. New in 06c
(adapted in its loaders and its X1 draw from 06b's analyse06b.py, a record kept unchanged in
state/scripts_06b_reference/).

Scores X2-X5, draws X1's sample and scores X1 from x1_reference.py's output, by the rules of
state/predictions.md (Part A, the prompt's text; Part B, X1's selection rule and the readings),
reading:
- 06a's run (prov.RUN06A): its 1e-7 checkpoint lines, for the scan (sign changes of theta' per
  group call, and the 720 stationary-point rows);
- 06b's run (prov.RUN06B): its checkpoint (the "before"), read only;
- this census run (RUN_DIR): its checkpoint, and its per-worker pickles for the leaves' trust-check
  numbers.

``select RUN_DIR``: X1's eligible calls and the draw; writes state/x1_selection.json (the 24
calls, each with this run's value, abserr, regions, depth and converged for the call, and 06b's
value and abserr), and prints the eligible counts by category and stratum.

``report RUN_DIR [X1_REFERENCE_JSON]``: X1 (from the reference output), X2-X5, and the descriptive
items; writes out/census_rows_06c.csv.gz (one line per unit, both runs side by side) and
out/deep_lines_06c.json (every depth_max call's line, compact).

Read-only outside state/x1_selection.json and out/. Imports nothing from either tree.
Usage: python -B analyse06c.py select RUN_DIR ; python -B analyse06c.py report RUN_DIR [REF_JSON]
"""

import collections
import csv
import gzip
import hashlib
import json
import math
import pickle
import statistics
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import RUN06A, RUN06B, WIDER, header, sha256  # noqa: E402

ROWS_JSON = WIDER / "state" / "rows.json"
SELECTION_JSON = WIDER / "state" / "x1_selection.json"
DEPTH_MAX = 20
RTOLS = ("1e-7", "1e-8")
BAND_ORDER = {"low": 0, "mid": 1, "seam": 2}
NAMED_06B = (("1e-8", "2-1704", 16), ("1e-8", "2-13964", 5), ("1e-8", "3-34987", 9))


def rowkey(uid):
    return "-".join(uid.rsplit("-", 2)[-2:])


def hx(d):
    return None if d is None else d["hex"]


def fl(d):
    return float.fromhex(d["hex"])


def read_rtol(run_dir, rtol):
    out = {}
    with open(Path(run_dir) / "checkpoint.jsonl") as f:
        for line in f:
            if f'"rtol": "{rtol}"' not in line:
                continue
            r = json.loads(line)
            if r["data"]["rtol"] == rtol:
                out[rowkey(r["unit"])] = r["data"]
    return out


def load_rec(run_dir, d):
    with open(Path(run_dir) / d["pickle"], "rb") as f:
        f.seek(d["pickle_offset"])
        blob = f.read(d["pickle_length"])
    if hashlib.sha256(blob).hexdigest() != d["pickle_record_sha256"]:
        raise RuntimeError(f"{d['unit']}: pickle record sha256 differs")
    return pickle.loads(blob)


def scan06a():
    """{rowkey: {call index: n_sign_changes}} from 06a's 1e-7 units, and the stationary-point rows."""
    scans, sp = {}, set()
    with open(RUN06A / "checkpoint.jsonl") as f:
        for line in f:
            if '"rtol": "1e-7"' not in line:
                continue
            d = json.loads(line)["data"]
            if d["rtol"] != "1e-7":
                continue
            rk = f"{d['shard']}-{d['serial']}"
            sc = {s["index"]: s["n_sign_changes"] for s in (d.get("scan") or []) if s}
            scans[rk] = sc
            if any(v > 0 for v in sc.values()):
                sp.add(rk)
    return scans, sp


def all_calls(d):
    return [("group", c) for c in d["calls"]] + [("analytic", c) for c in d["other_calls"]]


def match_calls(dg, db):
    """[(kind, cg, cb)] matched by index with equal labels, and the problems."""
    out, probs = [], []
    if len(dg["calls"]) != len(db["calls"]) or len(dg["other_calls"]) != len(db["other_calls"]):
        probs.append(f"{dg['unit']}: call counts differ ({len(dg['calls'])}/{len(dg['other_calls'])} vs "
                     f"{len(db['calls'])}/{len(db['other_calls'])})")
    bi = {c["index"]: c for c in db["calls"] + db["other_calls"]}
    for kind, cg in all_calls(dg):
        cb = bi.get(cg["index"])
        if cb is None or cb["notify_label"] != cg["notify_label"]:
            probs.append(f"{dg['unit']} call {cg['index']}: no 06b call with its label")
            out.append((kind, cg, None))
            continue
        out.append((kind, cg, cb))
    return out, probs


def moved(cg, cb):
    return (
        hx(cg["value"]) != hx(cb["value"])
        or hx(cg["abserr"]) != hx(cb["abserr"])
        or cg["num_regions"] != cb["num_regions"]
        or cg["max_depth"] != cb["max_depth"]
    )


def x2_type(cb):
    return (
        not cb["converged"]
        and cb["max_depth"] < DEPTH_MAX
        and cb["n_phase_limited_regions"] == 0
        and math.isfinite(fl(cb["abserr"]))
    )


def sha_key(uid, idx):
    return hashlib.sha256(f"{uid}:{idx}".encode()).hexdigest()


def rows_by_key():
    rows = json.loads(ROWS_JSON.read_text())["rows"]
    return {f"{r['shard']}-{r['serial']}": r for r in rows}


# ------------------------------------------------------------------------------------------ select


def round_robin(strata_items, n):
    """strata_items: [(stratum key, [calls ordered])] in stratum order; one per stratum per pass."""
    lists = [list(v) for _k, v in strata_items]
    taken = []
    while len(taken) < n and any(lists):
        for lst in lists:
            if lst and len(taken) < n:
                taken.append(lst.pop(0))
    return taken


def cmd_select(run_dir):
    run_dir = Path(run_dir)
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06c: X1's sample",
                    [ROWS_JSON, RUN06A / "checkpoint.jsonl", RUN06B / "checkpoint.jsonl", run_dir / "checkpoint.jsonl"]):
        print(l)
    by_key = rows_by_key()
    scans, _sp = scan06a()
    G = read_rtol(run_dir, "1e-7")
    B = read_rtol(RUN06B, "1e-7")
    cat1 = collections.defaultdict(list)
    cat2 = collections.defaultdict(list)
    n = collections.Counter()
    for rk, dg in G.items():
        R = by_key[rk]
        if dg["status"] != "done":
            n["unit not done"] += 1
            continue
        db = B[rk]
        bi = {c["index"]: c for c in db["calls"]}
        for cg in dg["calls"]:
            n["group calls"] += 1
            cb = bi.get(cg["index"])
            if cb is None or cb["notify_label"] != cg["notify_label"]:
                n["unmatched"] += 1
                continue
            if not moved(cg, cb):
                n["not moved"] += 1
                continue
            sc = scans.get(rk, {}).get(cg["index"])
            if sc is None or sc != 0:
                n["sign change or no scan"] += 1
                continue
            e = {
                "unit": dg["unit"],
                "index": cg["index"],
                "notify_label": cg["notify_label"],
                "row": R["name"],
                "band": R["band"],
                "k_key": R["k_key"],
                "k": R["k"],
                "census_value": hx(cg["value"]),
                "census_abserr": hx(cg["abserr"]),
                "census_num_regions": cg["num_regions"],
                "census_max_depth": cg["max_depth"],
                "census_converged": cg["converged"],
                "value_06b": hx(cb["value"]),
                "abserr_06b": hx(cb["abserr"]),
                "converged_06b": cb["converged"],
                "num_regions_06b": cb["num_regions"],
                "max_depth_06b": cb["max_depth"],
                "dir": str(run_dir),
                "order_key": sha_key(dg["unit"], cg["index"]),
            }
            if x2_type(cb):
                e["category"] = 1
                cat1[(R["band"], R["k_key"])].append(e)
                n["eligible, category 1"] += 1
            elif R["band"] in ("low", "mid"):
                e["category"] = 2
                cat2[(R["band"], R["k_key"])].append(e)
                n["eligible, category 2"] += 1
            else:
                n["eligible but seam, not category 1"] += 1

    def ordered(strata):
        keys = sorted(strata, key=lambda s: (BAND_ORDER[s[0]], float(strata[s][0]["k"])))
        return [(s, sorted(strata[s], key=lambda e: e["order_key"])) for s in keys]

    print(f"\ncounts: {dict(n)}")
    o1 = ordered(cat1)
    print("category 1 strata (band, k_key): eligible")
    for s, v in o1:
        print(f"   {s}: {len(v)}")
    n1 = min(12, sum(len(v) for _s, v in o1))
    take1 = round_robin(o1, n1)
    n2 = 24 - n1
    o2 = ordered(cat2)
    print("category 2 strata (band, k_key): eligible")
    for s, v in o2:
        print(f"   {s}: {len(v)}")
    want = {"low": (n2 + 1) // 2, "mid": n2 // 2}
    by_band = {b: [(s, v) for s, v in o2 if s[0] == b] for b in ("low", "mid")}
    avail = {b: sum(len(v) for _s, v in by_band[b]) for b in by_band}
    share = dict(want)
    for b, other in (("low", "mid"), ("mid", "low")):
        if avail[b] < share[b]:
            short = share[b] - avail[b]
            share[b] = avail[b]
            share[other] = min(avail[other], share[other] + short)
    take2 = []
    for b in ("low", "mid"):
        take2 += round_robin(by_band[b], share[b])
    shortfall = 24 - n1 - len(take2)
    sel = take1 + take2
    print(f"\ndrawn: category 1 {len(take1)} (n1 {n1}); category 2 {len(take2)} (low {share['low']}, mid {share['mid']}); shortfall {shortfall}")
    for i, e in enumerate(sel):
        print(f"   #{i + 1:>2} cat {e['category']} {e['unit']} call {e['index']} ({e['band']}, {e['k_key']}): g value {e['census_value']} abserr "
              f"{e['census_abserr']} regions {e['census_num_regions']} depth {e['census_max_depth']} conv {e['census_converged']} | 06b "
              f"{e['value_06b']} {e['abserr_06b']} conv {e['converged_06b']} regions {e['num_regions_06b']} depth {e['max_depth_06b']}")
    if SELECTION_JSON.exists():
        raise SystemExit(f"{SELECTION_JSON} exists")
    SELECTION_JSON.write_text(json.dumps(sel, indent=1))
    print(f"wrote {SELECTION_JSON} sha256 {sha256(SELECTION_JSON)}")


# ------------------------------------------------------------------------------------------ report


def q(vals, ps=(0.0, 0.1, 0.5, 0.9, 0.99, 1.0)):
    if not vals:
        return "n 0"
    v = sorted(vals)
    out = []
    for p in ps:
        i = min(len(v) - 1, int(round(p * (len(v) - 1))))
        out.append(f"p{int(p * 100)} {v[i]:.3g}")
    return f"n {len(v)}: " + ", ".join(out)


def verdict(fail, undecided):
    return "FAILS" if fail else ("UNDECIDED" if undecided else "HOLDS")


def cmd_report(run_dir, ref_json=None):
    run_dir = Path(run_dir)
    inputs = [ROWS_JSON, RUN06A / "checkpoint.jsonl", RUN06B / "checkpoint.jsonl", run_dir / "checkpoint.jsonl"]
    if ref_json:
        inputs.append(Path(ref_json))
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06c: the census's scores and descriptive items", inputs):
        print(l)
    by_key = rows_by_key()
    scans, sp = scan06a()
    print(f"\n06a's stationary-point rows: {len(sp)}")

    # ---------------------------------------------------------------- X1
    print("\n== X1 — honest where it moved (rtol 1e-7; state/x1_selection.json, out/x1_reference.json)")
    x1_fail = x1_und = False
    if ref_json:
        res = json.loads(Path(ref_json).read_text())["results"]
        left = 0
        for r in res:
            tag = f"   {r['unit']} call {r['index']} (cat {r.get('category')}, {r.get('band')}, {r.get('k_key')})"
            if r.get("error"):
                print(f"{tag}: ERROR (no reference): {r['error'].strip().splitlines()[-1]}")
                x1_und = True
                continue
            if not (r.get("capture_equals_census") and r.get("rerun_equals_census")):
                print(f"{tag}: capture {r.get('capture_equals_census')}, re-run {r.get('rerun_equals_census')} -- not the census call; undecided")
                x1_und = True
                continue
            eg = float.fromhex(r["abserr_g"])
            sdiff = float.fromhex(r["S_diff"]) / eg if eg > 0 else float("inf")
            if not r["agree"]:
                print(f"{tag}: DROPPED (|S9-S10| = {sdiff:.3g} abserr_g > 1/100); err_g/abserr_g {r['err_g_over_abserr_g']:.3g} "
                      f"(descriptive); err_06b/abserr_06b {r.get('err_06b_over_abserr_06b')}")
                continue
            left += 1
            if not r["holds"]:
                x1_fail = True
            print(f"{tag}: |S9-S10|/abserr_g {sdiff:.3g}; err_g/abserr_g {r['err_g_over_abserr_g']:.4g} -> {'holds' if r['holds'] else 'FAILS'}; "
                  f"err_06b/abserr_06b {r.get('err_06b_over_abserr_06b')}")
        if left == 0:
            x1_und = True
        print(f"   calls left in the rule: {left} of {len(res)}")
        print(f"   X1 {verdict(x1_fail, x1_und and not x1_fail)}")
    else:
        print("   no reference given")

    rows_csv = []
    deep_out = []
    desc = {}
    scores = {}
    unmatched_all = []
    over_wall = []
    for rtol in RTOLS:
        G = read_rtol(run_dir, rtol)
        B = read_rtol(RUN06B, rtol)
        x2 = {"n": 0, "fail": [], "und": 0}
        x3 = {"inf": 0, "inf_no_cap": [], "nan": [], "und": 0}
        x4 = {"n": 0, "viol": [], "nonfinite": 0, "und": 0}
        x5 = {"moved": 0, "moved_capped": [], "over": 0}
        conv = collections.Counter()
        depth_g = collections.Counter()
        depth_b = collections.Counter()
        depth_g_grp = collections.Counter()
        depth_b_grp = collections.Counter()
        cost = collections.Counter()
        walls_g, walls_b = [], []
        largest = collections.Counter()
        shares = collections.Counter()
        trust = collections.Counter()
        widen_ratio = []
        widen_flip = 0
        tot_move = collections.defaultdict(list)
        sp_move = {"rows": 0, "moved": 0, "over_abserr": 0, "rel": []}
        n_moved_kind = collections.Counter()
        named = {}
        for rk, dg in sorted(G.items()):
            R = by_key[rk]
            db = B[rk]
            in_sp = rk in sp
            if dg["wall_s"] is not None and dg["wall_s"] > 300:
                over_wall.append((dg["unit"], round(dg["wall_s"], 1)))
            over = dg["status"] == "over budget"
            if over:
                x5["over"] += 1
            walls_g.append(dg["wall_s"] or 0.0)
            walls_b.append(db["wall_s"] or 0.0)
            matched, probs = match_calls(dg, db)
            unmatched_all += probs
            rowdepth_g = max([c["max_depth"] for _k, c in all_calls(dg)] or [0])
            rowdepth_b = max([c["max_depth"] for _k, c in all_calls(db)] or [0])
            depth_g[rowdepth_g] += 1
            depth_b[rowdepth_b] += 1
            depth_g_grp[max([c["max_depth"] for c in dg["calls"]] or [-1])] += 1
            depth_b_grp[max([c["max_depth"] for c in db["calls"]] or [-1])] += 1
            for kind, cg, cb in matched:
                conv[(kind, "g", cg["converged"])] += 1
                cost[(kind, "regions g")] += cg["num_regions"]
                cost[(kind, "solves g")] += cg["evaluations"]
                if cb is not None:
                    conv[(kind, "06b", cb["converged"])] += 1
                    cost[(kind, "regions 06b")] += cb["num_regions"]
                    cost[(kind, "solves 06b")] += cb["evaluations"]
                # X2
                if cb is not None and x2_type(cb) and not in_sp:
                    x2["n"] += 1
                    if over:
                        x2["und"] += 1
                    elif not cg["converged"]:
                        x2["fail"].append((dg["unit"], cg["index"], cg["notify_label"], hx(cg["abserr"]), cg["max_depth"], cg["num_untrusted_regions"]))
                # X3
                ab = fl(cg["abserr"])
                if math.isnan(ab):
                    x3["nan"].append((dg["unit"], cg["index"], cg["max_depth"]))
                elif math.isinf(ab):
                    x3["inf"] += 1
                    if cg["n_at_depth_max"] == 0:
                        x3["inf_no_cap"].append((dg["unit"], cg["index"], cg["notify_label"]))
                # X5
                if cb is not None and moved(cg, cb):
                    x5["moved"] += 1
                    n_moved_kind[kind] += 1
                    if cg["max_depth"] >= DEPTH_MAX:
                        x5["moved_capped"].append((dg["unit"], cg["index"], cg["notify_label"], cg["num_regions"], cg["evaluations"],
                                                   cg["converged"], cg["num_untrusted_regions"]))
                elif cb is not None:
                    n_moved_kind[kind + " (not moved)"] += 1
                # where the budget went
                largest[(kind, cg["E_largest_share"])] += 1
                for k_, v_ in cg["E_share"].items():
                    shares[(kind, k_)] += fl(v_)
                # trust
                lv = cg["leaves"]
                for k_, v_ in lv.items():
                    trust[(kind, k_)] += v_
                b_ = cg.get("budget") or {}
                trust[(kind, "untrusted found during the call")] += b_.get("num_untrusted_found", 0)
                trust[(kind, "checks")] += b_.get("num_checks", 0)
                trust[(kind, "check passes")] += b_.get("num_check_passes", 0)
                trust[(kind, "running/returned disagree")] += int(bool(b_.get("disagree")))
                trust[(kind, "resum rejections")] += b_.get("num_resum_rejections", 0)
                trust[(kind, "untrusted returned")] += cg["num_untrusted_regions"] or 0
                for k_, v_ in (cg.get("solves_by_purpose") or {}).items():
                    trust[(kind, "solves: " + k_)] += v_
                eu = fl(cg["E_unwidened"])
                if eu > 0 and math.isfinite(eu) and math.isfinite(ab):
                    widen_ratio.append(ab / eu)
                req = max(cg["atol"], cg["rtol"] * abs(fl(cg["value"])))
                if eu <= req and not cg["converged"] and (cg["num_untrusted_regions"] or 0) == 0:
                    widen_flip += 1
                # named
                for nr, nrk, nidx in NAMED_06B:
                    if nr == rtol and nrk == rk and nidx == cg["index"]:
                        named[f"{nrk} call {nidx}"] = {"g": {k_: cg[k_] for k_ in ("notify_label", "converged", "num_regions", "max_depth", "evaluations", "num_untrusted_regions", "n_at_depth_max")}
                                                      | {"value": hx(cg["value"]), "abserr": hx(cg["abserr"]), "wall_unit": dg["wall_s"]},
                                                      "06b": {k_: cb[k_] for k_ in ("converged", "num_regions", "max_depth", "evaluations")}
                                                      | {"value": hx(cb["value"]), "abserr": hx(cb["abserr"]), "wall_unit": db["wall_s"]}}
            # X4 and the totals
            rg, rb = dg.get("result"), db.get("result")
            if rg is not None and rb is not None:
                tg, tb = fl(rg["total"]), fl(rb["total"])
                eg, eb = fl(rg["total_abserr"]), fl(rb["total_abserr"])
                conv[("rows", "g", rg["total_converged"])] += 1
                conv[("rows", "06b", rb["total_converged"])] += 1
                fin = all(math.isfinite(x) for x in (tg, tb, eg, eb))
                d_abs = abs(tg - tb) if fin else float("nan")
                if in_sp:
                    sp_move["rows"] += 1
                    if hx(rg["total"]) != hx(rb["total"]):
                        sp_move["moved"] += 1
                    if fin and eb > 0:
                        sp_move["rel"].append(d_abs / eb)
                        sp_move["over_abserr"] += d_abs > eb
                else:
                    if not (math.isfinite(eg) and math.isfinite(eb)):
                        x4["nonfinite"] += 1
                    else:
                        x4["n"] += 1
                        lhs = abs(Fraction(tg) - Fraction(tb))
                        if lhs > Fraction(eb) + Fraction(eg):
                            x4["viol"].append((dg["unit"], hx(rg["total"]), hx(rb["total"]), rg["total_abserr"]["repr"], rb["total_abserr"]["repr"],
                                               float(lhs / (Fraction(eb) + Fraction(eg)))))
                    if fin and eb > 0:
                        tot_move[(R["band"], R["k_key"])].append((d_abs / eb, d_abs / abs(tb) if tb != 0 else float("inf"), eg / eb))
            elif dg["status"] == "over budget":
                x4["und"] += 1
            rows_csv.append([dg["unit"], R["band"], R["k_key"], int(in_sp), dg["status"], f"{dg['wall_s']:.3f}", f"{db['wall_s']:.3f}",
                             hx((rg or {}).get("total")), hx((rg or {}).get("total_abserr")), (rg or {}).get("total_converged"),
                             hx((rb or {}).get("total")), hx((rb or {}).get("total_abserr")), (rb or {}).get("total_converged"),
                             rowdepth_g, rowdepth_b])
            for ln in (dg.get("deep_lines") or {}).get("lines") or []:
                deep_out.append({"unit": dg["unit"], **ln})
        scores[rtol] = (x2, x3, x4, x5)
        desc[rtol] = dict(conv=conv, depth_g=depth_g, depth_b=depth_b, depth_g_grp=depth_g_grp, depth_b_grp=depth_b_grp, cost=cost,
                          walls_g=walls_g, walls_b=walls_b, largest=largest, shares=shares, trust=trust, widen_ratio=widen_ratio,
                          widen_flip=widen_flip, tot_move=tot_move, sp_move=sp_move, n_moved_kind=n_moved_kind, named=named)
        del G, B

    print(f"\n== unmatched calls or differing call counts: {len(unmatched_all)}" + (f"; first {unmatched_all[:10]}" if unmatched_all else ""))
    print(f"== units whose wall time exceeded 300 s but which the dispatcher's budget did not stop (status done): {len(over_wall)} {over_wall}")

    print("\n== X2 — budget misses converge")
    f_any = u_any = False
    for rtol in RTOLS:
        x2 = scores[rtol][0]
        f_any |= bool(x2["fail"])
        u_any |= bool(x2["und"])
        print(f"   {rtol}: 06b X2-type calls outside the 720 rows: {x2['n']}; unconverged under g: {len(x2['fail'])}; in over-budget units {x2['und']}")
        kinds = collections.Counter("untrusted at cap" if (f[5] or 0) > 0 else ("depth_max" if f[4] >= DEPTH_MAX else "budget unmet") for f in x2["fail"])
        print(f"      unconverged by cause: {dict(kinds)}; first: {x2['fail'][:6]}")
    print(f"   X2 {verdict(f_any, u_any)}")

    print("\n== X3 — no infinite declared error")
    f_any = False
    nan_any = False
    for rtol in RTOLS:
        x3 = scores[rtol][1]
        f_any |= bool(x3["inf_no_cap"])
        nan_any |= bool(x3["nan"])
        print(f"   {rtol}: calls with abserr +inf: {x3['inf']}, of them with no leaf at depth_max: {len(x3['inf_no_cap'])} {x3['inf_no_cap'][:6]}; "
              f"NaN abserr: {len(x3['nan'])} {x3['nan'][:6]}")
    print(f"   X3 {verdict(f_any, False)}" + ("" if not nan_any else " (NaN abserr present: see above)"))

    print("\n== X4 — the rows agree (outside the 720 rows; exact)")
    f_any = u_any = False
    for rtol in RTOLS:
        x4 = scores[rtol][2]
        f_any |= bool(x4["viol"])
        u_any |= bool(x4["und"])
        print(f"   {rtol}: rows scored {x4['n']}; violations {len(x4['viol'])}; rows with a non-finite total_abserr in either {x4['nonfinite']}; over budget {x4['und']}")
        for v in sorted(x4["viol"], key=lambda t: -t[5])[:10]:
            print(f"      {v}")
    print(f"   X4 {verdict(f_any, u_any)}")

    print("\n== X5 — the cap")
    f_any = False
    for rtol in RTOLS:
        x5 = scores[rtol][3]
        f_any |= bool(x5["moved_capped"]) or x5["over"] > 0
        print(f"   {rtol}: calls that moved {x5['moved']} ({dict(desc[rtol]['n_moved_kind'])}); moved calls reaching depth_max {len(x5['moved_capped'])}; "
              f"units over budget {x5['over']}")
        byk = collections.Counter("analytic" if (t[2] or "").startswith("analytic") else "group" for t in x5["moved_capped"])
        print(f"      depth_max calls by kind {dict(byk)}; converged {sum(1 for t in x5['moved_capped'] if t[5])}; "
              f"with an untrusted leaf {sum(1 for t in x5['moved_capped'] if (t[6] or 0) > 0)}")
        for t in sorted(x5["moved_capped"], key=lambda t: -t[4])[:8]:
            print(f"      {t}")
    print(f"   X5 {verdict(f_any, False)}  (with the 9-unit wall-time reading too: {len(over_wall)} units exceeded 300 s)")

    # ---------------------------------------------------------------- descriptive
    print("\n== descriptive: convergence (calls and rows), g against 06b")
    for rtol in RTOLS:
        c = desc[rtol]["conv"]
        for kind in ("group", "analytic", "rows"):
            print(f"   {rtol} {kind:<8}: g converged {c[(kind, 'g', True)]}, unconverged {c[(kind, 'g', False)]} | "
                  f"06b converged {c[(kind, '06b', True)]}, unconverged {c[(kind, '06b', False)]}")
    print("\n== descriptive: depth (row depth over every Levin call; over group calls in brackets), g against 06b")
    for rtol in RTOLS:
        d = desc[rtol]
        print(f"   {rtol} g:   {dict(sorted(d['depth_g'].items()))}")
        print(f"   {rtol} 06b: {dict(sorted(d['depth_b'].items()))}")
        print(f"   {rtol} [group] g {dict(sorted(d['depth_g_grp'].items()))} | 06b {dict(sorted(d['depth_b_grp'].items()))}")
    print("\n== descriptive: cost")
    for rtol in RTOLS:
        d = desc[rtol]
        c = d["cost"]
        for kind in ("group", "analytic"):
            print(f"   {rtol} {kind:<8}: regions g {c[(kind, 'regions g')]} vs 06b {c[(kind, 'regions 06b')]}; solves g {c[(kind, 'solves g')]} vs 06b {c[(kind, 'solves 06b')]}")
        print(f"   {rtol} unit wall: g sum {sum(d['walls_g']) / 3600:.2f} h, max {max(d['walls_g']):.1f} s, {q(d['walls_g'])} | "
              f"06b sum {sum(d['walls_b']) / 3600:.2f} h, max {max(d['walls_b']):.1f} s")
    print("\n== descriptive: where the budget went (per call, the largest share of E; and the summed shares)")
    for rtol in RTOLS:
        d = desc[rtol]
        for kind in ("group", "analytic"):
            lg = {k_[1]: v for k_, v in d["largest"].items() if k_[0] == kind}
            sh = {k_[1]: f"{v:.3g}" for k_, v in d["shares"].items() if k_[0] == kind}
            print(f"   {rtol} {kind:<8}: calls by largest share {lg}; summed shares {sh}")
    print("\n== descriptive: the trust check (leaves returned, counters, cost; r distribution below)")
    for rtol in RTOLS:
        d = desc[rtol]
        for kind in ("group", "analytic"):
            t = {k_[1]: v for k_, v in sorted(d["trust"].items()) if k_[0] == kind}
            print(f"   {rtol} {kind:<8}: {t}")
            tot = d["cost"][(kind, "solves g")]
            chk = t.get("solves: trust check: a quarter", 0) + t.get("solves: fallback half, for _check_leaf", 0)
            print(f"      solves the check made: {chk} of {tot} ({chk / tot:.3f})" if tot else "")
        print(f"   {rtol} abserr / E_unwidened (the widening's effect): {q(d['widen_ratio'])}; "
              f"calls unconverged though E_unwidened met the request (no untrusted leaf): {d['widen_flip']}")
    print("\n== descriptive: totals against 06b, outside the 720 rows, by (band, k): |dtotal|/total_abserr_06b, |dtotal|/|total_06b|, total_abserr_g/total_abserr_06b")
    for rtol in RTOLS:
        tm = desc[rtol]["tot_move"]
        for key in sorted(tm, key=lambda s: (BAND_ORDER[s[0]], s[1])):
            v = tm[key]
            a = [x[0] for x in v]
            r = [x[1] for x in v if math.isfinite(x[1])]
            e = [x[2] for x in v]
            print(f"   {rtol} {key}: rows {len(v)}; median {statistics.median(a):.3g}, max {max(a):.3g}, >1: {sum(x > 1 for x in a)}; "
                  f"rel median {statistics.median(r) if r else float('nan'):.3g} max {max(r) if r else float('nan'):.3g}; abserr ratio median {statistics.median(e):.3g}")
    print("\n== descriptive: the 720 stationary-point rows (07's; not scored)")
    for rtol in RTOLS:
        s = desc[rtol]["sp_move"]
        print(f"   {rtol}: rows {s['rows']}, total moved {s['moved']}, |dtotal| > total_abserr_06b on {s['over_abserr']}; |dtotal|/total_abserr_06b {q(s['rel'])}")
    print("\n== descriptive: 06b's three depth-20 calls, by name")
    for rtol in RTOLS:
        for k_, v in desc[rtol]["named"].items():
            print(f"   {rtol} {k_}: {json.dumps(v)}")

    # flat residuals
    print("\n== descriptive: flat residuals -- calls at depth_max, their deepest line (leftmost leaf at the greatest depth)")
    flat = []
    n_err = n_norep = 0
    for ln in deep_out:
        if "error" in ln:
            n_err += 1
            continue
        if not ln["line_end_equals_leaf_and_reproduces_its_residual_and_floor"]:
            n_norep += 1
        lev = ln["levels"]
        res = [float.fromhex(x[1]) for x in lev]
        ratios = [x[5] for x in lev[:-1]]
        last5 = res[-5:]
        # "flat": over the line's last five levels the residual falls by less than a factor 10, and
        # never below a tenth of its value five levels up (both read from the recomputed line)
        is_flat = len(last5) == 5 and last5[0] > 0 and min(last5) >= 0.1 * last5[0]
        ln["flat_last5"] = is_flat
        if is_flat:
            flat.append(ln)
    rt = collections.Counter(ln["unit"].split("-")[1] + "-" + ln["unit"].split("-")[2] for ln in deep_out)
    print(f"   lines: {len(deep_out)} ({dict(rt)}); errors {n_err}; not reproducing the leaf {n_norep}; flat over the last five levels: {len(flat)}")
    for ln in flat[:20]:
        rs = [x[5] for x in ln["levels"][:-1]]
        print(f"      {ln['unit']} call {ln['index']} {(ln['notify_label'] or '')[-40:]}: residuals last5 "
              f"{[f'{float.fromhex(x[1]):.2g}' for x in ln['levels'][-5:]]}; (eL+eR)/eP along the line "
              f"{[None if r is None else round(r, 3) for r in rs]}")
    out = WIDER / "out" / "deep_lines_06c.json"
    out.write_text(json.dumps(deep_out))
    print(f"   wrote {out} sha256 {sha256(out)}")
    out = WIDER / "out" / "census_rows_06c.csv.gz"
    with gzip.open(out, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit", "band", "k_key", "stationary_point_row", "status", "wall_g", "wall_06b", "total_g", "total_abserr_g", "total_converged_g",
                    "total_06b", "total_abserr_06b", "total_converged_06b", "row_depth_g", "row_depth_06b"])
        w.writerows(rows_csv)
    print(f"   wrote {out} sha256 {sha256(out)}")


def cmd_trust_r(run_dir):
    """The distribution of the convergence ratio r over every leaf the census returned, by rtol,
    branch (Levin, fallback) and band, read from the pickles' region arrays."""
    run_dir = Path(run_dir)
    import numpy as np

    for l in header(Path(__file__).resolve(), "levin-phase-floor 06c: the trust check's r over every returned leaf", [ROWS_JSON, run_dir / "checkpoint.jsonl"]):
        print(l)
    by_key = rows_by_key()
    labels = ["not taken", "r=0", "(0,1e-6)", "[1e-6,1e-4)", "[1e-4,1e-2)", "[1e-2,0.1)", "[0.1,0.5)", "[0.5,1)", ">=1 or inf"]
    hist = collections.defaultdict(lambda: np.zeros(len(labels), dtype=np.int64))
    trusted_ct = collections.Counter()
    n_rec = 0
    for rtol in RTOLS:
        G = read_rtol(run_dir, rtol)
        for rk, d in G.items():
            if d.get("pickle") is None:
                continue
            rec = load_rec(run_dir, d)
            n_rec += 1
            band = by_key[rk]["band"]
            for kind, arrs in (("group", rec["regions"]), ("analytic", rec["other_regions"])):
                for a in arrs:
                    r = a["convergence_ratio"]
                    t = a["type"]
                    tr = a["trusted"]
                    for branch, mask in (("Levin", t == 0), ("fallback", t == 1)):
                        rr = r[mask]
                        h = hist[(rtol, branch, band)]
                        nt = np.isnan(rr)
                        h[0] += int(nt.sum())
                        v = rr[~nt]
                        h[1] += int((v == 0).sum())
                        edges = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0]
                        for i in range(6):
                            lo, hi = edges[i], edges[i + 1]
                            h[2 + i] += int(((v > lo) & (v < hi)).sum()) if i == 0 else int(((v >= lo) & (v < hi)).sum())
                        h[8] += int((v >= 1.0).sum())
                        for code in (-1, 0, 1):
                            trusted_ct[(rtol, branch, code)] += int((tr[mask] == code).sum())
        del G
    print(f"records read {n_rec}")
    print(f"{'rtol':<5} {'branch':<9} {'band':<5} " + " ".join(f"{l:>12}" for l in labels))
    for key in sorted(hist):
        print(f"{key[0]:<5} {key[1]:<9} {key[2]:<5} " + " ".join(f"{int(x):>12}" for x in hist[key]))
    print("trusted codes (-1 not taken / floor-limited, 0 untrusted, 1 trusted):")
    for key in sorted(trusted_ct):
        print(f"   {key}: {trusted_ct[key]}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "select":
        cmd_select(sys.argv[2])
    elif cmd == "report":
        cmd_report(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == "trust_r":
        cmd_trust_r(sys.argv[2])
    else:
        raise SystemExit(__doc__)
