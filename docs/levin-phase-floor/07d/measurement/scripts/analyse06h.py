"""levin-phase-floor prompt 06h section 5: the census's scores H3 (as written, on total; and its second
reading, on the analytic oracle), H4 (split by kind of call) and the descriptive items. New in 06h; its
loaders follow 06e's analyse06e.py (kept in state/scripts_06e_reference/), and its declared-term counts
06f's declared_vs_request_06f.py. Reads, read only: this census run's checkpoint (RUN_DIR), 06e's census
run's checkpoint (prov.RUN06E), 06a's run's 1e-7 checkpoint lines (the 720 stationary-point rows, as
analyse06e.py's scan06a_sp reads them) and state/rows.json. Every rule is state/predictions.md's (Part A
the prompt's text, Part B its readings). Writes out/census_rows_06h.csv.gz (one line per unit, both runs
side by side) and out/h4_changed_calls_06h.json (every call whose convergence changed, both kinds, inside
and outside the 720).
Usage: python -B analyse06h.py RUN_DIR"""
import collections
import csv
import gzip
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT06E_SHA256, RUN06A, RUN06E, WIDER, header, sha256  # noqa: E402

RTOLS = ("1e-7", "1e-8")
DEPTH_MAX = 20
CC_TERMS = ("t_value_rounding", "t_argument", "node_term", "declared")


def H(x):
    return None if x is None else float.fromhex(x["hex"] if isinstance(x, dict) else x)


def F(x):
    return Fraction(H(x))


def hexof(x):
    return None if x is None else (x["hex"] if isinstance(x, dict) else x)


def rowkey(uid):
    return "-".join(uid.rsplit("-", 2)[-2:])


def slim_call(kind, c, with_h):
    out = {k: c.get(k) for k in ("index", "notify_label", "group", "converged", "max_depth", "num_regions",
                                  "num_subregion_solves", "atol", "rtol", "n_at_depth_max")}
    out["kind"] = kind
    out["value"] = c["value"]["hex"]
    out["abserr"] = c["abserr"]["hex"]
    out["x_span"] = [hexof(c["x_span"][0]), hexof(c["x_span"][1])]
    f = c.get("cc_floor") or {}
    if f and "sums_floor_attributed_cc_leaves" in f:
        out["cc"] = {
            "n_cc": f["n_cc_leaves"], "n_fa": f["n_floor_attributed"],
            "fa": {t: H(f["sums_floor_attributed_cc_leaves"][t]) for t in CC_TERMS},
            "all_declared": H(f["sums_all_cc_leaves"]["declared"]),
        }
    else:
        out["cc"] = None
    if with_h:
        d = c.get("declared_06h") or {}
        if d.get("supplies_theta_abserr"):
            out["d6h"] = {
                "request": H(d["request"]),
                "n": d["n_leaves"], "n_rep": d["n_reproduced"],
                **{f"{b}_{t}": H(d[b][t]) for b in ("levin", "cc") for t in
                   ("declared", "declared_constituents", "declared_abscissa", "declared_phi_eval", "floor")},
                "levin_n": d["levin"]["n"], "cc_n": d["cc"]["n"],
            }
        else:
            out["d6h"] = None
    return out


def load(run_dir, rtol, with_h):
    """{rowkey: slim unit} at one rtol."""
    out = {}
    with open(Path(run_dir) / "checkpoint.jsonl") as f:
        for line in f:
            if f'"rtol": "{rtol}"' not in line:
                continue
            r = json.loads(line)
            d = r["data"]
            if d["rtol"] != rtol:
                continue
            res = d.get("result")
            u = {
                "unit": r["unit"], "status": d["status"], "wall_s": d["wall_s"], "row": d["row"],
                "total": None if res is None else res["total"]["hex"],
                "total_abserr": None if res is None else res["total_abserr"]["hex"],
                "total_converged": None if res is None else res["total_converged"],
                "analytic_rad": None if res is None else res["analytic_rad"]["hex"],
                "calls": [slim_call("group", c, with_h) for c in d["calls"]]
                + [slim_call("analytic", c, with_h) for c in d["other_calls"]],
            }
            if with_h:
                an = d.get("analytic_06h") or {}
                u["an_h"] = an.get("h")
                u["an_cf"] = an.get("parent_counterfactual")
                post = 0.0
                for k_ in ("scan_s", "cc_floor_s", "declared_06h_s"):
                    post += d.get(k_) or 0.0
                post += (d.get("w3") or {}).get("seconds") or 0.0
                post += (d.get("deep_lines") or {}).get("seconds") or 0.0
                post += an.get("seconds") or 0.0
                u["post_s"] = post
                u["post_parts"] = {"declared_06h_s": d.get("declared_06h_s") or 0.0, "analytic_06h_s": an.get("seconds") or 0.0,
                                   "cc_floor_s": d.get("cc_floor_s") or 0.0, "scan_s": d.get("scan_s") or 0.0,
                                   "w3_s": (d.get("w3") or {}).get("seconds") or 0.0,
                                   "deep_s": (d.get("deep_lines") or {}).get("seconds") or 0.0,
                                   "rebuild_s": d.get("rebuild_s") or 0.0}
            out[rowkey(r["unit"])] = u
    return out


def scan06a_sp():
    sp = set()
    with open(RUN06A / "checkpoint.jsonl") as f:
        for line in f:
            if '"rtol": "1e-7"' not in line:
                continue
            d = json.loads(line)["data"]
            if d["rtol"] != "1e-7":
                continue
            if any(s and s["n_sign_changes"] > 0 for s in (d.get("scan") or [])):
                sp.add(f"{d['shard']}-{d['serial']}")
    return sp


def match(uh, ue):
    """[(ch, ce)] by index, with equal labels and x_span bit for bit; problems."""
    probs = []
    if len(uh["calls"]) != len(ue["calls"]):
        probs.append(f"{uh['unit']}: call counts differ {len(uh['calls'])} vs {len(ue['calls'])}")
    bi = {c["index"]: c for c in ue["calls"]}
    out = []
    for ch in uh["calls"]:
        ce = bi.get(ch["index"])
        if ce is None or ce["notify_label"] != ch["notify_label"] or ce["x_span"] != ch["x_span"] or ce["kind"] != ch["kind"]:
            probs.append(f"{uh['unit']} call {ch['index']}: no 06e call with its label, span and kind")
            ce = None
        out.append((ch, ce))
    hi = {c["index"] for c in uh["calls"]}
    for ce in ue["calls"]:
        if ce["index"] not in hi:
            probs.append(f"{uh['unit']} call {ce['index']}: in 06e only")
            out.append((None, ce))
    return out, probs


def verdict(fail, undecided):
    return "fails" if fail else ("undecided" if undecided else "holds")


def q(vals, ps=(0.0, 0.1, 0.5, 0.9, 0.99, 1.0)):
    v = sorted(vals)
    if not v:
        return "none"
    return ", ".join(f"p{int(p*100)} {v[min(len(v)-1, int(p*(len(v)-1)))]:.3g}" for p in ps)


def main():
    run_dir = Path(sys.argv[1])
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06h: the census's scores and descriptive items",
                    [run_dir / "checkpoint.jsonl", RUN06E / "checkpoint.jsonl", WIDER / "state" / "rows.json",
                     WIDER / "state" / "predictions.md"]):
        print(l)
    assert sha256(RUN06E / "checkpoint.jsonl") == CHECKPOINT06E_SHA256
    rows = {f"{r['shard']}-{r['serial']}": r for r in json.loads((WIDER / "state" / "rows.json").read_text())["rows"]}
    sp = scan06a_sp()
    print(f"\n06a's stationary-point rows: {len(sp)} (must be 720: {len(sp) == 720})")
    assert len(sp) == 720

    csv_rows = []
    changed_calls = []
    summary = {}
    for rtol in RTOLS:
        Hh = load(run_dir, rtol, True)
        E = load(RUN06E, rtol, False)
        S = summary[rtol] = {}
        print(f"\n================ rtol {rtol}: units h {len(Hh)}, 06e {len(E)}")
        print(f"status h {dict(collections.Counter(u['status'] for u in Hh.values()))}; 06e {dict(collections.Counter(u['status'] for u in E.values()))}")
        over = [k for k, u in Hh.items() if u["status"] == "over budget"]
        over_out = [k for k in over if k not in sp]
        over300 = [(u["unit"], round(u["wall_s"], 1)) for u in Hh.values() if u["wall_s"] and u["wall_s"] > 300]
        print(f"over budget: {len(over)} ({len(over_out)} outside the 720); done with wall_s > 300 s: {over300}")
        probs = []
        h3_n = h3_nonfinite = 0
        h3_viol = []
        h3_bitident_total = h3_bitident_abserr = 0
        h3b_n = h3b_nonfinite = h3b_nocf = 0
        h3b_viol = []
        h3b_ratio = []
        h3b_moved = 0
        cf_equal = cf_differ = 0
        h4 = collections.Counter()
        h4_lost, h4_gained = [], []
        h4_und = 0
        group_bitident = collections.Counter()
        for key, uh in Hh.items():
            ue = E.get(key)
            in_sp = key in sp
            if ue is None:
                probs.append(f"{key}: no 06e unit")
                continue
            pairs, pr = match(uh, ue)
            probs += pr
            for ch, ce in pairs:
                if ch is None or ce is None:
                    if not in_sp and ce is not None and ce["converged"]:
                        h4_und += 1
                    continue
                if ch["kind"] == "group":
                    same = all(ch[k_] == ce[k_] for k_ in ("value", "abserr", "converged", "num_regions", "max_depth"))
                    group_bitident["identical" if same else "differ"] += 1
                if ch["converged"] != ce["converged"]:
                    rec = {"rtol": rtol, "unit": uh["unit"], "index": ch["index"], "kind": ch["kind"], "label": ch["notify_label"],
                           "stationary_point_row": in_sp, "06e": [ce["value"], ce["abserr"], ce["converged"], ce["num_regions"], ce["max_depth"]],
                           "h": [ch["value"], ch["abserr"], ch["converged"], ch["num_regions"], ch["max_depth"]],
                           "lost" if ce["converged"] else "gained": True}
                    changed_calls.append(rec)
                if not in_sp:
                    h4[f"{ch['kind']} calls scored"] += 1
                    if ce["converged"]:
                        h4[f"{ch['kind']} 06e converged"] += 1
                        if not ch["converged"]:
                            h4_lost.append(rec)
                    elif ch["converged"]:
                        h4_gained.append(rec)
            if in_sp:
                continue
            # H3 as written, on total
            if uh["total"] is not None and ue["total"] is not None:
                h3_bitident_total += uh["total"] == ue["total"]
                h3_bitident_abserr += uh["total_abserr"] == ue["total_abserr"]
                ah, ae = H(uh["total_abserr"]), H(ue["total_abserr"])
                if not (math.isfinite(ah) and math.isfinite(ae)):
                    h3_nonfinite += 1
                else:
                    h3_n += 1
                    lhs = abs(F(uh["total"]) - F(ue["total"]))
                    rhs = F(uh["total_abserr"]) + F(ue["total_abserr"])
                    if lhs > rhs:
                        h3_viol.append((uh["unit"], uh["total"], ue["total"], uh["total_abserr"], ue["total_abserr"]))
            # H3, the second reading, on the analytic oracle
            if uh["analytic_rad"] is not None and ue["analytic_rad"] is not None:
                cf = uh.get("an_cf") or {}
                hh = uh.get("an_h") or {}
                h3b_moved += uh["analytic_rad"] != ue["analytic_rad"]
                if "value" not in cf or cf["value"]["hex"] != ue["analytic_rad"]:
                    cf_differ += 1
                    h3b_nocf += 1
                    continue
                cf_equal += 1
                ah, ae = H(hh["abserr"]), H(cf["abserr"])
                if not (math.isfinite(ah) and math.isfinite(ae)):
                    h3b_nonfinite += 1
                    continue
                h3b_n += 1
                lhs = abs(F(hh["value"]) - F(ue["analytic_rad"]))
                rhs = F(hh["abserr"]) + F(cf["abserr"])
                if rhs > 0:
                    h3b_ratio.append(float(lhs / rhs))
                if lhs > rhs:
                    h3b_viol.append((uh["unit"], hh["value"]["hex"], ue["analytic_rad"], hh["abserr"]["hex"], cf["abserr"]["hex"],
                                     hh["converged"], cf["converged"]))
        print(f"matching problems: {len(probs)} {probs[:10]}")
        h3v = verdict(bool(h3_viol), bool(over_out))
        print(f"\n== H3 as written, on total (rtol {rtol}): rows scored {h3_n}; non-finite total_abserr in either {h3_nonfinite}; "
              f"violating {len(h3_viol)} -> {h3v}")
        print(f"   (fact 1) rows outside the 720 with total bit-identical to 06e's: {h3_bitident_total}; total_abserr bit-identical: {h3_bitident_abserr}")
        for v in h3_viol:
            print(f"   {v}")
        h3bv = verdict(bool(h3b_viol), bool(over_out) or h3b_nocf > 0)
        print(f"\n== H3, second reading, on the analytic oracle (rtol {rtol}): rows outside the 720 whose oracle moved {h3b_moved}; "
              f"06e's counterfactual equal to 06e's recorded analytic_rad bit for bit on {cf_equal} rows, not on {cf_differ}; "
              f"rows scored {h3b_n}; non-finite abserr in either {h3b_nonfinite}; violating {len(h3b_viol)} -> {h3bv}")
        print(f"   |a_h - a_06e| / (abserr_06e + abserr_h): {q(h3b_ratio)}")
        for v in h3b_viol:
            print(f"   {v[0]}: a_h {v[1]} a_06e {v[2]} abserr_h {v[3]} abserr_06e {v[4]} converged h {v[5]} 06e {v[6]}")
        print(f"   the two readings {'AGREE' if h3v == h3bv else 'DIFFER'}: {h3v} / {h3bv}")
        h4v = verdict(bool(h4_lost), h4_und > 0 or bool(over_out))
        print(f"\n== H4 (rtol {rtol}): Levin calls outside the 720 scored: analytic-block {h4['analytic calls scored']}, group {h4['group calls scored']}; "
              f"06e converged: analytic-block {h4['analytic 06e converged']}, group {h4['group 06e converged']}; "
              f"lost (06e converged, h not) {len(h4_lost)} (analytic {sum(1 for x in h4_lost if x['kind']=='analytic')}, group {sum(1 for x in h4_lost if x['kind']=='group')}); "
              f"gained {len(h4_gained)} (analytic {sum(1 for x in h4_gained if x['kind']=='analytic')}, group {sum(1 for x in h4_gained if x['kind']=='group')}); "
              f"unmatched 06e-converged {h4_und} -> {h4v}")
        for v in h4_lost + h4_gained:
            print(f"   {'LOST' if v.get('lost') else 'gained'} {v['unit']} #{v['index']} {v['kind']} {v['label']}: 06e {v['06e']} | h {v['h']}")
        print(f"   group calls (all rows) bit-identical to 06e's in value, abserr, converged, regions, depth: {dict(group_bitident)}")
        S.update({"H3": h3v, "H3_oracle": h3bv, "H4": h4v, "h3_n": h3_n, "h3_viol": len(h3_viol), "h3_nonfinite": h3_nonfinite,
                  "h3_bitident_total": h3_bitident_total, "h3b_n": h3b_n, "h3b_viol": len(h3b_viol), "h3b_nonfinite": h3b_nonfinite,
                  "h3b_nocf": h3b_nocf, "h3b_moved": h3b_moved, "h4": dict(h4), "h4_lost": len(h4_lost), "h4_gained": len(h4_gained),
                  "h4_unmatched": h4_und, "group_bitident": dict(group_bitident), "over": len(over), "match_probs": len(probs)})

        # ---- descriptive: what the declaration costs (analytic-block calls)
        print(f"\n-- descriptive (rtol {rtol}): what the declaration costs, analytic-block calls (all rows)")
        for name, U in (("06e", E), ("h", Hh)):
            st = collections.defaultdict(collections.Counter)
            for u in U.values():
                for c in u["calls"]:
                    if c["kind"] != "analytic" or not c["cc"] or c["cc"]["n_fa"] <= 0:
                        continue
                    s = c["cc"]["fa"]
                    big = max(CC_TERMS, key=lambda t: s[t])
                    req = max(c["atol"], c["rtol"] * abs(H(c["value"])))
                    for grp in (["declared-dominated"] if big == "declared" else []) + (["any declared > 0"] if s["declared"] > 0 else []):
                        k = st[grp]
                        k["calls"] += 1
                        r = s["declared"] / req
                        k["D_attr >= request"] += r >= 1
                        k["D_attr >= 0.1 request"] += r >= 0.1
                        k["D_attr >= 0.01 request"] += r >= 0.01
                        k["unconverged"] += not c["converged"]
                        k["unconverged and D_attr >= 0.1 request"] += (not c["converged"]) and r >= 0.1
            print(f"   CC branch, 06f's method ({name}): " + "; ".join(f"{g} {dict(st[g])}" for g in sorted(st)))
            S[f"cc_06f_{name}"] = {g: dict(st[g]) for g in st}
        # matched CC declared (all fallback leaves), h over 06e
        rat = []
        for key, uh in Hh.items():
            ue = E.get(key)
            if not ue:
                continue
            be = {c["index"]: c for c in ue["calls"] if c["kind"] == "analytic"}
            for c in uh["calls"]:
                if c["kind"] != "analytic" or not c["cc"]:
                    continue
                ce = be.get(c["index"])
                if ce and ce["cc"] and ce["cc"]["all_declared"] > 0 and c["cc"]["all_declared"] > 0:
                    rat.append(c["cc"]["all_declared"] / ce["cc"]["all_declared"])
        print(f"   CC branch, declared over all fallback leaves, h / 06e on matched analytic calls with both > 0 ({len(rat)}): {q(rat)}")
        # the declared term by branch under h, against the request
        lev, ccb, tot, absc, cons_cf, n_calls = [], [], [], [], [], 0
        thr = collections.Counter()
        n_unrep = 0
        for u in Hh.values():
            for c in u["calls"]:
                d = c.get("d6h")
                if c["kind"] != "analytic" or not d:
                    continue
                n_calls += 1
                n_unrep += d["n"] - d["n_rep"]
                req = d["request"]
                if not req:
                    continue
                L, C_ = d["levin_declared"], d["cc_declared"]
                T = L + C_
                lev.append(L / req)
                ccb.append(C_ / req)
                tot.append(T / req)
                for t in (1, 0.1, 0.01, 0.001):
                    thr[f"total declared >= {t} request"] += T / req >= t
                ab = d["levin_declared_abscissa"] + d["cc_declared_abscissa"]
                co = d["levin_declared_constituents"] + d["cc_declared_constituents"]
                if T > 0:
                    absc.append(ab / T)
                if co > 0:
                    cons_cf.append(T / co)
                thr["calls with a declared Levin term > 0"] += L > 0
                thr["calls with a declared CC term > 0"] += C_ > 0
                thr["unconverged"] += not c["converged"]
                thr["unconverged with total declared >= 0.1 request"] += (not c["converged"]) and T / req >= 0.1
        print(f"   h, by branch ({n_calls} analytic calls supplying theta_abserr; leaves not reproduced {n_unrep}):")
        print(f"      Levin declared / request: {q(lev)}")
        print(f"      CC declared / request: {q(ccb)}")
        print(f"      total declared / request: {q(tot)}; {dict(thr)}")
        print(f"      the abscissa part's share of the declared term: {q(absc)}")
        print(f"      declared / its constituents part (the counterfactual: what 06e's declaration would charge on h's leaves): {q(cons_cf)}")
        S["declared_h"] = {"n_calls": n_calls, "unreproduced": n_unrep, "thresholds": dict(thr),
                           "total_over_request_q": q(tot), "levin_over_request_q": q(lev), "cc_over_request_q": q(ccb),
                           "abscissa_share_q": q(absc), "over_constituents_q": q(cons_cf)}

        # ---- descriptive: convergence and depth
        def conv_counts(U):
            c = collections.Counter()
            for u in U.values():
                if u["total_converged"] is False:
                    c["rows total_converged false"] += 1
                for cl in u["calls"]:
                    c[f"{cl['kind']} calls"] += 1
                    if not cl["converged"]:
                        c[f"{cl['kind']} unconverged"] += 1
                    if cl["max_depth"] >= DEPTH_MAX:
                        c[f"{cl['kind']} at depth {DEPTH_MAX}"] += 1
                if any(not cl["converged"] for cl in u["calls"]):
                    c["rows with an unconverged call"] += 1
            return c
        print(f"\n-- descriptive (rtol {rtol}): convergence and depth, h vs 06e")
        for lab, keep in (("all rows", lambda k: True), ("outside the 720", lambda k: k not in sp), ("the 720", lambda k: k in sp)):
            ch_ = conv_counts({k: v for k, v in Hh.items() if keep(k)})
            ce_ = conv_counts({k: v for k, v in E.items() if keep(k)})
            print(f"   {lab}: " + "; ".join(f"{k} h {ch_[k]} 06e {ce_[k]}" for k in sorted(set(ch_) | set(ce_))))
            S[f"conv_{lab}"] = {k: (ch_[k], ce_[k]) for k in sorted(set(ch_) | set(ce_))}

        def row_depth(U):
            h = collections.Counter()
            for u in U.values():
                if u["calls"]:
                    h[max(c["max_depth"] for c in u["calls"])] += 1
            return h

        def call_depth(U, kind):
            h = collections.Counter()
            for u in U.values():
                for c in u["calls"]:
                    if c["kind"] == kind:
                        h[c["max_depth"]] += 1
            return h
        hg, he = row_depth(Hh), row_depth(E)
        print("   row depth (max over the row's Levin calls), depth: h / 06e")
        print("   " + "; ".join(f"{d}: {hg[d]}/{he[d]}" for d in sorted(set(hg) | set(he))))
        for kind in ("analytic", "group"):
            dg, de = call_depth(Hh, kind), call_depth(E, kind)
            print(f"   {kind} call depth, depth: h / 06e")
            print("   " + "; ".join(f"{d}: {dg[d]}/{de[d]}" for d in sorted(set(dg) | set(de))))
            S[f"call_depth_{kind}"] = {int(d): (dg[d], de[d]) for d in sorted(set(dg) | set(de))}
        S["row_depth"] = {int(d): (hg[d], he[d]) for d in sorted(set(hg) | set(he))}

        # ---- descriptive: cost
        def cost(U, kind=None):
            return (sum(c["num_regions"] or 0 for u in U.values() for c in u["calls"] if kind in (None, c["kind"])),
                    sum(c["num_subregion_solves"] or 0 for u in U.values() for c in u["calls"] if kind in (None, c["kind"])))
        print(f"\n-- descriptive (rtol {rtol}): cost")
        for kind in ("analytic", "group", None):
            rh, sh = cost(Hh, kind)
            re_, se = cost(E, kind)
            print(f"   {kind or 'all'} calls: regions h {rh} / 06e {re_} (x{rh/re_:.4f}); solves h {sh} / 06e {se} (x{sh/se:.4f})")
            S[f"cost_{kind or 'all'}"] = {"regions": (rh, re_), "solves": (sh, se)}
        wh = math.fsum(u["wall_s"] or 0.0 for u in Hh.values())
        we = math.fsum(u["wall_s"] or 0.0 for u in E.values())
        mh = max((u["wall_s"] or 0.0, u["unit"]) for u in Hh.values())
        me = max((u["wall_s"] or 0.0, u["unit"]) for u in E.values())
        post = collections.Counter()
        for u in Hh.values():
            for k_, v_ in u["post_parts"].items():
                post[k_] += v_
        print(f"   unit wall (wall_s, the timed section) summed h {wh/3600:.3f} h / 06e {we/3600:.3f} h (x{wh/we:.3f}); "
              f"largest unit h {mh[1]} {mh[0]:.1f} s / 06e {me[1]} {me[0]:.1f} s")
        print(f"   h, outside wall_s, summed (h): " + ", ".join(f"{k_} {v_/3600:.3f}" for k_, v_ in sorted(post.items())))
        S["wall_h"] = (wh / 3600, we / 3600)
        S["post_h"] = {k_: v_ / 3600 for k_, v_ in post.items()}

        # ---- descriptive: the stationary-point rows
        moved_t = moved_a = beyond_a = 0
        lost_sp = sum(1 for x in changed_calls if x["rtol"] == rtol and x["stationary_point_row"] and x.get("lost"))
        gained_sp = sum(1 for x in changed_calls if x["rtol"] == rtol and x["stationary_point_row"] and x.get("gained"))
        rel = []
        for key in sp:
            uh, ue = Hh.get(key), E.get(key)
            if not uh or not ue or uh["total"] is None or ue["total"] is None:
                continue
            moved_t += uh["total"] != ue["total"]
            if uh["analytic_rad"] != ue["analytic_rad"]:
                moved_a += 1
            cf = uh.get("an_cf") or {}
            if "abserr" in cf and cf["value"]["hex"] == ue["analytic_rad"]:
                den = H(cf["abserr"]) + H(uh["an_h"]["abserr"])
                if math.isfinite(den) and den > 0:
                    r_ = abs(H(uh["analytic_rad"]) - H(ue["analytic_rad"])) / den
                    rel.append(r_)
                    beyond_a += r_ > 1
        print(f"\n-- descriptive (rtol {rtol}): the 720: totals moved {moved_t}; analytic oracle moved {moved_a}, beyond the "
              f"second reading's bound {beyond_a}; |da|/(e_06e + e_h) {q(rel)}; calls lost convergence {lost_sp}, gained {gained_sp}")
        S["sp"] = {"totals_moved": moved_t, "oracle_moved": moved_a, "oracle_beyond": beyond_a, "lost": lost_sp, "gained": gained_sp}

        for key, uh in Hh.items():
            ue = E.get(key) or {}
            csv_rows.append([uh["unit"], rows[key]["band"], rows[key]["k_key"], key in sp, uh["status"], uh["wall_s"], ue.get("wall_s"),
                             uh["total"], uh["total_abserr"], uh["total_converged"], ue.get("total"), ue.get("total_abserr"), ue.get("total_converged"),
                             uh["analytic_rad"], (uh.get("an_h") or {}).get("abserr", {}).get("hex"), (uh.get("an_h") or {}).get("converged"),
                             ue.get("analytic_rad"), ((uh.get("an_cf") or {}).get("abserr") or {}).get("hex"), (uh.get("an_cf") or {}).get("converged"),
                             max((c["max_depth"] for c in uh["calls"]), default=None), max((c["max_depth"] for c in ue.get("calls", [])), default=None),
                             uh["post_s"]])
        del Hh, E

    with gzip.open(WIDER / "out" / "census_rows_06h.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit", "band", "k_key", "stationary_point_row", "status", "wall_h", "wall_06e", "total_h", "total_abserr_h",
                    "total_converged_h", "total_06e", "total_abserr_06e", "total_converged_06e", "analytic_rad_h", "analytic_abserr_h",
                    "analytic_converged_h", "analytic_rad_06e", "analytic_abserr_06e_counterfactual", "analytic_converged_06e_counterfactual",
                    "row_depth_h", "row_depth_06e", "post_timed_s_h"])
        w.writerows(csv_rows)
    (WIDER / "out" / "h4_changed_calls_06h.json").write_text(json.dumps(changed_calls, indent=0))
    print("\n== summary")
    print(json.dumps(summary, indent=1, default=str))
    print(f"wrote out/census_rows_06h.csv.gz ({len(csv_rows)} units) and out/h4_changed_calls_06h.json ({len(changed_calls)} calls)")


if __name__ == "__main__":
    main()
