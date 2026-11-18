"""Orchestrator's own re-score of U2, U3, U5 and U7, with descriptive cross-checks (levin-phase-floor 07b Part C,
§C.2). Read only but for OUT_JSON. Adapted from 07a's review/final/scripts/w_rescore.py (W2, W3, W5, W6); independent
of the census agent's scripts.

usage: python -I -B u_rescore.py OUT_JSON

Checkpoints read at my own indices, file SHA-256s asserted, each line accepted only if its SHA-256 matches; region
records read only where needed, accepted only against pickle_record_sha256:
  07b's census (h): review/stop3/index07b.json (396784bc...); 07a's census: review/stop3/index07a.json (461df2ec...);
  07's census (U7's population only): 07a's review/stop3/index07.json (03dfe786...).
- U2 (on h): a returned Levin leaf (type 0) holding a scan sign change, by the scan's grid test recomputed from the
  pickle's full scan (sign_change_x +- halfwidth inside the leaf), exact; my grid set compared with the scan's own; the
  scan's 'ends' listings of Levin leaves counted apart.
- U3 (on h): on every scanned call, each sign change has a located root within h and each located root lies within h
  of a sign change (exact). Every location_skipped_phases entry at h is listed; a reason other than 'zero_rate' is a
  violation; scanned calls with a skipped phase and a sign change listed.
- U5 (on h, gated solves): sum of route_07a.n_cc_with_levin_half; sum of route_07a.n_partition_nonforced; sum of
  guard_07.num_guarded_regions. Forced partitions (route_07a.partition.forced) descriptive.
- U7: population = 07's calls with guard_07.locates or .guards (either rtol), paired across rtols by row, kind, index;
  r_g = |v_7 - v_8| / e_7 at h, exact; also at 07a. Holds iff every pair r_g <= 1 or a member unconverged.
- Descriptive: class (untouched / range / edge) by rtol and kind; converged before->after by class and rtol; the newly
  unconverged calls; abserr ratio h/07a by class (quantiles); regions and wall; over-budget units.
"""
import hashlib, json, pickle, sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
RB = MAIN / "var/runs/levin-phase-floor-07b-census-20261117T010200"
RA = MAIN / "var/runs/levin-phase-floor-07a-census-20261112T132744"
R07 = MAIN / "var/runs/levin-phase-floor-07-census-20261108T171137"
IXB = MAIN / "var/levin-phase-floor-07b/review/stop3"
IX07 = MAIN / "var/levin-phase-floor-07a/review/stop3"


def fh(v):
    return Fraction(float.fromhex(v["hex"] if isinstance(v, dict) else float(v).hex()))


def readline(f, e):
    f.seek(e[0]); raw = f.read(e[1])
    assert hashlib.sha256(raw).hexdigest() == e[2]
    return json.loads(raw)["data"]


def klass(c):
    e, r = c["num_edge_rule_refusals"], c["num_range_rule_refusals"]
    assert isinstance(e, int) and isinstance(r, int)
    return "untouched" if e + r == 0 else ("edge" if e else "range")


def q(xs):
    xs = sorted(xs)
    if not xs:
        return None
    return {p: xs[min(len(xs) - 1, int(p * len(xs)))] for p in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99)} | {"max": xs[-1], "n": len(xs)}


def main():
    out = Path(sys.argv[1])
    ib = json.load(open(IXB / "index07b.json")); ia = json.load(open(IXB / "index07a.json"))
    i07 = json.load(open(IX07 / "index07.json"))
    assert ib["sha256"] == "396784bcf2dff82267f37caf3b547c786aaac86e7abacd37afab72596cd88253"
    assert ia["sha256"] == "461df2ec66969a113e9f6d590f05a4d5b300d7c1e0cb62d05c7385d23e8de56b"
    assert i07["sha256"] == "03dfe786f69343a0e86f436936161ace728630f26d687061e186d0883dd5cb10"
    assert set(ib["units"]) == set(ia["units"]) == set(i07["units"])
    C = Counter(); cls_rt = Counter(); conv = Counter(); newly_unconv = []; newly_conv = []
    u2_viol, u2_ends_levin, u2_grid_mismatch, u2_nonlevin = [], [], [], Counter()
    u3_viol, u3_skipped, u3_bad_reason, u3_skip_sc, u3_end_pts = [], Counter(), [], [], []
    u5 = Counter(); u5_calls = defaultdict(list); forced_calls = 0
    pop = {}
    over_budget = []
    cost = Counter(); eratio = defaultdict(list)
    fb = open(RB / "checkpoint.jsonl", "rb"); fa = open(RA / "checkpoint.jsonl", "rb"); f7 = open(R07 / "checkpoint.jsonl", "rb")
    for key in sorted(ib["units"]):
        dh, da, d7 = readline(fb, ib["units"][key]), readline(fa, ia["units"][key]), readline(f7, i07["units"][key])
        rt, row = dh["rtol"], dh["row"]
        assert da["row"] == row == d7["row"] and da["rtol"] == rt == d7["rtol"]
        C["units"] += 1
        C["units_done"] += dh["status"] == "done"
        if dh.get("wall_s") is not None and dh.get("budget_s") is not None and dh["wall_s"] > dh["budget_s"]:
            over_budget.append((key, dh["wall_s"]))
        cost[f"wall_h_{rt}"] += dh.get("wall_s") or 0; cost[f"wall_07a_{rt}"] += da.get("wall_s") or 0
        cas = {(k, c["index"]): c for k in ("calls", "other_calls") for c in da[k]}
        c7s = {(k, c["index"]): c for k in ("calls", "other_calls") for c in d7[k]}
        need_pickle = False
        for kind in ("calls", "other_calls"):
            for ch in dh[kind]:
                i = ch["index"]; ca = cas[(kind, i)]; c7 = c7s[(kind, i)]
                assert ch["notify_label"] == ca["notify_label"] == c7["notify_label"]
                k = klass(ch)
                kd = "group" if kind == "calls" else "analytic"
                cls_rt[(k, rt, kd)] += 1; C["calls"] += 1
                cost[f"regions_h_{rt}_{kd}"] += ch["num_regions"]; cost[f"regions_07a_{rt}_{kd}"] += ca["num_regions"]
                conv[(k, rt, bool(ca["converged"]), bool(ch["converged"]))] += 1
                if ca["converged"] and not ch["converged"]:
                    newly_unconv.append(dict(key=key, kind=kd, index=i, label=ch["notify_label"], cls=k,
                                             edge=ch["num_edge_rule_refusals"], range=ch["num_range_rule_refusals"],
                                             regions=[ca["num_regions"], ch["num_regions"]],
                                             max_depth=[ca["max_depth"], ch["max_depth"]],
                                             abserr_ratio=float(fh(ch["abserr"]) / fh(ca["abserr"])) if fh(ca["abserr"]) else None))
                if ch["converged"] and not ca["converged"]:
                    newly_conv.append((key, kd, i, k))
                ea = fh(ca["abserr"])
                if ea > 0:
                    eratio[(k, kd)].append(float(fh(ch["abserr"]) / ea))
                gh, ra = ch["guard_07"], ch["route_07a"]
                a = ra["n_cc_with_levin_half"]; b = ra["n_partition_nonforced"]; fo = ra["partition"]["forced"]
                g = gh["num_guarded_regions"]
                u5["cc_with_levin_half"] += a; u5["partition_nonforced"] += b; u5["partition_forced"] += fo
                u5["num_guarded_regions"] += g
                forced_calls += fo > 0
                for nm, v in (("cc_with_levin_half", a), ("partition_nonforced", b), ("num_guarded_regions", g)):
                    if v:
                        u5_calls[nm].append((key, kind, i, v))
                g7 = c7["guard_07"]
                if g7["locates"] or g7["guards"]:
                    pop.setdefault((row, kind, i), {})[rt] = (ch, ca)
                for e in gh["location_skipped_phases"]:
                    u3_skipped[e.get("reason", "<no reason>")] += 1
                    if e.get("reason") != "zero_rate":
                        u3_bad_reason.append((key, kind, i, e))
        callsh = {c["index"]: c for c in dh["calls"] + dh["other_calls"]}
        for s in dh["scan_07"] or []:
            C[f"scanned_{s['kind']}_{rt}"] += 1
            x = callsh[s["index"]]; g = x["guard_07"]
            sc = [fh(v) for v in s["sign_change_x"]]
            assert s["n_sign_changes"] == len(sc)
            if sc:
                need_pickle = True
            hh = fh(s["h"])
            if g["location_skipped_phases"] and sc:
                u3_skip_sc.append((key, s["index"], s["kind"], len(sc)))
            roots = [fh(p["x"]) for p in g["points"] if p["kind"] == "root"]
            if any(p["kind"] == "end" for p in g["points"]):
                u3_end_pts.append((key, s["index"]))
            bad = [float(z) for z in sc if not any(abs(z - r) <= hh for r in roots)]
            bad += [float(r) for r in roots if not any(abs(z - r) <= hh for z in sc)]
            C["u3_scored"] += 1
            if sc or roots:
                C["u3_with_change_or_root"] += 1
            if bad:
                u3_viol.append((key, s["index"], s["kind"], bad[:4]))
        if not need_pickle:
            continue
        with open(RB / dh["pickle"], "rb") as f:
            f.seek(dh["pickle_offset"]); raw = f.read(dh["pickle_length"])
        assert hashlib.sha256(raw).hexdigest() == dh["pickle_record_sha256"], key
        p = pickle.loads(raw); C["pickles_read"] += 1
        ra_ = {r["index"]: r for r in p["regions"] + p["other_regions"]}
        sfull = {e["index"]: e["scan"] for e in p["scan_07"]}
        for s in dh["scan_07"]:
            if not s["n_sign_changes"]:
                continue
            i = s["index"]; r = ra_[i]; sf = sfull[i]
            lo, hi = r["start"].astype(float), r["end"].astype(float)
            mine = set()
            for xc, hw in zip(sf["sign_change_x"], sf["sign_change_halfwidth"]):
                a, b = Fraction(xc) - Fraction(hw), Fraction(xc) + Fraction(hw)
                for kk in range(len(lo)):
                    l, e = sorted((Fraction(float(lo[kk])), Fraction(float(hi[kk]))))
                    if l <= a and b <= e:
                        mine.add(kk)
            sg = {x["region_index"] for x in sf["listed"] if x["how"] == "grid"}
            if mine != sg:
                u2_grid_mismatch.append((key, i, sorted(mine), sorted(sg)))
            for kk in mine:
                if int(r["type"][kk]) == 0:
                    u2_viol.append((key, i, kk))
                else:
                    u2_nonlevin[("window" if int(r["window_07"][kk]) else "", "guarded" if int(r["guarded_07"][kk]) else "")] += 1
            for x in sf["listed"]:
                if x["how"] == "ends" and x["branch"] == "Levin":
                    u2_ends_levin.append((key, i, x["region_index"]))
    u7 = []; unpaired = []
    for (row, kind, i), d in sorted(pop.items()):
        if set(d) != {"1e-7", "1e-8"}:
            unpaired.append((row, kind, i)); continue
        (h7, a7), (h8, a8) = d["1e-7"], d["1e-8"]
        assert json.dumps(h7["x_span"]) == json.dumps(h8["x_span"])
        rg_h = abs(fh(h7["value"]) - fh(h8["value"])) / fh(h7["abserr"])
        rg_a = abs(fh(a7["value"]) - fh(a8["value"])) / fh(a7["abserr"])
        u7.append((row, kind, i, float(rg_h), float(rg_a), h7["converged"], h8["converged"], klass(h7), klass(h8)))
    u7_fail = [x for x in u7 if x[3] > 1 and x[5] and x[6]]
    res = dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        counts=dict(C), class_by_rtol_kind={" ".join(k): n for k, n in sorted(cls_rt.items())},
        converged_07a_to_h={f"{k} {r} {a}->{b}": n for (k, r, a, b), n in sorted(conv.items())},
        newly_unconverged=dict(n=len(newly_unconv), by_class_kind_rtol={str(k): v for k, v in Counter(
            (x["cls"], x["kind"], x["key"].split("@")[1]) for x in newly_unconv).items()}, calls=newly_unconv),
        newly_converged=newly_conv,
        U2=dict(n=len(u2_viol), violations=u2_viol[:50], ends_levin=u2_ends_levin, grid_mismatch=u2_grid_mismatch[:20],
                n_grid_mismatch=len(u2_grid_mismatch), nonlevin_by_tag={str(k): v for k, v in u2_nonlevin.items()}),
        U3=dict(n=len(u3_viol), violations=u3_viol[:50], skipped_by_reason=dict(u3_skipped), bad_reason=u3_bad_reason[:50],
                skipped_with_sign_change=u3_skip_sc, calls_with_end_points=len(u3_end_pts)),
        U5=dict(sums=dict(u5), calls_with_forced=forced_calls, calls={k: v[:50] for k, v in u5_calls.items()},
                n_calls={k: len(v) for k, v in u5_calls.items()}),
        U7=dict(pairs=len(u7), unpaired=unpaired, n_rg_h_le1=sum(x[3] <= 1 for x in u7), n_rg_07a_le1=sum(x[4] <= 1 for x in u7),
                gt1_h=[x for x in u7 if x[3] > 1], gt1_07a=[x for x in u7 if x[4] > 1], failing=u7_fail,
                max_rg_h=max(u7, key=lambda x: x[3])),
        abserr_ratio_h_over_07a={" ".join(k): q(v) for k, v in sorted(eratio.items())},
        over_budget=over_budget, cost={k: v for k, v in sorted(cost.items())},
    )
    out.write_text(json.dumps(res, indent=1, default=str))
    short = {k: v for k, v in res.items() if k not in ("newly_unconverged",)}
    print(json.dumps(short, indent=1, default=str)[:25000])
    nu = res["newly_unconverged"]
    print("newly unconverged:", nu["n"], nu["by_class_kind_rtol"])


if __name__ == "__main__":
    main()
