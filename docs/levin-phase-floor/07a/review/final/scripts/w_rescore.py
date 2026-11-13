"""Orchestrator's own re-score of W2, W3, W5, W6 and W7, with descriptive cross-checks (levin-phase-floor 07a Part C,
§C.2). Read only but for OUT_JSON. Independent of the census agent's scripts; W2/W3 adapted from 07's review
v235_rescore.py, W6 from xcheck_recount.py.

usage: python -I -B w_rescore.py OUT_JSON

Both checkpoints are read at my own indices (review/stop3/index07{a,}.json, file SHA-256s asserted), each line accepted
only if its SHA-256 matches; region records read only where needed, accepted only against pickle_record_sha256.
- W2 (on h): a returned Levin leaf (type 0) holding a scan sign change, by the scan's grid test recomputed from the
  pickle's full scan (sign_change_x +- halfwidth inside the leaf), exact; my grid set compared with the scan's own; the
  scan's 'ends' listings of Levin leaves counted apart.
- W3 (on h): on every scanned call, each sign change has a located root within h and each located root lies within h
  of a sign change (exact). Every location_skipped_phases entry at h is listed; any whose reason is not 'zero_rate' is
  a violation; scanned calls with a skipped phase and a sign change listed.
- W5 (on h, every call): sum of route_07a.n_cc_with_levin_half; sum of route_07a.n_partition_nonforced (and
  partition.forced); sum of guard_07.num_guarded_regions; and the post-unit solves' n_partition, reported beside.
- W6: population = 07's calls with guard_07.locates or .guards (either rtol), paired across rtols by row, kind, index;
  r_g = |v_7 - v_8| / e_7 at h, exact; also at 07.
- W7: calls located at 07 and at h, route_07a n_route_differs == n_lu_differs == 0, and every phase located at h
  located at 07: |v_h - v_07| <= 1e-5 * e_07 and converged equal.
- Descriptive cross-checks: class counts by rtol; converged up/down by class and rtol; converged->unconverged calls with
  a route change and no window (10.4's list); over-budget units; summed wall and regions; theta_end_max and min margins.
"""
import hashlib, json, pickle, sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
R7A = MAIN / "var/runs/levin-phase-floor-07a-census-20261112T132744"
R07 = MAIN / "var/runs/levin-phase-floor-07-census-20261108T171137"
IX = MAIN / "var/levin-phase-floor-07a/review/stop3"


def fh(v):
    return Fraction(float.fromhex(v["hex"] if isinstance(v, dict) else float(v).hex()))


def fv(v):
    return float.fromhex(v["hex"]) if isinstance(v, dict) else v


def readline(f, e):
    f.seek(e[0]); raw = f.read(e[1])
    assert hashlib.sha256(raw).hexdigest() == e[2]
    return json.loads(raw)["data"]


def klass(ch, c7):
    gh, g7 = ch["guard_07"], c7["guard_07"]
    lh = gh["n_points"] > 0 or gh["n_windows"] > 0
    l7 = g7["n_points"] > 0 or g7["n_windows"] > 0
    if lh and l7: return "window_both"
    if lh: return "window_h_only"
    if l7: return "window_07_only"
    ra = ch["route_07a"]
    return "own" if ra["own_differs"] else ("aux_only" if ra["aux_differs"] else "untouched")


def main():
    out = Path(sys.argv[1])
    i7a = json.load(open(IX / "index07a.json")); i07 = json.load(open(IX / "index07.json"))
    assert i7a["sha256"] == "461df2ec66969a113e9f6d590f05a4d5b300d7c1e0cb62d05c7385d23e8de56b"
    assert i07["sha256"] == "03dfe786f69343a0e86f436936161ace728630f26d687061e186d0883dd5cb10"
    C = Counter(); cls_rt = Counter(); conv = Counter()
    w2_viol, w2_ends_levin, w2_grid_mismatch, w2_nonlevin = [], [], [], Counter()
    w3_viol, w3_skipped, w3_bad_reason, w3_skip_sc, w3_end_pts = [], Counter(), [], [], []
    w5 = Counter(); w5_calls = defaultdict(list)
    pop = {}   # (row, kind, index) -> {rtol: (ch, c7)}
    w7_rows, w7_viol = [], []
    c104 = []
    over_budget = []
    cost = Counter()
    theta = {}; m6 = {}; m20 = {}
    f7a = open(R7A / "checkpoint.jsonl", "rb"); f07 = open(R07 / "checkpoint.jsonl", "rb")
    for key in sorted(i7a["units"]):
        dh, d7 = readline(f7a, i7a["units"][key]), readline(f07, i07["units"][key])
        rt, row = dh["rtol"], dh["row"]
        C["units"] += 1
        C["units_done"] += dh["status"] == "done"
        if dh.get("wall_s") is not None and dh.get("budget_s") is not None and dh["wall_s"] > dh["budget_s"]:
            over_budget.append((key, dh["wall_s"]))
        cost[f"wall_h_{rt}"] += dh.get("wall_s") or 0; cost[f"wall_07_{rt}"] += d7.get("wall_s") or 0
        c7s = {(k, c["index"]): c for k in ("calls", "other_calls") for c in d7[k]}
        need_pickle = False
        for kind in ("calls", "other_calls"):
            for ch in dh[kind]:
                i = ch["index"]; c7 = c7s[(kind, i)]
                k = klass(ch, c7)
                cls_rt[(k, rt)] += 1; C["calls"] += 1
                cost[f"regions_h_{rt}"] += ch["num_regions"]; cost[f"regions_07_{rt}"] += c7["num_regions"]
                conv[(k, rt, bool(c7["converged"]), bool(ch["converged"]))] += 1
                gh, ra = ch["guard_07"], ch["route_07a"]
                # W5
                a = ra["n_cc_with_levin_half"]; b = ra["n_partition_nonforced"]; fo = ra["partition"]["forced"]
                g = gh["num_guarded_regions"]
                w5["cc_with_levin_half"] += a; w5["partition_nonforced"] += b; w5["partition_forced"] += fo
                w5["num_guarded_regions"] += g
                for nm, v in (("cc_with_levin_half", a), ("partition_nonforced", b), ("num_guarded_regions", g)):
                    if v:
                        w5_calls[nm].append((key, kind, i, v))
                for pk, pv in (ch.get("route_07a_post") or {}).items():
                    w5[f"post_{pk}_n_partition"] += pv.get("n_partition", 0)
                # descriptive margins
                lab = ("group" if kind == "calls" else "analytic", rt)
                for nm, store in (("theta_end_max_nonforced", theta), ("theta_end_max_forced", theta)):
                    v = ra.get(nm)
                    if v is not None:
                        x = abs(fv(v)); kk = lab + (nm,)
                        if x > store.get(kk, (-1,))[0]:
                            store[kk] = (x, key, i)
                for nm, store in (("min_margin_6pi_same", m6), ("min_margin_20pi_same_levin", m20)):
                    v = ra.get(nm)
                    if v is not None:
                        x = abs(fv(v["margin"]))
                        if lab not in store or x < store[lab][0]:
                            store[lab] = (x, key, i, fv(v["V"]), fv(v["est_07"]))
                # W6 population
                g7 = c7["guard_07"]
                if g7["locates"] or g7["guards"]:
                    pop.setdefault((row, kind, i), {})[rt] = (ch, c7)
                # W7
                if k == "window_both":
                    ph_h = {p["phase"] for p in gh["points"]}; ph_7 = {p["phase"] for p in g7["points"]}
                    if ra["n_route_differs"] == 0 and ra["n_lu_differs"] == 0 and ph_h <= ph_7:
                        mv = abs(fh(ch["value"]) - fh(c7["value"])) / fh(c7["abserr"])
                        w7_rows.append((key, kind, i, float(mv), ch["converged"], c7["converged"]))
                        if mv > Fraction(1, 10 ** 5) or ch["converged"] != c7["converged"]:
                            w7_viol.append(w7_rows[-1])
                    else:
                        C["window_both_not_in_w7"] += 1
                # 10.4
                if k in ("own", "aux_only") and c7["converged"] and not ch["converged"]:
                    c104.append(dict(key=key, kind=kind, index=i, cls=k, abserr_07=fv(c7["abserr"]), abserr_h=fv(ch["abserr"]),
                                     n_route_differs=ra["n_route_differs"], n_lu_differs=ra["n_lu_differs"]))
                # W3 skipped
                for e in gh["location_skipped_phases"]:
                    w3_skipped[e.get("reason", "<no reason>")] += 1
                    if e.get("reason") != "zero_rate":
                        w3_bad_reason.append((key, kind, i, e))
        # W3 and W2 on scanned calls (h)
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
                w3_skip_sc.append((key, s["index"], s["kind"], len(sc)))
            roots = [fh(p["x"]) for p in g["points"] if p["kind"] == "root"]
            if any(p["kind"] == "end" for p in g["points"]):
                w3_end_pts.append((key, s["index"]))
            bad = [float(z) for z in sc if not any(abs(z - r) <= hh for r in roots)]
            bad += [float(r) for r in roots if not any(abs(z - r) <= hh for z in sc)]
            C["w3_scored"] += 1
            if sc or roots:
                C["w3_with_change_or_root"] += 1
            if bad:
                w3_viol.append((key, s["index"], s["kind"], bad[:4]))
        if not need_pickle:
            continue
        with open(R7A / dh["pickle"], "rb") as f:
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
                w2_grid_mismatch.append((key, i, sorted(mine), sorted(sg)))
            for kk in mine:
                if int(r["type"][kk]) == 0:
                    w2_viol.append((key, i, kk))
                else:
                    w2_nonlevin[("window" if int(r["window_07"][kk]) else "", "guarded" if int(r["guarded_07"][kk]) else "")] += 1
            for x in sf["listed"]:
                if x["how"] == "ends" and x["branch"] == "Levin":
                    w2_ends_levin.append((key, i, x["region_index"]))
    # W6
    w6 = []; unpaired = []
    for (row, kind, i), d in sorted(pop.items()):
        if set(d) != {"1e-7", "1e-8"}:
            unpaired.append((row, kind, i)); continue
        (h7, o7), (h8, o8) = d["1e-7"], d["1e-8"]
        assert json.dumps(h7["x_span"]) == json.dumps(h8["x_span"])
        rg_h = abs(fh(h7["value"]) - fh(h8["value"])) / fh(h7["abserr"])
        rg_7 = abs(fh(o7["value"]) - fh(o8["value"])) / fh(o7["abserr"])
        w6.append((row, kind, i, float(rg_h), float(rg_7), h7["converged"], h8["converged"]))
    w6_gt1 = [x for x in w6 if x[3] > 1]
    res = dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        counts=dict(C), class_by_rtol={f"{k} {r}": n for (k, r), n in sorted(cls_rt.items())},
        converged_07_to_h={f"{k} {r} {a}->{b}": n for (k, r, a, b), n in sorted(conv.items())},
        W2=dict(n=len(w2_viol), violations=w2_viol[:50], ends_levin=w2_ends_levin, grid_mismatch=w2_grid_mismatch[:20],
                n_grid_mismatch=len(w2_grid_mismatch), nonlevin_by_tag={str(k): v for k, v in w2_nonlevin.items()}),
        W3=dict(n=len(w3_viol), violations=w3_viol[:50], skipped_by_reason=dict(w3_skipped), bad_reason=w3_bad_reason[:50],
                skipped_with_sign_change=w3_skip_sc, calls_with_end_points=len(w3_end_pts)),
        W5=dict(sums=dict(w5), calls={k: v[:50] for k, v in w5_calls.items()}, n_calls={k: len(v) for k, v in w5_calls.items()}),
        W6=dict(pairs=len(w6), unpaired=unpaired, n_rg_h_le1=sum(x[3] <= 1 for x in w6), n_rg_07_le1=sum(x[4] <= 1 for x in w6),
                gt1=w6_gt1, max_rg_h=max(w6, key=lambda x: x[3])),
        W7=dict(n_calls=len(w7_rows), n_viol=len(w7_viol), violations=w7_viol[:50],
                max_move=max(w7_rows, key=lambda x: x[3]) if w7_rows else None,
                n_moved=sum(x[3] > 0 for x in w7_rows)),
        c104=dict(n=len(c104), calls=c104[:200]),
        over_budget=over_budget, cost={k: v for k, v in sorted(cost.items())},
        theta_end_max={" ".join(k): v for k, v in theta.items()},
        min_margin_6pi={" ".join(k): v for k, v in m6.items()}, min_margin_20pi={" ".join(k): v for k, v in m20.items()},
    )
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "c104"}, indent=1, default=str)[:20000])
    print("c104 n", len(c104))


if __name__ == "__main__":
    main()
