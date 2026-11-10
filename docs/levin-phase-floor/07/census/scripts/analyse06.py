"""
levin-phase-floor prompt 06 section 6: the analysis of the registered run. New in 06 (05a's
analyse.py, which scored T1-T4, is copied unchanged and unused). Reads the run's checkpoint.jsonl
and unit pickles, state/sample.json, state/predictions.md's readings, and 05's run (which rows were
capped under B1's code: 05's p at depth 20, as 05a's analyse.py read it). Runs nothing in either
tree; imports only numpy and 04's depth20_floor (for eta, F = eps k eta(z_response)).

Scores S1-S5 against prompt 06 section 6 as amended, with the readings of state/predictions.md:
S1 under reading (I) of record (a sample is pure-WKB iff its value's z_init == its z_source
exactly; the user's ruling) and reading (L) beside it (z_init null); writes every probe window,
matched across trees, to state/probe_windows.json so that S1 and S2 can be re-scored.

Usage: python -I -B analyse06.py RUN_DIR OUT_JSON PROBE_JSON
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import RUN05, S_TREE, WIDER, header  # noqa: E402

sys.path.insert(0, str(S_TREE / "docs" / "handover" / "a3-v2" / "findings"))
import depth20_floor  # noqa: E402

EPS = 2.0**-52
R_ROWS = ("R1", "R2", "R3", "R4", "R5", "R6", "R7")


def fval(x):
    return None if x is None else float.fromhex(x["hex"])


def main(run_dir, out_json, probe_json):
    import ray.cloudpickle as cloudpickle

    run_dir = Path(run_dir)
    lines = header(
        Path(__file__).resolve(),
        "levin-phase-floor 06 section 6: S1-S5 and the descriptive items",
        [run_dir / "checkpoint.jsonl", WIDER / "state" / "sample.json",
         WIDER / "state" / "predictions.md", RUN05 / "checkpoint.jsonl",
         Path(depth20_floor.__file__)],
    )
    out = lines.append
    sample = json.loads((WIDER / "state" / "sample.json").read_text())
    rows = sample["rows"]
    by_name = {r["name"]: r for r in rows}
    recs = {}
    for l in (run_dir / "checkpoint.jsonl").read_text().splitlines():
        r = json.loads(l)
        recs[r["unit"]] = r
    out(f"# {len(recs)} checkpoint records; script_sha256 {sorted({r['script_sha256'] for r in recs.values()})}")
    heads = sorted({(r["data"]["tree"], r["data"]["tree_head"], r["data"]["tree_status_short"]) for r in recs.values()})
    out(f"# trees on the records: {heads}")

    def U(tree, rtol, name):
        r = recs.get(f"{tree}-{rtol}-{name}")
        return None if r is None else r["data"]

    # ---- status, budget
    statuses = {}
    worst = (0.0, None)
    for uid, r in recs.items():
        d = r["data"]
        statuses[d["status"]] = statuses.get(d["status"], 0) + 1
        if d["budget_s"]:
            q = (d["wall_s"] or 0) / d["budget_s"]
            if q > worst[0]:
                worst = (q, uid)
    over = [uid for uid, r in recs.items() if r["data"]["status"] == "over budget"]
    errs = [uid for uid, r in recs.items() if r["data"]["status"] == "error"]
    out(f"# statuses {statuses}; over budget {over}; errors {errs}")
    out(f"# largest s wall / budget: {worst[0]:.3f} ({worst[1]})")
    warn = sum((r["data"].get("policy_warnings") or {}).get("count", 0) for r in recs.values())
    out(f"# '!! WARNING (GkSourcePolicyData' lines counted in all units: {warn}")
    probe_err = [uid for uid, r in recs.items() if r["data"].get("probe_error")]
    out(f"# probe errors: {probe_err}")

    # ---- capped under B1
    r05 = {}
    for l in (RUN05 / "checkpoint.jsonl").read_text().splitlines():
        rr = json.loads(l)
        r05[rr["unit"]] = rr["data"]
    d20 = [r["name"] for r in rows if r["role"] == "d20"]
    capped = [
        n for n in d20
        if r05[f"p-1e-7-{n}"]["status"] == "done"
        and (r05[f"p-1e-7-{n}"]["result"]["WKB_Levin_data"] or {}).get("max_depth") == 20
    ]
    out(f"# rows capped under B1's code (05's p at depth 20): {len(capped)}: {capped}")

    res = {}
    # ---- S3
    out("")
    out("== S3: on every row (all 39), under s at 1e-7, no group has a region at depth >= 18; an over-budget unit counts as false")
    s3 = []
    for r in rows:
        n = r["name"]
        d = U("s", "1e-7", n)
        if d is None or d["status"] != "done":
            s3.append((n, False, None))
            continue
        pk = cloudpickle.loads((run_dir / d["pickle"]).read_bytes())
        maxd = max([int(c["depth"].max()) for c in pk["regions"] if len(c["depth"])] or [0])
        maxd_ck = max([c["max_depth"] for c in d["calls"]] or [0])
        assert maxd == maxd_ck, (n, maxd, maxd_ck)
        s3.append((n, maxd < 18, maxd))
    s3_holds = all(ok for _n, ok, _m in s3)
    out(f"   true on {sum(ok for _n, ok, _m in s3)} of {len(s3)}; largest depth {max(m or 0 for _n, _ok, m in s3)}; "
        f"on the {len(capped)} capped under B1 largest {max(m or 0 for n, _ok, m in s3 if n in capped)}")
    out(f"   false on: {[(n, m) for n, ok, m in s3 if not ok]}")
    out(f"== S3: {'HOLDS' if s3_holds else 'FAILS'}")

    # ---- S4
    out("")
    out("== S4: on every row (all 39), |total_s - total_t5| <= total_abserr_t5, at 1e-7")
    s4 = []
    for r in rows:
        n = r["name"]
        a, b = U("t5", "1e-7", n), U("s", "1e-7", n)
        if a is None or a["status"] != "done" or b is None or b["status"] != "done":
            s4.append((n, None, None, None, None))
            continue
        ta, tb = fval(a["result"]["total"]), fval(b["result"]["total"])
        ea = fval(a["result"]["total_abserr"])
        diff = abs(tb - ta)
        s4.append((n, diff <= ea, diff, ea, ta))
    und = [x[0] for x in s4 if x[1] is None]
    fails = [x for x in s4 if x[1] is False]
    out(f"   {'row':<14s} {'role':<7s} {'capped':6s} {'total_t5':>24s} {'|s-t5|':>11s} {'abserr_t5':>11s} {'ratio':>10s} {'rel':>10s}")
    for n, ok, diff, ea, ta in s4:
        if ok is None:
            out(f"   {n:<14s} undecided (unit not done)")
            continue
        ratio = diff / ea if ea not in (0.0,) and math.isfinite(ea) else (0.0 if diff == 0 else (0.0 if math.isinf(ea) else float("inf")))
        rel = diff / abs(ta) if ta else float("nan")
        out(f"   {n:<14s} {by_name[n]['role']:<7s} {str(n in capped):6s} {ta!r:>24s} {diff:11.3e} {ea:11.3e} {ratio:10.3e} {rel:10.3e} {'ok' if ok else 'FALSE'}")
        res.setdefault(n, {})["S4"] = dict(ok=ok, diff=diff, abserr_t5=ea, ratio=ratio, rel=rel)
    s4_verdict = "FAILS" if fails else ("UNDECIDED" if und else "HOLDS")
    out(f"   true on {sum(1 for x in s4 if x[1])} of {len(s4)}; false on {[(x[0], round(x[2] / x[3], 3)) for x in fails]}; undecided {und}")
    bitid = [x[0] for x in s4 if x[2] == 0.0]
    out(f"   bit-identical totals on {len(bitid)} rows: {bitid}")
    out(f"== S4: {s4_verdict}")

    # ---- S5
    out("")
    out("== S5: on the 31 depth-20 rows with both s units in budget, |total_s(1e-7) - total_s(1e-8)| <= total_abserr_s(1e-7)")
    s5 = []
    excl = []
    for n in d20:
        a, b = U("s", "1e-7", n), U("s", "1e-8", n)
        if a is None or b is None or a["status"] != "done" or b["status"] != "done":
            excl.append(n)
            continue
        t7, t8 = fval(a["result"]["total"]), fval(b["result"]["total"])
        e7 = fval(a["result"]["total_abserr"])
        diff = abs(t7 - t8)
        ok = diff <= e7
        ratio = diff / e7 if (e7 and math.isfinite(e7)) else 0.0
        s5.append((n, ok, ratio))
        res.setdefault(n, {})["S5"] = dict(ok=ok, diff=diff, abserr_s7=e7, ratio=ratio)
    for n, ok, ratio in s5:
        out(f"   {n:<14s} capped-B1 {str(n in capped):5s} ratio {ratio:.3g} {'ok' if ok else 'FALSE'}")
    s5_holds = all(ok for _n, ok, _r in s5)
    out(f"   true on {sum(ok for _n, ok, _r in s5)} of {len(s5)} scored; excluded {excl}; false on "
        f"{[(n, round(r, 3)) for n, ok, r in s5 if not ok]}; on the capped: true on "
        f"{sum(1 for n, ok, _r in s5 if ok and n in capped)} of {sum(1 for n, _o, _r in s5 if n in capped)}, largest "
        f"{max([r for n, _o, r in s5 if n in capped] or [0]):.3g}")
    out(f"== S5: {'HOLDS' if s5_holds else 'FAILS'}")

    # ---- S1 / S2
    out("")
    out("== the phase probe (R1-R7): windows matched across trees")
    allwin = []
    for n in R_ROWS:
        r = by_name[n]
        F = EPS * float(r["k"]) * depth20_floor.eta(float(r["z_response"]))
        a, b = U("t5", "1e-7", n), U("s", "1e-7", n)
        wa = {(w["sub"], w["factor"], w["i0"]): w for w in (a.get("probe") or [])}
        wb = {(w["sub"], w["factor"], w["i0"]): w for w in (b.get("probe") or [])}
        only = sorted(set(wa) ^ set(wb))
        for key in sorted(set(wa) | set(wb)):
            x, y = wa.get(key), wb.get(key)
            e = {"row": n, "F": F, "sub": key[0], "factor": key[1], "i0": key[2]}
            if x is None or y is None:
                e["matched"] = False
                allwin.append(e)
                continue
            e["matched"] = x["z"] == y["z"]
            e.update(
                z_centre=y["z_centre"], z=y["z"], sigma_t5=x["rms"], sigma_s=y["rms"],
                abs_theta_centre=y["abs_theta_centre"], abs_theta_max=y["abs_theta_max"],
                theta_deriv_u_t5=x["theta_deriv_u_centre"], theta_deriv_u_s=y["theta_deriv_u_centre"],
                phi_t5=x["phi"], phi_s=y["phi"], regime=y["regime"],
            )
            if key[1] == "G":
                zs = [float.fromhex(h) for h in y["z"]]
                pure_I_t5 = all(zi is not None and zi == z for zi, z in zip(x["z_init"], zs))
                pure_I_s = all(zi is not None and zi == z for zi, z in zip(y["z_init"], zs))
                pure_parent_s = all(y.get("rerun_pure_WKB") or [False])
                e.update(
                    z_init_t5=x["z_init"], z_init_s=y["z_init"],
                    pure_I=pure_I_t5 and pure_I_s, pure_I_agree=(pure_I_t5 == pure_I_s == pure_parent_s),
                    pure_L=all(v is None for v in x["z_init"]) and all(v is None for v in y["z_init"]),
                    dtheta_rms_unwrapped=y.get("dtheta_rms_unwrapped"), dtheta_rms_raw=y.get("dtheta_rms_raw"),
                    rerun_pure_WKB=y.get("rerun_pure_WKB"),
                )
            bound = max(1e-3 * e["sigma_t5"], 1e-6 * F)
            e["bound"] = bound
            e["true"] = e["sigma_s"] <= bound
            allwin.append(e)
        out(f"   {n}: F {F:.4g}; windows t5 {len(wa)}, s {len(wb)}, in one tree only {len(only)}")
    Path(probe_json).write_text(json.dumps(allwin))

    def score(ws, label):
        unmatched = [w for w in ws if not w["matched"]]
        scored = [w for w in ws if w["matched"]]
        false = [w for w in scored if not w["true"]]
        if false:
            v = "FAILS"
        elif not scored:
            v = "UNDECIDED (no window)"
        elif unmatched:
            v = "UNDECIDED (unmatched windows)"
        else:
            v = "HOLDS"
        out(f"   {label}: {len(scored)} scored windows, {len(unmatched)} unmatched; true in {len(scored) - len(false)}; false in {len(false)}")
        if scored:
            rs = [w["sigma_s"] / w["F"] for w in scored]
            rt = [w["sigma_t5"] / w["F"] for w in scored]
            q = [w["sigma_s"] / w["bound"] for w in scored]
            out(f"      sigma_t5/F: median {np.median(rt):.3g}, max {max(rt):.3g}; sigma_s/F: median {np.median(rs):.3g}, max {max(rs):.3g}; sigma_s/bound max {max(q):.3g}")
            out(f"      sigma_t5 == 0 in {sum(1 for w in scored if w['sigma_t5'] == 0.0)} windows")
            byrow = {}
            for w in scored:
                byrow.setdefault(w["row"], []).append(w)
            for rn, wl in byrow.items():
                out(f"      {rn}: {len(wl)} windows; sigma_t5/F median {np.median([w['sigma_t5'] / w['F'] for w in wl]):.3g} max {max(w['sigma_t5'] / w['F'] for w in wl):.3g}; "
                    f"sigma_s/F median {np.median([w['sigma_s'] / w['F'] for w in wl]):.3g} max {max(w['sigma_s'] / w['F'] for w in wl):.3g}; false {sum(1 for w in wl if not w['true'])}")
        for w in false[:12]:
            out(f"      FALSE: {w['row']} sub {w['sub']} {w['factor']} i0 {w['i0']} z {w['z_centre']:.6g} sigma_s {w['sigma_s']:.3e} sigma_t5 {w['sigma_t5']:.3e} bound {w['bound']:.3e} F {w['F']:.3e}")
        return v, scored, false

    out("")
    out("== S1 (reading (I), of record): every pure-WKB G window: sigma_s <= max(1e-3 sigma_t5, 1e-6 F)")
    G = [w for w in allwin if w["factor"] == "G"]
    disagree = [w for w in G if w.get("matched") and not w.get("pure_I_agree", True)]
    out(f"   G windows {len(G)}; tag (I) disagrees between t5, s and the s re-run's parents in {len(disagree)}")
    s1, s1_scored, s1_false = score([w for w in G if not w["matched"] or (w.get("pure_I") and w.get("pure_I_agree"))], "pure-WKB (I)")
    if disagree and s1 == "HOLDS":
        s1 = "UNDECIDED (tag disagreement)"
    out(f"== S1 (I): {s1}")
    out("")
    out("== S1 (reading (L), unscored): z_init null in all 9 samples")
    nL = sum(1 for w in G if w.get("pure_L"))
    out(f"   pure-WKB windows under (L): {nL}; S1 by (L): {'UNDECIDED (no window)' if nL == 0 else 'see windows'}")
    out("")
    out("== numerically seeded G windows (not scored): sigma_s beside the window RMS of the re-run deltaTheta")
    seeded = [w for w in G if w["matched"] and not w.get("pure_I")]
    out(f"   {len(seeded)} windows")
    if seeded:
        for rn in R_ROWS:
            wl = [w for w in seeded if w["row"] == rn]
            if not wl:
                continue
            ss = np.array([w["sigma_s"] for w in wl])
            st = np.array([w["sigma_t5"] for w in wl])
            du = np.array([w["dtheta_rms_unwrapped"] for w in wl])
            F = wl[0]["F"]
            out(f"   {rn}: {len(wl)} windows; sigma_s/F median {np.median(ss) / F:.3g} max {ss.max() / F:.3g}; "
                f"sigma_t5/F median {np.median(st) / F:.3g} max {st.max() / F:.3g}; dtheta RMS/F median {np.median(du) / F:.3g} max {du.max() / F:.3g}; "
                f"sigma_s / dtheta RMS median {np.median(ss / np.where(du > 0, du, np.nan)):.3g}")
    out("")
    out("== S2: every probe window of a T factor: sigma_s <= max(1e-3 sigma_t5, 1e-6 F)")
    T = [w for w in allwin if w["factor"] in ("Tq", "Tr")]
    s2, s2_scored, s2_false = score(T, "T windows")
    out(f"== S2: {s2}")

    # ---- descriptive
    out("")
    out("== descriptive: theta_deriv (log) at the window centres, s against t5")
    for fac in ("G", "Tq", "Tr"):
        wl = [w for w in allwin if w["factor"] == fac and w["matched"]]
        if not wl:
            continue
        rel = np.array([abs(w["theta_deriv_u_s"] - w["theta_deriv_u_t5"]) / abs(w["theta_deriv_u_t5"]) for w in wl if w["theta_deriv_u_t5"]])
        absd = np.array([abs(w["theta_deriv_u_s"] - w["theta_deriv_u_t5"]) for w in wl])
        out(f"   {fac}: {len(wl)} windows; |d theta_deriv| median {np.median(absd):.3g}, max {absd.max():.3g}; relative median {np.median(rel):.3g}, max {rel.max():.3g}")
    out("")
    out("== descriptive per row: totals, convergence, regions, depth, Levin_z, policy")
    out(f"   {'row':<14s} {'role':<7s} {'cap':5s} {'conv t5/s7/s8':14s} {'regions t5->s7 (s8)':22s} {'depth t5->s7 (s8)':18s} Levin_z t5 / s; type,quality t5 / s; wall t5, s7/budget")
    groups_changed = 0
    groups_total = 0
    for r in rows:
        n = r["name"]
        a, b, c = U("t5", "1e-7", n), U("s", "1e-7", n), U("s", "1e-8", n)
        def summ(d):
            if d is None or d["status"] != "done":
                return None, None, None
            return d["result"]["total_converged"], sum(x["num_regions"] for x in d["calls"]), max([x["max_depth"] for x in d["calls"]] or [0])
        ca, ra, ma = summ(a)
        cb, rb, mb = summ(b)
        cc, rc, mc = summ(c)
        if a and b and a["status"] == b["status"] == "done":
            for x, y in zip(a["calls"], b["calls"]):
                groups_total += 1
                if (x["num_regions"], x["max_depth"], x["converged"]) != (y["num_regions"], y["max_depth"], y["converged"]):
                    groups_changed += 1
            assert [(x["regime"], x["group"]) for x in a["calls"]] == [(x["regime"], x["group"]) for x in b["calls"]], n
        ga, gb = (a or {}).get("GkPolicy") or {}, (b or {}).get("GkPolicy") or {}
        lz = "same" if ga.get("Levin_z") == gb.get("Levin_z") else f"{ga.get('Levin_z')} -> {gb.get('Levin_z')}"
        tq = "same" if (ga.get("type"), ga.get("quality"), ga.get("crossover_z")) == (gb.get("type"), gb.get("quality"), gb.get("crossover_z")) else f"{ga.get('type')},{ga.get('quality')},{ga.get('crossover_z')} -> {gb.get('type')},{gb.get('quality')},{gb.get('crossover_z')}"
        res.setdefault(n, {}).update(conv=(ca, cb, cc), regions=(ra, rb, rc), depth=(ma, mb, mc), Levin_z=(ga.get("Levin_z"), gb.get("Levin_z")), policy=tq)
        out(f"   {n:<14s} {r['role']:<7s} {str(n in capped):5s} {str(ca)[0]}/{str(cb)[0]}/{str(cc)[0] if cc is not None else '-'}{'':9s} {ra}->{rb} ({rc}){'':8s} {ma}->{mb} ({mc}){'':8s} Levin_z {lz}; policy {tq}; "
            f"wall {a['wall_s']:.3f}, {b['wall_s']:.3f}/{b['budget_s']:.3f}")
    out(f"   group calls whose (regions, max depth, converged) changed t5 -> s at 1e-7: {groups_changed} of {groups_total}")
    for label, sel in (("all 39", lambda n: True), ("depth-20 (31)", lambda n: by_name[n]["role"] == "d20"), ("controls (8)", lambda n: by_name[n]["role"] == "control"), ("capped under B1 (11)", lambda n: n in capped), ("R1-R7", lambda n: n in R_ROWS)):
        names = [r["name"] for r in rows if sel(r["name"])]
        def tot(i):
            return sum((res[n]["regions"][i] or 0) for n in names)
        out(f"   regions {label}: t5 {tot(0)} -> s@1e-7 {tot(1)}; s@1e-8 {tot(2)}; unconverged t5 {sum(1 for n in names if res[n]['conv'][0] is False)}, s@1e-7 {sum(1 for n in names if res[n]['conv'][1] is False)}, s@1e-8 {sum(1 for n in names if res[n]['conv'][2] is False)}")
    infs = [n for n in res if "S4" in res[n] and math.isinf(res[n]["S4"]["abserr_t5"])]
    out(f"   rows with an infinite total_abserr_t5: {infs}")
    out("")
    out("== descriptive: the re-run producers against A3 v2's stored values (s@1e-7 units)")
    agg = dict(G_vals=0, G_div=0, G_div_far=[], G_mod=0.0, T_vals=0, T_div=0, T_div_far=[], T_mod=0.0, G_amp=0.0, T_amp=0.0, pol_changed=[], timing=0.0)
    for r in rows:
        d = U("s", "1e-7", r["name"])
        rr = (d or {}).get("rerun")
        if not rr:
            continue
        Gv, Tv = rr["G_values_vs_stored"], rr["T_values_vs_stored"]
        agg["G_vals"] += Gv["n_values"]; agg["G_div"] += Gv["theta_div_2pi_changes"]; agg["G_div_far"] += Gv["theta_div_2pi_changes_not_one_cycle_near_boundary"]
        agg["G_mod"] = max(agg["G_mod"], Gv["max_abs_dmod_over_eps_theta_same_div"]); agg["G_amp"] = max(agg["G_amp"], Gv["max_rel_change_G_WKB"])
        agg["T_vals"] += Tv["n_values"]; agg["T_div"] += Tv["theta_div_2pi_changes"]; agg["T_div_far"] += Tv["theta_div_2pi_changes_not_one_cycle_near_boundary"]
        agg["T_mod"] = max(agg["T_mod"], Tv["max_abs_dmod_over_eps_theta_same_div"]); agg["T_amp"] = max(agg["T_amp"], Tv["max_rel_change_T_WKB"])
        agg["timing"] += rr["timing"]["total"]
        P = rr["GkSourcePolicyData"]
        if any(P["rerun"][k] != P["stored"][k] for k in ("type", "quality", "crossover_z", "Levin_z")):
            agg["pol_changed"].append(r["name"])
    out(f"   G values re-run {agg['G_vals']}: theta_div_2pi changes {agg['G_div']} (not one cycle within 4 eps|theta| of a boundary: {agg['G_div_far']}); "
        f"max |d theta_mod_2pi|/(eps|theta|) at the same cycle {agg['G_mod']:.3g}; max relative G_WKB change {agg['G_amp']:.3g}")
    out(f"   T values re-run {agg['T_vals']}: theta_div_2pi changes {agg['T_div']} (not near a boundary: {agg['T_div_far']}); max |d mod|/(eps|theta|) {agg['T_mod']:.3g}; max relative T_WKB change {agg['T_amp']:.3g}")
    out(f"   rows whose recomputed GkSourcePolicyData type/quality/crossover_z/Levin_z differs from A3 v2's: {agg['pol_changed']}")
    out(f"   re-run time over the 39 s@1e-7 units: {agg['timing']:.0f} s")

    verdicts = {"S1 (I)": s1, "S1 (L)": "UNDECIDED (no window)" if nL == 0 else "see windows", "S2": s2,
                "S3": "HOLDS" if s3_holds else "FAILS", "S4": s4_verdict, "S5": "HOLDS" if s5_holds else "FAILS"}
    out("")
    out(f"== verdicts: {verdicts}")
    Path(out_json).write_text(json.dumps({"verdicts": verdicts, "rows": res, "capped": capped,
                                          "over_budget": over, "errors": errs, "worst_budget": worst,
                                          "policy_warnings": warn, "rerun": {k: v for k, v in agg.items()}}, default=str, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:4])
