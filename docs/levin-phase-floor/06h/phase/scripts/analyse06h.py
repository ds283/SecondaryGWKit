"""levin-phase-floor prompt 06h section 5, the phase re-measurement: H1 and H2 scored against the rules
as prompt 06h section 5 wrote them, the parent's column beside them, and the descriptive split by
the declaration's dominant term. New in 06h (its M2 tables follow 06f's analyse06f.py).

Reads, read only: out/sample/selection_{p,h}.json, out/capture_{p,h}.json, out/replay_{p,h}/*/
(result.json, nodes.json.gz, dense.json.gz), out/cross/{p_at_h,h_at_p}/*/ (cross.json,
rows.json.gz), out/cmd/32_method_check.txt and 33_refcheck_50_digits.txt, and 06f's own nodes
(var/levin-phase-floor-06f/out/replay/<unit>_<index>/nodes.json.gz) for the parent's reproduction of
06f. Writes out/phase_calls.csv, out/phase_leaves.csv, out/scores.json; prints everything.

**H1's points** are h's: every node of every returned leaf of h's replay (Clenshaw-Curtis fine nodes,
Levin collocation nodes) **and** 06f's seeded dense sample of 1,000 points per call ("at every point
sampled"). Each is scored |delta theta(u)| <= theta_abserr_h(u), the reference at the exact e^u.
**The parent's column** is the parent's code -- its theta_mod_2pi and its theta_abserr -- on the same
points (cross06h.py, p at h's points); and, beside it, on its own returned leaves' nodes and the dense
sample (its own replay).

**H2** is scored on h's returned leaves: |dI_measured| (06f's item 4, the first-order response to the
measured error at the exact e^u, at the points where the rule evaluates theta) <= the leaf's declared
term (the one the driver added, from the replay's pass-through at _declared_endpoint_phase_err).

Usage: python -B analyse06h.py   (from var/levin-phase-floor-06h/phase/)"""

import csv
import gzip
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import REC06F, header, sha256  # noqa: E402

PARTS = ("constituents", "abscissa", "phi_eval")


def gz(p):
    return json.load(gzip.open(p, "rt"))


def rest(unit):
    return unit.split("-", 1)[1]


def main():
    out = ROOT / "out"
    sel = {
        t: json.loads((out / "sample" / f"selection_{t}.json").read_text())
        for t in ("h", "p")
    }
    caps = {t: json.loads((out / f"capture_{t}.json").read_text()) for t in ("h", "p")}
    inputs = [out / "sample" / f"selection_{t}.json" for t in ("h", "p")]
    inputs += [out / f"capture_{t}.json" for t in ("h", "p")]
    inputs += [
        out / "cmd" / "32_method_check.txt",
        out / "cmd" / "33_refcheck_50_digits.txt",
    ]
    calls = []
    for e in sel["h"]:
        k = (rest(e["unit"]), e["index"])
        ep = [x for x in sel["p"] if (rest(x["unit"]), x["index"]) == k][0]
        tag = {"h": f"{e['unit']}_{e['index']}", "p": f"{ep['unit']}_{ep['index']}"}
        c = {
            "key": k,
            "unit_06f": e["unit_06f"],
            "label": e["notify_label"],
            "tag": tag,
        }
        for t in ("h", "p"):
            d = out / f"replay_{t}" / tag[t]
            for n in ("result.json", "nodes.json.gz", "dense.json.gz"):
                inputs.append(d / n)
            c[f"res_{t}"] = json.loads((d / "result.json").read_text())
            c[f"nodes_{t}"] = gz(d / "nodes.json.gz")
            c[f"dense_{t}"] = gz(d / "dense.json.gz")
            other = "p" if t == "h" else "h"
            x = out / "cross" / f"{t}_at_{other}" / tag[t]
            inputs += [x / "cross.json", x / "rows.json.gz"]
            c[f"cross_{t}"] = json.loads((x / "cross.json").read_text())
            c[f"cross_rows_{t}"] = gz(x / "rows.json.gz")
        f06 = (
            REC06F
            / "out"
            / "replay"
            / f"{e['unit_06f']}_{e['index']}"
            / "nodes.json.gz"
        )
        inputs.append(f06)
        c["nodes_06f"] = gz(f06)
        calls.append(c)
    for l in header(
        Path(__file__).resolve(),
        "levin-phase-floor 06h: the phase re-measurement, H1 and H2",
        inputs,
    ):
        print(l)

    # ---------------- the captures and replays reproduce their census
    print("\n== The captures and the replays against each tree's census (bit for bit)")
    capok = {}
    for t in ("p", "h"):
        n = sum(1 for e in caps[t] if e.get("capture_matches_census"))
        ray = sorted({str(e.get("capture_ray_initialized")) for e in caps[t]})
        capok[t] = n == len(sel[t]) == len(caps[t])
        print(
            f"   {t}: captures matching the census {n} of {len(sel[t])}; ray_initialized {ray}"
        )
    f0 = {}
    for t in ("p", "h"):
        held = [c for c in calls if c[f"res_{t}"]["F0"]["holds"]]
        f0[t] = len(held) == len(calls)
        print(
            f"   {t}: replays with F0 holding (replay = census on value, abserr, regions, depth, converged; "
            f"terms = cc_floor; every leaf reproduced) {len(held)} of {len(calls)}"
        )
    errs = [
        (c["tag"][t], k)
        for c in calls
        for t in ("h", "p")
        for k in ("M1_error", "M2_error")
        if k in c[f"res_{t}"]
    ]
    print(f"   replay errors: {errs if errs else 'none'}")

    # ---------------- the cross checks
    print("\n== The two trees' calls: spans, groups and dense samples")
    cross_ok = True
    for c in calls:
        ch = c["cross_p"]["checks"]
        bad = [k for k, v in ch.items() if not v]
        cross_ok &= not bad
        if bad:
            print(f"   {c['unit_06f']} #{c['key'][1]}: NOT EQUAL {bad}")
    print(
        f"   every call: spans, coefficients, signs, orders, K, dense points and dense references equal: {cross_ok}"
    )

    # ---------------- the parent reproduces 06f (06f's error at x_hat, at 06f's own nodes)
    print(
        "\n== The parent's replay reproduces 06f's M2 item 1 (its error at x_hat, 06f's convention)"
    )
    rep06f = []
    for c in calls:
        a = [
            (
                r["leaf"],
                r["k"],
                r["u"],
                r["theta_mod_2pi"],
                float(r["delta_theta"]).hex(),
                float(r["theta_abserr"]).hex(),
            )
            for r in c["nodes_06f"]
        ]
        b = [
            (
                r["leaf"],
                r["k"],
                r["u"],
                r["theta_mod_2pi"],
                float(r["delta_theta_float"]).hex(),
                float(r["theta_abserr"]).hex(),
            )
            for r in c["nodes_p"]
        ]
        rep06f.append(a == b)
        if a != b:
            print(
                f"   {c['unit_06f']} #{c['key'][1]}: differs ({len(a)} vs {len(b)} nodes)"
            )
    n06 = sum(len(c["nodes_06f"]) for c in calls)
    v06 = sum(
        1
        for c in calls
        for r in c["nodes_p"]
        if r["ratio_float"] is not None and r["ratio_float"] > 1
    )
    m06 = max(
        r["ratio_float"]
        for c in calls
        for r in c["nodes_p"]
        if r["ratio_float"] is not None
    )
    print(
        f"   calls whose parent nodes equal 06f's (u, theta_mod_2pi, delta theta at x_hat, theta_abserr, hex): "
        f"{sum(rep06f)} of {len(calls)}; nodes {n06}; at x_hat the parent violates at {v06}, worst {m06:.4g} (06f: 49, 148)"
    )

    # ---------------- H1
    def pts(c, t):
        rows = [dict(r, set="node") for r in c[f"nodes_{t}"]]
        rows += [dict(r, set="dense") for r in c[f"dense_{t}"]]
        return rows

    def dom(r):
        p = r.get("theta_abserr_parts")
        if not p:
            return None
        return PARTS[int(np.argmax(p))]

    print(
        "\n== H1 -- the declaration is honest (h's points; reference at the exact e^u)"
    )
    callrows = []
    worst = None
    n_pts = n_viol = 0
    by_set = {"node": [0, 0, 0.0], "dense": [0, 0, 0.0]}
    for c in calls:
        P = pts(c, "h")
        rat = [r["ratio"] for r in P]
        nv = sum(1 for x in rat if x > 1.0)
        n_pts += len(P)
        n_viol += nv
        for r in P:
            s = by_set[r["set"]]
            s[0] += 1
            s[1] += r["ratio"] > 1.0
            s[2] = max(s[2], r["ratio"])
            if worst is None or r["ratio"] > worst[0]:
                worst = (r["ratio"], c, r)
        # the parent on h's points
        X = c["cross_rows_p"]
        xr = [r["ratio"] for r in X]
        # the parent on its own points
        Pp = pts(c, "p")
        pr = [r["ratio"] for r in Pp]
        lv = c["res_h"]["M2"]["per_leaf"]
        lp = c["res_p"]["M2"]["per_leaf"]
        callrows.append(
            {
                "unit_06f": c["unit_06f"],
                "index": c["key"][1],
                "label": c["label"],
                "leaves_h": len(lv),
                "levin_leaves_h": sum(1 for L in lv if L["branch"] == "levin"),
                "leaves_p": len(lp),
                "levin_leaves_p": sum(1 for L in lp if L["branch"] == "levin"),
                "same_leaves": [(L["a"], L["b"], L["branch"]) for L in lv]
                == [(L["a"], L["b"], L["branch"]) for L in lp],
                "points_h": len(P),
                "nodes_h": len(c["nodes_h"]),
                "violations_h": nv,
                "max_ratio_h": max(rat),
                "median_ratio_h": float(np.median(rat)),
                "max_abs_dtheta_h": max(abs(r["delta_theta"]) for r in P),
                "parent_at_h_points_violations": sum(1 for x in xr if x > 1.0),
                "parent_at_h_points_max_ratio": max(xr),
                "points_p": len(Pp),
                "parent_own_violations": sum(1 for x in pr if x > 1.0),
                "parent_own_max_ratio": max(pr),
                "h_at_p_points_violations": c["cross_h"]["summary"]["all"][
                    "n_violating"
                ],
                "h_at_p_points_max_ratio": c["cross_h"]["summary"]["all"]["max_ratio"],
                "theta_abserr_h_min": min(r["theta_abserr"] for r in P),
                "theta_abserr_h_max": max(r["theta_abserr"] for r in P),
                "D_h": c["res_h"]["M2"]["D"],
                "D_p": c["res_p"]["M2"]["D"],
                "abserr_h": c["res_h"]["M2"]["declared_error"],
                "abserr_p": c["res_p"]["M2"]["declared_error"],
                "request_h": c["res_h"]["M2"]["request"],
            }
        )
    H1 = n_viol == 0
    wr, wc, wp = worst
    parts = dict(zip(PARTS, wp["theta_abserr_parts"]))
    x_parent = sum(r["parent_at_h_points_violations"] for r in callrows)
    x_parent_max = max(r["parent_at_h_points_max_ratio"] for r in callrows)
    x_parent_call = max(callrows, key=lambda r: r["parent_at_h_points_max_ratio"])
    own_parent = sum(r["parent_own_violations"] for r in callrows)
    own_parent_n = sum(r["points_p"] for r in callrows)
    own_parent_max = max(r["parent_own_max_ratio"] for r in callrows)
    print(
        f"   points scored {n_pts} (nodes {by_set['node'][0]}, dense {by_set['dense'][0]}); violating {n_viol} "
        f"(nodes {by_set['node'][1]}, dense {by_set['dense'][1]}) -> H1 {'holds' if H1 else 'fails'}"
    )
    print(
        f"   worst |delta theta|/theta_abserr_h = {wr:.6g} on {wc['unit_06f']} #{wc['key'][1]} ({wc['label']}), "
        f"{wp['set']} {'leaf ' + str(wp['leaf']) + ' node ' + str(wp['k']) if wp['set'] == 'node' else 'j ' + str(wp['j'])}, "
        f"u = {float.fromhex(wp['u'])!r} ({wp['u']}); delta theta {wp['delta_theta']:+.4e}, theta_abserr {wp['theta_abserr']:.4e} = "
        f"constituents {parts['constituents']:.4e} + abscissa {parts['abscissa']:.4e} + phi_eval {parts['phi_eval']:.4e}; "
        f"dominant: {dom(wp)}"
    )
    print(
        f"   the parent on the same points: violating {x_parent} of {n_pts}, worst ratio {x_parent_max:.6g} "
        f"({x_parent_call['unit_06f']} #{x_parent_call['index']})"
    )
    print(
        f"   the parent on its own leaves' nodes and the dense sample: violating {own_parent} of {own_parent_n}, worst {own_parent_max:.6g}"
    )
    print(
        "\n   call | label | leaves h (Levin) | leaves p (Levin) | same | points | viol h | max h | parent at h's pts: viol, max | parent own: viol, max"
    )
    for r in callrows:
        print(
            f"   {r['unit_06f']} #{r['index']} | {r['label']} | {r['leaves_h']} ({r['levin_leaves_h']}) | {r['leaves_p']} ({r['levin_leaves_p']}) | "
            f"{r['same_leaves']} | {r['points_h']} | {r['violations_h']} | {r['max_ratio_h']:.4g} | "
            f"{r['parent_at_h_points_violations']}, {r['parent_at_h_points_max_ratio']:.4g} | {r['parent_own_violations']}, {r['parent_own_max_ratio']:.4g}"
        )

    # ---------------- descriptive: by dominant part (h's points)
    print(
        "\n== Descriptive: measured/declared at h's points by the declaration's dominant part (test 4's split)"
    )
    split = {}
    for c in calls:
        for r in pts(c, "h"):
            split.setdefault(dom(r), []).append(r["ratio"])
    split_out = {}
    for part in PARTS:
        rr = np.array(split.get(part, []))
        if not len(rr):
            print(f"   {part:12s} points 0")
            split_out[part] = {"n": 0}
            continue
        split_out[part] = {
            "n": int(len(rr)),
            "median": float(np.median(rr)),
            "p90": float(np.quantile(rr, 0.9)),
            "max": float(rr.max()),
            "min": float(rr.min()),
        }
        print(
            f"   {part:12s} points {len(rr):5d}; median {np.median(rr):.3e}; 90 % {np.quantile(rr, 0.9):.3e}; "
            f"max {rr.max():.3e}; min {rr.min():.3e}"
        )

    # ---------------- H2
    print("\n== H2 -- the declared term covers the response (h's returned leaves)")
    leafrows = []
    for c in calls:
        for t in ("h", "p"):
            for L in c[f"res_{t}"]["M2"]["per_leaf"]:
                dec = L["declared"]
                resp = abs(L["dI_measured"])
                leafrows.append(
                    {
                        "tree": t,
                        "unit_06f": c["unit_06f"],
                        "index": c["key"][1],
                        "label": c["label"],
                        "leaf": L["i"],
                        "branch": L["branch"],
                        "a": L["a"],
                        "b": L["b"],
                        "depth": L["depth"],
                        "declared": dec,
                        "abs_dI_measured": resp,
                        "ratio": (
                            (resp / dec)
                            if dec > 0
                            else (float("inf") if resp > 0 else 0.0)
                        ),
                        "abs_dI_measured_float": abs(L["dI_measured_float"]),
                        "max_abs_delta_theta": L["max_abs_delta_theta"],
                        "theta_abserr_a": L["theta_abserr_a"],
                        "theta_abserr_b": L["theta_abserr_b"],
                    }
                )
    hl = [r for r in leafrows if r["tree"] == "h"]
    pl = [r for r in leafrows if r["tree"] == "p"]
    H2 = all(r["abs_dI_measured"] <= r["declared"] for r in hl)
    wl = max(hl, key=lambda r: r["ratio"])
    med = statistics.median(r["ratio"] for r in hl)
    pv = [r for r in pl if r["abs_dI_measured"] > r["declared"]]
    wpl = max(pl, key=lambda r: r["ratio"])
    print(
        f"   leaves scored {len(hl)} ({sum(1 for r in hl if r['branch'] == 'levin')} Levin); violating "
        f"{sum(1 for r in hl if r['abs_dI_measured'] > r['declared'])} -> H2 {'holds' if H2 else 'fails'}"
    )
    print(
        f"   worst |dI_measured|/declared = {wl['ratio']:.6g} on {wl['unit_06f']} #{wl['index']} leaf {wl['leaf']} "
        f"({wl['branch']}, depth {wl['depth']}): {wl['abs_dI_measured']:.4e} against {wl['declared']:.4e}; median {med:.4g}"
    )
    print(
        f"   the parent (its own leaves, its own declared terms): {len(pl)} leaves, violating {len(pv)}, worst "
        f"{wpl['ratio']:.6g} on {wpl['unit_06f']} #{wpl['index']} leaf {wpl['leaf']} ({wpl['branch']}); median "
        f"{statistics.median(r['ratio'] for r in pl):.4g}"
    )
    for r in sorted(hl, key=lambda r: -r["ratio"])[:10]:
        print(
            f"   h {r['unit_06f']} #{r['index']} leaf {r['leaf']} {r['branch']}: |dI| {r['abs_dI_measured']:.3e}, declared "
            f"{r['declared']:.3e}, ratio {r['ratio']:.3g}"
        )

    # ---------------- tables
    with open(out / "phase_calls.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(callrows[0]))
        w.writeheader()
        w.writerows(callrows)
    with open(out / "phase_leaves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(leafrows[0]))
        w.writeheader()
        w.writerows(leafrows)
    scores = {
        "captures_match_census": capok,
        "replays_F0_hold": f0,
        "two_trees_calls_equal": cross_ok,
        "parent_reproduces_06f_nodes": f"{sum(rep06f)} of {len(calls)}",
        "H1": {
            "verdict": "holds" if H1 else "fails",
            "points": n_pts,
            "points_nodes": by_set["node"][0],
            "points_dense": by_set["dense"][0],
            "violations": n_viol,
            "worst_ratio": wr,
            "worst_call": f"{wc['unit_06f']} #{wc['key'][1]}",
            "worst_point": {
                k: wp.get(k)
                for k in (
                    "set",
                    "leaf",
                    "k",
                    "j",
                    "u",
                    "delta_theta",
                    "theta_abserr",
                    "theta_abserr_parts",
                )
            },
            "worst_dominant_part": dom(wp),
            "parent_on_same_points_violations": x_parent,
            "parent_on_same_points_worst_ratio": x_parent_max,
            "parent_own_points": own_parent_n,
            "parent_own_violations": own_parent,
            "parent_own_worst_ratio": own_parent_max,
        },
        "H2": {
            "verdict": "holds" if H2 else "fails",
            "leaves": len(hl),
            "violations": sum(1 for r in hl if r["abs_dI_measured"] > r["declared"]),
            "worst_ratio": wl["ratio"],
            "worst_leaf": f"{wl['unit_06f']} #{wl['index']} leaf {wl['leaf']} ({wl['branch']})",
            "median_ratio": med,
            "parent_leaves": len(pl),
            "parent_violations": len(pv),
            "parent_worst_ratio": wpl["ratio"],
        },
        "split_by_dominant_part": split_out,
    }
    (out / "scores.json").write_text(json.dumps(scores, indent=1))
    for p in ("phase_calls.csv", "phase_leaves.csv", "scores.json"):
        print(f"wrote out/{p} sha256 {sha256(out / p)}")


if __name__ == "__main__":
    main()
