"""levin-phase-floor prompt 06f sections 1-4: Gate F0's table, M1's and M2's tables, and F1-F6 scored
against the rules as prompt section 4 wrote them. New in 06f. Reads, read only: out/sample/sample.json,
out/capture.json, out/replay/*/result.json and out/cmd/31_method_check.txt. Writes out/f0.csv,
out/m1_calls.csv, out/m1_z2_by_depth.csv, out/m1_z2_subtrees.csv, out/m2_calls.csv,
out/m2_leaves.csv, out/scores.json; prints everything.

Usage: python -B analyse06f.py   (from var/levin-phase-floor-06f/)"""

import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402

Z2 = ("g-1e-8-0-33995", 5)


def main():
    rep = ROOT / "out" / "replay"
    sample = json.loads((ROOT / "out" / "sample" / "sample.json").read_text())
    inputs = [
        ROOT / "out" / "sample" / "sample.json",
        ROOT / "out" / "capture.json",
        ROOT / "out" / "cmd" / "31_method_check.txt",
    ]
    results = {}
    for d in sorted(rep.iterdir()):
        p = d / "result.json"
        inputs.append(p)
        results[d.name] = json.loads(p.read_text())
    for l in header(
        Path(__file__).resolve(), "levin-phase-floor 06f: F0, M1, M2 and F1-F6", inputs
    ):
        print(l)
    caps = {
        (e["unit"], e["index"]): e
        for e in json.loads((ROOT / "out" / "capture.json").read_text())
    }

    # ---------------- F0
    print("\n== Gate F0")
    rows = []
    for c in sample["M1"] + sample["M2"]:
        key = (c["unit"], c["index"])
        r = results.get(f"{c['unit']}_{c['index']}")
        f0 = r["F0"] if r else None
        row = {
            "sample": c["sample"],
            "unit": c["unit"],
            "index": c["index"],
            "capture_matches_census": caps[key]["capture_matches_census"],
            "replay_equals_census": f0 and f0["replay_equals_census"],
            "terms_equal_census_sums": f0 and f0["terms_equal_census_sums"],
            "every_leaf_reproduced": f0 and f0["every_leaf_reproduced"],
            "leaves_sum_to_value_and_abserr": f0
            and f0["leaves_sum_to_value_and_abserr"],
            "n_cc_leaves": f0 and f0["n_cc_leaves"]["replay"],
            "holds": f0 and f0["holds"],
            "value": f0 and f0["replay"]["value"],
            "abserr": f0 and f0["replay"]["abserr"],
            "num_regions": f0 and f0["replay"]["num_regions"],
            "max_depth": f0 and f0["replay"]["max_depth"],
            "converged": f0 and f0["replay"]["converged"],
            "digest": r and r["digest"],
        }
        rows.append(row)
    det = results.get(f"{Z2[0]}_{Z2[1]}_det2")
    z2r = results[f"{Z2[0]}_{Z2[1]}"]
    det_ok = (
        det is not None
        and det["digest"] == z2r["digest"]
        and det["leaves_sha256"] == z2r["leaves_sha256"]
        and det.get("solves_cc_sha256") == z2r.get("solves_cc_sha256")
        and det["header"]["pid"] != z2r["header"]["pid"]
    )
    with open(ROOT / "out" / "f0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_fail = sum(1 for r in rows if not (r["capture_matches_census"] and r["holds"]))
    print(
        f"calls {len(rows)}; capture and replay equal 06e's census record, terms equal the checkpoint's sums: "
        f"{len(rows) - n_fail}; failing {n_fail}"
    )
    for r in rows:
        if not (r["capture_matches_census"] and r["holds"]):
            print("   FAILS:", r)
    print(
        f"Z2's call twice in fresh processes (pids {z2r['header']['pid']}, {det and det['header']['pid']}): digest "
        f"{z2r['digest'][:16]} / {det and det['digest'][:16]}; per-leaf table and every CC solve's E identical: {det_ok}"
    )
    f0_holds = n_fail == 0 and det_ok
    print(f"F0 {'holds' if f0_holds else 'FAILS'}; replacements: none")

    # ---------------- M1
    print("\n== M1")
    m1rows = []
    for c in sample["M1"]:
        r = results[f"{c['unit']}_{c['index']}"]
        m = r["M1"]
        m1rows.append(
            {
                "unit": c["unit"],
                "index": c["index"],
                "stratum": c["stratum"],
                "kind": c["kind"],
                "rtol": m["rtol"],
                "N_cc": m["N_cc"],
                "sum_absE": m["sum_absE"],
                "abs_sumE": m["abs_sumE"],
                "ratio": m["ratio_sum_absE_over_abs_sumE"],
                "sqrt_N_cc": m["sqrt_N_cc"],
                "rms_E": m["rms_E"],
                "abs_sumE_over_sqrtN_rms": m["abs_sumE_over_sqrtN_rms"],
                "sum_R": m["sum_R"],
                "request": m["request"],
                "declared_error": m["declared_error"],
                "signed_combination_error": m["signed_combination_error"],
                "signed_combination_error_linear": m["signed_combination_error_linear"],
                "signed_combination_error_sibling_pairs": m[
                    "signed_combination_error_sibling_pairs"
                ],
                "n_sibling_pairs": m["n_sibling_pairs"],
                "n_unpaired_cc": m["n_unpaired_cc"],
                "converged": m["converged"],
                "max_depth": m["max_depth"],
                "n_floor_limited_cc": m["n_floor_limited_cc"],
                "runs": m["signs_along_call"].get("runs"),
                "runs_expected": m["signs_along_call"].get("runs_expected_if_random"),
                "runs_z": m["signs_along_call"].get("runs_z"),
                "longest_run": m["signs_along_call"].get("longest_run"),
                "n_positive": m["signs_along_call"].get("n_positive"),
                "n_negative": m["signs_along_call"].get("n_negative"),
            }
        )
    with open(ROOT / "out" / "m1_calls.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(m1rows[0]))
        w.writeheader()
        w.writerows(m1rows)
    hdr = (
        "unit",
        "#",
        "N_cc",
        "sum|E|",
        "|sumE|",
        "ratio",
        "sqrtN",
        "|sumE|/(sqrtN rms)",
        "sumR",
        "request",
        "declared",
        "signed",
        "signed(pairs)",
        "runs/exp (z)",
    )
    print(" | ".join(hdr))
    for m in m1rows:
        print(
            f"{m['unit']} | {m['index']} | {m['N_cc']} | {m['sum_absE']:.3e} | {m['abs_sumE']:.3e} | "
            f"{m['ratio'] if m['ratio'] is None else round(m['ratio'], 2)} | {m['sqrt_N_cc']:.1f} | "
            f"{m['abs_sumE_over_sqrtN_rms'] if m['abs_sumE_over_sqrtN_rms'] is None else round(m['abs_sumE_over_sqrtN_rms'], 3)} | "
            f"{m['sum_R']:.3e} | {m['request']:.3e} | {m['declared_error']:.3e} | {m['signed_combination_error']:.3e} | "
            f"{m['signed_combination_error_sibling_pairs']:.3e} | {m['runs']}/{m['runs_expected'] and round(m['runs_expected'], 1)} "
            f"({m['runs_z'] and round(m['runs_z'], 2)})"
        )
    z2 = z2r["M1"]
    print(
        f"\nZ2's call: N_cc {z2['N_cc']}, sum|E| {z2['sum_absE']:.6e}, |sum E| {z2['abs_sumE']:.6e}, ratio "
        f"{z2['ratio_sum_absE_over_abs_sumE']:.4g}, sqrt N {z2['sqrt_N_cc']:.2f}; sum R {z2['sum_R']:.4e}"
    )
    print(
        f"   request max(atol, rtol|v|) = {z2['request']:.6e} (atol {z2['atol']:.6e}, rtol|v| {z2['rtol'] * abs(z2['value']):.6e}); "
        f"declared {z2['declared_error']:.6e}"
    )
    print(
        f"   signed-combination error {z2['signed_combination_error']:.6e} (leafwise: each leaf's floor without |E|, max'd "
        f"with its resolution term as the code does, summed, plus |sum E|); linear form abserr - sum|E| + |sum E| "
        f"{z2['signed_combination_error_linear']:.6e}; with |E| per sibling pair {z2['signed_combination_error_sibling_pairs']:.6e} "
        f"({z2['n_sibling_pairs']} pairs, {z2['n_unpaired_cc']} unpaired); the rest alone {z2['rest_of_error_without_absE']:.6e}"
    )
    print(f"   signs along the call: {z2['signs_along_call']}")
    print("   by depth (returned CC leaves): depth n sum|E| |sumE|")
    zd = []
    for d, v in z2["by_depth"].items():
        print(f"      {d} {v['n']} {v['sum_absE']:.4e} {v['abs_sumE']:.4e}")
        zd.append({"depth": int(d), **v})
    with open(ROOT / "out" / "m1_z2_by_depth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(zd[0]))
        w.writeheader()
        w.writerows(zd)
    print(
        f"   subtrees (all-CC) against their root's own |E| ({z2['n_subtrees_compared']} subtrees): root depth: n, median leaves, "
        f"median sum|E|/|E_root| [p10, p90], median |sumE|/|E_root|"
    )
    zs = []
    for d, v in z2["subtrees_vs_root"].items():
        print(
            f"      {d}: {v['n_subtrees']}, {v['median_n_leaves']}, {v['median_sum_absE_over_root_absE']:.3g} "
            f"[{v['p10_sum_absE_over_root_absE']:.3g}, {v['p90_sum_absE_over_root_absE']:.3g}], {v['median_abs_sumE_over_root_absE']:.3g}"
        )
        zs.append({"root_depth": int(d), **v})
    with open(ROOT / "out" / "m1_z2_subtrees.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(zs[0]))
        w.writeheader()
        w.writerows(zs)
    # the largest subtrees: the root and its first levels
    big = sorted(z2.get("subtree_rows", []), key=lambda r: -r["n_leaves"])[:10]
    print(
        "   the ten largest all-CC subtrees: root depth, leaves, |E_root|, sum|E|, |sumE|"
    )
    for r in big:
        print(
            f"      {r['root_depth']} {r['n_leaves']} {r['root_absE']:.3e} {r['sub_sum_absE']:.3e} {r['sub_abs_sumE']:.3e}"
        )

    F1 = z2["sum_absE"] >= 10 * z2["abs_sumE"]
    F2 = z2["signed_combination_error"] <= z2["request"]
    F2_lin = z2["signed_combination_error_linear"] <= z2["request"]
    big_calls = [m for m in m1rows if m["N_cc"] >= 100]
    big_ratios = [
        m["ratio"] if m["ratio"] is not None else float("inf") for m in big_calls
    ]
    med_big = statistics.median(big_ratios) if big_ratios else None
    big_noz2 = [m["ratio"] for m in big_calls if (m["unit"], m["index"]) != Z2]
    med_big_noz2 = statistics.median(big_noz2) if big_noz2 else None
    F3 = "undecided" if len(big_calls) < 5 else ("holds" if med_big >= 10 else "fails")
    print(
        f"\nF1: sum|E| {z2['sum_absE']:.4e} >= 10 |sum E| = {10 * z2['abs_sumE']:.4e}? -> {'holds' if F1 else 'fails'}"
    )
    print(
        f"F2: signed-combination error {z2['signed_combination_error']:.4e} <= request {z2['request']:.4e}? -> "
        f"{'holds' if F2 else 'fails'} (linear form {z2['signed_combination_error_linear']:.4e}: {'holds' if F2_lin else 'fails'})"
    )
    print(
        f"F3: {len(big_calls)} sampled node-dominated calls with N_cc >= 100 (Z2's included); ratios "
        f"{sorted(round(x, 2) for x in big_ratios)}; median {med_big:.4g}; without Z2's call median {med_big_noz2:.4g} -> {F3}"
    )

    # ---------------- M2
    print("\n== M2")
    m2rows, leafrows = [], []
    for c in sample["M2"]:
        r = results[f"{c['unit']}_{c['index']}"]
        m = r["M2"]
        m2rows.append(
            {
                "unit": c["unit"],
                "index": c["index"],
                "block": c["notify_label"],
                "rtol": m["rtol"],
                "n_leaves": m["n_leaves"],
                "n_cc": m["n_cc"],
                "D": m["D"],
                "D_cc": m["D_cc"],
                "D_levin": m["D_levin"],
                "dI_plus": m["dI_plus"],
                "dI_minus": m["dI_minus"],
                "dI_plus_half_resolution": m["dI_plus_at_half_resolution"],
                "dI_converged_rel": m["dI_converged_rel"],
                "max_s_abs_dI": m["max_s_abs_dI"],
                "sum_abs_dI_leaf": m["sum_abs_dI_leaf"],
                "D_over_max_abs_dI": m["D_over_max_abs_dI"],
                "sum_abs_dI_leaf_over_max_abs_dI": m["sum_abs_dI_leaf_over_max_abs_dI"],
                "dI_declared_rule_plus": m["dI_declared_rule_plus"],
                "dI_measured": m["dI_measured"],
                "D_honest": m["D_honest"],
                "D_honest_over_max_abs_dI": m["D_honest_over_max_abs_dI"],
                "D_over_D_honest": m["D_over_D_honest"],
                "request": m["request"],
                "declared_error": m["declared_error"],
                "total_phase_abs": m["total_phase_abs"],
                "n_nodes": m["n_nodes"],
                "max_abs_delta_theta": m["max_abs_delta_theta"],
                "median_abs_delta_theta": m["median_abs_delta_theta"],
                "max_ratio": m["max_ratio_delta_theta_over_theta_abserr"],
                "median_ratio": m["median_ratio"],
                "n_nodes_violating": m["n_nodes_violating"],
                "theta_abserr_min": m["theta_abserr_range"][0],
                "theta_abserr_max": m["theta_abserr_range"][1],
                "dense_max_ratio": m["dense"]["max_ratio"],
                "dense_n_violating": m["dense"]["n_violating"],
                "nu": "/".join(str(x) for x in m["group"]["nu"]),
                "has_near": "/".join(str(int(x)) for x in m["group"]["has_near"]),
            }
        )
        for L in m["per_leaf"]:
            leafrows.append(
                {
                    "unit": c["unit"],
                    "index": c["index"],
                    **{k: v for k, v in L.items() if k != "dI_history_plus"},
                }
            )
    for name, rr_ in (("m2_calls.csv", m2rows), ("m2_leaves.csv", leafrows)):
        keys = []
        for r in rr_:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(ROOT / "out" / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rr_)
    print(
        "unit | # | block | leaves (cc) | D | |dI| | sum|dI_l| | D/|dI| | sum/|dI| | D_honest/|dI| | dI_meas | request | declared | "
        "|theta(b)-theta(a)| | max|dth| | med|dth| | max ratio | med ratio | conv"
    )
    for m in m2rows:
        print(
            f"{m['unit']} | {m['index']} | {m['block']} | {m['n_leaves']} ({m['n_cc']}) | {m['D']:.3e} | {m['max_s_abs_dI']:.3e} | "
            f"{m['sum_abs_dI_leaf']:.3e} | {m['D_over_max_abs_dI']:.3g} | {m['sum_abs_dI_leaf_over_max_abs_dI']:.3g} | "
            f"{m['D_honest_over_max_abs_dI']:.3g} | {m['dI_measured']:.2e} | {m['request']:.3e} | {m['declared_error']:.3e} | "
            f"{m['total_phase_abs']:.3g} | {m['max_abs_delta_theta']:.2e} | {m['median_abs_delta_theta']:.2e} | "
            f"{m['max_ratio']:.3g} | {m['median_ratio']:.3g} | {m['dI_converged_rel']:.1e}"
        )
    F4_viol = sum(m["n_nodes_violating"] for m in m2rows)
    F4 = F4_viol == 0
    r5 = [m["D_over_max_abs_dI"] for m in m2rows]
    r6 = [m["sum_abs_dI_leaf_over_max_abs_dI"] for m in m2rows]
    med5, med6 = statistics.median(r5), statistics.median(r6)
    conv_worst = max(m["dI_converged_rel"] for m in m2rows)
    print(
        f"\nF4: nodes sampled {sum(m['n_nodes'] for m in m2rows)} (every node of every returned leaf); violating "
        f"{F4_viol}; largest |dtheta|/theta_abserr {max(m['max_ratio'] for m in m2rows):.4g} -> {'holds' if F4 else 'fails'}"
        f" (dense descriptive sample: {sum(m['dense_n_violating'] for m in m2rows)} of {1000 * len(m2rows)} violate; largest "
        f"ratio {max(m['dense_max_ratio'] for m in m2rows):.4g})"
    )
    print(
        f"F5: D/max_s|dI| sorted {sorted(round(x, 3) for x in r5)}; median {med5:.4g} -> {'holds' if med5 >= 10 else 'fails'}"
    )
    print(
        f"F6: sum|dI_l|/max_s|dI| sorted {sorted(round(x, 4) for x in r6)}; median {med6:.4g} -> {'holds' if med6 >= 10 else 'fails'}"
    )
    print(
        f"item 3 convergence: worst relative change on doubling the resolution {conv_worst:.2e} (two figures need < 5e-3)"
    )
    print(
        f"descriptive: D_honest/max|dI| median {statistics.median(m['D_honest_over_max_abs_dI'] for m in m2rows):.4g}; "
        f"D/D_honest median {statistics.median(m['D_over_D_honest'] for m in m2rows):.4g} [min {min(m['D_over_D_honest'] for m in m2rows):.4g}, "
        f"max {max(m['D_over_D_honest'] for m in m2rows):.4g}]; |dI_measured|/max|dI| median "
        f"{statistics.median(abs(m['dI_measured']) / m['max_s_abs_dI'] for m in m2rows):.3g}; D/declared error median "
        f"{statistics.median(m['D'] / m['declared_error'] for m in m2rows):.4g}; D/request median "
        f"{statistics.median(m['D'] / m['request'] for m in m2rows):.3g} [max {max(m['D'] / m['request'] for m in m2rows):.3g}]; "
        f"rule response to the declared size vs Gauss-Legendre: worst relative difference "
        f"{max(abs(m['dI_declared_rule_plus'] - m['dI_plus']) / abs(m['dI_plus']) for m in m2rows):.2e}"
    )
    mc = (ROOT / "out" / "cmd" / "31_method_check.txt").read_text()
    mc_pass = "method check: passes" in mc
    print(f"method check (out/cmd/31): {'passes' if mc_pass else 'FAILS'}")
    scores = {
        "F0": "holds" if f0_holds else "fails",
        "F1": "holds" if F1 else "fails",
        "F2": "holds" if F2 else "fails",
        "F3": F3,
        "F4": "holds" if F4 else "fails",
        "F5": "holds" if med5 >= 10 else "fails",
        "F6": "holds" if med6 >= 10 else "fails",
        "numbers": {
            "z2_sum_absE": z2["sum_absE"],
            "z2_abs_sumE": z2["abs_sumE"],
            "z2_ratio": z2["ratio_sum_absE_over_abs_sumE"],
            "z2_signed": z2["signed_combination_error"],
            "z2_signed_linear": z2["signed_combination_error_linear"],
            "z2_signed_pairs": z2["signed_combination_error_sibling_pairs"],
            "z2_request": z2["request"],
            "z2_declared": z2["declared_error"],
            "F3_n": len(big_calls),
            "F3_median": med_big,
            "F3_median_without_z2": med_big_noz2,
            "F4_violations": F4_viol,
            "F5_median": med5,
            "F6_median": med6,
            "method_check": mc_pass,
            "item3_worst_convergence": conv_worst,
        },
    }
    (ROOT / "out" / "scores.json").write_text(json.dumps(scores, indent=1))
    print("\nscores:", json.dumps({k: v for k, v in scores.items() if k != "numbers"}))
    for n in (
        "f0.csv",
        "m1_calls.csv",
        "m1_z2_by_depth.csv",
        "m1_z2_subtrees.csv",
        "m2_calls.csv",
        "m2_leaves.csv",
        "scores.json",
    ):
        print(f"wrote out/{n} sha256 {sha256(ROOT / 'out' / n)}")


if __name__ == "__main__":
    main()
