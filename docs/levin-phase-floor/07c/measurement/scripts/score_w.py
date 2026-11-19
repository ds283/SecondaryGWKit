"""levin-phase-floor prompt 06j, Part W: score W1-W4 and fill section 3's tables, by
state/predictions_and_rules_20261103T0905Z.md sections 1, 5 and 6 (written before any reading). New in 06j. Reads
only, except out/w_scores.json, out/w_calls.csv and out/w_tables.md.

For each W call (f00-f09, c00-c11): its reference readings (out/readings/<ref sid>-<impl>-r<res>.json; the W-5
1e-8 calls f05-f09 use their 1e-7 twins' readings f00-f04, whose integrands are equal, out/w_prep_v2/summary.json);
each reading's usable clause (tmap.all_monotone, or every non-monotone segment integrated directly:
direct_segments > 0 with all_monotone False) and whether Ray was initialised; the spread (largest minus smallest
value_30, tools.filon_reference.scatter) against the tolerance min(e_g, e_06h)/100; the reference, the FCC r2
reading; |v - ref|/e for both codes in mpmath at 40 digits; and from out/w_prep_v2/<sid>.json what each code
returned and the constant move.

Usage: PYTHONPATH=<tree>:<this directory> python -B score_w.py
"""

import csv
import json
import sys
from pathlib import Path

import mpmath as mp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

mp.mp.dps = 40
IMPL_RES = (("qawo", 1), ("qawo", 2), ("fcc", 1), ("fcc", 2))


def main():
    import tools.filon_reference as F

    sel = {e["sid"]: e for e in json.loads((RC.ROOT / "out" / "sample" / "selection.json").read_text())}
    wsam = json.loads((RC.ROOT / "out" / "sample" / "w_sample.json").read_text())
    wsum = json.loads((RC.ROOT / "out" / "w_prep_v2" / "summary.json").read_text())
    twin = {}
    for row, eq in wsum["w5_equality"].items():
        a, b = eq["pair"]
        if not eq["one_reference_serves_both"]:
            raise SystemExit(f"{row}: the rtol twins are not equal")
        twin[a] = a
        twin[b] = a
    sids = [x["sid"] for x in wsam["w5"]] + [x["sid"] for x in wsam["w12"]]
    rfiles = sorted((RC.ROOT / "out" / "readings").glob("*.json"))
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06j: Part W scored (W1-W4) and section 3's tables",
                 [RC.ROOT / "out" / "sample" / "selection.json", RC.ROOT / "out" / "w_prep_v2" / "summary.json"] + rfiles)
    for l in hdr:
        print(l)
    calls = []
    for sid in sids:
        e = sel[sid]
        ref_sid = twin.get(sid, sid)
        rd = {}
        for impl, res in IMPL_RES:
            p = RC.ROOT / "out" / "readings" / f"{ref_sid}-{impl}-r{res}.json"
            if not p.exists():
                rd[f"{impl}-r{res}"] = None
                continue
            d = json.loads(p.read_text())
            tm = d.get("tmap") or {}
            usable = bool(tm.get("all_monotone")) or (tm.get("all_monotone") is False and (tm.get("direct_segments") or 0) > 0)
            rd[f"{impl}-r{res}"] = {"value_30": d["value_30"], "value_hex": d["value_hex"], "imag_30": d["imag_30"],
                                    "wall_s": d["wall_s"], "all_monotone": tm.get("all_monotone"),
                                    "direct_segments": tm.get("direct_segments"), "usable": usable,
                                    "ray_initialized": d.get("ray_initialized"), "tool_sha256": d.get("tool_sha256"),
                                    "pickle_sha256": d.get("pickle_sha256"), "n_segments": d.get("n_segments")}
        have = [r for r in rd.values() if r is not None and r["usable"]]
        g, h = e["g_record"], e["06h_record"]
        vg, eg = mp.mpf(float.fromhex(g["value"]["hex"])), mp.mpf(float.fromhex(g["abserr"]["hex"]))
        vh, eh = mp.mpf(float.fromhex(h["value"]["hex"])), mp.mpf(float.fromhex(h["abserr"]["hex"]))
        tol = min(eg, eh) / 100
        spread = mp.mpf(F.scatter(have)) if len(have) >= 2 else None
        ref = rd.get("fcc-r2")
        out = {"sid": sid, "ref_sid": ref_sid, "part": e["part"], "unit": e["unit"], "index": e["index"],
               "group": e["g_record"].get("group") or e["notify_label"], "rtol": e["rtol"], "readings": rd,
               "n_usable": len(have), "tolerance": float(tol), "spread": None if spread is None else float(spread),
               "spread_over_tol": None if spread is None else float(spread / tol),
               "four_agree": len(have) == 4 and spread is not None and spread <= tol,
               "v_g": g["value"]["repr"], "e_g": g["abserr"]["repr"], "g_regions": g["num_regions"], "g_depth": g["max_depth"],
               "g_converged": g["converged"], "v_06h": h["value"]["repr"], "e_06h": h["abserr"]["repr"],
               "h_regions": h["num_regions"], "h_depth": h["max_depth"], "h_converged": h["converged"]}
        if ref is not None and ref["usable"]:
            R = mp.mpf(ref["value_30"])
            out["ref_30"] = ref["value_30"]
            out["g_ratio"] = float(abs(vg - R) / eg)
            out["h_ratio"] = float(abs(vh - R) / eh)
            out["g_within"] = abs(vg - R) <= eg
            out["h_within"] = abs(vh - R) <= eh
            # each reading's distance in units of each error, for the record
            out["readings_g_ratio"] = {k: float(abs(vg - mp.mpf(r["value_30"])) / eg) for k, r in rd.items() if r}
            out["readings_h_ratio"] = {k: float(abs(vh - mp.mpf(r["value_30"])) / eh) for k, r in rd.items() if r}
        wp = json.loads((RC.ROOT / "out" / "w_prep_v2" / f"{sid}.json").read_text())
        out["g_leaves"] = [{k: L.get(k) for k in ("start", "end", "depth", "convergence_ratio", "trusted", "value_kind", "n_seams")}
                           for L in wp["g"]["leaves"]]
        out["h_leaves"] = [{k: L.get(k) for k in ("start", "end", "depth", "convergence_ratio", "trusted", "n_seams", "value_recomputed")}
                           for L in wp["h06"]["leaves"]]
        out["h_values_reproduced"] = wp["h06"]["recomputed_sum_equals_value"]
        cm = wp["constant_move"]
        big = max(cm["cells"], key=lambda c: abs(c["share_of_move"] or 0)) if cm["cells"] else None
        out["move"] = {"v_g_minus_v_06h": cm["v_g_minus_v_06h"], "localised_90pct": cm["localised_90pct"],
                       "carrying_cell": big, "n_cells": len(cm["cells"])}
        # the move against the reference: which code's partial on the carrying cell is off
        if "ref_30" in out and big is not None:
            out["move"]["g_minus_ref"] = float(vg - mp.mpf(out["ref_30"]))
            out["move"]["h06_minus_ref"] = float(vh - mp.mpf(out["ref_30"]))
        calls.append(out)
    # verdicts
    w = {c["sid"]: c for c in calls}
    w5 = [w[x["sid"]] for x in wsam["w5"]]
    w12 = [w[x["sid"]] for x in wsam["w12"]]
    allc = w5 + w12
    lacking = [c["sid"] for c in allc if c["n_usable"] < 4]
    disagree = [c["sid"] for c in allc if c["n_usable"] == 4 and not c["four_agree"]]
    W1 = "fails" if disagree else ("undecided" if lacking else "holds")
    noref = [c["sid"] for c in allc if "ref_30" not in c]
    W2 = "undecided" if any(c["sid"] in noref for c in w5) else ("holds" if all(c["g_within"] for c in w5) else "fails")
    W3 = "undecided" if any(c["sid"] in noref for c in w5) else ("holds" if all(not c["h_within"] for c in w5) else "fails")
    n_g_in = sum(1 for c in w12 if c.get("g_within"))
    n_h_out = sum(1 for c in w12 if "h_within" in c and not c["h_within"])
    W4 = "undecided" if any(c["sid"] in noref for c in w12) else ("holds" if n_g_in >= 11 and n_h_out >= 6 else "fails")
    verd = {"W1": {"verdict": W1, "lacking_four_usable": lacking, "disagreeing": disagree,
                   "spread_over_tol": {c["sid"]: c["spread_over_tol"] for c in allc}},
            "W2": {"verdict": W2, "g_ratio": {c["sid"]: c.get("g_ratio") for c in w5}},
            "W3": {"verdict": W3, "h_ratio": {c["sid"]: c.get("h_ratio") for c in w5}},
            "W4": {"verdict": W4, "n_g_within": n_g_in, "n_h_outside": n_h_out, "n": len(w12),
                   "g_ratio": {c["sid"]: c.get("g_ratio") for c in w12}, "h_ratio": {c["sid"]: c.get("h_ratio") for c in w12}}}
    for k, v in verd.items():
        print(f"{k}: {v['verdict']}")
    print(f"W1: four usable readings on {sum(1 for c in allc if c['n_usable'] == 4)} of {len(allc)}; agree within tolerance on "
          f"{sum(1 for c in allc if c['four_agree'])}; disagreeing {disagree}; lacking {lacking}")
    print(f"W4: g within on {n_g_in} of {len(w12)}; 06h outside on {n_h_out} of {len(w12)}")
    print("\n| call | unit #index | group | readings usable | spread / tol | ref (FCC r2) | g: |v-ref|/e (regions, depth, conv) | 06h: |v-ref|/e (regions, depth, conv) | move carried by |")
    print("|---|---|---|---|---|---|---|---|---|")
    lines = []
    for c in allc:
        cell = c["move"]["carrying_cell"]
        lines.append(f"| {c['sid']} | {c['unit']} #{c['index']} | {c['group']} | {c['n_usable']} | {c['spread_over_tol']:.3g} | "
                     f"{float(mp.mpf(c['ref_30'])):.10e} | {c['g_ratio']:.4g} ({c['g_regions']}, {c['g_depth']}, {c['g_converged']}) | "
                     f"{c['h_ratio']:.4g} ({c['h_regions']}, {c['h_depth']}, {c['h_converged']}) | "
                     f"{cell['coarse']} leaf [{cell['start']:.6g}, {cell['end']:.6g}], {cell['share_of_move']:.4f} of {c['move']['v_g_minus_v_06h']:.4e} |")
    for l in lines:
        print(l)
    (RC.ROOT / "out" / "w_scores.json").write_text(json.dumps({"header": hdr, "verdicts": verd, "calls": calls}, indent=1, default=str))
    with open(RC.ROOT / "out" / "w_calls.csv", "w", newline="") as f:
        f.write("".join(f"{l}\n" for l in hdr))
        keys = ["sid", "ref_sid", "part", "unit", "index", "group", "rtol", "n_usable", "tolerance", "spread", "spread_over_tol",
                "four_agree", "ref_30", "v_g", "e_g", "g_ratio", "g_within", "g_regions", "g_depth", "g_converged", "v_06h",
                "e_06h", "h_ratio", "h_within", "h_regions", "h_depth", "h_converged"]
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        for c in allc:
            wr.writerow(c)
    (RC.ROOT / "out" / "w_tables.md").write_text("\n".join(["<!-- " + l + " -->" for l in hdr]) + "\n\n" + "\n".join(lines) + "\n")
    print(f"wrote out/w_scores.json sha256 {RC.sha256(RC.ROOT / 'out' / 'w_scores.json')}")


if __name__ == "__main__":
    main()
