"""Y1 references: assemble the per-call and per-row results.

Reads (all under state/): rows.json; the pickling captures capture2_R1{7,8}_*.json; moved.json; and,
per call, ref/<row>-<idx>.panels.json, .ref.json, .cmp17.json (and .twin.json for a call whose
reference is its twin's). Writes results.json (every number in hex as well) and a markdown table
file, and prints both.

Per moved call with a reference: |value - ref| / abserr for each code, and "within" iff
|value - ref| <= abserr; the reference's accuracy is |S_high - S_low| against
min(abserr17, abserr18) / 100.

Per row: the reference total is formed exactly (fractions.Fraction) as the store forms the total,
numeric_quad + sum over sub-intervals and groups of (1 + z_response) * value, with each moved call
that has a reference at its reference value and every unmoved call at its (shared) value. A moved
call without a reference is taken at R17's value and, separately, at R18's: two reference totals.
The verdict is given when both agree on it; otherwise "undetermined". As a check, the same sum
with every call at R17's (R18's) own value is compared with the stored T17 (T18).

Usage: python -B assemble.py OUT_JSON OUT_MD
"""

import glob
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from yprov import Y1, header  # noqa: E402

STATE = Y1 / "state"
REF = STATE / "ref"


def fx(h):
    return float.fromhex(h)


def hx(x):
    return None if x is None else float(x).hex()


def region_at(regions, x):
    for r in regions:
        if fx(r["start"]) <= x <= fx(r["end"]):
            return r
    return None


def main():
    out_json, out_md = sys.argv[1:3]
    inputs = sorted(glob.glob(str(STATE / "capture2_*.json"))) + [str(STATE / "rows.json"), str(STATE / "moved.json")]
    hdr = header(Path(__file__).resolve(), "Y1 references: assembled results", [Path(p) for p in inputs])
    for l in hdr:
        print(l)
    rows = {r["id"]: r for r in json.loads((STATE / "rows.json").read_text())}
    cap = {"R17": {}, "R18": {}}
    for t in cap:
        for p in sorted(glob.glob(str(STATE / f"capture2_{t}_*.json"))):
            for r in json.loads(Path(p).read_text())["rows"]:
                cap[t][r["row"]] = r
    moved = {(e["row"], e["index"]) for e in json.loads((STATE / "moved.json").read_text())}

    def load_ref(tag):
        p = REF / f"{tag}.ref.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        d["_panels"] = json.loads((REF / f"{tag}.panels.json").read_text())
        c = REF / f"{tag}.cmp17.json"
        d["_cmp17"] = json.loads(c.read_text()) if c.exists() else None
        return d

    results = {"header": hdr, "rows": []}
    md = []
    md.append("| row | k, q, r (/Mpc) | z_resp | moved calls (ref'd / moved) | call | R17 |v-ref|/abserr | R18 |v-ref|/abserr | ref accuracy (conv / min abserr) | row: R17 |T-Tref|/e17 | R18 |T-Tref|/e18 | verdict |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for rid in sorted(rows):
        R = rows[rid]
        a, b = cap["R17"].get(rid), cap["R18"].get(rid)
        if a is None or b is None:
            continue
        scale = 1.0 + fx(a["z_response"])
        assert a["z_response"] == b["z_response"]
        Fs = Fraction(scale)
        calls = []
        sum17 = Fraction(fx(a["numeric_quad"]))
        sum18 = Fraction(fx(b["numeric_quad"]))
        ref_lo = Fraction(fx(a["numeric_quad"]))  # uncomputed moved calls at R17's value
        ref_hi = Fraction(fx(b["numeric_quad"]))  # ... at R18's
        assert a["numeric_quad"] == b["numeric_quad"]
        ref_err = Fraction(0)
        n_moved = n_ref = 0
        for ca, cb in zip(a["calls"], b["calls"]):
            assert (ca["index"], ca["label"], ca["x_span"]) == (cb["index"], cb["label"], cb["x_span"])
            v17, e17, v18, e18 = fx(ca["value"]), fx(ca["abserr"]), fx(cb["value"]), fx(cb["abserr"])
            sum17 += Fs * Fraction(v17)
            sum18 += Fs * Fraction(v18)
            key = (rid, ca["index"])
            if key not in moved:
                assert (ca["value"], ca["abserr"]) == (cb["value"], cb["abserr"])
                ref_lo += Fs * Fraction(v17)
                ref_hi += Fs * Fraction(v17)
                continue
            n_moved += 1
            tag = f"{rid}-{ca['index']:02d}"
            twin = None
            tw = REF / f"{tag}.twin.json"
            if tw.exists():
                twin = json.loads(tw.read_text())
                d = load_ref(twin["twin_of"]) if twin["r18_identical"] and twin["r17_identical"] else None
            else:
                d = load_ref(tag)
            cr = {"row": rid, "index": ca["index"], "label": ca["label"], "sub": ca["sub"], "x_span": ca["x_span"],
                  "R17": {"value": ca["value"], "abserr": ca["abserr"], "num_regions": ca["num_regions"], "max_depth": ca["max_depth"],
                          "converged": ca["converged"], "phase_limited": ca["phase_limited"]},
                  "R18": {"value": cb["value"], "abserr": cb["abserr"], "num_regions": cb["num_regions"], "max_depth": cb["max_depth"],
                          "converged": cb["converged"], "phase_limited": cb["phase_limited"]},
                  "twin_of": twin["twin_of"] if twin else None}
            if d is None:
                cr["ref"] = None
                ref_lo += Fs * Fraction(v17)
                ref_hi += Fs * Fraction(v18)
                calls.append(cr)
                continue
            n_ref += 1
            ref = fx(d["ref"])
            conv = fx(d["conv_diff"])
            emin = min(e17, e18)
            cr.update({
                "ref": d["ref"], "ref_conv_diff": d["conv_diff"], "ref_rounding_bound_linear": d["rounding_bound_linear"],
                "ref_conv_over_min_abserr": conv / emin if emin > 0 else None, "ref_accurate": conv <= emin / 100,
                "ref_nodes": d["nodes"], "ref_panels": d["npanels"], "cancellation": d["cancellation"],
                "R17_err_over_abserr": abs(v17 - ref) / e17 if e17 > 0 else None, "R17_within": abs(v17 - ref) <= e17,
                "R18_err_over_abserr": abs(v18 - ref) / e18 if e18 > 0 else None, "R18_within": abs(v18 - ref) <= e18,
                "R17_err": hx(abs(v17 - ref)), "R18_err": hx(abs(v18 - ref)),
                "replay_R18": d["replay"], "replay_R17": (d["_cmp17"] or {}).get("replay"),
                "f_theta_bit_identical_R17_R18": (d["_cmp17"] or {}).get("all_bit_identical") if not twin else twin["r17_identical"],
                "sample_nodes": (d["_cmp17"] or {}).get("columns", {}).get("f_sin", {}).get("nodes"),
                "theta_deriv_consistency": d["theta_deriv_consistency"],
                "zeros_of_theta_deriv": d["_panels"]["zeros_of_theta_deriv"],
                "theta_deriv_max_abs": d["_panels"]["theta_deriv_max_abs"],
                "theta_deriv_at_ends": d["_panels"]["theta_deriv_at_ends"],
                "oscillations_total_variation": d["_panels"]["oscillations_total_variation"],
            })
            # the regions around a zero of theta'
            ns = []
            for zr in d["_panels"]["zeros_of_theta_deriv"]:
                x0 = zr["x"]
                ent = {"x": zr["x_hex"], "z": zr["z"], "theta_second_deriv": zr["theta_second_deriv"],
                       "stationary_width_x": zr["stationary_width_x"]}
                for t, c in (("R17", ca), ("R18", cb)):
                    rg = region_at(c["regions"], x0)
                    w = fx(rg["end"]) - fx(rg["start"])
                    ent[t] = {"start": rg["start"], "end": rg["end"], "width": w, "width_over_stationary_width": w / zr["stationary_width_x"],
                              "depth": rg["depth"], "type": rg["type"], "abserr": rg["abserr"], "phase_err": rg["phase_err"],
                              "total_err": rg["total_err"],
                              "regions_within_10_widths": sum(1 for r in c["regions"] if fx(r["end"]) >= x0 - 10 * zr["stationary_width_x"]
                                                              and fx(r["start"]) <= x0 + 10 * zr["stationary_width_x"])}
                ns.append(ent)
            cr["stationary_points"] = ns
            ref_lo += Fs * Fraction(ref)
            ref_hi += Fs * Fraction(ref)
            ref_err += Fs * Fraction(conv)
            calls.append(cr)
        T17, e17r, T18, e18r = fx(R["R17"]["total"]), fx(R["R17"]["total_abserr"]), fx(R["R18"]["total"]), fx(R["R18"]["total_abserr"])
        row = {"id": rid, "rank": R["rank"], "Y1_ratio": R["ratio"], "k": R["k"], "q": R["q"], "r": R["r"], "z_response": R["z_response"],
               "equilateral": R["equilateral"], "T17": R["R17"]["total"], "e17": R["R17"]["total_abserr"], "T18": R["R18"]["total"],
               "e18": R["R18"]["total_abserr"], "moved_calls": n_moved, "referenced_calls": n_ref,
               "sum_check_R17": float(abs(sum17 - Fraction(T17)) / abs(Fraction(T17))),
               "sum_check_R18": float(abs(sum18 - Fraction(T18)) / abs(Fraction(T18))),
               "Tref_uncomputed_at_R17": hx(float(ref_lo)), "Tref_uncomputed_at_R18": hx(float(ref_hi)),
               "Tref_error": hx(float(ref_err)), "calls": calls}
        verdicts = []
        for Tref in (ref_lo, ref_hi):
            w17 = abs(Fraction(T17) - Tref) <= Fraction(e17r)
            w18 = abs(Fraction(T18) - Tref) <= Fraction(e18r)
            verdicts.append("both" if w17 and w18 else "R17 within" if w17 else "R18 within" if w18 else "neither")
        row["R17_row_err_over_e17"] = [float(abs(Fraction(T17) - t) / Fraction(e17r)) for t in (ref_lo, ref_hi)]
        row["R18_row_err_over_e18"] = [float(abs(Fraction(T18) - t) / Fraction(e18r)) for t in (ref_lo, ref_hi)]
        row["verdict"] = verdicts[0] if verdicts[0] == verdicts[1] else "undetermined"
        row["verdict_pair"] = verdicts
        results["rows"].append(row)
        # markdown
        first = True
        for c in calls:
            def r_(x):
                return "-" if x is None else f"{x:.3g}"
            if c["ref"] is None:
                ctext = f"{c['index']}:{c['label']} (no ref)"
                cols = ["-", "-", "-"]
            else:
                ctext = f"{c['index']}:{c['label']}" + (f" (= {c['twin_of']})" if c["twin_of"] else "")
                cols = [r_(c["R17_err_over_abserr"]), r_(c["R18_err_over_abserr"]), r_(c["ref_conv_over_min_abserr"])]
            if first:
                rc = row["R17_row_err_over_e17"]
                rr = row["R18_row_err_over_e18"]
                rtxt = [f"{rc[0]:.3g}" if abs(rc[0] - rc[1]) <= 1e-3 * max(rc) else f"{rc[0]:.3g} / {rc[1]:.3g}",
                        f"{rr[0]:.3g}" if abs(rr[0] - rr[1]) <= 1e-3 * max(rr) else f"{rr[0]:.3g} / {rr[1]:.3g}", f"**{row['verdict']}**"]
                md.append(f"| {rid} | {R['k']:.4g}, {R['q']:.4g}, {R['r']:.4g} | {R['z_response']:.6g} | {n_ref} / {n_moved} | {ctext} | "
                          + " | ".join(cols) + " | " + " | ".join(rtxt) + " |")
                first = False
            else:
                md.append(f"| | | | | {ctext} | " + " | ".join(cols) + " | | | |")
    Path(out_json).write_text(json.dumps(results, indent=1))
    Path(out_md).write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print()
    for row in results["rows"]:
        print(f"{row['id']}: verdict {row['verdict']} {row['verdict_pair']}; R17 |T-Tref|/e17 {row['R17_row_err_over_e17']}; "
              f"R18 |T-Tref|/e18 {row['R18_row_err_over_e18']}; Tref err {fx(row['Tref_error']):.3g} vs e17 {fx(row['e17']):.3g} e18 {fx(row['e18']):.3g}; "
              f"sum checks {row['sum_check_R17']:.2g} {row['sum_check_R18']:.2g}")
        for c in row["calls"]:
            for sp in c.get("stationary_points", []):
                print(f"   {c['index']}:{c['label']} theta'=0 at z={sp['z']:.6g} (x {fx(sp['x']):.6f}), width {sp['stationary_width_x']:.3g}; "
                      f"max|theta'| {c['theta_deriv_max_abs']:.3g}; R17 region depth {sp['R17']['depth']} width {sp['R17']['width']:.3g} "
                      f"({sp['R17']['width_over_stationary_width']:.3g} widths) abserr {sp['R17']['abserr']}; R18 region depth {sp['R18']['depth']} "
                      f"width {sp['R18']['width']:.3g} ({sp['R18']['width_over_stationary_width']:.3g} widths) abserr {sp['R18']['abserr']}")
    print(f"wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
