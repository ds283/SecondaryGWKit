"""levin-phase-floor prompt 06j, Part S: tabulate S-a to S-e from out/partS/<sid>.json.gz and score S1-S4 by the
rules of state/predictions_and_rules_20261103T0905Z.md (sections 1, 3 and 6). New in 06j. Reads only, except its
outputs. Writes out/partS_leaves.csv (one row per sampled leaf), out/partS_summary.json and prints the report.

Usage: python -B score_s.py
"""

import csv
import gzip
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402


def q(v, ps=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)):
    v = sorted(x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not v:
        return "none"
    return ", ".join(f"p{int(p*100)} {v[min(len(v)-1, int(p*(len(v)-1)))]:.4g}" for p in ps)


def main():
    files = sorted((RC.ROOT / "out" / "partS").glob("s*.json.gz"))
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06j: Part S tabulated and scored (S1-S4)", files)
    for l in hdr:
        print(l)
    rows, calls = [], []
    for p in files:
        d = json.loads(gzip.open(p).read())
        calls.append(d)
        for L in d["leaves"]:
            sb = L["S_b"]
            br = {}
            if L["S_a"]["holds_seam"]:
                names = (("A", "seam_branch"), ("B", "off_seam_branch"))
            else:
                names = (("A", "max_branch"), ("B", "min_branch"))
            for tag, nm in names:
                for node in sb[nm]:
                    if "ended" in node:
                        br[f"{tag}_ended_at"] = node["depth"]
                        continue
                    dd = node["depth"]
                    br[f"{tag}_r{dd}"] = node["r"]
                    br[f"{tag}_d1phi{dd}"] = node["d1_over_phi"]
                    br[f"{tag}_trusted{dd}"] = node["trusted_b"]
                    br[f"{tag}_seam{dd}"] = node["seams_held"]
            sc = L["S_c"]
            row = {"sid": d["sid"], "unit": d["unit"], "index": d["index"], "group": d["group"], "rtol": d["rtol"],
                   "start": L["start"], "end": L["end"], "width": L["end"] - L["start"], "cluster": L["cluster"],
                   "r": L["census"]["convergence_ratio"], "delta1": L["census"]["delta1"], "phi": L["census"]["phase_err"],
                   "d1_over_phi": L["census"]["delta1"] / L["census"]["phase_err"],
                   "reproduces_census": L["check"]["reproduces_census"], "holds_seam": L["S_a"]["holds_seam"],
                   "n_seams": L["S_a"]["n_seams"], "seam_kinds": "|".join(L["S_a"]["kinds"]),
                   "dist_end_widths": L["S_a"]["dist_end_widths"], "dist_centre_widths": L["S_a"]["dist_centre_widths"],
                   "null20": d["null_depth20"], "branch_A": names[0][1], "branch_B": names[1][1],
                   "root_is_direct": sb["root"]["is_direct"],
                   "scatter_i": sc["i_nodes"]["rms_V_P"], "scatter_ii": sc["ii_inputs"]["rms_V_P"],
                   "scatter_i_halves": sc["i_nodes"]["rms_halves"], "scatter_ii_halves": sc["ii_inputs"]["rms_halves"],
                   "scatter_i_delta1": sc["i_nodes"]["rms_delta1"], "scatter_ii_delta1": sc["ii_inputs"]["rms_delta1"],
                   "n_changed_i": sc["i_nodes"]["n_V_P_changed"], "n_changed_ii": sc["ii_inputs"]["n_V_P_changed"],
                   "delta1_over_max_scatter": sc["delta1_over_max_scatter"],
                   "max_scatter_over_phi": sc["max_scatter_over_phi"]}
            row.update(br)
            if "S_d" in L:
                row["Sd_all_trusted"] = L["S_d"]["all_trusted"]
                row["Sd_sub"] = json.dumps([{k: s[k] for k in ("width", "r", "d1_over_phi", "trusted_b")} for s in L["S_d"]["sub_spans"]])
            rows.append(row)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(RC.ROOT / "out" / "partS_leaves.csv", "w", newline="") as f:
        f.write("".join(f"{l}\n" for l in hdr))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    n = len(rows)
    ok = [r for r in rows if r["reproduces_census"]]
    excl = n - len(ok)
    out = {"header": hdr, "n_calls": len(calls), "n_leaves": n, "n_reproduce": len(ok)}
    print(f"\nPart S: {len(calls)} calls, {n} leaves; the probe reproduces the census numbers on {len(ok)}; excluded {excl}")
    # S1
    seam = [r for r in ok if r["holds_seam"]]
    frac = len(seam) / len(ok)
    null = sum(r["null20"] for r in ok) / len(ok)
    s1 = "holds" if (frac >= 0.8 and frac >= 5 * null) else "fails"
    if excl > 0.2 * n:
        s1 = "undecided"
    out["S1"] = {"n": len(ok), "n_hold_seam": len(seam), "fraction": frac, "null_weighted": null,
                 "fraction_over_null": frac / null if null else None, "verdict": s1}
    print(f"S1: {len(seam)} of {len(ok)} hold a seam ({frac:.4f}); null {null:.4e} (x{frac/null:.3g}); verdict {s1}")
    noseam = [r for r in ok if not r["holds_seam"]]
    print(f"   distance to the nearest seam (widths), from the nearer end: {q([r['dist_end_widths'] for r in noseam])}")
    print(f"   from the centre: {q([r['dist_centre_widths'] for r in noseam])}")
    comp = [c for d in calls for c in d["comparison_trusted_depth_ge_15"]]
    cfrac = sum(1 for c in comp if c["holds_seam"]) / len(comp) if comp else None
    cnull = sum(c["null_at_depth"] for c in comp) / len(comp) if comp else None
    out["S1_comparison_trusted_depth_ge_15"] = {"n": len(comp), "n_hold_seam": sum(1 for c in comp if c["holds_seam"]),
                                                 "fraction": cfrac, "null_weighted": cnull,
                                                 "by_depth": {str(dd): [sum(1 for c in comp if c["depth"] == dd),
                                                                        sum(1 for c in comp if c["depth"] == dd and c["holds_seam"])]
                                                              for dd in sorted({c["depth"] for c in comp})}}
    print(f"   comparison: the calls' trusted returned leaves at depth >= 15: {len(comp)}, holding a seam "
          f"{out['S1_comparison_trusted_depth_ge_15']['n_hold_seam']} ({cfrac}); null at their depths {cnull}")
    # S2
    excl2 = [r for r in seam if any(f"A_r{d}" not in r for d in (21, 22, 23, 24))]
    den = [r for r in seam if r not in excl2]
    hold2 = [r for r in den if all(0.4 <= r[f"A_r{d}"] <= 0.6 for d in (21, 22, 23, 24))]
    if not seam:
        s2 = "undecided"
    elif len(excl2) > 0.2 * len(seam):
        s2 = "undecided"
    else:
        s2 = "holds" if len(hold2) >= 0.8 * len(den) else "fails"
    out["S2"] = {"n_seam_leaves": len(seam), "excluded_branch_ended": len(excl2), "n": len(den), "n_within": len(hold2),
                 "seam_leaves": [{k: r.get(k) for k in ("sid", "start", "end", "seam_kinds", "r", "A_r21", "A_r22", "A_r23",
                                                       "A_r24", "B_r21", "B_r22", "B_r23", "B_r24", "A_ended_at", "B_ended_at")}
                                 for r in seam], "verdict": s2}
    print(f"S2: seam leaves {len(seam)}, excluded {len(excl2)}, r in [0.4, 0.6] at 21-24 on {len(hold2)} of {len(den)}; verdict {s2}")
    for r in seam:
        print(f"   {r['sid']} [{r['start']!r}, {r['end']!r}] kinds {r['seam_kinds']}: seam branch r "
              + ", ".join(f"{d}: {r.get(f'A_r{d}', float('nan')):.4g}" for d in (21, 22, 23, 24))
              + "; off-seam r " + ", ".join(f"{d}: {r.get(f'B_r{d}', float('nan')):.4g}" for d in (21, 22, 23, 24)))
    # descriptive: the no-seam leaves' branches
    for tag, nm in (("A", "max"), ("B", "min")):
        for d in (20, 21, 22, 23, 24):
            key = "r" if d == 20 else f"{tag}_r{d}"
            v = [r[key] for r in noseam if key in r]
            inside = sum(1 for x in v if 0.4 <= x <= 0.6)
            tr = [r.get(f"{tag}_trusted{d}") for r in noseam if f"{tag}_trusted{d}" in r]
            print(f"   no-seam leaves, {nm} branch, depth {d}: r {q(v, (0.1, 0.5, 0.9))}; in [0.4, 0.6] {inside} of {len(v)}"
                  + (f"; trusted under (b) {sum(1 for x in tr if x)} of {len(tr)}" if tr else ""))
            out.setdefault("S_b_noseam", {})[f"{nm}_{d}"] = {"r_quantiles": q(v), "n_in_04_06": inside, "n": len(v),
                                                             "n_trusted": sum(1 for x in tr if x) if tr else None}
        for d in (21, 22, 23, 24):
            v = [r[f"{tag}_d1phi{d}"] for r in noseam if f"{tag}_d1phi{d}" in r]
            print(f"   no-seam leaves, {nm} branch, depth {d}: delta1/phi {q(v, (0.1, 0.5, 0.9))}")
    stay = [r for r in noseam if all(f"{t}_r{d}" in r and 0.4 <= r[f"{t}_r{d}"] <= 0.6 for t in "AB" for d in (21, 22, 23, 24))]
    out["S_b_noseam_both_branches_in_04_06_all_depths"] = len(stay)
    print(f"   no-seam leaves with r in [0.4, 0.6] at every depth 21-24 on both branches: {len(stay)} of {len(noseam)}")
    # S3
    s3n = [r for r in ok if r["delta1"] > 10 * max(r["scatter_i"], r["scatter_ii"])]
    s3 = "holds" if len(s3n) >= 0.8 * len(ok) else "fails"
    if excl > 0.2 * n:
        s3 = "undecided"
    out["S3"] = {"n": len(ok), "n_delta1_gt_10x_scatter": len(s3n),
                 "delta1_over_max_scatter": q([r["delta1_over_max_scatter"] for r in ok]),
                 "max_scatter_over_phi": q([r["max_scatter_over_phi"] for r in ok]),
                 "scatter_i_over_delta1": q([r["scatter_i"] / r["delta1"] for r in ok]),
                 "scatter_ii_over_delta1": q([r["scatter_ii"] / r["delta1"] for r in ok]),
                 "scatter_halves_i_over_delta1": q([r["scatter_i_halves"] / r["delta1"] for r in ok]),
                 "scatter_delta1_i_over_delta1": q([r["scatter_i_delta1"] / r["delta1"] for r in ok]),
                 "n_draws_changed_i": q([r["n_changed_i"] for r in ok]), "n_draws_changed_ii": q([r["n_changed_ii"] for r in ok]),
                 "n_delta1_gt_max_scatter": sum(1 for r in ok if r["delta1"] > max(r["scatter_i"], r["scatter_ii"])),
                 "verdict": s3}
    print(f"S3: delta1 > 10 x max scatter on {len(s3n)} of {len(ok)}; verdict {s3}")
    for k in ("delta1_over_max_scatter", "max_scatter_over_phi", "scatter_i_over_delta1", "scatter_ii_over_delta1",
              "scatter_halves_i_over_delta1", "scatter_delta1_i_over_delta1", "n_draws_changed_i", "n_draws_changed_ii"):
        print(f"   {k}: {out['S3'][k]}")
    print(f"   delta1 > max scatter (factor 1): {out['S3']['n_delta1_gt_max_scatter']} of {len(ok)}")
    # S4
    sd = [r for r in seam if "Sd_all_trusted" in r]
    s4n = [r for r in sd if r["Sd_all_trusted"]]
    s4 = "undecided" if not sd else ("holds" if len(s4n) >= 0.8 * len(sd) else "fails")
    out["S4"] = {"n": len(sd), "n_all_trusted": len(s4n), "detail": [{"sid": r["sid"], "sub": json.loads(r["Sd_sub"])} for r in sd],
                 "verdict": s4}
    print(f"S4: both sides trusted on {len(s4n)} of {len(sd)}; verdict {s4}")
    for r in sd:
        print(f"   {r['sid']} [{r['start']!r}, {r['end']!r}]: {r['Sd_sub']}")
    # by-group, by-rtol summaries of r and d1/phi for the record
    out["leaves_r"] = q([r["r"] for r in ok])
    out["leaves_d1_over_phi"] = q([r["d1_over_phi"] for r in ok])
    out["leaves_branch"] = {"clenshaw_curtis": sum(1 for r in ok if r["root_is_direct"]), "levin": sum(1 for r in ok if not r["root_is_direct"])}
    print(f"leaves: r {out['leaves_r']}; delta1/phi {out['leaves_d1_over_phi']}; branch {out['leaves_branch']}")
    (RC.ROOT / "out" / "partS_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote out/partS_leaves.csv sha256 {RC.sha256(RC.ROOT / 'out' / 'partS_leaves.csv')}; out/partS_summary.json sha256 "
          f"{RC.sha256(RC.ROOT / 'out' / 'partS_summary.json')}")


if __name__ == "__main__":
    main()
