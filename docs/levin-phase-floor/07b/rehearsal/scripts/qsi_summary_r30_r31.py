"""Summaries of the R31-against-R30 tabulation (tabulate_qsi_r30_r31.py's two CSVs), for stop 1.

levin-phase-floor 07b, rehearsal agent, prompt 07b section 5 item 3. 07a's rehearsal agent's
qsi_summary_r29_r30.py (cc7bc729...), with the store tags 29 -> 30 and 30 -> 31, the tree path and
this docstring changed, and one new function, `additions_07b` (called at the end of main), which
adds: the |dtotal|/abserr bands on rows whose total moved; regions per moved row and by (k, q, r);
max-depth changes and the rows at depth 20; the analytic and group records' regions; the group
records whose value moved, against both abserrs; and the group records' abserr ratios. Reads only
the two CSVs.

    python -B qsi_summary_r30_r31.py <rows csv> <records csv>
"""

import csv
import hashlib
import subprocess
import sys
from collections import Counter
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows_p, recs_p = sys.argv[1:3]
    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    print(f"# input {rows_p} sha256 {sha256(rows_p)}")
    print(f"# input {recs_p} sha256 {sha256(recs_p)}")
    t = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-0a103b5"
    head = subprocess.run(["git", "--no-optional-locks", "-C", t, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    print(f"# tree {t} HEAD {head}")
    rows = list(csv.DictReader(open(rows_p)))
    recs = list(csv.DictReader(open(recs_p)))
    T = lambda s: s == "True"
    key = lambda r: (r["k"], r["q"], r["r"], r["z_response"])
    print(f"moved rows: {len(rows)}; Levin records differing: {len(recs)}")
    what = Counter()
    for r in rows:
        what[(not T(r["total_bit_identical"]), not T(r["total_abserr_bit_identical"]),
              not T(r["analytic_rad_bit_identical"]))] += 1
    print("moved rows by (total moved, total_abserr moved, analytic_rad moved):")
    for k, v in sorted(what.items()):
        print(f"  {k}: {v}")
    tm = [r for r in rows if not T(r["total_bit_identical"])]
    print(f"rows whose total moved: {len(tm)}")
    for r in sorted(tm, key=lambda r: -float(r["abs_dtotal_over_abserr30"])):
        print(f"  {key(r)} |dT|/e30 {float(r['abs_dtotal_over_abserr30']):.3e} |dT|/e31 {float(r['abs_dtotal_over_abserr31']):.3e} "
              f"S3 {float(r['s3_abs_dtotal_over_sum_abserr']):.3e} e30 {r['total_abserr_30']} e31 {r['total_abserr_31']} "
              f"reg {r['WKB_Levin_num_regions_30']}->{r['WKB_Levin_num_regions_31']} depth {r['WKB_Levin_max_depth_30']}->{r['WKB_Levin_max_depth_31']} "
              f"conv {r['total_converged_30']}->{r['total_converged_31']} grp sp/wp/gr 30 {r['group_stationary_points_30']}/{r['group_window_pieces_30']}/{r['group_guarded_regions_30']} "
              f"31 {r['group_stationary_points_31']}/{r['group_window_pieces_31']}/{r['group_guarded_regions_31']}")
    ea = [r for r in rows if T(r["total_bit_identical"]) and not T(r["total_abserr_bit_identical"])]
    print(f"rows whose total_abserr moved with total bit-identical: {len(ea)}")
    print("rows whose regions or maximum depth changed:")
    for r in rows:
        if r["WKB_Levin_num_regions_30"] != r["WKB_Levin_num_regions_31"] or r["WKB_Levin_max_depth_30"] != r["WKB_Levin_max_depth_31"]:
            print(f"  {key(r)} reg {r['WKB_Levin_num_regions_30']}->{r['WKB_Levin_num_regions_31']} depth "
                  f"{r['WKB_Levin_max_depth_30']}->{r['WKB_Levin_max_depth_31']} total moved {not T(r['total_bit_identical'])} "
                  f"|dT|/e30 {float(r['abs_dtotal_over_abserr30']):.3e} conv {r['total_converged_30']}->{r['total_converged_31']}")
    print("Levin records whose four counts changed:")
    gc = Counter()
    for x in recs:
        c29 = tuple(int(x[f"{g}_30"]) for g in ("stationary_points", "window_pieces", "guarded_regions", "guard_at_depth_max"))
        c30 = tuple(int(x[f"{g}_31"]) for g in ("stationary_points", "window_pieces", "guarded_regions", "guard_at_depth_max"))
        if c29 != c30:
            gc[(x["kind"], x["record"].split(".", 1)[1] if x["kind"] == "analytic" else x["label"], c29, c30,
                x["converged_30"], x["converged_31"])] += 1
    for k, v in sorted(gc.items()):
        print(f"  {k}: {v}")
    print("group records with value bit-identical and abserr moved (no location in either):")
    n = 0
    for x in recs:
        f = set(x["fields_differing"].split())
        if x["kind"] == "group" and "value" not in f and "abserr" in f:
            n += 1
            a, b = float.fromhex(x["abserr_30"]), float.fromhex(x["abserr_31"])
            print(f"  {key(x)} {x['record']} {x['label']} fields {sorted(f)} abserr {x['abserr_30']} -> {x['abserr_31']} "
                  f"rel {(b - a) / a:+.3e} regions {x['regions_30']}->{x['regions_31']} sp {x['stationary_points_30']}/{x['stationary_points_31']}")
    print(f"  count: {n}")
    print("group records whose value moved:")
    for x in recs:
        if x["kind"] == "group" and "value" in x["fields_differing"].split():
            print(f"  {key(x)} {x['record']} {x['label']} |dv|/e30 {float(x['abs_dvalue_over_abserr30']):.3e} "
                  f"|dv|/e31 {float(x['abs_dvalue_over_abserr31']):.3e} regions {x['regions_30']}->{x['regions_31']} "
                  f"sp {x['stationary_points_30']}/{x['stationary_points_31']} wp {x['window_pieces_30']}/{x['window_pieces_31']} "
                  f"conv {x['converged_30']}->{x['converged_31']} pl {x['phase_limited_30']}->{x['phase_limited_31']}")
    print("phase_limited flips (records):")
    pf = Counter()
    for x in recs:
        if x["phase_limited_30"] != x["phase_limited_31"]:
            pf[(x["kind"], x["record"].split(".", 1)[1] if x["kind"] == "analytic" else x["label"],
                x["phase_limited_30"], x["phase_limited_31"])] += 1
    for k, v in sorted(pf.items()):
        print(f"  {k}: {v}")
    print("analytic block records by (block, regions changed, abserr ratio band):")
    bc = Counter()
    for x in recs:
        if x["kind"] != "analytic":
            continue
        a, b = float.fromhex(x["abserr_30"]), float.fromhex(x["abserr_31"])
        ratio = b / a if a else float("inf")
        band = ("x<0.5" if ratio < 0.5 else "0.5<=x<1" if ratio < 1 else "1" if ratio == 1 else "1<x<=2" if ratio <= 2 else "x>2")
        bc[(x["record"].split(".", 1)[1], x["regions_30"] != x["regions_31"], band)] += 1
    for k, v in sorted(bc.items()):
        print(f"  {k}: {v}")
    additions_07b(rows, recs)


def additions_07b(rows, recs):
    """07b additions (new): aggregates for the stop-1 report."""
    import math
    T = lambda s: s == "True"
    print()
    print("== 07b additions")
    tm = [r for r in rows if not T(r["total_bit_identical"])]
    for col in ("abs_dtotal_over_abserr30", "abs_dtotal_over_abserr31", "s3_abs_dtotal_over_sum_abserr"):
        xs = sorted(float(r[col]) for r in tm)
        bands = Counter()
        for x in xs:
            bands["<1e-6" if x < 1e-6 else "<1e-4" if x < 1e-4 else "<1e-2" if x < 1e-2 else "<0.1" if x < 0.1 else "<1" if x < 1 else ">=1"] += 1
        med = xs[len(xs) // 2] if xs else float("nan")
        print(f"rows whose total moved ({len(xs)}): {col} median {med:.3e} max {xs[-1] if xs else float('nan'):.3e} bands {dict(sorted(bands.items()))}")
    reg = [(int(r["WKB_Levin_num_regions_30"]), int(r["WKB_Levin_num_regions_31"])) for r in rows
           if r["WKB_Levin_num_regions_30"] not in ("", "None")]
    up = sum(b > a for a, b in reg); dn = sum(b < a for a, b in reg); eq = sum(b == a for a, b in reg)
    print(f"moved rows with WKB_Levin regions: {len(reg)}; R31 more {up}, fewer {dn}, equal {eq}; "
          f"summed over moved rows {sum(a for a, b in reg)} -> {sum(b for a, b in reg)}")
    ratios = sorted(b / a for a, b in reg if a > 0 and b != a)
    if ratios:
        print(f"  per-row regions ratio where changed: min {ratios[0]:.3f} median {ratios[len(ratios)//2]:.3f} max {ratios[-1]:.3f}")
    dep = Counter((int(r["WKB_Levin_max_depth_30"]), int(r["WKB_Levin_max_depth_31"])) for r in rows
                  if r["WKB_Levin_max_depth_30"] not in ("", "None") and r["WKB_Levin_max_depth_30"] != r["WKB_Levin_max_depth_31"])
    print(f"  max-depth changes (R30 -> R31: rows): {dict(sorted(dep.items()))}")
    for st in ("30", "31"):
        print(f"  rows at depth 20, R{st}: " + "; ".join(
            str((r["k"], r["q"], r["r"], r["z_response"])) for r in rows if r[f"WKB_Levin_max_depth_{st}"] == "20"))
    # by k triple
    byk = {}
    for r in rows:
        kk = (r["k"], r["q"], r["r"])
        d = byk.setdefault(kk, [0, 0, 0, 0])
        d[0] += 1
        d[1] += not T(r["total_bit_identical"])
        if r["WKB_Levin_num_regions_30"] not in ("", "None"):
            d[2] += int(r["WKB_Levin_num_regions_30"]); d[3] += int(r["WKB_Levin_num_regions_31"])
    print("moved rows by (k, q, r): rows moved, total moved, WKB regions R30 -> R31")
    for kk, d in sorted(byk.items(), key=lambda t: tuple(float(x) for x in t[0])):
        print(f"  {kk}: {d[0]}, {d[1]}, {d[2]} -> {d[3]}")
    # analytic block regions
    an = [(int(x["regions_30"]), int(x["regions_31"])) for x in recs if x["kind"] == "analytic"]
    print(f"analytic block records differing: {len(an)}; regions summed {sum(a for a, b in an)} -> {sum(b for a, b in an)}")
    gr = [(int(x["regions_30"]), int(x["regions_31"])) for x in recs if x["kind"] == "group"]
    print(f"group records differing: {len(gr)}; regions summed {sum(a for a, b in gr)} -> {sum(b for a, b in gr)}")
    gv = [x for x in recs if x["kind"] == "group" and "value" in x["fields_differing"].split()]
    xs = sorted(float(x["abs_dvalue_over_abserr30"]) for x in gv)
    ys = sorted(float(x["abs_dvalue_over_abserr31"]) for x in gv)
    if xs:
        print(f"group records whose value moved: {len(gv)}; |dv|/e30 median {xs[len(xs)//2]:.3e} max {xs[-1]:.3e}; "
              f"|dv|/e31 median {ys[len(ys)//2]:.3e} max {ys[-1]:.3e}; |dv|/e30 > 1: {sum(x > 1 for x in xs)}; |dv|/e31 > 1: {sum(y > 1 for y in ys)}")
        for x in sorted(gv, key=lambda x: -max(float(x["abs_dvalue_over_abserr30"]), float(x["abs_dvalue_over_abserr31"])))[:10]:
            print(f"  top: {(x['k'], x['q'], x['r'], x['z_response'])} {x['record']} {x['label']} |dv|/e30 {float(x['abs_dvalue_over_abserr30']):.3e} "
                  f"|dv|/e31 {float(x['abs_dvalue_over_abserr31']):.3e} regions {x['regions_30']}->{x['regions_31']} conv {x['converged_30']}->{x['converged_31']}")
    both = [x for x in gv if float(x["abs_dvalue_over_abserr30"]) * float.fromhex(x["abserr_30"])
            > float.fromhex(x["abserr_30"]) + float.fromhex(x["abserr_31"])]
    print(f"group records whose |dv| exceeds abserr30 + abserr31 (record-level S3 form, float arithmetic): {len(both)}")
    for x in sorted(both, key=lambda x: -abs(float.fromhex(x["value_31"]) - float.fromhex(x["value_30"]))
                    / (float.fromhex(x["abserr_30"]) + float.fromhex(x["abserr_31"]))):
        d = abs(float.fromhex(x["value_31"]) - float.fromhex(x["value_30"]))
        print(f"  {(x['k'], x['q'], x['r'], x['z_response'])} {x['record']} {x['label']} |dv|/(e30+e31) "
              f"{d / (float.fromhex(x['abserr_30']) + float.fromhex(x['abserr_31'])):.3e} v30 {x['value_30']} v31 {x['value_31']} "
              f"e30 {x['abserr_30']} e31 {x['abserr_31']} regions {x['regions_30']}->{x['regions_31']} conv {x['converged_30']}->{x['converged_31']}")
    cd = [x for x in recs if x["converged_30"] != x["converged_31"]]
    print(f"Levin records whose converged changed: {len(cd)}")
    for x in cd:
        print(f"  {(x['k'], x['q'], x['r'], x['z_response'])} {x['kind']} {x['record']} {x['label']} conv {x['converged_30']}->{x['converged_31']} "
              f"v {x['value_30']} -> {x['value_31']} e {x['abserr_30']} -> {x['abserr_31']} regions {x['regions_30']}->{x['regions_31']} "
              f"pl {x['phase_limited_30']}->{x['phase_limited_31']} sp {x['stationary_points_30']}/{x['stationary_points_31']} wp {x['window_pieces_30']}/{x['window_pieces_31']}")
    ab = sorted(float.fromhex(x["abserr_31"]) / float.fromhex(x["abserr_30"]) for x in recs
                if x["kind"] == "group" and float.fromhex(x["abserr_30"]) > 0)
    if ab:
        print(f"group records differing: abserr31/abserr30 min {ab[0]:.3e} median {ab[len(ab)//2]:.3e} max {ab[-1]:.3e}; "
              f"<1 {sum(a < 1 for a in ab)}, =1 {sum(a == 1 for a in ab)}, >1 {sum(a > 1 for a in ab)}")


if __name__ == "__main__":
    main()
