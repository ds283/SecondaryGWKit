"""Summaries of the R30-against-R29 tabulation (tabulate_qsi_r29_r30.py's two CSVs), for stop 1.

levin-phase-floor 07a, rehearsal agent, prompt 07a section 5 item 3. New. Reads only the two CSVs.
Prints: the moved rows by what moved; the |dtotal|/abserr distribution on rows whose total moved;
the Levin records whose location or guard counts changed (R29's Rule-A-skipped blocks, newly
located at R30); the group records whose abserr moved with their value bit-identical; the rows whose
regions or maximum depth changed; and the phase_limited flips.

    ./venv/bin/python -I -B qsi_summary_r29_r30.py <rows csv> <records csv>
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
    t = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-4bf50d3"
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
    for r in sorted(tm, key=lambda r: -float(r["abs_dtotal_over_abserr29"])):
        print(f"  {key(r)} |dT|/e29 {float(r['abs_dtotal_over_abserr29']):.3e} |dT|/e30 {float(r['abs_dtotal_over_abserr30']):.3e} "
              f"S3 {float(r['s3_abs_dtotal_over_sum_abserr']):.3e} e29 {r['total_abserr_29']} e30 {r['total_abserr_30']} "
              f"reg {r['WKB_Levin_num_regions_29']}->{r['WKB_Levin_num_regions_30']} depth {r['WKB_Levin_max_depth_29']}->{r['WKB_Levin_max_depth_30']} "
              f"conv {r['total_converged_29']}->{r['total_converged_30']} grp sp/wp/gr 29 {r['group_stationary_points_29']}/{r['group_window_pieces_29']}/{r['group_guarded_regions_29']} "
              f"30 {r['group_stationary_points_30']}/{r['group_window_pieces_30']}/{r['group_guarded_regions_30']}")
    ea = [r for r in rows if T(r["total_bit_identical"]) and not T(r["total_abserr_bit_identical"])]
    print(f"rows whose total_abserr moved with total bit-identical: {len(ea)}")
    print("rows whose regions or maximum depth changed:")
    for r in rows:
        if r["WKB_Levin_num_regions_29"] != r["WKB_Levin_num_regions_30"] or r["WKB_Levin_max_depth_29"] != r["WKB_Levin_max_depth_30"]:
            print(f"  {key(r)} reg {r['WKB_Levin_num_regions_29']}->{r['WKB_Levin_num_regions_30']} depth "
                  f"{r['WKB_Levin_max_depth_29']}->{r['WKB_Levin_max_depth_30']} total moved {not T(r['total_bit_identical'])} "
                  f"|dT|/e29 {float(r['abs_dtotal_over_abserr29']):.3e} conv {r['total_converged_29']}->{r['total_converged_30']}")
    print("Levin records whose four counts changed:")
    gc = Counter()
    for x in recs:
        c29 = tuple(int(x[f"{g}_29"]) for g in ("stationary_points", "window_pieces", "guarded_regions", "guard_at_depth_max"))
        c30 = tuple(int(x[f"{g}_30"]) for g in ("stationary_points", "window_pieces", "guarded_regions", "guard_at_depth_max"))
        if c29 != c30:
            gc[(x["kind"], x["record"].split(".", 1)[1] if x["kind"] == "analytic" else x["label"], c29, c30,
                x["converged_29"], x["converged_30"])] += 1
    for k, v in sorted(gc.items()):
        print(f"  {k}: {v}")
    print("group records with value bit-identical and abserr moved (no location in either):")
    n = 0
    for x in recs:
        f = set(x["fields_differing"].split())
        if x["kind"] == "group" and "value" not in f and "abserr" in f:
            n += 1
            a, b = float.fromhex(x["abserr_29"]), float.fromhex(x["abserr_30"])
            print(f"  {key(x)} {x['record']} {x['label']} fields {sorted(f)} abserr {x['abserr_29']} -> {x['abserr_30']} "
                  f"rel {(b - a) / a:+.3e} regions {x['regions_29']}->{x['regions_30']} sp {x['stationary_points_29']}/{x['stationary_points_30']}")
    print(f"  count: {n}")
    print("group records whose value moved:")
    for x in recs:
        if x["kind"] == "group" and "value" in x["fields_differing"].split():
            print(f"  {key(x)} {x['record']} {x['label']} |dv|/e29 {float(x['abs_dvalue_over_abserr29']):.3e} "
                  f"|dv|/e30 {float(x['abs_dvalue_over_abserr30']):.3e} regions {x['regions_29']}->{x['regions_30']} "
                  f"sp {x['stationary_points_29']}/{x['stationary_points_30']} wp {x['window_pieces_29']}/{x['window_pieces_30']} "
                  f"conv {x['converged_29']}->{x['converged_30']} pl {x['phase_limited_29']}->{x['phase_limited_30']}")
    print("phase_limited flips (records):")
    pf = Counter()
    for x in recs:
        if x["phase_limited_29"] != x["phase_limited_30"]:
            pf[(x["kind"], x["record"].split(".", 1)[1] if x["kind"] == "analytic" else x["label"],
                x["phase_limited_29"], x["phase_limited_30"])] += 1
    for k, v in sorted(pf.items()):
        print(f"  {k}: {v}")
    print("analytic block records by (block, regions changed, abserr ratio band):")
    bc = Counter()
    for x in recs:
        if x["kind"] != "analytic":
            continue
        a, b = float.fromhex(x["abserr_29"]), float.fromhex(x["abserr_30"])
        ratio = b / a if a else float("inf")
        band = ("x<0.5" if ratio < 0.5 else "0.5<=x<1" if ratio < 1 else "1" if ratio == 1 else "1<x<=2" if ratio <= 2 else "x>2")
        bc[(x["record"].split(".", 1)[1], x["regions_29"] != x["regions_30"], band)] += 1
    for k, v in sorted(bc.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
