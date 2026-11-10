"""R28 against R27 inside QuadSourceIntegral: every Levin record (group call or analytic block) of R28
with a nonzero guard count, one line each, beside R27's record of the same call.

levin-phase-floor 07, rehearsal agent, prompt 07 section 5.4. New. Descriptive companion to
tabulate_qsi_guard.py (which classes the rows); reuses its reader by import from this directory.
For each such record: the row's physical key, the record's path, label (groups), the subinterval's
z range and regime, the four counts, value (groups only; the analytic blocks keep no value), abserr,
converged, phase_limited and regions in each store, |dv|/abserr in each (groups), and the row's
|dtotal|/total_abserr in each. Writes a CSV and prints summaries by group label and by block.

    ./venv/bin/python -I -B qsi_guard_records.py <R27 primary> <R28 primary> <out csv>
"""

import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabulate_qsi_guard as T  # noqa: E402


def main():
    a_path, b_path, csv_path = sys.argv[1:4]
    print(f"# script {Path(__file__).resolve()} sha256 {T.sha256(__file__)}")
    print(f"# imports tabulate_qsi_guard.py sha256 {T.sha256(T.__file__)}")
    for p in (a_path, b_path):
        prim = Path(p)
        for f in sorted(prim.parent.glob(prim.stem + "*.sqlite")):
            print(f"# input {f} sha256 {T.sha256(f)}")
    A, _ = T.read(a_path)
    B, _ = T.read(b_path)
    out = []
    by_label = defaultdict(Counter)
    by_block = defaultdict(Counter)
    for key in sorted(B):
        a, b = A[key], B[key]
        _, ma, _ = T.fields(a)
        _, mb, _ = T.fields(b)
        ra = {r[1]: r for r in T.levin_records(ma)}
        dT = abs(b["total"] - a["total"])
        for kind, name, rec, sub in T.levin_records(mb):
            if not any(rec[g] for g in T.GUARD_LEAVES):
                continue
            r27 = ra[name][2]
            row = {
                "k": key[0], "q": key[1], "r": key[2], "z_response": key[3], "kind": kind, "record": name,
                "label": rec.get("label", ""),
                "z_min": sub["z_min"] if sub else "", "z_max": sub["z_max"] if sub else "",
                "regime": sub["regime"] if sub else "",
            }
            for g in T.GUARD_LEAVES:
                row[g] = rec[g]
            for tag, x in (("27", r27), ("28", rec)):
                row[f"value_{tag}"] = T.hx(x.get("value"))
                row[f"abserr_{tag}"] = T.hx(x["abserr"])
                row[f"converged_{tag}"] = x["converged"]
                row[f"phase_limited_{tag}"] = x["phase_limited"]
                row[f"regions_{tag}"] = x["regions"]
            if kind == "group":
                dv = abs(rec["value"] - r27["value"])
                row["abs_dv_over_abserr27"] = repr(T.ratio(dv, r27["abserr"]))
                row["abs_dv_over_abserr28"] = repr(T.ratio(dv, rec["abserr"]))
                row["value_bit_identical"] = T.bits(rec["value"]) == T.bits(r27["value"])
            row["row_abs_dtotal_over_abserr27"] = repr(T.ratio(dT, a["total_abserr"]))
            row["row_abs_dtotal_over_abserr28"] = repr(T.ratio(dT, b["total_abserr"]))
            out.append(row)
            c = by_label[rec["label"]] if kind == "group" else by_block[name.split(".", 1)[1]]
            c["records"] += 1
            for g in T.GUARD_LEAVES:
                c[g] += rec[g]
            c["converged_27"] += r27["converged"]
            c["converged_28"] += rec["converged"]
            c["regions_27"] += r27["regions"]
            c["regions_28"] += rec["regions"]
            if kind == "group":
                c["value_bit_identical"] += row["value_bit_identical"]
                for tag in ("27", "28"):
                    x = float(row[f"abs_dv_over_abserr{tag}"])
                    c[f"dv_gt_abserr{tag}"] += (not math.isnan(x)) and x > 1
    print(f"records with a nonzero count: {len(out)}")
    print("group records, by label:")
    for lab, c in sorted(by_label.items()):
        print(f"  {lab}: {dict(c)}")
    print("analytic blocks, by block:")
    for blk, c in sorted(by_block.items()):
        print(f"  {blk}: {dict(c)}")
    g = [r for r in out if r["kind"] == "group"]
    sp = sorted(r["stationary_points"] for r in g)
    if sp:
        print(f"group records: stationary_points per record min {sp[0]}, median {sp[len(sp)//2]}, max {sp[-1]}; "
              f"distribution {sorted(Counter(sp).items())[:40]}")
    reg = Counter((r["label"], str(r["regime"])) for r in g)
    print(f"group records by (label, subinterval regime): {dict(reg)}")
    an = [r for r in out if r["kind"] == "analytic"]
    print(f"analytic blocks: stationary_points per block {sorted(Counter(r['stationary_points'] for r in an).items())}; "
          f"window_pieces per block {sorted(Counter(r['window_pieces'] for r in an).items())}")
    print("group records ordered by |dv|/abserr28, top 15:")
    for r in sorted(g, key=lambda r: -float(r["abs_dv_over_abserr28"]))[:15]:
        print(f"  ({r['k']:.6g}, {r['q']:.6g}, {r['r']:.6g}, z {r['z_response']:.8g}) {r['label']} sp {r['stationary_points']} "
              f"wp {r['window_pieces']} z [{r['z_min']:.6g}, {r['z_max']:.6g}] |dv|/e27 {float(r['abs_dv_over_abserr27']):.4g} "
              f"|dv|/e28 {float(r['abs_dv_over_abserr28']):.4g} conv {r['converged_27']}->{r['converged_28']} "
              f"regions {r['regions_27']}->{r['regions_28']} v27 {r['value_27']} v28 {r['value_28']} e27 {r['abserr_27']} e28 {r['abserr_28']}")
    with open(csv_path, "w", newline="") as f:
        cols = []
        for r in out:
            for k in r:
                if k not in cols:
                    cols.append(k)
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"# wrote {len(out)} records to {csv_path} sha256 {T.sha256(csv_path)}")


if __name__ == "__main__":
    main()
