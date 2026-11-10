"""Summaries of out/qsi_rows.csv and out/qsi_guard_records.csv (R28 against R27, the touched rows).

levin-phase-floor 07, rehearsal agent, prompt 07 section 5.4. New. Reads only the two CSVs written by
tabulate_qsi_guard.py and qsi_guard_records.py. Reports: the touched rows by what located (group
G+q+r, group q-r, analytic only, both), their total_converged in each store, the rows still
unconverged in R28, and the largest |dtotal|/total_abserr per class, with the rows above 1 counted.

    ./venv/bin/python -I -B qsi_rows_summary.py <qsi_rows.csv> <qsi_guard_records.csv>
"""

import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows_path, recs_path = sys.argv[1:3]
    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    for p in (rows_path, recs_path):
        print(f"# input {p} sha256 {sha256(p)}")
    rows = list(csv.DictReader(open(rows_path)))
    recs = list(csv.DictReader(open(recs_path)))
    key = lambda r: (r["k"], r["q"], r["r"], r["z_response"])
    labels = defaultdict(set)
    for r in recs:
        labels[key(r)].add(r["label"] if r["kind"] == "group" else "analytic")
    cls = Counter()
    by = defaultdict(list)
    for r in rows:
        s = labels[key(r)]
        c = "+".join(sorted(s))
        cls[c] += 1
        by[c].append(r)
    print(f"touched rows: {len(rows)}; by what located: {dict(cls)}")
    for c, lst in sorted(by.items()):
        t27 = sum(r["total_converged_27"] == "True" for r in lst)
        t28 = sum(r["total_converged_28"] == "True" for r in lst)
        r27 = [float(r["abs_dtotal_over_abserr27"]) for r in lst]
        r28 = [float(r["abs_dtotal_over_abserr28"]) for r in lst]
        bit = sum(r["total_bit_identical"] == "True" for r in lst)
        print(f"  {c}: {len(lst)} rows; total_converged R27 {t27}, R28 {t28}; total bit-identical {bit}; "
              f"max |dT|/e27 {max(r27):.6g}, max |dT|/e28 {max(r28):.6g}; rows with |dT| > e27: "
              f"{sum(x > 1 for x in r27)}, > e28: {sum(x > 1 for x in r28)}, > both: "
              f"{sum(a > 1 and b > 1 for a, b in zip(r27, r28))}")
    group_rows = [r for r in rows if r["group_stationary_points"] != "0"]
    print(f"rows in which a group call located a point (the stationary-point rows of the group calls): {len(group_rows)}; "
          f"total_converged R27 {sum(r['total_converged_27'] == 'True' for r in group_rows)}, "
          f"R28 {sum(r['total_converged_28'] == 'True' for r in group_rows)}")
    print("touched rows unconverged in R28:")
    for r in rows:
        if r["total_converged_28"] != "True":
            print(f"  {key(r)} located by {sorted(labels[key(r)])}; R27 converged {r['total_converged_27']}; "
                  f"unconverged group calls 27/28 {r['group_calls_unconverged_27']}/{r['group_calls_unconverged_28']}; "
                  f"max depth 27/28 {r['WKB_Levin_max_depth_27']}/{r['WKB_Levin_max_depth_28']}")
    print("touched rows ordered by max(|dT|/e27, |dT|/e28), top 20:")
    for r in sorted(rows, key=lambda r: -max(float(r["abs_dtotal_over_abserr27"]), float(r["abs_dtotal_over_abserr28"])))[:20]:
        print(f"  {key(r)} {sorted(labels[key(r)])} |dT|/e27 {float(r['abs_dtotal_over_abserr27']):.4g} "
              f"|dT|/e28 {float(r['abs_dtotal_over_abserr28']):.4g} T27 {r['total_27']} T28 {r['total_28']} "
              f"e27 {r['total_abserr_27']} e28 {r['total_abserr_28']} conv {r['total_converged_27']}->{r['total_converged_28']}")


if __name__ == "__main__":
    main()
