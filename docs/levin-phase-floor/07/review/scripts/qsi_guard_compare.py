"""Orchestrator's own read of QuadSourceIntegral between R27 (A, parent) and R28 (B, guard).

Usage: python -I -B qsi_guard_compare.py A_PRIMARY B_PRIMARY OUT_CSV

levin-phase-floor 07 Part B, planned stop 1 and review check 4. Read only (sqlite mode=ro).

Rows are read exactly as qsi_full_compare.py (06m's orchestrator, imported from this directory):
keyed by (k, q, r, z_response, log10 atol, log10 rtol); every column but store bookkeeping and
timing; `metadata` flattened leaf by leaf, timing leaves dropped.

B's count leaves are the leaves whose last key is one of the four counts prompt 07 section 1.6
adds. Each Levin record in B (a WKB_Levin group record, or an analytic *_data record under
`.Levin.`) must carry all four. A row is UNTOUCHED iff every count leaf in it is the integer 0.
For an untouched row every other field must equal A's by float.hex / type-and-value. For every
other row the change in total is written to OUT_CSV against both stores' total_abserr.
"""

import collections
import csv
import hashlib
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qsi_full_compare import read, same, generic  # noqa: E402

COUNTS = ("stationary_points", "window_pieces", "guarded_regions", "guard_at_depth_max")


def lastkey(path):
    return re.split(r"[.\[]", path)[-1].rstrip("]")


def record_prefix(path):
    return path.rsplit(".", 1)[0]


def main():
    a_p, b_p, out_csv = sys.argv[1:4]
    me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(f"# qsi_guard_compare.py sha256 {me}")
    for p in (a_p, b_p):
        print(f"# {p} sha256 {hashlib.sha256(Path(p).read_bytes()).hexdigest()}")
    a, b = read(a_p), read(b_p)
    print(f"rows A {len(a)} B {len(b)}; in one only {len(set(a) ^ set(b))}")

    n_untouched = n_untouched_identical = 0
    untouched_diffs = collections.Counter()
    touched_rows = []
    count_problems = 0
    group_recs = analytic_recs = 0
    touched_group_recs = touched_analytic_recs = 0
    count_totals = collections.Counter()
    conv = collections.Counter()
    for key in sorted(a):
        x, y = a[key], b[key]
        cnt = {f: v for f, v in y.items() if f.startswith("metadata.") and lastkey(f) in COUNTS}
        # every Levin record in B carries all four counts
        recs = collections.defaultdict(dict)
        for f, v in cnt.items():
            recs[record_prefix(f)][lastkey(f)] = v
        # Levin records in A: group records and analytic *_data records (by their 'regions' leaf)
        a_recs = {
            record_prefix(f)
            for f in x
            if f.startswith("metadata.")
            and lastkey(f) == "regions"
            and (".groups[" in f or ".Levin." in f)
        }
        if set(recs) != a_recs or any(set(r) != set(COUNTS) for r in recs.values()):
            count_problems += 1
            if count_problems <= 5:
                print("COUNT LAYOUT PROBLEM", key, sorted(set(recs) ^ a_recs)[:4])
        bad_type = [f for f, v in cnt.items() if type(v) is not int]
        if bad_type:
            count_problems += 1
            print("NON-INT COUNT", key, bad_type[:3])
        for pfx, r in recs.items():
            is_group = ".groups[" in pfx
            group_recs += is_group
            analytic_recs += not is_group
            if any(r.get(c) for c in COUNTS):
                touched_group_recs += is_group
                touched_analytic_recs += not is_group
            for c in COUNTS:
                count_totals[(("group" if is_group else "analytic"), c)] += r.get(c) or 0
        untouched = all(v == 0 for v in cnt.values())
        y_rest = {f: v for f, v in y.items() if f not in cnt}
        diff_fields = [
            f for f in set(x) | set(y_rest) if not same(x.get(f, "<absent>"), y_rest.get(f, "<absent>"))
        ]
        if untouched:
            n_untouched += 1
            if not diff_fields:
                n_untouched_identical += 1
            for f in diff_fields:
                untouched_diffs[generic(f)] += 1
            if diff_fields and n_untouched - n_untouched_identical <= 5:
                print("UNTOUCHED ROW DIFFERS", key, sorted(diff_fields)[:6])
        else:
            ta, tb = x["total"], y["total"]
            ea, eb = x["total_abserr"], y["total_abserr"]
            d = abs(Fraction(tb) - Fraction(ta))
            sp = sum(v for f, v in cnt.items() if lastkey(f) == "stationary_points")
            wp = sum(v for f, v in cnt.items() if lastkey(f) == "window_pieces")
            gr = sum(v for f, v in cnt.items() if lastkey(f) == "guarded_regions")
            gd = sum(v for f, v in cnt.items() if lastkey(f) == "guard_at_depth_max")
            sp_an = sum(
                v for f, v in cnt.items() if lastkey(f) == "stationary_points" and ".groups[" not in f
            )
            touched_rows.append(
                dict(
                    k=key[0], q=key[1], r=key[2], z=key[3],
                    stationary_points=sp, stationary_points_analytic=sp_an,
                    window_pieces=wp, guarded_regions=gr, guard_at_depth_max=gd,
                    total_a=ta.hex(), total_b=tb.hex(),
                    abserr_a=ea.hex(), abserr_b=eb.hex(),
                    d_over_abserr_a=float(d / Fraction(ea)) if ea else float("inf"),
                    d_over_abserr_b=float(d / Fraction(eb)) if eb else float("inf"),
                    total_bitident=same(ta, tb),
                    conv_a=x["total_converged"], conv_b=y["total_converged"],
                    regions_a=x.get("WKB_Levin_num_regions"), regions_b=y.get("WKB_Levin_num_regions"),
                    maxdepth_a=x.get("WKB_Levin_max_depth"), maxdepth_b=y.get("WKB_Levin_max_depth"),
                    analytic_bitident=same(x.get("analytic_rad"), y.get("analytic_rad"))
                    and same(x.get("metadata.analytic.abserr"), y.get("metadata.analytic.abserr")),
                    n_fields_differing=len(diff_fields),
                )
            )
        conv[(bool(x["total_converged"]), bool(y["total_converged"]))] += 1

    print(f"Levin records in B: group {group_recs}, analytic {analytic_recs}; count-layout problems {count_problems}")
    print(f"records with a nonzero count: group {touched_group_recs}, analytic {touched_analytic_recs}")
    for (kind, c), v in sorted(count_totals.items()):
        print(f"  sum of {c} over {kind} records: {v}")
    print(f"UNTOUCHED rows {n_untouched}; bit-identical (every field but the counts) {n_untouched_identical}")
    for f, n in sorted(untouched_diffs.items(), key=lambda t: -t[1]):
        print(f"  untouched diff {n:6d}  {f}")
    print(f"OTHER rows {len(touched_rows)}")
    if touched_rows:
        print(f"  total bit-identical {sum(r['total_bitident'] for r in touched_rows)}")
        print(f"  analytic column bit-identical {sum(r['analytic_bitident'] for r in touched_rows)}")
        print(f"  rows with an analytic call that located a point {sum(1 for r in touched_rows if r['stationary_points_analytic'])}")
        print(f"  guarded_regions sum {sum(r['guarded_regions'] for r in touched_rows)}; guard_at_depth_max sum {sum(r['guard_at_depth_max'] for r in touched_rows)}")
        ma = max(touched_rows, key=lambda r: r["d_over_abserr_a"])
        mb = max(touched_rows, key=lambda r: r["d_over_abserr_b"])
        print(f"  largest |dtotal|/abserr_R27 {ma['d_over_abserr_a']:.6g} at {(ma['k'], ma['q'], ma['r'], ma['z'])}")
        print(f"  largest |dtotal|/abserr_R28 {mb['d_over_abserr_b']:.6g} at {(mb['k'], mb['q'], mb['r'], mb['z'])}")
        tc = collections.Counter((r["conv_a"], r["conv_b"]) for r in touched_rows)
        print(f"  total_converged (R27,R28) among other rows: {dict(tc)}")
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(touched_rows[0]))
            w.writeheader()
            w.writerows(touched_rows)
    print(f"total_converged (R27,R28) over all rows: {dict(conv)}")
    for (ca, cb), n in sorted(conv.items()):
        if ca != cb:
            print(f"  changed {ca}->{cb}: {n}")
    gained = [k for k in sorted(a) if not a[k]["total_converged"] and b[k]["total_converged"]]
    lost = [k for k in sorted(a) if a[k]["total_converged"] and not b[k]["total_converged"]]
    print(f"gained {len(gained)}: {gained}")
    print(f"lost {len(lost)}: {lost}")


if __name__ == "__main__":
    main()
