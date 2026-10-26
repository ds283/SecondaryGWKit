"""Prompt 05 §5.4: tabulate the QuadSourceIntegral differences between R9 (parent) and R10 (code).

Reads every store file sqlite `mode=ro`, and store_value_diff's JSON. Rows are paired by their
physical (k, q, r, z_response): k, q and r are each `wavenumber.k_inv_Mpc` through
`wavenumber_exit_time`, and z_response is `redshift.z`. Both stores have one atol and one rtol.
The pairing is checked against the tool's identity-matched report in three ways. The tool's
864 identities must give the same (k, q, r, z_response) set. Every `total` the tool reports as
differing must have the same `float.hex` on both sides here. Every row the tool does not report
for `total` must be bit-identical here.

Per row, it writes |Δtotal|/|total_A|, |Δtotal|/total_abserr_A and |Δtotal|/total_abserr_B, the
change in `total_converged`, `total_phase_limited`, `WKB_Levin_num_regions` and
`WKB_Levin_max_depth`, and which metadata keys differ. Metadata keys are split into wall-clock
keys (ending `elapsed` or `compute_time`), reported only as a count, and the rest, which are
named. A = R9 (parent e648951), B = R10 (code 120fdfe).

    ./venv/bin/python -I tabulate_qsi.py <R9 primary> <R10 primary> <store_value_diff.json> <out.csv>
"""

import csv
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path

COLS = ("total", "total_abserr", "total_converged", "total_phase_limited", "numeric_quad",
        "WKB_Levin", "analytic_rad", "WKB_Levin_num_regions", "WKB_Levin_max_depth",
        "WKB_Levin_evaluations")


def read(primary):
    primary = Path(primary)
    con = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
    shards = [r[0] for r in con.execute("SELECT filename FROM shards ORDER BY serial")]
    con.close()
    out = {}
    for name in shards:
        con = sqlite3.connect(f"file:{primary.parent / Path(name).name}?mode=ro", uri=True)
        kq = ("(SELECT w.k_inv_Mpc FROM wavenumber_exit_time e JOIN wavenumber w "
              "ON w.serial = e.wavenumber_serial WHERE e.serial = q.{})")
        sql = (f"SELECT {kq.format('k_wavenumber_exit_serial')}, "
               f"{kq.format('q_wavenumber_exit_serial')}, {kq.format('r_wavenumber_exit_serial')}, "
               "(SELECT z FROM redshift WHERE serial = q.z_response_serial), "
               "(SELECT log10_tol FROM tolerance WHERE serial = q.atol_serial), "
               "(SELECT log10_tol FROM tolerance WHERE serial = q.rtol_serial), "
               + ", ".join(f'q."{c}"' for c in COLS) + " FROM QuadSourceIntegral q")
        for row in con.execute(sql):
            key = row[:4]
            assert key not in out, f"duplicate {key}"
            out[key] = dict(zip(("atol", "rtol") + COLS, row[4:]))
        con.close()
    return out


def hx(x):
    return None if x is None else float(x).hex()


def timing_key(k):
    return k.endswith("elapsed") or k.endswith("compute_time")


def main():
    a_path, b_path, report_path, out_csv = sys.argv[1:5]
    A, B = read(a_path), read(b_path)
    print(f"rows: A {len(A)}, B {len(B)}; same (k,q,r,z_response) set: {set(A) == set(B)}")
    print(f"tolerances A: {sorted(set((v['atol'], v['rtol']) for v in A.values()))}; "
          f"B: {sorted(set((v['atol'], v['rtol']) for v in B.values()))}")

    rep = json.load(open(report_path))["classes"]["QuadSourceIntegral"]["rows"]
    tool = {}
    for e in rep["rows"]:
        k = e["key"]
        key = (k["k"]["k"]["k_inv_Mpc"], k["q"]["k"]["k_inv_Mpc"], k["r"]["k"]["k_inv_Mpc"],
               k["z_response"]["z"])
        assert key not in tool
        tool[key] = e["columns"]
    print(f"tool: {len(tool)} rows differ, {rep['identical']} identical; "
          f"same key set as the stores: {set(tool) == set(A)}")
    mism = 0
    for key in A:
        t = tool.get(key, {}).get("total")
        if t is None:
            mism += hx(A[key]["total"]) != hx(B[key]["total"])
        else:
            mism += (t["a"], t["b"]) != (hx(A[key]["total"]), hx(B[key]["total"]))
    print(f"cross-check of `total` against the tool: {mism} mismatches")
    print()

    rows = []
    meta_named = Counter()
    for key in sorted(A):
        a, b = A[key], B[key]
        d = b["total"] - a["total"]
        meta = tool.get(key, {}).get("metadata", {"keys": []})["keys"]
        named = sorted(m for m in meta if not timing_key(m))
        meta_named.update(named)
        rows.append({
            "k": key[0], "q": key[1], "r": key[2], "z_response": key[3],
            "total_A": hx(a["total"]), "total_B": hx(b["total"]),
            "total_bit_identical": hx(a["total"]) == hx(b["total"]),
            "rel_dtotal": abs(d) / abs(a["total"]) if a["total"] else math.inf,
            "dtotal_over_abserr_A": abs(d) / a["total_abserr"] if a["total_abserr"] else None,
            "dtotal_over_abserr_B": abs(d) / b["total_abserr"] if b["total_abserr"] else None,
            "abserr_A": a["total_abserr"], "abserr_B": b["total_abserr"],
            "converged_A": a["total_converged"], "converged_B": b["total_converged"],
            "phase_limited_A": a["total_phase_limited"], "phase_limited_B": b["total_phase_limited"],
            "regions_A": a["WKB_Levin_num_regions"], "regions_B": b["WKB_Levin_num_regions"],
            "max_depth_A": a["WKB_Levin_max_depth"], "max_depth_B": b["WKB_Levin_max_depth"],
            "numeric_quad_identical": hx(a["numeric_quad"]) == hx(b["numeric_quad"]),
            "analytic_rad_identical": hx(a["analytic_rad"]) == hx(b["analytic_rad"]),
            "WKB_Levin_identical": hx(a["WKB_Levin"]) == hx(b["WKB_Levin"]),
            "metadata_keys_named": ";".join(named),
            "metadata_timing_keys": sum(1 for m in meta if timing_key(m)),
        })
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    ch = [r for r in rows if not r["total_bit_identical"]]
    print(f"`total` differs in {len(ch)} of {n} rows; bit-identical in {n - len(ch)}")
    print(f"numeric_quad identical in {sum(r['numeric_quad_identical'] for r in rows)}/{n}; "
          f"analytic_rad identical in {sum(r['analytic_rad_identical'] for r in rows)}/{n}; "
          f"WKB_Levin identical in {sum(r['WKB_Levin_identical'] for r in rows)}/{n}")
    print(f"rows with total differing but WKB_Levin identical: "
          f"{sum(1 for r in ch if r['WKB_Levin_identical'])}")

    def q(vals, p):
        s = sorted(vals)
        return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]

    for name in ("rel_dtotal", "dtotal_over_abserr_A", "dtotal_over_abserr_B"):
        v = [r[name] for r in ch if r[name] is not None]
        print(f"{name} over the {len(v)} differing rows: median {q(v, .5):.3e}, "
              f"90% {q(v, .9):.3e}, max {max(v):.3e}")
    for thr in (1.0, 0.1):
        print(f"rows with |dtotal|/abserr_A > {thr}: "
              f"{sum(1 for r in ch if r['dtotal_over_abserr_A'] and r['dtotal_over_abserr_A'] > thr)}; "
              f"/abserr_B > {thr}: "
              f"{sum(1 for r in ch if r['dtotal_over_abserr_B'] and r['dtotal_over_abserr_B'] > thr)}")
    print()
    print("transitions (A -> B), all rows:")
    for f in ("converged", "phase_limited", "max_depth"):
        c = Counter((r[f + "_A"], r[f + "_B"]) for r in rows)
        print(f"  {f}: " + ", ".join(f"{x}->{y}: {m}" for (x, y), m in sorted(c.items(), key=str)))
    rc = [r for r in rows if r["regions_A"] != r["regions_B"]]
    print(f"  regions: {len(rc)} rows change; total regions A {sum(r['regions_A'] or 0 for r in rows)}, "
          f"B {sum(r['regions_B'] or 0 for r in rows)} (None counted as 0)")
    for r in sorted(rc, key=lambda r: -abs((r["regions_B"] or 0) - (r["regions_A"] or 0)))[:42]:
        print(f"    k={r['k']:.4g} q={r['q']:.4g} r={r['r']:.4g} z_resp={r['z_response']:.6g}: "
              f"regions {r['regions_A']}->{r['regions_B']}, depth {r['max_depth_A']}->"
              f"{r['max_depth_B']}, converged {r['converged_A']}->{r['converged_B']}, "
              f"|dtotal|/abserr_A {r['dtotal_over_abserr_A']:.3g}")
    print()
    print("non-wall-clock metadata keys that differ, by number of rows (top 30):")
    for k, m in meta_named.most_common(30):
        print(f"  {m:4d}  {k}")
    print(f"distinct non-wall-clock keys differing: {len(meta_named)}; rows with any: "
          f"{sum(1 for r in rows if r['metadata_keys_named'])}; rows with wall-clock keys only: "
          f"{sum(1 for r in rows if not r['metadata_keys_named'] and r['metadata_timing_keys'])}")
    print()
    print("the 15 rows of largest |dtotal|/abserr_A:")
    for r in sorted(ch, key=lambda r: -(r["dtotal_over_abserr_A"] or 0))[:15]:
        print(f"  k={r['k']:.4g} q={r['q']:.4g} r={r['r']:.4g} z_resp={r['z_response']:.6g}: "
              f"rel {r['rel_dtotal']:.3e}, /abserr_A {r['dtotal_over_abserr_A']:.3e}, "
              f"/abserr_B {r['dtotal_over_abserr_B']:.3e}, depth {r['max_depth_A']}->{r['max_depth_B']}, "
              f"regions {r['regions_A']}->{r['regions_B']}, converged {r['converged_A']}->{r['converged_B']}")
    print()
    print("by max depth under A: rows, rows where total differs, median and max |dtotal|/abserr_A")
    for dep in sorted(set(r["max_depth_A"] for r in rows), key=lambda x: (x is None, x)):
        g = [r for r in rows if r["max_depth_A"] == dep]
        gc = [r["dtotal_over_abserr_A"] for r in g if not r["total_bit_identical"]
              and r["dtotal_over_abserr_A"] is not None]
        print(f"  depth {dep}: {len(g)} rows, {len(gc)} differ"
              + (f", median {q(gc, .5):.3e}, max {max(gc):.3e}" if gc else ""))


if __name__ == "__main__":
    main()
