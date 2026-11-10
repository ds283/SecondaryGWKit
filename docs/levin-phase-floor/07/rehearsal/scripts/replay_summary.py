"""Summaries of the replay's per-call kind and location table (out/replay_calls.csv).

levin-phase-floor 07, rehearsal agent, prompt 07 section 5.4 (located points, kind and z) with the user's
terms at planned stop 1 (the 26 q-r calls summarised, not listed point by point). New. Reads only the CSV.

    ./venv/bin/python -I -B replay_summary.py <replay_calls.csv>
"""

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main():
    src = sys.argv[1]
    print(f"# script {Path(__file__).resolve()} sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}")
    print(f"# input {src} sha256 {hashlib.sha256(Path(src).read_bytes()).hexdigest()}")
    rows = list(csv.DictReader(open(src)))
    calls = defaultdict(list)
    for r in rows:
        calls[(r["k"], r["q"], r["r"], r["z_response"], r["record"])].append(r)
    print(f"lines {len(rows)}; calls {len(calls)}; all reproduce R28: {all(r['reproduces'] == 'True' for r in rows)}")
    by_label = Counter(v[0]["label"] for v in calls.values())
    print(f"calls by label: {dict(by_label)}")
    # q-r
    qr = [v[0] for v in calls.values() if v[0]["label"] == "q-r"]
    print(f"\nq-r calls (q = r): {len(qr)}")
    ok = True
    for r in sorted(qr, key=lambda r: (float(r["k"]), float(r["q"]), float(r["z_response"]))):
        s = json.loads(r["summary"])
        cond = (r["num_stationary_points"] == "4097" and s["all_x_on_scan_grid"] and s["n_distinct_x"] == 4097
                and s["grid_theta_deriv_exact_zeros"] == 4097 and float.fromhex(s["grid_theta_deriv_max_abs"]["hex"]) == 0.0)
        ok &= cond
        zlo, zhi = r["location"].split("..")
        print(f"  ({float(r['k']):.6g}, {float(r['q']):.6g}, {float(r['r']):.6g}, z_resp {float(r['z_response']):.6g}): "
              f"{r['num_stationary_points']} points {r['point_kind']} found_as {r['found_as']}; every scan sample an exact "
              f"zero of theta' ({s['grid_theta_deriv_exact_zeros']} of 4097, max |theta'| {s['grid_theta_deriv_max_abs']['repr']}); "
              f"points on the grid {s['all_x_on_scan_grid']}; theta in [{s['grid_theta_min']['repr']}, {s['grid_theta_max']['repr']}]; "
              f"z {float(zlo):.6g} .. {float(zhi):.6g}; window pieces {r['num_window_pieces']}")
    print(f"  every q-r call: 4097 points, each a sample of the scan grid, theta' exactly 0 at all 4097 samples: {ok}")
    # G+q+r
    gq = [r for r in rows if r["label"] == "G+q+r"]
    print(f"\nG+q+r calls: {len({(r['k'], r['q'], r['r'], r['z_response']) for r in gq})}; points {len(gq)}; "
          f"kinds {dict(Counter(r['point_kind'] for r in gq))}; found_as {dict(Counter(r['found_as'] for r in gq))}")
    zs = defaultdict(list)
    for r in gq:
        zs[(r["k"], r["q"], r["r"])].append(float(r["location"]))
    for kqr, v in sorted(zs.items(), key=lambda t: tuple(float(x) for x in t[0])):
        print(f"  (k, q, r) = ({float(kqr[0]):.6g}, {float(kqr[1]):.6g}, {float(kqr[2]):.6g}): {len(v)} calls, located z "
              f"{min(v)!r} .. {max(v)!r}" + (f" (distinct: {sorted(set(v))})" if len(set(v)) <= 3 else ""))
    # analytic
    an = [r for r in rows if r["kind"] == "analytic"]
    print(f"\nanalytic blocks: {len({(r['k'], r['q'], r['r'], r['z_response'], r['record']) for r in an})}; points {len(an)}; "
          f"kinds {dict(Counter(r['point_kind'] for r in an))}; blocks {dict(Counter(r['record'].split('.', 1)[1] for r in an))}")
    etas = defaultdict(list)
    for r in an:
        etas[(r["k"], r["q"], r["r"], r["record"].split(".")[-1])].append(float(r["location"]))
    for kqr, v in sorted(etas.items(), key=lambda t: tuple(float(x) for x in t[0][:3]) + (t[0][3],)):
        print(f"  ({float(kqr[0]):.6g}, {float(kqr[1]):.6g}, {float(kqr[2]):.6g}) {kqr[3]}: {len(v)} calls, located eta "
              f"{min(v):.6g} .. {max(v):.6g}")
    tds = [abs(float(r["theta_deriv_at_point"])) for r in an + gq if r["theta_deriv_at_point"] not in ("", None)]
    print(f"\n|theta'| at the located points (G+q+r and analytic): max {max(tds)!r}")
    print(f"minima located: {sum(1 for r in rows if r['point_kind'] == 'minimum')}; points at an end: "
          f"{sum(1 for r in rows if r['point_kind'] == 'end')} (q-r: 2 per call, in the summaries)")


if __name__ == "__main__":
    main()
