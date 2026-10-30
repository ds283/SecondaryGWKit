"""Prompt 06c §4.3, descriptive: where Y1's violating rows move, call by call.

For every row tabulate_qsi_global.py marks VIOLATES in its CSV, prints each WKB Levin call (phase
group of a sub-interval) of R17 and R18 side by side from the `metadata` column: value, abserr,
converged, phase_limited and regions, with |dV| against abserr17 + abserr18 per call, and the
numeric_quad part. Read-only (sqlite `mode=ro`), physical-key pairing as the tabulation's.

    python -I y1_violations_groups.py <R17 primary> <R18 primary> <qsi_rows.csv>
"""

import csv
import json
import sqlite3
import sys
from pathlib import Path


def read(primary, keys):
    primary = Path(primary)
    con = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
    shards = [r[0] for r in con.execute("SELECT filename FROM shards ORDER BY serial")]
    con.close()
    out = {}
    kq = (
        "(SELECT w.k_inv_Mpc FROM wavenumber_exit_time e JOIN wavenumber w "
        "ON w.serial = e.wavenumber_serial WHERE e.serial = q.{})"
    )
    sql = (
        f"SELECT {kq.format('k_wavenumber_exit_serial')}, {kq.format('q_wavenumber_exit_serial')}, "
        f"{kq.format('r_wavenumber_exit_serial')}, (SELECT z FROM redshift WHERE serial = q.z_response_serial), "
        "q.total, q.total_abserr, q.numeric_quad, q.WKB_Levin, q.metadata FROM QuadSourceIntegral q"
    )
    for name in shards:
        con = sqlite3.connect(f"file:{primary.parent / Path(name).name}?mode=ro", uri=True)
        for row in con.execute(sql):
            if row[:4] in keys:
                out[row[:4]] = row[4:]
        con.close()
    return out


def main():
    a_path, b_path, csv_path = sys.argv[1:4]
    viol = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["Y1"] == "VIOLATES":
                viol.append(
                    (
                        (float(r["k"]), float(r["q"]), float(r["r"]), float(r["z_response"])),
                        float(r["Y1_ratio_dtotal_over_sum"]),
                    )
                )
    keys = {k for k, _ in viol}
    A, B = read(a_path, keys), read(b_path, keys)
    print(f"violating rows: {len(viol)}")
    summary = []
    for key, ratio in sorted(viol, key=lambda t: -t[1]):
        ta, ea, nqa, wla, ma = A[key]
        tb, eb, nqb, wlb, mb = B[key]
        ma, mb = json.loads(ma), json.loads(mb)
        print()
        print(
            f"k={key[0]:.6g} q={key[1]:.6g} r={key[2]:.6g} z={key[3]:.8g}  Y1 ratio {ratio:.4g}; "
            f"total {ta:.10e} -> {tb:.10e}; abserr {ea:.3e} -> {eb:.3e}"
        )
        print(
            f"  numeric_quad {nqa.hex()} -> {nqb.hex()} (identical {nqa.hex() == nqb.hex()}); "
            f"WKB_Levin {wla:.10e} -> {wlb:.10e}; numeric_quad abserr "
            f"{ma['numeric_quad']['abserr']:.3e} -> {mb['numeric_quad']['abserr']:.3e}"
        )
        worst = None
        for i, (sa, sb) in enumerate(
            zip(ma["WKB_Levin"]["subintervals"], mb["WKB_Levin"]["subintervals"])
        ):
            assert (sa["z_max"], sa["z_min"]) == (sb["z_max"], sb["z_min"])
            for j, (ga, gb) in enumerate(zip(sa["groups"], sb["groups"])):
                assert ga["label"] == gb["label"]
                dv = abs(gb["value"] - ga["value"])
                s = ga["abserr"] + gb["abserr"]
                rr = dv / s if s > 0 else (0.0 if dv == 0 else float("inf"))
                print(
                    f"  sub {i} z({sa['z_max']:.5g},{sa['z_min']:.5g}) {sa['regime']} group {ga['label']:<10} "
                    f"value {ga['value']:+.10e} -> {gb['value']:+.10e} |dV| {dv:.3e}; abserr {ga['abserr']:.3e} -> {gb['abserr']:.3e}; "
                    f"|dV|/sum {rr:.3g}; conv {int(ga['converged'])}>{int(gb['converged'])} "
                    f"pl {int(ga['phase_limited'])}>{int(gb['phase_limited'])} regions {ga['regions']}>{gb['regions']}"
                )
                if worst is None or rr > worst[0]:
                    worst = (rr, i, ga["label"], sa["regime"])
        summary.append((key, ratio, worst))
    print()
    print("per violating row, the call with the largest |dV|/(abserr17+abserr18):")
    for key, ratio, worst in summary:
        print(
            f"  k={key[0]:.6g} q={key[1]:.6g} r={key[2]:.6g} z={key[3]:.8g}: row ratio {ratio:.4g}; "
            f"call sub {worst[1]} {worst[3]} group {worst[2]} ratio {worst[0]:.4g}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
