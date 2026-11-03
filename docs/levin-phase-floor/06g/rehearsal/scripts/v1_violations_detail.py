"""Prompt 06g §4, descriptive, not scored: where V1's violating rows moved, group by group.

For every QuadSourceIntegral row that `tabulate_qsi_leafvalues.py` reports as violating V1 on
`total` (read back from its CSV, column `V1` == "VIOLATES"), this prints, from both stores
(sqlite `mode=ro`, through `tabulate_qsi_leafvalues.read`, rows paired by physical key):
  - the row's columns: total, total_abserr (float.hex), total_converged, total_phase_limited,
    WKB_Levin_num_regions, WKB_Levin_max_depth, numeric_quad (float.hex; its bit-identity);
  - every WKB Levin phase group, from the metadata at
    `WKB_Levin.subintervals[i].groups[j]` (`value`, `abserr`, `converged`, `phase_limited`,
    `regions`; the sub-interval's `regime`, `z_max`, `z_min`): |d value| against
    abserr_23 + abserr_24, and the group's share of the row's |d total|.
A summary over the violating rows follows: how many groups violate the same rule at group level,
and their converged / phase_limited / regions in each store. The stores hold no per-leaf data
(06c §8), so the cause inside a group is not visible here.

    python -I -B v1_violations_detail.py <R23 primary> <R24 primary> <qsi_rows.csv>
"""

import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tabulate_qsi_leafvalues import read  # noqa: E402


def hx(x):
    return None if x is None else float(x).hex()


def main():
    a_path, b_path, rows_csv = sys.argv[1:4]
    A, _ = read(a_path)
    B, _ = read(b_path)
    viol = []
    with open(rows_csv) as fh:
        for r in csv.DictReader(fh):
            if r["V1"] == "VIOLATES":
                viol.append(
                    (float(r["k"]), float(r["q"]), float(r["r"]), float(r["z_response"]))
                )
    print(f"violating rows read from {rows_csv}: {len(viol)}")
    summ = Counter()
    gtab = []
    for key in viol:
        a, b = A[key], B[key]
        dT = abs(b["total"] - a["total"])
        print()
        print(
            f"== k={key[0]:.6g} q={key[1]:.6g} r={key[2]:.6g} z_response={key[3]!r}: "
            f"|dT| {dT:.4e}, ratio {dT / (a['total_abserr'] + b['total_abserr']):.6g}"
        )
        for tag, s in (("R23", a), ("R24", b)):
            print(
                f"   {tag}: total {hx(s['total'])} ({s['total']:.10e}) abserr {hx(s['total_abserr'])} "
                f"({s['total_abserr']:.4e}) converged {s['total_converged']} phase_limited "
                f"{s['total_phase_limited']} regions {s['WKB_Levin_num_regions']} max_depth "
                f"{s['WKB_Levin_max_depth']} numeric_quad {hx(s['numeric_quad'])}"
            )
        print(f"   numeric_quad bit-identical: {hx(a['numeric_quad']) == hx(b['numeric_quad'])}")
        ma = json.loads(a["metadata"])["WKB_Levin"]["subintervals"]
        mb = json.loads(b["metadata"])["WKB_Levin"]["subintervals"]
        assert len(ma) == len(mb)
        for i, (sa, sb) in enumerate(zip(ma, mb)):
            assert sa["regime"] == sb["regime"] and len(sa["groups"]) == len(sb["groups"])
            for j, (ga, gb) in enumerate(zip(sa["groups"], sb["groups"])):
                assert ga["label"] == gb["label"]
                d = abs(gb["value"] - ga["value"])
                es = ga["abserr"] + gb["abserr"]
                ratio = d / es if es > 0 else float("inf")
                share = d / dT if dT else 0.0
                gviol = d > es
                summ["groups"] += 1
                summ["groups_violating"] += gviol
                gtab.append(
                    (gviol, ga["converged"], gb["converged"], ga["phase_limited"], gb["phase_limited"])
                )
                if gviol or share > 0.05:
                    print(
                        f"   sub[{i}] regime {sa['regime']} z ({sa['z_max']:.6g}, {sa['z_min']:.6g}) "
                        f"group {ga['label']!r}: |dv| {d:.4e} = {share:.3f} of |dT|; ratio {ratio:.6g}; "
                        f"v23 {hx(ga['value'])} v24 {hx(gb['value'])} e23 {hx(ga['abserr'])} "
                        f"e24 {hx(gb['abserr'])}; conv {ga['converged']}>{gb['converged']} "
                        f"pl {ga['phase_limited']}>{gb['phase_limited']} regions "
                        f"{ga['regions']}>{gb['regions']}"
                    )
    print()
    print(f"groups in the violating rows: {summ['groups']}; violating at group level: {summ['groups_violating']}")
    print(
        "violating groups by (converged 23, converged 24, phase_limited 23, phase_limited 24): "
        f"{dict(Counter(t[1:] for t in gtab if t[0]))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
