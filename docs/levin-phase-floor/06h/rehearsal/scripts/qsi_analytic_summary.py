"""Prompt 06h §4.3, descriptive: what moved inside QuadSourceIntegral, R21 against R22.

06h changes only `BesselPhaseGroup` (LiouvilleGreen/three_bessel_integrals.py). In
QuadSourceIntegral that class is used only by `_three_bessel_Levin`, the analytic oracle, whose
result is the column `analytic_rad` and the metadata `analytic.*`; `total` is
`numeric_quad + WKB_Levin` (ComputeTargets/QuadSourceIntegral.py, the return of
evaluate_QuadSource_integral), and WKB_Levin's phase groups are `ComputeTargets/phase_groups.py`'s,
which 06h does not change (06h §0.3). This script describes the analytic side, which H0 (on
`total`) does not reach. Nothing here is scored.

Reads both stores with sqlite `mode=ro`, through `tabulate_qsi_groupphase.read` (same directory),
rows paired by the physical key (k, q, r, z_response). Per row:
  - `analytic_rad`: bit-identical or not (float.hex); |d analytic_rad| against the row's
    `analytic.abserr` in each store, and against their sum (the H0 form on the oracle);
  - `analytic.abserr` (the oracle's declared error), R22/R21;
  - `analytic.converged`, R21 -> R22.
Per three-Bessel Levin block (`analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data`): `abserr` R22/R21,
`regions` (the block's `len(regions)`, a work counter), `converged`, `phase_limited`. No per-block
value and no per-block depth is stored.

    python -I -B qsi_analytic_summary.py <R21 primary> <R22 primary> <out.csv>
"""

import csv
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tabulate_qsi_groupphase import read  # noqa: E402

BLOCKS = [
    (o, f"{f}{i}_data") for o in ("0pt5", "2pt5") for f in ("J", "Y") for i in (1, 2, 3, 4)
]


def q(vals, ps=(0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)):
    v = sorted(vals)
    if not v:
        return "none"
    return ", ".join(
        f"p{p * 100:g} {v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]:.4g}" for p in ps
    )


def fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def main():
    a_path, b_path, out_csv = sys.argv[1:4]
    A, _ = read(a_path)
    B, _ = read(b_path)
    assert set(A) == set(B)
    rows = []
    blk = []
    for key in sorted(A):
        ma = json.loads(A[key]["metadata"] or "{}").get("analytic", {})
        mb = json.loads(B[key]["metadata"] or "{}").get("analytic", {})
        va, vb = A[key]["analytic_rad"], B[key]["analytic_rad"]
        ea, eb = ma.get("abserr"), mb.get("abserr")
        nblocks = 0
        for o, name in BLOCKS:
            da = ma.get(o, {}).get("Levin", {}).get(name)
            db = mb.get(o, {}).get("Levin", {}).get(name)
            assert (da is None) == (db is None), (key, o, name)
            if da is None:
                continue
            nblocks += 1
            blk.append(
                {
                    "key": key,
                    "block": f"{o}.{name}",
                    "abserr21": da["abserr"],
                    "abserr22": db["abserr"],
                    "regions21": da["regions"],
                    "regions22": db["regions"],
                    "conv21": da["converged"],
                    "conv22": db["converged"],
                    "pl21": da["phase_limited"],
                    "pl22": db["phase_limited"],
                    "abserr_bit_identical": float(da["abserr"]).hex() == float(db["abserr"]).hex(),
                }
            )
        d = abs(vb - va) if fin(va) and fin(vb) else None
        rows.append(
            {
                "k": key[0],
                "q": key[1],
                "r": key[2],
                "z_response": key[3],
                "analytic_levin_blocks": nblocks,
                "analytic_rad_R21": None if va is None else float(va).hex(),
                "analytic_rad_R22": None if vb is None else float(vb).hex(),
                "analytic_rad_bit_identical": (va is None and vb is None)
                or (va is not None and vb is not None and float(va).hex() == float(vb).hex()),
                "abs_d_analytic_rad": d,
                "analytic_abserr_R21": ea,
                "analytic_abserr_R22": eb,
                "d_over_abserr_R21": d / ea if d is not None and fin(ea) and ea else None,
                "d_over_abserr_R22": d / eb if d is not None and fin(eb) and eb else None,
                "d_over_sum_abserr": (
                    d / (ea + eb) if d is not None and fin(ea) and fin(eb) and ea + eb > 0 else None
                ),
                "analytic_abserr_ratio_R22_R21": eb / ea if fin(ea) and fin(eb) and ea else None,
                "analytic_converged_R21": ma.get("converged"),
                "analytic_converged_R22": mb.get("converged"),
            }
        )
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    withblk = [r for r in rows if r["analytic_levin_blocks"]]
    print(f"rows {n}; rows with a three-Bessel Levin block {len(withblk)}; blocks {len(blk)}")
    print(
        f"analytic_rad bit-identical: {sum(r['analytic_rad_bit_identical'] for r in rows)}/{n} "
        f"(rows with a block {sum(r['analytic_rad_bit_identical'] for r in withblk)}/{len(withblk)}, "
        f"rows without {sum(r['analytic_rad_bit_identical'] for r in rows if not r['analytic_levin_blocks'])}"
        f"/{n - len(withblk)})"
    )
    print(
        f"analytic_rad None in R21/R22: {sum(r['analytic_rad_R21'] is None for r in rows)}/"
        f"{sum(r['analytic_rad_R22'] is None for r in rows)}; analytic.abserr non-finite or absent "
        f"R21/R22: {sum(not fin(r['analytic_abserr_R21']) for r in rows)}/"
        f"{sum(not fin(r['analytic_abserr_R22']) for r in rows)}"
    )
    print(
        f"analytic.abserr bit-identical: "
        f"{sum(1 for r in rows if fin(r['analytic_abserr_R21']) and fin(r['analytic_abserr_R22']) and float(r['analytic_abserr_R21']).hex() == float(r['analytic_abserr_R22']).hex())}/{n}"
    )
    moved = [r for r in rows if not r["analytic_rad_bit_identical"]]
    print(f"rows whose analytic_rad moved: {len(moved)}")
    print(f"  |d analytic_rad|/(abserr21+abserr22): {q([r['d_over_sum_abserr'] for r in moved if r['d_over_sum_abserr'] is not None])}")
    print(f"  |d analytic_rad|/abserr21: {q([r['d_over_abserr_R21'] for r in moved if r['d_over_abserr_R21'] is not None])}")
    print(f"  |d analytic_rad|/abserr22: {q([r['d_over_abserr_R22'] for r in moved if r['d_over_abserr_R22'] is not None])}")
    print(f"  |d analytic_rad|/|analytic_rad21|: {q([r['abs_d_analytic_rad'] / abs(float.fromhex(r['analytic_rad_R21'])) for r in moved if r['abs_d_analytic_rad'] is not None and float.fromhex(r['analytic_rad_R21'])])}")
    viol = [r for r in rows if r["d_over_sum_abserr"] is not None and r["d_over_sum_abserr"] > 1]
    print(f"  rows with |d analytic_rad| > abserr21 + abserr22: {len(viol)}")
    for r in viol:
        print(f"    {r}")
    w = max((r for r in moved if r["d_over_sum_abserr"] is not None), key=lambda r: r["d_over_sum_abserr"], default=None)
    if w:
        print(
            f"  largest: {w['d_over_sum_abserr']:.6g} at k={w['k']:.6g} q={w['q']:.6g} r={w['r']:.6g} "
            f"z={w['z_response']:.8g}: rad21 {w['analytic_rad_R21']} rad22 {w['analytic_rad_R22']} "
            f"abserr21 {float(w['analytic_abserr_R21']).hex()} abserr22 {float(w['analytic_abserr_R22']).hex()}"
        )
    print(f"analytic.abserr R22/R21 (rows with a block): {q([r['analytic_abserr_ratio_R22_R21'] for r in withblk if r['analytic_abserr_ratio_R22_R21'] is not None])}")
    print(
        f"  rows abserr up {sum(1 for r in withblk if (r['analytic_abserr_ratio_R22_R21'] or 1) > 1)}, "
        f"down {sum(1 for r in withblk if (r['analytic_abserr_ratio_R22_R21'] or 1) < 1)}, "
        f"equal {sum(1 for r in withblk if r['analytic_abserr_ratio_R22_R21'] == 1)}"
    )
    print(f"analytic.converged R21 -> R22: {dict(Counter((r['analytic_converged_R21'], r['analytic_converged_R22']) for r in rows))}")
    print()
    print("three-Bessel Levin blocks:")
    print(f"  abserr bit-identical: {sum(b['abserr_bit_identical'] for b in blk)}/{len(blk)}")
    print(f"  abserr R22/R21: {q([b['abserr22'] / b['abserr21'] for b in blk if b['abserr21']])}")
    print(f"  converged R21 -> R22: {dict(Counter((b['conv21'], b['conv22']) for b in blk))}")
    print(f"  phase_limited R21 -> R22: {dict(Counter((b['pl21'], b['pl22']) for b in blk))}")
    print(
        f"  regions summed R21 {sum(b['regions21'] for b in blk)}, R22 {sum(b['regions22'] for b in blk)}; "
        f"blocks more {sum(1 for b in blk if b['regions22'] > b['regions21'])}, "
        f"fewer {sum(1 for b in blk if b['regions22'] < b['regions21'])}, "
        f"equal {sum(1 for b in blk if b['regions22'] == b['regions21'])}"
    )
    print(f"  regions per block R21: {dict(sorted(Counter(b['regions21'] for b in blk).items()))}")
    print(f"  regions per block R22: {dict(sorted(Counter(b['regions22'] for b in blk).items()))}")
    for o in ("0pt5", "2pt5"):
        s = [b for b in blk if b["block"].startswith(o)]
        print(
            f"  {o}: blocks {len(s)}; regions R21 {sum(b['regions21'] for b in s)} R22 {sum(b['regions22'] for b in s)}; "
            f"abserr R22/R21 {q([b['abserr22'] / b['abserr21'] for b in s if b['abserr21']], ps=(0.0, 0.5, 1.0))}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
