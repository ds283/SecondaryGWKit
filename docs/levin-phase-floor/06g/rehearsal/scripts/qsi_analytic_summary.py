"""Prompt 06g §4, descriptive: what moved in the analytic oracle of QuadSourceIntegral, R23
against R24.

(06h's `qsi_analytic_summary.py`, copied and adapted for 06g: the store names R21/R22 -> R23/R24
and their suffixes, the import from this directory's `tabulate_qsi_leafvalues`, and this
paragraph. The diff against 06h's copy is saved as out/qsi_analytic_summary.diff.)

06g changes `_adaptive_levin` (AdaptiveLevin/levin_quadrature.py), which serves every Levin call
in QuadSourceIntegral: the WKB phase groups behind `total` and the three-Bessel blocks of the
analytic oracle (`_three_bessel_Levin`), whose result is the column `analytic_rad` and the
metadata `analytic.*` (06g §0.3, "`total` moves"). V1 on both is scored by
`tabulate_qsi_leafvalues.py`; this script describes the oracle's blocks. Nothing here is scored.

Reads both stores with sqlite `mode=ro`, through `tabulate_qsi_leafvalues.read` (same directory),
rows paired by the physical key (k, q, r, z_response). Per row:
  - `analytic_rad`: bit-identical or not (float.hex); |d analytic_rad| against the row's
    `analytic.abserr` in each store, and against their sum (V1's form on the oracle);
  - `analytic.abserr` (the oracle's declared error), R24/R23;
  - `analytic.converged`, R23 -> R24.
Per three-Bessel Levin block (`analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data`): `abserr` R24/R23,
`regions` (the block's `len(regions)`, a work counter), `converged`, `phase_limited`. No per-block
value and no per-block depth is stored.

    python -I -B qsi_analytic_summary.py <R23 primary> <R24 primary> <out.csv>
"""

import csv
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tabulate_qsi_leafvalues import read  # noqa: E402

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
                    "abserr23": da["abserr"],
                    "abserr24": db["abserr"],
                    "regions23": da["regions"],
                    "regions24": db["regions"],
                    "conv23": da["converged"],
                    "conv24": db["converged"],
                    "pl23": da["phase_limited"],
                    "pl24": db["phase_limited"],
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
                "analytic_rad_R23": None if va is None else float(va).hex(),
                "analytic_rad_R24": None if vb is None else float(vb).hex(),
                "analytic_rad_bit_identical": (va is None and vb is None)
                or (va is not None and vb is not None and float(va).hex() == float(vb).hex()),
                "abs_d_analytic_rad": d,
                "analytic_abserr_R23": ea,
                "analytic_abserr_R24": eb,
                "d_over_abserr_R23": d / ea if d is not None and fin(ea) and ea else None,
                "d_over_abserr_R24": d / eb if d is not None and fin(eb) and eb else None,
                "d_over_sum_abserr": (
                    d / (ea + eb) if d is not None and fin(ea) and fin(eb) and ea + eb > 0 else None
                ),
                "analytic_abserr_ratio_R24_R23": eb / ea if fin(ea) and fin(eb) and ea else None,
                "analytic_converged_R23": ma.get("converged"),
                "analytic_converged_R24": mb.get("converged"),
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
        f"analytic_rad None in R23/R24: {sum(r['analytic_rad_R23'] is None for r in rows)}/"
        f"{sum(r['analytic_rad_R24'] is None for r in rows)}; analytic.abserr non-finite or absent "
        f"R23/R24: {sum(not fin(r['analytic_abserr_R23']) for r in rows)}/"
        f"{sum(not fin(r['analytic_abserr_R24']) for r in rows)}"
    )
    print(
        f"analytic.abserr bit-identical: "
        f"{sum(1 for r in rows if fin(r['analytic_abserr_R23']) and fin(r['analytic_abserr_R24']) and float(r['analytic_abserr_R23']).hex() == float(r['analytic_abserr_R24']).hex())}/{n}"
    )
    moved = [r for r in rows if not r["analytic_rad_bit_identical"]]
    print(f"rows whose analytic_rad moved: {len(moved)}")
    print(f"  |d analytic_rad|/(abserr23+abserr24): {q([r['d_over_sum_abserr'] for r in moved if r['d_over_sum_abserr'] is not None])}")
    print(f"  |d analytic_rad|/abserr23: {q([r['d_over_abserr_R23'] for r in moved if r['d_over_abserr_R23'] is not None])}")
    print(f"  |d analytic_rad|/abserr24: {q([r['d_over_abserr_R24'] for r in moved if r['d_over_abserr_R24'] is not None])}")
    print(f"  |d analytic_rad|/|analytic_rad23|: {q([r['abs_d_analytic_rad'] / abs(float.fromhex(r['analytic_rad_R23'])) for r in moved if r['abs_d_analytic_rad'] is not None and float.fromhex(r['analytic_rad_R23'])])}")
    viol = [r for r in rows if r["d_over_sum_abserr"] is not None and r["d_over_sum_abserr"] > 1]
    print(f"  rows with |d analytic_rad| > abserr23 + abserr24: {len(viol)}")
    for r in viol:
        print(f"    {r}")
    w = max((r for r in moved if r["d_over_sum_abserr"] is not None), key=lambda r: r["d_over_sum_abserr"], default=None)
    if w:
        print(
            f"  largest: {w['d_over_sum_abserr']:.6g} at k={w['k']:.6g} q={w['q']:.6g} r={w['r']:.6g} "
            f"z={w['z_response']:.8g}: rad23 {w['analytic_rad_R23']} rad24 {w['analytic_rad_R24']} "
            f"abserr23 {float(w['analytic_abserr_R23']).hex()} abserr24 {float(w['analytic_abserr_R24']).hex()}"
        )
    print(f"analytic.abserr R24/R23 (rows with a block): {q([r['analytic_abserr_ratio_R24_R23'] for r in withblk if r['analytic_abserr_ratio_R24_R23'] is not None])}")
    print(
        f"  rows abserr up {sum(1 for r in withblk if (r['analytic_abserr_ratio_R24_R23'] or 1) > 1)}, "
        f"down {sum(1 for r in withblk if (r['analytic_abserr_ratio_R24_R23'] or 1) < 1)}, "
        f"equal {sum(1 for r in withblk if r['analytic_abserr_ratio_R24_R23'] == 1)}"
    )
    print(f"analytic.converged R23 -> R24: {dict(Counter((r['analytic_converged_R23'], r['analytic_converged_R24']) for r in rows))}")
    print()
    print("three-Bessel Levin blocks:")
    print(f"  abserr bit-identical: {sum(b['abserr_bit_identical'] for b in blk)}/{len(blk)}")
    print(f"  abserr R24/R23: {q([b['abserr24'] / b['abserr23'] for b in blk if b['abserr23']])}")
    print(f"  converged R23 -> R24: {dict(Counter((b['conv23'], b['conv24']) for b in blk))}")
    print(f"  phase_limited R23 -> R24: {dict(Counter((b['pl23'], b['pl24']) for b in blk))}")
    print(
        f"  regions summed R23 {sum(b['regions23'] for b in blk)}, R24 {sum(b['regions24'] for b in blk)}; "
        f"blocks more {sum(1 for b in blk if b['regions24'] > b['regions23'])}, "
        f"fewer {sum(1 for b in blk if b['regions24'] < b['regions23'])}, "
        f"equal {sum(1 for b in blk if b['regions24'] == b['regions23'])}"
    )
    print(f"  regions per block R23: {dict(sorted(Counter(b['regions23'] for b in blk).items()))}")
    print(f"  regions per block R24: {dict(sorted(Counter(b['regions24'] for b in blk).items()))}")
    for o in ("0pt5", "2pt5"):
        s = [b for b in blk if b["block"].startswith(o)]
        print(
            f"  {o}: blocks {len(s)}; regions R23 {sum(b['regions23'] for b in s)} R24 {sum(b['regions24'] for b in s)}; "
            f"abserr R24/R23 {q([b['abserr24'] / b['abserr23'] for b in s if b['abserr23']], ps=(0.0, 0.5, 1.0))}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
