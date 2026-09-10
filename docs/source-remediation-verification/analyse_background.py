"""Audit section 4.4 -- the A7 grid-end bias on a real GenericEOS (QCD) run, and the A1 fix in
the stored background columns.

`LambdaCDM_GenericEOS` (and therefore `QCD_Cosmology`) supplies no analytic derivative methods,
so all five derivative columns of `BackgroundModelValue` are produced by the padded, refined,
quintic spline stack prompt 03 installed in `ComputeTargets/BackgroundModel.py`. This script
compares the stored columns against an independent high-order central-difference derivative of
the cosmology's own `Hubble` and `wPerturbations`, taken in x = log(1+z) exactly as the
production code does, at the two grid ends and in the interior; the end/interior ratio is the
statistic the audit's TK-5 table and prompt 03's acceptance criterion use.

It also checks the A1 fix as it appears in stored data: `wPerturbations` must be Lambda-free,
i.e. equal to (1/3) rho_r / (rho_m + rho_r) rather than (1/3) rho_r / rho_total.

Read-only; the shard files are opened in `mode=ro`.
"""

import argparse
import glob
import sqlite3
from math import log, exp, log1p, expm1

import numpy as np

from CosmologyModels.LambdaCDM import Planck2018
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.LambdaCDM import LambdaCDM
from Units import Mpc_units

# central-difference weights in x = log(1+z)
STENCILS = {
    1: ([-2, -1, 1, 2], [1.0 / 12, -2.0 / 3, 2.0 / 3, -1.0 / 12]),
    2: ([-2, -1, 0, 1, 2], [-1.0 / 12, 4.0 / 3, -5.0 / 2, 4.0 / 3, -1.0 / 12]),
    3: ([-3, -2, -1, 1, 2, 3], [1.0 / 8, -1.0, 13.0 / 8, -13.0 / 8, 1.0, -1.0 / 8]),
}


def dx_derivative(f, x, order, h):
    offsets, weights = STENCILS[order]
    return sum(w * f(x + o * h) for o, w in zip(offsets, weights)) / h**order


def z_derivatives(f_of_z, z, h):
    """d/dz, d2/dz2, d3/dz3 of f, from central differences in x = log(1+z)."""
    f_of_x = lambda x: f_of_z(expm1(x))
    x = log1p(z)
    opz = 1.0 + z
    d1x = dx_derivative(f_of_x, x, 1, h)
    d2x = dx_derivative(f_of_x, x, 2, h)
    d3x = dx_derivative(f_of_x, x, 3, h)
    # d/dz = (1/(1+z)) d/dx, applied repeatedly
    d1 = d1x / opz
    d2 = (d2x - d1x) / opz**2
    d3 = (d3x - 3.0 * d2x + 2.0 * d1x) / opz**3
    return d1, d2, d3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-glob", required=True)
    parser.add_argument("--max-z", type=float, default=1e20)
    parser.add_argument("--fd-step", type=float, default=1e-4)
    parser.add_argument("--interior-samples", type=int, default=25)
    args = parser.parse_args()

    units = Mpc_units()
    params = Planck2018()
    cosmology = QCD_Cosmology(store_id=0, units=units, params=params, max_z=args.max_z)
    reference = LambdaCDM(store_id=0, units=units, params=params)

    rows = []
    for db in sorted(glob.glob(args.shards_glob)):
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        redshifts = {r[0]: r[1] for r in conn.execute("select serial, z from redshift")}
        model_serials = [
            r[0] for r in conn.execute("select serial from BackgroundModel")
        ]
        for serial in model_serials:
            for r in conn.execute(
                "select z_serial, wBackground, wPerturbations, d_lnH_dz, d2_lnH_dz2, "
                "d3_lnH_dz3, d_wPerturbations_dz, d2_wPerturbations_dz2 "
                "from BackgroundModelValue where model_serial = ?",
                (serial,),
            ):
                rows.append((redshifts[r[0]],) + tuple(r[1:]))
        conn.close()
    rows.sort(key=lambda t: t[0])
    print(f"# BackgroundModelValue rows: {len(rows)}")
    if not rows:
        return
    print(f"# z range: {rows[0][0]:.5g} ... {rows[-1][0]:.5g}")

    # ------------------------------------------------------------------ A1
    print("\n## A1 -- stored wPerturbations excludes Lambda")
    worst = 0.0
    worst_with_lambda = 0.0
    for z, wB, wP, *_ in rows[:: max(1, len(rows) // 12)]:
        rho = cosmology._rho_fluid(z)
        # the numerator is the EOS's own w(T) times rho_r, not (1/3) rho_r: the QCD EOS is not
        # exactly radiation (LambdaCDM_GenericEOS.py:281-292)
        numerator = cosmology._eos.w(rho["T"]) * rho["radiation"]
        without_lambda = numerator / (rho["matter"] + rho["radiation"])
        with_lambda = numerator / cosmology.rho(z)
        e_without = abs(wP - without_lambda) / max(abs(without_lambda), 1e-300)
        e_with = abs(wP - with_lambda) / max(abs(with_lambda), 1e-300)
        worst = max(worst, e_without)
        worst_with_lambda = max(worst_with_lambda, e_with)
        if z < 100.0:
            print(
                f"   z={z:<12.5g} stored={wP:.10e}  Lambda-free={without_lambda:.10e} "
                f"(rel {e_without:.2e})  with-Lambda={with_lambda:.10e} (rel {e_with:.2e})"
            )
    print(f"   worst relative difference from the Lambda-free form: {worst:.3e}")
    print(
        f"   worst relative difference from the with-Lambda form: {worst_with_lambda:.3e}"
    )

    # ------------------------------------------------------------------ A7
    print("\n## A7 -- grid-end bias of the spline-derived derivative columns")
    h = args.fd_step
    lnH = lambda z: log(cosmology.Hubble(z))
    wP = cosmology.wPerturbations

    columns = [
        ("d_lnH_dz", 3, lnH, 1),
        ("d2_lnH_dz2", 4, lnH, 2),
        ("d3_lnH_dz3", 5, lnH, 3),
        ("d_wPerturbations_dz", 6, wP, 1),
        ("d2_wPerturbations_dz2", 7, wP, 2),
    ]

    n = len(rows)
    interior_idx = list(
        range(
            n // 4,
            3 * n // 4,
            max(1, (n // 2) // args.interior_samples),
        )
    )
    probes = {
        "low-z end": [0, 1, 2],
        "high-z end": [n - 1, n - 2, n - 3],
        "interior": interior_idx,
    }

    print(
        f"   reference: central differences in log(1+z), step {h}, of the QCD cosmology's own "
        f"Hubble/wPerturbations"
    )
    print(
        f"   {'column':<24}{'low-z end':>14}{'2nd':>12}{'3rd':>12}"
        f"{'interior median':>18}{'end/interior':>14}{'high-z end':>14}"
    )
    for name, idx, f, order in columns:
        errs = {}
        for label, indices in probes.items():
            values = []
            for i in indices:
                z = rows[i][0]
                stored = rows[i][idx]
                ref = z_derivatives(f, z, h)[order - 1]
                scale = max(abs(ref), 1e-300)
                values.append(abs(stored - ref) / scale)
            errs[label] = values
        interior_median = sorted(errs["interior"])[len(errs["interior"]) // 2]
        ratio = (
            errs["low-z end"][0] / interior_median
            if interior_median > 0
            else float("nan")
        )
        print(
            f"   {name:<24}{errs['low-z end'][0]:>14.3e}{errs['low-z end'][1]:>12.3e}"
            f"{errs['low-z end'][2]:>12.3e}{interior_median:>18.3e}{ratio:>14.3g}"
            f"{errs['high-z end'][0]:>14.3e}"
        )

    print(
        "\n   (relative error against the finite-difference reference; the audit's TK-5 table "
        "quoted 3.0e-01 for eps'' and 3.8e-01 for w'' at the z=0.1 end before prompt 03)"
    )

    # epsilon itself, for comparison with the audit's table
    print("\n   epsilon = (1+z) d lnH/dz at the same probes:")
    for label, indices in probes.items():
        vals = []
        for i in indices:
            z = rows[i][0]
            stored = (1.0 + z) * rows[i][3]
            ref = (1.0 + z) * z_derivatives(lnH, z, h)[0]
            vals.append(abs(stored - ref) / max(abs(ref), 1e-300))
        summary = sorted(vals)[len(vals) // 2] if label == "interior" else vals[0]
        print(f"     {label:<14} {summary:.3e}")


if __name__ == "__main__":
    main()
