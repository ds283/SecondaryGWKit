"""
Compare the error A3 v2 declares on its depth-20 QuadSourceIntegral rows with the double-precision
floor of the phase handed to the Levin driver, eps * k * eta(z_response).

Reads only the committed depth census, docs/handover/a3-v2/after/out/depth_census_08.txt, which
lists every one of the 2265 rows at WKB_Levin_max_depth = 20 with (k, q, r, z_response,
compute_time, depth, regions, total_converged, abserr/|total|). No store, no Ray, no repository
imports.

eta(z) is the conformal time of a flat LambdaCDM background written out here (h, Omega_m, Omega_r
below), not the pipeline's BackgroundModel. It is checked against the phase spans
docs/gktk-remedial-verification.md Layer 1 measured through the production functions: there
|theta_G| at z = 0.1, 1.0006, ... divided by k is the pipeline's own conformal-time difference from
the source end, which at these redshifts is eta(z) to the precision this comparison needs.

The seam band (z_response > 1e5) is left out: eps * k * eta is ~1e-12 there and nothing is
expected to be floor-limited.

Usage, from the repository root:

    ./venv/bin/python docs/handover/a3-v2/findings/depth20_floor.py \
        docs/handover/a3-v2/after/out/depth_census_08.txt
"""

import hashlib
import subprocess
import sys
from math import expm1, log

import numpy as np
from scipy.integrate import quad

EPS = np.finfo(float).eps  # 2.22e-16
RTOL = 1e-7  # the pair A3 v2 was built at, (1e-32, 1e-7)

C_KM_S = 2.99792458e5
H = 0.677
OMEGA_M = 0.31
OMEGA_R = 9.1e-5
OMEGA_L = 1.0 - OMEGA_M - OMEGA_R
H0_INV_MPC = 100.0 * H / C_KM_S

# |theta_G| / k at z_response, from docs/gktk-remedial-verification.md Layer 1
# (LambdaCDM, k = 3e8/Mpc; theta_ref column)
LAYER1_CHECK = [
    (0.1, 4.1184e12 / 3e8),
    (1.0006, 3.229e12 / 3e8),
    (10.012, 1.357e12 / 3e8),
    (100.19, 3.991e11 / 3e8),
    (1002.5, 8.952e10 / 3e8),
]

BANDS = [("mid", 9.0, 1e5), ("low", 0.0, 9.0)]


def hubble(z):
    return H0_INV_MPC * np.sqrt(
        OMEGA_M * (1 + z) ** 3 + OMEGA_R * (1 + z) ** 4 + OMEGA_L
    )


def eta(z):
    """conformal time in Mpc, integrated in u = log(1 + z) from z to z = 1e15"""
    u0, u1 = log(1.0 + z), log(1.0 + 1e15)
    value, _ = quad(lambda u: (1.0 + expm1(u)) / hubble(expm1(u)), u0, u1, limit=500)
    return value


def read_depth20_rows(path):
    rows = []
    reading = False
    with open(path) as handle:
        for line in handle:
            if line.startswith("-- every row at depth 20"):
                reading = True
                continue
            if reading and line.startswith("--"):
                break
            if reading:
                fields = line.split()
                if len(fields) == 11 and fields[0][0].isdigit():
                    rows.append([float(x) for x in fields[:9]])
    return np.array(rows)


def provenance(path):
    with open(__file__, "rb") as handle:
        script_hash = hashlib.sha256(handle.read()).hexdigest()
    with open(path, "rb") as handle:
        input_hash = hashlib.sha256(handle.read()).hexdigest()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return script_hash, input_hash, head


def main(path):
    script_hash, input_hash, head = provenance(path)
    print(f"script sha256 {script_hash}")
    print(f"input  {path} sha256 {input_hash}")
    print(f"git HEAD {head}")
    print(
        f"background: flat LambdaCDM, h = {H}, Omega_m = {OMEGA_M}, Omega_r = {OMEGA_R}; eps = {EPS:.3e}"
    )

    print("\n== eta(z) against the pipeline's |theta_G|/k (gktk-remedial Layer 1)")
    for z, eta_pipeline in LAYER1_CHECK:
        here = eta(z)
        print(
            f"  z = {z:9.4f}: here {here:10.1f} Mpc, pipeline {eta_pipeline:10.1f} Mpc, ratio {here / eta_pipeline:.4f}"
        )

    rows = read_depth20_rows(path)
    k, q, r, z, t, depth, regions, conv, rel = rows.T
    print(f"\n== depth-20 rows read: {len(rows)}")

    in_bands = z <= 1e5
    floor = np.full(len(rows), np.nan)
    floor[in_bands] = EPS * k[in_bands] * np.array([eta(zz) for zz in z[in_bands]])

    print(
        "\n== declared abserr/|total| against the phase floor eps*k*eta(z_response), by k and band"
    )
    print(
        "        k /Mpc  band     n  median t (s)  max t (s)  unconverged  median abserr/|total|  median eps*k*eta  median ratio"
    )
    for kk in sorted(set(k)):
        for name, lo, hi in BANDS:
            mask = (
                (k == kk) & (z >= lo) & (z <= hi)
                if name == "mid"
                else (k == kk) & (z < hi)
            )
            if not mask.any():
                continue
            print(
                f"  {kk:12.4e}  {name:4s}  {mask.sum():4d}  {np.median(t[mask]):12.2f}  {t[mask].max():9.1f}"
                f"  {int((conv[mask] == 0).sum()):11d}  {np.median(rel[mask]):21.3e}"
                f"  {np.median(floor[mask]):16.3e}  {np.median(rel[mask] / floor[mask]):12.2f}"
            )

    infinite = ~np.isfinite(rel)
    print(
        f"\n-- rows whose abserr/|total| is +inf, presumably total_abserr = +inf (the census writes nan, not inf, for total = 0): {int(infinite.sum())}"
    )
    for row in rows[infinite]:
        print(
            f"   k {row[0]:.4e}  q {row[1]:.4e}  r {row[2]:.4e}  z {row[3]:.6g}  t {row[4]:.2f} s  converged {int(row[7])}"
        )

    mask = in_bands & np.isfinite(rel) & (rel > 0)
    slope, _ = np.polyfit(np.log10(floor[mask]), np.log10(rel[mask]), 1)
    print(
        f"\n-- log-log slope of abserr/|total| on eps*k*eta over the {mask.sum()} mid/low-band rows: {slope:.3f}"
    )
    ratio = rel[mask] / floor[mask]
    print(
        f"-- ratio abserr/|total| / (eps*k*eta): 10th, 50th, 90th percentile "
        f"{np.percentile(ratio, 10):.3f}, {np.percentile(ratio, 50):.3f}, {np.percentile(ratio, 90):.3f}"
    )

    print("\n== total_converged against the declared error, depth-20 rows")
    flagged = (conv == 1) & np.isfinite(rel)
    for threshold in (10 * RTOL, 1e-5, 1e-4):
        n = int((flagged & (rel > threshold)).sum())
        print(
            f"  converged = 1 and abserr/|total| > {threshold:.0e}: {n} of {int(flagged.sum())} converged rows"
        )
    print(f"  max abserr/|total| on a converged row: {rel[flagged].max():.3e}")

    expensive = t > 60.0
    print(f"\n== geometry of the {int(expensive.sum())} depth-20 rows over 60 s")
    print(f"  r == k: {int((r[expensive] == k[expensive]).sum())}")
    print(f"  q == r == k: {int(((q == k) & (r == k))[expensive].sum())}")
    print(
        f"  by k: "
        + ", ".join(
            f"{kk:.3e}: {int(((k == kk) & expensive).sum())}" for kk in sorted(set(k))
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} depth_census_08.txt")
    main(sys.argv[1])
