import os
"""QS_02: (a) confirm that the 'Tprime' consumed by QuadSource is dT/dz (not
d/dlog(1+z), not H^{-1} dT/deta); (b) build f from the spec 03 R22 formula
independently and compare with ComputeTargets.QuadSource.source_function on the
constant-w analytic transfer function in radiation domination.

Exact constant-w background used here (radiation, w = 1/3):
   H(z) = H0 (1+z)^2
   tau  == a0*eta, defined in ComputeTargets/BackgroundModel.py:56 by
           d(a0 eta)/dz = -1/H   =>  a0 eta = 1/(H0 (1+z))
   (matches BackgroundModel's tau_init = (1+z)/H = 1/(H0(1+z)))
General constant w:
   H = H0 (1+z)^{3(1+w)/2},  a0 eta = 2/((1+3w) H0) (1+z)^{(1+3w)/2}
"""

import sys

sys.path.insert(0, os.getcwd())  # run from the repository root

import numpy as np

from ComputeTargets.analytic_Tk import compute_analytic_T, compute_analytic_Tprime
from ComputeTargets.QuadSource import source_function

H0 = 1.0


def Hubble(z, w):
    return H0 * (1.0 + z) ** (1.5 * (1.0 + w))


def a0eta(z, w):
    # solves d(a0 eta)/dz = -1/H with a0 eta -> 0 as z -> inf (for w > -1/3)
    p = 0.5 * (1.0 + 3.0 * w)
    return 2.0 / ((1.0 + 3.0 * w) * H0) * (1.0 + z) ** (-p)


# ---------------------------------------------------------------- part (a)
print("=== (a) which derivative is compute_analytic_Tprime / TkNumericValue.Tprime ? ===")
print(f"{'w':>6} {'z':>10} {'k':>8} {'code Tprime':>16} {'FD dT/dz':>16} {'rel diff':>10}")
worst = 0.0
for w in (1.0 / 3.0, 0.2, 0.5):
    for k in (1.0, 30.0, 1000.0):
        for z in (1.0e4, 1.0e3, 1.0e2, 10.0):
            tau = a0eta(z, w)
            H = Hubble(z, w)
            code = compute_analytic_Tprime(k, w, tau, H)
            h = z * 1.0e-6
            fd = (
                compute_analytic_T(k, w, a0eta(z + h, w))
                - compute_analytic_T(k, w, a0eta(z - h, w))
            ) / (2.0 * h)
            rel = abs(code - fd) / max(abs(fd), 1e-300)
            worst = max(worst, rel)
            if abs(w - 1.0 / 3.0) < 1e-12 and k in (1.0, 1000.0) and z in (1.0e3, 10.0):
                print(f"{w:6.3f} {z:10.3g} {k:8.3g} {code:16.8e} {fd:16.8e} {rel:10.2e}")
print(f"worst relative difference over the whole scan: {worst:.3e}")
print("  => Tprime is dT/dz (a d/dlog(1+z) reading would be off by 1+z; an")
print("     H^{-1}dT/deta reading by (1+z)^{-1}; both are O(1)-to-O(1e4) here)\n")

# for the record, show the size of the two wrong readings at one point
w = 1.0 / 3.0
z, k = 1.0e3, 30.0
tau, H = a0eta(z, w), Hubble(z, w)
tp = compute_analytic_Tprime(k, w, tau, H)
print(f"  at w=1/3, z={z}, k={k}: dT/dz={tp:.6e}; (1+z)dT/dz={(1+z)*tp:.6e}; dT/dz/(1+z)={tp/(1+z):.6e}\n")


# ---------------------------------------------------------------- part (b)
def f_spec(Tq, Tr, dTq, dTr, z, w):
    """spec 03 R22 / §0.3, literally as written"""
    return Tq * Tr + 2.0 / (3.0 * (1.0 + w)) * (Tq - (1.0 + z) * dTq) * (
        Tr - (1.0 + z) * dTr
    )


print("=== (b) source_function vs an independent transcription of spec 03 R22 ===")
print(f"{'w':>6} {'z':>10} {'q':>8} {'r':>8} {'f_spec':>17} {'f_code':>17} {'rel':>10}")
worst = 0.0
for w in (1.0 / 3.0, 0.1, 0.5):
    for (q, r) in ((1.0, 1.0), (1.0, 30.0), (30.0, 30.0), (300.0, 700.0)):
        for z in (1.0e5, 1.0e4, 1.0e3, 1.0e2, 10.0, 1.0):
            tau, H = a0eta(z, w), Hubble(z, w)
            Tq = compute_analytic_T(q, w, tau)
            Tr = compute_analytic_T(r, w, tau)
            dTq = compute_analytic_Tprime(q, w, tau, H)
            dTr = compute_analytic_Tprime(r, w, tau, H)
            fs = f_spec(Tq, Tr, dTq, dTr, z, w)
            fc = source_function(Tq, Tr, dTq, dTr, z, w)["source"]
            rel = abs(fs - fc) / max(abs(fs), 1e-300)
            worst = max(worst, rel)
            if abs(w - 1.0 / 3.0) < 1e-12 and (q, r) == (30.0, 30.0):
                print(f"{w:6.3f} {z:10.3g} {q:8.3g} {r:8.3g} {fs:17.9e} {fc:17.9e} {rel:10.2e}")
print(f"worst relative difference over the whole scan: {worst:.3e}  (float round-off only)\n")

# ---------------------------------------------------------------- part (c)
print("=== (c) 'analytic_source_rad' is NOT an independent oracle ===")
print("QuadSource.py:147-154 feeds the analytic T,T' into the SAME source_function,")
print("so analytic_source_rad == f_spec(analytic T) by construction; the numbers")
print("printed in (b) ARE the analytic_source_rad values. Reproduced explicitly:")
w = 1.0 / 3.0
for z in (1.0e4, 1.0e2):
    tau, H = a0eta(z, w), Hubble(z, w)
    Tq = compute_analytic_T(30.0, w, tau)
    Tr = compute_analytic_T(30.0, w, tau)
    dTq = compute_analytic_Tprime(30.0, w, tau, H)
    dTr = compute_analytic_Tprime(30.0, w, tau, H)
    out = source_function(Tq, Tr, dTq, dTr, z, w)
    print(
        f"  z={z:9.3g}  undiff={out['undiff']:14.6e} diff={out['diff']:14.6e} "
        f"source={out['source']:14.6e}   f_spec={f_spec(Tq,Tr,dTq,dTr,z,w):14.6e}"
    )

# ---------------------------------------------------------------- part (d)
print("\n=== (d) symmetry f(q,r) == f(r,q) numerically ===")
w = 1.0 / 3.0
z = 1.0e3
tau, H = a0eta(z, w), Hubble(z, w)
Tq, Tr = compute_analytic_T(11.0, w, tau), compute_analytic_T(97.0, w, tau)
dTq, dTr = (
    compute_analytic_Tprime(11.0, w, tau, H),
    compute_analytic_Tprime(97.0, w, tau, H),
)
a = source_function(Tq, Tr, dTq, dTr, z, w)["source"]
b = source_function(Tr, Tq, dTr, dTq, z, w)["source"]
print(f"  f(q,r)={a:.16e}\n  f(r,q)={b:.16e}\n  identical: {a == b}")
