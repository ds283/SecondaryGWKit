import os
"""QS_05:
(a) how much of the analytic_Tprime float64 cancellation (QS_02c) survives into f?
(b) the analytic_source_w column mixes two different w's: T is built with
    wPerturbations (TkNumericIntegration.py:436-439) but the f coefficients with
    wBackground (QuadSource.py:142-162).  Size of the mismatch in a LambdaCDM
    background.
"""

import sys

sys.path.insert(0, os.getcwd())  # run from the repository root

import mpmath as mp
from ComputeTargets.QuadSource import source_function
from ComputeTargets.analytic_Tk import compute_analytic_T, compute_analytic_Tprime

mp.mp.dps = 60
H0 = mp.mpf(1)


def a0eta(z, w):
    return 2 / ((1 + 3 * w) * H0) * (1 + z) ** (-(1 + 3 * w) / 2)


def Hubble(z, w):
    return H0 * (1 + z) ** (mp.mpf(3) * (1 + w) / 2)


def T_mp(k, w, z):
    b = (1 - 3 * w) / (1 + 3 * w)
    n1 = mp.mpf("1.5") + b
    x = k * mp.sqrt(w) * a0eta(z, w)
    return 2**n1 * mp.gamma(mp.mpf("2.5") + b) * x ** (-n1) * mp.besselj(n1, x)


def f_mp(q, r, w, z):
    Tq, Tr = T_mp(q, w, z), T_mp(r, w, z)
    dTq = mp.diff(lambda zz: T_mp(q, w, zz), z)
    dTr = mp.diff(lambda zz: T_mp(r, w, zz), z)
    return Tq * Tr + 2 / (3 * (1 + w)) * (Tq - (1 + z) * dTq) * (Tr - (1 + z) * dTr)


print("(a) relative error of the float64 analytic source f vs 60-digit mpmath")
print(f"{'w':>7} {'q':>7} {'r':>7} {'z':>9} {'x_q':>11} {'f (mp)':>16} {'rel err f':>11}")
w = mp.mpf(1) / 3
for (q, r) in ((mp.mpf(1), mp.mpf(1)), (mp.mpf(30), mp.mpf(30)), (mp.mpf(1), mp.mpf(1000))):
    for ze in (7, 6, 5, 4, 2, 0):
        z = mp.mpf(10) ** ze
        t, h = a0eta(z, w), Hubble(z, w)
        fc = source_function(
            compute_analytic_T(float(q), float(w), float(t)),
            compute_analytic_T(float(r), float(w), float(t)),
            compute_analytic_Tprime(float(q), float(w), float(t), float(h)),
            compute_analytic_Tprime(float(r), float(w), float(t), float(h)),
            float(z),
            float(w),
        )["source"]
        fe = f_mp(q, r, w, z)
        rel = float(abs(mp.mpf(fc) - fe) / abs(fe))
        xq = float(q * mp.sqrt(w) * t)
        print(f"{float(w):7.4f} {float(q):7.3g} {float(r):7.3g} {float(z):9.3g} "
              f"{xq:11.3e} {mp.nstr(fe,8):>16} {rel:11.3e}")

print("\n  => the cancellation in Tprime is harmless inside f: super-horizon the")
print("     T' terms are O(x^2) suppressed, so f keeps ~1e-15 accuracy throughout.\n")

# ------------------------------------------------------------------ (b)
print("(b) wBackground vs wPerturbations in a LambdaCDM background")
Om, Or, Occ = 0.3, 9.1e-5, 0.7


def wB(z):
    o = 1.0 + z
    return ((1.0 / 3.0) * Or * o**4 - Occ) / (Om * o**3 + Or * o**4 + Occ)


def wP(z):
    o = 1.0 + z
    return (1.0 / 3.0) * Or * o / (Om + Or * o)


print(f"{'z':>10} {'wBackground':>13} {'wPerturbations':>15} "
      f"{'(5+3wB)/(3(1+wB))':>19} {'same with wP':>13} {'ratio':>9}")
for z in (1e6, 1e5, 1e4, 3400.0, 1e3, 100.0, 10.0, 1.0, 0.1):
    a, b = wB(z), wP(z)
    ca = (5 + 3 * a) / (3 * (1 + a))
    cb = (5 + 3 * b) / (3 * (1 + b))
    print(f"{z:10.4g} {a:13.6f} {b:15.6f} {ca:19.6f} {cb:13.6f} {ca/cb:9.4f}")
print("\n  spec 03 sec 0.5 fixes w_0 = wBackground inside f: the code's choice is correct.")
print("  The point is only that the 'analytic_source_w' ORACLE column is internally")
print("  inconsistent -- its T comes from a constant-w = wPerturbations(z) solution")
print("  while its prefactors use wBackground(z).")
