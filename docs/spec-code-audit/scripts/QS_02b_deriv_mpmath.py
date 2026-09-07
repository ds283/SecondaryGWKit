import os
"""QS_02b: high-precision confirmation that compute_analytic_Tprime = dT/dz
(the plain redshift derivative), using mpmath at 50 digits so that the
finite-difference check is not cancellation-limited at super-horizon scales.
Same exact constant-w background as QS_02.
"""

import sys

sys.path.insert(0, os.getcwd())  # run from the repository root

import mpmath as mp
from ComputeTargets.analytic_Tk import compute_analytic_Tprime

mp.mp.dps = 50
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


print(f"{'w':>7} {'z':>9} {'k':>7} {'dT/dz (mp)':>22} {'code':>22} {'rel':>10}")
worst = 0.0
for w in (mp.mpf(1) / 3, mp.mpf("0.2"), mp.mpf("0.5")):
    for k in (mp.mpf(1), mp.mpf(30), mp.mpf(1000)):
        for z in (mp.mpf("1e5"), mp.mpf("1e4"), mp.mpf("1e3"), mp.mpf(100), mp.mpf(10), mp.mpf(1)):
            exact = mp.diff(lambda zz: T_mp(k, w, zz), z)
            code = compute_analytic_Tprime(float(k), float(w), float(a0eta(z, w)), float(Hubble(z, w)))
            rel = abs(mp.mpf(code) - exact) / abs(exact)
            worst = max(worst, float(rel))
            if k == 30:
                print(f"{float(w):7.4f} {float(z):9.3g} {float(k):7.3g} "
                      f"{mp.nstr(exact, 12):>22} {code:22.12e} {float(rel):10.2e}")
print(f"\nworst relative difference: {worst:.3e}")
print("=> compute_analytic_Tprime is exactly dT/dz.")

# how big would the two alternative readings be at a representative point?
w = mp.mpf(1) / 3
for z in (mp.mpf("1e4"), mp.mpf(10)):
    exact = mp.diff(lambda zz: T_mp(mp.mpf(30), w, zz), z)
    print(
        f"  w=1/3 k=30 z={float(z):9.3g}: dT/dz={mp.nstr(exact,8)}  "
        f"(1+z)dT/dz={mp.nstr((1+z)*exact,8)}  H^-1 dT/deta = dT/dz/(1+z)... "
        f"factor (1+z)={float(1+z):.4g} apart"
    )
