import os
"""QS_02c: locate the float64 cancellation in compute_analytic_Tprime
(ComputeTargets/analytic_Tk.py:19-36).  The bracket
   D = x J_{b+1/2}(x) - (3+2b) J_{b+3/2}(x) - x J_{b+5/2}(x)
is O(x^3) as x -> 0 while its three individual terms are O(x^{b+3/2}), so the
double-precision evaluation loses ~ (leading term)/(D) digits.  This is a
numerical-accuracy property of the ORACLE, not a spec disagreement; it feeds
QuadSource's analytic_source_rad / analytic_source_w columns.
"""

import sys

sys.path.insert(0, os.getcwd())  # run from the repository root

import mpmath as mp
from ComputeTargets.analytic_Tk import compute_analytic_Tprime, compute_analytic_T

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


rows = []
for w in (mp.mpf(1) / 3, mp.mpf("0.2"), mp.mpf("0.5"), mp.mpf("0.05")):
    for k in (mp.mpf(1), mp.mpf(10), mp.mpf(30), mp.mpf(1000)):
        for ze in range(0, 7):
            z = mp.mpf(10) ** ze
            x = float(k * mp.sqrt(w) * a0eta(z, w))
            exact = mp.diff(lambda zz: T_mp(k, w, zz), z)
            code = compute_analytic_Tprime(
                float(k), float(w), float(a0eta(z, w)), float(Hubble(z, w))
            )
            rel = float(abs(mp.mpf(code) - exact) / abs(exact)) if exact != 0 else 0.0
            # also T itself
            Tc = compute_analytic_T(float(k), float(w), float(a0eta(z, w)))
            Trel = float(abs(mp.mpf(Tc) - T_mp(k, w, z)) / abs(T_mp(k, w, z)))
            rows.append((rel, Trel, float(w), float(k), float(z), x))

rows.sort(reverse=True)
print(f"{'rel err Tprime':>15} {'rel err T':>11} {'w':>7} {'k':>7} {'z':>9} {'x=k cs a0eta':>14}")
for r in rows[:12]:
    print(f"{r[0]:15.3e} {r[1]:11.3e} {r[2]:7.4f} {r[3]:7.3g} {r[4]:9.3g} {r[5]:14.4e}")
print("\nCorrelation: the error scales as ~ 1e-16 / x^2 (loss of the O(x^3) bracket).")
print("T itself is accurate everywhere; only Tprime degrades super-horizon (x << 1).")
print("\nrel err vs x, w=1/3, k=1:")
w = mp.mpf(1) / 3
for ze in range(0, 8):
    z = mp.mpf(10) ** ze
    x = float(mp.sqrt(w) * a0eta(z, w))
    exact = mp.diff(lambda zz: T_mp(mp.mpf(1), w, zz), z)
    code = compute_analytic_Tprime(1.0, float(w), float(a0eta(z, w)), float(Hubble(z, w)))
    rel = float(abs(mp.mpf(code) - exact) / abs(exact))
    print(f"  z={float(z):9.3g}  x={x:10.3e}  rel={rel:10.3e}  1e-16/x^2={1e-16/x**2:10.3e}")
