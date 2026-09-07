import os
"""GK_08: (cross-check, confirmation only) spec 02 R43 (= spec 01 R27/R30) transfer-function
omega_eff^2 and its logarithmic derivative, vs ComputeTargets/WKB_Tk.py. Confirms the
coefficient fix recorded in commit 641bb51.
"""

import sympy as sp

z = sp.Symbol("z")
opz = 1 + z
kH2 = sp.Function("kH2")(z)  # k_phys^2/H^2
w = sp.Function("w")(z)  # c_s^2 = wPerturbations
eps = sp.Function("eps")(z)

# d/dz (k^2/H^2) = -2 (k^2/H^2) dlnH/dz = -2 (k^2/H^2) eps/(1+z)
kH2_deriv = -2 * kH2 * eps / opz

N = (
    sp.Rational(3, 2) * (1 + eps) * (1 + w)
    - eps * (3 + eps / 2) / 2
    - sp.Rational(9, 4) * (1 + w) ** 2
)
omega_sq = (
    w * kH2
    + (sp.Rational(3, 2) * sp.diff(w, z) - sp.diff(eps, z) / 2) / opz
    + N / opz**2
)


def dz(expr):
    """differentiate, then substitute the correct derivative of k^2/H^2"""
    return sp.diff(expr, z).subs(sp.Derivative(kH2, z), kH2_deriv)


true = dz(omega_sq)

wp, wpp = sp.diff(w, z), sp.diff(w, z, 2)
ep, epp = sp.diff(eps, z), sp.diff(eps, z, 2)

A = wp * kH2
B = (sp.Rational(3, 2) * wpp - epp / 2 - 2 * eps * w * kH2) / opz
C = (
    ep / 2 * (3 * w - eps + 1) + sp.Rational(3, 2) * wp * (eps - 3 * (1 + w))
) / opz**2
D = -(3 * (1 + eps) * (1 + w) - eps * (3 + eps / 2) - sp.Rational(9, 2) * (1 + w) ** 2) / opz**3
code = A + B + C + D

print("R43 omega_eff^2: code form matches spec by inspection (c_s^2 = w, 3 c_s c_s' = 3w'/2)")
print("WKB_Tk.Tk_d_ln_omegaEff_dz numerator - d(omega^2)/dz =", sp.simplify(code - true))

# and the pre-641bb51 form, with (3/2)(1+w) in place of 3(1+w) in the w' coefficient
C_old = (ep / 2 * (3 * w - eps + 1) + sp.Rational(3, 2) * wp * (eps - sp.Rational(3, 2) * (1 + w))) / opz**2
print("pre-fix numerator - d(omega^2)/dz =", sp.simplify((A + B + C_old + D) - true))
